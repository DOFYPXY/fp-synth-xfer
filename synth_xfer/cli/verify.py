from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import perf_counter

from xdsl.dialects.func import FuncOp
from z3 import ModelRef

from synth_xfer._util.domain import AbstractDomain
from synth_xfer._util.parse_mlir import (
    HelperFuncs,
    get_fns,
    get_helper_funcs,
    parse_mlir_mod,
)
from synth_xfer._util.verifier import verify_fp_transfer_function, verify_transfer_function
from synth_xfer.cli.args import int_list
from synth_xfer.cli.eval_final import resolve_xfer_name


def verify_function(
    func: FuncOp,
    xfer_helpers: list[FuncOp | None],
    helper_funcs: HelperFuncs,
    timeout: int,
    domain: AbstractDomain = AbstractDomain.KnownBits,
    solver: str = "z3",
    bw: int | None = None,
) -> tuple[bool | None, ModelRef | None]:
    xfer_helpers += [
        helper_funcs.get_top_func,
        helper_funcs.instance_constraint_func,
        helper_funcs.domain_constraint_func,
        helper_funcs.op_constraint_func,
        helper_funcs.meet_func,
    ]
    helpers = [x for x in xfer_helpers if x is not None]

    if domain == AbstractDomain.FPRange:
        return verify_fp_transfer_function(
            func, helper_funcs.crt_func, helpers, timeout, solver=solver
        )

    assert bw is not None, "bw is required for bitvector domains"
    return verify_transfer_function(func, helper_funcs.crt_func, helpers, bw, timeout, solver=solver)


def _register_parser() -> Namespace:
    p = ArgumentParser()

    p.add_argument(
        "--bw",
        type=int_list,
        help="Bitwidth range (e.g. `--bw 4`, `--bw 4-64` or `--bw 4,8,16`). Required for bitvector domains.",
    )

    p.add_argument(
        "-d",
        "--domain",
        type=str,
        choices=[str(x) for x in AbstractDomain],
        required=True,
        help="Abstract Domain to evaluate",
    )

    p.add_argument("--op", type=Path, required=True, help="Concrete op")
    p.add_argument("--xfer-file", type=Path, required=True, help="Transformer file")
    p.add_argument("--xfer-name", type=str, help="Transformer to verify")
    p.add_argument("--timeout", type=int, default=30, help="Solver timeout (per bitwidth)")
    p.add_argument(
        "--solver",
        type=str,
        choices=["z3", "cvc5"],
        default="z3",
        help="SMT solver to use (default: z3)",
    )

    return p.parse_args()


def main() -> None:
    args = _register_parser()
    domain = AbstractDomain[args.domain]
    xfer_fns = get_fns(parse_mlir_mod(args.xfer_file))
    xfer_name = resolve_xfer_name(xfer_fns, args.xfer_name)

    xfer_fn = xfer_fns[xfer_name]
    del xfer_fns[xfer_name]
    helper_funcs = get_helper_funcs(args.op, domain)

    if domain == AbstractDomain.FPRange:
        start_time = perf_counter()
        is_sound, model = verify_function(
            xfer_fn, list(xfer_fns.values()), helper_funcs, args.timeout, domain,
            solver=args.solver,
        )
        run_time = perf_counter() - start_time
        _print_result(None, is_sound, model, run_time)
    else:
        if args.bw is None:
            raise SystemExit("error: --bw is required for bitvector domains")
        for bw in args.bw:
            start_time = perf_counter()
            is_sound, model = verify_function(
                xfer_fn, list(xfer_fns.values()), helper_funcs, args.timeout, domain,
                solver=args.solver, bw=bw,
            )
            run_time = perf_counter() - start_time
            _print_result(bw, is_sound, model, run_time)


def _print_result(
    bw: int | None,
    is_sound: bool | None,
    model: object,
    run_time: float,
) -> None:
    prefix = f"{bw:<2} bits | " if bw is not None else ""
    if is_sound is None:
        print(f"{prefix}timeout | took {run_time:.4f}s")
    elif is_sound:
        print(f"{prefix}sound   | took {run_time:.4f}s")
    else:
        print(f"{prefix}unsound | took {run_time:.4f}s")
        print("counterexample:")
        print(model)


if __name__ == "__main__":
    main()
