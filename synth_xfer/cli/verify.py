from argparse import ArgumentParser, ArgumentTypeError, Namespace
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
from synth_xfer._util.verifier import verify_transfer_function


def verify_function(
    bw: int,
    func: FuncOp,
    xfer_helpers: list[FuncOp | None],
    helper_funcs: HelperFuncs,
    timeout: int,
) -> tuple[bool | None, ModelRef | None]:
    xfer_helpers += [
        helper_funcs.get_top_func,
        helper_funcs.instance_constraint_func,
        helper_funcs.domain_constraint_func,
        helper_funcs.op_constraint_func,
        helper_funcs.meet_func,
    ]
    helpers = [x for x in xfer_helpers if x is not None]

    return verify_transfer_function(func, helper_funcs.crt_func, helpers, bw, timeout)


def _parse_int_range(s: str) -> range:
    parts = s.split(":")
    if len(parts) == 1:
        try:
            n = int(parts[0])
        except ValueError:
            raise ArgumentTypeError(f"Invalid integer '{s}'")
        return range(n, n + 1)

    if len(parts) == 2:
        try:
            start = int(parts[0])
            end = int(parts[1])
        except ValueError:
            raise ArgumentTypeError(f"Invalid range '{s}', expected INT or INT:INT")

        if start > end:
            raise ArgumentTypeError(f"Range must be non-decreasing: {start}>{end}")

        return range(start, end + 1)

    raise ArgumentTypeError(f"Invalid range '{s}', expected INT or INT:INT")


def _register_parser() -> Namespace:
    p = ArgumentParser()

    p.add_argument(
        "-bw",
        type=_parse_int_range,
        required=True,
        help="Bitwidth range (e.g. `-bw 4` or `-bw 4:64`)",
    )

    p.add_argument(
        "-domain",
        type=str,
        choices=[str(x) for x in AbstractDomain],
        required=True,
        help="Abstract Domain to evaluate",
    )

    p.add_argument("-op", type=Path, required=True, help="Concrete op")
    p.add_argument("-xfer_file", type=Path, required=True, help="Transformer file")
    p.add_argument("-xfer_name", type=str, required=True, help="Transformer to verify")
    p.add_argument("-timeout", type=int, default=30, help="z3 timeout")

    return p.parse_args()


def main() -> None:
    args = _register_parser()
    domain = AbstractDomain[args.domain]
    xfer_name = str(args.xfer_name)

    xfer_fns = get_fns(parse_mlir_mod(args.xfer_file))
    xfer_fn = xfer_fns[xfer_name]
    del xfer_fns[xfer_name]

    helper_funcs = get_helper_funcs(args.op, domain)

    for bw in args.bw:
        start_time = perf_counter()
        is_sound, model = verify_function(
            bw, xfer_fn, list(xfer_fns.values()), helper_funcs, args.timeout
        )
        run_time = perf_counter() - start_time

        if is_sound is None:
            print(f"{bw:<2} bits | timeout | took {run_time:.4f}s")
        elif is_sound:
            print(f"{bw:<2} bits | sound   | took {run_time:.4f}s")
        else:
            print(f"{bw:<2} bits | unsound | took {run_time:.4f}s")
            print("counterexample:")
            print(model)


if __name__ == "__main__":
    main()
