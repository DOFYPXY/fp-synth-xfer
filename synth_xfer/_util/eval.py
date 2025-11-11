from enum import Enum
from pathlib import Path
from subprocess import PIPE, run
from typing import TYPE_CHECKING, Callable

from synth_xfer._eval_engine import (
    ToEvalKnownBits4,
    ToEvalKnownBits8,
    ToEvalKnownBits16,
    ToEvalKnownBits32,
    ToEvalKnownBits64,
    enum_low_knownbits_4,
    enum_low_knownbits_8,
    enum_low_knownbits_16,
    enum_low_knownbits_32,
    enum_low_knownbits_64,
    enum_mid_knownbits_4,
    enum_mid_knownbits_8,
    enum_mid_knownbits_16,
    enum_mid_knownbits_32,
    enum_mid_knownbits_64,
    eval_knownbits_4,
    eval_knownbits_8,
    eval_knownbits_16,
    eval_knownbits_32,
    eval_knownbits_64,
)
from synth_xfer._util.eval_result import EvalResult, PerBitRes, get_per_bit
from synth_xfer._util.helper_funcs import HelperFuncs
from synth_xfer.jit import Jit
from synth_xfer.lower_to_llvm import LowerToLLVM

if TYPE_CHECKING:
    from synth_xfer._eval_engine import BW, KnownBitsToEval


class AbstractDomain(Enum):
    KnownBits = "KnownBits", 2
    UConstRange = "UConstRange", 2
    SConstRange = "SConstRange", 2
    IntegerModulo = "IntegerModulo", 6

    vec_size: int

    def __new__(
        cls,
        value: str,
        vec_size: int,
    ):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.vec_size = vec_size
        return obj

    def __str__(self) -> str:
        return self.name


def _parse_engine_output(output: str) -> list[EvalResult]:
    bw_evals = output.split("---\n")
    bw_evals.reverse()
    per_bits = [get_per_bit(x) for x in bw_evals if x != ""]

    ds: list[list[PerBitRes]] = [[] for _ in range(len(per_bits[0]))]
    for es in per_bits:
        for i, e in enumerate(es):
            ds[i].append(e)

    return [EvalResult(x) for x in ds]


def setup_eval(
    bw: "BW",
    samples: int | None,
    seed: int,
    helper_funcs: HelperFuncs,
    jit: Jit,
) -> "KnownBitsToEval":
    lowerer = LowerToLLVM(bw)
    crt = lowerer.add_fn(helper_funcs.crt_func, shim=True)
    op_constraint = (
        lowerer.add_fn(helper_funcs.op_constraint_func, shim=True)
        if helper_funcs.op_constraint_func
        else None
    )

    jit.add_mod(str(lowerer))
    concrete_fn_ptr = jit.get_fn_ptr(crt.name)
    constraint_fn_ptr = jit.get_fn_ptr(op_constraint.name) if op_constraint else None

    low_fns: dict[BW, Callable[[int, int | None], "KnownBitsToEval"]] = {
        4: enum_low_knownbits_4,
        8: enum_low_knownbits_8,
        16: enum_low_knownbits_16,
        32: enum_low_knownbits_32,
        64: enum_low_knownbits_64,
    }

    mid_fns: dict[BW, Callable[[int, int | None, int, int], "KnownBitsToEval"]] = {
        4: enum_mid_knownbits_4,
        8: enum_mid_knownbits_8,
        16: enum_mid_knownbits_16,
        32: enum_mid_knownbits_32,
        64: enum_mid_knownbits_64,
    }

    if samples:
        return mid_fns[bw](concrete_fn_ptr, constraint_fn_ptr, samples, seed)
    else:
        return low_fns[bw](concrete_fn_ptr, constraint_fn_ptr)


# TODO may want to just pass whole jit in here
def eval_transfer_func(
    to_eval: "KnownBitsToEval",
    xfers: list[int],
    bases: list[int],
    # domain: AbstractDomain,
) -> list[EvalResult]:
    if isinstance(to_eval, ToEvalKnownBits4):
        res = eval_knownbits_4(to_eval, xfers, bases)
    elif isinstance(to_eval, ToEvalKnownBits8):
        res = eval_knownbits_8(to_eval, xfers, bases)
    elif isinstance(to_eval, ToEvalKnownBits16):
        res = eval_knownbits_16(to_eval, xfers, bases)
    elif isinstance(to_eval, ToEvalKnownBits32):
        res = eval_knownbits_32(to_eval, xfers, bases)
    elif isinstance(to_eval, ToEvalKnownBits64):
        res = eval_knownbits_64(to_eval, xfers, bases)

    return _parse_engine_output(str(res))


# TODO
def eval_final(
    data_dir: str,
    xfer_name: str,
    xfer_src: str,
    op_name: str,
    helper_srcs: list[str],
    domain: AbstractDomain,
) -> list[EvalResult]:
    engine_path = Path("xdsl_smt").joinpath("eval_engine", "build", "eval_engine")
    if not engine_path.exists():
        raise FileNotFoundError(f"Eval Engine not found at: {engine_path}")

    engine_params = ""
    engine_params += f"{data_dir}\n"
    engine_params += f"{domain}\n"
    engine_params += f"{op_name}\n"
    engine_params += "\n"
    engine_params += f"{[xfer_name]}\n"
    engine_params += "using A::APInt;\n"
    engine_params += "\n".join(helper_srcs + [xfer_src])

    eval_output = run(
        [engine_path],
        input=engine_params,
        text=True,
        stdout=PIPE,
        stderr=PIPE,
    )

    if eval_output.returncode != 0:
        print("EvalEngine failed with this error:")
        print(eval_output.stderr, end="")
        exit(eval_output.returncode)

    return _parse_engine_output(eval_output.stdout)
