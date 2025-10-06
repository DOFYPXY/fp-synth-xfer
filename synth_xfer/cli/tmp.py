from pathlib import Path
import sys

from synth_xfer.jit import Jit, LowerToLLVM, parse_mlir_funcs


def main() -> None:
    bw = 8
    mlir_path = Path(sys.argv[1])
    mlir_funcs = parse_mlir_funcs(mlir_path)

    lowerer = LowerToLLVM(bw, mlir_path.stem)
    fns = lowerer.add_fns(mlir_funcs)

    print(lowerer.shim_xfer(fns[1]))
    print(lowerer.shim_conc(fns[0]))

    jit = Jit(lowerer)
    fn_ptr = jit.get_fn_ptr("kb_and_shim")
    and_fn_ptr = jit.get_fn_ptr("concrete_op_shim")

    print("######################")
    from synth_xfer import _eval_engine

    print([x for x in dir(_eval_engine) if not x.startswith("__")])
    print("######################")

    from synth_xfer._eval_engine import enum_low_knownbits_4, eval_knownbits_4

    to_eval_kb_4 = enum_low_knownbits_4(and_fn_ptr)
    r = eval_knownbits_4(to_eval_kb_4, [fn_ptr], [])
    print(r)


if __name__ == "__main__":
    main()
