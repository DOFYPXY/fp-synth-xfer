from pathlib import Path

import pytest
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import FuncOp

from synth_xfer._util.lower import LowerToLLVM
from synth_xfer._util.parse_mlir import parse_mlir

MLIR_FILES = [
    "mlir/FPRange/top.mlir",
    "mlir/FPRange/meet.mlir",
    "mlir/FPRange/get_constraint.mlir",
    "mlir/FPRange/get_instance_constraint.mlir",
    "tests/data/fpr_add.mlir",
    "tests/data/fpr_nop.mlir",
]


@pytest.mark.parametrize("rel_path", MLIR_FILES)
def test_lower_fp_mlir_files(rel_path: str) -> None:
    path = Path(rel_path)
    mlir_op = parse_mlir(path)

    lowerer = LowerToLLVM([16], name=path.stem)

    if isinstance(mlir_op, ModuleOp):
        lowerer.add_mod(mlir_op)
    elif isinstance(mlir_op, FuncOp):
        lowerer.add_fn(mlir_op)
    else:
        raise AssertionError(f"Unexpected MLIR op type: {type(mlir_op).__name__}")

    # print(f"\n--- Lowering for {rel_path} ---")
    # print(lowerer)

    # Ensure we produced at least one LLVM function.
    assert lowerer.fns
