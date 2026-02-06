from llvmlite import ir
from xdsl.ir import Attribute

from synth_xfer.dialects.fp import AddOp, SubOp


def lower_fp_type(typ: Attribute) -> ir.Type:
    """Lower floating point types to LLVM IR types.

    Assumes fp16 for now.
    """
    # For now, assume fp16
    return ir.HalfType()


class _LowerFPToLLVM:
    """Helper class to lower FP dialect operations to LLVM IR.

    This class provides methods to lower FP operations to their LLVM equivalents,
    focusing on fp16 operations.
    """

    _fp_intrinsics: dict = {
        # binary operations
        AddOp: ir.IRBuilder.fadd,
        SubOp: ir.IRBuilder.fsub,
    }

    def __init__(self, ir_builder: ir.IRBuilder, ssa_map: dict):
        """Initialize the FP lowering helper.

        Args:
            ir_builder: The LLVM IR builder to use for creating instructions
            ssa_map: Mapping from MLIR SSA values to LLVM IR values
        """
        self.b = ir_builder
        self.ssa_map = ssa_map

    def operands(self, op) -> tuple:
        """Get the LLVM operands for an MLIR operation."""
        return tuple(self.ssa_map[x] for x in op.operands)

    def lower_binop(self, op) -> ir.Value:
        """Lower a binary floating point operation.

        Args:
            op: The MLIR operation (AddOp, SubOp, etc.)

        Returns:
            The LLVM IR value representing the operation result
        """
        if type(op) not in self._fp_intrinsics:
            raise ValueError(f"Unsupported FP operation: {type(op)}")

        llvm_op = self._fp_intrinsics[type(op)]
        lhs, rhs = self.operands(op)

        # Get a reasonable name for the result
        result_name = op.results[0].name_hint or f"fp_{type(op).__name__}"

        return llvm_op(self.b, lhs, rhs, name=result_name)
