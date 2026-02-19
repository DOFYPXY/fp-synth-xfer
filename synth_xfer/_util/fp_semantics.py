import struct
from typing import Callable, Mapping, Sequence

from xdsl.ir import (
    Attribute,
    Operation,
    SSAValue,
)
from xdsl.pattern_rewriter import (
    PatternRewriter,
)
from xdsl.dialects.builtin import FloatAttr

import xdsl_smt.dialects.smt_bitvector_dialect as smt_bv
import xdsl_smt.dialects.smt_floatingpoint_dialect as smt_fp
from xdsl_smt.dialects.smt_floatingpoint_dialect import FloatingPointType
from xdsl_smt.semantics.semantics import (
    OperationSemantics,
    TypeSemantics,
)

from synth_xfer.dialects.fp import *

# IEEE 754 float16 layout: 1 sign bit, 5 exponent bits, 10 mantissa bits
_FP16_EB = 5
_FP16_SB = 11  # significand bits *including* the hidden bit


class OpSemantics(OperationSemantics):
    """Semantics for fp ops that map directly (no rounding mode).

    Maps ``fp.abs(x)`` → ``smt_fp.AbsOp(x)``,
         ``fp.max(a, b)`` → ``smt_fp.MaxOp(a, b)``, etc.
    """

    def __init__(self, ctor: Callable[..., Operation]):
        self._ctor = ctor

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        new_op = self._ctor(*operands)
        rewriter.insert_op_before_matched_op([new_op])
        return ((new_op.results[0],), effect_state)


class RoundingModeOpSemantics(OperationSemantics):
    """Semantics for fp ops that require a rounding-mode first operand.

    Maps ``fp.sqrt(x)`` → ``smt_fp.SqrtOp(RNE, x)``,
         ``fp.add(a, b)`` → ``smt_fp.AddOp(RNE, a, b)``, etc.
    """

    def __init__(self, ctor: Callable[..., Operation]):
        self._ctor = ctor

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        rne = smt_fp.RNEOp()
        rewriter.insert_op_before_matched_op([rne])

        new_op = self._ctor(rne.res, *operands)
        rewriter.insert_op_before_matched_op([new_op])
        return ((new_op.results[0],), effect_state)


class FPConstantOpSemantics(OperationSemantics):
    """Lower ``fp.constant`` to ``smt.fp.constant``.

    Decodes the float16 value into sign / exponent / mantissa bitvectors
    and constructs an ``(fp sign exponent mantissa)`` SMT term.
    """

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        value_attr = attributes["value"]
        assert isinstance(value_attr, FloatAttr)
        float_val: float = value_attr.value.data

        # Pack to IEEE 754 binary16 as a bit string.
        # Layout: [1 sign | EB exponent | SB-1 mantissa]
        # The hidden bit is not stored.
        raw = int.from_bytes(struct.pack(">e", float_val), "big")
        total_bits = 1 + _FP16_EB + (_FP16_SB - 1)
        bit_str = format(raw, f"0{total_bits}b")

        # Slice the bit string according to the IEEE 754 layout
        sign = int(bit_str[0], 2)
        exponent = int(bit_str[1 : 1 + _FP16_EB], 2)
        mantissa = int(bit_str[1 + _FP16_EB :], 2)

        sign_op = smt_bv.ConstantOp(sign, 1)
        exp_op = smt_bv.ConstantOp(exponent, _FP16_EB)
        mant_op = smt_bv.ConstantOp(mantissa, _FP16_SB - 1)

        fp_const = smt_fp.ConstantOp(sign_op.res, exp_op.res, mant_op.res)

        rewriter.insert_op_before_matched_op(
            [sign_op, exp_op, mant_op, fp_const]
        )
        return ((fp_const.result,), effect_state)


class FloatingPointTypeSemantics(TypeSemantics):
    """Convert a FloatType to a FloatingPointType."""

    def get_semantics(self, type: Attribute) -> Attribute:
        assert isinstance(type, FloatType)
        return FloatingPointType(_FP16_EB, _FP16_SB)


fp_semantics: dict[type[Operation], OperationSemantics] = {
    # No rounding mode
    FPAbsOp: OpSemantics(smt_fp.AbsOp),
    FPNegOp: OpSemantics(smt_fp.NegOp),
    FPMaxOp: OpSemantics(smt_fp.MaxOp),
    FPMinOp: OpSemantics(smt_fp.MinOp),
    # With rounding mode
    FPSqrtOp: RoundingModeOpSemantics(smt_fp.SqrtOp),
    FPAddOp: RoundingModeOpSemantics(smt_fp.AddOp),
    FPSubOp: RoundingModeOpSemantics(smt_fp.SubOp),
    FPMulOp: RoundingModeOpSemantics(smt_fp.MulOp),
    FPDivOp: RoundingModeOpSemantics(smt_fp.DivOp),
    # Constant
    FPConstantOp: FPConstantOpSemantics(),
}
