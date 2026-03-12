import struct
from typing import Callable, Mapping, Sequence

from xdsl.dialects.builtin import FloatAttr
from xdsl.dialects.smt import BitVectorType, BoolType
from xdsl.ir import (
    Attribute,
    Operation,
    SSAValue,
)
from xdsl.pattern_rewriter import (
    PatternRewriter,
)
import xdsl_smt.dialects.smt_bitvector_dialect as smt_bv
import xdsl_smt.dialects.smt_floatingpoint_dialect as smt_fp
from xdsl_smt.dialects.smt_floatingpoint_dialect import FloatingPointType
from xdsl_smt.dialects.smt_utils_dialect import FirstOp, PairOp, PairType, SecondOp
from xdsl_smt.semantics.semantics import (
    OperationSemantics,
    TypeSemantics,
)

from synth_xfer.dialects.fp import (
    FloatType,
    FPAbsOp,
    FPAbsValueType,
    FPAddOp,
    FPCmpOp,
    FPConstantOp,
    FPDivOp,
    FPGetOp,
    FPIsNanOp,
    FPMakeOp,
    FPMaximumFOp,
    FPMaxOp,
    FPMinimumFOp,
    FPMinOp,
    FPMulOp,
    FPNegInfOp,
    FPNegOp,
    FPPosInfOp,
    FPSqrtOp,
    FPSubOp,
)

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

        rewriter.insert_op_before_matched_op([sign_op, exp_op, mant_op, fp_const])
        return ((fp_const.result,), effect_state)


class FloatingPointTypeSemantics(TypeSemantics):
    """Convert a FloatType to a FloatingPointType."""

    def get_semantics(self, type: Attribute) -> Attribute:
        assert isinstance(type, FloatType)
        return FloatingPointType(_FP16_EB, _FP16_SB)


class FloatingPointAbsTypeSemantics(TypeSemantics):
    """Convert an FPAbsValueType to its SMT representation."""

    def get_semantics(self, type: Attribute) -> Attribute:
        assert isinstance(type, FPAbsValueType)
        fp_sort = FloatingPointType(_FP16_EB, _FP16_SB)
        # has_nan is i1, which arith ops expect as PairType(BV1, Bool)
        i1_lowered = PairType(BitVectorType(1), BoolType())
        return PairType(fp_sort, PairType(fp_sort, i1_lowered))


class FPGetOpSemantics(OperationSemantics):
    """Lower ``fp.get`` to pair projections on Pair(lo, Pair(hi, has_nan))."""

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        from xdsl.dialects.builtin import IntegerAttr

        index = attributes["index"]
        assert isinstance(index, IntegerAttr)
        idx = index.value.data
        arg = operands[0]
        ops: list[Operation] = []
        if idx == 0:
            ops.append(FirstOp(arg))
        elif idx == 1:
            ops.append(SecondOp(arg))
            ops.append(FirstOp(ops[-1].results[0]))
        elif idx == 2:
            ops.append(SecondOp(arg))
            ops.append(SecondOp(ops[-1].results[0]))
        else:
            raise ValueError(f"FPGetOp index {idx} out of range for FPAbsValueType")
        rewriter.insert_op_before_matched_op(ops)
        return ((ops[-1].results[0],), effect_state)


class FPIsNanOpSemantics(OperationSemantics):
    """Lower ``fp.is_nan`` to smt_fp.IsNaNOp, producing PairType(BV1, Bool)."""

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        from xdsl.dialects.smt import ConstantBoolOp
        from xdsl_smt.dialects.smt_dialect import IteOp

        is_nan = smt_fp.IsNaNOp(operands[0])
        bv1_true = smt_bv.ConstantOp(1, 1)
        bv1_false = smt_bv.ConstantOp(0, 1)
        ite = IteOp(is_nan.res, bv1_true.res, bv1_false.res)

        no_poison = ConstantBoolOp(False)
        pair = PairOp(ite.res, no_poison.result)
        rewriter.insert_op_before_matched_op(
            [is_nan, bv1_true, bv1_false, ite, no_poison, pair]
        )
        return ((pair.results[0],), effect_state)


class FPCmpOpSemantics(OperationSemantics):
    """Lower ``fp.cmp`` to the appropriate SMT FP comparison predicate(s).

    Handles ordered (oeq/one/olt/ole/ogt/oge), unordered (ueq/une/ult/ule/ugt/uge),
    and constant (false/true) predicates.  The result is ``PairType(BV1, Bool)``
    to match the ``IntegerType(1)`` lowering.
    """

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        from xdsl.dialects.builtin import StringAttr
        from xdsl.dialects.smt import ConstantBoolOp, NotOp, OrOp
        from xdsl_smt.dialects.smt_dialect import IteOp

        pred_attr = attributes["predicate"]
        assert isinstance(pred_attr, StringAttr)
        pred = pred_attr.data

        lhs, rhs = operands[0], operands[1]
        ops: list[Operation] = []
        bool_val: SSAValue  # the Bool-typed SSA value for the comparison

        if pred == "false":
            c = ConstantBoolOp(False)
            ops.append(c)
            bool_val = c.result
        elif pred == "true":
            c = ConstantBoolOp(True)
            ops.append(c)
            bool_val = c.result
        elif pred == "oeq":
            cmp = smt_fp.EqOp(lhs, rhs)
            ops.append(cmp)
            bool_val = cmp.res
        elif pred == "one":
            # ordered not-equal: lt(a,b) OR gt(a,b)
            lt = smt_fp.LtOp(lhs, rhs)
            gt = smt_fp.GtOp(lhs, rhs)
            or_op = OrOp(lt.res, gt.res)
            ops.extend([lt, gt, or_op])
            bool_val = or_op.result
        elif pred == "olt":
            cmp = smt_fp.LtOp(lhs, rhs)
            ops.append(cmp)
            bool_val = cmp.res
        elif pred == "ole":
            cmp = smt_fp.LeqOp(lhs, rhs)
            ops.append(cmp)
            bool_val = cmp.res
        elif pred == "ogt":
            cmp = smt_fp.GtOp(lhs, rhs)
            ops.append(cmp)
            bool_val = cmp.res
        elif pred == "oge":
            cmp = smt_fp.GeqOp(lhs, rhs)
            ops.append(cmp)
            bool_val = cmp.res
        elif pred == "ueq":
            # unordered or equal = NOT (lt(a,b) OR gt(a,b))
            lt = smt_fp.LtOp(lhs, rhs)
            gt = smt_fp.GtOp(lhs, rhs)
            or_op = OrOp(lt.res, gt.res)
            neg = NotOp(or_op.result)
            ops.extend([lt, gt, or_op, neg])
            bool_val = neg.result
        elif pred == "une":
            # unordered or not-equal = NOT fp.eq(a, b)
            eq = smt_fp.EqOp(lhs, rhs)
            neg = NotOp(eq.res)
            ops.extend([eq, neg])
            bool_val = neg.result
        elif pred == "ult":
            # unordered or less-than = NOT fp.geq(a, b)
            geq = smt_fp.GeqOp(lhs, rhs)
            neg = NotOp(geq.res)
            ops.extend([geq, neg])
            bool_val = neg.result
        elif pred == "ule":
            # unordered or less-equal = NOT fp.gt(a, b)
            gt = smt_fp.GtOp(lhs, rhs)
            neg = NotOp(gt.res)
            ops.extend([gt, neg])
            bool_val = neg.result
        elif pred == "ugt":
            # unordered or greater-than = NOT fp.leq(a, b)
            leq = smt_fp.LeqOp(lhs, rhs)
            neg = NotOp(leq.res)
            ops.extend([leq, neg])
            bool_val = neg.result
        elif pred == "uge":
            # unordered or greater-equal = NOT fp.lt(a, b)
            lt = smt_fp.LtOp(lhs, rhs)
            neg = NotOp(lt.res)
            ops.extend([lt, neg])
            bool_val = neg.result
        else:
            raise ValueError(f"Unsupported fp.cmp predicate: {pred}")

        # Convert Bool → BV1 via IteOp, then wrap with no-poison
        bv1_true = smt_bv.ConstantOp(1, 1)
        bv1_false = smt_bv.ConstantOp(0, 1)
        ite = IteOp(bool_val, bv1_true.res, bv1_false.res)
        no_poison = ConstantBoolOp(False)
        pair = PairOp(ite.res, no_poison.result)
        ops.extend([bv1_true, bv1_false, ite, no_poison, pair])

        rewriter.insert_op_before_matched_op(ops)
        return ((pair.results[0],), effect_state)


class FPMakeOpSemantics(OperationSemantics):
    """Lower ``fp.make(lo, hi, has_nan)`` to ``Pair(lo, Pair(hi, has_nan))``.

    This assembles an abstract value from its components, matching the
    nested-pair layout of ``FloatingPointAbsTypeSemantics``:
    ``PairType(fp, PairType(fp, i1_lowered))``.
    """

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        lo, hi, has_nan = operands[0], operands[1], operands[2]

        inner = PairOp(hi, has_nan)
        outer = PairOp(lo, inner.results[0])
        rewriter.insert_op_before_matched_op([inner, outer])
        return ((outer.results[0],), effect_state)


fp_semantics: dict[type[Operation], OperationSemantics] = {
    # No rounding mode
    FPAbsOp: OpSemantics(smt_fp.AbsOp),
    FPNegOp: OpSemantics(smt_fp.NegOp),
    FPMaxOp: OpSemantics(smt_fp.MaxOp),
    FPMinOp: OpSemantics(smt_fp.MinOp),
    FPMaximumFOp: OpSemantics(smt_fp.MaxOp),
    FPMinimumFOp: OpSemantics(smt_fp.MinOp),
    # With rounding mode
    FPSqrtOp: RoundingModeOpSemantics(smt_fp.SqrtOp),
    FPAddOp: RoundingModeOpSemantics(smt_fp.AddOp),
    FPSubOp: RoundingModeOpSemantics(smt_fp.SubOp),
    FPMulOp: RoundingModeOpSemantics(smt_fp.MulOp),
    FPDivOp: RoundingModeOpSemantics(smt_fp.DivOp),
    # Constant
    FPConstantOp: FPConstantOpSemantics(),
    # Abs value ops
    FPGetOp: FPGetOpSemantics(),
    FPMakeOp: FPMakeOpSemantics(),
    FPIsNanOp: FPIsNanOpSemantics(),
    FPPosInfOp: OpSemantics(lambda _: smt_fp.PositiveInfinityOp(_FP16_EB, _FP16_SB)),
    FPNegInfOp: OpSemantics(lambda _: smt_fp.NegativeInfinityOp(_FP16_EB, _FP16_SB)),
    FPCmpOp: FPCmpOpSemantics(),
}
