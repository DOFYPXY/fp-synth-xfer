from __future__ import annotations

from abc import ABC
from typing import ClassVar, Mapping, Sequence

from xdsl.dialects.builtin import (
    Float16Type,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    OpResult,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    Operand,
    VarConstraint,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class FloatType(ParametrizedAttribute, TypeAttribute):
    name = "fp.float"


@irdl_attr_definition
class FPAbsValueType(ParametrizedAttribute, TypeAttribute):
    name = "fp.abs_value"


class UnaryOp(IRDLOperation, ABC):
    """Base class for unary floating point operations."""

    T: ClassVar = VarConstraint("T", AnyAttr())

    op: Operand = operand_def(T)
    result: OpResult = result_def(T)

    @staticmethod
    def infer_result_type(
        operand_types: Sequence[Attribute], attributes: Mapping[str, Attribute] = {}
    ) -> Sequence[Attribute]:
        match operand_types:
            case [op]:
                return [op]
            case _:
                raise VerifyException("Unary operation expects exactly one operand")

    def __init__(
        self,
        op: SSAValue,
    ):
        super().__init__(
            operands=[op],
            result_types=[op.type],
        )


class BinOp(IRDLOperation, ABC):
    """Base class for binary floating point operations."""

    T: ClassVar = VarConstraint("T", AnyAttr())

    lhs: Operand = operand_def(T)
    rhs: Operand = operand_def(T)
    result: OpResult = result_def(T)

    @staticmethod
    def infer_result_type(
        operand_types: Sequence[Attribute], attributes: Mapping[str, Attribute] = {}
    ) -> Sequence[Attribute]:
        match operand_types:
            case [lhs, _]:
                return [lhs]
            case _:
                raise VerifyException("Binary operation expects exactly two operands")

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[lhs.type],
        )


# Unary operations
@irdl_op_definition
class FPAbsOp(UnaryOp):
    name = "fp.abs"


@irdl_op_definition
class FPNegOp(UnaryOp):
    name = "fp.neg"


@irdl_op_definition
class FPSqrtOp(UnaryOp):
    name = "fp.sqrt"


@irdl_op_definition
class FPConstantOp(IRDLOperation):
    name = "fp.constant"

    result: OpResult = result_def(FloatType)
    value: FloatAttr[Float16Type] = attr_def(FloatAttr[Float16Type])

    def __init__(self, value: float):
        super().__init__(
            operands=[],
            result_types=[FloatType()],
            attributes={"value": FloatAttr(value, 16)},
        )


# Binary operations
@irdl_op_definition
class FPAddOp(BinOp):
    name = "fp.add"


@irdl_op_definition
class FPSubOp(BinOp):
    name = "fp.sub"


@irdl_op_definition
class FPMulOp(BinOp):
    name = "fp.mul"


@irdl_op_definition
class FPDivOp(BinOp):
    name = "fp.div"


@irdl_op_definition
class FPMaxOp(BinOp):
    name = "fp.max"


@irdl_op_definition
class FPMinOp(BinOp):
    name = "fp.min"


@irdl_op_definition
class FPMaximumFOp(BinOp):
    name = "fp.maximumf"


@irdl_op_definition
class FPMinimumFOp(BinOp):
    name = "fp.minimumf"


@irdl_op_definition
class FPCmpOp(IRDLOperation):
    name = "fp.cmp"

    lhs: Operand = operand_def(FloatType)
    rhs: Operand = operand_def(FloatType)
    result: OpResult = result_def(IntegerType(1))
    predicate: StringAttr = attr_def(StringAttr)

    @staticmethod
    def infer_result_type(
        operand_types: Sequence[Attribute], attributes: Mapping[str, Attribute] = {}
    ) -> Sequence[Attribute]:
        match operand_types:
            case [_, _]:
                return [IntegerType(1)]
            case _:
                raise VerifyException("Cmp operation expects exactly two operands")

    def __init__(self, lhs: SSAValue, rhs: SSAValue, predicate: str | StringAttr):
        if isinstance(predicate, str):
            predicate = StringAttr(predicate)
        super().__init__(
            operands=[lhs, rhs],
            result_types=[IntegerType(1)],
            attributes={"predicate": predicate},
        )


@irdl_op_definition
class FPIsNanOp(IRDLOperation):
    name = "fp.is_nan"

    op: Operand = operand_def(FloatType)
    result: OpResult = result_def(IntegerType(1))

    @staticmethod
    def infer_result_type(
        operand_types: Sequence[Attribute], attributes: Mapping[str, Attribute] = {}
    ) -> Sequence[Attribute]:
        match operand_types:
            case [_]:
                return [IntegerType(1)]
            case _:
                raise VerifyException("IsNan operation expects exactly one operand")

    def __init__(self, op: SSAValue):
        super().__init__(
            operands=[op],
            result_types=[IntegerType(1)],
        )


@irdl_op_definition
class FPGetOp(IRDLOperation):
    name = "fp.get"

    value: Operand = operand_def(FPAbsValueType)
    result: OpResult = result_def(AnyAttr())
    index: IntegerAttr[IntegerType] = attr_def(IntegerAttr[IntegerType])

    def __init__(self, value: SSAValue, index: int, result_type: Attribute):
        super().__init__(
            operands=[value],
            result_types=[result_type],
            attributes={"index": IntegerAttr(index, IntegerType(64))},
        )


@irdl_op_definition
class FPMakeOp(IRDLOperation):
    name = "fp.make"

    lo: Operand = operand_def(FloatType)
    hi: Operand = operand_def(FloatType)
    has_nan: Operand = operand_def(IntegerType(1))
    result: OpResult = result_def(FPAbsValueType)

    def __init__(self, lo: SSAValue, hi: SSAValue, has_nan: SSAValue):
        super().__init__(
            operands=[lo, hi, has_nan],
            result_types=[FPAbsValueType()],
        )


@irdl_op_definition
class FPPosInfOp(UnaryOp):
    name = "fp.pos_inf"


@irdl_op_definition
class FPNegInfOp(UnaryOp):
    name = "fp.neg_inf"


FP = Dialect(
    "fp",
    [
        # Unary
        FPAbsOp,
        FPNegOp,
        FPSqrtOp,
        FPConstantOp,
        # Binary
        FPAddOp,
        FPSubOp,
        FPMulOp,
        FPDivOp,
        FPMaxOp,
        FPMinOp,
        FPMaximumFOp,
        FPMinimumFOp,
        FPCmpOp,
        FPIsNanOp,
        FPGetOp,
        FPMakeOp,
        FPPosInfOp,
        FPNegInfOp,
    ],
    [FloatType, FPAbsValueType],
)
