from __future__ import annotations

from abc import ABC
from typing import ClassVar, Mapping, Sequence

from xdsl.dialects.builtin import Float16Type, FloatAttr
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
    ],
    [FloatType],
)
