from __future__ import annotations
from abc import ABC

from typing import ClassVar, Mapping, Sequence

from xdsl.ir import (
    ParametrizedAttribute,
    Dialect,
    TypeAttribute,
    OpResult,
    Attribute,
    SSAValue,
)

from xdsl.irdl import (
    operand_def,
    result_def,
    Operand,
    VarConstraint,
    irdl_attr_definition,
    irdl_op_definition,
    IRDLOperation,
    AnyAttr,
)
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class FloatType(ParametrizedAttribute, TypeAttribute):
    name = "fp.float"


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


@irdl_op_definition
class AddOp(BinOp):
    name = "fp.add"


@irdl_op_definition
class SubOp(BinOp):
    name = "fp.sub"


FP = Dialect(
    "fp",
    [
        AddOp,
        SubOp,
    ],
    [FloatType],
)
