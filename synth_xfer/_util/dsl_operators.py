from xdsl.dialects import arith
from xdsl.ir import Operation
from xdsl_smt.dialects.transfer import (
    AddOp,
    AndOp,
    AShrOp,
    ClearHighBitsOp,
    ClearLowBitsOp,
    ClearSignBitOp,
    CmpOp,
    CountLOneOp,
    CountLZeroOp,
    CountROneOp,
    CountRZeroOp,
    LShrOp,
    MulOp,
    NegOp,
    OrOp,
    PopCountOp,
    SDivOp,
    SelectOp,
    SetHighBitsOp,
    SetLowBitsOp,
    SetSignBitOp,
    ShlOp,
    SMaxOp,
    SMinOp,
    SRemOp,
    SubOp,
    UDivOp,
    UMaxOp,
    UMinOp,
    UnaryOp,
    URemOp,
    XorOp,
)

INT_T = "int"
BOOL_T = "bool"


def get_operand_kinds(op_type: type[Operation]) -> tuple[str, ...]:
    """Infer operand kinds (int or bool) from the op definition."""
    if op_type == SelectOp:
        return (BOOL_T, INT_T, INT_T)
    if op_type in {arith.AndIOp, arith.OrIOp, arith.XOrIOp}:
        return (BOOL_T, BOOL_T)
    if op_type == CmpOp:
        return (INT_T, INT_T)
    if issubclass(op_type, UnaryOp):
        return (INT_T,)
    return (INT_T, INT_T)


def get_result_kind(op_type: type[Operation]) -> str:
    if op_type in {arith.AndIOp, arith.OrIOp, arith.XOrIOp, CmpOp}:
        return BOOL_T
    return INT_T


basic_int_ops: list[type[Operation]] = [
    NegOp,
    AndOp,
    OrOp,
    XorOp,
    AddOp,
]

custom_int_ops1: list[type[Operation]] = [
    NegOp,
    AndOp,
    OrOp,
    XorOp,
    AddOp,
    SubOp,
    SelectOp,
    UMinOp,
    UMaxOp,
    MulOp,
]

custom_int_ops_w_mul: list[type[Operation]] = [
    NegOp,
    AndOp,
    OrOp,
    XorOp,
    AddOp,
    SubOp,
    SelectOp,
    LShrOp,
    ShlOp,
    UMinOp,
    UMaxOp,
    SMinOp,
    SMaxOp,
    UDivOp,
    SDivOp,
    URemOp,
    SRemOp,
    MulOp,
]

custom_int_ops_w_bit: list[type[Operation]] = [
    NegOp,
    AndOp,
    OrOp,
    XorOp,
    AddOp,
    SubOp,
    SelectOp,
    LShrOp,
    ShlOp,
    UMinOp,
    UMaxOp,
    SMinOp,
    SMaxOp,
    SetHighBitsOp,
    SetLowBitsOp,
    ClearHighBitsOp,
    ClearLowBitsOp,
    SetSignBitOp,
    ClearSignBitOp,
]

full_int_ops: list[type[Operation]] = [
    NegOp,
    AndOp,
    OrOp,
    XorOp,
    AddOp,
    SubOp,
    SelectOp,
    LShrOp,
    AShrOp,
    ShlOp,
    UMinOp,
    UMaxOp,
    SMinOp,
    SMaxOp,
    UDivOp,
    SDivOp,
    URemOp,
    SRemOp,
    MulOp,
    SetHighBitsOp,
    SetLowBitsOp,
    ClearHighBitsOp,
    ClearLowBitsOp,
    SetSignBitOp,
    ClearSignBitOp,
    CountLOneOp,
    CountLZeroOp,
    CountROneOp,
    CountRZeroOp,
    PopCountOp,
]

full_i1_ops: list[type[Operation]] = [
    arith.AndIOp,
    arith.OrIOp,
    arith.XOrIOp,
    CmpOp,
]

basic_i1_ops: list[type[Operation]] = [
    CmpOp,
]


i1_prior_uniform: dict[type[Operation], int] = {k: 1 for k in full_i1_ops}

int_prior_uniform: dict[type[Operation], int] = {k: 1 for k in full_int_ops}

int_prior_uniform_stronger: dict[type[Operation], int] = {k: 10 for k in full_int_ops}

int_prior_bias: dict[type[Operation], int] = {
    NegOp: 10,
    AndOp: 10,
    OrOp: 10,
    XorOp: 10,
    AddOp: 10,
    SubOp: 10,
    SelectOp: 0,
    LShrOp: 0,
    ShlOp: 0,
    UMinOp: 0,
    UMaxOp: 0,
    SMinOp: 0,
    SMaxOp: 0,
    MulOp: 0,
    SetHighBitsOp: 0,
    SetLowBitsOp: 0,
    ClearHighBitsOp: 0,
    ClearLowBitsOp: 0,
    SetSignBitOp: 0,
    ClearSignBitOp: 0,
}
