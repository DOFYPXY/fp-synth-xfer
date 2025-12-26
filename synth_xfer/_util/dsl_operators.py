import json
from pathlib import Path
from typing import Dict, List

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


DslOpSet = Dict[str, List[type[Operation]]]

_OP_NAME_MAP: dict[str, type[Operation]] = {
    "CmpOp": CmpOp,
    "arith.AndIOp": arith.AndIOp,
    "arith.OrIOp": arith.OrIOp,
    "arith.XOrIOp": arith.XOrIOp,
    "NegOp": NegOp,
    "AndOp": AndOp,
    "OrOp": OrOp,
    "XorOp": XorOp,
    "AddOp": AddOp,
    "SubOp": SubOp,
    "SelectOp": SelectOp,
    "LShrOp": LShrOp,
    "AShrOp": AShrOp,
    "ShlOp": ShlOp,
    "UMinOp": UMinOp,
    "UMaxOp": UMaxOp,
    "SMinOp": SMinOp,
    "SMaxOp": SMaxOp,
    "UDivOp": UDivOp,
    "SDivOp": SDivOp,
    "URemOp": URemOp,
    "SRemOp": SRemOp,
    "MulOp": MulOp,
    "SetHighBitsOp": SetHighBitsOp,
    "SetLowBitsOp": SetLowBitsOp,
    "ClearHighBitsOp": ClearHighBitsOp,
    "ClearLowBitsOp": ClearLowBitsOp,
    "SetSignBitOp": SetSignBitOp,
    "ClearSignBitOp": ClearSignBitOp,
    "CountLOneOp": CountLOneOp,
    "CountLZeroOp": CountLZeroOp,
    "CountROneOp": CountROneOp,
    "CountRZeroOp": CountRZeroOp,
    "PopCountOp": PopCountOp,
}


def load_dsl_ops(path: Path) -> DslOpSet:
    """Load DSL op sets from a JSON file (e.g., dsl/ops_set_*.json)."""
    with path.open() as f:
        data = json.load(f)

    def _load_kind(kind: str) -> list[type[Operation]]:
        ops: list[type[Operation]] = []
        for entry in data.get(kind, []):
            if isinstance(entry, str):
                op_name = entry
            elif isinstance(entry, dict) and "op_name" in entry:
                # Backward compatibility with older JSON shape.
                op_name = entry["op_name"]
            else:
                raise ValueError(
                    f"Invalid DSL op entry for '{kind}' in {path}: {entry!r}"
                )
            if op_name not in _OP_NAME_MAP:
                raise ValueError(f"Unknown op_name '{op_name}' in {path}")
            op_type = _OP_NAME_MAP[op_name]
            if op_type not in ops:
                ops.append(op_type)
        return ops

    return {INT_T: _load_kind("int_ops"), BOOL_T: _load_kind("i1_ops")}


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


def make_uniform_weights(
    ops: list[type[Operation]], weight: int = 1
) -> dict[type[Operation], int]:
    return {op: weight for op in ops}


DEFAULT_DSL_OPS: DslOpSet = {INT_T: full_int_ops, BOOL_T: full_i1_ops}

i1_prior_uniform: dict[type[Operation], int] = make_uniform_weights(full_i1_ops)

int_prior_uniform: dict[type[Operation], int] = make_uniform_weights(full_int_ops)

int_prior_uniform_stronger: dict[type[Operation], int] = make_uniform_weights(
    full_int_ops, weight=10
)

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
