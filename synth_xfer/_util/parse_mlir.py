from dataclasses import dataclass
from pathlib import Path

from xdsl.context import Context
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func, FuncOp
from xdsl.parser import Parser
from xdsl_smt.dialects.transfer import AbstractValueType, Transfer, TransIntegerType
from xdsl_smt.passes.transfer_inline import FunctionCallInline

from synth_xfer._util.domain import AbstractDomain

_ctx = Context()
_ctx.load_dialect(Arith)
_ctx.load_dialect(Builtin)
_ctx.load_dialect(Func)
_ctx.load_dialect(Transfer)


def parse_mlir_func(p: Path | str) -> FuncOp:
    func_str = p if isinstance(p, str) else p.read_text()
    func_name = "<text>" if isinstance(p, str) else p.name
    mod = Parser(_ctx, func_str, func_name).parse_op()

    if isinstance(mod, FuncOp):
        return mod
    else:
        raise ValueError(f"mlir in '{func_name}' is not a FuncOp")


def parse_mlir_mod(p: Path | str, inline: bool = False) -> ModuleOp:
    func_str = p if isinstance(p, str) else p.read_text()
    func_name = "<text>" if isinstance(p, str) else p.name
    mod = Parser(_ctx, func_str, func_name).parse_op()

    if isinstance(mod, ModuleOp):
        if inline:
            FunctionCallInline(False, get_fns(mod)).apply(_ctx, mod)

        return mod
    else:
        raise ValueError(f"mlir in '{func_name}' is not a ModuleOp")


def get_fns(mod: ModuleOp) -> dict[str, FuncOp]:
    return {x.sym_name.data: x for x in mod.ops if isinstance(x, FuncOp)}


@dataclass
class HelperFuncs:
    crt_func: FuncOp
    instance_constraint_func: FuncOp
    domain_constraint_func: FuncOp
    op_constraint_func: FuncOp | None
    get_top_func: FuncOp
    transfer_func: FuncOp
    meet_func: FuncOp


def get_helper_funcs(p: Path, d: AbstractDomain) -> HelperFuncs:
    mod = parse_mlir_mod(p, inline=True)
    fns = get_fns(mod)

    assert "concrete_op" in fns
    crt_func = fns["concrete_op"]
    op_con_fn = fns.get("op_constraint", None)

    # TODO xfer fn is only ever used as a type to construct a top fn
    ty = AbstractValueType([TransIntegerType() for _ in range(d.vec_size)])
    xfer_fn = FuncOp.from_region("empty_transformer", [ty, ty], [ty])

    def get_domain_fns(fp: str) -> FuncOp:
        dp = p.resolve().parent.parent.joinpath(str(d), fp)
        return parse_mlir_func(dp)

    top = get_domain_fns("top.mlir")
    meet = get_domain_fns("meet.mlir")
    constraint = get_domain_fns("get_constraint.mlir")
    instance_constraint = get_domain_fns("get_instance_constraint.mlir")

    return HelperFuncs(
        crt_func=crt_func,
        instance_constraint_func=instance_constraint,
        domain_constraint_func=constraint,
        op_constraint_func=op_con_fn,
        get_top_func=top,
        transfer_func=xfer_fn,
        meet_func=meet,
    )
