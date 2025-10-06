from functools import singledispatchmethod
from pathlib import Path
from typing import Callable

from llvmlite import ir
import llvmlite.binding as llvm
from xdsl.context import Context
from xdsl.dialects.arith import Arith, ConstantOp
from xdsl.dialects.builtin import Builtin, IntegerType, ModuleOp
from xdsl.dialects.func import Func, FuncOp, ReturnOp
from xdsl.ir import Attribute, Operation
from xdsl.parser import Parser
from xdsl_smt.dialects.transfer import (
    AbstractValueType,
    CmpOp,
    Constant,
    CountLOneOp,
    CountLZeroOp,
    CountROneOp,
    CountRZeroOp,
    GetAllOnesOp,
    GetBitWidthOp,
    GetOp,
    GetSignedMaxValueOp,
    GetSignedMinValueOp,
    IsNegativeOp,
    MakeOp,
    SAddOverflowOp,
    SMaxOp,
    SMinOp,
    SMulOverflowOp,
    SShlOverflowOp,
    # SSubOverflowOp,
    Transfer,
    TransIntegerType,
    TupleType,
    UAddOverflowOp,
    UMaxOp,
    UMinOp,
    UMulOverflowOp,
    UShlOverflowOp,
    # USubOverflowOp,
)

# TODO make sure to add all proper constraints so that it works like z3
# TODO clean up big map
# TODO type this map better
# BinaryFunc: Callable[[ir.IRBuilder, ir.Value, ir.Value, str], ir.Value] = ir.IRBuilder.not_
# TODO just use str.endswith calls to remove duplicate elements in the bigmap
_big_map: dict[str, Callable] = {
    # unary
    "transfer.neg": ir.IRBuilder.neg,
    # binary
    "transfer.and": ir.IRBuilder.and_,
    "arith.andi": ir.IRBuilder.and_,
    "transfer.add": ir.IRBuilder.add,
    "arith.addi": ir.IRBuilder.add,
    "transfer.or": ir.IRBuilder.or_,
    "arith.ori": ir.IRBuilder.or_,
    "transfer.xor": ir.IRBuilder.xor,
    "arith.xori": ir.IRBuilder.xor,
    "transfer.sub": ir.IRBuilder.sub,
    "arith.subi": ir.IRBuilder.sub,
    "transfer.mul": ir.IRBuilder.mul,
    "transfer.udiv": ir.IRBuilder.udiv,
    "transfer.sdiv": ir.IRBuilder.sdiv,
    "transfer.urem": ir.IRBuilder.urem,
    "transfer.srem": ir.IRBuilder.srem,
    "transfer.ashr": ir.IRBuilder.ashr,
    "transfer.lshr": ir.IRBuilder.lshr,
    "transfer.shl": ir.IRBuilder.shl,
    # # ternery
    "transfer.select": ir.IRBuilder.select,
    "arith.select": ir.IRBuilder.select,
    # not impl'd yet
    # "transfer.get_high_bits": ".getHiBits",
    # "transfer.get_low_bits": ".getLoBits",
    # "transfer.set_high_bits": ".setHighBits",
    # "transfer.set_low_bits": ".setLowBits",
    # "transfer.clear_high_bits": ".clearHighBits",
    # "transfer.clear_low_bits": ".clearLowBits",
    # "transfer.set_sign_bit": ".setSignBit",
    # "transfer.clear_sign_bit": ".clearSignBit",
    # "transfer.intersects": ".intersects",
    # "transfer.extract": ".extractBits",
    # "transfer.reverse_bits": ".reverseBits",
}


# TODO broken rn
# @add_op.register
# def _(op: CallOp, b: ir.IRBuilder, vals: dict[str, ir.Value]) -> None:
#     oprnds = get_operands(op, vals)
#     res_name = get_res_name(op)
#     callee = op.callee.string_value()
#     res = b.call(callee, oprnds, name=res_name)
#     vals[res_name] = res


def parse_mlir_funcs(p: Path | str) -> list[FuncOp]:
    ctx = Context()
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(Transfer)

    func_str = p if isinstance(p, str) else p.read_text()
    func_name = "<text>" if isinstance(p, str) else p.name

    mod = Parser(ctx, func_str, func_name).parse_op()

    if isinstance(mod, ModuleOp):
        return [x for x in mod.ops if isinstance(x, FuncOp)]
    elif isinstance(mod, FuncOp):
        return [mod]
    else:
        raise ValueError(f"mlir in '{func_name}' is neither a ModuleOp, nor a FuncOp")


def lower_type(typ: Attribute, bw: int) -> ir.Type:
    # TODO only works for arity 2 domains (no IM)
    if isinstance(typ, TransIntegerType):
        return ir.IntType(bw)
    elif isinstance(typ, IntegerType):
        return ir.IntType(typ.width.data)
    elif isinstance(typ, AbstractValueType) or isinstance(typ, TupleType):
        fields = typ.get_fields()
        sub_type = lower_type(fields[0], bw)

        for other_type in fields:
            assert lower_type(other_type, bw) == sub_type

        return ir.ArrayType(sub_type, len(fields))

    raise ValueError("Unsupported Type", typ)


class LowerToLLVM:
    def __init__(self, bw: int, name: str) -> None:
        self.bw = bw
        self.llvm_mod = ir.Module(name=name)
        self.fns: dict[str, ir.Function] = {}

    def __str__(self) -> str:
        return str(self.llvm_mod)

    @staticmethod
    def add_attrs(fn: ir.Function) -> ir.Function:
        fn.attributes.add("nounwind")
        fn.attributes.add("alwaysinline")
        fn.attributes.add("readnone")
        fn.attributes.add("norecurse")

        return fn

    @staticmethod
    def is_concrete_op(mlir_fn: FuncOp) -> bool:
        fn_ret_type = lower_type(mlir_fn.function_type.outputs.data[0], 64)
        fn_arg_types = tuple(lower_type(x.type, 64) for x in mlir_fn.args)

        i64 = ir.IntType(64)
        ret_match = fn_ret_type == i64
        arg_match = fn_arg_types == (i64, i64)

        return ret_match and arg_match

    @staticmethod
    def is_transfer_fn(mlir_fn: FuncOp) -> bool:
        fn_ret_type = lower_type(mlir_fn.function_type.outputs.data[0], 64)
        fn_arg_types = tuple(lower_type(x.type, 64) for x in mlir_fn.args)

        i64 = ir.IntType(64)
        abst = ir.ArrayType(i64, 2)
        ret_match = fn_ret_type == abst
        arg_match = fn_arg_types == (abst, abst)

        return ret_match and arg_match

    def add_fn(
        self, mlir_fn: FuncOp, fn_name: str | None = None, shim: bool = False
    ) -> ir.Function:
        fn_name = fn_name if fn_name else mlir_fn.sym_name.data
        fn_ret_type = lower_type(mlir_fn.function_type.outputs.data[0], self.bw)
        fn_arg_types = (lower_type(x.type, self.bw) for x in mlir_fn.args)

        fn_type = ir.FunctionType(fn_ret_type, fn_arg_types)
        llvm_fn = ir.Function(self.llvm_mod, fn_type, name=fn_name)
        llvm_fn = self.add_attrs(llvm_fn)

        self.fns[fn_name] = LowerFuncToLLVM(mlir_fn, llvm_fn, self.bw).llvm_fn

        # TODO factor out this out into a class shim function
        if shim and self.is_concrete_op(mlir_fn):
            return self.shim_conc(self.fns[fn_name])
        elif shim and self.is_transfer_fn(mlir_fn):
            return self.shim_xfer(self.fns[fn_name])
        elif shim:
            raise ValueError(f"Cannot shim non concrete and non transfer function {fn_name}")

        return self.fns[fn_name]

    def add_fns(self, fns: list[FuncOp]) -> list[ir.Function]:
        return [self.add_fn(x) for x in fns]

    def shim_conc(self, old_fn: ir.Function) -> ir.Function:
        lane_t = ir.IntType(self.bw)
        wide_t = ir.IntType(64)

        fn_name = f"{old_fn.name}_shim"
        shim_ty = ir.FunctionType(wide_t, [wide_t, wide_t])
        shim_fn = ir.Function(self.llvm_mod, shim_ty, name=fn_name)
        shim_fn = self.add_attrs(shim_fn)

        entry = shim_fn.append_basic_block(name="entry")
        b = ir.IRBuilder(entry)

        a64, b64 = shim_fn.args
        a64.name = "a64"
        b64.name = "b64"

        a_n = a64 if self.bw == 64 else b.trunc(a64, lane_t)
        b_n = b64 if self.bw == 64 else b.trunc(b64, lane_t)
        r_n = b.call(old_fn, [a_n, b_n])
        r64 = r_n if self.bw == 64 else b.zext(r_n, wide_t)
        b.ret(r64)

        return shim_fn

    # def shim_xfer(self, old_fn: ir.Function) -> ir.Function:
    #     lane_t = ir.IntType(self.bw)
    #     wide_t = ir.IntType(self.bw * 2)

    #     fn_name = f"{old_fn.name}_shim"
    #     shim_ty = ir.FunctionType(wide_t, (wide_t, wide_t))
    #     shim_fn = ir.Function(self.llvm_mod, shim_ty, name=fn_name)
    #     shim_fn = self.add_attrs(shim_fn)

    #     b = ir.IRBuilder(shim_fn.append_basic_block(name="entry"))
    #     a0, a1 = shim_fn.args

    #     shift_amt = ir.Constant(wide_t, self.bw)

    #     def split_to_pair(wide_val: ir.Value) -> ir.Value:
    #         high_wide = b.lshr(wide_val, shift_amt)

    #         low_lane = b.trunc(wide_val, lane_t)
    #         high_lane = b.trunc(high_wide, lane_t)

    #         pair = ir.Constant(ir.ArrayType(lane_t, 2), None)
    #         pair = b.insert_value(pair, low_lane, 0)
    #         pair = b.insert_value(pair, high_lane, 1)
    #         return pair

    #     def combine_pair(pair_val: ir.Value) -> ir.Value:
    #         low_lane = b.extract_value(pair_val, 0)
    #         high_lane = b.extract_value(pair_val, 1)

    #         low_wide = b.zext(low_lane, wide_t)
    #         high_wide = b.zext(high_lane, wide_t)
    #         high_wide = b.shl(high_wide, shift_amt)

    #         return b.or_(high_wide, low_wide)  # type: ignore

    #     a0_pair = split_to_pair(a0)
    #     a1_pair = split_to_pair(a1)

    #     res_pair = b.call(old_fn, (a0_pair, a1_pair))

    #     res_wide = combine_pair(res_pair)
    #     b.ret(res_wide)

    #     self.fns[fn_name] = shim_fn

    #     return shim_fn

    def shim_xfer(self, old_fn: ir.Function) -> ir.Function:
        lane_t = ir.IntType(self.bw)
        i64 = ir.IntType(64)
        lane_arr_t = ir.ArrayType(lane_t, 2)
        i64_arr_t = ir.ArrayType(i64, 2)

        fn_name = f"{old_fn.name}_shim"
        shim_ty = ir.FunctionType(i64_arr_t, [i64_arr_t, i64_arr_t])
        shim_fn = ir.Function(self.llvm_mod, shim_ty, name=fn_name)
        shim_fn = self.add_attrs(shim_fn)

        b = ir.IRBuilder(shim_fn.append_basic_block(name="entry"))
        a64, b64 = shim_fn.args

        def to_lane(v):
            return v if self.bw == 64 else b.trunc(v, lane_t)

        a0 = to_lane(b.extract_value(a64, 0))
        a1 = to_lane(b.extract_value(a64, 1))
        b0 = to_lane(b.extract_value(b64, 0))
        b1 = to_lane(b.extract_value(b64, 1))

        empty_arr = ir.Constant(lane_arr_t, None)
        a_n = b.insert_value(empty_arr, a0, 0)
        a_n = b.insert_value(a_n, a1, 1)
        b_n = b.insert_value(empty_arr, b0, 0)
        b_n = b.insert_value(b_n, b1, 1)

        def to_i64(v):
            return v if self.bw == 64 else b.zext(v, i64)

        r_n = b.call(old_fn, [a_n, b_n])
        r0 = to_i64(b.extract_value(r_n, 0))
        r1 = to_i64(b.extract_value(r_n, 1))

        empty_i64_arr = ir.Constant(i64_arr_t, None)
        r = b.insert_value(empty_i64_arr, r0, 0)
        r = b.insert_value(r, r1, 1)
        b.ret(r)

        return shim_fn


class LowerFuncToLLVM:
    bw: int
    b: ir.IRBuilder
    ssa_map: dict[str, ir.Value]
    llvm_fn: ir.Function
    # TODO bind fn_map from Foo class for func calls
    # func_map

    def __init__(self, mlir_fn: FuncOp, llvm_fn: ir.Function, bw: int) -> None:
        self.bw = bw

        self.b = ir.IRBuilder(llvm_fn.append_basic_block(name="entry"))
        self.ssa_map = dict(zip((x.name_hint for x in mlir_fn.args), llvm_fn.args))  # type: ignore

        [self.add_op(op) for op in mlir_fn.walk() if not isinstance(op, FuncOp)]

        self.llvm_fn = llvm_fn

    def __str__(self) -> str:
        return str(self.llvm_fn)

    @staticmethod
    def result_name(op: Operation) -> str:
        ret_val = op.results[0].name_hint
        assert ret_val
        return ret_val

    def operands(self, op: Operation) -> tuple[ir.Value, ...]:
        return tuple(self.ssa_map[x.name_hint] for x in op.operands if x.name_hint)

    @singledispatchmethod
    def add_op(self, _: Operation) -> None:
        pass

    @add_op.register
    def _(self, op: Operation) -> None:
        llvm_op = _big_map[op.name]
        res_name = self.result_name(op)
        self.ssa_map[res_name] = llvm_op(self.b, *self.operands(op), name=res_name)

    @add_op.register
    def _(self, op: CountLOneOp | CountLZeroOp) -> None:
        res_name = self.result_name(op)
        true_const = ir.Constant(ir.IntType(1), 1)

        operand = self.operands(op)[0]
        if isinstance(op, CountLOneOp):
            operand = self.b.not_(operand, name=f"{res_name}_not")

        self.ssa_map[res_name] = self.b.ctlz(operand, true_const, name=res_name)  # type: ignore

    @add_op.register
    def _(self, op: CountROneOp | CountRZeroOp) -> None:
        res_name = self.result_name(op)
        true_const = ir.Constant(ir.IntType(1), 1)

        operand = self.operands(op)[0]
        if isinstance(op, CountROneOp):
            operand = self.b.not_(operand, name=f"{res_name}_not")

        self.ssa_map[res_name] = self.b.ctrz(operand, true_const, name=res_name)  # type: ignore

    @add_op.register
    def _(
        self,
        op: UAddOverflowOp | SAddOverflowOp | UMulOverflowOp | SMulOverflowOp,
        # | USubOverflowOp
        # | SSubOverflowOp,
    ) -> None:
        res_name = self.result_name(op)
        oprands = self.operands(op)

        d = {
            UAddOverflowOp: self.b.uadd_with_overflow,
            SAddOverflowOp: self.b.sadd_with_overflow,
            UMulOverflowOp: self.b.umul_with_overflow,
            SMulOverflowOp: self.b.smul_with_overflow,
            # USubOverflowOp: self.b.usub_with_overflow,
            # SSubOverflowOp: self.b.ssub_with_overflow,
        }

        ov = d[type(op)](oprands[0], oprands[1], name=f"{res_name}_ov")
        self.ssa_map[res_name] = self.b.extract_value(ov, 1, name=res_name)

    @add_op.register
    def _(self, op: UShlOverflowOp | SShlOverflowOp) -> None:
        res_name = self.result_name(op)
        oprnds = self.operands(op)
        typ = lower_type(op.results[0].type, self.bw)

        bw_const = ir.Constant(typ, self.bw)
        true_const = ir.Constant(ir.IntType(1), 1)
        cmp = self.b.icmp_unsigned(">=", oprnds[0], bw_const, name=f"{res_name}_cmp")

        shl = self.b.shl(oprnds[0], oprnds[1], name=f"{res_name}_shl")
        if isinstance(op, SShlOverflowOp):
            shr = self.b.ashr(shl, oprnds[1], name=f"{res_name}_ashr")
        elif isinstance(op, UShlOverflowOp):
            shr = self.b.lshr(shl, oprnds[1], name=f"{res_name}_ashr")

        ov = self.b.icmp_signed("!=", shr, oprnds[0], name=f"{res_name}_ov")
        self.ssa_map[res_name] = self.b.select(cmp, true_const, ov, name=f"{res_name}_ov")

    @add_op.register
    def _(self, op: GetOp) -> None:
        idx: int = op.attributes["index"].value.data  # type: ignore
        res_name = self.result_name(op)
        self.ssa_map[res_name] = self.b.extract_value(
            self.operands(op)[0], idx, name=res_name
        )

    @add_op.register
    def _(self, op: MakeOp) -> None:
        res_name = self.result_name(op)

        res = ir.Constant(lower_type(op.results[0].type, self.bw), None)
        for i, oprnd in enumerate(self.operands(op)):
            res = self.b.insert_value(res, oprnd, i, name=res_name)

        self.ssa_map[res_name] = res

    @add_op.register
    def _(self, op: ReturnOp) -> None:
        self.b.ret(self.operands(op)[0])

    @add_op.register
    def _(
        self,
        op: GetSignedMaxValueOp
        | GetSignedMinValueOp
        | GetAllOnesOp
        | GetBitWidthOp
        | Constant
        | ConstantOp,
    ) -> None:
        typ = lower_type(op.results[0].type, self.bw)
        res_name = self.result_name(op)

        if isinstance(op, GetSignedMaxValueOp):
            val = (2 ** (self.bw - 1)) - 1
        elif isinstance(op, GetSignedMinValueOp):
            val = 2 ** (self.bw - 1)
        elif isinstance(op, GetAllOnesOp):
            val = (2**self.bw) - 1
        elif isinstance(op, GetBitWidthOp):
            val = self.bw
        elif isinstance(op, Constant) or isinstance(op, ConstantOp):
            val: int = op.value.value.data  # type: ignore

        self.ssa_map[res_name] = ir.Constant(typ, val)

    @add_op.register
    def _(self, op: UMaxOp | UMinOp | SMaxOp | SMinOp) -> None:
        oprnds = self.operands(op)
        res_name = self.result_name(op)

        if isinstance(op, UMaxOp):
            cmp = self.b.icmp_unsigned(">", oprnds[0], oprnds[1], name=f"{res_name}_cmp")
        elif isinstance(op, UMinOp):
            cmp = self.b.icmp_unsigned("<", oprnds[0], oprnds[1], name=f"{res_name}_cmp")
        elif isinstance(op, SMaxOp):
            cmp = self.b.icmp_signed(">", oprnds[0], oprnds[1], name=f"{res_name}_cmp")
        elif isinstance(op, SMinOp):
            cmp = self.b.icmp_signed("<", oprnds[0], oprnds[1], name=f"{res_name}_cmp")

        self.ssa_map[res_name] = self.b.select(cmp, oprnds[0], oprnds[1], name=res_name)

    @add_op.register
    def _(self, op: IsNegativeOp) -> None:
        oprnds = self.operands(op)
        res_name = self.result_name(op)
        typ = lower_type(op.results[0].type, self.bw)

        const_zero = ir.Constant(typ, 0)
        self.ssa_map[res_name] = self.b.icmp_signed(
            "<", oprnds[0], const_zero, name=res_name
        )

    @add_op.register
    def _(self, op: CmpOp) -> None:
        s = self.b.icmp_signed
        us = self.b.icmp_unsigned
        cmp_sign_map = [s, s, s, s, s, s, us, us, us, us]
        cmp_pred_map = ["==", "!=", "<", "<=", ">", ">=", "<", "<=", ">", ">="]

        oprnds = self.operands(op)
        cmp_pred = op.predicate.value.data
        res_name = self.result_name(op)

        self.ssa_map[res_name] = cmp_sign_map[cmp_pred](
            cmp_pred_map[cmp_pred], oprnds[0], oprnds[1], name=res_name
        )


def _create_exec_engine() -> tuple[llvm.ExecutionEngine, llvm.TargetMachine, llvm.Target]:
    "This engine is reusable for an arbitrary number of modules."

    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    target = llvm.Target.from_default_triple()
    tm = target.create_target_machine(
        cpu=llvm.get_host_cpu_name(),
        features=llvm.get_host_cpu_features().flatten(),
        opt=2,
    )
    backing_mod = llvm.parse_assembly("")

    return llvm.create_mcjit_compiler(backing_mod, tm), tm, target


class Jit:
    engine, tm, target = _create_exec_engine()

    def __init__(self, llvm_ir: str | ir.Module | LowerToLLVM) -> None:
        if not isinstance(llvm_ir, str):
            llvm_ir = str(llvm_ir)

        mod = llvm.parse_assembly(llvm_ir)
        mod.triple = self.target.triple
        mod.data_layout = str(self.tm.target_data)
        mod.verify()

        pb = llvm.PassBuilder(self.tm, llvm.PipelineTuningOptions())
        mpm = pb.getModulePassManager()
        mpm.add_aggressive_dce_pass()
        mpm.add_aa_eval_pass()
        mpm.add_aggressive_instcombine_pass()
        mpm.add_simplify_cfg_pass()
        mpm.add_constant_merge_pass()
        mpm.add_rpo_function_attrs_pass()
        mpm.run(mod, pb)

        self.engine.add_module(mod)
        self.engine.finalize_object()
        self.engine.run_static_constructors()

        self.mod = mod

    def get_fn_ptr(self, fn: str | ir.Function) -> int:
        if isinstance(fn, ir.Function):
            fn = str(fn.name)

        return self.engine.get_function_address(fn)
