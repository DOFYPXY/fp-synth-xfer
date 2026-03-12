from dataclasses import dataclass
from typing import Callable

import xdsl.dialects.arith as arith
from xdsl.dialects.builtin import FunctionType, IntegerAttr, IntegerType, UnitAttr, i1
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Operation, OpResult
from xdsl_smt.dialects.transfer import (
    AbstractValueType,
    AddOp,
    AndOp,
    CmpOp,
    Constant,
    GetAllOnesOp,
    GetBitWidthOp,
    GetOp,
    MakeOp,
    TransIntegerType,
)

from synth_xfer._util.cost_model import (
    abduction_cost,
    precise_cost,
    sound_and_precise_cost,
)
from synth_xfer._util.domain import AbstractDomain
from synth_xfer._util.dsl_operators import BOOL_T, INT_T, get_operand_kinds
from synth_xfer._util.eval_result import EvalResult
from synth_xfer._util.mutation_program import MutationProgram
from synth_xfer._util.random import Random
from synth_xfer._util.synth_context import (
    SynthesizerContext,
    get_ret_type,
    is_int_op,
    not_in_main_body,
)
from synth_xfer.dialects.fp import (
    FloatType,
    FPAbsValueType,
    FPAddOp,
    FPCmpOp,
    FPConstantOp,
    FPGetOp,
    FPMakeOp,
)


@dataclass
class MutationFlags:
    replace_entire_op: bool = True  # Exists
    replace_operand: bool = True  # Exists
    rewire_make_op: bool = False  # New
    perturb_constant: bool = False  # New
    replace_op_window: bool = False  # New


class MCMCSampler:
    current: MutationProgram
    current_cmp: EvalResult
    context: SynthesizerContext
    random: Random
    cost_func: Callable[[EvalResult, float], float]
    step_cnt: int
    total_steps: int
    bw: int
    is_cond: bool
    length: int
    flags: MutationFlags

    def __init__(
        self,
        func: FuncOp,
        context: SynthesizerContext,
        cost_func: Callable[[EvalResult, float], float],
        length: int,
        total_steps: int,
        bw: int = 4,
        reset_init_program: bool = True,
        random_init_program: bool = True,
        is_cond: bool = False,
        flags: MutationFlags | None = None,
    ):
        self.bw = bw
        self.is_cond = is_cond
        self.flags = flags
        if is_cond:
            cond_type = FunctionType.from_lists(
                func.function_type.inputs,  # pyright: ignore [reportArgumentType]
                [i1],
            )
            func = FuncOp("cond", cond_type)
        self.context = context
        self.cost_func = cost_func
        self.total_steps = total_steps
        self.step_cnt = 0
        self.random = context.get_random_class()
        self.length = length
        if reset_init_program:
            self.current = self.construct_init_program(func, self.length)
            if random_init_program:
                self.reset_to_random_prog()

    def compute_cost(self, cmp: EvalResult) -> float:
        return self.cost_func(cmp, self.step_cnt / self.total_steps)

    def compute_current_cost(self):
        return self.compute_cost(self.current_cmp)

    def get_current(self):
        return self.current.func

    def accept_proposed(self, proposed_cmp: EvalResult):
        if self.current.old_ops is not None:
            self.current.remove_history_window()
        elif self.current.old_op is not None:
            self.current.remove_history()
        # else: mutation was a no-op (e.g. rewire_make_op found no valid operands)
        self.current_cmp = proposed_cmp
        self.step_cnt += 1

    def reject_proposed(self):
        if self.current.old_ops is not None:
            self.current.revert_window()
        elif self.current.old_op is not None:
            self.current.revert_operation()
        # else: mutation was a no-op, nothing to revert
        self.step_cnt += 1

    def replace_entire_operation(self, idx: int, history: bool):
        """
        Random pick an operation and replace it with a new one
        """
        old_op = self.current.ops[idx]
        valid_operands = {
            ty: self.current.get_valid_operands(idx, ty) for ty in [INT_T, BOOL_T]
        }
        new_op = None
        while new_op is None:
            new_op = self.context.get_random_op(get_ret_type(old_op), valid_operands)

        self.current.subst_operation(old_op, new_op, history)

    def replace_operand(self, idx: int, history: bool):
        op = self.current.ops[idx]
        new_op = op.clone()

        self.current.subst_operation(op, new_op, history)

        ith = self.context.random.randint(0, len(op.operands) - 1)
        operand_kinds = get_operand_kinds(type(op))

        vals = self.current.get_valid_operands(idx, operand_kinds[ith])

        success = False
        while not success:
            success = self.context.replace_operand(new_op, ith, vals)

    def rewire_make_op(self, history: bool):
        ops = self.current.ops
        make_op = next(
            (op for op in reversed(ops) if isinstance(op, (MakeOp, FPMakeOp))), None
        )
        if make_op is None:
            return

        make_op_idx = ops.index(make_op)
        ith = self.context.random.randint(0, len(make_op.operands) - 1)

        new_make_op = make_op.clone()
        self.current.subst_operation(make_op, new_make_op, history)

        # FPMakeOp operand 2 (has_nan) is i1 (BOOL_T), others are float (INT_T)
        if isinstance(new_make_op, FPMakeOp) and ith == 2:
            vals = self.current.get_valid_operands(make_op_idx, BOOL_T)
        else:
            vals = self.current.get_valid_operands(make_op_idx, INT_T)

        if not vals:
            self.current.subst_operation(new_make_op, new_make_op.clone(), False)
            return

        success = False
        while not success:
            success = self.context.replace_operand(new_make_op, ith, vals)

    def _has_useful_perturbation(self) -> bool:
        """Returns False for domains where constant perturbation adds no value."""
        domain = self.context.domain
        if domain == AbstractDomain.KnownBits:
            return False  # only 0/1 are meaningful, already covered by replace_entire
        return True

    def perturb_constant(self, history: bool):
        ops = self.current.ops
        const_ops = [(op, idx) for idx, op in enumerate(ops) if isinstance(op, Constant)]
        if not const_ops:
            return

        old_const, idx = self.random.choice(const_ops)
        ref_val = old_const.op

        new_val = self._sample_perturbed_constant(old_const.value.value.data)
        new_const = Constant(ref_val, new_val)
        self.current.subst_operation(old_const, new_const, history)

    def _sample_perturbed_constant(self, current_val: int) -> int:
        domain = self.context.domain
        if domain == AbstractDomain.FPRange:
            candidates = self._fp_constant_candidates()
        elif domain == AbstractDomain.KnownBits:
            candidates = self._kb_constant_candidates()
            max_val = (1 << self.bw) - 1
            candidates = list({c & max_val for c in candidates})
        else:
            candidates = self._cr_constant_candidates()
            max_val = (1 << self.bw) - 1
            candidates = list({c & max_val for c in candidates})

        filtered = [c for c in candidates if c != current_val]
        if not filtered:
            filtered = candidates
        return self.random.choice(filtered)

    def _kb_constant_candidates(self) -> list[int]:
        return [0, 1]

    def _cr_constant_candidates(self) -> list[int]:
        return [0, 1, 2, 4, 8, 16, 32, 64, 128]

    def _fp_constant_candidates(self) -> list[int]:
        return [
            0,  # 0.0
            0x80000000,  # -0.0
            0x3F800000,  # 1.0
            0xBF800000,  # -1.0
            0x7F800000,  # +inf
            0xFF800000,  # -inf
            0x7F7FFFFF,  # MAX_FLOAT
            0xFF7FFFFF,  # -MAX_FLOAT
            0x00800000,  # MIN_FLOAT (smallest normal)
            0x7FC00000,  # NaN
        ]

    def replace_op_window(self, history: bool):
        live_ops = self.current.get_modifiable_operations()
        if len(live_ops) < 2:
            return

        window_size = min(self.random.randint(2, 3), len(live_ops))
        start = self.random.randint(0, len(live_ops) - window_size)
        window_indices = [idx for _, idx in live_ops[start : start + window_size]]

        # Save a clone of the entire function as backup
        backup_func = self.current.func.clone()

        for orig_idx in window_indices:
            ops = self.current.ops
            op = ops[orig_idx]

            valid_operands = {
                ty: self.current.get_valid_operands(orig_idx, ty)
                for ty in [INT_T, BOOL_T]
            }
            new_op = None
            while new_op is None:
                new_op = self.context.get_random_op(get_ret_type(op), valid_operands)

            block = self.current.func.body.block
            block.insert_op_before(new_op, op)
            if len(op.results) > 0 and len(new_op.results) > 0:
                op.results[0].replace_by(new_op.results[0])
            block.detach_op(op)
            op.erase()

        if history:
            self.current.old_ops = []
            self.current.new_ops = []
            self.current._backup_func = backup_func

    def construct_init_program(self, _func: FuncOp, length: int):
        func = _func.clone()
        block = func.body.block
        for op in block.ops:
            block.detach_op(op)
        if self.context.weighted:
            func.attributes["from_weighted_dsl"] = UnitAttr()

        if self.context.domain == AbstractDomain.FPRange:
            return self._construct_init_program_fp(func, block, length)
        else:
            return self._construct_init_program_int(func, block, length)

    def _construct_init_program_int(self, func: FuncOp, block, length: int):
        # Part I: GetOp
        for arg in block.args:
            if isinstance(arg.type, AbstractValueType):
                for i, field_type in enumerate(arg.type.get_fields()):
                    op = GetOp(arg, i)
                    block.add_op(op)

        assert isinstance(block.last_op, GetOp)
        tmp_int_ssavalue = block.last_op.results[0]

        # Part II: Constants
        true = arith.ConstantOp(IntegerAttr.from_int_and_width(1, 1), i1)
        false = arith.ConstantOp(IntegerAttr.from_int_and_width(0, 1), i1)
        all_ones = GetAllOnesOp(tmp_int_ssavalue)
        zero = Constant(tmp_int_ssavalue, 0)
        one = Constant(tmp_int_ssavalue, 1)
        get_bw = GetBitWidthOp(tmp_int_ssavalue)
        block.add_op(true)
        block.add_op(false)
        block.add_op(zero)
        block.add_op(one)
        block.add_op(all_ones)
        block.add_op(get_bw)

        if not self.is_cond:
            # Part III: Main Body
            last_int_op = block.last_op
            for i in range(length):
                if i % 6 == 0:
                    nop_bool = CmpOp(tmp_int_ssavalue, tmp_int_ssavalue, 0)
                    block.add_op(nop_bool)
                elif i % 3 == 0:
                    last_int_op = AddOp(tmp_int_ssavalue, tmp_int_ssavalue)
                    block.add_op(last_int_op)
                else:
                    last_int_op = AndOp(tmp_int_ssavalue, tmp_int_ssavalue)
                    block.add_op(last_int_op)

            # Part IV: MakeOp
            output = list(func.function_type.outputs)[0]
            assert isinstance(output, AbstractValueType)
            operands: list[OpResult] = []
            for i, field_type in enumerate(output.get_fields()):
                assert isinstance(field_type, TransIntegerType) or isinstance(
                    field_type, IntegerType
                )
                assert last_int_op is not None
                operands.append(last_int_op.results[0])
                while True:
                    last_int_op = last_int_op.prev_op
                    assert last_int_op is not None
                    if is_int_op(last_int_op):
                        break

            return_val = MakeOp(operands)
            block.add_op(return_val)

        else:
            # Part III: Main Body (cond)
            last_bool_op = true
            for i in range(length):
                if i % 4 == 0:
                    last_int_op = AndOp(tmp_int_ssavalue, tmp_int_ssavalue)
                    block.add_op(last_int_op)
                else:
                    last_bool_op = CmpOp(tmp_int_ssavalue, tmp_int_ssavalue, 0)
                    block.add_op(last_bool_op)
            return_val = last_bool_op.results[0]

        # Part V: Return
        block.add_op(ReturnOp(return_val))
        return MutationProgram(func)

    def _construct_init_program_fp(self, func: FuncOp, block, length: int):
        # Part I: FPGetOp — extract (lo: FloatType, hi: FloatType, has_nan: i1)
        # from each FPAbsValueType arg
        fp_field_types = [FloatType(), FloatType(), IntegerType(1)]
        for arg in block.args:
            if isinstance(arg.type, FPAbsValueType):
                for i, field_type in enumerate(fp_field_types):
                    op = FPGetOp(arg, i, field_type)
                    block.add_op(op)

        assert isinstance(block.last_op, FPGetOp), (
            f"Expected at least one FPAbsValueType arg, got last_op={block.last_op}"
        )
        tmp_fp_ssavalue = block.last_op.results[0]  # FloatType

        # Part II: FP Constants (seed values for body construction)
        true = arith.ConstantOp(IntegerAttr.from_int_and_width(1, 1), i1)
        false = arith.ConstantOp(IntegerAttr.from_int_and_width(0, 1), i1)
        fp_zero = FPConstantOp(0.0)
        fp_one = FPConstantOp(1.0)
        block.add_op(true)
        block.add_op(false)
        block.add_op(fp_zero)
        block.add_op(fp_one)

        if not self.is_cond:
            # Part III: Main Body — nop FP ops seeded from tmp_fp_ssavalue
            last_fp_op: Operation = fp_one
            for i in range(length):
                if i % 3 == 0:
                    # bool nop
                    nop_bool = FPCmpOp(tmp_fp_ssavalue, tmp_fp_ssavalue, "oeq")
                    block.add_op(nop_bool)
                else:
                    last_fp_op = FPAddOp(tmp_fp_ssavalue, tmp_fp_ssavalue)
                    block.add_op(last_fp_op)

            # Part IV: FPMakeOp(lo, hi, has_nan) → FPAbsValueType
            output = list(func.function_type.outputs)[0]
            assert isinstance(output, FPAbsValueType), (
                f"Expected FPAbsValueType output, got {output}"
            )
            # Find two distinct FloatType ops for lo/hi, use `true` for has_nan
            lo_op = last_fp_op
            # Walk back to find a second FloatType op for hi
            hi_op: Operation | None = lo_op.prev_op
            while hi_op is not None:
                if hi_op.results and isinstance(hi_op.results[0].type, FloatType):
                    break
                hi_op = hi_op.prev_op
            if hi_op is None:
                hi_op = fp_zero  # fallback: both lo and hi from same constant

            return_val = FPMakeOp(
                lo_op.results[0],
                hi_op.results[0],
                true.results[0],  # has_nan=true is conservative
            )
            block.add_op(return_val)

        else:
            # Part III (cond): bool body seeded from FP comparisons
            last_bool_op: Operation = true
            for i in range(length):
                if i % 4 == 0:
                    last_fp_op = FPAddOp(tmp_fp_ssavalue, tmp_fp_ssavalue)
                    block.add_op(last_fp_op)
                else:
                    last_bool_op = FPCmpOp(tmp_fp_ssavalue, tmp_fp_ssavalue, "oeq")
                    block.add_op(last_bool_op)
            return_val = last_bool_op.results[0]

        # Part V: Return
        block.add_op(ReturnOp(return_val))
        return MutationProgram(func)

    def sample_next(self):
        live_ops = self.current.get_modifiable_operations()
        live_op_indices = [x[1] for x in live_ops]

        choices = []
        if self.flags.replace_entire_op and live_op_indices:
            choices.append(("replace_entire", 2))
        if self.flags.replace_operand and live_op_indices:
            choices.append(("replace_operand", 2))
        if self.flags.rewire_make_op:
            choices.append(("rewire_make", 2))
        if self.flags.perturb_constant and self._has_useful_perturbation():
            choices.append(("perturb_const", 2))
        if self.flags.replace_op_window and len(live_op_indices) >= 2:
            choices.append(("window", 2))
        if not choices:
            return self

        names = [c[0] for c in choices]
        weights = [c[1] for c in choices]
        mutation = self.random.choice_weighted(names, dict(zip(names, weights)))

        if mutation == "replace_entire":
            self.replace_entire_operation(self.random.choice(live_op_indices), True)
        elif mutation == "replace_operand":
            self.replace_operand(self.random.choice(live_op_indices), True)
        elif mutation == "rewire_make":
            self.rewire_make_op(True)
        elif mutation == "perturb_const":
            self.perturb_constant(True)
        elif mutation == "window":
            self.replace_op_window(True)

        return self

    def reset_to_random_prog(self):
        # Part III-2: Random reset main body
        total_ops_len = len(self.current.ops)
        # Only modify ops in the main body
        for i in range(total_ops_len):
            if not not_in_main_body(self.current.ops[i]):
                self.replace_entire_operation(i, False)


def setup_mcmc(
    transfer_func: FuncOp,
    precise_set: list[FuncOp],
    num_abd_proc: int,
    num_mcmc: int,
    context_regular: SynthesizerContext,
    context_weighted: SynthesizerContext,
    context_cond: SynthesizerContext,
    program_length: int,
    num_steps: int,
    cond_length: int,
    bw: int = 4,
    mutation_flags: set[str] | None = None,
) -> tuple[list[MCMCSampler], list[FuncOp], tuple[range, range, range]]:
    """
    A mcmc sampler use one of 3 modes: sound & precise, precise, condition
    This function specify which mode should be used for each mcmc sampler
    For example, mcmc samplers with index in sp_range should use "sound&precise"
    """

    p_size = 0
    c_size = num_abd_proc
    sp_size = num_mcmc - p_size - c_size

    if len(precise_set) == 0:
        sp_size += c_size
        c_size = 0

    sp_range = range(0, sp_size)
    p_range = range(sp_size, sp_size + p_size)
    c_range = range(sp_size + p_size, sp_size + p_size + c_size)

    prec_set_after_distribute: list[FuncOp] = []

    if c_size > 0:
        # Distribute the precise funcs into c_range
        prec_set_size = len(precise_set)
        base_count = c_size // prec_set_size
        remainder = c_size % prec_set_size
        for i, item in enumerate(precise_set):
            for _ in range(base_count + (1 if i < remainder else 0)):
                prec_set_after_distribute.append(item.clone())

    mutation_flags_obj = MutationFlags(
        replace_entire_op="replace_entire_op" in mutation_flags,
        replace_operand="replace_operand" in mutation_flags,
        rewire_make_op="rewire_make_op" in mutation_flags,
        perturb_constant="perturb_constant" in mutation_flags,
        replace_op_window="replace_op_window" in mutation_flags,
    )

    mcmc_samplers: list[MCMCSampler] = []
    for i in range(num_mcmc):
        if i in sp_range:
            spl = MCMCSampler(
                transfer_func,
                context_regular
                if i < (sp_range.start + sp_range.stop) // 2
                else context_weighted,
                sound_and_precise_cost,
                program_length,
                num_steps,
                random_init_program=True,
                bw=bw,
                flags=mutation_flags_obj,
            )
        elif i in p_range:
            spl = MCMCSampler(
                transfer_func,
                context_regular
                if i < (p_range.start + p_range.stop) // 2
                else context_weighted,
                precise_cost,
                program_length,
                num_steps,
                random_init_program=True,
                bw=bw,
                flags=mutation_flags_obj,
            )
        else:
            spl = MCMCSampler(
                transfer_func,
                context_cond,
                abduction_cost,
                cond_length,
                num_steps,
                random_init_program=True,
                is_cond=True,
                bw=bw,
                flags=mutation_flags_obj,
            )

        mcmc_samplers.append(spl)

    return mcmc_samplers, prec_set_after_distribute, (sp_range, p_range, c_range)
