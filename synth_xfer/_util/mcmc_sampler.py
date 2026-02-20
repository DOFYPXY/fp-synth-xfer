from typing import Callable

import xdsl.dialects.arith as arith
from xdsl.dialects.builtin import FunctionType, IntegerAttr, UnitAttr, i1
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import OpResult
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
from synth_xfer._util.domain import AbstractDomain
from dataclasses import dataclass

# We add this dataclass to test mutations in isolations
# TODO: Remove this after testing 
@dataclass
class MutationFlags:
    replace_entire_op: bool = True # Exists 
    replace_operand: bool = True # Exists
    rewire_make_op: bool = False # New
    perturb_constant: bool = False # New
    replace_op_window: bool = False # New
    
class MCMCSampler:
    current: MutationProgram
    current_cmp: EvalResult
    context: SynthesizerContext
    random: Random
    cost_func: Callable[[EvalResult, float], float]
    step_cnt: int
    total_steps: int
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
        reset_init_program: bool = True,
        random_init_program: bool = True,
        is_cond: bool = False,
        flags: MutationFlags | None = None,
    ):
        self.is_cond = is_cond
        self.flags = flags or MutationFlags() 
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
        self.current.remove_history()
        self.current_cmp = proposed_cmp
        self.step_cnt += 1

    def reject_proposed(self):
        self.current.revert_operation()
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
        make_op_idx = len(ops) - 2
        make_op = ops[make_op_idx]
        assert isinstance(make_op, MakeOp)

        # Pick a random operand of MakeOp to rewire
        ith = self.context.random.randint(0, len(make_op.operands) - 1)

        # Clone so we can track history
        new_make_op = make_op.clone()
        self.current.subst_operation(make_op, new_make_op, history)

        # Get valid int operands before MakeOp
        vals = self.current.get_valid_operands(make_op_idx, INT_T)
        if not vals:
            # Nothing to rewire, revert silently
            self.current.subst_operation(new_make_op, new_make_op.clone(), False)
            return

        success = False
        while not success:
            success = self.context.replace_operand(new_make_op, ith, vals)

    def perturb_constant(self, history: bool):
        ops = self.current.ops
        const_ops = [
            (op, idx) for idx, op in enumerate(ops)
            if isinstance(op, Constant)
        ]
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
        else:
            candidates = self._cr_constant_candidates()

        filtered = [c for c in candidates if c != current_val]
        if not filtered:
            filtered = candidates  # fallback if all candidates match current
        return self.random.choice(filtered)

    def _kb_constant_candidates(self) -> list[int]:
        return [0, 1]

    def _cr_constant_candidates(self) -> list[int]:
        return [0, 1, 2, 4, 8, 16, 32, 64, 128]

    def _fp_constant_candidates(self) -> list[int]:
        return [
            0,              # 0.0
            0x80000000,     # -0.0
            0x3F800000,     # 1.0
            0xBF800000,     # -1.0
            0x7F800000,     # +inf
            0xFF800000,     # -inf
            0x7F7FFFFF,     # MAX_FLOAT
            0xFF7FFFFF,     # -MAX_FLOAT
            0x00800000,     # MIN_FLOAT (smallest normal)
            0x7FC00000,     # NaN
        ]
            
    def construct_init_program(self, _func: FuncOp, length: int):
        func = _func.clone()
        block = func.body.block
        for op in block.ops:
            block.detach_op(op)

        if self.context.weighted:
            func.attributes["from_weighted_dsl"] = UnitAttr()

        # Part I: GetOp
        for arg in block.args:
            if isinstance(arg.type, AbstractValueType):
                for i, field_type in enumerate(arg.type.get_fields()):
                    op = GetOp(arg, i)
                    block.add_op(op)

        assert isinstance(block.last_op, GetOp)
        tmp_int_ssavalue = block.last_op.results[0]

        # Part II: Constants
        true: arith.ConstantOp = arith.ConstantOp(
            IntegerAttr.from_int_and_width(1, 1), i1
        )
        false: arith.ConstantOp = arith.ConstantOp(
            IntegerAttr.from_int_and_width(0, 1), i1
        )
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
                if i % 4 == 0:
                    nop_bool = CmpOp(tmp_int_ssavalue, tmp_int_ssavalue, 0)
                    block.add_op(nop_bool)
                elif i % 4 == 1:
                    int_nop = AddOp(tmp_int_ssavalue, tmp_int_ssavalue)
                    block.add_op(int_nop)
                elif i % 4 == 2:
                    last_int_op = AndOp(tmp_int_ssavalue, tmp_int_ssavalue)
                    block.add_op(last_int_op)
                else:
                    last_int_op = AndOp(tmp_int_ssavalue, tmp_int_ssavalue)
                    block.add_op(last_int_op)

            # Part IV: MakeOp
            output = list(func.function_type.outputs)[0]
            assert isinstance(output, AbstractValueType)
            operands: list[OpResult] = []
            for i, field_type in enumerate(output.get_fields()):
                assert isinstance(field_type, TransIntegerType)
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
            # Part III: Main Body
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

    def sample_next(self):
        live_ops = self.current.get_modifiable_operations()
        live_op_indices = [x[1] for x in live_ops]

        choices = []
        if self.flags.replace_entire_op and live_op_indices:
            choices.append(('replace_entire', 3))
        if self.flags.replace_operand and live_op_indices:
            choices.append(('replace_operand', 7))
        if self.flags.rewire_make_op:
            choices.append(('rewire_make', 1))
        if self.flags.perturb_constant:
            choices.append(('perturb_const', 1))
        if not choices:
            return self

        names = [c[0] for c in choices]
        weights = [c[1] for c in choices]
        mutation = self.random.choice_weighted(names, dict(zip(names, weights)))

        if mutation == 'replace_entire':
            self.replace_entire_operation(self.random.choice(live_op_indices), True)
        elif mutation == 'replace_operand':
            self.replace_operand(self.random.choice(live_op_indices), True)
        elif mutation == 'rewire_make':
            self.rewire_make_op(True)
        elif mutation == 'perturb_const':
            self.perturb_constant(True)
        elif mutation == 'window':
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
            )

        mcmc_samplers.append(spl)

    return mcmc_samplers, prec_set_after_distribute, (sp_range, p_range, c_range)
