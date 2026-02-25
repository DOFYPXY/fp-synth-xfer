from pathlib import Path

from xdsl.dialects.func import ReturnOp
from xdsl_smt.dialects.transfer import Constant, MakeOp

from synth_xfer._util.cost_model import sound_and_precise_cost
from synth_xfer._util.domain import AbstractDomain
from synth_xfer._util.dsl_operators import BOOL_T, INT_T
from synth_xfer._util.mcmc_sampler import MCMCSampler, MutationFlags
from synth_xfer._util.mutation_program import MutationProgram
from synth_xfer._util.parse_mlir import get_helper_funcs
from synth_xfer._util.random import Random
from synth_xfer._util.synth_context import (
	SynthesizerContext,
	get_ret_type,
	not_in_main_body,
)

PROJ_DIR = Path(__file__).parent.parent


# Helper functions
def make_context(seed: int = 42, domain: AbstractDomain | None = None) -> SynthesizerContext:
	random = Random(seed)
	return SynthesizerContext(random, domain=domain)


def make_sampler(seed: int = 42, length: int = 8, flags: MutationFlags | None = None, domain: AbstractDomain | None = None) -> MCMCSampler:
	conc_nop_f = PROJ_DIR / "mlir" / "Operations" / "Nop.mlir"
	helpers = get_helper_funcs(conc_nop_f, AbstractDomain.KnownBits)
	transfer_func = helpers.transfer_func 
	context = make_context(seed, domain)
	return MCMCSampler(
		transfer_func,
		context,
		sound_and_precise_cost,
		length=length,
		total_steps=100,
		reset_init_program=True,
		random_init_program=False,
  		flags=flags
	)


def assert_program_valid(prog: MutationProgram):
	"""
	Check structural invariants that must hold after any mutation:
	1. Last op is ReturnOp
	2. Second-to-last op is MakeOp (for non-cond programs)
	3. Every operand's owner is defined before its use (SSA dominance)
	4. No detached ops with live uses
	"""
	ops = prog.ops
	assert len(ops) >= 2, "Program must have at least MakeOp + ReturnOp"

	# Invariant 1: last op is ReturnOp
	assert isinstance(ops[-1], ReturnOp), \
		f"Last op must be ReturnOp, got {type(ops[-1])}"

	# Invariant 2: second-to-last is MakeOp (regular, non-cond program)
	assert isinstance(ops[-2], MakeOp), \
		f"Second-to-last op must be MakeOp, got {type(ops[-2])}"

	# Invariant 3: SSA dominance — every operand defined before its use
	defined = set()
	for arg in prog.func.body.block.args:
		defined.add(arg)
	for op in ops:
		for operand in op.operands:
			assert operand in defined, \
				f"Op {type(op)} uses value not yet defined (SSA violation)"
		for result in op.results:
			defined.add(result)

	# Invariant 4: MakeOp operands are defined
	make_op = ops[-2]
	for operand in make_op.operands:
		assert operand in defined, \
			"MakeOp uses undefined value"


# Mutation primitives
class TestMutationProgramPrimitives:
	def test_construct_init_program_is_valid(self):
		spl = make_sampler()
		assert_program_valid(spl.current)

	def test_ops_count_matches_length(self):
		length = 8
		spl = make_sampler(length=length)
		ops = spl.current.ops
		main_body_ops = [op for op in ops if not not_in_main_body(op)]
		assert len(main_body_ops) == length, \
			f"Expected {length} main body ops, got {len(main_body_ops)}"

	def test_subst_operation_preserves_validity(self):
		spl = make_sampler()
		live_ops = spl.current.get_modifiable_operations()
		assert len(live_ops) > 0, "Should have at least one modifiable op"

		op, idx = live_ops[0]
		ret_type = get_ret_type(op)
		valid_operands = {
			ty: spl.current.get_valid_operands(idx, ty)
			for ty in [INT_T, BOOL_T]
		}
		new_op = None
		while new_op is None:
			new_op = spl.context.get_random_op(ret_type, valid_operands)

		spl.current.subst_operation(op, new_op, history=True)
		assert_program_valid(spl.current)

	def test_revert_operation_restores_program(self):
		spl = make_sampler(seed=0)
		# Snapshot ops before mutation
		ops_before = [type(op) for op in spl.current.ops]

		spl.replace_entire_operation(
			spl.current.get_modifiable_operations()[0][1],
			history=True
		)
		# After mutation, history should be set
		assert spl.current.old_op is not None
		assert spl.current.new_op is not None

		spl.current.revert_operation()

		# History should be cleared
		assert spl.current.old_op is None
		assert spl.current.new_op is None

		# Op types should match original
		ops_after = [type(op) for op in spl.current.ops]
		assert ops_before == ops_after, \
			"revert_operation should restore original op sequence"

	def test_remove_history_clears_state(self):
		spl = make_sampler()
		live_ops = spl.current.get_modifiable_operations()
		spl.replace_entire_operation(live_ops[0][1], history=True)

		assert spl.current.old_op is not None
		spl.current.remove_history()

		assert spl.current.old_op is None
		assert spl.current.new_op is None
		assert_program_valid(spl.current)

	def test_get_modifiable_operations_are_live(self):
		"""All returned ops should be reachable from MakeOp."""
		spl = make_sampler()
		live_ops = spl.current.get_modifiable_operations(only_live=True)
		all_ops = spl.current.get_modifiable_operations(only_live=False)
		# Live set should be a subset of all ops
		live_indices = {idx for _, idx in live_ops}
		all_indices = {idx for _, idx in all_ops}
		assert live_indices.issubset(all_indices)

	def test_get_valid_operands_respects_dominance(self):
		"""Valid operands at index x must all come from ops before x."""
		from synth_xfer._util.dsl_operators import INT_T
		spl = make_sampler()
		ops = spl.current.ops
		for idx in range(len(ops)):
			valid = spl.current.get_valid_operands(idx, INT_T)
			for val in valid:
				owner_idx = ops.index(val.owner)
				assert owner_idx < idx, \
					f"Operand at owner idx {owner_idx} violates dominance for use at {idx}"


# SampleNext
class TestSampleNext:
	def test_single_sample_preserves_validity(self):
		spl = make_sampler()
		spl.sample_next()
		assert_program_valid(spl.current)

	def test_many_samples_preserve_validity(self):
		spl = make_sampler(seed=7)
		for _ in range(50):
			spl.sample_next()
			assert_program_valid(spl.current)
			# Simulate accept (removes history)
			if spl.current.old_op is not None:
				spl.current.remove_history()

	def test_reject_then_sample_is_valid(self):
		spl = make_sampler(seed=13)
		for _ in range(20):
			spl.sample_next()
			assert_program_valid(spl.current)
			# Always reject
			if spl.current.old_op is not None:
				spl.current.revert_operation()
			assert_program_valid(spl.current)

	def test_replace_entire_operation_changes_op_type(self):
		"""replace_entire_operation should produce a structurally valid program
		and the op at that index should potentially differ."""
		spl = make_sampler(seed=99)
		live_ops = spl.current.get_modifiable_operations()
		assert len(live_ops) > 0
		idx = live_ops[0][1]

		# Run multiple times so that we can see a change
		changed = False
		original_type = type(spl.current.ops[idx])
		for _ in range(20):
			spl.replace_entire_operation(idx, history=True)
			assert_program_valid(spl.current)
			if type(spl.current.ops[idx]) != original_type:
				changed = True
			spl.current.revert_operation()

		# With 20 tries and a random op set, we ideally want to see at least one change
		assert changed, "replace_entire_operation never changed the op type"

	def test_replace_operand_preserves_validity(self):
		spl = make_sampler(seed=5)
		live_ops = spl.current.get_modifiable_operations()
		assert len(live_ops) > 0
		idx = live_ops[0][1]

		for _ in range(10):
			spl.replace_operand(idx, history=True)
			assert_program_valid(spl.current)
			spl.current.revert_operation()

	def test_reset_to_random_prog_is_valid(self):
		spl = make_sampler()
		spl.reset_to_random_prog()
		assert_program_valid(spl.current)

	def test_no_dangling_history_after_full_cycle(self):
		"""After accept+remove_history, old_op and new_op must both be None."""
		spl = make_sampler()
		spl.sample_next()
		if spl.current.old_op is not None:
			spl.current.remove_history()
		assert spl.current.old_op is None
		assert spl.current.new_op is None
		
	
# Test new mutations (use flags to isolate)	
class TestRewireMakeOp:
	def test_rewire_make_op_preserves_validity(self):
		flags = MutationFlags(
			replace_entire_op=False,
			replace_operand=False,
			rewire_make_op=True,
		)
		spl = make_sampler(flags=flags)
		for _ in range(20):
			spl.sample_next()
			assert_program_valid(spl.current)
			if spl.current.old_op is not None:
				spl.current.revert_operation()
			assert_program_valid(spl.current)

	def test_rewire_make_op_changes_operand(self):
		flags = MutationFlags(
			replace_entire_op=False,
			replace_operand=False,
			rewire_make_op=True,
		)
		spl = make_sampler(flags=flags, seed=0)
		ops = spl.current.ops
		make_op = ops[-2]
		original_operand = make_op.operands[0]

		changed = False
		for _ in range(20):
			spl.sample_next()
			if spl.current.ops[-2].operands[0] != original_operand:
				changed = True
			if spl.current.old_op is not None:
				spl.current.revert_operation()

		assert changed, "rewire_make_op never changed the operand"

class TestPerturbConstant:
	def test_perturb_constant_preserves_validity(self):
		flags = MutationFlags(
			replace_entire_op=False,
			replace_operand=False,
			perturb_constant=True,
		)
		spl = make_sampler(flags=flags)
		for _ in range(20):
			spl.sample_next()
			assert_program_valid(spl.current)
			if spl.current.old_op is not None:
				spl.current.revert_operation()
			assert_program_valid(spl.current)

	def test_perturb_constant_revert_restores_value(self):
		flags = MutationFlags(
			replace_entire_op=False,
			replace_operand=False,
			perturb_constant=True,
		)
		spl = make_sampler(flags=flags, seed=1)

		def get_const_values():
			return [
				op.value.value.data
				for op in spl.current.ops
				if isinstance(op, Constant)
			]

		for _ in range(20):
			before = get_const_values()
			spl.sample_next()
			if spl.current.old_op is not None:
				spl.current.revert_operation()
			after = get_const_values()
			assert before == after, \
				f"revert_operation did not restore constant values: {before} -> {after}"

	def test_perturb_constant_no_op_when_no_constants(self):
		flags = MutationFlags(
			replace_entire_op=False,
			replace_operand=False,
			perturb_constant=True,
		)
		spl = make_sampler(flags=flags)
		# TODO: How do we enforce this?
		spl.sample_next()
		assert_program_valid(spl.current)
  
	def test_perturb_constant_kb_samples_valid_values(self):
		flags = MutationFlags(
			replace_entire_op=False,
			replace_operand=False,
			perturb_constant=True,
		)
		spl = make_sampler(flags=flags, seed=0,
						domain=AbstractDomain.KnownBits)
		for _ in range(30):
			spl.sample_next()
			for op in spl.current.ops:
				if isinstance(op, Constant):
					assert op.value.value.data in [0, 1], \
						f"KnownBits constant outside {{0,1}}: {op.value.value.data}"
			if spl.current.old_op is not None:
				spl.current.remove_history()

	def test_perturb_constant_cr_samples_valid_values(self):
		flags = MutationFlags(
			replace_entire_op=False,
			replace_operand=False,
			perturb_constant=True,
		)
		cr_candidates = [0, 1, 2, 4, 8, 16, 32, 64, 128]
		spl = make_sampler(flags=flags, seed=0,
						domain=AbstractDomain.UConstRange)
		for _ in range(30):
			spl.sample_next()
			for op in spl.current.ops:
				if isinstance(op, Constant):
					assert op.value.value.data in cr_candidates, \
						f"CR constant outside candidates: {op.value.value.data}"
			if spl.current.old_op is not None:
				spl.current.remove_history()

	def test_perturb_constant_always_changes_value(self):
		flags = MutationFlags(
			replace_entire_op=False,
			replace_operand=False,
			perturb_constant=True,
		)
		spl = make_sampler(flags=flags, seed=0)

		def get_const_values():
			return [
				op.value.value.data
				for op in spl.current.ops
				if isinstance(op, Constant)
			]

		for _ in range(30):
			before = get_const_values()
			spl.sample_next()
			after = get_const_values()
			if spl.current.old_op is not None:
				assert before != after, \
					"perturb_constant should always change the value"
				spl.current.revert_operation()

	def test_perturb_constant_fp_samples_valid_values(self):
		flags = MutationFlags(
			replace_entire_op=False,
			replace_operand=False,
			perturb_constant=True,
		)
		fp_candidates = [
			0, 0x80000000, 0x3F800000, 0xBF800000,
			0x7F800000, 0xFF800000, 0x7F7FFFFF,
			0xFF7FFFFF, 0x00800000, 0x7FC00000,
		]
		spl = make_sampler(flags=flags, seed=0,
						domain=AbstractDomain.FPRange)
		for _ in range(30):
			spl.sample_next()
			for op in spl.current.ops:
				if isinstance(op, Constant):
					assert op.value.value.data in fp_candidates, \
						f"FP constant outside candidates: {op.value.value.data}"
			if spl.current.old_op is not None:
				spl.current.remove_history()
	
class TestReplaceOpWindow:
	def test_window_preserves_validity(self):
		flags = MutationFlags(
			replace_entire_op=False,
			replace_operand=False,
			replace_op_window=True,
		)
		spl = make_sampler(flags=flags)
		for _ in range(20):
			spl.sample_next()
			assert_program_valid(spl.current)
			if spl.current.old_ops is not None:
				spl.current.revert_window()
			assert_program_valid(spl.current)

	def test_window_revert_restores_program(self):
		flags = MutationFlags(
			replace_entire_op=False,
			replace_operand=False,
			replace_op_window=True,
		)
		spl = make_sampler(flags=flags, seed=0)

		for _ in range(20):
			ops_before = [type(op) for op in spl.current.ops]
			spl.sample_next()
			if spl.current.old_ops is not None:
				spl.current.revert_window()
			ops_after = [type(op) for op in spl.current.ops]
			assert ops_before == ops_after, \
				"revert_window did not restore original op sequence"

	def test_window_commit_clears_history(self):
		flags = MutationFlags(
			replace_entire_op=False,
			replace_operand=False,
			replace_op_window=True,
		)
		spl = make_sampler(flags=flags)
		spl.sample_next()
		if spl.current.old_ops is not None:
			spl.current.remove_history_window()
		assert spl.current.old_ops is None
		assert spl.current.new_ops is None
		assert_program_valid(spl.current)

	def test_existing_mutations_unaffected(self):
		"""Existing single-op mutations should still work after window changes."""
		spl = make_sampler()  # default flags
		for _ in range(30):
			spl.sample_next()
			assert_program_valid(spl.current)
			if spl.current.old_op is not None:
				spl.current.remove_history()
			elif spl.current.old_ops is not None:
				spl.current.remove_history_window()

	def test_window_changes_multiple_ops(self):
		flags = MutationFlags(
			replace_entire_op=False,
			replace_operand=False,
			replace_op_window=True,
		)
		spl = make_sampler(flags=flags, seed=1)

		multi_op_count = 0
		for _ in range(20):
			spl.sample_next()
			if spl.current.old_ops is not None:
				assert len(spl.current.old_ops) >= 2, \
					"window mutation should always replace at least 2 ops"
				assert len(spl.current.new_ops) >= 2, \
					"window mutation should always insert at least 2 ops"
				multi_op_count += 1
				spl.current.revert_window()

		assert multi_op_count > 0, \
			"replace_op_window was never triggered in 20 steps"