from io import StringIO

from xdsl.context import Context
from xdsl.dialects.builtin import FunctionType, IntegerType, ModuleOp
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.smt import BoolType, ConstantBoolOp
from xdsl.ir import Operation, SSAValue
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl_smt.dialects.smt_dialect import CheckSatOp, DefineFunOp
from xdsl_smt.dialects.smt_bitvector_dialect import ConstantOp
from xdsl_smt.dialects.transfer import (
    AbstractValueType,
    GetOp,
    MakeOp,
    TransIntegerType,
    TupleType,
)
from xdsl_smt.passes.dead_code_elimination import DeadCodeElimination
from xdsl_smt.passes.lower_effects import LowerEffectPass
from xdsl_smt.passes.lower_pairs import LowerPairs
from xdsl_smt.passes.lower_to_smt import func_to_smt_patterns
from xdsl_smt.passes.lower_to_smt.lower_to_smt import LowerToSMTPass
from xdsl_smt.passes.lower_to_smt.smt_lowerer import SMTLowerer
from xdsl_smt.passes.merge_func_results import MergeFuncResultsPass
from xdsl_smt.passes.transfer_inline import FunctionCallInline
from xdsl_smt.passes.transfer_unroll_loop import UnrollTransferLoop
from xdsl_smt.semantics.arith_semantics import arith_semantics
from xdsl_smt.semantics.builtin_semantics import IntegerTypeSemantics
from xdsl_smt.semantics.comb_semantics import comb_semantics
from xdsl_smt.semantics.transfer_semantics import (
    AbstractValueTypeSemantics,
    TransferIntegerTypeSemantics,
    transfer_semantics,
)
from xdsl_smt.traits.smt_printer import print_to_smtlib
from xdsl_smt.utils.transfer_function_check_util import (
    backward_soundness_check,
    forward_soundness_check,
)
from xdsl_smt.utils.transfer_function_util import (
    FunctionCollection,
    SMTTransferFunction,
    TransferFunction,
    call_function_and_assert_result_with_effect,
    call_function_with_effect,
    get_argument_instances_with_effect,
)
from z3 import ModelRef, Solver, parse_smt2_string, sat, unknown

from .fp_semantics import FloatingPointAbsTypeSemantics, FloatingPointTypeSemantics, fp_semantics
from synth_xfer.dialects.fp import FloatType, FPAbsValueType

try:
    from cvc5 import InputParser, InputLanguage
    from cvc5 import Solver as Cvc5Solver
    from cvc5 import SymbolManager
except ImportError:
    pass

# TODO do we still need this
_TMP_MODULE: list[ModuleOp] = []


def _verify_pattern(
    ctx: Context,
    op: ModuleOp,
    timeout: int,
    solver: str = "z3",
    solver_options: dict[str, str] | None = None,
) -> tuple[bool | None, ModelRef | None]:
    cloned_op = op.clone()
    LowerPairs().apply(ctx, cloned_op)
    CanonicalizePass().apply(ctx, cloned_op)
    DeadCodeElimination().apply(ctx, cloned_op)

    stream = StringIO()
    print_to_smtlib(cloned_op, stream)

    smt2_string = stream.getvalue()
    opts = solver_options or {}

    if solver == "cvc5":
        return _check_sat_cvc5(smt2_string, timeout, opts)
    return _check_sat_z3(smt2_string, timeout)


def _check_sat_z3(
    smt2_string: str, timeout: int
) -> tuple[bool | None, ModelRef | None]:
    s = Solver()
    s.set(timeout=timeout * 1000)
    s.add(parse_smt2_string(smt2_string))
    r = s.check()

    if r == unknown:
        return None, None
    elif r == sat:
        return False, s.model()
    else:
        return True, None


def _check_sat_cvc5(
    smt2_string: str, timeout: int, options: dict[str, str]
) -> tuple[bool | None, object]:
    # Strip (check-sat) — we call checkSat() ourselves
    smt2_string = smt2_string.replace("(check-sat)", "")

    solver = Cvc5Solver()
    for key, value in options.items():
        solver.setOption(key, value)
    solver.setOption("produce-models", "true")
    solver.setOption("tlimit-per", str(timeout * 1000))
    solver.setLogic("ALL")

    sm = SymbolManager(solver)
    parser = InputParser(solver, sm)
    parser.setStringInput(InputLanguage.SMT_LIB_2_6, smt2_string, "input")

    while True:
        cmd = parser.nextCommand()
        if cmd.isNull():
            break
        cmd.invoke(solver, sm)

    result = solver.checkSat()

    if result.isUnsat():
        return True, None
    elif result.isSat():
        terms = sm.getDeclaredTerms()
        model_lines = [f"  {t} = {solver.getValue(t)}" for t in terms]
        return False, "\n".join(model_lines)
    else:
        return None, None


def _lower_to_smt_module(module: ModuleOp, width: int, ctx: Context):
    SMTLowerer.rewrite_patterns = {**func_to_smt_patterns}
    SMTLowerer.type_lowerers = {
        IntegerType: IntegerTypeSemantics(),
        AbstractValueType: AbstractValueTypeSemantics(),
        TransIntegerType: TransferIntegerTypeSemantics(width),
        TupleType: AbstractValueTypeSemantics(),
        FloatType: FloatingPointTypeSemantics(),
        FPAbsValueType: FloatingPointAbsTypeSemantics(),
    }
    SMTLowerer.op_semantics = {
        **arith_semantics,
        **transfer_semantics,
        **comb_semantics,
    }

    LowerToSMTPass().apply(ctx, module)
    MergeFuncResultsPass().apply(ctx, module)
    LowerEffectPass().apply(ctx, module)


def _add_poison_to_conc_fn(concrete_func: FuncOp) -> FuncOp:
    """
    Input: a concrete function with shape (trans.integer, trans.integer) -> trans.integer
    Output: a new function with shape (tuple<trans.integer, bool>
    """
    result_func = concrete_func.clone()
    block = result_func.body.block
    # Add poison to every args
    new_arg_type = TupleType([TransIntegerType(), BoolType()])
    while isinstance(result_func.args[0].type, TransIntegerType):
        new_arg = block.insert_arg(new_arg_type, len(result_func.args))
        new_get_op = GetOp(new_arg, 0)
        assert block.first_op is not None
        block.insert_op_before(new_get_op, block.first_op)
        result_func.args[0].replace_by(new_get_op.result)
        block.erase_arg(result_func.args[0])
    last_op = block.last_op
    assert last_op is not None
    poison_val = ConstantBoolOp(False)
    poison_val.is_ancestor
    new_return_val = MakeOp([last_op.operands[0], poison_val.result])
    block.insert_ops_before([poison_val, new_return_val], last_op)
    last_op.operands[0] = new_return_val.result
    new_args_type = [arg.type for arg in result_func.args]
    new_return_type = new_return_val.result.type
    result_func.function_type = FunctionType.from_lists(new_args_type, [new_return_type])
    return result_func


def _create_smt_function(func: FuncOp, width: int, ctx: Context) -> DefineFunOp:
    """
    Input: a function with type FuncOp
    Return: the function lowered to SMT dialect with specified width

    We might reuse some function with specific width so we save it to global TMP_MODULE
    Class FunctionCollection is the only caller of this function and maintains all generated SMT functions
    """

    global _TMP_MODULE
    _TMP_MODULE.append(ModuleOp([func.clone()]))
    _lower_to_smt_module(_TMP_MODULE[-1], width, ctx)
    resultFunc = _TMP_MODULE[-1].ops.first
    assert isinstance(resultFunc, DefineFunOp)
    return resultFunc


def _soundness_check(
    smt_transfer_function: SMTTransferFunction,
    domain_constraint: FunctionCollection,
    instance_constraint: FunctionCollection,
    int_attr: dict[int, int],
    ctx: Context,
    timeout: int,
    solver: str = "z3",
) -> tuple[bool | None, ModelRef | None]:
    query_module = ModuleOp([])
    if smt_transfer_function.is_forward:
        added_ops: list[Operation] = forward_soundness_check(
            smt_transfer_function,
            domain_constraint,
            instance_constraint,
            int_attr,
        )
    else:
        added_ops: list[Operation] = backward_soundness_check(
            smt_transfer_function,
            domain_constraint,
            instance_constraint,
            int_attr,
        )
    query_module.body.block.add_ops(added_ops)
    FunctionCallInline(True, {}).apply(ctx, query_module)

    return _verify_pattern(ctx, query_module, timeout, solver=solver)


def _verify_smt_transfer_function(
    smt_transfer_function: SMTTransferFunction,
    domain_constraint: FunctionCollection,
    instance_constraint: FunctionCollection,
    ctx: Context,
    timeout: int,
    solver: str = "z3",
) -> tuple[bool | None, ModelRef | None]:
    assert smt_transfer_function.concrete_function is not None
    assert smt_transfer_function.transfer_function is not None

    int_attr: dict[int, int] = {}
    soundness_result = _soundness_check(
        smt_transfer_function,
        domain_constraint,
        instance_constraint,
        int_attr,
        ctx,
        timeout,
        solver=solver,
    )

    return soundness_result


def _build_init_module(
    transfer_function: FuncOp,
    concrete_func: FuncOp,
    helper_funcs: list[FuncOp],
    ctx: Context,
):
    INSTANCE_CONSTRAINT = "getInstanceConstraint"
    DOMAIN_CONSTRAINT = "getConstraint"

    func_name_to_func: dict[str, FuncOp] = {}
    module_op = ModuleOp([])
    functions: list[FuncOp] = [transfer_function.clone()]
    functions.append(_add_poison_to_conc_fn(concrete_func))
    module_op.body.block.add_ops(functions + [func.clone() for func in helper_funcs])
    domain_constraint: FunctionCollection | None = None
    instance_constraint: FunctionCollection | None = None
    transfer_function_obj: TransferFunction | None = None
    transfer_function_name = transfer_function.sym_name.data
    for func in module_op.ops:
        assert isinstance(func, FuncOp)
        func_name = func.sym_name.data
        if func_name in func_name_to_func:
            print(
                [func.sym_name.data for func in helper_funcs]
                + [transfer_function.sym_name.data]
            )
            raise ValueError("Found function with the same name in the input")

        func_name_to_func[func_name] = func

        # Check func validity
        assert len(func.function_type.inputs) == len(func.args)
        for func_type_arg, arg in zip(func.function_type.inputs, func.args):
            assert func_type_arg == arg.type
        return_op = func.body.block.last_op
        assert return_op is not None and isinstance(return_op, ReturnOp)
        assert return_op.operands[0].type == func.function_type.outputs.data[0]
        # End of check function type

        if func_name == transfer_function_name:
            assert transfer_function_obj is None
            transfer_function_obj = TransferFunction(func)
        if func_name == DOMAIN_CONSTRAINT:
            assert domain_constraint is None
            domain_constraint = FunctionCollection(func, _create_smt_function, ctx)
        elif func_name == INSTANCE_CONSTRAINT:
            assert instance_constraint is None
            instance_constraint = FunctionCollection(func, _create_smt_function, ctx)

    assert domain_constraint is not None
    assert instance_constraint is not None
    assert transfer_function_obj is not None

    func_name_to_func[transfer_function.sym_name.data] = transfer_function
    return (
        module_op,
        func_name_to_func,
        transfer_function_obj,
        domain_constraint,
        instance_constraint,
    )


def _build_fp_module(
    transfer_function: FuncOp,
    concrete_func: FuncOp,
    helper_funcs: list[FuncOp],
    width: int,
    ctx: Context,
) -> tuple[
    dict[str, DefineFunOp],
    TransferFunction,
    dict[str, FuncOp],
]:
    """Build, validate, inline, unroll, lower, and collect SMT functions for FP.

    Like _build_init_module but without poison wrapping on the concrete function,
    and returns the SMT-lowered function map and pre-lowered FuncOp map directly
    (FP constraints are not width-parameterised so no FunctionCollection needed).
    """
    func_name_to_func: dict[str, FuncOp] = {}
    module_op = ModuleOp([])
    functions: list[FuncOp] = [transfer_function.clone(), concrete_func.clone()]
    module_op.body.block.add_ops(functions + [func.clone() for func in helper_funcs])

    transfer_function_obj: TransferFunction | None = None
    transfer_function_name = transfer_function.sym_name.data

    for func in module_op.ops:
        assert isinstance(func, FuncOp)
        func_name = func.sym_name.data
        if func_name in func_name_to_func:
            raise ValueError("Found function with the same name in the input")
        func_name_to_func[func_name] = func

        # Check func validity
        assert len(func.function_type.inputs) == len(func.args)
        for func_type_arg, arg in zip(func.function_type.inputs, func.args):
            assert func_type_arg == arg.type
        return_op = func.body.block.last_op
        assert return_op is not None and isinstance(return_op, ReturnOp)
        assert return_op.operands[0].type == func.function_type.outputs.data[0]
        # End of check function type

        if func_name == transfer_function_name:
            assert transfer_function_obj is None
            transfer_function_obj = TransferFunction(func)

    assert transfer_function_obj is not None

    # Save pre-lowered funcs before inlining mutates them
    pre_lowered_funcs = {name: fn.clone() for name, fn in func_name_to_func.items()}

    func_name_to_func[transfer_function.sym_name.data] = transfer_function

    # Inline → unroll → lower to SMT
    FunctionCallInline(False, func_name_to_func).apply(ctx, module_op)
    smt_module = module_op.clone()
    UnrollTransferLoop(width).apply(ctx, smt_module)
    _lower_to_smt_module(smt_module, width, ctx)

    # Collect SMT-lowered functions
    func_name_to_smt_func: dict[str, DefineFunOp] = {}
    for op in smt_module.ops:
        if isinstance(op, DefineFunOp):
            assert op.fun_name is not None
            func_name_to_smt_func[op.fun_name.data] = op

    return func_name_to_smt_func, transfer_function_obj, pre_lowered_funcs


def _fp_forward_soundness_check(
    transfer_function: SMTTransferFunction,
    domain_constraint: DefineFunOp,
    instance_constraint: DefineFunOp,
) -> list[Operation]:
    """Build a soundness query for a forward FP transfer function.

    Unlike the BV version, concrete values have no poison wrapping and
    constraint functions are single pre-lowered DefineFunOp (no
    width-parameterised lookup).
    """
    assert transfer_function.is_forward
    abstract_func = transfer_function.transfer_function
    concrete_func = transfer_function.concrete_function
    abs_op_constraint = transfer_function.abstract_constraint
    op_constraint = transfer_function.op_constraint
    is_abstract_arg = transfer_function.is_abstract_arg

    assert abstract_func is not None
    assert concrete_func is not None

    abs_arg_ops = get_argument_instances_with_effect(abstract_func, {})
    abs_args: list[SSAValue] = [arg.res for arg in abs_arg_ops]
    crt_arg_ops = get_argument_instances_with_effect(concrete_func, {})
    crt_args: list[SSAValue] = [arg.res for arg in crt_arg_ops]

    assert len(abs_args) == len(crt_args)

    effect = ConstantBoolOp(False)
    constant_bv_0 = ConstantOp(0, 1)
    constant_bv_1 = ConstantOp(1, 1)

    abs_arg_include_crt_arg_constraints_ops: list[Operation] = []
    abs_domain_constraints_ops: list[Operation] = []
    for i, (abs_arg, crt_arg) in enumerate(zip(abs_args, crt_args)):
        if is_abstract_arg[i]:
            abs_arg_include_crt_arg_constraints_ops += (
                call_function_and_assert_result_with_effect(
                    instance_constraint,
                    [abs_arg, crt_arg],
                    constant_bv_1,
                    effect.result,
                )
            )
            abs_domain_constraints_ops += (
                call_function_and_assert_result_with_effect(
                    domain_constraint,
                    [abs_arg],
                    constant_bv_1,
                    effect.result,
                )
            )

    abs_arg_constraints_ops: list[Operation] = []
    if abs_op_constraint is not None:
        abs_arg_constraints_ops = call_function_and_assert_result_with_effect(
            abs_op_constraint, abs_args, constant_bv_1, effect.result
        )
    crt_args_constraints_ops: list[Operation] = []
    if op_constraint is not None:
        crt_args_constraints_ops = call_function_and_assert_result_with_effect(
            op_constraint, crt_args, constant_bv_1, effect.result
        )

    call_abs_func_op, call_abs_func_first_op = call_function_with_effect(
        abstract_func, abs_args, effect.result
    )
    call_crt_func_op, call_crt_func_first_op = call_function_with_effect(
        concrete_func, crt_args, effect.result
    )

    abs_result_not_include_crt_result_ops = (
        call_function_and_assert_result_with_effect(
            instance_constraint,
            [call_abs_func_first_op.res, call_crt_func_first_op.res],
            constant_bv_0,
            effect.result,
        )
    )

    return (
        [effect]
        + abs_arg_ops
        + crt_arg_ops
        + [constant_bv_0, constant_bv_1]
        + abs_domain_constraints_ops
        + abs_arg_include_crt_arg_constraints_ops
        + abs_arg_constraints_ops
        + crt_args_constraints_ops
        + [
            call_abs_func_op,
            call_abs_func_first_op,
            call_crt_func_op,
            call_crt_func_first_op,
        ]
        + abs_result_not_include_crt_result_ops
        + [CheckSatOp()]
    )


def verify_transfer_function(
    transfer_function: FuncOp,
    concrete_func: FuncOp,
    helper_funcs: list[FuncOp],
    width: int,
    timeout: int,
    solver: str = "z3",
) -> tuple[bool | None, ModelRef | None]:
    ctx = Context()

    (
        module_op,
        func_name_to_func,
        transfer_function_obj,
        domain_constraint,
        instance_constraint,
    ) = _build_init_module(transfer_function, concrete_func, helper_funcs, ctx)

    FunctionCallInline(False, func_name_to_func).apply(ctx, module_op)

    smt_module = module_op.clone()

    # expand for loops
    unrollTransferLoop = UnrollTransferLoop(width)
    assert isinstance(smt_module, ModuleOp)
    unrollTransferLoop.apply(ctx, smt_module)

    concrete_func_name: str = concrete_func.sym_name.data
    assert concrete_func_name is not None

    _lower_to_smt_module(smt_module, width, ctx)

    func_name_to_smt_func: dict[str, DefineFunOp] = {}
    for op in smt_module.ops:
        if isinstance(op, DefineFunOp):
            op_func_name = op.fun_name
            assert op_func_name is not None
            func_name = op_func_name.data
            func_name_to_smt_func[func_name] = op

    func_name = transfer_function.sym_name.data
    assert func_name is not None

    smt_concrete_func = func_name_to_smt_func.get(concrete_func_name, None)
    assert smt_concrete_func is not None
    smt_transfer_function = func_name_to_smt_func.get(func_name, None)
    abs_op_constraint = func_name_to_smt_func.get("abs_op_constraint", None)
    op_constraint = func_name_to_smt_func.get("op_constraint", None)

    smt_transfer_function_obj = SMTTransferFunction(
        transfer_function_obj,
        func_name,
        concrete_func_name,
        abs_op_constraint,
        op_constraint,
        None,
        None,
        None,
        None,
        smt_transfer_function,
        smt_concrete_func,
    )

    return _verify_smt_transfer_function(
        smt_transfer_function_obj,
        domain_constraint,
        instance_constraint,
        ctx,
        timeout,
        solver=solver,
    )


def verify_fp_transfer_function(
    transfer_function: FuncOp,
    concrete_func: FuncOp,
    helper_funcs: list[FuncOp],
    timeout: int,
    solver: str = "z3",
) -> tuple[bool | None, ModelRef | None]:
    """Verify an FP transfer function (no poison, single-width constraints)."""
    ctx = Context()
    _FP_DUMMY_WIDTH = 16

    func_name_to_smt_func, transfer_function_obj, pre_lowered = _build_fp_module(
        transfer_function, concrete_func, helper_funcs, _FP_DUMMY_WIDTH, ctx,
    )

    # FP uses FPAbsValueType instead of AbstractValueType
    transfer_function_obj.is_abstract_arg = [
        isinstance(arg.type, FPAbsValueType)
        for arg in transfer_function_obj.transfer_function.args
    ]

    func_name = transfer_function.sym_name.data
    concrete_func_name = concrete_func.sym_name.data

    smt_transfer_function_obj = SMTTransferFunction(
        transfer_function_obj,
        func_name,
        concrete_func_name,
        func_name_to_smt_func.get("abs_op_constraint"),
        func_name_to_smt_func.get("op_constraint"),
        None, None, None, None,
        func_name_to_smt_func[func_name],
        func_name_to_smt_func[concrete_func_name],
    )

    # Pre-generate constraint functions at a single dummy width
    smt_domain_constraint = _create_smt_function(
        pre_lowered["getConstraint"], _FP_DUMMY_WIDTH, ctx
    )
    smt_instance_constraint = _create_smt_function(
        pre_lowered["getInstanceConstraint"], _FP_DUMMY_WIDTH, ctx
    )

    assert smt_transfer_function_obj.is_forward
    added_ops = _fp_forward_soundness_check(
        smt_transfer_function_obj,
        smt_domain_constraint,
        smt_instance_constraint,
    )

    query_module = ModuleOp([])
    query_module.body.block.add_ops(added_ops)
    FunctionCallInline(True, {}).apply(ctx, query_module)
    return _verify_pattern(
        ctx, query_module, timeout,
        solver=solver,
        solver_options={"fp-exp": "true"},
    )
