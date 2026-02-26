"""FP-specific verification pipeline.

Like the BV verifier but without the poison model — FP concrete values
are raw FloatingPointType, not Pair(value, poison).
"""

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Operation, SSAValue

from xdsl_smt.dialects.smt_bitvector_dialect import ConstantOp
from xdsl_smt.dialects.smt_dialect import CheckSatOp, ConstantBoolOp, DefineFunOp
from xdsl_smt.passes.transfer_inline import FunctionCallInline
from xdsl_smt.passes.transfer_unroll_loop import UnrollTransferLoop
from xdsl_smt.utils.transfer_function_util import (
    SMTTransferFunction,
    TransferFunction,
    call_function_and_assert_result_with_effect,
    call_function_with_effect,
    get_argument_instances_with_effect,
)
from z3 import ModelRef

from synth_xfer.dialects.fp import FPAbsValueType

from .verifier import _create_smt_function, _lower_to_smt_module, _verify_pattern


def fp_forward_soundness_check(
    transfer_function: SMTTransferFunction,
    domain_constraint: DefineFunOp,
    instance_constraint: DefineFunOp,
    int_attr: dict[int, int],
) -> list[Operation]:
    """Check soundness of a forward FP transfer function.

    Differences from the BV version:
    - Concrete args/result have no poison pair wrapping.
    - Constraint functions are pre-lowered DefineFunOp (no bitwidth).
    """
    assert transfer_function.is_forward
    abstract_func = transfer_function.transfer_function
    concrete_func = transfer_function.concrete_function
    abs_op_constraint = transfer_function.abstract_constraint
    op_constraint = transfer_function.op_constraint
    is_abstract_arg = transfer_function.is_abstract_arg

    assert abstract_func is not None
    assert concrete_func is not None

    abs_arg_ops = get_argument_instances_with_effect(abstract_func, int_attr)
    abs_args: list[SSAValue] = [arg.res for arg in abs_arg_ops]
    crt_arg_ops = get_argument_instances_with_effect(concrete_func, int_attr)
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


def verify_fp_transfer_function(
    transfer_function: FuncOp,
    concrete_func: FuncOp,
    helper_funcs: list[FuncOp],
    width: int,
    timeout: int,
) -> tuple[bool | None, ModelRef | None]:
    """Full FP verification pipeline."""
    ctx = Context()

    INSTANCE_CONSTRAINT = "getInstanceConstraint"
    DOMAIN_CONSTRAINT = "getConstraint"

    # Build module — no poison wrapping for FP
    func_name_to_func: dict[str, FuncOp] = {}
    module_op = ModuleOp([])
    functions: list[FuncOp] = [transfer_function.clone(), concrete_func.clone()]
    module_op.body.block.add_ops(functions + [func.clone() for func in helper_funcs])

    domain_constraint_func: FuncOp | None = None
    instance_constraint_func: FuncOp | None = None
    transfer_function_obj: TransferFunction | None = None
    transfer_function_name = transfer_function.sym_name.data

    for func in module_op.ops:
        assert isinstance(func, FuncOp)
        func_name = func.sym_name.data
        if func_name in func_name_to_func:
            raise ValueError("Found function with the same name in the input")
        func_name_to_func[func_name] = func

        # Validate function types
        assert len(func.function_type.inputs) == len(func.args)
        for func_type_arg, arg in zip(func.function_type.inputs, func.args):
            assert func_type_arg == arg.type
        return_op = func.body.block.last_op
        assert return_op is not None and isinstance(return_op, ReturnOp)
        assert return_op.operands[0].type == func.function_type.outputs.data[0]

        if func_name == transfer_function_name:
            assert transfer_function_obj is None
            transfer_function_obj = TransferFunction(func)
        if func_name == DOMAIN_CONSTRAINT:
            domain_constraint_func = func
        elif func_name == INSTANCE_CONSTRAINT:
            instance_constraint_func = func

    assert domain_constraint_func is not None
    assert instance_constraint_func is not None
    assert transfer_function_obj is not None

    # Fix is_abstract_arg to recognise FPAbsValueType
    transfer_function_obj.is_abstract_arg = [
        isinstance(arg.type, FPAbsValueType)
        for arg in transfer_function_obj.transfer_function.args
    ]

    func_name_to_func[transfer_function.sym_name.data] = transfer_function

    # Inline, unroll, lower to SMT
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

    func_name = transfer_function.sym_name.data
    concrete_func_name = concrete_func.sym_name.data

    smt_transfer_function_obj = SMTTransferFunction(
        transfer_function_obj,
        func_name,
        concrete_func_name,
        func_name_to_smt_func.get("abs_op_constraint"),
        func_name_to_smt_func.get("op_constraint"),
        None,
        None,
        None,
        None,
        func_name_to_smt_func[func_name],
        func_name_to_smt_func[concrete_func_name],
    )

    # Pre-generate SMT constraint functions
    smt_domain_constraint = _create_smt_function(domain_constraint_func, width, ctx)
    smt_instance_constraint = _create_smt_function(
        instance_constraint_func, width, ctx
    )

    # Build and verify the soundness query
    assert smt_transfer_function_obj.is_forward
    added_ops = fp_forward_soundness_check(
        smt_transfer_function_obj,
        smt_domain_constraint,
        smt_instance_constraint,
        {},
    )

    query_module = ModuleOp([])
    query_module.body.block.add_ops(added_ops)
    FunctionCallInline(True, {}).apply(ctx, query_module)

    return _verify_pattern(ctx, query_module, timeout)
