"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %result = "transfer.sdiv"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) ->!transfer.integer
    "func.return"(%result) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer,!transfer.integer) -> !transfer.integer, sym_name = "concrete_op"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %const0 = "transfer.constant"(%arg1) {value=0:index}:(!transfer.integer)->!transfer.integer
    %arg1_neq_0 = "transfer.cmp"(%const0, %arg1) {predicate=1:i64}: (!transfer.integer, !transfer.integer) -> i1
    %arg0_eq_0 = "transfer.cmp"(%const0, %arg0) {predicate=0:i64}: (!transfer.integer, !transfer.integer) -> i1
    %arg0_plus_arg0 = "transfer.add"(%arg0, %arg0) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %arg0_plus_arg0_neq_0 = "transfer.cmp"(%arg0_plus_arg0, %const0) {predicate=1:i64}: (!transfer.integer, !transfer.integer) -> i1
    %arg0_neq_smin = "transfer.or"(%arg0_eq_0, %arg0_plus_arg0_neq_0) : (i1, i1) -> i1
    %minus1 = "transfer.get_all_ones"(%arg0) : (!transfer.integer) -> !transfer.integer
    %arg1_neq_minus1 = "transfer.cmp"(%minus1, %arg1) {predicate=1:i64}: (!transfer.integer, !transfer.integer) -> i1
    %not_ub2 = "transfer.or"(%arg0_neq_smin, %arg1_neq_minus1) : (i1, i1) -> i1
    %not_ub = "transfer.and"(%arg1_neq_0, %not_ub2) : (i1, i1) -> i1
    "func.return"(%not_ub) : (i1) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> i1, sym_name = "op_constraint"} : () -> ()
}) : () -> ()
