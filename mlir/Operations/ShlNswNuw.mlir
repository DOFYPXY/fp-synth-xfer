"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %result = "transfer.shl"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) ->!transfer.integer
    "func.return"(%result) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer,!transfer.integer) -> !transfer.integer, sym_name = "concrete_op"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %bitwidth = "transfer.get_bit_width"(%arg0): (!transfer.integer) -> !transfer.integer
    %check = "transfer.cmp"(%arg1, %bitwidth) {predicate=9:i64}: (!transfer.integer, !transfer.integer) -> i1

    %cl0 = "transfer.countl_zero"(%arg0) : (!transfer.integer) -> !transfer.integer
    %nuw = "transfer.cmp"(%cl0, %arg1) {predicate=9:i64}: (!transfer.integer, !transfer.integer) -> i1

    %cl1 = "transfer.countl_one"(%arg0) : (!transfer.integer) -> !transfer.integer
    %is_non_neg = "transfer.cmp"(%arg0, %const0) {predicate=5:i64}: (!transfer.integer, !transfer.integer) -> i1
    %shamt_lt_cl0 = "transfer.cmp"(%arg1, %cl0) {predicate=6:i64}: (!transfer.integer, !transfer.integer) -> i1
    %shamt_lt_cl1 = "transfer.cmp"(%arg1, %cl1) {predicate=6:i64}: (!transfer.integer, !transfer.integer) -> i1
    %nsw = "transfer.select"(%is_non_neg, %shamt_lt_cl0, %shamt_lt_cl1): (i1, i1, i1) -> i1

    %nuw_and_nsw = "transfer.and"(%nuw, %nsw) : (i1, i1) -> i1
    %res = "transfer.and"(%check, %nuw_and_nsw) : (i1, i1) -> i1
    "func.return"(%res) : (i1) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> i1, sym_name = "op_constraint"} : () -> ()
}) : () -> ()
