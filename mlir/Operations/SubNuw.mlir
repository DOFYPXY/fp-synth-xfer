"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %result = "transfer.sub"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) ->!transfer.integer
    "func.return"(%result) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer,!transfer.integer) -> !transfer.integer, sym_name = "concrete_op"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %check = "transfer.cmp"(%arg0, %arg1) {predicate=9:i64}: (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%check) : (i1) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> i1, sym_name = "op_constraint"} : () -> ()
}): () -> ()
