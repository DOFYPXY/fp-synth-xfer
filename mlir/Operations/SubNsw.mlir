"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %result = "transfer.sub"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) ->!transfer.integer
    "func.return"(%result) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer,!transfer.integer) -> !transfer.integer, sym_name = "concrete_op"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %res = "transfer.sub"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %xor0 = "transfer.xor"(%arg0, %res) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %xor1 = "transfer.xor"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %andres = "transfer.and"(%xor0, %xor1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %zero = "transfer.constant"(%arg0){value=0:index} : (!transfer.integer) -> !transfer.integer
    %nsw = "transfer.cmp"(%andres, %zero) {predicate=5:i64}: (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%nsw) : (i1) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> i1, sym_name = "op_constraint"} : () -> ()
}): () -> ()
