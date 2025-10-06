"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %const0 = "transfer.constant"(%arg0){value=0:index} : (!transfer.integer) -> !transfer.integer
    %subRes = "transfer.sub"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %overflow = "transfer.cmp"(%arg0, %arg1){predicate=6:i64}:(!transfer.integer,!transfer.integer)->i1
    %result = "transfer.select"(%overflow, %const0, %subRes) : (i1, !transfer.integer, !transfer.integer) ->!transfer.integer
    "func.return"(%result) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer,!transfer.integer) -> !transfer.integer, sym_name = "concrete_op"} : () -> ()
}): () -> ()
