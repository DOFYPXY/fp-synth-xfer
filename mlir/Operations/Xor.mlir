"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %result = "transfer.xor"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) ->!transfer.integer
    "func.return"(%result) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer,!transfer.integer) -> !transfer.integer, sym_name = "concrete_op"} : () -> ()
}): () -> ()
