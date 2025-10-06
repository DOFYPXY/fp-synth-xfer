"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %constMax = "transfer.get_all_ones"(%arg0) : (!transfer.integer) -> !transfer.integer
    %mulRes = "transfer.mul"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %overflow = "transfer.umul_overflow"(%arg0, %arg1):(!transfer.integer,!transfer.integer)->i1
    %result = "transfer.select"(%overflow, %constMax, %mulRes) : (i1, !transfer.integer, !transfer.integer) ->!transfer.integer
    "func.return"(%result) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer,!transfer.integer) -> !transfer.integer, sym_name = "concrete_op"} : () -> ()
}): () -> ()
