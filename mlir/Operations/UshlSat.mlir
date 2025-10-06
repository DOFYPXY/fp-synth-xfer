"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %constMax = "transfer.get_all_ones"(%arg0) : (!transfer.integer) -> !transfer.integer
    %shlRes = "transfer.shl"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %overflow = "transfer.ushl_overflow"(%arg0, %arg1) {predicate=9:i64}: (!transfer.integer, !transfer.integer) -> i1
    %result = "transfer.select"(%overflow, %constMax, %shlRes) : (i1, !transfer.integer, !transfer.integer) ->!transfer.integer
    "func.return"(%result) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer,!transfer.integer) -> !transfer.integer, sym_name = "concrete_op"} : () -> ()
}): () -> ()
