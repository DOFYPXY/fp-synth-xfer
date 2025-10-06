"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %andi = "transfer.and"(%arg0, %arg1) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    %xori = "transfer.xor"(%arg0, %arg1) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    %one = "transfer.constant"(%arg0){value=1:index} : (!transfer.integer) -> !transfer.integer
    %lshri = "transfer.lshr"(%xori, %one) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    %result = "transfer.add"(%andi, %lshri) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%result) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer,!transfer.integer) -> !transfer.integer, sym_name = "concrete_op"} : () -> ()
}): () -> ()
