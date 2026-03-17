"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: !fp.float):
    %result = "fp.floor"(%arg0) : (!fp.float) ->!fp.float
    "func.return"(%result) : (!fp.float) -> ()
  }) {function_type = (!fp.float) -> !fp.float, sym_name = "concrete_op"} : () -> ()
}): () -> ()
