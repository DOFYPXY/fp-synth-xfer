"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: !fp.float):
    %one = "fp.constant"(%arg0) {value = 1.0} : (!fp.float) -> !fp.float
    %result = "fp.add"(%arg0, %one) : (!fp.float, !fp.float) ->!fp.float
    "func.return"(%result) : (!fp.float) -> ()
  }) {function_type = (!fp.float) -> !fp.float, sym_name = "concrete_op"} : () -> ()
}): () -> ()
