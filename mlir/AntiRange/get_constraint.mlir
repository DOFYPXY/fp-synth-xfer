"func.func"() ({
^bb0(%arg0: !transfer.abs_value<[!transfer.integer, !transfer.integer]>):
  %arg00 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
  %arg01 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer

  %const0 = "transfer.constant"(%arg00){value=0:index} : (!transfer.integer) -> !transfer.integer
  %const1 = "transfer.constant"(%arg00){value=1:index} : (!transfer.integer) -> !transfer.integer
  %constMax = "transfer.sub"(%const0, %const1) : (!transfer.integer, !transfer.integer) -> !transfer.integer

  %lb_ne_0 = "transfer.cmp"(%arg00, %const0){predicate=1:i64}:(!transfer.integer, !transfer.integer) -> i1
  %ub_ne_max = "transfer.cmp"(%arg01, %constMax){predicate=1:i64}:(!transfer.integer, !transfer.integer) -> i1

  %result = "arith.andi"(%lb_ne_0, %ub_ne_max) : (i1, i1) -> i1
  "func.return"(%result) : (i1) -> ()
}) {function_type = (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> i1, sym_name = "getConstraint"} : () -> ()
