"func.func"() ({
^bb0(%a: !fp.abs_value):
  %lo = "fp.get"(%a) {index=0:index}: (!fp.abs_value) -> !fp.float
  %hi = "fp.get"(%a) {index=1:index}: (!fp.abs_value) -> !fp.float
  %has_nan = "fp.get"(%a) {index=2:index}: (!fp.abs_value) -> i1
  %lo_is_nan = "fp.is_nan"(%lo) : (!fp.float) -> i1
  %hi_is_nan = "fp.is_nan"(%hi) : (!fp.float) -> i1
  %result = "arith.ori"(%has_nan, %lo_is_nan) : (i1, i1) -> i1
  "func.return"(%result) : (i1) -> ()
}) {function_type = (!fp.abs_value) -> i1, sym_name = "getConstraint"} : () -> ()
