"func.func"() ({
^bb0(%a: !fp.abs_value):
  %lo = "fp.get"(%a) {index=0:index}: (!fp.abs_value) -> !fp.float
  %hi = "fp.get"(%a) {index=1:index}: (!fp.abs_value) -> !fp.float
  %has_nan = "fp.get"(%a) {index=2:index}: (!fp.abs_value) -> i1
  %result = "fp.cmp"(%lo, %hi){predicate = "ole"} : (!fp.float, !fp.float) -> i1
  "func.return"(%result) : (i1) -> ()
}) {function_type = (!fp.abs_value) -> i1, sym_name = "getConstraint"} : () -> ()
