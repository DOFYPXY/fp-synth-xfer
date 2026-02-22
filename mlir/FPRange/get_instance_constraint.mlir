"func.func"() ({
^bb0(%a: !fp.abs_value, %c: !fp.float):
  %lo = "fp.get"(%a) {index=0:index}: (!fp.abs_value) -> !fp.float
  %hi = "fp.get"(%a) {index=1:index}: (!fp.abs_value) -> !fp.float
  %has_nan = "fp.get"(%a) {index=2:index}: (!fp.abs_value) -> i1
  %c_is_nan = "fp.is_nan"(%c) : (!fp.float) -> i1
  %c_is_nan_and_has_nan = "arith.andi"(%c_is_nan, %has_nan) : (i1, i1) -> i1
  %c_ge_lo = "fp.cmp"(%c, %lo) {predicate = "oge"} : (!fp.float, !fp.float) -> i1
  %c_le_hi = "fp.cmp"(%c, %hi) {predicate = "ole"} : (!fp.float, !fp.float) -> i1
  %c_in_range = "arith.andi"(%c_ge_lo, %c_le_hi) : (i1, i1) -> i1
  %result = "arith.ori"(%c_is_nan_and_has_nan, %c_in_range) : (i1, i1) -> i1
  "func.return"(%result) : (i1) -> ()
}) {function_type = (!fp.abs_value, !fp.float) -> i1, sym_name = "getInstanceConstraint"} : () -> ()
