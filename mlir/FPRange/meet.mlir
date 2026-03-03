"func.func"() ({
^bb0(%a: !fp.abs_value, %b: !fp.abs_value):
  %a_lo = "fp.get"(%a) {index=0:index}: (!fp.abs_value) -> !fp.float
  %a_hi = "fp.get"(%a) {index=1:index}: (!fp.abs_value) -> !fp.float
  %a_has_nan = "fp.get"(%a) {index=2:index}: (!fp.abs_value) -> i1
  %b_lo = "fp.get"(%b) {index=0:index}: (!fp.abs_value) -> !fp.float
  %b_hi = "fp.get"(%b) {index=1:index}: (!fp.abs_value) -> !fp.float
  %b_has_nan = "fp.get"(%b) {index=2:index}: (!fp.abs_value) -> i1
  %res_lo = "fp.maximumf"(%a_lo, %b_lo) : (!fp.float, !fp.float) -> !fp.float
  %res_hi = "fp.minimumf"(%a_hi, %b_hi) : (!fp.float, !fp.float) -> !fp.float
  %res_has_nan = "arith.ori"(%a_has_nan, %b_has_nan) : (i1, i1) -> i1
  %result = "fp.make"(%res_lo, %res_hi, %res_has_nan) : (!fp.float, !fp.float, i1) -> !fp.abs_value
  "func.return"(%result) : (!fp.abs_value) -> ()
}) {function_type = (!fp.abs_value, !fp.abs_value) -> !fp.abs_value, sym_name = "meet"} : () -> ()
