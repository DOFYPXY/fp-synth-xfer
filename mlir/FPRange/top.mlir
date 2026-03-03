"func.func"() ({
^bb0(%a: !fp.abs_value):
  %a_lo = "fp.get"(%a) {index=0:index}: (!fp.abs_value) -> !fp.float
  %lo = "fp.neg_inf"(%a_lo) : (!fp.float) -> !fp.float
  %hi = "fp.pos_inf"(%a_lo) : (!fp.float) -> !fp.float
  %has_nan = "arith.constant"() {value = 1 : i1} : () -> i1
  %result = "fp.make"(%lo, %hi, %has_nan) : (!fp.float, !fp.float, i1) -> !fp.abs_value
  "func.return"(%result) : (!fp.abs_value) -> ()
}) {function_type = (!fp.abs_value) -> !fp.abs_value, sym_name = "getTop"} : () -> ()
