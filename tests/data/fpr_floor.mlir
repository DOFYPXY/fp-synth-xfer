"func.func"() ({
  ^0(%a : !fp.abs_value):
    %a_lo = "fp.get"(%a) {index=0:index}: (!fp.abs_value) -> !fp.float
    %a_hi = "fp.get"(%a) {index=1:index}: (!fp.abs_value) -> !fp.float
    %a_has_nan = "fp.get"(%a) {index=2:index}: (!fp.abs_value) -> i1
    %res_lo = "fp.floor"(%a_lo) : (!fp.float) -> !fp.float
    %res_hi = "fp.floor"(%a_hi) : (!fp.float) -> !fp.float
    %result = "fp.make"(%res_lo, %res_hi, %a_has_nan) : (!fp.float, !fp.float, i1) -> !fp.abs_value
    "func.return"(%result) : (!fp.abs_value) -> ()
  }) {"sym_name" = "fpr_floor", "function_type" = (!fp.abs_value) -> !fp.abs_value} : () -> ()
