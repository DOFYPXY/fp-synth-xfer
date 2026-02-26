"func.func"() ({
  ^0(%a : !fp.abs_value, %b : !fp.abs_value):
    %a_lo = "fp.get"(%a) {index=0:index}: (!fp.abs_value) -> !fp.float
    %a_hi = "fp.get"(%a) {index=1:index}: (!fp.abs_value) -> !fp.float
    %a_has_nan = "fp.get"(%a) {index=2:index}: (!fp.abs_value) -> i1
    %b_lo = "fp.get"(%b) {index=0:index}: (!fp.abs_value) -> !fp.float
    %b_hi = "fp.get"(%b) {index=1:index}: (!fp.abs_value) -> !fp.float
    %b_has_nan = "fp.get"(%b) {index=2:index}: (!fp.abs_value) -> i1
    %ab_has_nan = "arith.ori"(%a_has_nan, %b_has_nan) : (i1, i1) -> i1
    %sum_lo = "fp.add"(%a_lo, %b_lo) : (!fp.float, !fp.float) -> !fp.float
    %sum_hi = "fp.add"(%a_hi, %b_hi) : (!fp.float, !fp.float) -> !fp.float
    %sum_lo_nan = "fp.is_nan"(%sum_lo) : (!fp.float) -> i1
    %sum_hi_nan = "fp.is_nan"(%sum_hi) : (!fp.float) -> i1
    %sum_has_nan = "arith.ori"(%sum_lo_nan, %sum_hi_nan) : (i1, i1) -> i1
    %res_has_nan = "arith.ori"(%ab_has_nan, %sum_has_nan) : (i1, i1) -> i1
    %pos_inf = "fp.pos_inf"(%a_lo) : (!fp.float) -> !fp.float
    %neg_inf = "fp.neg_inf"(%a_lo) : (!fp.float) -> !fp.float
    %res_lo = "transfer.select"(%sum_lo_nan, %pos_inf, %sum_lo) : (i1, !fp.float, !fp.float) -> !fp.float
    %res_hi = "transfer.select"(%sum_hi_nan, %neg_inf, %sum_hi) : (i1, !fp.float, !fp.float) -> !fp.float
    %result = "fp.make"(%res_lo, %res_hi, %res_has_nan) : (!fp.float, !fp.float, i1) -> !fp.abs_value
  }) {"sym_name" = "fpr_add", "function_type" = (!fp.abs_value, !fp.abs_value) -> !fp.abs_value} : () -> ()
