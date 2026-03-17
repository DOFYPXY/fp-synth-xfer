#pragma once

#include <cstdint>
#include <ostream>
#include <random>
#include <string_view>
#include <vector>
#include <iostream>
#include "fp16_ops.hpp"

// FPRange: Abstract domain for FP16 floating point values
// Represents a range [lo, hi] of FP16 values, plus a flag for NaN
// Always operates on 16-bit floats, no template parameter needed
struct FPRange {
  std::uint16_t lo;      // FP16 bit pattern for lower bound
  std::uint16_t hi;      // FP16 bit pattern for upper bound
  bool has_nan;          // Whether NaN is in the abstract set

  // Constructors
  constexpr FPRange() : lo(0), hi(0), has_nan(false) {}
  constexpr FPRange(std::uint16_t l, std::uint16_t h, bool nan)
      : lo(l), hi(h), has_nan(nan) {}

  // Equality
  constexpr bool operator==(const FPRange &other) const noexcept {
    return lo == other.lo && hi == other.hi && has_nan == other.has_nan;
  }

  constexpr bool operator!=(const FPRange &other) const noexcept {
    return !(*this == other);
  }

  // Pretty printing: [lo, hi, T/F]
  friend std::ostream &operator<<(std::ostream &os, const FPRange &x) {
    auto print_val = [&](std::uint16_t v) {
      if (fp16::is_pos_inf(v))      os << "+inf";
      else if (fp16::is_neg_inf(v)) os << "-inf";
      else if (fp16::is_nan(v))     os << "nan";
      else                          os << fp16::to_float(v);
    };

    os << "[";
    print_val(x.lo);
    os << ", ";
    print_val(x.hi);
    os << (x.has_nan ? "; NaN" : "") << "]";

    if (x.isBottom())       os << "(bottom)";
    else if (x.isTop())     os << "(top)";
    else if (x.isOnlyNan())  os << "(only_nan)";

    return os;
  }

  // Lattice predicates
  bool constexpr isTop() const noexcept {
    // TODO: Top should be when range covers all possible FP16 values and has_nan=true
    // use fp16::POS_INF and NEG_INF to check if lo and hi cover the whole range
    return lo == fp16::NEG_INF && hi == fp16::POS_INF && has_nan;
  }

  bool constexpr isBottom() const noexcept {
    // Bottom is the empty set: no values in range and no NaN
    // Xuanyu: Treat all invalid ranges as bottom for now
    return !is_valid(lo, hi, has_nan);
  }

  bool isOnlyNan() const noexcept {
    // Check if this range represents only NaN
    return has_nan && !fp16::le(lo, hi);
  }

  bool isCanonicalOnlyNan() const noexcept {
    // Check if this range represents only NaN and is in canonical form (lo=POS_INF, hi=NEG_INF)
    return has_nan && lo == fp16::POS_INF && hi == fp16::NEG_INF;
  }

  // Lattice operations
  FPRange meet(const FPRange &rhs) const noexcept {
    // Intersection of two ranges
    // meet([lo1, hi1], [lo2, hi2], nan1, nan2) =
    //   [max(lo1, lo2), min(hi1, hi2)], (nan1 && nan2)
    // If resulting lo > hi, return bottom (empty set)
    if (isBottom() || rhs.isBottom()) {
      return bottom();
    }
    std::uint16_t new_lo = fp16::maximum(lo, rhs.lo);
    std::uint16_t new_hi = fp16::minimum(hi, rhs.hi);
    bool new_nan = has_nan && rhs.has_nan;
    return FPRange(new_lo, new_hi, new_nan);
  }

  FPRange join(const FPRange &rhs) const noexcept {
    // Union of two ranges
    // join([lo1, hi1], [lo2, hi2], nan1, nan2) =
    //   [min(lo1, lo2), max(hi1, hi2)], (nan1 || nan2)
    if (isBottom()) return rhs;
    if (rhs.isBottom()) return *this;
    std::uint16_t new_lo = fp16::minnum(lo, rhs.lo);
    std::uint16_t new_hi = fp16::maxnum(hi, rhs.hi);
    bool new_nan = has_nan || rhs.has_nan;
    return FPRange(new_lo, new_hi, new_nan);
  }

  // Concrete operations
  std::vector<std::uint16_t> toConcrete() const {
    // Enumerate all concrete FP16 values in this abstract range
    // Returns vector of uint16_t where each represents a FP16 bit pattern
    // Iterates through all 65536 possible FP16 bit patterns
    std::vector<std::uint16_t> res;
    // Iterate through all possible FP16 bit patterns
    for (std::uint32_t i = 0; i <= 0xFFFFu; ++i) {
      std::uint16_t bits = static_cast<std::uint16_t>(i);
      // Skip NaN patterns if has_nan is false
      // (compare() returns 0 for NaN, making ge/le both true, so NaNs would leak in)
      if (!has_nan && fp16::is_nan(bits)) {
        continue;
      }
      if (fp16::ge(bits, lo) && fp16::le(bits, hi)) {
        res.push_back(bits);
      }
    }
    if (has_nan) {
      res.push_back(fp16::NAN_PATTERN);
    }
    return res;
  }

  // Returns true if the given FP16 bit pattern is contained in this range.
  bool contains(std::uint16_t bits) const noexcept {
    if (isBottom()) return false;
    if (fp16::is_nan(bits)) return has_nan;
    return fp16::ge(bits, lo) && fp16::le(bits, hi);
  }

  // Returns representative concrete values for this range: lo, hi, and NaN if has_nan.
  std::vector<std::uint16_t> get_rep_conc() const {
    std::vector<std::uint16_t> res;
    if (contains(lo)) res.push_back(lo);
    if (hi != lo && contains(hi)) res.push_back(hi);
    if (contains(fp16::NAN_PATTERN)) res.push_back(fp16::NAN_PATTERN);
    return res;
  }

  std::uint16_t sample_concrete(std::mt19937 & rng) const {
    std::uniform_int_distribution<std::uint16_t> bits_dist(0, 0xFFFFu);
    // std::uniform_int_distribution<int> nan_check(0, 999);

    // // With small probability, sample NaN if it's in the range
    // if (has_nan && nan_check(rng) == 0) {
    //   return fp16::NAN_PATTERN;
    // }

    // Rejection sampling: repeatedly sample until we get a value in [lo, hi]
    for (int attempts = 0; attempts < 1000; ++attempts) {
      std::uint16_t candidate = bits_dist(rng);

      // Skip NaN bit patterns if they're not allowed
      if (!has_nan && fp16::is_nan(candidate)) {
        continue;
      }
      else if(has_nan && fp16::is_nan(candidate)) {
        return candidate;
      }

      // Check if candidate is within [lo, hi] using FP16 comparison
      if (fp16::ge(candidate, lo) && fp16::le(candidate, hi)) {
        return candidate;
      }
    }

    // Fallback: return lo (guaranteed to be in range)
    return lo;
  }

  // Distance metric
  std::uint64_t distance(const FPRange &rhs) const noexcept {
    // TODO: Define distance metric on the lattice
    // Example: count of bit patterns that differ between the two ranges
    // or symmetric difference of concrete sets
    uint64_t this_size = norm();
    uint64_t rhs_size = rhs.norm();
    // calculate size diff
    uint64_t size_diff = (this_size > rhs_size) ? (this_size - rhs_size) : (rhs_size - this_size);
    return size_diff;
  }

private:
  // Helper: count exponents in the range [lo, hi]
  std::uint16_t count_exponents() const noexcept {
    if (isBottom()) return 0;

    std::uint16_t lo_sign = fp16::extract_sign(lo);
    std::uint16_t hi_sign = fp16::extract_sign(hi);
    std::uint16_t lo_exp = fp16::extract_exp(lo);
    std::uint16_t hi_exp = fp16::extract_exp(hi);

    if (lo_sign == 0 && hi_sign == 0) {
      // Positive range: simple exponent range
      return hi_exp - lo_exp + 1;
    }
    else if (lo_sign == 1 && hi_sign == 1) {
      // Negative range: simple exponent range
      return lo_exp - hi_exp + 1;
    }
    else {
      // Crosses zero: sum positive and negative sections
      return lo_exp + hi_exp + 2;
    }
  }

public:
  std::uint64_t norm() const noexcept {
    if (isBottom()) return 0;
    if (isOnlyNan()) return 1;  // Only NaN contributes to size
    std::uint64_t exp_count = const_cast<FPRange*>(this)->count_exponents();
    return exp_count + (has_nan ? 1 : 0);
  }

  std::uint64_t size() const noexcept {
    // Estimate upper bound of concrete space size without enumerating all values
    // Returns an estimate of how many concrete FP16 values are in [lo, hi]

    if (isBottom()) return 0;

    std::uint64_t exp_count = const_cast<FPRange*>(this)->count_exponents();

    // Each exponent has ~1024 mantissa values (2^10 for 10-bit mantissa in FP16)
    // Plus some subnormal values, so use 1024 as conservative estimate
    const std::uint64_t mantissa_values_per_exp = 1024;
    std::uint64_t size_estimate = exp_count * mantissa_values_per_exp;

    // Add 1 if NaN is present
    if (has_nan) {
      size_estimate += 1;
    }

    return size_estimate;
  }

  // Validation helper. This should be equivalent to get_constraint.mlir
  static constexpr bool is_valid(std::uint16_t lo, std::uint16_t hi, bool has_nan) noexcept {
    if (fp16::is_nan(lo) || fp16::is_nan(hi)) {
      // If either bound is NaN, the range is invalid (NaN cannot be a bound)
      return false;
    }
    // If NaN is included, the range can be anything (even invalid bounds) because at least the NaN is there
    if (has_nan)
      return true;

    // For interval without nan: lo must be <= hi in FP comparison
    return fp16::le(lo, hi);
  }

  // Static constructors
  static constexpr FPRange bottom() noexcept {
    // Return canonical bottom element (empty set)
    // Convention: lo = +inf (0x7C00u), hi = -inf (0xFC00u), has_nan = false
    // This makes lo > hi in FP order, representing an impossible range
    return FPRange(fp16::POS_INF, fp16::NEG_INF, false);
  }

  static constexpr FPRange top() noexcept {
    // TODO: Return top element (all FP16 values including NaN)
    // Convention: lo = -inf, hi = +inf, has_nan = true
    return FPRange(fp16::NEG_INF, fp16::POS_INF, true);
  }

  static constexpr FPRange canonicaOnlyNan() noexcept {
    // Return canonical NaN-only range
    // Convention: lo = +inf, hi = -inf, has_nan = true (same as bottom but with NaN)
    return FPRange(fp16::POS_INF, fp16::NEG_INF, true);
  }



  static constexpr FPRange fromConcrete(std::uint16_t fp16_bits) noexcept {
    // TODO: Lift a concrete FP16 bit pattern to abstract domain
    // Should return a singleton range containing just this value
    if (fp16::is_nan(fp16_bits)) {
      return FPRange(fp16::POS_INF, fp16::NEG_INF, true);  // NaN-only range
    } else {
      return FPRange(fp16_bits, fp16_bits, false);  // Singleton range
    }
  }

  // Lattice enumeration
  static std::vector<FPRange> enumLattice() {
    // This method should not be used in practice due to the huge size of the lattice, throw an runtime error if called
    throw std::runtime_error("Lattice enumeration is not supported for FPRange due to its large size");
    return {};
  }

  // Returns representative FPRange elements by enumerating all valid (lo, hi) pairs
  // from fp16::get_rep_values(), each with and without NaN, plus the NaN-only range.
  static std::vector<FPRange> get_representative_rand() {
    auto vals = fp16::get_rep_values();
    std::vector<FPRange> reps;
    // All valid (lo, hi) combinations from representative values
    for (auto lo_val : vals) {
      for (auto hi_val : vals) {
        FPRange a_w_nan = FPRange(lo_val, hi_val, true);
        FPRange a_wo_nan = FPRange(lo_val, hi_val, false);
        if (!a_wo_nan.isBottom())
          reps.push_back(a_wo_nan);
        if (!a_w_nan.isBottom() && !a_w_nan.isOnlyNan())
          reps.push_back(a_w_nan);
      }
    }
    reps.push_back(canonicaOnlyNan());  // Add the NaN-only range as a representative

    return reps;
  }

  static constexpr std::uint64_t num_levels() noexcept {
    // TODO: Return the height of the lattice
    // Largest chain: bottom -> ... -> top
    // For a flat lattice this might be 2, for ranges it's more complex
    return 0;  // placeholder
  }

  static FPRange rand(std::mt19937 &rng, std::uint64_t /* level */) {
    // Generate random valid FPRange using uniform distribution
    std::uniform_int_distribution<std::uint16_t> dist(0, 0xFFFFu);
    std::uniform_int_distribution<int> bool_dist(0, 1);

    // Try generating until we get a valid FPRange (usually first try)
    for (int attempts = 0; attempts < 100; ++attempts) {
      std::uint16_t lo = dist(rng);
      std::uint16_t hi = dist(rng);
      bool has_nan = bool_dist(rng);
      // Ensure lo <= hi in FP order by swapping if needed
      if (fp16::gt(lo, hi)) {
        std::swap(lo, hi);
      }
      if (is_valid(lo, hi, has_nan)) {
        return FPRange(lo, hi, has_nan);
      }
    }
    // Fallback: return bottom
    return bottom();
  }
};

// LLVM ABI representation for {half, half, i1}
struct FPRangeRepr {
  std::uint16_t lo;      // half as 16 bits
  std::uint16_t hi;      // half as 16 bits
  std::uint8_t has_nan;  // i1 as byte (LLVM ABI for booleans)
} __attribute__((packed));

// Conversion helpers
inline FPRangeRepr to_repr(const FPRange &r) {
  return FPRangeRepr{r.lo, r.hi, static_cast<std::uint8_t>(r.has_nan)};
}

inline FPRange from_repr(const FPRangeRepr &r) {
  return FPRange{r.lo, r.hi, r.has_nan != 0};
}

// Helper: Free function that returns FPRange max_dist
// Needed because Results constructor expects a function pointer
inline std::uint64_t fprange_get_max_dist() {
  return 64;
}
