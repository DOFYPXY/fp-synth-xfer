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

  // Pretty printing
  friend std::ostream &operator<<(std::ostream &os, const FPRange &x) {
    // Display as: [lo, hi] (has_nan) where lo/hi are FP16 values
    // Example output: "[0.5, 1.5] (has_nan)" or "[0.5, 1.5]"

    os << "[";

    // Handle special cases for lo
    if (fp16::is_pos_inf(x.lo)) {
      os << "+inf";
    } else if (fp16::is_neg_inf(x.lo)) {
      os << "-inf";
    } else if (fp16::is_nan(x.lo)) {
      os << "nan";
    } else {
      os << fp16::to_float(x.lo);
    }

    os << ", ";

    // Handle special cases for hi
    if (fp16::is_pos_inf(x.hi)) {
      os << "+inf";
    } else if (fp16::is_neg_inf(x.hi)) {
      os << "-inf";
    } else if (fp16::is_nan(x.hi)) {
      os << "nan";
    } else {
      os << fp16::to_float(x.hi);
    }

    os << "]";

    if (x.has_nan) {
      os << " (has_nan)";
    }

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
    // Convention: has_nan=false AND (lo > hi in FP order)
    // For FP16: use sentinel lo=0x7C00u (+inf), hi=0xFC00u (-inf) to represent empty
    if (has_nan) return false;
    return fp16::lt(hi, lo);  // hi < lo in FP order means empty range
  }

  // Lattice operations
  FPRange meet(const FPRange &rhs) const noexcept {
    // Intersection of two ranges
    // meet([lo1, hi1], [lo2, hi2], nan1, nan2) =
    //   [max(lo1, lo2), min(hi1, hi2)], (nan1 && nan2)
    // If resulting lo > hi, return bottom (empty set)
    std::uint16_t new_lo = fp16::max(lo, rhs.lo);
    std::uint16_t new_hi = fp16::min(hi, rhs.hi);
    bool new_nan = has_nan && rhs.has_nan;
    return FPRange(new_lo, new_hi, new_nan);
  }

  FPRange join(const FPRange &rhs) const noexcept {
    // Union of two ranges
    // join([lo1, hi1], [lo2, hi2], nan1, nan2) =
    //   [min(lo1, lo2), max(hi1, hi2)], (nan1 || nan2)
    std::uint16_t new_lo = fp16::min(lo, rhs.lo);
    std::uint16_t new_hi = fp16::max(hi, rhs.hi);
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

  std::uint16_t sample_concrete(std::mt19937 & /* rng */) const {
    // TODO: Sample a concrete FP16 value uniformly from this range
    // If has_nan, occasionally sample a NaN bit pattern
    // Otherwise pick a random FP value in [lo, hi]
    return 0;
  }

  std::uint64_t size() const noexcept {
    if (isBottom()) return 0;

    std::uint16_t lo_sign = fp16::extract_sign(lo);
    std::uint16_t hi_sign = fp16::extract_sign(hi);
    std::uint16_t lo_exp = fp16::extract_exp(lo);
    std::uint16_t hi_exp = fp16::extract_exp(hi);

    std::uint16_t exp_count = 0;

    if (lo_sign == 0 && hi_sign == 0) {
      // Same sign: simple exponent range
      exp_count = hi_exp - lo_exp + 1;
    }
    else if (lo_sign == 1 && hi_sign == 1) {
      // Same sign: simple exponent range
      exp_count = lo_exp - hi_exp + 1;
    }
    else {
      // Crosses zero: sum positive and negative sections
      exp_count = lo_exp + hi_exp + 2;
    }
    // clog to print the FPRange itself
    // std::clog << *this << "\n";
    // std::clog << "Size: exp_count=" << exp_count << ", has_nan=" << has_nan << "\n";
    uint64_t total_size = exp_count + (has_nan ? 1 : 0);
    // std::clog << "Total size: " << total_size << "\n";
    return total_size;
  }

  // Distance metric
  std::uint64_t distance(const FPRange &rhs) const noexcept {
    // TODO: Define distance metric on the lattice
    // Example: count of bit patterns that differ between the two ranges
    // or symmetric difference of concrete sets
    uint64_t this_size = size();
    uint64_t rhs_size = rhs.size();
    // calculate size diff
    uint64_t size_diff = (this_size > rhs_size) ? (this_size - rhs_size) : (rhs_size - this_size);
    return size_diff;
  }

  // Validation helper
  static constexpr bool is_valid(std::uint16_t lo, std::uint16_t hi, bool /* has_nan */) noexcept {
    if (fp16::is_nan(lo) || fp16::is_nan(hi)) {
      // If either bound is NaN, the range is invalid (NaN cannot be a bound)
      return false;
    }
    // For non-bottom: lo must be <= hi in FP comparison
    return fp16::le(lo, hi);
  }

  // Static constructors
  static constexpr FPRange bottom() noexcept {
    // Return bottom element (empty set)
    // Convention: lo = +inf (0x7C00u), hi = -inf (0xFC00u), has_nan = false
    // This makes lo > hi in FP order, representing an impossible range
    return FPRange(fp16::POS_INF, fp16::NEG_INF, false);
  }

  static constexpr FPRange top() noexcept {
    // TODO: Return top element (all FP16 values including NaN)
    // Convention: lo = -inf, hi = +inf, has_nan = true
    return FPRange(fp16::NEG_INF, fp16::POS_INF, true);
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
    // Enumerate elements of the FPRange lattice
    // Enumerate lo, hi, and has_nan combinations that form valid FPRange elements
    // Use representative FP16 values to keep size tractable
    std::vector<FPRange> res;

    // Select representative FP16 values for enumeration
    std::vector<std::uint16_t> representative_values = {
        fp16::NEG_INF,   // -inf
        0xBC00u,         // -1.0
        fp16::NEG_ZERO,  // -0.0
        fp16::POS_ZERO,  // +0.0
        0x3C00u,         // +1.0
        fp16::POS_INF,   // +inf
    };

    // Enumerate all valid (lo, hi, has_nan) combinations
    for (std::uint16_t lo : representative_values) {
      for (std::uint16_t hi : representative_values) {
        for (bool has_nan : {false, true}) {
          if (is_valid(lo, hi, has_nan)) {
            res.push_back(FPRange(lo, hi, has_nan));
          }
        }
      }
    }

    return res;
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
