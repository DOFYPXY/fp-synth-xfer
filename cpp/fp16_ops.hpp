#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

// FP16 Operations: Utilities for FP16 (half-precision) floating point operations
// FP16 is represented as a 16-bit unsigned integer (bit pattern)
// This file provides classification, comparison, and min/max operations.

namespace fp16 {

// Special FP16 bit patterns
constexpr std::uint16_t POS_INF = 0x7C00u;   // +infinity
constexpr std::uint16_t NEG_INF = 0xFC00u;   // -infinity
constexpr std::uint16_t POS_ZERO = 0x0000u;  // +0.0
constexpr std::uint16_t NEG_ZERO = 0x8000u;  // -0.0
constexpr std::uint16_t NAN_PATTERN = 0x7E00u;  // One of many NaN patterns
constexpr std::uint16_t MAX_POS = 0x7FFFu;  // Maximum positive value (all positive)
constexpr std::uint16_t MIN_NEG = 0x8000u;  // Minimum negative value (sign bit only)

// Check if an FP16 bit pattern represents NaN
// NaN has exponent = 31 (0x1F) and non-zero mantissa
constexpr bool is_nan(std::uint16_t bits) noexcept {
  std::uint16_t exponent = (bits >> 10) & 0x1Fu;
  std::uint16_t mantissa = bits & 0x3FFu;
  return exponent == 0x1Fu && mantissa != 0;
}

// Check if an FP16 bit pattern represents positive infinity
constexpr bool is_pos_inf(std::uint16_t bits) noexcept {
  return bits == POS_INF;
}

// Check if an FP16 bit pattern represents negative infinity
constexpr bool is_neg_inf(std::uint16_t bits) noexcept {
  return bits == NEG_INF;
}

// Check if an FP16 bit pattern represents infinity (positive or negative)
constexpr bool is_inf(std::uint16_t bits) noexcept {
  return is_pos_inf(bits) || is_neg_inf(bits);
}

// Check if an FP16 bit pattern represents zero (positive or negative)
constexpr bool is_zero(std::uint16_t bits) noexcept {
  return (bits & 0x7FFFu) == 0;  // Exponent and mantissa both 0
}

// Extract sign bit: 1 = negative, 0 = positive
constexpr bool is_negative(std::uint16_t bits) noexcept {
  return (bits & 0x8000u) != 0;
}

// Extract sign bit value (0 for positive, 1 for negative)
constexpr std::uint16_t extract_sign(std::uint16_t bits) noexcept {
  return (bits >> 15) & 0x01u;
}

// Extract exponent bits (bits 10-14, 5 bits total)
// Returns value in range [0, 31]
constexpr std::uint16_t extract_exp(std::uint16_t bits) noexcept {
  return (bits >> 10) & 0x1Fu;
}

// Compare two FP16 bit patterns
// Returns -1 if a < b, 0 if a == b, 1 if a > b
// For comparison purposes, treats bit patterns as signed integers
// (This works because FP16 format is sign-magnitude with special handling)
inline int compare(std::uint16_t a, std::uint16_t b) noexcept {
  if (is_nan(a) || is_nan(b)) return 0;  // NaN is unordered

  // Sign-magnitude comparison for FP16 bit patterns
  bool a_neg = is_negative(a);
  bool b_neg = is_negative(b);

  if (a_neg != b_neg) {
    return a_neg ? -1 : 1;
  }

  // Same sign: compare magnitudes
  std::uint16_t a_mag = a & 0x7FFFu;
  std::uint16_t b_mag = b & 0x7FFFu;

  if (a_mag < b_mag) return a_neg ? 1 : -1;   // Negative: reversed
  if (a_mag > b_mag) return a_neg ? -1 : 1;    // Negative: reversed
  return 0;
}

// Return the maximum of two FP16 values (IEEE 754-2008 maxNum)
// If one operand is NaN, return the other; if both are NaN, return NaN
inline std::uint16_t maxnum(std::uint16_t a, std::uint16_t b) noexcept {
  if (is_nan(a)) return b;
  if (is_nan(b)) return a;
  return compare(a, b) >= 0 ? a : b;
}

// Return the minimum of two FP16 values (IEEE 754-2008 minNum)
// If one operand is NaN, return the other; if both are NaN, return NaN
inline std::uint16_t minnum(std::uint16_t a, std::uint16_t b) noexcept {
  if (is_nan(a)) return b;
  if (is_nan(b)) return a;
  return compare(a, b) <= 0 ? a : b;
}

// Return the maximum of two FP16 values (IEEE 754-2019 maximum)
// If either operand is NaN, return NaN
inline std::uint16_t maximum(std::uint16_t a, std::uint16_t b) noexcept {
  if (is_nan(a)) return a;
  if (is_nan(b)) return b;
  return compare(a, b) >= 0 ? a : b;
}

// Return the minimum of two FP16 values (IEEE 754-2019 minimum)
// If either operand is NaN, return NaN
inline std::uint16_t minimum(std::uint16_t a, std::uint16_t b) noexcept {
  if (is_nan(a)) return a;
  if (is_nan(b)) return b;
  return compare(a, b) <= 0 ? a : b;
}

// Greater than: returns true if a > b (false if either is NaN)
inline bool gt(std::uint16_t a, std::uint16_t b) noexcept {
  return compare(a, b) > 0;
}

// Greater than or equal: returns true if a >= b (false if either is NaN)
inline bool ge(std::uint16_t a, std::uint16_t b) noexcept {
  return compare(a, b) >= 0;
}

// Less than: returns true if a < b (false if either is NaN)
inline bool lt(std::uint16_t a, std::uint16_t b) noexcept {
  return compare(a, b) < 0;
}

// Less than or equal: returns true if a <= b (false if either is NaN)
inline bool le(std::uint16_t a, std::uint16_t b) noexcept {
  return compare(a, b) <= 0;
}

// Convert FP16 bit pattern to float for printing
inline float to_float(std::uint16_t bits) noexcept {
  // FP16 format: 1 sign bit, 5 exponent bits, 10 mantissa bits
  std::uint32_t sign = (bits & 0x8000u) << 16;  // Move sign to bit 31
  std::uint32_t exp = (bits & 0x7C00u) >> 10;   // Extract exponent (5 bits)
  std::uint32_t mantissa = (bits & 0x03FFu);     // Extract mantissa (10 bits)

  std::uint32_t fp32_bits;

  if (exp == 0x1F) {  // Inf or NaN
    fp32_bits = sign | 0x7F800000u | (mantissa << 13);
  } else if (exp == 0) {  // Subnormal or zero
    if (mantissa == 0) {
      fp32_bits = sign;  // Zero
    } else {
      // Subnormal: normalize it
      exp = 0x71;  // -14 in FP32 bias
      while ((mantissa & 0x400) == 0) {
        mantissa <<= 1;
        exp--;
      }
      mantissa &= 0x3FF;
      fp32_bits = sign | (exp << 23) | (mantissa << 13);
    }
  } else {  // Normal number
    fp32_bits = sign | ((exp + 112) << 23) | (mantissa << 13);  // 112 = 127 - 15 (bias adjust)
  }

  float result;
  std::memcpy(&result, &fp32_bits, sizeof(result));
  return result;
}

// Returns a small set of representative FP16 bit patterns covering key boundary values.
// Used by FPRange::get_representative_rand() to generate representative ranges.
inline std::vector<std::uint16_t> get_rep_values() {
  return {
      NEG_INF,   // -inf
      // 0xBC00u,   // -1.0
      NEG_ZERO,  // -0.0
      POS_ZERO,  // +0.0
      // 0x3C00u,   // +1.0
      POS_INF,   // +inf
      NAN_PATTERN // NaN
  };
}

}  // namespace fp16
