#pragma once

#include <array>
#include <optional>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <iostream>

#include "fprange.hpp"
#include "rand.hpp"

// EnumFP: Generate test cases for FPRange domain
// Specialized for FP16 operations

namespace detail {

// Function pointer types for N-ary concrete operations on FP16
template <std::size_t N, std::size_t... Is>
auto fp16_concrete_fn_ptr(std::index_sequence<Is...>)
    -> std::uint16_t (*)(std::conditional_t<true, std::uint16_t,
                                            std::integral_constant<std::size_t, Is>>...);

template <std::size_t N>
using fp16_concrete_fn_t =
    decltype(fp16_concrete_fn_ptr<N>(std::make_index_sequence<N>{}));

// Function pointer types for N-ary constraint operations
template <std::size_t N, std::size_t... Is>
auto fp16_constraint_fn_ptr(std::index_sequence<Is...>)
    -> std::uint8_t (*)(std::conditional_t<true, std::uint16_t,
                                           std::integral_constant<std::size_t, Is>>...);

template <std::size_t N>
using fp16_constraint_fn_t =
    decltype(fp16_constraint_fn_ptr<N>(std::make_index_sequence<N>{}));

} // namespace detail

// EnumFP<N>: Enumerate test cases for N-ary FPRange operations
template <std::size_t N>
class EnumFP {
public:
  // Type aliases
  using ArgsTuple = decltype([]<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::tuple<std::conditional_t<true, FPRange,
                                         std::integral_constant<std::size_t, Is>>...>{};
  }(std::make_index_sequence<N>{}));

  using EvalVec = std::vector<decltype([]<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::tuple<std::conditional_t<true, FPRange,
                                         std::integral_constant<std::size_t, Is>>...,
                      FPRange>{};
  }(std::make_index_sequence<N>{}))>;

  using ConcOpFn = detail::fp16_concrete_fn_t<N>;
  using OpConFn = detail::fp16_constraint_fn_t<N>;

  // Constructor
  constexpr EnumFP(const std::uintptr_t concOpAddr,
                   const std::optional<std::uintptr_t> opConAddr)
      : concOp(reinterpret_cast<ConcOpFn>(concOpAddr)),
        opCon(opConAddr ? std::optional<OpConFn>(
                              reinterpret_cast<OpConFn>(*opConAddr))
                        : std::nullopt) {}

  // Generate "low" test cases: Cartesian product of lattice elements
  EvalVec genLows() const {
    EvalVec result;

    // Get lattice for each argument position
    auto lattices = make_lattices_tuple(std::make_index_sequence<N>{});

    // Enumerate Cartesian product
    ArgsTuple current{};
    for_each_combination<0>(lattices, current, [&](const ArgsTuple &args) {
      FPRange best = toBestAbst(args);
      result.emplace_back(tuple_cat_result(args, best));
    });

    return result;
  }

  // TODO: Generate "mid" test cases: Random lattice sampling
  EvalVec genMids(unsigned int num_lat_samples, std::mt19937 &rng,
                  const rngdist::Sampler &sampler) {
    EvalVec result;
    result.reserve(num_lat_samples);

    for (unsigned int i = 0; i < num_lat_samples; ++i) {
      ArgsTuple args = make_random_args(rng, sampler);
      FPRange best = toBestAbst(args);
      result.emplace_back(tuple_cat_result(args, best));
    }

    return result;
  }

  // TODO: Generate "high" test cases: Random lattice + concrete sampling
  EvalVec genHighs(unsigned int /* num_lat_samples */, unsigned int /* num_conc_samples */,
                   std::mt19937 & /* rng */, const rngdist::Sampler & /* sampler */) {
    // TODO: Implement mixed sampling: random lattice points + concrete value sampling
    return EvalVec();
  }

private:
  ConcOpFn concOp;
  std::optional<OpConFn> opCon;

  // Generate random arguments based on lattice sampling
  ArgsTuple make_random_args(std::mt19937 &rng,
                             const rngdist::Sampler &sampler) const {
    ArgsTuple result{};
    std::apply(
        [&](auto &...elems) {
          (
              [&] {
                const std::uint64_t hi = FPRange::num_levels();
                const std::uint64_t level = sampler(rng, 0ULL, hi);
                elems = FPRange::rand(rng, level);
              }(),
              ...);
        },
        result);
    return result;
  }

  // Compute best abstraction by enumerating all concrete combinations
  FPRange toBestAbst(const ArgsTuple &args) const {
    auto concSets = build_concrete_sets(args);
    FPRange result = FPRange::bottom();
    std::array<std::uint16_t, N> current{};

    for_each_conc_combination<0>(
        concSets, current, [&](const std::array<std::uint16_t, N> &vals) {
          // Check constraint if present
          if (opCon && apply_n_ary(*opCon, vals) == 0)
            return;

          // Apply concrete operation and lift result
          auto out = apply_n_ary(concOp, vals);
          result = result.join(FPRange::fromConcrete(out));
        });

    return result;
  }

  // Apply N-ary function to array of arguments
  template <typename F>
  static auto apply_n_ary(F f, const std::array<std::uint16_t, N> &vals) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return f(vals[Is]...);
    }(std::make_index_sequence<N>{});
  }

  // Create tuple of lattices (one for each argument position)
  template <std::size_t... Is>
  static auto make_lattices_tuple(std::index_sequence<Is...>) {
    return std::tuple<std::conditional_t<true, std::vector<FPRange>,
                                         std::integral_constant<std::size_t, Is>>...>{
        (static_cast<void>(Is), FPRange::enumLattice())...};
  }

  // Concatenate arguments tuple with result
  template <std::size_t... Is>
  static auto tuple_cat_result(const ArgsTuple &args, const FPRange &result) {
    return std::tuple_cat(args, std::make_tuple(result));
  }

  // Enumerate Cartesian product of lattice elements
  template <std::size_t I, typename LatticeTuple, typename CurrentTuple,
            typename Body>
  static void for_each_combination(const LatticeTuple &lattices,
                                   CurrentTuple &current, Body &&body) {
    if constexpr (I == N) {
      body(current);
    } else {
      const auto &vec = std::get<I>(lattices);
      for (const auto &v : vec) {
        std::get<I>(current) = v;
        for_each_combination<I + 1>(lattices, current,
                                    std::forward<Body>(body));
      }
    }
  }

  // Enumerate Cartesian product of concrete values
  template <std::size_t I, typename ConcSetsTuple, typename Body>
  static void for_each_conc_combination(const ConcSetsTuple &concSets,
                                        std::array<std::uint16_t, N> &current,
                                        Body &&body) {
    if constexpr (I == N) {
      body(current);
    } else {
      const auto &vec = std::get<I>(concSets);
      for (const auto &val : vec) {
        current[I] = val;
        for_each_conc_combination<I + 1>(concSets, current,
                                         std::forward<Body>(body));
      }
    }
  }

  // Build tuple of concrete value vectors from argument ranges
  static auto build_concrete_sets(const ArgsTuple &args) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return std::make_tuple(std::get<Is>(args).toConcrete()...);
    }(std::make_index_sequence<N>{});
  }
};
