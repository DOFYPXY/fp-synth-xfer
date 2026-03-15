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
#include "fp16_ops.hpp"

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

  // Generate "mid" test cases: representative combinations first, then random sampling
  EvalVec genMids(unsigned int num_lat_samples, std::mt19937 &rng,
                  const rngdist::Sampler &sampler) {
    EvalVec result;

    // Enumerate all combinations of representative elements
    auto rep_list = make_rep_args_list();
    for (const auto &args : rep_list) {
      FPRange best = toBestAbst(args);
      result.emplace_back(tuple_cat_result(args, best));
    }

    // If num_lat_samples exceeds the representative list size, do random sampling
    if (num_lat_samples > rep_list.size()) {
      unsigned int extra =
          num_lat_samples - static_cast<unsigned int>(rep_list.size());

      for (unsigned int i = 0; i < extra; ++i) {
        ArgsTuple args = make_random_args(rng, sampler);
        FPRange best = toBestAbst(args);
        result.emplace_back(tuple_cat_result(args, best));
      }
    }

    return result;
  }

  // Generate "high" test cases: representative combinations first, then random lattice + concrete sampling
  EvalVec genHighs(unsigned int num_lat_samples, unsigned int num_conc_samples,
                   std::mt19937 &rng, const rngdist::Sampler &sampler) {
    EvalVec result;

    // Enumerate all combinations of representative elements
    auto rep_list = make_rep_args_list();
    for (const auto &args : rep_list) {
      process_high_args(args, num_conc_samples, rng, result);
    }

    // If num_lat_samples exceeds the representative list size, do random sampling
    if (num_lat_samples > rep_list.size()) {
      unsigned int extra =
          num_lat_samples - static_cast<unsigned int>(rep_list.size());
      result.reserve(result.size() + extra);
      for (unsigned int i = 0; i < extra; ++i) {
        process_high_args(make_random_args(rng, sampler), num_conc_samples, rng, result);
      }
    }

    return result;
  }

private:
  ConcOpFn concOp;
  std::optional<OpConFn> opCon;

  // Process a single args tuple for genHighs: exact abstraction or concrete sampling.
  // Always enumerates all rep_conc combinations first, then does random sampling
  // for remaining budget if num_conc_samples >= product of rep_conc sizes.
  void process_high_args(const ArgsTuple &args, unsigned int num_conc_samples,
                         std::mt19937 &rng, EvalVec &result) const {
    FPRange res = FPRange::bottom();
    const std::uint64_t cap = static_cast<std::uint64_t>(num_conc_samples);
    const std::uint64_t total_space = capped_concrete_space(args, cap);
    if (total_space <= cap) {
      res = toBestAbst(args);
    } else {
      // Always enumerate all rep_conc combinations first
      auto rep_sets = build_rep_conc_sets(args);
      std::uint64_t rep_product = 1;
      std::apply([&](auto const &...vecs) {
        ((rep_product *= vecs.size()), ...);
      }, rep_sets);

      std::array<std::uint16_t, N> concretes{};
      for_each_conc_combination<0>(rep_sets, concretes, [&](const std::array<std::uint16_t, N> &vals) {
        if (opCon && apply_n_ary(*opCon, vals) == 0)
          return;
        res = res.join(FPRange::fromConcrete(apply_n_ary(concOp, vals)));
      });

      // Additional random sampling for remaining budget
      if (num_conc_samples >= rep_product) {
        unsigned int extra = num_conc_samples - static_cast<unsigned int>(rep_product);
        for (unsigned int j = 0; j < extra; ++j) {
          fill_sampled_concretes(args, rng, concretes);

          if (opCon && apply_n_ary(*opCon, concretes) == 0)
            continue;

          res = res.join(FPRange::fromConcrete(apply_n_ary(concOp, concretes)));
        }
      }
    }
    result.emplace_back(tuple_cat_result(args, res));
  }

  // Build the full list of N-ary Cartesian product combinations of representative elements
  std::vector<ArgsTuple> make_rep_args_list() const {
    auto reps = FPRange::get_representative_rand();
    auto rep_lattices = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return std::tuple<std::conditional_t<true, std::vector<FPRange>,
                                           std::integral_constant<std::size_t, Is>>...>{
          (static_cast<void>(Is), reps)...};
    }(std::make_index_sequence<N>{});

    std::vector<ArgsTuple> list;
    ArgsTuple current{};
    for_each_combination<0>(rep_lattices, current,
                            [&](const ArgsTuple &args) { list.push_back(args); });
    return list;
  }

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

  // Fill array with sampled concrete FP16 values from argument ranges
  void fill_sampled_concretes(const ArgsTuple &args, std::mt19937 &rng,
                              std::array<std::uint16_t, N> &out) const {
    std::size_t i = 0;
    std::apply(
        [&](auto const &...elems) {
          ((out[i++] = elems.sample_concrete(rng)), ...);
        },
        args);
  }

  // Compute product of concrete space sizes with overflow cap
static std::uint64_t capped_concrete_space(const ArgsTuple &args,
                                             std::uint64_t cap) {
    std::uint64_t product = 1;
    bool exceeded = false;

    std::apply(
        [&](auto const &...elems) {
          auto handle = [&](auto const &elem) {
            if (exceeded)
              return;

            std::uint64_t s = elem.size();

            if (s == 0) {
              s = 1;
            }

            if (product > cap / s) {
              exceeded = true;
              return;
            }

            product *= s;
          };

          (handle(elems), ...);
        },
        args);

    return exceeded ? (cap + 1) : product;
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

  // Build tuple of rep_conc value vectors from argument ranges
  static auto build_rep_conc_sets(const ArgsTuple &args) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return std::make_tuple(std::get<Is>(args).get_rep_conc()...);
    }(std::make_index_sequence<N>{});
  }
};
