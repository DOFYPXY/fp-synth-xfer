#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "fprange.hpp"
#include "results.hpp"

// EvalFP: Evaluator for FPRange abstract domain
// Specialized for FP16 operations, no need for template bitwidths


namespace detail {

// Function pointer types for N-ary operations on FPRangeRepr
template <std::size_t N, std::size_t... Is>
auto fprange_xfer_fn_ptr(std::index_sequence<Is...>)
    -> FPRangeRepr (*)(std::conditional_t<true, FPRangeRepr,
                                          std::integral_constant<std::size_t, Is>>...);

template <std::size_t N>
using fprange_xfer_fn_t =
    decltype(fprange_xfer_fn_ptr<N>(std::make_index_sequence<N>{}));

} // namespace detail

// EvalFP<N>: Evaluate N-ary operations on FPRange
// N = number of arguments (e.g., 2 for binary operations)
template <std::size_t N>
class EvalFP {
public:
  // Type aliases
  using ArgsTuple = decltype([]<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::tuple<std::conditional_t<true, FPRange,
                                         std::integral_constant<std::size_t, Is>>...>{};
  }(std::make_index_sequence<N>{}));

  using Row = decltype([]<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::tuple<std::conditional_t<true, FPRange,
                                         std::integral_constant<std::size_t, Is>>...,
                      FPRange>{};
  }(std::make_index_sequence<N>{}));

  using EvalVec = std::vector<Row>;
  using XferFn = detail::fprange_xfer_fn_t<N>;

private:
  std::vector<XferFn> xfrFns;
  std::vector<XferFn> refFns;

public:
  // Constructor: takes function pointers for transfer functions and reference implementations
  constexpr EvalFP(const std::vector<std::uintptr_t> &xfrAddrs,
                   const std::vector<std::uintptr_t> &refAddrs)
      : xfrFns(xfrAddrs.size(), nullptr), refFns(refAddrs.size(), nullptr) {
    for (std::size_t i = 0; i < xfrFns.size(); ++i)
      xfrFns[i] = reinterpret_cast<XferFn>(xfrAddrs[i]);
    for (std::size_t i = 0; i < refFns.size(); ++i)
      refFns[i] = reinterpret_cast<XferFn>(refAddrs[i]);
  }

  // Main evaluation function
  Results eval(const EvalVec &toEval) const {
    // Fixed bitwidth of 16 for FP16
    constexpr unsigned int ResBw = 16;

    Results r{static_cast<unsigned int>(xfrFns.size()), ResBw,
              &fprange_get_max_dist};

    for (const Row &row : toEval) {
      const ArgsTuple args = take_args(row);
      const FPRange &best = std::get<N>(row);
      evalSingle(args, best, r);
    }

    return r;
  }

private:
  // Extract first N elements as argument tuple
  static ArgsTuple take_args(const Row &row) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return ArgsTuple{std::get<Is>(row)...};
    }(std::make_index_sequence<N>{});
  }

  // Evaluate a single row
  void evalSingle(const ArgsTuple &args, const FPRange &best,
                  Results &r) const {
    constexpr auto idxs = std::make_index_sequence<N>{};

    // Helper: run all functions in a vector and collect results
    auto run_fns = [&](const std::vector<XferFn> &fns) {
      std::vector<FPRange> out;
      out.reserve(fns.size());

      for (XferFn f : fns) {
        // Convert FPRange -> FPRangeRepr, call function, convert back
        FPRangeRepr result = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
          return f(to_repr(std::get<Is>(args))...);
        }(idxs);

        out.emplace_back(from_repr(result));
      }

      return out;
    };

    // Run synthesized transfer functions
    std::vector<FPRange> synth_results = run_fns(xfrFns);

    // Run reference implementations and meet them
    FPRange ref = meetAll(run_fns(refFns));

    // print args, synth[0], best
    // std::clog<< "Args: ";
    // std::apply(
    //     [&](auto const &...elems) {
    //       ((std::clog << elems << " "), ...);
    //     },
    //     args);
    // std::clog << "\nSynth: " << synth_results[1] << "\nBest: " << best << "\n";

    // Evaluate quality metrics
    bool solved = (ref == best);
    unsigned long baseDis = ref.distance(best);

    for (unsigned int i = 0; i < synth_results.size(); ++i) {
      FPRange synth_after_meet = ref.meet(synth_results[i]);
      bool sound = isSuperset(synth_after_meet, best);
      bool exact = (synth_after_meet == best);
      unsigned long dis = synth_after_meet.distance(best);
      unsigned long soundDis = sound ? dis : baseDis;

      r.incResult(Result(sound, dis, exact, solved, soundDis), i);
    }
    // std::clog << "Updated result for function " << r << "\n";
    r.incCases(solved, baseDis);
  }

  // Helper: check if lhs is superset of rhs (lhs.meet(rhs) == rhs)
  static bool isSuperset(const FPRange &lhs, const FPRange &rhs) {
    return lhs.meet(rhs) == rhs;
  }

  // Helper: meet all elements in vector
  static FPRange meetAll(const std::vector<FPRange> &v) {
    if (v.empty())
      return FPRange::top();

    FPRange result = v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      result = result.meet(v[i]);
    }

    return result;
  }
};
