#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include "domain.hpp"
#include "results.hpp"

// TODO join all should take a vec of uint64_t and do everything in bulk for
// faster perf, in these funcs (toBestAbst, genLows, genMids, genHighs)
//
// maybe use namespace DomainHelpers in this file

// TODO put this in a different file
template <template <std::size_t> class D, std::size_t BW>
  requires Domain<D, BW>
using ToEval = std::vector<std::tuple<D<BW>, D<BW>, D<BW>>>;
// TODO put this in a different file

// TODO make a domain helper or put in one of these classes?
template <std::size_t BW>
inline const constexpr std::array<std::uint64_t, 2>
pack(const std::array<APInt<BW>, 2> &arr) {
  return std::array<std::uint64_t, 2>{arr[0].getZExtValue(),
                                      arr[1].getZExtValue()};
}

// TODO make a domain helper or put in one of these classes?
template <std::size_t BW>
inline const constexpr std::array<APInt<BW>, 2>
unpack(const std::array<std::uint64_t, 2> &value) {
  return {APInt<BW>{value[0]}, APInt<BW>{value[1]}};
}

template <template <std::size_t> class Dom, std::size_t BW>
  requires Domain<Dom, BW>
class EnumDomain {
private:
  using D = Dom<BW>;
  using BV = APInt<BW>;
  using ConcOpFn = std::uint64_t (*)(std::uint64_t, std::uint64_t);
  // TODO idk if I'll shim this with an i64 or a bool
  using OpConFn = std::uint64_t (*)(std::uint64_t, std::uint64_t);
  using ToEval = ToEval<Dom, BW>;

  ConcOpFn concOp;
  std::optional<OpConFn> opCon;

  const D constexpr toBestAbst(const D &lhs, const D &rhs) const {
    const std::vector<BV> lhsCncSet = lhs.toConcrete();
    const std::vector<BV> rhsCncSet = rhs.toConcrete();

    D res = D::bottom();
    for (const BV &lhsCnc : lhsCncSet)
      for (const BV &rhsCnc : rhsCncSet)
        if (!opCon ||
            opCon.value()(lhsCnc.getZExtValue(), rhsCnc.getZExtValue()))
          res = res.join(D::fromConcrete(
              BV(concOp(lhsCnc.getZExtValue(), rhsCnc.getZExtValue()))));

    return res;
  }

public:
  constexpr EnumDomain(const std::uintptr_t concOpAddr,
                       const std::optional<std::uintptr_t> opConAddr)
      : concOp(reinterpret_cast<ConcOpFn>(concOpAddr)),
        opCon(opConAddr ? std::optional<OpConFn>(
                              reinterpret_cast<OpConFn>(*opConAddr))
                        : std::nullopt) {}

  // TODO this is a bad function name
  const ToEval genLows() const {
    const std::vector<D> lattice = D::enumLattice();

    ToEval r;
    r.reserve(lattice.size() * lattice.size());

    for (const D &lhs : lattice)
      for (const D &rhs : lattice)
        r.emplace_back(lhs, rhs, toBestAbst(lhs, rhs));

    return r;
  }

  const ToEval genMids(unsigned int num_lat_samples, std::mt19937 &rng) {

    ToEval r;
    r.reserve(num_lat_samples);
    for (unsigned int i = 0; i < num_lat_samples; ++i) {
      while (true) {
        const D lhs = D::rand(rng);
        const D rhs = D::rand(rng);
        const D res = toBestAbst(lhs, rhs);
        if (!res.isBottom()) {
          r.emplace_back(lhs, rhs, res);
          break;
        }
      }
    }

    return r;
  }

  const ToEval genHighs(unsigned int num_lat_samples,
                        unsigned int num_conc_samples, std::mt19937 &rng) {
    ToEval r;
    r.reserve(num_lat_samples);
    for (unsigned int i = 0; i < num_lat_samples; ++i) {
      const D lhs = D::rand(rng);
      const D rhs = D::rand(rng);
      D res = D::bottom();

      for (unsigned int j = 0; j < num_conc_samples; ++j) {
        const std::uint64_t lhsConc = lhs.sample_concrete(rng).getZExtValue();
        const std::uint64_t rhsConc = rhs.sample_concrete(rng).getZExtValue();
        if (!opCon || opCon.value()(lhsConc, rhsConc))
          res = res.join(D::fromConcrete(BV(concOp(lhsConc, rhsConc))));
      }

      r.emplace_back(lhs, rhs, res);
    }

    return r;
  }
};

template <template <std::size_t> class Dom, std::size_t BW>
  requires Domain<Dom, BW>
class Eval {
private:
  using D = Dom<BW>;
  using XferFn = std::array<std::uint64_t, 2> (*)(std::array<std::uint64_t, 2>,
                                                  std::array<std::uint64_t, 2>);

  std::vector<XferFn> xfrFns;
  std::vector<XferFn> refFns;

  std::vector<D> synFnWrapper(const D &lhs, const D &rhs) const {
    std::vector<D> r;
    r.reserve(xfrFns.size());

    for (const XferFn &f : xfrFns)
      r.emplace_back(D(unpack<BW>(f(pack<BW>(lhs.v), pack<BW>(rhs.v)))));

    return r;
  }

  std::vector<D> refFnWrapper(const D &lhs, const D &rhs) const {
    std::vector<D> r;
    r.reserve(refFns.size());

    for (const XferFn &f : refFns)
      r.emplace_back(D(unpack<BW>(f(pack<BW>(lhs.v), pack<BW>(rhs.v)))));

    return r;
  }

  void evalSingle(const D &lhs, const D &rhs, const D &best, Results &r) const {
    std::vector<D> synth_results(synFnWrapper(lhs, rhs));
    D ref = DomainHelpers::meetAll(refFnWrapper(lhs, rhs));
    bool solved = ref == best;
    unsigned long baseDis = ref.distance(best);
    for (unsigned int i = 0; i < synth_results.size(); ++i) {
      D synth_after_meet = ref.meet(synth_results[i]);
      bool sound = DomainHelpers::isSuperset(synth_after_meet, best);
      bool exact = synth_after_meet == best;
      unsigned long dis = synth_after_meet.distance(best);
      unsigned long soundDis = sound ? dis : baseDis;

      r.incResult(Result(sound, dis, exact, solved, soundDis), i);
    }

    r.incCases(solved, baseDis);
  }

public:
  constexpr Eval(const std::vector<std::uintptr_t> &_xfrFns,
                 const std::vector<std::uintptr_t> &_refFns)
      : xfrFns(_xfrFns.size(), nullptr), refFns(_refFns.size(), nullptr) {
    for (std::size_t i = 0; i < xfrFns.size(); ++i)
      xfrFns[i] = reinterpret_cast<XferFn>(_xfrFns[i]);

    for (std::size_t i = 0; i < refFns.size(); ++i)
      refFns[i] = reinterpret_cast<XferFn>(_refFns[i]);
  }

  const Results eval(const ToEval<Dom, BW> toEval) const {
    Results r{static_cast<unsigned int>(xfrFns.size()), BW, D::maxDist};

    for (const auto [lhs, rhs, best] : toEval)
      evalSingle(lhs, rhs, best, r);

    return r;
  }
};
