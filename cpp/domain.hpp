#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "apint.hpp"

// TODO rand should take latice_level as a param so we can change the dist of
// the abst vals we sample

template <template <std::size_t> class D, std::size_t BW>
concept Domain =
    requires(const D<BW> d, const APInt<BW> &a, std::size_t i, std::ostream &os,
             std::mt19937 &rng) {
      typename D<BW>::BV;
      requires std::same_as<typename D<BW>::BV, APInt<BW>>;

      { D<BW>::arity } -> std::convertible_to<std::size_t>;
      { D<BW>::name } -> std::convertible_to<const std::string>;
      requires(D<BW>::arity >= 2 && D<BW>::arity <= 6);
      { d[i] } noexcept -> std::same_as<const APInt<BW> &>;

      // Static methods
      { D<BW>::rand(rng) } -> std::same_as<const D<BW>>;
      { D<BW>::bottom() } noexcept -> std::same_as<const D<BW>>;
      { D<BW>::top() } noexcept -> std::same_as<const D<BW>>;
      {
        D<BW>::enumLattice()
      } noexcept -> std::same_as<const std::vector<D<BW>>>;
      { D<BW>::fromConcrete(a) } noexcept -> std::same_as<const D<BW>>;
      { D<BW>::maxDist() } noexcept -> std::same_as<double>;

      // Instance methods
      { d.isBottom() } noexcept -> std::same_as<bool>;
      { d.meet(d) } noexcept -> std::same_as<const D<BW>>;
      { d.join(d) } noexcept -> std::same_as<const D<BW>>;
      { d.toConcrete() } noexcept -> std::same_as<const std::vector<APInt<BW>>>;
      { d.distance(d) } noexcept -> std::same_as<std::uint64_t>;
      { d.sample_concrete(rng) } -> std::same_as<const APInt<BW>>;
      { os << d } -> std::same_as<std::ostream &>;
    } &&
    std::constructible_from<D<BW>, const std::array<APInt<BW>, D<BW>::arity> &>;
;

template <template <std::size_t> class D, std::size_t BW>
  requires Domain<D, BW>
constexpr bool operator==(const D<BW> &lhs, const D<BW> &rhs) {
  for (std::size_t i = 0; i < D<BW>::arity; ++i)
    if (!(lhs[i] == rhs[i]))
      return false;

  return true;
}

template <template <std::size_t> class D, std::size_t BW>
  requires Domain<D, BW>
constexpr bool operator!=(const D<BW> &lhs, const D<BW> &rhs) {
  return !(lhs == rhs);
}

namespace DomainHelpers {
template <template <std::size_t> class D, std::size_t BW>
  requires Domain<D, BW>
bool constexpr isTop(const D<BW> &d) {
  return d == D<BW>::top();
}

template <template <std::size_t> class D, std::size_t BW>
  requires Domain<D, BW>
bool constexpr isSuperset(const D<BW> &lhs, const D<BW> &rhs) {
  return lhs.meet(rhs) == rhs;
}

template <template <std::size_t> class D, std::size_t BW>
  requires Domain<D, BW>
const D<BW> constexpr joinAll(const std::vector<D<BW>> &v) {
  if (v.empty())
    return D<BW>::bottom();

  D<BW> d = v[0];
  for (std::size_t i = 1; i < v.size(); ++i) {
    d = d.join(v[i]);
  }

  return d;
}

template <template <std::size_t> class D, std::size_t BW>
  requires Domain<D, BW>
const D<BW> constexpr meetAll(const std::vector<D<BW>> &v) {
  if (v.empty())
    return D<BW>::top();

  D<BW> d = v[0];
  for (unsigned int i = 1; i < v.size(); ++i) {
    d = d.meet(v[i]);
  }

  return d;
}
} // namespace DomainHelpers
