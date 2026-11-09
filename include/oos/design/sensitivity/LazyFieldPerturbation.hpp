#pragma once

#include "oos/design/sensitivity/FieldPerturbation.hpp"

#include <functional>
#include <mutex>
#include <optional>

namespace oos::design
{

// The perturbation of one designed field, made at its first use: the semi-analytic derivatives
// read it only where a family has no closed form for the field (ElementDerivatives) or an item's explicit part is
// differenced, so that a field whose every computation is in closed form (the AnisoMat2D terms and the T of a
// lamination zone) never builds the four overlaid snapshots and their element sets.
//
// PERFORMANCE: a laminated model at 1/1024 of its full size (94 zones, 1 786 fields)
// spent 3.4 s of its three linearisations' 4.8 s in the fields' perturbations (some 0.6 ms each: four snapshots of the
// point overlaid and indexed), which the closed forms of the shells in their AnisoMat2D terms no longer read; a field
// that needs them builds the same perturbation as before, so the same bits. Measured: 3.4 s of 4.8 s in the
// perturbations before, on the model above.
//
// Thread-safety: get() may be called concurrently (the element terms of a wave of fields run on the pool); the
// perturbation is built once, by the first caller, under a std::once_flag. Neither copyable nor movable. Ownership:
// owns the perturbation once built and the function that builds it (whose captures must outlive the first get()).
class LazyFieldPerturbation
{
public:
    // A perturbation built already (a member of a batch, a shape's).
    explicit LazyFieldPerturbation(FieldPerturbation built);
    // A perturbation `make()` builds at the first get().
    explicit LazyFieldPerturbation(std::function<FieldPerturbation()> make);

    LazyFieldPerturbation(const LazyFieldPerturbation&) = delete;
    LazyFieldPerturbation& operator=(const LazyFieldPerturbation&) = delete;

    // The perturbation, built at the first call.
    [[nodiscard]] const FieldPerturbation& get() const;
    // Whether get() was called (the field was differenced somewhere); meaningful once the concurrent callers are done.
    [[nodiscard]] bool isBuilt() const noexcept { return m_value.has_value(); }

private:
    std::function<FieldPerturbation()> m_make;
    mutable std::once_flag m_once;
    mutable std::optional<FieldPerturbation> m_value;
};

} // namespace oos::design
