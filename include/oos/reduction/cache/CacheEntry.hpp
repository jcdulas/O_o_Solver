#pragma once

#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/reduction/condense/ModalReduction.hpp"
#include "oos/reduction/condense/ReducedUnit.hpp"

#include <cstdint>
#include <memory>
#include <string_view>

namespace oos::reduction
{

// The two kinds of what the condensation cache holds ("the recovery data, that is the Craig-Bampton
// modes", under their own volume limit).
enum class CacheEntryKind : std::uint8_t
{
    Reduction = 1,   // a reduced unit (CachedReduction)
    RecoveryData = 2 // the fixed-interface modes of a Craig-Bampton unit (CachedRecoveryData)
};

[[nodiscard]] constexpr std::string_view cacheEntryKindName(CacheEntryKind kind) noexcept
{
    return kind == CacheEntryKind::Reduction ? "reduction" : "recovery_data";
}

// The cached static condensation of one unit: all that a cache may keep of it (never an interior factor nor a recovery
// matrix, which are large): the reduced unit (its boundary labels, K*, the condensed load patterns f*_k, the condensed
// inertial patterns, the rigid-body mass properties) and two values derived from the patterns' condensation:
// - patternWork G = F_oᵀ·K_oo⁻¹·F_o (patterns × patterns, symmetric), the interior work of the condensed load
//   patterns, with which the external work of a case is known without recovering the unit
//   (StaticCondensation::patternWork); a few numbers per pair of patterns, a quantity of the condensed patterns;
// - largestPivotRatio, the MAXRATIO diagnostic of the interior factorisation, reported as the factorisation would.
// Never an interior factor, never a recovery matrix K_oo⁻¹·K_oa: the recovery of a unit taken from the cache factorises
// its interior again (StaticCondensation::recomputeInteriorFactor), to the same bits.
//
// Thread-safety: immutable once made; shared between the cache's memory layer and the analyses that use it. Ownership:
// owns its matrices, budgeted in the budget given when it was made, which must outlive it.
//
// Craig-Bampton ("the reduced stiffness and mass are stored like K*") stores two reduction
// entries per unit, so that the boundary part serves every mode request:
// - the boundary part (CacheKey::modeCount 0): the entry above with boundaryMass, M_aa = Ψ̃ᵀ·M·Ψ̃, the reduced mass of
//   the boundary (the statically condensed model's mass, whose eigenvalues give the cut-off of an ND request);
// - the modal part (CacheKey::modeCount N, or a cut-off): `modal` alone (the q set: the eigenvalues of the
//   fixed-interface modes, K_qq, M_aq, M_qq), `reduced` null; the modes themselves are recovery data
//   (CachedRecoveryData) under the same key.
struct CachedReduction
{
    std::shared_ptr<const ReducedUnit> reduced;
    linalg::DenseMatrix patternWork;
    double largestPivotRatio = 0.0;
    std::int64_t interiorDofCount = 0;
    linalg::DenseMatrix boundaryMass;            // M_aa (boundary DOFs square), or empty
    std::shared_ptr<const ModalReduction> modal; // the modal part entry's, or null
    // The checksum of the equations it was condensed from (CacheKey::equationStamp, equationStampOf): what a hit whose
    // equations are built again compares them with.
    std::uint64_t equationStamp = 0;

    // The bytes it holds (what its entry weighs in memory; its file is about this size).
    [[nodiscard]] std::uint64_t byteCount() const noexcept
    {
        const auto work = static_cast<std::uint64_t>(patternWork.rows() * patternWork.columns()) * sizeof(double);
        const auto mass = static_cast<std::uint64_t>(boundaryMass.rows() * boundaryMass.columns()) * sizeof(double);
        return (reduced ? reduced->byteCount() : 0) + work + mass + (modal ? modal->byteCount() : 0);
    }
};

// The recovery data of a Craig-Bampton unit ("the fixed-interface modes of a unit (its interior shapes
// and their eigenvalues, 8·n_o·N bytes) ... stored in the cache as recovery data, within their own volume limit"): the
// N fixed-interface modes on the unit's interior equations and their eigenvalues. Made by the Craig-Bampton reduction
// (CraigBampton::computeModes), stored under the key of the modal part's reduction entry; the cache stores
// and evicts it under cache_recovery_max_size, before the reductions, and an analysis that recovers the unit without
// it computes the modes again, to the same bits.
//
// Thread-safety and ownership: as CachedReduction.
struct CachedRecoveryData
{
    linalg::DenseMatrix modes;       // interior equations × N
    linalg::DenseMatrix eigenvalues; // N × 1

    [[nodiscard]] std::uint64_t byteCount() const noexcept
    {
        return static_cast<std::uint64_t>(modes.rows() * modes.columns() + eigenvalues.rows() * eigenvalues.columns()) *
               sizeof(double);
    }
};

} // namespace oos::reduction
