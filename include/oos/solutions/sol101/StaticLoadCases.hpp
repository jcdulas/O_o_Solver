#pragma once

#include "oos/assembly/constraints/SpcSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/loads/LoadBasis.hpp"
#include "oos/assembly/loads/LoadFamily.hpp"
#include "oos/assembly/loads/LoadVectorBuilder.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/input/tables/PointIndex.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace oos::solutions
{

// The load cases of one constraint set of a static analysis and their load basis (many load cases solved on
// one basis): for each family of loads (LoadFamily), the compact load matrix of the cases (the loaded DOFs of the
// concentrated loads, the card values of the distributed loads, the load sets of the inertial loads, the temperature
// sets of the thermal loads, the enforced values of the enforced displacements) and, where the deck gives them, the
// elementary load sets with the factors of the LOAD combinations (loadsTable: `source_sid` and `scale`); then the
// basis of each family (LoadBasis), every case being Σ α_i times the basis vectors of every family.
//
// The basis vectors are made on demand (basisVector), one g-set load vector and one g-set vector of enforced
// displacements each: a load set as written, a unit load on a loaded DOF (or a unit enforced displacement), or the
// family part of a case.
//
// Thread-safety: const methods are safe concurrently (each call fills its own vectors; what they call holds no mutable
// state: the load builder, the temperature fields, the inertial loads, the element families' load facets; the budget
// and the sink are thread-safe, so the basis vectors are made in parallel). Ownership: owns the bases and
// the coefficients; borrows the snapshot, the DOF map, the load builder, the cases and their SPC sets, and the mass.
class StaticLoadCases
{
public:
    struct Inputs
    {
        const input::ModelSnapshot* snapshot = nullptr;
        const input::PointIndex* points = nullptr;
        const assembly::DofMap* dofMap = nullptr;
        const assembly::LoadVectorBuilder* loads = nullptr;
        std::vector<const StaticSubcase*> cases;         // the solved subcases of the constraint set
        std::vector<const assembly::SpcSet*> spcs;       // their SPC sets (enforced displacements)
        const linalg::BlockSparseMatrix* mass = nullptr; // Mgg, needed by the inertial loads
        assembly::LoadBasisTolerance tolerance = assembly::LoadBasisTolerance::Roundoff;
        base::ThreadPool* pool = nullptr;
        // The kernel family of the analysis: the matrix products of the rank test's certificate (LoadBasis).
        linalg::KernelKind kernel = linalg::KernelKind::Reference;
        // Keep the description of every family (familyDescription): the condensed path chooses a basis per unit on
        // it.
        bool keepsFamilies = false;
        // The full path streamed: no mass, the inertial rows checked by StreamedLoads.
        bool checksInertialLoads = true;
    };

    // Where a compact row of the distributed loads acts: its element, or the grid points of a PointsPressureLoad face
    // (elementId 0).
    struct DistributedRowPlace
    {
        std::int32_t elementId = 0;
        std::array<std::int32_t, 4> faceGrids{};
    };

    // The loads of one family as the load basis chooses on them (assembly::LoadBasis::Input), kept with
    // Inputs::keepsFamilies: the compact matrix of the cases, the load sets as written with their factors, and what
    // each compact row is (the DOF of a concentrated load or an enforced displacement; the element or face of a
    // distributed card value; the inertial and thermal rows are load sets, fields over the whole model).
    struct FamilyDescription
    {
        assembly::LoadFamily family = assembly::LoadFamily::Concentrated;
        assembly::CompactLoadMatrix cases;
        std::optional<assembly::CompactLoadMatrix> loadSets;
        std::optional<linalg::DenseMatrix> setFactors; // with loadSets (sets × cases)
        bool offersLoadedDofs = false;
        assembly::LoadBasisTolerance tolerance = assembly::LoadBasisTolerance::Roundoff;
        std::span<const base::DofIndex> dofOfRow;         // concentrated and enforced
        base::BudgetedBuffer<DistributedRowPlace> places; // distributed
        // The cases are exact combinations of the load sets with their factors
        // (assembly::LoadBasis::Input::isCombinationOfSets): `cases` is then empty, the rows those of the sets.
        bool isCombinationOfSets = false;
        std::int64_t rowCount = 0; // the compact rows (of the cases and the sets)
    };

    // Throws MemoryBudgetError; the problems of the loads (a card the solver cannot apply) are reported to `sink` when
    // the basis vectors are made (basisVector), as LoadVectorBuilder reports them.
    StaticLoadCases(Inputs inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink);

    // What one basis vector is, for the load basis handed over: its
    // family, the basis kind its family took, and its source with its deck name: the ID of a load set as written
    // (source LoadSet), the g-set DOF of a unit load (LoadedRow), or the label of the subcase whose family part it is
    // (Case: a case of the Cases basis, a pivot of the Rank basis, or a case that joined the basis).
    struct VectorName
    {
        assembly::LoadFamily family = assembly::LoadFamily::Concentrated;
        assembly::LoadBasisKind kind = assembly::LoadBasisKind::Empty;
        assembly::BasisVector::Source source = assembly::BasisVector::Source::Case;
        base::EntityId loadSet = 0;
        base::DofIndex dof = -1;
        base::EntityId subcase = 0;
    };

    [[nodiscard]] std::size_t caseCount() const noexcept { return m_inputs.cases.size(); }
    [[nodiscard]] std::size_t basisSize() const noexcept { return m_vectors.size(); }
    [[nodiscard]] assembly::LoadFamily familyOf(std::size_t vector) const { return m_vectors[vector].family; }
    // α (basisSize × caseCount): case c is Σ_i α(i, c) times basis vector i.
    [[nodiscard]] const linalg::DenseMatrix& coefficients() const noexcept { return m_coefficients; }
    // The report of each family that has loads, in the order of ALL_LOAD_FAMILIES.
    [[nodiscard]] const std::vector<assembly::LoadBasisReport>& reports() const noexcept { return m_reports; }
    // Whether some case has a thermal load (the thermal family is not empty).
    [[nodiscard]] bool hasThermalLoads() const noexcept;

    // The name of basis vector `vector`.
    [[nodiscard]] VectorName nameOf(std::size_t vector) const;
    // Fills the g-set loads and enforced displacements of basis vector `vector` (both zeroed first).
    void basisVector(std::size_t vector, std::span<double> loadsG, std::span<double> enforcedG,
                     base::MemoryBudget& budget, base::DiagnosticSink& sink) const;
    // Fills the exact loads and enforced displacements of case `c` (every family), for the deviation check of the deck
    // tolerance.
    void exactCase(std::size_t c, std::span<double> loadsG, std::span<double> enforcedG, base::MemoryBudget& budget,
                   base::DiagnosticSink& sink) const;

    // The description of a family (null when Inputs::keepsFamilies is false or no case has loads of it).
    [[nodiscard]] const FamilyDescription* familyDescription(assembly::LoadFamily family) const noexcept;
    // Fills the g-set loads and enforced displacements of one basis vector named in a family's description (a load set
    // as written, a unit load or enforced displacement on a loaded row, the family part of a case): what basisVector()
    // makes for the vectors of the model's own basis, for the vectors of a unit's (reduction::UnitBasisVector). Both
    // zeroed first.
    void familyVector(assembly::LoadFamily family, const assembly::BasisVector& source, std::span<double> loadsG,
                      std::span<double> enforcedG, base::MemoryBudget& budget, base::DiagnosticSink& sink) const;

private:
    struct Vector
    {
        assembly::LoadFamily family = assembly::LoadFamily::Concentrated;
        assembly::BasisVector source;
    };
    // The compact rows of each family, to map a LoadedRow basis vector back to its DOF.
    struct FamilyRows
    {
        base::BudgetedBuffer<base::DofIndex> dofOfRow;    // concentrated and enforced: the DOF of each compact row
        base::BudgetedBuffer<base::EntityId> setOfColumn; // load sets as written: their IDs, sorted
    };

    void addFamily(assembly::LoadFamily family, base::MemoryBudget& budget);
    void addFamilyCase(assembly::LoadFamily family, std::size_t c, std::span<double> loadsG,
                       std::span<double> enforcedG, base::MemoryBudget& budget, base::DiagnosticSink& sink) const;

    Inputs m_inputs;
    std::vector<Vector> m_vectors;                         // bounded by the cases and their loaded DOFs
    std::vector<FamilyRows> m_rows;                        // per family
    std::vector<linalg::DenseMatrix> m_familyCoefficients; // per family
    std::vector<assembly::LoadBasisReport> m_reports;
    linalg::DenseMatrix m_coefficients;
    std::vector<std::optional<FamilyDescription>> m_descriptions; // per family, with Inputs::keepsFamilies
};

} // namespace oos::solutions
