#pragma once

#include "oos/api/AeroTransfer.hpp"
#include "oos/api/Analyses.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"

namespace oos::api
{
class AeroCoupling;

class AeroStaticPreload
{
    friend class AeroCoupling;
    std::shared_ptr<const input::ModelSnapshot> snapshot;
    std::shared_ptr<base::MemoryBudget> budget;
    std::shared_ptr<const elements::SnapshotIndex> index;
    base::BudgetedBuffer<double> displacement;
    base::EntityId subcase = 0;
};

class AeroModalBasis
{
public:
    [[nodiscard]] std::vector<double> generalizedLoads(std::span<const double> boxForces) const;

private:
    friend class AeroCoupling;
    explicit AeroModalBasis(std::shared_ptr<const AeroTransfer> geometry) : transfer(std::move(geometry)) {}
    std::shared_ptr<const AeroTransfer> transfer;
    std::vector<std::vector<double>> attachmentModes;
};

struct AeroCoupledResult
{
    AnalysisOutput analysis;
    // Static motion or one field per mode, in the order of the eigenvalue table.
    std::vector<aero::BoxMotion> boxMotions;
    std::shared_ptr<AeroStaticPreload> preload;
    std::shared_ptr<AeroModalBasis> modalBasis;
};

// All structural analyses and transfers use the same immutable model and spline attachments.
class AeroCoupling
{
public:
    explicit AeroCoupling(std::shared_ptr<const input::ModelSnapshot> snapshot);
    [[nodiscard]] auto nodeIds() const { return m_transfer->nodeIds(); }
    [[nodiscard]] auto boxIds() const { return m_transfer->boxIds(); }
    [[nodiscard]] auto boxMotion(std::span<const double> motion) const { return m_transfer->boxMotion(motion); }
    [[nodiscard]] auto structuralLoads(std::span<const double> forces) const { return m_transfer->structuralLoads(forces); }

    [[nodiscard]] AeroCoupledResult solveStatic(base::EntityId subcase, std::span<const double> boxForces) const;
    [[nodiscard]] AeroCoupledResult solveModes(base::EntityId subcase) const;
    [[nodiscard]] AeroCoupledResult solveBuckling(base::EntityId subcase, const AeroStaticPreload& preload) const;

private:
    struct Attachment { base::DofIndex start; base::Mat3 cdToBasic; };
    [[nodiscard]] std::vector<double> attachmentMotion(std::span<const double> field) const;
    std::shared_ptr<const input::ModelSnapshot> m_snapshot;
    std::shared_ptr<const AeroTransfer> m_transfer;
    std::vector<Attachment> m_attachments;
};
} // namespace oos::api
