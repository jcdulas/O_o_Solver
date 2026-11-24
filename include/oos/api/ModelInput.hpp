#pragma once

#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/input/snapshot/SnapshotBuilder.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// The simple entry points of the solver: what the WebAssembly module (wasm/) offers to JavaScript, in plain C++ so that
// they also build and run natively. They take plain types (byte vectors, numbers, texts) and hide the snapshot, the
// solution options and the memory budget.
namespace oos::api
{

// A real column added after the columns of a table, one value per row (NaN for none): what the caller computes for a
// table of O_o_Model that the solver reads with one more column (the neutral axes n1, n2 of sections_table).
struct AddedColumn
{
    std::string name;
    std::vector<double> values;
};

struct SourceCard
{
    std::string card;
    std::int64_t count = 0;
};

// The model of one analysis, gathered table by table: the tables of O_o_Model as table packages (the bytes of its
// QueryTable.toPackage(), which keep the exact Arrow layout), the rotations of the displacement systems and the version
// counters of the model. build() checks every table against what the solver reads and makes the snapshot the analyses
// run on.
//
// The names are those of the snapshot: the query tables under their own names ("nodes", "singlePointConstrTable",
// "subcasesTable", ...; see input/contract/TableRole.cpp), and the card tables, by their neutral card types
// ("ShellQuad4"), through their own methods.
//
// Thread-safety: none (one object per model, on one thread). Ownership: owns the tables until build().
class ModelInput
{
public:
    ModelInput() = default;

    void setVersions(std::int64_t structureVersion, std::int64_t contentVersion);

    // All source card types and counts, including data not present in the exported tables.
    void setSourceCards(std::vector<SourceCard> cards);

    // A table of a snapshot role ("nodes", "singlePointConstrTable", ...), with optional added columns. Throws
    // ContractError for a name that is not a role or bytes that are not a table package.
    void setTable(const std::string& name, std::vector<std::uint8_t> package, std::vector<AddedColumn> added = {});

    // The card tables, by neutral card type: the elements of one card type ("ShellQuad4"), the properties
    // ("ShellProp"), the eigenvalue methods ("LanczosEigenMethod", with the column of their ID), the material cards the
    // shells and solids read field by field ("OrthoMat2D"), the load cards with their field formats
    // ("PointForceLoad"), the aeroelastic cards ("AeroBeamSpline2", with their ID column).
    void addElementTable(const std::string& card, std::vector<std::uint8_t> package);
    void addPropertyTable(const std::string& card, std::vector<std::uint8_t> package);
    void addMethodTable(const std::string& card, const std::string& idColumn, std::vector<std::uint8_t> package);
    void addMaterialTable(const std::string& card, std::vector<std::uint8_t> package);
    void addLoadCardTable(const std::string& card, std::vector<std::uint8_t> package);
    void addAeroCardTable(const std::string& card, const std::string& idColumn, std::vector<std::uint8_t> package);

    // The rotation of each grid point's displacement system to the basic system, 9 values per grid point in the rows
    // of the "nodes" table (O_o_Model's displacementRotations).
    void setDisplacementRotations(std::vector<double> rotations);

    // ModelParams OffsetDefinition: "ElemOffsetsOnly" (the default), "LoadRecoveryOffsets" or "ElemOffsetZ"; another
    // value is reported by the support check.
    void setOffsetDefinition(const std::string& value);

    // The formulation of every element of one card type ("ShellQuad4", "ClassicFormulation"; elementFormulations()
    // lists them), before the model's <Card>Formulation parameter; a second call for the same card replaces the first.
    // The card is the neutral card type exactly as elementFormulations() names it: a deck keyword ("ShellQuad4") is
    // refused, so that a program names a card one way only. A formulation the card does not have is reported by the
    // support check. Throws std::invalid_argument for an empty name or a card that is not an element card type of the
    // solver.
    void setElementFormulation(const std::string& card, const std::string& formulation);

    // The formulation of the whole model ("ClassicFormulation", "MixedHybridFormulation"), before the model's
    // ElementFormulation parameter. Under MixedHybridFormulation every shell, line and solid card takes it together
    // (docs/mixed-hybrid-formulation-spec.md section 4.5). Throws std::invalid_argument for a name the solver does not
    // know.
    void setModelFormulation(const std::string& formulation);

    // The geometry declarations of the mixed hybrid formulation (section 4.4 of the specification), as flat lists of
    // IDs. Mandatory for a model with shells or solids in that formulation; empty lists say that the model's
    // ShellNormalAngle decides every edge. Throws std::invalid_argument for a list whose length is not a multiple of
    // its group, an edge of one grid point or a mesh line that names a grid point twice.
    struct FormulationGeometryLists
    {
        // Three values per edge: its two grid point IDs, then the shell element whose side is meant (0: every side).
        std::vector<std::int32_t> smoothEdges;
        std::vector<std::int32_t> sharpEdges;
        // Two grid point IDs per edge.
        std::vector<std::int32_t> rawEdges;
        std::vector<std::int32_t> rawElements;
        // Three grid point IDs per continuation: the far end of one edge, the grid point the line goes through, the
        // far end of the edge it goes on into.
        std::vector<std::int32_t> lineContinuations;
        // Two grid point IDs per end: the far end of the edge, then the grid point its line ends at.
        std::vector<std::int32_t> lineEnds;
    };
    void setFormulationGeometry(const FormulationGeometryLists& lists);

    // Checks every table and makes the snapshot; the object is empty afterwards. Throws ContractError naming the
    // table, column, expected type and actual type when a table is not what the solver reads. Messages use the model's
    // neutral card and field names.
    [[nodiscard]] std::shared_ptr<const input::ModelSnapshot> build();

private:
    input::SnapshotBuilder m_builder;
};

} // namespace oos::api
