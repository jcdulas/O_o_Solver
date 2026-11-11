#pragma once

#include "oos/output/sections/CutDefinition.hpp"
#include "oos/output/sections/SectionDefinition.hpp"
#include "oos/output/sections/ZoneDefinition.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace oos::design
{

// The DesignResp1 response types (the RTYPE of the entry): those of a static analysis, and the eigenvalue types of the
// MODES and BUCK analyses of SOL 200: EIGN (the eigenvalue λ = ω² of a normal mode, in (radians/time)²), FREQ (its
// natural frequency √λ / 2π, in cycles/time) and LAMA (the load factor of a buckling mode); the types of the SAERO
// analysis: TRIM (the value of a trim variable, its
// AeroRigidTrimVariable or AeroControlSurface in ATTA) and STABDER (an elastic stability derivative coefficient of a
// trim variable, ATTA its AeroRigidTrimVariable or AeroControlSurface, ATTB the restraint flag, 0 unrestrained or 1
// restrained, ATT1 the component 1 to 6: CX, CY, CZ, CMX, CMY, CMZ). The other types of the format (FLUTTER, DIVERG...)
// are refused, naming them.
//
// Compliance to Rotation are the response kinds of the optimisation aids, SectionLoad and CutLoad those of the
// analytical sizing tools, ContourDisplacement and ContourForce those of the external local models: extensions of
// O_o_Solver that no entry of the input format writes. A caller's declaration makes them
// (add_response(kind="compliance"...)), never a deck (ResponseTables refuses their names); to_model() refuses
// them, but for a section load, which it writes as a DesignResp2 of DesignResp1 FORCE items with constant coefficients.
enum class ResponseType : std::uint8_t
{
    Weight,                 // WEIGHT: an entry of the rigid-body mass matrix of the grid point weight generator
    Volume,                 // VOLUME: the volume of the structural elements
    FractionalMass,         // FRMASS: the mass of the selected elements over their mass at the initial design
    Displacement,           // DISP: grid point components
    Stress,                 // STRESS: element stress items
    Strain,                 // STRAIN: element strain items
    Force,                  // FORCE: element force items
    SpcForce,               // SPCFORCE: single-point constraint force components
    StrainEnergy,           // ESE: element strain energy items
    CompositeStress,        // CSTRESS: ply stress items of CompProp/CompPropGlobal elements
    CompositeStrain,        // CSTRAIN: ply strain items
    CompositeFailure,       // CFAILURE: ply failure indices
    CompositeStrengthRatio, // CSTRAT: ply strength ratios
    Compliance,             // the work of the loads on the displacements, Pᵀu, one item per subcase (self-adjoint)
    MeanDisplacement,       // the compliance over the magnitude of the applied forces, one item per subcase
    DeflectionToThickness,  // per shell grid point, |normal displacement| / thickness of the shells around it
    Rotation,               // per grid point, the magnitude of its rotation vector, in radians
    SectionLoad,            // per section and component, its load (output/sections/SectionLoads)
    CutLoad,                // per cut and component, its load (output/sections/CutLoads)
    ContourDisplacement,    // per zone, contour DOF and subcase, its displacement (output/sections/ZoneContour)
    ContourForce,           // per zone, contour DOF and subcase, its contour force (output/sections/ZoneContour)
    Eigenvalue,             // EIGN: the eigenvalue of a normal mode (ATTA) of a MODES subcase
    Frequency,              // FREQ: the natural frequency of a normal mode (ATTA) of a MODES subcase
    BucklingFactor,         // LAMA: the load factor of a buckling mode (ATTA) of a BUCK subcase
    TrimVariable, // TRIM: the value of a trim variable (ATTA its AeroRigidTrimVariable or AeroControlSurface) of a
                  // SAERO subcase
    StabilityDerivative // STABDER: an elastic stability derivative coefficient of a trim variable
};

// The response type of an RTYPE ("Weight", "Displacement"...), or nothing for a type that is not a static DesignResp1
// type. The names of the extension types ("Compliance", "MeanDisplacement", "DeflectionToThickness", "Rotation") are
// found too: the reader of the deck's tables refuses them (isExtensionType).
[[nodiscard]] std::optional<ResponseType> findResponseType(std::string_view rtype) noexcept;
[[nodiscard]] std::string_view responseTypeName(ResponseType type) noexcept;
// Whether the type is a response kind that no entry of the input format has (Compliance ... Rotation, SectionLoad,
// CutLoad).
[[nodiscard]] bool isExtensionType(ResponseType type) noexcept;
// Whether the type is self-adjoint (a response whose adjoint vector is known needs no adjoint solve): its
// displacement weights are the loads of its subcase, so its adjoint vector is the displacement itself (Compliance,
// MeanDisplacement; design/aids/SelfAdjointResponses).
[[nodiscard]] bool isSelfAdjointType(ResponseType type) noexcept;
// Whether a response of this type has one value for the whole model, evaluated once and not per subcase (WEIGHT,
// VOLUME, FRMASS: they take no REGION, and they are invariant under the load).
[[nodiscard]] bool isGlobalType(ResponseType type) noexcept;
// Whether the type reads element results (STRESS ... CSTRAT): its ATTi are elements or properties (PTYPE).
[[nodiscard]] bool isElementType(ResponseType type) noexcept;
// Whether the type is a response of a MODES analysis (EIGN, FREQ), or of an eigenvalue analysis (EIGN, FREQ, LAMA):
// one item per subcase of its analysis, the root of the mode its ATTA names. Field responses instead select MODES
// with DesignResp1Definition::mode.
[[nodiscard]] bool isModalType(ResponseType type) noexcept;
[[nodiscard]] bool isEigenType(ResponseType type) noexcept;
// Whether the type is a response of the SAERO analysis only (TRIM, STABDER): one item per subcase of that analysis.
[[nodiscard]] bool isAeroelasticType(ResponseType type) noexcept;

// The analysis of a subcase (Case Control ANALYSIS = STATICS, MODES, BUCK, SAERO): a response is evaluated in the
// subcases of the analysis its type reads (DesignResponses::analysisOf), a global one once; the types of a
// static analysis are evaluated in the STATICS and SAERO subcases (the trimmed state of a SAERO subcase is a static
// state: a DesignResp1 DISP, STRESS ... of a SOL 200 SAERO subcase reads it).
enum class SubcaseAnalysis : std::uint8_t
{
    Statics,
    Modes,
    Buckling,
    Aeroelastic
};
[[nodiscard]] SubcaseAnalysis analysisOfType(ResponseType type) noexcept;

// A DesignResp1 entry, or a response declared by the caller with the meaning of one:
//
//   | DesignResp1 | ID | LABEL | RTYPE | PTYPE | REGION | ATTA | ATTB | ATT1 |
//   |        | ATT2 | -etc.- |
//
// `propertyType` is the ownerKind: "Elements" when the ATTi are element IDs, a property card ("ShellProp", "RodProp")
// when they are property IDs, empty when blank (grid responses, weight). An empty `targets` means every target (ATTi
// blank or ALL for WEIGHT, VOLUME, FRMASS; the whole model for a caller's response without elements).
struct DesignResp1Definition
{
    std::int32_t id = 0;
    std::string label;
    ResponseType type = ResponseType::Displacement;
    std::string propertyType;
    std::optional<std::int32_t> region;
    std::optional<std::int32_t> attributeA; // ATTA: component digits, item code, weight row
    std::optional<double> attributeB;       // ATTB: weight column, lamina number (a blank one: nothing)
    std::vector<std::int32_t> targets;      // ATTi: grid points, elements or properties
    // The caller's selectors: `componentName` names the items by a column of the result tables ("vonMises", "axial",
    // "o11") in place of ATTA (every item code of each element whose column has that name: Z1 and Z2 of a shell);
    // `plies` the laminae of a composite response in place of ATTB.
    std::string componentName;
    std::vector<std::int32_t> plies;
    // The subcases the response is restricted to (the subcases= of a caller's declaration); empty: as its constraints,
    // its objective or every analysed subcase decide (DesignResponses).
    std::vector<std::int32_t> subcases;
    // SECTLOAD: its sections; CUTLOAD: its cuts (componentName then selects one component, empty: every one).
    std::vector<output::SectionDefinition> sections;
    std::vector<output::CutDefinition> cuts;
    // CONTDISP, CONTFORC: the IDs of its zones (ResponseParts::zones); componentName selects one component of the
    // zone's system ("t1" ... "r3"), empty: every contour component.
    std::vector<std::int32_t> zones;
    // EIGN, FREQ, LAMA: ATTA is the mode number, ATTB the approximation code (blank, 1 or 2); `tracksMode`
    // (track=True of a caller's declaration, an extension of O_o_Solver) follows the mode
    // from one design point to the next by its shape (ModeTracking: the modal assurance criterion against the shape the
    // reference point gave), instead of taking the ATTA-th root of each point.
    bool tracksMode = false;
    // A field of one vibration mode (Displacement, Stress, Strain, Force, CompositeStress or CompositeStrain), from 1.
    // With tracksMode, this mode is matched against the caller's reference shape.
    std::optional<std::int32_t> mode;
};

// What an argument of an equation response refers to (the flags of DesignResp2 and DesignResp3, in the order the
// entries list their argument groups).
enum class ArgumentKind : std::uint8_t
{
    DesignVariable, // "DesignVar" DVIDi: the value of the variable
    TableConstant,  // "DesignConstants" LABLi: a constant of the design model
    DesignResp1,    // "DesignResp1" NRk: every item of the response, in its item order (a positional argument each)
    Node,           // "DesignNodes" Gm Cm: the basic coordinate Cm (1 to 3) of grid point Gm
    DesignPropRelation1, // "DesignPropRelation1" ... "DesignMatRelation2": the value of the designed field of that
                         // relation
    DesignElemRelation1,
    DesignMatRelation1,
    DesignPropRelation2,
    DesignElemRelation2,
    DesignMatRelation2,
    DesignResp2 // "DesignResp2" NRRk: another equation response (its value in the same subcase)
};

[[nodiscard]] std::optional<ArgumentKind> findArgumentKind(std::string_view flag) noexcept;
[[nodiscard]] std::string_view argumentKindName(ArgumentKind kind) noexcept;

// One argument of an equation response. `id` is the DESVAR, DesignResp1, relation or DesignResp2 ID, or the grid point
// of a DNODE (with its `component`); `label` the DesignConstants label. `name` is the keyword the argument is given
// under to a caller's callable (empty for a deck DesignResp3: the arguments are then passed by kind, DESVAR,
// DesignConstants, DesignResp1...).
struct ResponseArgument
{
    ArgumentKind kind = ArgumentKind::DesignVariable;
    std::int32_t id = 0;
    std::int32_t component = 0;
    std::string label;
    std::string name;
    // A design variable given per label of a by-label equation (add_equation_response(by="label")): the variable ID of
    // each label entity; empty: `id` for every label.
    std::vector<std::pair<std::int32_t, std::int32_t>> variableByLabel;
};

// How an equation response computes its value.
enum class EquationForm : std::uint8_t
{
    Expression, // a DesignEquation program (DRESP2, add_equation_response(expression=)): exact derivatives
    Callable    // a caller's callable by batch (DRESP3 by GROUP and TYPE, add_equation_response(function=))
};

// A DesignResp2 or DesignResp3 entry, or an equation response declared by the caller:
//
//   | DesignResp2 | ID | LABEL | EQID | REGION | ...  then the argument lists "DESVAR", "DesignConstants",
//   "DesignResp1", "DNODE", | DesignResp3 | ID | LABEL | GROUP | TYPE | REGION |   "DesignPropRelation1" ...
//   "DesignResp2" (in that order)
//
// `callable` names the registered callable of a DesignResp3 ("GROUP/TYPE") or of a caller's declaration. `byLabel`
// (add_equation_response(by="label")) gives the response one item per label of its response arguments (an entity and
// a subcase), the arguments matched by label; otherwise the rules of the cards apply: the response has one value per
// subcase (or one for the model when every argument is global), its DesignResp1 arguments expanded into one positional
// argument per item.
struct EquationResponseDefinition
{
    std::int32_t id = 0;
    std::string label;
    std::string card = "DesignResp2"; // the card type: "DesignResp2" or "DesignResp3"
    EquationForm form = EquationForm::Expression;
    std::int64_t equationId = 0; // the DesignEquation program of an Expression
    // Header arguments of an Expression bound by name (constants= of a caller's declaration); the arguments fill the
    // other header arguments in order. A deck DesignResp2 binds none: its DesignConstants constants are positional
    // arguments.
    std::vector<std::pair<std::string, double>> constants;
    std::string callable; // the registered callable of a Callable
    std::optional<std::int32_t> region;
    std::vector<ResponseArgument> arguments;
    bool byLabel = false;
    std::vector<std::int32_t> subcases; // as DesignResp1Definition::subcases
};

// What an argument of an external response is (an external local model receives the arguments it needs: contour
// displacements or forces, section loads, zone loads, variables): the contour data of its zone (one row per
// contour DOF of the zone, its size the zone's), the loads of the interior grid points of its zone, or a design
// variable (one value; per zone with `variableByLabel`). A variable that drives no field of the model is allowed: it
// enters only the responses that read it (a variable of the local model only).
enum class ExternalArgumentKind : std::uint8_t
{
    ContourDisplacement,
    ContourForce,
    ZoneLoads,
    DesignVariable
};

struct ExternalArgument
{
    ExternalArgumentKind kind = ExternalArgumentKind::ContourDisplacement;
    std::string name; // the keyword the plugin gets it under
    std::int32_t variable = 0;
    std::vector<std::pair<std::int32_t, std::int32_t>> variableByLabel; // (zone, DESVAR) per label
};

// How the values of an external response are given (one-phase or two-phase coupling): by its callable, called
// during the evaluation by label and batch, or supplied by value by the program (problem.supply) for the design point.
enum class ExternalCoupling : std::uint8_t
{
    Callback,
    Supplied
};

// An external response (an external local model, a plugin of the caller): the outputs of a local model of each
// of its zones, one item per zone (its label: the item's entity), subcase and output, computed by a plugin from the
// contour data of the zone, its loads and design variables. `isLinear` (linear=True): the outputs are linear in the
// contour data and the loads of the zone, so the plugin is called on the load basis and its outputs combined with α;
// `hasPost` then: post= computes the outputs of each case from its combined linear outputs (von Mises, a reserve
// factor). `hasSelection` (select=): the evaluator gives, per zone, the subcases to evaluate; the others are
// not_evaluated. `outputs`: per zone (in the order of `zones`) the number of outputs when declared, 0 when it is the
// plugin's first answer. `evaluator`: its name in the ExternalRegistry.
struct ExternalResponseDefinition
{
    std::int32_t id = 0;
    std::string label;
    std::vector<std::int32_t> zones;
    std::vector<ExternalArgument> arguments;
    bool isLinear = false;
    bool hasPost = false;
    bool hasSelection = false;
    ExternalCoupling coupling = ExternalCoupling::Callback;
    std::string evaluator;
    std::vector<std::int32_t> outputs;
    std::optional<std::int32_t> region;
    std::vector<std::int32_t> subcases; // as DesignResp1Definition::subcases
};

// The responses of a design problem, as read (ResponseTables) or declared (the caller's DesignProblem): plain data,
// checked when DesignResponses is made of them. `zones` are the zones of local models the contour and external
// responses read, `zoneSubcases` the constraints of the subcases that decided their contours.
struct ResponseParts
{
    std::vector<DesignResp1Definition> designResp1;
    std::vector<EquationResponseDefinition> equations;
    std::vector<ExternalResponseDefinition> externals;
    std::vector<output::ZoneDefinition> zones;
    std::vector<output::ZoneSubcaseConstraint> zoneSubcases;
};

} // namespace oos::design
