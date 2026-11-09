#include "oos/elements/springs/bushing/BushingElement.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementReferences.hpp"
#include "oos/elements/core/MatrixKernels.hpp"

#include <optional>
#include <string>

namespace oos::elements
{

namespace
{

// O_o_Reader gives a blank CID or OCID as −1 (CBUSH: OCID default −1; CID blank means "from the geometry").
constexpr std::int32_t BLANK_SYSTEM = -1;

// Writes the 3 × 3 block `block` at (3·blockRow, 3·blockColumn) of H.
void setBlock(base::MatN<6, 12>& h, std::size_t blockRow, std::size_t blockColumn, const base::Mat3& block)
{
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            h(3 * blockRow + i, 3 * blockColumn + j) = block(i, j);
}

} // namespace

BushingElement::Columns::Columns(const ElementBatchView& view)
    : ga(view.columns().values<std::int32_t>("nodeA")),
      gb(view.columns().values<std::int32_t>("nodeB")),
      go(view.columns().values<std::int32_t>("orientationNode", base::NullPolicy::Allow)),
      x(view.columns().fixedLists<double>("orientation", base::NullPolicy::Allow)),
      isXGiven(view.columns().validity("orientation")),
      frameId(view.columns().values<std::int32_t>("frameId")),
      s(view.columns().values<double>("offsetFraction")),
      offsetFrameId(view.columns().values<std::int32_t>("offsetFrameId")),
      si(view.columns().fixedLists<double>("offset", base::NullPolicy::Allow)),
      properties(view, "BushProp")
{
    if (properties.table == nullptr)
        return;
    const base::ContractReader& bushProp = properties.table->reader();
    k = bushProp.fixedLists<double>("stiffness");
    rcv = bushProp.fixedLists<double>("recovery");
    mass = bushProp.values<double>("mass");
    if (bushProp.hasColumn("thermal"))
        thermal = bushProp.fixedLists<double>("thermal");
}

Resolution<BushingElement::Bush> BushingElement::resolve(const ElementBatchView& view, const Columns& columns,
                                                           std::int64_t row)
{
    using Result = Resolution<Bush>;
    const auto r = static_cast<std::size_t>(row);
    Bush bush;
    Resolution<std::int32_t> rowA = resolveGridRow(view, row, columns.ga[r], "nodeA");
    if (!rowA)
        return Result::failure(std::move(rowA.problem));
    bush.rowA = *rowA.value;
    bush.isGrounded = columns.gb[r] == 0;
    if (!bush.isGrounded)
    {
        Resolution<std::int32_t> rowB = resolveGridRow(view, row, columns.gb[r], "nodeB");
        if (!rowB)
            return Result::failure(std::move(rowB.problem));
        bush.rowB = *rowB.value;
    }
    Resolution<PropertyRef> property = resolveProperty(view, columns.properties, row);
    if (!property)
        return Result::failure(std::move(property.problem));
    const std::span<const double> k = columns.k[property.value->row];
    for (std::size_t c = 0; c < 6; ++c)
        bush.stiffness[c] = k[c];
    const std::span<const double> coefficients = columns.rcv[property.value->row];
    for (std::size_t c = 0; c < 4; ++c)
        bush.recovery[c] = coefficients[c];
    bush.mass = columns.mass[static_cast<std::size_t>(property.value->row)];
    // BushProp "T" line: ALPHA (the thermal expansion) and COINL (the length of a Bushing whose grid points coincide)
    // are not implemented; TREF alone has no effect.
    if (columns.thermal.size() > 0)
    {
        const std::span<const double> thermal = columns.thermal[property.value->row];
        if (thermal[0] != 0.0 || thermal[2] != 0.0)
            return Result::failure(elementProblem(
                view, row, problem_code::UNSUPPORTED_FIELD, "propId",
                base::concatenate(
                    {"BushProp ", std::to_string(columns.properties.propIds[r]),
                     ": expected the T line blank (BushProp.alphaCoefficient and "
                     "BushProp.coincidentLength "
                     "0), found BushProp.alphaCoefficient ",
                     std::to_string(thermal[0]), " and BushProp.coincidentLength ", std::to_string(thermal[2]),
                     "; the thermal expansion and the coincidence length of a Bushing are not supported: "
                     "leave them blank"})));
    }

    const base::Vec3 positionA = view.positionInBasic(bush.rowA);
    const base::Vec3 positionB = bush.isGrounded ? positionA : view.positionInBasic(bush.rowB);
    const base::Mat3 frameA = view.displacementFrame(bush.rowA);
    const base::Mat3 frameB = bush.isGrounded ? frameA : view.displacementFrame(bush.rowB);

    // Element axes E (columns in the basic system).
    base::Mat3 axes;
    if (columns.frameId[r] != BLANK_SYSTEM)
    {
        Resolution<input::CoordinateFrame> frame = resolveRectangularFrame(view, row, columns.frameId[r], "frameId");
        if (!frame)
            return Result::failure(std::move(frame.problem));
        axes = frame.value->axes;
    }
    else
    {
        const base::Vec3 span = positionB - positionA;
        const double length = base::norm(span);
        if (bush.isGrounded || length < BUSH_COINCIDENT_DISTANCE)
            return Result::failure(
                elementProblem(view, row, problem_code::INVALID_FIELD, "frameId",
                               base::concatenate({"expected a Bushing.frameId, found it blank with ",
                                                  bush.isGrounded ? std::string("Bushing.nodeB blank")
                                                                  : std::string("Bushing.nodeA and Bushing.nodeB ")
                                                                        .append(std::to_string(length))
                                                                        .append(" apart"),
                                                  " (Bushing: Bushing.frameId must be given when Bushing.nodeB is "
                                                  "blank or Bushing.nodeA and "
                                                  "Bushing.nodeB coincide)"})));
        const base::Vec3 xAxis = span / length;
        base::Vec3 orientation;
        const char* orientationField = "orientation1";
        if (columns.go[r] > 0)
        {
            orientationField = "orientationNode";
            Resolution<std::int32_t> rowO = resolveGridRow(view, row, columns.go[r], "orientationNode");
            if (!rowO)
                return Result::failure(std::move(rowO.problem));
            orientation = view.positionInBasic(*rowO.value) - positionA;
        }
        else if (columns.isXGiven.isValid(row))
        {
            const std::span<const double> components = columns.x[row];
            // Bushing: X1-X3 are in the displacement system of GA.
            orientation = frameA * base::Vec3{components[0], components[1], components[2]};
        }
        const base::Vec3 zDirection = base::cross(xAxis, orientation);
        const double orientationLength = base::norm(orientation);
        if (orientationLength == 0.0)
            return Result::failure(elementProblem(
                view, row, problem_code::UNSUPPORTED_FIELD, "orientation1",
                "expected an orientation (Bushing.orientationNode, or "
                "Bushing.orientation1-Bushing.orientation3 not all "
                "zero) or a Bushing.frameId, found none; a Bushing without them is not supported yet: give "
                "Bushing.orientationNode, Bushing.orientation1-Bushing.orientation3 or "
                "Bushing.frameId"));
        if (base::norm(zDirection) <= BUSH_PARALLEL_ORIENTATION_FACTOR * orientationLength)
            return Result::failure(
                elementProblem(view, row, problem_code::DEGENERATE_GEOMETRY, orientationField,
                               "expected an orientation vector not parallel to Bushing.nodeA-Bushing.nodeB, found a "
                               "parallel one; "
                               "the element y and z axes are undefined: choose another Bushing.orientationNode or "
                               "Bushing.orientation1-Bushing.orientation3"));
        const base::Vec3 zAxis = base::normalized(zDirection);
        const base::Vec3 yAxis = base::cross(zAxis, xAxis);
        axes = base::matrixFromRows(xAxis, yAxis, zAxis).transposed();
    }

    // Spring point p.
    base::Vec3 springPoint;
    if (columns.offsetFrameId[r] == BLANK_SYSTEM)
        springPoint = bush.isGrounded ? positionA : positionA + columns.s[r] * (positionB - positionA);
    else
    {
        Resolution<input::CoordinateFrame> frame =
            resolveRectangularFrame(view, row, columns.offsetFrameId[r], "offsetFrameId");
        if (!frame)
            return Result::failure(std::move(frame.problem));
        // Bushing S1-S3: the components of the spring-damper offset in OCID, measured from GA (a vector, not a point
        // of OCID: the origin of OCID plays no part). Checked against the same bush at its offset point tied by
        // RigidBar.
        const std::span<const double> offset = columns.si[row];
        springPoint = positionA + frame.value->axes * base::Vec3{offset[0], offset[1], offset[2]};
    }

    // H: (Δu, Δθ) in the element system from the DOFs in the displacement systems, u_basic = R·u_CD.
    const base::Mat3 toElement = axes.transposed();
    const base::Mat3 translationA = toElement * frameA;
    const base::Mat3 armA = toElement * crossProductMatrix(springPoint - positionA) * frameA;
    setBlock(bush.relativeMotion, 0, 0, -1.0 * translationA); // −u_A
    setBlock(bush.relativeMotion, 0, 1, armA);                // −θ_A × (p − x_A) = [p − x_A]×·θ_A
    setBlock(bush.relativeMotion, 1, 1, -1.0 * translationA); // −θ_A
    if (!bush.isGrounded)
    {
        const base::Mat3 translationB = toElement * frameB;
        const base::Mat3 armB = toElement * crossProductMatrix(springPoint - positionB) * frameB;
        setBlock(bush.relativeMotion, 0, 2, translationB); // u_B
        setBlock(bush.relativeMotion, 0, 3, -1.0 * armB);  // θ_B × (p − x_B) = −[p − x_B]×·θ_B
        setBlock(bush.relativeMotion, 1, 3, translationB); // θ_B
    }
    return Result::success(bush);
}

void BushingElement::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

} // namespace oos::elements
