#pragma once

#include <array>
#include <cmath>
#include <span>
#include <string>
#include <vector>

namespace oos::design
{

// The section of a BarShapeProp or of one station of a BeamShapeProp: its type in the beam library ("ISection", "Tube",
// "Hat"
// ...) and its dimensions DIM1 ... DIMn.
struct SectionShape
{
    std::string type;
    std::vector<double> dimensions;
    // The dimensions whose meshes the finite element properties are computed on (O_o_Reader's reference dimensions: the
    // properties smooth in the dimensions around it); empty: the section's own (the bits of O_o_Reader's
    // sectionsTable, which reference = dimensions gives too).
    std::vector<double> reference;
};

// The properties of a section as the elements read them from the sections table (input::SectionTable: O_o_Reader's
// sectionsTable): the area, the inertias I1, I2, I12, the torsion constant J, the shear factors K1, K2, the stress
// recovery points C, D, E, F as (y, z) pairs (c1 c2 d1 d2 e1 e2 f1 f2), and the neutral axis from the shear centre
// (n1, n2: O_o_Reader's centroid − shear_center, the columns O_o_Solver's snapshot adds for a BeamShapeProp). NaN
// everywhere for an impossible section.
struct SectionValues
{
    double area = 0.0;
    double i1 = 0.0;
    double i2 = 0.0;
    double i12 = 0.0;
    double j = 0.0;
    double k1 = 0.0;
    double k2 = 0.0;
    std::array<double, 8> points{};
    std::array<double, 2> neutralAxis{}; // as the library gives it, before neutralAxisOffset()
    // With the derivatives asked: d(value)/d(DIMk) of the SECTION_VALUE_COUNT values in the order area, i1, i2, i12, j,
    // k1, k2, c1 ... f2, n1, n2, value-major (gradient[v · dimensions + k]); empty otherwise.
    std::vector<double> gradient;
};

// The values a SectionValues holds, in the order of its gradient.
inline constexpr std::size_t SECTION_VALUE_COUNT = 17;

// A component of the neutral axis' offset from the shear centre within this share of √A is taken as 0: the finite
// element shear centre of a doubly or point-symmetric section (BOX, I of equal flanges, Z, BAR, H...) departs from its
// centroid by round-off, up to 1.2e-11·√A measured on O_o_Reader's sections (the barrels' Z frames), where a section
// whose shear centre is off its centroid has offsets of 0.1 to 1·√A (a T: 0.58). So such a section keeps its axis on
// its centroid to the bit. The same rule in O_o_Solver's snapshot (snapshot.SHEAR_CENTRE_ON_CENTROID).
inline constexpr double SHEAR_CENTRE_ON_CENTROID = 1.0e-8;

// The offset a station takes for the component `offset` of its neutral axis (centroid − shear centre) and its `area`.
[[nodiscard]] inline double neutralAxisOffset(double offset, double area)
{
    return std::abs(offset) <= SHEAR_CENTRE_ON_CENTROID * std::sqrt(area) ? 0.0 : offset;
}

// The section library a design point derives the section properties of its designed BarShapeProp and BeamShapeProp from
// (library sections as design targets): the formulas of O_o_Reader's sectionsTable, so that a design point gives the
// bits of the same cards edited by model.design.apply (the shear factors of most sections come from a finite element
// model of the section there, which O_o_Solver does not compute). The caller implements it with O_o_Reader's section
// properties; a C++ test gives the closed forms of its sections.
//
// Thread-safety: called by batch from the thread that makes a design point or its perturbations (the caller's
// implementation may run code that is not thread-safe); the implementation must be safe for that. Ownership: none.
class SectionLibrary
{
public:
    virtual ~SectionLibrary() = default;

    // One SectionValues per shape, in order, with their derivatives with respect to the dimensions when `gradient`
    // (O_o_Reader's section_properties_batch: exact on the meshes of the reference for the finite element properties).
    // Throws std::invalid_argument for a type the library does not know.
    [[nodiscard]] virtual std::vector<SectionValues> properties(std::span<const SectionShape> shapes,
                                                                bool gradient = false) const = 0;
};

} // namespace oos::design
