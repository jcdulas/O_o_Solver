#pragma once

namespace oos::solutions
{

// The parts of the steps of an analysis before its solves, in seconds, summed over its constraint sets: what a new
// design point builds again and what the reuse of the analysis structure between design points saves.
// StaticTimings::setup, assembly and factorisation hold them with the rest of their stages.
struct AnalysisPartTimings
{
    double checks = 0.0;        // the support check, the snapshot index, the element check, the subcases' plans
    double connectivity = 0.0;  // the DOF map, the element matrices of the connectivity pass and their node groups
    double constraints = 0.0;   // the MPCs, fasteners, SPCs, glued contact, Gm and the DOF sets
    double graph = 0.0;         // the m-set couplings, the node graph and the sparse pattern
    double stiffness = 0.0;     // Kgg (from the kept element matrices) and Knn (the m set eliminated)
    double singularities = 0.0; // the grid point singularity processor (AUTOSPC) and the solved sets
    double ordering = 0.0;      // the fill-reducing ordering of the solved set's nodes
    double symbolic = 0.0;      // the symbolic factor and the numbering of the equations
    double conversion = 0.0;    // Knn to the SymmetricCsc of the equations
    double numeric = 0.0;       // the numeric factorisation and its solve plan

    AnalysisPartTimings& operator+=(const AnalysisPartTimings& other) noexcept
    {
        checks += other.checks;
        connectivity += other.connectivity;
        constraints += other.constraints;
        graph += other.graph;
        stiffness += other.stiffness;
        singularities += other.singularities;
        ordering += other.ordering;
        symbolic += other.symbolic;
        conversion += other.conversion;
        numeric += other.numeric;
        return *this;
    }
};

} // namespace oos::solutions
