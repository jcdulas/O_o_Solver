# Static input checks and non-structural mass

## Support checks

`modelFromReader` transfers the complete bulk-card inventory as well as the analysis tables. A card that a query
does not export can therefore still produce an `unsupported-card` refusal. The inventory is retained when an
optimization creates an overlaid model.

`model.checkSupport("statics")` reports unsupported source card types and incompatible selections exported by
`subcasesTable`. Nonlinear static parameters, dynamic loads, frequency sets, time steps and material-temperature
selections are refused by the linear analysis. For example, `CreepMat` and `NonlinearStaticParams` cannot silently
produce an elastic static result. `solver.run` applies the same support check before assembly.
An explicitly selected unsupported `AnalysisKind`, including `NonlinearStatics`, is also refused.

Known damping, plot and unused eigenvalue cards produce explicit warnings. Design declarations are reported
separately as declarations processed by the optimization API. Support for a card type does not validate every field:
the element, constraint and load checks run afterwards. This is not a certification of arbitrary control commands
or parameter combinations outside the transferred contract.

**JavaScript / WebAssembly**

~~~javascript
const model = solver.modelFromReader(oor, deck);
try {
  const report = model.checkSupport("statics");
  if (!report.isComplete) {
    console.error(report.refusals);
  } else {
    const result = solver.run(model, "statics");
  }
} finally {
  model.delete();
}
~~~

**Python**

In Python, run the analysis and inspect its diagnostics.

```python
from oo_solver import model_from_reader

result = model_from_reader(deck).run("statics")
print(result["diagnostics"])
```

**C++20**

```cpp
const auto model = oos::api::modelFromDeck(deck);
const auto report = oos::api::checkSupport(*model, oos::api::Analysis::Statics);
for (const auto& refusal : report.refusals)
    std::cerr << refusal.message << '\n';
if (report.refusals.empty()) {
    const auto result = oos::api::runAnalysis(*model, oos::api::Analysis::Statics, {});
    for (const auto& diagnostic : result.diagnostics)
        std::cout << diagnostic.message << '\n';
}
```

## Selecting added mass

Select a bulk non-structural mass set in Case Control:

~~~text
[case]
NonStructMassSelection;20
SubcaseStart;1
SinglePointConstrSelection;1
StaticLoadSelection;2
DisplacementOutput;All
[model]
# Additional mass per unit length for the elements of RodProp 1.
NonStructMass;20;RodProp;1;2.
~~~

Model exports the effective selection in the `nonStructMass` column of `subcasesTable`. Global selections are
inherited and each subcase can override them. An unselected set contributes nothing. `NonStructMassUnion`
members are expanded by Model; their contributions are added once by the solver.

The full static path and its streamed assembly include selected `NonStructMass` and `NonStructMass1` values
in the mass matrix used by gravity, acceleration, rotational loads and inertia relief. The values add to the
non-structural mass stored in element properties. `Parameter;WeightToMass` scales the assembled element mass.
Line elements take a mass per length; shell and shear-panel elements take a mass per area. Subcases selecting
different sets have separate mass matrices and load bases.

A real eigenvalue analysis (SOL 103) on the full path adds the set its subcase selects to the mass of its
eigenproblem: subcases that select different sets are solved apart, each with its mass, and the modal effective
masses and the grid point weight read that mass. In a buckling analysis (SOL 105) the set weighs in the inertial
loads of the static preload, whose subcase selects it, and so in the load factors.

Unknown selected sets, missing targets, invalid values and incompatible target dimensions stop the run before
solving. Lumped set cards (`NonStructMassLumped`, `NonStructMassLumped1`) are still refused when selected:
their distribution is not implemented. A selected set is refused, naming the subcase, where its mass would be left
out: on the condensed path of every analysis, in a static aeroelastic analysis (SOL 144), in an analysis with
superelements, and in the eigenvalue and buckling analyses of a design problem, whose derivatives do not form the
mass of the set. A static design problem takes it. These restrictions concern selected bulk mass sets; the existing
mass fields in element properties remain available.

The end-to-end checks in `wasm/test/cases/static-inputs.mjs` compare gravity displacements with the independent
axial-rod solution, test inherited and overridden selections, unions, property mass and unit scaling, and verify
that unsupported input stops the calculation. For the eigenvalue analyses they check that a set that doubles or
triples the mass divides the eigenvalues by two or three (a rod, a shell), that a selected set gives the eigenvalue
of the same mass written in the property, and that a set that doubles the weight of a column halves its buckling
load factor.

The [fast combination suite](testing.md) extends these checks across element families, loads, constraints and mass
options, with a watchdog below one second per test.
