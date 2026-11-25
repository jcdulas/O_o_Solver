import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { oor, solver, test } from "./support.mjs";

const wing = readFileSync(new URL("../fixtures/trim-rbe3-wing.oos", import.meta.url), "utf8");
const surface = "AeroSurfaceSpline1;60;100;100;103;50";
const beam = "AeroBeamSpline2;60;100;100;103;50;0.01;1.2;0;0.01;0.02;Both";

// Prepare once and drop both source handles before the first transfer.
function prepare(text, options = {}) {
  oor.writeFile("/work/aero-transfer.oos", text);
  const deck = oor.readModel("/work/aero-transfer.oos", { strict: true });
  try {
    const model = solver.modelFromReader(oor, deck, options);
    try { return model.aeroTransfer(); } finally { model.delete(); }
  } finally { deck.delete(); }
}

function near(a, b) {
  assert.ok(Math.abs(a - b) < 1e-10 * Math.max(1, Math.abs(b)), a + " differs from " + b);
}

for (const [kind, card] of [["surface", surface], ["beam", beam]]) {
  test("direct " + kind + " spline transfer preserves virtual work", () => {
    const transfer = prepare(wing.replace(surface, card));
    let loads;
    try {
      assert.deepEqual(Array.from(transfer.boxIds), [100, 101, 102, 103]);
      assert.deepEqual(Array.from(transfer.nodeIds), [1, 2, 3, 6]);
      const u = Float64Array.from({ length: 24 }, (_, i) => Math.sin(i + 1));
      const forces = Float64Array.of(1, -2, 3, 5);
      const { plunge, slope } = transfer.boxMotion(u);
      loads = transfer.structuralLoads(forces);
      assert.equal(loads.length, 24);
      // Each box chord is 0.5. Its quarter-chord force acts 0.125 upstream of its centre.
      const aeroWork = forces.reduce((sum, f, i) => sum + f * (plunge[i] - 0.125 * slope[i]), 0);
      near(loads.reduce((sum, load, i) => sum + load * u[i], 0), aeroWork);
      near(loads.filter((_, i) => i % 6 === 2).reduce((sum, f) => sum + f), 7);
      if (kind === "beam") assert.ok(loads.some((v, i) => i % 6 >= 3 && Math.abs(v) > 1e-5));
      const constant = Float64Array.from(u, (_, i) => i % 6 === 2 ? 2 : 0);
      const rigid = transfer.boxMotion(constant);
      rigid.plunge.forEach(z => near(z, 2));
      rigid.slope.forEach(s => near(s, 0));
      assert.throws(() => transfer.boxMotion([0]), /nodal motion/);
      assert.throws(() => transfer.structuralLoads([1, 2]), /box forces/);
      assert.throws(() => transfer.structuralLoads([0, 0, NaN, 0]), /finite/);
    } finally { transfer.delete(); }
    assert.ok(loads.every(Number.isFinite));
  });
}

test("shared attachment loads accumulate and directional spline usage is preserved", () => {
  const a = "AeroSurfaceSpline1;60;100;100;101;50", b = "AeroSurfaceSpline1;61;100;102;103;50";
  const reference = prepare(wing);
  const split = prepare(wing.replace(surface, a + "\n" + b));
  const oneWay = prepare(wing.replace(surface, a + ";;;Force\n" + b + ";;;Displacement"));
  try {
    const f = [3, -7, 9, 2], u = Float64Array.from({length: 24}, (_, i) => Math.cos(i));
    const total = reference.structuralLoads(f), first = reference.structuralLoads([3, -7, 0, 0]);
    split.structuralLoads(f).forEach((v, i) => near(v, total[i]));
    oneWay.structuralLoads(f).forEach((v, i) => near(v, first[i]));
    const full = reference.boxMotion(u), partial = oneWay.boxMotion(u);
    for (const field of ["plunge", "slope"])
      partial[field].forEach((v, i) => near(v, i < 2 ? 0 : full[field][i]));
  } finally { oneWay.delete(); split.delete(); reference.delete(); }
});

function structuralRun(text, kind, subcase) {
  oor.writeFile("/work/structural-reference.oos", text);
  const deck = oor.readModel("/work/structural-reference.oos", { strict: true });
  try {
    const model = solver.modelFromReader(oor, deck);
    try { return solver.run(model, kind, { subcases: [subcase] }); }
    finally { model.delete(); }
  } finally { deck.delete(); }
}

test("coupled SOL 101, 103 and 105 agree with an independent structural deck", () => {
  const base = wing.replace("TrimSelection;5", "StaticLoadSelection;10")
    .replace("[case]", "[case]\nDisplacementOutput;All")
    + "\nPointForceLoad;10;3;;0.2;0.;0.;-1.\n";
  const t = prepare(base), other = prepare(base);
  const forces = [1, -2, 3, 5];
  const reference = base.split("\n").filter(line => !line.startsWith("Aero")).join("\n")
    .replace("SolutionSequence;144", "SolutionSequence;101");
  const loads = t.structuralLoads(forces);
  let loaded = reference;
  for (const [i, id] of t.nodeIds.entries())
    for (const [offset, card] of [[0, "PointForceLoad"], [3, "PointMomentLoad"]]) {
      const values = loads.slice(6*i+offset, 6*i+offset+3);
      if (values.some(v => v !== 0))
        loaded += [card, 10, id, "", "1.", ...values].join(";") + "\n";
    }
  const statics = t.solveStatic(1, forces);
  const modes = t.solveModes(2);
  try {
    const expected = structuralRun(loaded, "statics", 1);
    const displacements = expected.nodalTables.find(table => table.kind === "displacements");
    const computed = statics.nodalTables.find(table => table.kind === "displacements");
    computed.values.forEach((v,i) => near(v, displacements.values[i]));
    const attachments = table => Array.from(t.nodeIds).flatMap(id => {
      const start = 6 * table.ids.indexOf(id);
      assert.ok(start >= 0);
      return Array.from(table.values.slice(start, start + 6));
    });
    const motion = t.boxMotion(attachments(displacements));
    statics.boxMotions[0].plunge.forEach((v,i) => near(v, motion.plunge[i]));
    assert.ok(motion.plunge.some(v => Math.abs(v) > 1e-8));
    const referenceModes = structuralRun(reference, "modes", 2);
    const q = modes.modalBasis.generalizedLoads(forces);
    assert.equal(q.length, 2);
    modes.boxMotions.forEach((m, k) => {
      near(modes.eigenvalues[0].rows[k].cycles, referenceModes.eigenvalues[0].rows[k].cycles);
      // Read the same normalisation and phase from the structural output of the coupled solve.
      const shape = attachments(modes.nodalTables.find(table => table.step === k+1 && table.kind === "eigenvectors"));
      const projected = t.boxMotion(shape);
      m.slope.forEach((v,i) => near(v, projected.slope[i]));
      near(q[k], shape.reduce((sum,v,i) => sum + v*loads[i], 0));
    });
    const buckling = t.solveBuckling(2, statics.preload);
    const expectedBuckling = structuralRun(loaded, "buckling", 2);
    assert.ok(buckling.eigenvalues[0].rows.length > 0);
    buckling.eigenvalues[0].rows.forEach((r,i) => near(r.eigenvalue, expectedBuckling.eigenvalues[0].rows[i].eigenvalue));
    assert.throws(() => other.solveBuckling(2, statics.preload), /different model snapshot/);
    const wrongCase = t.solveStatic(2, forces);
    try {
      assert.throws(() => t.solveBuckling(2, wrongCase.preload), /different static preload subcase/);
    } finally { wrongCase.preload.delete(); }
  } finally {
    modes.modalBasis.delete(); statics.preload.delete(); other.delete(); t.delete();
  }
});

test("rotating the displacement frame of a dependent attachment preserves the physical response", () => {
  const original = prepare(wing);
  const rotated = prepare(wing.replace("Node;6;;0.18;1.75;0.", "Node;6;;0.18;1.75;0.;11")
    + "\nRectFrame2;11;;0.;0.;0.;1.;0.;0.;0.;1.;0.\n");
  const f = [0.5, 1.2, -0.8, 2.1], a = original.solveStatic(1, f), b = rotated.solveStatic(1, f);
  try {
    const ma = a.boxMotions[0], mb = b.boxMotions[0];
    mb.plunge.forEach((v,i) => near(v, ma.plunge[i]));
    mb.slope.forEach((v,i) => near(v, ma.slope[i]));
  } finally { a.preload.delete(); b.preload.delete(); rotated.delete(); original.delete(); }
});

test("mixed Auto buckling uses the complete static solution and scales inversely with its load", () => {
  const t = prepare(wing + "\nParameter;MixedHybridDegree;Auto\nParameter;CoupledMass;1\n",
    { elementFormulation: "MixedHybridFormulation", formulationGeometry: {} });
  const f = [1,-2,3,5], a = t.solveStatic(1, f), b = t.solveStatic(1, f.map(v => 2*v));
  try {
    const first = t.solveBuckling(2, a.preload).eigenvalues[0].rows;
    const second = t.solveBuckling(2, b.preload).eigenvalues[0].rows;
    assert.ok(first.length > 0);
    first.forEach((row,i) => near(row.eigenvalue, 2*second[i].eigenvalue));
    assert.deepEqual(a.selectedQuarticShells, b.selectedQuarticShells);
    const modes = t.solveModes(2);
    try {
      assert.ok(modes.modalBasis.generalizedLoads(f).some(v => Math.abs(v) > 1e-8));
    } finally { modes.modalBasis.delete(); }
  } finally { a.preload.delete(); b.preload.delete(); t.delete(); }
});

test("the equilibrated box forces of a restrained SOL 144 recover the trim displacements in SOL 101", () => {
  const text = wing.replace("[case]", "[case]\nDisplacementOutput;All");
  const trim = structuralRun(text, "aeroelastic", 1);
  const transfer = prepare(text);
  const state = transfer.solveStatic(1, trim.aeroelastic.subcases[0].boxForces);
  try {
    const u = trim.nodalTables.find(table => table.kind === "displacements");
    const v = state.nodalTables.find(table => table.kind === "displacements");
    assert.deepEqual(v.ids, u.ids);
    assert.ok(u.values.some(value => Math.abs(value) > 1e-8));
    v.values.forEach((value, i) => near(value, u.values[i]));
  } finally { state.preload.delete(); transfer.delete(); }
});
