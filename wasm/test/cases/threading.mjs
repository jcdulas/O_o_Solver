import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "../helpers/register.mjs";
import { solver, oor, info, dist, checks, loadOosSolver, supportsWasmThreads } from "../helpers/runtime.mjs";

function memoryFlags(bytes) {
  let position = 8;
  function leb() {
    let value = 0, scale = 1, byte;
    do { byte = bytes[position++]; value += (byte & 127) * scale; scale *= 128; } while (byte & 128);
    return value;
  }
  while (position < bytes.length) {
    const section = bytes[position++], size = leb(), end = position + size;
    if (section === 5) { assert.equal(leb(), 1); return leb(); }
    if (section === 2) {
      const count = leb();
      for (let i = 0; i < count; ++i) {
        const moduleLength = leb(); position += moduleLength;
        const nameLength = leb(); position += nameLength;
        const kind = bytes[position++];
        if (kind === 2) return leb(); // shared memories are imported by the worker runtime
        if (kind === 0) leb();
        else throw new Error(`unexpected import kind ${kind} before the memory`);
      }
    }
    position = end;
  }
  throw new Error("module has no memory");
}

test("the package describes the actual memory and rejects invalid worker counts", async () => {
  const flags = memoryFlags(readFileSync(path.join(dist, "oos_solver_core.wasm")));
  assert.equal(Boolean(flags & 4), true);
  assert.equal(Boolean(flags & 2), info.enabled);
  assert.equal(info.activeWorkers, 0);
  if (info.enabled) assert.equal(supportsWasmThreads(), true);
  for (const threads of [0, -1, 1.5, NaN, info.maximumThreads + 1])
    await assert.rejects(() => loadOosSolver({ threads }), /threads/);
});

for (const threads of [1, 2, 4, 8].filter(n => n <= info.maximumThreads))
  test(`real workers execute nested tasks and propagate failures at ${threads} threads`, {
    skip: !checks,
  }, () => {
    assert.equal(checks.checkThreads(threads), threads);
    assert.equal(checks.checkThreads(threads), threads);
  });

// An original clamped plate: many elements exercise assembly, sparse factorisation and recovery.
function plate() {
  const nx = 8, ny = 2, id = (i, j) => j * (nx + 1) + i + 1;
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1",
    "SinglePointConstrSelection;1", "StaticLoadSelection;2", "EigenMethodSelection;3",
    "DisplacementOutput;All", "StressOutput;All", "[model]"];
  for (let j = 0; j <= ny; ++j) for (let i = 0; i <= nx; ++i)
    lines.push(`Node;${id(i, j)};;${i * 0.1};${j * 0.1};0.`);
  let element = 1;
  for (let j = 0; j < ny; ++j) for (let i = 0; i < nx; ++i)
    lines.push(`ShellQuad4;${element++};1;${id(i,j)};${id(i+1,j)};${id(i+1,j+1)};${id(i,j+1)}`);
  lines.push("ShellProp;1;1;0.02;1;;1", "IsoMat;1;7.E10;;0.3;2700.", "LanczosEigenMethod;3;;;2");
  for (let j = 0; j <= ny; ++j) lines.push(`SinglePointConstrList;1;123456;${id(0,j)}`);
  lines.push(`PointForceLoad;2;${id(nx, ny)};;-100.;0.;0.;1.`);
  return lines.join("\n") + "\n";
}

for (const analysis of ["statics", "modes"])
  for (const threads of [2, 4, 8].filter(n => n <= info.maximumThreads))
    test(`${analysis} results agree at one and ${threads} threads`, { skip: !info.enabled }, () => {
      oor.writeFile("/work/threaded-plate.oos", plate());
      const deck = oor.readModel("/work/threaded-plate.oos", { strict: true });
      let model;
      try {
        model = solver.modelFromReader(oor, deck);
        const reference = solver.run(model, analysis, { threads: 1 });
        const actual = solver.run(model, analysis, { threads });
        assert.deepEqual(actual.nodalTables, reference.nodalTables);
        assert.deepEqual(actual.elementTables, reference.elementTables);
        assert.deepEqual(actual.eigenvalues, reference.eigenvalues);
        assert.equal(solver.threading().activeWorkers, 0);
      } finally { model?.delete(); deck.delete(); }
    });

test("optimization retains its workers and releases them with its last evaluation", {
  skip: !info.enabled,
}, () => {
  oor.writeFile("/work/threaded-design.oos", plate() +
    "DesignVar;1;THICKNESS;0.02;0.01;0.1\nDesignPropRelation1;1;ShellProp;1;thickness;;;;1;1.\n");
  const deck = oor.readModel("/work/threaded-design.oos", { strict: true });
  const threads = Math.min(4, info.maximumThreads);
  const problem = solver.optimizationFromReader(oor, deck, {
    responses: [{ id: 10, type: "Volume" }, { id: 11, type: "Displacement", attributeA: 3, targets: [27] }],
  }, { threads });
  const point = problem.evaluate();
  const other = solver.optimizationFromReader(oor, deck, {}, { threads });
  try {
    assert.equal(solver.threading().activeWorkers, threads - 1); // same-sized problems share the workers
    if (threads > 1 && threads < info.maximumThreads)
      assert.throws(() => solver.optimizationFromReader(oor, deck, {}, { threads: info.maximumThreads }),
        /capacity/);
    assert.equal(solver.threading().activeWorkers, threads - 1);
    const serial = solver.optimizationFromReader(oor, deck, {
      responses: [{ id: 10, type: "Volume" }, { id: 11, type: "Displacement", attributeA: 3, targets: [27] }],
    }, { threads: 1 });
    const reference = serial.evaluate();
    try {
      const expected = reference.gradients().responses;
      const actual = point.gradients().responses;
      assert.deepEqual(actual, expected);
      const weights = { responses: { 11: [1] } };
      assert.deepEqual(point.hessianVector(weights, [1]).values, reference.hessianVector(weights, [1]).values);
    } finally { reference.delete(); serial.delete(); }
    other.delete(); problem.delete();
    assert.equal(solver.threading().activeWorkers, threads - 1);
    point.gradients();
  } finally { other.delete(); problem.delete(); point.delete(); deck.delete(); }
  assert.equal(solver.threading().activeWorkers, 0);
});
