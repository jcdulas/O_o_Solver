# External optimizer client benchmark

The axial rod and the 16 by 8 quadrilateral membrane have two independent variables, two responses
and three normalized constraint rows. All clients construct the same model in memory from the fixture
cards. The fixtures use a total axial load of 10, length 3, Young's modulus 1000 and area or thickness 2.
Both cases give a loaded-edge displacement of 0.015 and a volume of 6.
The panel has 153 nodes and 128 elements. Its zero Poisson ratio and distributed edge load give a uniform axial strain.

Run from the Solver root with a compatible Model and the Solver SDK or packages built from this revision.
Set `CMAKE_PREFIX_PATH` to the installed C++ SDK prefix or supply the configured vcpkg toolchain.
For JavaScript set `OOR_READER_DIST` and `OOS_SOLVER_DIST` to the package directories and use a Memory64-capable Node.js.

```sh
python docs/benchmarks/optimizer-api/fixtures.py build/optimizer-fixtures
cmake -S docs/benchmarks/optimizer-api -B build/optimizer-client -DCMAKE_BUILD_TYPE=Release
cmake --build build/optimizer-client --config Release
build/optimizer-client/optimizer_api build/optimizer-fixtures/rod.json
build/optimizer-client/optimizer_api build/optimizer-fixtures/panel.json
node docs/benchmarks/optimizer-api/js-client.mjs build/optimizer-fixtures/rod.json build/optimizer-fixtures/panel.json
python docs/benchmarks/optimizer-api/python_client.py build/optimizer-fixtures/rod.json build/optimizer-fixtures/panel.json
```

The timers exclude fixture I/O and model editing. They separate preparation, evaluation, the first complete gradient,
and the mean of 20 two-direction JVPs after priming the linearization. The output size counts the numerical product
returned to the client. It excludes temporary buffers and JS/WASM transfers. The reported gradient factorization count
covers derivative work after evaluation. The clients also check a three-row CSR Jacobian
at a second point against normalized constraint gradients.

With SciPy installed in the client environment, run the independent SLSQP example on either fixture.
The minimum volume is 2.25 at size 0.75 and Young's modulus 2000. The upper displacement bound of 0.02 is active.
The example shares one native evaluation between optimizer callbacks and checks the accepted point by a fresh solve.

```sh
python docs/benchmarks/optimizer-api/external_slsqp.py build/optimizer-fixtures/rod.json build/optimizer-fixtures/panel.json
```

## Measurements on Windows x64

Measured on an Intel Core i5-12400F. Each entry is the median of five fresh processes for that client
and fixture, in milliseconds. The panel tests use a single solver thread. Preparation begins after the
model is constructed and the package is loaded. First-call WASM initialization and JIT work can still
appear inside these timings. C++ uses MSVC 19.39, the Python 3.13.3 extension uses GCC 16.1, and
JavaScript uses Node 24.19 and Memory64. Cross-client times include these differences in compiler and runtime.

| Model | Client | Prepare | Evaluate | First full gradient | Warm two-direction JVP |
| --- | --- | ---: | ---: | ---: | ---: |
| Rod | JavaScript/WASM | 270.73 | 83.08 | 21.78 | 0.0541 |
| Rod | Python | 6.05 | 1.56 | 2.77 | 0.0040 |
| Rod | C++ | 4.91 | 1.30 | 2.56 | 0.0029 |
| Panel | JavaScript/WASM | 279.96 | 116.37 | 97.20 | 0.2121 |
| Panel | Python | 6.42 | 8.72 | 38.57 | 0.0600 |
| Panel | C++ | 5.88 | 15.30 | 56.36 | 0.0984 |

All six gradient reports showed zero additional factorizations after evaluation. Each two-direction product
returned 32 bytes of numerical values. The JS wrapper fetches two numbers per response once on the first JVP
and reuses them. A VJP needs no response-size transfer. A single output block crosses into JS as an owned array
and is returned directly.

On the panel, eight Python JVPs with preallocated outputs took 0.4365 ms separately and 0.0969 ms
in one batch (4.5×). On the rod the corresponding times were 0.0216 and 0.0053 ms (4.1×).
The script verifies that separate and batched derivatives agree.

For the WASM comparison, the client requests all 153 panel nodal displacements as an extra response,
then repeatedly asks for the gradient of volume alone. The median time per warm VJP over five fresh
processes (200 calls each) was 0.6157 ms with the released 0.2.1 WASM package and 0.0540 ms with
this build (11.4×). Both used the same Model package. This measures the derivative callback on a retained
point. The structural analysis runs before the timer starts. To reproduce, run the command separately with
each Solver package directory in `OOS_SOLVER_DIST`.

```sh
node docs/benchmarks/optimizer-api/js-client.mjs --vjp build/optimizer-fixtures/panel.json
```

The external SLSQP run reached the analytical optimum in nine distinct trial evaluations on each mesh,
followed by a fresh evaluation of the accepted point.
