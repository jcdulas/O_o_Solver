API reference
=============

The table maps the most common operations across both packages. For Python signatures, see :ref:`native-python-api`.
The TypeScript declarations and C++ headers below are included directly from the source tree.

For a runnable example, start with :doc:`getting-started`. The module guides explain implementation details.
For analytic coverage, fallback behavior and report flags, see :doc:`derivatives`. Element implementers can use
the live C++ contexts and facets in :doc:`element-derivative-api`.

Native C++20 with vcpkg
-----------------------

Configure the `O_o Git registry <https://github.com/jcdulas/O_o_vcpkg>`_ in your consumer manifest, add ``oo-solver`` to its dependencies and link the installed target:

.. code-block:: cmake

   find_package(oo-solver CONFIG REQUIRED)
   target_link_libraries(my_application PRIVATE oos::api)

Include ``<oos/api/ModelFromDeck.hpp>`` to call ``oos::api::modelFromDeck(deck)``. The returned snapshot retains
the imported column buffers. The source deck can be released before calling ``oos::api::runAnalysis``.
The `standalone rod example <https://github.com/jcdulas/O_o_vcpkg/tree/main/examples/rod>`_ builds and solves a model in memory.

.. list-table::
   :header-rows: 1

   * - Operation
     - JavaScript / WebAssembly
     - Python / native C++
   * - Build a solver snapshot from an in-memory deck
     - ``solver.modelFromReader(reader, deck)``
     - ``oo_solver.model_from_reader(deck)``
   * - Run statics
     - ``solver.run(model, "statics")``
     - ``model.run("statics")``
   * - Evaluate a design
     - ``problem.evaluate(x)``
     - ``problem.evaluate(x)``

.. _spline-transfer-for-an-external-solver:

Spline transfer for an external solver
--------------------------------------

Prepare one transfer from the structural model and reuse it throughout a coupling loop.
The transfer owns the spline matrices and remains usable after the model is released.
Prepare a new transfer when the geometry, spline membership or coordinate systems change.

* ``nodeIds`` / ``node_ids`` lists distinct spline attachment node IDs in increasing order.
  Supply six components per node in the basic frame, translations XYZ followed by small rotations XYZ in radians.
  Use the physical motion of each attachment, including any dependent RBE3 node.
* ``boxIds`` / ``box_ids`` gives the aerodynamic box order. The motion output contains the normal
  displacement at each box centre (``plunge``, in length units) and the dimensionless chordwise ``slope``.
* Supply one signed normal **force** per box, positive along the box normal and acting at its force point.
  For pressure jumps use force = pressure jump times box area. The returned node rows contain force XYZ
  followed by moment XYZ in the basic frame. Shared attachments accumulate contributions. A structural
  solve distributes the loads applied at dependent nodes through its constraint equations.
* Each spline honours its force or displacement usage. Boxes without a motion spline have zero output motion.
  A box contributes structural load only through a force-enabled spline.

**JavaScript / WebAssembly**

.. code-block:: javascript

   const transfer = model.aeroTransfer();
   try {
     const nodeIds = transfer.nodeIds, boxIds = transfer.boxIds;
     const u = new Float64Array(6 * nodeIds.length);
     for (let i = 0; i < nodeIds.length; ++i) u[6*i + 2] = 0.01;
     const { plunge, slope } = transfer.boxMotion(u);
     const forces = new Float64Array(boxIds.length).fill(1);
     const loads = transfer.structuralLoads(forces); // six consecutive values per node
   } finally { transfer.delete(); }

**Python / native C++**

.. code-block:: python

   import numpy as np

   transfer = model.aero_transfer()
   u = np.zeros((len(transfer.node_ids), 6))
   u[:, 2] = 0.01
   motion = transfer.box_motion(u)  # motion["plunge"], motion["slope"]
   forces = np.ones(len(transfer.box_ids))
   loads = transfer.structural_loads(forces)  # shape (nodes, 6)

**C++20**

.. code-block:: cpp

   #include <oos/api/AeroTransfer.hpp>

   // snapshot is the model returned by oos::api::modelFromDeck(deck).
   oos::api::AeroTransfer transfer(*snapshot);
   std::vector<double> u(6 * transfer.nodeIds().size());
   for (std::size_t i = 0; i < transfer.nodeIds().size(); ++i) u[6*i + 2] = 0.01;
   const auto motion = transfer.boxMotion(u);
   std::vector<double> forces(transfer.boxIds().size(), 1.0);
   const auto loads = transfer.structuralLoads(forces);

The examples use a uniform basic-Z translation and a unit normal force on each box. In a coupled run,
fill the same arrays from the structural and aerodynamic solutions using the returned IDs. The external
driver controls the coupling iterations and convergence criterion.

Coupling SOL 101, 103 and 105
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :doc:`external-aeroelastic-coupling` for the complete call sequence, subcase selection,
handle lifetimes and reference results for a four-box wing in JavaScript, Python and C++.

.. list-table::
   :header-rows: 1

   * - Operation
     - JavaScript / WebAssembly
     - Python
     - C++20
   * - Solve with external box forces
     - ``t.solveStatic(subcase, f)``
     - ``t.solve_static(subcase, f)``
     - ``t.solveStatic(subcase, f)``
   * - Obtain structural modes and box motion
     - ``t.solveModes(subcase)``
     - ``t.solve_modes(subcase)``
     - ``t.solveModes(subcase)``
   * - Calculate generalized loads
     - ``m.modalBasis.generalizedLoads(f)``
     - ``m["modal_basis"].generalized_loads(f)``
     - ``m.modalBasis->generalizedLoads(f)``
   * - Use the equilibrated preload
     - ``t.solveBuckling(subcase, s.preload)``
     - ``t.solve_buckling(subcase, s["preload"])``
     - ``t.solveBuckling(subcase, *s.preload)``

Here ``t`` is the prepared coupling, ``s`` is its static result and ``m`` is its modal result.

JavaScript and TypeScript
-------------------------

Load the built package from ``build/wasm/dist/index.js``. The declarations below describe its public interface.

.. literalinclude:: ../wasm/js/index.d.ts
   :language: typescript
   :linenos:

Optimization declarations
~~~~~~~~~~~~~~~~~~~~~~~~~

``point.nodalCoordinateGradient(responseId, nodeIds, item)`` in JavaScript and
``point.nodal_coordinate_gradient(response_id, node_ids, item)`` in Python return the basic-frame XYZ
derivatives for a scalar response item at the evaluated point. Node IDs follow the requested order.
The optional item index starts at zero. See :doc:`external-optimization` for paired examples and
the C++ entry point.

``DesignResponse.mode`` selects a modal field with a positive, one-based mode number. A modal result item includes
its ``mode`` label. ``ProductReport`` distinguishes static Hessian blocks from modal, buckling and trim tangent
paths; the flags are described in :doc:`derivatives`.

.. literalinclude:: ../wasm/js/optimization.d.ts
   :language: typescript
   :linenos:

External studies and workers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../wasm/js/study.d.ts
   :language: typescript
   :linenos:

.. literalinclude:: ../wasm/js/study-workers.d.ts
   :language: typescript
   :linenos:

.. _native-python-api:

Python
------

Import ``model_from_reader(deck, *, formulation=None)`` and
``prepare_optimization(deck, *, analysis="statics", formulation=None, threads=1)`` from ``oo_solver``.

* ``Model.run(analysis="statics", *, threads=1, subcases=[])`` returns a dictionary with
  read-only NumPy result arrays. Analyses are ``statics``, ``modes``, ``buckling`` and ``aeroelastic``.
* ``OptimizationProblem.variables`` and ``initial_values`` give the independent variables in vector order.
  ``problem.evaluate(x)`` returns a retained ``DesignPoint``.
* ``point.objective``, ``point.constraints`` and ``point.responses`` describe that evaluation.
  ``point.gradients()``, ``point.jvp(direction)`` and ``point.vjp(weights)`` reuse it.
* ``analysis={"aeroelastic": [1], "buckling": [2]}`` selects a trim and a buckling
  subcase in one design problem. Set ``StaticSubcaseSelection`` in the buckling
  subcase to select its preload.
* ``point.hessian_vector(weights, direction)`` returns a dictionary with a NumPy
  array of products and a tangent report. Directions have shape (variables,) or
  (variables, directions). Weights map response IDs to vectors of item weights.

See :doc:`optimization` for batched NumPy inputs, reusable output buffers and CSR layouts.
See :doc:`python` for paired examples and installation details. The public
`Python module <https://github.com/jcdulas/O_o_Solver/blob/main/python/oo_solver/__init__.py>`_ and
`native bindings <https://github.com/jcdulas/O_o_Solver/blob/main/python/bindings.cpp>`_ define the available calls.

C++ entry points
----------------

ModelInput.hpp
~~~~~~~~~~~~~~

.. literalinclude:: ../include/oos/api/ModelInput.hpp
   :language: cpp
   :linenos:

Analyses.hpp
~~~~~~~~~~~~

.. literalinclude:: ../include/oos/api/Analyses.hpp
   :language: cpp
   :linenos:

AeroTransfer.hpp and AeroCoupling.hpp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../include/oos/api/AeroTransfer.hpp
   :language: cpp

.. literalinclude:: ../include/oos/api/AeroCoupling.hpp
   :language: cpp

Optimization.hpp
~~~~~~~~~~~~~~~~

.. literalinclude:: ../include/oos/api/Optimization.hpp
   :language: cpp
   :linenos:

ModelFromDeck.hpp and ConstraintJacobian.hpp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../include/oos/api/ModelFromDeck.hpp
   :language: cpp

.. literalinclude:: ../include/oos/api/ConstraintJacobian.hpp
   :language: cpp

Capabilities.hpp
~~~~~~~~~~~~~~~~

.. literalinclude:: ../include/oos/api/Capabilities.hpp
   :language: cpp
   :linenos:

Threading.hpp
~~~~~~~~~~~~~

.. literalinclude:: ../include/oos/api/Threading.hpp
   :language: cpp
   :linenos:
