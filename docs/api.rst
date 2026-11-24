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

JavaScript and TypeScript
-------------------------

Load the built package from ``build/wasm/dist/index.js``. The declarations below describe its public interface.

.. literalinclude:: ../wasm/js/index.d.ts
   :language: typescript
   :linenos:

Optimization declarations
~~~~~~~~~~~~~~~~~~~~~~~~~

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
``optimization_from_reader(deck, *, analysis="statics", formulation=None, threads=1)`` from ``oo_solver``.

* ``Model.run(analysis="statics", *, threads=1, subcases=[])`` returns a dictionary with
  read-only NumPy result arrays. Analyses are ``statics``, ``modes``, ``buckling`` and ``aeroelastic``.
* ``OptimizationProblem.variables`` and ``initial_values`` give the independent variables in vector order.
  ``problem.evaluate(x)`` returns a retained ``DesignPoint``.
* ``point.objective``, ``point.constraints`` and ``point.responses`` describe that evaluation.
  ``point.gradients()``, ``point.jvp(direction)`` and ``point.vjp(weights)`` reuse it.

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

Optimization.hpp
~~~~~~~~~~~~~~~~

.. literalinclude:: ../include/oos/api/Optimization.hpp
   :language: cpp
   :linenos:

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
