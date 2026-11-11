API reference
=============

These declarations are read from the same source files as the packages and C++ facade. A documentation build
therefore reflects the checked-out revision rather than a separately maintained copy of the signatures.

For a runnable example, start with :doc:`getting-started`. The module guides explain implementation details.
For analytic coverage, fallback behavior and report flags, see :doc:`derivatives`. Element implementers can use
the live C++ contexts and facets in :doc:`element-derivative-api`.

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
