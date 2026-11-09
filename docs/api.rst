API reference
=============

These declarations are read from the same source files as the packages and C++ facade. A documentation build
therefore reflects the checked-out revision rather than a separately maintained copy of the signatures.

For a runnable example, start with :doc:`getting-started`. The module guides explain implementation details.

JavaScript and TypeScript
-------------------------

Load the built package from ``build/wasm/dist/index.js``. The declarations below describe its public interface.

.. literalinclude:: ../wasm/js/index.d.ts
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
