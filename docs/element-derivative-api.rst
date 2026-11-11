Native element derivative API
=============================

Package consumers normally use the optimization facade in :doc:`api`. This reference documents the C++ extension
points used by element implementers and native consumers. Declarations are included from the compiled headers,
so the documentation follows the checked-out source revision.

See :doc:`derivatives` for solution coverage and reports, and :doc:`beam-derivatives` for the bar/beam formulas.

Designed fields and use flags
----------------------------------------

``DesignedField`` identifies a raw card field. Its item is zero-based for list columns; scalar fields use ``-1``.
``FieldUse`` selects which element computations can change. The flags are conservative dependencies, rather than
a promise that a closed derivative is available. Beam equivalent line loads can depend on section/material
properties, and the beam families mark those dependencies explicitly.

.. literalinclude:: ../include/oos/elements/core/DesignFieldUse.hpp
   :language: cpp
   :start-at: struct DesignedField
   :end-before: // One rule of a family's Derivatives facet
   :linenos:

Load and recovery contexts
----------------------------------------

``ElementLoadContext`` borrows temperatures and element-specific applied loads. Inertial-load derivatives follow
the mass matrix and are assembled separately by the design layer.

.. literalinclude:: ../include/oos/elements/core/ElementFamily.hpp
   :language: cpp
   :start-at: struct ElementLoadContext
   :end-before: // The displacements of the whole model
   :linenos:

``NodalDisplacements`` borrows the state, temperature data and optional line loads used during recovery.
The ordinary bar/beam recovery layout stays the same when its line-load context is present. Each recovered row
selects its own batch-row loads. For a linear output at fixed geometry, loads change the affine term while the
displacement slopes and their field rates can be reused across subcases.

.. literalinclude:: ../include/oos/elements/core/ElementFamily.hpp
   :language: cpp
   :start-at: struct NodalDisplacements
   :end-before: // The plug-in interface of an element type
   :linenos:

Derivative facets
-----------------

These optional methods append matrix, load or recovered-row rates in the same DOF/row layout as their ordinary
facets. A return value of ``true`` with an empty matrix/load block means a known zero; ``false`` means unavailable
and leaves the design layer's numerical fallback in place. Recovery columns without a supplied formula use NaN
according to the individual method's contract. Only claim a closed derivative when all terms of that requested
field and context are included.

The geometric-stiffness methods return partials at fixed preload displacements and temperatures. The solution
layer adds preload tangents where supported; those element partials alone do not provide a total buckling Hessian.
``loadFieldMixedDerivative`` also supplies a same-field second derivative when its two fields are identical.

.. literalinclude:: ../include/oos/elements/core/ElementFamily.hpp
   :language: cpp
   :start-at:     // dK/dv and dM/dv of the element
   :end-before: protected:
   :linenos:

Shared bar and beam implementation
----------------------------------------

``BeamFieldDerivatives`` implements the shared sizing facets. A null ``other`` selects a first partial; an identical
second field selects the same-field second partial. Unknown fields or unsupported contexts return ``false``.
The implementation carries values and two first partials plus their mixed partial through local arithmetic;
it does not perturb the model to obtain these rates.

.. literalinclude:: ../include/oos/elements/bars_beams/BeamFieldDerivatives.hpp
   :language: cpp
   :start-at: class BeamFieldDerivatives
   :end-before: } // namespace oos::elements
   :linenos:

Field lists, load conventions, station meanings and recovery limits are described in :doc:`beam-derivatives`.
