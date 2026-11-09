# Legal notes

The rules this project follows to stay independent of the software vendors that sell finite element solvers. This
document is not legal advice.

## Input and output

- O_o_Reader reads the project's `.oos` model format. The solver consumes its typed table packages and produces
  plain result arrays. O_o_Converter handles conversions from external input formats.
- The solver writes no vendor file format, binary or printed.

## Names

- Code, model records, interfaces and diagnostics use neutral project names.
- The name NASTRAN-95 identifies NASA's open-source release when cited as a scientific source.
- The product, namespace (`oos`), libraries (`oos_*`) and package (`oos-solver`) carry no vendor name.

## Sources

- The code describes model fields, defaults and validation rules in the project's own words. The numerical methods cite their public sources (textbooks
  and papers).
- No vendor documentation, example deck or result file is part of the repository, and none may be added or quoted.
- The ShellQuad4 and ShellTria3 elements follow the formulation of NASTRAN-95 (https://github.com/nasa/NASTRAN-95), released
  by NASA under the NASA Open Source Agreement 1.3. They were written from a mathematical description of that
  formulation, not by translating its Fortran, and contain no NASTRAN-95 code; the source is cited in the comments.
- Where an open-source solver's published documentation describes a method (the CalculiX user's manual for the tied
  contact and the incompatible-modes hexahedron, the NASA reports of the ADS and CONMIN optimisers for the design
  defaults), the code cites that document; no code of those programs is used.

## Third-party components

| Component | Use | Licence |
| --- | --- | --- |
| Emscripten | the WebAssembly build | MIT / University of Illinois NCSA |

The solver uses nothing else: its linear algebra (sparse factorisation, eigenvalue solvers, dense kernels) is its own.
