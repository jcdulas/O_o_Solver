# Module guide

The modules below follow the dependency order of the implementation. Each guide describes its responsibilities,
entry points and numerical or data conventions.

- [base](../src/base/README.md)
- [linalg](../src/linalg/README.md)
- [input](../src/input/README.md)
- [elements](../src/elements/README.md)
- [assembly](../src/assembly/README.md)
- [reduction](../src/reduction/README.md)
- [aero](../src/aero/README.md)
- [output](../src/output/README.md)
- [solutions](../src/solutions/README.md)
- [design](../src/design/README.md)

[Bar and beam derivatives](beam-derivatives.md) describes the shared flexibility, release, mass and recovery
derivative kernels and the native optimization paths that use them.

~~~{toctree}
:hidden:
:maxdepth: 2

../src/base/README
../src/linalg/README
../src/input/README
../src/elements/README
../src/assembly/README
../src/reduction/README
../src/aero/README
../src/output/README
../src/solutions/README
../src/design/README
~~~
