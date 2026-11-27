# Solver scaling

- Machine: 12th Gen Intel(R) Core(TM) i5-12400F, 6 cores, 12 logical cores, 32 GiB of memory, Windows-11-10.0.26200-SP0.
- Software: oo-solver 0.2.5, oo-model 0.2.4, Python 3.13.3; element formulation: the solver's default.
- Measured on 2026-11-27T09:13:31+00:00.

Times in seconds. *Read + snapshot*: reading the deck and building the solver's model (one thread). *1 thread*, *12 threads*: the wall time of the analysis. *Peak memory*: the peak resident memory of the process, 1 thread / 12 threads. *Check*: the relative difference from the model's reference value.

*Over budget*: the solver stopped the run with a message naming what it needed, because the factorisation needed more than the memory budget of the run, which is the memory free when the run starts (16.3 to 18.6 GB on this machine).

## Linear statics

| Model | DOF | Read + snapshot | 1 thread | 12 threads | Speed-up | Peak memory | Check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Square plate, ShellQuad4 | 9,720 | 0.0332 | 0.100 | 0.0621 | 1.61 | 87 MiB / 92 MiB | +0.14 % |
| Square plate, ShellQuad4 | 29,610 | 0.0338 | 0.301 | 0.158 | 1.90 | 160 MiB / 171 MiB | +0.08 % |
| Square plate, ShellQuad4 | 98,688 | 0.0665 | 1.08 | 0.533 | 2.03 | 440 MiB / 450 MiB | +0.06 % |
| Square plate, ShellQuad4 | 301,728 | 0.145 | 3.65 | 1.45 | 2.51 | 1.26 GiB / 1.29 GiB | +0.05 % |
| Square plate, ShellQuad4 | 1,000,008 | 0.487 | 14.0 | 5.23 | 2.67 | 4.30 GiB / 4.31 GiB | +0.05 % |
| Square plate, ShellQuad4 | 2,006,238 | 0.835 | 32.9 | 11.6 | 2.84 | 8.93 GiB / 8.94 GiB | +0.05 % |
| Cylinder, ShellQuad4 | 9,576 | 0.0252 | 0.0823 | 0.0568 | 1.45 | 83 MiB / 90 MiB | -0.10 % |
| Cylinder, ShellQuad4 | 29,884 | 0.0327 | 0.282 | 0.155 | 1.82 | 165 MiB / 174 MiB | -0.04 % |
| Cylinder, ShellQuad4 | 98,724 | 0.0597 | 1.03 | 0.485 | 2.13 | 443 MiB / 461 MiB | -0.02 % |
| Cylinder, ShellQuad4 | 299,772 | 0.135 | 3.56 | 1.49 | 2.39 | 1.26 GiB / 1.30 GiB | -0.01 % |
| Cylinder, ShellQuad4 | 999,844 | 0.413 | 14.9 | 5.39 | 2.76 | 4.28 GiB / 4.34 GiB | -0.01 % |
| Cylinder, ShellQuad4 | 2,003,968 | 0.885 | 35.3 | 11.8 | 2.99 | 8.82 GiB / 8.89 GiB | -0.01 % |
| Wing box, ShellQuad4 | 10,440 | 0.0355 | 0.136 | 0.0640 | 2.13 | 93 MiB / 101 MiB | -0.29 % |
| Wing box, ShellQuad4 | 24,240 | 0.0373 | 0.233 | 0.117 | 1.98 | 158 MiB / 167 MiB | -0.26 % |
| Wing box, ShellQuad4 | 100,200 | 0.0581 | 1.05 | 0.451 | 2.34 | 545 MiB / 564 MiB | -0.23 % |
| Wing box, ShellQuad4 | 282,120 | 0.118 | 3.24 | 1.23 | 2.63 | 1.48 GiB / 1.50 GiB | -0.22 % |
| Wing box, ShellQuad4 | 1,028,400 | 0.428 | 15.4 | 5.04 | 3.07 | 5.32 GiB / 5.35 GiB | -0.22 % |
| Wing box, ShellQuad4 | 1,931,400 | 0.771 | 34.2 | 10.6 | 3.21 | 10.10 GiB / 10.12 GiB | -0.22 % |
| Block, SolidHexa | 10,752 | 0.0292 | 0.190 | 0.0854 | 2.22 | 139 MiB / 149 MiB | -0.99 % |
| Block, SolidHexa | 29,040 | 0.0487 | 0.630 | 0.244 | 2.58 | 353 MiB / 365 MiB | -0.92 % |
| Block, SolidHexa | 92,160 | 0.101 | 2.97 | 0.995 | 2.99 | 1.25 GiB / 1.29 GiB | -0.88 % |
| Block, SolidHexa | 317,952 | 0.309 | 20.6 | 5.28 | 3.90 | 5.58 GiB / 5.63 GiB | -0.85 % |
| Block, SolidHexa | 999,600 | – | over budget | over budget | – | – / – | – |

- Square plate, ShellQuad4: centre deflection against Navier's thin-plate series.
- Cylinder, ShellQuad4: mean radial displacement at mid-length against the membrane value p R²/(E t).
- Wing box, ShellQuad4: mean tip deflection against Timoshenko beam theory (bending + shear of the webs).
- Block, SolidHexa: mean end deflection against Timoshenko beam theory (Cowper's shear coefficient).

## Normal modes (first 10)

| Model | DOF | Read + snapshot | 1 thread | 12 threads | Speed-up | Peak memory | Check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Square plate, ShellQuad4 | 9,720 | 0.0235 | 0.325 | 0.311 | 1.04 | 111 MiB / 119 MiB | -0.12 % |
| Square plate, ShellQuad4 | 29,610 | 0.0323 | 0.903 | 0.620 | 1.46 | 239 MiB / 249 MiB | -0.06 % |
| Square plate, ShellQuad4 | 98,688 | 0.0566 | 3.35 | 2.17 | 1.55 | 709 MiB / 722 MiB | -0.04 % |
| Square plate, ShellQuad4 | 301,728 | 0.127 | 13.5 | 7.89 | 1.71 | 2.17 GiB / 2.15 GiB | -0.03 % |
| Square plate, ShellQuad4 | 1,000,008 | 0.472 | 58.3 | 32.4 | 1.80 | 7.60 GiB / 7.43 GiB | -0.03 % |
| Square plate, ShellQuad4 | 2,006,238 | – | over budget | over budget | – | – / – | – |
| Cylinder, ShellQuad4 | 9,576 | 0.0236 | 0.294 | 0.260 | 1.13 | 109 MiB / 120 MiB | -0.04 % |
| Cylinder, ShellQuad4 | 29,884 | 0.0319 | 1.03 | 0.686 | 1.50 | 242 MiB / 252 MiB | -0.01 % |
| Cylinder, ShellQuad4 | 98,724 | 0.0576 | 4.45 | 2.63 | 1.69 | 701 MiB / 719 MiB | +0.01 % |
| Cylinder, ShellQuad4 | 299,772 | 0.125 | 13.0 | 6.80 | 1.90 | 2.06 GiB / 2.05 GiB | +0.01 % |
| Cylinder, ShellQuad4 | 999,844 | 0.435 | 60.7 | 36.6 | 1.66 | 6.90 GiB / 6.81 GiB | +0.01 % |
| Cylinder, ShellQuad4 | 2,003,968 | 0.802 | 122 | 64.6 | 1.89 | 14.04 GiB / 13.84 GiB | +0.01 % |
| Wing box, ShellQuad4 | 10,440 | 0.0242 | 0.394 | 0.380 | 1.04 | 122 MiB / 138 MiB | -0.57 % |
| Wing box, ShellQuad4 | 24,240 | 0.0299 | 0.959 | 0.697 | 1.38 | 230 MiB / 239 MiB | -0.59 % |
| Wing box, ShellQuad4 | 100,200 | 0.0599 | 4.65 | 2.72 | 1.71 | 863 MiB / 869 MiB | -0.60 % |
| Wing box, ShellQuad4 | 282,120 | 0.126 | 14.4 | 7.29 | 1.97 | 2.38 GiB / 2.33 GiB | -0.61 % |
| Wing box, ShellQuad4 | 1,028,400 | 0.410 | 65.3 | 30.5 | 2.14 | 8.41 GiB / 8.29 GiB | -0.61 % |
| Wing box, ShellQuad4 | 1,931,400 | – | over budget | over budget | – | – / – | – |
| Block, SolidHexa | 10,752 | 0.0293 | 0.478 | 0.227 | 2.10 | 199 MiB / 207 MiB | +0.63 % |
| Block, SolidHexa | 29,040 | 0.0478 | 1.52 | 0.620 | 2.45 | 542 MiB / 550 MiB | +0.60 % |
| Block, SolidHexa | 92,160 | 0.0984 | 7.58 | 2.56 | 2.96 | 1.88 GiB / 1.87 GiB | +0.58 % |
| Block, SolidHexa | 317,952 | 0.293 | 51.4 | 16.4 | 3.14 | 8.32 GiB / 8.16 GiB | +0.56 % |
| Block, SolidHexa | 999,600 | – | over budget | over budget | – | – / – | – |

- Square plate, ShellQuad4: first frequency against the thin-plate value π/a² √(D/ρt).
- Cylinder, ShellQuad4: first frequency against Sanders' shell theory (end diaphragms).
- Wing box, ShellQuad4: first frequency against a Timoshenko beam with the ribs' mass spread along the span.
- Block, SolidHexa: first frequency against a Timoshenko beam.

## Linear buckling (first factor)

| Model | DOF | Read + snapshot | 1 thread | 12 threads | Speed-up | Peak memory | Check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Square plate, ShellQuad4 | 9,720 | 0.0251 | 0.309 | 0.209 | 1.47 | 107 MiB / 115 MiB | -0.08 % |
| Square plate, ShellQuad4 | 29,610 | 0.0320 | 0.929 | 0.485 | 1.92 | 224 MiB / 233 MiB | -0.07 % |
| Square plate, ShellQuad4 | 98,688 | 0.0556 | 3.43 | 1.62 | 2.12 | 659 MiB / 673 MiB | -0.06 % |
| Square plate, ShellQuad4 | 301,728 | 0.137 | 13.7 | 5.98 | 2.29 | 2.02 GiB / 2.02 GiB | -0.06 % |
| Square plate, ShellQuad4 | 1,000,008 | 0.415 | 60.3 | 24.4 | 2.47 | 7.10 GiB / 6.94 GiB | -0.06 % |
| Square plate, ShellQuad4 | 2,006,238 | – | over budget | over budget | – | – / – | – |
| Cylinder, ShellQuad4 | 9,576 | 0.0239 | 0.541 | 0.468 | 1.16 | 105 MiB / 121 MiB | +3.09 % |
| Cylinder, ShellQuad4 | 29,884 | 0.0327 | 1.92 | 1.18 | 1.62 | 228 MiB / 237 MiB | +1.20 % |
| Cylinder, ShellQuad4 | 98,724 | 0.0575 | 7.06 | 3.57 | 1.98 | 652 MiB / 666 MiB | +0.53 % |
| Cylinder, ShellQuad4 | 299,772 | 0.129 | 24.0 | 10.7 | 2.24 | 1.91 GiB / 1.90 GiB | +0.35 % |
| Cylinder, ShellQuad4 | 999,844 | 0.380 | 91.8 | 50.3 | 1.83 | 6.40 GiB / 6.32 GiB | +0.28 % |
| Cylinder, ShellQuad4 | 2,003,968 | 0.825 | 211 | 108 | 1.96 | 13.04 GiB / 12.85 GiB | +0.26 % |
| Wing box, ShellQuad4 | 10,440 | 0.0267 | 0.487 | 0.372 | 1.31 | 118 MiB / 134 MiB | – |
| Wing box, ShellQuad4 | 24,240 | 0.0303 | 1.18 | 0.698 | 1.69 | 218 MiB / 226 MiB | – |
| Wing box, ShellQuad4 | 100,200 | 0.0575 | 5.60 | 2.72 | 2.06 | 812 MiB / 815 MiB | – |
| Wing box, ShellQuad4 | 282,120 | 0.120 | 17.4 | 7.40 | 2.35 | 2.24 GiB / 2.19 GiB | – |
| Wing box, ShellQuad4 | 1,028,400 | 0.413 | 98.7 | 34.0 | 2.90 | 7.91 GiB / 7.78 GiB | – |
| Wing box, ShellQuad4 | 1,931,400 | 0.800 | 164 | 73.4 | 2.24 | 14.76 GiB / 14.74 GiB | – |
| Block, SolidHexa | 10,752 | 0.0304 | 0.887 | 0.391 | 2.27 | 186 MiB / 195 MiB | +0.39 % |
| Block, SolidHexa | 29,040 | 0.0446 | 2.45 | 0.910 | 2.69 | 504 MiB / 511 MiB | +0.34 % |
| Block, SolidHexa | 92,160 | 0.0985 | 11.6 | 3.76 | 3.09 | 1.76 GiB / 1.75 GiB | +0.30 % |
| Block, SolidHexa | 317,952 | 0.410 | 75.8 | 22.6 | 3.36 | 7.89 GiB / 7.72 GiB | +0.28 % |
| Block, SolidHexa | 999,600 | – | over budget | over budget | – | – / – | – |

- Square plate, ShellQuad4: first factor against the thin-plate value 4π²D/(a² N).
- Cylinder, ShellQuad4: first factor against Sanders' shell theory (end diaphragms).
- Wing box, ShellQuad4: no reference (skin panel buckling between ribs and spars).
- Block, SolidHexa: first factor against the column with shear (Engesser).
