# Structural benchmarks across formulations and solvers

How much does the choice of formulation change a result on the same mesh? On the Scordelis–Lo roof with an 8 × 8 grid split into distorted triangles, the classical formulation gives a free-edge deflection of **0.260842 ft** and mixed Auto gives **0.301329 ft**, against the conventional **0.3024 ft**. The corresponding errors are **13.74%** and **0.35%**. On a pin-ended vibrating beam, the analytical first frequency is **28.75826 Hz** with section rotary inertia. With four elements, mixed Auto gives **28.75826 Hz** and Classic gives **28.76644 Hz**.

The comparison covers ten physical problems, from curved-shell bending to buckling. Each of the 41 input meshes was run twice, with the default classical formulation and with `MixedHybridFormulation` in Auto. Both use the same nodes, connectivity, materials, supports and loads. The commercial results come from the linked verification manuals or the cited authors' calculations. Element type, mesh density and how the result was sampled matter when comparing the columns.

## Shells and plates

The entries are displacement magnitudes in the load direction. The cylinder and hemisphere use their respective benchmark's consistent length units. Shell mesh sizes count cells on the modeled portion. An 8 × 8 triangular roof has 128 triangles on the quarter roof.

| Benchmark and measurement | O_o mesh | Reference | Classic | Mixed Auto | Published commercial result |
| --- | --- | ---: | ---: | ---: | --- |
| Scordelis–Lo, free-edge midpoint (ft) | QUAD4, 8 × 8 | 0.3024 | 0.306110 | 0.301508 | 0.30845, OptiStruct QUAD4 [OS-V 0730][roof] |
| Scordelis–Lo, same point (ft) | TRIA3, 8 × 8 | 0.3024 | 0.259222 | 0.301343 | 0.26793, OptiStruct TRI3 [OS-V 0730][roof] |
| Scordelis–Lo, same point (ft) | Distorted TRIA3, 8 × 8 | 0.3024 | 0.260842 | 0.301329 | Matching distorted mesh unavailable in [OS-V 0730][roof] |
| Pinched cylinder, inward motion at the force | QUAD4, 10 × 10 | 1.8248 × 10⁻⁵ | 1.73996 × 10⁻⁵ | 1.85469 × 10⁻⁵ | 1.51895 × 10⁻⁵, Abaqus/Standard S4, 10 × 10 [parametric study][pinch] |
| Open hemisphere, outward motion at the force | QUAD4, 8 × 8 | 0.0940 | 0.0946728 | 0.0935749 | 0.09257, Abaqus S4, regular mesh, N = 8 [Ko et al.][ko] |
| Twisted cantilever, in-plane tip load (in) | QUAD4, 12 × 2 | 0.005424 | 0.00537751 | 0.00540073 | 0.00539, MSC Nastran QUAD4 [§1.8][twist] |
| Twisted cantilever, out-of-plane tip load (in) | QUAD4, 12 × 2 | 0.001754 | 0.00172939 | 0.00175325 | 0.00173, MSC Nastran QUAD4 [§1.8][twist] |
| Simply supported square plate, centre (m) | QUAD4, 8 × 8 quarter | 1.689 × 10⁻⁴ | 1.69959 × 10⁻⁴ | 1.69081 × 10⁻⁴ | 1.675 × 10⁻⁴, ADINA, 64 plate elements on the quarter [A.27][adina] |

**Scordelis–Lo.** The roof has rigid end diaphragms, a free longitudinal edge, a vertical load of 90 lbf/ft² and Poisson's ratio zero. For the distorted case the interior nodes move by up to 20% of a cell in each surface parameter, with the boundary nodes fixed. OptiStruct publishes ratios of 1.020 and 0.886 in its column labeled 8. Multiplication by 0.3024 ft gives the commercial values above. A separate refinement study proposes **0.30200–0.30204 ft** for a shear-flexible shell and **0.3006 ft** for a shear-rigid shell. Against 0.30202 ft, the mixed QUAD4 result on 16 × 16 has **0.052%** error, compared with **0.177%** against the historical target. The local inputs include transverse shear. [OptiStruct verification][roof] · [Krysl and Chen, §2.1][limits]

**Pinched shells.** The cylinder uses an octant, rigid end diaphragms and a quarter of a unit force at the loaded node. The reference comes from thin-shell theory. On the same 10 × 10 octant Abaqus also reports **1.80581 × 10⁻⁵** with quadratic S8R shells. The hemisphere has an 18° polar opening, a free equator and alternating radial point loads. Ko et al. report the S4 ratio **0.9954** relative to **0.0930**. The table converts their ratio to displacement before comparing it with the historical **0.0940**. Krysl and Chen extrapolate **0.093597** for the open hemisphere. Against that value the 16 × 16 errors are **0.187%** for Classic and **0.010%** for mixed Auto. [Abaqus pinched cylinder][pinch] · [Ko et al., cut-out hemisphere][ko] · [Krysl and Chen, §2.2][limits]

**Twisted cantilever.** The strip twists uniformly through 90° over 12 in, is 1.1 in wide and 0.32 in thick, and has a clamped root. The local model uses the MSC guide's stated material ($E=29$ Mpsi, $\nu=0.30$), distributes a 1 lbf resultant over the tip edge and reads the centre of that edge. The reference and commercial columns reproduce the guide's targets and `macneal009` results. The guide identifies the element type in the results table without giving a subdivision count there. [MSC Nastran 2022.2, §1.8][twist]

**Flat plate.** The full side is 0.8 m, thickness 8 mm, pressure 1 kPa, $E=210$ GPa and $\nu=0.30$. The shell uses simple supports, quarter symmetry and the in-plane restraints of the ADINA plate example. ADINA quotes a rounded thin-plate bending solution for centre deflection. The solid version below uses a separate three-dimensional elasticity reference. [ADINA verification A.27][adina]

## Solids and thermoelasticity

| Benchmark and measurement | O_o mesh | Reference | Classic | Mixed Auto | Published commercial result |
| --- | --- | ---: | ---: | ---: | --- |
| Straight cantilever, in-plane tip motion (in) | Regular HEX8, 6 × 1 × 1 | 0.1081 | 0.105996 | 0.107554 | 0.10572, OptiStruct HEX8, rectangular [OS-V 0720][straight] |
| Same cantilever and load (in) | Trapezoidal HEX8, 6 × 1 × 1 | 0.1081 | 0.00512412 | 0.107549 | 0.00508, OptiStruct HEX8, trapezoidal [OS-V 0720][straight] |
| Same trapezoidal cantilever (in) | HEX20, 6 × 1 × 1 | 0.1081 | 0.0957380 | 0.107545 | 0.09751, OptiStruct HEX20, trapezoidal [OS-V 0720][straight] |
| Square plate, centre deflection (m) | TET4, 8 × 8 × 1 full plate | 1.69055 × 10⁻⁴ | 1.36551 × 10⁻⁶ | 1.67591 × 10⁻⁴ | Related plate-element result above. [ADINA A.27][adina] gives no TET4 value. |
| Thick cylinder, bore expansion (in) | HEX20, 4 × 8 × 1 | 0.00786667 | 0.00786734 | 0.00786665 | 0.0078667, Ansys PLANE183 [VM25][vessel] |
| Thick cylinder, bore hoop stress (psi) | Same HEX20 mesh | 50,000 | 50,589.25 | 50,000.30 | 49,908.05, Ansys PLANE183 [VM25][vessel] |
| Heated bar after gap closure, axial stress (psi) | HEX8, 6 × 1 × 1 | −6,125 | −6,125 | −6,125 | −6,122.4, Ansys Mechanical [VM-WB-MECH-065][thermal] |

**Distortion in bending.** The 6 in cantilever has a 0.2 × 0.1 in section, a fixed root and a 1 lbf shear load distributed to the end-face nodes. The local output is the mean displacement of the tip corner nodes along the load. Both element orders receive the consistent nodal forces of a uniform end traction. Altair gives a beam-theory target of 0.1081 in and HEX8 ratios of 0.978 and 0.047. Changing only the six bricks from rectangular to trapezoidal takes the local classical error from **1.95% to 95.26%**. Mixed Auto gives **0.50% and 0.51%**, respectively. Subdividing each distorted brick into eight brings the classical error to **30.01%** and the mixed error to **0.29%**. [OptiStruct verification, tables 1 and 3][straight]

With the same trapezoidal corner geometry, adding mid-edge nodes reduces the classical error to **11.44%** on six HEX20s and **1.21%** on the refined mesh. OptiStruct publishes a ratio of 0.902 on the coarse HEX20 mesh, or 9.8% error. [OptiStruct verification, table 3][straight]

**One-layer tetrahedra.** The solid plate has the dimensions and material of the ADINA case. Diaphragm supports act through the thickness, pressure acts on the top face, and the probe averages the top and bottom centre displacements. Here 8 × 8 means **full-plate** cells, giving 4 × 4 × 1 cells and 96 tetrahedra on the modeled quarter. The reference scales the three-dimensional Fourier solution for the 1 m plate by 0.8 for the length scale and by 1/3 for the change in modulus. On the regular 8 × 8 × 1 mesh the classical and mixed deflection errors are **99.19%** and **0.87%**. The companion study also reports surface stresses, distorted grids and additional layers. [Thin-plate bending with solid elements](thin-plate-solid-bending.md)

**Pressure vessel.** The internal pressure is 30,000 psi, the inner and outer radii are 4 and 8 in, and the cylinder can contract axially. The reference values are the Lamé solution at the bore for zero axial stress. The local model uses a quarter ring with quadratic geometry, meshed radially, circumferentially and axially in that order. At $(4,0,0)$ the hoop component is $\sigma_{yy}$, averaged over incident element corners. Ansys uses an eight-node axisymmetric solid. Both displacement and stress are reported because near agreement in bore expansion can coexist with a larger stress error. [Ansys Mechanical APDL VM25, internal-pressure load case][vessel]

**Heated bar.** A 3 in bar expands across a 0.002 in gap under a 100 °F temperature rise. After closure its axial stress is $E(g/L-\alpha\Delta T)=-6125$ psi. The local models prescribe the final end displacement and sample axial stress at midspan. This compares the final thermoelastic equilibrium. Ansys models closure in its verification case using a nominal 0.25 in mesh. [Ansys Mechanical verification][thermal]

## Vibration and stability

| Benchmark and measurement | O_o mesh | Reference | Classic | Mixed Auto | Published commercial result |
| --- | --- | ---: | ---: | ---: | --- |
| Pin-ended beam, first bending frequency (Hz) | 4 BAR2 | 28.758260893 (Rayleigh) | 28.76644 | 28.75826 | 28.761, Ansys BEAM189, 4 quadratic beam elements (published target 28.766) [VM50][modes] |
| Pin-ended column, first buckling load (lbf) | 10 BAR2 on half-length | 38.553142 | 38.553175 | 38.553142 | 38.553, Ansys BEAM188, 10 cubic beam elements on half-length [VM127][buckling] |

**First bending frequency.** The local beam is simply supported, with span $L=80$ in, square section $A=4$ in² and $I=4/3$ in⁴, $E=30\times10^6$ psi and $\rho=0.000728$ lbf·s²/in⁴. Both formulations use consistent mass with section rotary inertia and specify rigid transverse shear. For the first half-wave, $k=\pi/L$, the matching analytical frequency is the Rayleigh value

$$
f_1=\frac{k^2}{2\pi}\sqrt{\frac{EI}{\rho(A+Ik^2)}}=28.7582608934\ \mathrm{Hz}.
$$

Omitting the rotary inertia term $Ik^2$ gives 28.7656514 Hz, which rounds to the 28.766 Hz target printed in VM50. Ansys reports 28.761 Hz from four quadratic BEAM189 elements. The classical four-element O_o result lies close to the printed target because its discretization error offsets the effect of rotary inertia. With refinement it approaches the Rayleigh frequency. [Ansys VM50][modes] · [VM50 input][modes-input]

| BAR2 elements on the span | Classic (Hz) | Classic absolute error (Hz) | Mixed Auto (Hz) | Mixed absolute error (Hz) |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 28.766442726 | 8.18 × 10⁻³ | 28.758261055 | 1.61 × 10⁻⁷ |
| 8 | 28.758920687 | 6.60 × 10⁻⁴ | 28.758260894 | 6.35 × 10⁻¹⁰ |
| 16 | 28.758337820 | 7.69 × 10⁻⁵ | 28.758260894 | 8.70 × 10⁻¹¹ |

For both formulations on all three meshes the first mode's nodal transverse displacements match the analytical half-sine (modal assurance criterion above 0.999999). In the mixed section law, a rigid shear plane now has exactly zero shear compliance. The former finite stiffness of $10^4EA$ left a frequency error of about $7.4\times10^{-7}$ Hz on the two finer meshes. These errors measure numerical agreement for the specified uniform beam.

The buckling model uses the equivalent fixed-free half of a 200 in pin-ended column with a 0.5 in square section. The Euler critical load is $\pi^2 EI/L^2$, with $L=200$ in. The compressive reference load is 1 lbf, so the first positive load multiplier equals the critical load numerically in lbf. [Ansys VM127][buckling] · [VM127 input][buckling-input]

## Closest result observed under refinement

For each measurement and formulation, the table gives the computed value, its absolute relative error $100|q-q_{\rm ref}|/|q_{\rm ref}|$, and the mesh that attained it. Each row uses the target and units from the corresponding comparison above. The search covers only the listed meshes, with fixed Auto settings throughout. Agreement finer than a published target's rounding should be interpreted at the precision of that target.

| Measurement | Meshes tried | Classic best (error), mesh | Mixed Auto best (error), mesh |
| --- | --- | --- | --- |
| Roof, QUAD4 | 4², 8², 16² | 0.302922 (0.173%), 16² | 0.301864 (0.177%), 16² |
| Roof, TRIA3 | 4², 8², 16² | 0.289393 (4.30%), 16² | 0.301639 (0.251%), 16² |
| Roof, distorted TRIA3 | 4², 8², 16² | 0.288579 (4.57%), 16² | 0.301658 (0.245%), 16² |
| Pinched cylinder | 5², 10², 20² | 1.86173 × 10⁻⁵ (2.02%), 20² | 1.85469 × 10⁻⁵ (1.64%), 10² |
| Open hemisphere | 4², 8², 16² | 0.0937719 (0.243%), 16² | 0.0936068 (0.418%), 16² |
| Twisted cantilever, in plane | 12 × 2, 24 × 4 | 0.00539060 (0.616%), 24 × 4 | 0.00540080 (0.428%), 24 × 4 |
| Twisted cantilever, out of plane | 12 × 2, 24 × 4 | 0.00174325 (0.613%), 24 × 4 | 0.00175330 (0.040%), 24 × 4 |
| Square plate, QUAD4, quarter | 4², 8², 16² | 1.69307 × 10⁻⁴ (0.241%), 16² | 1.69081 × 10⁻⁴ (0.107%), 4² |
| Regular HEX8 cantilever | 6 × 1 × 1, 12 × 2 × 2 | 0.106228 (1.73%), 12 × 2 × 2 | 0.107783 (0.293%), 12 × 2 × 2 |
| Trapezoidal HEX8 cantilever | 6 × 1 × 1, 12 × 2 × 2 | 0.0756639 (30.01%), 12 × 2 × 2 | 0.107782 (0.294%), 12 × 2 × 2 |
| Trapezoidal HEX20 cantilever | 6 × 1 × 1, 12 × 2 × 2 | 0.106788 (1.21%), 12 × 2 × 2 | 0.107781 (0.296%), 12 × 2 × 2 |
| Square plate, TET4, full plate | 4², 8², 16², one layer each | 5.07753 × 10⁻⁶ (97.00%), 16² | 1.68978 × 10⁻⁴ (0.046%), 16² |
| Thick cylinder, expansion | 4 × 8 × 1, 8 × 16 × 1 | 0.00786671 (0.00057%), 8 × 16 × 1 | 0.007866666 (0.000013%), 8 × 16 × 1 |
| Thick cylinder, hoop stress | Same two meshes | 50,190.97 (0.382%), 8 × 16 × 1 | 49,999.78 (0.00045%), 8 × 16 × 1 |
| Heated bar, stress | 6 × 1 × 1, 12 × 1 × 1 | −6,125 (<0.000001%), both | −6,125 (<0.000001%), both |
| First bending frequency (Rayleigh) | 4, 8, 16 | 28.758337820 (0.000267%), 16 | 28.758260894 (<0.000001%), 16 |
| First buckling load | 5, 10, 20 | 38.553144 (0.0000053%), 20 | 38.553142 (<0.000001%), 10 |

## Reproduce the comparison

The archive contains all **41 model files**. The results file identifies each mesh, the reference, and the exact nodes, stress component or eigenvalue used for all **86 measurements** from **82 solves**. It retains the values before rounding. The two formulations used the same single-thread WebAssembly build, with the classical EAS-21 HEX8 and reader checkout `518ba32`.

{download}`Model files <benchmarks/structural-comparison-inputs.zip>` ·
{download}`Full numerical results <benchmarks/structural-comparison-results.json>` ·
{download}`Replay script <benchmarks/replay-structural-comparison.mjs>`

With the Reader and Solver packages built and Node.js 24 available, run from the repository root. The replay evaluates every probe and compares it with the archived value using a tolerance of one millionth of the reference magnitude. [Build and validation instructions](testing.md)

```bash
python -m zipfile -e docs/benchmarks/structural-comparison-inputs.zip build/structural-comparison-inputs
node docs/benchmarks/replay-structural-comparison.mjs
```

[roof]: https://help.altair.com/hwsolvers/os/topics/solvers/os/scordelis_lo_roof_verification_r.htm
[straight]: https://help.altair.com/hwsolvers/os/topics/solvers/os/straight_cantilever_beam_verification_r.htm
[pinch]: https://docs.software.vt.edu/abaqusv2026/English/SIMACAEBMKRefMap/simabmk-c-pinchcyl.htm
[ko]: https://web.mit.edu/kjb/www/Principal_Publications/Performance_of_the_MITC3%2B_and_MITC4%2B_shell_elements_in_widely_used_benchmark_problems.pdf
[limits]: https://link.springer.com/article/10.1007/s11831-022-09798-5
[twist]: https://documentation-be.hexagon.com/bundle/MSC_Nastran_2022.2_Verification_Guide/raw/resource/enus/MSC_Nastran_2022.2_Verification_Guide.pdf
[adina]: https://docs.bentley.com/LiveContent/web/ADINA%20Examples-v26.00.0/Verification%20Manual/en/topics/verify/a_27_simply_supported_square_plate_under_uniform_pressure.html
[vessel]: https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/ans_vm/Hlp_V_VM25.html
[thermal]: https://ansyshelp.ansys.com/public/Views/Secured/corp/v251/en/wb_vm/wbvt_VM-WB-MECH-065.html
[modes]: https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/ans_vm/Hlp_V_VM50.html
[modes-input]: https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/ans_vm/Hlp_V_VM50TXT.html
[buckling]: https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/ans_vm/Hlp_V_VM127.html
[buckling-input]: https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/ans_vm/Hlp_V_VM127TXT.html
