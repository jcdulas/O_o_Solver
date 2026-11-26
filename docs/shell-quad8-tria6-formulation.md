# Eight-node and six-node shell elements — formulation

This document states the linear eight-node quadrilateral and six-node triangular shell elements that go with the
four-node element of [shell-quad4-formulation.md](shell-quad4-formulation.md) ("the quadrilateral document" below)
and the three-node element of [shell-tria3-formulation.md](shell-tria3-formulation.md) ("the triangle document"):
shells with a director at each grid, on the quadratic shape functions, with the conventions of these two documents
where they apply, so that a model may mix the four elements. Section 7 gives the results measured on the cases of the
triangle document, and sections 4 and 6 what the eight-node element does not take from them. Only what differs from
the two documents is stated here. The methods are those of the open
literature listed in section 9.

## 1. Data and conventions

- N grids $\mathbf X_a$: the corners, counter-clockwise, then the grids at the middle of the sides (N = 8: sides 1-2,
  2-3, 3-4, 4-1; N = 6: sides 1-2, 2-3, 3-1), every one of them given; six degrees of freedom per grid. The thickness
  is given at the corners, and is their mean at the middle of a side.
- Section stiffnesses $A, B, D, H$ per unit length, mass moments, material axis, offset and drilling factor $c_d$: as
  in section 1 of the quadrilateral document, with the same conventions for $\mathbf e_0$, $\boldsymbol\kappa$,
  $\mathbf N$, $\mathbf M$, $\mathbf Q$.
- A director $\boldsymbol\nu_a$ at each grid: the unit normal of the quadratic surface of the element there, or the
  normal the grid shares with its neighbours (section 3).

## 2. Element axes

The axes are those of the corners: $\mathbf e_z$ along the cross product of the diagonals of the quadrilateral (the
normal of the plane of the three corners of the triangle), $\mathbf e_x$ along the side 1-2 projected on the plane
normal to $\mathbf e_z$, $\mathbf e_y=\mathbf e_z\times\mathbf e_x$; the origin is the mean of the corners. The
material angle is measured from $\mathbf e_x$, and the results of a place are given along the projection of
$\mathbf e_x$ on the tangent plane of the surface there, as for the four-node and three-node elements.

An element whose grids are all within $10^{-8}$ of its size of the plane of its corners is flat: its grids are
projected on that plane. An element whose surface normal turns by more than 45° from $\mathbf e_z$ is refused.

## 3. Directors and kinematics

**Kinematics.** The reference surface and the fibres are interpolated from the grids by the quadratic shape functions
$N_a$ (the eight-node serendipity functions, the six-node functions of the triangle):

$$
\mathbf X(\xi,\eta,z)=\mathbf X_0+z\,\mathbf d,\quad \mathbf X_0=\sum_aN_a\mathbf X_a,\quad
\mathbf d=\sum_aN_a\boldsymbol\nu_a,\qquad
\mathbf u(\xi,\eta,z)=\mathbf u_0+z\,\mathbf b,\quad \mathbf u_0=\sum_aN_a\mathbf u_a,\quad
\mathbf b=\sum_aN_a\,\boldsymbol\theta_a\times\boldsymbol\nu_a .
$$

At a point, the local frame is $\mathbf e_3$ the unit normal of the reference surface, $\mathbf e_1$ the unit vector
along the projection of $\mathbf e_x$ on the tangent plane, $\mathbf e_2=\mathbf e_3\times\mathbf e_1$; with the
derivatives $\partial/\partial s_i$ along $\mathbf e_i$, the strains are those of a shell whose strains are linear
through the thickness:

$$
\varepsilon_{ij}=\tfrac12\big(\mathbf e_i\cdot\mathbf u_{0,j}+\mathbf e_j\cdot\mathbf u_{0,i}\big),\qquad
\gamma_i=\mathbf e_i\cdot\mathbf b+\mathbf d\cdot\mathbf u_{0,i},
$$

$$
k_{ij}=\tfrac12\big(\mathbf e_i\cdot\mathbf b_{,j}+\mathbf e_j\cdot\mathbf b_{,i}
+\mathbf d_{,i}\cdot\mathbf u_{0,j}+\mathbf d_{,j}\cdot\mathbf u_{0,i}\big)
-\tfrac12\big(\beta^T\varepsilon+\varepsilon\beta\big)_{ij},\qquad \beta_{ij}=\mathbf e_i\cdot\mathbf d_{,j},
$$

with $\mathbf e_0=(\varepsilon_{11},\varepsilon_{22},2\varepsilon_{12})$ and
$\boldsymbol\kappa=-(k_{11},k_{22},2k_{12})$. The last term of $k$ is the correction of the first-approximation shell
theory (Sanders): a uniform dilatation of the reference surface bends nothing. On a flat element whose directors are
its normal these are the strains of the plane element and of the plate of the quadrilateral document. The section
acts on these strains by its resultants (section 4 of the quadrilateral document).

**Directors shared at the grids.** As section 3 of the triangle document, for the four elements of a model
together: at each grid, the elements whose normals there make, in direction, an angle of at most $\alpha_n$ (20° by
default) are in one smooth group, and a group whose normals are not equal shares the normalised sum of its unit
normals; a normal may also be given at a grid. The normals of two elements are compared in direction: two neighbours
numbered in opposite senses are in one group, and each takes the shared normal on its own side. The own normals of
two quadratic elements that meet at a grid of a curved surface differ by the interpolation error of their surfaces,
a small angle: without shared normals the rotation about them is held by a stiffness smaller as the mesh is refined.

**Drilling.** Section 8 of the quadrilateral document with these directors, at each point:
$r=\omega-\theta_n$, $\omega=\tfrac12(\mathbf e_2\cdot\mathbf u_{0,1}-\mathbf e_1\cdot\mathbf u_{0,2})$,
$\theta_n=\sum_aN_a(\boldsymbol\theta_a-\bar{\boldsymbol\theta})\cdot\boldsymbol\nu_a+\bar{\boldsymbol\theta}\cdot\mathbf e_3$,
$\bar{\boldsymbol\theta}=\tfrac1N\sum\boldsymbol\theta_a$, and the stiffness $c_dA_{33}\,r^Tr$ per unit area for a
section with both a membrane and a bending stiffness; $c_d=10^{-6}k$, k = 100 by default, no term with $k\le0$. It is
integrated with the full rule of the element (3 × 3 points, seven points): with the 2 × 2 rule of the quadrilateral
the rotations $\xi^2-\eta^2$ of the grids at the middle of its sides about their directors, zero at its four points,
would have no stiffness across a whole mesh.

## 4. The eight-node quadrilateral

The strains are those of the displacements, and the stiffness is integrated with the 2 × 2 Gauss rule (uniform
reduced integration: Zienkiewicz, Taylor and Too). An element alone keeps two modes without energy beside its six
rigid motions, one of its membrane and one of its bending; two elements do not share them, so a mesh does not have
them. One element alone, wider than long, is soft in bending for that reason (section 7).

The eight functions hold the complete quadratic fields on a parallelogram only: a state of uniform curvature is
exact on parallelograms, and approximate on other quadrilaterals, the more the thinner the plate. On a mesh of
3 × 3 cells with their inner grids moved by 0.3 of a cell, held in a uniform curvature by its boundary: a deflection
off by $4\cdot10^{-4}$ and moments at the places off by 2 % for a plate of thickness a tenth of its side; a
deflection off by $3\cdot10^{-3}$ and moments off by 50 % for a thickness of a thousandth (the six-node triangle and
the four-node and three-node elements are exact there). Thin plates and shells of eight-node elements take cells
close to parallelograms. A state of uniform membrane strain is exact on any quadrilateral of straight sides.

## 5. The six-node triangle

The quadratic triangle has no reduced rule of full rank, and with the strains of its displacements it locks in the
bending of thin plates and of curved shells (section 7). Its covariant transverse shears and membrane strains are
tied: taken from their values at tying points, in the natural coordinates $(\xi,\eta)$ of the element, whose sides
are $\eta=0$, $\xi=0$ and $\xi+\eta=1$. With $r_1=\tfrac12-\tfrac1{2\sqrt3}$ and $r_2=\tfrac12+\tfrac1{2\sqrt3}$ the
Gauss points of a side:

**Transverse shears.** The covariant shears of the displacements are
$\tilde\gamma_\alpha=\mathbf X_{0,\alpha}\cdot\mathbf b+\mathbf d\cdot\mathbf u_{0,\alpha}$. The tied shears are the
field of eight terms

$$
\tilde\gamma_\xi=a_1+b_1\xi+c_1\eta+\eta\,(d\,\xi+e\,\eta),\qquad
\tilde\gamma_\eta=a_2+b_2\xi+c_2\eta-\xi\,(d\,\xi+e\,\eta),
$$

whose tangential component along each side is linear, which takes the eight values: $\tilde\gamma_\xi$ at
$(r_1,0)$ and $(r_2,0)$, $\tilde\gamma_\eta$ at $(0,r_1)$ and $(0,r_2)$, $\tilde\gamma_\eta-\tilde\gamma_\xi$ at
$(r_2,1-r_2)$ and $(r_1,1-r_1)$, and both shears at the centroid.

**Membrane strains.** The covariant membrane strains of the displacements are
$\tilde e_{\alpha\beta}=\tfrac12(\mathbf X_{0,\alpha}\cdot\mathbf u_{0,\beta}+\mathbf X_{0,\beta}\cdot\mathbf u_{0,\alpha})$.
The strain along each side, $\tilde e_{\xi\xi}$, $\tilde e_{\eta\eta}$ and
$\tilde e_{qq}=\tilde e_{\xi\xi}+\tilde e_{\eta\eta}-2\tilde e_{\xi\eta}$, is tied as a linear field
$a+b\,\xi+c\,\eta$ through its values at three points: the two Gauss points of its side, and the middle of the
segment the element cuts on the parallel to that side at the fraction $s=1/\sqrt3$ of the way to the opposite corner:

| Strain | Tying points |
|---|---|
| $\tilde e_{\xi\xi}$ | $(r_1,0)$, $(r_2,0)$, $(\tfrac{1-s}2,\ s)$ |
| $\tilde e_{\eta\eta}$ | $(0,r_1)$, $(0,r_2)$, $(s,\ \tfrac{1-s}2)$ |
| $\tilde e_{qq}$ | $(r_2,1-r_2)$, $(r_1,1-r_1)$, $(\tfrac{1-s}2,\ \tfrac{1-s}2)$ |

and $\tilde e_{\xi\eta}=\tfrac12(\tilde e_{\xi\xi}+\tilde e_{\eta\eta}-\tilde e_{qq})$.

**Local strains.** With $\mathbf g^\alpha$ the contravariant vectors of the reference surface at the point,

$$
\varepsilon_{ij}=\sum_{\alpha\beta}(\mathbf g^\alpha\cdot\mathbf e_i)(\mathbf g^\beta\cdot\mathbf e_j)\,\tilde e_{\alpha\beta},
\qquad \gamma_i=\sum_\alpha(\mathbf g^\alpha\cdot\mathbf e_i)\,\tilde\gamma_\alpha .
$$

The curvatures are those of the displacements. The stiffness is integrated with the seven-point rule, exact to the
fifth degree. On a triangle of straight sides the membrane strains of the displacements are linear and the tied ones
are the same; a state of uniform membrane strains and curvatures is exact in the element. The six rigid motions are
its only modes without energy.

**The inner point.** The fraction s is the one that gives the best curved shells among those measured (two triangles
per cell, cut along the diagonal 1-3 and alternating; the pinched hemisphere and the barrel vault of section 7):

| s | Hemisphere 4 × 4 | 8 × 8 | 16 × 16 | Vault 4 × 4 | 8 × 8 |
|---|---|---|---|---|---|
| 1/3 (the centroid) | 0.09, 0.19 | 0.34, 0.50 | 0.86, 0.92 | 0.84, 0.91 | 0.981, 0.991 |
| 0.5 | 0.52, 0.62 | 0.87, 0.89 | 0.983, 0.985 | 0.978, 0.987 | 0.996, 0.998 |
| 0.55 | 0.78, 0.77 | 0.946, 0.932 | 0.990, 0.988 | 0.993, 0.996 | 0.999, 0.999 |
| 1/√3 | 0.81, 0.79 | 0.949, 0.933 | 0.990, 0.988 | 0.997, 0.999 | 1.000, 1.000 |
| 0.6 | 0.74, 0.76 | 0.930, 0.925 | 0.988, 0.987 | 0.998, 1.001 | 1.001, 1.001 |
| 0.7 | 0.32, 0.48 | 0.73, 0.82 | 0.966, 0.975 | 0.986, 1.005 | 1.003, 1.003 |
| membrane strains and shears of the displacements | 0.01, 0.01 | 0.13, 0.17 | 0.68, 0.74 | 0.65, 0.71 | 0.956, 0.966 |

(the last row with the seven-point rule; with the three-point rule that went with these strains before: 0.05 and
0.07, 0.38 and 0.44, 0.88 and 0.91 on the hemisphere). The values are those of a program of the formulation written
apart from the solver, whose eigenvalues section 8 gives.

## 6. Stiffness, mass, loads, resultants

- **Section without shear material:** the transverse shear stiffness is that of the bending material,
  $H=\kappa\,G\,t$ with G its in-plane shear modulus in the material axes (the term (3, 3) of the bending matrix
  there, $G_{12}$ of an orthotropic layer: not that term in the element axes, which turns with the element) and
  $\kappa$ the shear factor of the section: negligible for a thin plate. This differs from the two documents, whose
  elements give such a section
  $H=c_K\,12\max(D_{11},D_{22})/S$: that stiffness makes a plate of any thickness a thin one for its elements, and the
  eight-node quadrilateral does not hold the uniform curvature of a thin plate on irregular quadrilaterals (section
  4). Measured on the mesh of section 4 for a plate of thickness a tenth of its side: moments at the places off by
  2 % with the shear of the material, by 50 % with $c_K$. The six-node triangle keeps the same rule.
- **Mass:** with the full rule; the consistent mass of the translations and the rotary inertia of the fibres in the
  coupled mass; the lumped mass is the diagonal of the consistent mass of the translations scaled to the mass of the
  element (Hinton, Rock and Zienkiewicz), without rotary inertia.
- **Pressure:** with the full rule, the corner pressures interpolated linearly, along the normal of the surface at
  each point or along a given direction.
- **Thermal loads:** the resultants of the free thermal strain with the rule of the stiffness. As in the two
  documents, the dilatation $\bar\varepsilon$ of the free strain (the mean over the element of half the trace of its
  free membrane strain) is taken apart: its loads are $K\,\mathbf q_h$ for the homothety
  $\mathbf q_h=\bar\varepsilon\,\mathbf x$ of the element. A free uniform temperature rise is then that homothety,
  without stress, on a curved element and with shared directors, whose transverse shear in a homothety the
  correction of section 3 does not remove (the displacements of a free quarter cylinder were off the homothety by
  $2\cdot10^{-5}$ to $9\cdot10^{-5}$ of $\alpha\Delta T$ without it; $2\cdot10^{-9}$ with it). As the pressure, the
  thermal loads are carried to the grids through $\mathsf T_f$ of section 11 of the quadrilateral document: an
  element with an offset is free of stress under that rise only when the model puts the loads on the offset; by
  default its loads are taken at the grids, without the moments of its forces about them, and it bends (section 12
  of that document).
- **Differential stiffness:** that of the membrane forces of the centre on the gradients of the translations, with
  the rule of the stiffness.
- **Resultants:** at the centre and at the corners, in the local frame of the place. For the triangle, from its tied
  strains at the place. For the quadrilateral, from the strains at the four points of its 2 × 2 rule, where the
  strains of a reduced integration are the most accurate (Barlow), carried to the place by the bilinear functions
  through these points, $\tfrac14(1+3\xi\xi_g)(1+3\eta\eta_g)$. The strains of a place are those of the displacements
  less what the strains of the homothety of the free dilatation are off $(\bar\varepsilon,\bar\varepsilon,0)$.

## 7. Results measured

The cases of section 6 of the triangle document, run through the solver
(`wasm/test/validation/classic-shells.test.mjs`) with one eight-node quadrilateral per cell, and with two six-node
triangles per cell cut along the diagonal 1-3 or alternating, the grids of the middles of the sides on the surface;
default drilling factor and shared normals. The values are the result over the reference; for the triangles, the
range over the two ways of cutting. The last column is the six-node triangle with the strains of its displacements
and the three-point rule, before its strains were tied.

| Case | Mesh (cells) | Eight-node quadrilaterals | Six-node triangles | Triangles, not tied |
|---|---|---|---|---|
| Cantilever strip 10 × 1, t = 0.1, ν = 0, tip force: deflection over the beam's | 1, 2, 4, 8 along | 1.0000 to 1.0001 | 0.751, 0.942 to 0.944, 0.997 to 0.998, 1.001 | 0.754, 0.942, 0.989 to 0.990, 0.999 |
| The same with a section without shear material, over the beam's without shear deflection | 1, 4 along | 1.0001 | 0.751, 0.997 to 0.998 | 0.754, 0.989 to 0.990 |
| The strip in torsion: rotation over T L / (G b t³ / 3) | 10 × 1, 20 × 2, 40 × 4 | 0.996, 1.025, 1.050 | 0.988 to 0.991, 1.009 to 1.011, 1.030 to 1.032 | 0.986 to 1.022, 1.015 to 1.022, 1.043 to 1.047 |
| The strip bent in its plane: deflection over the beam's | 4 × 1, 8 × 1 | 0.999, 0.999 | 0.989, 0.997 | the same |
| Simply supported square plate, a/t = 100, pressure: centre deflection over 0.004062 q a⁴ / D | 8 × 8, 16 × 16 | 1.0005, 1.0006 | 1.007 to 1.011, 1.003 | 0.946 to 0.972, 0.995 to 0.996 |
| Clamped square plate, a/t = 100, pressure: over 0.001265 q a⁴ / D | 8 × 8, 16 × 16 | 0.995, 1.002 | 1.016 to 1.029, 1.011 to 1.012 | 0.846 to 0.901, 0.985 to 0.991 |
| Barrel vault under its weight (Scordelis and Lo): deflection of the free edge over 0.3024 | 8 × 8, 16 × 16 | 1.001, 0.999 | 1.001 to 1.003, 0.998 | 0.984 to 0.987, 0.997 |
| Pinched hemisphere with a hole (MacNeal and Harder): displacement over 0.0940 | 8 × 8, 16 × 16 | 0.992, 0.997 | 0.933 to 0.949, 0.988 to 0.990 | 0.38 to 0.44, 0.88 to 0.91 |
| Twisted beam (MacNeal and Harder), t = 0.32: tip displacement over 0.005424 (force in the plane of the tip section) and over 0.001754 (force across it) | 12 × 2, 24 × 4 | 0.998 and 0.999, 0.998 and 0.999 | 0.998 and 0.998, 0.999 and 0.999 | 0.995 and 0.996, 0.998 and 0.999 |
| Free quarter cylinder under a temperature rise: displacements off the homothety, over α ΔT | 8 × 8 | 1·10⁻⁹ (2·10⁻⁵ before) | 2·10⁻⁹ | 2·10⁻⁵ to 9·10⁻⁵ |
| Simply supported square plate, a/t = 100: first four frequencies over the thin plate's | 16 × 16 | 0.994 to 0.996 | 0.992 to 0.997 | 0.998 to 1.010 |

What the table shows:

1. **The eight-node quadrilateral** is within 1 % of the reference on every case from the coarsest mesh, the torsion
   of the strip apart (whose reference leaves out the factor 1 − 0.63 t / b: the bar of this section turns 7 % more).
   Its displacements are those it had before this document within $10^{-6}$: the drilling term and the normals of
   section 3 are a stabilisation.
2. **The six-node triangle in bending.** Its deflection is quadratic along a beam: one cell gives three quarters of
   the deflection of the cantilever, as one quadratic beam element does, and four cells 0.997. Tying the shears
   removes the locking of the plates (a clamped plate of 8 × 8 cells: 0.85 to 0.90 before, 1.02 to 1.03 now); the
   triangle now comes to the plate from above.
3. **The six-node triangle on curved shells.** Tying the membrane strains removes the locking of the hemisphere
   (8 × 8 cells: 0.38 to 0.44 before, 0.93 to 0.95 now) and brings the vault within 0.3 % at 8 × 8. At 16 × 16 it is
   within 1.3 % of the reference on every case.
4. **One element alone.** One eight-node quadrilateral wider than long is soft in bending, by the mode of its reduced
   rule (a cantilever of one cell of 1 × 1: 1.06; of 0.5 × 1: 1.90; four cells along: 1.001 and 1.017).
5. **Drilling factor and normals.** On the hemisphere of 8 × 8 cells, the drilling factors 0, $10^{-4}$ and
   $4\cdot10^{-4}$ give the same displacement within $3\cdot10^{-5}$, and the shared normals, the own normals of the
   elements and the normals of the sphere given at every grid the same within $5\cdot10^{-5}$.

## 8. Checks of an implementation

The triangle with the grids (0, 0), (1, 0.1), (0.2, 0.9), (0.5, 0.05), (0.6, 0.5), (0.1, 0.45) in the plane z = 0 and
the directors (0, 0, 1), then with the same grids on the surface $z=0.3x^2-0.2xy+0.15y^2$ and the directors along
$(-0.6x+0.2y,\ 0.2x-0.3y,\ 1)$; the section of section 14 of the quadrilateral document (E = 1000, ν = 0.3, t = 0.05,
shear factor 5/6) and $c_d=4\cdot10^{-4}$. The stiffness has six zero eigenvalues; its thirty others are

| Triangle | Eigenvalues |
|---|---|
| flat | 1.0302017488e-04, 1.0302355163e-04, 1.2677982131e-04, 3.2935952021e-04, 3.2940027670e-04, 3.6463043352e-03, 3.7386196036e-03, 4.1451505498e-03, 5.1601610265e-03, 6.3515575223e-03, 2.3295301698e-02, 2.7112900382e-02, 3.9404288983e-02, 3.2238012652e-01, 3.4791540961e-01, 1.1246721615, 6.7170271604, 7.5442365300, 8.9151509545, 12.365821688, 20.628986675, 25.071946027, 40.427067946, 44.496940019, 49.248788252, 60.887553700, 78.620768740, 124.20085128, 142.06316405, 192.17600848 |
| curved | 1.0466005218e-04, 1.1073214702e-04, 1.2200014831e-04, 3.3835105705e-04, 3.4996839290e-04, 2.8265818096e-03, 3.3146152445e-03, 4.5014643338e-03, 5.2545851524e-03, 7.1340433407e-03, 2.2332154047e-02, 2.6686492958e-02, 3.9923731347e-02, 3.3473684575e-01, 3.5071952303e-01, 1.2696040811, 6.3466570477, 7.2568882960, 8.7748897042, 12.488558896, 19.306719051, 26.228696892, 38.923552657, 43.834977468, 49.873838588, 63.848962229, 79.281887474, 125.44212292, 143.71456798, 197.38813743 |

Other checks, for both elements, flat and curved: a rigid motion gives no force; the stiffness of the triangle has
six zero eigenvalues and that of the quadrilateral eight; the loads of a uniform temperature rise are the forces
$K\,\mathbf q_h$ of its homothety, and the element in that homothety has no force, moment nor shear force at any
place; a state of uniform membrane strains and curvatures is recovered at every place of a triangle of straight
sides and of a parallelogram, and its membrane strains at every place of any quadrilateral of straight sides.

## 9. Literature

- The literature of the quadrilateral and triangle documents, for the directors defined at the nodes of the mesh, the
  rotation about the normal and the reference values of section 7.
- S. Ahmad, B. M. Irons, O. C. Zienkiewicz, Analysis of thick and thin shell structures by curved finite elements,
  Int. J. Numer. Methods Eng. 2 (1970): the shell with a director at each node.
- O. C. Zienkiewicz, R. L. Taylor, J. M. Too, Reduced integration technique in general analysis of plates and shells,
  Int. J. Numer. Methods Eng. 3 (1971): the uniform reduced integration of the eight-node element.
- T. J. R. Hughes, M. Cohen, The "heterosis" finite element for plate bending, Comput. Struct. 9 (1978): the modes of
  the reduced rule, which a mesh does not have.
- K.-J. Bathe, E. N. Dvorkin, A formulation of general shell elements — the use of mixed interpolation of tensorial
  components, Int. J. Numer. Methods Eng. 22 (1986): covariant strains tied at points.
- P.-S. Lee, K.-J. Bathe, Development of MITC isotropic triangular shell finite elements, Comput. Struct. 82 (2004):
  the tied transverse shears and membrane strains of the six-node triangle.
- J. Barlow, Optimal stress locations in finite element models, Int. J. Numer. Methods Eng. 10 (1976): the points of
  the reduced rule for the strains.
- J. L. Sanders, An improved first-approximation theory for thin shells, NASA TR R-24 (1959); B. Budiansky,
  J. L. Sanders, On the "best" first-order linear shell theory, Progress in Applied Mechanics (1963): the bending
  strain of section 3.
- E. Hinton, T. Rock, O. C. Zienkiewicz, A note on mass lumping and related processes in the finite element method,
  Earthquake Eng. Struct. Dyn. 4 (1976): the lumped mass.
