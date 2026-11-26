# Three-node triangular shell element with 18 degrees of freedom — formulation

This document states the linear three-node shell element that goes with the four-node element of
[shell-quad4-formulation.md](shell-quad4-formulation.md) ("the quadrilateral document" below): it is that element on
three nodes, with the same kinematics, section, conventions and constants where they apply, so that two triangles on
the grids of a quadrilateral behave closely to it. Section 6 gives the closeness measured and its limits. The
notation is that of the quadrilateral document; only what differs is stated here. Section 3 also states how the
directors of both elements are shared at the grids of a model. The methods are those of the open literature listed
in section 8.

## 1. Data and conventions

- Three grids $\mathbf X_i$ (i = 1..3, counter-clockwise) with the thicknesses $t_i$; six degrees of freedom per grid.
- Section stiffnesses $A, B, D, H$, mass moments, material axis, offset and drilling factor $c_d$: as in section 1 of
  the quadrilateral document, with the same conventions for $\mathbf e_0$, $\boldsymbol\kappa$, $\mathbf N$,
  $\mathbf M$, $\mathbf Q$.
- A director $\mathbf n_i$ at each grid: the normal of the triangle, or the normal the grid shares with its
  neighbours (section 3).

## 2. Element axes

A triangle is taken as the half of a parallelogram, and has the axes and the widths the quadrilateral document gives
that parallelogram.

Let O be the grid opposite the longest side (the earlier side in the order 1-2, 2-3, 3-1 when two sides are equal),
and A, B the ends of that side in the order of the grids. With the sides $\mathbf l_A=\mathbf X_A-\mathbf X_O$ and
$\mathbf l_B=\mathbf X_B-\mathbf X_O$, the parallelogram has the corners O, A, O + $\mathbf l_A$ + $\mathbf l_B$, B and
the diagonals $\mathbf d_{13}=\mathbf l_A+\mathbf l_B$, $\mathbf d_{24}=\mathbf l_B-\mathbf l_A$:

$$
\mathbf e_z=\frac{\mathbf l_A\times\mathbf l_B}{\|\mathbf l_A\times\mathbf l_B\|},\qquad
\mathbf e_x=\frac{\hat{\mathbf d}_{13}-\hat{\mathbf d}_{24}}{\|\hat{\mathbf d}_{13}-\hat{\mathbf d}_{24}\|},\qquad
\mathbf e_y=\mathbf e_z\times\mathbf e_x ,
$$

origin $\mathbf X_c=\tfrac13\sum\mathbf X_i$, local coordinates $\mathbf x_i=T(\mathbf X_i-\mathbf X_c)$. The widths
are those of the parallelogram,

$$
a=|\mathbf l_A\cdot\mathbf e_x|,\qquad b=|\mathbf l_B\cdot\mathbf e_y| .
$$

On a right triangle $\mathbf e_x$ and $\mathbf e_y$ are along the sides of the right angle, and a, b are their
lengths.

Reference direction: $\mathbf X_2-\mathbf X_1$. The material direction is this direction turned by φ about
$\mathbf e_z$, or the projection of the given direction on the element plane.

## 3. Directors and kinematics

**Kinematics.** Those of section 3 of the quadrilateral document with the linear shape functions
$N_1=1-\xi-\eta$, $N_2=\xi$, $N_3=\eta$:

$$
\mathbf X(\xi,\eta,\zeta)=\sum_iN_i\big(\mathbf x_i+\zeta\tfrac{t_i}2\mathbf n_i\big),\qquad
\mathbf u(\xi,\eta,\zeta)=\sum_iN_i\big(\mathbf u_i+\zeta\tfrac{t_i}2\,\boldsymbol\theta_i\times\mathbf n_i\big),
$$

the covariant vectors, the displacement gradient L with its three rows kept, and the local frame of a point, which
is the element frame (the reference surface is flat). The membrane strains and the curvatures are those of section 5
of the quadrilateral document; they are constant over a triangle whose directors are its normal, and there
$\boldsymbol\kappa=(\theta_{y,x},\ -\theta_{x,y},\ \theta_{y,y}-\theta_{x,x})$. There are no enhanced strains: a
constant strain has no use for them.

**Directors shared at the grids.** Each element has a normal at each of its grids: that of its bilinear surface
there for a quadrilateral, its own for a triangle. On a curved surface the elements that meet at a grid have normals
that differ by the angle of their facets, and the rotation about the normal of one is a bending rotation for the
others: they hold it by their bending alone, the less the finer the mesh against the thickness, and the surface
softens as the mesh is refined (a twisted beam of triangles gives 1.19, 1.38 and 1.89 times its tip displacement at
12 × 2, 24 × 4 and 48 × 8 cells; a thick vault of quadrilaterals drifts by 4 % over the same refinements). So the
directors are defined at the grids of the mesh, for the quadrilaterals and the triangles of a model together (and
the eight-node and six-node elements of [shell-quad8-tria6-formulation.md](shell-quad8-tria6-formulation.md)):

- at each grid the elements are put in smooth groups: two elements whose normals there make, in direction, an angle
  of at most $\alpha_n$ (20° by default) are in one group, and the groups are the connected sets of that relation.
  A fold, a junction of three walls or a stiffener on a skin keeps one normal per smooth side; two neighbours
  numbered in opposite senses are in one group;
- a group of one element, a group whose normals are equal (a flat mesh), or a group one of whose normals is farther
  than $\alpha_n$ from their mean keeps the normal of each element;
- any other group shares the normalised sum of the unit normals of its elements (each turned to the side of the
  first), and each element takes it on the side of its own normal.

The rotation of a grid about its shared normal then moves nothing in any element of the group, and the two other
rotations are bending rotations for all of them. With $\alpha_n\le0$ no normal is shared. A normal may also be given
at a grid: the elements whose own normal there is within $\alpha_n$ of it, on either side, take it; a zero vector
leaves every element of the grid its own normal.

**Drilling.** Section 8 of the quadrilateral document with these directors:
$r=\omega-\theta_n$, $\theta_n=\sum_iN_i(\boldsymbol\theta_i-\bar{\boldsymbol\theta})\cdot\mathbf n_i+\bar{\boldsymbol\theta}\cdot\mathbf e_3$,
$\bar{\boldsymbol\theta}=\tfrac13\sum\boldsymbol\theta_i$. The drilling factor is a datum of the model, $c_d=10^{-6}k$
with k = 100 by default; with $k\le0$ there is no drilling term.

## 4. Transverse shear

**Tied strains.** The covariant shears of section 7 of the quadrilateral document,
$\tilde\gamma_\alpha=\mathbf g_3\cdot\mathbf u_{,\alpha}+\mathbf g_\alpha\cdot\sum_iN_i\tfrac{t_i}2\,\boldsymbol\theta_i\times\mathbf n_i$,
are tied at the mid-sides. The field is the one of lowest order whose component along each side is constant on it:

$$
\tilde\gamma_\xi=a+c\,\eta,\qquad\tilde\gamma_\eta=b-c\,\xi ,
$$

with $a=\tilde\gamma_\xi(\tfrac12,0)$ on side 1-2, $b=\tilde\gamma_\eta(0,\tfrac12)$ on side 3-1, and c from the
shear along side 2-3 at its middle, $(\tilde\gamma_\eta-\tilde\gamma_\xi)(\tfrac12,\tfrac12)=b-a-c$. The local
components $\gamma_x,\gamma_y$ of a point follow by the formula of that section.

On a triangle whose directors are its normal the field is $\boldsymbol\gamma(x,y)=(\bar\gamma_x-q\,y,\ \bar\gamma_y+q\,x)$,
x and y from the centroid: $\bar{\boldsymbol\gamma}$ is its mean and q its rotational part, and by the circulation of
$\boldsymbol\gamma$ around the element $q=-\tfrac12(\theta_{x,x}+\theta_{y,y})$: q measures how far the rotations are
from the slopes of a deflection, and vanishes with the transverse shear in a thin plate.

**Factors.** With the section of the point, $s(z)=z/(1+z)$, $D^*_{kk}$ as in section 7 of the quadrilateral document
and the widths a, b of section 2:

$$
s_{x0}=s\Big(\frac{12D^*_{11}}{H_{11}a^2}\Big),\qquad
s_{x1}=s\Big(\frac{12\,(D^*_{11}+c_tD^*_{33})}{H_{11}a^2}\Big),
$$

$$
s_{y0}=s\Big(\frac{12D^*_{22}}{H_{22}b^2}\Big),\qquad
s_{y1}=s\Big(\frac{12\,(D^*_{22}+c_tD^*_{33})}{H_{22}b^2}\Big).
$$

$s_{x0}$ and $s_{y0}$ are those of the quadrilateral: the residual bending flexibility of a beam segment of length a
or b. With them a strip of right triangles whose cells are longer than wide bends as the beam (section 6).

The rotational part cannot take the factors of the linear part of the quadrilateral's shear. In the quadrilateral that
part is free of the rotations of a constant twist, and $c_t=99$ leaves it nearly rigid. In a triangle q is a
constraint on the rotations themselves, $\theta_{x,x}+\theta_{y,y}=0$, which the linear rotations of a coarse mesh
cannot meet together with the two mean shears. With a nearly rigid q a coarse mesh is too stiff: with $c_t=99$ a
clamped plate of 4 × 4 cells under a pressure gives 0.50 to 0.63 of its deflection (0.80 to 0.95 with the value
below). With a free q a strip one cell wide is too soft in torsion: with $c_t=0$ the strip of section 6 turns 1.23
times the reference (1.13 with the value below). The triangle takes

$$
c_t=8
$$

on the width of the direction itself; section 6 gives what it leaves.

**Modified strains.** With $\bar{\boldsymbol\gamma}$ the mean of the local shears of the three integration points:

$$
\gamma'_x=\sqrt{s_{x0}}\,\bar\gamma_x+\sqrt{s_{x1}}\,(\gamma_x-\bar\gamma_x),\qquad
\gamma'_y=\sqrt{s_{y0}}\,\bar\gamma_y+\sqrt{s_{y1}}\,(\gamma_y-\bar\gamma_y),\qquad\boldsymbol\gamma'=B_s\mathbf d .
$$

**Without transverse shear flexibility:** $H=c_K\,\dfrac{12\max(D_{11},D_{22})}{j}\begin{bmatrix}1&0\\0&1\end{bmatrix}$,
j the area of the parallelogram (the 4j of the quadrilateral).

## 5. Stiffness, mass, loads, resultants

Integration at the three mid-sides, of weight w = j/6 each (exact for a polynomial of degree 2), with the section at
the thickness $t=\sum N_it_i$ of the point.

- **Stiffness.** With $\mathsf B$ the rows of $(\mathbf e_0,\boldsymbol\kappa)$ and
  $\mathsf C=\begin{bmatrix}A&B\\B&D\end{bmatrix}$:

$$
K=\sum_pw\Big(\mathsf B^{\mathsf T}\mathsf C\mathsf B+B_s^{\mathsf T}HB_s+c_dA_{33}\,\mathbf b_d^{\mathsf T}\mathbf b_d\Big).
$$

  A section without bending stiffness has no shear term; the drilling term is present when the section has both a
  membrane and a bending stiffness.
- **Mass.** Section 10 of the quadrilateral document:
  $M=\sum_pw\,(I_0N_u^{\mathsf T}N_u+I_2N_\beta^{\mathsf T}N_\beta)$ for a homogeneous section ($I_0=\rho t$ plus
  the non-structural mass, $I_2=\rho t^3/12$). Lumped form: on the three translations of grid i,
  $m_i=\sum_pw\,I_0N_i$; nothing on the rotations.
- **Offset and global system.** As section 11 of the quadrilateral document.
- **Pressure** p along the element normal: $\mathbf f_i=\sum p\,N_i\,j/6\ \mathbf e_z$ on the translations of grid i,
  at the three interior points $(\tfrac16,\tfrac16),(\tfrac16,\tfrac23),(\tfrac23,\tfrac16)$, p interpolated from the
  values at the grids.
- **Initial (thermal) resultants.** The free thermal strain is split into its dilatation ε (the mean of its two
  in-plane strains) and the rest. The rest gives $\mathbf f=\sum_pw\,\mathsf B^{\mathsf T}\mathsf C\,\boldsymbol\varepsilon_{th}$
  as in section 12 of the quadrilateral document. The dilatation gives $\mathbf f=\varepsilon\,K\mathbf d_h$,
  $\mathbf d_h$ the homothety of the element (the positions $\mathbf x_i$ on the translations, no rotation): a free
  thermal dilatation is the homothety $\mathbf u=\varepsilon\mathbf x$, and leaves the element without stress in
  every one of its strain measures, whatever its directors. The quadrilateral takes the same rule, so that a warped
  mesh, or a mesh whose normals are shared, expands freely under a uniform temperature rise. As the other loads, the
  thermal loads are carried to the grids through $\mathsf T_f$ of section 11 of the quadrilateral document: an
  element with an offset is free of stress under that rise only when the model puts the loads on the offset; by
  default its loads are taken at the grids and it bends (section 12 of that document).
- **Resultants** at the centroid, at the mean thickness:
  $\begin{Bmatrix}\mathbf N\\\mathbf M\end{Bmatrix}=\mathsf C\,(\mathsf B\mathbf d-\boldsymbol\varepsilon_{th})$, the
  dilatation of $\boldsymbol\varepsilon_{th}$ taken as ε times the strains of the homothety, and the transverse shear
  forces conjugate to the mean shear in the energy of the element,
  $\mathbf Q=\mathsf S H\mathsf S\,\bar{\boldsymbol\gamma}$ with $\mathsf S=\operatorname{diag}(\sqrt{s_{x0}},\sqrt{s_{y0}})$.
- **Differential stiffness** of the membrane forces: as section 12 of the quadrilateral document, with the constant
  gradients of the translations and the area j/2.

## 6. Two triangles against one quadrilateral

The cases below are run through the solver with quadrilaterals and with two triangles on the grids of each
quadrilateral, cut along the diagonal 1-3, along 2-4, or alternating from cell to cell
(`wasm/test/validation/classic-shells.test.mjs`), with the default drilling factor and shared normals. The values are
the result over the reference; for the triangles, the range over the three ways of cutting.

| Case | Mesh (cells) | Quadrilaterals | Two triangles per cell |
|---|---|---|---|
| Cantilever strip 10 × 1, t = 0.1, ν = 0, tip force: deflection over the beam's | 1, 2, 4, 8 along | 1.0000 | 1.0000 to 1.0003 |
| The same without transverse shear flexibility | 1, 4 along | 1.0000 | 1.0000 to 1.0001 |
| The same, one cell of 2 × 1, 1 × 1, 0.5 × 1 (length × width) | 1 | 1.000 | 1.004, 1.031, 1.101 |
| The same, four cells of 2 × 1, 1 × 1, 0.5 × 1 | 4 along | 1.000 | 1.0001 to 1.0002, 1.0006 to 1.0016, 1.0032 to 1.0044 |
| The strip in torsion: rotation over T L / (G b t³ / 3) | 10 × 1, 20 × 2, 40 × 4 | 1.006, 1.028, 1.036 | 1.130, 1.071, 1.041 to 1.043 |
| The strip bent in its plane: deflection over the beam's | 8 × 1 | 0.995 | 0.19 to 0.21 |
| | 40 × 4, 80 × 8 | | 0.82 to 0.84, 0.95 to 0.96 |
| Simply supported square plate, a/t = 100, pressure: centre deflection over 0.004062 q a⁴ / D | 8 × 8, 16 × 16 | 1.017, 1.006 | 0.955 to 0.964, 0.990 to 0.992 |
| Clamped square plate, a/t = 100, pressure: over 0.001265 q a⁴ / D | 8 × 8, 16 × 16 | 1.059, 1.021 | 0.966 to 0.976, 0.994 to 0.997 |
| Barrel vault under its weight (Scordelis and Lo): deflection of the free edge over 0.3024 | 8 × 8, 16 × 16 | 1.012, 1.002 | 0.857 to 0.883, 0.957 to 0.965 |
| Pinched hemisphere with a hole (MacNeal and Harder): displacement over 0.0940 | 8 × 8, 16 × 16 | 1.007, 0.998 | 0.991 to 0.993, 0.987 to 0.989 |
| Twisted beam (MacNeal and Harder), t = 0.32: tip displacement over 0.005424 (force in the plane of the tip section) and over 0.001754 (force across it) | 12 × 2, 24 × 4 | 0.996 and 0.987, 0.997 and 0.994 | 0.978 and 0.831 to 0.841, 0.990 and 0.922 to 0.931 |
| Simply supported square plate, a/t = 100: first four frequencies over the thin plate's | 16 × 16 | 0.998, 1.005, 1.005, 1.001 | 1.002 to 1.003, 1.010 to 1.013, 1.013 to 1.020, 1.024 to 1.026 |

What the table shows:

1. **Bending.** A strip of triangles whose cells are at least twice as long as wide bends as the beam within 0.4 %
   with any number of elements, one included, where the quadrilateral is exact. A single cell is softer as it widens
   (3 % for a square cell, 10 % for a cell twice as wide as long); with four cells along, 0.2 % and 0.4 %.
2. **Plates and the hemisphere.** The triangles converge from below and the quadrilaterals from above; at 16 × 16
   cells both are within 2.2 % of the reference, and within 3 % of each other.
3. **Torsion of a strip.** With one cell across the width the triangles turn 13 % more than the reference, where the
   quadrilateral is within 1 %; 7 % with two cells across, 4 % with four, as the quadrilateral then. (The bar of this
   section turns 7 % more than the reference, whose J = b t³ / 3 leaves out the factor 1 − 0.63 t / b, less the
   restraint of its clamped end.)
4. **Bending in the plane of the element.** This is the limit of the three-node triangle. Its membrane strains are
   constant, where the enhanced strains of the quadrilateral bend with one element across the depth: a web or a
   flange meshed with one row of triangles is five times too stiff in its plane, and eight cells across the depth
   are needed for 5 %. The barrel vault, whose response is carried by its membrane, shows the same slower
   convergence, and the twisted beam under the force across its tip, which bends its root in its plane. Where a
   part bends in its plane, it takes quadrilaterals, or several triangles across its depth.
5. **Warped and thick surfaces.** With the shared normals of section 3 the triangles converge on the twisted beam,
   whose quadrilaterals are all warped (0.978, then 0.990 under the force in the plane of the tip), and a twisted
   stiffened panel whose cells come to the size of the thickness of its skin
   (`wasm/test/validation/classic-stiffened-panel.test.mjs`: stringers, frames and edge caps as line elements) is
   within 0.3 % of the same panel with webs of shells under bending loads, with quadrilaterals, with triangles or
   with both. Without shared normals the triangles give 4 to 29 % too much displacement there, more as the mesh is
   refined. The results do not depend on the drilling factor between 0 and 4·10⁻⁴.

## 7. Checks of an implementation

1. Six zero-energy modes and no other; $K\boldsymbol\Phi=0$ for the six rigid motions to round-off, with directors
   along the normal of the triangle or up to 20° off it.
2. Constant membrane, bending and coupled states are reproduced exactly on any flat mesh (patch test), for any
   section.
3. A free thermal expansion and its free curvature are balanced by the thermal loads and leave no resultant; a free
   quarter of a cylinder under a uniform temperature rise takes the displacements of the homothety to round-off,
   with shared normals.
4. One cell of 2 × 1 (length × width), ν = 0, t = 0.05, clamped at one end and loaded at the other: the two triangles
   give 1.003744 times the deflection $PL^3/(3EI)+PL/(GA_s)$ that one quadrilateral gives exactly, on either
   diagonal.
5. Numerical values. The section of section 14 of the quadrilateral document (E = 1000, ν = 0.3, t = 0.05,
   $c_d=4\cdot10^{-4}$) and $c_t=8$. Grids (0, 0, 0), (1.2, 0.1, 0), (0.3, 0.9, 0.2): a = 1.190020, b = 0.941215.
   The twelve non-zero eigenvalues of the 18 × 18 stiffness and the trace of the consistent mass:

| Section and directors | Eigenvalues of K | trace M |
|---|---|---|
| with $H=\frac56\frac{Et}{2(1+\nu)}\mathbf I$, the normal of the triangle | 3.452772614·10⁻⁴, 3.452772614·10⁻⁴, 4.584678133·10⁻³, 5.950790620·10⁻³, 6.213710696·10⁻³, 1.436811758·10⁻², 2.426647736·10⁻², 8.562176106·10⁻², 2.097010749·10⁻¹, 29.79552950, 34.63211936, 69.15264118 | 4.040305034·10⁻² |
| without transverse shear flexibility, the normal of the triangle | 3.452772614·10⁻⁴, 3.452772614·10⁻⁴, 4.584678133·10⁻³, 5.950907217·10⁻³, 6.220952221·10⁻³, 1.436816311·10⁻², 2.502438656·10⁻², 8.613741876·10⁻², 2.116889453·10⁻¹, 29.79552950, 34.63211936, 69.15264118 | 4.040305034·10⁻² |
| with H, the directors (0.1163474672, −0.2676832426, 0.9564564541), (−0.0625593640, −0.1046684690, 0.9925375749), (0.0493105131, −0.1348232982, 0.9896419310) | 3.412781254·10⁻⁴, 3.450049511·10⁻⁴, 4.378774519·10⁻³, 5.975466350·10⁻³, 6.544627910·10⁻³, 1.462007979·10⁻², 2.443904227·10⁻², 8.569121971·10⁻², 2.105586843·10⁻¹, 29.79652551, 34.63327897, 69.15416522 | 4.040305034·10⁻² |

## 8. Literature

- The literature of the quadrilateral document, for the kinematics, the residual bending flexibility of the
  transverse shear and the rotation about the normal.
- T. J. R. Hughes, W. K. Liu, Nonlinear finite element analysis of shells: part I. Three-dimensional shells, Comput.
  Methods Appl. Mech. Eng. 26 (1981); K.-J. Bathe, Finite Element Procedures (1996), section 5.4: the directors
  defined at the nodes of the mesh.
- J.-C. Nédélec, Mixed finite elements in R³, Numer. Math. 35 (1980): the lowest-order field with continuous
  tangential components.
- P.-S. Lee, K.-J. Bathe, Development of MITC isotropic triangular shell finite elements, Comput. Struct. 82 (2004):
  covariant transverse shear strains of a triangular shell tied at the mid-sides.
- J. L. Sanders, An improved first-approximation theory for thin shells, NASA TR R-24 (1959): a uniform dilatation
  bends nothing.
- A. C. Scordelis, K. S. Lo, Computer analysis of cylindrical shells, J. Am. Concrete Inst. 61 (1964); R. H. MacNeal,
  R. L. Harder, A proposed standard set of problems to test finite element accuracy, Finite Elem. Anal. Des. 1 (1985);
  S. P. Timoshenko, S. Woinowsky-Krieger, Theory of plates and shells (1959): the reference values of section 6.
