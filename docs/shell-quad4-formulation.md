# Four-node quadrilateral shell element with 24 degrees of freedom — formulation

This document states a linear four-node shell element completely enough to implement
it: stiffness, mass, equivalent loads, stress resultants and differential stiffness. It
is self-contained. Every constant is listed in section 13 with its origin; the methods
used are those of the open literature listed in section 15.

## 1. Data

- Four grids $\mathbf X_i$ (i = 1..4, counter-clockwise) in a global Cartesian system, with
  the thicknesses $t_i$.
- Six degrees of freedom per grid: translations $\mathbf u_i$ and rotations
  $\boldsymbol\theta_i$ (global components).
- Section stiffnesses in the material axes, functions of the local thickness t:
  membrane $A$, coupling $B$, bending $D$ (3 × 3, engineering shear), transverse shear
  $H$ (2 × 2), or no $H$ when the section has no transverse shear flexibility. Mass
  moments per unit area $I_0, I_1, I_2$. For a homogeneous section
  $A\propto t$, $B\propto t^2$, $D\propto t^3$, $H\propto t$.
- Material axis: an angle φ from the reference direction of section 2, or a given
  direction projected on the element plane.
- Offset d of the reference surface from the grids, along the element normal.
- Drilling factor $c_d$.

Conventions: the director increment of a rotation is
$\boldsymbol\beta=\boldsymbol\theta\times\mathbf n$; in a local frame $(x,y,z)$ with z along the
normal,

$$
\mathbf e_0=(\varepsilon_x,\varepsilon_y,\gamma_{xy}),\quad
\boldsymbol\kappa=(\beta_{x,x},\ \beta_{y,y},\ \beta_{x,y}+\beta_{y,x}),\quad
\begin{Bmatrix}\mathbf N\\ \mathbf M\end{Bmatrix}=\begin{bmatrix}A&B\\B&D\end{bmatrix}
\begin{Bmatrix}\mathbf e_0\\ \boldsymbol\kappa\end{Bmatrix},\quad \mathbf Q=H\boldsymbol\gamma .
$$

## 2. Element axes

With the diagonals $\mathbf d_{13}=\mathbf X_3-\mathbf X_1$, $\mathbf d_{24}=\mathbf X_4-\mathbf X_2$:

$$
\mathbf e_z=\frac{\mathbf d_{13}\times\mathbf d_{24}}{\|\mathbf d_{13}\times\mathbf d_{24}\|},\qquad
\mathbf e_x=\frac{\hat{\mathbf d}_{13}-\hat{\mathbf d}_{24}}{\|\hat{\mathbf d}_{13}-\hat{\mathbf d}_{24}\|},\qquad
\mathbf e_y=\mathbf e_z\times\mathbf e_x ,
$$

origin $\mathbf X_c=\tfrac14\sum\mathbf X_i$. $T$ = 3 × 3 matrix whose rows are
$\mathbf e_x,\mathbf e_y,\mathbf e_z$; local coordinates $\mathbf x_i=T(\mathbf X_i-\mathbf X_c)$.
Everything below is written in these axes until section 11.

Reference direction: the projection of $\mathbf X_2-\mathbf X_1$ on the plane normal to
$\mathbf e_z$. The material direction $\mathbf m$ is this direction turned by φ about
$\mathbf e_z$, or the projection of the given direction on that plane.

## 3. Geometry and kinematics

Natural coordinates $(\xi_i,\eta_i)=(-1,-1),(1,-1),(1,1),(-1,1)$,
$N_i=\tfrac14(1+\xi\xi_i)(1+\eta\eta_i)$, $\zeta\in[-1,1]$ through the thickness.

Directors: $\mathbf n_i=\widehat{\mathbf x_{,\xi}\times\mathbf x_{,\eta}}$ evaluated at grid i,
with $\mathbf x=\sum N_i\mathbf x_i$.

$$
\mathbf X(\xi,\eta,\zeta)=\sum_iN_i\big(\mathbf x_i+\zeta\tfrac{t_i}2\mathbf n_i\big),\qquad
\mathbf u(\xi,\eta,\zeta)=\sum_iN_i\big(\mathbf u_i+\zeta\tfrac{t_i}2\,\boldsymbol\theta_i\times\mathbf n_i\big).
$$

Covariant vectors $\mathbf g_1=\mathbf X_{,\xi}$, $\mathbf g_2=\mathbf X_{,\eta}$,
$\mathbf g_3=\mathbf X_{,\zeta}=\sum N_i\tfrac{t_i}2\mathbf n_i$; $J(\zeta)$ = matrix of rows
$\mathbf g_1,\mathbf g_2,\mathbf g_3$.

Local frame at $(\xi,\eta)$, from the vectors at ζ = 0:

$$
\mathbf e_3=\widehat{\mathbf g_1\times\mathbf g_2},\qquad
\mathbf e_1=\widehat{\mathbf e_y\times\mathbf e_3},\qquad \mathbf e_2=\mathbf e_3\times\mathbf e_1 ,
$$

E = matrix of rows $\mathbf e_1,\mathbf e_2,\mathbf e_3$; $j=\|\mathbf g_1\times\mathbf g_2\|$ at ζ = 0.

Displacement gradient in the local frame, $L_{ab}=\partial u_b/\partial x_a$, at $(\xi,\eta,\zeta)$:

$$
L=\Big(E\,J(\zeta)^{-1}\Big)\begin{bmatrix}\mathbf u_{,\xi}\\ \mathbf u_{,\eta}\\ \mathbf u_{,\zeta}\end{bmatrix}E^{\mathsf T},
$$

the three rows $\mathbf u_{,\xi},\mathbf u_{,\eta},\mathbf u_{,\zeta}$ being kept (the third column of
$J^{-1}$ is not dropped).

## 4. Section at a point

Thickness $t=\sum N_it_i$; $A, B, D, H$ at that thickness. Angle of the material direction
in the local frame, $\varphi=\operatorname{atan2}(\mathbf m\cdot\mathbf e_2,\ \mathbf m\cdot\mathbf e_1)$,
$c=\cos\varphi$, $s=\sin\varphi$:

$$
(A,B,D)\leftarrow T_\varphi^{\mathsf T}(A,B,D)\,T_\varphi,\quad H\leftarrow R_\varphi^{\mathsf T}HR_\varphi,\qquad
T_\varphi=\begin{bmatrix}c^2&s^2&cs\\ s^2&c^2&-cs\\ -2cs&2cs&c^2-s^2\end{bmatrix},\quad
R_\varphi=\begin{bmatrix}c&s\\-s&c\end{bmatrix}.
$$

Without transverse shear flexibility:
$H=c_K\,\dfrac{12\max(D_{11},D_{22})}{4j}\begin{bmatrix}1&0\\0&1\end{bmatrix}$.

## 5. Membrane strains and curvatures

In-plane strains $\boldsymbol\varepsilon=(L_{11},\ L_{22},\ L_{12}+L_{21})$ at ζ = +1 and ζ = −1:

$$
\mathbf e_0=\tfrac12(\boldsymbol\varepsilon^++\boldsymbol\varepsilon^-)=B_m\mathbf d,\qquad
\boldsymbol\kappa=\frac{\boldsymbol\varepsilon^+-\boldsymbol\varepsilon^-}{2h}=B_b\mathbf d,\qquad h=\mathbf g_3\cdot\mathbf e_3 ,
$$

$\mathbf d$ = the 24 local degrees of freedom $(\mathbf u_i,\boldsymbol\theta_i)$.

## 6. Enhanced strains

Eight internal parameters $\boldsymbol\alpha=(\boldsymbol\alpha_m,\boldsymbol\alpha_b)$, four on the
membrane strains and four on the curvatures. At the element centre: local frame $E_0$,
$J_0=\begin{bmatrix}\mathbf g_1\cdot\mathbf e_1&\mathbf g_1\cdot\mathbf e_2\\ \mathbf g_2\cdot\mathbf e_1&\mathbf g_2\cdot\mathbf e_2\end{bmatrix}$,
$j_0=\det J_0$, $\mathbf j=J_0^{-1}$. With

$$
\mathsf M=\begin{bmatrix}\xi&0&0&0\\0&\eta&0&0\\0&0&\xi&\eta\end{bmatrix},\qquad
T_0=\begin{bmatrix}j_{11}^2&j_{12}^2&j_{11}j_{12}\\ j_{21}^2&j_{22}^2&j_{21}j_{22}\\ 2j_{11}j_{21}&2j_{12}j_{22}&j_{11}j_{22}+j_{12}j_{21}\end{bmatrix},\qquad
G=\frac{j_0}{j}\,T_0\,\mathsf M ,
$$

$$
\mathbf e_0\leftarrow B_m\mathbf d+G\boldsymbol\alpha_m,\qquad
\boldsymbol\kappa\leftarrow B_b\mathbf d+G\boldsymbol\alpha_b .
$$

## 7. Transverse shear

**Tied strains.** Covariant shear at ζ = 0:

$$
\tilde\gamma_\alpha=\mathbf g_3\cdot\mathbf u_{,\alpha}+\mathbf g_\alpha\cdot\sum_iN_i\tfrac{t_i}2\,\boldsymbol\theta_i\times\mathbf n_i ,\qquad\alpha=\xi,\eta ,
$$

evaluated at the mid-sides: $\tilde\gamma_\xi$ at $(0,-1)$ and $(0,1)$, $\tilde\gamma_\eta$ at
$(-1,0)$ and $(1,0)$, and interpolated:

$$
\tilde\gamma_\xi=\tfrac{1-\eta}2\tilde\gamma_\xi(0,-1)+\tfrac{1+\eta}2\tilde\gamma_\xi(0,1),\qquad
\tilde\gamma_\eta=\tfrac{1-\xi}2\tilde\gamma_\eta(-1,0)+\tfrac{1+\xi}2\tilde\gamma_\eta(1,0).
$$

Local components at a point, with the contravariant vectors $\mathbf g^k$ (rows of
$J(0)^{-\mathsf T}$):

$$
\gamma_a=\sum_{\alpha=1}^2\tilde\gamma_\alpha\big[(\mathbf g^\alpha\!\cdot\mathbf e_a)(\mathbf g^3\!\cdot\mathbf e_3)+(\mathbf g^3\!\cdot\mathbf e_a)(\mathbf g^\alpha\!\cdot\mathbf e_3)\big],\qquad a=1,2 .
$$

**Side lengths.** With $(\bar x_i,\bar y_i)$ the components of $\mathbf x_i$ on
$\mathbf e_1,\mathbf e_2$ of the centre frame:
$a=\tfrac12|\bar x_2+\bar x_3-\bar x_1-\bar x_4|$, $b=\tfrac12|\bar y_3+\bar y_4-\bar y_1-\bar y_2|$.

**Factors.** At each of the 2 × 2 integration points, with the local section,
$s(z)=z/(1+z)$ and

$$
D^*_{kk}=D_{kk}-\frac{B_{kk}^2}{A_{kk}}\quad(k=1,2,3;\ D^*_{kk}=D_{kk}\text{ when }B=0\text{ or when this value is not positive}),
$$

$$
s_{x0}=s\Big(\frac{12D^*_{11}}{H_{11}a^2}\Big),\qquad
s_{x1}=s\Big(\frac{12\,(D^*_{11}/a^2+c_tD^*_{33}/b^2)}{H_{11}}\Big),
$$

$$
s_{y0}=s\Big(\frac{12D^*_{22}}{H_{22}b^2}\Big),\qquad
s_{y1}=s\Big(\frac{12\,(D^*_{22}/b^2+c_tD^*_{33}/a^2)}{H_{22}}\Big).
$$

**Modified strains.** At the point $p=(\xi_p,\eta_p)$, with $p_\eta=(\xi_p,-\eta_p)$ and
$p_\xi=(-\xi_p,\eta_p)$, each $\gamma$ being evaluated with the frame of its own point:

$$
\gamma'_x=\sqrt{s_{x0}}\,\tfrac12\big[\gamma_1(p)+\gamma_1(p_\eta)\big]+\sqrt{s_{x1}}\,\tfrac12\big[\gamma_1(p)-\gamma_1(p_\eta)\big],
$$

$$
\gamma'_y=\sqrt{s_{y0}}\,\tfrac12\big[\gamma_2(p)+\gamma_2(p_\xi)\big]+\sqrt{s_{y1}}\,\tfrac12\big[\gamma_2(p)-\gamma_2(p_\xi)\big],
$$

$\boldsymbol\gamma'=(\gamma'_x,\gamma'_y)=B_s\mathbf d$.

## 8. Drilling

At a point, at ζ = 0:

$$
r=\omega-\theta_n,\qquad \omega=\tfrac12(L_{12}-L_{21}),\qquad
\theta_n=\sum_iN_i(\boldsymbol\theta_i-\bar{\boldsymbol\theta})\cdot\mathbf n_i+\bar{\boldsymbol\theta}\cdot\mathbf e_3,\quad
\bar{\boldsymbol\theta}=\tfrac14\sum_i\boldsymbol\theta_i ,
$$

$r=\mathbf b_d\mathbf d$. The term is zero for a rigid motion.

## 9. Stiffness

2 × 2 Gauss points $(\pm1/\sqrt3,\pm1/\sqrt3)$, unit weights. With
$\mathsf B=\begin{bmatrix}B_m\\B_b\end{bmatrix}$ (6 × 24),
$\mathsf G=\begin{bmatrix}G&0\\0&G\end{bmatrix}$ (6 × 8),
$\mathsf C=\begin{bmatrix}A&B\\B&D\end{bmatrix}$:

$$
K_{dd}=\sum_pj_p\Big(\mathsf B^{\mathsf T}\mathsf C\mathsf B+B_s^{\mathsf T}HB_s+c_dA_{33}\,\mathbf b_d^{\mathsf T}\mathbf b_d\Big),\qquad
K_{d\alpha}=\sum_pj_p\,\mathsf B^{\mathsf T}\mathsf C\mathsf G,\qquad
K_{\alpha\alpha}=\sum_pj_p\,\mathsf G^{\mathsf T}\mathsf C\mathsf G,
$$

$$
K=K_{dd}-K_{d\alpha}K_{\alpha\alpha}^{-1}K_{d\alpha}^{\mathsf T}.
$$

A section without membrane stiffness, or without bending stiffness, drops the
corresponding four parameters; the drilling term is present when the section has both.

## 10. Mass

With $N_u=[N_i\mathbf I_3]$ on the translations and $N_\beta$ such that
$\boldsymbol\beta=\sum N_i\,\boldsymbol\theta_i\times\mathbf n_i$:

$$
M=\sum_pj_p\big(I_0N_u^{\mathsf T}N_u+I_1(N_u^{\mathsf T}N_\beta+N_\beta^{\mathsf T}N_u)+I_2N_\beta^{\mathsf T}N_\beta\big).
$$

Lumped form: on the three translations of grid i, $m_i=\sum_pj_pI_0N_i$; nothing on the
rotations.

## 11. Offset and global system

Per grid, with the offset d along $\mathbf e_z$:

$$
\mathbf u_{ref}=T\mathbf u_g+d\,(\theta_y,-\theta_x,0),\quad\boldsymbol\theta=T\boldsymbol\theta_g
\ \Rightarrow\ T_6=\begin{bmatrix}T&d\begin{bmatrix}T_2\\-T_1\\0\end{bmatrix}\\0&T\end{bmatrix},
$$

$T_1, T_2$ = first and second rows of T. With $\mathsf T=\operatorname{diag}(T_6,T_6,T_6,T_6)$:
$K_g=\mathsf T^{\mathsf T}K\mathsf T$, $M_g=\mathsf T^{\mathsf T}M\mathsf T$.

## 12. Loads, resultants, differential stiffness

- **Pressure** p along the element normal:
  $\mathbf f_i=\sum_pp\,N_i\,(\mathbf X_{,\xi}\times\mathbf X_{,\eta})$ on the translations of grid i
  (global components, 2 × 2 points, surface of the grids).
- **Initial (thermal) resultants** $\mathbf S_0=(\mathbf N_0,\mathbf M_0)$ given in the material axes
  at each point, turned to the local frame by $T_\varphi^{\mathsf T}$:

$$
\mathbf f_d=\sum_pj_p\mathsf B^{\mathsf T}\mathbf S_0,\quad\mathbf f_\alpha=\sum_pj_p\mathsf G^{\mathsf T}\mathbf S_0,\quad
\mathbf f=\mathsf T^{\mathsf T}\big(\mathbf f_d-K_{d\alpha}K_{\alpha\alpha}^{-1}\mathbf f_\alpha\big).
$$

- **Resultants** at the integration points, in the local frames:

$$
\boldsymbol\alpha=K_{\alpha\alpha}^{-1}(\mathbf f_\alpha-K_{d\alpha}^{\mathsf T}\mathbf d),\qquad
\begin{Bmatrix}\mathbf N\\\mathbf M\end{Bmatrix}=\mathsf C\,(\mathsf B\mathbf d+\mathsf G\boldsymbol\alpha)-\mathbf S_0 .
$$

- **Values at the grids.** With the transverse shear forces of an integration point taken as
  those conjugate to its tied shears $\gamma_a(p)$ in the energy $\sum_pj_p\boldsymbol\gamma'^{\mathsf T}H\boldsymbol\gamma'$
  ($\mathbf Q'=H\boldsymbol\gamma'$, $p_\eta$ and $p_\xi$ as in section 7),

$$
Q_x(p)=\tfrac12\big(\sqrt{s_{x0}}+\sqrt{s_{x1}}\big)_pQ'_x(p)+\frac{j_{p_\eta}}{j_p}\tfrac12\big(\sqrt{s_{x0}}-\sqrt{s_{x1}}\big)_{p_\eta}Q'_x(p_\eta),
$$

  and $Q_y(p)$ alike with $p_\xi$, the values $f_p$ of the four integration points (resultants,
  strains $\mathsf B\mathbf d+\mathsf G\boldsymbol\alpha$, fibre stresses), each brought from its local
  frame to the element axes turned onto its tangent plane, give at grid i the value of the
  bilinear surface through them:

$$
f_i=\sum_p\tfrac14\,(1+3\,\xi_i\xi_p)(1+3\,\eta_i\eta_p)\,f_p .
$$

  The curvature along a direction in which the rotations vary linearly is constant over the
  element: the moment of a grid differs from the moment of the centre only by what the
  enhanced strains and the variation across that direction give, the linear part of the moment
  under a shear force being carried by the factors of section 7.
- **Differential stiffness** of the membrane forces $\mathbf N=(N_{11},N_{22},N_{12})$: with
  $G_a$ (3 × 24) the local gradient $\partial\mathbf U/\partial x_a$ of the translations of the
  reference surface (the part of L at ζ = 0 that multiplies the $\mathbf u_i$),

$$
K_\sigma=\mathsf T^{\mathsf T}\Big[\sum_pj_p\big(N_{11}G_1^{\mathsf T}G_1+N_{22}G_2^{\mathsf T}G_2+N_{12}(G_1^{\mathsf T}G_2+G_2^{\mathsf T}G_1)\big)\Big]\mathsf T .
$$

## 13. Constants

| Constant | Value | Origin |
|---|---|---|
| 12 in $s_{x0}$, $s_{y0}$ | exact | A beam segment of length a under end loads has a linear moment and a constant shear force; its complementary energy gives the stiffness of the mean shear strain, $1/K=1/H+a^2/(12D)$. |
| 12 and $D^*_{11}/a^2$ in $s_{x1}$ | exact | The same relation on each line across the element, for a shear strain varying linearly. |
| $c_t$ | 99 | Tolerance. The twisting curvature measured with the rotations can be relieved by the linear part of the shear; a thin strip one element wide keeps $\mu/(1+\mu)$ of its torsional rigidity, $\mu=c_t+(D_{11}/D_{33})(b/a)^2$. $c_t=1/\epsilon-1$ with ε = 1 %. |
| $D^*_{kk}$ | — | Bending stiffness of a strip in cylindrical bending whose membrane force along the strip is zero. |
| $c_d$ | 4·10⁻⁴ by default | Stabilisation of the rotation about the normal; results are insensitive to it between 10⁻⁷ and 10⁻³. |
| $c_K$ | 10⁴ | Sections without transverse shear flexibility: shear flexibility below 10⁻⁴ of the bending one. |

## 14. Checks of an implementation

1. Six zero-energy modes and no other, on flat, tapered, skewed and warped elements;
   $K\boldsymbol\Phi=0$ for the six rigid motions to round-off.
2. Constant membrane, bending and coupled states are reproduced exactly on any mesh of
   flat elements (patch test), for any section.
3. A cantilever strip with ν = 0 under a tip force has the tip deflection
   $PL^3/(3EI)+PL/(GA_s)$ with any number of elements, one included.
4. A thin strip one element wide, without transverse shear flexibility, twisted by end
   torques: torsional rigidity $\mu/(1+\mu)$ times $Gbt^3/3$, μ of section 13.
5. Numerical values. Isotropic section, E = 1000, ν = 0.3, t = 0.05 at the four grids:
   $A=\frac{Et}{1-\nu^2}\begin{bmatrix}1&\nu&0\\\nu&1&0\\0&0&\frac{1-\nu}2\end{bmatrix}$,
   $B=0$, $D=\frac{t^2}{12}A$, $H=\frac56\frac{Et}{2(1+\nu)}\mathbf I$, $I_0=0.05$, $I_1=0$,
   $I_2=t^3/12$, φ = 0, d = 0, $c_d=4\cdot10^{-4}$, $c_t=99$. The eighteen non-zero
   eigenvalues of the 24 × 24 stiffness and the trace of the consistent mass:

| Grids | Eigenvalues of K | trace M |
|---|---|---|
| (0, 0, 0), (1.2, 0, 0), (1.0, 0.9, 0), (−0.1, 0.8, 0) | 2.099463·10⁻⁴, 6.305329·10⁻⁴, 6.315727·10⁻⁴, 2.485962·10⁻³, 4.669159·10⁻³, 5.546885·10⁻³, 6.624600·10⁻³, 7.210094·10⁻³, 1.706078·10⁻², 1.015927·10⁻¹, 2.908098·10⁻¹, 3.286997·10⁻¹, 2.864322, 11.95446, 22.46793, 33.57019, 40.52063, 82.02336 | 6.567579·10⁻² |
| (0, 0, 0), (1.2, 0, 0), (1.0, 0.9, 0.1), (−0.1, 0.8, 0) | 2.107253·10⁻⁴, 6.341382·10⁻⁴, 6.352187·10⁻⁴, 2.500274·10⁻³, 4.684838·10⁻³, 5.421407·10⁻³, 7.279753·10⁻³, 8.967955·10⁻³, 1.706005·10⁻², 1.017682·10⁻¹, 2.899826·10⁻¹, 3.303452·10⁻¹, 2.877774, 12.00676, 22.45304, 33.60344, 40.46907, 81.86705 | 6.592780·10⁻² |

## 15. Literature

- S. Ahmad, B. M. Irons, O. C. Zienkiewicz, Analysis of thick and thin shell structures by
  curved finite elements, Int. J. Numer. Methods Eng. 2 (1970): shell kinematics with
  directors at the nodes.
- E. N. Dvorkin, K.-J. Bathe, A continuum mechanics based four-node shell element for
  general non-linear analysis, Eng. Comput. 1 (1984): transverse shear strains tied at the
  mid-sides.
- T. H. H. Pian, K. Sumihara, Rational approach for assumed stress finite elements, Int. J.
  Numer. Methods Eng. 20 (1984); J. C. Simo, M. S. Rifai, A class of mixed assumed strain
  methods and the method of incompatible modes, Int. J. Numer. Methods Eng. 29 (1990):
  five-parameter stress field and its enhanced-strain form.
- R. H. MacNeal, A simple quadrilateral shell element, Comput. Struct. 8 (1978): residual
  bending flexibility of the transverse shear.
- T. J. R. Hughes, F. Brezzi, On drilling degrees of freedom, Comput. Methods Appl. Mech.
  Eng. 72 (1989): rotation about the normal.
- E. Hinton, J. S. Campbell, Local and global smoothing of discontinuous finite element
  functions using a least squares method, Int. J. Numer. Methods Eng. 8 (1974); J. Barlow,
  Optimal stress locations in finite element models, Int. J. Numer. Methods Eng. 10 (1976):
  values at the grids from those of the 2 × 2 points.
- S. P. Timoshenko, J. M. Gere, Mechanics of Materials: beam with shear deformation.
