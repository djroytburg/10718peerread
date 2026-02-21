## Neural Hamiltonian Diffusions for Modeling Structured Geometric Dynamics

## Sungwoo Park

Department of Computer Science and Engineering Korea University sungwoo\_park@korea.ac.kr

## Abstract

We propose Neural Hamiltonian Diffusion (NHD), a unified framework for learning stochastic Hamiltonian dynamics on differentiable manifolds. Unlike conventional Hamiltonian Neural Networks (HNNs), which assume noise-free dynamics in flat Euclidean spaces, our approach models stochastic differential equations (SDEs) on curved manifolds endowed with both a Riemannian metric and a Poisson structure. Specifically, we parameterize a neural Hamiltonian and define the dynamics via a Stratonovich SDE whose drift is the Poisson vector field lifted horizontally to the orthonormal frame bundle. This construction ensures coordinate-invariant, gaugeconsistent dynamics across (pseudo-)Riemannian manifolds, enabling physically plausible modeling in systems with geometric constraints, periodicity, or relativistic structure. We establish generalization guarantees under curvature-dependent complexity and demonstrate applications across diverse scientific domains, including toroidal molecular dynamics, quantum spin systems, and relativistic n -body problems in Schwarzschild spacetime.

## 1 Introduction

Modeling physical dynamics from data is a fundamental challenge in machine learning, with applications ranging from molecular simulations and protein folding [Karplus and McCammon, 2002, Noé et al., 2020] to planetary motion and gravitational systems with curved spacetime [Rein and Liu, 2019, Pretorius, 2005]. A central goal in this context is to learn a dynamical model that not only predicts future states accurately but also reflects the underlying physical laws such as symplectic structure, and geometric invariance [Hairer et al., 2006]. However, many existing learning-based approaches focus on approximating state transitions in Euclidean spaces without explicitly encoding the physics or geometry of the system. In practice, physical systems often evolve on non-Euclidean domains. In molecular dynamics, for example, internal coordinates such as dihedral angles naturally reside on toroidal or pseudo-Riemannian manifolds [Zhou et al., 2020, Townsend et al., 2021]. Likewise, in modeling N -body interactions near massive celestial bodies, the trajectories of particles evolve in strongly curved spacetimes, where the underlying geometry plays a crucial role in determining the causal and dynamical structure [Rezzolla and Zanotti, 2013]. Quantum spin systems modeled on compact Lie groups also exhibit inherently curved dynamics due to their non-Euclidean group geometry [Sakurai and Napolitano, 2017]. These systems are inherently geometric, and their governing laws are often described by Hamiltonian mechanics on manifolds equipped with symplectic structures.

Recent advances in Hamiltonian learning including Hamiltonian Neural Networks (HNNs) [Greydanus et al., 2019] and their variants [Zhong et al., 2020, Chen et al., 2021, Cranmer et al., 2020, Wang et al., 2023, Dierkes et al., 2023, Khoo et al., 2023] have shown that incorporating symplectic structure can significantly improve generalization and long-term stability. Complementary approaches such as Symplectic ODE-Nets [Zhong et al., 2020], Symplectic Recurrent Neural Networks [Chen et al., 2021], and Symplectic Transformers [Finzi et al., 2020] embed symplectic constraints directly

into the learning architecture. However, these models typically operate in flat Euclidean phase space and do not generalize to curved configuration spaces or non-canonical geometries. Orthogonally, a growing line of research on modeling learnable stochastic dynamics on Riemannian manifolds [De Bortoli et al., 2022, Huang et al., 2022, Park et al., 2022, Mathieu et al., 2023] has enabled geometry-aware stochastic modeling. Yet, these approaches are not physically grounded: they do not enforce Hamiltonian or symplectic structure to preserve physical fidelity. In this context, our contribution lies in unifying these previously disconnected pillars. We propose a stochastic Hamiltonian modeling framework that incorporates both the geometric complexity of differentiable manifolds and the structural constraints of Hamiltonian mechanics. We extend Hamiltonian learning to curved, periodic, and causally structured domains with stochasticity. We highlight the following contributions:

- Neural Hamiltonian Diffusions on Curved Manifolds. We propose a novel framework that unifies stochastic diffusion processes and Hamiltonian mechanics on general curved spaces. By incorporating gauge consistency into the modeling, we ensure that the learned dynamics remain physically meaningful and independent of local coordinate choices. Our approach respects both the symplectic structure and the intrinsic geometry of the system, enabling faithful simulation of stochastic physical processes beyond flat spaces.
- Geometry-Consistent Learning via Frame Bundle Lifts. Instead of learning dynamics directly on the base manifold, we lift the formulation to the frame bundle to handle curvature and coordinatedependence explicitly. This allows the model to represent vector fields in a unified way across locally varying orthonormal frames, ensuring compatibility between overlapping charts. This geometric design provides consistency across varying local frames and improves the physical reliability of the learned vector fields.
- Theoretical Guarantees and Empirical Superiority. We establish theoretical generalization bounds that link curvature, network capacity, and frame symmetry, and prove that gauge consistency intrinsically reduces worst-case deviations. Our method achieves superior performance compared to existing approaches across various structured geometric systems, demonstrating the practical benefits of incorporating geometric and physical consistency into learning.

## 2 Neural Hamiltonian Diffusion

Hamiltonian Dynamics. Let m t := ( q t , p t ) ∈ R 2 d denote the canonical position and momentum coordinates in phase space, and let H ∈ C ∞ ( R 2 d ) be a smooth Hamiltonian function. We briefly recall the canonical formulation of Hamiltonian dynamics in Euclidean phase space, where the system evolves according to a smooth Hamiltonian function H via the associated Poisson bracket structure:

<!-- formula-not-decoded -->

The operator { f, g } := ∇ f ⊤ J ∇ g defines the canonical Poisson bracket for any smooth functions f, g ∈ C ∞ ( R 2 d ) , and {· , H } denotes the Hamiltonian vector field applied to observables. This formulation is analytically tractable and serves as the foundation of classical conservative dynamics.

However, such Euclidean and deterministic formulations may face fundamental limitations when applied to structured data domain. First, they assume a globally flat phase space, making them illsuited for modeling systems with curved or topologically structured configuration spaces such as those encountered in relativistic, periodic, or molecular settings. Second, real-world physical dynamics are often inherently stochastic , due to latent variables, thermal fluctuations, or observational uncertainty, none of which are reflected in the deterministic formulation. To overcome these limitations, we move beyond the classical regime and explore a generalized class of Hamiltonian systems that operate over differentiable manifolds and evolve according to stochastic dynamics . Specifically, we adopt the framework of Hamiltonian diffusion Bismut [1981], which preserves the structural fidelity of Hamiltonian flows while incorporating both the intrinsic geometry of the underlying space and the probabilistic nature of physical systems. This generalized formulation enables the modeling of structured, curved, and noisy dynamics in a principled and physically consistent manner.

Hamiltonian Diffusion on Manifolds 1 . Throughout, we work on 2 d -dimensional symplectic manifold ( M , ω ) equipped with a Poisson structure i . e ., {· , ·} with local coordinates m := ( q , p ) ∈

1 For a detailed explanation of the background, we refer the reader to the Appendix A.

Figure 1: Horizontally-lifted Hamiltonian Diffusion and Gauge Equivariance on Frame Bundle . (Left) A red stochastic trajectory X t evolves on M = T ∗ Q . The learned horizontal vector field G Hor θ transports an orthonormal frame U t to U s that spans T M , t ≤ s ; its connection-driven rotation is highlighted by the blue arrow ( U · h ), realizing the symplectic structure in the principal O( d ) -bundle. (Right) We visualize a specific types of equivariant Hamiltonian vector fields along with their fiber rotations under O( d ) . The preserved structure under transformations highlights the gauge equivariance property of our model.

<!-- image -->

T ∗ Q := M , where the configuration space is set to Riemannian manifolds ( Q , g ) . Now, we give a formal definition of Hamiltonian diffusion on manifolds:

Definition 2.1. Let (Ω , F t := F B t , P ) be the augmented probability space with 2 d -dimensional Brownian motion B t . Given the standard non-degenerate symplectic 2-form ω := ∑ d i =1 d q i ∧ d p i , we define a M -valued semi-martingale that solves the system of stochastic differential equations:

<!-- formula-not-decoded -->

where π q : T ∗ Q → Q and π p : T ∗ Q → T ∗ q Q are canonical projections onto configuration manifold and the fiber, and ι and d denote the interior product and the exterior derivative on M , respectively.

A neural Hamiltonian diffusion (NHD) refers to a stochastic process X t evolving on a manifold under the Hamiltonian flow, where the Hamiltonian H θ is modeled by neural networks with parameters θ ∈ Θ so as to reflect the induced physical structure of the system. As can be seen, the Poisson bracket formulation naturally leads to geometry-aware dynamics, ensuring that the induced flow respects the curvature and structure of the underlying manifold. To be more specific, the vector field on the manifold takes the form

<!-- formula-not-decoded -->

where m = ( q, p ) ∈ T ∗ Q , and G -1 ( q ) is the inverse of the Riemannian metric on the configuration manifold Q . The resulting vector field retains the skew-symmetric structure of Hamiltonian flows while encoding local geometric information through the metric G := [ g ij ] , and can be viewed as a geometry-aware generalization of Euclidean Hamiltonian vector fields in Eq (1).

In deterministic Hamiltonian systems in Eq (1), energy conservation is encoded by the identity ˙ H = 0 , which holds along every trajectory, ensuring exact invariance of the Hamiltonian over time. This reflects the fact that the energy function remains constant along deterministic flows. In contrast, our stochastic Hamiltonian framework characterizes energy conservation through the stationarity of the equilibrium distribution, given by L θ π = 0 , where π ∝ e -H ( X ∞ ) . Rather than preserving energy along individual sample paths, the Hamiltonian in this case governs the long-term statistical behavior of the system via the generator L θ . Table 3 summarizes the distinction between these two paradigms.

Horizontal Lift of Hamiltonian Diffusion. One major difficulty in the simulation of Eq. (2) lies in the absence of canonical coordinates, as well as the lack of a principled method to define stochasticity on manifolds. Recently, [De Bortoli et al., 2022] suggested geodesic random walks (GRWs) which harness the property of extrinsic geometry by using Riemannian exponential maps. Yet, there are open questions to respect Hamiltonian and symplectic structures by using GRWs. In contrast to extrinsic approaches, we formulate the intrinsic geometry by lifting the process to the frame bundle [Hsu,

2002], where geometry-consistent noise can be defined, naturally allowing for a principled realization of stochastic Hamiltonian dynamics on manifolds Lázaro-Camí and Ortega [2008].

Formally, we introduce a horizontal lift ( i . e ., U t ) of the diffusion process ( i . e ., X t ) to the frame bundle O ( M ) , where the stochastic dynamics admit local frame coordinates adapted to the manifold:

Proposition 2.2 (Horizontal Hamiltonian Diffusion) . Let U t ∈ O ( M ) be the horizontal lift of the diffusion process X t = π ( U t ) , where π : O ( M ) →M is the canonical projection and m denotes a local coordinate function on M . The lifted process U t evolves according to the Stratonovich SDE:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where [Γ M ] ♭ ∈ M(2 d × d 2 ) is the index-lowered connection tensor (i.e., Christoffel symbol), and vec( E ) ∈ M( d 2 , 1) is the vectorized local orthonormal frame. ∇ x and ∇ e denote the vectorized gradients with respect to the configuration point and the frame coordinates, respectively.

Here, the horizontal process U t ∈ O ( M ) augments the manifold trajectory with a local orthonormal frame, enabling noise to be defined in the tangent space and lifted via the horizontal distribution. In the formulation to ensure horizontal transport, we remove the vertical fiber part using the connection one-form ω 2 . The resulting vector field G Hor θ ( U t ) ∈ H U t is obtained by projecting ̂ G θ onto the horizontal distribution as G Hor θ = (Id -ω ♯ ) ̂ G θ , which ensures that the Stratonovich increment lies in H U = ker ω . Figure 1 schematically illustrates this construction. Note that we lift the base metric g to a Sasaki-type metric g M to induce geometric structure on the total space M = T ∗ Q . The resulting connection tensor Γ M serves as a geometric object defined on the total manifold of trajectory X t :

Definition 2.3 (Lifted Metric on the Total Space) . Let Q be a configuration manifold equipped with Riemannian metric g with its total space M := T ∗ Q . We define Sasaki-type metric on M as g M := π ∗ q g ⊕ g -1 , where π ∗ q g is a pull-back metric with respect to π q . Then, the connection tensor associated with g M take the following block tensor form:

<!-- formula-not-decoded -->

Having the complete definition of Hamiltonian diffusion on M by using horizontal lifts in Proposition 2.2, we now shift our focus from defining the model to learning it from physical data.

Learning Geometric Dynamics. Main feature of modeling physical systems is that the system is required to reconstruct both positional and complementary information such as velocity and momentum. This additional physical information is represented as trajectories evolving in the secondorder tangent bundle of the phase space, formally modeled as γ t = ( q t , p t , ˙ q t , ˙ p t ) ∈ T ( T ∗ Q ) . Given observational trajectory sampled from a data path distribution ˜ γ t ∼ P t, data on physical data space, we seek to recover the underlying Hamiltonian H θ that approximately generates the observed dynamics. The objective is to align the empirical drift of the process with the vector field induced by the learned Hamiltonian. To this end, we define the training loss over path distribution:

Definition 2.4 (Hamiltonian Learning) . Let γ t = ( q t , p t , ˙ q t , ˙ p t ) ∈ T ( T ∗ Q ) be a sample trajectory on the physical data space, and let ˜ γ t ∼ P t, data denote the corresponding data distribution. Given a parameterized Hamiltonian H θ : T ∗ Q → R , the geometric learning objective is defined as follows:

<!-- formula-not-decoded -->

where ∥·∥ g M is a Riemannian norm, and d Q denotes a distance function on configuration space.

2 Abusing notation, we denote both the connection one-form and the symplectic form by ω for simplicity (with ω # is its pull-back), as the meaning will be unambiguous from context.

Eq. (7) defines a hybrid loss that combines a vector field alignment term and a trajectory reconstruction term. The first term ensures that the learned Hamiltonian vector field aligns with the empirical time derivatives of the state to reflect the underlying physical dynamics. The second term penalizes deviations between the predicted and observed positions on the configuration manifold. The entire objective is conditioned on a fixed initial state U 0 = π -1 ([ q 0 , p 0 ]) , reflecting the initial value nature of Hamiltonian systems where the full trajectory is determined by the initial condition. Together, these objectives encourage the model to learn a Hamiltonian function that faithfully captures both the evolution of the system and its observed behavior.

Designing Neural Hamiltonian Functions. To model physically consistent dynamics in curved or structured spaces, we aim to construct Hamiltonian functions that reflect the underlying geometry of the configuration space. As motivated from conventional force-field modeling Salomon-Ferrer et al. [2013], we begin by formulating N -body interaction systems on (pseudo-)Riemannian manifolds parameterized by neural networks.

<!-- formula-not-decoded -->

Eq. (8) defines a parameterized Hamiltonian function for a system of interacting particles, where the joint state ( q N := { q 1 , · · · , q N } , p N := { p 1 , · · · , p N } ) ∈ M N evolves on the cotangent space of the joint configuration manifold ( M N , ( g M ) N ) . The proposed Hamiltonian consists of three distinct components: (i) the first kinetic energy term represents the kinetic energy of each particle and incorporates geometry-awareness by using the local inverse Riemannian metric, (ii) the second potential term captures single-particle effects through neural potentials that depend on individual states, including temporal, environmental, or local structural influences, and (iii) the third term accounts for pairwise interactions that model spatial dependencies and mutual influence across particles, which are essential for capturing correlated behavior on the manifold. Having established the parameterized Hamiltonian function H θ , we now derive the associated Poisson bracket {· , H θ } expressed explicitly in local coordinates on M as follows:

<!-- formula-not-decoded -->

In what follows, we impose a structural constraint on the neural Hamiltonian H θ that respects the gauge symmetry of the underlying frame bundle, leading to the formulation of gauge equivariance on O ( M ) to ensure the efficient learning of the parameterized Poisson bracket { m , H θ } .

Gauge Equivariance on Frame Bundle. While the choice of frame can be arbitrary, the proposed horizontal lift canonically projects dynamics from the base manifold to its frame bundle in a manner that is independent of specific frame parametrization. In this context, gauge equivariance is an essential property for ensuring that learned dynamics remain consistent across arbitrary frame choices and preserve geometric coherence during transport Cohen et al. [2019]. This principle can also formally be realized through the geometry of the orthonormal frame bundle in our framework.

Let U ∈ O ( M ) and h ∈ O( d ) be orthonormal frame and the rotation defined by R h ( U ) := U · h , then the orthonormal frame bundle O ( M ) admits a natural right action of the structure group O( d ) := { h ∈ GL( d, R ) | h ⊤ h = I d } . This action describes local gauge transformations within each fiber, and naturally lifts geometric quantities from base manifold M into an equivariant bundle.

Definition 2.5 (Gauge Equivariance on the Frame Bundle) . Let f : O ( M ) → V be a map into a representation space V ∼ = R d q ⊕ R d p ∼ = R 2 d , corresponding to position and momentum components in the cotangent bundle. We say that f is gauge equivariant if, for every h ∈ O( d ) ,

<!-- formula-not-decoded -->

where U ∈ O ( M ) is an orthonormal frame, and U · h denotes the right action of h on U .

Figure 1 provides an illustrative example of gauge equivariance defined over the orthonormal frame bundle. The equivariance property reflects that under frame transformation, local momentum vectors

p ∈ R d transform covariantly as p ↦→ hp , while position vectors q ∈ R d remain invariant as base coordinates. In the context of Hamiltonian learning, the goal is then to construct the neural Hamiltonian H θ such that the resulting geometry-induced drift term G Hor θ ( U ) is gauge equivariant under the frame transformation U ↦→ U · h , which holds if the following condition is satisfied:

<!-- formula-not-decoded -->

To realize equivariant learning within frame bundle coordinates, we propose the Frame Equivariant Transformer U-Net 3 . This architecture integrates canonicalization by transforming coordinates into the local orthonormal frame via U ⊤ p , performing invariant computations, and reconstructing outputs via U ˆ p , thereby ensuring gauge-consistent predictions. In Appendix, we provide the algorithm and pseudo-code for sampling frame-equivariant and neural network architectures.

## 3 Theoretical Analysis

In this section, we present two theoretical results: a generalization bound linking curvature and model capacity, and a deviation bound showing how gauge equivariance improves stability across frames.

Uniform-in-time Generalization of Hamiltonian Learning. With the objective function posed earlier, a natural question arises: If the model achieves near-perfect trajectory matching, why is Hamiltonian learning still necessary? This question is fundamental, as trajectory matching alone does not guarantee physically meaningful generalization. To analyze this rigorously, we consider the neural network θ ⋆ , which exactly reproduces the physical trajectories.

<!-- formula-not-decoded -->

Here, the radius R of the ball Θ reflects the capacity of neural networks. Unfortunately, although the neural network θ ∗ perfectly fits the physical trajectory in the training phase, it fails to capture the holistic physical information such as velocity and momentum. Proposition 3.1 demonstrates that our proposed geometric Hamiltonian learning significantly improves generalization.

Proposition 3.1 (Informal) . Let P t ( θ ) := Law( γ t ( θ )) be an associated probability measure of model trajectory, and assume the condition ( C1 ) in Eq. (11) holds, Under the mild regularity conditions of Hamiltonian function, the learned model distribution fails to remain close to the data distribution uniformly over time with high probability:

<!-- formula-not-decoded -->

where W := W 2 , 2 T ( T ∗ Q ) denotes the squared Wasserstein distance on the physical data space T ( T ∗ Q ) , and Ω is a constant depending on geometric and model-specific quantities.

Generalization in geometric Hamiltonian learning hinges on two key factors: the curvature of the configuration manifold ( i . e ., Γ ) and the network capacity ( i . e ., R ) . High curvature ( ∥ Γ ∥ ∞ ) intensifies stochastic distortion, while large R increases variance. This induces a trade-off-expressive models capture complex geometry but risk overfitting under curvature. Trajectory matching alone is insufficient, often neglecting velocity and momentum structure. Our method resolves this by enforcing physically consistent dynamics beyond position-level fitting.

Gauge Equivariance Ensures Smaller Deviations. In the second theoretical finding, Proposition 3.2 shows that enforcing gauge equivariance not only yields uniformly smaller worst-case Wasserstein deviations between the learned trajectories and the reference geodesic across all admissible frames, but also tightens the resulting generalization bounds by eliminating spurious frame-dependent variance.

Proposition 3.2 (Informal) . Let γ : [0 , T ] → T ∗ Q be a reference physical data represented as a geodesic. For any Hamiltonian H θ define the frame-rotated trajectory by X t ( h ) := ( q t , π p ◦ π ( U t · h )) for h ∈ O( d ) . Then, for arbitrary h ′ ∈ O( d ) , there exists constants κ, C &gt; 0 such that the following inequality holds:

<!-- formula-not-decoded -->

where X eq t ( h ) ∼ P eq t ( h ) is generated by a gauge-equivariant Hamiltonian function H θ .

3 Appendix D contains in-detailed information of model architecture.

Figure 2: Visualization of Three-body Quantum Spin Dynamics via Hopf Projection. Each subplot shows the spin trajectory of a single body on the 3-sphere S 3 , projected to two orthogonal complex planes: z 1 = x + iy (orange) and z 2 = z + iw (magenta). (Left) Ground-truth trajectories reveal nonlinear yet phase-coherent dynamics. (Right) Our model (HDM) accurately reproduces the spin geometry across bodies.

<!-- image -->

## 4 Related Works

Hamiltonian Neural Networks. Hamiltonian Neural Networks (HNNs) [Greydanus et al., 2019] introduced the idea of learning a scalar energy function H ( q, p ) whose gradients define conservative dynamics. Several extensions have since been proposed to improve generality, structure preservation, or application-specific modeling. [Cranmer et al., 2020] proposed learning Lagrangian dynamics as an alternative to the Hamiltonian formulation. [Chen et al., 2021] introduced symplectic recurrence for better long-term stability. [Wang et al., 2023] incorporated symplectic constraints into the learning process. [Simiao et al., 2023] adapted HNNs to rigid-body dynamics with energy-aware formulations. [Dierkes et al., 2023] focused on automatic symmetry detection and exploitation. [Khoo et al., 2023] proposed modeling separable Hamiltonians to reflect physical modularity. Unlike prior work constrained to Euclidean domains, our model learns Hamiltonian dynamics directly on manifolds, enabling faithful modeling of geometry-aware physical systems.

Neural Diffusion on Manifolds. Recent work has extended neural stochastic modeling to nonEuclidean spaces, particularly Riemannian manifolds, by incorporating geometric structure into diffusion or score-based generative models. [De Bortoli et al., 2022] proposed Riemannian score-based generative modeling on smooth manifolds, generalizing Langevin dynamics and score matching to curved spaces. [Huang et al., 2022] developed Riemannian diffusion models by extending continuoustime stochastic differential equations (SDEs) to arbitrary manifolds. [Park et al., 2022] introduced Riemannian Neural SDEs, enabling stochastic representation learning directly on manifolds using intrinsic geometry. [Lou et al., 2023] addressed the scalability of Riemannian diffusion models for high-dimensional and complex manifold settings. [Fishman and Cunningham, 2023] tackled constrained diffusion in non-Euclidean domains by incorporating boundary-aware mechanisms. These works lay the foundation for stochastic modeling on manifolds. Building on this line of research, our approach extends geometric diffusion models with a Hamiltonian perspective, enabling structured modeling of physical dynamics on curved spaces.

## 5 Experiments

Problem Formulation. In this section, we validate our proposed framework across three distinct physical scenarios that reflect a diverse range of geometric structures: (i) an interacting spin system evolving on the compact Lie group manifold SU(2) ∼ = S 3 , (ii) relativistic N -body dynamics formulated on Lorentzian spacetimes such as the Schwarzschild manifolds and (iii) molecular dynamics of protein backbones represented on high-dimensional toroidal configuration spaces T N . Each setting highlights a unique combination of curvature, topology, and physical constraints, allowing us to assess the generality and fidelity of neural Hamiltonian diffusion on non-Euclidean domains. We compare our method with recent state-of-the art methodologies in geometric sequential modeling including GeoTDM Han et al. [2024], EqMotion Xu et al. [2023], EGNN Satorras et al. [2021], SE-3 transformer Fuchs et al. [2020]. Hamiltonian learning based such as HNN Greydanus et al. [2019], SympHNN David and Méhats [2023], Noether van der Ouderaa et al. [2024] are also considered.

In all scenarios, we formulate physical dynamics prediction as a sequence modeling problem on non-Euclidean manifolds. Let { γ t } T t =1 denote a trajectory of geometric states γ t ∈ T ( T ∗ Q ) sampled from a Hamiltonian system. During training, each model is conditioned on a single initial state γ 1 and trained to autoregressively predict the subsequent states { γ t } T obs t =2 . At test time, the predicted state is re-fed into the model to generate the next one, allowing the model to learn long-range extrapolation dynamics at each step. This setup reflects realistic forecasting settings where long-term evolution must be inferred from geometric observations. A comprehensive summary of the experimental setups is included in the Appendix.

Table 1: Comparison of toroidal protein trajectory prediction and curved-space N -body dynamics. The first three columns (AD-3, 2AA, 4AA) report ADE/FDE on protein torsion angle trajectories. The last two columns (Spin and Schwarzschild) report ADE from N -body simulations of spin-based and Schwarzschild-metric particle systems with N = 3 and N = 5 particles. The first and second best is highlighted with bold and blue .

| Model      | AD-3              | 2AA               | 4AA               | Spin              | Schwarzschild     |
|------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| HNN        | 0 . 413 / 0 . 779 | 0 . 612 / 0 . 859 | ≥ 1 . 0           | 0 . 141 / 0 . 275 | 0 . 106 / 0 . 218 |
| Noether    | 0 . 554 / 0 . 614 | 0 . 580 / 0 . 723 | ≥ 1 . 0           | 0 . 077 / 0 . 162 | 0 . 053 / 0 . 116 |
| SympHNN    | 0 . 596 / 0 . 717 | 0 . 519 / 0 . 736 | ≥ 1 . 0           | 0 . 083 / 0 . 124 | 0 . 063 / 0 . 175 |
| SE(3)-Tr.  | 0 . 312 / 0 . 513 | 0 . 445 / 0 . 596 | 0 . 649 / 0 . 830 | 0 . 384 / 0 . 665 | 0 . 338 / 0 . 437 |
| EGNN       | 0 . 251 / 0 . 501 | 0 . 367 / 0 . 405 | 0 . 417 / 0 . 474 | 0 . 182 / 0 . 242 | 0 . 155 / 0 . 246 |
| EqMotion   | 0 . 081 / 0 . 117 | 0 . 062 / 0 . 152 | 0 . 131 / 0 . 174 | 0 . 090 / 0 . 102 | 0 . 081 / 0 . 149 |
| GeoTDM     | 0 . 045 / 0 . 102 | 0 . 079 / 0 . 145 | 0 . 093 / 0 . 179 | 0 . 037 / 0 . 085 | 0 . 026 / 0 . 046 |
| NHD (Ours) | 0 . 023 / 0 . 084 | 0 . 055 / 0 . 103 | 0 . 117 / 0 . 186 | 0 . 019 / 0 . 097 | 0 . 012 / 0 . 035 |

Molecular Dynamics of Protein Backbones. In modeling Hamiltonian formulations of protein molecular dynamics, we are motivated by classical force fields used in molecular modeling Cornell et al. [1995], Maier et al. [2015], Tian et al. [2019], which incorporate structured physical interactions such as bond stretching, angle bending, torsional rotations, and non-bonded forces. We reinterpret these force fields as learnable neural potentials while preserving underlying geometric and physical consistency. The configuration space is set to high-dimensional torus Q := T N angle , where N angle denotes the number of torsional degrees of freedom ( e . g ., , backbone dihedral angles ϕ, ψ, ω and side-chain angles χ i ). To evaluate the proposed framework, we perform experiments on three representative peptide systems of increasing complexity: AD-3 Alanine dipeptide , which exhibits

Figure 3: Spatiotemporal Ramachandran Map . Torsional state evolution over time compared between true and predicted trajectories.

<!-- image -->

simple dynamics on T 2 , 2AA dipeptides , where T 4 arises from backbone and occasional side-chain torsions, and 4AA tetrapeptides , which form structured dynamics on T 12 due to multiple interacting torsional modes. To extract geometric Hamiltonian states, we post-process time-aligned atomic trajectories from the Timewarp Klein et al. [2023] to compute angles ( ϕ, ψ, ω, χ i ) as generalized coordinates, and approximate their corresponding momenta by estimating the reduced moment of inertia associated with each torsional mode. Time derivatives are computed via finite differences across consecutive frames. We evaluate trajectory quality using standard geodesic metrics on the torus manifold, including average displacement error (ADE) and final displacement error (FDE), where distances are measured along T N angle = S 1 ×··· × S 1 . As summarized in Table 1, our method outperforms existing benchmarks by a significant margin across all evaluated metrics.

Quantum Spin System. In quantum physics, a spin system refers to a collection of particles, each possessing an intrinsic angular momentum (spin) that interacts with neighboring spins according to specified coupling rules. Mathematically, spin states are modeled as unit vectors on a sphere or, as elements of compact Lie groups. We model the dynamics of mutually interacting quantum spins on the unit 3 -sphere S 3 ⊂ R 4 , where each spin is represented as a unit quaternion that evolves under rigid-body dynamics. The system is equipped with anisotropic inertia and pairwise coupling Hamiltonians, giving rise to nonlinear, geometry-constrained motion. The total Hamiltonian of the system takes the following form: H( q N , p N ) = 1 2 ∑ N i =1 ( p i ) ⊤ I -1 p i -∑ i&lt;j λ ij ( ⟨ q i , q j ⟩ ) 2 where ω i ∈ R 3 is the body angular velocity of the i -th spin, I ∈ R 3 × 3 is the moment of inertia tensor, and λ ij is the coupling strength promoting alignment between spins q i and q j . The inner product ⟨ q i , q j ⟩ = x i x j + y i y j + z i z j + w i w j measures the similarity of unit quaternions on S 3 . The time evolution is governed by the Hamiltonian equations ˙ q i = 1 2 Ω( ω i ) q i and ˙ ω i = I -1 τ i , where Ω( ω ) encodes angular velocity and τ i is the coupling torque promoting spin alignment. The induced dynamics are thus constrained to a Riemannian manifold, specifically the 3-sphere endowed with its canonical metric. We visualize the resulting trajectories using Hopf projection π : S 3 → S 2 in Figure 2, where each spin is mapped to complex plane components ( z 1 , z 2 ) ∈ C 2 with z 1 = x + iy , z 2 = z + iw , | z 1 | 2 + | z 2 | 2 = 1 . Both the qualitative trajectories in Figure 2 and the quantitative metrics in Table 1 demonstrate that our dynamics accurately capture the underlying spin system evolution.

Figure 5: Three-body Trajectories in the Spacetime of Schwarzschild Black Holes. Left : The ground-truth simulation obtained by numerically integrating the exact relativistic dynamics, shows three mutually interacting bodies (labels 1 -3 ) spiraling toward the event horizon ( i . e ., Schwarzschild Radius = 2 M ). Center : The proposed method accurately captures the relativistic deflection and inward inspiral of all three trajectories, remaining faithful to the ground-truth. Right : Existing Euclidean HNN trained without explicit geometric conditioning yields inconsistent trajectories that indicate an incorrect physical regime.

<!-- image -->

Relativistic Particle Dynamics. In the last experiment, we consider the dynamics N interacting bodies in the curved spacetime surrounding compact astrophysical objects such as Schwarzschild black hole. In formulation, the background force field is derived from general relativity, encapsulating the relativistic geometry of spacetime. Meanwhile, the interaction forces between bodies follow classical modeling assumptions, e . g ., pairwise

Figure 4: Comparison of total Hamiltonian H( t ) and cumulative relative drift E t | ∆H t | / H 0 across models.

<!-- image -->

Newtonian-like potentials. This setup allows us to generalize classical N -body systems to curved spacetime environments beyond the fat spaces Satorras et al. [2021]. The Hamiltonian consists of a kinetic term defined via the inverse Schwarzschild metric and a classical pairwise potential: H( q N , p N ) = 1 2 M ∑ N i =1 ∑ 3 µ,ν =0 g µν ( q i ) p µ i p ν i -∑ i&lt;j GM 2 √ ∥ ⃗ q i -⃗ q j ∥ 2 E + ε 2 . The geodesic structure of the spacetime introduces non-Euclidean curvature effects in the momentum transport, while inter-body forces remain Newtonian-like. We implement a symplectic leapfrog integrator adapted to relativistic Hamiltonian flow and simulate multi-body systems initialized near stable orbital radii. The results clearly indicate that Euclidean methods struggle to model particle behavior in curved geometry. In contrast, our proposed HDM achieves superior reconstruction accuracy and substantially lower energy drift in Table 1, reflecting improved alignment with the intrinsic geometry of the system.

Ablation Study. We assess the numerical stability and scalability of our model via two ablation criteria: (i) energy conservation, and (ii) robustness across varying system sizes. Figure 4 shows that our method yields significantly lower energy drift compared to Euclidean baselines ( i . e ., SympHNN), indicating better consistency with the underlying geometric structure. In addition, Table 2 reports how performance degrades as the number of spin particles increases. While both models exhibit reduced accuracy for larger N , the proposed gauge-equivariant

Table 2: Performance degradation as the number of spin particles increases.

|   N | G -equiv   | Non-equiv      |
|-----|------------|----------------|
|   3 | 0 . 019    | 0 . 024 (+24%) |
|   5 | 0 . 097    | 0 . 103 (+6%)  |
|  10 | 0 . 120    | 0 . 173 (+44%) |
|  20 | 0 . 148    | 0 . 169 (+14%) |

model remains consistently more stable. For instance, while both models experience increasing error as N grows, the non-equivariant variant exhibits a sharp deterioration at N = 10 , with over a sevenfold increase in ADE relative to N = 3 . In contrast, the gauge-equivariant model maintains a more gradual degradation, reflecting improved scalability under growing system complexity.

## 6 Conclusion

This work presented Neural Hamiltonian Diffusion (NHD), a unified framework that integrates geometry-aware diffusion processes with structure-preserving Hamiltonian learning. We formulated a diffusion process lifted to the frame bundle and constructed neural Hamiltonian vector fields that are equivariant under frame transformations. We provided theoretical results characterizing the generalization properties of the proposed method, including uniform-in-time bounds and framewise deviation under gauge transformations. Experiments results across diverse scientific domains demonstrated that our NHD consistently improves physical fidelity and predictive stability compared to Euclidean or non-Hamiltonian baselines.

## Acknowledgments

This work was supported by ICT Creative Consilience Program through the Institute of Information &amp; Communications Technology Planning &amp; Evaluation (IITP) grant funded by the Korea government (MSIT) (IITP-2025-RS-2020-II201819)

## References

Jean-Michel Bismut. Mécanique aléatoire, volume 866 of lecture notes in mathematics, 1981.

- Zhengdao Chen, Jian Zhang, Martin Arjovsky, Léon Bottou, and Joan Bruna. Symplectic recurrent neural networks. arXiv preprint arXiv:2010.07003 , 2021.
- Taco Cohen, Maurice Weiler, Berkay Kicanaoglu, and Max Welling. Gauge equivariant convolutional networks and the icosahedral cnn. In International conference on Machine learning , pages 1321-1330. PMLR, 2019.
- Wendy D Cornell, Piotr Cieplak, Christopher I Bayly, Ian R Gould, Kenneth M Merz, David M Ferguson, David C Spellmeyer, Thomas Fox, James W Caldwell, and Peter A Kollman. A second generation force field for the simulation of proteins, nucleic acids, and organic molecules. Journal of the American Chemical Society , 117(19):5179-5197, 1995.
- Miles Cranmer, Alvaro Sanchez-Gonzalez, Peter Battaglia, Rui Xu, Kyle Cranmer, David Spergel, and Shirley Ho. Lagrangian neural networks. In International Conference on Learning Representations (ICLR) , 2020.
- Marco David and Florian Méhats. Symplectic learning for hamiltonian neural networks. Journal of Computational Physics , 494:112495, 2023.
- Valentin De Bortoli, James Thornton, Jeremy Heng, Alexandre Bouchard-Côté, Paul Vicol, and Yee Whye Teh. Riemannian score-based generative modeling. arXiv preprint arXiv:2202.02763 , 2022.
- Eva Dierkes, Christian Offen, Sina Ober-Blöbaum, and Kathrin Flaßkamp. Hamiltonian neural networks with automatic symmetry detection. arXiv preprint arXiv:2301.07928 , 2023.
- Marc Finzi, Samuel Stanton, Pavel Izmailov, and Andrew Gordon Wilson. Generalizing hamiltonian mechanics with learned symplectic structure. In International Conference on Machine Learning (ICML) , pages 3146-3154. PMLR, 2020.
- Nic Fishman and John P. Cunningham. Diffusion models for constrained domains. arXiv preprint arXiv:2304.05364 , 2023.
- Fabian Fuchs, Daniel Worrall, Volker Fischer, and Max Welling. Se (3)-transformers: 3d rototranslation equivariant attention networks. Advances in neural information processing systems , 33: 1970-1981, 2020.
- Samuel Greydanus, Misko Dzamba, and Jason Yosinski. Hamiltonian neural networks. In Advances in Neural Information Processing Systems , volume 32, 2019.
- Ernst Hairer, Christian Lubich, and Gerhard Wanner. Geometric numerical integration: structurepreserving algorithms for ordinary differential equations , volume 31. Springer Science &amp; Business Media, 2006.
- Jiaqi Han, Minkai Xu, Aaron Lou, Haotian Ye, and Stefano Ermon. Geometric trajectory diffusion models. arXiv preprint arXiv:2410.13027 , 2024.
- Elton P Hsu. Stochastic analysis on manifolds . Number 38. American Mathematical Soc., 2002.
- Chin-Wei Huang, Milad Aghajohari, Joey Bose, Prakash Panangaden, and Aaron Courville. Riemannian diffusion models. In Advances in Neural Information Processing Systems , 2022.
- Martin Karplus and J Andrew McCammon. Molecular dynamics simulations of biomolecules. Nature Structural Biology , 9:646-652, 2002.

- Zi-Yu Khoo, Dawen Wu, Jonathan Sze Choong Low, and Stéphane Bressan. Separable hamiltonian neural networks. arXiv preprint arXiv:2309.01069 , 2023.
- Leon Klein, Andrew Foong, Tor Fjelde, Bruno Mlodozeniec, Marc Brockschmidt, Sebastian Nowozin, Frank Noé, and Ryota Tomioka. Timewarp: Transferable acceleration of molecular dynamics by learning time-coarsened dynamics. Advances in Neural Information Processing Systems , 36: 52863-52883, 2023.
- Joan-Andreu Lázaro-Camí and Juan-Pablo Ortega. Stochastic hamiltonian dynamical systems. Reports on Mathematical Physics , 61(1):65-122, 2008.
- Aaron Lou, Minkai Xu, Adam Farris, and Stefano Ermon. Scaling riemannian diffusion models. In Thirty-seventh Conference on Neural Information Processing Systems , 2023.
- James A Maier, Carlos Martinez, Krishna Kasavajhala, Lauren Wickstrom, Kevin E Hauser, and Carlos Simmerling. ff14sb: improving the accuracy of protein side chain and backbone parameters from ff99sb. Journal of chemical theory and computation , 11(8):3696-3713, 2015.
- Pascal Massart. About the constants in talagrand's concentration inequalities for empirical processes. The Annals of Probability , 28(2):863-884, 2000.
- Emile Mathieu, Vincent Dutordoir, Michael John Hutchinson, Valentin De Bortoli, Yee Whye Teh, and Richard E Turner. Geometric neural diffusion processes. In Thirty-seventh Conference on Neural Information Processing Systems , 2023.
- Frank Noé, Alexandre Tkatchenko, Klaus-Robert Müller, and Cecilia Clementi. Machine learning for molecular simulation. Annual Review of Physical Chemistry , 71:361-390, 2020.
- Bernt Øksendal and Bernt Øksendal. Stochastic differential equations . Springer, 2003.
- Sung Woo Park, Seongmin Kim, Junhyun Lee, Seonguk Joo, Jaehyung Choi, and Eunho Yang. Riemannian neural sde: Learning stochastic representations on manifolds. In Advances in Neural Information Processing Systems (NeurIPS) , 2022.
- Frans Pretorius. Evolution of binary black hole spacetimes. Physical Review Letters , 95(12):121101, 2005.
- Hanno Rein and Shang-Fei Liu. Rebound: an open-source multi-purpose n-body code for collisional dynamics. Monthly Notices of the Royal Astronomical Society , 485(4):5490-5513, 2019.
- Luciano Rezzolla and Olindo Zanotti. Relativistic hydrodynamics . Oxford University Press, 2013.
- J.J. Sakurai and Jim Napolitano. Modern quantum mechanics . Cambridge University Press, 2017.
- Romelia Salomon-Ferrer, David A Case, and Ross C Walker. An overview of the amber biomolecular simulation package. Wiley Interdisciplinary Reviews: Computational Molecular Science , 3(2): 198-210, 2013.
- Vıctor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E (n) equivariant graph neural networks. In International conference on machine learning , pages 9323-9332. PMLR, 2021.
- Fan Simiao, Linlin Huo, Fei Wang, Lei Zheng, and Shuang Yang. Hamiltonian neural network 6-dof rigid-body dynamic modeling based on energy variation estimation. International Journal of Intelligent Systems , 2023. doi: 10.1155/2023/8882781.
- Michel Talagrand. The generic chaining: upper and lower bounds of stochastic processes . Springer Science &amp; Business Media, 2005.
- Cong Tian, Krishna Kasavajhala, Kaërl A A Belfon, Lauren Raguette, Hao Huang, Andre N Migues, Jessica Bickel, Yumin Wang, Jennifer Pincay, Qi Wu, et al. ff19sb: amino-acid-specific protein backbone parameters trained against quantum mechanics energy surfaces in solution. Journal of chemical theory and computation , 16(1):528-552, 2019.

- James Townsend, Eric Vogeley, Christoph Wehmeyer, and Frank Noé. Geometric constraints in molecular simulations via differential geometry. Journal of Chemical Physics , 154(23):234108, 2021.
- Tycho F. A. van der Ouderaa, Mark van der Wilk, and Pim De Haan. Noether's razor: Learning conserved quantities. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024. URL https://openreview.net/forum?id=dpvqBkEp1f .
- Yiming Wang, Yujie Sun, Yiqun Liu, and Hao Xu. Symplectic learning for hamiltonian neural networks. Journal of Computational Physics , 484:112060, 2023.
- Jon Wellner et al. Weak convergence and empirical processes: with applications to statistics . Springer Science &amp; Business Media, 2013.
- Chenxin Xu, Robby T Tan, Yuhong Tan, Siheng Chen, Yu Guang Wang, Xinchao Wang, and Yanfeng Wang. Eqmotion: Equivariant multi-agent motion prediction with invariant interaction reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 1410-1420, 2023.
- Yicheng Zhong, Debasmit Dey, Rudrasis Chakraborty, and Vikas Singh. Symplectic ode-net: Learning hamiltonian dynamics with control. In International Conference on Learning Representations (ICLR) , 2020.
- Yao Zhou, James Zou, Will Grathwohl, and Frank Noé. Torchmd: A deep learning framework for molecular simulations. arXiv preprint arXiv:2012.12106 , 2020.

## A Backgrounds

## A.1 Stochastic Riemannian Geometry, Hamiltonian Dynamics

Let M be a smooth d -dimensional Riemannian manifold with metric g . The tangent bundle T M is the disjoint union of all tangent spaces T x M for x ∈ M . The orthonormal frame bundle O ( M ) is a principal O ( n ) -bundle over M , where each point U ∈ O ( M ) is an ordered orthonormal basis ( e 1 , . . . , e n ) of T x M at some x ∈ M . The canonical projection π : O ( M ) → M maps a frame to its base point. This allows lifting trajectories from M to O ( M ) in a geometrically structured way. A connection on O ( M ) decomposes the tangent space T U O ( M ) into vertical and horizontal components:

<!-- formula-not-decoded -->

The connection form ω ∈ Ω 1 ( O ( M ); so ( n )) is a Lie algebra-valued 1-form satisfying:

- ω ( A ∗ ) = A for all A ∈ so ( n ) , where A ∗ is the fundamental vertical vector field,
- R ∗ g ω = Ad g -1 ω under the right action R g of g ∈ O ( n ) .

Avector field X on M lifts horizontally to O ( M ) if ω ( ˜ X ) = 0 . This horizontal lift allows stochastic processes on M to be lifted into O ( M ) while preserving the connection structure.

Let X t ∈ M be a semimartingale on a smooth manifold M . Let U t ∈ O ( M ) denote a process on the frame bundle such that π ( U t ) = X t . We say U t is the horizontal lift of X t if it satisfies the following condition from Hsu [2002]:

<!-- formula-not-decoded -->

where { H i } is the canonical horizontal vector field associated with the standard basis vectors e i ∈ R n , and B t is an R n -valued Brownian motion. This construction allows noise to be defined canonically on R n , lifted horizontally via H i , and transported on the manifold through U t . The projected process X t = π ( U t ) then inherits stochastic dynamics that respect the geometry induced by the connection.

An orthonormal frame U ∈ O ( M ) at x = π ( U ) is represented as an isometry U : R n → T x M , i.e., for a standard basis vector e i ∈ R n , U e i = e x i ∈ T x M . Thus, the process U t evolves according to the geometry of M with stochastic noise applied in the frame coordinates and mapped into the tangent bundle via horizontal transport. This perspective ensures that noise is not only intrinsic to the manifold but also compatible with the Levi-Civita connection and geometric constraints of M .

Hamiltonian Vector Fields on Manifolds. Let H : T ∗ Q → R be a Hamiltonian. The Hamiltonian vector field X H is defined implicitly via the symplectic form ω on T ∗ Q , i . e ., ι X H ω = d H . In local Darboux coordinates ( q i , p i ) , X H takes the standard form:

<!-- formula-not-decoded -->

This is the core generator of deterministic Hamiltonian evolution and provides the basis for its stochastic extension. For smooth functions f, g : T ∗ Q → R , the Poisson bracket is defined as:

<!-- formula-not-decoded -->

Horizontal Lift and Connection form. In Section 2 and Figure 1, we described the evolution of lifted Hamiltonian dynamics on the orthonormal frame bundle O ( M ) . The key construction relies on decomposing a lifted vector field into horizontal and vertical components with respect to the principal connection. Let ̂ G θ ( U t ) ∈ T U t O ( M ) denote the full lifted vector field constructed from the Hamiltonian flow { m , H θ } . This field is not guaranteed to lie in the horizontal distribution H U t := ker ω U t and must be projected to ensure that the resulting SDE respects the manifold connection structure. The canonical projection is defined via the connection 1-form ω ∈ Ω 1 ( O ( M ); so ( d )) , which satisfies:

<!-- formula-not-decoded -->

The vertical component is extracted using ω , and the projection onto the horizontal space is given by:

<!-- formula-not-decoded -->

where ω U t ( ̂ G θ ) ♯ ∈ V U t denotes the vertical lift of the Lie algebra element associated to the vertical component, and ♯ maps Lie algebra elements to fundamental vector fields. This ensures that the resulting direction G Hor θ ∈ H U t lies entirely in the horizontal distribution, satisfying the condition ω ( G Hor θ ) = 0 . In local coordinates used in Eq. (5), the vertical component explicitly appears as the second term involving the connection tensor [Γ M ] ♭ and the vectorized frame vec( E ) . The subtraction in Eq. (5) therefore realizes the above projection in local form, decomposing the lifted vector field into:

<!-- formula-not-decoded -->

This decomposition plays a crucial role in ensuring that the Stratonovich increment d U t = G Hor θ ( U t ) ◦ d B t evolves along a direction consistent with the geometry of T ∗ Q . This geometric consistency is essential for transporting noise on the manifold without introducing spurious curvature-induced distortions, and forms the foundation of the gauge-consistent stochastic Hamiltonian dynamics defined in this paper. Note that the proof of Proposition 2.2 will deliver the detailed calculation of deriving the vanishing connection 1-form to show the validity.

Infinitesimal Generator and Fokker-Planck Equation . The stochastic process X t ∈ M := T ∗ Q governed by the Stratonovich SDE proposed in main manuscript

<!-- formula-not-decoded -->

defines a diffusion process on the symplectic manifold M with Hamiltonian vector field { m , H θ } . The corresponding infinitesimal generator L θ acts on smooth test functions f ∈ C ∞ ( M ) as:

<!-- formula-not-decoded -->

where the index i runs over local coordinates ( q 1 , . . . , q d , p 1 , . . . , p d ) on T ∗ Q , and { m , H θ } i denotes the i -th component of the Hamiltonian vector field. Let ρ t ( x ) ∈ Dens( M ) denote the time-dependent probability density function of X t . Then, the evolution of ρ t is governed by the Fokker-Planck equation associated with the generator L θ :

<!-- formula-not-decoded -->

where L ∗ θ is the formal adjoint of L θ in the L 2 ( M ) sense. Explicitly, using integration by parts, this yields:

<!-- formula-not-decoded -->

where the density ρ t allows the Radon-Nikodym derivative with respect to probability measure P t by the formula d P t = ρ t d x with Lebesgue measure d x . This formulation describes how the probability mass of the stochastic process spreads over the symplectic manifold under the influence of the geometry-aware Hamiltonian noise.

Table 3: Comparison between Euclidean HNNs and Neural Hamiltonian Diffusions (Ours).

| Comparison Item     | Euclidean HNNs          | Neural Hamiltonian Diffusions (Ours)          |
|---------------------|-------------------------|-----------------------------------------------|
| Space               | R 2 d (flat)            | General manifold T ∗ Q                        |
| Structure           | Fixed symplectic form J | Poisson structure from lifted geometry        |
| Noise               | None (deterministic)    | Intrinsic stochasticity (via horizontal lift) |
| Energy Conservation | Pathwise ˙ H = 0        | Statistical: L θ π = 0                        |

## B Experimental Setup

## B.1 Relativistic Dynamics

We consider a Hamiltonian system defined on full spacetime phase space ( M , g ) , where M is a Lorentzian manifold with Schwarzschild metric g . The canonical geodesic Hamiltonian governing N interacting particles takes the form:

<!-- formula-not-decoded -->

where g µν ( x ) is the inverse Schwarzschild metric and p i µ is the four-momentum conjugate to the spacetime coordinate x µ i = ( t i , r i , θ i , ϕ i ) . The second term encodes softened pairwise gravitational interactions.

The inverse Schwarzschild metric in spherical coordinates is given by:

<!-- formula-not-decoded -->

The dynamics follow the relativistic Hamilton equations:

<!-- formula-not-decoded -->

We summarize the physical-to-code variable mapping as:

<!-- formula-not-decoded -->

Each particle i ∈ { 1 , . . . , N } is initialized with a four-position q i = ( t i , r i , θ i , ϕ i ) drawn from:

<!-- formula-not-decoded -->

centered around a stable orbital radius with small angular spread.

Initial four-momentum is sampled to induce slightly perturbed circular orbits:

<!-- formula-not-decoded -->

Table 4: Initial state (reproducible random seed 42 ).

|   Body | x         | y       | z       | w         | ω x       | ω y       | ω z       |
|--------|-----------|---------|---------|-----------|-----------|-----------|-----------|
|      1 | - 0 . 496 | 0 . 647 | 0 . 576 | - 0 . 011 | 0 . 228   | - 0 . 238 | 0 . 040   |
|      2 | 0 . 261   | 0 . 400 | 0 . 812 | - 0 . 314 | - 0 . 141 | 0 . 136   | - 0 . 113 |
|      3 | - 0 . 183 | 0 . 861 | 0 . 318 | 0 . 351   | - 0 . 119 | - 0 . 438 | 0 . 176   |

Frame Bundle Structure and Spatial Diffusion. Let ( M , g ) be a pseudo-Riemannian manifold with signature ( -, + , + , +) . The pseudo-orthonormal frame bundle O (1 , 3) ( M ) is a principal SO(1 , 3) -bundle. A point in this bundle is written as:

<!-- formula-not-decoded -->

and η = diag( -1 , 1 , 1 , 1) is the Minkowski metric. The frame e µ a forms a local orthonormal basis of T x M . To respect relativistic causality, we restrict stochastic diffusion to spatial directions a = 1 , 2 , 3 . Let B t = ( B (1) t , B (2) t , B (3) t ) be Brownian noise on the spatial frame. The spatiallyrestricted horizontal SDE on the frame bundle is then:

<!-- formula-not-decoded -->

No perturbation is applied to the temporal component ( a = 0 ), and the drift term is entirely deterministic, maintaining consistency with the Lorentzian structure. The simulation integrates this Hamiltonian system using a symplectic leapfrog method, with symbolic metric evaluation via SymPy . The background mass M = 1 . 0 determines the Schwarzschild radius r s = 2 GM = 2 . 0 .

## B.2 Spin Dynamics

Our goal in the expereiment is to model the time-evolution of three mutually-interacting rigid-body spins living on the unit 3-sphere S 3 ⊂ R 4 , simulate three dynamical regimes, and render the results through the Hopf fibration. Let q = ( x, y, z, w ) ∈ S 3 be a unit quaternion representing a rigid rotation. We split q into the complex pair ( z 1 , z 2 ) ∈ C 2 , z 1 = x + iy, z 2 = z + iw, so that | z 1 | 2 + | z 2 | 2 = 1 . We treat each spin as a point mass with principal inertia (2 , 1 , 0 . 5) and equip the system with the pair-exchange Hamiltonian defined as follows:

̸

<!-- formula-not-decoded -->

where ω i ∈ R 3 is the spatial angular velocity, I =diag(2 , 1 , 0 . 5) , and J &gt; 0 promotes alignment. Then, the Hamilton equations read ˙ q i = 1 2 Ω( ω i ) q i , ˙ ω i = I -1 τ i , with Ω( ω ) and τ i = -J ∑ j = i ( q i -q j ) 1:3 the coupling torque. To simulate the dynamics, we employ an explicit Euler step of size ∆ t =0 . 02 s and renormalize q i to unit length after each step to avoid drift off S 3 . All runs start from the randomized seed quaternion/velocity ensemble (Table 4 for seed 42 ). Each trajectory contains T = 300 frames. While a unit quaternion ( z 1 , z 2 ) ∈ S 3 ⊂ C 2 projects via the Hopf projection π ( z 1 , z 2 ) = ( 2 ℜ ( z 1 ¯ z 2 ) , 2 ℑ ( z 1 ¯ z 2 ) , | z 1 | 2 -| z 2 | 2 ) ∈ S 2 . , we display each trajectory in polar arg-magnitude coordinates of ( z 1 , z 2 ) : ( θ k , r k ) = (arg z k , | z k | ) , k = 1 , 2 .

## B.3 Toroidal Protein Sequence

We convert the raw Cartesian trajectory contained in traj-arrays.npz into generalized coordinates ( θ, ˙ θ, p, ˙ p ) on the dihedral-torus to support Hamiltonian learning. The .npz file provides positions x t ∈ R N × 3 , velocities v t , and time stamps t ∈ R + . Dihedral angles θ t = ( ϕ t , ψ t , ω t ) ∈ ( -π, π ] 3 are computed from atomic coordinates using standard torsion definitions applied to atom quadruplets A ϕ , A ψ , A ω ⊂ { 1 , . . . , N } 4 , which are in turn extracted from the molecular topology file traj-state0.pdb via MDTRAJ. This topology file also provides element-wise atomic masses { m i } N i =1 for moment of inertia computation.

We first estimate a uniform time step ∆ t = mean k ( t k +1 -t k ) in femtoseconds from the raw time array. Angular velocities for each torsion angle θ ∈ { ϕ, ψ, ω } are then computed by finite differencing:

̸

Table 5: Dimensions of the augmented physical variables.

| Variable             | Symbol   | Shape        |
|----------------------|----------|--------------|
| Angular velocities   | ˙ θ      | ( T - 1) × 3 |
| Angular momenta      | p        | ( T - 1) × 3 |
| Momentum derivatives | ˙ p      | ( T - 2) × 3 |

˙ θ k = ( θ k +1 -θ k ) / ∆ t , followed by periodic unwrapping using ˙ θ k ← wrap ( -π,π ] ( ˙ θ k ) where the wrap function applies atan2(sin · , cos · ) elementwise. The resulting angular velocity matrix ˙ θ ∈ R ( T -1) × 3 is stored under the key torsion\_dots . To compute the scalar moment of inertia I k for a given torsion at time t k , we approximate the rotation axis as the normalized vector e = ( x a 2 -x a 1 ) / ∥ · ∥ and use only the two terminal atoms to define transverse distances. Specifically,

<!-- formula-not-decoded -->

This produces a single scalar inertia for each dihedral and time step. Using this, we compute the conjugate angular momentum as p k = I k ˙ θ k and aggregate the result as a matrix p ∈ R ( T -1) × 3 stored as torsion\_momentum . To obtain generalized forces, we compute the time derivatives of angular momentum by finite differencing:

<!-- formula-not-decoded -->

where we again apply periodic unwrapping. This finally yields the data ˙ p ∈ R ( T -2) × 3 . Table 5 summarizes the shape of the augmented tensors.

## B.4 Experimental Details

All experiments were conducted on a single NVIDIA RTX 5090 GPU using Python 3.11 and PyTorch 2.1.0 with CUDA 11.8 support. The proposed framework is evaluated on a pre-processed protein trajectory dataset embedded in a curved configuration space. We use an 80%/20% temporal split for training and testing. Each sub-trajectory consists of 0 . 8 T frames: the initial frame t 0 serves as the input, and the following 0 . 8 T frames ( t 1:0 . 8 T ) are used for supervision. The model input and target sequences include generalized coordinates, velocities, momenta, and their time derivatives:

<!-- formula-not-decoded -->

where T denotes the total number of time steps in each sequence.

The neural architecture in Algorithm 1 is a Gauge-Equivariant Transformer UNet designed to model Hamiltonian vector fields on curved manifolds by incorporating symmetry-preserving inductive biases. The input to the network is a concatenation of configuration and momentum coordinates ( q, p ) ∈ R 2 d , transformed into a canonical local gauge frame using a Cholesky-based projection with gauge matrix G ∈ R d × d . This projected input is passed through a linear embedding layer and fused with a temporal encoding via sinusoidal or MLP-based TimeEmbedding .

The model consists of an encoder-decoder Transformer with L = 16 residual blocks, each using multi-head self-attention, GELU activations, and layer normalization. Skip connections and projection layers link encoder and decoder stages. Predictions for time derivatives are generated in a local gauge frame and mapped back to the global frame using a Cholesky-based projection, ensuring gauge equivariance. The network uses a hidden size of 128 and contains approximately 3M parameters. Training is performed for 10 5 epochs using the Adam optimizer with learning rate 10 -4 and batch size 128. The loss combines a local alignment term and a long-range reconstruction objective.

<!-- formula-not-decoded -->

To simulate future trajectories, we employ a geometry-aware simulator that integrates stochastic Hamiltonian dynamics via Stratonovich SDEs on the cotangent bundle. This simulator leverages

| Algorithm 1 GAUGE EQUIVARIANT TRANSFORMER UNET                                                                                                 |                                            |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
|------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|---------|-----------|---------|-----|----|----|----|----|----|----|----|----|----|----|----|----|----|-----|----|----|-----|-----|
| Input: x = [ q N , p N ] ∈ 2 dN , time t , bundle metric G Output: [ ̂ ˙ q N , ̂ ˙ p N ]                                                         |                                            |         |           | 1: 2:   |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| 3: // Canonical gauge transform 4: q can ← G ⊤ q N , x can ← [ q can , p N ]                                                                   |                                            |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| 5: // Encoder-decoder trunk (shared context) 6: h ← Linear ( x can )+ TimeEmbed ( t ) 7: h ← Unsqueeze ( h, 1) for i = 1 to L do h ← ( h ) ← h |                                            |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| end for i = 1 to L do h ← Concat ( h, enc L - i +1 ) ; h ← DecProj i ( h ) ; h ← DecBlock i ( h )                                              |                                            |         |           | 9: 10:  |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| for                                                                                                                                            |                                            |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| end for c ← GlobalPool ( h ) // context vector shared by all potentials                                                                        |                                            |         |           | 11: 12: |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| // Shared trunk T ϕ and two heads H sp , H pair                                                                                                |                                            |         |           | 13:     |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| for i = 1 to N do ϕ ←T ( [ q i , p i ] ,                                                                                                       | ▷ single-particle branch // shared weights |         |           | 14: 15: |     |    |    | c  |    |    | )  |    |    |    |    |    | ϕ  |    |     |    |    | i   |     |
| E i sp ←H sp ( ϕ i )                                                                                                                           | // head 1                                  |         |           | 16:     |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| end for for all pairs                                                                                                                          | ▷ pairwise branch                          |         |           | 17: 18: |     | <  | do |    | i, | )  | j  | i  | )  | j  | i  | (  |    | ,  |     | j  | (  |     |     |
| ϕ ij ←T ϕ [ q , q ] , c E ij pair ←H pair ( ϕ ij )                                                                                             | // same trunk                              |         |           | 19:     |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| for                                                                                                                                            | // head 2                                  |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| KE ← 2 i ( p ) G p SP ← ∑ E i , PI ← ∑ E                                                                                                       | // kinetic                                 |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| i sp i<j ij pair                                                                                                                               |                                            |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
|                                                                                                                                                | back to global frame                       |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| H θ = KE+SP+PI                                                                                                                                 |                                            |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| 27: return [ ̂ ˙ q , ̂ ˙ p ]                                                                                                                     | //                                         |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| N N                                                                                                                                            |                                            |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| ̂ ˙ q ← G ̂ ˙ q                                                                                                                                  |                                            |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| N N                                                                                                                                            |                                            |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| 26:                                                                                                                                            |                                            |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| q N θ                                                                                                                                          |                                            |         |           | 25:     |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| ∇ H                                                                                                                                            |                                            |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| θ , ̂ ˙ p =                                                                                                                                     |                                            |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| N                                                                                                                                              |                                            |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| p                                                                                                                                              |                                            |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| = ∇ N H                                                                                                                                        |                                            |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| ̂                                                                                                                                               |                                            |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
| ˙ q N                                                                                                                                          |                                            |         |           |         |     |    |    |    |    |    |    |    |    |    |    |    |    |    |     |    |    |     |     |
|                                                                                                                                                |                                            | 22: 1 ∑ | i ⊤ - 1 i | 23:     | 24: |    |    |    |    |    |    |    |    |    |    |    |    |    | end |    |    | 21: | 20: |

gauge-equivariant drift fields and lifts the dynamics into a frame bundle, where the noise is transported horizontally. Chart transitions are handled using a manifold-aware update rule. The numerical integrator uses a fixed step size ∆ t = 2 . 0 fs and isotropic Gaussian noise of magnitude 10 -2 √ ∆ t . Performance is evaluated using two common metrics: the Average Displacement Error (ADE) and the Final Displacement Error (FDE), defined as

<!-- formula-not-decoded -->

Random seeds for torch , numpy , and random are fixed to 42 for reproducibility. The codebase and configuration files will be made publicly available at https://github.com/Anonymous/HDM .

Simulation of Neural Hamiltonian Diffusion. Algorithm 2 outlines the simulation process of our Hamiltonian Diffusion Model (HDM) based on Proposition 2.2. Starting from an initial state ( q 0 , p 0 ) , we construct a lifted representation U 0 on the frame bundle using the inverse metric and its Cholesky decomposition. At each step, the model predicts a gauge-equivariant Hamiltonian drift, and isotropic noise is projected onto the horizontal space to ensure geometric consistency. The state is updated via a Stratonovich integrator that respects the manifold structure, and chart transitions are handled as needed. The algorithm outputs both the lifted trajectory on the frame bundle and its projection onto the base manifold, enabling structured simulation over curved geometric spaces.

Initialisation of the Lifted State U 0 = ( x 0 , E 0 ) ∈ O ( M ) . The horizontal SDE of Proposition2.2 requires an initial condition on the orthonormal-frame bundle . This amounts to choosing (i) a base point x 0 ∈ M -which fixes the particle's initial configuration-and (ii) an orthonormal frame E 0 ∈ SO( T x 0 M ) that serves as the local gauge in which all subsequent tangent-space computations are expressed.

Algorithm 2 SIMULATE NEURAL HAMILTONIAN DIFFUSION ( q 0 , p 0 , t 0 , T, n f ; H θ )

Require: Initial state ( q 0 , p 0 ) ∈ T ∗ Q , start-end times ( t 0 , T ) , # Stratonovich steps n f , parameterized Hamilto-

```
nian H θ Ensure: Lifted trajectory { U t } t 0 ≤ t ≤ T and its projection { X t = π ( U t ) } 1: // Initial frame lift 2: g ♯ ← g ♯ ( q 0 , p 0 ) // inverse bundle metric 3: ν 0 ← chol ( g ♯ ) // local orthonormal frame 4: U 0 ← ( q 0 , p 0 , vec( ν 0 )) // coordinates on O ( T ∗ Q ) 5: C 0 ← 1 d // initial chart index 6: // Stratonovich SDE integration 7: for k = 0 to n f -1 do 8: t ← t 0 + k ∆ t, ∆ t ← T -t 0 n f 9: q t , p t , U t ← unpack( U k ) 10: // Horizontal Hamiltonian drift 11: [ w q , w p ] ← MODELFORWARD ( [ q t , p t ] , t, U t ) 12: G hor θ ( U k ) ← [ w q , -w p ] // { m , H θ } part 13: // Horizontal diffusion term 14: H t ← H frame ( q t , U t ) // horizontal projector 15: ξ ∼ N (0 , I 2 d ) , sto = √ ∆ t H t ξ 16: // Stratonovich increment 17: U k +1 ← U k + G hor θ ( U k ) ∆ t +sto 18: // (optional) Chart update on frame bundle 19: C k +1 , U k +1 ← CHARTUPDATE ( U k +1 , C k ) 20: end for return { U k } n f k =0 , { X k = π ( U k ) }
```

1. Base point x 0 . In practice x 0 is dictated by the task: for trajectory prediction one sets x 0 = x data (the observed configuration at time 0 ); for sampling or controlled experiments one may draw x 0 from a prescribed distribution on M (e.g. the uniform measure on S 2 ).
2. Orthonormal frame E 0 . Given a coordinate chart m = ( q 1 , . . . , q d ) around x 0 with metric matrix g ( x 0 ) , one constructs E 0 by orthonormalising the coordinate basis via the Gram-Schmidt (or Cholesky) procedure:

<!-- formula-not-decoded -->

For the sphere example ( d = 2 ) with ( χ, φ ) coordinates one obtains

<!-- formula-not-decoded -->

3. Phase-space variables. If the model evolves on T ∗ M one also specifies the initial momentum p 0 ∈ T ∗ x 0 M . Typical choices are (a) the empirical momentum if one starts from real data, or (b) a draw from the canonical Gibbs distribution p 0 ∼ N ( 0 , g -1 ( x 0 ) ) , which is consistent with the kinetic term 1 2 p ⊤ g -1 p in the Hamiltonian.
4. Vectorised form for the SDE. For implementation the frame is flattened, vec( E 0 ) ∈ R d 2 , and concatenated with ( q 0 , p 0 ) to produce the full initial vector fed into Algorithm ?? . Because E 0 is already orthonormal, the integrator starts in the horizontal sub-bundle, and the structural properties guaranteed by Proposition2.2 are preserved from the first step onward without any corrective projection.

## C Lemmas

Lemma C.1. Let Γ := Γ i jk be the connection (i.e., Christoffel symbols) of a smooth configuration manifold ( Q , g ) , and let R i jkl be the components of the Riemann curvature tensor. Let us denote ∥ Γ ∥ ∞ := sup x ∈Q max i,j,k | Γ i jk ( x ) | , and ∥ ∂ Γ ∥ ∞ := sup x ∈Q max i,j,k,l | ∂ k Γ i jl ( x ) | . Then the following sup-norm inequality holds:

<!-- formula-not-decoded -->

Proof of Lemma C.1. Recall that, in a local coordinate chart on a smooth Riemannian manifold ( Q , g ) , the components of the Riemann curvature tensor are

<!-- formula-not-decoded -->

where Γ i jk are the Christoffel symbols. Define the sup-norms

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

For the derivative part of R i jkl , we have

<!-- formula-not-decoded -->

For other quadratic parts, we control the terms by showing that

<!-- formula-not-decoded -->

Applying above results and then we take the maximum over all indices at each point to have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, this gives

If ∥ Γ ∥ ∞ ≡ 0 (so every Γ i jk vanishes identically), then ∂ k Γ i jl ≡ 0 as well, and R i jkl ≡ 0 by ( ∗ ) . Consequently ∥R∥ ∞ ≡ 0 . This concludes the proof.

Lemma C.2. Let ( M,g ) be a Riemannian manifold. Fix a smooth reference curve γ : [0 , T ] → M and define

<!-- formula-not-decoded -->

where X t is a solution to the proposed Stratonovich Hamiltonian Diffusion Model (HDM). Assume that X H ( γ ( t ) ) = 0 for all t ∈ [0 , T ] . Then, to first order in the stochastic differential, we have

<!-- formula-not-decoded -->

Proof. We start by recalling the behavior of the exponential map and its derivative. For v ∈ T γ ( t ) M , define the geodesic γ v ( s ) := exp γ ( t ) ( sv ) . Given ξ ∈ T exp γ ( t ) ( v ) M , the differential of the exponential map satisfies

<!-- formula-not-decoded -->

where J ξ denotes the unique Jacobi field along γ v satisfying the initial conditions

<!-- formula-not-decoded -->

At the origin v = 0 , the exponential map behaves simply:

<!-- formula-not-decoded -->

This tells us that near v = 0 , the differential D exp γ ( t ) ( v ) is close to the identity up to second-order terms. Thus, for y close to γ ( t ) (writing v = exp -1 γ ( t ) ( y ) ), the differential of the inverse exponential map satisfies

<!-- formula-not-decoded -->

Next, we apply a Taylor expansion to the Hamiltonian vector field X H around γ ( t ) . Since J t = exp -1 γ ( t ) ( X t ) represents a small deviation, we have

<!-- formula-not-decoded -->

By the assumption X H ( γ ( t ) ) = 0 , this simplifies to

<!-- formula-not-decoded -->

Thus, the stochastic differential d X t is given by

<!-- formula-not-decoded -->

Now, we apply the differential of the inverse exponential map to both sides. Using (22), we find

<!-- formula-not-decoded -->

Finally, since J t = O ( | B t | ) under small-noise scaling, the remainder term O ( ∥ J t ∥ 3 ) ◦ dB t becomes negligible compared to dB t in the Stratonovich limit. Thus, we conclude the desired first-order approximation:

<!-- formula-not-decoded -->

Lemma C.3. Let ( Q , g ) be a d -dimensional Riemannian manifold. Assume that the scalar curvature Scal( x ) satisfies

<!-- formula-not-decoded -->

for some constant S ≥ 0 . Then the operator sup-norm of the Riemann curvature tensor satisfies the estimate

<!-- formula-not-decoded -->

In particular, the curvature-induced deviation term ( B ) in the stochastic Jacobi analysis can be uniformly bounded in terms of the scalar curvature bound S .

Proof. Recall that for any point x ∈ Q and any orthonormal basis { e i } d i =1 of T x Q , the scalar curvature is given by

<!-- formula-not-decoded -->

where K ( e i , e j ) denotes the sectional curvature of the 2-plane spanned by e i and e j :

<!-- formula-not-decoded -->

There are exactly ( d 2 ) = d ( d -1) 2 independent pairs ( i, j ) , and each sectional curvature contributes linearly to the scalar curvature. Therefore, taking absolute values and using the triangle inequality, we obtain

<!-- formula-not-decoded -->

which rearranges to

Since by definition

<!-- formula-not-decoded -->

up to constants depending on the wedge norm ∥ u ∧ v ∥ = 1 for orthonormal pairs, we obtain

<!-- formula-not-decoded -->

where the factor 2 arises from symmetrization conventions in the definition of the Riemann tensor versus the sectional curvature. Thus, the curvature tensor's sup-norm is explicitly controlled by the scalar curvature bound S .

Lemma C.4. Øksendal and Øksendal [2003] Let ( B t ) t ∈ [0 ,T ] be a standard Brownian motion. Then, the mean-squared expectation of Stratonovich SDEs can be calculated as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## D Proofs

This section serves to rigorously formalize all theoretical results that were informally stated in the main text. The goal is to provide complete proofs that fill in the technical gaps and support the conceptual developments discussed earlier.

## D.1 Proof of Proposition 2.2

Proposition D.1 (Horizontal Hamiltonian Diffusion) . Let U t ∈ O ( M ) be the horizontal lift of the diffusion process X t = π ( U t ) , where π : O ( M ) →M is the canonical projection and m denotes a local coordinate function on M . The lifted process U t evolves according to the Stratonovich SDE:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where [Γ M ] ♭ ∈ M(2 d × d 2 ) is the index-lowered connection tensor (i.e., Christoffel symbol), and vec( E ) ∈ M( d 2 , 1) is the vectorized local orthonormal frame. ∇ x and ∇ e denote the vectorized gradients with respect to the configuration point and the frame coordinates, respectively.

Proof. We begin by deriving the horizontal lift of stochastic Hamiltonian dynamics in local coordinates. This involves expressing the dynamics { X t } on the cotangent bundle T ∗ Q under a local trivialization of the frame bundle OM , equipped with a moving frame E t . For explicit analytical and numerical handling, we represent the Stratonovich SDEs in terms of Euclidean coordinates via the horizontal lift operator. In this coordinate system, the horizontal lift of the Stratonovich-type stochastic Hamiltonian dynamics is written as:

<!-- formula-not-decoded -->

The above system describes the horizontal lift of Hamiltonian dynamics where the curvature of the base manifold Q encoded by the Christoffel symbols Γ i jk and the stochastic transport along local orthonormal frames E t jointly modulate the diffusion. The geometric structure is embedded via the lifted noise term on T ∗ Q , ensuring that Brownian motion remains horizontal with respect to the Levi-Civita connection.

To enable stochastic calculus, we now transform the Stratonovich integrals in (34) into Itô form. This allows the introduction of correction terms due to the nonlinear dependence of the coefficients on the stochastic process. The position dynamics in Itô form become:

<!-- formula-not-decoded -->

Similarly, the momentum equation is converted as:

<!-- formula-not-decoded -->

Here, the additional drift induced by curvature and moving frames is absorbed into the auxiliary term V j , defined as:

<!-- formula-not-decoded -->

In Euclidean coordinates with Cartesian frames, all connection coefficients vanish ( Γ = 0 ), and the orthonormal frame E t becomes static. As a result, the geometric correction term V j also disappears, recovering the standard stochastic Hamiltonian flow.

To summarize and simplify the geometric formulation, we now express the entire Hamiltonian diffusion in matrix notation. Let us define the following tensorial representations:

<!-- formula-not-decoded -->

Let a := Γ ♭ · vec( E t ) denote the geometric distortion vector. Then, the matrix form of the Hamiltonian diffusion reads:

<!-- formula-not-decoded -->

where a j denotes the j -th column of a , and D 2 { m, H θ } ∈ R 2 d × 2 d × 2 d is a rank-3 tensor containing Hessian of Hamiltonian. This reformulation makes explicit the second-order geometry-aware correction arising from horizontal noise transport in local coordinates. With the form of Stratonovich's diffusion, one can recover the original definition used in Eq (5) as follows:

<!-- formula-not-decoded -->

The first part of the proof is complete by rewriting the above dynamics presented as Stratonovich SDEs.

As a next step, we aim to establish the theoretical validity of our geometric construction by verifying whether the proposed vector fields G Hor θ indeed lie in the horizontal distribution of the orthonormal frame bundle O ( M ) , where M := T ∗ Q is the cotangent bundle equipped with a Sasaki-type metric.

We introduce canonical coordinates on M as

<!-- formula-not-decoded -->

With block index conventions where q -indices are i, j, k ∈ { 1 , . . . , d } and fibre indices are ¯ ı := d + i , the Sasaki-type metric on M is given by

<!-- formula-not-decoded -->

where g is the base Riemannian metric on Q . The Christoffel symbols of the Levi-Civita connection on M , denoted Γ M α βγ , have the following block structure (with ∂ i := ∂/∂q i ):

<!-- formula-not-decoded -->

Let U = ( x, E ) ∈ O ( M ) be a point on the orthonormal frame bundle, where E = ( E α a ) ∈ R 2 d × 2 d is an orthonormal frame at x . We define the vectorized frame by

<!-- formula-not-decoded -->

which stacks the columns of E . We also define the index-lowered connection tensor

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

via the transformation

Then for any v ∈ R 2 d , we have the key identity:

<!-- formula-not-decoded -->

where Γ M ( v ) α β := Γ M α βγ v γ , and mat( · ) reshapes a vector of length (2 d ) 2 into a 2 d × 2 d matrix.

We now set

<!-- formula-not-decoded -->

which defines the base vector field associated with the Hamiltonian dynamics. Then the lifted horizontal vector field on the frame bundle is given by

G

Hor

θ

=

v

∂

x

α

-

[

I

2

d

⊗

v

]

[Γ

]

vec(

E

)

·

∂

E

To confirm horizontality, we define the temporal derivative of the frame:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The Levi-Civita connection 1-form evaluated at a vector field V is given by

<!-- formula-not-decoded -->

Substituting the expression for ˙ E α b , we obtain

<!-- formula-not-decoded -->

α

⊤

M

♭

.

(48)

which is equivalent to which implies

<!-- formula-not-decoded -->

Therefore, the connection 1-form vanishes:

<!-- formula-not-decoded -->

which confirms that the vector field lies in the horizontal distribution:

<!-- formula-not-decoded -->

In conclusion, the proposed Stratonovich SDE

<!-- formula-not-decoded -->

is the horizontal lift of the Hamiltonian diffusion process on the cotangent bundle T ∗ Q , ensuring geometric consistency with the underlying connection on M .

## D.2 Proof of Proposition 3.1

Proposition D.2 (Time-uniform Generalization Bound of Hamiltonian Diffusion) . Let P t ( θ ⋆ ) := Law ( γ t ( θ ⋆ )) be an associated probability measure of model trajectory, and assume that the proposed neural networks lie in Sobolev ball i . e ., ∥ θ ∥ W 2 ,s ≤ R, ∀ θ ∈ Θ , and first and second derivatives of Hamiltonian are Lipschitzian.

<!-- formula-not-decoded -->

While the first bound captures the uniform deviation of the learned trajectory distribution from the target data measure across time, we next provide a concentration result that controls the deviation between the empirical Wasserstein distance and its population expectation.

<!-- formula-not-decoded -->

where W := W 2 , 2 T ( T ∗ Q ) stands for the squared Wasserstein distance on physical data space T ( T ∗ Q ) , Ω := Ω( σ, λ max , L H , L ∇ H , d, s ) , d &gt; 2 s is a constant depending on metric tensor g and the smoothness, Lipschitz constant of Hamiltonian.

Proof. While the proposed stochastic system is semi-martingale, the chain rule with respect to Poisson bracket ( i . e ., Eq.(2.8) Lázaro-Camí and Ortega [2008]) direct gives the following result:

<!-- formula-not-decoded -->

where δ b a denotes the Kronecker delta. This reveals that the stochastic evolution of the velocity field depends explicitly on the derivatives with respect to momentum coordinates in Eq. (59), highlighting the necessity of incorporating additional physical information. For further discussion, we first give a Sasaki-type fiber metric ( i . e ., norm) on T M = T ( T ∗ Q ) . Then, the squared distance between γ 1 and γ 2 on tangent bundle T M can be naturally defined as follows:

<!-- formula-not-decoded -->

where g -1 Q ( α # , β # ) = g Q ( α, β ) is a dual metric on configuration space. Following by the definition of the norm on tangent bundle of total manifold T ( T ∗ Q ) , the discrepancy between model trajectory γ ( θ and data trajectory ˜ γ can be calculated as follows:

<!-- formula-not-decoded -->

Let assume that the test neural network θ satisfies perfectly matches particle trajectories almost surely i . e ., π q ◦ π ( U t ) = q t ( θ ) = ˜ q t , and assume both mapping ∇ p H( q , · ; θ ) and ∇ q H( p , · ; θ ) is

an injective mapping for each fixed q and p . Consider another neural network θ 0 both matches both particle trajectory and their corresponding momentum behavior e . g ., θ 0 := arg min θ L ( θ ) .

If there exists a inverse Lipschitz constant L -1 H of second mapping, then the assumptions leads to the second and third inequality in Eq. (62):

<!-- formula-not-decoded -->

Given the fact that velocity field lies in the tangent of configuration space ˙ q t ∈ T Q ∼ = R d , taking a supremum with respect neural networks in both side of inequality in Eq. (62) gives

<!-- formula-not-decoded -->

For readability, we simplify the notation as W := W 2 , 2 T ( T ∗ Q ) . Note that the expectation in this context is taken from µ q t ∈ P ( T q t Q ) for each t ∈ [0 , T ] . In first inequality, we normalize the Riemannian inner product with the Euclidean correspondence by using the property: ∥ v ∥ 2 g Q ≤ λ 2 max ( g Q ) ∥ v ∥ 2 E for any vectors v ∈ T p t Q . Next, our goal is to obtain the following type of decomposition

<!-- formula-not-decoded -->

where the time-dependent constant f := f ( t, Γ , ∂ Γ , ∂ I H θ ) depends on the connection form Γ and their derivative ∂ Γ , and the Lipschitz constants of higher-order derivatives Lip ( ∂ I H θ ) , ∀ I ≤ 2 . To this end, we first define four auxiliary processes as follows:

<!-- formula-not-decoded -->

where the mean-squared norm of processes ∥ A ∥ 2 E , ∥ δ B ∥ 2 E , ∥ Z ∥ 2 E , ∥ D ∥ 2 E , ∥∇ Z ∥ 2 E is bounded above with some constants C A , C B , C Z , C D , C ∇ Z . Having the definition in hands, the proposed velocity vector fields for arbitrary network θ can be simplified with the following form:

<!-- formula-not-decoded -->

With the definition δ ˙ q t := ˙ q t ( θ ) -˙ q t ( θ 0 ) for deviation between two velocity vector fields, direct calculation leads to have norm-squared expectation as follows:

<!-- formula-not-decoded -->

Following by the conversion of Stratonovich SDE into Ito's SDE in Lemma C.4, the second term of right-hand in last inequality can be upper-bounded with the following form:

<!-- formula-not-decoded -->

This bound reflects the second-moment structure of the Stratonovich integral, where the dominant contribution arises from the squared noise norm C 2 Z , and the correction terms involve both the norm and gradient of the stochastic vector field Z t .s

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This shows that the magnitude of the noise vector Z t grows quadratically with the model distance ∥ θ -θ 0 ∥ , and is modulated by the geometric curvature ∥ Γ ∥ ∞ and the Hamiltonian smoothness L H .

<!-- formula-not-decoded -->

This decomposition separates the gradient of the stochastic vector field Z t into two terms: one involving the spatial derivative of the geometry-aware coefficient A j,α , and the other involving the gradient of the perturbation δB j,α , both of which are influenced by the manifold structure and the Hamiltonian model.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since the orthonormal frame is locally updated by parallel transport, one can obtain

<!-- formula-not-decoded -->

Therefore, the spatial variation of each frame component E α,i t is entirely determined by the Christoffel symbol and remains uniformly bounded under smooth parallel transport.s

<!-- formula-not-decoded -->

Hence, the total spatial gradient ∇ Z t scales linearly with the parameter deviation ∥ θ -θ 0 ∥ , and is tightly controlled by the geometry through ∥ Γ ∥ ∞ , ∥ ∂ Γ ∥ ∞ and the Lipschitz constants L H , L ∇ H .

<!-- formula-not-decoded -->

For L = L H ∨ L ∇ H , f is defined as follows:

<!-- formula-not-decoded -->

Now, our goal is to simplify the inequality, making f is related to the curvature skewness. Using Lemma C.1, the curvature tensor provides a lower bound on ∥ ∂ Γ ∥ ∞ , which allows us to eliminate the explicit derivative dependence and reparameterize R 1 in terms of ∥R∥ ∞ and ∥ Γ ∥ ∞ .

<!-- formula-not-decoded -->

This inequality gives a curvature-dependent upper bound on the Riemann tensor norm in terms of the supremum of the partial derivatives and ∥ Γ ∥ ∞ , which allows us to replace ∥R∥ ∞ by ∥ Γ ∥ ∞ in subsequent expressions. By combining the previous inequality with the triangle inequality, we obtain a uniform bound on ∥ ∂ Γ ∥ ∞ + ∥ Γ ∥ ∞ (1 + ∥ Γ ∥ ∞ ) that is linear in ∥R∥ ∞ and ∥ Γ ∥ ∞ , facilitating simplification of higher-order terms.

<!-- formula-not-decoded -->

Thus, the curvature dependence can be simplified to a function of ∥ R ∥ ∞ and ∥ Γ ∥ ∞ only.

<!-- formula-not-decoded -->

Squaring both sides, we derive a bound for the squared norm ( ∥ ∂ Γ ∥ ∞ + ∥ Γ ∥ ∞ (1 + ∥ Γ ∥ ∞ )) 2 , which ensures that second-order curvature contributions can be expressed as a function of ∥R∥ ∞ and ∥ Γ ∥ 2 ∞ alone.

<!-- formula-not-decoded -->

As a result, the original function f ( t, Γ , R ) can now be written in terms of ∥ Γ ∥ ∞ only, up to a multiplicative constant, removing explicit curvature dependence from the generalization bound.

We now consider the set of neural networks θ 0 constrained within a metric ball defined by a Sobolevtype functional distance. Let Θ denote the set of such neural networks whose Sobolev norm and supremum norm are simultaneously bounded by a constant R &gt; 0 . We then define the associated function class with respect to the L 2 -norm, and consider a probability space (Θ , Σ µ , ˜ P µ ) supported on Θ .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the random variable θ ( ω ) ∈ Σ µ , let us define auxiliary processes as

<!-- formula-not-decoded -->

For the metric d F (( t, F ) , ( s, G )) = ∥ F ( θ t ) -G ( θ s ) ∥ L 2 ( µ q ) where s ≤ t ∈ [0 , T ] , the ε -covering number on product space [0 , T ] ×F r θ can be interpreted as a product of two sub-coverings in separate spaces:

<!-- formula-not-decoded -->

This shows that the metric entropy can be decomposed as summation of two sub-terms:

<!-- formula-not-decoded -->

where the algebraic constraint on ε 1 and ε 2 is given as

<!-- formula-not-decoded -->

As a next step, we derive the upper bound of expectation for the variable Z , assuming that F ( θ t ) has controlled by Gaussian-like long-tail property, Proposition 1.2.1 Talagrand [2005]). Specifically, we apply Dudley's entropy integral bound to have the following result:

<!-- formula-not-decoded -->

where the expectation ˜ E θ is taken with the probability measure ˜ P µ . The third inequality naturally follows from the embedding W s, 2 ↪ → L 2 and the fact that φ ( a ) = ∥ θ -a ∥ 2 L 2 is 1 -Lipschitz for some a ∈ B W s, 2 ( r ) where we consider the function composition F θ = φ ◦ B W s, 2 ( r ) . In the fourth

inequality, we follow the entropy number in metric ball with radius r in Sobolev space W s, 2 Wellner et al. [2013], and taking supremum under the constraint ∥·∥ W 2 ,s ≤ R .

Since the final expression is non-integrable in general case, we only provide their approximation bound with Taylor expansion in the case when the first term in square root nominates the other term.

<!-- formula-not-decoded -->

where d &gt; 2 s , and we simply set the variable β ε = 0 . 5 . After optimizing with respect to the radius r the right-hand side, we finally have the time-uniform upper-bound of empirical estimates.

<!-- formula-not-decoded -->

Let us now turn our attention to the empirical concentration behavior of the Wasserstein distance, initiating our analysis with the standard symmetrization lemma.

<!-- formula-not-decoded -->

where the empirical version of metrics can be rewritten as following form:

<!-- formula-not-decoded -->

Here, a i ∼ U [ {± 1 } ] and R n are denote both Rademacher variables and their corresponding empirical Rademacher complexity. As with the similar calculation conducted in Eq. (89), the expectation of X admits an upper bound that involves both the function class radius R and sample complexity n .

<!-- formula-not-decoded -->

The result was obtained by using the identical metric entropy calculated in Eq (89) to derive the upper bound. Note that the additional assumption of Gaussian-like property is not considered here as opposite to first inequality in Eq. (89). Combining the result in Eq. (63), Eq. (64) and the definition of auxiliary processes in Eq. (85), we obtain two inequalities

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, we show the exponential probability inequality associated with ( L 2 ( µ q ) , Σ µ , ˜ P µ ) by introducing the classical result from Theorem 3 Massart [2000]. Here, the probability space is considered as one specific choice of generic metric space.

<!-- formula-not-decoded -->

where σ 2 = sup F ∈F r θ Var ( F ( θ )) is a maximal variance of function class, and two positive constants ε, x &gt; 0 are arbitrary. Collecting the result obtained from Eq. (89), (95), (96), (98), we have

<!-- formula-not-decoded -->

Our goal is to find the optimal constant D &gt; 0 such that the quadratic expression in right-hand side serves as an upper bound for the left-hand side polynomial of linear order in Eq. (100).

<!-- formula-not-decoded -->

Here, the constants A , B , and C capture the contributions from the empirical moments, noise level, and smoothness complexity, and are given by:

<!-- formula-not-decoded -->

By minimizing the right-hand side with respect to the free variable x , we obtain the optimal constant D that balances the quadratic and linear terms:

<!-- formula-not-decoded -->

Substituting the identity Dx 2 = δ into the exponential tail inequality, we arrive at the following probabilistic bound for ( R1 ) :

<!-- formula-not-decoded -->

Observing that √ 9 B 2 +32 AC ≥ √ 32 AC and setting ϵ = 1 , we obtain the following upper bound for D ( α, A ) by simplifying the denominator expression:

<!-- formula-not-decoded -->

By substituting the upper bound of D ( α, A ) obtained in Eq. (104) into the general exponential inequality in Eq. (103), we derive the following explicit bound on the probability of the event ( R1 ) , which reflects the asymptotic decay behavior in terms of R and the structural parameters:

˜

P

µ

[(

R1

)

≤

δ

]

<!-- formula-not-decoded -->

We now further simplify the expression by isolating the leading-order terms and observing that the dominant contribution arises from the linear dependence on R 1 / 2 (log R ) 1 / 4 in the denominator. This leads to a more interpretable asymptotic bound expressed in terms of the supremum norm of the function class.

<!-- formula-not-decoded -->

Similarly, we consider the case where the effective complexity α scales with the number of samples n as α = R √ log R/n . This reflects a regime where the resolution increases with sample size, leading to the following generalization bound:

<!-- formula-not-decoded -->

As before, we simplify the expression by extracting the leading dependence on R , log R , and n to arrive at an asymptotic bound that reveals the effect of sample size scaling on the generalization rate:

<!-- formula-not-decoded -->

where Ω := Ω( σ, λ max , L H , L ∇ H , d, s ) , d &gt; 2 s . This final bound highlights that under the samplesize-aware complexity scaling α = R √ log R/n , the generalization error decays exponentially in the effective resolution scale R and the logarithmic complexity log R , with an additional improvement in rate proportional to n 1 / 4 . The result reveals how incorporating geometric inductive bias and adaptive complexity can yield sharper generalization guarantees in high-dimensional structured models.

## D.3 Proof of Proposition 3.2

Proposition D.3 (Worst-case geodesic deviation under frame rotations) . Let γ : [0 , T ] → T ∗ Q be a reference physical data represented as a geodesic. For any Hamiltonian H θ define the frame-rotated trajectory X t ( h ) by X t ( h ) := ( q t , hp t ) for h ∈ O( d ) . Write κ θ := ( L θ + ∥ R ∥ ∞ C θ ) D , where L θ and C θ are the Lipschitz bounds of ∇ 2 { m, H θ } and { m, H θ } , D is the diameter of some compact domain K , and R is the Riemann tensor. Then, for every t ∈ [0 , T ] ,

<!-- formula-not-decoded -->

where X eq t is generated by a Φ -gauge-equivariant Hamiltonian H θ satisfying the same bound with smaller constants L Φ θ ≤ L θ , C Φ θ ≤ C θ . Thus gauge equivariance minimizes the worst-case geodesic deviation over all frame actions h .

Proof. Let us assume that the physical data trajectory γ : [0 , T ] → T ∗ Q forms a geodesic, which satisfies the vanishing connection:

<!-- formula-not-decoded -->

This identity ensures that the acceleration of γ with respect to the connection vanishes, meaning that γ locally minimizes path length and follows the intrinsic geometry of the manifold. In order to quantify small deviations from the reference geodesic γ , we define a deviation vector field J t as the logarithmic map from γ ( t ) to a nearby perturbed point X t :

<!-- formula-not-decoded -->

This construction allows us to express perturbations within a common tangent space at γ ( t ) , facilitating differential analysis. To derive the stochastic differential equation governing J t , we apply the chain rule for Stratonovich differentials adapted to manifold settings. This yields

<!-- formula-not-decoded -->

where F ( t, x ) = exp -1 γ ( t ) ( x ) and { E k } denotes an orthonormal basis of the tangent space. Expanding each term individually, we observe that the time derivative of the logarithmic map corresponds to the covariant derivative along the base curve γ ( t ) , giving

<!-- formula-not-decoded -->

Additionally, the differentials involving d X t are computed via the pullback under F t , while secondorder corrections involving ∇ 2 F are responsible for curvature effects, although these higher-order terms will vanish to leading order under our assumptions.

To measure the growth of deviations quantitatively, we introduce an energy functional E ( t ) defined by the Riemannian norm of the differential of J t :

<!-- formula-not-decoded -->

Here, g denotes the Riemannian metric lifted to the tangent bundle, such as the Sasaki metric if necessary. For notational clarity, we introduce two auxiliary processes:

<!-- formula-not-decoded -->

The term K t represents the covariant derivative of J t along the trajectory γ ( t ) , while L t captures how the Hamiltonian vector field varies along the perturbation direction J t . Following by standard estimation of SDEs ( i . e ., dt 2 = 0 , dt ◦ d B t = 0 , ( d B t ) 2 = dt ), we have

<!-- formula-not-decoded -->

Finally, differentiating once more with respect to time and expanding the covariant derivatives using standard curvature identities yields

<!-- formula-not-decoded -->

where R denotes the Riemannian curvature tensor. This equation connects the second derivative of the energy with the curvature of the manifold and the structure of the Hamiltonian flow.

While the lifted Riemannian metric g is compatible with the Levi-Civita connection (i.e., ∇ g = 0 ), we can relate the time derivative of the metric pairing along the trajectory as follows:

<!-- formula-not-decoded -->

where U = V = { m, H θ } denotes the Hamiltonian vector field evaluated along the curve γ ( t ) . This relation reflects the fundamental property of metric compatibility and provides a way to track how the energy associated with the Hamiltonian flow evolves along the trajectory. To simplify further, we notice that the covariant derivative of a vector field composed with γ ( t ) can be expanded by the product rule of covariant derivatives along curves:

<!-- formula-not-decoded -->

where W = { m, H θ } ( γ ( t )) , and R denotes the Riemannian curvature tensor. This decomposition separates the effects of directional covariant changes along the flow from the intrinsic curvatureinduced distortions arising from the manifold's geometry.

Given the above expansion, we can derive an upper bound for the second derivative of the energy functional ¨ E ( t ) in terms of the norms of relevant geometric quantities:

<!-- formula-not-decoded -->

Here, term (A) corresponds to the covariant second derivative contribution, while term (B) captures the effect of curvature-induced deviations along the geodesic trajectory. These two contributions govern the overall behavior of the energy growth along the stochastic Hamiltonian flow.

Applying Grönwall's inequality to this differential inequality, we obtain an exponential upper bound on the evolution of the energy deviation:

<!-- formula-not-decoded -->

where ( A ) and ( B ) represent the supremum bounds of the two terms over the time interval of interest.

This estimate provides a key control over the divergence between the stochastic trajectory X t and the reference geodesic γ ( t ) under Hamiltonian diffusion dynamics. We now aim to show that imposing an equivariance constraint on the Hamiltonian function intrinsically reduces the upper bound on the growth of geodesic distance. Specifically, the two key terms in the upper bound (92) can be estimated separately as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ∥ R ∥ ∞ denotes the sup-norm of the Riemannian curvature tensor over the manifold. Next, consider the action of the orthogonal group O( d ) on the configuration and momentum coordinates ( q, p ) . For any h ∈ O( d ) , the equivariant Hamiltonian satisfies

<!-- formula-not-decoded -->

where ρ ( h ) = [ h 0 0 I d ] ∈ GL (2 d ) acts on ( q, p ) coordinates. Differentiating both sides with respect to ( q, p ) and using the chain rule, we obtain

<!-- formula-not-decoded -->

where we have used the fact that ρ ( h ) is orthogonal, and that differentiation of ρ ( h -1 ) introduces a right multiplication by ρ ( h ) . Differentiating once more yields the second derivative relation:

<!-- formula-not-decoded -->

Since ρ ( h ) is orthogonal, it preserves the operator norm of tensors. Therefore, for any matrix A ,

<!-- formula-not-decoded -->

This property ensures that the norm of the differential operators remains invariant under frame transformations. Using these observations, we can describe the differential operators acting on the Hamiltonian as

<!-- formula-not-decoded -->

From the orthogonality and isometry of ρ ( h ) , it follows that

<!-- formula-not-decoded -->

and similarly for the second derivative,

<!-- formula-not-decoded -->

Let Φ ≤ O( d ) be the gauge subgroup that leaves the Hamiltonian H θ invariant, that is,

<!-- formula-not-decoded -->

where ρ : Φ → GL(2 d ) is the canonical block embedding ρ ( h ) = diag( h, 1 d ) acting on ( q, p ) , and the right action U ↦→ U · h is free and proper. Consequently, O( d ) ↠ O( d ) / Φ is a principal Φ -bundle with a smooth projection π Φ : O( d ) → O( d ) / Φ , and the base O( d ) / Φ is a smooth homogeneous manifold homeomorphic to the coset space O( d ) / Φ ≃ G/K for some closed subgroup K ≃ Φ .

Endow the bundle with the canonical Ehresmann connection induced by the Levi-Civita connection of the configuration manifold. The tangent space at U ∈ O( d ) splits as T U O( d ) = H U ⊕V U , where V U is the vertical subspace associated with the Φ -action. Since Φ acts by isometries, horizontal lifts preserve the Sasaki metric and the Itô-Stratonovich structure; Brownian noise injected along H U descends canonically to the base O( d ) / Φ . Let [ U ] ∈ O( d ) / Φ denote a frame class and fix a smooth section σ : O( d ) / Φ → O( d ) . Define the lifted stochastic flow by

<!-- formula-not-decoded -->

where X t ( · ) solves the stochastic Hamiltonian system d X t = { m, H θ } ( X t ) ◦ dB t . Because H θ is Φ -equivariant, the Hamiltonian vector fields satisfy

<!-- formula-not-decoded -->

where the lift ρ respects the orthogonal structure. Consequently, both the first and second derivatives obey

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies the norm invariance relations

<!-- formula-not-decoded -->

Let J [ U ] t := exp -1 γ ( t ) ( X [ U ] t ) be the deviation Jacobi field and define the Sasaki energy

<!-- formula-not-decoded -->

Using the stochastic Jacobi equation and applying the curvature identity

<!-- formula-not-decoded -->

we obtain a differential inequality controlling the energy:

<!-- formula-not-decoded -->

where A [ U ] := ∥∇ 2 { m, H θ }∥ op and B [ U ] := ∥ R ∥ ∞ ∥{ m, H θ }∥ g evaluated along the flow. Due to (96), the terms A [ U ] and B [ U ] are invariant across [ U ] ∈ O( d ) / Φ . Thus, applying Grönwall's inequality yields the uniform bound

<!-- formula-not-decoded -->

In contrast, for the non-equivariant model (without Φ symmetry), the best bound achievable is

<!-- formula-not-decoded -->

with A max ≥ A and B max ≥ B in general. Hence, we conclude that

<!-- formula-not-decoded -->

Since O( d ) / Φ is compact (being the quotient of the compact Lie group O( d ) by a closed subgroup), the uniform estimate (97) implies exponential W 2 -stability of the equivariant diffusion. In contrast, the non-equivariant dynamics suffer from a generally larger exponential factor ( A max + B max ) &gt; ( A + B ) . Therefore, gauge-equivariant Hamiltonian learning provides strictly better control over stochastic deviation and ensures tighter uniform generalization error bounds on curved configuration spaces.

We now further refine the upper bounds for the error terms ( A ) and ( B ) by exploiting the geometric structure of the Hamiltonian system. First, we recall that the canonical symplectic matrix J g satisfies ∥ J g ∥ = 1 , as it acts isometrically on T ∗ ( Q ) and preserves the standard Riemannian norm induced by the Sasaki metric. Moreover, since the reference trajectory γ ( t ) is parametrized by arc-length, we have

<!-- formula-not-decoded -->

Furthermore, assuming that both the stochastic trajectory X t and the geodesic γ ( t ) remain within a compact subset K ⊂ T ∗ Q , we have

<!-- formula-not-decoded -->

where D := diam( K ) denotes the geodesic diameter of K . Under these simplifications, the two contributions ( A ) and ( B ) can be bounded more explicitly. Using the fact that ∥ J g ∥ = 1 and ∥ ˙ γ ( t ) ∥ g = 1 , the covariant second derivative term satisfies

<!-- formula-not-decoded -->

where L θ := sup x ∈ T ∗ Q ∥∇ 2 { m , H θ } ( x ) ∥ op denotes the global Lipschitz bound on the Hessian of the Hamiltonian vector field. Similarly, the curvature-induced deviation term satisfies

<!-- formula-not-decoded -->

where ∥ R ∥ ∞ := sup x ∈Q sup ∥ u ∥ g = ∥ v ∥ g =1 ∥ R x ( u, v ) ∥ g is the global sup-norm of the Riemannian curvature tensor, and C θ := sup x ∈ T ∗ Q ∥{ m , H θ } ( x ) ∥ g is the global growth bound on the Hamiltonian vector field. Therefore, the sum ( A ) + ( B ) admits the uniform estimate

<!-- formula-not-decoded -->

which depends linearly on the diameter D of the compact domain K and the regularity constants L θ and C θ . If the Hamiltonian H θ is Φ -equivariant under a subgroup Φ ⊂ O( d ) , the constants L θ

and C θ improve to L Φ θ and C Φ θ respectively, reflecting the additional regularity induced by gauge symmetry. In this case, we obtain the sharper bounds

<!-- formula-not-decoded -->

and the total deviation is controlled by

<!-- formula-not-decoded -->

where L Φ θ &lt; L θ and C Φ θ &lt; C θ due to the symmetry reduction. If the space is assumed to have bounded scalar curvature κ max , one can improve the by following Lemma C.3 as follows:

<!-- formula-not-decoded -->

Substituting these improved bounds into the Grönwall estimate derived earlier, we conclude that the equivariant stochastic Hamiltonian dynamics exhibits exponentially tighter control of the geodesic deviation compared to the general non-equivariant case, with an explicit exponent that scales linearly with the curvature bounds, Hamiltonian regularity, and the diameter of the compact reachable set.

Corollary D.4 (Worst-case W 2 deviation) . Under the setting of Proposition D.4 assume, in addition, that the initial state is deterministic, X 0 = γ (0) . For each frame rotation h ∈ O( d ) , let us set

<!-- formula-not-decoded -->

Then, for every t ∈ [0 , T ] ,

<!-- formula-not-decoded -->

with the same κ θ as in Proposition D.4. Hence gauge equivariance minimises the worst-case 2 -Wasserstein divergence from the reference geodesic over all frame actions h .

Proof. Fix a rotation h ∈ O( d ) and let P t ( h ) = Law [ X t ( h )] . Because the reference point γ ( t ) is deterministic, the unique optimal coupling between P t ( h ) and δ γ ( t ) is the map X t ( h ) ↦→ γ ( t ) . Hence

<!-- formula-not-decoded -->

Proposition C.3 provides the uniform pathwise bound d 2 ( X t ( h ) , γ ( t ) ) ≤ d 2 ( X 0 , γ (0)) e κ θ t for every h . Taking expectations preserves the same right-hand side yields

<!-- formula-not-decoded -->

The same reasoning with the gauge-equivariant trajectory X eq t ( h ) gives an analogous inequality with the smaller constants L Φ θ , C Φ θ ; by Proposition C.3 this already realizes the infimum over h . By taking supremum operator over h ∈ O( d ) in both, it completes the chain of inequalities stated in the corollary.

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (1-2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

## IMPORTANT, please:

- Delete this instruction block, but keep the section heading 'NeurIPS Paper Checklist" ,
- Keep the checklist subsection headings, questions/answers and guidelines below.
- Do not modify the questions and only use the provided macros for your answers .

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The manuscript effectively communicates the primary goals of the research.

## Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification:

## Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: The appendix includes the necessary theoretical background and underlying assumptions for the analysis.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The appendix provides a clear account of how each experimental setup was constructed.

Guidelines:

- The answer NA means that the paper does not include experiments.

- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: The full description (PyTorch code) of the proposed method will be released in GitHub.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification:

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification:

Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification:

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification:

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification:

[TODO]

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the paper poses no such risks.

- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.