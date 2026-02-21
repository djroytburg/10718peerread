## A multiscale analysis of mean-field transformers in the moderate interaction regime

## Giuseppe Bruno

Department of Mathematics and Statistics University of Bern giuseppe.bruno@unibe.ch

## Federico Pasqualotto

Department of Mathematics University of California, San Diego fpasqualotto@ucsd.edu

## Andrea Agazzi

Department of Mathematics and Statistics University of Bern andrea.agazzi@unibe.ch

## Abstract

We study the evolution of tokens through the depth of encoder-only transformer models at inference time by modeling them as a mean-field interacting particle system, and analyzing the corresponding dynamics. More specifically, we consider this problem in the moderate interaction regime , where the number N of tokens is large and the inverse temperature parameter β of the model scales together with N . In this regime, the dynamics of the system displays a multiscale behavior: a fast phase, where the token empirical measure collapses on a low-dimensional subspace, an intermediate phase, where the measure further collapses into clusters, and a slow phase, where such clusters sequentially merge into a single one. We characterize the limiting dynamics in each phase, exemplifying our results with some simulations.

## 1 Introduction

The transformer architecture [49], through its extensive use in Large Language Models, has played a crucial role in the recent, unprecedented developments in machine learning and artificial intelligence. One of the key innovations at the heart of this architecture are self-attention modules [6], allowing to capture long-range dependencies in the data, e.g., in prompts with a large number N of tokens. To further improve their performance, practitioners have implemented these models in different hyperparameter regimes, e.g., choosing the model's inverse temperature parameter β (a parameter that scales the query-key dot products in the self-attention layer) as a function of N [37, 43]. However, the groundbreaking empirical success of these machine learning models remains largely unexplained from the theoretical perspective. In particular, a precise mathematical description of the internal representations learned by transformers, and of how these representations behave in different hyperparameter regimes, is still lacking.

A promising approach to fill this gap was presented in the work [46], where the authors interpret tokens traveling through a deep stack of transformer layers as particles evolving in time and interacting in a mean-field way. A subsequent line of work [24] has then observed that tokens in this model tend to organize into clusters, offering - in a simplified setting - a compelling qualitative explanation of how transformer models build representation of complex input data.

Despite its apparent simplicity, this interacting particle system exhibits a remarkably rich dynamical behavior. Indeed, recent studies have identified distinct dynamical phases, characterized by quali-

tatively different clustering patterns, which depend on specific choices of parameters, timescales, and initial conditions. However, these results often rely on restrictive - and sometimes unrealistic - assumptions, and they typically capture only limited aspects of the collapse dynamics, providing partial views of the transformer's complex dynamical landscape that are difficult to reconcile into a consistent and global dynamical picture. Such a global characterization of the clustering phenomenon in a realistic parameter regime is arguably fundamental to understanding how internal representations form in deep models and how to operate such models in the optimal hyperparameter regimes.

Contributions In this work, we study the dynamics of the mean-field transformer model developed in [46, 25], constrained on the d -dimensional sphere, in the limit of large context size, i.e., when the number N of input tokens is large. Motivated by recent scaling strategies in state-of-the-art LLMs such as SSMax [37] and YaRN [43] (used respectively in Llama 4 and Qwen 3), we consider the setting where the inverse temperature parameter β grows with N . In this regime, our contributions can be summarized as follows:

1. We identify three distinct dynamical phases (respectively denoted the alignment , heat and pairing phases), corresponding to different scales of time as a function of the parameter β . In each phase the model dynamics displays, asymptotically in β , qualitatively distinct behavior, characterized by a different limiting equation.
2. In the alignment phase, occurring on a fast timescale of order O (1) , we prove our main technical result: under general assumptions on the parameter matrices, the finite particle dynamics converges to a linear transport equation modeling the collapse of the token measure onto a low-dimensional manifold dictated by the spectral properties of such matrices. To the best of the authors' knowledge, this phase was not yet identified in the literature.
3. In the heat and pairing phases, occurring respectively on timescales of order O ( β ) and O ( e cβ ) , we identify, under stronger conditions on the parameter values, the limiting dynamics as a forward or backward heat equation on the aligned manifold (leading, in the backward case, to metastable clustering) and a finite-dimensional system of ODEs describing sequential cluster merging along geodesics.

Together, these phases reconcile various previously identified dynamical regimes as different timescales of a single unified dynamical picture. Furthermore, our multi-phase analysis allows to relax some of the restrictive assumptions imposed by previous works, extending their applicability to more realistic scenarios.

Related works The model studied in this paper was introduced in [46], where the authors also identify the heat equation as describing the dynamics of the particle system in R d in the large β regime. This limit emerges as a correction term in their analysis upon subtracting an appropriate leading order term from the prelimit equation. In this paper, by considering the dynamics on the sphere resulting from the inclusion of the layer normalization in our model - we provide a justification for the spontaneous collapse of the system's state to a subspace where this correction term becomes of leading order, dominating the dynamics of the model on a certain timescale.

In [25, 24, 29, 13, 44], the authors identified the clustering behavior occurring in this and closely related models as t →∞ for β, N fixed. Analogous convergence results, under different assumptions, are provided in [18, 34], while quantitative contraction rates for such convergence are given in [16]. These works, however, do not address the dynamically meta-stable phases characterized by partial clustering numerically highlighted in [25]. This intermediate phase is explored in [9] in the large N limit for tokens distributed uniformly at initialization and in [23], where the authors study the formation of meta-stable clusters under the assumption that the system is initialized into wellseparated configurations. Furthermore, in [23] the authors identify different dynamical timescales in the finite N case and characterize for the first time what we refer to in this paper as the pairing phase in the large β limit. In all these cases, however, the results are limited to the setting where the model's key, query and value matrices, Q,K and V , were multiples of the identity. More recently, the work [11] analyzes the stability of fixed points of the same model based on the eigendecomposition of Q,K,V under the weaker assumption that parameters satisfy a modified Wasserstein gradient flow condition ( Q T K = V = D ), but does not study the dynamical landscape connecting such fixed points. Finally, in [3, 4], the authors discuss clustering for hardmax transformers. Our work provides a framework to combine the observations listed above in a unique dynamical picture.

Our modeling approach shares conceptual roots with the neural ODEs literature [14, 21]. However, a key distinction is that we consider N particles interacting through a mean-field PDE, as opposed to one in the previous references. This connects our work to the broader literature on mean-field models for neural networks [45, 36, 17, 2, 19], where timescale analysis has also been a subject of interest (see [7]). In contrast to these works, which typically focus on training dynamics, our study centers on the inference-time evolution of representations through network depth.

Finally, our research relates to the study of moderate scaling limits in interacting particle systems. For instance, Oelschläger [39] proved the convergence of certain systems to the porous medium equation with noise. These results were subsequently extended to cases without noise [41, 40], with different exponents [22], or employing different techniques and equations [12, 10, 42]. Another relevant line of work investigates the convergence of specific interacting particle systems to the heat equation, explored both numerically and theoretically [20, 8, 33].

## 2 Framework and notation

We consider the framework introduced in [46, 25, 24], modeling the transformer architecture as a discrete-time dynamical system governing the evolution of N tokens { x i ( ℓ ) } i =1 ,...,N through its layers via:

<!-- formula-not-decoded -->

where N : R d → S d -1 denotes the normalization operator on the d -dimensional unit sphere S d -1 , L denotes the depth of the transformer architecture and Z β,i ( ℓ ) = ∑ N j =1 e β ⟨ Q ℓ x i ( ℓ ) ,K ℓ x j ( ℓ ) ⟩ is a normalization constant. The dynamics depends on the parameters with matrix values Q ℓ , K ℓ , and V ℓ that represent the query, key, and value matrices at each layer, respectively.

In the spirit of neural ODEs [14], the authors then consider the infinite-depth limit of (1), leading to the following continuous-time model, describing the evolution of x i ( t ) : [0 , ∞ ) → S d -1 :

<!-- formula-not-decoded -->

Here and throughout, P x y := y -⟨ x, y ⟩ x denotes the orthogonal projection of y onto the tangent space T x S d -1 , ⟨· , ·⟩ is the Euclidean inner product in R d , and Z β,i ( t ) = ∑ N j =1 e β ⟨ Q t x i ( t ) ,K t x j ( t ) ⟩ is the time-dependent normalization factor. The parameter β &gt; 0 is interpreted as the inverse temperature .

Remark 2.1. The MLP would act as a drift term in the dynamics, whose consequence should still be investigated further. We expect different dynamical behavior depending on the relative scale of the MLP coefficients and the attention part. Although the framework allows for the inclusion of feedforward layers via a Lie-Trotter splitting scheme (see [26]), we choose to isolate exclusively the self-attention mechanism, both because our interest lies specifically in its dynamics, and for the sake of clarity. For the same reason, we assume, as in the works cited above, the parameter matrices to be shared across layers: Q t ≡ Q , K t ≡ K , and V t ≡ V .

As the positional information of each token is encoded in its initial condition, the dynamics (SA) is invariant under permutations of the particles' indices. This symmetry allows us to fully characterize the system's state through the particles' empirical measure µ ( t ) := 1 N ∑ N i =1 δ x i ( t ) , where δ x denotes the Dirac measure centered at x . The measure µ ( t ) evolves according to the continuity equation:

<!-- formula-not-decoded -->

where the vector field χ β [ µ ] : S d -1 → T S d -1 is defined as

<!-- formula-not-decoded -->

with Z β,µ ( x ) := ∫ S d -1 e β ⟨ Qx,Ky ⟩ d µ ( y ) . This formulation extends the token dynamics to a flow on the space P ( S d -1 ) of probability measures over the sphere S d -1 , encompassing both empirical and absolutely continuous distributions.

## 3 Main results

As discussed in the introduction, in this paper we consider the limit as β →∞ of the dynamics (3). To present the dynamical scales arising in this limit, consider a formal Taylor expansion of the vector field χ β [ µ ] generated by a sufficiently smooth measure µ :

<!-- formula-not-decoded -->

where σ denotes the Lebesgue measure on S d -1 and x ′ = K ⊤ Qx . For large β , Laplace approximation suggests that (I) typically dominates (II) at initialization, giving rise to a first, fast dynamical phase:

- Alignment Phase : on a timescale of O (1) , the limiting dynamics are governed by a linear transport equation (Eq. (5) below) and the token distribution rapidly collapses onto a lowerdimensional subspace determined by the spectral properties of the matrix V K T Q .

After the dynamics collapses to this low-dimensional subspace, we identify some classes of parameters for which the leading-order contribution to the vector field, approximated by term (I), vanishes. In such scenarios, the dynamics becomes governed by term (II), which involves the gradient of the measure µ . This gives rise to a second, intermediate phase:

- Heat Phase : operating on a timescale of O ( β ) (achieved by rescaling time as t ′ = t/β ), the dynamics within the previously identified subspace exhibits diffusive or anti-diffusive behavior. Depending on the model parameters (specifically the sign related to V K T Q restricted to the subspace), this phase can lead to further concentration into distinct clusters (backward heat equation) or to smoothing/spreading of the distribution (forward heat equation).

In the attractive case, we identify the limiting dynamics up until the formation of clusters. We expect the clusters to be invariant in this timescale, and to interact only on much longer ones.

- Pairing Phase : on an exponentially long timescale in β (e.g., O ( e cβ ) for some c &gt; 0 , where c depends on the distance between clusters), the clusters formed in the previous phase sequentially merge. Typically, the closest pair of clusters collapses first, governed by a system of ODEs describing their interaction, eventually leading to a single clustered state.

We refer to Appendix E for a graphical representation of the three phases introduced above.

We outline the structure of the remainder of this Section. In in Section 3.1, we recall a quantitative result connecting the large N behavior of the ODE system with the behavior of the corresponding PDE in the relevant timescale. This will allow us to focus solely on the PDE analysis when we describe the three main dynamical phases in Sections 3.2, 3.3, and 3.4.

## 3.1 Large N convergence

To connect the timescales analysis above to the N -particle system of ODEs (SA), we consider the regime where N → ∞ and β = β N → ∞ slowly enough with respect to N . This is relevant for context scaling techniques and LLMs (see introduction). To proceed, we use the following lemma:

Lemma 3.1. Assume that the initial tokens { x i (0) } i ∈ [ N ] are sampled independently and identically distributed from a reference measure µ 0 ∈ P ( S d -1 ) . Let µ N,β t be the empirical measure for particles { x i ( t ) } i ∈ [ N ] evolving via the ODEs (SA) , and let µ β t be the solution to the continuity equation (2) with initial condition µ 0 . Fix a time interval [0 , T β ] where T β is a β -dependent timescale. If β = β N depends on N and diverges slowly enough as N →∞ , then:

<!-- formula-not-decoded -->

uniformly on [0 , T β ] .

Proof. This follows from the Dobrushin-type stability estimate: W 1 ( µ N,β t , µ β t ) ≤ W 1 ( µ N 0 , µ 0 ) e L β t , where L β is a positive constant depending on the Lipschitz constant of the vector field χ β , as discussed in [9]. The claimed convergence follows from W 1 ( µ N 0 , µ 0 ) → 0 provided that L β N T β N grows sufficiently slowly with N such that the overall term tends to zero.

Our goal is to understand the behavior of µ N,β t in the joint limit N,β N →∞ . We denote the limiting distribution of µ β t as β →∞ by µ ∞ t . Following an argument analogous to that in [22], though in a different setting, we can decompose the convergence problem as:

<!-- formula-not-decoded -->

In our regime, Lemma 3.1 guarantees that the first term vanishes as N → ∞ . Consequently, the analysis of the N -particle system in this coupled limit reduces to studying the behavior of the solution µ β t to the continuity equation (2) as β → ∞ . The PDE analysis in this limit will therefore be the focus of the following sections.

## 3.2 The Alignment Phase

To characterize the limiting dynamics in the first phase we make the following assumptions:

Assumption 1. Q,K,V are invertible square matrices.

Assumption 2. The probability measure µ 0 on S d -1 is absolutely continuous with respect to the Lebesgue measure on S d -1 . Its density is bounded from above and below ( min x ∈ S d -1 µ 0 ( x ) &gt; 0 ) and Lipschitz continuous.

These technical assumptions, significantly milder than the ones made in most related works, are needed to guarantee that the terms appearing in the analysis of the limiting equation, e.g., the denominator in (5), are sufficiently well behaved. Under these conditions, we show that the limiting dynamics in this regime coincides with the formal Laplace approximation of term (I) in (4), i.e., the integrals in the definition of the vector fields can be replaced by the value of the integrand at the maximum point x ′ = K T Qx/ | K T Qx | , leading to the significantly simplified expression (5) below.

Theorem 3.2. Let Assumptions 1, 2 hold, then the solutions { µ β } β of the continuity equation (2) converge in C ([0 , T ] , P ( S d -1 )) to the solution µ ∞ of the partial differential equation:

<!-- formula-not-decoded -->

Proof Sketch. To establish the result we must prove well-posedness of the family of equations leading to the desired limit and obtain sufficient regularity uniformly in β to ensure that the formal simplifications from (4)(I) to (5) are allowed. This is particularly important as the derivatives of the kernel tend to infinity in the limit β → ∞ . The core argument proceeds in three main steps. First, we establish the relative compactness of the family of trajectories { µ β } β&gt; 0 in the space C ([0 , T ] , P ( S d -1 )) using a variant of Ascoli-Arzelà theorem and the boundedness of the vector field χ β [ µ β ] . The second, crucial step involves deriving uniform in β estimates on the regularity (i.e., Lipschitz bounds) of the vector field χ β [ µ β ] along the solution trajectories µ β . This is achieved by analyzing the concentration behavior of the kernel e β ⟨ Qx,Ky ⟩ as β →∞ , leveraging properties related to the cumulants of the V on Mises-Fisher distribution, and employing a continuation argument to propagate regularity over time. Finally, using the compactness and uniform regularity, we pass to the limit β →∞ in the weak formulation of the continuity equation (2). The uniform estimates allow us to conclude that µ ∞ is a solution of (5), while uniqueness follows from [5]. The full proof is deferred to Appendix A.1.

A consequence of Theorem 3.2 is that in the largeβ limit, the tokens, to leading order, evolve independently of each other, driven primarily by the structure of the Q , K , and V matrices. In this regime, self-attention behaves like a composition of linear layers followed by layer normalization, with minimal influence from inter-token interactions.

Combining the above result with Lemma 3.1 we obtain the following convergence result:

Corollary 3.3. Under Assumptions 1, 2, for every t &gt; 0 we have W 1 ( µ N,β t , µ ∞ t ) → 0 as N →∞ , provided that β N →∞ slowly enough.

Having established that µ ∞ is a solution of equation (5), we can investigate its long-time behavior. In particular, we show below that the support of µ ∞ t is asymptotically flattened onto a lower-dimensional subspace determined by the spectral properties of the matrix V K ⊤ Q ,

Proposition 3.4. Let µ 0 be a probability measure on S d -1 absolutely continuous with respect to the Lebesegue measure, and let µ t be the corresponding solution of (5) . Then for every ν ∈ ω ( µ 0 ) (the ω -limit set of µ 0 ) it holds:

<!-- formula-not-decoded -->

where E max is the generalized eigenspace associated to the eigenvalue of V K T Q with largest real part.

Proof. The proof is provided in Appendix A.2, where we reduce the analysis to a linear system of ODEs in R d with matrix V K T Q , identifying the corresponding asymptotics with the ones of (5).

Remark 3.5. At a first glance, this result might appear inconsistent with those of [11], since in some cases measures supported on E max do not maximize the energy. However, this apparent discrepancy is a consequence of the order of the limits being taken, with β →∞ preceding t →∞ in our case.

Proposition 3.4 demonstrates that the token representations rapidly collapse onto a lower-dimensional subspace determined by the model's matrices. This can be interpreted as the initial phase of the inference process, where information is compressed into a smaller, more relevant subspace. This phenomenon is consistent with the rank collapse observed, e.g., in [38, 27].

Remark 3.6. Apart from the collapse to E max , one cannot in general conclude the existence of a limiting (stationary) dynamics for (5) . Indeed, it is not difficult to construct examples where the particles continue to rotate on the sphere indefinitely, e.g., when V is a rotation and Q T K = Id .

Remark 3.7. Recent works have studied transformer models with stochastic perturbations [48], where the token dynamics is influenced by random noise. In this setting, the convergence to the corresponding equation (5) (with an additional Laplacian term) is typically easier to establish due to the regularizing effect of the noise (see [39]).

## 3.3 The Heat Phase

Having established the rapid collapse onto the subspace E max ∩ S d -1 , we now investigate the slower evolution within this subspace, assuming that the initial measure µ 0 is supported in E max ∩ S d -1 as a consequence of the previous analysis:

Assumption 3. The initial condition µ 0 in the heat phase satisfies supp ( µ 0 ) ⊆ E max ∩ S d -1 .

Since the intersection E max ∩ S d -1 can be identified with a lower-dimensional sphere, specifically S dim( E max ) -1 , we will, with a slight abuse of notation, continue to denote it by S d -1 .

To demonstrate that the heat equation, described in a different setting in [46], emerges as an intermediate dynamical phase due to the spherical geometry induced by LayerNorm, we assume:

Assumption 4. Q T K | E max = λ 1 I and V | E max = ± λ 2 I when restricted to E max , with λ 1 , λ 2 &gt; 0 .

Under this condition, E max is an invariant subspace for Eq. (2) and, without loss of generality, we can suppose λ 1 , λ 2 = 1 .

Remark 3.8. Assumption 4, for example, is satisfied under the global assumption Q ⊤ K = S and V = ± S , with S symmetric definite positive matrix. This is a fairly standard assumption in recent studies within this framework and it endows the model with an additional structure of gradient flow on P ( S d -1 ) with respect to a modified metric (see [11, 25]).

In this regime, the vector field χ β [ µ ] vanishes on the support of µ as β →∞ , but its rescaled version admits the formal limit (see Corollary B.2):

<!-- formula-not-decoded -->

where γ := ± 1 , depending on the sign choice in the definition of V . This scaling of the vector field by β corresponds to a time rescaling dt = βds , explaining the phase duration of order O ( β ) .

Proposition 3.9. Let Assumption 4 hold and let µ ∞ 0 ∈ P ( S d -1 ) be the initial measure. Assume that there exist T &gt; 0 , k positive integer, and µ ∞ t ∈ C ([0 , T ] , C k +3 ( S d -1 )) , with min x ∈ S d -1 µ ∞ t ( x ) &gt; 0 for all t ∈ [0 , T ] , such that µ ∞ t solves the heat equation on [0 , T ] × S d -1 :

<!-- formula-not-decoded -->

where ∆ denotes the Laplace-Beltrami operator on S d -1 . Then, for large β , µ ∞ t solves the mean-field PDE:

<!-- formula-not-decoded -->

where the residual term satisfies R β → 0 in C ([0 , T ] , C k ( S d -1 )) as β →∞ .

This proposition, whose proof is provided in Appendix B, characterizes the limiting dynamics within the lower-dimensional manifold, connecting the transformer model with a heat flow on the sphere, thereby justifying the name of this phase. Remarkably, this connection holds without the need for correction terms, in contrast to [46]. We now need to distinguish between two different cases:

- Forward diffusion . When γ &lt; 0 in equation (6), the dynamics corresponds to a forward heat equation. In this setting, local existence and regularity for µ ∞ t (and in particular the assumptions of Prop. 3.9) are automatically satisfied due to the smoothing properties of forward diffusion, provided that µ 0 ∈ C k +3 ( S d -1 ) . Notably, interacting particle systems of the specific form given by equation (SA) (under the assumption Q ⊤ K = Id = -V ) have been studied in the literature and are known as diffusion-velocity methods ; see, for instance, [33, 20, 8, 31, 32, 35].
- Backward diffusion . When γ &gt; 0 , the dynamics corresponds to a backward heat equation. In this case, the regularity assumptions on µ ∞ 0 ensuring local existence and regularity are significantly more restrictive (e.g. requiring that µ 0 is in the Gevrey1 2 space). Nonetheless, we construct explicit examples of solutions below. The backward heat equation is a prototypical ill-posed problem, which explains why the statement of Proposition 3.9 is necessarily weaker than that of Theorem 3.2.

A family of initial conditions µ 0 that satisfies the assumptions of Proposition 3.9 is given by

<!-- formula-not-decoded -->

where α j ≥ 0 , ∑ M j =1 α j = 1 , and N S d -1 ( m,σ 2 ) denotes the heat kernel (the forward-in-time evolution under the heat semi-group exp( t ∆) of a Dirac delta, analogous to a Gaussian N R d ( m,σ 2 ) in Euclidean space) centered at m ∈ S d -1 with concentration related to σ 2 . By linearity of ∆ , the explicit solution to ∂ t µ = -γ ∆ µ is then given by:

<!-- formula-not-decoded -->

For forward diffusion ( γ &lt; 0 ), this solution is a smooth function for all t ≥ 0 . while in the backward case ( γ &gt; 0 ) this only holds for t ∈ [0 , T min ) , where T min = min j σ 2 j is the time at which the first Gaussian component collapses to a Dirac delta δ m j . Amore general class in which local existence and well-posedness hold in both the forward and backward directions is the set of positive, Gevrey1 / 2 , functions.

Motivated by the aggregation behavior observed in the finiteβ particle system, we conjecture that the collapsed δ m j remains invariant under the limiting dynamics, while other components continue evolving independently according to the backward heat equation until their respective collapse times. From the practical perspective, this observation suggests that the transformer's behavior in this regime can be interpreted as a form of regularized denoising (when β is finite) acting on the input. This aligns with the clustering phenomena extensively studied in previous works on the model. The dynamics in this phase, governed by a heat equation, drive the formation of distinct token clusters (via backward diffusion) or the smoothing of the token distribution (via forward diffusion). This behavior can be interpreted as a representation refinement stage, where tokens are organized into more defined semantic groups.

Remark 3.10. In [25], a simplified model, referred to as the Unnormalized Self-Attention (USA) model, is proposed, where the normalization factor Z β,µ ( x ) is replaced by a constant Z β , significantly simplifying the mathematical analysis. By choosing Z β = 1 β ∫ S d -1 e β ⟨ x,y ⟩ dσ ( y ) (or equivalently by rescaling time), the limiting behavior of the model no longer yields the heat equation, but rather the porous medium equation: ∂ t µ = ∆( µ 2 ) . Even in this case, the convergence of particles system to this nonlinear PDE has been extensively studied (see for example [22, 33, 39, 41] or [47] for S d -1 ).

## 3.4 Pairing Phase

The initial conditions we consider for the dynamics on longer timescales must be compatible with the steady states of the preceding phase. Motivated by the discussion at the end of the previous section, we therefore formulate the following assumption:

Assumption 5. The initial condition µ 0 in the pairing phase can be written as µ 0 = ∑ m j =1 α j δ x j for an m ∈ N , with x j ∈ S d -1 , α j &gt; 0 ∀ j ∈ { 1 , ..., m } and ∑ m j =1 α j = 1 .

̸

Under this assumption, further supposing for the sake of clarity that α j = 1 /m for all j ∈ { 1 , . . . , m } , we can interpret each cluster as a particle, and the dynamics of the system is given by the set of ODEs (SA). In the regime of large β , clusters interact very weakly due to their separation and the exponential tails (in β ) of the interaction kernel, resulting in exponentially long timescales for the nontrivial dynamics. Here, analogously to [3], interactions are dominated by the closest pair of clusters ( i, j ) , assumed unique, satisfying ⟨ x i , x j ⟩ = max i = j ⟨ x i , x j ⟩ at initialization. We note that this hardmax particle interaction, as well as the timescale where it arises in the large β limit, was introduced in [23, Section 6] in the case d = 2 . We present an analogous result here in arbitrary dimension, without claiming originality, to provide a complete dynamical picture across phases.

Proposition 3.11. The solutions x i ( t ) of the ODE system (SA) , under Assumptions 4 and positive V , with the rescaled time dt = e β (1 -⟨ x i ,x j ⟩ ) ds , converge as β →∞ to the solutions of the system:

<!-- formula-not-decoded -->

on finite intervals [0 , T ϵ ] , with T ϵ such that ⟨ y i , y j ⟩ ≤ 1 -ϵ throughout the interval, for any ϵ &gt; 0 .

In other words, all clusters remain stationary except for the closest pair, which collapses along the geodesic connecting them, in a time exponential in β . Note that this result only holds up to an arbitrary moment before the first collapse. We refer to [23, Section 6] for a detailed explanation of the challenges to bypass this limit and a proof of an analogous result until and beyond the collapse time in a related but simplified model. The above proposition is proven for completeness in Appendix C.

This final, slow phase models the sequential merging of the closest token clusters. This can be interpreted as the construction of higher-order abstractions, where previously formed groups are hierarchically combined to create more complex representations.

## 4 Numerical experiments

This section presents numerical simulations of the transformer model in Eq. (1). All experiments are conducted in dimension d = 3 or or d = 2 to facilitate visualization and are designed to validate our theoretical findings. The attention mechanism is implemented using the official PyTorch function torch.nn.functional.scaled\_dot\_product\_attention() and the experiments are performed on a single Nvidia H100. The code is available at [28].

First Phase Dynamics. Figure 1 illustrates the dynamics of the alignment phase, showing distinct behaviors based on the parameters choices for Q,K,V . For both scenarios presented in Figure 1, the initial state consists of N = 10 4 tokens sampled independently and identically uniformly from the sphere S 2 . We set the inverse temperature parameter β = 30 and use a time step of dt = 10 -2 .

- Scenario 1a (collapse to 1D subspace). The matrix V K T Q is chosen such that it possesses a unique eigenvalue with maximal real part. As predicted by our theory, this configuration leads to the tokens collapsing onto a one-dimensional subspace (i.e. two antipodal points).
- Scenario 1b (non-gradient flow dynamics and rotation). This example employs a parameter choice for Q,K,V that falls outside the gradient flow regime. Nevertheless, the tokens are observed to collapse toward a two-dimensional subspace (a great circle), accompanied by a collective rotation of the particles along this circle.

Both observed behaviors are consistent with the results in Theorem 3.2 and Proposition 3.4.

Figure 1: Simulations of two different scenarios (one per row, four timesteps) for the first phase.

<!-- image -->

Second Phase Dynamics. We support the conclusions of section 3.3 through two examples, comparing empirical dynamics with analytical solutions of backward and forward diffusion equations.

- Scenario 2a (collapse to 2D subspace and backward diffusion). For the experiment in Figure 2 we set β = 10 . The initial configuration comprises N = 10 4 i.i.d. tokens. Their elevation angle ψ is sampled uniformly on [ -π 2 , π 2 ] , while their azimuthal angles, θ i ∈ [0 , 2 π ) , are distributed according to the mixture density g ( θ ) :

<!-- formula-not-decoded -->

where N ( · ; µ, σ 0 ) denotes the probability density function of a wrapped normal distribution on S 1 with mean µ and standard deviation σ 0 = 0 . 2 . The parameters Q , K , and V are chosen so that, after the first phase, the tokens collapse onto the xy -plane, with distribution g ( θ ) . The analytical solution to the backward heat equation with initial condition g ( θ ) (computed as in Eq. (7)) is plotted as a red curve in Figure 2. The positions of the clusters agree with this solution, numerically confirming the predictions of Proposition 3.9.

- Scenario 2b (forward diffusion comparison). In Figure 3, we compare the empirical token distribution with the analytical solution of the forward heat equation characterizing a possible example of the second phase of the dynamics. Specifically, we simulate the evolution of 5 × 10 4 tokens, initially sampled from a superposition of three Gaussian densities on S 1 , through the transformer model with parameters β = 50 , d = 2 , Q = K = Id , V = -Id , and dt = 10 -3 . The analytical solution of the forward heat equation (in red) closely matches the token distribution histogram (in blue) over time (i.e., depth). Note that, as expected, the forward diffusion process is significantly more stable numerically than the backward one.

Figure 2: Numerical simulation of the backward scenario for the second phase on S 2 .

<!-- image -->

Figure 3: Evolution of the tokens distribution in the forward scenario for the second phase in S 1 .

<!-- image -->

## 5 Conclusions

This work provides a mathematical analysis of token dynamics in mean-field transformer models within the moderate interaction regime, where interaction strength ( β ) scales with context size ( N ), motivated by scaling practices in modern LLMs. Our study reveals a fundamental multiscale structure governing the evolution of token representations through network depth in this setting. More specifically, we showed that, under this scaling, the system progresses through a sequence of three different dynamical phases characterized by qualitatively distinct dynamical behavior, operating on separated timescales. Through our analysis, we offer a unified dynamical picture describing how deep transformers might achieve progressive representation refinement.

This unified dynamical picture is, however, not yet fully rigorous. Indeed, while we establish convergence results for the alignment phase and provide partial justification for the intermediate and late phases under specific assumptions, a complete mathematical treatment of the full dynamics - particularly of the backward heat regime beyond the first collapse and of the slow clustering interactions - remains an open challenge due to significant technical difficulties in the analysis of the strongly unstable limiting equations. Additionally, characterizing phase transitions in self-attention for different N,β scalings is an interesting separate question, with progress made in [15] under assumptions on inter-token angles.

Furthermore, while the first dynamical phase has quite general assumptions on the parameter matrices, the following phases still require relatively limited assumptions (although less limited than in most previous works). Relaxing these assumptions further, in particular in the case of non-gradient dynamics, would constitute an interesting, but also technically quite challenging, avenue of future research.

There are several important directions in which our work could be extended. Most notably, incorporating the MLP, which could be interpreted as introducing a drift term in the dynamics, acting independently on each token without accounting for mutual interactions. Another natural extension involves studying the dynamics under more general parameter settings. For instance, during the heat phase, we assume that Q T K = S is symmetric positive definite, which holds, for example, when Q = K . This 'shared-QK' assumption is not novel and has been adopted in prior empirical work (e.g., in [30]). While different choices of these parameters (both MLP and Attention matrices) can have a dramatic effect on model behavior, with adversarial choices potentially leading to qualitatively different dynamics from the one predicted in this paper, we believe our results to be a relevant first step towards understanding the development of representations in transformers, capturing some important qualitative features of these models as shown in [25].

A further direction of future research consists in providing sufficient conditions for the stability of the space E max emerging in the alignment phase under the prelimit model (i.e., for large but finite β ), thereby justifying Assumption 3 and, ultimately, connecting in a rigorous way the alignment and heat phases identified in this paper.

While the path from this theoretical analysis to direct application is not immediate, we believe our work opens several potential avenues for future investigation. The characterization of the alignment phase, for instance, offers a potential mechanism for interpreting how token representations evolve into learned subspaces. Finally, by focusing on the "moderate interaction regime", we hope our analysis provides a theoretical foundation for a more principled understanding of parameter scaling, particularly as models are adapted for longer contexts.

## Acknowledgements

AA and GB acknowledge partial support by the Institute of Mathematical Statistics and Actuarial Sciences at the University of Bern. FP acknowledges support by the American Mathematical Society through the Bergman fellowship. AA thanks B. Geshkovski, J. Mattingly, Y. Polyanskiy, and P. Rigollet for interesting discussions. All authors thank the anonymous reviewers for comments that have improved the quality of the paper. Calculations were performed on UBELIX, the HPC cluster at the University of Bern.

## References

- [1] Milton Abramowitz and Irene A Stegun. Handbook of mathematical functions with formulas, graphs, and mathematical tables , volume 55. US Government printing office, 1948.
- [2] Andrea Agazzi and Jianfeng Lu. Global optimality of softmax policy gradient with single hidden layer neural networks in the mean-field regime. In International Conference on Learning Representations (ICLR 2021) , 2021.
- [3] Albert Alcalde, Giovanni Fantuzzi, and Enrique Zuazua. Clustering in pure-attention hardmax transformers and its role in sentiment analysis. SIAM Journal on Mathematics of Data Science , 7(3):1367-1393, 2025.
- [4] Albert Alcalde, Borjan Geshkovski, and Domènec Ruiz-Balet. Attention's forward pass and frank-wolfe. arXiv preprint arXiv:2508.09628 , 2025.
- [5] Luigi Ambrosio, Luis Caffarelli, Michael G Crandall, Lawrence C Evans, and Nicola Fusco. Transport equation and cauchy problem for non-smooth vector fields. Calculus of Variations and Nonlinear Partial Differential Equations: With a historical overview by Elvira Mascolo , pages 1-41, 2008.
- [6] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473 , 2014.
- [7] Raphaël Berthier, Andrea Montanari, and Kangjie Zhou. Learning time-scales in two-layers neural networks. Foundations of Computational Mathematics , pages 1-84, 2024.
- [8] Yann Brenier. Geometric diffusions of 1-currents. In Annales de la Faculté des sciences de Toulouse: Mathématiques , volume 26, pages 831-846, 2017.
- [9] Giuseppe Bruno, Federico Pasqualotto, and Andrea Agazzi. Emergence of meta-stable clustering in mean-field transformer models. In International Conference on Learning Representations (ICLR 2025) , 2025.
- [10] Martin Burger and Antonio Esposito. Porous medium equation and cross-diffusion systems as limit of nonlocal interaction. Nonlinear Analysis , 235:113347, 2023.
- [11] Martin Burger, Samira Kabri, Yury Korolev, Tim Roith, and Lukas Weigand. Analysis of mean-field models arising from self-attention dynamics in transformer architectures with layer normalization. Philosophical Transactions A , 383(2298):20240233, 2025.
- [12] José Antonio Carrillo, Katy Craig, and Francesco S Patacchini. A blob method for diffusion. Calculus of Variations and Partial Differential Equations , 58:1-53, 2019.
- [13] Valérie Castin, Pierre Ablin, José Antonio Carrillo, and Gabriel Peyré. A unified perspective on the dynamics of deep transformers. arXiv preprint arXiv:2501.18322 , 2025.
- [14] Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. Advances in neural information processing systems , 31, 2018.
- [15] Shi Chen, Zhengjiang Lin, Yury Polyanskiy, and Philippe Rigollet. Critical attention scaling in long-context transformers. arXiv preprint arXiv:2510.05554 , 2025.
- [16] Shi Chen, Zhengjiang Lin, Yury Polyanskiy, and Philippe Rigollet. Quantitative clustering in mean-field transformer models. arXiv preprint arXiv:2504.14697 , 2025.

- [17] Lenaic Chizat and Francis Bach. On the global convergence of gradient descent for overparameterized models using optimal transport. Advances in neural information processing systems , 31, 2018.
- [18] Christopher Criscitiello, Quentin Rebjock, Andrew D McRae, and Nicolas Boumal. Synchronization on circles and spheres with nonlinear interactions. arXiv preprint arXiv:2405.18273 , 2024.
- [19] Valentin De Bortoli, Alain Durmus, Xavier Fontaine, and Umut Simsekli. Quantitative propagation of chaos for sgd in wide neural networks. Advances in Neural Information Processing Systems , 33:278-288, 2020.
- [20] Pierre Degond and Francisco-José Mustieles. A deterministic approximation of diffusion equations using particles. SIAM Journal on Scientific and Statistical Computing , 11(2):293-310, 1990.
- [21] Weinan E. A proposal on machine learning via dynamical systems. Communications in Mathematics and Statistics , 1(5):1-11, 2017.
- [22] Alessio Figalli and Robert Philipowski. Convergence to the viscous porous medium equation and propagation of chaos. ALEA Lat. Am. J. Probab. Math. Stat , 4:185-203, 2008.
- [23] Borjan Geshkovski, Hugo Koubbi, Yury Polyanskiy, and Philippe Rigollet. Dynamic metastability in the self-attention model. arXiv preprint arXiv:2410.06833 , 2024.
- [24] Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, and Philippe Rigollet. The emergence of clusters in self-attention dynamics. Advances in Neural Information Processing Systems , 36, 2024.
- [25] Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, and Philippe Rigollet. A mathematical perspective on transformers. Bulletin of the American Mathematical Society , 62(3):427-479, 2025.
- [26] Borjan Geshkovski, Philippe Rigollet, and Domènec Ruiz-Balet. Measure-to-measure interpolation using transformers. arXiv preprint arXiv:2411.04551 , 2024.
- [27] Alessio Giorlandino and Sebastian Goldt. Two failure modes of deep transformers and how to avoid them: a unified theory of signal propagation at initialisation. arXiv preprint arXiv:2505.24333 , 2025.
- [28] GitHub-Repository. https://github.com/gbruno16/multiscale\_transformers .
- [29] Nikita Karagodin, Yury Polyanskiy, and Philippe Rigollet. Clustering in causal attention masking. Advances in Neural Information Processing Systems , 37:115652-115681, 2024.
- [30] Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020 . OpenReview.net, 2020.
- [31] Gilles Lacombe. Analyse d'une équation de vitesse de diffusion. Comptes Rendus de l'Académie des Sciences-Series I-Mathematics , 329(5):383-386, 1999.
- [32] Gilles Lacombe and Sylvie Mas-Gallic. Presentation and analysis of a diffusion-velocity method. In Esaim: Proceedings , volume 7, pages 225-233. EDP Sciences, 1999.
- [33] Pierre-Louis Lions and Sylvie Mas-Gallic. Une méthode particulaire déterministe pour des équations diffusives non linéaires. Comptes Rendus de l'Académie des Sciences-Series IMathematics , 332(4):369-376, 2001.
- [34] Johan Markdahl, Johan Thunberg, and Jorge Gonçalves. Almost global consensus on the n -sphere. IEEE Transactions on Automatic Control , 63(6):1664-1675, 2017.
- [35] Sylvie Mas-Gallic. The diffusion velocity method: a deterministic way of moving the nodes for solving diffusion equations. Transport Theory and Statistical Physics , 31(4-6):595-605, 2002.

- [36] Song Mei, Andrea Montanari, and Phan-Minh Nguyen. A mean field view of the landscape of two-layer neural networks. Proceedings of the National Academy of Sciences , 115(33):E7665E7671, 2018.
- [37] Ken M Nakanishi. Scalable-softmax is superior for attention. arXiv preprint arXiv:2501.19399 , 2025.
- [38] Lorenzo Noci, Sotiris Anagnostidis, Luca Biggio, Antonio Orvieto, Sidak Pal Singh, and Aurelien Lucchi. Signal propagation in transformers: Theoretical perspectives and the role of rank collapse. Advances in Neural Information Processing Systems , 35:27198-27211, 2022.
- [39] Karl Oelschläger. A law of large numbers for moderately interacting diffusion processes. Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete , 69(2):279-322, 1985.
- [40] Karl Oelschläger. Large systems of interacting particles and the porous medium equation. Journal of differential equations , 88(2):294-346, 1990.
- [41] Karl Oelschläger. A sequence of integro-differential equations approximating a viscous porous medium equation. Zeitschrift für Analysis und ihre Anwendungen , 20(1):55-91, 2001.
- [42] Thierry Paul and Emmanuel Trélat. Universal approximations of quasilinear pdes by finite distinguishable particle systems. arXiv preprint arXiv:2501.11387 , 2025.
- [43] Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. Yarn: Efficient context window extension of large language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024 . OpenReview.net, 2024.
- [44] Yury Polyanskiy, Philippe Rigollet, and Andrew Yao. Synchronization of mean-field models on the circle. arXiv preprint arXiv:2507.22857 , 2025.
- [45] Grant Rotskoff and Eric Vanden-Eijnden. Trainability and accuracy of artificial neural networks: An interacting particle system approach. Communications on Pure and Applied Mathematics , 75(9):1889-1935, 2022.
- [46] Michael E Sander, Pierre Ablin, Mathieu Blondel, and Gabriel Peyré. Sinkformers: Transformers with doubly stochastic attention. In International Conference on Artificial Intelligence and Statistics , pages 3515-3530. PMLR, 2022.
- [47] Anna Shalova. Noisy gradient flows: with applications in machine learning. PhD Thesis , 2025.
- [48] Anna Shalova and André Schlichting. Solutions of stationary mckean-vlasov equation on a high-dimensional sphere and other riemannian manifolds. arXiv preprint arXiv:2412.14813 , 2024.
- [49] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems , 30, 2017.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Abstract and introduction summarize the claims made in the paper that are proved or discussed in the following sections.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Yes, see section conclusions.

Guidelines:

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

## Answer: [Yes]

Justification: The assumptions are reported in the corresponding sections, while the proofs are in the supplementary materials

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

Justification: The details of the experiments are reported in the corresponding section.

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

Justification: An anonymous github repository is provided.

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

Answer: [NA]

Justification: The experiments don't have training and test.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The kind of experiments does not need statistical significance, they are solution of ODEs.

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

Justification: See the section about numerical experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Anonymity and all the other rules have been respected.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: It is a theoretical paper.

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

Justification: It is a theoretical paper.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: It is a theoretical paper.

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

Answer: [Yes]

Justification: See numerical experiments section and the github link.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Proofs of the alignment phase

This section is divided into two parts. The first part contains the proof of Theorem 3.2, while the second one contains the proof of Proposition 3.4.

## A.1 Moderate scaling limit

Consider the family { µ β } β ≥ 0 of solutions to the usual continuity equation (2):

<!-- formula-not-decoded -->

where χ β is the vector field given by:

<!-- formula-not-decoded -->

Remark A.1. For notational simplicity, we will refer to the matrix Q t K in the main body as B .

Remark A.2. In the following C will be a constant depending only on V, B, d and µ 0 . Its value may change line by line.

Under assumptions 1, 2, i.e.:

- A1: Q,B are invertible square matrices.
- A2: The probability measure µ 0 on S d -1 is absolutely continuous with respect to the Lebesgue measure. Its density is bounded and Lipschitz continuous and its minimum satisfies min x ∈ S d -1 µ 0 ( x ) &gt; 0 .

one can prove Theorem 3.2: µ β converges weakly to µ ∞ in C ([0 , T ]; P ( S d -1 )) where µ ∞ is the unique solution of the PDE:

<!-- formula-not-decoded -->

and the metric in C ([0 , T ] , P ( S d -1 )) is given by:

<!-- formula-not-decoded -->

where BL ( S d -1 ) is the set of all the Lipschitz continuous functions on S d -1 which are bounded together with their Lipschitz constant by 1 .

The idea of the proof follows five steps:

- Relative compactness in C ([0 , T ]; P ( S d -1 )) ,
- Bounds on cumulants of the Von Mises-Fisher distribution,
- Prove a relationship between the derivatives of µ and the regularity of the vector field,
- Apply a continuation argument to show the uniform regularity of χ β [ µ β ] ,
- Use the regularity to pass to the limit in the PDE.

## A.1.1 Relative compactness

Proposition A.3. The set { µ β } β ≥ 0 is relatively compact in C ([0 , T ] , P ( S d -1 )) .

Proof. By Prokhorov's theorem, since S d -1 is compact, we can conclude that P ( S d -1 ) is weakly compact. Since ρ metricizes the weak topology, then also ( P ( S d -1 ) , ρ ) is compact. To apply Ascoli-Arzelà theorem, we just need the equicontinuity of the set { µ β } β&gt; 0 .

Given 0 ≤ s ≤ t ≤ T and β &gt; 0 :

<!-- formula-not-decoded -->

This is sufficient to conclude the proof.

## A.1.2 Bounds on the vector field

The aim of the following paragraphs is to obtain some bounds on D i χ β [ µ ] , i = 0 , 1 , 2 . To fix the notation we define the probability measure ν µ,β,B x on S d -1 as:

<!-- formula-not-decoded -->

Remark A.4. The measure ν σ,β,B x is the Von Mises-Fisher distribution with mean direction B T x | B T x | and concentration parameter β | B T x | . Some properties of this distribution are studied later.

Then the vector field χ β [ µ ] can be written as:

<!-- formula-not-decoded -->

µ,β,B

where ◦ denotes the composition with respect to the parameter x of the measure ν x . Lemma A.5. The derivatives of the vector field χ β [ µ ] are bounded by:

| χ β [ µ ]( x ) | ≤ C,

<!-- formula-not-decoded -->

where C is a constant depending only on V, B, d .

Proof. Let's compute the derivatives of χ β [ µ ] :

<!-- formula-not-decoded -->

Hence we need to compute the derivatives with respect to x of E ν µ,β,B x [ Y ] = E ν µ, 1 ,Id x [ Y ] ◦ ( βB T x ) . Thanks to the Faa di Bruno formula:

<!-- formula-not-decoded -->

with Π n the set of all the possible partitions of { 1 , ..., n } and ⊗ the tensor product. The previous expression can be bounded by C B ∑ n l =0 β l ∥ ( D l x E ν µ, 1 ,Id x [ Y ]) | βB T x ∥ , where C B is a constant depending on the matrix B .

Thus, the aim is to compute a bound for D n x E ν µ, 1 ,Id x [ Y ] . This is related to the d -dimensional cumulants (tensors) of the distribution ν µ, 1 ,Id x . Indeed, we can write:

<!-- formula-not-decoded -->

It is well known that the first three cumulants correspond to the central moments:

<!-- formula-not-decoded -->

The thesis then follows by replacing these equalities in the initial bounds (after renaming C V and C B ).

Lemma A.6. Let σ be the uniform measure on S d -1 . Then:

<!-- formula-not-decoded -->

Proof. By (8) and Schwarz's theorem the three tensors are invariant by permutations of the indices and by definition of ν σ,β,Id x they are also invariant by rotations that fix x . Hence (see lemma D.2) they must have the form:

<!-- formula-not-decoded -->

Where Sym ( x ⊗ Id ) = x i δ jk + x j δ ik + x k δ ij . Weneed to compute the coefficients α 1 , α 2 , β 2 , α 3 , β 3 . Define A ( β ) = ∫ S d -1 ⟨ x, y ⟩ ν σ,β,Id x ( dy ) . Similarly to what has been done in (8), one can relate this to the cumulants of ⟨ x, Y ⟩ by noticing that:

<!-- formula-not-decoded -->

This give us immediately the following identities:

<!-- formula-not-decoded -->

Suppose without loss of generality that x = e 1 . Then α 1 is given by:

<!-- formula-not-decoded -->

The coefficients α 2 and β 2 can be obtained comparing the representations in (10) with the representations in (9), and exploiting the relations in (11):

<!-- formula-not-decoded -->

And the same can be done for α 3 and β 3 :

<!-- formula-not-decoded -->

To conclude it is sufficient to show that 1 -A 2 = O ( 1 β ) and A ′ , A ′′ = O ( 1 β 2 ) . Now, using the identity:

<!-- formula-not-decoded -->

we can explicitly compute A ( β ) as:

<!-- formula-not-decoded -->

where we used the derivatives rules for the modified Bessel function:

<!-- formula-not-decoded -->

and its asymptotic behavior (in both cases see [1]).

In a similar way we can also compute:

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where we used the asymptotics in (14). This is sufficient to conclude the proof.

Lemma A.7. Given a strictly positive probability measure µ the following holds:

<!-- formula-not-decoded -->

where x B = B T x | B T x |

Proof. Indeed:

<!-- formula-not-decoded -->

where we used:

<!-- formula-not-decoded -->

and the last inequality is a consequence of ν σ,β,B x = ν σ,β | B T x | ,Id x B and lemma D.1.

β satisfy:

Proposition A.8. The derivatives of the vector field χ [ µ ]

<!-- formula-not-decoded -->

Proof. Thanks to Lemma A.5 we just need to bound the cumulants. The first one is already done. Second cumulant:

<!-- formula-not-decoded -->

We have already shown in lemma A.6 that the first term is ≤ C 1 β . Now we need to bound the second term. R can be expanded by multi-linearity and using lemma A.7 the worst case is either of the form:

<!-- formula-not-decoded -->

where in the last line we used lemma D.1, or of the form:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

These are the worst cases because every | y -z | produces an additional β -1 / 2 by lemma D.1. Hence we proved, thanks to lemma A.5, that:

<!-- formula-not-decoded -->

The bound for the third cumulant is similar to what we have done above:

<!-- formula-not-decoded -->

We have already shown that the first term is O ( 1 β 2 ) . Now we need to bound the second term. R can be expanded again as in lemma A.7 and the worst case is either of the form:

<!-- formula-not-decoded -->

or of the form:

<!-- formula-not-decoded -->

Thus, we can replace the bounds on the second and third cumulants that we obtained above in the estimates of Lemma A.5 to conclude that:

<!-- formula-not-decoded -->

## Lemma A.9. If µ solves the PDE:

then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Let x t be a point of maximum for | µ t | . Then ∇ S d -1 µ t ( x t ) = 0 and:

<!-- formula-not-decoded -->

And for min µ we can use the same argument.

Now, let x t be a point of maximum for |∇ S d -1 µ | 2 , then H S d -1 µ ( x t ) ∇ S d -1 µ ( x t ) = 0 , hence:

<!-- formula-not-decoded -->

Using that ∂ t |∇ µ ( x t ) | 2 = 2 |∇ µ ( x t ) | ∂ t |∇ µ ( x t ) | and dividing by |∇ µ ( x t ) | we get the thesis.

Lemma A.10. Consider again µ t solution of the PDE:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, for β large enough (depending just on µ 0 and C ):

- ∥ µ t ∥ ∞ ≤ 2( ∥ µ 0 ∥ ∞ ) e 2 Ct ,
- min µ t ≥ 1 2 (min µ 0 ) e -2 Ct ,

<!-- formula-not-decoded -->

Proof. The thesis is true at time t = 0 . Let us assume that it is true on [0 , t ] . Then ∃ β big enough (where "big" depends only on C 1 , C 2 , i.e. just µ 0 , B, V, d ) such that ∥∇ µ ∥ ∞ min | µ | β -1 / 2 ≤ 1 on [0 , t ] . Hence Dχ [ µ β ] ≤ 2 C on [0 , t ] thanks to proposition A.8. By Gronwall applied to the first two bounds in lemma A.9 we can conclude:

<!-- formula-not-decoded -->

For ∥∇ µ ∥ ∞ we have, again by lemma A.9:

<!-- formula-not-decoded -->

Define:

where in the second row we used the assumption on [0 , t ] and proposition A.8. Hence by Gronwall:

<!-- formula-not-decoded -->

This concludes the continuation argument and the proof.

Corollary A.11. For β large enough (depending on µ 0 , B, V, d ) the vector fields { χ β [ µ ] } β,t are jointly Lipschitz in β and t ∈ [0 , T ] .

Proof. This is a consequence of lemma A.10 and proposition A.8.

Corollary A.12. For every x ∈ S d -1 and t ∈ [0 , T ] :

<!-- formula-not-decoded -->

Proof. With the usual notations one can write:

<!-- formula-not-decoded -->

The reminder R is bounded using lemma A.7 by:

<!-- formula-not-decoded -->

where the last line follows from lemma D.1 and lemma A.10. Hence, the proof can be concluded by noticing that:

<!-- formula-not-decoded -->

where we used the identities in equations 10, 12, 14.

We can finally pass to the limit in the PDE. Indeed consider a subsequence µ β of solutions to the PDE converging in C ([0 , T ] , P ( S d -1 )) to a certain probability measure µ ∞ . If we define the vector field χ ∞ ( x ) := P x V B T x | B T x | , then for every f ∈ C 2 b ( S d -1 ) :

<!-- formula-not-decoded -->

where we used that µ ∞ 0 = µ 0 = µ β 0 and that the PDE in weak form for µ β is:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

thanks to the fact that f is Lipschitz and by the definition of convergence in C ([0 , T ] , P ( S d -1 )) . For the second term:

<!-- formula-not-decoded -->

The first part goes to 0 by dominated convergence ( ∇ f, χ ∞ , χ β [ µ β ] are bounded, and χ β [ µ β ] → χ ∞ point-wise by lemma A.12). The second part goes to 0 by definition of the convergence µ ∞ → µ β and by equi-lipschitzianity of χ β [ µ β ] (see corollary A.11).

The uniqueness is standard, since µ is a probability measure and the vector field is smooth (see, for example, [5]).

Moreover, as β →∞ :

## A.2 Asymptotic behavior

This section studies the asymptotic behavior of the support of the solution to the partial differential equation:

<!-- formula-not-decoded -->

and in particular, we prove Proposition 3.4.

Lemma A.13. The ODE:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where y ( t ) = x ( f -1 ( t )) and f ( t ) = ∫ t 0 1 | B T x ( s ) | ds .

Remark A.14. The reparameterization is well defined since B is invertible.

Proof. : We have:

<!-- formula-not-decoded -->

This shows that the ODE is a time-reparameterization of the ODE for y ( t ) .

Lemma A.15. If z ( t ) solves:

<!-- formula-not-decoded -->

then y ( t ) = z ( t ) | z ( t ) | .

Proof. We have:

<!-- formula-not-decoded -->

This concludes the proof.

Corollary A.16. For Lebesgue almost every x 0 , the ω -limit set ω ( x 0 ) ⊂ E max .

Proof. This is a consequence of lemma A.13, lemma A.15 and of the classical theory for linear ODEs, after reducing to the Jordan canonical form of the matrix V B T .

Remark A.17. This technical result parallels Lemma 3.1 in [29], which was used to analyze the dynamics of the first token in causal attention.

And now we can finally prove Proposition 3.4:

is a time-reparameterization of:

Proof. Denote Φ t the flow of the ODE (17) and let ϕ ∈ C 2 b ( S d -1 ) be a test function with supp ( ϕ ) ⊂ E C max ∩ S d -1 . Fix µ ∞ ∈ ω ( µ 0 ) . Then there exists a divergent sequence of times { t k } k such that µ t k → µ ∞ weakly. As a consequence:

<!-- formula-not-decoded -->

where we used corollary A.16 and the dominated convergence theorem.

## B Proofs of the heat phase

In this section, we prove Proposition 3.9, which characterizes the second phase using the heat equation on the sphere.

Lemma B.1. Given a measure µ ∈ C 2 ( S d -1 ) ∩ P ( S d -1 ) strictly positive, then the following holds:

<!-- formula-not-decoded -->

with the gradient ∇ x and Hessian H x defined with respect to the standard Riemannian metric on S d -1 .

Proof.

<!-- formula-not-decoded -->

where, by lemma D.1:

<!-- formula-not-decoded -->

Hence:

<!-- formula-not-decoded -->

and noticing that E σ,β,I [ Y ] is parallel to x

, ν x (see proof of lemma A.6):

1

<!-- formula-not-decoded -->

where α 2 , β 2 are defined in the proof of Lemma A.6) and we used equation 13. To conclude, it suffices to replace equations 18 and the asymptotic estimates 14 and 15:

<!-- formula-not-decoded -->

Corollary B.2. Given a family { µ β } β of probability measures on S d -1 , suppose that there exist c, C &gt; 0 such that ∥ µ β || C 2 ( S d -1 ) ≤ C and µ β ≥ c for every β ≥ 0 . Moreover, assume there exists µ ∞ such that µ β → µ ∞ in C 1 ( S d -1 ) . Then

<!-- formula-not-decoded -->

where ∇ x is the gradient with respect to the standard Riemannian metric on S d -1 .

Proof of Proposition 3.9. Without loss of generality, set γ = -1 , the other case is analogous. The residual term is given by:

<!-- formula-not-decoded -->

It is sufficient to show that:

<!-- formula-not-decoded -->

Corollary B.2 guarantees convergence in C 0 ( S d -1 ) thanks to the assumptions on µ t . To improve this to higher regularity, we can use an interpolation argument through uniform bounds in C k +2 .

Define the kernel W β ( t ) := e βt K β , where K β := ∫ S d -1 e β ⟨ x,y ⟩ dσ ( y ) . Then,

<!-- formula-not-decoded -->

By the product rule, for every 0 ≤ j ≤ k +2 , there exists a polynomial p j such that:

<!-- formula-not-decoded -->

where we used that min x ∈ S d -1 W β ∗ µ ≥ min x ∈ S d -1 µ . The only thing left is to notice that ∥ W β ∗ µ ∥ C j ≤ C k ∥ µ ∥ C j , though proving this on S d -1 requires some care.

Consider the case j = 1 ( j &gt; 1 follows by induction) and fix v ∈ T x ( S d -1 ) . Let A ∈ so ( d ) (a skew-symmetric matrix) satisfying Ax = v , and define R ( t ) := e tA . In such a way R (0) x = x and R ′ (0) x = v . Then:

<!-- formula-not-decoded -->

where we used the change of variable z = R ( t ) T y , and the invariance of the measure on the sphere. Since ∥ W β ∥ L 1 = 1 , it follows that

<!-- formula-not-decoded -->

Higher derivatives follow similarly, completing the proof.

## C Proofs of the pairing phase

In this section we provide the proof of Proposition 3.11. Consider the interacting particle system on S d -1 described by the following ODEs corresponding to the case ( Q T K = V = Id ):

<!-- formula-not-decoded -->

where Z β ( x i ) = ∑ N j =1 e β ⟨ x i ,x j ⟩ and P x i ( x j ) = x j - ⟨ x i , x j ⟩ x i is the projection on the hyperplane orthogonal to x i . Suppose that there exists a unique pair ( i, j ) such that at initialization ⟨ x i , x j ⟩ = max i = j ⟨ x i , x j ⟩ and denote ⟨ x i ( t ) , x j ( t ) ⟩ := d t . Define also m t := max {⟨ x i , x j ⟩| i = j and { i, j } ̸ = { i, j }} .

̸

̸

Let α := arccos( m 0 ) -arccos( d 0 ) &gt; 0 and consider the time rescaling given by the inverse of dτ = e β (1 -d t ) dt , that we will still denote by t . Then:

<!-- formula-not-decoded -->

As usual the constant C can change from line to line, but it does not depend on α or β .

Lemma C.1. If β is such that Ce -β (1 -cos( α/ 4)) T ≤ 1 2 α , then:

<!-- formula-not-decoded -->

Proof. We proceed by a standard continuation argument. At t = 0 we have d 0 -m 0 ≥ 1 -cos( α ) ≥ 1 -cos( α/ 4) . Suppose the thesis holds on [0 , t ] . Then we have:

̸

- if i = i and j = j :

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

by Cauchy-Schwarz inequality, ⟨ P x i x j , P x i x j | P x i x j | ⟩ ≤ | P x i x j | , hence:

̸

<!-- formula-not-decoded -->

And in the last line we used that:

<!-- formula-not-decoded -->

In both cases the following holds:

<!-- formula-not-decoded -->

hence:

<!-- formula-not-decoded -->

that implies

<!-- formula-not-decoded -->

We can conclude:

<!-- formula-not-decoded -->

This is sufficient to close the continuation argument.

Remark C.2. We used the fact that, if arccos( x ) -arccos( y ) ≥ α , then y -x ≥ 1 -cos( α ) .

Now, recall Proposition 3.11:

Proposition C.3. The solutions x i ( t ) of the ODE system (SA) , under Assumptions 4 and positive V , with the rescaled time dt = e β (1 -⟨ x i ,x j ⟩ ) ds , converge as β →∞ to the solutions of the system:

<!-- formula-not-decoded -->

on finite intervals [0 , T ϵ ] , with T ϵ such that ⟨ y i , y j ⟩ ≤ 1 -ϵ throughout the interval, for any ϵ &gt; 0 .

First we need the following lemma:

LemmaC.4. If β is large enough then δ t := ⟨ y i , y j ⟩ ≤ 1 -c on [0 , T ] implies d t = ⟨ x i , x j ⟩ ≤ 1 -c/ 2 on [0 , T ] .

Proof. We proceed again using a continuation argument. The derivatives of the differences are bounded by:

<!-- formula-not-decoded -->

where we used | P x ( y ) | 2 = 1 -⟨ x, y ⟩ 2 and the previous lemma. The conclusion is again an application of Gronwall's lemma.

Proof of Proposition 3.11. Thanks to the previous lemma for β large enough, on [0 , T ϵ ] we have d t &lt; 1 -c ϵ . Now we can proceed with the proof:

Consider the case k = i .

<!-- formula-not-decoded -->

̸

where we used the previous lemma and the fact that e β Z β ( x i ) ≈ 1 -e -β (1 -d s ) ≈ 1 thanks to the propertyon T ϵ . The case k = i, j is similar. Hence:

<!-- formula-not-decoded -->

The conclusion is then just an application of Gronwall's lemma.

## D Useful lemmas

Lemma D.1. Let k &gt; 0 , β →∞ . Then:

<!-- formula-not-decoded -->

Proof. In the following C d,k is a constant that depends just on the dimension and on k and could change at each line:

<!-- formula-not-decoded -->

where M is the Kummer's confluent hypergeometric function and its asymptotic behavior for β →∞ (see [1]) is given by:

<!-- formula-not-decoded -->

Lemma D.2. Suppose that T is a tensor such that:

- T is invariant under permutations of the indices.
- T is invariant under rotations that fix a unit vector x .

Then if T is a 2-tensor, then it must be of the form:

<!-- formula-not-decoded -->

If T is a 3-tensor, then it must be of the form:

<!-- formula-not-decoded -->

̸

Proof. For simplicity, suppose d ≥ 5 . Without loss of generality we can assume x = e 1 . Let's start with the case of the 2 -tensor. For every i = j , we can consider another index l / ∈ { 1 , i, j } and the rotation R such that Re i = -e i (wlog i = 1 , otherwise use j ), Re l = -e l and elsewhere is the identity. Then:

̸

<!-- formula-not-decoded -->

that implies T [ e i , e j ] = 0 . If i = j &gt; 1 , then there exists a rotation R such that Re i = e 2 , Re 2 = -e i and the identity elsewhere:

<!-- formula-not-decoded -->

This conclude the proof for the 2-tensor.

For the 3-tensor: consider i, j, k such that i &gt; 1 and ( i / ∈ { j, k } or i = j = k ) . Consider another index l / ∈ { 1 , i, j, k } and the rotation R such that Re i = -e i , Re l = -e l and identity elsewhere. Then:

<!-- formula-not-decoded -->

that implies T [ e i , e j , e k ] = 0 . The only cases left are given by i = j and k = 1 and their permutations. If i = j &gt; 1 and k = 1 , then construct the rotation R such that Re i = e 2 and Re 2 = -e i to conclude, as above, that the tensor must be of the form α ( x i δ jk + x j δ ik + x k δ ij ) + βx i x j x k .

## E Supplementary figures

This experiment uses the same settings as in Figure 2 in the backward regime, but with an initial distribution given by a mixture of four wrapped Gaussians on the xy-plane. The red curve shows the interaction energy of the system over time on a logarithmic timescale. We highlight the three distinct timescales and the corresponding behaviors discussed in the paper.

Figure 4: Evolution of the dynamics with Q t K = V = S definite positive.

<!-- image -->