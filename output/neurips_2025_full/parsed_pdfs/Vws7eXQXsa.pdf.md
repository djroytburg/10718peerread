## Graph-Smoothed Bayesian Black-Box Shift Estimator and Its Information Geometry

## Masanari Kimura

School of Mathematics and Statistics The University of Melbourne m.kimura@unimelb.edu.au

## Abstract

̸

Label shift adaptation aims to recover target class priors when the labelled source distribution P and the unlabelled target distribution Q share P ( X | Y ) = Q ( X | Y ) but P ( Y ) = Q ( Y ) . Classical black-box shift estimators invert an empirical confusion matrix of a frozen classifier, producing a brittle point estimate that ignores sampling noise and similarity among classes. We present Graph-Smoothed Bayesian BBSE (GS-B 3 SE), a fully probabilistic alternative that places Laplacian-Gaussian priors on both target log-priors and confusion-matrix columns, tying them together on a label-similarity graph. The resulting posterior is tractable with HMC or a fast block Newton-CG scheme. We prove identifiability, N -1 / 2 contraction, variance bounds that shrink with the graph's algebraic connectivity, and robustness to Laplacian misspecification. We also reinterpret GS-B 3 SE through information geometry, showing that it generalizes existing shift estimators.

## 1 Introduction

Modern machine-learning systems are rarely deployed in exactly the same environment in which they were trained. When the distribution of class labels drifts but the class-conditional features remain stable, a phenomenon known as label shift, even a high-capacity model can produce arbitrarily biased predictions [54, 12, 41, 29, 44, 45, 51]. Practical examples include sudden changes in click-through behaviour of online advertising, evolving pathogen prevalence in medical diagnostics, and seasonal increments of certain object categories in autonomous driving. Because re-labelling target data is often prohibitively expensive, methods that recover the new class priors from a small unlabelled sample are indispensable precursors to reliable downstream decisions.

A popular remedy, the Black-Box Shift Estimator (BBSE) is to keep a single, frozen classifier ˆ h : X → Y trained on labelled source data and to link its predictions on the target domain to the unknown target priors through the confusion matrix C [39, 7]. The resulting formulation converts density shift into a linear system ˜ q = Cq that can be solved in closed form, after plugging in (i) an empirical estimate ˜ C computed on a small labelled validation set and (ii) the empirical prediction histogram ˜ q measured on unlabelled target instances. The elegance of BBSE has made it the default baseline for label-shift studies [45, 13, 38, 30, 42, 19].

Despite its popularity, the BBSE pipeline overlooks two sources of uncertainty that become debilitating in realistic, high-class-count regimes. (i) Finite-sample noise. Each column of ˜ C is estimated from at most a few hundred examples, so the matrix inversion layer can amplify small fluctuations into large errors on ˆ q . Regularised variants such as RLLS [7] or MLLS [18] damp variance but still return a point estimate whose uncertainty remains opaque to the user. (ii) Semantic structure. Classes in vision and language problems live on rich ontologies [35]: car and bus are more alike than

car and daisy . Standard BBSE fits each class independently and cannot borrow statistical strength across such related labels, leading to particularly fragile estimates for rare classes.

We introduce Graph-Smoothed Bayesian BBSE ( GS-B 3 SE ), a fully probabilistic alternative that attacks both weaknesses in a single hierarchical model. The key idea is to couple both the target log-prior vector and the columns of the confusion matrix through a Gaussian Markov random field defined on a label-similarity graph. Graph edges are obtained once from off-the-shelf text or image embeddings, and the resulting Laplacian precision shrinks parameters of semantically adjacent classes towards each other. Sampling noise is handled naturally by Bayesian inference: we place Gamma-Laplacian hyper-priors on the shrinkage strengths and sample the joint posterior with either Hamiltonian Monte Carlo [8, 9, 16] or a fast block Newton-conjugate-gradient optimizer [23, 10]. Moreover, we provide interpretation of GS-B 3 SE through the lens of information geometry [5, 6]. The analysis of statistical procedures or algorithms in this framework is known to help us understand them better [4, 1, 40, 28, 27, 25, 26, 2, 24].

Contributions. i) We formulate the joint Bayesian model that simultaneously regularizes the target prior and the confusion matrix with graph-based smoothness, reducing variance without hand-tuned penalties (in Section 4). ii) We provide theoretical guarantees: (a) posterior identifiability, (b) N -1 / 2 contraction, (c) class-wise variance bounds that tighten with the graph's algebraic connectivity, and (d) robustness to Laplacian misspecification (in Section 5). Moreover, we provide interpretation of GS-B 3 SE through the lens of information geometry framework and it shows that our algorithm is a natural generalization of existing methods. iii) An empirical study on several datasets demonstrates that GS-B 3 SE produces sharper prior estimates and improves downstream accuracy after Saerens correction compared to state-of-the-art baselines (in Section 7).

## 2 Related Literature

Classical estimators under label shift. Saerens et al. [47] proposed an EM algorithm that alternates between estimating the target prior and re-weighting posterior probabilities calculated by a fixed classifier. Lipton et al. [39] later formalised the Black-Box Shift Estimator (BBSE), showing that a single inversion of the empirical confusion matrix suffices when P ( X | Y ) is preserved. Subsequent refinements introduced regularisation to cope with ill-conditioned inverses: RLLS adds an ℓ 2 penalty to the normal equations [7], while MLLS frames the problem as a constrained maximum-likelihood optimisation [18]. All three methods remain point estimators and ignore uncertainty in ˜ C .

Bayesian and uncertainty-aware approaches. Caelen [11] derived posterior credible intervals for precision and recall by coupling Beta priors with multinomial counts; Ye et al. [53] extended the idea to label-shift estimation under class imbalance. Most of these models assume that classes are a priori independent, so posterior variance remains high for rare labels. Our work instead imposes structured Gaussian Markov random field (GMRF) priors that borrow strength across semantically related classes.

Graph-based smoothing and Laplacian priors. In spatial statistics, Laplacian-Gaussian GMRFs are a standard device for sharing information among neighbouring regions [49]. Recent machine learning studies exploit the same idea for discrete label graphs: Alsmadi et al. [3] introduced a graph-Dirichlet-multinomial model for text classification, and Ding et al. [15] used graph convolutions to smooth class logits. We follow this line but couple both the prior vector and every confusion-matrix column to the same similarity graph, yielding a joint posterior amenable to HMC and Newton-CG.

Domain-shift benchmarks and failure modes. Large-scale empirical studies such as WILDS [30] and Mandoline [13] describe how brittle point estimators become under distribution drift and class imbalance. Rabanser et al. [45] demonstrated that label-shift detectors without calibrated uncertainty frequently produce over-confident but wrong alarms. By delivering credible intervals whose width shrinks with the graph's algebraic connectivity, our method directly tackles this shortcoming.

Positioning of this work. GS-B 3 SE unifies three strands of research: (i) black-box label-shift estimation, (ii) Bayesian confusion-matrix modelling, and (iii) graph-structured smoothing. To justify our proposed method, we show posterior identifiability and N -1 / 2 contraction, and derive variance bounds that scale with λ 2 ( L ) . Empirically, our method plugs seamlessly into existing shift-benchmark pipelines, providing calibrated uncertainty absent from earlier regularised or EM-style alternatives.

## 3 Preliminarily

Problem Setting Let X be an input space and Y = { 1 , . . . , K } be a label set, where K ≥ 2 is the number of classes. For source and target distributions P and Q , the label shift assumption states that the class-conditional feature laws remain unchanged while the class priors may differ:

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Black-Box Shift Estimator Train once, on source data, an arbitrary measurable classifier ˆ h : X → Y . Denote its confusion matrix under P by C ∈ (0 , 1) K × K , where C j,i = Pr P [ ˆ h ( X ) = j | Y = i ] and ∑ j C j,i = 1 . Notice that C depends only on the source distribution and can be estimated on a held-out validation set with known labels. Write the empirical estimate as ˜ C . Let m labelled source-validation points ( x i , y i ) m i =1 be used to form ˜ C . Apply the same fixed ˆ h to unlabeled target instances { x ′ t } n ′ t =1 of size n ′ . Let ˜ q j = Pr Q [ ˆ h ( X ) = j ] , ˜ q = (˜ q 1 , . . . , ˜ q K ) ⊤ . Because the class-conditionals are shared, Bayes' rule gives

<!-- formula-not-decoded -->

and in vector form, it can be written as ˜ q = Cq . Eq. 2 is the identifiability equation for label shift.

Assume C is invertible or full column rank. Then the population target prior is q = C -1 ˜ q . Under this observation, BBSE framework [39] converts black-box classifier predictions on unlabeled target data into an estimate of the unknown target class prior by solving the linear system.

## 4 Methodology

The usual BBSE treats the confusion matrix C as fixed. In practice C is estimated from a finite validation set and is itself ill-conditioned when some classes are rare. To address this problem, we consider extending BBSE framework by the joint Bayesian model for both confusion matrix and target priors with the graph coupling. Let n S i be a number of source examples with label i , and n S i = ( n S 1 ,i , . . . , n S K,i ) ⊤ be the counts of ˆ h ( X ) = j among those n S i . Also, let ˜ n = (˜ n 1 , . . . , ˜ n K ) ⊤ be the counts of ˆ h ( X ) = j on unlabeled target data. For each true class i ,

<!-- formula-not-decoded -->

where C : ,i is the i -column of C and Multi( · , · ) is the multinomial distribution. Similarly,

<!-- formula-not-decoded -->

The complete-data likelihood factorises

<!-- formula-not-decoded -->

Wenowconsider to utilize similarity information between classes. Let G = ( Y , E , W ) be a similarity graph on labels with the weight matrix W , E is the edges and L is the graph Laplacian. In G , each vertex corresponds to a class label; an edge ( i, j ) ∈ E indicates that labels i and j are semantically or visually similar. Weights W ij ∈ [0 , 1] quantify that similarity, with W ij = 0 when no edge is present. In our methodology, we use the unnormalised Laplacian L = D -W where D ii = ∑ j W ij . Because we enforce connectivity, L has exactly one zero eigenvalue, making λ 2 ( L ) (the algebraic connectivity) strictly positive as required by our theory. Introduce log-odds vector

<!-- formula-not-decoded -->

and consider the following Gaussian Markov random field (GMRF) prior

<!-- formula-not-decoded -->

where a q , b q &gt; 0 are hyper-parameters. A Laplacian-based precision shrinks log-odds differences along graph edges, promoting smooth class priors across semantically similar labels [31], and recover q = softmax ( θ ) .

Treat each column C : ,i as a latent simplex vector with Dirichlet-log-normal hierarchy:

- i) Latent log-odds ϕ i ∈ R K , ϕ ⊤ i 1 = 0 .
2. ii) Conditional prior p ( ϕ i | τ C ) ∝ exp ( -τ C 2 ϕ ⊤ i Lϕ i ) . All ϕ i share the same Laplacian L over predicted labels so that columns corresponding to neighbouring predicted classes exhibit similar shape.
3. iii) Transformation to the simplex C j,i = exp ϕ j,i ∑ K ℓ =1 exp ϕ ℓ,i for i = 1 , . . . , K .
4. iv) Hyper-prior τ C ∼ Gamma( a C , b C ) , with a C , b C &gt; 0 .

̸

The resulting distribution on each C : ,i is a logistic-Normal on the simplex and it reduces to an ordinary Dirichlet when L = 0 but gains graph-coupled precision for L = 0 . The full hierarchical model is as follows.

<!-- formula-not-decoded -->

Here, the Moore-Penrose pseudoinverse L † appears because L is singular, and the constraint θ ⊤ 1 = 0 ensures uniqueness.

For the posterior inference, consider the following log-joint distribution.

<!-- formula-not-decoded -->

All terms are differentiable, enabling Hamiltonian Monte Carlo (HMC) in the unconstrained variables. Because Eq. 4 is concave in each block after reparameterization, a block-Newton scheme alternates, i) update { ϕ i } K i =1 by one Newton-CG step using sparse Laplacian Hessian, ii) update θ likewise, iii) closed-form updates for τ C , τ q from Gamma posteriors. Convergence is super-linear due to the strict convexity induced by the Laplacian energies. The posterior predictive distribution of the confusion-weighted target counts is

<!-- formula-not-decoded -->

Credible intervals for each q i reflect both sampling noise and model-induced graph smoothing, an advantage over plug-in BBSE. If no prior similarity information exists one may default to W ij = 1 { i = j } , in which case our model reduces to an independent logistic-Normal prior and all theoretical guarantees still hold.

## Relationship to Existing Work

- Replaces the point estimate of BBSE C -1 ˜ q with a full posterior, relating Bayesian confusion matrix treatments [11].
- Laplacian GMRFs generalise classical Dirichlet priors by borrowing strength along graph edges, extending recent graph-Dirichlet-multinomial models [31].
- When L = 0 and Gamma hyper-priors degenerate to delta masses, our hierarchical model reduces exactly to deterministic BBSE.

## 5 Theory

This section provides the theoretical foundations of the proposed method. See Appendix A for the detailed proofs. First, the following lemma on identifiability is introduced. Although an analogous statement has been implicitly argued in the prior work [39], the full proof is included in the Appendix A to make this study self-contained.

̸

Lemma 1. Let C and C ′ be two column-stochastic matrices with strictly positive entries: C j,i &gt; 0 , C ′ j,i &gt; 0 , ∑ K j =1 C j,i = ∑ K j =1 C ′ j,i = 1 for 1 ≤ i ≤ K . In addition, assume C and C ′ are invertible, or equivalently, det C = 0 and det C ′ = 0 . For any deterministic sample sizes n S i ∈ { 1 , 2 , . . . } and n ′ ∈ { 1 , 2 , . . . } , define the data-generating distributions

̸

<!-- formula-not-decoded -->

Suppose that, for every choice of the sample sizes { n S i =1 } K i =1 and n ′ , ( { N S i } K i =1 , ˜ N ) d = ( { N S ′ i } K i =1 , ˜ N ′ ) , as random vectors in N K 2 + K , where the left-hand side is generated by ( C , q ) and the right-hand side by ( C ′ , q ′ ) . Then, C = C ′ and q = q ′ .

Lemma 1 implies that the mapping ( q , C ) ↦→ {{ n S i } , ˜ n } is injective up to measure-zero label permutations when the graph is connected and all source classes appear.

Let ∆ K -1 be the ( K -1) -dimensional probability simplex:

<!-- formula-not-decoded -->

The following lemma provides the support condition needed in statements described later.

̸

Lemma 2. Let ( q 0 , C 0 ) be the true parameter pair, where q 0 ∈ ∆ K -1 and C 0 ∈ (0 , 1) K × K with det C 0 = 0 . Define the Euclidean small ball as

<!-- formula-not-decoded -->

for some radius ϵ &gt; 0 small enough that all vectors in the ball stay strictly inside the simplex. Then, for every ϵ , the joint prior distribution Π on ( q , C ) assigns strictly positive mass to the ball: Π( B ϵ ( q 0 , C 0 )) &gt; 0 .

This lemma states that positivity of Gaussian density and the smooth bijection yield the positive push-forward density, and it is the classical strategy used for logistic-Gaussian process priors in density estimation [50].

Lemmas 1, 2 and the classical results from Ghosal et al. [20], Van der Vaart [52] gives the following statement.

̸

Proposition 1. Let K ≥ 2 be fixed and ( q 0 , C 0 ) be the true parameters pair with q 0 ∈ ˚ ∆ K -1 and C 0 ∈ (0 , 1) K × K , where ˚ ∆ K -1 is the interior of ∆ K -1 , and det C 0 = 0 . Suppose that the data consist of { N S i } K i =1 and ˜ N where conditionally on ( q 0 , C 0 ) ,

<!-- formula-not-decoded -->

Also suppose that sample sizes diverge with the single index: N := n ′ + ∑ K i =1 n S i → ∞ , and min i n S i →∞ . Then, for every ϵ &gt; 0 , Π( B c ϵ | data ) P ( q 0 , C 0 ) - - - - -→ N →∞ 0 .

Under the same assumption in Proposition 1, the following statements about the posterior contraction rate are obtained.

Theorem 1. Let the data-generating model, true parameter pair ( q 0 , C 0 ) , and diverging sample sizes N = n ′ + ∑ K i =1 n S i →∞ satisfy the setup spelled out before Proposition 1. Define the Euclidean radius ϵ N := M/ √ N, M &gt; 0 arbitrary but fixed. Let

<!-- formula-not-decoded -->

Under Lemma 1 and 2, the joint posterior Π( · | data ) for the Laplacian-Gaussian hierarchy satisfies

<!-- formula-not-decoded -->

That is, the posterior contracts around the truth at the parametric rate N -1 / 2 .

Corollary 1. Retain the setting and notation of Theorem 1. For each class i , write

<!-- formula-not-decoded -->

under the joint posterior Π( · | data ) . Let L be the connected-graph Laplacian used in the GMRF prior and let λ 2 ( L ) := min { λ &gt; 0 : λ is an eigenvalue of L } be its algebraic connectivity. Assume the hyper-parameter τ q is fixed, or sampled from a Gamma prior independent of N . Then, there exists a constant C &gt; 0 , depending only on the true ( q 0 , C 0 ) and on K , such that for every sample size N large enough,

<!-- formula-not-decoded -->

Finally, we can show the following statement about the robustness to graph Laplacian misspecification.

̸

Proposition 2. Let L 0 be the true Laplacian, and F 0 := diag( C 0 q 0 ) -( C 0 q 0 )( C 0 q 0 ) ⊤ ⪰ 0 be the Fisher information of θ in the target multinomial likelihood. For L = L 0 , let ¯ θ N := E [ θ | data ] be the posterior mean of θ under the misspecified prior. Then, for all sample sizes N large enough,

<!-- formula-not-decoded -->

In particular,

<!-- formula-not-decoded -->

where λ min ( F 0 ) &gt; 0 and λ 2 ( L ) &gt; 0 are, respectively, the smallest eigenvalue of F 0 and the algebraic connectivity of L .

̸

Thus, we can see that the bias decays as N -1 when L = L 0 and if the graphs coincide the leading term vanishes and the posterior mean is unbiased up to the usual N -1 / 2 noise. Moreover, Proposition 2 states that a larger algebraic connectivity λ 2 ( L ) reduces bias, emphasising the benefit of rich similarity structures.

## 6 Interpretation via Information Geometry

The basic notations of information geometry used in this section are summarized in Appendix B. The K -1 simplex ∆ K -1 := { q &gt; 0 : 1 ⊤ K q = 1 } is a Riemannian manifold when equipped with the Fisher-Rao metric g q ( v , w ) = ∑ K i =1 v i w i q i , for v , w ∈ T q ∆ K -1 , where T q ∆ K -1 := { v : 1 ⊤ v = 0 } is the tangent space. The natural potential on this manifold is minus entropy ψ ( q ) = ∑ i q i log q i whose Euclidean gradient is the centred log-odds vector θ used in the previous section. These facts allow us to cast GS-B 3 SE as a Riemannian penalised likelihood. The dual affine coordinates are m -coordinates q i (mixture parameters) and e -coordinates θ i = log q i -1 ∑ log q j (centred log-odds). The convex potential ψ ( q ) = ∑ i =1 q i log q i is minus Shannon entropy and satisfies ∇ Euc q ψ ( q ) = θ ; together ( ψ, θ ) endow ∆ K -1 with the classical dually-flat structure of information geometry [5, 6]. See standard textbooks for detailed explanation of concepts in differential geometry and Riemannian manifold [36, 37, 34, 17, 43, 21, 33].

K j K

Denote by ˆ r = ˜ n /n ′ and M = { Cq : q ∈ ∆ K -1 } the empirical prediction histogram and the m -flat sub-manifold induced by the frozen classifier. The negative log-posterior derived in Section 4 can be written as

<!-- formula-not-decoded -->

Thus Eq. (8) is a sum of an m -convex and an e -convex potential, so it is geodesically convex under the Fisher-Rao metric (see Table 1).

Table 1: Information geometric identification of GS-B 3 SE.

| Term             | Geometric meaning                                                                                                                 |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| D KL [ˆ r ∥ Cq ] | Canonical divergence between ˆ r and the m -flat model M .                                                                        |
| τ q 2 θ ⊤ L θ    | Quadratic form in e -coordinates ⇒ e -convex barrier that bends the manifold in the directions encoded by the graph Laplacian L . |

Theorem 2 (Geodesic convexity of F ) . For every q ∈ ˚ ∆ K -1 the Riemannian Hessian of F satisfies

<!-- formula-not-decoded -->

where F 0 = diag( Cq ) -( Cq )( Cq ) ⊤ is the Fisher information of the multinomial likelihood and λ 2 ( L ) the algebraic connectivity of the label graph. Hence F is α -strongly geodesically convex with α = n ′ λ min ( F 0 ) + τ q λ 2 ( L ) &gt; 0 .

The proof in Appendix A explicitly decomposes any tangent direction into an m -straight and an e -straight component and shows that the lower bound remains positive because both components contribute additively.

## 6.1 Natural-Gradient Dynamics

The natural gradient of F is

<!-- formula-not-decoded -->

where ⊙ is component-wise product. The associated flow ˙ q ( t ) = -grad FR F ( q ( t ) ) is the steepest-descent curve in the Fisher-Rao geometry.

Proposition 3 (Natural-gradient flow of the penalised objective) . Under the Fisher-Rao metric g q ( v , w ) = ∑ K i =1 v i w i /q i the natural gradient grad FR F ( q ) of F is

<!-- formula-not-decoded -->

where the division is element-wise. Consequently the un-constrained natural-gradient flow

<!-- formula-not-decoded -->

preserves the simplex and coincides with the replicator-Laplacian dynamical system: ˙ q t,j = -q t,j [ n ′ [ C ⊤ ( 1 -ˆ r / r )] + τ q [ Lθ t ] ] j .

## Remarks

- i) Role of Laplacian When the Laplacian term is absent ( τ q = 0 ), the flow reduces to the classical replicator equation that drives every class-probability q j proportionally to the (signed) log-likelihood residual [ C ⊤ ( 1 -ˆ r / r )] j . The graph-Laplacian contribution -τ q q j [ Lθ ] j plays the role of a mutation / diffusion force that mixes mass along edges of the label graph and prevents degenerate solutions.
2. ii) Role of the algebraic connectivity. From Theorem 2 the strong-convexity modulus is α = n ′ λ min ( F 0 ) + τ q λ 2 ( L ) . Along the flow we have d dt F ( q t ) = -∥ grad FR F ( q t ) ∥ 2 g q t ≤ -2 α ( F ( q t ) -F ∗ ) , so F decays exponentially fast with rate proportional to the algebraic connectivity λ 2 ( L ) ; a denser-connected label graph therefore accelerates convergence.
3. iii) Link to Saerens EM correction. If we freeze the confusion matrix and drop the Laplacian term, the stationary condition C ⊤ (ˆ r / r ) = 1 is exactly the fixed point solved (iteratively) by the Saerens EM method [47].

Table 2: Baseline methods and their key ideas.

| Method           | Key idea                                                                                                                                                                                                                                  |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| BBSE [39]        | Solve ˆ C ˆ q = ˆ y with the empirical confusion matrix (no re-training).                                                                                                                                                                 |
| EM [47]          | Expectation-Maximization that iteratively re-estimates priors and re-weights posteriors.                                                                                                                                                  |
| RLLS [7]         | Adds an ℓ 2 penalty to the BBSE normal equations to control variance for small n ′ .                                                                                                                                                      |
| MLLS [18]        | Maximum-likelihood estimation of the label-ratio vector; unifies BBSE & RLLS and optimizes q directly.                                                                                                                                    |
| GS-B 3 SE (ours) | Joint Bayesian inference of both target priors q and confusion matrix C . The hierarchical model couples classes along a label-similarity graph, shrinking estimates in low-count regimes and yielding full posterior credible intervals. |

## 6.2 Dual Projections and the Pythagorean Identity

Let Π m (ˆ r ) be the m -projection of the data onto M and Π e ( q 0 ) the e -projection of the hyper-prior center onto the same manifold. At the optimum q ⋆ we have Π m (ˆ r ) = Π e ( q 0 ) = Cq ⋆ , and the generalized Pythagorean theorem [5] gives D KL [ ˆ r ∥ q 0 ] = D KL [ ˆ r ∥ Cq ⋆ ] + D KL [ Cq ⋆ ∥ q 0 ] , where the second term equals the Laplacian regulariser τ q 2 n ′ θ ⊤ L θ . Hence GS-B 3 SE can be summarized as find the unique intersection of an m -geodesic (data fit) and an e -ellipsoid (graph prior).

## 7 Experiments

## 7.1 Experimental Protocol and Implementation

Datasets and synthetic label shifts. We evaluate on MNIST ( K = 10) [14], CIFAR-10 ( K = 10) and CIFAR-100 ( K = 100) datasets [32]. For each dataset we treat the official training split as the source domain and the official test split as the pool from which an unlabelled target domain is drawn. Source class-priors are kept uniform p = (1 /K,... , 1 /K ) . Target priors are deliberately perturbed:

<!-- formula-not-decoded -->

where u K = (1 , . . . , K ) ⊤ . In our experiments, we set α = 0 . 05 and b = 1 . 1 . The procedure is: i) Source set: Sample 10 , 000 instances from the training partition according to p and train a backbone classifier (ResNet-18 [22, 48]) for 100 epochs with standard data-augmentation. ii) Validation set: Hold out 5 , 000 labelled source instances, stratified by p , to estimate the empirical confusion matrix ˜ C . iii) Target set: Draw n ′ = 10 , 000 unlabelled instances from the test partition using probabilities q . These labels are revealed only for evaluation.

Graph construction on labels. For every dataset we embed the class names with the frozen CLIP ViT-B/32 text encoder [46], obtain { e i } K i =1 ⊂ R 512 , | e i | 2 = 1 , and build a k -nearest-neighbour graph

<!-- formula-not-decoded -->

Edge weights are W ij = exp ( -∥ e i -e j ∥ 2 2 /σ 2 ) with σ set to the median pairwise distance inside E . The resulting k -NN graph is connected, so its unnormalised Laplacian L = D -W satisfies λ 2 ( L ) &gt; 0 . For MNIST, where class names are single digits, we instead construct E from 4-NN in the Euclidean space of 128-d penultimate-layer features averaged over the training images.

Hyper-priors and inference. Gamma hyper-priors: a q = b q = a C = b C = 1 , giving vague Gamma(1 , 1) on τ q and τ C . Four independent HMC chains, each with 500 warm-up (NUTS) and 1,000 posterior iterations; leap-frog step-size adaptively tuned. Block Newton-CG inner optimizer:

Table 3: Label shift estimation and downstream performance. Lower is better for ∥ ˆ q -q ∥ 1 ; higher is better for post-correction accuracy. Best results are bold . ± shows one bootstrap standard error. ( 1 000 resamples).

|           | MNIST ( K =10 )   | MNIST ( K =10 )   | CIFAR-10 ( K =10 )   | CIFAR-10 ( K =10 )   | CIFAR-100 ( K =100 )   | CIFAR-100 ( K =100 )   |
|-----------|-------------------|-------------------|----------------------|----------------------|------------------------|------------------------|
| Method    | ∥ ˆ q - q ∥ 1 ↓   | Acc ↑             | ∥ ˆ q - q ∥ 1 ↓      | Acc ↑                | ∥ ˆ q - q ∥ 1 ↓        | Acc ↑                  |
| BBSE      | 0.038 ± 0.007     | 0.942 ± 0.002     | 0.112 ± 0.015        | 0.781 ± 0.004        | 1.62 ± 0.05            | 0.690 ± 0.006          |
| EM        | 0.052 ± 0.015     | 0.935 ± 0.008     | 0.194 ± 0.033        | 0.732 ± 0.012        | 2.10 ± 0.14            | 0.632 ± 0.026          |
| RLLS      | 0.016 ± 0.004     | 0.959 ± 0.003     | 0.072 ± 0.010        | 0.803 ± 0.004        | 0.92 ± 0.03            | 0.712 ± 0.006          |
| MLLS      | 0.010 ± 0.003     | 0.963 ± 0.002     | 0.052 ± 0.008        | 0.812 ± 0.004        | 0.71 ± 0.03            | 0.734 ± 0.006          |
| GS-B 3 SE | 0.002 ± 0.001     | 0.986 ± 0.002     | 0.025 ± 0.004        | 0.844 ± 0.003        | 0.22 ± 0.02            | 0.783 ± 0.005          |

tolerance 10 -4 , at most eight iterations per Newton step, stop when the relative change of the joint log-density falls below 10 -3 . All routines implemented in PyMC and run on a single NVIDIA T4.

Baselines. We compare against a) plug-in BBSE [39], b) the EM-style Saerens re-weighting [47], c) RLLS [7] with ℓ 2 -regularisation and d) MLLS [18] tuned on a held-out split. All baselines receive the same ˜ C and target predictions ˆ h ( x ) . Table 2 summarizes the baseline methods and their key ideas, including our method.

Evaluation. We report prior-error | ˆ q -q | 1 and downstream accuracy after Saerens likelihood correction using the estimated priors. Significance is assessed with 1,000 paired bootstrap resamples of the target set.

## 7.2 Main Empirical Findings

Table 3 compares GS-B 3 SE with four widely-used point estimators on three datasets.

Sharper prior estimates. Across all datasets GS-B 3 SE reduces the ℓ 1 error ∥ ˆ q -q ∥ 1 by large margins: i) MNIST ( K =10) -already a benign scenario- error falls from 0 . 010 (best baseline, MLLS) to 0 . 002 ( × 5 improvement), ii) CIFAR-10 ( K =10) -richer images and heavier label skew- error halves from 0 . 052 to 0 . 025 , CIFAR-100 ( K =100) -the high-class-count regime- graph smoothing is essential: GS-B 3 SE reaches 0 . 22 versus 0 . 71 . The advantage widens with K , confirming the benefit of borrowing strength along the label graph when per-class counts are scarce.

Better downstream accuracy. Feeding the estimated priors into Saerens post-processing improves final accuracy in proportion to the quality of the prior. GS-B 3 SE attains 0.986 on MNIST, 0.844 on CIFAR-10 and 0.783 on CIFAR-100-absolute gains of +2 . 3 ,pp, +3 . 2 ,pp and +4 . 9 ,pp over the strongest non-Bayesian competitor (MLLS) on the respective datasets.

## 8 Conclusion

We presented GS-B 3 SE, a graph-smoothed Bayesian generalization of the classical black-box shift estimator. By tying both the target prior q and every column of the confusion matrix C together through a Laplacian-Gaussian hierarchy, the model simultaneously i) shares statistical strength across semantically related classes, ii) quantifies all uncertainty arising from finite validation and target samples, and admits scalable inference with either HMC or a Newton-CG variational surrogate. We proved that the resulting posterior is identifiable, contracts at the optimal N -1 / 2 rate, and that its class-wise variance decays inversely with the graph's algebraic connectivity λ 2 ( L ) . A robustness bound further shows that even with a misspecified graph the bias vanishes as N -1 . Because our approach is a pure post-processing layer that needs only a frozen classifier, a tiny labelled validation set, and a pre-computed label graph, it can be retro-fitted to virtually any deployed model.

Limitations and future work. i) Our current graph is built from CLIP or feature embeddings; learning the graph jointly with the posterior could adapt it to the task. ii) Although inference is already tractable, further speed-ups via structured variational approximations would make GS-B 3 SE attractive for extreme-label settings. iii) As declared in Section 7.1, our experiments used a single NVIDIA T4; scaling to larger datasets remains future work.

## References

- [1] Shotaro Akaho. The e-pca and m-pca: Dimension reduction of parameters by information geometry. In 2004 IEEE International Joint Conference on Neural Networks (IEEE Cat. No. 04CH37541) , volume 1, pages 129-134. IEEE, 2004.
- [2] Shotaro Akaho and Kazuya Takabatake. Information geometry of contrastive divergence. In ITSL , pages 3-9, 2008.
- [3] Mutasem K Alsmadi, Malek Alzaqebah, Sana Jawarneh, Ibrahim ALmarashdeh, Mohammed Azmi Al-Betar, Maram Alwohaibi, Noha A Al-Mulla, Eman AE Ahmed, and Ahmad Al Smadi. Hybrid topic modeling method based on dirichlet multinomial mixture and fuzzy match algorithm for short text clustering. Journal of Big Data , 11(1):68, 2024.
- [4] Shun-Ichi Amari. Natural gradient works efficiently in learning. Neural computation , 10(2): 251-276, 1998.
- [5] Shun-ichi Amari. Information geometry and its applications , volume 194. Springer, 2016.
- [6] Shun-ichi Amari and Hiroshi Nagaoka. Methods of information geometry , volume 191. American Mathematical Soc., 2000.
- [7] Kamyar Azizzadenesheli, Anqi Liu, Fanny Yang, and Animashree Anandkumar. Regularized learning for domain adaptation under label shifts. In International Conference on Learning Representations , 2019.
- [8] Michael Betancourt. A conceptual introduction to hamiltonian monte carlo. arXiv preprint arXiv:1701.02434 , 2017.
- [9] Michael Betancourt and Mark Girolami. Hamiltonian monte carlo for hierarchical models. Current trends in Bayesian methodology with applications , 79(30):2-4, 2015.
- [10] Albert G Buckley. A combined conjugate-gradient quasi-newton minimization algorithm. Mathematical Programming , 15(1):200-210, 1978.
- [11] Olivier Caelen. A bayesian interpretation of the confusion matrix. Annals of Mathematics and Artificial Intelligence , 81(3):429-450, 2017.
- [12] Yee Seng Chan and Hwee Tou Ng. Word sense disambiguation with distribution estimation. In IJCAI , volume 5, pages 1010-5, 2005.
- [13] Mayee Chen, Karan Goel, Nimit S Sohoni, Fait Poms, Kayvon Fatahalian, and Christopher Ré. Mandoline: Model evaluation under distribution shift. In International conference on machine learning , pages 1617-1629. PMLR, 2021.
- [14] Li Deng. The mnist database of handwritten digit images for machine learning research [best of the web]. IEEE signal processing magazine , 29(6):141-142, 2012.
- [15] Yun Ding, Yanwen Chong, Shaoming Pan, and Chun-Hou Zheng. Class-imbalanced graph convolution smoothing for hyperspectral image classification. IEEE Transactions on Geoscience and Remote Sensing , 62:1-18, 2024.
- [16] Simon Duane, Anthony D Kennedy, Brian J Pendleton, and Duncan Roweth. Hybrid monte carlo. Physics letters B , 195(2):216-222, 1987.
- [17] Luther Pfahler Eisenhart. Riemannian geometry , volume 19. Princeton university press, 1997.
- [18] Saurabh Garg, Yifan Wu, Sivaraman Balakrishnan, and Zachary Lipton. A unified view of label shift estimation. Advances in Neural Information Processing Systems , 33:3290-3300, 2020.
- [19] Saurabh Garg, Nick Erickson, James Sharpnack, Alex Smola, Sivaraman Balakrishnan, and Zachary Chase Lipton. Rlsbench: Domain adaptation under relaxed label shift. In International Conference on Machine Learning , pages 10879-10928. PMLR, 2023.
- [20] Subhashis Ghosal, Jayanta K Ghosh, and Aad W Van Der Vaart. Convergence rates of posterior distributions. Annals of Statistics , pages 500-531, 2000.

- [21] Heinrich W Guggenheimer. Differential geometry . Courier Corporation, 2012.
- [22] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 770-778, 2016.
- [23] Magnus R Hestenes, Eduard Stiefel, et al. Methods of conjugate gradients for solving linear systems. Journal of research of the National Bureau of Standards , 49(6):409-436, 1952.
- [24] Hideitsu Hino, Shotaro Akaho, and Noboru Murata. Geometry of em and related iterative algorithms. Information Geometry , 7(Suppl 1):39-77, 2024.
- [25] Masanari Kimura. Generalized t-sne through the lens of information geometry. IEEE Access , 9: 129619-129625, 2021.
- [26] Masanari Kimura and Howard Bondell. Density ratio estimation via sampling along generalized geodesics on statistical manifolds. In The 28th International Conference on Artificial Intelligence and Statistics , 2025. URL https://openreview.net/forum?id=v13muX4Q3i .
- [27] Masanari Kimura and Hideitsu Hino. α -geodesical skew divergence. Entropy , 23(5):528, 2021.
- [28] Masanari Kimura and Hideitsu Hino. Information geometrically generalized covariate shift adaptation. Neural Computation , 34(9):1944-1977, 2022.
- [29] Masanari Kimura and Hideitsu Hino. A short survey on importance weighting for machine learning. Transactions on Machine Learning Research , 2024. ISSN 2835-8856. URL https: //openreview.net/forum?id=IhXM3g2gxg . Survey Certification.
- [30] Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, et al. Wilds: Abenchmark of in-the-wild distribution shifts. In International conference on machine learning , pages 5637-5664. PMLR, 2021.
- [31] Bartosz Kołodziejek, Jacek Wesołowski, and Xiaolin Zeng. Discrete parametric graphical models with dirichlet type priors. arXiv preprint arXiv:2301.06058 , 2023.
- [32] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
- [33] Wolfgang Kühnel. Differential geometry , volume 77. American Mathematical Soc., 2015.
- [34] Serge Lang. Differential and Riemannian manifolds . Springer Science &amp; Business Media, 1995.
- [35] Patrice Latinne, Marco Saerens, and Christine Decaestecker. Adjusting the outputs of a classifier to new a priori probabilities may significantly improve classification accuracy: Evidence from a multi-class problem in remote sensing. In ICML , volume 1, pages 298-305, 2001.
- [36] John M Lee. Riemannian manifolds: an introduction to curvature , volume 176. Springer Science &amp; Business Media, 2006.
- [37] John M Lee. Introduction to Riemannian manifolds , volume 2. Springer, 2018.
- [38] Jian Liang, Ran He, and Tieniu Tan. A comprehensive survey on test-time adaptation under distribution shifts. International Journal of Computer Vision , 133(1):31-64, 2025.
- [39] Zachary Lipton, Yu-Xiang Wang, and Alexander Smola. Detecting and correcting for label shift with black box predictors. In International conference on machine learning , pages 3122-3130. PMLR, 2018.
- [40] Noboru Murata, Takashi Takenouchi, Takafumi Kanamori, and Shinto Eguchi. Information geometry of u-boost and bregman divergence. Neural Computation , 16(7):1437-1481, 2004.
- [41] Tuan Duong Nguyen, Marthinus Christoffel, and Masashi Sugiyama. Continuous target shift adaptation in supervised learning. In Asian Conference on Machine Learning , pages 285-300. PMLR, 2016.

- [42] Sunghyun Park, Seunghan Yang, Jaegul Choo, and Sungrack Yun. Label shift adapter for testtime adaptation under covariate and label shifts. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 16421-16431, 2023.
- [43] Peter Petersen. Riemannian geometry , volume 171. Springer, 2006.
- [44] Joaquin Quiñonero-Candela, Masashi Sugiyama, Anton Schwaighofer, and Neil D Lawrence. Dataset shift in machine learning . Mit Press, 2022.
- [45] Stephan Rabanser, Stephan Günnemann, and Zachary Lipton. Failing loudly: An empirical study of methods for detecting dataset shift. Advances in Neural Information Processing Systems , 32, 2019.
- [46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning , pages 8748-8763. PmLR, 2021.
- [47] Marco Saerens, Patrice Latinne, and Christine Decaestecker. Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. Neural computation , 14(1):21-41, 2002.
- [48] Muhammad Shafiq and Zhaoquan Gu. Deep residual learning for image recognition: A survey. Applied sciences , 12(18):8972, 2022.
- [49] Gavin Simpson. First steps with mrf smooths. From the Bottom of the Heap , 2017.
- [50] Surya T Tokdar and Jayanta K Ghosh. Posterior consistency of logistic gaussian process priors in density estimation. Journal of statistical planning and inference , 137(1):34-42, 2007.
- [51] Antonio Torralba and Alexei A Efros. Unbiased look at dataset bias. In CVPR 2011 , pages 1521-1528. IEEE, 2011.
- [52] Aad W Van der Vaart. Asymptotic statistics , volume 3. Cambridge university press, 2000.
- [53] Changkun Ye, Russell Tsuchida, Lars Petersson, and Nick Barnes. Label shift estimation for class-imbalance problem: A bayesian approach. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision , pages 1073-1082, 2024.
- [54] Kun Zhang, Bernhard Schölkopf, Krikamol Muandet, and Zhikun Wang. Domain adaptation under target and conditional shift. In International conference on machine learning , pages 819-827. Pmlr, 2013.

## A Proofs

Proof for Lemma 1. Fix an index i ∈ { 1 , . . . , K } and an arbitrary source sample size n S i = n ≥ 1 . Denote N S i = ( N S 1 ,i , . . . , N S K,i ) ⊤ . Under parameter pair ( C , q ) , we have

<!-- formula-not-decoded -->

Under ( C ′ , q ′ ) , the same vector has pmf

<!-- formula-not-decoded -->

By the assumption, these two pmfs coincide for every integer vector k with the given total n . Canceling the multinomial coefficient yields

<!-- formula-not-decoded -->

Pick the K specific count vectors

<!-- formula-not-decoded -->

Plugging k ( ℓ ) into Eq. 11 gives

<!-- formula-not-decoded -->

Because n ≥ 1 , taking the n -th root yields C ℓ,i = C ′ ℓ,i . Since i is arbitrary, Eq. 11 implies C = C ′ . Therefore, equality in distribution of the target count vector ˜ N implies the underlying multinomial parameter vectors must coincide: Cq = Cq ′ . Under the standing assumption that C is invertible, it gives q = q ′ . Hence, the mapping

<!-- formula-not-decoded -->

is injective under the stated positivity and invertibility conditions.

Proof for Lemma 2. The prior on ( q , C ) is the push-forward measure of a product Gaussian:

<!-- formula-not-decoded -->

under the smooth, one-to-one map

<!-- formula-not-decoded -->

where ψ : R K → ∆ K -1 is the softmax map. Here, injectivity holds because the log-odds representation is unique once the sum-zero constraint is imposed. Because L † is positive definite on the subspace

<!-- formula-not-decoded -->

the Gaussian distributions in Eq. 12 have everywhere positive Lebesgue densities on that subspace. Formally, for the target prior block

<!-- formula-not-decoded -->

where det ⋆ is the product of the positive eigenvalues. Since τ q L is non-singular on H , f q ( θ ) &gt; 0 for every θ ∈ H . Analogous positivity holds for each f C ,i ( ϕ i ) .

The softmax ψ is C ∞ with non-zero Jacobian everywhere on H . Therefore, T is a C ∞ diffeomorphism between

<!-- formula-not-decoded -->

with the Jacobian determinant is ∏ i q -1 i (1 -∑ i q i ) = · · · and hence non-zero, preserving positivity. Hence T preserves positivity of densities, the image measure Π on S posesses a density

<!-- formula-not-decoded -->

with respect to the product Lebesgue measure on simplices. Since each factor is positive everywhere, π ( q , C ) &gt; 0 for every ( q , C ) ∈ S . Because ( q 0 , C 0 ) lies in the interior of S and π is continuous and strictly positive, there exists

<!-- formula-not-decoded -->

Note that the minimum exists by compactness of the closed ϵ -ball. Furthermore, the Euclidean volume of the ball, under the ambient dimension dim∆ K -1 = K -1 for q and K ( K -1) for C , is finite and strictly positive:

<!-- formula-not-decoded -->

where c d is the volume if the unit ball in R d . Hence,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This concludes the proof.

Proof for Proposition 1. For every η &gt; 0 ,

<!-- formula-not-decoded -->

where P ( q , C ) denotes the K ( K +1) -dimensional multinomial law of the complete data. Because the parameter space is finite-dimensional and smooth,

<!-- formula-not-decoded -->

with a deterministic radius δ ( η ) → 0 as η ↓ 0 . This follows from a second-order Taylor expansion of the multinomial log-likelihood around the interior point. Lemma 2 says that the prior puts strictly positive mass on every Euclidean ball and hence on U η :

<!-- formula-not-decoded -->

Let θ = ( q , C ) and θ 0 = ( q 0 , C 0 ) . For any fixed ϵ &gt; 0 , define the alternative set B ϵ above. Because the model is an i.i.d. exponential family (multinomials) of finite dimension, Ghosal et al. [20] show there exist tests φ N : data →{ 0 , 1 } satisfying, for some constants c 1 , c 2 &gt; 0 independent of N ,

<!-- formula-not-decoded -->

Write the complete log-likelihood ratio as

<!-- formula-not-decoded -->

Let φ N = 1 { l N ( θ 0 ) &lt; -1 2 cN } , where c ∈ (0 , c ∗ ) and c ∗ is the minimized KL-divergence over B ϵ :

<!-- formula-not-decoded -->

The above inequality is strict by Lemma 1. Chernoff bounds for sums of bounded log-likelihood ratios gives (i) in Eq. 13. Under any θ ∈ B ϵ , the expected log-likelihood ratio equals -N · D KL [ P θ 0 ∥ P θ ] ≤ -c ∗ N . A one-sided Hoeffding inequality for sums of independent, bounded random variables then yields (ii) in Eq. 13, with c 2 = ( c ∗ -c ) / 2 .

Therefore,

<!-- formula-not-decoded -->

Because the metric ∥ ( q , C ) -( q 0 , C 0 ) ∥ is continuous, this convergence in outer probability equals convergence in probability, completing the proof.

Proof for Theorem 1. Put the log-odds vector for the target prior

<!-- formula-not-decoded -->

and for the i -th column of the confusion matrix

<!-- formula-not-decoded -->

Each vector lives in the centered linear subspace H := { v ∈ R K : v ⊤ 1 K = 0 } of dimension K -1 . Define the global parameter

<!-- formula-not-decoded -->

The map Ψ : η ↦→ ( q , C ) given by component-wise softmax is C ∞ and has a Jacobian of full rank everywhere on R d . Hence, η ↦→ ( q , C ) is a local diffeomorphism. We fix η 0 corresponding to ( q 0 , C 0 ) . Let

<!-- formula-not-decoded -->

where C and q in the right-hand side are the softmax images of η . Because each count is bounded by N and the mapping Ψ is smooth, ℓ N ( η ) is twice countinuously differentiable in a neighbourhood of η 0 .

Compute the score vector and observed information:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where F ( η 0 ) is the Fisher information matrix of dimension d × d . Here, F ( η 0 ) is positive definite because ( q 0 , C 0 ) is interior and C 0 is invertible (ensures different parameters yield different probability mass functions). The positive definiteness can be checked by observing that the Fisher information of a finite multinomial family with parameters inside the simplex is positive definite and the Jacobian of Ψ is full rank.

Set local parameter h = N ( η -η 0 ) . Asecond-order Taylor expansion yields the Local Asymptotic Normality (LAN) representation

√

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

The convergence of ∆ N uses the multivariate central limit theorem for sums of independent bounded variables. Uniform control of r N follows from third derivative boundedness in a neighborhood of η 0 .

In η -coordinates, the Laplacian-Gaussian prior has a density

<!-- formula-not-decoded -->

where φ is strictly positive and smooth Gamma density. Because L is positive semi-definite on H , π is continuous and strictly positive in a neighborhood of η 0 . Therefore, there exsit c 0 &gt; 0 and δ &gt; 0 such that

<!-- formula-not-decoded -->

Consequently, under the true distribution

<!-- formula-not-decoded -->

and for any M &gt; 0 ,

Hence,

<!-- formula-not-decoded -->

Because M &gt; 0 is arbitrary, this limit verifies the claim of the theorem.

Proof for Corollary 1. Consider the centred log-odds vector θ ∈ H for the target prior. Its conditional posterior density is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Take any point θ ∗ in a O ( N -1 / 2 ) -ball around θ 0 . By Theorem 1 this contains essentially all posterior mass. Inside that ball Taylor expansion gives

<!-- formula-not-decoded -->

with F the Fisher information matrix. Hence the negative Hessian of the log-posterior satisfies

<!-- formula-not-decoded -->

For N beyond some N 0 , the error term is dominated by N F , so there exists c 0 &gt; 0 such that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Because Ψ is C 1 with Jacobian D Ψ( η 0 ) of full rank, there exists a constant K 0 &gt; 0 such that, for all η in a neighborhood of η 0 ,

<!-- formula-not-decoded -->

Thus, where

Set the estimation error

<!-- formula-not-decoded -->

Because L has eigen-pair (0 , 1 ) and eigenvalues λ k ≥ λ 2 ( L ) on H , the restricted inverse satisfies

<!-- formula-not-decoded -->

For N ≥ N 0 , the second term dominates 1 / ( τ q λ 2 ( L )) , and hence

<!-- formula-not-decoded -->

The softmax map σ : θ ↦→ q has derivative

<!-- formula-not-decoded -->

Every entry of Dσ is bounded by 1 / 4 (attained at uniform prior), hence the operator norm obeys ∥ Dσ ∥ ≤ 1 / 2 . For any random vector θ with covariance Σ ,

<!-- formula-not-decoded -->

Applying to the posterior distribution with bound 15 gives

<!-- formula-not-decoded -->

The constant c 0 is λ min ( F ) which is independent of the graph but positive. Tightening Eq. 14 using the τ q L , term, we keep only the τ q λ 2 ( L ) contribution:

<!-- formula-not-decoded -->

Repeating the delta-method argument with this sharper bound scales Eq. 16 by the same 1 /λ 2 ( L ) . Set C = K ( τ q + c 0 ) / 2 τ q c 0 . Then,

<!-- formula-not-decoded -->

which is precisely the claimed finite-sample variance control.

Proof for Proposition 2. Write the complete-data log-posterior (ignoring normalising constants)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let ˆ θ N be the MAP with respect to the misspecified prior, and it satisfies the score equation

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taylor expansion of the gradient in Eq. 17 yields

<!-- formula-not-decoded -->

where R N = O P ( ∥ ∆ N ∥ 2 ) because the third derivative of ℓ ( q ) N is bounded on a neighbourhood of θ 0 . Using E [ ˜ N j ] = n ′ ( C 0 q 0 ) j and by the Markov inequality,

<!-- formula-not-decoded -->

For quadratic priors and log-concave likelihoods the posterior is asymptotically normal, so the difference between posterior mean and MAP is O ( N -1 ) [52]. Therefore

<!-- formula-not-decoded -->

This is the bound in Eq. 6 in the proposition.

The sub-space H = { v ∈ R K : v ⊤ 1 K = 0 } is the ( K -1) -dimensional hyper-plane of vectors whose components sum to zero. Its orthogonal projection operator is the K × K symmetric matrix P H = I K -1 K 1 K 1 ⊤ K , because for any v ∈ R K , P H v = v -( 1 K 1 ⊤ K v ) 1 K , and the subtracted term is exactly the scalar mean of v replicated in every coordinate, making the result mean-zero. Because F 0 ⪰ 0 and L ⪰ λ 2 ( L ) P H , the smallest eigenvalue of N F 0 + τ q L is at least Nλ min ( F 0 )+ τ q λ 2 ( L ) . Hence,

<!-- formula-not-decoded -->

Combine them to get inequality in Eq. 7, completing the proof.

Proof of Theorem 2. Throughout the proof we fix an arbitrary interior point q ∈ ˚ ∆ K -1 and an arbitrary tangent direction v ∈ T q ∆ K -1 = { v ∈ R K | 1 ⊤ v = 0 } . To establish geodesic convexity it suffices to show

<!-- formula-not-decoded -->

because the latter is the Rayleigh quotient form of the desired positive-definite bound. Recall the form of the negative log-posterior (constant terms omitted):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Write r ( q ) := Cq , r i ( q ) = ∑ j C ij q j . For the multinomial log-likelihood term we have the usual identity

<!-- formula-not-decoded -->

Indeed, start with F lik ( q ) = ∑ i ˆ r i log ˆ r i r i ( q ) and differentiate twice with respect to q j while keeping the tangent constraint 1 ⊤ v = 0 , and we have Eq. (20). Now evaluate the quadratic form:

<!-- formula-not-decoded -->

To relate the Euclidean norm ∥ · ∥ 2 with the Fisher metric g q notice that

<!-- formula-not-decoded -->

Hence for every v ,

<!-- formula-not-decoded -->

Accordingly

because q i &lt; 1 inside the simplex. Thus, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where J ( q ) ∈ R K × ( K -1) is the Jacobian ∂ θ /∂ q restricted to the tangent space. Concretely

<!-- formula-not-decoded -->

with P T = I -1 K 11 ⊤ the projection onto T q ∆ K -1 .

Applying Eq. (24) we expand the quadratic form:

<!-- formula-not-decoded -->

Because J ( q ) ⊤ J ( q ) = diag( q ) -1 P T one checks

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, and

<!-- formula-not-decoded -->

Since the inequality holds for every v ∈ T q ∆ K -1 , the matrix inequality announced in the theorem follows, and the strong-convexity constant is

<!-- formula-not-decoded -->

Proof for Proposition 3. Write F lik ( q ) = n ′ ∑ K i =1 ˆ r i log ˆ r i r i ( q ) with r i ( q ) = ∑ j C ij q j . For j ∈ { 1 , . . . , K } we differentiate:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so that in vector form

Recall θ = ( θ i ) with θ i = log q i -1 K ∑ K k =1 log q k . Let s ( q ) = 1 K ∑ k log q k . For j :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In vector notation

<!-- formula-not-decoded -->

and because P T annihilates the 1 -component we may drop it inside the tangent space: P T L = L since 1 is the null-eigenvector of L . On the simplex g -1 q acts as left-multiplication by diag( q ) (followed by projection onto T q ∆ K -1 , automatic here because every gradient we computed is already centered). Hence

<!-- formula-not-decoded -->

which is exactly (9). The rightmost expression is automatically orthogonal to 1 because each bracketed term sums to 0 ; therefore the evolution does not leave the simplex:

<!-- formula-not-decoded -->

Finally, inserting (9) in the natural gradient flow yields the replicator part ˙ q t,j = -q t,j [ n ′ [ C ⊤ ( 1 -ˆ r / r )] + τ q [ Lθ t ] ] j , confirming the announced replicator-Laplacian dynamics.

## B Background on Information Geometry

Table 4 summarizes the basic notations of information geometry required in our study, to make the manuscript self-contained. See textbooks in information geometry for more details [5, 6].

Table 4: Basic notations of information geometry.

| Concept                         | Definition                                                                                                                          |
|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Fisher-Rao metric               | g q ( v , w ) = ∑ i v i w i q i on the open simplex.                                                                                |
| e -coordinates                  | θ i = log q i - 1 K ∑ j log q j (exponential-family / natural parameters constrained to T q ∆ K - 1 ).                              |
| m -coordinates                  | The usual probabilities q i (mixture parameters).                                                                                   |
| e - / m -flat sub-manifold      | A subset whose image is affine in the corresponding coordi- nates; e.g. M = Cq is m-flat because r = Cq is linear in q .            |
| Dual flatness, potentials       | State ψ ( q ) = ∑ i q i log q i and φ ( θ ) = log (∑ i e θ i ) satisfy- ing q = ∇ θ φ and θ = ∇ q ψ .                               |
| e - / m -projection             | For a point p and sub-manifold S , the minimizer of D KL ( p &#124; s ) (m-projection) or D KL ( s &#124; p ) (e-projection).       |
| e - / m -convex function        | A function whose restriction to every e -(resp. m -) geodesic is convex in the ordinary sense.                                      |
| Generalized Pythagorean theorem | For p , q , r where q is the m -projection of p onto S and r ∈ S , D KL [ p &#124; r ] = D KL [ p &#124; q ]+ D KL [ q &#124; r ] . |
| Natural gradient                | grad FR F = g - 1 q ∇ ! q F .                                                                                                       |

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We summarized our contributions, referring the corresponding sections.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations are discussed in the conclusion section.

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

Answer: [Yes]

Justification: Full proofs for all statements are provided in the appendix.

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

Justification: Full experimental protocol is described in the experiments section.

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

Justification: The codes for numerical experiments are submitted as the supplemental material.

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

Justification: Full experimental protocol is described in experiments section.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All results are reported with standard error.

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

Justification: The computing resource is described in experiments section.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The authors reviewed the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This work is a foundational research.

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

Justification: The paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All required libraries and resources are correctly cited.

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

Justification: The paper does not release new assets.

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

## Answer: [NA]

Justification: The core method development in this research does not involve LLMs

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.