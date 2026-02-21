## CAMO: Convergence-Aware Multi-Fidelity Bayesian Optimization ∗

Wei W. Xing

University of Sheffield w.xing@sheffield.ac.uk

## Zhenjie Lu †

SUSTeh &amp; Shenzhen University 12432976@mail.sustech.edu.cn

## Abstract

Existing Multi-fidelity Bayesian Optimization (MFBO) methods ignore the convergence behavior of the multi-fidelity surrogate as the fidelity increases, leading to inefficient exploration and suboptimal performance. We introduce CAMO (Convergence-Aware Multi-fidelity Optimization), a principled framework based on Linear Fidelity Differential Equations (LFiDEs) that explicitly encodes convergence of fidelity-indexed outputs and employs a closed-form nonstationary kernel. We rigorously prove the existence and pointwise/uniform convergence to the high fidelity surrogate under mild restrictions and provide new convergence results for general FiDEs using smooth, non-smooth and even non-convex Lyapunov functions, establishing a bridge between MFBO and the theory of subgradient flows in non-smooth optimization theory. Combined with a fidelity-aware acquisition function, CAMO outperforms state-of-the-art MFBO methods on a majority of synthetic and real-world benchmarks, with up to a four-fold improvement in optimization performance and a dramatic speed-up in convergence. CAMO offers a tractable and theoretically grounded approach to convergence-aware MFBO.

## 1 Introduction

Bayesian optimization (BO) is an efficient method for optimising costly black-box functions, finding applications in various domains of engineering and science [1]. BO leverages a surrogate model, typically a Gaussian Process (GP), to approximate the objective function and guide the search towards promising regions of parameter space using an acquisition function. This approach is particularly effective when the black-box function evaluations are costly or time-consuming. In many applications, black-box solutions can be obtained at varying levels of fidelity. Lower-fidelity solutions are associated with lower accuracy and lower computational costs, the latter of which typically increase dramatically as fidelity increases. The fidelity can be controlled in a number of ways, by varying the modelling choices or the solver settings. For example, electronic-structure methods involve increasing levels of theory, with exponentially increasing costs [2]. Different fidelities can also be defined via the numerical formulation, e.g., use different mesh sizes, time steps or convergence thresholds. For the purposes of prediction, and especially for optimization, it is of course desirable and sometimes essential to predict at the highest fidelity. In practice, however, the number of high-fidelity results obtainable within a typical computational budget or time frame is often constrained.

Multi-Fidelity Bayesian optimization (MFBO) methods aim to reduce the cost of optimising highfidelity black-box functions by leveraging lower-fidelity approximations [3, 4]. In typical scenarios, the multi-fidelity objective evolves towards the highest fidelity function as the fidelity parameter increases. Existing MFBO approaches ignore this evolution and treat the fidelity as an unstructured input, which can lead to inefficient exploration in high-cost regions and, ultimately, suboptimal

∗ Code is available at https://github.com/IceLab-X/CAMO . † The majority of this work was conducted while Zhenjie Lu was at Shenzhen University. ‡ Corresponding author

Akeel A. Shah ‡

Chongqing University akeelshah@cqu.edu.cn

performance. In this paper we propose CAMO, a principled framework for MFBO that uses a convergence-aware surrogate model derived from linear fidelity differential equations (LiFiDEs). The 'convergence' here refers to a systematic progression towards a high-fidelity function. In contrast to conventional methods, CAMO structurally enforces convergence as fidelity increases, ensuring that predictions at lower fidelities provably approach the high-fidelity objective under mild regularity assumptions. Our key contributions are: (1) a convergence-aware surrogate that encodes the structural tendency of simulations to converge as fidelity increases; (2) a closed form non-stationary LiFiDE kernel that captures fidelity-wise correlations, eliminating the need to specify a relationship between fidelities a-priori (CAMO adaptively learns convergence behaviour from the data); (3) a general theory for fidelity-indexed systems using Lyapunov functions, establishing uniform and pointwise convergence under smooth, non-smooth and even non-convex scenarios, and revealing a formal link to subgradient-based dynamics in modern non-smooth optimization theory; (4) when combined with BOCA (Bayesian optimization with Continuous Approximations) [3] to leverage both fidelity-aware exploration and convergence-aware modelling we show that CAMO consistently outperforms MFBO baselines on synthetic and real-world benchmarks.

## 2 Related Work

Bayesian optimization (BO) [5] uses a surrogate model, typically a Gaussian Process (GP) [6] (see Appendix A), to approximate the objective function, together with an acquisition function to guide the search towards promising regions. Acquisition functions balance exploration and exploitation to efficiently search for a global optimum. Popular choices include the Expected Improvement (EI) [7], the Upper Confidence Bound (UCB) [8], and the Probability of Improvement (PI) [9]. MFBO extends standard BO by utilising solutions at different fidelity levels, aiming to reduce the overall optimization cost and improve convergence speed. Most of the work in MFBO has focused on discrete fidelity sets. Huang et al. [10] proposed Sequential Kriging optimization (SKO), which employs a hierarchical GP approach to capture the correlations between different fidelity levels. Kandasamy et al. [11] developed the Multi-Fidelity (MF) Gaussian Process Upper Confidence Bound (MF-GP-UCB) method based on a generic GP formulation. Le Gratiet and Garnier [12] proposed Recursive Co-Kriging, which builds a hierarchy of GP models of the residual between successive fidelity levels. Perdikaris et al. [13] used low-fidelity information as an input to a high-fidelity GP, leading to a deep GP structure, while Cutajar et al. [14] directly employed a deep GP to learn nonlinear mappings between fidelities.

In many real-world applications, the fidelity level can be treated as a continuous variable (e.g., a mesh size or time step). This has led to the development of continuous-fidelity MFBO methods such as BOCA [3], which extends MF-GP-UCB by using a two-step procedure to select the next query point. Poloczek et al. [15] introduced the Multi-Information Source optimization (MISO) framework, which treats the objective function as a linear combination of GPs, each corresponding to a different fidelity level. Klein et al. [16] proposed FABOLAS, combining a GP with a linear fidelity kernel and an Entropy Search acquisition function. Wu and Frazier [17] developed the continuous fidelity knowledge gradient (cfKG) method based on a GP with a knowledge gradient acquisition function. In contrast, Li et al. [18] took a deep learning approach with Deep Neural Network MF Bayesian optimization (DNN-MFBO), which uses a fidelity-wise Gauss-Hermite quadrature and a moment-matching mutual information acquisition function. These studies illustrated the benefits of continuous-fidelity MFBO over discrete formulations. Most practical GP-based methods, however, lack insights into the convergence behaviour of the objective function across fidelity levels, leading to the over-exploration of costly regions and, therefore, suboptimal performance. Although DNNs can in theory capture the complex relationships between fidelity levels, observations at high fidelity are typically sparse, leading to overfitting and high model variance. Recent advances in MF modelling include fidelity differential equations (FiDE) [19], modelled by NeuralODE [20] or continuous autoregression (CAR) [19] to capture the convergence behaviour of the objective function. The training times for NeuralODE are up to 10 4 -fold higher than those for CAR. In this work we therefore build upon the FiDE concept to propose a tractable and scalable MFBO method.

Hyperparameter optimization (HPO) and neural architecture search (NAS) are not considered core applications of CAMO, which is focused on expensive blackbox solvers in science and engineering. In such applications, continuous fidelity indices are unambiguously defined. They arise naturally as controllable, continuous parameters intrinsic to the physical model or numerical approximation. Their effect on the objective function is deterministic, with typically a monotonic convergence toward the highest fidelity. In contrast, HPO/NAS operate in a setting in which fidelity is stochastic and algorithm-dependent, e.g., epoch number and budget are algorithmic artefacts with no deterministic

convergence law or trajectory. The cross-fidelity structure is non-smooth and data-dependent, which requires empirical (ideally discrete-fidelity) MFBO methods such as BOHB [21], DyHPO [22] and FastBO [23]. Adapting CAMO to HPO/NAS would potentially require reformulating the underlying convergence model as a stochastic process, which is orthogonal to the contributions of this paper.

## 3 Background

## 3.1 Problem Formulation

We consider the problem of optimising a function y : X × [0 , T ) → R , ( x , t ) ↦→ y ( x , t ) , in which X ⊆ R d is a design variable space, and [0 , T ) ⊂ R is a range of fidelity levels. T = ∞ is allowed and is of particular interest. We assume that both the accuracy and computational costs of evaluating y ( x, t ) increases with t . The goal is to find the design variable x ∗ = arg max x ∈X y ( x , T ) that maximises the high-fidelity objective while minimising the total cost of evaluations. The observations y i at different x and t incorporate additive Gaussian noise (see Appendix A). L p ( X ) ( 1 ≤ p &lt; ∞ ) denotes the space of functions f : X → R such that ∥ f ∥ p L p ( X ) = ∫ X | f ( x ) | p d x &lt; ∞ , where d x denotes integration w.r.t. Lebesgue measure. C ( X ) denotes the space of continuous functions f : X → R . C a,b ( X × [0 , ∞ )) denotes the space of functions f : X × [0 , ∞ ) → R with continuous derivatives up or orders a and b in x and t , with similar notation for functions of more than two variables (the superscript is omitted if a = b = 0 ). ∥ f ∥ L ∞ ( A ) = ess sup x ∈X f ( x ) denotes the L ∞ norm. AC ([0 , ∞ )) denotes the space of absolutely continuous functions and L 1 loc ([0 , ∞ )) is the space of locally integrable functions on [0 , ∞ ) .

## 4 CAMOand Theoretical Results

## 4.1 Multi-Fidelity via FiDE: Well-Posedness and Link to Subgradient Flows

MFBO requires a surrogate to model y ( x , t ) , often in the form of a GP: y ( x , t ) ∼ GP (0 , k ( x , t, x ′ , t ′ )) with some kernel k . Li et al. [20] and Xing et al. [19] proposed that y ( x , t ) can be modelled as the solution to a Fidelity Differential Equation (FiDE), namely

<!-- formula-not-decoded -->

subject to some initial condition y ( x , 0) = y 0 ( x ) . Here, ϕ ( x , t, y ) describes the system 'dynamics', with the fidelity parameter t playing the role of physical time. Without loss of generality, we set T = ∞ for the theoretical results. Treating the MF system as a dynamical system is entirely natural since the index t is strictly ordered. To conduct MFBO with such a model we need guarantees on the well-posedness of the FiDE problem (1), i.e, that unique solutions exist and that y ( x , t ) converges to a unique equilibrium solution y ∞ ( x ) = lim t →∞ y ( x , t ) . The convergence can be interpreted in a pathwise sense, e.g., for each fixed input x , the solution trajectory converges deterministically to the high-fidelity target. In this work we do not consider stochastic FiDEs, i.e., convergence in stochastic process topologies. The dynamics are entirely deterministic once the training data are fixed.

Convergence can be studied using a Lyapunov analysis. A Lyapunov function V : ( x , t, y ) ↦→ V ( x , t, y ) (dependent on ϕ ) serves as a energy-like functional that measures the deviation of y ( x , t ) from an equilibrium. Our first result establishes minimal conditions for the existence and uniqueness of classical solutions pointwise in x (see Appendix B.1 for the proof)

Lemma 1. [Existence and Regularity for General FiDEs] Let X ⊆ R d be a non-empty set. Consider the system (1) and assume that: (1) for each fixed x ∈ X , the function ( t, y ) ↦→ ϕ ( x , t, y ) is continuous on [0 , ∞ ) × R ; (2) for each fixed x ∈ X , ϕ ( x , t, y ) is locally Lipschitz in y , uniformly for t in compact subsets of [0 , ∞ ) ; (3) the initial condition y 0 ( x ) ∈ R is finite for each x . Then for each fixed x ∈ X , there exists a unique maximal solution y ( x , t ) ∈ C 1 ([0 , t ∗ ( x ))) defined on some maximal interval [0 , t ∗ ( x )) , with 0 &lt; t ∗ ( x ) ≤ ∞ . Moreover: (a) if y ( x , t ) remains bounded for all t ∈ [0 , ∞ ) , then t ∗ ( x ) = ∞ and the solution exists globally on [0 , ∞ ) ; (b) if ϕ is globally Lipschitz in y , uniformly in t , then the solution exists globally on [0 , ∞ ) .

The following Lemmas (see Appendix B.2 for the proofs) establish minimal regularity requirements on V , y 0 and X for pointwise and uniform (in x ) convergence to hold, respectively.

Lemma 2 (Pointwise Convergence for FiDE) . Let y ( x , · ) ∈ C 1 ([0 , ∞ )) , x ∈ X be a solution to (1) with y 0 ( x ) &lt; ∞ for each x ∈ X . Suppose there exists a Lyapunov function V : X × [0 , ∞ ) × R → R , ( x , t, y ) ↦→ V ( x , t, y ) , and that the following conditions are satisfied: (1) V ∈ C 0 , 1 , 1 ( X × [0 , ∞ ) × R ) ; (2) V ( x , t, y ) ≥ 0 , ∀ ( x , t, y ) ∈ X × [0 , ∞ ) × R and V ( x , t, y ) = 0 iff y = y ∞ ( x ) ; (3) for all

( x , t ) ∈ X × [0 , ∞ )

<!-- formula-not-decoded -->

Then y ( x , t ) → y ∞ ( x ) pointwise on X , i.e., lim t →∞ y ( x , t ) = y ∞ ( x ) for each x ∈ X .

Lemma 3 (Uniform Convergence for FiDE) . Let y ( x , · ) ∈ C 1 ([0 , ∞ )) , x ∈ X is a solution to (1). Suppose there exists a Lyapunov function V : X × [0 , ∞ ) × R → R , ( x , t, y ) ↦→ V ( x , t, y ) and the following conditions are satisfied: (1) y 0 ( x ) ∈ C ( X ) ; (2) V ∈ C 0 , 1 , 2 ( X × [0 , ∞ ) × R ) ; (3) V ( x , t, y ) ≥ 0 , ∀ ( x , t, y ) ∈ X × [0 , ∞ ) × R and V ( x , t, y ) = 0 iff y = y ∞ ( x ) ; (4) ∃ c min &gt; 0 such that c ( x , t ) ≥ c min &gt; 0 , ∀ ( x , t ) ∈ X × [0 , ∞ ) , in which c ( x , t ) = 1 2 ∂ 2 yy V ( x , t, y ∞ ( x )) ; (5) for all x , t ∈ X × [0 , ∞ ) , ∃ α &gt; 0 such that

<!-- formula-not-decoded -->

(6) ∃ V max such that ∥ V ( x , 0 , y ( x , 0)) ∥ L ∞ ( X ) ≤ V max . Then lim t →∞ y ( x , t ) = y ∞ ( x ) uniformly on X , with

<!-- formula-not-decoded -->

If y ( x , t ) ∈ C ( X × [0 , ∞ )) then y ∞ ∈ C ( X ) since the uniform limit of continuous functions is continuous. The uniform-convergence conditions are readily satisfied for functions V that have at most a mild dependence on t , such as exponential decay or bounded variations, notably quadratic functions V ( x , t, y ) = a ( x , t )( y -y ∞ ( x )) 2 , a ( x , t ) ≥ a 0 &gt; 0 uniformly. Non-smooth Lyapunov functions such as V |·| = | y ( x, t ) -y ∞ ( x ) | permit a convergence analysis even if y ( x , t ) exhibits non-smooth behaviour due to noise or sparse evaluations at low fidelities (in which case ∂ t y has to be interpreted in a distributional sense). Non-smooth V allow for the certification of asymptotic behaviour and exponential convergence in such non-smooth settings. Lemma B1 in Appendix B.3 establishes uniform convergence for V |·| ( x , t, y ) , which is readily extended to Elastic-Net-type and Group-Sparsity-type V . In fact, the result extends to non-convex Lyapunov functions via the Clarke subdifferential [24] (see Appendix B.3 for a proof)

Lemma 4. Let X be a nonempty set, and let y : X × [0 , ∞ ) → R be such that for each x ∈ X , y ( x , · ) ∈ C 1 [0 , ∞ ) . Let V : X × R → R , ( x , y ) → V ( x , y ( x , t )) satisfy the following conditions. (1) For every fixed ( x , t ) , the map y ↦→ V ( x , y, t ) is locally Lipschitz. (2) For each fixed ( x , t ) ∈ X × [0 , ∞ ) , define u [ x ]( · ) : R → R , y ↦→ V ( x , y ) , and for a fixed x ∈ X , define v [ x ]( · ) : [0 , ∞ ) → R , t ↦→ V ( x , y ( x , t )) ; assume that for all ( x , t ) ∈ X × [0 , ∞ ) and all ξ ( x , t ) ∈ ∂ C u [ x ]( y ) the inequality ξ ( x , t ) ϕ ( x , t, y ( x , t )) ≤ -αv [ x ]( t ) holds for some constant α &gt; 0 , independent of x and t . (3) The initial deviation is bounded, i.e., ∥ y ( · , 0) -y ∞ ( · ) ∥ L ∞ ( X ) &lt; ∞ . Then, for all t ∈ [0 , ∞ ) , we have

<!-- formula-not-decoded -->

These lemmas reveal a close analogy between the convergence of FiDEs and subgradient-based dynamics in non-smooth convex optimization. For a non-smooth Lyapunov function V , the evolution of y ( x , t ) can be analysed using differential inclusions involving the subdifferential ∂ y V . This structure mirrors subgradient flows and proximal-map methods commonly encountered in sparse learning and variational optimization. Under mild conditions, such dynamics guarantee exponential convergence toward the high-fidelity target y ∞ ( x ) , with y ( x , t ) following a descent trajectory that reduces the fidelity error encoded by V . Quantitatively, subgradient dynamics induced by non-smooth V can be interpreted as the limit of smooth gradient flows applied to a Moreau-Yosida regularisation [25]. For instance, define the smoothed functional

<!-- formula-not-decoded -->

as a regularized approximation of V ( x , y ) = | y -y ∞ ( x ) | . The associated gradient flow ∂ t y λ ( x , t ) = -∇ y V λ ( x , y λ ( x , t ) converges (in the sense of graphical or Mosco convergence) to the subgradient inclusion ∂ t y ( x , t ) ∈ -∂ | y ( x , t ) -y ∞ ( x ) | as λ → 0 . Although FiDEs do not explicitly follow this gradient flow, the analogy provides a useful interpretation: the FiDE-induced surrogate evolution resembles a continuoust analogue of proximal or subgradient descent, where Moreau smoothing offers stability while preserving convergence guarantees. We refer to Appendix B.4 for further discussion.

## 4.2 Linear FiDEs and Convergence Guarantees

While Lemmas 2-4 provide a robust and general framework for ensuring convergence, finding a suitable Lyapunov function for a specific ϕ is complicated by the fact that the latter is generally unknown and at the very least is difficult to model. Li et al. [20] resort to learning ϕ and y 0 ( x ) using neural networks. In contrast, Xing et al. [19] considered a more tractable case in which ϕ admits a linear form in y . In this paper we adopt the same approach, first establishing the existence of a unique equilibrium and the validity of a constructive variation-of-constants approach. If ϕ admits a linear form in y , we obtain the following linear FiDE for each fixed x ∈ X

<!-- formula-not-decoded -->

subject to some y 0 ( x ) = y ( x , 0) ∈ R (i.e., finite for each x ). This is a first-order linear, nonautonomous ODE in t . We first establish the global existence of unique solutions and validity of the variation-of-constants formula pointwise in x ∈ X (see Appendix C.1 for a proof).

Lemma 5. (Existence and Uniqueness for Linear FiDE) Let X ⊂ R d be given and suppose that: (A1) β ( x , · ) , u ( x , · ) ∈ C ([0 , ∞ )) ; and (A2) y 0 ( x ) ∈ R . Then there exists a maximal time t ∗ ( x ) such that a unique local solution y ( x , · ) ∈ C 1 ([0 , t ∗ ( x ))) satisfying (7) exists and is given by

<!-- formula-not-decoded -->

Moreover, either t ∗ ( x ) = ∞ or lim sup t → t ∗ ( x ) -| y ( x , t ) | = ∞ (finite-time blowup). If in addition: (A3) ∥ β ( x , t ) ∥ L ∞ ( X× [0 , ∞ )) &lt; ∞ and ∥ u ( x , t ) ∥ L ∞ ( X× [0 , ∞ )) &lt; ∞ , then t ∗ ( x ) = ∞ . Now let X ⊂ R d be compact and suppose instead that: (B1) β ( x , t ) , u ( x , t ) ∈ C ( X × [0 , ∞ )) ; (B2) y 0 ( x ) ∈ C ( X ) ; and (A3) above. Then there exists a unique solution y ( x , t ) ∈ C ( X × [0 , ∞ )) , y ( x , · ) ∈ C 1 ([0 , ∞ )) of (7) given by (8).

Having established the conditions for existence, the following two lemmas guarantee pointwise or uniform convergence of y ( x , t ) on X towards an equilibrium under some general conditions on X , y 0 ( x ) , β ( x , t ) and u ( x , t ) (see Appendices C.2 and C.3 for proofs).

Theorem 1 (Linear FiDE Pointwise convergence) . Let y ( x , t ) satisfy the LiFiDE (7) for each fixed x ∈ X and assume: (1) for each x ∈ X , β ( x , · ) ∈ C ([0 , ∞ )) ∩ L 1 loc [0 , ∞ ) , β ( x , t ) &lt; 0 , ∀ ( x , t ) ∈ X × [0 , ∞ ) and ∃ β ∗ ( x ) ∈ R such that lim t →∞ β ( x , t ) = β ∗ ( x ) &lt; 0 ; (2) for each x ∈ X , u ( x , · ) ∈ C ([0 , ∞ )) and lim t →∞ u ( x , t ) = u ∗ ( x ) ∈ R ; (3) ∥ u ( x , t ) ∥ L ∞ ( X× [0 , ∞ )) = M &lt; ∞ and ∥ β ( x , t ) ∥ L ∞ ( X× [0 , ∞ )) = λ &lt; ∞ . Then

<!-- formula-not-decoded -->

The 'long-time' behavior of the solution y ( x , t ) is governed by the limiting values of the coefficients β ( x , t ) and u ( x , t ) . As t →∞ , the original non-autonomous equation (7) effectively behaves like the constant-coefficient equation

<!-- formula-not-decoded -->

which has the unique equilibrium point y ∞ ( x ) = -u ∗ ( x ) /β ∗ ( x ) . Theorem 1 shows that the solution y ( x , t ) does indeed tend to this equilibrium as t →∞ for each fixed x ∈ X .

Theorem 2 (Linear FiDE Uniform Convergence) . Let y ( x , t ) satisfy (7). Assume: (1) y 0 ( x ) ∈ C ( X ) ; (2) β ( x , t ) ∈ C ( X× [0 , ∞ )) , β ( x , t ) &lt; 0 , ∀ ( x , t ) ∈ X× [0 , ∞ ) , lim t →∞ ∥ β ( x , t ) -β ∗ ( x ) ∥ L ∞ ( X ) = 0 , β ∗ ( x ) ∈ C ( X ) ; (3) u ( x , t ) ∈ C ( X × [0 , ∞ )) , lim t →∞ ∥ u ( x , t ) -u ∗ ( x ) ∥ L ∞ ( X ) = 0 , u ∗ ( x ) ∈ C ( X ) ; (4) ∃ λ ′ &gt; 0 such that β ∗ ( x ) ≤ -λ ′ , ∀ x ∈ X . Then

<!-- formula-not-decoded -->

Remark 1 (Consistency of assumptions) . The conditions for convergence in Theorems 1 and 2 build naturally upon the conditions established in Lemma 5. In particular, the continuity assumptions β ( x , · ) , u ( x , · ) ∈ C ([0 , ∞ )) ensure the existence of a unique classical solution y ( x , t ) ∈ C 1 . The additional assumptions required for convergence (pointwise or uniform convergence of β ( x , t ) → β ∗ ( x ) and u ( x , t ) → u ∗ ( x ) ) serve to control the long-time asymptotics of the system. That is, while continuity ensures a solution exists, convergence of the coefficients guarantees that the solution stabilises to an equilibrium. Moreover, the boundedness conditions in Theorem 1 and uniformity in Theorem 2 (along with compactness of X ) allow us to lift the pointwise result to uniform convergence.

Examples of coefficients that satisfy the assumptions in Theorems 1 and 2 are as follows.

1. Exponential decay: if β ( x , t ) = β ∗ ( x ) + δ β ( x ) e -µt , u ( x , t ) = u ∗ ( x ) + δ u ( x ) e -µt with µ &gt; 0 and δ β , δ u ∈ C ( X ) , then uniform convergence holds.
2. Polynomial decay: slower convergence rates, e.g., with β ( x , t ) = β ∗ ( x ) + δ β ( x ) t -γ , γ &gt; 1 , will suffice for pointwise convergence, since integrability over [0 , ∞ ) is preserved.
3. Time-invariant case: when β ( x , t ) ≡ β ∗ ( x ) &lt; 0 , u ( x , t ) ≡ u ∗ ( x ) , the solution converges exponentially to the equilibrium y ∞ ( x ) = -u ∗ ( x ) /β ∗ ( x ) .
4. Bounded perturbations: β ( x , t ) = β ∗ ( x ) + ϵ ( x , t ) with bounded ϵ ( x , t ) , oscillatory or decaying. If ϵ ( x , t ) → 0 , the function converges to β ∗ ( x ) . If it does so uniformly, then Theorem 2 applies. If only pointwise convergence holds and ϵ ( x , · ) ∈ L 1 ([0 , ∞ )) , then Theorem 1 applies.

## 4.3 Tractable Autoregressive Gaussian Process Multi-Fidelity Surrogate

Based on the linear FiDE surrogate, we now introduce a tractable MF model that guarantees convergence at least pointwise while maintaining the probabilistic framework of GPs. Let (Ω , F , P ) be a probability space supporting two independent stochastic processes y 0 : X × Ω → R and u : X × [0 , ∞ ) × Ω → R . We place Gaussian process priors over both

<!-- formula-not-decoded -->

The joint law of ( y 0 , u ) is the product measure induced by P on Y 0 ×U , reflecting their independence. Under these assumptions we have the following result (see Appendix D).

Proposition 1. The linear model solution via the variation-of-constants formula

<!-- formula-not-decoded -->

is a zero-mean Gaussian process with covariance function k ( x , t, x ′ , t ′ ) = E ω ∼ P [ y ( x , t ; ω ) y ( x ′ , t ′ ; ω )] , given explicitly by

<!-- formula-not-decoded -->

The integrals in the second term in Eq. (14) would in general require numerical quadrature. However, they can be evaluated analytically for various forms of β ( x , t ) , e.g., a constant β ( x , t ) = -β with stationary kernels such as Matérn and periodic for k u ( x , t, x , t ′ ) (see Appendix E). We also assume that k u ( x , t, x , t ′ ) = k x ( x , x ′ ) k t ( t, t ′ ) , in which k t ( t, t ′ ) = exp ( -( t -t ′ ) 2 / 2 ℓ 2 ) , while k x and k 0 ( x , x ′ ) are kept arbitrary. This leads to the 'LiFiDE kernel'

<!-- formula-not-decoded -->

in which I ( t, t ′ ) = ( √ πℓ/ 2)( h ( t ′ , t ) + h ( t, t ′ )) , where h ( t ′ , t ) is given by [26]

<!-- formula-not-decoded -->

with erf ( · ) denoting the error function, α = βℓ/ √ 2 , τ = ( t -t ′ ) /ℓ , and ˆ τ = ( t + t ′ ) /ℓ . The kernel (15) is non-stationary. Its structure allows us to reduce the training time complexity of the MF surrogate from O (( ∑ f N f ) 3 ) to O ( ∑ f N 3 f ) , where N f is the number of fidelity f data points [19].

## 4.4 optimization Strategies Using Continuous Fidelity Acquisition Functions

There are several strategies for guiding the selection of the input x and fidelity t at each iteration. MFGP-UCB [11] uses fidelity-specific upper confidence bounds and selects query inputs by maximising the minimum of a UCB acquisition across discrete fidelities. BOCA [3] extends this method to continuous t , with the fidelity selected from a filtered set. cfKG [4] extends the KG acquisition function to continuous fidelities by selecting the point ( x , t ) that maximises the expected reduction in the high-fidelity posterior minimum. FABOLAS [16] is based on a cost-aware EI acquisition function, in which the fidelity is typically a proxy. It models the validation loss and cost jointly as GPs and selects ( x , t ) values that maximise improvement per cost. MF-DNN [18] approximates mutual information between the function optimum and observations using a variational approximation, enabling efficient acquisition in large neural network tuning problems.

While some acquisition functions can be re-used across MF models, many (e.g., BOCA and cfKG) are tightly coupled to specific fidelity models. In our experiments, BOCA consistently delivered the best empirical performance among the methods above. It also provides theoretical guarantees alongside practical performance, and an improved rate of convergence to the optimum compared to UCB. The original formulation considers the fidelity set T = [0 , 1] and input space X = [0 , 1] d , with y ( x , t ) ∼ GP (0 , κ ) , κ = κ t ( t, t ′ ) κ x ( x , x ′ ) . The key idea in BOCA is to exploit cheaper, approximate fidelities t &lt; 1 when they are sufficiently informative about the target function f ( x ) = y ( x , 1) . BOCA filters the fidelity set using a cost constraint and a minimum uncertainty condition. By concentrating expensive evaluations in a small, polynomially-dilated variant X ρ,n of a 'high-information' region X ρ , BOCA achieves tighter regret bounds than GP-UCB and faster convergence. Additionally, BOCA spends the majority of its evaluations on low-cost but informative fidelities, leading to improved capital ( Λ ) efficiency. The region X ρ depends on a 'fidelity gap' ξ ( t ) = √ 1 -κ t ( t, 1) 2 , which controls the tightness of the bound. The following is an informal version of the simple regret r result in terms of the mutual information γ n (see Appendix F for full details).

Theorem 3 (Kandasamy et al. [3]) . Let X = [0 , 1] d , T = [0 , 1] and y ( x , t ) ∼ GP (0 , κ ) , κ = κ t ( t, t ′ ) κ x ( x , x ′ ) . Choose δ ∈ (0 , 1) and execute BOCA with β n = O ( d log( n/δ )) . Then, for any α ∈ (0 , 1) , there exist ρ ( α ) &gt; 0 such that for Λ large enough, with probability at least 1 -δ

<!-- formula-not-decoded -->

For a fixed x , x ′ , the standard SE kernel, k SE enforces high correlation near the diagonal t ≈ t ′ , with O ( | t -t ′ | 2 ) decay that is independent of t . This implies all fidelities are equally smooth and informative. In practical scenarios, however, low-fidelity evaluations are often noisy or unstable. In contrast, the LiFiDE kernel variance k ( t, t ) increases with t , and k ( t, t ′ ) has O ( | t -t ′ | ) decay along the diagonal t ≈ t ′ , decreasing with t (see Appendix G). This models a convergent fidelity process: evaluations become more stable and informative as fidelity increases, reflecting a more realistic structure. For the SE kernel, ξ ( t ) = O ( | t -T | ) for | t -T | ≪ 1 for a maximum fidelity T . In contrast, for the LiFiDE kernel ξ ( t ) = O ( √ | t -T | ) (see Remark G1). Although this leads to looser regret guarantees via X ρ , the LiFiDE kernel offers stronger empirical performance due to its convergence-aware structure and because it does not concentrate as sharply at t → T , discarding potentially useful results at low fidelity. Indeed, the experimental results will show that CAMO conducts most of its exploration in low-fidelity regions, which reduces the query cost and leads to a fast convergence rate in the early stages of the optimization process, without sacrificing accuracy.

## 5 Experimental Results

We assess CAMO on synthetic benchmarks, including continuous and discrete MFBO tasks, as well as real-world engineering design tasks. We compare the results to those of: (1) BOCA with a standard GP [3], (2) Fabolas [16], and (3) SMAC3 [27]. On discrete fidelity tasks, we compare the results to: (1) AR [28], (2) ResGP [29], and (3) a GP [6], and discrete MFBO baselines (1) MF-UCB [30], (2) MF-EI [10], and (3) cfKG [17].

Settings. We assess CAMO with the LiFiDE kernel Eq. (15). Except for DNN-MFBO, Fabolas, and SMAC3 (original implementations and default settings), all methods were implemented in Pytorch. All GP models used the SE Kernel for a fair comparison. Each model is updated for 200 steps using an Adam optimizer with a learning rate of 0 . 01 to ensure model convergence. In each case, 10 low-fidelity and 4 high-fidelity designs were randomly selected to form the initial training set. We repeated the experiments 20 times with random seeds and report the mean values. Figures showing the actual optimization progress are provided in Appendix H. The optimization performance was measured by the simple regret ( γ ), defined as the difference between the global optimum and the best-queried design so far: γ i = max f ( x , T ) -max j&lt;i f ( x j , T ) . All experiments were performed on a workstation with an AMD 7800x CPU, Nvidia RTX4080 GPU, and 32GB RAM.

## 5.1 Synthetic Benchmark Evaluation

We consider: (1) three canonical continuous-fidelity tasks [31], the Park, Currin and Branin functions; and (2) three further synthetic continuous-fidelity tasks [32], the nonlinear sin, Forrester, and Bohachevsky functions. In the latter 3 we set f ( x , t ) = (1 -w ( t )) f low ( x ) + w ( t ) f high ( x ) with w ( t ) = ln(9 t +1) . All functions are defined in Appendix I. The query costs were set to c ( t ) = 10 t .

Figure 2: Simple regret for the Borehole, Colville and Himmelblau functions.

<!-- image -->

We show γ on the six tasks under increasing query cost in Fig. 1, which clearly demonstrates the superiority of CAMO across all tasks, with the exception of the non-linear sin at low cost. BOCA also performs well on the Branin and Park functions. The results for different random seeds are shown in Fig. H1-Fig. H9 in Appendix H to demonstrate the consistency of CAMO. These figures also show that methods equipped with cfKG (including CAMO) are generally less competitive in terms of γ and convergence speed compared to those with BOCA. To investigate the scalability of the results in terms of d , we further evaluated CAMO on the Borehole, Colville [33], and Himmelblau [34] functions (6, 4, and 2 design variables, respectively). The results are shown in Fig. 2. In these cases the superiority of CAMO is just as pronounced, suggesting robustness as d increases.

One of the key factors in MFBO is the cost c ( t ) of querying the high-fidelity function. We conducted experiments with different costs: c ( t ) = 10 t , c ( t ) = 5 t , and c ( t ) = log 2 (2 + t ) for the Currin and Bohachevsky functions. The results are shown in Fig. 3, and for the Forrester function in Fig. H10. Clearly, CAMO outperforms all other methods under all cost settings. An exponential cost setting is more challenging for all methods. The advantage of CAMO is increasingly significant from c ( t ) = log 2 (2 + t ) → 5 t → 10 t , which highlights the advantage of being convergence-aware in MFBOby maximising the benefit/cost ratio. For a logarithmic cost setting, CAMO performs similarly to BOCA because being 'convergence-aware' is less 'rewarded' when the cost also increases logarithmically. In contrast, if the cost increases beyond linear, the advantage of CAMO becomes significant. Such a setting is common in real-world applications, e.g., FEM (cubic complexity with mesh resolution [35]) and Monte Carlo estimation (quadratic sample requirements [36]). Importantly, CAMOalways exhibits a fast convergence rate in the early stages of the optimization since it does not discard useful low-fidelity information. This is consistent with the theoretical analysis.

Figure 3: Continuous MFBO for the Currin (top row) and Bohachevsky (bottom row) functions using logarithmic ( log 2 (2 + t ) ), linear ( 5 t ), and exponential ( 10 t ) cost c ( t ) .

<!-- image -->

Figure 4: Discrete MFBO on the Branin (left) and Currin functions (right).

<!-- image -->

Discrete MFBO Assessment. To examine performance on discrete MFBO tasks, we discretise the Branin and Currin into ten discrete fidelities. We also compare with the DNN-MFBO method [18]. The results under increasing query costs are shown in Fig. 4. We can see that CAMO again outperforms all other methods by a wide margin on the Branin function, while the advantages are clear but less dramatic on the Currin function. DNN-MFBO essentially failed.

For MFBO, the time for model update and acquisition-function based optimization is also crucial. The average times (over all the benchmarks, iterations, and seeds) of the optimization queries and model training are shown in Fig. 5 for different query and MF approaches, respectively. Despite the favourable performance reported in Li et al. [18], MutualInfo and FABOLAS are impractically slow in terms of query time. SMAC3 has the shortest simulation time, but it is not competitive in terms of accuracy and convergence rate. CAMO is intermediate between these methods for both query time using BOCA and training, achieving a good trade-off between performance and computational cost.

## 5.2 Real-World Applications in Engineering Design

We now consider real-world continuous tasks (see Appendix J). (1) Mechanical Plate Vibration. We optimize the natural vibration frequency of a 3-D, supported, square elastic plate ( 10 × 10 × 1 in m) over the Young's modulus ( ∈ [100 , 500] [GPa]), Poisson ratio ( ∈ [0 . 2 , 0 . 6] ) and mass density ( ∈ [6 × 10 3 , 9 × 10 3 ] [kgm -3 ]). This is a parametric FE modal analysis. The maximum element size

Figure 5: From left to right: average query time on discretised tasks, average query time on continuous tasks, average training on discretised tasks, and average training time on continuous tasks.

<!-- image -->

Figure 6: MFBO on the Mechanical Plate Vibration (left) and Thermal Conductor Design (right) problems with different query cost.

<!-- image -->

h max ∈ [0 . 2 , 1 . 2][ m ] is the fidelity. (2) Thermal Conductor. We optimize the shape of an elliptical central hole in which a conductor is placed in order to maximise a heat conduction rate. The hole shape is parameterised by the semi-major and semi-minor axes and orientation angle. We used the time to reach 70 degrees as the objective function value. h max ∈ [0 . 1 , 2][ m ] is the fidelity.

The γ versus simulation cost is shown in Fig. 6. The cost scales according to the inverse cubic of the maximal element size, which is the standard computational complexity for FEM problems. CAMO and BOCA outperform the other methods in both tasks by a significant margin. In particular, the fast convergence of CAMO carries over to these real-world problems, with BOCA yielding comparable performance on the Thermal Conductor and a slightly worse performance than CAMO on the Mechanical Plate Vibration. In terms of wall clock time to a given γ , CAMO and BOCA are very similar, with FABOLAS again impractically slow.

## 6 Conclusions

We propose CAMO, a convergence-aware MFBO framework based on LiFiDEs for continuous fidelity problems. CAMO captures the fidelity-wise evolution of the objective function and provides a theoretically-grounded surrogate model. For general FiDE we used a Lyapunov-based analysis to establish convergence guarantees even for non-smooth objectives. Combined with BOCA, CAMO delivers strong empirical performance, consistently outperforming state-of-the-art MFBO methods on synthetic and real-world tasks. Crucially, CAMO adapts to and exploits informative low-fidelity queries rather than discarding them, enabling more efficient use of the evaluation budget and fast convergence.

CAMO has several limitations that are worth noting. Computationally, the kernel optimization adds roughly 20% overhead compared to standard GP methods, and the approach requires sufficient low-fidelity data to properly model convergence behaviour. Methodologically, the LiFiDE assumption may not capture all (especially non-monotonic) convergence patterns, and performance depends on the existence of a good cost model. CAMO is suited to scenarios in which the computational budget is the primary constraint and early progress is valuable. For unlimited budget scenarios, more aggressive high-fidelity exploration might be preferable.

## Funding Disclosure

The authors declare no external funding.

## References

- [1] Eric Brochu, Vlad M. Cora, and Nando de Freitas. A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning. arXiv:1012.2599 [cs], December 2010.
- [2] Akeel A Shah, PK Leung, and WW Xing. Rapid high-fidelity quantum simulations using multi-step nonlinear autoregression and graph embeddings. npj Computational Materials, 11(1):57, 2025.
- [3] Kirthevasan Kandasamy, Gautam Dasarathy, Jeff Schneider, and Barnabás Póczos. Multi-fidelity bayesian optimisation with continuous approximations. In International conference on machine learning, pages 1799-1808. PMLR, 2017.
- [4] Jian Wu, Saul Toscano-Palmerin, Peter I. Frazier, and Andrew Gordon Wilson. Practical Multifidelity Bayesian Optimization for Hyperparameter Tuning. In Proceedings of The 35th Uncertainty in Artificial Intelligence Conference, pages 788-798. PMLR, August 2020.
- [5] Bobak Shahriari, Kevin Swersky, Ziyu Wang, Ryan P Adams, and Nando De Freitas. Taking the human out of the loop: A review of bayesian optimization. Proceedings of the IEEE, 104(1):148-175, 2015.
- [6] Carl Edward Rasmussen and Christopher K. I. Williams. Gaussian Processes for Machine Learning. Adaptive Computation and Machine Learning. MIT Press, Cambridge, Mass, 2006. ISBN 978-0-26218253-9.
- [7] Jonas Mockus. The application of bayesian methods for seeking the extremum. Towards global optimization, 2:117, 1978.
- [8] Niranjan Srinivas, Andreas Krause, Sham M Kakade, and Matthias Seeger. Gaussian process optimization in the bandit setting: No regret and experimental design. arXiv preprint arXiv:0912.3995, 2009.
- [9] HJ Kushner. A new method of locating the maximum point of an arbitrary multipeak curve in the presence of noise. Journal of Basic Engineering, 86(1):97-106, 1964.
- [10] Deng Huang, Theodore T Allen, William I Notz, and R Allen Miller. Sequential kriging optimization using multiple-fidelity evaluations. Structural and Multidisciplinary Optimization, 32:369-382, 2006.
- [11] Kirthevasan Kandasamy, Gautam Dasarathy, Junier B Oliva, Jeff Schneider, and Barnabás Póczos. Gaussian process bandit optimisation with multi-fidelity evaluations. Advances in neural information processing systems, 29, 2016.
- [12] Loic Le Gratiet and Josselin Garnier. Recursive co-kriging model for design of computer experiments with multiple levels of fidelity. International Journal for Uncertainty Quantification, 4(5), 2014.
- [13] Perdikaris P., Raissi M., Damianou A., Lawrence N. D., and Karniadakis G. E. Nonlinear information fusion algorithms for data-efficient multi-fidelity modelling. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 473(2198):20160751, February 2017. doi: 10.1098/rspa.2016.0751.
- [14] Kurt Cutajar, Mark Pullin, Andreas Damianou, Neil Lawrence, and Javier González. Deep Gaussian Processes for Multi-fidelity Modeling. arXiv:1903.07320 [cs, stat], March 2019.
- [15] Matthias Poloczek, Jialei Wang, and Peter Frazier. Multi-Information Source Optimization. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.
- [16] Aaron Klein, Stefan Falkner, Simon Bartels, Philipp Hennig, and Frank Hutter. Fast bayesian optimization of machine learning hyperparameters on large datasets. In Artificial intelligence and statistics, pages 528-536. PMLR, 2017.
- [17] Jian Wu and Peter I Frazier. Continuous-fidelity bayesian optimization with knowledge gradient. 2018.
- [18] Shibo Li, Wei Xing, Robert Kirby, and Shandian Zhe. Multi-Fidelity Bayesian Optimization via Deep Neural Networks. Advances in Neural Information Processing Systems, 33:8521-8531, 2020.
- [19] Wei Xing, Yuxin Wang, and Zheng Xing. Continuar: Continuous autoregression for infinite-fidelity fusion. Advances in Neural Information Processing Systems, 36, 2024.

- [20] Shibo Li, Zheng Wang, Robert Kirby, and Shandian Zhe. Infinite-fidelity coregionalization for physical simulation. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 25965-25978. Curran Associates, Inc., 2022.
- [21] Stefan Falkner, Aaron Klein, and Frank Hutter. Bohb: Robust and efficient hyperparameter optimization at scale. In International conference on machine learning, pages 1437-1446. PMLR, 2018.
- [22] Martin Wistuba, Arlind Kadra, and Josif Grabocka. Supervising the multi-fidelity race of hyperparameter configurations. Advances in Neural Information Processing Systems, 35:13470-13484, 2022.
- [23] Jiantong Jiang and Ajmal Mian. Fastbo: Fast hpo and nas with adaptive fidelity identification. arXiv preprint arXiv:2409.00584, 2024.
- [24] Frank H Clarke. Optimization and nonsmooth analysis. SIAM, 1990.
- [25] Neal Parikh, Stephen Boyd, et al. Proximal algorithms. Foundations and trends® in Optimization, 1(3): 127-239, 2014.
- [26] Neil Lawrence, Guido Sanguinetti, and Magnus Rattray. Modelling transcriptional regulation using gaussian processes. Advances in Neural Information Processing Systems, 19, 2006.
- [27] Tobias Domhan, Jost Tobias Springenberg, and Frank Hutter. Speeding up automatic hyperparameter optimization of deep neural networks by extrapolation of learning curves. In Twenty-fourth international joint conference on artificial intelligence, 2015.
- [28] M. Kennedy. Predicting the output from a complex computer code when fast approximations are available. Biometrika, 87(1):1-13, March 2000. ISSN 0006-3444, 1464-3510. doi: 10.1093/biomet/87.1.1.
- [29] W. W. Xing, A. A. Shah, P. Wang, S. Zhe, Q. Fu, and R. M. Kirby. Residual gaussian process: A tractable nonparametric bayesian emulator for multi-fidelity simulations. Applied Mathematical Modelling, 97: 36-56, September 2021. ISSN 0307-904X. doi: 10.1016/j.apm.2021.03.041.
- [30] Kirthevasan Kandasamy, Gautam Dasarathy, Junier B Oliva, Jeff Schneider, and Barnabas Poczos. Gaussian Process Bandit Optimisation with Multi-fidelity Evaluations. In Advances in Neural Information Processing Systems, volume 29. Curran Associates, Inc., 2016.
- [31] Shifeng Xiong, Peter ZG Qian, and CF Jeff Wu. Sequential design and analysis of high-accuracy and low-accuracy computer codes. Technometrics, 55(1):37-46, 2013.
- [32] Bruce Ankenman, Barry L Nelson, and Jeremy Staum. Stochastic kriging for simulation metamodeling. In 2008 Winter simulation conference, pages 362-370. IEEE, 2008.
- [33] Jialin Song, Yuxin Chen, and Yisong Yue. A General Framework for Multi-fidelity Bayesian Optimization with Gaussian Processes. In Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, pages 3158-3167. PMLR, April 2019.
- [34] Huachao Dong, Baowei Song, Peng Wang, and Shuai Huang. Multi-fidelity information fusion based on prediction of kriging. Structural and Multidisciplinary Optimization, 51(6):1267-1280, June 2015. ISSN 1615-147X, 1615-1488. doi: 10.1007/s00158-014-1213-9.
- [35] Thomas JR Hughes. The finite element method: linear static and dynamic finite element analysis. Courier Corporation, 2012.
- [36] Russel E Caflisch. Monte carlo and quasi-monte carlo methods. Acta numerica, 7:1-49, 1998.
- [37] Gerald Teschl. Ordinary differential equations and dynamical systems, volume 140. American Mathematical Soc., 2012.
- [38] R Tyrrell Rockafellar and Roger J-B Wets. Variational analysis, volume 317. Springer Science &amp; Business Media, 2009.

## A Gaussian process

Consider observations y i = f ( x i ) + ε , i = 1 , . . . , N , in which ε ∼ N (0 , σ 2 ) is additive noise. In a GP model, a prior distribution is placed over f ( x )

<!-- formula-not-decoded -->

in which m 0 ( x ) = E [ f ( x )] is the mean function and k ( x , x ′ | θ ) = E [( f ( x ) -m 0 ( x ))( f ( x ′ ) -m 0 ( x ′ ))] is the covariance function. A set of hyperparameters θ fully characterises the kernel function and in most cases the data is centred (the empirical mean is subtracted) to justify setting m 0 ( x ) ≡ 0 as a simplification. Various forms can be adopted for the covariance function, with the following exponential ARD kernel being the most widely favoured

<!-- formula-not-decoded -->

By the key property of GPs, the joint distribution of f ( x i ) , i = 1 , . . . , N , is a multivariate Gaussian. This leads to the following conditional predictive posterior conditioned on y = ( y 1 , . . . , y N ) ⊤ and x

<!-- formula-not-decoded -->

in which K = [ K ij ] , K ij = k ( x i , x j ) , i, j = 1 , . . . , N , is the covariance matrix, and k = ( k ( x i , x , . . . , k ( x N , x )) . The hyperparameters are typically inferred from the likelihood, given by

<!-- formula-not-decoded -->

## B Well-posedness of general FiDEs

## B.1 Existence and uniqueness

Here we consider the well-posedness of solutions to the general Fidelity Differential Equation (FiDE)

<!-- formula-not-decoded -->

in which X is any nonempty set, subject to some initial condition y ( x , 0) = y 0 ( x ) .

Lemma 1. [Existence and Regularity for General FiDEs] Let X ⊆ R d be a non-empty set. Consider the system (B1) and assume that:

1. for each fixed x ∈ X , the function ( t, y ) ↦→ ϕ ( x , t, y ) is continuous on [0 , ∞ ) × R
2. for each fixed x ∈ X , ϕ ( x , t, y ) is locally Lipschitz continuous in y , uniformly for t in compact subsets of [0 , ∞ )
3. the initial condition y 0 ( x ) ∈ R is finite for each x .

Then for each fixed x ∈ X , there exists a unique maximal solution y ( x , t ) ∈ C 1 ([0 , t ∗ ( x ))) defined on some maximal interval [0 , t ∗ ( x )) , with 0 &lt; t ∗ ( x ) ≤ ∞ . Moreover:

1. if y ( x , t ) remains bounded for all t ∈ [0 , ∞ ) , then t ∗ ( x ) = ∞ and the solution exists globally on [0 , ∞ )
2. if ϕ is globally Lipschitz in y , uniformly in t , then the solution exists globally on [0 , ∞ ) .

Proof. Fix x ∈ X , then the equation becomes an ordinary differential equation (ODE) in the variable t

<!-- formula-not-decoded -->

## Appendices

Since by assumption (1), the function ( t, y ) ↦→ ϕ ( x , t, y ) is continuous, and by assumption (2), ϕ ( x , t, y ) is locally Lipschitz in y , it follows from the Picard-Lindelöf Theorem that there exists a local time t ∗ ( x ) &gt; 0 and a unique solution t ↦→ y ( x , t ) ∈ C 1 ([0 , t ∗ ( x ))) that solves the ODE for t ∈ [0 , t ∗ ( x )) . Uniqueness of the solution y ( x , t ) follows directly from the local Lipschitz property of ϕ in y , which prevents branching of solutions and ensures that the solution is unique once y 0 ( x ) is fixed.

By the general theory of ODEs (continuation theorems) [37], the maximal existence time t ∗ ( x ) satisfies

<!-- formula-not-decoded -->

the latter representing blow-up of the solution in a finite time t ∗ . Thus, if we can guarantee that y ( x , t ) remains bounded for all t ∈ [0 , ∞ ) , then it follows that t ∗ ( x ) = ∞ , and the solution extends globally. If ϕ is globally Lipschitz in y uniformly in t , then no blow-up can occur, and the solution is guaranteed to exist globally. Since ϕ is continuous in ( t, y ) by assumption (1), and y ( x , t ) solves the ODE, standard regularity theory tells us that y ( x , · ) ∈ C 1 ([0 , t ∗ ( x ))) . Specifically, ∂ t y ( x , t ) = ϕ ( x , t, y ( x , t )) is continuous, as the composition of continuous functions.

## B.2 Pointwise and Uniform Convergence for FiDE

We now consider the convergence of solutions to (B1). The following results provide a practical set of conditions under which y ( x , t ) → y ∞ ( x ) pointwise or uniform in x .

Lemma 2 (Pointwise Convergence for FiDE) . Let y ( x , · ) ∈ C 1 ([0 , ∞ )) , x ∈ X be a solution to (1) with y 0 ( x ) &lt; ∞ for each x ∈ X . Suppose there exists a Lyapunov function V : X × [0 , ∞ ) × R → R , ( x , t, y ) ↦→ V ( x , t, y ) , and that the following conditions are satisfied:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then y ( x , t ) → y ∞ ( x ) pointwise on X , i.e., lim t →∞ y ( x , t ) = y ∞ ( x ) for each x ∈ X .

Proof. For each t , define the function y t : X → R , x ↦→ y ( x , t ) . Under the assumptions, we will show that the family of functions ( y t ) t ≥ 0 converges pointwise to y ∞ on X , that is

<!-- formula-not-decoded -->

Fix x ∈ X and consider the function t ↦→ V ( x , t, y ( x , t )) . By the chain rule, and using the assumptions, we have

<!-- formula-not-decoded -->

using ∂ t y ( x , t ) = ϕ ( x , t, y ) . By assumption, d dt V ( x , t, y ) is strictly negative, so that V ( x , t, y ) is strictly decreasing along the trajectory for each fixed x . Furthermore, V ( x , t, y ) &gt; 0 for all t ≥ 0 by assumption, and V ( x , t, y ) is bounded below by 0 . Now t ↦→ V ( x , y ( x , t ) , t ) is a continuous-time function and so for a fixed x , ( V ( x , y ( x , t ) , t )) t ∈ [0 , ∞ ) ⊂ R defines a bounded, monotone net on the directed set ([0 , ∞ ) , ≤ ) , where ≤ is the usual order relation. Therefore, this net converges to its infimum

<!-- formula-not-decoded -->

Since V is strictly decreasing in t and positive, its limit must be 0 (otherwise it would eventually fall below L ( x ) , a contradiction) and thus L ( x ) = 0 , i.e.

<!-- formula-not-decoded -->

Finally, V ( x , t, y ) = 0 if and only if y = y ∞ ( x ) , and thus

<!-- formula-not-decoded -->

Since x ∈ X was arbitrary, we obtain pointwise convergence on X .

Lemma 3 (Uniform Convergence for FiDE) . Let y ( x , · ) ∈ C 1 ([0 , ∞ )) , x ∈ X is a solution to (1). Suppose there exists a Lyapunov function V : X × [0 , ∞ ) × R → R , ( x , t, y ) ↦→ V ( x , t, y ) and the following conditions are satisfied:

1. y 0 ( x ) ∈ C ( X )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

4. ∃ c min &gt; 0 such that c ( x , t ) ≥ c min &gt; 0 , ∀ ( x , t ) ∈ X × [0 , ∞ ) , in which c ( x , t ) = 1 2 ∂ 2 yy V ( x , t, y ∞ ( x ))
5. for all x , t ∈ X × [0 , ∞ ) , ∃ α &gt; 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

6. ∃ V max such that ∥ V ( x , 0 , y ( x , 0)) ∥ L ∞ ( X ) ≤ V max

Then lim t →∞ y ( x , t ) = y ∞ ( x ) uniformly on X , with

<!-- formula-not-decoded -->

Proof. From assumption 5 and Grönwall's inequality [37], we have for all ( x , t ) ∈ X × [0 , ∞ )

<!-- formula-not-decoded -->

By the smoothness assumption V ∈ C 0 , 1 , 2 ( X × [0 , ∞ ) × R ) , a Taylor expansion around y = y ∞ ( x ) yields, for each ( x , t )

<!-- formula-not-decoded -->

in which c ( x , t ) = 1 2 ∂ 2 yy V ( x , y ∞ ( x ) , t ) , and by assumption 4, c ( x , t ) ≥ c min &gt; 0 uniformly.

Thus

<!-- formula-not-decoded -->

Taking the supremum over X yields

<!-- formula-not-decoded -->

which proves uniform convergence.

## B.3 Non-smooth Lyapunov functions

Lemma B1. [Convergence for non-smooth convex Lyapunov function] Let X be a nonempty set and let ( x , t ) satisfy (B1). Define V ( x , y, t ) = | y -y ∞ ( x ) | for ( x , y, t ) ∈ X × R × [0 , ∞ ) . Suppose that for each x ∈ X :

1. The map t ↦→ y ( x , t ) is C 1 ([0 , ∞ ))
2. Let ∂ y V ( x , y, t ) be the subdifferential mapping of y ↦→ V ( x , y, t ) . For every ( x , t ) ∈ X × [0 , ∞ ) and every ξ ( x , t ) ∈ ∂ y V ( x , y ( x , t ) , t ) , we have

<!-- formula-not-decoded -->

for some constant α &gt; 0 independent of x and t .

Then for each x ∈ X and every t ∈ [0 , ∞ ) ,

<!-- formula-not-decoded -->

If, moreover

<!-- formula-not-decoded -->

the convergence is uniform in x ∈ X

<!-- formula-not-decoded -->

Proof. For each fixed ( x , t ) ∈ X × [0 , ∞ ) , the map

<!-- formula-not-decoded -->

is convex and 1-Lipschitz continuous in y . For a fixed x ∈ X , we also define the function

<!-- formula-not-decoded -->

y ( x , · ) ∈ AC ([0 , ∞ )) (absolutely continuous) since it is in C 1 ([0 , ∞ )) , and thus the composite map v [ x ]( · ) is absolutely continuous on [0 , ∞ ) . By the properties of absolutely continuous functions, v [ x ]( · ) is differentiable almost everywhere, and its upper right Dini derivative

<!-- formula-not-decoded -->

exists for every t ∈ [0 , ∞ ) .

The subdifferential at a point y ∗ of the convex function u [ x , t ]( · ) is defined as the set of subgradients ∂u [ x , t ]( y ∗ ) = { g ∈ R : u [ x , t ]( y ) ≥ u [ x , t ]( y ∗ ) + g ( y -y ∗ ) , ∀ y ∈ R } [24, 38]. In the present case, the set-valued subdifferential mapping y ↦→ ∂u [ x , t ]( y ) , R → 2 R is given by

<!-- formula-not-decoded -->

Applying the chain rule to u [ x , t ]( · ) and using the definition of the subdifferential mapping of a convex function we obtain

<!-- formula-not-decoded -->

by using the definition of y . Given assumption (2), for any subgradient ξ ( x , t ) ∈ ∂u [ x , t ]( y ) ∈ R , we have

<!-- formula-not-decoded -->

which holds for t ∈ [0 , ∞ ) a.e and pointwise in x ∈ X , with y = y ( x , t ) fixed. Thus, for t ∈ [0 , ∞ ) a.e.

<!-- formula-not-decoded -->

Since v [ x ]( t ) ∈ AC ([0 , ∞ )) , by the standard properties of Dini derivatives, this inequality extends to the upper Dini derivative, yielding

<!-- formula-not-decoded -->

Applying the generalised Grönwall inequality for upper Dini derivatives (which states that if D + u ( t ) ≤ -αu ( t ) , then u ( t ) ≤ u (0) e -αt ), we obtain

<!-- formula-not-decoded -->

Recalling that V ( x , y, t ) = | y -y ∞ ( x ) | , this yields

<!-- formula-not-decoded -->

pointwise x ∈ X . By assumption (3) this holds in the L ∞ ( X ) norm by taking the sup on both sides, i.e.

<!-- formula-not-decoded -->

This result can be extended to non-convex Lyapunov functions by using the Clarke subdifferential [24]. We first introduce some definitions. Let X be a Banach space with dual X ∗ . The weak ∗ topology on X ∗ is the coarsest topology such that all evaluation maps f ↦→ f ( x ) for x ∈ X are continuous. That is, f α → f in the weak ∗ topology iff f α ( x ) → f ( x ) for all x ∈ X . Now let f : R n → R be locally Lipschitz. The Clarke subdifferential of f at x is defined as

<!-- formula-not-decoded -->

in which co denotes the convex hull of a set. This set is always nonempty, convex, and closed. If f = g ◦ h , where g : R → R and h : R → R n are locally Lipschitz, then for any t ∈ R ([24], Chain Rule Theorem 2.3.9)

<!-- formula-not-decoded -->

in which co w ∗ {·} denotes the closure of the convex hull in the weak ∗ topology on R ∗ ∼ = R . In finite dimensions, notably in our case R , this is simply the norm-closed convex hull and the standard-norm, weak, and weak ∗ topologies are equivalent.

If we set h ( t ) = y ( x , t ) and g ( y ) = u [ x ]( y ) (we assume V has no explicit dependence on t ), then f ( t ) = v [ x ]( t ) as previously defined. Since y ( x , · ) ∈ C 1 ([0 , ∞ )) , ∂ C y ( t ) = { ∂ t y ( x , t ) } , so that

<!-- formula-not-decoded -->

Now, the image of a convex set under a linear map is convex, and therefore (B32) reduces to

<!-- formula-not-decoded -->

since co w ∗ ≡ co and furthermore the Clarke subdifferential is already closed and convex.

We know that the composition v [ x ]( t ) = V ( x , y ( x , t ) , t ) ∈ AC ([0 , ∞ )) because V is locally Lipschitz in y and v is differentiable. Then by Clarke's Chain Rule Theorem above

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If we assume that then

We can then use Grönwall's inequality and take the supremum over x ∈ X to obtain a uniform bound. This proves the following lemma.

Lemma 4. Let X be a nonempty set, and let y : X × [0 , ∞ ) → R be such that for each x ∈ X , y ( x , · ) ∈ C 1 [0 , ∞ ) . Let V : X × R → R , ( x , y ) → V ( x , y ( x , t )) satisfy the following conditions

1. For every fixed ( x , t ) , the map y ↦→ V ( x , y, t ) is locally Lipschitz.
2. For each fixed ( x , t ) ∈ X × [0 , ∞ ) , define u [ x ]( · ) : R → R , y ↦→ V ( x , y ) , and for a fixed x ∈ X , define v [ x ]( · ) : [0 , ∞ ) → R , t ↦→ V ( x , y ( x , t )) . Assume that for all ( x , t ) ∈ X × [0 , ∞ ) and all ξ ( x , t ) ∈ ∂ C u [ x ]( y ) the inequality

<!-- formula-not-decoded -->

holds for some constant α &gt; 0 , independent of x and t .

3. The initial deviation is bounded: ∥ y ( · , 0) -y ∞ ( · ) ∥ L ∞ ( X ) &lt; ∞ .

Then, for all t ∈ [0 , ∞ ) , we have

<!-- formula-not-decoded -->

## B.4 Connection to Subgradient Flows and Proximal Methods

We have analysed the convergence of fidelity-indexed systems y : X × [0 , ∞ ) → R , which serve as multi-fidelity surrogate models for a high-fidelity target function y ∞ : X → R . These surrogates evolve over fidelity index t according to the FiDE dynamics (1). When a non-smooth Lyapunov function such as V ( x , y, t ) = | y -y ∞ ( x ) | is used to study convergence, the time evolution must be interpreted using subdifferential calculus. In particular, the upper Dini derivative satisfies the inclusion

<!-- formula-not-decoded -->

in which ξ ( x , t ) ∈ ∂ y V ( x , y ( x , t ) , t ) is a subgradient. For example, when V ( x , y, t ) = | y -y ∞ ( x ) | , ∂ y V ( x , y, t ) is given by (B22). This structure defines a differential inclusion, and is directly analogous to subgradient flows commonly used in convex optimization, particularly for non-smooth regularisation problems such as LASSO. In these cases, the evolution of a state x ( t ) under a non-smooth objective g is given by

<!-- formula-not-decoded -->

For the case in Lemma B1, the surrogate y ( x , t ) evolves by descent on the 'fidelity error' g ( y ) = | y -y ∞ ( x ) | , driving it asymptotically towards the equilibrium y ∞ ( x ) . Under mild monotonicity assumptions on ϕ , this descent yields exponential convergence, analogous to the behavior of proximal-point iterations in optimization.

This framework extends naturally to more general non-smooth Lyapunov functions, such as the elastic-net form

<!-- formula-not-decoded -->

or group sparsity objectives for vector-valued surrogates y : X → R d

<!-- formula-not-decoded -->

in which G is a partitioning of the coordinates. In all such cases, the surrogate y ( x , t ) evolves via a subdifferential inclusion that mirrors proximal-map dynamics in optimization, linking multi-fidelity modelling with tools from modern sparse learning and non-smooth convex analysis.

The subgradient dynamics discussed above can also be viewed as a limiting case of smooth gradient flows applied to a smoothed surrogate objective. In convex analysis, the Moreau envelope provides a classical smoothing of a non-smooth function. For a convex, lower semi-continuous function f : R → R , the envelope is defined as [25]

<!-- formula-not-decoded -->

with ∇ f λ ( x ) = 1 λ ( x -prox λf ( x )) . This provides a differentiable approximation of f , and as λ → 0 , we have f λ ( y ) → f ( y ) pointwise, and ∇ f λ ( y ) → ∂f ( y ) in the sense of graphical (or Mosco) convergence.

While the actual FiDE dynamics are not given by a gradient flow, the Lyapunov convergence analysis admits a natural interpretation via these smoothed flows. For instance, using the fidelity error function f ( y ) = | y -y ∞ ( x ) | , we define the Moreau envelope

<!-- formula-not-decoded -->

This yields the auxiliary gradient flow

<!-- formula-not-decoded -->

which approximates the subgradient dynamics

<!-- formula-not-decoded -->

as λ → 0 . While this flow is not the governing equation for y ( x , t ) in the FiDE, it serves as a useful analytical proxy, capturing the key descent structure and convergence behaviour of the surrogate. From this point of view, the Lyapunov function represents a fidelity error whose decay over t encodes asymptotic convergence to the high-fidelity limit.

## C Well-Posedness and Convergence of Linear FiDE: Proofs

We consider the linear FiDE

<!-- formula-not-decoded -->

subject to an initial condition y 0 ( x ) = y ( x , 0) .

## C.1 Proof of Lemma 5

Proof. Let X ⊂ R d be non-empty and fix x ∈ X . Suppose that:

<!-- formula-not-decoded -->

(A2) y 0 ( x ) ∈

<!-- formula-not-decoded -->

Define ϕ ( x , t, y ) = β ( x , t ) y + u ( x , t ) . For fixed x , the map ϕ ( x , · , · ) is continuous in ( t, y ) since both β ( x , t ) and u ( x , t ) are continuous in t . Moreover, ϕ is globally Lipschitz in y since

<!-- formula-not-decoded -->

Thus, | β ( x , t ) | acts as a Lipschitz constant (depending only on t ). By the Picard-Lindelöf Theorem, there exists a unique local solution y ( x , t ) ∈ C 1 defined on some interval [0 , t ∗ ( x )) .

Now suppose additionally that: (A3) ∥ β ( x , t ) ∥ L ∞ ([0 , ∞ )) &lt; ∞ and ∥ u ( x , t ) ∥ L ∞ ([0 , ∞ )) &lt; ∞ . Then the right-hand side of the ODE

<!-- formula-not-decoded -->

remains bounded on compact intervals of [0 , ∞ ) . Therefore, the solution y ( x , t ) cannot blow up in finite time, and the local solution can be extended globally to [0 , ∞ ) , i.e., t ∗ ( x ) = ∞ , and y ( x , t ) ∈ C 1 ([0 , ∞ )) .

Define the integrating factor µ ( x , t ) = e ∫ t 0 β ( x ,s ) ds . Then

<!-- formula-not-decoded -->

and integrating both sides yields the variation of constants formula

<!-- formula-not-decoded -->

Now Let X ⊂ R d be compact, and suppose that:

- (B1) β ( x , t ) , u ( x , t ) ∈ C ( X × [0 , ∞ ))
- (B2) y 0 ( x ) ∈ C ( X )

in addition to (A3). Then for each ( x , t ) ∈ X × [0 , ∞ ) , the solution is given by the variation-of-constants formula

<!-- formula-not-decoded -->

We now show that y ( x , t ) ∈ C ( X × [0 , ∞ )) . Define

<!-- formula-not-decoded -->

Since β ( x , s ) ∈ C ( X × [0 , ∞ )) , the map is continuous on X × [0 , ∞ ) .

Now define and denote the integrand by

## C.2 Proof of Theorem 1

Assume the following:

1. For each x ∈ X , we have β ( x , · ) ∈ C ([0 , ∞ )) ∩ L 1 loc ([0 , ∞ )) , β ( x , t ) &lt; 0 , ∀ ( x , t ) ∈ X × [0 , ∞ ) and lim t →∞ β ( x , t ) = β ∗ ( x ) &lt; 0
2. For each x ∈ X , u ( x , · ) ∈ C ([0 , ∞ )) , and lim t →∞ u ( x , t ) = u ∗ ( x ) ∈ R
3. ∥ u ( x , t ) ∥ L ∞ ( X× [0 , ∞ )) = M &lt; ∞ and ∥ β ( x , t ) ∥ L ∞ ( X× [0 , ∞ )) = λ &lt; ∞

To prove Theorem 1 we will require the following Proposition.

Proposition C1. For each fixed x ∈ X and each r ≥ 0 , the following holds

<!-- formula-not-decoded -->

Proof. Since β ( x , t ) → β ∗ ( x ) pointwise in x ∈ X , for any δ &gt; 0 , ∃ t 0 &gt; 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is continuous. The exponential function is continuous, and y 0 ( x ) ∈ C ( X ) , so the product

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since β ( x , z ) ∈ C ( X × [0 , ∞ )) , the map ( x , t, s ) ↦→ ∫ t s β ( x , z ) dz is continuous. Since u ( x , s ) ∈ C ( X × [0 , ∞ )) , it follows that f ( x , t, s ) ∈ C ( X × [0 , ∞ ) 2 ) . Due to the compactness of X and β, u ∈ L ∞ , the integrand is uniformly bounded. Therefore, the parameter-dependent integral

<!-- formula-not-decoded -->

is continuous in ( x , t ) . Since both terms F 1 ( x , t ) and F 2 ( x , t ) are continuous, it follows that

<!-- formula-not-decoded -->

For a fixed r &gt; 0 , let δ = ε/r and define t ∗ = t 0 + r . Then for all t ≥ t ∗ , we have t -r ≥ t 0 , and so for all u ∈ [ t -r, t ] , with t ≥ t ∗

Therefore, for all t ≥ t ∗

<!-- formula-not-decoded -->

From this we conclude that for each x ∈ X and each r ≥ 0

<!-- formula-not-decoded -->

Proof of Theorem 1. Since we are concerned with pointwise convergence we need only employ the standard topology on R . Under the assumptions above, the variation-of-constants formula for a fixed x ∈ X yields the solution

<!-- formula-not-decoded -->

Since β ( x , · ) ∈ C ([0 , ∞ )) ⊂ L 1 loc ([0 , ∞ )) , the integrals are well-defined. Moreover, since β ( x , t ) &lt; 0 and β ( x , t ) → β ∗ ( x ) &lt; 0 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so that

Let us now define the inhomogeneous integral

<!-- formula-not-decoded -->

Using the change of variable r = t -s , we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If we set then ( f t ( r )) t ∈ [0 ,T ) defines a net and

<!-- formula-not-decoded -->

Now define the following limiting function

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For each fixed r ≥ 0 , u ( x , t -r ) → u ∗ ( x ) . Moreover, from Proposition C1, for each x ∈ X and each r ≥ 0

<!-- formula-not-decoded -->

and therefore f t ( r ) → f ( r ) pointwise in r .

Since the Dominated Convergence Theorem applies only to sequences, we now extract a sequence ( f t n ) n ∈ N such that t n →∞ and consider the corresponding sequence of functions ( f t n ) . This sequence satisfies

1. f t n ( r ) → f ( r ) for each r ≥ 0 by the assumption on u and Proposition C1
2. | f t n ( r ) | ≤ Me rλ = g ( r ) , in which g ∈ L 1 ([0 , ∞ )) (since λ &lt; 0 ) is a dominating function

Therefore, by the Dominated Convergence Theorem, we have

<!-- formula-not-decoded -->

Since the limit of the integral is independent of the particular sequence ( t n ) , and because R is first-countable (i.e., sequential), it follows that the net (∫ ∞ 0 f t ( r ) dr ) t ∈ [0 , ∞ ) converges to the same value

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This yields the final result

## C.3 Proof of Theorem 2

We again consider the linear FiDE problem (C1), now with X ⊂ R d a compact set (closed and bounded). We make the following assumptions:

1. y 0 ( x ) ∈ C ( X ) ;
2. β ( x , t ) ∈ C ( X × [0 , ∞ )) , with

<!-- formula-not-decoded -->

3. u ( x , t ) ∈ C ( X × [0 , ∞ )) , with

<!-- formula-not-decoded -->

4. There exists λ ′ &gt; 0 such that β ∗ ( x ) ≤ -λ ′ for all x ∈ X .

We will require the following two Propositions in order to prove Theorem 2.

Proposition C2. There exist finite λ &gt; 0 and M &gt; 0 such that

<!-- formula-not-decoded -->

Proof. By compactness of X , β ∗ ∈ C ( X ) , and by assumption (4), we have

<!-- formula-not-decoded -->

Furthermore, since β ( · , t ) → β ∗ ( · ) uniformly in L ∞ ( X ) , there exists T &gt; 0 such that for all t ≥ T

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

On the compact set X × [0 , T ] , β is continuous, hence bounded above. Let

<!-- formula-not-decoded -->

noting that β ( x , t ) &lt; 0 , ∀ ( x , t ) ∈ X × [0 , ∞ ) , otherwise we can use λ = min { λ ′ / 2 , inf X× [0 ,T ] ( -β ( x , t )) } . Then for all x , t , β ( x , t ) ≤ -λ .

Similarly, uniform convergence of u ( · , t ) → u ∗ ( · ) ∈ C ( X ) implies there exists M &gt; 0 such that

<!-- formula-not-decoded -->

Proposition C3. For each fixed s ≥ 0 , the following

<!-- formula-not-decoded -->

holds uniformly for x ∈ X as t →∞ .

Hence, for all x ∈ X , t ≥ T

Proof. By the uniform convergence of β , for every ϵ &gt; 0 , there exists U ∗ ( ϵ ) &gt; 0 such that for all u ≥ U ∗

∗

<!-- formula-not-decoded -->

Let t ≥ U ∗ , then for any fixed s ≥ 0

<!-- formula-not-decoded -->

The first integral is over a finite interval, and is thus uniformly bounded in x . For the second term, observe that β ( x , u ) = β ∗ ( x ) + δ ( x , u ) , where | δ ( x , u ) | &lt; ϵ for all u ≥ U ∗ . Thus

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with

Therefore, since s and U ∗ are fixed

<!-- formula-not-decoded -->

Thus

<!-- formula-not-decoded -->

in which c &gt; 0 is a constant independent of t and x . As t →∞ , the right-hand side tends to ϵ , and since ϵ &gt; 0 was arbitrary, (C39) follows, uniformly in x for each fixed s ≥ 0 .

Proof of Theorem 2. By the variation of constants formula, the solution is given by

<!-- formula-not-decoded -->

By assumption (4), and Proposition C2, we have β ( x , t ) ≤ -λ &lt; 0 for all x , t , and so

<!-- formula-not-decoded -->

Furthermore, y 0 ( x ) ∈ C ( X ) with compact X , so that ∥ y 0 ( x ) ∥ L ∞ ( X ) = K &lt; ∞ . Thus,

<!-- formula-not-decoded -->

and therefore the first term in (C46) vanishes uniformly in x .

Now let

<!-- formula-not-decoded -->

From Proposition C3

<!-- formula-not-decoded -->

uniformly in x for each fixed s ≥ 0 , and since β ∗ ( x ) &lt; 0 we obtain

<!-- formula-not-decoded -->

also uniformly in x for each fixed s ≥ 0 . When analysing the behavior of f t ( x , s ) for all s ∈ [0 , t ] , s and t may diverge as t →∞ . In order to handle the behavior across the entire range s ∈ [0 , t ] , we use change of variable r = t -s , so that

<!-- formula-not-decoded -->

Now, for each fixed r ≥ 0 , as t →∞ , s = t -r →∞ . Thus, both

<!-- formula-not-decoded -->

hold uniformly in x . Therefore and therefore

<!-- formula-not-decoded -->

uniformly in x , for each fixed r ≥ 0 . Moreover, since β ( x , t ) ≤ -λ &lt; 0

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

( f t ( x , r )) t ≥ 0 forms a net rather than a sequence, so we pick any sequence ( t n ) n ≥ 1 ⊂ [0 , ∞ ) such that t n →∞ . For each fixed r ≥ 0 , f t n ( x , r ) → u ∗ ( x ) e β ∗ ( x ) r uniformly in x , and | f t n ( x , r ) | ≤ Me -λr with Me -λr ∈ L 1 ([0 , ∞ )) . Thus, by the Dominated Convergence Theorem, we have

<!-- formula-not-decoded -->

Since the limit is independent of the chosen sequence, it follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## D Joint Covariance Function For Linear FiDE

Let (Ω , F , P ) be a probability space supporting two independent stochastic processes y 0 : X × Ω → R and u : X × [0 , ∞ ) × Ω → R . We place Gaussian process priors over both

<!-- formula-not-decoded -->

The solution to the linear fidelity-indexed dynamical system is given by

<!-- formula-not-decoded -->

in which β ( x , t ) is a deterministic function satisfying β ( x , t ) &lt; 0 and regular enough to ensure well-defined integrals. Thus, y ( x , t ) is a zero-mean GP (as an affine transformations of GPs), and its expectation is taken over the joint measure induced by both GP priors. The covariance function of y ( x , t ) is therefore given by k ( x , t, x ′ , t ′ ) = E ω ∼ P [ y ( x , t ; ω ) y ( x ′ , t ′ ; ω )] and expanding the product in the argument yields

<!-- formula-not-decoded -->

noting that the cross terms disappear due to independence. Taking expectations and using linearity, the zero-mean assumptions and Fubini's Theorem (which holds for GPs under mild assumptions), we obtain

<!-- formula-not-decoded -->

## E Closed-from Kernels

Constant β ( x , t ) = β and Periodic Kernel.

Consider the periodic exponential sine squared (ESS) kernel, defined as

<!-- formula-not-decoded -->

We finally conclude that

in which σ 2 is the signal variance, ℓ is the length scale, and p is the period. Assuming a constant β ( x , t ) = β we obtain

<!-- formula-not-decoded -->

in which B n ( · ) is the modified Bessel function of the first kind of order n .

## Constant β ( x , t ) = β and Matérn Kernel.

The Matérn kernel with parameter ν is defined as

<!-- formula-not-decoded -->

in which σ 2 is the signal variance, ℓ is the characteristic length scale, Γ( · ) is the gamma function, and K ν ( · ) is the modified Bessel function of the second kind of order ν . For ν = 3 2 , the covariance function is

<!-- formula-not-decoded -->

The double integral in the second term can be evaluated analytically, yielding

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Constant β ( x , t ) = β and Random Fourier Features.

Since SE, periodic, and Matérn kernels are all stationary, we can extend the closed-form solution to any stationary kernel. Recall that any stationary kernel can be represented as a Fourier series, and the random Fourier features (RFF) kernel is defined as

<!-- formula-not-decoded -->

in which σ 2 is the signal variance, m is the number of random features, and ω i are randomly sampled frequencies from a distribution p ( ω ) (typically a Gaussian distribution with zero mean and variance 1 /ℓ 2 , where ℓ is the length scale of the squared exponential kernel being approximated). Assuming a constant β ( x , t ) = β , the covariance function becomes

<!-- formula-not-decoded -->

## F Background on BOCA and Theorem 3

Here we describe the BOCA method of Kandasamy et al. Kandasamy et al. [3] (adapted to our problem). We set X = [0 , 1] d and T = [0 , 1] and let f ( x ) = y ( x , 1) be the black-box function we aim to optimize, with different fidelities t given by a function y ( x , t ) : X × T → R . Assume that

1. y ( x , t ) ∼ GP (0 , κ ) with a product kernel

<!-- formula-not-decoded -->

in which κ 0 &gt; 0 and both k t , k x are valid kernels.

2. λ ( t ) is the cost function for evaluating y at fidelity t .

Let A ⊂ X , and let A n = { x 1 , . . . , x n } ⊂ A be a finite-cardinality subset. Define noisy observations y A n = f A n + ϵ A n with ϵ n ∼ N (0 , η 2 ) . Here, y A n = ( y 1 , . . . , y n ) T , f A n = ( f ( x 1 , . . . , x n ) ⊤ and ϵ A n = ( ϵ 1 , . . . , ϵ n ) ⊤ . For a continuous random vector X ∼ N ( µ, Σ) , the differential entropy is defined as

<!-- formula-not-decoded -->

The mutual information or information gain I ( y A n ; f A n ) is a measure of the reduction in uncertainty in the values of f A n after acquiring observations y A n . For Gaussian random vectors, information gain is given by the difference in differential entropy

<!-- formula-not-decoded -->

in which K A n ∈ R n × n is the kernel matrix with entries ( K A n ) ij = κ ( x i , x j ) . The maximal information gain after n evaluations on a subset A ⊂ X is then defined as

<!-- formula-not-decoded -->

Let x ∗ = arg max x ∈X f ( x ) and f ∗ = f ( x ∗ ) . The simple regret after capital Λ is defined as

<!-- formula-not-decoded -->

Λ refers to the total budget or cumulative cost allowed for querying the objective function, and therefore bounds the cumulative cost of all function evaluations.

A measure of how informative fidelity t is with regards to the maximum fidelity t = 1 is given by the following information-gap function

<!-- formula-not-decoded -->

A smaller ξ ( t ) implies greater informativeness. If ξ ( t ) is smooth, the gap decreases as t → 1 . Now define

<!-- formula-not-decoded -->

noting that ξ (0) is the maximal fidelity gap. We note here that this expression appears to be different from that in Kandasamy et al. [3], in which the authors use ξ ( √ p ) for T = [0 , 1] p throughout their paper. Clearly √ p / ∈ [0 , 1] p and this is a typing error (the correct expression is ξ ( 0 ) ). Most queries at the highest fidelity t = 1 are made from the set X ρ . In relation to X ρ , let

̸

<!-- formula-not-decoded -->

in which B 2 ( x, ε ) denotes the Euclidean ( ℓ 2 ) ball of radius ε centred at x , n is the number of queries at any fidelity and α ∈ (0 , 1) . The set X ρ,n includes all points in X that lie within a Euclidean distance √ d n α/ 2 d of X ρ , and can be interpreted as a polynomial-rate dilation of the latter. As n →∞ , we have X ρ,n →X ρ at a rate of n -α/ 2 d .

At iteration n , the BOCA algorithm selects a query point ( x n , t n ) ∈ X × [0 , T ) in two stages. First, it constructs an upper confidence bound (UCB) acquisition function φ n ( x ) over the domain X , conditioned on the fidelity t

<!-- formula-not-decoded -->

in which µ n -1 ( x , 1) and σ n -1 ( x , 1) are the mean and standard deviation of the posterior GP slice at t = 1 over y ( x , t ) (i.e., over f ( x ) = y ( x , 1) ) given all previous observations { y ( x i , t i ) } n -1 i =1 , and β n is a t -varying confidence parameter. The latter is given by

<!-- formula-not-decoded -->

in which δ ∈ (0 , 1) and a, b &gt; 0 are constants depending on the kernel k x . Specifically, noting that f ( x ) ∼ GP (0 , k x ( x , x ′ ) since k t (1 , 1) = 1 , it is assumed that there exist constants a, b &gt; 0 such that

<!-- formula-not-decoded -->

Having chosen the next input x n ∈ arg max x ∈X φ n ( x ) , the fidelity level t n ∈ [0 , 1) is chosen from the filtered set

<!-- formula-not-decoded -->

̸

where γ ( t ) = √ κ 0 ξ ( t ) ( λ ( t ) λ (1) ) q . q depends on the kernel and for a SE kernel q = (1 + d + 2) -1 . If T n ( x n ) = ∅ , the algorithm chooses the cheapest fidelity from this set, i.e.,

<!-- formula-not-decoded -->

otherwise it sets t n = 1 . With the definitions above we can state the main result.

Theorem F1 (Kandasamy et al. [3], Theorem 8) . Let X = [0 , 1] d , T = [0 , 1] and y ( x , t ) ∼ GP (0 , κ ) with κ ( x , t, x ′ , t ′ ) = κ 0 k t ( | t -t ′ | ) k x ( ∥ x -x ′ ∥ ) , in which k x satisfies Assumption (F11). Choose δ ∈ (0 , 1) and execute BOCA with β n chosen as in F10). Then, for any α ∈ (0 , 1) , there exist ρ &gt; 0 and Λ 0 such that with probability at least 1 -δ

<!-- formula-not-decoded -->

in which C 1 is a universal constant and n Λ = ⌊ Λ λ (1) ⌋ , while ρ &gt; max { 2 , 1 + √ (1 + 2 /α ) / (1 + d ) .

## G Asymptotic behaviour of the LiFiDE kernel

We expand I ( t, t ′ ) around t ′ = t + δ for small δ as

<!-- formula-not-decoded -->

in which I 0 ( t ) and I 1 ( t ) are given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

I 0 ( t ) increases monotonically as t →∞ . The crucial term in I 1 ( t ) is

<!-- formula-not-decoded -->

which is strictly negative. For the squared exponential k SE, due to symmetry and stationarity

<!-- formula-not-decoded -->

in which I 0 and I 2 are independent of t . Thus, the decay is O ( δ 2 ) compared to O ( δ ) for the LiFiDE kernel. Therefore, correlations between values of y ( x , t ) and y ( x , t ′ ) around t ′ = t + δ are more concentrated, and do not depend on t .

Remark G1 (On Regret Bounds and Kernel Choice) . BOCA's theoretical regret bound depends on the fidelitygap function ξ ( t ) , which in turn is determined by the fidelity kernel through

<!-- formula-not-decoded -->

for a maximum fidelity T . For the SE kernel k t ( | t -T | ) = exp ( -( t -T ) 2 2 ℓ 2 ) we have the approximation

<!-- formula-not-decoded -->

around t = T , which decays linearly in the distance to the highest fidelity T . Thus, X ρ is sharply localised and regret is relatively easy to control.

We can simplify the expression for the LiFiDE kernel as follows

<!-- formula-not-decoded -->

in which c 1 and c 2 are the values of k 0 and k x at a fixed ( x , x ′ ) ∈ X 2 . Consider t = T -δ in the limit δ → 0 + . Expanding the kernel around t ′ = t + δ we obtain

<!-- formula-not-decoded -->

in which I 1 ( t ) is defined in (G3). We now normalise the kernel so that it takes values in [0 , 1] by defining

<!-- formula-not-decoded -->

Since k ( t, t + δ ) = k ( t, t ) + O ( δ ) and k ( t + δ, t + δ ) = k ( t, t ) + O ( δ ) , we have

<!-- formula-not-decoded -->

The square fidelity gap becomes

<!-- formula-not-decoded -->

Since I 1 ( t ) &lt; 0 , ∀ t , we finally obtain

<!-- formula-not-decoded -->

which shows that near the highest fidelity level, the fidelity gap decays as √ | T -t | , i.e., at a much slower rate than that for the SE kernel.

## H Additional Experimental Results

Below we provide more detailed information on the sampling functions for all continuous MFBO experiments conducted in Section 5. Each marker represents a sampling point. Among the numerous seeds, we selected the first four for display. The results for the Branin, Currin, Park, nonlinear sin, Forrester, Bohachevsky, Borehole, Colville, and Himmelblau functions are shown in Fig. H1-Fig. H9, respectively.

We can see the advantage of CAMO in almost all cases in terms of the query cost and simple regret. The query cost is significantly reduced compared to the other methods, and the simple regret is also competitive. In particular, CAMO always exhibits a fast convergence rate in the early stage of the optimization process. Upon closer inspection, CAMO conducts most of its exploration in the low-fidelity region, reducing the query cost, which is consistent with the theoretical analysis. To explore the performance of different models combined with different acquisition functions, we provide additional results with different costs for the Forrester function in Fig. H10. The results are consistent with those in the main text. CAMO is the optimal choice, particularly when the cost function is exponential-like, which is more realistic in real-world applications such as finite element analysis and neural network architecture search (NAS).

Figure H1: Branin function MFBO results from four random seeds.

<!-- image -->

Figure H2: Currin function MFBO results from four random seeds.

<!-- image -->

Figure H3: Park function MFBO results from four random seeds.

<!-- image -->

Figure H4: Nonlinear sin function MFBO results from four random seeds.

<!-- image -->

Figure H5: Forrester function MFBO results from four random seeds.

<!-- image -->

Figure H6: Bohachevsky function MFBO results from four random seeds.

<!-- image -->

Figure H7: Borehole function MFBO results from four random seeds.

<!-- image -->

Figure H8: Colvile function MFBO results from four random seeds.

<!-- image -->

Figure H9: Himmelblau function MFBO results from four random seeds.

<!-- image -->

Figure H10: MFBO for the Forrester function with different cost functions.

<!-- image -->

## I Synthetic Benchmarks

The definitions of the objective functions in the synthetic benchmark experiments are given below.

## Park Function

The input is two dimensional, x ∈ [0 , 1] 2 and there is one fidelity index t ∈ [0 , 1]

<!-- formula-not-decoded -->

## Currin Function

The input is two dimensional, x ∈ [0 , 1] 2 and there is one fidelity index t ∈ [0 , 1]

<!-- formula-not-decoded -->

## Branin Function

The input is two dimensional, x ∈ [0 , 1 . 5] 2 and there is one fidelity index t ∈ [0 , 1]

<!-- formula-not-decoded -->

in which b = 5 . 1 2 , c = 5 , r = 6 and s = 1

<!-- formula-not-decoded -->

## Nonlinear Sin Function

This is a two-level multi-fidelity function where input is one dimensional x ∈ [0 , 1 . 5] . Low and high fidelity are given by

<!-- formula-not-decoded -->

The low and high fidelity functions are shown in Figure I1.

## Forrester Function

Two-level multi-fidelity function with input x ∈ [0 , 1 . 5]

<!-- formula-not-decoded -->

The low and high fidelity functions are shown in Figure I1.

## Bohachevsky Function

Two-level multi-fidelity function with input x ∈ [ -5 , 5] 2

f

high

f

low

(

(

) =

x

) =

x

x

2

1

f

+2

high

(0

x

2

2

.

7

x

-

1

0

, x

.

3 cos(3

2

) +

x

1

πx

x

1

)

2

-

## Borehole Function

Two-level multi-fidelity function with input variables r w ∈ [0 . 05 , 0 . 15] , r ∈ [100 , 50000] , T u ∈ [63070 , 115600] , H u ∈ [990 , 1110] , T l ∈ [63 . 1 , 116] , H l ∈ [700 , 820] , L ∈ [1120 , 1680] , K w ∈ [9855 , 12045]

<!-- formula-not-decoded -->

-

12

0

.

.

4 cos(4

πx

2

) + 0

.

7

,

(G19)

<!-- image -->

X

Figure I1: Low and high fidelity nonlinear sin function (left) and Forrester function (right).

## Colville Function

Two-level multi-fidelity function where input is four dimensional, x ∈ [ -1 , 1] 4 and coefficient A &lt; = 0 . 68 , where low and high fidelity is given by:

<!-- formula-not-decoded -->

## Himmelblau Function

Two-level multi-fidelity function with input x ∈ [ -1 , 1] 2 and coefficient A ≤ 0 . 68

<!-- formula-not-decoded -->

## J Details of Real-World Applications

## J.1 Mechanical Plate Vibration Design

The objective is to optimize the design of a 3-D simply supported, square, elastic plate with dimensions 10 × 10 × 1 [m], as illustrated in Fig. J1. The primary goal is to identify materials that maximise the fourth vibration mode frequency, thereby minimising the risk of resonance-induced damage caused by interactions with other components. The material properties under consideration include the Young's modulus (ranging from 1 × 10 11 to 5 × 10 11 [Pa]), the Poisson ratio (between 0 . 2 and 0 . 6 ), and mass density (varying from 6 × 10 3 to 9 × 10 3 [kgm -3 ]). The plate is discretised using quadratic tetrahedral elements, as depicted in Fig. J1.

Figure J1: Quadratic tetrahedral element discretisation of the plate ( h max = 1 . 2 ).

<!-- image -->

## J.2 Thermal Conductor Design

The second application focuses on optimising the design of a thermal conductor, as shown in Fig. J2(a). The heat source is located on the left side of the conductor, with the temperature initially at 0 and increasing to 100 degrees within 0 . 5 seconds. The heat transfer occurs through the conductor towards the right end. The conductor dimensions and material properties, including thermal conductivity and mass density (both equal to 1 ), are fixed. To facilitate installation, a hole must be bored into the center of the conductor. The top, bottom, and inner surfaces of the hole are thermally insulated, preventing heat transfer across these boundaries. The

Figure J2: (a) Snapshot of the thermal conductor temperature solution; (b) the heat response curve on the right edge.

<!-- image -->

size and angle of the hole play a crucial role in determining the rate of heat transfer. In general, the hole is an ellipse, characterised by three parameters: the semi-minor and semi-major axes, and the orientation angle. The objective is to minimise the time required for the temperature at the right end to reach 70 degrees, thus maximising the heat conduction rate from left to right. To evaluate the time taken to reach the target temperature, the conductor is discretised using quadratic tetrahedral elements. The finite element method is then applied to solve the problem, yielding a response heat curve at the right edge, as illustrated in Fig. J2(b). By analysing this response curve, the time at which the temperature reaches 70 degrees can be determined.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We have carefully ensured that the main ideas and finding are accurately reflected in both the abstract and introduction. The introduction provides a detailed description of the proposed method and the results.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We provide a detailed list of the limitations in the conclusions section.

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

Justification: Each result lists and numbers the detailed assumptions in the statement. These assumptions are further discussed in the text and in remarks.

Guidelines:

- The answer NA means that the paper does not include theoretical results.

- All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

## Answer: [Yes]

Justification: We provide full descriptions of the examples and of settings used in the method and competing methods (hyperparameter choices, kernels used for each method, statement of original formulations used if that is the case, details of the software implementation and hardware used, etc).

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

## Answer: [Yes]

Justification: Code is publicly available at https://github.com/IceLab-X/CAMO .

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/public/ guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https://nips.cc/public/ guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: The settings are listed in the experimental section in detail, including training/test split, number of random seeds, the optimizer used, and so on.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We discuss the results from different random seeds and in the Appendix provide additional results from individual seeds in all cases to demonstrate consistency.

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

Justification: We list the computer harware used, the software employed and provides details on the simulation costs.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have carefully adhered to the NuerIPS code of Ethics, ensuring that none of the rules are violated.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: While the work can have an impact on Multi-Fidelity Bayesian optimization, we cannot claim any direct societal impact of the work at this stage.

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

Justification: no models/data with dual-use risk are released

Guidelines:

- The answer NA means that the paper poses no such risks.

- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: When comparing to existing methods we use the orignal codes if available and have cited the relevant papers (with discussions).

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

Justification: no human-subject or crowdsourcing study is involved

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: no human-subject or crowdsourcing study is involved

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