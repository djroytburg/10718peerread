## Second-Order Convergence in Private Stochastic Non-Convex Optimization

## Youming Tao

TU Berlin &amp; Shandong University tao@ccs-labs.org

## Zuyuan Zhang

The George Washington University zuyuan.zhang@gwu.edu

Dongxiao Yu Shandong University dxyu@sdu.edu.cn

Xiuzhen Cheng

Shandong University xzcheng@sdu.edu.cn

Falko Dressler TU Berlin dressler@ccs-labs.org

## Di Wang

KAUST

di.wang@kaust.edu.sa

## Abstract

We investigate the problem of finding second-order stationary points (SOSP) in differentially private (DP) stochastic non-convex optimization. Existing methods suffer from two key limitations: (i) inaccurate convergence error rate due to overlooking gradient variance in the saddle point escape analysis, and (ii) dependence on auxiliary private model selection procedures for identifying DP-SOSP, which can significantly impair utility, particularly in distributed settings. To address these issues, we propose a generic perturbed stochastic gradient descent (PSGD) framework built upon Gaussian noise injection and general gradient oracles. A core innovation of our framework is using model drift distance to determine whether PSGD escapes saddle points, ensuring convergence to approximate local minima without relying on second-order information or additional DP-SOSP identification. By leveraging the adaptive DP-SPIDER estimator as a specific gradient oracle, we develop a new DP algorithm that rectifies the convergence error rates reported in prior work. We further extend this algorithm to distributed learning with heterogeneous data, providing the first formal guarantees for finding DP-SOSP in such settings. Our analysis also highlights the detrimental impacts of private selection procedures in distributed learning under high-dimensional models, underscoring the practical benefits of our design. Numerical experiments on real-world datasets validate the efficacy of our approach.

## 1 Introduction

Stochastic optimization is a fundamental problem in machine learning and statistics, aimed at training models that generalize well to unseen data using a finite sample drawn from an unknown distribution. As the volume of sensitive data continues to grow, privacy has become a pressing concern. This has led to the widespread adoption of differential privacy (DP) [11], which provides rigorous privacy guarantees while preserving model utility in learning tasks.

In the past decade, significant progress has been made in DP stochastic optimization, particularly for convex objectives [8, 29, 41, 39, 43]. While convex problems are relatively well understood, non-convex optimization introduces unique challenges, primarily due to the presence of saddle points.

Most existing DP algorithms for non-convex problems focus on finding first-order stationary points (FOSP), characterized by small gradient norms [2, 5, 54]. However, FOSP include not only local minima but also saddle points and local maxima, often leading to suboptimal solutions [21, 42]. Consequently, second-order stationary points (SOSP), where the gradient is small and the Hessian is positive semi-definite, are more desirable as they guarantee convergence to local minima.

Motivated by this, substantial research has been devoted to finding SOSP in non-convex optimization [14, 24, 10, 22, 17]. However, the study of SOSP under differential privacy constraints (DPSOSP) remains limited. At the same time, distributed learning has become increasingly important for training large-scale models across decentralized edge devices. Yet, no existing work has addressed DP-SOSP in non-convex stochastic optimization under distributed settings. Compared to singlemachine setups, distributed learning introduces additional challenges, including data heterogeneity, cross-participant privacy, and communication efficiency.

Limitations in the State-of-the-Art. A notable exception in the study of DP-SOSP for stochastic optimization is the recent work by [30], which injects additional Gaussian noise into the DP gradient estimator near saddle points to facilitate escape. Despite its contributions, this method suffers from two key limitations. (i) Its saddle point escape analysis overlooks the variance of gradients, leading to incorrect error bounds. A direct correction of the analysis would unfortunately yield a weaker type of SOSP guarantee than originally targeted. This is because their design relies on additional injected noise beyond the inherent DP noise for escape, highlighting the need for an effective way of exploiting the DP noise already present. (ii) Their learning algorithm outputs all model iterates and guarantees only the existence of a DP-SOSP, requiring an auxiliary private model selection procedure to identify one. While effective in single-machine settings, it faces critical issues in distributed environments due to decentralized data access. In particular, auxiliary private selection introduces non-negligible error and communication overhead, especially when sharing high-dimensional second-order information. These drawbacks also underscore the necessity of a new learning algorithm that inherently outputs a DP-SOSP without dependence on any additional private selection procedure.

Our Contributions. We refer to Appendix A for more detailed discussions of the limitations outlined above. To address the challenges identified above, we propose a generic algorithmic and analytical framework for finding DP-SOSP in stochastic non-convex optimization. Our approach not only corrects existing error rates but also extends naturally to distributed learning. The main contributions are summarized as follows:

1. A generic non-convex stochastic optimization framework: We introduce a perturbed stochastic gradient descent (PSGD) framework that employs Gaussian noise and general stochastic gradient oracles. This framework serves as a versatile optimization tool for non-convex stochastic problems beyond the DP setting. A key innovation is a novel criterion based on model drift distance, which enables provable saddle point escape and guarantees convergence to approximate local minima with low iteration complexity and high probability.

2. Corrected error rates for DP non-convex optimization: By incorporating the adaptive DPSPIDER estimator as the gradient oracle, we develop a differentially private algorithm that achieves a corrected error rate bound of ˜ O ( 1 n 1 / 3 + ( √ d ϵn ) 2 / 5 ) , where n is the number of samples. This corrects the suboptimal bound of ˜ O ( 1 n 1 / 3 + ( √ d ϵn ) 3 / 7 ) reported in [30].

3. Application to distributed learning: We extend the adaptive DP-SPIDER estimator to distributed learning. Via adaptivity, our learning algorithm improves upon the DIFF2 [37], which only guarantees convergence to DP-FOSP under homogeneous data. In contrast, our method provides the first error bound for converging to DP-SOSP under heterogeneous data: ˜ O ( 1 ( mn ) 1 / 3 + ( √ d ϵmn ) 2 / 5 ) , where m is the number of participants and n is the number of samples per participant. Furthermore, we analyze the adverse effects of private model selection, showing that it deteriorates utility guarantees in high-dimensional regimes, thereby highlighting the necessity of our framework.

Due to the space limit, technical lemmata , omitted proofs , experimental results and broader impacts , conclusions are all included in the Appendix.

## 2 Related Work

Private Stochastic Optimization. Differential privacy (DP) has become a crucial consideration in stochastic optimization due to increasing concerns about data privacy. The pioneering work by [11] established the foundational principles of DP, and its application in stochastic optimization has since seen significant progress. Early efforts primarily focused on convex optimization, achieving strong privacy guarantees while ensuring efficient learning, with a long list of representative works e.g., [6, 51, 48, 4, 47, 49, 15, 5, 20, 43, 41, 8, 40]. Recent advances have extended DP to non-convex settings, mainly focusing on first-order stationary points (FOSP). Notable works in this area include [46, 54, 5, 52, 2], which improved error rates in non-convex optimization with balanced privacy and utility in stochastic gradient methods. However, these works generally fail to address the more stringent criterion of second-order stationary points (SOSP). The very recent work [30] tired to narrow this gap, but unfortunately has some issues in their results as we discussed before. Our work builds on this foundation by correcting error rates and proposing a framework that ensures convergence to SOSP while maintaining DP.

Finding SOSP. In non-convex optimization, convergence to FOSP is often insufficient, as saddle points can lead to sub-optimal solutions [21, 42]. Achieving SOSP, where the gradient is small and the Hessian is positive semi-definite, ensures that the optimization converges to a local minimum rather than a saddle point. Techniques for escaping saddle points, such as perturbed SGD with Gaussian noise, have been explored in works like [17] and [24]. [17] first showed that SGD with a simple parameter perturbation can escape saddle points efficiently. Later, the analysis was refined by [22, 24]. Recently, variance reduction techniques have been applied to second-order guaranteed methods [18, 28].These methods ensure escape from saddle points by introducing noise to the gradient descent process. In contrast, the studies of SOSP under DP are quite limited, and most of them only consider the empirical risk minimization objective, such as [46, 50, 3]. Very recently, [30] addressed the population risk minimization objective, but with notable gaps in their error analysis, particularly in the treatment of gradient variance. Moreover, all of these works are limited to the single-machine setting and cannot be directly extended to the more general distributed learning setting.

Distributed Learning. With the rise of large-scale models and decentralized data, distributed learning has gained significant attention. Methods like federated learning [34] have enabled multiple clients to collaboratively train models without sharing their local data. Recent studies, such as [16, 32, 33] have investigated DP learning in distributed settings, but these works are limited to first-order optimality. While some studies have investigated SOSP in distributed learning, their focus was primarily on Byzantine-fault tolerance [53], and communication efficiency [36, 7]. No effort, to our knowledge, has been made to to ensure DP-SOSP in distributed learning scenarios with heterogeneous data. Our proposed framework fills this gap by introducing the first distributed learning algorithm with DP-SOSP guarantees while effectively handling data heterogeneity across clients.

## 3 Preliminaries

Notations. We denote by ∥ · ∥ the ℓ 2 norm and by λ min ( · ) the smallest eigenvalue of a matrix. The symbol I d represents the d -dimensional identity matrix. We use O ( · ) and Ω( · ) to hide constants independent of problem parameters, while ˜ O ( · ) and ˜ Ω( · ) additionally hide polylogarithmic factors.

Stochastic Optimization. Let f : R d ×Z → R be a (potentially non-convex) loss function, where x ∈ R d denotes the d -dimensional model parameter and z ∈ Z is a data point.

Assumption 1. The loss function f ( · ; z ) is G -Lipschitz, M -smooth, and ρ -Hessian Lipschitz. Specifically, for any z ∈ Z and any x 1 , x 2 ∈ R d , we have: (i) | f ( x 1 ; z ) -f ( x 2 ; z ) | ≤ G ∥ x 1 -x 2 ∥ ; (ii) ∥∇ f ( x 1 ; z ) -∇ f ( x 2 ; z ) ∥ ≤ M ∥ x 1 -x 2 ∥ ; (iii) ∥∇ 2 f ( x 1 ; z ) -∇ 2 f ( x 2 ; z ) ∥ ≤ ρ ∥ x 1 -x 2 ∥ .

Let D denote the unknown data distribution. The population risk is defined as the expected loss: F D ( x ) := E z ∼D [ f ( x ; z )] for ∀ x ∈ R d . When clear from context, we omit D and simply write F ( x ) .

Assumption 2. Let x ∗ denote a minimizer of the population risk and F ∗ = F ( x ∗ ) its minimum value. There exists U ∈ R such that max x F ( x ) -F ∗ ≤ U .

ˆ

Let D denote a dataset of n i.i.d. samples from D . The empirical risk is defined as f D ( x ) := 1 | D | ∑ z ∈ D f ( x ; z ) . Given access to D , the goal is to find an approximate second-order stationary point (SOSP) of the unknown population risk F ( · ) . In general, we have the notion of ( α g , α H ) -SOSP: Definition 1 ( ( α g , α H ) -SOSP) . A point x is an ( α g , α H ) -SOSP of a twice differentiable function F ( · ) if x satisfies ∥∇ F ( x ) ∥ ≤ α g and ∇ 2 F ( x ) ⪰ -α H · I d .

As shown in [53, Proposition 1], there exists a lower bound of ˜ O ( α 1 / 2 g ) for α H given α g , implying that an ( α, ˜ O ( √ α )) -SOSP is the best second-order guarantee achievable. Accordingly, we target the notion of α -SOSP in this work, following [30].

Definition 2 ( α -SOSP) . A point x is an α -SOSP of a twice differentiable function F ( · ) if x satisfies ∥∇ F ( x ) ∥ ≤ α and ∇ 2 F ( x ) ⪰ - √ ρα · I d .

An α -SOSP excludes α -strict saddle points where ∇ 2 F ( x ) ⪯ - √ ρα I d , thereby ensuring convergence to an approximate local minimum. Following prior work [30, 24], we assume M ≥ √ ρα so that finding an SOSP is strictly more challenging than finding an FOSP.

Distributed Learning. In the distributed (federated) learning setting, m clients collaboratively learn under the coordination of a central server. Each client j ∈ [ m ] has a local dataset D j of size n , sampled from an unknown local distribution D j . The population risk for client j is defined as F D j ( x ) := E z ∼D j [ f ( x ; z )] or simply F j ( x ) . The global population risk is defined as the average of the local population risks: F D ( x ) := 1 m ∑ j ∈ [ m ] F j ( x ) , or simply F ( x ) . We allow for heterogeneous local datasets, meaning that the local distributions {D j } j ∈ [ m ] may differ.

Differential Privacy. We aim to find an α -SOSP under the requirment of Differential Privacy (DP), which is referred to as an α -DP-SOSP. We say two datasets D and D ′ are adjacent if they differ by at most one record. DP ensures that the output of the stochastic optimization algorithm on any pair of adjacent datasets is statistically indistinguishable.

Definition 3 (Differential Privacy (DP) [11]) . Given ϵ, δ &gt; 0 , a randomized algorithm A : Z → X is ( ϵ, δ )-DP if for any pair of adjacent datasets D,D ′ ⊆ Z , and any measurable subset S ⊆ X ,

<!-- formula-not-decoded -->

In distributed learning, we focus on inter-client record-level DP (ICRL-DP) , which assumes that clients do not trust the server or other clients with their sensitive local data. This notion has been widely adopted in state-of-the-art distributed learning works, such as [16, 32, 33].

Definition 4 (Inter-Client Record-Level DP (ICRL-DP)) . Given ϵ, δ &gt; 0 , a randomized algorithm A : Z m →X satisfies ( ϵ, δ )-ICRL-DP if, for any client j ∈ [ m ] and any pair of local datasets D j and D ′ j , the full transcript of client j 's sent messages during the learning process satisfies (3), assuming fixed local datasets for other clients.

Variance Reduction via SPIDER. Since the population risk F ( · ) is unknown, standard SGD approximates the true gradient ∇ F ( x t -1 ) at iteration t using a stochastic estimate g t . However, such estimates often exhibit high variance, degrading convergence. The Stochastic Path Integrated Differential Estimator (SPIDER) [13] mitigates this variance using two gradient oracles O 1 and O 2 . For a mini-batch B t at iteration t , we define

<!-- formula-not-decoded -->

SPIDER queries O 1 every p iterations to refresh the gradient estimate. Between these updates, it uses O 2 to incrementally refine the estimate:

<!-- formula-not-decoded -->

For smooth functions, the variance of O 2 ( x t -1 , x t -2 , B t ) scales with ∥ x t -1 -x t -2 ∥ , which is typically small when updates are minimal. This allows SPIDER to achieve low-variance gradient estimates while maintaining accuracy.

We choose SPIDER because it achieves state-of-the-art error rates for privately finding first-order stationary points (DP-FOSP) [2]. Our goal is to investigate whether its variance reduction can extend to DP-SOSP. Importantly, the insights in this paper are not specific to SPIDER; they also apply to other variance-reduced methods such as STORM [9] or SARAH [38]. However, since these algorithms are conceptually similar, no significant improvement is expected from substituting them.

## 4 Our Generic Perturbed SGD Framework

In this section, we introduce a generic framework for finding an α -SOSP of the population risk F D ( · ) by escaping saddle points. Our framework is a Gaussian perturbed stochastic gradient descent method, denoted as Gauss-PSGD .

## 4.1 Gradient Oracle Setup

Since ∇ F D ( · ) is unknown, direct gradient descent is infeasible. As in standard stochastic optimization, we assume access to a stochastic gradient oracle g t that approximates ∇ F D ( x t -1 ) at iteration t . For example, g t can be computed as an empirical gradient over a mini-batch B t sampled from D . We model the oracle as

<!-- formula-not-decoded -->

where ζ t represents inherent gradient noise. Following [24, 30], we assume ζ t ∼ nSG( σ ) , where nSG denotes a norm-sub-Gaussian distribution (Definition 7 in Appendix B).

To enable saddle point escape, we introduce an additional Gaussian perturbation to form a perturbed gradient oracle ˆ g t :

<!-- formula-not-decoded -->

where ξ t ∼ N (0 , r 2 I d ) . We define the effective noise magnitude in ˆ g t as

<!-- formula-not-decoded -->

The model update is then performed by

<!-- formula-not-decoded -->

<!-- image -->

Our problem setting fundamentally differs from that in [24]. In their setting, the target error α is given, and the perturbation magnitude r is determined accordingly. In contrast, in our privacy-constrained setting, r is dictated by the privacy parameters ( ϵ, δ ) , and the goal is to achieve the smallest possible α under this constraint. Crucially, their parameterization r = O ( √ ( σ 2 + α 3 / 2 ) /d ) implies that r depends on both σ and α , determined by max { σ/ √ d, α 3 / 4 / √ d } . This non-invertible relationship between r and α makes their setting incompatible with ours. First, under DP constraints, r is determined by ( ϵ, δ ) and may be smaller than σ/ √ d in weak privacy regimes, violating the required lower bound. Second, because r and α are not uniquely determined by each other, it is not meaningful to directly translate their error bounds into our setting. Thus, their analysis and results cannot be directly applied to our problem.

## 4.2 Our Approach: A General Gaussian-Perturbed SGD Framework

We present our Gauss-PSGD framework in Algorithm 1, which finds an α -SOSP with high probability at least 1 -ω . As specified in (2), we employ a general Gaussian-perturbed stochastic gradient oracle, denoted as P\_Grad\_Oracle ( ∗ ) in steps 4 and 10, where ∗ abstracts the specific arguments required by the oracle implementation. This abstraction allows Gauss-PSGD to serve as a flexible optimization framework for non-convex stochastic problems, applicable beyond the differential privacy (DP) setting.

At each iteration, the gradient estimate ˆ g t is computed by P\_Grad\_Oracle ( ∗ ) , and the model parameter is updated via the gradient descent step in (4). The algorithm proceeds until it encounters a

point ˜ x satisfying ∥ ˆ g t ∥ ≤ 3 χ , where χ is specified in (5). This point ˜ x may lie near a saddle point with a large negative eigenvalue of the Hessian. To escape such a saddle point, the framework enters an escape procedure (steps 6-20), which performs Q rounds of Γ -descent (steps 9-16).

In each round, the algorithm executes at most Γ perturbed SGD iterations starting from ˜ x . If at any iteration we observe ∥ x t -˜ x ∥ ≥ R for a threshold R (specified in (5)), indicating that the iterate has moved sufficiently far from ˜ x , we declare that the algorithm has successfully escaped the saddle point and resume normal PSGD from x t . If no such movement is observed after Q rounds, we declare ˜ x an α -SOSP of the population risk F D ( · ) and output ˜ x . The repetition over Q rounds ensures a high probability of escape: as we will prove later, each Γ -descent succeeds in escaping a saddle point with constant probability, and multiple repetitions reduce the failure probability to any desired level.

A central innovation of our framework is using model drift distance as the escape criterion (step 12), replacing the function value decrease criterion used in [22, 24]. This design enables the algorithm to identify an SOSP with high probability during the optimization process itself, eliminating the need for an auxiliary private model selection step. Our key insight is as follows: escaping a saddle point not only causes a decrease in the objective function [22, 24] but also induces a substantial displacement of the model parameter beyond a threshold R . Shifting from monitoring function values to tracking parameter movement is critical in population risk settings, where the objective function is unknown and function evaluations are unavailable, unlike in empirical risk minimization [22]. However, the model iterates and their deviations are observable. By leveraging this property, our framework can directly output an SOSP, rather than merely guaranteeing its existence among the iterates.

## 4.3 Main Results for Gauss-PSGD Framework

We begin by introducing the parameter setup and notations used throughout the analysis:

<!-- formula-not-decoded -->

where s is a sufficiently large absolute constant to be chosen later, and µ is a logarithmic factor:

<!-- formula-not-decoded -->

Here C is an absolute constant that may change across expressions. The rationale behind these parameter choices is further discussed in Remark 2 following Theorem 1. Let ˜ x denote a saddle point of the population risk F ( · ) , and H := ∇ 2 F (˜ x ) . Let v min be the eigenvector corresponding to λ min ( H ) , and P -v min be the projection onto the orthogonal complement of v min . Set γ := -λ min ( H ) .

Definition 5 (Coupling Sequence) . Let { x i } and { x ′ i } be two PSGD sequences initialized at ˜ x . We say they are coupled if they share the same randomness for P -v min ξ t and ζ t at each iteration t , but use opposite perturbations in the v min direction: v ⊤ min ξ t = -v ⊤ min ξ ′ t .

The following lemma ensures that under Γ -descent , at least one of the coupled sequences escapes the saddle point with constant probability (proof in Appendix C.1).

Lemma 1 (Escaping Saddle Points) . Let { x i } and { x ′ i } be coupled PSGD sequences initialized at ˜ x such that ∥∇ F (˜ x ) ∥ ≤ α and λ min ( ∇ 2 F (˜ x )) ≤ - √ ρα . Then, with probability at least 1 / 4 , there exists τ ≤ Γ such that max {∥ x τ -˜ x ∥ , ∥ x ′ τ -˜ x ∥} ≥ R .

From this, we immediately obtain a corollary that applies to any PSGD sequence:

Corollary 1. For any PSGD sequence { x i } starting at ˜ x with ∥∇ F (˜ x ) ∥ ≤ α and λ min ( ∇ 2 F (˜ x )) ≤ - √ ρα , with probability at least 1 / 8 , there exists t ≤ Γ such that ∥ x t -˜ x ∥ ≥ R .

To ensure a high-probability escape from a saddle point, we repeat Γ -descent for Q rounds:

Lemma 2 (Escape Amplification via Repetition) . Given any ω 0 ∈ (0 , 1) , repeating Γ -descent independently for Q = 26 5 log( 1 ω 0 ) rounds ensures escape with probability at least 1 -ω 0 .

The proof is deferred to Appendix C.2. We now analyze the total number of PSGD steps needed for convergence. Let ν t := ζ t + ξ t denote the combined noise in the gradient estimate.

Lemma 3 (Descent Lemma) . For any t 0 , the following holds:

<!-- formula-not-decoded -->

Since ν t can be bounded with high probability, we have:

Corollary 2. For any t 0 and some constant c , with probability at least 1 -2 e -ι ,

<!-- formula-not-decoded -->

Proofs of Lemma 3 and Corollary 2 are in Appendix C.3 and C.4. These imply that large gradients lead to rapid function decrease. We next show in Lemma 4 that a successful saddle point escape via Γ -descent leads to a significant decrease in function value, whose proof is in Appendix C.5.

Lemma 4 (Value Decrease per Escape) . Let a Γ -descent starting from x t 0 succeed after τ ≤ Γ steps. With probability at least 1 -2 e -ι , F ( x t 0 + τ ) -F ( x t 0 ) ≤ -s 8 ι 3 √ α 3 ρ = -Φ .

We bound the total number of PSGD steps required for convergence, based on the following estimate: Lemma 5 (Gradient Estimate Error Bound) . With probability at least 1 -ω/ 2 , for all t ∈ [ T ] , ∥ ν t ∥ ≤ C √ 2 log ( 4 T ω ) ψ ≤ χ .

Lemma 6 (Maximum Number of Descent Steps) . Given failure probability ω , set Q = 26 5 log ( 16 ι 3 ( F 0 -F ∗ ) sω √ ρ χ 3 ) . Gauss-PSGD returns an α -SOSP within at most ˜ O (1 /α 2 . 5 ) PSGD steps.

Proofs of Lemmas 5 and 6 are in Appendix C.6 and C.7, respectively.

Remark 1 (On Gradient Complexity) . While Lemma 6 appears to improve gradient complexity from O (1 /α 4 ) in [24] to O (1 /α 2 . 5 ) , the two results are not directly comparable. In [24], the error target α is treated as an input and can be arbitrarily small, with gradient variance σ typically treated as a constant. In contrast, in our setting, the perturbation r and variance σ are fixed by privacy constraints, and α emerges as a function of these. Thus, our gradient complexity fundamentally depends on σ and r , though we express it in terms of α for clarity.

Combining all the above, we obtain the final convergence guarantee:

Theorem 1 (Convergence Guarantee of Gauss-PSGD ) . Let Assumptions 1 and 2 hold. For any failure probability ω ∈ (0 , 1) , using the parameter settings in (5) and setting Q = 26 5 log ( 16 ι 3 ( F 0 -F ∗ ) sω √ ρ χ 3 ) , then with probability at least 1 -ω , Gauss-PSGD (Algorithm 1) re- turns an α -SOSP of F ( · ) , where α = 4 χ , within at most ˜ O (1 /α 2 . 5 ) PSGD steps.

Remark 2 (On the setting of parameters) . The parameters introduced in (5) are chosen in accordance with our convergence and privacy analysis. Specifically, the escape threshold χ matches the gradient estimation error, ensuring a uniform expected decrease in the objective value per PSGD step (cf. Lemma 5 and Lemma 6). The model drift threshold κ balances the cumulative error from the gradient oracles O 1 and O 2 , while the maximum drift threshold R and maximum escape steps Γ jointly control the curvature-dependent term P h ( t ) and keep the stochastic gradient noise P sg ( t ) bounded (see Eq. (41) and (43)). Finally, the repeat number Q is chosen to grow logarithmically in the failure probability parameter to amplify the overall success probability, as established in Lemma 2.

## 5 Rectified Error Rate for finding SOSP in DP Stochastic Optimization

## 5.1 Adaptive Gradient Oracle: Ada-DP-SPIDER

In this section, we derive the upper bound on the error rate for DP stochastic optimization by instantiating the Gauss-PSGD framework with a specific gradient oracle. We adopt an adaptive version

of the DP-SPIDER estimator, referred to as Ada-DP-SPIDER , which is presented in Algorithm 2. This adaptive version refines the original SPIDER by dynamically adjusting gradient queries based on model drift. Unlike standard SPIDER, which queries O 1 at fixed intervals and may suffer from growing estimation error over time, Ada-DP-SPIDER tracks the cumulative model drift defined as

<!-- formula-not-decoded -->

where τ ( t ) is the last iteration at which the full gradient oracle O 1 was queried.

The intuition is that, for smooth functions, the error of O 2 , which estimates ∇ F ( x t -1 ) -∇ F ( x t -2 ) , is proportional to ∥ x t -1 -x t -2 ∥ . When the model drift is small, O 2 remains accurate, allowing for continued use to reduce variance (steps 9-11). However, when the drift becomes large, further use of O 2 can accumulate significant errors. To mitigate this, the algorithm triggers a fresh query to O 1 (steps 4-7). A threshold κ is used in step 3 to determine when the drift is large. This enables adaptive switching between oracles based on the model drift, ensuring the total error remains well controlled.

Our approach differs fundamentally from that of [30]. In their method, in addition to using model drift to trigger O 1 , they also invoke O 1 when approaching potential saddle points and inject an additional Gaussian noise on top of the DP gradient estimator to escape. To prevent excessive noise injection, they introduce a Frozen state to restrict how frequently this occurs. In contrast, our method leverages the inherent Gaussian noise from the DP gradient estimator for saddle point escape and uses model drift as the sole trigger for querying O 1 . This results in a simpler, more efficient estimator without auxiliary state tracking or redundant noise injection.

## 5.2 Error Rate Analysis for DP-SOSP with Ada-DP-SPIDER

To minimize the error rate α for DP-SOSP using Ada-DP-SPIDER , we must carefully tune algorithmic parameters, including the mini-batch sizes b 1 , b 2 , and the drift threshold κ . These parameters directly influence the gradient estimation error, which, according to Theorem 1, dominates the learning error. The following lemma characterizes how these parameters affect the estimation quality:

Lemma7. Let Assumption 1 hold. For all t ∈ [ T ] , the gradient estimate ˆ g given by Ada-DP-SPIDER

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof is given in Appendix D.1. To ensure that b 1 and b 2 remain valid mini-batch sizes under a fixed sample budget, we must control the number of times O 1 is queried. Lemma 8 bounds the count:

Lemma 8. Let Assumption 1 and 2 hold. Define T := { t ∈ [ T ] : drift t ≥ κ } as the set of rounds where the drift exceeds the threshold κ . With high probability (as in Theorem 1), |T | ≤ O ( Uη/κ ) .

Proof is in Appendix D.2. Guided by Lemmas 7 and 8, we now derive the error bound for α via appropriate choices of b 1 , b 2 , and κ in Theorem 2. The proof is provided in Appendix D.3.

Theorem 2. Let Assumption 1 and 2 hold. Define b 1 = nκ 2 Uη , b 2 = nηχ 2 2 U and κ = max { G 3 / 2 U 1 / 2 ρ 1 / 2 M 5 / 2 n 1 / 2 , G 14 / 15 d 2 / 5 U 4 / 5 ρ 8 / 15 M 34 / 15 ( nϵ ) 4 / 5 } . Then, running Gauss-PSGD with gradient oracle instantiated by Ada-DP-SPIDER ensures ( ϵ, δ ) -DP for constants c 1 , c 2 and returns an α -SOSP with α = ˜ O ( 1 n 1 / 3 + ( √ d nϵ ) 2 / 5 ) 1 .

Remark 3 (No Cyclic Dependency Among Parameters) . All algorithmic parameters are consistently defined in terms of the problem parameters n , d , and ϵ . Specifically, Gauss-PSGD parameters such as the step size η and the noise scale χ depend on the target error α (see (5)), and the gradient oracle parameters b 1 and b 2 are defined through η and χ , and thus also indirectly depend on α . In the proof of Theorem 2, by utilizing the relationship α = ˜ O ( √ σ 2 + r 2 d ) , we obtain the closed-form expression of α that depends solely on the problem parameters n , d , and ϵ . As a result, all algorithm parameters are ultimately determined by n , d , and ϵ , and there is no cyclic dependency in the parameter design.

1 For clarity, the bound stated here omits constant factors stemming from the Lipschitzness, smoothness, and Hessian Lipschitz assumptions. The complete expression, including these constants and their dependencies, is provided in the proof in Appendix. The same convention applies to Theorem 3.

<!-- image -->

## 6 Extension to Distributed SGD

By adapting the centralized gradient oracle Ada-DP-SPIDER (Algorithm 2) to the distributed setting, we obtain Distributed Ada-DP-SPIDER (Algorithm 3), enabling our Gauss-PSGD framework to extend seamlessly to distributed learning scenarios. The primary difference lies in the computation and communication scheme: in the distributed variant, each client performs local gradient estimation with private noise and communicates the privatized estimate to the server, which then aggregates the results. This avoids centralized access to raw data while still leveraging collective information.

The learning algorithm using Distributed Ada-DP-SPIDER can be viewed as an adaptive extension of the DIFF2 algorithm [37], which uses standard SPIDER and is limited to convergence to DP-FOSP under homogeneous data. To the best of our knowledge, our method is the first to achieve convergence to a DP-SOSP in a distributed setting with heterogeneous data.

Following the same analytical strategy as in Section 5, we first quantify in Lemma 9 the gradient estimation quality in the distributed case. The proof is provided in Appendix E.1.

<!-- formula-not-decoded -->

Based on this, we derive the error bound for α in the distributed setting. The proof is in Appendix E.2.

<!-- formula-not-decoded -->

```
Algorithm 4: Private Model Selection in Distributed Learning Input: Model iterates { x t } T t =1 , DP budget ϵ, δ 1 for t ← 1 , · · · , T do 2 for every client j in parallel do 3 Compute ∇ ¯ F j ( x t ) ←∇ ˆ f S j ( x t ) + θ i,t , where θ i,t ∼ N ( 0 , c 1 G 2 T log(1 /δ ) n 2 ϵ 2 I d ) ; 4 Compute ∇ 2 ¯ F j ( x t ) ←∇ 2 ˆ f S j ( x t ) + H j,t , where H j,t is a symmetric matrix with its upper triangle (including the diagonal) being i.i.d. samples from N ( 0 , c 2 M 2 dT log(1 /δ ) n 2 ϵ 2 ) and each lower triangle entry is copied from its upper triangle counterpart; 5 Send ∇ ¯ F j ( x t ) and ∇ 2 ¯ F j ( x t ) to the server; 6 ∇ ¯ F ( x t ) ← 1 m ∑ m j =1 ∇ ¯ F j ( x t ) , ∇ 2 ¯ F ( x t ) ← 1 m ∑ m j =1 ∇ 2 ¯ F j ( x t ) ; 7 if ∥∇ ¯ F ( x t ) ∥ 2 ≤ α + G log ( 8 d/ω ′ ) √ mn + G √ dT log(1 /δ ) log(16 /ω ′ ) √ mnϵ and λ min ( ∇ 2 ¯ F ( x t ) ) ≥ -( √ ρα + M √ log(8 d/ω ′ ) mn + Md √ T log(1 /δ ) log(32 /ω ′ ) √ mnϵ ) then 8 Return x t
```

Remark 4. The error rate shown in Theorem 3 highlights the collaborative synergy among clients, indicating the learning performance benefits from distributed learning. Specifically, the first nonprivate term of α exhibits a linear dependence on m before n , while the second term, which accounts for the privacy cost, demonstrates a square root dependence √ m before n . This separation reflects the impact of data heterogeneity in distributed setting. The benefit of distributed collaboration under DP constraints is consistent with prior results in heterogeneous federated learning [16].

We conclude by demonstrating the advantages of our Gauss-PSGD framework in distributed learning by eliminating the need for a separate private model selection procedure. Without the guarantee of directly outputting an α -SOSP, one must resort to evaluating all model iterates generated during the learning process and privately selecting an approximate SOSP from them. As discussed in Appendix A, the AboveThreshold mechanism used in [30] for the single-machine case is not applicable in distributed settings due to decentralized data access. To overcome this, we adapt [46, Algorithm 5] to the distributed setting, resulting in Algorithm 4. In this scheme, each client computes privatized gradients and Hessian estimates using additional local data, which are then aggregated by the server to evaluate the stationary point conditions. Suppose a distributed learning algorithm produces a sequence { x t } t ∈ [ T ] that contains at least one α -DP-SOSP. The following result characterizes the quality of the point selected by Algorithm 4, whose proof is provided in Appendix E.3:

Theorem 4. Algorithm 4 satisfies ( ϵ, δ ) -ICRL-DP. Let Assumption 1 hold and mn ≥ 4 9 log 8 d ω ′ , then with probability at least 1 -ω ′ , if there exists an α -SOSP x p ∈ { x t } T t =1 , then the selected point x o is an α ′ -SOSP with α ′ = ˜ O ( α + 1 mn + 1 √ mn + α √ mn + √ d √ mnϵα 5 / 4 + d √ mnϵα 3 / 4 + d 2 mn 2 ϵ 2 α 5 / 2 ) .

Remark 5. To ensure that the selected model's error α ′ does not exceed the training error α , the following must hold: √ d √ mnϵα 5 / 4 + d √ mnϵα 3 / 4 + d 2 mn 2 ϵ 2 α 5 / 2 ≤ ˜ O ( α ) . This implies a constraint on the model dimension: d ≤ min { ( √ mnϵ ) 2 , ( √ mnϵ ) 6 / 13 } . Thus, in high-dimensional regimes, private model selection degrades the overall error rate, marking the limitation of selection-based approaches.

Remark 6. The error bound α ′ in Theorem 4 can be improved by estimating the smallest eigenvalue of the Hessian via Hessian-vector products using iterative methods such as the power method [26]. This reduces the dimensional dependence in the noise scale from O ( d ) to O ( √ d ) . However, the remaining √ d factor is sill problematic in high-dimensional settings. In contrast, in the single-machine case, private model selection only requires perturbing scalar quantities, making the error independent of dimension, preserving the error guarantee of the learning algorithm. In distributed settings, sharing perturbed vectors becomes unavoidable. This emphasizes the necessity and superiority of our Gauss-PSGD framework that inherently avoids the need for any separate model selection step.

## Acknowledgments and Disclosure of Funding

Youming Tao was supported in part by the National Science Foundation of China (NSFC) under Grant 623B2068. Dongxiao Yu is supported in part by the Major Basic Research Program of Shandong Provincial Natural Science Foundation under Grant ZR2025ZD18. Xiuzhen Cheng is supported in part by the Major Basic Research Projects of Shandong Natural Science Foundation under Grant ZR2022ZD02. Di Wang is supported in part by the funding BAS/1/1689-01-01 and funding from KAUST - Center of Excellence for Generative AI, under award number 5940.

## References

- [1] Martin Abadi, Andy Chu, Ian Goodfellow, H Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC conference on computer and communications security , pages 308-318, 2016.
- [2] Raman Arora, Raef Bassily, Tomás González, Cristóbal A Guzmán, Michael Menart, and Enayat Ullah. Faster rates of convergence to stationary points in differentially private optimization. In International Conference on Machine Learning , pages 1060-1092. PMLR, 2023.
- [3] Dmitrii Avdiukhin, Michael Dinitz, Chenglin Fan, and Grigory Yaroslavtsev. Noise is all you need: Private second-order convergence of noisy sgd. arXiv preprint arXiv:2410.06878 , 2024.
- [4] Raef Bassily, Vitaly Feldman, Kunal Talwar, and Abhradeep Guha Thakurta. Private stochastic convex optimization with optimal rates. Advances in neural information processing systems , 32, 2019.
- [5] Raef Bassily, Cristóbal Guzmán, and Michael Menart. Differentially private stochastic optimization: New results in convex and non-convex settings. Advances in Neural Information Processing Systems , 34:9317-9329, 2021.
- [6] Raef Bassily, Adam Smith, and Abhradeep Thakurta. Private empirical risk minimization: Efficient algorithms and tight error bounds. In 2014 IEEE 55th annual symposium on foundations of computer science , pages 464-473. IEEE, 2014.
- [7] Sijin Chen, Zhize Li, and Yuejie Chi. Escaping saddle points in heterogeneous federated learning via distributed sgd with communication compression. In International Conference on Artificial Intelligence and Statistics , pages 2701-2709. PMLR, 2024.
- [8] Christopher A Choquette-Choo, Arun Ganesh, and Abhradeep Thakurta. Optimal rates for dp-sco with a single epoch and large batches. arXiv preprint arXiv:2406.02716 , 2024.
- [9] Ashok Cutkosky and Francesco Orabona. Momentum-based variance reduction in non-convex sgd. Advances in neural information processing systems , 32, 2019.
- [10] Hadi Daneshmand, Jonas Kohler, Aurelien Lucchi, and Thomas Hofmann. Escaping saddles with stochastic gradients. In International Conference on Machine Learning , pages 1155-1164. PMLR, 2018.
- [11] Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography: Third Theory of Cryptography Conference, TCC 2006, New York, NY, USA, March 4-7, 2006. Proceedings 3 , pages 265-284. Springer, 2006.
- [12] Cynthia Dwork, Aaron Roth, et al. The algorithmic foundations of differential privacy. Foundations and Trends® in Theoretical Computer Science , 9(3-4):211-407, 2014.
- [13] Cong Fang, Chris Junchi Li, Zhouchen Lin, and Tong Zhang. Spider: Near-optimal nonconvex optimization via stochastic path-integrated differential estimator. Advances in neural information processing systems , 31, 2018.
- [14] Cong Fang, Zhouchen Lin, and Tong Zhang. Sharp analysis for nonconvex sgd escaping from saddle points. In Conference on Learning Theory , pages 1192-1234. PMLR, 2019.

- [15] Vitaly Feldman, Tomer Koren, and Kunal Talwar. Private stochastic convex optimization: optimal rates in linear time. In Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing , pages 439-449, 2020.
- [16] Changyu Gao, Andrew Lowy, Xingyu Zhou, and Stephen J Wright. Private heterogeneous federated learning without a trusted server revisited: Error-optimal and communication-efficient algorithms for convex losses. arXiv preprint arXiv:2407.09690 , 2024.
- [17] Rong Ge, Furong Huang, Chi Jin, and Yang Yuan. Escaping from saddle points-online stochastic gradient for tensor decomposition. In Conference on learning theory , pages 797-842. PMLR, 2015.
- [18] Rong Ge, Zhize Li, Weiyao Wang, and Xiang Wang. Stabilized svrg: Simple variance reduction for nonconvex optimization. In Conference on learning theory , pages 1394-1448. PMLR, 2019.
- [19] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision , pages 1026-1034, 2015.
- [20] Lijie Hu, Shuo Ni, Hanshen Xiao, and Di Wang. High dimensional differentially private stochastic optimization with heavy-tailed data. In Proceedings of the 41st ACM SIGMODSIGACT-SIGAI Symposium on Principles of Database Systems , pages 227-236, 2022.
- [21] Prateek Jain, Chi Jin, Sham M Kakade, and Praneeth Netrapalli. Computing matrix squareroot via non convex local search. arXiv preprint arXiv:1507.05854 , 2015.
- [22] Chi Jin, Rong Ge, Praneeth Netrapalli, Sham M Kakade, and Michael I Jordan. How to escape saddle points efficiently. In International conference on machine learning , pages 1724-1732. PMLR, 2017.
- [23] Chi Jin, Praneeth Netrapalli, Rong Ge, Sham M Kakade, and Michael I Jordan. A short note on concentration inequalities for random vectors with subgaussian norm. arXiv preprint arXiv:1902.03736 , 2019.
- [24] Chi Jin, Praneeth Netrapalli, Rong Ge, Sham M Kakade, and Michael I Jordan. On nonconvex optimization for machine learning: Gradients, stochasticity, and saddle points. Journal of the ACM (JACM) , 68(2):1-29, 2021.
- [25] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
- [26] Cornelius Lanczos. An iteration method for the solution of the eigenvalue problem of linear differential and integral operators. 1950.
- [27] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE , 86(11):2278-2324, 1998.
- [28] Zhize Li. Ssrgd: Simple stochastic recursive gradient descent for escaping saddle points. Advances in Neural Information Processing Systems , 32, 2019.
- [29] Daogao Liu and Hilal Asi. User-level differentially private stochastic convex optimization: Efficient algorithms with optimal rates. In International Conference on Artificial Intelligence and Statistics , pages 4240-4248. PMLR, 2024.
- [30] Daogao Liu, Arun Ganesh, Sewoong Oh, and Abhradeep Guha Thakurta. Private (stochastic) non-convex optimization revisited: Second-order stationary points and excess risks. Advances in Neural Information Processing Systems , 36, 2024.
- [31] Ruixuan Liu, Yang Cao, Hong Chen, Ruoyang Guo, and Masatoshi Yoshikawa. Flame: Differentially private federated learning in the shuffle model. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 35, pages 8688-8696, 2021.
- [32] Andrew Lowy, Ali Ghafelebashi, and Meisam Razaviyayn. Private non-convex federated learning without a trusted server. In International Conference on Artificial Intelligence and Statistics , pages 5749-5786. PMLR, 2023.

- [33] Andrew Lowy and Meisam Razaviyayn. Private federated learning without a trusted server: Optimal algorithms for convex losses. In The Eleventh International Conference on Learning Representations , 2023.
- [34] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics , pages 1273-1282. PMLR, 2017.
- [35] Frank D McSherry. Privacy integrated queries: an extensible platform for privacy-preserving data analysis. In Proceedings of the 28th ACM SIGMOD International Conference on Management of data (SIGMOD) , pages 19-30, 2009.
- [36] Tomoya Murata and Taiji Suzuki. Escaping saddle points with bias-variance reduced local perturbed sgd for communication efficient nonconvex distributed learning. Advances in Neural Information Processing Systems , 35:5039-5051, 2022.
- [37] Tomoya Murata and Taiji Suzuki. Diff2: Differential private optimization via gradient differences for nonconvex distributed learning. In International Conference on Machine Learning , pages 25523-25548. PMLR, 2023.
- [38] Lam M Nguyen, Jie Liu, Katya Scheinberg, and Martin Takáˇ c. Sarah: A novel method for machine learning problems using stochastic recursive gradient. In International conference on machine learning , pages 2613-2621. PMLR, 2017.
- [39] Jinyan Su, Lijie Hu, and Di Wang. Faster rates of private stochastic convex optimization. In International Conference on Algorithmic Learning Theory , pages 995-1002. PMLR, 2022.
- [40] Jinyan Su, Lijie Hu, and Di Wang. Faster rates of differentially private stochastic convex optimization. Journal of Machine Learning Research , 25(114):1-41, 2024.
- [41] Jinyan Su, Changhong Zhao, and Di Wang. Differentially private stochastic convex optimization in (non)-euclidean space revisited. In Uncertainty in Artificial Intelligence , pages 2026-2035. PMLR, 2023.
- [42] Ju Sun, Qing Qu, and John Wright. A geometric analysis of phase retrieval. In 2016 IEEE International Symposium on Information Theory (ISIT) , pages 2379-2383. IEEE, 2016.
- [43] Youming Tao, Yulian Wu, Xiuzhen Cheng, and Di Wang 0015. Private stochastic convex optimization and sparse learning with heavy-tailed data revisited. In IJCAI , pages 3947-3953, 2022.
- [44] Joel A Tropp. User-friendly tail bounds for sums of random matrices. Foundations of computational mathematics , 12:389-434, 2012.
- [45] Roman Vershynin. High-dimensional probability. University of California, Irvine , 10:11, 2020.
- [46] Di Wang, Changyou Chen, and Jinhui Xu. Differentially private empirical risk minimization with non-convex loss functions. In International Conference on Machine Learning , pages 6526-6535. PMLR, 2019.
- [47] Di Wang, Marco Gaboardi, Adam Smith, and Jinhui Xu. Empirical risk minimization in the non-interactive local model of differential privacy. Journal of machine learning research , 21(200):1-39, 2020.
- [48] Di Wang, Marco Gaboardi, and Jinhui Xu. Empirical risk minimization in non-interactive local differential privacy revisited. Advances in Neural Information Processing Systems , 31, 2018.
- [49] Di Wang, Hanshen Xiao, Srinivas Devadas, and Jinhui Xu. On differentially private stochastic convex optimization with heavy-tailed data. In International Conference on Machine Learning , pages 10081-10091. PMLR, 2020.
- [50] Di Wang and Jinhui Xu. Escaping saddle points of empirical risk privately and scalably via dptrust region method. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2020, Ghent, Belgium, September 14-18, 2020, Proceedings, Part III , pages 90-106. Springer, 2021.

- [51] Di Wang, Minwei Ye, and Jinhui Xu. Differentially private empirical risk minimization revisited: Faster and more general. Advances in Neural Information Processing Systems , 30, 2017.
- [52] Hanshen Xiao, Zihang Xiang, Di Wang, and Srinivas Devadas. A theory to instruct differentiallyprivate learning via clipping bias reduction. In 2023 IEEE Symposium on Security and Privacy (SP) , pages 2170-2189. IEEE, 2023.
- [53] Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett. Defending against saddle point attack in byzantine-robust distributed learning. In International Conference on Machine Learning , pages 7074-7084. PMLR, 2019.
- [54] Yingxue Zhou, Xiangyi Chen, Mingyi Hong, Zhiwei Steven Wu, and Arindam Banerjee. Private stochastic non-convex optimization: Adaptive algorithms and tighter generalization bounds. arXiv preprint arXiv:2006.13501 , 2020.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly state the main contributions-namely, the development of a PSGD-based framework that corrects prior analytical errors, eliminates reliance on private model selection, and extends to distributed learning with heterogeneous data. We also provide a list of our core contributions explicitly in our introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We provide the limitation discussion in Section H in the Appendix, where we highlight the theoretical assumption of unbiased gradient oracles and discuss its potential divergence from practical DP optimizers. We also outline the challenges and directions for extending the framework to handle biased gradient estimates.

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

Justification: We state all assumptions for our theoretical results in Section 2. For each theoretical result (lemma, theorem, etc.), we explicitly indicate the assumptions it relies on and provide a complete proof, with the location of each proof clearly referenced in the Appendix.

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

Justification: We provide a comprehensive description of our experimental setup, including running environments, datasets, learning models, hyperparameter settings, and evaluation metrics, in Section F of the Appendix. This ensures that the main experimental results are reproducible and support the core claims of the paper.

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

Justification: The datasets used in our experiments are publicly available. While our code is not yet released at the time of submission, we plan to open-source it with detailed instructions to reproduce all experimental results as described in the Appendix.

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

Justification: We provide all necessary experimental details in the experiment section (Section F in Appendix). These details ensure that the experimental setup and results can be fully understood and independently reproduced.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We reported error bars in our experimental results.

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

Justification: We specify all the computational resources used for our experiments in the experiment section (Section F in Appendix).

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our research fully complies with the NeurIPS Code of Ethics. We have carefully reviewed and ensured adherence to all relevant standards.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

## Answer: [Yes]

Justification: We provide a Broader Impact Statement in Section G, discussing both the potential positive impacts-such as enabling trustworthy and privacy-preserving machine learning in sensitive domains like healthcare and finance-and the broader limitations of differentially private learning, including potential reductions in model accuracy. This balanced discussion reflects both societal benefits and possible drawbacks.

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

Answer: [NA]

Justification: The paper does not use existing assets.

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

Justification: The paper does not involve research with human subjects.

Guidelines:

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