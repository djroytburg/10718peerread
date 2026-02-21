## Decreasing Entropic Regularization Averaged Gradient for Semi-Discrete Optimal Transport

Ferdinand Genans 1 ∗ Antoine Godichon-Baggioni 1 François-Xavier Vialard 2 Olivier Wintenberger 1 , 3

Sorbonne Université, CNRS, LPSM 1 Université Gustave Eiffel, CNRS, LIGM 2 Wolfgang Pauli Institute 3

{ferdinand.genans-boiteux, antoine.godichon\_baggioni, olivier.wintenberger}@sorbonne-universite.fr francois-xavier.vialard@u-pem.fr

## Abstract

Adding entropic regularization to Optimal Transport (OT) problems has become a standard approach for designing efficient and scalable solvers. However, regularization introduces a bias from the true solution. To mitigate this bias while still benefiting from the acceleration provided by regularization, a natural solver would adaptively decrease the regularization as it approaches the solution. Although some algorithms heuristically implement this idea, their theoretical guarantees and the extent of their acceleration compared to using a fixed regularization remain largely open. In the setting of semi-discrete OT, where the source measure is continuous and the target is discrete, we prove that decreasing the regularization can indeed accelerate convergence. To this end, we introduce DRAG: Decreasing (entropic) Regularization Averaged Gradient, a stochastic gradient descent algorithm where the regularization decreases with the number of optimization steps. We provide a theoretical analysis showing that DRAG benefits from decreasing regularization compared to a fixed scheme, achieving an unbiased O (1 /t ) sample and iteration complexity for both the OT cost and the potential estimation, and a O (1 / √ t ) rate for the OT map. Our theoretical findings are supported by numerical experiments that validate the effectiveness of DRAG and highlight its practical advantages.

## 1 Introduction

Optimal transport is now a widely used framework to compare probability distributions in different areas of data science such as machine learning [14, 25, 6], computational biology [47], imaging [21, 7], even economics [22] or material sciences [9]. The computational and statistical efficiency of OT solvers is the key to facilitating their use in practical applications. Therefore, both computational methods and the statistical bottleneck in OT, often referred to as the curse of dimensionality, have received significant attention over the past decade [42, 57]. Regularization methods such as Entropic OT (EOT) [16] are popular techniques to mitigate these two issues. It consists of adding an entropic regularization term to the objective function. OT and its entropic regularization apply to different contexts of interest. The most general context is when the two distributions are accessed via samples and one wants to estimate the OT distance and the corresponding plan or map. Another context of interest in some applications, that has recently gained popularity in generative modeling [2, 10, 36], is the case of semi-discrete OT. In this setting, one of the two distributions is discrete and the other continuous. The OT problem, while being a natural proxy to the continuous case, is then slightly simpler since (i) it reduces to the estimation of Laguerre cells and (ii) the curse of dimensionality is alleviated [45].

Related works. The incorporation of entropic regularization for OT was pioneered in the discrete setting by Cuturi [16], showing that the Sinkhorn algorithm [53] can efficiently solve the EOT problem. Sinkhorn leads to an ε -accurate OT plan in O ( n 2 /ε 2 ) time when both measures have n points [19]. However, the poor dependence on ε , observed both theoretically and empirically, motivated annealing strategies (also called ε -scaling), which gradually decrease the regularization during optimization to better approximate the true OT solution. Such schemes appear to significantly improve performance in practice [34, 48, 20], but their theoretical analysis remains largely open [52, 48, 12]. From a statistical perspective, it has recently been shown that decreasing the regularization is not only computationally beneficial but also statistically necessary when using EOT: [45] demonstrate in the semi-discrete setting that with t samples from the source and target measures, taking ε ≍ 1 / √ t allows one to achieve a minimax-optimal O (1 / √ t ) rate for OT map estimation, thereby escaping the curse of dimensionality without assuming smoothness of the transport map. Their analysis leverages convergence results for entropic potentials [1, 18] and builds upon the entropic map estimator developed in [51, 44].

In parallel, there has been increasing interest in solving semi-discrete OT problems, where the source distribution is continuous while the target measure is known and discrete [2, 54]. In low dimensions, when the source density is fully known, Newton-type solvers [38, 35, 31] have been proposed, offering highly efficient methods for solving the semi-dual problem. In higher dimensions, or when the source measure is only accessible via samples [24] propose solving the semi-dual formulation of semi-discrete (E)OT using an SGD scheme. SGD solvers are a natural choice here since they are well-suited for large-scale applications: they can operate in an online setting using one sample at a time without storage requirements, and they avoid discretization bias. The study of SGD and Averaged SGD (ASGD) for solving the semi-dual of EOT was further investigated in [5], revealing, however, prohibitive constants in terms of 1 /ε and higher. This study shows that, as in the discrete case, the use of entropic regularization introduces a computational trade-off.

Contributions. We introduce DRAG (Decreasing Regularization Averaged Gradient), an SGDbased algorithm for solving the non-regularized semi-discrete OT. DRAG employs a decreasing entropic regularization scheme decaying with the sample count, aiming to have the best regularization/accuracy trade-off at any time. While matching the computational and memory efficiency of vanilla SGD, DRAG attains superior convergence compared to fixed-regularization methods by exploiting the enhanced properties of the entropic semi-dual without incurring adverse dependencies on the vanishing regularization. Concretely, given t iid samples from the source measure, DRAG achieves rates up to O (1 /t ) for both the OT cost and the potential. A key technical result (Lemma 2) underlying DRAG is that, within an ε -ball around the optimum, the entropic semi-dual with regularization ε satisfies a restricted strong-convexity property independent of ε . DRAG is designed to remain, with high probability, within this ε -ball, even as ε decreases. Furthermore, by analyzing the difference between the Laguerre cells induced by the true OT potential and our estimate, we establish a O (1 / √ t ) convergence rate for the OT map. Both our theoretical analysis and numerical experiments confirm the benefits of decreasing regularization incorporated in DRAG.

Notations. We note ∥ · ∥ the Euclidean norm, and for C ⊂ R d , D C := sup {∥ x -y ∥ : x, y ∈ C} denote its diameter. For a, b ∈ R , a ∨ b := max { a, b } and a ∧ b := min { a, b } . For v ∈ R d , v min := min 1 ≤ j ≤ d v j . 1 d and 0 d denote the vectors (1 , . . . , 1) and (0 , . . . , 0) in R d . λ R d is the Lebesgue measure in R d . P ( R d ) is the set of probabilities in R d , and for ρ ∈ P ( R d ) , Supp( ρ ) is its support. O ( · ) and o ( · ) are the usual approximation orders. We use f ≲ g if there exists a constant C &gt; 0 such that f ( · ) ≤ Cg ( · ) . We write a ≍ b if both a ≲ b and b ≲ a .

## 2 Behind Stochastic Approximation for Optimal Transport

## 2.1 Background on (Entropic) Optimal Transport

Given a source and target probability measures µ, ν ∈ P ( R d ) , a cost function c : R d × R d → R + and a regularization parameter ε ≥ 0 , the Entropic Optimal Transport (EOT) problem is where Π( µ, ν ) is the set of joints probability measures on R d × R d with marginals µ and ν . Mild conditions on µ, ν and the cost can be made so that this problem is well-posed, see [56]. When ε = 0 ,

<!-- formula-not-decoded -->

Problem (1) recovers the Kantorovich formulation of OT. In this article, we focus on the quadratic cost c ( x, y ) = 1 2 ∥ x -y ∥ 2 , although some of our results can be extended to other costs. Our analysis relies on the semi-dual formulation of the convex problem (1) given by

<!-- formula-not-decoded -->

where for all y ∈ R d ,

<!-- formula-not-decoded -->

Under mild conditions on the cost or densities, a positive ε makes the semi-dual formulation 1 /ε -smooth [17]. The key property of this semi-dual formulation of (E)OT is to retain more convexity than the standard dual of (1) (see [28, 55]).

Optimal map and Brenier's theorem. We consider the quadratic cost, ε = 0 and µ, ν having second-order moments. Under the additional assumption that the measure µ is absolutely continuous, the optimal potential f ∗ , called Kantorovich potential, is (locally) Lipschitz and the map

<!-- formula-not-decoded -->

pushes forward µ onto ν (see [8]). In addition, T µ,ν is the gradient of a convex function. This optimal map has more importance than the OT cost in subfields of machine learning such as generative modeling [29, 36] or domain adaptation [15].

## 2.2 Semi-discrete OT

Semi-discrete (E)OT is when the source measure µ is absolutely continuous and the target measure ν = ∑ M j =1 w j δ y j is a finite sum of M ≥ 1 Dirac masses with weights w j &gt; 0 . In this case, the semi-dual formulation reduces to a finite-dimensional convex optimization problem on R M

<!-- formula-not-decoded -->

where for all x ∈ R d , g c,ε ( x ) is a (vectorial) ( c, ε ) -transform with respect to a vector g = ( g 1 , . . . , g M ) ∈ R M , defined by

<!-- formula-not-decoded -->

The vector g corresponds to the value of the potential function at the points y j . For notational convenience, we write H ε ( g ) = ∫ R d h ε ( x, g )d µ ( x ) with h ε ( x, g ) = -g c,ε ( x ) -∑ M j =1 g j w j . For all g ∈ R M and given X ∼ µ , an unbiased estimator of the gradient is given by

<!-- formula-not-decoded -->

where for ( x, g ) ∈ R d × R M , we have

<!-- formula-not-decoded -->

For ε = 0 , χ j ( x, g ) = ✶ L j ( g ) ( x ) is an indicator function and we have a partition R d = ⋃ M j =1 L j ( g ) , where for all j ∈ J 1 , M K ,

̸

<!-- formula-not-decoded -->

The convex sets L j ( g ) are called power or Laguerre cells and µ ( L i ( g ) ∩ L j ( g )) = 0 when i = j . By the first-order optimality condition, solving semi-discrete OT amounts to finding g such that for all j ∈ J 1 , M K , µ ( L j ( g )) = w j . Semi-discrete OT is a case of application of Brenier's theorem. Given the optimal potential g ∗ , the OT map is T µ,ν ( x ) = x -∇ ( g ∗ ) c ( x ) = y j for x inside L j ( g ∗ ) .

## 2.3 Solving semi-discrete (E)OT with the semi-dual formulation

Exploiting its finite-dimensional nature, solving semi-discrete OT by optimizing its semi-dual formulation has become a popular approach. Notably, Newton methods are highly effective to solve H 0 in scenarios with low dimensions and known source densities, utilizing meshes to approximate the source density [38, 35, 31]. In scenarios involving arbitrary dimensions or when only sample-based access to the source measure is available, EOT emerges as a favored strategy. Notably, to avoid working with a discretized version of the source measure, such as with the Sinkhorn Algorithm, [24] recommend employing stochastic optimization to solve (4). Indeed, the semi-dual EOT problem has a convex objective of the form

<!-- formula-not-decoded -->

with X as a random variable under µ . As noted in [24], the main advantage of stochastic optimization algorithms is that they are suited for really large-scale problems, keeping in memory only the discrete measure ν . Moreover, avoiding discretization enables unbiased estimation of (E)OT quantities. SGD-based solvers also naturally operate in an online fashion, progressively refining their solutions as more samples become available.

For a given fixed regularization parameter ε &gt; 0 , stochastic first-order methods are predominantly employed to solve (4). Starting with an initial value g 0 ∈ R M , these algorithms consider at each iteration one or many samples X t ∼ µ and rely on an update of the form

<!-- formula-not-decoded -->

At time t , the Averaged Stochastic Gradient Descent (ASGD) returns the averaged estimate g t = 1 t +1 ∑ t k =0 g k , while Stochastic Gradient Descent (SGD) returns g t . ASGD, as an acceleration of SGD, has been widely studied in the literature (see [43, 41, 4], and [5] for the specific case of EOT).

Choosing the regularization parameter ε for EOT. Approximating the EOT problem rather than the OT one benefits from an enhanced convergence rate, especially in the discrete setting. The introduction of the Sinkhorn Algorithm for solving the EOT problem, as highlighted by [16], has led to a resurgence of interest in OT within the Machine Learning community.

The choice of the regularization parameter ε then becomes a practical and/or statistical problem:

1. Selecting the regularization parameter is a practical issue that aims to strike an optimal balance between convergence speed and accuracy [16, 19]. To address this trade-off, some heuristics, such as ε -scaling [49], which involves a decreasing regularization scheme, are employed in the discrete setting, although they lack sharp theoretical guarantees.
2. In the semi-discrete and continuous settings, the initial statistical problem is to determine the number of samples needed to accurately approximate the OT quantities. In this line of work, the use of EOT to construct estimators has also been proven to be satisfactory. In this case, studies show that regularization must decrease as the number of samples increases [44, 45]. However, discrete solvers do not adjust to the number of drawn points, as the solver is initiated once the points to approximate the measures have been sampled.

## 3 DRAG: Decreasing Regularization Averaged Gradient

## 3.1 The setting.

We focus here on the one-sample setting of semi-discrete OT. Specifically, we sample from the source measure µ and leverage the full information of the discrete measure ν . Furthermore, fixing R &gt; 0 and α ∈ (0 , 1] , we make the following mild assumption, already present in [18, 45].

The target ν is assumed to be discrete, of the form ν = 1 M ∑ M j =1 δ y j , with ( y 1 , . . . , y M ) ∈ B (0 , R ) M .

Assumption 1. We assume that µ ∈ P ( R d ) has support contained in the convex ball B (0 , R ) and admits a density dµ that is α -Hölder continuous and satisfies 0 &lt; dµ &lt; ∞ on its support. We denote by P α ( B (0 , R )) the set of such measures.

## 3.2 DRAG: A gradient-based algorithm adaptive to both the sample size and the regularization parameter

To accurately estimate the non-regularized OT cost and map, it is crucial to use a regularization parameter ε that decreases as the number of drawn samples increases. However, no existing algorithm in the OT literature simultaneously adapts to both entropic regularization and sample size. Inspired by ε -annealing [49], a decreasing regularization scheme from the discrete OT setting, which is known for accelerating the convergence of the Sinkhorn algorithm in practice, and considering that SGD algorithms are inherently adaptive to the number of samples, we introduce the Decreasing entropic Regularization projected Averaged stochastic Gradient descent (DRAG) to solve the semi-dual (2). Our algorithm employs a decreasing regularization sequence ( ε t ) t and replaces the usual gradient step in SGD with a projected step using adaptive regularization

<!-- formula-not-decoded -->

where for U ⊂ R M convex, we define the projector as Proj U ( g ) := arg min {∥ g -g ′ ∥ , g ′ ∈ U } . This method can be interpreted as a decreasing bias SGD scheme. For such a method, employing a projection step can be highly effective in ensuring convergence [13, 23]. In the context of EOT with bounded cost, it is well established that the ( c, ε ) -transform enables the localization of a ∥ . ∥ ∞ -ball, where a minimum of the semi-dual problem lies [40]. Specifically, since sup { c ( x, y j ); x ∈ Supp( µ ) , j ∈ J 1 , M K } &lt; 2 R 2 by Assumption 1, a preliminary projection set can be expressed as C ∞ := [0 , 2 R 2 ] M and we know that we can search for a minimum in this set. Nonetheless, leveraging the regularity of the cost function, we can have a projection set with a unique optimizer, as described in the following Lemma.

Lemma 1. (Proof in Appendix B.6) Under Assumption 1, for all ε ≥ 0 , there exists a unique solution g ∗ ε to (4) in C u := { g ∈ R M ; g 1 = 0 and | g j | ≤ R ∥ y 1 -y j ∥ , j ∈ J 1 , M K } .

Note that the choice g 1 = 0 is arbitrary. In what follows, we refer to C = C ∞ or C = C u as our projection set. Note that for both sets, the projection's computational complexity is only O ( M ) , as it involves merely clipping each coordinate of our vector.

Finally, we consider the Decreasing Regularization projected Averaged stochastic Gradient descent (DRAG) defined by

<!-- formula-not-decoded -->

with g 0 = g 0 . The pseudo-code of our algorithm is given in Algorithm 1. A main advantage of DRAG is its O ( dtM ) computational complexity and O ( dM ) spatial complexity, which make it well suited for large-scale problems.

## Algorithm 1 DRAG

<!-- formula-not-decoded -->

## 3.3 Key properties of the semi-dual H ε for the design of DRAG

The design and convergence analysis of DRAG rely on two fundamental properties of the semi-dual objective H ε . First, the fast convergence of entropic potentials ensures that the optimal solution does not change abruptly as the regularization parameter varies. Second, the enhanced Restricted Strong Convexity (RSC) of H ε around its optimum. These two properties are crucial to the construction of DRAG and are detailed below.

Convergence of the entropic potentials. The following result from [18] establishes that the convergence of entropic optimal potentials is faster than linear as ε ′ → ε . Note that this result is only given for the quadratic cost and is, therefore, the limiting factor in our analysis for broadening the class of cost functions for DRAG.

Proposition 1. [Corollary 2.2 [18]] For 0 ≤ ε ′ ≤ ε , under Assumption 1 , there exists a constant K 0 , notably depending on the characteristics of ν , such that for any α ′ ∈ (0 , α ) ,

<!-- formula-not-decoded -->

The constant K 0 can depend on the source measure µ , through the radius R and the constants m 1 , m x 2 such that m 1 &lt; d µ &lt; m 2 on its support, see Remark 2.1 in [18]. However, terms depending on K 0 will be asymptotically negligible in the analysis of DRAG.

Note that, the semi-dual H has the invariance H ( g ) = H ( g + a 1 M ) for any g ∈ R M and a ∈ R . Therefore, the minimizer g ∗ of the semi-dual is unique only up to a transformation of the form g ∗ ε t + a 1 M , where a ∈ R ∗ . Consequently, our analysis on the orthogonal complement of the subspace spanned by 1 M , denoted as Vect( 1 M ) ⊥ . For simplicity, for g , g ′ ∈ R M , we denote for p ∈ [1 , ∞ ]

<!-- formula-not-decoded -->

Global and local Restricted Strong Convexity. The convergence behavior of gradient-based methods is a central topic in convex optimization. The Restricted Strong Convexity (RSC) condition [58] offers a strictly weaker alternative to strong convexity while still providing comparable guarantees in many settings [58, 50]. The following lemma characterizes the RSC of H ε . Notably, while the global RSC constant on C scales linearly with ε -1 , the local RSC in a neighborhood of radius ε/ 2 around the optimum becomes independent of ε . This motivates the decreasing regularization scheme in DRAG: by gradually reducing ε , we ensure that iterates remain within regions where the improved convexity properties can be fully exploited.

Lemma 2 (Global and local RSC of H ε , proof in Appendix B.7) . For any ε ∈ (0 , 1] , under Assumption 1, there exists ρ ∗ &gt; 0 independant of ε , such that for all g ∈ C ,

<!-- formula-not-decoded -->

Here, ρ ∗ provides a lower bound on the strong convexity constant of H ε restricted to the subspace Vect ( 1 ⊥ ) , and it holds uniformly over ε ∈ (0 , 1] (see Theorem 3.2 in [18] for further details).

## 3.4 Convergence rate of DRAG

Convergence rate before averaging. The following proposition provides a key high-probability control, ensuring that the iterates g t remain uniformly close to the optimal potential g ∗ ε t at all times t .

Proposition 2. (Proof in Appendix B.5) Under Assumption 1 with µ ∈ P α ( B (0 , R )) , taking the parameters ( γ 1 , a, b ) of DRAG such that γ 1 &gt; 0 , b ∈ ( 1 2 , 1 ) , with constraints 2 a &lt; b, a + b &lt; 1 , 1 + a + aα &gt; 2 b, we have for any δ &gt; 0 and every q &gt; 0 ,

<!-- formula-not-decoded -->

This result is key to leveraging the locally enhanced RSC of H ε t and guides how quickly the regularization can decay. When b &gt; 2 a , it yields a convergence rate of o ( t p ) for all p . This proposition leads to the convergence rate of the non-averaged DRAG iterates stated in Theorem 1.

Dependence on a , b , and α . As we can see, the convergence rate depends on a, b from DRAG and the Hölder regularity. While the constraints may seem difficult to interpret, setting a arbitrarily close to 1 3 (denoted a = 1 3 -) and b = 2 3 ensures that the constraints are satisfied for any α ∈ (1 / 2 , 1] and the best converge rate for DRAG in our convergence analysis.

Theorem 1. (Proof in Appendix B.1) Under the same assumptions as in Proposition 2, we have for any α ′ ∈ (0 , α )

<!-- formula-not-decoded -->

Remarkably, we achieve a convergence rate without any undesirable dependence on regularization. In contrast, [5] derived a convergence rate of the form O ( ε -c t -b ) for a fixed regularization, with c at least equal to 1 . Note that having no adverse dependence on the regularization parameter is a key necessary characteristic of DRAG, which aims to solve the non-regularized OT problem at fast rates. This is indeed the case here, and there is no trade-off with the choice of the admissible a and b since the result holds for all q &gt; 0 .

Enhanced convergence rate with averaging. In convex stochastic optimization, it is known that averaging SGD iterations can lead to acceleration. More precisely, ASGD can adapt to the possibly unknown local strong convexity of the objective function at the optimizer [43, 3] and achieve optimal O (1 /t ) converge rates. Despite the fact that our objective function changes at each time t , Theorem 2 shows that DRAG fully exploits the acceleration thanks to averaging.

Theorem 2. (Proof in Appendix B.2) Under the same assumptions as in Theorem 1, noting s = min { 1 , 2 a +2 aα ′ } , we have for all α ′ ∈ (0 , α ) ,

<!-- formula-not-decoded -->

The rates again depend on a , b , and α . The key message here is that by taking a = 1 3 -and b = 2 3 , we recover the optimal O (1 /t ) rate for any α ∈ ( 1 2 , 1] .

## 4 Optimal Transport cost and map estimation with DRAG

In the previous section, we established the convergence rate of DRAG to the OT potential. While this result was central to our theoretical analysis, our final objective is to estimate the OT cost and transport map. Leveraging the convergence rate of the potential, we derive estimation guarantees for these key OT quantities.

## 4.1 OT cost estimation

Corollary 1. (Proof in Appendix B.4) Taking the same assumptions as Theorem 2, with s = min { 1 , 2 a +2 aα ′ } , we have

<!-- formula-not-decoded -->

Once again, when α &gt; 1 / 2 and setting a = 1 3 -and b = 2 3 , we achieve an O (1 /t ) convergence rate, which is optimal for strongly convex objectives. This rate is obtained by leveraging the locally enhanced RSC property and our adaptive decreasing regularization scheme. The fact that H 0 is locally smooth, as noted in Theorem 4.1 of [32], also plays a crucial role in the proof of Corollary 2 .

## 4.2 Brenier map estimation

When employing entropic regularization, a popular choice to approximate the OT map involves using the estimator of the entropic Brenier map [44] T ε µ,ν ( g ∗ ε )( x ) = x -∇ ( g ∗ ε ) c,ε . Indeed, for ˆ g ∈ R M , T ε µ,ν (ˆ g )( x ) could serve as an estimator. The objective is then to find an accurate estimator, ˆ g , close to g ∗ ε , and to analyze its performance based on the bias-variance decomposition

<!-- formula-not-decoded -->

using the fact that ∥ T µ,ν -T ε µ,ν ( g ∗ ε ) ∥ 2 L 2 ( µ ) ≲ ε ([45], Theorem 3.4). However, the mapping g ↦→ T ε µ,ν ( g ) is ε -1 -Lipschitz, complicating the bias-variance trade-off given that ε t = t -a . Instead, we rely on the gradient computed thanks to the c -transform of the estimator g t of DRAG. In fact, for any x ∈ R d , if there exists j ∈ J 1 , M K such that x is in the interior of L j ( g ∗ ) ∩ L j ( g t ) , we have T µ,ν ( x ) = x -∇ ( g t ) c ( x ) . Indeed, no matter g , whenever x ∈ R d is in the interior of L j ( g ) , the gradient of g c is given by

<!-- formula-not-decoded -->

By analyzing the differences of Laguerre cells partitions between L ( g t ) and L ( g ∗ ) , we derive the following theorem.

Theorem 3. (Proof in Appendix B.3) Under the same assumptions as Theorem 1, defining for all x ∈ R d and time t ≥ 0 T ( g t )( x ) = x -∇ g c t , s = min { 1 , 2 a +2 aα ′ } , we have for all 1 ≤ p &lt; ∞

<!-- formula-not-decoded -->

When α &gt; 1 / 2 , setting a = 1 3 -and b = 2 3 , we recover an O (1 / √ t ) convergence rate. This matches the rate obtained in [45], but in our case it is achieved in the one-sample setting with an algorithm that refines its estimate online, whereas their approach relies on the Sinkhorn algorithm in a batched setting. Note that, unlike our method, theirs achieves the optimal rate for any α ∈ (0 , 1) . However, the use of online algorithms is crucial in high-dimensional applications where data is sampled sequentially, such as in generative modeling. In such settings, although the source measure is not compact, it is often chosen to be the standard Gaussian, which has a Lipschitz density and therefore corresponds to the case α = 1 .

## 5 Numerical experiments

Convergence rates of DRAG on synthetic data. We numerically verify here our convergence rate guarantees through various examples. For each example, we know the theoretical OT map, cost, and discrete potential. The first two examples are similar to those in [45]. In all figures and experiments, we set the parameters of DRAG to ε t = 0 . 1 /t a , a = 0 . 33 , b = 2 / 3 , γ 1 = Diam ( Supp ( µ )) . Our numerical investigation found that our parameter selection is robust without further hypertuning.

Examples settings: (1) µ ∼ U ([0 , 1] 10 3 ) , Supp( ν ) = { y j = ( j -1 / 2 M , 1 2 , ..., 1 2 ) , j ∈ J 1 , M K } , w = 1 M 1 M , M = 1000 . (2) µ ∼ U ([0 , 1] 10 3 ) , M = 30 and y 1 , ..., y M randomly generated in [0 , 1] 10 3 . We then also randomly generate g ∗ ∈ R 30 and approximate w with Monte Carlo (MC), such that g ∗ is the discrete optimum potential. This setting led to non-uniform weights, with w min = 0 . 001 . (3) µ ∼ U ([ δ, 1 + δ ]) , δ = 0 . 5 , Supp( ν ) = { k M ; k ∈ J 1 , M K } , w = 1 M 1 M , M = 1000 .

Figure 1: Convergence rate to the OT potential, cost and map for Examples 1 , 2 and 3 .

<!-- image -->

In Figure 1, we show the convergence rates of the OT cost, map, and discrete potential. As we can see, our theoretical rates are matched for all OT quantities. The higher variance in the OT cost estimations in Example 3 is likely due to the use of 10 8 Monte Carlo samples to approximate H , which introduces an additional approximation error beyond the one caused by DRAG alone.

Dependence on the dimension. For DRAG, the ambient dimension of the measures does not change the convergence rate but the number of points M does. This is due to the fact that, with the semi-dual formulation, we solve an expected minimization problem, and the dimension is then the number of points of our discrete measure. For instance, changing the dimension from d = 10 to d = 1000 does not change the convergence rate. However,if one wants to approximate continuous OT with semi-discrete OT and DRAG, the accuracy of the approximation of the continuous measure with M points will depend on the dimension and thus will have an impact on DRAG.

DRAG compared to fixed regularization ASGD. In Figures 2 and 3, we compare the effectiveness and robustness of decreasing vs. fixed regularization schemes. Figure 2 shows that DRAG consistently outperforms projected ASGD with various fixed regularization values, achieving a better trade-off between convergence speed and solution quality. Fixed schemes either converge to biased solutions when regularization is large or fail to converge in time when it is too small (e.g., ε = 5 · 10 -3 ). This highlights DRAG's advantage during the entire optimization process and supports the idea that starting with high regularization and gradually reducing it yields more stable and accurate solutions in semi-discrete optimal transport. Figure 3 shows DRAG's robustness to the decay parameter a : both a = 0 . 3 and a = 0 . 5 yield similar convergence, indicating low sensitivity. All decreasing regularization variants also clearly outperform the non-regularized projected ASGD. While all

regularization schemes eventually converge with more iterations, DRAG remains one to two orders of magnitude more accurate, due to its improved start visible in both figures (see Appendix A).

<!-- image -->

Iterations

Figure 2: DRAG compared to ASGD with a fixed regularization on Example 1 .

<!-- image -->

Iterations

Figure 3: DRAG, with different decreasing regularization rate ε t = 0 . 1 /t a on Example 1 .

Generative modeling task. We illustrate the practical benefits of our solver in the context of generative modeling. In [2], semi-discrete OT is used to map a simple prior onto encoded data points in latent space, with the goal of reducing mode collapse.

To generate new samples, they approximate a semidiscrete OT map from a standard gaussian to the empirical distribution of encoded data points in the latent space and then apply a specific interpolation scheme to obtain a continuous mapping from prior to latent space. We replicate their pipeline on toy datasets from their repository [27], replacing their ADAMbased solver with DRAG, using the same number of samples. As we can see, while both solvers yield good results on the "swissroll" target data, DRAG outperforms the ADAM solver on the "spiralarms" data, being able to almost completely generate it, whereas ADAM shows poorer coverage. Since the Gaussian is not compactly supported, this setup falls

Figure 4: Comparison of DRAG and ADAM for a generative model task

<!-- image -->

under Assumption 1, and DRAG was run without projections. This further underscores its robustness in more general settings.

Monge-Kantorovich Quantiles. We visualize our OT map estimator on a concrete example of Monge-Kantorovich (MK) quantiles [11]. In this context, having a target measure ν to investigate, the source measure is set to be the uniform measure on the unit Euclidean ball µ ∼ U ( B (0 , 1)) . Given the OT map T µ,ν (or its approximation), T µ,ν ( B (0 , k/ 10) ) for k ∈ J 1 , 10 K define MK quantile regions. We used M = 10 5 points to approximate ν , a discrete version of a boomerang-shaped measure and benchmarked DRAG against two OT solvers that can solve semi-discrete OT: Online Sinkhorn [37], using the EOT map estimator and Neural OT [33]. DRAG and Online Sinkhorn used 10 7 source samples; for the latter, the entropic regularisation was tuned to ε = 10 -3 . Both ran in under one minute. Neural OT, following Appendix B of [33], processed over 10 8 samples using a three hidden layers MLP, was ten times slower, even on a A100 GPU. Figure 5 displays the estimated MKquantile regions of the target measure ν , color-coding each centered annulus region. As visible in the figure, DRAG is the only method producing an unbiased estimate of the MK quantiles, fully covering the support of ν while keeping every MK region convex, as expected in theory.

Figure 5: Comparison of Monge-Kantorovich quantiles approximation with different solvers

<!-- image -->

Additional experiments. The appendix presents further experiments that, while not affecting our theoretical results, may benefit practitioners. We show that mini-batching with GPU acceleration and weighted averaging of iterates g t can significantly speed up the algorithm. We also compare DRAG to Adam on synthetic data, highlighting its superior performance.

## 6 Conclusion

In EOT, a decreasing regularization parameter naturally appeals to practitioners who employ annealing schemes to accelerate Sinkhorn-like algorithms. Similarly, in the statistical community, regularization that decreases with the number of samples is preferred for more accurately approximating true OT quantities. With our algorithm, DRAG, we show that these two motivations for decreasing regularization can successfully coexist by proving that a decreasing entropic regularization scheme can indeed improve the convergence rate, whereas it was only a heuristic beforehand. Moreover, we prove that DRAG achieves optimal convergence rates: O (1 /t ) for both the OT potential and cost, and O (1 / √ t ) for the OT map. These rates are obtained by leveraging decreasing regularization as a form of acceleration. To the best of our knowledge, this is the first algorithm in the OT literature that adapts to both regularization strength and sample size. Our results also motivate further investigation of decreasing regularization in (i) discrete OT, by adapting our approach to demonstrate the acceleration benefits of annealing schemes, and in (ii) semi-discrete OT, by developing new optimized versions of DRAG, such as those incorporating adaptive step sizes in a similar way to Adam or Adagrad.

## 7 Acknowledgements

The work of François-Xavier Vialard is partly supported by the Bézout Labex (New Monge Problems), funded by ANR, reference ANR-10-LABX-58.

## References

- [1] J. M. Altschuler, J. Niles-Weed, and A. J. Stromme. Asymptotics for semidiscrete entropic optimal transport. SIAM Journal on Mathematical Analysis , 54(2):1718-1741, 2022.
- [2] D. An, Y. Guo, N. Lei, Z. Luo, S.-T. Yau, and X. Gu. Ae-ot: A new generative model based on extended semi-discrete optimal transport. ICLR 2020 , 2019.
- [3] F. Bach. Adaptivity of averaged stochastic gradient descent to local strong convexity for logistic regression. The Journal of Machine Learning Research , 15(1):595-627, 2014.
- [4] F. Bach and E. Moulines. Non-strongly-convex smooth stochastic approximation with convergence rate o (1/n). Advances in neural information processing systems , 26, 2013.
- [5] B. Bercu and J. Bigot. Asymptotic distribution and convergence rates of stochastic algorithms for entropic optimal transportation between probability measures. Annals of Statistics , 49(2): 968-987, 2021.
- [6] J. Bigot, R. Gouet, T. Klein, and A. Lopez. Geodesic pca in the wasserstein space by convex pca. In Annales de l'Institut Henri Poincaré (B) Probabilités et Statistiques , volume 53, pages 1-26, 2017.
- [7] N. Bonneel and J. Digne. A survey of optimal transport for computer graphics and computer vision. Computer Graphics Forum , 42(2):439-460, 2023. doi: https://doi.org/10.1111/cgf.14778. URL https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.14778 .
- [8] Y. Brenier. Polar factorization and monotone rearrangement of vector-valued functions. Communications on Pure and Applied Mathematics , 44(4):375-417, 1991.
- [9] M. Buze, J. Feydy, S. M. Roper, K. Sedighiani, and D. P. Bourne. Anisotropic power diagrams for polycrystal modelling: efficient generation of curved grains via optimal transport, 2024.
- [10] Y. Chen, M. Telgarsky, C. Zhang, B. Bailey, D. Hsu, and J. Peng. A gradual, semi-discrete approach to generative network training via explicit wasserstein minimization. In International Conference on Machine Learning , pages 1071-1080. PMLR, 2019.

- [11] V. Chernozhukov, A. Galichon, M. Hallin, and M. Henry. Monge-kantorovich depth, quantiles, ranks and signs. The Annals of Statistics , 45(1):223-256, 2017.
- [12] L. Chizat. Annealed sinkhorn for optimal transport: convergence, regularization path and debiasing. arXiv preprint arXiv:2408.11620 , 2024.
- [13] K. Cohen, A. Nedi´ c, and R. Srikant. On projected stochastic gradient descent algorithm with weighted averaging for least squares regression. IEEE Transactions on Automatic Control , 62 (11):5974-5981, 2017.
- [14] N. Courty, R. Flamary, and D. Tuia. Domain adaptation with regularized optimal transport. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2014, Nancy, France, September 15-19, 2014. Proceedings, Part I 14 , pages 274-289. Springer, 2014.
- [15] N. Courty, R. Flamary, D. Tuia, and A. Rakotomamonjy. Optimal transport for domain adaptation. IEEE Transactions on Pattern Analysis and Machine Intelligence , 39(9):1853-1865, 2017. doi: 10.1109/TPAMI.2016.2615921.
- [16] M. Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Advances In Neural Information Processing Systems , 26, 2013.
- [17] M. Cuturi and G. Peyré. Semidual regularized optimal transport. SIAM Review , 60(4):941-965, 2018. doi: 10.1137/18M1208654. URL https://doi.org/10.1137/18M1208654 .
- [18] A. Delalande. Nearly tight convergence bounds for semi-discrete entropic optimal transport. In International Conference On Artificial Intelligence And Statistics , pages 1619-1642, 2022.
- [19] P. Dvurechensky, A. Gasnikov, and A. Kroshnin. Computational optimal transport: Complexity by accelerated gradient descent is better than by sinkhorn's algorithm. In International Conference On Machine Learning , pages 1367-1376, 2018.
- [20] J. Feydy. Geometric data analysis, beyond convolutions. Applied Mathematics , 2020.
- [21] J. Feydy, B. Charlier, F.-X. Vialard, and G. Peyré. Optimal transport for diffeomorphic registration. In M. Descoteaux, L. Maier-Hein, A. Franz, P. Jannin, D. L. Collins, and S. Duchesne, editors, Medical Image Computing and Computer Assisted Intervention - MICCAI 2017 , pages 291-299, Cham, 2017. Springer International Publishing. ISBN 978-3-319-66182-7.
- [22] A. Galichon. Optimal transport methods in economics . Princeton University Press, 2018.
- [23] C. Geiersbach and G. C. Pflug. Projected stochastic gradients for convex constrained problems in hilbert spaces. SIAM Journal on Optimization , 29(3):2079-2099, 2019.
- [24] A. Genevay, M. Cuturi, G. Peyré, and F. Bach. Stochastic optimization for large-scale optimal transport. In Advances In Neural Information Processing Systems , volume 29, 2016.
- [25] A. Genevay, G. Peyré, and M. Cuturi. Learning generative models with sinkhorn divergences. In International Conference on Artificial Intelligence and Statistics , pages 1608-1617. PMLR, 2018.
- [26] A. Godichon and B. Portier. An averaged projected robbins-monro algorithm for estimating the parameters of a truncated spherical distribution. Electronic Journal of Statistics , 11(1): 1890-1927, 2017.
- [27] Y. Guo. pyomt: A pytorch implementation of adaptive monte carlo optimal transport. https: //github.com/k2cu8/pyOMT , 2024. Accessed: 2025-05-07.
- [28] J.-C. Hütter and P. Rigollet. Minimax estimation of smooth optimal transport maps. The Annals of Statistics , 49(2), 2021.
- [29] V. Khrulkov and I. Oseledets. Understanding ddpm latent codes through optimal transport. ArXiv , abs/2202.07477, 2022. URL https://api.semanticscholar.org/CorpusID: 246863713 .

- [30] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 , 2014.
- [31] J. Kitagawa, Q. Mérigot, and B. Thibert. A newton algorithm for semi-discrete optimal transport. Journal of the European Mathematical Society , 21, 03 2016. doi: 10.4171/JEMS/889.
- [32] J. Kitagawa, Q. Mérigot, and B. Thibert. Convergence of a newton algorithm for semi-discrete optimal transport. Journal of the European Mathematical Society , 21(9):2603-2651, 2019.
- [33] A. Korotin, D. Selikhanovych, and E. Burnaev. Neural optimal transport. arXiv preprint arXiv:2201.12220 , 2022.
- [34] J. J. Kosowsky and A. L. Yuille. The invisible hand algorithm: Solving the assignment problem with statistical physics. Neural Networks , 7:477-490, 1994. URL https://api. semanticscholar.org/CorpusID:6967348 .
- [35] B. Lévy. A numerical algorithm for l 2 semi-discrete optimal transport in 3d. ESAIM: Mathematical Modelling and Numerical Analysis , 49(6):1693-1715, 2015.
- [36] Z. Li, S. Li, Z. Wang, N. Lei, Z. Luo, and D. X. Gu. Dpm-ot: a new diffusion probabilistic model based on optimal transport. In Proceedings of the ieee/cvf international conference on computer vision , pages 22624-22633, 2023.
- [37] A. Mensch and G. Peyré. Online sinkhorn: optimal transport distances from sample streams. In Proceedings of the 34th International Conference on Neural Information Processing Systems , NIPS'20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546.
- [38] Q. Mérigot. A multiscale approach to optimal transport. In Computer Graphics Forum , volume 30, pages 1583-1592. Wiley Online Library, 2011.
- [39] A. Mokkadem and M. Pelletier. A generalization of the averaging procedure: The use of two-time-scale algorithms. SIAM Journal on Control and Optimization , 49(4):1523-1543, 2011.
- [40] M. Nutz and J. Wiesel. Entropic optimal transport: Convergence of potentials. Probability Theory and Related Fields , 184(1-2):401-424, 2022.
- [41] M. Pelletier. Asymptotic almost sure efficiency of averaged stochastic algorithms. SIAM Journal on Control and Optimization , 39(1):49-72, 2000.
- [42] G. Peyré, M. Cuturi, et al. Computational optimal transport: With applications to data science. Foundations and Trends® in Machine Learning , 11(5-6):355-607, 2019.
- [43] B. T. Polyak and A. B. Juditsky. Acceleration of stochastic approximation by averaging. SIAM journal on control and optimization , 30(4):838-855, 1992.
- [44] A.-A. Pooladian and J. Niles-Weed. Entropic estimation of optimal transport maps. arXiv preprint arXiv:2109.12004 , 2021.
- [45] A.-A. Pooladian, V. Divol, and J. Niles-Weed. Minimax estimation of discontinuous optimal transport maps: The semi-discrete case. In International Conference on Machine Learning , pages 28128-28150. PMLR, 2023.
- [46] F. Santambrogio. Optimal transport for applied mathematicians . Progress in nonlinear differential equations and their applications. Birkhauser, 1 edition, Oct. 2015.
- [47] G. Schiebinger, J. Shu, M. Tabaka, B. Cleary, V. Subramanian, A. Solomon, J. Gould, S. Liu, S. Lin, P. Berube, et al. Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming. Cell , 176(4):928-943, 2019.
- [48] B. Schmitzer. Stabilized sparse scaling algorithms for entropy regularized transport problems. SIAM Journal on Scientific Computing , 41(3):A1443-A1481, 2019. doi: 10.1137/16M1106018. URL https://doi.org/10.1137/16M1106018 .

- [49] B. Schmitzer. Stabilized sparse scaling algorithms for entropy regularized transport problems. SIAM Journal On Scientific Computing , 41:A1443-A1481, 2019.
- [50] F. Schopfer. Linear convergence of descent methods for the unconstrained minimization of restricted strongly convex functions. SIAM Journal on Optimization , 26(3):1883-1911, 2016.
- [51] V. Seguy, B. B. Damodaran, R. Flamary, N. Courty, A. Rolet, and M. Blondel. Large-scale optimal transport and mapping estimation. arXiv preprint arXiv:1711.02283 , 2017.
- [52] M. Sharify, S. Gaubert, and L. Grigori. Solution of the optimal assignment problem by diagonal scaling algorithms. arXiv preprint arXiv:1104.3830 , 2011.
- [53] R. Sinkhorn. Diagonal equivalence to matrices with prescribed row and column sums. The American Mathematical Monthly , 74(4):402-405, 1967.
- [54] B. Ta¸ skesen, S. Shafieezadeh-Abadeh, and D. Kuhn. Semi-discrete optimal transport: Hardness, regularization and numerical solution. Mathematical Programming , 199(1):1033-1106, 2023.
- [55] A. Vacher and F.-X. Vialard. Semi-dual unbalanced quadratic optimal transport: fast statistical rates and convergent algorithm. In Proceedings of the 40th International Conference on Machine Learning , ICML'23. JMLR.org, 2023.
- [56] C. Villani. Optimal transport: old and new , volume 338. Springer, 2009.
- [57] J. Weed and F. Bach. Sharp asymptotic and finite-sample rates of convergence of empirical measures in wasserstein distance. Bernoulli , 25, 06 2017. doi: 10.3150/18-BEJ1065.
- [58] H. Zhang and W. Yin. Gradient methods for convex minimization: better rates under weaker conditions. arXiv preprint arXiv:1303.4645 , 2013.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Proofs and numerical experiments are given for all the claims made in the abstract and the introduction.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The main limitations, such as the assumptions or dependence on specific parameters, are clearly stated in the paper.

## 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: The assumptions are always clearly stated and all the proofs are given in appendix.

## 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: All the settings for the experiments are clearly stated and the parameters of our algorithm are given.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: All the code to reproduce our experiments is provided in Python Notebooks and attached in a zip file in the supplementary materials.

## 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: The parameters of our algorithm are given at the beginning of the Numerical Experiments section, and the example settings are explicitly provided.

## 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All the experiments are run several times, and the error plots represent the average error across the experiments.

## 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: No specific computing resources are required for our experiments; they can be run on any modern machine, even without a GPU. The only exception is the Neural OT algorithm, which does require a GPU.

## 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The authors have read the NeurIPS Code of Ethics and guarantee that the paper conforms to it.

## 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The code provided is original and is included for reproducibility under the correct license.

## 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: The algorithms and experiments are clearly explained in the paper, and the code to reproduce the experiments is provided in an anonymized zip file.

## 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

## 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

## Appendix

## Table of Contents

| A Additonnal Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   | A Additonnal Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   | .                                                                                                                                                                 |   17 |
|------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
|                                                                                          | A.1                                                                                      | Mini-batch DRAG. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                  |   17 |
|                                                                                          | A.2                                                                                      | Weighted Averaging: Maintaining a better trade-off between averaged and non- averaged iterations. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |   17 |
|                                                                                          | A.3                                                                                      | DRAG compared to Adam. . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                      |   18 |
|                                                                                          | A.4                                                                                      | DRAG compared to non regularized ASGD . . . . . . . . . . . . . . . . . . .                                                                                       |   18 |
|                                                                                          | B Proofs . . . . . . . . . . . . . . . . . . . .                                         | . . . . . . . . . . . . . . . . . . . . . .                                                                                                                       |   19 |
|                                                                                          | B.1                                                                                      | Proof of Theorem 1: Convergence rate of the non averaged iterates. . . . . . . .                                                                                  |   19 |
|                                                                                          | B.2                                                                                      | Proof of Theorem 2: Convergence rate of DRAG . . . . . . . . . . . . . . . .                                                                                      |   22 |
|                                                                                          | B.3                                                                                      | Proof of Theorem 3: Convergence of the OT map estimator . . . . . . . . . . .                                                                                     |   25 |
|                                                                                          | B.4                                                                                      | Proof of Corollary 1: OT cost estimation . . . . . . . . . . . . . . . . . . . . .                                                                                |   26 |
|                                                                                          | B.5                                                                                      | Proof of Proposition 2: High probability being in B ( g ∗ ε t , ε t ) . . . . . . . . . . .                                                                       |   27 |
|                                                                                          | B.6                                                                                      | Proof of Lemma 1: Projection step . . . . . . . . . . . . . . . . . . . . . . . .                                                                                 |   31 |
|                                                                                          | B.7                                                                                      | Proof of Lemma 2: Global and local RSC condition of H ε . . . . . . . . . . .                                                                                     |   32 |
|                                                                                          | B.8                                                                                      | Other Technical results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                               |   34 |

## A Additonnal Experiments

## A.1 Mini-batch DRAG.

As for Vanilla SGD, we can take advantage of GPU parallelization and replace the gradient estimator using one sample X ∼ µ

<!-- formula-not-decoded -->

by a mini-batch estimator, using n b ≥ 1 i.i.d samples X 1 , ..., X n b samples of the source measure at once

<!-- formula-not-decoded -->

Of course, no matter the choice n b , (7) defines an unbiased estimator of ∇ H ε ( g ) .

Using a mini-batch of size n b , we suggest multiplying γ 1 by √ n b , as is usual with mini-batch SGD. The following figure shows the acceleration due to mini-batching on Example 1, 2 and 3, while maintaining the same computational time when using a GPU. Indeed, each mini-batch estimator has an error an order of magnitude lower than the non-batched ones, even with a small mini-batch size of n b = 16 .

Figure 6: Comparison of the non mini-batched and mini-batched estimators on Example 1, 2 and 3, n b = 16 .

<!-- image -->

## A.2 Weighted Averaging: Maintaining a better trade-off between averaged and non-averaged iterations.

I is well known that the averaged algorithm can suffer from bad initialization. One strategy to overcome this is weighted averaging [39]. Namely, we replace the averaged estimator g t = 1 t +1 ∑ t k =0 g t , by

<!-- formula-not-decoded -->

with a parameter ω &gt; 0 . The parameter ω balances the weights assigned to the estimators g k . As ω increases, greater importance is given to the more recent estimates, while we retrieve g t when ω goes to 0 . As for the usual averaged estimators, we can perform the weighted average online, without having to store all the iterates, with the recursion

<!-- formula-not-decoded -->

It is important to note that g ( ω ) t will have the same asymptotic convergence guarantees as g t .

Figure 7: Comparison between g t , g t and g ( ω ) t on Examples 1, 2 and 3, with ω = 2

<!-- image -->

As illustrated in Figure 7, the weighted average estimator consistently outperforms g t , achieving orders of magnitude better performance in Examples 2 and 3. Note that a mini-batch size of 16 was used for all experiments, each repeated 10 times.

## A.3 DRAG compared to Adam.

We compare here the performance of our algorithm DRAG to that of the Adam algorithm [30], on Example 1, with M ∈ 200 , 2000 . The experiment was repeated 10 times. For this comparison, we fixed the parameters of DRAG to ( √ M, 1 / 3 , 2 / 3) and ran the algorithm for t = 10 5 iterations. The parameters for Adam were set to β 1 = 0 . 9 , β 2 = 0 . 999 , and λ = 10 -3 (learning rate/weight decay). As shown in Figure 8, DRAG clearly outperforms Adam on this example, particularly in the early iterations and as the number of points increases.

Figure 8: Comparison of DRAG with Adam on Example 1, for different values of M .

<!-- image -->

## A.4 DRAG compared to non regularized ASGD

As discussed in the numerical section, we observed empirically that all methods, including the nonregularized projected ASGD, eventually converge to the true solution given a sufficient number of iterations. In Figure 9, we report additional experiments with a larger iteration budget to confirm this behavior. Notably, even the non-regularized ASGD converges, albeit much more slowly. In contrast, DRAG converges significantly faster and achieves higher accuracy earlier in the optimization process. Among the DRAG variants, we observe that choices of a ∈ 0 . 2 , 0 . 4 yield the best performance, which aligns with our theoretical analysis suggesting that a = 1 3 -achieves the optimal convergence rate in this setting, since b = 2 3 . These results reinforce our claim from the main text that, while all schemes converge given enough iterations, DRAG remains consistently one to two orders of magnitude more accurate due to its improved early-stage performance (see Appendix A).

Note also that the convergence of the non-regularized ASGD is encouraging, as it highlights that, with a sufficiently large number of steps, the effect of vanishing regularization is not a deterrent. Indeed, as t →∞ , the regularized gradient becomes numerically indistinguishable from the non-regularized one, essentially corresponding to the difference between a tempered softmax with temperature ε and

an argmax. This further supports the view that DRAG serves as an effective acceleration mechanism in the early stages of optimization and that by decreasing the regularization, we will not hit a plateau.

<!-- image -->

Iterations

Figure 9: Comparison of DRAG with different a and non-regularized ASGD

## B Proofs

## Additionnal notations.

For any c &gt; 0 we define the function t ↦→ Ψ c ( t ) such that

<!-- formula-not-decoded -->

For a sequence ( u t ) t ∈ N , if t 2 / ∈ N , u t 2 must be understood as u ⌈ t 2 ⌉ .

<!-- formula-not-decoded -->

Remark that the dependence in t is both in the estimator g t and the optimizer g ∗ ε t . We also recall that we note D C := sup ′ g g ′ &lt; .

In all the sequel, we note

<!-- formula-not-decoded -->

## B.1 Proof of Theorem 1: Convergence rate of the non averaged iterates.

Proof. Using Lemma 3, for any t ≥ t a,α , we have

<!-- formula-not-decoded -->

Let F t denote the filtration generated by the samples X 1 , . . . , X t iid ∼ µ , that is F t = σ ( X 1 , . . . , X t ) and taking the conditional expectation, we have

<!-- formula-not-decoded -->

Using Lemma 2 on the restricted strong convexity of H ε t , we have

Therefore, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using Proposition 2, for all p and β ∈ (0 , 1) , there exists C β,p such that for all t ≥ 0 , E [∆ p t ] ≤ C β,p γ p t ε βp t ε p t ≤ C β,p t -bp + a (1 -β ) p . Therefore,

<!-- formula-not-decoded -->

Note that, since 2 a &lt; b , we can always choose β such that the inequality a (3 -β ) &lt; b holds. Using the fact that ∆ t ≤ D 2 C and taking the expectation in (10), we obtain

<!-- formula-not-decoded -->

Let t γ := min { t, 2 λγ t +1 ≤ 1 } and t 0 := max { t a,α , t γ } , we apply Proposition 5 to obtain

<!-- formula-not-decoded -->

Applying Corollary 2, the exponential product converges exponentially fast to 0 and an asymptotic comparison gives

<!-- formula-not-decoded -->

We conclude by using Proposition 1, using the bound ∥ g ∗ ε t -g ∗ ∥ ≲ ε 1+ α ′ t .

Proposition 3. Under the same assumptions as in Theorem 1 , we have for any α ′ ∈ (0 , α )

<!-- formula-not-decoded -->

Remark: Note that this proposition directly proves Theorem 1, but we decided to split them, to have a cleaner proof of Theorem 1.

Proof. We begin by squaring equation (13) of Lemma 3. For t ≥ t a,α , where t a,α is defined in (15), we have

<!-- formula-not-decoded -->

Taking the conditional and using Lemma 2, we have

We also use the simple bound

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

These two inequalities lead to

<!-- formula-not-decoded -->

Using that the gradient norm is bounded by two, we apply the Cauchy-Schwarz inequality to obtain

<!-- formula-not-decoded -->

Applying Hölder's inequality yields

<!-- formula-not-decoded -->

Summing up these inequalities, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly to the case p = 1 , using that P [ ∥ g t -g ∗ ε t ∥ ≥ ε t / 2] ≤ C β, 4 b b -a (3 -β ) t -4 b by (11) and that ∆ 2 t ≤ D 4 C for all t , taking the expectation yields

Again, as in the case p = 1 , applying Proposition 5 and Corollary 2 and using that ∥ g ∗ ε t -g ∗ ∥ ≲ ε 1+ α ′ t concludes the proof.

Lemma 3. Under the assumptions of Theorem 1, there exists a finite time t a,α , depending on a and α , such that for all t ≥ t a,α , we have

<!-- formula-not-decoded -->

Proof. By definition of the gradient step at time t +1 and since g ∗ ε t +1 ∈ C , we have

<!-- formula-not-decoded -->

Then, incorporating the change of optimum between time t and t +1 , we get

<!-- formula-not-decoded -->

Using Corollary 2.2 in [18] (see Proposition 1), there exists K 0 &gt; 0 such that for any α ′ ∈ ]0 , α [

<!-- formula-not-decoded -->

For clarity, we define r t := aK 0 t -(1+ a + aα ′ ) and R t := (2 D C +2 γ t +1 + r t ) r t . Using that for all t , g t ∈ C , and that for all x ∈ R d , g ∈ R M , ∥∇ g h ε t ( g , x ) ∥ ≤ 2 , we obtain

<!-- formula-not-decoded -->

Note that, since we have 1 + a + aα &gt; 2 b , we can also take α ′ ∈ ]0 , α [ such that 1 + a + aα ′ &gt; 2 b . Consequently, the sequence R t /γ 2 t is decreasing and tends to 0. For conciseness, we note

<!-- formula-not-decoded -->

For any t ≥ t a,α , we then obtain the following upper bound of ∆ t +1 in terms of ∆ t and the gradient direction:

<!-- formula-not-decoded -->

## B.2 Proof of Theorem 2: Convergence rate of DRAG

Proof. We start with a decomposition of the gradient step, similar to [26]. By abuse of notation, we note and define the following differences:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The term p k represents the difference between the projected and non-projected steps. Note that p k = 0 if g k -γ k +1 ∇ g h ε k ( g k , X k +1 ) ∈ C . The term ξ k is a martingale difference ξ k representing the difference between the regularized gradient and its non-biased estimator. σ k represents the difference between the ε k -regularized gradient and the non-regularized gradient.Finally, δ k represents the difference between the gradient at g k and its linear approximation given by the Hessian at the optimum.

Let I M denote identity matrix of M M ( R ) , observe that for any k ∈ N

incorporating δ k

<!-- formula-not-decoded -->

Thus, we have that

<!-- formula-not-decoded -->

Observe that there is an orthogonal matrix U such that ∇ 2 ∗ = U diag ( λ 1 , . . . , λ M -1 , 0) U ⊤ . Therefore, in the following, we denote

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

the inverse of ∇ 2 ∗ , restricted to the subspace Vect( 1 M ) ⊥ . Note that we have [18, Theorem 3.2]

Taking all the equalities in Vect( 1 M ) ⊥ , that is, considering all our vectors in the subspace Vect( 1 M ) ⊥ , we have

<!-- formula-not-decoded -->

We will now give the convergence rate for each sum. Note that thanks to the introduction of σ k , we will directly be able to use the local smoothness and strong convexity of H 0 , proved in our setting in [32].

- Convergence rate for 1 t +1 ∑ t k =0 g k -g k +1 γ k +1 .

<!-- formula-not-decoded -->

Remark that γ -1 t +1 -γ -1 t ≤ 2 γ -1 1 n b -1 . By Theorem 1 (non-averaged iterates), E [ ∥ g n -g ∗ ∥ 2 v ] ≲ γ 1 ρ ∗ ( t +1) -b . Therefore

<!-- formula-not-decoded -->

We thus have the convergence rate

<!-- formula-not-decoded -->

- Convergence rate for 1 t +1 ∑ t k =0 δ k .

By [32, Theorem 1.3], there exists a ball B ( g ∗ , d 1 ) with d 1 &gt; 0 where H is α -Hölder. Therefore, by applying a Taylor expansion of ∇ H 0 ( g ) around g ∗ , if g k ∈ B ( g ∗ , d 1 ) , we have

<!-- formula-not-decoded -->

Otherwise, since the Hessian H 0 is uniformly bounded [32, Theorem 1.1], there exists a constant C δ such that for any g ∈ C , ∥∇ H ( g ) -∇ 2 H ( g ∗ ) ( g -g ∗ ) ∥ ≤ C δ .

Since P ( g k / ∈ B ( g ∗ , d 1 )) = P ( ∥ g k -g ∗ ∥ &gt; d 1 ) , we obtain by Markov's inequality

<!-- formula-not-decoded -->

Therefore, using Minkowski's inequality, we have

<!-- formula-not-decoded -->

- Convergence rate for 1 t +1 ∑ t k =0 ξ k +1 .

We recall that ξ k +1 = ∇ H ( g k ) -∇ g h ( g k , X k +1 ) and thus E [ ξ k +1 ] = 0 .

<!-- formula-not-decoded -->

Thus, since E [ ∥ ξ k ∥ 2 ] ≤ 4 for all k , we have the convergence rate

<!-- formula-not-decoded -->

## · Convergence rate of 1 t +1 ∑ t k =0 σ k .

<!-- formula-not-decoded -->

Using Proposition 4, we have uniformly in g k ∈ C that, for all α ′ ∈ (0 , α ) ,

Therefore

<!-- formula-not-decoded -->

## · Convergence rate for 1 t +1 ∑ t k =0 p k γ k .

Take d 0 such that B ( g ∗ , d 0 ) ⊂ C . Defining ∇ k := ∇ g h ( g k , X k +1 ) for conciseness, we obtain

<!-- formula-not-decoded -->

Since for any y ∈ C , one has ∥ x -Proj C ( x ) ∥ v ≤ ∥ x -y ∥ v , taking y = g k , and since g k -γ k +1 ∇ k / ∈ C is satisfied only if ∥ g k -γ k +1 ∇ k -g ∗ ∥ v &gt; d 0 , we have

<!-- formula-not-decoded -->

We thus have

<!-- formula-not-decoded -->

## · Conclusion.

<!-- formula-not-decoded -->

Finally, summing up all the convergence rates, using Cauchy-Schwarz inequality and that ( A + B ) 2 ≤ 2( A 2 + B 2 ) for any A,B ∈ R we obtain

Since b &gt; 2 a and b + αa &gt; 2 a +2 aα ′ and so noting s = min { 1 , 2 a +2 aα ′ } , and since the Hessian norm is uniformly bounded, we finally obtain

<!-- formula-not-decoded -->

## B.3 Proof of Theorem 3: Convergence of the OT map estimator

Proof. We show that the convergence rate of g t to g ∗ 0 implies a convergence rate a convergence rate for the map estimation. The Brenier map is given by T µ,ν ( x ) = x -∇ ( g ∗ 0 ) c ( x ) ; see for instance [46], Theorem 1.17. We thus focus on the convergence of ∇ g c t to ∇ ( g ∗ 0 ) c .

<!-- formula-not-decoded -->

For all j ∈ J 1 , M K , if x lies in the interior of L j ( g ) , we have ∇ c -j

Therefore, given g , g ′ ∈ R M , if there exists a j ∈ J 1 , M K such that x is the interior of L j ( g ) ∩ L j ( g ′ ) we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We will now follow arguments from [46], Section 6.4.2. Fix j, j ′ ∈ J 1 , M K such that j = j ′ and x is in the interior of L j ( g ) ∩ L j ′ ( g ′ ) . By definition of the c -transform, we observe that L j ( g ) is defined by M -1 linear inequalities of the form

Similarly, interchanging the role of g , g ′ and j , j ′ we have

<!-- formula-not-decoded -->

We obtain that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

Moreover, noting h = ( h 1 , ..., h M ) = g -g ′ , we see that

We have

<!-- formula-not-decoded -->

Under Assumption 1, µ is a measure such that Supp( µ ) ⊂ B (0 , R ) and it admits a density d µ bounded by d µ max . Thus,

<!-- formula-not-decoded -->

̸

by the rotational invariance of the Lebesgue measure. Combining this remark with (17) yields

<!-- formula-not-decoded -->

Similarly, for the L p norm of the map difference, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Plugging in the convergence rate of g t to g ∗ concludes the proof.

## B.4 Proof of Corollary 1: OT cost estimation

Proof. For any vector g ∈ R M , we recall the definition of L ( g ) = ⋃ M j =1 L j ( g ) :

<!-- formula-not-decoded -->

̸

Note that L ( g ) defines a partition of R d up to µ -null sets , i.e. µ ( L i ( g ) ∩ L j ( g )) = 0 when i = j , and the convex sets L j ( g ) are called power or Laguerre cells. We define the set

<!-- formula-not-decoded -->

Using Theorem 4.1 in [32], under Assumption 1, H 0 is uniformly C 2 ,α on K δ . That is, there exists a constant L such that H 0 is L -smooth on K δ . Note that the constant L depends on µ min , δ , R . We refer to [32], Remark 4.1 for more details.

By the first order condition, as soon as δ ≤ w min , we have g ∗ ∈ K δ . Indeed, at the optimum, we have for all i ∈ J 1 , M K , µ ( L i ( g ∗ )) = w i . We fix here δ = 1 10 w min .

<!-- formula-not-decoded -->

Thanks to the L -smoothness, for any g ∈ K δ , we have

Note that, for any g ∈ R M and i ∈ J 1 , M K , the difference of measure of the Laguerre cells L i ( g ) and L i ( g ∗ ) is at most linear with respect to ∥ g -g ∗ ∥ ∞ . We refer to Theorem 3 or Section 6.4.2 in [46] for more details.

Therefore, there exists a constant C L such that, as soon as ∥ g -g ∗ ∥ 2 ≤ C L , we have that g ∈ K δ . This constant depends on δ, µ max , R and d as in Theorem 3. Using Theorem 2, E [ ∥ g t -g ∗ ∥ 2 ] = O ( t -s ) with s &gt; 0 . Then where the Markov inequality of order 1 was used on E [ ✶ ∥ g t -g ∗ ∥ 2 &gt;C L ] .

<!-- formula-not-decoded -->

So, in particular, there exists a constant C ∆ &gt; 0 , independent of the location of the points y j , which grows at least linearly in M such that

<!-- formula-not-decoded -->

## B.5 Proof of Proposition 2: High probability being in B ( g ∗ ε t , ε t ) .

Proof. The proof of this proposition relies heavily on the technical Lemma 4, which we state and prove immediately after this proof.

We start with a base case at δ = u 0 = 0 , which provides an initial convergence rate for E [∆ p t ] . Then, by an inductive argument, we gradually increase u n to improve this rate till the limit when n tends to infinity, namely min { b -a, a } .

<!-- formula-not-decoded -->

Base case ( u 0 = 0 ). Using Lemma 4, with λ c,t = ρ ∗ 1 -e -4 C ∞ 4 C ∞ ε t if c = 0 , and λ c,t = ρ ∗ (1 -e -1 ) ε t t c if c ∈ (0 , a ] , we have

By applying Proposition 5 and Corollary 2, we obtain the following baseline convergence rate, for all p &gt; 0 :

<!-- formula-not-decoded -->

Inductive step (improving the rate). Suppose that for some u n ∈ [0 , min { b -a, a } ) , we already have

<!-- formula-not-decoded -->

Choose c &lt; b -a + u n 2 and set d = b -a + u n -2 c &gt; 0 , which is positive by construction. By Markov's inequality, we then get for all q &gt; 0

<!-- formula-not-decoded -->

We take q chosen large enough so that dq &gt; p +1 .

Consequently, applying Lemma 4,

<!-- formula-not-decoded -->

Therefore, if we pick any u n +1 ∈ ( 0 , b -a + u n 2 ) , applying Proposition 5 and Corollary 2, we see that

<!-- formula-not-decoded -->

As soon as b -a &gt; u n , we have ( b -a + u n ) / 2 &gt; u n as a valid range upper range for u n +1 , so we can take u n +1 &gt; u n and strictly improve our convergence rate.

Achievability for all δ ∈ [0 , min { b -a, a } ) . Finally, note that the sequence defined by u 0 = 0 and u n +1 = b -a + u n 2 converges to ( b -a ) , showing that every value δ up to ( b -a ) can be reached through successive improvements. Since c = a is the upper bound in Lemma 4, we can continue the limit min { b -a, a } , so for all δ ∈ [0 , min { b -a, a } ) , we have

<!-- formula-not-decoded -->

Using Markov's inequality concludes the proof.

Lemma 4. For any a, b &gt; 0 , such that 1 + a + aα &gt; 2 b , there exists constants C 1 ,p , C 2 ,p , only depending on γ 1 and p , such that defining

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have for any c ∈ [0 , a ]

̸

Proof. Let us fix c ∈ [0 , a ] . Starting from equation (13), raising to the power p gives

<!-- formula-not-decoded -->

We note ( p i,j,k ) = p ! i ! j ! k ! and apply the trinomial expansion to obtain

<!-- formula-not-decoded -->

We divide the set

<!-- formula-not-decoded -->

into the following partition

<!-- formula-not-decoded -->

̸

̸

̸

̸

In what follows, the constants C k may depend on the constant γ 1 from the learning rate γ t = γ 1 /t b , since we will often use the crude bound γ p +1+ k t ≤ γ k 1 γ p +1 t for k ∈ N . Note, however, that λ c,t ≤ 1 for all t , and therefore λ -q + k c,t ≤ λ -q c,t for all q, k ∈ N , so the constants C k will not depend on λ c,t . We also introduce, for all p , the constant

<!-- formula-not-decoded -->

Case where ( i, j, k ) ∈ P a .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Case where ( i, j, k ) ∈ P b .

We have j + k = p such that j +2 k ≥ p +1 since k = 0 . Using the bound ∥ g t -g ∗ ε t ∥ ≤ D C , we obtain:

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Case where ( i, j, k ) ∈ P c .

<!-- formula-not-decoded -->

Case where ( i, j, k ) ∈ P d .

<!-- formula-not-decoded -->

i

by Young:

q

=

+

j/

2

, q

′

=

j

+2

k

.

Taking c 6 = γ 2 /q t , it comes c -q ′ 6 = γ -2 q ′ /q t = γ -2 / ( q -1) t . Since we are only considering cases with i, j, k ≥ 1 (which forces p ≥ 3 ) and we are excluding the particular case ( i, j, k ) = ( p -2 , 1 , 1) , one can show that the parameter q = 2 p 2 p -2 i -j = 2 p 2 k + j satisfies

Thus, since 2 q -1 ≤ p -2 , it follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, using the crude bound ∑ ( i,j,k ) ∈P d ( p i,j,k ) ≤ 3 p and defining a constant C 6 readily, we obtain

<!-- formula-not-decoded -->

Case where ( i, j, k ) ∈ P e .

Since j = 0 , i + k = p , and ( p -1 , 0 , 1) ∈ P a , we have k ≥ 2 . We use Young's inequality with q = p i , q ′ = p k to obtain

<!-- formula-not-decoded -->

Summing up the inequalities , we obtain

<!-- formula-not-decoded -->

By convexity of H ε t , taking the conditional expectation gives

<!-- formula-not-decoded -->

Applying Lemma 2, recalling that λ c,t = ρ ∗ 1 -e -1 2 √ 2 c ∞ ε t if c = 0 , and λ c,t = ρ ∗ 1 -e -1 √ 2 ε t t c if c ∈ (0 , a ] , we have

Therefore,

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

We now just have to sum up the inequalities. Fixing Γ p = 3 p +1 2 p -1 such that -2 p + 3 p +1 Γ p = -1 , C 1 ,p = 8 p +3 p +2 p , C 2 ,p = 8 p + ∑ 7 k =1 C k , and taking the expectation, we have the desired form E [∆ p t +1 ] ≤ E [∆ p t ] ( 1 -γ t +1 λ c,t + C 1 ,p γ 2 t +1 ) + γ t +1 D 2 p C E [ 1 ∆ t ≥ t -2 c ] 1 c =0 + C 2 ,p λ -p +1 c,t γ p +1 t +1 .

## B.6 Proof of Lemma 1: Projection step

Proof. According to [40], any optimal pair of functions ( f ε , g ε ) solving the dual formulation of entropic OT with regularization ε ≥ 0 satisfies the Schrödinger equations. That is, we can take for all y ∈ R d , g ε ( y ) = f c,ε ε ( y ) . Moreover, 1 2 ∥ x -y ∥ 2 is R -Lipschitz on B (0 , R ) . Therefore, since by Assumption 1, we have Supp( µ ) ⊂ B (0 , R ) and Supp( ν ) ⊂ B (0 , R ) , we can exploit the Lipschitz property of our cost function on B (0 , R ) . Using that the ( c, ε ) -transform has the same modulus of continuity as c (see Lemma 3.1 in [40]), we get, for all y, y ′ ∈ R d :

That is, coming back to the function g , we have for all j, j ′ ∈ 1 , M K :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By writing back our dual potential as a vector, that is g ∗ = ( g ∗ 1 , . . . , g ∗ M ) , where for all j ∈ J 1 , M K , g ∗ j = g ε ( y j ) , we have

Moreover, if g ∗ optimizes the semi-dual H ε , then for any β ∈ R , the vector g ∗ + β 1 M optimizes H ε . In particular, g ∗ -g ε ( y 1 ) 1 M , which we rename g ∗ , optimizes the semi-dual, with g ∗ 1 = 0 . Hence, for all j ∈ 1 , ..., M

<!-- formula-not-decoded -->

That is, there exists an optimizer in the desired closed convex set.

Remark: Note that for other costs such as c ( x, y ) = ∥ x -y ∥ which defines the 1-Wasserstein distance, this projection set can be more relevant. Indeed, in this case, the cost is 1 -Lipschitz and the projection set depends only on the target measure ν and no assumption of bounded cost is needed. In this case, the practitioner could choose the index k such that g k = 0 , minimizing for instance the Euclidean diameter of the corresponding set.

## B.7 Proof of Lemma 2: Global and local RSC condition of H ε

Proof. For any g ∈ C and s ∈ [0 , 1] , note g s = g ∗ ε + s ( g -g ∗ ε ) , where g ∗ ε is the minimizer of H ε satisfying ∑ M i =1 g i = ∑ M i =1 g ∗ ε,i and define φ by

<!-- formula-not-decoded -->

Applying Lemma 5, whose proof is postponed until after this one, we have that

<!-- formula-not-decoded -->

where for all x ∈ R d : m ( x, g s ) := ∑ M j =1 χ ε j ( x, g s )( g s -g ∗ ε ) .

Using Hölder's inequality with the Hölder conjugates p = 1 , q = + ∞ for δ 0 and Cauchy-Schwarz inequality as in [5] for δ 1 , we obtain

<!-- formula-not-decoded -->

Use δ = δ 0 or δ 1 = 1 . Since ∑ M i =1 g i = ∑ M i =1 g ∗ ε,i , φ is strictly convex, and therefore, we can divide by φ ′′ ( s ) to obtain for s ∈ [0 , 1]

<!-- formula-not-decoded -->

Integrating between 0 and S and using that ∫ S 0 φ ′′′ ( s ) φ ′′ ( s ) ds = ln | φ ′′ ( S ) | -ln | φ ′′ (0) | gives

<!-- formula-not-decoded -->

Since φ ′′ ( s ) = ( g -g ∗ ε ) ⊤ ∇ 2 H ε ( g s ) ( g -g ∗ ε ) , recalling that ρ ∗ is the second smallest eigenvalue of ∇ 2 H ε ( g ∗ ε ) gives the upper bound

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, since φ ′ ( s ) = ⟨∇ H ε ( g s ) , g -g ∗ ε ⟩ , an integration of (23) between 0 and 1 gives

Note that the function δ ∈ (0 , ∞ ) ↦→ 1 δ (1 -exp( -δ )) is strictly decreasing and upper bounded by 1 .

<!-- formula-not-decoded -->

If ε = 1 , take δ = δ 0 = 2 ∥ g -g ε ∥ ∞ and use the fact that ∥ g -g ε ∥ ∞ ≤ 2 C ∞ to obtain

In the same way, for ε &lt; 1 and ∥ g -g ∗ ε ∥ ≤ ε √ 2 , taking δ = δ 1 = √ 2 ∥ g -g ∗ ε ∥ ≤ 1 we obtain

<!-- formula-not-decoded -->

which concludes the proof.

Lemma 5 (Helping Lemma for the RSC condition of H ε ) . For any g ∈ C and t ∈ [0 , 1] , define g s = g ∗ ε + s ( g -g ∗ ε ) , where g ∗ ε is the minimizer of H ε satisfying ∑ M i =1 g i = ∑ M i =1 g ∗ ε,i . The function φ , defined by

<!-- formula-not-decoded -->

satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. The proof is an adaptation of the proof of Lemma A.2 in [5]. For completeness, we recall all the steps of their proof that are needed for our results. Note that their recent erratum regarding Lemma A.1 has no impact on Lemma A.2.

For any g ∈ R M and s ∈ [0 , 1] , define g s = g ∗ ε + s ( g -g ∗ ε ) , where g ∗ ε is the minimizer of H ε satisfying ∑ M i =1 g i = ∑ M i =1 g ∗ ε,i . We also define the function φ by

<!-- formula-not-decoded -->

Its first to third-order derivatives are given by

<!-- formula-not-decoded -->

Since for all g ∈ R M , ∇ H ε ( g ) = -E X ∼ µ [ χ ε ( X, g )] + w , defining for all x ∈ R d , m ( x, g s ) = ∑ M i =1 χ ε i ( x, g s )( g i -g ∗ ε,i ) .

<!-- formula-not-decoded -->

Using that ∇ g χ ε ( x, g ) = 1 ε ( diag ( χ ε ( x, g )) -χ ε ( x, g ) χ ε ( x, g ) ⊥ ) , we have

<!-- formula-not-decoded -->

Therefore, using the expression of m yields to

<!-- formula-not-decoded -->

defining for all x ∈ R d

<!-- formula-not-decoded -->

A derivation of σ 2 leads to (see [5] eq. (A.19),) for more details)

<!-- formula-not-decoded -->

Since ε 2 φ ′′′ ( s ) = E X ∼ µ [ d ds σ 2 ( X, g s ) ] , we conclude

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B.8 Other Technical results

Proposition 4. For all g ∈ C and all α ′ ∈ (0 , α ) , we have

<!-- formula-not-decoded -->

Proof. We adopt the decomposition of X from Appendix A.3 of [18]. See Figure 1 in [18] for an illustration of this decomposition. Fix i ∈ { 1 , . . . , M } , let X ⊂ B (0 , R ) be the support of µ , and choose parameters

<!-- formula-not-decoded -->

Define, for all i ∈ J 1 , M K , the function

<!-- formula-not-decoded -->

and use these to define the following sets:

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We also recall the point-wise definitions of the regularized and non-regularized functions constituting the gradients

<!-- formula-not-decoded -->

and define the constant

## Error decomposition.

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

## 1. Interior regions X i,η, + and X i,η, -.

<!-- formula-not-decoded -->

For x ∈ X i,η, + one has f i -f j ≥ η ∥ y i -y j ∥ ≥ ηc y for every j = i , hence

̸

̸

<!-- formula-not-decoded -->

̸

In a same way, we obtain the same bound if x ∈ X i,η, -, therefore

<!-- formula-not-decoded -->

## 2. Simple slabs A i,η,γ .

Inside one slab T ij = { x + t d ij ∣ ∣ x ∈ H γ ij , t ∈ [ -η ∥ y i -y j ∥ , η ∥ y i -y j ∥ ] } , and we have f i ( x ) -f j ( x ) = t ∥ y i -y j ∥ . for a certain t ∈ [ -η ∥ y i -y j ∥ , η ∥ y i -y j ∥ ] . All other indices satisfy f k ( x ) -f i ( x ) ≤ -c y γ , so

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

with p ˜ ε ( t ) = ( 1 + e -t/ ˜ ε ) -1 , ˜ ε = ε/ ∥ y i -y j ∥ .

<!-- formula-not-decoded -->

Introduce coordinates x = z + tn ij with n ij = y i -y j ∥ y i -y j ∥ and z ∈ H ij ; the Jacobian of this change of coordinate is 1 . Since f µ is α -Hölder, there exists L &gt; 0 such that we can write f µ ( z + tn ij ) = f µ ( z ) + r α ( z, t ) with | r α ( z, t ) | ≤ L | t | α . Note that the function p ˜ ε ( t ) -1 t&gt; 0 is odd. Writing σ the Hausdorff measure of dimension d -1 , we obtain

Summing over j = i yields

̸

## 3. Corner set B i,η,γ .

As shown in [18], denoting by θ the maximum angle that can be formed by three non-aligned points of the target measure, each corner that constitutes B i,η,γ is included in a cylinder of volume at most 4 π diam( B (0 ,R )) d -2 cos( θ/ 2) 2 γ 2 . Moreover, there are at most M 2 such corners. Therefore, µ ( B i,η,γ ) = O ( γ 2 ) = O ( η 2 ) , and so

## 4. Choice of the exponent β .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let α ′ ∈ (0 , α ) and pick

Then η 1+ α = ε β (1+ α ) ≤ ε 1+ α ′ and η 2 = ε 2 β ≤ ε 1+ α ′ . Exponential terms are even smaller, hence

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proposition 5. Let ( γ t ) t ≥ 0 and ( ν t ) t ≥ 0 be some positive and decreasing sequences and let ( δ t ) t ≥ 0 , satisfying the following:

- The sequence δ t follows the recursive relation:

<!-- formula-not-decoded -->

- Let γ t converge to 0 .
- Let t 0 = inf { t ≥ 1 : ωγ t +1 ≤ 1; ηγ t ≤ ω } .

Then, for all t ≥ t 0 , we have the upper bound:

<!-- formula-not-decoded -->

Proof. For all t ≥ t 0 , since 1 -2 ωγ t +1 + ηγ 2 t +1 ≤ 1 -ωγ t +1 , one has

t

δ

<!-- formula-not-decoded -->

+

ν

t

+1

γ

t

+1

One can consider two cases: ⌈ t/ 2 ⌉ -1 ≤ t 0 and ⌈ t/ 2 ⌉ -1 &gt; t 0 .

<!-- formula-not-decoded -->

Case where ⌈ t/ 2 ⌉ -1 ≤ t 0 &lt; t : Since ν k is decreasing,

Since ν k is decreasing, it comes U 2 ,t ≤ 1 ω ν ⌈ t/ 2 ⌉ .

Case where ⌈ t/ 2 ⌉ -1 &gt; t 0 : As in [3], for all m = t 0 +1 , . . . , t , one has

<!-- formula-not-decoded -->

Then, taking m = ⌈ t/ 2 ⌉ -1 , it comes

<!-- formula-not-decoded -->

Corollary 2. Let ( γ t ) t ≥ 0 and ( ν t ) t ≥ 0 be some positive and decreasing sequences and let ( δ t ) t ≥ 0 be a sequence satisfying the following:

- The sequence δ t follows the recursive relation:

<!-- formula-not-decoded -->

with δ 0 ≥ 0 and ω, η &gt; 0

<!-- formula-not-decoded -->

- Let γ t = c γ t -α with α ∈ (0 , 1) .
- Let t 0 = inf { t ≥ 1 : ωγ t +1 ≤ 1; ηγ t ≤ ω } .

Then, for all t ∈ N , we have the upper bound:

<!-- formula-not-decoded -->

Proof. Applying Proposition 5, for all t ≥ t 0 , we have the upper bound:

<!-- formula-not-decoded -->

Approximating the sum ∑ t s = t 0 γ s via a Riemann sum lower bound for the function x ↦→ 1 x α , and applying the logarithmic inequality log(1 -x ) ≤ -x , one can now bound ∏ t i = t 0 +1 (1 -ωγ i ) δ t 0 as

<!-- formula-not-decoded -->

In a same way, since we obtain

<!-- formula-not-decoded -->

Since the product involving exponential terms converges exponentially fast, we finally obtain the desired convergence rate

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->