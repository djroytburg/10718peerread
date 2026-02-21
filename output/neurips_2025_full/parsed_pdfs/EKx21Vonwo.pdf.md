## Nearly Dimension-Independent Convergence of Mean-Field Black-Box Variational Inference

## Kyurae Kim

University of Pennsylvania kyrkim@seas.upenn.edu

Trevor Campbell University of British Columbia trevor@stat.ubc.ca

Yi-An Ma

University of California San Diego yianma@ucsd.edu

Jacob R. Gardner University of Pennsylvania jacobrg@seas.upenn.edu

## Abstract

Weprove that, given a mean-field location-scale variational family, black-box variational inference (BBVI) with the reparametrization gradient converges at a rate that is nearly independent of any explicit dimension dependence. Specifically, for a d -dimensional strongly log-concave and log-smooth target, the number of iterations for BBVI with a sub-Gaussian family to obtain a solution ϵ -close to the global optimum has an explicit dimension dependence no larger than O(log d ) . This is a significant improvement over the O( d ) dependence of full-rank locationscale families. For heavy-tailed families, we prove a weaker O( d 2 /k ) dependence, where k is the number of finite moments of the family. Additionally, if the Hessian of the target log-density is constant, the complexity is free of any explicit dimension dependence. We also prove that our bound on the gradient variance, which is key to our result, cannot be improved using only spectral bounds on the Hessian of the target log-density.

## 1 Introduction

Variational inference (VI; Blei et al., 2017; Hinton and van Camp, 1993; Jordan et al., 1999; Peterson and Hartman, 1989) is an effective method for approximating intractable high-dimensional distributions and models with tall datasets. Among various VI algorithms, black-box VI (BBVI; Kucukelbir et al., 2017; Ranganath et al., 2014; Titsias and Lázaro-Gredilla, 2014; Wingate and Weber, 2013), which minimizes the exclusive KL divergence (Kullback and Leibler, 1951) via stochastic gradient descent (SGD; Bottou et al., 2018; Robbins and Monro, 1951) in the space of parameters, is widely used due to its flexibility to apply to a wide range of variational families with only minor modifications (Bingham et al., 2019; Carpenter et al., 2017; Fjelde et al., 2025; Ge et al., 2018; Patil et al., 2010). Specifically, location-scale variational families-in which a base distribution is mutated by an affine transformation-remain a popular choice, encompassing those with diagonal scale matrices (the 'mean-field' approximation; Hinton and van Camp, 1993; Peterson and Hartman, 1989), as well as scale matrices with low rank (Ong et al., 2018; Rezende et al., 2014; Tomczak et al., 2020) and full-rank (Kucukelbir et al., 2017; Titsias and Lázaro-Gredilla, 2014) factors.

The choice of the variational family is generally known to affect the convergence speed of BBVI, where families that are more 'expressive,' those that contain more complex distributions, result in slower convergence. For example, in location-scale families, it has been empirically observed that mean-field families often provide faster convergence to an accurate posterior approximation than full-rank families (Agrawal et al., 2020; Giordano et al., 2018, 2024; Ko et al., 2024; Zhang et al., 2022). This is because full-rank families often require running SGD with a smaller step size and

for longer; even given a large computation budget, BBVI on a full-rank family may not converge adequately (Ko et al., 2024). Therefore, choosing the expressiveness of the family corresponds to trading statistical accuracy for computational efficiency (Bhatia et al., 2022). In order to control this trade-off for our benefit, a clear theoretical understanding of the relationship between convergence speed and expressiveness is needed.

Formally, consider the setting of approximating a µ -strongly log-concave and L -log-smooth target, where κ ≜ L/µ is the condition number. For BBVI with the reparametrization gradient (Kingma and Welling, 2014; Rezende et al., 2014; Titsias and Lázaro-Gredilla, 2014) on a full-rank locationscale family, an ϵ -close solution to the global optimum in squared distance in parameter space can be obtained after at least O( dκ 2 ϵ -1 ) iterations (Domke, 2019; Kim et al., 2023a). For mean-field location-scale families, on the other hand, the iteration complexity improves to O( √ d κ 2 ϵ -1 ) (Kim et al., 2023a). While this is clearly better than the O( d ) explicit dimension dependence of full-rank families, it has been conjectured that a better dependence is more likely (Kim et al., 2023a).

In this work, we positively resolve this conjecture by obtaining stronger convergence guarantees for BBVI on mean-field location-scale families (Section 3). In particular, under the conditions stated above, we prove that BBVI with a mean-field location-scale family with sub-Gaussian tails can obtain an ϵ -accurate solution in squared distance after O((log d ) κ 2 ϵ -1 ) iterations. Heavier-tailed families achieve a weaker O( d 2 /k κ 2 ϵ -1 ) iteration complexity guarantee, where k is the number of finite moments of the variational family. For the Studentt variational family with a high-enough degrees of freedom ν , this corresponds to a O( d 2 / ( ν -2) ) explicit dimension dependence. In addition, if the Hessian of the target log-density is constant, any mean-field location-scale family attains a O( κ 2 ϵ -1 ) iteration complexity without any explicit dependence on d .

The key element of the proof is a careful probabilistic analysis of the variance of the reparametrization gradient (Section 4): In general, the reparametrization gradient of the scale parameters contains heavy-tailed components that grow not-so-slowly in d . However, for mean-field families, only a single random coordinate turns out to be heavy-tailed. Through a probabilistic decomposition, the influence of this heavy-tailed component can be averaged out over all d coordinates. Then the lighter-tailed components of the gradient dominate as d increases, resulting in a benign dimension dependence (Lemma 4.1). We also provide a lower bound (Proposition 4.2) showing that our analysis cannot be improved when using only spectral bounds on the Hessian of the target log-density.

## 2 Preliminaries

Notation We denote random variables in sans serif ( e.g. , u , U ). S d ≻ 0 ⊂ R d × d denotes the set of d × d positive definite (PD) matrices, D d ⊂ R d × d denotes the set of diagonal matrices, and D d ≻ 0 ⊂ D d ∩ S d × d ≻ 0 is its positive definite subset. 〈· , ·〉 and ‖·‖ 2 denote the Euclidean inner product and norm. For a matrix A ∈ R d × d , ‖ A ‖ F = √ tr( A ⊤ A ) is the Frobenius norm, ‖ A ‖ 2 = σ max ( A ) is the ℓ 2 operator norm, where σ max ( · ) and σ min ( · ) are the largest and smallest singular values.

## 2.1 Problem Setup

Our problem of interest is an optimization problem over some space Λ ⊆ R p of the form of

ℓ : R d → R is a measurable function we refer to as the 'target function', h : Λ → R is a potentially non-smooth convex regularizer, and the expectation E z ∼ q λ ℓ ( z ) is assumed to be intractable.

<!-- formula-not-decoded -->

BBVI is a special case of Eq. (1) where ℓ = -log π is the negative (unnormalized) log-density of some distribution π with respect to the Lebesgue measure and h ( λ ) = -H [ q λ ] is the negative differential entropy of q λ . Then F is the exclusive Kullback-Leibler divergence D KL (Kullback and Leibler, 1951) up to an additive constant (Jordan et al., 1999), where Eq. (1) reduces to

We assume π is supported on R d , which, unless discrete-valued variables are involved, is often valid after appropriate support transformations (Kim et al., 2023a, §2.2). Such a setup for BBVI has been proposed by Kucukelbir et al. (2017), and now encompasses most practical use of BBVI with the

<!-- formula-not-decoded -->

reparametrization gradient as implemented in Stan (Carpenter et al., 2017), PyMC (Patil et al., 2010), Pyro (Bingham et al., 2019), and Turing (Fjelde et al., 2025; Ge et al., 2018).

For the purpose of a quantitative theoretical analysis, we will consider the following properties:

Definition (Smoothness) . For some ϕ : R d → R , we say ϕ is L-(Lipschitz )smooth if there exists some L ∈ (0 , + ∞ ) such that, for all z, z ′ ∈ R d ,

Definition (Strong Convexity) . For some ϕ : R d → R , we say ϕ is µ -strongly convex if there exists some constant µ ∈ (0 , L ] such that, for all z, z ′ ∈ R d ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the context of BBVI, assuming that ℓ = -log π is both µ -strongly convex and L -smooth is equivalent to assuming π is µ -strongly log-concave and L -log-Lipschitz smooth, respectively, which is common in the analysis of MCMC (Chewi, 2024) and VI (Arnese and Lacker, 2024; Diao et al., 2023; Domke et al., 2023; Kim et al., 2023a; Lambert et al., 2022; Lavenant and Zanella, 2024).

## 2.2 Variational Family

We consider the location-scale family (Casella and Berger, 2001, §3.5):

Definition 2.1 (Location-Scale Variational Family) . A family of distributions Q is referred to as a location-scale variational family if there exists some univariate distribution φ dominated by the Lebesgue measure such that each member of Q indexed by λ = ( m,C ) ∈ R d ×C , where C ⊂ R d × d and q λ ∈ Q , satisfies where

and d = is equivalence in distribution. Then T λ is referred to as the 'reparametrization function, ' while m and C are referred to as the location and scale parameters, respectively.

<!-- formula-not-decoded -->

In addition, we impose mild regularity assumptions on the moments of the base distribution:

Assumption 2.2. φ satisfies the following: (i) It is standardized such that E u i = 0 and E u 2 i = 1 , (ii) symmetric such that E u 3 i = 0 , and (iii) its kurtosis is finite such that E u 4 i = r 4 &lt; ∞ .

The location-scale family with Assumption 2.2 encompasses many variational families used in practice, such as Gaussians, Studentt with a high-enough degrees of freedom ν , Laplace, and so on, and enables the use of the reparametrization gradient.

While the choice of φ gives control over the tail behavior of the family, the choice of the structure of the scale matrix C gives control over how much correlation between coordinates of ℓ the variational approximation can represent. This ability to represent correlations is often referred to as the 'expressiveness' of a variational family, where the most expressive choice is the following:

Definition 2.3 (Full-Rank Location-Scale Family) . We say Q is a full-rank location-scale family if it satisfies Definition 2.1 and, for any C ∈ C , C is invertible and the squared C s, CC ⊤ , span the whole space of dense R d × d positive definite matrices as { CC ⊤ | C ∈ C} = S d ≻ 0 .

Typically, full-rank location-scale families are formed by setting C to be the set of invertible triangular matrices (the 'Cholesky factor parametrization'; Kucukelbir et al., 2017; Titsias and LázaroGredilla, 2014) or the set of symmetric square roots (Domke, 2020; Domke et al., 2023). Adding further restrictions on C forms various subsets of the broader location-scale family. In this work, we focus on the case where C ∈ C is restricted to be diagonal such that C ⊂ D d , which is known as the mean-field approximation (Hinton and van Camp, 1993; Peterson and Hartman, 1989):

Definition 2.4 (Mean-Field Location-Scale Family) . We say Q is a mean-field location-scale family if it satisfies Definition 2.1 and all C ∈ C are diagonal such that C ⊂ D d .

## 2.3 Algorithm Setup

Recall that BBVI is essentially SGD in the space of parameters of the variational distribution. Therefore, we have to define the space of parameters. For this, we use the 'linear' parametrization:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Under this parametrization, the desirable properties of ℓ easily transfer to f . For instance, if ℓ is µ -strongly convex and L -smooth, f is also µ -strongly convex and L -smooth (Domke, 2020). This contrasts with 'non-linear parametrizations' commonly used in practice, such as making the diagonal positive by C ii = exp( λ C ii ) . Such practice rules out transfer of strong convexity and smoothness (Kim et al., 2023a) unless constraints such as C ii ≥ δ for some δ &gt; 0 are enforced (Hotti et al., 2024). (Though they can sometimes be beneficial by reducing gradient variance; Hotti et al., 2024; Kim et al., 2023b.) The flip side of using the linear parametrization is that we must now enforce the constraint C glyph[follows] 0 . Furthermore, h then becomes non-smooth with respect to C :

h ( λ ) = -H ( q λ ) = -log | det C | -d H ( φ ) = -∑ d i =1 log | C ii | -d H ( φ ) . (4) This corresponds to log-barrier functions (Parikh and Boyd, 2014, §6.7.5), which are non-smooth. Thus, the optimization algorithm must somehow deal with these difficulties (Domke, 2020).

In this work, we will rely on the proximal variant of stochastic gradient descent (SGD; Bottou, 1999; Bottou et al., 2018; Nemirovski et al., 2009; Robbins and Monro, 1951; Shalev-Shwartz et al., 2011), often referred to as stochastic proximal gradient descent (SPGD; Nemirovski et al., 2009). Proximal methods are a family of methods that rely on proximal operators (Parikh and Boyd, 2014), which are well defined as long as the following hold:

Assumption 2.5. h : Λ → R ∪ { + ∞} is convex, bounded below, and lower semi-continuous.

<!-- formula-not-decoded -->

The non-smoothness of h and the domain constraint are handled by the proximal operator while the intractability of f is handled through stochastic estimates of ∇ f (Definition 2.6). For a step size schedule ( γ t ) t ≥ 0 , ̂ ∇ f , an unbiased estimator of ∇ f ( λ t ) = E ̂ ∇ f ( λ t ; u ) , and a sequence of i.i.d. noise ( u t ) t ≥ 0 , for each t ≥ 0 , SPGD iterates

<!-- formula-not-decoded -->

λ t +1 = prox γ t h ( λ t -γ t ̂ ∇ f ( λ ; u t ) ) . In the case of BBVI with a mean-field location-scale family, the proximal operator of Eq. (4) is identical to that of log-barrier functions (Parikh and Boyd, 2014, §6.7.5):

Instead of using SPGD, one can also use projected SGD, where C is projected to a subset where F is smooth (Domke, 2020) and use the 'closed-form entropy' gradient ̂ ∇ F ≜ ̂ ∇ f + ∇ h (Kucukelbir et al., 2017; Titsias and Lázaro-Gredilla, 2014). However, the resulting theoretical guarantees are indistinguishable (Domke et al., 2023), and the need for setting a closed domain of C is inconvenient. Therefore, we only consider SPGD. But our results can easily be applied to projected SGD.

For ̂ ∇ f , we will use the classic reparametrization gradient (Ho and Cao, 1983; Rubinstein, 1992): Definition 2.6 (Reparametrization Gradient) . For a differentiable function ℓ : R d → R , is an unbiased estimator of ∇ f such that ∇ λ E z ∼ q λ ℓ ( z ) = ∇ f ( λ ) .

<!-- formula-not-decoded -->

The reparametrization gradient, also known as the push-in gradient or pathwise gradient, was introduced to VI by Kingma and Welling (2014); Rezende et al. (2014); Titsias and Lázaro-Gredilla (2014). (See also the reviews by Glasserman 1991; Mohamed et al. 2020; Pflug 1996.) It is empirically observed to outperform alternatives (Kucukelbir et al., 2017; Mohamed et al., 2020) such as the score gradient (Glynn, 1990; Williams, 1992) and de facto standard whenever ℓ is differentiable. (Though theoretical evidence of this superiority is limited to the quadratic setting; Xu et al., 2019.)

## 2.4 General Analysis of Stochastic Proximal Gradient Descent

Analyzing the convergence of BBVI corresponds to analyzing the convergence of SPGD (or more broadly, of SGD) for the class of problems that corresponds to BBVI. For this, we will first discuss sufficient conditions for the convergence of SPGD and the resulting consequences.

Assumption 2.7 (Lipschitz Gradients in Expectation) . There exists some constant L ∈ [0 , ∞ ) such that, for all λ, λ ′ ∈ Λ ,

<!-- formula-not-decoded -->

Assumption 2.8 (Bounded Variance) . There exists some constant σ ∈ [0 , ∞ ) such that, for all λ ∗ ∈ arg min λ ∈ Λ F ( λ ) ,

E ‖ ̂ ∇ f ( λ ∗ ; u ) ‖ 2 2 ≤ σ 2 . Both assumptions were initially used by Bach and Moulines (2011, Assumptions H2 and H4) to analyze the convergence of SGD. Here, Assumption 2.7 serves as an analog of L -smoothness, and thus determines the largest stepsize we can use. The strategy of combining Assumptions 2.7 and 2.8 is referred to as 'variance transfer' (Garrigos and Gower, 2023, §4.3.3). Previously, for analyzing BBVI, a slightly different assumption called quadratically-bounded variance (QV)-which assumes the existence of α, β ∈ [0 , + ∞ ) such that, for all λ ∈ Λ , E ‖ ̂ ∇ f ( λ ; u ) ‖ 2 2 ≤ α ‖ λ -λ ∗ ‖ 2 2 + β holdshas been commonly used (Domke, 2019; Domke et al., 2023; Kim et al., 2024b). While similar, our assumptions result in a constant-factor improvement in the resulting bounds.

For the analysis, we will use a two-stage step size schedule (Gower et al., 2019, Theorem 3.2):

<!-- formula-not-decoded -->

This operates by first maintaining a fixed step size γ 0 until some switching time t ∗ ∈ { 0 , . . . , T } , and then switches to the 1 /t schedule of Lacoste-Julien et al. (2012) with an offset τ ≥ 0 .

Under Assumptions 2.7 and 2.8, we can now provide a complexity guarantee for solving Eq. (1) via SPGD. Since BBVI consists of a subset of Eq. (1), establishing Assumptions 2.7 and 2.8 and invoking the following result will constitute our complexity guarantee for BBVI.

Proposition 2.9. Suppose f is µ -strongly convex, h satisfies Assumption 2.5, and ̂ ∇ f satisfies Assumptions 2.7 and 2.8. Then, for the global optimum λ ∗ = arg min λ ∈ Λ F ( λ ) , ∆ ≜ ‖ λ 0 -λ ∗ ‖ 2 , some t ∗ , τ , and γ 0 (explicit in the proof), SPGD with the step size schedule in Eq. (5) guarantees

<!-- formula-not-decoded -->

Proof . See the full proof

<!-- formula-not-decoded -->

This result is a slight improvement over past analysis of SPGD with Eq. (5) (Domke et al., 2023, Theorem 7). In particular, the dependence on the initialization ∆ has been improved to be logarithmic instead of polynomial. Furthermore, it encompasses the case where we have 'interpolation' ( σ 2 = 0 ; Kim et al., 2024b; Schmidt and Roux, 2013; Vaswani et al., 2019) automatically resulting in a O(log 1 /ϵ ) complexity. A similar result for vanilla SGD, where the dependence on both ϵ and ∆ is optimized simultaneously, was reported by Stich (2019). However, this result required a schedule that depends on the maximum number of iterations T . Compared with this, our result provides an any-time guarantee that holds for any number of iterations.

For a non-strongly convex f , using the strategy of Domke et al. (2023, Theorem 8 and 11) should yield a corresponding O ( 1 /ϵ 2 ) complexity guarantee under the same set of assumptions. However, this requires fixing the horizon T in advance, and it is currently unknown how to obtain an anytime O(1 / √ T ) convergence bound for SGD under Assumptions 2.7 and 2.8 or QV. If one moves away from the canonical SGD update by incorporating Halpern iterations (Halpern, 1967), it is possible to obtain any-time convergence under a QV-like assumption (Alacaoglu et al., 2025).

## 3 Main Results

## 3.1 General Result

For our results, we impose an additional assumption that is a generalization of L -smoothness under twice differentiability of ℓ .

Assumption 3.1. ℓ is twice differentiable and, for all z ∈ R d , there exist some matrix H ∈ R d × d and constant δ ∈ [0 , ∞ ) satisfying

<!-- formula-not-decoded -->

Notably, if ℓ is twice differentiable, µ -strongly convex, L -smooth, it already satisfies Assumption 3.1 with H = L + µ 2 I d and δ = L -µ 2 . If ℓ is only L -smooth, it satisfies it with H = 0 d × d and δ = L . The key advantage of this assumption, however, is that it characterizes Hessians that are not necessarily well-conditioned, but almost constant. This crucially affects the dimension dependence.

Given our assumptions on the target function ℓ , variational family Q , and our choice of gradient estimator, we can guarantee that SPGD applied to a problem structure corresponding to BBVI (Eq. (1)) achieves a given level of accuracy ϵ after O( g ( d, H, δ, µ, φ ) ϵ -1 ) number of iterations:

Theorem 3.2. Suppose the following hold:

1. ℓ is µ -strongly convex and satisfies Assumption 3.1 and µ ≤ σ min ( H ) ≤ σ max ( H ) ≤ L .
2. h satisfies Assumption 2.5.
3. Q is a mean-field location-scale family, where Assumption 2.2 holds.

T ≥ O { g ( d, H, δ, µ, φ ) ( σ 2 ∗ ϵ -1 + σ ∗ log ( ‖ λ 0 -λ ∗ ‖ 2 2 ) ϵ -1 / 2 )} ⇒ E ‖ λ T -λ ∗ ‖ 2 2 ≤ ϵ , where g ( d, H, δ, µ, φ ) ≜ 2 (1 + r 4 ) ( ‖ H ‖ 2 2 /µ 2 ) +4 ( δ 2 /µ 2 ) ( (1 / 2) + r 4 + E max j =1 ,...,d u 2 j ) . Proof. The full proof can be found in Appendix B.2.1, p. 25.

4. ̂ ∇ f is the reparametrization gradient. Denote the global optimum λ ∗ = ( m ∗ , C ∗ ) = arg min λ ∈ Λ F ( λ ) , the irreducible gradient noise as σ 2 ∗ ≜ ‖ m ∗ -¯ z ‖ 2 2 + ‖ C ∗ ‖ 2 F , and the stationary point of ℓ as ¯ z ≜ arg min z ∈ R d ℓ ( z ) . Then, for some t ∗ , τ , and γ 0 (explicit in the proof), SPGD with the step size schedule in Eq. (5) guarantees

Due to the identity ‖ λ -λ ′ ‖ 2 2 = E u ∼ φ ⊗ d ‖T λ ( u ) - T λ ′ ( u ) ‖ 2 2 (Lemma A.3), which is the squared cost of a coupling between q λ T and q λ ∗ , our guarantee also translates to a guarantee in Wasserstein2 distance: E ‖ λ T -λ ∗ ‖ 2 2 ≤ ϵ ⇒ E W 2 ( q λ T , q λ ∗ ) 2 ≤ ϵ . In the general case where δ &gt; 0 , the dimension dependence enters through E max j =1 ,...,d u 2 j , which depends on the order-statistics of the base distribution φ . In case ℓ is a quadratic, corresponding to π being a Gaussian target distribution in the BBVI context, there exists some H such that ∇ 2 ℓ ( z ) = H for all z ∈ R d . Thus, Assumption 3.1 holds with δ = 0 , implying a dimension-independent convergence rate. We will present additional special cases with more explicit choices of φ in the next section.

In case we do not want to assume Assumption 3.1 and only assume that ℓ is µ -strongly convex and L -smooth instead, we can replace them with the generic choices of H = L + µ 2 I d and δ = L -µ 2 , which hold for all ℓ s that are µ -strongly convex, L -smooth, and twice differentiable. This then makes the role of the condition number κ ≜ L/µ more explicit.

Corollary 3.3. Suppose ℓ is is twice differentiable, µ -strongly convex, and L -smooth. Then, denoting the condition number as κ ≜ L/µ , Theorem 3.2 holds with

This makes the O( κ 2 ) condition number dependence explicit, but the downside is that we lose dimension independence in the case of ill-conditioned quadratic ℓ s. This fact suggests that dimension dependence is more fundamentally related to how close the Hessian is to a constant rather than how well-conditioned it is.

<!-- formula-not-decoded -->

## 3.2 Special Cases with Benign Dimension Dependence

We now present some special cases of Theorem 3.2, which has yet to exhibit an explicit dependence on dimensionality. As mentioned in the previous section, dimension dependence depends on the order statistics of φ , which is related to the tail behavior of φ .

Variational Families with Sub-Gaussian Tails. The most commonly used variational family in practice is the Gaussian variational family. More broadly, for sub-Gaussian variational families, u 2 i is sub-exponential and therefore admits a moment generating function (MGF) (Wainwright, 2019, Theorem 2.6), which leads to a O(log d ) explicit dimension dependence.

Proposition 3.4. Suppose there exists some t &gt; 0 such that the MGF of u 2 i satisfies M u 2 i ( t ) &lt; ∞ . Then

For example, if φ is a standard Gaussian, then

<!-- formula-not-decoded -->

Proof. The full proof can be found in Appendix B.2.2, p. 26.

<!-- formula-not-decoded -->

Variational Families with Finite Higher Moments. For families with tails heavier than subGaussian, however, u 2 i may not have an MGF. While we then lose the O(log d ) dependence, we may still obtain a polynomial dependence that can be better than O( √ d ) obtained in previous works (Kim et al., 2023b). In particular, the result that will follow states that the highest order of the available moments determines the order of dimension dependence. For Studentt families, this implies that using a high-enough degree of freedom ν can make the dimension dependence benign.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For example, if φ is a Studentt with ν &gt; 4 degrees of freedom and unit variance, then

<!-- formula-not-decoded -->

Proof . See the full proof in Appendix B.2.3, p. 28.

<!-- formula-not-decoded -->

## 4 Analysis of Gradient Variance

## 4.1 Overview

The key technical contribution of this work is analyzing the gradient variance and thus establishing the constants L (Assumption 2.7) and σ 2 (Assumption 2.8), which boils down to analyzing

∥ ∥ where the equality follows from the fact that the Jacobian ∂ T λ ( u ) / ∂λ does not depend on λ . For mean-field location-scale variational families, the squared Jacobian follows as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where U ≜ diag( u 1 , . . . , u d ) (Kim et al., 2023b). This implies that

E ‖ ̂ ∇ f ( λ ; u ) -̂ ∇ f ( λ ′ ; u ) ‖ 2 2 = E ‖∇ ℓ ( T λ ( u )) -∇ ℓ ( T λ ′ ( u )) ‖ 2 2 ︸ ︷︷ ︸ ≜ V loc + E ‖ U ( ∇ ℓ ( T λ ( u )) -∇ ℓ ( T λ ′ ( u ))) ‖ 2 2 ︸ ︷︷ ︸ ≜ V scale . (6) Our goal is to bound each term by ‖ λ -λ ′ ‖ 2 2 .

In order to solve the expectations, we need to simplify the ∇ ℓ terms. For instance, for the gradient of the location V loc, assuming that ℓ is L -smooth allows for a quadratic approximation. That is,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last equality is by Lemma A.3.

Now, it is tempting to use the same quadratic approximation strategy for the gradient of the scale V scale . Indeed, this strategy was used by Domke (2019) to bound the gradient variance of full-rank location-scale variational families and by Ko et al. (2024) for structured location-scale variational families. Unfortunately, this strategy does not immediately apply to mean-field families due to the matrix U . We somehow have to decouple ∇ ℓ ( T λ ( u )) -∇ ℓ ( T λ ′ ( u )) and U , but in a way that does not lose the correlation between the two; the correlation leads to cancellations critical to obtaining a tight bound. Kim et al. (2023b) used the inequality

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which resulted in a dimension dependence of O( r 4 √ d ) after solving the expectation. The key question is whether this dimension dependence can be improved. Due to the ordering of norms ‖·‖ 2 ≤ ‖·‖ F , it is natural to consider the tighter inequality

(This step corresponds to Eq. (8) in the proof sketch of the upcoming result.) The main challenge, however, is solving the resulting expectation in a way that is also tight with respect to d . We will see that this requires a careful probabilistic analysis.

## 4.2 Upper Bound on Gradient Variance

We now formally state our upper bound on the gradient variance. In the context of proving Theorem 3.2, the following lemma implies both Assumption 2.7 and Assumption 2.2. (See the proof of Theorem 3.2.) We provide a corresponding unimprovability result in Section 4.3.

Lemma 4.1. Suppose Assumptions 2.2 and 3.1 hold, Q is a mean-field location-family, and ̂ ∇ f is the reparametrization gradient. Then, for any λ, λ ′ ∈ R d × D d .

Proof Sketch. For the proof sketch, we will assume that ℓ is L -smooth instead of taking Assumption 3.1. This will vastly simplify the analysis and let us focus on the key elements.

<!-- formula-not-decoded -->

Recall V scale in Eq. (6). Applying the operator norm and the L -smoothness of ℓ yields

<!-- formula-not-decoded -->

It remains to solve the expectation over u , . . . , u

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

where i ∗ = arg max i =1 ,...,d u 2 i is the coordinate of maximum magnitude. Here, u 4 i ∗ = max i =1 ,...,d u 4 i is a heavy-tailed quantity that generally grows fast in d , unlike u 2 i ∗ . ( e.g. , for a Gaussian u i , u 2 i ∗ has an MGF but u 4 i ∗ does not.) Therefore, a benign dimension dependence might appear futile. Notice, however, that the problematic term only affects a single dimension: the coordinate indicated by i ∗ . A probabilistic analysis reveals that as d increases, the effect of u 4 i ∗ becomes averaged out and the effect of the remaining term involving u 2 i dominates. More formally, recall that T λ ( u ) = Cu + m (Definition 2.1), and notice that, since U is a diagonal matrix, ‖ U ‖ 2 2 = max i =1 ,...,d u 2 j . Then we can rewrite Eq. (8) as

The problematic term is

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

Since the maximum of d random variable is always smaller than their sum, the probability of the maximally random event, P [ i ∗ = i ] = 1 /d , kills off the dimensional growth of u 4 i ∗ . In fact, using the crude bound E u 4 i ∗ ≤ E ∑ d i =1 u 4 i = dr 4 , where the last equality is due to Assumption 2.2, is enough to make this term independent of d . The remaining dimension dependence comes from u 2 i ∗ :

glyph[negationslash]

E [ u 2 i ∗ u 2 i 1 { i ∗ = i } ] = E [ max j = i u 2 j u 2 i 1 { i ∗ = i } ] ≤ E [ max j =1 ,...,d -1 u 2 j ] E [ u 2 i ] = E max j =1 ,...,d -1 u 2 j , where the last equality follows from Assumption 2.2. Therefore, we finally obtain

̸

glyph[negationslash]

<!-- formula-not-decoded -->

The full proof performs an analogous analysis under the more general Assumption 3.1. See the full proof in Appendix B.3.1, p. 30.

## 4.3 Unimprovability

Wealso demonstrate a lower bound, which implies that Lemma 4.1 cannot be improved when relying on the spectral bounds on ∇ 2 ℓ . From Eq. (6) and the fundamental theorem of calculus,

E ‖ ̂ ∇ f ( λ ; u ) ‖ 2 2 ≥ E ‖ U ( ∇ ℓ ( z ) -∇ ℓ (¯ z )) ‖ 2 2 = E ∥ ∥ U ∫ 1 0 ∇ 2 ℓ ( z w )( z -¯ z )d w ∥ ∥ 2 2 , (9) where ¯ z ∈ { z | ∇ ℓ ( z ) = 0 } is a stationary point of ℓ , z ≜ T λ ( u ) , and z w ≜ w z +(1 -w )¯ z . There exists a matrix-valued function with bounded singular values that lower-bounds this quantity:

Proposition 4.2. Suppose Assumption 2.2 holds and Q is a mean-field location-scale family. Then, for any t &gt; 0 , d &gt; 0 , µ, L ∈ (0 , + ∞ ) satisfying µ ≤ L , there exists a matrix-valued function H ( z ) : R d → S d ≻ 0 satisfying µ I d glyph[precedesequal] H glyph[precedesequal] L I d almost surely and a set of parameters λ = ( m,C ) ∈ R d × D d ≻ 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For Gaussians, E max i u 4 i is upper bounded as O( √ d ) (Gumbel, 1954, Eq. 1.6), which means the negative term vanishes at a 1 / √ d rate. Furthermore, E max i u 2 i ≥ ( E max i u i ) 2 = Ω(log d ) by the well-known lower bound on the expected maximum of i.i.d. Gaussians (Wainwright, 2019, Exercise 2.11.(b)). Combining these facts with Proposition 4.2 yield a Ω ( L 2 log d ) bound on Eq. (9).

Remark 4.3. It is not obvious that the rows of our worst-case example H worst form conservative vector fields. This means that Proposition 4.2 does not assert the existence of a function ℓ that satisfies ∇ 2 ℓ = H worst . However, it does suggest that one cannot improve Lemma 4.1 by relying only on spectral bounds on the Hessian.

## 5 Discussion

## 5.1 Related Works

Early results analyzing VI had to rely on assumptions that either: (i) do not hold on Gaussian targets, (ii) are difficult to verify, or (iii) require bounds on the domain (Alquier and Ridgway, 2020; Buchholz et al., 2018; Fan et al., 2015; Fujisawa and Sato, 2021; Khan et al., 2016; Liu and Owen, 2021; Nguyen et al., 2025; Regier et al., 2017). coordinate-ascent VI (CAVI), in particular, was studied on specific models (Ghorbani et al., 2019; Zhang and Zhou, 2020) only. Under general and verifiable assumptions, Xu and Campbell (2022) obtained asymptotic convergence guarantees, while partial results, such as bounds on the gradient variance (Domke, 2019; Fan et al., 2015; Kim et al., 2023b), or regularity of the ELBO (Challis and Barber, 2013; Domke, 2020; Titsias and LázaroGredilla, 2014), were known.

It was only recently that non-asymptotic quantitative convergence under realizable and verifiable assumptions was established. For BBVI specifically, Hoffman and Ma (2020) first proved convergence on Gaussian targets (quadratic ℓ ), while Domke et al. (2023); Kim et al. (2023a, 2024b) proved the first results on strongly convex and smooth functions with location-scale families. Surendran et al. (2025) extended these results to non-convex smooth functions and more complex variational family parametrizations, and Cheng et al. (2024) analyzed a variant of semi-implicit VI. The results of Domke et al.; Hoffman and Ma; Kim et al., who focused on full-rank families, suggest a O( d ) dimension dependence in the iteration complexity. On the other hand, Kim et al. (2023a) reported a O( √ d ) dimension dependence for mean-field location-scale families, while conjecturing O(log d ) dependence, based on the partial result of Kim et al. (2023b). For targets with a diagonal Hessian structure, Ko et al. (2024, Corollary 1) show that mean-field families are dimension-independent.

Apart from BBVI, Wasserstein VI algorithms-which minimize the KL divergence on the Wasserstein geometry-provide non-asymptotic convergence guarantees. In particular, the algorithms by Diao et al. (2023); Lambert et al. (2022) optimize over the full-rank Gaussian family, while that of Jiang et al. (2025) optimizes over all mean-field families with bounded second moments. To guarantee E W 2 ( q λ T , q λ ∗ ) 2 ≤ ϵ on strongly log-concave and log-smooth targets, they all report an iteration complexity of O( dϵ -1 log ϵ -1 ) . Meanwhile, under the same conditions, Arnese and Lacker

(2024); Lavenant and Zanella (2024) analyzed (block) CAVI, and reported an iteration complexity of O( d log ϵ -1 ) . Bhattacharya et al. (2025) provides a concurrent result on CAVI, but relies on an assumption that departs from log-concavity and smoothness. Finally, Bhatia et al. (2022) analyzes a specialized algorithm optimizing over only the scale of Gaussians, which has a gradient query complexity of O( dkϵ -3 ) , where k is the user-chosen number of rank-1 factors in the scale matrix.

## 5.2 Conclusion

In this work, we proved that BBVI with mean-field location-scale families is able to converge with an iteration complexity with only a O(log d ) dimension dependence, as long as the tails of the family are sub-Gaussian. For high-dimensional targets, this suggests a substantial speed advantage over BBVI with full-rank families. In practice, the mean-field approximation can be combined with other design elements such as control variates (Boustati et al., 2020; Geffner and Domke, 2018, 2020; Miller et al., 2017; Roeder et al., 2017; Wang et al., 2024) and data-point subsampling (Kucukelbir et al., 2017; Titsias and Lázaro-Gredilla, 2014). Our analysis strategy should easily be combined with existing analyses (Kim et al., 2024a,b) for such design elements.

For a target distribution π with a condition number of κ and a target accuracy level ϵ , we now know how to improve the dependence on d and ϵ in the iteration complexity: Using less-expressive families such as mean-field (Theorem 3.2) or structured (Ko et al., 2024) families improves the dependence on d , while applying control variates to gradient estimators (Kim et al., 2024b) improves the dependence on ϵ . However, it is currently unclear whether the dependence on κ is tight or improvable. If it is tight, it would be worth investigating whether this can be provably improved through algorithmic modifications, for example, via stochastic second-order optimization methods (Byrd et al., 2016; Fan et al., 2015; Liu and Owen, 2021; Meng et al., 2020; Regier et al., 2017).

Another future direction would be to develop methods that are able to adaptively adjust the computational cost between O(log d ) and O( d ) by trading statistical accuracy akin to the method of Bhatia et al. (2022). Existing BBVI schemes with 'low-rank(-plus-diagonal)' families (Ong et al., 2018; Rezende et al., 2014; Tomczak et al., 2020) result in a non-smooth, non-Lipschitz, and non-convex landscape. This not only rules out typical theoretical convergence guarantees but also exhibits unstable and slow convergence in practice (Modi et al., 2025). Furthermore, understanding the statistical side of this trade-off will be an important direction. As of now, our understanding is restricted to either mean-field or full-rank families (Katsevich and Rigollet, 2024; Margossian and Saul, 2023, 2025; Wang and Blei, 2019a,b; Yang et al., 2020; Zhang and Gao, 2020) with little in between except for the work of Bhatia et al. (2022).

## Acknowledgments and Disclosure of Funding

The authors thank Anton Xue for helpful discussions and the reviewers for helpful suggestions.

K. Kim, J. R. Gardner were supported through the NSF award [IIS2145644]; Y.-A. Ma was supported by the NSF Award CCF-2112665 (TILOS), the DARPA AIE program, and the CDC-RFAFT-23-0069; T. Campbell was supported by the NSERC Discovery Grant RGPIN-2025-04208.

## References

- Abhinav Agrawal, Daniel R Sheldon, and Justin Domke. Advances in black-box VI: Normalizing flows, importance weighting, and optimization. In Advances in Neural Information Processing Systems , volume 33, pages 17358-17369. Curran Associates, Inc., 2020. (page 1)
- Ahmet Alacaoglu, Yura Malitsky, and Stephen J. Wright. Towards weaker variance assumptions for stochastic optimization. arXiv Preprint arXiv:2504.09951, 2025. (page 5)
- Pierre Alquier and James Ridgway. Concentration of tempered posteriors and of their variational approximations. The Annals of Statistics , 48(3):1475-1497, 2020. (page 9)
- Manuel Arnese and Daniel Lacker. Convergence of coordinate ascent variational inference for logconcave measures via optimal transport. arXiv Preprint arXiv:2404.08792, 2024. (pages 3, 9)
- Francis Bach and Eric Moulines. Non-asymptotic analysis of stochastic approximation algorithms for machine learning. In Advances in Neural Information Processing Systems , volume 24, pages 451-459. Curran Associates, Inc., 2011. (page 5)
- Kush Bhatia, Nikki Lijing Kuang, Yi-An Ma, and Yixin Wang. Statistical and computational trade-offs in variational inference: A case study in inferential model selection. arXiv Preprint arXiv:2207.11208, 2022. (pages 2, 10)
- Anirban Bhattacharya, Debdeep Pati, and Yun Yang. On the convergence of coordinate ascent variational inference. The Annals of Statistics , 53(3):929-962, 2025. (page 10)
- Eli Bingham, Jonathan P. Chen, Martin Jankowiak, Fritz Obermeyer, Neeraj Pradhan, Theofanis Karaletsos, Rohit Singh, Paul Szerlip, Paul Horsfall, and Noah D. Goodman. Pyro: Deep universal probabilistic programming. Journal of Machine Learning Research , 20(28):1-6, 2019. (pages 1, 3)
- David M. Blei, Alp Kucukelbir, and Jon D. McAuliffe. Variational inference: A review for statisticians. Journal of the American Statistical Association , 112(518):859-877, 2017. (page 1)
- Léon Bottou. On-line learning and stochastic approximations. In On-Line Learning in Neural Networks , pages 9-42. Cambridge University Press, 1 edition, 1999. (page 4)
- Léon Bottou, Frank E. Curtis, and Jorge Nocedal. Optimization methods for large-scale machine learning. SIAM Review , 60(2):223-311, 2018. (pages 1, 4)
- Ayman Boustati, Sattar Vakili, James Hensman, and S. T. John. Amortized variance reduction for doubly stochastic objective. In Proceedings of the Conference on Uncertainty in Artificial Intelligence , volume 124 of PMLR , pages 61-70. JMLR, 2020. (page 10)
- Alexander Buchholz, Florian Wenzel, and Stephan Mandt. Quasi-Monte Carlo variational inference. In Proceedings of the International Conference on Machine Learning , volume 80 of PMLR , pages 668-677. JMLR, 2018. (page 9)
- R. H. Byrd, S. L. Hansen, Jorge Nocedal, and Y. Singer. A stochastic quasi-Newton method for large-scale optimization. SIAM Journal on Optimization , 26(2):1008-1031, 2016. (page 10)
- Bob Carpenter, Andrew Gelman, Matthew D. Hoffman, Daniel Lee, Ben Goodrich, Michael Betancourt, Marcus Brubaker, Jiqiang Guo, Peter Li, and Allen Riddell. Stan: A probabilistic programming language. Journal of Statistical Software , 76(1):1-32, 2017. (pages 1, 3)
- George Casella and Roger L. Berger. Statistical Inference . Cengage Learning, 2 edition, 2001. (page 3)
- Edward Challis and David Barber. Gaussian Kullback-Leibler approximate inference. Journal of Machine Learning Research , 14(68):2239-2286, 2013. (page 9)

- Ziheng Cheng, Longlin Yu, Tianyu Xie, Shiyue Zhang, and Cheng Zhang. Kernel semi-implicit variational inference. In Proceedings of the International Conference on Machine Learning , volume 235 of PMLR , pages 8248-8269. JMLR, 2024. (page 9)
- Sinho Chewi. Log-Concave Sampling . Unpublished draft, november 3, 2024 edition, 2024. URL https://chewisinho.github.io/main.pdf . (page 3)
- Michael Ziyang Diao, Krishna Balasubramanian, Sinho Chewi, and Adil Salim. Forward-backward Gaussian variational inference via JKO in the Bures-Wasserstein space. In Proceedings of the International Conference on Machine Learning , volume 202 of PMLR , pages 7960-7991. JMLR, 2023. (pages 3, 9)
- Justin Domke. Provable gradient variance guarantees for black-box variational inference. In Advances in Neural Information Processing Systems , volume 32, pages 329-338. Curran Associates, Inc., 2019. (pages 2, 5, 7, 9)
- Justin Domke. Provable smoothness guarantees for black-box variational inference. In Proceedings of the International Conference on Machine Learning , volume 119 of PMLR , pages 2587-2596. JMLR, 2020. (pages 3, 4, 9, 25)
- Justin Domke, Robert Gower, and Guillaume Garrigos. Provable convergence guarantees for blackbox variational inference. In Advances in Neural Information Processing Systems , volume 36, pages 66289-66327. Curran Associates, Inc., 2023. (pages 3, 4, 5, 9, 19)
- Kai Fan, Ziteng Wang, Jeff Beck, James Kwok, and Katherine A Heller. Fast second order stochastic backpropagation for variational inference. In Advances in Neural Information Processing Systems , volume 28, pages 1387-1395. Curran Associates, Inc., 2015. (pages 9, 10)
- Tor Erlend Fjelde, Kai Xu, David Widmann, Mohamed Tarek, Cameron Pfiffer, Martin Trapp, Seth D. Axen, Xianda Sun, Markus Hauru, Penelope Yong, Will Tebbutt, Zoubin Ghahramani, and Hong Ge. Turing.jl: A general-purpose probabilistic programming language. ACM Transactions on Probabilistic Machine Learning , 1(3):1-48, 2025. (pages 1, 3)
- Masahiro Fujisawa and Issei Sato. Multilevel Monte Carlo variational inference. Journal of Machine Learning Research , 22(278):1-44, 2021. (page 9)
- Guillaume Garrigos and Robert M. Gower. Handbook of convergence theorems for (stochastic) gradient methods. arXiv Preprint arXiv:2301.11235, 2023. (pages 5, 22)
- Hong Ge, Kai Xu, and Zoubin Ghahramani. Turing: A language for flexible probabilistic inference. In Proceedings of the International Conference on Machine Learning , volume 84 of PMLR , pages 1682-1690. JMLR, 2018. (pages 1, 3)
- Tomas Geffner and Justin Domke. Using large ensembles of control variates for variational inference. In Advances in Neural Information Processing Systems , volume 31, pages 9960-9970. Curran Associates, Inc., 2018. (page 10)
- Tomas Geffner and Justin Domke. Approximation based variance reduction for reparameterization gradients. In Advances in Neural Information Processing Systems , volume 33, pages 2397-2407. Curran Associates, Inc., 2020. (page 10)
- Behrooz Ghorbani, Hamid Javadi, and Andrea Montanari. An instability in variational inference for topic models. In Proceedings of the International Conference on Machine Learning , volume 97 of PMLR , pages 2221-2231. JMLR, 2019. (page 9)
- Ryan Giordano, Tamara Broderick, and Michael I. Jordan. Covariances, robustness, and variational Bayes. Journal of Machine Learning Research , 19(51):1-49, 2018. (page 1)
- Ryan Giordano, Martin Ingram, and Tamara Broderick. Black box variational inference with a deterministic objective: Faster, more accurate, and even more black box. Journal of Machine Learning Research , 25:1-39, 2024. (page 1)
- Paul Glasserman. Gradient Estimation via Perturbation Analysis . Number 116 in The Springer International Series in Engineering and Computer Science. Springer, New York, NY, 1991. (page 4)
- Peter W. Glynn. Likelihood ratio gradient estimation for stochastic systems. Communications of the ACM , 33(10):75-84, 1990. (page 4)
- Eduard Gorbunov, Filip Hanzely, and Peter Richtarik. A unified theory of SGD: Variance reduction, sampling, quantization and coordinate descent. In Proceedings of the International Conference on Artificial Intelligence and Statistics , volume 108 of PMLR , pages 680-690. JMLR, 2020. (page 22)

- Robert Mansel Gower, Nicolas Loizou, Xun Qian, Alibek Sailanbayev, Egor Shulgin, and Peter Richtárik. SGD: General analysis and improved rates. In Proceedings of the International Conference on Machine Learning , volume 97 of PMLR , pages 5200-5209. JMLR, 2019. (pages 5, 22)
- E. J. Gumbel. The maxima of the mean largest value and of the range. The Annals of Mathematical Statistics , 25(1):76-84, 1954. (page 9)
- Benjamin Halpern. Fixed points of nonexpanding maps. Bulletin of the American Mathematical Society , 73(6):957-961, 1967. (page 5)
- Geoffrey E. Hinton and Drew van Camp. Keeping the neural networks simple by minimizing the description length of the weights. In Proceedings of the Annual Conference on Computational Learning Theory , pages 5-13. ACM Press, 1993. (pages 1, 3)
- Y. C. Ho and X. Cao. Perturbation analysis and optimization of queueing networks. Journal of Optimization Theory and Applications , 40(4):559-582, 1983. (page 4)
- Matthew Hoffman and Yian Ma. Black-box variational inference as a parametric approximation to Langevin dynamics. In Proceedings of the International Conference on Machine Learning , volume 119 of PMLR , pages 4324-4341. JMLR, 2020. (page 9)
- Alexandra Maria Hotti, Lennart Alexander Van der Goten, and Jens Lagergren. Benefits of nonlinear scale parameterizations in black box variational inference through smoothness results and gradient variance bounds. In Proceedings of the International Conference on Artificial Intelligence and Statistics , volume 238 of PMLR , pages 3538-3546. JMLR, 2024. (page 4)
- Yiheng Jiang, Sinho Chewi, and Aram-Alexandre Pooladian. Algorithms for mean-field variational inference via polyhedral optimization in the Wasserstein space. Foundations of Computational Mathematics , 2025. (page 9)
- Norman L. Johnson, Samuel Kotz, and Narayanaswamy Balakrishnan. Continuous univariate distributions. volume 1 of Wiley Series in Probability and Mathematical Statistics . Wiley, New York, 2 edition, 1994. (page 26)
- Norman L. Johnson, Samuel Kotz, and Narayanaswamy Balakrishnan. Continuous univariate distributions. volume 2 of Wiley Series in Probability and Mathematical Statistics . Wiley, New York, 2 edition, 1995. (pages 26, 28, 29)
- Michael I. Jordan, Zoubin Ghahramani, Tommi S. Jaakkola, and Lawrence K. Saul. An introduction to variational methods for graphical models. Machine Learning , 37(2):183-233, 1999. (pages 1, 2)
- Anya Katsevich and Philippe Rigollet. On the approximation accuracy of Gaussian variational inference. The Annals of Statistics , 52(4):1384-1409, 2024. (page 10)
- Ahmed Khaled, Othmane Sebbouh, Nicolas Loizou, Robert M. Gower, and Peter Richtárik. Unified analysis of stochastic gradient methods for composite convex and smooth optimization. Journal of Optimization Theory and Applications , 199:499-540, 2023. (page 22)
- Mohammad Emtiyaz Khan, Reza Babanezhad, Wu Lin, Mark Schmidt, and Masashi Sugiyama. Faster stochastic variational inference using proximal-gradient methods with general divergence functions. In Proceedings of the Conference on Uncertainty in Artificial Intelligence , UAI'16, pages 319-328, Jersey City, New Jersey, USA, 2016. AUAI Press. (page 9)
- Kyurae Kim, Jisu Oh, Kaiwen Wu, Yian Ma, and Jacob R. Gardner. On the convergence of blackbox variational inference. In Advances in Neural Information Processing Systems , volume 36, pages 44615-44657. Curran Associates Inc., 2023a. (pages 2, 3, 4, 9, 22)
- Kyurae Kim, Kaiwen Wu, Jisu Oh, and Jacob R. Gardner. Practical and matching gradient variance bounds for black-box variational Bayesian inference. In Proceedings of the International Conference on Machine Learning , volume 202 of PMLR , pages 16853-16876. JMLR, 2023b. (pages 4, 7, 9)
- Kyurae Kim, Joohwan Ko, Yi-An Ma, and Jacob R. Gardner. Demystifying SGD with doubly stochastic gradients. In Proceedings of the International Conference on Machine Learning , volume 235 of PMLR , pages 24210-24247. JMLR, 2024a. (page 10)
- Kyurae Kim, Yian Ma, and Jacob R. Gardner. Linear convergence of black-box variational inference: Should we stick the landing? In Proceedings of the International Conference on Artificial Intelligence and Statistics , volume 238 of PMLR , pages 235-243. JMLR, 2024b. (pages 5, 9, 10)

- Diederik P. Kingma and Max Welling. Auto-encoding variational Bayes. In Proceedings of the International Conference on Learning Representations , Banff, AB, Canada, 2014. (pages 2, 4)
- Joohwan Ko, Kyurae Kim, Woo Chang Kim, and Jacob R. Gardner. Provably scalable black-box variational inference with structured variational families. In Proceedings of the International Conference on Machine Learning , volume 235 of PMLR , pages 24896-24931. JMLR, 2024. (pages 1, 2, 7, 9, 10, 22)
- Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, and David M. Blei. Automatic differentiation variational inference. Journal of Machine Learning Research , 18(14):1-45, 2017. (pages 1, 2, 3, 4, 10)
- S. Kullback and R. A. Leibler. On information and sufficiency. The Annals of Mathematical Statistics , 22(1):79-86, 1951. (pages 1, 2)
- Simon Lacoste-Julien, Mark Schmidt, and Francis Bach. A simpler approach to obtaining an O(1 /t ) convergence rate for the projected stochastic subgradient method. arXiv Preprint arXiv:1212.2002, 2012. (page 5)
- Marc Lambert, Sinho Chewi, Francis Bach, Silvère Bonnabel, and Philippe Rigollet. Variational inference via Wasserstein gradient flows. In Advances in Neural Information Processing Systems , volume 35, pages 14434-14447. Curran Associates, Inc., 2022. (pages 3, 9)
- Hugo Lavenant and Giacomo Zanella. Convergence rate of random scan coordinate ascent variational inference under log-concavity. SIAM Journal on Optimization , 34(4):3750-3761, 2024. (pages 3, 10)
- Sifan Liu and Art B. Owen. Quasi-Monte Carlo quasi-Newton in variational Bayes. Journal of Machine Learning Research , 22(243):1-23, 2021. (pages 9, 10)
- Charles C. Margossian and Lawrence K. Saul. The shrinkage-delinkage trade-off: An analysis of factorized Gaussian approximations for variational inference. In Proceedings of the Conference on Uncertainty in Artificial Intelligence , volume 216 of PMLR , pages 1358-1367. JMLR, 2023. (page 10)
- Charles C. Margossian and Lawrence K. Saul. Variational inference in location-scale families: Exact recovery of the mean and correlation matrix. In Proceedings of the International Conference on Artificial Intelligence and Statistics , volume 258 of PMLR , pages 3466-3474. JMLR, 2025. (page 10)
- Si Yi Meng, Sharan Vaswani, Issam Hadj Laradji, Mark Schmidt, and Simon Lacoste-Julien. Fast and furious convergence: Stochastic second order methods under interpolation. In Proceedings of the International Conference on Artificial Intelligence and Statistics , volume 108 of PMLR , pages 1375-1386. JMLR, 2020. (page 10)
- Andrew Miller, Nick Foti, Alexander D' Amour, and Ryan P Adams. Reducing reparameterization gradient variance. In Advances in Neural Information Processing Systems , volume 30, pages 3708-3718. Curran Associates, Inc., 2017. (page 10)
- Chirag Modi, Diana Cai, and Lawrence K. Saul. Batch, match, and patch: Low-rank approximations for score-based variational inference. In Proceedings of the International Conference on Artificial Intelligence and Statistics , volume 258 of PMLR , pages 4510-4518. JMLR, 2025. (page 10)
- Shakir Mohamed, Mihaela Rosca, Michael Figurnov, and Andriy Mnih. Monte Carlo gradient estimation in machine learning. Journal of Machine Learning Research , 21(132):1-62, 2020. (page 4)
- A. Nemirovski, A. Juditsky, G. Lan, and A. Shapiro. Robust stochastic approximation approach to stochastic programming. SIAM Journal on Optimization , 19(4):1574-1609, 2009. (page 4)
- Dai Hai Nguyen, Tetsuya Sakurai, and Hiroshi Mamitsuka. Wasserstein gradient flow over variational parameter space for variational inference. In Proceedings of the International Conference on Artificial Intelligence and Statistic , volume 258 of PMLR , pages 1756-1764. JMLR, 2025. (page 9)
- Victor M.-H. Ong, David J. Nott, and Michael S. Smith. Gaussian variational approximation with a factor covariance structure. Journal of Computational and Graphical Statistics , 27(3):465-478, 2018. (pages 1, 10)
- Neal Parikh and Stephen P. Boyd. Proximal Algorithms , volume 1 of Foundations and Trends® in Optimization . Now Publishers, Norwell, MA, 2014. (page 4)

- Anand Patil, David Huard, and Christopher Fonnesbeck. PyMC: Bayesian stochastic modelling in Python. Journal of Statistical Software , 35(4):1-81, 2010. (pages 1, 3)
- Carsten Peterson and Eric Hartman. Explorations of the mean field theory learning algorithm. Neural Networks , 2(6):475-494, 1989. (pages 1, 3)
- Georg Pflug. Optimization of Stochastic Models: The Interface between Simulation and Optimization . Number v.373 in The Springer International Series in Engineering and Computer Science Ser. Springer, New York, NY, 1996. (page 4)
- Rana. Answer to 'A bound for the expectation of the maximum independent random variables', 2017. URL https://math.stackexchange.com/q/2177201 . (page 27)
- Rajesh Ranganath, Sean Gerrish, and David Blei. Black box variational inference. In Proceedings of the International Conference on Artificial Intelligence and Statistics , volume 33 of PMLR , pages 814-822. JMLR, 2014. (page 1)
- Jeffrey Regier, Michael I Jordan, and Jon McAuliffe. Fast black-box variational inference through stochastic trust-region optimization. In Advances in Neural Information Processing Systems , volume 30, pages 2399-2408. Curran Associates, Inc., 2017. (pages 9, 10)
- Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In Proceedings of the International Conference on Machine Learning , volume 32 of PMLR , pages 1278-1286. JMLR, 2014. (pages 1, 2, 4, 10)
- Herbert Robbins and Sutton Monro. A stochastic approximation method. The Annals of Mathematical Statistics , 22(3):400-407, 1951. (pages 1, 4)
- Geoffrey Roeder, Yuhuai Wu, and David K Duvenaud. Sticking the landing: Simple, lower-variance gradient estimators for variational inference. In Advances in Neural Information Processing Systems , volume 30, pages 6928-6937. Curran Associates, Inc., 2017. (page 10)
- Reuven Y. Rubinstein. Sensitivity analysis of discrete event systems by the 'push out' method. Annals of Operations Research , 39(1):229-250, 1992. (page 4)
- Mark Schmidt and Nicolas Le Roux. Fast convergence of stochastic gradient descent under a strong growth condition. arXiv Preprint arXiv:1308.6370, 2013. (page 5)
- Shai Shalev-Shwartz, Yoram Singer, Nathan Srebro, and Andrew Cotter. Pegasos: Primal estimated sub-gradient solver for SVM. Mathematical Programming , 127(1):3-30, 2011. (page 4)
- Sebastian U. Stich. Unified optimal analysis of the (stochastic) gradient method. Preprint arXiv:1907.04232, arXiv, 2019. URL http://arxiv.org/abs/1907.04232 . (pages 5, 19)
- Sobihan Surendran, Antoine Godichon-Baggioni, and Sylvain Le Corff. Theoretical convergence guarantees for variational autoencoders. In Proceedings of the International Conference on Artificial Intelligence and Statistics , volume 258 of PMLR . JMLR, 2025. (page 9)
- Symbol-1. Answer to 'Meaningful lower-bound of √ a 2 + b -a when a glyph[greatermuch] b &gt; 0 ', 2022. URL https://math.stackexchange.com/q/4360503 . (page 21)
- Michalis Titsias and Miguel Lázaro-Gredilla. Doubly stochastic variational Bayes for non-conjugate inference. In Proceedings of the International Conference on Machine Learning , volume 32 of PMLR , pages 1971-1979. JMLR, 2014. (pages 1, 2, 3, 4, 9, 10)
- Marcin Tomczak, Siddharth Swaroop, and Richard Turner. Efficient low rank Gaussian variational inference for neural networks. In Advances in Neural Information Processing Systems , volume 33, pages 4610-4622. Curran Associates, Inc., 2020. (pages 1, 10)
- Lloyd N. Trefethen and David Bau. Numerical Linear Algebra . Society for Industrial and Applied Mathematics, Philadelphia, 1997. (page 36)
- Sharan Vaswani, Francis Bach, and Mark Schmidt. Fast and faster convergence of SGD for overparameterized models and an accelerated perceptron. In Proceedings of the International Conference on Artificial Intelligence and Statistics , volume 89 of PMLR , pages 1195-1204. JMLR, 2019. (page 5)
- Martin J. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint . Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, New York, NY, 1st ed edition, 2019. (pages 6, 9)

- Xi Wang, Tomas Geffner, and Justin Domke. Joint control variate for faster black-box variational inference. In Proceedings of the International Conference on Artificial Intelligence and Statistics , volume 238 of PMLR , pages 1639-1647. JMLR, 2024. (page 10)
- Yixin Wang and David Blei. Variational bayes under model misspecification. In Advances in Neural Information Processing Systems , volume 32, pages 13357-13367. Curran Associates, Inc., 2019a. (page 10)
- Yixin Wang and David M. Blei. Frequentist consistency of variational Bayes. Journal of the American Statistical Association , 114(527):1147-1161, 2019b. (page 10)
- Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning , 8(3):229-256, 1992. (page 4)
- David Wingate and Theophane Weber. Automated variational inference in probabilistic programming. arXiv Preprint arXiv:1301.1299, 2013. (page 1)
- Ming Xu, Matias Quiroz, Robert Kohn, and Scott A. Sisson. Variance reduction properties of the reparameterization trick. In Proceedings of the International Conference on Artificial Intelligence and Statistics , volume 89 of PMLR , pages 2711-2720. JMLR, 2019. (page 4)
- Zuheng Xu and Trevor Campbell. The computational asymptotics of Gaussian variational inference and the Laplace approximation. Statistics and Computing , 32(4), 2022. (page 9)
- Yun Yang, Debdeep Pati, and Anirban Bhattacharya. α -variational inference with statistical guarantees. The Annals of Statistics , 48(2):886-905, 2020. (page 10)
- Anderson Y. Zhang and Harrison H. Zhou. Theoretical and computational guarantees of mean field variational inference for community detection. The Annals of Statistics , 48(5):2575-2598, 2020. (page 9)
- Fengshuo Zhang and Chao Gao. Convergence rates of variational posterior distributions. The Annals of Statistics , 48(4):2180-2207, 2020. (page 10)
- Lu Zhang, Bob Carpenter, Andrew Gelman, and Aki Vehtari. Pathfinder: Parallel quasi-Newton variational inference. Journal of Machine Learning Research , 23(306):1-49, 2022. (page 1)

## Contents

| 1 Introduction   | 1 Introduction                | 1 Introduction                                               | 1   |
|------------------|-------------------------------|--------------------------------------------------------------|-----|
| 2                | Preliminaries                 | Preliminaries                                                | 2   |
|                  | 2.1                           | Problem Setup . . . . . . . . . . . . . . . . . . . . . . .  | 2   |
|                  | 2.2                           | Variational Family . . . . . . . . . . . . . . . . . . . . . | 3   |
|                  | 2.3                           | Algorithm Setup . . . . . . . . . . . . . . . . . . . . . .  | 3   |
|                  | 2.4                           | General Analysis of Stochastic Proximal Gradient Descent     | 4   |
| 3                | Main Results                  | Main Results                                                 | 5   |
|                  | 3.1                           | General Result . . . . . . . . . . . . . . . . . . . . . .   | 5   |
|                  | 3.2                           | Special Cases with Benign Dimension Dependence . . .         | 6   |
| 4                | Analysis of Gradient Variance | Analysis of Gradient Variance                                | 7   |
|                  | 4.1                           | Overview . . . . . . . . . . . . . . . . . . . . . . . . .   | 7   |
|                  | 4.2                           | Upper Bound on Gradient Variance . . . . . . . . . . . .     | 8   |
|                  | 4.3                           | Unimprovability . . . . . . . . . . . . . . . . . . . . . .  | 9   |
| 5                | Discussion                    | Discussion                                                   | 9   |
|                  | 5.1 Related                   | Works . . . . . . . . . . . . . . . . . . . . . . .          | 9   |
|                  | 5.2                           |                                                              | 10  |
|                  |                               | Conclusion . . . . . . . . . . . . . . . . . . . . . . . .   |     |
| A                | Auxiliary Lemmas              | Auxiliary Lemmas                                             | 18  |
| B                | Proofs                        | Proofs                                                       | 19  |
|                  | B.1                           | Proofs of Results in Section 2 . . . . . . . . . . . . . . . | 19  |
|                  |                               | B.1.1 Proof of Proposition 2.9 . . . . . . . . . . . . .     | 19  |
|                  | B.1.2                         | Proof of Lemma B.1 . . . . . . . . . . . . . . .             | 22  |
|                  | B.2                           | Proofs of Results in Section 3 . . . . . . . . . . . . . . . | 25  |
|                  | B.2.1                         | Proof of Theorem 3.2 . . . . . . . . . . . . . . .           | 25  |
|                  | B.2.2                         | Proof of Proposition 3.4 . . . . . . . . . . . . .           | 26  |
|                  | B.2.3                         | Proof of Proposition 3.5 . . . . . . . . . . . . .           | 27  |
|                  | B.3                           | of Results in Section 4                                      |     |
|                  | Proofs                        | . . . . . . . . . . . . . . .                                | 30  |
|                  | B.3.1                         | Proof of Lemma 4.1 . . . . . . . . . . . . . . .             | 30  |
|                  | B.3.2                         | Proof of Lemma B.4 . . . . . . . . . . . . . . .             | 32  |
|                  | B.3.3                         | Proof of Lemma B.5 . . . . . . . . . . . . . . .             | 33  |
|                  | B.3.4                         | Proof of Proposition 4.2 . . . . . . . . . . . . .           | 35  |

## A Auxiliary Lemmas

Lemma A.1. Suppose Assumption 2.2 holds. Then r 4 = E u 4 i ≥ 1 .

<!-- formula-not-decoded -->

Lemma A.2. Suppose Assumption 2.2 holds and denote U = diag( u 1 , . . . , u d ) . Then we have the following identities: (i) E uu ⊤ = I d , (ii) E U 2 = I d .

Proof. From Assumption 2.2, we know that E u 2 i = 1 . Then (i) follows from glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

For (ii), we only need to focus on the diagonal since the off-diagonal is already zero.

<!-- formula-not-decoded -->

Lemma A.3. Suppose Assumption 2.2 holds, T λ is the reparametrization function for a locationscale family, and the linear parametrization is used. Then

<!-- formula-not-decoded -->

Proof. ′ ′ ′

<!-- formula-not-decoded -->

Lastly,

<!-- formula-not-decoded -->

Combining this with Eq. (10) yields the result.

Lemma A.4. Suppose f is µ -strongly convex and Assumption 2.7 holds. Then f is L -Lipschitz smooth, while the constants satisfy the ordering

<!-- formula-not-decoded -->

Proof. For all λ, λ ′ ∈ Λ , the unbiasedness of ̂ ∇ f and Jensen's inequality states that ‖∇ f ( λ ) -∇ f ( λ ′ ) ‖ 2 2 = ‖ E ̂ ∇ f ( λ ; u ) -E ̂ ∇ f ( λ ′ ; u ) ‖ 2 2 ≤ E ‖ ̂ ∇ f ( λ ; u ) -̂ ∇ f ( λ ′ ; u ) ‖ 2 2 . Then the µ -strong convexity of f and Assumption 2.7 yields the inequality

µ 2 ‖ λ -λ ′ ‖ 2 2 ≤ ‖∇ f ( λ ) -∇ f ( λ ′ ) ‖ 2 2 ≤ E ‖ ̂ ∇ f ( λ ; u ) -̂ ∇ f ( λ ′ ; u ) ‖ 2 2 ≤ L 2 ‖ λ -λ ′ ‖ 2 2 , from which the statement follows immediately.

## B Proofs

## B.1 Proofs of Results in Section 2

## B.1.1 Proof of Proposition 2.9

Under the stated assumptions, we first establish a convergence bound which bounds E ‖ λ T -λ ∗ ‖ 2 2 after T iterations under a given step size schedule. We will invert this convergence bound into a complexity guarantee by identifying the conditions on T , t ∗ , τ , and γ 0 that guarantee E ‖ λ T -λ ∗ ‖ 2 2 ≤ ϵ for a given ϵ &gt; 0 .

Lemma B.1. Suppose f is µ -strongly convex, h satisfies Assumption 2.5, and ̂ ∇ f satisfies Assumptions 2.7 and 2.8. Then, for the global optimum λ ∗ = arg min λ ∈ Λ F ( λ ) , and the step size schedule in Eq. (5) with any t ∗ ≥ 0 and τ ≥ 4 L 2 /µ 2 , the contraction coefficient ρ ≜ 1 -µγ 0 satisfies ρ ∈ (0 , 1) and the last iterate of SPGD after T iterations, λ T , satisfies

<!-- formula-not-decoded -->

Proof . The full proof is deferred to Appendix B.1.2, p. 22.

This is a slight generalization of the result by Domke et al. (2023, Theorem 7), where the switching time t ∗ was fixed to some t ∗ ∝ L 2 /µ 2 . While the choice of t ∗ ∝ L 2 /µ 2 results in the typical O(1 /ϵ ) asymptotic complexity, it suffers from a suboptimal polynomial dependence on the initialization error ∆ = ‖ λ 0 -λ ∗ ‖ 2 . Picking an alternative t ∗ , which is what we do in the proof, improves the iteration complexity to O ( 1 /ϵ +1 / √ ϵ log ∆ 2 +log(∆ 2 /ϵ ) ) . Now, a similar O(1 /ϵ ) result for vanilla SGD was demonstrated by Stich (2019), where the dependence on ∆ is also logarithmic. Their step size schedule, however, requires the maximum number of iterations T , which means T must be fixed before running the algorithm. Our step size, on the other hand, does not require T and is therefore an any-time result.

Proposition 2.9. Suppose f is µ -strongly convex, h satisfies Assumption 2.5, and ̂ ∇ f satisfies Assumptions 2.7 and 2.8. Then, for the global optimum λ ∗ = arg min λ ∈ Λ F ( λ ) , ∆ ≜ ‖ λ 0 -λ ∗ ‖ 2 , some t ∗ , τ , and γ 0 (explicit in the proof), SPGD with the step size schedule in Eq. (5) guarantees

<!-- formula-not-decoded -->

Proof. Since f is strongly convex and h is convex, F is also strongly convex. This implies that, by the property of strictly convex functions, F has a unique global optimum, which we denote as λ ∗ . Then we can invoke Lemma B.1. We will optimize the upper bound over the parameters t ∗ , τ , γ 0 , and T so that we can ensure the ϵ -accuracy guarantee E ‖ λ T -λ ∗ ‖ 2 2 ≤ ϵ .

Consider the choice

<!-- formula-not-decoded -->

We will separately analyze the cases of t ∗ ≥ T and t ∗ &lt; T . When t ∗ ≥ T , the second-stage never kicks in. Therefore, we can apply Eq. (27) with t ∗ = T . Furthermore,

<!-- formula-not-decoded -->

is true. An immediate implication is that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Considering this fact, applying Eq. (27) with t ∗ = T ,

<!-- formula-not-decoded -->

The number of required steps for achieving the ϵ -accuracy requirement follows from

<!-- formula-not-decoded -->

For the case t ∗ &lt; T ,

<!-- formula-not-decoded -->

This implies

Substituting for this in Lemma B.1,

<!-- formula-not-decoded -->

which is a quadratic function of 1 /T with the coefficients

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Achieving the ϵ -accuracy guarantee is equivalent to finding the largest x = 1 /T satisfying the inequalities x &gt; 0 and

<!-- formula-not-decoded -->

By the quadratic formula, this is equivalent to finding the largest x satisfying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is sufficient to obtain an ϵ -accurate solution. To make the bound more interpretable, after defining α = 4 aϵ and β = b , we can use the inequality (Symbol-1, 2022)

Then

<!-- formula-not-decoded -->

where we used the inequality √ a + b ≤ √ a + √ b . Thus, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substituting t ∗ and γ 0 with the expressions in Eq. (11),

<!-- formula-not-decoded -->

Considering this, the sufficient condition for E ‖ λ T -λ ∗ ‖ 2 2 ≤ ϵ is now

<!-- formula-not-decoded -->

Combining both cases, that is, Eqs. (13) and (17), we have

<!-- formula-not-decoded -->

This implies the stated result.

Therefore, picking any

## B.1.2 Proof of Lemma B.1

The proof closely mirrors the strategy of Garrigos and Gower (2023, Theorem 12.9), which is a combination of previous analyses of SPGD (Gorbunov et al., 2020; Khaled et al., 2023) with the analysis of SGD strongly convex objectives with a decreasing step size schedule (Gower et al., 2019). The main difference is that Garrigos and Gower utilize a different condition on the gradient variance instead of Assumption 2.7. Specificially, they assume that, for all λ, λ ′ ∈ Λ , there exists some function of L ( u ) : supp( u ) → [0 , ∞ ) such that, for each u ∈ supp( u ) , the function ̂ ∇ f ( λ ; u ) : Λ → R p is L ( u ) -smooth with respect to λ . This then enables the use of the 'convex expected smoothness' (Gorbunov et al., 2020; Khaled et al., 2023) condition, which postulates that, for all λ ∈ Λ , there exists some L &lt; ∞ such that where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is the Bregman divergence associated with f . Note that Assumption 2.7 and the µ -strong convexity of f implies Eq. (18). Therefore, under our assumptions, one can invoke the results that assume Eq. (18), which was the strategy by some previous analyses of BBVI (Kim et al., 2023a; Ko et al., 2024). Here, we will take a more straightforward approach that uses Assumption 2.7 directly in the convergence proof, but the results are identical to the indirect approach of establishing Eq. (18).

Lemma B.1. Suppose f is µ -strongly convex, h satisfies Assumption 2.5, and ̂ ∇ f satisfies Assumptions 2.7 and 2.8. Then, for the global optimum λ ∗ = arg min λ ∈ Λ F ( λ ) , and the step size schedule in Eq. (5) with any t ∗ ≥ 0 and τ ≥ 4 L 2 /µ 2 , the contraction coefficient ρ ≜ 1 -µγ 0 satisfies ρ ∈ (0 , 1) and the last iterate of SPGD after T iterations, λ T , satisfies

<!-- formula-not-decoded -->

Proof. Since f is strongly convex and h is convex, F is also strongly convex. This implies that F has a unique global optimum, which we denote as λ ∗ . Furthermore, under the stated assumptions on h , the proximal operator prox γh ( · ) is non-expansive for any γ ∈ (0 , ∞ ) (Garrigos and Gower, 2023, Lemma 8.17) and any λ, λ ′ ∈ R p such that

<!-- formula-not-decoded -->

and λ ∗ is the fixed-point of the deterministic proximal gradient descent step (Garrigos and Gower, 2023, Lemma 8.18) such that

<!-- formula-not-decoded -->

Using these facts,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

‖ λ t +1 -λ ∗ ‖ 2 2 ≤ ‖ λ t -λ ∗ ‖ 2 2 -2 γ t 〈 ̂ ∇ f ( λ t ; u ) -∇ f ( λ ∗ ) , λ t -λ ∗ 〉 + γ 2 t ‖ ̂ ∇ f ( λ t ; u ) -∇ f ( λ ∗ ) ‖ 2 2 . Denoting the filtration of the σ -field of the iterates generated up to iteration t as F t , where the equality follows from the fact that ̂ ∇ f is unbiased conditional on any λ t ∈ Λ . From the µ -strong convexity of f ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The gradient variance at λ t , on the other hand, can be compared against the gradient variance at λ ∗ through the variance transfer strategy as

<!-- formula-not-decoded -->

Applying Eqs. (23) and (24) to Eq. (22),

<!-- formula-not-decoded -->

Taking expectation over all randomness, we obtain our general partial contraction bound

<!-- formula-not-decoded -->

Due to the form of the step size schedule, SPGD operates in two different regimes: the first stage with a fixed step size γ t = γ 0 ( t ∈ { 0 , . . . , t ∗ -1 } ) and the second stage with a decreasing step size γ t +1 &lt; γ t ( t ∈ { t ∗ , . . . , T } ). In the first stage, γ t = γ 0 ≤ µ 2 L 2 . Then the Bregman divergence term in Eq. (25) is negative such that

<!-- formula-not-decoded -->

Unrolling the recursion yields

<!-- formula-not-decoded -->

From Lemma A.4, we deduce that γ 0 µ = µ 2 / (2 L 2 ) ≤ 1 / 2 , which implies ρ ∈ (0 , 1) .

We now turn to the second stage, where the step size starts decreasing. Notice that, for any τ ≥ 4 L 2 /µ 2 , Eq. (5) satisfies

<!-- formula-not-decoded -->

Therefore, γ t ≤ µ 2 L 2 for all t ≥ 0 . Again, the Bregman term in Eq. (25) is negative such that

<!-- formula-not-decoded -->

Subtituting γ t with the choice in Eq. (5), we obtain

<!-- formula-not-decoded -->

Multiplying ( t + τ +1) 2 to both sides,

<!-- formula-not-decoded -->

Let us choose the Lyapunov function V t ≜ ( t + τ +1) 2 E ‖ λ t +1 -λ ∗ ‖ 2 2 . Then the discrete derivative of the Lyapunov,

<!-- formula-not-decoded -->

shows that the energy is increasing only by a constant. By integrating the Lyapunov over the time interval t = t ∗ , . . . , T -1 ,

<!-- formula-not-decoded -->

Substuting ‖ λ t ∗ -λ ∗ ‖ 2 2 with the error in Eq. (27),

<!-- formula-not-decoded -->

which is our stated result.

## B.2 Proofs of Results in Section 3

## B.2.1 Proof of Theorem 3.2

Theorem 3.2. Suppose the following hold:

1. ℓ is µ -strongly convex and satisfies Assumption 3.1 and µ ≤ σ min ( H ) ≤ σ max ( H ) ≤ L .
2. h satisfies Assumption 2.5.
3. Q is a mean-field location-scale family, where Assumption 2.2 holds.

<!-- formula-not-decoded -->

4. ̂ ∇ f is the reparametrization gradient. Denote the global optimum λ ∗ = ( m ∗ , C ∗ ) = arg min λ ∈ Λ F ( λ ) , the irreducible gradient noise as σ 2 ∗ ≜ ‖ m ∗ -¯ z ‖ 2 2 + ‖ C ∗ ‖ 2 F , and the stationary point of ℓ as ¯ z ≜ arg min z ∈ R d ℓ ( z ) . Then, for some t ∗ , τ , and γ 0 (explicit in the proof), SPGD with the step size schedule in Eq. (5) guarantees

Proof. The proof consists of establishing the sufficient conditions of Proposition 2.9 as follows:

- (i) ℓ is µ -strongly convex ⇒ f is µ -strongly convex.
- (ii) Assumption 3.1 ⇒ Assumptions 2.7 and 2.8.

Under the linear parametrization, (i) was established by Domke (2020, Thm. 9). It remains to establish (ii). Therefore, the proof focuses on analyzing the variance of the gradient estimator ̂ ∇ f . Since Assumption 3.1 holds, Lemma 4.1 states that, for all λ, λ ′ ∈ R d × D d , the inequality holds. Since Λ ⊂ R d × D d under the linear parametrization, this implies we satisfy Assumption 2.7 with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, For the specific choice of λ ∗ = ( m ∗ , C ∗ ) = arg min λ ∈ Λ F ( λ ) and ¯ λ = (¯ z, 0 d × d ) (which is not part of Λ ), we have the equality

E ‖ ̂ ∇ f ( λ ∗ ; u ) -̂ ∇ f ( ¯ λ ; u ) ‖ 2 2 = E ‖ ̂ ∇ f ( λ ∗ ; u ) -̂ ∇ f (¯ z ; u ) ‖ 2 2 = E ‖ ̂ ∇ f ( λ ∗ ; u ) ‖ 2 2 . This means Lemma 4.1 also implies Assumption 2.8 with the constant

We are now able to invoke Proposition 2.9. Substituting L and σ 2 in Eq. (28) with the expressions above, we obtain the condition

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, substituting for Eq. (29) yields our stated result.

## B.2.2 Proof of Proposition 3.4

The result follows from a well-known bound on the expected maximum of sub-exponential random variables. We state the proof for completeness.

Lemma B.2. Let x 1 , . . . , x d be i.i.d. random variables. Suppose there exists some t &gt; 0 such that their moment-generating function (MGF) satisfies M x i ( t ) &lt; ∞ . Then

Proof.

<!-- formula-not-decoded -->

Dividing both sides by t yields the statement.

Applying Lemma B.2 to u 2 i yields the result.

Proposition 3.4. Suppose there exists some t &gt; 0 such that the MGF of u 2 i satisfies M u 2 i ( t ) &lt; ∞ . Then

For example, if φ is a standard Gaussian, then

<!-- formula-not-decoded -->

Proof. The first part of the statement is a re-statement of Lemma B.2.

<!-- formula-not-decoded -->

For the special case of u i ∼ N (0 , 1) , we know that u 2 i ∼ χ 2 1 (Johnson et al., 1995, Eq. 29.1), which is the χ 2 distribution with 1 degree of freedom. The MGF of χ 2 1 is given as

<!-- formula-not-decoded -->

for t ∈ (0 , 1 / 2) . Then we can invoke Lemma B.2, which suggests

Any fixed choice of t ∈ (0 , 1 / 2) is a valid upper bound. Picking t = 1 2 ( 1 -1 e ) ≥ 1 4 yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, the kurtosis of the standard Gaussian is r 4 = 3 (Johnson et al., 1994, Eq. 13.11). Plugging r 4 and Eq. (31) into g in Theorem 3.2 yields the statement.

<!-- formula-not-decoded -->

## B.2.3 Proof of Proposition 3.5

The result follows from the following moment-based bound on the expected maximum of random variables, which is a non-asymptotic refinement of the proof by Rana (2017).

Lemma B.3. Let x 1 , . . . , x d be i.i.d. non-negative random variables where, for k ≥ 2 , their k th moment is finite. That is, E x k i = r k &lt; ∞ . Then

<!-- formula-not-decoded -->

Proof. For any ϵ &gt; 0 , we have

<!-- formula-not-decoded -->

(Decreased lower limit of integral) .

<!-- formula-not-decoded -->

Now, from the definition of moments, we know that

Therefore,

<!-- formula-not-decoded -->

The bound is minimized when setting

Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If the k th moment of u 2 i is finite, this then immediately implies a polynomial O( d 1 /k ) bound on g . Proposition 3.5. Suppose, for k ≥ 2 , the k th moment of u 2 i is finite as r 2 k = E u 2 k i &lt; ∞ . Then E max u 2 i √ 2 d 1 /k r 1 /k 2 k .

<!-- formula-not-decoded -->

For example, if φ is a Studentt with ν &gt; 4 degrees of freedom and unit variance, then g ( d, H, δ, µ, φ ) ≤ 8( ‖ H ‖ 2 2 /µ 2 ) + ( δ 2 /µ 2 ) ( 16 + √ 2 ν 3 d 2 ν -2 ) .

Proof. The first part of the statement directly follows from Lemma B.3, where we simplified ( k / k -1 ) ( k -1) /k . In particular, for k ≥ 2 , ( k / k -1 ) ( k -1) /k is monotonically decreasing. Since an order k ≥ 2 moment exists by the assumption on the degrees of freedom, ( k / k -1 ) ( k -1) /k ≤ √ 2 .

Let's turn to the second part of the statement. We will denote a Studentt distribution with ν -degrees of freedom as t ν . Since t ν does not have unit variance (Johnson et al., 1995, Eq. 28.7a), we have to set the sampling process from φ to be

<!-- formula-not-decoded -->

Now, it is known that v 2 i d = w i ∼ FDist(1 , ν 2 ) (Johnson et al., 1995, §28.7), where FDist( ν 1 , ν 2 ) is Fisher's F -distribution with ( ν 1 , ν 2 ) degrees of freedom. The k th raw moment of FDist( ν 1 , ν 2 ) , denoted as m k ≜ E w k i , exists up to 2 k &lt; ν 2 = ν and is given as

This means that we can invoke Lemma B.3 as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For m 1 /k k , we can use the fact that the gamma function satisfies the recursion Γ( z +1) = z Γ( z ) , which implies Γ( a/ 2 + k ) = Γ( a/ 2) ∏ k -1 i =0 ( a/ 2 + i ) for any a &gt; 0 . Therefore,

(geometric series sum formula)

<!-- formula-not-decoded -->

Applying this bound to a = ν 2 = ν and a = ν 1 = 1 respectively,

<!-- formula-not-decoded -->

Also, choosing k = glyph[ceilingleft] ν/ 2 -1 glyph[ceilingright] , we have d 1 /k ≤ d 2 / ( ν -2) . This yields

<!-- formula-not-decoded -->

Lastly, the kurtosis of u i = ( ν -2) /ν v i follows as (Johnson et al., 1995, Eq. 28.5)

<!-- formula-not-decoded -->

Plugging the bound in Eq. (32) and the value of r 4 into g in Theorem 3.2 yields the statement.

## B.3 Proofs of Results in Section 4

## B.3.1 Proof of Lemma 4.1

Under the assumption that ∇ 2 ℓ glyph[precedesequal] L I d and twice differentiability, it is well known that ∇ 2 ℓ glyph[precedesequal] L I d ⇒ ℓ is L -smooth. We will prove a supporting result analogous to this under Assumption 3.1, which will allow us to bound the relative growth of ∇ ℓ .

<!-- formula-not-decoded -->

Lemma B.4. Suppose ℓ : R d → R satisfies Assumption 3.1. Then, for any W ∈ R d × d satisfying ‖ W ‖ 2 &lt; ∞ ,

Proof. The full proof is deferred to Appendix B.3.2, p. 32.

Using this, we can now simplify the ∇ ℓ terms in Eq. (6). Applying Lemma B.4 to V loc with W = I d and Young's inequality,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

V const corresponds to the constant component of the Hessian ∇ 2 ℓ , whereas V non-const corresponds to the non-constant residual. Denote the location and scale parameters of λ and λ ′ as

<!-- formula-not-decoded -->

For V const, we can use the following lemma:

Lemma B.5. Suppose T λ is the reparameterization operator of a mean-field location-family and Assumption 2.2 holds. Then, for any matrix H ∈ R d × d and any λ, λ ′ ∈ R d × D d ,

<!-- formula-not-decoded -->

See the full proof in Appendix B.3.3, p. 33.

The remaining part of the proof closely resembles the proof sketch of Lemma 4.1. For convenience, we first restate Lemma 4.1 and then proceed to the full proof.

Lemma 4.1. Suppose Assumptions 2.2 and 3.1 hold, Q is a mean-field location-family, and ̂ ∇ f is the reparametrization gradient. Then, for any λ, λ ′ ∈ R d × D d .

Proof. Recall Eq. (34). The proof consists of bounding the two terms V const and V non-const . First, for V const, under Assumption 3.1,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It remains to bound V non-const, which is our main challenge.

Denote ¯ m ≜ m -m ′ and ¯ C ≜ C -C ′ such that

<!-- formula-not-decoded -->

Then

<!-- formula-not-decoded -->

glyph[negationslash]

We will focus on the first term. Denoting i ∗ = arg max i =1 ,...,d u 2 i , the coordinate of maximum magnitude, we can decompose the expectation by the contribution of the event i ∗ = i and i ∗ = i . That is, glyph[negationslash]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The expectation of the event i ∗ = i follows as

On the other hand, for the event i ∗ = i , glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

Therefore, we finally obtain

<!-- formula-not-decoded -->

Combining Eqs. (6), (33) to (35) and (37) yields the statement.

<!-- formula-not-decoded -->

̸

̸

̸

glyph[negationslash]

glyph[negationslash]

## B.3.2 Proof of Lemma B.4

Lemma B.4. Suppose ℓ : R d → R satisfies Assumption 3.1. Then, for any W ∈ R d × d satisfying ‖ W ‖ 2 &lt; ∞ ,

<!-- formula-not-decoded -->

Proof. From twice differentiability of ℓ (Assumption 3.1) and the fundamental theorem of calculus, we know that

Denoting z t ≜ tz +(1 -t ) z ′ for clarity,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B.3.3 Proof of Lemma B.5

Lemma B.5. Suppose T λ is the reparameterization operator of a mean-field location-family and Assumption 2.2 holds. Then, for any matrix H ∈ R d × d and any λ, λ ′ ∈ R d × D d ,

<!-- formula-not-decoded -->

Proof. For clarity, let us denote ¯ C ≜ C -C ′ and ¯ m ≜ m -m ′ such that

Then

V loc and V cross are straightforward. Under Assumption 2.2, it immediately follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

On the other hand,

The expectation follows as

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Thus, the cross term V cross vanishes.

V scale requires careful elementwise inspection in order to apply Assumption 2.2. That is, V scale = E ‖ U H ¯ C u ‖ 2 2

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

Combining everything,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From the property of the Frobenius norm, for any matrix A ∈ R d × d , we can decompose

̸

<!-- formula-not-decoded -->

where off( A ) is a function that zeroes-out the diagonal of A . Then from Eq. (38),

<!-- formula-not-decoded -->

which is the stated result.

̸

̸

̸

̸

## B.3.4 Proof of Proposition 4.2

For any µ, L ∈ (0 , ∞ ) such that µ ≤ L , our goal is to obtain a matrix-valued function H worst : R d → S d ≻ 0 satisfying

<!-- formula-not-decoded -->

that, under the choice H = H worst , maximizes the quantity

<!-- formula-not-decoded -->

∥ ∥ where ¯ z ∈ { z | ∇ ℓ ( z ) = 0 } is any stationary point of ℓ , z ≜ T λ ( u ) , and z w ≜ w z + (1 -w )¯ z . Given the norm constraint, the worst-case example that maximizes Eq. (39) will be the matrix-valued function that approximately results in for any realization of u on R d . For this, we will establish the relations

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The first equality in Eq. (40) follows from identifying the conditions where H ( z w ) is independent of the value of w . For the specific choice of

<!-- formula-not-decoded -->

H ( z w ) is independent of w if it only depends on the quantities

<!-- formula-not-decoded -->

That is, with some abuse of notation, H ( z w ) = H (ˆ z w , i ∗ ) .

Lemma B.6. Suppose m = ¯ z = 0 d , and for any δ &gt; 0 , C = diag( δ, . . . , δ ) . If H ( z w ) is a function of only i ∗ and ˆ z w , then H ( z w ) is constant with respect to w ∈ [0 , 1] .

Proof. It suffices to show that, under the stated conditions, the values of i ∗ and ˆ z w are invariant to w . For ˆ z w , this trivially follows from the assumption that ¯ z = 0 as

<!-- formula-not-decoded -->

For i ∗ , we use the fact that the diagonal matrix C is isotropic as

<!-- formula-not-decoded -->

From H ( z w ) = H (ˆ z w , i ∗ ) , the integral in Eq. (39) can be solved as

It remains to construct H in a way that depends only on ˆ z w and i ∗ such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Recalling the spectral constraints, this is equivalent to, for all z ∈ R d , H solving the equation

<!-- formula-not-decoded -->

Notice the equivalence

<!-- formula-not-decoded -->

Thus, ˆ z w and i ∗ contain all the information we need. The following matrix-valued function almost solves Eq. (42):

<!-- formula-not-decoded -->

This function is reminiscent of a Householder reflector (Trefethen and Bau, 1997, Eq. 10.4), where some modifications were made to make it satisfy the eigenvalue constraint. From the fact that both e i ∗ and ˆ z have a unit norm, it is apparent that this matrix satisfies Assumption 3.1 with H = α I d and δ = β . Furthermore, by setting the constants as

<!-- formula-not-decoded -->

the triangle inequality asserts that the eigenvalue constraint µ I d ≤ H worst ≤ L I d is satisfied almost surely.

Given the specific form of H worst , we are now ready to formally prove Proposition 4.2. Let us first restate the proposition for convenience and then proceed to the proof.

Proposition 4.2. Suppose Assumption 2.2 holds and Q is a mean-field location-scale family. Then, for any t &gt; 0 , d &gt; 0 , µ, L ∈ (0 , + ∞ ) satisfying µ ≤ L , there exists a matrix-valued function H ( z ) : R d → S d ≻ 0 satisfying µ I d glyph[precedesequal] H glyph[precedesequal] L I d almost surely and a set of parameters λ = ( m,C ) ∈ R d × D d ≻ 0 such that

<!-- formula-not-decoded -->

Proof. Recall H worst in Eq. (43). By inspection, we know that H worst ( z w ) only depends on the quantities i ∗ and z w . Then Lemma B.6 states that w ↦→ H worst ( z w ) is a constant function. Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, the first term β 2 / 4 u 2 i ∗ ‖ z ‖ 2 2 is the worst-case behavior we expect from solving Eq. (42). The remaining terms V 1 and V 2 are the error caused by inexactly solving Eq. (42). It suffices to show that β 2 / 4 u 2 i ∗ ‖ z ‖ 2 2 dominates lower bounds on V 1 and V 2 asymptotically in L and d .

V 1 ≥ 0 trivially holds and can immediately be lower-bounded. V 2 , on the other hand, is not necessarily non-negative. Therefore, we will use the bound V 2 ≥ -| E V 2 | .

For V 3 , we can use an argument similar to Eq. (36) where we distribute the influence of the maximum coordinate over the d coordinates.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The last equality follows by applying Lemma A.2 to the identity E ‖ z ‖ 2 2 = E u ⊤ C ⊤ C u . By applying Eq. (48) into Eq. (47), we can now notice that V 2 decreases by a factor of E u 4 i ∗ /d .

It is clear that V 2 vanishes as d →∞ .

Applying the lower bound on V 2 into Eqs. (45) and (46), we have

<!-- formula-not-decoded -->

It remains to solve the expectation.

<!-- formula-not-decoded -->

glyph[negationslash]

Let us decompose the events where the i th coordinate attains the maximum ( i ∗ = i ) or not ( i ∗ = i ) as done in Lemma 4.1.

̸

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

̸

̸

By introducing a free variable t &gt; 0 , we can break up the indicator

̸

̸

<!-- formula-not-decoded -->

̸

̸

This then allows the expectation to break up between terms depending on u 2 i and max j = i u j , which is the independence that we were after. That is, applying Eq. (51) to Eq. (50),

̸

̸

Notice that the function ( t, φ ) ↦→ ∫ t 0 P [ u 2 i &gt; s ] d s is strictly positive as long as t &gt; 0 and only dependent on t and the base distribution φ .

<!-- formula-not-decoded -->

̸

We are left with the expectation over the event i ∗ = i . For the upper bound in Lemma 4.1, the expectation was solved by noticing that u 2 i and u 2 i ∗ can be made independent after upper bounding the indicator. For a lower bound, however, breaking up the expectation for u 2 i and u 2 i ∗ is more involved.

̸

̸

̸

We now obtain our final result by combining the results into Eq. (49). With explicit constants,

<!-- formula-not-decoded -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Yes, we obtain a O((log d ) κ 2 ϵ -1 ) iteration complexity result for BBVI on strong log-concave and log-smooth target distributions. This corresponds to a nearly dimension-independent iteration complexity.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: As stated in Remark 4.3, the main limitation of our work is that the unimprovability result Proposition 4.2 does not fully assert that Lemma 4.1 is tight for any target function ℓ . It only shows that our specific proof strategy, which uses only spectral bounds on the Hessian of ℓ , is unimprovable. In principle, using additional properties of the Hessian could result in a tighter bound.

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

Justification: All assumptions are stated in either Sections 2 and 3 or the propositional statements.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA]

Justification: The paper does not contain any experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.

In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: The paper does not contain any experiments.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so No is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Justification: The paper does not contain any experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The paper does not contain any experiments.

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

Answer: [NA]

Justification: The paper does not contain any experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification:

Guidelines: The content of the paper is a theoretical study of an inference algorithm and does not involve real data.

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: The content of the paper is a theoretical study of an inference algorithm and does not have direct societal consequences.

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

Justification: The paper does not involve real data or models.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper does not involve real data.

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

Justification: The paper does not involve real data.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [No]

Justification: Part of the supporting results were obtained after some minor interaction with LLMs. However, all of the proofs were written and proofread by humans. Therefore, LLMs did not play an important, original role nor did they contribute any non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.