## Beyond Least Squares: Uniform Approximation and the Hidden Cost of Misspecification

## Davide Maran

Politecnico di Milano davide.maran@polimi.it

## Csaba Szepesvári

Google DeepMind and University of Alberta szepi@google.com

## Abstract

We study the problem of controlling worst-case errors in misspecified linear regression under the random design setting, where the regression function is estimated via (penalized) least-squares. This setting arises naturally in value function approximation for bandit algorithms and reinforcement learning (RL). Our first main contribution is the observation that the amplification of the misspecification error when using least-squares is governed by the Lebesgue constant , a classical quantity from approximation theory that depends on the choice of the feature subspace and the covariate distribution. We also show that this dependence on the misspecification error is tight for least-squares regression: in general, no method minimizing the empirical squared loss, including regularized least-squares, can improve it substantially. We argue this explains the empirical observation that some feature-maps (e.g., those derived from the Fourier bases) 'work better in RL' than others (e.g., polynomials): given some covariate distribution, the Lebesgue constant is known to be highly sensitive to choice of the feature-map. As a second contribution, we propose a method that augments the original feature set with auxiliary features designed to reduce the error amplification. We then prove that the method successfully competes with an 'oracle' that knows the best way of using the auxiliary features to reduce this amplification. For example, when the domain is a real interval and the features are monomials, our method reduces the amplification factor to O (1) as d →∞ , while without our method, least-squares with the monomials (and in fact polynomials) will suffer a worst-case error amplification of order Ω( d ) . It follows that there are functions and feature maps for which our method is consistent, while least-squares is inconsistent.

## 1 Introduction

Value function approximation plays a central role in modern reinforcement learning (RL) and contextual bandit algorithms [Sutton and Barto, 2018, Lattimore and Szepesvári, 2020]. In many such settings, policies are evaluated or selected based on value estimates obtained by regressing observed returns. To this end, (penalized) linear regression-based on empirical squared loss-serves as a core subroutine due to its simplicity and favorable computational properties [Ernst et al., 2005, Antos et al., 2008]. A fundamental challenge arises, however, when the true value function or reward model lies outside the span of the chosen features-a situation referred to as model misspecification . Recent work by Du et al. [2020] highlighted that in this setting for any d , there are feature maps so that the worst-case prediction error incurred by least-squares regression can be √ d larger than the misspecification error, even if the learner can control the covariate distribution. 1 This amplification is concerning because it implies that adding more features may not improve the learner's performance if

1 Du et al. [2020] prove the stronger result that any learner that has access to a polynomially sized sample is ought to suffer a 'large' worst-case prediction error no matter the method they use.

the misspecification error decreases at a rate of Ω(1 / √ d ) . As such, in the RL and bandits communities much ink was spilled on the implications and the control of error amplification [Lattimore et al., 2020, Dong and Yang, 2023, Amortila et al., 2023, Maran et al., 2024, Amortila et al., 2024].

In this paper, we further investigate this error amplification and suggest a method designed to drastically reduce it. We study this problem in the context of misspecified linear regression under the random design setting , where inputs are drawn from an unknown distribution. Our first contribution is to identify how the amplification of the misspecification error depends directly on the interaction between the sampling distribution and the feature subspace. Specifically, we show that this amplification is governed by the Lebesgue constant -aclassical quantity in approximation theory that captures how well the 2 -norm projection underlying least-squares regression projects arbitrary functions onto the span of the features. This result provides a significant refinement of previous results in this direction. While prior work established a worst-case, feature-agnostic amplification factor of √ d (which is known to be tight for some feature maps), our approach identifies the governing principle for this amplification, explaining why the true factor can range from as low as 1 for favorable features. This distinction is critical, as it allows for significantly tighter finite-sample guarantees than those derived from the universal √ d scaling. Moreover, we prove that this dependence on the Lebesgue constant is tight: no estimator based on least-squares can substantially improve upon this bound. This sensitivity of least-squares regression to misspecification is in fact a modern re-emergence of a phenomenon that is well known in classical approximation theory. When polynomial bases are used on equispaced grids, the Runge phenomenon causes the uniform error to explode near the boundary despite good L 2 behaviour, precisely because the associated Lebesgue constants grow in an uncontrolled way. Likewise, in Fourier approximation, the well-known Gibbs phenomenon, although far milder than the Runge blow-up, causes localized oscillatory overshoots near discontinuities [Gottlieb and Shu, 1997].

The disparity in error severity is not accidental. It corresponds directly to their differing Lebesgue constant growth rates: logarithmic for Fourier bases, exponential (in the case of equispaced grids) for polynomials. Yet, despite its central role in approximation theory since the early works of Gregory et al. [1848], De La Vallée Poussin et al. [1919] and Szeg˝ o [1939], the link between the Lebesgue constant and misspecification error has, to the best of our knowledge, been overlooked in the modern statistical and reinforcement learning literature. In fact, while the superiority of Fourier bases over polynomials for value function approximation has been empirically observed (see Konidaris et al. [2011]), this phenomenon has lacked a precise theoretical explanation.

Motivated by this connection to the Lebesgue constant, our second contribution is a method for reducing misspecification error amplification . The approach works by augmenting the original feature set with auxiliary features and then using a weighted ridge regression approach to explicitly regularize the corresponding projection operator to give small error amplification. To illustrate, we show that when the domain is an interval and the base and auxiliary features are monomials, our method reduces the amplification factor to O (1) as d →∞ . In contrast, standard least squares suffers from arbitrarily large worst-case errors in the same setting.

## 2 Problem Formulation

We consider the problem of estimating a function with uniform accuracy using a misspecified linear model. We first detail the statistical setting, introduce the standing assumptions and define the performance criterion that will be used in the rest of the paper.

Let R denote the set of reals, X be a measurable input space, and let f : X → R denote an unknown measurable target (or regression) function that we wish to estimate from a sample D n = (( x 1 , y 1 ) , . . . , ( x n , y n )) of n independent, identically distributed pairs that belong to X × R and which satisfy that almost surely for all t ∈ [ n ] := { 1 , . . . , n } ,

<!-- formula-not-decoded -->

We denote the marginal distribution of the inputs x t by µ , while the distribution underlying ( x t , y t ) is denoted by P .

Assumption 1 (Sub-Gaussian Noise) . For some σ &gt; 0 , y 1 -f ( x 1 ) is σ -subgaussian conditionally on x 1 . That is, almost surely,

<!-- formula-not-decoded -->

.

We are interested in the problem of linear function approximation . That is, the goal is to approximate f as a linear combination of d basis functions , φ 1 , . . . , φ d : X → R . When these are clear from the context, for θ ∈ R d , we let f θ : X → R be defined via

<!-- formula-not-decoded -->

We shall also collect them into a feature-map φ : X → R d and write f θ ( x ) = φ ( x ) ⊤ θ , where φ ( · ) = ( φ 1 ( · ) , . . . , φ d ( · )) ⊤ . Motivated by the applications mentioned earlier, we depart from the bulk of the literature in this setting and evaluate performance via the uniform (or maximum) norm. For a function g : X → R , this is defined as ∥ g ∥ ∞ = sup x ∈X | g ( x ) | . We let L ∞ ( X ) denote the set of functions with finite maximum norm. In what follows, we assume that both f and our basis functions φ i belong to this set. For f ∈ L ∞ ( X ) and θ ∈ R d we let

<!-- formula-not-decoded -->

Thus, E ∞ ( θ, f ) is the maximum error suffered when approximating f with f θ . The quantity E ∞ ( f ) represents the best possible uniform approximation error achievable by our basis functions, and its value is unknown to the learner. When E ∞ ( f ) &gt; 0 , we refer to this as the misspecified setting , and refer to E ∞ ( f ) as the misspecification error . In the next section, we investigate the behavior of the error E ∞ ( ˆ θ n , f ) when ˆ θ n is the ordinary least-squares (OLS) parameter estimate 2 :

<!-- formula-not-decoded -->

As we will see, while the OLS estimator is simple, a rigorous analysis of its uniform error under misspecification is involved.

## 3 Characterizing the behavior of OLS

Let F = { f θ : θ ∈ R d } denote the subspace of L ∞ ( X ) spanned by the basis functions underlying the feature-map φ . From now on we assume that the population Gram matrix V µ = ∫ φ ( x ) φ ( x ) ⊤ µ ( dx ) is non-singular. As n →∞ , the Strong Law of Large Numbers ensures that f ˆ θ n converges almost surely to

<!-- formula-not-decoded -->

where ∥ · ∥ µ denotes the L 2 ( µ ) -norm, defined for any measurable g : X → R by ∥ g ∥ 2 µ = ∫ X g 2 ( x ) µ ( dx ) . Since µ is a probability measure, we have ∥ g ∥ 2 µ ≤ ∥ g ∥ 2 ∞ . The map Π d,µ defined in Eq. (1) is the orthogonal projection onto F with respect to the L 2 ( µ ) inner product and is well-defined thanks to our assumption on the Gram matrix V µ . The map Π d,µ is linear, idempotent and satisfies Π d,µ f = f for all f ∈ F . Moreover, it is non-expansive in the L 2 ( µ ) -norm.

By continuity, the almost sure convergence of the OLS estimate implies that, almost surely, lim n →∞ E ∞ ( ˆ θ n , f ) = ∥ Π d,µ f -f ∥ ∞ . The first question we must address is how large ∥ Π d,µ f -f ∥ ∞ can be relative to the best possible error, E ∞ ( f ) . In other words, by how much is the misspecification error E ∞ ( f ) amplified when we use the L 2 ( µ ) projection of f onto F ? The following classical result provides an answer:

Lemma 1 (Lebesgue's lemma, e.g., Proposition 4.1 from Chapter 2 of DeVore and Lorentz [1993] ) . Let ( S , ∥ · ∥ ) be a normed vector space, F a subspace of S and Π : S → S be a linear map such that Π( S ) ⊆ F and Π is an identity on F . 3 Then, for any f ∈ S ,

<!-- formula-not-decoded -->

where ∥ Π ∥ := sup f ∈S : ∥ f ∥≤ 1 ∥ Π f ∥ is the operator norm of Π .

2 In principle, this minimizer may not be unique; however, it is unique under the assumptions required for the results we are about to show.

3 A map Π with this property is called a projection. Note that orthogonal projections are projections, but there are projections of course which are not orthogonal with respect to any inner product.

When Π maps between two normed spaces, ( X, ∥ · ∥ X ) and ( Y, ∥ · ∥ Y ) , we denote its operator norm by ∥ Π ∥ X → Y . Applying this result to our setting with ( S , ∥ · ∥ ) = ( L ∞ ( X ) , ∥ · ∥ ∞ ) we obtain

<!-- formula-not-decoded -->

where ∥ Π ∥ ∞ denotes the operator norm ∥ Π ∥ L ∞ ( X ) → L ∞ ( X ) . In honor of its discoverer, this quantity is formally named as follows:

Definition 1 (Lebesgue constant) . For a linear operator Π : L ∞ ( X ) → L ∞ ( X ) , its induced norm is called the Lebesgue constant associated with Π , and is denoted by

<!-- formula-not-decoded -->

Since the Lebesgue constant of our projection operators will be frequently needed, to minimize clutter we introduce the shorthand

<!-- formula-not-decoded -->

Notably, it is the subspace F -not the specific feature map used to define it-that is the fundamental object: F alone determines the intrinsic misspecification error E ∞ ( f ) , while its interplay with µ governs the projection Π d,µ and the amplification factor Λ d,µ . With the notation just introduced, Lemma 1 yields the upper bound

<!-- formula-not-decoded -->

It is easy to see that Λ d,µ ≥ 1 (consider any nonzero f ∈ F ). Unfortunately, there is no general upper limit on how large this constant can be. Moreoever, the bound in Lemma 1 is essentially tight:

Theorem 1. For any ε &gt; 0 and any subspace F there exist f ∈ L ∞ ( X ) such that

<!-- formula-not-decoded -->

We relegate all proofs to the Appendix. The proof of this specific result can be found in Appendix C.1.

Given the tightness of the lower bound established above, we expect any guarantee on the uniform error E ∞ ( ˆ θ n , f ) to inherently involve Λ d,µ E ∞ ( f ) . Our main result of this section confirms this.

To state the result, we need one additional quantity characterizing the feature space. Let ( φ i ) 1 ≤ i ≤ d be an orthonormal basis of F with respect to L 2 ( µ ) (obtained, for instance, via the Gram-Schmidt procedure on the original basis functions). We let φ : X → R d denote the orthogonalized featuremap defined via φ ( x ) = ( φ 1 ( x ) , . . . , φ d ( x )) ⊤ . We further define ϑ d, 2 = sup x ∈X ∥ φ ( x ) ∥ 2 .

While the orthonormal basis ( φ i ) 1 ≤ i ≤ d is not unique, the quantity ϑ d, 2 is uniquely defined . In particular (see Proposition 16), it is the L 2 ( µ ) → L ∞ ( X ) operator norm of the projection Π d,µ :

<!-- formula-not-decoded -->

Being µ a probability measure, Λ d,µ = ∥ Π d,µ ∥ L ∞ ( X ) → L ∞ ( X ) ≤ ∥ Π d,µ ∥ L 2 ( µ ) → L ∞ ( X ) = ϑ d, 2 .

Theorem 2. Let X be finite and Assumption 1 hold. Let X be finite and Assumption 1 hold. For any δ ∈ (0 , 1 / 3] and any n ≥ 20 ϑ 2 d, 2 log( d/δ ) , the OLS estimate ˆ θ n satisfies, with probability at least 1 -3 δ ,

<!-- formula-not-decoded -->

The first term in the bound matches the deterministic amplification derived in Eq. (2), accounting for the irreducible approximation gap between Π d,µ f and f . The remaining terms bound the additional finite-sample stochastic error. While stated for finite X for simplicity, the result extends to continuous domains via standard covering arguments For example, if X = [ -1 , 1] and the basis functions are L φ -Lipschitz, a uniform bound over X can be obtained by considering an ε/L φ -cover of X . This

extension incurs only a mild logarithmic factor proportional to log( L φ /ε ) . Alternatively, a sometimes tighter bound can be achieved by covering the feature set { φ ( x ) : x ∈ X} ⊂ R d directly.

Beyond Λ d,µ , the bound also depends on the constant ϑ d, 2 , which scales with the dimension d . In particular, we show that ϑ d, 2 ≥ √ d holds regardless of the chosen feature map (see Proposition 18 in the appendix). The scaling of the terms in Theorem 2 aligns with standard expectations. The first term represents the unavoidable approximation error (bias) discussed previously. The remaining terms quantify the estimation error (variance) due to finite sampling. Specifically, the term that involves σ captures the effect of additive noise, while the remaining term accounts for the additional variance induced by the random design, which itself is amplified by the Lebesgue constant and the intrinsic misspecification level. Below we show that when an a priori upper bound ε on E ∞ ( f ) is available (as can be the case in certain numerical applications when the target function belongs to some known class of functions, such as a smoothness class), we can obtain a semi-empirical bound . This bound, which relies on data-dependent quantities, has the potential to significantly improve upon the worst-case bound given in Theorem 2.

A uniform, semi-empirical bound Our purpose here is to bound the uniform error of the OLS estimate using empirical quantities. Let µ n = 1 n ∑ n t =1 δ x t denote the empirical measure associated with the inputs ( x 1 , . . . , x n ) . A key advantage of this analysis is that it relies on the empirical Lebesgue constant Λ d,µ n associated with Π d,µ n , allowing us to drop all assumptions regarding how the inputs are generated (i.e., it applies to fixed design). The empirical operator Π d,µ n takes the form

<!-- formula-not-decoded -->

and Φ ∈ R n × d is the design matrix with rows φ ⊤ ( x t ) . Analogous to the population case, assuming that V µ n is non-singular , we let ( ̂ φ i ) 1 ≤ i ≤ d be an orthonormal basis of F in L 2 ( µ n ) , define the feature map ̂ φ : X → R d via ̂ φ ( x ) = ( ̂ φ 1 ( x ) , . . . , ̂ φ d ( x )) ⊤ , and let ϑ ( n ) d, 2 = sup x ∈X ∥ ̂ φ ( x ) ∥ 2 .

Theorem 3. Let X be finite, Assumption 1 hold, ˆ θ n be the OLS estimate. Then, for any fixed δ &gt; 0 , with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Compared to Theorem 2, this bound offers several improvements: it eliminates the lower-order O (1 /n ) term entirely and it removes the dependence on the misspecification error E ∞ ( f ) from the leading stochastic term (the 1 / √ n term). Furthermore, the population Lebesgue constant Λ d,µ is replaced by its empirical counterpart Λ d,µ n , which may be smaller than Λ d,µ . When X is finite, Λ d,µ n , which is a matrix maximum norm, can be calculated in O ( n |X| ) time.

If the input points ( x 1 , . . . , x n ) can be chosen, one may attempt to optimize this bound directly. Both Λ d,µ n and ϑ ( n ) d, 2 depend on µ n . In experimental optimal design, a G -optimal design minimizes ϑ ( n ) d, 2 by carefully selecting µ n . A fundamental result by Kiefer and Wolfowitz [1960] establishes that for n = Ω( d ) , there exists a design µ n such that ϑ ( n ) d, 2 = O ( √ d ) , which is the best possible scaling.

Assuming that µ n is a G -optimal design, we can compare our result with Proposition 5.1 from Lattimore et al. [2020] (see their equation (2) and the corresponding bound in high probability). Rephrased in our notation, their bound (adapted to high probability with σ = 1 ) states that if µ is an optimal design, then

<!-- formula-not-decoded -->

This result is recoverable as a special case of our Theorem 2. Indeed, for a G -optimal design, ϑ ( n ) d, 2 = √ d , and it is straightforward to show that Λ d,µ n ≤ ϑ ( n ) d, 2 always holds. In the large sample limit ( n →∞ ), the term √ d E ∞ ( f ) dominates their bound. Therefore, our finer bound involving the Lebesgue constant yields a strictly better guarantee whenever Λ d,µ n is smaller than √ d . For example, consider a partition-based feature map where each φ i is an indicator function of a distinct region X i in a partition of X . In this case, Λ d,µ = 1 regardless of µ , offering a massive improvement over the worst-case √ d -factor. While Proposition 5.1 can be refined to replace √ d by ϑ ( n ) d, 2 in the variance term, this improvement still falls short of recovering the Λ d,µ n factor in the bias term.

Table 1: Examples of Lebesgue constants. Domain is X = [ -1 , 1] .

| Basis functions      | µ                          | Λ d,µ        | Source                          | Note       |
|----------------------|----------------------------|--------------|---------------------------------|------------|
| Polynomial           | uniform on regular d -grid | Ω(2 d )      | [Quarteroni et al., 2010]       |            |
| Polynomials          | uniform                    | Θ( d )       | DeVore and Lorentz [1993]       | ϑ d, 2 ≈ d |
| Fourier              | uniform                    | O (log( d )) | [Katznelson, 2004, p.59, Ex. 1] |            |
| Continuous B-splines | uniform                    | O (1)        | Huang [2003]                    |            |
| Wavelets             | uniform                    | O (1)        | Chen and Christensen [2013]     |            |

## 3.1 The Lebesgue constant: properties and particular cases

While we established with Eq. (3) that Λ d,µ ≤ ϑ d, 2 always hold, ϑ d, 2 itself is lower-bounded by √ d (see Proposition 18). To find cases where Λ d,µ is significantly smaller than √ d , we must look at specific feature maps. Table 1 summarizes known results for several classical bases on [ -1 , 1] .

As shown in the table, the Lebesgue constant varies dramatically depending on the basis and measure. Polynomials on a regular d -grid exhibit the worst behavior, with exponential growth Ω(2 d ) . Even with a uniform measure, polynomials still suffer from a linear growth Θ( d ) . In stark contrast, Fourier series enjoy a much slower logarithmic growth O (log d ) . It follows that if a target function's L 2 approximation error decreases as O (1 /d s ) with some s &gt; 0 , the additional uniform error incurred by least-squares is minimal for d large. Interestingly, as was noted earlier, empirical work in reinforcement learning has identified Fourier bases as a strong general-purpose choice [Konidaris et al., 2011]. Our analysis provides a theoretical justification for this: their slowly growing Lebesgue constant ensures reasonable error control even under misspecification.

Finally, localized basis functions like wavelets and B-splines achieve the ideal constant scaling O (1) , independent of d . This makes them excellent candidates when uniform accuracy is paramount. We speculate that tile coding , a popular localized representation in RL, likely shares these favorable extrapolation properties.

A practical limitation of relying on tabulated Lebesgue constants is their dependence on the specific sampling distribution µ . Calculating these constants is non-trivial, and standard results typically only exist for simple, idealized distributions (e.g., uniform). The following proposition provides a way to transfer these known bounds to other distributions, provided they are not too dissimilar:

Proposition 4. Let µ, ν be two discrete probability measures supported on a countable set X such that for all x ∈ X , 0 ≤ c ≤ µ ( x ) ≤ C . Then, Λ d,µ ≤ C Λ d,ν .

<!-- formula-not-decoded -->

## 4 Regularized estimators

Crefthm:lowerboundone establishes a fundamental limitation of the OLS estimator: its worst-case error is inescapably amplified by the Lebesgue constant. Importantly, this bound holds even in the infinite data limit, meaning the issue is not standard overfitting to finite-sample noise. Rather, the problem stems from the geometry of the L 2 ( µ ) -projection itself: due to the rigidity of the feature subspace, minimizing the average error can force the projection Π d,µ f to exhibit large oscillations entirely absent from the target f , particularly in low-density regions.

A natural strategy to dampen such oscillations is regularizing the loss. In the next theorem, however, we show that the standard Ridge Regression approach is ineffective for this purpose, even when the ideal orthonormal basis ¯ φ d is known and used.

Theorem 5. Let ˆ θ n, RIDGE be the λ -ridge regression estimate. For any feature map φ d ( · ) : X → R d , there exists a target function f ∈ L ∞ ( X ) such that, in the infinite data limit,

<!-- formula-not-decoded -->

This result highlights a 'damned if you do, damned if you don't' dilemma for ridge regression. If we choose a large penalty λ ≈ Λ d,µ / 2 to counteract the amplification, the second term in the lower bound

approaches 1 , preventing convergence even as the misspecification error E ∞ ( f ) → 0 . Conversely, if λ is small, we essentially recover the poor Ω(Λ d,µ ) worst-case bound of OLS. Crucially, this phenomenon persists even in the infinite data regime, indicating that it is not merely a sample size issue, but a geometric defect of the projection operator itself. Consequently, standard techniques designed to achieve small test mean-squared error -such as cross-validation or early stopping [Ghojogh and Crowley, 2019]-cannot overcome this fundamental geometric limitation, as they will asymptotically converge to the OLS solution, which is what minimizes the test mean-squared error.

Let us examine why ridge regression fails. The proof of Theorem 5 relies on the explicit form of the corresponding ridge operator Π Ridge d,µ in the infinite data limit:

<!-- formula-not-decoded -->

Importantly, this is not a projection operator because it does not preserve functions in F . For example, applying this operator to the basis function f = φ 1 yields an error of φ 1 -Π Ridge d,µ φ 1 = λ 1+ λ φ 1 . To obtain an error bound that scales with the misspecification E ∞ ( f ) (i.e., a bound that is zero when f ∈ F ), every function in F must be a fixedpoint of the operator. In Eq. (4), this requires α = 1 , which forces λ = 0 , bringing us back to OLS and its associated amplification problems.

Stabilization via Feature Augmentation Instead of ridge regression, we propose a different approach: we augment the feature map and use the additional degrees of freedom purely to 'stabilize' the operator, rather than to improve approximation: After all, our bounds depend on both E ∞ ( f ) and Λ d,µ . Selecting sufficiently many features E ∞ ( f ) may be under control; hence, the idea is to use additional features to control Λ d,µ . Let us denote our original feature map by φ d and its corresponding subspace by F d . We now augment this map with D -d additional features, yielding the extended map φ D . While these extra features could be arbitrary, many standard bases (Fourier, polynomials, splines) have a natural nested structure that provides a canonical sequence of extensions.

Let φ D denote the orthonormal basis for this extended space obtained via Gram-Schmidt on φ D with respect to L 2 ( µ ) . We now define a weighted ridge regression operator on this extended basis. For any sequence of weights λ = ( λ 1 , . . . , λ D ) ∈ [0 , ∞ ) D , let

<!-- formula-not-decoded -->

Crucially, we want this new operator to serve as a superior replacement for Π d,µ while maintaining the approximation power of our original space F d . We do not aim to target the potentially better (but likely much less stable) approximation of the full space F D . This requirement forces us to use zero regularization on the first d components ( λ i = 0 = ⇒ α i = 1 for i ≤ d ), ensuring that every function in F d is fixed by the operator. This leads to the set of valid attenuation parameters :

<!-- formula-not-decoded -->

For any α ∈ A D d , the operator Π Ridge α ,µ fixes every function in F d , thereby avoiding the pitfall of standard ridge regression.

Remark 1 (Connection to Averaging Projections) . This formulation generalizes the classical technique of averaging projections to increase stability. For instance, the averaged operator ¯ Π = 1 D -d +1 ∑ D k = d Π k,µ is exactly equivalent to Eq. (5) with a specific choice of linearly decaying weights α ∈ A D d . The intuition is that while individual high-degree projections Π k,µ may oscillate wildly, these oscillations often cancel out when averaged, leaving a stable estimate. Our framework allows for optimizing these weights directly.

## 4.1 Weighted ridge estimator and the Oracle Operator

Every operator Π Ridge α ,µ with α ∈ A D d maintains the elements of the original subspace F d as its fixed points. Hence, the operators satisfy the conditions of Lebesgue's Lemma (Lemma 1) with F = F d . Denoting by Λ α ,µ the Lebesgue constant of Π Ridge α ,µ , we immediately obtain the following result:

Proposition 6. Let α ∈ A D d (see (6) ) and Π Ridge α ,µ be defined as in (5) . Then, for any f ∈ L ∞ ( X ) ,

<!-- formula-not-decoded -->

Established in Proposition 6 that the error amplification is governed by Λ α ,µ , our goal is to select the parameter α ∈ A D d that minimizes this constant. We refer to this ideal choice as the ORACLE choice:

<!-- formula-not-decoded -->

Unfortunately, α Oracle µ is unknown to the learner, as it depends on the unknown distribution µ . The remainder of this section addresses two key questions:

- Q1 Can we design a finite sample estimator whose error, for fixed α ∈ A D d , asymptotically scales with Λ α ,µ ?
- Q2 Can we design a finite sample estimator whose error asymptotically scales with the optimal Λ Oracle µ ?

Q1 To answer these questions, we first generalize Theorem 3 to incorporate regularization with a chosen parameter α . Although µ is unknown, we can define an empirical counterpart of the operator in Eq. (5) using the empirical measure µ n . Recalling that ̂ φ D is the feature map obtained by orthogonalizing φ D w.r.t. µ n , we have

<!-- formula-not-decoded -->

This empirical operator has two key properties: (1) it depends on the evaluations f ( x t ) , and (2) its output is a linear combination of the basis functions ̂ φ i . Property (1) allows us to estimate Π Ridge α ,µ n f ( · ) from noisy data by simply replacing the unknown values f ( x t ) with the observed targets y t . Property (2) ensures that the resulting estimate can be parameterized as φ D ( · ) ⊤ ˆ θ for some coefficient vector ˆ θ . Specifically, let R n be the upper triangular matrix from the Gram-Schmidt procedure, such that φ D ( · ) ⊤ = ̂ φ D ( · ) ⊤ R n . Letting I α = diag ( α ) be the the regularization weights matrix, we define our estimator as follows

<!-- formula-not-decoded -->

Theorem 7. Let Assumption 1 hold. Then, for any δ &gt; 0 , with probability 1 -δ ,

<!-- formula-not-decoded -->

This result confirms that the amplification error of our estimator scales with Λ α ,µ n . To fully answer question Q1 , we must show that for large n , this empirical constant is a good proxy for the population constant Λ α ,µ . The following proposition establishes this convergence:

Proposition 8. Fix δ &gt; 0 . With probability 1 -δ , the following bounds holds simultaneously for every α ∈ A D d : | ϑ D, 2 -ϑ ( n ) D, 2 | = ˜ O ( ϑ 2 D, 2 √ log(1 /δ ) /n ) , and

<!-- formula-not-decoded -->

Q2 For this more challenging goal, we must optimize α to converge to the oracle value, despite not knowing the true distribution µ . Our strategy is to rely on computable quantities: the empirical operator Π Ridge α ,µ n (Eq. (7)) and its associated empirical Lebesgue constant Λ α ,µ n . A key observation enables this approach: the Lebesgue constant is convex with respect to α :

Proposition 9. The function J : A D d → (0 , + ∞ ) defined by J ( α ) := Λ α ,µ n is convex in α .

<!-- image -->

x

x

Figure 1: Comparison between the OLS estimator and the BWR estimator using polynomial features on [ -1 , 1] , with d = 10 features (left) and d = 15 features (right). The inputs are chosen uniformly at random from [ -1 , 1] . Even if the true function is bounded, OLS suffers from large oscillations near the boundaries due to the high Lebesgue constant. In contrast, BWR achieves a much more uniform approximation error across the domain by effectively controlling the amplification effect.

This convexity allows us to provably find an approximate minimizer in a finite number of iterations. We employ a standard subgradient method [Boyd et al., 2003]: starting from an arbitrary α ∈ A D d , we iteratively update it until convergence. The details are provided in Algorithm 1 (Appendix D.4).

<!-- formula-not-decoded -->

By definition of J , this result guarantees that α ( I ) is an approximate minimizer of the empirical Lebesgue constant Λ α ,µ n . To finally answer Q2 , we define the BWR ( Best Weighted Regularizer ) estimator by plugging α ( I ) into Eq. (8):

<!-- formula-not-decoded -->

Theorem 11. Let Assumption 1 hold and fix δ &gt; 0 . Then, with probability 1 -δ ,

<!-- formula-not-decoded -->

This oracle inequality affirmatively answers Q2 : our estimator is asymptotically able to compete with the Oracle Lebesgue constant.

## 5 Case study: polynomial basis

The method introduced in Section 4 aims to reduce error amplification by explicitly controlling the Lebesgue constant. While broadly applicable, its impact is best illustrated in settings where standard estimators suffer from poor uniform behavior. A canonical example is polynomial regression on a compact interval, where the feature map φ d consists of the first d monomials, { 1 , x, x 2 , . . . , x d -1 } .

Consider the standard setting where X = [ -1 , 1] and the data-generating distribution µ is uniform on this interval. Even in this favorable scenario, the Lebesgue constant for the polynomial basis grows linearly with the degree, Λ d,µ ≈ d . Consequently, the worst-case uniform error for OLS scales as O ( d · E ∞ ( f )) , meaning small misspecification errors can be amplified into large prediction errors.

In contrast, the BWR estimator augments the feature space-for instance, by doubling the degree to D = 2 d -and optimizes the attenuation vector α to minimize the empirical Lebesgue constant. This yields a projection operator that preserves the original degreed polynomials exactly while using the

extra degrees of freedom to stabilize the approximation. Theoretically, this reduces the amplification factor from O ( d ) to O (1) , as the following theorem shows:

Theorem 12. Let µ be the uniform distribution on [ -1 , 1] . There exists a constant C &gt; 0 independent of d such that, if we choose D = 2 d and φ D ( x ) = [1 , . . . x 2 d -1 ] as the augmented feature map for the target space spanned by φ d ( x ) = [1 , . . . , x d -1 ] ⊤ , we have Λ Oracle µ ≤ C.

This theoretical improvement translates directly to empirical performance, as shown in Fig. 1. While OLS exhibits characteristic large oscillations near the boundaries (a manifestation of the classical Runge phenomenon), BWR remains stable across the entire domain. By effectively controlling the Lebesgue constant, BWR achieves a significantly smaller uniform error despite using the same base features for the final representation.

The above simulations visually demonstrate how the amplification factor is exacerbated by increasing d . We complement them with an asymptotic result showing just how severe this factor can be, even for target functions where the approximation error E ∞ ( f ) → 0 as d → ∞ . In fact, there exist a bounded function that can be uniformly approximated by polynomials, yet for which the OLS estimator diverges with a uniform error roughly of order Ω( d ) :

Proposition 13. Fix γ &gt; 0 . Let ˆ θ n be the OLS estimator, and ˆ θ n, BWR be our estimator defined in equation (9) . There exists a function f : [ -1 , 1] → R such that, E ∞ ( f ) → 0 as d →∞ , and under Assumption 1 with µ = U ([ -1 , 1]) , the following hold with probability one:

<!-- formula-not-decoded -->

## 6 Related works

The problem we address, while motivated by the goal of designing principled algorithms for bandits and reinforcement learning, has roots in several fields, including mathematical analysis, econometrics, and approximation theory. We provide a brief overview here, with an extended discussion in Appendix A.

In mathematical analysis the problem of projecting onto a linear subspace of L ∞ ( X ) in a way that minimizes the uniform error have long been a central topic. Classical results on orthogonal polynomials Szeg˝ o [1939] and Fourier series Katznelson [2004] share this goal. More recently, Kobos and Lewicki [2024] proposed an approach for general feature maps. In econometrics , a related line of research studies pointwise estimators based on least-squares from noisy samples [Newey, 1997, Belloni et al., 2015, Li and Liao, 2020], which can be naturally adapted to yield uniform convergence guarantees. Most recently, this problem has resurfaced in bandits and reinforcement learning under the name misspecified linear function approximation [Du et al., 2020, Lattimore et al., 2020, Maran et al., 2024, Dong and Yang, 2023, Amortila et al., 2024].

The specific regularization technique we propose in Section 4 is inspired by classical methods for regularizing Fourier series [de la Vallée Poussin, 1918, De La Vallée Poussin et al., 1919]. Variants of this technique remain an active topic of study in numerical mathematics today [Németh, 2016, Themistoclakis and Van Barel, 2017, Occorsio and Themistoclakis, 2025].

## 7 Conclusion

We investigated the problem of uniform error control in misspecified linear regression under the random design setting. Our key insight is that the amplification of E ∞ ( f ) by least-squares methods is governed by the Lebesgue constant, a fundamental concept from approximation theory. We showed that this amplification is tight and intrinsic to the geometry of L 2 -projection, thereby exposing a fundamental limitation of ordinary and ridge least-squares methods, even in the infinite data regime.

To overcome this limitation, we introduced a novel regularization framework based on weighted ridge regression over extended feature sets, which preserves the approximation power of the base features while using the auxiliary features to stabilize the projection operator. We proved that this approach allows us to, asymptotically for n →∞ , compete with the best possible (oracle) projection in terms of uniform error, and we proposed an efficient algorithm for learning such weights from data. In the canonical case of polynomial features, we demonstrated a dramatic improvement: from Ω( d ) amplification with OLS to the optimal O (1) with our method.

## References

- Luigi Ambrosio, Nicola Fusco, and Diego Pallara. Functions of bounded variation and free discontinuity problems . Oxford university press, 2000.
- Philip Amortila, Nan Jiang, and Csaba Szepesvári. The optimal approximation factors in misspecified off-policy value function estimation. In ICML , pages 768-790, 2023.
- Philip Amortila, Tongyi Cao, and Akshay Krishnamurthy. Mitigating covariate shift in misspecified regression with applications to reinforcement learning. In The Thirty Seventh Annual Conference on Learning Theory , pages 130-160. PMLR, 2024.
- Andras Antos, Csaba Szepesvári, and Rémi Munos. Fitted Q-iteration in continuous action-space MDPs. In Advances in neural information processing systems , pages 9-16, 2008.
- Alexandre Belloni, Victor Chernozhukov, Denis Chetverikov, and Kengo Kato. Some new asymptotic theory for least squares series: Pointwise and uniform results. Journal of Econometrics , 186(2): 345-366, 2015.
- Stéphane Boucheron, Gábor Lugosi, and Olivier Bousquet. Concentration inequalities. In Summer school on machine learning , pages 208-240. Springer, 2003.
- Stephen Boyd, Lin Xiao, and Almir Mutapcic. Subgradient methods. lecture notes of EE392o, Stanford University, Autumn Quarter , 2004(01), 2003.
- Xiaohong Chen and Timothy Christensen. Optimal uniform convergence rates for sieve nonparametric instrumental variables regression. arXiv preprint arXiv:1311.0412 , 2013.
- E.W. Cheney. Introduction to approximation theory . McGraw-Hill, 1966.
- Ch de la Vallée Poussin. Sur la meilleure approximation des fonctions d'une variable réelle par des expressions d'ordre donné. CR Acad. Sci. Paris , 166:799-802, 1918.
- Ch J De La Vallée Poussin et al. Leçons sur l'approximation des fonctions d'une variable réelle . Paris, 1919.
- Ronald A DeVore and George G Lorentz. Constructive Approximation , volume 303 of Grundlehren der mathematischen Wissenschaften . Springer Berlin Heidelberg, Berlin, Heidelberg, January 1993.
- Jialin Dong and Lin Yang. Does sparsity help in learning misspecified linear bandits? In International Conference on Machine Learning , pages 8317-8333. PMLR, 2023.
- Zlatko Drmaˇ c, Matjaž Omladiˇ c, and Krešimir Veseli´ c. On the perturbation of the cholesky factorization. SIAM Journal on Matrix Analysis and Applications , 15(4):1319-1332, 1994.
- Simon S Du, Sham M Kakade, Ruosong Wang, and Lin F Yang. Is a good representation sufficient for sample efficient reinforcement learning? In International Conference on Learning Representations , 2020.
- Damien Ernst, Pierre Geurts, and Louis Wehenkel. Tree-based batch mode reinforcement learning. Journal of Machine Learning Research , 6, 2005.
- Benyamin Ghojogh and Mark Crowley. The theory behind overfitting, cross validation, regularization, bagging, and boosting: tutorial. arXiv preprint arXiv:1905.12787 , 2019.
- D Gottlieb and Chi-Wang Shu. On the gibbs phenomenon and its resolution. SIAM Rev. Soc. Ind. Appl. Math. , 39(4):644-668, December 1997.
- Duncan Farquharson Gregory, Robert Leslie Ellis, William Thomson Baron Kelvin, and Norman Macleod Ferrers. The Cambridge and Dublin Mathematical Journal , volume 7. E. Johnson;[etc., etc.]; Macmillan, Barclay and Macmillan, 1848.
- Daniel Hsu, Sham M Kakade, and Tong Zhang. Random design analysis of ridge regression. Foundations of Computational Mathematics , 14:569-600, 2014.

- Jianhua Z Huang. Local asymptotics for polynomial spline regression. The Annals of Statistics , 31 (5):1600-1635, 2003.
- Yitzhak Katznelson. An introduction to harmonic analysis . Cambridge University Press, 2004.
- Jack Kiefer and Jacob Wolfowitz. The equivalence of two extremum problems. Canadian Journal of Mathematics , 12:363-366, 1960.
- Tomasz Kobos and Grzegorz Lewicki. On the dimension of the set of minimal projections. Journal of Mathematical Analysis and Applications , 529(2):127250, 2024.
- Andrei Kolmogoroff. Über die beste annäherung von funktionen einer gegebenen funktionenklasse. Annals of Mathematics , 37(1):107-110, 1936.
- George Konidaris, Sarah Osentoski, and Philip Thomas. Value function approximation in reinforcement learning using the Fourier basis. In Proceedings of the AAAI conference on artificial intelligence , volume 25, pages 380-385, 2011.
- Tor Lattimore and Csaba Szepesvári. Bandit algorithms . Cambridge University Press, 2020.
- Tor Lattimore, Csaba Szepesvári, and Gellért Weisz. Learning with good feature representations in bandits and in RL with a generative model. In International conference on machine learning , pages 5662-5670. PMLR, 2020.
- Jia Li and Zhipeng Liao. Uniform nonparametric inference for time series. Journal of Econometrics , 219(1):38-51, 2020.
- G.G. Lorentz. Approximation of functions . Athena. New York, Holt, Rinehart and Winston, 1966.
- Davide Maran, Alberto Maria Metelli, Matteo Papini, and Marcello Restelli. Local linearity: the key for no-regret reinforcement learning in continuous MDPs. arXiv preprint arXiv:2410.24071 , 2024.
- Zsolt Németh. De la Vallée Poussin Type Approximation Methods . PhD thesis, Eötvös Loránd University, Hungary, 2016.
- Whitney K Newey. Convergence rates and asymptotic normality for series estimators. Journal of econometrics , 79(1):147-168, 1997.
- Donatella Occorsio and Woula Themistoclakis. De la Vallée Poussin filtered polynomial approximation on the half-line. Applied Numerical Mathematics , 207:569-584, 2025.
- Allan Pinkus. N -widths in Approximation Theory , volume 7. Springer Science &amp; Business Media, 2012.
- Alfio Quarteroni, Riccardo Sacco, and Fausto Saleri. Numerical mathematics , volume 37. Springer Science &amp; Business Media, 2010.
- Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction . The MIT Press, second edition, 2018.
- Gábor Szeg˝ o. Orthogonal polynomials , volume 23. American Mathematical Society, 1939.
- Woula Themistoclakis and Marc Van Barel. Generalized de la Vallée Poussin approximations on [ -1 , 1] . Numerical Algorithms , 75:1-31, 2017.
- Joel A Tropp et al. An introduction to matrix concentration inequalities. Foundations and Trends® in Machine Learning , 8(1-2):1-230, 2015.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: -

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discussed the limits of the paper and the future research directions in order to address them.

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

Justification: All the statements are provided with proofs in the appendix.

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

Justification: We include the code in the supplementary material (very simple, just one very short Jupyter notebook)

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

Justification: We include the code in the supplementary material (very simple, just one very short Jupyter notebook)

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

Justification: We include the code in the supplementary material (very simple, just one very short Jupyter notebook)

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: -

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

Justification: The simulation is straightforward and its computational time is negligible.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The paper is coherent with NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: -

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

Justification: -

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: -

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

Justification: -

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: -

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: -

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: -

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/ LLM ) for what should or should not be described.

## A Related Works

Classical approximation theory The idea of approximating a class of functions with a family of vector spaces in a uniform sense has always been an important topic in mathematical analysis. On the more general level, this theory takes the name of Kolmogorov's n-width (Kolmogoroff [1936]; see Lorentz [1966] and Pinkus [2012] for a more modern formalization). The idea, central to this paper, of finding a linear operator that well approximates the non-linear L ∞ projection operator has also been the main topic of multiple line of research. In particular, many result about orthogonal polynomials Szeg˝ o [1939] or Fourier Series Katznelson [2004] approximation have this goal. More recently, Kobos and Lewicki [2024] studied the problem for general feature map, investigating the class of linear operators that achieve the lower bound.

Asymptotic pointwise and uniform convergence of LS series in the econometric literature In the econometric literature, the series least squares (LS) estimators have been analyzed primarily through an asymptotic lens: with the sample size n → + ∞ and the basis dimension d → + ∞ , one studies asymptotic Gaussianity of the estimator of the function in each single point. Newey [1997] provided seminal results for this literature, which were then improved by Belloni et al. [2015], the first to use the Lebesgue constant in this field, and by Li and Liao [2020], who generalize the result to time series data. All these contributions, however, remain asymptotic : they provide limiting distributions or rates without explicit high-probability bounds, and-crucially-they do not propose algorithmic modifications capable of reducing the amplification factor induced by the Lebesgue constant.

̸

Uniform bounds for linear regression in the context of Online Learning As anticipated in the introduction, the problem of getting L ∞ bounds for regression over a domain naturally arises in the context of Online Learning with linear function approximation; bandits and RL in particular. Du et al. [2020] established the first √ d amplification lower bound in some specific cases, which was then refined by Lattimore et al. [2020], who also derives the corresponding an upper bound of √ d , using an optimal design argument. In fact, it can be proved that the factor √ d is precisely the maximal Lebesgue constant of any feature map for µ that is the optimal design. These lower bound hold for a worst-case feature map, but allowing the learner to choose the data distribution . Following these works, many papers tried to understand how this amplification factor could be reduced. Maran et al. [2024] shows how to remove it in case of a locally linear feature map; Dong and Yang [2023] improves the √ d amplification in case of sparsity. Perhaps, the most similar paper to our one is Amortila et al. [2024], which proposes a method to mitigate the effect of misspecification w.r.t. the least-squares fitting. Still, the latter focuses on a different objective, i.e. the error under covariate shift (measuring the MSE under a distribution ν = µ ), and scales with the density ratio ν ( · ) /µ ( · ) . Generalizing to the uniform error would mean to take ν ( · ) as a Dirac's delta, which would make this bound vacuous.

De la Valleè Poussin approach The to reduce the Lebesgue constant by adding auxiliary features is rooted in a concept that dates back in the history of mathematics to Baron de la Vallée Poussin [de la Vallée Poussin, 1918, De La Vallée Poussin et al., 1919]. The technique he invented is still studied today in numerical mathematics [Németh, 2016, Themistoclakis and Van Barel, 2017, Occorsio and Themistoclakis, 2025].

Finite-sample bounds for ridge regression Hsu et al. [2014] gives finite-sample bounds for ridge regression under random design. The results, when translated into our setting, bound the error between f ˆ θ n and ¯ f where ¯ f := g ◦ φ and the bound is expressed in terms of ¯ f -Π µ,d f . Here for u ∈ R d , g ( u ) = ∫ f ( x ) µ ( dx | u ) where µ ( dx | u ) is the disintegration of µ with respect to the push-forward of µ under φ . In particular, for S ⊂ X , u ∈ R d , µ ( S | u ) = ∫ I ( x ∈ S, φ ( x ) = u ) µ ( dx ) . In the special case when φ is injective, ¯ f = f . Just like in the result that can be extracted from the work of Lattimore et al. [2020], the bounds in this work depend on ϑ d, 2 (or ϑ ( n ) d, 2 ) and scale similarly.

In fact, papers like Lattimore et al. [2020] adopt the following way to bound the uniform error of least squares. Let V n = ∑ n t =1 φ ( x t ) φ ( x t ) ⊤ be the Gramian matrix and θ ⋆ be the vector realizing the L ∞ projection, so that φ ( x ) ⊤ θ ⋆ = Π ∞ f ( x ) . Then each y t takes the form φ ( x t ) ⊤ θ ⋆ + ε ( x t ) + η t ,

meaning

<!-- formula-not-decoded -->

While the stochastic part is bounded as in our Theorems 2 and 3, the one containing the misspecification is treated as follows

<!-- formula-not-decoded -->

By definition, ( V n /n ) -1 / 2 φ ( x ) = ̂ φ ( x ) . Therefore, when making the supremum over x ∈ X we end up with ϑ ( n ) d, 2 E ∞ ( f ) . As we pointed out in the main paper and also noted by Lattimore et al. [2020], whatever the choice of ( x t ) t and the feature map, ϑ ( n ) d, 2 ≥ √ d . Therefore, this strategy is doomed to achieve sub-optimal guarantees, whenever Λ d,µ n &lt; O ( √ d ) .

## B General-interest results

We start from the usual Bernstein's inequality Boucheron et al. [2003], here written for variables that are bounded in [ -B,B ] and in the "high probability" form.

Theorem 14. Let ( x t ) n t =1 be a sequence of zero-mean random variable bounded in [ -B,B ] . Let σ 2 := ∑ n t =1 Var ( X t ) . Then, with probability at least 1 -δ

<!-- formula-not-decoded -->

Lemma 2. Let φ d be an orthonormal feature map w.r.t. ρ .

<!-- formula-not-decoded -->

where I d is the d -dimensional identity matrix.

Proof. In this proof, let us denote with e i , for i = 1 , . . . d , the standard basis of R d . By definition of outer product between two vectors we get what follows.

<!-- formula-not-decoded -->

This completes the proof.

Lemma 3. Let { v t } k t =1 be a sequence of independent d -dimensional random vectors such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

2. W.p. at least 1 -δ

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Note that, as λ max ( v t v ⊤ t ) = ∥ v t ∥ 2 s ≤ B , we can then apply Theorem 5.1.1 from Tropp et al. [2015] taking

<!-- formula-not-decoded -->

Let V := ∑ k t =1 v t v ⊤ t . Then,

1. W.p. at least 1 -δ

which ensures that

<!-- formula-not-decoded -->

while

<!-- formula-not-decoded -->

The thesis is going to follow by just simplifying the previous expressions. We recall from elementary Taylor expansions that

<!-- formula-not-decoded -->

and

This tells us that and

<!-- formula-not-decoded -->

We can reformulate the previous results in the high-probability notation. Indeed, taking δ = de -kσ 2 ε 2 / (5 B ) , we get

<!-- formula-not-decoded -->

which entails that

<!-- formula-not-decoded -->

Doing the same for the other result, we get

<!-- formula-not-decoded -->

which completes the proof.

Proposition 15. The Lebesgue constant satisfies Λ d,µ = sup x ∈X ∫ X ∣ ∣ ∣ ∑ d i =1 φ i ( z ) φ i ( x ) ∣ ∣ ∣ dµ ( z ) .

Proof. See Cheney [1966], chapter 4.

<!-- formula-not-decoded -->

Therefore, we have, for ε &lt; 0 . 5

<!-- formula-not-decoded -->

On the other side, for ε ≤ 1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C Proofs from Section 3

## C.1 Lower bound for LS

Let Π ∞ f = arg min g ∈F ∥ f -g ∥ ∞ with ties broken arbitrarily. Theorem 1.1 of Chapter 3 in the book of DeVore and Lorentz [1993] guarantees that at least one minimizer exists. (As discussed there, uniqueness may or may not hold.)

Lemma 4. We have

<!-- formula-not-decoded -->

Proof. By definition of Lebesgue constant, for every ε &gt; 0 there is a function g such that

<!-- formula-not-decoded -->

Take f = Π ∞ g -g . Wewill use twice that for any h ∈ F , ∥ h ∥ ∞ = ∥ 0 -h ∥ ∞ ≥ inf u ∈F ∥ u -h ∥ ∞ = ∥ Π ∞ h -h ∥ ∞ . Now,

<!-- formula-not-decoded -->

The result follows by letting ε → 0 .

Theorem 1. For any ε &gt; 0 and any subspace F there exist f ∈ L ∞ ( X ) such that

<!-- formula-not-decoded -->

Proof. The result is immediate from Lemma 4.

For the next result we abbreviate ∥ · ∥ L 2 ( µ ) → L ∞ ( X ) by ∥ · ∥ 2 →∞ .

## Proposition 16.

<!-- formula-not-decoded -->

Proof. The following equalities hold:

<!-- formula-not-decoded -->

In particular, the first is by definition of induced 2 → ∞ -norm, the second one by definition of projection operator, and the third by definition of infinity norm. The fourth passage follows exchanging the two supremum, while the fifth from Parseval's theorem and the sixth one by duality of the two-norm (i.e. for any w , w = sup ∥ v ∥ 2 =1 ⟨ v , w ⟩ ).

## C.2 Towards the proof of Theorem 2

Lemma 5. Fix δ &gt; 0 , and n ≥ 20 ϑ 2 d, 2 log( d/δ ) . Let

<!-- formula-not-decoded -->

Then, λ min ( V n ) ≥ n/ 2 .

Proof. The matrices we are summing correspond to φ ( x t ) φ ( x t ) ⊤ each one being semi-positive definite with the biggest eigenvalue bounded by ϑ 2 d, 2 almost surely (indeed, v ⊤ φ ( x t ) φ ( x t ) ⊤ v is maximized for v parallel to φ ( x t ) and produces ∥ φ ( x t ) ∥ 2 2 ). Moreover, as we have seen in Lemma 2,

<!-- formula-not-decoded -->

These two ingredients allow us to apply Lemma 3 part one, which ensures that with probability at least 1 -δ

<!-- formula-not-decoded -->

if ( 1 -√ 5 ϑ 2 d, 2 log( d/δ ) n ) ≤ 1 / 2 . Therefore, taking n ≥ 20 ϑ 2 d, 2 log( d/δ ) , we get λ min ( V n ) ≥ n/ 2 , which completes the proof.

Lemma 6. Let ζ ( · ) := f ( · ) -Π d,µ f ( · ) . With probability at least 1 -δ ,

<!-- formula-not-decoded -->

plus a lower-order term depending on n -1 which takes the form of ˜ O ( n -1 d 1 / 2 ϑ 2 d, 2 Λ d,µ E ∞ ( f ) + n -3 / 2 dϑ 3 d, 2 Λ d,µ E ∞ ( f ) ) .

Proof. We start rearranging the equation as follows

<!-- formula-not-decoded -->

For ∆ n := ( V n /n ) -1 -I d . To bound both parts, we start by giving a result for 1 n ∑ n t =1 v ⊤ φ ( x t ) ζ ( x t ) that holds for one fixed v ∈ R d . Indeed,

1. Every random variable v ⊤ φ ( x t ) ζ ( x t ) is bounded by ∥ v ∥ 2 ϑ d, 2 Λ d,µ E ∞ ( f ) a.s.
2. The variance of the same random variable is

<!-- formula-not-decoded -->

the main step following from Lemma 2.

So by Bernstein's inequality (Theorem 14),

<!-- formula-not-decoded -->

We can use the previous equation to bound both parts. For the first, we just take v = φ ( z ) , which respects ∥ v ∥ 2 ≤ ϑ d, 2 , in Eq. (10) and get

<!-- formula-not-decoded -->

Let us now focus on the second part. Indeed,

<!-- formula-not-decoded -->

Now, using Lemma 3 as done in the proof of Lemma 5, we have

<!-- formula-not-decoded -->

while for the last part we can write

<!-- formula-not-decoded -->

where B 1 /n d is a 1 /n covering of the set of vectors such that ∥ v ∥ 2 = 1 . It is well-known that we can choose B 1 /n d so that | B 1 /n d | ≈ n -d , so that, making a union bound together with Eq. (10), we get

<!-- formula-not-decoded -->

As a consequence,

<!-- formula-not-decoded -->

This completes the proof.

## C.3 Proof of Theorem 2

Theorem 2. Let X be finite and Assumption 1 hold. Let X be finite and Assumption 1 hold. For any δ ∈ (0 , 1 / 3] and any n ≥ 20 ϑ 2 d, 2 log( d/δ ) , the OLS estimate ˆ θ n satisfies, with probability at least 1 -3 δ ,

<!-- formula-not-decoded -->

Proof. In this proof, let ζ d,µ ( · ) := f ( · ) -Π d,µ f ( · ) and η t = y t -f ( x t ) . Moreover, we will call ˆ θ n the OLS estimator parametrized w.r.t. φ , rather than φ . We will also call ̂ f n ( · ) = φ ( · ) ⊤ ˆ θ n the corresponding estimated function (which does not change with the parameterization of the basis, as it only depends on F ).

We start making the following decomposition:

<!-- formula-not-decoded -->

To bound the first part, we let θ ⋆ be such that Π d,µ f ( · ) = φ ( · ) ⊤ θ ⋆ . By Assumption 1, the samples take the form y t = φ ( x t ) ⊤ θ ⋆ + ζ d,µ ( x t ) + η t , where ( η t ) n t =1 is a family of independent σ -subgaussian random variables. By definition, letting V n = ∑ n t =1 φ ( x t ) φ ( x t ) ⊤ , the LS solution takes the form φ ( x t ) ⊤ ˆ θ n , where

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

We are going to bound the two terms separately. First, let E := { λ min ( V n ) ≥ n/ 2 } . From Lemma 5, under the assumptions of this theorem, we have P ( E ) ≥ 1 -δ .

- (I) Since η t are independent and σ -subgaussian conditionally to ( x t ) n t =1 (Assumption 1), Lemma 5.4 and Theorem 5.3 from Lattimore and Szepesvári [2020] ensure that, with probability at least 1 -2 δ

<!-- formula-not-decoded -->

Moreover, if event E holds,

<!-- formula-not-decoded -->

so that the full term is bounded by √ 8 log(1 /δ ) σϑ d, 2 n -1 / 2 .

- (II) This term is bounded by Lemma 6 which, with probability at least 1 -δ gives

<!-- formula-not-decoded -->

plus lower-order terms of the form poly ( d,ϑ d, 2 , Λ d,µ E ∞ ( f )) n .

Note that, thanks to Lemma 5, event E holds with probability 1 -δ under the assumptions of this theorem. Moreover, imposing that both events in ( I ) and ( II ) verify, we get, with probability at least 1 -3 δ ,

<!-- formula-not-decoded -->

plus lower-order terms of the form poly ( d,ϑ d, 2 , Λ d,µ E ∞ ( f )) n . This completes the proof.

## C.4 Bound scaling with the empirical Lebesgue constant

Theorem 3. Let X be finite, Assumption 1 hold, ˆ θ n be the OLS estimate. Then, for any fixed δ &gt; 0 , with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Proof. In this proof, let ζ ( · ) := f ( · ) -Π d,µ n f ( · ) and η t = y t -f ( x t ) . Moreover, we will call ˆ θ n the OLS estimator parametrized w.r.t. ̂ φ , rather than φ . We will also ̂ f n ( · ) = ̂ φ ( · ) ⊤ ˆ θ n the corresponding estimated function (which does not change with the parameterization of the basis, as it only depends on F ).

The following decomposition holds:

<!-- formula-not-decoded -->

Now, we focus on the second term. As done in the previous proof of Theorem 2, we let θ ⋆ be such that Π d,µ n f ( · ) = ̂ φ ( · ) ⊤ θ ⋆ and ζ ( · ) := f ( · ) -̂ φ ( · ) ⊤ θ ⋆ . In this way, our samples take the form y t = ̂ φ ( x t ) ⊤ θ ⋆ + ζ ( x t ) + η t .

For any fixed x ∈ X we have

<!-- formula-not-decoded -->

Here, the last passage is due to the fact that, being ̂ φ ( · ) orthogonal w.r.t. µ n ( · ) , it follows 1 n ∑ n t =1 ̂ φ ( x t ) ̂ φ ( x t ) ⊤ = I d . Now, we analyze the two terms ( I ) and ( II ) separately.

<!-- formula-not-decoded -->

In fact, by definition of orthogonal projection, ζ ( · ) is orthogonal in L 2 ( µ n ) to the span of ̂ φ ( · ) , so to each of its components in particular.

Let us look at the second term. Since η t are independent and σ -subgaussian conditionally on ( x t ) n t =1 , Lemma 5.4 and Theorem 5.3 from Lattimore and Szepesvári [2020] ensure that, with probability at least 1 -2 δ

<!-- formula-not-decoded -->

Where the second passage comes once again from the fact that 1 n ∑ n t =1 ̂ φ ( x t ) ̂ φ ( x t ) ⊤ = I d . This proves that ( II ) is bounded by √ 2 log(1 /δ ) σn -1 / 2 ̂ φ 2 ,d . Making a union bound over x ∈ X , this entails w.p. 1 -δ ,

<!-- formula-not-decoded -->

We have proved that

<!-- formula-not-decoded -->

## C.5 Proofs from Section 3.1

Proposition 17. The Lebesgue constant is bounded by Λ d,µ ≤ ϑ d, 2 .

Proof. Let f ∈ L ∞ ( X ) with ∥ f ∥ ∞ = 1 . We have, for any x ∈ X ,

<!-- formula-not-decoded -->

the last passage coming from the fact that as ρ is a probability measure, ∥ f ∥ µ ≤ ∥ f ∥ ∞ . The thesis follows taking the supremum on f, x .

Proposition 18. Let φ d : X → R d be any feature map, and ρ a probability measure. Then,

<!-- formula-not-decoded -->

Proof. The key for this result is to note that, being ρ a probability measure, ϑ 2 d, 2 ≥ E x ∼ ρ [ ∥ φ d ( x ) ∥ 2 2 ] (the supremum of a function upper bounds its integral on any probability measure). Then,

<!-- formula-not-decoded -->

Where the passage ( ∗ ) comes from Lemma 2.

Proposition 19. Let X = [ k ] and φ i ( j ) = X ij , with all the X ij being independent bounded zero-mean unit variance random variables. Then, if d = O ( √ k ) , the feature map φ d , satisfies

<!-- formula-not-decoded -->

with probability at least 1 -δ . Moreover, E [Λ d,µ ] ≥ Ω( √ d ) .

Proof. By convenience, we call Φ ∈ R k × d the matrix having, as columns, the features of φ d . Precisely, the i -th column of Φ corresponds to φ i . It is well-known that, in a finite dimensional space the orthogonal projection operator writes as

<!-- formula-not-decoded -->

We call Φ m · the m -th row of Φ which, by assumption, is a random vector of independent entries bounded in [ -B,B ] and with variance one. We have

<!-- formula-not-decoded -->

At this point, we can apply Lemma 3, that ensures with probability 1 -2 δ , for k sufficiently large,

<!-- formula-not-decoded -->

Now, we can fix σ = 1 as in the assumption and rewrite the projection operator in the following form

<!-- formula-not-decoded -->

where ∆ has all the eigenvalues of magnitude less than √ 5 dB 2 log( d/δ ) kσ 2 , by the previous result.

We now bound the infinity norm of the two terms separately. First,

<!-- formula-not-decoded -->

where ∗ holds since the infinity norm of a matrix corresponds to the maximum 1 -norm between its rows. Now, note that, as the rows are independent, each variable ∑ d i =1 Φ mi Φ ni , for m = n is a sum of i.i.d. random variables such that

- Φ mi Φ ni is bounded in [ -B 2 , B 2 ] almost surely.
- The variance is

<!-- formula-not-decoded -->

Therefore, Bernstein's inequality (14) ensures that, w.p. 1 -δ

<!-- formula-not-decoded -->

̸

Making a union bound over the k 2 -k pairs m = n , we get, still with probability at least 1 -δ ,

̸

<!-- formula-not-decoded -->

At this point, we simply have, with probability 1 -δ ,

̸

<!-- formula-not-decoded -->

For the second term, we have

<!-- formula-not-decoded -->

̸

̸

where ∗ comes from the bound on the eigenvalues of ∆ . Putting everything together, we have proved that

<!-- formula-not-decoded -->

To show that we cannot go much lower than this quantity, note that, even ignoring the contribution of ∆ we have

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

̸

̸

The last passage comes from the fact that, for n = m , we have the expected value of the modulus a sum of d independent random variables, which grows as √ d .

Proposition 20. Let µ, ν be two discrete probability measures supported on a countable set X such that for all x ∈ X , 0 ≤ c ≤ µ ( x ) ν ( x ) ≤ C . Then, Λ d,µ ≤ C c Λ d,ν .

Proof. The following identity holds for the Lebesgue constant

<!-- formula-not-decoded -->

where G ( µ ) = ∫ X φ ( x ) φ ( x ) ⊤ dµ ( x ) and R ( µ ) is its Cholesky factor, such that R ( µ ) ⊤ R ( µ ) = G ( µ ) ; here, the second passage comes from the fact that the Cholesky factor of a matrix corresponds to the R factor in the QR factorization, which is the one giving Graham-Schmidt orthogonalization Quarteroni et al. [2010]. In fact, letting φ ( x ) be the basis orthonomalized w.r.t. µ , we have

<!-- formula-not-decoded -->

Note that, by absolute continuity, we have, for any x ∈ X

<!-- formula-not-decoded -->

Passing to the supremum, we get the thesis.

## D Proofs from Section 4

## D.1 Lower bound for standard ridge regression

Lemma 7. Let Π λ d,µ be the operator defined in this way:

<!-- formula-not-decoded -->

Then, we have

<!-- formula-not-decoded -->

This completes the proof.

<!-- formula-not-decoded -->

Proof. We start from the definition of θ λ :

<!-- formula-not-decoded -->

where the last passage comes from Parseval's theorem, as ζ d,µ is orthogonal in L 2 to the span of φ , while Π d,µ f ( · ) , φ ( · ) ⊤ θ belongs to this vector space. We then write the operator Π d,µ f explicitly:

<!-- formula-not-decoded -->

The last passage holds from Parseval's theorem since φ i are orthonormal in L 2 . Note that, as the θ i in the last minimization problem are disentangled, we can find as explicit solution

<!-- formula-not-decoded -->

Lemma 8. Let Π λ d,µ be defined according to Eq. (12) . For every feature map φ we have

<!-- formula-not-decoded -->

Proof. By definition of Lebesgue constant, for every ε &gt; 0 there is a function g such that

<!-- formula-not-decoded -->

Take f = Π ∞ g -g . We have, by Lemma 7,

<!-- formula-not-decoded -->

At this point, note that as follows from

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using this property, we have

<!-- formula-not-decoded -->

At this point, using the definition of g ,

<!-- formula-not-decoded -->

The thesis follows letting ε → 0 .

Theorem 5. Let ˆ θ n, RIDGE be the λ -ridge regression estimate. For any feature map φ d ( · ) : X → R d , there exists a target function f ∈ L ∞ ( X ) such that, in the infinite data limit,

<!-- formula-not-decoded -->

Proof. Let ̂ f n be the output of λ -ridge regression, that is the function φ ( · ) ⊤ ˆ θ n , where

<!-- formula-not-decoded -->

By the uniform law of large numbers, in the limit, the minimizer ̂ f n converges to Π λ d,µ f , the regularized projection operator is defined as follows

<!-- formula-not-decoded -->

We start showing the λ λ +1 lower bound. Taking any function in the span of φ ( · ) with ∥ f ∥ ∞ = 1 we have, by Lemma 7,

<!-- formula-not-decoded -->

To show the other part, use Lemma 8 to define a function f such that

<!-- formula-not-decoded -->

Replacing ∥ Π ∞ f -f ∥ ∞ = E ∞ ( f ) completes the proof.

## D.2 Proofs from Section 4.1

Theorem 7. Let Assumption 1 hold. Then, for any δ &gt; 0 , with probability 1 -δ ,

<!-- formula-not-decoded -->

Proof. In this proof, let ζ ( · ) := f ( · ) -Π Ridge α ,µ n f ( · ) and η t = y t -f ( x t ) and ˆ θ n be the estimator corresponding to Π Ridge α ,µ n in the parameterization of ̂ φ D ( · ) , so that

<!-- formula-not-decoded -->

The following decomposition holds:

<!-- formula-not-decoded -->

where we have applied Proposition 6 for µ n . Let us focus on the second term. As in the proof of the previous theorems, we call θ ⋆ the vector corresponding to the orthogonal projection over ̂ φ D ( · ) so that we have, for every x ∈ X

<!-- formula-not-decoded -->

By orthogonality, the first term corresponds to

<!-- formula-not-decoded -->

The second term is

<!-- formula-not-decoded -->

by definition of orthogonal projection. The third term is

<!-- formula-not-decoded -->

which can be bounded as the corresponding terms in Theorems 2 and 3: as η t are independent and σ -subgaussian subgaussian conditionally on ( x t ) n t =1 , Lemma 5.4 and Theorem 5.3 from Lattimore and Szepesvári [2020] ensure that, with probability at least 1 -2 δ

<!-- formula-not-decoded -->

Where the only difference w.r.t. the other proofs is the presence of I α , which is erased after the first step since, being α ∈ A D d , its norm is ≤ 1 . This proves that the last term is bounded by √ 2 log(1 /δ ) σn -1 / 2 ̂ φ 2 ,D . Making a union bound over X gives, w.p. 1 -δ ,

<!-- formula-not-decoded -->

Putting everything together, we have proved that

<!-- formula-not-decoded -->

Proposition 21. Fix δ &gt; 0 . With probability 1 -δ , the following bounds holds simultaneously for every α ∈ A D d : | ϑ D, 2 -ϑ ( n ) D, 2 | = ˜ O ( ϑ 2 D, 2 √ log(1 /δ ) /n ) , and

<!-- formula-not-decoded -->

We prove this theorem for a generic d ∈ N . The result follows for d = D .

We define V n := 1 n ∑ n t =1 φ d ( x t ) φ d ( x t ) ⊤ . Let ̂ φ d ( · ) the basis obtained from φ d by Gram-Schmidt orthogonalization w.r.t. µ n , the empirical distribution of the { x t } t . As in the main paper, we let R n = Chol ( V n ) and, since the Cholesky factor corresponds to the matrix given by Graham Schmidt orthogonalization (proposition 3.4 in Quarteroni et al. [2010]),

<!-- formula-not-decoded -->

so that, under this convenient normalization, we can pass from φ d ( x t ) to ̂ φ d ( x t ) trough a matrix that is exactly the Cholesky factor of V n . In this setting, Theorem 2.1. in Drmaˇ c et al. [1994], which provides a stability result for the Cholesky decomposition which, combined with our theorem gives

<!-- formula-not-decoded -->

We can now proceed with the proof.

## Proof. Bounding norm difference

We have to measure

## Lebesgue constants difference

Let us bound the distance between the estimated and the true Lebesgue constant, for any α ∈ A D d ,

<!-- formula-not-decoded -->

As we said, the relation between the two is φ d ( x ) = R ⊤ n ̂ φ d ( x ) which we can also wite as R -⊤ n φ d ( x ) = ̂ φ d ( x ) , so that

<!-- formula-not-decoded -->

At this point, equation (14) ensures that ∥ I d -R -⊤ n ∥ 2 → 2 = O ( ϑ d, 2 √ log(1 /δ ) /n log( d ) ) , so we get

<!-- formula-not-decoded -->

A simple yet useful consequence of this result is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the following, we call

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

## Bound the first term .

Fix α ∈ A D d ,

<!-- formula-not-decoded -->

Where, I α = diag ( α ) . At this point, we can replace the result of Eq. (13): getting

<!-- formula-not-decoded -->

This formulation allows us to apply Eq. (14): As I α is diagonal matrix with elements in [0 , 1] , we have

<!-- formula-not-decoded -->

This gives the following

<!-- formula-not-decoded -->

Here, the first equality is due to the fact that, being ̂ φ d orthonormal w.r.t. µ n , we have ∑ n t =1 ∥ ̂ φ d ( x t ) ∥ 2 2 = nd . This holds uniformly for every α , as we have only used the fact that ∥ I α ∥ 2 ≤ 1 .

## Bounding the second term.

The second term corresponds to

<!-- formula-not-decoded -->

First, we fix x ∈ X and α ∈ A D d and use the scalar product to write it as

<!-- formula-not-decoded -->

Note that by definition

<!-- formula-not-decoded -->

Moreover,

<!-- formula-not-decoded -->

where the last step comes from the fact that I 2 α ⪯ I d . For the same reason, we also have | φ d ( x ) ⊤ I α φ d ( x t ) | ≤ ϑ 2 d, 2 almost surely. These three results allow us to apply Bernstein's inequality (14) for

<!-- formula-not-decoded -->

This gives, with probability at least 1 -δ ,

<!-- formula-not-decoded -->

So, we can bound Eq. (20), which corresponds to 1 n | ∑ n t =1 X t | , as follows.

<!-- formula-not-decoded -->

The former holds for any fixed α ∈ A D d . To have a uniform bound, let

<!-- formula-not-decoded -->

so that log |A ′ | ≤ d log( nϑ d, 2 ) . Making a union bound gives, ∀ α ∈ A ′

<!-- formula-not-decoded -->

To pass to the general case, note that for every α ∈ A D d there is α ′ ∈ A ′ such that ∣ ∣ 1 n ∑ n t =1 ∣ ∣ φ d ( x ) ⊤ I α φ d ( x t ) ∣ ∣ -∫ X ∣ ∣ φ d ( x ) ⊤ I α φ d ( z ) ∣ ∣ dµ ( z ) ∣ ∣ changes no more than 2 ϑ d, 2 between the two, by definition of ε -cover. Therefore, we have, with probability at least 1 -δ over all α ∈ A D d at the same time

This means,

<!-- formula-not-decoded -->

Putting the two results together. By the two bounds that we got for the two terms, it follows with probability at least 1 -δ

<!-- formula-not-decoded -->

To end the proof, note that, using Eq. (19), the difference between ϑ d, 2 and ϑ ( n ) d, 2 is of order ϑ 2 d, 2 √ log(1 /δ ) /n , so that

<!-- formula-not-decoded -->

Finally, note that, as √ d ≤ ϑ d, 2 , the term √ dϑ 3 d, 2 log(1 /δ ) n dominates over dϑ 2 d, 2 n log(1 /δ ) that we had before.

## D.3 Proofs about gradient method

Proposition 22. The function J : A D d → (0 , + ∞ ) defined by J ( α ) := Λ α ,µ n is convex in α .

Proof. By definition,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where M ( α ) = 1 n ∑ d i =1 α i ̂ φ i ( x ) ̂ φ i ( x t ) . Therefore, in particular

<!-- formula-not-decoded -->

This function is convex, being the supremum of a family of linear functions 1 n ∑ d i =1 α i ̂ φ i ( x ) ̂ φ i ( x t ) in α .

Theorem 10. Fix ϵ &gt; 0 . After I = ˜ O ( ϵ -2 ϑ ( n ) D, 2 2 ( D -d )) iterations, Algorithm 1 outputs α ( I ) ∈ A D d such that J ( α ( I ) ) ≤ inf α ∈A D d J ( α ) + ϵ .

Proof. The first step of this proof consists in finding an upper bound for any sub-gradient of α . As we said,

<!-- formula-not-decoded -->

where I α = diag ( α ) is a D × D diagonal matrix and ̂ Φ is the n × d matrix having, as rows, ̂ φ D ( x t ) for each t = 1 , . . . n . At this point note that, by duality

<!-- formula-not-decoded -->

where {} t denotes the t -th component of ̂ φ D ( x ) ⊤ I α ̂ Φ ⊤ , which is a 1 × n row vector. Now, assuming 4 that the supremum is obtained by just one value x ∗ ∈ X , we can compute the gradient as

<!-- formula-not-decoded -->

In the last line, we have used the Hadamard product ⊙ , that is defined, for two vectors of length D like ̂ φ D ( x ∗ ) ⊤ and { ̂ Φ } ⊤ t , as the component-wise product, generating another vector of length D . Now, we are going to bound the two-norm of this gradient:

<!-- formula-not-decoded -->

where the last passage holds since the features ̂ φ i ( · ) are orthonormal w.r.t. µ n ( · ) . Under these assumption, namely

1. J is convex
2. Each sub-gradient has norm bounded by G := ϑ ( n ) D, 2
3. The diameter of the optimization space H D d is R := √ D -d

4 if there are ties, the argument applied to each of them still holds bounding the norm of the sub-gradient

equation (3) on Boyd et al. [2003] guarantees that running the subgradient method for I iterations with step size

<!-- formula-not-decoded -->

(corresponding to line 7), achieves suboptimality ϵ I bounded by

<!-- formula-not-decoded -->

Therefore, a number of iterations I = 4 ϵ -2 ϑ ( n ) D, 2 2 ( D -d ) log 3 (4 ϑ ( n ) D, 2 2 ( D -d )) allows to ensure ϵ I ≤ ϵ . In this way, we have

<!-- formula-not-decoded -->

which completes the proof.

Theorem 11. Let Assumption 1 hold and fix δ &gt; 0 . Then, with probability 1 -δ ,

<!-- formula-not-decoded -->

Proof. By Theorem 7 and the definition of ˆ θ n, BWR,

<!-- formula-not-decoded -->

By Theorem 10, for fixed ϵ , we have Λ α ( I ) ,µ n ≤ min α ∈A D d Λ α ,µ n + ϵ . Moreover, note that

<!-- formula-not-decoded -->

Replacing this relation in Eq. (21) we get the result.

## D.4 Gradient method

The algorithm we use for our estimator is called Subgradient Method in the literature, and is presented in Algorithm 1.

## E Proofs of Section 5

Theorem 12. Let µ be the uniform distribution on [ -1 , 1] . There exists a constant C &gt; 0 independent of d such that, if we choose D = 2 d and φ D ( x ) = [1 , . . . x 2 d -1 ] as the augmented feature map for the target space spanned by φ d ( x ) = [1 , . . . , x d -1 ] ⊤ , we have Λ Oracle µ ≤ C.

## Algorithm 1 Subgradient Method

Require: Feature map φ , d , Number I of iterations

D ∗ D

Ensure: Sequence α ∈ A

- 1: Compute ̂ φ D from φ D via Gram-Schmidt orthogonalization

<!-- formula-not-decoded -->

- 3: Initialize α (0) ← [ ones ( d ) , zeros ( D -d )] ⊤
- 4: for ℓ = 1 to I do
- ̂ 6: Compute a subgradient g ℓ ∈ ∂J ( α ( ℓ -1) )
- 5: Compute step size γ ℓ = √ D -d φ 2 ,d √ ℓ +1
- 7: Update: α ( ℓ ) = α ( ℓ -1) -γ ℓ g ℓ
- 8: if α ( ℓ ) / ∈ A D d then

<!-- formula-not-decoded -->

- 10: end if
- 11: end for
- 12: return α ∗ = α ( I )

Proof. See Theorem 3.1 by Themistoclakis and Van Barel [2017]

Proposition 23. Fix γ &gt; 0 . Let ˆ θ n be the OLS estimator, and ˆ θ n, BWR be our estimator defined in equation (9) . There exists a function f : [ -1 , 1] → R such that, E ∞ ( f ) → 0 as d →∞ , and under Assumption 1 with µ = U ([ -1 , 1]) , the following hold with probability one:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Most of the proof of this proposition is about in building the function, that we are calling f ( · ) .

The construction of the function in this proof is going to be quite involved. The function is going to be a sum over n of terms of the form ˜ f n ( · ) . The following notation will be used

1. Let d n dimension of the basis function used at step n
2. Let a n = d -γ n , for a parameter γ &gt; 0 to be defined
3. Let h n width of the mollifier
4. Let M n ( · ) = M ( · /h n ) , where M ( · ) is the standard mollifier, that is, a nonnegative function M ( · ) ∈ C ∞ (( -1 , 1)) with integral one and compact support.
5. f n ( · ) := sgn ( φ d ( · ) ⊤ φ d ( x n )) , where x n

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We are able to prove the following lemmas:

Lemma 9. For every n ,

<!-- formula-not-decoded -->

Proof. In order to perform this proof, we need one result from mathematical analysis. In fact, call bounded variation a function X = ( -1 , 1) → R such that the following norm is bounded

<!-- formula-not-decoded -->

A well-known characterization of this space Ambrosio et al. [2000] ensures that the former norm is equivalent to

d

- 2: Define the following loss:

<!-- formula-not-decoded -->

Now, we can proceed to the proof. First, note that by definition f n is in the BV (( -1 , 1)) class with ∥ f n ∥ BV = O ( d n ) . Indeed, f n ( · ) takes only values in {-1 , +1 } , and can only jump between the two values when φ d n ( · ) ⊤ φ d n ( x n ) = 0 , which happens at most d n times, as the previous is a polynomial of degree d n . At this point, by the properties of convolution,

<!-- formula-not-decoded -->

Where we have moved the derivative in the first term. At this point, the properties of convolution allow us to say that for any pair of functions g 1 , g 2 , ∥ g 1 ∗ g 2 ∥ L 2 ≤ ∥ g 1 ∥ M ∥ g 2 ∥ L 2 . Therefore, we have

<!-- formula-not-decoded -->

At this point, note that by definition M n ( t ) ≥ 0 , its integral is one and its support is contained in ( -h n , h n ) . Therefore,

<!-- formula-not-decoded -->

so that its L 2 norm is bounded by 4 √ h n . This completes the proof.

Lemma 10. For every m ≤ n , and s &gt; 0

<!-- formula-not-decoded -->

Proof. First, let us examine the smoothness of ˜ f m . Indeed, we have, for any s &gt; 0

<!-- formula-not-decoded -->

Therefore, by Jackson's theorem, we have for any s ,

<!-- formula-not-decoded -->

Theorem 24. For any γ &lt; 1 / 4 there is f ∗ such that

- lim d ∥ f ∗ -Π d, ∞ f ∗ ∥ ∞ = 0

<!-- formula-not-decoded -->

Proof. Let

## First part

Fix ε &gt; 0 . As a n goes to zero faster than exponentially and ∥ ˜ f n ( · ) ∥ ∞ ≤ 1 , we can find n 0 such that

<!-- formula-not-decoded -->

Now, ∑ n 0 n =1 a n ˜ f n ( · ) is a finite sum of C ∞ ([ -1 , 1]) functions, so it is uniformly continuous, in particular. Therefore, by Stone-Weierstrass theorem, for sufficiently large d ,

<!-- formula-not-decoded -->

Putting the two results together, we have proved that, for sufficiently large d ,

<!-- formula-not-decoded -->

Second part Let us fix n = ℓ and consider

<!-- formula-not-decoded -->

We are going to analyze the three terms separately.

- (A) We start bounding the first term from below,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, the second inequality comes from Cauchy-Schwartz, the sequent equality from Parseval's theorem and the last comes from Lemma 9. Note that, for the polynomial basis, φ 2 ,d ℓ ≈ Λ d ℓ ,µ ≈ d ℓ , so we get

<!-- formula-not-decoded -->

- (B) This term is

<!-- formula-not-decoded -->

The last passage holds as Π d ℓ ,µ Π d ℓ , ∞ ˜ f n ( · ) = Π d ℓ , ∞ ˜ f n ( · ) . Now, we can apply Lemma 10, as n &lt; ℓ , which ensures

<!-- formula-not-decoded -->

- (C) The last term can be simply bounded due to the fact that ∥ ˜ f n ∥ ∞ ≤ 1 :

<!-- formula-not-decoded -->

Now, fix any γ &lt; 1 / 4 and take

<!-- formula-not-decoded -->

We get

<!-- formula-not-decoded -->

For term B , we have

<!-- formula-not-decoded -->

Last term:

<!-- formula-not-decoded -->

Again, this term satisfies C = O (1) , as the term exp( -1 /γ m ) in the last sum decays faster than 2 -m .

where the last passage holds as s = 2 . Therefore we get B ≤ O ( ∑ ℓ -1 n =1 a n ) = O (1) , since a n decays faster than 2 -n which already generates a convergent seqeuence.

All together, these passages prove

<!-- formula-not-decoded -->

Therefore, taking this d n sequence entails lim sup d →∞ ∥ f ∗ -Π d,µ f ∗ ∥ ∞ d 1 -γ &gt; 0 .

Proof. (of Proposition 13) . Let f = f ∗ defined before, for the specific value of γ &gt; 0 . Thanks to part one of Theorem 24 5 , assumption E ∞ ( f ) d → 0 is satisfied:

<!-- formula-not-decoded -->

Then, we prove the two theses point by point. Point one: for fixed d , Theorem 11 gives

<!-- formula-not-decoded -->

As X is [ -1 , 1] and the feature map is Lipschitz continuous, we can get rid of the |X| by a covering argument. As n →∞ , the former gives

<!-- formula-not-decoded -->

5 formally, the result holds for γ &gt; 1 / 4 but, for what we are trying to prove, the validity of the statement for γ implies its validity for every γ ′ &gt; γ , therefore we can proceed w.l.o.g.

For µ = U ([ -1 , 1]) , Theorem 12 ensured that Λ Oracle µ &lt; C, a universal constant independent on d . Therefore,

<!-- formula-not-decoded -->

Let us pass to the second thesis:

<!-- formula-not-decoded -->

This follows from the fact that, for n →∞ , φ d ( · ) ⊤ ˆ θ n, OLS → Π d,µ f ( · ) and that Theorem 24 ensures lim sup d ∥ f ∗ -Π d,µ f ∗ ∥ ∞ d 1 -γ &gt; 0 .