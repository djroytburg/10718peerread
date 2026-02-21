## A Geometric Analysis of PCA

Ayoub El Hanchi University of Toronto &amp; Vector Institute aelhan@cs.toronto.edu

Murat A. Erdogdu University of Toronto &amp; Vector Institute erdogdu@cs.toronto.edu

## Abstract

What property of the data distribution determines the excess risk of principal component analysis? In this paper, we provide a precise answer to this question. We establish a central limit theorem for the error of the principal subspace estimated by PCA, and derive the asymptotic distribution of its excess risk under the reconstruction loss. We obtain a non-asymptotic upper bound on the excess risk of PCA that recovers, in the large sample limit, our asymptotic characterization. Underlying our contributions is the following result: we prove that the negative block Rayleigh quotient, defined on the Grassmannian, is generalized self-concordant along geodesics emanating from its minimizer of maximum rotation less than π/ 4 .

## 1 Introduction

Principal Component Analysis (PCA) is a core method of machine learning and statistics, prized for its simplicity and consistently strong empirical performance across diverse tasks. In contrast, analyzing its theoretical performance is challenging: an explicit characterization of its asymptotic excess risk is known for the special case of Gaussian data [RW20], while non-asymptotic results on the excess risk are in general limited to upper bounds [Sha+05; BBZ07; Nad08; RW20].

Traditionally, two main approaches have been adopted to analyze PCA. In the first, it is treated as a plug-in estimator: the empirical covariance replaces the population one, and its principal components estimate the true ones. Matrix perturbation bounds [SS90], most famously the Davis-Kahan theorem [DK70; YWS15], are then used to control the error of PCA by the deviation of the empirical covariance from the population one. In the second approach, adopted in [Sha+05; BBZ07], PCA is viewed as an instance of empirical risk minimization, for which variants of the uniform convergence analysis apply. Unfortunately, neither of these approaches leads to provably accurate bounds.

The recent work of Reiss and Wahl [RW20] takes a different approach and analyzes the projector found by PCA directly, building on earlier work of Dauxois et al. [DPR82] who established its asymptotic normality. The excess risk bounds obtained therein are powerful enough that under Gaussian data and certain eigenvalue decay assumptions, they recover the leading term in the exact asymptotic expansion of the excess risk. In the general case, however, asymptotically tight bounds do not appear to be available in the existing literature.

In this paper, we take a different approach that allows us, among other things, to obtain such bounds: we view PCA as an M-estimator, and use tools from the theory of asymptotic statistics [Van00], and its accompanying non-asymptotic theory [e.g. OB21], to analyze its performance. From this point of view, PCA is similar to linear regression with ordinary least squares. A significant difference, and a major source of difficulty, resides in the nature of their respective search spaces: for linear regression, it is R d , whereas for PCA, it is the manifold of k -dimensional subspaces of R d - the Grassmannian [e.g. EAS98]. We build extensively on the accessible expositions in [Bou23; BZA24] to overcome this difficulty.

Chris J. Maddison University of Toronto &amp; Vector Institute cmaddis@cs.toronto.edu

Our contributions are as follows. In Theorem 1, we establish a central limit theorem for the error of the principal subspace obtained by PCA, and use this result to obtain the asymptotic distribution of its excess risk under the reconstruction loss, all under a necessary moment assumption. We then establish, in Theorem 2, a non-asymptotic upper bound on the excess risk that, in the large-sample limit, accurately recovers our asymptotic characterization. At the heart of our analysis is the following key result (Proposition 1): we prove that the reconstruction risk is generalized self-concordant - in a sense analogous to the one introduced by Bach [Bac10] in his analysis of logistic regression - when restricted to geodesics originating from its global minimizer of maximum rotation less than π/ 4 .

The rest of the paper is organized as follows. In Section 2 we formalize our problem and provide an overview of the Grassmann manifold, restricting ourselves to the objects needed to state our theorems. In Section 3 we characterize the asymptotic performance of PCA. In Section 4 we establish the self-concordance of the reconstruction risk. In Section 5 we provide a non-asymptotic bound on the error of PCA, and conclude with a discussion in Section 6. Proofs of our statements are provided in the Appendix.

## 2 Problem setup &amp; Background

The goal of linear dimensionality reduction is to project high-dimensional data onto a lower dimensional subspace while preserving as much information about the original data as possible. Specifically, given i.i.d. data points ( X i ) n i =1 in R d and a choice of dimension k ∈ [ d ] , PCA finds an orthogonal projector UU T ∈ R d × d onto a k -dimensional subspace of R d such that the following empirical reconstruction error is as small as possible

<!-- formula-not-decoded -->

Here U ∈ R d × k is a matrix whose columns form an orthonormal basis of the aforementioned subspace. Denote by Σ n := n -1 ∑ n i =1 X i X T i the empirical covariance matrix, and fix an orthonormal basis ( u n,j ) d j =1 of eigenvectors of Σ n , ordered non-increasingly according to their eigenvalues, with ties broken arbitrarily. The d × k matrix U n whose j -th column is u n,j is a minimizer of (1).

Typically, however, we care about the population reconstruction error of this projector. If X is a random vector with the same distribution as that of the data points, this error is given by

<!-- formula-not-decoded -->

Denote by Σ := E[ XX T ] the population covariance matrix, and fix an orthonormal basis ( u j ) d j =1 of eigenvectors of Σ , ordered non-increasingly according to their eigenvalues ( λ j ) d j =1 , with ties broken arbitrarily. The d × k matrix U ∗ whose j -th column is u j is a minimizer of (2).

A redundancy in the parametrization. The analysis of PCA is complicated by the fact that the map U ↦→ UU T is a redundant parametrization of the set of orthogonal projectors. Fortunately, this redundancy is well-structured: two such matrices U, V represent the same projector (i.e. UU T = V V T ) if and only if there exists an orthogonal matrix Q ∈ R k × k such that V = UQ . This defines an equivalence relation ∼ on the set St( d, k ) := { U ∈ R d × k | U T U = I k } . The space of equivalence classes under this relation is known as the Grassmann manifold Gr( d, k ) := St( d, k ) / ∼ . We denote a generic element in this space by [ U ] .

To gain some intuition about this abstract set, note that every element of Gr( d, k ) can be identified with a k -dimensional subspace of R d through the map that sends [ U ] to the column space of U . Therefore for the sake of intuition we can think of [ U ] as the column space of U . In the special case of Gr(3 , 2) , which we will use as a running example, we can visualize [ U ] as a plane passing through the origin embedded in 3 dimensions.

Equipped with this new space, we define our final population and empirical risks by R ([ U ]) := ˜ R ( U ) and R n ([ U ]) := ˜ R n ( U ) , and note that, with the definitions of U n and U ∗ given above,

<!-- formula-not-decoded -->

To be precise, we will call PCA the abstract procedure that takes as input ( X i ) n i =1 and outputs the subspace [ U n ] , even though in practice it returns the representative U n . In this paper, our goal is to understand the performance of this procedure in terms of the distribution of X , as measured both by how close [ U n ] is to [ U ∗ ] , and by the excess risk R ([ U n ]) -R ([ U ∗ ]) .

## 2.1 Background on the Grassmann manifold

Our analysis takes place on the Grassmannian Gr( d, k ) , which admits the structure of a Riemannian manifold. In this section, we give a high-level description of the objects needed to state our results. For a more rigorous yet accessible introduction, see for example [BZA24] and [Bou23, Chapter 9]. Throughout, we let [ U ] , [ V ] ∈ Gr( d, k ) and let U, V ∈ St( d, k ) be corresponding representatives.

Tangent space. The tangent space of Gr( d, k ) at [ U ] , denoted by T [ U ] Gr( d, k ) , is a vector space of dimension k ( d -k ) . Our results are stated in terms of a more concrete, yet equivalent, set

<!-- formula-not-decoded -->

For our purposes, H U is easier to work with and there exists a canonical, invertible, linear map 1 lift U : T [ U ] Gr( d, k ) → H U that lifts tangent vectors ξ ∈ T [ U ] Gr( d, k ) into H U . The elements of H U can be thought of concretely as 'velocities' in the sense that the basis U with 'velocity' lift U ( ξ ) moves infinitesimally to ' U + ϵ lift U ( ξ ) ' while remaining an orthonormal basis. By requiring that lift U ( ξ ) has columns in the orthogonal complement of U , we are guaranteed that the subspace [ U ] changes when U moves in the direction of lift U ( ξ ) .

The tangent space is equipped with an inner product ⟨· , ·⟩ [ U ] known as the Riemannian metric at [ U ] . To calculate ⟨· , ·⟩ [ U ] , we lift tangent vectors into H U and apply the Frobenius inner product,

<!-- formula-not-decoded -->

for all ξ 1 , ξ 2 ∈ T [ U ] Gr( d, k ) .

Geodesics and the exponential map. Let ξ ∈ T [ U ] Gr( d, k ) . The geodesic starting at [ U ] in the direction ξ is the curve γ : [0 , 1] → Gr( d, k ) with γ (0) = [ U ] of constant velocity ξ . Intuitively, one may think of γ ( t ) as the straight line " [ U ] + tξ ", in the sense that γ is a curve with zero acceleration, properly defined. γ ( t ) can be calculated with the SVD lift U ( ξ ) = PSQ T 2 via the identity

<!-- formula-not-decoded -->

The exponential map at [ U ] in the direction ξ is then defined by Exp [ U ] ( ξ ) = γ (1) . Returning to our running example, Gr(3 , 2) , these geodesics correspond to "constant velocity" rotations of the plane [ U ] along the pitch or roll axes, as specified by the directions of the columns of lift U ( ξ ) , and the extent of rotation between [ U ] and Exp [ U ] ( ξ ) depends on the magnitudes of the columns of lift U ( ξ ) .

Principal angles and Riemannian distance. The j -th principal angle θ j ([ U ] , [ V ]) ∈ [0 , π/ 2] is defined by cos( θ j ([ U ] , [ V ])) = s j where s j is the j -th largest singular value of U T V . These angles generalize the notion of angles between lines to angles between subspaces, and they measure the magnitude of the most efficient rotation that aligns [ U ] with [ V ] . For Gr( d, k ) , the principal angles give us an explicit expression for the Riemannian distance between [ U ] and [ V ] , which, properly defined, is the length of a shortest curve connecting [ U ] and [ V ] ,

<!-- formula-not-decoded -->

Logarithmic map. Where well-defined, the logarithmic map at [ U ] evaluated at [ V ] is the inverse of the exponential map, i.e. , Exp [ U ] (Log [ U ] ([ V ])) = [ V ] . It can be thought of as " [ V ] -[ U ] ". For Gr( d, k ) , the logarithmic map can be calculated by lift U (Log [ U ] ([ V ])) = ( P arctan( S ) Q T ) where ( I -UU T ) V ( U T V ) -1 = PSQ T is a SVD. This map is only well-defined for θ k ([ U ] , [ V ]) &lt; π/ 2 . The singular values of lift U (Log [ U ] ([ V ])) are the principal angles, and hence the following holds ∥ lift U (Log [ U ] ([ V ])) ∥ F = dist([ U ] , [ V ]) , see also [AV24]. In Gr(3 , 2) , the logarithmic map at [ U ] evaluated at [ V ] gives us the most efficient rotation that transforms the plane [ U ] into [ V ] .

1 The inverse of the differential of the quotient map U ↦→ [ U ] at U restricted to H U .

2 P ∈ R d × k , S ∈ R k × k , Q ∈ R k × k .

## 3 Asymptotic characterization

Before stating our first main result, we briefly recap our notation. The empirical and population covariance matrices are denoted by Σ n = n -1 ∑ n i =1 X i X T i and Σ = E[ XX T ] , ( u n,j ) is an orthonormal basis of eigenvectors of Σ n ordered non-increasingly according their corresponding eigenvalues, and ( u j ) is an orthonormal basis of eigenvectors of Σ ordered non-increasingly according to their corresponding eigenvalues ( λ j ) . U n ∈ St( d, k ) is the matrix whose j -th column is u n,j , and corresponds to the output of PCA, while U ∗ ∈ St( d, k ) is the matrix whose j -th column is u j .

We further define U ⊥ ∗ ∈ St( d, d -k ) to be the matrix whose i -th column is u k + i . It is easy to verify that the map Γ ↦→ U ⊥ ∗ Γ for ( d -k ) × k matrices Γ is linear and its image is H U ∗ as defined in (4). It is invertible and preserves the Frobenius inner product, so H U ∗ can be identified with R ( d -k ) × k through it. Finally, recall that the logarithm at [ U ∗ ] of [ U n ] is only well-defined when all the principal angles between them are strictly less than π/ 2 . In what follows, this logarithm can be defined arbitrarily when this condition fails - the validity of the statement is unaffected by this choice.

The following is the first main result of the paper.

Theorem 1. Assume that λ k &gt; λ k +1 , E[ ∥ X ∥ 2 2 ] is finite, and for all i, s ∈ [ d -k ] and j, t ∈ [ k ] ,

<!-- formula-not-decoded -->

is finite. Define δ ij := λ j -λ k + i . Then as n →∞ , the following holds.

- Consistency:

<!-- formula-not-decoded -->

for all ε &gt; 0 , where dist is the Riemannian distance given by (7).

- Asymptotic normality:

<!-- formula-not-decoded -->

where G is a mean zero ( d -k ) × k Gaussian matrix with E[ G ij G st ] = Λ ijst /δ ij δ st .

- Excess risk:

<!-- formula-not-decoded -->

where H is a mean zero ( d -k ) × k Gaussian matrix with E[ H ij H st ] = Λ ijst / √ δ ij δ st .

Under an eigengap and moment condition, Theorem 1 characterizes the performance of PCA in the large sample limit. The consistency statement says that, with enough data, the principal subspace found by PCA gets arbitrarily close to the true one under the Riemannian distance (7) with overwhelming probability. The asymptotic normality result refines this statement: it says that the fluctuations of PCA around the true principal subspace are asymptotically normal with the prescribed covariance structure. Finally, the last statement expresses the asymptotic distribution of the excess risk as the squared Frobenius norm of a Gaussian matrix.

Relationship with existing work and assumptions. The consistency result is a direct consequence of the Davis-Kahan theorem [DK70]. To the best of our knowledge, the asymptotic normality result in Theorem 1 is new. The finiteness of (8) is necessary, in the same way that finite variance is for the classical central limit theorem. Tripuraneni et al. [Tri+18] obtained a similar expression for the asymptotic variance of averaged Riemannian SGD on PCA, albeit under an unverified assumption. Dauxois et al. [DPR82] established the asymptotic normality of the Euclidean fluctuations of empirical projectors and eigenvectors - our result may be viewed as a Riemannian analogue of theirs. Under the eigengap condition, the excess risk bound in Theorem 1 extends the result of Reiss and Wahl [RW20, Proposition 2.14]. While their statement is restricted to Gaussian data, the underlying argument carries over directly to any distribution with finite fourth moments. Theorem 1 strengthens this result further by requiring only the finiteness of (8). A detailed discussion of the eigengap condition is deferred to Section 6; for now, we simply note that it is a mild assumption.

Our main interest is in the excess risk, as it directly measures how well PCA performs on the reconstruction task. Corollary 1 below offers a more interpretable version of the result in Theorem 1, and serves as a benchmark for our non-asymptotic analysis. To motivate it, we briefly digress.

Performance guarantees in machine learning are typically stated as: with probability at least 1 -δ , the excess risk is at most some quantity. While intuitive, the accuracy of such statements is hard to quantify: what is a "high-probability lower bound"? This ambiguity can be avoided by interpreting such statements as upper bounds on the 1 -δ quantile of the excess risk. Specifically, recall that for a random variable Z , its 1 -δ quantile, for δ ∈ [0 , 1] , is defined by

<!-- formula-not-decoded -->

In words, this quantile describes the best upper bound on the random variable Z that holds with probability at least 1 -δ . We may then make sense of a "high-probability lower bound" on the excess risk as a lower bound on its 1 -δ quantile. The following corollary, a simple consequence of Gaussian concentration, gives matching upper and lower bounds on the asymptotic quantiles of the excess risk.

Corollary 1. In the setting of Theorem 1, and for all δ ∈ [0 , 0 . 1)

<!-- formula-not-decoded -->

where E n = R ([ U n ]) -R ([ U ∗ ]) is the excess risk of PCA, τ 2 = sup ∥ A ∥ F =1 E[ ⟨ A,H ⟩ 2 F ] , and a ≍ b means that cb ≤ a ≤ Cb for some C, c &gt; 0 . Here C = 1 and c = 1 / 64 are valid choices.

To interpret this statement, it will be useful to introduce the following definition. For a random matrix W ∈ R d × k , we define its covariance operator to be the linear map Cov( W ) : R d × k → R d × k given by Cov( W )[ A ] := E[ ⟨ W,A ⟩ F W ] . It is positive semi-definite, i.e. ⟨ A, Cov( W )[ A ] ⟩ F ≥ 0 for all A , and so has non-negative eigenvalues, whose sum is the trace of Cov( W ) , which equals E[ ∥ W ∥ 2 F ] .

Corollary 1 then says that for sufficiently small failure probability δ , the 1 -δ asymptotic quantile of the excess risk is equivalent, up to explicit constants, to a sum of two terms. The first is E[ ∥ H ∥ 2 F ] which equals the trace of Cov( H ) - that is the sum of its eigenvalues. It admits an explicit expression in terms of (i) a second-order covariance between pairs of projections of X onto a top k and a bottom d -k eigenvector of Σ , (ii) the eigenvalue gaps between a top k and a bottom d -k eigenvalue of Σ . The second term is the product of the largest eigenvalue of Cov( H ) and log(1 /δ ) . In typical regimes where δ is moderately small, this second term is much smaller than the first. Returning to the question we raised in the abstract, Corollary 1 thus identifies the first term as the key property of the distribution of X that determines the excess risk of PCA, at least in the large sample regime.

Our goal in the next sections will be to derive a non-asymptotic upper bound on the 1 -δ quantile of the excess risk that matches its expression from Corollary 1. We conclude this section by offering two remarks highlighting other aspects of Theorem 1, accompanied by an example illustrating our result on the spiked covariance model.

Remark 1 (Empirical projectors) . In some applications it is of more interest to measure the performance of PCA through the closeness of the empirical projector U n U T n to the population one U ∗ U T ∗ in a given norm. Dauxois et al. [DPR82] derive the exact asymptotic distribution of √ n ( U n U T n -U ∗ U T ∗ ) from which the result we are about to discuss can potentially be deduced. Here we would like to point out that the asymptotic normality result of Theorem 1 can also be used to establish that as n →∞ ,

<!-- formula-not-decoded -->

for all p ∈ [1 , ∞ ] , where G is the Gaussian matrix defined in Theorem 1 and ∥ G ∥ p is the Schattenp norm of G , i.e. the p -norm of its singular values. Compare with equation (2.22) in [RW20].

The case p = 2 corresponds to the Frobenius norm ∥ U n U n -U ∗ U T ∗ ∥ F , and a statement analogous to Corollary 1 holds. Specifically, for δ ∈ [0 , 0 . 1) it holds that

<!-- formula-not-decoded -->

where P n := ∥ U n U n -U ∗ U T ∗ ∥ 2 F , τ 2 = sup ∥ A ∥ F =1 E[ ⟨ A,G ⟩ 2 F ] , and the constants are the same as in Corollary 1. A similar result can be obtained for the other values of p using the noncommutative Khintchine inequality [Tro+15; Van17], though the upper and lower bounds differ by a logarithmic factor in the dimension d for large values of p .

Example 1 (Spiked covariance model) . As an application of Theorem 1, we consider the spiked covariance model [Joh01; Nad08]. Specifically, we assume that X = Z + ε such that Z and ε are independent, ε ∼ N (0 , σ 2 I d ) , and S = E[ ZZ T ] has rank k . Then Σ = S + I , the support of Z is contained in the k -dimensional subspace spanned by ( u j ) k j =1 , and λ j = η j + σ 2 where ( η j ) are the eigenvalues of S ordered non-increasingly. Taking ξ j := ⟨ Z, u j ⟩ we recover the standard form

<!-- formula-not-decoded -->

This spiked covariance model captures the scenario where we observe a noisy version X of the true lower dimensional data point Z corrupted with isotropic noise ε . The goal is to recover, using PCA, the subspace span( { u j | j ∈ [ k ] } ) on which the noise-free data Z is supported. In this setting, Theorem 1 simplifies significantly. Specifically, under this model, the Gaussian matrices G and H in the statements (9) and (10) have independent entries with variances

<!-- formula-not-decoded -->

which are constant along rows. From Remark 1 and Theorem 1, we have the distributional results

<!-- formula-not-decoded -->

as n →∞ . The asymptotic quantiles of P n = ∥ U n U T n -U ∗ U T ∗ ∥ 2 F have the equivalent expression

<!-- formula-not-decoded -->

while those of the excess risk E n = R ([ U n ]) -R ([ U ∗ ]) have the equivalent expression

<!-- formula-not-decoded -->

both for δ ∈ [0 , 0 . 1) and the same constants as in Corollary 1. We conclude this example by noting that, using a recent result of Latała et al. [LHY18] and leveraging the independence of the entries of G , an analogue of (11) can be derived for large values of p without suffering from the inefficiency of the noncommutative Khintchine inequality highlighted at the end of Remark 1.

Remark 2 (Generalized PCA) . While our results are framed for PCA, we remark here that they apply to the more general problem of estimating the leading k -dimensional eigenspace of a symmetric matrix. Specifically, let ( A i ) n i =1 be i.i.d. realizations of a random symmetric matrix A , and suppose that we are interested in estimating the leading k -dimensional eigenspace of M := E[ A ] . A natural and common procedure is to estimate it using the leading k -dimensional eigenspace of M n := n -1 ∑ n i =1 A i . While the reconstruction loss does not make sense for this problem, we may still cast this procedure as an instance of ERM where the loss is given by the negative block Rayleigh quotient. The population and empirical risks are then given by

<!-- formula-not-decoded -->

PCA then corresponds to the special case A = XX T where X is a random vector, and the population and empirical reconstruction risks are, up to additive constants, equal to those in (12). Theorem 1 applies almost verbatim to this generic setting, with the only change being that (8) is generalized to

<!-- formula-not-decoded -->

As an example different from PCA, consider the case where M is the adjacency matrix of an undirected weighted graph with non-negative weights. Suppose that we observe n i.i.d. edges { J i , K i } n i =1 of the graph, sampled from the distribution on the edges that is proportional to their weights. Then one may take A i = e J i e T K i + e K i e T J i , and Theorem 1 with (8) replaced by (13) applies. A similar argument can be made for the estimation of the trailing k -dimensional eigenspace of the Laplacian matrix. As examples of potential applications, we mention spectral clustering [e.g. NJW01], community detection [e.g. Abb18], and contrastive learning [e.g. Hao+21].

## 4 Self-concordance of the block Rayleigh quotient

The main ingredient behind Theorem 1 is a Taylor expansion of the population and excess risks at [ U n ] around [ U ∗ ] , which becomes exact in the large sample limit. In order to make Corollary 1 nonasymptotic, an explicit control of the error in these expansions in a reasonably large neighbourhood of [ U ∗ ] is what is needed. In this section, we show that the population and empirical reconstruction risks are geodesically generalized self-concordant, in a sense analogous to the one introduced by Bach [Bac10]. This provides the needed control for the non-asymptotic analysis. As this self-concordance result can potentially be of broader interest, we frame it here in more general terms.

Let A be a d × d symmetric matrix, and let ( v j ) be a basis of eigenvectors of A ordered nonincreasingly in terms of their eigenvalues ( µ j ) . Recall that eigenvectors corresponding to the largest eigenvalue are maximizers of the Rayleigh quotient. The following known construction is a generalization of this familiar identity. Let F : Gr( d, k ) → R be given by

<!-- formula-not-decoded -->

To see how F relates to the reconstruction risk, note that it can be expressed in terms of it

<!-- formula-not-decoded -->

The trace expression in (14) is known as the block Rayleigh quotient of A . Let k 1 and k 2 be the smallest and largest indices i such that µ i = µ k respectively. The set of minimizers of F is given by (see for example [Tao12, Proposition 1.3.4])

<!-- formula-not-decoded -->

where ⊕ is the direct sum of subspaces, and S is a subspace of dimension k -k 1 +1 . In the case where µ k &gt; µ k +1 that we have been operating under, k 1 = k 2 = k and V ∗ becomes a singleton.

Recall the definition of geodesics and principal angles from Section 2.1. The following is the main result of this section [c.f. Bac10, Lemma 1].

<!-- formula-not-decoded -->

for all t ∈ [0 , 1] where we shortened θ k ([ V ∗ ] , [ V ]) to θ . As a consequence,

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

satisfies ψ ( θ ) → 1 as θ → 0 and ψ ( θ ) → c as θ → π/ 4 for c ≈ 1 . 485 .

In words, Proposition 1 says that for any [ V ] ∈ Gr( d, k ) that is less than π/ 4 away from the minimizer of F in maximum principal angle, the restriction of F to the geodesic connecting [ V ] to this minimizer is well approximated by its second order Taylor expansion, up to a factor of approximately 2 in the second term of this expansion. We have the immediate corollary [cf. Bac10, Proposition 1].

Corollary 2. In the setting of Proposition 1, the following estimates hold.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ξ = Log [ V ∗ ] ([ V ]) , and where for any [ U ] ∈ Gr( d, k ) and ζ ∈ T [ U ] Gr( d, k ) with ∆ = lift U ( ζ ) lift U (grad F ([ U ])) = -( I -UU T ) AU, lift U (Hess F ([ U ])[ ζ ]) = ∆ U T AU -( I -UU T ) A ∆ . The statement remains true when [ V ] and [ V ∗ ] are interchanged in the above inequalities.

Related results include those of [ZJS16] who showed that F satisfies a version of the PolyakŁojasiewicz inequality for k = 1 , and [AV24] who showed that F , when restricted as in Proposition 1, satisfies a version of strong convexity - Proposition 1 was inspired by the latter work. The result of the next section builds on Corollary 2 to provide a non-asymptotic analogue of Corollary 1.

## 5 Non-asymptotic bound

The following is the second main result of the paper. The parameters V and ν appearing in it are defined in Remark 3 below. We compute them explicitly under a Gaussian model in Example 2.

Theorem 2. Assume that λ k +1 &gt; λ k , E[ X 4 j ] &lt; ∞ for all j ∈ [ d ] , and let δ ∈ [0 , 1) . If

<!-- formula-not-decoded -->

then with probability at least 1 -δ

<!-- formula-not-decoded -->

where for c ( d ) = 4(1 + 2 ⌈ log( d ) ⌉ ) ,

<!-- formula-not-decoded -->

Focusing on (17), Theorem 2 says that, up to a worse dependence on δ , the tight asymptotic upper bound on the 1 -δ quantile of the excess risk in Corollary 1 holds true for finitely many samples, provided that the sample size is larger than a certain distribution-dependent constant. Given the weak moment condition assumed, the dependence on δ in Theorem 2 is likely unimprovable. It is possible to obtain a log(1 /δ ) dependence as in Corollary 1 under the assumption that X is bounded. We favour the above statement as it highlights an important shortcoming of ERM, and thus PCA: its performance degrades under heavy-tailed data - we refer the interested reader to the literature on robust estimation [e.g. LM19]. The results in [RW20] are the most closely related, though they do not capture the right dependence on the distribution of X identified in Corollary 1 and recovered in (17). They however hold under different assumptions and can cover a wider range of sample sizes.

The sample size restriction (16) consists of three terms. They arise from two distinct steps of the analysis: a global and a local one. The global one ensures that with high probability, [ U n ] is within a maximum principal angle of π/ 4 from [ U ∗ ] . This step is carried out using standard existing tools namely the Davis-Kahan theorem [e.g. YWS15] - and is likely loose. It results in the third term of (16), the largest of the three. The second step is a local analysis that uses our new self-concordance result from Proposition 1, and is where our original contribution lies. This step results in the first two terms of (16), the first of which typically dominates: their role is to ensure that the curvature of the empirical risk at [ U ∗ ] is strong enough to force [ U n ] to be near it. Qualitatively, the explicit expression of V and ν in Remark 3 below indicate that they induce a quadratic dependence on the inverse of the eigengap on the sample size restriction. Example 2 below gives an easily interpretable expression for V and ν in the special case when X is Gaussian and centered.

Remark 3 (Variance parameters) . The parameters V and ν appearing in Theorem 2 admit an explicit expression, though it is quite involved in the general case. Recall from Theorem 1 the definition of the eigengaps δ ij = λ j -λ k + i . Let ˜ X denote the coordinates of X in the basis of eigenvectors ( u j ) , i.e. ˜ X j := ⟨ X,u j ⟩ . Define and recall from Theorem 1

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the coefficients are given by

<!-- formula-not-decoded -->

Finally, define the parameter

<!-- formula-not-decoded -->

where the coefficients are given by, suppressing the dependence on M

<!-- formula-not-decoded -->

Typically, V is much larger than ν , and we always have V ≥ ν .

Example 2 (Gaussian model) . The variance parameters appearing in Theorem 2 and defined in Remark 3 simplify greatly when X has mean zero and its coordinates in a basis of eigenvectors of Σ are independent. When X ∼ N (0 , Σ) we can compute them exactly in terms of the spectrum of Σ :

<!-- formula-not-decoded -->

## 6 Discussion

We started this paper with a simple question: which property of the distribution of X governs the performance of PCA as measured by its excess risk? In the large sample limit, and under very mild assumptions, we found an equally simple answer (see (10) and Corollary 1):

<!-- formula-not-decoded -->

Our second main contribution was to derive an upper bound on the critical sample size - that beyond which the excess risk of PCA is governed by (22) (see (16)). In the general case, this bound admits an explicit expression in terms of the fourth moments of X and the eigengaps appearing in (22) (see Remark 3), though its precise description is quite involved. In the special case of Gaussian X , we showed that the terms in this bound take on an exceptionally simple form (see Example 2).

There are two main limitations of our results. The first is that they rely on the eigengap condition λ k +1 -λ k &gt; 0 . While this is a mild assumption, it would be desirable to relax it, though this is quite challenging with our approach. To see why, note that without it, the minimizers of the reconstruction risk form a submanifold (itself a Grassmannian) of Gr( d, k ) (see (15)). The classical theory of asymptotic statistics, upon which our results rely, does not immediately apply in such a degenerate setting [Van+96; Van00], and we leave this problem to future work.

The second limitation we would like to point out is related to Theorem 2. As described after its statement, the global component of the analysis leading to it is unlikely to be tight. To accurately capture the sample complexity of this step, we suspect that one would need to leverage analytical properties of the reconstruction risk as we do in our local analysis. This is however quite challenging as globally the reconstruction risk is ill-behaved and has for example many critical points [SI14]. We anticipate that new insights are needed to fully capture the sample complexity of this global step.

Finally, let us mention that while the setting we consider is both classical and quite general, there are potentially interesting cases that our framework does not cover. For example, our approach does not directly apply to Kernel PCA [SSM98] or functional PCA [RS02], and extending it to these settings would require us to work with an infinite-dimensional analogue of the Grassmannian - a daunting task. Similarly our focus is on characterizing the performance of PCA on a fixed but unknown data distribution, and to do this in as much generality as possible. This in contrast with the literature on high-dimensional PCA which typically considers sequences of Gaussian problems indexed by their dimension, but provides potentially finer-grained results [e.g. Joh01; Pau07; CMW13]. Finally, we mention that in practice, PCA is typically performed with an initial centering step. This neatly fits in our setting by changing the search space from the Grassmannian to the Grassmannian of affine subspaces [LWY21], though some work is required to make this approach viable.

Beyond addressing the limitations discussed above, there are a few directions that are potentially worth exploring. From an analysis standpoint, the output of PCA - along with its generalized version described in Remark 2 - is often used as a preprocessing step for downstream tasks such as regression or classification. It would be interesting to investigate whether the techniques developed in this paper can be extended to provide end-to-end guarantees for such two-stage procedures. Another intriguing direction would be to explore whether the self-concordance result established in Proposition 1 can be leveraged to obtain improved convergence guarantees for optimization algorithms applied to the block Rayleigh quotient (14), particularly for Edelman et al. [EAS98]'s version of Newton's method.

## References

- [Abb18] E. Abbe. 'Community detection and stochastic block models: recent developments'. In: Journal of Machine Learning Research 18.177 (2018), pp. 1-86.
- [AV24] F. Alimisis and B. Vandereycken. 'Geodesic convexity of the symmetric eigenvalue problem and convergence of steepest descent'. In: Journal of Optimization Theory and Applications 203.1 (2024), pp. 920-959.
- [Bac10] F. Bach. 'Self-concordant analysis for logistic regression'. In: (2010).
- [BBZ07] G. Blanchard, O. Bousquet, and L. Zwald. 'Statistical properties of kernel principal component analysis'. In: Machine Learning 66 (2007), pp. 259-294.
- [BLM13] S. Boucheron, G. Lugosi, and P. Massart. Concentration Inequalities: A Nonasymptotic Theory of Independence . Oxford University Press, Feb. 2013.
- [Bou02] O. Bousquet. 'A Bennett concentration inequality and its application to suprema of empirical processes'. In: Comptes Rendus Mathematique 334.6 (2002), pp. 495-500.
- [Bou23] N. Boumal. An introduction to optimization on smooth manifolds . Cambridge University Press, 2023.
- [BZA24] T. Bendokat, R. Zimmermann, and P.-A. Absil. 'A Grassmann manifold handbook: Basic geometry and computational aspects'. In: Advances in Computational Mathematics 50.1 (2024), p. 6.
- [CMW13] T. T. Cai, Z. Ma, and Y. Wu. 'Sparse PCA: Optimal rates and adaptive estimation'. In: (2013).
- [DK70] C. Davis and W. M. Kahan. 'The rotation of eigenvectors by a perturbation. III'. In: SIAM Journal on Numerical Analysis 7.1 (1970), pp. 1-46.
- [DPR82] J. Dauxois, A. Pousse, and Y. Romain. 'Asymptotic theory for the principal component analysis of a vector random function: some applications to statistical inference'. In: Journal of multivariate analysis 12.1 (1982), pp. 136-154.
- [EAS98] A. Edelman, T. A. Arias, and S. T. Smith. 'The geometry of algorithms with orthogonality constraints'. In: SIAM journal on Matrix Analysis and Applications 20.2 (1998), pp. 303-353.
- [EME24] A. El Hanchi, C. Maddison, and M. Erdogdu. 'Minimax Linear Regression under the Quantile Risk'. In: The Thirty Seventh Annual Conference on Learning Theory . PMLR. 2024, pp. 1516-1572.
- [Hao+21] J. Z. HaoChen, C. Wei, A. Gaidon, and T. Ma. 'Provable guarantees for self-supervised deep learning with spectral contrastive loss'. In: Advances in neural information processing systems 34 (2021), pp. 5000-5011.
- [Joh01] I. M. Johnstone. 'On the distribution of the largest eigenvalue in principal components analysis'. In: The Annals of statistics 29.2 (2001), pp. 295-327.
- [LHY18] R. Latała, R. van Handel, and P. Youssef. 'The dimension-free structure of nonhomogeneous random matrices'. In: Inventiones mathematicae 214 (2018), pp. 10311080.
- [LM19] G. Lugosi and S. Mendelson. 'Mean estimation and regression under heavy-tailed distributions: A survey'. In: Foundations of Computational Mathematics 19.5 (2019), pp. 1145-1190.
- [LWY21] L.-H. Lim, K. S.-W. Wong, and K. Ye. 'The Grassmannian of affine subspaces'. In: Foundations of Computational Mathematics 21 (2021), pp. 537-574.
- [Nad08] B. Nadler. 'Finite sample approximation results for principal component analysis: A matrix perturbation approach'. In: (2008).
- [NJW01] A. Ng, M. Jordan, and Y. Weiss. 'On spectral clustering: Analysis and an algorithm'. In: Advances in neural information processing systems 14 (2001).
- [OB21] D. M. Ostrovskii and F. Bach. 'Finite-sample analysis of m-estimators using selfconcordance'. In: (2021).
- [Pau07] D. Paul. 'Asymptotics of sample eigenstructure for a large dimensional spiked covariance model'. In: Statistica Sinica (2007), pp. 1617-1642.
- [RS02] J. O. Ramsay and B. W. Silverman. Applied functional data analysis: methods and case studies . Springer, 2002.

- [RW20] M. Reiss and M. Wahl. 'Nonasymptotic upper bounds for the reconstruction error of PCA'. In: The Annals of Statistics 48.2 (2020), pp. 1098-1123.
- [Sha+05] J. Shawe-Taylor, C. K. Williams, N. Cristianini, and J. Kandola. 'On the eigenspectrum of the Gram matrix and the generalization error of kernel-PCA'. In: IEEE Transactions on Information Theory 51.7 (2005), pp. 2510-2522.
- [SI14] H. Sato and T. Iwai. 'Optimization algorithms on the Grassmann manifold with application to matrix eigenvalue problems'. In: Japan Journal of Industrial and Applied Mathematics 31 (2014), pp. 355-400.
- [SS90] G. Stewart and J. Sun. Matrix Perturbation Theory . Computer Science and Scientific Computing. Elsevier Science, 1990. ISBN: 9780126702309.
- [SSM98] B. Schölkopf, A. Smola, and K.-R. Müller. 'Nonlinear component analysis as a kernel eigenvalue problem'. In: Neural computation 10.5 (1998), pp. 1299-1319.
- [Tao12] T. Tao. Topics in random matrix theory . Vol. 132. American Mathematical Soc., 2012.
- [Tri+18] N. Tripuraneni, N. Flammarion, F. Bach, and M. I. Jordan. 'Averaging stochastic gradient descent on Riemannian manifolds'. In: Conference On Learning Theory . PMLR. 2018, pp. 650-687.
- [Tro+15] J. A. Tropp et al. 'An introduction to matrix concentration inequalities'. In: Foundations and Trends® in Machine Learning 8.1-2 (2015), pp. 1-230.
- [Tro16] J. A. Tropp. 'The expected norm of a sum of independent random matrices: An elementary approach'. In: High Dimensional Probability VII: The Cargese Volume . Springer. 2016, pp. 173-202.
- [Van+16] S. A. Van de Geer et al. Estimation and testing under sparsity . Springer, 2016.
- [Van+96] A. W. Van Der Vaart, J. A. Wellner, A. W. van der Vaart, and J. A. Wellner. Weak convergence . Springer, 1996.
- [Van00] A. W. Van der Vaart. Asymptotic statistics . Vol. 3. Cambridge university press, 2000.
- [Van17] R. Van Handel. 'Structured random matrices'. In: Convexity and concentration (2017), pp. 107-156.
- [YWS15] Y. Yu, T. Wang, and R. J. Samworth. 'A useful variant of the Davis-Kahan theorem for statisticians'. In: Biometrika 102.2 (2015), pp. 315-323.
- [ZJS16] H. Zhang, S. J Reddi, and S. Sra. 'Riemannian SVRG: Fast stochastic optimization on Riemannian manifolds'. In: Advances in Neural Information Processing Systems 29 (2016).

## A Further background on the Grassmannian

In Section 2, we focused on the properties of the Grassmannian necessary to state our results. For the analysis we require more tools, which we briefly describe here. As before, we adopt a computationoriented description, and refer the reader to Chapter 10 in [Bou23] for a rigorous treatment.

Parallel transport. Let [ U ] ∈ Gr( d, k ) and ξ ∈ T [ U ] Gr( d, k ) , and consider the curve,

<!-- formula-not-decoded -->

for t ∈ [0 , 1] , and where lift U ( ξ ) = PSQ T is a SVD. This is the geodesic starting at [ U ] in direction ξ as defined in (6) at the level of representatives in St( d, k ) , i.e. [ α ( t )] = Exp [ U ] ( tξ ) . For each t ∈ [0 , 1] and ζ ∈ T [ U ] Gr( d, k ) , define the map P ξ,t by

<!-- formula-not-decoded -->

This is the parallel transport map along the geodesic [ α ( t )] . See equation (3.18) in [BZA24]. For any fixed t , ζ ↦→ P ξ,t ( ζ ) is an invertible linear map that preserves the inner product between T [ U ] Gr( d, k ) and T [ α ( t )] Gr( d, k ) . Informally, t ↦→ P ξ,t ( ζ ) transports ζ along the tangent spaces of [ α ( t )] such that it stays "constant", i.e. the derivative of t ↦→ P ξ,t ( ζ ) , properly defined, is zero.

For the rest of this section, let ˜ f : R d × k → R be such that ˜ f ( U ) = ˜ f ( UQ ) for any orthogonal Q ∈ R k × k and U ∈ St( d, k ) . We define f : Gr( d, k ) → R by f ([ U ]) := ˜ f ( U ) . We assume that f is sufficiently smooth, under the proper notion of smoothness, to justify the computations below.

Gradients and Hessians. The Riemannian gradient of f at [ U ] is the element of T [ U ] Gr( d, k ) given by

<!-- formula-not-decoded -->

where ∇ ˜ f is the Euclidean gradient of ˜ f . See equation (9.84) in [Bou23]. Similarly, the Riemannian Hessian of f is the linear map Hess f ([ U ]) : T [ U ] Gr( d, k ) → T [ U ] Gr( d, k ) given by

<!-- formula-not-decoded -->

for any ξ ∈ T [ U ] Gr( d, k ) , and where ∇ 2 ˜ f is the Euclidean Hessian of ˜ f , viewed as a linear map R d × k → R d × k . See equation (9.86) in [Bou23].

Higher order derivatives. The total s -th order covariant derivative of f is denoted by ∇ m f . See Definition 10.77 and Example 10.78 in [Bou23]. It is a map that takes an m +1 tuple ([ U ] , ξ 1 , . . . , ξ m ) where [ U ] ∈ Gr( d, k ) and ξ j ∈ T [ U ] Gr( d, k ) for all j ∈ [ m ] and outputs a real number. It is linear in each of its last m arguments, and can be computed iteratively as follows: ∇ 0 f = f and

<!-- formula-not-decoded -->

See equation (10.53) in [Bou23]. In particular, for m = 1 and m = 2

we have the identities

<!-- formula-not-decoded -->

See Example 10.78 in [Bou23].

Taylor expansions. Let [ U ] ∈ Gr( d, k ) and ξ ∈ T [ U ] Gr( d, k ) , and consider the geodesic γ ( t ) = Exp [ U ] ( tξ ) for t ∈ [0 , 1] . Define the function g : [0 , 1] → R by g ( t ) := f ( γ ( t )) . Then we have

<!-- formula-not-decoded -->

See Example 10.81 in [Bou23]. By Taylor's theorem applied to g around 0 and (27) we have

<!-- formula-not-decoded -->

for some s ∈ [0 , 1] , and where we used the mean value form of the remainder. We also have the following Taylor expansion the gradient of f

<!-- formula-not-decoded -->

See Step 2 in the proof of Proposition 10.55 in [Bou23].

Lipschitz continuous derivatives. The function f is said to have an L -Lipschitz continuous m -th derivative if

<!-- formula-not-decoded -->

See Proposition 10.83 in [Bou23]. For the special case m = 2 , this is equivalent to Hessian L -Lipschitzness, which states that for all [ U ] ∈ Gr( d, k ) and ξ ∈ T [ U ] Gr( d, k )

<!-- formula-not-decoded -->

See Exercise 10.89 in [Bou23].

## B Analysis of the block Rayleigh quotient

In this section, we state and prove the two main technical results behind Theorems 1 and 2. We state them here in terms of the negative block Rayleigh quotient (14). We will use them in subsequent sections on the empirical and population reconstruction risk, which we recall from Section 4 are up to an additive constant equal to the negative block Rayleigh quotient of Σ n and Σ respectively.

Recall the setup of Section 4. We have a symmetric matrix A ∈ R d × d and its associated negative block Rayleigh quotient F : Gr( d, k ) → R given by

<!-- formula-not-decoded -->

In the context of Appendix A, this function can be obtained from the one defined on Euclidean space ˜ F : R d × k → R given by ˜ F ( B ) = -(1 / 2) Tr( B T AB ) . Hence the Riemannian gradient and Hessian of F are given by, using (24) and (25),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ∆ = lift V ( ξ ) .

## B.1 Hessian Lipschitzness of the block Rayleigh quotient

Recall the discussion on the higher order derivatives ∇ s F of F from Appendix A.

Proposition 2. For all [ V ] ∈ Gr( d, k ) and ξ 1 , ξ 2 , ξ 3 ∈ T [ V ] Gr( d, k ) , it holds that

<!-- formula-not-decoded -->

where ∆ j = lift V ( ξ j ) for j ∈ { 1 , 2 , 3 } . As a consequence

<!-- formula-not-decoded -->

and for all [ V ] ∈ Gr( d, k ) and ξ ∈ T [ V ] Gr( d, k ) ,

<!-- formula-not-decoded -->

Proof. Fix [ V ] ∈ Gr( d, k ) and ξ 1 , ξ 2 , ξ 3 ∈ T [ V ] Gr( d, k ) . Then we have by (26)

<!-- formula-not-decoded -->

Let ∆ 3 := lift V ( ξ 3 ) = PSQ T be a SVD, ∆ 1 := lift V ( ξ 1 ) , and ∆ 2 := lift V ( ξ 2 ) . Define

<!-- formula-not-decoded -->

where [ V ( t )] = Exp [ V ] ( tξ 3 ) by (6) and where we used (23) for the parallel transport map. Using these definitions, (27), (33), and the fact that the map lift preserves the inner product

<!-- formula-not-decoded -->

where we used the identity ( I d -V ( t ) V ( t ) T )∆ 2 ( t ) = ∆ 2 ( t ) , which holds since V ( t ) T ∆ 2 ( t ) = 0 by (4), to simplify the resulting expression. Now taking derivatives with respect to t , dropping the dependence on t in the notation, and writing ˙ V for the derivative of V ( t ) , we get

<!-- formula-not-decoded -->

Noting that ˙ B ( t ) = -V Q cos( tS ) SP T -P sin( tS ) SP T , we get

<!-- formula-not-decoded -->

Replacing in (36) and simplifying, then using (35) and (34) finishes the proof of the first statement. The second follows from the Cauchy-Schwarz inequality, the inequality ∥ V C ∥ F ≤ ∥ V ∥ op ∥ C ∥ F and ∥ V ∥ op = 1 , the submultiplicativity of the Frobenius norm, and the fact that ∥ lift V ( ξ ) ∥ F = ∥ ξ ∥ for all ξ ∈ T [ U ] Gr( d, k ) . See the end of Appendix A for the last statement.

## B.2 Generalized self-concordance of the block Rayleigh quotient

Proof of Proposition 1. Recall that ( v j ) d j =1 is a basis of eigenvectors of A ordered non-increasingly according to their corresponding eigenvalues ( µ j ) d j =1 . Let V ∗ ∈ St( d, k ) be the matrix whose j -th column is v j . We start with the first statement, and with the case where γ ( t ) = Exp [ V ∗ ] ( t Log [ V ∗ ] ([ V ])) . Define ξ := Log [ V ∗ ] ([ V ]) , and let lift V ∗ ( ξ ) = PSQ T be a SVD. Let r be the rank of lift V ∗ ( ξ ) , and let P r be the d × k matrix whose first r columns match those of P , and whose last k -r columns are 0 . Then we have by (6)

<!-- formula-not-decoded -->

where we used that the post-multiplication by Q T , an orthogonal matrix, can be dropped without affecting the equivalence class, and we used that P sin( tS ) = P r sin( tS ) since the last k -r singular values in S are zero. Now let V ( t ) = V ∗ Q cos( tS ) + P r sin( tS ) . Then we have

<!-- formula-not-decoded -->

and its derivatives are given by

<!-- formula-not-decoded -->

A straightforward computation shows that

<!-- formula-not-decoded -->

Replacing yields

<!-- formula-not-decoded -->

To further simplify this expression, let V ⊥ ∗ ∈ St( d, d -k ) be the matrix whose j -th column is v k + j . Then since V T ∗ lift V ∗ ( ξ ) = 0 by (4) and by definition of P r , we have V T ∗ P r = 0 . Hence there exists

Γ ∈ R ( d -k ) × k whose first r columns are orthonormal and last k -r columns are zero such that P r = V ⊥ ∗ Γ . Finally, we may write an eigendecomposition of A as

<!-- formula-not-decoded -->

where D = diag( µ 1 , . . . , µ d ) . Performing block-wise matrix multiplication we obtain the identities

<!-- formula-not-decoded -->

where D &lt;k = diag( d 1 , . . . , d k ) and D &gt;k = diag( d k +1 , . . . , d d ) . Replacing in (38) V ( t ) and ˙ V ( t ) by their expressions and using the identities (39) yields

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where in the penultimate line we have used Holder's inequality for Schatten p -norms, and in the last we have used the fact that the matrix S √ cos(2 tS ) { Q T D ≤ k Q -Γ T D &gt;k Γ } S √ cos(2 tS ) is positivesemidefinite, so its nuclear norm equals its trace. To see why the latter matrix is positive-semidefinite, note that for any k -dimensional unit vector y ,

<!-- formula-not-decoded -->

where these inequalities follow from the fact that Qy is unit norm, and Γ y is at most unit norm by definition of Γ , and furthermore the singular values in S correspond to the principal angles between [ V ∗ ] and [ V ] , which by assumption are less than π/ 4 , so that cos(2 tS ) &gt; 0 . This last observation also shows that 2 ∥ S tan(2 tS ) ∥ ∞ = 2 θ tan(2 tθ ) where θ is the maximum principal angle between [ V ∗ ] and [ V ] . This concludes the proof of the first statement for the case γ ( t ) = Exp [ V ∗ ] ( t Log [ V ∗ ] ([ V ])) .

For the second statement, we first show that g ′′ ( t ) &gt; 0 for all t ∈ [0 , 1] . Indeed, expanding the trace expression in (40) we obtain

<!-- formula-not-decoded -->

where in the second line we used that the columns of Q are orthonormal, and that the columns of Γ are either of length one or zero, and in the last line we used the assumption µ k -µ k +1 &gt; 0 . We use this result to justify rearranging the first statement of Proposition 1 as follows

<!-- formula-not-decoded -->

Integrating once, exponentiating, then integrating twice yields the second statement in Proposition 1. See the proof of Lemma 1 in [Bac10] for a very similar calculation.

Finally, the case γ ( t ) = Exp [ V ] ( t Log [ V ] ([ V ∗ ])) follows from the first one using the identity Exp [ V ] ( t Log [ V ] ([ V ∗ ])) = Exp [ V ∗ ] ((1 -t ) Log [ V ∗ ] ([ V ])) . This holds since both curves parametrize the unique length-minimizing geodesic from [ V ] to [ V ∗ ] .

Proof of Corollary 2. This is an immediate consequence of Proposition 1. In particular, it is enough to replace the occurrences of g ′ (0) and g ′′ (0) with their expressions in terms of the gradient and Hessian of F using (28) and (27) and using the coarse bounds

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

valid for θ ∈ (0 , π/ 4) .

## C Technical lemmas and computations

This section collects several supporting lemmas and explicit computations used in the proofs of the main results in Section D. Throughout we make the assumption that E[ ∥ X ∥ 2 2 ] &lt; ∞ , so that Σ is well-defined. Recall also the definition of U ⊥ ∗ from the second paragraph of Section 3.

## C.1 Gradient and Hessian computations

Define the function ˜ ℓ : R d × k × R d → [0 , ∞ ) by

<!-- formula-not-decoded -->

The reconstruction loss ℓ : Gr( d, k ) × R d → [0 , ∞ ) is given by

<!-- formula-not-decoded -->

which is well defined, as the right-hand side does not depend on the choice of representative in [ U ] . We thus have by (24) and (25), for any [ U ] ∈ Gr( d, k ) and ξ ∈ T [ U ] Gr( d, k )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ∆ = lift U ( ξ ) , and where we computed the Euclidean gradient and Hessian of ˜ ℓ with respect to U to obtain these expressions. We can express the empirical and population risk defined in Section 2 in terms of the reconstruction loss (41) as

<!-- formula-not-decoded -->

By linearity of the grad and Hess operators along with (42) and (43), or alternatively by working with ˜ R and ˜ R n defined in (2) and (1) and formulas (24) and (25), the gradients of R and R n satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and similarly for their Hessians

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 1. Let i ∈ [ d -k ] , j ∈ [ k ] , and E ij ∈ R ( d -k ) × k be the matrix whose ( i, j ) -th entry is one and its remaining entries are zero. Define ξ ij by lift U ∗ ( ξ ij ) = U ⊥ ∗ E i,j . Then

<!-- formula-not-decoded -->

i.e. ( ξ ij ) are a basis of eigenvectors of Hess R ([ U ∗ ]) and their associated eigenvalues are ( λ j -λ k + i ) .

Proof. We have by (46) and (43)

<!-- formula-not-decoded -->

where in the first line we used the identity ( I -U ∗ U T ∗ ) = U ⊥ ∗ U ⊥ ,T ∗ , and in the second we expanded Σ = [ U ∗ | U T ∗ ] · Λ · [ U ∗ | U T ∗ ] T and performed block-wise matrix multiplication.

Corollary 3. Assume that λ k &gt; λ k +1 . Then Hess R ([ U ∗ ]) is positive definite, and for any ξ ∈ T [ U ∗ ] Gr( d, k )

<!-- formula-not-decoded -->

where lift U ∗ ( ξ ) = U ⊥ ∗ C and

<!-- formula-not-decoded -->

Proof. The positive definiteness of Hess R ([ U ∗ ]) follows directly from the positiveness of its eigenvalues from Lemma 1 under the assumed condition λ k &gt; λ k +1 . This shows that Hess R ([ U ∗ ]) is invertible. Expanding ξ into the basis of eigenvectors ( ξ ij ) from Lemma 1 yields

<!-- formula-not-decoded -->

where the second equality holds by (5). Therefore we have

<!-- formula-not-decoded -->

Applying lift U ∗ to both sides and using its linearity yields the first identity of the corollary. The second follows from a similar argument.

## C.2 Convergence of empirical gradients and Hessians

Lemma 2. For all n ∈ N , it holds that

<!-- formula-not-decoded -->

where H = Hess R ([ U ∗ ]) and where ∥ ξ ∥ 2 H -1 = ⟨ H -1 ( ξ ) , ξ ⟩ [ U ∗ ] for ξ ∈ T [ U ∗ ] Gr( d, k ) .

Proof. By (45) we have

<!-- formula-not-decoded -->

and the statement follows since the elements of the sum are all equal to E[ ∥ grad ℓ ([ U ∗ ] , X ) ∥ 2 H -1 ] , since ( X i ) are i.i.d. with the same distribution as X . The third equality holds since the cross-terms vanish by the independence of ( X i ) , E[grad ℓ ([ U ∗ ] , X )] = grad R ([ U ∗ ]) by (44), and grad R ([ U ∗ ]) = 0 since [ U ∗ ] is a minimizer of R .

Lemma 3. It holds that

<!-- formula-not-decoded -->

where H = Hess R ([ U ∗ ]) and where ∥ ξ ∥ 2 H -1 = ⟨ H -1 ( ξ ) , ξ ⟩ [ U ∗ ] for ξ ∈ T [ U ∗ ] Gr( d, k ) .

Proof. By (42), and using the identity ( I d -U ∗ U T ∗ ) = U ⊥ ∗ U ⊥ ,T ∗ we have

<!-- formula-not-decoded -->

Let C = ( U ⊥ ,T ∗ X )( U ∗ X ) T . It has entries C i,j = ⟨ u k + i , X ⟩ · ⟨ u j , X ⟩ . Hence by Corollary 3

<!-- formula-not-decoded -->

where C ′ = C ij / ( λ j -λ k + i ) . Now

<!-- formula-not-decoded -->

Replacing C ij by its value yields the result.

Lemma 4. Assume that for all i, s ∈ [ d -k ] and j, t ∈ [ k ]

<!-- formula-not-decoded -->

Then

<!-- formula-not-decoded -->

where H n = Hess R n ([ U ∗ ]) , and where G is a ( d -k ) × k matrix with jointly Gaussian mean zero entries with covariances E[ G ij G st ] = Λ ijst / ( δ ij δ st ) where δ ij = λ j -λ k + i .

Proof. Recall the global assumption E ∥ X ∥ 2 2 &lt; ∞ so that the population covariance matrix Σ exists. By (46) and (43) this implies that H = Hess R ([ U ∗ ]) exists. Thus by (47) and the weak law of large numbers, we have H n p - → H , and by the continuous mapping theorem H -1 n p - → H -1 . On the other hand consider the random matrix

<!-- formula-not-decoded -->

By the proof of Lemma (3) we have Z = -C ′ where C ′ ij = ( ⟨ u k + i , X ⟩ · ⟨ u j , X ⟩ ) / ( λ j -λ k + i ) , thus E[ Z ij ] = 0 and E[ Z ij Z st ] = Λ ijst / ( δ ij δ st ) . Hence by the central limit theorem and (45), as n →∞ ,

<!-- formula-not-decoded -->

where G is the Gaussian random matrix in the statement. Finally, by another application of the central limit theorem, we have that √ n · grad R n ([ U ∗ ]) converges in distribution to a random Gaussian element, and hence ( H -1 n -H -1 )[ √ n · grad R n ([ U ∗ ])] converges to 0 in probability by Slutsky's theorem. Therefore

<!-- formula-not-decoded -->

and the final statement of the lemma is obtained by another application of Slutsky's theorem, an an application of the continuous mapping theorem with the map C ↦→ U ⊥ ∗ C , recalling that U ⊥ ∗ U ⊥ ,T ∗ ∆ = ∆ for all ∆ ∈ H U ∗ .

Lemma 5. Assume that λ k &gt; λ k +1 and that E[ X 4 j ] &lt; ∞ for all j ∈ [ d ] . If

<!-- formula-not-decoded -->

then with probability at least 1 -δ/ 4 , where H = H -1 / 2 ◦ H ◦ H -1 / 2 and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where M = H -1 / 2 ◦ Hess ℓ ([ U ∗ ] , X ) ◦ H -1 / 2 , H n = Hess R n ([ U ∗ ]) , and H = Hess R ([ U ∗ ]) .

<!-- formula-not-decoded -->

Proof. Let M i = H -1 / 2 ◦ Hess ℓ ([ U ∗ ] , X i ) ◦ H -1 / 2 . We have the variational characterization

<!-- formula-not-decoded -->

Thus by Bousquet's inequality [Bou02], specifically the version in [Van+16, Corollary 16.1], with probability at least 1 -δ/ 4

<!-- formula-not-decoded -->

Now by the Matrix Bernstein inequality [Tro+15, Theorem 6.6.1], we have

<!-- formula-not-decoded -->

Combining the two bounds and solving for n yields the result.

Lemma 6. The parameters V and ν defined in Lemma 5 admit the explicit expression given in Remark 3.

Proof. Let ξ ∈ T [ U ∗ ] Gr( d, k ) , and let U T ∗ C = ∆ = lift U ∗ ( ξ ) . We have by Corollary 3, for ξ 1 = H -1 / 2 [ ξ ]

<!-- formula-not-decoded -->

where C ′ ij = C ij / √ λ j -λ k + i . Now by (43), we have for ξ 2 = Hess ℓ ([ U ∗ ] , X )[ ξ 1 ]

<!-- formula-not-decoded -->

Defining ˜ X j = ⟨ X,u j ⟩ , and writing ˜ X ≤ k for its first k entries, and ˜ X &gt;k for its remaining d -k entries, we have

<!-- formula-not-decoded -->

Denote the term in brackets by D . Then again by Corollary 3, for ξ 3 = H -1 / 2 [ ξ 2 ] , we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking the square of this expression, expanding, and taking expectations yields an explicit expression of ⟨ M [ ξ ] , ξ ⟩ 2 [ U ∗ ] in terms of C . Noting that the map that sends ξ to C is an isometric isomorphism, the supremum of the former over vectors ∥ ξ ∥ = 1 is equal to the supremum of the latter over ∥ C ∥ F = 1 . This concludes the proof for ν . For V , note that

<!-- formula-not-decoded -->

Expanding the square and taking expectations gives an explicit expression of ∥ M [ ξ ] ∥ 2 2 in terms of C , and by the same argument as for ν , V is the supremum over ∥ C ∥ F = 1 of this explicit expression.

## C.3 Localization argument for ERM

The following two corollaries are immediate consequences of Corollary 2 and Proposition 2, since the population and empirical reconstruction risks R and R n are, up to an insignificant constant, equal to the negative block Rayleigh quotients -(1 / 2) Tr( U T Σ U ) and -(1 / 2) Tr( U T Σ n U ) respectively.

Corollary 4. Assume that λ k &gt; λ k +1 , and let [ U ] ∈ Gr( d, k ) such that θ k ([ U ∗ ] , [ U ]) &lt; π/ 4 . Then

<!-- formula-not-decoded -->

where ξ = Log [ U ∗ ] ([ U ]) .

Corollary 5. It holds that

<!-- formula-not-decoded -->

and for all [ U ] ∈ Gr( d, k ) and ξ ∈ T [ U ] Gr( d, k ) ,

<!-- formula-not-decoded -->

Lemma 7. On the event that θ k ([ U n ] , [ U ∗ ]) &lt; π/ 4 , it holds that

<!-- formula-not-decoded -->

where ˜ H n = H -1 / 2 ◦ H n ◦ H -1 / 2 , λ min ( ˜ H n ) is its smallest eigenvalue, H = Hess R ([ U ∗ ]) , H n = Hess R n ([ U ∗ ]) , and ∥ ξ ∥ 2 H = ⟨ H ( ξ ) , ξ ⟩ [ U ∗ ] for ξ ∈ T [ U ∗ ] Gr( d, k ) .

Proof. Denote Log [ U ∗ ] ([ U n ]) by ξ n . Since [ U n ] minimizes the empirical risk we have

<!-- formula-not-decoded -->

On the other hand by Corollary 4 we have

<!-- formula-not-decoded -->

By the Cauchy-Schwartz inequality

<!-- formula-not-decoded -->

And we also have

<!-- formula-not-decoded -->

̸

Combining (48), (49), (50), and (51) we obtain the result.

## C.4 Matrix identities and perturbation bounds

Lemma 8. Assume that λ k -λ k +1 &gt; 0 and that ∥ Σ -Σ n ∥ op ≤ ( λ k -λ k +1 ) / 2 . Then

<!-- formula-not-decoded -->

Proof. Let δ = λ k -λ k +1 . Recall that ( u n,j ) is an orthonormal basis of eigenvectors of Σ n ordered non-increasingly according to their corresponding eigenvalues ( λ n,j ) , with ties broken arbitrarily. On the one hand, we have by inequality (1.63) in [Tao12]

<!-- formula-not-decoded -->

for all j ∈ [ d ] . This implies that

<!-- formula-not-decoded -->

where the inequality follows by the assumed bound in the lemma. On the other hand, we have by the operator norm version of Theorem 1 in [YWS15] 3 , and taking r = 1 and s = k in the statement

<!-- formula-not-decoded -->

where we have used (53) to simplify the denominator appearing in the original statement. Using again (52) to lower bound the denominator we get

<!-- formula-not-decoded -->

Using the inequality x/ ( c -x ) ≤ 2 x/c valid for x ∈ [0 , c/ 2]

finishes the proof.

Corollary 6. Assume that λ k &gt; λ k +1 and that E[ X 4 j ] &lt; ∞ for all j ∈ [ d ] . If

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then with probability at least 1 -δ/ 2

where

<!-- formula-not-decoded -->

Proof. By the second item of Theorem 5.1 in [Tro16], we have

<!-- formula-not-decoded -->

Applying Markov's inequality, using Lemma 8, and solving for n yields the result.

Lemma 9. Let U, V ∈ St( d, k ) . Let ( σ i ) d i =1 be the singular values of UU T -V V T ordered non-increasingly. Then for i ∈ [2 k ] ,

<!-- formula-not-decoded -->

and the remaining singular values are zero.

Proof. See Theorem 5.5 in Chapter 1 of [SS90].

## D Proofs of main results

## D.1 Proof of Theorem 1

Consistency. Since E[ ∥ X ∥ 2 2 ] &lt; ∞ , we have by the weak law of large numbers Σ n p - → Σ , i.e. for all ε &gt; 0

<!-- formula-not-decoded -->

Let A n be the event that ∥ Σ n -Σ ∥ op &lt; ( λ k -λ k +1 ) / 2 . By assumption the right-hand side is strictly larger than 0 , so lim n →∞ P( A n ) = 1 . On the one hand, by Lemma 8, we have on the event A n

<!-- formula-not-decoded -->

On the other we have

<!-- formula-not-decoded -->

3 See the beginning of the second paragraph after the statement of the theorem.

Now let 0 &lt; ε &lt; √ kπ/ 2 . Using the above two bounds we obtain

<!-- formula-not-decoded -->

and both terms go to zero as n →∞ .

√ n -consistency. For the rest of the proof, let ξ n = Log [ U ∗ ] ([ U ∗ ]) when well defined, and 0 otherwise. Let H = Hess R ([ U ∗ ]) and H n = Hess R n ([ U ∗ ]) , and for a positive semi-definite linear map L : T [ U ∗ ] Gr( d, k ) → T [ U ∗ ] Gr( d, k ) , let ∥ ξ ∥ 2 L = ⟨ L ( ξ ) , ξ ⟩ denote the (squared) semi-norm it induces, for ξ ∈ T [ U ∗ ] Gr( d, k ) . We note that under the eigengap assumption of the theorem, ξ ↦→∥ ξ ∥ H is a norm since H is positive-definite by Corollary 3.

Let A n be the event that θ k ([ U n ] , [ U ∗ ]) &lt; π/ 4 . By the consistency result, lim n →∞ P( A n ) = 1 . Now on this event we have by Lemma 7

<!-- formula-not-decoded -->

where ˜ H n = H -1 / 2 ◦ H n ◦ H -1 / 2 and λ min ( ˜ H n ) is its smallest eigenvalue.

Onthe one hand, by the weak law of large numbers ˜ H n p - → Id , and by the continuous mapping theorem λ -1 min ( ˜ H n ) p - → 1 . Let B n be the event that λ -1 min ( ˜ H n ) ≤ 2 , which thus satisfies lim n →∞ P( B n ) = 1 .

On the other hand, we have by Lemma 2

<!-- formula-not-decoded -->

and by the moment assumption of the theorem and Lemma 3 the expectation on the right-hand side is finite.

Therefore we have for any ε &gt; 0 and α ∈ [0 , 1 / 2) , using the above two displays and Markov's inequality

<!-- formula-not-decoded -->

and all three terms go to zero as n →∞ so that for all ε &gt; 0 and α ∈ [0 , 1 / 2)

<!-- formula-not-decoded -->

Asymptotic normality. Let A n be the event that θ k ([ U n ] , [ U ∗ ]) &lt; π/ 4 . By the consistency result, lim n →∞ P( A n ) = 1 .

Since [ U n ] minimizes the empirical reconstruction risk, it holds that grad R n ([ U n ]) = 0 . On the event A n , we have by the Taylor expansion (30)

<!-- formula-not-decoded -->

where we used that the parallel transport map introduced at the beginning of Appendix A sends 0 to 0 , and where the term E n is given by

<!-- formula-not-decoded -->

By the weak law of large numbers H n p - → H which is invertible by the eigengap assumption of the theorem and Corollary 3. The event B n on which H n is invertible thus satisfies lim n →∞ P( B n ) = 1 . On the event A n ∩ B n we thus have, rearranging the first display and scaling by √ n

<!-- formula-not-decoded -->

Now we claim that √ n · E n p - → 0 . Indeed, we have by Corollary 5

<!-- formula-not-decoded -->

where in the first line we used the easy to check identity P ξ,s = P sξ, 1 . Once again using the weak law of large numbers and the continuous mapping theorem we obtain ∥ Σ n ∥ F p - → ∥ Σ ∥ F . Thus using (54) we obtain √ n · E n p - → 0 . The asymptotic normality statement in the Theorem then follows from Lemma 4, Slutsky's theorem, and the fact that the events A c n and B c n have vanishing probability as n →∞ .

Excess Risk. Let A n be the event that θ k ([ U n ] , [ U ∗ ]) &lt; π/ 4 . By the consistency result, lim n →∞ P( A n ) = 1 . By the Taylor expansion (29), we have on this event

<!-- formula-not-decoded -->

where the error term E n is given by

<!-- formula-not-decoded -->

for some s ∈ [0 , 1] , and where we again used the easy to check identity P ξ,s = P sξ, 1 . We claim that n · E n p - → 0 . Indeed by Corollary 5, we have

<!-- formula-not-decoded -->

where we used in the equality that the parallel transport map is an isometry as described in the beginning of Appendix A. Thus using (54) we obtain n · E n p - → 0 . Finally, we have, using the asymptotic normality statement in the theorem, proven above, the continuous mapping theorem, and the explicit description of H 1 / 2 from Lemma 1 that lift U ∗ ( H 1 / 2 [ √ n · ξ n ]) converges in distribution U ⊥ ∗ H where H is the Gaussian matrix in the statement of the theorem. The result then follows from an application of Slutsky's theorem, the fact that the event A c n has vanishing probability as n →∞ , and that ∥ U ⊥ ∗ C ∥ F = ∥ C ∥ F for any ( d -k ) × k matrix C .

## D.2 Proof of Remark 1

We have on the one hand by Lemma 9 that

<!-- formula-not-decoded -->

where we have shortened θ j = θ j ([ U n ] , [ U ∗ ]) . Recall from Section 2 that the singular values of ∆ n = lift U ∗ (Log [ U ∗ ] ([ U n ])) are the principal angles between [ U n ] and [ U ∗ ] . Define the function φ : R d × k → R d × k as follows. For a matrix A ∈ R d × k , let A = PSQ T be a SVD of A . Then we define φ ( A ) = P sin( S ) Q T where sin is applied element-wise to the singular values. Now

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

where s max is the maximum singular value of A , and we used the fact that ( x -sin( x )) /x → 0 as x → 0 . Therefore φ is differentiable at 0 , and its derivative there is the identity map, so by the delta method [e.g. Van00, Theorem 3.1]) and the asymptotic normality result in Theorem 1 we obtain

<!-- formula-not-decoded -->

Applying the continuous mapping theorem with the map A ↦→ 2 1 /p ∥ A ∥ p , using (55), and noting that ∥ U ⊥ ∗ G ∥ p = ∥ G ∥ p since U ⊥ ∗ ∈ St( d, d -k ) finishes the proof.

## D.3 Proof of Theorem 2

Let ξ n = Log [ U ∗ ] ([ U ∗ ]) when well defined, and 0 otherwise. Let H = Hess R ([ U ∗ ]) and H n = Hess R n ([ U ∗ ]) , and for a positive semi-definite linear map L : T [ U ∗ ] Gr( d, k ) → T [ U ∗ ] Gr( d, k ) , let ∥ ξ ∥ 2 L = ⟨ L ( ξ ) , ξ ⟩ denote the (squared) semi-norm it induces, for ξ ∈ T [ U ∗ ] Gr( d, k ) .

Let A n be the event that θ k ([ U n ] , [ U ∗ ]) &lt; π/ 4 . Under the sample size restriction of the theorem, and in particular the third term in this restriction, this event happens with probability at least 1 -δ/ 2 by Corollary 6.

Now on this event, we have by Lemma 7 that the following inequality holds

<!-- formula-not-decoded -->

where ˜ H n = H -1 / 2 ◦ H n ◦ H -1 / 2 .

Let B n be the event that λ min ( ˜ H n ) ≥ 1 / 2 . Under the sample size restriction of the theorem, and in particular the first two terms in this restriction, this event happens with probability at least 1 -δ/ 4 by Lemmas 5 and 6.

Now by Lemmas 2 and 3 and Markov's inequality we also have that on an event C n that holds with probability at least 1 -δ/ 4

<!-- formula-not-decoded -->

Therefore, on the event A n ∩ B n ∩ C n that holds with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

and on this same event, we have by Corollary 4, noting that on this event ξ n = Log [ U ∗ ] ([ U ∗ ])

<!-- formula-not-decoded -->

which concludes the proof.

## D.4 Proof Sketch for Example 2

When X ∼ N (0 , Σ) , we have

<!-- formula-not-decoded -->

and and finally

<!-- formula-not-decoded -->

̸

Using these identities in the sums in Remark 3 and simplifying shows that the maximization problem becomes one over the simplex which can be solved directly for V or very well-approximated for ν .

̸

̸

̸

<!-- formula-not-decoded -->

̸

̸

## D.5 Remarks on omitted proofs

Quantile bounds in Corollary 1 and Remark 1. The upper bounds in these statements are a direct consequence of the Gaussian concentration inequality [e.g. BLM13, Theorem 5.6]. For the lower bounds and a step by step derivation, see for example [EME24, Appendix A]. Note also that the quantile bounds in Example 1 are a direct consequence of those in Corollary 1 and Remark 1.

Claim in Remark 2. As is clear from the Proof of Theorem 1, the key properties leading to it are the self-concordance and Hessian Lipschitzness of the empirical and populations reconstruction risks, i.e. Corollaries 4 and 5. These are themselves derived from the more general results we obtained in Propositions 1 and 2 that holds for the negative block Rayleigh quotient. These can be used directly with minor adjustments to prove the claim we made in Remark 2. Furthermore, up to generalizing the parameters appearing in the sample size restriction, one may use Propositions 1 and 2 to obtain a more general version of Theorem 2 that holds in the setting described in Remark 2.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Theorem 1, Proposition 1, and Theorem 2 are the formal statements of what is described in the abstract.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: the section "Discussion" contains three paragraphs describing the limitations of our results.

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

Justification: Yes. The assumptions are stated clearly in the theorems, and the proofs are provided in the supplemental material.

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

Justification: the paper does not include experiments.

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

Answer: [NA]

Justification: the paper does not include experiments.

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

Justification: the paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: the paper does not include experiments.

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

Justification: the paper does not include experiments.

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

Justification: there is no societal impact of the work performed.

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

Justification: the paper does not use existing assets.

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

Justification: the paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: the paper does not involved crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: the paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: the core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.