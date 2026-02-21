## A Private Approximation of the 2nd-Moment Matrix of Any Subsamplable Input

## Bar Mahpud

Faculty of Engineering Bar-Ilan University Israel mahpudb@biu.ac.il

Or Sheffet

Faculty of Engineering Bar-Ilan University Israel or.sheffet@biu.ac.il

## Abstract

We study the problem of differentially private second moment estimation and present a new algorithm that achieve strong privacy-utility trade-offs even for worst-case inputs under subsamplability assumptions on the data. We call an input ( m,α,β ) -subsamplable if a random subsample of size m (or larger) preserves w.p ≥ 1 -β the spectral structure of the original second moment matrix up to a multiplicative factor of 1 ± α . Building upon subsamplability, we give a recursive algorithmic framework similar to Kamath et al. (2019) that abides zero-Concentrated Differential Privacy (zCDP) while preserving w.h.p the accuracy of the second moment estimation upto an arbitrary factor of (1 ± γ ) . We then show how to apply our algorithm to approximate the second moment matrix of a distribution D , even when a noticeable fraction of the input are outliers.

## 1 Introduction

Estimating the second moment matrix (or equivalently, the covariance matrix) of a dataset is a fundamental task in machine learning, statistics, and data analysis. In a typical setting, given a dataset of n points in R d , one aims to compute an empirical second moment (or covariance) matrix that is close, in spectral norm, to the true second moment matrix. However, as modern datasets increasingly contain sensitive information, maintaining strong privacy guarantees has become a key consideration.

Anatural way to protect sensitive data is through differential privacy (DP). In this paper, we focus on the zero-Concentrated Differential Privacy (zCDP) framework (Bun &amp; Steinke, 2016), which offers elegant composition properties and somewhat tighter privacy-utility trade-offs compared to traditional ( ϵ, δ ) -DP. While there have been works regarding the estimation of the second moment matrix (and PCA), they mostly focused on Gaussian input or well-conditioned input (see Related Work below). In contrast, our work focuses on a general setting, where the input's range is significantly greater than λ min , the least eigenvalue of the 2nd-moment matrix.

Suppose indeed we are in a situation where the first and least eigenvalues of the input's 2nd moment matrix are very different. By and large, this could emanate from one of two options: either it is the result of a few outliers, in which case it is unlikely to approximate the 2nd moment matrix well with DP; or it is the case that the underlying distribution of the input does indeed have very different variances along different axes, and here DP approximation of the input is plausible. Our work is focuses therefore on the latter setting, which we define using the notion of subsamplability . Namely that from a sufficiently large random subsample, one can recover a spectral approximation to the original second moment matrix with high probability. This property resonates with classical matrix-concentration results (namely, matrix Bernstein bounds), yet - as our analysis shows - our subsamplability assumption offers a less nuanced path to controlling the tail behavior of the data. In this work we formalize this notion of subsamplability - which immediately gives a non-private

approximation the data's second moment matrix, and show how it can be integrated into a privacypreserving algorithm with some overhead.

Subsamplability Assumption. Throughout this paper, we assume that our n -size input dataset is subsamplable, which we formally define as follows.

Definition 1.1. ( ( m,α,β ) -subsamplability) Let X ⊆ R d be a dataset of n points. Fix m ≤ n, α &gt; 0 , β ∈ (0 , 1) . Let ˆ X 1 , . . . , ˆ X m ′ be a random subsample of m ′ ≥ m points i.i.d from X . Denote Σ = 1 n ∑ i ∈ [ n ] X i X T i and ˆ Σ = 1 m ′ ∑ i ∈ [ m ′ ] ˆ X i ˆ X T i , then the dataset X is ( m,α,β ) -subsamplable if:

<!-- formula-not-decoded -->

By assuming subsamplability, we ensure that the critical spectral properties of the data are retained, enabling efficient and accurate private estimation. This assumption provides a tractable way to manage the inherent complexity of the problem while maintaining robustness to variations in the data. On the contrapositive - when the data isn't subsamplable, estimating the second moment matrix becomes significantly more challenging. Furthermore, in the case where the n input points are drawn i.i.d. from some distribution (the case we study in Section 4) subsamplability follows directly from the convergence of a large enough sample to the true (distributional) second moment matrix; alternatively, sans subsamplability we cannot estimate the distribution's second moment matrix.

It is important to note that our subsamplability assumption is weaker than standard concentration bounds, which state that for any α, β there exists m ( α, β ) such that a random subsample of m (or more) points preserve w.p. ≥ 1 -β the spectral structure of Σ upto a (1 ± α ) -factor. Here we only require that for some α, β there exists such a m ( α, β ) , a distinction that allows us to cope even with a situation of a well-behaved distribution with outliers, as we discuss in Section 4. In our analysis we require that α = O (1) (we set it as α ≤ 1 / 2 purely for the ease of analysis); however, our analysis does require a bound on the β parameter, namely having β = O ( α/ log( R )) , with R denoting the bound on the L 2 norm of all points. It is an interesting open problem to replace this requirement on β with a O (1) -bound as well. It is also important to note that our result is stronger than the baseline we establish: subsamplability implies that using (roughly) m / ϵ samples, one can succesfully apply the subsample-and-aggregate framework (Nissim et al., 2007) and obtain a 1 ± O ( α ) -approximation of the spectrum of the second moment matrix. In contrast, our algorithm can achieve a (1 ± γ ) -approximation of the second moment matrix of the 'nice' portion of the input even for γ ≪ α (provided we have enough input points). Details below.

Contributions. First, we establish a baseline for this problem. Under our subsamplability assumption, we use an off-the-shelf algorithm of Ashtiani &amp; Liaw (2022) that follows the 'subsample-andaggregate paradigm (Nissim et al., 2007) to privately return a matrix ˜ Σ satisfying (1 -2 α )Σ ⪯ ˜ Σ ⪯ (1 + 2 α )Σ .

We then turn our attention to our algorithm, which is motivated by the same recursive approach given in Kamath et al. (2019): In each iteration we deal with an input X whose 2nd moment matrix satisfy I ⪯ Σ ⪯ κI , add noise (proportional to κ / m ) to its 2nd-moment matrix and find the subspace of large eigenvalue (those that are greater than ψκ for ψ ≈ 1 / m ), and then apply a linear transformation Π reducing the projection onto the subspace of large eigenvalues by 1 / 2 , thereby reducing the second moment matrix of Π X so that it is ⪯ 3 κ 7 I . So we shrink R , the range of the input, to √ 3 / 7 R and continue by recursion. Yet unlike Kamath et al. (2019) who work with the underlying assumption that the input is Gaussian, we only know that our input is subsamplable, and so in our setting there could be input points whose norm is greater than √ 3 / 7 R after applying Π and whose norm we must shrink to fit in the √ 3 / 7 R -ball. So the bulk of our analysis focuses on these points that undergo shrinking, and show that they all must belong to a particular set we refer to as P tail (see Definition 3.2). We argue that there aren't too many of them (just roughly a β / m -fraction of the input) and that even with shrinking these points, the second moment matrix of the input remains ⪰ I . This allows us to recurse all the way down to a setting where κ ∝ m , where all we have to do is to simply add noise to the 2nd moment matrix to obtain a (1 ± γ ) -approximation w.h.p.

We then apply our algorithm to an ensemble of points drawn from a general distribution (even a heavy-tailed one). So next we consider any distribution D with a finite second moment Σ D where

the vector y = Σ -1 / 2 D x for x ∼ D exhibits particular bounds (See Claim 4.2 for further details), and give concrete sample complexity bounds for our algorithm to approximate Σ D up to a factor of 1 ± γ w.p. 1 -ξ . We then consider a mixture of such a well-behaved D with an η -fraction of outliers . We show that our algorithm allows us to cope with the largest fraction of outliers (roughly ˜ O (1 /d ) ) provided that the second moment matrix of the outliers Σ out satisfies Σ out ⪯ O ( 1 / η )Σ D . In contrast, the subsample and aggregate baseline (and other baselines too) not only requires a smaller bound on η but also has a significantly large sample complexity bound. Details appear in Section 4.

Organization. After surveying related work in the remainder of Section 1, we introduce necessary definitions and background in Section 2. In Section 3.1 we survey the baseline of Ashtiani &amp; Liaw (2022), and in Section 3.2 we discuss using existing algorithms to estimate the initial parameters of the input we require, namely its range R and its least eigenvalue λ min . Multiplying R 2 by 1 /λ min we obtain an input that indeed satisfies I ⪯ Σ ⪯ κI (with κ = R 2 /λ min ). Then, in Section 3.3 we present our algorithm and state its utility theorem, which we prove in Section 3.4. Finally, Section 4 illustrates how to apply our framework to a general (potentially heavy tailed) distribution, including the case of a noticeable fraction of outliers.

Related Work. Differential privacy has been extensively studied in the context of mean and covariance estimation, particularly in high-dimensional regimes. Early work by Dwork et al. (2014) proposed private PCA for worst-case bounded inputs via direct perturbation of the second moment matrix, laying foundational tools for differentially private matrix estimation. Subsequently, Nissim et al. (2007) introduced the subsample-and-aggregate framework, which has since become a standard paradigm for constructing private estimators under structural assumptions.

A significant body of research has focused on learning high-dimensional Gaussian distributions under differential privacy. Kamath et al. (2019) introduced a recursive private preconditioning technique for Gaussian and product distributions, achieving nearly optimal bounds while relying on the assumption of a well-behaved (Gaussian) input. Their approach underlies several subsequent advances in private estimation. Building on these ideas, Kamath et al. (2022) proposed a polynomialtime algorithm for privately estimating the mean and covariance of unbounded Gaussians. Their algorithm, which incorporated a novel private preconditioning step, improved both accuracy and computational efficiency.

Ashtiani &amp; Liaw (2022) proposed a general framework that reduces private estimation to its nonprivate analogue. This yielded efficient, approximate-DP estimators for unrestricted Gaussians with optimal (up to logarithmic factors) sample complexity. Their method also demonstrated the power of reduction-based techniques in bridging private and non-private statistics. Aden-Ali et al. (2021) gave near-optimal bounds for agnostically learning multivariate Gaussians under approximate DP, while Amin et al. (2019) and Dong et al. (2022) revisited the task of private covariance estimation under ϵ -DP and zCDP, respectively. These works introduced trace- and tail-sensitive algorithms for better handling of data heterogeneity.

Recent work has emphasized robustness and practical applicability. For example, Biswas et al. (2020) introduced a robust and accurate mean/covariance estimator for sub-Gaussian data, and Kothari et al. (2022) developed a robust, polynomial-time estimator resilient to adversarial outliers. Further, Alabi et al. (2023) presented near-optimal, computationally efficient algorithms for privately estimating multivariate Gaussian parameters in both pure and approximate DP models.

A particularly notable contribution is by Brown et al. (2023), who studied the problem of differentially private covariance-aware mean estimation under sub-Gaussian assumptions. They introduced a polynomial-time algorithm that achieves strong Mahalanobis distance guarantees with nearly optimal sample complexity. Their techniques also extend to distribution learning tasks with provable guarantees on total variation distance.

Our algorithm outperforms prior methods that rely on per-point bounded leverage and residual conditions-most notably the private covariance estimation algorithm of Brown et al. Brown et al. (2023)-in settings where the dataset may contain a small fraction of outliers or where individual points may exhibit high leverage scores, but the global spectral structure is preserved in random subsamples. Unlike their algorithm, which requires strong uniform constraints on every data point (i.e., no large leverage scores), our method only assumes a subsamplability condition that holds with high probability over random subsamples. This allows us to tolerate the presence of many multiple

outliers, provided they do not dominate the overall spectrum. Moreover, our algorithm is tailored for second moment estimation , and achieves strong utility guarantees even when the second moment matrix has a large condition number - a regime where the estimator of Brown et al. (2023) may incur significant error with the presence of outlier correlated with the directions of small eigenvalues. A more elaborated discussion demonstrating this setting appears in Section 4.2.

## 2 Preliminaries

Throughout the paper, we assume that our instance of dataset is subsamplable, as given in Definition 1.1.

Notations. Let S d -1 denote The unit sphere in R d , which is defined as the set of all points in d -dimensional Euclidean space that have unit norm, i.e., S d -1 = { x ∈ R d | ∥ x ∥ 2 = 1 } . Here, the superscript d -1 indicates that the unit sphere is an object of intrinsic dimension d -1 embedded in R d .

Fact 2.1 (see e.g. Tao (2012) Corollary 2.3.6) . For d sufficiently large, there exist absolute constants C, c &gt; 0 such that: Pr N ∼ GUE( σ 2 ) [ ∥ N ∥ 2 &gt; Aσ √ d ] ≤ Ce -cAd for all A ≥ C .

Let GUE( σ 2 ) denote the distribution over d × d symmetric matrices N where for all i ≤ j , we have N ij ∼ N (0 , σ 2 ) i.i.d.. From basic random matrix theory, we have the following guarantee.

Definition 2.2 (Differential Privacy (Dwork et al., 2006)) . Arandomized algorithm A satisfies ( ϵ, δ ) -differential privacy if, for all datasets D and D ′ differing in at most one element, and for all measurable subsets S of the output space of A , it holds that:

<!-- formula-not-decoded -->

Definition 2.3 (Zero-Concentrated Differential Privacy (zCDP) (Bun &amp; Steinke, 2016)) . Arandomized algorithm A satisfies ρ -zero-concentrated differential privacy ( ρ -zCDP) if, for all datasets D and D ′ differing in at most one element, and for all α &gt; 1 , the Rényi divergence of order α between the output distributions of A on D and D ′ is bounded by ρα , i.e., D α ( A ( D ) ∥A ( D ′ )) ≤ ρα . Here, ρ ≥ 0 is the privacy parameter that controls the trade-off between privacy and utility, and D α denotes the Rényi divergence of order α .

Theorem 2.4 (Bun &amp; Steinke (2016)) . If a randomized algorithm A satisfies ρ -zero-concentrated differential privacy ( ρ -zCDP), then A also satisfies ( ϵ, δ ) -differential privacy for any δ &gt; 0 , where: ϵ = ρ + √ 2 ρ ln ( 1 δ ) .

Theorem 2.5 (Composition Theorem for ρ -zCDP) . Let M 1 and M 2 be two independent mechanisms that satisfy ρ 1 -zCDP and ρ 2 -zCDP, respectively. Then their composition M 1 ◦ M 2 satisfies ( ρ 1 + ρ 2 ) -zCDP.

## 3 Technical Analysis

## 3.1 Baseline

In this section, we provide a baseline for the problem of 2nd-moment estimation using subsample and aggregate framework (Nissim et al., 2007). For the lack of space we move the entire discussion of the baseline to Appendix A, and only cite here the conclusion.

Theorem 3.1. Let ξ , ϵ , δ be parameters, and let X ⊆ R d be a ( m,α,β ) -subsamplable set of n points. Then, there exists an algorithm for which the following properties hold:

1. The algorithm is (2 ϵ, 4 e ϵ δ ) -Differential Private.
2. The algorithm returns ˜ Σ satisfying ∥ Σ -1 / 2 ˜ ΣΣ -1 / 2 -I ∥ ≤ 2 α , where Σ = 1 n XX ⊤ .

These guarantees hold under the following conditions:

<!-- formula-not-decoded -->

2. The subsamplability parameters satisfy m ≥ 2 βn/ξ.

In particular, Item 1 suggests a sample complexity bound of n = Ω( d · m ( α,β ) ϵα ) .

## 3.2 Finding Initial Parameters

Our recursive algorithm requires as input two parameters that characterize the 'aspect ratio' of the input, namely -R max , the maximum distance of any point from the origin, and λ min , the minimum eigenvalue of the input. These two parameters give us the initial bounds, as they imply that λ min I ⪯ Σ ⪯ R 2 max I . Due to space constraints, we move the entire discussion, regarding how to apply off-the-shelf algorithms, or modify such algorithms, to obtain these initial parameters to Appendix B.

## 3.3 Main Algorithm and Theorem

Next, we detail our algorithm that approximates the second moment of the input. Its starting point is the assumption that the input has a known bound on the L 2 -norm of each point R , and that the second moment matrix of the input, Σ , satisfies I ⪯ Σ ⪯ R 2 I .

## Algorithm 1 DP Second Moment Estimation

Input: a ( m,α,β ) -subsamplable set of n points X ⊆ R d , parameters: error parameter ξ ∈ (0 , 1) , privacy parameter ρ , covering radius R .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 3: return (1 -α ) ˜ Σ

## Algorithm 2 Recursive DP Second Moment Estimation (RecDPSME)

Input: a set of n points X ⊆ R d , parameters: linear shrinking η &lt; 1 , eigenvalue threshold ψ &lt; 1 , stopping value C, noise c , eigenvalue upper bound κ , radius R , iterations bound T , error parameter ξ , privacy loss ρ .

- 1: Set σ ← 4 R 2 √ T n √ 2 ρ
- 3: ˜ Σ ← 1 n XX T + N
- 2: Sample N ∼ GUE( σ 2 )
- 4: if κ ≤ C then
- 5: return ˜ Σ .
- 6: end if
- 8: Π ← η Π V +Π V
- 7: V ← Span ( { v i : eigenvector of ˜ Σ with eigenvalue ≥ ψκ } )
- ⊥ { Π U denotes the projection matrix onto U .}

<!-- formula-not-decoded -->

- 9: Y ← √ 8 7 Π X .

In our analysis, the following definition plays a key role.

Definition 3.2. Let X be a ( m,α,β ) -subsamplable set. We denote P tail as the set of points whose projection onto some direction u in R d is m times greater than expected, namely

<!-- formula-not-decoded -->

Theorem 3.3. Fix parameters ξ ∈ (0 , 1) , ρ &gt; 0 , γ &gt; 0 and κ ≥ 1 . Let X ⊂ R d be a ( m,α,β ) -subsamplable set of n points bounded in L 2 norm by R 2 , with α 1 / 2 and β α R 2

s.t. I ⪯ Σ ⪯ R 2 I where Σ = 1 n XX T . Then, denoting T = log 7 / 3 ( 1 1 -α ) R 2 640 m = O (log( R / m )) , we have that Algorithm 1 satisfies ρ -zCDP, 1 , and if

≤ ≤ 4(1+ α ) log( (1+ α ) m ) ( )

<!-- formula-not-decoded -->

Then (1) P tail holds at most a ( β + β 2 m ) -fraction of | X | , and (2) w.p. ≥ 1 -2 ξ it outputs ˜ Σ such that:

where Σ eff = 1 n ∑ x ∈ X \ P tail xx T .

## 3.4 Algorithm's Analysis

Next we prove Theorem 3.3. Momentarily we shall argue that Algorithm 2 repeats for at most T = log 7 / 3 ( ( 1 1 -α ) R 2 640 m ) iterations. Based on Fact 2.1 and on the bound on the number of iterations, it is simple to argue that the following event holds w.p. ≥ 1 -ξ :

<!-- formula-not-decoded -->

which follows from the fact that in each iteration the upper bound on the largest eigenvalue of Σ is at most κ = R 2 . We continue our analysis conditioning on E holding.

The analysis begins with the following lemma, that shows that under E we have that in each iteration the eigenvalues of Σ decrease. Its proof is very similar to the proof in Kamath et al. (2019) and so it is deferred to Appendix C.

Lemma 3.4. Given X = { X 1 , ..., X n } ⊂ R d , C &gt; 0 , c &gt; 0 , 0 &lt; η &lt; 1 , 0 &lt; ψ &lt; 1 , and κ ≥ 1 s.t. I ⪯ Σ ⪯ κI where Σ = 1 n XX T . Let V ← Span ( { v i : λ i ≥ ψκ } ) of the largest eigenvalues of the noisy ˜ Σ and let Π = η Π V +Π V ⊥ . Given that: n ≥ Ω ( m √ dT ρ ln( T / ξ ) ) Then:

<!-- formula-not-decoded -->

In particular, if κ &gt; C for some C then: I ⪯ 1 (1 -1 η 2 ψ -c ) ΠΣΠ ⪯ η 2 + ψ +2 c (1 -1 C ( η 2 ψ -c ) ) κI .

Corollary 3.5. Given X = { X 1 , ..., X n } ⊂ R d and κ ≥ 1 s.t. I ⪯ Σ ⪯ κI where Σ = 1 n XX T . Let V, Π be as in Algorithm 2, and set η = 1 / 2 , ψ = 1 / 10 m , c = 1 / 80 m and C = 640 m in Lemma 3.4. Then w.p. ≥ 1 -ξ :

In particular, if κ &gt; C then: I ⪯ 8 7 ΠΣΠ ⪯ 3 7 κI .

<!-- formula-not-decoded -->

Based on Corollary 3.5 we can bound the number of iterations of the algorithm.

Corollary 3.6. Algorithm 2 has T = log 7 / 3 ( ( 1 1 -α ) R 2 640 m ) iterations.

Proof. The algorithm halts when ( 3 7 ) T ( 1 1 -α ) κ ≤ C = 640 m so: T ≥ log 3 / 7 ( 640 m ( 1 1 -α ) κ ) = log 7 / 3 ( ( 1 1 -α ) R 2 640 m ) .

1 Note that the privacy of Algorithm 1 holds for any input X with bounded L 2 -norm, regardless of X being subsamplable or not.

<!-- formula-not-decoded -->

With Π reducing the largest eigenvalues of Σ from κ to 3 κ / 7 , we now proceed and bound the radius of all datapoints by from R to √ 3 / 7 R . This is where our analysis diverges from the analysis in Kamath et al. (2019). Whereas Kamath et al rely on the underlying Gaussian distribution to argue they have no outliers, we have to deal with outliers. For our purpose, a datapoint x is an outlier if the shrinking function S (Step 10 in Algorithm 2) reduces the norm of Π x since ∥ Π x ∥ &gt; √ 3 / 7 R . In the following claims we argue that all outliers lie in P tail (Definition 3.2), and moreover, that by shrinking the outliers we do not alter the second moment matrix all too much. We begin by arguing that there aren't too many outliers.

Claim 3.7. Analogously to Definition 3.2, fix any m ′ ≥ m and define P tail ( m ′ ) = { x ∈ X : ∃ u ∈ R d : ( x T u ) 2 &gt; m ′ (1 + α ) 1 n ∑ i ( x T i u ) 2 } . Then it holds that

<!-- formula-not-decoded -->

Proof sketch. The proof applies the ( m,α,β ) -subsamplability property: if a point violates the bound, it would contradict subsamplability with non-negligible probability. A simple union bound and tail approximation then yield the claimed bound. Full details are deferred to Appendix C.2.

Lemma 3.8. Let X be a ( m,α,β ) -subsamplable with β ≤ α 4(1+ α ) log( R 2 (1+ α ) m ) . Let P = X \ P tail .

Then:

Proof sketch. We partition the tail points according to the magnitude of their contribution and apply Claim 3.7 to bound the measure of each bucket. Summing across buckets shows that the overall loss from removing the tail points is small. Full proof is deferred to Appendix C.3.

<!-- formula-not-decoded -->

Lemma 3.9. At each iteration t of Algorithm 2, only points belonging to P tail are subjected to shrinking, given that α &lt; 1 / 2 , ψ = 1 10 m and c = 1 80 m .

Proof sketch. The proof uses induction over iterations. Shrinking happens only if a point's mass in a low-eigenvalue subspace is too large. By carefully tracking how shrinking operates and applying Weyls theorem and Lemma 3.8, we show that only initially bad points (i.e., those in P tail) can cause such violations. Full proof appears in Appendix C.4.

Corollary 3.10. In all iterations of the algorithm it holds that Σ ⪰ I , namely, that the least eigenvalue of the second moment matrix of the input is ≥ 1 .

Proof sketch. Weargue by induction that removing or shrinking tail points preserves a spectral lower bound. Using Lemma 3.8 and the shrinkage structure from Lemma 3.9, the transformation at each step maintains the least eigenvalue above 1 . Full proof is provided in Appendix C.5.

Proof of Theorem 3.3. First we argue that Algorithm 1 is ρ -zCDP. Given two neighboring data sets X , X ′ of size n which differ in that one contains X i and the other contains X ′ i , the covariance matrix of these two data sets can change in Spectral norm by at most:

<!-- formula-not-decoded -->

Since Algorithm 1 invokes T calls to Algorithm 2 each preserving ρ / T -zCDP, thus the privacy guarantee of Algorithm 1 follows from sequential composition of zCDP.

We now turn to proving the algorithm's utility. From Claim 3.7 we conclude that | P tail | is indeed at most ( β + β 2 m ) -fraction of | X | . We prove by recursion that: (1 -γ )Σ eff ⪯ ˜ Σ ⪯ (1 + γ )Σ .

Stopping Rule: Let X T be the input at the final iteration T and let P = X \ P tail . Denote Σ( · ) as the second moment matrix operator. We know that throughout the algorithm, the points from P were not shrunk. Moreover, Corollary 3.10 assures that the least eigen value of Σ( X ) is ≥ 1 . Additionally, our bound on n yields that when κ ≤ C then the noise matrix N we add satisfy that ∥ N ∥ 2 ≤ γ w.p. ≥ 1 -ξ . It thus follows that (1 -γ )Σ( X T ) ⪯ Σ+ N ⪯ (1+ γ )Σ( X T ) as required.

Recursive Step: Let X t be the input at iteration t ≤ T . Then, by Lemma 3.4, we have: I ⪯ Σ ( 8 7 Π X t ) ⪯ 3 7 κI . Lemma 3.9 ensures that S (Π X t ) shrinks only points from P tail and so Corollary 3.10 assures that the eigenvalue is ≥ 1 throughout the recursive iterations. Hence, by the inductive hypothesis, our recursive call returns Σ rec such that:

which implies:

<!-- formula-not-decoded -->

Proving the required for any intermediate iteration of Algorithm 2.

## 4 Applications: Coping with Outliers

## 4.1 Input Drawn from 'Nice' Distributions

First, we show our algorithm returns an approximation of the 2nd-moment matrix when the input is drawn from a distribution D . Throughout this section, we apply the Matrix-Bernestein Inequality.

<!-- formula-not-decoded -->

Fact 4.1. Let Z be the sum of m i.i.d. matrices Z = ∑ i Z i , whose mean is 0 and have norm bounded by ∥ Z i ∥ ≤ R almost surely. Then, denoting σ 2 = ∥ E [ ZZ T ] ∥ , it holds that

We can apply Fact 4.1 above to measure how well the sample covariance estimator approximates the true covariance matrix of a general distribution using the following claim (proof deferred to Appendix C.6.)

Claim 4.2. Let D be a distribution on R d with a finite second moment Σ . Consider a random vector y chosen by drawing x ∼ D and then multiplying y = Σ -1 / 2 x . Suppose that ∥ y ∥ ≤ M 1 a.s. that we also have a bound ∥ E [( y T y ) yy T ] ∥ ≤ M 2 . Fix α, β &gt; 0 . If we draw m = max { 2 M 2 α 2 , 2(1+ M 2 1 ) 3 α } ·

<!-- formula-not-decoded -->

ln( 4 d / β ) examples from D and compute the empirical second moment matrix ˆ Σ , then w.p. ≥ 1 -β it holds that

Recall that we (1 ± γ ) -approximate the 2nd-moment matrix of the input w.p. ≥ 1 -ξ . Thus, we need the input itself to be a (1 ± γ ) -approximation of the 2nd-moment matrix of the distribution. (We can then apply Fact A.3 to argue we get a 1 ± O ( γ ) approximation of the distribution's second moment matrix.) This means our algorithm requires

<!-- formula-not-decoded -->

for α = 1 / 2 and β = O ( 1 log( R/λ min ) ) in order to return a (1 ± O ( γ )) -approximation of the 2nd moment of the distribution w.p. ≥ 1 -O ( ξ ) . In Appendix D we give concrete examples of distributions for which this bound is applicable, including (bounded) heavy-tail distributions.

## 4.2 Distributional Input with Outliers

Next, we consider an application to our setting, in which we take some well-behaved distribution D and add to it outliers. Consider D to be a distribution that for any γ, ξ &gt; 0 is m ( γ, ξ ) -subsamplable for m = O ( d ln( d / ξ ) γ 2 ). We consider here inputs that are composed of (1 -η ) -fraction of good points and η -fraction of outliers. We thus denote the second moment matrix of the input as

<!-- formula-not-decoded -->

Weassume throughout that the least eigenvalue of Σ D is λ min . Our goal is to return, w.h.p. ( ≥ 1 -ξ ) an approximation of Σ D using a DP algorithm.

<!-- formula-not-decoded -->

Inapplicability of Brown et al. (2023). The work of Brown et al. (2023) shows that if the input has λ -bounded leverage scores, namely, if ∀ x, x T ( 1 n XX T ) -1 x ≤ λ , then they recover the second moment of the input with O ( λ √ d ϵ ) overhead to the sampling complexity. However, in this case one can set outliers so that their leverage scores is R 2 / λ min (provided the input has L 2 -norm bound of R ). We argue that the algorithm of Brown et al. (2023) is unsuited for such a case. Indeed, the algorithm of Brown et al. (2023) has an intrinsic 'counter' of outliers (referred to as score), which when reached O (1 /ϵ ) causes the algorithm to return 'Failure'. 2 So either it holds that η is so small that the overall number of outliers is a constant (namely, ηn = O ( 1 / ϵ ) ), or we set the bound on the leverage scores to be R 2 /λ min and suffer the cost in sample complexity.

A Private Learner. Suppose η is very small. In this case we can simply take some off-the-shelf ( ϵ, δ ) -DP algorithm with sample complexity m ( γ, ξ, ϵ, δ ) that approximates the second moment matrix, and run in over a subsample of m points out of that input. In order for this to work we require that η would be smaller than O ( ξ m ( γ,ξ,ϵ,δ ) ) , so that a subset of size m would be clean of any outliers.

Subsample and Aggregate. The framework of Subsample and Aggregate (Nissim et al., 2007) is in a way a 'perfect fit' for the problem: we subsample t datasets of size m ( γ, ξ ) each, and then wisely aggregate the (majority of the) t results into one. However, in order for this to succeed, it is required that most of the t subsamples are clean of outliers. In other words, we require that the probability of a dataset to be clean ought to be &gt; 1 / 2 , namely (1 -η ) m ( γ,ξ ) &gt; 1 / 2 or alternatively that η = O ( 1 m ( γ,ξ ) ) , which in our case means η = O ( γ 2 d log( d/ξ ) ) . Weanalyze this paradigm as part of the subsample-and-aggregate baseline we establish (Appendix A), and the subsample-and-aggregate baseline requires

<!-- formula-not-decoded -->

in order to return a (1 ± O ( γ )) -approximation of the 2nd moment of the distribution w.p. ≥ 1 -O ( ξ ) .

Our Work. Our work poses an alternative to the above mentioned techniques. Rather than having n &lt; 1 m ( γ,ξ ) , we have a slightly more delicate requirement. We require that there exists α = 1 / 2 and β ≤ 1 12 log( R λ min ) such that η = O ( β m ( α, β 2 ) ) . (In particular, for the given D it implies that we require that η = O ( 1 d log( d ) log( R/λ min ) ) , which is considerably higher value than in the case of subsample and aggregate discussed above.) This way, we can argue that w.p. ≥ 1 -β 2 it holds that a subsample of size m ( α, β 2 ) contains only points from D and that w.p. ≥ 1 -β 2 that sample is 'good' in the sense that its empirical second moment satisfy ˆ Σ ≈ Σ D .

However, we also require that the subsample of size m ( α, β 2 ) would satisfy that its empirical second moment matrix ˆ Σ satisfies that (1 -1 2 )Σ ⪯ ˆ Σ ⪯ (1 + 1 2 )Σ since we set α = 1 2 . As ˆ Σ ≈ Σ D it follows that it suffices to require that

<!-- formula-not-decoded -->

Some arithmetic shows that the upper bound is easily satisfied when η 1 -η ≤ 1 8 (which clearly holds for our value of η ), yet the lower bound requires that we have

<!-- formula-not-decoded -->

Under these two conditions, our work returns w.p. 1 -O ( ξ ) a matrix ˜ Σ that satisfies that ˜ Σ ⪰ (1 -O ( γ ))Σ D , with sample complexity of

<!-- formula-not-decoded -->

2 Moreover, in their algorithm, this 'score' intrinsically cannot be greater than k = O ( 1 / ϵ ) as they use a particular bound of the form e k/ϵ .

## Acknowledgments and Disclosure of Funding

O.S. is supported by the BIU Center for Research in Applied Cryptography and Cyber Security in conjunction with the Israel National Cyber Bureau in the Prime Ministers Office, and by ISF grant no. 2559/20. Both authors thank the anonymous reviewers for their suggestions and advice on improving this paper.

## References

- Aden-Ali, I., Ashtiani, H., and Kamath, G. On the sample complexity of privately learning unbounded high-dimensional gaussians. In Feldman, V., Ligett, K., and Sabato, S. (eds.), Proceedings of the 32nd International Conference on Algorithmic Learning Theory , volume 132 of Proceedings of Machine Learning Research , pp. 185-216. PMLR, 16-19 Mar 2021. URL https://proceedings.mlr.press/v132/aden-ali21a.html .
- Alabi, D., Kothari, P. K., Tankala, P., Venkat, P., and Zhang, F. Privately estimating a gaussian: Efficient, robust, and optimal. In Proceedings of the 55th Annual ACM Symposium on Theory of Computing , STOC 2023, pp. 483496, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9781450399135. doi: 10.1145/3564246.3585194. URL https://doi.org/ 10.1145/3564246.3585194 .
- Amin, K., Dick, T., Kulesza, A., Munoz, A., and Vassilvitskii, S. Differentially private covariance estimation. In Wallach, H., Larochelle, H., Beygelzimer, A., d'Alché-Buc, F., Fox, E., and Garnett, R. (eds.), Advances in Neural Information Processing Systems , volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper\_files/paper/2019/ file/4158f6d19559955bae372bb00f6204e4-Paper.pdf .
- Ashtiani, H. and Liaw, C. Private and polynomial time algorithms for learning gaussians and beyond. In Loh, P.-L. and Raginsky, M. (eds.), Proceedings of Thirty Fifth Conference on Learning Theory , volume 178 of Proceedings of Machine Learning Research , pp. 1075-1076. PMLR, 02-05 Jul 2022. URL https://proceedings.mlr.press/v178/ashtiani22a.html .
- Biswas, S., Dong, Y., Kamath, G., and Ullman, J. Coinpress: Practical private mean and covariance estimation. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems , volume 33, pp. 14475-14485. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/ file/a684eceee76fc522773286a895bc8436-Paper.pdf .
- Brown, G., Hopkins, S., and Smith, A. Fast, sample-efficient, affine-invariant private mean and covariance estimation for subgaussian distributions. In Neu, G. and Rosasco, L. (eds.), Proceedings of Thirty Sixth Conference on Learning Theory , volume 195 of Proceedings of Machine Learning Research , pp. 5578-5579. PMLR, 12-15 Jul 2023. URL https://proceedings.mlr.press/ v195/brown23a.html .
- Bun, M. and Steinke, T. Concentrated differential privacy: Simplifications, extensions, and lower bounds. Cryptology ePrint Archive, Paper 2016/816, 2016. URL https://eprint.iacr.org/ 2016/816 .
- Dong, W., Liang, Y., and Yi, K. Differentially private covariance revisited. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), Advances in Neural Information Processing Systems , volume 35, pp. 850-861. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/file/ 057405fd73dd7ba7f32a7cb34fb7c7f5-Paper-Conference.pdf .
- Dwork, C., McSherry, F., Nissim, K., and Smith, A. Calibrating noise to sensitivity in private data analysis. In Halevi, S. and Rabin, T. (eds.), Theory of Cryptography , pp. 265-284, Berlin, Heidelberg, 2006. Springer Berlin Heidelberg. ISBN 978-3-540-32732-5.
- Dwork, C., Talwar, K., Thakurta, A., and Zhang, L. Analyze gauss: optimal bounds for privacypreserving principal component analysis. In Shmoys, D. B. (ed.), Symposium on Theory of Computing, STOC 2014, New York, NY, USA, May 31 - June 03, 2014 , pp. 11-20. ACM, 2014.

- Kamath, G., Li, J., Singhal, V., and Ullman, J. Privately learning high-dimensional distributions. In Beygelzimer, A. and Hsu, D. (eds.), Proceedings of the Thirty-Second Conference on Learning Theory , volume 99 of Proceedings of Machine Learning Research , pp. 1853-1902. PMLR, 25-28 Jun 2019. URL https://proceedings.mlr.press/v99/kamath19a.html .
- Kamath, G., Mouzakis, A., Singhal, V., Steinke, T., and Ullman, J. A private and computationallyefficient estimator for unbounded gaussians. In Loh, P.-L. and Raginsky, M. (eds.), Proceedings of Thirty Fifth Conference on Learning Theory , volume 178 of Proceedings of Machine Learning Research , pp. 544-572. PMLR, 02-05 Jul 2022. URL https://proceedings.mlr.press/ v178/kamath22a.html .
- Kothari, P., Manurangsi, P., and Velingker, A. Private robust estimation by stabilizing convex relaxations. In Loh, P.-L. and Raginsky, M. (eds.), Proceedings of Thirty Fifth Conference on Learning Theory , volume 178 of Proceedings of Machine Learning Research , pp. 723-777. PMLR, 02-05 Jul 2022. URL https://proceedings.mlr.press/v178/kothari22a.html .
- Mahpud, B. and Sheffet, O. A differentially private linear-time fptas for the minimum enclosing ball problem. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), Advances in Neural Information Processing Systems , volume 35, pp. 31640-31652. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/ file/cd830afc6208a346e4ec5caf1b08b4b4-Paper-Conference.pdf .
- Nissim, K. and Stemmer, U. Clustering algorithms for the centralized and local models. In Janoos, F., Mohri, M., and Sridharan, K. (eds.), Proceedings of Algorithmic Learning Theory , volume 83 of Proceedings of Machine Learning Research , pp. 619-653. PMLR, 07-09 Apr 2018. URL https://proceedings.mlr.press/v83/nissim18a.html .
- Nissim, K., Raskhodnikova, S., and Smith, A. D. Smooth sensitivity and sampling in private data analysis. In Johnson, D. S. and Feige, U. (eds.), Proceedings of the 39th Annual ACM Symposium on Theory of Computing, San Diego, California, USA, June 11-13, 2007 , pp. 75-84. ACM, 2007.
- Nissim, K., Stemmer, U., and Vadhan, S. Locating a small cluster privately. In Proceedings of the 35th ACM SIGMOD-SIGACT-SIGAI Symposium on Principles of Database Systems , SIGMOD/PODS16, pp. 413427. ACM, June 2016. doi: 10.1145/2902251.2902296. URL http: //dx.doi.org/10.1145/2902251.2902296 .
- Tao, T. Topics in Random Matrix Theory . Graduate studies in mathematics. American Mathematical Soc., 2012. ISBN 9780821885079. URL https://books.google.co.il/books?id=Hjq\_ JHLNPT0C .

## A Baseline

In this section, we provide a baseline for the problem of 2nd-moment estimation using subsample and aggregate framework (Nissim et al. (2007)).

In this baseline, we work with the following notion of a convex semimetric space. The key property to keep in mind is that for semimetric spaces, we only have an approximate triangle inequality, as long as the points are significantly close to one another.

Definition A.1. Let Y be a convex set and let dist : Y × Y → R ≥ 0 . We say ( Y , dist ) is a convex semimetric space if there exist absolute constants t ≥ 1 , ϕ ≥ 0 , and r &gt; 0 such that for every k ∈ N and every Y, Y 1 , Y 2 , . . . , Y k ∈ Y , the following conditions hold:

1. dist ( Y, Y ) = 0 and dist ( Y 1 , Y 2 ) ≥ 0 .
2. Symmetry. dist ( Y 1 , Y 2 ) = dist ( Y 2 , Y 1 ) .
3. t -approximate r -restricted triangle inequality. If both dist ( Y 1 , Y 2 ) , dist ( Y 2 , Y 3 ) ≤ r , then

<!-- formula-not-decoded -->

4. Convexity. For all α ∈ ∆ k ,

<!-- formula-not-decoded -->

5. ϕ -Locality. For all α, α ′ ∈ ∆ k ,

<!-- formula-not-decoded -->

where ∆ k denotes the k -dimensional probability simplex. When r is unspecified, we take it to mean r = ∞ and refer to it as a t -approximate triangle inequality.

The following technical lemma (whose proof appears in Appendix C in Ashtiani &amp; Liaw (2022)) is helpful for learning second moment matrices.

Lemma A.2. Let S d be the set of all d × d positive definite matrices. For A,B ∈ S d , let

<!-- formula-not-decoded -->

Then ( S d , dist ) is a convex semimetric which satisfies a ( 3 / 2 ) -approximate 1 -restricted triangle inequality and 1 -locality.

Based on Lemma A.2, the following distance function forms a semimetric space for positive definite matrices:

<!-- formula-not-decoded -->

Fact A.3. Let A,B be d × d matrices and suppose that ∥ A -1 / 2 BA -1 / 2 -I ∥ ≤ γ ≤ 1 / 2 . Then ∥ B -1 / 2 AB -1 / 2 -I ∥ ≤ 4 γ

## Algorithm 3 Baseline DP Second Moment Estimation

Input: a set of n points X ⊆ R d , subsamplability parameters m,α,β , error parameter ξ ∈ (0 , 1) , privacy parameters ϵ, δ .

- 1: Randomly split X into T = ⌊ n / m ⌋ subgroups X 1 , . . . , X T of size m .
- 3: Σ t ← 1 m X t X T t
- 2: for t ∈ [ T ] do
- 4: end for
- 5: for t ∈ [ T ] do
- 7: end for
- 6: q t ← 1 T |{ t ′ ∈ [ T ] : dist (Σ t , Σ t ′ ) ≤ 2 α 1 -α }|
- 10: ˜ Q ← Q + Z
- 8: Q ← 1 T ∑ t ∈ [ T ] q t 9: Z ∼ TLap ( 2 / T , ϵ, δ )
- 11: if ˜ Q &lt; 0 . 8 + 2 Tϵ ln(1 + e ϵ -1 2 δ ) then
- 13: end if
- 12: fail and return ⊥
- 14: for t ∈ [ T ] do
- 16: end for
- 15: w t = min(1 , 10 max(0 , q t -0 . 6))
- 19: η ← α 48 C ( √ d + √ ln( 4 / β )) { C some large constant}
- 17: ˆ Σ ← ∑ t ∈ [ T ] w t Σ t / ∑ t ∈ [ T ] w t 18: N ∼ N (0 , 1) d × d
- 20: return ˜ Σ = ˆ Σ 1 / 2 ( I + ηN )( I + ηN ) T ˆ Σ 1 / 2

Lemma A.4. (Utility Analysis) Let Σ = 1 n XX T and set η = α 48 C ( √ d + √ ln( 4 / ξ )) for a sufficiently large constant C &gt; 0 . Then w.p. ≥ 1 -ξ Algorithm 3 returns ˜ Σ such that dist (Σ , ˜ Σ) ≤ 2 α given that:

<!-- formula-not-decoded -->

Proof. Indeed, we have that

<!-- formula-not-decoded -->

ˆ Σ

where we used the fact that ∥ N ∥ ≤ C ( √ d + √ ln( 4 / ξ )) w.p. ≥ 1 -ξ / 2 . Applying η as defined in the lemma gives that:

∥ Following from Fact A.3 we have that:

So we have that dist ( ˜ Σ , ˆ Σ) ≤ α / 3 .

-

1

/

2

˜

Σ ˆ Σ

-

1

/

2

-

I

∥ ≤

α

/

12

<!-- formula-not-decoded -->

Now we show that dist (Σ , ˆ Σ) ≤ α w.p. ≥ 1 -ξ / 2 :

Based on the subsamplability assumption, we know that w.p. ≥ (1 -β ) T :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It means that w.p. ≥ (1 -β ) T all t = t ′ ∈ [ T ] satisfy:

̸

Hence w.p. ≥ (1 -β ) T ≥ e -2 βn m ≥ 1 -ξ / 2 all t ∈ [ T ] satisfy:

<!-- formula-not-decoded -->

Finally, Q = 1 &gt; 0 . 8 + 2 Tϵ ln(1 + e ϵ -1 2 δ ) w.p. ≥ 1 -ξ / 2 and therefore the algorithm does not fail w.p. ≥ 1 -ξ .

Now obviously dist (Σ , ˆ Σ) ≤ α since ˆ Σ is a weighted average of Σ t .

So we have that dist ( ˜ Σ , ˆ Σ) ≤ α / 3 and dist (Σ , ˆ Σ) ≤ α w.p. ≥ 1 -ξ . Applying the 3 / 2 -approximate triangle inequality for the dist function, we get:

<!-- formula-not-decoded -->

Lemma A.5. (Privacy Analysis) Suppose that:

<!-- formula-not-decoded -->

Then Algorithm 3 is (2 ϵ, 4 e ϵ δ ) -DP.

Proof. By basic composition of Truncated Laplace Mechanism and Lemma 3.6 of Ashtiani &amp; Liaw (2022).

Both Lemma A.4 and Lemma A.5 together imply the following theorem:

Theorem A.6. Let ξ , ϵ , δ be parameters, and let X ⊆ R d be a ( m,α,β ) -subsamplable set of n points. Then, for Algorithm 3, the following properties hold:

1. Algorithm 3 satisfies (2 ϵ, 4 e ϵ δ ) -Differential Privacy.
2. The algorithm returns ˜ Σ such that dist (Σ , ˜ Σ) ≤ 2 α , where Σ = 1 n XX ⊤ .

These guarantees hold under the following conditions:

1. The dataset size satisfies:

<!-- formula-not-decoded -->

where η = α 48 C ( √ d + √ ln( 4 / ξ )) for a sufficiently large constant C &gt; 0 .

2. The subsamplability parameters satisfy m ≥ 2 βn ξ .

## B Finding Initial Parameters

Our recursive algorithm requires as input two parameters that characterize the 'aspect ratio' of the input, namely -R max , the maximum distance of any point from the origin, and λ min , the minimum eigenvalue of the input. These two parameters give us the initial bounds, as they imply that λ min I ⪯ Σ ⪯ R 2 max I . We detail here the algorithms that allow us to retrieve these parameters. Finding R max is fairly simple, as it requires us only to apply off-the-shelf algorithms that find an enclosing ball of the n input points (Nissim et al. (2016); Nissim &amp; Stemmer (2018); Mahpud &amp; Sheffet (2022)). Finding the minimal eigenvalue is fairly simple as well, and we use a subsampleand-aggregate framework. Details follow.

## Finding a Covering Radius of The Data

Definition B.1. A 1-cluster problem ( X d , n, t ) consists of a d -dimensional domain X d and parameters n ≥ t . We say that algorithm M solves ( X d , n, t ) with parameters (∆ , ω, β ) if for every input database S ∈ ( X d ) n it outputs, with probability at least 1 -β , a center c and a radius r such that: (i) the ball of radius r around c contains at least t -∆ points from S ; and (ii) r ≤ w · r opt, where r opt is the radius of the smallest ball in X d containing at least t points from S .

Theorem B.2 (Nissim &amp; Stemmer (2018)) . Let n, t, β, ϵ, δ be s.t.

<!-- formula-not-decoded -->

There exists an ( ϵ, δ ) -differentially private algorithm that solves the 1 -cluster problem X d , n, t with parameters (∆ , ω ) and error probability β , where ω = O (1) and

<!-- formula-not-decoded -->

In words, there exists an efficient ( ϵ, δ ) -differentially private algorithm that (ignoring logarithmic factors) is capable of identifying a ball of radius O ( r opt ) containing t -˜ O ( n 0 . 1 ϵ ) points, provided that t ≥ ˜ O ( n 0 . 1 · √ d / ϵ ) .

Finding Minimal Eigenvalue The algorithm described in Kamath et al. (2022) (Section 3) privately estimates all eigenvalues of the second moment matrix of the data. However, for the purpose of this study, we focus solely on identifying the minimum eigenvalue while maintaining the privacy guarantees provided by the algorithm. To adapt the algorithm, we modify its structure to prioritize the computation of the minimum eigenvalue directly, rather than estimating the full spectrum of eigenvalues. This simplification not only reduces computational overhead but also aligns with the specific objectives of our work. Below, we detail the adjusted methodology and highlight the changes made to the original theorem.

## Algorithm 4 DP Minimum Eigenvalue Estimator

Input: a set of n points X ⊆ R d , subsamplability parameters m,α,β , error parameter ξ ∈ (0 , 1) , privacy parameters ϵ, δ .

- 1: Randomly split X into T = ⌊ n / m ⌋ subgroups X 1 , . . . , X T of size m .
- 3: Let λ t min be the minimum eigenvalue of 1 m X t X T t .
- 2: for t ∈ [ T ] do
- 4: end for
- 5: Ω ←{ . . . , [(1 -α ) 2 , 1 -α ) , [1 -α, 1) , [1 , 1 1 -α ) , [ 1 1 -α , 1 (1 -α ) 2 ) , . . . } ∪ { [0 , 0] } .
- 7: Run ( ϵ, δ ) -DP histogram on all λ t min over Ω .
- 6: Divide [0 , ∞ ) into Ω .
- 8: if no bucket is returned then
- 9: return ⊥ .
- 11: Let [ ℓ, u ] be a non-empty bucket returned and set ˜ λ min ← ℓ .
- 10: end if
- 12: return ˜ λ min

Theorem B.3. ((Differentially Private EigenvalueEstimator) from Kamath et al. (2022)) For every ϵ, δ, ξ &gt; 0 , the following properties hold for Algorithm 4:

1. The algorithm is ( ϵ, δ ) -differentially private.
2. The algorithm runs in time poly ( n / m , ln( 1 / ϵξ ))
3. if:

then it outputs ˜ λ min such that with probability at least 1 -ξ , ˜ λ min ∈ [(1 -α ) λ min , (1 + α ) λ min ] where λ min is the minimum eigenvalue of 1 n XX T .

<!-- formula-not-decoded -->

Proof. Privacy and running time is proven by the theorem of stability-based private histograms (See Lemma 2.6 in Kamath et al. (2022)). Now, we move on to the accuracy guarantees. By subsamplability, with probability at least 1 -ξ / 2 , the non-private estimates of λ min must be within a factor of (1 ± α ) due to our subsample complexity. Therefore, at most two consecutive buckets would be filled with λ t min s. Due to our sample complexity and private histograms utility, those buckets are released with probability at least (1 -ξ / 2 ) , which proves our theorem.

## C Missing Proofs

Lemma C.1. (Lemma 3.4 restated.) Given X = { X 1 , ..., X n } ⊂ R d , C &gt; 0 , c &gt; 0 , 0 &lt; η &lt; 1 , 0 &lt; ψ &lt; 1 , and κ ≥ 1 s.t. I ⪯ Σ ⪯ κI where Σ = 1 n XX T . Let V ← Span ( { v i : λ i ≥ ψκ ) and Π ← η Π V +Π V ⊥ . Given that: n ≥ O ( m √ dT ρ ln( T / ξ ) ) Then:

<!-- formula-not-decoded -->

In particular, if κ &gt; C for some constant C then:

<!-- formula-not-decoded -->

Proof. First we prove the upper bound. Note that ∥ ΠΣΠ ∥ 2 = ∥ Π( ˜ Σ -N )Π ∥ ≤ ∥ Π ˜ ΣΠ ∥ + ∥ N ∥ . So we bound the two terms separately. Using Fact 2.1 with A = O ( 1 d ln( 1 / ξ )) for sufficiently large n we get ∥ N ∥ 2 ≤ cκ w.p. 1 -ξ / 2 . Additionally ∥ Π ˜ ΣΠ ∥ 2 ≤ η 2 ∥ Π V ˜ ΣΠ V ∥ + ∥ Π V ⊥ ˜ ΣΠ V ⊥ ∥ ≤ η 2 ( κ + c ) + ψκ ≤ ( η 2 + ψ + c ) κ . So overall,

<!-- formula-not-decoded -->

Next we prove the lower bound. Let u ∈ S d -1 . Our lower bound requires we show that u T ΠΣΠ u ≥ (1 -1 η 2 ψ -c · 1 κ ) . We consider two cases:

- Case 1: ∥ Π V u ∥ 2 &lt; 1 η 2 ψ -c · 1 κ . Since ∥ Π V ⊥ u ∥ 2 + ∥ Π V u ∥ 2 = 1 we have ∥ Π V ⊥ u ∥ 2 &gt; (1 -1 η 2 ψ -c · 1 κ ) fact that Σ ⪰ I we have that

, hence, using the

<!-- formula-not-decoded -->

- Case 2: ∥ Π V u ∥ 2 ≥ 1 η 2 ψ -c · 1 κ . Note that u T ΠΣΠ u = u T Π( ˜ Σ -N )Π u = u T Π ˜ ΣΠ u -u T Π N Π u . term separately. We know that

<!-- formula-not-decoded -->

- So we bound each

Additionally, based on the bound on the spectral norm of N we have that u T Π N Π u ≤ cκ ∥ Π u ∥ 2 ≤ cκ ∥ Π V u ∥ 2 . So overall:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Claim C.2. (Claim 3.7 restated.) Analogously to Definition 3.2, fix any m ′ ≥ m and define P tail ( m ′ ) = { x ∈ X : ∃ u ∈ R d : ( x T u ) 2 &gt; m ′ (1 + α ) 1 n ∑ i ( x T i u ) 2 } . Then it holds that

Proof. Let x ∈ X be a datapoint and ˆ X = { ˆ X 1 , . . . , ˆ X m ′ } be a subsample of m ′ points i.i.d from X . From subsamplability we know that: Pr[(1 -α )Σ ⪯ ˆ Σ ⪯ (1+ α )Σ] ≥ 1 -β where Σ = 1 n ∑ x ∈ X xx T

<!-- formula-not-decoded -->

and ˆ Σ = 1 m ′ ∑ i ∈ [ m ′ ] ˆ X i ˆ X T i . In other words:

Clearly, if even a single x 0 ∈ P tail ( m ′ ) belongs to ˆ X then we'd have that for some direction u we have

<!-- formula-not-decoded -->

contradicting subsamplability. It follows that w.p. ≥ 1 -β no point in ˆ X belongs to P tail ( m ′ ) . Thus, if we denote p = Pr x ∈ R X [ x ∈ P tail ( m ′ )] then we get that 1 -β ≤ (1 -p ) m ′ . Using known inequalities

<!-- formula-not-decoded -->

Lemma C.3. (Lemma 3.8 restated.) Let X be a ( m,α,β ) -subsamplable with β ≤ α 4(1+ α ) log( R 2 (1+ α ) m ) . Let P = X \ P tail . Then:

<!-- formula-not-decoded -->

Assume that P tail holds a p -fraction of X , and denote also λ tail = 1 | P tail | ∑ x ∈ P tail ⟨ x, u ⟩ 2 . Hence:

Proof. Fix direction u , and denote λ P = 1 n ∑ x ∈ P ⟨ x, u ⟩ 2 and λ = 1 n ∑ x ∈ X ⟨ x, u ⟩ 2 . Our goal is to prove P

<!-- formula-not-decoded -->

Now split the interval [(1 + α ) mλ,R 2 ] , the interval of P tail, into buckets B 0 , B 1 , B 2 , ..., B k where that λ -λ ≤ αλ .

which implies that our goal is to prove that pλ tail ≤ αλ .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

.

k ⋃ i =0 B i ] . Recall that Claim 3.7 implies that for any i we can set m ′ = 2 i m and get:

Now we can bound pλ P tail using B i :

<!-- formula-not-decoded -->

which is upper bounded by λ for β ≤ α 4(1+ α ) k

Lemma C.4. (Lemma 3.9 restated.) At each iteration t of Algorithm 2, only points belonging to P tail are subjected to shrinking, given that α &lt; 1 / 2 , ψ = 1 10 m and c = 1 80 m .

Proof. The proof works by induction on the iterations of the algorithm. Clearly, at t = 0 , before the algorithm begins, no points were subjected to shrinking so the argument is vacuously correct.

Let t ≤ T and let x ∈ X t denote a point that undergoes shrinking for the first time at iteration t , with X t denoting the input of the t -th time we apply Algorithm 2. Our goal is to show that x stems from a point in P tail. Since x hasn't been shrunk prior to iteration t , then there exists some x i in the original input such that x = Λ t x i for the linear transformation Λ t = Π t · Π t -1 · ... · Π 1 . Our goal is to show that x i ∈ P tail .

We assume x is shrunk at iteration t . This shrinking happens since x T Π V ⊥ x &gt; 3 7 R 2 . With α ≤ 1 2 we have 3 7 ≥ 1+ α 4 , then it follows that x satisfies:

<!-- formula-not-decoded -->

For the given parameters ψ and c , observe that:

<!-- formula-not-decoded -->

Recall that V ⊥ denotes the subspace spanned by all eigenvectors of ˜ Σ = Σ + N corresponding to eigenvalues ≤ ψκ and that ∥ N ∥ ≤ cκ . Now denote U ⊥ the subspace spanned by all eigenvectors of Σ corresponding to eigenvalues ≤ ( ψ + c ) κ . By Weyl's theorem it holds that

<!-- formula-not-decoded -->

Thus we infer the existence of u ∈ U ⊥ (unit length vector in the direction Π U ⊥ x ) such that:

<!-- formula-not-decoded -->

But as U ⊥ is spanned by all eigenvalues ≤ ( ψ + c ) κ of Σ then it holds that 1 n ∑ x ∈ X t ( x T u ) 2 ≤ ( ψ + c ) κ , so

<!-- formula-not-decoded -->

Now recall that x undergoes shrinking for the first time at iteration t . That means that there exists some x i in the original input such that x = Λ t x i for the linear transformation Λ t = Π t · Π t -1 · ... · Π 1 . Moreover, by the induction hypothesis all points that were shrunk upto iteration t are from P tail. So for any point z ∈ X t \ P tail it holds that z = Λ t y for the corresponding y in the original input X . We get that 1 n ∑ x ∈ X t ( x T u ) 2 ≥ 1 n ∑ x ∈ X t \ P tail ( x T u ) 2 = 1 n ∑ y ∈ X \ P tail ( y T Λ T u ) 2 .

We now apply Lemma 3.8 to infer that

<!-- formula-not-decoded -->

seeing as α ≤ 1 / 2 . Plugging this into Equation (2) we

<!-- formula-not-decoded -->

which by definition proves that x i belongs to P tail .

Corollary C.5. (Corollary 3.10 restated.) In all iterations of the algorithm it holds that Σ ⪰ I , namely, that the least eigenvalue of the second moment matrix of the input is ≥ 1 .

Proof. Again, we prove this by induction of t , the iteration of Algorithm 2. In fact, denoting X t as the input of of Algorithm 2 at iteration t , then we argue that the least eigenvalue of the matrix 1 n ∑ x ∈ X t \ P tail xx T is at least 1 .

that Algorithm 1 invokes Algorithm 2 on the input multiplied a 1 1 -α -factor, and so it holds that 1 n ∑ x ∈ X 0 \ P tail xx T ≥ 1 for X 0 , the very first input on which Algorithm 2 is run.

Consider t = 0 , prior to the execution of Algorithm 2 even once. Apply Lemma 3.8 with u being the direction of the least eigenvalue of Σ , and we get that 1 n ∑ x ∈ X \ P tail xx T ≥ (1 -α ) · 1 . Observe

Consider now any intermediate t , where we assume that input satisfies 1 n ∑ x ∈ X t \ P tail xx T ≥ 1 .

We can apply Lemma 3.4 and Corollary 3.5 solely to the points in X t \ P tail and have that 1 n ∑ x ∈ X t \ P tail 8 7 x Π x T ≥ 1 . Since Lemma 3.9 asserts no point in X t \ P tail is shrunk then we get that the required also holds at the invocation of the next iteration.

Claim C.6. (Claim 4.2 restated.) Let D be a distribution on R d with a finite second moment Σ . Consider a random vector y chosen by drawing x ∼ D and then multiplying y = Σ -1 / 2 x . Suppose that ∥ y ∥ ≤ M 1 a.s. that we also have a bound ∥ E [( y T y ) yy T ] ∥ ≤ M 2 . Fix α, β &gt; 0 . If we draw m = max { 2 M 2 α 2 , 2(1+ M 2 1 ) 3 α } · ln( 4 d / β ) examples from D and compute the empirical second moment matrix ˆ Σ , then w.p. 1 β it holds that

<!-- formula-not-decoded -->

≥ -

Proof. Denote our sample of drawn points as x 1 , x 2 , ..., x m . Define ∀ i : y i = Σ -1 / 2 x i so that E [ y i y T i ] = I . This transforms the problem to bounding: ∥ ˆ Σ y -I ∥ 2 ≤ α where ˆ Σ y = 1 m m ∑ i =1 y i y T i . Now ∥ y i y T i ∥ 2 = ∥ y i ∥ 2 2 ≤ M 2 1 .

Next, define the random deviation Z of the estimator ˆ Σ y from the true covariance matrix I :

<!-- formula-not-decoded -->

The random matrices Z i are independent, identically distributed, and centered. To apply Fact 4.1, we need to find a uniform bound R for the summands, and we need to control the matrix variance statistic σ 2 . First, let us develop a uniform bound on the spectral norm of each summand. We can calculate that:

<!-- formula-not-decoded -->

Second, we need to bound the matrix variance statistic σ 2 defined in 4.1, with σ 2 = ∥ E [ ZZ T ] ∥ = ∥ m ∑ i =1 E [ Z i Z T i ] ∥ . We need to determine the variance of each summand. By direct calculation:

<!-- formula-not-decoded -->

Then we have:

<!-- formula-not-decoded -->

We now invoke the matrix Bernstein inequality, Fact 4.1:

<!-- formula-not-decoded -->

which is upper bounded by β given that m ≥ m ( α, β ) = max { 2 M 2 α 2 , 2(1+ M 2 1 ) 3 α } · ln( 4 d / β ) .

## D More Applications

## D.1 Application: The Uniform Distribution Over Some Convex Ellipsoid

Fix a PSD matrix 0 ⪯ A ⪯ I . Consider the uniform distribution D over the surface of some convex ellipsoid K = { x ∈ R d | x T A -1 x = 1 } . Our goal in this section is to argue that our algorithm is able to approximate the 2nd moment matrix Σ D . To that end, we want to determine the size m of a subsample drawn from D , such that with probability at least 1 -β : ∥ Σ -1 / 2 D ˆ Σ D Σ -1 / 2 D -I ∥ 2 ≤ α where Σ D = 1 d A is the second moment of D .

To utilize Claim 4.2, it is necessary to compute the bounds M 1 and M 2 . First, note that if x is drawn from the surface of K then ∥ Σ -1 / 2 D x ∥ = √ d ∥ y ∥ for unit-length vector y , implying M 1 = √ d . Second, consider y = Σ -1 / 2 D x and observe that:

<!-- formula-not-decoded -->

where inequality ( ∗ ) follows from the fact that x T A -1 x = 1 for all i and equality ( ∗∗ ) follows since E [ x i x T i ] = 1 d A .

Hence we have ∥ E [ y i y T i y i y T i ] ∥ ≤ d = M 2 , and we conclude that D is ( O ( d α 2 · ln( 4 d / β ) ) , α, β ) -subsamplable.

Recall that we (1 ± γ ) -approximate the 2nd-moment matrix of the input w.p. ≥ 1 -ξ . Thus, we need the input itself to be a (1 ± γ ) -approximation of the 2nd-moment matrix of the distribution. This means our algorithm requires

<!-- formula-not-decoded -->

for α = 1 / 2 and β = O ( 1 log(1 /λ min ) ) in order to return a (1 ± O ( γ )) -approximation of the 2nd moment of the distribution w.p. ≥ 1 -O ( ξ ) .

Plugging the m ( α, β ) of D we conclude that in order to return a (1 ± O ( γ )) -approximation of the 2nd moment of D w.p. ≥ 1 -O ( ξ ) our sample complexity ought to be

<!-- formula-not-decoded -->

While, for comparison, our baseline algorithm requires in order to return a (1 ± O ( γ )) -approximation of the 2nd moment of the distribution w.p. ≥ 1 -O ( ξ ) .

<!-- formula-not-decoded -->

## D.2 Examples of Heavy-Tailed Distributions

The above discussion holds for a general distribution. Next we demonstrate our algorithms performance a few heavy-tailed distributions. However, we also emphasize that many more applications are possible, since the subsamplability assumption is broad and encompasses a wide range of input distributions. For example, we further analyze datasets drawn from uniform distributions over ellipsoids and from Gaussian mixtures with stochastic outliers in Appendix D.1 and ?? respectively.

The Truncated Pareto Distribution. Throughout we use the following distribution truncated Pareto distribution, denoted P 6 , that is supported on the interval [1 , B ] for some B &gt; 1 (say B = 10 ), and whose PDF is ∝ x -6 . Formally, its PDF is

<!-- formula-not-decoded -->

so that f integrates to 1 . Simple calculations show that µ 6 def = E λ ∼P 6 [ λ ] = 5 4 · B ( B 4 -1) B 5 -1 , that σ 2 6 def =

<!-- formula-not-decoded -->

We consider here two distributions composed of a λ ∼ P 6 and v ∈ R S d -1 . The first is λv , namely a vector with direction distributed uniformly over the unit sphere and magnitude distributed according to the P 6 distribution; and the second is λ ◦ v , namely a ( d + 1) -dimensional vector with first coordinate drawn from P 6 , concatenated with a uniformly chosen vector from the unit sphere on the remaining d coordinates.

The λv Distribution. Define the random variable x = λv , where v is uniformly distributed from the unit sphere, and λ ∼ P 6 . Let D be the distribution of x . Our goal is to argue that our algorithm is able to approximate the 2nd moment matrix Σ D and outperforms the baseline(s). To that end, we want to determine the size m ( α, β ) of a subsample drawn from D so that with probability at least 1 β : Σ -1 / 2 ˆ Σ Σ -1 / 2 I 2 α .

<!-- formula-not-decoded -->

To utilize Claim 4.2, we compute Σ D = E [ λv ( λv ) T ] = E [ λ 2 ] E [ vv T ] = σ 2 6 d I , since E [ vv T ] = 1 d I . It follows that y = Σ -1 / 2 D x = √ d σ 6 λv , so we can bound ∥ y ∥ = M 1 def = B √ d σ 6 . Lastly, it is necessary to compute M 2 . To this end, we evaluate the expectation:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( ∗ ) follows from v T v = 1 and ( ∗∗ ) holds since E [ λ 4 ] = 5 B 4 ( B -1) B 5 -1 and E [ vv T ] = 1 d I . Hence we have ∥ E [ y i y T i y i y T i ] ∥ ≤ 2 d = M 2 . We now plug-in our values of B , λ min and M and infer this distribution is m ( α, β ) -sampleable for m = O (max { d α 2 , dB 2 α } ln( d / β )) . Plugging this into (1) we conclude that in order to return a (1 ± O ( γ )) -approximation of the 2nd moment of D w.p. ≥ 1 -O ( ξ ) our sample complexity ought to be

<!-- formula-not-decoded -->

While, for comparison, our baseline algorithm requires

<!-- formula-not-decoded -->

in order to return a (1 ± O ( γ )) -approximation of the 2nd moment of the distribution w.p. ≥ 1 -O ( ξ ) .

The λ ◦ v Distribution. Consider the ( d + 1) -dimensional distribution D where the first coordinate is drawn from the truncated Pareto distribution P 6 and the remaining coordinates are drawn uniformly over the unit sphere S d -1 : I.e. x = [ λ v ] ∼ D . Our goal in this section is to argue that our algorithm is able to approximate the 2nd moment matrix Σ D . To that end, we want to determine the size m of a subsample drawn from D , such that with probability at least 1 -β : ∥ Σ -1 / 2 D ˆ Σ D Σ -1 / 2 D -I ∥ 2 ≤ α .

First it is easy to see that the L 2 -norm of any x ∼ D is at most B +1 . Next, we compute Σ D :

It follows that the vector y = Σ -1 / 2 D x = [ σ -1 6 0 T 0 √ dI ] [ λ u ] = [ σ -1 6 λ √ du ] , so its norm is upper

<!-- formula-not-decoded -->

bounded by M 1 = √ d + σ -2 6 B 2 . So now we can evaluate the expectation:

<!-- formula-not-decoded -->

seeing as σ -4 6 E [ λ 4 ] ≈ 9 5 = O (1) we can infer that M 2 = ∥ E [ yy T yy T ] ∥ = O ( d ) .

We now plug-in our values of B , λ min and M and infer this distribution is m ( α, β ) -samplable for m = O (max { d α 2 , d + B 2 α } ln( d / β )) . Plugging this into (1) we conclude that in order to return a (1 ± O ( γ )) -approximation of the 2nd moment of D w.p. ≥ 1 -O ( ξ ) our sample complexity ought to be

<!-- formula-not-decoded -->

The analysis suggests we can even set B = O ( √ d ) and get a sample complexity of ˜ O ( d γ 2 + d 3 / 2 γ √ ρ ) . While, for comparison, our baseline algorithm requires

<!-- formula-not-decoded -->

in order to return a (1 ± O ( γ )) -approximation of the 2nd moment of the distribution w.p. ≥ 1 -O ( ξ ) .

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly state the main contributions of the paper, including the subsamplability assumption, the design of a recursive zCDP algorithm for private second moment estimation, and its robustness to outliers.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The paper explicitly discusses key limitations, including the reliance on the subsamplability assumption and the requirement that the failure probability β be small relative to α/ log( R ) .

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

Justification: The paper provides precise and complete statements for all theoretical results, with clearly stated assumptions such as subsamplability parameters ( m,α,β ) and bounds on input norms and eigenvalues. Full proofs are included either in the main text or the appendix, and all necessary external results (e.g., matrix concentration inequalities) are appropriately cited and referenced to support correctness.

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

Justification: This is a theoretical paper that does not include empirical experiments. All claims are supported by formal definitions, theorems, and complete proofs, making experimental reproducibility not applicable.

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

Justification: This paper is entirely theoretical and does not include empirical experiments or datasets. As such, no code or data is necessary to reproduce the main results, and this question is not applicable.

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

Justification: This is a theoretical work that does not involve empirical experiments, training, or testing. Therefore, specifications such as data splits, hyperparameters, or optimizers are not applicable.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The paper does not include empirical experiments or statistical tests. All results are theoretical and supported by formal analysis, so reporting statistical significance or error bars is not applicable.

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

Justification: The paper contains no empirical experiments or statistical evaluations; all results are theoretical and supported by formal analysis and proofs. Hence, reporting error bars or statistical significance is not applicable.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The paper is theoretical in nature and does not involve human subjects, personal data, or deployment of systems. The research adheres to the NeurIPS Code of Ethics in full, including principles of transparency, rigor, and integrity.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This work is foundational and theoretical in nature, focusing on differential privacy and second moment estimation without reference to specific applications or deployments. As such, it does not have direct societal impact, either positive or negative, and broader impact considerations are not applicable in this context.

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

Justification: This paper does not release any models, datasets, or tools that pose a risk of misuse. It presents purely theoretical results in differential privacy and statistical estimation, and therefore no safeguards are necessary or applicable.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The paper does not use any external assets such as datasets, codebases, or pretrained models. All results are derived from original theoretical work, and all prior research is properly cited in accordance with academic standards.

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

Justification: The paper does not introduce or release any new datasets, code, or models. It presents only theoretical contributions, so no asset documentation is required.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve any crowdsourcing, user studies, or research with human subjects. It is purely theoretical and does not require participant interaction or data collection from individuals.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve any research with human subjects or participant data. It is purely theoretical, so IRB approval or equivalent review is not applicable.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The paper does not involve the use of large language models (LLMs) in the development of core methods or theoretical results. Any use of LLMs, if any, was limited to writing assistance and does not impact the scientific rigor or originality of the research.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.