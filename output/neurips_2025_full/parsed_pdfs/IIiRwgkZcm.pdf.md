## How many measurements are enough? Bayesian recovery in inverse problems with general distributions

## Ben Adcock

Department of Mathematics Simon Fraser University Canada

## Nick Huang

Department of Mathematics Simon Fraser University Canada

## Abstract

We study the sample complexity of Bayesian recovery for solving inverse problems with general prior, forward operator and noise distributions. We consider posterior sampling according to an approximate prior P , and establish sufficient conditions for stable and accurate recovery with high probability. Our main result is a non-asymptotic bound that shows that the sample complexity depends on (i) the intrinsic complexity of P , quantified by its approximate covering number , and (ii) concentration bounds for the forward operator and noise distributions. As a key application, we specialize to generative priors, where P is the pushforward of a latent distribution via a Deep Neural Network (DNN). We show that the sample complexity scales log-linearly with the latent dimension k , thus establishing the efficacy of DNN-based priors. Generalizing existing results on deterministic (i.e., non-Bayesian) recovery for the important problem of random sampling with an orthogonal matrix U , we show how the sample complexity is determined by the coherence of U with respect to the support of P . Hence, we establish that coherence plays a fundamental role in Bayesian recovery as well. Overall, our framework unifies and extends prior work, providing rigorous guarantees for the sample complexity of solving Bayesian inverse problems with arbitrary distributions.

## 1 Introduction

Inverse problems are of fundamental importance in science, engineering and industry. In a standard setting, the aim is to recover an unknown vector (e.g., an signal or image) x ∗ ∈ R n from measurements

<!-- formula-not-decoded -->

Here e ∈ R m is measurement noise and A ∈ R m × n , often termed the measurement matrix , represents the forwards operator. While simple, the discrete, linear problem (1.1) is sufficient to model many important applications [5, 25, 64, 68]. It is common to solve (1.1) using a Bayesian approach (see, e.g., [30, 71]), where one assumes that x ∗ is drawn from some prior distribution R . However, in practice, R is never known exactly. Especially in modern settings that employ Deep Learning (DL) [12,33], it is typical to learn an approximate prior P and then recover x ∗ from y by approximate posterior sampling , i.e., sampling ˆ x from the posterior P ( ·| y, A ) . An increasingly popular approach involves using generative models to learn P (see, e.g., [12, 21, 33, 68, 70, 80] and references therein).

A major concern in many inverse problems is that the number of measurements m is highly limited, due to physical constraints such as time (e.g., in Magnetic Resonance Imaging (MRI)), power (e.g., in portable sensors), money (e.g., seismic imaging), radiation exposure (e.g., X-Ray CT), or other factors [5, 64, 68]. Hence, one aims to recover x ∗ well while keeping the number of measurements m as small as possible. With this in mind, in this work we address the following broad question: How many measurements suffice for stable and accurate recovery of x ∗ ∼ R via approximate posterior

sampling ˆ x ∼ P ( ·| y, A ) , and what are conditions on R , P and the distributions A and E of the measurement matrix A and noise e , respectively, that ensure this recovery?

## 1.1 Overview

In this work, we strive to answer to this question in the broadest possible terms, with a theoretical framework that allows for very general types of distributions. We now describe the corresponding conditions needed and present simplified versions of our main results.

(i) Closeness of the real and approximate distributions. We assume that W p ( R , P ) is small for some 1 ≤ p ≤ ∞ , where W p denotes the Wassertstein p -metric.

(ii) Low-complexity of P . Since m ≪ n in many applications, to have any prospect for accurate recovery we need to impose that P (or equivalently R , in view of the previous assumption) has an inherent low complexity. Following [49], we quantify this in terms of its approximate covering number Cov η,δ ( P ) . This is equal to the minimum number of balls of radius η required to cover a region of R n having P -measure at least 1 -δ . See Definition 2.1 for the full definition.

(iii) Concentration of A . We consider constants C low ( t ) = C low ( t ; A , D ) ≥ 0 and C upp ( t ) = C upp ( t ; A , D ) ≥ 0 (see Definition 2.3 for the full definition) such that

<!-- formula-not-decoded -->

for all x ∈ D := supp( P ) -supp( P ) and t &gt; 0 . Here, and throughout this paper, we write S -S = { x 1 -x 2 : x 1 , x 2 ∈ S } ⊆ R n for the difference set associated with a set S ⊆ R n . We also write supp( P ) for the support of the measure P (see §2.1 for the definition). Furthermore, we write ∥·∥ for the ℓ 2 -norm and ∥·∥ ∞ for the ℓ ∞ -norm. If A is isotropic, i.e., E A ∼A ∥ Ax ∥ 2 = ∥ x ∥ 2 , ∀ x ∈ R n , as is often the case in practice, these constants measure how fast ∥ Ax ∥ 2 concentrates around its mean. Notice that this condition is imposed only on D = supp( P ) -supp( P ) , rather than the whole space R n . As we see later, this is crucial in obtaining meaningful recovery guarantees.

Finally, in order to present a simplified result in this section, we now make several simplifying assumptions. Both of these will be relaxed in our full result, Theorem 3.1.

(iv) Gaussian noise. Specifically, we assume that E = N (0 , σ 2 m I ) for some σ &gt; 0 .

(v) Bounded forwards operators. We assume that ∥ A ∥ ≤ θ a.s. for A ∼ A and some θ &gt; 0 .

Theorem 1.1 (Simplified main result) . Let 1 ≤ p ≤ ∞ , 0 &lt; δ ≤ 1 / 4 , ε, η &gt; 0 and suppose that conditions (i)-(v) hold with W p ( R , P ) ≤ ε/ (2 mθ ) and σ ≥ ε/δ 1 /p . Suppose that x ∗ ∼ R , A ∼ A , e ∼ E independently and ˆ x ∼ P ( ·| y, A ) , where y = Ax ∗ + e . Then, for any d ≥ 2 ,

<!-- formula-not-decoded -->

Note that in this and all subsequent results, the term on the left-hand side of (1.2) is the probability with respect to all variables, i.e., x ∗ ∼ R , A ∼ A , e ∼ E and ˆ x ∼ P ( ·| y, A ) . Theorem 1.1 is extremely general, in that it allows for essentially arbitrary (real and approximate) signal distributions R and P and an essentially arbitrary distribution A for the forwards operator. In broad terms, it bounds the probability that the error ∥ x ∗ -ˆ x ∥ of posterior sampling exceeds a constant times the noise level σ plus an arbitrary parameter η . It does so in terms of the approximate covering number Cov η,δ ( P ) , which measures the complexity of the approximate distribution P , the concentration bounds C low and C upp for A , which measure how much A elongates or shrinks a fixed vector, and an exponentially-decaying term e -m/ 16 , which stems from the (Gaussian) noise. In particular, by analyzing these terms for different classes of distributions P and A , we can derive concrete bounds for various exemplar problems. We next describe two such problems.

## 1.2 Examples

We first consider A to be a distribution of subgaussian random matrices . Here A ∼ A if its entries are i.i.d. subgaussian random variables with mean zero and variance 1 /m (see Definition 3.4).

Theorem 1.2 (Subgaussian measurement matrices, simplified) . Consider the setup of Theorem 1.1, where A is a distribution of subgaussian random matrices. Then there is a constant c &gt; 0 (depending on the subgaussian parameters β, κ &gt; 0 ; see Definition 3.4) such that

<!-- formula-not-decoded -->

This theorem shows the efficacy of Bayesian recovery with subgaussian random matrices: namely, the sample complexity scales linearly in the distribution complexity, i.e., the log of the approximate covering number. Later in Theorem 3.5, we slightly refine and generalize this result.

Gaussian random matrices are very commonly studied, due to their amenability to analysis and tight theoretical bounds [5,22,34,49], with Theorem 1.2 being a case in point. However, they are largely irrelevant to practical inverse problems, where physical constraints impose certain structures on the forwards operator distribution A [5, 64, 68]. For instance, in MRI physical constraints mean that the measurements are samples of the Fourier transform of the image. This has motivated researchers to consider much more practically-relevant distributions, in particular, so-called subsampled orthogonal transforms (see, e.g., [5]). Here U ∈ R n × n is a fixed orthogonal matrix - for example, the matrix of the Discrete Fourier Transform (DFT) in the case of MRI - and the distribution A is defined by randomly selecting m rows of U . See Definition 3.7 for the formal definition.

Theorem 1.3 (Subsampled orthogonal transforms, simplified) . Consider the setup of Theorem 1.1, where A is a distribution of subsampled orthogonal matrices based on a matrix U . Then there is a universal constant c &gt; 0 such that

<!-- formula-not-decoded -->

where D = supp( P ) -supp( P ) and µ ( U ; D ) is the coherence of U relative to D , defined as

̸

<!-- formula-not-decoded -->

This result shows that similar stable and accurate recovery to the Gaussian case can be achieved using subsampled orthogonal matrices, provided the number of measurements scales linearly with the coherence. We discuss this term further in §3.2, where we also present the full result, Theorem 3.9.

In general, Theorems 1.2 and 1.3 establish a key condition for successful Bayesian recovery in inverse problems, in each case relating the number of measurements m to the intrinsic complexity log(Cov η,δ ( P )) of P . It is therefore informative to see how this complexity behaves for cases of interest. As noted, it is common to use a generative model to learn P . This means that P = G♯γ , where G : R k → R n is a Deep Neural Network (DNN) and γ is some fixed probability measure on the latent space R k . Typically, γ = N (0 , I ) . If G is L -Lipschitz, we show in Proposition 4.1 that

√

<!-- formula-not-decoded -->

This scales log-linearly in the latent space dimension k , confirming the intrinsic low-complexity of P . Combining (1.3) with Theorems 1.2-1.3 we see that posterior sampling achieves stable and accurate recovery, provided the number of measurements m scales near-optimally with k , i.e., as O ( k log( k )) .

Further, in order to compare with deterministic settings such as classical compressed sensing, which concerns the recovery of s -sparse vectors, we also consider distributions P = P s of s -sparse vectors. In this case, we show in Proposition 4.3 that

<!-- formula-not-decoded -->

Hence s measurements, up to log terms, are sufficient for recovery of approximately sparse vectors. This extends a classical result for deterministic compressed sensing to the Bayesian setting.

## 1.3 Significance

The significance of this work is as follows. See §1.4 for additional discussion.

1. We provide the first results for Bayesian recovery with arbitrary real and approximate prior distributions R , P and forwards operator and noise distributions A , E .
2. Unlike much of the theory of Bayesian inverse problems, which is asymptotic in nature [12,30,71], our results are non-asymptotic . They hold for arbitrary values of the various parameters within given ranges (e.g., the failure probability δ , the noise level σ , the number of measurements m , and so forth).
3. For priors defined by Lipschitz generative DNNs, we establish the first result demonstrating that the sample complexity of Bayesian recovery depends log-linearly on the latent dimension k and logarithmically on the Lipschitz constant.

4. For the important class of subsampled orthogonal transforms, we show that the sample complexity of Bayesian recovery depends on the coherence , thus resolving several key open problems in the literature (see next).

5. It is increasingly well-known that DL-based methods for inverse problems are susceptible to hallucinations and other undesirable effects [11,15,18,25,29,40,42,45,47,60-63,65,66,79]. This is a major issue that may limit the uptake of these methods in safety-critical domains such as medical imaging [25,55,57,61,74,76,77]. Our results provide theoretical guarantees for stable and accurate, and therefore show conditions under which hallucinations provably cannot occur. This is not only theoretically interesting, but it also has practical consequences in the development of robust DL methods for inverse problems - a topic we intend to explore in future work.

## 1.4 Related work

Bayesian methods for inverse problems have become increasingly popular over the last several decades [30,71], and many state-of-the-art DL methods for inverse problems now follow a Bayesian approach (see [7,12,21,46,56,64] and references therein). Learned priors, such as those stemming from generative models, are now also increasingly used in applications [1,7,12,21,28,33,38,46,48, 51,53,54,58,64,68,70,78,80].

This work is motivated in part by the (non-Bayesian) theory of generative models for solving inverse problems (see [22,33,43,44,48,68-70,80] and references therein). This was first developed in [22], where compressed sensing techniques were used to show recovery guarantees for a Gaussian random matrix A when computing an approximate solution to (1.1) in the range Σ := ran( G ) of a Lipschitz map G : R k → R n , typically assumed to be a generative DNN. This is a deterministic approach. Besides the random forward operator and (potentially) random noise, it recovers a fixed (i.e., nonrandom) underlying signal x ∗ in a deterministic fashion with a point estimator ˆ x that is obtained as a minimizer of the empirical ℓ 2 -loss min z ∈ Σ ∥ Az -y ∥ 2 . In particular, no information about the latent space distribution γ is used. In this work, following, e.g., [1,21,48,49,70] and others, we consider a Bayesian setting, where x ∗ ∼ R is random and where we quantify the number of measurements that suffice for accurate and stable recovery via posterior sampling ˆ x ∼ P ( ·| y, A ) .

Our work is a generalization of [49], which considered Bayesian recovery with Gaussian random matrices and standard Gaussian noise. We significantly extend [49] to allow for arbitrary distributions A and E for the forward operator and noise, respectively. A key technical step in doing this is the introduction of the concentration bounds C low and C upp . In particular, these bounds are imposed only over the subset D = supp( P ) -supp( P ) . This is unnecessary in the Gaussian case considered in Theorem 1.2, but crucial to obtain meaningful bounds in, for instance, the case of subsampled orthogonal transforms considered in Theorem 1.3 (see Remarks 3.2 and 3.10 for further discussion). As noted, this case is very relevant to applications. In particular, when U is taken as a DFT matrix our work addresses open problems posed in [48, §3] and [68, §II.F] on recovery guarantees with Fourier measurements. We also derive bounds for the approximate covering number of distributions given by Lipschitz generative DNNs (see (1.3) and Proposition 4.1) and distributions of sparse vectors (see (1.4) and Proposition 4.3), addressing an open problem posed in [49, §6]. In particular, we demonstrate stable and accurate recovery in a Bayesian sense with a number of measurements that scales linearly in the model complexity, i.e., k in the former case and s in the latter case.

Recently [16,17] generalized the results of [22] in the non-Bayesian setting from Gaussian random matrices to subsampled orthogonal transforms. Theorem 1.3 provides a Bayesian analogue of this work, as discussed above, where we consider posterior sampling rather than a deterministic point estimator. We also extend the setup of [16, 17] by allowing for general measurement distributions A and priors P . In particular, in [16, 17] the quantity Σ , which is the deterministic analogue of the prior distribution P in our work, was assumed to be the range of a ReLU generative DNN. Like in [16], we make use of the concept of coherence (see Theorem 1.3). However, our proof techniques are completely different to those of [16, 17].

Classical compressed sensing considers the recovery of approximately s -sparse vectors from (1.1). However, it has been extended to consider much more general types of low-complexity signal models, such as joint or block sparse vectors, tree sparse vectors, cosparse vectors and many others [4, 6, 13, 23, 31, 36, 73]. However, most recovery guarantees for general model classes consider only (sub)Gaussian measurement matrices (see e.g., [13, 34]). Recently, [3] introduced a general

framework for compressed sensing that allows for general low-complexity models Σ ⊆ R n contained within a finite union of finite-dimensional subspaces and arbitrary (random) measurement matrices. Our work is a Bayesian analogue of this deterministic setting. Similar to [3], a key feature of our work is that we consider arbitrary real and approximate signal distributions P , R (analogous to arbitrary low-complexity models Σ ) and arbitrary distributions A for the forwards operator. Unsurprisingly, a number of the conditions in our main result, Theorem 1.1 - namely, the low-complexity condition (ii) and concentration condition (iii) - share similarities to those that ensure stable and accurate recovery in non-Bayesian compressed sensing. See Remarks 2.2 and 3.3 for further discussion. However, the proof techniques used in this work are once more entirely different.

Finally, while the focus of this work is to establish guarantees for posterior sampling, we mention in passing related work on information-theoretically optimal recovery methods. Methods such as Approximate Message Passing (AMP) [14,35] are well studied, and asymptotic information-theoretic bounds for Gaussian random matrices are known [52]. AMP methods are fast point estimation algorithms, whereas we focus on sampling-based methods and do not consider computational implementations (see §5 for some further discussion). Note that information-theoretic lower bounds for posterior sampling have also been shown for the Gaussian case in [49].

## 2 Preliminaries

## 2.1 Notation

We now introduce some further notation. We let B r ( x ) = { z ∈ R n : ∥ z -x ∥ ≤ r } and, when x = 0 , we write B r := B r (0) . Given a set X ⊆ R n , we write X c = R n \ X for its complement. We also write B r ( X ) = ⋃ x ∈ X B r ( x ) for the r -neighbourhood of X .

Let ( X, F , µ ) be a Borel probability space. We write supp( µ ) for its support, i.e., the smallest closed set A ⊆ X for which µ ( A ) = 1 . Given probability spaces ( X, F 1 , µ ) , ( Y, F 2 , ν ) , we write Γ = Γ µ,ν for the set of couplings, i.e., probability measures on the product space ( X × Y, σ ( F 1 ⊗F 2 )) whose marginals are µ and ν , respectively. Given a cost function c : X × Y → [0 , ∞ ) and 1 ≤ p &lt; ∞ , the Wassersteinp metric is defined as

<!-- formula-not-decoded -->

If p = ∞ , then W ∞ ( µ, ν ) = inf γ ∈ Γ (esssup γ c ( x, y )) . In this paper, unless stated otherwise, X = Y = R n and the cost function c is the Euclidean distance.

## 2.2 Approximate covering numbers

As a measure of complexity of measures, we use the concept of approximate covering numbers as introduced in [49].

Definition 2.1 (Approximate covering number) . Let ( X, F , P ) be a probability space and δ, η ≥ 0 . The η, δ -approximate covering number of P is defined as

<!-- formula-not-decoded -->

This quantity measures how many balls of radius η are required to cover at least 1 -δ of the P -mass of R n . See [49] for further discussion. Note that [49] does not require the centres x i of the approximate cover belong to supp( P ) . However, this is useful in our more general setting and presents no substantial restriction. At worst, this requirement changes η by a factor of 1 / 2 .

Remark 2.2 (Relation to non-Bayesian compressed sensing) Note that when δ = 0 , the approximate covering number Cov η, 0 ( P ) ≡ Cov η (supp( P )) is just the classical covering number of the set supp( P ) , i.e., the minimal number of balls of radius η that cover supp( P ) . Classical covering numbers play a key role in (non-Bayesian) compressed sensing theory. Namely, the covering number of (the unit ball of) the model class Σ ⊆ R n directly determines the number of measurements that suffice for stable and accurate recovery. See, e.g., [3, 34]. In the Bayesian setting, the approximate covering number plays the same role; see Theorem 1.1.

## 2.3 Bounds for A and E

Since our objective is to establish results that hold for arbitrary measurement and noise distributions A and E , we require several key definitions. These are variety of (concentration) bounds.

Definition 2.3 (Concentration bounds for A ) . Let A be a distribution on R m × n , t ≥ 0 and D ⊆ R n . Then a lower concentration bound for A is any constant C low ( t ) = C low ( t ; A , D ) ≥ 0 such that

<!-- formula-not-decoded -->

Similarly, an upper concentration bound for A is any constant C upp ( t ) = C upp ( t ; A , D ) ≥ 0 such that

<!-- formula-not-decoded -->

Finally, given t, s ≥ 0 an (upper) absolute concentration bound for A is any constant C abs ( s, t ; A , D ) such that

<!-- formula-not-decoded -->

Notice that if A is isotropic, i.e., E ∥ Ax ∥ 2 = ∥ x ∥ 2 , ∀ x ∈ R n , then C low and C upp determine how well ∥ Ax ∥ 2 concentrates around its mean ∥ x ∥ 2 for any fixed x ∈ D . To obtain desirable sample complexity estimates (e.g., Theorems 1.2 and 1.3), we need concentration bounds that decay exponentially in m . A crucial component of this analysis is considering concentration bounds over some subset D (related to the support of P ), as, in general, one cannot expect fast concentration over the whole of R n . See Remarks 3.2 and 3.10.

Definition 2.4 (Concentration bound for E ) . Let E be a distribution in R m and t ≥ 0 . Then an (upper) concentration bound for E is any constant D upp ( t ) = D upp ( t ; E ) ≥ 0 such that

<!-- formula-not-decoded -->

Notice that this bound just measures the probability that the noise is large. We also need the following concept, which estimates how much the density of E changes in a τ -neighbourhood of the origin when perturbed by an amount ε .

Definition 2.5 (Density shift bounds for E ) . Let E be a distribution in R m with density p E and ε, τ ≥ 0 . Then a density shift bound for E is any constant D shift ( ε, τ ) = D shift ( ε, τ ; E ) ≥ 0 (possibly + ∞ ) such that

<!-- formula-not-decoded -->

## 3 Main results

We now present our main results. The first, an extension of Theorem 1.1, is a general result that holds for arbitrary distributions R , P , A and E .

Theorem 3.1. Let 1 ≤ p ≤ ∞ , 0 ≤ δ ≤ 1 / 4 , ε, η, t &gt; 0 , c, c ′ ≥ 1 and σ ≥ ε/δ 1 /p . Let E be a distribution on R m and R , P be distributions on R n satisfying W p ( R , P ) ≤ ε and

<!-- formula-not-decoded -->

for some k ∈ N . Suppose that x ∗ ∼ R , A ∼ A , e ∼ E independently and ˆ x ∼ P ( ·| y, A ) , where y = Ax ∗ + e . Then p := P [ ∥ x ∗ -ˆ x ∥ ≥ ( c +2)( η + σ )] satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This theorem bounds the probability p of unstable or inaccurate recovery in terms of the various parameters using the constants introduced in the previous section and the approximate covering numbers of R , P . This result is powerful in its generality, but as a consequence, rather opaque. In particular, since it considers arbitrary measurement and noise distributions, the number of measurements m does not explicitly enter the bound. For typical distributions, a dependence on m is found in the concentration bounds C low , C upp , D upp , as well as the terms D shift and C abs . For instance, the former decay exponentially-fast in m for the examples introduced in §1.2, and therefore compensate for the exponentially-large scaling in k in the main bound (see §B for precise estimates, as well as the discussion in §3.1-3.2). Note that Theorem 3.1 also considers general noise distributions E . While Gaussian noise is arguably the most important example - and will be used in all our subsequent examples - this additional generality comes at little cost in terms of the technicality of the proofs.

Remark 3.2 (The concentration bounds in Theorem 3.1) A particularly important facet of this result, for the reasons discussed above, is that the various concentration bounds C abs , C low and C upp are taken over sets D 1 , D 2 - given by (3.2) and (3.3), respectively, and related to the support of P and R - rather than the whole space R n . We exploit this fact crucially later in Theorem 3.9.

Remark 3.3 (Relation to non-Bayesian compressed sensing) The constants C low and C upp are similar, albeit not identical to similar conditions such as the Restricted Isometry Property (RIP) (see, e.g., [5, Chpt. 5]) or Restricted Eigenvalue Condition (REC) [19,22] that appear in non-Bayesian compressed sensing. There, one considers a fixed model class Σ ⊆ R n , such as the set Σ s of s -sparse vectors or, as in [22], the range ran( G ) of a generative DNN. Conditions such as the RIP or REC impose that ∥ Ax ∥ is concentrated around ∥ x ∥ for all x belonging to the difference set Σ -Σ . In Theorem 3.1, assuming P attains the minimum in (3.1), there is a similar condition with Σ replaced by supp( P ) . Indeed, C low (2 √ 2 / √ c ; A , D 2 ) measures how small ∥ Ax ∥ is in relation to ∥ x ∥ and C upp ( √ c/ (2 √ 2); A , D 2 ) measures how large ∥ Ax ∥ is in relation to ∥ x ∥ .

## 3.1 Example: Subgaussian random matrices with Gaussian noise

We now apply this theorem to the first example introduced in §1.2. Recall that a random variable X on R is subgaussian with parameters β, κ &gt; 0 if P ( | X | ≥ t ) ≤ β e -κt 2 for all t &gt; 0 .

Definition 3.4 (Subgaussian random matrix) . A random matrix A ∈ R m × n is subgaussian with parameters β, κ &gt; 0 if A = 1 √ m ˜ A , where the entries of ˜ A are independent mean-zero sugaussian random variables with variance 1 and the same subgaussian parameters β , κ .

Note that 1 / √ m is a scaling factor that ensures that A is isotropic , i.e., E ∥ Ax ∥ 2 = ∥ x ∥ 2 , ∀ x ∈ R n .

Theorem 3.5. Let 1 ≤ p ≤ ∞ , 0 ≤ δ ≤ 1 / 4 , ε, η &gt; 0 and σ ≥ ε/δ 1 /p . Let E = N (0 , σ 2 m I ) and A be a distribution of subgaussian random matrices with parameters β, κ &gt; 0 . Let R , P be distributions on R n and suppose that x ∗ ∼ R , A ∼ A , e ∼ E independently and ˆ x ∼ P ( ·| y, A ) , where y = Ax ∗ + e . Then there is a constant c ( β, κ ) &gt; 0 depending on β, κ only such that

<!-- formula-not-decoded -->

provided W p ( R , P ) ≤ ε/c ( β, κ ) and

<!-- formula-not-decoded -->

This theorem is derived from Theorem 3.1 by showing that the various concentration bounds are exponentially small in m for subgaussian random matrices (see §B). It is a direct generalization of [49], which considered the Gaussian case only. It shows that subgaussian random matrices are near-optimal for Bayesian recovery, in the sense that m scales linearly with the log of the approximate covering number (3.4). We estimate these covering numbers for several key cases in §4.

It is worth at this stage discussing how (3.4) behaves with respect to the various parameters. First, suppose that η decreases so that the error bound becomes smaller. Then Cov η,δ ( · ) increases, meaning, as expected, that more measurements are required to meet (3.4). Second, suppose that δ decreases, so that the failure probability shrinks. Then Cov η,δ ( · ) and log(1 /δ ) both increase, meaning, once again, that more measurements are needed for (3.4) to hold. Both behaviours are as expected.

Remark 3.6 (Relation to Johnson-Lindenstrauss) Suppose that P = R is a sum of d Diracs located at X = { x 1 , . . . , x d } ⊆ R n . Since the matrix A ∈ R m × n is a linear dimensionality-reducing map, the Johnson-Lindenstrauss Lemma states that distances in X are preserved under A if and only if m = O (log( d )) . In this setting, preserving the distances in X is equivalent to being able to stably identify the mode from which a signal is drawn when observing its measurements. In agreement with this argument, Theorem 3.5 also predicts recovery from roughly m = O (log( d )) measurements.

## 3.2 Example: Randomly-subsampled orthogonal transforms with Gaussian noise

As discussed, subgaussian random matrices are largely impractical. We now consider the more practical case of subsampled orthogonal transforms.

Definition 3.7 (Randomly-subsampled orthogonal transform) . Let U ∈ R n × n be orthogonal (i.e., U ⊤ U = UU ⊤ = I ) and write u 1 , . . . , u n ∈ R n for its rows. Let X 1 , . . . , X n ∼ i . i . d . Ber( m/n ) be independent Bernoulli random variables with P ( X i = 1) = m/n and P ( X i = 0) = 1 -m/n . Then we define a distribution A as follows. We say that A ∼ A if

<!-- formula-not-decoded -->

where { i 1 , . . . , i q } ⊆ { 1 , . . . , n } is the set of indices i for which X i = 1 .

The factor √ n/m ensures that E ( A ⊤ A ) = I . Note that the number of measurements q in this model is itself a random variable, with E ( q ) = m . However, q concentrates exponentially around its mean.

n × n n

Definition 3.8. Let U ∈ R and D ⊆ R . The coherence of U relative to D is

̸

<!-- formula-not-decoded -->

Coherence is a well-known concept in classical compressed sensing with sparse vectors. Definition 3.8 is a generalization that allows for arbitrary model classes D . This definition is similar to that of [16], which considered non-Bayesian compressed sensing with generative models. It is also related to the more general concept of variation introduced in [3].

Theorem 3.9. Let 1 ≤ p ≤ ∞ , 0 ≤ δ ≤ 1 / 4 , ε, η &gt; 0 and σ ≥ ε/δ 1 /p . Let E = N (0 , σ 2 m I ) and A be a distribution of randomly-subsampled orthogonal matrices based on a matrix U . Let R , P be distributions on R n and suppose that x ∗ ∼ R , A ∼ A , e ∼ E independently and ˆ x ∼ P ( ·| y, A ) , where y = Ax ∗ + e . Then there is a universal constant c &gt; 0 such that

<!-- formula-not-decoded -->

provided W p ( R , P ) ≤ ε/ (2 mn ) and

<!-- formula-not-decoded -->

This theorem is a Bayesian analogue of the deterministic results shown in [3, 16]. In [3, 16], the measurement conditions scale linearly with µ ( U ; D ) , where D = Σ -Σ and Σ ⊆ R n is the lowcomplexity model class. Similarly, the number of measurements (3.5) scales linearly with respect to the coherence relative to D = supp( P ) -supp( P ) , which, as discussed in Remark 3.3, plays the role of the low-complexity model class in the Bayesian setting. Note that the measurement condition (3.5) involves the approximate distribution P only. This is relevant, since the quantities µ ( U ; D ) and Cov η,δ ( P ) can be estimated either numerically or analytically in various cases, such as when P is given by a generative model. Indeed, we estimate Cov η,δ ( P ) analytically for Lipschitz generative models in Proposition 4.1 below. The coherence µ ( U ; D ) was estimated analytically in [16] for ReLU DNNs with random weights (see Remark 4.2 below). It can also be estimated numerically for more general types of generative models [2,16]. Overall, by estimating these quantities, one can use (3.5) to gauge how well one can recover with a given P . Note that this may not be possible if (3.5) involved R as well, since this distribution is typically unknown.

In classical compressed sensing, coherence determines the sample complexity of recovering sparse vectors from randomly-subsampled orthogonal transforms [26]. A similar argument can be made in the Bayesian setting. Notice that µ ( U ; D ) ≤ µ ( U ; R n ) = n . However, we are particularly interested in cases where µ ( U ; D ) ≪ n , in which case (3.5) may be significantly smaller than the ambient dimension n . We discuss this in the context of several examples in the next section.

Remark 3.10 (Concentration over subsets) Theorem 3.9 illustrates why it is important that Theorem 3.1 involves concentration bounds over subsets of R n . To derive Theorem 3.9 from Theorem 3.1 (see §B), we show exponentially-fast concentration in m/µ ( U ; D ) . Had we considered the whole of R n , then, since µ ( U ; R n ) = n , this would have lead to an undesirable measurement condition of the form m = O ( n ) scaling linearly in the ambient dimension n .

## 4 Covering number and sample complexity estimates

We conclude by applying our results to two different types of approximate prior distributions.

## 4.1 Generative DNNs

Proposition 4.1 (Approximate covering number for a Lipschitz pushforward of a Gaussian measure) . Let G : R k → R n be Lipschitz with constant L ≥ 0 , i.e., ∥ G ( x ) -G ( z ) ∥ ≤ L ∥ x -z ∥ , ∀ x, z ∈ R k , and define P = G♯γ , where γ = N (0 , I ) is the standard normal distribution on R k . Then

<!-- formula-not-decoded -->

This result shows that P has low complexity, since log(Cov η,δ ( P )) scales log-linearly in k . Combined with Theorem 3.5, it shows that accurate and stable Bayesian recovery with such a prior with a sample complexity that is near-optimal in the latent dimension k , i.e., O ( k log( k )) .

Notice that L only appears logarithmically in (4.1). While it is not the main focus of this work, we note that Lipschitz constants of DNNs have been studied quite extensively [37,72,75]. Moreover, it is also possible to design and train DNNs with small Lipschitz constants [59].

Remark 4.2 (Quadratic bottleneck and high coherence) In Theorem 3.9, the measurement condition (3.5) also depends on the coherence. This quantity has been considered in [16] for the case of ReLU DNNs. In particular, if a ReLU DNN G : R k → R n has random weights drawn from a standard normal distribution, then its coherence µ ( U ; D ) scales like O ( k ) up to log factors [16, Thm. 3]. Combining this with Theorem 3.9 and Proposition 4.1, we see that the overall sample complexity for Bayesian recovery scales like O ( k 2 log( k )) in this case. This is worse than the subgaussian case, where there is no coherence factor and the sample complexity, as noted above, is O ( k log( k )) . Such a quadratic bottleneck also arises in the non-Bayesian setting [16]. Its removal is an open problem (see §5). Note that the coherence is also not guaranteed to be small for general (in particular, trained) DNNs. However, [16] also discuss strategies for training generative models to have small coherence. Numerically, they show that generative models with smaller coherence achieve better recovery from the same number of measurements than those with larger coherence.

## 4.2 Distributions of sparse vectors

Let s ∈ { 1 , . . . , n } . We define a distribution P = P s of s -sparse vectors in R n as follows. To draw x ∼ P , we first choose a support set S ⊆ { 1 , . . . , n } , | S | = s , uniformly at random amongst all possible ( n s ) such subsets. We then define x i = 0 , i / ∈ S , and for each i ∈ S we draw x i randomly and independently from N (0 , 1) . Note that there are other ways to define distributions of sparse vectors, which can be analyzed similarly. However, for brevity we only consider the above setup.

Proposition 4.3 (Approximate covering number for distributions of sparse vectors) . Let P = P s be a distribution of s -sparse vectors in R n . Then

<!-- formula-not-decoded -->

As in the previous case, we deduce Bayesian recovery from O ( s log( n/s )) subgaussian measurements, i.e., near-optimal, log-linear sample complexity. In the case of randomly-subsampled orthogonal matrices, we also need to consider the coherence. For P = P s as above, one can easily show that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The term µ ∗ ( U ) is often referred to as the coherence of U (see, e.g., [5, Defn. 5.8] or [26]). Notice that µ ∗ ( U ) ≈ 1 for DFT matrices, which is one reason why subsampled Fourier transforms are particularly effective in (non-Bayesian) compressed sensing. Our work implies as similar conclusion in the Bayesian setting: indeed, substituting (4.2) and (4.3) into (3.5) we immediately deduce that the measurement condition for Bayesian recovery with P s behaves like m = O ( s 2 ) up to log terms.

Remark 4.4 (Quadratic bottleneck) Once more we witness a quadratic bottleneck. In the nonBayesian setting, one can show stable and accurate recovery of sparse vectors from O ( s ) measurements, up to log terms (see, e.g., [5, Cor. 13.15]). However, this requires specialized theoretical techniques that heavily leverage the structure of the set of sparse vectors. In the setting of this paper, the bottleneck arises from the generality of the approach considered in this work: specifically, the fact that our main results hold for arbitrary probability distributions P .

## 5 Limitations and future work

We end by discussing a number of limitations and avenues for future work. First, although our main result Theorem 3.1 is very general, we have only applied it to a number of different cases, such as Lipschitz pushforward measures and Gaussian random matrices or subsampled orthogonal transforms. We believe many other important problems can be studied as corollaries of our main results. This includes sampling with heavy-tailed vectors [50], sampling with random convolutions [67], multi-sensor acquisition problems [27], generative models augmented with sparse deviations [32], block sampling [3,20,24], with applications to practical MRI acquisition, sparse tomography [8], deconvolution and inverse source problems [9]. We believe our framework can also be applied to various types of non-Gaussian noise, as well as problems involving sparsely-corrupted measurements [50]. We are actively investigating applying our framework to these problems.

Second, as noted in Remarks 4.2 and 4.4 there is a quadratic bottleneck when considering subsampled orthogonal transforms. In the non-Bayesian case, this can be overcome in the case of (structured) sparse models using more technical arguments [3]. We believe similar ideas could also be exploited in the Bayesian setting. On a related note, both [16, 22] consider ReLU generative models in the non-Bayesian setting, and derive measurement conditions that do not involve the Lipschitz constant L of the DNN. It is unclear whether analogous results can be established in the Bayesian setting.

Third, our main result involves the density shift bound (Definition 2.5). In particular, the noise distribution should have a density. This rather unpleasant technical assumption stems from Lemma C.6, which is a key step in proving the main result, Theorem 3.1. This lemma allows one to replace the 'real' distribution R in the probability term p in Theorem 3.1 by the approximate distribution P . This is done in order to align the prior and the posterior, which is necessary for the subsequent steps of the proof of Theorem 3.1. It would be interesting to see if this assumption on the noise could be removed through a refined analysis.

Fourth, as noted in [16], the coherence µ ( U ; D ) arising in Theorem 3.9 may be high. In the nonBayesian setting, this has been addressed in [2, 3, 17] by using a nonuniform probability distribution for drawing rows of U , with probabilities given in terms of so-called local coherences of U with relative to D . As shown therein, this can lead to significant performance gains over sampling uniformly at random. We believe a similar approach can be considered in the Bayesian setting as a consequence of our general framework. We intend to explore this in future work.

Finally, our results in this paper are theoretical, and strive to study the sample complexity of Bayesian recovery. We do not address the practical problem of sampling from the posterior. This is a key computational challenge in Bayesian inverse problems [12]. However, efficient techniques for doing this are emerging. See, e.g., [48,54,70] and references therein. We believe an advantage of our results is their independence from the choice of posterior sampling algorithm, whose analysis can therefore be performed separately. This is an interesting problem we intend to examine in the future. In future work we also intend present numerical experiments for various practical settings that further support the theory developed in this paper.

## Acknowledgments and Disclosure of Funding

BA acknowledges the support of the Natural Sciences and Engineering Research Council of Canada of Canada (NSERC) through grant RGPIN-2021-611675. NH acknowledges support from an NSERC Canada Graduate Scholarship. Both authors would like to thank Paul Tupper and Weiran Sun for helpful comments and feedback.

## References

- [1] A. Aali, M. Arvinte, S. Kumar, and J. I. Tamir. Solving inverse problems with score-based generative priors learned from noisy data. In 2023 57th Asilomar Conference on Signals, Systems, and Computers , pages 837-843, 2023.
- [2] B. Adcock, J. M. Cardenas, and N. Dexter. CS4ML: A general framework for active learning with arbitrary data based on Christoffel functions. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems , volume 36, pages 19990-20037, 2023.
- [3] B. Adcock, J. M. Cardenas, and N. Dexter. A unified framework for learning with nonlinear model classes from arbitrary linear samples. In International Conference on Machine Learning , 2024.
- [4] B. Adcock, A. Gelb, G. Song, and Y. Sui. Joint sparse recovery based on variances. SIAM J. Sci. Comput. , 41(1):A246-A268, 2019.
- [5] B. Adcock and A. C. Hansen. Compressive Imaging: Structure, Sampling, Learning . Cambridge University Press, Cambridge, UK, 2021.
- [6] B. Adcock, A. C. Hansen, C. Poon, and B. Roman. Breaking the coherence barrier: a new theory for compressed sensing. Forum Math. Sigma , 5:e4, 2017.
- [7] J. Adler and O. Öktem. Deep Bayesian inversion. In Data-Driven Models in Inverse Problems , pages 359-412. De Gruyter, Berlin, Boston, 2025.
- [8] G. S. Alberti, A. Felisi, M. Santacesaria, and S. I. Trapasso. Compressed sensing for inverse problems and the sample complexity of the sparse Radon transform. J. Eur. Math. Soc. (in press) , 2025.
- [9] G. S. Alberti, A. Felisi, M. Santacesaria, and S. I. Trapasso. Compressed sensing for inverse problems ii: applications to deconvolution, source recovery, and MRI. arXiv:2501.01929 , 2025.
- [10] L. Ambrosio, E. Brué, and D. Semola. Lectures on Optimal Transport . UNITEXT. Springer, Cham, Switzerland, 2nd edition, 2024.
- [11] V. Antun, F. Renna, C. Poon, B. Adcock, and A. C. Hansen. On instabilities of deep learning in image reconstruction and the potential costs of AI. Proc. Natl. Acad. Sci. USA , 117(48):3008830095, 2020.
- [12] S. Arridge, P. Maass, O. Öktem, and C.-B. Schönlieb. Solving inverse problems using datadriven models. Acta Numer. , 28:1-174, 2019.
- [13] R. G. Baraniuk, V. Cevher, M. F. Duarte, and C. Hedge. Model-based compressive sensing. IEEE Trans. Inform. Theory , 56(4):1982-2001, 2010.
- [14] M. Bayati and A. Montanari. The dynamics of message passing on dense graphs, with applications to compressed sensing. IEEE Trans. Inform. Theory , 57(2):764-785, 2011.
- [15] C. Belthangady and L. A. Royer. Applications, promises, and pitfalls of deep learning for fluorescence image reconstruction. Nature methods , 16(12):1215-1225, 2019.
- [16] A. Berk, S. Brugiapaglia, B. Joshi, Y. Plan, M. Scott, and O. Yilmaz. A coherence parameter characterizing generative compressed sensing with fourier measurements. IEEE J. Sel. Areas Inf. Theory , 3(3):502-512, 2022.
- [17] A. Berk, S. Brugiapaglia, Y. Plan, M. Scott, X. Sheng, and O. Yilmaz. Model-adapted Fourier sampling for generative compressed sensing. In NeurIPS 2023 Workshop on Deep Learning and Inverse Problems , 2023.
- [18] S. Bhadra, V. A. Kelkar, F. J. Brooks, and M. A. Anastasio. On hallucinations in tomographic image reconstruction. IEEE Trans. Med. Imaging , 40(11):3249-3260, 2021.

- [19] P. J. Bickel, Y . Ritov, and A. B. Tsybakov. Simultaneous analysis of Lasso and Dantzig selector. Ann. Statist. , 37(4):1705-1732, 2009.
- [20] J. Bigot, C. Boyer, and P. Weiss. An analysis of block sampling strategies in compressed sensing. IEEE Trans. Inform. Theory , 62(4):2125-2139, 2016.
- [21] P. Bohra, J. Pham, T.-A. Dong, and M. Unser. Bayesian inversion for nonlinear imaging models using deep generative priors. IEEE Trans. Comput. Imag. , 8:1237-1249, 2023.
- [22] A. Bora, A. Jalal, E. Price, and A. G. Dimakis. Compressed sensing using generative models. In International Conference on Machine Learning , pages 537-546, 2017.
- [23] A. Bourrier, M. E. Davies, T. Peleg, P. Pérez, and R. Gribonval. Fundamental performance limits for ideal decoders in high-dimensional linear inverse problems. IEEE Trans. Inform. Theory , 60(12):7928-7946, 2014.
- [24] C. Boyer, J. Bigot, and P. Weiss. Compressed sensing with structured sparsity and structured acquisition. Appl. Comput. Harmon. Anal. , 46(2):312-350, 2019.
- [25] M. Burger and T. Roith. Learning in image reconstruction: A cautionary tale. SIAM News , 57(08), Oct 2024.
- [26] E. J. Candès and Y. Plan. A probabilistic and RIPless theory of compressed sensing. IEEE Trans. Inform. Theory , 57(11):7235-7254, 2011.
- [27] I.-Y. Chun and B. Adcock. Compressed sensing and parallel acquisition. IEEE Trans. Inform. Theory , 63(8):4860-4882, 2017.
- [28] H. Chung and J. C. Ye. Score-based diffusion models for accelerated mri. Medical Image Analysis , 80:102479, 2022.
- [29] M. J. Colbrook, V. Antun, and A. C. Hansen. The difficulty of computing stable and accurate neural networks: On the barriers of deep learning and smale's 18th problem. Proc. Natl. Acad. Sci. USA , 119(12):e2107151119, 2022.
- [30] M. Dashti and A. M. Stuart. The bayesian approach to inverse problems. In R. Ghanem et al., editor, Handbook of Uncertainty Quantification . Springer, 2017.
- [31] M. A. Davenport, M. F. Duarte, Y. C. Eldar, and G. Kutyniok. Introduction to compressed sensing. In Y. C. Eldar and G. Kutyniok, editors, Compressed Sensing: Theory and Applications , pages 1-64. Cambridge University Press, Cambridge, UK, 2012.
- [32] M. Dhar, A. Grover, and S. Ermon. Modeling sparse deviations for compressed sensing using generative models. In International Conference on Machine Learning , pages 1214-1223. PMLR, 2018.
- [33] A. G. Dimakis. Deep generative models and inverse problems. In P. Grohs and G. Kutyniok, editors, Mathematical Aspects of Deep Learning , chapter 9, pages 400-421. Cambridge University Press, Cambridge, UK, 2022.
- [34] S. Dirksen. Dimensionality reduction with subgaussian matrices: a unified theory. Found. Comput. Math. , 16:1367-1396, 2016.
- [35] D. L. Donoho, A. Maleki, and A. Montanari. Message-passing algorithms for compressed sensing. Proc. Natl. Acad. Sci. USA , 106(45):18914-18919, 2009.
- [36] M. F. Duarte and Y. C. Eldar. Structured compressed sensing: from theory to applications. IEEE Trans. Signal Process. , 59(9):4053-4085, 2011.
- [37] M. Fazlyab, A. Robey, H. Hassani, M. Morari, and G. J. Pappas. Efficient and accurate estimation of Lipschitz constants for deep neural networks. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 33. Curran Associates, Inc., 2019.
- [38] B. T. Feng, J. Smith, M. Rubinstein, H. Chang, K. L. Bouman, and W. T. Freeman. Score-based diffusion models as principled priors for inverse imaging. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV) , pages 10486-10497, 2023.
- [39] S. Foucart and H. Rauhut. A Mathematical Introduction to Compressive Sensing . Appl. Numer. Harmon. Anal. Birkhäuser, New York, NY, 2013.
- [40] M. Genzel, J. Macdonald, and M. Marz. Solving inverse problems with deep neural networks robustness included? IEEE Trans. Pattern Anal. Mach. Intell. , 45(1):1119-1134, 2023.

- [41] C. R. Givens and R. M. Shortt. A class of Wasserstein metrics for probability distributions. Michigan Math. J. , 31(2):231-240, 1984.
- [42] N. M. Gottschling, V. Antun, A. C. Hansen, and B. Adcock. The troublesome kernel - on hallucinations, no free lunches and the accuracy-stability trade-off in inverse problems. SIAM Rev. , 67(1):73-104, 2025.
- [43] P. Hand, O. Leong, and V. Voroninski. Phase retrieval under a generative prior. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 31. Curran Associates, Inc., 2018.
- [44] P. Hand and V. Voroninski. Global guarantees for enforcing deep generative priors by empirical risk. In S. Bubeck, V. Perchet, and P. Rigollet, editors, Proceedings of the Thirty-First Conference on Learning Theory , volume 75 of Proceedings of Machine Learning Research , pages 970-978. PMLR, 2018.
- [45] D. P. Hoffman, I. Slavitt, and C. A. Fitzpatrick. The promise and peril of deep learning in microscopy. Nature Methods , 18(2):131-132, 2021.
- [46] M. Holden, M. Pereyra, and K. C. Zygalakis. Bayesian imaging with data-driven priors encoded by neural networks. SIAM J. Imaging Sci. , 15(2):892-924, 2022.
- [47] Y. Huang, T. Würfl, K. Breininger, L. Liu, G. Lauritsch, and A. Maier. Some investigations on robustness of deep learning in limited angle tomography. In International Conference on Medical Image Computing and Computer-Assisted Intervention , pages 145-153, 2018.
- [48] A. Jalal, M. Arvinte, G. Daras, E. Price, A. G. Dimakis, and J. Tamir. Robust compressed sensing mri with deep generative priors. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P. S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems , volume 34, pages 14938-14954. Curran Associates, Inc., 2021.
- [49] A. Jalal, S. Karmalkar, A. Dimakis, and E. Price. Instance-optimal compressed sensing via posterior sampling. In 38th International Conference on Machine Learning , pages 4709-4720, 2021.
- [50] A. Jalal, L. Liu, A. G. Dimakis, and C. Caramanis. Robust compressed sensing using generative models. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems , volume 33, pages 713-727. Curran Associates, Inc., 2020.
- [51] Z. Kadkhodaie and E. Simoncelli. Stochastic solutions for linear inverse problems using the prior implicit in a denoiser. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems , volume 34, pages 13242-13254. Curran Associates, Inc., 2021.
- [52] A. Karan, K. Shah, S. Chen, and Y. C. Eldar. Unrolled denoising networks provably learn to perform optimal Bayesian inference. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems , volume 37, pages 135264-135298. Curran Associates, Inc., 2024.
- [53] B. Kawar, M. Elad, S. Ermon, and J. Song. Denoising diffusion restoration models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems , volume 35, pages 23593-23606. Curran Associates, Inc., 2022.
- [54] B. Kawar, G. Vaksman, and M. Elad. SNIPS: solving noisy inverse problems stochastically. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems , volume 34, pages 21757-21769. Curran Associates, Inc., 2021.
- [55] R. F. Laine, I. Arganda-Carreras, R. Henriques, and G. Jacquemet. Avoiding a replication crisis in deep-learning-based bioimage analysis. Nature Methods , 18(10):1136-1144, 2021.
- [56] R. Laumont, V. D. Bortoli, A. Almansa, J. Delon, A. Durmus, and M. Pereyra. Bayesian imaging using plug &amp; play priors: When langevin meets tweedie. SIAM J. Imaging Sci. , 15(2):701-737, 2022.
- [57] X. Liu, B. Glocker, M. M. McCradden, M. Ghassemi, A. K. Denniston, and L. Oakden-Rayner. The medical algorithmic audit. The Lancet Digital Health , 4(5):e384-e397, 2022.

- [58] G. Luo, M. Blumenthal, M. Heide, and M. Uecker. Bayesian MRI reconstruction with joint uncertainty estimation using diffusion models. Magn. Reson. Med. , 90(1):295-311, 2023.
- [59] T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida. Spectral normalization for generative adversarial networks. In International Conference on Learning Representations , 2018.
- [60] J. N. Morshuis, S. Gatidis, M. Hein, and C. F. Baumgartner. Adversarial robustness of MR image reconstruction under realistic perturbations. arXiv:2208.03161 , 2022.
- [61] Matthew J Muckley, Bruno Riemenschneider, Alireza Radmanesh, Sunwoo Kim, Geunu Jeong, Jingyu Ko, Yohan Jun, Hyungseob Shin, Dosik Hwang, Mahmoud Mostapha, et al. Results of the 2020 fastMRI challenge for machine learning MR image reconstruction. IEEE Trans. Med. Imaging , 2021.
- [62] M. Neyra-Nesterenko and B. Adcock. NESTANets: stable, accurate and efficient neural networks for analysis-sparse inverse problems. Sampl. Theory Signal Process. Data Anal. , 21:4, 2023.
- [63] C. R. Noordman, D. Yakar, J. Bosma, F. F. J. Simonis, and H. Huisman. Complexities of deep learning-based undersampled MR image reconstruction. Eur. Radiol. Exp. , 7:58, 2023.
- [64] G. Ongie, A. Jalal, C. A. Metzler, R. G. Baraniuk, A. G. Dimakis, and R. Willett. Deep learning techniques for inverse problems in imaging. IEEE J. Sel. Areas Inf. Theory , 1(1):39-56, 2020.
- [65] A. Raj, Y. Bresler, and B. Li. Improving robustness of deep-learning-based image reconstruction. In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning , volume 119 of Proceedings of Machine Learning Research , pages 7932-7942. PMLR, 13-18 Jul 2020.
- [66] A. J. Reader and B. Pan. AI for PET image reconstruction. Brit. J. Radiol. , 96(1150):20230292, 2023.
- [67] J. Romberg. Compressive sensing by random convolution. SIAM J. Imaging Sci. , 2(4):10981128, 2009.
- [68] J. Scarlett, R. Heckel, M. R. D. Rodrigues, P. Hand, and Y. C. Eldar. Theoretical perspectives on deep learning methods in inverse problems. IEEE J. Sel. Areas Inf. Theory , 3(3):433-453, 2022.
- [69] V. Shah and C. Hegde. Solving linear inverse problems using GAN priors: An algorithm with provable guarantees. In 2018 IEEE international conference on Acoustics, Speech and Signal Processing (ICASSP) conference on acoustics, speech and signal processing (ICASSP) , pages 4609-4613. IEEE, 2018.
- [70] Y. Song, L. Shen, L. Xing, and S. Ermon. Solving inverse problems in medical imaging with score-based generative models. In International Conference on Learning Representations , 2022.
- [71] A. M. Stuart. Inverse problems: a Bayesian perspective. Acta Numer. , 19:451-559, 2010.
- [72] C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, J. Ian Goodfellow, and R. Fergus. Intriguing properties of neural networks. In Proceedings of the International Conference on Learning Representations , 2014.
- [73] Y. Traonmilin and R. Gribonval. Stable recovery of low-dimensional cones in Hilbert spaces: one RIP to rule them all. Appl. Comput. Harmon. Anal. , 45(1):170-205, 2018.
- [74] G. Varoquaux and V. Cheplygina. Machine learning for medical imaging: methodological failures and recommendations for the future. NPJ digital medicine , 5(1):1-8, 2022.
- [75] A. Virmaux and K. Scaman. Lipschitz regularity of deep neural networks: analysis and efficient estimation. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 31. Curran Associates, Inc., 2018.
- [76] E. Wu, K. Wu, R. Daneshjou, D. Ouyang, D. E. Ho, and J. Zou. How medical AI devices are evaluated: limitations and recommendations from an analysis of FDA approvals. Nature Medicine , 27(4):582-584, 2021.
- [77] T. Yu, T. Hilbert, G. G. Piredda, A. Joseph, G. Bonanno, S. Zenkhri, P. Omoumi, M. B. Cuadra, E. J. Canales-Rodríguez, T. Kober, et al. Validation and generalizability of self-supervised image reconstruction methods for undersampled MRI. arXiv:2201.12535 , 2022.
- [78] M. Zach, F. Knoll, and T. Pock. Stable deep MRI reconstructions using generative priors. IEEE Trans. Med. Imag. , 42(12):3817-3831, 2023.

- [79] C. Zhang, J. Jia, B. Yaman, S. Moeller, S. Liu, M. Hong, and M. Akçakaya. Instabilities in conventional multi-coil MRI reconstruction with small adversarial perturbations. In 2021 55th Asilomar Conference on Signals, Systems, and Computers , pages 895-899, 2021.
- [80] Z. Zhao, J. C. Ye, and Y. Bresler. Generative models for inverse imaging problems: from mathematical foundations to physics-driven applications. IEEE Signal Process. Mag. , 40(1):148163, 2023.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We thoroughly discuss the main claims made in the abstract and introduction and the necessary assumptions to show them. Our main theoretical contributions directly address these claims. We also have several further remarks after these results to provide additional context for our work.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss limitations at the end of the paper in §5.

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

Justification: We have a detailed discussion of the assumptions needed to show our theoretical results in §2. We provide further discussion and present the results themselves in §3-4. We provide full proofs of our results in the supplemental material.

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

Justification: There are no experiments in the paper.

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

Justification: There are no experiments in the paper.

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

Justification: There are no experiments in the paper.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: There are no experiments in the paper.

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

Justification: There are no experiments in the paper.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have complied with the NeurIPS Code of Ethics in the preparation of this manuscript.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This work is primarily foundational, and the examples considered do not directly impact society.

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

Justification: This paper is theoretical, and does not involve any data or models.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: This paper is theoretical, and does not use any existing assets.

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

Justification: This paper is theoretical, and does not introduce any new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: No experiments were conducted in this paper.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: No experiments were conducted in this paper.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this paper does not involve LLMs as an important, original, or non-standard component.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Covering number estimates and the proofs of Propositions 4.1 and 4.3

The proof of Proposition 4.1 relies on the following two lemmas.

Lemma A.1 (Approximate covering number under a Lipschitz pushforward map) . Let G : R k → R n be Lipschitz with constant L ≥ 0 , i.e.,

<!-- formula-not-decoded -->

and define P = G♯γ , where γ is any probability distribution on R k . Then

<!-- formula-not-decoded -->

Proof. Let { x i } k i =1 ⊆ supp( γ ) and define z i = G ( x i ) ∈ supp( G♯γ ) for i = 1 , . . . , k . Let z ∈ G ( B η/L ( x i )) and write z = G ( x ) for some x ∈ B η/L ( x i ) . Then

<!-- formula-not-decoded -->

Hence z ∈ B η ( z i ) . Since z was arbitrary, we deduce that G ( B η/L ( x i )) ⊆ B η ( z i ) . It follows that B η/L ( x i ) ⊆ G -1 ( B η ( z i )) . Now suppose that γ [ ⋃ k i =1 B η/L ( x i ) ] ≥ 1 -δ . Then, by definition of the pushforward measure

<!-- formula-not-decoded -->

This gives the result.

Lemma A.2 (Approximate covering number of a normal distribution) . Let P = N (0 , σ 2 I ) on R n . Then its approximate covering number (Definition 2.1) satisfies

<!-- formula-not-decoded -->

Proof. Observe that, for t ≥ 0 ,

<!-- formula-not-decoded -->

where X ∼ χ 2 n is a chi-squared random variable, and the inequality follows from a standard Chernoff bound. Now t 2 ≤ e 2 t gives that

<!-- formula-not-decoded -->

Now set t = 1 + √ 2 n log(1 /δ ) so that P ( B c √ nσt ) ≤ δ . Hence, we have shown that

<!-- formula-not-decoded -->

where Cov η is the classical covering number of a set, i.e.,

<!-- formula-not-decoded -->

Using standard properties of covering numbers (see, e.g., [5, Lem. 13.22], we get

<!-- formula-not-decoded -->

as required.

Proof of Proposition 4.1. By Lemma A.1,

<!-- formula-not-decoded -->

The result now follows from Lemma A.2.

To prove Proposition 4.3, we first require the following lemma.

Lemma A.3 (Approximate covering number of a mixture) . Let P = ∑ r i =1 p i P i be a mixture of probability distributions P i on R n , where p i ≥ 0 , ∀ i , and ∑ r i =1 p i = 1 . Then

<!-- formula-not-decoded -->

Proof. For each i = 1 , . . . , r , let { x ( i ) j } k i j =1 ⊆ R n , and, in particular, 3 x ( i ) j ∈ supp( P i ) , be such that

<!-- formula-not-decoded -->

Then

<!-- formula-not-decoded -->

Notice that supp( P i ) ⊆ supp( P ) , therefore, x ( i ) j ∈ supp( P ) . The result now follows.

Proof of Proposition 4.3. Let S = { S : S ⊆ { 1 , . . . , n } , | S | = s } . Then we can write P s as the mixture

<!-- formula-not-decoded -->

where P S is defined as follows: x ∼ P S if x i = 0 for i / ∈ S and, for i ∈ S , x i is drawn independently from the standard normal distribution on R . Notice that P S = G♯γ , where γ = N (0 , I ) is the standard, multivariate normal distribution on R s and G : R s → R n is a zero-padding map. The map G is Lipschitz with constant L = 1 . Hence, by Lemmas A.1 and A.2,

<!-- formula-not-decoded -->

We now apply Lemma A.3 and the fact that

<!-- formula-not-decoded -->

the latter being a standard bound, to obtain

<!-- formula-not-decoded -->

Taking logarithms gives the result.

## B Concentration inequalities and the proofs of Theorems 3.5 and 3.9

We now aim to prove Theorems 3.5 and 3.9. To do this, we first derive concentration inequalities for subsgaussian random matrices and subsampled orthogonal transforms.

## B.1 Gaussian concentration and density shift bounds

Lemma B.1 (Concentration and density shift bounds for Gaussian noise) . Let E = N (0 , σ 2 m I ) . Then the upper concentration bound D upp ( t ; E ) (Definition 2.4) can be taken as

<!-- formula-not-decoded -->

and the density shift bound D shift ( ε, τ ; E ) (Definition 2.5) can be taken as

<!-- formula-not-decoded -->

Proof. Write e ∼ E as e = σ √ m n , where n ∼ N (0 , I ) . Then

<!-- formula-not-decoded -->

where X = ∥ n ∥ 2 ∼ χ 2 m is a chi-squared random variable with m degrees of freedom. Using a standard Chernoff bound once more, we have

<!-- formula-not-decoded -->

for any z &gt; 1 . Setting z = t 2 σ 2 , we have

<!-- formula-not-decoded -->

which gives the first result.

For the second result, we recall that E has density

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore

Now suppose that ∥ u ∥ ≤ τ and ∥ u -v ∥ ≤ ε . Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence which gives the second result.

## B.2 Subgaussian concentration inequalities

Lemma B.2 (Lower and upper concentration bounds for subgaussian random matrices) . Let A be a distribution of subgaussian random matrices with parameters β, κ &gt; 0 (Definition 3.4). Then the lower ad upper concentration bounds for A (Definition 2.3) can be taken as

<!-- formula-not-decoded -->

for any t &gt; 1 , where c ( t, β, κ ) &gt; 0 depends on β, κ only.

Proof. Let x ∈ R n and observe that

<!-- formula-not-decoded -->

We now use [39, Lem. 9.8]. Note that this result only considers a bound of the form P ( |∥ Ax ∥ 2 -∥ x ∥ 2 | ≥ s ∥ x ∥ 2 ) for s ∈ (0 , 1) . But the proof straightforwardly extends to s &gt; 0 .

## B.3 Concentration inequalities for randomly-subsampled orthogonal transforms

Lemma B.3 (Concentration bounds for randomly-subsampled orthogonal transforms) . Let D ⊆ R n and A be a distribution of randomly-subsampled orthognal transforms based on a matrix U (Definition 3.7). Then the lower and upper concentration bounds for A (Definition 2.3) can be taken as

<!-- formula-not-decoded -->

for any t &gt; 1 , where µ ( U ; D ) is a in Definition 3.8 and c ( t ) &gt; 0 depend on t only.

Proof. Due to (B.1), it suffices to bound

<!-- formula-not-decoded -->

for s &gt; 0 . The result uses Bernstein's inequality for bounded random variables (see, e.g., [5, Thm. 12.18]). Let x ∈ D . By definition of A and the fact that U is orthogonal, we can write

<!-- formula-not-decoded -->

Notice that the random variables Z i are independent, with E ( Z i ) = 0 . We also have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Therefore, by Bernstein's inequality,

<!-- formula-not-decoded -->

for any s &gt; 0 and x ∈ D . The result now follows.

Lemma B.4 (Absolute concentration bounds for subsampled orthogonal transforms) . Let D ⊆ R n and A be a distribution of randomly-subsampled orthogonal transforms based on a matrix U (Definition 3.7). Then ∥ A ∥ ≤ √ n/m a.s. for A ∼ A , and consequently the absolute concentration bound for A (Definition 2.3) can be taken as C abs ( s, t ; A , D ) = 0 for any s ≥ 0 and t ≥ s √ n/m .

Proof. Recall that A consists of q rows of an orthogonal matrix U multiplied by the scalar √ n/m . Hence ∥ A ∥ ≤ √ n/m ∥ U ∥ = √ n/m . Now let x ∈ D with ∥ x ∥ ≤ s . Then ∥ Ax ∥ ≤ ∥ A ∥∥ x ∥ ≤ ∥ A ∥ s . Therefore ∥ Ax ∥ √ n/ms ≤ t , meaning that P ( ∥ Ax ∥ &gt; t ) = 0 . This gives the result.

## B.4 Proofs of Theorems 3.5 and 3.9

Proof of Theorem 3.5. Let p = P [ ∥ x ∗ -ˆ x ∥ ≥ 34( η + σ )] . Weuse Theorem 3.1 with c = 32 , c ′ = 2 , t = 2 and ε replaced by ε/d , where d ≥ 1 is a constant that will be chosen later. Let ε ′ = ε/ ( dδ 1 /p ) . Then Theorem 3.1 gives

<!-- formula-not-decoded -->

Consider C abs ( ε ′ , 2 ε ′ ; A , R n ) . If x ∈ R n with ∥ x ∥ ≤ s , then

<!-- formula-not-decoded -->

Hence, in this case, we may take

<!-- formula-not-decoded -->

Now by Lemma B.2, we have that

<!-- formula-not-decoded -->

where c ( β, κ ) &gt; 0 depends on β, κ only. Also, by Lemma B.1, we have

<!-- formula-not-decoded -->

for some universal constant c &gt; 0 and

<!-- formula-not-decoded -->

where we used the facts that σ ≥ ε/δ 1 /p = dϵ ′ and d ≥ 1 . We deduce that

<!-- formula-not-decoded -->

for a possibly different constant c ( β, κ ) &gt; 0 . We now choose d = d ( β, κ ) = 12 /c ( β, κ ) . Up to another possible change in c ( β, κ ) , the condition (3.4) on m and (3.1) now give that p ≲ δ , as required.

Proof of Theorem 3.9. Let p = P [ ∥ x ∗ -ˆ x ∥ ≥ 34( η + σ )] . In this case, the forwards operator A satisfies ∥ A ∥ ≤ √ n/m (Lemma B.4). Hence we may apply Theorem 1.1 with θ = √ n/m and d = 2 to obtain

<!-- formula-not-decoded -->

for some universal constant c &gt; 0 , where D = supp( P ) -supp( P ) . Lemma B.3 now gives that

<!-- formula-not-decoded -->

for a possibly different constant c &gt; 0 . The result now follows from the condition (3.5) on m .

## C Proofs of Theorems 1.1 and 3.1

We finally consider the proofs of the two general results, Theorems 1.1 and 3.1. Our main effort will be in establishing the latter, from which the former will follow after a short argument.

To prove Theorem 3.1, we first require some additional background on couplings, along with several lemmas. This is given in §C.1. We then establish a series of key technical lemmas, presented in §C.2, which are used in the main proof. Having shown these, the proof of Theorem 3.1 proceeds in §C.3 via a series of step. We now briefly describe these steps, and by doing so explain how the sets D 1 , D 2 defined in (3.2)-(3.3) arise.

- (i) First, using Lemma C.7, we decompose P , R into distributions P ′ , R ′ that are supported in balls of a given radius, plus remainder terms. The distributions P ′ , R ′ are close (in W ∞ ) to a discrete distribution Q supported at the centres of the balls that give the approximate cover satisfying (3.1).
- (ii) Next, we replace x ∗ ∼ R in the definition of the probability p in Theorem 3.1 by z ∗ ∼ P ′ . The is done to align the prior with the posterior P ( ·| y, A ) , which is needed later in the proof. We do this using Lemma C.6. Here we have to consider the action of A on vectors of the form x -z , where x ∈ supp( R ′ ) and z ∈ supp( P ′ ) . After determining the supports of R ′ , P ′ , we see that x -z ∈ D 1 , where D 1 is the set defined in (3.2).
- (iii) We now decompose P ′ into a mixture over the balls mentioned in (i). After a series of arguments, we reduce the task to that of considering the probability that the conditional distribution is drawn from one ball when the prior is drawn from another. Lemmas C.4 and C.5 handle this. They involve estimating the action of A on vectors x -z , where z is the centre of one of the balls and x comes from another ball. Since the balls are supported in supp( P ) we have x ∈ supp( P ) , and since the centres come from the approximate covering number bound (3.1), we have that z ∈ supp( P ) if P attains the minimum and z ∈ supp( R ) otherwise. Hence x -z ∈ D 2 , with D 2 as in (3.3).

## C.1 Background on couplings

For a number of our results, we require some background on couplings. We first recall some notation. Given probability spaces ( X, F 1 , µ ) , ( Y, F 2 , ν ) , we write Γ = Γ µ,ν for the set of couplings, i.e., probability measures on the product space ( X × Y, σ ( F 1 ⊗ F 2 )) whose marginals are µ and ν , respectively. For convenience, we write π 1 : X × Y → X and π 2 : X × Y → Y for the projections π 1 ( x, y ) = x and likewise π 2 ( x, y ) = y . In particular, for any coupling γ we have π 1 ♯γ = µ and π 2 ♯γ = ν , where ♯ denotes the pushforward operation. As an immediate consequence, we observe that for any measurable function φ : X → R ,

<!-- formula-not-decoded -->

Given a cost function c : X × Y → [0 , ∞ ) , the Wassersteinp metric is defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We say that γ ∈ Γ is a W p -optimal coupling of µ and ν if

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

when p = ∞ . Note that such a coupling exists whenever X,Y are Polish spaces, and when the cost function is lower semicontinuous [41]. In our case, we generally work with Euclidean spaces with the cost function being the Euclidean norm, hence both conditions are satisfied.

for 1 ≤ p &lt; ∞ and for 1 ≤ p &lt; ∞ or

For convenience, if γ is probability measure on the product space ( X × Y, σ ( F 1 ⊗F 2 )) , we will often write γ ( E 1 , E 2 ) instead of γ ( E 1 × E 2 ) for E i ∈ F i , i = 1 , 2 . Moreover, if x ∈ X is a singleton, we write γ ( x, E 2 ) for γ ( { x } × E 2 ) and likewise for γ ( E 1 , y ) .

We now need several lemmas on couplings.

Lemma C.1. Suppose that ( X, F 1 , µ ) , ( Y, F 2 , ν ) are Borel probability spaces, and let γ be a coupling of µ, ν on the space ( X × Y, σ ( F 1 ⊗F 2 )) . Then supp( γ ) ⊆ supp( µ ) × supp( ν ) .

Proof. Let ( x, y ) ∈ supp( γ ) . Then γ ( U x,y ) &gt; 0 for every open set U x,y ⊆ X × Y that contains ( x, y ) . Now, to show that ( x, y ) ∈ supp( µ ) × supp( ν ) , we show that x ∈ supp( µ ) and y ∈ supp( ν ) .

Let U x ⊆ X be open with x ∈ U x . By definition, µ ( U x ) = γ ( U x × R n ) . Since U x × R n is open and contains ( x, y ) , it follows that γ ( U x × R n ) &gt; 0 . Since U x was arbitrary, we deduce that x ∈ supp( µ ) . The argument that y ∈ supp( ν ) is identical.

Lemma C.2. Let X be a Polish space with a complete metric d . Let µ, ν be Borel probability measures on X . Let d H be the Hausdorff metric with respect to d and W ∞ be the Wasserstein-∞ metric with cost function d . Then

<!-- formula-not-decoded -->

In particular, supp( µ ) ⊆ B η (supp( ν )) for any η ≥ W ∞ ( µ, ν ) .

Proof. Since

<!-- formula-not-decoded -->

we may, without loss of generality, assume that the maximum is achieved by sup x ∈ supp( ν ) d ( x, supp( µ )) =: D . Take a sequence { x n } n ∈ N ⊆ supp( ν ) such that D n := d ( x n , supp( µ )) → D . Since x n ∈ supp( ν ) , for any ε &gt; 0 , we have ν ( B ε ( x n )) &gt; 0 . Note that B ε ( x n ) is measurable as we assume X is Borel. For each n ∈ N , define ε n = 1 n D n . We show that for all x ∈ B ε n ( x n ) , y ∈ supp( µ ) , d ( x, y ) &gt; D n (1 -1 n ) . By triangle inequality, we have

<!-- formula-not-decoded -->

Notice also that D n (1 -1 n ) ≤ D and converges to D as n →∞ . This implies that

<!-- formula-not-decoded -->

Now consider any coupling γ ∈ Γ µ,ν . We have

<!-- formula-not-decoded -->

Therefore ess sup γ d ( x, y ) &gt; D n (1 -1 /n ) . Now since D n (1 -1 n ) → D we have ess sup γ d ( x, y ) ≥ D . This is holds for any coupling, therefore the result follows.

When working with a coupling between a finitely-supported distribution and a continuous distribution, the following lemma is often useful.

LemmaC.3. Let ( X, F 1 , µ ) , ( Y, F 2 , ν ) be probability spaces, such that ν is finitely supported on a set S ⊆ Y . Let γ be a coupling of µ, ν and E ⊆ X × Y be γ -measurable. Write E y = { x : ( x, y ) ∈ E } for the slice of E at y ∈ Y . Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˆ E = E \ ⋃ s ∈ S,s ∈ π 2 ( E ) ( E s ×{ s } ) . It suffices to show that γ ( ˆ E ) = 0 . Since F ⊆ π -1 2 ( π 2 ( F )) for any set F , we have

<!-- formula-not-decoded -->

But

<!-- formula-not-decoded -->

and therefore π 2 ( ˆ E ) ∩ S = ∅ . The result now follows.

Proof. Write

## C.2 Technical lemmas

## C.2.1 Separation lemma

The lemma considers a scenario where two random variables are drawn for a mixture of k probability distributions. The second random variable is conditioned on the draw of the first. It then considers the probability that the two random variables are drawn from different distributions in the mixture, bounding this in terms of their Total Variation (TV) distance. It generalizes [49, Lem. 3.1].

Lemma C.4 (Separation lemma) . Let H 1 , . . . , H k be Borel probability measures and consider the mixture H = ∑ k i =1 a i H i . Let y ∗ ∼ H and ˆ y ∼ ∑ k i =1 P ( y ∗ ∼ H i | y ∗ ) H i ( ·| y ∗ ) where P ( y ∗ ∼ H i | y ∗ ) are the posterior weights. Then

<!-- formula-not-decoded -->

To clarify, in this lemma and elsewhere we use the notation y ∗ ∼ H i (and similar) to mean the event that y ∗ is drawn from the i th distribution H i .

Proof. Note that if the H i have densities h i with respect to some measure, these weights are given by

<!-- formula-not-decoded -->

We now write

<!-- formula-not-decoded -->

Since E [ y ∗ | y ∗ ∼ H i ] ∼ H i , we have

<!-- formula-not-decoded -->

Now, because of the mixture property, H i ≪ H and therefore its Radon-Nikodym derivative h i = d H i d H exists. This means we may write

<!-- formula-not-decoded -->

By definition, we have P (ˆ y ∼ H j ( ·| y ∗ )) = P ( y ∗ ∼ H j | y ∗ ) and using (C.2), we deduce that

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

We now write

as required.

## C.2.2 Disjointly-supported measures induce well-separated measurement distributions

The following lemma pertains to the pushforwards of measures supported in R n via the forward operator A and noise e . Specifically, it states that if two distributions P int and P ext are disjointly supported then their corresponding pushforwards H int ,A and H ext ,A are, on average with respect to A ∼ A , well-separated, in the sense of their TV-distance. It is generalization of [49, Lem. 3.2] that allows for arbitrary distributions A of the forward operators, as opposed to just distributions of Gaussian random matrices.

Lemma C.5 (Disjointly-supported measures induce well-separated measurement distributions) . Let ˜ x ∈ R n , σ ≥ 0 , η ≥ 0 , c ≥ 1 , P ext be a distribution supported in the set

<!-- formula-not-decoded -->

and P int be a distribution supported in the set

<!-- formula-not-decoded -->

Given A ∈ R m × n , let H int ,A be the distribution of y = Ax ∗ + e where x ∗ ∼ P int and e ∼ E independently, and define H ext ,A in a similar way. Then

<!-- formula-not-decoded -->

where D ext = { x -˜ x : x ∈ supp( P ext ) } , D int = { x -˜ x : x ∈ supp( P int ) } and C upp ( · ; A ) , C low ( · ; A ) and D upp ( · ; E ) are as in Definitions 2.3 and 2.4, respectively.

Notice that the average TV-distance is bounded below by the concentration bounds C low and C upp for A (Definition 2.3) and the concentration bound D upp for E (Definition 2.4). This is unsurprising. The pushforward measures are expected to be well-separated if, firstly, the action of A approximately preserves the lengths of vectors (which explains the appearance of C low and C upp ) and, secondly, adding noise by E does not, with high probability, cause well-separated vectors to become close to each other (which explains the appearance of D upp ). Also as expected, as c increases, i.e., the distributions P int and P ext become further separated, the average TV-distance increases.

Proof. Given A ∈ R m × n , let

<!-- formula-not-decoded -->

We claim that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Notice that these claims immediately imply the result, since

<!-- formula-not-decoded -->

Therefore, the rest of the proof is devoted to showing (C.3) and (C.4). For the former, we write

<!-- formula-not-decoded -->

where B A -Ax = { b -Ax : b ∈ B A } . We now bound E x ∼P ext [ E A ∼A E ( B A -Ax )] . Given x ∈ R n , let C x = { A : ∥ Ax -A ˜ x ∥ &lt; 2 √ c ( η + σ ) } ⊆ R m × n and write

<!-- formula-not-decoded -->

so that

<!-- formula-not-decoded -->

We will bound I 1 , I 2 separately. For I 1 , we first write

<!-- formula-not-decoded -->

where the inequality follows from the fact that E ( B A -Ax ) ≤ 1 . Now since x ∼ P ext , we have x ∈ S ˜ x, ext and therefore ∥ x -˜ x ∥ ≥ c ( η + σ ) . Hence

<!-- formula-not-decoded -->

Since the outer expectation term has x ∼ P ext , we have that x ∈ supp( P ext ) with probability one. Using Definition 2.3, we deduce that

<!-- formula-not-decoded -->

We now bound I 2 . Let x ∈ S ˜ x, ext and A ∈ C c x , i.e., ∥ A ( x -˜ x ) ∥ &gt; 2 √ c ( η + σ ) . We now show that B A ⊆ B A,x , where B A,x = { y ∈ R m : ∥ y -Ax ∥ ≥ √ c ( η + σ ) } . Suppose that y ∈ B A , i.e., ∥ y -A ˜ x ∥ ≤ √ c ( η + σ ) . We have

<!-- formula-not-decoded -->

and therefore y ∈ B A,x , as required. Using this, we have

<!-- formula-not-decoded -->

and therefore

<!-- formula-not-decoded -->

But we notice that B A,x -Ax = B c √ c ( η + σ ) . Now since η ≥ 0 , we have B c √ c ( η + σ ) ⊆ B c σ √ c . Hence

<!-- formula-not-decoded -->

and therefore I 2 ≤ D upp ( σ √ c ; E ) . Combining this with (C.5), (C.6) and (C.7), we deduce that E A [ H ext ,A ( B A )] ≤ E x ∼P ext [ E A ∼A [ E ( B A -Ax )]] = I 1 + I 2 ≤ C low (2 / √ c ; A , D ext )+ C upp ( σ √ c ; E ) , which shows (C.3).

We will now establish (C.4). With similar reasoning to (C.5), we have

<!-- formula-not-decoded -->

Proceeding as before, let D x = { A : ∥ Ax -A ˜ x ∥ &lt; √ c 2 ( η + σ ) } , I 1 = E x ∼P int [ E A ∼A [ E ( B c A -Ax )]1 D c x ] , and I 2 = E x ∼P int [ E A ∼A [ E ( B c A -Ax )]1 D x ] so that

<!-- formula-not-decoded -->

The terms I 1 , I 2 are similar to those considered in the previous case. We bound them similarly. For I 1 , we have, by dropping the inner probability terms,

<!-- formula-not-decoded -->

Since x ∈ S ˜ x, int , we have ∥ x -˜ x ∥ ≤ η ≤ η + σ which gives

<!-- formula-not-decoded -->

and therefore

<!-- formula-not-decoded -->

We now bound I 2 . Let x ∈ S ˜ x, int and suppose that A ∈ D x , i.e., ∥ x -˜ x ∥ ≤ η and ∥ A ( x -˜ x ) ∥ &lt; √ c 2 ( η + σ ) . Define ˆ B A,x = { y ∈ R m : ∥ y -Ax ∥ &lt; √ c 2 ( η + σ ) } . We will show ˆ B A,x ⊆ B A in this case. Let y ∈ B A,x . Then

<!-- formula-not-decoded -->

as required. This implies that B c A ⊆ ˆ B c A,x . Hence

<!-- formula-not-decoded -->

which implies that I 2 ≤ D upp ( √ c 2 σ ; E ) . Combining with (C.8) and (C.9) we get

<!-- formula-not-decoded -->

which implies (C.4). This completes the proof.

## C.2.3 Replacing the real distribution with the approximate distribution

We next establish a result that allows one to upper bound the failure probability based on draws from the real distribution R with the failure probability based on draws from the approximate distribution P . This lemma is a key technical step that aligns the prior distribution with the posterior. The specific bound is given in terms of the Wasserstein distance between R and P and several of the concentration bounds defined in §2. This is a significant generalization of [49, Lem. 3.3] that allows for arbitrary distributions A , E for the forwards operator and noise.

Lemma C.6 (Replacing the real distribution with the approximate distribution) . Let ε, σ, d, t ≥ 0 , c ≥ 1 , E be a distribution on R m and R , P be distributions on R n such that W ∞ ( R , P ) ≤ ε . Let Π be an W ∞ -optimal coupling of R and P and define the set D = { x ∗ -z ∗ : ( x ∗ , z ∗ ) ∈ supp(Π) } . Let

<!-- formula-not-decoded -->

and

Then

<!-- formula-not-decoded -->

where C abs ( ε, tε ; A , D ) , D upp ( cσ ; E ) and D shift ( tε, cσ ; E ) are as in Definitions 2.3, 2.4 and 2.5, respectively.

As expected, this lemma involves a trade-off. The constant C abs ( ε, tε ; A , D ) is made smaller (for fixed ε ) by making the constant t larger. However, this increases D shift ( tε, cσ ; E ) , which is compensated by making c smaller. However, this in turn increases the constant D upp ( cσ ; E ) .

Proof. Define the events so that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Observe that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and similarly

<!-- formula-not-decoded -->

Therefore, to obtain the result, it suffices to replace samples from the real distribution R with samples from the approximate distribution P and to replace the indicator function of B 1 , ˆ x by the indicator function over B 2 , ˆ z . For the first task, we use couplings. Since W ∞ ( R , P ) ≤ ε , there exists a coupling Π between R , P with Π( ∥ x ∗ -z ∗ ∥ ≤ ε ) = 1 . By (C.1), we can write

<!-- formula-not-decoded -->

Define E = { ( x ∗ , z ∗ ) : ∥ x ∗ -z ∗ ∥ ≤ ε } and observe that Π( E ) = 1 . Then, for fixed A , e , we have

<!-- formula-not-decoded -->

We now show 1 B 1 , ˆ x ( x ∗ ) ≤ 1 B 2 , ˆ x ( z ∗ ) for ( x ∗ , z ∗ ) ∈ E . Let ( x ∗ , z ∗ ) ∈ E and suppose that x ∗ ∈ B 1 , ˆ x . Then ∥ x ∗ -ˆ x ∥ ≥ d + ε and, since ∥ x ∗ -z ∗ ∥ ≤ ε , we also have that ∥ z ∗ -ˆ x ∥ ≥ d and therefore z ∗ ∈ B 2 , ˆ x , as required. Hence

<!-- formula-not-decoded -->

for ( x ∗ , z ∗ ) ∈ E . Now, since indicator functions are non-negative, Fubini's theorem immediately implies that

<!-- formula-not-decoded -->

Having introduced the coupling Π and replaced 1 B 1 , ˆ x by 1 B 2 , ˆ x , to establish the result it remains to replace the conditional distribution P ( ·| Ax ∗ + e, A ) by P ( ·| Az ∗ + e, A ) . With a similar technique to that used in the proof of Lemma C.5, we define C x ∗ ,z ∗ = { A : ∥ A ( x ∗ -z ∗ ) ∥ &gt; tε } and

<!-- formula-not-decoded -->

so that

<!-- formula-not-decoded -->

We first bound I 1 . As before, we write

<!-- formula-not-decoded -->

Recalling the definition of the set E above, we get

<!-- formula-not-decoded -->

Using the definition of C 0 , E and D , we deduce that

<!-- formula-not-decoded -->

Now we bound I 2 . We further split the integral I 2 as follows:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let us first find an upper bound for I 2 1 . We have

<!-- formula-not-decoded -->

and therefore, by Definition 2.4,

<!-- formula-not-decoded -->

We now find a bound for I 2 2 . We first use Definition 2.5 to write

<!-- formula-not-decoded -->

Now define the new variable e ′ = e + A ( x ∗ -z ∗ ) . Since, in the integrand, ∥ e ∥ ≤ cσ (due to the indicator function 1 B cσ ( e ) ) and ∥ A ( x ∗ -z ∗ ) ∥ ≤ tε (due to the indicator function 1 C c x ∗ ,z ∗ ( A ) ), Definition 2.5 yields the bound

<!-- formula-not-decoded -->

We now drop the first two indicator functions and relabel the variables e ′ and ˆ x as e and ˆ z , respectively, to obtain

<!-- formula-not-decoded -->

This gives

<!-- formula-not-decoded -->

where q is as in (C.10). Combining this with (C.13), we deduce that

<!-- formula-not-decoded -->

To complete the proof, we combine this with (C.11) and (C.12), and then recall (C.10) once more.

## C.2.4 Decomposing distributions

The following lemma is in large part similar to [49, Lem. A.1]. However, we streamline and rewrite its proof for clarity and completeness, fix a number of small issues and make an addition to the statement (see item (v) below) that is important for proving our main result.

Lemma C.7 (Decomposing distributions) . Let R , P be arbitrary distributions on R n , p ≥ 1 and η, ρ, δ &gt; 0 . If W p ( R , P ) ≤ ρ and k ∈ N is such that

<!-- formula-not-decoded -->

then there exist distributions R ′ , R ′′ , P ′ , P ′′ , a constant 0 &lt; δ ′ ≤ δ and a discrete distribution Q with supp( Q ) = S satisfying

- (i) min { W ∞ ( P ′ , Q ) , W ∞ ( R ′ , Q ) } ≤ η ,
- (ii) W ∞ ( R ′ , P ′ ) ≤ ρ δ 1 /p ,

<!-- formula-not-decoded -->

- (iv) | S | ≤ e k ,
- (v) and S ⊆ supp( P ) if P attains the minimum in (C.15) with S ⊆ supp( R ) otherwise.

This lemma states that two distributions that are close in Wasserstein p -distance, and for which at least one has small approximate covering number (C.15), can be decomposed into mixtures (iii) of distributions, where the following holds. One of the distributions, say P ′ , is close (i) in Wasserstein-∞ distance to a discrete distribution Q with the cardinality of its support (iv) bounded by the approximate covering number. The other R ′ is close in Wasserstein-∞ distance to P ′ . Moreover, both mixtures (iii) are dominated by these distributions: the 'remainder' terms P ′′ and R ′′ are associated with a small constant δ ′ ≤ δ , meaning they are sampled with probability ≤ δ when drawing from either P or R . Note that if p &lt; ∞ then the Wasserstein-∞ distance between R ′ and P ′ may get larger as δ shrinks, i.e., as the remainder gets smaller. However, this does not occur when p = ∞ , as (ii) is independent of δ in this case.

Proof. Without loss of generality, we assume that log Cov η,δ ( P ) ≤ k . Then Cov η,δ ( P ) ≤ e k and hence there is a set S = { u i } l i =1 ⊆ supp( P ) with l ≤ e k , where the u i are the centres of the balls used to cover at least 1 -δ of the measure of P . That is,

<!-- formula-not-decoded -->

We now define f : R n → R so that f ( x ) = 0 if x lies outside these balls, and otherwise, f ( x ) is the equal to the reciprocal of the number of balls in which x is contained. Namely,

<!-- formula-not-decoded -->

We divide the remainder of the proof into a series of steps.

1. Construction of Q ′ . We will now define a finite measure Q ′ . The point of Q ′ is to, concentrate the mass of the measure P into the centres of the balls u i . If the sets B ( u i , η ) are disjoint, then this is straightforward. However, to ensure that Q ′ is indeed a probability measure, we need to normalize and account for any non-trivial intersections. This is done via the function f . Pick some arbitrary ˆ u / ∈ { u 1 , . . . , u l } and define

<!-- formula-not-decoded -->

Observe that

<!-- formula-not-decoded -->

and therefore Q ′ is a probability distribution supported on the finite set S ∪ { ˆ u } .

2. Coupling Q ′ , P . Now that we have associated all the mass of P with the points u i , we can define a coupling Π between Q ′ and P that associates the mass of P and u i with a single measure. Moreover, this measure will keep points within η distance of each other with high probability. We define Π as follows for measurable sets E,F ⊆ R n :

<!-- formula-not-decoded -->

To see that this is a coupling, we first observe that

<!-- formula-not-decoded -->

which gives the result for the first marginal. For the other, we have

<!-- formula-not-decoded -->

By definition of f , this is precisely

<!-- formula-not-decoded -->

which gives the result for the second marginal. Note that Π was only defined for product sets, but, since Q ′ is finitely supported, it follows directly from Lemma C.3 that it extends to arbitrary measurable sets in the product sigma-algebra. We now show that Π[ ∥ x 1 -x 2 ∥ &gt; η ] ≤ c ∗ ≤ δ . That is, we show that most points drawn from Π are within η distance of each other. By law of total probability we have

<!-- formula-not-decoded -->

where U i = { x : ∥ x -u i ∥ &gt; η } and ˆ U = { x : ∥ x -ˆ u ∥ &gt; η } . Notice that U i ∩ B ( u i , η ) = ∅ and therefore

Hence

<!-- formula-not-decoded -->

and, since Q ′ (ˆ u ) = c ∗ , we have Π( ˆ U, ˆ u ) Q ′ (ˆ u ) ≤ c ∗ ≤ δ . This gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

as required.

3. Coupling P , R . The next step is to introduce R . With the assumption that W p ( R , P ) ≤ ρ , by definition there exists a coupling Γ between P and R such that E Γ [ ∥ x 1 -x 2 ∥ p ] ≤ ρ p . Markov's inequality then gives that

<!-- formula-not-decoded -->

4. Coupling P , Q ′ , R . We next couple P , Q ′ and R . Before doing so, we first discuss the goal of our final coupling. Recall that we have the distribution P , the distribution Π that couples P , Q ′ closely except for up to δ mass of P , and Γ which keeps P , R close again except for up to δ of the mass of Γ . We want to decompose P into the portions that are η close to Q ′ , and points that are not. These will become P ′ and P ′′ , respectively. At the same time, we want to decompose R to points that are ρ δ 1 /p close to P ′ , and points that are not. Naturally this will become R ′ and R ′′ . To achieve this, we couple P , Q ′ and R in this step and then use this to construct the final decomposition in the next step.

We have measures P , Q ′ and R and couplings Π of P , Q ′ and Γ of P , R . We will in a sense, couple Π , Γ . Since ( R n ) 3 is a Polish space, by [10, Lem. 8.4], there exists a coupling Ω with

<!-- formula-not-decoded -->

where π 1 , 2 ( x 1 , x 2 , x 3 ) = ( x 1 , x 2 ) and likewise for π 1 , 3 . One should intuitively think of the x 1 component as samples from P , the x 2 component as samples from Q , and the x 3 component as samples from R . With the base measure defined, we still want to ensure that x 1 , x 3 are sampled

closely, and x 1 , x 2 are as well. Consider the event such that x 1 , x 3 are ρ δ 1 /p close and x 1 , x 2 are η close: namely,

<!-- formula-not-decoded -->

Split up the negation of the two events of ∥ x 1 -x 3 ∥ ≤ ρ/δ 1 /p and ∥ x 1 -x 2 ∥ ≤ η into the events

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so that E c = E 1 ∪ E 2 . We will now show Ω( E 1 ) ≤ δ . Write E 1 = E ′ 1 × R n where E ′ 1 = { ( x 1 , x 2 ) : ∥ x 1 -x 2 ∥ &gt; η } satisfies Π( E ′ 1 ) ≤ δ ′ by (C.16). Then

<!-- formula-not-decoded -->

as required. Using (C.17), we also have the analogous result for E 2 . Hence Ω( E c ) = Ω( E 1 ∪ E 2 ) ≤ Ω( E 1 ) + Ω( E 2 ) =: 2 δ ′ ≤ 2 δ ′ , and consequently,

<!-- formula-not-decoded -->

4. Decomposing P , R . Finally, we define P ′ , P ′′ , R ′ , R ′′ and Q by conditioning on the events E and E c , as follows:

<!-- formula-not-decoded -->

This gives

<!-- formula-not-decoded -->

and similarly

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now claim that these distributions satisfy (i)-(v) in the statement of the lemma. We have already shown that (iii) holds. To show (i), we define a coupling γ of P ′ , Q by γ ( B ) = Ω( B, R n | E ) for any B ⊆ ( R n ) 2 . Observe that γ ( A, R n ) = Ω( A, R n , R n | E ) = P ′ ( A ) and γ ( R n , A ) = Ω( R n , A, R n | E ) = Q ( A ) . Hence this is indeed a coupling of P ′ , Q . Therefore it suffices to show that γ ( B ) = 0 , where B is the event {∥ x 1 -x 2 ∥ &gt; η } . We have γ ( B ) = Ω( B, R n | E ) . Recall that for ( x 1 , x 2 , x 3 ) ∈ E , we have ∥ x 1 -x 2 ∥ ≤ η . Hence Ω( B, R n | E ) = 0 . Therefore, W ∞ ( P ′ , Q ) ≤ η , which gives (i).

Similarly, for (ii) we define a coupling γ ′ of R ′ , P ′ by γ ′ ( B ) = Ω( B | E ) where B := { ( x 1 , x 2 , x 3 ) : ( x 1 x 3 ) ∈ B,x 2 ∈ R n } . With similar reasoning as the previous case, γ ′ is a coupling of R ′ , P ′ and, for ( x 1 , x 2 , x 3 ) ∈ E we have ∥ x 1 -x 3 ∥ ≤ ρ/δ 1 /p , so letting B be the event {∥ x 1 -x 3 ∥ &gt; ρ/δ 1 /p } , we conclude that W ∞ ( R ′ , P ′ ) ≤ ρ/δ 1 /p . This gives (ii).

Finally we verify (iv) and (v). First recall that both properties hold for Q ′ by construction. The results now follow from the fact that Q ′ ( · ) = Ω( R n , · , R n ) and Q ( · ) = Ω( R n , · , R n | E ) .

## C.3 Proof of Theorem 3.1

We now prove Theorem 3.1. This follows a similar approach to that of [49, Thm. 3.4], but with a series of significant modifications to account for the substantially more general setup considered in this work. We also streamline the proof and clarify a number of key steps.

Proof of Theorem 3.1. By Lemma C.7, we can decompose P , R into measures P ′ , P ′′ and R ′ , R ′′ , and construct a finite distribution Q supported on a finite set S such that

- (i) min { W ∞ ( P ′ , Q ) , W ∞ ( R ′ , Q ) } ≤ η ,
- (ii) W ∞ ( R ′ , P ′ ) ≤ ε ′ := ε δ 1 /p ,
- (iii) P = (1 -2 δ ′ ) P ′ +(2 δ ′ ) P ′′ and R = (1 -2 δ ′ ) R ′ +(2 δ ′ ) R ′′ for some 0 ≤ δ ′ ≤ δ ,
- (iv) | S | ≤ e k ,
- (v) and S ⊆ supp( P ) if P attains the minimum in (C.15) with S ⊆ supp( R ) otherwise.

It is helpful to briefly recall the construction of these sets. Beginning with δ, η as parameters for the approximate covering numbers, the distribution Q concentrates 1 -δ of the mass of P into the centres of the η -radius balls used. Then the distributions P ′ , R ′ are the measures P , R within the balls. We now write

<!-- formula-not-decoded -->

Here, in the first inequality we used the fact that σ ≥ ε ′ , and in the second, we used the decomposition R = (1 -2 δ ′ ) R ′ +2 δ ′ R ′′ and the fact that δ ′ ≤ δ . We now bound q by using Lemma C.6 to replace the distribution R ′ by the distribution P ′ . Writing u = Az ∗ + e , this lemma and (ii) give that

<!-- formula-not-decoded -->

and D = { x ∗ -z ∗ : ( x ∗ , z ∗ ) ∈ supp(Π) } , for Π being the W ∞ -optimal coupling of R ′ , P ′ guaranteed by (ii). Lemma C.1 implies that supp(Π) ⊆ supp( R ′ ) × supp( P ′ ) and therefore

<!-- formula-not-decoded -->

Now (iii) implies that supp( P ′ ) ⊆ supp( P ) . Similarly, (iii) implies that supp( R ′ ) ⊆ supp( R ) . But Lemma C.2 and (ii) imply that supp( R ′ ) ⊆ B ε ′ (supp( P ′ )) . Therefore

<!-- formula-not-decoded -->

where D 1 as in (3.2).

We now bound r . Observe first that

<!-- formula-not-decoded -->

Indeed, from (i) either W ∞ ( P ′ , Q ) ≤ η or W ∞ ( R ′ , Q ) ≤ η . In the former case, the inequality trivially holds. In the latter case, we can use the triangle inequality and (ii) to obtain the desired bound. This implies that there is a coupling Γ of P ′ , Q with esssup Γ ∥ x -y ∥ ≤ η ′ . Fix ˜ z ∈ S and, for any Borel set E ⊆ R n , define

<!-- formula-not-decoded -->

Then it is readily checked that Γ ˜ z ( · ) defines a probability measure. Note also that Γ ˜ z is supported on a ball of radius η ′ around ˜ z , since esssup Γ ( ∥ x -y ∥ ) ≤ η ′ . Recall that Γ is a coupling between P ′ and Q . Let E ⊆ R n be a Borel set. Then Lemma C.3 gives that

<!-- formula-not-decoded -->

Therefore, we can express P ′ as the mixture

<!-- formula-not-decoded -->

Define the event E = {∥ z ∗ -ˆ z ∥ ≥ ( c +1)( η + σ ) } ⊆ R n × R n so that the probability r defined in (C.20) can be expressed as

<!-- formula-not-decoded -->

Using the above expression for P ′ we now write

<!-- formula-not-decoded -->

where the last line holds as Q (˜ z ) is a constant. Hence

<!-- formula-not-decoded -->

Now we bound each term in this sum. We do this by decomposing P into a mixture of three probability measures depending on ˜ z ∈ S . To do this, let θ = c ( η + σ ) and observe that, for any Borel set E ⊆ R n ,

<!-- formula-not-decoded -->

Now define the constants c ˜ z, mid = P ( B θ (˜ z )) -(1 -2 δ ′ ) Q (˜ z )Γ ˜ z ( B θ (˜ z )) , c ˜ z, ext = P ( B c θ (˜ z )) -(1 -2 δ ′ ) Q (˜ z )Γ ˜ z ( B c θ (˜ z )) . and let

<!-- formula-not-decoded -->

Then P can be expressed as the mixture

<!-- formula-not-decoded -->

To ensure this is a well-defined mixture, we need to show that P ˜ z, mid and P ˜ z, ext are probability measures. However, by (iii) we have, for any Borel set E ⊆ R n ,

<!-- formula-not-decoded -->

Therefore, P ˜ z, mid and P ˜ z, ext are well-defined, provided the constants c ˜ z, int , c ˜ z, ext &gt; 0 . However, if one of these constants is zero, then we can simply exclude this term from the mixture (C.22). For the rest of the theorem, we will assume that, at least, c ˜ z, ext &gt; 0 .

It is now useful to note that

<!-- formula-not-decoded -->

which follows immediately from their definitions, and also that

<!-- formula-not-decoded -->

where in the second inclusion we used the fact that η ′ = η + ε/δ 1 /p ≤ η + σ ≤ c ( η + σ ) = θ , as σ ≥ ε/δ 1 /p and c ≥ 1 .

We now return to the sum (C.21). Consider an arbitrary term. First, observe that, for z ∗ ∼ P , we have P ( z ∗ ∼ Γ ˜ z ) = Q (˜ z )(1 -2 δ ′ ) by (C.22). Hence

<!-- formula-not-decoded -->

Recall that z ∗ ∼ Γ ˜ z is supported in B η ′ (˜ z ) . Therefore, for the event E to occur, i.e., ∥ z ∗ -ˆ z ∥ &gt; ( c +1)( η + σ ) , it must be that ˆ z ∈ B c θ (˜ z ) , which means that ˆ z ∼ P ˜ z, ext ( ·| A,u ) . Hence

<!-- formula-not-decoded -->

Now fix A ∈ R m × n . Let H ˜ z, int ,A be the distribution of y ∗ = Az ∗ + e for z ∗ ∼ P ˜ z, int and e ∼ E independently, and define H ˜ z, ext ,A similarly. Then, by Fubini's theorem, we have

<!-- formula-not-decoded -->

Now let H ˜ z,A be the distribution of y = Az + e for z ∼ P and e ∼ E independently. Then Lemma C.4 (with H = H ˜ z,A , H 1 = H ˜ z, int ,A , H 2 = H ˜ z, mid ,A , H 3 = H ˜ z, ext ,A and a 1 = (1 -2 δ ′ ) Q (˜ z ) , a 2 = c ˜ z, mid , a 3 = c ˜ z, ext ) gives

<!-- formula-not-decoded -->

Finally, summing over all ˜ z we deduce that

<!-- formula-not-decoded -->

Now recall that H ˜ z, int ,A is the pushforward of a measure P ˜ z, int supported in B η ′ (˜ z ) , where η ′ = η + ε ′ ≤ η + σ and H ˜ z, ext ,A is the pushforward of a measure P ˜ z, ext supported in B c θ (˜ z ) , where θ = c ( η + σ ) ≥ c 2 ( η ′ + σ ) . Therefore, Lemma C.5 (with c replaced by c/ 2 ) gives that

<!-- formula-not-decoded -->

where D ˜ z, ext = { x -˜ z : x ∈ supp( P ˜ z, ext ) } and D ˜ z, int = { x -˜ z : x ∈ supp( P ˜ z, int ) } . It follows immediately from (C.22) that

<!-- formula-not-decoded -->

Moreover, ˜ z ∈ S and therefore

<!-- formula-not-decoded -->

where D 2 is as in (3.3). Using this, the previous bound and (C.23), we deduce that

<!-- formula-not-decoded -->

To complete the proof, now substitute this into (C.19) and (C.20), to obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The result now follows after recalling (iv), i.e., | S | ≤ e k , and the fact that δ ′ ≤ δ ≤ 1 / 4 .

## C.4 Proof of Theorem 1.1

Finally, we now show how Theorem 3.1 implies the simplified result, Theorem 1.1.

Proof of Theorem 1.1. Let p = P [ ∥ x ∗ -ˆ x ∥ ≥ (8 d 2 +2)( η + σ ) ] . We use Theorem 3.1 with ε replaced by ε/ (2 mθ ) . Let c = 8 d 2 , c ′ = 2 , t = θ and ε ′ = ε/ (2 δ 1 /p mθ ) . Then Theorem 3.1 gives that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

D 2 = D = supp( P ) -supp( P ) and k = ⌈ log Cov η,δ ( P ) ⌉ . Now since ∥ Ax ∥ ≤ ∥ A ∥∥ x ∥ ≤ θ ∥ x ∥ , ∀ x ∈ R n , we make take C abs ( ε ′ , θε ′ ; A , D 1 ) = 0 . Moreover, by Lemma B.1, we have

<!-- formula-not-decoded -->

where we used the facts that m ≥ 1 and σ ≥ ε/δ 1 /p . Hence

<!-- formula-not-decoded -->

Finally, Lemma B.1 implies that

<!-- formula-not-decoded -->

where in the final step we used the fact that d ≥ 2 . This gives the result.