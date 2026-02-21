## Privacy amplification by random allocation

## Vitaly Feldman

Apple vitaly.edu@gmail.com

## Moshe Shenfeld ∗

The Hebrew university of Jerusalem moshe.shenfeld@mail.huji.ac.il

## Abstract

We consider the privacy amplification properties of a sampling scheme in which a user's data is used in k steps chosen randomly and uniformly from a sequence (or set) of t steps. This sampling scheme has been recently applied in the context of differentially private optimization [Chua et al., 2024a, Choquette-Choo et al., 2025] and is also motivated by communication-efficient high-dimensional private aggregation [Asi et al., 2025]. Existing analyses of this scheme either rely on privacy amplification by shuffling which leads to overly conservative bounds or require Monte Carlo simulations that are computationally prohibitive in most practical scenarios.

We give the first theoretical guarantees and numerical estimation algorithms for this sampling scheme. In particular, we demonstrate that the privacy guarantees of random k -out-oft allocation can be upper bounded by the privacy guarantees of the well-studied independent (or Poisson) subsampling in which each step uses the user's data with probability (1 + o (1)) k/t . Further, we provide two additional analysis techniques that lead to numerical improvements in several parameter regimes. Altogether, our bounds give efficiently-computable and nearly tight numerical results for random allocation applied to Gaussian noise addition.

## 1 Introduction

One of the central tools in the analysis of differentially private algorithms are so-called privacy amplification guarantees, where amplification results from sampling of the inputs. In these results one starts with a differentially private algorithms (or a sequence of such algorithms) and a randomized selection (or sampling) to determine which of the n elements in a dataset to run each of the t algorithms on. Importantly, the random bits of the sampling scheme and the selected data elements are not revealed. For a variety of sampling schemes this additional uncertainty is known to lead to improved privacy guarantees of the resulting algorithm, that is, privacy amplification.

In the simpler, single step case a DP algorithm is run on a randomly chosen subset of the dataset. As first shown by Kasiviswanathan et al. [2011], if each element of the dataset is included in the subset with probability λ (independently of other elements) then the privacy of the resulting algorithm is better (roughly) by a factor λ . This basic result has found numerous applications, most notably in the analysis of the differentially private stochastic gradient descent (DP-SGD) algorithm [Bassily et al., 2014]. In DP-SGD gradients are computed on randomly chosen batches of data points and then privatized through clipping and Gaussian noise addition. Privacy analysis of this algorithm is based on the so called Poisson sampling: elements in each batch and across batches are chosen randomly and independently of each other. The absence of dependence implies that the algorithm can be analyzed relatively easily as a direct composition of single step amplification results. The downside of this simplicity is that such sampling is less efficient and harder to implement within the standard ML pipelines. As a result, in practice some form of shuffling is used to define the batches in

∗ Work partially done while author was an intern at Apple

DP-SGD leading to a well-recognized discrepancy between the implementations of DP-SGD and their analysis [Chua et al., 2024b,c, Annamalai et al., 2024].

Figure 1: Upper bounds on privacy parameter ε as a function of the noise parameter σ for various schemes and the local algorithm (no amplification), all using the Gaussian mechanism with fixed parameters δ = 10 -10 , t = 10 6 . In the Poisson scheme λ = 1 /t . The "flat" part of the RDP based calculation is due to computational limitations, which was computed for the range α ∈ [2 , 60] .

<!-- image -->

Motivated by the shuffle model of federated data analysis [Bittau et al., 2017], Cheu et al. [2019], Erlingsson et al. [2019] have studied the privacy amplification of the shuffling scheme. In this scheme, the n elements are randomly and uniformly permuted and the i -th element in the permuted order is used in the i -th step of the algorithm. This sampling scheme can be used to analyze the implementations of DP-SGD used in practice [Erlingsson et al., 2019, Feldman et al., 2021]. However, the analysis of this sampling scheme is more involved and nearly tight results are known only for relatively simple pure DP ( δ = 0 ) algorithms [Feldman et al., 2021, 2023, Girgis et al., 2021]. In particular, applying these results to Gaussian noise addition requires using ( ε, δ ) -guarantees of the Gaussian noise. This leads to an additional √ ln(1 /δ ) factor in the asymptotic analysis and significantly worse numerical results (see Fig. 1 for comparison and discussion in Section 4.2).

Note that shuffling differs from Poisson subsampling in that participation of elements is dependent both in each step (or batch) and across the steps. If the participation of elements in each step is dependent (by fixing the total number of participating elements) but the steps are independent then the sampling scheme can be tightly analyzed as a direct composition of fixed subset size sampling steps (e.g., using bound in Balle et al. [2018], Zhu et al. [2022]). However, a more problematic aspect of Poisson sampling is the stochasticity in the number of times each element is used in all steps. For example, using Poisson sampling with sampling rate 1 /t over t batches will result in a roughly 1 /e probability of not using the sample which implies dropping approximately 37% of the data, and the additional sampling randomness may increase the resulting variance as demonstrated in Appendix H. In a distributed setting it is also often necessary to limit the maximum number of times a user participates in the analysis due to time or communication constraints on the protocol [Chen et al., 2024, Asi et al., 2025]. Poisson sampling does not allow to fully exploit the available limit which hurts the utility.

Motivated by the privacy analysis of DP-SGD and the problem of communication-efficient highdimensional private aggregation with two servers [Asi et al., 2025], we analyze sampling schemes where each element participates in exactly k randomly chosen steps out of the total t , independently of other elements. We refer to this sampling as k -out-oft random allocation . For k = 1 , this scheme is a special case of the random check-in model of defining batches for DP-SGD in [Balle et al., 2020]. Their analysis of this variant relies on the amplification properties of shuffling and thus does not lead

to better privacy guarantees than those that are known for shuffling. Very recently, Chua et al. [2024a] have studied such sampling (referring to it as balls-and-bins sampling ) in the context of training neural networks via DP-SGD. Their main results show that from the point of view of utility (namely, accuracy of the final model) random allocation with k = 1 is essentially identical to shuffling and is noticeably better than Poisson sampling. Concurrently, Choquette-Choo et al. [2025] considered the same sampling scheme for the matrix mechanism in the context of DP-FTRL. The privacy analysis in these two works reduces the problem to analyzing the divergence of a specific pair of distributions on R t . They then used Monte Carlo simulations to estimate the privacy parameters of this pair. Their numerical results suggest that privacy guarantees of 1 -out-oft random allocation are similar to those of the Poisson sampling with rate of 1 /t . While very encouraging, such simulations have several limitations which we discuss in Appendix G.1, most notably, achieving high-confidence estimates for small δ and supporting composition appear to be computationally impractical. This approach also does not lead to provable privacy guarantees and does not lend itself to asymptotic analysis (such as the scaling of the privacy guarantees with t ).

## 1.1 Our contribution

We provide three new analyses for the random allocation setting that result in provable guarantees that nearly match or exceed those of the Poisson subsampling at rate k/t . The analyses rely on different techniques and lead to incomparable numerical results. We describe the specific results below and illustrate the resulting bounds in Fig. 1.

In our main result we show that the privacy of random allocation is upper bounded by that of the Poisson scheme with sampling probability ≈ k/t up to lower order terms which are asymptotically vanishing in t/k . Specifically, we upper bound it by the k -wise composition of Poisson subsampling with rate (1+ γ ) k/t applied to a dominating pair of distributions for each step of the original algorithm

<!-- formula-not-decoded -->

ε 0 , δ 0 are the privacy parameters of the original algorithm. The formal statement of this result that includes all the constants can be found in Thm. 4.1. Additionally, we show in Thm. 4.1 this lower order term can be recursively bounded using ( ε ′ , δ ′ ) parameters of the same algorithm for some ε ′ &gt; ε . This leads to significant numerical improvements in our results.

Our analysis relies on several simplification steps. Using a dominating pair of distributions for the steps of the original algorithm, we first derive an explicit dominating pair of distributions for random allocation (extending a similar result for Gaussian noise in [Chua et al., 2024a, Choquette-Choo et al., 2025]). Equivalently, we reduce the allocation for general multi-step adaptive algorithms to the analysis of random allocation for a single (non-adaptive) randomizer on two inputs. We also analyze only the case of k = 1 and then use a reduction from general k to k = 1 . Finally, our analysis of the non-adaptive randomizer for k = 1 relies on a decomposition of the allocation scheme into a sequence of posterior sampling steps for which we then prove a high-probability bound on subsampling probability in each step.

We note that, in general, the privacy of the composition of subsampling of the dominating pair of distributions can be worse than the privacy of the sampling scheme of a concrete algorithm, even if this pair tightly dominates it. This is true for both Poisson and random allocation schemes. However, all existing analyses of the Poisson sampling are effectively based on composition of subsampling for a dominating pair of distributions. Moreover, if the algorithm has a pair of neighboring datasets inducing this dominating pair, then our upper bound can be stated directly in terms of the Poisson subsampling scheme with respect to this pair. Such dominating input exists for many standard algorithms including those based on Gaussian and Laplace noise addition.

While our result shows asymptotic equivalence of allocation and Poisson subsampling, it may lead to suboptimal bounds for small values of t/k and large ε 0 . We address this using two additional techniques which are also useful as starting points for the recursive version of our main result.

We first show that ε of random allocation with k = 1 is at most a constant ( ≈ 1 . 6 ) factor times larger than ε of the Poisson sampling with rate 1 /t for the same δ (see Theorem 4.3). This upper bound does not asymptotically approach Poisson subsampling but applies in all parameter regimes. To prove this upper bound we observe that Poisson subsampling is essentially a mixture of random allocation schemes with various values of k . We then prove a monotonicity property of random allocations

showing that increasing k leads to worse privacy. Combining these results with the advanced joint convexity property Balle et al. [2018] gives the upper bound.

Finally, we give a direct analysis of the divergence for the dominating pair of distributions. Due to the asymmetric nature of the add/remove privacy our bounds require different techniques for each of the directions. In the remove direction we derive a closed form expression for the Rényi DP [Mironov, 2017] of the dominating pair of distributions for allocation in terms of the RDP parameters of the original algorithm (Theorem 4.4). This method has two important advantages. First, it gives a precise bound on the RDP parameters of integer order (as opposed to just an upper bound). Secondly, it is particularly easy to use in the typical setting where composition is used in addition to a sampling scheme (for example, when k &gt; 1 or in multi-epoch DP-SGD). The primary disadvantage of this technique is that the conversion from RDP bounds to the regular ( ε, δ ) bounds is known to be somewhat lossy (typically within 10 -20% range in multi-epoch settings). The same loss is also incurred when Poisson sampling is analyzed via RDP (referred to as moment accounting [Abadi et al., 2016]). Two more limitations of this technique result from the restriction to the range α ≥ 2 , and the computational complexity when α is in the high tens.

For the add direction we give an approximate upper bound in terms of the usual composition of a different, explicitly defined randomizer over the same domain. While this bound is approximate, the divergence for the add direction is typically significantly lower than the one for the remove direction and therefore, in our evaluations, this approximation has either minor or no effect on the maximum. Overall, in our evaluations of this method for Gaussian distribution in most regimes the resulting bounds are almost indistinguishable from those obtained via RDP for Poisson distribution (see Fig. 6 for examples). In fact, in some regimes it is better than Poisson sampling (Figure 5).

Numerical evaluation: In Section 5 we provide numerical evaluation and comparisons of our bounds to those for Poisson sampling as well as other relevant bounds. 2 Our evaluations across many parameter regimes give bounds on the privacy of random allocation that are very close, typically within 10% of those for the Poisson subsampling with rate k/t . This means that random allocation can be used to replace Poisson subsampling with only a minor loss in privacy. At the same time, in many cases, the use of random allocation can improve utility. In the context of training neural networks via DP-SGD this was shown in [Chua et al., 2024a]. Application of our bounds also lead to improvement over Poisson subsampling in [Asi et al., 2025]. We demonstrate that even disregarding some practical disadvantages of Poisson subsampling, random allocation has a better privacy-utility trade-off for mean estimation in low-dimensional regime. This improvement stems from the fact that random allocation computes the sum exactly whereas Poisson subsampling introduces additional variance. At the same time in the high-dimensional regime noise due to privacy dominates the final error and thus the trade-off boils down to the difference in the privacy bounds.

## 1.2 Related work

Our work builds heavily on tools and ideas developed for analysis of privacy amplification by subsampling, composition and shuffling. We have covered the work directly related to ours earlier and will describe some of the tools and their origins in the preliminaries. A more detailed technical and historical overview of subsampling and composition for DP can be found in the survey by Steinke [2022]. The shuffle model was first proposed by Bittau et al. [2017]. The formal analysis of the privacy guarantees in this model was initiated in [Erlingsson et al., 2019, Cheu et al., 2019]. The sequential shuffling scheme that we discuss here was defined by Erlingsson et al. [2019] who proved the first general privacy amplification results for this scheme albeit only for pure DP algorithms. Improved analyses and extensions to approximate DP were given in [Balle et al., 2019, 2020, Feldman et al., 2021, 2023, Girgis et al., 2021, Koskela et al., 2022].

DP-SGD was first defined and theoretically analyzed in the convex setting by Bassily et al. [2014]. Its use in machine learning was spearheaded by the landmark work of Abadi et al. [2016] who significantly improved the privacy analysis via the moments accounting technique and demonstrated the practical utility of the approach. In addition to a wide range of practical applications, this work has motivated the development of more advanced techniques for analysis of sampling and composition. At the same time most analyses used in practice still assume Poisson subsampling when selecting batches whereas some type of shuffling is used in implementation. It was recently shown that it

2 Python implementation of all methods is available in a GitHub project and in a package.

results in an actual difference between the reported and true privacy level in some regimes [Chua et al., 2024b,c, Annamalai et al., 2024].

In a concurrent and independent work Dong et al. [2025] considered the same sampling method (referring to it as Balanced Iteration Subsampling ). Their results are closest in spirit to our direct bounds. Specifically, they provide RDP-based bounds for the same dominating pair of distributions in the Gaussian case for both add and remove directions. Their bound for general k is incomparable to ours as it is based on a potentially loose upper bound for divergences of order α &gt; 2 , while using an exact extension of their approximation to k &gt; 1 . In contrast, our RDP-based bound uses a reduction from general k to k = 1 that is potentially loose but our computation for the k = 1 case is exact (for the remove direction which is typically larger than the add direction). In our numerical comparisons, the bounds in Dong et al. [2025] are comparable or worse than our direct bounds and are often significantly worse than the bounds from our main result. We discuss these differences in more detail and provide numerical comparison in Appendix G.2.

## 2 Preliminaries

We denote the domain of elements by X and the set of possible outputs by Y . Given a dataset s ∈ X ∗ and an output y ∈ Y , we denote by P M ( y | s ) := P Y ∼ M ( s ) ( Y = y ) the probability of observing the output y as the output of some randomized algorithm M which was given dataset s as input.

## 2.1 Sampling schemes

In this work, we consider t -step algorithms defined using an algorithm M that takes some subset of the dataset and a sequence of previous outputs as input. Formally, let Y &lt;t = ⋃ i&lt;t Y i . M takes a dataset in X ∗ and a view v ∈ Y &lt;t as its inputs and outputs a value in Y . A t -step algorithm defined by M first uses some scheme to define t subsets s 1 , . . . , s t ⊆ s , then sequentially computes y i = M ( s i , v i -1 ) , where v i := ( y 1 , . . . , y i ) are the intermediate views consisting of the outputs produced so far, and v 0 = ∅ . Such algorithms include DP-SGD, where each step consists of a call to the Gaussian mechanism (A.2), with gradient vectors adaptively defined as a function of previous outputs.

The assignment of the elements in s to the various subsets can be done in a deterministic manner (e.g., s 1 = . . . = s t = s ) , or randomly using a sampling scheme . We consider the following three sampling schemes.

1. Poisson scheme parametrized by sampling probability λ ∈ [0 , 1] , where each element is added to each subset s i with probability λ independent of the other elements and other subsets,
2. Shuffling scheme which uniformly samples a permutation π over [ n ] where n is the sample size, and sets s i = { x π ( i ) } (in this case, the sample size and number of steps must match).
3. Random allocation scheme parameterized by a number of selected steps k ∈ [ t ] , which uniformly samples k indices i = ( i 1 , . . . , i k ) ⊆ [ t ] for each element and adds them to the corresponding subsets s i 1 , . . . , s i k .

For a t -step algorithm defined by M , we denote by P t,λ ( M ) : X ∗ →Y t an algorithm using M with the Poisson sampling scheme, S n ( M ) : X n →Y n for the shuffling scheme, and A t,k ( M ) : X ∗ → Y t when M is used with the random allocation scheme. When k = 1 we omit it from the notation for clarity.

## 2.2 Privacy notions

We consider the standard add/remove adjacency notion of privacy in which datasets s , s ′ ∈ X ∗ are neighboring if s can be obtained from s ′ via adding or removing one of the elements. To appropriately define sampling schemes that operate over a fixed number of elements we augment the domain with a 'null' element ⊥ , that is, we define X ′ := X ∪ {⊥} . When a t -step algorithm assigns ⊥ to M we treat it as an empty set, that is, for any s ∈ X ∗ , v ∈ Y ∗ we have M ( s , v ) = M (( s , ⊥ ) , v ) . We say

that two datasets s , s ′ ∈ X n are neighbors and denote it by s ≃ s ′ , if one of the two can be created by replacing a single element in the other dataset by ⊥ .

We rely on the hockey-stick divergence to quantify the privacy loss.

Definition 2.1. Given κ ≥ 0 and two distributions P, Q over some domain Ω , the hockey-stick divergence between them is defined to be

<!-- formula-not-decoded -->

where ℓ ( ω ; P, Q ) := ln ( P ( ω ) Q ( ω ) ) ; P ( ω ) Q ( ω ) is the ratio of the probabilities for countable domain or the Radon Nikodym derivative in the continuous case, and [ x ] + := max { 0 , x } . When P, Q are distributions induced by neighboring datasets s ≃ s ′ and an algorithm M , we refer to the log probability ratio as the privacy loss random variable and denote it by ℓ M ( y ; s , s ′ ) . We omit M from the notation when it is clear from the context.

Definition 2.2 ([Balle et al., 2018]) . Given an algorithm M : X ∗ → Y , the privacy profile δ M : R → [0 , 1] is defined to be the maximal hockey-stick divergence between the distributions induced by any neighboring datasets and past view. Formally, δ M ( ε ) := sup s ≃ s ′ ∈X ∗ , v ∈Y ∗ ( H e ε ( M ( s , v ) ∥ M ( s ′ , v ))) .

Since the hockey-stick divergence is asymmetric in the general case, we use ⃗ δ M and ∼ → to denote the remove direction where ⊥ ∈ s ′ and ⃗ δ M , ∼ ← to denote the add direction when ⊥ ∈ s . Notice that δ M ( ε ) = max { ⃗ δ M ( ε ) , ⃗ δ M ( ε ) } .

Another useful divergence notion is the Rényi divergence .

Definition 2.3. Given α &gt; 1 and two distributions P, Q over some domain Ω , the Rényi divergence between them is defined to be R α ( P ∥ Q ) := 1 α -1 ln ( E ω ∼ Q [ e α · ℓ ( ω ; P,Q ) ] ) .

We can now formally define our privacy notions.

Definition 2.4 ([Dwork et al., 2006]) . Given ε &gt; 0 ; δ ∈ [0 , 1] , an algorithm M will be called ( ε, δ ) -differentially private (DP) , if δ M ( ε ) ≤ δ .

Definition 2.5 ([Mironov, 2017]) . Given α ≥ 1 ; ρ &gt; 0 , an algorithm M will be called ( α, ρ ) -Rényi differentially private (RDP) , if sup s ≃ s ′ ∈X ∗ , v ∈Y ∗ ( R α ( M ( s , v ) ∥ M ( s ′ , v ))) ≤ ρ .

One of the most common algorithms is the Gaussian mechanism N σ , which simply reports the sum of (some function of) the elements in the dataset with the addition of Gaussian noise. One of its main advantages is that we have closed form expressions of its privacy (Lemma A.2).

## 2.3 Dominating pair of distributions

A key concept for characterizing the privacy guarantees of an algorithm is that of a dominating pair of distributions.

Definition 2.6 ([Zhu et al., 2022]) . Given distributions P, Q over some domain Ω , and P ′ , Q ′ over Ω ′ , we say ( P ′ , Q ′ ) dominate ( P, Q ) if for all κ ≥ 0 we have H κ ( P ∥ Q ) ≤ H κ ( P ′ ∥ Q ′ ) . If δ M ( ε ) ≤ H e ε ( P ∥ Q ) for all ε ∈ R , we say ( P, Q ) is a dominating pair of distributions for M . If the inequality can be replaced by an equality for all ε , we say it is a tightly dominating pair . If there exist some s ≃ s ′ ∈ X ∗ such that P = M ( s ) , Q = M ( s ′ ) we say ( s , s ′ ) are the dominating pair of datasets for M . By definition, a dominating pair of input datasets is tightly dominating.

We use the notion of dominating pair to define a dominating randomizer, which captures the privacy guarantees of the algorithm independently of its algorithmic adaptive properties.

Definition 2.7. Given a t -step algorithm defined by M , we say that R : {⊥ , ∗} → Y is a dominating randomizer for M and set R ( ∗ ) = P and R ( ⊥ ) = Q , where ( P, Q ) is the tightly dominating pair of M ( · , · ) w.r.t. ∼ → over all indexes i ∈ [ t ] and input partial views v i -1 . 3 .

3 Such a pair always exists [Zhu et al., 2022, Proposition 8]

The definition of the Poisson and random allocation schemes naturally extends to the case where the internal algorithm is a randomizer. In this case P t,λ ( R ) : {∗ , ⊥} → Y t and A t,λ ( R ) : {∗ , ⊥} → Y t .

## 3 General reduction

We first prove two general claims which reduce the bound on arbitrary algorithms, datasets, and number of allocations, to the case of a single allocation ( k = 1 ) of a simple non-adaptive randomizer receiving a single element. Missing proofs can be found in Appendix B

From the definition of the dominating randomizer, for any ε ∈ R we have δ M ( ε ) ≤ δ R ( ε ) . We now prove that this is also the case for allocation scheme, that is δ A t,k ( M ) ( ε ) ≤ δ A t,k ( R ) ( ε ) , and that the supremum over neighboring datasets for A t,k ( R ) is achieved by the pair of datasets containing a single element, that is s = {∗} , s ′ = {⊥} . This results from the fact that random allocation can be viewed as a two steps process, where first all elements but one are allocated, then the remaining one is allocated and the algorithm is ran for t steps. From the convexity of the hockey-stick divergence we can upper bound the privacy profile of the random allocation scheme by the worst case allocation of all elements but the removed one. From Lemma A.3, each intermediate call to the mechanism is a post process of the randomizer, which can be used to recursively define a randomized mapping from the random allocation over the randomizer to the allocation over the mechanism. Using the same lemma, this mapping implies that δ A t,k ( M ) ( ε ) ≤ δ A t,k ( R ) ( ε ) .

Theorem 3.1. Given t ∈ N ; k ∈ [ t ] and a t -step algorithm defined by M dominated by a randomizer R , we have δ A t,k ( M ) ( ε ) ≤ δ A t,k ( R ) ( ε ) .

A special case of this result for Gaussian noise addition and k = 1 was given by Chua et al. [2024a, Theorem 1], and in the context of the matrix mechanism by Choquette-Choo et al. [2025, Lemma 3.2]. The same bound for the Poisson scheme is a direct result from the combination of Claim C.9 and Zhu et al. [2022, Theorem 11].

Next we show how to translate any bound on the privacy profile of the random allocation with k = 1 to the case of k &gt; 1 by decomposing it to k calls to a 1 out of t/k steps allocation process.

Lemma 3.2. For any k ∈ N , ε &gt; 0 we have δ A t,k ( R ) ( ε ) ≤ δ ⊗ k A ⌊ t/k ⌋ ( R ) ( ε ) , where ⊗ k denotes the composition of k runs of the algorithm or scheme which in our case is A ⌊ t/k ⌋ ( R ) .

Combining these two results, the privacy profile of the random allocation scheme is bounded by the (composition of the) hockey-stick divergence between A t ( R ; ∗ ) and A t ( R ; ⊥ ) = R ⊗ t ( ⊥ ) in both directions, which we bound in three different ways in the following section.

## 4 Privacy bounds

## 4.1 Truncated Poisson bound

Roughly speaking, our main theorem states that random allocation is asymptotically identical to the Poisson scheme with sampling probability ≈ k/t up to lower order terms. Formal proofs and missing details of this section can be found in Appendix C.

Theorem 4.1. Given ε 0 &gt; 0 ; δ 0 ∈ [0 , 1] and a ( ε 0 , δ 0 ) -DP randomizer R , for any ε, δ &gt; 0 we have δ A t ( R ) ( ε ) ≤ δ P t,η ( R ) ( ε )+ tδ 0 + δ , where η := 1 t (1 -γ ) and γ := min { cosh( ε 0 ) · √ 2 t ln ( 1 δ ) , 1 -1 t } .

Furthermore, for any ε, ε ′ &gt; 0 and randomizer R we have ⃗ δ A t ( R ) ( ε ) ≤ ⃗ δ P t,η ( R ) ( ε ) + τ · ⃗ δ A t ( R ) ( ε ′ ) and ⃗ δ A t ( R ) ( ε ) ≤ ⃗ δ P t,η ( R ) ( ε ) + τe 2 ε ′ · ⃗ δ A t ( R ) ( ε ′ ) , where η := e 2 ε ′ t and τ := 1 e ε ′ ( e ε ′ -1) .

Since η corresponds to a sampling probability of 1 t up to a lower order term in t , this implies that the privacy of random allocation scheme is asymptotically upper-bounded by the Poisson scheme. While this holds for sufficiently large value of t , in many practical parameter regimes the second part of the theorem provides tighter bounds.

While the recursive expression might seem to lead to a vacuous loop, it is in fact a useful tool. Notice that ⃗ δ A t ( R ) ( ε ′ ) / ⃗ δ A t ( R ) ( ε ) quickly diminishes as ε ′ /ε grows, so it suffices to set ε ′ = Cε

for some constant 1 ≪ C for the second term to become negligible. Both parts of this theorem follow from Lemma C.1 which bounds the privacy profile of the random allocation scheme by that of the corresponding Poisson scheme with sampling probability η , with an additional term roughly corresponding to a tail bound on the privacy loss of the allocation scheme.

## 4.2 Asymptotic analysis

Combining Theorem 4.1 with Lemma 3.2 and applying it to the Gaussian mechanism results in the next corollary.

<!-- formula-not-decoded -->

Using this Corollary we can derive asymptotic bounds on the privacy guarantees of the Gaussian mechanism amplified by random allocation. Since the Gaussian mechanism is dominated by the one-dimensional Gaussian randomizer (Claim D.3) where R ( ∗ ) = N (1 , σ 2 ) and R ( ⊥ ) = N (0 , σ 2 ) , this corollary implies that for sufficiently large σ , the random allocation scheme with the Gaussian

<!-- formula-not-decoded -->

constant C (Lemma D.2). We note that the dependence of ε on σ ; δ ; k ; and t matches that of the Poisson scheme for λ = k/t (Lemma D.1) up to an additional logarithmic dependence on t (Poisson scales with ln(1 /δ ) ), unlike the shuffle scheme which acquires an additional √ ln(1 /δ ) by converting approximate the DP randomizer to pure DP first, resulting in the bound ε ≥ C ′ · k · ln(1 /δ ) σ √ t [Feldman et al., 2021]. A detailed comparison can be found in Appendix D.

The recursive bound (second part of Theorem 4.1) provides similar asymptotic guarantees for arbitrary mechanisms, when ε 0 ≤ 1 for the local mechanism M . In this case, the privacy parameter of its corresponding Poisson scheme ε P is approximately linear in the sampling probability. Setting the sampling probability to 1 /t and combining amplification by subsampling with advanced composition

<!-- formula-not-decoded -->

While Theorem 4.1 provides a full asymptotic characterization of the random allocation scheme, the bounds it induces could be suboptimal for small t or large ε 0 . In the following section we provide several bounds that hold in all parameter regimes. We also use these to 'bootstrap' the recursive bound.

## 4.3 Poisson Decomposition

<!-- formula-not-decoded -->

We remark that while this theorem provides separate bounds for the add and remove adjacency notions 4 , numerical analysis seems to indicate that the bound on the remove direction is always larger than the one for the add direction.

Setting λ := 1 /t yields ⃗ γ ≈ e/ ( e -1) ≈ 1 . 6 , which bounds the difference between these two sampling methods up to this factor in ε in the ε &lt; 1 regime.

Formal proofs and missing details can be found in Appendix E.

## 4.4 Direct analysis

The previous bounds rely on a reduction to Poisson scheme. In this section we bound the privacy profile of the random allocation scheme directly, which is especially useful in the low privacy regime

4 An earlier version of this work has mistakenly stated that an upper bound for the remove direction applies to both directions.

where the privacy profile of random allocation is lower than that of Poisson. Formal proofs and missing details of this section can be found in Appendix F.

Our main result expresses the RDP of the random allocation scheme in the remove direction in terms of the RDP parameters of the randomizer, and provides an approximate bound in the the add direction 5 . While the privacy bounds induced by RDP are typically looser than those relying on full analysis and composition of the privacy loss distribution (PLD), the gap nearly vanishes as the number of composed calls to the randomizer grows, as depicted in Figure 6.

Theorem 4.4. Given two integers t, α ∈ N , we denote by Π t ( α ) the set of integer partitions of α consisting of ≤ t elements. Given a partition Π ∈ Π t ( α ) , we denote by ( α Π ) = α ! ∏ p ∈ Π p ! , and denote by C (Π) the list of counts of unique values in P (e.g. if α = 9 and Π = [1 , 2 , 3 , 3] then C (Π) = [1 , 1 , 2] ). For any α ≥ 2 and randomizer R we have

<!-- formula-not-decoded -->

For the add direction we use a different bound.

Theorem 4.5. Given γ ∈ [0 , 1] and a randomizer R , we define a new randomizer R γ which given an input x samples y ∝ P R ( y | x ) γ · P R ( y |⊥ ) 1 -γ Z γ , where Z γ is the normalizing factor.

<!-- formula-not-decoded -->

These two theorems follow from Lemmas F.1 and F.3 for P = R ( ∗ ) and Q = R ( ⊥ ) .

As is the case in Theorem 4.3, numerical analysis seems to indicate the bound on the remove direction always dominates the one for the add direction.

Since we have an exact expression for the hockey-stick and Rényi divergences of the Gaussian mechanism, these two theorems immediately imply the following corollary.

Corollary 4.6. Given σ &gt; 0 , and a Gaussian mechanism N σ , we have for any integer α ≥ 2

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Corollary 4.6 gives a simple way to exactly compute integer RDP parameters of random allocation with Gaussian noise in the remove direction. Interestingly, they closely match RDP parameters of the Poisson scheme with rate 1 /t in most regimes (e.g. Fig. 6). In fact, in some (primarily large ε ) parameter regimes the bounds based on RDP of allocation are lower than the PLD-based bounds for Poisson subsampling (Fig. 5). The restriction to integer values has negligible effect, which can be further mitigated using [Wang et al., 2019, Corollary 10], which upper bounds the fractional Rényi divergence by a linear combination of the Rényi divergence of its rounded integer values. We also note that | Π t ( α ) | is sub-exponential in α which leads to performance issues in the very high privacy ( ε ≪ 1 ) regime (Large σ values in Fig 1). Since the typical value of α used for accounting is in the low tens, this quantity can be efficiently computed using several technical improvements which we discuss in Appendix F.1. On the other hand, in the very low privacy regime ( ε ≫ 1 ), the α that leads to the best bound on ε is typically in the range [1 , 2] which cannot be computed exactly using our method. Finally, we remark that while this result is stated only for k = 1 , it can be extended to k &gt; 1 using the same argument as in Lemma 3.2. In fact, RDP based bounds are particularly convenient for subsequent composition which is necessary to obtain bounds for k &gt; 1 or multi-epoch training algorithms.

Figure 2: Bounds on privacy parameter ε as a function of the noise parameter σ for various values of t , all using the Gaussian mechanism with δ = 10 -10 . We compare the minimum over all our methods to the independent results in Dong et al. [2025], lower bound by Chua et al. [2024a], and to the Poisson scheme with λ = 1 /t .

<!-- image -->

## 5 Numerical evaluation

In this section we demonstrate that numerical implementations of our results give the first nearly-tight and provable bounds on privacy amplification of random allocation with Gaussian noise, notably showing (Fig. 1, 2) that they nearly match bounds known for Poisson subsampling. Compared to the Monte Carlo-based technique by Chua et al. [2024a] (G.1), we show in many regimes our results match these bounds up to constants in δ (logarithmic in ε ), and the computational limitation of the MCtechnique in the low δ and high confidence level regime. We additionally compare our results to the RDP-based approximation by Dong et al. [2025] (G.2), and demonstrate the advantage of our tight RDP analysis in the regime where k ≪ t . Their bound is tighter than our direct analysis in the k ≈ t regime, where the effect of amplification is small and ε is prohibitively large.

We demonstrate the utility degradation induced by Poisson subsampling relative to random allocation using the simple setting of estimating the mean of a Bernoulli distribution from a sampled dataset (App. H). We derive theoretical approximations for the mean square error of the two schemes and match them with numerical simulations, that demonstrate random allocation always has lower error for sufficiently large sample size. Together with the results of Chua et al. [2024a], our results imply that random allocation (or balls-and-bins sampling) has the utility benefits of shuffling while having the privacy benefits of Poisson subsampling. This provides a (reasonably) practical way to reconcile a long-standing and concerning discrepancy between the practical implementations of DP-SGD and its commonly-used privacy analyses.

## 6 Discussion

This work provides the first theoretical guarantees and numerical estimation algorithms for the random allocation sampling scheme. Its main analysis shows that its privacy guarantees are asymptotically identical to those of the Poisson scheme. We provide two additional analyses which lead to tighter bounds in some setting (Fig. 1). The resulting combined bound of the random allocation remains close to that of the Poisson scheme in many practical regimes (Fig. 2, 9), and even exceeds it in some. Unlike the Poisson scheme, our bounds are analytical and do not rely on numerical PLD analysis, which results in some remaining slackness. Further, unlike PLD-based bounds, our ( ε, δ ) bounds do not lend themselves for tight privacy accounting of composition. Both of these limitations are addressed in our subsequent work [Feldman and Shenfeld, 2025] where we show that PLD of random allocation can be approximated efficiently, leading to tighter and more general numerical bounds.

5 An earlier version of this work has mistakenly stated that an upper bound for the remove direction applies to both directions.

## Acknowledgments and Disclosure of Funding

We are grateful to Kunal Talwar for proposing the problem that we investigate here as well as suggesting the decomposition-based approach for the analysis of this problem (established in Theorem 4.3). We thank Thomas Steinke and Christian Lebeda for pointing out that an earlier version of our work mistakenly did not analyze the add direction of our direct bound. We also thank Hilal Asi, Hannah Keller, Guy Rothblum and Katrina Ligett for thoughtful comments and motivating discussions of these results and their application in [Asi et al., 2025]. Shenfeld's work was supported by the Apple Scholars in AI/ML PhD Fellowship.

## References

- Martin Abadi, Andy Chu, Ian Goodfellow, H Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC conference on computer and communications security , pages 308-318, 2016.
- Meenatchi Sundaram Muthu Selva Annamalai, Borja Balle, Emiliano De Cristofaro, and Jamie Hayes. To shuffle or not to shuffle: Auditing dp-sgd with shuffling. arXiv preprint arXiv:2411.10614 , 2024.
- Hilal Asi, Vitaly Feldman, Hannah Keller, Guy N. Rothblum, and Kunal Talwar. PREAMBLE: Private and efficient aggregation of block sparse vectors and applications. Cryptology ePrint Archive, Paper 2025/490, 2025.
- Borja Balle and Yu-Xiang Wang. Improving the gaussian mechanism for differential privacy: Analytical calibration and optimal denoising. In International Conference on Machine Learning , pages 394-403. PMLR, 2018.
- Borja Balle, Gilles Barthe, and Marco Gaboardi. Privacy amplification by subsampling: Tight analyses via couplings and divergences. Advances in neural information processing systems , 31, 2018.
- Borja Balle, James Bell, Adrià Gascón, and Kobbi Nissim. The privacy blanket of the shuffle model. In Advances in Cryptology-CRYPTO 2019: 39th Annual International Cryptology Conference, Santa Barbara, CA, USA, August 18-22, 2019, Proceedings, Part II 39 , pages 638-667. Springer, 2019.
- Borja Balle, Peter Kairouz, Brendan McMahan, Om Thakkar, and Abhradeep Guha Thakurta. Privacy amplification via random check-ins. Advances in Neural Information Processing Systems , 33: 4623-4634, 2020.
- Raef Bassily, Adam Smith, and Abhradeep Thakurta. Private empirical risk minimization: Efficient algorithms and tight error bounds. In 2014 IEEE 55th annual symposium on foundations of computer science , pages 464-473. IEEE, 2014.
- Andrea Bittau, Úlfar Erlingsson, Petros Maniatis, Ilya Mironov, Ananth Raghunathan, David Lie, Mitch Rudominer, Ushasree Kode, Julien Tinnes, and Bernhard Seefeld. Prochlo: Strong privacy for analytics in the crowd. In Proceedings of the 26th symposium on operating systems principles , pages 441-459, 2017.
- Clément L Canonne, Gautam Kamath, and Thomas Steinke. The discrete gaussian for differential privacy. Advances in Neural Information Processing Systems , 33:15676-15688, 2020.
- Wei-Ning Chen, Dan Song, Ayfer Ozgur, and Peter Kairouz. Privacy amplification via compression: Achieving the optimal privacy-accuracy-communication trade-off in distributed mean estimation. Advances in Neural Information Processing Systems , 36, 2024.
- Albert Cheu, Adam Smith, Jonathan Ullman, David Zeber, and Maxim Zhilyaev. Distributed differential privacy via shuffling. In Advances in Cryptology-EUROCRYPT 2019: 38th Annual International Conference on the Theory and Applications of Cryptographic Techniques, Darmstadt, Germany, May 19-23, 2019, Proceedings, Part I 38 , pages 375-403. Springer, 2019.

- Christopher Choquette-Choo, Arun Ganesh, Saminul Haque, Thomas Steinke, and Abhradeep Guha Thakurta. Near-exact privacy amplification for matrix mechanisms. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, editors, International Conference on Representation Learning , volume 2025, pages 98772-98802, 2025.
- Christopher A Choquette-Choo, Arun Ganesh, Thomas Steinke, and Abhradeep Guha Thakurta. Privacy amplification for matrix mechanisms. In The Twelfth International Conference on Learning Representations , 2023.
- Lynn Chua, Badih Ghazi, Charlie Harrison, Pritish Kamath, Ravi Kumar, Ethan Jacob Leeman, Pasin Manurangsi, Amer Sinha, and Chiyuan Zhang. Balls-and-bins sampling for dp-sgd. In The 28th International Conference on Artificial Intelligence and Statistics , 2024a.
- Lynn Chua, Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Amer Sinha, and Chiyuan Zhang. How private are dp-sgd implementations? In Forty-first International Conference on Machine Learning , 2024b.
- Lynn Chua, Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Amer Sinha, and Chiyuan Zhang. Scalable dp-sgd: Shuffling vs. poisson subsampling. Advances in Neural Information Processing Systems , 37:70026-70047, 2024c.
- Andy Dong, Wei-Ning Chen, and Ayfer Ozgur. Leveraging randomness in model and data partitioning for privacy amplification. In Forty-second International Conference on Machine Learning , 2025.
- Cynthia Dwork and Aaron Roth. The algorithmic foundations of differential privacy. Foundations and Trends® in Theoretical Computer Science , 9(3-4):211-407, 2014.
- Cynthia Dwork, Krishnaram Kenthapadi, Frank McSherry, Ilya Mironov, and Moni Naor. Our data, ourselves: Privacy via distributed noise generation. In Advances in Cryptology-EUROCRYPT 2006: 24th Annual International Conference on the Theory and Applications of Cryptographic Techniques, St. Petersburg, Russia, May 28-June 1, 2006. Proceedings 25 , pages 486-503. Springer, 2006.
- Úlfar Erlingsson, Vitaly Feldman, Ilya Mironov, Ananth Raghunathan, Kunal Talwar, and Abhradeep Thakurta. Amplification by shuffling: From local to central differential privacy via anonymity. In Proceedings of the Thirtieth Annual ACM-SIAM Symposium on Discrete Algorithms , pages 2468-2479. SIAM, 2019.
- Vitaly Feldman and Moshe Shenfeld. Efficient computation of the privacy loss distribution for random allocation, 2025. URL https://openreview.net/forum?id=DuFNAlQ8Lw .
- Vitaly Feldman, Audra McMillan, and Kunal Talwar. Hiding among the clones: A simple and nearly optimal analysis of privacy amplification by shuffling. In 2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS) , pages 954-964. IEEE, 2021.
- Vitaly Feldman, Audra McMillan, and Kunal Talwar. Stronger privacy amplification by shuffling for rényi and approximate differential privacy. In Proceedings of the 2023 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA) , pages 4966-4981. SIAM, 2023.
- Antonious M. Girgis, Deepesh Data, Suhas Diggavi, Peter Kairouz, and Ananda Theertha Suresh. Shuffled model of federated learning: Privacy, accuracy and communication trade-offs. IEEE Journal on Selected Areas in Information Theory , 2(1):464-478, 2021.
- Peter Kairouz, Sewoong Oh, and Pramod Viswanath. The composition theorem for differential privacy. In International conference on machine learning , pages 1376-1385. PMLR, 2015.
- Shiva P Kasiviswanathan and Adam Smith. On the'semantics' of differential privacy: A bayesian formulation. Journal of Privacy and Confidentiality , 6(1), 2014.
- Shiva Prasad Kasiviswanathan, Homin K Lee, Kobbi Nissim, Sofya Raskhodnikova, and Adam Smith. What can we learn privately? SIAM Journal on Computing , 40(3):793-826, 2011.
- Antti Koskela, Mikko A Heikkilä, and Antti Honkela. Numerical accounting in the shuffle model of differential privacy. Transactions on Machine Learning Research , 2022.

- Seng Pei Liew and Tsubasa Takahashi. Shuffle gaussian mechanism for differential privacy. arXiv preprint arXiv:2206.09569 , 2022.
- Xin Lyu. Composition theorems for interactive differential privacy. Advances in Neural Information Processing Systems , 35:9700-9712, 2022.
- Ilya Mironov. Rényi differential privacy. In 2017 IEEE 30th computer security foundations symposium (CSF) , pages 263-275. IEEE, 2017.
- Mehta Neelesh B., Wu Jingxian, Molisch Andreas F., and Zhang Jin. Approximating a sum of random variables with a lognormal. Transactions on Wireless Communications , 6(7):2690-2699, 2007.
- Thomas Steinke. Composition of differential privacy &amp; privacy amplification by subsampling. arXiv preprint arXiv:2210.00597 , 2022.
- Salil Vadhan and Tianhao Wang. Concurrent composition of differential privacy. In Theory of Cryptography: 19th International Conference, TCC 2021, Raleigh, NC, USA, November 8-11, 2021, Proceedings, Part II 19 , pages 582-604. Springer, 2021.
- Salil Vadhan and Wanrong Zhang. Concurrent composition theorems for differential privacy. In Proceedings of the 55th Annual ACM Symposium on Theory of Computing , pages 507-519, 2023.
- Yu-Xiang Wang, Borja Balle, and Shiva Prasad Kasiviswanathan. Subsampled rényi differential privacy and analytical moments accountant. In The 22nd International Conference on Artificial Intelligence and Statistics , pages 1226-1235. PMLR, 2019.
- Yuqing Zhu, Jinshuo Dong, and Yu-Xiang Wang. Optimal accounting of differential privacy via characteristic function. In International Conference on Artificial Intelligence and Statistics , pages 4782-4817. PMLR, 2022.

## A Missing definitions and claims

Since Rényi divergence is effectively a bound on the moment generating function, it can be used to bound the hockey-stick divergence which is effectively a tail bound.

Lemma A.1 (Rényi bounds Hockey-stick, Prop. 12 in Canonne et al. [2020]) . Given two distributions P, Q , if R α ( P ∥ Q ) ≤ ρ then H e ε ( P ∥ Q ) ≤ 1 α -1 e ( α -1)( ρ -ε ) ( 1 -1 α ) α .

.

Lemma A.2 (Gaussian mechanism DP guarantees, [Balle and Wang, 2018, Mironov, 2017]) . Given d ∈ N ; σ &gt; 0 , let X = Y := R d . The Gaussian mechanism N σ is defined as N σ ( s ) := N ( ∑ x ∈ s x, σ 2 I d ) .

If the domain of N σ is the unit ball in R d , we have δ N σ ( ε ) = Φ ( 1 2 σ -εσ ) -e ε Φ ( -1 2 σ -εσ ) , where Φ is the CDF of the standard Normal distribution, and for any α ≥ 1 N σ is ( α, α/ (2 σ 2 ) -RDP.

An important property of domination is its equivalence to existence of postprocessing.

Lemma A.3 (Post processing, Thm. II.5 [Kairouz et al., 2015]) . Given distributions P, Q over some domain Ω , and P ′ , Q ′ over Ω ′ , ( P, Q ) dominate ( P ′ , Q ′ ) if and only if there exists a randomized function φ : Ω → Ω ′ such that P ′ = φ ( P ) and Q ′ = φ ( Q ) .

## B Missing proofs from Section 3

Proof of Lemma 3.1. Given n ∈ N , a dataset s ∈ X n -1 and element x ∈ X , the random allocation scheme A t,k ( M ; ( s , x )) can be decomposed into two steps. First all elements in s are allocated, then x is allocated and the outputs are sampled based on the allocations. We denote by a t,k ( n -1) the set of all possible allocations of n -1 elements into k out of t steps, and for any a ∈ a t,k ( n -1) denote by A a t,k ( M ; ( s , x )) the allocation scheme conditioned on the allocation of s according to a . Given the neighboring datasets ( s , x ) and ( s , ⊥ ) we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and similarly,

<!-- formula-not-decoded -->

where the inequality results from the quasi-convexity of the hockey-stick divergence.

Fixing a to be the allocation that maximizes the right-hand side of the inequality, we denote by s 1 , . . . , s t the subsets defined by a . 6 From the definition of the randomizer, for any index j ∈ [ t ] and input prefix view v j -1 we have ( R ( ∗ ) , R ( ⊥ )) dominates ( M (( s j , x ) , v j -1 ) , M (( s j , ⊥ ) , v j -1 )) , so from Lemma A.3, there exists a randomized mapping φ v j -1 such that M (( s j , x ) , v j -1 ) = φ v j -1 ( R ( ∗ )) and M (( s j , ⊥ ) , v j -1 ) = φ v j -1 ( R ( ⊥ )) . 7 Using these mappings, we will recursively define another randomized mapping φ . Given an output view v ∈ Y t , we define v ′ ∼ φ ( v ) by sequentially sampling y ′ j ∼ φ v ′ j -1 ( y j ) for j = 1 , . . . , t , where v ′ j := ( y 1 , . . . , y j ) and v ′ 0 := ∅ .

We will now prove that A a t,k ( M ; ( s , x )) = φ ( A t,k ( R ; ∗ )) and A a t,k ( M ; ( s , ⊥ )) = φ ( A t,k ( R ; ⊥ )) , which by invoking Lemma A.3 again, implies A t,k ( M ) is dominated by A t,k ( R ) and completes the proof.

6 Conceptually, this is equivalent to considering the random allocation scheme over a single element x , with a sequence of mechanisms M s j defined by the various subsets.

7 We note that φ depends on s j and x as well. We omit them from notations for simplicity, since they are fixed at this point of the argument.

From the law of total probability we have,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where i ⊆ [ t ] denots the allocation of the single element x .

Using these identities, it suffices to prove that for any subset of indexes i , index j ∈ [ t ] , and input prefix view v j -1 , we have P A a t,k ( M ;( s ,x )) ( v j | v j -1 , i ) = P φ ( A t,k ( R ; ∗ )) ( v j | v j -1 , i ) and P A a t,k ( M ;( s , ⊥ )) ( v j | v j -1 , i ) = P φ ( A t,k ( R ; ⊥ )) ( v j | v j -1 , i ) .

From the definition,

<!-- formula-not-decoded -->

In the case of the null element, the allocation doesn't have any effect so we have,

<!-- formula-not-decoded -->

which completes the proof.

Proof. [Proof of Lemma 3.2] Notice that the random allocation of k indexes out of t can be described as a two steps process, first randomly splitting t into k subsets of size t/k , 8 then running A t/k, 1 ( R ) on each of the k copies of the scheme. Using the same convexity argument as in the proof of Lemma 3.1, the privacy profile of A t,k ( R ) is upper bounded by the composition of k copies of A t/k, 1 ( R ) .

We remark that this lemma holds for arbitrary t -step algorithms (and not just non-adaptive randomizers) but in the adaptive case the usual sequential composition should be replaced by concurrent composition [Vadhan and Wang, 2021], which was recently proven to provide the same privacy guarantees [Lyu, 2022, Vadhan and Zhang, 2023].

## C Missing proofs from Section 4.1

<!-- formula-not-decoded -->

Theorem 4.1 follows from this lemma by identifying the total loss with a sum of the independent losses per step and using maximal Azuma-Hoeffding inequality. Theorem4.1 follows from this lemma by using a simple relationship between the privacy loss tail bound and the privacy profile.

The proof of this lemma consists of a sequences of reductions.

8 For simplicity we assume that t is divisible by k .

Proof. Following [Erlingsson et al., 2019], we introduce the posterior sampling scheme (Definition C.3), where the sampling probability depends on the previous outputs. Rather than selecting in advance a single step, at each step the scheme chooses to include the element with posterior probability induced by the previously released outputs λ v i := P A t ( R ; ∗ ) ( i +1 ∈ I | v i ) , where I is the subset of chosen steps.

Though this scheme seems like a variation of the Poisson scheme, we prove (Lemma C.4) that in fact its output is distributed like the output of random allocation, which implies they share the same privacy guarantees. The crucial difference between these two schemes is the fact that unlike random allocation, the distribution over the outputs of any step of the posterior scheme is independent of the distribution over output of previous steps given the view and the dataset, since there is no shared randomness (such as the chosen allocation).

We then define a truncated variant of the posterior distribution (Definition C.5), where the sampling probability is capped by some threshold, and bound the difference between the privacy profile of the truncated and original posterior distributions, by the probability that the posterior sampling probability will exceed the truncation threshold (Lemma C.6).

Finally, we bound the privacy profile of the truncated posterior scheme by the privacy profile of the Poisson scheme with sampling probability corresponding to the truncation threshold, using the fact the privacy loss is monotonically increasing in the sampling probability (Lemma C.8), which completes the proof. Part of these last two lemmas is a special case of the tail bound that recently proved in [Choquette-Choo et al., 2023, Theorem 3.1].

Formally,

<!-- formula-not-decoded -->

where (1) results from Lemma C.4, (2) from Lemma C.6, and (3) from Lemma C.8.

The same proof can be repeated as is for the add direction.

Remark C.2. Repeating the previous lemmas while changing the direction of the inequalities and the sign of the lower order terms, we can similarly prove that the random allocation scheme upper bounds the Poisson scheme up to lower order terms, which implies they are asymptotically identical.

Throughout the rest of the section, claims will be stated in terms of R and ∗ for simplicity, but can be generalized to M and x . We note that A t,k ( R ; ⊥ ) = R ⊗ t ( ⊥ ) where R ⊗ t ( ⊥ ) denotes t sequential calls to R ( ⊥ ) , an identity which will be used several times throughout the next section

## C.1 Posterior scheme

We start by introducing the posterior sampling scheme, where the sampling probability depends on the previous outputs.

Definition C.3 (Posterior probability and scheme) . Given a subset size k ∈ [ t ] , an index i ∈ [ t -1] , a view v i ∈ Y i , and a randomizer R , the i + 1 posterior probability of the k allocation out of t given v i is the probability that the index i + 1 was one of the k steps chosen by the random allocation scheme, given that the view v i was produced by the first i rounds of A t ( R ; ∗ ) . Formally, λ v i ,k := P A t,k ( R ; ∗ ) ( i +1 ∈ I | v i ) , where I is the subset of chosen steps.

The posterior scheme is a function T t,k ( R ) : {∗ , ⊥} → Y t parametrized by a randomizer R , number of steps t , and number of selected steps k , which given ∗ , sequentially samples

<!-- formula-not-decoded -->

where λ v 0 ,k = k/t , and T t,k ( R ; ⊥ ) = A t,k ( R ; ⊥ ) . As before, we omit k from the notations where k = 1 .

Though this scheme seems like a variation of the Poisson scheme, the following lemma shows that in fact its output is distributed like the output of random allocation.

Lemma C.4. For any subset size k ∈ [ t ] and randomized R , A t,k ( R ; ∗ ) and T t,k ( R ; ∗ ) are identically distributed, which implies ⃗ δ A t,k ( R ) ( ε ) = ⃗ δ T t,k ( R ) ( ε ) and ⃗ δ A t,k ( R ) ( ε ) = ⃗ δ T t,k ( R ) ( ε ) for any randomizer and all ε ≥ 0 .

Proof. We notice that for all j ∈ [ t -1] and v j ∈ Y j ,

<!-- formula-not-decoded -->

where (1) denotes the subset of steps selected by the allocation scheme by I so I = i denotes the selected subset was i , (2) results from the definition v j +1 = ( v j , y j +1 ) and Bayes law, (3) from the fact that if j +1 ∈ I then y j +1 depends only on a ∗ and if j +1 / ∈ I then y j +1 depends only on ⊥ , and (4) is a direct result of the posterior scheme definition.

<!-- formula-not-decoded -->

## C.2 Truncated scheme

Next we define a truncated variant of the posterior distribution and use it to bound its privacy profile.

Definition C.5 (Truncated scheme) . The truncated posterior scheme is a function T t,k,η ( R ) : {∗ , ⊥} → Y t parametrized by a randomized R , number of steps t , number of selected steps k , and threshold η ∈ [0 , 1] , which given ∗ , sequentially samples

<!-- formula-not-decoded -->

where λ η v i ,k := min { λ v i ,k , η } , and T t,k,η ( R ; ⊥ ) = T t,k ( R ; ⊥ ) .

Next we relate the privacy profiles of the posterior and truncated schemes.

Lemma C.6. Given a randomizer R , for any η ∈ [ k/t, 1] ; ε &gt; 0 we have

<!-- formula-not-decoded -->

where ⃗ β A t ( R ) ( η ) and ⃗ β A t ( R ) ( η ) were defined in Lemma C.1.

The proof of this lemma relies on the relation between λ v i and the privacy loss of the random allocation scheme, stated in the next claim.

Claim C.7. Given i ∈ [ t -1] , a randomizer R , and a view v i we have λ v i = 1 t e ℓ A t ( R ) ( v i ; ⊥ , ∗ ) .

Proof. From the definition,

<!-- formula-not-decoded -->

where (1) results from Bayes law and (2) from the fact P ( I = i +1) = 1 t and P A t ( R ; ∗ ) ( v i | I = i +1 ) = P A t ( R ; ⊥ ) ( v i ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which after reordering the terms implies

<!-- formula-not-decoded -->

and similarly,

<!-- formula-not-decoded -->

where (1) results from the definition of the truncated posterior scheme and the set B t η , (2) from the fact that for any couple of distributions P, Q over some domain Y

<!-- formula-not-decoded -->

and (3) from the definition of ⃗ β A t ( R ) ( η ) .

Combining this with Claim C.7 we get

<!-- formula-not-decoded -->

and similarly

<!-- formula-not-decoded -->

We now relate the truncated scheme's privacy profile to Poisson.

Lemma C.8. Given η ∈ [0 , 1] and a randomizer R , we have ⃗ δ T t,η ( R ) ( ε ) ≤ ⃗ δ P t,η ( R ) ( ε ) and ⃗ δ T t,η ( R ) ( ε ) ≤ ⃗ δ P t,η ( R ) ( ε ) for all ε &gt; 0 .

The proof makes use of the following result.

Claim C.9 (Theorem 10 in [Zhu et al., 2022]) . If a pair of distributions ( P, Q ) dominates a randomizer R and ( P ′ , Q ′ ) dominate R ′ , then ( P × P ′ , Q × Q ′ ) dominate the composition of R and R ′ .

Proof of Lemma C.8. We first notice that the the hockey-stick divergence between a mixture distribution λP +(1 -λ ) Q and its second component Q is monotonically increasing in its mixture parameter λ . For any 0 ≤ λ ≤ λ ′ ≤ 1 and two distributions P 0 , P 1 over some domain, denoting Q λ := (1 -λ ) P 0 + λP 1 we have, Q λ ′ = 1 -λ ′ 1 -λ Q λ + λ ′ -λ 1 -λ P 1 . From the quasi-convexity of the hockey-stick divergence, for any κ ≥ 0 we have

<!-- formula-not-decoded -->

and similarly, H κ ( P 1 ∥ Q λ ′ ) ≤ H κ ( P 1 ∥ Q λ ) .

Using this fact we get that the privacy profile of a single call to a Poisson subsampling algorithm is monotonically increasing in its sampling probability w.r.t. both ∼ → and ∼ ← , so the privacy profile of every step of T t,η ( R ) is upper bounded by that of P 1 ,η ( R ) , and from Claim C.9 its t times composition is the dominating pair of P t,η ( R ) , which completes the proof.

## C.3 Proof of Lemma C.1

Proof. The proof directly results from combining the previous lemmas.

<!-- formula-not-decoded -->

where (1) results from Lemma C.4, (2) from Lemma C.6, and (3) from Lemma C.8.

The same proof can be repeated as is for the add direction.

## C.4 Proof of the first part of Theorem 4.1

We first reduce the analysis of general approximate-DP algorithms to that of pure-DP ones, paying an additional tδ 0 term in the probability.

Claim C.10. Given ε 0 &gt; 0 ; δ 0 ∈ [0 , 1] and a ( ε 0 , δ 0 ) -DP randomizer R , there exists a randomized ˆ R which is ε 0 -DP, such that ⃗ β A t,k ( R ) ( η ) ≤ ⃗ β A t,k ( ˆ R ) ( η ) + tδ 0 and ⃗ β A t,k ( R ) ( η ) ≤ ⃗ β A t,k ( ˆ R ) ( η ) + tδ 0 , where β A t,k ( R ) ( η ) was defined in Lemma C.1.

Proof of Claim C.10. From Lemma 3.7 in [Feldman et al., 2021], there exists a randomizer ˆ R which is ε 0 -DP, such that ˆ R ( ⊥ ) = R ( ⊥ ) and D TV ( R ( ∗ ) ∥ ˆ R ( ∗ )) ≤ δ 0 .

For any i ∈ [ t ] consider the posterior scheme T t,k, ( i ) ( ˆ R ) which ∀ j &lt; i returns

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Notice that T t,k, (0) ( ˆ R ) = T t,k ( R ) and T t,k, ( t ) ( ˆ R ) = T t,k ( ˆ R ) . From the definition, for any i ∈ [ t ] we have D TV ( T t,k, ( i -1) ( ˆ R ; ∗ ) ∥T t,k, ( i ) ( ˆ R ; ∗ )) ≤ δ 0 , which implies D TV ( T t,k ( R ; ∗ ) ∥T t,k ( ˆ R ; ∗ )) ≤ tδ 0 .

Combining this inequality with the fact that for any two distributions P, Q over domain Ω and a subset C ⊆ Ω we have P ( C ) ≤ Q ( C ) + D TV ( P ∥ Q ) completes the proof.

and ∀ j ≥ i returns

Next we provide a closed form expression for the privacy loss of the random allocation scheme. Claim C.11. Given an index i ∈ [ t ] and a view v i ∈ Y i we have,

<!-- formula-not-decoded -->

Proof. From the definition,

<!-- formula-not-decoded -->

which implies,

Using this identity we get,

<!-- formula-not-decoded -->

We are now ready to prove the corollary.

Proof of Theorem 4.1. Using Claim C.10 we can limit our analysis to a ε 0 -pure DP randomizer. We have,

<!-- formula-not-decoded -->

where (1) results from Claim C.11, and similarly,

<!-- formula-not-decoded -->

We can now define the following martingale; D 0 := 0 , ∀ j ∈ [ t -1] : D j := 1 -e ℓ R ( Y j ; ∗ , ⊥ ) , and S i := ∑ i j =0 D j . Notice that this is a sub-martingale since for any j ∈ [ t -1]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where R α is the α -Rényi divergence (Definition 2.3).

From the fact R is ε 0 -DP we have 1 -e -ε 0 ≤ D j ≤ 1 -e ε 0 almost surely, so the range of D j is bounded by e ε 0 -e -ε 0 = 2cosh( ε 0 ) , and we can invoke the Maximal Azuma-Hoeffding inequality and get for any l ∈ [ t ] ,

<!-- formula-not-decoded -->

where the last two steps result from the definition of η and γ

## C.5 Proof of the second part of Theorem 4.1

The proof makes use of the following result.

Claim C.12 (Part 2 of lemma 3.3 in Kasiviswanathan and Smith [2014]) . Given ε &gt; 0 ; δ ∈ [0 , 1] and a ( ε, δ ) -DP algorithm M , for any neighboring datasets s ≃ s ′ we have,

<!-- formula-not-decoded -->

Proof. Consider the following algorithm based on the randomizer R η : {∗ , ⊥} × Y ∗ →Y which is defined by R η ( ⊥ ; v ) = R ( ⊥ ) , and R η ( ∗ ; v ) = { R ( ∗ ) ℓ A t ( R ) ( v i ; ⊥ , ∗ ) &lt; ln( tη ) R ( ⊥ ) ℓ A t ( R ) ( v i ; ⊥ , ∗ ) ≥ ln( tη ) .

Given a view v , denote

<!-- formula-not-decoded -->

the first index where the privacy loss exceeds ln( tη ) if such index exists and 0 otherwise, and notice that,

<!-- formula-not-decoded -->

where (1) results from Claim C.11 and (2) from the definition of R η which implies the distribution of R η ( ∗ ; v j ) = R η ( ⊥ ; v j ) for all j &gt; i v .

9 We note the journal version has a typo in the δ part of the statement, which does not match the proof. We use the corrected version which appears in the Arxiv version.

.

Using this fact we get,

<!-- formula-not-decoded -->

and repeating all steps but (5) we similarly get,

<!-- formula-not-decoded -->

where (1) results from the definition of i v , (2) from Claim C.11, (3) from the definition of R η , (4) from the previous identity, (5) from the definition of the privacy loss, (6) from Claim C.12, (7) from the fact R η is dominated by R , and (8) from the definition of η and τ .

## C.6 Separate directions

For completeness we present the results of figure 1 for the add and remove directions separately for a single and multiple allocations.

Figure 3: Upper bounds on privacy parameter ε or the add and remove directions as a function of the noise parameter σ for various schemes, all using the Gaussian mechanism with fixed parameters δ = 10 -10 , t = 10 6 , the same setting as Figure 1

<!-- image -->

Figure 4: Upper bounds on privacy parameter ε or the add and remove directions as a function of the noise parameter σ for various schemes, all using the Gaussian mechanism with fixed parameters δ = 10 -10 , t = 10 6 , k = 10 , the same setting as Figure 1

<!-- image -->

## D Asymptotic analysis

We start by recalling the asymptotic bounds for the Poisson scheme due to Abadi et al. [2016]. 10 Lemma D.1 ([Abadi et al., 2016]) . There exists constants c 1 , c 2 &gt; 0 such that for any t ∈ N ; λ ∈ [0 , 1 / 16] ; δ ∈ [0 , 1] , if t ≥ ln(1 /δ ) and σ &gt; max { 1 , c 1 √ ln(1 /δ ) λ √ t } then the Poisson scheme with the Gaussian mechanism P t,λ ( N σ ) is ( ε, δ ) -DP for any ε ≥ c 2 max { λ √ t · ln(1 /δ ) σ , λ 2 √ t · ln(1 /δ ) } .

This is a direct result of the fact the Gaussian mechanism is dominated by the one-dimensional Gaussian randomizer (Claim D.3) where R ( ∗ ) = N (1 , σ 2 ) and R ( ⊥ ) = N (0 , σ 2 ) . Combining this Lemma with Corollary 4.2 implies a similar result for the random allocation scheme.

Lemma D.2. There exist constants c 1 , c 2 such that for any t ∈ N ; k ∈ [ t/ 16] ; δ ∈ [0 , 1] ; if

<!-- formula-not-decoded -->

then the random allocation scheme with the Gaussian mechanism A t,k ( N σ ) is ( ε, δ ) -DP for any

<!-- formula-not-decoded -->

The second term in the bound on ε is due to the privacy profile of the Poisson scheme, and applies only in the uncommon regime when σ &gt; t/k . One important difference between the privacy guarantees of the Poisson and random allocation schemes is in the bounds on σ , which are stricter for random allocation in the k &gt; √ t regime (Remark D.4).

The proof of this Lemma is based on the identity of the dominating pair of the Gaussian mechanism.

Claim D.3 (Gaussian randomizer [Abadi et al., 2016]) . Given σ &gt; 0 , the Gaussian mechanism N σ is tightly dominated by the pair of distributions ( N (1 , σ 2 ) , N (0 , σ 2 )) , which induce a Gaussian randomizer where ∗ := 1 and ⊥ := 0 . This pair can be realized by datasets of arbitrary size n of n -1 times n times vectors in dimension

d

by the pair

((

¯

0

, . . . ,

¯

0

, e

1

)

,

(

¯

0

, . . . ,

¯

0))

.

We note that the dominating pair of the Gaussian is one dimensional, regardless of the dimension of the original algorithm.

10 This is a variant of Abadi et al. [2016, Theorem 1] that is better suited for comparison. We prove this version in Appendix C.3.

︷

︸︸

︷

︷

︸︸

︷

Proof of Lemma D.2. From Theorem 4.1, each of the schemes has a privacy profile δ A t/k ( R ) ( ε ) ≤ δ P t/k,η ( R ) ( ε ) + t/kδ 0 + δ/k . Applying the union bound to the t/kδ 0 and δ/k terms, and using the fact that the composition of Poisson schemes is a longer Poisson scheme completes the proof of the first part.

From Lemma A.2 we have ε 0 = √ 2 ln(1 . 25 /δ 0 ) σ = √ 2 ln(1 . 25 t/δ ) σ (see e.g., Dwork and Roth [2014] for exact derivation). From the first bound on σ we get ε 0 ≤ 1 and therefore cosh( ε 0 ) = ( e ε 0 -e ε 0 ) / 2 ≤ 3 ε 0 / 2 . Combining this with the second bound on σ we get,

<!-- formula-not-decoded -->

which implies η ≤ 2 k t and δ P t,η ( N σ ) ( ε ) ≤ δ P t, 2 k/t ( N σ ) ( ε ) , since the Poisson scheme's privacy profile is monotonic in the sampling probability as proven in Lemma C.8.

Remark D.4. While the asymptotic bound on ε for the Poisson and random allocation schemes is identical up to the additional logarithmic dependence on t , only the third bound on σ stated for random allocation is required for Poisson. Notice that if √ t &gt; k the third term upper bounds the first one, and if additionally ln(1 /δ ) ≤ t 2 k 3 the second term is bounded by the third one as well. While the first condition might not hold when each element is allocated to many steps, the latter does not hold only when t &lt; ln 2 (1 /δ ) which is an uncommon regime of parameters.

## E Missing proofs from Section 4.3

Lemma E.1. Given 1 ≤ k ≤ k ′ ≤ t and a randomizer R dominated by a randomizer R we have ⃗ δ A t,k ( R ) ( ε ) ≤ ⃗ δ A t,k ′ ( R ) ( ε ) and ⃗ δ A t,k ( R ) ( ε ) ≤ ⃗ δ A t,k ′ ( R ) ( ε ) . Furthermore, for any sequence of integers k ≤ k 1 &lt; . . . &lt; k j ≤ t , and non-negative λ 1 , . . . , λ j s.t. λ 1 + . . . + λ j = 1 , the privacy profile of A t,k ( R ) is upper-bounded by the privacy profile of λ 1 A t,k 1 ( R )+ . . . + λ j A t,k j ( R ) , where we use convex combinations of algorithms to denote an algorithm that randomly chooses one of the algorithms with probability given in the coefficient.

Proof. To prove this claim, we recall the technique used in the proof of Lemma C.1. We proved in Lemma C.4 that A t,k ( R ; ∗ ) and T t,k ( R ; ∗ ) are identically distributed. From the non-adaptivity assumption, this is just a sequence of repeated calls to the mixture algorithm λ v i ,k, ∗ · R ( ∗ ) + (1 -λ v i ,k, ∗ ) · R ( ⊥ ) .

Next we recall the fact proven in Lemma C.8 that the hockey-stick divergence between this mixture algorithm and R ( ⊥ ) is monotonically increasing in λ . Since λ v i ,k ′ , ∗ ≥ λ v i ,k, ∗ for any k ′ &gt; k , this means the pair of distributions ( λ v i ,k ′ , ∗ · R ( ∗ ) + (1 -λ v i ,k ′ , ∗ ) · R ( ⊥ ) , R ( ⊥ )) dominates the pair ( λ v i ,k ′ , ∗ · R ( ∗ ) + (1 -λ v i ,k ′ , ∗ ) · R ( ⊥ ) , R ( ⊥ )) for any iteration i and view v i (this domination holds in both directions). Using Claim C.9 this implies we can iteratively apply this for all step and get δ A t,k ( R ) ( ε ) ≤ δ A t,k ′ ( R ) ( ε ) for any ε &gt; 0 , thus completing the proof of the first part.

The proof of the second part is identical, since the posterior sampling probability induced by any mixture of A t,k 1 ( R ) , . . . , A t,k j ( R ) is greater than the one induced by A t,k ( R ) the same reasoning follows.

Lemma E.2. For any λ ∈ [0 , 1] , element x ∈ X , and randomizer R we have,

<!-- formula-not-decoded -->

where B t,λ is the PDF of the binomial distribution with parameters t, λ and A t, 0 ( R ; x ) := R ⊗ t ( ⊥ ) .

Proof. This results from the fact that flipping t coins with bias λ can be modeled as first sampling an integer k ∈ { 0 , 1 , . . . , t } from a binomial distribution with parameters ( t, λ ) , then uniformly sampling i 1 , . . . , i k ∈ [ t ] , and setting the coins to 1 for those indexes.

The proof of the next claim is a generalization of the advance joint convexity property [Balle et al., 2018, Theorem 2].

Claim E.3. Given λ ∈ [0 , 1] ; κ &gt; 1 and two distribution P, Q over some domain, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where κ ′ := 1 + κ -1 λ , κ ′′ := 1 + κ -1 1 -κ + κλ , and β := 1 -κ + κλ .

Proof. The identity for H κ ((1 -λ ) P + λQ ∥ P ) is a direct result of the advanced joint convexity property [Balle et al., 2018, Theorem 2]. For the second part notice that,

<!-- formula-not-decoded -->

Lemma E.4. For any λ ∈ [0 , 1] ; ε &gt; 0 and randomizer R we have

<!-- formula-not-decoded -->

where is the Poisson scheme conditioned on allocating the element at least once, and ⃗ γ , ⃗γ , ⃗ ε , and ⃗ε were defined in Theorem 4.3.

Proof. First notice that,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (1) results from Lemma E.2, (2) from the definition of the binomial distribution, λ ′ and P + t,λ ( R ) , and (3) from the definition of λ ′ and the fact A t, 0 ( R ; x ) = R ⊗ t ( ⊥ ) .

Taking the converse of Claim E.3 we have,

<!-- formula-not-decoded -->

Similarly, from the previous claim we get,

<!-- formula-not-decoded -->

which completes the proof.

<!-- formula-not-decoded -->

We can now prove the main theorem.

Proof. [Proof of Theorem 4.3] The proof of this theorem consists of several key steps. First, we show (Lemma E.1) that increasing the number of allocations can only harm the privacy, that is, for any sequence of integers k ≤ k 1 &lt; . . . &lt; k j ≤ t , and non-negative λ 1 , . . . , λ j s.t. λ 1 + . . . + λ j = 1 , the privacy profile of A t,k ( R ) is upper-bounded by the privacy profile of λ 1 A t,k 1 ( R )+ . . . + λ j A t,k j ( R ) , where we use convex combinations of algorithms to denote an algorithm that randomly chooses one of the algorithms with probability given in the coefficient.

Next, we notice (Lemma E.2) that the Poisson scheme can be decomposed into a sequence of random allocation schemes, by first sampling the number of steps in which the element will participate from the Binomial distribution and then running the random allocation scheme for the corresponding number of steps,

<!-- formula-not-decoded -->

where B t,λ is the PDF of the binomial distribution with parameters t, λ and A t, 0 ( R ; x ) := R ⊗ t ( ⊥ ) . We then define the Poisson scheme conditioned on allocating the element at least once

<!-- formula-not-decoded -->

and use a generalized version of the advanced joint convexity (Claim E.3) to relate its privacy profile that of the Poisson scheme (Lemma E.4). Formally,

<!-- formula-not-decoded -->

where (1) results from the fact ∑ k ∈ [ t ] B t,λ ( k ) = 1 /⃗ γ , (2) from Lemma E.1, (3) from the definition of P + t,λ and the fact that P t,λ ( R ; ⊥ ) = R ⊗ t ( ⊥ ) = A t,λ ( R ; ⊥ ) , and (4) from the first part of Lemma E.4.

Repeating the same proof using the second part of Lemma E.4 proves the bound on ⃗ δ A t ( R ) .

Combining the Poisson decomposition perspective shown in Lemma E.2 with the monotonicity in number of allocations shown in Lemma E.1, additionally implies the following corollary.

Corollary E.5. For any λ ∈ [0 , 1] ; k ∈ [ t ] , we have δ P t,λ,k ( R ) ( ε ) ≤ δ P t,λ ( R ) ( ε ) , where P t,λ,k ( R ) denotes the Poisson scheme where the number of allocations is upper bounded by k .

Proof Theorem 4.3. The proof directly results from combining the previous lemmas,

<!-- formula-not-decoded -->

where (1) results from the fact ∑ k ∈ [ t ] B t,λ ( k ) = 1 /⃗ γ , (2) from Lemma E.1, (3) from the definition of P + t,λ and the fact that P t,λ ( R ; ⊥ ) = R ⊗ t ( ⊥ ) = A t,λ ( R ; ⊥ ) , and (4) from the first part of Lemma E.4.

Repeating the same proof using the second part of Lemma E.4 proves the bound on ⃗ δ A t ( R ) .

Proof. Notice that,

<!-- formula-not-decoded -->

where (1) results from Lemma 3.1, (2) from Lemma E.2 and the definition of P t,λ,k ( R ) , and (3) from Lemma E.1.

## F Missing proofs from Section 4.4

Lemma F.1. Given t, α ∈ N and two distributions P, Q over some domain Ω , we have

<!-- formula-not-decoded -->

where ¯ P := 1 t ∑ i ∈ [ t ] ( Q ⊗ ( i -1) · P · Q ⊗ ( t -i ) ) , Q ⊗ ( i -1) · P · Q t -i denotes the distribution induced by sampling all elements from Q , except for the i th one which is sampled from P .

We start by proving a supporting claim.

Claim F.2. Given α, t ∈ N and a list of integers i 1 , . . . , i t ≥ 0 such that i 1 + . . . + i t = α , denote by P ( i 1 , . . . , i t ) the integer partition of α associated with this list, e.g. if i 1 = 1 , i 2 = 0 , i 3 = 2 , i 4 = 1 , then P = [1 , 1 , 2] . Given an integer partition P of α , we have | B P | = ( t C (Π) ) where,

<!-- formula-not-decoded -->

and C (Π) was defined in Theorem 4.4.

Proof. Given a partition P with unique counts C (Π) = ( c 1 , . . . , c j ) , and an assignments i 1 , . . . , i t such that i 1 , . . . , i t ≥ 0 and P ( i 1 , . . . , i t ) = P , there are ( t c 1 ) ways to assign the first value to c 1 indexes of the possible t , ( t -c 1 c 2 ) ways to assign the second value to c 2 indexes of of the remaining t -c 1 indexes, and so on. Multiplying these terms completes the proof.

Proof of Lemma F.1. Given a set of integers i 1 , . . . , i t ≥ 0 such that i 1 + . . . + i t = α we have,

<!-- formula-not-decoded -->

where P is the integer partition of α defined by i 1 , . . . , i t , e.g. if i 1 = 1 , i 2 = 0 , i 3 = 2 , i 4 = 1 , then P = [1 , 1 , 2] . This is a result of the fact Y k are all identically distributed. Notice that the same partition corresponds to many assignments, e.g. P = [1 , 1 , 2] corresponds to i 1 = 0 , i 2 = 1 , i 3 = 1 , i 4 = 2 as well. The number of assignments that correspond to a partition P is ( t C (Π) ) . Using this fact we get,

<!-- formula-not-decoded -->

̸

where (1) results from the definition of ¯ P , (2) is the multinomial theorem, (3) results from the fact ω i and ω j are independent for any i = j , (4) from the fact ω k are all identically and independently distributed with P ( i 1 , . . . , i t ) defined in Claim F.2, and (5) results from Claim F.2.

LemmaF.3. Given λ ∈ [0 , 1] and two distributions P, Q over some domain Ω , denote P λ := P λ Q 1 -λ Z λ where Z λ is the normalizing factor.

Given t ∈ N , for any ε ∈ R we have H e ε ( Q ⊗ t ∥ ∥ ¯ P ) ≤ H e ε ′ ( Q ⊗ t ∥ ∥ ∥ P ⊗ t 1 /t ) , where ¯ P := 1 t ∑ i ∈ [ t ] ( Q ⊗ ( i -1) · P · Q ⊗ ( t -i ) ) , Q ⊗ ( i -1) · P · Q t -i denotes the distribution induced by sampling all elements from Q , except for the i th one which is sampled from P .

Proof. By definition,

<!-- formula-not-decoded -->

Given ε

<!-- formula-not-decoded -->

where (1) results from Jensen's inequality and (2) from the definition of P 1 /t and the previous claim.

Proof of Corollary 4.6. From the definition of the Rényi divergence for the Gaussian mechanism,

<!-- formula-not-decoded -->

For the add direction we notice that from the definition, for any x ∈ R

<!-- formula-not-decoded -->

so R λ ( ∗ ) = N ( λ, σ 2 ) and Z λ = e λ (1 -λ ) 2 σ 2 .

Setting λ = 1 /t we get,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (1) results from Theorem 4.5, (2) from the fact that the hockey-stick divergence between two Gaussians with the same scale depends only on the ratio of the difference between their means and their scale, and (3) from the fact that the t -composition hockey-stick divergence between two Gaussians with the same scale amounts to dividing their scale by √ t .

We remark that the expression in Corollary 4.6 for the remove direction was previously computed in Liew and Takahashi [2022], up to the improvement of using integer partitions. In this (unpublished) work the authors give an incorrect proof that datasets (0 , . . . , 0 , 1) and (0 , . . . , 0) are a dominating pair of datasets for the shuffle scheme applied to Gaussian mechanism. Their analysis of the RDP bound for this pair of distributions is correct (even if significantly longer) and the final expression is identical to ours.

Figure 5 provides a clear example of the advantage of direct analysis, in regimes where the privacy guarantees of the random allocation scheme are better than those of the Poisson scheme. Even though RDP-based bounds on Poisson are not as tight, the RDP-based of allocation is superior to other methods that rely on reduction to Poisson. On the other hand, figure 6 indicates that the gap between our RDP-based bound and Poisson's PLD-based one in other regimes, is mainly due to the fact it relies on RDP, and not a property of random allocation. It additionally reflects the fact that the gap between PLD and RDP based analysis vanishes as the number of epochs grows.

Figure 5: Upper bounds on privacy ε as a function of the number of steps t for the Poisson and random allocation schemes, for fixed parameters σ = 0 . 3 , δ = 10 -4 .

<!-- image -->

## F.1 Implementation details

Computation time of the naive implementation of our RDP calculation ranges between second and minutes on a typical personal computer, depending on the α value and other parameters, but can be improved by several orders of magnitude using several programming and analytic steps which we briefly discuss here.

On the programming side, we used vectorization and hashing to reduce runtime. To avoid overflow we computed most quantities in log form, and used and the LSE trick. While significantly reducing the runtime, programming improvements cannot escape the inevitable exponential (in α ) nature of this method. Luckily, in most settings, α ∗ - the α value which induces the tightest bound on ε is typically in the low 10 s. Unfortunately, finding α ∗ requires computing R α , so reducing the range of α values for which R α is crucial.

We do so by proving an upper bound on α ∗ in terms of a known bound on ε .

Figure 6: Upper bounds on privacy parameter ε for various schemes all using the Gaussian mechanism, as a function of E the number of 'epochs' - times the scheme was sequentially computed, for fixed parameters σ = 1 , δ = 10 -8 , t = 10 4 .

<!-- image -->

Claim F.4. Given δ ∈ (0 , 1) and two distributions P, Q and, denote by ε ( δ ) := inf x&gt; 0 ( δ ( x ) &lt; δ ) . Given ε &gt; 0 , if ε ( δ ) ≤ ε and R α ( P ∥ Q ) &gt; ε , then α ∗ &lt; α .

A direct implication of this Lemma is that searching on monotonically increasing values of α and using the best bound on ε achieved at any point to check the relevancy of α , we don't have to compute many values of α greater than α ∗ before we stop.

Proof. Denote by γ δ ( α ) the bound on ε achieved using R α ( P ∥ Q ) . From Lemma A.1, γ δ ( α ) = R α ( P ∥ Q ) + ϕ ( α ) for a non negative ϕ (except for the range α &gt; 1 / (2 δ ) which provides a vacuous bound). Since R α ( P ∥ Q ) is monotonically non-decreasing in α we have for any α ′ ≥ α ,

<!-- formula-not-decoded -->

so it cannot provide a better bound on α .

## G Comparison to other techniques

For completeness, we state how one can directly estimate the hockey-stick divergence of the entire random allocation scheme. This technique was first presented in the context of the Gaussian mechanism by Chua et al. [2024a].

We first provide an exact expression for the privacy profile of the random allocation scheme.

Lemma G.1. For any randomizer R and ε &gt; 0 we have,

<!-- formula-not-decoded -->

Given σ &gt; 0 , if N σ is a Gaussian mechanism with noise scale σ we have,

<!-- formula-not-decoded -->

We note that up to simple algebraic manipulations, this hockey-stick divergence is essentially the expectation of the right tail of the sum of t independent log-normal random variables, which can be approximated as a single log-normal random variable [Neelesh B. et al., 2007], but this approximation typically provide useful guarantees only for large number of steps.

Proof. Denote by I the index of the selected allocation. Notice that for any i ∈ [ t ] we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using this identity we get,

<!-- formula-not-decoded -->

Plugging this into the definition of the hockey-stick divergence completes the proof of the first part. The second part is a direct result of the dominating pair for the random allocation scheme of the Gaussian mechanism (Claim D.3).

## G.1 Monte Carlo simulation Chua et al. [2024a]

Using Monte Carlo simulation to estimate this quantity, is typically done using the E ω ∼ P [ [ 1 -αe -ℓ ( ω ; P,Q ) ] + ] representation of the hockey-stick divergence, so that numerical stability can be achieved by bounding the estimates quantity ∈ [0 , 1] .

A naive estimation will require an impractical number of experiments, especially in the low δ and high confidence level regimes. These challenges can be partially mitigated using importance sampling and order statistics, a new technique recently presented by Chua et al. [2024a]. Still, this technique suffers from several limitations. It can only account for the setting of k = 1 and does not provide a full PLD, and so cannot be composed. It can only estimate δ , so plotting ε as a function of some other parameter is computationally prohibitive. Figure 7 illustrates simultaneously the tightness of our bounds, which are within a constant from the lower bound in the delta regime, and the limitations of the MC methods which become loose in the δ &lt; 10 -4 regime for the chosen parameters.

Figure 7: Comparison of δ bound for Poisson scheme with various bounds for the random allocation scheme, for several σ and t values; our combined methods, the high probability and the average estimations of using Monte Carlo simulation with order statistics, 5 · 10 5 samples and 99% confidence level, by Chua et al. [2024a].

<!-- image -->

We additionally repeat the analysis using the experimental setting presented in Chua et al. [2024a, Figure 2], both in the form of Figure 1 and Figure 7. The choice of t depends on the experimental settings, the dataset (Criteo pCTR or Criteo search conversion logs) and the batch size.

Figure 8: Upper bounds on privacy parameter ε as a function of the noise parameter σ for various schemes and the local algorithm (no amplification), all using the Gaussian mechanism, with privacy parameters δ = 10 -7 and various values of t , following the experimental parameters following the experimental settings of Chua et al. [2024a, Figure 2]. In the Poisson scheme λ = 1 /t .

<!-- image -->

Figure 9: Comparison of δ bound for Poisson scheme with various bounds for the random allocation scheme, for several σ and t values; our combined methods, the high probability and the average estimations of using Monte Carlo simulation with order statistics, 5 · 10 5 samples and 99% confidence level, following the experimental settings of Chua et al. [2024a, Figure 2].

<!-- image -->

## G.2 RDP-based bound by Dong et al. [2025]

A recent independent work by Dong et al. [2025] considered the same setting under the name Balanced Iteration Subsampling. In Theorem 3.1 they provide two RDP bounds for the remove direction and one for add, that are comparable to Theorem 4.4 in our work. Since the bound for the remove direction always dominated the add direction, we focus on it. The first one is tight but computationally expensive even for the case of k = 1 , as it sums over O ( t kα ) terms (in the case of k = 1 their expression matches the one proposed by Liew and Takahashi [2022], which is mathematically identical to our, but requires O ( t α ) summands rather than our O (2 α ) ones.). The second bound they propose requires summing only over a linear (in k ) number of terms which is significantly more efficient than our term, but is lossy. This gap is more pronounced in some parameter regimes, mainly when the α used for inducing the best bound on ε is large. On the other hand, this method allows for direct analysis of the k &gt; 1 case, while our analysis relies on the reduction to composition of k runs of the random allocation process with a selection of 1 out of t/k steps.

Figure 10 depicts the spectrum of these effects. For small values of k , our RDP based bounds are tighter than the loose bound proposed by Dong et al. [2025], while for the large values of k when ε is quite large our composition based analysis is looser.

Figure 10: Upper bounds on privacy parameter ε as a function of the the number of allocations k for the Poisson and random allocation schemes, all using the Gaussian mechanism with fixed parameters δ = 10 -6 , t = 2 10 , σ = 0 . 6 . The y-axis uses logarithmic scale to emphasize the relative performance.

<!-- image -->

## H Privacy-utility tradeoff

The results of Chua et al. [2024a] show that in the context of training DP-SGD, random allocation (or balls-and-bins sampling) has the utility benefits of shuffling while having the privacy benefits of Poisson subsampling. Here we investigate the privacy-utility trade-off in a simple-to-analyze setting of mean estimation over a Boolean hypercube, that illustrates one possible source of this relative advantage.

We start with the one-dimensional setting. Consider a dataset s ∈ { 0 , 1 } n sampled iid from a Bernoulli distribution with expectation p ∈ [0 , 1] , where p is estimated from the data elements using one of the two schemes. Formally, at each iteration, the algorithm reports a noisy sum of the elements in the corresponding subset y i , and the estimated expectation is ˆ p := 1 n ∑ i ∈ [ t ] y i .

Since ˆ p is averaged over the various steps, in the case of random allocation with the Gaussian mechanism we have ˆ p A = 1 n ( ∑ x ∈ s x + ∑ i ∈ [ t ] ξ i ) where ξ i is the noise added at step i . From the property of the Gaussian mechanism ∑ i ∈ [ t ] ξ i is a Gaussian random variable with variance tσ 2 , and from the definition of the distribution, ∑ x ∈ s x ∼ Bin ( n, p ) , so ˆ p A ∼ 1 n ( Bin ( n, p ) + N (0 , tσ 2 ) ) . In particular this implies E [ˆ p A ] = p and Var (ˆ p A ) = p (1 -p ) n + tσ 2 n 2 , where the first term is the sampling noise and the second is the privacy noise.

Poisson subsampling adds some complexity to the analysis, but can be well approximated for large sample size. The estimation ˆ p P follows a similar distribution to that of ˆ p A , with an additional step. First we sample u ∼ Bin ( n, p ) , then - following the insight introduced in Lemma E.2 - we sample v i ∼ Bin ( t, 1 /t ) for all i ∈ [ u ] , which amounts to sampling m ∼ Bin ( u · t, 1 /t ) . We note that E [ m ] = u and Var ( m ) = u · t · 1 t ( 1 -1 t ) ≈ u . Since w.h.p. u ≈ p · n , we get E [ˆ p P ] = p and Var (ˆ p A ) ≈ p (1 -p ) n + p n + tσ 2 n 2 , where the first term is the sampling noise, the second is the Poisson sampling noise, and the third is the privacy noise.

The noise scale required to guarantee some fixed privacy parameters using the random allocation scheme is typically larger than the one required by the Poisson scheme, as shown in Figures 1. But following the asymptotic analysis discussed in Section 4.2 we have tσ 2 ≈ ln(1 /δ ) ε 2 for both schemes, with constants differing by ≈ 10% in most practical parameter regimes, as shown in Figures 1 and 2, which implies the privacy noise tσ 2 n 2 is typically slightly larger for the random allocation scheme. On the other hand, for p → 1 , the Poisson sampling noise is arbitrarily larger than the sampling noise. Since the privacy bound becomes negligible as n increases, we get that the random allocation scheme asymptotically (in n ) dominates the Poisson scheme, as illustrated in Figure 11. In the ε = 1 , d = 1

case, the induced σ is sufficiently small, so the gap is dominated by the additional Poisson sampling noise, when ε = 0 . 1 , this effect becomes dominate only for relatively large sample size.

In the high-dimensional setting the privacy noise dominates the sampling noise and therefore the privacy-utility tradeoff is dominated by the difference in the (known) privacy guarantees of the two schemes. In Figure 11 we give an example of this phenomenon for d = 1000 .

Figure 11: Analytical and empirical square error for the Poisson and random allocation scheme for the setting discussed in Appendix H, for various values of ε and d (which corresponds to an increase in sensitivity). We set p = 0 . 9 , t = 10 3 , δ = 10 -10 . The experiment was carried 10 4 times, so the 3 -std confidence intervals are barely visible.

<!-- image -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: All claims are mathematically proven.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss both tightness of the results and computation limitations when applied.

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

[Yes]

Justification: All the claims are detailed and all the proofs appear in the appendices.

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

Justification: The paper discusses the parameters and the supplementary material includes the code.

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

Justification: The code is provided as supplementary material. No data is involved.

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

Answer: or [NA]

Justification: The paper only contains numerical analysis without any usage of data. All relevant details were provided.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The paper mostly contains numerical analysis without any usage sampling. when MC based methods are considered, CI are provided.

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

The paper only contains numerical analysis, the longest of which runs for several minutes on a personal computer.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: This is a purely theoretical work.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: This paper provides several new tools for analyzing the privacy of machine learning algorithms. We do not anticipate any impacts beyond those typical for such results.

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

Justification: No data or models were released in this work.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Details can be found in the README file.

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

Justification: No new assets were introduced.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: No such thing was done in this paper.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: No such thing was done in this paper.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: LLMs only assisted with some of the technical coding tasks.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.