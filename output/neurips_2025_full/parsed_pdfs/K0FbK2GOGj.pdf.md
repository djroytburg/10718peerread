## Instance-Optimality for Private KL Distribution Estimation

Jiayuan Ye ∗ National University of Singapore

Vitaly Feldman Apple

## Abstract

We study the fundamental problem of estimating an unknown discrete distribution p over d symbols, given n i.i.d. samples from the distribution. We are interested in minimizing the KL divergence between the true distribution and the algorithm's estimate. We first construct minimax optimal private estimators. Minimax optimality however fails to shed light on an algorithm's performance on individual (non-worst-case) instances p and simple minimax-optimal DP estimators can have poor empirical performance on real distributions. We then study this problem from an instance-optimality viewpoint, where the algorithm's error on p is compared to the minimum achievable estimation error over a small local neighborhood of p . Under natural notions of local neighborhood, we propose algorithms that achieve instance-optimality up to constant factors, with and without a differential privacy constraint. Our upper bounds rely on (private) variants of the Good-Turing estimator. Our lower bounds use additive local neighborhoods that more precisely captures the hardness of distribution estimation in KL divergence, compared to ones considered in prior works.

## 1 Introduction

Accurately estimating a discrete distribution (over d symbols) from the empirical samples is a fundamental task in statistical machine learning. An especially important distribution estimation objective is the Kullback-Leibler (KL) divergence error, as it is crucial in promoting diversity and smoothness by penalizing zero-mass assignment to unseen symbols. Noticeably in speech-recognition and language modeling communities, negative log likelihood on a test set (which up to translation is the KL divergence) is the measure that has been found to best correlate with the performance of the model [13, 14], and is therefore the standard loss being optimized for. Additionally, KL error is well-established in the coding community [18, 42] in the context of data compression (i.e., identifying encoding that represents the information with fewer bits). Due to this reason, KL distribution estimation has been extensively studied [62, 11, 10, 51, 52, 39, 49].

However, overly fine-grained release of statistics make it possible to infer the memberships [37, 55, 28, 56] or even reconstruct the values [20, 12] of input data records. To this end, differential privacy [26] offers a powerful mathematical definition to provably control the associated privacy risk.

Definition 1.1 (Differential Privacy (DP) [26]) . A (randomized) algorithm A is ( ε, δ ) -differentially private ( ( ε, δ ) -DP) if for all neighboring datasets x, x ′ that differ in at most one sample, and any measurable output set T , Pr[ A ( x ) ∈ T ] ≤ e ε · Pr[ A ( x ′ ) ∈ T ] + δ .

Our work aims to provide an understanding of the KL error of private distribution estimation. Below we present the problem setting and summarize known results and our contributions.

* Research done while at Apple

Kunal Talwar Apple

Problem Setting Let p ∈ ∆( d ) := { p ∈ R d : p 1 , · · · , p d ≥ 0 and ∑ d i =1 p i = 1 } be an unknown distribution over d symbols. Let x ∈ N d be the histogram representation of a dataset consisting of empirical samples drawn from distribution p , where x i denotes the count of symbol i in the dataset. Following prior works [2, 49], we assume that the count of each symbol i is independently drawn from a Poisson distribution with mean np i , i.e., ( x 1 , · · · , x d ) ∼ Poi ( np ) := Poi ( np 1 ) ×··· × Poi ( np d ) . This assumption is a convenient choice for analysis as it ensures independent sampling of each symbol, while enjoying the benefit of being equivalent to sampling from multinomial distribution when conditioned on a fixed dataset size n . Our goal is to design an algorithm that accurately estimates the unknown distribution p in KL divergence given a sampled dataset x ∼ Poi ( np ) .

Minimax Optimality Results (and Their Limitations) We start by looking at the (private) KL distribution estimation problem from the standard minimax optimality objective, where the goal is to design estimators A that minimizes the KL error on the worst-case distribution instance.

<!-- formula-not-decoded -->

where KL ( p ∥A ( x )) = ∑ i p i log p i A ( x ) i , and the expectation is over the randomness of estimator A and over the sampling of x ∼ Poi ( np ) . This minimax objective (1) is well-studied for non-DP algorithms, where simple add-constant estimators are proved to be minimax optimal with O ( ln ( 1 + d n )) rates [10, 52] (see Appendix C.1 for a complete discussion of the existing results). For the DP setting, to the best of our knowledge, there are no known results for the KL minimax rates. Nevertheless, in Appendix C.2, we show that a similarly simple algorithm (that truncates the Laplace perturbations of empirical counts) achieves the minimax optimal O ( ln ( 1 + d εn )) rates.

However, such simple estimators achieve poor performance in experiments (Section 4) on commonly occurring distributions such as power-law distributions. Similar observations, i.e., the poor performance of the minimax-optimal add-constant estimator (compared to the practical Good-Turing [33] estimator), have long existed in the non-DP setting. This is intuitively because minimax optimality only captures the worst-case error over all possible distributions, thus failing to indicate whether an algorithm performs well for each non-worst-case distribution p .

Instance-optimality Instance-optimality is a promising framework to address the above limitation of minimax optimality and shed light on the per-instance provable performance of an estimator. In the non-DP setting, many works have studied instance-optimality in different contexts, such as local minimax estimation [21, 47, 23], competitive distribution estimation [2, 49] or instance-by-instance analysis [59]. Remarkably, the seminal work of [49] proved that a simple variant of Good-Turing estimators is nearly instance-optimal, in that it estimates every distribution nearly as well as the best estimator designed with prior knowledge of the distribution up to a permutation. The recent work by Feldman, McMillan, Sivakumar and Talwar [30], further generalize this instance-optimality definition to other natural definitions of prior knowledge in the language of per-instance neighborhood. Formally, we follow Feldman, McMillan, Sivakumar and Talwar [30] and define instance-optimality as follows.

Definition 1.2 (Instance-Optimality to Neighborhood Map N [30]) . We say an estimator A is instance-optimal with respect to a neighborhood map N if:

<!-- formula-not-decoded -->

That is, instance-optimality says that the algorithm A is competitive with any hypothetical algorithm A ′ that has auxiliary knowledge about the neighborhood N ( p ) of the input distribution p . We can further constrain the algorithm to be ( ε, δ ) -DP in establishing the per-instance lower bound as follows.

<!-- formula-not-decoded -->

This is the lower bound that we will use for establishing instance-optimality of private algorithms.

Limitations of Prior Instance-Optimality Objectives for KL Error In instance-optimality results, the smaller the neighborhood, the stronger the result. This is because we are proving that our algorithm is competitive with a hypothetical algorithm that already knows that the input distribution is in the neighborhood. However, if the neighborhood is too small, then no algorithm can be instance-optimal, as the baseline estimator's knowledge of the instance is overly precise. (As an extreme example, when N ( p ) = { p } contains only the target distribution p , the per-instance lower bounds (2) and (3) trivially become zero (as achieved by an estimator that always outputs A ( x ) = p regardless of the input dataset).) Below we review existing choices of neighborhood N ( p ) in the literature, and show they are either too large or too small to accurately capture the per-instance estimation hardness in KL.

- Permutation neighborhood [2, 49, 59]: N π ( p ) = { q : ∃ permutation π s.t. q i = p π ◦ i , ∀ i ∈ [ d ] } , i.e., all distributions obtained by permuting the probability values. On the one hand, prior work [30] has argued that this neighborhood is too large, as practical estimators often have stronger knowledge about which symbols are more frequent than others (e.g., the word 'and' is often more frequent than 'differential', thus assuming the two words are permutable is not reasonable). On the other hand, permutation neighborhood can be too small, in that it provably does not allow for multiplicative instance optimality guarantees, as discussed in Section 2.2. Indeed, permutation neighborhood assumes precise knowledge of the set of true probability values, which can be overly strong for any realistic estimator to satisfy.
- Two-point neighborhood [23, 47, 8]: N 2 ( p ) = { p, q } contains the target instance p and another alternative distribution q that is 'close' to p . This neighborhood reduces the perinstance hardness to binary hypothesis testing of whether the observed samples are from p or q . Under appropriate notions of 'close', this neighborhood allows matching per-instance upper and lower bounds for estimating one-parameter (exponential) families distribution under central DP [47] and local DP [23], as well as for general one-dimensional statistical estimation problems [8]. However, for high-dimensional problems, this neighborhood is provably too small, as the per-instance lower bound remain constant under growing data dimension, despite the growing hardness for distribution estimation under higher dimension. For example, there exist instances (such as long-tailed instances in our particular distribution estimation problem) whose estimation error inevitably grows with data dimension (e.g., of scale ln d/ 16 in our Theorem G.9 construction), indicating that the two-point neighborhood and multiplicative neighborhood are too small. Indeed testing can be provably easier than learning in simple settings.
- Multiplicative neighborhood [30]: N × ( p ) = { q : 1 2 p i ≤ q i ≤ 2 p i , ∀ i ∈ [ d ] } . It allows matching per-instance upper and lower bounds for distribution estimation in Wasserstein distance. However, this neighborhood is too small to capture the KL error: the per-instance lower bound is at most a constant (since an algorithm that is tailored to N ( p ) can achieve O (1) error), whereas such an error bound is provably unachievable for some distributions (e.g., our Theorem G.9 construction).

Besides the above three representative types of neighborhoods, earlier works [7, 6] on instanceoptimal DP estimation also considered other variants of neighborhoods that have similar limitations. [7] only deals with one-dimensional quantities, and [6] discusses high-dimensional problems defined on the dataset (rather than on the distribution as in our work). The notion of local minimax optimality there includes all datasets at a certain distance, whereas their second notion only competes with unbiased algorithms. These limitations call for new definitions of local neighborhood to precisely capture the per-instance hardness of distribution estimation in KL divergence, which we study in this paper.

## Our Main Contributions: Instance-Optimal Results

- We define instance-optimality objectives that more precisely capture the hardness of private KL distribution estimation. This is by constructing new neighborhoods (4) and (6) that additively perturb the probability values p i of individual symbols. We use small perturbation scales 1 /n , 1 /nϵ , and √ p i /n - to ensure the dataset (sampled by p ) could plausibly have come from other distribution in the neighborhood. (See Section 2 and 3 for more rationales on the neighborhood sizes.) This is stronger than permutation neighborhood [2, 49, 59] in allowing auxiliary knowledge of frequency order among symbols, and is thus a very strong notion of instance-optimality. Furthermore, we show in Section 3.1 that our

additive neighborhoods are (up to constants) the smallest that still allow for instance-optimal algorithms.

- Under such additive neighborhoods, we propose a new non-DP algorithm and prove it to be instance-optimal (Section 2.1). Our algorithm resembles the idea of the Good-Turing estimator (which is known to be instance-optimal in prior works), while ensuring significantly smaller sensitivity to adjacent datasets (in the DP sense). This reduced sensitivity is the key that makes our algorithm easier to privatize. We then propose a DP version of this algorithm and show that it is instance-optimal amongst DP algorithms (Section 3.1).
- We validate the performance of our instance-optimal estimators via experiments (Section 4), and show the reward from studying instance-optimality: while the Add-constant (DP) algorithm is already minimax-optimal, our instance-optimal algorithms achieve significantly better performance on many instances of practical interest, such as power-law distributions and real-word token distributions.

## 1.1 Technical Contributions

Generalized Assouad's method for decomposable statistical distance Standard tools for proving lower bounds, such as (DP) Assouad's method [65, 3, 30], only apply to symmetric statistical distance, which is not satisfied by KL divergence. To prove KL per-instance lower bound, we propose generalized (DP) Assouad's method (Theorem A.3 and A.4) that applies to general decomposable statistical distance (Definition A.1). This allows us to prove strong per-instance lower bounds.

Reducing the Sensitivity of Good-Turing Estimator via 'Sampling Twice' The challenge of privatizing the prior (near) instance-optimal Good-Turing estimators lie in its excessively high sensitivity to neighboring datasets. At the core, Good-Turing estimator is motivated by observing that estimating 'unseen' symbols as zero-probability is biased, as the symbols that appeared exactly once would intuitively have similar probability values (which are non-zero). To correct this bias, it recursively uses the counts of symbols with frequency t +1 to estimate the probability of symbols that appeared t times, for all low-frequency symbols (e.g., t = 1 , · · · , n 1 / 3 in Orlitsky and Suresh [49]). Such computations suffer from high sensitivity, as one record could change the combined counts of symbols with frequency t by t = n 1 / 3 in the worst-case. To address this limitation, we perform bias-correction via an alternative "sampling twice" approach: partitioning the dataset into two halves, using one for identifying 'unseen' symbols and the other for estimating their combined mass. The resulting Algorithm 1 is conceptually simpler than prior Good-Turing estimators, while achieving tight instance-optimality guarantees (up to constants) and being empirically competitive in experiments. Crucially, this 'Sampling Twice' design reduces sensitivity to just one (making the estimator easy to privatize), and is the key to achieving instance-optimality in the DP setting.

Effectively Privatizing Good-Turing Estimator via Calibrated Thresholding The Good-Turing estimator is based on the intuition that the probability values of symbols that appeared zero times (in the dataset) are similar to those of symbols that appeared exactly once. However, this simple zero-or-one thresholding does not remain effective for private algorithms, because DP estimates cannot differ significantly on neighboring input datasets. This forces us to use a larger threshold. To identify the best threshold for DP distribution estimation, given every possible threshold, we separately analyze (Theorem E.2 and G.6) the DP estimation error due to False Negatives (FN) (i.e., high-probability symbols being below-threshold) versus False Positives (FP) (i.e., low-probability symbols being above-threshold). This analysis allows us to choose a threshold that balances the FN and FP errors, and achieves DP instance-optimality up to a constant factor (Section 3.1).

## 1.2 Other Related Works

The Good-Turing estimator, originally developed by [33] and simplified by [31, 49], has been widely observed to yield empirically accurate distribution estimates (especially for language modeling tasks [40, 13] under large vocabulary size). Several later works prove it to be minimax optimal [46, 22, 50] as well as instance-optimal [2, 49] for discrete distribution estimation in various metrics. However, to our knowledge, no prior works have designed or analyzed DP variants of Good-Turing estimator, which is the main technical innovation of this paper.

For DP discrete distribution estimation, minimax optimality is well-studied across a variety of models for DP, including central DP [19, 3], local DP [24, 38, 64, 25, 1] and user-level DP [44]. Existing works also focus on different error metrics, such as total variation distance (i.e., ℓ 1 error) [19] and ℓ 2 error [3] (see a summary of results in [3, Table 1-2]). Our work falls into the central DP setting, and is the first to study the KL divergence error. Additionally, we establish the stronger instance-optimality (rather than minimax optimality). The closest work to our paper in the literature is [30], where the authors study instance-optimal density estimation in Wasserstein distance (that covers discrete distribution estimation in ℓ 1 error as a special case). The results in [30] are largely incomparable to our analysis, due to the lack of (tight) conversions between total variation distance error and KL divergence error. (See Appendix C.2 for a nuanced comparison between prior TV distance lower bounds and our KL lower bound for DP minimax distribution estimation.) Indeed, achieving optimality under KL often requires non-trivial change to the algorithm and analysis compared to ℓ 1 error, as evidenced by the abundant literature on distribution estimation in KL [10, 51, 52, 2, 49].

A closely related problem is frequency estimation (a.k.a. histogram estimation), where the goal is to privately and accurately estimate the empirical distribution. DP histogram estimation is extensively studied in a variety of models for DP, including central DP [34, 63], local DP [9, 61], and shuffle DP [29, 15, 32]. Due to sampling error, algorithms for empirical frequency estimation typically do not directly yield good utility for distribution estimation.

## 2 Tighter Instance-Optimality for Non-DP Estimation

We first define instance-optimality under additive neighborhoods, that get around the limitations of previously studied neighborhood notions as discussed in Section 1.

Neighborhood Choices As discussed in Section 1, a good neighborhood should be as small as possible and contain distributions that are similarly hard to estimate compared to the target distribution instance. Additive neighborhood is thus a natural choice as intuitively, the hardness of distribution estimation does not change a lot when perturbing each symbol's probability by a small amount. The key question is how small should the scale be. To this end, our main design choices are

1. For every symbol, we allow up to t/n perturbation, for a small t ≥ 1 . This only changes its expected count up to t , and thus from the the empirically sampled count, it is hard to distinguish the perturbed distribution from the target distribution.
2. For symbol with large p i &gt; t/n , we allow a larger √ p i /n perturbation, for a small t ≥ 1 . This captures the fact that with constant probability, counts sampled from Poi ( np i ) and Poi ( np i + √ np i ) are 'indistinguishable' due to the statistical variance in sampling. Indeed these are related to the standard confidence intervals for binomial estimation [16, 4].

On top of these design choices, we try to reduce the size of the neighborhood as much as possible. Thus we add a constraint that the combined mass of small symbols (with p i ≤ t n given a small t ≥ 1 ) should not change by more than t n - this added condition only makes the results stronger (as the resulting neighborhood is smaller). As a result, we obtain the following additive neighborhood.

<!-- formula-not-decoded -->

Per-Instance Lower Bound We next prove per-instance lower bound under this neighborhood N + ( p ) . Our analysis requires generalized variants of Assouad's method (discussed in Section 1.1). We defer all proofs to Appendix D, and only present the per-instance lower bound below.

Theorem 2.1. Let p ∈ ∆( d ) be an arbitrarily fixed distribution instance. Let N + ( p ) be the additive neighborhood defined in (4) . If d ≥ 2 and n ≥ 4 , then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Table 1: Prior instance-optimality results of Non-DP estimators

|                      | KL Error Bound                                                                                   | Reference                                       |
|----------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------|
| lower ( p,n,N π )    | Ω ( ∑ ∞ t =0 E x ∼ Poi ( np ) [ ∑ i : x i = t p i ln ( p i ∑ j : x j = t 1 ∑ j : x j = t p j )]) | [49, Lemma 4 &5] N π : permutation neighborhood |
| Add-constant         | lower ( p,n,N π )+Ω ( min { n - 1 / 3 , d n })                                                   | [2, Lemma 1]                                    |
| Good-Turing          | lower ( p,n,N π )+ O ( min { n - 1 / 3 , d n })                                                  | [49, Theorem 1]                                 |
| Smoothed Good-Turing | lower ( p,n,N π )+ O ( min { n - 1 / 2 , d n })                                                  | [49, Theorem 2] [2, Theorem 2]                  |
| Any estimator        | lower ( p,n,N π )+Ω ( min { n - 2 / 3 , d n })                                                   | [49, Theorem 3]                                 |

## Algorithm 1 Non-DP 'Sampling Twice'

Input: Data partition ratio α = 0 . 5 . Independently sampled datasets x ∼ Poi ( α · np ) and x ′ ∼ Poi ((1 -α ) · np ) s.t. x + x ′ ∼ Poi ( np ) . Threshold τ = 0 . Thresholding: L = { i ∈ [ d ] : x i ≤ τ }

Estimate combined mass for symbols in L : ˜ c = max {∑ x ′ , 1

Truncate individual estimates: let ˜ x = max { x , 1 } for i = 1 , · · · , d

<!-- formula-not-decoded -->

i ∈ L i } i ′ i

To interpret this lower bound, first observe that (5) is always smaller than the Ω ( 1 + d n ) minimax lower bound, especially when there are many small symbols (with p i &lt; 1 n ) that jointly takes a small combined mass p small ( L ′ ) . As an extreme example, imagine a highly concentrated distribution p = (1 / 3 , 2 / 3 , 0 , · · · , 0) , then our per-instance lower bound (5) is as small as ln(1+ d ) n . This is significantly smaller than the ln(1 + d n ) minimax lower bound for high-dimensional setting ( d →∞ ), thus correctly indicating that p is an extremely easy-to-estimate distribution. Achieving error upper bound that matches this small per-instance lower bound (5), then serves as a strong requirement for instance-optimal algorithm to fulfill (which we prove in the next section).

## 2.1 An Instance-Optimal 'Sampling Twice' Estimator

We now present a simple 'sampling twice' Algorithm 1, and prove that it is instance-optimal up to constants. Compared to the minimax optimal add-constant estimator, this algorithm observes that simply estimating 'unseen' symbols as zero-probability (or constant-probability) is heavily biased, and use a conceptual 'sampling-twice' procedure to correct the bias of low-frequency symbols.

As discussed in Section 1.1, this bias-correction idea originates from prior (near) instance-optimal Good-Turing algorithm [33, 31, 49] and maintains their utility benefit. The key benefit of our 'sampling-twice' design lies in significantly reducing the estimator's sensitivity to neighboring dataset (making it easier to privatize), enabling instance-optimality under DP (as we will show in Section 3).

Instance-optimality Guarantee In Theorem E.2, we prove a per-instance KL error upper bounds for the 'sampling twice' estimator (Algorithm 1). In Corollary E.3, we further prove that this upper bound can be rewritten as E x ∼ Poi ( np ) [ KL ( p, A ( x ))] ≤ O ( lower ( p, n, N + )) for any choice of neighborhood size t ≥ 1 s.t. t · e -t ≤ 1 / ln d . Specifically, we can choose t = min { 1 , 2 ln ln d } .

## 2.2 Comparison to Prior Instance-Optimality Results

In this section, we discuss and compare with prior instance-optimality results, which mainly cover two representative non-DP estimators - the add-constant estimator and the Good-Turing (GT) estimator.

Add-constant Estimator Given by A ( x ) = x i + c, ∀ i ∈ [ d ] for a constant c &gt; 0 , the add-constant estimator is one of the oldest and simplest distribution estimator. Its variants cover several known estimators, such as Laplace smoothing ( c = 1 ) and Krichevsky-Trofimov estimator [42] ( c = 1 / 2 ).

Good-Turing Estimator [33] The Good-Turing estimator [33], when combined with add-constant estimator, has long been observed to yield strong empirical performance. Many variants of GT estimator exist in the literature [33, 31, 49]. In the comparison, we use the simplified variant of GoodTuring estimators proposed in Orlitsky and Suresh [49], that is also provably (near) instance-optimal under the 'permutation neighborhood', i.e. all distributions obtained by permuting the distribution.

We summarize these prior instance-optimality results under permutation neighborhood in Table 1. Apart from the fact that this neighborhood may be too large to be appropriate for some applications (as discussed in Section 1), the last row of Table 1 shows a lower bound for the suboptimality - it shows that no algorithm can do better than an additive min { n -2 / 3 , d n } compared to a hypothetical algorithm that already knows the neighborhood. Observe that this additive gap is a lot larger than the per-instance lower bounds for certain distributions, e.g., highly concentrated distributions. Thus, multiplicative instance optimality is not achievable with permutation neighborhood. By contrast, tight instance-optimality (up to constant) is achievable under our additive neighborhood, as proved in Section 2.1. We will also discuss in Section 3.1 that our additive neighborhoods are the smallest (up to constants) that can still allow for the possibility of any DP instance-optimal algorithms.

Additionally, the existing analysis for Good-Turing would result in a ln n/n additive gap compared to the lower bound we obtain under additive neighborhood. The main gap is that the Good-Turing's error for combined probability estimate is ln n/n (as in Lemma 19 of [49]) due to delicate correlations between partitioned buckets (of small symbols) and their combined mass estimates, rather than 1 /n in our sampling-twice estimator (as in (191) that applies Lemma B.10) facilitated by using fresh counts for small symbols to estimate combined mass. We will add this discussion in the revised version to clarify the comparison.

## 3 DP Instance-Optimality

In this section, we present the first results for DP instance-optimality of KL distribution estimation.

Neighborhood Choices To define appropriate DP instance-optimality objective, we modify the N + neighborhood (4) with perturbations calibrated to the privacy parameters. For a small t ≥ 1 , let

<!-- formula-not-decoded -->

Compared to (4), the main change is that we allow the perturbation scale for individual symbols to be proportional to 1 /ε . This corresponds to the obliviousness of ( ε, δ ) -DP algorithm to dataset changes up to O ( 1 ε ) hamming distance (which allows an additional 1 nε change in probability).

DP Per-Instance Lower Bound Under the above neighborhood N ≤ t nε ( p, n ) that is calibrated to ( ε, δ ) -DP guarantee, we prove the below per-instance lower bound.

Theorem 3.1. Let p ∈ ∆( d ) be an arbitrary distribution instance, and let N ≤ t nε ( p ) be the additive neighborhood defined in (6) . If δ ≤ ε , d ≥ 2 , and nε ≥ 1 , then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Algorithm 2 ε -DP 'Sampling Twice' (Instance-Optimal)

Inputs: Data partition ratio α = 0 . 5 . Independently sampled datasets x ∼ Poi ( α · np ) and x ′ ∼ Poi ((1 -α ) · np ) s.t. x + x ′ ∼ Poi ( np ) . Threshold τ = 4ln d . L = ∅

for symbol i = 1 , · · · , d do

Private Thresholding:

## end for

If

˜

x

i

:=

x

i

+

z

i

≤

min

τ

{

ε,

1

}

for

z

i

∼

Lap

0

,

1

ε

, add

i

to

L

Estimate small symbols' combined mass: ˜ c = max { ∑ i ∈ L x ′ i + Lap ( 0 , 1 ε ) , 1 min { ε, 1 } } Estimate individual large symbols: for i ∈ [ d ] \ L , ˜ x ′ i = x ′ i + z ′ i for z ′ i ∼ Lap ( 0 , 1 ε )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proofs are deferred to Appendix F. The lower bound (7) consists of three terms:

1. Cost of privacy for large symbols p i ≥ 1 nε is 1 n 2 ε 2 p i , which is significantly smaller than the corresponding non-DP lower bound Ω( 1 n ) in (5) when p i is large enough, i.e., privacy is free for sufficiently large p i .
2. Cost of privacy for combined mass estimate of small symbols p i ≤ t nε is ln ( 1+ d small ( L ′ ) ) nε , which is non-zero whenever there exist small symbols in L ′ .
3. Cost of privacy for small symbols p i &lt; t nε is p small ( L ′ ) · ln ( 1 + d small ( L ′ ) nε · p small ( L ′ ) ) + ∑ i&lt; 1 nε p i , which is larger than the corresponding non-DP lower bound in (5) for small symbols whenever ε ≪ 1 , i.e., in the high privacy regime.

## 3.1 Privatized Instance-Optimal 'Sampling Twice' Estimator

We then propose a privatized variant of 'sampling-twice' Algorithm 2 by applying the Laplace Mechanism [26]. The ε -DP guarantee follows from standard results in DP by observing that the ℓ 1 -sensitivity of ( x 1 , · · · , x d , x ′ 1 , · · · , x ′ d ) is one. The main novel ingredient (as discussed in Section 1.1) is to choose a calibrated threshold for privately selecting a set of small symbols. This selected set is then used for estimating the combined mass of small symbols on fresh samples. An accurate combined mass estimate is the key to reducing KL error.

Instance-optimality Guarantee In Theorem G.6, we prove a per-instance upper bound for Algorithm 2. In Corollary G.7, we further prove that this upper bound can be controlled by the per-instance lower bound under the combination of our Non-DP addditive neighborhood N + ( p ) in (4) and our ( ε, δ ) -DP calibrated neighborhood in (6). Specifically, under neighborhood size t = 24ln d , we prove that E x ∼ Poi ( np ) [ KL ( p, A ( x ))] ≤ O ( lower ( p, n, N + ∪ N ≤ t nε )) .

Discussions on the Neighborhood Size We have established instance-optimality under local neighborhood N 24 ln d nε . It is a natural question to ask whether the size of such neighborhoods could be further reduced to enable stronger instance-optimality notions. In Theorem G.9, we prove that this neighborhood size is necessary up to constants-no DP estimator can be instance-optimal for a neighborhood N ≤ γ · ln d nε with γ ≤ o (1) .

Discussions on Approximate DP variants Our per-instance lower bounds hold for ( ε, δ ) -approxDP with δ &lt; ε , and match the upper bound achieved by pure-DP algorithms. This indicates that our pure-DP algorithm, while enjoying better privacy guarantees, is also instance-optimal under approximate DP. Similar phenomena have previously been observed for the histogram estimation

(

)

problem under approx-DP, where in the asymptotic sense, approx DP (e.g., Gaussian mechanism) offers little advantage over pure DP (e.g., Laplace).

In the practical sense, however, the Gaussian mechanism may be practically better in some regime of parameters (due to the lighter distribution tails compared to Laplace mechanism). And our algorithm (after substituting Laplace mechanism with Gaussian mechanism and changing threshold to calibrate to delta) achieves instance optimality up to a log(1 /δ ) factor (see the Gaussian variants of our algorithm and its instance-optimality proof in Appendix G.5).

## 4 Experiments

We now evaluate the performance of our algorithms and compare them with baselines. All experiments are performed on a MacOS intergrated CPU ( ≤ 30 minutes) with 18GB RAM. All reported performance numbers are averaged across five random trials of data sampling and estimators.

Figure 1: (Reddit Token Distribution Estimation) KL error versus dataset size n , distribution dimension d , and DP guarantee ε for our methods compared with the simple minimax optimal Add-constant (DP) baseline, and the strongest non-DP baseline of prior (near) instance-optimal Good-Turing estimator.

<!-- image -->

Figure 2: (Power law distribution p i ∝ 1 i ) KL error versus dataset size n , distribution dimension d , and DP guarantee ε for our methods compared with the simple minimax optimal Add-constant (DP) baseline, and the strongest non-DP baseline of prior (near) instance-optimal Good-Turing estimator.

<!-- image -->

Datasets We evaluate on both synthetic and real-world data distributions. For synthetic data, we evaluate power law distributions p i ∝ 1 i β over a range of parameters β &gt; 0 . We choose power law because of its practical relevance: word frequencies from a text corpus have long been observed to roughly follow power law distributions [54, 53, 48, 17]. For real-word data, we experiment on randomly drawn tokens from Reddit [60], Enron-email [41] and MMLU [36, 35], where each token is one (sensitive) record. We chose Reddit and Enron-email datasets because they are user-specific and thus bear a natural notion of privacy risk (compared to e.g., wikipedia), and are standard and widely used text datasets in the private learning literature [58, 43, 45]. We additionally evaluate on MMLUto simulate diverse text domains. To vary the distribution dimension (specified by the number of all possible tokens), we use two types of tokenizers (GPT2 and Bert).

One challenge for evaluating real-word dataset is the unknown ground-truth distribution p (as only empirical samples are given). To address this, we independently sample two equal-size datasets x

and x ′ , thus ensuring E [ x ′ i / ∥ x ′ ∥ 1 ] = p i for any i ∈ [ d ] . We then compute that

<!-- formula-not-decoded -->

Observe that the entropy term is independent of the estimator. Thus in experiments we only report the negative log likelihood ratio term to compare different estimators A .

Hyperparameters Although our proposed Algorithm 1 and 2 are provably instance-optimal, they separately use two disjoint fractions of the dataset. Thus in the extreme scenarios when all symbols are above or below threshold (e.g., when n, d are exceedingly large or small), Algorithm 1 and 2 incur twice as much noise compared to the add-one estimator, incurring a suboptimal multiplicative constant of two. To address this limitation, we perform grid search for the optimal hyperparameters over α ∈ { 0 . 01 , 0 . 1 , 0 . 25 , 0 . 5 , 0 . 75 , 0 . 9 , 0 . 99 } and τ ∈ { 0 , 0 . 0625 , 0 . 125 , 0 . 25 , 0 . 5 , 1 , 2 , 4 } × ln d , and use the tuned hyperparameters τ = 0 , α = 0 . 5 for Algorithm 1, and τ = min { 1 ε , 1 . 0 } × ln d, α = 0 . 9 for Algorithm 2 in all experiments.

Observations We show results on power law distributions p i ∝ 1 i β for β = 1 , 1 . 5 , 2 (Figure 2, 3, and 4), Reddit corpus (Figure 1), Enron-email corpus(Figure 5); and MMLU corpus (Figure 6). In all experiments, our DP instance-optimal Algorithm 2 consistently outperforms the simple minimax optimal Add-constant (DP) baseline, and our non-DP instance-optimal Algorithm 1 is consistently competitive (within constants) to the strongest prior non-DP baseline of (near) instance-optimal Good-Turing [50]. The gain of our algorithms are especially significant for real-world datasets, validating the effectiveness of our algorithms. We also remark that instance optimality means that our algorithm will provably adapt to any input distribution, and no other algorithm can be significantly better (across a neighborhood).

## 5 Conclusion

We provide tight instance-optimality analysis for private KL distribution estimation, in terms of achieving provably competitive error to the best possible estimator in a small additive local neighborhood of each instance. Furthermore, our constructed neighborhood's size is necessary up to constants for instance-optimality on the worst-case instances. Additionally, we proved instance-optimality up to constants, and leave open the question of whether exact instance-optimality is achievable. Such results, if possible, would require improving the constants in per-instance privacy lower bounds and/or designing better estimators.

## Acknowledgements

The authors thank Audra McMillan and Satchit Sivakumar for insightful discussions on the literature, and Daogao Liu for valuable feedback on earlier drafts. Jiayuan Ye is supported by the Apple Scholars in AI/ML PhD fellowship.

## References

- [1] Jayadev Acharya and Ziteng Sun. Communication complexity in locally private distribution estimation and heavy hitters. In International Conference on Machine Learning , pages 51-60. PMLR, 2019.
- [2] Jayadev Acharya, Ashkan Jafarpour, Alon Orlitsky, and Ananda Theertha Suresh. Optimal probability estimation with applications to prediction and classification. In Conference on Learning Theory , pages 764-796. PMLR, 2013.
- [3] Jayadev Acharya, Ziteng Sun, and Huanyu Zhang. Differentially private assouad, fano, and le cam. In Algorithmic Learning Theory , pages 48-78. PMLR, 2021.

- [4] Alan Agresti and Brent A Coull. Approximate is better than 'exact' for interval estimation of binomial proportions. The American Statistician , 52(2):119-126, 1998.
- [5] David Aldous. Random walks on finite groups and rapidly mixing markov chains. In Séminaire de Probabilités XVII 1981/82: Proceedings , pages 243-297. Springer, 1983.
- [6] Hilal Asi and John C Duchi. Instance-optimality in differential privacy via approximate inverse sensitivity mechanisms. Advances in neural information processing systems , 33:14106-14117, 2020.
- [7] Hilal Asi and John C Duchi. Near instance-optimality in differential privacy. arXiv preprint arXiv:2005.10630 , 2020.
- [8] Hilal Asi, John C Duchi, Saminul Haque, Zewei Li, and Feng Ruan. Universally instanceoptimal mechanisms for private statistical estimation. In The Thirty Seventh Annual Conference on Learning Theory , pages 221-259. PMLR, 2024.
- [9] Raef Bassily and Adam Smith. Local, private, efficient protocols for succinct histograms. In Proceedings of the forty-seventh annual ACM symposium on Theory of computing , pages 127-135, 2015.
- [10] Dietrich Braess and Thomas Sauer. Bernstein polynomials and learning theory. Journal of Approximation Theory , 128(2):187-206, 2004.
- [11] Dietrich Braess, Jürgen Forster, Tomas Sauer, and Hans U Simon. How to achieve minimax expected kullback-leibler distance from an unknown finite distribution. In International Conference on Algorithmic Learning Theory , pages 380-394. Springer, 2002.
- [12] Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security 21) , pages 2633-2650, 2021.
- [13] Stanley F Chen and Joshua Goodman. An empirical study of smoothing techniques for language modeling. Computer Speech &amp; Language , 13(4):359-394, 1999.
- [14] Stanley F Chen, Douglas Beeferman, and Roni Rosenfeld. Evaluation metrics for language models. 1998.
- [15] Albert Cheu, Adam Smith, Jonathan Ullman, David Zeber, and Maxim Zhilyaev. Distributed differential privacy via shuffling. In Advances in Cryptology-EUROCRYPT 2019: 38th Annual International Conference on the Theory and Applications of Cryptographic Techniques, Darmstadt, Germany, May 19-23, 2019, Proceedings, Part I 38 , pages 375-403. Springer, 2019.
- [16] Charles J Clopper and Egon S Pearson. The use of confidence or fiducial limits illustrated in the case of the binomial. Biometrika , 26(4):404-413, 1934.
- [17] Brian Conrad and Michael Mitzenmacher. Power laws for monkeys typing randomly: the case of unequal probabilities. IEEE Transactions on information theory , 50(7):1403-1414, 2004.
- [18] Thomas Cover. Broadcast channels. IEEE Transactions on Information Theory , 18(1):2-14, 1972.
- [19] Ilias Diakonikolas, Moritz Hardt, and Ludwig Schmidt. Differentially private learning of structured discrete distributions. Advances in Neural Information Processing Systems , 28, 2015.
- [20] Irit Dinur and Kobbi Nissim. Revealing information while preserving privacy. In Proceedings of the twenty-second ACM SIGMOD-SIGACT-SIGART symposium on Principles of database systems , pages 202-210, 2003.
- [21] David L Donoho and Iain M Johnstone. Ideal spatial adaptation by wavelet shrinkage. biometrika , 81(3):425-455, 1994.
- [22] Evgeny Drukh and Yishay Mansour. Concentration bounds for unigram language models. Journal of Machine Learning Research , 6(8), 2005.

- [23] John C Duchi and Feng Ruan. The right complexity measure in locally private estimation: It is not the fisher information. The Annals of Statistics , 52(1):1-51, 2024.
- [24] John C Duchi, Michael I Jordan, and Martin J Wainwright. Local privacy, data processing inequalities, and minimax rates. arXiv preprint arXiv:1302.3203 , 2013.
- [25] John C Duchi, Michael I Jordan, and Martin J Wainwright. Minimax optimal procedures for locally private estimation. Journal of the American Statistical Association , 113(521):182-201, 2018.
- [26] Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography: Third Theory of Cryptography Conference, TCC 2006, New York, NY, USA, March 4-7, 2006. Proceedings 3 , pages 265-284. Springer, 2006.
- [27] Cynthia Dwork, Aaron Roth, et al. The algorithmic foundations of differential privacy. Foundations and Trends® in Theoretical Computer Science , 9(3-4):211-407, 2014.
- [28] Cynthia Dwork, Adam Smith, Thomas Steinke, Jonathan Ullman, and Salil Vadhan. Robust traceability from trace amounts. In 2015 IEEE 56th Annual Symposium on Foundations of Computer Science , pages 650-669. IEEE, 2015.
- [29] Úlfar Erlingsson, Vitaly Feldman, Ilya Mironov, Ananth Raghunathan, Kunal Talwar, and Abhradeep Thakurta. Amplification by shuffling: From local to central differential privacy via anonymity. In Proceedings of the Thirtieth Annual ACM-SIAM Symposium on Discrete Algorithms , pages 2468-2479. SIAM, 2019.
- [30] Vitaly Feldman, Audra McMillan, Satchit Sivakumar, and Kunal Talwar. Instance-optimal private density estimation in the wasserstein distance. arXiv preprint arXiv:2406.19566 , 2024.
- [31] William A Gale and Geoffrey Sampson. Good-turing frequency estimation without tears. Journal of quantitative linguistics , 2(3):217-237, 1995.
- [32] Badih Ghazi, Noah Golowich, Ravi Kumar, Rasmus Pagh, and Ameya Velingker. On the power of multiple anonymous messages: Frequency estimation and selection in the shuffle model of differential privacy. In Annual International Conference on the Theory and Applications of Cryptographic Techniques , pages 463-488. Springer, 2021.
- [33] Irving J Good. The population frequencies of species and the estimation of population parameters. Biometrika , 40(3-4):237-264, 1953.
- [34] Michael Hay, Vibhor Rastogi, Gerome Miklau, and Dan Suciu. Boosting the accuracy of differentially-private histograms through consistency. arXiv preprint arXiv:0904.0942 , 2009.
- [35] Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob Steinhardt. Aligning ai with shared human values. Proceedings of the International Conference on Learning Representations (ICLR) , 2021.
- [36] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR) , 2021.
- [37] Nils Homer, Szabolcs Szelinger, Margot Redman, David Duggan, Waibhav Tembe, Jill Muehling, John V Pearson, Dietrich A Stephan, Stanley F Nelson, and David W Craig. Resolving individuals contributing trace amounts of dna to highly complex mixtures using high-density snp genotyping microarrays. PLoS genetics , 4(8):e1000167, 2008.
- [38] Peter Kairouz, Keith Bonawitz, and Daniel Ramage. Discrete distribution estimation under local privacy. In International Conference on Machine Learning , pages 2436-2444. PMLR, 2016.
- [39] Sudeep Kamath, Alon Orlitsky, Dheeraj Pichapati, and Ananda Theertha Suresh. On learning distributions from their samples. In Conference on Learning Theory , pages 1066-1100. PMLR, 2015.

- [40] Slava Katz. Estimation of probabilities from sparse data for the language model component of a speech recognizer. IEEE transactions on acoustics, speech, and signal processing , 35(3): 400-401, 1987.
- [41] Bryan Klimt and Yiming Yang. The enron corpus: A new dataset for email classification research. In Jean-François Boulicaut, Floriana Esposito, Fosca Giannotti, and Dino Pedreschi, editors, Machine Learning: ECML 2004 , pages 217-226, Berlin, Heidelberg, 2004. Springer Berlin Heidelberg. ISBN 978-3-540-30115-8.
- [42] Raphail Krichevsky and Victor Trofimov. The performance of universal encoding. IEEE Transactions on Information Theory , 27(2):199-207, 1981.
- [43] Xuechen Li, Florian Tramer, Percy Liang, and Tatsunori Hashimoto. Large language models can be strong differentially private learners. arXiv preprint arXiv:2110.05679 , 2021.
- [44] Yuhan Liu, Ananda Theertha Suresh, Felix Xinnan X Yu, Sanjiv Kumar, and Michael Riley. Learning discrete distributions: user vs item-level privacy. Advances in Neural Information Processing Systems , 33:20965-20976, 2020.
- [45] Nils Lukas, Ahmed Salem, Robert Sim, Shruti Tople, Lukas Wutschitz, and Santiago ZanellaBéguelin. Analyzing leakage of personally identifiable information in language models. In 2023 IEEE Symposium on Security and Privacy (SP) , pages 346-363. IEEE, 2023.
- [46] David A McAllester and Robert E Schapire. On the convergence rate of good-turing estimators. In COLT , pages 1-6, 2000.
- [47] Audra McMillan, Adam Smith, and Jon Ullman. Instance-optimal differentially private estimation. arXiv preprint arXiv:2210.15819 , 2022.
- [48] Michael Mitzenmacher. A brief history of generative models for power law and lognormal distributions. Internet mathematics , 1(2):226-251, 2004.
- [49] Alon Orlitsky and Ananda Theertha Suresh. Competitive distribution estimation: Why is good-turing good. Advances in Neural Information Processing Systems , 28, 2015.
- [50] Alon Orlitsky, Narayana P Santhanam, and Junan Zhang. Always good turing: Asymptotically optimal probability estimation. Science , 302(5644):427-431, 2003.
- [51] Liam Paninski. Estimation of entropy and mutual information. Neural computation , 15(6): 1191-1253, 2003.
- [52] Liam Paninski. Variational minimax estimation of discrete distributions under kl loss. Advances in Neural Information Processing Systems , 17, 2004.
- [53] Steven T Piantadosi. Zipf's word frequency law in natural language: A critical review and future directions. Psychonomic bulletin &amp; review , 21:1112-1130, 2014.
- [54] David MW Powers. Applications and explanations of zipf's law. In New methods in language processing and computational natural language learning , 1998.
- [55] Sriram Sankararaman, Guillaume Obozinski, Michael I Jordan, and Eran Halperin. Genomic privacy and limits of individual detection in a pool. Nature genetics , 41(9):965-967, 2009.
- [56] Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In 2017 IEEE symposium on security and privacy (SP) , pages 3-18. IEEE, 2017.
- [57] Michael Short. Improved inequalities for the poisson and binomial distribution and upper tail quantile functions. International Scholarly Research Notices , 2013, 2013.
- [58] Congzheng Song, Thomas Ristenpart, and Vitaly Shmatikov. Machine learning models that remember too much. In Proceedings of the 2017 ACM SIGSAC Conference on computer and communications security , pages 587-601, 2017.

- [59] Gregory Valiant and Paul Valiant. An automatic inequality prover and instance optimal identity testing. SIAM Journal on Computing , 46(1):429-455, 2017.
- [60] Michael V"olske, Martin Potthast, Shahbaz Syed, and Benno Stein. TL;DR: Mining Reddit to learn automatic summarization. In Proceedings of the Workshop on New Frontiers in Summarization , pages 59-63, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/W17-4508. URL https://www.aclweb.org/anthology/ W17-4508 .
- [61] Tianhao Wang, Jeremiah Blocki, Ninghui Li, and Somesh Jha. Locally differentially private protocols for frequency estimation. In 26th USENIX Security Symposium (USENIX Security 17) , pages 729-745, 2017.
- [62] Qun Xie and Andrew R Barron. Minimax redundancy for the class of memoryless sources. IEEE Transactions on Information Theory , 43(2):646-657, 1997.
- [63] Jia Xu, Zhenjie Zhang, Xiaokui Xiao, Yin Yang, Ge Yu, and Marianne Winslett. Differentially private histogram publication. The VLDB journal , 22:797-822, 2013.
- [64] Min Ye and Alexander Barg. Optimal schemes for discrete distribution estimation under locally differential privacy. IEEE Transactions on Information Theory , 64(8):5662-5676, 2018.
- [65] Bin Yu. Assouad, fano, and le cam. In Festschrift for Lucien Le Cam: research papers in probability and statistics , pages 423-435. Springer, 1997.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: All claims in the abstract are elaborated in the introduction, and all claims in the introduction are followed by references to their discussed sections.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: See the conclusion Section 5.

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

Justification: See the statements in Section 2 and 3.1 and proofs in the appendices.

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

Justification: Algorithm pseudocodes are given in Algorithm 1 and 2, and hyperparameters are given in Section 4.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often

one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [No]

Justification: The paper does not provide access to the code, but only uses opensource datasets in Section 4, and provides algorithm pseudocode in Algorithm 1 and 2.

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

Justification: Hyperparameters are given in Section 4

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All our experiments (Figure 2, 1 and Appendix G.4) report 1-sigma error bars across five runs.

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

Justification: See Section 4.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We reviewed the NeurIPS Code of Ethics and ensured it is respected.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: We do not see potential such risks of this paper, as we only used synthetic and open-source datasets for validation.

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

Justification: We do not foresee any no potential ethical harms, as the paper only uses synthetic and open-source datasets for validation.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All used datasets are referenced in Section 4, and have licenses that allow research use.

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

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: LLMs are only used for checking grammar and formatting.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Contents

| 1 Introduction   | 1 Introduction                                         | 1 Introduction                                               | 1   |
|------------------|--------------------------------------------------------|--------------------------------------------------------------|-----|
|                  | 1.1 . .                                                | Technical Contributions . . . . . . . . . . . . . . . . . .  | 4   |
|                  | 1.2 . . .                                              | Other Related Works . . . . . . . . . . . . . . . . . .      | 4   |
| 2                | Tighter Instance-Optimality for Non-DP Estimation      |                                                              | 5   |
|                  | 2.1                                                    | An Instance-Optimal 'Sampling Twice' Estimator . . . . . .   | 6   |
|                  | 2.2                                                    | Comparison to Prior Instance-Optimality Results . . . . . .  | 6   |
| 3                | DP                                                     | Instance-Optimality                                          | 7   |
|                  | 3.1                                                    | Privatized Instance-Optimal 'Sampling Twice' Estimator . .   | 8   |
| 4                | Experiments                                            |                                                              | 9   |
| 5                | Conclusion                                             |                                                              | 10  |
| A                | Tools for Lower Bounds                                 | Tools for Lower Bounds                                       | 22  |
| B                | Useful Lemmas for Poisson and Laplace Random Variables | Useful Lemmas for Poisson and Laplace Random Variables       | 26  |
| C                | Deferred Minimax Optimality Results                    | Deferred Minimax Optimality Results                          | 30  |
|                  | C.1                                                    | Recap: Non-DP Minimax Results . . . . . . . . . . . . . .    | 30  |
|                  |                                                        | C.1.1 Recap: Non-DP Minimax KL Error Lower Bound .           | 30  |
|                  |                                                        | C.1.2 Recap: Non-DP Minimax KL Error Upper Bound . .         | 31  |
|                  | C.2 Our .                                              | DP Minimax Results . . . . . . . . . . . . . . . . . .       | 33  |
|                  | C.2.1                                                  | DP Minimax Lower Bound . . . . . . . . . . . . . .           | 33  |
|                  | C.2.2                                                  | DP Minimax Upper Bound . . . . . . . . . . . . . .           | 34  |
| D                | D.1 Useful                                             | )                                                            | 36  |
|                  |                                                        | Non-DP t                                                     |     |
|                  | D.2                                                    | Per-Instance Lower Bound under N ≤ n ( p . . . . .           |     |
|                  | D.3                                                    | Non-DP Per-Instance Lower Bound under N stat ( p ) . . . . . | 39  |
| E                | Deferred Proofs for Non-DP Per-Instance Upper          | Bounds                                                       | 41  |
|                  | E.2                                                    | Proof for Matching Lower and Upper Bound . . . . . . . . .   | 42  |
| F                | Deferred Proofs for DP Per-Instance Lower Bound        | Theorem 3.1                                                  | 43  |
|                  | F.1                                                    | DP Lower Bound: N 1 nε neighborhood . . . . . . . . . . . .  | 43  |
|                  | F.2                                                    | DP Lower Bound: N t nε neighborhood . . . . . . . . . . . .  | 44  |
|                  | G Deferred Proofs for DP Per-Instance Upper            | Bounds                                                       | 47  |
|                  | G.1 Upper Bound: DP Sampling Twice Algorithm           | . . . . . . . .                                              | 47  |

| G.2   | Proof for Matching Lower and Upper Bound . . . . . . . . . . . . . . . . . . . . . 53   |
|-------|-----------------------------------------------------------------------------------------|
| G.3   | Discussions on the Neighborhood Size . . . . . . . . . . . . . . . . . . . . . . . . 54 |
| G.4   | Additional Experiments on More Datasets . . . . . . . . . . . . . . . . . . . . . . 56  |
| G.5   | Instance Optimality of Gaussian Variant of Our Algorithm . . . . . . . . . . . . . 56   |

## A Tools for Lower Bounds

Below we first define decomposable statistical distance.

Definition A.1 (Decomposable Statistical Distance) . Let d ∈ N and let dist : ∆( d ) × ∆( d ) → R be a non-negative function such that dist ( p, q ) = 0 if and only if p = q . We say dist is decomposable, if for any disjoint sets of symbols B 1 , · · · , B k ⊆ [ d ] , and for any distributions p, q ∈ ∆( d ) , it is the case that

<!-- formula-not-decoded -->

where for any distribution p ∈ ∆( d ) , and any i = 1 , · · · , k , we have denoted

<!-- formula-not-decoded -->

Lemma A.2 (KL divergence is decomposable) . KL divergence is decomposable, that is, for any disjoint sets of symbols B 1 , · · · , B k and any p, q ∈ ∆( d ) , it is the case that

<!-- formula-not-decoded -->

Proof. Let p, q ∈ ∆( d ) . Denote B c = [ d ] \ ( ∪ k i =1 B i ) , then by definition we compute

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality is by non-negativity of KL divergence KL ( p | B c , q | B c ) and ∑ k i =1 ( ∑ j ∈B i p j ) · ln ( ∑ j ∈B i p j ∑ j ∈B i q j ) + ( ∑ j ∈B c p j ) · ln ( ∑ j ∈B c p j ∑ j ∈B c q j ) .

Theorem A.3 (Generalized Assouad's method for decomposable statistical distance) . Let dist be a decomposable statistic distance as per Definition A.1. Let B 1 , · · · , B k ⊆ [ d ] be k disjoint sets of symbols. For each i = 1 , · · · , k , let P i be a set of distributions supported on B i . For fixed w 1 , · · · , w k ≥ 0 such that ∑ k i =1 w i = 1 , let P be the following composed packing set:

<!-- formula-not-decoded -->

If the following two conditions hold:

1. There exists a non-negative function f such that

<!-- formula-not-decoded -->

̸

2. For each i ∈ [ k ] , there exists τ i ≥ 0 and ¯ p i ∈ P i , such that for any fixed p j ∈ P j , j = i , it holds that

̸

<!-- formula-not-decoded -->

̸

where we have denoted S ( n, p ) as the distribution of histogram representation of dataset sampled from distribution p ∈ ∆( d ) with target sample size n .

Then for any fixed ε &gt; 0 , δ ≤ ε ,

<!-- formula-not-decoded -->

Proof. We will reduce the estimation problem over all d symbols to the estimation problem over each bucket B i . For any distribution p ∈ ∆( d ) , denote its conditional distribution on B i as

<!-- formula-not-decoded -->

And for any ( ε, δ ) -DP estimator A given dataset x supported on B , similarly denote

<!-- formula-not-decoded -->

Then by definition, for any fixed i = 1 , · · · , k , we have

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

̸

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

where (20) is by Markov inequality, (22) is by definition and by moving the sum of p i ∈ P i inside expectation (as ¯ x and min p i ∈P i d S ( n, ∑ k j =1 w j p j ) d S ( n,w i ¯ p i + ∑ j = i w j p j ) (¯ x ) are independent of the choice of p i ), (23) is by the packing assumption (23), and (24) is by the dataset distance assumption (16).

We now use (23) to prove lower bound on the expected KL error for estimating the whole distribution over all possible distributions in the packing set P . By Lemma A.2, it follows that

<!-- formula-not-decoded -->

where the last inequality (26) is by (23). This suffice to prove (17) by observing that average is smaller than maximum.

Theorem A.4 (Generalized DP Assouad's method for decomposable statistical distance) . Let dist be a decomposable statistic distance as per Definition A.1. Let B 1 , · · · , B k ⊆ [ d ] be k disjoint sets of symbols. For each i = 1 , · · · , k , let P i be a set of distributions supported on B i . For fixed w 1 , · · · , w k ≥ 0 such that ∑ k i =1 w i = 1 , let P be the following composed packing set:

<!-- formula-not-decoded -->

For any p ∈ ∆( d ) , denote S ( n, p ) as the distribution of histogram representation of dataset sampled from p with target sample size n . If the following two conditions hold:

1. There exists a non-negative function f such that

<!-- formula-not-decoded -->

2. For each i ∈ [ k ] , there exists τ i ≥ 0 and ¯ p i ∈ P i , such that for any fixed p j ∈ P j , j = 1 , · · · , k , it holds that

<!-- formula-not-decoded -->

̸

for a coupling ( x, ¯ x ) between distributions S ( n, ∑ k j =1 w j p j ) and S ( n, w i ¯ p i + ∑ j = i w j p j ) .

Then for any fixed ε &gt; 0 , δ ≤ ε ,

<!-- formula-not-decoded -->

Proof. We will reduce the estimation problem over all d symbols to the estimation problem over each bucket B i . For any distribution p ∈ ∆( d ) , we denote p | i as its conditional distribution on B i as defined in (18). And for any ( ε, δ ) -DP estimator A given dataset x supported on B , we denote A ( x ) | i as the conditional estimate on B i as defined in (19). Then by definition, for any fixed i = 1 , · · · , k , we have

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

where (31) is by Markov inequality, (33) is by group privacy and holds for arbitrary constant C &gt; 0 , (35) is by moving the sum of p i ∈ P i inside expectation (as ¯ x is independent of the choice of p i given coupling ( x, ¯ x ) between distributions S ( n, ∑ d j =1 w j p j ) and S ( n, w i ¯ p i + ∑ j = i w j p j ) ), (36) is by the packing assumption (28) and δ ≤ ε and the dataset distance assumption (29), and (37) is by choosing C = 1 4 ε · τ i .

We now use (37) to prove a lower bound on the expected error for estimating the whole distribution over all possible distributions in the packing set P . By Lemma A.2, it follows that

<!-- formula-not-decoded -->

where the last inequality (39) is by (37). This suffice to prove (30) by observing that average is smaller than maximum.

Finally, we provide two useful constructions of packing set that satisfy (15) and (28).

Lemma A.5 (Packing over Two Symbols) . Let B = { j 1 , j 2 } be a set of two symbols. Given 0 ≤ ∆ ≤ a ≤ 1 , let P := { p, p -} be a packing set containing the following two distributions on B .

<!-- formula-not-decoded -->

If ∆ &lt;

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We will prove this claim by separating the analysis for different q .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (41) is by ln(1 + x ) ≤ x -x 2 2 for x ≤ 0 , and by ln(1 + x ) ≤ x for x ≥ 0 , and the last inequality is by q j 1 + q j 2 ≤ 1 and by using the condition that q j 1 ≤ a -∆ 2 .

2. If q j 1 &gt; a -∆ 2 : by definition, we compute that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (43) is by ln(1 + x ) ≤ x -x 2 4 for 0 ≤ x &lt; 1 2 , and by ln(1 + x ) ≤ x for x ≥ 0 , and the last inequality is by q j 1 + q j 2 ≤ 1 and by using the condition that q j 1 &gt; a -∆ 2 .

Lemma A.6 (Dirac Distribution Packing over κ Symbols) . Let B = { j 1 , · · · , j κ } be a set of κ ≥ 2 symbols. Let P := { p, p -} be a packing set containing the following dirac distributions on B .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We prove by contradiction, suppose that there exists q ∈ ∆( d ) , such that

<!-- formula-not-decoded -->

Then by definition of KL divergence, we have

<!-- formula-not-decoded -->

Observe that ∑ κ l =1 1 q j l ≤ 1 1+ κ 4 is integer, (47) implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus

<!-- formula-not-decoded -->

where (50) is by (47). This contradicts q ∈ ∆( d ) .

## B Useful Lemmas for Poisson and Laplace Random Variables

Lemma B.1 (Expectation of inverse Poisson random variable) . Let p ∈ [0 , 1] , m ∈ N , and let x ∼ Poi ( mp ) . Then

<!-- formula-not-decoded -->

Then it holds that

Proof. By definition, we compute that

<!-- formula-not-decoded -->

Lemma B.2 (Tail bound for Sum of Poisson and Laplace Random Variables) . For any a, b, c &gt; 0 . Suppose that x ∼ Poi ( a ) , z ∼ Lap (0 , b ) , then we have that

<!-- formula-not-decoded -->

Proof. We first compute the moment generating function of x + z , i.e., the convolution of Poisson random variable and Laplace noise. For any θ ∈ [ -1 b , 1 b ] , by definition, we have that

<!-- formula-not-decoded -->

Then, by using Markov inequality, for θ ∈ [0 , 1 b ) , we have

<!-- formula-not-decoded -->

By choosing θ = 1 2 max { b, 1 } , then by observing e -θ -1 ≤ -2 3 θ for θ = 1 2 max { b, 1 } &lt; 1 2 , we prove that

<!-- formula-not-decoded -->

Similarly, we prove a bound for the upper tail by Markov inequality.

<!-- formula-not-decoded -->

where the last inequality is by choosing θ = 1 2 max { b, 1 }

.

Lemma B.3 (Bias of Truncated Laplace Random Variable) . Let λ ≥ 0 , n ∈ N . Let x ∼ Poi ( λ ) and let Z ∼ Lap (0 , b ) . Then the noisy estimator given by ˜ x = max { x + Z, c } satisfies

<!-- formula-not-decoded -->

Proof. We first prove the left inequality in Lemma B.3. By definition,

<!-- formula-not-decoded -->

We then prove the right inequality in Lemma B.3. By ˜ x = max { x + Z, c } , we compute that

<!-- formula-not-decoded -->

Lemma B.4 (Conditional Bias under Thresholding) . Let X be a random variable over R . Then for any c ∈ R ,

<!-- formula-not-decoded -->

Proof. We first prove the first inequality in Lemma B.4. If c ≥ E [ X ] , then E [ X | x ≥ c ] ≥ c ≥ E [ X ] . If c &lt; E [ X ] , then by definition,

<!-- formula-not-decoded -->

where the inequality is by x -E [ X ] ≤ c -E [ X ] ≤ 0 for x ≤ c . The proof for the second inequality in Lemma B.4 is similar.

Corollary B.5 (Conditional Bias of Truncated Sum of Poisson and Laplace Random Variable) . Let b &gt; 0 , λ &gt; 0 , c, d ∈ R ∪ {-∞ , + ∞} . Let x ∼ Poi ( λ ) and let Z ∼ Lap (0 , b ) . Then the noisy estimator given by ˜ x = max { x + Z, c } satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

Proof. We first prove (62). By Lemma B.4, we have that

<!-- formula-not-decoded -->

where the last inequality is by Lemma B.3. We then prove (63). By Lemma B.4, we have that

<!-- formula-not-decoded -->

where the last inequality is by Lemma B.3.

Lemma B.6. Let b &gt; 0 and c ∈ R , and let Z ∼ Lap (0 , b )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We separate our discussion for c ≥ 0 and c &lt; 0 .

1. If c ≥ 0 , we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

2. If c &lt; 0 , we have that

<!-- formula-not-decoded -->

Thus

<!-- formula-not-decoded -->

Lemma B.7. Let λ &gt; 0 , b &gt; 0 and c ∈ R . Let X ∼ Poi ( λ ) and Z ∼ Lap (0 , b ) be independent random variables. Then

<!-- formula-not-decoded -->

Thus

Proof. By definition,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (71) is by applying Lemma B.6

Lemma B.8 (KL Divergence between Poisson Distributions [57, Theorem 2]) . Let m,k &gt; 0 be fixed. Then the KL divergence between two Poisson distributions Poi ( m ) and Poi ( k ) satisfies

<!-- formula-not-decoded -->

Lemma B.9 (Total Variation Distance between Poisson Distributions) . Let λ 1 , λ 2 &gt; 0 be fixed. If λ 1 ≤ λ 2 , Then

<!-- formula-not-decoded -->

Proof. By Pinsker's inequality, we have

<!-- formula-not-decoded -->

where (76) is by Lemma B.8, and the last inequality is by ln(1 + x ) ≥ x -x 2 2 for x ≥ 0 .

Lemma B.10. Let p &gt; 0 , m ∈ N , c 1 , c 2 ∈ R ∪ { + ∞ , -∞} , b ≥ 0 , and c ≥ max { b, 1 } . Let x ∼ Poi ( mp ) and let Z ∼ Lap (0 , b ) be independent of x . Then the noisy estimator given by ˜ x = max { x + Z, c } satisfies

<!-- formula-not-decoded -->

Proof. We separate the discussions for p ≤ c m and p &gt; c m .

- If p ≤ c , by ln(1 + t ) ≤ t for any t &gt; -1

<!-- formula-not-decoded -->

where the (81) is by ˜ x ≥ c and by ( mp -˜ x ) 2 ≥ 0 , (82) is by ( a + b ) 2 ≤ 2 a 2 +2 b 2 , and the last inequality is by E [( mp -x ) 2 ] = mp , E [ ( x -˜ x ) 2 ] ≤ E [ Z 2 + c 2 ] = 2 b 2 + c 2 , and p ≤ c m .

- c 2 1

<!-- formula-not-decoded -->

where the first inequality in (87) is by ( a + b ) 2 ≤ 2( a 2 + b 2 ) and (˜ x -x ) 2 ≤ Z 2 + c 2 , and by Z 2 + c 2 + ( x -mp ) 2 ≥ 0 ; the last equality in (87) is by E [ Z 2 ] = 1 ε 2 and E [ ( x -mp ) 2 ] = mp .

On the other hand, by Lemma B.2, we have that Pr[ x + Z &lt; 1 3 mp ] ≤ O ( e -mp 3 · 1 max { b, 1 } ) . Thus we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the inequality in (90) is by x ln( x ) e x/ 6 ≤ O ( 1 x ) and x e x/ 6 ≤ O ( 1 x ) for x ≥ 0 , and by c ≥ max { b, 1 } This suffice to prove the bound in the statement.

## C Deferred Minimax Optimality Results

## C.1 Recap: Non-DP Minimax Results

The minimax rate for non-DP distribution estimation in KL divergence error is well-studied [10, 51, 52], as summarized below in Table 2.

Table 2: Non-DP Minimax Rates for Distribution Estimation in KL Divergences

| Estimator/Bound                      | Expected KL Error   | Reference                          |
|--------------------------------------|---------------------|------------------------------------|
| Upper Bound (Add-Constant Estimator) | ln ( 1+ d n )       | [10, Theorem 8] Recap: Theorem C.2 |
| Lower Bound                          | Ω ( ln ( 1+ d n ))  | Recap: Theorem C.1                 |

For completeness, below we offer simple proofs for the minimax upper and lower bounds.

## C.1.1 Recap: Non-DP Minimax KL Error Lower Bound

The minimax lower bound for distribution estimation in KL divergence error is well-understood to be Ω(1 + d n ) , e.g., see the variational lower bounds in [52]. For completeness, below we provide an alternative proof via the tools in this paper, i.e., the generalized DP Assouad's method for decomposable statistical distance (in our case KL divergence) in Lemma A.2.

Theorem C.1 (Lower Bound for Non-DP Estimation) . Let d , n be fixed. Then for any estimator A , we have that

<!-- formula-not-decoded -->

Proof sketch The idea is to design p to be uniform over a random support of symbols with small probability mass, and then reduce the problem to the difficulty of inferring the support given limited samples.

Proof. · Dense Case d -1 ≤ 2 n : Assume d mod 2 = 1 for convenience. We decompose d symbols into d -1 2 buckets B i = { 2 i -1 , 2 i } for i = 1 , · · · , d -1 2 . We then construct the

packing as follows. For i = 1 , · · · , d -1 2 .

<!-- formula-not-decoded -->

where δ i means point distribution on symbol i . We then construct a set of distributions.

<!-- formula-not-decoded -->

Then by Lemma A.2, for any algorithm A we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(97)

Thus for any A , there must exist one p ∈ P such that E x ∼ Mult ( n,p ) [ KL ( p, A ( x ))] ≥ Ω( d -1 n ) .

- Sparse Case: d -1 &gt; 2 n : Denote κ = ⌊ d -1 n ⌋ ≥ 2 and k = n for convenience. We decompose d symbols into k buckets B i = { κ · i -κ +1 , κ · i } for i = 1 , · · · , k . We then construct the packing as follows.

<!-- formula-not-decoded -->

where δ i means point distribution on symbol i . We then construct a set of distributions.

<!-- formula-not-decoded -->

Then by Lemma A.2, for any algorithm A we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(103)

Thus for any A , there must exist one p ∈ P such that E x ∼ Mult ( n,p ) [ KL ( p, A ( x ))] ≥ Ω(ln ( d -1 n ) ) .

## C.1.2 Recap: Non-DP Minimax KL Error Upper Bound

The minimax upper bound for distribution estimation is well-studied, and various works [10, 51, 52] have shown that simple add-constant estimators could achieve the optimal O ( ln ( 1 + d n )) minimax KL error. Below we offer a simple proof for completeness.

<!-- formula-not-decoded -->

Theorem C.2 (Non-DP Distribution Estimation KL Upper Bound) . There exists estimator A such that for any p ∈ ∆( d ) , given n empirical data samples x ∼ Mult ( n, p ) , it satisfies that

<!-- formula-not-decoded -->

Proof. We use that the following simple add-constant estimator.

<!-- formula-not-decoded -->

Then by definition

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (107) is by concavity of the function ln( x ) , and (108) is by Lemma C.3.

Lemma C.3. Let s 1 , · · · , s K be K independent Bernoulli random variables with parameter p 1 , · · · , p K . Let c ≥ 1 be a positive constant. Then we have that

<!-- formula-not-decoded -->

Proof. Conditioned on any fixed value for s 3 , · · · , s K , denote c ′ = c + ∑ K k =3 s k we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality is by 1 c ′ +2 -2 c ′ +1 + 1 c ′ &gt; 0 for c ′ &gt; 0 . By similar argument, we have that

<!-- formula-not-decoded -->

where s ∼ Bin ( K, ¯ p ) where ¯ p = 1 K ∑ K k =1 p k . Then by simple algebraic tricks, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Table 3: DP Minimax KL error bounds

| Method                                           | KL Error Bound                  | Conditions    | Reference   |
|--------------------------------------------------|---------------------------------|---------------|-------------|
| Any ( ε, δ ) -DP                                 | Ω ( ln(1+ d n min { ε, 1 } ) )  | d ≥ 4 , δ ≤ ε | Theorem C.4 |
| Add-constant ( ( ε, δ ) -DP) (Laplace Mechanism) | O ( ln ( 1+ d n min { ε, 1 } )) | ε > 0 , δ ≥ 0 | Theorem C.5 |

## C.2 Our DP Minimax Results

We first summarize our derived DP minimax rates in Table 3 - observe that simple variants of Laplace mechanism is minimax optimal.

We comment that our KL minimax lower bound is stronger than naive conversions of prior TV distance lower bounds to KL lower bounds - applying Pinsker inequality to the prior Ω ( d εn ) TV lower bounds [19, 3] only gives a KL lower bound of Ω ( min { d 2 ε 2 n 2 , 1 }) , which is significantly weaker than our bound Ω ( ln ( 1 + d εn )) in Table 3, especially for large d .

Below we present the proofs for the minimax lower bounds and upper bounds.

## C.2.1 DP Minimax Lower Bound

Theorem C.4 (Minimax Lower Bound for DP estimation) . Let d, n ∈ N , ε ≥ 0 , and δ ∈ [0 , 1] be fixed. If d ≥ 4 , δ ≤ ε , then

<!-- formula-not-decoded -->

Proof. Let κ, k ∈ N be defined as follows.

<!-- formula-not-decoded -->

Thus by d ≥ 2 , we have

<!-- formula-not-decoded -->

For i = 1 , · · · , k , we construct a packing set of distributions supported on symbols B i = { k · i -κ + 1 , · · · , δ κ · i } as follows.

<!-- formula-not-decoded -->

We construct the following set of distributions that lie in the additive neighborhood of p .

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

One can verify that the distributions in P are well-defined (i.e., normalized). Now by applying Lemma A.6 to P i for each i = 1 , · · · , k , we prove that for any q ′ ∈ ∆( d ) , it holds that

<!-- formula-not-decoded -->

Thus the first condition of Theorem A.4 holds. Below we analyze the second condition of Theorem A.4. For each i ∈ [ k ] and fixed q j ∈ P j , j = 1 , · · · , k and fixed ¯ q i ∈ P i . Let x ∼ Poi ( n, ( 1 -k 160 nε ) · q c + k ∑ j =1 w j · q j ) and

̸

<!-- formula-not-decoded -->

By definition, we compute that

<!-- formula-not-decoded -->

where the inequality is by triangle inequality for ℓ 1 distance. By applying Theorem A.4 under our proved conditions (124) and (126), we finally prove that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (128) is by using the definitions of κ and k in (119), (129) is by ⌊ x ⌋ ≥ x 2 for any x ≥ 1 , (130) is by ln(1 + λx ) ≤ λ · ln (1 + x ) ≤ x for 0 ≤ λ ≤ 1 and x &gt; 0 , and by λ ln(1 + x ) ≥ ln(1 + λx ) for λ &gt; 1 and x &gt; 0 .

## C.2.2 DP Minimax Upper Bound

Theorem C.5 (Upper Bound - Pure DP) . There exists an ( ε, δ ) -DP estimator A such that for any fixed d, n ∈ N ,

<!-- formula-not-decoded -->

Proof. This can be achieved by a simple estimator that combines an add-constant estimator and Laplace mechanism as follows.

<!-- formula-not-decoded -->

where z ∼ Lap ( 0 , 1 ε ) d , and N = ∑ i ˜ x i is the normalization constant. The ( ε, 0) -DP guarantee follows by observing that the ℓ 1 -sensitivity of vector release ( x 1 , · · · , x d ) is one, and by applying the DP guarantee for Laplace Mechanism in [26, Theorem 1]. Below we focus on bounding the KL

̸

Table 4: Non-DP per-instance lower bounds, where p small ( L ′ ) = ∑ i ∈ L ′ ,p i ≤ t n p i and d small ( L ′ ) = ∑ i ∈ L ′ ,p i ≤ t n 1 .

| Neighborhood N      | lower ( p,n,N ) (2)                                                           | Conditions                        | Reference   |
|---------------------|-------------------------------------------------------------------------------|-----------------------------------|-------------|
| N ≤ t/n ( p ) (136) | Ω ( ln(1+ d small ( L ′ )) n + p small ( L ′ ) ln ( 1+ d small ( L np small ( | L ′ ⊆ [ d ] , t ≥ 1 d ≥ 2 , n ≥ 4 | Theorem D.3 |
| N stat ( p ) (137)  | Ω ( ∑ i min { p i , 1 n } )                                                   | d ≥ 2                             | Theorem D.4 |

error. By concavity of ln( t ) on t &gt; 0 , we have

<!-- formula-not-decoded -->

where (133) is by p i ln ( p i · n ˜ x i ) ≤ 0 for p i &lt; 1 n min { 1 ,ε } , by E [˜ x i -x i ] ≥ 0 for any i (by Lemma B.3), and by ln(1 + x + y ) ≤ x +ln(1 + y ) for any x, y &gt; 0 ; (134) is by applying Lemma B.10 under setting m = n , c 1 = -∞ , c 2 = + ∞ , b = 1 ε and c = 1 min { 1 ,ε } for p i &gt; 1 n min { 1 ,ε } and by applying Lemma B.3 for p i &lt; 1 n min { 1 ,ε } ; and (135) is by a +ln(1 + b ) ≤ ln(1 + 2 a +2 b ) for 0 &lt; a &lt; 1 and b &gt; 0 .

## D Deferred Proofs for Non-DP Per-Instance Lower Bound Theorem 2.1

For ease of presentation and understanding, we break up the neighborhood N + ( p ) into two subneighborhoods: one N ≤ t n ⊆ N + ( p ) with the same small perturbation for every symbol, and the other N stat ⊆ N + ( p ) (based on statistical variance) with larger perturbations for large symbols, defined as follows.

<!-- formula-not-decoded -->

We then prove per-instance lower bounds under the sub-neighborhoods N ≤ t n ( p ) and N stat ( p ) respectively. By definition (2), their average is a lower bound for the per-instance lower bound under N + ( p ) , i.e., lower ( p, n, N + ) ≥ 1 2 · lower ( p, n, N ≤ t n )+ 1 2 · lower ( p, n, N stat ) . (This is because one can construct a distribution over hard instances, choosing the hard instance(s) in N ≤ t n ( p ) and N stat ( p ) with 1 / 2 probability respectively.) Our per-instance lower bounds under the two sub-neighborhood are summarized in Table 4.

## D.1 Useful Lemmas

Lemma D.1 (Coupling Lemma [5, Lemma 3.6]) . Let random variables Z 1 , Z 2 have distributions ν 1 , ν 2 . Then

̸

<!-- formula-not-decoded -->

Conversely, given probability distributions ν 1 , ν 2 , there exists ( Z 1 , Z 2 ) such that

̸

<!-- formula-not-decoded -->

where Z 1 , Z 2 have distributions ν 1 , ν 2 respectively.

Lemma D.2 (Total Variation Inequality between Product Measures) . Let µ 1 , µ 2 be probability distributions over Ω and let µ ′ 1 , µ ′ 2 be probability distributions over Ω ′ . Denote µ 1 × µ ′ 1 as the product measure of µ 1 and µ ′ 1 over Ω × Ω ′ , and similarly denote µ 2 × µ ′ 2 as the product measure of µ 2 and µ ′ 2 over Ω × Ω ′ . Then

<!-- formula-not-decoded -->

Proof. By the coupling lemma Lemma D.1, there exists ( Z 1 , Z ′ 1 ) such that Z 1 ∼ µ 1 , Z ′ 1 ∼ µ ′ 1 and TV = Pr[ Z 1 = Z ′ 1 ] . Similarly, there exists ( Z 2 , Z ′ 2 ) such that Z 2 ∼ µ 2 , Z ′ 2 ∼ µ ′ 2 and TV = Pr[ Z 2 = Z ′ 2 ] . Let Z = ( Z 1 , Z ′ 1 ) and Z ′ = ( Z 2 , Z ′ 2 ) . Then Z ∼ µ 1 × µ ′ 1 and Z ′ ∼ µ 2 × µ ′ 2 . By again using the coupling lemma Lemma D.1, we prove that

̸

̸

<!-- formula-not-decoded -->

where the second inequality is by union bound.

## D.2 Non-DP Per-Instance Lower Bound under N ≤ t n ( p )

We now prove the per-instance lower bound in Table 4 under additive neighborhood N ≤ t n ( p ) for low-probability symbols.

Theorem D.3. Let p ∈ ∆( d ) be fixed. For t &gt; 0 , let N ≤ t n ( p ) be the below additive local neighborhood of p as defined in (136) .

<!-- formula-not-decoded -->

If t ≥ 1 , d ≥ 2 and n ≥ 4 , then for any L ′ ⊆ [ d ] ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We separate the discussions for different d small ( L ′ ) .

1. If d small ( L ′ ) = 0 , then p small ( L ′ ) = 0 and thus (143) trivially holds.
2. If 1 ≤ d small ( L ′ ) ≤ 3 : Without loss of generality, assume that { i ∈ L ′ : p i ≤ t n } = { 1 , · · · , d small ( L ′ ) } .
3. (a) If max i ∈ [ d ] \ [ d small ( L ′ )] p i ≤ 1 n : Then the neighborhood N stat defined in (137) is a subset of N ≤ t n defined in (142), i.e., N stat ⊆ N ≤ t n . Thus (143) holds by observing that (143) is dominated by the lower bound in Theorem D.4. Specifically, by ln( x ) ≤ x -1 for any x &gt; 0 and by d small ( L ′ ) ≤ 3 , we prove that ln(1+ d small ( L ′ )) n + p small ( L ′ ) ln ( 1 + d small ( L ′ ) np small ( L ′ ) ) ≤ 2 d small ( L ′ ) n ≤ 6 n ≤ O ( ∑ d i =1 min { p i , 1 n } ) where the last inequality is by d ≥ 2 .

̸

̸

̸

- (b) If max i ∈ [ d ] \ [ d small ( L ′ )] p i &gt; 1 n , without loss of generality, assume that p d &gt; 1 n : Then we construct a packing set of distributions P = { p + , p -} that contains the following two distributions.

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

One can validate that the distributions in P are well-defined and are in the additive neighborhood N ≤ t n ( p ) (by observing that p 1 ≤ t n and d ≥ 2 , and thus p 1 d -1 ≤ t n ). By applying Lemma A.5 (under setting a = 1 2 n and ∆ = 1 4 n ), we further prove that for any q ′ ∈ ∆( d ) ,

<!-- formula-not-decoded -->

Thus the first condition of Theorem A.3 holds. Below we analyze the second condition of Theorem A.3. By definition of P we compute that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (147) is by definition for P that only contains two distributions, (148) is by using the definition of total variation distance, (149) is by the total variation inequality for product measures (Lemma D.2), (150) is by Lemma B.9. Thus the second condition of Theorem A.3 holds. By applying Theorem A.3 under our proved conditions (146) and (151), we finally prove that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (153) is by observing that ln(1+ d small ( L ′ )) n + p small ( L ′ ) ln ( 1 + d small ( L ′ ) np small ( L ′ ) ) ≤ 2 d small ( L ′ ) n ≤ 6 n (due to ln( x ) ≤ x -1 for any x &gt; 0 and the condition that d small ( L ′ ) ≤ 3 ).

3. If d small ( L ′ ) ≥ 4 : Without loss of generality, assume that { i ∈ L ′ : p i ≤ t n } = { 1 , · · · , d small ( L ′ ) } . For brevity, denote ˆ d = ⌊ d small ( L ′ ) 2 ⌋ ≥ 2 and ˆ p = ∑ ˆ d i =1 p i . Without

loss of generality, also assume that p 1 ≥ · · · ≥ p d small ( L ′ ) , then

<!-- formula-not-decoded -->

Let κ, k ∈ N be defined as follows.

<!-- formula-not-decoded -->

Thus by definition, we have

<!-- formula-not-decoded -->

For i = 1 , · · · , k , we construct a packing set of distributions supported on symbols B i = { k · i -κ +1 , · · · , δ κ · i } as follows.

<!-- formula-not-decoded -->

We construct the following set of distributions that lie in the additive neighborhood of p .

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

One can verify that the distributions in P are well-defined (i.e., normalized) and lie in the neighborhood N ≤ t n ( p ) defined in (136). This is by p j ≤ t n for j = 1 , · · · , ˆ d , and by 0 ≤ max { ˆ p, 1 n } -k n d small ( L ′ ) -ˆ d ≤ max { ˆ p ˆ d , 1 n } ≤ t n under (156) and ˆ d &lt; d small ( L ′ ) , and by 0 ≥ ˆ p -max { ˆ p, 1 n } 1 -p small ( L ′ ) ≥ -1 /n 1 -3 /n ≥ -1 under (154) and n ≥ 4 , and by observing that 0 ≥ p j · ˆ p -max { ˆ p, 1 n } 1 -p small ( L ′ ) ≥ ˆ p -max { ˆ p, 1 n } ≥ -1 n for any j = d small ( L ′ ) + 1 , · · · , d .

Now by applying Lemma A.6 to P i for each i = 1 , · · · , k , we prove that for any q ′ ∈ ∆( d ) , it holds that

<!-- formula-not-decoded -->

Thus the first condition of Theorem A.3 holds. Below we analyze the second condition of Theorem A.3. By definition of P , for any fixed i ∈ [ k ] and any fixed ¯ p := ( 1 -k n ) · q c + ∑ k i =1 1 n · ¯ q i ∈ P , we compute that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

where (162) is by probability mass function of Poisson distributions Poi ( n · 0) and Poi ( n · 1 e ) . By applying Theorem A.3 under our proved conditions (160) and (162), we finally prove that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (164) is by using the definitions of κ and k in (155), (165) is by ⌊ x ⌋ ≥ x 2 for any x ≥ 1 , (166) is by λ ln(1 + x ) ≥ ln (1 + λx ) for λ ≥ 1 and x &gt; 0 , and by ln(1 + λx ) ≤ λ · ln (1 + x ) ≤ x for 0 ≤ λ ≤ 1 and x &gt; 0 , and the last inequality is by applying (154).

## D.3 Non-DP Per-Instance Lower Bound under N stat ( p )

We now prove the per-instance lower bound in Table 4 under neighborhood N stat ( p ) (137).

Theorem D.4. Let n, d ∈ N and p ∈ ∆( d ) be fixed. Let N stat ( p ) be the following additive local neighborhood of p as defined in (137) .

<!-- formula-not-decoded -->

Then if d ≥ 2 , it holds that

<!-- formula-not-decoded -->

Proof. We will apply Theorem A.3 to prove the lower bound. Below we first construct the packing set. Without loss of generality, assume that p 1 ≥ · · · ≥ p d . For brevity, denote

<!-- formula-not-decoded -->

Let P be the following packing set of distributions.

<!-- formula-not-decoded -->

One can verify that distributions in the packing set P are normalized and lie in the statistical neighborhood N stat ( p ) by observing that min { p i -1 , √ p i -1 n } ≥ min { p i , √ p i n } for any i . By applying Lemma A.5 (under setting a = p 2 k -1 w k and ∆ = ∆ k w k ), we further prove that for any q ∈ ∆( d ) ,

<!-- formula-not-decoded -->

̸

Thus the first condition of Theorem A.3 holds. We now analyze the second condition of Theorem A.3. For any k and any fixed q l ∈ P l , l = k , by definition of P k we compute that

̸

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (174) is because P k only contains two distributions, (175) is by using the definition of total variation distance, (176) is by the total variation inequality for product measures (Lemma D.2), (177) is by Lemma B.9, (178) is by p 2 k -1 ≥ p 2 k and by definition (170) of ∆ k = 1 2 min { p 2 k , √ p 2 k n } ≤ p 2 k 2 . Thus the second condition of Theorem A.3 holds. By applying Theorem A.3 under our proved conditions (172) and (178), we finally prove that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (180) is by observing that for p 1 ≥ ·· · ≥ p d , it is the case that ∑ ⌊ d 2 ⌋ k =1 min { p 2 k , 1 n } ≥ ∑ ⌊ d 2 ⌋ k =1 min { p min { 2 k +1 ,d } , 1 n } . We now separate the discussions for the remaining (at most two) symbols.

1. If ∑ d i =2 min { p i , 1 n } ≥ 1 2 n , then (180) suffice to prove the bound (169) in the statement (by observing that there is only one remaining symbol).
2. If ∑ d i =2 min { p i , 1 n } &lt; 1 2 n , then it must be the case that p 1 &gt; 1 -1 2 n , p 2 &lt; 1 2 n , and ∑ d i =1 min { p i , 1 n } &lt; 3 2 n . By repeating the proof for a new packing P ′ = { ˆ p -, p } where

<!-- formula-not-decoded -->

we similarly prove a new lower bound of 1 256 · 1 n . One can also validate that the new packing P ′ is also in the neighborhood N stat (137) This suffice to prove (169) in the statement.

## E Deferred Proofs for Non-DP Per-Instance Upper Bounds

## E.1 Non-DP Per-Instance Upper Bound (Sampling Twice Algorithm)

We will use the following lemma for bounding the estimation error on zero-count symbols.

Lemma E.1 (Error of Algorithm 1 on zero-count Symbols) . Let p ∈ ∆( d ) be a discrete distribution over d symbols. Let n ∈ N . Then Algorithm 1 satisfies

<!-- formula-not-decoded -->

Proof. By definition, we have that

<!-- formula-not-decoded -->

We first analyze 1 . By concavity of ln( t ) over t &gt; 0 and by ˜ x i ≤ x ′ i +1 , we have that

<!-- formula-not-decoded -->

where the last inequality is by Lemma B.1 under m = n 2 . We then analyze 2 . By concavity of ln( t ) over t &gt; 0 , we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality is by applying Lemma B.3 with b = 0 and c = 1 .

We are now ready to prove the Non-DP per-instance upper bound for the non-dp 'sampling twice' algorithm Algorithm 1.

Theorem E.2 (Per-Instance Upper Bound - Sampling Twice Algorithm) . Let A be the estimator given by Algorithm 1. If d ≥ 2 , then for any n and any p ∈ ∆( d ) ,

<!-- formula-not-decoded -->

Proof. By definition,

<!-- formula-not-decoded -->

where (190) is by ln( t ) ≤ t -1 for any t &gt; 0 and by applying Lemma E.1, (191) is by applying Lemma B.10 under m = n/ 2 , c 1 = -∞ , c 2 = + ∞ , b = 0 and c = 1 , and the last inequality is by Pr[ i / ∈ L ] = 1 -e -np i ≤ min { np i , 1 } , and by ∑ i min { p i , 1 n } ≥ 1 n for d ≥ 2 .

## E.2 Proof for Matching Lower and Upper Bound

Corollary E.3. Let A be the estimator given by Algorithm 1. Let N stat ( p ) and N ≤ t n ( p ) be the additive neighborhoods defined in (137) and (136) respectively. Then for any n and any p ∈ ∆( d ) ,

<!-- formula-not-decoded -->

for any choice of neighborhood size t ≥ 1 such that t · e -t ≤ 1 ln d . Specifically, we can always choose t = min { 1 , 2 ln ln d } .

Proof. We will use the upper bound given by Theorem E.2. Observe that by ∑ j : x j =0 1 ∑ j : x j =0 p j

≤

<!-- formula-not-decoded -->

Additionally, by definition, we compute that

<!-- formula-not-decoded -->

| Neighborhood N   | lower ε,δ ( p,n,N ) (3)                                                                             | Conditions                               | Reference   |
|------------------|-----------------------------------------------------------------------------------------------------|------------------------------------------|-------------|
| N ≤ 1 nε (6)     | Ω ( ∑ i : p i < 1 nε p i + ∑ i : p i ≥ 1 nε 1 p i · 1 n 2 ε 2                                       | δ ≤ ε , d ≥ 2                            | Theorem F.1 |
| N ≤ t nε (6)     | Ω ( ln ( 1+ d small ( L ′ ) ) nε + p small ( L ′ ) · ln ( 1+ d small ( L ′ ) nε · p small ( L ′ ) ) | L ′ ⊆ [ d ] , t ≥ 1 , d ≥ 2 δ ≤ ε,nε ≥ 1 | Theorem F.2 |

<!-- formula-not-decoded -->

where (197) is by L = { i ∈ [ d ] : x i = 0 } and by probability mass function of Poisson random variables, and (198) is by choosing t ≥ 1 such that t · e -t ≤ 1 ln d . Specifically, we can choose t = min { 1 , 2 ln ln d } .

## F Deferred Proofs for DP Per-Instance Lower Bound Theorem 3.1

For ease of presentation and understanding, we break up the neighborhood N ≤ t nε ( p ) for a given small t ≥ 1 into two sub-neighborhoods: one N ≤ 1 n ⊆ N ≤ t nε ( p ) with the same small perturbation for every symbol, and the other N ≤ t nε ( p ) with larger perturbations for small symbols, defined as follows.

We then prove per-instance lower bounds under the sub-neighborhoods N ≤ t nε ( p ) and N ≤ t nε ( p ) respectively. By definition (2), their average is a lower bound for the per-instance lower bound under N ≤ t nε ( p ) , i.e., lower ( p, n, N ≤ t nε ) ≥ 1 2 · lower ( p, n, N ≤ 1 nε ) + 1 2 · lower ( p, n, N ≤ t nε ) . (This is because one can construct a distribution over hard instances, choosing the hard instance(s) in N ≤ t n ( p ) and N stat ( p ) with 1 / 2 probability respectively.)

Our results for DP per-instance lower bounds under neighborhoods N ≤ t nε , t ≥ 1 are summarized in Table 5.

## F.1 DP Lower Bound: N 1 nε neighborhood

In this section, we prove the per-instance DP estimation lower bound in Table 5 under additive neighborhood. We first prove a lower bound under add1 nε neighborhood.

Theorem F.1 (Lower Bound N ≤ 1 nε Neighborhood) . Let d ∈ N , ε ≥ 0 and 0 ≤ δ ≤ 1 , p ∈ ∆( d ) be fixed. Let N ≤ 1 nε ( p ) be the additive neighborhood defined in (6) for t = 1 as follows.

<!-- formula-not-decoded -->

If δ ≤ ε and d ≥ 2 , then the expected KL error of any ( ε, δ ) -DP estimator A satisfies

<!-- formula-not-decoded -->

Proof. We will apply Theorem A.4 to prove the lower bound. Below we first construct the packing set. Without loss of generality, assume that that d mod 2 = 0 . (Otherwise, sort the symbols to satisfy min { p 1 , 1 p 1 · 1 n 2 ε 2 } ≥ · · · ≥ min { p d , 1 p d · 1 n 2 ε 2 } and ignore the last symbol in the below constructions.) For brevity, denote

<!-- formula-not-decoded -->

Let P be the following packing set of distributions.

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

One can verify that distributions in the packing set P are normalized and lie in the additive neighborhood N ≤ 1 nε ( p ) by observing that 1 nε ≥ min { p i -1 , 1 nε } ≥ min { p i , 1 nε } for any i . By applying Lemma A.5 (under setting a = p 2 k -1 w k and ∆ = ∆ k w k ), we further prove that for any q ∈ ∆( d ) ,

<!-- formula-not-decoded -->

Thus the first condition of Theorem A.4 holds. We now analyze the second condition of Theorem A.4. For any k and any fixed q l ∈ P l , l = 1 , · · · , d 2 and fixed ¯ q k ∈ P k . Let x ∼ Poi ( n, ∑ d 2 l =1 w l · q l ) and y ∼ Poi ( n · ∆ k ) and y ′ ∼ Poi ( n · ∆ k ) be independent Poisson random variables. Then we could construct the following coupling ( x, ¯ x ) between distributions Poi ( n, ∑ d 2 l =1 w l · q l ) and

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By definition, we compute that

<!-- formula-not-decoded -->

where the inequality is by definition ∆ k ≤ 1 160 nε . Thus the second condition of Theorem A.4 holds. By applying Theorem A.4 under our proved conditions (203) and (205), we prove that

<!-- formula-not-decoded -->

By repeating the constructions under ∆ k = -1 160 min { p 2 k -1 , 1 nε } for k = 1 , · · · , d 2 , we similarly prove that

<!-- formula-not-decoded -->

By combining (206) and (207), we prove the bound (200) in the statement.

## F.2 DP Lower Bound: N t nε neighborhood

In this section, we prove the per-instance DP estimation lower bound in Table 5 for low probability symbols.

Theorem F.2 (Lower Bound - low probability symbols) . Let p ∈ ∆( d ) be fixed. For any t &gt; 0 , let N t nε be the additive neighborhood defined as follows.

<!-- formula-not-decoded -->

Then if δ ≤ ε , nε ≥ 1 , t ≥ 1 and d ≥ 2 , then for any L ′ ⊆ [ d ] , the expected KL error of any ( ε, δ ) -DP estimator A satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. 1. If d small ( L ′ ) = 0 , then p small ( L ′ ) = 0 and thus (209) trivially holds.

2. If 1 ≤ d small ( L ′ ) ≤ 3 : Without loss of generality, assume that { i ∈ L ′ : p i ≤ t nε } = { 1 , · · · , d small ( L ′ ) } . We construct a packing set of distributions P = { p + , p -} that contains the following two distributions.

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

One can validate that the distributions in P are well-defined and are in the additive neighborhood N ≤ t nε ( p ) (by observing that 0 ≤ p 1 ≤ t nε for d ≥ 2 and thus p 1 d -1 ≤ t nε ). By applying Lemma A.5 (under setting a = 1 80 nε and ∆ = 1 160 nε ), we further prove that for any q ′ ∈ ∆( d ) ,

<!-- formula-not-decoded -->

Thus the first condition of Theorem A.4 holds. Below we analyze the second condition of Theorem A.4. Let ¯ q = p + . For any q ∈ P , let y ∼ Poi ( n min { p + , p -} ) and let y ′ ∼ Poi ( n max { p + , p -} -n min { p + , p -} ) . Then we could construct the following coupling ( x, ¯ x ) between distributions Poi ( n, q ) and Poi ( n, ¯ q ) :

<!-- formula-not-decoded -->

By definition, we compute that

<!-- formula-not-decoded -->

Thus the second condition of Theorem A.4 holds. By applying Theorem A.4 under our proved conditions (212) and (151), we finally prove that

<!-- formula-not-decoded -->

where (216) is by the assumption that d small ( L ′ ) ≤ 3 and by ln (1 + x ) ≤ x for x &gt; 0 .

3. If d small ( L ′ ) ≥ 4 : Without loss of generality, assume that { i ∈ L ′ : p i ≤ t nε } consists of symbols 1 , · · · , d small ( L ′ ) . For brevity, denote ˆ d = ⌊ d small ( L ′ ) 2 ⌋ ≥ 2 and ˆ p = ∑ ˆ d i =1 p i . Without loss of generality, assume that p 1 ≥ · · · ≥ p d small ( L ′ ) , then it follows that

<!-- formula-not-decoded -->

Let κ, k ∈ N be defined as follows.

<!-- formula-not-decoded -->

Thus by definition,

<!-- formula-not-decoded -->

For i = 1 , · · · , k , we construct a packing set of distributions supported on symbols B i = { k · i -κ +1 , · · · , δ κ · i } as follows.

<!-- formula-not-decoded -->

We construct the following set of distributions that lie in the additive neighborhood of p .

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

One can verify that the distributions in P are well-defined (i.e., normalized) and lie in the neighborhood N ≤ t nε ( p ) defined in (6). This is by observing that p j ≤ t nε for j = 1 , · · · , ˆ d , and that 0 ≤ max { ˆ p, 1 160 nε } -k 160 · nε d small ( L ′ ) -ˆ d ≤ max { ˆ p ˆ d , 1 160 nε } due to (219) and ˆ d &lt; d small ( L ′ ) , and by 0 ≥ ˆ p -max { ˆ p, 1 160 nε } 1 -p small ( L ′ ) ≥ -1 160 nε 1 -3 160 nε &gt; -1 under (217) and nε ≥ 1 , and by 0 ≥ p j · ˆ p -max { ˆ p, 1 160 nε } 1 -p small ( L ′ ) ≥ -1 160 nε .

Now by applying Lemma A.6 to P i for each i = 1 , · · · , k , we prove that for any q ′ ∈ ∆( d ) ,

<!-- formula-not-decoded -->

Thus the first condition of Theorem A.4 holds. Below we analyze the second condition of Theorem A.4. For each i ∈ [ k ] and fixed q j ∈ P j , j = 1 , · · · , k and fixed ¯ q i ∈ P i . Let x ∼ Poi ( n, ( 1 -k 160 nε ) · q c + k ∑ j =1 w j · q j ) and

̸

<!-- formula-not-decoded -->

random variables. Then we could construct the following coupling ( x, ¯ x ) between distribution Poi ( n, ( 1 -k 160 nε ) · q c + k ∑ j =1 w j · q j ) and distribution

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

By definition, we compute that

<!-- formula-not-decoded -->

where the inequality is by triangle inequality for ℓ 1 distance. By applying Theorem A.4 under our proved conditions (223) and (225), we finally prove that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (227) is by using the definitions of κ and k in (218), (228) is by ⌊ x ⌋ ≥ x 2 for any x ≥ 1 , (229) is by λ ln(1 + x ) ≥ ln (1 + λx ) for λ ≥ 1 and x &gt; 0 , and by ln(1 + λx ) ≤ λ · ln (1 + x ) ≤ x for 0 ≤ λ ≤ 1 and x &gt; 0 , and the last inequality is by (217).

## G Deferred Proofs for DP Per-Instance Upper Bounds

## G.1 Upper Bound: DP Sampling Twice Algorithm

For convenience, we first prove the following lemma about the probability of small symbols going above a threshold, and the probability of large symbols going below a threshold.

Lemma G.1 (Probability of False Positive) . Let m ∈ N , p &gt; 0 , τ &gt; 0 and ε &gt; 0 . Let x ∼ Poi ( mp ) and z ∼ Lap ( 1 ε ) . If p ≤ 1 m min { ε, 1 } , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. By applying Lemma B.2 under setting a = mp i , b = 1 ε and c = τ min { ε, 1 } , we prove that

<!-- formula-not-decoded -->

where the last inequality is by p ≤ 1 m min { ε, 1 } .

Lemma G.2 (Probability of False Negative) . Let m ∈ N , p &gt; 0 , τ &gt; 0 , and ε &gt; 0 . Let x ∼ Poi ( mp ) and z ∼ Lap ( 1 ε ) . If p ≥ t m min { ε, 1 } for t = 6 τ ≥ 1 , then for any constant γ ≥ 0 ,

<!-- formula-not-decoded -->

Proof. By applying Lemma B.2 under setting a = mp i , b = 1 ε and c = τ min { ε, 1 } , we prove that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (234) is by τ 2 -mp i min { 1 ,ε } 3 &lt; -τ 2 -mp i min { 1 ,ε } 6 for p i &gt; t m min { 1 ,ε } undersetting t = 6 τ , and the last inequality is by e -x 6 ≤ O ( 1 x γ ) for x = mp i min { 1 , ε } ≥ t ≥ 1 and γ ≥ 0 .

We then prove two lemmas that bounds KL estimation error on (small) symbols below threshold and (large) symbols above threshold.

Lemma G.3 (Error of Algorithm 2 on Small Symbols - reused samples) . Algorithm 2 satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have denoted L = { i : x i + Lap ( 0 , 1 ε ) ≤ τ min { ε, 1 } } as the randomized instancedependent subset in Algorithm 2.

Proof. By definition, we have that

<!-- formula-not-decoded -->

We first analyze 1 . By αnp i ≤ min { ε, 1 } ≤ ¯ x i under p i ≤ 1 αn min { ε, 1 } , we compute that

<!-- formula-not-decoded -->

We then analyze 2 . By concavity of ln( t ) over t &gt; 0 , we have that

<!-- formula-not-decoded -->

where we have denoted L = { i : x i + Lap ( 0 , 1 ε ) ≤ τ min { ε, 1 } } as the randomized instancedependent subset in Algorithm 2. (241) is by applying Corollary B.5; (242) is by ln(1 + x + y ) ≤ ln(1 + x ) + y for any x ≥ 0 and any y .

By plugging (239) and (242) into (238), we prove that

<!-- formula-not-decoded -->

where (244) is by applying Lemma B.10 under setting m = αn , c 1 = -∞ and c 2 = τ min { ε, 1 } , b = 1 ε and c = 1 min { ε, 1 } for p i &gt; 1 αn min { 1 ,ε } .

Lemma G.4 (Error of Algorithm 2 on Large Symbols - reused samples) . Algorithm 2 satisfies

<!-- formula-not-decoded -->

where we have denoted L = { i : x i + Lap ( 0 , 1 ε ) ≤ τ min { ε, 1 } } as the randomized instancedependent subset in Algorithm 2.

Proof. By definition and by τ ≥ 1 , we compute that

<!-- formula-not-decoded -->

We now analyze 1 . By applying Lemma B.10 under setting m = αn , c 1 = τ min { ε, 1 } , c 2 = + ∞ , b = 1 ε and c = 1 min { 1 ,ε } , we prove that

<!-- formula-not-decoded -->

We finally analyze 2 . By ˜ x i ≥ αnp i for i / ∈ L and p i &lt; 1 αn min { 1 ,ε } , we prove that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second inequality in (250) is by applying Lemma G.1 under setting m = αn , and by applying Lemma B.7 under setting λ = αnp i , b = 1 ε and c = τ min { ε, 1 } . By plugging (249) and (251) into (248), we obtain the bound in the statement.

Lemma G.5 (Error of Algorithm 2 on Large Symbols - fresh samples) . Algorithm 2 satisfies

<!-- formula-not-decoded -->

where we have denoted L = { i : x i + Lap ( 0 , 1 ε ) ≤ τ min { ε, 1 } } as the randomized instancedependent subset in Algorithm 2.

Proof. By the independence between set L and noisy estimate ˜ x ′ i (as they are computed on independently sampled datasets x and x ′ respectively), we compute that

<!-- formula-not-decoded -->

where the first term in (255) is by Pr[ i / ∈ L ] ≤ 1 for p i ≥ 1 αn min { ε, 1 } (by definition) and by applying Lemma B.10 under m = (1 -α ) n , b = 1 ε , c = 1 min { 1 ,ε } , c 1 = -∞ and c 2 = + ∞ ; the second term in (255) is by applying Lemma G.1 under m = αn ; (256) is by definition.

We are now ready to prove the per-instance upper bound for Algorithm 2.

Theorem G.6 (DP 'Sampling Twice' Algorithm) . The estimator A given by Algorithm 2 is ε -DP and satisfies the following error bound for any fixed p ∈ ∆( d ) .

̸

<!-- formula-not-decoded -->

where L = { i : x i + Lap ( 0 , 1 ε ) ≤ τ min { ε, 1 } } is as defined in Algorithm 2.

Proof. By the definition of Algorithm 2, we compute that

<!-- formula-not-decoded -->

where the last inequality is by ln( t ) ≤ t -1 for any t &gt; 0 . We first analyze 1 . By applying Lemma G.3, we prove that

<!-- formula-not-decoded -->

where (263) is by α = 0 . 5 in Algorithm 2. We then analyze 2 . By the independence between L and ˜ c in Algorithm 2 (due to independent sampling of datasets x and x ′ ), conditioned on fixed L , we apply Lemma B.10 under setting m = (1 -α ) n , p = ∑ i ∈ L p i , c 1 = -∞ , c 2 = + ∞ , b = 1 ε and c = 1 min { 1 ,ε } and prove that

̸

<!-- formula-not-decoded -->

̸

where (264) is by setting α = 0 . 5 in Algorithm 2. We finally analyze 3 . By definition of ¯ x i for i / ∈ L , we compute that

̸

<!-- formula-not-decoded -->

where (267) is by the joint convexity of the function x ln ( x y ) with regard to arguments x, y ≥ 0 ; (268) is by applying Lemma G.4 and Lemma G.5 under setting α = 0 . 5 and τ = 4ln d as in Algorithm 2; and (269) is by Pr[ i ∈ L ] ≥ Ω(1) for p i &lt; 2 n min { 1 ,ε } under α = 0 . 5 and τ = 4ln d in Algorithm 2 (by applying Lemma B.2 under setting a = αnp i , b = 1 ε and c = τ n min { 1 ,ε } ). By plugging our proved bound (263), (264) and (269) for 1 , 2 , and 3 into (261), we prove the bound in the statement.

## G.2 Proof for Matching Lower and Upper Bound

Corollary G.7. Let A be the estimator given by Algorithm 2. Let N stat , N ≤ t n , N 1 nε , N ≤ t nε be the additive neighborhoods defined in (137) , (136) , (6) respectively. Then for any n and any p ∈ ∆( d ) ,

<!-- formula-not-decoded -->

under choosing neighborhood size t = 6 τ for τ = 4ln d .

Proof. We will use the upper bound given by Theorem G.6. Observe that for any t &gt; 0 ,

<!-- formula-not-decoded -->

∑ i ∈ L : p i ≤ t αn min { ε, 1 } p i . Thus the first term in Theorem G.6 satisfies

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

where (272) is by ∑ i ∈ L 1 min { 1 ,ε } ∑ i ∈ L np i ≤ d small ( L ) min { 1 ,ε } np small ( L ) , for any t &gt; 0 and d small ( L ) = ∑ i ∈ L : p i ≤ t n min { ε, 1 } 1 and p small ( L ) = ∑ i ∈ L : p i ≤ t n min { ε, 1 } p i ; (273) is Pr[ L = ∅ ] ≤

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma G.2 under setting m = αn and choosing t = 6 τ ; and (275) is by setting τ = 4ln d as defined in Algorithm 2, and (276) is by definition. Additionally, observe that by definition, the second term in Theorem G.6 is ∑ i : p i &gt; 1 n min { 1 ,ε } 1 p i · 1 n 2 min { ε, 1 } 2 ≤ O ( Theorem D.4 + Theorem F.1 ) . Combining this with (276) suffice to prove the bound in the statement.

## G.3 Discussions on the Neighborhood Size

Lemma G.8 (Generalized Packing Argument) . Let d ≥ 16 ∈ N and let O be an output space. Let err : O × O → R be an error function. Given p 1 , · · · , p d ∈ O . For i ∈ [ d ] , denote S ( p i ) as the distribution of histogram sampled from distribution p i . Assume that

1. for any i, j ∈ [ d ] ,

<!-- formula-not-decoded -->

for a coupling ( x, x ′ ) between the distributions S ( p i ) and S ( p j ) ;

2. for any q ∈ O and any S ⊆ [ d ] such that | S | ≥ d 1 / 4 , it holds that

<!-- formula-not-decoded -->

Then for any ( ε, δ ) -DP algorithm A with δ &lt; ε d 1 / 4 ln d , we have

<!-- formula-not-decoded -->

Proof. We will prove the lemma by contradiction. Consider a bipartite graph ( V, E ) on V = [ d ] ×O , where ( i, v ) ∈ E if and only if err ( p i , q ) &lt; ln d 4 . Then

<!-- formula-not-decoded -->

Otherwise the set of neighbors Nbr ( v ) = { i ∈ [ d ] : ( i, v ) ∈ E } violates (278).

Suppose that (279) does not hold, then

<!-- formula-not-decoded -->

Denote the neighbor set Nbr ( i ) = { v ∈ O : ( i, v ) ∈ E } . Then by Markov's inequality

<!-- formula-not-decoded -->

where the last equality is by (281).

On the other hand, for any j ∈ [ d ] , let ( x, x ′ ) be the coupling between S ( p i ) and S ( p j ) in (277), we prove that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (283) is by applying union bound; the first term in (284) is by recursive usage of definition of ( ε, δ ) -DP to datasets ( x, x ′ ) with hamming distance bounded by ln d 4 ε ; the second term in (284) is by applying Markov's inequality; and (285) is by applying condition (277) and δ ≤ ε d 1 / 4 ln d .

By combining (282) and (285), it follows that

<!-- formula-not-decoded -->

By summing (286) over all i , we prove that

<!-- formula-not-decoded -->

Thus there exists v ∈ O , such that

Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality is by d ≥ 16

. This contradicts with (280).

Theorem G.9. Let n, d ≥ 4 ∈ N , ε = ln( d ) 16 n , and 0 ≤ δ ≤ ε d 1 / 4 ln d . For γ ≤ 1 32 , let N γ ( p ) be the local neighborhood as defined in (6) for t = γ ln d .

<!-- formula-not-decoded -->

Then there exists a set P of distribution instances on ∆( d ) , and a per-neighborhood estimator A N γ ( p ) under neighborhood size γ , such that

<!-- formula-not-decoded -->

while for any ( ε, δ ) -DP estimator A , we have

<!-- formula-not-decoded -->

Thus if γ ≤ o (1) , then no ( ε, δ ) -DP estimator A could satisfy (2) (otherwise it contradicts (290) and (291) ).

Proof. Consider the following construction of P .

<!-- formula-not-decoded -->

̸

Thus the following construction of per-neighborhood estimator satisfies (290).

̸

<!-- formula-not-decoded -->

This is because

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (296) is by γ · ln d nε ≤ 1 2 under γ ≤ 1 32 and by ε = ln d 2 n .

Below we focus on proving (291) by applying Lemma G.8. We only need to validate that the two conditions of Lemma G.8 hold.

1. The first condition (277) holds by ε = ln d 16 n .

2. The second condition of (278) holds by convexity of the function ln( 1 t ) on t &gt; 0 , which ensures that for any S ⊆ [ d ] with | S | ≥ d 1 / 4 and any q ∈ ∆( d ) , we have

<!-- formula-not-decoded -->

where the second-to-last inequality is by ∑ i ∈ S q i ≤ 1 , and the last inequality is by | S | ≥ d 1 / 4 .

## G.4 Additional Experiments on More Datasets

In this section, we further evaluate on more data distributions: power law distributions p i ∝ 1 i β for β = 1 . 5 , 2 in Figure 3, and 4; Enron-email corpus in Figure 5; and MMLU corpus in Figure 6.

Figure 3: (Power law distribution p i ∝ 1 i 1 . 5 ) KL error versus dataset size n , distribution dimension d , and DP guarantee ε for our methods compared with the simple minimax optimal Add-constant (DP) baseline, and the strongest non-DP baseline of prior (near) instance-optimal Good-Turing estimator.

<!-- image -->

Figure 4: (Power law distribution p i ∝ 1 i 2 ) KL error versus dataset size n , distribution dimension d , and DP guarantee ε for our methods compared with the simple minimax optimal Add-constant (DP) baseline, and the strongest non-DP baseline of prior (near) instance-optimal Good-Turing estimator.

<!-- image -->

## G.5 Instance Optimality of Gaussian Variant of Our Algorithm

Below we present a Gaussian variant of our algorithm, and prove that it is near instance optimal up to a log(1 /δ ) factor.

Figure 5: (Enron-emails Token Distribution Estimation) KL error versus dataset size n , distribution dimension d , and DP guarantee ε for our methods compared with the simple minimax optimal Add-constant (DP) baseline, and the strongest non-DP baseline of prior (near) instance-optimal Good-Turing estimator.

<!-- image -->

Figure 6: (MMLU Token Distribution Estimation) KL error versus dataset size n , distribution dimension d , and DP guarantee ε for our methods compared with the simple minimax optimal Add-constant (DP) baseline, and the strongest non-DP baseline of prior (near) instance-optimal Good-Turing estimator.

<!-- image -->

## Algorithm 3 ( ε, δ ) -DP 'Sampling Twice' (Instance-Optimal)

Inputs: Data partition ratio α = 0 . 5 . Independently sampled datasets x ∼ Poi ( α · np ) and x ′ ∼ Poi ((1 -α ) · np ) s.t. x + x ′ ∼ Poi ( np ) . Threshold τ = 4ln d . Noise magnitude σ = √ ln(1 /δ ) ε . L = ∅

for symbol i = 1 , · · · , d do

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem G.10. The estimator A given by Algorithm 3 is ( ε, δ ) -DP and satisfies the following error bound for any fixed p ∈ ∆( d ) .

̸

<!-- formula-not-decoded -->

where L is the randomized set as defined in Algorithm 3.

Proof. The ( ε, δ ) -DP guarantee follows by observing that the ℓ 2 -sensitivity of vector release ( x 1 , · · · , x d , x ′ 1 , · · · , x ′ d ) is one, and by applying the ( ε, δ ) -DP guarantee for Gaussian mechanism in [27, Theorem A.1]. The KL error bound follows identically as Appendix G.1, after replacing all Laplace Tail bounds under Lap (1 /ε ) with Gaussian Tail bounds under N (0 , σ 2 ) for σ = √ 2 ln(1 . 25 /δ ) .

<!-- formula-not-decoded -->