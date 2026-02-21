## Optimal Online Change Detection via Random Fourier Features

## Florian Kalinke ∗

Information Systems Karlsruhe Institute of Technology (KIT) Karlsruhe, Germany

florian.kalinke@kit.edu

## Shakeel Gavioli-Akilagun *

Department of Decision Analytics and Operations City University Hong Kong Hong Kong, China sgavioli@cityu.edu.hk

## Abstract

This article studies the problem of online non-parametric change point detection in multivariate data streams. We approach the problem through the lens of kernelbased two-sample testing and introduce a sequential testing procedure based on random Fourier features, running with logarithmic time complexity per observation and with overall logarithmic space complexity. The algorithm has two advantages compared to the state of the art. First, our approach is genuinely online, and no access to training data known to be from the pre-change distribution is necessary. Second, the algorithm does not require the user to specify a window parameter over which local tests are to be calculated. We prove strong theoretical guarantees on the algorithm's performance, including information-theoretic bounds demonstrating that the detection delay is optimal in the minimax sense. Numerical studies on real and synthetic data show that our algorithm is competitive with respect to the state of the art.

## 1 Introduction

In the online change point detection problem, data is observed sequentially, and the goal is to flag a change if the distribution of the data changes. The problem dates back to the early work of Page [43], and has now been extensively studied in the statistics and machine learning literature [30, 3, 52]. However, these classical approaches assume that the data are low-dimensional and that the preand post-change distributions belong to a known parametric family. In modern applications, both assumptions are usually not satisfied. Examples of modern online change point detection problems include: detecting changes in audio streams [4], in videos [1, 25], in highway traffic data [15], in internet traffic data [31], or in cardiac time series [61]. Further, such data frequently has high volume, and online change point detection procedures should still be able to process new data in real time and with limited memory.

While algorithms for the problem of non-parametric online change point detection have been proposed-see Section 2 for a brief overview-state-of-the-art methods suitable for modern data suffer from at least one of the following two limitations. First, most procedures are not genuinely online from a statistical perspective, in the sense that they either assume the pre-change distribution to be known completely or they assume having access to historical data known to be from the pre-change distribution. Second, most approaches require the user to specify a window parameter over which local tests for a change in distribution will be applied. Choosing such a window is notoriously difficult [47], and choosing the window too small or too large leads to a reduction in power or an increase in the detection delay, respectively. Moreover, the window size imposes a limit on the historic data considered, rendering detecting sufficiently small changes impossible.

∗ Contributed equally.

Motivated by the above limitations and challenges, we propose a new algorithm called Online RFFMMD (Random Fourier Feature Maximum Mean Discrepancy). On a high level, the algorithm performs sequential two-sample tests based on the kernel-based maximum mean discrepancy (MMD; [53, 14]). Crucially, approximating the maximum mean discrepancy using random Fourier features (RFFs; [45]) leads to a detection statistic that can be computed in linear and updated in constant time. By embedding these local tests in a sequential testing scheme on a dyadic grid of candidate change point locations, we obtain an algorithm that does not require a window parameter and features a time and space complexity logarithmic in the amount of data observed.

This article makes the following contributions:

- Computational efficiency: We propose Online RFF-MMD, a fully non-parametric change point detection algorithm that does not require access to historical data, has no window parameter, and features logarithmic runtime and space complexity.
- Minimax optimality: Online RFF-MMD comes with strong theoretical guarantees. In particular, we derive information-theoretic bounds showing that the detection delay incurred by Online RFF-MMD is optimal up to logarithmic terms in the minimax sense. While related results are known in the offline setting [41, 42] and in the parametric online setting [30, 64], ours is the first result of this kind for kernel-based online change point detection.
- Empirical validation: We perform a suite of benchmarks on synthetic data, the MNIST data, and the HASC data to demonstrate the applicability of the proposed method. Our approach achieves competitive results throughout all experiments.

The article is structured as follows. We recall related work in Section 2 and introduce our notations and the problem in Section 3. Section 4 presents our algorithm and its guarantees, and Section 5 its minimax optimality. Experiments are in Section 6 and limitations in Section 7. Additional results and all proofs are in the appendices.

Table 1: Comparison of kernel-based change detectors. Genuinely online -whether the algorithm can be executed without reference data known to be from the pre-change distribution; Window free -whether the algorithm requires selection of a window parameter over which the detection statistic is calculated; Time comp. - runtime complexity per new observation; Space comp. - total space complexity; n -total number of observations. 2

| Algorithm                | Genuinely online   | Window free   | Time comp.   | Space comp.   |
|--------------------------|--------------------|---------------|--------------|---------------|
| Scan B -statistics [33]  | ✗                  | ✗             | O ( NW 2 )   | O ( NW )      |
| NEWMA[24]                | ✓                  | ✗             | O ( r )      | O ( r )       |
| Online kernel CUSUM [59] | ✗                  | ✗             | O ( NW 2 )   | O ( NW )      |
| Online RFF-MMD           | ✓                  | ✓             | ( r log n )  | ( r log n )   |

## 2 Related work

In the case of univariate data, numerous methods for non-parametric online change point detection have been proposed; we refer to Ross [49] for a more comprehensive overview. The most successful among these approaches exploit the fact that all information is contained in the data's empirical distribution function, which is a functional of the ranks. Rank-based online change point detection methods have been proposed by [13, 19, 48]. However, their extension to multivariate data is challenging as this requires a computationally efficient multivariate analogue to scalar ranks.

To tackle change point detection on multivariate data, many approaches exist; see Wang and Xie [58] for a survey. For example, Yilmaz [62], Kurt et al. [28] introduce procedures using summary statistics based on geometric entropy minimization, but assume knowledge of the pre-change distribution. An alternative non-parametric approach is using information on distances between data points [7, 8]. However, here one can construct alternatives which the procedures will always fail to detect as the limits of the test statistics employed do not metrize the space of probability distributions.

2 In Table 1, where applicable, r denotes the number of random Fourier features, W denotes the size of the window, and N denotes the number of blocks of historical data. We treat the dimension of the data as fixed.

O

O

One principled approach to tackle this challenging setting is using kernel-based two sample tests via the MMD, which we recall briefly in Section 3.2. The key property, if the underlying kernel is characteristic [11, 55], is that the MMD metrizes the space of probability distributions and thus allows detecting any change. However, the sample estimators of the MMD typically have a quadratic runtime complexity, which prohibits their direct application in the online setting. To overcome this challenge, Zaremba et al. [66] propose Scan B -statistics, which achieve sub-quadratic time complexity by splitting the data into blocks and computing the quadratic-time MMD on each block. Consequently, Li et al. [32, 33] introduce an online change point detection algorithm that recursively estimates the Scan B -statistic over a sliding window. Extending this idea, Wei and Xie [59] propose an algorithm which performs the same operation over grid of windows of different sizes. However, these algorithms all require the choice of a window parameter and are not genuinely online as they require historical data known to be from the pre-change distribution. Keriven et al. [24] propose comparing two exponentially smoothed MMD statistics, where the MMD is approximated by using random Fourier features. However, the window selection problem is not avoided because, as shown by the authors, the smoothed statistic can be interpreted as computing differences between MMDs calculated on two windows of different sizes.

We summarize the kernel-based approaches coming with theoretical guarantees in Table 1 and note that we present two extensions of our proposed Online RFF-MMD approach in the appendix. The first allows taking historical data into account (Appendix A.1). The second permits detecting multiple change points within the same stream, with theoretical guarantees (Appendix A.2).

## 3 Preliminaries

In this section, we formally introduce the online change point detection problem (Section 3.1) and recall kernel-based two sample testing together with its RFF-based approximation (Section 3.2).

## 3.1 Problem statement

We consider a data stream X 1 , X 2 , . . . observed online, where the X t -s are independent random variables taking values in R d , with d arbitrary but fixed. Let P , Q ∈ M + 1 := M + 1 ( R d ) and P = Q , where M + 1 ( R d ) denotes the set of all Borel probability measures on R d . We assume that there exists an η ∈ N := { 1 , 2 , . . . } such that

.

<!-- formula-not-decoded -->

The goal is to stop the process with minimal delay as soon as η is reached, but not before. Note that we may have η = ∞ in which case the process should never be stopped. Formally, one wants to test

̸

for each n ∈ N , until a local null is rejected. A secondary aim, once a local null has been rejected, is to accurately estimate η . Solving the aforementioned problems boils down to constructing an extended stopping time N , which, in a sense we make precise in Section 4.3, is close to η . Let F t = σ ( X s | s = 1 , . . . , t ) be the natural filtration generated by the X s 's up to time t and recall that a random variable N is an extended stopping time if (i) N takes vales in N ∪ {∞} and (ii) for each t ∈ N the event { N ≤ t } is F t -measurable.

<!-- formula-not-decoded -->

Minimizing the distance between η and N is analogous to maximizing the power of a particular sequential testing procedure, and it is therefore natural to impose some conditions on the sequential testing analogue of statistical size; we recall the two most frequent ones in the following. In the sequel, let P k be the joint distribution of { X t } t&gt; 0 when { η = k } ( k ∈ N ), and let E k be the expectation under this distribution.

1. The average run length until a spurious rejection under the global null should be bounded from below by a chosen quantity [36]. Specifically, for a given γ &gt; 1 it should hold that

<!-- formula-not-decoded -->

H 0 ,n : X t ∼ P for each t ≤ n and some P ∈ M + 1 versus

̸

2. The uniform false alarm probability should be bounded from above by a chosen quantity [30]. Specifically, for a given α ∈ (0 , 1) it should hold that

We show in Section 4.3 that Online RFF-MMD is able to satisfy either of the conditions (1) or (2).

<!-- formula-not-decoded -->

## 3.2 Fast kernel-based two sample tests

To resolve the change point detection problem (Section 3.1), we embed fast two sample tests based on RFF approximations to the MMD into a particular sequential testing scheme. In this section, we briefly recall the MMD statistic and its RFF approximation.

Let H K be a reproducing kernel Hilbert space (RKHS; [2, 56]) on R d with (reproducing) kernel K : R d × R d → R . Denote by supp (Λ) = { A ∈ σ ( R d ) | Λ( A ) &gt; 0 } the support of a Borel measure Λ on R d , where A denotes the closure of a set A and σ ( R d ) the Borel σ -algebra on R d . Throughout the article, we make the following assumption on the kernel:

Assumption 1. The kernel K : R d × R d → R is non-negative, continuous, bounded ( ∃ B &gt; 0 s.t. sup x ∈ R d K ( x , x ) ≤ B ), translation-invariant ( K ( x , y ) = ψ ( x -y ) for some positive definite ψ : R d → R ), and characteristic (supp (Λ) = R d with ψ ( x ) = ∫ e -iω T x dΛ( ω ) ).

To simplify exposition, we also assume that K ( 0 , 0 ) = 1 , which can be achieved by scaling any bounded kernel. The conditions in Assumption 1 are satisfied by several commonly used kernels, including the Gaussian kernel, mixtures of Gaussians, inverse multi-quadratic kernels, Matérn kernels, Laplace kernels, or B-spline kernels [55]. Assumption 1 permits to approximate the respective kernel function by finite-dimensional feature maps through Bochner's theorem (elaborated below), which, in turn, allows the effective online estimation of MMD detailed in Section 4.

To any P ∈ M + 1 , one can associate the kernel mean embedding µ K ( P ) ∈ H K , taking the form 3

<!-- formula-not-decoded -->

where the integral is meant in Bochner's sense [10, Chapter II.2]. The continuity and boundedness assumptions ensure the existence of µ K ( P ) for any P ∈ M + 1 [55, Proposition 2]. Expression (3) gives rise to the MMD, which quantifies the distance between two measures P , Q ∈ M + 1 via the distance between their mean embeddings in terms of the RKHS norm, and takes the form

<!-- formula-not-decoded -->

Crucially, the characteristic assumption ensures that (3) is injective and that the MMD metrizes the space M + 1 [55, Theorem 9], implying MMD K [ P , Q ] = 0 iff. P = Q . Given samples X 1 , . . . , X n i.i.d. ∼ P and Y 1 , . . . , Y m i.i.d. ∼ Q with associated empirical measures ˆ P n =: X 1: n and ˆ Q m =: Y 1: m , respectively, the squared plug-in estimator of MMD K [ P , Q ] takes the form

<!-- formula-not-decoded -->

The computation of (5) costs O ( (max( m,n )) 2 ) , rendering its use in an online testing procedure computationally infeasible.

Random Fourier features [45, 54] alleviate this bottleneck in the offline setting; we recall the method in the following. For some ω ∈ R d write ζ ω ( x ) = e -iω T x , where i = √ -1 . By Bochner's theorem,

<!-- formula-not-decoded -->

3 For x ∈ R d , K ( · , x ) : R d → R denotes the map x ′ ↦→ K ( x ′ , x ) .

where ∗ denotes the complex conjugate and, using that K (0 , 0) = 1 , one has that Λ ∈ M + 1 . As Λ and K are real-valued, Euler's identity implies that ∫ e -iω T ( x -y ) dΛ( ω ) = ∫ cos ( ω T ( x -y ) ) dΛ( ω ) . Therefore, using that cos( α -β ) = cos α cos β +sin α sin β , picking some r ∈ N , and sampling

<!-- formula-not-decoded -->

ω 1 , . . . , ω r i.i.d. ∼ Λ , a low variance estimator for K ( x , y ) is given by

By noting that ˆ K : R d × R d → R is the kernel associated with the RKHS H ˆ K = R 2 r , we may approximate (5) by

<!-- formula-not-decoded -->

Importantly, as the mean embeddings in (7) are Euclidean vectors, their distance can be computed with the standard Euclidean norm. This leads to a statistic which can computed in linear time and updated in constant time, allowing its use for sequential testing.

## 4 Online change point detection via random Fourier features

In this section, we present our proposed stopping time for resolving the change point detection problem. In particular, we give a precise definition in Section 4.1, an efficient algorithm in Section 4.2, and theoretical guarantees in Section 4.3.

## 4.1 The RFF-MMD stopping time

The intuitive construction of our RFF-MMD stopping time is as follows. We begin by choosing an r ∈ N and a kernel K , and construct its RFF approximation (6) using r random features. For every n ≥ 2 , having observed data { X 1 , . . . , X n } , we consider log 2 n possible sample splits of the domain { 1 , . . . , n } at locations n -2 j with j = 0 , . . . , ⌊ log 2 n ⌋ -1 . For every such split, we approximate the MMD between the two samples using (7). A change is declared at the first n for which at least one such statistic, appropriately normalized so that it is O P (1) under its local null, is larger than a given threshold. Formally, we have

<!-- formula-not-decoded -->

where { λ n | n ∈ N } is a non-decreasing sequence that we make precise in Section 4.3, taking requirements (1) or (2) into account.

The first use of an exponential grid in online change point detection appears to be due to Lai [29]. Recently, similar techniques have been used by Yu [63], Kalinke et al. [23], Moen [39]. The dyadic grid used in (8) has two advantages. First, only a logarithmic number of tests must be performed with each new observation. Second, the grid is sufficiently dense, so the obtained stopping time has essentially the same behavior as the computationally infeasible variant, which considers every possible candidate change point location.

## 4.2 The RFF-MMD algorithm

We now present an efficient implementation of (8) and analyze its runtime and space complexity. We show the pseudo code of our proposed method in Algorithm 1; see also Example 1 and Figure 1 for a summary. The details are as follows. For each new observation X t , we create a new window W , storing z = ˆ z K ( X t ) and c = 1 (Lines 3-4). The window W is then added to the list of all windows W (Line 5). The remaining algorithm has two main parts.

1. Change point detection. To detect changes, we iterate all |W| 1 dyadic points i (Line 6), and, for each i , merge the feature maps coming before i and coming after i (along with their counts) to compute the MMD statistic (Lines 7-9). If the statistic exceeds the threshold (Line 10; see Section 4.3 for its value), a change is flagged and we drop the data coming before the change.

2. Structure maintenance. To set up and maintain the dyadic structure, we merge windows that have the same counts (Line 16) by first summing their z -s and their c -s (Lines 17-18), and then replacing them in the list of windows W accordingly (Line 19). We note that pop removed the windows beforehand.

## Algorithm 1 Online RFF-MMD change point detection

```
Input: Stream X 1 , X 2 , . . . and a sequence of thresholds { λ t | t ∈ N } . Output: Changepoint location and detection time. 1: W ← empty list 2: for X t ∈ X 1 , X 2 , . . . do ▷ Main loop 3: W.z ← ˆ z K ( X t ) 4: W.c ← 1 5: W ← W .append ( W ) 6: for i ∈ 1 , . . . , |W| 1 do ▷ Detect changes 7: c 1 ← ∑ |W| j = i +1 W j .c 8: c 2 ← ∑ i j =1 W j .c 9: MMD ˆ K ← ∥ ∥ ∥ 1 c 1 ∑ |W| j = i +1 W j .z -1 c 2 ∑ i j =1 W j .z ∥ ∥ ∥ 2 10: if √ c 1 c 2 c 1 + c 2 MMD ˆ K > λ t then 11: print Change detected at element X t ; most likely at position i . 12: return 13: while |W| ≥ 2 do ▷ Maintain exponential structure 14: W 1 ← pop W 15: W 2 ← pop W 16: if W 1 .c = W 2 .c then 17: W.c ← W 1 .c + W 2 .c 18: W.z ← W 1 .z + W 2 .z 19: W ← W .append ( W ) 20: else 21: W ← W .append ( W 1 ) .append ( W 2 ) 22: break
```

We now analyze the runtime and space complexity of Online RFF-MMD. For each insert operation, Algorithm 1 performs three steps, which we analyze independently.

1. Setup. The computation of ˆ z K ( X t ) , defined in (6), requires computing 2 r trigonometric functions of d -dimensional inner products and thus is in O ( rd ) .
3. Maintenance. In the worst case, O ( |W| ) merge operations need to be performed. Each merge requires O ( r ) operations, which yields a total cost of O ( |W| r ) .
2. Change point detection. The dominating term is computing MMD ˆ K , which requires O ( |W| r ) computations. Repeating the computation |W| times leads to a cost of O ( |W| 2 r ) . The calculation of the threshold is in O (1) , which gives an overall cost of O ( |W| 2 r ) . However, we note that memoization of all sums allows to implement the change point detection in a single sweep over W (at each step, the attributes of one W ∈ W are subtracted from one sum and added to another sum) and thereby reduces the runtime complexity to O ( |W| r ) .

Adding the results obtained in steps 1.-3. shows that the algorithm has an overall runtime complexity of O ( |W| r ) = O ( r log n ) per insert operation. As the algorithm stores, for each W ∈ W , a number ( W.c ) and a vector ( W.z ∈ R 2 r ), the total space complexity when having observed n samples is O ( |W| r ) = O ( r log n ) . We note that r is a fixed parameter in practice and thus constant.

The following Example 1 and the corresponding Figure 1 illustrate how Algorithm 1 operates upon observing the first 6 samples X 1 , . . . , X 6 .

Example 1. When observing the first element X 1 , the algorithm creates a new window W , storing the feature map ˆ z K ( X 1 ) and that W has one element ( c = 1 ). Similarly, when observing X 2 , the algorithm creates a new window W ′ , storing ˆ z K ( X 2 ) and c = 1 . As both windows, W and W ′ , have the same counts, the algorithm merges them into a new window W , storing ˆ z K ( X 1 ) + ˆ z K ( X 2 ) and c = 2 , thereby maintaining the dyadic structure. The algorithm proceeds in this manner, resulting

Figure 1: Schematic representation of the proposed algorithm upon observing the first n = 6 elements. Merging equal sized 'windows' yields the division along dyadic points.

<!-- image -->

in the construction of the dyadic grid outlined in Section 4.1. For example, when observing X 6 , there are again two windows of size 1 , which the algorithm merges to store a total of two windows, one capturing X 1 , . . . , X 4 and the other one capturing X 5 , X 6 . We recall that the observations themselves are never stored explicitly.

Before presenting out theoretical results in the following section, we emphasize that Algorithm 1 matches (8), that is, there is no approximation error.

## 4.3 Theoretical results

In this section, we analyze the theoretical behavior of the RFF-MMD algorithm. We first study the behavior of the stopping time defined in (8) under the global null of no change. Theorems 1 and 2 show that with an appropriately chosen sequence of thresholds the stopping time can be made to attain, respectively, a desired average run length (1) or a desired uniform false alarm probability (2).

Theorem 1. Let N be the extended stopping time defined via (8). For any γ &gt; 1 , if the sequence of thresholds satisfies λ n ≥ √ 2 + √ 2 log (4 γ log 2 (2 γ )) for all n ∈ N , it holds that E ∞ [ N ] ≥ γ .

Theorem 2. Let N be the extended stopping time defined via (8). For any α ∈ (0 , 1) , if the sequence of thresholds satisfies λ n ≥ √ 2 + √ 2 (log( n/α ) + 2 log (log 2 ( n )) + log (log 2 (2 n ))) for each n ∈ N , it holds that P ∞ ( N &lt; ∞ ) ≤ α .

We emphasize that these guarantees do not depend on the number of random features used in constructing (6) and the bounds on the threshold sequences do not require any knowledge of the pre-change distribution.

Next, we study the detection delay incurred by (8) when the threshold sequence is chosen to control the uniform false alarm probability at some level α ∈ (0 , 1) . In the following, we assume that the data take values in a compact subset of R d , and denote the Lebesgue measure of a set by | · | . The following result shows that with high probability, provided the number of RFFs is chosen sufficiently large, the detection delay incurred by (8) is bounded from above by a quantity depending only on the chosen α , the number of pre-change observations, and the squared MMD between the pre- and post-change distributions.

Theorem 3. Let N be the extended stopping time defined via (8) with threshold sequence { λ n | n ∈ N } defined as in Theorem 2 for a chosen α ∈ (0 , 1) . If supp( P ) ∪ supp( Q ) ⊆ X for some compact set X ⊂ R d , the quantities η , α , and MMD K [ P , Q ] jointly satisfy

<!-- formula-not-decoded -->

and the number of random features in (7) is chosen so that

<!-- formula-not-decoded -->

then with probability at least 1 -α , it holds that

<!-- formula-not-decoded -->

where C 1 , C 2 , and C 3 are absolute constants independent of η , α , and MMD K [ P , Q ] , and, with σ 2 = ∫ R d ∥ ω ∥ 2 2 dΛ( ω ) , we have put

<!-- formula-not-decoded -->

which is likewise independent of η , α , and MMD K [ P , Q ] .

Condition (9) can be interpreted as a signal strength requirement, measuring the strength according to the number of observations from the pre-change distribution and the squared MMD between P and Q . The term log(2 η/α ) reflects the cost of multiple testing when the data are drawn from P . Such requirements are unavoidable from the minimax perspective in the corresponding offline problem [63, 42, 41], and the discussion in Yu et al. [64, Section 4.1] suggests that the same is true for genuinely online change point problems.

The requirement on the number of RFFs in (10) depends on MMD K [ P , Q ] , which is unknown in practice. However, if one assumes an asymptotic setting with a fixed distance between P and Q , and α ↓ 0 , then (10) suggests that the number of RFFs should be chosen as r = Θ(log 1 /α ) . To put this result into perspective, we compare it to online change procedures having a window. Here, the optimal window length also depends on the distance between the pre- and post-change distributions [33, 59]. However, choosing the window larger than this quantity can lead to an increase in the detection delay. This is not the case for RFF-MMD: in practice, one may choose the number of RFFs as large as possible subject to computational constraints, and choosing a larger number of RFFs does not negatively impact the detection delay.

## 5 Minimax optimality of RFF-MMD

Recall that with the conditions of Assumption 1, the underlying kernel is characteristic and MMD K metrizes the space M + 1 . Therefore, MMD K [ P , Q ] &gt; 0 for any P = Q and we have that (11) guarantees that our stopping time (8) obtains a finite detection delay, with high probability for any fixed alternative. Still, one may ask whether the detection delay is optimal. The following theorem resolves this question and shows that the detection delay of RFF-MMD is essentially optimal from a minimax perspective, up to logarithmic terms.

̸

Theorem 4. For every kernel K : R d × R d → R satisfying Assumption 1 there is a constant C K depending only on K and absolute constants α 0 , β 0 ∈ (0 , 1) independent of K , such that for any α ≤ α 0 it holds that

<!-- formula-not-decoded -->

with the infimum being over all extended stopping times.

We remark that in the online change point detection literature [40, 46], it is more common to study the expected risk of a stopping time. For example, for fixed P , Q , it is common to work with the so-called worst-worst-case average detection delay [37] of a given stopping time N , which is defined via

<!-- formula-not-decoded -->

However, in the absence of further restrictions, studying this quantity for the problem at hand does not seem possible. In fact, as long as P ⊗ η := P ⊗··· ⊗ P and Q ⊗ η := Q ⊗··· ⊗ Q have a total variation distance smaller than 1 , one can couple the given process and a process where all X t 's are drawn from Q so that with non-zero probability the two processes have identical sequences. In this case, we either lose control of the null and α cannot be arbitrarily close to zero, or we maintain control over the null but with non-zero probability the detection delay is infinite.

## 6 Experiments

We collect our experiments on synthetic data in Section 6.1 and on the MNIST data set in Section 6.2. We refer to Appendix A.3 for additional experiments and a numerical comparison of different thresholds for the stopping rule. To interpret the change point detection performance of the proposed method, we compare its average run length (ARL) and expected detection delay (EDD) to the existing kernel-based methods presented in Table 1. 4 For all experiments, we use the Gaussian kernel K ( x , y ) = e -γ ∥ x -y ∥ 2 2 with γ set by the median heuristic [12] or its RFF approximation, depending on the algorithm. All results were obtained on a PC with Ubuntu 20.04 LTS, 124GB RAM, and 32 cores with 2GHz each.

## 6.1 Synthetic data

In this section, we evaluate the runtimes of different configurations of the proposed Online RFF-MMD algorithm and compare its change point detection performance on synthetic data to that of other kernel-based approaches.

Runtime. Figure 2 summarizes the runtime results of Algorithm 1 with the number of random Fourier features r ∈ { 10 , 50 , 100 , 500 , 1 000 } and for streams of length up to n = 250000 . The experiments verify the O ( r log n ) runtime complexity of the proposed algorithm, derived analytically in Section 4.2. We note that the dependence on d is linear; we consider d = 1 only.

Figure 2: Average runtime ( 10 repetitions) of RFF-MMD per insert operation (left) and total (right).

<!-- image -->

ARL vs. expected detection delay. To illustrate the EDD for a given target ARL, we reproduce the experiments of Wei and Xie [59, Figure 4], also taking our method into account. Specifically, we consider the pre-change distribution P = N ( 0 20 , I 20 ) and set the parameters of each algorithm as follows. Matching the settings of the reproduced experiment, we choose B max = 50 and N = 15 for online kernel CUSUM; for Scan B-statistics and NewMA, we set B 0 = 50 . The remaining parameters of NewMA then follow from the heuristics detailed by the authors [24]. For Online RFF-MMD, we set r = 1000 . We compute the thresholds for a given target ARL by processing 10 × ( target ARL ) samples with each algorithm, repeating for 100 Monte Carlo (MC) iterations, and computing the 1 -1 / ( target ARL ) quantile of the resulting test statistics. For approximating the EDD of each algorithm, we draw 64 samples from P , respectively, before sampling from Q ; we report the average over 100 repetitions. OKCUSUM and ScanB additionally receive 1 000 samples from P upfront, to use as a reference sample. For NewMA, we process 400 additional samples from P for both the MC estimate and the EDD experiment, to reduce its variance.

Having processed the indicated number of samples from P , we then start sampling from either a mixed normal, a Laplace, or a Uniform distribution. Figure 3 collects the average detection delay; the respective post-change distribution Q is given on top. The results show that our algorithm achieves a smaller detection delay than the competitors for all considered post-change distributions, sometimes by a large margin.

4 All code replicating our experiments is available in the supplement and at https://github.com/ FlopsKa/rff-change-detection .

Figure 3: Average detection delay from P = N ( 0 20 , I 20 ) to the Q indicated on top ( d = 20 , σ = 2 ).

<!-- image -->

## 6.2 MNIST data

In this section, we interpret the MNIST data set [9] as high-dimensional data stream, similar to Wei and Xie [59, Figure 7], with the goal of detecting a change when the digit changes from 0 to a different digit. The experimental setup matches that of Section 6.1, but with d = 784 . Figure 4 collects our results; the results for the digits 4-6 are similar and in Appendix A.3.1. Similar to Figure 3, the proposed Online RFF-MMD algorithm shows very good performance. In particular, it achieves a lower EDD than all tested competitors throughout.

Figure 4: Average detection delay from MNIST digit 0 to digits 1 , 2 , and 3 (left to right).

<!-- image -->

## 7 Limitations

As with all kernel-based tests, the choice of kernel impacts the power of the test. While our theoretical guarantees (Section 4.3) hold for any kernel satisfying Assumption 1 and do not require any prechange data, one usually selects the kernel or its parameters using a few available samples in practice. While kernel optimization is not the focus of this work, there exist works [20-22, 34, 50, 51, 16-18] to (approximately) achieve this goal; it is interesting future work to tackle this problem in the sequential setting. A separate future direction is considering non-i.i.d. data, for example, data that is strongly mixing [6] or data exhibiting functional dependence [60].

## Acknowledgments and Disclosure of Funding

The authors thank Zoltán Szabó and Tengyao Wang for helpful discussions. FK thanks Georg Gntuni and Marius Bohnert for exchanges on the algorithm's implementation. This work was supported by the pilot program Core-Informatics of the Helmholtz Association (HGF).

## References

- [1] Abdalbassir Abou-Elailah, Valérie Gouet-Brunet, and Isabelle Bloch. Detection of abrupt changes in spatial relationships in video sequences. In International Conference on Pattern

Recognition Applications and Methods (ICPRAM) , pages 89-106, 2015.

- [2] Nachman Aronszajn. Theory of reproducing kernels. Transactions of the American Mathematical Society , 68:337-404, 1950.
- [3] Michèle Basseville and Igor V. Nikiforov. Detection of abrupt changes: theory and application . Prentice Hall, 1993.
- [4] Alberto Bietti, Francis R. Bach, and Arshia Cont. An online EM algorithm in hidden (semi-) Markov models for audio segmentation and clustering. In International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 1881-1885, 2015.
- [5] Stéphane Boucheron, Gábor Lugosi, and Pascal Massart. Concentration inequalities . Oxford University Press, 2013.
- [6] Richard C. Bradley. Basic properties of strong mixing conditions. A survey and some open questions. Probability Surveys , 2:107-144, 2005. Update of, and a supplement to, the 1986 original.
- [7] Hao Chen. Sequential change-point detection based on nearest neighbors. The Annals of Statistics , 47(3):1381-1407, 2019.
- [8] Lynna Chu and Hao Chen. Sequential change-point detection for high-dimensional and nonEuclidean data. IEEE Transactions on Signal Processing , 70:4498-4511, 2022.
- [9] Li Deng. The MNIST database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine , pages 141-142, 2012.
- [10] Joseph Diestel and John J. Uhl. Vector Measures . American Mathematical Society, 1977.
- [11] Kenji Fukumizu, Arthur Gretton, Xiaohai Sun, and Bernhard Schölkopf. Kernel measures of conditional dependence. In Advances in Neural Information Processing Systems (NeurIPS) , pages 489-496, 2007.
- [12] Damien Garreau, Wittawat Jitkrittum, and Motonobu Kanagawa. Large sample analysis of the median heuristic. Technical report, 2018. https://arxiv.org/abs/1707.07269 .
- [13] Louis Gordon and Moshe Pollak. An efficient sequential nonparametric scheme for detecting a change of distribution. The Annals of Statistics , 22(2):763-804, 1994.
- [14] Arthur Gretton, Karsten Borgwardt, Malte Rasch, Bernhard Schölkopf, and Alexander Smola. A kernel two-sample test. Journal of Machine Learning Research , 13(25):723-773, 2012.
- [15] Robert Grossman, Michal Sabala, Anushka Aanand, Steve Eick, Leland Wilkinson, Pei Zhang, John Chaves, Steve Vejcik, John Dillenburg, Peter Nelson, et al. Real time change detection and alerts from highway traffic data. In ACM/IEEE Conference on Supercomputing (SC) , pages 69-69, 2005.
- [16] Omar Hagrass, Bharath K. Sriperumbudur, and Bing Li. Spectral regularized kernel two-sample tests. The Annals of Statistics , 52(3):1076-1101, 2024.
- [17] Omar Hagrass, Bharath K. Sriperumbudur, and Bing Li. Spectral regularized kernel goodnessof-fit tests. Journal of Machine Learning Research , 25(309):1-52, 2024.
- [18] Omar Hagrass, Bharath Sriperumbudur, and Krishnakumar Balasubramanian. Minimax optimal goodness-of-fit testing with kernel Stein discrepancy. Bernoulli , 2025. (accepted; preprint: https://arxiv.org/abs/2404.08278 ).
- [19] Douglas M. Hawkins and Qiqi Deng. A nonparametric change-point control chart. Journal of Quality Technology , 42(2):165-173, 2010.
- [20] Wittawat Jitkrittum, Zoltán Szabó, Kacper Chwialkowski, and Arthur Gretton. Interpretable distribution features with maximum testing power. In Advances in Neural Information Processing Systems (NeurIPS) , pages 181-189, 2016.

- [21] Wittawat Jitkrittum, Zoltán Szabó, and Arthur Gretton. An adaptive test of independence with analytic kernel embeddings. In International Conference on Machine Learning (ICML) , pages 1742-1751, 2017.
- [22] Wittawat Jitkrittum, Wenkai Xu, Zoltán Szabó, Kenji Fukumizu, and Arthur Gretton. A linear-time kernel goodness-of-fit test. In Advances in Neural Information Processing Systems (NeurIPS) , pages 262-271, 2017.
- [23] Florian Kalinke, Marco Heyden, Georg Gntuni, Edouard Fouché, and Klemens Böhm. Maximum mean discrepancy on exponential windows for online change detection. Transactions on Machine Learning Research , 2025.
- [24] Nicolas Keriven, Damien Garreau, and Iacopo Poli. NEWMA: A new method for scalable model-free online change-point detection. IEEE Transactions on Signal Processing , 68:35153528, 2020.
- [25] Albert Y. Kim, Caren Marzban, Donald B. Percival, and Werner Stuetzle. Using labeled data to evaluate change detectors in a multivariate streaming environment. Signal Processing , 89(12): 2529-2536, 2009.
- [26] Katerina Kosta, Rebecca Killick, Oscar Bandtlow, and Elaine Chew. Dynamic change points in music audio capture dynamic markings in score. In International Society for Music Information Retrieval Conference (ISMIR) , 2017.
- [27] Katerina Kosta, Oscar F. Bandtlow, and Elaine Chew. MazurkaBL: score-aligned loudness, beat, and expressive markings data for 2000 Chopin Mazurka recordings. In International Conference on Technologies for Music Notation and Representation (TENOR) , pages 85-94, 2018.
- [28] Mehmet N. Kurt, Yasin Yilmaz, and Xiaodong Wang. Real-time nonparametric anomaly detection in high-dimensional settings. IEEE Transactions on Pattern Analysis and Machine Intelligence , 43(7):2463-2479, 2020.
- [29] Tze Leung Lai. Sequential changepoint detection in quality control and dynamical systems. Journal of the Royal Statistical Society: Series B (Statistical Methodology) , 57(4):613-644, 1995.
- [30] Tze Leung Lai. Information bounds and quick detection of parameter changes in stochastic systems. IEEE Transactions on Information Theory , 44(7):2917-2929, 1998.
- [31] Céline Lévy-Leduc and François Roueff. Detection and localization of change-points in highdimensional network traffic data. The Annals of Applied Statistics , 3(2):637-662, 2009.
- [32] Shuang Li, Yao Xie, Hanjun Dai, and Le Song. M-statistic for kernel change-point detection. In Advances in Neural Information Processing Systems (NeurIPS) , pages 3366-3374, 2015.
- [33] Shuang Li, Yao Xie, Hanjun Dai, and Le Song. Scan B -statistic for kernel change-point detection. Sequential Analysis , 38(4):503-544, 2019.
- [34] Feng Liu, Wenkai Xu, Jie Lu, Guangquan Zhang, Arthur Gretton, and Danica J. Sutherland. Learning deep kernels for non-parametric two-sample tests. In International Conference on Machine Learning (ICML) , pages 6316-6326, 2020.
- [35] Song Liu, Makoto Yamada, Nigel Collier, and Masashi Sugiyama. Change-point detection in time-series data by relative density-ratio estimation. Neural Networks , 43:72-83, 2013.
- [36] Gary Lorden. On excess over the boundary. Annals of Mathematical Statistics , 41:520-527, 1970.
- [37] Gary Lorden. Procedures for reacting to a change in distribution. Annals of Mathematical Statistics , 42:1897-1908, 1971.
- [38] Colin McDiarmid. On the method of bounded differences. Surveys in combinatorics , 141(1): 148-188, 1989.

- [39] Per August Jarval Moen. A general methodology for fast online changepoint detection. Technical report, 2025. https://arxiv.org/abs/2504.09573 .
- [40] George V. Moustakides. Optimal stopping times for detecting changes in distributions. The Annals of Statistics , 14(4):1379-1387, 1986.
- [41] Carlos Misael Madrid Padilla, Haotian Xu, Daren Wang, Oscar Hernan Madrid Padilla, and Yi Yu. Change point detection and inference in multivariate non-parametric models under mixing conditions. In Advances in Neural Information Processing Systems (NeurIPS) , 2023.
- [42] Oscar Hernan Madrid Padilla, Yi Yu, Daren Wang, and Alessandro Rinaldo. Optimal nonparametric multivariate change point detection and localization. IEEE Transactions on Information Theory , 68(3):1922-1944, 2021.
- [43] Ewan S. Page. Continuous inspection schemes. Biometrika , 41:100-115, 1954.
- [44] Moshe Pollak and David Siegmund. Sequential detection of a change in a normal mean when the initial value is unknown. The Annals of Statistics , 19(1):394-416, 1991.
- [45] Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. In Advances in Neural Information Processing Systems (NeurIPS) , pages 1177-1184, 2007.
- [46] Ya'acov Ritov. Decision theoretic optimality of the CUSUM procedure. The Annals of Statistics , 18(3):1464-1469, 1990.
- [47] Gaetano Romano, Idris A. Eckley, Paul Fearnhead, and Guillem Rigaill. Fast online changepoint detection via functional pruning CUSUM statistics. Journal of Machine Learning Research , 24 (81):1-36, 2023.
- [48] Gaetano Romano, Idris A. Eckley, and Paul Fearnhead. A log-linear non-parametric online changepoint detection algorithm based on functional pruning. IEEE Transactions on Signal Processing , 72:594-606, 2024.
- [49] Gordon J. Ross. Parametric and nonparametric sequential change detection in R: The cpm package. Journal of Statistical Software , 66:1-20, 2015.
- [50] Antonin Schrab, Benjamin Guedj, and Arthur Gretton. KSD aggregated goodness-of-fit test. In Advances in Neural Information Processing Systems (NeurIPS) , pages 32624-32638, 2022.
- [51] Antonin Schrab, Ilmun Kim, Benjamin Guedj, and Arthur Gretton. Efficient aggregated kernel tests using incomplete U-statistics. In Advances in Neural Information Processing Systems (NeurIPS) , pages 18793-18807, 2022.
- [52] David Siegmund. Sequential analysis: tests and confidence intervals . Springer, 2013.
- [53] Alexander Smola, Arthur Gretton, Le Song, and Bernhard Schölkopf. A Hilbert space embedding for distributions. In Algorithmic Learning Theory (ALT) , pages 13-31, 2007.
- [54] Bharath K. Sriperumbudur and Zoltán Szabó. Optimal rates for random Fourier features. In Advances in Neural Information Processing Systems (NeurIPS) , pages 1144-1152, 2015.
- [55] Bharath K. Sriperumbudur, Arthur Gretton, Kenji Fukumizu, Bernhard Schölkopf, and Gert Lanckriet. Hilbert space embeddings and metrics on probability measures. Journal of Machine Learning Research , 11(50):1517-1561, 2010.
- [56] Ingo Steinwart and Andreas Christmann. Support Vector Machines . Springer, 2008.
- [57] Nicolas Verzelen, Magalie Fromont, Matthieu Lerasle, and Patricia Reynaud-Bouret. Optimal change-point detection and localization. The Annals of Statistics , 51(4):1586-1610, 2023.
- [58] Haoyun Wang and Yao Xie. Sequential change-point detection: Computation versus statistical performance. Wiley Interdisciplinary Reviews: Computational Statistics , 16(1):e1628, 2024.
- [59] Song Wei and Yao Xie. Online kernel CUSUM for change-point detection. Technical report, 2022. https://arxiv.org/abs/2211.15070 .

- [60] Wei Biao Wu. Nonlinear system theory: another look at dependence. National Academy of Sciences of the United States of America , 102(40):14150-14154, 2005.
- [61] Ping Yang, Guy Dumont, and John M. Ansermino. Adaptive change detection in heart rate trend monitoring in anesthetized children. IEEE Transactions on Biomedical Engineering , 53 (11):2211-2219, 2006.
- [62] Yasin Yilmaz. Online nonparametric anomaly detection based on geometric entropy minimization. In IEEE International Symposium on Information Theory (ISIT) , pages 3010-3014, 2017.
- [63] Yi Yu. A review on minimax rates in change point detection and localisation. Technical report, 2020. https://arxiv.org/abs/2011.01857 .
- [64] Yi Yu, Oscar Hernan Madrid Padilla, Daren Wang, and Alessandro Rinaldo. A note on online change point detection. Sequential Analysis , 42(4):438-471, 2023.
- [65] Vadim Yurinsky. Sums and Gaussian vectors . Springer, 1995.
- [66] Wojciech Zaremba, Arthur Gretton, and Matthew B. Blaschko. B-test: A non-parametric, low variance kernel two-sample test. In Advances in Neural Information Processing Systems (NeurIPS) , pages 755-763, 2013.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction motivate and summarize the theoretical and experimental results that we establish in the article.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We make all our assumptions explicit in Assumption 1 and in the statements of the results. One theoretical difference to the offline setting is additionally elaborated right after Theorem 4. A runtime analysis is provided in Section 4.2, with a comparison to existing kernel-based approaches in Table 1. We elaborate limitations w.r.t. kernel choice in Section 7.

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

Justification: The assumptions on the kernel function are listed in Assumption 1 and each statement lists the additional assumptions that it requires. All our proofs are in the appendix, where we also collect the external statements that we use, ensuring self-completeness.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We include pseudo code describing an efficient implementation of our algorithm as Algorithm 1; the corresponding Figure 1 gives additional intuition. Further, we elaborate the experimental setup in Section 6, where we extend existing benchmarks, to simplify comparison and reproducibility. Additionally, all source code replicating our experiments is available online (see footnote 4) and in the supplement.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might

suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We include a link to the repository hosting our source code (see footnote 4) and also make it available in the supplement. The code contains scripts and additonal instructions for reproducing all results and figures.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/public/ guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We elaborate the experimental setup in Section 6, where we also specify all parameter choices for the proposed algorithm and the tested competitors. Our results extend known benchmarks.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We include the estimated 95% -error quantile in our runtime plots (Figure 2) but these are negligbly small. The remaining experiments show averages over 100 repetitions, respectively. Here, as is common for these experiments in the literature, we do not include error bars.

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

Justification: We make the resources used for our experiments explicit at the end of the introduction to Section 6. Explicit runtime results are in Figure 2.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in this work complies with the NeurIPS Code of Ethics. Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: Our proposed change detection algorithm does not have any direct societal impact. Guidelines:

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

Answer: [Yes]

Justification: We reference all competitors that we compare against in the main part of the paper; in particular, we summarize these with references in Table 1. The MNIST data set that we use is referenced in Section 6.2, HASC is referenced in Section A.3.2, and the MazurkaML data set is referenced in Section A.3.3.

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

Answer: [Yes]

Justification: We release all code replicating our experiments online (see footnote 4) and in the supplement. The code contains detailed instructions for reproducing all results in the 'README.md' file.

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
15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals

(or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or nonstandard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: This paper does not involve LLMs in any direct way.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Appendix

This appendix is structured as follows. We detail the extension of Online RFF-MMD to take a known or observed pre-change distribution into account in Appendix A.1. In particular, we show that in these settings tighter thresholds are possible. Appendix A.2 shows how to adapt our algorithm to detect multiple change points and gives corresponding guarantees. Additional numerical results of MMD-RFF are in Appendix A.3; we include numerical results w.r.t. our tighter thresholds in Appendix A.3.5. Appendix A.4 collects all our proofs, with auxiliary results in Appendix A.5 and external statements in Appendix A.6.

## A.1 Known or estimable pre-change distribution

In this section, we discuss practical extensions of Online RFF-MMD when additional information about the pre-change distribution is available. Specifically, in Section A.1.1, we show how the algorithm can be adapted to settings in which (i) one has access to historical data known to be from the pre-change distribution, or (ii) one knows the pre-change distribution exactly. In Section A.1.2, we describe how this additional information allows sharpening the thresholds proposed in Theorems 1 and 2 in the main text.

## A.1.1 Incorporating information of the pre-change distribution

To begin with, we consider the setting in which, for some ν ∈ N , historical data X -ν +1 , . . . , X 0 known to be from the (nonetheless unknown) pre-change distribution P is available. This setting has been studied in the literature from both the parametric [44] and non-parametric [59] perspectives. It is straightforward to extend the Online RFF-MMD stopping time defined in the main text to take advantage of the additional information. Intuitively, for each local test one may prepend the historical data to the block of data taken to be from the pre-change distribution. More formally, the following stopping time can be used:

<!-- formula-not-decoded -->

This stopping time can be implemented similarly to Algorithm 1, and such an implementation features the same time and space complexity as the original algorithm.

Next, we consider the setting in which the pre-change distribution P is known exactly. With this additional information, rather than performing local two sample tests, one may perform local one sample tests where the RFF approximation to the mean embedding of the data's empirical distribution is compared to the RFF approximation to the mean embedding of P . More formally, the following stopping time can be used:

<!-- formula-not-decoded -->

The exact knowledge of P permits a precise approximation or the exact computation of E X ∼ P [ˆ z K ( X ) | ω 1 , . . . , ω r ] , given ω 1 , . . . , ω r sampled from Λ . Again, this stopping time can be implemented similarly to Algorithm 1, enjoying the same time and space complexity as the original algorithm.

## A.1.2 Sharper thresholds through knowledge of the pre-change distribution

Access to additional information about the pre-change distribution paves the way to sharpening the thresholds proposed in Theorems 1 and 2. Indeed, although Theorem 4 suggests the thresholds proposed in the main paper are unimprovable up to constants, in practice these thresholds will be quite conservative as they are completely agnostic to the distribution of the data.

The main tool for proving Theorems 1 and 2 is Lemma 5, which controls the tail behavior of the tests performed by Online RFF-MMD under their respective local nulls. With additional knowledge of the pre-change distribution, Lemma 5 can be significantly sharpened by taking the second moment of the feature map into account. To that end, we have the following result.

Lemma 1. Given two independent samples { X 1 , . . . , X n } and { Y 1 , . . . , Y m } each with mutually independent entries drawn from some P ∈ M + 1 , for any ε &gt; 0 , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 1 allows the following improvements of Theorems 1 and 2.

Corollary 1. For any γ &gt; 1 , replacing the thresholds in Theorem 1 with the scale dependent thresholds

<!-- formula-not-decoded -->

it holds that E ∞ [ N ] ≥ γ where N is as defined in (8).

<!-- formula-not-decoded -->

Corollary 2. For any α ∈ (0 , 1) , replacing the thresholds in Theorem 2 with the scale dependent thresholds it holds that P ∞ ( N &lt; ∞ ) ≤ α where N is as defined in (8).

To put the above results in context, we detail Corollary 1. On large scales (large j -s), where detections tend to occur, the scale dependent thresholds behave approximately like ˜ σ √ 2 log(4 γ log 2 (2 γ )) , which differs from the threshold used in Theorem 1 by a factor of ˜ σ . Therefore, when ˜ σ is significantly smaller than 1 , the thresholds in (13) will be significantly smaller than those suggested by Theorem 1. As the kernel is assumed to be bounded from above by 1 , these smaller thresholds generally occur.

## A.2 Online detection of multiple change points

In this section, we explain how the algorithm in the main part can be extended to detect multiple change points in an online fashion. Indeed, consider the setting where data X 1 , X 2 , . . . are observed sequentially. There is a sequence of integers ( η k ) k ∈ N with η 0 = 0 and η k &lt; η k +1 for each k ∈ N , and a sequence of measures ( Q k ) k ∈ N each in M 1 + with Q k = Q k +1 for each k ∈ N such that for each t ∈ N the data are distributed according to

̸

<!-- formula-not-decoded -->

Consider running Algorithm 1 on this data stream using the threshold sequence given in Theorem 2. As soon as a change is detected, say at location n , we re-start the algorithm at X n +1 . However, the index on the threshold sequence is maintained. More precisely: having detected a change at time n , at the ( n + j ) -th iteration (for j ≥ 1 ), we construct local statistics over the interval { n +1 , . . . , n + j } according to Section 4.1, however the local statistics continue to be compared with the threshold λ n + j . We are able to prove the following guarantee on the change point locations recovered in this way:

Theorem 5. For any deterministic time T &lt; ∞ let M denote the number of η k -s taking values in { 1 , . . . , T } . Write δ k = min( η k -η k -1 , η k +1 -η k ) for k = 1 , . . . , M -1 and δ M = min( η M -η M -1 , T -η M ) for the effective sample size associated with the k -th change point location. For some α ∈ (0 , 1) , let ˆ M be the number of change points detected up to time T using the threshold sequence given in Theorem 2 and the procedure described above, and let ˆ η 1 , . . . , ˆ η ˆ M be their locations. If ∪ M k =1 supp( Q k ) ⊆ X for some compact set X ⊂ R d ,

<!-- formula-not-decoded -->

and the number of random Fourier features is chosen so that

<!-- formula-not-decoded -->

then with probability at least 1 -2 α it holds that ˆ M = M and

<!-- formula-not-decoded -->

Here h ( d, |X| , σ ) is as given in Theorem 3, and C 1 , C 2 , and C 3 are absolute constants independent of α and the sequences { δ k } k =1 ,...,M and { MMD K [ Q k , Q k +1 ] } k =1 ,...,M .

## A.3 Additional experiments

In this section, we summarize additional numerical results. The MNIST results with the MC threshold as in the main text are in Section A.3.1. We include results obtained without threshold estimation in Section A.3.4. A comparison of the distribution dependent and distribution-free bounds is in Section A.3.5. Numerical studies on the Human Activity Sensing Consortium (HASC) and on the MarzukaBL data sets are in Section A.3.2 and in Section A.3.3, respectively.

## A.3.1 MNIST data digits 4-9

In the following Figure 5, we collect the change detection performances of the algorithms of Table 1 on MNIST data, where the digit changes from 0 to one of 4 -9 ; the results on the other digits and the experimental setup are in Section 6.2. As in the corresponding experiment in the main part of this article, our proposed Online RFF-MMD algorithm consistently achieves the lowest expected detection delay, highlighting its good practical performance.

Figure 5: Average detection delay from MNIST digit 0 to digits 4-9 (left to right).

<!-- image -->

## A.3.2 HASC (Human Activity Sensing Consortium) data

This section collects our experiments on the Human Activity Sensing Consortium (HASC; available at http://hasc.jp/hc2011/ ) challenge 2011 data set, which is also considered in Liu et al. [35], Li et al. [33], Wei and Xie [59]. As in Wei and Xie [59], we consider the change from walking to staying of participant 101 . For preprocessing, we order the corresponding csv-files in the data set lexicographically, omitting the first 1 596 (detailed below) samples of walking , and then concatenating 100 walking observations and 100 staying observations to obtain a total of 10 data sets (with d = 3 ) with a single change point each. For obtaining the thresholds for the proposed Online RFF MMD, NewMA, Scan-B statistics, and OKCUSUM change detection algorithms, we proceed as elaborated in Section 6.1, that is, we use a Monte Carlo approach with α = 0 . 1 on all 10 walking data sets. As per Wei and Xie [59], for ScanB and OKCUSUM, we set the number of windows N = 14 and the window length w = 114 , implying that both algorithms receive 14 · 114 = 1 596 samples from walking upfront. Likewise, we process 1 596 elements with NewMA upfront. All kernel-based approaches use the (approximated) Gaussian kernel with the γ &gt; 0 parameter set by the median heuristic.

For the density ratio-based (i.e., non-kernel-based) RuLSIF algorithm-which showed the best performance on HASC in Liu et al. [35]-, we use the python changepoynt implementation and consider the l 2 -norm of each three-dimensional observation. 5 We then obtain the change scores for the full data set and select the point with the highest score as predicted change point, as done in Liu et al. [35, Section 4.2]. To match their setup, we set the window length to 50 , the number of windows to 10 , and α = 0 . 1 . Additionally, due to the large number of 'too early' cases reported in this setup, we introduce an offset and take the window length to be equal to the offset. In this setup, we consider the point with the highest score plus the offset as reported change point. A 'miss' can not occur, due to the scores reported to RuLSIF having at least one maximum. Table 2 collects our results.

Table 2: Comparison of change detection algorithms on 10 data sets derived from the HASC data set. 'Too early' refers to the number of times an algorithm reported a change point before the actual change occurred. 'Miss' reports cases in which no change was reported.

| Algorithm             |   Average delay |   Too early |   Miss |
|-----------------------|-----------------|-------------|--------|
| Online RFFMMD         |           21.86 |           2 |      1 |
| NewMA                 |           34.25 |           1 |      5 |
| ScanB                 |           31.2  |           0 |      0 |
| OKCUSUM               |           17.44 |           1 |      0 |
| RuLSIF                |            4.5  |           8 |      0 |
| RuLSIF (offset = 30 ) |           20.38 |           2 |      0 |
| RuLSIF (offset = 40 ) |           35    |           3 |      0 |
| RuLSIF (offset = 50 ) |           39.2  |           0 |      0 |

We emphasize that in the above case all algorithms except for the proposed one and RuLSIF receive 1 596 samples upfront, to use as a reference sample. Even though this fundamentally favors the kernel-based competitors, our proposed method achieves results that are comparable to those of the other kernel-based approaches and to RuLSIF.

## A.3.3 MazurkaBL data

In Kosta et al. [26], the authors found that change points in loudness information of Chopin's Mazurkas correspond to score positions having dynamic markings, tempo, or expression markings, among others. To further validate our algorithm's performance, we additionally run the proposed method on the 'M17-4' sample (illustrated in Figure 6) of pianist 'pid50534-05' of their MazurkaBL [27] data set with the goal of detecting these changes. Here, the dimensionality is d = 1 .

/brace189

<!-- image -->

sotto voce

Figure 6: An excerpt of Frédéric Chopin's Mazurka Op. 17 No. 4.

As in the main article (see Section 6), we approximate the Gaussian kernel with γ &gt; 0 set by the median heuristic. For obtaining the thresholds using Monte Carlo iterations, we slice the data along each annotated change point, where we consider the annotations provided by Kosta et al. [27] as ground truth, and compute the test statistics obtained by the proposed Online RFF MMD algorithm individually on each one. We then select the 1 -0 . 1 -quantile across all test statistics so obtained as threshold.

To detect change points, we again slice the loudness data, but now such that each slice contains precisely one change point, which yields a total of 25 samples with an average length of 1 561 . 32 .

5 The changepoynt library is available at https://github.com/Lucew/changepoynt .

We process each one of these with our proposed method and consider the first time the test statistic exceeds the threshold as 'change'. In total, our proposed method flags 10 change points too early, and, on the remaining 15 has an average detection delay of 73 . 67 , with a median detection delay of 64 . 0 .

While, when contrasting these results with the results obtained on the MNIST (Section A.3) and HASC (Section A.3.2) data sets, detecting changes in the selected loudness data of a Mazurka seems substantially more challenging, the proposed algorithm still manages to detect many changes with a relatively small delay.

## A.3.4 Distribution-free bound

In this section, we show the change detection performance of the proposed Online RFF-MMD algorithm if no pre-change sample is used to estimate the threshold. Instead, we use Theorem 1 to compute the distribution-free threshold sequence { λ n | n ∈ N } for a given target ARL. To obtain an EDD estimate, we sample and process 512 observations from MNIST digit 0 (pre-change) and 1 024 samples from digits 1-9 (post-change), respectively, averaging the detection delay over 100 repetitions. The results are in Figure 7. When comparing to Figure 4 and Figure 5, the figure shows that our method has an increased detection delay, which is due to the looser distribution-free bound (see Section A.3.5 for a numerical comparison). Still, except for the change to the digit 5 with a guaranteed ARL of 10 5 , Online RFF-MMD detects all changes reliably.

Figure 7: Average detection delay from MNIST digit 0 to digits 1-3, 4-6, 7-9 (top to bottom) with the distribution-free threshold sequence of Theorem 1.

<!-- image -->

## A.3.5 Threshold comparison

In this section, we compare the tightness of our thresholds in the offline two-sample testing setting. Specifically, we fix the level α = 0 . 01 and let P = Q = N (0 , 1) . We then approximate the 1 -α quantile of MMD ˆ K ( ˆ P n , ˆ Q n ) (with n = 1000 , ˆ K approximating the Gaussian kernel with r = 1000 RFFs, and γ &gt; 0 set by the median heuristic) by (i) obtaining new samples from P , Q and (ii) permuting a fixed sample from P , Q for 1 000 rounds. Figure 8 shows the respective histograms and the estimated quantiles along with the thresholds obtained by Lemma 1 and Lemma 5, respectively. As one expects, the figure shows that the variance estimate used in Lemma 1 allows to obtain a tighter bound, where we consider the resampling/permutation-based thresholds as ground truth. We emphasize that independent of the threshold used, the resulting test is consistent against any fixed alternative.

## A.4 Proofs

This section is dedicated to our proofs. The proof of Theorem 1 is in Appendix A.4.1, that of Theorem 2 is in Appendix A.4.2, that of Theorem 3 is in Appendix A.4.3, and that of our minimax result (Theorem 4) is in Section A.4.4. The tighter threshold detailed in Lemma 1 is proved in

Figure 8: Comparison of different thresholds for the acceptance region of the MMD two-sample test.

<!-- image -->

Appendix A.4.5 and we state the proof for our multiple change point detection result (Theorem 5) in Appendix A.4.6.

## A.4.1 Proof of Theorem 1

Proof. For ease of reading, for each n ≥ 2 and j = 0 , . . . , ⌊ log 2 ( n ) ⌋ -1 , put

By the law of total expectation, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, plugging (16) into (15) proves the desired result.

## A.4.2 Proof of Theorem 2

Proof. Write for each n ∈ N . Applying standard peeling arguments [64, 57], we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in line (18a), we apply a union bound and bound the resulting sum by its maximum. Then, applying Lemma 5 as was done in (16), we obtain that

P

∞

(

N &lt;

)

<!-- formula-not-decoded -->

∞

Finally using the facts that (i) exp ( -π 2 2 l / 2 ) ≤ αl -2 ( l +1) -1 2 -l for all l ∈ N and (ii) ∑ ∞ l =1 l -1 ( l + 1) -1 = 1 , we obtain that P ∞ ( N &lt; ∞ ) ≤ α .

## A.4.3 Proof of Theorem 3

Proof. We first observe that for any triplet of integers ( m,n,ν ) satisfying (i) m ≤ n and (ii) ν ≤ n/ 2 , given two samples

<!-- formula-not-decoded -->

with mutually independent entries taking values in some bounded set X ⊂ R d where the X 's are sampled from P and the Y 's and ˜ Y 's are sampled from Q for some P , Q ∈ M + 1 ( X ) not identical, for any ε &gt; 0 , it holds that

<!-- formula-not-decoded -->

where MMD K is as in (4) and MMD ˆ K is as defined in (7). To show (19), let the ˜ X 's below be sampled independently from P and introduce the quantities

<!-- formula-not-decoded -->

Note that by the reverse triangle inequality

<!-- formula-not-decoded -->

Consequently, using the above and by repeated applications of the triangle inequality, one has that

<!-- formula-not-decoded -->

For term (20a), applying Lemmas 2 and 3 together with the fact that the X 's and Y 's take values in some compact X ⊂ R d , we obtain that

<!-- formula-not-decoded -->

For the penultimate term in (20c), applying the bound

<!-- formula-not-decoded -->

for X 's sampled from P and Y 's sampled from Q , whose proof can be found for instance in Section A.2 of [14], together with the bound √ x + y ≥ ( √ 2 / 2 ) ( √ x + √ y ) for all x, y ≥ 0 , which holds due to the concavity of the square root, one has that

<!-- formula-not-decoded -->

Identical arguments together with the fact that m ≤ n implies that (2 m ) / ( m + n ) ≤ 1 give

<!-- formula-not-decoded -->

Therefore, combining (21), (23), and (24), rearranging, and applying the rough bound

<!-- formula-not-decoded -->

which holds for any x ∈ R , K ∈ N and any random variables Z 1 , . . . , Z K , we obtain that

<!-- formula-not-decoded -->

Note that we assume 0 ≤ K ( · , · ) ≤ 1 . Therefore, arguing as in Lemma 4, one can show that MMD as defined in (5) has the bounded differences property with constants (41). Hence, applying Theorem 6 to terms (25a) and (25b) one arrives at (19). Turning to the problem of interest we first make explicit the constants in Theorem 3:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, define the quantities

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that condition (9) guarantees that (26a) exists and it can be checked that C 2 &gt; 0 . Let j ∗ be the j appearing in (26b). Consequently, using (19) and the fact that k ∗ / 2 ≤ t k ∗ ≤ k ∗ , we obtain that

<!-- formula-not-decoded -->

where for typographical reasons we have put:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for j = 0 , . . . , ⌊ log 2 ( η + k ∗ ) ⌋-1 . Now(26a) together with the fact that λ 2 η + 10 √ 2 ≤ ( √ 50 + √ 6 ) × √ log (2 η/α ) for all α ∈ (0 , 1) and η ∈ N guarantees that the term on the right of the inequality in (27a) is no larger than C 2 × √ k ∗ MMD K [ P , Q ] . Hence, appealing to (10) and Theorem 7, we obtain that (27 a ) ≤ α/ 2 . Moreover, since for each k ≤ η it holds that with k ∗ defined as in (26a), we obtain that (27 b ) ≤ 4 × ( α/ 2 η ) 3 ≤ α/ 2 . With these facts in place the theorem is proved.

## A.4.4 Proof of Theorem 4

Proof. Let 1 = (1 , . . . , 1) T ∈ R d and 0 = (0 , . . . , 0) T ∈ R d , let δ x denote the Dirac measure (for any set A ∈ σ ( R d ) and any x ∈ R d , δ x ( A ) = 1 if x ∈ A and 0 otherwise), and let

<!-- formula-not-decoded -->

Therefore, for any P , Q ∈ M ∗ , making use of the symmetry of K , we have that

<!-- formula-not-decoded -->

Moreover, for any P , Q ∈ M ∗ , we also have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (30a) holds due to the bound x -1 x +1 ≤ log( x ) ≤ x -1 for x ≥ 1 , (30b) holds because p, q ∈ [1 / 4 , 3 / 4] , and (30c) holds due to the bound x ≤ 4 x 2 for x ≥ 1 / 4 . Combining (29) and (30c), and additionally making use of the shift invariance of K , we obtain that

<!-- formula-not-decoded -->

Therefore, putting C K = (1 / 2)(3 / 17) ( K ( 0 , 0 ) -K ( 1 , 0 )) for the constant in (12), using (31) along with the fact that M ∗ ⊂ M + 1 ( R d ) , we have that

<!-- formula-not-decoded -->

Consequently the theorem is proved if we can find absolute constants α 0 , β 0 ∈ (0 , 1) and pre- and post-change distributions P , Q ∈ M ∗ such that for all α ≤ α 0 it holds that

<!-- formula-not-decoded -->

To show (33), one can use a change of measure argument originally due to Lai [30]. In fact one can directly use the version of Lai's argument adapted to finite sample analysis by Yu et al. [64, Proposition 4.1]. For clarity of exposition, we repeat the argument below. The following holds for arbitrary P , Q ∈ M ∗ . For each n ∈ N let F n be the σ -field generated by { X i } n i =1 and let P ⊗ n be the restriction of the joint law to F n . We can write

<!-- formula-not-decoded -->

where, as in the main text, the subscripts indicate the time at which the change occurs. For a chosen α ∈ (0 , 1) and an arbitrary stopping time satisfying P ∞ ( N &lt; ∞ ) ≤ α introduce the events

<!-- formula-not-decoded -->

For the first event we have that

<!-- formula-not-decoded -->

where the first inequality is due to the definition of E 1 and the second inequality holds because the probability of N being finite when no change occurs is bounded from above by α . For the second

event we have that

<!-- formula-not-decoded -->

where in particular (35a) holds by subtracting t × KL ( Q ∥ P ) from both sides of the inequality and using the fact that for every t in the union it holds that t × KL ( Q ∥ P ) &lt; (1 / 2) log(1 /α ) , (35b) holds due to a union bound argument followed by an application of Hoeffding's inequality, and (35c) holds for all α ≤ α 0 where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since the above arguments do not depend on the stopping time N or the change point location η , the bounds (34) and (35c) together imply that R.H.S. of (32) ≥ 1 -2 α 1 / 4 0 , which proves the desired result.

## A.4.5 Proof of Lemma 1

Proof. It suffices to show that for any ϵ &gt; 0 it holds that

<!-- formula-not-decoded -->

which implies the stated result by using ± E | ω ˆ z K ( X ) , the triangle inequality, and a union bound. Here and in the following E | ω (resp. P | ω ) refers to the expectation (resp. probability) with ω fixed. To prove (36), we fix ω = ( ω 1 , . . . , ω r ) and let η i = ˆ z K ( X i ) -E | ω ˆ z K ( X ) . Then, all η i -s are zero mean w.r.t. P and i.i.d. Further, we have for any p ≥ 2 that

<!-- formula-not-decoded -->

where the last inequality follows by using the triangle inequality and ⟨ ˆ z K ( X 1 ) , ˆ z K ( X 1 ) ⟩ = 1 to obtain

Hence, we have the bound

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and the η i -s satisfy the Bernstein condition with B 2 = E | ω ∥ η 1 ∥ 2 and H = 2 . The application of Theorem 8 yields that for any ε &gt; 0

<!-- formula-not-decoded -->

To lift the conditioning, we observe that

<!-- formula-not-decoded -->

where the last inequality holds by the concavity of exp( -1 /x ) for x &gt; 0 and Jensen's inequality. Finally, we conclude the proof by using that

<!-- formula-not-decoded -->

where the boundedness of all terms allows exchanging the expectations by the Fubini-Tonelli theorem.

## A.4.6 Proof of Theorem 5

Proof. We first show that with probability at least 1 -α no local tests conducted on null intervals will cause the algorithm to wrongly declare a change. For each n ≤ T define k n = max { k ≤ M +1 | η k ≤ n } . With π n defined as in (17), introduce the event

<!-- formula-not-decoded -->

We therefore have that

<!-- formula-not-decoded -->

where (38a) follows by arguments similar to (18a) and (38b) holds due to Lemma 5. Therefore, if M = 0 we are done. We now tackle the case M &gt; 0 . We will show that when M &gt; 0 with probability at least 1 -2 α all true change points are detected and localized at the stated rate. We first make the constants in Theorem 5 explicit:

<!-- formula-not-decoded -->

Introduce also the event

<!-- formula-not-decoded -->

on which the random Fourier feature kernel ˆ K is close to K in sup norm sense. Note that by Theorem 7 the event (39) holds with probability at least 1 -α/ 2 . Introduce also the sequence of events

<!-- formula-not-decoded -->

such that on the event B k the k -th change point is detected and localized at the required rate. If ˆ M &lt; M we employ the convention ˆ η k = T for all k &gt; ˆ M . We now show that on (37) and (39) the event B 1 holds with probability at least T/ (2 α ) . Introduce the quantity

<!-- formula-not-decoded -->

To ease the notation put I ∗ 1 = 1 : η 1 , J ∗ 1 = ( η 1 +1) : n ∗ 1 , and V ∗ 1 = 2 j ∗ 1 ( n ∗ 1 -2 j ∗ 1 ) n ∗ 1 . Observe that

<!-- formula-not-decoded -->

Where (40a) holds by the triangle inequality and arguments similar to (21), and (40b) holds by arguments similar to (22) together with the fact that on (39) we will have that

<!-- formula-not-decoded -->

Using the above we therefore have that

<!-- formula-not-decoded -->

where the final inequality follows from the definition of C 3 , the bounded difference property of MMD K , and Theorem 6. By the definition we must have that 2 j ∗ 1 ≤ δ 1 / 2 , therefore on the event A 1 the first change point is detected and the algorithm starts at most mid way between η 1 and η 2 . Identical arguments to those above therefore give that, on the events A 1 , A 2 , and B 1 the event B 2 holds with probability at least 1 -α/ (2 T ) and having detected the second change point the algorithm re-starts at most mid way between η 2 and η 3 . By induction and a union bound argument, on the events (37) and (39) the events B 1 , . . . B M hold with probability at least 1 -Mα/ (2 T ) ≥ 1 -α/ 2 . Since (37) and (39) jointly hold with probability 1 -3 α/ 2 we are done.

## A.5 Auxiliary results

In this section, we collect a few auxiliary results. Besides establishing useful bounds on real numbers in Lemma 2 and Lemma 3, we show that the bounded differences property of RFF-MMD (Lemma 4) leads to its exponential concentration (Lemma 5). The latter is one of the key ingredients for deriving our threshold sequences elaborated Section 4.3.

Lemma 2. For any x, y &gt; 0 it holds that

<!-- formula-not-decoded -->

and, moreover, both inequalities are tight.

Proof. We first note that

<!-- formula-not-decoded -->

For the lower bound we use the fact that 1 + min( x,y ) max( x,y ) ≤ 2 , where equality holds when x = y . For the upper bound, we use that 1 + min( x,y ) max( x,y ) ≥ 1 , where equality holds in the limit when, for instance, x is fixed and y → + ∞ .

Lemma 3. For x, y &gt; 0 it holds that ∣ ∣ √ x - √ y ∣ ∣ ≤ √ | x -y | .

Proof. When x = y the statement is trivially true. When x = y it holds that

<!-- formula-not-decoded -->

̸

Lemma 4. The RFF-MMD as defined in (7) between two empirical measures composed respectively of n and m sample points is a function mapping from ( R d ) m + n → R . This function has the bounded differences property with constants

<!-- formula-not-decoded -->

Proof. Recall that if K : R d × R d → R is the reproducing kernel for some RKHS H K , for any P , Q ∈ M + 1 , one has that

<!-- formula-not-decoded -->

Note that ˆ K : R d × R d → R as defined in (6) is the reproducing kernel for an RKHS H ˆ K whose elements are vectors in R 2 r . Introduce the set

<!-- formula-not-decoded -->

Let MMD ˆ K ( x 1: n , y 1: m ) (˜ x i ′ ) stand for (7) with inputs { x 1 , . . . , x n } and { y 1 , . . . , y m } with the i ′ -th x replaced by ˜ x i ′ . We therefore have that

<!-- formula-not-decoded -->

where we used the reverse triangle inequality, the reproducing property, CBS for obtaining the supremum over a unit ball and that, for any x ∈ R d one has that

<!-- formula-not-decoded -->

The same calculations can be applied to MMD ˆ K ( x 1: n , y 1: m ) (˜ y j ′ ) . This proves the desired result.

Lemma 5. Given two independent samples { X 1 , . . . , X n } and { Y 1 , . . . , Y m } , each with mutually independent entries drawn from some P ∈ M + 1 , for any ε &gt; 0 , it holds that

<!-- formula-not-decoded -->

Proof. It is an immediate consequence of Lemma 4 and Theorem 6 that for any ε ′ &gt; 0

<!-- formula-not-decoded -->

Moreover, arguing as in the last step of the proof of Proposition 4 in [23] gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality follows from Jensen's inequality. Consequently, setting ε ′ = ε √ n + m nm , plugging (43) into (42), and integrating over the ω -s with respect to the product measure Λ ⊗ r := Λ ⊗··· ⊗ Λ yields the desired result.

## A.6 External statements

In this section, we collect the external statements that we use, to ensure self-completeness. Theorem 6 recalls McDiarmid's inequality from Boucheron et al. [5, Section 6.1], which is also known as bounded differences inequality [38]. Theorem 7 is about the concentration of random Fourier features and part of the proof of Sriperumbudur and Szabó [54, Theorem 1]. We recall the concentration result Yurinsky [65, Theorem 3.3.4] on random variables taking values in a separable Hilbert space in Theorem 8.

Theorem 6 (Bounded differences inequality) . Let X be a measurable space. A function f : X n → R has the bounded difference property for some constants c 1 , . . . , c n if, for each i = 1 , . . . , n ,

<!-- formula-not-decoded -->

Then, if X 1 , . . . , X n is a sequence of independently distributed random variables and (44) holds, putting Z = f ( X 1 , . . . , X n ) and ν = 1 4 ∑ n i =1 c 2 i for any t &gt; 0 , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 7 (RFF exponential concentration) . Let ˆ K be defined as in (6). Let X be a proper subset of R d and denote by |X| its Lebesgue measure. For any t &gt; 0 , it holds that

where σ 2 = ∫ R d ∥ ω ∥ 2 2 dΛ( ω ) and

<!-- formula-not-decoded -->

Theorem 8 (Hilbert space Bernstein inequality) . Let X 1 , . . . , X n be a sequence of zero mean independent random variables taking values in a real and separable Hilbert space X with inner product ⟨· , ·⟩ and norm ∥·∥ = √ ⟨· , ·⟩ . Write S ∗ n = sup m ≤ n ∥ X 1 + · · · + X m ∥ . If the random variables satisfy the moment condition

<!-- formula-not-decoded -->

for some constants B &gt; 0 and H &gt; 0 , then for any x &gt; 0 it holds that

<!-- formula-not-decoded -->