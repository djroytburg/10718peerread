## Nearly-Linear Time Private Hypothesis Selection with the Optimal Approximation Factor

Maryam Aliakbarpour 1 , 2 , Zhan Shi 1 , Ria Stevens 1 , Vincent X. Wang

1

1 Rice University, Department of Computer Science 2 Ken Kennedy Institute {maryama, zs50, ria.stevens, vw12}@rice.edu

## Abstract

Estimating the density of a distribution from its samples is a fundamental problem in statistics. Hypothesis selection addresses the setting where, in addition to a sample set, we are given n candidate distributions-referred to as hypotheses -and the goal is to determine which one best describes the underlying data distribution. This problem is known to be solvable very efficiently, requiring roughly O (log n ) samples and running in ˜ O ( n ) time. The quality of the output is measured via the total variation distance to the unknown distribution, and the approximation factor of the algorithm determines how large this distance is compared to the optimal distance achieved by the best candidate hypothesis. It is known that α = 3 is the optimal approximation factor for this problem. We study hypothesis selection under the constraint of differential privacy . We propose a differentially private algorithm in the central model that runs in nearly-linear time with respect to the number of hypotheses, achieves the optimal approximation factor, and incurs only a modest increase in sample complexity, which remains polylogarithmic in n . This resolves an open question posed by [Bun, Kamath, Steinke, Wu, NeurIPS 2019]. Prior to our work, existing upper bounds required quadratic time.

## 1 Introduction

The task of accurately estimating the underlying probability distribution that generates a dataset is a fundamental theoretical problem in statistical inference with broad applicability in practical data analysis. A growing concern in modern data analysis is preserving the privacy of the individuals whose data informs these estimations, specifically when dealing with sensitive information. Differential privacy (DP) has emerged as a widely adopted standard in privacy-preserving data analysis [DMNS06a] and is currently employed by major entities, such as Google [EPK14], Apple [Dif17], and the U.S. Census Bureau [Abo18]. See Section A.1 for more examples.

In this paper, we study a specific instance of distribution estimation under the constraint of differential privacy, referred to as Hypothesis Selection . In this problem, we are given a finite collection of n candidate distributions H := { H 1 , H 2 , . . . , H n } , known as hypotheses, and a dataset of i.i.d. samples drawn from an unknown distribution P . The goal is to select a hypothesis in H that well-approximates the true data distribution.

Along line of research has studied hypothesis selection in the non-private setting [Yat85, DL96, DL97, DL01, MS08, DK14, SOAJ14, AJOS14, AFJ + 18, BBK + 21, ABS23, AAC + 23, AAC + 24, ABS24]. These algorithms' performances are evaluated across three key aspects: i) sample complexity, ii) time complexity, and iii) approximation factor. Early work has shown that hypothesis selection admits highly sample-efficient algorithms, requiring only Θ(log n ) samples, a logarithmic dependence on the number of hypotheses, and no dependence on the domain size of distributions [Yat85, DL96, DL97, DL01]. The sample efficiency is achieved by only needing to estimate probabilities of O ( n 2 ) special sets, called Scheffé sets, according to P (see Equation (1) for a definition). Moreover, several

works [DK14, AFJ + 18, ABS23, ABS24] have shown that one can find a valid hypothesis in roughly linear time in n . Recently, [AAC + 24] characterized the statistical-computational trade-off of the problem when the distributions have finite domain.

Another key aspect is the accuracy of the selected hypothesis, measured by the total variation distance to the true distribution. The approximation factor (denoted by α ) measures this distance relative to the minimum distance between P and a hypothesis in H . Anotable lower bound established in [BKM19] shows that achieving α &lt; 3 is impossible unless the number of samples is polynomial in the domain size of P . In an interesting development, Aliakbarpour et al. [ABS24] recently proposed an algorithm that simultaneously achieves a logarithmic sample complexity, nearly-linear time complexity, and the optimal approximation factor α = 3 , representing a compelling performance in all three critical aspects of this problem.

Despite this desirable performance in the non-private setting, the state of the art in the private setting falls short of optimal performance. A naive privatization of Scheffé estimates via Laplace noise leads to an O ( n 2 ) sample complexity [BKSW19]. More advanced techniques, including the work of Bun et al. [BKSW19], offer better sample complexity (logarithmic) but suffer from quadratic time and suboptimal approximation. While Aden-Ali et al. [AAAK21] makes progress on accuracy and proof simplicity, their algorithm remains computationally expensive with a quadratic time complexity.

These limitations naturally lead us to address an open question, first raised in part by Bun et al. [BKSW19]: Does there exist an ideal private hypothesis selection algorithm that offers logarithmic sample complexity, nearly-linear time complexity, and the optimal approximation factor? We present a significant step forward towards this ideal: our algorithm achieves nearly-linear time complexity and the optimal approximation factor with a polylogarithmic sample complexity (which, while not logarithmic, is still considered a modest dependence on the number of hypotheses). A formal description of our results can be found in Section 1.2.

Applications: Estimating data distributions is a central component of many scientific tasks, such as estimating species abundance in ecology or analyzing survey results in the social sciences. Hypothesis selection describes a broadly applicable scenario where we can form a finite set of interpretable, noisefree, or otherwise manageable distributions as our candidate hypotheses, and we aim to approximate the potentially noisy and complex unknown distribution with one of the so-called 'nicer' candidates (e.g., modeling customer arrival time with Poisson processes).

One notable theoretical application of hypothesis selection is agnostic learning of a parametric class of distributions (e.g., mixtures of Gaussians [SOAJ14, DK14, ABM18, ABH + 20], and junta distributions [ABR16]) via the cover method . The approach is to first select a representative set of parametric distributions, and hypothesis selection then identifies the closest approximation in this set. For a survey, see Diakonikolas [Dia16].

## 1.1 Problem setup

Let P denote an unknown distribution over a domain X , and let H := { H 1 , H 2 , . . . , H n } be a set of n public and known distributions over X . We define OPT to be the total variation distance of P and the closest hypothesis in H .

We seek to design a semi-agnostic proper learner such that for every H and P , the algorithm outputs a hypothesis ˆ H ∈ H such that the total variation distance between ˆ H and P is within α -times OPT plus an additive error parameter σ , which can be made arbitrarily small with sufficiently many samples.

Weassume the standard access model of [ DL01 ] , where an algorithm accesses distributions by making queries of the following types:

1. The algorithm can draw i.i.d. samples from the unknown distribution P .
2. The algorithm can compare the PDFs of any two known distributions H i , H j at a given point x ∈ X . Specifically, it can ask if H i ( x ) &lt; H j ( x ) . This is equivalent to determining if x is in the Scheffé set of H i and H j (defined in Equation 1).
3. The algorithm can query the probability mass of the Scheffé set of any two known distributions. Precisely, it can ask for H i ( S i,j ) for all H i , H j ∈ H .

More formally, we have:

Definition 1.1 (Proper learner for private hypothesis selection) . Let α &gt; 0 , and let A be an algorithm with input parameters ϵ, β, σ ∈ (0 , 1) , sample access to an unknown distribution P , and query access to a finite class of n hypotheses H = { H 1 , H 2 , . . . , H n } (according to the access model described above). We say A is an ( α, ϵ, β, σ ) -proper learner for the private hypothesis selection problem if:

1. A is ϵ -differentially private in the central model (defined in Definition 1.2) with respect to the samples drawn from P .
2. A outputs ˆ H ∈ H such that with probability at least 1 -β ,

<!-- formula-not-decoded -->

We call α the approximation factor, ϵ the privacy parameter, β the confidence parameter, and σ the error parameter.

Remark 1. As mentioned in [DK14, ABS24], the third type of query in the standard access model can be relaxed. Our algorithms only need estimates , rather than exact values, of the probability masses of the Scheffé sets. Assuming all estimates of H i ( S i,j ) are accurate to an additive error of O ( σ ) with high probability, the analysis of our algorithms remain essentially unchanged. These estimates could be obtained by sampling from each H i or numerically integrating density functions when analytic forms are available.

## 1.2 Our result

We present an algorithm for private hypothesis selection with the following guarantee:

Theorem 2 (Informal version of Theorem 3) . For every ϵ, β, σ ∈ (0 , 1) , Algorithm 1 is an ( α = 3 , ϵ, β, σ ) -proper learner for the private hypothesis selection problem that uses s = Θ(log 3 ( n/β ) / ( β 2 σ 2 ϵ )) samples and runs in time ˜ Θ( n/ ( β 4 σ 3 ϵ )) .

Our result is the first algorithm for private hypothesis selection that runs in nearly-linear time, resolving an open question first posed by [BKSW19] about the existence of such an algorithm. In addition, we also maintain the optimal approximation factor α = 3 . However, our algorithm introduces an additional factor of O (log 2 n / σ ) in the sample complexity compared to existing work on private hypothesis selection. We summarize this tradeoff and compare with existing private algorithms in Table 1.

The overhead in the sample complexity stems from our privatization strategy that is informed by the structure of our algorithm. Similar to most time efficient algorithms in previous works, our algorithm consists of multiple interdependent components, where each component directs us to focus only on a small set of Scheffé estimates, as opposed to computing all of them. While these interdependencies allow us to achieve a highly time-efficient algorithm, they make direct privatization of the final output analytically very challenging. Rather than attempting to analyze the privacy loss of the full computation, we enforce differential privacy at each component of the algorithm, resulting in the sub-optimality of our sample complexity. Despite this added overhead, our algorithm still maintains sample complexity polylogarithmic in number of hypotheses with no dependence on the domain size.

Our algorithm also has a polynomial dependence on 1 /β , contrasting with the typical log(1 /β ) dependence on the confidence parameter that arises in many learning theory problems when amplifying a result (e.g., by taking an 'average' of the results of running log(1 /β ) many runs of an algorithm). However, in hypothesis selection, this amplification process is unlikely to succeed without a significant increase in α because it would require running hypothesis selection twice , which would push the total approximation factor to at least α = 9 . Achieving a polylogarithmic dependence on β is a difficult task, even in the absence of privacy constraints; both [ABS23, ABS24] present nearly-linear time algorithms with α &lt; 9 but suffer from similar polynomial dependencies on 1 /β .

Open directions: This leaves two open directions for further work: i) Does there exist a nearlylinear time algorithm that uses O ( log n σ 2 + log n σϵ ) samples in terms of its dependence on σ, ϵ ? ii) Can the dependence on the confidence parameter β be improved to O (log(1 /β )) while maintaining nearly-linear runtime?

Table 1: Summary of past hypothesis selection results under central DP

| Result                                        | α    | Sample complexity ( s )        | Time complexity   |
|-----------------------------------------------|------|--------------------------------|-------------------|
| Private Scheffé tournament [BKSW19] (Thm 3.6) | 9    | O ( log n σ 2 + n 2 log n σϵ ) | O ( n 2 · s )     |
| [BKSW19] (Thm 3.5)                            | > 54 | ˜ O ( log n σ 2 + log n σϵ )   | ˜ O ( n 2 · s )   |
| Minimum distance estimate [AAAK21] (Thm 2.24) | 3    | O ( log n σ 2 + log n σϵ )     | O ( n 2 · s )     |
| This work                                     | 3    | O ( log 3 n σ 2 ϵ )            | ˜ O ( n · s/σ )   |

Significance of the approximation factor: One may argue that it is possible to improve the accuracy by decreasing OPT by selecting more candidate hypotheses in H , as opposed to decreasing the approximation factor α . However, as mentioned in [ ABS24 ] , this is not feasible in many spaces, especially multivariate distributions, without substantially increasing the size of H . For instance, in the cover method, we require H to cover the space of a parametric class of distributions with a γ -cover, where each distribution in the class is at most γ away from an element in H , enforcing OPT &lt; γ . As an example, the mixture of k -Gaussians in [SOAJ14] requires O ( γ -3 k ) distributions to create a γ -cover, making algorithms with high approximation factor more time-consuming.

## 1.3 Related work

Most of the previous work on hypothesis selection falls under two approaches: i) tournament-based algorithms that compare pairs of hypotheses based on their Scheffé estimates, and then perform a series of comparisons to find a final winner ii) the minimum distance estimate (MDE) algorithms that compute an approximate distance based on the Scheffé estimates for each hypothesis, and then select the hypothesis with the minimum approximate distance.

The Scheffé tournament algorithm, first proposed by Devroye and Lugosi [DL01], runs in O ( n 2 · s ) time and achieves an approximation factor of α = 9 . Other works in this line include [ DK14 , AJOS14 , AFJ + 18 , AAC + 23 , ABS23 ] . Algorithms that follow the tournament structure typically exhibit a high approximation factor. Moreover, privatizing such algorithms raises extra challenges because changing a single data entry might affect the result of every comparison, thus greatly increasing sensitivity.

The other approach [DL01, MS08, ABS24] for non-private hypothesis selection is based on the minimum distance estimate (MDE) method introduced in [DL01], which has been the only type of algorithm that achieves the optimal approximation factor α = 3 . Mahalanabis and Stefankovic [MS08] later improved the initial O ( n 3 · s ) runtime of [DL01] to O ( n 2 · s ) , as well as proposed a nearlylinear time algorithm that requires exponential pre-processing. Aliakbarpour et al. [ABS24] recently demonstrated that the optimal approximation factor α = 3 could be achieved with a nearly-linear time algorithm.

For an extended version of the related work, see Appendix A.

## 1.4 Background: differential privacy

Differential privacy: We adopt the central model of DP [DMNS06b], where a sensitive dataset is given to a trusted data curator who performs the algorithm and publicly publishes the outcome. Differential privacy protects each entry in the dataset from an adversary who observes the outcome.

Adataset D = [ x 1 , . . . , x s ] ∈ X ⊗ s is a collection of s i.i.d. samples from an unknown distribution P . The Hamming distance between two datasets D and D ′ is defined as the number of differing entries and denoted as Ham ( D,D ′ ) . We consider an algorithm to be private with respect to the samples drawn from P if it satisfies the following definition:

Definition 1.2 (Pure differential privacy) . Let ϵ &gt; 0 . An algorithm A is ϵ -differentially private if for all measurable subsets S ⊆ Range( A ) and D,D ′ ∈ X ⊗ s such that Ham ( D,D ′ ) = 1 :

<!-- formula-not-decoded -->

Standard methods for calibrating noise in DP rely on the concept of sensitivity:

Definition 1.3 (Sensitivity) . Let f : X ⊗ s → R be a function. Then, ∆( f ) denotes the sensitivity of f and is defined by:

<!-- formula-not-decoded -->

Exponential mechanism: A widely used algorithm in DP is the exponential mechanism. This mechanism relies on a real-valued utility function u that maps a dataset D ∈ X ⊗ s and a candidate output H j ∈ H to a real-valued score, quantifying the 'quality' of H j with respect to D . Outputs with higher utility are more likely to be selected.

Definition 1.4 (Exponential mechanism [MT07, DR + 14]) . Given a utility function u : X ⊗ s ×H → R with sensitivity ∆( u ) , the exponential mechanism selects an element H j ∈ H with probability proportional to exp ( ϵu ( D,H j ) 2∆( u ) ) .

Fact 1.5 ([MT07]) . The exponential mechanism is ϵ -differentially private.

We also have the following utility guarantee of the exponential mechanism:

Fact 1.6 (Corollary 3.12 of [DR + 14]) . Let u : X ⊗ s ×H→ R be a utility function with sensitivity ∆( u ) . Fix a dataset D ∈ X ⊗ s . Let H ∈ H denote the output of the exponential mechanism with parameter ϵ and utility function u . Then, for any β ∈ (0 , 1) :

<!-- formula-not-decoded -->

Sparse vector technique [DNR + 09, DR + 14, LSL16]: Given a numerical statistic g : X → R , a threshold query counts the number of entries x ∈ D such that g ( x ) is above or below a fixed cutoff. We will employ the sparse vector technique (SVT) [DNR + 09], which enables processing a large number of threshold queries while incurring a privacy cost only for the small subset of queries whose values exceed a specified threshold. Dwork et al. [DNR + 09] presents a simple ϵ -differentially private algorithm ABOVETHRESHOLD that takes in a stream of queries and identifies the first meaningful query above a predefined threshold while privately ignoring queries that fall below the threshold.

Composition and post-processing: Two properties make DP particularly well-suited for modular algorithm design. Composition bounds the cumulative privacy loss after performing multiple differentially private subroutines, and post-processing ensures that no further transformation of the output of a differentially private algorithm can further degrade the privacy guarantees. We state the following theorems:

Fact 1.7 (Composition, Theorem 3.14 of [DR + 14]) . Let A 1 , . . . , A k be algorithms that access the same dataset D ∈ X ⊗ s , and suppose each A i is ϵ i -differentially private. Let A be the composed algorithm of A 1 , . . . , A k . Then, A is ∑ k i =1 ϵ i -differentially private.

Fact 1.8 (Post-processing, Proposition 2.1 of [DR + 14]) . Let A be an ϵ -differentially private algorithm. Let g be a (possibly random) mapping. Then, g ◦ A is ϵ -differentially private.

## 2 Preliminaries

We begin by introducing notation and a list of key definitions in Section 2.1. These concepts are expanded upon in later sections. Section 2.2 presents the framework of minimum distance estimate algorithms, and Section 2.3 describes the nearly-linear time optimization proposed by [ABS24].

## 2.1 Notation and basic concepts

Notation and basic definitions: For n ∈ Z + , we use [ n ] to denote the set { 1 , . . . , n } . For an arbitrary probability distribution P over X , let P ( x ) be the PDF of P at x ∈ X . For a measurable subset S ⊆ X , let P ( S ) be the probability mass of the set S according to P . We use X ∼ P to denote a random variable X that is drawn from the distribution P . Let ∥ P 1 -P 2 ∥ TV := sup S ⊆X | P 1 ( S ) -P 2 ( S ) | be the total variation distance between two distributions P 1 and P 2 . For a sample space Ω

and an event E ⊆ Ω , the indicator function 1 E evaluates to 1 when E occurs and 0 otherwise. We use the standard O, Ω , Θ notation for asymptotic functions, as well as ˜ O ( x ) , ˜ Ω( x ) , ˜ Θ( x ) to indicate additional polylog ( x ) factors. A dataset D = [ x 1 , . . . , x s ] ∈ X ⊗ s is a collection of s i.i.d. samples from an unknown distribution P .

Optimal hypothesis: We use H i ∗ to indicate a hypothesis in a finite hypothesis class H that achieves the smallest total variation distance to P , which is called OPT . If there are ties, then we pick one such hypothesis as H i ∗ arbitrarily. Therefore, OPT := min H ∈H ∥ H -P ∥ TV = ∥ H i ∗ -P ∥ TV.

Scheffé sets: For every pair of hypotheses H i , H j ∈ H , we define the Scheffé set of H i and H j as:

<!-- formula-not-decoded -->

It is not difficult to show that the difference of probability masses of two distributions on the Scheffé set of H i and H j is precisely the total variation distance between H i and H j :

<!-- formula-not-decoded -->

Semi-distances: We adopt the definitions of semi-distances from [ABS24], building on earlier work in [DL01, MS08]. For every pair of hypotheses H i , H j ∈ H , the semi-distance w i ( H j ) is the distance between H j and P measured on the Scheffé set of H i and H j ; that is, w i ( H j ) := | H j ( S i,j ) -P ( S i,j ) | . The maximum semi-distance of H j is defined as W ( H j ) := max H i ∈H w i ( H j ) .

Empirical semi-distances: Given a measurable set S ⊆ X , we define the empirical distribution ˆ P of a dataset D = [ x 1 , . . . , x s ] as ˆ P ( S ) := 1 s ∑ s k =1 1 x k ∈ S . The empirical semi-distance ˆ w i ( H j ) is similarly defined as ˆ w i ( H j ) := | H j ( S i,j ) -ˆ P ( S i,j ) | , where ˆ P is based on the observed samples drawn from P . Observe that ˆ w i ( H j ) is an estimation of w i ( H j ) . For a given hypothesis H j ∈ H and a set of hypotheses A ⊆ H , the proxy distance ˆ W A ( H j ) is defined as ˆ W A ( H j ) := max H i ∈ A ˆ w i ( H j ) . Here, A is a set of hypotheses that is updated throughout the algorithm to improve ˆ W A ( H j ) as an approximation for W ( H j ) .

Refined access model: As in prior work [DL01, MS08, ABS24], our algorithms will have query access to ˆ w i ( H j ) , which follows from the standard access model. In the lemma below, we show that ˆ w i ( H j ) is within some σ ′ of w i ( H j ) with sufficiently large samples, where σ ′ can be taken to be Θ( σ ) . The time complexity of our algorithms is measured in number of queries to ˆ w i ( H j ) , and each query takes Θ( s ) to compute.

Lemma 2.1. Let β, σ ′ ∈ (0 , 1) . If the number of samples s ≥ 1 2 σ ′ 2 log(2 n/β ) , then with probability at least 1 -β , the empirical semi-distances are accurate to an additive error of σ ′ :

<!-- formula-not-decoded -->

Proof. The estimates ˆ w i ( H j ) = | H j ( S i,j ) -ˆ P ( S i,j ) | can be computed via sampling from P and counting the fraction of samples in the Scheffé set of H i and H j , which is ˆ P ( S i,j ) . By a reverse triangle inequality:

<!-- formula-not-decoded -->

Therefore, by a standard application of the Hoeffding and union bounds, we can estimate each w i ( H j ) using 1 2 σ ′ 2 log(2 n/β ) samples.

Lifting: Let H i , H j ∈ H . We define the lift value H i induces on H j by ˆ w i ( H j ) -ˆ W A ( H j ) . In other words, the lift value quantifies the improvement of the proxy distance ˆ W A ( H j ) if H i is added to the set of hypotheses A used to compute ˆ W A ( H j ) = max H k ∈ A w k ( H i ) . For some σ ′ ∈ (0 , 1) , we say that H i σ ′ -lifts H j if the lift value is at least σ ′ , or equivalently that H i lifts H j by at least σ ′ .

Prompting: Let Q be a distribution over H . For two parameters σ ′ , η ∈ (0 , 1) , we say a hypothesis H i ∈ H is ( σ ′ , η ) -prompting with respect to Q if H i σ ′ -lifts a random hypothesis H j sampled from Q with probability at least η . In other words, we have:

<!-- formula-not-decoded -->

We now consider an empirical analog for a list of hypotheses K that is sampled from Q . Let K = [ H j 1 , . . . , H j t ] be a list of t hypotheses in H . For two parameters σ ′ , η ∈ (0 , 1] , we say a hypothesis H i ∈ H is ( σ ′ , η ) -empirical-prompting with respect to K if H i σ ′ -lifts at least an η -fraction of the hypotheses in K . In other words, we have:

<!-- formula-not-decoded -->

## 2.2 Background: minimum distance estimate

In this section, we sketch the key ideas behind previous approaches that use a minimum distance estimate [DL01, MS08]. For a more detailed treatment, see Section 3.1 of [ABS24].

Our algorithms rely on computations of ˆ w i ( H j ) that approximate the true semi-distances w i ( H j ) . For clarity, we will assume in this section that the approximations ˆ w i ( H j ) are exact by Lemma 2.1. Observe that w i ( H j ) provides a lower bound for ∥ H j -P ∥ TV: specifically, w i ( H j ) = | H j ( S i,j ) -P ( S i,j ) | ≤ sup S ⊆X | H j ( S ) -P ( S ) | = ∥ H j -P ∥ TV. Thus, we can view w i ( H j ) as an attempt to lower bound ∥ H j -P ∥ TV. In particular, if we discover that w i ( H j ) is large, then this suggests that H j is far from P . However, the inverse is not true: when w i ( H j ) is small, this does not imply that H j is close to P .

Fortunately, when the particular semi-distance w i ∗ ( H j ) is small, we can upper bound ∥ H j -P ∥ TV. The semi-distance w i ∗ ( H j ) has the following property: if w i ∗ ( H j ) ≤ OPT , then H j satisfies ∥ H j -P ∥ TV ≤ 3 · OPT , making H j a valid hypothesis to output. This follows from repeated applications of the triangle inequality:

<!-- formula-not-decoded -->

An issue with using this metric ( w i ∗ ( H j ) ≤ OPT ) is that we know neither the optimal hypothesis H i ∗ nor OPT . This is remedied by minimizing a maximum semi-distance W ( H j ) := max H i ∈H w i ( H j ) . Let ˆ H be the hypothesis that minimizes W ( H j ) over all hypotheses in H . Then, we claim that ˆ H satisfies the desired property w i ∗ ( ˆ H ) ≤ OPT , implying that ∥ ˆ H -P ∥ TV ≤ 3 · OPT as in the above discussion. This follows from:

<!-- formula-not-decoded -->

The first inequality follows from W ( ˆ H ) being a maximum over all semi-distances of ˆ H . The second inequality follows from minimality of W ( ˆ H ) over all other hypotheses. The last inequality follows from the fact that every semi-distance measured against the optimal hypothesis is itself bounded by OPT . Therefore, we reduce the problem of hypothesis selection to finding a ˆ H that minimizes W ( ˆ H ) .

## 2.3 Background: approximating the maximum semi-distance

The MDE framework is sample-optimal and achieves an optimal error parameter of α = 3 . However, computing all W ( H j ) 's exactly requires ˜ O ( n 2 ) time, which is expensive for large hypotheses classes. Instead, [ABS24] computes a proxy distance ˆ W A ( H j ) = max H k ∈ A ˆ w k ( H j ) for each H j , which serves as an updateable approximation that lower bounds W ( H j ) . All ˆ W A ( H j ) 's are initially set to 0. The proxy distances are updated throughout the algorithm via an iterative process. At every iteration, a selectively chosen hypothesis, called a prompting hypothesis , is added to A . Carefully selecting A

ensures that for all H j , ˆ W A ( H j ) is a good approximation of W ( H j ) without exhaustively computing all pairwise semi-distances. At the end of this process, a hypothesis with a low proxy distance is selected as the output. Because only O ( | A | · n ) semi-distances are queried, this strategy enables a nearly-linear time algorithm in ˜ O ( n ) that still maintains the α = 3 guarantee.

To identify prompting hypotheses, [ABS24] keeps track of 'buckets' of candidate hypotheses. Each hypothesis H j is assigned a bucket according to its proxy distance ˆ W A ( H j ) . Because the algorithm seeks a hypothesis with approximately the smallest maximum semi-distance, it only focuses on the 'lowest' bucket with the smallest proxy distances. Because a large proxy distance implies that the hypothesis is far from P , the algorithm can disregard hypotheses with a large proxy distance. Hence, only the hypothesis in the lowest bucket are required to have accurate proxy distances.

The set of prompting hypotheses A will ensure that the lowest bucket has hypotheses with proxy distances close to the maximum semi-distances.

Recall that if a hypothesis ˆ H satisfies w i ∗ ( ˆ H ) ≤ OPT , then it is a valid output. Conversely, we would like to avoid outputting a hypothesis for which we have w i ∗ ( ˆ H ) &gt; OPT . The algorithm of [ABS24] filters out such hypotheses in the lowest bucket by updating their proxy distance and effectively sending them to 'higher' buckets. Specifically, a key observation is that H i ∗ will always significantly lift the proxy distances of hypotheses that are poor choices. Therefore, in every iteration of the algorithm, the goal is to find a prompting hypothesis H i that substantially improves a large portion of the proxy distances and empties out the candidate hypotheses in the lowest bucket.

It can be shown after roughly Θ(log n ) iterations either the lowest bucket is fully emptied out, and the algorithm can move forward to the next bucket. Or, there are no more prompting hypotheses that can be identified. This condition implies that H i ∗ is not prompting. That is, H i ∗ could not lift most hypotheses in the lowest bucket. For those hypotheses, we must have that their w i ∗ ( H j ) ≤ OPT . Hence, a random hypothesis in the lowest bucket under this condition is a valid output.

## 3 Private Hypothesis Selection Algorithm

## 3.1 Overview of our algorithm

In this section, we present an overview of Algorithm 1, our main algorithm for solving the hypothesis selection problem in the central model of DP that obtains an α = 3 guarantee. Building on the previous work described in Section 2.2, our goal is to find a hypothesis ˆ H that approximately minimizes W ( ˆ H ) . For each hypothesis H j ∈ H , our algorithm keeps track of the proxy distances ˆ W A ( H j ) = max H k ∈ A ˆ w k ( H j ) . The set A will store a small set of prompting hypotheses accumulated so far. In every round t of our algorithm, we privately identify a prompting hypothesis H i t that improves a substantial portion of current proxy distances ˆ W A ( H j ) and update each ˆ W A ( H j ) to max ( ˆ W A ( H j ) , ˆ w i t ( H j ) ) . This effectively adds the privately selected H i t to the set A .

The primary challenge of privatizing the process of identifying a prompting hypothesis in [ABS24] arises due to the bucketing scheme. This membership to a bucket is highly sensitive due to its discrete nature. In particular, changing one sample of the dataset could potentially shift the membership of every single hypothesis in H by changing every ˆ W A ( H j ) .

Therefore, to privately select hypotheses with low proxy distances without relying on buckets, we use the exponential mechanism. We maintain a distribution of hypotheses Q that assigns each hypothesis H j a probability that favors hypotheses with low proxy distance ˆ W A ( H j ) . Based on this change, we modify the notion of prompting from [ABS24]: we say that a hypothesis is prompting with respect to the distribution Q over H , rather than with respect to the hypotheses in a bucket. We also introduce the notion of ( σ ′ , η ) -prompting to quantify the 'prompting-ness' of a hypothesis in Equation 2, where η is the probability mass of hypotheses in Q that can be σ ′ -lifted. Recall that a hypothesis H i σ ′ -lifts a hypothesis H j if ˆ w i ( H j ) -ˆ W A ( H j ) ≥ σ ′ . After sufficiently many rounds, when no more prompting hypotheses can be found, we sample an output hypothesis ˆ H from distribution Q .

Identifying prompting hypotheses: To privately identify a prompting hypothesis H i , we test whether H i significantly lifts a large probability mass of hypotheses in Q . Astraightforward approach

is to first create a list K of hypotheses that are sampled from Q . Then, we may choose a prompting hypothesis using a threshold query for hypotheses in K that are lifted significantly. Unfortunately, such a threshold query would have a very high sensitivity with respect to the dataset D , as a single change in the dataset can shift every lift value from below the threshold to being above the threshold.

To solve the issue of the high sensitivity, we replace the exact count of hypotheses lifted with a new type of query called score η, K ,D ( H i ) . This query instead returns a quantile of the lift values of each hypothesis, which is a much more stable type of query with a lower sensitivity.

More specifically, score η, K ,D ( H i ) is computed by Algorithm 2 as follows: we first calculate the lift value H i induces on each element in K . Then, we sort these values in non-increasing order and return the ⌈ η/ 2 · |K|⌉ -th largest lift value. This significantly reduces the sensitivity of score η, K ,D ( H i ) , as shown in section E.1. Even if every single value in H i shifts by some amount due to a change in the dataset, the quantile that we return should not shift significantly. In Section E.2, we show that score η, K ,D ( H i ) can be used to identify whether or not H i is prompting.

Applying the SVT: Wenowwish to determine exactly which hypotheses have high score η, K ,D ( H i ) 's to find hypotheses that are significantly prompting. For every round of our algorithm, we attempt to find a prompting hypothesis. This task is equivalent to answering n threshold queries. In general, this is very costly from the perspective of privacy. However, because we are only interested in one hypothesis that passes the threshold, we use the sparse vector technique (SVT) [DNR + 09] to find this hypothesis with minimal privacy cost.

Algorithm 3 describes an algorithm using the SVT, which privately outputs either the index i of the hypothesis that was detected to have a high score η, K ,D ( H i ) , or ⊥ if no hypotheses have sufficiently high scores. This algorithm has guarantees on finding a prompting hypothesis formalized in Theorem 6. First, whenever the SVT returns a hypothesis, we guarantee that it is at least somewhat prompting, so we make progress at every round by updating the proxy distances. Second, whenever the SVT fails to find any prompting hypotheses, all hypotheses have small score η, K ,D ( H j ) 's and are therefore unlikely to be prompting.

As in [ABS24], H i ∗ is typically prompting for hypotheses far from P . When the SVT is unable to find any prompting hypotheses, it implies that H i ∗ is not prompting with respect to Q . Consequently, a large probability mass of hypotheses in Q -namely, those with minimal proxy distances-must have proxy distances that cannot be lifted by H i ∗ . Recall that all the poor hypotheses can be lifted by H i ∗ , so in this case, outputting any random hypothesis in Q will be valid with high probability.

Number of rounds: Because we do not allow a hypothesis to be added to the prompting set more than once, our algorithm will certainly halt after at most n rounds. However, this leads to a quadratic bound on the time complexity. We show that the algorithm will halt after O (log n ) rounds, yielding a nearly-linear time complexity.

Arguing about this round complexity is a key hurdle that arises in the private setting. The non-private algorithm in [ABS24] iteratively eliminates hypotheses from buckets. At every round, upon finding a prompting hypothesis, a significant (say a constant) fraction of hypotheses within the bucket have their proxy distances updated, leading to their removal from the bucket. This ensures that even with an initial bucket of all n hypotheses, the algorithm concentrates on this bucket for only O (log n ) iterations. After this, the bucket is either empty or the algorithm halts due to the lack of a prompting hypothesis. However, adapting the notion of prompting to the case where we update only a set of hypotheses with a constant probability mass according to Q fundamentally changes the analysis. Determining the actual fraction of updated hypotheses becomes much more complex. For example, we might update only one hypothesis, since it may hold a constant probability mass under Q .

To resolve this issue, we provide a refined analysis of the progress of the exponential mechanism. This analysis relies on the fact that the normalization term in the exponential mechanism's probabilities must decrease with each prompting hypothesis added to the set A , as some proxy distances will increase but none can decrease. As a result, hypotheses whose proxy distances do not increase significantly will see their probabilities rise. As these probabilities cannot exceed one, changes to the proxy distances must be able to keep up with the decrease in the normalization term. However, they can only keep up for so long, as each proxy distance is itself upper bounded by one. As a result, we can bound the number of rounds of our algorithm by O (log n ) .

Enforcing privacy: Throughout every round of the algorithm, we incur two types of privacy costs: one from drawing |K| hypotheses from the exponential mechanism and another from identifying a prompting hypothesis through the sparse vector technique. As we have discussed above, the types of queries our algorithm makes have low sensitivities with respect to the dataset. In Lemma B.1, we give a complete proof of privacy by using basic additive composition.

## 3.2 Algorithm

In Algorithm 1, we begin by sampling s samples from the unknown distribution to make up a dataset D . We initialize A , the set of prompting hypotheses to be empty, and the proxy distance of each hypothesis to 0. We then iteratively search, over at most T rounds, for prompting hypotheses to add to A . In each round, we re-calculate the probabilities of the exponential mechanism in Line 9 and draw a list K of k hypotheses from this mechanism in Line 10. We call the FIND-PROMPTING-HYPOTHESIS procedure of Algorithm 3 to identify a hypothesis which is empirically prompting over K using the sparse vector technique. In Section F, we thoroughly describe this procedure. In Section E.1, we describe the COMPUTE-SCORE procedure used to assign a score to each hypothesis throughout FIND-PROMPTING-HYPOTHESIS. If FIND-PROMPTING-HYPOTHESIS returns a hypothesis, we add that hypothesis to A in Line 13 and update the proxy estimates of all hypotheses in Line 14. If FIND-PROMPTING-HYPOTHESIS returns ⊥ , indicating that it could not find a prompting hypothesis, we break from the 'for' loop, draw a hypothesis using the exponential mechanism, and output that final hypothesis.

̸

```
Algorithm 1 A private algorithm for hypothesis selection 1: procedure SELECT-HYPOTHESIS( H , ϵ, σ, β ) 2: s ← Θ ( 1 β 2 σ 2 ϵ log 3 ( n/β ) ) , T ← min ( Θ ( 1 βσ log ( n/β ) ) , n ) , k ← Θ ( 1 β log ( n/β ) ) 3: ϵ 1 ← ϵ 2( kT +1) , ϵ 2 ← ϵ 2 T 4: 5: D ← s samples drawn from P ▷ we will use these samples to compute semi-distances 6: A ←∅ 7: ˆ W A ( H j ) ← 0 for every H j ∈ H 8: for t = 1 , . . . , T do 9: Q ( H j ) ∝ exp ( -ϵ 1 ˆ W A ( H j ) 2∆ ( ˆ W A ) ) for every H j ∈ H 10: K ← k hypotheses drawn from Q ▷ sample using exponential mechanism 11: H i t ← FIND-PROMPTING-HYPOTHESIS ( ϵ 2 , 2 s , σ ′ 4 , β 4 , H\ A, K , D ) ▷ Algorithm 3 12: if H i t = ⊥ then 13: A ← A ∪ { H i t } ▷ add H i t to prompting set 14: ˆ W A ( H j ) ← max ( ˆ W A ( H j ) , ˆ w i t ( H j ) ) for every H j ∈ H 15: else 16: break ▷ failed to find a prompting hypothesis 17: return ˆ H ∼ Q and halt
```

Theorem 3. Let ϵ, β, σ ∈ (0 , 1) . Algorithm 1 is an ( α = 3 , ϵ, β, σ ) -private learner for hypothesis selection that uses s = Θ ( 1 β 2 σ 2 ϵ log 3 ( n/β ) ) samples and has time complexity Θ ( min ( 1 β 4 σ 3 ϵ · n · log 5 ( n/β ) , 1 β 3 σ 2 ϵ · n 2 · log 4 ( n/β ) )) .

Proof sketch: In Section B, we prove that Algorithm 1 is ϵ -differentially private. In Section C, we prove the correctness of Algorithm 1. Specifically, in Section C.1, we show that each hypothesis added to A will be prompting with high probability. In Section C.2, we show that, if this is the case, then the algorithm will halt after at most O (log n ) rounds. In Section C.3, we show that, if the algorithm halts early, then it will output a valid hypothesis with high probability. In Section C.5, we give exact settings for s, T , and k such that our proof of correctness holds. Finally, in Section D, we prove the time complexity of Algorithm 1.

## 4 Acknowledgments

R.S. acknowledges partial support from the Ken Kennedy Institute Research Cluster Fund and the Ken Kennedy Institute Computational Science and Engineering Recruiting Fellowship, funded by the Energy HPC Conference and the Rice University Department of Computer Science.

## References

- [AAAK21] Ishaq Aden-Ali, Hassan Ashtiani, and Gautam Kamath. On the sample complexity of privately learning unbounded high-dimensional gaussians. In Proceedings of the 32nd International Conference on Algorithmic Learning Theory, ALT , Proceedings of Machine Learning Research. PMLR, 2021.
- [AAC + 23] Anders Aamand, Alexandr Andoni, Justin Y. Chen, Piotr Indyk, Shyam Narayanan, and Sandeep Silwal. Data structures for density estimation. In Proceedings of the 40th International Conference on Machine Learning . PMLR, 2023.
- [AAC + 24] Anders Aamand, Alexandr Andoni, Justin Chen, Piotr Indyk, Shyam Narayanan, Sandeep Silwal, and Haike Xu. Statistical-computational trade-offs for density estimation. Advances in Neural Information Processing Systems , 37:97907-97927, 2024.
- [ABH + 20] Hassan Ashtiani, Shai Ben-David, Nicholas J. A. Harvey, Christopher Liaw, Abbas Mehrabian, and Yaniv Plan. Near-optimal sample complexity bounds for robust learning of gaussian mixtures via compression schemes. J. ACM , 67(6):32:1-32:42, 2020.
- [ABM18] Hassan Ashtiani, Shai Ben-David, and Abbas Mehrabian. Sample-efficient learning of mixtures. In Sheila A. McIlraith and Kilian Q. Weinberger, editors, Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence , pages 2679-2686. AAAI Press, 2018.
- [Abo18] John M. Abowd. The us census bureau adopts differential privacy. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mining , page 2867, 2018.
- [ABR16] Maryam Aliakbarpour, Eric Blais, and Ronitt Rubinfeld. Learning and testing junta distributions. In Proceedings of the 29th Conference on Learning Theory, COLT 2016, New York, USA, June 23-26, 2016 , pages 19-46, 2016.
- [ABS23] Maryam Aliakbarpour, Mark Bun, and Adam Smith. Hypothesis selection with memory constraints. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems , volume 36, pages 5045350481. Curran Associates, Inc., 2023.
- [ABS24] Maryam Aliakbarpour, Mark Bun, and Adam Smith. Optimal hypothesis selection in (almost) linear time. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems , volume 37, pages 141490-141527. Curran Associates, Inc., 2024.
- [ACFT19] J. Acharya, C. Canonne, C. Freitag, and H. Tyagi. Test without trust: Optimal locally private distribution testing. In Proceedings of Machine Learning Research , volume 89 of Proceedings of Machine Learning Research , pages 2067-2076. PMLR, 2019.
- [ADH + 24] Hilal Asi, John C Duchi, Saminul Haque, Zewei Li, and Feng Ruan. Universally instance-optimal mechanisms for private statistical estimation. In The Thirty Seventh Annual Conference on Learning Theory , pages 221-259. PMLR, 2024.
- [ADKR19] Maryam Aliakbarpour, Ilias Diakonikolas, Daniel Kane, and Ronitt Rubinfeld. Private testing of distributions via sample permutations. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, 8-14 December 2019, Vancouver, BC, Canada , pages 1087710888, 2019.

- [ADR18] Maryam Aliakbarpour, Ilias Diakonikolas, and Ronitt Rubinfeld. Differentially private identity and equivalence testing of discrete distributions. In Proceedings of the 35th International Conference on Machine Learning (ICML) , volume 80, pages 169-178, 2018.
- [AFJ + 18] Jayadev Acharya, Moein Falahatgar, Ashkan Jafarpour, Alon Orlitsky, and Ananda Theertha Suresh. Maximum selection and sorting with adversarial comparators. The Journal of Machine Learning Research , 19:59:1-59:31, 2018.
- [AGJ + 22] Kareem Amin, Jennifer Gillenwater, Matthew Joseph, Alex Kulesza, and Sergei Vassilvitskii. Plume: Differential privacy at scale. CoRR , abs/2201.11603, 2022.
- [AJM20] Kareem Amin, Matthew Joseph, and Jieming Mao. Pan-private uniformity testing. In Jacob D. Abernethy and Shivani Agarwal, editors, Conference on Learning Theory, COLT 2020, 9-12 July 2020, Virtual Event [Graz, Austria] , volume 125 of Proceedings of Machine Learning Research , pages 183-218. PMLR, 2020.
- [AJOS14] Jayadev Acharya, Ashkan Jafarpour, Alon Orlitsky, and Ananda Theertha Suresh. Sorting with adversarial comparators and application to density estimation. In 2014 IEEE International Symposium on Information Theory , pages 1682-1686, 2014.
- [ASZ18] Jayadev Acharya, Ziteng Sun, and Huanyu Zhang. Differentially private testing of identity and closeness of discrete distributions. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 31. Curran Associates, Inc., 2018.
- [BB20] Thomas Berrett and Cristina Butucea. Locally private non-asymptotic testing of discrete distributions is faster using interactive mechanisms. In Hugo Larochelle, Marc'Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual , 2020.
- [BBK + 21] Olivier Bousquet, Mark Braverman, Gillat Kol, Klim Efremenko, and Shay Moran. Statistically near-optimal hypothesis selection. In 62nd IEEE Annual Symposium on Foundations of Computer Science, FOCS 2021, Denver, CO, USA, February 7-10, 2022 , pages 909-919. IEEE, 2021.
- [BKM19] Olivier Bousquet, Daniel Kane, and Shay Moran. The optimal approximation factor in density estimation. In Alina Beygelzimer and Daniel Hsu, editors, Proceedings of the Thirty-Second Conference on Learning Theory , volume 99 of Proceedings of Machine Learning Research , pages 318-341. PMLR, 25-28 Jun 2019.
- [BKSW19] Mark Bun, Gautam Kamath, Thomas Steinke, and Zhiwei Steven Wu. Private hypothesis selection. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d'Alché-Buc, Emily B. Fox, and Roman Garnett, editors, Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada , pages 156-167, 2019.
- [CDK17] B. Cai, C. Daskalakis, and G. Kamath. Priv'it: Private and sample efficient identity testing. In International Conference on Machine Learning, ICML , pages 635-644, 2017.
- [CKM + 19] Clément L. Canonne, Gautam Kamath, Audra McMillan, Adam D. Smith, and Jonathan R. Ullman. The structure of optimal private tests for simple hypotheses. In Moses Charikar and Edith Cohen, editors, Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing, STOC 2019, Phoenix, AZ, USA, June 23-26, 2019 , pages 310-321. ACM, 2019.
- [CKM + 20] Clément L Canonne, Gautam Kamath, Audra McMillan, Jonathan Ullman, and Lydia Zakynthinou. Private identity testing for high-dimensional distributions. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems , volume 33, pages 10099-10111. Curran Associates, Inc., 2020.

- [Dia16] Ilias Diakonikolas. Learning structured distributions. In CRC Handbook of Big Data , pages 267-283. 2016.
- [Dif17] Differential Privacy Team, Apple. Learning with privacy at scale. https://machinelearning.apple.com/research/ learning-with-privacy-at-scale , 2017. Accessed: 2024-05-22.
- [DK14] Constantinos Daskalakis and Gautam Kamath. Faster and sample near-optimal algorithms for proper learning mixtures of gaussians. In Conference on Learning Theory , pages 1183-1213. PMLR, 2014.
- [DL96] Luc Devroye and Gábor Lugosi. A universally acceptable smoothing factor for kernel density estimates. The Annals of Statistics , 24(6):2499 - 2512, 1996.
- [DL97] Luc Devroye and Gábor Lugosi. Nonasymptotic universal smoothing factors, kernel complexity and Yatracos classes. The Annals of Statistics , 25(6):2626 - 2637, 1997.
- [DL01] Luc Devroye and Gábor Lugosi. Combinatorial methods in density estimation . Springer, 2001.
- [DMNS06a] Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Shai Halevi and Tal Rabin, editors, Theory of Cryptography , pages 265-284, Berlin, Heidelberg, 2006. Springer Berlin Heidelberg.
- [DMNS06b] Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography: Third Theory of Cryptography Conference, TCC 2006, New York, NY, USA, March 4-7, 2006. Proceedings 3 , pages 265-284. Springer, 2006.
- [DNR + 09] Cynthia Dwork, Moni Naor, Omer Reingold, Guy N. Rothblum, and Salil Vadhan. On the complexity of differentially private data release: efficient algorithms and hardness results. In Proceedings of the Forty-First Annual ACM Symposium on Theory of Computing , STOC '09, pages 381-390, New York, NY, USA, 2009. Association for Computing Machinery.
- [DR + 14] Cynthia Dwork, Aaron Roth, et al. The algorithmic foundations of differential privacy. Foundations and Trends® in Theoretical Computer Science , 9(3-4):211-407, 2014.
- [DVG20] Damien Desfontaines, James Voss, and Bryant Gipson. Differentially private partition selection. CoRR , abs/2006.03684, 2020.
- [EPK14] Úlfar Erlingsson, Vasyl Pihur, and Aleksandra Korolova. Rappor: Randomized aggregatable privacy-preserving ordinal response. In Proceedings of the 2014 ACM SIGSAC Conference on Computer and Communications Security , pages 1054-1067, 2014.
- [GKK + 20] Sivakanth Gopi, Gautam Kamath, Janardhan Kulkarni, Aleksandar Nikolov, Zhiwei Steven Wu, and Huanyu Zhang. Locally private hypothesis selection. In Jacob D. Abernethy and Shivani Agarwal, editors, Conference on Learning Theory, COLT 2020, 9-12 July 2020, Virtual Event [Graz, Austria] , volume 125 of Proceedings of Machine Learning Research , pages 1785-1816. PMLR, 2020.
- [GR18] M. Gaboardi and R. Rogers. Local private hypothesis testing: Chi-square tests. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018 , pages 1612-1621, 2018.
- [KLN + 11] Shiva Prasad Kasiviswanathan, Homin K Lee, Kobbi Nissim, Sofya Raskhodnikova, and Adam Smith. What can we learn privately? SIAM Journal on Computing , 40(3):793-826, 2011.
- [KT18] Krishnaram Kenthapadi and Thanh T. L. Tran. Pripearl: A framework for privacypreserving analytics and reporting at linkedin. In Alfredo Cuzzocrea, James Allan, Norman W. Paton, Divesh Srivastava, Rakesh Agrawal, Andrei Z. Broder, Mohammed J. Zaki, K. Selçuk Candan, Alexandros Labrinidis, Assaf Schuster, and Haixun Wang,

- editors, Proceedings of the 27th ACM International Conference on Information and Knowledge Management, CIKM 2018, Torino, Italy, October 22-26, 2018 , pages 2183-2191. ACM, 2018.
- [LSL16] Min Lyu, Dong Su, and Ninghui Li. Understanding the sparse vector technique for differential privacy. arXiv preprint arXiv:1603.01699 , 2016.
- [LSV + 19] Mathias Lécuyer, Riley Spahn, Kiran Vodrahalli, Roxana Geambasu, and Daniel Hsu. Privacy accounting and quality control in the sage differentially private ML platform. CoRR , abs/1909.01502, 2019.
- [MS08] Satyaki Mahalanabis and Daniel Stefankovic. Density estimation in linear time. In 21st Annual Conference on Learning Theory - COLT , pages 503-512, 2008.
- [MT07] Frank McSherry and Kunal Talwar. Mechanism design via differential privacy. In 48th Annual IEEE Symposium on Foundations of Computer Science (FOCS'07) , pages 94-103, 2007.
- [NH12] Arjun Narayan and Andreas Haeberlen. Djoin: Differentially private join queries over distributed databases. In Chandu Thekkath and Amin Vahdat, editors, 10th USENIX Symposium on Operating Systems Design and Implementation, OSDI 2012, Hollywood, CA, USA, October 8-10, 2012 , pages 149-162. USENIX Association, 2012.
- [PAA24] Alireza Fathollah Pour, Hassan Ashtiani, and Shahab Asoodeh. Sample-optimal locally private hypothesis selection and the provable benefits of interactivity. In The Thirty Seventh Annual Conference on Learning Theory, COLT , Proceedings of Machine Learning Research. PMLR, 2024.
- [PAJL25] Ankit Pensia, Amir R. Asadi, Varun Jog, and Po-Ling Loh. Simple binary hypothesis testing under local differential privacy and communication constraints. IEEE Trans. Inf. Theor. , 71(1):592-617, January 2025.
- [PJL24] Ankit Pensia, Varun S. Jog, and Po-Ling Loh. The sample complexity of simple binary hypothesis testing. In Shipra Agrawal and Aaron Roth, editors, The Thirty Seventh Annual Conference on Learning Theory, June 30 - July 3, 2023, Edmonton, Canada , volume 247 of Proceedings of Machine Learning Research , pages 4205-4206. PMLR, 2024.
- [SOAJ14] Ananda Theertha Suresh, Alon Orlitsky, Jayadev Acharya, and Ashkan Jafarpour. Near-optimal-sample estimators for spherical gaussian mixtures. In Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems , volume 27, 2014.
- [Ver18] Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science . Cambridge University Press, 2018.
- [Yat85] Yannis G. Yatracos. Rates of convergence of minimum distance estimators and kolmogorov's entropy. The Annals of Statistics , 13(2):768 - 774, 1985.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Our abstract and introduction adhere to Theorem 3, which states the main contribution of the paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Please refer to the paragraph after Theorem 2 in the Introduction regarding extra factors in our sample complexity.

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

Justification: Our problem setup is described in Section 1.1. All the proofs are included in the appendices.

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

Justification: This paper is theoretical and does not contain experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- 4.1 If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- 4.2 If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- 4.3 If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- 4.4 We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: This paper is theoretical and does not contain experiments.

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

Justification: This paper is theoretical and does not contain experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: This paper is theoretical and does not contain experiments.

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

Justification: This paper is theoretical and does not contain experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: There are no violations of the NeurIPS Code of Ethics in this paper.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: While algorithms using hypothesis selection can have practical impacts, this paper describes the theoretical underpinning of the problem, making it difficult to meaningfully comment on its social impact. We believe its main impact will be improving scientific understanding of private hypothesis selection.

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

Justification: This paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: This paper uses no real asset other than publicly available papers we have cited.

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

Justification: This paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper is theoretical and does not contain any crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper is theoretical and does not contain any crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: This paper does not use LLMs for any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Other Related Works

Hypothesis selection has also been studied in the local model of DP [KLN + 11], where the data curator is not trusted and thus only has access to the privatized version of users' data. Gopi et al. [GKK + 20] showed that the sample complexity in the local model is linear in n , exponentially larger than the central DP setting. Specifically, they proved a lower bound of Ω ( n σ 2 ϵ 2 ) . They also proved two upper bounds of O ( n log 3 n σ 4 ϵ 2 ) for non-interactive algorithms and O ( n log n log log n σ 2 ϵ 2 ) (with α = 27 ) for sequentially interactive algorithms with O (log log n ) rounds. A recent algorithm by Pour et al. [PAA24] closed the gap between upper and lower bounds by designing a sequentially interactive algorithm using O ( n log 2 1 /β σ 2 min( ϵ 2 , 1) ) samples, with α = 9 and O (log log n ) rounds.

Another related problem in statistics is simple hypothesis testing. Given two distributions P and Q , and a dataset D sampled from one of them, the goal is to determine whether D was drawn from P or Q . This setting resembles hypothesis selection where we have only two candidate distributions. Cannone et al. [CKM + 19] developed a private algorithm for simple hypothesis testing that achieves optimal sample complexity. Further developments on sample complexity on private simple hypothesis selection include [PJL24, PAJL25] under the local model of DP. Other examples within the broader topic of private hypothesis testing include [CDK17, GR18, ADR18, ASZ18, ADKR19, ACFT19, AJM20, CKM + 20, BB20].

Asi et al. [ADH + 24] give an instance-optimal algorithm for hypothesis selection when the hypothesis class H contains the true distribution P (i.e. OPT = 0 ). Their algorithm has logarithmic sample complexity and time complexity O ( n 2 s ) .

## A.1 Industrial applications of DP

We highlight several lines of research in differentially private data analysis motivated by realworld challenges. One example is the partition selection problem, where users aim to compute aggregate statistics over data grouped according to user-specified criteria. To ensure privacy, designers must bound the sensitivity of the statistics, decide which data partition to release, and maintain computational efficiency. Relevant works include Desfontaines et al. [DVG20] and Google's Plume system [AGJ + 22].

Another direction concerns private analytics of user actions, where the goal is to prevent attackers from learning a user's past behavior by repeated observation of public analytics. LinkedIn's PriPeARL [KT18] provides such protection, even towards persistent attackers who observe the analytics overtime.

Privacy leakages also emerge from releasing models trained on sensitive data. To mitigate this problem, Lécuyer et al. [LSV + 19] developed Sage , a machine learning platform that distributes training across data blocks, monitors privacy loss per blocks, and retires blocks once their privacy budget is depleted.

Lastly, the problem of answering queries across multiple private databases has also been studied. A representative system is DJoin by Narayan et al. [NH12].

## B Proof of Privacy of Algorithm 1

Lemma B.1. Algorithm 1 is ϵ -differentially private.

Proof. In each iteration, Algorithm 1 samples K , a hypothesis list of size k , where each hypothesis H j is drawn with probability proportional to its proxy distance, ˆ W A ( H j ) . In Lemma E.3, we show that each proxy distance has sensitivity 1 /s . We privatize this sampling of hypotheses using the exponential mechanism (Definition 1.4).

After drawing K , the algorithm calculates score η, K ,D ( H j ) for each hypothesis H j , and searches for a hypothesis with a high score η, K ,D ( H j ) . For each H j , recall that score η, K ,D ( H j ) is an empirical quantile of the promptingness of H j on K . K is treated as publicly available when computing each score. This fact, along with the choice to use quantiles to calculate each score, leads to a low sensitivity of the score function. Specifically, in Lemma E.4, we prove that the score function has sensitivity 2 /s . We privately select a hypothesis with a sufficiently high score using the sparse vector technique (Algorithm 3), allowing us to incur a loss of privacy independent of the number of hypotheses.

All remaining steps in the round either post-process K and the chosen prompting hypothesis H i t , or reveal no further information about the dataset. Therefore, in each round, we only consider the privacy loss of drawing s hypotheses and finding a prompting hypothesis. By basic composition (Fact 1.7), sampling k hypotheses from Q and calling FIND-PROMPTING-HYPOTHESIS once gives the following privacy loss in each round:

<!-- formula-not-decoded -->

Algorithm 1 takes at most T iterations. The composition of T rounds of the above procedure results in a privacy loss of:

<!-- formula-not-decoded -->

Finally, Algorithm 1 samples from Q to obtain output distribution in the last round. Hence, Algorithm 1 has a total privacy loss of:

<!-- formula-not-decoded -->

## C Proof of Correctness of Algorithm 1

The correctness proof of Algorithm 1 proceeds by first showing that, with high probability, three key events occur: i ) the empirical semi-distances are accurate, ii ) in each round, the score of every hypothesis accurately reflects its ability to lift many hypotheses, and iii ) in each round, FINDPROMPTING-HYPOTHESIS either outputs a high-scoring hypothesis or correctly identifies that no such hypothesis remains. We then show that if these events hold, the algorithm will eventually fail to find a prompting hypothesis and will halt after less than T rounds. Finally, we prove that if the key events hold and the algorithm halts early, the output is, with high probability, less than (3OPT + σ ) -far from P .

## C.1 Key events occur

Let σ 1 = σ/ 4 and σ 2 = σ/ 4 . The correctness of Algorithm 1 relies on the following key events:

1. With the s samples that we draw from P , we calculate the empirical semi-distances between all pairs of hypotheses to within an σ 1 -additive error.
2. In each round of the algorithm, if the score of a hypothesis is at least σ 2 / 2 , then that hypothesis is ( σ 2 / 2 , η/ 4) -prompting with respect to Q . If the score of a hypothesis is less than σ 2 , then that hypothesis is not ( σ 2 , η ) -prompting with respect to Q .
3. In each round of the algorithm, if FIND-PROMPTING-HYPOTHESIS outputs a hypothesis, then the score of that hypothesis is at least σ 2 / 2 . If it does not output a hypothesis, then no hypothesis had a score greater than σ 2 .

Each of these events fails to occur with low probability, provided s , the number of samples, and k , the number of hypotheses sampled at each round, are sufficiently large. We recall the specific assumptions on s and k that guarantee each event as follows:

Empirical semi-distances are σ 1 -accurate: Lemma 2.1 ensures that, with probability at least 1 -β/ 6 , the empirical semi-distances are accurate up to an additive factor of σ 1 if

<!-- formula-not-decoded -->

Scores reflect prompting ability: In Lemma E.5, we show that, in each round, with high probability, the score of each hypothesis reflects whether that hypothesis is prompting with respect to the hypotheses in Q . Specifically, the lemma ensures that, with probability at least 1 -β/ (6 T ) , if

<!-- formula-not-decoded -->

then, in round t for all H i , if the score of H i is at least σ 2 / 2 , H i is ( σ 2 / 2 , η/ 4) -prompting with respect to Q , and, if the score of H i is less than σ 2 , H i is not ( σ 2 , η ) -prompting with respect to Q .

FIND-PROMPTING-HYPOTHESIS succeeds: In Theorem 6, we show that, in each round, with high probability, the FIND-PROMPTING-HYPOTHESIS procedure succeeds in either finding a prompting hypothesis or identifying that there are no such hypotheses. Specifically, the theorem ensures that, with probability at least 1 -β/ (6 T ) , if

<!-- formula-not-decoded -->

then, in round t , if FIND-PROMPTING-HYPOTHESIS ( ϵ 2 , ∆ = 2 /s, σ 2 , η, H\ A, K ) outputs H i t , the score of H i t is at least σ 2 / 2 , and, if the procedure outputs ⊥ , each H i t has a score less than σ 2 . Note that this bound on ∆ implies the following bound on s , as ∆ = 2 /s :

<!-- formula-not-decoded -->

For now, assume that s, T and k satisfy these requirements. In Section C.5, we will give a precise choice of parameters that satisfies these, along with other constraints. Then, the probability that at

least one of the three key events does not occur is:

Pr [ at least one key event fails ] ≤ Pr [ semi-distance estimation fails ]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C.2 Algorithm halts early

If FIND-PROMPTING-HYPOTHESIS outputs a hypothesis with score at least σ 2 / 2 , then that hypothesis must be able to σ 2 / 2 -lift at least an η/ 2 -fraction of K . If its scores are an accurate representation of its ability to lift Q , then H i is ( σ 2 / 2 , η/ 4) -prompting over Q . In this section, we show that, if each hypothesis added to the prompting set is truly ( σ 2 / 2 , η/ 4) -prompting-that is, if the second and third events described in the previous section hold-, then the algorithm will reach Line 16 and halt before executing T rounds.

Theorem 4. Let σ ′ = σ 2 / 2 , η ′ = η/ 4 . Assume exp ( -ϵ 1 σ ′ 2∆( ˆ W A ) ) &lt; 1 2 . Further, assume that each hypothesis added to the prompting set is ( σ ′ , η ′ ) -prompting with respect to Q in the round it is added. Then Algorithm 1 terminates after at most min ( 1 log ( 1+ η ′ 2 ) ( log ( n ) + ϵ 1 2∆( ˆ W A ) · OPT ) , n ) rounds.

Proof Sketch Initially, Q is the uniform distribution over { H 1 , . . . , H n } , as each proxy distance ˆ W ( t ) A ( H j ) is initialized to 0. After each round of the algorithm, a new hypothesis H i t is added to the prompting set and the proxy distance of each hypothesis either increases or remains constant. If each H i t is truly prompting over Q , the normalization term of the exponential mechanism decreases at each round, amplifying the probabilities of the hypotheses with proxy distances that are relatively unchanged by the addition of H t . By characterizing these changes, we give an upper bound ˜ T on the number of rounds our algorithm will execute before returning a hypothesis. Note that this upper bound must be less than n , as every hypothesis is added to the prompting set at most once. If we choose T to be greater than ˜ T , the algorithm will halt before executing T rounds.

To formally prove Theorem 4, we introduce the following notation, exactly describing the probability distribution induced by the exponential mechanism in each round.

Definition C.1 (Exponential mechanism at round t ) . Recall that, at round t , the hypothesis sampling distribution, Q ( t ) , is defined as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where and ˆ W ( t ) A ( H ) is the proxy distance of H at round t .

We also require the following lemma, which demonstrates that Z ( ℓ ) decreases with each round, and bounds the amount of this decrease.

Lemma C.2. Let ℓ ∈ { 1 , 2 , . . . } . Let σ ′ , η ′ ∈ (0 , 1] . Assume exp ( -ϵ 1 σ ′ 2∆( ˆ W A ) ) &lt; 1 2 . Define Z ( ℓ ) and Z ( ℓ +1) as in Definition 8. Assume the hypothesis H i ℓ added to the prompting set in round ℓ is ( σ ′ , η ′ ) -prompting with respect to Q ( ℓ ) . Then, we have:

<!-- formula-not-decoded -->

Before proving Lemma C.2, we introduce, for each hypothesis H j , a value u ( ℓ +1) ( H j ) describing the increase in H j 's proxy distance between rounds ℓ and ℓ +1 :

<!-- formula-not-decoded -->

Recall that we define ˆ W ( ℓ +1) A ( H j ) to be the maximum empirical semi-distance between H j and the hypotheses in the prompting set A . Thus, when we add H i ℓ to the prompting set in round ℓ , ˆ W ( ℓ +1) A ( H j ) is the maximum of ˆ w i ℓ ( H j ) and ˆ W ( ℓ ) A ( H j ) :

<!-- formula-not-decoded -->

The difference between ˆ W ( ℓ +1) A ( H j ) and ˆ W ( ℓ ) A ( H j ) is thus lower bounded by the difference between ˆ W ( ℓ +1) A ( H j ) and ˆ w i ℓ ( H j ) :

<!-- formula-not-decoded -->

If H ℓ is ( σ ′ , η ′ ) -prompting, the definition of prompting allows us to lower bound the probability that this difference is greater than σ ′ :

<!-- formula-not-decoded -->

To prove Lemma C.2, we relate the ratio in question to the expected value of a function of these differences. If the hypothesis added to the prompting set at round ℓ is prompting, Equation 11 gives a tail bound on each of these differences. Combining this expected value and tail bound with careful consideration yields the desired bound.

Proof of Lemma C.2. Considering the ratio in question, we expand Z ( ℓ +1) as follows:

<!-- formula-not-decoded -->

Rewriting ˆ W ( ℓ +1) A ( H j ) in terms of u ( ℓ +1) ( H j ) and ˆ W ( ℓ ) A ( H j ) for every j , we have:

<!-- formula-not-decoded -->

The second equality follows from the definition of Q ( ℓ ) ( H j ) . Then, notice that the final line above forms an expectation over Q ( ℓ ) . We can rewrite it as such:

<!-- formula-not-decoded -->

Assuming the hypothesis added to the prompting set at round ℓ is prompting, Equation 11 applies. We will use that bound on the probability to bound the expectation in Equation 13. To do so, as the argument of the expectation is non-negative, we begin by applying the integral identity [Ver18]:

<!-- formula-not-decoded -->

Rearranging to isolate u ( ℓ +1) ( H j ) within the probability gives:

<!-- formula-not-decoded -->

We apply a change of variables, letting v = 2∆( ˆ W A ) ϵ 1 log 1 t :

<!-- formula-not-decoded -->

Because u ( ℓ +1) ( H j ) is non-negative, the integrand is 0 when v &lt; 0 , and we need only to evaluate the integral from v = 0 to ∞ :

<!-- formula-not-decoded -->

By Equation 11, we know that, for any v ≤ σ ′ , Pr [ u ( ℓ +1) ( H j ) ≥ v ] ≥ η ′ . This implies that, for any v ∈ [0 , σ ′ ] , we can upper bound Pr [ u ( ℓ +1) ( H j ) &lt; v ] by 1 -η ′ . At the same time, for v &gt; σ ′ , we can upper bound Pr [ u ( ℓ +1) ( H j ) &lt; v ] by 1. Applying these bounds, the resulting integral is:

<!-- formula-not-decoded -->

Under the assumption exp ( -ϵ 1 σ ′ 2∆( ˆ W A ) ) &lt; 1 2 , the following holds:

<!-- formula-not-decoded -->

Lemma C.2 implies that when hypothesis H i ℓ is added to the prompting set, the probability of sampling a hypothesis H j which is not significantly lifted by H i ℓ increases. As we continue to add prompting hypotheses which do not significantly lift H j , Q ( H j ) continues to increase. As this probability approaches 1 and H j is repeatedly sampled by the exponential mechanism, either we will find a hypothesis which lifts H j , in which case its probability will decrease, or we will halt, as we could not find such a hypothesis. At any given round, we know that the probability of sampling any hypothesis cannot grow until it exceeds 1. We use this fact to upper bound the number of rounds of Algorithm 1, thus proving Theorem 4.

Proof of Theorem 4. Note that the algorithm cannot add more than n hypotheses to the set A , as it will only check for a prompting hypothesis among the hypotheses in H\ A . As a result, the algorithm does not run for more than n rounds.

Fix t ≥ 0 . Consider the probability that H j is selected by the exponential mechanism at round t +1 , Q ( t +1) ( H j ) :

<!-- formula-not-decoded -->

Unfolding this expression over all rounds prior to t , we obtain:

<!-- formula-not-decoded -->

Recall that Q (0) follows a uniform distribution, implying Q (0) ( H j ) = 1 n . Further, as ˆ W (0) A ( H j ) = 0 and ˆ W ( t ) A ( H j ) = ˆ W (0) A ( H j ) + ∑ t ℓ =1 u ( ℓ ) ( H j ) , we have ˆ W ( t ) A ( H j ) = ∑ t ℓ =1 u ( ℓ ) ( H j ) . As a result, we can simplify Equation 14 to:

<!-- formula-not-decoded -->

By Lemma C.2, we know Z ( ℓ +1) Z ( ℓ ) ≤ 1 -η ′ 2 for all ℓ . Consequently, Z ( ℓ -1) Z ( ℓ ) ≥ 1 1 -η ′ 2 ≥ 1 + η ′ 2

Additionally, W A ( H j ) is upper bounded by W ( H j ) . Combining these two bounds and Equation 15, we have:

. ˆ ( ℓ )

<!-- formula-not-decoded -->

The above expression of Q ( t ) ( H j ) holds for all H j -including H i ∗ . If we focus specifically on H i ∗ , we know W ( H i ∗ ) ≤ OPT, leading to the bound:

<!-- formula-not-decoded -->

Because Q ( t ) ( H i ∗ ) represents a probability, it is upper bounded by 1. The lower bound given in Equation 17 is thus also upper bounded by 1. We use these bounds to establish an upper bound on t :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, our algorithm will terminate after at most log( n )+ 1 2∆( ˆ WA ) · OPT log ( 1+ η ′ 2 )

## C.3 Algorithm outputs a valid hypothesis

Theorem 5. Assume σ ≥ 8∆( ˆ W A ) ϵ 1 log (4 n/β ) . Assume the key events hold and Algorithm 1 reaches Line 16 and outputs ˆ H . Then, with probability at most β/ 2 , ∥ ˆ H -P ∥ TV &gt; 3 · OPT+ σ .

Proof. If Algorithm 1 breaks in round t , it must be that FIND-PROMPTING-HYPOTHESIS was unable to find a prompting hypothesis and returned ⊥ . Under the assumption that FIND-PROMPTINGHYPOTHESIS succeeds, this implies that, at round t , all hypotheses had scores less than σ 2 , implying that they were unable to σ 2 -lift more than an η/ 2 fraction of K . We've additionally assumed that the score of each hypothesis reflects its promptingness. That is, if the score of a hypothesis is less than σ 2 , it is not ( σ 2 , η ) -prompting. Therefore, under our assumptions, no H i ∈ H is ( σ 2 , η ) -prompting. Specifically, H i ∗ is not ( σ 2 , η ) -prompting. This guarantees that:

<!-- formula-not-decoded -->

We also know, by the utility guarantee of the exponential mechanism given in Lemma 1 . 6 , that, with probability at least 1 -β/ 4 , for ˆ H ∼ Q :

<!-- formula-not-decoded -->

We can bound min i ∈ [ n ] ˆ W A ( H i ) as follows:

<!-- formula-not-decoded -->

Recall that we have assumed that the empirical semi-distance estimates are accurate up to an additive factor of σ 1 . This implies:

<!-- formula-not-decoded -->

Combining Equation 19, and Equation 21, we have, with probability at least 1 -β/ 4 , the following bound on, ˆ W A ( ˆ H ) , the proxy distance of our outputted hypothesis:

<!-- formula-not-decoded -->

Because ˆ H is drawn from Q , we know, from Equation 18, that, with probability at least 1 -η, we can bound the empirical semi-distance ˆ w i ∗ ( ˆ H ) as:

<!-- formula-not-decoded -->

Combining Equation 22 and Equation 23, we have, with probability at least 1 -( β/ 4 + η ) , the following bound on ˆ w i ∗ ( ˆ H ) :

<!-- formula-not-decoded -->

By Equation 3, we have ∥ ˆ H -P ∥ TV ≤ 2OPT + w i ∗ ( ˆ H ) . Together with Equation 24, this yields:

<!-- formula-not-decoded -->

The second-to-last inequality holds because we set σ 2 = σ/ 4 , σ 1 = σ/ 4 , and we assume σ ≥ 8∆( ˆ W A ) ϵ 1 log (4 n/β ) . Recall that our algorithm sets η = β/ 4 . Ultimately, we have shown:

<!-- formula-not-decoded -->

## C.4 Overall correctness

We have shown, under certain constraints on s, T and k , that the probability that key events do not all hold is at most β/ 2 , and that, if the key events do hold and the algorithm halts before T rounds, then the probability that the algorithm outputs a far hypothesis ˆ H is at most β/ 2 .

Hence, if s, T and k satisfy the required constraints, we can bound the probability that the algorithm outputs a hypothesis ˆ H greater than (3OPT + σ ) -far as:

<!-- formula-not-decoded -->

## C.5 Sample complexity

In this section, we give exact settings of s, T , and k that satisfy the constraints given throughout our proof of correctness.

Recall that, throughout our proof of correctness, we make the following assumptions:

1. To accurately estimate semi-distances:

<!-- formula-not-decoded -->

2. To accurately approximate prompting-ness:

<!-- formula-not-decoded -->

3. For FIND-PROMPTING-HYPOTHESIS to succeed:

<!-- formula-not-decoded -->

4. To ensure the final output has a low proxy distance:

<!-- formula-not-decoded -->

5. To bound the number of rounds the algorithm will execute:

<!-- formula-not-decoded -->

6. To ensure the algorithm does not halt prematurely:

<!-- formula-not-decoded -->

Throughout the algorithm and analysis, we also have the following parameter settings:

<!-- formula-not-decoded -->

Combining these settings with our above assumptions and constraints, we have the following set of requirements for s, T and k :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We claim that the following settings of s, T , and k will satisfy these requirements:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

s satisfies constraints As both T and k are logarithmic in n , the fourth argument in the constraint on s will dominate. For completeness, we show that each argument of this constraint is less than our choice of s .

Beginning with the first argument, we have:

<!-- formula-not-decoded -->

For the second argument, we have:

<!-- formula-not-decoded -->

Then, because n ≥ T , we have the following sequence of inequalities, continued from Equation 34:

<!-- formula-not-decoded -->

Note that the third argument is entirely subsumed by the fourth. Therefore, for both the third and fourth argument, we have:

<!-- formula-not-decoded -->

T satisfies constraints We now show that our choice of T exceeds the bound established in Theorem 4. First, observe that we can lower bound T by:

<!-- formula-not-decoded -->

We can then expand the second term of this equation and rewrite it in terms of s, T and k , yielding:

<!-- formula-not-decoded -->

Applying the fact that β &lt; 1 and the inequality log(1 + x ) ≥ x 1+ x for all x &gt; -1 , we can then show:

<!-- formula-not-decoded -->

Finally, because OPT ≤ 1 , we know T must be greater than the number of rounds required for the algorithm to terminate.

<!-- formula-not-decoded -->

k satisfies constraints Recall that T must be less than n , as we do not allow any hypothesis to be added to the prompting set more than once. Then, our choice of k satisfies the constraint as follows:

<!-- formula-not-decoded -->

## D Proof of Time Complexity of Algorithm 1

In this section, we prove the time complexity of Algorithm 1.

<!-- formula-not-decoded -->

Proof. We walk through each step in Algorithm 1. Recall that each semi-distance query ˆ w i ( H j ) takes Θ( s ) to compute.

First, we draw s samples from P . As we go through the algorithm, we estimate the semi-distances ˆ w i ( H j ) as needed using these s samples. We begin by setting each ˆ W A ( H i ) to be 0 . The process of drawing samples and initializing proxy estimates takes Θ( s + n ) time.

The algorithm runs in at most T rounds. During a single round, it performs the following actions:

1. Creates Q and draws k samples from the exponential mechanism. Computing the probability of every H i according to Q takes Θ( n · s ) time. Obtaining k samples from Q can be done in Θ( n + k · log k ) time via computing the CDF and inverse sampling.
2. Invokes FIND-PROMPTING-HYPOTHESIS (Algorithm 3). For each hypothesis H j ∈ H , we compute its score η, K ,D ( H j ) . Computing the score involves computing the lift value of every hypothesis in K and then sorting, which takes Θ( k · s + k · log k ) time. Notice that our choice of s in Equation 31 dominates over our choice of log k . Therefore, FIND-PROMPTING-HYPOTHESIS takes Θ( k · n · s ) time.

3. Updates all n proxy distances (unless this is the last round). This takes Θ( n · s ) time.

Therefore, each round takes Θ( k · n · s ) .

Finally, when the algorithm halts, it samples a distribution from Q as output. Given there are at most T iterations, the total time spent is Θ( T · k · n · s ) . Substituting the values of s, T , and k (Equations 31, 32, 33) yields the desired result.

## E Computing the Prompting Scores

The wrapper algorithm iteratively seeks a hypothesis H i ∈ H that can lift a significant portion of other hypotheses in H . Concretely, we want to characterize the following cases, when H j is drawn from Q :

<!-- formula-not-decoded -->

Computing the exact probabilities in Equation 36 is costly, so we estimate them by sampling. Specifically, we draw a list of hypotheses K = [ H j 1 , · · · , H j k ] from Q , where k is the number of samples. We compute the empirical fraction of hypotheses in K that H i lifts significantly. If k is sufficiently large, then the empirical estimate closely approximates the probability.

However, if we naively count the number of hypotheses in K who have lift values above a threshold σ ′ , then the result is highly sensitive: a single change in the dataset could shift every lift value from below σ ′ to above σ ′ , thus shifting the count from 0 to n . To reduce sensitivity, we instead calculate an empirical quantile of the lift values. Specifically, we use the η/ 2 -quantile of all lift values that H i induces for sampled hypotheses in K . We call this value score η, K ,D ( H i ) , which is an approximation of the probabilities in Equation 36 and it allows us to distinguish between the two cases. In particular, if score η, K ,D ( H i ) is η/ 4 -close to the true probability, then we guarantee:

- If H i can σ ′ -lift H j with probability at least η , then score η, K ,D ( H i ) is at least σ ′ .
- If H i can σ ′ / 2 -lift H j with probability less than η/ 4 , then score η, K ,D ( H i ) is less than σ ′ / 2 .

In Section E.1, we prove in Lemma E.4 that the output of Algorithm 2 has low sensitivity with respect to changes in the input dataset D . In Section E.2, we prove that if the number of hypotheses sampled is large enough, then the value of score η, K ,D ( H i ) accurately approximates how often H i could lift hypotheses in H by σ ′ .

## Algorithm 2 Compute score η, K ,D ( H i )

| 1:   | procedure COMPUTE-SCORE( H i , η , K , D )                                                                   |
|------|--------------------------------------------------------------------------------------------------------------|
| 2:   | T = [] ▷ initialize a list to store lift values induced by H i for each H j ℓ ∈ K                            |
| 3:   | for H j ℓ ∈ K do                                                                                             |
| 4:   | Append ˆ w i ( H j ℓ ) - ˆ W ( H j ℓ ) to T ▷ assume query access to ˆ w i ( H j ℓ ) , access to ˆ W ( H j ℓ |
| 5:   | Sort T in non-increasing order                                                                               |
| 6:   | return T [ ⌈ η/ 2 · &#124;K&#124;⌉ ] ▷ return ⌈ η/ 2 · &#124;K&#124;⌉ -th largest lift value                 |

## E.1 Sensitivity of the score

In the following lemmas, we compute sensitivities to support the privacy analysis of our scoring mechanism. Throughout, we consider sensitivity with respect to the private dataset D ; all other inputs to each function are assumed to be public and fixed. Lemma E.1 shows that any quantile of a sorted list can change by at most the size of the individual perturbations of elements in that list. Lemma E.2 shows the sensitivity of the empirical semi-distance ˆ w i ( H j ) by 1 /s , and Lemma E.3 shows that the sensitivity of ˆ W A ( H j ) is also 1 /s . Finally, Lemma E.4 combines these results to prove that the overall score function score η, K ,D ( H i ) has sensitivity 2 /s .

Lemma E.1. Let x = [ x 1 , . . . , x n ] be a sorted non-increasing list. For all i ∈ [ n ] , let x ′ i := x i + δ i , where | δ i | ≤ ∆ . Sort the set { x ′ i } n i =1 into a non-increasing list y = [ y 1 , ..., y n ] . Then, for all i ∈ [ n ] , | y i -x i | ≤ ∆ .

Proof. Fix i ∈ [ n ] . We claim y i ≤ x i +∆ . Suppose for a contradiction that y i &gt; x i +∆ . Define the set of indices

<!-- formula-not-decoded -->

Because y is ordered, there are i values in y , namely y 1 , . . . , y i , that must be greater than x i +∆ . Therefore, there must be at least i values of j such that x ′ j &gt; x i +∆ , so | S | ≥ i . However, notice that if k ≥ i , then x ′ k / ∈ S because

<!-- formula-not-decoded -->

Therefore, only indices j &lt; i can be contained in S , which is a contradiction since there are only i -1 such indices. The other direction y i ≥ x i -∆ is proved similarly. To verify that this bound is tight, consider δ i = ∆ for all i ∈ [ n ] .

Lemma E.2. Let H i , H j be two hypotheses in H . With respect to the dataset D , the sensitivity of ˆ w i ( H j ) is 1 /s .

Proof. Notice that H j ( S i,j ) has no dependence on D . However, ˆ P ( S i,j ) can vary by at most 1 /s depending on if the differing data point is in S i,j . Therefore, ∆( ˆ w i ( H j )) = 1 /s .

Lemma E.3. Let H j ∈ H and A ⊆ H . With respect to the dataset D , the sensitivity of ˆ W A ( H j ) = max H k ∈ A ˆ w k ( H i ) is 1 /s .

Proof. Notice that ˆ W A ( H j ) is a maximum taken over a set of empirical semi-distances. Since the maximum is a particular quantile, by Lemma E.1 and Lemma E.2, the sensitivity of ˆ W A ( H j ) is 1 /s .

Lemma E.4. Fix H i ∈ H . Let K be a public list that consists of hypotheses in H . In other words, consider K to be given and fixed. With respect to the dataset D , the sensitivity of score η, K ,D ( H i ) is 2 /s . Precisely,

<!-- formula-not-decoded -->

Proof. Consider the sensitivity of the lift value H i induces on H j , defined as ˆ w i ( H j ) -ˆ W A ( H j ) . By LemmaE.2 and Lemma E.3, the sensitivity of each term is 1 /s , so the sensitivity of ˆ w i ( H j ) -ˆ W A ( H j ) is 2 /s . Since Algorithm 2 returns a fixed quantile of the lift values, and each lift value has sensitivity at most 2 /s , Lemma E.1 implies that score η, K ,D ( H i ) also has sensitivity 2 /s .

## E.2 Accuracy of the score

In the following lemma, we discuss the accuracy of the score. In fact, we show that the score helps us to distinguish the two cases defined in Equation 36.

Lemma E.5. Let K = [ H j 1 , ..., H j k ] be a list of hypotheses, where each H j ℓ ∈ K represents an i.i.d. sample from Q . If k is at least

<!-- formula-not-decoded -->

then with probability at least 1 -β SCO (taken over the randomness of H j ℓ 's) the following holds for every H i ∈ H :

1. If H i is ( σ ′ , η ) -prompting with respect to Q , then score η, K ,D ( H i ) is at least σ ′ .
2. If H i is not ( σ ′ / 2 , η/ 4) -prompting with respect to Q , then score η, K ,D ( H i ) is less than σ ′ / 2 .

Proof. Fix a candidate hypothesis H i ∈ H . For each H j ℓ ∈ K , use an indicator variable 1 ˆ w i ( H j ℓ ) -ˆ W A ( H j ℓ ) ≥ t , where t ∈ [0 , 1) , to determine whether H i can lift H j by at least t or not.

Notice the expectation of this indicator variable evaluates to the probability that H i lifts H j by at least t ,

<!-- formula-not-decoded -->

We set t = σ ′ in case 1 and σ ′ / 2 in case 2. We estimate the probability above using the following empirical estimators, ¯ Z σ ′ i and ¯ Z σ ′ / 2 i , defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

1. If H i is ( σ ′ , η ) -prompting with respect to Q , then

<!-- formula-not-decoded -->

Using a Chernoff bound, we obtain

<!-- formula-not-decoded -->

Therefore, with probability ≥ 1 -β SCO /n , at least η/ 2 fraction of the hypotheses in K can be lifted by H i by at least σ ′ . This implies H i is ( σ ′ , η/ 2) -empirical-prompting with respect to K . This further implies that there are at least ⌈ kη/ 2 ⌉ entries among the lift values in T that are at least σ ′ . Therefore, the ⌈ kη/ 2 ⌉ -th largest lift values must be at least σ ′ . Thus, score η, K ,D ( H i ) is at least σ ′ as desired in the statement of the lemma.

2. If H i is not ( σ ′ / 2 , η/ 4) -prompting with respect to Q , then

<!-- formula-not-decoded -->

Take X as a binomial random variable with parameter ( k, η/ 4) . Clearly, X/k is stochastically larger than ¯ Z σ ′ / 2 i , meaning for any fix threshold x ∈ (0 , 1] , the probability of X/k &gt; x is larger than the probability of ¯ Z σ ′ / 2 i &gt; x . Setting x = σ ′ / 2 attains

<!-- formula-not-decoded -->

Thus, with probability ≥ 1 -β SCO /n , fewer than η/ 2 fraction of lift values exceed σ ′ . This implies that H i is not ( σ ′ , η/ 2) -empirical-prompting with respect to K . This further implies that there are less than ⌈ kη/ 2 ⌉ entries of lift values at least σ ′ . Thus, score η, K ,D ( H i ) is less than σ ′ .

Using a union bound, we can show the above holds for all the hypotheses in H . Hence, the proof is complete.

and

## F Finding a Prompting Hypothesis

To privately find a prompting hypothesis given the scores computed in Algorithm 2, we use the sparse vector technique [DR + 14, LSL16]. We feed a stream of scores into Algorithm 3, which privately outputs either the index of the hypothesis which was detected to have a score above 3 σ ′ 4 , or ⊥ , if no hypotheses have sufficiently high scores. With high probability, we can guarantee that, if the mechanism outputs i , then score η, K ,D ( H i ) &gt; σ ′ 2 , and, if the mechanism sees a hypothesis H i with score η, K ,D ( H i ) &gt; σ ′ , it will not output ⊥ .

## Algorithm 3 An algorithm for privately finding a prompting hypothesis

```
1: procedure FIND-PROMPTING-HYPOTHESIS( ϵ, ∆ , σ ′ , η, H , K , D ) 2: ϵ 1 ← ϵ 2 3: ϵ 2 ← ϵ -ϵ 1 4: ρ ← Lap ( ∆ ϵ 1 ) 5: τ ← 3 σ ′ 4 6: ˆ τ ← τ + ρ 7: for H i ∈ H do 8: ν i ← Lap ( 2∆ ϵ 2 ) 9: score η, K ,D ( H i ) ← COMPUTE-SCORE( H i , η, K , D ) ▷ Algorithm 2 10: if score η, K ,D ( H i ) + ν i ≥ ˆ τ then 11: return H i and halt 12: return ⊥ and halt
```

Theorem 6 (Theorems 3.23 and 3.24 of [DR + 14]) . Suppose we are given parameters ϵ, ∆ &gt; 0 and σ ′ , η ∈ (0 , 1] . Assume we are given two lists of hypotheses H and K such that score η, K ,D ( · ) has sensitivity at most ∆ ≤ σ ′ ϵ 2 32 log(2 /β SVT ) . Then the FIND-PROMPTING-HYPOTHESIS Procedure in Algorithm 3 receives ϵ, ∆ , σ ′ , η, H , K , and D as its input and outputs H i or ⊥ with ϵ -privacy such that, with probability at least 1 -β SVT:

1. If FIND-PROMPTING-HYPOTHESIS outputs H i ∈ H , then score η, K ,D ( H i ) &gt; σ ′ 2 .
2. If there exists H i such that score η, K ,D ( H i ) &gt; σ ′ , then FIND-PROMPTING-HYPOTHESIS does not output ⊥ .

Proof overview: Theorem 6 follows from Theorems 3.23 and 3.24 in [DR + 14]. The privacy guarantee and the first statement of the theorem's accuracy guarantee are stated directly in [DR + 14], while the second accuracy statement is the contrapositive of the second statement in [DR + 14], Theorem 3.24.

Proof. FIND-PROMPTING-HYPOTHESIS is an instance of the sparse vector technique [DR + 14]. When ∆ is an upper bound on the sensitivity of the outputs of COMPUTE-SCORE, the privacy of the algorithm holds by Theorem 3.23 in [DR + 14].

The proof of accuracy of FIND-PROMPTING-HYPOTHESIS is a simple modification of Theorem 3.24 in [DR + 14], removing the requirement that the last query (or, in this case, hypothesis) is the only query with a score close to being above the threshold.

Let τ = 3 σ ′ 4 be the threshold of the mechanism. We want to find conditions on σ ′ such that if Algorithm 3 outputs H i , then score η, K ,D ( H i ) &gt; τ -σ ′ 4 = σ ′ 2 , and, if Algorithm 3 outputs ⊥ , then for all i , score η, K ,D ( H i ) &lt; τ + σ ′ 4 = σ ′ . We will then use these conditions to establish bounds on ∆ . Note that the second statement here is the contrapositive of the second statement in our theorem statement.

Observe that it is sufficient to find conditions on σ ′ such that, with probability at most 1 -β SVT:

<!-- formula-not-decoded -->

Recall that we don't halt at i if:

<!-- formula-not-decoded -->

Further, if we do halt at i , then:

<!-- formula-not-decoded -->

Thus, to find σ ′ satisfying Equation 37, we can equivalently find conditions on σ ′ such that:

<!-- formula-not-decoded -->

By the properties of the Laplace distribution and the union bound, we find σ ′ 4 ≥ 8∆ ϵ 2 log ( 2 n β SVT ) . This results in a bound on ∆ of:

<!-- formula-not-decoded -->