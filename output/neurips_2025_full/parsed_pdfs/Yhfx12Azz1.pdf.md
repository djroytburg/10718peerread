## Coreset for Robust Geometric Median: Eliminating Size Dependency on Outliers

Ziyi Fang Nanjing University

Lingxiao Huang ∗

Nanjing University

## Abstract

We study the robust geometric median problem in Euclidean space R d , with a focus on coreset construction. A coreset is a compact summary of a dataset P of size n that approximates the robust cost for all centers c within a multiplicative error ε . Given an outlier count m , we construct a coreset of size ˜ O ( ε -2 · min { ε -2 , d } ) when n ≥ 4 m , eliminating the O ( m ) dependency present in prior work [39, 40]. For the special case of d = 1 , we achieve an optimal coreset size of ˜ Θ( ε -1 / 2 + m n ε -1 ) , revealing a clear separation from the vanilla case studied in [40, 1]. Our results further extend to robust ( k, z ) -clustering in various metric spaces, eliminating the m -dependence under mild data assumptions. The key technical contribution is a novel non-component-wise error analysis, enabling substantial reduction of outlier influence, unlike prior methods that retain them. Empirically, our algorithms consistently outperform existing baselines in terms of size-accuracy tradeoffs and runtime, even when data assumptions are violated across a wide range of datasets.

## 1 Introduction

Geometric median, also known as the Fermat-Weber problem, is a foundational problem in computational geometry, whose objective is to identify a center c ∈ R d for a given dataset P ⊂ R d of size n that minimizes the sum of Euclidean distances from each data point to c . The given objective function, while simple and user-friendly, suffers from significant robustness problems when exposed to noisy or adversarial data [13, 15, 14, 32, 24]. For example, an adversary could add a few distant noisy outliers to the main cluster. These points could deceive the geometric median algorithm into incorrectly positioning the center closer to these outliers to minimize the cost function. Such susceptibility to outliers has been a considerable obstacle in data science and machine learning, prompting a substantial amount of algorithmic research on the subject [15, 32, 29, 53, 49].

Robust geometric median. We consider robust versions of the geometric median problem, specifically a widely-used variant that introduces outliers [13]. Formally, given an integer m ≥ 0 , the goal of robust geometric median is to find a center c ∈ R d that minimizes the objective function:

<!-- formula-not-decoded -->

where L represents the set of m outliers w.r.t. c and dist( p, c ) = ∥ p -c ∥ 2 denotes the Euclidean distance from p to center c . Intuitively, outliers capture the points that are furthest away and these are typically considered to be noise. When the number of outliers m = 0 , the robust geometric median problem simplifies to the vanilla geometric median. This problem (together with its generalization: robust ( k, z ) -clustering) has been well studied in the literature [15, 7, 29, 2, 23, 53]. However, the presence of outliers introduces significant computational challenges, particularly in the context of large-scale datasets [52, 50, 51, 53]. For example, the approximation algorithm for robust geometric

∗ Alphabetical order. Correspondence to: huanglingxiao@nju.edu.cn . LH's affiliations are the State Key Laboratory of Novel Software Technology and the New Cornerstone Science Laboratory.

Runkai Yang Nanjing University

median proposed by [52] requires ˜ O ( n d +1 ( n -m ) d ) 2 ; and [2] presents a fixed-parameter tractable (FPT) algorithm with f ( ε, m ) · n O (1) time. These challenges have driven extensive research on data reduction algorithms designed for limited hardware and time constraints.

Coreset. To tackle the computational challenge, we study coresets , which are (weighted) subsets S ⊆ P such that the clustering cost cost ( m ) ( S, c ) approximates cost ( m ) ( P, c ) within a factor of (1 ± ε ) for all center sets c ∈ R d , where ε &gt; 0 is a given error parameter. 3 A coreset preserves key geometric information while substantially reducing the dataset size, thus serving as a compact proxy for the original dataset P . Consequently, applying existing approximation algorithms to the coreset significantly improves computational efficiency. Moreover, coresets can be reused in future analyses of P , reducing redundant computation and saving storage resources. Finally, the size of the coreset reflects the intrinsic complexity of the problem, making the study of optimal coreset size a research topic of independent interest.

Coreset for vanilla geometric median ( m = 0 ) has been extensively studied across different dimensions d [28, 16, 18, 19, 1, 22] (see Appendix A.2 for details). When d = 1 , [37] proposed a coreset of size ˜ Θ( ε -1 / 2 ) . Subsequently, [1] extended this result by constructing an optimal coreset of size ˜ Θ( ε -d/ ( d +1) ) for dimension d = O (1) . Moreover, for high dimensions d &gt; ε -2 , the optimal coreset size is ˜ Θ( ε -2 ) [18, 21]. In contrast, the study of coreset for robust geometric median is far from optimal, even in the simplest case of d = 1 . Early work either required an exponentially large coreset size [27] or involved relaxing the outlier constraint [39]. A notable recent advancement by [39] overcame these limitations, proposing a coreset of size O ( m ) + ˜ O ( ε -3 · min { ε -2 , d } ) , using a hierarchical sampling framework based on [9]. Further improvements reduced the coreset size to O ( m ) + ˜ O ( ε -2 · min { ε -2 , d } ) [40]. A recent paper [42] presented a coreset size O ( mε -1 + Vanilla size ) via a novel reduction from the robust case to the vanilla case.

All previous coreset sizes for robust geometric median contain the factor m . However, the outlier number m could approach Ω( n ) in real-world scenarios [25, 12, 30, 10]. For instance, the PageBlocks dataset has n = 5473 points with m ≈ 0 . 1 n outliers [10]. Such m results in an inefficient coreset size, raising a natural question: Can we eliminate the O ( m ) term from the coreset size?

At first glance, one might assume the answer is negative, as [39] establishes a coreset lower bound of Ω( m ) . However, their worst-case instance critically relies on an extremely large number of outliers, specifically with m = n -1 . This requirement can be relaxed to n -m = o ( n ) (see Theorem 1.1). Motivated by this, we investigate the possibility of eliminating the O ( m ) dependency when n -m = Ω( n ) , and show that this condition is both necessary and sufficient. Under this setting, we obtain a coreset of size ˜ O ( ε -2 · min { ε -2 , d } ) (see Theorem 1.3).

Nevertheless, this size is substantially larger than that of the vanilla case across all dimensions. For instance, when d = 1 , our coreset size is ˜ O ( ε -2 ) , whereas the vanilla case achieves a size of only ˜ Θ( ε -1 / 2 ) [37]; thus, our bound is likely not optimal. On the other hand, our lower bound in Theorem 1.1 suggests that the optimal coreset size should indeed depend on m . This raises a natural question: What is the optimal coreset size for robust geometric median, and how does it vary with m ? Answering this question sheds light on when the complexity of robust geometric median fundamentally diverges from that of the vanilla case.

## 1.1 Our contributions

In this paper, we study (optimal) coreset sizes for the robust geometric median problem. See Table 3 in the appendix for a summary. We begin with a lower bound result (proof in Section C).

Theorem 1.1 (Coreset lower bound for robust geometric median) . Let 0 &lt; ε &lt; 0 . 5 and n &gt; m ≥ 1 . There exists a dataset P ⊂ R of size n such that any ε -coreset of P for the robust geometric median problem must have size Ω( m n -m ) .

2 Here, ˜ O ( n ) denotes O ( n · polylog( n )) , hiding logarithmic factors.

3 Here, the computation of cost ( m ) ( S, c ) excludes points of total weight m ; see Appendix A.1 for the formal definition.

This theorem indicates that when n -m = o ( n ) , which implies m = Θ( n ) , the coreset size is at least Ω ( m n -m ) = m o ( m ) , which depends on m . This extends the previous result in [39] to general m . Thus, n -m = Ω( n ) is a necessary condition for eliminating the O ( m ) term from the coreset size.

Accordingly, we assume n ≥ 4 m for the algorithmic results, where the constant 4 ensures the inlier number n -m is sufficiently larger than the outlier number m . We first state our result when d = 1 . Theorem 1.2 (Optimal coreset for robust 1D geometric median) . Let n, m ≥ 1 be integers and ε ∈ (0 , 1) . Assume n ≥ 4 m . There is a linear algorithm that given dataset P ⊂ R of size n , outputs an ε -coreset for robust 1D geometric median of size ˜ O ( ε -1 2 + m n ε -1 ) in O ( n ) time. Moreover, there exists a dataset X ⊂ R of size n such that any ε -coreset must have size Ω( ε -1 2 + m n ε -1 ) .

This theorem provides the optimal coreset size for the robust 1D geometric median problem. Since m ≤ n , the coreset size is at most ˜ O ( ε -1 ) , which is independent of m . Thus, we eliminate the O ( m ) dependency in the coreset size compared to previous bounds [40, 42]. In particular, relative to the bound O ( m ) + ˜ O ( ε -2 ) in [40], our result also improves the ε -dependence from ε -2 to at most ε -1 .

The theorem also shows how the coreset size increases as m grows. When m ≤ √ εn , our coreset size is dominated by ˜ O ( ε -1 / 2 ) since m n ε -1 ≤ ε -1 / 2 . This size matches the coreset size ˜ O ( ε -1 / 2 ) for vanilla 1D geometric median as given in [40]. As m increases from √ εn to n 4 , our coreset size is dominated by the term ˜ O ( m n ε -1 ) , which is a linear function of m growing from ˜ O ( ε -1 / 2 ) to ˜ O ( ε -1 ) . This size suggests that the complexity of robust 1D geometric median is higher than vanilla 1D geometric median in this range of outlier number m .

We remark that the size lower bound Ω( ε -1 2 + m n ε -1 ) holds for any dimension d ≥ 1 . In contrast, for d = O (1) , the tight coreset size for the vanilla geometric median is ˜ Θ( ε -d/ ( d +1) ) [1]. This implies that when m ≥ Ω( n · ε d/ ( d +1) ) , our coreset size lower bound exceeds Ω( ε -d/ ( d +1) ) , resulting in a gap between the coreset sizes for the robust and vanilla versions of geometric median.

Theorem 1.3 (Coreset for robust geometric median in R d ) . Let n, m, d ≥ 1 be integers and ε ∈ (0 , 1) . Assume n ≥ 4 m . There exists a randomized algorithm that given a dataset P ⊂ R d of size n , outputs an ε -coreset for the robust geometric median problem of size ˜ O ( ε -2 min { ε -2 , d } ) in O ( nd ) time.

Compared to previous bounds O ( m ) + ˜ O ( ε -2 min { ε -2 , d } ) [40] and O ( mε -1 ) + ˜ O ( ε -2 ) [42], this theorem eliminates the O ( m ) term in the coreset size when n ≥ 4 m . This result can be extended to various metric spaces (Section E.2).

Finally, we extend this theorem to handle the robust ( k, z ) -clustering problem (Definition F.1), which encompasses robust k -median ( z = 1 ) and robust k -means ( z = 2 ). To capture the additional complexity introduced by k , we propose the following geometric assumption.

Assumption 1.4 (Assumptions for robust ( k, z ) -clustering) . Given a dataset P ⊂ R d of n points and an ε -approximate center set C ⋆ ⊂ R d of P for robust ( k, z ) -clustering. Define P ⋆ I := arg min P I ⊂ P, | P I | = n -m ∑ p ∈ P I dist( p, C ⋆ ) to be the inlier points w.r.t. C ⋆ , where dist( p, C ⋆ ) := min c ∈ C ⋆ dist( p, c ) . Let { P ⋆ 1 , . . . , P ⋆ l } ⊂ P ⋆ I denote the k inlier clusters induced by C ⋆ , where each P ⋆ i contains points in P ⋆ I whose closest center is c ⋆ i . We assume the followings: 1) min i ∈ [ k ] | P ⋆ i | ≥ 4 m ; 2) max p ∈ P ⋆ I dist( p, C ⋆ ) z ≤ 4 k ∑ p ∈ P ⋆ I dist( p, C ⋆ ) z / | P ⋆ I | .

The first condition directly generalizes the assumption n ≥ 4 m , which requires that the size of each inlier cluster is more than the outlier number m . The second excludes 'remote inlier points' to C ⋆ , which could play a similar role as outliers. This condition holds for several real-world datasets (see Table 4), demonstrating its practicality. Now we propose the following result.

Theorem 1.5 (Coreset for robust ( k, z ) -clustering) . Let n, k, d ≥ 1 be integers and ε ∈ (0 , 1) . There exists a randomized algorithm that given a dataset P ⊂ R d of size n satisfying Assumption 1.4, outputs an ε -coreset for robust ( k, z ) -clustering of size ˜ O ( k 2 ε -2 z min { ε -2 , d } ) in O ( nkd ) time.

It improves upon the previous result O ( m ) + ˜ O ( k 2 ε -2 z min { ε -2 , d } ) in [39, 40] by eliminating the O ( m ) term. Furthermore, the result can also be extended to various metric spaces (Section F.3).

Empirically, in Section 4, we evaluate the performance of our coreset algorithms on six real-world datasets. We compare the size-error tradeoff against baselines [39, 40], and across all tested sizes, our

algorithms consistently achieve lower empirical error. For instance, for robust geometric median on the Census1990 dataset, our method produces a coreset of size 1000 with an empirical error of 0.012, while the baseline produces a coreset of size 2300 with an empirical error slightly higher than 0.013 (Figure 2). Moreover, our algorithms provide a speedup compared to baselines for achieving the same level of empirical error (see e.g., Table 2). Additionally, we show that our algorithms remain practically effective, regardless of which data assumption in the theoretical results is violated, further demonstrating the practical utility of our algorithms (Section G).

## 1.2 Technical overview

We outline the technical ideas and novelty for Theorems 1.2 and 1.3. Our approach introduces a novel non-component-wise error analysis for coreset construction, enabling a substantial reduction in the number of outlier points rather than preserving them all. In contrast, previous algorithms divide the dataset P into multiple components, and their approach requires aligning the number of outliers between P and S in every component. This idea, as we will show, inherently introduces an Ω( m ) -sized coreset. Furthermore, for the 1D case, our new algorithm offers a more refined partitioning of inlier points, enabling an adaptation of the vanilla coreset construction and leading to the optimal coreset size.

## 1.2.1 Overview for Theorem 1.2 ( d = 1 )

Revisiting coreset construction for vanilla 1D geometric median. Recall that [37] first partitions dataset P = { p 1 , . . . , p n } ⊂ R into ˜ O ( ε -1 / 2 ) buckets, where each bucket B i is a consecutive subsequence { p l , p l +1 . . . , p r } (see Definition 2.1). They select a mean point µ ( B ) , assign a weight | B | for each bucket B , and output their union as a coreset such that ∀ c ∈ R ,

<!-- formula-not-decoded -->

This follows that only the bucket containing c contributes a non-zero error at most ε · cost( P, c ) (see Lemma B.1). To handle the additional outliers for the robust case, a natural idea is to extend Inequality (2) in the following manner: for each center c ∈ R and each tuple ( m 1 , . . . , m T ) ∈ Z T ≥ 0 of outlier numbers per bucket (with ∑ i ∈ [ T ] m i = m ),

<!-- formula-not-decoded -->

This bounds the induced error | cost ( m i ) ( B i , c ) -( | B i | -m i ) · dist( µ ( B i ) , c ) | of each bucket B i when aligning the outlier numbers within this bucket for dataset P and coreset S .

Prior work [39, 40] utilized this component-wise error analysis for robust coreset construction. They show that at most three buckets B i could induce a non-zero error, including a bucket that contains c and two 'partially intersected' buckets with 0 &lt; m i &lt; | B i | that contain both inlier and outlier points relative to c . Thus, to prove Inequality (3), it suffices to ensure that the maximum induced error | cost ( m i ) ( B i , c ) -( | B i | -m i ) · dist( µ ( B i ) , c ) | of these three buckets is at most ε · cost ( m ) ( P, c ) / 3 . However, it is possible that cost ( m ) ( P, c ) ≪ cost( P, c ) , which presents a significant challenge for this guarantee compared to the vanilla case. To overcome this obstacle, prior work [39, 40] includes the 'outmost' m points of P into the coreset to ensure zero induced error from them, resulting in an O ( m ) size dependency.

First attempt to eliminate the O ( m ) dependency. We first observe that a subset P M = { p m +1 , . . . , p n -m } ⊂ P of size n -2 m acts as inliers w.r.t. any center c ∈ R . The condition n ≥ 4 m guarantees the existence of this P M with | P M | = n -2 m ≥ 2 m . Since cost ( m ) ( P, c ) ≥ cost( P M , c ) by the construction of P M , cost ( m ) ( P, c ) is intuitively not 'too small', which is useful for eliminating the O ( m ) dependency in analysis. A natural idea is to apply the vanilla coreset construction method given by [37] to P M and also include P -P M in the coreset S . This attempt yields a coreset of size 2 m + ˜ O ( ε -1 / 2 ) , which already improves the previous bound of O ( m ) + ˜ O ( ε -2 ) by [40]. To eliminate the O ( m ) term, it is essential to significantly reduce points in P -P M .

Our first attempt is to further decompose P -P M into buckets and utilize the component-wise error analysis in Inequality (3). As discussed earlier, the critical step is to ensure | cost ( m i ) ( B i , c ) -( | B i | -m i ) · dist( µ ( B i ) , c ) | ≤ ε 3 · cost ( m ) ( P, c ) for two partially intersecting buckets induced by c with 0 &lt; m i &lt; | B i | . To achieve this, we study the relative location of such buckets with respect to c

(Lemma 2.3), enabling us to partition the buckets based on the scale of cost ( m ) ( P, c ) , similar to the vanilla method (see Lines 1-5 of Algorithm 1). However, this partitioning approach is applicable only for c ∈ P M , where the scale of cost ( m ) ( P, c ) is well-controlled (see Lemma D.1). Consequently, the challenge remains to control the induced error of these buckets for c / ∈ P M .

Obstacle for c / ∈ P M . Unfortunately, to ensure Inequality (3) holds for c / ∈ P M , the number of constructed buckets in P -P M must be at least Ω( m ) . To see this, we provide an illustrative example in Section A.3, where there is a collection P 1 of ( n -m ) points condensed into a small interval, and a collection P 2 of m points that are exponentially far from each other. We show that for any bucket containing two points p, q ∈ P 2 with p &lt; q , the induced error of this bucket could be much larger than cost ( m ) ( P, c ) . Thus, to ensure Inequality (3) holds, all points in P 2 must be included in the coreset, leading to a size of Ω( m ) . Therefore, a new error analysis beyond Inequality (3) is required.

Key approach. W.L.O.G., we assume c &gt; p n -m , i.e., c is to the right of P M . The key idea is to develop a novel non-component-wise error analysis beyond Inequality (3): we regard | cost ( m ) ( P, c ) -cost ( m ) ( S, c ) | as a single entity and study how it changes as c shifts to the right from p n -m . Let f ′ P ( c ) and f ′ S ( c ) denote the derivative of cost ( m ) ( P, c ) and cost ( m ) ( S, c ) respectively. Then the induced error can be rewritten as | cost ( m ) ( P, c ) -cost ( m ) ( S, c ) | ≤ | cost ( m ) ( P, p n -m ) -cost ( m ) ( S, p n -m ) | + ∫ c p n -m | f ′ P ( x ) -f ′ S ( x ) | dx . We can ensure cost ( m ) ( P, p n -m ) = cost ( m ) ( S, p n -m ) by an extra bucket-partitioning step (see Line 8 of Algorithm 1). Thus, it suffices to ensure ∫ c p n -m | f ′ P ( x ) -f ′ S ( x ) | dx ≤ ε · cost ( m ) ( P, c ) .

A key geometric observation is that f ′ P ( c ) equals the difference between the number of inliers in P located to the left of c and those to the right of c ; a similar relationship holds for f ′ S ( c ) . For c , let m i and m ′ i denote the number of outliers in bucket B i relative to dataset P and coreset S , respectively. By the geometric observation, we conclude that | f ′ P ( c ) -f ′ S ( c ) | ≤ ∑ i | m i -m ′ i | +2 | B c | , where B c denotes the bucket containing c . Therefore, ∫ c p n -m | f ′ P ( x ) -f ′ S ( x ) | dx ≤ ( ∑ i | m i -m ′ i | +2 | B c | ) · dist( p n -m , c ) (see Lemma D.3). Thus, it suffices to ensure ( ∑ i | m i -m ′ i | +2 | B c | ) · dist( p n -m , c ) ≤ ε · cost ( m ) ( P, c ) . This desired property can be achieved by limiting the size of each bucket in P -P M to be at most O ( εn ) , which ensures ∑ i | m i -m ′ i | ≤ O ( εn ) for all c ∈ P M (see Lemma D.2).

We remark that, due to the misalignment of outlier counts, a single bucket may induce an arbitrarily large error in our analysis. However, these bucket-level errors can cancel out, and the overall error remains well controlled. To the best of our knowledge, this represents the first non-component-wise error analysis in the coreset literature, which may be of independent research interest. To demonstrate the power of our new error analysis, we apply it to the aforementioned obstacle instance-a case that previous component-wise analyses cannot solve (Appendix A.3).

Coreset size analysis. Note that the coreset size equals the number of buckets. We partition P M into ˜ O ( ε -1 / 2 ) buckets using the vanilla coreset construction method [37]. The analysis for c ∈ P M and c / ∈ P M results in at most ˜ O ( ε -1 / 2 ) buckets for the former and O ( m n ε -1 ) buckets for the latter. Therefore, the total coreset size is ˜ O ( ε -1 / 2 + m n ε -1 ) (see Lemma 2.4).

This coreset size is shown to be tight. To see this, we construct a worst-case example in Section D.6, where the m outliers are partitioned into m n ε -1 intervals, each containing εn points, with the interval scales increasing exponentially. If the coreset omits all points from any such interval, each point within it contributes an error of 2 · cost ( m ) ( P,c ) n , resulting in a total error of 2 ε · cost ( m ) ( P, c ) -which is unacceptably large. Thus, to control the error, the coreset must include at least one point from each interval, yielding a lower bound of m n ε -1 on the coreset size.

## 1.2.2 Overview for Theorem 1.3 (general d )

The obstacle of the traditional component-wise analysis remains in the high-dimensional case, motivating us to adopt the non-component-wise analysis framework. However, due to the increased complexity of candidate center distribution, a straightforward extension of the 1D analysis would involve using an ε -net to partition the center space, as in [35], but this introduces an O ( ε -d ) factor in the coreset size. To overcome this, we leverage the concept of the ball range space, as explored in previous works [9, 39], which allows us to effectively describe high-dimensional spaces.

Our Algorithm 2 takes a uniform sample S O of size ˜ O ( ε -2 min { ε -2 , d } ) from the 'outmost' m points of P to include in the coreset, in contrast to including them all as in [39, 38]. To analyze the error induced by S O , we examine errors caused by 'outlier-misaligned' points-those that act as outliers (or inliers) in P but as inliers (or outliers) in S with respect to a fixed c . The induced error for each such point is bounded by cost ( m ) ( P,c ) m when n ≥ 4 m (see Lemmas E.3, E.4, and E.5). To ensure the total error remains within O ( ε ) · cost ( m ) ( P, c ) , it suffices for the number of such outlier-misaligned points to be O ( εm ) . The key geometric insight is that this condition holds when S O serves as an ε -approximation for the ball range space on L ⋆ (see Lemma E.6). This approximation is guaranteed by ensuring | S O | = ˜ O ( ε -2 min { ε -2 , d } ) , as established in Lemma E.2.

To our knowledge, this analysis is the first to apply the range space argument to outlier points. The range space argument leverages the fact that the VC dimension of R d is at most O ( d ) and can be further reduced to ˜ O ( ε -2 ) by dimension reduction. This enables us to generalize results to various metric spaces via the notion of VC dimension (or doubling dimension), as well as to robust ( k, z ) -clustering; see Appendix E.2 and F.

## 2 Optimal coreset size when d = 1

Let P = { p 1 , . . . , p n } ⊂ R with p 1 &lt; p 2 &lt; . . . &lt; p n . Denote L ( c ) := arg min L ⊆ P, | L | = m cost( P -L, c ) to be the set of outliers of P w.r.t. a center c , and P ( c ) I = P -L ( c ) to be the set of inliers. Let c ⋆ := arg min c ∈ R cost ( m ) ( P, c ) be an optimal center. Let L ⋆ = L ( c ⋆ ) and P ⋆ I = P ( c ⋆ ) I .

Buckets and cumulative error. We introduce the definition of bucket proposed by [35] associated with related notions, which is useful for coreset construction when d = 1 [34, 37].

Definition 2.1 (Bucket and associated statistics) . Abucket B is a continuous subset { p l , p l +1 , . . . , p r } of P for some 1 ≤ l ≤ r ≤ n . Let N ( B ) := r -l +1 represents the number of points within B , µ ( B ) := ∑ p ∈ B p N ( B ) represents the mean point of B , and δ ( B ) := ∑ p ∈ B | p -µ ( B ) | represents the cumulative error of B .

A basic idea for vanilla coreset construction in 1D case is to partition P into multiple buckets B and then retain a point µ ( B ) with weight N ( B ) as the representative point of B in coreset. This idea works since each bucket induces an error at most δ ( B ) ; see Lemma B.1. Thus, we have the following theorem that provides the optimal coreset for vanilla 1D geometric median [37]

Theorem 2.2 (Coreset for vanilla 1D geometric median [37]) . There exists algorithm A , that given an input data set P ⊂ R and ε ∈ (0 , 1) , A ( P, ε ) outputs an ε -coreset of P for vanilla 1D geometric median with size ˜ O ( ε -1 2 ) in O ( | P | ) time.

We will apply A to the 'inlier subset' of P , say the set of middle n -2 m points P M = { p m +1 , . . . , p n -m } . Let P L = { p 1 , . . . , p m } and P R = { p n -m +1 , . . . , p n } . Any continuous subsequence of length n -m of P must contain P M , implying that all points in P M must be inliers w.r.t. any center c ∈ R . This motivates us to apply the vanilla method A to P M .

As discussed in Section 1.2, we then partition P L an P R into buckets, which requires an understanding of the relative position between the inlier subset P ( c ) I and c . Define r max := max p ∈ P ⋆ I | p -c ⋆ | , c L := c ⋆ -r max and c R := c ⋆ + r max . We present the following lemma (proof in Appendix B.2).

̸

Lemma 2.3 (Location of P ( c ) I ) . Let c ∈ P M be a center with c &lt; c ⋆ and P ( c ) I = P ⋆ I . Let p l be the leftmost point of P ( c ) I . Then dist( p l , c L ) ≤ 2 · dist( c, c ⋆ ) .

A symmetric observation can be made for c &gt; c ⋆ . Note that cost ( m ) ( P, c ) &gt; cost( P M , c ) ≥ O ( n ) · dist( c, c ⋆ ) ≥ O ( n ) · dist( p l , c L ) . This lower bound for cost ( m ) ( P, c ) motivates us to select the cumulative error bound for the bucket containing point p l as εn · dist( p l , c L ) . Thus, we partition P L ∪ P R into disjoint blocks according to points' distance to c L and c R , a concept inspired by [37]. Concretely, we partition the four collections P L ∩ ( -∞ , c L ) , P L ∩ [ c L , ∞ ) , P R ∩ ( -∞ , c R ] , P R ∩ ( c R , ∞ ) into blocks, respectively. Due to symmetry, we only define the blocks in P L ∩ ( -∞ , c L ) below. Blocks for other parts follow similarly and are provided in Appendix B.3; see Figure 1 for a visualization.

Figure 1: Illustration of the block partition. The blue square marks the optimal solution c ⋆ , and the blue triangle c L denotes the left boundary of the inlier set P ⋆ I , with distance r max = dist( c L , c ⋆ ) . Figure 1(a) partitions the one-dimensional space left of P M into disjoint blocks based on each point's position relative to c L : points farther than r max form B far , and those within 2 εr max form B 0 . Figure 1(b) shows the logarithmic subdivision of inner blocks B ( L ) i within distance r max from c L .

<!-- image -->

Outer blocks ( B far ): Define B ( L ) far as the set of points that are far from c L , where

<!-- formula-not-decoded -->

Inner blocks ( B ): Define B as the set of points that are close to c i ( L ) 0 L , where

<!-- formula-not-decoded -->

For the remaining points, partition them into blocks B ( L ) i based on exponentially increasing distance ranges for i = 1 , . . . , ⌈ log 2 ( ε -1 ) ⌉ , where

<!-- formula-not-decoded -->

The algorithm. Our algorithm (Algorithm 1) consists of three stages. In Stage 1, we construct a coreset S M for P M using Algorithm A for vanilla 1D geometric median (Theorem 2.2), ensuring that cost( S M , c ) ∈ (1 ± ε ) · cost( P M , c ) for any center c ∈ R . In Stage 2, we divide sets P L and P R into outer and inner blocks by Equations (4)-(6), and greedily partition these blocks into disjoint buckets B with bounded δ ( B ) and N ( B ) in Lines 3-6. In Stage 3, we ensure that cost ( m ) ( P, p n -m ) = cost ( m ) ( S, p n -m ) and cost ( m ) ( P, p m +1 ) = cost ( m ) ( S, p m +1 ) to control induced error when c / ∈ P M . Finally, we add the mean point µ ( B ) of each bucket B with weight N ( B ) into S O , and return S O ∪ S M as the coreset of P .

By construction, the coreset size | S | is exactly the number of buckets. Therefore, we have the following lemma that proves the coreset size in Theorem 1.2. Its proof can be found in Section D.2.

Lemma 2.4 (Number of buckets) . Algorithm 1 constructs at most ˜ O ( ε -1 2 + m n ε -1 ) buckets.

Key proof idea of Theorem 1.2. Using Lemma 2.3, we can control the error from partially intersected buckets when c ∈ P M (Lemma D.1). For c / ∈ P M , let m i and m ′ i be the number of outliers in bucket B i for P and S , respectively. We show that the total misaligned outliers satisfy ∑ i ∈ [ q ] | m i -m ′ i | ≤ εn 4 (Lemma D.2). In this case, the error is also bounded by | cost ( m ) ( P, c ) -cost ( m ) ( S, c ) | ≤ ( ∑ i ∈ [ q ] | m i -m ′ i | + εn 8 ) · dist( c, P M ) ≤ ε · cost ( m ) ( P, c ) (Lemma D.3). The complete proof can be found in Appendix D.

## 3 Improved coreset sizes for general d ≥ 1

We present Algorithm 2 for Theorem 1.3. In Line 1, we construct L ⋆ as the set of outliers of P w.r.t. c ⋆ and P ⋆ I = P -L ⋆ as the set of inliers. We construct coresets for P ⋆ I and L ⋆ separately. In Line 2, we take a uniform sample S O from L ⋆ as the coreset of L ⋆ . This step is the key for eliminating the O ( m ) dependency in the coreset size. In Line 3, we use the following theorem by [40] to construct a coreset S I for P ⋆ I . The coreset S is the union of S O and S I (Line 5). In Section F, we show how to generalize this algorithm to robust ( k, z ) -clustering (Algorithm 3).

Theorem 3.1 (Restatement of corollary 5.4 in [40]) . There exists a randomized algorithm A d that in O ( nd ) time constructs a weighted subset S I ⊆ P ⋆ I of size ˜ O ( ε -2 min { ε -2 , d } ) , such that for every dataset P O of size m , every integer 0 ≤ t ≤ m and every center c ∈ R d , | cost ( t ) ( P O ∪ P ⋆ I , c ) -cost ( t ) ( P O ∪ S I , c ) | ≤ ε · cost ( t ) ( P O ∪ P ⋆ I , c ) + 2 ε · cost( P ⋆ I , c ⋆ ) .

## Algorithm 1 Coreset construction for 1D

Input: A dataset P = { p 1 , . . . , p n } ⊂ R with p 1 &lt; . . . &lt; p n , and ε ∈ (0 , 1) . Output: An ε -coreset S

- 1: Set P M ←{ p m +1 , . . . , p n -m } , P L ←{ p 1 , . . . , p m } , P R ←{ p n -m +1 , . . . , p n } .
- 3: Compute an optimal center c ⋆ of P for robust 1D geometric median. Let P ⋆ I be the set of inliers w.r.t. c ⋆ , and r max := max p ∈ P ⋆ I dist( p, c ⋆ ) , c L ← c ⋆ -r max , c R ← c ⋆ + r max .
- 2: Construct S M ←A ( P M , ε 3 ) by Theorem 2.2.
- 4: Given c L and c R , divide P L and P R into outer blocks B ( L ) far and B ( R ) far by Equation (8), and inner blocks B ( L ) i , B ( LR ) i , B ( RL ) i and B ( R ) i ( 0 ≤ i ≤ ⌈ log 2 ( ε -1 ) ⌉ ) by Equations (9)-(10).
- 5: For each non-empty inner block B i , divide B i into disjoint buckets { B i,j } j ≥ 0 in a greedy way: each bucket B i,j is a maximal set with δ ( B i,j ) ≤ 2 i · ε 2 nr max 288 and N ( B i,j ) ≤ εn 16 .
- 6: If B ( L ) far is non-empty, divide B far into disjoint buckets { B ( L ) far ,j } j ≥ 0 in a greedy way: each bucket B ( L ) far ,j is a maximal set with N ( B ( L ) far ,j ) ≤ εn 16 . The same for B ( R ) far .
- 7: Compute the inlier set P ( p m +1 ) I with respect to center c = p m +1 and the inlier set P ( p n -m ) I with respect to center c = p n -m respectively.
- 8: If there exists some bucket B such that both B ∩ P ( p m +1 ) I and B \ P ( p m +1 ) I are non-empty, divide B into two buckets B ∩ P ( p m +1 ) I and B \ P ( p m +1 ) I . Do the same thing for P ( p n -m ) I .
- 9: For every B , add µ ( B ) with weight N ( B ) into S O .
- 10: Return S ← S O ∪ S M .

## Algorithm 2 Coreset Construction for General d

Input: A dataset P ⊂ R d , ε ∈ (0 , 1) and an O (1) -approximate center c ⋆ ∈ R d Output: An ε -coreset S

- 1: L ⋆ ← arg min L : | L | = m cost( P -L, c ⋆ ) , P ⋆ I ← P -L ⋆
- 2: Uniformly sample S O ⊆ L ⋆ of size ˜ O ( ε -2 min { ε -2 , d } ) . Set ∀ p ∈ S O , w O ( p ) ← m | S O | .
- 3: Construct ( S I , w I ) ←A d ( P ⋆ I ) by Theorem 3.1.
- 4: For any p ∈ S O , define w ( p ) = w O ( p ) and for any p ∈ S I , define w ( p ) = w I ( p ) .
- 5: Return S ← S O ∪ S I and w ;

The theorem tells that S I serves as an ε -coreset for the combination of P ⋆ I and any possible set of outliers P O . The flexible choice of P O is useful for our analysis. To estimate the error induced by S O , we introduce the key lemma below, whose proof can be found in Section E.1.

Lemma 3.2 (Induced error of S O ) . For any center c ∈ R d , we have | cost ( m ) ( P, c ) -cost ( m ) ( S O ∪ P ⋆ I , c ) | ≤ O ( ε ) · cost ( m ) ( P, c ) .

Proof of Theorem 1.3. Fix a center c ∈ R d . Let P O = S O and t = m in Theorem 3.1, we have | cost ( m ) ( S O ∪ P ⋆ I , c ) -cost ( m ) ( S O ∪ S I , c ) | ≤ ε · cost ( m ) ( S O ∪ P ⋆ I , c ) + 2 ε · cost( P ⋆ I , c ⋆ ) . By Lemma 3.2, we have | cost ( m ) ( P, c ) -cost ( m ) ( S O ∪ P ⋆ I , c ) | ≤ O ( ε ) · cost ( m ) ( P, c ) . Adding the two inequalities above, we have | cost ( m ) ( P, c ) -cost ( m ) ( S, c ) | ≤ O ( ε ) · cost ( m ) ( P, c ) .

The runtime is dominated by Line 1 and Line 3 that costs O ( nd ) time by Theorem 3.1, making the total overhead O ( nd ) . This completes the proof of Theorem 1.3.

## 4 Empirical results

We implement our coreset construction algorithm and compare its performance to several baselines. All experiments are conducted on a PC with an Intel Core i9 CPU and 16GB of memory, and the algorithms are implemented in C++ 11.

Baselines. We compare our algorithm with two baselines: 1) Method HJLW23 proposed by [39], which directly includes L ⋆ in the coreset and samples points from P ⋆ I . 2) Method HLLW25 proposed by [40], improves the sample size from P ⋆ I in [39].

Figure 2: Tradeoff between coreset size | S | and empirical error ̂ ε ( S ) .

<!-- image -->

Setup. We conduct experiments on six datasets from diverse domains, including social networks, demographics, and disease statistics, with sample sizes ranging from ( 10 4 ) to ( 10 5 ) and feature dimensions from 2 to 68 , as summarized in Table 4. These datasets cover those used in baseline [39], ensuring fair comparison. In each dataset, numerical features are extracted to create a vector for each record and the outlier number is set to 2% of the dataset size. To simplify computation, we subsample 10 5 points from the Twitter and Census1990 datasets, and 10 4 points from the Athlete and Diabetes datasets, respectively. We use k -means++ to compute an approximate center c ⋆ .

Size-error tradeoff. We evaluate the tradeoff between coreset size and empirical error. Given a (weighted) subset S ⊆ P and a center c ⊂ R d , we define the empirical error ̂ ε ( S, c ) := | cost ( m ) ( P,c ) -cost ( m ) ( S,c ) | cost ( m ) ( P,c ) , where lower values indicate better coreset performance for c . It is difficult to estimate the performance for every center. So we sample 500 centers c i ∈ R d , where each c i is drawn uniformly from P without replacement. Like in the literature [39], we evaluate the empirical error ̂ ε ( S ) := max i ∈ [500] ̂ ε ( S, c i ) . We vary the coreset size | S | from m to 2 m , and compute the empirical error ̂ ε ( S ) . For each size and each algorithm, we run the algorithm 10 times, compute their empirical errors ε ( S ) , and report the average of 10 empirical errors.

̂

Figure 2 presents the empirical results illustrating the size-error tradeoff on the Census1990 , Twitter and Adult datasets. As shown in Figures 2, our coreset algorithm consistently achieves the lowest empirical error among all methods. Moreover, unlike the baselines, which require a coreset of size at least m , our method attains the same level of error with a coreset size smaller than m . For example, with the Census1990 dataset, our method yields a coreset of size 1000 with an empirical error of 0.012, while the best baseline needs size 2300 to achieve a worse error of 0.013. Results for other datasets are presented in Section G. We also perform statistical tests across six real-world datasets by comparing the ratio of empirical errors between our algorithm and baselines, which further demonstrates that our algorithm consistently outperforms the baselines; see Table 1. The results show that both ̂ ε ( S 2 ) / ̂ ε ( S 1 ) and ̂ ε ( S 3 ) / ̂ ε ( S 1 ) are consistently bigger than 1 , demonstrating that our coreset consistently yields lower empirical error than the baselines. This confirms the applicability of our coreset across real-world datasets.

Speed-up baselines. In this experiment, we compare the coreset of size 2 m constructed by the HLLW25 baseline and coreset of size m constructed by Algorithm 2. We repeat the experiment 10 times and report the averages. The result is listed in Table 2. The construction time of our coreset is similar to that of the baseline HLLW25 . However, our algorithm achieves a speed-up over HLLW25 (a 2 × reduction in the running time on the coreset), while achieving the same level of empirical error.

Additional experiment. In Section G.1, we demonstrate the robustness of our Algorithm 2 when the assumption n ≥ 4 m is violated or dataset is noisy. In Sections G.2, G.3, and G.4, we implement our algorithms for the 1D case, robust k -median, and k -means, respectively, and conduct similar experiments. The results show improved performance on real-world datasets, even when the theoretical data assumptions are violated, further highlighting the practical robustness of our algorithms.

## 5 Conclusion and future work

We investigate coreset construction for robust geometric median problem, successfully eliminating the size dependency on the number of outliers. Specifically, for the 1D Euclidean case, we achieve the first optimal coreset size. Furthermore, our results generalize to robust clustering applications. Empirically, our algorithms achieve a superior size-error balance and a runtime acceleration.

Table 1: Statistical comparison of different coreset construction methods for robust geometric median. The coreset S 1 represents our coreset, S 2 represents the coreset constructed by the baseline HJLW23 , and S 3 the coreset constructed by baseline HLLW25 . For each empirical error ratio ̂ ε ( S 2 ) / ̂ ε ( S 1 ) and ̂ ε ( S 3 ) / ̂ ε ( S 1 ) , we report the mean value over 20 runs, with the subscript indicating the standard deviation.

| Coreset Size   | Census1990                | Census1990                | Twitter                   | Twitter                   |
|----------------|---------------------------|---------------------------|---------------------------|---------------------------|
| Coreset Size   | ̂ ε ( S 2 ) / ̂ ε ( S 1 ) | ̂ ε ( S 3 ) / ̂ ε ( S 1 ) | ̂ ε ( S 2 ) / ̂ ε ( S 1 ) | ̂ ε ( S 3 ) / ̂ ε ( S 1 ) |
| 2200           | 3 . 253 2 . 063           | 2 . 645 1 . 458           | 1 . 793 0 . 644           | 1 . 667 0 . 479           |
| 3200           | 1 . 257 0 . 842           | 1 . 251 0 . 632           | 1 . 343 0 . 234           | 1 . 283 0 . 197           |
| 4200           | 1 . 303 0 . 692           | 1 . 168 0 . 739           | 1 . 244 0 . 152           | 1 . 246 0 . 148           |
| Coreset Size   | Bank                      | Bank                      | Adult                     | Adult                     |
| Coreset Size   | ̂ ε ( S 2 ) / ̂ ε ( S 1 ) | ̂ ε ( S 3 ) / ̂ ε ( S 1 ) | ̂ ε ( S 2 ) / ̂ ε ( S 1 ) | ̂ ε ( S 3 ) / ̂ ε ( S 1 ) |
| 1200           | 1 . 647 0 . 972           | 1 . 360 1 . 018           | 1 . 467 0 . 287           | 1 . 094 0 . 542           |
| 1700           | 1 . 010 0 . 654           | 1 . 028 0 . 574           | 2 . 149 0 . 884           | 2 . 416 1 . 002           |
| 2200           | 1 . 010 0 . 654           | 1 . 026 0 . 674           | 1 . 089 0 . 360           | 1 . 172 0 . 537           |
| Coreset Size   | Athlete                   | Athlete                   | Diabetes                  | Diabetes                  |
| Coreset Size   | ̂ ε ( S 2 ) / ̂ ε ( S 1 ) | ̂ ε ( S 3 ) / ̂ ε ( S 1 ) | ̂ ε ( S 2 ) / ̂ ε ( S 1 ) | ̂ ε ( S 3 ) / ̂ ε ( S 1 ) |
| 210            | 5 . 172 3 . 634           | 4 . 200 1 . 944           | 5 . 700 3 . 303           | 5 . 868 2 . 952           |
| 310            | 2 . 467 1 . 564           | 1 . 427 0 . 660           | 1 . 567 0 . 800           | 1 . 332 0 . 653           |
| 410            | 1 . 658 0 . 881           | 1 . 045 0 . 449           | 1 . 360 1 . 103           | 1 . 216 0 . 943           |

This work opens several intriguing research directions. One immediate problem is to optimize the coreset size for d &gt; 1 or k &gt; 1 , particularly in cases where the size diverges from that of the vanilla setting. Extending robust coresets to the streaming model is a valuable but challenging direction. The primary obstacle is their lack of mergeability, as outlier interactions across different data chunks prevent the compositional updates essential for streaming algorithms. It is also interesting to explore whether our non-component-wise analysis can be applied to other robust machine learning problems, such as robust regression and robust PCA.

Table 2: Comparison of runtime between our Algorithm 2 and baseline HLLW25 . For each dataset, the coreset size of baseline HLLW25 is 2 m and the coreset size of ours is m . We use Lloyd algorithm given by [7] to compute approximate solutions c P and c S for both the original dataset P and coreset S , respectively. 'COST P ' denotes cost ( m ) ( P, c P ) on the original dataset P . 'COST S ' denotes cost ( m ) ( P, c S ) on the coreset constructed by METHOD. T X is the running time on the original dataset. T S is the running time on coreset. T C is the construction time of the coreset.

| DATASET    | COST P       | METHOD      | COST S                    |    T X | T C         | T S         |
|------------|--------------|-------------|---------------------------|--------|-------------|-------------|
| CENSUS1990 | 5.099 × 10 6 | OURS HLLW25 | 5.100 × 10 6 5.099 × 10 6 | 63.425 | 6.876 6.950 | 1.284 2.629 |
| TWITTER    | 7.307 × 10 6 | OURS HLLW25 | 7.310 × 10 6 7.307 × 10 6 | 41.816 | 3.233 3.259 | 0.633 1.278 |
| BANK       | 7.815 × 10 6 | OURS HLLW25 | 7.760 × 10 6 7.765 × 10 6 | 16.555 | 1.427 1.477 | 0.308 0.677 |
| ADULT      | 3.418 × 10 9 | OURS HLLW25 | 3.412 × 10 9 3.411 × 10 9 | 16.907 | 1.632 1.596 | 0.310 0.597 |
| ATHLETE    | 1.460 × 10 5 | OURS HLLW25 | 1.467 × 10 5 1.463 × 10 5 |  2.92  | 0.320 0.321 | 0.055 0.104 |
| DIABETES   | 1.781 × 10 5 | OURS HLLW25 | 1.786 × 10 5 1.788 × 10 5 |  3.977 | 0.366 0.360 | 0.062 0.135 |

## Acknowledgment

LH acknowledges support from the New Cornerstone Science Laboratory, and NSFC Grant No. 625707396.

## References

- [1] Peyman Afshani and Chris Schwiegelshohn. Optimal coresets for low-dimensional geometric median. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024 . OpenReview.net, 2024.
- [2] Akanksha Agrawal, Tanmay Inamdar, Saket Saurabh, and Jie Xue. Clustering what matters: Optimal approximation for clustering with outliers. Journal of Artificial Intelligence Research , 78:143-166, 2023.
- [3] Patrice Assouad. Plongements lipschitziens dans {{ r }} n . Bulletin de la Société Mathématique de France , 111:429-448, 1983.
- [4] Daniel N. Baker, Vladimir Braverman, Lingxiao Huang, Shaofeng H.-C. Jiang, Robert Krauthgamer, and Xuan Wu. Coresets for clustering in graphs of bounded treewidth. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event , volume 119 of Proceedings of Machine Learning Research , pages 569-579. PMLR, 2020.
- [5] Sayan Bandyapadhyay, Fedor V Fomin, and Kirill Simonov. On coresets for fair clustering in metric and Euclidean spaces and their applications. Journal of Computer and System Sciences , page 103506, 2024.
- [6] Barry Becker and Ronny Kohavi. Adult. UCI Machine Learning Repository, 1996. DOI: https://doi.org/10.24432/C5XW20.
- [7] Aditya Bhaskara, Sharvaree Vadgama, and Hong Xu. Greedy sampling for approximate clustering in the presence of outliers. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d'Alché-Buc, Emily B. Fox, and Roman Garnett, editors, Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada , pages 11146-11155, 2019.
- [8] Nicolas Bousquet and Stéphan Thomassé. VC-dimension and erd˝ os-pósa property. Discret. Math. , 338(12):2302-2317, 2015.
- [9] Vladimir Braverman, Vincent Cohen-Addad, Shaofeng H.-C. Jiang, Robert Krauthgamer, Chris Schwiegelshohn, Mads Bech Toftrup, and Xuan Wu. The power of uniform sampling for coresets. In 63rd IEEE Annual Symposium on Foundations of Computer Science, FOCS 2022, Denver, CO, USA, October 31 - November 3, 2022 , pages 462-473. IEEE, 2022.
- [10] Guilherme Oliveira Campos, Arthur Zimek, Jörg Sander, Ricardo J. G. B. Campello, Barbora Micenková, Erich Schubert, Ira Assent, and Michael E. Houle. On the evaluation of unsupervised outlier detection: measures, datasets, and an empirical study. Data Min. Knowl. Discov. , 30(4):891-927, 2016.
- [11] T-H. Hubert Chan, Arnaud Guerquin, and Mauro Sozio. Twitter data set. https://github. com/fe6Bc5R4JvLkFkSeExHM/k-center , 2018.
- [12] T.-H. Hubert Chan, Silvio Lattanzi, Mauro Sozio, and Bo Wang. Fully dynamic k -center clustering with outliers. Algorithmica , 86(1):171-193, 2024.
- [13] Moses Charikar, Samir Khuller, David M. Mount, and Giri Narasimhan. Algorithms for facility location problems with outliers. In S. Rao Kosaraju, editor, Proceedings of the Twelfth Annual Symposium on Discrete Algorithms, January 7-9, 2001, Washington, DC, USA , pages 642-651. ACM/SIAM, 2001.

- [14] Sanjay Chawla and Aristides Gionis. k -means-: A unified approach to clustering and outlier detection. In Proceedings of the 13th SIAM International Conference on Data Mining, May 2-4, 2013. Austin, Texas, USA , pages 189-197. SIAM, 2013.
- [15] Ke Chen. A constant factor approximation algorithm for k -median clustering with outliers. In Shang-Hua Teng, editor, Proceedings of the Nineteenth Annual ACM-SIAM Symposium on Discrete Algorithms, SODA 2008, San Francisco, California, USA, January 20-22, 2008 , pages 826-835. SIAM, 2008.
- [16] Ke Chen. On coresets for k -median and k -means clustering in metric and Euclidean spaces and their applications. SIAM J. Comput. , 39(3):923-947, 2009.
- [17] Flavio Chierichetti, Ravi Kumar, Silvio Lattanzi, and Sergei Vassilvitskii. Fair clustering through fairlets. pages 5029-5037, 2017.
- [18] Vincent Cohen-Addad, Kasper Green Larsen, David Saulpic, and Chris Schwiegelshohn. Towards optimal lower bounds for k -median and k -means coresets. In Stefano Leonardi and Anupam Gupta, editors, STOC '22: 54th Annual ACM SIGACT Symposium on Theory of Computing, Rome, Italy, June 20 - 24, 2022 , pages 1038-1051. ACM, 2022.
- [19] Vincent Cohen-Addad, Kasper Green Larsen, David Saulpic, Chris Schwiegelshohn, and Omar Ali Sheikh-Omar. Improved coresets for Euclidean k -means. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022 , 2022.
- [20] Vincent Cohen-Addad and Jason Li. On the fixed-parameter tractability of capacitated clustering. In ICALP , volume 132 of LIPIcs , pages 41:1-41:14. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2019.
- [21] Vincent Cohen-Addad, David Saulpic, and Chris Schwiegelshohn. Improved coresets and sublinear algorithms for power means in Euclidean spaces. In Marc'Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual , pages 21085-21098, 2021.
- [22] Jacobus Conradi, Benedikt Kolbe, Ioannis Psarros, and Dennis Rohde. Fast approximations and coresets for (k, l)-median under dynamic time warping. volume abs/2312.09838, 2023.
- [23] Rajni Dabas, Neelima Gupta, and Tanmay Inamdar. FPT approximation for capacitated clustering with outliers. Theoretical Computer Science , 1027:115026, 2025.
- [24] Hu Ding and Zixiu Wang. Layered sampling for robust optimization problems. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event , volume 119 of Proceedings of Machine Learning Research , pages 2556-2566. PMLR, 2020.
- [25] Olga Dorabiala, J. Nathan Kutz, and Aleksandr Y. Aravkin. Robust trimmed k -means. Pattern Recognit. Lett. , 161:9-16, 2022.
- [26] Dan Feldman and Michael Langberg. A unified framework for approximating and clustering data. In Lance Fortnow and Salil P. Vadhan, editors, Proceedings of the 43rd ACM Symposium on Theory of Computing, STOC 2011, San Jose, CA, USA, 6-8 June 2011 , pages 569-578. ACM, 2011.
- [27] Dan Feldman and Leonard J. Schulman. Data reduction for weighted and outlier-resistant clustering. In Yuval Rabani, editor, Proceedings of the Twenty-Third Annual ACM-SIAM Symposium on Discrete Algorithms, SODA 2012, Kyoto, Japan, January 17-19, 2012 , pages 1343-1354. SIAM, 2012.
- [28] Gereon Frahling and Christian Sohler. Coresets in dynamic geometric data streams. In Harold N. Gabow and Ronald Fagin, editors, Proceedings of the 37th Annual ACM Symposium on Theory of Computing, Baltimore, MD, USA, May 22-24, 2005 , pages 209-217. ACM, 2005.

- [29] Zachary Friggstad, Kamyar Khodamoradi, Mohsen Rezapour, and Mohammad R. Salavatipour. Approximation schemes for clustering with outliers. ACM Trans. Algorithms , 15(2):26:1-26:26, 2019.
- [30] Jing Gao, Feng Liang, Wei Fan, Chi Wang, Yizhou Sun, and Jiawei Han. On community outliers and their efficient detection in information networks. In Bharat Rao, Balaji Krishnapuram, Andrew Tomkins, and Qiang Yang, editors, Proceedings of the 16th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Washington, DC, USA, July 25-28, 2010 , pages 813-822. ACM, 2010.
- [31] Anupam Gupta, Robert Krauthgamer, and James R Lee. Bounded geometries, fractals, and low-distortion embeddings. In 44th Annual IEEE Symposium on Foundations of Computer Science, 2003. Proceedings. , pages 534-543. IEEE, 2003.
- [32] Shalmoli Gupta, Ravi Kumar, Kefu Lu, Benjamin Moseley, and Sergei Vassilvitskii. Local search methods for k -means with outliers. Proc. VLDB Endow. , 10(7):757-768, 2017.
- [33] Mohammad Taghi Hajiaghayi, Wei Hu, Jian Li, Shi Li, and Barna Saha. A constant factor approximation algorithm for fault-tolerant k -median. ACMTrans. Algorithms , 12(3):36:1-36:19, 2016.
- [34] Sariel Har-Peled and Akash Kushal. Smaller coresets for k -median and k -means clustering. In Joseph S. B. Mitchell and Günter Rote, editors, Proceedings of the 21st ACM Symposium on Computational Geometry, Pisa, Italy, June 6-8, 2005 , pages 126-134. ACM, 2005.
- [35] Sariel Har-Peled and Soham Mazumdar. On coresets for k -means and k -median clustering. In László Babai, editor, Proceedings of the 36th Annual ACM Symposium on Theory of Computing, Chicago, IL, USA, June 13-16, 2004 , pages 291-300. ACM, 2004.
- [36] Heesoo. 120 years of olympic history: Athletes and results. https://www.kaggle.com/ datasets/heesoo37/120-years-of-olympic-history-athletes-and-results , 2016. Data originally sourced from sports-reference.com.
- [37] Lingxiao Huang, Ruiyuan Huang, Zengfeng Huang, and Xuan Wu. On coresets for clustering in small dimensional Euclidean spaces. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA , volume 202 of Proceedings of Machine Learning Research , pages 13891-13915. PMLR, 2023.
- [38] Lingxiao Huang, Shaofeng H.-C. Jiang, Jian Li, and Xuan Wu. Epsilon-coresets for clustering (with outliers) in doubling metrics. In Mikkel Thorup, editor, 59th IEEE Annual Symposium on Foundations of Computer Science, FOCS 2018, Paris, France, October 7-9, 2018 , pages 814-825. IEEE Computer Society, 2018.
- [39] Lingxiao Huang, Shaofeng H.-C. Jiang, Jianing Lou, and Xuan Wu. Near-optimal coresets for robust clustering. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023 . OpenReview.net, 2023.
- [40] Lingxiao Huang, Jian Li, Pinyan Lu, and Xuan Wu. Coresets for constrained clustering: General assignment constraints and improved size bounds. In Proceedings of the 2025 ACM-SIAM Symposium on Discrete Algorithms (SODA) . SIAM, 2025.
- [41] Lingxiao Huang, Jian Li, and Xuan Wu. On optimal coreset construction for Euclidean (k, z)-clustering. In Bojan Mohar, Igor Shinkar, and Ryan O'Donnell, editors, Proceedings of the 56th Annual ACM Symposium on Theory of Computing, STOC 2024, Vancouver, BC, Canada, June 24-28, 2024 , pages 1594-1604. ACM, 2024.
- [42] Shaofeng H-C Jiang and Jianing Lou. Coresets for robust clustering via black-box reductions to vanilla case. In ICALP , 2025.
- [43] Michael Kahn. Diabetes. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5T59G.

- [44] Samir Khuller, Robert Pless, and Yoram J. Sussmann. Fault tolerant k -center problems. Theor. Comput. Sci. , 242(1-2):237-245, 2000.
- [45] Yi Li, Philip M. Long, and Aravind Srinivasan. Improved bounds on the sample complexity of learning. J. Comput. Syst. Sci. , 62(3):516-527, 2001.
- [46] Chris Meek, Bo Thiesson, and David Heckerman. Us census data (1990). UCI Machine Learning Repository, 2001. DOI: https://doi.org/10.24432/C5VP42.
- [47] S. Moro, P. Rita, and P. Cortez. Bank marketing. UCI Machine Learning Repository, 2012. DOI: https://doi.org/10.24432/C5K306.
- [48] Shyam Narayanan and Jelani Nelson. Optimal terminal dimensionality reduction in Euclidean space. In Moses Charikar and Edith Cohen, editors, Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing, STOC 2019, Phoenix, AZ, USA, June 23-26, 2019 , pages 1064-1069. ACM, 2019.
- [49] Debolina Paul, Saptarshi Chakraborty, Swagatam Das, and Jason Q. Xu. Uniform concentration bounds toward a unified framework for robust clustering. In Marc'Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual , pages 8307-8319, 2021.
- [50] Vladimir Shenmaier. A structural theorem for center-based clustering in high-dimensional Euclidean space. In Giuseppe Nicosia, Panos M. Pardalos, Renato Umeton, Giovanni Giuffrida, and Vincenzo Sciacca, editors, Machine Learning, Optimization, and Data Science - 5th International Conference, LOD 2019, Siena, Italy, September 10-13, 2019, Proceedings , volume 11943 of Lecture Notes in Computer Science , pages 284-295. Springer, 2019.
- [51] Vladimir Shenmaier. Some estimates on the discretization of geometric center-based problems in high dimensions. In Yury Kochetov, Igor Bykadorov, and Tatiana Gruzdeva, editors, Mathematical Optimization Theory and Operations Research , pages 88-101, Cham, 2020. Springer International Publishing.
- [52] Vladimir Shenmaier. Approximation and complexity of the capacitated geometric median problem. In Rahul Santhanam and Daniil Musatov, editors, Computer Science - Theory and Applications - 16th International Computer Science Symposium in Russia, CSR 2021, Sochi, Russia, June 28 - July 2, 2021, Proceedings , volume 12730 of Lecture Notes in Computer Science , pages 422-434. Springer, 2021.
- [53] Adiel Statman, Liat Rozenberg, and Dan Feldman. k-means: Outliers-resistant clustering+++. Algorithms , 13(12):311, 2020.

## Contents

| Introduction   | Introduction                                                                                                                                                     | Introduction                                                                                                                                                     | Introduction                                                                                                                                                     |   1 |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
|                | 1.1 Our contributions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                      | 1.1 Our contributions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                      | 1.1 Our contributions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                      |   2 |
|                | 1.2 Technical overview . . . . . . . . . . . . . . . . . . . .                                                                                                   | 1.2 Technical overview . . . . . . . . . . . . . . . . . . . .                                                                                                   | 1.2 Technical overview . . . . . . . . . . . . . . . . . . . .                                                                                                   |   4 |
|                |                                                                                                                                                                  | 1.2.1                                                                                                                                                            | Overview for Theorem 1.2 ( d = 1 ) . . .                                                                                                                         |   4 |
|                |                                                                                                                                                                  | 1.2.2                                                                                                                                                            | Overview for Theorem 1.3 (general d ) .                                                                                                                          |   5 |
| 2              | Optimal coreset size when d = 1                                                                                                                                  | Optimal coreset size when d = 1                                                                                                                                  | Optimal coreset size when d = 1                                                                                                                                  |   6 |
| 3              | Improved coreset sizes for general d ≥ 1                                                                                                                         | Improved coreset sizes for general d ≥ 1                                                                                                                         | Improved coreset sizes for general d ≥ 1                                                                                                                         |   7 |
| 4              |                                                                                                                                                                  | Empirical                                                                                                                                                        | results                                                                                                                                                          |   8 |
| 5              |                                                                                                                                                                  | Conclusion                                                                                                                                                       | and future work                                                                                                                                                  |   9 |
| A              | Omitted                                                                                                                                                          | details                                                                                                                                                          | in Section 1                                                                                                                                                     |  17 |
|                | A.1 Formal definition of coreset . . . . . . . . . . . . . . . . . . . . . . . .                                                                                 | A.1 Formal definition of coreset . . . . . . . . . . . . . . . . . . . . . . . .                                                                                 | A.1 Formal definition of coreset . . . . . . . . . . . . . . . . . . . . . . . .                                                                                 |  17 |
|                | A.2                                                                                                                                                              | Other related work . . . . . . . . . . .                                                                                                                         | . . . .                                                                                                                                                          |  17 |
|                | A.3                                                                                                                                                              | An illustrative example for the obstacle of Inequality (3) . . . . . . . . .                                                                                     | An illustrative example for the obstacle of Inequality (3) . . . . . . . . .                                                                                     |  18 |
| B              | Omitted details in Section 2                                                                                                                                     | Omitted details in Section 2                                                                                                                                     | Omitted details in Section 2                                                                                                                                     |  19 |
|                | B.1                                                                                                                                                              | Property of buckets . . . . . . . . . . . . . . . . . . . . .                                                                                                    | Property of buckets . . . . . . . . . . . . . . . . . . . . .                                                                                                    |  19 |
|                | B.2                                                                                                                                                              | Proof of Lemma 2.3 . . .                                                                                                                                         | . . . . . . . . . . . .                                                                                                                                          |  19 |
|                | B.3                                                                                                                                                              | Complete definition of blocks . . .                                                                                                                              | . . . . . . .                                                                                                                                                    |  20 |
| C              | B.4 Justifying the selection of the optimal center . . . . . . . . . . . . . . . . . . . . Proof of Theorem 1.1: coreset lower bound for robust geometric median | B.4 Justifying the selection of the optimal center . . . . . . . . . . . . . . . . . . . . Proof of Theorem 1.1: coreset lower bound for robust geometric median | B.4 Justifying the selection of the optimal center . . . . . . . . . . . . . . . . . . . . Proof of Theorem 1.1: coreset lower bound for robust geometric median |  20 |
| D              | Proof of Theorem 1.2: optimal coreset size for robust 1D geometric median                                                                                        | Proof of Theorem 1.2: optimal coreset size for robust 1D geometric median                                                                                        | Proof of Theorem 1.2: optimal coreset size for robust 1D geometric median                                                                                        |  21 |
|                | D.1                                                                                                                                                              | Proof of the upper bound in Theorem 1.2 . . . . . . . . . . . .                                                                                                  | Proof of the upper bound in Theorem 1.2 . . . . . . . . . . . .                                                                                                  |  21 |
|                | D.2                                                                                                                                                              | Proof of Lemma 2.4: Number of buckets                                                                                                                            | . . . .                                                                                                                                                          |  22 |
|                | D.3                                                                                                                                                              | Proof of Lemma D.1: Error analysis for c ∈ P M . . .                                                                                                             | Proof of Lemma D.1: Error analysis for c ∈ P M . . .                                                                                                             |  22 |
|                | D.4                                                                                                                                                              | Proof of Lemma D.2: Number of misaligned outliers . . . .                                                                                                        | Proof of Lemma D.2: Number of misaligned outliers . . . .                                                                                                        |  25 |
|                | D.5                                                                                                                                                              | Proof of Lemma D.3: Error analysis for c / ∈ P M . . . . . . .                                                                                                   | Proof of Lemma D.3: Error analysis for c / ∈ P M . . . . . . .                                                                                                   |  26 |
|                | D.6                                                                                                                                                              | Proof of the lower bound in Theorem 1.2 . . .                                                                                                                    | .                                                                                                                                                                |  27 |
| E              | Proofs of Theorem 1.3 ( d ≥ 1) and extension to metric spaces                                                                                                    | Proofs of Theorem 1.3 ( d ≥ 1) and extension to metric spaces                                                                                                    | Proofs of Theorem 1.3 ( d ≥ 1) and extension to metric spaces                                                                                                    |  28 |
|                | E.1                                                                                                                                                              | Proof of Lemma 3.2: induced error of S O .                                                                                                                       | . .                                                                                                                                                              |  28 |
|                | E.2                                                                                                                                                              | Extension to other metric spaces .                                                                                                                               | . . . . . . .                                                                                                                                                    |  35 |
| F              | Proof of Theorem 1.5: robust ( k, z ) -clustering                                                                                                                | Proof of Theorem 1.5: robust ( k, z ) -clustering                                                                                                                | Proof of Theorem 1.5: robust ( k, z ) -clustering                                                                                                                |  36 |
|                | F.1                                                                                                                                                              | Result for robust ( k, z ) -clustering . . . . . . . . . . . . . .                                                                                               | Result for robust ( k, z ) -clustering . . . . . . . . . . . . . .                                                                                               |  38 |
|                | F.2                                                                                                                                                              | Proof of Lemma F.7: Induced error                                                                                                                                | of S . . .                                                                                                                                                       |  38 |

O

| F.3   | Extension to other metric spaces . . . . . . . . . . . . .   |   40 |
|-------|--------------------------------------------------------------|------|
| G     | Additional empirical results                                 |   41 |
| G.1   | Additional empirical results for robust geometric median     |   41 |
| G.2   | Empirical results for robust 1D geometric median . . . .     |   42 |
| G.3   | Empirical results for robust k -median . . . . . . . . . .   |   43 |
| G.4   | Empirical results for robust k -means . . . . . . . . . . .  |   45 |

Table 3: Comparison of the state-of-the-art coreset size and our results for robust geometric median and robust k -median in R d . Robust k -median is a generalization of robust geometric median; see Definition F.1 when z = 1 .

| PARAMETERS d , k   | PARAMETERS d , k   | PRIOR RESULTS                                                                                                   | OUR RESULTS (ASSUMING n ≥ 4 m )                                                       |
|--------------------|--------------------|-----------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| d = 1              | k = 1              | O ( m )+ ˜ O ( ε - 2 ) [40] ˜ O ( mε - 1 )+ VANILLA SIZE [ 42 ] Ω( m ) [39]                                     | ˜ O ( ε - 1 2 + m n ε - 1 ) (THEOREM 1.2) Ω( ε - 1 2 + m n ε - 1 ) (THEOREM 1.2)      |
| d > 1              | k = 1              | O ( m )+ ˜ O ( ε - 2 · min { ε - 2 ,d } ) [40] ˜ O ( mε - 1 )+ VANILLA SIZE [ 42 ] Ω( m ) [39]                  | ˜ O ( ε - 2 · min { ε - 2 ,d } ) (THEOREM 1.3) Ω( ε - 1 2 + m n ε - 1 ) (THEOREM 1.2) |
| d > 1              | k > 1              | O ( m )+ ˜ O ( k 2 ε - 2 · min { ε - 2 ,d } ) [40] ˜ O (min { kmε - 1 ,mε - 2 } )+ VANILLA SIZE[42] Ω( m ) [39] | ˜ O ( k 2 ε - 2 · min { ε - 2 ,d } ) (THEOREM 1.5, UNDER ASSUMPTION 1.4)              |

## A Omitted details in Section 1

## A.1 Formal definition of coreset

In this section, we define the coreset for robust geometric median. For preparation, we first generalize the cost function cost ( m ) to handle weighted datasets.

Definition A.1 (Generalized cost function) . Let m be an integer. Let S ⊆ R d be a weighted dataset with weights w ( p ) for each point p ∈ S . Let w ( S ) := ∑ p ∈ S w ( p ) . Define a collection of weight functions W := { w ′ : S → R + | ∑ p ∈ S w ′ ( p ) = w ( S ) -m ∧ ∀ p ∈ S, w ′ ( p ) ≤ w ( p ) } . Moreover, we define the following cost function on S : ∀ c ∈ R d , cost ( m ) ( S, c ) := min w ′ ∈W ∑ p ∈ S w ′ ( p ) · dist( p, c ) .

Intuitively, to compute cost ( m ) ( S, c ) , we find a weighted subset S ′ of S with total weight w ( S ) -m that minimizes its vanilla cost to c . Thus, this cost ( m ) ( S, c ) serves as the cost function for robust geometric median on a weighted set. Note that for the unweighted case where w ( p ) = 1 for all p ∈ S , this cost function reduces to that in Equation (1).

Next, we define the notion of coreset for robust geometric median.

Definition A.2 (Coreset for robust geometric median) . Given a point set P ⊂ R d of size n ≥ 1 , integer m ≥ 1 and ε ∈ (0 , 1) , we say a weighted subset S ⊆ P together with a weight function w : S → R + is an ε -coreset of P for robust geometric median if w ( S ) = n and for any center c ∈ R d , cost ( m ) ( S, c ) ∈ (1 ± ε ) · cost ( m ) ( P, c ) .

This formulation ensures that the weighted coreset S provides an accurate approximation of the original dataset P 's cost for all centers c , within a tolerance specified by ε .

## A.2 Other related work

Coreset for robust clustering. Anatural extension of robust geometric median is called robust ( k, z ) -Clustering, attracting considerable interest for its coreset construction techniques in the literature [27, 38, 39, 41, 53]. In early work, [27] proposed a coreset construction method for the robust k -median problem, which requires an exponentially large size ( k + m ) O ( k + m ) ( ε -1 d log n ) 2 . Recently, [39] improved the coreset size to O ( m ) + ˜ O ( k 3 ε -3 z -2 ) via a hierarchical sampling framework proposed by [9]. Following this, [40] further improved the size to O ( m ) + ˜ O ( k 2 ε -2 z -2 ) . More recently, [42] proposed a new coreset of size O (min { kmε -1 , mε -2 z } ) + Vanilla size. We give Table 3 to compare our theoretical results with prior work.

Coreset for other clustering problems. Coreset construction for other variants of ( k, z ) -Clustering problems has also been extensively studied, including vanilla clustering [35, 16, 26, 9, 18, 19, 37, 41], capitalized clustering [9, 20], fair clustering [17, 5] and fault-tolerant clustering [44, 33]. Specifically, for vanilla ( k, z ) -clustering recent advancements by [19, 18, 41], produced a coreset

of size ˜ O (min { kε -z -2 , k 2 z +2 z +2 ε -2 } ) . When ε ≥ k -1 z +2 , the coreset upper bound kε -z -2 is shown to be optimal by a recent breakthrough [41]. Furthermore, a recent study [37] has investigated the coreset bounds when d is small.

## A.3 An illustrative example for the obstacle of Inequality (3)

Recall that we partition the input dataset P ⊂ R into disjoint buckets { B 1 , . . . , B T } , constructs a representative point µ ( B i ) with weight | B i | for each bucket and takes their union as a coreset S of P for robust 1D geometric median. Also, recall that prior work [39, 40] use Inequality (3) for error analysis, i.e., for any center c ∈ R and any tuple of outlier numbers ( m 1 , . . . , m T ) with ∑ i ∈ [ T ] m i = m ,

<!-- formula-not-decoded -->

In this section, we provide an example in which this inequality only holds if | S | = Ω( m ) .

Construct the dataset P as follows: for i = 1 , . . . , n -m , p i = i n ; for n -m + 1 ≤ i ≤ n , p i = n 3( i -n + m ) . We show that the Inequality (7) only holds if each p j forms its own isolated bucket. Assume, for the sake of contradiction, that there exists a bucket B q = { p i , . . . , p j } such that | B q | &gt; 1 and j &gt; n -m .

Case 1: If i ≤ n -m , let c = 0 . Then we have that the inlier set with respect to c is P ( c ) I = { p 1 , . . . , p n -m } . Thus,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which leads to a contradiction with Inequality (7).

<!-- formula-not-decoded -->

Note that | B q ∩ P ( c ) | = 1 , which means | B q | -m q = 1 , then we have

<!-- formula-not-decoded -->

which leads to a contradiction with Inequality (7). Overall, Inequality (7) does not hold if { p j } does not form a separated bucket.

Non-component-wise analysis breaks the obstacle. We then show how to adapt our new noncomponent-wise analysis to this example, which allows a more careful bucket decomposition. The key is that in our analysis, the induced error of the aforementioned bucket B q = { p i , . . . , p j } with j &gt; n -m is 0, due to allowing the misaligned outlier numbers in each bucket. Below illustrate the above two cases.

Case 1: If i ≤ n -m , c = 0 , then only the bucket B q may induce an error. Since j &gt; n -m , we have p j ≥ n 3 , thus

<!-- formula-not-decoded -->

Therefore, B q will be totally regarded as an outlier which induces 0 error, thus cost ( m ) ( S, c ) = cost ( m ) ( P, c ) .

Case 2: If i &gt; n -m , c = p i , then P ( c ) I = { p i -n + m +1 , . . . , p i } . Let the bucket B L = { p l , . . . , p r } with l ≤ i -n + m +1 , r ≥ i -n + m +1 denote the leftmost bucket containing at least one inlier. Therefore, only the buckets B L and B q may induce an error. For B q , we have dist( µ ( B q ) , c ) &gt; ( n 2 -1)dist( p 1 , c ) , thus B q induces 0 error. For B L , the induced error is bounded by εn · n 3( i -n -m ) ≤ ε · cost ( m ) ( P, c ) , since, in our framework, the size of each bucket in P -P M is restricted to at most O ( εn ) . In summary, the total error is bounded by ε · cost ( m ) ( P, c ) .

Thus, non-component-wise analysis significantly reduces bucket-wise errors in this example, which is crucial for proving S is a coreset.

## B Omitted details in Section 2

## B.1 Property of buckets

In Section 2, we introduce a useful notion called bucket. The following lemma shows that the coreset error on each bucket B is bounded by δ ( B ) . Recall that cost( B,c ) = ∑ p ∈ B dist( p, c ) for any point set B ⊂ R and center c ∈ R .

Lemma B.1 (Error analysis for buckets [34]) . Let B = { p l , . . . , p r } ⊆ P for 1 ≤ l ≤ r ≤ n be a bucket and c ∈ R be a center. We have

<!-- formula-not-decoded -->

## B.2 Proof of Lemma 2.3

Proof of Lemma 2.3. We prove the lemma by contradiction. Suppose there exists a center c ∈ P M satisfying that c &lt; c ⋆ and P ( c ) I = P I , and the leftmost point p l of P ( c ) I satisfies dist( p l , c L ) &gt; 2dist( c, c ⋆ ) . Then by the assumption of c , we have p l &lt; c L , thus p l / ∈ P ⋆ I . For any points p in P ⋆ I , we have

̸

<!-- formula-not-decoded -->

This implies that any point in P ⋆ I must be in P ( c ) I . However, since | P ⋆ I | = | P ( c ) I | = n -m , P ( c ) I cannot contain any other point not in P ⋆ I . This contradicts p l / ∈ P ⋆ I . Thus, dist( p l , c L ) ≤ 2dist( c, c ⋆ ) holds.

## B.3 Complete definition of blocks

Now we provide the complete definition of blocks, which are used in our algorithm 1. Recall that r max = max p ∈ P ⋆ I | p -c ⋆ | , c L = c ⋆ -r max and c R = c ⋆ + r max . Then we divide the sets P L and P R into disjoint blocks as follows:

Outer blocks ( B far ): Define B ( L ) far and B ( R ) far as the set of points that are far from c L and c R , where

<!-- formula-not-decoded -->

Inner blocks ( B i ): Define B ( L ) 0 , B ( LR ) 0 , B ( R ) 0 , and B ( RL ) 0 as the set of points that are close to c L or c R , where

<!-- formula-not-decoded -->

For the remaining points, partition them into blocks B ( L ) i , B ( LR ) i , B ( R ) i and B ( RL ) i based on exponentially increasing distance ranges for i = 1 , . . . , ⌈ log 2 ( ε -1 ) ⌉ , where

<!-- formula-not-decoded -->

## B.4 Justifying the selection of the optimal center

In previous coreset constructions [39], it is hard to obtain the exact value of optimal center c ⋆ , so an approximate center is generally used instead. However, when d = 1 , P ⋆ I is always a continuous subsequence of P . Suppose P ⋆ I = { p i , . . . , p j } , we have c ⋆ = p ⌊ j + i 2 ⌋ . This indicates that P ⋆ I and c ⋆ can be computed in polynomial time in robust 1D geometric median. Additionally, computing an O (1) -approximation of c ⋆ is also sufficient for our algorithm to remain valid, which only results in a factor of O (1) difference in the coreset size.

## C Proof of Theorem 1.1: coreset lower bound for robust geometric median

We first construct a bad instance P := { p 1 , . . . , p n } ⊂ R , where p i = i for i ∈ [ m ] and p j = x for m&lt;j ≤ n , x →∞ .

W.l.o.g, we assume n -m is an even number and n -m ≤ ( m -1) / 2 . Suppose ( S, w ) is an ε -coreset of size | S | &lt; m 2( n -m )+1 for robust geometric median on P . Define S I := S ∩ { p m +1 , . . . , p n } and S O := S ∩{ p 1 , . . . , p m } . There exists 2( n -m ) consecutive points p i +1 , ..., p i +2( n -m ) not in S for some i ∈ [ m -2( n -m )] . Let the center c = i +( n -m ) + 1 2 . Then we have

<!-- formula-not-decoded -->

We claim that w ( S O ) + w ( S I ) -m ≥ (1 -ε ) · ( n -m ) . Fix a center c ′ = x +1 . If w ( S O ) &gt; m , we have cost ( m ) ( P, c ′ ) = n -m and cost ( m ) ( S, c ′ ) →∞ , leading to an unbounded error. Therefore

we only need to consider the case w ( S O ) ≤ m and our claim can be verified by as follow:

<!-- formula-not-decoded -->

By this claim, we have

<!-- formula-not-decoded -->

In summary, we have cost ( m ) ( S, c ) ≥ (1 + ε ) · cost ( m ) ( P, c ) , which contradicts the definition of ε -coreset. We conclude that, each ε -coreset of P is of size Ω( m n -m ) .

## D Proof of Theorem 1.2: optimal coreset size for robust 1D geometric median

In this section, we provide the complete proof of Theorem 1.2 for both upper and lower bounds.

## D.1 Proof of the upper bound in Theorem 1.2

Now we show that our coreset S obtained by Algorithm 1 is an ε -coreset of P .

In the following discussion, we define S ( c ) I and S ⋆ I as follows. Given a weighted set S of total weight n and a center c ∈ R , let S ( c ) I := arg min S ( c ) I ⊆ S, ∑ s ∈ S ( c ) I w ( s )= n -m cost( S ( c ) I , c ) be the set of inliers of S , where w ( s ) represents the weight of s in S ( c ) I and is at most the weight of s in S . Moreover, let S ⋆ I represents S ( c ⋆ ) I , which exactly contains every µ ( B ) of each bucket B in P ⋆ I with weight | B | .

First, we consider the case that c ∈ P M . Note that regardless of the position of c , at most three buckets can induce the error: the bucket containing c , and the buckets containing the endpoints of P ( c ) I on either side. Actually, the coreset error in P M is already controlled by the vanilla coreset construction algorithm A . Thus, we only have to consider the cumulative error of the buckets that partially intersect with P ( c ) I on either side. We will show that this error is controlled by the carefully selected cumulative error bound of each inner block. Moreover, we adapt the analysis for the vanilla case in [37] to the robust case, as shown in Lemma D.1. This allows us to avoid considering the error caused by the misaligned outliers in P and S , which still suffice to ensure that the coreset error is bounded. The proof of Lemma D.1 is deferred to Section D.3.

<!-- formula-not-decoded -->

Now we analyze the case that c is not in P M . Assume P is divided into disjoint buckets B 1 , . . . , B q , from left to right. Fix a center c ∈ R , for each bucket B i ( i ∈ [ q ] ), define m i := | B i \ P ( c ) I | and m ′ i := | B i \ S ( c ) I | , which represent the number of outliers in B i with respect to c . Obviously we have ∑ i m i = ∑ i m ′ i = m . The following lemma shows that the number of inliers in each bucket for P ( c ) I and S ( c ) I remains roughly consistent, which indicates that cost ( m ) ( P, c ) and cost ( m ) ( S, c ) increase almost equally.

<!-- formula-not-decoded -->

Let Γ := sup c ∈ R ∑ i ∈ [ q ] | m i -m ′ i | , then we have Γ ≤ εn 4 . For c / ∈ P M , we consider the derivative of the cost value with respect to c , and gives an upper bound of the induced error in Lemma D.3. Combined with the upper bound of Γ , we conclude that the induced error is O ( εn · dist( c, c ⋆ )) , which is bounded by O ( ε ) · cost ( m ) ( P, c ) obviously. The main idea is similar to that of the proof of Theorem 2.1 in [37]. We defer the proof of Lemma D.2 and D.3 to Sec D.4 and D.5, respectively.

Lemma D.3 (Error analysis for c / ∈ P M ) . When the center c ≤ p m +1 or c ≥ p n -m , | cost ( m ) ( P, c ) -cost ( m ) ( S, c ) | ≤ (Γ + εn 8 ) · dist( c, P M ) ≤ ε · cost ( m ) ( P, c ) .

Now we are ready to prove the upper bound in Theorem 1.2.

Proof of Theorem 1.2 (upper bound). Given a dataset P of size n , we apply Algorithm 1 to P and obtain the output weighted set S . By Lemma 2.4, Algorithm 1 divides P into ˜ O ( ε -1 2 + m n ε -1 ) buckets. Note that S contains only the mean point µ ( B ) of each bucket B . Thus we have | S | = ˜ O ( ε -1 2 + m n ε -1 ) . Combined with Lemma D.1 and D.3, we conclude that S is an ε -coreset of P .

For the runtime, Line 3 cost O ( n ) time to obtain the optimal center. Recall that P ⋆ I = { p i , . . . , p j } is a continuous subsequence of P and c ⋆ = p ⌊ j + i ⌋ , since we only consider the one-dimensional case.

Thus, we can compute P ⋆ I and c ⋆ in O ( n ) time, since we can sequentially replace the leftmost point in the current inliers with the next point from the outliers, and the resulting cost difference can be computed in O (1) time. Lines 4-6 cost O ( n ) time, since we can sequentially check whether the next point can be added to the current bucket, otherwise, we place it into a new bucket. Obviously Lines 7-8 also cost O ( n ) time. This completes the proof.

2

## D.2 Proof of Lemma 2.4: Number of buckets

Proof of Lemma 2.4. Note that in Line 2, the number of buckets is ˜ O ( ε -1 2 ) in S M by Theorem 2.2. For Line 4, there are at most O (log( 1 ε )) non-empty blocks. For Line 6, the constraint on N ( B i,j ) generates at most O ( m n ε -1 ) buckets. For Lines 7-8, there are O (1) new one-point buckets.

What remains is to show that each non-empty block contains ˜ O ( ε -1 2 + m n ε -1 ) buckets in Line 5. Note that each block contains at most m points, thus the constraint on N ( B i,j ) generates at most O ( m n ε -1 ) buckets. Thus we only have to consider the constraint on δ ( B i,j ) .

Suppose we divide an inner block B i into t buckets { B i,j } 1 ≤ j ≤ t due to controlling δ ( B i,j ) . Since each B ij is the maximal bucket with δ ( B i,j ) ≤ 2 i · ε 2 nr max 288 , we have δ ( B i,j ∪ B i +1 ,j ) &gt; 2 i · ε 2 nr max 288 . Denote B i, 2 j -1 ∪ B i, 2 j by C j for j ∈ { 1 , . . . , ⌊ t 2 ⌋} . Let len ( B ) := max p ∈ B p -min p ∈ B p be the length of B . Note that δ ( B ) ≤ N ( B ) · len ( B ) holds for every bucket B . Thus we have:

<!-- formula-not-decoded -->

So we have ( ⌊ t 2 ⌋ ) 2 · 2 i · ε 2 nr max 288 &lt; m 2 i εr max , which implies t ≤ O ( ε -1 2 ) . The proof above is similar to Lemma 2.8 in [40]. Similarly, it is trivial to prove that B 0 also satisfies the above inequality. Since there are O (log( ε -1 )) non-empty blocks, the constraint on δ ( B i,j ) generates at most ˜ O ( ε -1 2 ) buckets. Thus Lemma 2.4 holds.

## D.3 Proof of Lemma D.1: Error analysis for c ∈ P M

Proof of Lemma D.1. Let L O := P L ∩ L ⋆ , R O := P R ∩ L ⋆ . Recall that L ⋆ denote the set of outliers w.r.t. the optimal center c ⋆ . W.L.O.G, assume c &gt; c ⋆ , thus P ( c ) I ∩ L O = ∅ . Next, we analyze the induced error in two cases, based on the scale of dist( c, c ⋆ ) . When εr max &gt; dist( c, c ⋆ ) , the center

c is sufficiently close to c ⋆ , so cost ( m ) ( P, c ) ≈ cost ( m ) ( P, c ⋆ ) , resulting in a small error. When εr max ≤ dist( c, c ⋆ ) , we have cost ( m ) ( P, c ) &gt; Ω( n · dist( c, c ⋆ )) &gt; Ω( εnr max ) , which matches the error from any outlier-misaligned bucket B , whose error is at most O ( εn )(dist( c, c ⋆ ) + εr max ) by Lemma 2.3.

Case 1: dist( c, c ⋆ ) &gt; ε 6 · r max .

If P ( c ) I ∩ R O = ∅ , then we have P ( c ) I = P ⋆ I , S ( c ) I = S ⋆ I . In this case, we directly have cost ( m ) ( S, c ) ∈ (1 ± ε )cost ( m ) ( P, c ) by Theorem 2.2.

̸

Next we assume P ( c ) I ∩ R O = ∅ . Same as the above lemma, denote the leftmost and rightmost buckets intersecting P ( c ) I as B L , B R , respectively. Recall that the coreset constructed by A ( P M , ε 3 ) is S M . By Theorem 2.2, Algorithm A ensures that

<!-- formula-not-decoded -->

Next, we bound the cumulative error of B L and B R . Define γ := max(0 , ⌈ log( dist( c,c ⋆ ) 2 ε · r max ) ⌉ ) . Obviously ε 6 r max ≤ dist( c, c ⋆ ) &lt; r max , thus γ ∈ [0 , ⌈ log( ε -1 ) ⌉ -1] . By the definition of γ , we have dist( c, c ⋆ ) ≤ 2 γ +1 εr max . Denote the rightmost point of B R as p r , then it follows that

<!-- formula-not-decoded -->

Recall that block B ( R ) i = { p ∈ P R | p &gt; c R , 2 i εr max ≤ dist( p, c R ) &lt; 2 i +1 εr max } . So the block B i which contains bucket B R satisfies i ≤ γ +1 . By Line 5 in Algorithm 1, any bucket B i,j in inner block B i satisfies δ ( B i,j ) ≤ 2 i · ε 2 nr max 288 . Thus,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that at least ⌊ n -m 2 ⌋ points are on the left of c ⋆ , among which there are at least ( ⌊ n -m 2 ⌋ -m ) inliers w.r.t center c . Moreover, when γ = 0 , we simply have dist( c, c ⋆ ) &gt; 2 γ εr max 6 by the assumption; when γ &gt; 0 , we have dist( c, c ⋆ ) &gt; 2 γ εr max by the definition of γ . Thus,

<!-- formula-not-decoded -->

Now we are ready to prove our goal | cost ( m ) ( P, c ) -∑ j ∈ [ q ] ( | B j | -m j ) · dist( µ ( B j ) , c ) | ≤ ε · cost ( m ) ( P, c ) . Recall that m j := | B j \ P ( c ) I | for each bucket B j . Thus for each bucket B j that is between B L and B R , we have m j = 0 ; for B L and B R , we have | B L | -m L = | B L ∩ P ( c ) I | and

Similarly,

<!-- formula-not-decoded -->

Thus by the definition of S ( c )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, it is easy to prove that the rightmost block B i ′ intersecting with S ( c ) I still satisfies i ′ ≤ γ +1 . Thus, similarly to the previous discussion, we have:

<!-- formula-not-decoded -->

Thus cost ( m ) ( S, c ) ∈ (1 ± ε )cost ( m ) ( P, c ) .

Case 2: dist( c, c ⋆ ) ≤ ε 6 · r max .

Let w l := ∑ p ∈ P ( c ) I \ P ⋆ I w ( p ) . Consider the points in P ( c ) I \ P ⋆ I , we have:

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

By the definition of cost ( m ) ( P, c ) , we have

<!-- formula-not-decoded -->

Similarly, we have

<!-- formula-not-decoded -->

By Theorem 2.2, Algorithm A ensures that | cost( S M , c ) -cost( P M , c ) | &lt; ε 3 · cost( P M , c ) . According to the definition of the block B 0 , we can obtain that there is no bucket that partially intersects with P ⋆ I . Thus,

<!-- formula-not-decoded -->

Combining the above equations, we have

<!-- formula-not-decoded -->

Similarly we have cost ( m ) ( P, c ) &lt; (1+ ε )cost ( m ) ( S, c ) , thus cost ( m ) ( S, c ) ∈ (1 ± ε ) · cost ( m ) ( P, c ) .

## D.4 Proof of Lemma D.2: Number of misaligned outliers

Proof of Lemma D.2. Note that for any center c , P ( c ) I is a continuous subset of size n -m . Let P ( c ) I = { p l , . . . , p l + n -m -1 } . Assume p l ∈ B L and p l + n -m -1 ∈ B R . Consider a point p j ∈ P L that satisfies dist( p j , c ) &lt; dist( p j + n -m , c ) and dist( µ ( B ) , c ) &gt; dist( µ ( B ′ ) , c ) , where p j ∈ B and p j + n -m ∈ B ′ . Next we show that the total number of points satisfying the above conditions is at most O ( εn ) , which also provides an upper bound for ∑ i | m i -m ′ i | .

̸

In this case p j ∈ P ( c ) I . By the definition of P ( c ) I , we have j ≥ l . If B = B L and B ′ = B R , then by j ≥ l , we have µ ( B ) &gt; max( B L ) and µ ( B ′ ) &gt; max( B R ) . Then we have

̸

<!-- formula-not-decoded -->

which contradicts the inequality dist( p j , c ) &lt; dist( p j + n -m , c ) . Thus, either B = B L or B ′ = B R holds. This indicates that the total number of points satisfying the above conditions is at most | B L | + | B R | . Formally, denote B ( i ) as the bucket containing p i , we have

<!-- formula-not-decoded -->

Symmetrically, consider the points in P R , we have

<!-- formula-not-decoded -->

Since | P ( c ) I | = n -m , for any i ∈ [1 , m ] , exactly one point in { p i , p i + n -m } is in P ( c ) I . This means that p i ∈ P ( c ) I if and only if p i + n -m / ∈ P ( c ) I . Thus, Inequality (21) is equivalent to the following form:

<!-- formula-not-decoded -->

Moreover, we'll show that I 1 and I 2 cannot both be greater than 0. Suppose I 1 &gt; 0 . In this case, there exists a point p j ∈ P L ∩ P ( c ) I such that dist( µ ( B ( j ) ) , c ) &gt; dist( µ ( B ( j + n -m ) ) , c ) . Consider a point p j ′ ∈ P L \ P ( c ) I , obviously j ′ &lt; j . Then we have

<!-- formula-not-decoded -->

Thus I 2 = 0 . Conversely, it still holds due to symmetry. Based on the above analysis, we have

<!-- formula-not-decoded -->

By p i ∈ P ( c ) I ⇐⇒ p i + n -m / ∈ P ( c ) I , it is easy to prove that ∑ i | m i -m ′ i | in P L and P R are exactly the same. Then it follows that

<!-- formula-not-decoded -->

## D.5 Proof of Lemma D.3: Error analysis for c / ∈ P M

Proof of Lemma D.3. W.L.O.G., we assume that c ≥ p n -m . Recall that P is divided into buckets B 1 , . . . , B q , from left to right. Suppose B t ( t ∈ [ q ] ) contains the center c , i.e., min p ∈ B t p ≤ c ≤ max p ∈ B t p . We define function f P ( c ) = cost ( m ) ( P, c ) and f S ( c ) = cost ( m ) ( S, c ) for every c ∈ R .

Note that f P ( c ) = f P ( p n -m ) + ∫ c p n -m f ′ P ( x ) and f S ( c ) = f S ( p n -m ) + ∫ c p n -m f ′ S ( x ) holds for any c &gt; p n -m . Next, we first show that f P ( p n -m ) = f S ( p n -m ) by Line 8 in Algorithm 1. Then we verify that | f ′ P ( c ) -f ′ S ( c ) | is bounded by ∑ i ∈ [ q ] | m i -m ′ i | , which suffices to prove the lemma. The main idea is similar to that of the proof of Theorem 2.1 in [37].

By Line 8 in Algorithm 1, there is no bucket that partially intersect with P ( p n -m ) I . In this case, for each bucket B ⊂ P ( p n -m ) I , dist( µ ( B ) , c ) ≤ max p ∈ P ( p n -m ) I dist( p, c ) ; for each bucket B ̸⊂ P ( p n -m ) I , dist( µ ( B ) , c ) &gt; max p ∈ P ( p n -m ) I dist( p, c ) . This indicates that S ( p n -m ) I contains µ ( B ) with weight | B | for each B ⊂ P ( p n -m ) I , which maintains consistent weights with P ( p n -m ) I . Then by Lemma B.1, we have f P ( p n -m ) = f S ( p n -m ) . Note that f p ( c ) is a linear function of c , and we have

<!-- formula-not-decoded -->

Similarly, when µ ( B t ) ≤ c , we have f ′ S ( c ) = ∑ i&lt;t ( | B i | -m ′ i ) + | B t | -∑ i&gt;t ( | B i | -m ′ i ) ; when µ ( B t ) &gt; c , we have f ′ S ( c ) = ∑ i&lt;t ( | B i | -m ′ i ) -| B t | -∑ i&gt;t ( | B i | -m ′ i ) . Thus

<!-- formula-not-decoded -->

Figure 3: A case for demonstrating the coreset lower bound for robust 1D geometric median. T i contains ⌊ m q ⌋ points where each point p ∈ T i satisfies p = m iα . T 0 contains the remaining points where each point p ∈ T 0 satisfies p = 0 .

<!-- image -->

<!-- formula-not-decoded -->

Moreover, since | P R | = m , at least n -2 m inliers w.r.t. c are on the left of p n -m . These points satisfy that dist( p, c ) ≥ dist( p n -m , c ) , which implies that cost ( m ) ( P, c ) ≥ ( n -2 m ) · dist( p n -m , c ) . Then we have

<!-- formula-not-decoded -->

This completes the proof.

## D.6 Proof of the lower bound in Theorem 1.2

Next we show that for n ≥ 4 m , the size lower bound of ε -coreset for robust 1D geometric median is Ω( ε -1 2 + m n ε -1 ) . This lower bound matches the upper bound in the above discussion, which completes the proof of Theorem 1.2. In the following discussion, we assume that the size of dataset P is sufficiently large such that εn &gt; 1 , which holds in nearly all practical scenarios.

Proof of Theorem 1.2 (lower bound). For vanilla 1D geometric median, [26] shows that the size lower bound of ε -coreset is Ω( ε -1 2 ) , which is obviously also the coreset size lower bound for robust 1D geometric median. Thus, it remains to show the coreset size is Ω( m n ε -1 ) when m n &gt; ε 1 2 .

Wefirst construct the dataset P of size n . Let q = ⌊ m 2 nε ⌋ . The dataset P is a union of 1+ q disjoint sets { T 0 , T 1 , . . . , T q } . For each i ∈ { 1 , . . . , q } , T i contains ⌊ m q ⌋ points, and every point p ∈ T i satisfies p = m iα , where α = 2+log m ( ε -2 ) . T 0 contains n -q ⌊ m q ⌋ points where each point p ∈ T 0 satisfies p = 0 . Correspondingly, define q disjoint intervals { I 1 , . . . , I q } , where I i = [ m ( i -1) α +1 , m iα +1 ] .

Suppose S is a ε -coreset of P with size | S | &lt; q . Then by the pigeonhole principle, there exists an interval I j ( j ∈ [1 , q ] ) such that S does not include any points located in I j . This implies that for any point p ∈ S , we have p ≤ m ( j -1) α +1 or p ≥ m jα +1 . Fix the center c = m jα , then we have

<!-- formula-not-decoded -->

Since n ≥ 4 m , we have | T 0 | = n -q · ⌊ m q ⌋ ≥ n -m . Consider the points in T 0 and T j as inliers, we have

<!-- formula-not-decoded -->

It follows that cost ( m ) ( S, c ) &gt; (1 + ε )cost ( m ) ( P, c ) , thus S is not a ε -coreset of P , which leads to a contradiction. This implies that any ε -coreset of P contains Ω( m n ε -1 ) points when m n &gt; ε 1 2 . Considering the discussion above, the lower bound of the coreset size is Ω( ε -1 2 + m n ε -1 ) .

Note that the dataset P we construct is a multiset, which is slightly different from the definition. However, this does not affect the proof, because we can move each point by a sufficiently small and distinct distance, making the cost value almost unchanged.

When εn &lt; 1 , we can show that the coreset size is Ω( ε -1 2 + m ) by the above discussion. Moreover, consider a trivial method that applies algorithm A on P M and keeps all points not in P M . It's easy to prove that this method constructs an ε -coreset of the original dataset under the assumption. In this case, the coreset size is ˜ O ( ε -1 2 + m ) , which matches the above lower bound.

## E Proofs of Theorem 1.3 ( d ≥ 1) and extension to metric spaces

In this section, we list out the missing proof in Section 3 and show how to extend Theorem 1.3 to various metric spaces.

## E.1 Proof of Lemma 3.2: induced error of S O

Below we briefly introduce the proof idea. We first observe that the induced error of S O is primarily caused by points that act as inliers in P but outliers in S , or vice versa, as shown in Lemmas E.3 and E.4. The error from a single point is bounded by O ( cost ( m ) ( P,c ) m ) (see Lemma E.5). The next task is ensuring that there are O ( εm ) such points in S O , which is guaranteed when S O provides an ε -approximation for the ball range space on L ⋆ (Lemma E.6). To achieve this, we sample ˜ O ( d/ε 2 ) points for S O (Lemma E.2).

Fix a center c ∈ R d . Let L ( c ) := arg min L ⊆ P : | L | = m cost( P -L, c ) be the set of outliers of P w.r.t. c and m P := | L ⋆ ∩ L ( c ) | represent the number of these outliers contained in L ⋆ . For S O ∪ P ⋆ I , we first define a family of weight functions W := { w S : S O ∪ P ⋆ I → R + | w S ( p ) ≤ w ( p ) , ∀ p ∈ S O ; w S ( p ) ≤ 1 , ∀ p ∈ P ⋆ I ; ∥ w S ∥ 1 = n -m } . Intuitively, W represents the collection of all possible

weight functions for n -m inliers of the weighted dataset S O ∪ P ⋆ I . Define a weight function w ′ as follows:

<!-- formula-not-decoded -->

i.e., ( S O ∪ P ⋆ I , w ′ ) represents the n -m inliers of S O ∪ P ⋆ I with respect to center c . Let m S := ∑ p ∈ S O ( w ( p ) -w ′ ( p )) denote the number of outliers of S O ∪ P ⋆ I w.r.t. c that are contained in S O .

For preparation, we introduce the concept of ball range space, which facilitates a precise analysis of point distributions in P and S .

Definition E.1 (Approximation of ball range space, Definition F.2 in [39]) . For a given dataset P ⊂ R d , the ball range space on P is ( P , P ) where P := { P ∩ Ball( c, u ) | c ∈ R d , u &gt; 0 } and Ball( c, u ) := { p ∈ R d | dist( p, c ) ≤ u } . A subset Y ⊂ P is called an ε -approximation of the ball range space ( P , P ) if for every c, u ∈ R d ,

<!-- formula-not-decoded -->

Based on this definition, we have the following preparation lemma that measures the performance of S O ; which is refined from [39].

Lemma E.2 (Refined from Lemma F.3 of [39]) . Given dataset P O ⊂ R d . Let S O be a uniform sampling of size ˜ O ( d ε 2 ) from P O , then with probability at least 1 -1 poly (1 /ε ) , S O is an ε -approximation of the ball range space on P O . Define a weight function w : w ( p ) = | P O | | S O | , for any p ∈ S O . Then for any c ∈ R d , u ∈ R + ,

<!-- formula-not-decoded -->

By the iterative method introduced by [48], the factor O ( d ) of coreset size can be replaced by ˜ O ( ε -2 ) . Therefore, S O is an ε -approximation of the ball range space on L ⋆ .

Then we are ready to prove Lemma 3.2. We first analyze where the induced error comes from.

Recall that we fix a center c in this section. Let T O := min p ∈ L ⋆ dist( p, c ) denote the minimum distance from points in L ⋆ to c . Let T I := max p ∈ P ⋆ I dist( p, c ) denote the maximum distance from points in P ⋆ I to c . We have the following Lemma.

Lemma E.3 (Comparing T O and T I ) . When m P = m , | cost ( m ) ( P, c ) -cost ( m ) ( S O ∪ P ⋆ I , c ) | = 0 holds. When m P &lt; m , T O ≤ T I holds.

Proof. If m P = m , for any point p ∈ L ⋆ , we have dist( p, c ) ≥ max p ∈ P ⋆ I dist( p, c ) . Since S O is sampled from L ⋆ , we know that for any point p ∈ S O , dist( p, c ) ≥ max p ∈ P ⋆ I dist( p, c ) . Therefore, we have m P = m S = m and then cost ( m ) ( P, c ) = cost( P ⋆ I , c ) = cost ( m ) ( S O ∪ P ⋆ I , c ) .

If m P &lt; m , then there exists a point ̂ p ∈ P ⋆ I such that dist( ̂ p, c ) ≥ T O . By definition, we also know that dist( ̂ p, c ) ≤ T I . Therefore, combining these two conditions, we have T O ≤ T I .

It remains to analyze the case that m P &lt; m . In this setting, we present the following lemma, which provides an upper bound on the estimation error | cost ( m ) ( P, c ) -cost ( m ) ( S O ∪ P ⋆ I , c ) | . Before stating the lemma, we introduce a notation that will also be used in subsequent lemmas. We sort all distances dist( p, c ) for each point p ∈ P ⋆ I in descending order, w.l.o.g. say dist( p 1 , c ) &gt; . . . &gt; dist( p | P ⋆ I | , c ) . Here, we can safely assume all distances dist( p, c ) are distinct, given that adding small values to the distances has only a subtle impact on the cost. Let d i := dist( p i , c ) for i ∈ [ m ] . Then d 1 , . . . , d m represent the distances from the m furthest points in P ⋆ I to the center c . Now, we are ready to provide the following lemmas.

Lemma E.4 (An upper bound of the induced error of S O ) . When m P &lt; m , suppose || P O ∩ Ball( c, u ) | -w ( S O ∩ Ball( c, u )) | ≤ ∆ for any u &gt; 0 , the following holds: | cost ( m ) ( P, c ) -cost ( m ) ( S O ∪ P ⋆ I , c ) | ≤ 2 · (∆ + | m P -m S | ) · ( T I -T O ) .

Proof. By Fact F.1 in [39], we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where ( a ) + = max { a, 0 } .

Let T := max p ∈ L ⋆ -L ( c ) dist( p, c ) and T S := max p ∈ S O : w ′ ( p ) &gt; 0 dist( p, c ) . Then we know that T O ≤ T , T S ≤ T I . By definition, for any distance u &gt; T , we have | L ⋆ ∩ Ball( c, u ) | ≥ m -m P . Then we can transform the Inequality (24) to

<!-- formula-not-decoded -->

Similarly, for any distance u &gt; T S , we have w ( S O ∩ Ball( c, u )) ≥ m -m S . We can transform the Inequality (25) to

<!-- formula-not-decoded -->

Based on the notation of d 1 , ..., d m , we know that each point p ∈ P ⋆ I satisfying dist( p, c ) ≤ d m -m P +1 is an inlier of P ⋆ I ∪ L ⋆ , and each point q ∈ P ⋆ I with weight w ′ ( q ) &gt; 0 satisfying dist( q, c ) ≤ d m -⌊ m S ⌋ is an inlier of S O ∪ P ⋆ I . Let m ′ S := ⌊ m S ⌋ . Let l := | m P -m S | denote the difference in the number of outliers. If m P &gt; m S , we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

If m P &lt; m S , we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

If m P = m S , we have

<!-- formula-not-decoded -->

By definition, we know that

̸

<!-- formula-not-decoded -->

When m S = m , we know the point p ∈ P ⋆ I that has distance dist( p, c ) = d m -m ′ S is an outlier on P ⋆ I ∪ S O . Then there exists a point q ∈ S O such that dist( q, c ) ≤ dist( p, c ) . Then we have

<!-- formula-not-decoded -->

Similarly, the point p ∈ P ⋆ I with distance dist( p, c ) = d m -m P is an outlier on P , thus there exists a point q ∈ L ⋆ such that dist( q, c ) ≤ dist( p, c ) . Then we have

<!-- formula-not-decoded -->

By Lemma E.3, it suffices to discuss the following three cases based on the values of m P and m S . Case 1: m S = m

Recall that l = m S -m P , we have

<!-- formula-not-decoded -->

Since m S = m and m P &lt; m S , each point p ∈ L ⋆ -L ( c ) satisfies dist( p, c ) &lt; min q ∈ S O dist( q, c ) . For each T O ≤ u ≤ T , we have w ( S O ∩ Ball( c, u )) = 0 . Since | L ⋆ ∩ Ball( c, u ) | -w ( S O ∩ Ball( c, u )) ≤ ∆ , we know | L ⋆ ∩ Ball( c, u ) | ≤ ∆ for T O ≤ u ≤ T and

<!-- formula-not-decoded -->

Since cost ( m S ) ( S O , c ) = 0 , we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Adding Inequality (30) and Inequality (36), we have

<!-- formula-not-decoded -->

Adding Inequality (31) into Inequality (37), we have

<!-- formula-not-decoded -->

Then we complete the proof of Case 2.

̸

<!-- formula-not-decoded -->

Without loss of generality, we assume that m P &gt; m S and T ≥ T S .

Firstly, we prove that cost ( m P ) ( P, c ) -cost ( m S ) ( S O ∪ P ⋆ I , c ) ≤ 2(∆ + | m P -m S | ) · ( T I -T O ) . We have

<!-- formula-not-decoded -->

For T S &lt; u &lt; T , we have | L ⋆ ∩ Ball( c, u ) | ≥ m -∆ , thus

<!-- formula-not-decoded -->

Adding Inequality (29) and the above inequality, we obtain

<!-- formula-not-decoded -->

Secondly, we prove that cost ( m S ) ( S O ∪ P ⋆ I , c ) -cost ( m P ) ( P, c ) ≤ (∆ + | m P -m S | ) · ( T I -T O ) . By Inequality (26) and Inequality (27), we have

<!-- formula-not-decoded -->

Adding Inequality (28) and the above inequality, we have

<!-- formula-not-decoded -->

In summary, we have

<!-- formula-not-decoded -->

Similarly, we can get the same conclusion when T &lt; T S . Moreover, for the case that m P &lt; m S , with the help of Inequalities (30)(31)(33)(35), the conclusion still holds.

̸

<!-- formula-not-decoded -->

By Inequality (32), we have

<!-- formula-not-decoded -->

By a similar argument as in Case 2, we have

<!-- formula-not-decoded -->

Overall, we complete the proof of Lemma E.4.

The Lemma E.4 tells us that at most 2(∆ + | m P -m S | ) points contribute to the error, with each point inducing at most T I -T O error. We now analyze the magnitude of T I -T O .

Lemma E.5 (Bounding induced error of each point) . T I -T O ≤ O (cost ( m ) ( P, c ) /m )

Proof. Let r max := max p ∈ P ⋆ I dist( p, c ⋆ ) denote the maximum distance from P ⋆ I to c ⋆ . Let d max := dist( c, c ⋆ ) denote the distance between the approximate center c ⋆ and the center c . Let ¯ r := cost( P ⋆ I ,c ⋆ ) n -m denote the average distance from P ⋆ I to c ⋆ .

Utilizing the triangle inequality, for any point p ∈ L ⋆ , we can assert that dist( p, c ) ≥ dist( p, c ⋆ ) -dist( c, c ⋆ ) ≥ r max -d max , thus T O ≥ r max -d max . Similarly, for any point p ∈ P ⋆ I , we have dist( p, c ) ≤ dist( p, c ⋆ ) + dist( c, c ⋆ ) = r max + d max , thus T I ≤ r max + d max . Consequently, depending on whether r max is greater than or less than d max , we derive different expressions for T I -T O : If r max ≥ d max , then T I -T O ≤ 2 · d max ; if r max &lt; d max , given that T O ≥ 0 , it follows that T I -T O ≤ r max + d max &lt; 2 · d max . Therefore, we turn to prove 2 · m · d max ≤ O (cost ( m ) ( P, c )) . We discuss the relationship between d max and ¯ r in two cases.

Case 1: d max ≤ 4¯ r By definition of ¯ r , we have

<!-- formula-not-decoded -->

since c ⋆ is an O (1) -approximate solution for robust geometric median, we have

<!-- formula-not-decoded -->

.

Case 2: d max &gt; 4¯ r Let ̂ P := { p ∈ P ⋆ I | dist( p, c ⋆ ) ≤ 2¯ r } . By definition, we have | ̂ P | ≥ 1 2 | P ⋆ I | = n -m 2 . Since d max &gt; 4¯ r , we obtain

<!-- formula-not-decoded -->

( m ) , which completes the proof.

Overall, we have T I -T O ≤ O (cost ( P, c ) /m )

Combining Lemmas E.4 and E.5, we obtain the bound: | cost ( m ) ( P, c ) -cost ( m ) ( S O ∪ P ⋆ I , c ) | ≤ O (1) · (∆ + | m P -m S | ) · cost ( m ) ( P,c ) m . To prove | cost ( m ) ( P, c ) -cost ( m ) ( S O ∪ P ⋆ I , c ) | ≤ O ( ε ) · cost ( m ) ( P, c ) , it remains to ensure that ∆ = O ( εm ) and | m P -m S | = O ( εm ) . We show that both conditions hold when S O is an ε -approximation for the ball range space on L ⋆ .

Lemma E.6 (Bounding misaligned outlier count) . Suppose S O is an ε -approximation for the ball range space on L ⋆ , we have | m P -m S | ≤ 2 · ε · m.

Before proving this lemma, we first analyze the properties of w ′ . Given a point p ∈ S O ∪ P ⋆ I , we say p is of partial weight w.r.t. c if 0 &lt; w ′ ( p ) &lt; w ( p ) when p ∈ S O and 0 &lt; w ′ ( p ) &lt; 1 when p ∈ P ⋆ I . Intuitively, such a point is partially an inlier and partially an outlier of S O ∪ P ⋆ I to c . The following claim indicates that the number of partial-weight points is at most one for any c .

Claim 1 (Properties of partial-weight points) . For every center c ⊂ R d , there exists at most one point v ∈ S O ∪ P ⋆ I of partial weight w.r.t. c . Moreover, for any other point p ∈ S O ∪ P ⋆ I with w ′ ( p ) = w ( p ) , we have dist( p, c ) ≤ dist( v, c ) .

Proof. By contradiction, we assume that there exist two points v, v ′ ∈ S O ∪ P ⋆ I of partial weight w.r.t. c . W.l.o.g., suppose dist( v, c ) ≤ dist( v ′ , c ) . We note that transferring weight from v to v ′ increases w ′ ( v ) by any constant δ &gt; 0 and decreases w ′ ( v ′ ) by δ does not increase the term cost ( m ) ( S O ∪ P ⋆ I , c ) . Then by selecting a suitable δ , we can make either v or v ′ no longer of partial weight. Hence, there exists at most one point v ∈ S O ∪ P ⋆ I of partial weight w.r.t. c .

For any other point p ∈ S O ∪ P ⋆ I with w ′ ( p ) = w ( p ) , suppose dist( v, c ) &lt; dist( p, c ) . In this case, increasing w ′ ( v ) by a small amount δ &gt; 0 and decreasing w ′ ( p ) by δ decreases cost ( m ) ( S O ∪ P ⋆ I , c ) . This contradicts the definition of w ′ , which completes the proof.

Recall that d 1 , . . . , d m represent the distances from the m furthest points in P ⋆ I to the center c . Now we are ready to prove Lemma E.6.

Proof of Lemma E.6. We only need to consider the case of m P &gt; m S and m P &lt; m S . Without loss of generality, we assume that m P &gt; m S . Let l 1 := m -m P +1 . Based on the definition of d 1 , ..., d m , each inlier point p satisfies dist( p, c ) ≤ d l 1 -1 . There are m -m P inlier points in L ⋆ , so we have:

<!-- formula-not-decoded -->

Let l 2 := m -⌊ m S ⌋ +1 . Since m P &gt; m S , we have m P ≥ ⌊ m S ⌋ +1 , then we get the inequality

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Claim 1, at most one point v ∈ S O ∪ P ⋆ I of partial weight w.r.t. c exists. If v ∈ P ⋆ I , we have dist( v, c ) = d l 2 . Any point v ′ ∈ P ⋆ I with dist( v ′ , c ) ≥ d l 2 -1 is an outlier of S O ∪ P ⋆ I . Then there are m -⌊ m S ⌋ -1 points in P ⋆ I that have distance to C at least d l 2 -1 . By contradiction, we assume that m -w ( S O ∩ Balls( c, d l 2 -1 )) &gt; ⌊ m S ⌋ +1 . Then there are no less than m points in P that have a distance to c greater than d l 2 -1 , which contradicts the fact that v ′ is an outlier of S O ∪ P ⋆ I . Hence, Inequality (40) holds. For the case that v ∈ S O or there is no point of partial weight w.r.t. c , the argument is similar.

Since m P &gt; m S , we have w ( S O ∩ Ball( c, d l 1 )) &gt; m -m P . Combining with Inequality (38), we conclude that

Moreover,

Then, we have

<!-- formula-not-decoded -->

Since S O is an ε -approximation for the ball range space on L ⋆ , by Lemma E.2, we have

<!-- formula-not-decoded -->

Combining the above two inequalities, we have

<!-- formula-not-decoded -->

Similarly, we can get the same conclusion when m P &lt; m S , which completes the proof of Lemma E.6.

Now we are ready to prove Lemma 3.2.

Proof of Lemma 3.2. Based on Lemmas E.3 and E.4, we have

<!-- formula-not-decoded -->

which completes the proof.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, we claim that

## E.2 Extension to other metric spaces

In this section, we explore the extension of Algorithm 2, designed for the robust geometric median in Euclidean space, to the metric spaces by leveraging the notions of VC dimension and doubling dimension since the VC dimension is known for various metric spaces [8, 9, 18, 40].

Let ( X , dist) denote the metric space, where X is the set under consideration, and dist : X × X → R ≥ 0 is a function that measures the distance between points in X , satisfying the triangle inequality. Specifically, in the Euclidean metric, X represents Euclidean space R d and dist is the Euclidean distance.

Similar to the analysis for robust geometric median on Euclidean space, we discuss the induced error of S I and S O separately. We first introduce how to bound the error induced by S O . Since the dist function satisfies the triangle inequality, our Lemmas E.3-E.4 hold, ensuring that each point in S O induces at most O (cost ( m ) ( P, c ) /m ) error. To ensure the number of points in S O inducing error is O ( εm ) , it suffices for S O to be an ε -approximation for the ball range space on L ⋆ (as shown in Lemma E.6). Next, we illustrate how this condition can be satisfied using the notions: 1) VC dimension; 2) doubling dimension.

VC dimension. We begin by introducing the concept of the VC dimension of the ball range space, which serves as a measure of the complexity of this ball range space.

Definition E.7 (VC dimension of ball range space) . Let M = ( X , dist) be the metric space and define Balls( X ) := { Ball( c, u ) | c ∈ X , u &gt; 0 } as the collection of balls in the space. The VC dimension of the ball range space ( X , Balls( X )) , denoted by d V C ( X ) , is the maximum | P | , P ⊆ X such that | P ∩ Balls( X ) | = 2 | P | , where P ∩ Balls( X ) := { P ∩ Ball( c, u ) | Ball( c, u ) ∈ Balls( X ) } .

The VC dimension of ( X , Balls( X )) aligns to the pseudo-dimension of ( X , Balls( X )) used by [45]. Based on this observation, we give out a refined lemma from [45].

Lemma E.8 (Refined from [45]) . Let M = ( X , dist) be the metric space with VC dimension d V C ( X ) . Given dataset P O ⊆ X , assume S O is a uniform sample of size ˜ O ( d V C ( X ) ε 2 ) from P O , then with probability 1 -1 /poly (1 /ε ) , S O is an ε -approximation of the ball range space on P O .

The Lemma E.2, used earlier for the Euclidean space case, is a direct corollary, since in Euclidean space R d , we have d V C ( X ) = O ( d ) . This lemma illustrates the number of samples required from L ⋆ .

Doubling dimension. Another notion to measure the complexity of the ball range space is the doubling dimension.

Definition E.9 (Doubling dimension [31, 3]) . The doubling dimension ddim( X ) of a metric space ( X , dist) is the least integer t such that for any Ball( c, u ) with c ∈ X , u &gt; 0 , it can be covered by 2 t balls of radius u/ 2 .

We denote the metric space with bounded doubling dimension as doubling metric . Based on [38], it is known that the VC dimension of a doubling metric ( X , dist) may not be bounded. Therefore, the previous Lemma E.8 does not apply for an effective bound. We want to directly relate the ball range space bound to the doubling dimension. This goal can be achieved by the following refined lemma.

Lemma E.10 (Ball range space approximation in doubling metrics [38]) . Let M = ( X , dist) be the metric space with doubling dimension ddim( X ) . Given P O ⊆ X , assume S O is a uniform sample of size ˜ O (ddim( X ) ε -2 ) from P O , then with probability 1 -1 /poly (1 /ε ) , S O is an ε -approximation of the ball range space on P O .

Proof. Suppose the distorted distance function dist ′ : X × X → R ≥ 0 satisfies: for any x, y ∈ X , we have (1 -ε )dist( x, y ) ≤ dist ′ ( x, y ) ≤ (1 + ε )dist( x, y ) . By Lemma 3.1 of [38], we know that with probability 1 -poly (1 /ε ) , S O is an ε -ball range space approximation of P O w.r.t. dist ′ , when S O = ˜ O (ddim( X ) ε -2 ) . By setting of dist ′ , we know S O is an O ( ε ) -ball range space approximation of P O w.r.t. dist , which completes the proof.

This lemma illustrates the number of points needed to be sampled in order to bound the induced error of S O when the metric space is a doubling metric.

The remaining problem is to bound the induced error of S I . We give out a generalized version of Theorem 3.1 as below.

Theorem E.11 (Refined from Corollary 5.4 in [40]) . Let M = ( X , dist) be the metric space. There exists a randomized algorithm that in O ( n ) time constructs a weighted subset S I ⊆ P ⋆ I of size:

- ˜ O ( d V C ( X ) · ε -4 ) , w.r.t. the VC dimension d V C ( X ) ,
- ˜ O (ddim( X ) · ε -2 ) , w.r.t. the doubling dimension ddim( X ) ,

such that for every dataset P O of size m , every integer 0 ≤ t ≤ m and every center c ∈ R d , | cost ( t ) ( P O ∪ P ⋆ I , c ) -cost ( t ) ( P O ∪ S I , c ) | ≤ ε · cost ( t ) ( P O ∪ P ⋆ I , c ) + 2 ε · cost( P ⋆ I , c ⋆ ) .

By combining the discussions of the errors induced by S O and S I , we present the main theorem for constructing a coreset for the robust geometric median across various metric spaces. We study the shortest-path metric ( X , dist) , where X is the vertex set of a graph G = ( V, E ) , and dist( · , · ) measures the shortest distance between two points in the graph. The treewidth of a graph measures how 'tree-like' the graph is; see Definition 2.1 of [4] for formal definition. A graph excluding a fixed minor is one that does not contain a particular substructure, known as the minor.

Theorem E.12 (Coreset size for robust geometric median in various metric spaces) . Let ε ∈ (0 , 1) . For a metric space M = ( X , dist) and a dataset X ⊂ X of size n ≥ 4 m , let S = S O ∪ S I be a sampled set of size

- ˜ O (log( |X| ) · ε -4 ) if M is general metric space.
- ˜ O (ddim( X ) · ε -2 ) if M is a doubling metric with doubling dimension ddim( X ) .
- ˜ O ( t · ε -4 ) if M is a shortest-path metric of a graph with bounded treewidth t .
- ˜ O ( | H | · ε -4 ) if M is a shortest-path metric of a graph that excludes a fixed minor H .

Then S is an ε -coreset for robust geometric median on X .

This theorem illustrates the coreset size with respect to different metric spaces, improving upon previous results for robust coresets by eliminating the O ( m ) dependency.

Proof of Theorem E.12. If M = ( X , dist) is a general metric space, then d V C ( X ) = O (log |X| ) . Thus, we have | S O | = ˜ O (log( |X| ) · ε -2 ) by Lemma E.8 and | S I | = ˜ O (log( |X| ) · ε -4 ) by Theorem E.11, leading to a coreset of size ˜ O (log( |X| ) · ε -4 ) .

If M = ( X , dist) is a doubling metric space, then by definition, ddim( X ) is bounded. Thus we have | S O | = ˜ O (ddim( X ) · ε -2 ) by Lemma E.10 and | S I | = ˜ O (ddim( X ) · ε -2 ) by Theorem E.11, leading to a coreset of size ˜ O (ddim( X ) · ε -2 ) .

If M = ( X , dist) is a shortest-path metric of a graph with bounded treewidth t , we have d V C ( X ) = O ( t ) by [8]. Thus, we have | S O | = ˜ O ( t · ε -2 ) by Lemma E.8 and | S I | = ˜ O ( t · ε -4 ) by Theorem E.11, leading to a coreset of size ˜ O ( t · ε -4 ) .

If M = ( X , dist) is a shortest-path metric of a graph that excludes a fixed minor H , we have d V C ( X ) = O ( | H | ) by [8]. Thus, we have | S O | = ˜ O ( | H | · ε -2 ) by Lemma E.8 and | S I | = ˜ O ( | H | · ε -4 ) by Theorem E.11, leading to a coreset of size ˜ O ( | H | · ε -4 ) .

## F Proof of Theorem 1.5: robust ( k, z ) -clustering

In this section, we provide a coreset construction for robust ( k, z ) -clustering when d ≥ 1 . We first extend the definition of the robust geometric median and its coreset to the robust ( k, z ) -clustering and its corresponding coreset.

Definition F.1 (Robust ( k, z ) -clustering) . Given a dataset P ⊂ R d of size n ≥ 1 and an integer m ≥ 0 , the robust ( k, z ) -clustering problem is to find a center set C ⊂ R d , | C | = k that minimizes the objective function below:

<!-- formula-not-decoded -->

where L represents the set of m outliers w.r.t. C and dist( p, C ) = min c ∈ C dist( p, c ) denotes the Euclidean distance from p to the closest center among C .

Before we define the coreset for robust ( k, z ) -clustering, we introduce the generalized cost function in the context of the robust ( k, z ) -clustering.

Definition F.2 (Generalized cost function for robust ( k, z ) -clustering) . Let m be an integer. Let S ⊆ R d be a weighted dataset with weights w ( p ) for each point p ∈ S . Let w ( S ) := ∑ p ∈ S w ( p ) . Define a collection of weight functions W := { w ′ : S → R + | ∑ p ∈ S w ′ ( p ) = w ( S ) -m ∧ ∀ p ∈ S, w ′ ( p ) ≤ w ( p ) } . Moreover, we define the following cost function on S :

<!-- formula-not-decoded -->

With this definition of the generalized cost function, we now define the notion of a coreset for robust ( k, z ) -clustering.

Definition F.3 (Coreset for robust ( k, z ) -clustering) . Given a point set P ⊂ R d of size n ≥ 1 , integer m ≥ 1 and ε ∈ (0 , 1) , we say a weighted subset S ⊆ P together with a weight function w : S → R + is an ε -coreset of P for robust ( k, z ) -clustering if w ( S ) = n and for any center set C ⊂ R d , | C | = k , cost ( m ) z ( S, C ) ∈ (1 ± ε ) · cost ( m ) z ( P, C ) .

Finally, we extend the concepts of inliers, outliers, and the ball range space.

Inliers and outliers. Throughout this section, we denote the approximate center set for robust ( k, z ) -clustering by C ⋆ = { c ⋆ 1 , . . . , c ⋆ k } ⊂ R d , which is an O (1) -approximation of the optimal solution. We then define the inliers and outliers with respect to this approximation center set.

Let L ⋆ := arg min L ⊂ P, | L | = m cost z ( P -L, C ⋆ ) denote the set of m outliers with respect to C ⋆ , and define P ⋆ I := P \ L ⋆ as the corresponding inlier set. The inlier set P ⋆ I is naturally partitioned by C ⋆ into k clusters { P ⋆ 1 , . . . , P ⋆ k } , where each cluster P ⋆ i contains the points in P ⋆ I closest to its corresponding center c ⋆ i . Additionally, define r max := max p ∈ P ⋆ I dist( p, C ⋆ ) which represents the maximum distance from the points in P ⋆ I to this center set C ⋆ . Let ¯ r := z √ cost z ( P ⋆ I ,C ⋆ ) n -m denote the average distance from P ⋆ I to C ⋆ . Under these notations, the second condition of Assumption 1.4 can be rewritten as ( r max ) z ≤ 4 k (¯ r ) z .

Ball range space. We introduce the concept of the k -balls range space, which is a direct extension of the ball range space defined in Definition E.1.

Definition F.4 (Approximation of k -balls range space, Definition F.2 in [39]) . Let Balls( C, u ) := ∪ c ∈ C Ball( c, u ) . For a given dataset P ⊂ R d , the k -balls range space on P is ( P, P ) where P := { P ∩ Balls( C, u ) | C ⊂ R d , | C | = k, u ∈ R + } . A subset Y ⊂ P is called an ε -approximation of the k -balls range space ( P, P ) if for every C ⊂ R d , | C | = k , u ∈ R + ,

<!-- formula-not-decoded -->

Based on this definition, we have the following lemma that measures the performance of S O for robust ( k, z ) -clustering.

Lemma F.5 (Refined from Lemma F.3 of [39]) . Given dataset P O ⊂ R d . Let S O be a uniform sampling of size ˜ O ( kd ε 2 ) from P O , then with probability at least 1 -1 poly ( k/ε ) , S O is an ε -approximation of the k -balls range space on P O . Define a weight function w : w ( p ) = | P O | | S O | , for any p ∈ S O . Then for any C ⊂ R d , | C | = k , u ∈ R + ,

<!-- formula-not-decoded -->

Then we are ready to give out our main result.

## F.1 Result for robust ( k, z ) -clustering

We first recall the following theorem.

Theorem F.6 (Restatement of Corollary 5.4 in [40]) . There exists a randomized algorithm A kd that in O ( nkd ) time constructs a weighted subset S I ⊆ P ⋆ I of size ˜ O ( k 2 ε -2 z min { ε -2 , d } )) , such that for every dataset P O of size m , every integer 0 ≤ t ≤ m and every center set C ⊂ R d , | C | = k , | cost ( t ) z ( P O ∪ P ⋆ I , C ) -cost ( t ) z ( P O ∪ S I , C ) | ≤ ε · cost ( t ) z ( P O ∪ P ⋆ I , C ) + 2 ε · cost z ( P ⋆ I , C ⋆ ) .

This theorem is a generalization of Theorem 3.1, describing the number of points that need to be sampled from P ⋆ I for robust ( k, z ) -clustering.

To adapt Algorithm 2 for Theorem 1.5, we only need to adjust the size of S O to ˜ O ( kε -2 z min { ε -2 , d } ) in Line 2 and modify the algorithm A d to A kd in Line 3 (see Algorithm 3). Consequently, the runtime remains dominated by Line 1 and Line 3, resulting in an overall complexity of O ( ndk ) according to Theorem F.6.

## Algorithm 3 Coreset Construction for General d and General k

Input: A dataset P ⊂ R d , ε ∈ (0 , 1) and an O (1) -approximate center set C ⋆ ⊂ R d Output: An ε -coreset S

- 1: L ⋆ ← arg min L ⊂ P, | L | = m cost z ( P -L, C ⋆ ) , P ⋆ I ← P -L ⋆
- 2: Uniformly sample S O ⊆ L ⋆ of size ˜ O ( kε -2 z min { ε -2 , d } ) . Set ∀ p ∈ S O , w O ( p ) ← m | S O | .
- 3: Construct ( S I , w I ) ←A kd ( P ⋆ I ) by Theorem F.6.
- 4: For any p ∈ S O , define w ( p ) = w O ( p ) and for any p ∈ S I , define w ( p ) = w I ( p ) .
- 5: Return S ← S O ∪ S I and w ;

Similar to the robust geometric median, we present the key lemma used to prove Theorem 1.5 below.

Lemma F.7 (Induced error of S O ) . For any center set C ⊂ R d , | C | = k , we have | cost ( m ) z ( P, C ) -cost ( m ) z ( S O ∪ P ⋆ I , C ) | ≤ O ( ε ) · cost ( m ) z ( P, C ) .

Theorem 1.5 follows as a corollary of Theorem F.6 and this lemma, with a proof similar to that of Theorem 1.3.

̸

Remark F.8 . The second condition in Assumption 1.4 can be replaced with dist( c ∗ i , c ∗ j ) z ≥ m · r z max min {| P ∗ i | , | P ∗ j |} for any c ∗ i , c ∗ j ∈ C ∗ , i = j . Under this modified assumption, Theorem 1.5 becomes a direct generalization of Theorem 1.3, as the second condition vanishes when k = 1 . This modification ensures that clusters are sufficiently well-separated, which explains why our algorithm performs well on the Bank dataset when k = 3 , as this modified assumption is satisfied in this case.

Below we illustrate why this assumption also works. If points in each cluster are mostly assigned to distinct centers, then cost ( m ) z ( P, C ) ≈ cost ( m ) z ( P, C ⋆ ) , making the error | cost ( m ) z ( P, C ) -cost ( m ) z ( P ⋆ I ∪ S O , C ) | easy to bound. Alternatively, if two clusters each have at least half of their points assigned to the same center, then by the modified assumption, cost ( m ) z ( P, C ) &gt; mr z max , which is large enough to bound the error induced by S O .

## F.2 Proof of Lemma F.7: Induced error of S O

In this section, we fix a center set C ⊂ R d of size k and define T O := min p ∈ L ⋆ (dist( p, C )) z , T I := max p ∈ P ⋆ I (dist( p, C )) z . We observe that Lemmas E.3 and E.4 can be directly extended to robust ( k, z ) -clustering. Therefore, the induced error of each point is at most T I -T O . Next, we show that T I -T O ≤ O (cost ( m ) z ( P, c ) /m ) , supported by a detailed discussion of the sizes of T I and T O (see Lemma F.9). The remaining task is to ensure that the number of points that may induce error is O ( εm ) . The result of Lemma E.6 can be extended to robust ( k, z ) -clustering when S O is an ε -approximation of the k -balls range space on L ⋆ . By Lemma F.5, we sample ˜ O ( kε -2 z min { ε -2 , d } ) points from L ⋆ to ensure the approximation.

We then demonstrate how to prove that T I -T O ≤ O (cost ( m ) z ( P, c ) /m ) .

Lemma F.9 (Bounds for T I and T O ) . When m P &lt; m , we have T I -T O ≤ O (cost ( m ) z ( P, c ) /m )

Proof. Let d max := max c ⋆ ∈ C ⋆ dist( c ⋆ , C ) denote the maximum distance of points in C ⋆ to C . Let d ′ max := max c ∈ C dist( c, C ⋆ ) denote the maximum distance of points in C to C ⋆ .

We first claim that T I ≤ ( r max + d max ) z . Let ̂ p be defined as max p ∈ P ⋆ I dist( p, C ) . Denote c ⋆ p as the closest approximate center to ̂ p , where c ⋆ p := arg min c ⋆ ∈ C ⋆ dist( ̂ p, c ⋆ ) , and c p as the closest center to c ⋆ p , where c p := arg min c ∈ C dist( c ⋆ p , c ) . This setup allows us to establish an upper bound for T I :

<!-- formula-not-decoded -->

thus, T I ≤ ( r max + d max ) z .

Next, we discuss the relationship between C and C ⋆ in two cases: 1) r max ≤ d max ; 2) r max &gt; d max .

Case 1: d max ≥ r max In this case, we have T I -T O ≤ 2 z d z max since T O ≥ 0 . Therefore, it suffices to prove 2 z d z max ≤ O (cost ( m ) z ( P, C ) /m ) . Suppose dist( c ⋆ i , C ) = d max , let ( ¯ r i ) z := cost z ( P ⋆ i , C ⋆ ) / | P ⋆ i | . Based on the scale of d max , we discuss in two cases.

Case 1.1: d max ≤ 8¯ r i By definition of ¯ r i , we have

<!-- formula-not-decoded -->

Since C ⋆ is an O (1) -approximate solution for robust ( k, z ) -clustering, we have

<!-- formula-not-decoded -->

Case 1.2: d max &gt; 8¯ r i

Let ̂ P i := { p ∈ P ⋆ i | dist( p, C ⋆ ) ≤ 4¯ r i } . By definition, we have | ̂ P i | ≥ 3 4 | P ⋆ i | . For any point p ∈ ̂ p i , suppose c p := arg min c ∈ C dist( p, c ) , then we have

<!-- formula-not-decoded -->

Since d max &gt; 8¯ r i , we obtain

<!-- formula-not-decoded -->

Overall, we have T I -T O ≤ O (cost ( m ) z ( P, C ) /m ) .

Case 2: d max &lt; r max We have T I -T O &lt; 2 z · r z max since T O ≥ 0 . By Item 2 of Assumption 1.4, we have m · r z max ≤ n · ¯ r z , thus T I -T O ≤ 2 z · cost ( m ) z ( P, C ) /m .

Combine Case 1 and Case 2, we complete the proof.

## F.3 Extension to other metric spaces

In this section, we explore the extension of Algorithm 3 for robust ( k, z ) -clustering in Euclidean space, to various other metric spaces.

Similar to the analysis for robust geometric median, we discuss the induced error of S I and S O separately. For the error induced by S O , Lemma F.9 still holds, so it suffices for S O to be an ε -approximation of the k -balls range space.

VC dimension We begin by introducing the concept of the VC dimension of the k -balls range space.

Definition F.10 (VC dimension of k -balls range space) . Let ( X , dist) be the metric space and define Balls k ( X ) := { Balls( C, u ) | C ⊂ X , | C | = k, u &gt; 0 } . The VC dimension of the k -balls range space ( X , Balls k ( X )) , denoted by d V C ( X ) , is the maximum | P | , P ⊆ X such that | P ∩ Balls k ( X ) | = 2 | P | , where P ∩ Balls k ( X ) := { P ∩ Balls( C, u ) | Balls( C, u ) ∈ Balls k ( X ) } .

The VC dimension of ( X , Balls k ( X )) aligns to the pseudo-dimension of ( X , Balls k ( X )) used by [45]. Based on this observation, we give out a refined lemma from [45].

Lemma F.11 (Refined from [45]) . Let ( X , dist) be the metric space. Given dataset P O ⊆ X , assume S O is a uniform sample of size ˜ O ( k · d V C ( X ) ε 2 ) from P O , then with probability 1 -1 /poly ( k/ε ) , S O is a ε -approximation of the k -balls range space on P O .

This lemma illustrates the number of points needed to be sampled from L ⋆ .

Doubling dimension Similar to the Lemma E.10, we give out the Lemma F.12 that illustrates the number of points needed to be sampled in order to bound the induced error of S O .

Lemma F.12 (Balls range space approximation for the doubling dimension) . Let M = ( X , dist) be the metric space with doubling dimension ddim( X ) . Given P O ⊆ X , assume S O is a uniform sample of size ˜ O (ddim( X ) · ε -2 z · k ) from P O , then with probability 1 -1 /poly ( k, 1 /ε ) , S O is an ε -approximation of the k -balls range space on P O .

The remaining problem is to bound the induced error of S I . We give out a generalized version of Theorem E.11 as below.

Theorem F.13 (Refined from Corollary 5.4 in [40]) . Let ( X , dist) be the metric space. There exists a randomized algorithm that in O ( nk ) time constructs a weighted subset S I ⊆ P ⋆ I of size:

- ˜ O ( k 2 · d V C ( X ) · ε -2 z -2 ) , w.r.t. VC dimension d V C ( X ) ,
- ˜ O ( k 2 · ddim( X ) · ε -2 z ) , w.r.t. doubling dimension ddim( X ) ,

, such that for every dataset P O of size m , every integer 0 ≤ t ≤ m and every center set C ⊂ R d , | C | = k , | cost ( t ) z ( P O ∪ P ⋆ I , C ) -cost ( t ) z ( P O ∪ S I , C ) | ≤ ε · cost ( t ) z ( P O ∪ P ⋆ I , C )+2 ε · cost z ( P ⋆ I , C ⋆ ) .

By combining the discussions of the errors induced by S O and S I , we present the main theorem for constructing a coreset for the robust ( k, z ) -clustering across various metric spaces.

Theorem F.14 (Coreset size for robust ( k, z ) -clustering in various metric spaces) . Let ε ∈ (0 , 1) . For a metric space M = ( X , dist) and a dataset X ⊂ X satisfying Assumption 1.4, let S = S O ∪ S I be a sampled set of size

- ˜ O ( k 2 · log( |X| ) · ε -2 z -2 ) if M is general metric space.
- ˜ O ( k 2 · ddim( X ) · ε -2 z ) if M is a doubling metric with doubling dimension ddim( X ) .
- ˜ O ( k 2 · t · ε -2 z -2 ) if M is a shortest-path metric of a graph with bounded treewidth t .
- ˜ O ( k 2 · | H | · ε -2 z -2 ) if M is a shortest-path metric of a graph that excludes a fixed minor H .

Then S is an ε -coreset for robust ( k, z ) -clustering on X .

Table 4: Datasets used in our experiments. For each dataset, we report its size, dimension (DIM), number of outliers ( m ). The number of centers ( k ) is used in robust ( k, z ) -clustering. We also provide the values of min i | P ⋆ i | and ( r max / ¯ r ) z for robust ( k, z ) -clustering. Our Assumption 1.4 requires that min i | P ⋆ i | ≥ 4 m and ( r max / ¯ r ) z ≤ 4 k . The value Y (resp. N ) indicates these assumptions are satisfied or not. Note that the Athlete dataset is sourced from Kaggle, while the other datasets are from the UCI repository.

| DATASET         |       | DIM.   | m    | min i &#124; P ⋆ i &#124; (   | min i &#124; P ⋆ i &#124; (   | r max ¯ r ) z   | r max ¯ r ) z   |
|-----------------|-------|--------|------|-------------------------------|-------------------------------|-----------------|-----------------|
|                 | SIZE  |        |      | k z = 1                       | z = 2                         | z = 1           | z = 2           |
| TWITTER[11]     | 10 5  | 2      | 2000 | 5 8968,Y                      | 588,N                         | 3.819,Y         | 1.083,Y         |
| CENSUS1990 [46] | 10 5  | 68     | 2000 | 5 5927,N                      | 5927,N                        | 1.873,Y         | 3.252,Y         |
| BANK[47]        | 41188 | 10     | 824  | 5 1361,N                      | 1361,N                        | 4.674,Y         | 15.731,Y        |
| ADULT[6]        | 48842 | 6      | 977  | 5 4508,Y                      | 186,N                         | 5.151,Y         | 5.834,Y         |
| ATHLETE[36]     | 10000 | 4      | 200  | 5 1018,Y                      | 1018,Y                        | 2.923,Y         | 7.062,Y         |
| DIABETES[43]    | 10000 | 10     | 200  | 5 1415,Y                      | 1459,Y                        | 2.551,Y         | 2.527,Y         |

This theorem illustrates the coreset size with respect to different metric spaces for robust ( k, z ) -clustering, improving upon previous results by eliminating the O ( m ) dependency.

Proof of Theorem F.14. If M = ( X , dist) is a general metric space, then d V C ( X ) = O (log |X| ) . Thus, we have | S O | = ˜ O ( k · log( |X| ) · ε -2 ) by Lemma E.8 and | S I | = ˜ O ( k 2 · log( |X| ) · ε -4 ) by Theorem E.11, leading to a coreset of size ˜ O ( k 2 · log( |X| ) · ε -4 ) .

If M = ( X , dist) is a doubling metric space, then by definition, ddim( X ) is bounded. Thus we have | S O | = ˜ O ( k · ddim( X ) · ε -2 ) by Lemma E.10 and | S I | = ˜ O ( k 2 · ddim( X ) · ε -2 ) by Theorem E.11, leading to a coreset of size ˜ O ( k 2 · ddim( X ) · ε -2 ) .

If M = ( X , dist) is a shortest-path metric of a graph with bounded treewidth t , we have d V C ( X ) = O ( t ) by [8]. Thus, we have | S O | = ˜ O ( k · t · ε -2 ) by Lemma E.8 and | S I | = ˜ O ( k 2 · t · ε -4 ) by Theorem E.11, leading to a coreset of size ˜ O ( k 2 · t · ε -4 ) .

If M = ( X , dist) is a shortest-path metric of a graph that excludes a fixed minor H , we have d V C ( X ) = O ( | H | ) by [8]. Thus, we have | S O | = ˜ O ( k · | H | · ε -2 ) by Lemma E.8 and | S I | = ˜ O ( k 2 · | H | · ε -4 ) by Theorem E.11, leading to a coreset of size ˜ O ( k 2 · | H | · ε -4 ) .

## G Additional empirical results

## G.1 Additional empirical results for robust geometric median

We first present Table 4, which lists the parameters of the datasets used in our experiments.

We show the missing result of Section 4 on Bank , Athlete and Diabetes dataset in Figure 4. This figure demonstrates that our coreset construction algorithm consistently outperforms the baselines. For instance, on the Diabetes dataset, our method produces a coreset of size 180 with an empirical error of 0 . 057 , while the best baseline, HLLW25 , results in a coreset size of 280 with an empirical error of 0 . 065 .

Analysis of our algorithm when n &lt; 4 m . For robust geometric median, we set m = n/ 2 in the six dataset, violating the assumption n ≥ 4 m , and report the corresponding size-error tradeoff in Figure 5. The results show that our method still consistently outperforms the baselines even when n &lt; 4 m . For instance, in Census1990 dataset, our method produces a coreset of size 50200 with an empirical error of 0 . 004 , while the best baseline produces a coreset of size 51200 with a much larger empirical error of 0 . 012 .

Figure 4: Tradeoff between coreset size | S | and empirical error ̂ ε ( S ) . The horizontal axis is | S | and the vertical axis is ̂ ε ( S ) .

<!-- image -->

Figure 5: Tradoff between coreset size | S | and empirical error ̂ ε ( S ) for robust geometric median when we set m = n/ 2 . In this scenario, the assumption n ≥ 4 m is violated.

<!-- image -->

Analysis under heavy-tailed contamination. For each dataset, we randomly perturb 10% of the points by adding independent Cauchy(0 , 1) 4 noise to every dimension, thereby simulating a heavy-tailed data environment. As shown in Figure 6, our method consistently outperforms the baselines on the robust geometric median task, demonstrating that its performance advantage remains stable even under heavy-tailed contamination. For example, on the perturbed Twitter dataset, our method achieves a coreset of size 1500 with an empirical error of 0 . 031 , whereas the best baseline, HLLW25 , requires a coreset twice as large ( 3000 ) to attain a higher error of 0 . 032 .

## G.2 Empirical results for robust 1D geometric median

We implement our 1D coreset construction algorithm and compare its performance to the previous baselines and our general dimension method. All experiments are conducted on a PC with an Intel Core i9 CPU and 16GB of memory, and the algorithms are implemented in C++ 11.

Setup. We select the first dimension of the Twitter dataset, the second dimension of the Adult dataset, and the second dimension of the Bank dataset to create the Twitter1D , Adult1D , and Bank1D datasets. We choose these dimensions because other dimensions in the four datasets listed in Table 4 contain no more than 130 distinct points, which would result in an overly small coreset. We conduct experiments on these 1D datasets. To compute an approximate center for each dataset, we use the k -means++ algorithm.

4 The probability density function of Cauchy(0 , 1) is f ( x ) = 1 π (1+ x 2 ) for any x ∈ R .

Figure 6: Tradoff between coreset size | S | and empirical error ̂ ε ( S ) for robust geometric median when we perturb 10% points.

<!-- image -->

Experiment result. We vary the coreset size from 200 to m +1000 , and compute the empirical error ̂ ε ( S ) w.r.t. the coreset size | S | . For each size and each algorithm, we independently run the algorithm 10 times and obtain 10 coresets, compute their empirical errors ̂ ε ( S ) , and report the average of 10 empirical errors. Figure 7 presents our results. This figure shows that our 1D coreset (denoted by Our1D ) outperforms the previous baselines and our coreset for general dimension. For example, with the Twitter1D dataset, our 1D method provides a coreset of size 320 with an empirical error 0 . 013 . The best empirical error achieved by our general dimension method for a coreset size 3500 is much larger, 0 . 087 . Note that the coreset size of Our1D method does not grow uniformly, since the number of buckets does not increase linearly with ε -1 or ε -1 / 2 .

Figure 7: Tradeoff between coreset size | S | and empirical error ̂ ε ( S ) in 1D datasets.

<!-- image -->

## G.3 Empirical results for robust k -median

We implement Algorithm 3 for robust k -median and compare its performance to the previous baselines.

Setup. We do experiments on the six datasets listed in Table 4, and set the number of centers k to be 5 . We use Lloyd version k -means++ to compute an approximate center C ⋆ .

Coreset size and empirical error tradeoff for robust k -median. We vary the coreset size from m to 2 m , and compute the empirical error ̂ ε ( S ) w.r.t. the coreset size | S | . For each size and each algorithm, we independently run the algorithm 10 times and obtain 10 coresets, compute their empirical errors ̂ ε ( S ) , and report the average of 10 empirical errors. Figure 8 presents our results.

This figure shows that our algorithm ( Ours ) outperforms the previous baselines. For example, with the Census1990 dataset, our method provides a coreset of size 2200 with empirical error 0 . 012 . The best empirical error achieved by baselines for a coreset size 3600 is larger, 0 . 013 .

Figure 8: Tradeoff between coreset size | S | and empirical error ̂ ε ( S ) for robust k -median on realworld datasets.

<!-- image -->

Statistical test. Similar to the robust geometric median, we evaluate the statistical performance between our method and baselines for robust k -median on all six real-world datasets. The results, listed in Table 5, further demonstrate that our algorithm consistently outperforms the baselines.

Table 5: Statistical comparison of different coreset construction methods for robust k -median. The coreset S 1 represents our coreset, S 2 represents the coreset constructed by the baseline HJLW23 , and S 3 the coreset constructed by baseline HLLW25 . For each empirical error ratio ̂ ε ( S 2 ) / ̂ ε ( S 1 ) and ̂ ε ( S 3 ) / ̂ ε ( S 1 ) , we report the mean value over 20 runs, with the subscript indicating the standard deviation.

| Coreset Size   | Census1990                | Census1990                | Twitter                   | Twitter                   |
|----------------|---------------------------|---------------------------|---------------------------|---------------------------|
| Coreset Size   | ̂ ε ( S 2 ) / ̂ ε ( S 1 ) | ̂ ε ( S 3 ) / ̂ ε ( S 1 ) | ̂ ε ( S 2 ) / ̂ ε ( S 1 ) | ̂ ε ( S 3 ) / ̂ ε ( S 1 ) |
| 2200           | 3 . 374 0 . 818           | 3 . 800 1 . 262           | 2 . 958 0 . 785           | 3 . 149 1 . 477           |
| 3200           | 1 . 800 0 . 659           | 1 . 483 0 . 534           | 1 . 654 0 . 788           | 1 . 574 0 . 710           |
| 4200           | 1 . 543 0 . 559           | 1 . 316 0 . 373           | 1 . 408 0 . 439           | 1 . 379 0 . 369           |
| Coreset Size   | Bank                      | Bank                      | Adult                     | Adult                     |
| Coreset Size   | ̂ ε ( S 2 ) / ̂ ε ( S 1 ) | ̂ ε ( S 3 ) / ̂ ε ( S 1 ) | ̂ ε ( S 2 ) / ̂ ε ( S 1 ) | ̂ ε ( S 3 ) / ̂ ε ( S 1 ) |
| 1200           | 1 . 657 0 . 415           | 2 . 019 0 . 740           | 2 . 098 0 . 551           | 2 . 153 0 . 772           |
| 1700           | 1 . 548 0 . 582           | 1 . 257 0 . 543           | 1 . 440 0 . 619           | 1 . 702 0 . 584           |
| 2200           | 1 . 393 0 . 558           | 1 . 460 0 . 737           | 1 . 450 0 . 677           | 1 . 373 0 . 545           |
| Coreset Size   | Athlete                   | Athlete                   | Diabetes                  | Diabetes                  |
| Coreset Size   | ̂ ε ( S 2 ) / ̂ ε ( S 1 ) | ̂ ε ( S 3 ) / ̂ ε ( S 1 ) | ̂ ε ( S 2 ) / ̂ ε ( S 1 ) | ̂ ε ( S 3 ) / ̂ ε ( S 1 ) |
| 210            | 11 . 481 3 . 633          | 8 . 851 0 . 740           | 12 . 688 3 . 443          | 8 . 080 1 . 652           |
| 310            | 2 . 142 0 . 525           | 1 . 798 0 . 553           | 1 . 902 0 . 512           | 1 . 402 0 . 395           |
| 410            | 2 . 104 0 . 570           | 1 . 523 0 . 481           | 1 . 691 0 . 546           | 1 . 257 0 . 470           |

Speed-up baselines. We compare the coreset of size 2 m constructed by the HLLW25 baselines and coreset of size m conducted by Algorithm 3. We repeat the experiment 10 times and report the aver-

ages. The result is listed in Table 6. Our algorithm achieves a speed-up over HLLW25 -specifically, a 2 × reduction in the running time on the coreset-while maintaining the same level of empirical error.

Validity of Assumption 1.4 for robust k -median. We evaluate the validity of Assumption 1.4 for robust k -median across six datasets. The results in Table 4 show that the assumptions are satisfied by the Twitter , Adult , Athlete , and Diabetes datasets, while the condition r max ≤ 4 k ¯ r holds for all six datasets. These results demonstrate that our assumptions are practical in real-world scenarios, and our algorithm performs well even when the assumption min i | P ⋆ i | ≥ 4 m is violated. Note that the condition r max ≤ 4 k ¯ r is satisfied by all six real-world datasets, even across different choices of k and m .

Analysis under heavy-tailed contamination. For each dataset, we randomly perturb 10% of the points by adding independent Cauchy(0 , 1) noise to every dimension, thereby simulating a heavytailed data environment. As shown in Figure 9, our method consistently outperforms the baselines on the robust k -median task, confirming that its superior performance remains stable even under heavy-tailed contamination. For example, on the perturbed Bank dataset, our method achieves a coreset of size 600 with an empirical error of 0 . 044 , whereas the best baseline, HLLW25 , requires a coreset of size 1300 to reach a higher error of 0 . 046 .

Figure 9: Tradoff between coreset size | S | and empirical error ̂ ε ( S ) for robust k -median when we perturb 10% points.

<!-- image -->

## G.4 Empirical results for robust k -means

We implement Algorithm 3 for robust k -means and compare its performance to the previous baselines.

Setup. We do experiments on the six datasets listed in Table 4, and set the number of centers k to be 5 . We use k -means++ to compute an approximate center C ⋆ .

Coreset size and empirical error tradeoff for Robust k -means. We vary the coreset size from m to 2 m , and compute the empirical error ̂ ε ( S ) w.r.t. the coreset size | S | . For each size and each algorithm, we independently run the algorithm 10 times and obtain 10 coresets, compute their empirical errors ̂ ε ( S ) , and report the average of 10 empirical errors. Figure 10 presents our results. This figure shows that our algorithm ( Ours ) outperforms the previous baselines. For example, with the Twitter dataset, our method provides a coreset of size 2200 with an empirical error 0 . 039 . The best empirical error achieved by our general dimension method for a coreset size 4000 is much larger, 0 . 056 .

Table 6: Comparison of runtime between our Algorithm 3 and baseline HLLW25 for robust k -median. For each dataset, the coreset size of baseline HLLW25 is 2 m and the coreset size of ours is m . We use Lloyd algorithm given by [7] to compute approximate solutions C P and C S for both the original dataset P and coreset S , respectively. 'COST P ' denotes cost ( m ) 1 ( P, C P ) on the original dataset P . 'COST S ' denotes cost ( m ) 1 ( P, C S ) on the coreset constructed by METHOD. T X is the running time on the original dataset. T S is the running time on coreset. T C is the construction time of the coreset.

| DATASET    | COST P       | METHOD      | COST S                    |     T X | T C           | T S          |
|------------|--------------|-------------|---------------------------|---------|---------------|--------------|
| CENSUS1990 | 1.032 × 10 6 | OURS HLLW25 | 1.030 × 10 6 1.040 × 10 6 | 312.606 | 38.862 38.703 | 5.532 10.815 |
| TWITTER    | 1.328 × 10 6 | OURS HLLW25 | 1.364 × 10 6 1.347 × 10 6 |  96.024 | 12.220 12.452 | 1.621 2.995  |
| BANK       | 3.179 × 10 6 | OURS HLLW25 | 3.194 × 10 6 3.207 × 10 6 | 147.324 | 9.021 9.210   | 1.484 3.445  |
| ADULT      | 9.221 × 10 8 | OURS HLLW25 | 9.153 × 10 8 9.166 × 10 8 | 144.185 | 9.324 9.266   | 1.854 3.517  |
| ATHLETE    | 7.251 × 10 4 | OURS HLLW25 | 7.503 × 10 4 7.389 × 10 4 |  41.541 | 1.825 1.872   | 0.206 0.421  |
| DIABETES   | 8.571 × 10 4 | OURS HLLW25 | 8.791 × 10 4 8.811 × 10 4 |  44.733 | 1.829 1.850   | 0.226 0.473  |

Figure 10: Tradeoff between coreset size | S | and empirical error ̂ ε ( S ) for robust k -means.

<!-- image -->

Statistical test. Similar to the previous cases, we evaluate the statistical performance between our method and baselines for robust k -means on all six real-world datasets. The results, listed in Table 7, further demonstrate that our algorithm consistently outperforms the baselines.

Speed-up baselines. We compare the coreset of size 2 m constructed by the HLLW25 baselines and coreset of size m conducted by Algorithm 3. We repeat the experiment 10 times and report the averages. The result is listed in Table 8. Our algorithm achieves a speed-up over HLLW25 -specifically, a 2 × reduction in the running time on the coreset-while maintaining the same level of empirical error.

Validity of Assumption 1.4 for robust k -means. We evaluate the validity of Assumption 1.4 for robust k -means across six datasets. The results in Table 4 show that our assumptions are satisfied by datasets Athlete and Diabetes , while the assumption r 2 max ≤ 4 k ¯ r 2 is satisfied by all six datasets.

Table 7: Statistical comparison of different coreset construction methods for robust geometric median. The coreset S 1 represents our coreset, S 2 represents the coreset constructed by the baseline HJLW23 , and S 3 the coreset constructed by baseline HLLW25 . For each empirical error ratio ̂ ε ( S 2 ) / ̂ ε ( S 1 ) and ̂ ε ( S 3 ) / ̂ ε ( S 1 ) , we report the mean value over 20 runs, with the subscript indicating the standard deviation.

| Coreset Size   | Census1990                | Census1990                | Twitter                   | Twitter                   |
|----------------|---------------------------|---------------------------|---------------------------|---------------------------|
| Coreset Size   | ̂ ε ( S 2 ) / ̂ ε ( S 1 ) | ̂ ε ( S 3 ) / ̂ ε ( S 1 ) | ̂ ε ( S 2 ) / ̂ ε ( S 1 ) | ̂ ε ( S 3 ) / ̂ ε ( S 1 ) |
| 2200           | 3 . 253 2 . 063           | 2 . 645 1 . 458           | 1 . 793 0 . 644           | 1 . 667 0 . 479           |
| 3200           | 1 . 257 0 . 842           | 1 . 251 0 . 632           | 1 . 343 0 . 234           | 1 . 283 0 . 197           |
| 4200           | 1 . 303 0 . 692           | 1 . 168 0 . 739           | 1 . 244 0 . 152           | 1 . 246 0 . 148           |
| Coreset Size   | Bank                      | Bank                      | Adult                     | Adult                     |
| Coreset Size   | ̂ ε ( S 2 ) / ̂ ε ( S 1 ) | ̂ ε ( S 3 ) / ̂ ε ( S 1 ) | ̂ ε ( S 2 ) / ̂ ε ( S 1 ) | ̂ ε ( S 3 ) / ̂ ε ( S 1 ) |
| 1200           | 1 . 647 0 . 972           | 1 . 360 1 . 018           | 1 . 467 0 . 287           | 1 . 094 0 . 542           |
| 1700           | 1 . 010 0 . 654           | 1 . 028 0 . 574           | 2 . 149 0 . 884           | 2 . 416 1 . 002           |
| 2200           | 1 . 010 0 . 654           | 1 . 026 0 . 674           | 1 . 089 0 . 360           | 1 . 172 0 . 537           |
| Coreset Size   | Athlete                   | Athlete                   | Diabetes                  | Diabetes                  |
| Coreset Size   | ̂ ε ( S 2 ) / ̂ ε ( S 1 ) | ̂ ε ( S 3 ) / ̂ ε ( S 1 ) | ̂ ε ( S 2 ) / ̂ ε ( S 1 ) | ̂ ε ( S 3 ) / ̂ ε ( S 1 ) |
| 210            | 5 . 172 3 . 634           | 4 . 200 1 . 944           | 5 . 700 3 . 303           | 5 . 868 2 . 952           |
| 310            | 2 . 467 1 . 564           | 1 . 427 0 . 660           | 1 . 567 0 . 800           | 1 . 332 0 . 653           |
| 410            | 1 . 658 0 . 881           | 1 . 045 0 . 449           | 1 . 360 1 . 103           | 1 . 216 0 . 943           |

Table 8: Comparison of runtime between our Algorithm 3 and baseline HLLW25 for robust k -means. For each dataset, the coreset size of baseline HLLW25 is 2 m and the coreset size of ours is m . We use Lloyd algorithm given by [7] to compute approximate solutions C P and C S for both the original dataset P and coreset S , respectively. 'COST P ' denotes cost ( m ) 2 ( P, C P ) on the original dataset P . 'COST S ' denotes cost ( m ) 2 ( P, C S ) on the coreset constructed by METHOD. T X is the running time on the original dataset. T S is the running time on coreset. T C is the construction time of the coreset.

| DATASET    | COST P        | METHOD      | COST S                      |     T X | T C           | T S          |
|------------|---------------|-------------|-----------------------------|---------|---------------|--------------|
| CENSUS1990 | 1.172 × 10 7  | OURS HLLW25 | 1.170 × 10 7 1.170 × 10 7   | 358.218 | 35.513 35.399 | 5.770 11.043 |
| TWITTER    | 2.657 × 10 7  | OURS HLLW25 | 2.664 × 10 7 2.662 × 10 7   | 106.327 | 14.084 13.330 | 1.806 3.494  |
| BANK       | 3.477 × 10 8  | OURS HLLW25 | 3.531 × 10 8 3.530 × 10 8   | 133.978 | 7.478 7.225   | 1.283 2.725  |
| ADULT      | 2.575 × 10 13 | OURS HLLW25 | 2.646 × 10 13 2.652 × 10 13 | 139.16  | 8.273 8.048   | 1.564 3.091  |
| ATHLETE    | 8.630 × 10 5  | OURS HLLW25 | 9.089 × 10 5 9.016 × 10 5   |  44.208 | 1.890 1.909   | 0.251 0.466  |
| DIABETES   | 6.490 × 10 5  | OURS HLLW25 | 6.814 × 10 5 6.838 × 10 5   |  39.57  | 1.622 1.602   | 0.196 0.413  |

These results show that our assumptions are practical in real-world datasets, and our algorithm performs well even when the assumption min i | P ⋆ i | ≥ 4 m is violated.

Analysis of our algorithm when Assumption 1.4 violates. We evaluate the applicability of our algorithm when both assumptions min i | P ⋆ i | ≥ 4 m and ( r max / ¯ r ) 2 ≤ 4 k are violated. In Bank dataset, we let k = 3 , then we have ( r max / ¯ r ) 2 = 15 . 001 &gt; 4 k and min i | P ⋆ i | = 1432 &lt; 4 m , violating the two assumptions. In Adult dataset, we let k = 8 , m = 500 , then we have ( r max / ¯ r ) 2 = 36 . 278 &gt; 4 k and min i | P ⋆ i | = 1592 &lt; 4 m , violating the two assumptions. We present the size-error tradeoff for robust k -means on these datasets in Figure 11. Our results show that our method still outperforms the baselines even when both assumptions are violated.

Figure 11: Tradeoff between coreset size | S | and empirical error ̂ ε ( S ) for robust k -means. We set k = 3 in the Bank dataset and set k = 8 , m = 500 in the Adult dataset. In these cases, the values ( r max / ¯ r ) 2 and min i ∈ [ k ] | P ⋆ i | violate both of our assumptions. The goal of this figure is to examine whether our algorithm remains applicable when the assumptions are not satisfied.

<!-- image -->

Analysis under heavy-tailed contamination. For each dataset, we randomly perturb 10% of the points by adding independent Cauchy(0 , 1) noise to every dimension, thereby simulating a heavytailed data environment. As shown in Figure 12, our method consistently outperforms the baselines on the robust k -means task. For example, on the perturbed Athlete dataset, our method yields a coreset of size 110 with an empirical error of 0 . 181 , whereas the best baseline, HLLW25 , attains a higher error of 0 . 196 even with a larger coreset of size 290 .

Figure 12: Tradoff between coreset size | S | and empirical error ̂ ε ( S ) for robust k -means when we perturb 10% points.

<!-- image -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: In Section 1.1, we list the contributions of this paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and empirical results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The Theorems 1.3 and 1.5 demonstrate that our coreset size is not tight for d &gt; 1 .

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

Justification: In Section 1.1, we present each theorem's assumptions and cite their formal definitions. Sections 2, 3, and F.2 list the lemmas required to prove the theorems.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. empirical result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main empirical results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Algorithms 1, 2, and 3 present the methods used in our experiments. The corresponding code is provided in the supplementary material.

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

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main empirical results, as described in supplemental material?

Answer: [Yes]

Justification: We provide our code in the supplemental material.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Plea the NeurIPS code and data submission guidelines ( https://nips.cc/public/ guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all empirical results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We demonstrate the parameters of datasets in Table 4.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: In Section 4, we report the empirical error across different coreset sizes for robust geometric median on each dataset. We list the results of the statistical significance experiments in Tables 1, 5 and 7.

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

Justification: We list out the compute resources in Section 4.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our research conducted in the paper conform with the NeurIPS Code of Ethics. Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This paper mainly propose the theoretical results, which has no societal impact of the work performed.

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

Answer: [Yes]

Justification: Table 4 lists the sources of our datasets.

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

Justification:

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.