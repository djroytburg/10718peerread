## Simple and Optimal Sublinear Algorithms for Mean Estimation

## Beatrice Bertolotti

University of Pavia, Italy beatrice.bertolotti02@universitadipavia.it

## Chris Schwiegelshohn

Aarhus University, Denmark schwiegelshohn@cs.au.dk

## Matteo Russo

Sapienza University of Rome, Italy mrusso@diag.uniroma1.it

## Sudarshan Shyam

Aarhus University, Denmark shyam@cs.au.dk

## Abstract

We study the sublinear multivariate mean estimation problem in d -dimensional Euclidean space. Specifically, we aim to find the mean µ of a ground point set A , which minimizes the sum of squared Euclidean distances of the points in A to µ . We first show that a multiplicative (1 + ε ) approximation to µ can be found with probability 1 -δ using O ( ε -1 log δ -1 ) many independent uniform random samples, and provide a matching lower bound. Furthermore, we give two estimators with optimal sample complexity that can be computed in optimal running time for extracting a suitable approximate mean:

1. The coordinate-wise median of log δ -1 sample means of sample size ε -1 . As a corollary, we also show improved convergence rates for this estimator for estimating means of multivariate distributions.
2. The geometric median of log δ -1 sample means of sample size ε -1 . To compute a solution efficiently, we design a novel and simple gradient descent algorithm that is significantly faster for our specific setting than all other known algorithms for computing geometric medians.

In addition, we propose an order statistics approach that is empirically competitive with these algorithms, has an optimal sample complexity and matches the running time up to lower order terms.

We finally provide an extensive experimental evaluation among several estimators which concludes that the geometric-median-of-means-based approach is typically the most competitive in practice.

## 1 Introduction

An extremely simple algorithmic paradigm is to sample a subset of the input independently and uniformly at random and solve a problem on the sample. Such algorithms are called sublinear algorithms and form a staple of large data analysis. The most desirable property of a sublinear algorithm is that the query of sample complexity is as small as possible. For most problems, it is rare for a sublinear algorithm with only constant queries to produce multiplicative guarantees. Remarkably, finding the multivariate mean of a point set A in Euclidean space is one of the cases where not only can we obtain a multiplicative error with constant queries, but can do so with arbitrary precision. Specifically, we are interested in the objective

<!-- formula-not-decoded -->

where ∥ x ∥ = √ ∑ d i =1 x 2 i denotes the Euclidean norm. It is well known that the mean of the point set A minimizes this expression. We say that c is a (1 + ε ) -approximate mean, if the cost of c is within a (1 + ε ) factor of the cost of the optimal mean.

The exact sample complexity for obtaining approximate means is still an open problem. Inaba et al. [1994] showed that ε -1 uniform and independent samples are necessary and sufficient for the empirical mean to be a (1 + ε ) -approximate mean on expectation. To extend this to a high probability guarantee, that is, a success probability of 1 -δ , O ( ε -1 δ -1 ) samples are sufficient. Unfortunately, this bound is tight when using the empirical mean (see Appendix E). Cohen-Addad et al. [2021] improved the dependence on the probability of failure by achieving a sample complexity of ˜ O ( ε -5 log 2 δ -1 ) , at the cost of a significantly worse dependency on ε . In the same paper, the authors also gave a lower bound of Ω( ε -1 ) for any sublinear algorithm succeeding with constant probability, showing that the result by Inaba et al. [1994] is optimal in the constant success probability regime. Subsequently, Musco et al. [2022] presented an algorithm with O ( ε -1 d (log 2 d · log ( dε -1 ) +log δ -1 ) log δ -1 ) queries, which combines the tight dependency on ε from Inaba et al. [1994] with the improved dependency on δ from Cohen-Addad et al. [2021], at the cost of a dependency on the dimension. Recently, Woodruff and Yasuda [2024] achieved almost best of all worlds trade-offs with a sample complexity of O ( ε -1 (log ε -1 +log δ -1 ) log δ -1 ) .

## 1.1 Our Contributions

Our main contributions are three algorithms that estimate the mean of a point set and that have optimal sample complexity. In addition, the running times are linear in the sample complexity for one and almost linear for the other two and thus optimal. In the first step of the algorithms, see also Algorithm 1, we draw O ( ε -1 log δ -1 ) points uniformly at random. We then split the sample into O (log δ -1 ) subsamples of size O ( ε -1 ) , compute the empirical mean for each subsample, and run an aggregation procedure, i.e., a subroutine AGGREGATE ( P ) that combines the sample set produced to output an estimate of the mean.

## Algorithm 1 MEANESTIMATE ( ε, δ )

for i = 1 , . . . , b log δ -1 do Sample S i points independently and uniformly at random with | S i | = aε -1 Compute the sample mean ˆ µ i = 1 | S i | · ∑ p ∈ S i p Output AGGREGATE (ˆ µ 1 , . . . , ˆ µ b log δ -1 )

The sample complexity of this algorithm, m = b log δ -1 · aε -1 turns out to be optimal, as shown in Section E. Phrased as a generalization question for bounding the excess risk given independent samples S from an underlying arbitrary but fixed distribution, we show an optimal convergence rate of Θ ( log δ -1 | S | ) .

The remaining question is to find a good solution (i.e., run AGGREGATE) quickly. Clearly, the sample complexity is also a lower bound for the running time of any algorithm. Achieving this running time and in particular, breaking through the O (log 2 δ -1 ) barrier inherent in all previous algorithms, is a substantial challenge, even for computing any constant approximation. It is fairly easy to show that most of the sample means are good estimators for the true mean. Selecting one of the good estimators is nevertheless difficult, as we have no way of estimating the cost and thus no easy way of distinguishing successful estimators from unsuccessful ones. Instead, we propose three algorithm to perform such a task.

Coordinate-Wise-Median-of-Means and Geometric-Median-of-Means. Both the coordinatewise-median-of-means and the geometric median-of-means are natural generalizations of the medianof-means estimator in a single dimension. It turns out both have optimal sample complexity, up to constant factors. The former can be straightforwardly computed in linear time. Achieving a good running time for the latter is more challenging. Black box applications of gradient descent methods do not offer fast running times to compute suitable medians in our setting, but we show that a relatively simple gradient descent (see Algorithm 2) finds a suitable solution in linear time.

Using theoretical benchmarks, it is not possible to determine the more favorable estimator and it is not too difficult to construct instances where one outperforms the other. We therefore conduct an extensive experimental evaluation on real world data sets to measure the performance of these two estimators, along with previously proposed algorithms and our third contribution, the MinSumSelect algorithm.

MinSumSelect. Our other algorithm, given in Algorithm 3 (in Appendix B), uses a carefully chosen order statistic combined with clustering ideas. Specifically, the observation is that the sum of distances of the t closest points allows us to distinguish between a successful estimate and an unsuccessful estimate, for an appropriate choice of t . Here, the running time improves to O (( ε -1 +log γ δ -1 ) · d log δ -1 ) , for any constant γ &gt; 0 . The algorithm can also be parameterized to achieve a similar running time to the gradient descent algorithm, at the cost of increasing the sample size by poly ( log log δ -1 ) factors.

## 1.2 Related Work

We covered the related work specific to sublinear algorithms for estimating the mean earlier, but there are several other closely related topics. Given a distribution, a classical question is to design good estimators for some underlying statistic. For high-dimensional means, there has been substantial attention recently, see Lee and Valiant [2022], Lugosi and Mendelson [2019, 2024] and references therein. There are common notions with our work, as the distance to the true mean and tail bounds for the estimators are an important quantity in our setting as well. Nevertheless, in this line of work, the running time, which is a very important focus for sublinear algorithms, is not a primary concern.

Another interesting line of work focuses on data dependent mean estimation, where the bounds derived are dependent on the quality of the data [Dang et al., 2023]. There are limited regimes in which estimators with better than sub-Gaussian estimators are possible. This is especially interesting given the recent efforts in the direction of beyond worst-case analysis of algorithms.

In a more general setting, one could ask for approximate power means minimizing the objective min c ∈ R d ∑ p ∈ A ∥ p -c ∥ z for some positive integer z . For z &gt; 2 , this problem was first posed by Cohen-Addad et al. [2021] who showed that ˜ O ( ε -z -3 log 2 δ -1 ) samples were sufficient for a high success probability while Ω( ε -z +1 ) samples were necessary for constant success probability. Woodruff and Yasuda [2024] improved this even further, obtaining a sample complexity of O ( ε -z +1 (log ε -1 +log δ -1 ) log δ -1 ) . Perhaps surprisingly, the case z = 1 , also known as the geometric median requires Θ( ε -2 log δ -1 ) many samples, see Chen and Derezinski [2021], Parulekar et al. [2021] for lower bounds and Cohen et al. [2016] for an optimal algorithm. Finally, the special case z →∞ corresponds to the minimum enclosing ball problem. The solution is highly susceptible to outliers, thus any sublinear algorithm must either be allowed to discard a fraction of the input, or rely on additive guarantees (see Clarkson et al. [2012] and Ding [2020] for examples).

The sample complexity for the k -means problem has been extensively studided in the context of generalization bounds. Here, we are given an arbitrary but fixed distribution D typically supported on the unit Euclidean sphere, we aim to find a set of k centers C minimizing ∫ p min c ∈ C ∥ p -c ∥ 2 P D [ p ] dp . Following a long line of work, the best currently known learning rates with a sample size n for this problem are of the order O (√ k log k | S | + log δ -1 | S | ) , see Bartlett et al. [1998], Clemençcon [2011], Cohen-Addad et al. [2025], Fefferman et al. [2016], Klochkov et al. [2021], Linder [2000], Narayanan and Mitter [2010], Pollard and references therein. Interestingly, the best known lower bounds on the learning rates for arbitrary values of k are at least Ω (√ k | S | ) , see Bartlett et al. [1998], Bucarelli et al. [2023].

Another related line of work is on coresets. For center-based objectives, a coreset approximates the cost of any given center with respect to the sum of distances raised to some power, where z = 2 is the power related to the 1 -means objective. While there are some exceptions that obtain very small coreset sizes via other techniques, see Afshani and Schwiegelshohn [2024], Braverman et al. [2022], Huang et al. [2023], Maalouf et al. [2021, 2022], the sample complexity of sensitivity sampling has by far been the most widely studied complexity measure in this line of work, see Bansal et al. [2024], Braverman et al. [2021], Cohen-Addad et al. [2021], Feldman and Langberg [2011], Langberg and Schulman [2010].

Further Comments and Limitations. We assume that the samples are drawn independently from a data set, and that there are no corruptions. Efficiency and scalability are not limitations, as the present work only has optimal results in that regard. Regarding notions of privacy: Private mean estimation is an important and well studied problem, but if we wish a purely multiplicative approximation, as we aim to do here, no privacy guarantees are possible. Regarding fairness notions, sublinear algorithms cannot be individually fair in a meaningful way, as we never access the entire data set. Other clustering fairness notions, such as representational fairness, do not apply for a single cluster.

## 2 Preliminaries

For a set of points A ⊂ R d , whose cardinality is | A | = n , we use µ ( A ) := 1 n · ∑ p ∈ A p to denote the mean, using µ if A is clear from the context. Denote by OPT = min c ∈ R d ∑ p ∈ A ∥ p -c ∥ 2 . In addition, for a set of points S sampled uniformly at random from A , we call ˆ µ ( S ) the empirical estimator of µ ( A ) , or simply ˆ µ if S is clear from context. A useful relationship between the goodness of ˆ µ and the size of | S | is established via the following identity.

<!-- formula-not-decoded -->

At the heart of most algorithms for Euclidean means lies the following (arguably folklore) identity. Lemma 2.2 (High Dimensional Mean-Variance Decomposition) . For any set of points A ⊂ R d and any c ∈ R d , we have ∑ p ∈ A ∥ p -c ∥ 2 = ∑ p ∈ A ∥ p -µ ∥ 2 + n · ∥ µ -c ∥ 2 .

Lemma 2.2 implies that ˆ µ is a (1 + ε ) appoximation if and only if ∥ ˆ µ -µ ∥ ≤ √ ε OPT n , which we will often use throughout this paper. For space reasons, some proofs are omitted in the main body and deferred to the appendix.

For the proofs in the next sections, we require a basic probabilistic lemma. We say that an empirical mean ˆ µ i is good , if ∥ ˆ µ i -µ ∥ ≤ r , where r = 1 11 √ ε OPT n . We denote the set of good sample means by G , and define event

<!-- formula-not-decoded -->

The following lemma is a straightforward application of the Chernoff bound.

Lemma 2.3. With probability 1 -δ , event E holds, for a sufficiently large absolute constants a and b .

Proof. We set the values of the constants to be a = 1440 , b = 50 and r = 1 11 √ ε OPT n (for reasons which will be clear in the proof). First, we show that a sample mean of aε -1 samples is good with probability at least 0 . 9 . This follows from Lemma 2.1 and Markov's inequality.

<!-- formula-not-decoded -->

Applying the standard Chernoff bound (with b = 50) , we get

<!-- formula-not-decoded -->

where the b is chosen so that -2 b 81 &lt; -1

.

## 3 Coordinate-Wise Median-of-Means

In this section, we show that the coordinate-wise median-of-means is an optimal estimator and thus leads to a linear time algorithm for mean estimation. That is, at the end of Algorithm 1 we simply

use AGGREGATE (ˆ µ 1 , . . . , ˆ µ b log δ -1 ) = COORDWISEMEDIAN (ˆ µ 1 , . . . , ˆ µ b log δ -1 ) as an aggregation procedure. We first do so in the context of sublinear algorithms and then extend the result to the case of distributional mean estimation in high dimensions.

## 3.1 Coordinate-Wise Median-of-Means in the Sublinear Model

We prove the following theorem:

Theorem 3.1. Algorithm 1 run with ν CWM := COORDWISEMEDIAN (ˆ µ 1 , . . . , ˆ µ b log δ -1 ) as an aggregation routine outputs a (1 + ε ) -approximate Euclidean mean with probability 1 -δ . The running time is O ( ε -1 · d log δ -1 ) .

Proof. Consider ν CWM, the coordinate-wise median of all the sample means. For each coordinate k , let L k and R k be the sets of sample means such that ˆ µ i,k ≤ ν CWM ,k and ˆ µ i,k ≥ ν CWM ,k respectively. We know that L k , R k have cardinality at least b log δ -1 2 . Depending on whether ν CWM ,k &gt; µ k or not, we show that at least one of the following statements is true:

1. For all ˆ µ in L k , we have that | ˆ µ i,k -µ k | ≥ | ν CWM ,k -µ k | .
2. For all ˆ µ in R k , we have that | ˆ µ i,k -µ k | ≥ | ν CWM ,k -µ k | .

We have | L k ∪ R k | = b log δ -1 which is the total number of sample means. First, from the definitions of L k and R k , we observe that there are points (such that ˆ µ k = ν CWM ,k ) which belong to both. Along with the definition of the coordinate-wise median, we have | L k | , | R k | ≥ b log δ -1 / 2 . Note that if there are no points such that ˆ µ k = ν CWM ,k , then | L k | = | R k | = b log δ -1 / 2 .

We provide a case analysis as to why at least 1 / 5 fraction of the samples are good and satisfy | ˆ µ i,k -µ k | ≥ | ν CWM ,k -µ k | .

First, we note that when the good event E holds, at least 1 / 5 fraction of the means are good in both the sets L k and R k . This follows from the fact that at least 7 / 10 fraction of the means are good and the number of good means in L k / R k is at least 7 / 10 -1 / 2 = 1 / 5 .

Case I: µ k ≤ ν CWM ,k . In this case, we have for all means in R k , | ˆ µ i,k -µ k | ≥ | ν CWM ,k -µ k | .

Case II: µ k &gt; ν CWM ,k . In this case, we have for all means in L k , | ˆ µ i,k -µ k | ≥ | ν CWM ,k -µ k | .

From the above statements, we have for at least one of L k and R k , we have that at least 1 / 5 fraction of the means are good and satisfy | ˆ µ i,k -µ k | ≥ | ν CWM ,k -µ k | .

From Lemma 1, we know that at least 7 10 fraction of the means are good (i.e., ∥ ˆ µ i -µ ∥ ≤ r ) with probability at least 1 -δ . Hence, we infer that at least 7 10 -1 2 = 1 5 of the sample means are good and satisfy | ˆ µ i,k -µ k | ≥ | ν CWM ,k -µ k | . On average,

<!-- formula-not-decoded -->

Summing over all coordinates k gives

<!-- formula-not-decoded -->

Interchanging the sums on the left-hand side, we get

<!-- formula-not-decoded -->

The theorem follows from the definition of r = 1 11 √ ε OPT n , and the running time follows by recognizing that we take O (log δ -1 ) sample means, each composed of O ( ε -1 ) many points in d dimensions.

## 3.2 Learning the Mean of a High-Dimensional Distribution

In this section, we relate the results in the literature with respect to mean estimation of high dimensional multivariate probability distributions and place our results in the context of these works.

Wefirst describe the setting. The goal is to estimate the mean µ of a probability distribution D in R d . A standard assumption is the existence of a covariance matrix Σ . We want the estimate to be close to the the true mean µ and the Euclidean distance to the mean is used as the objective. Given N i.i.d. samples X 1 , X 2 , . . . , X N ∼ D , the goal is to construct an estimator ν = ν ( X 1 , X 2 , . . . , X N ) such that, for any confidence interval δ ∈ [0 , 1] , with probability at least 1 -δ , we have ∥ µ -ν ∥ ≤ ε ( N,δ, Σ) . The function ε ( N,δ, Σ) is referred to as the rate of convergence of error henceforth.

The coordinate-wise median-of-means, while not optimal, is a simple and natural extension of the algorithm for the 1 -dimension. A simple analysis, found in several works, see Lugosi and Mendelson [2019] and Minsker [2015], applies a union bound over all dimensions and concludes that if the mean is concentrated along every dimension, it will be concentrated in general. This analysis proves a convergence rate of the order O (√ TR (Σ) log( dδ -1 ) N ) , where TR (Σ) is the trace of the covariance

matrix. Somewhat misleadingly, the dependency on log d is often implied to be necessary.

The analysis of the coordinate-wise median-of-means for sublinear algorithms can be adapted to yield convergence rate for the mean estimation problem. Specifically, the coordinate-wise median-of-means estimator has the following, dimension-free convergence rate.

Theorem 3.2. Let X 1 , X 2 , . . . , X N be independent samples from distribution D with variance µ and covariance matrix Σ , then the coordinate-wise median-of-means estimator ν CWM , with probability at least 1 -δ , satisfies

<!-- formula-not-decoded -->

The proof is an adaptation of the proof for our setting and the details are deferred to the appendix.

## 4 Geometric Median-of-Means and Gradient Descent

In the previous section, we showed that the coordinate-wise median of the sample means is a (1 + ε ) -approximation of the true mean and can be computed in linear time in the number of samples. However, one could generate instances where the coordinate-wise median is consistently a worse approximation to the true mean than the geometric median is, and vice-versa. This is shown in Appendix F. Therefore, we prove that the geometric median-of-means estimator also has an optimal sample complexity and give a gradient descent algorithm (Algorithm 2) for computing a sufficiently good estimate of it efficiently. It is important to note that Algorithm 2 does not always converge to the geometric median of the points. Take, for example, the triangle with coordinates (0 , 1) , (0 , -1) , (1 , 0) . The geometric median is the point ( 1 √ 3 , 0) . If we initialize the algorithm at the origin, the algorithm never makes any progress and stays at the origin.

The gradient of the geometric median objective ∑ p ∈ P ∥ p -c ∥ at any point q with respect to some reference point set P is ∇ ( q ) = ∑ p ∈ P ∇ p ( q ) , where ∇ p ( q ) := q -p ∥ q -p ∥ is the contribution to the gradient from point p (excluding co-located points q to make it well defined).

| Algorithm 2 FASTGD ( P )                                                                                                                                                                                                                                                                                                                                                 |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Compute ν CWM = COORDWISEMEDIAN (ˆ µ 1 , . . . , ˆ µ b log δ - 1 ) for j = 1 , . . .,T do Compute the gradient ∇ ( c j - 1 ) = ∑ p ∈ P c j - 1 - p ∥ c j - 1 - p ∥ Project all p i ∈ P onto c j - 1 -∇ ( c j - 1 ) , denoting the projection by p i,c j - 1 Compute the 1 -dimensional median c j of the p i,c j - 1 's along the line c j - 1 -∇ ( c j - 1 ) Output c T |

The proof that the geometric median-of-means estimator is optimal is algorithmic, i.e. it will follow as a result of the analysis of Algorithm 2. We initialize Algorithm 2 with P = { ˆ µ 1 , . . . , ˆ µ b log δ -1 } and

the coordinate-wise median ν CWM of P as the starting estimate. The goal is to prove the following theorem:

Theorem 4.1. Algorithm 1 run with Algorithm 2 as an aggregation routine outputs a (1 + ε ) -approximate Euclidean mean with probability 1 -δ if T ≥ γ for some absolute constant γ . The running time is O ( ε -1 · d log δ -1 ) .

The proof of Theorem 4.1 is articulated in two main steps. First, we show that, given a candidate estimate c j is very far from µ , the gradient of the geometric median of means evaluated at c j points in direction of µ . Second, we must determine a good updated solution along this gradient. Line-sweeps or second-order methods are all possible candidates, but computationally expensive. Instead, we perform and update by selecting the median-of-means along the gradient. Third, we show that once the algorithm determines a solution close to µ , the gradient updates will remain close thereafter.

Step 1: Fast convergence to close estimates from far ones. In order to estimate the mean quickly with gradient descent, we will make use of the following geometric median gradient properties whenever our current estimate c j is far from µ .

Lemma 4.2. Conditioned on event E , we have:

- (a) For any point q ∈ R d with ∥ q -µ ∥ &gt; 10 r , then ∥∇ ( q ) ∥ &gt; 3 10 · b log δ -1 ;
- (b) If, at iteration j , Algorithm 2 chooses a point c j such that ∥ c j -µ ∥ ≥ 10 r , then ∥ c j +1 -µ ∥ ≤ 7 10 · ∥ c j -µ ∥ .

Proof. We first show that every vector corresponding to a good mean is approximately in the same direction from q . Let ˆ µ be a good sample mean. Consider the triangle { q, µ, ˆ µ } and let α i and β i be the angle in q and ˆ µ respectively. We show an upper-bound for cos( α i ) . Using the law of sines,

<!-- formula-not-decoded -->

We get sin( α i ) ≤ ∥ µ -ˆ µ i ∥ ∥ q -µ ∥ ≤ 1 10 , which implies cos( α i ) ≥ √ 1 -sin 2 ( α i ) ≥ 0 . 99 . To bound ∥∇ ( q ) ∥ , we consider the contribution of good and bad means separately. We write ∇ ( q ) = ∑ i ∈ [ b log δ -1 ] ∇ i ( q ) = ∇ G ( q ) + ∇ B ( q ) , where ∇ G , ∇ B respectively denote good and bad points contributions to the gradient evaluated at q (see Figure 1).

<!-- formula-not-decoded -->

where the first inequality holds because projecting the gradient onto any direction only decreases the norm. Conversely, the contribution given by bad sample means to the gradient norm is at most

<!-- formula-not-decoded -->

where the inequality follows from the upper-bound on the number of bad means when E holds and the triangle inequality. Part (a) follows by combining both bounds: We have, due to the triangle inequality, an overall lower bound on the norm of the gradient of

<!-- formula-not-decoded -->

For part (b), similarly to the above, we focus on the triangle { q, q -∇ G , q -∇ G -∇ B } . Let β be the angle in vertex q (see Figure 1, with q instead of c j ) and let θ be the angle in vertex q -∇ G -∇ B . Using the bounds of ∥∇ G ∥ and ∥∇ B ∥ and the law of sines, we get

<!-- formula-not-decoded -->

which implies that sin( β ) ≤ ∥∇ B ∥ ∥∇ G ∥ ≤ 1 2 .

Next, we upper-bound ∥ c j +1 -µ ∥ to prove part (b). Consider the projection of µ onto c j -∇ G ( c j ) (the green line in Figure 1), call it y . Now, let us consider the projection of y onto c j -∇ ( c j ) , call it proj j ( y ) . Since the good samples are in the majority, the median c j +1 of the projected sample means (including both good and bad) lies within the convex hull of the good projected means. This holds because, under any projection, the median must be between two good points (analogously to the one-dimensional median-of-means estimator). This implies that, for c j +1 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the third inequality follows from the fact that ∥ c j -y ∥ ≤ ∥ c j -µ ∥ (given that c j -µ and c j -y are respectively the hypotenuse and the leg of the right-angle triangle { c j , µ, y } and ∥ c j -µ ∥ ≥ 10 r .

We wish to briefly remark on the sample complexity of the geometric median-of-means estimator and generalization bounds. Optimality follows from part (a) of Lemma 4.2, as if the gradient of the geometric median ν GM is 0 , ∥ ν GM -µ ∥ ≤ 10 r must hold.

Corollary 4.3. For sufficiently large absolute constants a and b , the geometric median ν GM := arg min c ∈ R d ∑ b log δ -1 i =1 ∥ ˆ µ i -c ∥ of b log δ -1 many sample means consisting of aε -1 many points is a

(1 + ε ) -approximate mean with probability at least 1 -δ .

Remark 4.4. The geometric median cannot be computed exactly, we can merely approximate it. The quality of the choice of approximation is important here. Using event E by itself, it is not clear that even a 2 -approximate geometric median will recover a (1 + O ( ε )) -approximate mean.

Indeed, if we assume that a single ˆ µ i has distance √ ε OPT δn to the optimal mean, we must compute a (1 + δ -O (1) ) -approximate median. In Appendix E, we will show that this assumption is warranted for worst case instances. The fastest algorithm for finding a (1 + γ ) -approximate geometric median by Cohen et al. [2016] runs in time O ( nd log 3 γ -1 ) . In our setting, n = b log δ -1 . The resulting running time of O (log 4 δ -1 · d ) is thus substantially slower that the other three algorithms presented in this paper.

## Step 2: Once the estimate is close, it remains close.

Lemma 4.5. Conditioned on event E , if, at iteration j , Algorithm 2 chooses a point c j such that ∥ c j -µ ∥ ≤ 10 r , then ∥ c j +1 -µ ∥ ≤ 11 r .

Proof. Consider a point c j chosen by Algorithm 2 such that ∥ c j -µ ∥ ≤ 10 r . Its gradient vector is ∇ ( c j ) and Algorithm 2 projects all other sample means onto c j -∇ ( c j ) and takes the median to be c j +1 . For every mean ˆ µ , we denote its projection by proj (ˆ µ ) . We make two observations. First, the set of proj (ˆ µ ) of all good points lie on a bounded line segment of length at most 2 r (see Figure 1). Second, due to the event E , we have that the median chosen lies within the bounded line segment.

From the second observation, we have ∥ c j +1 -proj ( µ ) ∥ ≤ r , which is the radius of the good ball (see Figure 1). Applying triangle inequality, we get

<!-- formula-not-decoded -->

where the second inequality follows from proj ( µ ) being a projection on a line passing through c j .

Proof of Theorem 4.1. We condition on event E . In Algorithm 2, the initial value ν CWM is taken to be the coordinate-wise median of all the b log δ -1 sample means. From Theorem 3.1, we know that ∥ ν CWM -µ ∥ ≤ √ 5 r . Using part (b) of Lemma 4.2, we have that as long as ∥ c j -µ ∥ ≥ 10 r , it holds that ∥ c j +1 -µ ∥ ≤ 7 10 · ∥ c j -µ ∥ . As the quantity ∥ c j -µ ∥ is decreasing exponentially, it follows that for some absolute constant γ there exists an iteration j ≤ γ , such that ∥ c j -µ ∥ ≤ 10 r . In the next

Figure 1: Good empirical means are represented in green and bad ones in red. The ball of good means centered at µ has radius r . Projection of all the good means lie on the bounded line segment of length at most 2 r .

<!-- image -->

iterations of the algorithm, from Lemma 4.5, we know that for the terminating value of the algorithm, c T satisfies ∥ c T -µ ∥ ≤ 11 r , which yields the desired approximation ratio.

Concerning the the running time, we first need to compute b log δ -1 sample means from aε -1 many uniform samples, and then, in each of the γ iterations, there are b log δ -1 sample means to project onto the gradient vector, an operation that takes O ( d log δ -1 ) time, with d being the dimension of the ambient space. To compute the median of all projections in every iteration, as well as finding the initial coordinate-wise median ν CWM can be done using a linear-time rank select procedure (see [Cormen et al., 2009, Chapter 9.3]). For each iteration, as well as the initialization, this therefore takes O ( d log δ -1 ) time. Thus, the overall running time is O ( ε -1 · d log δ -1 ) .

## 5 Experimental Evaluation

We conclude this paper with a short experimental evaluation. Code base and results can be found at https://github.com/matteorusso/sublinear\_mean\_estimation . The experiments were carried out on the Google Colab default CPU. Our goal is to understand the practical viability of various estimators, such as the three optimal estimators included here, as well as the previous results from Cohen-Addad et al. [2021], Woodruff and Yasuda [2024] and the empirical mean of the samples. In particular, we wish to understand if the gradient updates from Algorithm 2 improve over an initialization given by the coordinate-wise median. It is not difficult to generate instances where one of these algorithms outperforms the other, but ultimately the empirically best estimator can only be determined via experiments on real-world data.

Algorithms. The reference algorithms for fast sublinear mean estimation is the 1 -means coresetbased CSS algorithm Cohen-Addad et al. [2021], the active-regression coreset-based WY algorithm Woodruff and Yasuda [2024], and our Algorithms 2 and 3. In addition, we also used the standard empirical mean as a baseline. Each of these algorithms are given a set of m points sampled uniformly at random from the underlying point set and compared with respect to running time and quality of the solution.

We implement both MINSUMSELECTand FASTGD. As described in Section 4, we take as initial guess the coordinate-wise median of the computed sample means. We also simply compare against the coordinate-wise median initialization step of FASTGD, referred to as COORDWISEMEDIAN.

Data Sets, Setup and Results. We test our algorithms against the benchmarks mentioned above on the following datasets: MNIST and Fashion-MNIST, both of which are composed of 60,000 points, each with 784 features. We also consider CoverType, composed of 581,012 points, each with 54 features. We describe the gist of the results here, more details such as numerical figures are given in Appendix G. For each sample size m ∈ { 10 , 15 , 20 , 25 , 30 , 100 , 200 , 500 , 1000 , 2000 , 5000 , 10000 } , we repeat the execution of every algorithm 50 times and report averages and variances across runs. We plot the accuracy (i.e., ALG / OPT), and runing time as a function of the sample size for the various algorithms. Due to space constraints, we only report results on MNIST in this section (see Figure 2). The behaviour of accuracy and runtime as a function of sample size is similar across all datasets.

Figure 2: MNIST Dataset: Accuracy and runtime against sample size.

<!-- image -->

Our observations from the experimental evaluation on real-world datasets provide a few insights on the proposed methods. Among all algorithms, MINSUMSELECT and COORDWISEMEDIAN emerge as the fastest, consistently achieving superior runtime performance across all datasets. Their efficiency highlights their suitability for large-scale applications where computational speed is critical. In terms of accuracy, both MINSUMSELECT, COORDWISEMEDIAN and FASTGD perform comparably, with FASTGD showing a slight edge in most cases. Notably, FASTGD almost never underperforms relative to COORDWISEMEDIAN, despite both offering the same theoretical guarantees. This suggests that, in practice, applying a few gradient-based refinement steps toward the geometric median, starting from the coordinate-wise median, can yield measurable improvements.

Interestingly, despite having the weakest theoretical guarantees, the empirical mean consistently ranks among the best algorithms in terms of accuracy. Since it can also be computed extremely quickly via the closed form, it is also by far the fastest among the candidate algorithms. While there exist examples where the empirical mean does not provide a good and robust estimation, these examples do not seem to appear on any of the data sets we tried and may not be common in practice, see the appendix for a more detailed discussion. Nevertheless, the empirical mean is not consistently the best estimator, which goes to the CSS-estimator, which also requires extensive filtering and preprocessing and is among the slowest available methods. The WY algorithm performs worse in terms of accuracy. While it is not known whether the stated sample complexity upper bounds for CSS and WY are tight, these observation may indicate that these algorithms are not tightly analyzed. Additionally, they are considerably slower than our algorithms, regardless of sample size, due to inherent overhead in pre- and postprocessing steps. However, optimization is very fast for these algorithms because they simply output the mean.

In summary, the algorithms proposed in this paper are always competitive in terms of accuracy with the other methods, at least when given a moderately large sample size, while being among the fastest methods available. While the refinement via FASTGD can offer improvements in practice, the best candidate algorithms seem to be either COORDWISEMEDIAN or taking the empirical mean.

Figure 3: Fashion-MNIST Dataset: Accuracy and runtime against sample size.

<!-- image -->

## Acknowledgements

We thank the anonymous reviewer from a previous version of this paper for pointing an improvement of our analysis of the coordinate-wise median-of-means estimator. Matteo Russo was partially supported by the FAIR (Future Artificial Intelligence Research) project PE0000013, the NextGenerationEU program within the PNRR-PE-AI scheme (M4C2, investment 1.3, line on Artificial Intelligence), the PNRR MUR project IR0000013-SoBigData.it, and by the MUR PRIN grant 2022EKNE5K (Learning in Markets and Society). Chris Schwiegelshohn was supported by a Google Research Award and by the Independent Research Fund Denmark (DFF) under a Sapere Aude Research Leader grant No 1051-00106B. Sudarshan Shyam was partially supported by the Independent Research Fund Denmark (DFF) under grant 2032-00185B.

## References

- P. Afshani and C. Schwiegelshohn. Optimal coresets for low-dimensional geometric median. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024 . OpenReview.net, 2024. URL https://openreview.net/forum?id=8iWDWQKxJ1 .
- N. Bansal, V. Cohen-Addad, M. Prabhu, D. Saulpic, and C. Schwiegelshohn. Sensitivity sampling for k-means: Worst case and stability optimal coreset bounds. In 65th IEEE Annual Symposium on Foundations of Computer Science, FOCS 2024, Chicago, IL, USA, October 27-30, 2024 , pages 1707-1723. IEEE, 2024. doi: 10.1109/FOCS61266.2024.00106. URL https://doi.org/10. 1109/FOCS61266.2024.00106 .
- P. Bartlett, T. Linder, and G. Lugosi. The minimax distortion redundancy in empirical quantizer design. IEEE Transactions on Information Theory , 44(5):1802-1813, 1998. doi: 10.1109/18.705560.
- V. Braverman, S. H. Jiang, R. Krauthgamer, and X. Wu. Coresets for clustering in excluded-minor graphs and beyond. In D. Marx, editor, Proceedings of the 2021 ACM-SIAM Symposium on Discrete Algorithms, SODA 2021, Virtual Conference, January 10 - 13, 2021 , pages 2679-2696. SIAM, 2021.
- V. Braverman, V. Cohen-Addad, S. H. Jiang, R. Krauthgamer, C. Schwiegelshohn, M. B. Toftrup, and X. Wu. The power of uniform sampling for coresets. In 63rd IEEE Annual Symposium on Foundations of Computer Science, FOCS 2022, Denver, CO, USA, October 31 - November 3, 2022 , pages 462-473. IEEE, 2022. doi: 10.1109/FOCS54457.2022.00051. URL https: //doi.org/10.1109/FOCS54457.2022.00051 .
- M. S. Bucarelli, M. F. Larsen, C. Schwiegelshohn, and M. Toftrup. On generalization bounds for projective clustering. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023 , 2023.
- O. Catoni. Challenging the empirical mean and empirical variance: a deviation study. Ann. Inst. Henri Poincaré Probab. Stat. , 48(4):1148-1185, 2012. ISSN 0246-0203,1778-7017. doi: 10.1214/ 11-AIHP454. URL https://doi.org/10.1214/11-AIHP454 .
- X. Chen and M. Derezinski. Query complexity of least absolute deviation regression via robust uniform convergence. In M. Belkin and S. Kpotufe, editors, Conference on Learning Theory, COLT 2021, 15-19 August 2021, Boulder, Colorado, USA , volume 134 of Proceedings of Machine Learning Research , pages 1144-1179. PMLR, 2021.
- K. L. Clarkson, E. Hazan, and D. P. Woodruff. Sublinear optimization for machine learning. J. ACM , 59(5):23:1-23:49, 2012.
- S. Clemençcon. On u-processes and clustering performance. In J. Shawe-Taylor, R. Zemel, P. Bartlett, F. Pereira, and K. Weinberger, editors, Advances in Neural Information Processing Systems , volume 24. Curran Associates, Inc., 2011.

- M. B. Cohen, Y. T. Lee, G. L. Miller, J. Pachocki, and A. Sidford. Geometric median in nearly linear time. In D. Wichs and Y. Mansour, editors, Proceedings of the 48th Annual ACM SIGACT Symposium on Theory of Computing, STOC 2016, Cambridge, MA, USA, June 18-21, 2016 , pages 9-21. ACM, 2016.
- V. Cohen-Addad, D. Saulpic, and C. Schwiegelshohn. Improved coresets and sublinear algorithms for power means in euclidean spaces. In M. Ranzato, A. Beygelzimer, Y. N. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual , pages 21085-21098, 2021.
- V. Cohen-Addad, S. Lattanzi, and C. Schwiegelshohn. Almost optimal PAC learning for k-means. In M. Koucký and N. Bansal, editors, Proceedings of the 57th Annual ACM Symposium on Theory of Computing, STOC 2025, Prague, Czechia, June 23-27, 2025 , pages 2019-2030. ACM, 2025. doi: 10.1145/3717823.3718180. URL https://doi.org/10.1145/3717823.3718180 .
- T. H. Cormen, C. E. Leiserson, R. L. Rivest, and C. Stein. Introduction to Algorithms, 3rd Edition . MIT Press, 2009.
- T. Dang, J. C. H. Lee, M. R. Song, and P. Valiant. Optimality in mean estimation: Beyond worst-case, beyond sub-gaussian, and beyond 1+ α moments. In NeurIPS , 2023.
- H. Ding. A sub-linear time framework for geometric optimization with outliers in high dimensions. In 28th Annual European Symposium on Algorithms, ESA 2020, September 7-9, 2020, Pisa, Italy (Virtual Conference) , pages 38:1-38:21, 2020.
- C. Fefferman, S. Mitter, and H. Narayanan. Testing the manifold hypothesis. Journal of the American Mathematical Society , 29(4):983-1049, 2016.
- D. Feldman and M. Langberg. A unified framework for approximating and clustering data. In Proceedings of the 43rd ACM Symposium on Theory of Computing, STOC 2011, San Jose, CA, USA, 6-8 June 2011 , pages 569-578, 2011.
- L. Huang, R. Huang, Z. Huang, and X. Wu. On coresets for clustering in small dimensional euclidean spaces. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA , volume 202 of Proceedings of Machine Learning Research , pages 13891-13915. PMLR, 2023. URL https://proceedings.mlr.press/v202/huang23h.html .
- M. Inaba, N. Katoh, and H. Imai. Applications of weighted voronoi diagrams and randomization to variance-based k -clustering (extended abstract). In Proceedings of the Tenth Annual Symposium on Computational Geometry, Stony Brook, New York, USA, June 6-8, 1994 , pages 332-339, 1994.
- Y. Klochkov, A. Kroshnin, and N. Zhivotovskiy. Robust k-means clustering for distributions with two moments. The Annals of Statistics , 49(4):2206 - 2230, 2021.
- M. Langberg and L. J. Schulman. Universal ε -approximators for integrals. In Proceedings of the Twenty-First Annual ACM-SIAM Symposium on Discrete Algorithms, SODA 2010, Austin, Texas, USA, January 17-19, 2010 , pages 598-607, 2010. doi: 10.1137/1.9781611973075.50. URL http://dx.doi.org/10.1137/1.9781611973075.50 .
- J. C. H. Lee and P. Valiant. Optimal sub-gaussian mean estimation in very high dimensions. In M. Braverman, editor, 13th Innovations in Theoretical Computer Science Conference, ITCS 2022, January 31 - February 3, 2022, Berkeley, CA, USA , volume 215 of LIPIcs , pages 98:1-98:21. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2022. doi: 10.4230/LIPICS.ITCS.2022.98. URL https://doi.org/10.4230/LIPIcs.ITCS.2022.98 .
- T. Linder. On the training distortion of vector quantizers. IEEE Transactions on Information Theory , 46(4):1617-1623, 2000. doi: 10.1109/18.850705.
- G. Lugosi and S. Mendelson. Mean estimation and regression under heavy-tailed distributions: A survey. Found. Comput. Math. , 19(5):1145-1190, 2019.

- G. Lugosi and S. Mendelson. Multivariate mean estimation with direction-dependent accuracy. J. Eur. Math. Soc. (JEMS) , 26(6):2211-2247, 2024. ISSN 1435-9855,1435-9863. doi: 10.4171/jems/1321. URL https://doi.org/10.4171/jems/1321 .
- A. Maalouf, I. Jubran, and D. Feldman. Introduction to coresets: Approximated mean. CoRR , abs/2111.03046, 2021.
- A. Maalouf, I. Jubran, and D. Feldman. Fast and accurate least-mean-squares solvers for high dimensional data. IEEE Trans. Pattern Anal. Mach. Intell. , 44(12):9977-9994, 2022.
- S. Minsker. Geometric median and robust estimation in banach spaces. Bernoulli , 21(4):2308-2335, 2015. doi: 10.3150/14-BEJ645. URL https://www.jstor.org/stable/43590532 .
- C. Musco, C. Musco, D. P. Woodruff, and T. Yasuda. Active linear regression for ℓ p norms and beyond. In 2022 IEEE 63rd Annual Symposium on Foundations of Computer Science (FOCS) , pages 744-753, 2022.
- H. Narayanan and S. K. Mitter. Sample complexity of testing the manifold hypothesis. In J. D. Lafferty, C. K. I. Williams, J. Shawe-Taylor, R. S. Zemel, and A. Culotta, editors, Advances in Neural Information Processing Systems 23: 24th Annual Conference on Neural Information Processing Systems 2010. Proceedings of a meeting held 6-9 December 2010, Vancouver, British Columbia, Canada , pages 1786-1794. Curran Associates, Inc., 2010.
- A. Parulekar, A. Parulekar, and E. Price. L1 Regression with Lewis Weights Subsampling. In M. Wootters and L. Sanità, editors, Approximation, Randomization, and Combinatorial Optimization. Algorithms and Techniques (APPROX/RANDOM 2021) , volume 207 of Leibniz International Proceedings in Informatics (LIPIcs) , pages 49:1-49:21, Dagstuhl, Germany, 2021. Schloss Dagstuhl - Leibniz-Zentrum für Informatik. ISBN 978-3-95977-207-5.
- D. Pollard. A Central Limit Theorem for k -Means Clustering. The Annals of Probability , 10(4):919 926.
- D. P. Woodruff and T. Yasuda. Coresets for multiple ℓ p regression. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024 , 2024.

## A Preliminaries

Lemma A.1 (High Dimensional Mean-Variance Decomposition) . For any set of points A ⊂ R d and any c ∈ R d , we have ∑ p ∈ A ∥ p -c ∥ 2 = ∑ p ∈ A ∥ p -µ ∥ 2 + n · ∥ µ -c ∥ 2 .

Proof. We have

<!-- formula-not-decoded -->

where the last equality follows from ∑ p ∈ A ( p -µ ) = 0 .

## B Mean Estimation via Order Statistics

In this section, we leverage order statistics of the candidate means so as to allow for a quicker aggregation of a suitable candidate mean.

## Algorithm 3 MINSUMSELECT ( P, i )

Input: Set of points p 1 , . . . p | P | , recursion depth i if i = 0 or | P | = 1 then W ← P else Split P arbitrarily into √ | P | -sized clusters { P 1 , . . . , P √ | P | } W ←∅ for each P j do W ← W ∪ { MINSUMSELECT ( P j , i -1) } Output COMPUTEWINNER ( W )

## Algorithm 4 COMPUTEWINNER( P )

for each p j ∈ P do Let p ′ j ∈ P be the 7 10 | P | -closest point to p j Compute D j := ∑ p ∈ P, ∥ p -p j ∥≤∥ p ′ j -p j ∥ ∥ p -p j ∥ Output arg min p j ∈ P D j

The main result is the following:

Theorem B.1. Algorithm 1 run with Algorithm 3 as an aggregation routine outputs a (1 + ε ) approximation with probability 1 -δ , using a sample of size O ( 100 i ε -1 log δ -1 ) , and running in time

<!-- formula-not-decoded -->

for any non-negative integer i .

The key observation is that a good mean can always be identified via COMPUTEWINNER as an estimate ˆ µ j minimizing the sum of distances of the 7 10 th means closest to ˆ µ j . In a nutshell, any mean minimizing such a sum must be close to sufficiently many successful estimates. Unfortunately, a naive implementation of this idea takes time O (log 2 δ -1 · d ) , as proven in the following lemma.

Lemma B.2. COMPUTEWINNER( P ) takes time O ( | P | 2 · d ) .

Proof. We compute all pairwise distances between the points in P , which takes time O ( | P | 2 · d ) . To compute D j , we first have to find the 7 10 | P | -closest point to p j , which takes time O ( | P | ) with a sufficiently good rank select procedure, see [Cormen et al., 2009, Chapter 9.3]. Thereafter, summing up all the distances takes time O ( | P | ) per p j ∈ P .

Nevertheless, these scores yield an improved running time by arbitrarily partitioning the estimates into √ | P | groups, finding a good estimate in every group via the truncated sum statistic each in time | P | · d , for a total running time of | P | 3 / 2 · d , and then selecting the best estimate via another truncated sum statistic on all estimates returned by the groups in time | P | . The final algorithm consists of applying this idea recursively.

To prove that MINSUMSELECT outputs a good solution, we require a parameterized notion of a successful empirical mean. We say that a mean µ j is γ -good, if ∥ µ j -µ ∥ ≤ γ √ ε OPT n . A straightforward application of the Chernoff bound guarantees us that all but a very small fraction of the points are γ -good. Assuming this, the following lemma determines the quality of the computed solution.

Lemma B.3. Let P be a set of means such that at least ( 1 -( 3 10 ) i +1 ) of the means are γ -good. Then MINSUMSELECT ( P, i ) returns a mean that is 5 i +1 γ -good.

We prove this lemma by induction. We will use the following lemma in both the base case and the inductive step.

Lemma B.4. Given a set of means P , suppose that at least 7 10 | P | are γ -good. Then, the estimate returned by COMPUTEWINNER( P ) is 5 γ -good.

Proof. First, let ˆ µ ′ ∈ P be any γ -good estimator. We know that the 7 10 | P | -closest estimators to ˆ µ ′ have distance at most 2 γ √ ε OPT n , which likewise implies that min µ j ∈ P D j ≤ 7 10 | P | · 2 γ √ ε OPT n . Now suppose that ˆ µ j is the estimator returned by the algorithm. Let G (ˆ µ j ) be the set of γ -good estimators among the 7 10 | P | closest to ˆ µ j . By assumption, we have | G (ˆ µ j ) | ≥ 4 10 | P | , which implies

<!-- formula-not-decoded -->

which gives

Therefore,

<!-- formula-not-decoded -->

Proof of Lemma B.3. We proceed with the induction starting from i = 0 .

Base Case. For the base case i = 0 , MINSUMSELECT( P , 0 ) only calls COMPUTEWINNER. Thus, this case holds due to Lemma B.4.

Inductive Step. Let P 1 , . . . P √ | P | be the clusters of P computed when first calling MINSUMSELECT ( P, i ) . By assumption, we have at most ( 3 10 ) i +1 | P | means that are not γ -good. This implies that the number of clusters with more than ( 3 10 ) i √ | P | means that are not γ -good is at most

<!-- formula-not-decoded -->

Denote this set by B ( P ) and let G ( P ) be the remaining clusters. For each P j ∈ B ( P ) ∪ G ( P ) , let ˆ µ j returned by MINSUMSELECT( P j , i -1 ). If P j ∈ G ( P ) , we may use the inductive hypothesis

<!-- formula-not-decoded -->

which states that the mean ˆ µ j returned by MINSUMSELECT( P j , i -1 ) is 5 i γ -good. Since at least 7 10 √ | P | of the thus computed means are 5 i γ -good, we may use Lemma B.4 which shows that the final mean returned by COMPUTEWINNER ( ∪ P j ∈ B ( P ) ∪ G ( P ) { ˆ µ j } ) is 5 i +1 γ -good, which concludes the proof.

To conclude the proof, we require two more arguments. First, we must show that, for an appropriate choice of a and b , at least a ( 1 -( 3 10 ) i +1 ) fraction of the means are 1 -good with probability at least 1 -δ , allowing us to use Lemma B.3. Second, we will argue the running time. The first is a simple application of the Chernoff bound.

Lemma B.5. For a ≥ 2 · 25 i +1 · ( 10 3 ) i +1 and b = 3 , at least ( 1 -( 3 10 ) i +1 ) of the means are 5 -( i +1) -good with probability 1 -δ .

<!-- formula-not-decoded -->

2.1. This implies due to Markov's inequality that P [ X i = 1] ≤ 25 i +1 a which, by our choice of a , is less than 1 2 · ( 3 10 ) i +1 . Thus, by the Chernoff bound

<!-- formula-not-decoded -->

which is at most δ by our choice of b .

The approximation guarantee now follows from Lemma B.3 and Lemma B.5. What remains to be shown is the running time. Since we split a collection of t means into √ t clusters, the running time consists of the time required to recursively run the algorithm on the √ t clusters each of size √ t and consolidating via COMPUTEWINNER, which takes O ( t · d ) time. Thus, starting with an instance of size | P 0 | ∈ O (log δ -1 ) , we can solve for the recursion

<!-- formula-not-decoded -->

which yields the desired running time O ( i · d log δ -1 +(log δ -1 ) 1+2 -i · d ) = O ((log δ -1 ) 1+2 -i · d ) . We next formalize this and complete the proof of Theorem B.1.

Proof of Theorem B.1. Throughout this proof, assume a ≥ 2 · 25 i +1 ( 10 3 ) i +1 and b ≥ 3 .

We first argue correctness, then running time. Let P denote the entire set of means passed to the MINSUMSELECT algorithm. By Lemma B.3 and Lemma B.5 and with our choices of a and b , MINSUMSELECT ( P, i ) returns a 1 -good mean, that is a (1 + ε ) -approximate mean with probability at least 1 -δ .

What remains to be shown is the running time. Denote by | P | 0 = b log δ -1 the initial set of sample means. MINSUMSELECT ( P, i ) computes √ | P | children, each of which recursively calls MINSUMSELECT. Consolidating via COMPUTEWINNER takes time O ( √ | P | 2 · d ) = O ( | P | · d ) due to Lemma B.2. Thus the overall recursion takes time

<!-- formula-not-decoded -->

which solves to a running time of O ( i · | P | 0 · d + | P | 1+2 -i 0 · d ) = O (( b log δ -1 ) 1+2 -i · d ) . The time to compute the initial set of means is O ( aε -1 · bd log δ -1 ) , which with our choice of a and b , and some overestimation of the constants, leads to an overall running time of O (( 100 i ε -1 + ( log δ -1 ) 2 -i ) · d log δ -1 ) .

Remark B.6. For any constant choice of recursion depth, the sample complexity only increases by constants. We did not attempt to optimize the constants, but the exponential dependency on i is not avoidable. If we were to prioritize the running time over the sample complexity, we can set a recursion depth of i = log log log δ -1 , which achieves a running time and sample complexity of O ( ε -1 log δ -1 poly ( log log δ -1 ) · d ) .

Furthermore, if the recursion depth is shallow (i.e., for small values of i ), the recursion can be improved via a better choice of cluster size. Specifically, in the case i = 1 , that is with just one set of children, Theorem B.1 yields a running time of O ( ( ε -1 + √ log δ -1 ) · d log δ -1 ) . If we instead choose ( log δ -1 ) 2 / 3 many clusters, each consisting of 3 √ log δ -1 many estimators, we obtain a running time of O ( ( ε -1 + 3 √ log δ -1 ) · d log δ -1 ) . Similar improvements for other values i are also possible, but these improvements become increasingly irrelevant compared to the bounds given in Theorem B.1 as i gets larger.

## C Learning the Mean of a High-Dimensional Distribution

Theorem 3.2. Let X 1 , X 2 , . . . , X N be independent samples from distribution D with variance µ and covariance matrix Σ , then the coordinate-wise median-of-means estimator ν CWM , with probability at least 1 -δ , satisfies

<!-- formula-not-decoded -->

Proof. The N samples are partitioned into 8 log δ -1 subsamples of size N 8 log δ -1 each. The algorithm computes the empirical mean of each subsample and returns the coordinate-wise median of the set of empirical means.

Weprove the analog of Lemma 2.3 for this case, showing that a large fraction of sample means lie close to the true mean. For an empirical mean of N 8 log δ -1 samples, we have E [ ∥ ˆ µ -µ ∥ 2 ] = 8 TR (Σ) log δ -1 N .

We call an empirical mean good if ∥ ˆ µ -µ ∥ ≤ r where r = 13 √ TR (Σ) log δ -1 N and let G denotes the set of good means. Using TR (Σ) = E [ ∥ X -µ ∥ 2 ] and the Markov's inequality, we have P [ˆ µ ∈ G ] ≥ 1 -8 13 2 ≥ 0 . 95 . Applying the Chernoff bound, we get that with probability at least 1 -δ , we have | G | ≥ 7 10 log δ -1 .

Consider ν CWM, the coordinate-wise median of all the sample means. For each coordinate k , let L k and R k be the sets of sample means such that ˆ µ i,k ≤ ν CWM ,k and ˆ µ i,k ≥ ν CWM ,k respectively. We know that L k , R k have cardinality at least b log δ -1 2 . Depending on whether ν CWM ,k &gt; µ k or not, at least for one of L k , R k , we have that | ˆ µ i,k -µ k | ≥ | ν CWM ,k -µ k | .

We know that at least 7 10 fraction of the means are good (i.e., ∥ ˆ µ i -µ ∥ ≤ r ) with probability at least 1 -δ . Hence, we infer that at least 7 10 -1 2 = 1 5 of the sample means are good and satisfy | ˆ µ i,k -µ k | ≥ | ν CWM ,k -µ k | . On average,

<!-- formula-not-decoded -->

Summing over all coordinates k gives

<!-- formula-not-decoded -->

Interchanging the sums on the left-hand side, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## D Generalization Bounds

We place our results in the context of generalization bounds for clustering problems. In this setting, we are given an arbitrary but fixed distribution D supported on the unit Euclidean ball B d 2 . The cost of a point c with respect to D is defined as cost D ( c ) = ∫ p ∈ B d 2 ∥ p -c ∥ 2 · P [ p ] dp . The risk is defined as R := argmin c cost D ( c ) . Given independent samples S drawn from D , we wish to compute an estimate ˆ c with cost ˆ R = cost D (ˆ c ) such that the excess risk ˆ R-R is minimized. The cost of a distribution D is in a limiting sense the average cost of a point set with all points living in B d 2 . Since the average cost is at most the squared radius (i.e. 1 ), obtaining a (1 + ε ) approximate solution also yields a solution that has excess risk of at most ε OPT n ≤ ε , which we then rewrite in terms of the sample size | S | as O ( log δ -1 | S | ) . This discussion is summarized in the following corollary.

Corollary D.1. Given a set of independent samples S drawn from some underlying arbitrary but fixed distribution supported on B d 2 , the geometric median-of-means estimator has an excess risk for the least squared distances objective of

<!-- formula-not-decoded -->

with probability 1 -δ for some absolute constant γ &gt; 0 .

Note that the learning rate given by Corollary D.1 stands in contrast to learning rates that are achievable for other center-based problems such as indeed the geometric median with objective

<!-- formula-not-decoded -->

or for the k -means problem with objective

<!-- formula-not-decoded -->

for a k -center set C . As mentioned in the related work section, all of these objectives require learning rates of at least √ 1 | S | , ignoring problem specific parameters.

## E Lower Bounds

We complement our algorithmic results with matching lower bounds.

## E.1 A High Probability Lower Bound for Mean-Estimation

Theorem E.1. Any sublinear algorithm that outputs a (1 + ε ) -approximate Euclidean mean with probability at least 1 -δ must sample at least Ω( ε -1 log δ -1 ) many points.

The idea is to generate two instances that a sublinear algorithm for approximating means has to distinguish between and then bound the number of samples required to distinguish between such distributions. The first instance is virtually identical to the one used by Cohen-Addad et al. [2021]. It places n points at 0 and εn points at 1 . Thus, the optimal mean is ε 1+ ε . A routine calculation via Lemma 2.2 shows that the optimal cost is εn 1+ ε . The second instance places all points at 0 . Since 0 is not a sufficiently good approximate mean for the first instance, a sublinear algorithm can distinguish between the two instances, which yields a lower bound on the sample complexity.

The two instances lie within the unit Euclidean ball. By normalizing the number of points, it also yields a distribution supported on the unit Euclidean ball, which implies that the generalization bounds given in Corollary D.1 are sharp.

Proof. We give two instances. The first instance places n points at 0 and εn points at 1 . Thus, the optimal mean is placed at ε 1+ ε . A routine calculation via Lemma 2.2 shows that the optimal cost is OPT = εn 1+ ε . The second instance places all points at 0 .

First we argue that a sublinear algorithm can distinguish between these two instances. Any approximate mean for the second instance must output 0 . If we output 0 for the first instance, we incur a cost of εn = εn (1+ ε ) 1+ ε = OPT + ε · OPT, hence any algorithm improving over a (1 + ε ) approximation for the former cannot output 0 . Thus, a necessary condition to distinguish between the two instances is for the algorithm to sample at least one point at 1 .

Suppose we sample m points. Our goal is to show that for m ∈ Ω( ε -1 log δ -1 ) , the probability that all of the sampled points are drawn from 0 is less than δ for the first instance. Let X i denote the indicator variable of the i th sampled point. We have

<!-- formula-not-decoded -->

where the final inequality follows from the Mercator series ln(1 + ε ) = ∑ ∞ i =1 ( ε i i ) · ( -1) i +1 . Thus, for any m &lt; ε -1 · log δ -1 , we do not achieve the desired failure probability. Conversely, m ∈ Ω( ε -1 log δ -1 ) for an algorithm to succeed.

## E.2 The Empirical Mean is not a Good Estimator

We briefly show that the arguably most natural algorithm that outputs the empirical mean of a subsampled point set has a substantial increase in sample complexity in the high success probability regime and therefore also a substantial increase in running time compared to the algorithms presented in this paper. The result is folklore, but included for completeness.

Theorem E.2. For all ε, δ ∈ (0 , 1) , there exists an instance such that Ω( ε -1 δ -1 ) independently sampled points are required for the empirical mean to be a (1 + ε ) -approximate Euclidean mean with probability at least 1 -δ .

The proof is based on reducing the problem of finding a good approximate mean with high probability to giving an example distribution for which the Chebychev inequality is tight.

Proof. Consider a sample | S | , with | S | being specified later. We generate an instance as follows. We place a 1 2 | S | 2 ε -fraction of the points each at -| S | √ ε and | S | √ ε and the remaining 1 -1 | S | 2 ε fraction at 0 . Then the optimal solution places the mean at 0 and has average cost ( | S | √ ε ) 2 · 1 | S | 2 ε = 1 . By Lemma 2.2, this implies that the empirical mean ˆ µ = 1 | S | ∑ p ∈ S p is a better than (1+ ε ) -approximate mean if and only if | ˆ µ | &lt; √ ε . We set the failure probability, that is the probability P [ | ˆ µ | ≥ √ ε ] = δ . Then P [ | ˆ µ | ≥ √ ε ] is at least the probability that we sample exactly one point from either -| S | √ ε or | S | √ ε and the remaining | S | -1 points from 0 . Using Bernoulli's inequality and the density function of the binomial distribution, we therefore have

<!-- formula-not-decoded -->

Although the empirical mean is the fastest among all mean estimators (it can be computed in closed form), the above construction shows that it is not a robust estimator of the true mean-even in one dimension. Indeed, for heavy-tailed distributions, the best available tail bound is Chebyshev's inequality, which is exponentially weaker. In fact, Catoni [2012] (similarly to the construction above) shows that there exist distributions for which Chebyshev's inequality is tight. In contrast, the empirical mean performs well when estimating the mean of subgaussian distributions, where the associated tail bounds are sharply exponential. In practice, data distributions are often not as adversarial as those constructed to demonstrate the failure of the empirical mean (e.g., Catoni [2012]). This explains why the empirical mean is frequently observed to perform well in real-world scenarios, both in terms of computational efficiency and estimation accuracy.

## F Comparison of Geometric Median and the Coordinate-wise median

In this section, we give two instances which demonstrate that neither of the coordinate-wise median or the geometric median is strictly better than the other for the problem of mean-estimation.

Proposition F.1. Let δ &gt; 0 be fixed. There exist instances such that, with probability at 1 -O ( δ ) ,

- (1) The coordinate-wise median of K = log δ -1 2 ( 1 2 -e -1 ) 2 sample means of ε -1 sampled points coincides with the true mean, but the geometric median of the same K sample means does not;
- (2) The geometric median of log δ -1 sample means of ε -1 sampled points is a better approximation to the true mean than the coordinate-wise median is.

Proof. For part (1), we construct an instance for which the coordinate-wise median is a better estimator than the geometric median. Consider the uniform distribution on the d -dimensional simplex where d = O ( ε -2 δ -2 log 2 δ -1 ) . The value of d ensures that with probability at least 1 -δ , each of the ε -1 log δ -1 vertices sampled are distinct. We show that in this case the coordinatewise median is a better aggregation procedure than the geometric median. The coordinate median ν CWM at the origin, while the true mean is µ = (1 /d, 1 /d, . . . , 1 /d ) . The geometric median can be shown to lie at the empirical mean of all the samples due to symmetry, which is µ GM = ( ε/ log ( δ -1 ) , ε/ log ( δ -1 ) , . . . , 0 , . . . , 0) where there are exactly log ( δ -1 ) /ε non-zero coordinates. Comparing the distance, we see that ∥ µ -ν CWM ∥ ≤ ∥ µ -µ GM ∥ . To conclude, we note that with probability at least 1 -δ , the coordinate-wise median is a better estimator than the geometric median.

For part (2), we wish to prove that there exists an instance in which the geometric median is better than the coordinate-wise median. Consider the uniform distribution on the d -dimensional simplex, where d = cε -1 . We choose c to be a constant large enough so that in every subsample, the probability of a vertex sampled is 1 3 (we only require it to be bounded away from 1 2 ). As in the last example, we have that the actual mean is (1 /d, 1 /d, . . . , 1 /d ) . We observe that the coordinate-wise median ν CWM is the origin with high probability. While as δ - → 0 , the geometric median tends towards true mean µ .

## G Further Experimental Evaluation

We present here further experimental evaluation and specifically we plot the accuracy vs. sample size and runtime vs. sample size, as well as the accuracy vs. dimension and runtime vs. dimension.

We first describe some summary statistics for the datasets we consider:

Table 1: Summary statistics for datasets.

| Dataset       | Shape         |   Mean |   Std-Dev |   Min |   Max |
|---------------|---------------|--------|-----------|-------|-------|
| MNIST         | (60,000; 784) | 0.1306 |    0.3081 |     0 |     1 |
| Fashion-MNIST | (60,000; 784) | 0.286  |    0.353  |     0 |     1 |
| CoverType     | (581,012; 54) | 0.4567 |    0.4981 |     0 |     1 |

Accuracy and Runtime vs. Sample Size. We plot how accuracy and runtime vary as a function of the sample size. First, we plot this relationship in log-scale (Figures 4-5).

As they visually compress differences and obscure linear growth, especially when the slope is small, we replot accuracy vs. sample size in linear scale (Figure 6). The trends are indeed mildly increasing and consistent with near-linear growth, albeit with a small slope. We believe the soft growth observed on the original dataset is largely due to implementation-level factors. In particular, our use of NumPy's vectorized operations can lead to non-obvious runtime behavior, as such operations benefit from low-level optimizations (e.g., memory locality, multi-threaded backends) and often incur fixed overheads.

We also add further experimentation for larger sample sizes (Figure 7). Specifically, we generated a synthetic dataset from a 200 -dimensional multivariate Gaussian distribution and extended the sample

Figure 4: MNIST Dataset: Accuracy and runtime against sample size.

<!-- image -->

Figure 5: CoverType Dataset: Accuracy and runtime against sample size.

<!-- image -->

size up to 5 · 10 5 . The results are in line with theory: They show that the error decreases sharply as the sample size grows and that the runtime grows linearly with sample size. Unfortunately, we were unable to go beyond this sample size due to memory limitations in our environment, where the process exhausts the available RAM.

Accuracy and Runtime vs. Dimension. The experiments here illustrate how accuracy and runtime vary with dimensionality (Figure 8). Specifically, we generated a synthetic dataset of 10 , 000 samples drawn from multivariate Gaussian distributions with dimensions ranging from 10 to 1000 . As predicted by theory, the errors of both the coordinate-wise median and the fast gradient descent estimator remain low and stable across dimensions. Interestingly, the CSS algorithm shows rapid improvement as the dimension increases, stabilizing around dimension 500 . As expected, the empirical mean achieves the best accuracy since the data are drawn i.i.d. from a Gaussian distribution. In terms of runtime, we observe a roughly linear growth with dimension, in line with theory.

<!-- image -->

Sample Size

Figure 6: MNIST, Fashion-MNIST and CoverType Datasets: Accuracy against sample size (in linear scale).

<!-- image -->

(b) Runtime vs. sample size

Figure 7: Synthetic Dataset: Accuracy and runtime against sample size (in linear scale).

<!-- image -->

Figure 8: Synthetic Dataset: Accuracy and runtime against dimension.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We provide both verbal and visual intuition as well as formal proofs of all claims stated in the paper. We also give an in-depth experimental evaluation of the proposed algorithms.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Please refer to the Limitations subsection before the preliminaries.

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

Justification: Please refer to Sections 3, 4, and Appendices B, E.

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

Justification: We provide a full description of how to re-produce the experiments and include the code that is set up to automatically run all of them.

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

Justification: Code included in supplementary and anonymous link available in main body.

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

Justification: We provide several paragraphs describing how the experiments were set up and what the results mean. Please refer to Section 5.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We report, for each sample size we run the experiments , averages and variances of runtimes and accuracies arising from 50 executions of every algorithm. These are the most relevant statistical measures in our setting.

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

Justification: We include the hardware specifications for the experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We study a basic theoretical question that has been studied before.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This work concerns efficient mean estimation in high dimensions and therefore does not have negative societal impacts.

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

Justification: No such risk posed.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: No existing assets were used.

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

Justification: Our code is well-documented and should re-produce experiments out of the box (once the datasets are downloaded).

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

Justification: Not applicable.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: Not applicable.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.