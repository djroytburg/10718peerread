## A General-Purpose Theorem for High-Probability Bounds of Stochastic Approximation with Polyak Averaging

## Sajad Khodadadian ∗

Grado Department of Industrial and Systems Engineering Virginia Polytechnic Institute and State University Blacksburg, VA 24061

sajadk@vt.edu

## Martin Zubeldia ∗

Department of Industrial and Systems Engineering

University of Minnesota Minneapolis, MN 55455

zubeldia@umn.edu

## Abstract

Polyak-Ruppert averaging is a widely used technique to achieve the optimal asymptotic variance of stochastic approximation (SA) algorithms, yet its high-probability performance guarantees remain underexplored in general settings. In this paper, we present a general framework for establishing non-asymptotic concentration bounds for the error of averaged SA iterates. Our approach assumes access to individual concentration bounds for the unaveraged iterates and yields a sharp bound on the averaged iterates. We also construct an example, showing the tightness of our result up to constant multiplicative factors. As direct applications, we derive tight concentration bounds for contractive SA algorithms and for algorithms such as temporal difference learning and Q -learning with averaging, obtaining new bounds in settings where traditional analysis is challenging.

## 1 Introduction

Stochastic approximation (SA) algorithms are central to modern machine learning and optimization, serving as the foundation for methods ranging from stochastic gradient descent to Reinforcement Learning (RL) algorithms such as temporal difference (TD) learning and Q -learning. The general form of an SA algorithm is given by

<!-- formula-not-decoded -->

where α k = α/ ( k + h ) ξ for some h &gt; 1 , α &gt; 0 , and ξ ∈ [0 , 1] is the step size, and w k +1 represents random noise, possibly arising from sampling or environmental interactions. A significant body of work has been devoted to analyzing the convergence of such algorithms, providing both mean-square and high-probability error bounds under various assumptions (e.g., [49, 12, 19]).

∗ Equal contribution.

A key technique for improving the asymptotic variance and robustness of SA algorithms is PolyakRuppert averaging , where instead of using the iterates x k directly, one uses the averaged sequence

<!-- formula-not-decoded -->

This averaging has been shown to achieve optimal asymptotic behavior in a variety of settings, including stochastic gradient descent [40, 37, 1], contractive SA [10], and RL algorithms [36, 33]. Despite its empirical and theoretical effectiveness, sharp finite-time high-probability bounds for averaged SA in general settings remain underexplored.

## 1.1 Our Contributions

In this paper, we develop a general-purpose framework for deriving finite-time high-probability bounds for averaged SA iterates. Specifically, we assume that the base SA iterates x i , with probability at least 1 -δ ′ , satisfies a high-probability bound of the form ∥ x i -x ∗ ∥ 2 c ≤ α i f ( δ ′ , k ) for all δ ′ ∈ (0 , 1) and 0 ≤ i ≤ k , where x ∗ is the limit point (e.g., the fixed point or optimum), and f is a known function of δ ′ and k . Building on this assumption, we establish that for any δ ∈ (0 , 1) , with probability at least 1 -δ , the averaged iterates satisfy the bound

<!-- formula-not-decoded -->

where d is the dimension of the variables, ¯ σ 2 is a variance proxy, and c is a universal constant. We also construct an example to show that our result is tight up to the constant factor c .

Our general theorem has several important consequences. First, we provide a principled way to select the step size schedule { α k } k ≥ 0 to optimize the concentration of averaged SA under a variety of operators F . Second, we apply our framework to contractive SA algorithms , establishing sharp concentration bounds in these settings. Third, we leverage our results to derive new high-probability bounds on the convergence of averaged TD-learning improving upon the prior result. Furthermore, we establish concentration bounds on the convergence of averaged Q -learning and off-policy TDlearning, which to the best of our knowledge are the first in the literature. Finally, the general nature of our framework allows it to be used to obtain high-probability bounds of averaged SA iterates based on future refinements of non-average SA bounds (e.g., if high-probability bounds for SA under more general assumptions are developed in the future).

## 2 Related Work

Finite-Sample Guarantees for Contractive SA. Recent advances have significantly improved finite-time analyses of SA algorithms in contractive and strongly-monotone settings, often driven by applications in RL and optimization. Early finite-sample mean-square error bounds were established for specific SA instances such as TD-learning [6, 15, 16, 26, 34, 32, 17, 35] and Q -learning [20, 3], but lacked tail probability guarantees. Subsequent work provided high-probability concentration results. For instance, [49] introduced non-asymptotic bounds for unprojected linear SA, and [41] analyzed asynchronous Q -learning. Using martingale concentration techniques [21, 23], [51] obtained an O ( k -1 / 2 ) tail bound, while [12] utilized Lyapunov methods with convex envelopes to achieve finite-time results. This research culminated in general frameworks for Markovian SA: [13] provides polynomial tail bounds applicable to nonlinear contractive SA with Markovian noise. In particular, [14] (also [9, 8]) showed subgaussian concentration under minimal noise assumptions via exponential supermartingales, extending results beyond the i.i.d. setting. Sharper concentration bounds emerged for linear SA (LSA). [19] derived tight O (1 /k ) error rates for constant step size LSA. In nonlinear SA, notably Q -learning, averaging enhances performance. [53] established exponential tail bounds for averaged Q -learning, while [28] (also [30]) derived functional CLTs and optimal non-asymptotic error bounds for synchronous averaged Q -learning. Similarly, [55] demonstrated minimax-optimal complexity using averaging in variance-reduced Q -learning. Addressing practical concerns, [45] showed universal step sizes combined with tail-averaging yield near-optimal guarantees for linear TD-learning without projections or feature knowledge.

Polyak-Ruppert Averaging in Stochastic Approximation. Averaging is a classical variancereduction technique in SA, known to improve the asymptotic efficiency of iterates [40]. For TDlearning with linear function approximation, [31] provide finite-time high-probability bounds for averaging, albeit their step size depends on the confidence bound. [39, 46] strengthen this by showing that tail-averaged TD-learning with constant step sizes achieves optimal-order bias and variance, without requiring step size tuning. Furthermore, [36, 18] establish similar results for linear SA, which generalizes TD-learning with linear function approximation. In addition, via simulation, [24] provides a negative result on the convergence of averaged TD-learning. In Q -learning and other nonlinear SA settings, [29] establish tight sample complexity bounds. Follow-up work shows Polyak-averaged Q -learning achieves asymptotic efficiency [33]. [33] also provides a statistical analysis showing convergence to the minimax lower bound for tabular Q -learning. [48] derive a non-asymptotic CLT using Stein's method, applied to TD-learning. [44] prove Berry-Esseen bounds and establish bootstrap validity for linear SA. These enable rigorous confidence intervals in reinforcement learning. In actor-critic and other two-time-scale SA, [25] prove the first non-asymptotic CLT under averaging. [7] and [10] obtain finite-time bounds, showing that averaging improves rates from O ( k -2 / 3 ) to O ( k -1 ) .

High probability bounds vs bound on the distribution: Most of the prior work on the concentration of SA considers a step size which depends on the target probability δ [33, 46, 29]. Such analysis only provides a bound on a single point in the distribution of error. In contrast, in this paper, our step size does not depend on δ , and we establish a bound on the entire distribution of the error.

## 3 Stochastic Approximation and Polyak-Ruppert Averaging

In this section we review the classical SA framework, and the averaging technique.

Classical Stochastic Approximation : Let ¯ F : R d → R d be a (deterministic) operator, which has at least one fixed point x ∗ (i.e., ¯ F ( x ∗ ) = x ∗ ). The goal is to find this fixed point. If we have access to the deterministic operator ¯ F , under certain conditions (for instance, contractiveness of ¯ F or ¯ F ( x ) = x -∇ h ( x ) for some non-convex function h ) the iteration x k +1 = ¯ F ( x k ) converges to a fixed point. However, in many applications we do not have access to the deterministic operator ¯ F , and we only have access to noisy evaluations F ( · , w k +1 ) of this operator such that ¯ F ( · ) = E [ F ( · , w k +1 ) | F k ] , where F k is the sigma algebra generated by { w 1 , w 2 , . . . , w k } . The Robbins-Monro recursion [42] defines a sequence { x k } k ≥ 0 as in Equation (1.1), where { α k } k ≥ 0 is a step size sequence. Under some regularity conditions on the operator and the noise, and with the step size conditions ∑ ∞ k =0 α k = ∞ and ∑ ∞ k =0 α 2 k &lt; ∞ , it is known that the iterates converge to x ∗ almost surely [4]. SA recursions of the form of Equation (1.1) underpin a large class of algorithms in RL [50, 54] and optimization [22].

Polyak-Ruppert Averaging : This method was proposed independently by Ruppert and by Polyak and Juditsky [43, 40] to sharpen the statistical performance of SA. Given the raw iterates { x k } k ≥ 0 defined by Equation (1.1), Polyak-Ruppert averaging defines the averaged iterates { y k } k ≥ 0 as in Equation (1.2). It can be shown that √ k ( y k -x ∗ ) converges asymptotically to a normal distribution which has a covariance matrix that matches the Cramér-Rao lower bound [40]. An advantage of this averaging technique is that we can achieve the best asymptotic covariance with a robust choice of step size α k [38]. Besides the asymptotic results, there has been a long literature on the finite time analysis of SA algorithms, some of which are described in Section 2. Generally speaking, finite time bounds can be categorized into moment bounds (i.e., for some m ≥ 1 , finding a bound on E [ ∥ x k -x ∗ ∥ m ] or E [ ∥ y k -x ∗ ∥ m ] as a function of k ) and high-probability bounds (i.e., finding a bound on P ( ∥ x k -x ∗ ∥ ≥ ϵ ) or P ( ∥ y k -x ∗ ∥ ≥ ϵ ) as a function of ϵ &gt; 0 and k ).

## 4 Main Result

In this section, we present our main result. Throughout this section, we fix a given norm ∥ · ∥ c which satisfies the following assumption.

Assumption 4.1. The function ∥ · ∥ 2 c is M -smooth with respect to the ∥ · ∥ c norm, i.e., for all a, b ∈ R d , we have

<!-- formula-not-decoded -->

Remark. If the function ∥ · ∥ 2 c is non-smooth (e.g., if ∥ · ∥ c = ∥ · ∥ ∞ ), we can employ the machinery of the Moreau envelope [12] to obtain arbitrarily close smooth approximations of ∥ · ∥ 2 c . Therefore, Assumption 4.1 is without loss of generality.

Next, we impose an assumption on the operator ¯ F and its noisy version F , and on their Jacobians J ¯ F ( x ) := ∂ ¯ F ( x ) ∂x and J F w ( x, w ) := ∂F ( x,w ) ∂x .

Assumption 4.2. We assume the following.

- (i): The operator ¯ F admits at least one fixed point x ∗ , i.e., ¯ F ( x ∗ ) = x ∗
- (ii): The operator ¯ F is differentiable, and the matrix J ¯ F ( x ∗ ) -I is invertible. Hence, we have

<!-- formula-not-decoded -->

- (iii): The operator F ( x, w ) -x is ( N,R ) -locally psuedo smooth with respect to ∥·∥ c -norm. That is, there exists a radius R &gt; 0 such that, for all x satisfying ∥ x -x ∗ ∥ c ≤ R , we have:

<!-- formula-not-decoded -->

Note that Assumption 4.2(i) is relatively weak and it is satisfied for a wide range of operators. For instance, this assumption is satisfied for any contractive operator. Moreover, Lemma C.3 implies that Assumption 4.2(ii) is also satisfied for contractive operators. Furthermore, it can be readily verified that the linear operator ¯ F ( x ) = Ax + b satisfies Assumption 4.2(ii), provided that A is Hurwitz. Also, note that Assumption 4.2(iii) generalizes the notion of smoothness [2]. In fact, any smooth operator satisfies this assumption with R = ∞ .

Next, we impose an assumption on the noise of the operator at any fixed point x ∗ .

Assumption 4.3. For any d -dimensional, F k -measurable random vector v , and for any fixed point x ∗ , we have

<!-- formula-not-decoded -->

Moreover, for any d -dimensional, F k -measurable random vectors a and b , we have

<!-- formula-not-decoded -->

Remark. Assumption 4.3 is equivalent to assuming the noise in the operator and the Jacobian are subgaussian [53]. An example that satisfies Assumption 4.3 is when F ( x k , w k +1 ) = ¯ F ( x k ) + w k +1 , where { w i } i ≥ 1 are i.i.d. subgaussian random variables. Moreover, note that this assumption only needs to be satisfied at the fixed points x ∗ , and not for every x ∈ R d .

Next, we state the main result of our paper.

Theorem 4.1. Fix k ≥ 1 and ξ &lt; 1 . Consider the SA 1.1, and suppose that for any δ ′ ∈ (0 , 1) , with probability at least 1 -δ ′ , we have ∥ x i -x ∗ ∥ 2 c ≤ α i f ξ ( δ ′ , k ) for all 0 ≤ i ≤ k , for some function f ξ . Then, under assumptions 4.1, 4.2, and 4.3, for all δ ∈ (0 , 1) , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with 2

and

<!-- formula-not-decoded -->

The proof is given in Appendix B.1. Unlike the case of SA without averaging, where concentration bounds are obtained by establishing a one step recursion [14], here we need to take a more holistic approach and consider all the averaged iterates at once. This involves the added challenge of dealing with sums of noises directly, instead of tackling the noise of each iterate one at a time.

Theorem 4.1 provides a non-asymptotic high-probability bound for the SA iterates (1.1) with averaging. Our bound consists of two terms: ˜ ϵ ( k, δ/ 2) and ¯ ϵ ( k, δ/ 2) . For small enough k such that k 0 ( δ/ 2 , k ) = k +1 , the term ¯ ϵ ( k, δ/ 2) decays as 1 /k ξ , and thus it is the leading term as a function of k . For this range of time, the operator F is not assumed to be smooth, and thus our bound decays slower than the expected 1 /k rate. However, for all k large enough, with high probability, the iterates x k remain in the neighborhood where F is smooth. Theorem 4.1 states that from that point on, the term ¯ ϵ ( k, δ/ 2) decays as 1 /k 2 , and hence it is not a leading term anymore. Furthermore, for the SA with a globally smooth operator, Assumption 4.2(iii) is satisfied with R = ∞ , and Theorem 4.1 simplifies as follows.

Corollary 4.1. Under assumptions 4.1, 4.2 with R = ∞ , and 4.3, for all δ ∈ (0 , 1) , with probability at least 1 -δ , we have ∥ y k -x ∗ ∥ 2 c ≤ ˜ ϵ ( k, δ ) .

In the following subsections, we explore various implications of our main result.

## 4.1 Tail of the Leading and Higher Order Terms

We first look at the tail behavior in our bound. Throughout this discussion we assume that f ξ ( δ, · ) is non-increasing in δ and f ξ ( δ, · ) ∈ Ω(log(1 /δ )) , and that f ξ ( · , k ) ∈ O ( polylog ( k )) . Note that this assumption is reasonable as the rate of convergence of the error ∥ x k -x ∗ ∥ 2 c is typically ˜ O ( α k ) , and the tail of the error ∥ x k -x ∗ ∥ 2 c is typically sub-exponential or heavier [14]. In this case, our result implies that, if ξ &gt; 1 / 2 or N = 0 , then the error of the average variable y k is upper bounded as follows

<!-- formula-not-decoded -->

Note that the leading term is sub-exponential, and the higher order term has a tail that is potentially heavier than exponential, depending on f ξ . In particular, this implies that there exists a sequence of random variables { Z k } k ≥ 0 such that √ k ∥ y k -x ∗ ∥ ≤ Z k for all k ≥ 0 , and such that Z k converges to a subgaussian as k →∞ , although for every k &lt; ∞ , the error √ k ∥ y k -x ∗ ∥ could be heavier than subgaussian. Specifically, we have the following proposition.

Proposition 4.1. There exists an operator F and a sequence of random variables { w i } i ≥ 1 , which satisfy assumptions 4.2, and 4.3, such that √ k ∥ y k -x ∗ ∥ has a heavier tail than any Gaussian for all k ≥ 2 , while √

<!-- formula-not-decoded -->

where a, b &gt; 0 , and Φ( · ) is the CDF of the standard normal distribution.

2 Given two norms ∥·∥ a and ∥·∥ b , we define the constants ℓ ab and u ab such that ℓ ab ∥·∥ b ≤ ∥·∥ a ≤ u ab ∥·∥ b .

<!-- formula-not-decoded -->

This is consistent with CLT-style results on the limiting distribution of the error y k -x ∗ . However, this also suggests that Gaussian can be a poor approximation for the distribution of y k -x ∗ for finite k , as the distribution of the error could even be heavy-tailed.

Note that in Theorem 4.1, while we assume that the distribution of the squared error ∥ x k -x ∗ ∥ 2 c is upper bounded by f ξ , the distribution of the squared error of the average ∥ y k -x ∗ ∥ 2 c is bounded by f 2 ξ , which is heavier. Since the average of random variables cannot have a heavier tail than the original random variables, this must be an artifact of our proof. However, one can easily establish a high-probability bound which, for every finite k , the bound on the distribution of the squared error of the average has the same tail as the non-averaged iterates. Such a bound is given in the following lemma.

Lemma 4.1. Fix k ≥ 1 . Assume that for any δ ′ ∈ (0 , 1) with probability at least 1 -δ ′ , we have ∥ x i -x ∗ ∥ 2 c ≤ α i f ξ ( δ ′ , k ) for all 0 ≤ i ≤ k . Then with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

This result implies that that ∥ y k -x ∗ ∥ 2 c has at worst the same tail as ∥ x k -x ∗ ∥ 2 c . However, such tighter tail comes at the expense of a sup-optimal convergence rate of O (1 /k ξ ) instead of O (1 /k ) . Taking the minimum of the result in Theorem 4.1 and Lemma 4.1 we get the best of both worlds, as in the following corollary.

Corollary 4.2. Under the same assumptions as Theorem 4.1, with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

Corollary 4.2 shows that for any fixed time step k , as δ → 0 , we have ∥ y k -x ∗ ∥ 2 c ∈ O ( f ξ ( δ, 0)) , while for a fixed δ as k →∞ , we have ∥ y k -x ∗ ∥ 2 c ∈ O (1 /k ) . This shows that employing averaging will improve the convergence rate in k , while maintaining the same high-probability tail bound as the original iterates. We defer a more detailed comparison of averaged SA vs pure SA to Section 5.

## 4.2 Tightness of the Leading Term

Next, we discuss how tight the leading term in ˜ ϵ ( k, δ ) is. For that, we construct an example of a SA where the error is only a constant factor away from our leading term.

Suppose that the dimension d is even. Consider a 2 -dimensional i.i.d. sequence of random variables { z i } i ≥ 1 such that z i ∼ N (0 , (¯ σ 2 /d ) I ) . Using these, we construct a d -dimensional sequence of i.i.d. random variables { w i } i ≥ 0 such that the j 'th element of w i is equal to the first element of z i when j odd, i.e., w i,j = z i, 1 , and it is equal to the second element of z i when j even is even, i.e., we have w i,j = z i, 2 . Consider the operator F ( x, w ) = w , and the step sizes α k = α for all k ≥ 0 . This setting satisfies assumptions 4.2 and 4.3 for ∥ · ∥ c = ∥ · ∥ 2 , with ν = 1 and R = ∞ . In addition, we have the unique fixed point x ∗ = 0 . The following proposition bounds the error for this case.

Proposition 4.2. For the example above, for all even dimensions d , and for any δ ∈ (0 , 1) , we have

<!-- formula-not-decoded -->

Recall that for large k , small δ , and ∥ · ∥ c = ∥ · ∥ 2 , the leading term in Theorem 4.1 is 2 √ 6¯ σ/ √ k +1 . Proposition 4.2 implies that that this upper bound is at most a factor of 2 √ 6 away from the error for this example. Thus, Theorem 4.1 is at most this universal constant away from the tightest possible bound.

## 4.3 Linear versus Non-Linear Operators

We now discuss how our bound depends on the linearity or non-linearity of the operators, and on whether the noise is additive or not. In particular, this has an important effect on what choice of ξ maximizes the convergence rate of the higher order terms in our upper bound.

1. Linear Operator + Additive Noise: Suppose that F ( x k , w k ) = Ax k + w k . It is easy to see that Assumption 4.2 is satisfied with N = 0 and that Assumption 4.3 is satisfied with ˆ σ = 0 . Hence, with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

In this case a smaller ξ achieves a better convergence rate for the higher order terms. Furthermore, the tail of the higher order term is the same as the tail of the errors ∥ x i -x ∗ ∥ 2 for i ≥ 0 , which could be heavier than subgaussian.

2. Linear Operator + Non Additive Noise: Suppose that N = 0 and ˆ σ 2 &gt; 0 . Then, we have

<!-- formula-not-decoded -->

Here the best choice of ξ is 1 / 2 , and the tail of the higher order term is heavier than the error ∥ x i -x ∗ ∥ 2 by a factor of log(1 /δ ) . For instance, if ∥ x i -x ∗ ∥ 2 c is Weibull with shape parameter m , then the higher order term is sub-Weibull with shape parameter m +1 .

3. Nonlinear Operator: If N &gt; 0 , we have

<!-- formula-not-decoded -->

Our result suggest that the best choice of ξ is 2 / 3 . In this case, the tail of the higher order term is twice as heavy as the tail of ∥ x i -x ∗ ∥ 2 c .

In general, we observe that for nonlinear operators, we get the same rate of convergence for higherorder terms, regardless of whether the noise is additive or not, and this rate is worse than in the case of linear operators.

Remark. For the case of constant step sizes, i.e., for ξ = 0 , our high-probability bound is of order O ( N 2 α 2 ) + o (1) when the operator is non-linear. This is in line with [27] where they show that the bias of a SA with constant step size α and a non-linear operator is proportional to α . On the other hand, if the operator is linear, we have N = 0 , and the bias disappears. This is consistent with [36].

## 5 Application to Contractive Operators

In this section we will apply our results for the important class of contractive operators ¯ F , which is specified as follows.

Assumption 5.1. There exists a constant γ c ∈ [0 , 1) and a norm ∥ · ∥ c such that

<!-- formula-not-decoded -->

Recall that in Theorem 4.1 we require a high-probability bound on the SA iterates { x k } k ≥ 0 . To obtain such a high-probability bound, we use the prior work [14] that differentiates between two cases, additive and multiplicative noise, which will be studied in sections 5.1 and 5.2, respectively.

## 5.1 Additive Noise

We first study the additive noise setting (as it was defined in [14]), which is specified in the following assumption.

Assumption 5.2. The random vector F ( x k , w k +1 ) -¯ F ( x k ) is subgaussian, i.e., there exist σ &gt; 0 and a (possibly dimension-dependent) constant c d &gt; 0 such that for any k ≥ 0 and F k -measurable random vector v , the following two inequalities hold:

<!-- formula-not-decoded -->

where ∥ · ∥ ∗ c is the dual of the norm ∥ · ∥ c , and

<!-- formula-not-decoded -->

Employing [14, Theorem 2.4] in Theorem 4.1, we get the following result.

Proposition 5.1. Under assumptions 4.1, 4.2, 4.3, 5.1, and 5.2, using ξ &lt; 1 and h ≥ ( 2 ξ (1 -γ c ) α ) 1 / (1 -ξ ) , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

Next, we aim at comparing the high-probability bound achieved by averaging vs the high-probability bound achieved through Theorem 2.4 in [14] with ξ = 1 . For the sake of comparison, we assume that ∥ · ∥ c = ∥ · ∥ 2 . In that case we have u c 2 = 1 and M = 1 . By Proposition 5.1, with probability at least 1 -δ , we have (up to the leading term)

<!-- formula-not-decoded -->

Moreover, applying [14, Theorem 2.4] with ξ = 1 and k = K , we obtain

<!-- formula-not-decoded -->

for any µ &gt; 0 and a &gt; 1 (refer to Appendix A for calculations). Here, ¯ σ 2 is the variance of the subgaussian noise only at x ∗ , while σ 2 is a uniform bound on the variance over all x ∈ R d . Hence, we have ¯ σ 2 ≤ σ 2 .

Our results suggest that for SA with additive noise, averaging could improve the constant in the leading term. In particular, in the first term, SA has 32 aσ 2 which is at least 32 σ 2 . Averaging improves this term to 24¯ σ 2 . In the second term SA has 32 a 2 a -1 ec d (1 + µ ) σ 2 which is at least 128 ec d σ 2 . For the case that c d ≈ d (which happens when the noise is N (0 , I d × d ) ), averaging improves this term to 3( d +8log(2))¯ σ 2 . We will do a more specific comparison of SA with and without averaging for RL algorithms in Section 6.

Remark. It can be shown that, even if we use a smaller value of h than the lower bound ( 2 ξ (1 -γ c ) α ) 1 / (1 -ξ ) in Proposition 5.1, a concentration bound similar to the one of Equation (5.3) still holds, albeit with larger constants in the higher-order terms.

## 5.2 Multiplicative Noise

Next, we examine the multiplicative noise scenario (as it was defined in [14]), which is characterized by the following assumption:

Assumption 5.3 ([14]) . There exists σ &gt; 0 such that

<!-- formula-not-decoded -->

where ∥ · ∥ c is the norm from Assumption 5.1.

Utilizing [12, Corollary 2.2], Markov's inequality, and Theorem 4.1, we derive the following result:

Theorem 5.1. Under assumptions 4.1, 4.2, 4.3, 5.1, 5.3, and ξ &gt; 1 / 2 , for any δ ∈ (0 , 1) , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

Theorem 5.1 provides a heavy-tailed upper bound for the error. Although this is merely an upper bound, the actual distribution of the error is indeed heavy-tailed, as demonstrated by the following impossibility result:

Theorem 5.2. Under assumptions 5.1 and 5.3, when ξ ∈ (0 , 1) , there does not exist any c 0 &gt; 0 and m&gt; 0 , such that for all δ ∈ (0 , 1) with probability at least 1 -δ

<!-- formula-not-decoded -->

The above result highlights a fundamental difference between multiplicative and additive noise scenarios. Specifically, while Theorem 2.1 from [14] shows that simple SA with step size α k = α/ ( k + h ) attains a sub-Weibull distribution, our Theorem 5.2 demonstrates that averaging applied to general SA under multiplicative noise yields a heavier tail than any sub-Weibull distribution. This contrasts sharply with the additive noise scenario, where averaging maintains a subexponential tail similar to standard SA, while improving the constant factor of the leading term.

## 6 Application to Reinforcement Learning

In this section, we study the application of our result to TD-learning and Q -learning algorithms.

TD-Learning: Consider a finite Markov Decision Process characterized by state space S , action space A , reward function R ( s, a ) : S × A → [0 , R max ] , and transition probability matrix P . We assume |S| ≥ 2 . The objective of TD-learning is to estimate the value function associated with a policy π , defined as V π ( s ) = E [ ∑ ∞ i =0 γ i R ( S i , A i ) | S 0 = s ] . We focus on asynchronous TD ( n ) with i.i.d. noise, given by

<!-- formula-not-decoded -->

for all s ∈ S . Here, S 0 0 , S 0 1 , . . . iid ∼ µ π , where µ π denotes the stationary distribution of the induced Markov chain under policy π . For each k , { ( S i k , A i k ) } 0 ≤ i ≤ n represents a trajectory following policy π , with transitions A i k ∼ π ( ·| S i k ) , and S i +1 k ∼ P ( ·| S i k , A i k ) . We assume that the initial point of the algorithm satisfies V 0 ( s ) ∈ [0 , R max 1 -γ ] for all s ∈ S and that α k = α/ √ k + h with α/ √ h ≤ 1 . We denote ¯ V k = 1 k +1 ∑ k i =0 V i .

Theorem 6.1. Under the asynchronous TD ( n ) algorithm (6.1) , for any δ ∈ (0 , 1) , with probability at least 1 -δ , the following bound holds:

<!-- formula-not-decoded -->

To the best of our knowledge, Theorem 6.1 establishes the first bound on the entire distribution of the error for averaged TD-learning. In particular, prior works such as [19, 49, 36] provide bounds for the raw iterates or use step sizes that depend on a fixed confidence level δ , and thus only give pointwise control (i.e., one quantile of the error distribution). In contrast, since our step size is independent of δ , we obtain bounds that control the entire tail of the error distribution for the averaged iterates.

Q-Learning: Next, we study the asynchronous Q -learning algorithm with i.i.d. noise, given by

<!-- formula-not-decoded -->

for all ( s, a ) ∈ S × A , where S k ∼ µ π b , A k ∼ π b ( ·| S k ) , S ′ k ∼ P ( ·| S k , A k ) and π b is some fixed sampling policy. We denote ρ b = min s,a µ π b ( s ) π b ( a | s ) . We assume that the initial point of the algorithm satisfies Q 0 ( s, a ) ∈ [0 , R max 1 -γ ] for all ( s, a ) ∈ S ×A and α k = α/ √ k + h with α/ √ h &lt; 1 . Without loss of generality, in this section we also assume |A| ≥ 2 . We denote the optimal Q -function as Q ∗ and ¯ Q k = 1 k +1 ∑ k i =0 Q i . We further impose the following common [56] assumption on Q ∗ .

Assumption 6.1. Q ∗ is greedily unique, i.e. for every s ∈ S , a ∗ ( s ) = arg max a ′ { Q ∗ ( s, a ′ ) } is unique.

Theorem 6.2. For asynchronous Q-learning with i.i.d. noise and ξ = 1 / 2 , under Assumption 6.1, for any δ ∈ (0 , 1) , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

To the best of our knowledge, Theorem 6.2 establishes the first bound on the entire distribution of the error for averaged Q -learning.

We observe that, since the noise structure in tabular TD-learning and Q -learning is the same (both corresponding to additive noise), the bounds are of the same order of k and δ .

Off-policy TD-learning: Theorems 6.1 and 6.2 are two instances of RL algorithms which can be modeled with SA with additive noise, as in both of these settings the noise is bounded. Next, we study off-policy TD-learning with linear function approximation which has multiplicative noise. In this setting we assume we have access to a matrix Φ ∈ R |S|× d , and we denote the s -th row of this matrix by ϕ ( s ) . The goal is to find v π ∈ R d that estimates the value function as V π ( s ) ≈ v π ϕ ( s ) through the solution of the following fixed point equation Φ v π = Π π b Φ (( T π ) n Φ v π ) Here, Π π b Φ = Φ(Φ ⊤ K π b Φ) -1 Φ ⊤ K π b is a linear function that projects into the subspace spanned by the matrix Φ , and K π b is a diagonal matrix, with diagonal entries equal to the stationary distribution of the behavior policy π b . Furthermore, T π is the Bellman operator. We employ TD ( n ) as follows:

<!-- formula-not-decoded -->

where S 0 0 , S 0 1 , . . . iid ∼ µ π b , and µ π b is the stationary distribution over states of the induced Markov chain by following policy π b , and for every k , { ( S i k , A i k ) } 0 ≤ i ≤ n is a single trajectory of state-action pairs following policy π b as A i k ∼ π b ( ·| S i k ) , S i +1 k ∼ P ( ·| S i k , A i k ) , 0 ≤ i ≤ n . We assume that α k = α/ ( k + h ) ξ . We denote ¯ v k = 1 k +1 ∑ k i =0 v i .

Theorem 6.3. For large enough n , there exists a constant c &gt; 0 such that, for any δ ∈ (0 , 1) , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

To the best of our knowledge, Theorem 6.3 establishes the first bound on the entire distribution of the error for the averaged off-policy TD ( n ) with linear function approximation.

Note that for TD-learning with linear function approximation, the tail has O (1 /δ ) dependence, which is a sub-Pareto distribution. Moreover, TD-learning with linear function approximation is an example of SA with multiplicative noise, and as shown in Theorem 5.2, the tail has Ω((log(1 /δ )) n ) for all n ≥ 1 . In particular, this means that the tail of TD-learning with linear function approximation is heavier than any sub-weibull, but it is at most as heavy as sub-pareto. In contrast, as shown in Theorems 6.1 and 6.2, tabular TD-learning and Q -learning have O (log(1 /δ )) tail, which is sub-exponential.

## References

- [1] F. Bach and E. Moulines. Non-strongly-convex smooth stochastic approximation with convergence rate O(1/n). In Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS) , page 773-781, 2013.
- [2] A. Beck. First-order methods in optimization . SIAM, 2017.
- [3] C. L. Beck and R. Srikant. Error bounds for constant step-size Q-learning. Systems &amp; Control Letters , 61(12):1203-1208, 2012.
- [4] D. Bertsekas and J. N. Tsitsiklis. Neuro-dynamic programming . Athena Scientific, 1996.

- [5] D. P. Bertsekas, D. P. Bertsekas, D. P. Bertsekas, and D. P. Bertsekas. Dynamic programming and optimal control , volume 2. Athena scientific, 1995.
- [6] J. Bhandari, D. Russo, and R. Singal. A finite-time analysis of temporal-difference learning with linear function approximation. Operations Research , 69(3):950-973, 2021.
- [7] S. Chandak. O(1/k) Finite-Time Bound for Non-Linear Two-Time-Scale Stochastic Approximation. arXiv preprint arXiv:2504.19375 , 2024.
- [8] S. Chandak and V. S. Borkar. A Concentration Bound for TD(0) with Function Approximation. arXiv preprint arXiv:2312.10424 , 2023.
- [9] S. Chandak, V. S. Borkar, and P. R. Dodhia. Concentration of Contractive Stochastic Approximation and Reinforcement Learning. Stochastic Systems , 12(4):411-430, 2022.
- [10] S. Chandak, S. U. Haque, and N. Bambos. Finite-Time Bounds for Two-Time-Scale Stochastic Approximation with Arbitrary Norm Contractions and Markovian Noise. arXiv preprint arXiv:2503.18391 , 2025.
- [11] Z. Chen, S. Khodadadian, and S. T. Maguluri. Finite-Sample Analysis of Off-Policy Natural Actor-Critic With Linear Function Approximation. IEEE Control Systems Letters , 6:2611-2616, 2022.
- [12] Z. Chen, S. T. Maguluri, S. Shakkottai, and K. Shanmugam. Finite-sample analysis of contractive stochastic approximation using smooth convex envelopes. Advances in Neural Information Processing Systems , 33:8223-8234, 2020.
- [13] Z. Chen, S. T. Maguluri, S. Shakkottai, and K. Shanmugam. A Lyapunov theory for finite-sample guarantees of Markovian stochastic approximation. Operations Research , 71(1):147-164, 2023.
- [14] Z. Chen, S. T. Maguluri, and M. Zubeldia. Concentration of contractive stochastic approximation: Additive and multiplicative noise. The Annals of Applied Probability , 35, 2025.
- [15] G. Dalal, B. Szörényi, G. Thoppe, and S. Mannor. Finite sample analyses for TD(0) with function approximation. In Proc. AAAI , volume 32, 2018.
- [16] G. Dalal, G. Thoppe, B. Szörényi, and S. Mannor. Finite sample analysis of two-time-scale stochastic approximation with applications to reinforcement learning. In Proceedings of the Conference on Learning Theory (COLT) , pages 1199-1233, 2018.
- [17] R. Dorfman and K. Y. Levy. Adapting to mixing time in stochastic optimization with markovian data. In International Conference on Machine Learning , pages 5429-5446. PMLR, 2022.
- [18] A. Durmus, E. Moulines, A. Naumov, and S. Samsonov. Finite-time high-probability bounds for Polyak-Ruppert averaged iterates of linear stochastic approximation. Mathematics of Operations Research , 49(1):454-478, 2024.
- [19] A. Durmus, E. Moulines, A. Naumov, S. Samsonov, K. Scaman, and H.-T. Wai. Tight high probability bounds for linear stochastic approximation with fixed stepsize. Advances in Neural Information Processing Systems , 34:30063-30074, 2021.
- [20] E. Even-Dar and Y. Mansour. Learning rates for Q-learning. Journal of Machine Learning Research , 5:1-25, 2003.
- [21] D. A. Freedman. On tail probabilities for Martingales. The Annals of Probability , pages 100-118, 1975.
- [22] J. Harold, G. Kushner, and G. Yin. Stochastic approximation and recursive algorithm and applications. Application of Mathematics , 35(10), 1997.
- [23] W. Hoeffding. Probability inequalities for sums of bounded random variables. In The Collected Works of Wassily Hoeffding , pages 409-426. Springer, 1994.

- [24] K. Khamaru, A. Pananjady, F. Ruan, M. J. Wainwright, and M. I. Jordan. Is temporal difference learning optimal? an instance-dependent analysis. SIAM Journal on Mathematics of Data Science , 3(4):1013-1040, 2021.
- [25] S. T. Kong, S. Zeng, T. T. Doan, and R. Srikant. Nonasymptotic CLT and Error Bounds for Two-Time-Scale Stochastic Approximation. arXiv preprint arXiv:2502.09884 , 2024.
- [26] C. Lakshminarayanan and C. Szepesvári. Linear stochastic approximation: How far does constant step-size and iterate averaging go? In Proc. AISTATS , pages 1347-1355, 2018.
- [27] C. K. Lauand and S. Meyn. Bias in stochastic approximation cannot be eliminated with averaging. In Proceedings of the Annual Allerton Conference on Communication, Control, and Computing , pages 1-4, 2022.
- [28] G. Li, C. Cai, Y. Chen, Y. Gu, Y. Wei, and Y. Chi. Tightening the dependence on horizon in the sample complexity of Q-learning. In Proceedings of the Conference on Learning Theory (COLT) , 2021. Extended version: arXiv:2102.06548.
- [29] G. Li, C. Cai, Y. Chen, Y. Wei, and Y. Chi. Is Q-Learning Minimax Optimal? A Tight Sample Complexity Analysis. Operations Research , 72(1):222-236, 2024.
- [30] G. Li, Y. Wei, Y. Chi, Y. Gu, and Y. Chen. Sample complexity of asynchronous Q-learning: Sharper analysis and variance reduction. IEEE Transactions on Information Theory , 68(1):448473, 2022.
- [31] G. Li, W. Wu, Y. Chi, C. Ma, A. Rinaldo, and Y. Wei. High-probability sample complexities for policy evaluation with linear function approximation. IEEE transactions on information theory , 70(8):5969-5999, 2024.
- [32] T. Li, G. Lan, and A. Pananjady. Accelerated and instance-optimal policy evaluation with linear function approximation. arXiv preprint arXiv:2112.13109 , 2021.
- [33] X. Li, W. Yang, J. Liang, Z. Zhang, and M. I. Jordan. A Statistical Analysis of Polyak-Ruppert Averaged Q-Learning. In Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS) , volume 206, pages 2207-2261, 2023.
- [34] R. Liu and A. Olshevsky. Temporal difference learning as gradient splitting. In International Conference on Machine Learning , pages 6905-6913. PMLR, 2021.
- [35] A. Mitra. A simple finite-time analysis of td learning with linear function approximation. IEEE Transactions on Automatic Control , 2024.
- [36] W. Mou, C. J. Li, M. J. Wainwright, P. L. Bartlett, and M. I. Jordan. On linear stochastic approximation: Fine-grained Polyak-Ruppert and non-asymptotic concentration. In Proceedings of the Conference on Learning Theory (COLT) , pages 2947-2997, 2020.
- [37] E. Moulines and F. Bach. Non-asymptotic analysis of stochastic approximation algorithms for machine learning. Advances in neural information processing systems , 24, 2011.
- [38] A. Nemirovski, A. Juditsky, G. Lan, and A. Shapiro. Robust stochastic approximation approach to stochastic programming. SIAM Journal on optimization , 19(4):1574-1609, 2009.
- [39] G. Patil, L. Prashanth, D. Nagaraj, and D. Precup. Finite time analysis of temporal difference learning with linear function approximation: Tail averaging and regularisation. In International Conference on Artificial Intelligence and Statistics , pages 5438-5448. PMLR, 2023.
- [40] B. T. Polyak and A. B. Juditsky. Acceleration of stochastic approximation by averaging. SIAM journal on control and optimization , 30(4):838-855, 1992.
- [41] G. Qu and A. Wierman. Finite-time analysis of asynchronous stochastic approximation and Q-learning. In Proceedings of the Conference on Learning Theory (COLT) , pages 3185-3205, 2020.
- [42] H. Robbins and S. Monro. A stochastic approximation method. The annals of mathematical statistics , pages 400-407, 1951.

- [43] D. Ruppert. Efficient estimations from a slowly convergent Robbins-Monro process. Technical report, Cornell University Operations Research and Industrial Engineering, 1988.
- [44] S. Samsonov, E. Moulines, Q.-M. Shao, Z.-S. Zhang, and A. Naumov. Gaussian Approximation and Multiplier Bootstrap for Polyak-Ruppert Averaged Linear SA with Applications to TD Learning. In Advances in Neural Information Processing Systems , 2024.
- [45] S. Samsonov, D. Tiapkin, A. Naumov, and E. Moulines. Finite-sample analysis of the temporal difference learning. In Proceedings of the Conference on Learning Theory (COLT) , 2024.
- [46] S. Samsonov, D. Tiapkin, A. Naumov, and E. Moulines. Improved High-Probability Bounds for the TD Learning Algorithm via Exponential Stability. In Proceedings of the Conference on Learning Theory (COLT) . PMLR, 2024.
- [47] S. Shalev-Shwartz et al. Online learning and online convex optimization. Foundations and Trends® in Machine Learning , 4(2):107-194, 2012.
- [48] R. Srikant. Rates of Convergence in the Central Limit Theorem for Markov Chains, with an Application to TD Learning. arXiv preprint arXiv:2401.15719 , 2024.
- [49] R. Srikant and L. Ying. Finite-time error bounds for linear stochastic approximation and TD learning. In Proceedings of the Conference on Learning Theory (COLT) , pages 2803-2830. PMLR, 2019.
- [50] R. S. Sutton. Learning to predict by the methods of temporal differences. Machine learning , 3:9-44, 1988.
- [51] G. Thoppe and V. S. Borkar. A concentration bound for stochastic approximation via Alekseev's formula. Stochastic Systems , 9(1):1-26, 2019.
- [52] J. Tsitsiklis and B. Van Roy. An analysis of temporal-difference learning with function approximation. IEEE Transactions on Automatic Control , 42(5):674-690, 1997.
- [53] M. J. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint . Cambridge University Press, 2019.
- [54] C. J. Watkins and P. Dayan. Q-learning. Machine learning , 8:279-292, 1992.
- [55] E. Xia, K. Khamaru, M. Wainwright, and M. Jordan. Instance-optimality in optimal value estimation: Adaptivity via variance-reduced Q-learning. IEEE Transactions on Information Theory , page 1, 2024.
- [56] Y. Zhang and Q. Xie. Constant stepsize Q-learning: Distributional convergence, bias and extrapolation. Reinforcement Learning Journal , 3:1168-1210, 2024.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly state our contributions, which accurately match the theoretical results provided in Sections 4, 5, 6.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We clearly specify our assumptions for each theorem, and after each assumption we argue why the assumptions are reasonable.

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

Justification: All the necessary assumptions are clearly stated and/or cited in the main paper, and the complete proofs of all the results are provided in the appendix.

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

Justification: This paper is pure theory, and it does not include any experiments.

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

Justification: This paper is pure theory, and it does not include any experiments.

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

Justification: This paper is pure theory, and it does not include any experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: This paper is pure theory, and it does not include any experiments.

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

Justification: This paper is pure theory, and it does not include any experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have reviewed the NeurIPS Code of Ethics and our research conforms in all aspects with this document.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This paper is pure theory, and we are only analyzing already known algorithms. Hence, it does not have any societal impacts.

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

Justification: This paper is pure theory and it does not pose any risks for misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper is analyzing already well-known algorithms theoretically, and it does not use any existing assets.

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

Justification: This paper is pure theory, and it does not include any new asset.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper is pure theory, and it does not include any experiments.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper is pure theory, and it does not involve any crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/ LLM ) for what should or should not be described.

## A Analysis of the case study in Section 5

The generalized Moreau envelope norm ∥ · ∥ m was introduced in [14], and it is defined as

<!-- formula-not-decoded -->

By choosing ∥ · ∥ s = ∥ · ∥ c , as shown in [14], 1 2 ∥ · ∥ 2 m is M -smooth. Furthermore, we have ∥ · ∥ m = ∥ · ∥ c / √ 1 + µ . Hence, u cm = ℓ cm = √ 1 + µ . Also, we choose α = a/ (1 -γ c ) with a &gt; 1 . Note that u mc ∗ = u mc u cc ∗ = u cc ∗ / √ 1 + µ . Moreover, the contraction factor under the norm ∥ · ∥ m is γ c .

## B Proofs of main results

## B.1 Proof of Theorem 4.1

For ease of notation, we drop ξ from f ξ ( δ, k ) .

We define the event

<!-- formula-not-decoded -->

which we assumed to have /a80 ( E k ) ≥ 1 -δ . For ease of notation, we use E k instead of E k ( δ ) .

In order to obtain a tight high probability bound, we need to exploit Assumption 4.2(iii), which only holds whenever ∥ x k -x ∗ ∥ c ≤ R . We can ensure that with high probability, the SA is within the local smooth range of the operator after a certain time. Define ˜ k 0 ( δ, k ) ≥ 0 to be the smallest number such that α ˜ k 0 ( δ,k ) f ( δ, k ) ≤ R 2 . Solving for the smallest ˜ k 0 ( δ, k ) we have

<!-- formula-not-decoded -->

We define k 0 ( δ, k ) := min { ˜ k 0 ( δ, k ) , k +1 } . For ease of notation, unless otherwise stated, we denote k 0 ( δ, k ) by k 0 .

For any k ≥ 0 , we have

<!-- formula-not-decoded -->

We define

<!-- formula-not-decoded -->

It is clear that y k -x ∗ = ¯ y k +˜ y k . Applying norm on both sides, and using triangle inequality, we get

<!-- formula-not-decoded -->

We study ∥ ¯ y k ∥ c as follows. For every sample path in the event E k 0 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, for any k ≥ 0 , with probability at least 1 -δ we have

<!-- formula-not-decoded -->

Next, we study ∥ ˜ y k ∥ c . We have

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

By Assumption 4.1, we have

<!-- formula-not-decoded -->

Hence, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let us define

We have

<!-- formula-not-decoded -->

where in the last inequality we use Cauchy-Schwarz three times. We study each of the terms above separately.

<!-- formula-not-decoded -->

The term T 1 : By Assumption 4.3 and Lemma C.1, it follows that ∑ k i = k 0 ( F ( x ∗ , w i +1 ) -x ∗ ) is ( k -k 0 +1)¯ σ 2 -subgaussian. Furthermore, by Lemma C.2, we have

<!-- formula-not-decoded -->

where in last inequality we use that 1 / (1 -x ) ≤ exp(2 x ) for all x ≤ 1 / 2 .

## The term T 2 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( a ) and ( b ) are by triangle inequality and ( a + b + c ) 2 ≤ 3 a 2 + 3 b 2 + 3 c 2 , ( c ) is by the assumption of the theorem, ( d ) is by ( x +1) ξ -x ξ ≤ ξ/x 1 -ξ for x &gt; 0 , and ( e ) is by integral upper bound of summation.

The term T 3 :

<!-- formula-not-decoded -->

where in the last inequality we used the fact that ( k + h ) 2 -2 ξ ≤ ( k +1) 2 -2 ξ h 2 -2 ξ .

The term T 4 : It is easy to see that, almost surely, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now show that Assumption 4.3 implies that

<!-- formula-not-decoded -->

Indeed, by Taylor expansion of the inequality (4.2), we get

<!-- formula-not-decoded -->

̸

If we had E [ J ¯ F ( x ∗ ) -J F w ( x ∗ , w k +1 )) |F k ] = 0 , then we would have Ω( λ ) ≤ O ( λ 2 ) , which is a contradiction. It follows that the sequence of random variables given by [ J ¯ F ( x ∗ ) -J F w ( x ∗ , w i +1 )]( x i -x ∗ ) ✶ E i is a Martingale difference and, conditioned on F i , they are ˆ σ 2 ∥ ( x i -x ∗ ) ✶ E i ∥ 2 -subgaussian. Hence, by the definition of the event E i , the random variable [ J ¯ F ( x ∗ ) -J F w ( x ∗ , w i +1 )]( x i -x ∗ ) ✶ E i is ˆ σ 2 f ( δ, k ) α i -subgaussian. Thus, Lemma C.1 implies that ✶ E k +1 ∑ k i = k 0 [ J ¯ F ( x ∗ ) -J F w ( x ∗ , w i +1 )]( x i -x ∗ ) is ˆ σ 2 f ( δ, k ) ∑ k i = k 0 α i -subgaussian. Furthermore, by Lemma C.2, we have

<!-- formula-not-decoded -->

where in the second to last inequality we use the fact that 1 / (1 -x ) ≤ exp(2 x ) for all x ≤ 1 / 2 , and it holds when

<!-- formula-not-decoded -->

which is satisfied because we have

<!-- formula-not-decoded -->

Finally, putting everything together, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Making the right hand side equal to δ , and solving for ϵ we get that, with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Note that, for any two events A and B , we have

Define

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

Hence, with probability at least 1 -δ (with abuse of notation for substituting 2 δ by δ ), we have

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

where the last inequality follows from (B.2) and (B.6), and

<!-- formula-not-decoded -->

## B.2 Proof of Proposition 4.1

Consider the recursion

<!-- formula-not-decoded -->

where w k ∼ N (0 , 1) . It follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Without loss of generality, we assume that x 0 &gt; 0 . Consider the event E k = { w i ≥ 0 , i = 1 , . . . , k } , and suppose that k &gt; 2 . For any t &gt; 0 , we have

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

We also have

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

which is infinity for t &gt; ( k +1) / ( x 0 α 0 α 1 ) .

Next, consider α j = α/ ( j + h ) ξ such that α 0 &lt; 1 / 2 . By Markov's inequality, we have

<!-- formula-not-decoded -->

where the last inequality holds for some constant c &gt; 0 . Then, for h &gt; 2 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, with probability at least 1 -δ , for all 0 ≤ i ≤ k , we have | x i | 2 ≤ c ′′ α k /δ . This means that f ( δ, k ) = c ′′ /δ . Therefore, by Theorem 4.1, with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

The result follows.

## B.3 Proof of Lemma 4.1

The proof follows from Equation (B.2), and by assuming k 0 = k +1 . In particular, with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

## B.4 Proof of Proposition 4.2

First, note that

<!-- formula-not-decoded -->

In addition, this example satisfies Assumption 4.3 as

<!-- formula-not-decoded -->

For the sake of simplicity, we assume x 0 = 0 . Then, we have

<!-- formula-not-decoded -->

## B.5 Proof of Theorem 5.1

Since the c -norm can be non-smooth, to prove a concentration result on the error we use the generalized Moreau envelope

<!-- formula-not-decoded -->

where ∥ · ∥ s is an arbitrary smooth norm. Then, we have that M ( · ) is an L/µ - smooth function with respect to ∥ · ∥ s . Moreover, the Moreau envelope defines the norm ∥ · ∥ M = √ 2 M ( · ) . For a more detailed discussion about this function, please refer to [12].

Since Assumptions 5.1, and 5.2 are satisfied, and α i = α/ ( i + h ) ξ , where ξ ∈ (0 , 1) . Then, by choosing α &gt; 0 and h ≥ (2 ξ/ [(1 -˜ γ c ) α ]) 1 / (1 -ξ ) , for any δ &gt; 0 and K ≥ 0 , using the result from [14, Theorem 2.4], we know that with probability at least 1 -δ , we have for all i ≥ K that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with ˜ γ c = γ c (1 + µu 2 cs ) 1 / 2 / (1 + µℓ 2 cs ) 1 / 2 and ¯ D = 2(1 -˜ γ c ) . Next, choosing K = 1 , we get that Equation (B.7) holds for all i ≥ 1 . In addition, since ¯ c 2 ≥ 1 and ¯ c l ≥ 0 for l = 1 , 4 , 5 this bound holds i = 0 as well. Hence, for all i ≤ k , we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

We get the result by directly applying Theorem 4.1.

## B.6 Proof of Theorem 5.1

From the proof of Theorem 3 in [37], for h large enough, we have

<!-- formula-not-decoded -->

where u i = ∥ x i +1 -x ∗ ∥ 4 c + β 3 α i ∥ x i +1 -x ∗ ∥ 2 c for some positive constants β 1 , β 2 , and β 3 . Define

<!-- formula-not-decoded -->

Next, we prove that { M i } i ≥ 0 is a supermatringale:

<!-- formula-not-decoded -->

In the last inequality, we used the following

<!-- formula-not-decoded -->

Hence,

Hence,

<!-- formula-not-decoded -->

Applying Theorem 4.1 with f ξ ( δ, k ) = √ u 0 /α 2 0 +4 β 4 ∑ k j =0 α j δ = O ( k 1 / 2 -ξ/ 2 ) δ 1 / 2 , we get the result.

## B.7 Proof of Theorem 5.2

The idea of this proof is borrowed from the example in [14, Example 2.2].

Consider the SA presented in Equation (1.1) for a 1 -dimensional linear setting with F ( x, w ) = wx . In this case, let { w k +1 } k ≥ 0 be an i.i.d. sequence of real-valued random variables such that P ( w k = a + N ) = 1 / ( N +1) and P ( w k = a -1) = N/ ( N +1) , where a ∈ (0 , 1) and N ≥ 1 are tunable parameters. Note that the update equation can be equivalently written as

<!-- formula-not-decoded -->

In this example, we have x k ∈ R + for all k ≥ 0 and x ∗ = 0 . To begin with, there exists k 0 &gt; 0 such that

<!-- formula-not-decoded -->

As a result, we have for any λ &gt; 0 and k large enough that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, by Ville's maximal inequality, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By a change of variables, we get

<!-- formula-not-decoded -->

(for h large enough.)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( a ) is by lower bounding expectation with one of its terms, ( b ) is by a + N -1 &gt; 0 , ( c ) is by Equation (B.11), ( d ) follows from

<!-- formula-not-decoded -->

- ( e ) is due to the fact that for any r &gt; 0 , e r ≥ r m /m ! for all m ∈ N , ( f ) is due to picking k 1 ≥ k 0 +1 such that

<!-- formula-not-decoded -->

- ( g ) is due to an integral lower bound similar to Equation (B.12), and ( h ) is due to Bernoulli's inequality. Therefore, for any ˜ β &gt; 0 and ˜ β ′ &gt; 0 , we have

<!-- formula-not-decoded -->

As a result, according to Lemma [14, Lemma 4.1], there do not exist ¯ K ′ 1 , ¯ K ′ 2 &gt; 0 such that

<!-- formula-not-decoded -->

A change of variables finishes the proof.

## B.8 Proof of Theorem 6.1

First, we verify that the assumptions of Theorem 5.1 are satisfied.

- Assumption 5.1: Firstly, for tabular TD ( n ) algorithm we assumed that V k ( s ) ≤ R max / (1 -γ ) for all s ∈ S and k ≥ 0 . Furthermore, TD ( n ) can be written in the form of (1.1) with

<!-- formula-not-decoded -->

where w k +1 = [( S i k , A i k )] 0 ≤ i ≤ n . In addition, we have

<!-- formula-not-decoded -->

where the matrix P π is the transition probability of the Markov chain following policy π , i.e., we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where M π is a diagonal matrix with diagonal entries µ π , and T π is the Bellman operator. Hence, we can write

<!-- formula-not-decoded -->

Furthermore, we have

Moreover, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, consider the matrix B π ∈ ❘ |S|×|S| , where B π ss ′ = (1 -γ n ) µ π ( s ) / |S| . Note that C π = A π + B π is a stochastic matrix, and its corresponding stationary distribution is ν π such that ( ν π ) ⊤ C π = ( ν π ) ⊤ . Further, we have

<!-- formula-not-decoded -->

where µ min = min s { µ π ( s ) } . Let us pick an arbitrary p ≥ 2 , and denote ∥ · ∥ ν π ,p as the weighted p -norm with weights ν π . We then have

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

Therefore, ¯ F ( V ) is a contraction with respect to ∥ · ∥ ν π ,p with contraction factor γ c = [1 -(1 -γ n ) µ π min ] 1 -1 /p , and hence Assumption 5.1 is satisfied.

- Assumption 5.2: Since the state and action space are both bounded, the noise in the update of TD-learning is also bounded. Hence, Equation (5.1) is satisfied for some σ &gt; 0 by Hoeffding's lemma. In addition, Equation (5.2) is satisfied due to Lemma C.2.
- Assumption 4.1: This assumption is satisfied with M = p -1 due to Lemma C.4.
- Assumption 4.2: Since ¯ F is a contraction, Banach's fixed point theorem implies it has a unique fixed point, and Lemma C.3 implies that J ¯ F ( x ∗ ) -I is invertible. Moreover, since F is linear, it is (0 , ∞ ) -locally psuedo smooth.
- Assumption 4.3: We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(Hölder's inequality)

<!-- formula-not-decoded -->

where in the last inequality we used the fact the

<!-- formula-not-decoded -->

where q is such that 1 /p +1 /q = 1 , and ν π min = min s { ν π ( s ) } . It follows that

<!-- formula-not-decoded -->

(Hoffding's lemma)

<!-- formula-not-decoded -->

Therefore, the first part of Assumption 4.3 is satisfied with

<!-- formula-not-decoded -->

On the other hand, by the definition in (B.13), we have F ( V, w k +1 ) = A ( w k +1 ) V + b ( w k +1 ) . Hence, J F w ( V π , w k +1 ) = A ( w k +1 ) . Since we assume finite state and action spaces, and w k +1 can only take finitely many values, therefore, by Hoeffding's lemma there exist ˆ σ &lt; ∞ such that the second part of Assumption 4.3 holds.

Since ∥ x ∥ ν π ,p ≤ ∥ x ∥ p ≤ ∥ x ∥ 2 , we have u c 2 = 1 . By a direct application of Theorem 5.1, we have with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Moreover, we have

<!-- formula-not-decoded -->

Hence, with probability 1 -δ , we have

<!-- formula-not-decoded -->

This result holds for any p . In particular, by choosing

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

In addition, we have

<!-- formula-not-decoded -->

where in the last inequality we used the fact that for all 0 &lt; a &lt; 1 , we have

<!-- formula-not-decoded -->

for all x ≤ (1 -a ) / (2 log(1 /a )) , and we take x = 1 /p and a = 1 -(1 -γ n ) µ π min . Substituting equations (B.17) and (B.18) in Equation (B.16), we obtain

<!-- formula-not-decoded -->

## B.9 Proof of Theorem 6.2

First, we verify that the assumptions of Theorem 5.1 are satisfied.

- Assumption 5.1: Firstly, for Q -learning algorithm it is known that Q k ( s, a ) ≤ R max / (1 -γ ) for all s ∈ S and k ≥ 0 . Furthermore, Q -learning can be written in the form of Equation (1.1) with

<!-- formula-not-decoded -->

where w k +1 = ( S k , A k , S ′ k ) . In addition, we have

<!-- formula-not-decoded -->

Furthermore, we have

<!-- formula-not-decoded -->

where M π b is a diagonal matrix with diagonal entries µ π b ( s ) π b ( a | s ) , and H is the Bellman optimality operator.

Next, we have

<!-- formula-not-decoded -->

where in the second inequality we used that the Bellman optimality operator is a γ -contraction. It follows that

<!-- formula-not-decoded -->

Hence, we can write

<!-- formula-not-decoded -->

Then, for any

<!-- formula-not-decoded -->

Assumption 5.1 is satisfied with γ c = ( |S||A| ) 1 /p (1 -(1 -γ ) ρ b ) .

- Assumption 5.2: Since the state and action space are both bounded, the noise in the update of Q -learning is also bounded. Then, Equation (5.1) is satisfied for some σ &gt; 0 by Hoeffding's lemma. In addition, Equation (5.2) is satisfied due to Lemma C.2.
- Assumption 4.1: This assumption is satisfied with M = p -1 due to Lemma C.4.
- Assumption 4.2: Since ¯ F is a contraction, Banach's fixed point theorem implies it has a unique fixed point Q ∗ , and Lemma C.3 implies that J ¯ F ( x ∗ ) -I is invertible. Moreover, Assumption 6.1 implies that F is linear in a neighborhood of Q ∗ , and thus there exists R &gt; 0 such that F is (0 , R ) -locally psuedo smooth with respect to the ∥ · ∥ p norm.
- Assumption 4.3: We have

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

(Hoffding's lemma)

Therefore, the first part of Assumption 4.3 is satisfied with ¯ σ 2 = 4 R 2 max / (1 -γ ) 2 . On the other hand, since we assume finite state and action spaces, and w k +1 can only take finitely many values, therefore, by Hoeffding's lemma there exist ˆ σ &lt; ∞ such that the second part of Assumption 4.3 holds.

Furthermore, since p ≥ 2 and ∥ x ∥ p ≤ ∥ x ∥ 2 , we have u c 2 = 1 . By a direct application of Theorem 5.1 we have that, with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Furthermore, we have

<!-- formula-not-decoded -->

Hence, with probability 1 -δ , we have

<!-- formula-not-decoded -->

This result holds for any p . In particular, by choosing p = p min ( k +1) 1 / 4 , we have

<!-- formula-not-decoded -->

where in the last inequality we used the fact that

<!-- formula-not-decoded -->

for all 0 ≤ x ≤ (1 -a ) / (2 a ln( b )) , 0 &lt; a &lt; 1 and b &gt; 1 , and we take x = 1 /p , a = (1 -(1 -γ ) ρ b ) and b = |S||A| . Substituting Equation (B.20) in Equation (B.19), we get

<!-- formula-not-decoded -->

## B.10 Proof of Theorem 6.3

The update of off-policy TD ( n ) can be written as follows:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

and ¯ α k = ξα k . Hence, off-policy TD ( n ) with linear function approximation can be written in the form of Equation (1.1) with F ( v k , w k +1 ) = A k v k + b k , where w k +1 = [( S i k , A i k )] 0 ≤ i ≤ n .

Next, we verify that the assumptions of Theorem 5.1 are satisfied

- Assumption 5.1: As shown in [11, Proposition 3.1] we have

<!-- formula-not-decoded -->

and for large enough n , we have that

<!-- formula-not-decoded -->

is a Hurwitz matrix. Hence, by [5, Page 46 footnote] there exists ζ large enough such that ¯ F ( · ) is a contraction with respect to some weighted 2-norm. We denote this norm as ∥ · ∥ c .

## · Assumption 5.3:

- (i) As mentioned earlier, ∥ · ∥ c is a weighted 2-norm.
- (ii) Since F is linear, and the state-action space is bounded, this assumption is satisfied for some L &gt; 0 .
- (iii) This assumption holds due to [52, Lemma 9]
- (iv) Since F is linear, and the state-action space is bounded, this assumption is satisfied for some τ &gt; 0 .
- (v) Since F is linear, and the state-action space is bounded, this assumption is satisfied for some Σ ⪰ 0 .
- Assumption 4.1: Since the norm ∥ · ∥ 2 c is a weighted 2 -norm, then it is smooth.
- Assumption 4.2: Since ¯ F is a contraction, Banach's fixed point theorem implies it has a unique fixed point, and Lemma C.3 implies that J ¯ F ( x ∗ ) -I is invertible. Moreover, since F is linear, it is (0 , ∞ ) -locally psuedo smooth.
- Assumption 4.3: Since the noise is bounded, this assumption is satisfied for some ¯ σ and ˆ σ .

Applying Theorem 5.1 yields the result.

## C Technical lemmas

Lemma C.1. Assume X 1 , X 2 , . . . is a series of martingale difference random vectors in ❘ d . In addition, assume X i to be η 2 i -subgaussian conditioned on filtration F i -1 generated by the random variables { X 1 , X 2 , . . . , X i -1 } . Then X 1 + X 2 + · · · + X n is a ∑ n i =1 η 2 i -subgaussian random vector.

Proof: We prove this by induction in the number of terms n . The case n = 1 is immediate. Now suppose that, for any d -dimensional vector v , we have

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

Lemma C.2. Suppose we have a random vector X ∈ ❘ d which is η 2 -subgaussian. Then, for all λ &lt; 1 / (2 η 2 u 2 c 2 ) , we have

<!-- formula-not-decoded -->

Proof: Since X is subgaussian, we have

<!-- formula-not-decoded -->

Hence, for any λ ∈ (0 , 1) , we have

<!-- formula-not-decoded -->

We now integrate both sides with respect to the vector v . Integrating the left-hand side, we get

<!-- formula-not-decoded -->

where ( a ) is due to Fubini's theorem for non negative functions.

Integrating the right-hand side of Equation (C.2), we get

<!-- formula-not-decoded -->

Combining equations (C.3), (C.4), and (C.2), we obtain

<!-- formula-not-decoded -->

By a change of variable, we get the result.

Lemma C.3. Consider a differentiable operator ¯ F : R d → R d that is a γ c -pseudo-contraction with respect to some norm ∥ · ∥ c , with fixed point x ∗ . Then, we have

<!-- formula-not-decoded -->

Proof: We have

<!-- formula-not-decoded -->

where by the definition of the Jacobian, we used the fact that ∥ ˜ h ( y ) ∥ c = o ( ∥ y ∥ c ) / ∥ y ∥ c . By taking the supremum over all vectors y ∈ R d , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

Note that which is attained at some ˜ y (because it is equivalent to maximizing a continuous function over a compact set). Moreover, note that for every constant r = 0 , the vector r ˜ y also attains the maximum above. Let y n = ˜ y/n . Then,

<!-- formula-not-decoded -->

It follows that and thus

Hence,

<!-- formula-not-decoded -->

Lemma C.4. Let p ≥ 2 , q = p/ ( p -1) , and consider the weights w = ( w 1 , . . . , w d ) ∈ R d &gt; 0 . Define the weighted norms

<!-- formula-not-decoded -->

where w # i := w -1 / ( p -1) i . Let f ( x ) = 1 2 ∥ x ∥ 2 p,w . Then, for all x, y ∈ R d , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof: Introduce the diagonal map T : R d → R d such that T ( x ) := Wx , where W is a diagonal matrix with that i -th diagonal entry equal to w 1 /p i . Then, we have

<!-- formula-not-decoded -->

Let g ( z ) = 1 2 ∥ z ∥ 2 p . Since f = g ◦ T and W is diagonal, by the chain rule, we have

<!-- formula-not-decoded -->

On the other hand, as shown in [2, Example 5.11], for p ≥ 2 we have

<!-- formula-not-decoded -->

Fix x, y , and set z = T ( x ) , z ′ = T ( y ) . We have

<!-- formula-not-decoded -->

The chain above yields the desired Lipschitz constant L = p -1 :

<!-- formula-not-decoded -->