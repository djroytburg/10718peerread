## Statistical Guarantees for High-Dimensional Stochastic Gradient Descent

## Jiaqi Li

Department of Statistics University of Chicago Chicago, IL 60637 jqli@uchicago.edu

## Johannes Schmidt-Hieber

Department of Applied Mathematics University of Twente Enschede, Netherlands a.j.schmidt-hieber@utwente.nl

## Abstract

Stochastic Gradient Descent (SGD) and its Ruppert-Polyak averaged variant (ASGD) lie at the heart of modern large-scale learning, yet their theoretical properties in high-dimensional settings are rarely understood. In this paper, we provide rigorous statistical guarantees for constant learning-rate SGD and ASGD in high-dimensional regimes. Our key innovation is to transfer powerful tools from high-dimensional time series to online learning. Specifically, by viewing SGD as a nonlinear autoregressive process and adapting existing coupling techniques, we prove the geometric-moment contraction of high-dimensional SGD for constant learning rates, thereby establishing asymptotic stationarity of the iterates. Building on this, we derive the q -th moment convergence of SGD and ASGD for any q ≥ 2 in general ℓ s -norms, and, in particular, the ℓ ∞ -norm that is frequently adopted in high-dimensional sparse or structured models. Furthermore, we provide sharp high-probability concentration analysis which entails the probabilistic bound of high-dimensional ASGD. Beyond closing a critical gap in SGD theory, our proposed framework offers a novel toolkit for analyzing a broad class of high-dimensional learning algorithms.

## 1 Introduction

Stochastic gradient descent (SGD) has been a cornerstone in large-scale machine learning since the seminal work by Robbins and Monro [1951]. It is especially efficient in high-dimensional and overparameterized settings where the number of unknown parameters can exceed the number of training samples [Arpit et al., 2017, Zhang et al., 2017, He et al., 2016]. SGD can also be combined with regularization techniques such as dropout to prevent overfitting in large networks [Krizhevsky et al., 2012, Srivastava et al., 2014]. Despite the vast amount of theoretical work on SGD, generalization bounds of SGD in high-dimensional regimes remain limited [Garrigos and Gower, 2023]. Considering a strongly convex objective function, we provide statistical guarantees for constant learning-rate SGD and its Ruppert-Polyak averaged variant (ASGD) [Ruppert, 1988, Polyak and Juditsky, 1992] in high-dimensional settings.

## Zhipeng Lou

Department of Mathematics University of California, San Diego La Jolla, CA 92093 zlou@ucsd.edu

## Wei Biao Wu

Department of Statistics University of Chicago Chicago, IL 60637 wbwu@uchicago.edu

Specifically, we consider a general optimization problem

<!-- formula-not-decoded -->

g ( · ) is the noise-perturbed measurement of G ( · ) , and ξ denotes a random element sampled from some unknown distribution Π . Given i.i.d. random samples ξ 1 , ξ 2 , . . . and some initialization β 0 ∈ R d , the k -th SGD iteration is

<!-- formula-not-decoded -->

for some constant learning rate α &gt; 0 , and ∇ g ( β , ξ ) = ∇ β g ( β , ξ ) the stochastic gradient with respect to β . For k ≥ 1 , the ASGD variant is defined by

<!-- formula-not-decoded -->

We are interested in the high-dimensional setting where the parameter dimension d can be very large. Here, a notable divide between empirical success and theoretical understanding is that practitioners often employ a large constant learning rate α in (2) to accelerate convergence in high-dimensional problems [Wu et al., 2018, Cohen et al., 2021, Cai et al., 2024]. However, such choices can induce pronounced non-stationarity in the SGD iterates { β k } k ∈ N which will not converge to a point but oscillates around the mean of a stationary distribution. In other words, β k is non-stationary but asymptotically stationary, which converges only in distribution as k →∞ , while the mean of this distribution differs from the exact minimizer β ∗ due to the non-diminishing bias of order O ( α ) [Dieuleveut et al., 2020, Merad and Gaïffas, 2023]. Classical theory mostly relies on decaying learning rates [Zhang, 2004, Nemirovski et al., 2009, Jentzen and von Wurstemberger, 2020, Shi et al., 2023]. To address the non-stationarity issue, we apply powerful tools from nonlinear time series analysis [Wu and Shao, 2004] to online learning, particularly by adapting the coupling techniques to show the geometric-moment contraction of SGD for constant learning rates. Specifically, for any two SGD sequences { β k } k ∈ N and { β ′ k } k ∈ N that share the same random samples but have different initial vectors β 0 and β ′ 0 , we show in Theorem 1 that for all sufficiently small constant learning rates α , the initialization is forgotten exponentially fast in the sense that

<!-- formula-not-decoded -->

for contraction speed 0 ≤ r α,s,q &lt; 1 , and | · | s the ℓ s -norm, that is,

<!-- formula-not-decoded -->

This asserts the existence of a limiting stationary distribution of β k as k →∞ , thereby facilitating a systematic convergence theory of SGD even in nonlinear, overparameterized models.

Building on this new framework, we provide non-asymptotic bounds for higher-order moments of the SGD error in general ℓ s -norms for any finite s ≥ 2 beyond the usual ℓ 2 -norm, extendable to max-norm ℓ ∞ by choosing s ≈ log( d ) . Notably, the ℓ ∞ -norm is frequently adopted in highdimensional sparse or structured estimation [Wainwright, 2019]. See for instance, the max-norm convergence of the Lasso and Dantzig selector [Lounici, 2008]; the pivotal method for sup-norm bounds of the square-root Lasso [Belloni et al., 2011]; and the max-norm error control for confidence intervals in high-dimensional regression problems [Javanmard and Montanari, 2013]. In stochasticapproximation (SA), Wainwright [2019] derived ℓ ∞ -norm bounds for Q-learning with decaying learning rates; Chen et al. [2023] derived maximal concentration bounds for SA under arbitrary norms with decaying learning rates and with contraction as an assumption; Agarwal et al. [2012] considered high-dimensional SA for strongly convex objectives with a sparse optimum, but using decaying learning rates and restricting the tails of stochastic gradients to be sub-Gaussian. To date, all the existing results are restricted to low-dimensional settings or decaying learning rates and do not carry over to overparameterized models with constant learning rates. To address this gap, we derive a sharp high-dimensional moment inequality (see Lemma 2) valid for a broad class of learning problems, delivering explicit non-asymptotic bounds of E | β k -β ∗ | q s and its ASGD variant for any q, s ≥ 2 with mild conditions, together with matching complexity guarantees, i.e., given some target error ε &gt; 0 (see Proposition 2), the required number of iterations k such that

<!-- formula-not-decoded -->

Although moment bounds capture average-case performance, a single execution of (A)SGD in practice demands high-probability guarantees [Valiant, 1984, Vapnik, 2000, Bach and Moulines, 2013, Durmus et al., 2021, Zhong et al., 2024]. Recent advances include a generic high-probability framework for both convex and nonconvex SGD with sub-Gaussian gradient noises [Liu et al., 2023], high-probability rates for clipped-SGD with heavy-tailed noises [Nguyen et al., 2023], and high-probability guarantees for nonconvex stochastic approximation via robust gradient clipping [Li and Liu, 2022]. However, these established high-probability bounds focus again on decaying learning rates and low dimension. Moreover, early work primarily addressed light-tailed noises where the gradients are bounded or have exponential-type moments [Nemirovski et al., 2009, Rakhlin et al., 2012, Ghadimi and Lan, 2013, Cardot et al., 2017, Harvey et al., 2019, Mou et al., 2020, Chen et al., 2023]. For the cases that only admit a polynomial tail with finite q -th moment, Lou et al. [2022] were the first to derive a Nagaev-type inequality [Nagaev, 1979] for low-dimensional ASGD. The rate was shown to be optimal but their bound heavily relies on the linearity of gradients and is only suitable for decaying learning rates. By leveraging a dependency-adjusted functional dependence measure in high-dimensional time series [Zhang and Wu, 2017], we derive a high-probability concentration bounds for high-dimensional ASGD with constant learning rates. Given a tolerance level δ ∈ (0 , 1) and a target error ε &gt; 0 , we provide bounds for the required number of iterations k to guarantee that

<!-- formula-not-decoded -->

This tail-decay result (see Eq. (10)) is proved via a new Fuk-Nagaev-type inequality (see Theorem 4) and complements our moment and complexity characterizations of large-step stochastic optimization.

## 1.1 Our Contributions

This paper contributes to theoretical advancements for understanding constant learning-rate SGD and its averaged variant (ASGD) in the challenging high-dimensional regime. Our main technical innovations and results include:

(1) Handling Constant Learning Rates in High Dimensions. In practice, large-scale machine learning models commonly deploy fixed, large learning rates to speed up optimization in highdimensional settings. To address this, we introduce novel coupling techniques inspired by highdimensional nonlinear time series and establish the asymptotic stationarity of the SGD iterates with arbitrary initialization (Section 2) .

(2) Generalized Moment Convergence in ℓ s - and ℓ ∞ -Norms. By deriving a sharp high-dimensional moment inequality, we establish explicit, non-asymptotic q -th moment bounds for arbitrary ℓ s -norms of (A)SGD iterates for any q ≥ 2 and even integers s , generalizing previous theory primarily focusing on mean squared error (MSE) convergence with q = s = 2 . Our results extend naturally to the max-norm case (i.e., ℓ ∞ ) by selecting s ≈ log( d ) , that is essential for modern sparse and structured estimation in high-dimensional data (Section 3) .

(3) High-Probability Tail Bounds. While average-case (moment) bounds are informative, single runs require tail guarantees. We derive the first high-probability concentration bounds for ASGD in highdimensional settings with constant learning rates. By developing a tight Fuk-Nagaev-type inequality using the coupling techniques in nonlinear time series, we control the algorithmic complexity required to achieve targeted accuracy with high confidence (Section 4) .

## 1.2 Related Works

Stochastic Gradient Descent and its Variants. The SGD algorithm can be traced back to Robbins and Monro [1951], Kiefer and Wolfowitz [1952]. Popular SGD variants include Nesterov's accelerated gradient [Nesterov, 1983], AdaGrad [Duchi et al., 2011], AdaDelta [Zeiler, 2012], Adam [Kingma and Ba, 2014], AMSGrad [Reddi et al., 2018], AdamW [Loshchilov and Hutter, 2018], SAG [Schmidt et al., 2017], SVRG [Johnson and Zhang, 2013], SARAH [Nguyen et al., 2017], SPIDER [Fang et al., 2018] and Katyusha [Allen-Zhu, 2017]. The theoretical foundations of SGD under decaying learning rates were established in the early studies by [Blum, 1954, Dvoretzky, 1956, Sacks, 1958], with stronger almost-sure guarantees by Fabian [1968], Robbins and Siegmund [1971], Ljung [1977], Lai [2003], Wang and Gao [2010]. Existing works for smooth, strongly-convex objectives with decaying step sizes include Ruppert [1988], Polyak and Juditsky [1992], Nemirovski et al. [2009], Bach and Moulines [2013], Rakhlin et al. [2012], Mertikopoulos et al. [2020] among others. Despite the rich literature on SGD, the theoretical understanding in high-dimensional settings remains limited.

Exceptions are Paquette et al. [2021, 2022] who study high-dimensional SGD for the least-squares loss.

Constant Learning Rate. In high-dimensional scenarios, constant learning rates prevail due to simpler tuning procedures and faster convergence [Wang et al., 2022]. More recent theoretical and empirical studies of large-step SGD include Wu et al. [2018], Cohen et al. [2021] and the very recent Cai et al. [2024], which formalize the resurgence of constant-step methods in modern machine learning. A useful way to analyze constant-step SGD is to treat its iterates as a time-homogeneous Markov chain [Pflug, 1986], which makes it possible to characterize its long-run behavior and stationary law. However, previous works only derived convergence in Wasserstein distance [Dieuleveut et al., 2020, Merad and Gaïffas, 2023]. Such convergence in probability measures can hardly provide refined (non)-asymptotics such as higher-moment convergence and concentration inequalities, and seems nontrivial to extend to high-dimensional regimes.

High-Dimensional Nonlinear Time Series. An alternative approach for constant learning-rate SGD is to view it as an iterated random function [Dubins and Freedman, 1966, Barnsley and Demko, 1985, Diaconis and Freedman, 1999, Diaconis and Duflo, 2000], or a nonlinear autoregressive (AR) process. This interpretation facilitates the theory of online learning with non-stationarity and complex dependency structures; see, for example, the recent work by Li et al. [2024c] on SGD with dropout regularization building on the GMC framework [Wu and Shao, 2004]. To extend this systematic theory to high-dimensional settings, we adapt the coupling techniques in time series [Wu, 2005, 2007, 2009, 2011, Xiao and Wu, 2012, Berkes et al., 2014, Wu and Wu, 2016, Karmakar and Wu, 2020], especially the ones for high-dimensional regimes [Zhang and Wu, 2017, 2021, Li et al., 2024a] to online learning algorithms.

## 1.3 Notation

Denote column vectors in R d by lowercase bold letters x = ( x 1 , . . . , x d ) ⊤ and the ℓ s -norm of x by | x | s = ( ∑ d i =1 | x i | s ) 1 /s , s ≥ 1 . Write x ⊙ s = ( x s 1 , . . . , x s d ) ⊤ . The expectation and covariance of random vectors are respectively denoted by E [ · ] and Cov( · ) . For q &gt; 0 and a random variable X , we write X ∈ L q iff ∥ X ∥ q = [ E ( | X | q )] 1 /q &lt; ∞ . We denote matrices by uppercase letters. Given matrices A and B of compatible dimension, their matrix product is denoted by juxtaposition. Write A ⊤ for the transpose of A and I d for d × d identity matrix. For two positive number sequences ( a n ) and ( b n ) , we say a n = O ( b n ) (resp. a n ≍ b n ) if there exists c &gt; 0 such that a n /b n ≤ c (resp. 1 /c ≤ a n /b n ≤ c ) for all large n . Let ( x n ) and ( y n ) be two sequences of random variables. Write x n = O P ( y n ) if for ∀ ϵ &gt; 0 , there exists c &gt; 0 such that P ( | x n /y n | ≤ c ) &gt; 1 -ϵ for all large n .

Table 1: List of the sequences defined in the paper.

| Notation   | Definition                           | Reference   | Index Range   |
|------------|--------------------------------------|-------------|---------------|
| β ∗        | minimizer of the loss function G ( β | Eq. (1)     | /             |
| β k        | SGD iterates                         | Eq. (2)     | k ∈ N         |
| β ◦ k      | stationary SGD iterates              | Thm. (2)    | k ∈ Z         |
| ¯ β k      | ASGD iterates                        | Eq. (3)     | k ∈ N         |
| ¯ β ◦ k    | stationary ASGD iterates             | Eq. (9)     | k ∈ Z         |

## 2 Convergence of SGD to a Stationary Distribution

In this section, we establish the GMC property of high-dimensional SGD with constant learning rates. Our technique is to construct a smooth surrogate for the non-differentiable ℓ ∞ -norm via the ℓ s -norm, so that standard gradient-based tools become available. We defer the technical details to Section 6.1. Furthermore, we provide a novel high-dimensional moment inequality (see Section 6.2) and use it to derive the dimension-dependent range of the constant learning rate that guarantees the contraction.

We first impose the following assumptions on the objective function and the stochastic gradients.

Assumption 1 (Coercivity) . Assume that for any sequence β 1 , β 2 , . . . with | β n | s → ∞ the loss function G ( · ) in (1) satisfies lim n →∞ G ( β n ) = ∞ .

Assumption 2 (Strong Convexity ℓ s -norm) . Let s ≥ 2 be an even integer and write v ⊙ s := ( v s 1 , . . . , v s d ) ⊤ for a vector v = ( v 1 , . . . , v d ) ⊤ . Assume there exists µ &gt; 0 such that

<!-- formula-not-decoded -->

In Lemma 3 in the supplementary materials, we show that under Assumptions 1 and 2, a unique global minimizer β ∗ exists for the optimization problem (1). When s = 2 , Assumption 2 reduces to the regular strong convexity frequently adopted in the literature [Polyak and Juditsky, 1992, Moulines and Bach, 2011, Dieuleveut et al., 2020, Mies and Steland, 2023]. For general s and the linear regression model, Section 8.2 in the supplementary material interprets the ℓ s -type strong convexity assumption via the ℓ s -norm induced matrix norm. As different norms are involved, there does not seem to be an apparent relationship between the classical strong convexity and the case s &gt; 2 .

Assumption 3 (Stochastic Lipschitz Continuity ℓ s -norm) . Let β ∗ be the global minimizer. For some q ≥ 2 and an even integer s ≥ 2 , assume that

<!-- formula-not-decoded -->

Further assume there exists a constant L s,q &gt; 0 such that

<!-- formula-not-decoded -->

Later we will choose s = O (log( d )) to bound the max-norm. The above defined Lipschitz constant L s,q and the moments M s,q will then grow as d increases. Taking linear regression as an example, we investigate the dimension dependence of L s,q and M s,q in Section 8.2. All bounds in this work will contain the explicit dependence on ( L s,q , M s,q ) .

We now state the first main result of this paper, which plays a crucial role in establishing moment convergence and tail probability results in the following sections. The statement quantifies the exponential rate at which the initialization β 0 will be forgotten and the SGD iterates β k converges to a stationary distribution π α .

Theorem 1 (Convergence of SGD to stationary distribution) . Suppose that Assumptions 1-3 hold for some µ &gt; 0 , q ≥ 2 and even integer s ≥ 2 . Given a constant learning rate

<!-- formula-not-decoded -->

for any two d -dimensional SGD sequences { β k ( α ) } k ∈ N and { β ′ k ( α ) } k ∈ N sharing the same i.i.d. noise injections { ξ k } k ≥ 1 but possibly different initializations β 0 , β ′ 0 ∈ R d , the geometric-moment contraction (GMC)

<!-- formula-not-decoded -->

holds with contraction constant

<!-- formula-not-decoded -->

Moreover, there exists a unique stationary distribution π α with a finite q -th moment, that is, ∫ | u | q s π α ( d u ) &lt; ∞ , such that

<!-- formula-not-decoded -->

Equivalently, for any continuous function f ∈ C ( R d ) with | f | ∞ &lt; ∞ ,

<!-- formula-not-decoded -->

The result generalizes Li et al. [2024b] to large dimension d and extends the ℓ 2 -type GMC based on Lemma 9 to general ℓ s -norms. Moreover, choosing s = s d with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

shows the equivalence of the ℓ s d - and ℓ ∞ -norms. Consequently, by choosing s = s d , the previous theorem can also be used to derive the GMC property with respect to the ℓ ∞ -norm.

and using the inequality

## 3 Convergence of High-Dimensional SGD and ASGD

In this section, we derive convergence rates for the moments of the last iterate E | β k -β ∗ | q ∞ and the moments of the averaged SGD.

## 3.1 Convergence of SGD

Proposition 1. If Assumptions 1-3 hold for some q ≥ 2 , an even integer s ≥ 2 , and a constant M s,q , then,

<!-- formula-not-decoded -->

for all k ≥ 1 . The same inequality holds if β k is replaced by the stationary SGD iterates β ◦ k ∼ π α , k ≥ 1 .

Theorem 2 (Moment convergence of SGD) . Let 0 &lt; α &lt; α s,q / 7 with α s,q as defined in (5) . Suppose that Assumptions 1-3 hold for q ≥ 2 and even integer s ≥ 2 . Then for the stationary SGD iterates β ◦ k ∼ π α ,

<!-- formula-not-decoded -->

and for the SGD iterate β k with arbitrary initialization β 0 ,

<!-- formula-not-decoded -->

Choosing s = s d in (7) yields a bound with respect to the ℓ ∞ -norm.

## 3.2 Convergence of Ruppert-Polyak Averaged SGD

Consider now the Ruppert-Polyak Averaged SGD (ASGD) ¯ β k = 1 k ∑ k i =1 β i as defined in (3). For the initialization β ◦ 0 ∼ π α , define the stationary ASGD sequence

<!-- formula-not-decoded -->

Theorem 3. Consider the ASGD sequence { ¯ β k } k ≥ 1 . Suppose that Assumptions 1-3 hold with some q ≥ 2 and even integer s = s d in (7) , the conditions of Theorem 8 hold and the learning rate satisfies α ∈ (0 , α s d ,q ) with α s d ,q defined in (5) . For any k ≥ 1 and some universal constants C 1 , C 2 , C 3 &gt; 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have ∥| ¯ β k -β ∗ | ∞ ∥ q ≤ ε.

A proof outline is given in Section 6.3 and the full proof is deferred to the supplementary material. The sharpest complexity bound of SA for ℓ ∞ -norm known to date was derived by Wainwright [2019] proving that the number of iterations required to obtain an ε -accurate solution of Q-learning scales as (1 -γ ) -4 · ε -2 with the discount factor γ . In Proposition 2, our complexity bound for SGD is also of the order of O (1 /ε 2 ) if the dimension d is fixed, which is consistent with the degenerate Q-learning case in Wainwright [2019]. The derived result allows to determine the dependence on the dimension d .

## 4 Sharp Concentration and Gaussian Approximation

Via the following tail probability inequality for the averaged SGD estimator ¯ β k , one can further derive high-probability concentration bound of | ¯ β k -β ∗ | ∞ . Recall that s d = 2min { ℓ ∈ N : 2 ℓ &gt; log( d ) } . Theorem 4 (Fuk-Nagaev inequality) . Under the conditions of Theorem 3, for any z &gt; 0 , we have

<!-- formula-not-decoded -->

where the constants in ≲ are independent of k, d, s and α .

As an immediate consequence of Theorem 4, we obtain a sharp high-probability upper bound for | ¯ β k -β ∗ | ∞ , that is, for any given tolerance rate δ ∈ (0 , 1) , with at least probability 1 -δ , we have

<!-- formula-not-decoded -->

Notably, if the q -th moment of the gradient noise is finite ( M s d ,q &lt; ∞ ), the second term of the right hand side, involving k 1 -q , is generally unimprovable [Nagaev, 1979, Lou et al., 2022].

The distribution convergence for the high-dimensional ASGD relies on the following result. Let M 2 ,q be as defined in Assumption 3.

Theorem 5 (Gaussian approximation) . Consider stationary SGD iterates β ◦ k ∼ π α with π α as defined in Theorem 1, initialization β ◦ 0 ∼ π α , and learning rate α ∈ (0 , α s d ,q ) . Suppose that Assumptions 1-3 hold for some q &gt; 2 . Then, on a potentially different probability space, and for a number of iterations T satisfying d ≤ cT , where c &gt; 0 is some constant, there exist random vectors { ˜ β k } T k =1 D = { β ◦ k } T k =1 and independent Gaussian random vectors { z k } T k =1 with mean zero and covariance matrix

<!-- formula-not-decoded -->

such that

<!-- formula-not-decoded -->

with C ∗ α,q a constant that only depends on c , the learning rate α , and the moment index q.

For diverging moment index q → ∞ , the Gaussian approximation rate in (12) approaches the rate O ( √ log( T )( d 4 /T ) 1 / 6 ) . Thus, to obtain a nontrivial Gaussian approximation bound within T iterations, we need dimension dependence d = o ( T 1 / 4 -ζ ) with ζ &gt; 0 .

## 5 Constant Learning Rate for Large Dimension

Recall that L s,q is the Lipschitz constant introduced in Assumption 3. We established asymptotic stationarity and non-asymptotic convergence if α &lt; α s,q / 7 with α s,q defined in (5), leading to the upper bound

<!-- formula-not-decoded -->

if we choose s = s d in (7) and if L s d ,q ≍ d . We refer to Section 8.2 for the derivation of the dimension dependence of L s,q in the linear regression model.

Alternatively, the upper bound for the learning rate α can also be derived by a linear approximation technique (see Lemma 1), defined as the nontrivial solution to the following equation

<!-- formula-not-decoded -->

A derivation of this equation is provided in Section 6.1. The existence of a solution of (13) is shown below the proof of Lemma 1 in the supplementary materials. When q = 2 , the range of α simplifies to

<!-- formula-not-decoded -->

which is also proportional to 1 / [ d 2 log( d )] if we choose s = s d in (7) and if L s d , 2 ≍ d , matching the rate of α s,q in (5) derived by Lemma 2, though with a slightly more conservative constant for general s . In the special case with s = 2 , both bounds reduce to the classical α &lt; 2 µ/L 2 2 , 2 . If L 2 , 2 ≍ d for large dimension d , which is shown to be true for the linear regression model in Section 8.2 in the supplementary materials, the ℓ ∞ - and the ℓ 2 -norm yield similar upper bounds for the learning rate α.

## 6 Proof Sketches

## 6.1 Bridge between ℓ s - and ℓ ∞ - Norms

In high-dimensional regimes, convergence rates of constant-learning-rate SGD (2) with respect to the ℓ ∞ -norm are of particular interest [Wainwright, 2019, Chen et al., 2023]. However, it is extremely challenging to directly study the convergence of | β k -β ∗ | ∞ since the ℓ ∞ -norm is not differentiable thereby ruling out standard gradient-based tools for proving convergence rates or concentration. To address this issue, we instead study | · | s d with s d defined in (7). By the equivalence between ℓ s d -and ℓ ∞ -norms shown in (8), contraction in ℓ ∞ -norm follows from ℓ s d -norm contraction.

To prove the GMC property of SGD as introduced in (4), it suffices to show that for any two d -dimensional SGD sequences { β k } k ∈ N and { β ′ k } k ∈ N sharing the same i.i.d. observations { ξ k } k ≥ 1 but possibly different initializations β 0 , β ′ 0 ∈ R d , the contraction holds for | β k -β ′ k | s d for all k ≥ 1 . To this end, we need to determine a range of constant learning rates α such that for any q ≥ 2 and β , β ′ ∈ R d , the GMC in Theorem 1 holds, i.e.,

<!-- formula-not-decoded -->

To derive the inequality, we first provide a lemma based on linear approximation by considering the scalar function

<!-- formula-not-decoded -->

and linearizing it around α = 0 . Then, one only needs to prove that E | x -α z | q s ≤ r | x | q s . By the second-order Taylor expansion of | x -α z | q s in α , we have the linear approximation

<!-- formula-not-decoded -->

with remainder term of order α 2 , see Section 2 in the supplementary materials for details. Since a simple triangle inequality argument ∥| x -α z | s ∥ q ≤ ∥| x | s ∥ q + α ∥| z | s ∥ q fails to control this remainder sufficiently to yield a contraction constant r &lt; 1 , we establish a more precise bound.

Lemma 1. Recall that v ⊗ s = ( v s 1 , . . . , v s d ) ⊤ for a vector v = ( v 1 , . . . , v d ) ⊤ . For any q ≥ 2 , any even integer s ≥ 2 , any two vectors x , z ∈ R d , and any α &gt; 0 ,

<!-- formula-not-decoded -->

If s = 2 , q = 2 , the right-hand side is α 2 | z | 2 2 . This is consistent with the Taylor remainder of the right-hand side in Lemma 9 derived by Li et al. [2024c]. Using this inequality to prove the contraction in (14) is remarkably different from the approaches relying on the martingale decomposition (MD) that is frequently adopted in the literature [Dieuleveut et al., 2020, Mertikopoulos et al., 2020, Mies and Steland, 2023]. Our proposed method requires mild moment conditions on the stochastic gradients and yields simpler proofs that can be generalized to a broad class of online learning problems. We refer to Li et al. [2024b] for detailed discussion. Nevertheless, we remark in advance that a Rio-type inequality (Lemma 2) with slightly sharper constants will be used directly in our main contraction proof, while we retain Lemma 1 here for its intuitive appeal. Finally, by choosing s = s d as in (7) for (14), we can expect the ℓ ∞ -norm type GMC to hold for high-dimensional SGD iterates.

## 6.2 High-Dimensional Moment Inequality

To prove Theorem 1, we derive a high-dimensional version of Rio's inequality [Rio, 2009], adapted to the q -th moment of ℓ s -norm. This result provides a slightly sharper constant than Lemma 1 and is used directly in our moment-contraction analysis.

Lemma 2 (High-dimensional moment inequality) . For any q ≥ 2 , any even integer s ≥ 2 , and any two d -dimensional random vectors x , y , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, if E [ y | x ] = 0 almost surely, then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Repeatedly applying Lemma 2 leads to the high-dimensional maximal moment inequality in Lemma 8 in the supplementary materials, which is of independent interest.

## 6.3 Stationarity, Variation and Bias of ASGD

We prove the moment bound ∥| ¯ β k -β ∗ | ∞ ∥ q via the decomposition

<!-- formula-not-decoded -->

The first term accounts for the deviation due to the non-stationarity of ¯ β k as it is initialized from an arbitrarily fixed β 0 ; this can be bounded using the GMC property of β k shown in Theorem 1. The second term captures the stochastic variance of the stationary ASGD sequence. Bounding this term is more delicate because of the intricate dependency structure of ¯ β ◦ k . To address this, we deploy another powerful tool in time series - the functional dependence measure [Wu, 2005] in Section 8.6 of the supplementary materials, which can effectively quantify the contribution of the random sample ξ i to the k -th SGD iterate β ◦ k for all i ≤ k . As such, by controlling the cumulative dependence measures, we can bound this variance. Lastly, we handle the third term, which represents the non-diminishing bias of ¯ β ◦ k induced by the constant learning rate α [Dieuleveut et al., 2020, Huo et al., 2023]. This can be dealt with by extending the approach in Li et al. [2024b] to high-dimensional settings.

Theorem 6 (Asymptotic stationarity) . Consider the ASGD iterates ¯ β k and the stationary version ¯ β ◦ k . Suppose that Assumptions 1-3 are satisfied for some q ≥ 2 and some even integer s ≥ 2 . Then, for the learning rate α ∈ (0 , α s,q ) with α s,q defined in (5) ,

<!-- formula-not-decoded -->

As a direct consequence of Theorem 6, we have ∥| ¯ β k -¯ β ◦ k | s ∥ q ≲ | β 0 -β ◦ 0 | s / ( kα ) , which indicates the asymptotic stationarity of high-dimensional ASGD sequences. When the bias induced by the initialization is controlled, i.e., | β 0 -β ◦ 0 | s &lt; ∞ , as kα →∞ , the ASGD iterate ¯ β k approaches the stationary solution ¯ β ◦ k in the sense that ∥| ¯ β k -¯ β ◦ k | s ∥ q → 0 . By Theorem 6, we only need to show the convergence for stationary ASGD.

Theorem 7 (Stochasticity of stationary ASGD) . Consider the stationary SGD sequence { β ◦ k } k ≥ 1 . Suppose that Assumptions 1-3 hold with some q ≥ 2 and some even integer s ≥ 2 . Then there exists a constant c q &gt; 0 only depending on q , such that, for all k ≥ 1 ,

<!-- formula-not-decoded -->

In the low-dimensional case, we take s = 2 as a special example. Then, L s,q √ α max { q, s } = O (1) such that the bound is ∥| ¯ β ◦ k -E [ ¯ β ◦ k ] | s ∥ q = O { 1 / √ k } . This rate is optimal considering the central limit theorem of the stationary ASGD.

Next, we consider the bias induced by the constant learning rate. We first introduce some necessary notation. Recall G ( β ) = E [ ∇ g ( β , ξ )] and ∇ G ( β ) = ( ∂ 1 G ( β ) , . . . , ∂ d G ( β )) ⊤ , where β = ( β 1 , . . . , β d ) ⊤ . Denote ∂ i G ( β ) = ∂G ( β ) /∂β i , 1 ≤ i ≤ d ,

<!-- formula-not-decoded -->

We provide the non-asymptotic bound for the bias of stationary ASGD in the following lemma.

Theorem 8 (Bias of stationary ASGD) . Under Assumptions 1-3, consider the stationary ASGD ¯ β ◦ k . Assume that g ( β , ξ ) is twice differentiable with respect to β with positive definite Hessian matrix ∇ 2 G ( β ∗ ) , and uniformly bounded derivatives max 1 ≤ i ≤ d ∥∇ 3 G i ( β ) ∥ ∞ &lt; ∞ , where

<!-- formula-not-decoded -->

Then, we have

<!-- formula-not-decoded -->

## 7 Conclusions and Discussion

This work advances the theoretical understanding of the constant learning-rate SGD algorithms in high-dimensional settings. By introducing novel coupling techniques in nonlinear time series, we establish asymptotic stationarity of SGD with any initialization. We then derive non-asymptotic q -th moment bounds in general ℓ s - and ℓ ∞ -norms, and develop the first Fuk-Nagaev high-probability tail bound for ASGD. While this paper assumes strong convexity and smoothness of the objective, the nonlinear time series perspective offers a principled framework applicable to a broad class of overparameterized optimization tasks and can be extended to non-convex regimes, providing fundamental insights into the stability, convergence, and reliability of large-scale learning algorithms.

## Acknowledgments and Disclosure of Funding

We sincerely thank the program chair, senior area chair, area chair, and the four reviewers for their constructive feedback and involved discussion, which has greatly improved the clarity of our paper. Jiaqi Li's research is partially supported by the NSF (Grant NSF/DMS-2515926). Johannes Schmidt-Hieber has received funding from the Dutch Research Council (NWO) via the Vidi grant VI.Vidi.192.021. Wei Biao Wu's research is partially supported by the NSF (Grants NSF/DMS2311249, NSF/DMS-2027723). We would like to thank Insung Kong for helpful discussions.

## References

- A. Agarwal, S. Negahban, and M. J. Wainwright. Stochastic optimization and sparse statistical recovery: Optimal algorithms for high dimensions. In Advances in Neural Information Processing Systems , volume 25. Curran Associates, Inc., 2012.
- Z. Allen-Zhu. Katyusha: the first direct acceleration of stochastic gradient methods. In Proceedings of the 49th Annual ACM SIGACT Symposium on Theory of Computing , pages 1200-1205, Montreal Canada, 2017. ACM.
- D. Arpit, S. Jastrz˛ ebski, N. Ballas, D. Krueger, E. Bengio, M. S. Kanwal, T. Maharaj, A. Fischer, A. Courville, Y. Bengio, and S. Lacoste-Julien. A closer look at memorization in deep networks. In Proceedings of the 34th International Conference on Machine Learning , pages 233-242. PMLR, 2017.
- F. Bach and E. Moulines. Non-strongly-convex smooth stochastic approximation with convergence rate O(1/n). In Advances in Neural Information Processing Systems , volume 26. Curran Associates, Inc., 2013.
- M. F. Barnsley and S. Demko. Iterated function systems and the global construction of fractals. Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences , 399 (1817):243-275, 1985.
- A. Belloni, V. Chernozhukov, and L. Wang. Square-root lasso: pivotal recovery of sparse signals via conic programming. Biometrika , 98(4):791-806, 2011.
- I. Berkes, W. Liu, and W. B. Wu. Komlós-Major-Tusnády approximation under dependence. The Annals of Probability , 42(2):794-817, 2014.

- J. R. Blum. Approximation methods which converge with probability one. The Annals of Mathematical Statistics , 25(2):382-386, 1954.
- Y. Cai, J. Wu, S. Mei, M. Lindsey, and P. Bartlett. Large stepsize gradient descent for nonhomogeneous two-layer networks: Margin improvement and fast optimization. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- H. Cardot, P. Cénac, and A. Godichon-Baggioni. Online estimation of the geometric median in Hilbert spaces: Nonasymptotic confidence balls. The Annals of Statistics , 45(2):591-614, 2017.
- L. Chen, G. Keilbar, and W. B. Wu. Recursive quantile estimation: Non-asymptotic confidence bounds. Journal of Machine Learning Research , 24(91):1-25, 2023.
- Z. Chen, S. Theja Maguluri, and M. Zubeldia. Concentration of contractive stochastic approximation: Additive and multiplicative noise. arXiv e-prints , art. arXiv:2303.15740, 2023.
- V. Chernozhukov, D. Chetverikov, and K. Kato. Comparison and anti-concentration bounds for maxima of Gaussian random vectors. Probability Theory and Related Fields , pages 47-70, 2015.
- J. Cohen, S. Kaur, Y. Li, J. Z. Kolter, and A. Talwalkar. Gradient descent on neural networks typically occurs at the edge of stability. In International Conference on Learning Representations , 2021.
- P. Diaconis and M. Duflo. Random iterative models. In Journal of the American Statistical Association , volume 95, page 342, 2000.
- P. Diaconis and D. Freedman. Iterated random functions. SIAM Review , 41(1):45-76, 1999.
- A. Dieuleveut, A. Durmus, and F. Bach. Bridging the gap between constant step size stochastic gradient descent and Markov chains. The Annals of Statistics , 48(3):1348-1382, 2020.
- L. E. Dubins and D. A. Freedman. Invariant probabilities for certain Markov processes. The Annals of Mathematical Statistics , 37(4):837-848, 1966.
- J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research , 12:2121-2159, 2011.
- A. Durmus, E. Moulines, A. Naumov, S. Samsonov, K. Scaman, and H. T. Wai. Tight high probability bounds for linear stochastic approximation with fixed stepsize. In Advances in Neural Information Processing Systems , 2021.
- A. Dvoretzky. On stochastic approximation. In Proceedings of the Third Berkeley Symposium on Mathematical Statistics and Probability, Volume 1: Contributions to the Theory of Statistics , volume 3.1, pages 39-56. University of California Press, 1956.
- V. Fabian. On asymptotic normality in stochastic approximation. The Annals of Mathematical Statistics , 39(4):1327-1332, 1968.
- C. Fang, C. J. Li, Z. Lin, and T. Zhang. SPIDER: Near-optimal non-convex optimization via stochastic path-integrated differential estimator. Advances in Neural Information Processing Systems , 31, 2018.
- G. Garrigos and R. M. Gower. Handbook of convergence theorems for (stochastic) gradient methods. arXiv e-prints , art. arXiv:2301.11235, 2023. doi: 10.48550/arXiv.2301.11235.
- S. Ghadimi and G. Lan. Stochastic first- and zeroth-order methods for nonconvex stochastic programming. SIAM Journal on Optimization , 23(4):2341-2368, 2013.
- N. J. A. Harvey, C. Liaw, Y. Plan, and S. Randhawa. Tight analyses for non-smooth stochastic gradient descent. In Proceedings of the Thirty-Second Conference on Learning Theory , pages 1579-1613. PMLR, 2019.
- K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , pages 770-778, 2016.
- R. A. Horn and C. R. Johnson. Matrix Analysis . Cambridge University Press, Cambridge, 1985.

- D. Huo, Y. Chen, and Q. Xie. Bias and extrapolation in Markovian linear stochastic approximation with constant stepsizes. In Abstract Proceedings of the 2023 ACM SIGMETRICS International Conference on Measurement and Modeling of Computer Systems , pages 81-82, 2023.
- A. Javanmard and A. Montanari. Confidence intervals and hypothesis testing for high-dimensional statistical models. In Advances in Neural Information Processing Systems , volume 26. Curran Associates, Inc., 2013.
- A. Jentzen and P. von Wurstemberger. Lower error bounds for the stochastic gradient descent optimization algorithm: Sharp convergence rates for slowly and fast decaying learning rates. Journal of Complexity , 57:101438, 2020.
- R. Johnson and T. Zhang. Accelerating stochastic gradient descent using predictive variance reduction. Advances in Neural Information Processing Systems , 26, 2013.
- S. Karmakar and W. B. Wu. Optimal Gaussian approximation for multiple time series. Statistica Sinica , 30(3):1399-1417, 2020.
- J. Kiefer and J. Wolfowitz. Stochastic estimation of the maximum of a regression function. The Annals of Mathematical Statistics , 23(3):462-466, 1952.
- D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In 3rd International Conference on Learning Representations , 2014.
- A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems , volume 25. Curran Associates, Inc., 2012.
- T. L. Lai. Stochastic approximation: invited paper. The Annals of Statistics , 31(2):391-406, 2003.
- J. Li, L. Chen, W. Wang, and W. B. Wu. ℓ 2 inference for change points in high-dimensional time series via a two-way MOSUM. Ann. Statist. , 52(2):602-627, 2024a.
- J. Li, Z. Lou, S. Richter, and W. B. Wu. The stochastic gradient descent from a nonlinear time series persective, 2024b. Manuscript.
- J. Li, J. Schmidt-Hieber, and W. B. Wu. Asymptotics of stochastic gradient descent with dropout regularization in linear models. arXiv preprint , 2024c. arXiv:2409.07434.
- S. Li and Y. Liu. High probability guarantees for nonconvex stochastic gradient descent with heavy tails. In Proceedings of the 39th International Conference on Machine Learning , pages 12931-12963. PMLR, 2022.
- Z. Liu, T. D. Nguyen, T. H. Nguyen, A. Ene, and H. L. Nguyen. High probability convergence of stochastic gradient methods. In Proceedings of the 40th International Conference on Machine Learning , volume 202 of ICML'23 , pages 21884-21914, Honolulu, Hawaii, USA, 2023. JMLR.org.
- L. Ljung. Analysis of recursive stochastic algorithms. IEEE Transactions on Automatic Control , 22 (4):551-575, 1977.
- I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations , 2018.
- Z. Lou, W. Zhu, and W. B. Wu. Beyond sub-Gaussian noises: Sharp concentration analysis for stochastic gradient descent. Journal of Machine Learning Research , 23(46), 2022.
- K. Lounici. Sup-norm convergence rate and sign concentration property of Lasso and Dantzig estimators. Electronic Journal of Statistics , 2(none):90-102, 2008.
- I. Merad and S. Gaïffas. Convergence and concentration properties of constant step-size SGD through Markov chains. arXiv e-prints , art. arXiv:2306.11497, 2023.
- P. Mertikopoulos, N. Hallak, A. Kavis, and V. Cevher. On the almost sure convergence of stochastic gradient descent in non-convex problems. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems , volume 33, pages 1117-1128. Curran Associates, Inc., 2020.

- F. Mies and A. Steland. Sequential Gaussian approximation for nonstationary time series in high dimensions. Bernoulli , 29(4):3114-3140, 2023.
- W. Mou, C. J. Li, M. J. Wainwright, P. L. Bartlett, and M. I. Jordan. On linear stochastic approximation: Fine-grained Polyak-Ruppert and non-asymptotic concentration. In Proceedings of Thirty Third Conference on Learning Theory , pages 2947-2997. PMLR, 2020.
- E. Moulines and F. Bach. Non-asymptotic analysis of stochastic approximation algorithms for machine learning. In Advances in Neural Information Processing Systems , volume 24. Curran Associates, Inc., 2011.
- S. V. Nagaev. Large deviations of sums of independent random variables. The Annals of Probability , 7(5):745-789, 1979.
- A. Nemirovski, A. Juditsky, G. Lan, and A. Shapiro. Robust stochastic approximation approach to stochastic programming. SIAM Journal on Optimization , 19(4):1574-1609, 2009.
- Y. E. Nesterov. A method for solving the convex programming problem with convergence rate O (1 /k 2 ) . Dokl. Akad. Nauk SSSR , 269(3):543-547, 1983.
- L. M. Nguyen, J. Liu, K. Scheinberg, and M. Takáˇ c. SARAH: A novel method for machine learning problems using stochastic recursive gradient. arXiv:1703.00102 , 2017.
- T. D. Nguyen, T. H. Nguyen, A. Ene, and H. Nguyen. Improved convergence in high probability of clipped gradient methods with heavy tailed noise. In Thirty-seventh Conference on Neural Information Processing Systems , 2023.
- C. Paquette, K. Lee, F. Pedregosa, and E. Paquette. SGD in the Large: Average-case Analysis, Asymptotics, and Stepsize Criticality. In Proceedings of Thirty Fourth Conference on Learning Theory , pages 3548-3626. PMLR, July 2021.
- C. Paquette, E. Paquette, B. Adlam, and J. Pennington. Implicit regularization or implicit conditioning? exact risk trajectories of SGD in high dimensions. In Proceedings of the 36th International Conference on Neural Information Processing Systems , NIPS '22, pages 35984-35999, Red Hook, NY, USA, Nov. 2022. Curran Associates Inc.
- G. C. Pflug. Stochastic minimization with constant step-size: Asymptotic laws. SIAM Journal on Control and Optimization , 24(4):655-666, 1986.
- B. T. Polyak and A. B. Juditsky. Acceleration of stochastic approximation by averaging. SIAM Journal on Control and Optimization , 30(4):838-855, 1992.
- A. Rakhlin, O. Shamir, and K. Sridharan. Making gradient descent optimal for strongly convex stochastic optimization. In Proceedings of the 29th International Coference on International Conference on Machine Learning , pages 1571-1578, 2012.
- S. J. Reddi, S. Kale, and S. Kumar. On the convergence of Adam and beyond. In International Conference on Learning Representations , 2018.
- E. Rio. Moment inequalities for sums of dependent random variables under projective conditions. Journal of Theoretical Probability , 22(1):146-163, 2009.
- H. Robbins and S. Monro. A stochastic approximation method. The Annals of Mathematical Statistics , 22(3):400-407, 1951.
- H. Robbins and D. Siegmund. A convergence theorem for non negative almost supermartingales and some applications. In J. S. Rustagi, editor, Optimizing Methods in Statistics , pages 233-257. Academic Press, 1971.
- D. Ruppert. Efficient estimations from a slowly convergent Robbins-Monro process. Technical report, Cornell University Operations Research and Industrial Engineering, 1988.
- J. Sacks. Asymptotic distribution of stochastic approximation procedures. The Annals of Mathematical Statistics , 29(2):373-405, 1958.

- M. Schmidt, N. Le Roux, and F. Bach. Minimizing finite sums with the stochastic average gradient. Math. Program. , 162(1-2):83-112, 2017.
- B. Shi, W. J. Su, and M. I. Jordan. On learning rates and Schrödinger operators. Journal of Machine Learning Research , 24:1-53, 2023.
- N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research , 15(56): 1929-1958, 2014.
- L. G. Valiant. A theory of the learnable. Commun. ACM , 27(11):1134-1142, 1984.
- V. N. Vapnik. The Nature of Statistical Learning Theory . Springer, New York, NY, 2000.
- R. Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science . Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge, 2018.
- M. J. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint . Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge, 2019.
- M. J. Wainwright. Stochastic approximation with cone-contractive operators: Sharp ℓ ∞ -bounds for Q -learning. arXiv e-prints , art. arXiv:1905.06265, 2019.
- X. Wang and N. Gao. Stochastic resource allocation over fading multiple access and broadcast channels. IEEE Transactions on Information Theory , 56(5):2382-2391, 2010.
- Y. Wang, M. Chen, T. Zhao, and M. Tao. Large learning rate tames homogeneity: Convergence and balancing effect. In International Conference on Learning Representations , 2022.
- N. Wiener. Nonlinear problems in random theory . MIT Press, 1958.
- L. Wu, C. Ma, and W. E. How SGD selects the global minima in over-parameterized learning: A dynamical stability perspective. In Advances in Neural Information Processing Systems , volume 31, 2018.
- W. B. Wu. Nonlinear system theory: Another look at dependence. PNAS , 102(40):14150-14154, 2005.
- W. B. Wu. Strong invariance principles for dependent random variables. The Annals of Probability , 35(6):2294-2320, 2007.
- W. B. Wu. Recursive estimation of time-average variance constants. Ann. Appl. Probab. , 19(4): 1529-1552, 2009.
- W. B. Wu. Asymptotic theory for stationary processes. Statistics and Its Interface , 4(2):207-226, 2011.
- W. B. Wu and X. Shao. Limit theorems for iterated random functions. Journal of Applied Probability , 41(2):425-436, 2004.
- W. B. Wu and Y. N. Wu. Performance bounds for parameter estimates of high-dimensional linear models with correlated errors. Electronic Journal of Statistics , 10(1):352-379, 2016.
- H. Xiao and W. B. Wu. Covariance matrix estimation for stationary time series. The Annals of Statistics , 40(1), 2012.
- M. D. Zeiler. Adadelta: An adaptive learning rate method. arXiv e-prints , art. arXiv:1212.5701, 2012.
- C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals. Understanding deep learning requires rethinking generalization. In 2017 International Conference on Learning Representations (ICLR) , 2017.
- D. Zhang and W. B. Wu. Gaussian approximation for high dimensional time series. The Annals of Statistics , 45(5):1895-1919, 2017.

- D. Zhang and W. B. Wu. Convergence of covariance and spectral density estimates for highdimensional locally stationary processes. The Annals of Statistics , 49(1), 2021.
- T. Zhang. Statistical behavior and consistency of classification methods based on convex risk minimization. The Annals of Statistics , 32(1):56-134, 2004.
- Y. Zhong, J. Li, and S. Lahiri. Probabilistic guarantees of stochastic recursive gradient in non-convex finite sum problems. In D.-N. Yang, X. Xie, V. S. Tseng, J. Pei, J.-W. Huang, and J. C.-W. Lin, editors, Advances in Knowledge Discovery and Data Mining , pages 142-154, Singapore, 2024. Springer Nature.

## 8 Technical Appendices and Supplementary Material

## 8.1 Existence and Uniqueness of Global Minimum

Lemma 3. Consider the minimization problem β ∗ ∈ arg min β ∈ R d G ( β ) . If the function G satisfies Assumptions 1 and 2, then a global minimizer β ∗ exists and is unique.

Proof of Lemma 3. We first show the existence of a global minimizer. By the coercivity condition in Assumption 1, lim | β | s →∞ G ( β ) = ∞ , which implies that we can choose some large δ ∈ R such that the sub-level set

<!-- formula-not-decoded -->

is non-empty and bounded. Since G is continuous by Assumption 2, S δ is also closed, and hence compact in R d by the Heine-Borel theorem. Finally, by applying the Weierstrass extreme value theorem, there exists β ∗ ∈ S δ such that G ( β ∗ ) = min β ∈S δ G ( β ) . Since for any β / ∈ S δ , G ( β ) &gt; δ ≥ G ( β ∗ ) , G ( β ∗ ) = min β ∈ R d G ( β ) .

̸

Next, we show the uniqueness of the global minium. Assume that there are two distinct minimizers β 1 = β 2 . By Assumption 2, there exists µ &gt; 0 such that

<!-- formula-not-decoded -->

However, since β 1 and β 2 are both minimizers, ∇ G ( β 1 ) = ∇ G ( β 2 ) = 0 , while µ | β 1 -β 2 | s s &gt; 0 . This leads to contradiction, which finishes the proof.

## 8.2 Example: Linear Regression

As example, we consider the SGD algorithm for the high-dimensional linear regression, observing independent and identically distributed (i.i.d.) pairs ξ 1 := ( x 1 , y 1 ) , ξ 2 := ( x 2 , y 2 ) , . . . satisfying

<!-- formula-not-decoded -->

for random noises ϵ k that are independent of x k with E [ ϵ k ] = 0 and E | ϵ k | q &lt; ∞ for some q ≥ 2 . We verify Assumptions 2 and 3 and derive the explicit dependency of the learning-rate, the Lipschitz constant, and the moments of the gradient noise on the dimension d .

Let ξ = ( y, x ) be an independent random sample from the same distribution as the data. The least-squares loss and the stochastic gradient are respectively given by

<!-- formula-not-decoded -->

Then

Let

<!-- formula-not-decoded -->

To verify the ℓ s -type strong convexity

<!-- formula-not-decoded -->

imposed in Assumption 2, observe that ∇ G ( β ) -∇ G ( β ′ ) = Σ v . Thus, the condition becomes

̸

<!-- formula-not-decoded -->

Lemma 4. Let s ∈ { 2 , 4 , 6 , . . . } . Writing Σ = (Σ ij ) i,j =1 ,...,d , we have

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma 4. Write v = ( v 1 , . . . , v d ) ⊤ . Because of ( | v i | s -1 -| v j | s -1 )( | v i | - | v j | ) ≥ 0 , we obtain | v s -1 i v j | + | v i v s -1 j | ≤ v s i + v s j and

̸

<!-- formula-not-decoded -->

̸

This shows that the rightmost 'Gershgorin gap' min i =1 ,...,d Σ ii -∑ j : j = i | Σ ij | is a universal lower bound for every s . The lower bound is non-trivial if Σ is sufficiently diagonally dominant.

̸

For large s, the inequality λ ( s ) min ≥ min i =1 ,...,d Σ ii -∑ j : j = i | Σ ij | is nearly sharp. To see this, let i ∗ be the index i that minimizes min i =1 ,...,d Σ ii -∑ j : j = i | Σ ij | . For a small δ &gt; 0 , pick v = ( v 1 , . . . , v d ) by choosing v i ∗ := 1 and for i = i ∗ , taking v i := -sign(Σ i ∗ i )(1 -δ ) . For large s, v ⊙ ( s -1) ≈ (0 , 0 , . . . , 1 , 0 , . . . , 0) with the 1 at the i ∗ -th position. Similarly, | v | s s ≈ 1 . The i ∗ -th entry of Σ v is given by Σ i ∗ i ∗ -∑ j = i ∗ | Σ i ∗ j | + O ( δ ) . Hence for suitable sequences δ → 0 and s →∞ , we obtain ⟨ v ⊙ ( s -1) , Σ v ⟩ / | v | s s → Σ i ∗ i ∗ -∑ j = i ∗ | Σ i ∗ j | = min i Σ ii -∑ j = i | Σ ij | .

̸

̸

Regarding Assumption 3, we investigate the dependence of the Lipschitz constant L s,q on the dimension d in high-dimensional linear regression models. If s ∗ is the dual exponent of s , satisfying 1 /s +1 /s ∗ = 1 , we show that the condition holds with

<!-- formula-not-decoded -->

To see this, for any two vectors β , β ′ ∈ R d , we have

<!-- formula-not-decoded -->

Taking the ℓ s -norm on both sides, we obtain

<!-- formula-not-decoded -->

By Hölder's inequality, for the dual exponent s ∗ satisfying 1 /s +1 /s ∗ = 1 , it follows that

<!-- formula-not-decoded -->

Therefore, for q ≥ 2 , we have the q -th moment bounded as follows,

<!-- formula-not-decoded -->

proving (23).

Recall s d defined in (7). To bound the ℓ ∞ -norm, we set the conjugates

<!-- formula-not-decoded -->

Recall that for the ℓ s -norm, we have | x | ∞ ≤ | x | s d ≤ d 1 /s d | x | ∞ ≤ e | x | ∞ . Similarly, for the conjugate ℓ s ∗ d -norm, d 1 s ∗ d -1 = d 1 s d ≤ e implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which together with (23) gives

The next two lemmas show that the tail behavior of the covariate vector x k determines the behavior of the Lipschitz constant L s,q and the moment M s,q defined in Assumption 3.

̸

̸

̸

Lemma 5. Consider the linear regression in (18) with i.i.d. generic random samples ( x , y ) , where x = ( x 1 , . . . , x d ) ⊤ . Let q ≥ 2 and recall s d in (7) .

- (i) (Sub-Gaussian) If there is a constant K such that for all u ∈ R d , | u ⊤ x | ψ 2 ≤ K | u | 2 , where | v | ψ 2 = inf { t &gt; 0 : E [ e v 2 /t 2 ] ≤ 2 } denotes the sub-Gaussian norm, then

<!-- formula-not-decoded -->

- (ii) (Sub-exponential) If there is a constant K such that for all u ∈ R d , | u ⊤ x | ψ 1 ≤ K | u | 2 , where | v | ψ 1 = inf { t &gt; 0 : E [ e | v | /t ] ≤ 2 } denotes the sub-exponential norm, then

<!-- formula-not-decoded -->

- (iii) (Finite moment) If there is some p ≥ 2 q and a finite constant K p such that for each 1 ≤ j ≤ d , E | x j | p ≤ K p , then

<!-- formula-not-decoded -->

- (iv) For all three cases (i)-(iii), when s = 2 , L 2 ,q = O ( d ) .

Proof of Lemma 5. We write x = x k to denote a generic covariate. By (29) and Hölder's inequality,

<!-- formula-not-decoded -->

The convexity of the function t ↦→ t 2 q and Jensen's inequality yield | x | 2 q 1 ≤ d 2 q -1 ∑ d j =1 | x j | 2 q and

<!-- formula-not-decoded -->

Therefore, for all the three cases (i)-(iii),

<!-- formula-not-decoded -->

Next, we study the order of ( E [ | x | 2 q ∞ ]) 1 / (2 q ) for fixed q ≥ 2 .

- (i) If each x j is sub-Gaussian, then by Section 2.5 in Vershynin [2018], we have

<!-- formula-not-decoded -->

- (ii) If each x j is sub-exponential, then by Section 2.7 in Vershynin [2018], we obtain

<!-- formula-not-decoded -->

- (iii) If each x j has the finite p -th moment for some p ≥ 2 q , then

<!-- formula-not-decoded -->

Finally, for case (iv) with s d = 2 , by (23),

<!-- formula-not-decoded -->

By the convexity of the function t ↦→ t q , we apply Jensen's inequality and obtain

<!-- formula-not-decoded -->

Therefore, for x satisfying case (iii),

<!-- formula-not-decoded -->

which yields L 2 ,q = O ( d ) . For the cases (i) and (ii), by Sections 3.4 and 2.7 in Vershynin [2018], respectively, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

both indicating L 2 ,q = O ( d ) . This completes the proof.

Lemma 6. Consider the linear regression model in (18) and assume the conditions on ϵ and x therein are satisfied. Recall that M s,q = ∥|∇ g ( β ∗ , ξ ) | s ∥ q is defined in Assumption 3 for some q ≥ 2 . For the same four cases (i)-(iv) as in Lemma 5 and s d defined in (7) , M s d ,q is respectively equal to (i) O ( √ log( d )) , (ii) O (log( d )) , (iii) O ( d 1 / (2 q ) ) and (iv) O ( √ d ) .

Proof of Lemma 6. In the linear regression model, the stochastic gradient at the global minimum β ∗ can be rewritten into

<!-- formula-not-decoded -->

Since the noise ϵ is independent of the covariate vector x , we obtain

<!-- formula-not-decoded -->

By inequality (8), it suffices to bound ∥| x | ∞ ∥ q . Since ∥| x | ∞ ∥ q ≤ ∥| x | ∞ ∥ 2 q , the same arguments in the proof of Lemma 5 carry over immediately. We omit the details here.

## 8.3 Some Useful Lemmas

Lemma 7 (Maximal inequality [Chernozhukov et al., 2015]) . Let z 1 , . . . , z n be independent, d -dimensional random vectors. Denote the j -th element of z i by z ij , 1 ≤ j ≤ d . Define M := max 1 ≤ i ≤ n max 1 ≤ j ≤ d | z ij | and σ 2 := max 1 ≤ j ≤ d ∑ n i =1 E [ z 2 ij ] . Then,

<!-- formula-not-decoded -->

where the universal constant in ≲ is positive and independent of n and d .

Lemma 8 ( L q maximal inequality) . Let x 1 , . . . , x n be independent, d -dimensional random vectors. Denote by x ij the j -th element of x i , 1 ≤ j ≤ d . Then,

<!-- formula-not-decoded -->

This moment inequality can be derived by repeatedly applying Lemma 2. It generalizes the maximal inequality for E [max 1 ≤ j ≤ d | ∑ n i =1 ( x ij -E [ x ij ]) | ] in Chernozhukov et al. [2015], reproduced above as Lemma 7, to general q -th moments.

Proof of Lemma 8. One can assume that the independent random vectors x 1 , . . . , x n have zero means. By repeatedly applying Lemma 2 and choosing s = log( d ) ,

<!-- formula-not-decoded -->

and

Lemma 9 (Moment inequality [Li et al., 2024c]) . Let q ≥ 2 . For any two random vectors x and y in R d with fixed d ≥ 1 , and let

<!-- formula-not-decoded -->

Then, the following inequalities holds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 10 (Equivalence of ℓ s -ℓ ∞ -induced matrix norms) . For matrix A ∈ R d × d , we have the equivalence of the ℓ s d -norm and ℓ ∞ -norm induced matrix norms as follows

<!-- formula-not-decoded -->

̸

where s d is defined as (7) and ∥ A ∥ s = max | x | s =0 | A x | s / | x | s . If in addition, A is symmetric, then

<!-- formula-not-decoded -->

Proof of Lemma 10. By Horn and Johnson [1985], for any 1 ≤ p ≤ q ≤ ∞ and matrix A ∈ R d × d , d (1 /q ) -(1 /p ) ∥ A ∥ q ≤ ∥ A ∥ p ≤ d (1 /p ) -(1 /q ) ∥ A ∥ q . (33)

For p = s and q = ∞ , we obtain

<!-- formula-not-decoded -->

Since d 1 /s ≤ e by choosing s = s d in (7), we obtain (31).

For symmetric A = ( a ij ) 1 ≤ i,j ≤ d , a ij = a ji for all i, j . Therefore,

<!-- formula-not-decoded -->

This completes the proof.

## 8.4 Proofs for Section 2

Derivation of (15) : Since s is an even integer, we can write

<!-- formula-not-decoded -->

Taking the derivative with respect to α , we obtain

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

A first-order Taylor expansion yields then (15).

Proof of Lemma 1. Recall that we have defined

<!-- formula-not-decoded -->

A second order Taylor expansion gives f ( α ) = f (0) + αf ′ (0) + 1 2 α 2 f ′′ ( η ) for some η ∈ [0 , α ] . It suffices to bound sup u ∈ [0 ,α ] | f ′′ ( u ) | . Defining

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and the first two derivatives of f ( u ) can be respectively expressed by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since s is an even integer, it follows from Hölder's inequality that

<!-- formula-not-decoded -->

By applying Hölder's inequality again, we obtain

<!-- formula-not-decoded -->

By the two results above, we have

<!-- formula-not-decoded -->

Since u ∈ [0 , α ] , it follows that

<!-- formula-not-decoded -->

This completes the proof.

Existence of solution to (13) : To see the existence of the solution α s,q in

<!-- formula-not-decoded -->

denote the function α ↦→ F ( α ) = -µ + cα (1+ L ) q -2 for the constant c = [ | q -s | +( s -1)] L 2 / 2 &gt; 0 and L = L s,q . For any q ≥ 2 , and any α &gt; 0 , F ′ ( α ) = c [(1+ Lα ) q -2 + α ( q -2) L (1+ Lα ) q -3 ] &gt; 0 , proving that F ( α ) is strictly increasing on α &gt; 0 . Since F (0) = -µ &lt; 0 and F ( ∞ ) = + ∞ , the unique root to F ( α ) = 0 exists.

we have f ( u ) = [ M ( u )] q s ,

Proof of Lemma 2. Define φ ( t ) = ∥| x + t y | s ∥ 2 q for t ∈ [0 , 1] . Then

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Case I. If q/s -1 ≤ 0 , then ∆ 2 ( t ) ≤ 0 and φ ′′ ( t ) ≤ ∆ 3 ( t ) . By Hölder's inequality,

<!-- formula-not-decoded -->

Consequently,

<!-- formula-not-decoded -->

Case II. If q/s -1 &gt; 0 , by Hölder's inequality,

<!-- formula-not-decoded -->

Therefore, and

Then, we have

<!-- formula-not-decoded -->

Proof of Theorem 1. Consider the iterated random function

<!-- formula-not-decoded -->

To prove GMC in Theorem 1, it suffices to show that, for some q ≥ 2 and even integer s ≥ 2 , for any fixed vectors β , β ′ ∈ R d ,

<!-- formula-not-decoded -->

Recall the inequality in Lemma 2. For x and y therein, we choose them to be x = β -β ′ and y = -α ( ∇ g ( β , ξ ) -∇ g ( β ′ , ξ )) respectively. Then, it directly follows from Lemma 2 that

<!-- formula-not-decoded -->

This along with Assumptions 2 and 3 yields

<!-- formula-not-decoded -->

which completes the proof.

## 8.5 Proofs for Section 3.1

Proof of Proposition 1. Recall (17) and let ∇ G ( β ) = ( ∇ G 1 ( β ) , . . . , ∇ G d ( β ) ) ⊤ with

<!-- formula-not-decoded -->

Since the random samples ξ k , k ≥ 1 , are independent, it follows that for the k -th iteration, ξ k is independent of β k -1 . Then, by the tower rule, for all k ≥ 1 ,

<!-- formula-not-decoded -->

Therefore, by applying the high-dimensional moment inequality (16) in Lemma 2, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the second part in (51), noting that ∇ G ( β ∗ ) = 0 , by the triangle inequality, we have

<!-- formula-not-decoded -->

Since | · | s is a convex function for s ≥ 1 , we have | E [ · ] | s ≤ E [ | · | s ] . Thus, for all q ≥ 1 , by Jensen's inequality, we can bound

<!-- formula-not-decoded -->

This along with Assumption 3 yields

<!-- formula-not-decoded -->

Inserting this result back into (52), we obtain a bound for the second term in (51) using

<!-- formula-not-decoded -->

For the first term in (51), by applying Lemma 2 again, it follows from Assumptions 2 and 3 that

<!-- formula-not-decoded -->

Inserting this inequality and (55) into (51), we obtain the inequality

<!-- formula-not-decoded -->

The desired result is achieved since ∥|∇ g ( β ∗ , ξ k ) | s ∥ q ≤ M s,q by Assumption 3. As a special case, for the stationary SGD iterates β ◦ k ∼ π α , k ≥ 1 , we obtain the same result.

Proof of Theorem 2. First, we denote the contraction constant in Proposition 1 as follows

<!-- formula-not-decoded -->

Given the range of the constant learning rate α , we have ˜ r α,s,q &lt; 1 . Moreover, notice that

<!-- formula-not-decoded -->

Therefore, for the stationary SGD iterates β ◦ k ∼ π α , by Proposition 1, we can obtain

<!-- formula-not-decoded -->

Since the SGD iterates β ◦ k satisfy the geometric-moment contraction in Theorem 1, following Remark 2 in Wu and Shao [2004], the recursion β ◦ k = β ◦ k -1 -α ∇ g ( β ◦ k -1 , ξ k ) also holds for k ≤ 0 . Thus, we can recursively apply the inequality above and achieve

<!-- formula-not-decoded -->

This finishes the proof for the stationary SGD sequence.

Furthermore, for the general SGD iterates β k in (2) that may not have the stationary initialization, we apply the geometric-moment contraction in Theorem 1 and obtain

<!-- formula-not-decoded -->

which completes the proof.

## 8.6 Functional Dependence Measure in Time Series

The functional dependence measure in time series [Wu, 2005] is a key concept in our analysis. For that we view the high-dimensional SGD iterates { β k } k ∈ N as a nonlinear autoregressive (AR) process. Recall that ξ k , k ∈ Z , are i.i.d. Define the shift process F k = ( ξ k , ξ k -1 , . . . ) and its coupled version F k, { l } = ( ξ k , . . . , ξ l +1 , ξ ′ l , ξ l -1 , . . . ) , l ≤ k , where ξ ′ l is an i.i.d. copy of ξ l .

<!-- formula-not-decoded -->

where h α is a measurable function that depends on α [Wiener, 1958, Wu, 2005]. Define the coupled version of β ◦ k by

<!-- formula-not-decoded -->

The next lemma provides a bound for the functional dependence measure ∥| β ◦ k -β ◦ k, { l } | s ∥ q . It is later used to derive the moment bounds and the tail probability of the ASGD iterates.

Lemma 11. Consider the stationary SGD sequence { β ◦ k } k ≥ 1 . Suppose that Assumptions 2 and 3 hold with some q ≥ 2 and even integer s ≥ 2 . Then, for all k ≥ 1 and l ≤ k , we have

<!-- formula-not-decoded -->

Proof of Lemma 11. By applying Lemma 2, it follows from similar arguments as in the proof of Proposition 1 that, for each l ≤ k -1 ,

<!-- formula-not-decoded -->

By Assumption 3, for all l ≥ 1 ,

<!-- formula-not-decoded -->

which yields

<!-- formula-not-decoded -->

Recall ∥|∇ g ( β ∗ , ξ k ) | s ∥ q ≤ M s,q by Assumption 3. Therefore,

<!-- formula-not-decoded -->

This completes the proof.

## 8.7 Proofs for Section 3.2

In this section, we provide the proofs for the convergence results of ASGD in Section 3.2, which can be decomposed into the proofs for Theorems 6 to 8 in Section 6.

Proof of Theorem 7. Recall the i.i.d. random samples ξ k = ( y k , x k ) , the filtration F k = ( ξ k , ξ k -1 , . . . ) and its coupled version F k, { l } = ( ξ k , . . . , ξ l +1 , ξ ′ l , ξ l -1 , . . . ) , l ≤ k , where ξ ′ l is an i.i.d. copy of ξ l . Following Wu [2005], we introduce the projection operator

<!-- formula-not-decoded -->

Then, we can rewrite the centered ASGD into

<!-- formula-not-decoded -->

Since {P i -l ( β ◦ i ) } i ≥ l +1 is a sequence of martingale differences over i for each l = 0 , 1 , . . . , i -1 , following Lemma D.2 in Zhang and Wu [2021] and triangle inequality, we can obtain

<!-- formula-not-decoded -->

By Theorem 1 in Wu [2005], we have

<!-- formula-not-decoded -->

This along with Lemma 11 and definition of ˜ r α,s,q in (58) yields

<!-- formula-not-decoded -->

Recall r α,s,q in (6) and ˜ r α,s,q in (58). For some constant ω &gt; 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consequently, we can further bound (69) by

<!-- formula-not-decoded -->

we have 1 -ωα ≥ 0 and

For the term I 1 , it follows from Theorem 2 and expression (58) that

<!-- formula-not-decoded -->

Inserting this back into (74) gives

<!-- formula-not-decoded -->

for some constant c 1 &gt; 0 , where the last inequality is due to

<!-- formula-not-decoded -->

Similarly, for some constant c 2 &gt; 0 ,

<!-- formula-not-decoded -->

Combining the results of I 1 and I 2 , we obtain the claimed inequality

<!-- formula-not-decoded -->

Proof of Theorem 6. For the ASGD sequence { ¯ β k } k ∈ N with arbitrarily fixed initialization β 0 ∈ R d and the stationary ASGD sequence { ¯ β ◦ k } k ∈ N with β ◦ 0 ∼ π α , we have

<!-- formula-not-decoded -->

For each 1 ≤ i ≤ k , it follows from the geometric-moment contraction in Theorem 1 that

<!-- formula-not-decoded -->

Recall that r α,s,q = 1 -2 µα +(max { q, s } -1) L 2 s,q α 2 &lt; 1 in (6). Therefore,

<!-- formula-not-decoded -->

The desired result is achieved.

Proof of Theorem 8. Without loss of generality, assume β ∗ = 0 . We use the notation (17) for the derivatives of G. Notice that

<!-- formula-not-decoded -->

Consider the stationary SGD recursion

<!-- formula-not-decoded -->

By taking the expectation on the both sides, we obtain, for all k ≥ 1 ,

<!-- formula-not-decoded -->

Throughout the rest of the proof, we omit the iteration index k and write β = β ◦ k -1 when no confusion is caused. For notational convenience, write β = ( β 1 , . . . , β d ) ⊤ .

A first-order Taylor expansion on ∇ G ( β ) at β ∗ = 0 gives

<!-- formula-not-decoded -->

where ∇ 2 G (0) is the d × d Jacobian matrix with entries defined by

<!-- formula-not-decoded -->

and R ( β ) is the d -dimensional remainder defined as

<!-- formula-not-decoded -->

The i -th entry of R ( β ) can be rewritten into

<!-- formula-not-decoded -->

where ∇ 3 G i ( β ) , 1 ≤ i ≤ d , is a d × d matrix whose entries are

<!-- formula-not-decoded -->

Since ∇ G (0) = 0 and ∇ 2 G (0) is invertible given that λ min [ ∇ 2 G (0)] &gt; 0 , it follows from equation (84) that

<!-- formula-not-decoded -->

We only need to bound | E [ R ( β )] | s using Theorem 2, that is E [ | β ◦ k -β ∗ | s ] 2 = O (max { q, s } α ) for all k ≥ 1 .

Let v = β / | β | s . For each i = 1 , . . . , d ,

<!-- formula-not-decoded -->

By Hölder's inequality, for 1 /p +1 /q = 1 ,

<!-- formula-not-decoded -->

Again by Hölder's inequality,

<!-- formula-not-decoded -->

Therefore, by Theorem 2 and Lemma 10,

<!-- formula-not-decoded -->

where the matrix norm

<!-- formula-not-decoded -->

Finally, given the uniform bound max 1 ≤ i ≤ d ∥∇ 3 G i ( β ) ∥ ∞ &lt; ∞ ,

<!-- formula-not-decoded -->

which finishes the proof.

## 8.8 Proofs for Section 4

Proof of Theorem 4. By Theorem 6, we have ∥| ¯ β k -¯ β ◦ k | s ∥ q ≲ 1 / ( kα ) ∥| β 0 -β ◦ 0 | s ∥ q and consequently, it follows that

<!-- formula-not-decoded -->

Then it suffices to upper bound P ( | ¯ β ◦ k -β ∗ | s &gt; z ) . To this end, we first bound the dependence adjusted norm (Section 2 in Zhang and Wu [2017]) for { β ◦ k } k ≥ 1 . By Theorem 1, elementary calculations yield

<!-- formula-not-decoded -->

Consequently, by Theorem 6.2 in Zhang and Wu [2017] and Theorem 8, we have

<!-- formula-not-decoded -->

Combining this with (96) completes the proof.

Theorem 9 (Theorem 3.1 in [Mies and Steland, 2023]) . Let ( ϵ i ) i ∈ Z be i.i.d. random variables and ϵ k = ( ϵ k , ϵ k -1 , . . . ) . Assume X k = G k ( ϵ k ) ∈ R d with E [ X k ] = 0 for some measurable function G k . For any k , denote ˜ ϵ k,j = ( ϵ k , . . . , ϵ j +1 , ˜ ϵ j , ϵ j -1 , . . . ) with ˜ ϵ j an i.i.d. copy of ϵ j . Assume there exist Θ &gt; 0 and q &gt; 2 , such that for all k ,

<!-- formula-not-decoded -->

Additionally, assume that for some Γ ≥ 1 ,

<!-- formula-not-decoded -->

If d ≤ cn for some c &gt; 0 , then on a potentially different probability space, there exist random vectors ( X ′ k ) n k =1 = D ( X k ) n k =1 and independent, mean zero, Gaussian random vectors

<!-- formula-not-decoded -->

such that

<!-- formula-not-decoded -->

for some constant C depending on ( q, c ) .

Instead of univariate ϵ i , we apply Theorem 3.1 with vector-valued i.i.d. inputs ξ i . The theorem still applies as the proof depends only on the i.i.d. random elements and their L q bounds but not on the dimension of ξ i .

Proof of Theorem 5. To prove the Gaussian approximation we will apply Theorem 9 (Theorem 3.1 in Mies and Steland [2023]) with G k ≡ G = h α defined in (62) since β ◦ k is stationary. We now verify the conditions (97) and (98).

Recall the functional dependence measure ∥| β ◦ k -β ◦ k, { l } | s ∥ q introduced in Section 8.6.Throughout the proof, the q -th moment of the Euclidean norm is denoted by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Set

for some constant c &gt; 0 . If c is chosen sufficiently large, then, by Lemma 11 and Theorem 2, for all k ≥ 1 and l ≤ k , we have

<!-- formula-not-decoded -->

For α ∈ (0 , α s d ,q ) , it follows that ρ α,q &lt; 1 . Let l = k -j . Then, for a sufficiently large constant C ′ α,q , we have

<!-- formula-not-decoded -->

Therefore, the condition (97) holds with Θ = C ′ α,q M 2 ,q . This verifies the first part of condition 97. For the second part of condition 97, by Assumption 3 and Theorem 2, for some constant C ′′ α,q &gt; 0 ,

<!-- formula-not-decoded -->

Moreover, since β ◦ k is stationary, G k = G k -1 = h α and the left hand side of (98) is zero. Thus, condition (98) is trivially satisfied with Γ = 1 .

Finally, we show that the long-run covariance matrix Ξ = ∑ ∞ k = -∞ Cov( β ◦ 0 , β ◦ k ) is well defined in the sense that the spectral norm ∥ Ξ ∥ s is finite. Following (63), denote

<!-- formula-not-decoded -->

Since β ◦ k, {≤ 0 } is independent of β ◦ 0 , we have

<!-- formula-not-decoded -->

We can rewrite the difference as a telescoping sum,

<!-- formula-not-decoded -->

By stationarity and (100), it follows that

<!-- formula-not-decoded -->

For the spectral norm,

<!-- formula-not-decoded -->

where the first inequality is by Cauchy-Schwarz and the last inequality uses ( u ⊤ β ◦ 0 ) 2 ≤ | β ◦ 0 | 2 with | u | 2 = 1 . This, along with M 2 , 2 &lt; ∞ (Assumption 3) yields

<!-- formula-not-decoded -->

for some constant C ′ α &gt; 0 . As a direct consequence,

<!-- formula-not-decoded -->

This completes the proof.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We clearly stressed our contributions and scope in the abstract and included a subsection in the introduction to list our key innovations.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We talked about the limitations in the last section, especially for the assumptions of strong convexity and smoothness.

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

Justification: We provide rigorous proofs for all the theoretical results in the supplementary material.

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

Justification: This paper contributes to theoretical guarantees of high-dimensional SGD. Guidelines:

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

Justification: This paper does not include experiments requiring code.

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

Justification: This paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: This paper does not include experiments.

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

Justification: This paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We followed the NeurIPS Code of Ethics as instructed.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: This paper discusses potential positive societal impacts, particularly through advancing the theoretical understanding of modern machine learning, which can inform the development of more robust and efficient algorithms. As the work is purely theoretical and does not propose or evaluate any deployable systems, we do not anticipate any direct negative societal impacts.

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

Justification: This paper does not use existing assets.

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

Justification:

This paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

This paper does not involve crowdsourcing nor research with human subjects.

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
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.