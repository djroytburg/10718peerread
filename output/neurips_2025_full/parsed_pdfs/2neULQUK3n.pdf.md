## Adversarial Robustness of Nonparametric Regression

Parsa Moradi ∗ University of Minnesota moradi@umn.edu

Hanzaleh Akbarinodehi ∗ University of Minnesota akbar066@umn.edu

## Abstract

In this paper, we investigate the adversarial robustness of nonparametric regression, a fundamental problem in machine learning, under the setting where an adversary can arbitrarily corrupt a subset of the input data. While the robustness of parametric regression has been extensively studied, its nonparametric counterpart remains largely unexplored. We characterize the adversarial robustness in nonparametric regression, assuming the regression function belongs to the second-order Sobolev space (i.e., it is square integrable up to its second derivative).

The contribution of this paper is two-fold: (i) we establish a minimax lower bound on the estimation error, revealing a fundamental limit that no estimator can overcome, and (ii) we show that, perhaps surprisingly, the classical smoothing spline estimator, when properly regularized, exhibits robustness against adversarial corruption. These results imply that if o ( n ) out of n samples are corrupted, the estimation error of the smoothing spline vanishes as n →∞ . On the other hand, when a constant fraction of the data is corrupted, no estimator can guarantee vanishing estimation error, implying the optimality of the smoothing spline in terms of maximum tolerable number of corrupted samples.

## 1 Introduction

In recent years, machine learning (ML) models have increasingly relied on data from diverse sources and are often deployed in distributed or decentralized computing environments [1-7]. These settings introduce new attack surfaces and create incentives for adversaries to corrupt data or disrupt learning algorithms [8-11]. This has motivated a growing body of research initiatives aimed at understanding and mitigating the impact of adversarial behavior [12-21].

One of the fundamental problems in ML is regression, which aims to estimate an unknown function f based on observed noisy data [22]. This task is generally categorized into two approaches: parametric regression, which assumes f is a parametric function with known structure, and nonparametric regression, which makes minimal assumptions on f , allowing it to belong to a wide class of functions such as Sobolev or Hölder spaces [23, 24]. Regression underpins many machine learning tasks, and understanding its robustness to adversarial corruption is a critical objective.

Adversarial robustness in parametric regression has been extensively studied [25-31]. Many approaches leverage tools from classical robust statistics [32, 33], adapting techniques like trimmed means, median-of-means, and M -estimators to modern high-dimensional settings [34-36]. These methods benefit from the structural constraints of parametric models, which narrow the hypothesis space and simplify the alleviation of adversarial attacks. In contrast, robustness in nonparametric regression is considerably more challenging due to the absence of such structure, which makes the models more vulnerable to adversarial attacks [37-40].

In this work, we address the problem of nonparametric regression under adversarial corruption. We consider a setting in which one observes n pairs { ( x i , ˜ y i ) } n i =1 , where the responses ˜ y i may be

∗ Equal contribution.

Mohammad Ali Maddah-Ali University of Minnesota maddah@umn.edu

partially corrupted by an adversary. Specifically, the adversary arbitrarily choose ˜ y i , for all i ∈ A , where A is an unknown subset of { 1 , . . . , n } with cardinality at most q &lt; n . For each i / ∈ A , the observed response is ˜ y i = f ( x i ) + ε i , where f : Ω → R is the unknown regression function with domain Ω ⊂ R , and { ε i } i ∈ [ n ] \A are i.i.d. noise variables with zero mean and variance at most σ 2 .

In this paper, we assume that regression function f belongs to the second-order Sobolev space, consisting of functions that are square-integrable up to the second derivative over Ω . The objective of non-parametric regression is to produce ˆ f as an estimation of f based on { ( x i , ˜ y i ) } n i =1 . To evaluate the performance of ˆ f in the presence of adversarial corruption, we use the following metrics [24]:

<!-- formula-not-decoded -->

where ε := [ ϵ 1 , . . . , ϵ n ] and S denotes the adversarial strategy that can corrupt up to q samples. Our goal is to characterize inf ˆ f R 2 ( f, ˆ f ) and inf ˆ f R ∞ ( f, ˆ f ) over all estimators, assuming f belongs to the second-order Sobolev space, under the setting where the adversary may corrupt up to q samples.

The contributions of this paper are two-fold:

- A Computationally-Efficient Estimator (Theorem 1): We prove that the classical smoothing spline estimator retains robustness against adversarial corruption. This estimator selects ˆ f from the second-order Sobolev space, by minimizing the empirical error 1 n ∑ n i =1 ( g ( x i ) -˜ y i ) 2 , regularized by λ ∫ ˆ f ′′ ( x ) 2 dx , where λ &gt; 0 controls the level of smoothness [41]. Smoothing splines are computationally efficient, with O ( n ) complexity of fitting and evaluating, leveraging B-spline basis functions [42, 43], and have found wide applications in statistics and machine learning [44-47]. Note that classical nonparametric methods are not necessarily adversarially robust. For instance, the Nadaraya-Watson (NW) estimator [48] can be fragile even under a small number of adversarial corruptions [39]. It is therefore surprising that a computationally efficient nonparametric regression method, such as smoothing splines, also exhibits adversarial robustness.

While smoothing splines have been extensively studied in non-adversarial settings [41, 4955], their robustness properties against adversarial corruption were not previously understood. In this paper, we show that if the adversary corrupts at most q = o ( n ) samples and the regression function f belongs to a second-order Sobolev space, then the smoothing spline estimator achieves R 2 → 0 and R ∞ → 0 as n →∞ . This result further provides an upper bound on inf ˆ f R 2 ( f, ˆ f ) and inf ˆ f R ∞ ( f, ˆ f ) as functions of n and q .

- ˆ )
- Minimax Lower-Bound (Theorem 2): We derive minimax lower bounds on inf ˆ f R 2 ( f, f and inf ˆ f R ∞ ( f, ˆ f ) , expressed as functions of n and q . These bounds characterize the fundamental limits of estimation accuracy: no estimator can achieve better rates over the second-order Sobolev space under adversarial corruption.

A key implication of this result is that when q = Θ( n ) , no estimator can achieve vanishing error as n →∞ . This highlights that smoothing splines are not only computationally efficient but also optimal in terms of the maximum number of tolerable adversarial corruptions (see Corollary 4).

To better understand the results of this paper, we examine their implications in the regime of large n . In this regime, our results can be concisely summarized in Figure 1. This figure illustrates the rate of convergence, defined as -log n R 2 ( f, ˆ f ) or respectively, -log n R ∞ ( f, ˆ f ) , as a function of q , or equivalently, log q log n = log n q , as n →∞ . The red curves represent the impossibility result, indicating that no estimator can achieve a convergence rate beyond this bound for all functions in second-order Sobolev space (Theorem 2). The blue curve shows the convergence rate of the smoothing spline estimator in the presence of adversarial samples, as established in Theorem 1.

It is worth noting that, as shown in Figure 1a, when log n q &lt; 2 5 , the smoothing spline achieves the minimax-optimal convergence rate for metric R 2 . For 2 5 ≤ log n ( q ) &lt; 1 , in metric R 2 (Figure 1a), and for 0 ≤ log n ( q ) &lt; 1 in metric R ∞ (Figure 1b), while the smoothing splines offers vanishing estimation error for large n , the rate of convergence may not be optimum (there is a gap between the rate of convergence in smoothing splines and the minimax outer-bound). This theoretical results are supported with the simulation experiments results (see Section 4).

Figure 1: Rates of convergence for estimation error R 2 ( f, ˆ f ) and R ∞ ( f, ˆ f ) , as n → ∞ , and for any f belongs second-order Sobolev space (for non-asymptotic analysis, see Theorems 1 and 2). The blue curves represent the minimum rate achieved by the smoothing spline estimator. The red curves denote minimax outer bounds that are impossible to beat. Specifically, for q = o ( n ) , for the smoothing spline estimator, both R 2 and R ∞ converge to zero as n → ∞ . When q = Θ( n ) , we show that no estimator can achieve vanishing error, establishing a fundamental limit on robustness. This result highlights that smoothing splines are optimal in terms of the maximum tolerable number of adversarial corruptions (see Corollary 4).

<!-- image -->

This paper is organized as follows. Section 2 presents the problem formulation. Section 3 presents our main results by providing an upper and lower bounds under adversarial corruption. Section 4 provides simulation results, and Section 5 reviews related works.

Notation. Throughout the paper, we use [ n ] := { 1 , 2 , . . . , n } and denote the cardinality of a set A by |A| . Derivatives of scalar functions are written as f ′ , f ′′ , and, more generally, f ( k ) for the k -th derivative. The quantities ∥ g ∥ L 2 (Ω) and ∥ g ∥ L ∞ (Ω) denote the L 2 -norm and the supremum norm of a function g ( · ) over Ω . The space W 2 (Ω) refers to the second-order Sobolev space, consisting of square-integrable functions on Ω whose first and second derivatives are also square-integrable on Ω . We write a ≲ b to indicate that there exists a constant C &gt; 0 such that a ≤ Cb , and similarly a ≳ b to mean a ≥ Cb for some constant C &gt; 0 .

## 2 Problem Formulation

Let f : [ a, b ] → R be in W 2 ([ a, b ]) . The objective is to estimate f , from observations at fixed design points x i ∈ ( a, b ) for i ∈ [ n ] . Instead of observing a noisy version of responses (as in standard regression problem [22]), we are given (possibly) adversarially corrupted outputs { ˜ y i } n i =1 , defined as:

<!-- formula-not-decoded -->

where { ε i } i ∈ [ n ] \A are i.i.d. noise variables with zero mean and variance at most σ 2 , and A ⊆ [ n ] is an unknown subset of indices corresponding to adversarially corrupted observations. Here, ∗ denotes an arbitrary value chosen strategically by the adversary to mislead the estimator. We assume |A| ≤ q , for some known q ∈ N .

Let Ω := [ a, b ] denote the domain of the design points, and ε = ( ε i ) i ∈ [ n ] \A denote the noise vector. Following [24], we evaluate the performance of any estimator ˆ f using two metrics, R 2 ( f, ˆ f ) and R ∞ ( ˆ f ) , where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where S denotes the strategy, chosen by the adversary, in choosing the subset A and the value of ˜ y i , for i ∈ A , as long as |A| ≤ q . The supremum over S considers the worst-case adversarial attack, aiming to maximize estimation error for ˆ f .

In this paper, the objective is to find ˆ f , that minimizes R 2 ( f, ˆ f ) or R ∞ ( f, ˆ f ) , over all possible estimator functions ˆ f , where f is an arbitrary function in W 2 (Ω) .

## 3 Main Results

In this section, we present our main results on the adversarial robustness of nonparametric regression. Without loss of generality, we assume that Ω = [0 , 1] 2 . Let { x i } n i =1 ⊂ Ω denote the set of design points, and f ∈ W 2 (Ω) be the regression function.

First, we evaluate the robustness of the classical cubic smoothing spline estimator under adversarial corruption. In Theorem 1, we show that this estimator, as a computationally efficient [42, 43] and widely popular estimator [44-47], exhibits robustness to adversarial corruption.

The cubic smoothing spline estimator is defined as the solution to the following optimization problem:

<!-- formula-not-decoded -->

Here, λ &gt; 0 is a smoothing parameter that balances the fitness to the sample data, measured by 1 n ∑ n i =1 ( g ( x i ) -˜ y i ) 2 , and smoothness of the estimator, quantified by ∫ Ω g ′′ ( x ) 2 dx .

We define the empirical distribution function F n associated with the design points { x i } n i =1 as

<!-- formula-not-decoded -->

where 1 { x i ≤ x } is the indicator function. We assume that F n converges uniformly to a continuously differentiable cumulative distribution function (CDF) F ; that is,

<!-- formula-not-decoded -->

This is a standard assumption in the related literature [24]. For the limiting CDF, i.e., F ( x ) , we assume that the density function p ( x ) := F ′ ( x ) exists. In addition, similar to [39, 49], we assume that p ( x ) is bounded away from zero, i.e., there exists a constant p min &gt; 0 such that inf x ∈ Ω p ( x ) ≥ p min , and that p ( x ) is three times continuously differentiable on Ω .

We assume that the function f is bounded; that is, for all x ∈ Ω , | f ( x ) | ≤ m 1 for some constant m 1 ∈ R . Moreover, we assume that the adversary's corrupted values are also bounded, i.e., the adversary cannot inject arbitrarily large perturbations, satisfying | ˜ y i | ≤ m 2 for m 2 ∈ R and i ∈ A .

̸

Finally, let ∆ max := sup x ∈ Ω min i ∈ [ n ] | x -x i | , ∆ min := min i = j | x i -x j | , denote the maximum gap from any point in Ω to the nearest design point, and the minimum separation between any two design points, respectively. Likewise to [50], we assume that their ratio is bounded by a constant, i.e., ∆ max / ∆ min ≤ k, for some k &gt; 0 , ensuring that the design points are neither arbitrarily sparse nor overly clustered.

Theorem 1 (Upper Bound) . Let f ∈ W 2 (Ω) , and let ˆ f a SS denote the smoothing spline estimator defined in (3) . Let M = max { m 1 , m 2 } . Assume that λ → 0 as n → ∞ and λ &gt; n -2 . Then, for sufficiently large n , we have

<!-- formula-not-decoded -->

and also

<!-- formula-not-decoded -->

2 Note that any function f : [ a, b ] → R can be transformed with scaling and shifting into a function ˜ f : [0 , 1] → R without affecting its Sobolev regularity or the scaling of the associated metrics.

For the proof details, see Appendix A. Here, we present a proof sketch: We first decompose each metric into two components using the triangle inequality. Specifically, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˆ f SS denotes the smoothing spline estimator fitted on clean (uncorrupted) data. More precisely, we have

<!-- formula-not-decoded -->

with y i = ˜ y i , for i ∈ [ n ] \A , and otherwise, for i ∈ A , y i = f ( x i ) + ϵ i , for some i.i.d ϵ i . In addition, we have ˆ ϵ = ( ϵ i ) i ∈ [ n ] .

The first term in each decompositions (7) and (8), quantifies the estimator's error in the absence of adversarial contamination, reflecting the classical estimation error. The second term, referred to as adversarial deviation , captures how much the adversarial estimator ˆ f a SS deviates from its uncorrupted counterpart ˆ f SS .

To bound the first term in decomposition (7), we rely on an established upper bounds for smoothing spline estimation [50, 51], which guarantee that, for sufficiently large n , we have

<!-- formula-not-decoded -->

Applying (9) with j = 0 yields the desired bound for the first term in decomposition (7). For decomposition (8), the first term is bounded by combining (9) with j = 0 and j = 1 , applying norm inequalities for Sobolev spaces [56], and using the Cauchy-Schwarz inequality, leading to

<!-- formula-not-decoded -->

To bound the second term in decompositions (7) and (8), we leverage the fact that the smoothing spline estimator is a linear smoother [41]. Specifically, the solution to (3) can be expressed in a kernel form as

<!-- formula-not-decoded -->

where W n ( · , · ) denotes the smoothing spline kernel (or weight function), which depends on the design points { x i } i ∈ [ n ] , sample size n , and the smoothing parameter λ . Using this representation, we have

<!-- formula-not-decoded -->

where ( a ) follows from the fact that y i = ˜ y i , for i ∈ [ n ] \A . Using the Hölder inequality, and taking expectation, we show that

<!-- formula-not-decoded -->

Unfortunately, W n ( · , · ) does not admit an analytically tractable form [52, 53] for directly bounding its supremum in (13). However, a substantial body of research [52-55] has focused on approximating W n ( · , · ) with analytically tractable functions, known as equivalent kernels , denoted by ̂ W n ( x, s ) . We leverage such approximations in our analysis to derive an upper bound for (13), leading to

<!-- formula-not-decoded -->

We also take similar steps to derive

<!-- formula-not-decoded -->

Combining (9), (10), (14), and (15) completes the proof of Theorem 1.

Corollary 1 (Convergence Rate of R 2 ( f, ˆ f ) ) . Assume the conditions of Theorem 1 hold, and q = Θ( n β ) for some β ∈ [0 , 1] . Then, by choosing λ = O ( n -4 / 5 ) for β ≤ 0 . 4 and λ = O ( n -4 / 3(1 -β ) ) for β &gt; 0 . 4 , we have

<!-- formula-not-decoded -->

as depicted by the blue curve in Figure 1a.

Corollary 2 (Convergence Rate of R ∞ ( f, ˆ f ) ) . Under the same assumptions as in Corollary 1, by choosing λ = O ( n -4 / 5 ) for β ≤ 0 . 5 and λ = O ( n -8 / 5(1 -β ) ) for β &gt; 0 . 5 , we have

<!-- formula-not-decoded -->

as depicted by the blue curve in Figure 1b.

Based on Corollaries 1 and 2, the thresholds at β = 0 . 4 for R 2 ( f, ˆ f a SS ) and β = 0 . 5 for R ∞ ( f, ˆ f a SS ) indicate a phase transition: Below these points, the estimation error is dominated by noise, and adversarial corruption has no impact on the convergence rate. Beyond these thresholds, the adversary dictates the rate of convergence. In this scenario, we must choose a larger smoothing parameter λ to smooth out the adversarial contribution in the data points.

Furthermore, for all β ∈ [0 , 1) , the convergence rate of R ∞ ( f, ˆ f a SS ) is slower than that of R 2 ( f, ˆ f a SS ) , as established by Theorem 1. This reflects the greater sensitivity of R ∞ estimation error to adversarial attacks: while metric R 2 averages the estimation error across the entire domain, metric R ∞ is driven by the worst-case pointwise error, making it inherently more vulnerable to adversarial perturbations.

In the following theorem we provide a minimax lower bound for both metrics.

Theorem 2. Let P ε denote the probability density function of the noise vector ε , with i.i.d zero-mean σ 2 -variance entries, f ∈ W 2 (Ω) be the regression function. Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The full proof of Theorem 2 is provided in Appendix B. Here, we present a proof sketch. To establish Theorem 2, we reduce the minimax risk in (18) and (19) to a hypothesis testing problem [57]. Specifically, we construct two functions f 1 and f 2 in W 2 (Ω) with L 2 and L ∞ distance, bounded away from zero (see Figure 2). However, given n samples from either function, an adversary can corrupt up to q of them, making it impossible for any estimator to reliably distinguish between f 1 and f 2 . Consequently, no estimation approach can identify which function generated the data, and the average hypothesis testing error remains 1 / 2 . Applying [57, Proposition 5.1] yields the following lower bounds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, when q = 0 , the adversarial model reduces to the classical non-adversarial setting, for which the minimax lower bounds have been established as inf ˆ f sup f,P ε R 2 ( f, ˆ f ) ≳ n -4 / 5 and inf ˆ f sup f,P ε R ∞ ( f, ˆ f ) ≳ (log n/n ) 3 / 4 [24]. Combining these with (20) and (21) completes the proof of Theorem 2.

Figure 2: Construction of functions f 1 (blue) and f 2 (red) used in Theorem 2. Both functions belong to W 2 ([0 , 1]) , where f 1 ( x ) = 0 for all x , and f 2 ( x ) differs from f 1 only on the interval [0 , r q ] , with r q = q/n . The function f 2 is linear on [0 , r q -ε q ] , where ε q = r 2 q , and transitions smoothly to zero on [ r q -ε q , r q ] via a degree-5 polynomial, ensuring f 2 ∈ W 2 ([0 , 1]) . This construction induces a non-zero gap in both L 2 and L ∞ norms, while enabling the adversary to obscure the difference by corrupting only q samples, and making f 1 , f 2 statistically indistinguishable. The details of this construction is provided in Appendix B.

<!-- image -->

Corollary 3. Assuming q = Θ( n β ) for some β ∈ [0 , 1] , we conclude from Theorem 2 that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

as depicted by red curves in Figure 1.

Corollary 4 (On optimality of Smoothing Spline) . Assume the conditions of Theorem 1 hold and that q = o ( n ) . Then, by selecting the smoothing parameter λ = ( q n ) 4 / 3 when q ≥ n 0 . 4 and λ = n -0 . 8 when q &lt; n 0 . 4 , R 2 ( f, ˆ f a SS ) vanishes as n → ∞ . Similarly, for R ∞ ( f, ˆ f a SS ) , setting λ = ( q n ) 6 / 5 when q ≥ n 0 . 5 and λ = n -0 . 8 when q &lt; n 0 . 5 ensures that R ∞ ( f, ˆ f a SS ) also converges to zero. Consequently, as long as q = o ( n ) , then R 2 ( f, ˆ f a SS ) and R ∞ ( f, ˆ f a SS ) go to zero, as n →∞ . Conversely, if q = Θ( n ) , according to Corollary 3, there exists a function f ∈ W 2 (Ω) such that, for any estimator ˆ f , none of R 2 ( f, ˆ f ) and R ∞ ( f, ˆ f ) converges to zero as n →∞ . This implies that the classical cubic smoothing spline estimator is optimal with respect to maximum tolerable number of adversarial corruptions.

## 4 Experimental Results

In this section, we present numerical experiments to validate the theoretical results. All experiments are conducted on a single CPU-only machine. The smoothing spline estimator is implemented using the SciPy package [58]. We consider two regression functions: (i) f ( x ) = x sin( x ) over the domain Ω = [ -10 , 10] with M = 100 , and (ii) a three-layer MLP network with weights initialized in [ -1 , 1] and M = 500 . The noise vector ε is drawn independently from a Gaussian distribution with zero mean and variance σ 2 = 1 .

To evaluate the adversarial robustness of the cubic smoothing spline estimator, we consider three distinct attack strategies:

- Random Corruption Attack: The adversary randomly selects q out of the n samples and replaces their response values with M .
- Greedy Corruption Attack: In this process, the attacker:
1. Fits the baseline estimator on clean data.
2. Computes ℓ i = ( ˆ f ( x i ) -y i ) 2 for each sample.
3. Identifies i ⋆ = arg min i ℓ i .

Figure 3: Log-log plots of error convergence rates for the cubic smoothing spline estimator ˆ f = ˆ f a SS for f ( x ) = x sin( x ) in the uniform design setting, where the input points converge to a uniform distribution. The top row shows R 2 ( f, ˆ f ) and R ∞ ( f, ˆ f ) errors for q = n 0 . 3 , along with the corresponding theoretical upper bounds of O ( n -0 . 8 ) and O ( n -0 . 6 ) , respectively. The bottom row presents R 2 ( f, ˆ f ) and R ∞ ( f, ˆ f ) errors for q = n 0 . 6 , with theoretical upper bounds of O ( n -0 . 53 ) and O ( n -0 . 48 ) , respectively.

<!-- image -->

4. Updates y i ⋆ ← y i ⋆ + M · sign ( ˆ f ( x i ⋆ ) -y i ⋆ ) .
5. Repeats the process until q points are corrupted.
- Concentrated Corruption Attack: The adversary targets q consecutive samples centered around the median of the design points and modifies their corresponding labels to M .

For each attack strategy, we evaluate both R 2 ( f, ˆ f a SS ) and R ∞ ( f, ˆ f a SS ) across a range of sample sizes n , and examine how these metrics scale with n under varying levels of adversarial corruption. Additionally, for each experiment, we consider two settings for the design points: uniform and Gaussian. In the uniform and Gaussian settings, the design points { x i } n i =1 converge to a uniform and a truncated Gaussian distribution over Ω , respectively, as n →∞ .

For the function f ( x ) = x sin( x ) , Figures 3 and 5 illustrate the behavior of R 2 ( f, ˆ f a SS ) and R ∞ ( f, ˆ f a SS ) under uniform and Gaussian designs, respectively, for two corruption levels, q = n 0 . 3 and q = n 0 . 6 . Similarly, for the MLP network, Figures 4 and 6 present the corresponding results.

As shown in these figures, the empirical convergence rates align well with the theoretical upper bounds established in Theorem 1. Specifically, Theorem 1 establishes that R 2 ( f, ˆ f a SS ) ≤ O ( n -0 . 8 ) and R ∞ ( f, ˆ f a SS ) ≤ O ( n -0 . 6 ) for q = n 0 . 3 , and R 2 ( f, ˆ f a SS ) ≤ O ( n -0 . 53 ) and R ∞ ( f, ˆ f a SS ) ≤ O ( n -0 . 48 ) for q = n 0 . 6 . These theoretical predictions align with the empirical convergence trends observed in

Figure 4: Log-log plots showing the convergence behavior of the cubic smoothing spline estimator ˆ f = ˆ f a SS when the ground-truth function is the MLP network, under the uniform design setting. The top row corresponds to the case q = n 0 . 3 , with theoretical convergence rates of O ( n -0 . 8 ) for R 2 ( f, ˆ f ) and O ( n -0 . 6 ) for R ∞ ( f, ˆ f ) . The bottom row shows results for a higher corruption level, q = n 0 . 6 , with respective theoretical upper bounds of O ( n -0 . 53 ) and O ( n -0 . 48 ) .

<!-- image -->

the figures. Moreover, these figures show that the concentrated attack results in noticeably higher estimation error compared to the other two attack strategies.

It is important to note that these empirical rates are not expected to match the lower bounds from Theorem 2, since those bounds are minimax in nature. That is, they guarantee the existence of a worst-case function f ⋆ ∈ W 2 (Ω) for which no estimator can achieve faster convergence. Therefore, the lower bounds apply to such worst-case functions and not necessarily to all functions in W 2 (Ω) , including the two functions in our experiments.

## 5 Related Work

Unlike parametric regression, which benefits from structural assumptions on the model class, nonparametric regression imposes minimal assumptions on the underlying function. This flexibility, makes it substantially more challenging to evaluate and guarantee adversarial robustness. Consequently, the literature on adversarial robustness in nonparametric settings remains relatively sparse. Nonetheless, several notable efforts have begun to address this gap.

As discussed earlier, the Nadaraya-Watson (NW) estimator is not robust to adversarial corruption and can fail even in the presence of a single corrupted sample [38, 39]. Classical robust estimation techniques, such as the Median-of-Means (MoM) estimator [59] and trimmed means [60], have been extended to nonparametric settings [37, 38, 61] to improve the robustness of the NW estimator. In the MoM approach, the data are partitioned into several groups, an NW estimator is fitted to each

group, and the median of the resulting estimates is taken. While this method enhances robustness to outliers, its performance degrades sharply when even a single corrupted sample appears in each group. Trimmed-mean methods, on the other hand, discard a fixed fraction of samples with extreme response values and fit the NW estimator on the remaining data. However, their effectiveness is limited when adversarial corruption is not uniformly distributed across the input space.

Zhao et al. [39] study adversarial robustness in kernel-based nonparametric regression by analyzing an M -estimator variant of the Nadaraya-Watson (NW) estimator [48, 62], deriving upper and minimax lower bounds for metrics based on L 2 -norm and L ∞ -norm. In comparison to our setting, which assumes the regression function lies in a second-order Sobolev space, their work considers a first-order Hölder class. Their proposed estimator requires gradient descent with O ( n log(1 /ϵ )) complexity to produce an estimation with precision ϵ on new data point. In contrast, the cubic smoothing spline accurately evaluates new data point in O ( n ) time, offering greater computational efficiency.

Several works also have studied the robustness of nonparametric classification. In [40], the authors analyze the robustness of nonparametric linear classifiers under arbitrary norms and mild regularity assumptions. The robustness of nearest neighbor classifiers against adversarial perturbations has been studied in [63] and a general attack framework applicable to a wide class of nonparametric classifiers is introduced in [64] and a data-pruning defense strategy to mitigate such attacks is proposed.

## 6 Conclusion

In this paper, we study the adversarial robustness of nonparametric regression when the underlying regression function belongs to W 2 (Ω) . We prove that the cubic smoothing spline achieves vanishing R 2 and R ∞ errors as long as the number of corrupted samples satisfies q = o ( n ) . We also establish lower bounds using a minimax argument. Notably, we show that cubic smoothing splines are optimal with respect to the maximum number of tolerable adversarial corruptions.

## Acknowledgment

This material is based upon work supported by the National Science Foundation under Grant CIF2348638.

## References

- [1] Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Mark Mao, Marc'aurelio Ranzato, Andrew Senior, Paul Tucker, Ke Yang, et al. Large scale distributed deep networks. Advances in neural information processing systems , 25, 2012.
- [2] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics , pages 1273-1282. PMLR, 2017.
- [3] Yang You, Zhao Zhang, Cho-Jui Hsieh, James Demmel, and Kurt Keutzer. Imagenet training in minutes. In Proceedings of the 47th international conference on parallel processing , pages 1-10, 2018.
- [4] Jakub Koneˇ cn` y, Brendan McMahan, and Daniel Ramage. Federated optimization: Distributed optimization beyond the datacenter. arXiv preprint arXiv:1511.03575 , 2015.
- [5] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053 , 2019.
- [6] Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, Antonio Marcedone, H Brendan McMahan, Sarvar Patel, Daniel Ramage, Aaron Segal, and Karn Seth. Practical secure aggregation for privacy-preserving machine learning. In proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security , pages 1175-1191, 2017.

- [7] Hanzaleh Akbari Nodehi, Viveck R Cadambe, and Mohammad Ali Maddah-Ali. Game of coding: Sybil resistant decentralized machine learning with minimal trust assumption. arXiv preprint arXiv:2410.05540 , 2024.
- [8] Peiyu Xiong, Michael Tegegn, Jaskeerat Singh Sarin, Shubhraneel Pal, and Julia Rubin. It is all about data: A survey on the effects of data on adversarial robustness. ACM Computing Surveys , 56(7):1-41, 2024.
- [9] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199 , 2013.
- [10] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572 , 2014.
- [11] Zhiyi Tian, Lei Cui, Jie Liang, and Shui Yu. A comprehensive survey on poisoning attacks and countermeasures in machine learning. ACM Computing Surveys , 55(8):1-35, 2022.
- [12] Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In 2017 ieee symposium on security and privacy (sp) , pages 39-57. Ieee, 2017.
- [13] Lianghe Shi and Weiwei Liu. Adversarial self-training improves robustness and generalization for gradual domain adaptation. Advances in Neural Information Processing Systems , 36: 37321-37333, 2023.
- [14] Tao Bai, Jinqi Luo, Jun Zhao, Bihan Wen, and Qian Wang. Recent advances in adversarial training for adversarial robustness. arXiv preprint arXiv:2102.01356 , 2021.
- [15] Brendan Van Rooyen and Robert C Williamson. A theory of learning with corrupted labels. Journal of Machine Learning Research , 18(228):1-50, 2018.
- [16] Sainbayar Sukhbaatar and Rob Fergus. Learning from noisy labels with deep neural networks. arXiv preprint arXiv:1406.2080 , 2(3):4, 2014.
- [17] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083 , 2017.
- [18] Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer. Machine learning with adversaries: Byzantine tolerant gradient descent. Advances in neural information processing systems , 30, 2017.
- [19] Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett. Byzantine-robust distributed learning: Towards optimal statistical rates. In International conference on machine learning , pages 5650-5659. Pmlr, 2018.
- [20] Parsa Moradi, Hanzaleh Akbarinodehi, and Mohammad Ali Maddah-Ali. General coded computing: Adversarial settings. arXiv preprint arXiv:2502.08058 , 2025.
- [21] Hanzaleh Akbari Nodehi, Viveck R Cadambe, and Mohammad Ali Maddah-Ali. Game of coding: Beyond trusted majorities. In 2024 IEEE International Symposium on Information Theory (ISIT) , pages 2850-2855. IEEE, 2024.
- [22] Wolfgang Härdle and Enno Mammen. Comparing nonparametric versus parametric regression fits. The Annals of Statistics , pages 1926-1947, 1993.
- [23] Wolfgang Härdle. Applied nonparametric regression . Cambridge university press, 1990.
- [24] Alexandre B. Tsybakov. Introduction to Nonparametric Estimation . Springer Series in Statistics. Springer, New York, 2009.
- [25] Chen Dan, Yuting Wei, and Pradeep Ravikumar. Sharp statistical guaratees for adversarially robust gaussian classification. In International Conference on Machine Learning , pages 23452355. PMLR, 2020.

- [26] Yue Xing, Ruizhi Zhang, and Guang Cheng. Adversarially robust estimate and risk analysis in linear regression. In International Conference on Artificial Intelligence and Statistics , pages 514-522. PMLR, 2021.
- [27] Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certified adversarial robustness via randomized smoothing. In international conference on machine learning , pages 1310-1320. PMLR, 2019.
- [28] Aref Rekavandi, Farhad Farokhi, Olga Ohrimenko, and Benjamin Rubinstein. Certified adversarial robustness via randomized α -smoothing for regression models. Advances in Neural Information Processing Systems , 37:134127-134150, 2024.
- [29] Yiling Xie and Xiaoming Huo. High-dimensional (group) adversarial training in linear regression. arXiv preprint arXiv:2405.13940 , 2024.
- [30] Antonio Ribeiro, Dave Zachariah, Francis Bach, and Thomas Schön. Regularization properties of adversarially-trained linear regression. Advances in Neural Information Processing Systems , 36:23658-23670, 2023.
- [31] Edgar Dobriban, Hamed Hassani, David Hong, and Alexander Robey. Provable tradeoffs in adversarially robust classification. IEEE Transactions on Information Theory , 69(12):77937822, 2023.
- [32] Peter J Huber and Elvezio M Ronchetti. Robust statistics . John Wiley &amp; Sons, 2011.
- [33] Ricardo A Maronna, R Douglas Martin, Victor J Yohai, and Matías Salibián-Barrera. Robust statistics: theory and methods (with R) . John Wiley &amp; Sons, 2019.
- [34] Ilias Diakonikolas, Gautam Kamath, Daniel Kane, Jerry Li, Ankur Moitra, and Alistair Stewart. Robust estimators in high-dimensions without the computational intractability. SIAM Journal on Computing , 48(2):742-864, 2019.
- [35] Ilias Diakonikolas, Gautam Kamath, Daniel M Kane, Jerry Li, Ankur Moitra, and Alistair Stewart. Being robust (in high dimensions) can be practical. In International Conference on Machine Learning , pages 999-1008. PMLR, 2017.
- [36] Yu Cheng, Ilias Diakonikolas, Rong Ge, and David P Woodruff. Faster algorithms for highdimensional robust covariance estimation. In Conference on Learning Theory , pages 727-757. PMLR, 2019.
- [37] Anna Ben-Hamou and Arnaud Guyader. Robust non-parametric regression via median-of-means. https://hal.science/hal-03957385v1/document , 2023. HAL preprint, hal-03957385.
- [38] Subhra Dhar, Prashant Jha, and Prabrisha Rakshit. The trimmed mean in non-parametric regression function estimation. Theory of Probability and Mathematical Statistics , 107:133158, 2022.
- [39] Puning Zhao and Zhiguo Wan. Robust nonparametric regression under poisoning attack. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pages 17007-17015, 2024.
- [40] Elvis Dohmatob. Consistent adversarially robust linear classification: non-parametric setting. In Forty-first International Conference on Machine Learning , 2024.
- [41] Grace Wahba. Smoothing noisy data with spline functions. Numerische mathematik , 24(5): 383-393, 1975.
- [42] Grace Wahba. Spline models for observational data . SIAM, 1990.
- [43] Paul HC Eilers and Brian D Marx. Flexible smoothing with b-splines and penalties. Statistical science , 11(2):89-121, 1996.
- [44] Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljaˇ ci´ c, Thomas Y Hou, and Max Tegmark. Kan: Kolmogorov-arnold networks. arXiv preprint arXiv:2404.19756 , 2024.

- [45] Jian Zhao and Hui Zhang. Thin-plate spline motion model for image animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 3657-3666, 2022.
- [46] Chuanbo Hua, Federico Berto, Michael Poli, Stefano Massaroli, and Jinkyoo Park. Learning efficient surrogate dynamic models with graph spline networks. Advances in Neural Information Processing Systems , 36:52523-52547, 2023.
- [47] Junhui He, Ying Yang, and Jian Kang. Adaptive bayesian multivariate spline knot inference with prior specifications on model complexity. arXiv preprint arXiv:2405.13353 , 2024.
- [48] Elizbar A Nadaraya. On estimating regression. Theory of Probability &amp; Its Applications , 9(1): 141-142, 1964.
- [49] Felix Abramovich and Vadim Grinshtein. Derivation of equivalent kernel for general spline smoothing: a systematic approach. Bernoulli , 5(1):109-123, 1999.
- [50] Florencio I Utreras. Convergence rates for multivariate smoothing spline functions. Journal of approximation theory , 52(1):1-27, 1988.
- [51] David L Ragozin. Error bounds for derivative estimates based on spline smoothing of exact or noisy data. Journal of approximation theory , 37(4):335-355, 1983.
- [52] Bernard W Silverman. Spline smoothing: the equivalent variable kernel method. The annals of Statistics , pages 898-916, 1984.
- [53] K Messer. A comparison of a spline estimate to its equivalent kernel estimate. The Annals of Statistics , pages 817-829, 1991.
- [54] Karen Messer and Larry Goldstein. A new class of kernels for nonparametric curve estimation. The Annals of Statistics , pages 179-195, 1993.
- [55] Douglas Nychka. Splines as local smoothers. The Annals of Statistics , pages 1175-1197, 1995.
- [56] Giovanni Leoni. A first course in Sobolev spaces , volume 181. American Mathematical Society, 2024.
- [57] Martin J Wainwright. High-dimensional statistics: A non-asymptotic viewpoint , volume 48. Cambridge university press, 2019.
- [58] Pauli Virtanen, Ralf Gommers, Travis E Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, et al. Scipy 1.0: fundamental algorithms for scientific computing in python. Nature methods , 17(3):261-272, 2020.
- [59] Arkadij Semenovich Nemirovskij and David Borisovich Yudin. Problem Complexity and Method Efficiency in Optimization . Wiley-Interscience, 1983.
- [60] AH Welsh. The trimmed mean in the linear model. The Annals of Statistics , 15(1):20-36, 1987.
- [61] Pierre Humbert, Batiste Le Bars, and Ludovic Minvielle. Robust kernel density estimation with median-of-means principle. In International Conference on Machine Learning , pages 9444-9465. PMLR, 2022.
- [62] Geoffrey S Watson. Smooth regression analysis. Sankhy¯ a: The Indian Journal of Statistics, Series A , pages 359-372, 1964.
- [63] Yizhen Wang, Somesh Jha, and Kamalika Chaudhuri. Analyzing the robustness of nearest neighbors to adversarial examples. In International Conference on Machine Learning , pages 5133-5142. PMLR, 2018.
- [64] Yao-Yuan Yang, Cyrus Rashtchian, Yizhen Wang, and Kamalika Chaudhuri. Robustness for non-parametric classification: A generic attack and defense. In International Conference on Artificial Intelligence and Statistics , pages 941-951. PMLR, 2020.

- [65] Robert J Serfling. Approximation theorems of mathematical statistics . John Wiley &amp; Sons, 2009.
- [66] Parsa Moradi, Behrooz Tahmasebi, and Mohammad Maddah-Ali. Coded computing for resilient distributed computing: A learning-theoretic framework. Advances in Neural Information Processing Systems , 37:111923-111964, 2024.
- [67] Lucien LeCam. Convergence of estimates under dimensionality restrictions. The Annals of Statistics , pages 38-53, 1973.
- [68] Bin Yu. Assouad, fano, and le cam. In Festschrift for Lucien Le Cam: research papers in probability and statistics , pages 423-435. Springer, 1997.

## A Proof of Theorem 1

In this section, we prove Theorem 1. Without loss of generality, we assume that Ω = [0 , 1] 3 . Recall that the solution of (3) is unique and the explicit formula for ˆ f a SS is given by

<!-- formula-not-decoded -->

where W n ( x, x i ) denotes the smoothing spline weight function depending on { x i } n i =1 , the sample size n , and the smoothing parameter λ .

To facilitate the analysis, we define a second scenario in which the adversarial strategy is to be honest , that is, for any i ∈ A , the adversary does not deviate from the clean data generation process and behaves as if it were non-adversarial. This allows us to construct a one-to-one correspondence between the realizations of the adversarial and honest scenarios such that for each i / ∈ A , the observed responses y i are identical across both settings, while for i ∈ A , the responses may differ: in Scenario 1 (adversarial), the adversary may introduce arbitrary deviations, whereas in Scenario 2 (honest), the responses follow the true underlying model.

In this second setting, we apply the same smoothing spline estimator to the uncorrupted data. The resulting estimator, which we denote by ˆ f SS , is given by

<!-- formula-not-decoded -->

where y i denotes the uncorrupted response corresponding to input x i , i.e., y i = ˜ y i , for i ∈ [ n ] \A , and otherwise, for i ∈ A , y i = f ( x i ) + ϵ i , for some i.i.d ϵ i . We define ˆ ϵ := ( ϵ i ) i ∈ [ n ] .

We now proceed to prove the bounds stated in Theorem 1. We first establish the upper bound for R 2 ( f, ˆ f a SS ) in (5), and subsequently turn to the bound for R ∞ ( f, ˆ f a SS ) in (6).

By the definition of R 2 ( f, ˆ f a SS ) , we have

<!-- formula-not-decoded -->

where ε = ( ε i ) i ∈ [ n ] \A . First, observe that

<!-- formula-not-decoded -->

which follows from the fact that ∥ ∥ ∥ f -ˆ f a SS ∥ ∥ ∥ is independent of the noise terms ( ε i ) i ∈A . To proceed, we add and subtract ˆ f SS ( x ) inside the squared term:

<!-- formula-not-decoded -->

Using AM-GM inequality, we obtain

<!-- formula-not-decoded -->

Substituting this bound into the definition of R 2 ( f, ˆ f a SS ) , we get

<!-- formula-not-decoded -->

To prove the upper bound in (5), it suffices to find appropriate bounds for the two terms appearing in (29). We begin by analyzing the first term involving the honest estimator ˆ f SS :

<!-- formula-not-decoded -->

3 Note that any function f : [ a, b ] → R can be transformed with scaling and shifting into a function ˜ f : [0 , 1] → R without affecting its Sobolev regularity or the scaling of the associated metrics.

To do so, we use the following theorem, which is a direct consequence of [50, Therorem 1.1], specialized to the second-order Sobolev space setting:

Lemma 1. Let I = [ a, b ] ⊂ R be a bounded interval, and let the design points { x i } n i =1 ⊂ I satisfy the quasi-uniformity condition

<!-- formula-not-decoded -->

̸

for some constant k &gt; 0 , where

<!-- formula-not-decoded -->

Then, for any j = 0 , 1 , 2 , there exist constants λ 0 &gt; 0 , P 0 &gt; 0 , and Q 0 &gt; 0 , such that for all n -4 ≤ λ ≤ λ 0 , we have

<!-- formula-not-decoded -->

Here, ˆ f SS is the smoothing spline estimator applied to uncorrupted data, and f ( j ) denotes the j -th derivative of f . To bound the first term in (29), we invoke Lemma 1 with j = 0 , corresponding to the L 2 (Ω) error between the regression function f and the honest smoothing spline estimator ˆ f SS . This yields:

<!-- formula-not-decoded -->

where λ is the regularization parameter, and P 0 , Q 0 &gt; 0 are constants from Lemma 1.

To complete the proof of (5), we now seek to find an upper bound for the second term in (29), which captures the deviation between the adversarial and honest estimators:

<!-- formula-not-decoded -->

Note that from the kernel representations (24) and (25), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, for each x ∈ Ω ,

Note that for each i :

- If i / ∈ A , there is no corruption, and y i = ˜ y i .
- If i ∈ A , the adversary may modify y i , and since f ( x i ) , y i ∈ [ -M,M ] , we have

<!-- formula-not-decoded -->

Thus, the sum above reduces to and we can bound

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- ˜ .

This implies that

<!-- formula-not-decoded -->

where (a) and (b) follow from the Cauchy-Schwarz and AM-GM inequalities, respectively. Taking expectations and supremum over S yields

<!-- formula-not-decoded -->

Now, to complete the proof of (5), it remains to find an upper bound for the kernel supremum term

<!-- formula-not-decoded -->

Unfortunately, W n ( · , · ) does not admit an analytically tractable form [52, 53] for directly bounding its supremum in (13). However, a substantial body of research [52-55] has focused on approximating W n ( · , · ) with analytically tractable functions, known as equivalent kernels , denoted by ̂ W n ( x, s ) . We leverage such approximations in our analysis to derive an upper bound.

Recall that we define the empirical distribution function F n as

<!-- formula-not-decoded -->

We assume that the empirical distribution function F n converges to a cumulative distribution function F , i.e., α ( n ) := sup x ∈ Ω | F n ( x ) -F ( x ) | satisfies α ( n ) -→ 0 as n →∞ . Moreover, we assume that F ( x ) is differentiable on Ω with density p ( x ) = F ′ ( x ) , and that there exists a constant p min &gt; 0 such that

<!-- formula-not-decoded -->

To proceed, according to [49], we define the equivalent kernel ̂ W n ( x, s ) as

<!-- formula-not-decoded -->

where the phase function φ 0 ( x, s ) is given by

<!-- formula-not-decoded -->

Based on [49, Theorem 1], for sufficiently large n , we have

<!-- formula-not-decoded -->

where C &gt; 0 is a constant independent of n , and the bound holds uniformly over all x ∈ [0 , 1] and s ∈ [ τ 1 , τ 2 ] , where 0 &lt; τ 1 &lt; τ 2 &lt; 1 .

Now note that

<!-- formula-not-decoded -->

Using the uniform approximation property established in (43), we can bound the second term:

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

where (a) follows from the definition of ̂ W n ( x, x j ) in (41), and the fact that inf x ∈ Ω p ( x ) ≥ p min . Combining the decomposition in (29), the bound on the honest estimator error from (33), and the adversarial deviation bounds from (38) and (47), we obtain the final upper bound for R 2 ( f, ˆ f a SS ) stated in Theorem 1:

<!-- formula-not-decoded -->

Therefore, in the regime where λ → 0 as n → ∞ and λ &gt; n -2 &gt; n -4 , there exist constants E 1 , E 2 , E 3 such that for sufficiently large n ,

<!-- formula-not-decoded -->

Since λ 1 / 4 → 0 as n →∞ , the additive term λ 1 / 4 becomes negligible compared to 1 for sufficiently large n . Dropping this term and absorbing constants, we obtain

<!-- formula-not-decoded -->

For a continuous cumulative distribution function F , Serfling [65] shows that α ( n ) = n -1 / 2 log log n almost surely. Since λ &gt; n -2 , it follows that λ -1 / 4 α ( n ) → 0 as n →∞ . Therefore, for sufficiently large n , we have 1 + λ -1 / 4 α ( n ) &lt; 2 . As a result, we obtain

<!-- formula-not-decoded -->

This concludes the proof of the upper bound on R 2 ( f, ˆ f a SS ) in Theorem 1.

To complete the proof of Theorem 1, it remains to prove (6). To do so, we adopt a similar strategy as in the L 2 case, but adapted to the squared supremum norm. By the inequality ( a + b ) 2 ≤ 2 a 2 +2 b 2 , we have

<!-- formula-not-decoded -->

Taking expectation and supremum over S , we substitute into the definition of R ∞ and obtain

<!-- formula-not-decoded -->

From the pointwise bound established in (36), we have

<!-- formula-not-decoded -->

Applying the kernel estimate from (47), we conclude that

<!-- formula-not-decoded -->

Squaring both sides and taking expectation and supremum over S , we obtain

<!-- formula-not-decoded -->

To complete the proof of (6), it remains to find an upper bound for the first term in (53), namely

<!-- formula-not-decoded -->

To do so, Since f -ˆ f SS ∈ W 2 (Ω) , we can leverage Sobolev norms inequalities [56] and use the same arguments as in [66, Lemma 5] and obtain:

<!-- formula-not-decoded -->

Taking expectations on both sides of (57) and applying the Cauchy-Schwarz inequality, we obtain:

<!-- formula-not-decoded -->

Applying Lemma 1 with j = 0 and j = 1 , we can bound the right-hand side using:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substituting the bounds from (59) and (60) into (57), we obtain

<!-- formula-not-decoded -->

Combining the decomposition in (53) with the bounds from (61) and (56), we obtain the following upper bound in the regime where λ → 0 as n →∞ and λ &gt; n -2 ≥ n -4 :

<!-- formula-not-decoded -->

We now multiply and divide the first term by λ 1 / 4 , yielding:

<!-- formula-not-decoded -->

By arguments similar to those used in the bound for R 2 ( f, ˆ f a SS ) , we can neglect both λ 1 / 4 and λ -1 / 4 α ( n ) compared to 1 for sufficiently large n . Thus, we obtain

<!-- formula-not-decoded -->

This completes the proof of the upper bound on R ∞ ( f, ˆ f a SS ) in (6), and thereby concludes the proof of Theorem 1.

## B Proof of Theorem 2

To prove Theorem 2, we first state and prove Lemma 2.

Lemma 2. Let P 1 and P 2 denote two probability density functions of two distributions with common variance σ 2 &gt; 0 . Then, there exists α ∈ [0 , 1] , and two probability density functions Q 1 and Q 2 such that

<!-- formula-not-decoded -->

where Q 1 and Q 2 are explicitly constructed from P 1 and P 2 .

Proof. Define α as:

<!-- formula-not-decoded -->

Next, define Q 1 and Q 2 as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where 1 {·} denotes the indicator function.

By construction, both Q 1 ( u ) and Q 2 ( u ) are non-negative since the indicator functions restrict the support to regions where the corresponding differences are non-negative. We now show that Q 1 and Q 2 are valid probability density functions. Consider:

<!-- formula-not-decoded -->

By symmetry, the same argument shows that ∫ Q 2 ( u ) du = 1 as well.

Hence, both Q 1 and Q 2 are valid densities. With this choice of α , the following identity holds:

<!-- formula-not-decoded -->

This completes the proof.

We now prove Theorem 2, building on Lemma 2. We begin by establishing the lower bound for the metric R 2 , as stated in (18); the proof for R ∞ , given in (19), follows by a similar argument. To do so, we reduce the minimax risk in (18) and (19) to a hypothesis testing problem [57]. Specifically, we construct two functions f 1 and f 2 in W 2 (Ω) with L 2 and L ∞ distance, bounded away from zero (see Figure 2). However, given n samples from either function, an adversary can corrupt up to q of them, making it impossible for any estimator to reliably distinguish between f 1 and f 2 . Consequently,

no estimation approach can identify which function generated the data, and the average hypothesis testing error remains 1 / 2 . Applying [57, Proposition 5.1] yields the lower bounds in Theorem 2. The details of the proof is as follows.

Throughout the proof, we assume a fixed design given by x i = i/n and ε i ∼ N (0 , σ 2 ) are i.i.d noise samples drawn from a normal distribution with zero mean and variance σ 2 , for i ∈ [ n ] .

Let r q = q n and define ε q = r 2 q . We construct two functions, f 1 and f 2 , as follows. Set

<!-- formula-not-decoded -->

To define f 2 , we construct a degree-5 polynomial g ( x ) on the interval [ r q -ε q , r q ] that satisfies the following conditions:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

These six conditions uniquely determine a polynomial of degree 5, since there are six coefficients to solve for. Hence, such a polynomial g exists and can be explicitly constructed. Now, define f 2 on the interval [0 , 1] by

<!-- formula-not-decoded -->

It is straightforward to verify that f 2 ∈ W 2 ([0 , 1]) , since both f 2 and its first and second derivatives have bounded norms over Ω (See Figure 2).

Note that f 1 and f 2 are close but not identical; their differences are concentrated on the interval [0 , r q ] , and will be used to construct the lower bound.

For each sample x i , the adversary proceeds as follows:

- If x i ≥ r q , then f 1 ( x i ) = f 2 ( x i ) , so no corruption is needed: both models produce identical distributions for ˜ y i .

̸

<!-- formula-not-decoded -->

- If x i &lt; r q , then f 1 ( x i ) = f 2 ( x i ) , and the adversary applies Lemma 2 to the pair of normal distributions

obtaining a scalar α i ∈ [0 , 1] and auxiliary distributions Q ( i ) 1 and Q ( i ) 2 such that

<!-- formula-not-decoded -->

For each such i , the adversary acts:

- -With probability 1 -α i , leave y i uncorrupted (i.e., drawn from P ( i ) 1 if f = f 1 , or from P ( i ) 2 if f = f 2 ).
- -With probability α i , the adversary replaces y i by a draw from Q ( i ) 1 if the true function is f 1 , and from Q ( i ) 2 if the true function is f 2 .

For the above adversarial strategy, we have |A| ≤ r q n = q . In addition, note that under model f 1 , conditionally on x i , the corrupted response ˜ y i is distributed according to (1 -α i ) P ( i ) 1 + α i Q ( i ) 1 , and under model f 2 , it is distributed according to (1 -α i ) P ( i ) 2 + α i Q ( i ) 2 . By construction of Q ( i ) 1 and Q ( i ) 2 in Lemma 2, these two mixtures are identical for each i .

Therefore, after adversarial corruption, the distribution of all observed data { ˜ y i } n i =1 is identical under f 1 and f 2 . More precisely:

- For all i with x i &gt; r q , we have f 1 ( x i ) = f 2 ( x i ) , and hence P ( i ) 1 = P ( i ) 2 ; no corruption is needed, and the distribution of ˜ y i is the same under both models.
- For all i with x i ≤ r q , the adversary modifies the responses exactly so that the overall conditional distribution of ˜ y i is matched across the two models.

Note that the constructed functions f 1 and f 2 are not identical: by definition, their difference measured by the metrics introduced in (1) and (2) is nonzero. However, the adversarial corruption strategy described above renders the corrupted data distribution identical under both f 1 and f 2 . Consequently, no estimator can achieve better performance than random guessing between the two hypotheses. As a result, the minimax error under adversarial corruption remains bounded away from zero, establishing a nontrivial lower bound.

To prove (18), by starting from the definition of R 2 ( f, ˆ f ) , we have

<!-- formula-not-decoded -->

where the expectation is over the noise ε , and the supremum is taken over all admissible adversarial strategies S . Since Theorem 2 considers the worst-case function f , we obtain

<!-- formula-not-decoded -->

As established earlier, the adversary makes the corrupted data distribution identical under both f 1 and f 2 . Formally, let P ( A ) f 1 and P ( A ) f 2 denote the distributions over the corrupted datasets when the ground truth is f 1 or f 2 , respectively. Thus, we have:

<!-- formula-not-decoded -->

That is, the total variation distance satisfies:

<!-- formula-not-decoded -->

This guarantees that no estimator can distinguish between them better than random guessing. To formalize this, we use Le Cam's two-point method [67, 68] (the hypothesis testing between two points), which states that for any estimator ˆ f and any pair f 1 , f 2 ,

<!-- formula-not-decoded -->

Using (78), we obtain the following lower bound:

<!-- formula-not-decoded -->

Consequently, following (77) we have

<!-- formula-not-decoded -->

Recall that f 1 ( x ) = 0 , and

<!-- formula-not-decoded -->

where g ( x ) is a degree-5 polynomial satisfying the smoothness and boundary conditions described earlier. Therefore,

<!-- formula-not-decoded -->

Note that since ε q = r 2 q , we have

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

Moreover, even in the absence of adversarial corruption (i.e., q = 0 ), it is well known from classical minimax theory in nonparametric regression [24] that

<!-- formula-not-decoded -->

Combining the two regimes, we obtain the following lower bound on the adversarial error:

<!-- formula-not-decoded -->

This completes the proof of (18). To complete the proof of Theorem 2, we now establish a lower bound for R ∞ . Recall that

<!-- formula-not-decoded -->

where the expectation is taken over the noise ε , and the supremum is over all adversarial corruption strategies S . The norm ∥·∥ L ∞ (Ω) denotes the supremum norm over the interval [0 , 1] .

As in the case of R 2 , the adversary can construct corrupted data distributions under f 1 and f 2 that are indistinguishable. Consequently, no estimator can distinguish between the two hypotheses better than random guessing. Applying Le Cam's two-point method [67, 68] to the L ∞ loss, we obtain:

<!-- formula-not-decoded -->

Therefore, we have:

<!-- formula-not-decoded -->

Since f 1 ( x ) = 0 , we have ∥ f 1 -f 2 ∥ L ∞ (Ω) = ∥ f 2 ∥ L ∞ (Ω) ≥ f 2 (0) . From the definition of f 2 , we have f 2 (0) = r q . Therefore,

<!-- formula-not-decoded -->

Moreover, in the absence of adversarial corruption (i.e., q = 0 ), the standard minimax rate for estimation under the supremum norm is known to satisfy (see [24])

<!-- formula-not-decoded -->

Combining both contributions, we conclude that

<!-- formula-not-decoded -->

This completes the proof of (19), and thereby the proof of Theorem 2.

## C Gaussian Setting Experiments

Figure 5: Log-log plots showing the convergence rate of the cubic smoothing spline estimator ˆ f = ˆ f a SS for f ( x ) = x sin( x ) under a Gaussian design. The top row plots are results for q = n 0 . 3 , with theoretical rates of O ( n -0 . 8 ) for R 2 ( f, ˆ f ) and O ( n -0 . 6 ) for R ∞ ( f, ˆ f ) . The bottom row corresponds to a higher corruption level, q = n 0 . 6 , with respective theoretical upper bounds of O ( n -0 . 53 ) and O ( n -0 . 48 ) .

<!-- image -->

.

Figure 6: Log-log plots showing the convergence behavior of the cubic smoothing spline estimator ˆ f = ˆ f a SS when the ground-truth function is an MLP, under the Gaussian design. The top row corresponds to the case q = n 0 . 3 , with theoretical convergence rates of O ( n -0 . 8 ) for R 2 ( f, ˆ f ) and O ( n -0 . 6 ) for R ∞ ( f, ˆ f ) . The bottom row shows results for a higher corruption level, q = n 0 . 6 , with respective theoretical upper bounds of O ( n -0 . 53 ) and O ( n -0 . 48 ) .

<!-- image -->

.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We detailed our contributions clearly in the abstract and the introduction sections of the paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We provided all the details regarding the assumptions, conditions, and limitations of the main problem (Section 2), theorems (Section 3), the experiments (Section 4), as well as the conclusion (Section 6) in the paper.

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

Justification: Our paper includes theoretical results in Section 3. In our theorems, we clearly mentioned all the required assumptions, and a complete (and correct) proof of them is available in appendices. Please see Section 2 for a full definition of the problem and the notations used in the paper are described in introduction 1.

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

Justification: We have provided details regarding our empirical evaluations in Section 4 in the paper.

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

Answer: [No]

Justification: We provided references to all packages that we used in the paper. Regarding the code, we are happy to share it later if required.

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

Justification: We provided full experimental details in the paper (see Section 4).

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The details are provided in Section 4.

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

Justification: We provided all the details regarding our experiments in Section 4.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We followed the NeurIPS code of ethics in our paper.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our paper is focused on developing a theoretical foundation for adversarial robustness of non-parametric regression. We believe this work has no direct societal impact that should be explained in the paper.

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

Justification: This is not applicable to our work and this paper poses no such risks.

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

Justification: Our paper does not involve these.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our paper does not involve these.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: We just use LLM for writing and editing purposes.

Guidelines: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.