## Beyond Benign Overfitting in Nadaraya-Watson Interpolators

## Daniel Barzilai

Weizmann Institute of Science daniel.barzilai@weizmann.ac.il

## Guy Kornowski

Weizmann Institute of Science guy.kornowski@weizmann.ac.il

## Ohad Shamir

Weizmann Institute of Science ohad.shamir@weizmann.ac.il

## Abstract

In recent years, there has been much interest in understanding the generalization behavior of interpolating predictors, which overfit on noisy training data. Whereas standard analyses are concerned with whether a method is consistent or not, recent observations have shown that even inconsistent predictors can generalize well. In this work, we revisit the classic interpolating Nadaraya-Watson (NW) estimator (also known as Shepard's method), and study its generalization capabilities through this modern viewpoint. In particular, by varying a single bandwidth-like hyperparameter, we prove the existence of multiple overfitting behaviors, ranging non-monotonically from catastrophic, through benign, to tempered. Our results highlight how even classical interpolating methods can exhibit intricate generalization behaviors. In addition, for the purpose of tuning the hyperparameter, the results suggest that over-estimating the intrinsic dimension of the data is less harmful than under-estimating it. Numerical experiments complement our theory, demonstrating the same phenomena.

## 1 Introduction

The incredible success of over-parameterized machine learning models has spurred a substantial body of work, aimed at understanding the generalization behavior of interpolating methods (which perfectly fit the training data). In particular, according to classical statistical analyses, interpolating inherently noisy training data can be harmful in terms of test error, due to the bias-variance tradeoff. However, contemporary interpolating methods seem to defy this common wisdom [1, 2]. Therefore, a current fundamental question in statistical learning is to understand when models that perfectly fit noisy training data can still achieve strong generalization performance.

The notion of what it means to generalize well has somewhat changed over the years. Classical analysis has been mostly concerned with whether or not a method is consistent, meaning that asymptotically (as the training set size increases), the excess risk converges to zero. By now, several settings have been identified where even interpolating models may be consistent, a phenomenon known as 'benign overfitting' [3-6]. However, following Mallinar et al. [7], a more nuanced view of overfitting has emerged, based on the observation that not all inconsistent learning rules are necessarily unsatisfactory.

In particular, it has been argued both empirically and theoretically that in many realistic settings, benign overfitting may not occur, yet interpolating methods may still overfit in a 'tempered' manner, meaning that their excess risk is proportional to the Bayes error. On the other hand, in some situations

overfitting may indeed be 'catastrophic', leading to substantial degradation in performance even in the presence of very little noise. The difference between these regimes is significant when the amount of noise in the data is relatively small, and in such a case, models that overfit in a tempered manner may still generalize relatively well, while catastrophic methods do not. These observations led to several recent works aiming at characterizing which overfitting profiles occur in different settings beyond consistency, mostly for kernel regression and shallow ReLU networks [8-14]. We note that one classical example of tempered overfitting is 1 -nearest neighbor, which asymptotically achieves at most twice the Bayes error [15]. Moreover, results of a similar flavor are known for k -nearest neighbor where k &gt; 1 (see [16]). However, unlike the interpolating predictors we study here, k -nearest neighbors do not necessarily interpolate the training data when k &gt; 1 .

With this modern nuanced approach in mind, we revisit in this work one of the earliest and most classical learning rules, namely the Nadaraya-Watson (NW) estimator [17, 18]. In line with recent analysis focusing on interpolating predictors, we focus on an interpolating variant of the NW estimator, for binary classification: given (possibly noisy) classification data S = ( x i , y i ) i m =1 ⊂ R d ×{± 1 } sampled from some continuous distribution D , and given some β &gt; 0 , we consider the predictor

<!-- formula-not-decoded -->

The predictor in Eq. (1) has a long history in the literature and is known by many different names, such as Shepard's method, inverse distance weighting (IDW), the Hilbert kernel estimate, and singular kernel classification (see Section 2 for a full discussion).

̸

Notably, for any choice of β &gt; 0 , ˆ h β interpolates the training set, meaning that ˆ h β ( x i ) = y i . We will study the predictor's generalization in 'noisy' classification tasks: we assume there exists a ground truth f ∗ : R d →{± 1 } (satisfying mild regularity assumptions), so that for each sampled point x , its associated label y ∈ {± 1 } satisfies Pr[ y = f ∗ ( x ) | x ] = 1 -p for some p ∈ (0 , 0 . 49) . Clearly, for this distribution, no predictor can achieve expected classification error better than p &gt; 0 . However, interpolating predictors achieve 0 training error on the training set, and thus by definition overfit. We are interested in studying the ability of these predictors to achieve low classification error with respect to the underlying distribution. Factoring out the inevitable error due to noise, we can measure this via the 'clean' classification error Pr x ∼D x [ ˆ h β ( x ) = f ∗ ( x )] , which measures how well ˆ h β captures the ground truth function f ∗ .

As our starting point, we recall that ˆ h β is known to exhibit benign overfitting when β = d precisely:

Theorem 1.1 (Devroye et al. [19]) . Suppose D x has a density on R d , and let β = d . For any noise level p ∈ (0 , 0 . 49) , it holds that the clean classification error of ˆ h β goes to zero as m →∞ , i.e. ˆ h β exhibits benign overfitting.

̸

In other words, although training labels are flipped with probability p ∈ (0 , 0 . 49) , the predictor is asymptotically consistent, and thus predicts according to the ground truth f ∗ . Furthermore, Devroye et al. [19] also informally argued that setting β = d is inconsistent in general, and therefore excess risk should be expected. Nonetheless, the behavior of the predictor ˆ h β beyond the benign/consistent setting is not known prior to this work.

̸

In this paper, in light of the recent interest in inconsistent interpolation methods, we characterize the price of overfitting in the inconsistent regime β = d . What is the nature of the inconsistency for β = d ? Is the overfitting tempered, or in fact catastrophic? As our main contribution, we answer these questions and prove the following asymmetric behavior:

̸

Theorem 1.2 (Main results, informal) . For any dimension d ∈ N and noise level p ∈ (0 , 0 . 49) , the following hold asymptotically as m →∞ :

- ('Tempered' overfitting) For any β &gt; d , the clean classification error of ˆ h β is between Ω(poly( p )) and ˜ O ( p ) .
- ('Catastrophic' overfitting) For any β &lt; d , there is some f ∗ for which ˆ h β will suffer constant clean classification error, independently of p .

We summarize the overfitting profile that unfolds in Figure 1, with an illustration of the NadarayaWatson interpolator in one dimension. These results provide a modern analysis of a classical learning

rule, uncovering a range of generalization behaviors: By varying a single hyperparameter, these behaviors range non-monotonically from catastrophic to tempered overfitting, with a delicate sliver of benign overfitting behavior in between. Our results highlight how intricate generalization behaviors, including the full range from benign to catastrophic overfitting, can appear in simple and well-known interpolating learning rules. To the best of our knowledge, for kernel interpolators, there is no other example of a single kernel provably exhibiting all three types of overfitting as we do here (even with a varying bandwidth).

Moreover, the results provide an interesting insight about the optimal tuning of β : Although Theorem 1.1 might seem to suggest that the optimal value of β is simply the input dimension d , it does not cover the common case where the data has some intrinsic dimension d int &lt; d (due to the requirement that D x has a density on R d ). In that situation, our analysis suggests that the optimal value for β is in fact d int , not d . Unfortunately, d int is generally not known, and can only be estimated. In that case, our results suggest that setting β to an over-estimate of d int (namely, choosing some β &gt; d int ) is much preferable to under-estimating it, as the former leads to tempered overfitting, whereas the latter may lead to catastrophic overfitting. We further discuss this in Remark 5.2, and in Section 6 we present numerical evidence supporting this claim.

Figure 1: (a): Illustration of the entire overfitting profile of the NW interpolator given by Eq. (1). (b): Toy illustration of the NW interpolator in dimension d = 1 with noisy data. (Left) Catastrophic overfitting for β &lt; d : the prediction at each point is influenced too heavily by far-away points, and therefore the predictor does not capture the general structure of the ground truth function f ∗ . (Middle) Benign overfitting for β = d : asymptotically the excess risk will be Bayes-optimal. (Right) Tempered overfitting for β &gt; d , the prediction at each point is influenced too heavily by nearby points, so the predictor misclassifies large regions around label-flipped points, but only around them.

<!-- image -->

The paper is structured as follows. In Section 2, we review related work. In Section 3 we formally present the discussed setting. In Section 4, we present our result for the tempered regime β &gt; d . In Section 5 we present our result for the catastrophic regime β &lt; d . In Section 6 we provide some illustrative experiments to complement our theoretical findings. We conclude in Section 7. All of the results in the main text include proof sketches, while full proofs appear in the appendix.

## 2 Related work

Nadaraya-Watson kernel estimator. The Nadaraya-Watson (NW) estimator was introduced independently in the seminal works of Nadaraya [17] and Watson [18]. Later, and again independently, in the context of reconstructing smooth surfaces, Shepard [20] used a method referred to as Inverse Distance Weighting (IDW), which is in fact a NW estimator with respect to certain kernels leading to interpolation, identical to those we consider in this work. To the best of our knowledge, Devroye et al. [19] provided the first statistical guarantees for such interpolating NW estimators (which they

called the Hilbert kernel), showing that the predictor given by Eq. (1) with β = d is asymptotically consistent. For a more general discussion on so called 'kernel rules', see [16, Chapter 10]. In more recent works, Belkin et al. [21] derived non-asymptotic rates showing consistency under a slight variation of the kernel. Radhakrishnan et al. [22], Eilers et al. [23] showed that in certain cases, neural networks in the NTK regime behave approximately as the NW estimator, and leverage this to show consistency. Abedsoltan et al. [24] showed that interpolating NW estimators can be used in a way that enables in-context learning.

Overfitting and generalization. There is a substantial body of work aimed at analyzing the generalization properties of interpolating predictors that overfit noisy training data. Many works study settings in which interpolating predictors exhibit benign overfitting, such as linear predictors [3, 25-30], kernel methods [31, 32, 6], and other learning rules [19, 33, 1].

On the other hand, there is also a notable line of work studying the limitations of generalization bounds in interpolating regimes [34, 2, 35]. In particular, several works showed that various kernel interpolating methods are not consistent in any fixed dimension [36-38], or whenever the number of samples scales as an integer-degree polynomial with the dimension [39, 40, 12, 41].

Motivated by these results and by additional empirical evidence, Mallinar et al. [7] proposed a more nuanced view of interpolating predictors, coining the term tempered overfitting to refer to settings in which the asymptotic risk is strictly worse than optimal, but is still better than a random guess. A well-known example is the classic 1 -nearest-neighbor interpolating method, for which the excess risk scales linearly with the probability of a label flip [15]. Several works subsequently studied settings in which tempered overfitting occurs in the context of kernel methods [11, 12, 42], and for other interpolation rules [8, 9, 43].

Finally, some works studied settings in which interpolating with kernels is in fact catastrophic , meaning that the excess error is lower bounded by a constant which is independent of the noise level, leading to substantial risk even in the presence of very little noise [9, 10, 13, 14].

We note that our proof techniques differ from most known results for kernel interpolators, which typically rely on a spectral analysis. However, this often requires additional non-trivial assumptions (e.g. Gaussian universality). By contrast, our proofs are based on characterizing the 'locality' of the predictor.

Varying kernel bandwidth. Several works considered generalization bounds that hold uniformly over a family of kernels, parameterized by a bandwidth parameter [36, 44, 37, 38, 13]. The bandwidth plays the same role as the parameter β in this paper, controlling how local/global the kernel is. Specifically, these works showed that in fixed dimensions various kernels are asymptotically inconsistent for all bandwidths. Medvedev et al. [13] showed that with large enough noise, the Gaussian kernel with any bandwidth is at least as bad as a constant predictor, which we classify as catastrophic. As far as we know, our paper provides the first known example of a kernel method provably exhibiting all types of overfitting behaviors in fixed dimensions by varying the bandwidth alone.

## 3 Preliminaries

Notation. We use bold-faced font to denote vectors, e.g. x ∈ R d , and denote by ∥ x ∥ the Euclidean norm. We let [ n ] := { 1 , . . . , n } . Given some set A ⊆ R d and a function f , we denote its restriction by f | A : A → R , and by Unif( A ) the uniform distribution over A . We let B ( x , r ) := { z | ∥ x -z ∥ ≤ r } be the ball of radius r centered at x . We denote by d = equality in distribution. We use the standard big-O notation, with O ( · ) , Θ( · ) and Ω( · ) hiding absolute constants that do not depend on problem parameters, and ˜ O ( · ) , ˜ Ω( · ) additionally hiding logarithmic factors. Given some parameter (or set of parameters) θ , we denote by c ( θ ) , C ( θ ) , C 1 ( θ ) , ˜ C ( θ ) etc. positive constants that depend on θ .

Setting. Given some target function f ∗ : R d → {± 1 } , we consider a classification task based on noisy training data S = ( x i , y i ) i m =1 ⊂ R d × {± 1 } , such that x 1 , . . . , x m ∼ D x are sampled from some distribution D x with a density µ , and for each i ∈ [ m ] independently, y i = f ∗ ( x i ) with probability 1 -p or else y i = -f ∗ ( x i ) with probability p ∈ (0 , 0 . 49) . We note that while we focus

on a fixed noise level p for simplicity, our results can also be extended to the case where p varies smoothly with x .

Given the predictor ˆ h β introduced in Eq. (1), we denote the asymptotic clean classification error by 1

̸

<!-- formula-not-decoded -->

Throughout the paper we impose the following mild regularity assumptions on µ and f ∗ :

Assumption 3.1. We assume µ is continuous at almost every x ∈ R d . We also assume that for almost every x ∈ R d , there is a neighborhood B x ⊃ { x } such that f ∗ | B x ≡ f ∗ ( x ) .

We note that the assumptions above are very mild. Indeed, any density is Lebesgue integrable, whereas our assumption for µ is equivalent to it being Riemann integrable. As for f ∗ , the assumption asserts that its associated decision boundary has zero measure, ruling out pathological functions.

Types of overfitting. We study the asymptotic error guaranteed by ˆ h β in a 'minimax' sense, namely uniformly over µ, f ∗ that satisfy Assumption 3.1. Under the described setting with noise level p ∈ (0 , 0 . 49) , we say that:

- ˆ h β exhibits benign overfitting if L ( ˆ h β ) = 0 ;
- Else, ˆ h β exhibits tempered overfitting if L ( ˆ h β ) scales monotonically with p : there exists φ : [0 , 1] → [0 , 1] non-decreasing, continuous with φ (0) = 0 , so that L ( ˆ h β ) ≤ φ ( p ) ;
- ˆ h β exhibits catastrophic overfitting if there exist some µ, f ∗ (satisfying the regularity assumptions) such that L ( ˆ h β ) is lower bounded by a positive constant (independent of p ).

We remark that the latter definition of catastrophic overfitting slightly differs from the one of Mallinar et al. [7], which called the method catastrophic only if L ( ˆ h β ) = 1 2 . Medvedev et al. [13] noted that the latter definition can result in even the most trivial predictor, a function that is constant outside the training set, being classified as tempered instead of catastrophic. We therefore find the formalization above more suitable, which also coincides with previous works [8, 9, 12, 13, 43].

## 4 Tempered overfitting

We start by presenting our main result for the β &gt; d parameter regime, establishing tempered overfitting of the predictor ˆ h β :

Theorem 4.1. For any d ∈ N , any β &gt; d , any density µ and target function f ∗ satisfying Assumption 3.1, and any noise level p ∈ (0 , 0 . 49) , it holds that

<!-- formula-not-decoded -->

where c ( β/d ) = ( 8 · 2 β/d β/d -1 ) 1 β/d -1 &gt; 0 , and C 1 ( β/d ) , C 2 ( β/d ) &gt; 0 are constants that depend only on the ratio β/d .

In particular, the theorem implies that for any β &gt; d it holds that L ( ˆ h β ) = ˜ O ( p ) , hence in low noise regimes the error is never too large. Moreover, we note that the lower bound (of the form Ω(poly( p )) for any β &gt; d ) holds for any target function satisfying mild regularity assumptions. Therefore, the tempered cost of overfitting holds not only in a minimax sense, but for any instance.

Further note that since we know that β = d leads to benign overfitting, one should expect the lower bound in Theorem 4.1 to approach 0 as β → d + . Indeed, the lower bound's polynomial degree

(

·

8

2

β/d

β/d satisfies

c

(

β/d

) =

-

1

+

-→ ∞

, and thus

p

+

-→

0

.

1 Technically, the limit may not exist in general. In that case, our lower bounds hold for the lim inf m →∞ , while our upper bounds hold for the lim sup m →∞ , and therefore both hold for all partial limits.

2 To be precise, one needs to make sure that the constant C 1 ( β/d ) does not blow up, which is indeed the case.

)

1

β/d

-

β

→

d

c

(

β/d

)

β

→

d

1

2

We provide below a sketch of the main ideas that appear in the proof of Theorem 4.1, which is provided in Appendix B. In a nutshell, the proof establishes that when β &gt; d , the predictor ˆ h β is highly local , and thus prediction at a test point is affected by flipped labels nearby, yet only by them. The proof essentially shows that in this parameter regime, ˆ h β behaves similar to the k nearest neighbor ( k -NN) method for some finite k that depends on β/d (although notably, as opposed to ˆ h β , k -NN does not interpolate), and has a similarly tempered generalization guarantee accordingly.

Proof sketch of Theorem 4.1. Looking at some test point x ∈ R d , we are interested in understanding the prediction ˆ h β ( x ) . Clearly, by definition in Eq. (1), the prediction depends on the random variables ∥ x -x i ∥ -β for i ∈ [ m ] , so that closer datapoints have a great affect on the prediction at x . Denote by y (1) , . . . , y ( m ) the labels ordered according to the distance of their corresponding datapoints, namely ∥ ∥ x -x (1) ∥ ∥ ≤ ∥ ∥ x -x (2) ∥ ∥ ≤ · · · ≤ ∥ ∥ x -x ( m ) ∥ ∥ . By analyzing the distribution of distances from the sample to x , for datapoints sufficiently close to x we can jointly approximate the random variables by ∥ x -x ( i ) ∥ -β ≈ µ ( x )( ∑ m +1 i =1 E i ) β/d / ( ∑ i j =1 E j ) β/d , where E 1 , . . . , E m i.i.d. ∼ exp(1) are standard exponential random variables. Furthermore, datapoints which are more than some constant distance away from x can contribute at most a constant, so for some m ′ &lt; m we obtain

<!-- formula-not-decoded -->

Since E [ ∑ i j =1 E j ] = i , we apply concentration bounds for sums of exponential variables to argue that with high probability ∑ i j =1 E j ≈ i simultaneously over all i ∈ N , so the prediction is roughly

<!-- formula-not-decoded -->

since O ( m ) ≪ ( m +1) β/d is asymptotically negligible and µ ( x )( m +1) β/d &gt; 0 .

Crucially, for any β &gt; d , the sum above converges, and therefore there exists a constant k ∈ N (that depends only on the ratio β/d ) so that the tail is smaller than the first k summands:

<!-- formula-not-decoded -->

Therefore, under the event that all nearby labels coincide, the prediction depends only on the k nearest neighbors, and we would get that predictor returns their value. By Assumption 3.1, for sufficiently large sample size m and fixed k , for almost every x the k nearest neighbors should be labeled the same as x , namely f ∗ ( x ) = f ∗ ( x (1) ) = · · · = f ∗ ( x ( k ) ) . So overall, we see that

̸

̸

<!-- formula-not-decoded -->

and similarly

<!-- formula-not-decoded -->

̸

̸

The two inequalities above show the desired upper and lower bounds on the prediction error.

## 5 Catastrophic overfitting

We now turn to present our main result for the β &lt; d parameter regime, establishing that ˆ h β can catastrophically overfit:

Theorem 5.1. For any d ∈ N and any 0 &lt; β &lt; d , there exist a density µ and a target function f ∗ satisfying Assumption 3.1, such that for some absolute constants C 1 , C 2 ∈ (0 , 1) , and c ( β, d ) := C β 1 · (1 -β/d ) &gt; 0 , it holds for any p ∈ (0 , 0 . 49) that

<!-- formula-not-decoded -->

The theorem states that whenever β &lt; d , the error can be arbitrarily larger than the noise level, since L ( ˆ h β ) = Ω(1) even as p → 0 . Note that since the benign overfitting result for β = d holds over any distribution and target function (under the same regularity assumptions), the fact that the lower bound of Theorem 5.1 approaches 0 as β → d is to be expected.

Remark 5.2. Interestingly, the only role played by d in the proofs of Theorems 4.1 and 5.1 is the fact that locally, the probability mass scales as ∫ B ( x ,r ) µ ≍ r d (for almost all x and small r &gt; 0 ). Accordingly, when the data distribution is supported on a lower dimensional manifold of dimension d int &lt; d , the result suggests that tempered overfitting occurs whenever β &gt; d int , and that catastrophic overfitting can occur whenever β &lt; d int . Although we do not attempt to formalize it in this paper, 3 we conjecture that in general the parameter d can be replaced by d int in all our results. Since d int generally can only be estimated, it suggests a potential practical implication: Setting β to an over-estimate of d int is less harmful than under-estimating it, as the former leads to tempered overfitting whereas the latter may lead to catastrophic overfitting. This is further supported by our experiments in Section 6.

We provide below a sketch of the main ideas of the proof, which is provided in Appendix C. Notably, the main idea behind the proof is quite different from that of Theorem 4.1. There, the analysis was highly local , i.e. for every test point x we showed that we can restrict our analysis to a small neighborhood around that point. In contrast, the reason we will obtain catastrophic overfitting for β &lt; d is precisely that the predictor is too global , as we will see that for every test point x , all points x i in the training set have a non-negligible effect on ˆ h β ( x ) . Our proof essentially shows that whenever a small region of constant probability mass is surrounded by the opposite label, the predictor will mislabel it, incurring a constant error. Our construction is therefore quite generic, and we expect the same intuition to extend to many target functions f ∗ . The full proof can be found in the appendix.

Proof sketch of Theorem 5.1. We will construct an explicit distribution and target function for which ˆ h β exhibits catastrophic overfitting. The distribution we consider consists of an inner ball of constant probability mass labeled -1 , and an outer annulus labeled +1 , as illustrated in Figure 2. Specifically, we denote c := c ( β, d ) = C β 1 · (1 -β/d ) for some absolute constant C 1 &gt; 0 to be specified later, and consider the following density and target function:

<!-- formula-not-decoded -->

We consider a test point x with ∥ x ∥ ≤ 1 4 , and will show that for sufficiently large m , with high probability x will be misclassified as +1 . This implies the desired result, since then

<!-- formula-not-decoded -->

To that end, we decompose

<!-- formula-not-decoded -->

3 This should not be difficult in principle, but the proofs would become substantially more technical when the manifold is non-linear.

Figure 2: Illustration of the lower bound construction used in the proof of Theorem 5.1. When β &lt; d , the inner circle will be misclassified as +1 with high probability, inducing constant error.

<!-- image -->

where T 1 crudely bounds the contribution of points in the inner circle, T 2 is the expected contribution of outer points labeled 1 , and T 3 is a perturbation term. Noting that T 2 &gt; 0 , our goal is to show that T 2 dominates the expression above, implying that Eq. (3) is positive and thus h β ( x ) = 1 .

Let k := ∣ ∣ { i : ∥ x i ∥ ≤ 1 4 } ∣ ∣ denote the number of points inside the inner ball, and note that we can expect k ≈ E [ k ] = cm . To bound T 1 , we express its distribution using exponential random variables in a manner that is similar to the proof of Theorem 4.1. Specifically, for standard exponential random variables E 1 , . . . , E m i.i.d. ∼ exp(1) , we show that with high probability

<!-- formula-not-decoded -->

where (1) uses concentration bounds on the sums of exponential random variables to argue that ∑ i j =1 E j ≈ i , and (2) follows from showing k ≈ cm .

To show that T 2 is sufficiently large, we use the fact that ∥ x -x i ∥ ≤ ∥ x ∥ + ∥ x i ∥ ≤ 5 4 , and that ∣ ∣ { i : ∥ x i ∥ ≥ 3 4 }∣ ∣ ≈ (1 -c ) m ≥ 1 2 m with high probability to obtain

<!-- formula-not-decoded -->

Lastly, we show that T 3 is asymptotically negligible, by noting that E [ T 3 ] = 0 hence T 3 = o ( m ) with high probability by Hoeffding's inequality. Thus Eq. (3) becomes

<!-- formula-not-decoded -->

Overall, we see that the right-hand side above is positive as long as c = C β 1 · ( 1 -β d ) &lt; 1 -β/d 5 β , or equivalently C 1 &lt; 1 5 , meaning that ˆ h β ( x ) = 1 even though f ∗ ( x ) = -1 .

## 6 Experiments

In this section, we provide numerical simulations that illustrate and complement our theoretical findings. In all experiments, we sample m datapoints according to some distribution, flip each label independently with probability p , and plot the clean test error of ˆ h β for various values of β . We ran each experiment 50 times, and plotted the average error surrounded by a 95% confidence interval.

## 6.1 Synthetic data

We start by discussing several experiments with synthetic data distributions.

Warm up: one dimensional data. In our first experiment, we considered data in dimension d = 1 distributed according to the construction considered in the proof of Theorem 5.1. In particular, we consider

<!-- formula-not-decoded -->

In Figure 3, on the left we plot the results for m = 2000 and various values of p , and on the right we fix p = 0 . 04 and vary m .

<!-- image -->

β

β

Figure 3: The classification error of ˆ h β for varying values of β , with data in dimension d = 1 given by Eq. (4). On the left, m = 2000 is fixed, p varies. On the right, p = 0 . 04 is fixed, m varies. Best viewed in color.

As seen in Figure 3, the generalization is highly asymmetric with respect to β . For β &lt; 1 , the test error degrades independently of the noise level p , and quickly reaches 0 . 1 in all cases, illustrating that the predictor errors on the negative labels (which have 0 . 1 probability mass). On the other hand, for β &gt; 1 , the test error exhibits a gradual deterioration. Moreover, we see this deterioration is controlled by the noise level p , matching our theoretical finding. The right figure illustrates all of the discussed phenomena hold similarly for moderate sample sizes, which complements our asymptotic analysis.

Spherical data. In our second experiment, we consider a similar distribution over the unit sphere S 2 ⊂ R 3 , where the inner negatively labeled region is a spherical cap. In particular, consider the spherical cap defined by A := { x = ( x 1 , x 2 , x 3 ) ∈ S 2 | x 3 &gt; √ 3 / 2 } , and let

<!-- formula-not-decoded -->

In Figure 4, on the left we plot the results for m = 2000 and various values of p , and on the right we fix p = 0 . 04 and vary m .

Figure 4: The classification error of ˆ h β for varying values of β , with data on S 2 ⊂ R 3 given by Eq. (5). On the left, m = 2000 is fixed, p varies. On the right, p = 0 . 04 is fixed, m varies. Best viewed in color.

<!-- image -->

As seen in Figure 4, the same asymmetric phenomenon holds in which overly large β are more forgiving than overly small β , especially in low noise regimes. The main difference between the first

and second experiment is that the optimal 'benign' exponent in the second case is β = 2 , matching the intrinsic dimension of the sphere, even though the data is embedded in 3 -dimensional space. This agrees with our conjecture that for distributions with low intrinsic dimension d int &lt; d , the overfitting behavior depends on d int rather than d (as discussed in Remark 5.2).

In Appendix E we provide an extension of the spherical data experiment, in which the inputs are corrupted by Gaussian noise. As the noise variance increases, hence the dataset is drawn away from having a low intrinsic dimension, the β value with minimal test error gradually increases from 2 to 3 . This illustrates a robustness to input-noise which is prevalent in practice, complementing an aspect that our current formal results do not cover.

## 6.2 Intrinsic Dimension of MNIST

Next, we consider an experiment in which the data consists of images of handwritten 0 and 1 digits from the MNIST dataset. In Figure 5, on the left we plot the results with respect to the entire training set m = 12 , 665 and various values of p , and on the right we fix p = 0 . 1 and vary m .

Figure 5: The classification error of ˆ h β for varying values of β , with respect to MNIST's 0 / 1 data. On the left, m is fixed to the entire train set, p varies. On the right, p = 0 . 1 is fixed, m varies. Best viewed in color.

<!-- image -->

As seen in Figure 5, the same asymmetric phenomenon demonstrated by both our theory as well as the synthetic experiments, clearly holds once more. It is interesting to note that each image in MNIST is of size 28 × 28 = 784 pixels, and so the extrinsic dimension is 784 . Nonetheless, the optimal exponent is roughly β ≈ 8 ≪ 784 , which matches an estimate of the intrinsic dimension of MNIST measured by Pope et al. [45]. Moreover, as seen on the right, the asymptotic phenomenon manifests quite clearly already for small samples sizes, and only becomes more pronounced as the number of samples increases.

## 7 Discussion

In this work, we characterized the generalization behavior of the NW interpolator for any choice of the hyperparameter β . Specifically, NW interpolates in a tempered manner when β &gt; d , exhibits benign overfitting when β = d , and overfits catastrophically when β &lt; d . This substantially extends the classical analysis of this method, which only focused on consistency. In addition, it indicates that the NW interpolator is much more tolerant to over-estimating β as opposed to under-estimating it.

Our analysis and experiments both suggest that the dependence on d arises from the assumption that the distributions considered here have a density in R d , and that more generally over-estimating the intrinsic dimension of the data is preferable to under-estimating it when setting β .

Overall, our results highlight how intricate generalization behaviors, including the full range from benign through tempered to catastrophic overfitting, can already appear in simple and well-known interpolating learning rules. We hope these results will further motivate revisiting other fundamental learning rules using this modern viewpoint, going beyond the classical consistency-vs.-inconsistency dichotomy.

## Acknowledgments and Disclosure of Funding

This research is supported in part by European Research Council (ERC) grant 754705, by the Israeli Council for Higher Education (CHE) via the Weizmann Data Science Research Center and by research grants from the Estate of Harry Schutzman and the Anita James Rosen Foundation. GK is supported by an Azrieli Foundation graduate fellowship.

## References

- [1] Mikhail Belkin, Daniel Hsu, Siyuan Ma, and Soumik Mandal. Reconciling modern machinelearning practice and the classical bias-variance trade-off. Proceedings of the National Academy of Sciences , 116(32):15849-15854, 2019.
- [2] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning (still) requires rethinking generalization. Communications of the ACM , 64(3): 107-115, 2021.
- [3] Peter L Bartlett, Philip M Long, Gábor Lugosi, and Alexander Tsigler. Benign overfitting in linear regression. Proceedings of the National Academy of Sciences , 117(48):30063-30070, 2020.
- [4] Tengyuan Liang and Alexander Rakhlin. Just interpolate: Kernel 'ridgeless' regression can generalize. The Annals of Statistics , 48(3):1329-1347, 2020.
- [5] Spencer Frei, Niladri S Chatterji, and Peter Bartlett. Benign overfitting without linearity: Neural network classifiers trained by gradient descent for noisy linear data. In Conference on Learning Theory , pages 2668-2703. PMLR, 2022.
- [6] Alexander Tsigler and Peter L Bartlett. Benign overfitting in ridge regression. J. Mach. Learn. Res. , 24:123-1, 2023.
- [7] Neil Mallinar, James Simon, Amirhesam Abedsoltan, Parthe Pandit, Misha Belkin, and Preetum Nakkiran. Benign, tempered, or catastrophic: Toward a refined taxonomy of overfitting. Advances in Neural Information Processing Systems , 35:1182-1195, 2022.
- [8] Naren Sarayu Manoj and Nathan Srebro. Interpolation learning with minimum description length. arXiv preprint arXiv:2302.07263 , 2023.
- [9] Guy Kornowski, Gilad Yehudai, and Ohad Shamir. From tempered to benign overfitting in relu neural networks. Advances in Neural Information Processing Systems , 36, 2024.
- [10] Nirmit Joshi, Gal Vardi, and Nathan Srebro. Noisy interpolation learning with shallow univariate relu networks. In The Twelfth International Conference on Learning Representations , 2024.
- [11] Yicheng Li and Qian Lin. On the asymptotic learning curves of kernel ridge regression under power-law decay. Advances in Neural Information Processing Systems , 36, 2024.
- [12] Daniel Barzilai and Ohad Shamir. Generalization in kernel regression under realistic assumptions. In Forty-first International Conference on Machine Learning , 2024.
- [13] Marko Medvedev, Gal Vardi, and Nathan Srebro. Overfitting behaviour of gaussian kernel ridgeless regression: Varying bandwidth or dimensionality. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [14] Tin Sum Cheng, Aurelien Lucchi, Anastasis Kratsios, and David Belius. Characterizing overfitting in kernel ridgeless regression through the eigenspectrum. In Forty-first International Conference on Machine Learning , 2024.
- [15] Thomas Cover and Peter Hart. Nearest neighbor pattern classification. IEEE transactions on information theory , 13(1):21-27, 1967.
- [16] Luc Devroye, László Györfi, and Gábor Lugosi. A probabilistic theory of pattern recognition , volume 31. Springer Science &amp; Business Media, 2013.

- [17] Elizbar A Nadaraya. On estimating regression. Theory of Probability &amp; Its Applications , 9(1): 141-142, 1964.
- [18] Geoffrey S Watson. Smooth regression analysis. Sankhy¯ a: The Indian Journal of Statistics, Series A , pages 359-372, 1964.
- [19] Luc Devroye, Laszlo Györfi, and Adam Krzy˙ zak. The Hilbert kernel regression estimate. Journal of Multivariate Analysis , 65(2):209-227, 1998.
- [20] Donald Shepard. A two-dimensional interpolation function for irregularly-spaced data. In Proceedings of the 1968 23rd ACM national conference , pages 517-524, 1968.
- [21] Mikhail Belkin, Alexander Rakhlin, and Alexandre B Tsybakov. Does data interpolation contradict statistical optimality? In The 22nd International Conference on Artificial Intelligence and Statistics , pages 1611-1619. PMLR, 2019.
- [22] Adityanarayanan Radhakrishnan, Mikhail Belkin, and Caroline Uhler. Wide and deep neural networks achieve consistency for classification. Proceedings of the National Academy of Sciences , 120(14):e2208779120, 2023.
- [23] Luke Eilers, Raoul-Martin Memmesheimer, and Sven Goedeke. A generalized neural tangent kernel for surrogate gradient learning. arXiv preprint arXiv:2405.15539 , 2024.
- [24] Amirhesam Abedsoltan, Adityanarayanan Radhakrishnan, Jingfeng Wu, and Mikhail Belkin. Context-scaling versus task-scaling in in-context learning. arXiv preprint arXiv:2410.12783 , 2024.
- [25] Mikhail Belkin, Daniel Hsu, and Ji Xu. Two models of double descent for weak features. SIAM Journal on Mathematics of Data Science , 2(4):1167-1180, 2020.
- [26] Jeffrey Negrea, Gintare Karolina Dziugaite, and Daniel Roy. In defense of uniform convergence: Generalization via derandomization with an application to interpolating predictors. In International Conference on Machine Learning , pages 7263-7272. PMLR, 2020.
- [27] Frederic Koehler, Lijia Zhou, Danica J Sutherland, and Nathan Srebro. Uniform convergence of interpolators: Gaussian width, norm bounds and benign overfitting. Advances in Neural Information Processing Systems , 34:20657-20668, 2021.
- [28] Trevor Hastie, Andrea Montanari, Saharon Rosset, and Ryan J Tibshirani. Surprises in highdimensional ridgeless least squares interpolation. The Annals of Statistics , 50(2):949-986, 2022.
- [29] Lijia Zhou, Frederic Koehler, Danica J Sutherland, and Nathan Srebro. Optimistic rates: A unifying theory for interpolation learning and regularization in linear regression. ACM/IMS Journal of Data Science , 1, 2023.
- [30] Ohad Shamir. The implicit bias of benign overfitting. Journal of Machine Learning Research , 24(113):1-40, 2023.
- [31] Zitong Yang, Yu Bai, and Song Mei. Exact gap between generalization error and uniform convergence in random feature models. In International Conference on Machine Learning , pages 11704-11715. PMLR, 2021.
- [32] Song Mei and Andrea Montanari. The generalization error of random features regression: Precise asymptotics and the double descent curve. Communications on Pure and Applied Mathematics , 75(4):667-766, 2022.
- [33] Mikhail Belkin, Daniel J Hsu, and Partha Mitra. Overfitting or perfect fitting? risk bounds for classification and regression rules that interpolate. Advances in neural information processing systems , 31, 2018.
- [34] Mikhail Belkin, Siyuan Ma, and Soumik Mandal. To understand deep learning we need to understand kernel learning. In International Conference on Machine Learning , pages 541-549. PMLR, 2018.

- [35] Vaishnavh Nagarajan and J Zico Kolter. Uniform convergence may be unable to explain generalization in deep learning. Advances in Neural Information Processing Systems , 32, 2019.
- [36] Alexander Rakhlin and Xiyu Zhai. Consistency of interpolation with laplace kernels is a high-dimensional phenomenon. In Conference on Learning Theory , pages 2595-2623. PMLR, 2019.
- [37] Daniel Beaglehole, Mikhail Belkin, and Parthe Pandit. On the inconsistency of kernel ridgeless regression in fixed dimensions. SIAM Journal on Mathematics of Data Science , 5(4):854-872, 2023.
- [38] Moritz Haas, David Holzmüller, Ulrike Luxburg, and Ingo Steinwart. Mind the spikes: Benign overfitting of kernels and neural networks in fixed dimension. Advances in Neural Information Processing Systems , 36, 2024.
- [39] Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Generalization error of random feature and kernel methods: hypercontractivity and kernel matrix concentration. Applied and Computational Harmonic Analysis , 59:3-84, 2022.
- [40] Lechao Xiao, Hong Hu, Theodor Misiakiewicz, Yue Lu, and Jeffrey Pennington. Precise learning curves and higher-order scalings for dot-product kernel regression. Advances in Neural Information Processing Systems , 35:4558-4570, 2022.
- [41] Haobo Zhang, Weihao Lu, and Qian Lin. The phase diagram of kernel interpolation in large dimensions. Biometrika , page asae057, 2024.
- [42] Tin Sum Cheng, Aurelien Lucchi, Anastasis Kratsios, and David Belius. A comprehensive analysis on the learning curve in kernel ridge regression. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [43] Itamar Harel, William M Hoza, Gal Vardi, Itay Evron, Nathan Srebro, and Daniel Soudry. Provable tempered overfitting of minimal nets and typical nets. Advances in Neural Information Processing Systems , 37, 2024.
- [44] Simon Buchholz. Kernel interpolation in sobolev spaces is not consistent in low dimensions. In Conference on Learning Theory , pages 3410-3440. PMLR, 2022.
- [45] Phil Pope, Chen Zhu, Ahmed Abdelkader, Micah Goldblum, and Tom Goldstein. The intrinsic dimension of images and its impact on learning. In International Conference on Learning Representations , 2021.
- [46] Luc Devroye. Nonuniform random variate generation. Handbooks in operations research and management science , 13:83-121, 2006.
- [47] Galen R Shorack and Jon A Wellner. Empirical processes with applications to statistics . SIAM, 2009.
- [48] Elias M Stein and Rami Shakarchi. Real analysis: measure theory, integration, and Hilbert spaces . Princeton University Press, 2009.
- [49] Roman Vershynin. High-dimensional probability: An introduction with applications in data science , volume 47. Cambridge university press, 2018.
- [50] Roman Vershynin. Introduction to the non-asymptotic analysis of random matrices. arXiv preprint arXiv:1011.3027 , 2010.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction describe the concrete contributions of this paper, as indicated throughout the rest of the paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: As discussed in Remark 5.2 and elsewhere in the paper, our analysis does not formally cover the case in which the distribution is supported over a lower-dimensional curved manifold. Other than that, any theoretical paper is limited by the setting it considers, and we do not believe our paper is further missing pointers to any specific limitations.

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

Justification: All assumptions, theorems, lemmas and proof sketches are provided in the main text. The full proofs are provided in the appendices.

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

Justification: We clearly discuss all of the above in Section 6.

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

Justification: The experiments described in Section 6 are very easy to reproduce based on our accurate description. We believe there is no added value by sharing this code.

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

Justification: We provide a complete description of the details in Section 6. We remark that some of the variables described above are not directly relevant to the experiments in this paper (e.g., no optimizer etc.).

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The figures in Section 6 include confidence intervals, as described therein.

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

Answer: [No]

Justification: We do not explicitly mention these, as the experiments are extremely lightweight and take at most a couple of minutes to run locally on a standard computer.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We reviewed the NeurIPS Code of Ethics, and the research conforms with it.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: This work is theoretical, and its goal is to advance our understanding of a certain phenomenon observed the field of Machine Learning. While there are potential societal consequences of Machine Learning as a whole, we believe none of which must be specifically highlighted here.

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

Justification: [NA]

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: [NA]

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

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Notation and order statistics

We start by introducing some notation that we will use throughout the proofs to follow. We denote X 1 ≍ X 2 to abbreviate X 1 = Θ( X 2 ) , X 1 ≲ X 2 to abbreviate X 1 = O ( X 2 ) and X 1 ≳ X 2 to abbreviate X 1 = Ω( X 2 ) . Throughout the proofs we let α := β/d , and abbreviate h = ˆ h β . Given some x ∈ supp( µ ) and m ∈ N , we consider the one-dimensional random variables

<!-- formula-not-decoded -->

where V d is the volume of the d dimensional unit ball, and the randomness is over x i . We let F x be the CDF of W x i (which is clearly the same for all i ∈ [ m ] ). We also let U i ∼ U ([0 , 1]) , i ∈ [ m ] be standard uniform random variables, and denote by W x ( i ) and U ( i ) the ordered versions of the W x i s and U i s respectively, namely

<!-- formula-not-decoded -->

We will often omit the superscript/subscript x where it is clear by context and use the notations W i , W ( i ) and F to denote W x i , W x ( i ) and F x respectively. Lastly, we let F -1 : [0 , 1] → [0 , 1] , F -1 ( t ) = inf { s : F ( s ) ≥ t } be the quantile function, and note that it satisfies F ( w ) ≤ u if and only if w ≤ F -1 ( u ) .

<!-- formula-not-decoded -->

Note that since ( W i ) i ∈ [ m ] are independent, the lemma above further applies to the joint distribution, and to the joint distribution of the order statistics (see e.g. 46, Example 2.3). Since we will use this often, we state this as a separate lemma.

<!-- formula-not-decoded -->

The behavior of U ( i ) is best understood through the following lemma.

Lemma A.3 (47, Chapter 8, Proposition 1) . Let E 1 , . . . E m +1 be i.i.d. standard exponential random variables. Then

<!-- formula-not-decoded -->

## B Proof of Theorem 4.1

Throughout the proof, we will use the notation introduced in Appendix A.

Lemma B.1. For almost every x ∈ R d and ϵ &gt; 0 , there exists δ x &gt; 0 such that for any u ≤ V α d δ β x :

<!-- formula-not-decoded -->

Proof. By the Lebesgue differentiation theorem (cf. 48, Chapter 3), for almost every x ∈ R d and ϵ &gt; 0 , there exists some δ x &gt; 0 such that

<!-- formula-not-decoded -->

In particular, for any 0 &lt; u ≤ V α d δ β x , taking r = u 1 /β V 1 /d d (which in particular satisfies r ≤ δ x ) we have

<!-- formula-not-decoded -->

As a result,

<!-- formula-not-decoded -->

The result readily follows by plugging ϵ = 1 2 and inverting.

Lemma B.2. For any k ∈ N , α &gt; 1 and almost every x ∈ R d there exists a constant ˜ C ( x , k, α ) such that as long as m ≥ ˜ C ( x , k, α ) , the following holds: If the k nearest neighbors of x are all labeled the same y (1) = · · · = y ( k ) , then h ( x ) = y (1) with probability at least 1 -c 1 exp( -c 2 k ) -exp( -c α k 1 -1 α ) over the randomness of ( x i ) i m =1 .

Proof. Given x , let δ = δ x &gt; 0 be the radius given by Lemma B.1, and assume without loss of generality that δ is sufficiently small so that f ∗ is constant over B ( x , δ ) (otherwise replace it by the smaller radius given by Assumption 3.1). Note that for all indices i such that W ( i ) ≤ δ , it holds that y i are independent variables (that equal f ∗ ( x ) with probability 1 -p ). Furthermore, given k ∈ N , we assume m is sufficiently large so that the k nearest neighbors of x all lie in B ( x , δ ) with probability at least 1 -exp( -k ) . Under this likely event, we decompose

<!-- formula-not-decoded -->

We will show that whenever y (1) = · · · = y ( k ) , then with high probability ( I ) is the dominant term in the sum above. We start by noting that if y (1) = · · · = y ( k ) , then ( I ) = y (1) ∑ k i =1 1 F -1 ( U ( i ) ) , thus

<!-- formula-not-decoded -->

Similarly,

<!-- formula-not-decoded -->

So the probability of | ( II ) | &lt; 1 2 | ( I ) | is at least the probability of the event in which 8 ∑ m i = k +1 1 ( ∑ i j =1 E j ) α &lt; 1 E α 1 . To see this event is indeed likely, we apply Lemma D.1 to get that with probability at least 1 -c 1 exp( -c 2 k ) , for all i ≥ k + 1 : 1 ( ∑ i j =1 E j ) α ≤ 1 ( i/ 2) α , and therefore under this event we get

<!-- formula-not-decoded -->

The latter is smaller than 1 /E α 1 as long as E 1 ≤ ( α -1) 1 /α k 1 -1 /α 2 · 8 1 /α , which by definition, occurs with probability 1 -exp [ -( α -1) 1 /α 2 · 8 1 /α k 1 -1 /α ] . To complete the proof, we note that | ( III ) | ≤ m δ is asymptotically negligible for sufficiently large m , since using Eq. (6) we see that | I | ≥ µ ( x ) 2 E α 1 ( ∑ m +1 i =1 E i ) α ≥ µ ( x ) 2 E α 1 ( m/ 2) α = ω ( m ) with probability at least 1 -c 1 exp( -c 2 m ) by Lemma D.1.

Lemma B.3. Given x ∈ R d , let A k x be the event in which all of x 's k nearest neighbors ( x ( i ) ) k i =1 satisfy f ∗ ( x i ) = f ∗ ( x ) . Then for any fixed k , it holds for almost every x ∈ supp( µ ) that lim m →∞ Pr[ A k x ] = 1 .

Proof. Let x ∈ supp( µ ) be such that µ is continuous at x (which holds for a full measure set by assumption). Since µ ( x ) &gt; 0 , then there exists ρ &gt; 0 so that µ | B ( x ,ρ ) &gt; 0 , and assume ρ is sufficiently small so that f ∗ | B ( x ,ρ ) = f ∗ ( x ) . Note that B ( x , ρ ) has some positive probability mass which we denote by ϕ := ∫ B ( x ,ρ ) µ . Under this notation, we see that

<!-- formula-not-decoded -->

Proof of Theorem 4.1 We start by proving the upper bound. Let k := log α α -1 (1 /p ) , and for any x ∈ R d , consider the event A k x in which x 's k nearest neighbors ( x ( i ) ) k i =1 satisfy f ∗ ( x i ) = f ∗ ( x ) (as described in Lemma B.3). Using the law of total expectation, we have that

̸

̸

<!-- formula-not-decoded -->

̸

✶ Note that by Lemma B.3 lim m →∞ Pr S [ ¬ A k x ] = 0 , and therefore it remains to bound the first summand above.

̸

To that end, we continue by temporarily fixing x . Denote by B k x the event in which x 's k nearest neighbors are all labeled correctly (namely, their labels were not flipped), and note that Pr S [ B k x ] = (1 -p ) k ≥ 1 -kp , hence Pr S [ ¬ B k x ] &lt; kp . By Lemma B.2 we also know that for sufficiently large m :

̸

<!-- formula-not-decoded -->

̸

̸

Therefore,

<!-- formula-not-decoded -->

̸

where the last inequality follows by our assignment of k . Since this is true for any x , it is also true in expectation over x , thus completing the proof of the upper bound.

We proceed to prove the lower bound. We consider A k x to be the same event as before, yet now we set k := k α = ( 8 · 2 α α -1 ) 1 α -1 . By lower bounding Eq. (7) (instead of upper bounding it as before), we obtain

̸

̸

As lim m →∞ Pr S [ A k x ] = 1 and lim m →∞ Pr S [ ¬ A k x ] = 0 by Lemma B.3, it once again remains to bound ( ⋆ ) .

<!-- formula-not-decoded -->

To that end, we temporarily fix x , denote by D k x the event in which the labels of x 's k nearest neighbors were are all flipped. Note that since the label flips are independent of the location of the datapoints, it holds that Pr S [ D k x | A k x ] = Pr S [ D k x ] = p k . By Lemma B.2 we also know that for sufficiently large m :

̸

<!-- formula-not-decoded -->

̸

̸

̸

Therefore,

<!-- formula-not-decoded -->

̸

is due to our assignment of k (and the explicit form of c α in Lemma B.2).

## C Proof of Theorem 5.1

Setting for the proof. Throughout the proof, we will use the notation introduced in Appendix A. We start by specifying the target function and distribution for which we will prove that catastrophic overfitting occurs. We will consider a slightly more general version than mentioned in the main text. Fix R,r, c &gt; 0 that satisfy R &gt; 3 r . We define a distribution on B ( 0 , R ) whose density is given by

<!-- formula-not-decoded -->

where V d is the volume of the d -dimensional unit ball. We also define the target function

<!-- formula-not-decoded -->

The main lemma from we derive the proof of Theorem 5.1 is the following:

Lemma C.1. Under setting C suppose that c satisfies

<!-- formula-not-decoded -->

Then there exists some m 0 ∈ N , such that for any x ∈ B ( 0 , r ) , m&gt;m 0 and p ∈ (0 , 0 . 49) , it holds with probability at least 1 -˜ O m ( 1 m + 1 m 1 -β/d β/d ) over the randomness of the training set S that

<!-- formula-not-decoded -->

We temporarily defer the proof of Lemma C.1, and start by showing that it easily implies the theorem:

Proof. of Theorem 5.1 Fix R &gt; 3 r , let c = 1 -β/d 2400(1+ R r ) β and consider the distribution and target function given by Setting C. Using the law of total expectation, we have that

̸

̸

<!-- formula-not-decoded -->

̸

where ( ∗ ) follows from Lemma C.1. This completes the proof by sending m →∞ .

̸

̸

̸

̸

̸

## C.1 Proof of Lemma C.1

Fix some x with ∥ x ∥ &lt; r , we will show that for sufficiently large m , with high probability x will be misclassified as +1 . To that end, we decompose

<!-- formula-not-decoded -->

where T 1 crudely bounds the contribution of points in the inner circle, T 2 is the expected contribution of outer points labeled 1 , and T 3 is a perturbation term. Let k m := |{ i ∈ [ m ] | ∥ x i ∥ ≤ r }| denote the number of training points inside the inner ball. By Lemma C.3, whenever c ≤ 1 2 (we will ensure this happens) it holds with probability at least 1 -2 exp ( -m 8 ) that

<!-- formula-not-decoded -->

Throughout the rest of the proof, we assume this event indeed occurs.

Bounding T 1 : Using that ∥ x ∥ &lt; r and that the pdf µ is such that for all x i / ∈ B ( 0 , r ) , ∥ x -x i ∥ &gt; 3 r -r &gt; 2 r , we have that the k m nearest neighbors x (1) , . . . , x ( k m ) are precisely the points with ∥ x i ∥ ≤ r .

For any w ≤ (2 r ) β and any z ∈ B ( x , w 1 β ) it holds that ∥ z ∥ ≤ 3 r , and µ ( x ) ≤ c V d r d . Thus, for such a w ,

<!-- formula-not-decoded -->

Correspondingly, by substituting u = c r d w 1 α , we obtain for any u ≤ 2 d c that u ≥ F ( u α r αd c α ) and thus F -1 ( u ) ≥ u α r αd c α . Note that for any i ∈ [ k m ] , ∥ ∥ x -x ( i ) ∥ ∥ β &lt; (2 r ) αd so W ( i ) satisfies the condition that W ( i ) ≤ (2 r ) αd . As such, using Lemma A.2 we obtain

<!-- formula-not-decoded -->

Now for T 1 , we have from Eq. (10):

<!-- formula-not-decoded -->

where (1) holds by Lemma C.4 with probability at least 1 -˜ O k m ( 1 k m + 1 k 1 -α m α ) = O m ( 1 m + 1 m 1 -α α ) and (2) follows from Eq. (9).

Bounding T 2 : Using the fact that for any i ∈ [ m ] , ∥ x -x i ∥ ≤ ∥ x ∥ + ∥ x i ∥ ≤ R + r , and the bound on k m from Eq. (9), we have for any p &lt; 0 . 49 that

<!-- formula-not-decoded -->

Bounding T 3 : From Lemma C.2 and Eq. (9), it holds with probability at least 1 -2 exp ( -√ m 4 ) that

<!-- formula-not-decoded -->

Putting it Together: For any ϵ &gt; 0 there is some m 0 ∈ N , such that for any m&gt;m 0 , -T 3 ≥ -mϵ . So overall, we obtain that with probability at least 1 -˜ O m ( 1 m + 1 1 -α ) , m α

<!-- formula-not-decoded -->

where the last line follows by using that α &lt; 1 , and by fixing some sufficiently small ϵ . Finally, fixing some c ≤ (1 -α ) r αd 2400( R + r ) αd = 1 -α 2400 ( 1+ R r ) αd suffices to ensure that this is positive, implying ˆ h β ( x ) = 1 .

Lemma C.2. Under Setting C, let x ∈ B ( 0 , r ) and k m := |{ i ∈ [ m ] | ∥ x i ∥ ≤ r }| . It holds with probability at least 1 -2 exp ( - √ m -k m ) that

<!-- formula-not-decoded -->

Proof. Let ξ i be the random variable representing a label flip, meaning that ξ i is 1 with probability p and -1 with probability 1 -p , and y i = f ∗ ( x i ) ξ i by assumption. For any x i with ∥ x i ∥ ≥ 3 r , it holds that f ∗ ( x i ) = 1 , and that ∥ x -x i ∥ ≥ ∥ x i ∥ - ∥ x ∥ ≥ 2 r , and thus y i ∥ x -x i ∥ αd are bounded as

<!-- formula-not-decoded -->

We thus apply Hoeffding's Inequality (cf. 49, Theorem 2.2.6) yielding that for any t ≥ 0

<!-- formula-not-decoded -->

In particular, we have that with probability at least 1 -2 exp ( -1 2 √ m -k m ) that

<!-- formula-not-decoded -->

Lemma C.3. Under setting C, let k m := |{ i : ∥ x i ∥ ≤ r }| , then it holds with probability at least 1 -2 exp( -c 2 m 2 ) that

<!-- formula-not-decoded -->

Proof. We can rewrite k m = ∑ m i =1 B i where B i = 1 if ∥ x i ∥ ≤ r and 0 otherwise. Notice that each B i is a Bernoulli random variable with parameter c , i.e B i is 1 with probability c and 0 with probability 1 -c . So by Hoeffding's inequality (cf. 49, Theorem 2.2.6), we have for any t ≥ 0 that

<!-- formula-not-decoded -->

Taking t = cm 2 concludes the proof.

Lemma C.4. It holds for any k ≤ m ∈ N , 0 &lt; α &lt; 1 that with probability at least 1 -˜ O k ( 1 k + 1 k 1 -α α ) ,

<!-- formula-not-decoded -->

Proof. Fix some n 0 ≤ k which will be specified later. Using Lemma A.3, we can write

<!-- formula-not-decoded -->

By Lemma D.1, for some absolute constant C &gt; 0 it holds with probability 1 -2(1+ 1 C ) exp( -Cn 0 ) that for all n ≥ n 0 ,

<!-- formula-not-decoded -->

Conditioned on this even occurring, we use this to bound both T 1 and T 3 . For T 1 , Eq. (12) directly implies that T 1 ≤ ( 3 2 m ) α . For T 3 , using both Eq. (12) as well as the integral test for convergence we obtain

<!-- formula-not-decoded -->

It remains to bound T 2 . By definition of an exponential random variable, for any t ≥ 0 it holds for any E i with probability at least exp( -t ) (which is ≥ 1 -t ) that E i ≥ t . So taking t = ( n 0 k 1 -α ) 1 α , it holds with probability at least 1 -( n 0 k 1 -α ) 1 α that E 1 ≥ ( n 0 k 1 -α ) 1 α . As a result,

<!-- formula-not-decoded -->

To ensure that the probability that both Eq. (12) and Eq. (13) hold is sufficiently high, we take n 0 = max ( 1 C log( k ) , 2 ) . As such, we obtain that with probability at least 1 -˜ O ( 1 k + 1 k 1 -α α ) that Eq. (11) can be bounded as

<!-- formula-not-decoded -->

## D Auxiliary lemma

Lemma D.1. Suppose ( E i ) i ∈ N iid ∼ exp(1) are standard exponential random variables. Then there exists some absolute constant C &gt; 0 such that:

1. For any n ∈ N it holds that

<!-- formula-not-decoded -->

2. For any n 0 ∈ N it holds that

<!-- formula-not-decoded -->

Proof. Denote by ∥ · ∥ ψ 1 the sub-exponential norm of a random vector (for a reminder of the definition, see for example Vershynin 49, Definition 2.7.5). Each E i satisfies for any t &gt; 0 , Pr( E i ≥ t ) ≤ exp( -t ) implying that ∥ E i ∥ ψ 1 = 1 . By Vershynin [50, Remark 5.18], this implies ∥ E i -1 ∥ ψ 1 ≤ 2 . So Bernstein's inequality for sub exponential random variables [49, Corollary 2.8.3] states that there exists some absolute constant C ′ &gt; 0 such that for any t ≥ 0

<!-- formula-not-decoded -->

Taking t = 1 2 and taking C := C ′ 16 yields

<!-- formula-not-decoded -->

This proves the first statement. For the second statement, we union bound and apply the integral test for convergence, to get that

<!-- formula-not-decoded -->

## E Additional experiment

In this appendix, we provide an extension of the spherical data experiment discussed in the main paper, demonstrating the effect of noisy sampling on our results.

We repeated the experiment with data sampled from the sphere the sphere S 2 ⊂ R 3 , given by Eq. (5). This time, after sampling from the sphere, we added Gaussian noise distributed as N ( 0 , σ 2 I 3 ) to each data point independently, and examined the effect of the noise variance σ 2 on the exponent β achieving minimal test error (corresponding to the intrinsic dimension in our theory).

The results are presented in Figure 6, with m = 2000 , p = 0 . 04 and various values of σ 2 .

Figure 6: The classification error of ˆ h β for varying values of β and sampling noise σ 2 . Best viewed in color.

<!-- image -->

As the noise increases, we see that this 'best' β gradually increases from 2 to 3, as the data indeed becomes fully dimensional for significant noise in the data-points. For example, with variance 0 . 04 in each coordinate (equivalently, standard deviation 0 . 2 ), the optimal β is roughly 2 . 5 , which is notably still smaller than 3 .