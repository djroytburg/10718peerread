## Anytime-valid, Bayes-assisted, Prediction-Powered Inference

Valentin Kilian ∗ Department of Statistics, University of Oxford kilian@stats.ox.ac.uk

## Stefano Cortinovis ∗

Department of Statistics, University of Oxford cortinovis@stats.ox.ac.uk

## Abstract

Given a large pool of unlabelled data and a smaller amount of labels, predictionpowered inference (PPI) leverages machine learning predictions to increase the statistical efficiency of confidence interval procedures based solely on labelled data, while preserving fixed-time validity. In this paper, we extend the PPI framework to the sequential setting, where labelled and unlabelled datasets grow over time. Exploiting Ville's inequality and the method of mixtures, we propose predictionpowered confidence sequence procedures that are asymptotically valid uniformly over time and naturally accommodate prior knowledge on the quality of the predictions to further boost efficiency. We carefully illustrate the design choices behind our method and demonstrate its effectiveness in real and synthetic examples.

## 1 Introduction

Increasing the sample size of an experiment is arguably the single simplest way to improve the precision of the statistical conclusions drawn from it. However, in many fields - such as healthcare, finance, and social sciences - obtaining labelled data is often costly and time-consuming. In these settings, using machine learning (ML) models to impute additional labels represents a tempting alternative to expensive data collection, albeit at the risk of introducing bias. Prediction-powered inference (PPI) [1] is a recently introduced framework for valid statistical inference in the presence of a small labelled dataset and a large number of unlabelled examples paired with predictions from a black-box model.

Formally, given an input/output pair ( X,Y ) ∼ P = P X × P Y | X , consider the goal of estimating

<!-- formula-not-decoded -->

where ℓ θ ( x, y ) is a convex loss function parameterised by θ ∈ R . As an example, the mean θ ⋆ = E [ Y ] is the estimand induced by the squared loss ℓ θ ( x, y ) = ( θ -y ) 2 / 2 . For t = 1 , 2 , . . . , we observe a sequence of independent random variables Z t , either drawn from P (labelled sample) or from P X (unlabelled sample), and we are provided with a black-box prediction rule f that maps any input x to a prediction f ( x ) .

Let ( X i , Y i ) i ≥ 1 and ( ˜ X j ) j ≥ 1 denote the subsequences of labelled and unlabelled samples, respectively. For n = 1 , 2 , . . . , let N n denote the number of unlabelled samples observed before the n th labelled one, and assume that N n ≥ n , with N n ≫ n in typical settings. PPI constructs an (asymptotic) 1 -α confidence interval (CI) C pp α,n for θ ⋆ , that exploits the auxiliary information encoded in f . To this end, under mild assumptions, θ ⋆ can be expressed as the solution to

<!-- formula-not-decoded -->

∗ Equal contribution. Order decided by coin toss.

François Caron Department of Statistics, University of Oxford caron@stats.ox.ac.uk

where ℓ ′ θ is a subgradient of ℓ θ with respect to θ . The quantity g θ in Equation (2) can be decomposed as g θ = m θ +∆ θ , where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where m θ represents a measure of fit of the predictor, while ∆ θ , the rectifier , accounts for the discrepancy between the predicted outputs f ( X ) and the true labels Y . If C g α,θ,n is a (1 -α ) confidence interval for g θ , then the PPI confidence interval C pp α,n , defined as also achieves the desired coverage, i.e., Pr( θ ⋆ ∈ C pp α,n ) ≥ 1 -α . Constructing C g α,θ,n relies on estimating g θ , for which PPI defines an estimator leveraging both the unlabelled data and the prediction rule f . The resulting method outperforms standard CI procedures based on the labelled data alone when f is sufficiently accurate and N n ≫ n . Intuitively, this is because, in this case, ∆ θ is close to zero, while m θ can be estimated with low variance from the unlabelled data.

Crucially, coverage of the PPI CI (4) is guaranteed only at a fixed time, i.e., for a labelled sample size n fixed in advance. This is undesirable in many practical settings - such as online learning, real-time monitoring, or sequential decision-making - where it is essential to continuously draw conclusions as new data arrive. In this work, we address this by proposing an anytime-valid extension of the PPI CI (4). That is, we define a confidence sequence ( C avpp α,n ) n ≥ 1 , satisfying asymptotically the stronger coverage guarantee while still taking advantage of the prediction rule f . Analogously to standard PPI, we construct a confidence sequence ( C g α,θ,n ) n ≥ 1 for g θ and define C avpp α,n through Equation (4) for n ≥ 1 . While our approach is agnostic to the specific form of the confidence sequence ( C g α,θ,n ) n ≥ 1 , we mainly focus on asymptotic confidence sequences [2], as they provide a versatile time-uniform analogue of standard CLT-based CIs that applies to the PPI framework above in full generality. Moreover, being based on the method of mixtures [3, 4, 5], they can readily accommodate prior information on the quality of the prediction model f . In particular, by means of a zero-centred prior on the rectifier ∆ θ , we obtain tighter confidence sequences when the predictions are good, extending the fixed-time Bayes-assisted approach of Cortinovis and Caron [6].

<!-- formula-not-decoded -->

The remainder of the paper is organised as follows. Section 2 reviews related work. Section 3 provides background on (asymptotic) confidence sequences and discusses how prior information may be incorporated into their construction. Section 4 presents PPI in the context of control-variate estimators, whose asymptotic properties are crucial for our approach to anytime-valid, Bayes-assisted PPI, which is described in Section 5. Section 6 demonstrates the benefits of our method on synthetic and real data. Finally, Section 7 discusses limitations of our approach and further extensions. Proofs and additional experiments are provided in the Supplementary Material.

## 2 Related Work

PPI was introduced by Angelopoulos et al. [1] as a general framework for valid statistical inference with black-box ML predictors, and later extended in Angelopoulos et al. [7]. Closely related ideas appear in the literature on semi-supervised inference, missing-data methods, survey sampling, and double machine learning [8, 9, 10, 11, 12]. More recently, Cortinovis and Caron [6] proposed a Bayes-assisted variant of PPI. All of these contributions target fixed-time confidence intervals.

Confidence sequences (CS) were introduced by Darling and Robbins [13] and further developed by Robbins and Siegmund [4] and Lai [5], building on earlier work by Ville [3] and Wald [14]. Interest has surged again in recent years [15, 2], motivated by applications such as A/B testing. The notion is closely linked to e-values [16, 17]. Building on the e-value framework and on earlier work by Zrnic and Candès [18] and Waudby-Smith and Ramdas [15], Csillag et al. [19] proposed an exact, time-uniform PPI method that yields CS under stronger conditions (e.g., existence of bounded e-values) but does not leverage prior knowledge about the quality of the ML predictions. Furthermore, applying their method requires an active-learning setup in which, at each time t , the observation Z t can be labelled with strictly positive probability. In particular, it is not applicable to deterministic sequences of observations, such as those describing a large initial pool of unlabelled data followed by a stream of labelled data, which are the main focus of our experiments.

In the setting of double machine learning and semiparametric inference, Dalal et al. [20] and WaudbySmith et al. [2] derive asymptotic confidence sequences for target parameters in the presence of high-dimensional nuisance components.

## 3 Asymptotic (Bayes-assisted) confidence sequences

In this section, we first review background on (asymptotic) confidence sequences (CS), and then show how prior information can be incorporated into asymptotic CS procedures, leading to asymptotic Bayes-assisted confidence sequences.

## 3.1 Background

We start by defining an exact confidence sequence [13], a time-uniform analogue of classical CIs. Definition 1 (Confidence sequence) . Let ( C α,t ) t ≥ 1 be a sequence of random subsets of R . For α ∈ (0 , 1) , ( C α,t ) t ≥ 1 is a 1 -α confidence sequence for a fixed parameter µ ∈ R if

<!-- formula-not-decoded -->

We now introduce the notion of an asymptotic confidence sequence (AsympCS) [2, 20].

Definition 2 (Asymptotic confidence sequence) . Let α ∈ (0 , 1) and ( a t ) t ≥ 1 be a real sequence such that lim t →∞ a t = 0 . Let ( ̂ µ t ) t ≥ 1 be a consistent sequence of estimators of µ . The sequence of random intervals ( C α,t ) t ≥ 1 , with C α,t = [ ̂ µ t -L t , ̂ µ t + U t ] and L t &gt; 0 , U t &gt; 0 , is said to be an asymptotic confidence sequence with (little-o) approximation rate a t if there exists a (usually unknown) confidence sequence ( C ⋆ α,t ) t ≥ 1 , with C ⋆ α,t = [ ̂ µ t -L ⋆ t , ̂ µ t + U ⋆ t ] , such that and, almost surely as t →∞ , max { L ⋆ t -L t , U ⋆ t -U t } = o ( a t ) .

<!-- formula-not-decoded -->

Thus, an asymptotic CS may be regarded as an approximation of an exact CS that becomes arbitrarily accurate in the limit. It is worth noting that, while classical fixed-sample asymptotic CIs rely on convergence in distribution of scaled estimators, asymptotic confidence sequences rely on almost sure convergence at a given rate of the centred lower and upper bounds relative to those of an underlying exact CS. The following is an example of an asymptotic CS that applies to i.i.d. data.

Theorem 1. Let ( Y t ) t ≥ 1 be a sequence of i.i.d. random variables with mean µ and such that E | Y 1 | 2+ δ &lt; ∞ for some δ &gt; 0 . For any t ≥ 1 , let Y t be the sample mean, and ̂ σ 2 t be the sample variance based on the first t observations. For any parameter ρ &gt; 0 , the sequence of intervals defined as

<!-- formula-not-decoded -->

forms a (1 -α ) -AsympCS with approximation rate 1 / √ t log t for µ .

For the sequel, it is useful to highlight some aspects of the proof of this theorem. First, if the random variables ( Y t ) t ≥ 1 were Gaussian with variance σ 2 , then C NA α,t ( Y t , σ ; ρ ) would form an exact CS. This follows from combining the method of mixtures for nonnegative martingales with Ville's inequality [3, 4, 5, 21]. Second, the proof relies on KMT strong coupling [22, 23]: there exists i.i.d. Gaussian random variables ( W t ) t ≥ 1 with mean µ and variance var( Y ) such that

<!-- formula-not-decoded -->

Such a coupling plays a central role in constructing asymptotic confidence sequences, serving as a substitute for the CLT assumption underlying in classical fixed-sample CIs. The construction in Theorem 1 extends beyond the i.i.d. case, provided a similar coupling exists.

Theorem 2. Let ( ̂ µ t ) t ≥ 1 be a consistent sequence of estimators of µ . Assume that there exists a sequence of i.i.d. Gaussian random variables ( W i ) i ≥ 1 , with mean µ and variance σ 2 , such that

<!-- formula-not-decoded -->

Let ( ̂ σ 2 t ) t ≥ 1 be a consistent sequence of estimators of σ 2 with | ̂ σ t -σ | = o ( 1 log t ) a.s. Then, for any parameter ρ &gt; 0 , the sequence of intervals ( C NA α,t ( ̂ µ t , ̂ σ t ; ρ )) t ≥ 1 forms a (1 -α ) -AsympCS with approximation rate 1 / √ t log t for µ .

The asymptotic CS (6) includes a tuning parameter ρ , which can be chosen so as to minimise the width of the interval at a specified time t ; see [2, Appendix B.2]. However, this method does not allow the incorporation of prior information about the parameter of interest to yield tighter intervals when the data align with such assumptions: the width of Equation (6) is independent of Y t .

## 3.2 Asymptotic Bayes-assisted confidence sequences

To address this, we introduce a Bayes-assisted analogue of Theorem 1.

Theorem 3 (Bayes-assisted AsympCS - i.i.d. case) . Let ( Y t ) t ≥ 1 be a sequence of i.i.d. random variables with unknown mean µ and unknown variance σ 2 , and such that E | Y 1 | 2+ δ &lt; ∞ for some δ &gt; 0 . For any t ≥ 1 , let Y t be the sample mean, and ̂ σ 2 t be the sample variance based on the first t observations. Let η t : R → (0 , √ t/ (2 π )) be defined as

<!-- formula-not-decoded -->

where π is a continuous and proper prior density on R , strictly positive in a neighbourhood of µ/σ . Then

<!-- formula-not-decoded -->

forms a (1 -α ) -AsympCS with approximation rate 1 / √ t log t for µ .

In Theorem 3, the density π encodes prior beliefs about the ratio µ/σ . Under this prior, η t represents the marginal density of the standardised mean Y t /σ that would arise if the observations ( Y t ) t ≥ 1 were normally distributed. In contrast to the non-assisted AsympCS (6), the width of the Bayes-assisted AsympCS (9) varies with Y t / ̂ σ t : when the data align with the prior, η t ( Y t / ̂ σ t ) is large and the interval narrows; when they conflict, η t ( Y t / ̂ σ t ) is small and the interval widens. It is worth emphasising that, even when the prior is strongly misspecified, the Bayes-assisted AsympCS (9) remains valid. In the case of a Gaussian prior π centred at µ 0 with variance τ 2 , we obtain the following AsympCS :

<!-- formula-not-decoded -->

Setting ρ = τ allows a direct comparison between (10) and its non-assisted counterpart (6). When the data agree with the prior - i.e., Y t / ̂ σ t -µ 0 ≃ 0 - the Bayes-assisted interval is narrower than the non-assisted one. Conversely, if the data conflict with the prior, ( Y t / ̂ σ t -µ 0 ) 2 is large and the Bayes-assisted AsympCS becomes wider than (6). The proof of Theorem 3 is similar to that of [2, Theorem 2.2]. First, note that C BA α,t ( Y t , var( Y ); π ) would be an exact CS if the observations were normally distributed. This follows from an application of the method of mixtures for nonnegative martingales, using the prior π as mixing density, together with Ville's inequality. Second, we use KMT strong coupling to approximate in an almost sure sense Y t by a sample average of i.i.d. Gaussian random variables. As in the non-assisted case, Theorem 3 can be extended to the non-i.i.d. setting, as long as one can find such a strong coupling.

Theorem 4 (Asymptotic Bayes-assisted CS - non-i.i.d. case) . Consider the same notation and assumptions as in Theorem 2. Let π be a continuous and proper prior density on R , strictly positive in a neighbourhood of µ/σ , and let η t be the density (8) for any t ≥ 1 . Then, the sequence of intervals ( C BA α,t ( ̂ µ t , ̂ σ t ; π )) t ≥ 1 forms a (1 -α ) -AsympCS with approximation rate 1 / √ t log t for µ .

## 3.3 Asymptotic Type-I error control

The asymptotic confidence sequences defined above satisfy an asymptotic version of time-uniform Type-I error control (in the sense of [2, §2.5]; see also [24]).

Theorem 5 (Asymptotic Type-I error control) . Assume the hypotheses of one of Theorems 1 to 4, and let ( C α,t ) be the corresponding (1 -α ) -AsympCS for µ . Then

<!-- formula-not-decoded -->

## 4 Control variates and PPI: background and strong coupling

Prediction-powered inference (PPI) closely relates to control variates, a standard variance-reduction method in Monte Carlo estimation [25, §4.1]. In fact, each PPI estimator can be expressed as a control-variate estimator. We begin with a review of control variates and derive a KMT-type strong-coupling result for these estimators, before providing additional background on PPI.

## 4.1 Control variates: definitions and KMT strong coupling

Let ( U, V ) be real-valued random variables with finite variance, and consider the goal of estimating γ = E [ V ] from an i.i.d. sample ( U i , V i ) n i =1 . If µ = E [ U ] is known, the control-variate estimator (CVE) of γ is defined as

<!-- formula-not-decoded -->

where U and V denote the empirical means of ( U i ) n i =1 and ( V i ) n i =1 , respectively, λ ∈ R is a tunable coefficient, and the term U i -µ acts as a control variate. The estimator ̂ γ icv λ is unbiased, consistent, and has variance var( ̂ γ icv λ ) = (var( V ) -2 λ cov( U, V ) + λ 2 var( U )) /n . Compared to the standard sample mean estimator V , which attains variance var( V ) = var( V ) /n , using ̂ γ icv λ yields variance reduction when λ &lt; 2cov( U, V ) / var( U ) . The minimum variance is achieved at the optimal coefficient λ ⋆ = cov( U, V ) / var( U ) , for which var( ̂ γ icv λ ⋆ ) = (1 -ρ 2 U,V )var( V ) , where ρ U,V is the correlation between U and V . That is, stronger correlation leads to greater variance reduction.

In practice, µ and λ ⋆ are typically unknown. When this is the case, given an additional i.i.d. sample ( ˜ U j ) N n j =1 , independent of ( U i , V i ) n i =1 , where ˜ U 1 has the same distribution as U , one can estimate µ by ̂ µ = 1 N n ∑ N n j =1 ˜ U j and plug it into Equation (12). For fixed λ , this gives

<!-- formula-not-decoded -->

Similarly, λ ⋆ may be estimated from data as ̂ λ = ̂ cov(( U i , V i ) n i =1 ) / ̂ var(( U i ) n i =1 ) , where ̂ var( · ) and ̂ cov( · ) denote the sample variance and covariance, respectively. Plugging ̂ λ into (13) defines ̂ γ cv+ := ̂ γ cv ̂ λ , which is similar to the semi-supervised least squares estimator of Zhang et al. [11, Eq. (2.15)]. As discussed in Section 3, deriving an AsympCS requires a strong coupling between the estimator and a sequence of i.i.d. Gaussian random variables. We now establish this coupling, a key ingredient for constructing AsympCS for CVEs (and, in particular, for PPI estimators).

Proposition 1 (Asymptotics for CVEs) . Assume E | U | 2+ δ and E | V | 2+ δ &lt; ∞ for some 0 &lt; δ &lt; 1 . Then, almost surely as n →∞ ,

<!-- formula-not-decoded -->

Proposition 2 (KMT coupling for CVEs) . Assume E | U | 2+ δ and E | V | 2+ δ &lt; ∞ for some 0 &lt; δ &lt; 1 . Additionally, assume that | n N n -r | = O (1 /n 1 -a ) with 0 &lt; a &lt; 2 / (2 + δ ) , for some r ∈ [0 , 1] . Then, there exist i.i.d. Gaussian random variables ( W cv i ) i ≥ 1 with mean γ and variance

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

such that, almost surely as n →∞ ,

Likewise, there exist i.i.d. Gaussian random variables ( W cv+ i ) i ≥ 1 with mean γ and variance ν cv+ := ν cv λ ⋆ = var( V ) [ 1 -(1 -r ) ρ 2 U,V ] such that, almost surely as n →∞ ,

The estimators

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

are consistent estimators of ν cv λ and ν cv+ , respectively, where ̂ µ = 1 N n ∑ N n j =1 ˜ U j .

## 4.2 PPI estimators: definitions and asymptotic properties

Owing to Equation (2), the PPI estimator ̂ θ n is the value of θ that solves the equation ̂ g θ,n = 0 , where ̂ g θ,n = ̂ m θ,n + ̂ ∆ θ,n is an estimator of g θ . Here, ̂ m θ,n and ̂ ∆ θ,n are estimators of m θ and ∆ θ , respectively. A typical choice for ̂ m θ,n is the sample mean of the unlabelled data,

<!-- formula-not-decoded -->

Different choices for ̂ ∆ θ,n have been proposed in the literature, leading to different PPI estimators.

Standard PPI. Angelopoulos et al. [1] use the sample mean

<!-- formula-not-decoded -->

as an estimator for ∆ θ . Combining Equation (20) and Equation (19),

<!-- formula-not-decoded -->

is a CVE with control variate ℓ ′ θ ( X i , f ( X i )) -̂ m θ,n and control-variate parameter λ = 1 . For the squared loss, the estimator ̂ θ PP n solving ̂ g PP θ,n = 0 also takes the control-variate form

<!-- formula-not-decoded -->

with control variate f ( X i ) -1 N n ∑ N n j =1 f ( ˜ X j ) and λ = 1 .

PPI ++ . Angelopoulos et al. [7] extend the standard PPI estimator (21) by allowing the controlvariate parameter λ , which they call power-tuning parameter, to take values other than 1 . The resulting estimator is

<!-- formula-not-decoded -->

where ̂ λ θ,n is the estimator ̂ λ θ,n = ̂ cov (( ℓ ′ θ ( X i , Y i ) , ℓ ′ θ ( X i , f ( X i ))) n i =1 ) / ̂ var (( ℓ ′ θ ( X i , f ( X i ))) n i =1 ) . In this case, ̂ ∆ PP+ θ,n is a CVE with centred control variate ℓ ′ θ ( X i , f ( X i )) -̂ m θ,n , which depends only on the black-box predictions. As a result,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is also a CVE. Under the squared loss, we obtain

<!-- formula-not-decoded -->

where in this case ̂ λ θ,n = ̂ λ 0 ,n for all θ . Standard asymptotic confidence intervals for PPI and PPI ++ rely on CLTs for the estimators ̂ g θ,n , ̂ m θ,n , and ̂ ∆ θ,n . In contrast, constructing asymptotic confidence sequences requires almost sure approximations by averages of i.i.d. Gaussian variables. Since the estimators for g θ , m θ and ∆ θ are all CVEs, the asymptotic results of Proposition 1 and the KMT coupling of Proposition 2 both apply.

## 5 Anytime-valid, Bayes-assisted, prediction-powered inference

In this section we combine the results of Sections 3 and 4 within the PPI framework to obtain AsympCS for g θ . For any θ ∈ R and i ≥ 1 , let U θ,i = ℓ ′ θ ( X i , f ( X i )) , ˜ U θ,i = ℓ ′ θ ( ˜ X i , f ( ˜ X i )) and V θ,i = ℓ ′ θ ( X i , Y i ) . Define V θ,n = 1 n ∑ n i =1 V θ,i and U θ,n = 1 n ∑ n i =1 U θ,i . In the following, we assume E | U θ,i | 2+ δ , E | ˜ U θ,i | 2+ δ and E | V θ,i | 2+ δ &lt; ∞ for some 0 &lt; δ &lt; 1 , and that | n N n -r | = O (1 /n 1 -a ) with 0 &lt; a &lt; 2 / (2 + δ ) for some r ∈ [0 , 1] .

## 5.1 Anytime-valid PPI

We first derive AsympCS that do not incorporate prior information about the black-box predictor's accuracy. The following result follows directly from Proposition 2 and Theorem 2, owing to the control-variate form of the PPI estimator ̂ g PP θ,n (21) and of the PPI ++ estimator ̂ g PP+ θ,n (24).

Proposition 3. Let ̂ g θ,n be either the PPI (21) or the PPI ++ (24) estimator. For PPI, let ( ̂ σ g θ,n ) 2 = ̂ ν cv 1 (( U θ,i , V θ,i ) n i =1 , ( ˜ U θ,j ) N n j =1 ) (see (17) ). For PPI ++ , let ( ̂ σ g θ,n ) 2 = ̂ ν cv+ (( U θ,i , V θ,i ) n i =1 ) (see (18) ). Then, for any ρ &gt; 0 , the sequence of intervals defined as C g α,θ,n = C NA α,n ( ̂ g θ,n , ̂ σ g θ,n ; ρ ) forms a (1 -α ) -AsympCS with approximation rate 1 / √ n log n for g θ and asymptotic Type-I error control.

## 5.2 Anytime-valid, Bayes-assisted, PPI

In many modern applications extremely accurate black-box predictors are available (e.g., [26, 27, 28]). When this is the case, we can leverage this prior information to obtain tighter AsympCS for g θ via a zero-mean prior on ∆ θ . Following the decomposition in Equation (3), we combine an AsympCS for m θ (Proposition 4) with a Bayes-assisted AsympCS for ∆ θ (Proposition 5).

Proposition 4 (AsympCS for m θ ) . Let ̂ m θ,n and ( ̂ σ f θ,n ) 2 be the sample mean (19) and sample variance of ( ℓ ′ θ ( ˜ X j , f ( ˜ X j ))) N n j =1 . Let δ ∈ (0 , 1) . For any ρ &gt; 0 , R δ,θ,n = C NA δ,n ( ̂ m θ,n , ̂ σ f θ,n ; ρ ) forms a (1 -δ ) -AsympCS with approximation rate 1 / √ n log n for m θ and asymptotic Type-I error control.

Proposition 5 (Bayes-assisted AsympCS for ∆ θ ) . For PPI, let ̂ ∆ θ,n and ( ̂ σ ∆ θ,n ) 2 be the sample mean (20) and sample variance of ( V θ,i -U θ,i ) n i =1 . For PPI ++ , let ̂ ∆ θ,n be the control-variate estimator (23) and ( ̂ σ ∆ θ,n ) 2 = ̂ ν cv+ (( U θ,i , V θ,i -U θ,i ) n i =1 ) (see (18) ). Let κ ∈ (0 , 1) . For any continuous proper prior π , the sequence of Bayes-assisted intervals T κ,θ,n = C BA κ,n ( ̂ ∆ θ,n , ̂ σ ∆ θ,n ; π ) forms a (1 -κ ) -AsympCS with approximation rate 1 / √ n log n for ∆ θ and asymptotic Type-I error control.

Finally, for both PPI and PPI ++ , the confidence sequences R δ,θ,n and T α -δ,θ,n are combined via a Minkowski sum to obtain a (1 -α ) -AsympCS for g θ , with approximation rate 1 / √ n log n and asymptotic Type-I error control, of the form

<!-- formula-not-decoded -->

where ̂ g θ,n is either the PPI estimator (21) or the PPI ++ estimator (24). Solving Equation (4) gives the confidence region for θ ⋆ . In the case of the squared loss, C avpp α,n is an interval, given by

<!-- formula-not-decoded -->

where ̂ θ n is either the PPI estimator (22) or the PPI ++ estimator (25).

## 6 Experiments

We compare the PPI and PPI ++ AsympCS procedures introduced in Section 5 - with and without Bayes assistance - to the AsympCS relying solely on labelled data (obtained from Theorem 1 and referred to as 'classical') on several estimation problems. Bayes-assisted methods are annotated with (G), (L), or (T) to indicate Gaussian, Laplace, or Student-t priors with mean zero and scale depending on the task and reported in the Supplementary Material. For the Student-t prior, we set the degrees of freedom to 2 in all experiments. Since PPI is motivated by settings with scarce labelled data and abundant unlabelled data, we consider the following experimental setting: labelled data arrive sequentially, i.e., n = 1 , 2 , . . . , while a large unlabelled dataset is available from the start, i.e., N n = N for all n , with N ≫ n large enough to exclude any uncertainty on the measure of fit m θ . As discussed by Cortinovis and Caron [6], this simplifies the comparison between non-assisted and Bayesassisted PPI, as it rules out any potential loss of efficiency due to the Minkowski sum (26), thereby isolating the effect of the Bayes correction on the CS procedure. For synthetic data, we set N = ∞ to guarantee the simplification holds. For real data, we empirically verify that N is large enough to justify this assumption by confirming that anytime validity is preserved - specifically, that the cumulative miscoverage rate remains below the chosen threshold α = 0 . 1 for all n . As with CLT-based CIs, the n at which one starts counting the cumulative miscoverage rate of an asymptotic CS is inherently arbitrary; unless otherwise stated, we choose n = 40 , as we empirically find this to be a reasonably small labelled sample size at which the KMT coupling generally provides a good approximation.

## 6.1 Synthetic data

The synthetic experiments follow a general structure: we start with N = ∞ unlabelled samples { ˜ X j } N j =1 iid ∼ P X and successively sample n labelled observations ( X i , Y i ) n i =1 iid ∼ P with the goal of estimating the mean θ ⋆ = E [ Y ] .

Noisy predictions. This experiment demonstrates that our method can adapt to varying correlation levels between predictions and true labels by using the PPI ++ estimator (23). We sample Y i iid ∼ N (0 , 1) , so that θ ⋆ = E [ Y ] = 0 . The prediction rule is defined as f ( X i ) = Y i + ϵ i , where X i is only used for indexing and ϵ i iid ∼ N (0 , σ 2 Y ) , with the noise level σ Y ∈ { 0 . 1 , 0 . 8 , 3 } . In this case, the optimal control-variate parameter is given by λ ⋆ θ = λ ⋆ = cov( Y, f ( X )) / var( f ( X )) = (1 + σ 2 Y ) -1 , which decreases with σ Y . Figure 1 compares the interval volume achieved by classical and nonassisted CS procedures as a function of n , while results under informative priors are reported in Section S7.1. For small noise levels, PPI and PPI ++ achieve similar performance, and greatly outperform classical inference. As the noise level grows, the machine learning predictions become less informative and standard PPI loses ground to the classical CS. By contrast, PPI ++ adapts to the noise level and always performs similarly to, or better than, the other baselines.

Biased predictions. This experiment illustrates the potential benefits of incorporating prior information into our method. We sample X i iid ∼ N (0 , 1) and Y i = X i + ϵ i , where ϵ i iid ∼ t df (0 , 1) , so that θ ⋆ = E [ Y ] = 0 . The prediction rule is defined as f ( X i ) = X i + υ , where υ ∈ R controls its bias level. For all υ , λ ⋆ = 1 , so PPI and PPI ++ coincide. We vary υ between -1 . 2 and 1 . 2 , and df ∈ { 5 , 10 , ∞} to study the impact of bias level and noise distribution on the AsympCS procedures. Figure 2 compares the average interval volumes at n = 100 as a function of υ for each value of df . Classical inference and non-assisted PPI volumes remain essentially constant across bias levels, reflecting their lack of prior information, and with the latter consistently outperforming the former

Figure 1: Noisy predictions study. The left, middle and right panels show average interval volume over 1000 repetitions as a function of the labelled sample size n for noise levels σ Y ∈ { 0 . 1 , 0 . 8 , 3 . 0 } .

<!-- image -->

Figure 2: Biased predictions study. The left, middle and right panels show average interval volume over 100 repetitions as a function of the bias level υ for df = 5 , 10 , ∞ .

<!-- image -->

by leveraging imputed predictions. On the other hand, the volume of the Bayes-assisted procedures varies widely with the bias level υ : it is reduced for small υ , but grows with | υ | as the priors become increasingly misspecified. Notably, the volume under the Gaussian prior inflates the fastest with | υ | , while heavier-tailed Laplace and Student-t priors offer comparatively greater robustness. These conclusions hold for all values of df , which controls the accuracy of the KMT coupling approximation for a given n . Coverage results in Section S7.1 show that, while smaller values of df lead to slightly worse coverage, the approximation quality is overall satisfactory in this example.

## 6.2 Real data

We evaluate our method on several real-world datasets, which are described in Section S6.2. While each dataset is, in principle, static (providing label/prediction pairs ( Y i , f ( X i )) N + n 1 i =1 ), we simulate an online setting akin to Section 6.1 by randomly splitting the data into a labelled set of size n 1 , serving as a labelled data stream, and an unlabelled set of size N .

Figure 3 compares classical and PPI ++ AsympCS procedures on the FLIGHTS, FOREST, and GALAXIES datasets, where the goal is mean estimation. By taking advantage of the unlabelled data, PPI methods consistently yield smaller regions than the classical counterpart, while maintaining reliable coverage. Moreover, Bayes-assisted approaches further improve efficiency for moderate labelled sample sizes, as the quality of the predictions is generally high in these datasets.

Figure S8 reports results for three additional estimation tasks: linear regression (CENSUS), logistic regression (HEALTHCARE), and quantile estimation (GENES). For the first two tasks, the same conclusions as for mean estimation hold: PPI methods consistently outperform classical inference,

Figure 3: Mean estimation. The top and bottom rows show the average interval volume and cumulative miscoverage rate over 1000 repetitions for the FLIGHTS, FOREST, and GALAXIES datasets.

<!-- image -->

with Bayes-assisted approaches providing an additional efficiency boost. For the quantile estimation task, non-assisted PPI still improves over classical inference by leveraging the machine learning predictions; however, the Bayes-assisted methods yield larger regions than the other approaches, reflecting lower prediction quality in this dataset.

## 7 Discussion

We extended the PPI framework to the sequential setting via asymptotic confidence sequences, which allow for the seamless integration of prior information about the quality of the auxiliary predictions. However, several directions merit further investigation. The results developed here are for scalar parameter values θ . Extensions to multivariate settings are discussed in Section S4, building on earlier work by Waudby-Smith et al. [2, §B.10]. In the non-assisted case, we focused on asymptotic confidence sequences of the form (6), but other options are possible. In particular, as discussed in Section S8, the parameter-free CS proposed by Wang and Ramdas [29], which is based on an improper prior, may be used as an exact reference CS in place of Equation (6).

The AsympCS derived in this paper are asymptotically valid for i.i.d. data under mild, nonparametric assumptions. Promising directions include extensions to non-i.i.d. observations, as well as the development of nonasymptotic , nonparametric Bayes-assisted confidence sequences under stricter assumptions (e.g., bounded means), building on the work of Waudby-Smith and Ramdas [15]. In the non-assisted case, the parameter ρ was assumed to be fixed. Waudby-Smith et al. [2, §2.5] considered delayed-start sequences C α,t ( m ) that may depend on the start time m ; this includes allowing the tuning parameter ρ to depend on m . Their asymptotic Type-I error control result, derived under assumptions similar to those used here, also applies in our setting. Another interesting direction would be to adapt similar ideas to the Bayes-assisted construction.

PPI AsympCS procedures share the computational considerations of their fixed-time counterparts. Beyond mean estimation (e.g., Figure S8), they typically require constructing a grid over θ . When the marginal density η t is not available in closed form (e.g., for the Studentt prior), the Bayes-assisted version requires numerical integration. If computation is a concern, the Laplace prior offers a good compromise: it has heavier tails than the Gaussian while still admitting a closed-form expression for η t .

## Acknowledgments and Disclosure of Funding

Valentin Kilian is supported by the Clarendon Funds Scholarship. Stefano Cortinovis is supported by the EPSRC Centre for Doctoral Training in Modern Statistics and Statistical Machine Learning (EP/S023151/1). The authors thank the reviewers for their time and valuable feedback, especially the suggestion to incorporate a discussion on Type-I error control.

## References

- [1] A. N. Angelopoulos, S. Bates, C. Fannjiang, M. I. Jordan, and T. Zrnic. Prediction-powered inference. Science , 382(6671):669-674, 2023.
- [2] I. Waudby-Smith, D. Arbour, R. Sinha, E. Kennedy, and A. Ramdas. Time-uniform central limit theory and asymptotic confidence sequences. The Annals of Statistics , 52(6):2613-2640, 2024.
- [3] J. Ville. Etude critique de la notion de collectif . Gauthier-Villars Paris, 1939.
- [4] H. Robbins and D. Siegmund. Boundary crossing probabilities for the Wiener process and sample sums. The Annals of Mathematical Statistics , pages 1410-1429, 1970.
- [5] T. L. Lai. On confidence sequences. The Annals of Statistics , pages 265-280, 1976.
- [6] S. Cortinovis and F. Caron. FAB-PPI: Frequentist, assisted by Bayes, prediction-powered inference. In International Conference on Machine Learning (ICML'2025) , 2025.
- [7] A. Angelopoulos, J. Duchi, and T. Zrnic. PPI++: Efficient prediction-powered inference. arXiv preprint arXiv:2311.01453 , 2023.
- [8] J. M. Robins and A. Rotnitzky. Semiparametric efficiency in multivariate regression models with missing data. Journal of the American Statistical Association , 90(429):122-129, 1995.
- [9] C.-E. Särndal, B. Swensson, and J. Wretman. Model assisted survey sampling . Springer Science &amp;Business Media, 2003.
- [10] V. Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey, and J. Robins. Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal , 21(1):C1-C68, 2018.
- [11] A. Zhang, L. D. Brown, and T. T. Cai. Semi-supervised inference: general theory and estimation of means. The Annals of Statistics , 47(5):2538-2566, 2019.
- [12] Y. Zhang and J. Bradic. High-dimensional semi-supervised learning: in search of optimal inference of the mean. Biometrika , 109(2):387-403, 2022.
- [13] D. A. Darling and H. Robbins. Confidence sequences for mean, variance, and median. Proceedings of the National Academy of Sciences , 58(1):66-68, 1967.
- [14] A. Wald. Sequential tests of statistical hypotheses. The Annals of Mathematical Statistics , 16 (2):117-186, 1945.
- [15] I. Waudby-Smith and A. Ramdas. Estimating means of bounded random variables by betting. Journal of the Royal Statistical Society Series B: Statistical Methodology , 86(1):1-27, 2024.
- [16] A. Ramdas, P. Grünwald, V. Vovk, and G. Shafer. Game-theoretic statistics and safe anytimevalid inference. Statistical Science , 38(4):576-601, 2023.
- [17] A. Ramdas and R. Wang. Hypothesis testing with e-values. Foundations and Trends® in Statistics , 1(1-2):1-390, 2025. ISSN 2978-4212. doi: 10.1561/3600000002.
- [18] T. Zrnic and E. J. Candès. Active statistical inference. In Proceedings of the 41st International Conference on Machine Learning , ICML'24. JMLR.org, 2024.
- [19] D. Csillag, C. Jose Struchiner, and G. Tegoni Goedert. Prediction-powered e-values. In Proceedings of the 42nd International Conference on Machine Learning , 2025.
- [20] A. Dalal, P. Blöbaum, S. Kasiviswanathan, and A. Ramdas. Anytime-Valid Inference for Double/Debiased Machine Learning of Causal Parameters. arXiv:2408.09598 , 2024. doi: 10.48550/arXiv.2408.09598.
- [21] S. Howard, A. Ramdas, J. McAuliffe, and J. Sekhon. Time-uniform, nonparametric, nonasymptotic confidence sequences. The Annals of Statistics , 49(2), 2021.

- [22] J. Komlós, P. Major, and G. Tusnády. An approximation of partial sums of independent RV'-s, and the sample DF. I. Zeitschrift für Wahrscheinlichkeitstheorie und Verwandte Gebiete , 32(1): 111-131, March 1975. ISSN 1432-2064. doi: 10.1007/BF00533093.
- [23] P. Major. The approximation of partial sums of independent RV's. Zeitschrift für Wahrscheinlichkeitstheorie und Verwandte Gebiete , 35(3):213-220, September 1976. ISSN 1432-2064. doi: 10.1007/BF00532673.
- [24] A. Bibaut, N. Kallus, and M. Lindon. Near-optimal non-parametric sequential tests and confidence sequences with possibly dependent observations. arXiv preprint arXiv:2212.14411 , 2022.
- [25] P. Glasserman. Monte Carlo Methods in Financial Engineering . Springer, 2003.
- [26] A. Dal Pozzolo, O. Caelen, R. A Johnson, and G. Bontempi. Calibrating probability with undersampling for unbalanced classification. In 2015 IEEE symposium series on computational intelligence , pages 159-166. IEEE, 2015.
- [27] R. Lam, A. Sanchez-Gonzalez, M. Willson, P. Wirnsberger, M. Fortunato, F. Alet, S. Ravuri, T. Ewalds, Z. Eaton-Rosen, W. Hu, A. Merose, S. Hoyer, G. Holland, O. Vinyals, J. Stott, A. Pritzel, S. Mohamed, and P. Battaglia. Learning skillful medium-range global weather forecasting. Science , 382(6677):1416-1421, 2023. doi: 10.1126/science.adi2336.
- [28] J. M. Jumper, R. Evans, A. Pritzel, T. Green, M. Figurnov, O. Ronneberger, K. Tunyasuvunakool, R. Bates, A. Žídek, A. Potapenko, A. Bridgland, C. Meyer, S. A A Kohl, A. Ballard, A. Cowie, B. Romera-Paredes, S. Nikolov, R. Jain, J. Adler, T. Back, S. Petersen, D. Reiman, E. Clancy, M. Zielinski, M. Steinegger, M. Pacholska, T. Berghammer, S. Bodenstein, D. Silver, O. Vinyals, A. W. Senior, K. Kavukcuoglu, P. Kohli, and D. Hassabis. Highly accurate protein structure prediction with alphafold. Nature , 596:583 - 589, 2021.
- [29] H. Wang and A. Ramdas. The extended Ville's inequality for nonintegrable nonnegative supermartingales. arXiv preprint arXiv:2304.01163 , 2023.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction accurately reflect the paper's actual contributions.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification:The limitations are discussed in our Discussion section.

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

Justification: Proofs of all results are included in full detail in the supplementary material, except for some very classical results for which we provide only references. When relevant, we also provide a sketch of the proof in the main text.

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

Justification: As stated in the supplementary material, the code used to perform our experiments is made available online under a permissive licence.

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

Justification: As stated in the supplementary material, the code used to perform our experiments is made available online under a permissive licence.

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

Justification: All details necessary to understand the results are provided in the paper.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: For all experiments, results are averaged over many repetitions (100 or 1000 repetitions, depending on the experiment). Variations from the mean are negligible. Statistical guarantees (i.e., asymptotic time-uniform coverage) are checked empirically computing the average cumulative miscoverage rate for all experiments.

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

Justification: All experiments were run locally on an Apple Silicon M4 Pro CPU with 24GB of memory, and implementation details are provided in the supplementary material.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in the paper is mainly theoretical and uses only publicly available datasets, which do not contain any sensitive information.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The work performed is mainly theoretical, and we do not foresee any societal impact.

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

Justification: The work performed is mainly theoretical and doesn't pose such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All datasets have permissive licenses and are properly credited in the supplementary material.

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

Justification: As stated in the supplementary material, the code used to perform our experiments is made available online under a permissive licence.

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

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

## Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.