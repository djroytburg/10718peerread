## It's Hard to Be Normal: The Impact of Noise on Structure-agnostic Estimation

Jikai Jin ∗ Stanford University jkjin@stanford.edu

Lester Mackey Microsoft Research lmackey@microsoft.com

## Abstract

Structure-agnostic causal inference studies how well one can estimate a treatment effect given black-box machine learning estimates of nuisance functions (like the impact of confounders on treatment and outcomes). Here, we find that the answer depends in a surprising way on the distribution of the treatment noise. Focusing on the partially linear model of Robinson [1988], we first show that the widely adopted double machine learning (DML) estimator is minimax rate-optimal for Gaussian treatment noise, resolving an open problem of Mackey et al. [2018]. Meanwhile, for independent non-Gaussian treatment noise, we show that DML is always suboptimal by constructing new practical procedures with higher-order robustness to nuisance errors. These ACE procedures use structure-agnostic cumulant estimators to achieve r -th order insensitivity to nuisance errors whenever the ( r + 1) -st treatment cumulant is non-zero. We complement these core results with novel minimax guarantees for binary treatments in the partially linear model. Finally, using synthetic demand estimation experiments, we demonstrate the practical benefits of our higher-order robust estimators.

## 1 Introduction

Modern machine learning (ML) offers a rich toolbox of flexible methods for modeling complex, high-dimensional functions - ranging from regularized linear regression [Belloni et al., 2014, Zou and Hastie, 2005] and random forests [Breiman, 2001, Biau et al., 2008, Syrgkanis and Zampetakis, 2020] to neural networks [Schmidt-Hieber, 2020, Farrell et al., 2021] and hybrid combinations of these techniques [Dˇ zeroski and ˇ Zenko, 2004, Chernozhukov et al., 2022]. For statisticians and econometricians, it is natural to ask whether ML can improve both the accuracy and the robustness of estimating a target parameter of interest. Recently, Balakrishnan et al. [2023] introduced the paradigm of structure-agnostic estimation (SAE). SAE enables parameter inference by directly plugging in black-box ML-based estimates of the so-called nuisance functions - the other regression components that impact our observations while not being of primary inferential interest. This framework characterizes the best possible target estimation accuracy in terms of the error of these nuisance estimates. By contrast, the classical semiparametric approach derives optimal error rates under explicit structural assumptions on the nuisance - such as smoothness or sparsity - which must be exploited by the estimator and can render it fragile when those assumptions fail [Stone, 1982, Chen and White, 1999, Belloni et al., 2011].

In the field of causal inference, a typical goal is to estimate the impact of a treatment on an observed outcome in the presence of confounders X that impact both treatment T and outcome Y in largely unknown ways. When all confounders are observed, many causal estimands admit expressions in

∗ Jikai Jin was supported by NSF Award IIS-2337916.

† Vasilis Syrgkanis was supported by NSF Award IIS-2337916.

Vasilis Syrgkanis † Stanford University vsyrgk@stanford.edu

terms of regression functions that are themselves estimable from data. Consequently, causal parameter estimation falls squarely within the regime where ML-based nuisance estimates may provide benefits.

Recently, double/debiased machine learning (DML) [Chernozhukov et al., 2018, 2022] was proposed as an efficient way to estimate causal parameters from black-box nuisance estimates. Specifically, given n independent and identically distributed (i.i.d.) observations and nuisance estimates with mean squared error ϵ , these methods achieve O ( ϵ 2 + n -1 / 2 ) error rates, improving over the O ( ϵ + n -1 / 2 ) rate achieved by naively plugging in the nuisance estimates. In particular, the estimates are √ n -consistent even when the nuisance estimates converge as slowly as n -1 / 4 . Moreover, this rate is structure-agnostic, and recent works [Balakrishnan et al., 2023, Jin and Syrgkanis, 2024] proved that for several important causal parameters, this rate is in fact minimax rate-optimal in the SAE framework. In other words, one cannot achieve smaller estimation estimation rates without strong prior knowledge on the underlying structure of the nuisance. These optimality results provide a strong justification for the popularity of doubly robust learning methods in practice.

Acurious exception to this rule can be found in the work of Mackey et al. [2018]. There, the authors study the popular partially linear model of Robinson [1988]:

<!-- formula-not-decoded -->

Here, the observation Z = ( X,T,Y ) comprises covariates X ∈ X , a scalar treatment T ∈ R , and a scalar outcome Y ∈ R . The primary objective is the estimation of the parameter θ 0 , which represents the average causal effect of T on Y . Within this framework, the DML estimator for θ 0 is derived from the Neyman-orthogonal moment condition and typically takes the form:

<!-- formula-not-decoded -->

where ˆ g and ˆ q are machine learning estimators for the nuisance functions g 0 ( X ) = E [ T | X ] and q 0 ( X ) = E [ Y | X ] , respectively [Chernozhukov et al., 2018]. Common practice involves sample splitting, where one portion of the data is used to train ˆ g and ˆ q and another (independent) portion is used to compute ˆ θ DML via (2). However, Mackey et al. [2018] showed that for the model (1), one can design structure-agnostic estimators that are more efficient than DML if the treatment noise η is nonGaussian and independent of X . 3 Bonhomme et al. [2024] consider parameter estimation problems in a conditional likelihood setting and show that arbitrarily higher-order orthogonal estimators can be constructed if the conditional likelihood function is known up to the nuisance parameters. These works highlight a gap in our understanding of how noise impacts optimal SAE rates, and this is the main question that we will address in this paper.

Given access to black-box nuisance estimates, we make the following contributions:

- On the one hand, we prove in Section 3.2 the existence of Gaussian treatment barrier : for Gaussian treatment, no estimators can achieve a better rate than DML, even when the variance of the treatment is completely known. This implies that leveraging distributional information of the treatment noise as in Mackey et al. [2018] cannot yield a better estimate and that their restrictions to the non-Gaussian setting is not an algorithmic issue.
- On the other hand, for non-Gaussian treatment, we propose a general procedure to construct higher-order orthogonal moment functions and provide structure-agnostic guarantees in Section 4. Then, for treatment noise independent of X , we derive in Section 5 a new agnostic cumulant-based estimator (ACE) that achieves r -th order insensitivity to nuisance estimate error ϵ and O ( ϵ r + n -1 / 2 ) error rates for treatment effect estimation whenever the ( r + 1) -st cumulant of η is non-zero. To the best of our knowledge, this is the first structure-agnostic estimator that achieves arbitrarily high-order robustness.
- We complement these findings with additional contributions relevant to this setting. Specifically, we show in Section 3.1 that DML is minimax optimal for (1) with binary treatment, a case not covered in existing lower bounds that do not assume a partially linear outcome model.

3 The structure-agnostic rates are not explicitly stated in Mackey et al. [2018] but can be straightforwardly derived from their analysis.

We also derive new lower and upper bounds for structure-agnostic moment and cumulant estimation in a standard non-parametric model in Section 5.1 that might be of independent interest. Finally, in Section 6, we conduct a synthetic demand estimation experiment, highlighting the benefits of the ACE estimator compared with existing approaches.

Notation We introduce the shorthand [ m ] ≜ { 1 , . . . , m } for each m ∈ N . For a function g and distribution P on a domain Z , we let ∥ g ∥ P,s ≜ ∥ g ∥ L s ( P ) ≜ ( ∫ | g ( z ) | s dP ( z )) 1 /s with s ≥ 1 represent the L s ( P ) norm. For a vector ϵ ∈ R l , we define ∥ ϵ ∥ ∞ ≜ max 1 ≤ i ≤ l | ϵ i | . For two vectors α = ( α 1 , · · · , α l ) , β = ( β 1 , · · · , β l ) ∈ R l , we write α ≤ β if α i ≤ β i , 1 ≤ i ≤ l .

## 2 Structure-agnostic estimation and minimax error

To evaluate and compare method quality in this work, we adopt the minimax structure-agnostic framework of Balakrishnan et al. [2023]. Notably, structure-agnostic analyses make no explicit assumptions about nuisance smoothness, sparsity, or structure and instead simply assume access to black-box nuisance estimates with certain unobserved error levels.

We first define the class of all data generating distributions P on our data domain Z and assign to each a target parameter θ 0 ( P ) and a vector-valued nuisance function h : Z ↦→ R ℓ .

Definition 2.1 (Data generating distributions, target parameters, and nuisance functions) . Throughout, we let P be the set of candidate data generating distributions on the finite dimensional-domain Z , let H ⊆ ( R ℓ ) Z be the set of relevant vector-valued nuisance functions, and let Φ be a deterministic mapping from P to H . Then, for any P ∈ P , we say that θ 0 ( P ) ∈ R is the target parameter and h = Φ( P ) ∈ H is the nuisance function corresponding to the distribution P .

We will sometimes abuse notation and choose H to be a subset of H ⊆ ( R ℓ ) X where X is the domain of the covariate component X of Z . Departing from prior work [Balakrishnan et al., 2023, Jin and Syrgkanis, 2024], we leave the exact choice of nuisance mapping Φ unspecified in Definition 2.1. This allows us to study how the choice of nuisance functions affects the estimation error of the target parameter of interest.

Next, we introduce the ground-truth uncertainty sets associated with any nuisance estimate ˆ h and any target error level ϵ . For any vector-valued function h ⋆ : X ↦→ R ℓ , distribution P ∈ P , and ϵ ∈ R ℓ + , we define B P,s ( h ⋆ , ϵ ) := { h i ∈ L s ( P ) : ∥ h i -h ⋆ i ∥ P,s ≤ ϵ i , ∀ i ∈ [ ℓ ] } .

Definition 2.2 (Uncertainty sets) . For any nuisance estimate ˆ h ∈ H , error level vector ϵ ∈ R ℓ + , and power s ≥ 1 , we define the uncertainty set P s,ϵ ( ˆ h ; Φ) as the set of all P ∈ P 0 satisfying

<!-- formula-not-decoded -->

or, equivalently, Φ( P ) ∈ B P,s ( ˆ h, ϵ ) .

For convenience, we will omit the dependency of P s,ϵ ( ˆ h ; Φ) on P , which will always be clear from context. We will sometimes write P s,ϵ ( ˆ h ) when the choice of Φ is obvious. Finally, for a given estimator ˆ θ of a target parameter θ 0 ( P ) , we define the worst-case error over an uncertainty set.

Definition 2.3 (Minimax estimation error) . For any set of distributions P over Z , we define the worst-case (1 -γ ) -quantile error of an estimator ˆ θ : Z ⊗ n ↦→ R as R n, 1 -γ ( ˆ θ ; P ) ≜ sup P ∈P Q P, 1 -γ ( | ˆ θ -θ 0 ( P ) | ) , where Q P, 1 -γ ( | ˆ θ -θ 0 ( P ) | ) is a (1 -γ ) -quantile of | ˆ θ ( Z 1 , . . . , Z n ) -θ 0 ( P ) | when Z i i.i.d. ∼ P . We further define the minimax estimation error of P as M n, 1 -γ ( P ) ≜ inf ˆ θ : Z ⊗ n ↦→ R R 1 -γ ( ˆ θ ; P ) .

## 3 Structure-agnostic lower bounds

In this section, we establish structure-agnostic lower bounds for treatment effect estimation in the partially linear model (1).

## 3.1 Optimality of DML for binary treatment

We begin by establishing the minimax rate-optimality of DML when the treatment T is binary. Previous works [Balakrishnan et al., 2023, Jin and Syrgkanis, 2024] have established structureagnostic lower bounds of a similar form. However, Balakrishnan et al. [2023] consider the estimation of the expected conditional covariance defined as E [Cov( T, Y | X )] , while Jin and Syrgkanis [2024] consider the estimation of the average treatment effect with a different set of nuisance functions. Furthermore, neither work constrains the form of E [ Y | T, X ] , while we assume a partially linear structure for the outcome model. Our result implies that even with this additional assumption, it is still impossible to improve over DML. For convenience, we introduce the following definitions as the 'default' choice for the class of data generating distributions and nuisance functions:

Definition 3.1 (Set of feasible distributions) . We define P ⋆ as the set of all distributions of ( X,T,Y ) generated by (1) . Moreover, we define the following subsets of P ⋆ as follows:

- for any constants C θ , C T , C Y ∈ [0 , + ∞ ] , we use P r ( C θ , C T , C Y ) to denote all distributions P ∈ P ⋆ that satisfy | θ 0 | ≤ C θ , E [ | T | r ] 1 /r ≤ C T , and E [ | Y | r ] 1 /r ≤ C Y .
- for any constants C θ , C g , C q , ψ ξ , ψ η ∈ (0 , + ∞ ] , we use P ⋆ ( C θ , C g , C q ; ψ ξ , ψ η ) to denote the set of all distributions in P ∈ P ⋆ that satisfy
1. | θ 0 | ≤ C θ , | g 0 ( X ) | ≤ C g , | q 0 ( X ) | ≤ C q a.s. for ( g 0 , q 0 )( X ) = ( E [ T | X ] , E [ Y | X ])
2. ξ | X and η | X are ψ ξ and ψ η -sub-Gaussian a.s..
- for any constants C θ , C T , C Y ∈ [0 , + ∞ ] , we use P b ( C θ , C T , C Y ) to denote all distributions P ∈ P ⋆ that satisfy | θ 0 | ≤ C θ , | T | ≤ C T , and | Y | ≤ C Y .

Finally, we define Φ ⋆ as a mapping from P ∈ P ⋆ to the 'default' nuisance functions h 0 = ( g 0 , q 0 ) .

We are interested in the minimax structure-agnostic estimation error induced by P = P s,ϵ ( ˆ h, Φ) for some s ≥ 1 , in the sense of Definition 2.3, with P being chosen as a set of distributions that satisfies certain mild regularity conditions, such as the ones introduced in Definition 3.1. In other words, given black-box nuisance estimates of h 0 with L s ( P ) error rates, we would like to derive the optimal worst-case estimation error for the treatment effect θ 0 .

Our main result in this section is a lower bound for estimating θ 0 when T is a Bernoulli random variable. Our lower bound, Theorem 3.1, is established under the additional assumption that X has a uniform distribution on X = [0 , 1] K ; as a consequence, the minimax error rate of DML cannot be improved even when the marginal distribution of X is known.

Theorem 3.1 (Structure-agnostic lower bound for binary treatment) . Fix any C θ &gt; 0 , c q , δ ∈ (0 , 1 4 ) , and K ∈ N + , and let

<!-- formula-not-decoded -->

If ∥ ϵ ∥ ∞ ≤ δ/ 2 , then for any estimates ˆ h = (ˆ g, ˆ q ) with c q ≤ ˆ q ( X ) ≤ 1 -c q and ˆ g ( X )(1 -ˆ g ( X )) ≥ A -1 δ, a.s. , we have

<!-- formula-not-decoded -->

for any γ ∈ (1 / 2 , 1) , where c γ is a universal constant that only depends on γ .

When ∥ ϵ ∥ ∞ ≤ δ/ 2 , this matches the upper bound achieved by DML up to constant factors, stated in Theorem C.1 for completeness. The proof of Theorem 3.1 can be found in Appendix B.

## 3.2 The Gaussian treatment barrier

In the previous section, we established the rate-optimality of DML for binary treatment. Is it possible to improve over DML if we make different distributional assumptions? In the literature, it is not uncommon to model the treatment assignment rule using a Gaussian distribution, i.e. , η | X ∼ N (0 , σ ( X ) 2 ) for some function σ ( · ) [Imai and Van Dyk, 2004, Zhao et al., 2020]. However, in Mackey et al. [2018] it is shown that Gaussianality of the noise variable η is in fact a barrier for one to construct second-order orthogonal moments, thereby preventing them from deriving better error

rates than DML by leveraging distributional information of η . However, it is unclear whether this is an issue specific to their approach or if there indeed exists a fundamental, non-algorithmic barrier for Gaussian treatment.

In this section, we resolve this open question and show that the latter is true: if the treatment noise is Gaussian, then DML is already minimax rate-optimal even when η is independent of X and one has exact knowledge of its distribution. Our lower bound, stated next, is proved in Appendix D.

Theorem 3.2 (The Gaussian treatment barrier) . Let σ, C θ , C g , C q &gt; 0 be known constants and P = { P ∈ P b ( C θ , ∞ , C q ) : η | X ∼ N (0 , σ 2 ) and | g 0 ( X ) | ≤ C g a.s. } . If ϵ 1 ϵ 2 = o (log -1 / 2 n ) , then for any estimates ˆ h = (ˆ g, ˆ q ) satisfying | ˆ g | ≤ C g , | ˆ q | ≤ C q and any 1 ≤ s ≤ + ∞ , we have

<!-- formula-not-decoded -->

for any γ ∈ (1 / 2 , 1) , where c γ &gt; 0 is a constant that only depends on γ .

The assumption that | ˆ g | ≤ C g , | ˆ q | ≤ C q is natural, since for any P 0 ∈ P 0 , the ground-truth nuisance functions g 0 ( X ) = E [ T | X ] and q 0 ( X ) = E [ Y | X ] must satisfy | g 0 | ≤ C g and | q 0 | ≤ C q a.s. according to our assumption. Moreover, the lower bound does not depend on the value of s , meaning that no improvement is possible even if the nuisance estimates have small L ∞ -error.

Under the same assumptions, one can show that DML attains the minimax error rate up to the factor ( log(1 /ϵ 1 ) log n ) 3 and matches the minimax error rate whenever ϵ 1 = O ( n -c ) for some positive constant c . For completeness, we include the details in Theorem C.2. Notably, compared with the lower bound in Theorem 3.1, the term Θ( ϵ 2 1 ) disappears here since the variance σ 2 is assumed to be known. Theorem 3.2 establishes the existence of a method-agnostic Gaussian treatment barrier as suggested (but not proved) in Mackey et al. [2018].

The proof of Theorem 3.2 in Appendix D is based on a constrained risk inequality for testing composite hypothesis developed in Cai and Low [2011] combined with novel constructions of the fuzzy hypotheses using moment matching techniques. For Gaussian treatment, we show that such hypotheses can be constructed in a way such that their induced target parameters θ 0 are well-separated by leveraging a recursive property of Hermite polynomials.

## 4 Structure-agnostic upper bounds

Section 3 established the rate optimality of DML for binary and Gaussian treatments but left open the possibility of improvement for other treatment distributions. To exploit this opportunity, we will first introduce a new procedure that yields fast estimation rates whenever η | X is non-Gaussian and the cumulants of η | X are estimated accurately and next, in Section 5, show that cumulant estimation is easy when η is independent of X .

Our new procedure is based on the method of moments. Specifically, we will identify a moment function m satisfying E P [ m ( Z, θ, h ( X )) | X ] = 0 , a.s. for all P in some specified distribution set P , where θ = Φ( P ) is the ground-truth parameter of interest and h ( · ) is some vector-valued nuisance functions that we need to estimate from data. Moreover, we require that θ = Φ( P ) is the unique solution to the moment equation. We proceed by plug in estimates of the nuisance ˆ h derived from a sample D 0 , and select ˆ θ satisfying the empirical moment equation ∑ i ∈D m ( Z, θ, ˆ h ( X )) = 0 on a separate sample D . This procedure is widely adopted in the development of DML-type methods [Chernozhukov et al., 2018], and leads to efficient estimates as long as the moment function m is Neyman-orthogonal, meaning that it is insensitive, under expectation, to nuisance estimation errors. The precise definition will be presented in Theorem 4.1. The novelty of our construction lies in a specific recursive procedure that generates moment functions with arbitrarily high levels of insensitivity to nuisance estimation errors.

Consider the model (1) and let J 1 ( w,x ) be any function of w ∈ R and x ∈ X satisfying

<!-- formula-not-decoded -->

where each E [ a i 1 ( X ) 2 ] &lt; ∞ and each ρ i 1 is continuous. Without loss of generality, we assume that ρ 11 ( w ) ≡ 1 (as one can otherwise introduce a dummy summand into the expression (7) with

a 11 ( x ) 0 ). Let J r ( w,x ) ∞ be a series of functions defined by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The following lemma, proved in Appendix E.4, derives the general form of J r for each k ∈ Z + . Lemma 4.1 (Explicit formula for J ) . If M = M + r 1 , then r r 1 -

<!-- formula-not-decoded -->

In particular, for all r ≥ 2 we have ρ 2 r ( w ) = w .

We would ideally use the moment function

<!-- formula-not-decoded -->

to estimate θ 0 . However, by Lemma 4.1, J r depends on the unknown data generating distribution via the functions a ir ( · ) . Fortunately, our next theorem, proved in Appendix E.3, shows that an estimated moment function (11) based on an estimate of J r yields improved treatment effect estimation rates whenever θ 0 is identifiable and J r is estimated sufficiently well.

Theorem 4.1 (Structure-agnostic error from estimated moments) . Consider the datasets ( D 1 , D 2 ) = ( { Z i } n/ 2 i =1 , { Z i } n i = n/ 2+1 ) with each Z i ⊆ Z . Define the estimated moment function

<!-- formula-not-decoded -->

where h = ( g, q ) and ˆ J r ( w,x ; D 1 ) ≜ ∑ M r i =1 ˆ a ir ( x ; D 1 ) ρ ir ( w ) for ˆ a ir : X × Z n/ 2 ↦→ R and ρ ir ( · ) recursively defined via (9) . Fix any C θ , C g , C q ; ψ ξ , ψ η &gt; 0 , γ ∈ (0 , 1) , s ≥ r +1 , and ∆ ∈ R 2 + , and let P ⊆ P ⋆ ( C θ , C g , C q ; ψ ξ , ψ η ) contain all distributions P satisfying, with probability 1 -γ/ 2 over D 1 i.i.d. ∼ P , the following four conditions simultaneously:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˆ J ( j ) r ( w,x ) ≜ d j d w j ˆ J r ( w,x ) . Let ˆ h = (ˆ g, ˆ q ) be a possibly random function independent of D and ˆ θ be the solution of 1 n ∑ n i = n/ 2+1 ˆ m r ( Z i , θ, ˆ h ; D 1 ) = 0 . Then there exists a constant C γ &gt; 0 that only depends on γ , such that for all ϵ ≤ ∆ ,

<!-- formula-not-decoded -->

In particular, if ϵ ( j ) = O (max { ϵ 1 , ϵ 2 } r -j ) , then the treatment effect error rate (16) has r -th order dependencies on the nuisance errors ϵ i , i = 1 , 2 in place of the slower second-order dependencies of DML (see, e.g., Theorems C.1 and C.2). One caveat of applying this bound is that δ id , V m , and λ ⋆ all depend on the order of orthogonality r and need to be computed in a case-by-case manner. In Section 5.2, we will construct an explicit moment estimator that satisfies ϵ ( j ) = O (max { ϵ 1 , ϵ 2 } r -j ) and make the dependency on r explicit.

The identifiability assumption (12) is crucial and is why the construction of Theorem 4.1 does not work for Gaussian treatments. Indeed, as we show in Proposition I.2, if the treatment is Gaussian and (15) holds with ϵ ( j ) → 0 , then δ id → 0 , so that Theorem 4.1 cannot yield √ n -consistency.

A natural choice for J 1 ( w,x ) satisfying (7) is J 1 ( w,x ) ≡ w , which corresponds to selecting a 11 ≡ 0 , ρ 11 ≡ 1 , a 21 ≡ 1 , and ρ 21 ( w ) = w . In this case, Lemma 4.1 implies that each J r ( w,x ) is the following r -th order polynomial of w :

<!-- formula-not-decoded -->

Here, Π m denotes the set of all partitions of [ m ] 4 and κ i ( x ) is the i -th cumulant of η | X = x . In particular, if µ i ( x ) denotes the i -th moment of η | X = x , then J 2 ( w,x ) = 1 2 ( w 2 -µ 2 ( x )) and J 3 ( w,x ) = 1 6 ( w 3 -3 µ 2 ( x ) w -µ 3 ( x )) . In Section 5, we will show how to estimate these cumulant-based J r effectively whenever η is non-Gaussian and independent of X .

## 5 Agnostic cumulant-based estimation (ACE)

In this section, we apply the general guarantee in Theorem 4.1 to derive structure-agnostic estimators with better rates than DML when η is non-Gaussian and independent of X .

## 5.1 Structure-agnostic cumulant estimation

The moment function induced by the J k ( w,x ) defined in (17) requires estimating the cumulants κ i of the noise variable η in the treatment regression model T = g 0 ( X ) + η . In this subsection, we propose efficient structure-agnostic cumulant estimators assuming that η is independent of X . We will also see in Remark 5.2 that our approach has potential benefits even when this assumption fails.

Our main result, stated below, indicates that an r -th order error rate can be attained for estimating the r -th cumulant of η .

Theorem 5.1 (Efficient cumulant estimator for noise with finite moments) . Let C T &gt; 0 be a constant, r ≥ 2 be a positive integer, and P be the set of all distributions of ( X,T ) generated from T = g 0 ( X ) + η , such that E [ | T | r | X ] 1 /r ≤ C T holds a.s.. The target parameter θ 0 ( P ) : P ↦→ R is the r -th order cumulant of η = T -g 0 ( X ) under P . Let Φ map P ∈ P to the nuisance function g 0 . Let ˆ g : X ↦→ R be a nuisance estimate that satisfies | ˆ g ( X ) | ≤ C g and ˆ κ r : { ( X i , T i ) } n i =1 ↦→ R be the r -th order cumulant of the empirical residual distribution P n = 1 n ∑ n i =1 δ T i -ˆ g ( X i ) where δ z is the Dirac measure at z . Then for any γ ∈ (0 , 1) and s ≥ r , ˆ θ = ˆ κ r satisfies R n, 1 -γ ( ˆ θ ; P s,ϵ (ˆ g, Φ) ) ≤ C γ,r n -1 / 2 + (2 rϵ ) r where C γ,r = 10 r 1 / 2 γ -1 / 2 (2 C T ) r ( r -1)! .

Our next theorem derives a refined bound for sub-Gaussian treatment noise.

Theorem 5.2 (Efficient cumulant estimator for sub-Gaussian noise) . Let C g , ψ η &gt; 0 be a constant and P be the set of all distributions of ( X,T ) generated from T = g 0 ( X ) + η , such that | g 0 ( X ) | ≤ C g , a.s. and η is mean-zero, independent of X and ψ η -sub-Gaussian. The target parameter θ 0 ( P ) : P ↦→ R is the r -th order cumulant of η = T -g 0 ( X ) under P . Let Φ map P ∈ P to the nuisance function g . Let ˆ g : X ↦→ R be a nuisance estimate that satisfies | ˆ g ( X ) | ≤ C g and ˆ κ r : { ( X i , T i ) } n i =1 ↦→ R be the r -th order cumulant of the empirical residual distribution P n = 1 n ∑ n i =1 δ T i -ˆ g ( X i ) where δ z is the Dirac measure at z . Then for any γ ∈ (0 , 1) and s ≥ r , ˆ θ = ˆ κ r satisfies R n, 1 -γ ( ˆ θ ; P s,ϵ (ˆ g, Φ) ) ≤ C γ,r n -1 / 2 +(2 rϵ ) r where C γ,r = 3 [ 12 r ( C g + ψ η ) ] r l 1 / 2 γ -1 / 2 .

Remark 5.1 (Comparing the bounds in Theorem 5.1 and Theorem 5.2) . Under the assumptions in Theorem 5.2, C T in Theorem 5.1 would be O ( C g + √ rψ η ) , so that Theorem 5.1 implies a bound which scales as ( cr ) 3 r/ 2 n -1 / 2 +(2 rϵ ) r , while the bound in Theorem 5.2 scales as ( c ′ r ) r n -1 / 2 + (2 rϵ ) r , which is strictly tighter.

Remark 5.2 (Relaxing the independent noise assumption) . While Theorem 5.2 assumes that the noise variable η is independent of X , the estimator ˆ κ r can also be of value when this assumption does not hold. In Appendix F.3, we consider r = 3 and derive guarantees for this estimator when η is 'nearly' independent of X .

4 For example, Π 3 = { {{ 1 , 2 , 3 }} , {{ 1 , 2 } , { 3 }} , {{ 1 , 3 } , { 2 }} , {{ 2 , 3 } , { 1 }} , {{ 1 } , { 2 } , { 3 }} } .

̸

The fast rates in Theorems 5.1 and 5.2 hold specifically for estimating the cumulants. When the target estimand is instead a moment of η , our structure-agnostic lower bound in Theorem F.1 requires Ω( n -1 / 2 + ϵ 3 ) minimax error for r = 3 and Ω( n -1 / 2 + ϵ 2 ) minimax error for r = 3 . Notably, this cubic-quadratic bottleneck implies that, unlike the cumulant-based approach espoused here, the moment-based approach of Mackey et al. [2018] cannot attain arbitrarily high-order error rates.

The proofs of Theorems 5.1 and 5.2 can be found in Appendix F.2. To the best of our knowledge, these results are novel and may be of independent interest. Next, we will apply these result to construct more efficient structure-agnostic estimators of treatment effects.

## 5.2 Fast rates with independent treatment noise

In this subsection, we introduce agnostic cumulant-based estimation (ACE), a novel treatment effect estimator that leverages the efficient cumulant estimators of Section 5.1.

Throughout, we let ˆ κ i be the empirical cumulant estimate defined in Theorem 5.2 and define

<!-- formula-not-decoded -->

ˆ J r ( · ) can be viewed as an estimate of the cumulant-based function J r ( · ) (17) when X is independent of η . The key observation is that this ˆ J r ( · ) satisfies (15) with ϵ ( j ) = O ( ϵ j 1 ) , which follows from the key lemma stated below.

<!-- formula-not-decoded -->

Notably, each term in the RHS in Lemma 5.1 is a product of cumulant estimation errors. Recall in Theorem 5.2 we show that the estimation error of ˆ κ r is O ( ϵ r 1 ) when ∥ ˆ g -g ∥ P,s ≤ ϵ 1 , so that ∏ B ∈ π ( κ | B | -ˆ κ | B | ) = O (∏ B ∈ π ϵ | B | ) = O ( ϵ r -k 1 ) . We additionally bound the coefficient hidden in the O ( · ) in Lemma G.5. In view of this favorable property, we propose our estimation algorithm, ACE, in Algorithm 1.

## Algorithm 1: Agnostic Cumulant-based Estimation (ACE)

Input : Nuisance estimates ˆ g and ˆ q ; observations D = { Z i = ( X i , T i , Y i ) } n i =1 ; order r . Output: An estimate ˆ θ of the treatment effect θ 0 defined in (1).

- 1 Split the data into two sets: D 1 = { ( X i , T i , Y i ) } n/ 2 i =1 , D 2 = { ( X i , T i , Y i ) } n i = n/ 2+1 ;
- 3 µ ′ k ← 2 n ∑ n/ 2 i =1 ( Y i -ˆ g ( X i )) k ;
- 2 for k ← 1 to r +1 do
- 4 Define the cumulant-based function ˆ J r ( η ) as in (18);
- 5 Define the moment function ˆ m ( Z, θ, h ( X )) = [ Y q ( X ) θ ( T g ( X ))] ˆ J ( T g ( X ))
- 6 return ˆ θ ← solution of 2 n ∑ n i = n/ 2+1 ˆ m r ( Z i , θ, ˆ h ( X i )) = 0
- r ---r -;

The next two theorems, proved in Appendix G.3 and Appendix G.2 respectively, show that ACE can achieve higher-order error rates for treatment effect estimation when η is non-Gaussian.

Theorem 5.3 (ACE estimation error) . Let r ∈ Z + and δ id , C θ , C T , C Y &gt; 0 be constants and P be the set of all distributions in P 2 r +2 ( C θ , C T , C Y ) with η independent of X and | κ r +1 | ≥ δ id ,r . Then, for any γ ∈ (1 / 2 , 1) , there exists C γ &gt; 0 such that for all ϵ 1 , ϵ 2 &gt; 0 , if

<!-- formula-not-decoded -->

where b 1 = log( γn/ 100) , b 2 = 50 min { 1 , C θ } δ id ,r max { ϵ 1 , ϵ 2 , ( γn ) -1 / 2 C Y } -1 , a 1 = 2 log( C T ϵ -1 1 / 2) , and a 2 = C T , then then the r -th order ACE estimator ˆ θ satisfies

<!-- formula-not-decoded -->

Remark 5.3 (Power of non-Gaussianity) . When η is Gaussian, its cumulant κ r +1 = 0 for all r , violating the assumption that | κ r +1 | ≥ δ id ,r in Theorem 5.3. Conversely, for non-Gaussian η , this condition is always satisfied for some r by Levy's Inversion Formula [Durrett, 2019, Theorem 3.3.11], allowing us to obtain higher-order error rates.

Notably, the constant C T in (19) may itself grow with r . For example, if η = T -g 0 ( X ) is subGaussian, we can have C T = Θ( √ r ) . The theorem below makes this dependence explicit and delivers an even sharper bound in the sub-Gaussian regime.

Theorem 5.4 (ACE estimation error: sub-Gaussian noise) . Let δ id , C θ , C g , C q , ψ η , ψ ξ &gt; 0 and r ∈ Z + be constants and P be the set of all distributions in P ∗ ( C θ , C g , C q ; ψ ξ , ψ η ) with η independent of X and | κ r +1 | ≥ δ id ,r . Then, for any γ ∈ (1 / 2 , 1) , there exists C γ &gt; 0 such that ∀ ϵ 1 , ϵ 2 &gt; 0 , if

<!-- formula-not-decoded -->

where b 1 = log( γn/ 9) , b 2 = 200min { 1 , C θ } δ id ,r max { ϵ 1 , ϵ 2 , ( γn ) -1 / 2 ( ψ ξ + C θ ψ η ) } -1 , a 1 = 2 log(6( C g + ψ η ) ϵ -1 1 ) , and a 2 = 4( C g + ψ η ) then the r -th order ACE estimator ˆ θ satisfies

<!-- formula-not-decoded -->

Remark 5.4 (Scale of the leading coefficient under uniform noise) . As shown in (22) , the estimation error of r -th order ACE estimator depends not only on r, ϵ 1 , ϵ 2 , n , but also on κ r +1 . This is intuitive as κ r +1 is a measure of non-Gaussianity. An estimate of κ r +1 can also be used to estimate the variance of ˆ θ ; see Section 6 for more details. To understand the role of δ id ,r in the bound, consider the case when η follows a uniform distribution on [ -1 , 1] . Then for any m ∈ Z + , we have κ 2 m ∼ 4 √ π m ( m πe ) 2 m [Binet, 1839]. Plugging into (22) , we have

<!-- formula-not-decoded -->

Hence the leading coefficient is only exponential in r , rather than super-exponential.

When r = 1 , ACE is identical to DML. When r = 2 , 3 it recovers the 'second-order' orthogonal estimators proposed by Mackey et al. [2018]. Interestingly, for r = 3 , the rate given by Theorem 5.4 is faster than that of Mackey et al. [2018, Thm. 10], as the latter did not establish third-order orthogonality. When r ≥ 4 , to the best of our knowledge, ACE is novel, and we derive the explicit expressions for r = 3 , 4 in Appendix G.4.

As a concrete instantiation of Theorem 5.4, consider the setting of high-dimensional linear nuisance,

<!-- formula-not-decoded -->

where ( p, s 1 , s 2 ) all potentially grow with n , and the nuisance functions are estimated using Lasso regression [Hastie et al., 2015, Chap. 11]. In this setting, DML is known to provide order n -1 / 2 estimation error for θ 0 whenever the maximum sparsity level max( s 1 , s 2 ) = o ( n 1 / 2 / log p ) [Chernozhukov et al., 2018, Rem. 4.3]. Remarkably, as we prove in Appendix G.5, r -th order ACE provides the same guarantee when max( s 1 , s 2 ) = o ( n r/ ( r +1) / log p ) .

## 6 Numerical experiments

To evaluate the empirical effectiveness of our proposed estimators, we simulate a demand estimation scenario using purchase and pricing data. In this setting, Y represents observed demand, the treatment T corresponds to an observed product price, g 0 ( X ) denotes a baseline product price determined by covariates X that influence pricing policy, and the treatment noise η represents a random discount offered to customers for demand assessment. Notably, η is typically discrete (and thus distinctly non-Gaussian) and independent of X .

We replicate the experimental framework of Mackey et al. [2018, Section 5], where X ∼ N (0 , I ) , ϵ ∼ U ([ -3 , 3]) , and η follows a discrete distribution on { 0 . 5 , 0 , -1 . 5 , -3 . 5 } with probabilities

<!-- image -->

(d) First-order ACE estimates.

(e) Fifth-order ACE estimates.

2 × 101

(f) Fifth-order confidence intervals.

Figure 1: Comparison of first through fifth-order ACE estimation (Algorithm 1) in the synthetic demand estimation setting of Section 6. Fourth-order ACE is omitted due to substantially larger error. All quality measures and shaded 95% confidence bands are estimated using 20000 independent replicates of the experiment.

{ 0 . 65 , 0 . 2 , 0 . 1 , 0 . 05 } , respectively. Each nuisance function is specified as a sparse linear function in p = 100 dimensions with s = 40 non-zero coefficients.

We examine the r -th order ACE estimator introduced in Section 4 across different values of r . For r = 1 , 2 , this framework precisely recovers the first-order [Chernozhukov et al., 2018] and second-order [Mackey et al., 2018] orthogonal estimators. First-stage nuisance function estimates are obtained using Lasso regression [Tibshirani, 1996], following Corollary G.3. Complete Python code for replicating all experiments is available at https://github.com/JikaiJin/ACE .

In view of the high-probability bounds in Theorem 5.4, we empirically assess ACE performance for orders r ≤ 5 across varying sample sizes. A comparison of the total RMSE is provided in Figure 1a, demonstrating that the fifth-order ACE estimator achieves optimal performance. We further decompose RMSE into bias and variance components. Figure 1b compares bias across different orders, with fifth-order ACE exhibiting the smallest bias. Moreover, Figure 1c shows that the first-order ACE estimator achieves the lowest standard deviation, followed by the fifthorder estimator. Figures 1d and 1e present the distribution of estimated values using first- and fifth-order ACE estimators. Both distributions are approximately Gaussian, with the first-order estimator exhibiting substantially larger bias. Based on Theorem 4.1, the variance of ˆ θ is bounded by δ -1 id ( V m /n ) 1 / 2 , where δ id provides a lower bound for κ r +1 in the context of Theorem 5.4. This enables us to construct a direct plug-in variance estimate E var as E var = ˆ κ -1 r √ V m n for V m = 1 n ∑ n i =1 [ ( Y i -ˆ q ( X i )) 2 + ˆ θ 2 ( T i -ˆ g ( X i )) 2 ] ˆ J r ( T i -ˆ g ( X i )) 2 . Lastly, following Corollary E.1, we construct the approximate 95% confidence interval [ ˆ ϑ -1 . 96 E 1 / 2 var , ˆ ϑ +1 . 96 E 1 / 2 var ] for θ 0 . Figure 1f demonstrates that approximately 95% of independent experiments yield confidence intervals that contain the true parameter value, confirming the validity of our constructed intervals.

## 7 Conclusion and future directions

In this paper, we provide new insights into how distributional properties could change the statistical limit of structure-agnostic estimation. Focusing on a partial linear outcome model, we show that the Gaussianity of the treatment variable creates a fundamental barrier for improving over DML, while improvements upon DML is possible for non-Gaussian treatment. Moving forward, it would be of interest to exploit distributional properties to design estimators more efficient than DML for heterogeneous treatment effects.

## References

- Alireza Ansari. The gauss-airy functions and their properties. Annals of the University of CraiovaMathematics and Computer Science Series , 43(2):119-127, 2016.
- Sivaraman Balakrishnan, Edward H Kennedy, and Larry Wasserman. The fundamental limits of structure-agnostic functional estimation. arXiv preprint arXiv:2305.04116 , 2023.
- Alexandre Belloni, Victor Chernozhukov, and Christian Hansen. Inference for high-dimensional sparse econometric models. arXiv preprint arXiv:1201.0220 , 2011.
- Alexandre Belloni, Victor Chernozhukov, and Lie Wang. Pivotal estimation via square-root lasso in nonparametric regression. The Annals of Statistics , 42(2):757, 2014.
- G´ erard Biau, Luc Devroye, and G¨ abor Lugosi. Consistency of random forests and other averaging classifiers. Journal of Machine Learning Research , 9(9), 2008.
- Jacques Binet. M´ emoire sur les int´ egrales d´ efinies eul´ eriennes et sur leur application ` a la th´ eorie des suites, ainsi qu'` a l'´ evaluation des fonctions des grands nombres. Journal de l' ´ Ecole Polytechnique , 16 (27):123-343, 1839.
- St´ ephane Bonhomme, Koen Jochmans, and Martin Weidner. A neyman-orthogonalization approach to the incidental parameter problem. arXiv preprint arXiv:2412.10304 , 2024.
- Leo Breiman. Random forests. Machine learning , 45:5-32, 2001.
- T Tony Cai and Mark G Low. Testing composite hypotheses, hermite polynomials and optimal estimation of a nonsmooth functional. The Annals of Statistics , pages 1012-1041, 2011.
- Xiaohong Chen and Halbert White. Improved rates and asymptotic normality for nonparametric neural network estimators. IEEE Transactions on Information Theory , 45(2):682-691, 1999.
- Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins. Double/debiased machine learning for treatment and structural parameters: Double/debiased machine learning. The Econometrics Journal , 21(1), 2018.
- Victor Chernozhukov, Whitney K Newey, and Rahul Singh. Automatic debiased machine learning of causal and structural effects. Econometrica , 90(3):967-1027, 2022.
- Rick Durrett. Probability: theory and examples , volume 49. Cambridge university press, 2019.
- Samuel O Durugo. Higher-order Airy functions of the first kind and spectral properties of the massless relativistic quartic anharmonic oscillator . PhD thesis, Loughborough University, 2014.
- Saso Dˇ zeroski and Bernard ˇ Zenko. Is combining classifiers with stacking better than selecting the best one? Machine learning , 54:255-273, 2004.
- Max H Farrell, Tengyuan Liang, and Sanjog Misra. Deep neural networks for estimation and inference. Econometrica , 89(1):181-213, 2021.
- Trevor Hastie, Robert Tibshirani, and Martin Wainwright. Statistical learning with sparsity. Monographs on statistics and applied probability , 143(143):8, 2015.
- Kosuke Imai and David A Van Dyk. Causal inference with general treatment regimes: Generalizing the propensity score. Journal of the American Statistical Association , 99(467):854-866, 2004.
- Jikai Jin and Vasilis Syrgkanis. Structure-agnostic optimality of doubly robust learning for treatment effect estimation. arXiv preprint arXiv:2402.14264 , 2024.
- Lester Mackey, Vasilis Syrgkanis, and Ilias Zadik. Orthogonal machine learning: Power and limitations. In International Conference on Machine Learning , pages 3375-3383. PMLR, 2018.
- James Robins, Eric Tchetgen Tchetgen, Lingling Li, and Aad van der Vaart. Semiparametric minimax rates. Electronic journal of statistics , 3:1305, 2009.

- Peter M Robinson. Root-n-consistent semiparametric regression. Econometrica: Journal of the Econometric Society , pages 931-954, 1988.
- Leonas Saulis and VA Statulevicius. Limit theorems for large deviations , volume 73. Springer Science &amp; Business Media, 2012.
- Anselm Johannes Schmidt-Hieber. Nonparametric regression using deep neural networks with relu activation function. Annals of statistics , 48(4):1875-1897, 2020.
- Charles J Stone. Optimal global rates of convergence for nonparametric regression. The annals of statistics , pages 1040-1053, 1982.
- Vasilis Syrgkanis and Manolis Zampetakis. Estimation and inference with trees and forests in high dimensions. In Conference on learning theory , pages 3453-3454. PMLR, 2020.
- Robert Tibshirani. Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society Series B: Statistical Methodology , 58(1):267-288, 1996.
- Alexandre B Tsybakov. Introduction to nonparametric estimation . Springer Science &amp; Business Media, 2008.
- Roman Vershynin. High-dimensional probability: An introduction with applications in data science , volume 47. Cambridge university press, 2018.
- Lie Wang, Lawrence D Brown, T Tony Cai, and Michael Levine. Effect of mean on variance function estimation in nonparametric regression. The Annals of Statistics , pages 646-664, 2008.
- Shandong Zhao, David A van Dyk, and Kosuke Imai. Propensity score-based methods for causal inference in observational studies with non-binary treatments. Statistical methods in medical research , 29(3):709-727, 2020.
- Hui Zou and Trevor Hastie. Regularization and variable selection via the elastic net. Journal of the Royal Statistical Society Series B: Statistical Methodology , 67(2):301-320, 2005.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes] .

Justification: At the end of Section 1 we directly point to our main results.

Guidelines: See the main contributions listed at the end of Section 1.

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The main limitation is the assumed independence of noise, and this is discussed in Remark 5.2.

## Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate 'Limitations' section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: We explicitly point to the proof after stating each theorem.

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

Justification: Details are provided in Appendix H.

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

Justification: The open-source code is provided in the supplementary materials.

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

Justification: These are discussed in Section 6 and Appendix H.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Error bars and histogram under multiple runs are provided in Section 6.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer 'Yes' if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
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

Justification: The experiments are of small-scale and can be run on a laptop in a reasonable amount of time.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have checked the code of ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This paper is mainly theoretical and does not have direct societal impact.

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

Answer: [NA]

Justification: The paper does not use existing assets.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/ datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
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

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

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
- Please refer to our LLM policy ( https://neurips.cc/Conferences/ 2025/LLM ) for what should or should not be described.

## Appendix

## Table of Contents

| A   | Preliminaries                                                                  | Preliminaries                                                                                                                                                        | 21                                                                             |
|-----|--------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
|     | A.1                                                                            | Semiparametric bounds . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                    | 21                                                                             |
|     | A.2                                                                            | Useful properties of sub-Gaussian distributions . . . . . . . . . . . . . . . . . .                                                                                  | 21                                                                             |
| B   | Proof and discussion of Theorem 3.1: Structure-agnostic lower bound for binary | Proof and discussion of Theorem 3.1: Structure-agnostic lower bound for binary                                                                                       | Proof and discussion of Theorem 3.1: Structure-agnostic lower bound for binary |
|     |                                                                                | treatment                                                                                                                                                            | 22                                                                             |
|     | B.1                                                                            | Some remarks on the constants . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                    | 25                                                                             |
|     | B.2                                                                            | Discussion of the constant c q . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                 | 25                                                                             |
| C   | Proofs of upper bounds forDML                                                  | Proofs of upper bounds forDML                                                                                                                                        | 27                                                                             |
|     | C.1                                                                            | Proof of Theorem C.1: Structure-agnostic rate of DML . . . . . . . . . . . . . .                                                                                     | 27                                                                             |
|     | C.2                                                                            | Proof of Theorem C.2: Structure-agnostic upper bound with known treatment noise                                                                                      | 29                                                                             |
| D   | Proof of Theorem 3.2: The Gaussian treatment barrier                           | Proof of Theorem 3.2: The Gaussian treatment barrier                                                                                                                 | 29                                                                             |
| E   | General upper bounds under Neyman Orthogonality                                | General upper bounds under Neyman Orthogonality                                                                                                                      | 38                                                                             |
|     | E.1                                                                            | Estimation error of general moment estimators . . . . . . . . . . . . . . . . . .                                                                                    | 38                                                                             |
|     | E.2                                                                            | Proofs of Theorem E.1 and Corollary E.1 . . . . . . . . . . . . . . . . . . . . .                                                                                    | 39                                                                             |
|     | E.3                                                                            | Proof of Theorem 4.1: Structure-agnostic error from estimated moments . . . . .                                                                                      | 41                                                                             |
|     | E.4                                                                            | Proof of Lemma 4.1: Explicit formula for J r . . . . . . . . . . . . . . . . . . .                                                                                   | 42                                                                             |
| F   | Technical details in Section 5.1                                               | Technical details in Section 5.1                                                                                                                                     | 42                                                                             |
|     | F.1                                                                            | Proof of Theorem F.1: Structure-agnostic limit for estimating residual moments .                                                                                     | 43                                                                             |
|     | F.2                                                                            | Proofs of Theorem 5.1 and Theorem 5.2 . . . . . . . . . . . . . . . . . . . . .                                                                                      | 44                                                                             |
|     | F.3                                                                            | Relaxing the independent noise assumption . . . . . . . . . . . . . . . . . . . .                                                                                    | 49                                                                             |
| G   | Technical details in Section 5.2                                               | Technical details in Section 5.2                                                                                                                                     | 50                                                                             |
|     | G.1                                                                            | Proof of Lemma 5.1: Key lemma; higher-order insensitivity condition . . . . . .                                                                                      | 50                                                                             |
|     | G.2                                                                            | Proof of Theorem 5.4: ACE estimation error: sub-Gaussian noise . . . . . . . .                                                                                       | 50                                                                             |
|     | G.3                                                                            | Proof of Theorem 5.3: ACE estimation error . . . . . . . . . . . . . . . . . . .                                                                                     | 58                                                                             |
|     | G.4                                                                            | Special cases of the ACE estimator . . . . . . . . . . . . . . . . . . . . . . . .                                                                                   | 59                                                                             |
|     | G.5                                                                            | ACE estimation error for high-dimensional sparse linear regression . . . . . . .                                                                                     | 60                                                                             |
| H   | Additional experiment results and discussion                                   | Additional experiment results and discussion                                                                                                                         | 61                                                                             |
| I   | More results for orthogonal machine learning                                   | More results for orthogonal machine learning                                                                                                                         | 62                                                                             |
|     | I.1                                                                            | Construction of orthogonal moment functions . . . . . . . . . . . . . . . . . . .                                                                                    | 62                                                                             |
|     | I.2 I.3                                                                        | Proof of Theorem I.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . Example: heteroscedastic nonparametric regression . . . . . . . . . . . . . . . . | 63 65                                                                          |
|     | I.4                                                                            | Technical details in Appendix I.3 . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                  | 66                                                                             |
|     | I.5                                                                            | Proof of Corollary I.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                 | 67                                                                             |
|     | I.6                                                                            | Comparison with Mackey et al. [2018] . . . . . . . . . . . . . . . . . . . . . .                                                                                     | 68                                                                             |
|     | I.7                                                                            | More discussion on Assumption E.1 . . . . . . . . . . . . . . . . . . . . . . . .                                                                                    | 68                                                                             |

## A Preliminaries

## A.1 Semiparametric bounds

Our proofs of lower bounds are based on on the method of fuzzy hypothesis. A key lemma is stated below:

Lemma A.1. (Tsybakov [2008], Theorem 2.15) Let Z be an observation with supp( Z ) = Z , P ∈ P , P 1 ⊆ P and π be a probability distribution on P 1 , which induce the distribution Q 1 ( A ) = ∫ Q ⊗ n ( A ) dπ ( Q ) , ∀ A ⊂ P . Suppose that there exists a functional T : P ↦→ R which satisfies

<!-- formula-not-decoded -->

for some s &gt; 0 . If H 2 ( P ⊗ n , Q 1 ) ≤ ξ &lt; 2 , then:

<!-- formula-not-decoded -->

We will use the following lemma to bound the Hellinger distance as required in Lemma A.1

Lemma A.2. (Robins et al. [2009, Theorem 2.1], see also Jin and Syrgkanis [2024, Theorem 4]) let Z = ∪ j m =1 Z j be a measurable partition of the sample space. Given a vector λ = ( λ 1 , . . . , λ m ) in some product measurable space Λ = Λ 1 ×··· × Λ m , let P and Q λ be probability measures on Z such that the following statements hold:

- P ( Z j ) = Q λ ( Z j ) = p j for every λ ∈ Λ , and
- the probability measures P and Q λ restricted to Z j depend λ j only.

Let p and q λ be the densities of the measures P and Q λ that are jointly measurable in the parameter λ and the observation x , and π be a probability measure on Λ . Define b = m max j sup λ ∫ X j ( q λ -p ) 2 p dµ . Suppose that p = ∫ q λ d π ( λ ) and that n max { 1 , b } max j p j ≤ A for all j for some positive constant A , then there exists a constant C that depends only on A such that, for any product probability measure π = π 1 ⊗··· ⊗ π m ,

<!-- formula-not-decoded -->

## A.2 Useful properties of sub-Gaussian distributions

In this subsection, we recall a few useful properties of sub-Gaussian distributions. Recall that for a variable Z , its sub-Gaussian norm is defined as

<!-- formula-not-decoded -->

Proposition A.1 (Moment bounds for sub-Gaussian variables, see e.g. , Vershynin [2018] Proposition 2.5.2) . Suppose that Z is a sub-Gaussian random variable with Orlicz norm σ = ∥ Z ∥ ψ 2 . Then for every integer k ≥ 1

<!-- formula-not-decoded -->

where C &gt; 0 is an absolute constant; one may take C = 2 .

The following bound of cumulants is due to Saulis and Statulevicius [2012], Lemma 1.5.

Proposition A.2 (Cumulant bounds for sub-Gaussian variables) . Let Z be a centred sub-Gaussian random variable with Orlicz norm σ = ∥ Z ∥ ψ 2 , i.e. E [ e tZ ] ≤ exp ( σ 2 t 2 / 2 ) for all t ∈ R . Denote by κ r ( Z ) its r -th cumulant, r ∈ N . Then

<!-- formula-not-decoded -->

In particular, the sequence {| κ r ( Z ) | 1 /r } r ≥ 2 grows at most like 2 σ √ r , which is sharp up to the constant 2 (no smaller absolute constant works for all sub-Gaussians).

## B Proof and discussion of Theorem 3.1: Structure-agnostic lower bound for binary treatment

In this section, we present the proof of Theorem 3.1.

We define the following data generating distribution:

<!-- formula-not-decoded -->

where ˆ θ = 1 2 c q and ˆ f ( x ) = ˆ q ( x ) -ˆ θ ˆ g ( x ) . Since c q ≤ ˆ q ( x ) ≤ 1 -c q and 0 ≤ ˆ g ( x ) ≤ 1 by assumption, it is easy to see that ˆ f ( x ) + θt ∈ [0 , 1] , t ∈ { 0 , 1 } . Hence (27) defines a valid data generating distribution. We denote the joint distribution of ( X,T,Y ) as ˆ P .

Since γ ∈ ( 1 2 , 1 ) , there always exists some ζ ∈ (0 , 2) such that 1 -√ ζ (1 -ζ/ 4) 2 = γ . Let m ≥ Cn 2 ˆ θ 4 ζ be a positive integer and B i , i = 1 , 2 , · · · , 2 m be a partition of the covariate space X = [0 , 1] K such that each set has a Lebesgue measure of 1 2 m . We also define

<!-- formula-not-decoded -->

where and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let P λ be the joint distribution of ( X,T,Y ) induced by (28), µ be the uniform measure over X × T × Y and p λ = d P λ d µ . From (28) we can derive the expressions of p λ ( x, t, y ) as follows:

<!-- formula-not-decoded -->

Specifically, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Crucially, all the p λ -ˆ p 's are linear functions of ∆( λ, x ) .

Lemma B.1 ( L 2 -norm bounds) . Let ˜ ϵ 1 = A -1 / 2 δ -1 / 2 ϵ 1 , ˜ ϵ 2 = A -1 / 2 δ -1 / 2 ϵ 2 , then if ϵ 1 , ϵ 2 ≤ δ 2 , it holds that

<!-- formula-not-decoded -->

As a result, P λ ∈ P 2 ,ϵ h ) .

Proof : By definition, we have

<!-- formula-not-decoded -->

Similarly ∥ q λ -ˆ q ∥ L 2 ( P X ) = ϵ 2 . Hence it directly follows that P λ ∈ P 2 ,ϵ ( ˆ h ) . By our assumption, ˆ g ( x )(1 -ˆ g ( x )) ≥ A -1 δ , so that A -1 ˆ g ( x ) ≥ δ ≥ ˜ ϵ 2 1 and

By a similar argument, one can show that g λ ( x ) &lt; 1 and also q λ ( x ) ∈ (0 , 1) , concluding the proof. □

<!-- formula-not-decoded -->

To apply Lemma A.1, we now use Lemma A.2 to bounding the Hellinger distance between ˆ P ⊗ n and ∫ P ⊗ n λ d π ( λ ) . We recall the following lemma:

Lemma B.2 (Joint density lower bound) . If ϵ 1 , ϵ 2 ≤ δ 8 , we have p λ ( x, t, y ) ≥ 1 8 δ 2 , ∀ ( x, t, y ) ∈ X × T × Y .

Proof : Note that ˆ f ( x ) = ˆ q ( x ) δ ˆ g ( x ) [ δ , 1 δ ] and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so we have that f λ ( x ) ∈ [ δ 4 , 1 -δ 4 ] . Similarly, one can show that f λ ( x ) + θ ′ ∈ [ δ 4 , 1 -δ 4 ] . By Lemma B.1, g λ ( x ) ∈ [ δ 2 , 1 -δ 2 ] , so the conclusion directly follows from (30). □

<!-- formula-not-decoded -->

Lemma B.3 (Hellinger distance bound) . For any ζ &gt; 0 , as long as M ≥ , it holds that

Proof : Let Z = X × T × Y , where T = Y = {-1 , +1 } . We apply Lemma A.2 with Λ i = {-1 , +1 } , Z j = ( B 2 j -1 ∪ B 2 j ) ×{-1 , +1 }×{-1 , +1 } , P = ˆ P,Q λ = P λ and π being the uniform distribution on {-1 , +1 } m . Firstly, since λ i ∼ Uniform( {-1 , +1 } ) , we have E π [∆( λ, x )] = 0 for any fixed x ∈ X , so that (31) implies that E π P λ = ˆ P .

<!-- formula-not-decoded -->

By our choice of B j , we have ˆ P ( Z j ) = P λ ( Z j ) = 1 m for all j , so we have that p j = 1 m . Notice that where we recall that µ ( Z j ) = 1 m since µ is the uniform distribution on Z . When t = y = 1 , we have

where the last step holds since ˆ q ( x ) + 2 ˆ θ (1 -ˆ g ( x )) ≥ 0 by our choice of ˆ θ and ˆ q ( x ) ≥ 0 . Similarly, we can deduce the same bound for ( t, y ) ∈ { (0 , 1) , (1 , 0) , (1 , 1) } . Hence we have that b ≤ ˆ θ -1 . We can choose A = ˆ θ -1 and n max { 1 , b } max j p j ≤ ˆ θ -1 nm -1 ≤ A is satisfied. Therefore, by Lemma A.2 we can deduce that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last step holds since m ≥ Cn 2 ˆ θ 2 ζ .

□

Now we are ready to apply Lemma A.1. We choose Z = ( X,T,Y ) , P = ˆ P, P 1 = { P λ : λ ∈ {-1 , +1 }} , π be the uniform distribution on P 1 , and T that maps each observation distribution P generated by (1) to the corresponding θ . Then we have that

<!-- formula-not-decoded -->

where s = 1 2 ( θ ′ ˜ ϵ 2 1 + ˜ ϵ 1 ˜ ϵ 2 ) = (4 Aδ ) -1 ( c q ϵ 2 1 + ϵ 1 ϵ 2 ) . Moreover, our choice of m and Lemma B.3 together implies that H 2 ( P ⊗ n , ∫ Q ⊗ n d π ( Q ) ) ≤ ζ . Therefore, Lemma A.1 implies that for any estimator ˆ T , it holds that

<!-- formula-not-decoded -->

Equivalently, we have

<!-- formula-not-decoded -->

We now proceed to prove the n -1 / 2 component of the lower bound. Define

<!-- formula-not-decoded -->

and let ˜ P be the distribution generated by x ∼ P X , T | X = x ∼ Bernoulli(ˆ g ( x )) , Y | X = x, T = t ∼ Bernoulli(˜ q ( x ) -ˆ g ( x ) + ˜ θt ) and ˜ p ( x, t, y ) be the density. Then we have that

<!-- formula-not-decoded -->

so that

<!-- formula-not-decoded -->

We also have that so that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we use ˆ θ = 1 2 c q and 1 -ˆ q ( x ) ≥ c q in the last step. Similarly, one can show that

Combining all the inequalities above, we have

<!-- formula-not-decoded -->

By choosing ϵ = 0 . 1 ζc 1 / 2 q δ -1 / 2 n -1 / 2 , it holds that

<!-- formula-not-decoded -->

so Lemma A.1 directly implies that

Therefore, Lemma A.1 implies that for any estimator ˆ T , it holds that

<!-- formula-not-decoded -->

Equivalently, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining (33) and (35), we obtain the desired result.

## B.1 Some remarks on the constants

Notice that the assumptions that we make for deriving the upper and lower bounds are not exactly the same. Here we discuss some important aspects of their differences.

Remark B.1 (Assumptions on the uniform overlap) . Compared with Theorem C.1, we make the additional assumption that ˆ g ( x )(1 -ˆ g ( x )) ≥ A -1 δ , which we will refer to as uniform overlap. Equivalently, this assumption states that there exists some δ 1 = Θ( A -1 δ ) such that δ 1 ≤ ˆ g ( x ) ≤ 1 -δ 1 . This assumption is also made in previous work [Jin and Syrgkanis, 2024], albeit for a different causal estimand. If δ 1 is treated as a universal constant, then we can choose A = δ -1 1 and the expected overlap assumption is satisfied. In this case, our lower bound has tight dependency on both ϵ i , i = 1 , 2 and δ . However, the error rate of DML does not match the lower bound in δ if ˆ g is not uniformly extreme. The minimax optimal rate is unknown in this regime.

Remark B.2 (Assumptions on ˆ q ( · ) ) . Compared with Theorem C.1, another additional assumption that we make is that c q ≤ ˆ q ( x ) ≤ 1 -c q in Theorem 3.1. This assumption is also needed in the previous lower bound established in Balakrishnan et al. [2023] for E [Cov( Y, T | X )] . Interestingly, this assumption is not needed for upper bound. The c q ϵ 2 1 term in our lower bound (5) corresponds to the C θ ϵ 2 1 term in our upper bound, where we recall that C θ is an upper bound on the ground-truth θ 0 . In fact, the overlap assumption on ˆ q ( x ) also implicitly imposes a constraint on the magnitude of the ground-truth parameter θ 0 ; more discussions can be found in Appendix B.2.

## B.2 Discussion of the constant c q

For any pair of functions ( g, q ) that take values in [0 , 1] , we define their cross ratio to be

<!-- formula-not-decoded -->

First, note that Theorem 3.1 can be slightly strengthened as follows:

Theorem B.1 (Strengthened binary lower bound) . Let c q , δ ∈ (0 , 1 4 ) and P be the set of all possible P 's generated by (1) , such that the variables T, Y are binary and that the marginal distribution of P on X is P X . Let Φ be a mapping that maps a distribution P ∈ P to the nuisance functions h 0 = ( g 0 , q 0 ) ∈ H = range(Φ) . Then for any γ ∈ (1 / 2 , 1) , there exists a constant c γ &gt; 0 such that for any ϵ i ≤ δ/ 2 , i = 1 , 2 and any estimates ˆ h = (ˆ g, ˆ q ) with E P X [ˆ g ( X )(1 -ˆ g ( X ))] = 2 δ and ˆ g ( x )(1 -ˆ g ( x )) ≥ A -1 δ, ∀ x ∈ X , we have

<!-- formula-not-decoded -->

where γ is a universal constant that only depends on γ , and

<!-- formula-not-decoded -->

Proof : Without loss of generality, we assume that

<!-- formula-not-decoded -->

For any h = ( g, q ) ∈ P 2 ,ϵ/ 2 ( ˆ h ) , note that

<!-- formula-not-decoded -->

so that

<!-- formula-not-decoded -->

Now it suffices to show that

<!-- formula-not-decoded -->

Notice that this lower bound can be derived with exactly the same argument as we employed in the previous section, since the only place that we use the assumption c q ≤ ˆ q ( x ) ≤ 1 -c q is that

Now we replace ˆ g, ˆ q with g, q respectively, and choosing ˆ θ = c θ / 2 ensures that the above relationship holds. Therefore we obtain the desired lower bound. □

<!-- formula-not-decoded -->

The upper bound side can also be strengthened by replacing C θ with

<!-- formula-not-decoded -->

Theorem B.2 (Strengthened binary upper bound) . Let δ &gt; 0 and P be the set of all distributions P of ( X,T,Y ) generated from (1) that satisfies E P [( T -g 0 ( X )) 2 ] ≥ δ and T, Y are binary. Let Φ be a mapping that maps each P ∈ P to ( g 0 , q 0 ) ∈ H = range( H ) . Then there exists a constant n 0 = n 0 ( δ ) such that when n ≥ n 0 , for any estimates ˆ h = (ˆ g, ˆ q ) and any γ ∈ (0 , 1) , the DML estimator ϑ DML derived from the moment function

The main insight is that the nuisance estimates already tells us that | θ | ≤ C ′ θ .

<!-- formula-not-decoded -->

satisfies

<!-- formula-not-decoded -->

for any γ ∈ (0 , 1) , where A γ is a constant that only depends on γ .

Proof : We prove the theorem by applying Theorem C.1. First, since T and Y are binary, we can take C = 1 . It remains to show that for any P ∈ P 2 ,ϵ ( ˆ h ) , the corresponding θ 0 is bounded by C ′ θ . Indeed, let g 0 ( x ) = E P [ T | X = x ] and q 0 ( x ) = E P [ T | X = x ] , then ∥ g 0 -ˆ g ∥ L 2 ( P X ) ≤ ϵ 1 and ∥ q 0 -ˆ q ∥ L 2 ( P X ) ≤ ϵ 2 . Moreover, note that

<!-- formula-not-decoded -->

so we have that concluding the proof.

<!-- formula-not-decoded -->

□

## C Proofs of upper bounds for DML

In this section, we present the formal statements and proofs of Theorem C.1 and Theorem C.2. Both results are already known in the literature, and we present here the explicit structure-agnostic rates for completeness.

Theorem C.1 (Structure-agnostic rate of DML) . Let δ, C θ , C T , C Y &gt; 0 , P 0 = { P 0 ∈ P ⋆ 0 ( C θ , C T , C Y ) : E P 0 [( T -g 0 ( X )) 2 ] ≥ δ } , and Φ = Φ ⋆ . Then for any estimates ˆ h = (ˆ g, ˆ q ) and any γ ∈ (0 , 1) such that | ˆ g ( X ) | ≤ C T a.s., the DML estimator ϑ DML derived from the moment function (38) satisfies

<!-- formula-not-decoded -->

for any γ ∈ (0 , 1) and δ ≥ 15 C 1 / 2 T n -1 / 2 , where C γ is a constant that only depends on γ .

The proof of this result follows the standard arguments in the DML literature and can be found in Appendix C.1. Existing theoretical guarantees largely focus on establishing sufficient conditions for achieving O ( n -1 / 2 ) rate and establishing confidence intervals [Chernozhukov et al., 2018, 2022], while here we make the dependency of the error on ϵ i , i = 1 , 2 and δ explicit, which would be of interest when the rate is slower than n -1 / 2 . The existence a constant δ &gt; 0 that satisfies the assumption in Theorem C.1 is commonly referred to as the overlap assumption and is widely adopted in the DML literature. The estimation error still be large if δ is small compared to ϵ g , ϵ q . When the assumption on δ is violated, i.e. , δ = O ( n -1 / 2 ) , the second term in the upper bound becomes Ω( ϵ g ) , so that DML is not better than the naive estimator 1 n ∑ n i =1 (ˆ g (1 , X i ) -ˆ g (0 , X i )) .

On the other hand, since σ is known, the following upper bound can be easily established for a modified version of DML, using the moment function

<!-- formula-not-decoded -->

Theorem C.2 (Structure-agnostic upper bound with known treatment noise) . Let Φ , H , ˆ h be defined as in Theorem 3.2, and s 1 , s 2 &gt; 0 satisfy s -1 1 + s -1 2 ≤ 1 , then the estimator ˜ ϑ DML derived from the moment function (39) satisfies

<!-- formula-not-decoded -->

for any γ ∈ (0 , 1) , where C γ is a constant that only depends on γ .

Theorem C.2 can be derived in a similar way as Theorem C.1; the proof can be found in Appendix C.2. Since it considers a simplified setting where the treatment variance is known in X , the ϵ 2 1 term in Theorem C.1 induced by estimating the treatment variance vanishes in the current upper bound.

When the nuisance error of q 0 is small, i.e., ϵ 2 ≤ C θ , the upper bound matches the lower bound derived in Theorem 3.2 up to logarithmic factors. Moreover, if the estimation error ϵ g of g 0 is polynomial in n , then they differ by only a constant factor, implying that DML is minimax optimal.

## C.1 Proof of Theorem C.1: Structure-agnostic rate of DML

Given data { ( X i , T i , Y i ) } n i =1 , the DML estimator is defined by

<!-- formula-not-decoded -->

Let η i = T i -g 0 ( X i ) , ϵ i = Y i -q 0 ( X i ) -θ 0 η i , ∆ g = ˆ g -g and ∆ q = ˆ q -q , then

<!-- formula-not-decoded -->

For any γ ∈ (0 , 1) , by assumption there exists some constant N γ such that for any n ≥ N γ , we have

<!-- formula-not-decoded -->

Since { ∆ g ( X i ) η i } n i =1 are i.i.d random variables with E [∆ g ( X i ) η i ] = E [∆ g ( X i ) E [ η i | X i ]] = 0 and E [∆ g ( X i ) 2 η 2 i ] = E [∆ g ( X i ) 2 E [ η 2 i | X i ]] ≤ 4 C 2 T E [∆ g ( X i ) 2 ] ≤ 4 C 2 T ϵ 2 g

<!-- formula-not-decoded -->

we have by Chebyshev's inequality, where A = 0 . 1 γ -1 / 2 . Similarly, we also have

<!-- formula-not-decoded -->

From we deduce that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since E [ | ∆ g ( X )∆ q ( X ) | ] ≤ ϵ g ϵ q by Cauchy-Schwarz inequality, Chebyshev's inequality again implies that

<!-- formula-not-decoded -->

with a similar reasoning, we have

<!-- formula-not-decoded -->

Lastly, since E [ η 2 i ] ≥ δ by assumption, we also have that

<!-- formula-not-decoded -->

so that

<!-- formula-not-decoded -->

Let E be the event that all the above high-probability bounds hold, then

Under E , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

since E [ η 2 ] ≥ δ by assumption and (40) implies that δ ≥ 4 ACn -1 / 2 . Moreover,

It is easy to see that E [∆ g ( X ) 2 ∆ q ( X ) 2 ] ≤ 4 C 2 Y ϵ 2 g and E [∆ g ( X ) 4 ] ≤ 4 C 2 T ϵ 2 g . As a result, we can deduce that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

concluding the proof.

## C.2 Proof of Theorem C.2: Structure-agnostic upper bound with known treatment noise

Since the variance of η | X is assumed to be σ 2 where σ is a known constant, we have

<!-- formula-not-decoded -->

so that

<!-- formula-not-decoded -->

Note that the high-probability bounds for each term in the above expression can be obtained with similar arguments as in the previous subsection, with C = Θ( σ ) and δ = σ 2 . Hence it is straightforward to deduce that

<!-- formula-not-decoded -->

## D Proof of Theorem 3.2: The Gaussian treatment barrier

Our proof is based on a constrained risk inequality for testing composite hypothesis developed in Cai and Low [2011].

Lemma D.1. [Cai and Low, 2011, Corollary 1] Let X be an observation with distribution P ∈ P and P i , i = 0 , 1 be two subsets of P satisfying P 1 ∪P 2 = P , and µ i be some distribution supported on P . Define

<!-- formula-not-decoded -->

to be the mean and variance of a functional T : P ↦→ R , F i be the distribution of X with prior µ i and f i be its density with respect to some common dominating measure µ . Then for any estimator ˆ T ( X ) we have that

<!-- formula-not-decoded -->

as long as | m 1 -m 0 | -v 0 I ≥ 0 , where I = ( E f 0 [ ( f 1 ( X ) f 0 ( X ) -1 ) 2 ]) 1 / 2 is the χ 2 -distance between F 0 and F 1 .

To apply this inequality, we construct the null and alternative hypotheses as mixtures of data distributions with matching moments. While moment matching techniques are widely adopted in establishing minimax lower bounds, the structural nature of our causal model (1) brings additional challenges to our construction. Unlike most existing works where only moments of a single variable need to be matched, here we need to match moments that contain two variables: we seek for distributions ν 0 , ν 1 over P s,ϵ ( ˆ h ) with corresponding mixtures ¯ P 0 , ¯ P 1 respectively, such that both

<!-- formula-not-decoded -->

hold for k = 1 , 2 , · · · , k n . This would imply that χ 2 ( ¯ P 0 || ¯ P 1 ) is small, which further implies that χ 2 ( ¯ P ⊗ n 0 || ¯ P ⊗ n 1 ) is also small, where ¯ P ⊗ n i := ∫ P ⊗ n d ν i , i = 1 , 2 and P ⊗ is the n -fold product distribution.

To apply Lemma D.1, we need to show that there exists a sufficient gap between m 0 and m 1 , which correspond to the expected value of θ under ν 0 and ν 1 respectively. Our key insight is that, for Gaussian treatment, there is no need to match E [( Y -E [ Y | X ])( T -E [ T | X ])] since this term always vanishes. This fact is due to a recursive property of the Hermite polynomial H k ( x ) =

( -1) k φ ( k ) ( z ) /φ ( z ) (where φ ( · ) is the Gaussian density); we will elaborate on this connection in Lemma D.6. As a result, we can construct mixtures of distributions that are close in terms of χ 2 -distance (Corollary D.1) but their average values of the E [( Y -E [ Y | X ])( T -E [ T | X ])] term are well-separated. Given the structure-agnostic oracle, this separation can be as large as ˜ Ω( ϵ g ϵ q ) , and it further induces a separation between m 0 and m 1 at the same scale, yielding the desired lower bound.

In the following, we present the full proof of this theorem.

The following lemma turns out to be a useful tool for moment matching in establishing our lower bounds.

LemmaD.2 ( L ∞ -distance to univariate polynomial bases) . Let P k be a linear space of polynomials on [ -1 , 1] in the form of ∑ m i =1 a i λ u i where u i ∈ { 0 , 1 , · · · , k } \ { 1 } , and δ k be the L ∞ -distance of a ( λ ) = λ to P k . Then δ k ≥ 1 2 k 3 .

Proof : Let b ( λ ) ∈ P k be a polynomial that satisfies ∥ a -b ∥ L ∞ = δ k . Define r = a -b , then r satisfies ∥ r ∥ L ∞ = δ k and r ′ (0) = 1 .

Since deg( r ) ≤ k , the Lagrange interpolation formula implies that

̸

<!-- formula-not-decoded -->

̸

for any x i ∈ [ -1 , 1] , 1 ≤ i ≤ 2 k . Taking the derivative of both sides, we obtain

̸

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

In particular, we choose then it holds that

̸

<!-- formula-not-decoded -->

As a result, we have

̸

̸

̸

̸

<!-- formula-not-decoded -->

Hence δ k ≥ 1 2 k 3 as desired.

□

LemmaD.3 ( L ∞ -distance to bivariate polynomial bases) . Let P k, 1 be a linear space of polynomials on [ -1 , 1] 2 of the form ∑ m i =1 a i λ u i ρ v i where ( u i , v i ) ∈ { 0 , 1 , · · · , k } × { 0 , 1 } \ { (1 , 1) } , and δ k, 1 be the L ∞ -distance from a 1 ( λ, ρ ) = λρ to P k, 1 . Then δ k, 1 ≥ 1 2 δ k .

Proof : Assume the contrary holds, i.e. δ k, 1 &lt; 1 2 δ k , then there exists some b 1 ( λ, ρ ) ∈ P k, 1 such that ∥ b 1 -a 1 ∥ L ∞ &lt; 1 2 δ k . By definition, there exists polynomials r 1 ∈ P k and s 1 such that b 1 ( λ, ρ ) = ρr 1 ( λ ) + s 1 ( λ ) . In particular, setting ρ = 1 and ρ = -1 implies that ∥ r 1 + s 1 -λ ∥ L ∞ &lt; 1 2 δ k and ∥ r 1 -s 1 -λ ∥ L ∞ &lt; 1 2 δ k . The triangle inequality implies that ∥ r 1 -λ ∥ L ∞ &lt; δ k , which is a contradiction to the definition of δ k . Thus the conclusion follows. □

Lemma D.4 (Separation of measures under matching properties) . There exists two probability measures ν 0 and ν 1 on [ -1 , 1] 2 such that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Proof : The proof is similar to that of [Cai and Low, 2011, Lemma 1]. Let C ([ -1 , 1] 2 ) be the space of continuous functions on [ -1 , 1] 2 equipped with the L ∞ norm and F be linear space spanned by a 1 ( λ, ρ ) = λρ and P k, 1 . Define a linear functional T that maps any f = ca 1 + g ∈ F (where c ∈ R and g ∈ P k ) to cδ k, 1 , where δ k, 1 is defined in the previous lemma. Let g 1 ∈ P k be the best L ∞ -approximation of a 1 in P k , then ∥ a 1 -g 1 ∥ L ∞ = δ k, 1 and T ( a 1 -g 1 ) = δ k, 1 , so ∥ T ∥ ≥ 1 . On the other hand, for any f = ca 1 + g ∈ F , we have ∥ f ∥ ∞ ≥ | c | δ k, 1 since otherwise the L ∞ distance between a 1 and -c -1 g would be smaller than δ k, 1 , which is a contradiction. Thus | T ( f ) | = | c | δ k, 1 ≤ ∥ f ∥ L ∞ , which implies that ∥ T ∥ ≤ 1 .

Therefore, we must have ∥ T ∥ = 1 . By Hahn-Banach theorem, T can be extended to a linear functional on C ([ -1 , 1] 2 ) with unit norm, which we still denote by T . The Riesz representer theorem then implies that there exists a signed measure µ with unit total variation such that T ( f ) = ∫ f d µ, ∀ f ∈ C ([ -1 , 1] 2 ) . In particular, we have

<!-- formula-not-decoded -->

Finally, by the Hahn decomposition theorem, there exists (positive) measures ν 0 , ν 1 such that µ = ν 0 -ν 1 . Then it is easy to see that such ν 0 and ν 1 satisfiy the desired properties, concluding the proof. □

In the following, we provide the full proof of Theorem 3.2. For any A &gt; 0 and q ∈ (0 , 1) , we define a two-piece Bernoulli distribution , denoted by B 2 ( q ; A ) , as a distribution with PDF

<!-- formula-not-decoded -->

It is easy to see that such a distribution has mean ( q -1 2 ) A .

To begin with, note that we can assume that | ˆ g ( X ) | ≤ C T -ϵ 1 / 2 and | ˆ q ( X ) | ≤ C q -ϵ 2 / 2 without loss of generality. Indeed, since | ˆ g ( X ) | ≤ C T and | ˆ q ( X ) | ≤ C q , there exists ˜ h = (˜ g ( · ) , ˜ q ( · )) satisfying ∥ ˆ q -˜ q ∥ L ∞ ≤ ϵ 1 / 2 , ∥ ˆ g -˜ g ∥ L ∞ ≤ ϵ 2 / 2 and | ˜ g ( X ) | ≤ C T -ϵ 1 / 2 , | ˜ q ( X ) | ≤ C q -ϵ 2 / 2 . Then, P s,ϵ/ 2 ( ˜ h ) ⊆ P s,ϵ ( ˆ h ) , and any lower bound for P s,ϵ/ 2 ( ˜ h ) also applies to P s,ϵ ( ˆ h ) , implying the desired lower bound up to constants. In the following, we will replace ˆ h with ˜ h work with the uncertainty set P s,ϵ/ 2 ( ˆ h ) .

Now, let A = C q , Q = C q -ϵ 2 / 2 and G = C T -ϵ 1 / 2 . Let k n &gt; 0 be some even integer that will be specified later, and ν 0 , ν 1 be the corresponding distributions in Lemma D.4. For any λ, ρ ∈ R , we define the following data generating process:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

By choosing ˜ ϵ 1 = σ -1 ϵ 1 / 4 and ˜ ϵ 2 = ϵ 2 / 4 , we can ensure that for any h λ = ( g λ , q λ ) it holds that h λ ∈ P s,ϵ/ 2 ( ˆ h ) , ∀ s ∈ [1 , + ∞ ] 2 .

We use P λ,ρ to denote the joint distribution of ( X,T,Y ) in (44), and p λ,ρ be its density. Then we have that

<!-- formula-not-decoded -->

The following lemma derives an equivalent expression for E [ Y | X = x, T = t ] = θ λ,ρ t + f λ,ρ ( x ) :

Lemma D.5 (Expression for conditional mean outcome) . For any x, t we have

<!-- formula-not-decoded -->

Note that the last event in (45) happens with small probability. Indeed, we can define a good event

<!-- formula-not-decoded -->

An important property of the above definition is that the bound goes to infinity since we assumed that ϵ 1 = o (1) and ϵ 2 = o (1) , so that it would happen with high probability.

The following result summarizes the good properties enjoyed by E ˜ ϵ 1 , ˜ ϵ 2 :

Proposition D.1 (Properties of good events) . We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof : By definition, P λ,ρ [ E ˜ ϵ 1 , ˜ ϵ 2 ] = P T ∼N ( g λ ( x ) ,σ 2 ) [ | T | ≤ A -Q -˜ ϵ 2 ˜ ϵ 1 ˜ ϵ 2 max { 1 ,σ } 2 -G -1 ] . Since | g λ ( x ) | ≤ G + o (1) , (1) directly follows. To prove (2), it suffices to note that

<!-- formula-not-decoded -->

where the first equation is due to Lemma D.5.

□

Let H k be the k -th order Hermite polynomial and ¯ P 0 and ¯ P 1 be the mixture of P λ,ρ with priors ν 0 and ν 1 respectively, and ˆ p 0 and ¯ p 1 be their densities. Our next step would be to bound the χ 2 -divergence between ¯ P 0 and ¯ P 1 . To do this, we need to analyze the densities ¯ p i ( x, t, y ) , i = 0 , 1 for two cases ( x, t, y ) ∈ E ˜ ϵ 1 , ˜ ϵ 2 and ( x, t, y ) / ∈ E ˜ ϵ 1 , ˜ ϵ 2 separately. Our next two lemmas handle the first case.

Lemma D.6 (Taylor expansions of perturbed densities) . Suppose that t ∈ E ˜ ϵ 1 , ˜ ϵ 2 , then we have

<!-- formula-not-decoded -->

where z = t -ˆ g ( x ) σ .

Proof : We only prove the statement for y ∈ [ -A, 0) ; the other case y ∈ [0 , A ] can be handled similarly. Since

<!-- formula-not-decoded -->

and by Lemma D.5, we can deduce that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in the final step holds since H k ( z ) = zH k -1 ( z ) -( k -1) H k -2 ( z ) . The conclusion follows. □

Lemma D.7 (Bounding χ 2 -distance under good event) . We have

<!-- formula-not-decoded -->

Proof : Let ( x, t, y ) satisfies t ∈ E ˜ ϵ 1 , ˜ ϵ 2 and y ∈ [ -A, 0) , then Lemma D.6 implies that

<!-- formula-not-decoded -->

where z = t -ˆ g ( x ) σ and

<!-- formula-not-decoded -->

On the other hand, since x ↦→ e -x is convex, by Jensen's inequality we have

<!-- formula-not-decoded -->

for sufficiently large n , since ˜ ϵ 1 , ˜ ϵ 2 = o (1) . Hence,

<!-- formula-not-decoded -->

Fixing x, y and integrating both sides of (47) with respect to t , we can deduce that

<!-- formula-not-decoded -->

where the last step follows from the orthogonality of Hermite polynomials:

<!-- formula-not-decoded -->

Moreover, (48) implies that

<!-- formula-not-decoded -->

Plugging into (50), we can deduce that

<!-- formula-not-decoded -->

For t ∈ [0 , A ] , the above inequality can be established in a similar fashion. As a result, we have

<!-- formula-not-decoded -->

as desired.

Our next lemma, on the other hand, develops bounds for densities outside E ˜ ϵ 1 , ˜ ϵ 2 . It essentially shows that this part makes a negligible contribution to the overall χ 2 -divergence.

Lemma D.8 (Bounding χ 2 -distance under bad event) . For any t / ∈ E ˜ ϵ 1 , ˜ ϵ 2 , we have

<!-- formula-not-decoded -->

Proof : For any t / ∈ E ˜ ϵ 1 , ˜ ϵ 2 and any x ∈ X , define

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

where (51b) follows from (45), (51c) and (51d) follow from the inequality u 2 ≥ ρ ( u + v ) 2 -ρ 1 -ρ v 2 , ∀ ρ ∈ (0 , 1) , and (51e) holds because ˜ ϵ 1 , ˜ ϵ 2 = o (1) and | t -ˆ g ( x ) | = Ω(1) by (46). With a similar reasoning, we can also deduce from (51b) that

<!-- formula-not-decoded -->

Define

<!-- formula-not-decoded -->

where we drop the dependency on ( x, t, y ) for convenience. Then from (51) we can deduce that

<!-- formula-not-decoded -->

Thus, combining the above inequality (52) we have

<!-- formula-not-decoded -->

On the other hand, from (45) it is easy to see that for any ( λ, ρ ) ∈ Λ (0) x,t,y we have

1

4

A φ

(

z

)

≤

p

λ

(

x, t, y

)

≤

3

4

A φ

(

z

)

<!-- formula-not-decoded -->

where the last inequality is due to the bound I 0 , 0 ≤ A -1 exp ( -1 4 σ 2 ( t -ˆ g ( x ) ) 2 ) , which directly follows from its definition (45) and the argument used in (51e).

Combining (53) and (54), we can deduce that as desired.

<!-- formula-not-decoded -->

Combining the results of Lemma D.7 and Lemma D.8, we obtain the following:

Corollary D.1 (Bounding the whole χ 2 -distance between the null and alternative distributions) . Let ¯ P 0 and ¯ P 1 be the mixture distributions as defined before. Then we have that

<!-- formula-not-decoded -->

In particular, if ˜ ϵ 1 ˜ ϵ 2 = o ( log -1 / 2 n ) and k n ≥ -log n log ˜ ϵ 1 , then χ 2 ( ¯ P 0 || ¯ P 1 ) = o ( ( nk 3 n ) -1 ) and χ 2 ( ¯ P ⊗ n 0 || ¯ P ⊗ n 1 ) = o ( k -3 n ) .

Proof : By definition we have

<!-- formula-not-decoded -->

If ˜ ϵ 1 ˜ ϵ 2 = o ( log 1 / 2 n ) and k n is the smallest integer satisfying k n ≥ -log n log ˜ ϵ 1 , it is easy to see that both terms in (55) are o ( ( nk 3 n ) -1 ) , so χ 2 ( ¯ P 0 || ¯ P 1 ) = o ( ( nk 3 n ) -1 ) . It follows that

<!-- formula-not-decoded -->

which concludes the proof.

□

Finally, we can apply Lemma D.1 to deduce our lower bound. We define the following functional T : for any observation distribution P ⊗ n λ,ρ of { ( x i , t i , y i ) } n i =1 generated from a model in (44), T ( P ) equals the corresponding parameter value θ λ,ρ . Let ν 0 , ν 1 be the distributions that satisfy the property in Lemma D.4 corresponding to the k n in Corollary D.1; we can also view ν 0 and ν 1 as distributions on the P ⊗ n λ,ρ 's. Note that θ λ,ρ = σ -1 ˜ ϵ 1 ˜ ϵ 2 λρ by (44), we know from Lemma D.4 that the mean difference between ν 0 and ν 1 is

<!-- formula-not-decoded -->

On the other hand, we clearly have v 0 ≤ 2 σ -1 ˜ ϵ 1 ˜ ϵ 2 , and Corollary D.1 implies that I = χ 2 ( ¯ P ⊗ n 0 || ¯ P ⊗ n 1 ) = o ( k -3 n ) . So for sufficiently large n , we have m 1 -m 0 -v 0 I ≥ 1 8 σk 3 n ˜ ϵ 1 ˜ ϵ 2 . By Lemma D.1, the minimax mean-square error for any estimator ˆ T is at least

<!-- formula-not-decoded -->

In other words, we have

<!-- formula-not-decoded -->

It remains to prove the n -1 / 2 component of the lower bound. Our proof relies on the following lemma that derives the χ 2 -divergence between two Gaussian mixtures.

Lemma D.9 ( χ 2 -distance for a specific Gaussian model) . Let P i , i = 0 , 1 be the distribution of ( X,T,Y ) generated from

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof : It is easy to see that the density of P i can be written as

<!-- formula-not-decoded -->

thus

<!-- formula-not-decoded -->

where c ( x ) is some irrelevant function of x . By taking integration, we can deduce that

<!-- formula-not-decoded -->

as desired.

The next corollary highlights the special case of Lemma D.9 that we will use in our proof:

□

Corollary D.2 (Bounding the χ 2 -distance) . In the setting of Lemma D.9, if g 0 = g 1 , q 0 = q 1 and σ | θ 1 -θ 0 | ≤ 0 . 1 , then χ 2 ( P 1 , P 0 ) ≤ 2 σ 2 ( θ 1 -θ 0 ) 2 .

Proof : By Lemma D.9, we have χ 2 ( P 1 , P 0 ) ≤ [ 1 -2 σ 2 ( θ 1 -θ 0 ) 2 ] -1 / 2 -1 , and σ | θ 1 -θ 0 | ≤ 0 . 1 implies that concluding the proof.

We now define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and let ˆ P and ˜ P be distributions of ( X,T,Y ) generated from (57) with ( g, q, θ ) = (ˆ g, ˆ q, ˆ θ ) and (ˆ g, ˆ q, ˜ θ ) respectively. Then Corollary D.2 implies that χ 2 ( ˜ P, ˆ P ) ≤ ξn -1 / 2 and thus

<!-- formula-not-decoded -->

Therefore, Lemma A.1 implies that for any estimator ˆ T , it holds that

<!-- formula-not-decoded -->

Equivalently, we have

<!-- formula-not-decoded -->

Combining (56) and (58), we obtain the desired result.

## E General upper bounds under Neyman Orthogonality

## E.1 Estimation error of general moment estimators

In this section, we establish upper bounds for general orthogonal estimators beyond DML.

To state our first result, we require several assumptions, as stated below. These assumptions largely follows [Mackey et al., 2018]. We first define the Neyman orthogonality property of moment functions.

Definition E.1 (Orthogonality of moment function) . A moment function m ( Z, θ 0 , h 0 ( X )) : R K × R × R ℓ ↦→ R d is said to be ( S 0 , S 1 ) -orthogonal for some sets S 1 ⊆ S 0 ⊆ Z ℓ ≥ 0 , if for any α ∈ S 0 , we have E P [ D α m ( Z, θ 0 , h 0 ( X )) | X ] = 0 a.s., and for any α ′ ∈ S 1 , we have D α m ( Z, θ 0 , γ ) = 0 a.s., where D α m ( Z, θ 0 , γ ) := ∇ α 1 γ 1 ∇ α 2 γ 2 · · · ∇ α ℓ γ ℓ m ( Z, θ 0 , γ ) , ∀ γ ∈ R ℓ .

This property is the key to constructing efficient structure-agnostic estimators.

Assumption E.1 (Main assumptions) . Let S 1 ⊆ S 0 be non-empty sets and k ∈ Z + , then the following conditions hold:

- (1). The moment m is ( S 0 , S 1 ) -orthogonal.
- (2). E P [ m ( Z, θ 0 , h 0 ( X ))] = 0 for all θ = θ 0 .

̸

̸

- (3). ∣ ∣ E P [ ∇ θ m ( Z, θ 0 , h 0 ( X ))] ∣ ∣ ≥ δ id and Var P ( m ( Z, θ 0 , h 0 ( X ))) ≤ V m .
- (4). D α m exists and is continuous for all ∥ α ∥ 1 ≤ k +1 .

The specific choices of S 0 , S 1 and k will be explicitly stated in all our results. In Assumption E.1, (1) requires orthogonality of the moment function, (2) guarantees that θ 0 is the unique solution to the moment equation, (3) guarantees identifiability of θ 0 , and lastly, (4) requires sufficient regularity of the moment function. Finally, we assume the following regularity conditions:

Assumption E.2 (Additional regularity assumptions) . Define B h 0 ,r = { h ∈ H : max ∥ α ∥ 1 ≤ k +1 E [ ∏ ℓ i =1 | h i ( X ) -h 0 ,i ( X ) | 2 α i ] ≤ r } . Then there exists r &gt; 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let I k,ℓ the the set of all indices α ∈ Z ℓ ≥ 0 such that ∥ α ∥ 1 ≤ k and I k,ℓ, 0 = I k,ℓ \I k -1 ,ℓ . The following theorem shows that orthogonal moments as in Definition E.1 directly yields efficient structureagnostic estimators of θ 0 .

Theorem E.1 (Structure-agnostic guarantee for general orthogonal estimators) . Let S 1 ⊆ S 0 ⊆ Z ℓ ≥ 0 , k ∈ Z + , and p, q ∈ [1 , + ∞ ] be such that p -1 + q -1 = 1 . Let P be a set of distributions of ( X,T,Y ) generated from (1) , ϵ i &gt; 0 , i = 1 , 2 , · · · , ℓ and s ≥ q max α ∈ S ∑ l i =1 α i . Further let Φ be an arbitrary mapping that maps P ∈ P to some function h 0 : R ℓ ↦→ R in some vector space F . Consider the estimate ˆ θ OML obtained by solving the moment equations

<!-- formula-not-decoded -->

Suppose that the moment function m : R K × R × R ℓ ↦→ R d satisfies Assumption E.1 with S 0 , S 1 , k specified above and additional regularity conditions (stated in Assumption E.2) for all P ∈ P . then for any γ ∈ (0 , 1) , there exists a constant C γ &gt; 0 such that

<!-- formula-not-decoded -->

with probability ≥ 1 -γ , where

<!-- formula-not-decoded -->

Additionally, when the nuisance error rates are sufficiently fast, we have the following asymptotic normality guarantee for ˆ θ :

Corollary E.1 (Asymptotic normality) . Suppose that ∏ ℓ i =1 ϵ α i i = o ( n -1 / 2 ) for all α ∈ ( I k,ℓ \ S 0 ) ∪ ( I k +1 ,ℓ, 0 \ S 1 ) , then √ n ( ˆ θ -θ 0 ) d - → N (0 , δ -2 id V m ) .

The proof can be found in Appendix E.3.

## E.2 Proofs of Theorem E.1 and Corollary E.1

The proof is based on the standard arguments for bouding estimation errors of orthogonal estimators; see e.g. [Mackey et al., 2018, Section A]. The only major difference is that our bound is structureagnostic while their goal is to establish O ( n -1 / 2 ) convergence rate under assumptions on nuisance errors. For conciseness, we will not repeat the arguments that have already been covered in their paper.

To begin with, their eq.(10) shows that

̸

̸

where J ( ˆ h ) = 1 n ∑ n i =1 m ′ θ ( Z i , ˜ θ, ˆ h ( X i )) for some θ = λθ 0 + (1 -λ ) ˆ θ, λ ∈ [0 , 1] and J = E [ m ( Z, θ 0 , h 0 ( X ))] . They also show that J ( ˆ h ) -1 I [det J ( ˆ h ) = 0] p - → J -1 . Hence

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

We then consider the decomposition of B following [Mackey et al., 2018, eq.(11)]:

<!-- formula-not-decoded -->

where we recall that I k,ℓ = { α ∈ Z ℓ ≥ 0 : ∥ α ∥ 1 ≤ k } . First, it is easy to see that with high probability, it holds that B 1 ≲ Var( m ( Z, θ 0 , h 0 ( X ))) 1 / 2 .

Second, by our assumption on the error ˆ h -h 0 , we have

<!-- formula-not-decoded -->

where the last step follows from Holder's inequality:

<!-- formula-not-decoded -->

Similarly, we have

<!-- formula-not-decoded -->

Finally, the arguments in [Mackey et al., 2018, Section A.2] imply that B 2 , B 4 = o P (1) . Combining everything above, we conclude the proof of Theorem E.1.

Under the assumptions in Corollary E.1, it holds that B 3 , B 5 = o ( n -1 / 2 ) . As a result, the same arguments in [Mackey et al., 2018, Section A.2] would imply the desired asymptotic normality result in Corollary E.1.

## E.3 Proof of Theorem 4.1: Structure-agnostic error from estimated moments

The proof follows a similar argument as the proof of the previous theorem. Consider any probability distribution P ∈ P . We define E as the 'good' event that the dataset D 1 satisfies the conditions (12), (13), (14) and (15). By assumption, we known that P [ E ] ≥ 1 -γ/ 2 . Our subsequent analysis consider a fixed choice of D 1 that fails into E . Note that the moment function ˆ m r ( · ) is partially linear in q , so for any index α = ( α 1 , α 2 ) , if α 2 ≥ 2 , then D α ˆ m r = 0 . Now let's calculate the derivative for α 2 ∈ { 0 , 1 } .

When α 2 = 0 , we have

<!-- formula-not-decoded -->

so that under E , we have

<!-- formula-not-decoded -->

for all α 1 ≤ j +1 . When α 2 = 1 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so that for all α 1 ≤ j . The above derivations also imply that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where we use the assumed property that

<!-- formula-not-decoded -->

Let J ( D 1 ) = E Z ∼ P [ m ( Z, θ 0 , h 0 ( X ); D 1 ) | D 1 ] . Similar to the proof of the previous theorem, we consider the decomposition

<!-- formula-not-decoded -->

By Chebyshev's inequality, with probability ≥ 1 -γ/ 2 , we have

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

Finally, by Taylor's formula and the orthogonality condition,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the third step uses Holder's inequality and s ≥ 2 r +2 . This yields the desired bound. The total probability for this bound to hold is (1 -γ ) 2 ≥ 1 -2 γ , as desired.

## E.4 Proof of Lemma 4.1: Explicit formula for J r

We prove this lemma by induction on r . When r = 1 , the conclusion automatically holds by the assumed expression of J 1 ( w,x ) . Now suppose that r ≥ 2 , and the conclusion holds for r -1 , then by definition,

<!-- formula-not-decoded -->

and the conclusion follows.

## F Technical details in Section 5.1

In this section, we provide additional results and details that complement Section 5.1.

We first consider the problem of estimating µ r := E [ η r ] , where r ≥ 2 is some positive integer. It turns out that even under the independence noise assumption, there exists a fundamental bottleneck for estimating µ r , as stated in the following theorem. Interestingly, this statistical limit is different for r = 3 and all other values of r .

Theorem F.1 (Structure-agnostic limit for estimating residual moments) . Let C T &gt; 0 be a constant and P 0 be the set of all distributions of ( X,T ) generated from T = g 0 ( X ) + η , such that | T | ≤ C T , a.s. and η is some mean-zero noise variable independent of X . Let Φ be a mapping that maps P 0 ∈ P 0 to the nuisance function g 0 , and the target parameter is defined by θ ( P 0 ) = E [( T -g 0 ( X )) r ] . Then for any γ ∈ (1 / 2 , 1) and r ∈ Z + , there exists a constant c γ,r &gt; 0 such that M n, 1 -γ ( P ∞ ,ϵ (ˆ g ) ) ≥ c γ ( ϵ α r + n -1 / 2 ) , where α r = 3 if r = 3 and α r = 2 otherwise. Moreover, these rates can be attained by θ = θ r , where { θ k } + ∞ k =1 is recursively defined as θ 1 = 0 , θ k = µ ′ k -kθ k -1 µ ′ 1 ( k ≥ 2) , where µ ′ k = 1 n ∑ n i =1 ( T i -ˆ g ( X i )) k .

The remaining part of this section is devoted to proving Theorem F.1 and Theorem 5.2.

## F.1 Proof of Theorem F.1: Structure-agnostic limit for estimating residual moments

The proof is based on the method of fuzzy hypothesis, as introduced in Lemma A.1. Let X ∼ P X be uniformly distributed on [0 , 1] and η be a random variable independent of X , with density

<!-- formula-not-decoded -->

where a &gt; 0 is chosen such that E [ η ] = 0 . Let M and λ = ( λ 1 , λ 2 , · · · , λ M ) where λ i 's i.i.d. random variables such that λ i = 2 with probability 1 3 and = -1 with probability 2 3 . Let B 1 , B 2 , · · · , B M be a partition of X = [0 , 1] such that P X ( B i ) = 1 M , ∀ i ∈ [ M ] . Define P λ to be the joint distribution of ( X,T ) generated from

<!-- formula-not-decoded -->

where g λ ( x ) = ˆ g ( x ) + 1 2 ϵ g ∆( λ, x ) , and ˆ P = ∫ P λ d π ( λ ) . It is easy to see that P λ ∈ Q 0 .

Let ˆ p and p λ be the density of ˆ P and ˆ P λ respectively, then the above definitions and Taylor's formula together imply that

<!-- formula-not-decoded -->

For any given X = x , E π [∆( λ, x ) i ] = 2 3 (2 i -1 + ( -1) i ) is independent of x , thus E π [ p λ ( x, t )] only depends on x, t through t -ˆ g ( x ) . As a result, we can define a random variable ˆ η which is independent of X and has density p ˆ η ( t -ˆ g ( x )) = E π [ p λ ( x, t )] . The data generating process thus induces a density p X ( x ) p ˆ η ( t -ˆ g ( x )) = E π [ p λ ( x, t )] = ˆ p ( x, t ) .

<!-- formula-not-decoded -->

We choose P = ˆ P , Q λ = P λ and Z j = B j ×T in Lemma A.2, the corresponding p j = 1 M . For any t ∈ T , x ∈ X and λ ∈ supp( π ) , we have

<!-- formula-not-decoded -->

since ˆ g is assumed to be uniformly bounded, it is easy to see that x ↦→ ∫ R Γ( t -ˆ g ( x )) d t is uniformly bounded as well. Therefore, in the setting of Lemma A.2, we have

<!-- formula-not-decoded -->

is bounded by some universal constant, which we denote by ¯ b . Let C be the constant in Lemma A.2 that corresponds to A = 1 , and choose M ≥ max { n max { 1 , ¯ b } , Cδ -1 n 2 ¯ b 2 } , then we have that

<!-- formula-not-decoded -->

The final step is to verify the separation condition (25). Specifically, we choose P = Q 0 and define T ( P ) to be -µ r for any P ∈ P . We abuse notation and use µ r ( P ) to denote the value of µ r corresponds to P ∈ P . Then we have that

<!-- formula-not-decoded -->

Note that for i ≤ r we have

In particular, we consider the case where i = 2 . If r = 3 then the above equation is nonzero, implying that

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Therefore, Lemma A.1 can be applied with s = Θ( ϵ 2 g ) , which yields the desired result.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and the conclusion can be similarly derived.

## F.2 Proofs of Theorem 5.1 and Theorem 5.2

In this subsection, we present the proofs of Theorem 5.1 and Theorem 5.2. The proof techniques are largely similar. We choose to start with the proof of Theorem 5.2, which is more involved. 5

Proof of Theorem 5.2 For any P ∈ P , let ¯ µ ′ k = E P [( T -ˆ g ( X )) k ] , then it is easy to see that

<!-- formula-not-decoded -->

By Chebyshev's inequality, we have

<!-- formula-not-decoded -->

where the last step uses | g | , | ˆ g | ≤ C g and ∥ η ∥ ψ 2 ≤ ψ η . We choose δ k = r 1 / 2 ( γn ) -1 / 2 2 2 k ( C 2 k g + k k ψ 2 k η ) 1 / 2 , then it is easy to see that with probability ≥ 1 -γ , | µ ′ k -¯ µ ′ k | ≤ δ k for all k ∈ [ r ] . Let E be the event that all these inequalities hold. The following lemma bounds the difference between θ and its population version (with µ ′ l replaced by ¯ µ ′ l ), defined as

<!-- formula-not-decoded -->

Lemma F.1 (Moment-to-cumulant type bounds) . Let the sequences { θ k } k ≥ 1 , { ¯ θ k } k ≥ 1 , { µ ′ k } k ≥ 1 and { ¯ µ ′ k } k ≥ 1 satisfy the recursions

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

5 Although Theorem 5.1 consider a more general class of noise, the rate is strictly looser in the setting of Theorem 5.2, as discussed in Remark 5.1.

Assume there exist constants C g , ψ η &gt; 0 and l, γ, n ∈ (0 , ∞ ) such that for every k ≥ 1

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof : Throughout the argument write

<!-- formula-not-decoded -->

(70)-(71) imply that | ¯ µ ′ k | ≤ A k and | µ ′ k -¯ µ ′ k | ≤ D k . By Triangle inequality, (68) implies that

<!-- formula-not-decoded -->

Let ¯ θ k = A k ρ k , then this becomes

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now prove that

Indeed, since we can deduce that

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

proving (75). Plugging into (74) and rearranging, we obtain

Define

Then we have that

Moreover,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, for all k ≥ 1 ,

so it follows from (77) that S k ≤ 2 k . Finally, we have

<!-- formula-not-decoded -->

We now turn to bound ¯ θ k -θ k . Set ∆ k := ¯ θ k -θ k . Subtracting (68) from (69) gives

<!-- formula-not-decoded -->

Taking absolute values and invoking the bounds already proved yields

<!-- formula-not-decoded -->

Let ∆ k = A k δ k , then (75) and (78) implies that

<!-- formula-not-decoded -->

Define

Then we have that

<!-- formula-not-decoded -->

which further implies

Moreover,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so it follows from (80) that T r ≤ 3 r +1 r 1 / 2 ( γn ) -1 / 2 . Therefore, | ∆ r | = A r | δ r | ≤ r r/ 2 T r A r ≤ r r/ 2 3 r +1 r 1 / 2 ( γn ) -1 / 2 A r , concluding the proof. □

In view of the previous lemma, we only need to bound the difference between ¯ θ l and κ l . Note the following well-known property of cumulants { κ l } + ∞ l =1 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let ˆ η = T -ˆ g ( X ) and D X = g 0 ( X ) -ˆ g ( X ) , then ˆ η = η + D X and η, D X are independent by our assumption. By definition, ¯ θ l is the l -th order cumulant of T -ˆ g ( X ) . Hence,

<!-- formula-not-decoded -->

where d l is the l -th order cumulant of the variable D X . From Proposition A.2 we can deduce that d r ≤ (2 rϵ ) r . combining with Lemma F.1, the conclusion follows.

Proof of Theorem 5.1 For any P ∈ P , let ¯ µ ′ k = E P [( T -ˆ g ( X )) k ] , then it is easy to see that

<!-- formula-not-decoded -->

By Chebyshev's inequality, we have

<!-- formula-not-decoded -->

We choose δ k = r 1 / 2 ( γn ) -1 / 2 (2 C T ) k , then it is easy to see that with probability ≥ 1 -γ , | µ ′ k -¯ µ ′ k | ≤ δ k for all k ∈ [ l ] . Let E be the event that all these inequalities hold. The following lemma bounds the difference between θ and its population version (with µ ′ l replaced by ¯ µ ′ l ), defined as

<!-- formula-not-decoded -->

Lemma F.2 (Moment-to-cumulant type bounds) . Let the sequences { θ k } k ≥ 1 , { ¯ θ k } k ≥ 1 , { µ ′ k } k ≥ 1 and { ¯ µ ′ k } k ≥ 1 satisfy the recursions

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assume there exist constants C g , ψ η &gt; 0 and l, γ, n ∈ (0 , ∞ ) such that for every k ≥ 1

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof : By Triangle inequality, (68) implies that

<!-- formula-not-decoded -->

Let ¯ θ k = (2 C T ) k ρ k , then this becomes

<!-- formula-not-decoded -->

Then, for all k ≥ 1 ,

Since it is easy to prove by induction that

so that

<!-- formula-not-decoded -->

We now turn to bound ¯ θ k -θ k . Set ∆ k := ¯ θ k -θ k . Subtracting (68) from (69) gives

<!-- formula-not-decoded -->

Let D k = r 1 / 2 ( γn ) -1 / 2 (2 C T ) k . Taking absolute values and invoking the bounds already proved yields

<!-- formula-not-decoded -->

Let ∆ k = D k δ k , then (75) and (78) implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We also have that so by induction we can deduce that | δ k | ≤ 10( k -1)! , and

concluding the proof.

<!-- formula-not-decoded -->

In view of the previous lemma, we only need to bound the difference between ¯ θ l and κ l . Note the following well-known property of cumulants { κ l } + ∞ l =1 :

<!-- formula-not-decoded -->

Let ˆ η = T -ˆ g ( X ) and D X = g 0 ( X ) -ˆ g ( X ) , then ˆ η = η + D X and η, D X are independent by our assumption. By definition, ¯ θ l is the l -th order cumulant of T -ˆ g ( X ) . Hence,

<!-- formula-not-decoded -->

where d l is the l -th order cumulant of the variable D X , and we have that d l ≤ l ! E [ | g 0 ( X ) -ˆ g ( X ) | l ] ≤ l ! ϵ l , and the conclusion follows.

<!-- formula-not-decoded -->

## F.3 Relaxing the independent noise assumption

In this subsection, we consider a case where the noise η is almost independent of X and show that our estimator is still better than the plug-in estimator.

Proposition F.1 (Finite-sample accuracy of two cubic-moment estimators) . Let T = g 0 ( X ) + η with η = ϵ 0 η 0 + η 1 and assume

<!-- formula-not-decoded -->

- (ii) η 1 ⊥ ⊥ X (while making no restriction on the joint law of η 0 and X );
- (iii) η 0 , η 1 have finite third moments;
- (iv) an estimator ˆ g satisfies δ ( X ) := ˆ g 0 ( X ) -g 0 ( X ) with ∥ δ ∥ L 3 ( P X ) ≤ ϵ ;

<!-- formula-not-decoded -->

Given i.i.d. samples { ( X i , T i ) } n i =1 , put Z i := T i -ˆ g 0 ( X i ) and define

<!-- formula-not-decoded -->

For any 0 &lt; δ &lt; 1 , with probability at least 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where C &gt; 0 is an absolute constant (e.g. C = 10 ). Consequently, if ϵ 0 → 0 and ϵ → 0 while n → ∞ , the plug-in estimator ˆ ψ n is o ( ϵ ) -biased and O p ( n -1 / 2 ) -consistent, whereas the naive estimator ˆ ν n keeps a leading Θ( ϵ ) bias whenever σ 2 1 &gt; 0 .

Proof : With Z = T -ˆ g 0 ( X ) = η -δ and µ k := E [ Z k ] , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The first bracket in each line is the bias , the second the sampling error .

Now write η = ϵ 0 η 0 + η 1 , δ = δ ( X ) . Expanding moments and using η 1 ⊥ ⊥ X ,

<!-- formula-not-decoded -->

H¨ older yields | E [ δ ] | ≤ ϵ , E [ δ 2 ] ≤ ϵ 2 , | E [ δ 3 ] | ≤ ϵ 3 , while Cauchy-Schwarz gives | E [ η 0 δ 2 ] | ≤ σ 0 ϵ 2 . Substituting into (94)-(95) gives the bias terms displayed in (92)-(93).

Define ∆ k := ˆ µ k,n -µ k . By Bernstein's inequality for centred variables with sixth moment M 6 ,

<!-- formula-not-decoded -->

Taking t k := M k √ (2 log(6 /δ )) /n and a union bound over k ensures | ∆ k | ≤ t k with probability 1 -δ . Using | µ 1 | ≤ M , | µ 2 | ≤ M 2 and (94),

<!-- formula-not-decoded -->

An analogous bound holds for | ˆ ν n -µ 3 | . Combining with the bias bounds completes (92)-(93). □

## G Technical details in Section 5.2

In this section, we provide detailed proofs of results in Section 5.2. We let P m be the set of all possible partitions of the integer m , i.e., the set of all multisets of positive integers that sum to m ( e.g. , for m = 4 the possible partitions are (4) , (1 , 3) , (2 , 2) , (1 , 1 , 2) , (1 , 1 , 1 , 1) ), and P m,j be the set of partitions with j terms. We define p ( m ) = | P m | and p ( m,j ) = | P m,j | respectively. Note that P m is different from Π m defined in Section 5, which is the number of partitions of [ m ] into distinct subsets.

Proposition G.1 (Partition number bound) . We have p ( m ) ≤ 2 m for all m ≥ 1 .

Proof : Consider placing numbers 1 , 2 , · · · , m in a row and delimiters are placed between some consecutive numbers. Clearly, the total number of ways to place the delimiters is 2 m -1 . For each partition m = i 1 + · · · + i j , there exists at least one way of placing the delimiters, such that their induced partition of { 1 , 2 , · · · , m } contains subsets of sizes i 1 , · · · , i m . This creates an injective mapping from P m to the set of possible delimiters, implying that p ( m ) ≤ 2 m -1 . □

## G.1 Proof of Lemma 5.1: Key lemma; higher-order insensitivity condition

First by definition, E [ ˆ J ( k ) r ( T -g 0 ( X )) ] = ∑ r i = k i ! ( i -k )! ˆ a i +1 ,r µ i -k where µ i -k = ∑ π ′ ∈ Π i -k ∏ B ′ ∈ π ′ κ | B ′ | is the ( i -k ) -th moment of η . It suffices to show that

<!-- formula-not-decoded -->

To establish this, we note that the corresponding summands on each side are of the form ( -1) q κ i 1 · · · κ i p ˆ κ j 1 · · · ˆ κ j q . Fix i 1 , · · · , i p , j 1 , · · · , j q such that ∑ p s =1 i s + ∑ q t =1 j t = r -k , and let i = k + ∑ p s =1 i s ≤ j . Let N m, { α s } s 0 s =1 be the number of ways to partition [ m ] into subsets of sizes { α s } s 0 s =1 where m = ∑ s 0 s =1 α s . Then the coefficient of the term ( -1) q κ i 1 · · · κ i p ˆ κ j 1 · · · ˆ κ j q on the right-hand side (RHS) is ( r -k r -i ) N i -k, { i s } p s =1 N r -i, { j t } q t =1 . However, the left-hand side has precisely the same coefficient, because there exist ( r -k r -i ) ways to partition [ r -k ] into two subsets with sizes r -i and i -k respectively and inside each subset the number of partitions with desired subset sizes are N i -k, { i s } p s =1 and N r -i, { j t } q t =1 respectively.

## G.2 Proof of Theorem 5.4: ACE estimation error: sub-Gaussian noise

Lemma G.1 (Log inequalities) . Let a, b &gt; 0 .

<!-- formula-not-decoded -->

- (2) Assume, in addition, that ab ≥ e (so log( ab ) ≥ 1 ). Then for every 0 &lt; x ≤ b a log( ab ) we have x log( ax ) ≤ b.

Proof : Proof of Item (1). Define g ( x ) ≜ ax +log x -b, x &gt; 0 . Because g ′ ( x ) = a +1 /x &gt; 0 , the map g is strictly increasing. Put

<!-- formula-not-decoded -->

We show g ( x 0 ) ≤ 0 . Set y ≜ b a ; then

<!-- formula-not-decoded -->

Since y -1 a log y &lt; y , the argument of the second logarithm is smaller than y , hence log ( y -1 a log y ) &lt; log y and g ( x 0 ) &lt; 0 . Because g is increasing, x ≤ x 0 implies g ( x ) ≤ g ( x 0 ) ≤ 0 , i.e. ax +log x ≤ b .

Proof of Item (2). Define h ( x ) ≜ x log( ax ) -b, x &gt; 0 . We have h ′ ( x ) = log( ax ) + 1 , so h is strictly increasing for x ≥ e -1 /a . The equation h ( x ) = 0 has the unique positive root

<!-- formula-not-decoded -->

where W is the LambertW function (the solution of z = W ( z ) e W ( z ) ). When ab ≥ e we have W ( ab ) ≥ log( ab ) (standard lower bound for W on [ e, ∞ ) ), hence

<!-- formula-not-decoded -->

Thus every x ≤ b/ ( a log( ab )) satisfies x ≤ x ⋆ and, by monotonicity of h , h ( x ) ≤ h ( x ⋆ ) = 0 ; that is x log( ax ) ≤ b . □

Lemma G.2 (Condition for bias domination) . Let

<!-- formula-not-decoded -->

Let a ≜ 2 log ( 6( C g + ψ η ) ϵ -1 1 ) , b ≜ log( γn/ 9) with a, b &gt; 0 . If

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

i.e. the bias term (2 iϵ 1 ) i dominates the variance term in Lemma G.2.

Proof : Note that is equivalent to

<!-- formula-not-decoded -->

We can apply Lemma G.1 with a = 2log ( 6( C g + ψ η ) ϵ -1 1 ) and b = log( γn/ 9) to obtain the desired conclusion. □

Let

<!-- formula-not-decoded -->

be the exact version of ˆ J r . We first derive some bounds for the coefficients of J r and ˆ J r .

Lemma G.3 (Bounding polynomial coefficients) . For any i ∈ [ r +1] we have

<!-- formula-not-decoded -->

then for every 1 ≤ i ≤ r

<!-- formula-not-decoded -->

Proof : By definition, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (100) follows from triangle inequality, (101) follows from the cumulant bound in Proposition A.2, (103) rearranges the summation term according to the number of subsets in the partition π ∈ Π r +1 -i , (104) follows from and (105) follows from Proposition G.1.

Lemma G.4 (Bounding the estimation error of polynomial coefficients) . For any i ∈ [ r + 1] we have

<!-- formula-not-decoded -->

Proof : By definition, we have

<!-- formula-not-decoded -->

as desired. Here in the last but one step, we use the fact that

<!-- formula-not-decoded -->

and the last step follows from Proposition G.1.

Lemma G.5 (Key lemma; approximate orthogonality) . Let

<!-- formula-not-decoded -->

where a ≜ 2 log ( 6( C g + ψ η ) ϵ -1 1 ) , b ≜ log( γn/ 9) as in Lemma G.2, then under E we have E [ ˆ J ( k ) r ( T -g 0 ( X )) ] ≤ (4 eϵ 1 ) r -k .

Proof : For any k ∈ [ r ] we have that

<!-- formula-not-decoded -->

By Lemma G.2, we have κ i ˆ κ i 2(2 iϵ 1 ) i for all i [ r ] . Then for any 1 j r k , we have

<!-- formula-not-decoded -->

(108) | -| ≤ ∈ ≤ ≤ -

Plugging into (108), we have

<!-- formula-not-decoded -->

Lemma G.6 (Identifiability coefficient) . E [( T -g 0 ( X )) J r ( T -g 0 ( X ))] = 1 r ! κ r +1 .

Proof : By definition, we have

<!-- formula-not-decoded -->

Consider any partition π ∈ Π i . Without loss of generality, assume that 1 ∈ B 1 and | B 1 | = k, k ≤ i . There are a total of ( i -1 k -1 ) possible choices of B 1 , and the remaining sets form a partition of [ i -k ] . Hence we can write

<!-- formula-not-decoded -->

Plugging into (109), the right-hand side

<!-- formula-not-decoded -->

concluding the proof.

Lemma G.7 (Linear-inε 1 moment difference) . Let

<!-- formula-not-decoded -->

Assume an estimate ˆ g satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and fix an integer 1 ≤ i ≤ s/ 2 . Then

Proof : Write δ ( X ) ≜ g 0 ( X ) -ˆ g ( X ) so that T -g 0 ( X ) = η and T -ˆ g ( X ) = η + δ ( X ) . Binomial expansion gives

<!-- formula-not-decoded -->

Taking absolute values, expectations, and using independence ( η ⊥ ⊥ X ) together with H¨ older,

<!-- formula-not-decoded -->

Since η is sub-Gaussian with η ψ 2 = ψ η , we have E η j (2 ψ η √ j ) j (2 ψ η √ i 1) j , so that

<!-- formula-not-decoded -->

□

<!-- formula-not-decoded -->

Lemma G.8 (Identifiability guarantee) . We have

<!-- formula-not-decoded -->

Proof : By assumption (2) in Theorem 5.4 and Lemma G.6, we know that

<!-- formula-not-decoded -->

By Lemma G.3, we have | a ir | ≤ 1 ( i -1)! (8 ψ η ) r +1 -i . Moreover, by Lemma G.7, for i ∈ [ r + 1] it holds that

Hence, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (111) follows from triangle inequality, (112) follows from Lemma G.3, Lemma G.4, Lemma G.7 and (66). From our assumption (21) and Lemma G.1, we can deduce that this quantity is smaller than 1 2 r ! δ id , concluding the proof. □

Remark G.1. With a similar reasoning, one can also deduce that

<!-- formula-not-decoded -->

This inequality will be used later in the proof.

Lemma G.9 (Second-order moment bounds) . The following inequalities hold:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof : Recall that

<!-- formula-not-decoded -->

by Lemma G.3, and

<!-- formula-not-decoded -->

by (66), we can deduce that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (118) follows from (116), (119) follows from (117), (120) follows from ( i + j 2 ) i + j ≤ i i j j which is a direct consequence of Jensen's inequality, (103) follows from i ! ≥ i i/ 2 . This concludes the proof of (114).

With a similar reasoning, we can deduce that

<!-- formula-not-decoded -->

Since ξ is ψ ξ -sub-Gaussian, we have

<!-- formula-not-decoded -->

so that

<!-- formula-not-decoded -->

By definition, we have for any P ∈ P that

<!-- formula-not-decoded -->

□

Since m ( Z, θ, h ( X )) with h = ( g, q ) can be viewed as an ( l +2) -th order polynomial of g and q , we have

<!-- formula-not-decoded -->

where in the last but one step we use the fact that | θ 0 | ≤ C θ . By (114) and Chebyshev inequality, we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

with probability ≥ 1 -γ . Our assumption on l , Lemma G.8, and (125) together imply that

<!-- formula-not-decoded -->

It also directly follows from (126) that

<!-- formula-not-decoded -->

By triangle inequality,

<!-- formula-not-decoded -->

where the penultimate inequality follows from E P [ ˆ J r ( T -ˆ g ( X )) 2 ] ≤ 3 [ 8( C g + ψ η ) ] r ϵ 1 which can be shown in a similar fashion as Lemma G.9, and the final inequality follows from (113). Hence we can deduce that

<!-- formula-not-decoded -->

where (128) follows from Lemma G.6 and (129) uses the constraint on r given in (21). Recall that ˆ θ satisfies 2 n ∑ n i = n/ 2+1 m ( Z i , θ, ˆ h ( X i )) = 0 , so (127) and (129) together imply that

<!-- formula-not-decoded -->

As a result, (125) and (126) yield

<!-- formula-not-decoded -->

Subtracting this inequality from (124), we obtain

<!-- formula-not-decoded -->

Therefore, we conclude that

<!-- formula-not-decoded -->

## G.3 Proof of Theorem 5.3: ACE estimation error

In this section, we outline how the proof of Theorem 5.4 in the previous section can be slightly modified to obtain Theorem 5.3.

Lemma G.10 (Condition for bias domination) . Suppose

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then for every 1 ≤ i ≤ r i.e. the bias term (2 iϵ 1 ) i dominates the variance term in Lemma G.10.

Lemma G.11 (Bounding polynomial coefficients) . For any i ∈ [ r +1] we have

<!-- formula-not-decoded -->

Proof : The proof is the same as that of Lemma G.3.

□

Lemma G.12 (Bounding the estimation error of polynomial coefficients) . For any i ∈ [ r +1] we have

<!-- formula-not-decoded -->

Proof : The proof is the same as that of Lemma G.4.

□

Lemma G.13 (Key lemma; approximate orthogonality) . If r satisfies Lemma G.10, then under E (defined in the proof of Theorem 5.1 in Appendix F .2) we have

<!-- formula-not-decoded -->

Proof : The proof is the same as that of Lemma G.5.

Lemma G.14 (Linear-inϵ 1 moment difference) . Let

<!-- formula-not-decoded -->

Assume an estimate ˆ g satisfies

<!-- formula-not-decoded -->

and fix an integer 1 ≤ i ≤ s/ 2 . Then

<!-- formula-not-decoded -->

Proof : The proof is similar to Lemma G.7; the only difference is that the moment bound of η becomes E | η | j ≤ C j T . □

Lemma G.15 (Identifiability guarantee) . We have

<!-- formula-not-decoded -->

Proof : Similar to the proof of Lemma G.8, the left-hand-side can be shown to be ≤ 3(2 C T ) r ϵ 1 . Combining with the constraint (19) yields the desired conclusion. □

Lemma G.16 (Second-order moment bounds) . The following inequalities hold:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equipped with the above lemmas, we can then follow the arguments in Appendix G.2 to deduce that

<!-- formula-not-decoded -->

## G.4 Special cases of the ACE estimator

When r = 3 , Theorem 5.4 immediately implies the following result:

Corollary G.1 (Third-order ACE estimator) . Let δ id &gt; 0 and C θ , C g , C q , ψ η , ψ ξ ≥ 1 be constants and P 0 be the set of all distributions in P ( C θ , C g , C q ; ψ ξ , ψ η ) such that η is independent of X and | κ 4 | ≥ δ id . Suppose that θ is the solution to (59) with m ( Z, θ, h ( X )) = [ Y -q ( X ) -θ ( T -g ( X ))] [ ( T -g ( X )) 3 -3 µ ′ 2 ( T -g ( X )) -( µ ′ 3 -3 µ ′ 1 µ ′ 2 ) ] . Then for any γ ∈ (0 , 1) , there exists a constant C γ such that

<!-- formula-not-decoded -->

□

The choice of the moment function in Corollary G.1 has also been proposed in Mackey et al. [2018], though their results are restricted to the high-dimensional linear regression setting. However, the rate that we derive from Corollary G.1 is faster than theirs, and as a consequence, in Corollary G.3 we need a weaker sparsity assumption to achieve O ( n -1 / 2 ) rate. The main insight for deriving this improved rate is that the moment function is, in fact, third-order orthogonal. By contrast Mackey et al. [2018] only shows that it is second-order orthogonal. We will revisit this setting in Section 6, where we empirically verify the effectiveness of ACE for different choices of r .

For r ≥ 4 , to the best of our knowledge, the estimators derived from Theorem 5.4 are new. For illustration purpose, we derive the guarantee for r = 4 in the following:

Corollary G.2 (Fourth-order ACE estimator) . Let δ id &gt; 0 and C θ , C g , C q , ψ η , ψ ξ ≥ 1 be constants and P 0 be the set of all distributions in P ( C θ , C g , C q ; ψ ξ , ψ η ) such that η is independent of X and | κ 5 | ≥ δ id . Suppose that θ is the solution to (59) with

Then for any γ ∈ (0 , 1) , there exists a constant C γ such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark G.2 (Cumulants versus moments) . Generalizing the r = 3 case to r ≥ 4 is highly nontrivial. Indeed, given the construction in Corollary G.1, one might be tempted to consider

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with h = ( g, q ) . In this case, let ∆( x ) = g ( x ) -ˆ g ( x ) , then we have which is the same as the r = 3 case up to constants. As a result, this approach does not yield rates faster than Corollary G.1.

In Corollaries G.1 and G.2, we omit the constants in the upper bounds. As shown in Theorem 5.4, the constants for r -th order orthogonal estimators can be at most ( Cr ) r for some constant C , that grows super-exponentially, and, as demonstrated in Remark 5.4, the growth of this constant is in some cases offset by the growth of the absolute cumulant | κ r +1 | .

## G.5 ACE estimation error for high-dimensional sparse linear regression

Corollary G.3 (ACE estimation error for high-dimensional sparse linear regression) . In the setting of Theorem 5.4 with high-dimensional linear nuisance (24) , suppose that the nuisance estimators (ˆ g, ˆ q ) are respectively constructed via Lasso regression of T and Y onto X with an appropriately chosen regularization parameter. If max( s 1 , s r/ ( r +1) 1 s 1 / ( r +1) 2 ) = o ( n r/ ( r +1) / log p ) , then, with probability ≥ 1 -γ , we have | ˆ θ -θ 0 | ≤ C γ C r δ -1 id ,r n -1 / 2 for constants C γ and C r depending only on γ and r respectively.

Proof : As derived in Mackey et al. [2018, Sec. I], there exists some constant C ′ γ &gt; 0 such that on an event E with probability ≥ 1 -γ/ 2 , the Lasso nuisance estimates simultaneously provide the bounds

<!-- formula-not-decoded -->

where we recall that β 0 , α 0 ∈ R p . Since X ∼ N (0 , I ) , Cauchy-Schwarz and Khintchine's inequality [Vershynin, 2018, Corollary 2.6.4] imply that, on this event,

<!-- formula-not-decoded -->

<!-- image -->

- (a) The average MSE of ACE estimation with order l = 1 , 2 , 5 as functions of ξ .
- (b) The distribution of estimates for second- and fifth-order ACE estimation, with varying ξ .
- (a) The average MSE of ACE estimator with l = 1 , 2 , · · · , 6 as functions of sample size.
- (b) The distribution of estimates for second-order ACE estimation when n = 10000 , s = 200 .
- (c) The distribution of estimates for fifth-order ACE estimation when n = 10000 , s = 200 .

<!-- image -->

Figure 2: The sensitivity of ACE estimators to correlation of the covariate X and the noise variable η .

<!-- image -->

<!-- image -->

<!-- image -->

Figure 3: Experiment results for ACE estimators with fixed sample size n = 10000 and varying sparsity.

for some universal constant K &gt; 0 . Our assumption that

<!-- formula-not-decoded -->

further implies that, on the event E , max { ϵ r +1 1 , ϵ r 1 ϵ 2 } = o ( n -1 / 2 ) . Applying Theorem 5.4 with γ replaced by γ/ 2 , we can therefore conclude that the advertised bound holds with probability (1 -γ/ 2) 2 ≥ 1 -γ .

□

## H Additional experiment results and discussion

In view of Theorem 5.4, the error rate of r -th order ACE estimator depends on a bias term that scales as O ( ϵ r 1 ϵ 2 + ϵ r +1 1 ) and a variance term that scales as O ( n -1 / 2 ) , multiplied by a constant depending on r . In practice, this constant is often non-negligible. Hence, to choose an appropriate order r , one should take into consideration its effect on the final estimation error.

Varying sample size. We first investigate the performance of ACE estimators with r ≤ 6 with varying sample size. The results are reported in Figure 2. We set r = 1 as a baseline and only plot the results of estimators that are better than r = 1 . From Figure 1 one can see that the fifth order estimator performs the best, followed by the second order one. The fifth order estimator incurs large errors for small sample sizes ( n = 2000 ) but the error decreases rapidly when n grows larger.

When n ≥ 8000 , the decreasing rates of different estimators are roughly the same. This is because in this regime, variance becomes the dominating term in the total mean-squared error. From a theoretical perspective, this is because for fixed s , the LASSO estimates induce errors that scale as n -1 / 2 , so for any r ≥ 1 , the bias term becomes n -( r +1) / 2 ≪ n 1 / 2 . In Figure 1d and Figure 1e we plot the distributions of the estimates produced by first and fifth order ACE estimators. One can see that the variances of these two estimators are roughly at the same level, while the first-order estimator induces significantly larger biases, especially when n is small.

In Figure 1f, we plot the 95% confidence intervals for each individual estimates. We observe that the actual percentage of confidence intervals covering the ground-truth parameter is 94 . 9% , quite close to what Corollary E.1 predicts.

Correlation between covariate and treatment noise. The theoretical benefits of ACE estimators with r ≥ 3 crucially relies on the assumption that X and η are independent, which might be restrictive. However, it might be the case that they are weakly correlated, i.e. , only a small part of η is correlated with X . We would like to understand the sensitivity of our estimators' performance with respect to such correlation.

Specifically, we assume that the treatment variable is drawn from

<!-- formula-not-decoded -->

where X 1 is the first component of X and η is a mean-zero random variable independent of X . We set p = 100 , n = 20000 , s = 40 investigate the estimation error of ACE with different r 's as a function of the correlation coefficient ξ .

The results are reported in Figure 2, where we compare the top3 estimators among r = 1 , 2 , · · · , 6 . The first- and second-order estimators have stable performance across different correlations, while the performance of the fifth-order one deteriorates rapidly when ξ ≥ 0 . 1 . This suggests that it would be better to use the second-order estimator unless one has strong prior knowledge that X and η are weakly correlated. In the context of pricing experiments, the data is drawn from a company's historical experimentation records, so that scientists are likely to have knowledge about the highlevel design principles of such experiments.

Varying sparsity. Lastly, we investigate the relative performance of ACE estimators with different level of sparsity, for fixed p = 1000 , n = 10000 . Recall that sparsity affects the first-stage nuisance errors, which in turn affects the bias of our estimates.

The results are reported in Figure 3. From Figure 3a we can see that the performance of first-order estimator deteriorates rapidly when the support size grows. By contrast, the performances of secondand fifth-order estimators are quite stable, with the fifth-order one slightly better for smaller s . This is not surprising, since one can see from Theorem 5.4 that the bias term for the fifth-order estimator has a larger bias, so it would only be smaller than the second-order counterpart when the nuisance errors are small enough. As shown in Figures 3b and 3c, when s = 200 , the fifth-order estimator indeed incurs a larger bias.

## I More results for orthogonal machine learning

## I.1 Construction of orthogonal moment functions

Theorem I.1 (Construction of higher-order orthogonal moments) . Let a ik ( x ) and ρ ik ( w ) be as defined in Lemma 4.1, then the following statements hold:

- (1). If E [(1 + | η | ) | ρ ik ( η ) | | X = x ] &lt; + ∞ holds for all 1 ≤ i ≤ M r and x ∈ X , then the moment function

<!-- formula-not-decoded -->

with nuisance functions h 1 = g, h 2 = q and h i +2 = a ik , i ∈ [ M r ] is ( S 0 , S 1 ) -orthogonal where S 0 contains all α ∈ Z ℓ ≥ 0 that satisfies at least one of the following conditions:

(i). ∥ α ∥ 1 ≤ 1 ; (ii). α 1 + α 2 ≤ k ; (iii). α 4 = 1 , ( α 1 , α 2 ) ∈ { (0 , 1) , (1 , 0) } , the remaining α i 's equal zero;

and S 1 = { α ∈ Z ℓ ≥ 0 : max { α 2 , ∑ M r +2 i =3 α i } ≥ 2 } . Moreover, D α m exists and is continuous for all ∥ α ∥ 1 ≤ k +1 .

- (2). Let C θ , C T , C Y &gt; 0 be constants and P = P ⋆ b ( C θ , C T , C Y ) . Let θ be the solution of (59) with m = m r . Then under Assumption E.1 (2) and (3), for any γ ∈ (0 , 1) there exists a

so that

For any α ∈ Z M r +2 ≥ 0 with ∥ α ∥ 1 = 1 , we consider three cases, constant C γ &gt; 0 that only depends on γ, C θ , C T , C Y , such that

<!-- formula-not-decoded -->

̸

holds with probability ≥ 1 -γ , where ˜ ϵ = max i =1 , 2 , 4 ϵ i , s ≥ k +1 and

<!-- formula-not-decoded -->

Theorem I.1 is proven in Appendix I.2. It shows that by successive integration of J 0 , we can construct moment functions that are orthogonal with respect to g 0 ( X ) and q 0 ( X ) with arbitrarily high order, while being only first-order orthogonal with respect to the nuisances a ik ( X ) . In (3), the constants δ id and V m depend on the order k ; we will write them as δ id ,k and V m ,k to avoid confusion.

From another perspective, Theorem I.1 can be viewed as a special case of Theorem 4.1, because the construction of the moment function there ensures that E P [ ∑ M j i =1 a ij ( X ) ρ ij ( T -g 0 ( X )) | X = x ] = 0 , j = 1 , 2 , · · · , k . In Theorem I.1, the a ij ( · ) 's are viewed as nuisance functions which allows for exact orthogonalily properties. By contrast, in Theorem 4.1 only q ( · ) and g ( · ) are treated as nuisance functions and we only ask for approximate orthogonality. While they look similar at first glance, an important observation is that this result does not rely on any explicit assumptions on the estimation errors of ˆ a ij ( · ) 's. Indeed, it might be possible that the left-hand side of (15) is much smaller than the individual estimation errors of ˆ a ij ( · ) 's, because these individual errors cancel out in the summation. This observation will prove helpful in Section 5.

Recall that in Theorem 3.2 we show that higher-order orthogonality is impossible for the Gaussian treatment, even with known variance. In this case, since the distribution of η = T -g 0 ( X ) | X is known, the functions a ik 's are known as well. However, in the Gaussian case, any moment function constructed from (8) when k ≥ 2 would violate the identifiability assumption (Assumption E.1 (3)). We prove this in Proposition I.2.

## I.2 Proof of Theorem I.1

In this subsection, we provide a straightforward instantiation of Theorem E.1 using the moment functions constructed in Lemma 4.1. For simplicity, we restrict ourselfs to the case where the treatment and outcome are both bounded.

Proof of (1). The statement follows directly from induction. By assumption, the conclusion holds for k = 0 . Now assume that it holds for some k -1 ≥ 0 , then

<!-- formula-not-decoded -->

so we can choose M r = M r -1 + 1 , a ir ( x ) = a i -1 ,r -1 ( x ) , ρ ir ( w ) = ∫ w 0 ρ i -1 ,r -1 ( w ′ ) d w ′ , 2 ≤ i ≤ M r -1 and a 1 ,r ( x ) = -E [ I r ( T -g 0 ( X ) , X ) | X = x ] , ρ 1 ,r ( w ) ≡ w , proving the result for r .

Proof of (2). First note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- α 1 = 1 , then (8) implies that

<!-- formula-not-decoded -->

- α 2 = 1 , then E [ D α m ( Z, θ 0 , h 0 ( X ))] = θ 0 E [ J r ( T -g 0 ( X ) , X )] -E [( Y -q 0 ( X ) -θ 0 ( T -g 0 ( X ))) J r -1 ( T -g 0 ( X ) , X )] = 0 .
- α i +2 = 1 for some i ≥ 1 , then E [ D α m ( Z, θ 0 , h 0 ( X ))] = E [( Y -q 0 ( X ) -θ 0 ( T -g 0 ( X ))) ρ ir ( T -g 0 ( X ))] = 0 .

Next, we consider α 's of form ( α 1 , α 2 , 0 , · · · , 0) where α 1 + α 2 ≤ r . Since m is affine in q , when α 1 ≥ 2 we have D α m ( Z, θ 0 , h 0 ( X )) = 0 , so we only need to consider the case when α 1 ∈ { 0 , 1 } . Similar to the arguments above,

- If α 1 = 1 , α 2 ≤ r -1 then

<!-- formula-not-decoded -->

- If α 1 = 0 , α 2 ≤ r then

<!-- formula-not-decoded -->

Furthermore,

- If α 1 = 1 , α 4 = 1 , ρ 4 ,r ( w ) ≡ w and the remaining α j 's are all zero, then

<!-- formula-not-decoded -->

- If α 1 = 1 , α 4 = 1 , ρ 4 ,r ( w ) ≡ w and the remaining α j 's are all zero, then

<!-- formula-not-decoded -->

This proves the orthogonality properties related to S 0 . Since m ( Z, θ, γ ) is affine in γ i , i ≥ 2 (which corresponds to the nuisance functions q and a ir , i ∈ [ M r ] ), we have D α m ≡ 0 as long as ∑ i ≥ 2 α i ≥ 2 ⇔ α ∈ S 1 . Hence m is ( S 0 , S 1 ) -orthogonal as desired.

Finally, the continuity of D α m is obvious since m is a quadratic function in terms of the nuisance functions.

Proof of (3). By Theorem E.1,

<!-- formula-not-decoded -->

As shown in part (2), I r,ℓ \ S 0 only contains α with α 2 ∈ { 0 , 1 } , α 1 + α 2 ≥ 1 , min { α 1 , α 2 } + α 4 ≥ 3 and ∑ i ≥ 3 α i = 1 . Thus

<!-- formula-not-decoded -->

On the other hand, I r +1 ,ℓ, 0 \ S 1 contains α with α 2 ≤ 1 and ∑ i ≥ 3 α 1 ≤ 1 , so that

<!-- formula-not-decoded -->

Combining the above two inequalities, the conclusion follows.

## I.3 Example: heteroscedastic nonparametric regression

In general, a ik ( · ) may be hard to estimate since it is a linear combination of conditional moment functions. Generally speaking, there is no guarantee that estimating these conditional moment functions is easier than estimating the nuisance functions.

In this section, we revisit the nonparametric regression problem with heteroscedastic noise, where fast rates for estimating a ik 's are indeed achievable. Specifically, suppose that the treatment variable is sampled from the regression model

<!-- formula-not-decoded -->

where the noise variable η = V 0 ( X ) 1 / 2 η ⋆ and η ⋆ satisfies | η ⋆ | ≤ C η ⋆ a.s., E [ η ⋆ | X ] = 0 and E [ η ⋆ 2 | X ] = 1 . For this problem, the following result is known from Wang et al. [2008].

Proposition I.1. [Wang et al., 2008, Theorems 1,2 and Remark 3] Assuming that X = [0 , 1] and g 0 ( · ) , V 0 ( · ) are α and β -th order smooth respectively, then given i.i.d. data { ( X i , T i ) } n i =1 from some distribution such that the marginal density of X exists and is bounded away from 0 , there exists an estimator ˆ V ( · ) that achieves the optimal mean-square error rate ∥ ˆ V ( X ) -V 0 ( X ) ∥ P, 2 = O P ( n -min { 2 α,β/ (2 β +1) } ) .

In particular, when β &gt; α , one can in fact estimate V 0 ( · ) with higher accuracy than estimating g 0 ( · ) . In this subsection, we additionally assume that the distribution of η ⋆ is known and let µ ⋆ r ( x ) = E [ η ⋆r | X = x ] be its r -th moment. Thus we can estimate µ r ( x ) = E [ η r | x = x ] = V 0 ( x ) r/ 2 µ ⋆ r ( x ) with ˆ V ( x ) r/ 2 µ ⋆ r ( x ) . We conjecture that by using a similar approach as Wang et al. [2008] one can directly construct higher-order moment estimates for unknown η ⋆ , so the assumption that η ⋆ is known can be removed.

To state our main result for this setting, we assume that g 0 and V 0 are α and β -th order smooth respectively:

Assumption I.1. g ∈ Λ α ( M g ) and V ∈ Λ β ( M V ) for some constants M g , M V &gt; 0 and α, β &gt; 0 .

Let ˆ g be an optimal estimate of g 0 under the L ∞ -norm, that achieve the rate ϵ 1 = O ( (log n/n ) α/ (2 α +1) ) [Stone, 1982]. Also let ˆ V be the estimate of V 0 in Wang et al. [2008]

<!-- formula-not-decoded -->

that achieves the L 2 -rate

Consider the moment function m r defined in (10). Its nuisance functions a ik ( x ) , i = 0 , 1 , · · · , k are functions of the conditional moments of η | X . Then we can derive their estimates ˆ a ik ( x ) by directly plugging in the variance estimates: ˆ µ r ( x ) = ˆ V ( x ) r/ 2 µ ⋆ r ( x ) . The next theorem provides theoretical guarantee for the resulting estimate derived from this approach:

Theorem I.2 (Error rate for heteroscedastic nonparametric regression) . Let X = [0 , 1] , C T ≥ 1 be a real number and P be the set of all distributions in P ⋆ that satisfy Assumption I.1 and | T | ≤ C T . Let ˆ g, ˆ V be defined above and ˆ q be some estimator of q 0 such that ∥ ˆ q -q 0 ∥ L 2 ( P ) ≤ ϵ 2 . Also assume that | ˆ µ r ( x ) | ≤ (2 C T ) r 6 . Then the following statements hold:

- (1). Consider the nuisance functions and their estimates specified in the previous paragraph. Then for all i ∈ [ r +1] , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (2). Let ˆ θ OML be the solution to (59) with the moment function m = m r defined in (10) . Then in the setting of Theorem I.1, for any γ ∈ (0 , 1) , there exists a constant C γ &gt; 0 such that

<!-- formula-not-decoded -->

6 This is without loss of generality, since otherwise we can replace ˆ µ r with min { (2 C T ) r , max {-(2 C T ) r , ˆ µ r }} . By assumption we know that | µ r | ≤ (2 C T ) r , so this would only reduce the estimation error.

Theorem I.2 is proven in Appendix I.4.1. Note that Theorem I.2 makes structural assumptions on g 0 , V 0 but not on q 0 . As a result, the assumption is stronger than the fully structure-agnostic setting, while being weaker than the Holder-smoothness setting. Even in this interpolated regime, to the best of our knowledge, there is no existing results that achieve faster rates than DML.

It is worth noticing that the rate ϵ v can be faster than ϵ 1 to arbitrary order. Thus, there exists an optimal k that balances the dependency on n and the magnitude of the constants A r , δ id ,r , V m ,r , λ ⋆ . For r = 2 , 3 , assuming that these constants are uniformly bounded and that q also belongs to a Holder class, we can derive the following result:

Corollary I.1. In the setting of Theorem I.2, if we additionally assume that q ∈ Λ γ ( M q ) and ∥ ˆ q -q 0 ∥ L 2 ( P X ) ≤ ϵ 2 = O ( n -2 γ/ (2 γ +1) ) , then the following holds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Corollary I.1 implies that O P ( n -1 / 2 ) rate can be achieved under weaker smoothness requirements if one uses third-order orthogonal estimators. Its proof can be found in Appendix I.5.

## I.4 Technical details in Appendix I.3

## I.4.1 Proof of Theorem I.2

Proof of part (1). By definition, we have

<!-- formula-not-decoded -->

while ˆ κ i 's are cumulant estimates obtained by directly plugging in ˆ V ( · ) :

<!-- formula-not-decoded -->

By the mean value theorem, there exists some ˜ V ( · ) between ˆ V and V 0 such that ˆ V j/ 2 ( x ) -V j/ 2 0 ( x ) = j 2 ˜ V j/ 2 -1 ( x )( ˆ V ( x ) -V 0 ( x )) . Hence

<!-- formula-not-decoded -->

where we recall that C v is the assumed uniform upper bound on | V 0 ( X ) | and | ˆ V ( x ) | . We can then bound the estimation error of ˆ µ r ( · ) as follows:

<!-- formula-not-decoded -->

Via a similar reasoning as in Lemma F.1, we can deduce by induction that

<!-- formula-not-decoded -->

which then implies an upper bound on the error of ˆ a ik ( · ) :

<!-- formula-not-decoded -->

where we follow the arguments employed in Lemma G.4.

Proof of part (2). This is a direct consequence of Theorem I.1 and part (1).

## I.5 Proof of Corollary I.1

Let ρ = min { α, γ } &gt; 1 4 , then we have ∥ ˆ g -g 0 ∥ P 0 , ∞ , ∥ ˆ q -q 0 ∥ P 0 , ∞ = ˜ O P ( n -ρ 2 ρ +1 ) , so that ϵ 1 , ϵ 2 = ˜ O P ( n -ρ 2 ρ +1 ) = o P ( n -1 / 6 ) .

Also Proposition I.1 implies that

<!-- formula-not-decoded -->

It remains to show that

<!-- formula-not-decoded -->

Since 2 α &gt; 1 2 , it remains to check that

<!-- formula-not-decoded -->

which holds by assumption. This proves (1).

To prove (2), we use a similar argument except that the upper bound becomes

<!-- formula-not-decoded -->

Since ρ ≥ √ 3 -1 4 &gt; 1 6 , we have max { ϵ 1 , ϵ 2 } 4 = O ( n -4 ρ 2 ρ +1 ) = O ( n -1 / 2 ) . Moreover, the assumption guarantees that

<!-- formula-not-decoded -->

since √ 3 -1 4 is the positive root of the equation ρ 2 ρ +1 +2 ρ = 1 2 . This concludes the proof.

## I.6 Comparison with Mackey et al. [2018]

In Mackey et al. [2018] the authors consider polynomial-based moment functions. These constructions can be derived from our Theorem I.1 by choosing J 1 ( w,x ) = w k -µ k ( x ) where k is some positive integer and µ k ( x ) = E [ η k | X = x ] . For r = 2 , 3 , we obtain the following special cases:

Example I.1. By choosing r = 2 , we recover the moment function

<!-- formula-not-decoded -->

proposed by Mackey et al. [2018]. Thus, Theorem I.1 implies that this moment function satisfies all conditions in Assumption E.1 with the orthogonality set S = {∥ α ∥ 1 ≤ 2 }\{ (1 , 0 , 0 , 1) , (0 , 1 , 0 , 1) } .

Example I.2. By choosing r = 3 , we obtain

<!-- formula-not-decoded -->

with nuisance functions h ( X ) = ( q ( X ) , g ( X ) , µ 2 ( X ) , µ k ( X ) , µ k +1 ( X ) , µ k +2 ( X )) ∈ R 6 is orthogonal with respect to S = { ( a, b, 0 , 0 , 0 , 0) | a + b = 3 } ∪ { α | ∥ α ∥ 1 ≤ 2 } \ { ( a, b, c, d, 0 , e ) ∈ Z 6 ≥ 0 | a + b = c + d + e = 1 } .

## I.7 More discussion on Assumption E.1

The following result states that if the distribution of η | X = x does not depend on x and its density has certain good properties, then Assumption E.1 would not be violated with J 2 , unless η is Gaussian.

Proposition I.2 (Identifiability v.s. orthogonality) . The moment function m in Theorem I.1 (2) satisfies Assumption E.1 if and only if

<!-- formula-not-decoded -->

Moreover, the following statements hold:

- (1). If η | X = x is Gaussian for all x ∈ X , then E [( T -g 0 ( X )) J r ( T -g 0 ( X ) , X )] = 0 , ∀ r ≥ 2 .
- (2). If η | X = x is non-Gaussian with twice continuously differentiable density p ( · ) that does not depend on x , then there exists J 1 ( w,x ) in the form of (7) , such that E [( T -g 0 ( X )) J 3 ( T -g 0 ( X ) , X )] = 0 and E [(1 + | η | ) | ρ i 2 ( η ) | ] &lt; + ∞ , ∀ 1 ≤ i ≤ M 2 .

̸

- (3). Suppose that η ⊥ ⊥ X and η is non-Gaussian, and let { J r } + ∞ r =1 be the sequence generated from J 1 ( w,x ) = w (assuming that all of them are well-defined). Then there exists r ≥ 2 such that E [( T -g 0 ( X )) J r ( T -g 0 ( X ) , X )] = 0 .

̸

The first part of Proposition I.2 can be directly derived from the Stein's Lemma, while the second part is derived from a characterization of the solutions property of the Gauss-Airy's equation xy -ay ′ -by ′′ = 0 . [Durugo, 2014, Ansari, 2016] Generalizing this result to k ≥ 3 requires characterizing the solution properties of higher-order Gauss-Airy's equation, which we leave for future work.

Proof : (1) is straightforward from Stein's lemma. To prove (2), orthogonality implies that

<!-- formula-not-decoded -->

and

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we slightly abuse notation and use J ′ 3 , J ′′ 3 to represent the partial derivatives with respect to the first argument.

Note that m is partially linear in θ , we have

<!-- formula-not-decoded -->

so (2) and (3) are both equivalent to

<!-- formula-not-decoded -->

̸

̸

We argue that there must exist some J 3 ( w,x ) that is polynomial in w , such that the equations (146), (147) and (148) hold simultaneously. Since by assumption, the density p ( η ) of η = T -g 0 ( X ) | X does not depend on X , (147a) is equivalent to

<!-- formula-not-decoded -->

and similarly, (147b) is equivalent to

<!-- formula-not-decoded -->

Lemma I.1 at the end of this subsection implies that in L 2 ( R ) , wp ( w ) / ∈ span ⟨ p ′ ( w ) , p ′′ ( w ) ⟩ , so by Lemma I.2, there must exists some function ˜ J 3 ( w ) ∈ C 3 ( R ) such that

<!-- formula-not-decoded -->

Moreover, we can assume WLOG that ∫ ˜ J 3 ( w ) p ( w ) d w = 0 , since replacing ˜ J 3 with ˜ J 2 -c 0 , ∀ c 0 ∈ R does not affect the properties in (149). We define ˜ J 2 ( w ) = ˜ J ′ 3 ( w ) and ˜ J 1 ( w ) = ˜ J ′′ 3 ( w ) . In the following, we show that J 1 ( w,x ) = ˜ J 1 ( w ) satisfies all the desired properties. First, since ˜ J 3 is a polynomial, so is ˜ J 1 . Second, by (147b) we have

<!-- formula-not-decoded -->

Finally, recall that p ( · ) is the probability density function of η = T -g 0 ( X ) , so

<!-- formula-not-decoded -->

concluding the proof of (2).

Finally, under the conditions of (3), Lemma G.6 implies that E [( T -g 0 ( X )) J r ( T -g 0 ( X ) , X )] = 1 r ! κ r +1 , where κ i is the i -th order cumulant of η . We know from Levy's Inversion Formula [Durrett, 2019, Theorem 3.3.11] that non-Gaussian distributions must have at least one non-zero cumulant, so the conclusion immediately follows. □

Lemma I.1 (Solution of second-order Airy equation) . Let p ( · ) be the probability density function of a random variable η that is second-order continuous differentiable, and that

<!-- formula-not-decoded -->

for some a, b ∈ R , then we must have b = 0 , a &gt; 0 and thus η must be Gaussian.

Proof : Since p ( · ) is a density function, the Riemann-Lebesgue lemma implies that its Fourier transform ˆ p ( ξ ) = ∫ R e -ixξ p ( x ) d x must vanish at infinity. On the other hand, applying Fourier transform to both sides of (150) yields

<!-- formula-not-decoded -->

̸

Thus we must have a &gt; 0 . If b = 0 , [Ansari, 2016, 4.2] then implies that there exists constants c 1 , c 2 ∈ R such that p ( x ) has the same sign as the Airy function Ai( c 1 x + c 2 ) . However, it is well-known that Ai( x ) can take both positive and negative values, which is a contradiction. □

Lemma I.2 (Separability of inner products) . Suppose that f i ( w ) , i = 0 , 1 , 2 are continuous functions such that f 0 / ∈ span ⟨ f 1 , f 2 ⟩ . Then there exists a function J ( w ) ∈ C 3 ( R ) such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

Proof : Suppose that such J ( w ) does not exist. For any finite interval [ a, b ] and a sequence of C 3 functions S = { g j , j = 1 , 2 , · · · , n } supported on [ a, b ] (we denote the set of such functions by C 3 0 ([ a, b ]) ), we define the vector u S i = (∫ g r ( w ) f i ( w ) d w ) + ∞ k =0 , i = 0 , 1 , 2 . Then for any λ ∈ R | S | , by choosing J ( w ) = ∑ n i =1 λ i g i ( w ) , our assumption implies that

<!-- formula-not-decoded -->

Let v S 0 be the orthogonal projection of u S 0 onto span ⟨ u S 1 , u S 2 ⟩ and λ = u S 0 -v S 0 , then λ ⊤ u S 1 = λ ⊤ u S 2 = 0 and λ ⊤ u S 0 = ∥ λ ∥ 2 , so that λ = 0 and u S 0 ∈ span ⟨ u S 1 , u S 2 ⟩ .

We consider the following two cases:

1. For any finite S , rank ( span ⟨ u S 1 , u S 2 ⟩ ) ≤ 1 , then u S 0 is parallel to u S 1 for all S . Since u 1 = 0 , it is easy to show that u 0 = αu 1 for some α ∈ R . As a result, we have ∫ J ( w )( f 0 ( w ) -αf 1 ( w )) d w = 0 for any J ( · ) ∈ C 3 0 ([ a, b ]) , so we must have f 0 -αf 1 ≡ 0 on [ a, b ] .

̸

2. There exists some finite S such that rankspan ⟨ u S 1 , u S 2 ⟩ = 2 , then there exists unique α, β ∈ R such that u S 0 = αu S 1 + βu S 2 . By considering any set S 1 = S ∪{ s } , s / ∈ S , one can show that u S 1 0 = αu S 1 1 + βu S 1 2 as well, so u S 0 = αu S 1 + βu S 2 for any finite set S . As a result, we have ∫ J ( w )( f 0 ( w ) -αf 1 ( w ) -βf 2 ( w )) d w = 0 for any J ( · ) ∈ C 3 0 ([ a, b ]) , so we must have f 0 -αf 1 -βf 2 ≡ 0 on [ a, b ] .

Now we have shown that for any interval [ a, b ] , there exists α, β ∈ R such that f 0 -αf 1 -βf 2 ≡ 0 on [ a, b ] . It is easy to derive from this fact that f 0 -αf 1 -βf 2 ≡ 0 on R , i.e. , f 0 ∈ span ⟨ f 1 , f 2 ⟩ , which is a contradiction. Hence the conclusion follows. □