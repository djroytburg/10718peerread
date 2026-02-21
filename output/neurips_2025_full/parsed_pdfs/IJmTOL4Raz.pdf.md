## Stochastic Gradients under Nuisances

Facheng Yu

## ∗ Ronak Mehta Alex Luedtke Zaid Harchaoui

University of Washington

## Abstract

Stochastic gradient optimization is the dominant learning paradigm for a variety of scenarios, from classical supervised learning to modern self-supervised learning. We consider stochastic gradient algorithms for learning problems whose objectives rely on unknown nuisance parameters, and establish non-asymptotic convergence guarantees. Our results show that, while the presence of a nuisance can alter the optimum and upset the optimization trajectory, the classical stochastic gradient algorithm may still converge under appropriate conditions, such as Neyman orthogonality. Moreover, even when Neyman orthogonality is not satisfied, we show that an algorithm variant with approximately orthogonalized updates (with an approximately orthogonalized gradient oracle) may achieve similar convergence rates. Examples from orthogonal statistical learning/double machine learning and causal inference are discussed.

## 1 Introduction

Machine learning, statistics, and causal inference rely on risk minimization problems of the form

<!-- formula-not-decoded -->

where Θ ⊆ R d is a parameter space, Z is a Z -valued random variable, and ℓ 0 : Θ ×Z → R is a loss function. The quantity ℓ 0 ( θ ; z ) describes the performance of a model parametrized by θ ∈ Θ on a test example z ∈ Z . Given only an oracle that provides a stochastic gradient estimate of the objective (1), practitioners are able to train models ranging from linear functions on tabular data to billion-parameter neural networks on vision and language data.

The success of stochastic gradient descent (SGD) algorithms [Amari, 1993, Bottou and Le Cun, 2005, Bottou and Bousquet, 2007, Ward et al., 2020] has motivated an abundance of work on their theoretical properties under various algorithmic and risk conditions, such as class separability [Soudry et al., 2018], random reshuffling [G¨ urb¨ uzbalaban et al., 2021], decomposable objectives [Schmidt et al., 2017, Vaswani et al., 2019], quantization noise [Gorbunov et al., 2020], and noise dominance [Sclocchi and Wyart, 2024]. This success has been fueled by machine learning and AI software libraries such as JAX, PyTorch, TensorFlow, and others, which offer a wide range of SGD variants, as long as a loss function can be clearly specified. The gradient is then evaluated automatically on a mini-batch of datapoints and used for stochastic updates.

Though powerful, this recipe takes one thing for granted: that the learner can always compute the risk (or an unbiased estimate thereof). Indeed, many complex learning problems rely on a risk function that is only partially specified up to a class

<!-- formula-not-decoded -->

where G is a possibly infinite-dimensional set and L : Θ ×G → R is a function of both the target parameter θ ∈ Θ and an unknown nuisance parameter g ∈ G .

∗ email: fachengy@uw.edu .

This framework originates from semiparametric estimation and inference [Levit, 1979, Linnik, 2008, Bickel et al., 1993, Van der Vaart, 2000], where the risk is a Kullback-Leibler (KL) divergence and g provides information about the true data-generating distribution P , but is not of primary scientific interest. However, the partially specified loss formulation from (2) is not limited to semiparametric estimation and inference problems. This framework connects to many areas of interest, including profile likelihood based learning [Murphy and and, 2000, Pavlichin et al., 2019, Hao and Orlitsky, 2019] and distributionally robust learning [Shapiro, 2017, Levy et al., 2020, Mehta et al., 2024].

For instance, profile likelihood based learning reduces (2) by applying a pointwise minimum over g ∈ G to then construct a problem that can be solved in θ ∈ Θ . Another example arises in applications with distribution shifts, for which g represents an unknown test data distribution that may differ from the one from which the training data were drawn. Distributionally robust learning reduces (2) by instead taking a pointwise maximum over G and solving the resulting problem. Although the pointwise minimum and maximum are natural reductions, it is often the case that there is a 'true' g 0 ∈ G and the loss class is reduced by first estimating g 0 with auxiliary data to produce some ˆ g , which we refer to as double/debiased machine learning , or DML, following Chernozhukov et al. [2018a]. The problem (1) is then thought to be derived via L 0 ( θ ) ≡ L ( θ, g 0 ) in this case (see examples in Sec. 2). This is the focus of this paper.

Despite the prominence of SGD and DML individually, the convergence guarantees of SGD to recover the risk minimizer with a misspecified nuisance parameter remain unknown. Indeed, after producing ˆ g , the user typically solves a (full batch) empirical risk minimization problem, i.e. minimizing a sample average approximation of L ( · , ˆ g ) . In this paper, we aim to fill this gap by proving convergence guarantees on the sequence ( θ ( n ) ) n ≥ 1 generated by updates of the form

<!-- formula-not-decoded -->

where η &gt; 0 is a learning rate, D n := ( Z i ) n i =1 is a stream of independent data drawn from P , ˆ g is a nuisance parameter estimate, and S : Θ ×G × Z → R d is a stochastic gradient oracle satisfying of E Z ∼ P [ S ( θ, g ; Z )] = ∇ θ L ( θ, g ) for all ( θ, g ) ∈ Θ ×G . In particular, when G lies within a Banach space equipped with norm ∥·∥ G , we wish to compare θ ( n ) to

<!-- formula-not-decoded -->

given conditions on the degree of misspecification ∥ ˆ g -g 0 ∥ G and (approximate) Neyman orthogonality of the risk L [Neyman, 1959].

Intuitively, Neyman orthogonal classes of objectives are instances of (2) whose curvature with respect to θ is insensitive to the choice of g (see Sec. 2 for the formal description). When Neyman orthogonality is satisfied, the double machine learning framework is also known as orthogonal statistical learning (OSL) [Zadik et al., 2018, Liu et al., 2022, Foster and Syrgkanis, 2023]. In addition to the obvious computational considerations, we argue that the SGD perspective in this paper also sheds light on the methodological opportunities in DML/OSL. Indeed, while loss functions are typically specified by the chosen architecture, Neyman orthogonality is often achieved by specialized analytic calculations on the part of the user. Although this property is generally seen as a secondorder property of the loss, it can also be viewed as a first-order property of the gradient oracle S . As we detail in Sec. 3, it may be easier and more aligned with the spirit of modern machine learning, to craft Neyman orthogonal gradient oracles instead of losses.

Contributions. We prove the first theoretical convergence guarantees for SGD under an unknown nuisance model. We find that θ ( n ) converges linearly to a neighborhood of θ ⋆ -the optimum in the well-specified case-with a radius that has a fourth-power (resp. squared) dependence on ∥ ˆ g -g 0 ∥ G when Neyman orthogonality is (resp. is not) satisfied. Our analysis can also apply to two-stream settings in which the nuisance parameter is learned online alongside the target. We further analyze a new algorithm, called orthogonalized SGD (OSGD), wherein the gradient oracle of a possibly non-orthogonal loss can be iteratively made orthogonal using an 'approximately orthogonalized' gradient oracle, which is based on a separate estimation procedure. This algorithm enjoys a convergence guarantee that interpolates between the ∥ ˆ g -g 0 ∥ 4 G (nuisance insensitive) and ∥ ˆ g -g 0 ∥ 2 G (nuisance sensitive) regimes depending on the quality of the orthogonalizing operator.

We provide an introduction to the OSL/DML setting in Sec. 2. The SGD and OSGD algorithms are described and analyzed in Sec. 3. We discuss related work in Sec. 4; additional discussion can be found in Appx. F. All proofs and numerical illustrations can be found in the Appendix.

## 2 Orthogonal Statistical Learning

We first introduce various examples of risk functions in the form of (2), then formally introduce Neyman orthogonality and its implications. As is common in learning settings, the risk will be in the form of an expectation,

<!-- formula-not-decoded -->

where ℓ : Θ × G × Z → R is an instance-level loss function. Various assumptions used in the analysis in Sec. 3 (e.g. convexity) may be placed on either the loss ℓ or the risk L . In each example, we provide the structure of the data point Z , the set G , and the loss ℓ , and the true g 0 ∈ G to fully specify the problem. Here, we interpret 'true' to mean that g 0 is a parameter of the data-generating distribution (e.g. a propensity score in causal inference), or that g 0 satisfies a cost-minimizing or utility-maximizing criterion (as in the profile likelihood or distributional robustness examples from Sec. 1).

Table 1: Examples of Neyman Orthogonal Losses.

| Example   | ℓ ( θ, g ; z )                                                                                                       | g 0                                                                           |
|-----------|----------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| PLM       | 1 2 ( y - g Y ( w ) -⟨ θ,x - g X ( w ) ⟩ ) 2                                                                         | ( E P [ Y &#124; W ] , E P [ X &#124; W ])                                    |
| CATE      | 1 2 ( g (1) ( x ) - g (0) ( x )+ ( w - g prop ( x ))( y - g ( w ) ( x )) g prop ( x )(1 - g prop ( x )) -⟨ θ,x ⟩ ) 2 | ( E P [ Y &#124; W = 1 ,X ] , E P [ X &#124; W = 0 ,X ] , E P [ W &#124; X ]) |
| CRR       | - [ µ (1) g ( z ) log p θ ( x )+ µ (0) g ( z ) log(1 - p θ ( x )) ]                                                  | ( E P [ Y &#124; W = 1 ,X ] , E P [ X &#124; W = 0 ,X ] , E P [ W &#124; X ]) |

Example 1 (Partially Linear Model) . Let Z = ( X,Y,W ) ∼ P , where X is an R d -valued input, Y is a real-valued outcome, and W is a W -valued control or confounder. The space G is a nonparametric class containing functions of the form

<!-- formula-not-decoded -->

Following the construction of Robinson [1988], this g is supplied to the loss

<!-- formula-not-decoded -->

To ensure θ ⋆ can be interpreted via the projection of E P [ Y | X,W ] onto partially linear additive functions, the true nuisance is given by g 0 = ( g 0 ,X , g 0 ,Y ) , where

<!-- formula-not-decoded -->

The next example concerns a quantity widely studied in causal inference [Kennedy, 2023].

Example 2 (Conditional Average Treatment Effect) . We observe Z = ( X,Y,W ) ∼ P , where W is a binary treatment assignment. The functions in G are of the form

<!-- formula-not-decoded -->

and are evaluated (see van der Laan and Luedtke [2014, Thm. 1]) at the loss

<!-- formula-not-decoded -->

For g 0 = ( g (0) 0 , g (1) 0 , g prop 0 ) nuisance functions g (0) 0 and g (1) 0 represent the outcome regressions

<!-- formula-not-decoded -->

whereas g prop 0 ( x ) := E P [ W | X = x ] denotes the propensity score. The minimizer θ ⋆ indexes a projection of the conditional average treatment effect g (1) 0 -g (0) 0 onto linear functions.

Finally, we maintain the data structure from the previous example, but consider a loss corresponding to a different target parameter according to van der Laan et al. [2024].

Example 3 (Conditional Relative Risk) . We retain all components of the previous example, changing only the loss and assuming that the outcome Y is binary/non-negative. First, consider the 'label' function

<!-- formula-not-decoded -->

where 1 ( · ) denotes the indicator function, and the logit-linear predictor p θ ( x ) = e ⟨ θ,x ⟩ / (1 + e ⟨ θ,x ⟩ ) . To obtain a linear approximation of the log-relative risk log( g (1) 0 /g (0) 0 ) , we employ the cross entropy-type loss function

<!-- formula-not-decoded -->

While the choices of the loss function in Examples 2 and 3 might look opaque to readers outside of causal inference and statistics, they are carefully designed to be Neyman orthogonal . To motivate its definition, notice that, invariably, g 0 is unknown to the user. In DML, the user may produce or access some ˆ g ∈ G , which is an estimate of g 0 based on independent training data other than the stream ( Z i ) n i =1 used to produce θ ( n ) . It is of clear interest how stochastic optimization algorithms (and their resulting minimizers) behave in light of the misspecification of g 0 , and what precise theoretical conditions govern this behavior. Moreover, as we demonstrate in Sec. 3, these same conditions can be used to analyze procedures for which the user may access additional data to progressively improve the estimate ˆ g and learn θ ⋆ simultaneously. We now formally introduce Neyman orthogonality, and by extension, the orthogonal statistical learning (OSL) variant of DML.

Neyman Orthogonality. For a definition that accounts for a possibly infinite-dimensional function class G , we introduce the directional derivative , or equivalently, the derivative operator .

Definition 1 (Derivative Operator) . For a functional F mapping from a vector space F to R , we define the (directional) derivative operator D as D F ( f )[ h ] := d d t F ( f + th ) | t =0 for any f, h ∈ F . For a vector-valued F : F ↦→ R d , this derivative operator can be generalized by taking derivatives coordinate-wise. We define the second-order derivative as D 2 F ( f )[ h, h ′ ] := D(D F ( f )[ h ])[ h ′ ] for h, h ′ ∈ F and higher-order derivatives similarly. For functionals of multiple variables F : F × G → R , we use the subscript notation D f F ( f, g )[ h ] to indicate the directional derivative of f ↦→ F ( f, g ) with g ∈ G fixed.

We denote by S θ ( θ, g ; z ) = ∇ θ ℓ ( θ, g ; z ) the gradient of the loss function w.r.t. the target parameter θ ∈ Θ . Borrowing terminology from statistics, we call this the score , whether ℓ is based on a likelihood or not. 2 This constitutes one particular example of a stochastic gradient oracle S used in (3). Overloading notation, the population gradient oracle is defined as S θ ( θ, g ) = E Z ∼ P [ ∇ θ ℓ ( θ, g ; Z )] .

Definition 2 (Neyman Orthogonality) . The population gradient oracle S θ is Neyman orthogonal at ( θ ⋆ , g 0 ) over G ′ ⊆ G if

<!-- formula-not-decoded -->

For Θ ′ ⊆ Θ , the population loss L is Neyman orthogonal at ( θ ⋆ , g 0 ) over Θ ′ ×G ′ if

<!-- formula-not-decoded -->

2 This notion of (Fisher) score differs from the 'score' used in score-based generative modeling [Song et al., 2021]. If ℓ is based on a log-likelihood, then S θ is the gradient w.r.t. the parameter θ ∈ Θ , not the input z ∈ Z .

Figure 1: Illustration of Neyman Orthogonalization. The first two panels are contour plots of the risk function L ( θ, g ) , where θ varies on the x -axis and g varies on the y -axis. For the orthogonalized risk (center) the contours are approximately axis-aligned. The right plot shows the cross sections of the non-orthogonal risk when fixing g = g 0 , g 1 , g 2 . Due to non-orthogonality, the minimizers θ 1 and θ 2 shown in the first and third plots may drift significantly from θ ⋆ . In contrast, the minimizers in the center plot are less sensitive to the choice of g .

<!-- image -->

In Definition 2, we allow Θ ′ ×G ′ ⊆ Θ ×G to be a proper subset, which not only provides a weaker condition, but also accounts for localization-style arguments. Moreover, since D θ L ( θ ⋆ , g 0 )[ θ -θ ⋆ ] = ⟨ S θ ( θ ⋆ , g 0 ) , θ -θ ⋆ ⟩ , if the population risk L satisfies (5), then (6) holds for any target parameter class Θ ′ ⊆ R d . As mentioned above, the risk functions in Examples 1-3 are all Neyman orthogonal at their respective value of ( θ ⋆ , g 0 ) . In the next section, we will discuss a procedure to make a non-orthogonal gradient oracle 'approximately' orthogonal. We illustrate the intended outcome intuitively in Fig. 1 by comparing a generic loss and its orthogonalized counterpart.

## 3 Stochastic Gradient Optimization

In this section, we propose two stochastic gradient algorithms, which rely on different choices of the stochastic gradient oracle S used in (3). The first is the familiar stochastic gradient oracle that provides a sample estimate of the gradient ∇ L ( · , ˆ g ) for a fixed estimate ˆ g . The second employs an approximately orthogonalized gradient oracle , or OSGD oracle, to achieve a notion of approximate Neyman orthogonality (in a manner we make precise in this section). We analyze the first algorithm under both non-orthogonal and orthogonal settings, achieving an illustrative breakdown of 'nuisance sensitive' and 'nuisance insensitive' regimes regarding the theoretical convergence guarantee. For the OSGD algorithm, we prove a convergence guarantee that interpolates between the two regimes, depending on the accuracy of the oracle.

Notation and Assumptions. For readers' convenience, a table of all the notation we introduce throughout the paper is collected in Appx. A. We maintain the prototypical bias/variance conditions on the stochastic gradient oracle S , that is, an unbiasedness condition and a second-moment growth condition (see Asm. 3(d)). To dispel confusion, note that by 'unbiased', we mean specifically that E Z ∼ P [ S ( θ, g ; Z )] = ∇ θ L ( θ, g ) for all ( θ, g ) , as opposed to the 'bias' of replacing ˆ g with g 0 , a terminology sometimes used in DML/OSL. Our analysis will rely partly on the initial distance r of the nuisance estimate ˆ g to g 0 in G , which defines the ball

<!-- formula-not-decoded -->

Various assumptions on the risk will be required to hold only locally, that is, within G r ( g 0 ) as opposed to the entire linear space G . Thus, the assumptions become weaker as the estimate ˆ g improves.

## Assumption 3. The following conditions hold:

- (a) Differentiability: For any ( z, g ) ∈ Z × G , θ ↦→ ℓ ( θ, g ; z ) is twice continuously differentiable. For any ( θ, g ) , ( ¯ θ, ¯ g ) ∈ Θ × G , (i) D 2 g D θ ℓ ( ¯ θ, ¯ g ; z )[ θ -θ ⋆ , g -g 0 , g -g 0 ] exists and is continuous, (ii) D θ L ( ¯ θ, ¯ g )[ θ -θ ⋆ ] and D g L ( ¯ θ, ¯ g )[ g -g 0 ] exist,

<!-- formula-not-decoded -->

- (b) First-order optimality: The pair ( θ ⋆ , g 0 ) satisfies S θ ( θ ⋆ , g 0 ) = 0 .
- (c) Smoothness and strong convexity: There exist constants M ≥ µ &gt; 0 such that for all g ∈ G r ( g 0 ) , the population risk L ( · , g ) is M -smooth and µ -strongly convex for θ ∈ Θ .
- (d) Second-moment growth: There exist constants K 1 , κ 1 ≥ 0 such that

<!-- formula-not-decoded -->

- (e) Second-order smoothness: There exists a constant α 1 ≥ 0 such that

<!-- formula-not-decoded -->

Asm. 3 does not require Neyman orthogonality at ( θ ⋆ , g 0 ) . Instead, Asm. 3(a) is a standard differentiability condition. Asm. 3(b) and (c) implies that θ ⋆ is a unique global minimizer. Asm. 3(d) generalizes the uniformly bounded second moment condition in stochastic optimization (e.g. Cutler et al. [2023]) by adding a quadratic form in θ , which allows us to consider an unbounded feasible set Θ and more loss classes. Finally, Asm. 3(d) and (e) can be satisfied when the Hessian of the population risk is a bounded operator. Usually, K 1 , κ 1 , and α 1 would depend on the initial nuisance estimation distance r . We provide in Appx. B estimates of the constants in Asm. 3 for each motivating example. We proceed to the main results regarding the convergence of SGD and OSGD.

Stochastic Gradient Algorithm. Here, we use the standard single-sample stochastic gradient estimate S = S θ in (3). This leads to the update

<!-- formula-not-decoded -->

While the SGD procedure can be easily extended to using a batch of unbiased gradient estimates, we keep our single-observation construction to highlight the most important aspects of the analysis. In order to achieve quantitative guarantees in the Neyman orthogonal setting, which essentially removes certain second-order terms that include θ and g , we will consider the following higherorder condition in some cases.

Assumption 4 (Higher-Order Smoothness) . The risk L satisfies Definition 2 at ( θ ⋆ , g 0 ) , and there exists some constant β 1 &gt; 0 such that

<!-- formula-not-decoded -->

When satisfied, Asm. 4 results in the nuisance insensitivity alluded to at the beginning of this section. Notice that Neyman orthogonality is not necessary to construct a stochastic optimizer, and it is still possible to obtain a nuisance sensitive rate under only Asm. 3. We demonstrate this in Thm. 1.

Theorem 1. Define D n = ( Z 1 , . . . , Z n ) , sampled from the product measure P n . Suppose that Asm. 3 holds, ˆ g ∈ G r ( g 0 ) is estimated independently of D n , and θ (0) , . . . , θ ( n ) ∈ Θ almost surely. The iterates of (8) satisfy:

1. Nuisance sensitive: If η ≤ µ/ 2( Mµ + κ 1 ) , then

<!-- formula-not-decoded -->

2. Nuisance insensitive: If Asm. 4 also holds, then, for η ≤ µ/ 2( Mµ + κ 1 ) ,

<!-- formula-not-decoded -->

Note that the assumption that the iterates remain in Θ is satisfied in common cases. It is satisfied trivially for the first two examples in Sec. 2 because Θ = R d . Another case is when the loss decomposes into the sum of a G -Lipschitz continuous component and the ℓ 2 2 -norm regularizer, i.e.

ℓ ( θ, g ; z ) = h ( θ, g ; z ) + µ 2 ∥ θ ∥ 2 2 . Then, the iterates and the optimum remain in { θ : ∥ θ ∥ 2 ≤ G/µ } (see, e.g., Mehta et al. [2023, Appx. C]), so Definition 2 can be restricted to this compact set.

Thm. 1 states that SGD converges linearly to a ball around θ ⋆ with a radius that depends on the bias (due to the replacement of g 0 with ˆ g ) and the variance due to gradient noise. Moreover, the variance component decays proportionally to the learning rate η . Under Asm. 4, the bias component can have a significantly more favorable scaling with the error in the nuisance estimate ∥ ˆ g -g 0 ∥ G -specifically, ∥ ˆ g -g 0 ∥ 4 G instead of ∥ ˆ g -g 0 ∥ 2 G . A similar breakdown into two regimes of the bias scaling occurs in the works of both Foster and Syrgkanis [2023] and Liu et al. [2022] under Asm. 4 (called 'slow rate' and 'fast rate' there). Importantly, their bounds are based on an exact, offline empirical risk minimization procedure for a fixed training set, i.e. they provide excess risk bounds on the quantity L ( ˆ θ n , g 0 ) -L ( θ ⋆ , g 0 ) , where

<!-- formula-not-decoded -->

In contrast, Thm. 1 accounts for both the expected distance to optimum and the interplay between bias incurred by ˆ g and the progress achieved at each step. In particular, even when using a constant learning rate, the bias does not accrue on each iterate and is in fact constant in n . When ˆ θ n is designed to be doubly robust, using SGD can achieve double robustness ; see Appx. F.5 for an example.

Orthogonalized Stochastic Gradient Algorithm. Given the marked improvement in the rate of decay of the bias term when an orthogonal loss is used, it is clearly beneficial to do so when possible. We now describe how we can induce orthogonality by adjusting the stochastic gradient oracle using the solution of an auxiliary problem.

The construction of Neyman orthogonal losses has historically been motivated in semiparametric theory and statistical learning as a means to build efficient - minimum asymptotic variance full batch statistical estimators [Tsiatis, 2006, Van der Vaart, 2000, Foster and Syrgkanis, 2023, Chernozhukov et al., 2018b]. The approach we follow is inspired by the construction reviewed in Chernozhukov et al. [2018a, Section 2.2]; see also Luedtke [2024]. We also give an intuitive explanation based on least-squares estimation, instead of the usual differential/information geometry one.

While our construction holds in general spaces, let us first consider the illustrative case when G = R k . At the true parameters ( θ ⋆ , g 0 ) , consider the problem of finding the best predictor of the R d -valued target variable S θ ( θ ⋆ , g 0 ; Z ) = ∇ θ ℓ ( θ ⋆ , g 0 ; Z ) given the R k -valued predictor ∇ g ℓ ( θ ⋆ , g 0 ; Z ) variable in the space L ( G , Θ) containing all continuous and linear operators from G to Θ :

<!-- formula-not-decoded -->

In the special case where ℓ ( θ, g ; z ) = -log p θ,g ( z ) for a density p θ,g on Z that governs the random variable Z , the projection direction solving (9) can be shown to satisfy Γ 0 = H ⊤ θg H -1 gg , where H θg = ∇ g S θ ( θ ⋆ , g 0 ) ∈ R k × d is the transposed Jacobian and H gg = ∇ 2 g L ( θ ⋆ , g 0 ) ∈ R k × k is the Hessian. The prediction Γ 0 ∇ g ℓ ( θ ⋆ , g 0 ; Z ) accounts for the covariance between ∇ θ ℓ ( θ ⋆ , g 0 ; Z ) (the gradient w.r.t. θ ) and ∇ g ℓ ( θ ⋆ , g 0 ; Z ) (the gradient w.r.t. g ). It stands to reason that as θ → θ ⋆ , the random vector

<!-- formula-not-decoded -->

would be less sensitive to perturbations of g 0 , as the component of S θ ( θ, g 0 ; Z ) that is predictable through changes in g 0 is subtracted out. Furthermore, if we are aware that the expectation of S is made zero at θ ⋆ , then a stochastic gradient scheme based on (10) could conceivably achieve a nuisance insensitive rate guarantee in lieu of Thm. 1. From a variance reduction viewpoint, the correction term in (10) subtracts the regression of the θ gradient of the loss on the g 'gradient' of the loss. By the law of total variance, the variance of the gradient reduces and improves the trajectory of stochastic optimization; see Appx. F.4 for more details. This variational description (10) hints at how such an operator can be computed algorithmically, instead of the historical approach of deriving the operator via calculation by hand on case by case basis.

Supported by this illustration, we define a generalization that will provide a modified stochastic gradient oracle to use for optimization purposes. Without assuming that ℓ is a negative log-likelihood, we generalize the formulas for ∇ g ℓ ( θ, g ; z ) ∈ R k , H θg ∈ R k × d and H gg ∈ R k × k for when G ≡ ( G , ⟨· , ·⟩ G ) is an infinite-dimensional Hilbert space. Under regularity conditions on the directional derivatives of L , we have that ∇ g ℓ ( θ, g ; z ) ∈ G for all z ∈ Z , H θg = ( H (1) θg , . . . , H ( d ) θg ) ∈ G d , and H gg : G → G is a bounded and self-adjoint operator. The formal details of their construction are contained in Appx. D. Just as in (10), we may consider the operator Γ 0 : G → R d , defined elementwise by [Γ 0 g ] j = ⟨ H ( j ) θg , H -1 gg g ⟩ G , where the invertibility of H gg is satisfied by our assumptions preceding Thm. 3. As shown in (9), the orthogonalizing Γ 0 is defined by both the true nuisance g 0 and the target θ ⋆ , where g 0 can usually be learned as some conditional expectation and θ ⋆ can be learned by our proposed methods. We then construct the central object of the upcoming Thm. 3: the Neyman orthogonalized (NO) gradient oracle

<!-- formula-not-decoded -->

Lemma 2. Suppose that Asm. 3(a) holds and D 2 g L ( θ ⋆ , g 0 )[ · , · ] : G × G ↦→ R is a bounded and symmetric bilinear form. Then the NO gradient oracle S no ( θ, g ; z ) is Neyman orthogonal at ( θ 0 , g 0 ) .

We refer readers to Lem. 15 for the proof. In this context, we refer to the operator Γ 0 as the 'orthogonalizing operator'. As a natural sanity check, we note that for a risk function that is already Neyman orthogonal at ( θ ⋆ , g 0 ) , the NO score S no is exactly equal to score function S θ itself since Γ 0 = 0 . To construct S no for the non-orthogonal loss, we provide the following example in partially linear model where the corresponding derivations of Γ 0 and S no are included in Appx. B.1.2.

Example 4 (Partially Linear Model) . In addition to Example 1, suppose that Z = ( X,Y,W ) ∼ P satisfies

<!-- formula-not-decoded -->

where θ ⋆ ∈ R d is the true parameter, g 0 : W ↦→ R is the true nuisance function, and E P [ ϵ | X,W ] = 0 . The space G ∈ L 2 ( P ) with inner product ⟨ g 1 , g 2 ⟩ G = E P [ g 1 ( W ) g 2 ( W )] for any g 1 , g 2 ∈ G is a nonparametric class containing functions of the form

<!-- formula-not-decoded -->

Consider the following non-orthogonal squared loss function:

<!-- formula-not-decoded -->

The orthogonalizing operator for this non-orthogonal loss is

<!-- formula-not-decoded -->

and the NO gradient oracle is obtained as

<!-- formula-not-decoded -->

Motivated by the advantage of a Neyman orthogonal score, we now construct our OSGD algorithm using an estimated the NO score S no. While Γ 0 (like g 0 ) is unknown to the user in general, using an arbitrary estimate ˆ Γ , we can define the estimated NO score ˆ S no oracle via

<!-- formula-not-decoded -->

Usually, one can obtain such an estimate ˆ Γ using the same data stream of ˆ g ; we discuss possible strategies in Appx. F.3. Finally, using ˆ S no as the stochastic gradient oracle S in (3), we derive the OSGD update

<!-- formula-not-decoded -->

To measure the quality of ˆ Γ in our analysis, we use the Frobenius norm ∥ Γ ∥ 2 Fro = ∑ d j =1 ∥ Γ ( j ) ∥ 2 op where Γ : G → R d , Γ ( j ) : g ↦→ [Γ g ] j and ∥·∥ op denotes the usual operator norm for linear functionals. As an example, by the uniqueness of Riesz representations, ∥ Γ 0 ∥ 2 Fro = ∑ d j =1 ∥ H -1 gg H ( j ) θg ∥ 2 G .

Using this modified oracle (12) requires similar assumptions to those used in Thm. 1. For ease of presentation, we defer the formal assumption statement to Appx. E, but note that the result depends on the constants ( µ no , M no , α 2 , β 2 , K 2 ) , which are exactly analogous to ( µ, M, α 1 , β 1 , K 1 ) from Asm. 3.

Theorem 3. Consider the setting of Thm. 1, with the addition of Asm. 6. When ∥ ˆ Γ -Γ 0 ∥ Fro &lt; µ no / (4 α 1 ) and

<!-- formula-not-decoded -->

the iterates of (13) satisfy:

<!-- formula-not-decoded -->

Compared with Thm. 1, Thm. 3 shows that OSGD can outperform the nuisance sensitive rate through the correction term ∥ ˆ g -g 0 ∥ 2 G · ∥ ˆ Γ -Γ 0 ∥ 2 Fro , and can align with the nuisance insensitive rate when ∥ ˆ Γ -Γ 0 ∥ Fro is of the order O ( ∥ ˆ g -g 0 ∥ G ) . With slightly different assumptions, Thm. 3 can further simplified - see Appx. E for details.

Interleaving Target and Nuisance Estimation. The results seen thus far have considered for simplicity the estimate ˆ g to be a fixed element of G , and included terms that depend on the discrepancy ∥ ˆ g -g 0 ∥ G . Part of the convenience of these results is that if ˆ g ≡ ˆ g ( m ) is the result of a learning procedure with m independent data points, then statistical bounds on ∥ ˆ g ( m ) -g 0 ∥ G (either in expectation or high probability, depending on the situation) can be plugged in to quantify the bias. While the results naturally account for full batch learning procedures, they are also amenable to analyzing staggered procedures in which two data sources are queried to estimate θ ⋆ and g 0 , respectively. To our knowledge, this is the first theoretical analysis of such an orthogonal stochastic learning method.

To be precise, suppose that we update the nuisance estimator for m times, leading to the sequence ˆ g (1) , . . . , ˆ g ( m ) on a stream of W -valued data W 1 , . . . , W m , sampled i.i.d. from a probability measure Q . We define θ (0 ,n ) = θ (0) ∈ Θ , and for the update of ˆ g ( i ) for 1 ≤ i ≤ m , we define θ ( i, 0) = θ ( i -1 ,n ) and produce the sequence θ ( i, 1) , . . . , θ ( i,n ) using n steps of the SGD update (8) initialized at θ ( i, 0) . Consider, for example, the case in which G is a reproducing kernel Hilbert space (RKHS) with kernel k ( · , · ) . With the assumption that the eigenvalues ( λ j ) j ≥ 1 of covariance operator E Q [ k ( W, · ) ⊗ k ( W, · )] decay polynomially at order j -α , the nonparametric stochastic gradient algorithm of Dieuleveut and Bach [2016] satisfies E Q m [ ∥ ˆ g ( m ) -g 0 ∥ 2 G ] = O ( m -(2 α -1) / (2 α ) ) . This leads to the following nuisance sensitive rate for a non-Neyman orthogonal loss, by Prop. 22:

<!-- formula-not-decoded -->

As another example, suppose that, in addition, we can estimate ˆ Γ ≡ ˆ Γ ( m ) using the nonparametric stochastic gradient algorithm of Dieuleveut and Bach [2016] and using the same data stream ( W 1 , . . . , W m ) . If there are high probability bounds for ∥ ˆ g ( m ) -g 0 ∥ 2 G and ∥ ˆ Γ ( m ) -Γ 0 ∥ 2 Fro of the same order as O ( m -(2 α -1) / (2 α ) ) and ∥ θ ( m,n ) -θ ⋆ ∥ 2 2 decays as described in Thm. 3, then we have in Prop. 23 that ∥ θ ( m,n ) -θ ⋆ ∥ 2 2 = O p ( (1 -µη/ 2) mn + m -(2 α -1) /α + n -1 + η ) where the O p ( m -(2 α -1) /α ) nuisance bias term decays quadratically faster than the one for a non-Neyman orthogonal loss. We refer the reader to Appx. F.3 for further details of this analysis.

## 4 Related Work

We summarize in this section our discussion of the related work. Additional discussions, as well as calculations supporting them, can be found in Appx. F. Possible extensions to SGD variants such as SGD with momentum, averaged SGD, and Adam, are explored in Appx. H.

From an optimization perspective, it is helpful to know how our convergence bounds perform in the idealized case of a known nuisance, which is equivalent to (1). In this case, Thm. 1 gives the convergence rate E D n ∼ P n [ ∥ θ ( n ) -θ ⋆ ∥ 2 2 ] = O ((1 -µη/ 2) n + η ) , which aligns with the nonasymptotic SGD convergence rates, in mean-square error [Bach and Moulines, 2011] and in highprobability [Cutler et al., 2023]. Our result requires a smaller learning rate η &lt; µ/ 2( Mµ + 2 κ 1 ) when compared to the requirement η &lt; 1 / (2 M ) from Cutler et al. [2023]. This is entirely due to our bounded moment assumption (see Asm. 3(d)), which contrasts with a uniform boundedness assumption over all Θ × G r ( g 0 ) . In addition, when the uniform moment bound holds true, κ 1 becomes zero, and our learning rate requirement becomes η &lt; 1 / (2 M ) .

̸

The comparison with unbiased SGD, and biased SGD, respectively, is also valuable. In the biased SGD literature, the 'bias' refers to the fact that E Z ∼ P [ S ( θ, ˆ g ; Z )] = ∇ θ L ( θ, g 0 ) in general. The convergence radius then depends on the average value of ∥ E Z ∼ P [ S ( θ ( n ) , ˆ g ; Z ) ] -∇ θ L ( θ ( n ) , g 0 ) ∥ 2 2 . Results along this line result in a radius that may not scale with η ; see Demidovich et al. [2023, Thm. 3]. Although this form of bias may be related to ∥ ˆ g -g 0 ∥ G under Lipschitzness conditions on the oracle, it is unclear how to effectively incorporate Neyman orthogonality into these general-purpose approaches. Our approach naturally leverages Neyman orthogonality whenever it holds.

In the general case of an unknown nuisance, Foster and Syrgkanis [2023], Chernozhukov et al. [2018b] consider full batch learning methods based on analytically crafted Neyman orthogonal risk functions in various scenarios. For regression functionals, the procedure from Chernozhukov et al. [2022] using random forests or neural networks can ensure that the bias term ∥ ˆ g -g 0 ∥ 2 G is asymptotically negligible for large samples, in the sense that classical statistical confidence sets for θ ⋆ are asymptotically valid. These papers are focused on algorithm-independent statistical properties.

Our work fills this gap, by providing non-asymptotic convergence guarantees for stochastic gradient algorithms under unknown nuisances. Moreover, the modified stochastic gradient oracle moreover offers a flexible solution to deal with general risk functions. If deriving an orthogonalized risk by hand is difficult or impossible, then the strategy we propose can be applied, and Thm. 3 demonstrates that, when the learning rate η is set appropriately, the convergence rate using the modified stochastic gradient oracle can be improved to

<!-- formula-not-decoded -->

When we have the true orthogonalizing Γ 0 , the improved rate recovers the nuisance insensitive one from Thm. 1. Besides, when ˆ g converges but ∥ ˆ Γ -Γ 0 ∥ 2 G = O p (1) , the improved rate resembles the nuisance sensitive rate of Thm. 1, plus a O ( η ) bias term. Thus, the quality of the estimated orthogonalizing operator governs how the optimization interpolates between these two rates.

Having understood the performance of SGD when using an estimated orthogonalizing operator, one question is how to compute or approximate such an operator. Luedtke [2024] recently demonstrated that an orthogonalizing operator can be derived using algorithmic/reverse mode functional differentiation in many interesting cases. This can also be effective in our stochastic setting. In Sec. 3, using least-squares regression as an illustration, we developed a control variate [Johnstone and Velleman, 1985] interpretation of the variance reduction. This viewpoint offers another venue to develop approximate orthogonalizing operators.

Conclusion. We established non-asymptotic convergence guarantees for SGD algorithms under nuisances. We showed how the Neyman orthogonality of the loss function can mitigate the sensitivity of SGD algorithms to the effect of nuisances, and obtained results that align with recent ones from the DML/OSL literature in the batch setting. We also presented an iteratively orthogonalized SGD algorithm, whose convergence rate aligns with the rate in the nuisance insensitive regime. Extensions to hypothesis testing and reinforcement learning are interesting venues for future work.

Acknowledgments. The authors would like to thank L. Liu, V. Roulet, and J. Wellner for valuable comments and suggestions. This work was supported by NSF DMS-2023166, DMS-2134012, DMS-2210216, DMS-2502281, CCF-2019844, NIH, and IARPA 2022-22072200003. Part of this work was performed while R. Mehta and Z. Harchaoui were visiting the Simons Institute for the Theory of Computing, and A. Luedtke was visiting the Institute of Statistical Mathematics, and with the University of Washington.

Broader Impact. This work lies at the intersection of machine learning, mathematical optimization, learning theory, and mathematical statistics. While there are many applications of stochastic gradient optimization for practitioners, this particular work is of a theoretical nature and does not have any immediate positive or negative societal impact.

## References

- S. Amari. Backpropagation and Stochastic Gradient Descent Method. Neurocomputing , 1993.
- F. Bach and E. Moulines. Non-Asymptotic Analysis of Stochastic Approximation Algorithms for Machine Learning. In NeurIPS , 2011.
- P. J. Bickel, C. A. Klaassen, P. J. Bickel, Y . Ritov, J. Klaassen, J. A. Wellner, and Y . Ritov. Efficient and Adaptive Estimation for Semiparametric Models . Johns Hopkins University Press Baltimore, 1993.
- M. Bonvini and E. H. Kennedy. Fast Convergence Rates for Dose-Response Estimation, 2022.
- L. Bottou and O. Bousquet. The Tradeoffs of Large Scale Learning. In NeurIPS , 2007.
- L. Bottou and Y. Le Cun. On-Line Learning for Very Large Data Sets. Applied Stochastic Models in Business and Industry , 2005.
- S. P. Boyd and L. Vandenberghe. Convex Optimization . Cambridge University Press, 2004.
- M. Carone, A. R. Luedtke, and M. J. van Der Laan. Toward Computerized Efficient Estimation in Infinite-Dimensional Models. Journal of the American Statistical Association , 2019.
- V. Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey, and J. Robins. Double/Debiased Machine Learning for Treatment and Structural Parameters. The Econometrics Journal , 2018a.
- V. Chernozhukov, D. N. Nekipelov, V. Semenova, and V. Syrgkanis. Plug-In Regularized Estimation of High-Dimensional Parameters in Nonlinear Semiparametric Models. Technical report, Cemmap Working Paper, 2018b.
- V. Chernozhukov, W. K. Newey, and R. Singh. Automatic Debiased Machine Learning of Causal and Structural Effects. Econometrica , 2022.
- V. Chernozhukov, W. K. Newey, V. Quintas-Martinez, and V. Syrgkanis. Automatic Debiased Machine Learning via Riesz Regression, 2024.
- J. Clore, K. Cios, J. DeShazo, and B. Strack. Diabetes 130-US Hospitals for Years 1999-2008. UCI Machine Learning Repository, 2014.
- J. Cutler, D. Drusvyatskiy, and Z. Harchaoui. Stochastic Optimization under Distributional Drift. Journal of Machine Learning Research , 2023.
- A. D´ efossez and F. Bach. Averaged Least-Mean-Squares: Bias-Variance Trade-Offs and Optimal Sampling Distributions. In AISTATS , 2015.
- A. D´ efossez, L. Bottou, F. Bach, and N. Usunier. A Simple Convergence Proof of Adam and Adagrad. A Simple Convergence Proof of Adam and Adagrad , 2022.
- Y. Demidovich, G. Malinovsky, I. Sokolov, and P. Richt´ arik. A Guide Through the Zoo of Biased SGD. In NeurIPS , 2023.

- A. Dieuleveut and F. Bach. Nonparametric Stochastic Approximation with Large Step-Sizes. The Annals of Statistics , 2016.
- T. S. Ferguson. Mathematical Statistics: A Decision Theoretic Approach . Academic press, 2014.
- D. J. Foster and V. Syrgkanis. Orthogonal Statistical Learning. The Annals of Statistics , 2023.
- E. Gorbunov, F. Hanzely, and P. Richtarik. A Unified Theory of SGD: Variance Reduction, Sampling, Quantization and Coordinate Descent. In AISTATS , 2020.
- R. M. Gower, M. Schmidt, F. Bach, and P. Richt´ arik. Variance-Reduced Methods for Machine Learning. Proceedings of the IEEE , 2020.
- C. Graham and D. Talay. Stochastic Simulation and Monte Carlo Methods . Springer Berlin, Heidelberg, 2013.
- M. G¨ urb¨ uzbalaban, A. Ozdaglar, and P. A. Parrilo. Why Random Reshuffling Beats Stochastic Gradient Descent. Mathematical Programming , 2021.
- Y. Hao and A. Orlitsky. The Broad Optimality of Profile Maximum Likelihood. In NeurIPS , 2019.
- H. Ichimura and W. K. Newey. The Influence Function of Semiparametric Estimators. Quantitative Economics , 2022.
- I. M. Johnstone and P. F. Velleman. Efficient Scores, Variance Decompositions, and Monte Carlo Swindles. Journal of the American Statistical Association , 1985.
- M. Jordan, Y. Wang, and A. Zhou. Empirical Gateaux Derivatives for Causal Inference. In NeurIPS , 2022.
- E. H. Kennedy. Towards Optimal Doubly Robust Estimation of Heterogeneous Causal Effects. Electronic Journal of Statistics , 2023.
- M. J. Laan and J. M. Robins. Unified Methods for Censored Longitudinal Data and Causality . Springer, 2003.
- B. Y. Levit. Infinite-Dimensional Informational Inequalities. Theory of Probability &amp; Its Applications , 1979.
- D. Levy, Y. Carmon, J. Duchi, and A. Sidford. Large-Scale Methods for Distributionally Robust Optimization. In NeurIPS , 2020.
- X. Li, M. Liu, and F. Orabona. On the Last Iterate Convergence of Momentum Methods. In ALT , 2022.
- J. V. Linnik. Statistical Problems with Nuisance Parameters . American Mathematical Society, 2008.
- L. Liu, C. Cinelli, and Z. Harchaoui. Orthogonal Statistical Learning with Self-Concordant Loss. In COLT , 2022.
- A. Luedtke. Simplifying Debiased Inference via Automatic Differentiation and Probabilistic Programming, 2024.
- A. Luedtke and I. Chung. One-Step Estimation of Differentiable Hilbert-Valued Parameters. The Annals of Statistics , 2024.
- R. Mehta, V. Roulet, K. Pillutla, L. Liu, and Z. Harchaoui. Stochastic Optimization for Spectral Risk Measures. In AISTATS , 2023.
- R. Mehta, V. Roulet, K. Pillutla, and Z. Harchaoui. Distributionally Robust Optimization with Bias and Variance Reduction. In ICLR , 2024.
- S. A. Murphy and A. W. V. D. V. and. On Profile Likelihood. Journal of the American Statistical Association , 2000.
- W. K. Newey. The Asymptotic Variance of Semiparametric Estimators. Econometrica , 1994.

- J. Neyman. Optimal Asymptotic Tests for Composite Hypotheses. Probability and Statistics , 1959.
- J. Neyman. C ( α ) Tests and Their Use. Sankhy¯ a: The Indian Journal of Statistics, Series A , 1979.
- X. Nie and S. Wager. Quasi-Oracle Estimation of Heterogeneous Treatment Effects. Biometrika , 2021.
- D. S. Pavlichin, J. Jiao, and T. Weissman. Approximate Profile Maximum Likelihood. Journal of Machine Learning Research , 2019.
- J. Pfanzagl. Asymptotic Expansions for General Statistical Models . Springer-Verlag Berlin Heidelberg, 1985.
- A. Rahimi and B. Recht. Random Features for Large-Scale Kernel Machines. In NeurIPS , 2007.
- J. Robins, L. Li, E. Tchetgen, A. van der Vaart, et al. Higher Order Influence Functions and Minimax Estimation of Nonlinear Functionals. In Probability and Statistics: Essays in Honor of David A. Freedman . Institute of Mathematical Statistics, 2008.
- J. M. Robins and A. Rotnitzky. Semiparametric Efficiency in Multivariate Regression Models with Missing Data. Journal of the American Statistical Association , 1995.
- J. M. Robins, A. Rotnitzky, and L. P. Zhao. Estimation of Regression Coefficients When Some Regressors Are Not Always Observed. Journal of the American Statistical Association , 1994.
- P. M. Robinson. Root-N-Consistent Semiparametric Regression. Econometrica , 1988.
- A. Rotnitzky, E. Smucler, and J. M. Robins. Characterization of Parameters with a Mixed Bias Property. Biometrika , 2021.
- D. B. Rubin. Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies. Journal of Educational Psychology , 1974.
- M. Schmidt, N. Le Roux, and F. Bach. Minimizing Finite Sums with the Stochastic Average Gradient. Mathematical Programming , 2017.
- A. Sclocchi and M. Wyart. On the Different Regimes of Stochastic Gradient Descent. PNAS , 2024.
- A. Shapiro. Distributionally Robust Stochastic Programming. SIAM Journal on Optimization , 2017.
- C. Shi, D. Blei, and V. Veitch. Adapting Neural Networks for the Estimation of Treatment Effects. In NeurIPS , 2019.
- Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-Based Generative Modeling through Stochastic Differential Equations. In ICLR , 2021.
- D. Soudry, E. Hoffer, M. S. Nacson, S. Gunasekar, and N. Srebro. The Implicit Bias of Gradient Descent on Separable Data. Journal of Machine Learning Research , 2018.
- A. A. Tsiatis. Semiparametric Theory and Missing Data . Springer, 2006.
- L. van der Laan, M. Carone, and A. Luedtke. Combining T-learning and DR-Learning: A Framework for Oracle-Efficient Estimation of Causal Contrasts, 2024.
- L. van der Laan, A. Bibaut, N. Kallus, and A. Luedtke. Automatic Debiased Machine Learning for Smooth Functionals of Nonparametric M-Estimands, 2025.
- M. J. van der Laan and A. R. Luedtke. Targeted Learning of an Optimal Dynamic Treatment, and Statistical Inference for Its Mean Outcome. U.C. Berkeley Division of Biostatistics Working Paper Series , 2014.
- M. J. van der Laan, S. Rose, et al. Targeted Learning: Causal Inference for Observational and Experimental Data . Springer, 2011.
- A. W. Van der Vaart. Asymptotic Statistics . Cambridge University Press, 2000.

- S. Vaswani, A. Mishkin, I. Laradji, M. Schmidt, G. Gidel, and S. Lacoste-Julien. Painless Stochastic Gradient: Interpolation, Line-Search, and Convergence Rates. In NeurIPS , 2019.
- R. Ward, X. Wu, and L. Bottou. Adagrad Stepsizes: Sharp Convergence over Nonconvex Landscapes. Journal of Machine Learning Research , 2020.
- J. M. Wooldridge. Specification Testing and Quasi-Maximum-Likelihood Estimation. Journal of Econometrics , 1991.
- I. Zadik, L. Mackey, and V. Syrgkanis. Orthogonal Machine Learning: Power and Limitations. In ICML , 2018.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Our claims are theoretical convergence guarantees for various optimization algorithms. The results are included in Sec. 3 and the proofs are written in the appendix.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: A detailed discussion including limitations, relationships to other work, and future work is contained in Sec. 4.

Guidelines:

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

## Answer: [Yes]

Justification: Assumptions are given before the statement of each result and proofs are contained in the appendix.

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

Justification: While this work is primarily theoretical, we provide code that reproduces our numerical illustrations.

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

Justification: The code is provided in github.

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

Justification: Experiments are included in Appx. G.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: Experiments of this nature are not included in our paper.

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

Answer: [NA]

Justification: Our numerical illustration is not computationally prohibitive, and can run on an instance of Google Colab.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: There are no ethical violations, to the authors' knowledge.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: A 'Broader Impact' statement is included before the references.

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

Justification: We do not provide any models or datasets in this paper.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: We do not use any third party data/models that may incur licensing issues.

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

Answer: [Yes]

Justification: Our code is documented in notebook format.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Experiments of this nature are not included in this paper.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Experiments of this nature are not included in this paper.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: No substantive part of this research involved the use of large language models.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/ 2025/LLM ) for what should or should not be described.

## Appendix

## Table of Contents

| A   | Notation                                                  | Notation                                                                           | 23    |
|-----|-----------------------------------------------------------|------------------------------------------------------------------------------------|-------|
| B   | Detailed Examples                                         | Detailed Examples                                                                  | 24    |
|     | B.1                                                       | Partially Linear Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | 24    |
|     | B.2                                                       | Conditional Averaged Treatment Effect . . . . . . . . . . . . . . . . . . . . .    | 27    |
|     | B.3                                                       | Conditional Relative Risk . . . . . . . . . . . . . . . . . . . . . . . . . . . .  | 28    |
|     | B.4                                                       | Proofs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | 29    |
| C   | Convergence Proofs for Stochastic Gradient                | Convergence Proofs for Stochastic Gradient                                         | 47    |
|     | C.1                                                       | Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   | 47    |
|     | C.2                                                       | Technical Lemma . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    | 47    |
|     | C.3                                                       | Proof of Theorem 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   | 50    |
| D   | Orthogonalization with respect to Nuisance                | Orthogonalization with respect to Nuisance                                         | 52    |
|     | D.1                                                       | Orthogonalization via Riesz Representation . . . . . . . . . . . . . . . . . . .   | 52    |
|     | D.2                                                       | Technical Lemma . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    | 53    |
| E   | Convergence Proofs for Orthogonalized Stochastic Gradient | Convergence Proofs for Orthogonalized Stochastic Gradient                          | 54    |
|     | E.1                                                       | Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   | 54    |
|     | E.2                                                       | Technical Lemma . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    | 55    |
|     | E.3                                                       | Proof of Theorem 3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   | 57    |
| F   | Detailed Discussion                                       | Detailed Discussion                                                                | 60    |
|     | F.1                                                       | Comparison to Biased SGD . . . . . . . . . . . . . . . . . . . . . . . . . . .     | 60    |
|     | F.2                                                       | Discussion of Full-sample Orthogonal Statistical Learning and Related Methods      | 61    |
|     | F.3                                                       | Discussion of Interleaving Target and Nuisance Estimation . . . . . . . . . . .    | 63    |
|     | F.4                                                       | Interpretation as Control Variate for Variance Reduction . . . . . . . . . . . .   | 64    |
|     | F.5                                                       | Discussion of Double Robustness . . . . . . . . . . . . . . . . . . . . . . . .    | 65    |
|     | F.6                                                       | Proof of Proposition 22 . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  | 67    |
|     | F.7 Proof of Proposition Experiments                      | 23 . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                       | 69 71 |
| G   | Numerical                                                 | Numerical                                                                          |       |
|     | G.1                                                       | Numerical Illustration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | 71    |
|     | G.2                                                       | Simulations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  | 72    |
|     | G.3                                                       | Real Data Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   | 77    |
| H   | Extension to SGD Variants                                 | Extension to SGD Variants                                                          | 81    |
|     | H.1                                                       | SGD with Momentum and Averaged SGD . . . . . . . . . . . . . . . . . . .           | 81    |
|     | H.2                                                       | Adam . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   | 84    |

## A Notation

Table 2: Notation used throughout the paper.

| Symbol                                                                      | Description                                                                                                                                                                                                                                                                       |              |            |     |    |    |    |    |    |     |    |
|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|------------|-----|----|----|----|----|----|-----|----|
| Θ ⊆ R d ( G , ∥ · ∥ G ) ( G , ⟨· , ·⟩ G )                                   | Finite-dimensional parameter class. Possibly infinite-dimensional nuisance space. The nuisance space as a Hilbert space.                                                                                                                                                          |              |            |     |    |    |    |    |    |     |    |
| P Z ∈ Z θ ⋆ g                                                               | The unknown distribution of interest. The random variable under P . The target of interest. The true nuisance parameter.                                                                                                                                                          |              |            |     |    |    |    |    | 0  |     |    |
| ℓ ( θ, g ; z ) L ( θ, g ) S θ ( θ, g ; z ) S θ ( θ, g ) S no ( θ, g ; z ) S | loss function. loss E Z ∼ P [ ℓ ( θ, g ; Z )] . The score function ∇ θ ℓ ( θ, g ; z ) . The population score E Z ∼ P [ ∇ θ ℓ ( θ, g ; z )] . The Neyman orthogonalized score. The population Neyman orthogonalized score E Z ∼ P [ S no ( θ, g ; Z )] The gradient w.r.t. θ and g | prespecified | population | The | θ, | g  | no | (  |    | The | )  |
| ( ∇ θ , ∇ g ) (D θ , D g ) H θg H gg Γ 0                                    | The derivative operator w.r.t. θ and g . The transposed Jacobian defined by ∇ g S θ ( θ ⋆ , g 0 ) ∈ G d The nuisance Hessian operator defined by ∇ 2 g L ( θ ⋆ , g 0 ) Linear operator defined by [Γ 0 g ] j = ⟨ H ( j ) θg ,H - 1 gg g ⟩ G .                                     |              |            |     |    |    |    |    |    |     |    |
| µ M ( K 1 ,κ 1 ) ( α 1 ,α 2 ) β 1                                           | The strong convexity constant of L . The smoothness constant of L . Constants to bound the second moment of S θ ( θ, g ; Z ) . The second order smoothness constant of L . The orthogonal L                                                                                       |              |            |     |    |    |    |    |    |     |    |
| µ no M no ( K 2 ,κ 2 ) β 2                                                  | The strong convexity constant of ∇ θ S no ( θ ⋆ , g 0 ) . The smoothness constant of ∇ θ S no ( θ ⋆ , g 0 ) . Constants to bound the second moment of S no ( θ, g ; Z ) . The higher order smoothness constant of S no .                                                          |              |            |     |    |    |    |    |    |     |    |
| η n                                                                         | The learning rate of stochastic optimization. The iteration of stochastic gradient. The iteration of nuisance estimation.                                                                                                                                                         |              |            |     |    |    |    |    |    |     |    |
| m                                                                           |                                                                                                                                                                                                                                                                                   |              |            |     |    |    |    |    |    |     |    |
|                                                                             | higher order smoothness constant of a Neyman .                                                                                                                                                                                                                                    |              |            |     |    |    |    |    |    |     |    |

## B Detailed Examples

In this section, we describe in detail how the three examples in Sec. 2 from the main text satisfy Asm. 3 and Asm. 4. We first talk about the partially linear model (PLM) in Appx. B.1, and then introduce the conditional averaged treatment effect (CATE) based on the potential outcomes framework in Appx. B.1. Under the same framework, finally we talk about the conditional relative risk (CRR) in Appx. B.3. In addition, we also study a non-orthogonal loss usually used for PLM in Appx. B.1.2 and an unrestricted loss function for CATE in Appx. B.2.1. The constants for all examples are concluded in Tab. 3 and proofs of lemmas in this section are provided in Appx. B.4.

## B.1 Partially Linear Model

## B.1.1 Orthogonal Loss

We revisit Example 1 from the main text where we consider the target of interest as a solution of a partially linear model. Let Z = ( X,Y,W ) , where X is an R d -valued input, Y is a realvalued outcome, and W is a W -valued control or confounder. The space G is a nonparametric class containing functions of the form

<!-- formula-not-decoded -->

Following the construction of Robinson [1988], this g is supplied to the loss

<!-- formula-not-decoded -->

To ensure θ ⋆ can be interpreted via the projection of E P [ Y | X,W ] onto partially linear additive functions, the true nuisance is given by g 0 = ( g 0 ,X , g 0 ,Y ) , where

<!-- formula-not-decoded -->

We define the residual ϵ at ( θ ⋆ , g 0 ) as

<!-- formula-not-decoded -->

Lemma 4. Let ˜ Y = Y -g 0 ,Y ( w ) and ˜ X = X -g 0 ,X ( w ) . We assume the following conditions:

- (a) λ min ( E P [ ˜ X ˜ X ⊤ ]) ≥ λ 0 for some constant λ 0 &gt; 0 .
- (b) ∥ ˜ X ∥ 2 ≤ C X a.s. and E P [ ϵ 4 ] ≤ σ 4 for some constants C X , σ &gt; 0 .

Then Asm. 3 and Asm. 4 are satisfied. The target θ ⋆ is the minimizer of the squared loss:

<!-- formula-not-decoded -->

The proof of Lem. 4 is provided in Appx. B.4.1.

## B.1.2 Non-orthogonal Loss

Suppose that the outcome Y is generated under the partially linear model:

<!-- formula-not-decoded -->

where θ 0 ∈ R d is the true parameter, g 0 : W ↦→ R is the true nuisance function and E P [ ϵ | X,W ] = 0 . The space G is a nonparametric class containing functions of the form

<!-- formula-not-decoded -->

We can also consider the following non-orthogonal squared loss function:

<!-- formula-not-decoded -->

Table 3: Constants for All Examples.

|         |                          |                        | ))                                                      |                                    |
|---------|--------------------------|------------------------|---------------------------------------------------------|------------------------------------|
| β 1     | 2(1+ ∥ θ ⋆ ∥ 2 )         | -                      | (1 +4( C X ∥ θ ⋆ ∥ 2 + C τ                              | 4 c - 2 0 C X (1+ r )              |
| α 1     | 2(1+ ∥ θ ⋆ ∥ 2 ) r       | C X                    | O ( r )                                                 | 2 C X (2 r +3) r                   |
| κ 1     | 18 C 4 X + O ( r 2 )     | 2 C 4 X                | 27 C 4 X + O ( r 2 )                                    | 3 C 4 X                            |
| K 1     | 18 C 2 X σ 2 + O ( r 2 ) | 6 C 2 X ( σ 2 +2 r 2 ) | 12 C 2 X ( σ 2 +4( C X ∥ θ ⋆ ∥ 2 + C τ ) 2 )+ O ( r 2 ) | 27 C 2 X ( σ 2 +2 C 2 )+ O ( r 2 ) |
| M       | C 2 X + r 2              | C 2 X                  | C 2 X (1+ r 2                                           | C 2 X                              |
| µ       | λ 0                      | λ 0                    | c 2 0 λ 0                                               | λ 0                                |
| Example | (1) Orthogonal PLM       | (2) Non-Orthogonal PLM | (3) Unrestricted CATE                                   | (4) Restricted CATE                |

We define the residual ϵ at ( θ ⋆ , g 0 ) as

<!-- formula-not-decoded -->

Lemma 5. We assume the following conditions:

- (a) λ min ( E P [ XX ⊤ ] ) ≥ λ 0 for some constant λ 0 &gt; 0 .

<!-- formula-not-decoded -->

Then Asm. 3 is satisfied and the target θ ⋆ is the true parameter, i.e., θ ⋆ = θ 0 .

The proof of Lem. 5 is provided in Appx. B.4.2.

Orthogonalization. We can perform our orthogonalization method to obtain the Neyman orthogonal gradient oracle for this non-orthogonal loss. For any h 1 , h 2 ∈ G , we define the inner product of G as

<!-- formula-not-decoded -->

For any ( θ, g, z ) ∈ Θ × G ×Z By Definition 1 the derivative of non-orthogonal loss (17) along the direction of h 1 is given by

<!-- formula-not-decoded -->

Do derivative on D g ℓ ( θ, g ; z )[ h 1 ] along the direction of h 2 and we have

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

By the definition in (84), we have H gg = I the identity operator. In addition, do derivative on the score along the direction of h ∈ G and we have

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

By the definition in (83), we have H θg = E P [ X | W ] . Thus, by (85) we have

<!-- formula-not-decoded -->

Thus, the Neyman orthogonalized gradient oracle defined in (86) is given by

<!-- formula-not-decoded -->

Lemma 6. Consider the bounded linear operator ˆ Γ : G ↦→ R d such that [ ˆ Γ g ] j = ⟨ ˆ γ ( j ) , g ⟩ G , ∀ g ∈ G for some ˆ γ ( j ) ∈ G , j = 1 , . . . , d . Let ˜ Y = Y -g 0 ,Y ( w ) and ˜ X = X -g 0 ,X ( w ) . We assume the following conditions:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then Asm. 6 is satisfied.

The proof of Lem. 6 is provided in Appx. B.4.3.

## B.2 Conditional Averaged Treatment Effect

We now introduce examples in causal inference which are established based on the potential outcomes framework. The potential outcomes framework [Rubin, 1974] has been widely used in causal inference. Let Z = ( W,X,Y ) ∈ { 0 , 1 }× R d × R under some distribution P . We posit the existence of potential outcomes Y (1) , Y (0) ∈ R . The conditional averaged treatment effect (CATE) is then defined as

<!-- formula-not-decoded -->

To identify τ 0 ( x ) and the following causal assumptions are required:

Assumption 5. The following conditions hold:

- (a) (consistency) Y = Y ( W ) .
- (b) (unconfoundedness) Y ( w ) ⊥ W | X for all w ∈ { 0 , 1 } .
- (c) (positive overlap) c 0 ≤ P ( W = 1 | X ) ≤ 1 -c 0 a.s. for some c 0 &gt; 0 .

Under Asm. 5, τ 0 can be identified by observed data since

<!-- formula-not-decoded -->

## B.2.1 Unrestricted Nuisance

We observe Z = ( X,Y,W ) , where W is a binary treatment assignment. The functions in G are of the form

<!-- formula-not-decoded -->

and are evaluated (see Nie and Wager [2021, Eq. (2)]) at the loss

<!-- formula-not-decoded -->

For g 0 = ( g out 0 , g prop 0 ) nuisance functions g out 0 and g prop 0 represent the outcome regression and the propensity score, respectively:

<!-- formula-not-decoded -->

We define the residual ϵ under the true model as

<!-- formula-not-decoded -->

Lemma 7. We assume Asm. 5 and the following conditions hold:

- (a) λ min ( E P [ XX ⊤ ] ) ≥ λ 0 for some constant λ 0 &gt; 0 .
- (b) ∥ X ∥ 2 ≤ C X and | τ 0 ( X ) | ≤ C τ a.s. and E P [ ϵ 4 ] ≤ σ 4 for some constants C X , C τ , σ &gt; 0 .

Then Asm. 3 and Asm. 4 are satisfied. The target θ ⋆ is the minimizer of the squared loss:

<!-- formula-not-decoded -->

The proof of Lem. 7 is provided in Appx. B.4.4.

## B.2.2 Restricted Nuisance

We observe Z = ( X,Y,W ) , where W is a binary treatment assignment. Here we restrict the propensity model as g prop : R d ↦→ (0 , 1) . The functions in G are of the form

<!-- formula-not-decoded -->

and are evaluated (see van der Laan and Luedtke [2014, Thm. 1]) at the loss

<!-- formula-not-decoded -->

This loss also appears in Foster and Syrgkanis [2023, Eq. (23)]. For g 0 = ( g (0) 0 , g (1) 0 , g prop 0 ) nuisance functions g (0) 0 and g (1) 0 represent the outcome regressions

<!-- formula-not-decoded -->

We define the residual ϵ as

<!-- formula-not-decoded -->

Lemma 8. We assume Asm. 5 and the following conditions hold:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then Asm. 3 and Asm. 4 are satisfied. The target θ ⋆ is the minimizer of the squared loss:

<!-- formula-not-decoded -->

The proof of Lem. 8 is provided in Appx. B.4.5.

## B.3 Conditional Relative Risk

We retain all components of the previous example, changing only the loss and assuming that the outcome Y is binary/non-negative. First, consider the 'label' function

<!-- formula-not-decoded -->

where 1 ( · ) denotes the indicator function, and the log-linear predictor p θ ( x ) = e ⟨ θ,x ⟩ / (1 + e ⟨ θ,x ⟩ ) . Following Example 2 in van der Laan et al. [2024], we then employ the cross entropy-type loss function

<!-- formula-not-decoded -->

Lemma 9. We assume the following conditions:

- (a) λ min ( E P [ XX ⊤ ] ) ≥ λ 0 for some constant λ 0 &gt; 0 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then Asm. 3 and Asm. 4 are satisfied. The target θ ⋆ is the minimizer of the weighted cross entropy loss:

<!-- formula-not-decoded -->

The proof of Lem. 9 is provided in Appx. B.4.6.

## B.4 Proofs

## B.4.1 Proof of Lemma 4

Proof. We consider the following loss:

<!-- formula-not-decoded -->

with the corresponding risk function defined as

<!-- formula-not-decoded -->

Let ˜ Y = Y -g 0 ,Y ( w ) and ˜ X = X -g 0 ,X ( w ) . By definition, the target θ ⋆ is the minimizer of the squared loss:

<!-- formula-not-decoded -->

Differentiating ℓ ( θ, g ; z ) with respect to θ , we obtain the gradient and Hessian w.r.t. θ as

<!-- formula-not-decoded -->

The expected gradient and expected Hessian are then obtained as

<!-- formula-not-decoded -->

We consider the nuisance neighborhood such that for g ∈ G r ( g 0 ) ,

<!-- formula-not-decoded -->

We now verify that the loss function ℓ satisfies Asm. 3.

(a) We assume that g X ( w ) : W ↦→ R d and g Y ( w ) : W ↦→ R are continuous functions, thus Asm. 3(a) is satisfied.

(b) By (26), it follows from KKT conditions that

<!-- formula-not-decoded -->

(c) Since E P [ ˜ X | W ] = 0 and E P [ ˜ Y | W ] = 0 , we have H θθ ( θ, g ) = E P [ ˜ X ˜ X ⊤ ] + E P [ ( g X ( W ) -g 0 ,X ( W ))( g X ( W ) -g 0 ,X ( W )) ⊤ ] . For any g ∈ G r , when λ min ( E P [ ˜ X ˜ X ⊤ ]) ≥ λ 0 and ∥ ˜ X ∥ 2 ≤ C X a.s. , we have λ 0 I ≼ H θθ ( θ, g ) ≼ ( C 2 X + r 2 ) I = ⇒ µ = λ 0 and M = C 2 X + r 2 . (29)

(d) Consider the Taylor expansion around θ ⋆ , we have

<!-- formula-not-decoded -->

Since E P [ ϵ | W ] = 0 , E P [ ˜ X | W ] = 0 by definition and E P [ ϵ ˜ X ] = 0 by (28), then for any g ∈ G r ( g 0 ) ,

<!-- formula-not-decoded -->

Similarly, we have

<!-- formula-not-decoded -->

which implies that for g ∈ G r ( g 0 ) , when E P [ ϵ 4 ] ≤ σ 4 ,

<!-- formula-not-decoded -->

Thus, for any g ∈ G r ( g 0 ) ,

<!-- formula-not-decoded -->

On the other hand, since

<!-- formula-not-decoded -->

by (29) we have

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

which implies

Note that

<!-- formula-not-decoded -->

By identical proof of (32), we have that L satisfies Asm. 4 since

<!-- formula-not-decoded -->

which implies

## B.4.2 Proof of Lemma 5

Proof. We consider the following loss:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(e) For any θ ∈ Θ and g, ¯ g ∈ G r ( g 0 ) , by (30) we have

<!-- formula-not-decoded -->

Since ¯ g ∈ G r ( g 0 ) ,

<!-- formula-not-decoded -->

Similarly,

Thus,

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In addition, the risk L is Neyman orthogonal at ( θ ⋆ , g 0 ) since

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with the corresponding risk function defined as

<!-- formula-not-decoded -->

Under the true nuisance, the target is the minimizer of the following squared loss:

<!-- formula-not-decoded -->

Since ϵ = Y -g 0 ( W ) -⟨ θ 0 , X ⟩ satisfies E P [ ϵ | X,W ] = 0 under the true model, by bias-variance decomposition, we have

<!-- formula-not-decoded -->

Differentiating ℓ ( θ, g ; z ) with respect to θ , we obtain the gradient and Hessian w.r.t. θ as

<!-- formula-not-decoded -->

The expected gradient and expected Hessian are then obtained as

<!-- formula-not-decoded -->

We consider the nuisance neighborhood such that for g ∈ G r ( g 0 ) ,

<!-- formula-not-decoded -->

We now verify that the loss function ℓ satisfies Asm. 3.

(a) We assume that g : W ↦→ R is continuous, thus Asm. 3(a) is satisfied.

- (b) Since θ ⋆ = θ 0 by (35), we have

<!-- formula-not-decoded -->

(c) When λ min ( E P [ XX ⊤ ] ) ≥ λ 0 &gt; 0 and ∥ X ∥ 2 ≤ C X a.s. , L ( θ, g ) is λ 0 -strongly convex and C 2 X -smooth since

<!-- formula-not-decoded -->

- (d) Consider the Taylor expansion around θ ⋆ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

On the other hand, which implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For g ∈ G r ( g 0 ) , when E P [ ϵ 2 ] ≤ σ 2 we have

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

(e) For any θ ∈ Θ and g, ¯ g ∈ G r ( g 0 ) , we have

<!-- formula-not-decoded -->

which implies that

## B.4.3 Proof of Lemma 6

Proof. We consider the following loss:

<!-- formula-not-decoded -->

with the corresponding risk function defined as

<!-- formula-not-decoded -->

First by the same proof as Appx. B.4.2, we have θ ⋆ = θ 0 . Define the inner product ⟨· , ·⟩ G as (18) and define the norm ∥·∥ G such that ∥ g ∥ 2 G = ⟨ g, g ⟩ G ∀ g ∈ G . Consider a uniformly bounded neighborhood G r ( g 0 ) such that

<!-- formula-not-decoded -->

The NO gradient oracle for this non-orthogonal loss is derived as (22) such that

<!-- formula-not-decoded -->

We now verify that Asm. 6 is satisfied.

(a) Since ϵ = Y -g 0 ( W ) -⟨ θ 0 , X ⟩ satisfies E P [ ϵ | X,W ] = 0 under the true model, by (22) we first have

<!-- formula-not-decoded -->

Let γ ( j ) 0 = H -1 gg H ( j ) θg for j = 1 , . . . , d . By (85), we have [Γ 0 g ] j = ⟨ γ ( j ) 0 , g ⟩ G , ∀ g ∈ G . Thus, by (82) we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which, by (19), implies that

<!-- formula-not-decoded -->

Thus, Asm. 6(a) holds true due to (42) and (43).

(b) By (22), for any ( θ, g ) ∈ Θ ×G ,

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

Thus, Asm. 6(b) holds true for µ no = λ 0 .

(c) For any

(

θ, g,

¯

g

)

∈

Θ

×G × G

r

(

g

0

)

, by (19),

<!-- formula-not-decoded -->

Assume that E P [ ϵ 2 | W ] ≤ σ 2 and ∥ X ∥ ∞ ≤ C X a.s. . By (41) we have

<!-- formula-not-decoded -->

Thus, Asm. 6(c) holds true for K 2 = 3( σ 2 + r 2 ) and κ 2 = 3 C 2 X . (d) For any ( θ, ¯ g, g 1 , g 2 ) ∈ Θ ×G r ( g 0 ) ×G × G , by (20),

<!-- formula-not-decoded -->

In addition, for any ( θ, ¯ θ, g ) ∈ Θ × Θ ×G , by (19) we have

<!-- formula-not-decoded -->

Thus, Asm. 6(d) holds true for α 2 = 1 due to (46) and α 1 = C X due to (47). (e) Note that

<!-- formula-not-decoded -->

which implies that for any g 1 , g 2 ∈ G ,

<!-- formula-not-decoded -->

Thus, Asm. 6(e) holds true for β 2 = 0 .

## B.4.4 Proof of Lemma 7

Proof. We consider the following loss:

<!-- formula-not-decoded -->

with the corresponding risk function defined as

<!-- formula-not-decoded -->

Note that ϵ = Y -g out 0 ( X ) -( W -g prop 0 ( X ) ) τ 0 ( X ) . Under Asm. 5, we have E P [ ϵ | W,X ] = 0 , which implies that

<!-- formula-not-decoded -->

Thus, the target is the minimizer of the following squared loss:

<!-- formula-not-decoded -->

Differentiating ℓ ( θ, g ; z ) with respect to θ , we obtain the gradient and Hessian w.r.t. θ as

<!-- formula-not-decoded -->

The expected gradient and expected Hessian are then obtained as

<!-- formula-not-decoded -->

We consider the nuisance neighborhood such that for g ∈ G r ( g 0 ) ,

<!-- formula-not-decoded -->

We now verify that the loss function ℓ satisfies Asm. 3.

(a) We assume that g out : R d ↦→ R and g prop : R d ↦→ R are continuous, thus Asm. 3(a) is satisfied.

(b) Since θ ⋆ is a global minimizer of (49), we have

<!-- formula-not-decoded -->

(c) We assume that c 0 ≤ g prop 0 ( X ) ≤ 1 -c 0 a.s. for some c 0 &gt; 0 . When λ min ( E P [ XX ⊤ ] ) ≥ λ 0 &gt; 0 and ∥ X ∥ 2 ≤ C X a.s. , we have

<!-- formula-not-decoded -->

(d) Consider the Taylor expansion around θ ⋆ , we have

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

We assume that τ 0 : R d ↦→ R is continuous. Then when ∥ X ∥ 2 ≤ C X a.s. , | τ 0 ( X ) | ≤ C τ for some C τ &gt; 0 . It follows that

<!-- formula-not-decoded -->

Since E P [ ( W -g prop ( X )) 2 | X ] = g prop 0 (1 -g prop 0 )( X ) + (( g prop -g prop 0 )( X )) 2 and ( W -g prop ( X )) 2 ≤ 2 + 2(( g prop -g prop 0 )( X )) 2 , we have

<!-- formula-not-decoded -->

When E P [ ϵ 4 ] ≤ σ 4 , by H¨ older inequality,

<!-- formula-not-decoded -->

Similarly, use the fact that E P [ ϵ | W,X ] = 0 and by the stationary condition of (49), we have

<!-- formula-not-decoded -->

which implies

On the other hand,

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

- (e) For any θ ∈ Θ and g, ¯ g ∈ G r ( g 0 ) , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

which implies

We define the residual ϵ as

<!-- formula-not-decoded -->

In addition, L ( θ, g ) is Neyman orthogonal at ( θ ⋆ , g 0 ) since

<!-- formula-not-decoded -->

Since for any θ ∈ Θ and g, ¯ g ∈ G r ( g 0 ) ,

<!-- formula-not-decoded -->

Similarly, we can show that

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

## B.4.5 Proof of Lemma 8

Proof. Let

<!-- formula-not-decoded -->

We consider the following loss:

<!-- formula-not-decoded -->

with the corresponding risk function defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Under Asm. 5, we have E P [ ϵ | W,X ] = 0 . Since τ 0 ( x ) = g (1) 0 ( x ) -g (0) 0 ( x ) , we have

<!-- formula-not-decoded -->

Thus, the target is the minimizer of the following squared loss:

<!-- formula-not-decoded -->

Differentiating ℓ ( θ, g ; z ) with respect to θ , we obtain the gradient and Hessian w.r.t. θ as

<!-- formula-not-decoded -->

The expected gradient and expected Hessian are then obtained as

<!-- formula-not-decoded -->

We consider the nuisance neighborhood such that for g ∈ G r ( g 0 ) ,

<!-- formula-not-decoded -->

We now verify that the loss function ℓ satisfies Asm. 3.

(a) We assume that g ( w ) : R d ↦→ R , w = 0 , 1 , and g prop : R d ↦→ (0 , 1) are continuous, thus Asm. 3(a) is satisfied.

(b) Since θ ⋆ is a global minimizer of (55), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(d) Consider the Taylor expansion around θ ⋆ , we have

<!-- formula-not-decoded -->

Let τ = g (1) -g (0) and

<!-- formula-not-decoded -->

Under Asm. 5, we have

<!-- formula-not-decoded -->

We can decompose S θ ( θ ⋆ , g ; Z ) as

<!-- formula-not-decoded -->

where which implies

For I 3 , which implies

For I 5 ,

<!-- formula-not-decoded -->

For I 1 , when ∥ X ∥ 2 ≤ C X a.s. , we have | τ 0 ( X ) -⟨ θ ⋆ , X ⟩| ≤ C for some C &gt; 0 and ∥ I 1 ∥ 2 ≤ C X ( | ( τ -τ 0 )( X ) | + C ) , which implies

<!-- formula-not-decoded -->

For I 2 , when ∣ ∣ ∣ Y -g ( W ) 0 ( X ) ∣ ∣ ∣ ≤ C Y a.s. , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

For I 4 , since ∣ ∣ g prop -g prop 0 ∣ ∣ ≤ 2 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies

For I 6 ,

When E P [ ϵ 2 ] ≤ σ 2 , we have

<!-- formula-not-decoded -->

For I 7 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies

For I 8 , which implies

For I 9 , which implies

<!-- formula-not-decoded -->

By Cauchy-Schwarz inequality, it follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since H θθ ( θ ⋆ , g ; Z ) -H θθ ( θ ⋆ , g ) ≼ C 2 X I , we have

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

which implies

Similarly, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(e) Since Y = W ( Y (1) -Y (0))+ Y (0) and g ( W ) ( X ) = W ( g (1) ( X ) -g (0) ( X )) + g (0) ( X ) , ϕ ( g ; Z ) can be written as

<!-- formula-not-decoded -->

Under Asm. 5, we have

<!-- formula-not-decoded -->

Thus, for τ = g (1) -g (0) and for any θ ∈ Θ and g, ¯ g ∈ G r ( g 0 ) such that ¯ g = tg + (1 -t ) g 0 for some t ∈ (0 , 1) , we have

<!-- formula-not-decoded -->

Since ( a + b ) 4 ≤ 8 a 4 +8 b 4 for a, b ∈ R , we have

<!-- formula-not-decoded -->

Similarly, we have

It is easy to show that

<!-- formula-not-decoded -->

and

Thus,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

In addition, L ( θ, g ) is Neyman orthogonal at ( θ ⋆ , g 0 ) since

<!-- formula-not-decoded -->

We have the higher-order derivative such that for any θ ∈ Θ and g, ¯ g ∈ G r ( g 0 ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

Similarly, we have

<!-- formula-not-decoded -->

Then we can show that

<!-- formula-not-decoded -->

which implies that

## B.4.6 Proof of Lemma 9

Proof. Define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where 1 ( · ) denotes the indicator function, and the log-linear predictor p θ ( x ) = e ⟨ θ,x ⟩ / (1 + e ⟨ θ,x ⟩ ) . Under Asm. 5,

<!-- formula-not-decoded -->

We consider the following loss:

<!-- formula-not-decoded -->

with the corresponding risk function defined as

<!-- formula-not-decoded -->

By (63), we have E P [ µ ( s ) g 0 ( Z ) | X ] = g ( s ) 0 ( X ) . Thus, the target is the minimizer of the following squared loss:

<!-- formula-not-decoded -->

Since ∇ θ p θ ( x ) = p θ ( x )(1 -p θ ( x )) x , we have

<!-- formula-not-decoded -->

Differentiating ℓ ( θ, g ; z ) with respect to θ , we obtain the gradient and Hessian w.r.t. θ as

S

H

θ

θθ

(

(

θ, g

θ, g

;

;

z

z

) =

-

) = (

[

µ

µ

(1)

g

(1)

g

(

z

(

z

)(1

) +

µ

-

(0)

g

p

(

θ

z

(

x

))

p

))

θ

(

x

x

-

µ

)(1

(0)

g

-

(

p

z

θ

)

(

p

x

θ

(

))

x

)

xx

]

x

⊤

.

The expected gradient and expected Hessian are then obtained as

<!-- formula-not-decoded -->

We consider the nuisance neighborhood such that for g ∈ G r ( g 0 ) ,

<!-- formula-not-decoded -->

Weassume that δ ≤ g (0) 0 ( X )+ g (1) 0 ( X ) ≤ δ -1 for δ &gt; 0 . In addition, we assume that for g ∈ G r ( g 0 ) , f (1) ( g ; X ) + f (0) ( g ; X ) ≥ δ a.s. . Note that

<!-- formula-not-decoded -->

We now verify that the loss function ℓ satisfies Asm. 3.

(a) We assume that g ( w ) : R d ↦→ R , w = 0 , 1 , and g prop : R d ↦→ (0 , 1) are continuous, thus Asm. 3(a) is satisfied.

,

(b) Since θ ⋆ is a global minimizer of (64), we have

<!-- formula-not-decoded -->

(c) We assume that Θ is bounded such that C ≤ p θ ( X ) ≤ 1 -C a.s. for some C &gt; 0 . When λ min ( E P [ XX ⊤ ] ) ≥ λ 0 &gt; 0 and ∥ X ∥ 2 ≤ C X a.s. , we have

<!-- formula-not-decoded -->

- (d) Consider the Taylor expansion around θ ⋆ , we have

S θ ( θ, g ; Z ) -S θ ( θ, g ) = S θ ( θ ⋆ , g ; Z ) -S θ ( θ ⋆ , g ) + ( H θθ ( θ ⋆ , g ; Z ) -H θθ ( θ ⋆ , g ))( θ -θ ⋆ ) . Note that

<!-- formula-not-decoded -->

For s = 1 , when Y (1) -g (1) 0 ( X ) ≤ C Y a.s. for C Y &gt; 0 , we have

<!-- formula-not-decoded -->

Thus,

Since

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

∥ Similarly, we can show that

<!-- formula-not-decoded -->

It follows that

<!-- formula-not-decoded -->

- (e) For s = 1 , we have

<!-- formula-not-decoded -->

Similarly, for s = 0 ,

<!-- formula-not-decoded -->

For any ¯ g, g ∈ G r ( g 0 ) such that ¯ g = tg +(1 -t ) g 0 for some t ∈ (0 , 1) , we have

<!-- formula-not-decoded -->

Note that by (61),

<!-- formula-not-decoded -->

In addition,

<!-- formula-not-decoded -->

Thus, it is easy to show that

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

In addition, L ( θ, g ) is Neyman orthogonal at ( θ ⋆ , g 0 ) since

<!-- formula-not-decoded -->

Now we compute the the higher-order derivative. For s = 1 , we have

<!-- formula-not-decoded -->

Similarly, for s = 0 ,

<!-- formula-not-decoded -->

Then for any θ ∈ Θ and g, ¯ g ∈ G r ( g 0 ) such that ¯ g = tg +(1 -t ) g 0 for some t ∈ (0 , 1) ,

<!-- formula-not-decoded -->

By (61), we have

<!-- formula-not-decoded -->

In addition,

<!-- formula-not-decoded -->

Together we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

## C Convergence Proofs for Stochastic Gradient

This section is dedicated to demonstrate the SGD convergence in Thm. 1 from Sec. 3 of the main text using Asm. 3 and Asm. 4. We first give an overview of the problem settings and the expected results with the proof outline in Appx. C.1. We then provide all the technical lemmas needed for Thm. 1 in Appx. C.2, and finally prove our first main result in Appx. C.3.

## C.1 Overview

In this section, we demonstrate the convergence of SGD for a risk minimization problem with nuisance:

<!-- formula-not-decoded -->

where g 0 ∈ G is the true nuisance, L ( θ, g ) = E Z ∼ P [ ℓ ( θ, g ; Z )] , and ℓ is a prespecified loss function. We consider the stochastic gradient method for learning θ ⋆ when g 0 is unknown but an estimate ˆ g is accessible. Define D n = ( Z 1 , . . . , Z n ) , sampled from the product measure P n . Recall the SGD θ ( n ) defined as

<!-- formula-not-decoded -->

Throughout the section, we take the following notations for simplicity:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where S θ ( θ, g ; z ) = ∇ θ ℓ ( θ, g ; z ) is the gradient and S θ ( θ, g ) = E Z ∼ P [ S θ ( θ, g ; Z )] is the population gradient. We are interested in the mean squared error using an estimated nuisance ˆ g , and our results show that for non-Neyman orthogonal loss L , the error δ ( n ) satisfies

<!-- formula-not-decoded -->

where the nuisance estimator ˆ g would lead to a bias of order O ( ∥ ˆ g -g 0 ∥ 2 G ) for the SGD convergence. If L is Neyman orthogonal, this bias introduced by the nuisance estimation would be further removed, resulting in the following convergence

<!-- formula-not-decoded -->

Proof Outline. The proofs for both results (74) and (75) proceed through the following four steps:

1. Upper bound the excess risk L ( θ ( n ) , ˆ g ) -L ( θ ⋆ , ˆ g ) in terms of the SGD improvement.
2. Lower bound L ( θ ( n ) , ˆ g ) -L ( θ ⋆ , ˆ g ) using strong convexity and Neyman orthogonality.
3. Derive a recursive formula of E D n ∼ P n [ ∥ δ ( n ) ∥ 2 2 ] from these bounds.
4. Perform the recursion and obtain the final result.

Follow these steps above, we provide technical lemma in Appx. C.2, and then prove our first main result Thm. 1 in Appx. C.3.

## C.2 Technical Lemma

Lemma 10 (One-step improvement for SGD) . Suppose that Asm. 3 holds. If η &lt; 1 /M , θ ( n ) ∈ Θ , and ˆ g ∈ G r ( g 0 ) , it holds that

<!-- formula-not-decoded -->

Proof. We first define the η -1 -strongly convex function f n as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies that θ ( n ) = θ ( n -1) -ηS ( n ) is the global minimizer of (76) and ∇ θ f n ( θ ( n ) ) = 0 . Then

<!-- formula-not-decoded -->

Since L ( · , ˆ g ) is µ -strongly convex and f n ( θ ⋆ ) = ⟨ S ( n ) , -δ ( n -1) ⟩ +(2 η ) -1 ∥ δ n -1 ∥ 2 2 , we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Together with (77), it follows that

<!-- formula-not-decoded -->

Since L ( · , ˆ g ) is M -smooth and f n ( θ ( n ) ) = ⟨ S ( n ) , θ ( n ) -θ ( n -1) ⟩ +(2 η ) -1 ∥ θ ( n ) -θ ( n -1) ∥ 2 2 , we have that

<!-- formula-not-decoded -->

By (78), it follows that

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

For any ω &gt; 0 , by Cauchy-Schwarz inequality and Young's inequality, we have

<!-- formula-not-decoded -->

Note that

Take this into (79) and we have

<!-- formula-not-decoded -->

When η ≤ 1 /M , set M 2 -1 2 η + 1 2 ω = 0 , i.e., set ω = 1 / ( η -1 -M ) . It follows that

<!-- formula-not-decoded -->

We complete the proof by multiplying both sides of the inequality by 2 η .

Lemma 11. Suppose that Asm. 3 holds. If η &lt; 1 /M , θ ( n ) ∈ Θ , and ˆ g ∈ G r ( g 0 ) , it holds that

<!-- formula-not-decoded -->

Proof. Under Asm. 3,

<!-- formula-not-decoded -->

Since L ( · , ˆ g ) is µ -strongly convex,

<!-- formula-not-decoded -->

By (80), it follows that

<!-- formula-not-decoded -->

Together with Lemma 10, we have

<!-- formula-not-decoded -->

Rearranging it, we have

<!-- formula-not-decoded -->

By Young's inequality,

<!-- formula-not-decoded -->

Take this into (81) and we have

<!-- formula-not-decoded -->

Corollary 12. Suppose that Asm. 3 holds. If η &lt; 1 /M , θ ( n ) ∈ Θ , and ˆ g ∈ G r ( g 0 ) , it holds that

<!-- formula-not-decoded -->

Proof. Note that E Z n ∼ P [ ⟨ v ( n ) , δ ( n -1) ⟩ ] = 0 . By Lem. 11, we have

<!-- formula-not-decoded -->

Under Asm. 3, E Z n ∼ P [ ∥ v ( n ) ∥ 2 2 ] ≤ K 1 + κ 1 ∥ δ ( n ) ∥ 2 2 , and it follows that

<!-- formula-not-decoded -->

Lemma 13. Suppose that Asm. 3 and Asm. 4 hold. If η &lt; 1 /M , θ ( n ) ∈ Θ , and ˆ g ∈ G r ( g 0 ) , it holds that

<!-- formula-not-decoded -->

Proof. Under Asm. 3 and Asm. 4,

<!-- formula-not-decoded -->

The rest of the proof is similar to Lem. 11.

Corollary 14. Suppose that Asm. 3 and Asm. 4 hold. If η &lt; 1 /M , θ ( n ) ∈ Θ , and ˆ g ∈ G r ( g 0 ) , it holds that

<!-- formula-not-decoded -->

Proof. The proof is similar to Cor. 12 using Lem. 13.

## C.3 Proof of Theorem 1

Proof. Let c ( η ) = µ -κ 1 η/ (1 -Mη ) . When η &lt; µ/ ( Mµ + κ 1 ) , we have

<!-- formula-not-decoded -->

Under Asm. 3 and by Cor. 12, we have that

<!-- formula-not-decoded -->

By recursion, it follows that

E D n ∼ P n [ ∥ δ ( n ) ∥ 2 2 ] ≤ (1 -c ( η ) η ) n ∥ δ (0) ∥ 2 2 + α 2 1 η µ ∥ ˆ g -g 0 ∥ 2 G n -1 ∑ i =0 (1 -c ( η ) η ) i + K 1 η 2 1 -Mη n -1 ∑ i =0 (1 -c ( η ) η ) i ≤ (1 -c ( η ) η ) n ∥ δ (0) ∥ 2 2 + α 2 1 µc ( η ) ∥ ˆ g -g 0 ∥ 2 G + K 1 η c ( η )(1 -Mη ) . If η ≤ µ/ 2( Mµ + κ 1 ) , we have µ/ 2 ≤ c ( η ) ≤ µ . Thus, .

<!-- formula-not-decoded -->

In addition, if Asm. 4 holds, then by Cor. 14 and using identical proof as above, it follows that for a Neyman orthogonal risk L ,

<!-- formula-not-decoded -->

## D Orthogonalization with respect to Nuisance

In this section, we establish our orthogonalization method for the possibly infinite-dimensional nuisance introduced in Sec. 3 of the main text. We demonstrate how we construct the orthogonalizing operator in Appx. D.1, and provide all the technical lemmas in Appx. D.2.

## D.1 Orthogonalization via Riesz Representation

We consider G ≡ ( G , ⟨· , ·⟩ G ) as a possibly infinite-dimensional Hilbert space. Recall the derivative operator D g defined as for any h ∈ G ,

<!-- formula-not-decoded -->

This derivative operator is also known as the Gateaux derivative. We posit the usual assumption as Jordan et al. [2022] that the derivative operator D g ℓ ( θ, g ; z ) is linear and continuous in G for any ( θ, g, z ) ∈ Θ × G r ( g 0 ) × Z . We also assume regularity conditions such that D g D θ ℓ ( θ, g ; z ) is continuous and D g D θ ℓ ( θ, g ; z ) = D θ D g ℓ ( θ, g ; z ) at any ( θ, g, z ) .

Since D g ℓ ( θ, g ; z ) is linear and continuous, by the Riesz representation theorem, there uniquely exists some ∇ g ℓ ( θ, g ; z ) ∈ G such that for any g ∈ G ,

<!-- formula-not-decoded -->

Lem. 16 shows that the operator D g S θ ( θ ⋆ , g 0 ) is linear and continuous. By Riesz representation theorem, we can define H θg = ( H (1) θg , . . . , H ( d ) θg ) ∈ G d such that for all g ∈ G ,

<!-- formula-not-decoded -->

The Hessian operator H gg : G ↦→ G is defined as for any g 1 , g 2 ∈ G ,

<!-- formula-not-decoded -->

We will show in Lem. 17 that H gg uniquely exists and is an self-adjoint and bounded linear operator when D 2 g L ( θ ⋆ , g 0 ) is bounded and symmetric bilinear. Assuming that H gg is invertible, we define the orthogonalizing operator as

<!-- formula-not-decoded -->

We now construct the Neyman orthogonalized (NO) gradient oracle

<!-- formula-not-decoded -->

In addition, Γ 0 ∇ g ℓ ( θ, g ; z ) can be written as the derivative in the sense that for each j = 1 , . . . , d ,

<!-- formula-not-decoded -->

That is, the NO gradient oracle can be easily obtain by

<!-- formula-not-decoded -->

The following Lemma shows that S no ( θ, g ; z ) is Neyman orthogonal at ( θ ⋆ , g 0 ) .

Lemma 15. S no ( θ, g ; z ) is a Neyman orthogonal score at ( θ 0 , g 0 ) .

<!-- formula-not-decoded -->

## D.2 Technical Lemma

Lemma 16. D g S θ ( θ, g ; z ) : G ↦→ R d and D g S θ ( θ, g ) : G ↦→ R d are linear and continuous in G .

Proof. The continuity of D g S θ ( θ, g ; z ) and D g S θ ( θ, g ) follows from the continuity of D g D θ ℓ ( θ, g ; z ) . It suffices to prove that D g S θ ( θ, g ; z ) is linear. For all u ∈ R d , h 1 , h 2 ∈ G ,

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 17. Suppose that D 2 g L ( θ ⋆ , g 0 )[ · , · ] : G ×G ↦→ R is a bounded and symmetric bilinear form. Then H gg : G ↦→ G uniquely exists and is self-adjoint, bounded, and linear.

Proof. Given g 1 , g 2 ∈ G , since D 2 g L ( θ ⋆ , g 0 )[ g 1 , · ] is a bounded linear map from G to R , by Riesz representation theorem, for any g 2 ∈ G , there uniquely exists some Th 1 ∈ G such that

<!-- formula-not-decoded -->

Thus, we define the operator T : G ↦→ G . Note that D 2 g L ( θ ⋆ , g 0 )[ · , · ] is bilinear. For any a, a ′ ∈ R , and any g 1 , g ′ 1 , g 2 ∈ G , we have

<!-- formula-not-decoded -->

̸

which implies T is a linear operator. To show T is bounded, suppose that the norm of the bilinear form D 2 g L ( θ ⋆ , g 0 ) is bounded by B . Thus, for Tg 1 = 0 ,

<!-- formula-not-decoded -->

which implies T is bounded. Note that D 2 g L ( θ ⋆ , g 0 )[ · , · ] is symmetric, we have T being self-adjoint since

<!-- formula-not-decoded -->

Finally, we show that T is unique. If there exists some T ′ : G ↦→ G such that for any g 1 , g 2 ∈ G ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

That is, T = T ′ . We finish the proof by letting H gg = T .

which implies

## E Convergence Proofs for Orthogonalized Stochastic Gradient

This section is dedicated to demonstrate the OSGD convergence in Thm. 3 of the main text. In Appx. E.1, we give an overview of the OSGD settings, the additional assumptions, and the expected results with the proof outline. We then provide all the technical lemmas in Appx. E.2, and finally prove Thm. 3 in Appx. E.3.

## E.1 Overview

Following the same problem settings in Appx. C, we consider the orthogonalized SGD (OSGD) using the estimated NO score ˆ S no oracle defined as

<!-- formula-not-decoded -->

where ˆ Γ is an estimator for the orthogonalizing operator defined in (85). Specifically, we consider all continuous linear ˆ Γ : G ↦→ R d for estimating the orthogonalizing operator Γ 0 . By Riesz representation theorem, there exists some ˆ γ ( j ) ∈ G for j = 1 , . . . , d, such that

<!-- formula-not-decoded -->

For the orthogonalizing operator Γ 0 , we define γ ( j ) 0 = H -1 gg H ( j ) θg , j = 1 , . . . , d, such that

<!-- formula-not-decoded -->

We focus on the OSGD defined below:

<!-- formula-not-decoded -->

Throughout the section, we take the following notations for simplicity:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let ∇ g L ( θ, g ) = E Z ∼ P [ ∇ g ℓ ( θ, g ; Z )] and S no ( θ, g ) = E Z ∼ P [ S no ( θ, g ; Z )] . We need the following assumptions to establish the convergence result of the OSGD.

Assumption 6. The following conditions hold:

- (a) First-order optimality: S no ( θ ⋆ , g 0 ) = 0 and ( ˆ Γ -Γ 0 ) ∇ g L ( θ ⋆ , g 0 ) = 0 .
- (b) Smoothness and strong convexity: There exists some M no , µ no &gt; 0 such that for all θ ∈ Θ and g ∈ G r ( g 0 ) , ∥∇ θ S no ( θ, g ) ∥ 2 ≤ M no and

<!-- formula-not-decoded -->

- (c) Second-moment growth: There exist constants K 2 , κ 2 &gt; 0 such that

<!-- formula-not-decoded -->

for all θ ∈ Θ , ¯ g ∈ G r ( g 0 ) , and g ∈ G .

- (d) Second-order smoothness: There exists a constant α 2 &gt; 0 such that

<!-- formula-not-decoded -->

- (e) Higher-order smoothness: There exists a constants β 2 &gt; 0 such that

<!-- formula-not-decoded -->

Asm. 6(a) is necessary for the convergence of the OSGD to θ ⋆ . When S θ is Neyman orthogonal at ( θ ⋆ , g 0 ) , Γ 0 = 0 is accessible and thus, S no = S θ . When S θ is non-orthogonal, Asm. 6(a) can be satisfied whenever ∇ g L ( θ ⋆ , g 0 ) = 0 , implying that ( θ ⋆ , g 0 ) is a local minimizer of L ( θ, g ) . Asm. 6(b) is related to the Schur complement of the population Hessian. Thus, the hypothetical objective relating to S no inherits its strong convexity from that of the population risk L w.r.t. ( θ, g ) ∈ Θ × G r ( g 0 ) when G is finite-dimensional; see Boyd and Vandenberghe [2004]. Asm. 6(c) and (d) are exactly analogous to Asm. 3(d) and (e), while Asm. 6(e) is analogous to Asm. 4.

With Asm. 6, we aim to show that the error δ ( n ) no satisfies

<!-- formula-not-decoded -->

Proof Outline. The proof the (92) follows the following four steps:

1. Upper bound ∥ I -η ∇ θ ˆ S no ( θ, g ) ∥ 2 w.r.t. the operator estimation error ∥ ˆ Γ -Γ 0 ∥ Fro .
2. Upper bound ∥ ˆ S no ( θ ⋆ , ˆ g ) ∥ 2 using Neyman orthogonality and the first order optimality.
3. Derive a recursive formula of E D n ∼ P n [ ∥ δ ( n ) no ∥ 2 2 ] from these bounds.
4. Perform the recursion and obtain the final result.

Follow these steps above, we provide technical lemma in Appx. E.2, and then prove our second main result Thm. 3 in Appx. E.3.

Alternatively, the intuition of step 1 also suggests that we should focus on ˆ Γ that lies in the neighborhood of Γ 0 such that ∥ ˆ Γ -Γ 0 ∥ Fro ≤ R for a small R &gt; 0 . Then, instead of assuming Asm. 6(b), we can directly assume that for all θ ∈ Θ and g ∈ G r ( g 0 ) , ∥∇ θ ˆ S no ( θ, g ) ∥ 2 ≤ M no and

<!-- formula-not-decoded -->

With this assumption, one can still show the same OSGD convergence rate by the identical proof while the constraint of the learning rate η will be simplified.

## E.2 Technical Lemma

Lemma 18. Given η &gt; 0 . For any ω &gt; 0 and u, v ∈ R d ,

<!-- formula-not-decoded -->

Proof. By definition,

<!-- formula-not-decoded -->

By Young's inequality, for any ω &gt; 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 19. Suppose that Asm. 6 holds. For all θ ∈ Θ and g ∈ G r ( g 0 ) ,

<!-- formula-not-decoded -->

Proof. Note that

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

We now bound ∥∇ θ ( ˆ Γ -Γ 0 ) ∇ g L ( θ, g ) ∥ 2 . For each j = 1 , . . . , d , for any θ ∈ R d ,

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

which implies that ∥∇ θ ( ˆ Γ -Γ 0 ) ∇ g L ( θ, g ) ∥ 2 ≤ α 1 ∥ ˆ Γ -Γ 0 ∥ Fro and

<!-- formula-not-decoded -->

Additionally, we have

<!-- formula-not-decoded -->

In conclusion, we have

<!-- formula-not-decoded -->

Lemma 20. Suppose that Asm. 6 holds. When ˆ g ∈ G r ( g 0 ) ,

<!-- formula-not-decoded -->

Proof. Note that

<!-- formula-not-decoded -->

Since S no ( θ ⋆ , g 0 ) = 0 and S no is Neyman orthogonal at ( θ ⋆ , g 0 ) , we have for some ¯ g ∈ G r ( g 0 ) ,

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

Similarly, since ( ˆ Γ ( j ) -Γ 0 ) ∇ g L ( θ ⋆ , g 0 ) = 0 , we have for some ¯ g ′ ∈ G r ( g 0 ) ,

<!-- formula-not-decoded -->

which implies

In conclusion,

<!-- formula-not-decoded -->

## Lemma 21. Suppose that Asm. 3 and Asm. 6 holds. Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Asm. 3,

Since

In conclusion,

<!-- formula-not-decoded -->

## E.3 Proof of Theorem 3

Proof. Since θ ( n ) = θ ( n -1) -η ˆ S ( n ) no , by Taylor's theorem we have that for some ¯ θ ( n -1) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that E Z n ∼ P [ v ( n ) no ] = 0 . Take the expectation of the squared norm of both sides w.r.t. Z n and we have

<!-- formula-not-decoded -->

By Lem. 18, Lem. 19, and Lem. 20, for any ω &gt; 0 ,

<!-- formula-not-decoded -->

Set ω = µ no. For η ≤ 2 /µ no, we have

<!-- formula-not-decoded -->

where b ( η ) = µ no -2 α 1 ∥ ˆ Γ -Γ 0 ∥ Fro -(6 M 2 no -2 µ 2 no +2 µ no α 1 ∥ ˆ Γ -Γ 0 ∥ Fro +12 α 2 1 ∥ ˆ Γ -Γ 0 ∥ 2 Fro ) η. By Lem. 21,

<!-- formula-not-decoded -->

It follows that

<!-- formula-not-decoded -->

Thus, it is clear that when ∥ ˆ Γ -Γ 0 ∥ Fro ≤ µ no / (4 α 1 ) and the learning rate satisfies

<!-- formula-not-decoded -->

we have 1 -µ no η/ 2 ≥ 0 and

<!-- formula-not-decoded -->

When η satisfies (93), it follows that

<!-- formula-not-decoded -->

Finally, perform the same recursion in Appx. C.3 and we have

<!-- formula-not-decoded -->

## F Detailed Discussion

This section provides details on comparisons and remarks following the statements of the main results in the main text, and details on the discussions summarized in Sec. 4 from the main text. In Appx. F.1, we compare our results to generic state-of-the-art results on biased SGD, that is SGD with errors in the stochastic gradients. In Appx. F.2, we discuss different orthogonalization method in orthogonal statistical learning. In Appx. F.3, we discuss how to interleave the target and nuisance estimation. In Appx. F.4, we describe the connection between our orthogonalized gradient to variance reduction method in the Monte Carlo estimation literature. In Appx. F.5, we discuss the double robustness of SGD for dose-response estimation.

## F.1 Comparison to Biased SGD

There are several ways to think about the bias induced by using an imperfect estimate ˆ g as opposed to the true nuisance g 0 ∈ G . For the sake of discussion, we will define L ( · , g 0 ) and L ( · , ˆ g ) as the 'original objective' and 'shifted objective', respectively. Accordingly, we will call θ ⋆ the 'original minimizer' and denote by

<!-- formula-not-decoded -->

the 'shifted minimizer'. The bias can be measured in terms of (i) the error ∥ ˆ θ ⋆ -θ ⋆ ∥ 2 2 between the original and shifted minimizers, (ii) the uniform error sup θ | L ( θ, g 0 ) -L ( θ, ˆ g ) | between the original and shifted objectives, and (iii) some summary of the gradient bias

<!-- formula-not-decoded -->

of the oracle S (a vector-valued quantity) for step t = 1 , . . . , n of the algorithm. Whether one appeals to (i) or (ii) depends on whether the convergence guarantees are stated in terms of iterate convergence or function value convergence; because we analyze convergence of iterates, our discussion will cover (i) and (iii).

On (i), one applies the decomposition

<!-- formula-not-decoded -->

and plugs an analysis of unbiased SGD from the current literature for the ∥ θ ( n ) -ˆ θ ⋆ ∥ 2 2 term. The purpose of this substitution is to check how our theoretical results align with the known results on unbiased SGD .

Bach and Moulines [2011, Thm. 1] show that for constant learning rate η = O ( µ/M 2 ) , the iterate θ ( n ) satisfies

<!-- formula-not-decoded -->

Cutler et al. [2023, Thm. 3] demonstrate that with the learning rate η = O (1 /M ) , the iterates would satisfy the following bound:

<!-- formula-not-decoded -->

In addition, Cutler et al. [2023, Thm. 6] provide the high probability bound of θ ( n ) that for η = O (1 /M ) , with probability at least 1 -δ ,

<!-- formula-not-decoded -->

All of these bounds essentially agree, as we may apply (1 -µη/ 2) ≤ exp( -µη/ 2) . In comparison to our Thm. 1, our bias term is stated directly in terms of the nuisance error ∥ ˆ g -g 0 ∥ 2 G . This can be viewed as a refinement of the less transparent bias measurement ∥ ˆ θ ⋆ -θ ⋆ ∥ 2 2 . Moreover, although (96)-(98) are of the same order as our results in Thm. 1 when the true nuisance g 0 is

available, all of the three bounds above require κ 1 = 0 in Asm. 3(d). In this case, provide (97) and (98) use a learning rate of the order O (1 /M ) (whereas the learning rate of (96) encounters an additional condition number M/µ ). Our learning rate recovers O (1 /M ) when κ 1 = 0 , and adapts via the setting η = O ( µ/ ( Mµ + κ 1 )) when κ 1 &gt; 0 . Finally, the high probability bound (98) requires a stronger assumption in the sense that S θ ( θ, g ; Z ) -S θ ( θ, g ) is sub-Gaussian with uniform parameter K 1 / 2 for all θ ∈ Θ and g ∈ G r ( g 0 ) .

Returning to the bias in the stochastic gradient oracle (95), this case is handled quite generally in Demidovich et al. [2023]. Their 'ABC assumption' considers constants A,B,C,b,c ≥ 0 such that the inequalities

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

hold for all θ ∈ R d (where the expectations are conditional on any randomness in ˆ g ). 3 The bias is really captured in the first of the three inequalities, whereas the third inequality places conditions on the parameters of the problem that are not in the hands of the algorithm user. By strong convexity, our Asm. 3(d) satisfies (100) with A = κ 1 /µ , B = 0 , C = K 1 . The resulting convergence guarantee [Demidovich et al., 2023, Thm. 5] gives

<!-- formula-not-decoded -->

for a learning rate set as

<!-- formula-not-decoded -->

We would like to check how our theoretical results compare with the above application of generic results on biased SGD . In our setting, the condition (101) reads κ 1 /µ + M (1 -2 b ) &lt; µ/ 2 . We assume neither this nor (99), and in our Thm. 1, replace the K 1 + c µ 2 term with a bias term that depends directly on the nuisance error ∥ ˆ g -g 0 ∥ 2 G , either in the nuisance sensitive regime, or in the nuisance insensitive regime.

## F.2 Discussion of Full-sample Orthogonal Statistical Learning and Related Methods

Comparison of Orthogonalizing Operators. Constructing orthogonal losses or scores has been widely studied in semiparametric inference, hypothesis testing, and machine learning. In semiparametric statistics, such constructions often rely on the efficient influence function, which characterizes the asymptotic efficient estimation bound; see Bickel et al. [1993, Ch. 3], Tsiatis [2006, Ch. 4], Van der Vaart [2000, Ch. 25], Luedtke and Chung [2024]. In hypothesis testing, orthogonal scores were used by Neyman [1959, 1979] and Ferguson [2014] to guarantee the local unbiasedness of specific tests based on the likelihood with finite-dimensional nuisance. In machine learning, the construction of orthogonal scores was latter extended to non-likelihood losses in Wooldridge [1991] and Liu et al. [2022], which aligns with our construction limited to the finite-dimensional nuisance case. Recent work of orthogonalization in machine learning with infinite-dimensional nuisance relies on the approach named concentrating-out [Newey, 1994, Chernozhukov et al., 2018a]. However, although all these constructions produce Neyman orthogonal losses or scores, none of them consider the stochastic design. Our work is complementary to these, providing non-asymptotic guarantees for stochastic optimization.

Although these constructions might lead to different orthogonal scores, they can be the same at both the target and the true nuisance. Specifically, when ℓ is the negative log-likelihood and G = R k , the concentrating-out approach and our NO gradient oracle S no both produce the efficient score in the semiparametric theory literature; see Newey [1994, Page 1359], Van der Vaart [2000, Ch. 25.4], and Tsiatis [2006, Def. 8]. This identity can happen for infinite-dimensional nuisances as well. As

3 The third inequality is actually A + M ( B +1 -2 b ) &lt; µ , but the constant (1 / 2) to make the resulting bound more comparable, in that their bound can only improve over ours for the stronger inequality.

an example, consider the partial linear model from Appx. B.1.2, where the non-orthogonal loss is defined as

<!-- formula-not-decoded -->

Chernozhukov et al. [2018a, Sec. 2.2.2] showed that the concentrating-out approach would produce an orthogonal score under the concentrated-out nuisance φ 0 ( θ ) = Z ↦→ E P [ Y -⟨ θ, X ⟩ | W ] as

<!-- formula-not-decoded -->

On the other hand, it is easy to verify that H gg = I , H θg = E P [ X | W ] , and Γ 0 ∇ g ℓ ( θ, g ; z ) = D g ℓ ( θ, g ; z )[ H -1 gg H θg ] , which implies that our orthogonal gradient oracle S no in (11) has the same form under the target θ ⋆ and true nuisance g 0 ( W ) = E P [ Y -⟨ θ ⋆ , X ⟩ | W ] :

<!-- formula-not-decoded -->

Comparison with Debiased Machine Learning. In machine learning, debiasing typically refers to reducing the impact of model selection error on the parameter or quantity of interest. In particular, mitigating the bias introduced by nuisance estimation is known as debiased machine learning (DML), which has been recently studied by van der Laan et al. [2011], Shi et al. [2019], Chernozhukov et al. [2024], van der Laan et al. [2025]. Some of the calculations used by DML estimators have been shown to be amenable to computerization, simplifying their construction [Carone et al., 2019, Ichimura and Newey, 2022, Luedtke, 2024]. Statistical learning methods that use debiasing are also called orthogonal statistical learning (OSL) and have been studied in Foster and Syrgkanis [2023], Liu et al. [2022], Zadik et al. [2018]. While the earlier studies focus on the empirical risk minimization, our paper provide a stochastic approximation method in DML/OSL and establish the convergence rate of the debiased estimation.

To strengthen the debiasing effect, one possible approach is to consider the higher-order Neyman orthogonality. If the loss function satisfies the k -th order orthogonality at ( θ ⋆ , g 0 ) , Zadik et al. [2018, Cor. 4] show that we only need the nuisance estimator to converge at rate O p ( n -1 2( k +1) ) to have the nuisance effect in the order of O p ( n -1 ) , which aligns with the nuisance insensitive rate in Thm. 1, where k = 1 and the nuisance effect ∥ ˆ g -g 0 ∥ 4 G = O p ( n -1 ) when ∥ ˆ g -g 0 ∥ G = O p ( n -1 / 4 ) . Similar improvements in sensitivity to nuisance estimation rates have been developed previously using higher-order influence functions [Pfanzagl, 1985, Robins et al., 2008].

For a range of problems, debiasing methods often lead to cross-product estimations consisting of two nuisance estimators [Rotnitzky et al., 2021]. Such remainders frequently result from orthogonalization procedures used in missing data problems and causal inference problems [Robins et al., 1994, Robins and Rotnitzky, 1995, Laan and Robins, 2003]. Chernozhukov et al. [2024] consider cases where Z = ( W,Y ) , g 0 ( W ) = E P [ Y | W ] , and the target can be written as the averaged moment of the form

<!-- formula-not-decoded -->

where E [ m ( g ; W )] : G × W ↦→ R d is a continuous linear functional of g : W ↦→ R . By Riesz representation theorem, there uniquely exists a Riesz representer (RR) g RR 0 ∈ G such that E P [ m ( g ; W )] = E P [ g RR 0 ( W ) g ( W )] . Then the debiased score for estimating θ ⋆ is defined as

<!-- formula-not-decoded -->

The debiasing effect on the nuisance turns out depending on the cross-product ∥ ˆ g RR -g RR 0 ∥ G · ∥ ˆ g -g 0 ∥ G . Specifically, Chernozhukov et al. [2024, Asm. 4] requires the cross product to be in the order O p ( n -1 / 2 ) to construct O p ( n -1 ) consistent target estimator. This aligns with the cross-product ∥ ˆ g -g 0 ∥ G · ∥ ˆ Γ -Γ 0 ∥ Fro in Thm. 3 where the same requirement needs to be satisfied to obtain a O p ( n -1 / 2 ) consistent estimator. However, Thm. 3 also has a second, non-cross-product remainder ∥ ˆ g -g 0 ∥ 4 G that will only be small if ˆ g approximates g 0 , making it so that our consistency guarantee is robust to misspecification of ˆ Γ , but not to misspecification of ˆ g .

## F.3 Discussion of Interleaving Target and Nuisance Estimation

To propose the interleaving approach, we consider the case where we learn the nuisance from the W -valued data W = ( U, V ) from a probability measure Q . We assume that the true nuisance g 0 satisfies g 0 : U ↦→ R and is the minimizer of the mean squared error over G :

<!-- formula-not-decoded -->

Suppose that we observe another data stream W 1 , . . . , W m sampled i.i.d. from Q , and that S m = { W 1 , . . . , W m } is independent of the parameter stream D n . We define the sigma algebra H m = σ ( S m ) , m ≥ 1 as the nuisance filtration and the sigma algebra F m,t = σ ( S m ∪ D ( m -1) n + t ) , 0 ≤ t ≤ n as the parameter filtration. We assume that there are two stochastic processes ˆ g ( m ) , m ≥ 1 adapted to H m and θ ( m,t ) , 0 ≤ t ≤ n adapted to F m,t , to which we refer as the nuisance estimator and the parameter estimator, respectively. Intuitively, this means that the nuisance estimator ˆ g ( m ) can be updated now instead of being the fixed ˆ g , and the parameter estimator θ ( m,t ) can be updated n times between every two nuisance updates. Specifically, we use SGD as the parameter estimator. We define θ (0 ,n ) = θ (0) ∈ Θ and θ ( i, 0) = θ ( i -1 ,n ) for 1 ≤ i ≤ m , and produce the sequence θ ( i, 1) , . . . , θ ( i,n ) using n steps of the SGD update (8) initialized at θ ( i, 0) .

Under Non-orthogonality. Consider the case that G is a reproducing kernel Hilbert space (RKHS) with kernel k ( · , · ) . To obtain a sequence of nuisance estimator ˆ g ( m ) on H m , one possible approach is to adopt the non-parametric stochastic approximation. With the assumption that the eigenvalues ( λ j ) j ≥ 1 of covariance operator E Q [ k ( W, · ) ⊗ k ( W, · )] decay polynomially at order j -α , Dieuleveut and Bach [2016, Cor. 3] suggests that the non-parametric stochastic approximation ˆ g ( m ) satisfies for some C &gt; 0 ,

<!-- formula-not-decoded -->

This leads to the following nuisance sensitive rate for non-Neyman orthogonal losses.

Proposition 22. Suppose that ˆ g ( m ) satisfies (102) and that ˆ g ( m ) ∈ G r ( g 0 ) and θ ( m,t ) ∈ Θ almost surely for all m ≥ 1 and 0 ≤ t ≤ n . Under the same conditions to Thm. 1, it holds that

<!-- formula-not-decoded -->

In addition, when ( ηn ) -1 = O (1) , it holds that

<!-- formula-not-decoded -->

The proof is provided in Appx. F.6. Prop. 22 demonstrates that interleaving the target and nuisance estimation allows η ≍ n -1 since the nuisance update iterations guarantees the shrinking of the term (1 -µη/ 2) mn in this case. This is an improvement to Thm. 1 where η should satisfy ( ηn ) -1 = o (1) to ensure (1 -µη/ 2) n shrinking to zero.

Under Orthogonalized SGD. To establish a similar probability bound for OSGD, we assume that the orthogonalizing operator Γ 0 can be written as the minimizer of the following program:

<!-- formula-not-decoded -->

where G ∗ is the dual space of G . When d is fixed, we assume that Γ 0 can be estimated (coordinatewisely) from the data stream S m using the stochastic approximation of Dieuleveut and Bach [2016], which leads to a sequence of operator estimators ˆ Γ ( m ) , m ≥ 1 . For any s &gt; 0 , we define the

following events for i = 0 , 1 , . . . , m ,

<!-- formula-not-decoded -->

We assume that for some constant c ≥ 1 the nuisance estimator ˆ g ( i ) satisfies

<!-- formula-not-decoded -->

Additionally, we assume that ˆ Γ ( i ) decays in the same rate such that

<!-- formula-not-decoded -->

With all the assumptions above, it is possible to provide a convergence bound of ∥ θ ( m,n ) -θ ⋆ ∥ 2 2 in probability. The following proposition shows that estimations from S m using OSGD contribute to a nuisance insensitive rate of O ( m -2 α -1 α ) , compared to the nuisance sensitive rate O ( m -2 α -1 2 α ) in Prop. 22 for non-Neyman orthogonal losses.

Proposition 23. Suppose that { ˆ g ( m ) , m ≥ 1 } satisfies (103) , and that { ˆ Γ ( m ) , m ≥ 1 } satisfies (104) . Assume that θ ( m,t ) ∈ Θ almost surely for all m ≥ 1 and 0 ≤ t ≤ n . For any s ≥ 0 , define δ ( s ) = O ( ms ) as (109) . Under the same conditions to Thm. 1, with probability at least 1 -δ ( s ) , it holds that

<!-- formula-not-decoded -->

In addition, when ( ηn ) -1 = O (1) , with probability at least 1 -δ ( s ) , it holds that

<!-- formula-not-decoded -->

We refer the reader to Appx. F.7 for the proof.

## F.4 Interpretation as Control Variate for Variance Reduction

The regression equation (9), which provides an alternate characterization of the orthogonalized stochastic gradient oracle in the case of negative log-likelihood losses, yields an interesting connection to the Monte Carlo estimation literature. Variance reduction techniques (or 'swindles') are used in problems such as estimating the mean or variance of a statistic via Monte Carlo simulation. Consider a probability space (Ω , F , P ) with expectation denoted by E and an unknown vector-valued target v ∈ R d . We have ˆ v : Ω → R d , where we interpret ˆ v as a (not necessarily unbiased) sample estimate of v . Several variance reduction techniques fall into the category of control variates [Graham and Talay, 2013], where a random variable ˆ u : Ω → R k with known expectations u = E [ˆ u ] and a matrix Γ ∈ R d × k are used in the variance-reduced estimator

<!-- formula-not-decoded -->

A mean squared error decomposition yields the identity

<!-- formula-not-decoded -->

indicating that for sufficiently 'small' Γ , ˜ v provides an improved estimator if ˆ v -v and ˆ u -u have high (multiple) correlation. While in the Monte Carlo literature, ˆ u and Γ can be chosen optimally provided knowledge of the underlying data-generating mechanism, as Γ can be interpreted as the regression function of ˆ v -v on ˆ u -u . 4 Outside of Monte Carlo simulation, this procedure can be applied more widely if the user chooses ˆ u and ˆ Γ based on intuition or limiting arguments.

4 In the Monte Carlo settings, it often holds that d = k and Γ = αI for some constant α ∈ R . Then, E ⟨ ˆ v -v, Γ(ˆ u -u ) ⟩ can be replaced by α Tr ( C ov(ˆ v, ˆ u )) and o ( ∥ Γ ∥ op ) can be replaced by o ( α ) .

In the stochastic optimization setting, v represents the true gradient of the objective at a particular parameter, while ˆ v represents a stochastic gradient estimate from an oracle. Variance reduction techniques have previously been applied in an incremental setting, in which a fixed data set of size n is provided at initialization, and the algorithm may only make multiple passes through this same data set [Gower et al., 2020]. Note that this differs from our fully stochastic setting, in which we receive a fresh sample Z t on each iterate t = 1 , . . . , n . For negative log-likelihood losses, our orthogonalized oracle can be viewed in a similar light to control variate-based variance reduction methods (although in an infinite-dimensional setting). To summarize, we have from (9) that

<!-- formula-not-decoded -->

̸

using the idealized parameters. Using the approximations for θ = θ ⋆ , we have

<!-- formula-not-decoded -->

In fact, using the derivative of the log likelihood in a control variate procedure has been explored in the simulation literature as early as Johnstone and Velleman [1985], as the correlation between a statistic and the score function has tight connections to the Cram´ er-Rao lower variance bound and exponential families. We emphasize, however, that our method does not require the loss to be of negative log-likelihood form nor any specific distributional knowledge to be applied.

## F.5 Discussion of Double Robustness

We now study the double robustness of SGD for dose-response estimation as discussed in Bonvini and Kennedy [2022]. Consider estimating the effect of the continuous treatment A ∈ A ⊂ R on the outcome Y ∈ Y ⊂ R , which is defined as E Y ( a ) (known as the dose-response function, DRF) under the potential outcomes framework. Under standard assumptions, the DRF takes the form

<!-- formula-not-decoded -->

where X ∈ X ⊂ R d is the measured confounders. Let Z = ( Y, A, X ) ∼ P with density p . We take the following notations:

<!-- formula-not-decoded -->

We can rewrite θ 0 ( t ) equivalently as

<!-- formula-not-decoded -->

We also take the notations P ( g ( Z )) = ∫ g ( z )d P ( z ) , P n ( g ( Z )) = n -1 ∑ n i =1 g ( Z i ) , ∥ g ∥ L 2 ( P ) = [ P ( g 2 ( Z ))] 1 / 2 to denote the L 2 ( P ) norm, and ∥ g ∥ L 4 ( P ) = [ P ( g 4 ( Z ))] 1 / 4 to denote the L 4 ( P ) norm. We now establish the procedure to estimate θ 0 ( t ) as Algorithm 1 in Bonvini and Kennedy [2022] with slightly modification to apply SGD:

1. Observe i.i.d. samples { Z ′ i } i m =1 for the nuisance estimation and i.i.d. samples { Z i } n i =1 for the parameter estimation.
2. Estimate µ , w , and m ( a ) = P µ ( a, · ) using { Z ′ i } i m =1 with ˆ µ , ˆ w , and ˆ m ( a ) = P n (ˆ µ ( a, · )) , respectively.

3. Construct the pseudo-outcome

<!-- formula-not-decoded -->

We also define the true nuisance as

<!-- formula-not-decoded -->

4. Define the loss function via a parametric function class F Θ = { f θ : A ↦→ R | θ ∈ Θ ⊂ R d } as

<!-- formula-not-decoded -->

Define the stochastic gradient oracle as

<!-- formula-not-decoded -->

5. Solve the optimization problem

<!-- formula-not-decoded -->

using SGD with the stochastic gradient S θ ( θ, ˆ φ ; Z ) by

<!-- formula-not-decoded -->

As demonstrated in Bonvini and Kennedy [2022], this procedure would yield a doubly robust ERM estimator. In the following proposition, we claim that double robustness would be preserved if the SGD estimator is adopted instead.

Proposition 24. Assume that E [ ∥∇ θ f θ ⋆ ( A ) ∥ 2 2 ] 1 / 2 ≤ C A . Suppose that Asm. 3 holds and θ (0) , . . . , θ ( n ) ∈ Θ almost surely for θ ( n ) in (106) . If η ≤ µ/ 2( Mµ + κ 1 ) , the iterates of (106) satisfy

<!-- formula-not-decoded -->

Prop. 24 follows directly from the following two lemmas, Lem. 25 and Lem. 26, and the nuisance sensitive rate in Thm. 1. Whenever the empirical estimation max a ∈A | ( P n -P ) { ˆ µ ( a, X ) }| 2 shrinks, Prop. 24 suggests that the θ ( n ) would converge to the the target parameter when either ˆ w or ˆ µ is correctly specified.

Lemma 25. Assume that E [ ∥∇ θ f θ ⋆ ( A ) ∥ 2 2 ] 1 / 2 ≤ C A . Then for the loss defined in (105) , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Let ˆ r ( t ) = E [ ˆ φ ( Z ) | A = t ] -θ 0 ( t ) . Note that

<!-- formula-not-decoded -->

Thus, by the assumption that E [ ∥∇ θ f θ ⋆ ( A ) ∥ 2 2 ] 1 / 2 ≤ C A ,

<!-- formula-not-decoded -->

Lemma 26. For the norm defined in Lem. 25, we have

<!-- formula-not-decoded -->

Proof. Lemma 1 of Bonvini and Kennedy [2022] demonstrates that

<!-- formula-not-decoded -->

where ∥ f ∥ 2 = ∫ f 2 ( z )d P ( z | A = t ) . By (107), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## F.6 Proof of Proposition 22

Proof. For simplicity, we use the notation E m,n to replace E D mn ∪S m ∼ P mn ⊗ Q m . Let q n = (1 -µη/ 2) n , δ ( m,n ) = θ ( m,n ) -θ ⋆ , and δ (0) = δ (0 ,n ) . Thus, by Thm. 1,

<!-- formula-not-decoded -->

This recursive formula gives a complete bound for θ ( m,n ) as

<!-- formula-not-decoded -->

By (96), we assume that ξ m ≤ Cm -2 α -1 2 α for some C &gt; 0 . Note that

<!-- formula-not-decoded -->

For the second term, when q n ∈ (0 , 1) we have

<!-- formula-not-decoded -->

The last term is easy to bound since for q n ∈ (0 , 1) ,

<!-- formula-not-decoded -->

We claim that for some constant c &gt; 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

With (108), we have which implies that

<!-- formula-not-decoded -->

When ( ηn ) -1 = O (1) , the bound above reduces to

<!-- formula-not-decoded -->

We will finish the proof by showing (108). The key step is to show 1 -e -x ≥ c min( x, 1) for all x &gt; 0 and some constant c &gt; 0 .

Let f ( x ) = 1 -e -x -x/ 2 , for x ∈ (0 , 1) we have

<!-- formula-not-decoded -->

Thus, f ( x ) ≥ f (log 2) = (1 -log 2) / 2 &gt; 0 for x ∈ (0 , 1) , which implies that 1 -e -x &gt; x/ 2 for x ∈ (0 , 1) . Note that 1 -e -x ≥ 1 -e -1 for x ≥ 1 . Let c = min(2 -1 , 1 -e -1 ) . Then we have 1 -e -x ≥ c min( x, 1) .

It follows that

<!-- formula-not-decoded -->

Since x -1 ≥ log x for all x &gt; 0 , we have

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

Thus, we complete the proof.

## F.7 Proof of Proposition 23

Proof. Given s &gt; 0 , we define A i = A i ( s 1 /c ) and B i = B i ( s 1 /c ) for i = 0 , . . . , m for simplicity. First, since c ≥ 1 , by (103) and Markov inequality, for i = 1 , . . . , m,

<!-- formula-not-decoded -->

We assume that P [ A 0 ] = P [ B 0 ] = 1 , and we have

<!-- formula-not-decoded -->

Similarly, we have

<!-- formula-not-decoded -->

we consider the conditional mean squared error of θ ( m,n ) given ( ∩ i m =0 A i ) ∩ ( ∩ i m =0 B i ) . By similar proof to Prop. 22, we can show that for some constant C 1 &gt; 0 ,

<!-- formula-not-decoded -->

We define the event of interest as

<!-- formula-not-decoded -->

where f s ( m,n ) is defined as

<!-- formula-not-decoded -->

By Markov inequality, we have

<!-- formula-not-decoded -->

Since

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

Define δ ( s ) as

<!-- formula-not-decoded -->

Then, with probability at least 1 -δ ( s ) , we have

<!-- formula-not-decoded -->

When ( ηn ) -1 = O (1) , it follows that

<!-- formula-not-decoded -->

## G Numerical Experiments

This section provides numerical experiments of the proposed stochastic methods in this paper. In Appx. G.1, we design a numerical experiment to illustrate our orthogonalization method. In Appx. G.2, we design simulations based on a partially linear model. In Appx. G.3, we conduct a real data analysis with synthetic outcome to evaluate the performance of our methods. Code for reproduction can be found at https://fachengyu.github.io/ .

## G.1 Numerical Illustration

In this section, we design a numerical experiment to illustrate how our orthogonalization method effects the target estimation as shown in Fig. 1 from the main text.

Settings. Consider Θ ∈ R and G = R . Let L ( θ, g ) be a real-valued risk function defined as

<!-- formula-not-decoded -->

where u = ( θ, g ) ⊤ ∈ R 2 , λ = 0 . 02 is the regularization parameter, and

<!-- formula-not-decoded -->

It is easy to see that (0 , 0) is the global minimizer of L since L ( θ, g ) ≥ 0 . Let q ( u ) = ⟨ u, Bu ⟩ . The gradient w.r.t. u is

<!-- formula-not-decoded -->

Since A + 2 λ sin(2 q ( u )) B ≽ A -0 . 04 B ≻ 0 , it is clear that (0 , 0) is the only stationary point, implying that (0 , 0) is the only minimizer of L . Furthermore, we can obtain the Hessian w.r.t. u as

<!-- formula-not-decoded -->

which implies that L ( · , g ) is not convex in R given any g ∈ G r ( g 0 ) . However, when Θ is a small neighborhood around zero, it is still possible to have L ( · , g ) strongly convex for in Θ given any g ∈ G r ( g 0 ) .

Orthogonalization. To orthogonalize L , we first derive the orthogonal gradient oracle using (11), and then integral the oracle w.r.t. θ to obtain the orthogonalized loss L no.

Let H be the Hessian at (0 , 0) . By (111), we know that H = A , implying H θg = A 12 and H gg = A 22 . Since the gradient w.r.t. θ satisfies

<!-- formula-not-decoded -->

and the gradient w.r.t. g satisfies

<!-- formula-not-decoded -->

follow the construction of (11) and we obtain the orthogonal gradient oracle as

<!-- formula-not-decoded -->

where a = A 11 -A 12 A -1 22 A 21 , b = B 11 -A 12 A -1 22 B 21 , and c = B 12 -A 12 A -1 22 B 22 . Finally, we can integral S no ( θ, g ) w.r.t. θ and recover the orthognalized loss L no as

<!-- formula-not-decoded -->

Numerical Computation. Usually, S no ( s, g ) contains a form of integral, which needs to be numerically computed. For the example introduced above, we can simplify L no ( θ, g ) to stabilize the numerical computation. Note that ∇ θ sin 2 ( q ( θ, g )) = sin(2 q ( θ, g ))( B 11 θ + B 12 g ) . Then

<!-- formula-not-decoded -->

It follows that the orthogonalized loss L no admits the following form

<!-- formula-not-decoded -->

which implies that only the integral of sin(2 q ( s, g )) w.r.t. s needs to be computed.

## G.2 Simulations

## G.2.1 Data Generating Process

To demonstrate Thm. 1 and Thm. 3, we revisit the partially linear model and the corresponding orthogonal and non-orthogonal losses in Appx. B.1. Specifically, ( X,W,Y ) ∈ R d × R d × R satisfies the following partially linear model where the nonlinear function is determined by the distribution of ( W,U ) ∈ R d × R :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where θ 0 ∈ R d is the true parameter, α 0 : W ↦→ R is the true nonlinear function, ϵ and ξ are independent noises that satisfy E [ ϵ | X,W ] = 0 and E [ ξ | W ] = 0 . It is clear that α 0 ( W ) = E [ U | W ] . In our simulations, we choose d = 2 and θ 0 = [ -0 . 5 1] ⊤ .

To get samples for simulations, we first generate ( X,W ) under the Gaussian model

<!-- formula-not-decoded -->

where µ X = [1 1] ⊤ , µ W = [2 2] ⊤ , I 2 ∈ R 2 × 2 is the identity matrix, λ ∈ [0 , 1] is used to control the correlation between X and W , and δ = 0 . 05 is used to prevent the degeneration of the covariance matrix. For simplicity, we define the nonlinear function α 0 as

<!-- formula-not-decoded -->

where w = [ w 1 w 2 ] ⊤ ∈ R 2 . We then generate Y and U using independent Gaussian noises ϵ ∼ N (0 , 1) and ξ ∼ N (0 , 1) based on (112) and (113), respectively.

## G.2.2 Stochastic Gradient Oracles

To estimate the true parameter θ 0 using stochastic gradients, we need to design a correspond loss whose minimizer θ ⋆ is equal to θ 0 . Based on Appx. B.1, there are two types of loss, the orthogonal loss and the non-orthogonal loss, available for this goal. We will derive the stochastic gradient oracle for these two losses and further derive the orthogonalized gradient oracle for the non-orthogonal loss.

Orthogonal loss. Recall the orthogonal loss in Appx. B.1.1:

<!-- formula-not-decoded -->

where g = ( g Y , g X ) : W → R × R d and the norm ∥·∥ G is defined in (27). The true nuisance for this loss is g 0 = ( g 0 ,X , g 0 ,Y ) , where g 0 ,Y ( w ) := E P [ Y | W = w ] and g 0 ,X ( w ) := E P [ X | W = w ] .

In fact, the explicit expression for g 0 can be easily obtained as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From (112) and (117), it is clear that

<!-- formula-not-decoded -->

which implies that θ ⋆ = θ 0 by Lem. 4. The stochastic gradient oracle for the orthogonal loss (116) is then defined as

<!-- formula-not-decoded -->

Non-orthogonal loss. We also provide the non-orthogonal loss in Appx. B.1.2 as

<!-- formula-not-decoded -->

where g : W ↦→ R and the norm ∥ · ∥ G is now defined in (36). The true nuisance for this nonorthogonal loss satisfies

<!-- formula-not-decoded -->

By Lem. 5, we have θ ⋆ = θ 0 . The stochastic gradient oracle for the orthogonal loss (120) is then defined as

<!-- formula-not-decoded -->

Orthogonalized gradient oracle. Since we perform orthogonalization on the non-orthogonal loss, we have θ ⋆ = θ 0 being the same target parameter. By (22) in Appx. B.1.2, the Neyman orthogonalized gradient oracle for this non-orthogonal loss (120) is given by

<!-- formula-not-decoded -->

## G.2.3 Estimation Methods

Throughout the experiments, we estimate the nuisances and the orthogonalizing operator using fullbatch data and stream data, respectively.

Nuisance estimation. Note that the true nuisances for the orthogonal loss and the non-orthogonal loss are conditional expectation given W . To conduct nonparametric regression, we use random Fourier feature (RFF) [Rahimi and Recht, 2007] using the kernel w ↦→ exp ( -γ · ∥ w ∥ 2 2 ) to generate a randomized feature map for W .

The nuisance estimation procedure for obtaining ˆ g ( m ) using full batch data can be summarized as

1. Fit RFF sampler with 20 components using m i.i.d. samples from P W | λ .
2. Fit Ridge regressions where the regularization parameter is set to be 0 . 01 /m . Specifically,
- For the orthogonal loss, fit two Ridge regressions using m i.i.d. samples from the joint distribution P X,W,Y | λ and the fitted RFF sampler to coordinate-wisely estimate E [ X | W ] . With the same data, fit one Ridge regression using the fitted RFF sampler to estimate E [ Y | W ] .
- For the non-orthogonal loss, fit one Ridge regression using m i.i.d. samples from the joint distribution P X,W,Y | λ and the fitted RFF sampler to estimate E [ U | W ] .

To estimate nuisances using stream data, instead of fit a Ridge regression each time, we perform SGD for the Ridge regression loss. The procedure can be summarized as

1. Initialize RFF sampler with 20 components using n 0 i.i.d. samples ( W i ) n 0 i =1 from P W | λ .
2. Perform SGD update once observing a mini-batch of i.i.d. samples from the joint distribution P X,W,Y | λ with size n g . Specifically,

- For the orthogonal loss, perform two SGD with the Ridge loss for m iterations to estimate E [ X | W ] coordinate-wisely. With the same data perform another SGD with the Ridge loss for m iterations to estimate E [ Y | W ] .
- For the non-orthogonal loss, perform one SGD with the Ridge loss for m iterations to estimate E [ U | W ] .

Orthogonalizing operator estimation. To approximate the orthogonalizing operator Γ 0 , it suffices to estimate E [ X | W ] by (21). To that end, we use the same method as the nuisance estimation. The orthogonalizing operator estimation procedure for obtaining ˆ Γ ( k ) can be summarized as

1. Fit RFF sampler with 20 components using k i.i.d. samples ( W ′ i ) k i =1 from P W | λ .
2. Fit two Ridge regressions with the regularization parameter being 0 . 01 /k using the fitted RFF sampler and another k i.i.d. samples ( X ′ i , W ′ i ) 2 k i = k to coordinate-wisely estimate E [ X | W ] .

Target estimation. After the estimation of nuisances and orthogonalizing operator, we perform stochastic gradient descent (SGD) to estimate θ ⋆ using each of the three stochastic gradient oracles in (119), (122), and (123) on n i.i.d. samples drawn from the joint distribution P X,W,U,Y . The learning rates of all the three SGDs are fixed during the training.

## G.2.4 Simulation Results

Setup. For each nuisance estimation setting, we study three types of estimation methods for learning θ 0 established in this paper: (1) (orthogonal loss) obtain nuisance estimator ˆ g ( m ) = (ˆ g ( m ) Y , ˆ g ( m ) X ) of (117) and (118) and then perform SGD to obtain θ ( n ) using the gradient oracle (119) after plugging in ˆ g ( m ) ; (2) (non-orthogonal loss) obtain the nuisance estimator ˆ g ( m ) = ˆ α ( m ) of (121) and then perform SGD to obtain θ ( n ) using the gradient oracle (122) after plugging in ˆ g ( m ) ; (3) (OSGD) obtain the nuisance estimator ˆ g ( m ) of (121) and the orthogonalizing operator estimator ˆ Γ ( k ) of (21), and then perform SGD to obtain θ ( n ) using the gradient oracle (123) after plugging in ˆ g ( m ) and ˆ Γ ( k ) . Each method is independently repeated 20 times. For nuisance estimated using stream data, we allow the procedure repeated by plugging in updated nuisance estimators and an updated operator estimator, where the nuisances get updated for 2000 iterations after every 2000 target SGD iterations.

Evaluation. We evaluate the performance of nuisance estimators using the corresponding norms defined in (27) and (36). Specifically, for method (1), we evaluate the nuisance estimation by

<!-- formula-not-decoded -->

For method (2) and (3), we evaluate the nuisance estimation by

<!-- formula-not-decoded -->

We evaluate ˆ Γ ( k ) : g ↦→ E [ˆ g ( k ) X ( W ) g ( W )] in method (3) using the Frobenius norm ∥ ˆ Γ ( k ) -Γ 0 ∥ Fro , which is defined as

<!-- formula-not-decoded -->

Finally, we evaluate the target estimation using two kinds of criterion: (a) the relative error ∥ θ ( n ) -θ 0 ∥ 2 ∥ θ 0 ∥ 2 , and (b) the risk L ( θ ( n ) , g 0 ) -L ( θ ⋆ , g 0 ) where L ( θ, g ) = E [ ℓ ( θ, g ; Z )] . For method (1), ℓ ( θ, g ; z ) is the orthogonal loss defined in (116) while for method (2) and (3), ℓ ( θ, g ; z ) is the non-orthogonal loss defined in (120).

Results using nuisances fitted on full-batch data. We first estimate the target using prefitted nuisances and operator. The estimation errors of nuisances and the operator fitted on full-batch data are shown in Fig. 2, where all estimation converges when the sample size m increases and less samples are usually required to obtain the same error level when λ increases.

Figure 2: The Nuisance and Orthogonalizing Operator Fitted on Full-Batch Simulated Data. The y-axis measures the corresponding error defined in (124) - (126) and the x-axis displays the sample size of data used to estimate the nuisance and operator.

<!-- image -->

Figure 3: SGD for Orthogonal Loss with the Nuisance Fitted on Full-Batch Simulated Data. The x-axis represents the SGD iteration. Top: The y-axis measures the relative error. Bottom: The y-axis measures the risk.

<!-- image -->

The performances of SGDs using prefitted nuisances and stochastic gradient oracles (119), (122), and (123) are shown in Fig. 3, Fig. 4, and Fig. 5, respectively. These figures suggest that when λ increases, i.e., the correlation between X and W increases, usually more iterations are required to have SGD converged due to the difficulty of separating the effect of X from W . In addition, a well prefitted nuisance estimator would largely reduce the SGD estimation error, which aligns with Thm. 1. This improvement would be more obvious as λ increases. Fig. 5 also shows that either using a well estimated nuisance or a well estimated orthogonalizing operator can improve the OSGD performance, and that OSGD using both well prefitted nuisance and operator would perform nearly the same as OSGD using the true nuisance and the true operator.

Results using nuisances fitted on stream data. We then study the interleaving the nuisance and target estimations discussed in Appx. F.3. Here, Both the nuisance and the operator are learned using the same data stream and the results are shown in Fig. 6. Compared with Fig. 2, nuisances estimated using stream data usually has larger error and need more iterations to converge due to mini-batch, learning rate, and other tuning parameters.

The performances of SGDs by interleaving nuisance and target updates with stochastic gradient oracles (119), (122), and (123) are shown in Fig. 7, Fig. 8, and Fig. 9, respectively. For all the three stochastic gradients, when λ increases, the relative errors of the target SGD always get larger and their convergence rates become slower. There are obvious errors for SGDs using gradient oracles (119) and (122) in Fig. 7 and Fig. 8 since nuisances are not well estimated. However, OSGD

<!-- image -->

Figure 4: SGD for Non-Orthogonal Loss with the Nuisance Fitted on Full-Batch Simulated Data. The x-axis represents the SGD iteration. Top: The y-axis measures the relative error. Bottom: The y-axis measures the risk.

Figure 5: OSGD with the Nuisance and Operator Fitted on Full-Batch Simulated Data. Here, m 1 = 500 , m 2 = 10000 , k 1 = 300 , k 2 = 10000 . The x-axis represents the OSGD iteration. Top: The y-axis measures the relative error. Bottom: The y-axis measures the risk.

<!-- image -->

Figure 6: Nuisance and Orthogonalizing Operator Fitted on Simulated Stream Data. The yaxis measures the corresponding error defined in (124) - (126) and the x-axis displays the sample size of data used to estimate the nuisance and operator.

<!-- image -->

Figure 7: SGD for Orthogonal Loss with the Nuisance Fitted on Simulated Stream Data. The x-axis represents the SGD iteration. Top: The y-axis measures the relative error. Bottom: The y-axis measures the risk.

<!-- image -->

Figure 8: SGD for Non-Orthogonal Loss with the Nuisance Fitted on Simulated Stream Data. The x-axis represents the SGD iteration. Top: The y-axis measures the relative error. Bottom: The y-axis measures the risk.

<!-- image -->

performs perfectly as shown in Fig. 9, which verifies the analysis of Thm. 3 that using an estimated orthogonalizing operator would reduce the bias from nuisance estimation.

## G.3 Real Data Analysis

We consider the Diabetes 130-Hospitals Dataset [Clore et al., 2014] as the real dataset example. We use six of these features as covariates, which are summarized in Tab. 4. We take the binary feature 'change' as the input X ∈ { 0 , 1 } and take the rest five features as the control W ∈ R 5 .

## G.3.1 Synthetic outcomes

To evaluate the performance of our proposed methods, we use the synthetic outcome instead of a real outcome to examine the performance of our proposed methods. Using the synthetic outcome is common in causal inference; see Nie and Wager [2021, Sec. 4.1]. In this real data analysis, we generate outcome according to the following partially linear model:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Figure 9: OSGD with the Nuisance Fitted on Simulated Stream Data. The x-axis represents the SGD iteration. Top: The y-axis measures the relative error. Bottom: The y-axis measures the risk.

<!-- image -->

Table 4: Features used for real data analysis.

| Feature                                                                                    | Description                                                                                                                                                                                                                                                                                                                                                               |
|--------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| change time in hospital num lab procedures num procedures num medications number diagnoses | Indicates if there was a change in diabetic medications. Integer number of days between admission and discharge. Integer number of lab tests performed during the encounter. Integer number of procedures (other than lab tests) performed during the encounter. Integer number of distinct generic names administered during the encounter. Integer number of diagnoses. |

where ˜ θ 0 = -1 , ϵ ∼ N (0 , 1) and ξ ∼ N (0 , 1) are independent noises, and ˜ α 0 : R 5 ↦→ R satisfies that for w = ( w 1 , . . . , w 5 ) ,

<!-- formula-not-decoded -->

Similar to Appx. G.2.2, we have θ ⋆ = ˜ θ 0 in this case.

## G.3.2 Real Data Results

Setup. We consider the same three stochastic gradient oracles as Appx. G.2.2 and the same two nuisance estimation methods as Appx. G.2.3 except that we use logistic regression on full batch data and SGD of the logistic loss on stream data for estimating E [ X | W ] . The setup of SGD using prefitted nuisances for this real data analysis is the same as Appx. G.2.4. For nuisance estimated using stream data, we update nuisances for 100 iterations after every 500 target SGD iterations.

Evaluation. Since the true nuisances E [ X | W ] and E [ Y | W ] are unknown, we evaluate the performance of nuisance estimation ˆ g ( m ) = (ˆ g ( m ) Y , ˆ g ( m ) X ) for the orthogonal loss by the mean squared error:

<!-- formula-not-decoded -->

We adopt the nuisance estimation error ∥ ˆ g ( n ) -g 0 ∥ G defined in (125) as the nuisance evaluation for non-orthogonal loss due to the synthetic outcome, where now g 0 = ˜ α 0 . For the operator estimation

Figure 10: Nuisance and Orthogonalizing Operator Fitted on Full-Batch Real Data. The xaxis displays the sample size of data used to estimate the nuisance and operator. Left. The y-axis measure the nuisance error defined in (125). Middle. The y-axis measure the nuisance estimation MSE defined in (129). Right. The y-axis measure the operator estimation MSE defined in (130).

<!-- image -->

Figure 11: Stochastic Gradients with Nuisance Fitted on Full-Batch Real Data. Here, m 1 = 32 , m 2 = 64 , m 3 = 128 , m 4 = 8 , m 5 = 128 , k 1 = 32 and k 2 = 128 . The x-axis represents the SGD iteration using corresponding stochastic gradient. Top: The y-axis measures the relative error. Bottom: The y-axis measures the risk.

<!-- image -->

ˆ Γ ( m ) : g ↦→ E [ˆ g ( m ) X ( W ) g ( W )] , evaluate its performance by the mean squared error:

<!-- formula-not-decoded -->

Results using nuisances fitted on full-batch data. We first estimate the target using prefitted nuisances and operator. The estimation errors of nuisances and the operator using full-batch real data are shown in Fig. 10, which suggests that the estimation of ˜ α 0 converges to zero due to our design while there exists obvious bias for estimating the nuisance ( g 0 ,X , g 0 ,Y ) and the orthogonalizing operator Γ 0 possibly due to model misspecification.

The performances of SGDs using prefitted nuisances and stochastic gradient oracles (119), (122), and (123) are shown in Fig. 11. Overall, the relative error and the risk are small when well estimated nuisances are used. In addition, both relative errors and risks become smaller when we use more samples to estimate nuisances for the orthogonal loss and the non-orthogonal loss.

Results using nuisances fitted on stream data. We then estimate the target by interleaving the nuisance and target updates. Here, Both the nuisance and the operator are learned using the same

Figure 12: Estimation Errors of Nuisance and Orthogonalizing Operator Fitted on Stream Data. The x-axis displays the sample size of data used to estimate the nuisance and operator. Left. The y-axis measure the nuisance error defined in (125). Middle. The y-axis measure the nuisance estimation MSE defined in (129). Right. The y-axis measure the operator estimation MSE defined in (130).

<!-- image -->

Figure 13: Stochastic Gradients with Nuisance Fitted on Real Stream Data. The x-axis represents the SGD iteration. Top: The y-axis measures the relative error. Bottom: The y-axis measures the risk.

<!-- image -->

data stream and the results are shown in Fig. 12. Compared with Fig. 10, nuisances estimated using stream data converges similarly.

The performances of SGDs by interleaving nuisance and target updates with stochastic gradient oracles (119), (122), and (123) are shown in Fig. 13. The figure on the left in Fig. 13 shows that the target estimation has small relative error using the estimated nuisance sequence { ˆ g ( m ) : m ≥ 1 } . The figure in the middle suggests that there is still some bias for the target estimation while this bias is negligible. The figure on the right shows the performance of OSGD, where the relative error of OSGD using the estimated nuisance sequence is similar to OSGD using the true nuisance, which aligns with Thm. 3.

## H Extension to SGD Variants

In this section, we discuss strategies for analyzing other variants of SGD under nuisances. In Appx. H.1, we discuss the relationship between SGD with momentum and averaged SGD and provide a convergence analysis example of the averaged SGD. In Appx. H.2, we discuss Adam as a generalization of SGD with momentum and the difficulties to analyze the convergence rate of Adam.

## H.1 SGD with Momentum and Averaged SGD

For the gradient oracle sequence S ( n ) , SGD with momentum following the description of Li et al. [2022] can be expressed as

<!-- formula-not-decoded -->

where ¯ θ ( n ) is the SGD estimation sequence, m ( n ) is the momentum sequence, and ( α n ) n ≥ 0 and ( β n ) n ≥ 0 can be any positive sequence. The following example shows that the averaged SGD is a special case of SGD with momentum.

Example 5 (Averaged SGD). Let β n = 1 /n and α n = η (1 -β n +1 ) for all n ≥ 1 . The momentum updates implied by this sequence are

<!-- formula-not-decoded -->

which implies that ¯ θ ( n +1) is the averaged SGD such that

<!-- formula-not-decoded -->

Example 5 demonstrates that the convergence rate of SGD with momentum can be analyzed in the same way as averaged SGD. While it is not the focus of this paper, we provide a convergence result of the averaged SGD based on the analysis of D´ efossez and Bach [2015].

Proposition 27 (Convergence rate of averaged SGD) . Consider the partially linear model and the non-orthogonal loss ℓ ( θ, g ; z ) in Appx. B.1.2. Define D n = ( Z 1 , . . . , Z n ) , sampled from the product measure P n . Choose the gradient oracle S ( n ) to be the score S θ ( θ, ˆ g ; Z n ) where ˆ g is estimated independently of D n . Let ¯ θ ( n ) be the averaged SGD defined in (132) . Suppose the same assumptions as Lem. 5. If 0 &lt; η &lt; η max , then

<!-- formula-not-decoded -->

where η max = sup { η &gt; 0 : tr ( A ⊤ E P [ XX ⊤ ] A ) -η E P [ ( X ⊤ AX ) 2 ] &gt; 0 , ∀ A ∈ S ( R d ) } and S ( R d ) is the set of all d × d symmetric matrices.

Before we prove Prop. 27, recall the example of non-orthogonal loss for the partially linear model in Appx. B.1.2, where Z = ( X,W,Y ) ∼ P satisfies

<!-- formula-not-decoded -->

The target parameter θ ⋆ = arg min θ ∈ Θ E P [ ℓ ( θ, g ; Z )] where ℓ is the non-orthogonal loss defined as

<!-- formula-not-decoded -->

By Lem. 5, we have θ ⋆ = θ 0 . The stochastic gradient oracle for this non-orthogonal loss is

<!-- formula-not-decoded -->

and the SGD iteration is defined by θ (0) ∈ Θ and

<!-- formula-not-decoded -->

where ˆ g ∈ G r ( g 0 ) is any nuisance estimator independent of { Z i } n i =1 . Note that (135) can be written as

<!-- formula-not-decoded -->

Let β ( n ) = θ ( n ) -θ ⋆ , r n = ˆ g ( W n ) -g 0 ( W n ) , and

<!-- formula-not-decoded -->

By recursion, we have

<!-- formula-not-decoded -->

In the above formula, first two terms are usually interpreted as the bias term and the variance term under the true nuisance, respectively according to D´ efossez and Bach [2015], and the last term can be viewed as the error term caused by the nuisance estimation.

To analyze the bias term and the variance term, we adopt the notations of D´ efossez and Bach [2015] for matrices and operators. First, Define H = E P [ XX ⊤ ] . Let H L (resp. H R ) be the matrix operator representing left multiplication (resp. right multiplication) by H , and T be the linear operator such that for any square matrix M ∈ R d × d , TM = HM + MH -η E P [ ( X ⊤ MX ) XX ⊤ ] . Let ρ = max {∥ I -ηH ∥ op , ∥ I -ηT ∥ op } where the operator norm ∥·∥ op is defined as the largest singular value. Finally, let η max be the same as in Prop. 27. With definitions above, the asymptotic covariances of the bias and the variance follow directly from Theorems 1 and 2 of D´ efossez and Bach [2015, Appx. 3].

Lemma 28 (Asymptotic covariance of the bias) . Let Ξ 0 = E P [ β (0) β (0) T ] . If 0 &lt; η &lt; η max , then

<!-- formula-not-decoded -->

where B n = 1 n +1 ∑ n j =0 M 0 ,j β (0) .

Lemma 29 (Asymptotic covariance of the variance) . Let Σ 0 = V ar( X n ( ϵ n -r n ) + E [ X n r n ]) . If 0 &lt; η &lt; η max , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In fact, the convergence rate of averaged SGD depends on tr ( B n B ⊤ n ) and tr ( V n V ⊤ n ) . When ρ &lt; 1 , Lem. 28 demonstrate that tr ( B n B ⊤ n ) is of order n -2 , while Lem. 29 shows that tr ( V n V ⊤ n ) is of order n -1 , which is reasonable due to the randomness of the noise ϵ k -r k .

For the error term, we can analyze it in a similar way to the bias term. Let ∆ = E P [ X n r n ] and

<!-- formula-not-decoded -->

Note that by Jensen's inequality,

<!-- formula-not-decoded -->

It is clear to see that the asymptotic covariance of ∆ k,n := ∑ n j = k M k,j ∆ is of the same form as the bias term in Lem. 28. Let G 0 = E P [ ∆∆ ⊤ ] and we have

<!-- formula-not-decoded -->

Thus, the trace of the covariance summation over k = 1 , . . . , n satisfies

<!-- formula-not-decoded -->

Gathering the bias term, the variance term, and the error term, we are now ready to proof Prop. 27.

Proof of Prop. 27. By Lemma 1 of D´ efossez and Bach [2015], 0 &lt; ρ &lt; 1 when 0 &lt; η &lt; η max . Suppose that ∥ X ∥ ∞ ≤ C X almost surely. By Jensen's inequality we have that

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

By Lem. 28, Lem. 29, and (136), we have

<!-- formula-not-decoded -->

## H.2 Adam

The primary updates for Adam under nuisance estimate ˆ g are given by the following recursive equations. Below, we let i ∈ { 1 , . . . , d } denote a particular dimension of the finite-dimensional parameter of interest. Following the description of D´ efossez et al. [2022], for the gradient oracle sequence S ( n ) the Adam generates the target estimator ˜ θ ( n ) as below:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where β 2 ∈ (0 , 1] , β 1 ∈ [0 , β 2 ] are the momentum and variance parameters, m ( n ) , v ( n ) ∈ R d are the momentum and variance sequences, and ϵ &gt; 0 is a numerical stability parameter. Adam differs from the SGD with momentum by adding a variance sequence v ( n ) . When v ( n ) is chosen to be a constant, then (137) and (139) would reduce to the special case of SGD with momentum where β n and α n are constant.

The analysis of Adam is often done in the case of smooth non-convex optimization, in which it is shown that the gradient of the objective tends to zero [Ward et al., 2020, D´ efossez et al., 2022]. Specifically, D´ efossez et al. [2022] consider a momentum-free Adam ( β 1 = 0 ) to analyze the essential ingredients that differ from momentum: the variance pre-conditioning and element-wise updates, which suggests that under the true nuisance, i.e., S ( n ) = S θ ( θ ( n ) , g 0 ; Z n ) for an i.i.d. sample { Z i } n i =1 ∼ P n , the convergence result of Adam satisfies

<!-- formula-not-decoded -->

Note that this result is not comparable to our convergence criterion (in terms of iterations), which differs non-trivially from a stationarity or function value analysis. While the convergence of Adam without nuisance has been studied in the literature, it still remains unclear that how to analysis Adam under an estimated nuisance ˆ g and what should be the nuisance effect on the gradient norm criterion.