## Multiclass Loss Geometry Matters for Generalization of Gradient Descent in Separable Classification

## Matan Schliserman

Blavatnik School of Computer Science and AI Tel Aviv University schliserman@mail.tau.ac.il

## Tomer Koren

Blavatnik School of Computer Science and AI Tel Aviv University and Google Research tkoren@tauex.tau.ac.il

## Abstract

We study the generalization performance of unregularized gradient methods for separable linear classification. While previous work mostly deal with the binary case, we focus on the multiclass setting with 𝑘 classes and establish novel population risk bounds for Gradient Descent for loss functions that decay to zero. In this setting, we show risk bounds that reveal that convergence rates are crucially influenced by the geometry of the loss template , as formalized by Wang and Scott [24], rather than of the loss function itself. Particularly, we establish risk upper bounds that holds for any decay rate of the loss whose template is smooth with respect to the 𝑝 -norm. In the case of exponentially decaying losses, our results indicates a contrast between the 𝑝 = ∞ case, where the risk exhibits a logarithmic dependence on 𝑘 , and 𝑝 = 2 where the risk scales linearly with 𝑘 . To establish this separation formally, we also prove a lower bound in the latter scenario, demonstrating that the polynomial dependence on 𝑘 is unavoidable. Central to our analysis is a novel bound on the Rademacher complexity of low-noise vector-valued linear predictors with a loss template smooth w.r.t. general 𝑝 -norms.

## 1 Introduction

The generalization properties of gradient-based learning methods, particularly in overparameterized regimes, is a central topic of study in contemporary machine learning. A key question is how unregularized gradient methods achieve good generalization despite their potential to overfit. Early work by Soudry et al. [21] demonstrated that gradient descent (GD) applied to linearly separable data with the logistic loss asymptotically converges to the max-margin solution. This result suggests that gradient descent, when properly tuned, can avoid overfitting without explicit regularization. Extensions of this result to other optimization algorithms and loss functions have further deepened our understanding of this phenomenon in various scenarios [3, 4, 12, 13, 5].

A particularly interesting regime for these investigations is multi-class classification. In this setting, Soudry et al. [21] achieved convergence to max-margin with the cross-entropy loss, and Lyu and Li [8], Lyu et al. [9] extended the results to homogeneous models and two-layer networks. More recently, Ravi et al. [15] generalized the implicit bias analysis to a broader class of exponentially tailed loss functions using the PERM framework [24], thereby bridging the binary and multi-class settings in this context.

Beyond these asymptotic results, recent work has focused on the generalization performance of gradient-based methods in finite-time regimes. In the binary classification setting, several recent works examined gradient-based methods applied to smooth loss functions that decay to zero at infinity [20, 17, 19, 23]. These results show that strong generalization without explicit regularization, even in finite time, can be achieved by gradient methods also beyond the regime of exponentially

tailed loss functions. In terms of bounds, their results reveal that generalization performance is fully characterized by the decay rate of the loss function.

Despite these advancements in understanding gradient methods in separable classification, finite-time generalization in the multi-class setting remains rather poorly understood-even for exponentially decaying loss functions, and particularly with regard to the dependence of risk bounds on the number of classes. In particular, several fundamental questions remains open: does unregularized GD generalize well also after finite number of iterations? Does the algorithm's generalization ability extend beyond the exponential decay setting? How do the properties of a loss function influence the achievable test loss bounds? Additionally, does the sole dependence of generalization performance on the decay rate of the loss function, as observed in the binary case, also extend to multi-class classification? In fact, the first two questions were stated as open problems by Ravi et al. [15].

In this work, we address these questions by studying the finite-time generalization properties of gradient descent when applied with a multi-class loss function ℓ : ℝ 𝑘 × 𝑘 → ℝ . Our findings reveal key distinctions from the binary classification case. Whereas in the binary regime risk bounds depend solely on the decay rate of the loss function, we show that in the multi-class setting risk bounds crucially depend on the geometry of the multi-class loss function, as determined by the norm with respect to which it is smooth. This differs from the results of Ravi et al. [15], that suggest that all exponentially tailed loss functions behave asymptotically similarly.

The class of functions that we consider is similar to the class considered by Wang and Scott [24], who showed that in the setting of classification with 𝑘 classes, losses are characterized by their template: a function ˜ ℓ : ℝ 𝑘 -1 → ℝ that has a simpler form than the original loss function ℓ . For multi-class classification losses with a template that is 𝛽 -smooth with respect to the 𝐿 𝑝 norm and decays to zero at infinity, we establish the following upper bounds on the risk of the output of gradient descent (when the step size is tuned optimally),

<!-- formula-not-decoded -->

where 𝜌 : ℝ → ℝ represents the decay rate of the loss function, 𝑘 denotes the number of classes, 𝑇 is the number of gradient steps, 𝑛 is the sample size, and 𝛾 is the separation margin. These results suggest that gradient descent can generalize well for a reasonable number of classes ( 𝑘 ≪ 𝑇, 𝑛 ). As with the bounds in the binary case established in [19], the risk bounds depend on the decay rate of the function, through the expression 𝜌 -1 ( 𝜖 / 𝑘 ) (though here the decay function 𝜌 -1 is evaluated at 𝜖 / 𝑘 , compared to 𝜖 in the binary case).

Next, noticing the fact that our upper bounds behave differently for templates that are smooth with respect to the 𝐿 𝑝 -norm for sufficiently large 𝑝 , and the case of 𝑝 = 2 , giving better generalization bounds in the former case, we establish this separation formally, by showing tight lower bounds for any decay rate in the 𝐿 2 regime. We also provide examples for this separation in popular loss classes.

In terms of techniques, our analysis requires some new technical tools. First, we derive a Rademacher complexity bound for multi-class losses whose templates are smooth with respect to the 𝐿 𝑝 norm ( 2 ≤ 𝑝 ≤ ∞ ), in the low-noise regime. Next, we show that the choice of step-size of gradient descent also depends on the geometry of the loss, achieving improved optimization performance as 𝑝 becomes larger. Putting these technical pieces together, we obtain the aforementioned risk upper bounds. We remark that this approach applies to essentially any gradient method that produces a model with low norm and low optimization error, making it applicable beyond gradient descent.

## 1.1 Summary of Contributions

To summarize, our contributions are as follows:

- Our first main result (Theorem 1) establishes an upper bound for unregularized gradient descent in separable multiclass classification for any loss function that decays to zero. Our bound suggests that the dependence on the number of classes improves as 𝑝 increases.
- Our second main result (Theorem 3) shows a tight lower bound for losses with templates that are smooth with respect to 𝐿 2 and decays to zero. Our lower bound reveals a strict separation between templates that are smooth with respect to the 𝐿 𝑝 -norm for sufficiently large 𝑝 , and the case of 𝑝 = 2 , where a polynomial dependence on the number of classes is unavoidable.

- As direct applications of our general bounds, we derive upper and lower bounds for templates with several decay rates (see Section 5). For example, in the exponential rate case, our result reveals that if the template is smooth with respect to the 𝐿 ∞ norm, the risk bounds align with those of the binary case and depend only logarithmically on 𝑘 ; in contrast, for 𝑝 = 2 the rate has an unavoidable linear dependence on the number of classes.
- Finally, as an additional technical contribution that underlies our analysis, we show a new upper bound on the Rademacher complexity for multi-class classification losses in the low-noise regime, where the loss template is smooth with respect to any 𝐿 𝑝 norm with 𝑝 ≥ 2 , refining and extending the results in [7, 16]. In particular, our assumptions apply to the template rather than the individual loss functions, which represents a new perspective (see Section 1.2 for further discussion).

Put together, our results reveal that the geometry of the loss template plays a crucial role in the generalization behavior of gradient descent. Prior work on separable classification showed that for exponentially tailed losses, gradient descent implicitly converges toward max-margin solutions [21, 15], and that in the particular case of binary classification with more general tails, generalization depends primarily on the decay rate of the loss [17, 19]. In the more general multiclass setting, our results indicate that this behavior is strongly influenced by the smoothness properties of the loss template with respect to geometries. In particular, losses with similar decay rates can induce very different generalization bounds, depending on their underlying geometry. This can serve to explain why ℓ ∞ -smooth losses such as the cross-entropy scale more favorably with the number of classes as compared to ℓ 2 -smooth losses.

## 1.2 Additional related work

Convergence rates for unregularized GD in separable classification. The risk of Gradient Descent in separable classification has been extensively studied. Firstly, the asymptotic analysis in the fundamental work of Soudry et al. [21] showed an upper bound of 1 / log ( 𝑇 ) for the classification error of gradient descent. Then, using a more refined analysis Shamir [20] established tight bounds on for gradient descent applied to binary cross-entropy loss. Later, Schliserman and Koren [17], Telgarsky [23], Schliserman and Koren [19] extended this analysis. Schliserman and Koren [17] showed generalization bounds for gradient-based methods with constant step sizes in using an additional self-boundedness assumption. Telgarsky [23] established a high-probability risk bound for 𝑇 ≤ 𝑛 for batch Mirror Descent with a non-constant step size for linear models. Schliserman and Koren [19] showed tight risk bounds for the binary case were given for any smooth loss decaying to zero. While all of the aforementioned work (except Schliserman and Koren [17] that discussed the particular case of the cross entropy loss), studied binary classification, in this work we address the multi-class setting and establish risk bounds applicable to any classification loss with smooth template that decays to zero, without any additional assumptions.

Lower bounds for unregularized GD in separable classification. There are several lower bounds in the context of binary classification. Firstly, Ji and Telgarsky [4] presented a lower bound of Ω ( log ( 𝑛 )/ log ( 𝑇 )) for the distance between the output of GD and a max margin solution with the same norm. In other work, Shamir [20] proved a lower bound of Ω ( 1 / 𝛾 2 𝑇 ) for the empirical risk of GD when applied to logistic loss. More recently, Schliserman and Koren [19] showed a tight lower bounds for the risk of GD, that are valid for any decay rate of the loss function. In this work, we establish the first lower bounds for unregularized GD when applied in the multi-class setting. Our lower bound is valid for losses with a template with any decay rate that is smooth with respect to the 𝐿 2 norm.

Vector-valued predictors (VVPs). Extensive research has been dedicated to understanding the sample complexity of vector-valued predictors. For the non-smooth regime with bounded domain, Maurer [11] established upper bounds scaling as 𝑂 ( 𝑘 ) for Lipschitz predictors with bounded Frobenius norm. In addition, Lei et al. [6] and Zhang and Zhang [26] derived logarithmic bounds in 𝑘 for ℓ ∞ -Lipschitz VVPs with arbitrary initialization. In another work, Magen and Shamir [10] studied the role of initialization and established bounds independent of 𝑘 when the algorithm is initialized at the origin. However, these bounds grow exponentially with the error 𝜖 , the Lipschitz constant 𝐿 , and the radius of the initialization ball. For lower bounds, Magen and Shamir [10] established a generalization lower bound of Ω ( log 𝑘 ) for convex predictors, while Schliserman and Koren [18]

improved this to match the upper bounds of Maurer [11] under the 𝐿 2 -Lipschitz condition for the nonsmooth case. Unlike these previous studies, our work focuses on the smooth and unregularized setting, where the effective norm of the iterates and the Lipschitz constant may be depend on 𝑘 , optionally introducing additional multiplicative factors in the bounds.

Fast rates for VVPs. There is a large body of work that achieves fast rates for VVPs. For example, Reeve and Kaban [16] showed Rademacher complexity bounds that are logarithmic in 𝑘 for smooth losses with respect to the 𝐿 ∞ norm with bounded domain, while Li et al. [7] provided rates linear in 𝑘 for 𝐿 2 -smooth losses. Another related work is the work of Wu et al. [25] that established fast rates generalization bounds for SGD in strongly convex settings. Importantly, in this study, we show that in multi-class classification, it suffices to assume the smoothness of the template of the loss function, rather than the actual loss function, and demonstrate that this property characterizes the generalization of gradient descent in this setting. In addition, we show Rademacher complexity bounds for the general 𝐿 𝑝 norm, recovering the bounds of Li et al. [7] and Reeve and Kaban [16] as special cases.

## 2 Problem Setup

We consider the following multi-class linear classification setting. Let D denote a distribution over pairs ( 𝑥, 𝑦 ) , where 𝑥 ∈ ℝ 𝑑 is a 𝑑 -dimensional feature vector, and 𝑦 ∈ [ 𝑘 ] is the class index corresponding to 𝑥 . We assume that the data is scaled such that ∥ 𝑥 ∥ 2 ≤ 1 with probability 1 with respect to D . Our focus is on the separable linear classification setting with margin. Specifically, denoting the Frobenius norm of a matrix 𝑊 ∈ ℝ 𝑘 × 𝑑 by ∥ 𝑊 ∥ 𝐹 and its 𝑗 'th row by 𝑊 𝑗 , we assume the following separability assumption:

Assumption 1 (Separability) . There exists a matrix 𝑊 ∗ ∈ ℝ 𝑘 × 𝑑 , with rows 𝑊 1 ∗ , . . . , 𝑊 𝑘 ∗ , such that ∥ 𝑊 ∗ ∥ 𝐹 ≤ 1 and, with probability 1 over ( 𝑥, 𝑦 ) ∼ D ,

<!-- formula-not-decoded -->

Given a multi-class loss function ℓ : ℝ 𝑘 × [ 𝑘 ] → ℝ + , the goal is to find a model 𝑊 ∈ ℝ 𝑘 × 𝑑 that minimizes the (population) risk, defined as the expected loss over the distribution D :

<!-- formula-not-decoded -->

For this, we use a dataset 𝑆 = {( 𝑥 1 , 𝑦 1 ) , . . . , ( 𝑥 𝑛 , 𝑦 𝑛 )} of training examples drawn i.i.d. from D , and optimize the empirical risk:

<!-- formula-not-decoded -->

For convenience, we define the function ℓ 𝑦 : ℝ 𝑘 → ℝ as ℓ 𝑦 = ℓ (· , 𝑦 ) . In addition, for every vector 𝑣 ∈ ℝ 𝑑 , we denote its 𝑗 'th entry by 𝑣 [ 𝑗 ] .

## 2.1 Loss Functions and Templates

Here we detail the class of loss functions that we consider. First, following [24], we define the template of a multi-classification loss function.

Definition 1 (Multi-class loss template) . Given a multi-class loss function ℓ : ℝ 𝑘 × [ 𝑘 ] → ℝ + , we say that ˜ ℓ : ℝ 𝑘 -1 → ℝ is a template of ℓ , if for every class 𝑦 ∈ [ 𝑘 ] , it holds that

<!-- formula-not-decoded -->

where 𝐷 𝑦 ∈ ℝ ( 𝑘 -1 ) × 𝑘 is the negative identity matrix when the 𝑦 th row is omitted and the 𝑦𝑡ℎ column is replaced by the vector that all of its entries are 1 .

Note that for every vector 𝑣 it holds that, 𝐷 𝑦 𝑣 = ( 𝑣 [ 𝑦 ] -𝑣 [ 1 ] , 𝑣 [ 𝑦 ] -𝑣 [ 2 ] , . . . , 𝑣 [ 𝑦 ] -𝑣 [ 𝑘 ]) , where the zero entry, 𝑣 [ 𝑦 ] -𝑣 [ 𝑦 ] , is omitted.

The templates considered in this work are 𝛽 -smooth with respect to 𝐿 𝑝 norm for 𝑝 ≥ 2 , as described in the following definition.

Definition 2 (smoothness w.r.t. 𝐿 𝑝 ) . A differentiable function 𝑓 : ℝ 𝑑 → ℝ is 𝛽 -smooth function w.r.t 𝐿 𝑝 norm if ∥∇ 𝑓 ( 𝑣 ) - ∇ 𝑓 ( 𝑢 )∥ 𝑞 ≤ 𝛽 ∥ 𝑣 -𝑢 ∥ 𝑝 for all 𝑢, 𝑣 ∈ ℝ 𝑑 , where 1 𝑞 + 1 𝑝 = 1 .

The primary goal of this paper is to quantify how the risk bounds depend on properties of the template ˜ ℓ , especially the rate at which it decays to zero as its input approaches infinity and the particular norm 𝐿 𝑝 which it is smooth with respect to it. To formalize this, we use the following definition, following [19]:

Definition 3 (Tail Function) . A function 𝜌 : [ 0 , ∞) → ℝ is called a tail function if 𝜌 :

- (i) is nonnegative, 1 -Lipschitz, and 𝛽 -smooth convex;
- (ii) is strictly decreasing and lim 𝑢 →∞ 𝜌 ( 𝑢 ) = 0 ;
- (iii) satisfies 𝜌 ( 0 ) ≥ 1 and | 𝜌 ′ ( 0 )| ≥ 1 2 .

In addition, we can define the following class of templates,

Definition 4 ( 𝜌 -Tailed Class) . For a given tail function 𝜌 , the class ˜ C 𝛽,𝑝 𝜌 is defined as of all nonnegative and convex functions ˜ ℓ : ℝ 𝑘 -1 → ℝ such that:

- (a) ˜ ℓ is 𝛽 -smooth with respect to the 𝐿 𝑝 norm.
- (b) lim 𝑡 →∞ ˜ ℓ ( 𝑡𝑢 ) = 0 for all 𝑢 ∈ ( ℝ + ) 𝑘 -1 .
- (c) ˜ ℓ ( 𝑢 ) ≤ ˝ 𝑘 -1 𝑗 = 1 𝜌 ( 𝑢 [ 𝑗 ]) for all 𝑢 ∈ ( ℝ + ) 𝑘 -1 .

Now, the actual class of functions we consider is the following class, which contains multi class classification losses.

Definition 5 ( 𝜌 -Tailed MCC Class) . The class C 𝛽,𝑝 𝜌 is defined as all loss functions ℓ : ℝ 𝑘 × [ 𝑘 ] → ℝ for which there exists ˜ ℓ ∈ ˜ C 𝛽,𝑝 𝜌 such that ˜ ℓ is a template of ℓ .

The vast majority of loss functions used in multi-class classification are in C 𝛽,𝑝 𝜌 for some tail function 𝜌 , 𝑝 and 𝛽 . In Section 5, we detail several applications of our bounds for popular multi-class functions.

## 2.2 Unregularized Gradient Descent

In this work, we focus on standard Gradient Descent with a fixed step size 𝜂 &gt; 0 , applied to the empirical risk b 𝐿 . The algorithm is initialized at 𝑊 1 = 0 and performs updates at each step 𝑡 = 1 , . . . , 𝑇 as follows:

<!-- formula-not-decoded -->

The algorithm outputs the final model 𝑊 𝑇 .

While our primary focus is on GD, the majority of our results can also be adapted to other gradient methods.

## 3 Risk Bounds for GD on Multiclass Losses

In this section we establish our upper bound for the risk of GD, when the loss function ℓ is taken from the class C 𝛽,𝑝 𝜌 . The bound appears in the following theorem,

Theorem 1. Let 𝜌 be a tail function and let ℓ be any loss function from the class C 𝛽,𝑝 𝜌 . Fix 𝑇 , 𝑛 and 𝛿 &gt; 0 . Then, with probability at least 1 -𝛿 (over the random sample 𝑆 of size 𝑛 ), the output of GD applied on b 𝐿 with step size 𝜂 = 1 / 6 𝑘 2 / 𝑝 𝛽 initialized at 𝑊 1 = 0 has for any 𝜖 ≤ 1 2 such that 𝜂𝛾 2 𝑇 ≤ ( 𝜌 -1 ( 𝜖 / 𝑘 )) 2 / 𝜖 , for 𝑝 ∈ ( 2 , ∞) , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the rest of the section we detail the main techniques which we use for proving Theorem 1.

In addition, if 𝑝 = ∞ ,

## 3.1 Bounds for the Rademacher Complexity of VVPs

Firstly, we explain our main technique, which is based on local Rademacher complexity of vectorvalued function classes. We first recall the definition of the Rademacher complexity (e.g., [1]).

Definition 6 (Rademacher complexity) . Let Z be a measurable space and D be a distribution over Z . Let F be a class of real-valued functions mapping from Z to F . Given a training set 𝑆 = { 𝑧 1 , . . . , 𝑧 𝑛 } of 𝑛 exmples that sampled i.i.d. from Z . The empirical Rademacher complexity of F is defined by

<!-- formula-not-decoded -->

where 𝜖 1 , . . . , 𝜖 𝑛 are i.i.d. Rademacher random variables. In addition, the worst-case Rademacher complexity is defined as ˆ ℜ 𝑛 (F) = sup 𝑆 ∈Z 𝑛 ℜ 𝑆 (F) .

In particular, in our work, given a loss function ℓ ∈ C 𝛽,𝑝 𝜌 , we are interested in bounding the worst case Rademacher complexity of the class

<!-- formula-not-decoded -->

where 𝔹 𝑘 × 𝑑 𝐵 = { 𝑊 ∈ ℝ 𝑘 × 𝑑 | ∥ 𝑊 ∥ 𝐹 ≤ 𝐵 } . We establish the following upper bound for the worst case Rademacher complexity of L 𝐵,𝑟 ℓ ,

Lemma 1. Let 𝜌 be a tail function and let ℓ ∈ C 𝛽,𝑝 𝜌 . Given 𝐵, 𝑟 ≥ 0 , let L 𝐵,𝑟 ℓ be as defined above. Moreover, let 𝑀 be such that every 𝑓 ∈ L 𝐵,𝑟 ℓ is bounded by 𝑀 . Then, it holds that,

<!-- formula-not-decoded -->

For the proof of Lemma 1, we use the approach of Lei et al. [6], Reeve and Kaban [16], that given a multi class classification training set 𝑆 = {( 𝑥 1 , 𝑦 1 ) . . . ( 𝑥 𝑛 , 𝑦 𝑛 ) , define a new training set with 𝑛𝑘 examples denoted as ˜ 𝑆 follows and is defined as follows

<!-- formula-not-decoded -->

where 𝜙 𝑗 ( 𝑥 ) ∈ ℝ 𝑑 × 𝑘 is the matrix which its 𝑗 th column is 𝑥 and the rest of the columns are zero. Then, it is possible to relate the covering number of L 𝐵,𝑟 ℓ , to the covering number of the following class of linear predictors when applied on ˜ 𝑆 ,

<!-- formula-not-decoded -->

The full proof of Lemma 1 appears in Appendix A. Notably, in contrast to those works, which uses the properties of the loss, we show that in the multi-class classification setting, it is sufficient to use the properties of the template ˜ ℓ .

The next step of the proof is to use Lemma 1, to bound the difference between the empirical risk and the population risk of a specific model in multi-class losses. Such a result appears in the following theorem,

Theorem 2. Let 𝜌 be a tail function and let ℓ ∈ C 𝛽,𝑝 𝜌 . Given 𝐵, 𝑟 ≥ 0 , Let L 𝐵,𝑟 ℓ be as defined above. Moreover, Let 𝑀 be such that every 𝑓 ∈ L 𝐵,𝑟 ℓ is bounded by 𝑀 . Then, for any 𝛿 &gt; 0 we have, with probability at least 1 -𝛿 over a random sample of size 𝑛 , for any 𝑊 ∈ 𝔹 𝑘 × 𝑑 𝐵 ,

<!-- formula-not-decoded -->

## 3.2 Implications of Template Geometry on Optimization

Next, we discuss how the geometry of the template influences the optimization error of GD. The key insight is that while the template ˜ ℓ is 𝑂 ( 1 ) -smooth, this smoothness does not necessarily extend to the loss function ℓ with respect to the model 𝑊 . In fact, the latter is highly dependent on the geometry of the template, as formalized in the following lemma (see proof in Appendix A):

Lemma 2. Let ∥ 𝑥 ∥ 2 ≤ 1 , 𝑦 ∈ [ 𝑘 ] and ˜ ℓ ∈ ˜ C 𝛽,𝑝 𝜌 for 𝑝 ≥ 2 . Let ℓ ( 𝑥,𝑦 ) : ℝ 𝑘 × 𝑑 → ℝ be ℓ ( 𝑥,𝑦 ) ( 𝑊 ) = ℓ 𝑦 ( 𝑊𝑥 ) = ˜ ℓ ( 𝐷 𝑦 𝑊𝑥 ) Then, for every 𝑊,𝑊 ′ ∈ ℝ 𝑘 × 𝑑 ,

<!-- formula-not-decoded -->

Since the optimal step size for GD on general 𝛽 -smooth functions (with respect to 𝑊 ) is approximately 𝜂 ≈ 1 / 𝛽 , Lemma 2 shows that the optimal step size increases with 𝑝 . Substituting this into the convergence bound for the optimization error of GD leads to improved convergence rates as 𝑝 grows, as formalized in the following lemma (see proof in Appendix A),

Lemma 3. Let 𝜌 be a tail function and let ℓ ∈ C 𝛽,𝑝 𝜌 . Fix any 𝜖 &gt; 0 and a point 𝑊 ∗ 𝜖 ∈ ℝ 𝑘 × 𝑑 such that b 𝐿 ( 𝑊 ∗ 𝜖 ) ≤ 𝜖 . Then, the output of 𝑇 -iterations GD, applied on b 𝐿 with step size 𝜂 = 1 / 6 𝑘 2 / 𝑝 𝛽 initialized at 𝑊 1 = 0 has,

<!-- formula-not-decoded -->

## 3.3 Proof of Theorem 1

We are now ready to prove Theorem 1. The proof proceeds by first showing that the iterates of GD remain within a bounded region around the origin; this is established in Lemma 14 (see Appendix A). Next, we combine the bound on the generalization gap bound from with the low-noise guarantee implied by Lemma 3 to complete the argument for Theorem 1. The full proof is detailed below.

Proof of Theorem 1. First, let 𝑝 ∈ ( 2 , ∞) . First, for 𝜖 such that 𝜂𝛾 2 𝑇 ≤ ( 𝜌 -1 ( 𝜖 𝑘 )) 2 / 𝜖 , we get by Lemma 14 and Lemma 12 (see Appendix A),

<!-- formula-not-decoded -->

For the same 𝜖 , by Lemmas 3 and 12,

<!-- formula-not-decoded -->

Now, we denote B 𝜖 = { 𝑊 ∈ ℝ 𝑘 × 𝑑 ∥ 𝑊 ∥ 𝐹 ≤ 𝐵 𝜖 } . Moreover, by Lemma 14 and Lemmas 12 to 14 (see Appendix A, we know that, with probability 1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, by Theorem 2, for any 𝛿 &gt; 0 we have, with probability at least 1 -𝛿 over a random sample of size 𝑛 , for any 𝑊 ∈ 𝜖 , there exists a constant 𝐶 &gt; 0 such that 𝐶 depends poly-logarithmically on 𝑘, 𝑛, 𝑀 𝜖 , 𝛽, 1 𝛿 and

<!-- formula-not-decoded -->

For 𝑊 𝑇 by the choice of 𝜂 , we get,

<!-- formula-not-decoded -->

For 𝑝 = ∞ , since any 𝛽 - smooth function w.r.t 𝐿 ∞ is also 𝛽 smooth with respect to the 𝐿 𝑘 norm, we get that, since 𝑥 1 / 𝑥 ≤ 𝑒 &lt; 3 for any 𝑥 ∈ ℝ ,

<!-- formula-not-decoded -->

## 4 Tightness in the Euclidean case

In this section, we show that the non-trivial dependence on 𝑘 given in Theorem 1 for 𝑝 = 2 is unavoidable. We prove this by establishing the following lower bound:

Theorem 3. Let 𝑝 = 2 and 𝛾 ≤ 1 8 . For any tail function 𝜌 , sample size 𝑛 ≥ 35 and any 𝑇 , there exist a distribution D and a loss function ℓ ∈ C 𝛽,𝑝 𝜌 , such that for 𝑇 -steps GD over a sample 𝑆 = {( 𝑥 𝑖 , 𝑦 𝑖 )} 𝑛 𝑖 = 1 sampled i.i.d. from D , initialized at 𝑊 1 = 0 with stepsize 𝜂 = 1 / 6 𝛽𝑘 , it holds that

<!-- formula-not-decoded -->

for any 𝜖 &lt; 1 256 such that 𝜂𝛾 2 𝑇 ≥ 1 𝜖 ( 𝜌 -1 (( 𝜖 𝑘 ))) 2 .

For the proof of Theorem 3, we prove two lemmas. first show the following lemma, which provides a tight lower bound for the case in which 𝑇 ≥ 𝑛 ,

Lemma 4. Let 𝛾 ≤ 1 8 and 𝜖 &gt; 0 be such that 𝜌 -1 ( 𝜖 𝑘 ) 2 𝜂𝛾 2 𝑇 ≤ 𝜖 ≤ 1 256 . For any tail function 𝜌 , sample size 𝑛 ≥ 35 and any and 𝑇 , there exist a distribution D with margin 𝛾 , a loss function ℓ ∈ C 𝛽,𝑝 𝜌 for 𝑝 = 2 such that for GD over a sample 𝑆 = { 𝑧 𝑖 } 𝑛 𝑖 = 1 sampled i.i.d. from D , initialized at 𝑊 1 = 0 with step size 𝜂 ≤ 1 6 𝛽𝑘 , it holds that

<!-- formula-not-decoded -->

Second, in the following lemma we give a tight lower bound for the case where 𝑇 ≤ 𝑛 .

Lemma 5. Let 𝛾 ≤ 1 8 and 𝜖 &gt; 0 be such that 𝜌 -1 ( 𝜖 𝑘 ) 2 𝛾 2 𝜂𝑇 ≤ 𝜖 ≤ 1 16 . For any tail function 𝜌 , 𝑇 , there exist a distribution D with margin 𝛾 , a loss function ℓ ∈ C 𝛽,𝑝 𝜌 such that for GD over a sample 𝑆 = { 𝑧 𝑖 } 𝑛 𝑖 = 1 sampled i.i.d. from D , initialized at 𝑊 1 = 0 with step size 𝜂 ≤ 1 6 𝛽𝑘 , it holds that

<!-- formula-not-decoded -->

Below, we provide a sketch of the proof for Lemmas 4 and 5. The full proofs and the derivation of Theorem 3 can be found in Appendix B.

To construct a hard instance for the Euclidean case and prove Lemmas 4 and 5, our main observation is that for a univariate loss function 𝜙 : ℝ → ℝ , the template ˜ ℓ : ℝ 𝑘 -1 → ℝ , which applies 𝜙 to each entry of its input and sums the results, satisfies ˜ ℓ ∈ ˜ C 𝛽,𝑝 𝜌 for 𝑝 = 2 . This is established in the following lemma (see proof in Appendix B):

Lemma 6. Let ˜ ℓ : ℝ 𝑘 -1 → ℝ such that there exists a function 𝜙 ∈ ℝ → ℝ and ˜ ℓ ( 𝑤 ) = ˝ 𝑘 -1 𝑗 = 1 𝜙 ( 𝑤 [ 𝑗 ]) . Then, if 𝜙 is nonnegative, convex, 𝛽 -smooth and monotonically decreasing loss function such that 𝜙 ( 𝑢 ) ≤ 𝜌 ( 𝑢 ) for all 𝑢 ≥ 0 and some function tail function 𝜌 , it holds that ˜ ℓ ∈ ˜ C 𝛽,𝑝 𝜌 for 𝑝 = 2 .

Next, to construct the hard instance, we design loss functions that represent the sum of 𝑘 hard binary classification instances. Combining this with a construction similar to that of Schliserman and Koren [19] for the latter case, we derive a multi-class classification lower bound for loss functions with smooth templates with respect to 𝐿 2 .

## 5 Examples

In this section, we apply our general generalization bounds for gradient methods in the setting of multi-class classification with several popular choices of loss function, demonstrating how the geometry of the loss function affect the generalization properties of Gradient Descent.

## 5.1 Exponentially-tailed losses

First, we show a risk bound for Gradient Descent, when the decay rate of loss the loss is exponential, i.e. when ℓ ∈ C 𝛽,𝑝 𝜌 for 𝜌 ( 𝑥 ) = 𝑒 -𝑥 . We can apply Theorem 1 with 𝜖 = 1 𝑇 and get the following,

Corollary 4. Let ℓ ∈ C 𝛽,𝑝 𝜌 for 𝜌 ( 𝑥 ) = 𝑒 -𝑥 . Then, the output of Gradient Descent on b 𝐿 with step size 𝜂 = 1 6 𝑘 2 𝑝 and 𝑊 1 = 0 satisfies

<!-- formula-not-decoded -->

A particular loss function in the class of losses with exponentially decaying template is the cross entropy loss, i.e., for every 𝑦 ∈ [ 𝑘 ] , ℓ 𝑦 ( ˆ 𝑦 ) = log GLYPH&lt;16&gt; 1 + ˝ 𝑗 ≠ 𝑦 exp ( ˆ 𝑦 [ 𝑦 ] -ˆ 𝑦 [ 𝑗 ]) GLYPH&lt;17&gt; , whose template is smooth with respect to the 𝐿 ∞ norm (see Lemma 22 in Appendix C). Next, we can derive an upper bound for GD which is logarithmic in the number of classes. For this, we apply Theorem 1 with 𝜖 = 1 𝑇 and obtain the following result,

Corollary 5. If ℓ is the cross entropy loss function, the output of Gradient Descent on b 𝐿 with step size 𝜂 = 1 12 and 𝑊 1 = 0 satisfies

<!-- formula-not-decoded -->

This bound matches the upper bound of Schliserman and Koren [17] for Gradient Descent on the cross entropy loss, and, up to logarithmic factors matches the bounds given in Schliserman and Koren [19] for the case of setting of binary classification with smooth losses with exponential tail. In contrast, using Theorem 3 with 𝜖 = log 2 ( 𝑘𝑇 ) 𝜂𝛾 2 𝑇 we get:

Corollary 6. There exists a function ℓ ∈ C 𝛽,𝑝 𝜌 for 𝑝 = 2 and 𝜌 ( 𝑥 ) = 𝑒 -𝑥 such that the output of Gradient Descent on b 𝐿 with step size 𝜂 = 1 6 𝑘 and 𝑊 1 = 0 holds,

<!-- formula-not-decoded -->

Combining Corollaries 5 and 6, we get a separation between exponentially tailed losses with templates that are smooth w.r.t the 𝐿 ∞ -norm-such as the cross-entropy loss, where the risk matches the binary case up to logarithmic factors, and the 𝐿 2 -norm case, the upper bounds exhibit an unavoidable linear dependence on the number of classes. This differ but not at odds with the results of [15], which suggest that exponentially tailed losses exhibit similar asymptotic behavior.

## 5.2 Polynomially-tailed losses

Now we show application of our generalization bound for Gradient Descent, when the decay rate of loss the loss is polynomial, i.e., when ℓ ∈ C 𝛽,𝑝 𝜌 for 𝜌 ( 𝑥 ) = 𝑥 -𝛼 for some 𝛼 &gt; 0 . For giving an upper bound for polynomially tailed losses, we can apply Theorem 1 with for this class of functions

<!-- formula-not-decoded -->

Corollary 7. Let ℓ ∈ C 𝛽,𝑝 𝜌 for 𝜌 ( 𝑥 ) = 𝑥 -𝛼 . Then, the output of Gradient Descent on b 𝐿 with step size 𝜂 = 1 6 𝑘 2 𝑝 and 𝑊 1 = 0 holds,

<!-- formula-not-decoded -->

«

‹

## 6 Discussion and Limitations

In this work, we provide the first finite-time population risk bounds for gradient descent in linearly separable multiclass classification. Our results show that the geometry of the loss, captured through the ℓ 𝑝 -smoothness of its template, plays a central role in both convergence and generalization. In contrast to prior views that emphasize the decay rate of the loss or the implicit bias of gradient methods, our analysis reveals that smoothness geometry determines how generalization of gradient descent depends on the number of classes across different multiclass regimes.

Our analysis assumes linear predictors and linearly separable data, which, while standard in theoretical studies, limits direct applicability to nonlinear or noisy settings. As a result, our results should be seen as a theoretical foundation that helps explain generalization in simpler settings, rather than a direct description of deep learning in practice. Despite these assumptions, our insights may suggest broader implications. The dependence of the bounds on ℓ 𝑝 -smoothness offers an explanation for the empirical success of cross-entropy and other ℓ ∞ -smooth losses in large-scale or extreme classification, where the number of classes is high.

Future work. Having established the first finite-time risk bounds for gradient descent in the multiclass separable setting, several open directions remain. A natural next step is to extend our analysis to nonlinear predictors and nonseparable data, and to examine empirically whether the geometric separation between smoothness norms also arises in more complex regimes. An especially relevant example is classifier-head fine-tuning in deep networks, where the data are typically nonseparable and multi-labeled, in contrast to the single-label setting considered in this work. Another promising direction is to further study the implicit bias of gradient methods for loss functions with general, potentially non-exponential tail decay rates (e.g., polynomial tails), and investigate whether it implies nontrivial multiclass risk bounds, similar to those established in this paper. This question is particularly interesting given that, in the binary case, the implicit-bias characterization of the gradient descent solutions leads to strictly suboptimal bounds as compared to the state-of-the-art [19] (see a more elaborate discussion therein).

## Acknowledgments

This project has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement No. 101078075). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them. This work received additional support from the Israel Science Foundation (ISF, grant numbers 2549/19 and 3174/23), a grant from the Tel Aviv University Center for AI and Data Science (TAD) and from the Len Blavatnik and the Blavatnik Family foundation.

## References

- [1] P. L. Bartlett and S. Mendelson. Rademacher and gaussian complexities: Risk bounds and structural results. Journal of Machine Learning Research , 3(Nov):463-482, 2002.
- [2] O. Bousquet. Concentration inequalities and empirical processes theory applied to the analysis of learning algorithms. Journal of Machine Learning Research , 01 2002.
- [3] Z. Ji and M. Telgarsky. Risk and parameter convergence of logistic regression. arXiv preprint arXiv:1803.07300 , 2018.
- [4] Z. Ji and M. Telgarsky. A refined primal-dual analysis of the implicit bias. Journal of Environmental Sciences (China) English Ed , 2019.
- [5] Z. Ji, M. Dudík, R. E. Schapire, and M. Telgarsky. Gradient descent follows the regularization path for general losses. In J. Abernethy and S. Agarwal, editors, Proceedings of Thirty Third Conference on Learning Theory , volume 125 of Proceedings of Machine Learning Research , pages 2109-2136. PMLR, 09-12 Jul 2020.
- [6] Y. Lei, Ü. Dogan, D.-X. Zhou, and M. Kloft. Data-dependent generalization bounds for multi-class classification. IEEE Transactions on Information Theory , 65(5):2995-3021, 2019.
- [7] J. Li, Y . Liu, R. Yin, H. Zhang, L. Ding, and W. Wang. Multi-class learning: From theory to algorithm. Advances in Neural Information Processing Systems , 31, 2018.
- [8] K. Lyu and J. Li. Gradient descent maximizes the margin of homogeneous neural networks. arXiv preprint arXiv:1906.05890 , 2019.
- [9] K. Lyu, Z. Li, R. Wang, and S. Arora. Gradient descent on two-layer nets: Margin maximization and simplicity bias. Advances in Neural Information Processing Systems , 34:12978-12991, 2021.
- [10] R. Magen and O. Shamir. Initialization-dependent sample complexity of linear predictors and neural networks. Advances in Neural Information Processing Systems , 36, 2024.
- [11] A. Maurer. A vector-contraction inequality for rademacher complexities. In Algorithmic Learning Theory: 27th International Conference, ALT 2016, Bari, Italy, October 19-21, 2016, Proceedings 27 , pages 3-17. Springer, 2016.
- [12] M. S. Nacson, J. Lee, S. Gunasekar, P. H. P. Savarese, N. Srebro, and D. Soudry. Convergence of gradient descent on separable data. In The 22nd International Conference on Artificial Intelligence and Statistics , pages 3420-3428. PMLR, 2019.
- [13] M. S. Nacson, N. Srebro, and D. Soudry. Stochastic gradient descent on separable data: Exact convergence with a fixed learning rate. In The 22nd International Conference on Artificial Intelligence and Statistics , pages 3051-3059. PMLR, 2019.
- [14] A. Rakhlin, K. Sridharan, and A. Tewari. Sequential complexities and uniform martingale laws of large numbers. Probability Theory and Related Fields , 161(1):111-153, 2015. doi: 10.1007/s00440-013-0545-5. URL https://doi.org/10.1007/s00440-013-0545-5 .
- [15] H. Ravi, C. Scott, D. Soudry, and Y. Wang. The implicit bias of gradient descent on separable multiclass data. arXiv preprint arXiv:2411.01350 , 2024.
- [16] H. Reeve and A. Kaban. Optimistic bounds for multi-output learning. In International Conference on Machine Learning , pages 8030-8040. PMLR, 2020.
- [17] M. Schliserman and T. Koren. Stability vs implicit bias of gradient methods on separable data and beyond. In P.-L. Loh and M. Raginsky, editors, Proceedings of Thirty Fifth Conference on Learning Theory , volume 178 of Proceedings of Machine Learning Research , pages 3380-3394. PMLR, 02-05 Jul 2022.
- [18] M. Schliserman and T. Koren. Complexity of vector-valued prediction: From linear models to stochastic convex optimization. arXiv preprint arXiv:2412.04274 , 2024.

- [19] M. Schliserman and T. Koren. Tight risk bounds for gradient descent on separable data. Advances in Neural Information Processing Systems , 36, 2024.
- [20] O. Shamir. Gradient methods never overfit on separable data. The Journal of Machine Learning Research , 22(1):3847-3866, 2021.
- [21] D. Soudry, E. Hoffer, M. S. Nacson, S. Gunasekar, and N. Srebro. The implicit bias of gradient descent on separable data. The Journal of Machine Learning Research , 19(1):2822-2878, 2018.
- [22] N. Srebro, K. Sridharan, and A. Tewari. Smoothness, low noise and fast rates. In J. Lafferty, C. Williams, J. Shawe-Taylor, R. Zemel, and A. Culotta, editors, Advances in Neural Information Processing Systems , volume 23. Curran Associates, Inc., 2010.
- [23] M. Telgarsky. Stochastic linear optimization never overfits with quadratically-bounded losses on general data. In P.-L. Loh and M. Raginsky, editors, Proceedings of Thirty Fifth Conference on Learning Theory , volume 178 of Proceedings of Machine Learning Research , pages 5453-5488. PMLR, 02-05 Jul 2022.
- [24] Y. Wang and C. Scott. Unified binary and multiclass margin-based classification. Journal of Machine Learning Research , 25(143):1-51, 2024.
- [25] L. Wu, A. Ledent, Y. Lei, and M. Kloft. Fine-grained generalization analysis of vector-valued learning. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 35, no. 12, pages 10338-10346, 2021.
- [26] Y. Zhang and M.-L. Zhang. Generalization analysis for multi-label learning. In R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning , volume 235 of Proceedings of Machine Learning Research , pages 60220-60243. PMLR, 21-27 Jul 2024.

## A Proofs for Section 3

We begin by the following standard lemma for smooth functions (e.g. Srebro et al. [22]).

Lemma 7. Let 𝑓 : ℝ 𝑑 → ℝ be a non-negative 𝛽 -smooth loss function with respect to 𝐿 𝑝 norm. Then, we have for every 𝑤 ∈ ℝ 𝑑 ,

<!-- formula-not-decoded -->

where 𝑞 is such that 1 𝑝 + 1 𝑞 = 1 .

Lemma 8. Let 𝑝 ∈ [ 1 , ∞] . Let 𝑓 : ℝ 𝑘 → ℝ be a non-negative 𝛽 -smooth function with respect to 𝐿 𝑝 norm. Then, for every 𝑢, 𝑣 ∈ ℝ 𝑘 , it holds that,

<!-- formula-not-decoded -->

Proof. Let 𝑞 ∈ [ 1 , ∞] be such that 1 𝑝 + 1 𝑞 = 1 . First, by the mean value theorem for any 𝑢, 𝑣 ∈ ℝ 𝑘 there exists 𝑥 on the line between 𝑣 and 𝑢 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, if ∥ 𝑢 -𝑣 ∥ 𝑝 ≤ ∥∇ 𝑓 ( 𝑣 ) ∥ 𝑞 5 𝛽 then ∥∇ 𝑓 ( 𝑥 )∥ 𝑞 ≤ 6 5 ∥∇ 𝑓 ( 𝑣 )∥ 𝑞 , by Cauchy-Schwartz inequality and Lemma 7, we get

<!-- formula-not-decoded -->

Otherwise, we know that ∥∇ 𝑓 ( 𝑥 )∥ 𝑞 ≤ 6 𝛽 ∥ 𝑢 -𝑣 ∥ 𝑝 , and derive

<!-- formula-not-decoded -->

By smoothness, we know that

As a result,

□

Now, for every class F defined on a space Z , 𝑝 ∈ [ 1 , ∞] , 𝜖 &gt; 0 and training set 𝑆 = { 𝑧 1 , . . . , 𝑧 𝑛 } ∈ Z 𝑛 , we denote by N 𝑝 (F , 𝜖 , 𝑛 ) the 𝐿 𝑝 -covering number of F , i.e., the size of a minimal cover C 𝜖 such that ∀ 𝑓 ∈ F , ∃ 𝑓 𝜖 ∈ C 𝜖 s.t. ∥ ˜ 𝑓 ( 𝑆 ) -˜ 𝑓 𝜖 ( 𝑆 )|| 𝑝 ≤ 𝜖 , where for every 𝑓 ∈ F , ˜ 𝑓 : Z 𝑛 → ℝ 𝑛 is the function that for every set 𝑆 = { 𝑧 1 , . . . , 𝑧 𝑛 } , the 𝑖 th entry of ˜ 𝑓 ( 𝑆 ) is 𝑓 ( 𝑧 𝑖 ) .

Lemma 9 ([22, 14, 6]) . Let F be a class of real-valued functions defined on a space e Z and 𝑆 ′ : = { ˜ 𝑧 1 , . . . , ˜ 𝑧 𝑛 } ∈ ˜ Z 𝑛 of cardinality 𝑛 .

1. If functions in F take values in [-𝐵, 𝐵 ] , then for any 𝜖 &gt; 0 with fat 𝜖 (F) &lt; 𝑛 we have

<!-- formula-not-decoded -->

2. For any 𝜖 &gt; 2 ˆ ℜ 𝑛 (F) , we have fat 𝜖 (F) &lt; 16 𝑛 𝜖 2 ˆ ℜ 𝑛 (F) 2 .

3. Let 𝑀 = sup 𝑓 . The Rademacher complexity ℜ 𝑆 ′ (F) satisfies

<!-- formula-not-decoded -->

Lemma 10. Let 𝑊 ∈ ℝ 𝑘 × 𝑑 , 𝑥 ∈ ℝ 𝑑 , 𝑗 ∈ [ 𝑘 ] . Then, for 𝜙 𝑗 ( 𝑥 ) defined in Theorem 1 it holds that,

<!-- formula-not-decoded -->

where 𝑊 𝑗 is the 𝑗 th row of 𝑊 .

Proof. By the definition of 𝜙 𝑗 ( 𝑥 ) , it holds that,

<!-- formula-not-decoded -->

Lemma 11. (Proposition 7 in [6]) Let H 𝐵 as defined above. Then, it holds that,

<!-- formula-not-decoded -->

Now we can prove Lemma 1 and Theorem 2.

Proof of Lemma 1. First, notice that for every 𝑦 ∈ [ 𝑘 ] and 𝑣 ∈ ℝ 𝑘 , for every 𝑗 ≠ 𝑦 , the 𝑗 th index of 𝐷 𝑦 𝑣 is 𝑣 [ 𝑦 ] -𝑣 [ 𝑗 ] , we obtain that, ∥ 𝐷 𝑦 𝑣 ∥∞ ≤ 2 ∥ 𝑣 ∥∞ .

Then, by Lemma 8 and the properties of ˜ ℓ we have,

<!-- formula-not-decoded -->

and get by Lemma 10

<!-- formula-not-decoded -->

We derive that

<!-- formula-not-decoded -->

Now by Lemma 9, for every training set 𝑆 it holds that

<!-- formula-not-decoded -->

,

□

<!-- formula-not-decoded -->

□

Proof of Theorem 2. By the displayed equation prior to the last one in the proof of the theorem Theorem 6.1 of [2] we have that if 𝜓 𝑛 is any sub-root function that satisfies for all 𝑟 &gt; 0 , ˆ ℜ 𝑛 GLYPH&lt;16&gt; L 𝐵,𝑟 ℓ GLYPH&lt;17&gt; ≤ 𝜓 𝑛 ( 𝑟 ) then, for any 𝛿 &gt; 0 , with probability at least 1 -𝛿 , for any 𝑊 ∈ 𝔹 𝑘 × 𝑑 𝐵 ,

<!-- formula-not-decoded -->

where 𝑟 ∗ 𝑛 is the largest solution to equation 𝜓 𝑛 ( 𝑟 ) = 𝑟 . Now by Lemma 1 there exists a constant 𝐶 &gt; 0 such that 𝐶 depends polylogarithmically on 𝑘, 𝑛, 𝑀, 𝛽 such that for 𝜓 𝑛 ( 𝑟 ) = 𝐶 √ 𝛽𝑟𝑘 1 𝑝 𝐵 + 1 √ 𝑛 , ˆ ℜ 𝑛 GLYPH&lt;16&gt; L 𝐵,𝑟 ℓ GLYPH&lt;17&gt; satisfies the property that for all 𝑟 &gt; 0 . Thus, for 𝑟 ∗ 𝑛 = 𝐶 2 𝛽𝑘 2 𝑝 ( 𝐵 + 1 ) 2 𝑛 (3) holds. Now by the fact that for any non-negative 𝐴, 𝐵, 𝐶 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we get

<!-- formula-not-decoded -->

The theorem holds with a factor of ˜ 𝐶 = 110 GLYPH&lt;16&gt; log 1 𝛿 + log log 𝑛 GLYPH&lt;17&gt; 𝐶 2

.

<!-- formula-not-decoded -->

Proof of Lemma 2. First, similarly to Lemma 4.2 in [15], note that the expression for the gradient of ℓ ( 𝑥,𝑦 ) w.r.t to 𝑊 is ∇ 𝑊 ℓ ( 𝑥,𝑦 ) ( 𝑊 ) = 𝑥 ∇ ˜ ℓ ( 𝐷 𝑦 𝑊𝑥 ) 𝑇 𝐷 𝑦 . Let 𝑞 be such that 1 𝑝 + 1 𝑞 = 1 . Then, it holds that

<!-- formula-not-decoded -->

where for every matrix 𝐴 , ∥ 𝐴 ∥ 𝑞, 2 is sup ∥ 𝑣 ∥ 𝑞 = 1 ∥ 𝐴𝑣 ∥ 2 . Now, by the expression for 𝐷 𝑦 it holds that, the 𝑦 th row of 𝐷 𝑇 𝑦 is the vector with all entries as 1 and the rest of the rows with index 𝑗 th row is a negative standard basis vector, we get that

<!-- formula-not-decoded -->

Moreover, since, for every 𝑦 ∈ [ 𝑘 ] and 𝑣 ∈ ℝ 𝑘 , for every 𝑗 ≠ 𝑦 , the 𝑗 th index of 𝐷 𝑦 𝑣 is 𝑣 [ 𝑦 ] -𝑣 [ 𝑗 ] , we obtain that, ∥ 𝐷 𝑦 𝑣 ∥∞ ≤ 2 ∥ 𝑣 ∥∞ . , and First, notice that for every 𝑦 ∈ [ 𝑘 ] and 𝑣 ∈ ℝ 𝑘 , it holds that,

<!-- formula-not-decoded -->

Then, we conclude that

<!-- formula-not-decoded -->

The lemma follows by taking a square root of both sides.

□

Lemma 12. Let 𝜌 be a tail function and let ℓ ∈ C 𝛽,𝑝 𝜌 . Fix any 0 &lt; 𝜖 &lt; 1 2 . The, there exists a model 𝑊 ∗ 𝜖 ∈ ℝ 𝑘 × 𝑑 such that ∥ 𝑊 ∗ 𝜖 ∥ 𝐹 ≤ 𝜌 -1 ( 𝜖 𝑘 ) 𝛾 and b 𝐿 ( 𝑊 ∗ 𝜖 ) ≤ 𝜖 .

Proof. By separability, there exists a model 𝑊 ∗ such that ∥ 𝑊 ∗ ∥ 𝐹 ≤ 1 and such that for every 𝑗 ∈ [ 𝑘 ] \ { 𝑦 𝑖 } , it holds that ( 𝑊 𝑦 𝑖 ∗ -𝑊 𝑗 ∗ ) ⊤ 𝑥 𝑖 ≥ 𝛾 for every ( 𝑥 𝑖 , 𝑦 𝑖 ) in the training set 𝑆 .

Now, let 𝑊 1 𝑖 , . . . , 𝑊 𝑘 -1 𝑖 ∈ 𝑅 𝑘 be the rows of 𝐷 𝑦 𝑖 𝑊 ∗ . Note that the seperability condition is equivalent to the fact that 𝑊 𝑗 𝑖 · 𝑥 𝑖 ≥ 𝛾 for any 𝑗 ∈ [ 𝑘 -1 ] . Then, for 𝑊 ∗ 𝜖 = 𝜌 -1 ( 𝜖 𝑘 ) 𝛾 𝑊 ∗ and every ( 𝑥 𝑖 , 𝑦 𝑖 ) ∈ 𝑆 ,

<!-- formula-not-decoded -->

b b

□

Lemma 13. Let 𝑦 ∈ 𝑘 and ℓ ∈ C 𝛽,𝑝 𝜌 for 𝑝 ≥ 2 . For every 𝑊,𝑊 ′ ∈ ℝ 𝑘 × 𝑑 such that ∥ 𝑊 -𝑊 ′ ∥ 𝐹 ≤ 𝑅 and 𝑥 ∈ ℝ 𝑑 with ∥ 𝑥 ∥ 2 ≤ 1 , it holds that,

<!-- formula-not-decoded -->

Proof of Lemma 13. Let 𝑊,𝑊 ′ ∈ ℝ 𝑘 × 𝑑 such that ∥ 𝑊 -𝑊 ′ ∥ 𝐹 ≤ 𝑅 and 𝑥 ∈ ℝ 𝑑 . Moreover, Let 𝑞 be such that 1 𝑝 + 1 𝑞 = 1 . First, notice that for every 𝑦 ∈ [ 𝑘 ] and 𝑣 ∈ ℝ 𝑘 , it holds that, ∥ 𝐷 𝑦 𝑣 ∥∞ ≤ 2 ∥ 𝑣 ∥∞ . Then, by smoothness w.r.t 𝐿 𝑝 and Lemma 7 it holds that

<!-- formula-not-decoded -->

where the second inequality follows by the fact that for every 𝛾 ≥ 0 and 𝑥, 𝑦 ∈ ℝ 𝑘 , it holds that 𝑥𝑦 ≤ 1 2 𝛾 𝑥 2 + 𝛾 2 𝑦 2 . □

Lemma 14. Fix any 𝜖 &gt; 0 and a point 𝑊 ∗ 𝜖 ∈ ℝ 𝑘 × 𝑑 such that b 𝐿 ( 𝑊 ∗ 𝜖 ) ≤ 𝜖 . Then, the output of 𝑇 -iterations GD, applied on b 𝐿 with step size 𝜂 ≤ 1 / 6 𝑘 2 𝑝 𝛽 initialized at 𝑊 1 = 0 has,

<!-- formula-not-decoded -->

Proof. Let ˜ 𝛽 = 3 𝑘 2 𝑝 𝛽 . First, by Lemma 2, b 𝐿 is ˜ 𝛽 -smooth with respect to 𝑊 and Lemma 7, we know that ∥∇ 𝐿 ( 𝑊 )∥ 2 ≤ 2 ˜ 𝛽 𝐿 ( 𝑊 ) for any 𝑊 . Therefore, by using 𝜂 ≤ 1 / ˜ 𝛽 , for every 𝜖 ,

<!-- formula-not-decoded -->

By summing until time 𝑇 ,

<!-- formula-not-decoded -->

The lemma follows by taking a square root and using triangle inequality.

□

Proof of Lemma 3. Let ˜ 𝛽 = 3 𝑘 2 𝑝 𝛽 . First, by Lemma 2, b 𝐿 is ˜ 𝛽 -smooth with respect to 𝑊 , thus, for every 𝑡 and 𝜂 ≤ 1 / ˜ 𝛽 ,

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

Moreover, from standard regret bounds for gradient updates, for any 𝑊 ∈ ℝ 𝑘 × 𝑑 ,

<!-- formula-not-decoded -->

By Lemma 7,

<!-- formula-not-decoded -->

Using 𝜂 ≤ 1 / 2 ˜ 𝛽 gives

<!-- formula-not-decoded -->

For 𝑊 = 𝑊 ∗ 𝜖 we get by Eq. (4),

<!-- formula-not-decoded -->

When 𝜂 = 1 6 𝛽𝑘 2 𝑝 , we get the lemma.

## B Proofs for Section 4

Proof of Lemma 6. The non-negativity and convexity is implied directly by the fact that ˜ ℓ is a sum of non-negative convex functions. Moreover, for every 𝑢 ∈ ( ℝ + ) 𝑘 ,

<!-- formula-not-decoded -->

and, since 𝜌 decays to zero at infinity

<!-- formula-not-decoded -->

It is left to prove the smoothness of ˜ ℓ . For every, 𝑢, 𝑣 ∈ ℝ 𝑘 -1 , it holds that

<!-- formula-not-decoded -->

□

Then,

Then,

We know that

Furthermore,

<!-- formula-not-decoded -->

Combining these, we obtain

𝐴

) ≥

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Pr

(

𝐴

1

) ·

Pr

(

𝐴

2

|

𝐴

1

) ≥

<!-- formula-not-decoded -->

1

120

𝑒𝑛

.

□

Now we turn to prove lemmas that we use in the proof of Theorem 3 We begin with probabilistic claims that similar to Schliserman and Koren [19].

Lemma 15. Let D be the distribution defined in Eq. (5) . Let 𝑆 ∼ D 𝑛 be a sample of size 𝑛 , and let ( 𝑥 ′ , 𝑦 ′ ) ∼ D be a validation example. Moreover, assume 𝑛 ≥ 35 and let 𝛿 2 be the fraction of ( 𝑥 2 , 1 ) in 𝑆 . We define the following event,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. The proof follows directly from Lemma 16 and Lemma 17.

We define the following events:

<!-- formula-not-decoded -->

By Lemma 16, we have

<!-- formula-not-decoded -->

By Lemma 17, we further have

Combining these results, we get

Pr

(

□

Lemma 16. Let D be the distribution defined in Eq. (5) . Let 𝑆 ∼ D 𝑛 be a sample of size 𝑛 , and let ( 𝑥 ′ , 𝑦 ′ ) ∼ D be a validation example. Let 𝐴 1 be the following event,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. First, observe that, since 𝑦 is deterministic,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

□

Lemma 17. Let D be the distribution defined in Eq. (5) . Assume 𝑛 ≥ 35 and let 𝛿 2 denote the fraction of ( 𝑥 2 , 1 ) in 𝑆 . We define the following events:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. For every 𝑥 𝑖 ∈ 𝑆 , let 𝑝 ′ 𝑖 = Pr ( 𝑥 𝑖 = 𝑥 2 | 𝐴 1 ) . Since 𝑥 𝑖 and 𝑥 𝑗 are independent for 𝑖 ≠ 𝑗 , it follows that 𝑝 ′ 𝑖 = 𝑝 ′ 𝑗 for all 𝑖 ≠ 𝑗 . Using independence, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The expected value of 𝛿 2 given 𝐴 1 is

<!-- formula-not-decoded -->

The variance is

<!-- formula-not-decoded -->

Using Chebyshev's inequality, for 𝑛 ≥ 35 , we have

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

Substituting the variance, we get

For 𝑛 ≥ 35 , this simplifies to

Then,

This simplifies to

Then, ℓ ∈ C 𝛽,𝑝 𝜌 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 18. Let 𝜌 be a tail function. and 𝜙 : ℝ → ℝ be the following function

<!-- formula-not-decoded -->

Next, we define the following loss function for every 𝑦 ,

<!-- formula-not-decoded -->

□

Proof. First, for ˜ ℓ ( ˆ 𝑦 ) = ˝ 𝑘 -1 𝑗 = 1 𝜙 ( ˆ 𝑦 𝑗 ) , ℓ 𝑦 ( ˆ 𝑦 ) = ˜ ℓ ( 𝐷 𝑦 ˆ 𝑦 ) . Then, it is left to prove that ˜ ℓ ∈ ˜ C 𝛽,𝑝 𝜌 . By Lemma 6, it is sufficient to prove that 𝜙 is nonnegative, convex, 𝛽 -smooth and monotonically decreasing loss functions such that 𝜙 ( 𝑢 ) ≤ 𝜌 ( 𝑢 ) for all 𝑢 ≥ 0 .

Second, 𝜙 is non negative: for 𝑥 ≥ 0 by the non negativity of 𝜌 and for 𝑥 &lt; 0 by the fact that 𝜌 ′ ( 0 ) ≤ 0 . Moreover, 𝜙 is convex. We need to prove that every 𝑥 &lt; 𝑦 , 𝜙 ′ ( 𝑥 ) ≤ 𝜙 ′ ( 𝑦 ) For 𝑥, 𝑦 &lt; 0 , we get it by the convexity of 𝜌 . For 𝑥, 𝑦 &gt; 0 , we get it by the fact 𝜙 there is a sum of convex function and linear function. For 𝑥 &lt; 0 &lt; 𝑦 , by the convexity of 𝜌 ,

<!-- formula-not-decoded -->

In addition, 𝜙 is 𝛽 -smooth. We need to prove that every 𝑥 &lt; 𝑦 , 𝜙 ′ ( 𝑦 ) -𝜙 ′ ( 𝑥 ) ≤ 𝛽 ( 𝑦 -𝑥 ) For 𝑥, 𝑦 ≥ 0 , we get it by the smoothness of 𝜌 . For 𝑥, 𝑦 ≤ 0 , we get it by the fact that 𝜙 is a sum of 𝛽 -smooth function and a linear function. For 𝑥 ≤ 0 ≤ 𝑦 , by the smoothness of 𝜌 ,

<!-- formula-not-decoded -->

Finally, 𝜙 is strictly monotonically decreasing. We need to prove that every 𝑥 &lt; 𝑦 , 𝜙 ( 𝑦 ) &gt; 𝜙 ( 𝑥 ) . For 𝑥, 𝑦 &gt; 0 , we get it by the monotonicity of 𝜌 . For 𝑥 &lt; 𝑦 &lt; 0 ,

<!-- formula-not-decoded -->

For 𝑥 &lt; 0 &lt; 𝑦 ,

<!-- formula-not-decoded -->

□

Lemma 19. Let 𝜙 : ℝ → ℝ a univariate funcation. For every 𝑥 ∈ ℝ 𝑑 , 𝑦 ∈ [ 𝑘 ] and let ℓ 𝑥,𝑦 be the following loss function

<!-- formula-not-decoded -->

where for every 𝑗 , 𝑊 𝑗 is the 𝑗 th row of 𝑊 . Moreover, let 𝑊 𝑡 the iterate of GD with step size 𝜂 &gt; 0 , initialized on 𝑊 1 = 0 . Then, for every 𝑡 ≥ 1 , it holds that 𝑊 𝑗 𝑡 = 𝑊 2 𝑡 for any 𝑗 ≠ 1 .

Proof. We prove by induction on 𝑡 . For 𝑡 = 0 , since 𝑊 1 = 0 , the lemma trivially holds. Now, assuming 𝑊 𝑗 𝑡 = 𝑊 2 𝑡 , it holds that for every 𝑗 ≠ 1 , and for every possible example 𝑥 that the 𝑗 th row of the gradient is 𝜙 ′ (⟨ 𝑊 1 -𝑊 𝑗 , 𝑥 ⟩) 𝑥 Then, we conclude that,

<!-- formula-not-decoded -->

□

Proof of Lemma 4. Let 𝛾 ≤ 1 8 . We define the following distribution D :

<!-- formula-not-decoded -->

and the following function 𝜙 : ℝ → ℝ :

<!-- formula-not-decoded -->

Then, we define the following loss function for every sample ( 𝑥, 𝑦 ) ,

<!-- formula-not-decoded -->

First, we show that the distribution is separable. Since 𝑦 = 1 with probability 1 for the matrix 𝑊 ∗ where its first row is 𝑊 1 ∗ = ( 𝛾, 1 2 , 1 4 ) and for any other 𝑗 th row 𝑊 𝑗 ∗ = 0 , it holds for any 𝑗 ≠ 1 that

( 𝑊 1 ∗ -𝑊 𝑗 ∗ ) 𝑥 𝑖 = 𝑊 1 ∗ 𝑥 𝑖 ≥ 𝛾 for every 𝑖 ∈ { 1 , 2 , 3 } . Moreover, Lemma 18 in Appendix B shows that indeed ℓ ∈ C 𝛽,𝑝 𝜌 .

Next, let 𝑆 be a sample of 𝑛 i.i.d. examples from D and let ( 𝑥 ′ , 𝑦 ′ ) ∼ D be a validation example independent from 𝑆 . We denote by 𝛿 2 ∈ [ 0 , 1 ] the fraction of appearances of ( 𝑥 2 , 1 ) in the sample 𝑆 , and by 𝐴 1 , 𝐴 2 the following events;

<!-- formula-not-decoded -->

In Lemma 15 (in Appendix B), we show that

<!-- formula-not-decoded -->

Then by Lemma 3 and the choice of 𝜖 ,

<!-- formula-not-decoded -->

Now, for every 𝑗 ≠ 1 , 𝑡 ∈ [ 𝑇 ] , we denote, 𝑈 𝑗 𝑡 = 𝑊 1 𝑡 -𝑊 𝑗 𝑡 . For the rest of the proof, we condition on the event 𝐴 1 ∩ 𝐴 2 .

First, we show that for every 𝑗 ≠ 1 it hold that 𝑈 𝑗 𝑡 · 𝑥 2 ≥ 0 . Indeed, if it were not the case, by Lemma 19, then 𝑈 2 𝑡 · 𝑥 2 ≥ 0 and it implies that 𝜙 ( 𝑈 2 𝑡 · 𝑥 2 ) &gt; 𝜌 ( 0 ) ; together with Eq. (8) we obtain,

<!-- formula-not-decoded -->

which is a contradiction to 𝜌 ( 0 ) ≥ 1 . Moreover, it holds for every 𝑗 ≠ 1 that 𝑈 𝑗 𝑇 [ 1 ] ≥ 0 . Again, we show this by contradiction for 𝑗 = 2 and it follows for any 𝑗 ≠ 1 by Lemma 19. Conditioned on 𝐴 2 , we have 𝛿 1 &gt; 7 8 . Then, if 𝑈 𝑗 𝑇 [ 1 ] &lt; 0 , 𝜙 ( 𝑈 2 𝑇 · 𝑥 1 ) &gt; 𝜌 ( 0 ) , and

<!-- formula-not-decoded -->

which is another contradiction to the fact tat 𝜌 ( 0 ) ≥ 1 . In addition, we notice that 𝑥 3 is the only possible example whose third entry is non zero. Given the event 𝐴 1 , we know that 𝑥 3 is not in 𝑆 . Equivalently, for every ( 𝑥, 𝑦 ) ∈ 𝑆 , 𝑥 [ 3 ] = 0 . As a result, since 𝑊 𝑗 1 [ 3 ] = 0 for every 𝑗 , it can be proved by induction that for every 𝑡 ≥ 1 , it holds for 𝑗 ≠ 1 that

<!-- formula-not-decoded -->

For 𝑗 = 1 , it holds that,

<!-- formula-not-decoded -->

Then, we get that for every 𝑗 ≠ 1 , it holds that,

<!-- formula-not-decoded -->

Then, since we showed that 𝑈 𝑗 𝑇 · 𝑥 2 ≥ 0 for every 𝑗 , ℓ ( 𝑊 𝑇 · 𝑥 2 ) = ˝ 𝑗 ≠ 1 𝜌 ( 𝑈 𝑗 𝑇 · 𝑥 2 ) , and conditioned on 𝐴 2 , we have

<!-- formula-not-decoded -->

which implies for every 𝑗 ≠ 1 that,

<!-- formula-not-decoded -->

Therefore, by combining Eq. (10) with the fact that 𝑈 𝑗 𝑇 [ 1 ] ≥ 0 ,

<!-- formula-not-decoded -->

This implies for every 𝑗 ≠ 1 , 𝑈 𝑗 𝑇 [ 2 ] ≥ 1 3 𝛾 𝜌 -1 ( 64 𝑘 b 𝐿 ( 𝑊 𝑇 )) . By Eq. (9),

<!-- formula-not-decoded -->

We conclude see that for every 𝜖 such that 𝜖 ≥ ( 𝜌 -1 ( 𝜖 𝑘 ) ) 2 𝛾 2 𝑇𝜂 , b 𝐿 ( 𝑤 𝑇 )) ≤ 4 𝜖 , and

<!-- formula-not-decoded -->

where in the final inequality we again used Eq. (8). Then the lemma follows using Eq. (7) and the law of total expectation,

<!-- formula-not-decoded -->

Lemma 20. Let 𝜌 be a tail function. and 𝜙 : ℝ → ℝ be the following function

<!-- formula-not-decoded -->

Next, we define the following loss function for every 𝑦 ∈ [ 𝑘 ] ,

<!-- formula-not-decoded -->

Then, ℓ ∈ C 𝛽,𝑝 𝜌 .

Proof. For ˜ ℓ ( ˆ 𝑦 ) = ˝ 𝑘 -1 𝑗 = 1 𝜙 ( ˆ 𝑦 𝑗 ) , ℓ 𝑦 ( ˆ 𝑦 ) = ˜ ℓ ( 𝐷 𝑦 ˆ 𝑦 ) . Then, it is left to prove that ˜ ℓ ∈ ˜ C 𝛽,𝑝 𝜌 . By Lemma 6, it is sufficient to prove that 𝜙 is nonnegative, convex, 𝛽 -smooth and monotonically decreasing loss functions such that 𝜙 ( 𝑢 ) ≤ 𝜌 ( 𝑢 ) for all 𝑢 ≥ 0 .

First, 𝜙 is non negative: for 𝑥 ≥ 0 by the non negativity of 𝜌 and for 𝑥 &lt; 0 by the fact that 𝜌 ′ ( 0 ) ≤ 0 . Moreover, 𝜙 is convex. We need to prove that every 𝑥 &lt; 𝑦 , 𝜙 ′ ( 𝑥 ) ≤ 𝜙 ′ ( 𝑦 ) For 𝑥, 𝑦 &lt; 0 , we get it by the convexity of 𝜌 . For 𝑥, 𝑦 &gt; 0 , we get it by the linearity of 𝜙 . For 𝑥 &lt; 0 &lt; 𝑦 , by the convexity of 𝜌 ,

<!-- formula-not-decoded -->

In addition, 𝜙 is 𝛽 -smooth. We need to prove that every 𝑥 &lt; 𝑦 , 𝜙 ′ ( 𝑦 ) -𝜙 ′ ( 𝑥 ) ≤ 𝛽 ( 𝑦 -𝑥 ) For 𝑥, 𝑦 ≥ 0 , we get it by the smoothness of 𝜌 . For 𝑥, 𝑦 ≤ 0 , we get it by the linearity of 𝜙 . For 𝑥 ≤ 0 ≤ 𝑦 , by the smoothness of 𝜌 ,

<!-- formula-not-decoded -->

Finally, 𝜙 is strictly monotonically decreasing. We need to prove that every 𝑥 &lt; 𝑦 , 𝜙 ( 𝑦 ) &gt; 𝜙 ( 𝑥 ) . For 𝑥, 𝑦 &gt; 0 , we get it by the monotonicity of 𝜌 . For 𝑥 &lt; 𝑦 &lt; 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For 𝑥 &lt; 0 &lt; 𝑦 ,

□

Proof of Lemma 5. Let 𝛾 ≤ 1 8 and 𝜖 ≤ 1 16 . We consider the following distribution;

<!-- formula-not-decoded -->

where 𝑝 = 𝜌 -1 ( 16 𝜖 𝑘 ) 72 𝛾 2 𝑇𝑘𝜂 . Note that by the condition of the theorem, 𝑝 ≤ 𝜖 ≤ 1 16 . Since 𝑦 = 1 with probability 1 for the matrix 𝑊 ∗ where its first row is 𝑊 1 ∗ = ( 𝛾, 1 2 , 1 4 ) and for any other 𝑗 th row 𝑊 𝑗 ∗ = 0 , it holds for any 𝑗 ≠ 1 that ⟨ 𝑊 1 ∗ -𝑊 𝑗 ∗ , 𝑥 𝑖 ⟩ = ⟨ 𝑊 ∗ 1 , 𝑥 𝑖 ⟩ ≥ 𝛾 for every 𝑖 ∈ { 1 , 2 } . In addition, we consider the following univariate function,

<!-- formula-not-decoded -->

and the loss function such that for every 𝑦 ∈ [ 𝑘 ] ,

<!-- formula-not-decoded -->

First, by Lemma 20 we get that ℓ ∈ C 𝛽,𝑝 𝜌 . Next, let 𝑆 be a sample of 𝑛 i.i.d. examples from D . We denote by 𝛿 2 ∈ [ 0 , 1 ] the fraction of appearances of ( 𝑥 2 , 1 ) in the sample 𝑆 , and by 𝐴 1 the event that 𝛿 2 ≤ 2 𝑝 . By Markov's inequality, we know that Pr ( 𝐴 1 ) ≥ 1 2 . Moreover, by Lemma 3 and the choice of 𝜖 ,

<!-- formula-not-decoded -->

By Lemma 19 we notice that all of the rows of 𝑊 𝑇 except the first row are equal. Then, defining 𝑈 𝑗 𝑇 = 𝑊 1 𝑇 -𝑊 𝐽 𝑇 , we get that for every 𝑗 ≠ 1 it holds that 𝑈 𝑗 𝑇 = 𝑈 2 𝑇 Now, we turn to assume that 𝐴 1 holds. We know that

<!-- formula-not-decoded -->

thus, conditioned on 𝐴 1 and by Eq. (11),

<!-- formula-not-decoded -->

Then, if 𝑈 2 𝑇 [ 1 ] &lt; 0 , we get that

<!-- formula-not-decoded -->

which is a contradiction to our assumption that 𝜖 ≤ 1 16 . Then 𝑈 2 𝑇 ( 1 ) ≥ 0 and by Eq. (13), we get that 16 𝜖 𝑘 ≥ 𝜙 ( 𝑈 2 𝑇 [ 1 ]) = 𝜌 ( 𝑈 2 𝑇 [ 1 ]) . This implies that

<!-- formula-not-decoded -->

Now, by the fact that 𝜌 ′ ( 0 ) ≤ 1 and 𝜌 is 1 -Lipschitz, it follows that 𝜙 is 1 -Lipschitz. Thus, by the GD update rule, it holds for every 𝑗 ≠ 1 , that,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We get that for any 𝑗 ≠ 1 , and for 𝑗 = 1

<!-- formula-not-decoded -->

As a result, by Eqs. (12), (14) and (15) we now obtain that

<!-- formula-not-decoded -->

By the fact that ∀ 𝑥 &lt; 0 : 𝜙 ( 𝑥 ) ≥ -𝑥 , this implies that in the event 𝐴 1 it holds that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, for a new validation example ( 𝑥 ′ , 𝑦 ′ ) ∼ D (independent from the sample 𝑆 ), 𝑦 ′ = 1 , and

<!-- formula-not-decoded -->

To conclude, from Eqs. (16) and (17) we have

<!-- formula-not-decoded -->

and,

□

Proof of Theorem 3. By Lemma 4, there exists a constant 𝐶 1 such that 𝔼 𝐿 ( 𝑊 𝑇 ) ≥ 𝐶 1 𝛽𝑘𝜌 -1 ( 256 𝜖 𝑘 ) 2 𝛾 2 𝑛 . By Lemma 5, there exists a constant 𝐶 2 such that 𝔼 𝐿 ( 𝑊 𝑇 ) ≥ 𝐶 2 𝜌 -1 ( 16 𝜖 𝑘 ) 2 𝜂𝛾 2 𝑇 . If ( 𝜌 -1 ( 16 𝜖 𝑘 ) 2 𝛾 2 𝑇𝜂 ≥ 𝛽𝑘 ( 𝜌 -1 256 𝜖 𝑘 ) 2 𝛾 2 𝑛 , the theorem follows from Lemma 5 with 𝜂 = 1 6 𝛽𝑘 ; otherwise, it follows from Lemma 4. □

## C Proofs for Section 5

Lemma 21. Let 𝛼 &gt; 0 . If for every 𝑦 , ℓ 𝑦 ( ˆ 𝑦 ) = 1 𝛼 log GLYPH&lt;16&gt; 1 + ˝ 𝑗 ≠ 𝑦 exp ( 𝛼 ( ˆ 𝑦 𝑦 -ˆ 𝑦 𝑗 )) GLYPH&lt;17&gt; . Then, ℓ ∈ C 𝛽,𝑝 𝜌 for 𝜌 ( 𝑥 ) = 1 𝛼 𝑒 -𝛼𝑥 , 𝛽 = 𝛼 2 and 𝑝 = ∞ .

Proof of Lemma 21. Here we notate the 𝑗 th entry of every vector 𝑤 by 𝑤 𝑗 .

First, for ˜ ℓ ( ˆ 𝑦 ) = 1 𝛼 log GLYPH&lt;16&gt; 1 + ˝ 𝑘 -1 𝑗 = 1 exp ( 𝛼 ˆ 𝑦 𝑗 ) GLYPH&lt;17&gt; , ℓ 𝑦 ( ˆ 𝑦 ) = ˜ ℓ ( 𝐷 𝑦 ˆ 𝑦 ) . Now, 𝑥 ≥ log ( 1 + 𝑥 ) ≥ 0 for every 𝑥 , it follows ˜ ℓ non-negative and,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the convexity of ˜ ℓ , let 𝑢, 𝑣 ∈ ℝ 𝑘 -1 and 𝜆 ∈ ( 0 , 1 ) . If ˜ 𝑢, ˜ 𝑣 are the vectors on ℝ 𝑘 whose the 𝑘 -1 first entries are 𝑢, 𝑣 , respectively and last entry is 0. It holds that,

<!-- formula-not-decoded -->

«

‹

<!-- formula-not-decoded -->

as required. For the smoothness, for every 𝑢 ∈ ℝ 𝑘 -1 the partial derivatives of ˜ ℓ are

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, if we denote by 𝑤 the vector that its 𝑗 th entry is 𝑤 𝑗 = 𝛼𝑒 𝛼𝑢 𝑗 1 + ˝ 𝑘 -1 𝑗 = 1 𝑒 𝛼𝑢 𝑗 , it holds that ∇ 2 ˜ ℓ ( 𝑤 ) = 𝑑𝑖𝑎𝑔 ( 𝑤 ) -𝑤𝑤 𝑇 . Now, let 𝑣 ∈ ℝ 𝑘 -1 . For 𝐿 ∞ smoothness it is sufficient to prove that 𝑣 𝑇 ∇ 2 ˜ ℓ ( 𝑢 ) 𝑣 ≤ 𝛼 2 ∥ 𝑣 ∥ 2 ∞ .

<!-- formula-not-decoded -->

□

Lemma 22. If ℓ is the cross entropy loss function, ℓ ∈ C 𝛽,𝑝 𝜌 for 𝜌 ( 𝑥 ) = 𝑒 -𝑥 , 𝛽 = 1 and 𝑝 = ∞ . Proof of Lemma 22. The proof is implied directly from Lemma 21 with 𝛼 = 1 . □

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The claims presented in the abstract and introduction accurately reflect the paper's contributions-specifically, the risk upper bounds (including the novel Rademacher complexity bound that leverages the template's properties), the lower bound for the Euclidean case, and the applications to widely-used loss functions.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Our work is theoretical and focuses on the setting where the loss function is convex and smooth, and the data is separable. The assumptions underlying our analysis are clearly outlined in Section 2.

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

Justification: The assumptions for all our theorems are detailed in Section 2 and explicitly stated within each theorem. The proofs provided are both correct and complete.

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

Justification: The paper does not include experiments.

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

Justification: The paper does not include experiments requiring code.

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

Justification: The paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The paper does not include experiments.

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

Justification: The paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The paper is theoretical and conforms with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The paper is theoretical and has no societal impact.

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

Justification: The paper is theoretical and poses no such risks.

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
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
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

Justification: The paper is theoretical and does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper is theoretical and does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: This research does not involve LLMs as any important, original, or nonstandard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.