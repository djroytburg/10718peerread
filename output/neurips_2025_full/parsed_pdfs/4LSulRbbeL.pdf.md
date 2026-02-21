## Dynamic Regret Reduces to Kernelized Static Regret

## Andrew Jacobsen ∗

Universit` a degli Studi di Milano Politecnico di Milano

## Alessandro Rudi ∗

Bocconi University alessandro.rudi@sdabocconi.it contact@andrew-jacobsen.com

## Francesco Orabona

King Abdullah University of Science and Technology (KAUST) Thuwal, 23955-6900, Kingdom of Saudi Arabia francesco@orabona.com

## Nicol` o Cesa-Bianchi

Universit` a degli Studi di Milano Politecnico di Milano nicolo.cesa-bianchi@unimi.it

## Abstract

We study dynamic regret in online convex optimization, where the objective is to achieve low cumulative loss relative to an arbitrary benchmark sequence. By observing that competing with an arbitrary sequence of comparators u 1 , . . . , u T in W ⊆ R d can be reframed as competing with a fixed comparator function u : [1 , T ] →W , we cast dynamic regret minimization as a static regret problem in a function space . By carefully constructing a suitable function space in the form of a Reproducing Kernel Hilbert Space (RKHS), our reduction enables us to recover the optimal R T ( u 1 , . . . , u T ) = O ( √∑ t ∥ u t -u t -1 ∥ T ) dynamic regret guarantee in the setting of linear losses, and yields new scale-free and directionallyadaptive dynamic regret guarantees. Moreover, unlike prior dynamic-to-static reductions-which are valid only for linear losses-our reduction holds for any sequence of losses, allowing us to recover O ( ∥ u ∥ 2 H + d eff ( λ ) ln T ) bounds when the losses have meaningful curvature, where d eff ( λ ) is a measure of complexity of the RKHS. Despite working in an infinite-dimensional space, the resulting reduction leads to algorithms that are computable in practice, due to the reproducing property of RKHSs.

## 1 Introduction

This paper introduces new techniques for Online Convex Optimization (OCO), a framework for designing and analyzing algorithms which learn on-the-fly from a stream of data [10, 11, 18, 39, 59]. Formally, consider T rounds of interaction between a learner and the environment. In each round, the learner chooses w t ∈ W from a convex set W ⊆ R d , the environment reveals a convex loss function ℓ t : W → R , and the learner incurs a loss of ℓ t ( w t ) . The classic objective in this setting is to minimize the learner's regret relative to any fixed benchmark u ∈ W :

<!-- formula-not-decoded -->

∗ Equal contribution.

Figure 1: Transformation from a sequence of comparators to a function. Many functions may implement the transformation. In Section 4 we will see that under mild assumptions on the chosen function space H we can always find a function u ∈ H such that u ( t ) = u t for all t and ∥ u ∥ 2 H d = O ( ∑ T t =2 ∥ u t -u t -1 ∥ ) .

<!-- image -->

In this paper, we study the more general problem of minimizing the learner's regret relative to any sequence of benchmarks u 1 , . . . , u T ∈ W [21, 22, 59]:

<!-- formula-not-decoded -->

This objective is typically referred to as dynamic regret, to distinguish it from the special case where the comparator sequence is fixed u 1 = · · · = u T (referred to as static regret). Intuitively, dynamic regret captures a notion of non-stationarity in learning problems. Problem instances where u 1 = · · · = u T model classic problem settings, wherein there is a fixed 'solution' whose performance we want to emulate, while a time-varying comparator sequence models problem settings where the learner needs to continuously adapt to a changing environment in which the solution is time-varying. The complexity of a given comparator sequence is typically characterized by its path-length :

<!-- formula-not-decoded -->

Clearly, if the path-length is large there is no hope to obtain low dynamic regret. The goal is thus to obtain performance guarantees that gracefully adapt to the level of non-stationarity. For instance, in the setting of G -Lipschitz losses and a bounded domain D = sup x,y ∈W ∥ x -y ∥ , the minimax optimal dynamic regret guarantee is of the order of O ( G √ ( D 2 + DP T ) T ) , which scales naturally with the complexity of the benchmark sequence and recovers the optimal O ( GD √ T ) static regret guarantee when the comparator is fixed. In unbounded domains (e.g., W = R d ) these bounds would be vacuous, so the guarantee should instead be adaptive to M := max t ∥ u t ∥ . In this case an analogous guarantee of ˜ O ( G √ ( M 2 + MP T ) T ) can be achieved at the expense of additional logarithmic terms. Throughout the paper we focus on the unbounded setting.

Contributions. In this work we introduce a new framework for reducing dynamic regret minimization to static regret minimization. Our key insight is that competing with a sequence u 1 , . . . , u T in W can be equivalently framed as competing with some fixed function u ( · ) such that u ( t ) = u t for all t . In this view, we effectively transform dynamic regret minimization over a domain W ⊆ R d into a static regret minimization problem over a domain of functions , depicted graphically in Figure 1.

The choice of the function space is crucial, as it controls the trade-offs of the resulting algorithm. To complete the construction, we carefully design a rich family of function spaces which embed the comparator sequence in a way that (1) optimizes the inherent trade-offs of the function class to achieve optimal dynamic regret guarantees and (2) ensures that the resulting algorithm is computable in practice, despite being stated as an infinite-dimensional optimization problem. Indeed, the family we design is an instance of a Reproducing Kernel Hilbert Space (RKHS), a well-studied class of functions endowed with the familiar structure of a Hilbert space. The reduction to learning in an RKHS is particularly natural in the context of online learning-the vast majority of modern online learning theory is developed for static regret minimization in Hilbert spaces , so our reduction enables the use of the familiar online learning toolkit while also allowing us to draw upon deep connections between dynamic regret minimization, kernel methods, and signal processing theory.

In the linear losses setting, our construction enables us to achieve the optimal dynamic regret guarantees of O ( √ MP T T ) up to poly-logarithmic terms. Notably, the resulting algorithm is naturally horizon independent , and is easily extended to a scale-free version. These are the first algorithms

that obtain the optimal √ P T dependence without prior knowledge of the horizon T natively, without resorting to the doubling trick. Our reduction also enables us to derive new directionally-adaptive guarantees, which scale as ˜ O (√ d eff ( λ ) ( ∥ u ∥ 2 H d + ∑ T t =1 ⟨ g t , u t ⟩ 2 ) ) , where ∥ u ∥ 2 H d and d eff ( λ ) are measures of the complexity of the comparator function and complexity of the function class H respectively.

Interestingly, because our reduction only involves viewing the comparator sequence through a different lens, it holds for any sequence of loss functions, contrasting prior works which are valid only for linear losses [26, 55]. We show that this allows us to account for loss curvature and obtain O ( λ ∥ u ∥ 2 H + d eff ( λ ) log T ) dynamic regret in the context of strongly-convex, exp-concave, and improper linear regression settings.

Related Works. Our work is most directly related to a recent thread of research in the linear loss setting initiated by Zhang et al. [55]. Their strategy approaches dynamic regret from a signal processing perspective, wherein the comparator sequence is stacked into a high-dimensional 'signal' ˜ u = vec ( u 1 , . . . , u T ) ∈ R dT , and 1 -dimensional static regret algorithms are employed to learn the coefficients of a basis of features which decompose that signal, leading to O ( √ MP T T ) dynamic regret via a carefully chosen dictionary of features. Jacobsen and Orabona [26] generalize this perspective by designing static regret algorithms that are applied directly in this high-dimensional space, and derive the O ( √ MP T T ) bound by choosing a suitable dual-norm pair in this space, such that ∥ ˜ u ∥ = O ( √ P T ) . Our work further extends this perspective by interpreting the comparator sequence as samples of a function in an RKHS H , and designing algorithms which obtain suitable static regret guarantees in function space. The reduction of Jacobsen and Orabona [26] can in fact be understood as a special case of our framework by choosing the discrete RKHS H associated with the Dirac kernel.

More broadly, the concept of dynamic regret was originally introduced by Herbster and Warmuth [21, 22]. Later, Zinkevich [59] showed that OGD naturally obtains O ( P T √ T ) dynamic regret and Yang et al. [52] showed that O ( √ DP T T ) can be achieved when prior knowledge of P T is available. The first to achieve the O ( √ DP T T ) rate without prior knowledge of P T was Zhang et al. [53], who also proved a matching lower bound, and the analogous bound of O ( √ MP T T ) has been achieved up to logarithmic terms in unbounded settings [23, 24, 33, 55]. There have also been several refinements to the result, replacing the T factor with data-dependent quantities such as ∑ T t =1 ∥ g t ∥ 2 or ∑ t sup x | ℓ t ( x ) -ℓ t -1 ( x ) | [9, 14, 19]. Going beyond linear losses, various improvements in adaptivity can be obtained when the losses are smooth or exp-concave, such as replacing the T factor with ∑ t ℓ t ( u t ) or ∑ t sup w ∥∇ ℓ t ( w ) -∇ ℓ t -1 ( w ) ∥ 2 [56-58]. In the squared loss setting ℓ t ( w ) = 1 2 ( y t -w ) 2 , minimax optimal rates of R T (˚ y 1 , . . . , ˚ y T ) = O ( C 2 / 3 T T 1 / 3 ) have been obtained where C T = ∑ T t =2 | ˚ y t -˚ y t -1 | is the path-length of the benchmark predictions [3-5, 29, 44, 54].

## 2 Preliminaries

Notations. Hilbert spaces are denoted by upper case calligraphic letters. Given a Hilbert space H , we denote the associated inner product by ⟨· , ·⟩ H . We denote L ( H , W ) the space of linear operators from H to W . L ( H , W ) is itself a Hilbert space when equipped with the Hilbert-Schmidt inner product, ⟨ A,B ⟩ HS = Tr( A ∗ B ) , where A ∗ ∈ L ( W , H ) is the adjoint of A . The subdifferential set of a function f at x is denoted by ∂f ( x ) . We will occasionally abuse notation and write ∇ f ( x ) to mean an arbitrary element of ∂f ( x ) . We will denote by [ T ] the set { 1 , 2 , . . . , T } . The Fourier transform of a function Q is denoted F [ Q ]( x ) = ∫ R Q ( ω ) e -2 πixω dω and, when clear from context, we will generally abbreviate F [ Q ]( x ) =: ̂ Q ( x ) .

Reproducing Kernel Hilbert Spaces. Let H = { h : T → R } be a Hilbert space of functions a on compact set T . The space H is a RKHS [1] if there exists a positive definite function k : T × T → R such that k ( · , x ) ∈ H for all x ∈ T , that has the reproducing property , i.e., we have f ( x ) = ⟨ f, k ( · , x ) ⟩ H for all f ∈ H and x ∈ T . The function k is called the kernel function associated with H and the function ϕ ( x ) = k ( · , x ) is called the feature map . It is known that the kernel function uniquely characterizes the RKHS H . Akernel is universal if it can approximate any

```
Algorithm 1: Kernelized Online Learning Input: Domain W ⊆ R d , feature map ϕ : T → H , algorithm A defined on L ( H , W ) for t = 1 : T do Receive W t ∈ L ( H , W ) from A and play w t = W t ϕ ( t ) ∈ W Observe loss function ℓ t : W → R and incur loss ℓ t ( w t ) Send ˜ ℓ t : W ↦→ ℓ t ( Wϕ ( t )) to A as the t th auxiliary loss end
```

real-valued continuous function on T to arbitrary accuracy. Many of the standard kernel functions are universal, including the Gaussian RBF kernel, the Mat´ ern kernel, and inverse multiquadratic kernel. All kernels considered in this work are universal kernels. For a detailed introduction to kernel methods, see, e.g., Berlinet and Thomas-Agnan [6], Paulsen and Raghupathi [42], Sch¨ olkopf and Smola [45], Wendland [51].

We will often be interested in functions taking values in W ⊆ R d . In this case the usual RKHS machinery extends in a straight-forward way via a coordinate-wise extension. Indeed, we can represent w : T → R d as a tuple w = ( w 1 , . . . , w d ) such that w i ∈ H for each i . 2 This naturally leads to an operator-based version of the reproducing property:

<!-- formula-not-decoded -->

where W ∈ L ( H , W ) . The space L ( H , W ) is itself a Hilbert space when equipped with the HilbertSchmidt norm, and under the coordinate-wise extension above we have ∥ W ∥ 2 HS = ∥ w ∥ 2 H d = ∑ d i =1 ∥ w i ∥ 2 H . Moreover, observe that when d = 1 this setup simply reduces back to the usual setup, wherein ∥ W ∥ HS = ∥ w ∥ H and w ( t ) = ⟨ w,ϕ ( t ) ⟩ H .

For notational clarity we will refer to functions w ( · ) ∈ H and their values w ( t ) ∈ W with lowercase letters, and their representation W ∈ L ( H , W ) using the upper-case. We will typically use the notation w t in place of w ( t ) when referring to the evaluations of w ( t ) at discrete time-points t ∈ [ T ] .

## 3 An Equivalence Between Static and Dynamic Regret

In this section, we present the main tool that we will use to develop dynamic regret guarantees for online learning. The key idea is to interpret the comparator sequence u 1 , . . . , u T ∈ W as the evaluations of a function u ( · ) at the discrete time-points t ∈ [ T ] , allowing us to re-frame dynamic regret minimization as a static regret minimization in function space .

Note that most existing work in online learning revolves around learning in Hilbert spaces, not general function spaces, so if we hope to leverage these existing tools we should embed the comparator sequence in a Hilbert space of functions . In particular, our approach will be to embed the comparator sequence in a Hilbert space H of functions representable by a reproducing kernel k ( s, t ) and feature map ϕ : T → H . Note that this is always possible by selecting a universal kernel on T . Our reduction is conceptually shown in Algorithm 1 and the following theorem shows that the dynamic regret w.r.t. comparator sequence u 1 , . . . , u T in W is equivalent to static regret w.r.t. a function u ( · ) ∈ H on the auxiliary loss sequence ˜ ℓ t : W ↦→ ℓ t ( Wϕ ( t )) .

Theorem 1 (Dynamic Regret via Kernelized Static Regret) . Let T be a compact set, W ⊆ R d , let H be an RKHS with associated feature map ϕ : T → H , and for any W ∈ L ( H , W ) let ˜ ℓ t ( W ) = ℓ t ( Wϕ ( t )) . Let W 1 , . . . , W T be an arbitrary sequence in L ( H , W ) and suppose that on each round we play w t = W t ϕ ( t ) ∈ W . Then, for any comparator sequence u 1 , . . . , u T in W and U ∈ L ( H , W ) satisfying u t = Uϕ ( t ) for all t ,

<!-- formula-not-decoded -->

Note that the reduction holds for any operator U ∈ L ( H , W ) which interpolates the comparator sequence, u t = Uϕ ( t ) ∀ t . Hence, we can always let u ( · ) ∈ H be the minimum norm function in H which interpolates these points, and take U to be its representation in L ( H , W ) . In fact, since k is a

2 This connection can be made more formally via Riesz representation theorem, see Appendix A for details.

universal kernel, we can assume that u ( · ) ∈ H approximates any continuous function on T ⊆ [1 , T ] to arbitrary accuracy, so the assumption that u ( · ) lives in an RKHS H does not actually restrict the functions that we can compare against in a significant way. We will see a concrete example of an RKHS H which reconstructs arbitrary comparator sequences in later sections (e.g., Theorem 4).

Our reduction bears a strong resemblance to the reduction recently proposed by Jacobsen and Orabona [26], which works by embedding the comparator sequence in R dT by simply 'stacking' the comparator sequence into one long vector, ˜ u = vec( u 1 , . . . , u T ) ∈ R dT . In fact, we show in Appendix B that our framework precisely recovers the reduction in Jacobsen and Orabona [26] by choosing the discrete RKHS H associated with the Dirac kernel. However, notice in particular that this means that their reduction is inherently tied to finite -dimensional features, whereas ours enables infinite -dimensional features. As we will see in Section 4, this distinction is key to obtaining the optimal path-length dependencies in a horizon-independent manner. Moreover, note that the regret equality in Jacobsen and Orabona [26] holds only in the context of linear losses, ℓ t ( w ) ↦→ ⟨ g t , w ⟩ , whereas in our framework the regret equality holds for any sequence of losses ℓ 1 , . . . , ℓ T . We will see that this distinction is important in Section 5-for example, our reduction allows us to preserve the curvature of the losses needed to obtain O ( ∥ u ∥ 2 H + d eff ( λ ) ln T ) bounds when the original losses ℓ t are strongly convex or exp-concave, where d eff ( λ ) is a measure of complexity of the RKHS.

## 4 Linear Losses

We first consider the setting of online linear optimization. In this setting, on round t the learner receives linear loss ℓ t ( w ) = ⟨ g t , w ⟩ W , so recalling the reduction in the previous section defines the auxiliary loss as ˜ ℓ t : W ∈ L ( H , W ) ↦→ ℓ t ( Wϕ ( t )) ∈ R , we have

<!-- formula-not-decoded -->

where G t = g t ⊗ ϕ ( t ) ∈ L ( H , W ) is the rank one operator such that ( g t ⊗ ϕ ( t ))( h ) = ⟨ ϕ ( t ) , h ⟩ H g t for any h ∈ H . As such, it is important that the base algorithm facilitates an application of the kernel trick to avoid explicitly evaluating the feature map ϕ ( t ) , which may be infinite-dimensional in general. To help make things concrete and provide intuitions, the following example shows that many of the common algorithms based on Follow the Regularized Leader (FTRL) with a radiallysymmetric regularizer w ↦→ Ψ t ( ∥ w ∥ ) are amenable to the kernel trick. This class captures many of the fundamental regularizers in online learning, such as quadratic regularizers and the 'linearithmic' [41] regularizers Ψ t ( ∥ w ∥ ) ≈ ∥ w ∥ √ t log( ∥ w ∥ /α +1) associated with the comparator-adaptive regret guarantees that the key result of this section (Proposition 1) will be derived from.

Example 1. (Kernel Trick for Kernelized FTRL) Let g 1 , . . . , g T be a sequence in W and let G t = g t ⊗ ϕ ( t ) ∈ L ( H , W ) for all t . Let θ t = -∑ t -1 s =1 G s , V t = ∑ t -1 s =1 ∥ G t ∥ 2 HS , let Ψ t ( · ; V t ) be a convex function with differentiable Fenchel conjugate Ψ ∗ t , and consider the following FTRL update:

<!-- formula-not-decoded -->

Then, V t = ∑ t -1 s =1 ∥ g s ∥ 2 W k ( t, t ) (Lemma 10), ∥ θ t ∥ 2 HS = ∑ t -1 s,s ′ =1 k ( s, s ′ ) ⟨ g s , g s ′ ⟩ W (Lemma 9), and on round t , Algorithm 1 plays

<!-- formula-not-decoded -->

The example shows that many common instances of FTRL can be kernelized without explicit computation of the feature map. The example also demonstrates an important consideration when applying static regret decompositions of this nature: the update described above would require O ( t ) time and memory to implement in general, while existing algorithms for dynamic regret can often be implemented using O (ln T ) computation and memory [23, 53, 55, 58]. Luckily, there is already a deep and well-developed literature on efficient approximations for kernel methods that can be leveraged to translate the algorithms developed from the kernelized OCO point-of-view into more practically implementable algorithms [see, e.g., 31, 47]. Since these extensions are already well-understood and since implementating these details would not yield any new insights in the current paper, we will not consider them further here, focusing instead on the theoretical development.

Nowthat we have seen how to translate an algorithm's updates to the kernelized setting, we turn now to how to translate its static regret guarantees into dynamic regret guarantees. The following result shows that an algorithm's kernelized static regret guarantee translates in a straight-forward way to a dynamic regret guarantee in the original problem. The proof is immediate by applying Theorem 1 and computing ∥ G t ∥ HS = ∥ g t ⊗ ϕ ( t ) ∥ HS = ∥ g t ∥ W √ k ( t, t ) by Lemma 10.

Theorem 2. Let A be an online learning algorithm defined on Hilbert space V . Suppose that for any sequence of convex loss functions h 1 , . . . , h T on V , A obtains a bound on the static regret of the form ˜ R T ( U ) ≤ B T ( ∥ U ∥ V , ∥∇ h 1 ( W 1 ) ∥ V , . . . , ∥∇ h T ( W T ) ∥ V ) for any comparator U ∈ V and some function B T : R T +1 ≥ 0 → R , where ∇ h t ( W t ) ∈ ∂h t ( W t ) for all t . If we apply A in V = L ( H , W ) with ∥·∥ V = ∥·∥ HS , then for any sequence u 1 , . . . , u T in W and U ∈ L ( H , W ) satisfying u t = Uϕ ( t ) for all t , Algorithm 1 with A guarantees

<!-- formula-not-decoded -->

where g t ∈ ∂ℓ t ( w t ) for all t , and k ( · , · ) is the reproducing kernel associated to the space H .

The value of the lemma is that it enables us to immediately translate static regret guarantees from OLO to guarantees in our RKHS formulation of dynamic regret, wherein the complexity of the comparator sequence is measured by the RKHS norm ∥ U ∥ HS = ∥ u ∥ H d . For instance, if we simply apply the standard (sub)gradient descent guarantee to Theorem 2 we get

<!-- formula-not-decoded -->

Optimally tuning η yields R T ( u 1 , . . . , u T ) ≤ ∥ u ∥ H d √ ∑ T t =1 ∥ g t ∥ 2 W k ( t, t ) , so achieving the optimal O ( √ P T T ) has effectively been reduced to the problem of designing a kernel such that 3 ∥ u ∥ H d = √ ∑ d i =1 ∥ u i ∥ 2 H = ˜ O ( √ P T ) while controlling k ( t, t ) . We will see in Section 4.1 that this can be accomplished by using a carefully chosen translation-invariant kernel.

In the above argument, the optimal choice of η would require prior knowledge of ∥ u ∥ H d and cannot be chosen in general. Luckily, there are static regret algorithms which can adapt to the comparator norm automatically to obtain the optimal trade-off up to logarithmic terms [16, 23, 35, 36, 40]. For our purposes we will refer to an algorithm A defined on Hilbert space V as parameter-free if for any sequence G -Lipschitz loss functions h 1 , . . . , h T and any U ∈ V , A guarantees

<!-- formula-not-decoded -->

There are many existing algorithms which satisfy this property; we provide a concrete example and its updates in our framework for completeness in Appendix C.2. Using such an algorithm in Algorithm 1 immediately yields the following regret guarantee.

Proposition 1. Let A be a static regret algorithm for Hilbert spaces satisfying Equation (1) . For any G &gt; 0 , any sequence of G -Lipschitz losses ℓ 1 , . . . , ℓ T , and any sequence u 1 , . . . , u T in W , and U ∈ L ( H , W ) satisfying u t = Uϕ ( t ) , Algorithm 1 applied with A guarantees

<!-- formula-not-decoded -->

where g t ∈ ∂ℓ t ( w t ) for all t .

## 4.1 Controlling the Trade-offs Induced by H

Proposition 1 demonstrates a clear trade-off between the RKHS norm ∥ u ∥ H and the associated kernel k ( t, t ) induced by the choice of function space: smaller the RKHS norms correspond to larger function spaces, hence higher values of k ( t, t ) . In order to obtain the optimal O ( √ P T T ) dynamic regret, we need to design a kernel such that ∥ U ∥ HS = ∥ u ∥ H d = O ( √ P T ) and k ( t, t ) is

3 Here and in the following the ˜ O notation will hide polylogarithmic factors.

controlled for all t . Throughout this section, we will assume for simplicity that d = 1 but note that the extension to d &gt; 1 is straightforward via the coordinate-wise extension in Section 2.

Recall that a translation invariant kernel over R is characterized by the Fourier transform of its spectral density Q [51], where Q is a real non-negative integrable function. In particular, a translation invariant kernel and its associated norm are

<!-- formula-not-decoded -->

where we use the short-hand notation ̂ g = F [ g ] to denote the Fourier transorm of a function g ( · ) . The intuition behind focusing on translation-invariant kernels is that the associated norm provides a natural connection to the √ MP T dependencies we would like like to achieve. Indeed, observe that with spectral density Q ( ω ) ≈ 1 /ω , we would have via Parseval's identity and the fact that F [ f ′ ( x )]( ω ) = 2 πiω f ( ω ) that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which is the continuous-time analogue of MP T . The key challenge is to choose an integrable Q ( ω ) which suitably trades off the comparator norm ∥ u ∥ 2 H and the magnitude of the associated kernel entries k ( t, t ) . Unfortunately, these trade-offs are non-trivial using standard translation-invariant kernels, as shown in the following example.

Example 2 (Existing kernels lead to sub-optimal trade-offs [51]) . At first glance, the spline kernel seems like a natural candidate since it has ∥ u ∥ 2 H = ∥∇ u ∥ 2 L 2 = O ( ∑ t ∥ u t -u t -1 ∥ 2 ) (Theorem 8). However, the spline kernel also has k ( t, t ) = t , leading to a suboptimal rate in Proposition 1. On the other hand, for the classical translation invariant kernels such as the Gaussian or the Matern kernels, we have k ( t, t ) = O (1) but ∥ u ∥ 2 H = ∥ u ∥ 2 L 2 + ∑ n ≥ 1 c n ∥∇ n u ∥ 2 for c n positive and summable. In this case, note that k ( t, t ) has the good rate but ∥ u ∥ 2 H ≥ ∥ u ∥ 2 L 2 and ∥ u ∥ 2 L 2 = c 2 T already for constant comparators on [0 , T ] , u ( t ) = c 1 [0 ,T ] ( t ) , c &gt; 0 , precluding the optimal rate.

Given the above, we next turn our attention to designing a new kernel that will achieve the desired trade-offs. Since we need to find a delicate balance in the trade-off of ∥ u ∥ H and k ( t, t ) to achieve optimal rates, in the first part of the section we first derive a result that identifies general sufficient conditions to bound the RKHS norm of a translation invariant kernel in terms of the continuous pathlength ∥∇ u ∥ L 1 = ∫ |∇ u ( t ) | dt (Theorem 3). Then, in Proposition 2, we design an explicit kernel satisfying such conditions, leading to a trade-off of O ( √ ∥∇ u ∥ L 1 T ) . Finally, in Theorem 4 we show that under mild conditions (which are satisfied by the kernel in the Proposition 2), it is always possible to find an u ( · ) ∈ H such that u ( t ) = u t for all t and ∥∇ u ∥ L 1 = O ( ∑ t ∥ u t -u t -1 ∥ W ) , so achieving dynamic regret scaling with ∥ u ∥ H = O ( ∥∇ u ∥ L 1 ) recovers the usual path-length. The proof of the following theorem can be found in Appendix E.1

Theorem 3. Let Q : R → R + be an integrable strictly positive even function on R \ { 0 } and such that R ( x ) := 2 π/ ( x (1 + ( x/ 2 π ) 2 m ) Q ( x )) is also integrable for some m ∈ N , m ≥ 1 . Let k be defined in terms of Q as in Eq. (3) . Then k is a translation invariant universal kernel with k ( t, t ) ≤ ∥ Q ∥ L 1 for all t ∈ R . The RKHS H associated to k contains the space of finitely supported functions with bounded derivatives up to order 2 m , and moreover, for any T &gt; 0 and any 2 m -times differentiable function f that is supported on [0 , T +1] ,

<!-- formula-not-decoded -->

where c ( T ) := ∥ F [ R ] ∥ L 1 ([ -T -1 ,T +1]) . If R is monotonically decreasing on (0 , ∞ ) , then,

<!-- formula-not-decoded -->

With this in hand, the following proposition provides an example of spectral density Q which will leads to the desired dependency ∥ u ∥ 2 H = O ( ∥∇ u ∥ L 1 ) , up to poly-logarithmic terms. Proof can be found in Appendix E.4.

Proposition 2. Let Q : R → R + be defined as

<!-- formula-not-decoded -->

Then we can apply Theorem 3 with m = 1 : the function k defined in terms of Q as in Eq. (3) is a translation invariant kernel with k ( t, t ) ≤ 8 π 2 , ∀ t ∈ R ; the associated RKHS norm satisfies

<!-- formula-not-decoded -->

for any f that is 2 -times differentiable and supported in [0 , T +1] , where T &gt; 2 and c ≤ (2 πe ) 2 .

A notable property of the kernel characterized by Proposition 2 is that it is horizon independent , requiring no upper bound on T to control ∥ f ∥ 2 H and k ( t, t ) . This is a non-trivial property to guarantee using existing methods without resorting to the doubling trick, which is well-known to perform poorly in practice. The intuitions behind the choice of Q ( ω ) follow from the discussion above: we would like to set Q ( ω ) ≈ 1 / | ω | so that ∥ u ∥ H relates to the path-length via Equation (4), but this would not be a valid choice because Q ( ω ) = 1 / | ω | is not integrable. Proposition 2 adds a small bit of additional regularization to ensure that Q ( ω ) is integrable while remaining close to 1 / | ω | . We provide additional intuition on the choice of regularization in Appendix D.

A subtlety that we have glossed over thus far is that the continuous path-length , ∥∇ u ∥ L 1 = ∫ ∥∇ u ( t ) ∥ W dt , does not necessarily compare favorably to the classic discrete path-length P T = ∑ t ∥ u t -u t -1 ∥ W since the function may vary wildly between the interpolated points. The next theorem shows that we can always find a function such that ∥∇ u ∥ L 1 = O ( P T ) . Proof can be found in Appendix E.2.

Theorem 4. Let v 1 , . . . , v T ∈ R d and let H be the RKHS associated to kernel k contain finitely supported functions with bounded derivatives up to order 2 m , with m ∈ N , m ≥ 1 . Then there exists a function u ∈ H supported on [0 , T +1] , such that u ( t ) = v t for all t ∈ [ T ] and

<!-- formula-not-decoded -->

with C, C ′ depending only on m and given in explicitly in the proof.

The theorem demonstrates that the continuous path-length can be bound by the usual discrete path-length under mild assumptions on the RKHS that are satisfied by the translation invariant kernel with spectral density Q chosen according to Theorem 3. Based on this observation, we immediately see that the RKHS characterized by the kernel in Proposition 2 satisfies the condition of the theorem, and has RKHS norm satisfying ∥ u ∥ 2 H = ˜ O ( ∥∇ u ∥ L 1 ∥ ∥ u -∇ 2 u ∥ ∥ L ∞ ) = ˜ O ( M 2 + M ∑ T t =2 ∥ u t -u t -1 ∥ W ) where M = max t ∥ u t ∥ W .

Optimal Path-length Dependencies. Applying our reduction Proposition 1 with the translation invariant kernel characterized by Proposition 2, followed by Theorem 4 to bound ∥∇ u ∥ L 1 = O ( √ M 2 + MP T ) immediately yields the following dynamic regret guarantee for OLO.

Theorem 5. Let G &gt; 0 and apply the algorithm characterized in Proposition 1 with the kernel with spectral density described by Proposition 2. Then for any T &gt; 3 , and any sequence g 1 , . . . , g T satisfying ∥ g t ∥ W ≤ G and sequence u 1 , . . . , u T in W ⊆ R d , the dynamic regret is bounded as

<!-- formula-not-decoded -->

where M = max t ∥ u t ∥ W and P T = ∑ T t =2 ∥ u t -u t -1 ∥ W .

As observed in Section 4, the kernel that produces this result is horizon independent, so the algorithm described above requires no prior knowledge of T . This is in fact the first dynamic regret algorithm we are aware of that achieves the optimal √ P T dependence in the absence of prior knowledge of T without resorting to a doubling trick. Likewise, in Appendix C.3 we show that these guarantees extend immediately to scale-free guarantees using the gradient-clipping argument of [12]. These are the first scale-free dynamic regret guarantees that we are aware of that achieve the optimal √ P T dependencies.

## 5 Curved Losses

An advantage of our reduction over the dynamic-to-static reduction of Jacobsen and Orabona [26] is that, by preserving the curvature of the losses, our reduction allows us to apply (quasi)second-order methods like Online Newton Step (ONS) [20].

Exp-concave Losses The following proposition shows that exp-concave losses retain the crucial property required to apply ONS under our reduction (proof in Appendix F).

Proposition 3. Let ℓ t : W → R be a β -exp-concave function, let H be an RKHS with feature map ϕ ( t ) ∈ H , and define ˜ ℓ t ( W ) = ℓ t ( Wϕ ( t )) for W ∈ L ( H , W ) . Then for any X,Y ∈ L ( H , W ) ,

<!-- formula-not-decoded -->

Note that this is precisely the curvature assumption that is required to run Kernelized ONS (KONS) [7, 8]. Hence, applying our reduction Theorem 1 with KONS to the loss sequence ˜ ℓ 1 , . . . , ˜ ℓ T leads immediately to the following dynamic regret guarantee, adapted from Calandriello et al. [8, Theorem 1].

Theorem 6. Let ℓ 1 , . . . , ℓ T be a sequence of β -exp-concave losses. For any sequence u 1 , . . . , u T ∈ W and U ∈ L ( H , W ) satisfying u t = Uϕ ( t ) for all t , Algorithm 1 applied with KONS guarantees

<!-- formula-not-decoded -->

where G ≥ ∥∇ ℓ t ( w ) ∥ for all w ∈ W , k max = max t k ( t, t ) , d eff ( λ ) = Tr ( K T ( K T + λI ) -1 ) , and K T = ( ⟨∇ ℓ i ( w i ) , ∇ ℓ j ( w j ) ⟩ W k ( i, j )) T i,j =1 ∈ R T × T .

In the previous section, we observed a direct trade-off between the complexity of the comparatormeasured in terms of ∥ u ∥ H -and a term measuring the complexity of the RKHS, max t k ( t, t ) . Here we again see a trade-off in measures of complexity, but now the complexity of the RKHS is characterized by the effective dimension d eff ( λ ) . Loosely speaking, the effective dimension represents the number of 'non-negligable directions' spanned by the features ϕ (1) , . . . , ϕ ( T ) , characterized by the number of eigenvectors of K T associated with non-negligable eigenvalues relative to λ .

Strongly-convex Losses Interestingly, for strongly-convex losses it can be shown that an analogous curvature condition to Proposition 3 holds under our reduction as well, leading to an analogous result to Theorem 6. Indeed, the main difference is that in the strongly-convex setting, one uses the feature covariance λI + ∑ t s =1 ϕ ( t ) ⊗ ϕ ( t ) to define a weighted norm while KONS the covariance matrix of the product kernel associated with features ˜ ϕ ( t ) = g t ⊗ ϕ ( t ) . Applying a similar argument then leads to a guarantee which is analogous to Theorem 6 (see Appendix F.1 for more details).

Online Linear Regression Similar results also apply in the context of online regression. In that setting, at the start of round t the learner first observes a context x t ∈ T , then predicts a ̂ y t ∈ R , and incurs a loss ℓ t ( ̂ y ) = 1 2 ( y t -̂ y ) 2 . In this setting, our reduction recovers kernelized online regression , by letting ̂ y t = ⟨ f, ϕ ( x t ) ⟩ where f ∈ L ( H , R ) and ϕ ( x t ) ∈ H is the feature map associated with H . Applying the Kernelized Vovk-Azoury-Warmuth forecaster [2, 27, 50] guarantees regret of the same form as above. The result follows from J´ ez´ equel et al. [27, Proposition 1 and Proposition 2].

Proposition 4. Let W = R and for all t let ℓ t ( ̂ y ) = 1 2 ( y t -̂ y ) 2 . Then for any sequence ( x 1 , y 1 ) , . . . , ( x T , y T ) in T × W and any benchmark sequence ˚ y 1 , . . . , ˚ y T in R and u ∈ H satisfying ˚ y t = ⟨ u, ϕ ( x t ) ⟩ for all t , the Kernelized VAW Forecaster guarantees

<!-- formula-not-decoded -->

where k max = max t k ( t, t ) and y 2 max = max t y 2 t .

It is known that the dependence on d eff ( λ ) for kernel ridge regression is optimal [30], demonstrating that these trade-offs are unimprovable in the context of dynamic regret as well.

In each of the results above, the main trade-off is between the comparator norm λ ∥ u ∥ 2 H and the effective dimension d eff ( λ ) . As an illustrative example, the following shows that the linear spline kernel can achieve non-trivial squared path-length guarantees, which were recently shown to be unattainable in the OLO setting [26].

Example 3. The linear spline kernel k ( s, t ) = min( s, t ) has well-known RKHS norm of ∥ u ∥ 2 H = ∥∇ u ∥ 2 L 2 = ∫ |∇ u ( t ) | 2 dt . Moreover, in Appendix F.2 we show that we can bound ∥∇ u ∥ L 2 ≤ O ( √∑ t | ˚ y t -˚ y t -1 | 2 W ) := C ′ T and that the effective dimension is d eff ( λ ) = O ( T/ √ λ ) (Theorems 8 and 9 respectively). Optimally tuning λ leads to R T (˚ y 1 , . . . , ˚ y T ) = ˜ O ( T 2 / 3 ( C ′ T ) 2 / 3 ) which matches the minimax optimal rate for forecasting in the class of discrete Sobolev sequences of bounded variation [3, 44]. Note that λ can be tuned without data-dependent prior knowledge using mixtureof-experts and a simple clipping argument [25, 34].

In the special case of the 1-dimensional squared loss ℓ t ( y ) = 1 2 ( y t -y ) 2 , it is possible to achieve R T (˚ y 1 , . . . , ˚ y T ) = ˜ O ( T 1 / 3 C 2 / 3 T ) where C T = ∑ t | ˚ y t -˚ y t -1 | is the (unsquared) path-length of the benchmark predictions , and this bound is minimax optimal among the class of discrete TV-bounded sequences, which is more general than the Sobolev class in the example above [3, 54]. Designing a kernel with a suitable effective dimension to achieve this trade-off has proven non-trivial and is left as a direction for future work.

## 6 Directional Adaptivity

An exciting benefit of reducing to static regret is that we can leverage more 'exotic' static regret guarantees to uncover new and interesting trade-offs in dynamic regret, essentially for free. For example, in recent years there has been an interest in algorithms which adapt to the directional covariance between the comparator and the losses [13, 15, 16, 37, 49], to guarantee

<!-- formula-not-decoded -->

These bounds recover the usual ˜ O ( ∥ u ∥ W √ ∑ T t =1 ∥ g t ∥ 2 W , ∗ ) bounds in the worst case, but could be significantly smaller if the comparator tends to be orthogonal to the losses. Passing from dynamic regret to static regret via Theorem 1, the following proposition shows that guarantees of this form translate into dynamic regret guarantees which naturally decouple the comparator variability ∥ U ∥ HS from the a per-round directional variance penalty ∑ T t =1 ⟨ g t , u t ⟩ 2 W . The full statement and proof of this result can be found in Appendix G.

Proposition 5. Let ℓ 1 , . . . , ℓ T be an arbitrary sequence of G -Lipschitz convex loss functions over W . There exists an algorithm such that for any sequence of u 1 , . . . , u T in W and U ∈ L ( H , W ) satisfying u t = Uϕ ( t ) for all t , the dynamic regret R T ( u 1 , . . . , u T ) is bounded by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where g t ∈ ∂ℓ t ( w t ) , L 2 k = G 2 max t k ( t, t ) , K T = ( ⟨ g t , g s ⟩ W k ( t, s )) t,s ∈ [ T ] , and d eff ( λ ) = Tr( K T ( λI + K T ) -1 ) .

## 7 Discussion

In this paper we developed a general reduction from dynamic regret to static regret based on embedding the comparator sequence as a function in an RKHS. We showed that the optimal √ P T path-length dependence of can be obtained via a carefully designed translation-invariant kernel. We also developed new scale-free and directionally-adaptive guarantees for online linear optimization and ∥ u ∥ 2 H + d eff ( λ ) ln T bounds for losses with curvature.

There are many promising directions for future work. As noted in Section 4, if implemented naively, the algorithms described here could be prohibitively expensive to run in practice. Future work should study how to best leverage kernel approximation techniques or sparse dictionary methods to achieve the standard O ( d ln T ) per-round computation without ruining the desired regret bounds. We also anticipate many interesting directions for future work by investigating the rich intersections between online learning, kernel methods, and signal processing that our reduction brings to light.

## Acknowledgments and Disclosure of Funding

AJ and NCB acknowledge the financial support from the FAIR project, funded by the NextGenerationEU program within the PNRR-PE-AI scheme (M4C2, investment 1.3, line on Artificial Intelligence), the MUR PRIN grant 2022EKNE5K (Learning in Markets and Society), funded by the NextGenerationEU program within the PNRR scheme (M4C2, investment 1.1), the EU Horizon CL4-2022-HUMAN-02 research and innovation action under grant agreement 101120237, project ELIAS (European Lighthouse of AI for Sustainability), and the One Health Action Hub, University Task Force for the resilience of territorial ecosystems, funded by Universit` a degli Studi di Milano (PSR 2021-GSA-Linea 6). AR acknowledges support from the European Research Council (grant REAL 947908).

## References

- [1] Nachman Aronszajn. Theory of reproducing kernels. Transactions of the American mathematical society , 1950.
- [2] K. S. Azoury and M. K. Warmuth. Relative loss bounds for on-line density estimation with the exponential family of distributions. Machine Learning , 2001.
- [3] Dheeraj Baby and Yu-Xiang Wang. Online forecasting of total-variation-bounded sequences. In Advances in Neural Information Processing Systems , 2019.
- [4] Dheeraj Baby and Yu-Xiang Wang. Optimal dynamic regret in exp-concave online learning. In Conference on Learning Theory . PMLR, 2021.
- [5] Dheeraj Baby and Yu-Xiang Wang. Second order path variationals in non-stationary online learning. In Francisco Ruiz, Jennifer Dy, and Jan-Willem van de Meent, editors, Proceedings of The 26th International Conference on Artificial Intelligence and Statistics . PMLR, 2023.
- [6] Alain Berlinet and Christine Thomas-Agnan. Reproducing kernel Hilbert spaces in probability and statistics . Springer Science &amp; Business Media, 2011.
- [7] Daniele Calandriello, Alessandro Lazaric, and Michal Valko. Efficient second-order online kernel learning with adaptive embedding. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems . Curran Associates, Inc., 2017.
- [8] Daniele Calandriello, Alessandro Lazaric, and Michal Valko. Second-order kernel online convex optimization with adaptive sketching. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning . PMLR, 2017.
- [9] Nicol` o Campolongo and Francesco Orabona. A closer look at temporal variability in dynamic online learning. arXiv preprint arXiv:2102.07666 , 2021.
- [10] Nicol` o Cesa-Bianchi and G´ abor Lugosi. Prediction, learning, and games . Cambridge university press, 2006.
- [11] Nicol` o Cesa-Bianchi and Francesco Orabona. Online learning algorithms. Annual Review of Statistics and Its Application , 2021.
- [12] Ashok Cutkosky. Artificial constraints and hints for unbounded online learning. In Alina Beygelzimer and Daniel Hsu, editors, Proceedings of the Thirty-Second Conference on Learning Theory . PMLR, 2019.
- [13] Ashok Cutkosky. Better full-matrix regret via parameter-free online learning. Advances in Neural Information Processing Systems , 2020.
- [14] Ashok Cutkosky. Parameter-free, dynamic, and strongly-adaptive online learning. In Hal Daum´ e III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning . PMLR, 2020.
- [15] Ashok Cutkosky and Zakaria Mhammedi. Fully unconstrained online learning. In The Thirtyeighth Annual Conference on Neural Information Processing Systems , 2024.
- [16] Ashok Cutkosky and Francesco Orabona. Black-box reductions for parameter-free online learning in banach spaces. In S´ ebastien Bubeck, Vianney Perchet, and Philippe Rigollet, editors, Proceedings of the 31st Conference On Learning Theory . PMLR, 2018.

- [17] CARLOS M Da Fonseca and VICTOR Kowalenko. Eigenpairs of a family of tridiagonal matrices: three decades later. Acta Mathematica Hungarica , 2020.
- [18] Geoffrey J. Gordon. Regret bounds for prediction problems. In Proc. of the twelfth annual conference on Computational learning theory (COLT) , 1999.
- [19] Eric C. Hall and Rebecca M. Willett. Online convex optimization in dynamic environments. IEEE Journal of Selected Topics in Signal Processing , 2015.
- [20] E. Hazan, A. Agarwal, and S. Kale. Logarithmic regret algorithms for online convex optimization. Machine Learning , 2007.
- [21] Mark Herbster and Manfred K Warmuth. Tracking the best regressor. In Proceedings of the eleventh annual conference on Computational learning theory , 1998.
- [22] Mark Herbster and Manfred K Warmuth. Tracking the best linear predictor. Journal of Machine Learning Research , 2001.
- [23] Andrew Jacobsen and Ashok Cutkosky. Parameter-free mirror descent. In Po-Ling Loh and Maxim Raginsky, editors, Proceedings of Thirty Fifth Conference on Learning Theory . PMLR, 2022.
- [24] Andrew Jacobsen and Ashok Cutkosky. Unconstrained online learning with unbounded losses. In International Conference on Machine Learning (ICML) . PMLR, 2023.
- [25] Andrew Jacobsen and Ashok Cutkosky. Online linear regression in dynamic environments via discounting. In International Conference on Machine Learning (ICML) . PMLR, 2024.
- [26] Andrew Jacobsen and Francesco Orabona. An equivalence between static and dynamic regret minimization. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [27] R´ emi J´ ez´ equel, Pierre Gaillard, and Alessandro Rudi. Efficient online learning with kernels for adversarial large scale problems. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alch´ eBuc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems . Curran Associates, Inc., 2019.
- [28] David W Kammler. A first course in Fourier analysis . Cambridge University Press, 2007.
- [29] Wouter M Koolen, Alan Malek, Peter L Bartlett, and Yasin Abbasi-Yadkori. Minimax time series prediction. In Advances in Neural Information Processing Systems , 2015.
- [30] Tor Lattimore. A lower bound for linear and kernel regression with adaptive covariates. In The Thirty Sixth Annual Conference on Learning Theory . PMLR, 2023.
- [31] Fanghui Liu, Xiaolin Huang, Yudong Chen, and Johan A. K. Suykens. Random features for kernel approximation: A survey on algorithms, theory, and beyond. IEEE Transactions on Pattern Analysis and Machine Intelligence , 2022.
- [32] Laszlo Losonczi. Eigenvalues and eigenvectors of some tridiagonal matrices. Acta Mathematica Hungarica , 1992.
- [33] Haipeng Luo, Mengxiao Zhang, Peng Zhao, and Zhi-Hua Zhou. Corralling a larger band of bandits: A case study on switching regret for linear bandits. In Po-Ling Loh and Maxim Raginsky, editors, Proceedings of Thirty Fifth Conference on Learning Theory . PMLR, 2022.
- [34] Jack J. Mayo, Hedi Hadiji, and Tim van Erven. Scale-free unconstrained online learning for curved losses. In Proceedings of Thirty Fifth Conference on Learning Theory , 2022.
- [35] Brendan Mcmahan and Matthew Streeter. No-regret algorithms for unconstrained online convex optimization. In F. Pereira, C. J. C. Burges, L. Bottou, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems . Curran Associates, Inc., 2012.
- [36] H. Brendan McMahan and Francesco Orabona. Unconstrained online linear learning in Hilbert spaces: Minimax algorithms and normal approximations. In Maria Florina Balcan, Vitaly Feldman, and Csaba Szepesv´ ari, editors, Proceedings of The 27th Conference on Learning Theory . PMLR, 2014.
- [37] Zakaria Mhammedi and Wouter M. Koolen. Lipschitz and comparator-norm adaptivity in online learning. In Jacob Abernethy and Shivani Agarwal, editors, Proceedings of Thirty Third Conference on Learning Theory . PMLR, 2020.
- [38] Frank WJ Olver. NIST handbook of mathematical functions . Cambridge university press, 2010.

- [39] Francesco Orabona. A modern introduction to online learning. arXiv preprint arXiv:1912.13213 , 2019.
- [40] Francesco Orabona and D´ avid P´ al. Coin betting and parameter-free online learning. In Proceedings of the 30th International Conference on Neural Information Processing Systems . Curran Associates Inc., 2016.
- [41] Francesco Orabona and D´ avid P´ al. Parameter-free stochastic optimization of variationally coherent functions. arXiv preprint arXiv:2102.00236 , 2021.
- [42] Vern I Paulsen and Mrinal Raghupathi. An introduction to the theory of reproducing kernel Hilbert spaces . Cambridge university press, 2016.
- [43] DE Rutherford. Some continuant determinants arising in physics and chemistry. Proceedings of the Royal Society of Edinburgh Section A: Mathematics , 1948.
- [44] Veeranjaneyulu Sadhanala, Yu-Xiang Wang, and Ryan J Tibshirani. Total variation classes beyond 1d: Minimax rates, and the limitations of linear smoothers. In D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing Systems . Curran Associates, Inc., 2016.
- [45] Bernhard Sch¨ olkopf and Alexander J Smola. Learning with kernels: support vector machines, regularization, optimization, and beyond . MIT press, 2002.
- [46] Elias M Stein and Guido Weiss. Introduction to Fourier analysis on Euclidean spaces . Princeton university press, 1971.
- [47] Shiliang Sun, Jing Zhao, and Jiang Zhu. A review of Nystr¨ om methods for large-scale machine learning. Information Fusion , 2015.
- [48] Ernest Oliver Tuck. On positivity of fourier transforms. Bulletin of the Australian Mathematical Society , 2006.
- [49] T. van Erven and W. M. Koolen. MetaGrad: Multiple learning rates in online learning. In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing Systems 29 . Curran Associates, Inc., 2016.
- [50] V. Vovk. Competitive on-line statistics. International Statistical Review , 2001.
- [51] Holger Wendland. Scattered data approximation . Cambridge university press, 2004.
- [52] Tianbao Yang, Lijun Zhang, Rong Jin, and Jinfeng Yi. Tracking slowly moving clairvoyant: Optimal dynamic regret of online learning with true and noisy gradient. In Maria Florina Balcan and Kilian Q. Weinberger, editors, Proceedings of The 33rd International Conference on Machine Learning . PMLR, 2016.
- [53] Lijun Zhang, Shiyin Lu, and Zhi-Hua Zhou. Adaptive online learning in dynamic environments. In Proceedings of the 32nd International Conference on Neural Information Processing Systems , 2018.
- [54] Yu-Jie Zhang, Peng Zhao, and Masashi Sugiyama. Non-stationary online learning for curved losses: Improved dynamic regret via mixability, 2025.
- [55] Zhiyu Zhang, Ashok Cutkosky, and Yannis Paschalidis. Unconstrained dynamic regret via sparse coding. Advances in Neural Information Processing Systems , 2024.
- [56] Peng Zhao, Yu-Jie Zhang, Lijun Zhang, and Zhi-Hua Zhou. Dynamic regret of convex and smooth functions. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems . Curran Associates, Inc., 2020.
- [57] Peng Zhao, Yan-Feng Xie, Lijun Zhang, and Zhi-Hua Zhou. Efficient methods for nonstationary online learning. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems . Curran Associates, Inc., 2022.
- [58] Peng Zhao, Yu-Jie Zhang, Lijun Zhang, and Zhi-Hua Zhou. Adaptivity and non-stationarity: Problem-dependent dynamic regret for online convex optimization. Journal of Machine Learning Research , 2024.
- [59] Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In Proceedings of the 20th international conference on machine learning (icml-03) , 2003.

## A Additional Functional Analysis Background

In this section we briefly recall some additional definitions and background from functional analysis, which will be useful for understanding the proofs of our results but were not relevant for the main text.

Additional Notations. The dual space of a normed space H is the space of bounded linear functionals H ∗ = L ( H , R ) , and the associated norm is the dual norm ∥ g ∥ H , ∗ = sup ∥ h ∥ H =1 g ( h ) for any g ∈ H ∗ . An operator T : H → W is Hilbert-Schmidt if for any orthonormal basis { h i } i of H we have ∥ T ∥ 2 HS := ∑ i ∥ Th i ∥ 2 W &lt; ∞ . The space L ( H , W ) is itself a Hilbert space with inner product ⟨ A,B ⟩ HS = ∑ i ⟨ Ah i , Bh i ⟩ W .

A Brief Review of Reproducing Kernel Hilbert Spaces. A linear functional φ ∈ L ( H , R ) is bounded if there exists a constant M such that | φ ( x ) | ≤ M ∥ x ∥ H for all h ∈ H . A reproducing kernel Hilbert space (RKHS) is a Hilbert space H of functions h : X → R for which the evaluation functional δ x : h ↦→ h ( x ) is bounded for all x ∈ X . For any such space, Riesz representation theorem tells us that for any x ∈ X there is a unique k x ∈ H such that δ x ( h ) = ⟨ h, k x ⟩ H for all h ∈ H . The function k ( x, x ′ ) = k x ( x ′ ) is called the reproducing kernel associated with H . The reproducing kernel is often expressed in terms of the feature map ϕ ( x ) = k x ∈ H as k ( x, x ′ ) = ⟨ ϕ ( x ) , ϕ ( x ′ ) ⟩ H .

We will often be interested in functions taking values in W ⊆ R d . In this case the preceding discussion can be extended in a straightforward way by considering a coordinate-wise extension. In particular, observe that in this setting an operator W ∈ L ( H , W ) can be represented as a tuple ( W 1 , . . . , W d ) such that W i ∈ L ( H , R ) = H ∗ for each i ∈ [ d ] . Riesz representation theorem then tells us that there is a w i ∈ H such that W i ( h ) = ⟨ w i , h ⟩ H for any h ∈ H , so using the reproducing property we have W i ( ϕ ( t )) = ⟨ w i , ϕ ( t ) ⟩ H = w i ( t ) . Hence, each W ∈ L ( H , W ) is identified by a tuple ( W 1 , . . . , W d ) ∈ H d and we can write

<!-- formula-not-decoded -->

Note the that space L ( H , W ) is itself a Hilbert space when equipped with the Hilbert-Schmidt norm, which in the coordinate-wise extension above can be expressed as ∥ W ∥ 2 HS = ∥ w ∥ 2 H d = ∑ d i =1 ∥ w i ∥ 2 H . This can be seen by definition of the Hilbert-Schmidt norm: let { h i } i be an orthonormal basis of H , then

<!-- formula-not-decoded -->

where the last line uses Parseval's identity.

## B Recovering Jacobsen and Orabona [26]

In this section we demonstrate that the reduction in Jacobsen and Orabona [26] is equivalent to the special case of our framework. Note that we assume linear losses in this section because the reduction of Jacobsen and Orabona [26] is only defined for linear losses (or by linearizing the losses ℓ t via convexity).

Let e t ∈ R T be the t th standard basis vector and consider Algorithm 1 with H = R T , ⟨ A,B ⟩ HS = Tr ( A ⊤ B ) , kernel feature map ϕ ( t ) = e t ∈ R T , and linear losses W ↦→ ⟨ G t , W ⟩ HS for G t =

g t ⊗ e t = g t e ⊤ t ∈ R d × T . Then for any sequence u 1 , . . . , u T in R d , let U = ( u 1 . . . u T ) ∈ R d × T and observe that we can write u t = Uϕ ( t ) . Moreover,

<!-- formula-not-decoded -->

Similarly, suppose A is an online learning algorithm and let W t = ( w (1) t . . . w ( T ) t ) ∈ R d × T denote its output on round t . Suppose on round t we play w t = W t ϕ ( t ) = w ( t ) t . Then

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

To see why this is precisely equivalent to the reduction of Jacobsen and Orabona [26], observe that their reduction is simply phrased in terms of the 'flattened' versions of each of the above quantities, and can be interpreted as working in the finite-dimensnional RKHS over ˜ H = R dT with ⟨ x, y ⟩ ˜ H = ⟨ x, y ⟩ being the canonical inner product on R dT . In particular, they instead define

<!-- formula-not-decoded -->

and run an algorithm ˜ A defined on ˜ H = R dT against the losses ˜ g t . As shown in their Proposition 1, under this setup it holds that ∑ T t =1 ⟨ g t , w t -u t ⟩ = ∑ T t =1 ⟨ ˜ g t , ˜ w t -˜ u ⟩ , from which it immediately follows that

<!-- formula-not-decoded -->

## C Proofs for Section 4 (Linear Losses)

## C.1 Proof of Theorem 2

Theorem 2. Let A be an online learning algorithm defined on Hilbert space V . Suppose that for any sequence of convex loss functions h 1 , . . . , h T on V , A obtains a bound on the static regret of the form ˜ R T ( U ) ≤ B T ( ∥ U ∥ V , ∥∇ h 1 ( W 1 ) ∥ V , . . . , ∥∇ h T ( W T ) ∥ V ) for any comparator U ∈ V and some function B T : R T +1 ≥ 0 → R , where ∇ h t ( W t ) ∈ ∂h t ( W t ) for all t . If we apply A in V = L ( H , W ) with ∥·∥ V = ∥·∥ HS , then for any sequence u 1 , . . . , u T in W and U ∈ L ( H , W ) satisfying u t = Uϕ ( t ) for all t , Algorithm 1 with A guarantees

<!-- formula-not-decoded -->

where g t ∈ ∂ℓ t ( w t ) for all t , and k ( · , · ) is the reproducing kernel associated to the space H .

Proof. We note that L ( H , W ) is a separable Hilbert space for separable H and W . Moreover, ˜ ℓ t is differentiable, with derivative

<!-- formula-not-decoded -->

where ⊗ is the tensor product. Then applying algorithm A to the loss sequence ( ˜ ℓ t ) t , we obtain

<!-- formula-not-decoded -->

The proof is concluded by noting that R T ( u 1 , . . . , u T ) = ˜ R T ( U ) by Theorem 1, and that

<!-- formula-not-decoded -->

since u ⊗ v ∈ L ( U , V ) is a rank one operator and so ∥ u ⊗ v ∥ L ( U , V ) = ∥ u ∥ U ∥ v ∥ V , for any u, v ∈ U , V and U , V Hilbert spaces. Finally, note that ∥ ϕ ( t ) ∥ 2 H = ⟨ ϕ ( t ) , ϕ ( t ) ⟩ = k ( t, t ) .

Remark 1. Note that the result of Theorem 2 applies more generally to algorithms that use dualweighted-norm pairs ( ∥·∥ M , ∥·∥ M -1 ) , since this amounts to transforming the decision space W ↦→ M 1 2 W , the losses G t ↦→ M -1 2 G t , and preserving the original inner product structure. We expect the theorem should also generalize to arbitrary dual-norm pairs on W , but this will require some additional care to interpret the norm ∥·∥ W ∗ ⊗H .

## C.2 A Concrete Example of Proposition 1

<!-- formula-not-decoded -->

There are many examples of algorithms which would produce the static regret guarantee stated in Proposition 1. In this section, we briefly provide an example which attains the result of the stated form, and provide the full regret guarantee and update in our framework.

Let us consider the algorithm characterized by Jacobsen and Cutkosky [23, Theorem 1] in an unconstrained setting. Their algorithm can be understood as a particular instance of FTRL, and so we can develop its kernelized version using the same reasoning as Example 1. Indeed, applying their algorithm in the space L ( H , W ) with inner product ⟨· , ·⟩ HS against losses G t = g t ⊗ ϕ ( t ) leads to updates of the form

<!-- formula-not-decoded -->

where V t +1 = 4 G 2 0 + ∑ t s =1 ∥ G s ∥ 2 HS and Ψ( S, V ) defined in Algorithm 2. Moreover, we have V t +1 = 4 G 2 0 + ∑ t s =1 ∥ g s ∥ 2 W k ( s, s ) and ∥ ∥ ∥ ∑ t s =1 G s ∥ ∥ ∥ HS = ∑ t i,j ⟨ g i , g j ⟩ k ( i, j ) via Lemma 10 and Lemma 9 respectively, and so in the context of our reduction Algorithm 1 the updates are

<!-- formula-not-decoded -->

leading to the procedure described in Algorithm 2. Notice that, as mentioned in Section 4, this naive implementation requires O ( t ) time and memory to update on round t due to having to re-weight the sum ∑ t s =1 k ( s, t ) g s and compute ∑ t -1 s =1 k ( s, t ) ⟨ g t , g s ⟩ W , so in practice one would ideally implement additional measures to reduce the complexity, such as implementing Nystrom projections or choosing a suitably sparse kernel.

Nowapplying Algorithm 2 with Theorem 2, we immediately get the following regret guarantee from Jacobsen and Cutkosky [23, Theorem 1].

Proposition 6. Let ℓ 1 , . . . , ℓ T be G -Lipschitz convex loss functions and let g t ∈ ∂ℓ t ( w t ) for all t . For any u 1 , . . . , u T in W and U ∈ L ( H , W ) satisfying u t = Uϕ ( t ) for all t , Algorithm 2 guarantees

<!-- formula-not-decoded -->

## C.3 Scale-free Guarantees

Now that we have seen how to obtain the optimal path-length dependencies on Lipschitz losses, we can extend these guarantees to be scale-free by simply changing the base algorithm. In particular, there are algorithms which are adaptive to both the comparator norm and the effective Lipschitz constant, L T = max t ∈ [ T ] ∥∇ ℓ t ( w t ) ∥ W . Algorithms which scale with L T rather than a given upper bound G ≥ L T are referred to as scale-free . We first consider the setting in which the domain is constrained W = { w ∈ R d : ∥ w ∥ W ≤ D } . 4

Proposition 7. There exists an algorithm which guarantees that for any sequence u 1 , . . . , u T in W = { w ∈ R d : ∥ w ∥ W ≤ D } ,

<!-- formula-not-decoded -->

where L t = max t ∥ g t ∥ W .

4 More generally, this assumption amounts to assuming prior knowledge on a bound D ≥ ∥ u t ∥ W for all t , which the learner can leverage by projecting to the same set, regardless of any boundedness of the underlying problem's domain.

The result follows by constraining ∥ w t ∥ W ≤ D and applying the gradient-clipping argument of Cutkosky [12]. 5 Indeed, if we've constrained our iterates to satisfy ∥ w t ∥ W ≤ D , then we can replace the gradients g t the the clipped gradients ̂ g t = g t min { 1 , max s&lt;t ∥ g s ∥ W ∥ g t ∥ W } to get

<!-- formula-not-decoded -->

following the same telescoping argument as Cutkosky [12]. With this in hand, we can simply apply our reduction Algorithm 1 with the losses ̂ G t = ̂ g t ⊗ ϕ ( t ) , which we now have an a priori bound on at the start of round t : ∥ ∥ ∥ ̂ G t ∥ ∥ ∥ HS ≤ max s&lt;t ∥ g s ∥ W √ k ( t, t ) := ̂ L t ≤ L t . Hence, even without prior knowledge of a G ≥ ∥ g t ∥ W we can obtain ̂ R T ( u 1 , . . . , u T ) ≤ ˜ O ( ∥ U ∥ HS √ max t ∥ g t ∥ 2 W k ( t, t ) + ∑ T t =1 ∥ g t ∥ 2 W k ( t, t ) ) .

To see why this is difficult using existing techniques, note that nearly all existing algorithms which achieve the optimal √ P T dependence do so by designing an algorithm which guarantees dynamic regret of the form

<!-- formula-not-decoded -->

from which G √ P T T regret is obtained by tuning η . 6 The tuning step is done by running several instances of the base algorithm in parallel for each η in some set S = { 2 i /G √ T ∧ 1 /G : i = 0 , 1 , . . . } , and combining the outputs-typically using a mixture-of-experts algorithm like Hedge. Note however that the set S requires prior knowledge of the Lipschitz constant G . There is no straightforward way to adapt to this argument without resorting to unsatisfying doubling strategies, which are well-known to perform poorly in practice. Instead, using our framework we avoid these issues entirely by simply applying a scale-free static regret guarantee to get the a L T ∥ U ∥ HS √ T dependence, and then designing a kernel which ensures ∥ U ∥ HS ≤ √ MP T .

More generally, when the domain W is not uniformly bounded, it is still possible to achieve a scale-free bound at the expense of an L T max t ∥ u t ∥ 3 W penalty, again using the same argument as [12]. One simply starts by replacing the comparator sequence u 1 , . . . , u T with a new one satisfying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Applying the same clipping argument as above and observing that ̂ P T = ∑ t ∥ ̂ u t -̂ u t -1 ∥ W ≤ O ( P T +max t ∥ u t ∥ W ) we get the following result.

Proposition 8. There exists an algorithm such that for any u 1 , . . . , u T in W ⊆ R d ,

<!-- formula-not-decoded -->

where L T = max t ∥ g t ∥ W √ k ( t, t ) .

5 Note that constraining the final outputs w t is straight-forward in our framework; one can simply apply a standard unconstrained-to-constrained reduction in W [14, 16] prior to applying our dynamic-to-static reduction.

6 The exception being Zhang et al. [55], which uses a similar high-dimensional embedding as [26], but neither works obtain the optimal √ P T while being scale-free.

## D Intuitions on the Spectral Density Q ( ω )

In this section we provide some additional high-level intuitions that motivate the choice of Q ( ω ) in Proposition 2.

Recall that we would like to choose Q ( ω ) ≈ 1 / | ω | , since this choice yields a continuous-time analogue of the MP T dependence that we want to achieve: ∥ u ∥ 2 H ≤ max t | u ( t ) | ∥∇ u ∥ L 1 . The key issue is that this choice of Q ( ω ) is not integrable, diverging as ω → 0 and ω → ∞ . We fix these issues by adding a small amount of additional regularization, setting

<!-- formula-not-decoded -->

where R ∞ ( ω ) ≈ 1 / (1 + | ω | 2 ) 1 / 4 is a tapering function that ensures integrability in the asymptotic regime, and R 0 ( ω ) ≈ 1 / log(1 + 1 / √ | ω | ) log 2 (log(1 + 1 / √ | ω | )) ensures that Q is well-behaved near zero. It is clear that R ∞ ( ω ) ensures integrability in the asymptotic regime since when ω is large we have Q ( ω ) ≈ R ∞ ( ω ) / | ω | ≤ 1 / | ω | 2 , which is integrable away from zero. On the other hand, near zero R ∞ ( ω ) ≈ 1 and we have

<!-- formula-not-decoded -->

which after a change of variables t = log( ω -1 / 2 ) integrates near zero as

<!-- formula-not-decoded -->

for an appropriately chosen ϵ .

## E Proofs for Section 4.1 (Controlling the Trade-offs Induced by H )

## E.1 Proof of Theorem 3

Theorem 3. Let Q : R → R + be an integrable strictly positive even function on R \ { 0 } and such that R ( x ) := 2 π/ ( x (1 + ( x/ 2 π ) 2 m ) Q ( x )) is also integrable for some m ∈ N , m ≥ 1 . Let k be defined in terms of Q as in Eq. (3) . Then k is a translation invariant universal kernel with k ( t, t ) ≤ ∥ Q ∥ L 1 for all t ∈ R . The RKHS H associated to k contains the space of finitely supported functions with bounded derivatives up to order 2 m , and moreover, for any T &gt; 0 and any 2 m -times differentiable function f that is supported on [0 , T +1] , where c ( T ) := ∥ F [ R ] ∥ 1 --. If R is monotonically decreasing on (0 , ∞ )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Consider a function f that is supported on [0 , τ ] for some τ &gt; 0 . To bound the norm above in terms of the L 1 norm we use the fact that for any u, v, w ∈ L 2 ( R ) ⊕ L 1 ( R ) we have F [ ∇ k u ] = (2 πi ) k ̂ u , for k ∈ N , F [ u ⋆ v ] = ̂ u ̂ v , where ⋆ is the convolution operator, and that by the Plancherel theorem we have ∫ R ̂ u ( ω ) ̂ v ( ω ) dt = ∫ R u ( t ) v ( t ) dt and so, in particular, ∫ R ̂ u ( ω ) ̂ v ( ω ) ̂ w ( ω ) dt = ∫ R u ( t )( v ⋆ w )( t ) dt . Moreover, note that, by construction R is an integrable real odd function, so its Fourier transform ̂ R is an odd purely imaginary function. So, we have

<!-- formula-not-decoded -->

where in the last two steps we used the fact that f is bounded in [0 , τ ] and the H¨ older inequality. Since also ∇ m f is supported on [0 , τ ] , we have

<!-- formula-not-decoded -->

The last step is finding a bound that is easy to compute for c ( τ ) := ∥ ̂ R ∥ L 1 ([ -τ,τ ]) . We start noting that since R is odd

<!-- formula-not-decoded -->

and, in particular also ̂ R is an odd function. The characterization we propose for c ( τ ) is a direct consequence of the fact that the sine transform of R is non-negative (also known as Polya criterion, which we recall in Lemma 3 and is applicable since R is positive and decreasing on (0 , ∞ ) ). Now, by expanding the definition of ̂ R and using the non-negativity of the sine transform, we have

<!-- formula-not-decoded -->

To conclude, let α &gt; 0 , since | sin( z ) | ≤ min( | z | , 1) for any z ∈ R

<!-- formula-not-decoded -->

The stated result then follows by choosing τ = T +1 .

## E.2 Proof of Theorem 4

Before proving Theorem 4 we need two auxiliary results

Lemma 1. Given m ∈ N with m ≥ 1 and T &gt; 0 there exists a function b T that is 2 m -times differentiable and that is identically equal to 0 on R \ (0 , T +1) and that is identically equal to 1 on the interval [1 , T ] . Moreover, for any 2 m -times differentiable function f , with derivatives in L p ( S ) for p ∈ [1 , ∞ ] and interval S ⊆ R , we have

<!-- formula-not-decoded -->

Proof. Consider the function

<!-- formula-not-decoded -->

where Γ is the gamma function. Then B is supported on ( -1 / 2 , 1 / 2) , integrates to 1 , and is 2 m -times differentiable everywhere. Its Fourier transform (see [46], Thm. 4.15) is

<!-- formula-not-decoded -->

where J ν is the Bessel J function of order ν [46]. Since J ν is analytic on [0 , ∞ ) for ν &gt; 0 and J ν ( z ) = O ( | z | ν ) when | z |→ 0 and also J ν ( z ) = O ( z -1 / 2 ) for z →∞ , then ̂ B is in L 1 ∩ L ∞ and analytic. We build b T as follows

<!-- formula-not-decoded -->

Figure 2: Plots demonstrating the functions used in the construction of b T ( t ) for m = 1 and T = 5 . The function B ( x ) is a simple bump function, designed such that ∫ R B ( x ) dx = 1 , shown on the left. The center demonstrates how we can combine translations of B ( x ) to get a function with two bumps which will eventually cancel out when integrated over [0 , T +1] , leading to the function b T shown on the right.

<!-- image -->

This function by construction is 2 m -times differentiable everywhere, moreover it is identically equal to 0 on R \ (0 , T +1) and identically equal to 1 on [0 , T ] . To help with intuitions, we show an example of the functions B ( x ) , B ( x -1 2 ) -B ( x -1 2 -T ) , and b T ( t ) for m = 1 and T = 5 in Figure 2. The Fourier transform of b T is

<!-- formula-not-decoded -->

Now we define u ( t ) := f ( t ) b T ( t ) . By construction, u is equal to f T on [1 , T ] since b T is identically 1 on this interval, let Z := S ∩ [0 , T +1] , we have

<!-- formula-not-decoded -->

Now, given the Fourier transform of ̂ b T ,

<!-- formula-not-decoded -->

Using the fact that | J ν ( z ) | ≤ min( z ν 2 -ν / Γ(1 + ν ) , ν -1 / 3 ) (see [38], Eq. 10.14.2, 10.14.4) for any z ≥ 0 , for any α &gt; 0 , we have

<!-- formula-not-decoded -->

Optimizing in α , we obtain

<!-- formula-not-decoded -->

leading to

<!-- formula-not-decoded -->

this leads to

<!-- formula-not-decoded -->

To conclude, note that

<!-- formula-not-decoded -->

Lemma 2. Let f be such that its Fourier transform ̂ f and its weak derivative ∇ ̂ f are both L 2 , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now by the H¨ older inequality we have

<!-- formula-not-decoded -->

where 1( x ) is the constant function 1 . Similarly,

<!-- formula-not-decoded -->

since ∥ 1 /x ∥ 2 L 2 ( R \ [ -t,t ]) = 2 ∫ ∞ t 1 /x 2 dx = 2 /t . This leads to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Optimizing on t we obtain,

The first case is concluded by applying the Plancherel theorem for which ∥ f ∥ L 2 = ∥ ̂ f ∥ L 2 and ∥ fx ∥ L 2 = ∥ F [ fx ] ∥ L 2 and by the fact that F [ f ( x ) x ]( ω ) = i/ (2 π ) ∇ ̂ f ( ω ) .

Theorem 4. Let v 1 , . . . , v T ∈ R d and let H be the RKHS associated to kernel k contain finitely supported functions with bounded derivatives up to order 2 m , with m ∈ N , m ≥ 1 . Then there exists a function u ∈ H supported on [0 , T +1] , such that u ( t ) = v t for all t ∈ [ T ] and

<!-- formula-not-decoded -->

with C, C ′ depending only on m and given in explicitly in the proof.

Proof. We will build u as follows

Proof. For any t &gt; 0 ,

<!-- formula-not-decoded -->

where b T is a m +1 -times differentiable function that is supported on [0 , T +1] and that is equal to 1 on [1 , T ] , while f is a function that interpolates v t , i.e. f ( t ) = v t for t ∈ { 1 , . . . , T } . We build f as follows. Consider the following function, that is a product of two sinc functions

<!-- formula-not-decoded -->

S satisfies S (0) = 1 and S ( t ) = 0 on t ∈ Z \ { 0 } . Now we can build f as follows

<!-- formula-not-decoded -->

By construction, for all t ∈ { 1 , . . . , T } the following holds

<!-- formula-not-decoded -->

Note that, by construction f is a band-limited function with band [ -3 / 4 , 3 / 4] that interpolates the given points.

Step 1. Bounding ∥∇ f ∥ L 1 . Now we bound pointwise |∇ f ( x ) | with | v ( x ) -v ( x -1) | . The Fourier transform ̂ f of u is equal to

<!-- formula-not-decoded -->

where g ( ω ) = ∑ T ℓ =1 v ℓ e 2 πiℓω and where ̂ S is Fourier Transform of S . Note that S is band-limited, i.e., ̂ S is equal to 0 on R \ [ -3 / 4 , 3 / 4] . In particular,

<!-- formula-not-decoded -->

Now passing by the Fourier transform of ∇ f , we obtain

<!-- formula-not-decoded -->

Note that ̂ M ( ω ) is bounded, continuous and supported in [ -3 / 4 , 3 / 4] , since ̂ S is supported in [ -3 / 4 , 3 / 4] and ω (1+ ω 2 ) 1 -e 2 πiω is bounded and analytic on such interval. So we have

<!-- formula-not-decoded -->

where ⋆ is the convolution operator and M = F -1 [ ̂ M ] , L = F -1 [ ̂ L ] . Now note that

<!-- formula-not-decoded -->

Since the inverse Fourier transform of e 2 πaω / (1 + ω 2 ) is e -| x -a | for any a ∈ R , we have

<!-- formula-not-decoded -->

By Young's inequality for the convolution, we have that ∥ f ⋆g ∥ L 1 ≤ ∥ f ∥ L 1 ∥ g ∥ L 1 for any integrable functions f, g , so in our case

<!-- formula-not-decoded -->

To conclude we have ∥ e -2 π | x -ℓ | ∥ L 1 = ∥ e -2 π | x | ∥ L 1 = 1 /π and we need to bound the L 1 norm of M . Note that ̂ M ( ω ) admits a weak derivative, since it is the product of a bounded analytic function on the support and ̂ S that admits a weak derivative that is the following

<!-- formula-not-decoded -->

In particular, we have for every ω ∈ R

<!-- formula-not-decoded -->

obtaining

<!-- formula-not-decoded -->

Using Lemma 2 we can bound ∥ M ∥ L 1 as follows,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 2. Bounding ∥∇ k f ∥ L ∞ . Let k ∈ N and k ≥ 1 . Since ̂ Z ( ω ) := ̂ S ( ω/ 4) is equal to 1 on [ -1 , 1] and it is supported on [ -3 , 3] , we have that ̂ f ( ω ) = ̂ Z ( ω ) ̂ f ( ω ) . So, by using the properties of the Fourier transform of convolutions, we have that for all t ∈ R ,

<!-- formula-not-decoded -->

where Z ( t ) = 4 S (4 t ) for all t ∈ R is the inverse Fourier transform of ̂ Z . So we have

<!-- formula-not-decoded -->

Now, we have

<!-- formula-not-decoded -->

Since | S ( t ) | ≤ 1 / (1 + 2 t 2 ) for any t ∈ R , we have

<!-- formula-not-decoded -->

To conclude this section, using Lemma 2 and since F [ ∇ k Z ] = ω k Z/ (2 πi ) k

<!-- formula-not-decoded -->

So, for any k ∈ N (including 0 , since we have Eq. (6))

<!-- formula-not-decoded -->

Step 3. Building b T and computing the final norms. Lemma 1 constructs a function b T that is 2 m -times differentiable and that is identically equal to 0 on R \ (0 , T + 1) and that is identically equal to 1 on [1 , T ] , moreover it proves that for any 2 m -times differentiable function that has the derivatives L p ( S ) integrable for some p ∈ [0 , ∞ ] and some interval S ⊆ R , we have

<!-- formula-not-decoded -->

Define u ( t ) := f ( t ) b T ( t ) . By construction u is equal to f T on [0 , T ] since b T is identically 1 on this interval, and so in particular

<!-- formula-not-decoded -->

moreover u ( t ) = f ( t ) b T ( t ) = 0 for t ∈ R \ (0 , T +1) . By applying the result above, together with Eq. (7)

<!-- formula-not-decoded -->

Applying the same lemma, with Eq. (5) we have

<!-- formula-not-decoded -->

This completes the proof.

## E.3 Proof of Lemma 3

Here, we recall a classical result about the positivity of the sine transform of a positive decreasing function (see e.g. [48] Eq. 4).

Lemma 3. Let R be an integrable, positive and strictly decreasing on (0 , ∞ ) . Then, for any t &gt; 0 we have

<!-- formula-not-decoded -->

Proof. Since sin(2 π ( z +1 / 2)) = -sin(2 πz ) for each z ∈ [ j, j +1 / 2] and j ∈ N , we have

<!-- formula-not-decoded -->

where the last step is due to the fact that R is decreasing, so R (( j + θ ) /t ) -R (( j + θ ) /t +1 / 2 t ) ≥ 0 for any j ∈ N , θ ∈ [0 , 1 / 2] and that sin(2 πθ ) ≥ 0 on the integration interval θ ∈ [0 , 1 / 2] .

## E.4 Proof of Proposition 2

Proposition 2. Let Q : R → R + be defined as

<!-- formula-not-decoded -->

Then we can apply Theorem 3 with m = 1 : the function k defined in terms of Q as in Eq. (3) is a translation invariant kernel with k ( t, t ) ≤ 8 π 2 , ∀ t ∈ R ; the associated RKHS norm satisfies

<!-- formula-not-decoded -->

for any f that is 2 -times differentiable and supported in [0 , T +1] , where T &gt; 2 and c ≤ (2 πe ) 2 .

Proof. Step 1. Characterization of Q . The function Q is even, strictly positive and analytic on R \ { 0 } . To study its integrability define the auxiliary function S : [0 , ∞ ) → [0 , ∞ ) as

<!-- formula-not-decoded -->

S is concave, strictly increasing, on (0 , ∞ ) and with S (0) = 0 and lim z →∞ S ( z ) = 1 / 2 . So its derivative S ′ corresponding to

<!-- formula-not-decoded -->

is positive and strictly decreasing on (0 , ∞ ) and since 0 &lt; s &lt; 2 m , the function ( · ) s/ 2 m is concave, and we have (1 + ( | ω | / 2 π ) 2 m ) s/ 2 m ≥ 2 s 2 m -1 (1 + ( | ω | / 2 π ) s ) , so

<!-- formula-not-decoded -->

So 0 &lt; Q ( ω ) ≤ LS ′ ( | ω | ) for any ω ∈ R , and since S ′ is positive and integrable, then Q is integrable too and we have

<!-- formula-not-decoded -->

To conclude this first step, since Q is integrable it admits a Fourier transform ̂ Q , and since it is also positive, k ( t, t ′ ) := ̂ Q ( t -t ′ ) for t, t ′ ∈ R is a translation invariant kernel. In particular, k ( t, t ) = ̂ Q (0) = ∫ R Q ( ω ) dω ≤ L , for any t ∈ R .

Step 2. Characterization of R and explicit bound for ∥ F [ R ] ∥ L 1 [ -T,T ] . The second condition to apply Theorem 3, concerns the function R defined as

<!-- formula-not-decoded -->

Note that R is an odd function, since x is odd, while both 1+( x 2 π ) 2 m and Q ( x ) are even. Moreover, R is analytic on R \{ 0 } since Q ( x ) is analytic on the same domain and x (1+( x/ 2 π ) 2 m ) is analytic on the whole axis. Expanding the definition of Q in R , we obtain

<!-- formula-not-decoded -->

where C 0 = 4 π/ ( s log log π ) . From which we observe that R is also positive and strictly decreasing on (0 , ∞ ) since, log( π + | x | -s ) , log(log( π + | x | -s )) 2 and 1 / (1 + ( ω/ 2 π ) 2 m ) 2 m -s 2 m ) are strictly positive and strictly decreasing on (0 , ∞ ) . So we can apply the bound on ∥ F [ R ] ∥ L 1 ([ -T,T ]) in Theorem 3, obtaining, for any α &gt; 0

<!-- formula-not-decoded -->

To bound such integrals, we first simplify R . Let β, γ &gt; 0 , since the following functions are bounded, non-negative, with a unique critical point that is a maximum, by equating their derivative to zero we obtain

<!-- formula-not-decoded -->

so we have for any x &gt; 0 ,

<!-- formula-not-decoded -->

where C 1 ( β, γ ) = (2 /γ ) 2 ( 1+ γ β ) 1+ γ e -3 -γ C 0 ≤ 16 C 0 / ( e 3 γ 2 β 1+ γ ) ≤ γ -2 β -1 -γ C 0 . Now we can control the integral of interest by using the bound above. First, we will split it in two regions of interest. For the first term, letting β &lt; 1 ,

<!-- formula-not-decoded -->

For the second term, we have

<!-- formula-not-decoded -->

So c ( T ) is bounded by

<!-- formula-not-decoded -->

By choosing α = 1 /T , β = 1 / log( T ) , γ = 1 / log(log( T )) , we have

<!-- formula-not-decoded -->

and, since s ≥ 1 (by assumption), 2 m -s ≥ 1 (by definition of m ) and T &gt; 3 (by assumption), we have

<!-- formula-not-decoded -->

## F Proofs for Section 5 (Curved Losses)

Proposition 3. Let ℓ t : W → R be a β -exp-concave function, let H be an RKHS with feature map ϕ ( t ) ∈ H , and define ˜ ℓ t ( W ) = ℓ t ( Wϕ ( t )) for W ∈ L ( H , W ) . Then for any X,Y ∈ L ( H , W ) ,

<!-- formula-not-decoded -->

Proof. Let x = Xϕ ( t ) and y = Y ϕ ( t ) . By definition and β -exp-concavity of ℓ t , we have

˜ ℓ t ( X ) -˜ ℓ t ( Y ) = ℓ t ( x ) -ℓ t ( y ) ≤ ⟨∇ ℓ t ( x ) , x -y ⟩ -β 2 ⟨∇ ℓ t ( x ) , x -y ⟩ 2 W = ⟨∇ ℓ t ( x ) , ( X -Y ) ϕ ( t ) ⟩ W -β 2 ⟨∇ ℓ t ( x ) , ( X -Y ) ϕ ( t ) ⟩ 2 W = ⟨∇ ℓ t ( x ) ⊗ ϕ ( t ) , ( X -Y ) ⟩ HS -β 2 ⟨∇ ℓ t ( x ) ⊗ ϕ ( t ) , ( X -Y ) ⟩ 2 HS . Observing that ∇ ˜ ℓ ( X ) = ∇ ℓ t ( x ) ⊗ ϕ ( t ) ∈ L ( H , W ) completes the proof.

## F.1 Strongly-convex Losses

In this section we show how to apply our static-to-dynamic reduction in the context of stronglyconvex losses. Interestingly, the algorithm ends up being essentially the same as the KernelizedONSalgorithm of [27], but with a weighted norm defined in terms of the feature covariance operator, Σ t = λI + β ∑ t s =1 ϕ ( s ) ⊗ ϕ ( s ) . The following lemma shows how to connect the instantaneous regret on round t to the kernelized linear losses g t ⊗ ϕ ( t ) and is analogous to Proposition 3.

Proposition 9. Let ℓ t : W → R be a β -strongly-convex function, let H be an RKHS with associated feature map ϕ ( t ) ∈ H , and define ˜ ℓ t ( Wϕ ( t )) for W ∈ L ( H , W ) . Then for any X,Y ∈ L ( H , W ) ,

<!-- formula-not-decoded -->

where ϕ ( t ) ⊗ ϕ ( t ) : H → H is the operator with action ( ϕ ( t ) ⊗ ϕ ( t )) h = ⟨ ϕ ( t ) , h ⟩ ϕ ( t )

<!-- formula-not-decoded -->

Proof. Let x = Xϕ ( t ) and y = Y ϕ ( t ) , and observe that by β -strong-convexity of ℓ t in W we have

<!-- formula-not-decoded -->

where ( ⋆ ) uses Lemma 8 and the last line observes that ∇ ℓ t ( x ) ⊗ ϕ ( t ) = ∇ ˜ ℓ t ( X ) .

Using this result it is straight-forward to see that the usual ONS arguments work in this setting. For instance, by running mirror descent with regularizer ψ t ( W ) = 1 2 ⟨ W Σ t , W ⟩ HS where Σ t = λI + β ∑ t s =1 ϕ ( t ) ⊗ ϕ ( t ) , we have the following regret guarantee.

Theorem 7. (K-ONS for Strongly-convex Losses) Let ℓ 1 , . . . , ℓ T be a sequence of β -strongly convex losses. Let λ &gt; 0 and for all t , define Σ t = λI + β ∑ t s =1 ϕ ( s ) ⊗ ϕ ( s ) and ∥ W ∥ 2 Σ t = ⟨ W Σ t , W ⟩ HS for W ∈ L ( H , W ) . Suppose that on each round A updates

<!-- formula-not-decoded -->

starting from W 1 = 0 ∈ L ( H , W ) . Then for any u 1 , . . . , u T in W and U ∈ L ( H , W ) satisfying u ( t ) = Uϕ ( t ) for all t , Algorithm 1 applied with A guarantees

<!-- formula-not-decoded -->

where K T = ( ⟨ ϕ ( s ) , ϕ ( t ) ⟩ H ) s,t ∈ [ T ] and d eff ( λ ) = Tr ( K T ( λI + K T ) -1 ) .

Proof. Applying Theorem 1 followed by Proposition 9, we have

<!-- formula-not-decoded -->

where ( a ) applies the standard bound for online mirror descent, ( b ) observes that

<!-- formula-not-decoded -->

and uses Fenchel-Young inequality to bound

<!-- formula-not-decoded -->

and ( c ) uses a mild generalization of the usual log-determinant lemma (Lemmas 6 and 7) and defines defines K T = ( ⟨ ϕ ( s ) , ϕ ( t ) ⟩ H ) s,t ∈ [ T ] .

Note that in the static regret setting, it is possible to avoid the dependence on the comparator norm entirely and pay only the logarithmic penalty-we do not expect such an improvement to be possible here since it would violate known Ω( P T ) lower bounds for strongly-convex losses [52].

## F.2 Additional Details for Example 3

In this section we provide some extra details showing that for the RKHS H associated with kernel k ( t, s ) = min( s, t ) , we can bound ∥ u ∥ 2 H = ∥∇ u ∥ 2 L 2 = O (√ ∑ t ∥ u t -u t -1 ∥ 2 W ) (Theorem 8) and d eff ( λ ) = O ( T/ √ λ ) (Theorem 9). We begin with the bound on the continuous squared path-lenth ∥∇ u ∥ 2 L 2 .

Theorem 8. Let H be the RKHS associated to the kernel k ( s, t ) = min( s, t ) on [0 , T ] . Then for any v 1 , . . . , v T ∈ R d there exists a function u ∈ H such that u ( t ) = v t for all t ∈ [ T ] and

<!-- formula-not-decoded -->

with C ≤ 5 4 .

Proof. We assume without loss of generality that v t ∈ R since the result extends immediately to R d via the coordinate-wise extension mentioned in Section 2. For brevity we will define v t = 0 for t / ∈ { 1 , . . . , T } so that we can write ∑ T t =1 | v t -v t -1 | 2 + | v 1 | 2 = ∑ t | v t -v t -1 | 2 .

Note that the RKHS associated with kernel k ( s, t ) = min( s, t ) is H = { f ∈ L 2 : f ′ ∈ L 2 , f (0) = 0 } , with associated norm ∥ f ∥ H = ∥∇ f ∥ L 2 = ∫ |∇ f ( x ) | 2 dx (see, e.g., Example 23 of Berlinet and Thomas-Agnan [6] with m = 1 ). Now suppose we define

<!-- formula-not-decoded -->

where sinc( x ) = sin( πx ) /πx . Then u and u ′ are square integrable and u (0) = 0 , so u ∈ H . Moreover, the norm associated with H is ∥ f ∥ 2 H = ∫ |∇ f ( x ) | 2 dx = ∥∇ f ∥ 2 L 2 , so we need only show that the constructed function u ( t ) has ∥∇ u ∥ 2 L 2 ≤ O ( ∑ t | v t -v t -1 | 2 ) .

Denote v ( t ) = ∑ i v i δ ( t -i ) = v t and observe that we can write u ( t ) = ∑ T i =1 v ( i )sinc( t -i ) = ( v⋆ sinc)( t ) , so using the fact that the Fourier transform of sinc is the rectangle function 1 [ -1 2 , 1 2 ] ( ω ) = I { ω ∈ [ -1 2 , 1 2 ] } (see, e.g., Kammler [28]), we have ̂ u ( ω ) = ̂ v ⋆ sinc( ω ) = ̂ v ( ω )1 [ -1 2 , 1 2 ] ( ω ) . Thus,

<!-- formula-not-decoded -->

via Parseval's identity. We proceed by relating ̂ v ( ω ) to the DFT of the difference sequence, ∆ v ( t ) = v t -v t -1 and then applying Parseval's inequality for sequences to get ∫ | ̂ ∆ v ( ω ) | 2 ≤ ∑ t | ∆ v ( t ) | 2 = ∑ t | v t -v t -1 | 2 .

Observe that the DFT of the difference sequence is

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

Now observe that using the identity 1 -cos( x ) = 2 sin 2 ( x/ 2) we have

<!-- formula-not-decoded -->

then, using ω 2 / sin 2 ( ω/ 2) ≤ 5 on [ -1 2 , 1 2 ] we have

<!-- formula-not-decoded -->

where the last line applies Parseval's identity for sequences.

Next, the following theorem shows that the effective dimension of the linear spline kernel is indeed O ( T/ √ λ ) .

Theorem 9. Let K T ∈ R T × T be the matrix with entries [ K T ] ij = min( i, j ) . Then

<!-- formula-not-decoded -->

Proof. By Lemma 12, the inverse of a matrix K T with entries [ K T ] ij = min( i, j ) is the tri-diagonal matrix of the form

<!-- formula-not-decoded -->

The eigenvalues of matrices of this form are well-known [17, 32, 43] and have a closed form expression:

<!-- formula-not-decoded -->

where the second equality uses the identity 1 -cos( x ) = 2sin 2 ( x/ 2) . Moreover, using the fact that sin( x ) is concave on [0 , π/ 2] we can bound sin( x ) ≥ 2 π x , so the eigenvalues of K -1 T can be bounded as

<!-- formula-not-decoded -->

Thus, via direct calculation of the effective dimension d eff ( λ ) = Tr ( K T ( λI + K T ) -1 ) = ∑ T k =1 λ k ( K T ) λ k ( K T )+ λ , we have

<!-- formula-not-decoded -->

where ( a ) makes a change of variables x = T/ √ λu , ( b ) uses the fact that ∫ b a 1 1+ u 2 du = arctan( x ) ∣ ∣ ∣ b a and ( c ) uses | arctan( x ) | ≤ π/ 2 for all x .

## G Proofs for Section 6 (Directional Adaptivity)

In this section we provide the full statement and proof of Proposition 5.

Proposition 10. Let ℓ 1 , . . . , ℓ T be a sequence of G -Lipschitz losses and for all t , let g t ∈ ∂ℓ t ( w t ) and define G t = g t ⊗ ϕ ( t ) ∈ L ( H , W ) , G 0 = G √ max t k ( t, t ) and Σ t = ( λ + G 2 0 ) I + ∑ t -1 s =1 G t ⊗ G t . 7 Let ( ∥·∥ t , ∥·∥ t, ∗ ) be a dual-norm pair characterized by ∥ W ∥ t = √ ⟨ W, Σ t W ⟩ HS . Let ϵ &gt; 0 , and for all t let V t = 4 G 2 0 + ∑ t -1 s =1 ∥ G t ∥ 2 t, ∗ , α t = ϵG 0 √ V t log 2 ( V t /G 2 0 ) and set ψ t ( W ) = k ∫ ∥ W ∥ t 0 min η ≤ 1 /G 0 [ ln( x/α t +1) η + ηV t ] dx .

Suppose on each round we set W t = arg min W ∈ L ( H , W ) 〈 ∑ t -1 s =1 G s , W 〉 + ψ t ( W ) and we play w t = W t ϕ ( t ) . Then for u 1 , . . . , u T in W and U ∈ L ( H , W ) satisfying u t = Uϕ ( t ) for all t ,

<!-- formula-not-decoded -->

Proof. The result follows as a special case of Theorem 10 with sequence of non-decreasing norms characterized by ∥ W ∥ t = √ ⟨ W, Σ t W ⟩ W and Lipschitz constant G 0 = G √ max t k ( t, t ) ≥ ∥ ∥ ∥ ∇ ˜ ℓ t ( W t ) ∥ ∥ ∥ for all t . First, observe that

<!-- formula-not-decoded -->

hence from the static regret guarantee of Theorem 10, we get

<!-- formula-not-decoded -->

Moreover, observing that

<!-- formula-not-decoded -->

we have via Lemma 7 that

<!-- formula-not-decoded -->

where K T is the gram matrix with entries [ K T ] ij = ⟨ g i , g j ⟩ k ( i, j ) and d eff ( λ ) = Tr ( K T ( λI + K T ) -1 ) = ∑ T k =1 λ k ( K T ) λ + λ k ( K T ) .

Hence the dynamic regret R T ( u 1 , . . . , u T ) = ˜ R T ( U ) can be bound above by

<!-- formula-not-decoded -->

7 Here, the tensor product G t ⊗ G t is the map such that for V ∈ L ( H , W ) , ( G t ⊗ G t )( V ) = ⟨ G t , V ⟩ HS G t ∈ L ( H , W ) . Note that Σ t is a self-adjoint operator.

## G.1 Directional Adaptivity via Varying-norms

For completeness we provide a mild generalization of the static regret algorithm of [23] to leverage an arbitrary sequence of increasing norms. A similar technique has been used to get full-matrix parameter-free rates by [13].

The analysis remains mostly the same as Jacobsen and Cutkosky [23], but their analysis of the stability term bounds -D ψ t ( w t +1 | w t ) via a lemma that assumes that ψ t ( w ) = Ψ t ( ∥ w ∥ 2 ) for w ∈ R d . To obtain a full-matrix version of their result, we would instead like to have Ψ t ( ∥ w ∥ M ) , where ∥·∥ M is a weighted norm w.r.t. to the inner product ⟨· , ·⟩ W on an arbitrary Hilbert space W . In what follows, we drop the dependence on W for brevity and simply write ⟨· , ·⟩ .

We first state and prove the main result of this section. The proof will rely on a few technical lemmas, which we state and prove at the end of the section in Appendix G.1.1.

Theorem 10. Let W be a Hilbert space and let ⟨· , ·⟩ denote the associated inner product, let ∥·∥ 1 , . . . , ∥·∥ T +1 be an arbitrary sequence of non-decreasing norms on W , and let ∥·∥ 0 := √ ⟨· , ·⟩ ≤ ∥·∥ t for all t . Let ℓ 1 , . . . , ℓ T be convex functions over W satisfying ∥ g t ∥ t, ∗ ≤ G for all t and g t ∈ ∂ℓ t ( w t ) . Let ϵ, λ &gt; 0 , V t = 4 G 2 + ∑ t -1 s =1 ∥ g s ∥ 2 s, ∗ , α t = ϵG √ V t log 2 ( V t /G 2 ) , and set ψ t ( w ) = 3 ∫ ∥ w ∥ t 0 min η ≤ 1 G [ ln( x/α t +1) η + ηV t ] dx , and on each round update w t +1 = arg min w ∈W 〈 ∑ t s =1 g s , w 〉 + ψ t +1 ( w ) . Then for all u ∈ W ,

<!-- formula-not-decoded -->

where ̂ O ( · ) hides constant and log(log) factors (but not log factors).

Proof. Begin by applying the standard FTRL regret template (see, e.g., Orabona [39, Lemma 7.1]):

<!-- formula-not-decoded -->

where F t ( w ) = 〈 ∑ t -1 s =1 g s , w 〉 + ψ t ( w ) . Observe that the summation can be written as

<!-- formula-not-decoded -->

where ( a ) uses the definition of Bregman divergence to write f ( a ) -f ( b ) = ⟨∇ f ( a ) , a -b ⟩ -D f ( b | a ) , ( b ) uses the fact that w t = arg min w ∈W F t ( w ) , hence ⟨∇ F t ( w t ) , w t -w t +1 ⟩ ≤ 0 by the first-order optimality condition, and ( c ) uses the fact that Bregman divergences are invariant to linear terms, so from the definition of F t we have D F t ( ·|· ) = D ψ t ( ·|· ) . Moreover, since ∥·∥ 1 , . . . , ∥·∥ T is a non-decreasing sequence of norms, we can bound the terms

<!-- formula-not-decoded -->

so overall the regret is bounded by

<!-- formula-not-decoded -->

From here, the rest of the proof follows using the same arguments as [23], but using our Lemma 5 to bound D ψ t ( w t +1 | w t ) ≥ 1 2 ∥ w t -w t +1 ∥ 2 Ψ t ( ∥ ˜ w ∥ t ) instead of their Lemma 7.

## G.1.1 A Stability Lemma for Weighted Norms

In this section generalize the stability lemma of Jacobsen and Cutkosky [23] to weighted norms ∥ x ∥ M = √ ⟨ x, Mx ⟩ . This is the main technical detail needed for the proof of Theorem 10 that is not covered by the proof of their static regret algorithm. Throughout this section we assume the domain W is a Hilbert space with associated inner product ⟨· , ·⟩ . The following helper lemma follows via a straight-forward but somewhat tedious computation.

Lemma 4. Let g : W → R be a convex function and let f ( x ) = √ g ( x ) . Then for x ∈ W s.t. g ( x ) &gt; 0 we have

<!-- formula-not-decoded -->

where ⊗ denotes the tensor product.

Using this, we have the following Hessian bounds for elliptically-symmetric functions:

Lemma 5. Let M ∈ L ( H , H ) be a positive definite linear operator and assume M is self-adjoint w.r.t. ⟨· , ·⟩ . Let ∥ x ∥ M = √ ⟨ x, Mx ⟩ be the weighted norm induced by M and let ψ ( w ) = Ψ( ∥ w ∥ M ) for some convex function Ψ : R → R . Then for any w ∈ W bounded away from 0 and any u ∈ W ,

<!-- formula-not-decoded -->

Moreover, if Ψ ′ ( · ) is concave and non-negative, then for any w ∈ W bounded away from 0 and u ∈ W ,

<!-- formula-not-decoded -->

Proof. The proof follows a similar argument to Orabona and P´ al [41, Lemma 23]. Let us first compute the gradients of f ( x ) = ∥ x ∥ M = √ ⟨ x, Mx ⟩ . Let g ( x ) = ⟨ x, Mx ⟩ and observe that if M is self-adjoint w.r.t. ⟨· , ·⟩ , we have ∇ g ( x ) = 2 Mx and ∇ 2 g ( x ) = 2 M . Hence, applying Lemma 4 we have

<!-- formula-not-decoded -->

Using this, we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Hence, for any u ∈ W we have

<!-- formula-not-decoded -->

Now decompose u = w + v for some v such that ⟨ Mw,v ⟩ = 0 ; such a v always exists for positive definite M . Then

<!-- formula-not-decoded -->

where the last step uses the fact that w and v are orthogonal w.r.t. M .

For the second statement of the lemma, we need only show that Ψ ′ ( ∥ w ∥ M ) / ∥ w ∥ M ≥ Ψ ′′ ( ∥ w ∥ M ) . This is indeed the case by concavity and non-negativity of Ψ ′ ( · ) :

<!-- formula-not-decoded -->

## H Supporting Lemmas

The following lemma is a straight-forward generalization of the usual log-determinant lemma (see, e.g. , [10, Lemma 11.11]), taking a bit of extra care to handle determinants of potentially infinitedimensional linear operators.

Lemma 6. Let H be a Hilbert space and for all t let v t ∈ H . Suppose v t ⊗ v t : H → H defines a bounded linear operator for all t and suppose A t = A t -1 + v t ⊗ v t for any t ≥ 1 , starting from A 0 = I . Then

<!-- formula-not-decoded -->

Proof. Observe that for any t , we have A t = A t -1 + v t ⊗ v t , so re-arranging terms, factoring, and taking determinants of both sides we have

<!-- formula-not-decoded -->

Note that each of these determinants are well-defined in terms of the Fredholm determinant: each of the three terms above is a trace-class perturbation of the identity operator. Moreover, observe that A -1 t ( v t ⊗ v t ) is a rank-one operator having single eigenvalue equal to λ = 〈 v t , A -1 t v t 〉 . Indeed, for any w ∈ H we have A -1 t ( v t ⊗ v t )( w ) = ⟨ v t , w ⟩ A -1 t v t , hence, for w = A -1 t v t we have

<!-- formula-not-decoded -->

Therefore, from the standard rank-one perturbation identity for the determinant we have Det ( I -A -1 t v t ⊗ v t ) = 1 -〈 v t , A -1 t v t 〉 , so re-arranging Equation (10) yields

<!-- formula-not-decoded -->

Lemma 7. Let H be a Hilbert space. For all t let G t ∈ H be a bounded linear operator and define S t = λI + ∑ t s =1 G s ⊗ G s for λ &gt; 0 . Then

<!-- formula-not-decoded -->

where K T = ( ⟨ g t , g s ⟩ k ( t, s )) t,s ∈ [ T ] and d eff ( λ ) = Tr ( K T ( λI + K T ) -1 ) .

Proof. First apply Lemma 6 with v t = G t / √ λ followed by the elementary inequality 1 -x ≤ ln (1 /x ) to get

<!-- formula-not-decoded -->

where the last line uses the well-known fact that the gram matrix K T = ( ⟨ g t , g s ⟩ k ( t, s )) t,s ∈ [ T ] and the empirical covariance operator ∑ T t =1 G t ⊗ G t have the same eigenvalues. Moreover, following J´ ez´ equel et al. [27, Proposition 2] we can use the inequality ln (1 + x ) ≤ x 1+ x (1 + ln (1 + x )) to expose a dependence on the effective dimension d eff ( λ ) = Tr ( K T ( λI + K T ) -1 ) as follows:

<!-- formula-not-decoded -->

Lemma 8. Let H be a separable RKHS with associated feature map ϕ ( t ) ∈ H and let x ∈ H satisfy x ( t ) = Xϕ ( t ) for some X ∈ L ( H , W ) . Then

<!-- formula-not-decoded -->

where ϕ ( t ) ⊗ ϕ ( t ) : H → H is the linear operator with action ( ϕ ( t ) ⊗ ϕ ( t )) h = ⟨ ϕ ( t ) , h ⟩ H ϕ ( t ) .

Proof. Let h 1 , h 2 , . . . be an orthonormal basis of H . By definition of the Hilbert-Schmidt inner product, we have

<!-- formula-not-decoded -->

where X ∗ : W → H is the adjoint of X and ( ⋆ ) uses Parseval's identity.

Lemma 9. Let W ⊆ R d and let H be an RKHS with associated feature map ϕ . For all t ∈ [ T ] , let G t = g t ⊗ ϕ ( t ) ∈ L ( H , W ) denote the rank-one operator mapping G t ( h ) = ⟨ ϕ ( t ) , h ⟩ g t ∈ W . Then for any t ,

<!-- formula-not-decoded -->

Proof. Let h 1 , h 2 , . . . be an orthonormal basis of H . Observe that for any h ∈ H , ( ∑ t s =1 G s ) ( h ) = ∑ t s =1 ⟨ ϕ ( s ) , h ⟩ g s . Hence, by definition of the Hilbert-Schmidt norm,

<!-- formula-not-decoded -->

where the last line observes that for orthonormal basis h i we have ∑ i ⟨ ϕ ( s ) , h i ⟩ H ⟨ ϕ ( s ′ ) , h i ⟩ H = ⟨ ϕ ( s ) , ϕ ( s ′ ) ⟩ H = k ( s, s ′ ) .

The following theorem shows how to compute the norm of G t = g t ⊗ ϕ ( t ) , which is the auxiliary loss for OLO under our framework. Here we state the result in terms of g t ∈ W ∗ for generality, but note that in the main text we implicitly invoke Riesz representation theorem to write g t ∈ W , G t ∈ L ( H , W ) , and ∥ G t ∥ = ∥ g t ∥ W √ k ( t, t ) .

Lemma 10. Let H be a RKHS with associated feature map ϕ ( t ) and let W be a Hilbert space. Let ℓ t : W → R be a differentiable function and for any W ∈ L ( H , W ) let ˜ ℓ t ( W ) = ℓ t ( Wϕ ( t )) . Then for any W ∈ L ( H , W ) , g t ∈ ∂ℓ t ( Wϕ ( t )) , and G t = g t ⊗ ϕ ( t ) ∈ ∂ ˜ ℓ t ( W ) ,

<!-- formula-not-decoded -->

where k ( s, t ) = ⟨ ϕ ( s ) , ϕ ( t ) ⟩ H is the kernel associated with H and ∥·∥ W , ∗ is the dual norm of ∥·∥ W .

Proof. We have via Lemma 11 that

<!-- formula-not-decoded -->

where g t ∈ ∂ℓ t ( Wϕ ( t )) ⊆ W ∗ . By Riesz representation theorem, we can identify a ̂ g t ∈ W such that for any w ∈ W , g t ( w ) = ⟨ ̂ g t , w ⟩ W , and likewise we can identify G t ∈ L ( H , W ) ∗ with a rank-one operator ̂ G t ∈ L ( H , W ) with action ̂ G t ( h ) = ⟨ ϕ ( t ) , h ⟩ H ̂ g t . Hence, we have by definition of the Hilbert-Schmidt norm that for any orthonormal basis { h i } i of H ,

<!-- formula-not-decoded -->

where the last line again uses Riesz representation theorem to write ∥ ̂ g t ∥ W = ∥ g t ∥ W , ∗ and then uses ∑ i ⟨ ϕ ( t ) , h i ⟩ 2 H = ∥ ϕ ( t ) ∥ 2 H by Parseval's identity. Moreover, since ϕ ( t ) are the features of an RKHS with kernel k , we have

<!-- formula-not-decoded -->

Lemma 11. Let H be a RKHS with feature map ϕ ( t ) ∈ H , and let W be a Hilbert space. Let ℓ t : W → R be a convex function and let ˜ ℓ t ( W ) = ℓ t ( Wϕ ( t )) for W ∈ L ( H , W ) . Then for any W ∈ L ( H , W ) and any g t ∈ ∂ℓ t ( Wϕ ( t )) ⊆ W ∗ ,

<!-- formula-not-decoded -->

where G t ∈ L ( H , W ) ∗ is the functional with action G t ( W ) = ⟨ g t , Wϕ ( t ) ⟩ W for all W ∈ L ( H , W ) .

Proof. Let W ∈ L ( H , W ) , w t = Wϕ ( t ) ∈ W , and let g t ∈ ∂ℓ t ( w t ) ⊆ W ∗ . Define G t = g t ⊗ ϕ ( t ) ∈ L ( H , W ) ∗ the functional on L ( H , W ) with action

<!-- formula-not-decoded -->

Now observe that for g t ∈ ∂ℓ t ( w t ) , for any w ∈ W we have

<!-- formula-not-decoded -->

hence for any V ∈ L ( H , W ) we can take w = V ϕ ( t ) to get

<!-- formula-not-decoded -->

that is,

<!-- formula-not-decoded -->

so G t = g t ⊗ ϕ ( t ) ∈ ∂ ˜ ℓ t ( W ) ⊆ L ( H , W ) ∗ .

For completeness, the following lemma provides the inverse of a matrix with entries K ij = min( i, j ) . A similar result can be seen in the proof of Jacobsen and Orabona [26, Lemma 4], where a variant of the matrix K appears as an intermediate calculation.

Lemma 12. Let K ∈ R T × T be a matrix with entries K i,j = min( i, j ) . Then K -1 is a tri-diagonal matrix of the form

<!-- formula-not-decoded -->

Proof. It can easily be checked that K has Cholesky decomposition K = U ⊤ U where U is the upper-triangular matrix of 1 ′ s . Hence, K -1 = U -1 ( U ⊤ ) -1 . Moreover, the inverse of U is the first-order finite-differences operator with entries

<!-- formula-not-decoded -->

Indeed, ( U Σ) ij = ∑ T k =1 U ik Σ kj = -U i,j -1 + U ij = 1 for i = j and zero otherwise. Computing K -1 = ΣΣ ⊤ yields the tri-diagonal matrix of the stated form.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The claims in the abstract and introduction accurately summarize our contributions: we present a reduction from dynamic regret minimization to static regret minimization in an RKHS. We design a class of function spaces which allow us to obtain optimal regret guarantees Section 4.1 and our result easily generalizes to scale-free updates using well-known clipping techniques (Appendix C.3). We show that our approach enables us to leverage loss curvature to obtain ∥ u ∥ 2 H + d eff ( λ ) log( T ) dynamic regret in Section 5. Finally, we show how to obtain directionally adaptive dynamic regret guarantees in Section 6.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the computational difficulties inherent to the RKHS approach approach in Section 4, and provide direction to well-known ways to address these issues from the RKHS literature. In Section 5 we discuss that we were unable to find a kernel with effective dimension which will enable the optimal T 1 / 3 P 2 / 3 T for the class of TV bounded sequences. We detail explicitly which specific classes of loss functions our results hold for (Lipschitz convex losses in Sections 4 and 6, exp-concave losses, strongly-convex, and regression losses in Section 5).

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

Answer: [Yes]

Justification: For each theoretical result, we explicitly state all necessary assumptions and provide complete proofs in the main text or appendix. The arguments are self-contained or discussed earlier in the relevant section.

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

Justification: We do not include experimental results in this work, as the current focus is on proving the theoretical guarantees and exploring potential applications.

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

Justification: Since this paper does not include experimental results, there is no data or code provided for reproduction.

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

Justification: As this paper does not include experimental results, there are no training or test details provided.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: As this paper does not include experimental results, no error bars or statistical significance are reported.

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

Justification: As the paper does not include experimental results, there is no information provided regarding computational resources.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in this paper fully conforms to the NeurIPS Code of Ethics. There are no ethical concerns related to data collection, experiments, or other aspects of the work, as it focuses purely on theoretical analysis.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This paper focuses on theoretical advancements in online learning algorithms and does not have direct societal applications. As such, it does not explicitly discuss potential positive or negative societal impacts. The work is foundational in nature and does not involve technologies that could be misused or present ethical concerns in its current form.

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

Justification: This paper focuses on theoretical algorithmic advancements in online learning and does not involve the release of models, data, or other resources that could pose risks for misuse. Therefore, no safeguards are necessary, as there is no high-risk data or models associated with the work.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: This paper does not use any external assets such as code, data, or models. All work presented is original and theoretical, and no third-party assets were incorporated.

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

Justification: This paper does not introduce any new assets such as datasets, code, or models. The work is purely theoretical, and no new assets were created or released as part of this research.

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

Justification: The paper does not involve crowdsourcing nor research with human subjects.

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