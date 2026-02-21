## New Perspectives on the Polyak Stepsize: Surrogate Functions and Negative Results

## Francesco Orabona

King Abdullah University of Science and Technology (KAUST) Thuwal, 23955-6900, Kingdom of Saudi Arabia francesco@orabona.com

## Ryan D'Orazio

Mila Québec AI Institute, Université de Montréal Montréal, QC, Canada ryan.dorazio @mila.quebec

## Abstract

The Polyak stepsize has been proven to be a fundamental stepsize in convex optimization, giving near optimal gradient descent rates across a wide range of assumptions. The universality of the Polyak stepsize has also inspired many stochastic variants, with theoretical guarantees and strong empirical performance. Despite the many theoretical results, our understanding of the convergence properties and shortcomings of the Polyak stepsize or its variants is both incomplete and fractured across different analyses. We propose a new, unified, and simple perspective for the Polyak stepsize and its variants as gradient descent on a surrogate loss. We show that each variant is equivalent to minimize a surrogate function with stepsizes that adapt to a guaranteed local curvature . Our general surrogate loss perspective is then used to provide a unified analysis of existing variants across different assumptions. Moreover, we show a number of negative results proving that the non-convergence results in some of the upper bounds is indeed real.

## 1 Introduction

The iterative optimization of complex functions forms a cornerstone of modern machine learning, scientific computing, and engineering. Among the most foundational first-order methods is gradient descent, which iteratively refines a solution by moving in the direction opposite to the function's gradient. A critical aspect of gradient descent is the selection of an appropriate stepsize (or learning rate), as it dictates both the speed of convergence and the stability of the algorithm. The wrong choice of the stepsizes can lead to slow convergence or, conversely, to divergence, making the tuning process a significant practical hurdle.

In this landscape of stepsize selection strategies, the Polyak stepsize, proposed by Polyak [36] stands out for its theoretical elegance and convergence properties. Starting from an arbitrary x 1 ∈ R d , the Polyak stepsize is defined as

̸

<!-- formula-not-decoded -->

where 0 = g t ∈ ∂f ( x t ) and f ⋆ = min x f ( x ) . If g t = 0 , then x t +1 = x t . This update rule can achieve linear convergence for strongly convex and smooth functions, O (1 /T ) rate for convex smooth functions, and O (1 / √ T ) rate for non-smooth convex ones. This is particularly interesting because all of these rates are achieved with a unique stepsize and without knowledge of smoothness

or curvature constants. In other words, this update rule is adaptive to the geometry of the functions to optimize.

Recently, the Polyak stepsize has seen a resurgence in the machine learning literature, with a plethora of variants. However, despite the big number of papers on this topic, one essential research question seems still to be missing: What makes the Polyak stepsize adaptive and when can it fail?

Contributions. This paper aims to provide a novel framework to understand the Polyak stepsize, providing a clear geometric explanation of its adaptivity. In particular, we show that the adaptivity is due to a simple but powerful observation: The Polyak stepsize minimizes a surrogate objective function that is always locally smooth. As for standard smooth functions, we will show that the knowledge of the local smoothness constant is enough to obtain the correct rates. In addition, the local smoothness will depend only on the gradient itself, removing the need to estimate it. Furthermore, we show that minimal curvature of the surrogate is inherited from the original function as well. We also use this framework to extend its core idea to a family of algorithms. Then, we will show a number of negative results when f ( x ⋆ ) is not known and for its use in the stochastic case. These negative results complete our understanding by showing that some non-vanishing terms in existing upper bounds are necessary.

## 2 Related Work

In the pionnering work of Ermol'ev [14] stepsizes of the form η t ∝ 1 / ∥ g t ∥ 2 were proposed for non-smooth optimization. Despite the many convergence guarantees enabled by Ermol'ev [14]'s framework, in Polyak [36] it is noted that linear convergence is not possible with such stepsizes. As an alternative, Polyak suggests the stepsize (1), which is shown to converge at favourable rates with non-smooth convex functions, and strongly convex and smooth functions. In fact, contrary to common belief, Polyak [36] was the first to show linear convergence with a rate comparable to gradient descent in the smooth and strongly convex case. Furthermore, the case where f ⋆ is estimated was also studied, showing convergence to a level set if f ⋆ is overestimated, and best-iterate convergence to a neighboorhood if it is underestimated. The Polyak stepsize has since been extended and studied across several applications and domains.

Non-smooth convex. In non-smooth convex optimization, several schemes have been developed to estimate f ⋆ on the fly [7, 6, 16, 41, 31, 27]. In the finite-sum case with interpolation, (1) and variants have been studied as an incremental subgradient method [31, 27].

Non-expansive operators. In the context of non-expansive operators, the update (1) has also been studied as a special case of the subgradient projector [2, 8, 10]; where it can be shown that subgradient descent with η t = ( f ( x ) -c ) + / ∥ g t ∥ 2 is a quasi-firmly non-expansive operator 1 if f is convex and c ≥ f ⋆ , and x → x ⋆ if f is continuous [2]. Moreover, in the finite-sum setting, interpolation can also be viewed as iterating different quasi-firmly non-expansive operators with a common fixed point. For example, applying a subgradient projector sequentially (i.e., cycling through the different component functions) in a way such that each function eventually gets visited guarantees that x t →{ f ( y ) ≤ c } [8, Example 5.9.7]. For a survey on this topic see Censor [9].

Deterministic. In modern optimization, (1) has been shown to achieve similar rates to gradient descent in various common assumptions (e.g., Lipschitz, smoothness and strong convexity) [22], and more recently with other assumptions such as weakly convex functions [12], ( L 0 , L 1 ) -smooth functions [42, 17], and directional smoothness [30].

Stochastic. The Polyak stepsize has also been extended to the stochastic case with emphasis on applications to machine learning [39, 5, 29, 38]. The ALI-G method [5] and SPS max [29] use stochastic estimates via the sampled function f ( x , ξ ) and its gradient ∇ f ( x , ξ ) to perform a Polyaklike stepsize. SPS max in addition uses inf x f ( x, ξ ) as an estimate to f ⋆ , and is shown to converge at fast rates without a neighbourhood under interpolation. Folowing SPS max many variants have been proposed for SGD: StoPS [24], DECSPS and SPS ℓ max [35], SPS + [15]. Beyond SGD, other extensions include: mirror descent [13], with preconditioning [1], with line-search [25], and with momentum [43, 40, 33].

1 An operator T is quasi-firmly non-expansive if for all fixed points x ⋆ , ∥ T ( x ) -x ⋆ ∥ 2 ≤ ∥ x -x ⋆ ∥ 2 -∥ T ( x ) -x ∥ 2 . Note that a quasi-firmly non-expansive operators are also referred to as cutters .

Neighbourhood of Convergence. For SPS max , Loizou et al. [29] proved that the suboptimality gap is only guaranteed to shrink up to a factor that depends on the loss themselves. As we will explain in Section 5.1, this is equivalent to the guarantees in online learning where the regret is proportional to the cumulative loss of the competitor, typically denoted by L ⋆ . Hence, these kind of guarantees are usually called in online learning L ⋆ bounds [see, e.g., 34, Section 4.2].

Surrogates. Gower et al. [19] show that the Polyak stepsize is equivalent to online gradient descent with fixed stepsize on a sequence of adversarially chosen self-bounded surrogate losses. Differently from our framework, the adversarial nature of their losses does not allow to show that the algorithm is minimizing a fixed function. In comparison, our surrogate approach considers a fixed surrogate loss with local smoothness, where the Polyak stepsize is chosen to be the inverse of the local smoothness.

Surprisingly enough, despite the adaptivity of the Polyak stepsize across various assumptions without modification, in previous literature there is no clear explanation why this is the case.

## 3 Definitions and Notation

We will use the following notation and definitions. All the norms in this paper are L2 norms and will be denoted by ∥ · ∥ . For a function f : R d → R , we define a subgradient of f in x ∈ R d as a vector g ∈ R d that satisfies f ( y ) ≥ f ( x ) + ⟨ g , y -x ⟩ , ∀ y ∈ R d . We denote the set of subgradients of f in x by ∂f ( x ) . For a differentiable function we have that ∂f ( x ) = {∇ f ( x ) } . A function f : V → R , differentiable in an open set containing V , is L -smooth w.r.t. ∥ · ∥ if f ( y ) ≤ f ( x ) + ⟨∇ f ( x ) , y -x ⟩ + L 2 ∥ x -y ∥ 2 for all x , y ∈ V .

Definition 1. We say that a function f has a s -sharp minimum in x ⋆ if

<!-- formula-not-decoded -->

Note that if f has a sharp minimum then it is not differentiable at x ∗ [37] and if the function is also convex and G -Lipschitz we immediately have G ≥ s .

<!-- formula-not-decoded -->

Definition 2. We say that a function f : R d → R is L -self-bounded if

It is known that L -smooth functions are L -self-bounded [see, e.g., Lemma 4 in 28], but this definition is strictly weaker because it does not assume differentiability.

## 4 Polyak Stepsize is Gradient Descent on a Surrogate Function

Let f be convex and x ⋆ ∈ argmin x f ( x ) . Consider the following function:

<!-- formula-not-decoded -->

Instead of viewing the Polyak stepsize (1) with respect to f we propose to view it equivalently as a subgradient method with respect to ϕ . By the chain rule of subgradients [2][Corollary 16.72], subgradient descent with the Polyak stepsize (1) is equivalent to subgradient descent on ϕ with stepsize η t = 1 ∥ g t ∥ 2 . This perspective may seem superfluous, howerever, we will show that 1 ∥ g t ∥ 2 is strongly related to a certain notion of local curvature of ϕ , local star upper curvature.

Definition 3 (Local star upper curvature (LSUC)) . We say that a function f with minimizer x ⋆ has λ y -local star upper curvature (LSUC) around y if there exists λ y &gt; 0 such that

<!-- formula-not-decoded -->

Note that if the function is LSUC everywhere, then it must be convex since we assume the existence of a subgradient. 2 It is also immediate to show that convex L -smooth functions are also L -LSUC. Indeed, for convex L -smooth functions we have that [32, Theorem 2.1.5]

<!-- formula-not-decoded -->

2 If g t is not a subgradient but a directional derivative then f would be guaranteed to be star-convex [26].

Figure 1: The function f ( x ) = | x +2 | + x 2 2 is non-smooth but is 2 -LSUC as demonstrated by the blue curve, f ( x ⋆ ) -⟨ g , x ⋆ -x ⟩ -1 / 4 ∥ g ∥ 2 , being larger than f ( x ) for all x and g ∈ ∂f ( x ) . Similarly, f is self-bounded but with the larger constant L = 9 .

<!-- image -->

So, it is enough to set x = x ⋆ to obtain the above definition. However, the inclusion is strict, because there exist functions that are not smooth and still satisfy the above definition. For example, as shown in Figure 1, one can easily verify that f ( x ) = | x +2 | + x 2 2 is 2 -LSUC and 9 -self-bounded but not differentiable x = -2 , hence it is not smooth.

Finally, if the star-upper-curvature holds globally, i.e., there exists 0 &lt; λ &lt; λ y for all y , then we can show that this condition is equivalent to the upper quadratic growth condition in Guille-Escuret et al. [21]. 3 This observation was first made by Goujaud et al. [18, Theorem 2.6], we include the precise statement and proof in the Appendix B.

The key observation in the next Theorem is that ϕ is always locally star upper curved, regardless of the curvature (or lack of it) of the function f . Moreover, it will inherit additional curvature from f . The proof can be found in Appendix A.

Theorem 1 (Curvature of the Polyak surrogate) . Let f ( x ) be convex and define x ⋆ ∈ argmin x f ( x ) . Define ϕ ( x ) = 1 2 ( f ( x ) -f ( x ⋆ )) 2 . Then, we have

- ϕ is ∥ g y ∥ 2 -LSUC around any y for any g y ∈ ∂f ( y ) .
- If f is s -sharp, then ϕ has s 2 -quadratic growth.
- If f has µ -quadratic growth and L -self bounded, then ϕ satisfies a local quadratic growth:

<!-- formula-not-decoded -->

This theorem tells us that, regardless of the curvature of f , we can always construct the function ϕ that is locally curved. It is well-known that for L -smooth functions one can use the stepsize η = 1 L and achieve a rate between O (1 /T ) and a linear one, depending on the presence of strong convexity. Here, we show a similar result: GD can use stepsizes that depend on the local star upper curvature in all cases. Note however, unlike GD with a constant stepsize and smoothness, we do not have a descent lemma with ϕ . Indeed this is not possible as it would guarantee a O (1 /T ) rate for the last iterate which was shown to be impossible by Goujaud et al. [18] for QG +( L ) functions (i.e., L-LUSC functions).

Lemma 1. Let ϕ convex and define x ⋆ ∈ argmin x ϕ ( x ) . Assume ϕ to be λ x -LSUC around any point x . Then, using subgradient descent with stepsizes η t = 1 λ x t guarantees

<!-- formula-not-decoded -->

Summing this inequality over time, we also have

<!-- formula-not-decoded -->

3 A function f satisfies the µ -quadratic growth condition if f ( x ) -f ( x ⋆ ) ≥ µ / 2 ∥ x -x ⋆ ∥ 2 .

<!-- formula-not-decoded -->

Proof. From the classic one-step analysis of GD [see, e.g., 34], we have

<!-- formula-not-decoded -->

for any g x t ∈ ∂ϕ ( x t ) . Now, we use the fact that ϕ is LSUC and the definition of η t we obtain the stated bound. Summing from t = 1 to T and discarding the negative term on the right hand side concludes the proof.

The above discussion can be summarized in the following theorem.

Theorem 2. The Polyak stepsize in (1) is equivalent to subgradient descent on the function ϕ in (2) , when using stepsizes η t equal to the inverse of the local-star-upper curvature of ϕ in x t .

This theorem and Lemma 1 do not give us a rate, however we can immediately observe that if ∑ t η t = + ∞ we also have ϕ ( x t ) → 0 , implying convergence of the last iterate (see Remark 6 for more details).

To obtain known rates for the Polyak stepsize, we can use additional assumptions on f . However, we want to stress that, differently from prior results, we explicitly get a convergence rate for the surrogate function ϕ , the function actually minimized by (1). The rates on the original function f are immediate by just taking the square root.

If f is G -Lipschitz, then ∑ T t =1 η t ≥ T/G 2 . Hence, the final rate is ϕ (¯ x T ) ≤ G 2 2 T ∥ x 1 -x ⋆ ∥ 2 . So, the surrogate loss converges as O (1 /T ) , as expected by a loss with upper curvature.

Now, instead let's assume that the function f is L -self-bounded. Using inequalities between harmonic and arithmetic means, we have

<!-- formula-not-decoded -->

where in the last inequality we use (4). This implies

<!-- formula-not-decoded -->

Similarly, it is equally easy to obtain rates for Hölder-smooth functions, see Theorem 7 for more details.

We can also assume that f is s -sharp and G -Lipschitz. So, from (3) and the fact that ϕ has s 2 quadratic growth from Theorem 1, then by Lemma 1 we have

<!-- formula-not-decoded -->

Using the fact that s 2 G 2 ≤ 1 , this immediately gives a linear convergence rate.

## 5 Generalizing the Polyak Stepsize: More Surrogates and Stochastic Setting

We have shown how the Polyak stepsize is just GD on a particular function with stepsizes adapted to the local curvature of the function. In this section, we show that we can construct an entire family of surrogate losses with similar guarantees, while also preparing ourselves for the stochastic setting.

Instead of the function (2), we consider more generally the surrogate

<!-- formula-not-decoded -->

where h : R d → R ≥ 0 is convex. As a special case we can recover (2) with h ( x ) = f ( x ) -f ⋆ , however, in general we do not need to know f ⋆ . For example we can take h = ( f ( x ) -a ) + for any a . Intuitively, the role of h is to transform f into a positive function. We show that ψ generally has an approximate local-star-upper curvature, where the approximation stems from h potentially being strictly positive in x ⋆ .

Definition 4 (Approximate local-star-upper curvature) . We will say that a function f with minimizer x ⋆ has ϵ -approximate λ y -star-upper-curvature around y if there exists ϵ such that

<!-- formula-not-decoded -->

Since we do not make explicit assumptions h with respect to f , we can only hope to achieve convergence to the minimum of h or ψ . So, from here onward we denote x ⋆ as as minizer of ψ .

Lemma 2. Let h : R d → R ≥ 0 be convex. Define ψ = 1 2 h 2 . Then, ψ is (2 √ ψ ( x ) ψ ( x ⋆ ) -ψ ( x ⋆ )) -approximate ∥ g ∥ -LSUC for any g ∈ ∂h ( x ) .

Proof. Given that the function ψ ( x ) might not be differentiable, we have to be careful in the calculation of its subgradients. We have

<!-- formula-not-decoded -->

where g ∈ ∂h ( x ) and the first inequality is due to the fact that 1 2 ( · ) 2 is a convex function. Hence, we see that ˜ g := h ( x ) g is a subgradient of ψ in x . Hence, for any u ∈ R d , we have

<!-- formula-not-decoded -->

where the inequality is due to the convexity of h and the fact that h ( x ) ≥ 0 . Setting u = x ⋆ , we have the stated bound.

With approximate local curvature, a generalization of Lemma 1 is immediate.

Lemma 3. Assume ψ : R d → R to be ϵ t -approximately λ t -star-upper-curve around x t . Then, for any η t &gt; 0 , any g t ∈ ∂ψ ( x t ) , and x t +1 = x t -η t g t we have

<!-- formula-not-decoded -->

̸

The last two lemmas tell us that the properties of the surrogate functions breaks if ψ ( x ⋆ ) = 0 . Hence, we will not be able to prove convergence results, but only that, for example, the suboptimality gap will converge up to a floor that depends on ψ ( x ⋆ ) . However, in Section 6 we will show that this is not an artifact of the proof. Indeed, we can construct simple one-dimensional functions where the generalized Polyak stepsize does not converge.

## 5.1 Stochastic Approximation Setting

Consider now the case that we are minimizing F ( x ) := E ξ ∼ D [ f ( x , ξ )] , where f : R d ×S → R , that covers both the stochastic approximation and finite-sum settings. We do not know the distribution D , but we assume that we can sample ξ i.i.d. from D .

In this setting, we argue that the Polyak stepsize makes sense only in restricted settings. In fact, the interpretation of the Polyak stepsize as minimizing a surrogate function implies that in the stochastic setting we will minimize the function E ξ ∼ D [ 1 2 h 2 ( x , ξ )] , where the function h ( · , ξ ) depends on the particular variant of the stochastic Polyak stepsize. It is clear that in general argmin x E ξ ∼ D [ f ( x , ξ )] can be completely different from argmin x E ξ ∼ D [ 1 2 h 2 ( x , ξ )] .

Here, starting from ALI-G [5] and SPS max [29] that use the idea of limiting the stepsizes, we propose a generalized Polyak stepsize algorithm. The proof is in Appendix C.

## Algorithm 1 Generalized Polyak Stepsize

Require: h : R d ×S → R , x 1 ∈ R d

- 1: for t = 1 , . . . , T do
- 2: Sample ξ t from D
- 3: Tranform f ( x , ξ t ) into h ( x , ξ t )
- 5: if g t = 0 then
- 4: Receive g t ∈ ∂h ( x , ξ t )

<!-- formula-not-decoded -->

̸

- 7: else
- 8: x t +1 = x t
- 9: end if
- 10: end for

Theorem 3. Let h : R d × S → R ≥ 0 be convex in its first argument. Denote by H ( x ) = E ξ ∼ D [ h ( x , ξ )] . Then, setting η t = min ( 1 ∥ g t ∥ 2 , γ h ( x t , ξ t ) ) in Algorithm 1, we have

- If h ( · , ξ t ) is L -self bounded, we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

- If h ( · , ξ t ) is G -Lipschitz, then we have

<!-- formula-not-decoded -->

- If h ( · , ξ ) is L -self-bounded and H ( x ) has µ -quadratic growth, then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Choosing the function h , the above theorem covers and extends a number of results in previous papers, for example:

- SPS max [29]: h ( x , ξ ) = f ( x , ξ ) -inf x f ( x , ξ ) , so H ( x ⋆ ) = E [ f ( x ⋆ , ξ ) -inf x f ( x , ξ )] .
- SPS + [15]: h ( x , ξ ) = ( f ( x , ξ ) -f ( x ⋆ , ξ )) + . In this case, H ( x ⋆ ) = 0 so we can also safely set γ = ∞ . Moreover, H ( x ) ≥ F ( x ) -F ( x ⋆ ) , hence any bound on H ( x ) translates to a bound on the suboptimality gap.
- SPS ℓ max [35]: h ( x , ξ ) = f ( x , ξ ) -q ( ξ ) , where q ( ξ ) is a lower bound to inf x f ( x , ξ ) . In this case, H ( x ⋆ ) = E [ f ( x ⋆ , ξ ) -q ( ξ )] .

Remark 1. If h ( x ⋆ , ξ t ) = 0 for all t , then ∥ x t +1 -x ⋆ ∥ ≤ ∥ x t -x ⋆ ∥ . Hence, in this case we only need to consider all the properties of h in the bounded domain { x ∈ R d : ∥ x -x ⋆ ∥ ≤ ∥ x 1 -x ⋆ ∥} . This is well-known via the subgradient projector perspective, as x t is guaranteed to approach the set ⋂ ξ { x : h ( x , ξ ) = 0 } at each iteration if it is non-empty. This property was observed in Gower et al. [20] for SPS + [15], but here it holds more generally. For example, h ( x , ξ ) = ( f ( x , ξ ) -a ) + , where a ≥ sup ξ f ( x ⋆ , ξ ) , would also have no neighbourhood of convergence and satisfies the assumption of the theorem.

Remark 2. The above theorem also applies to the case where some of the f ( · , ξ ) are non-convex functions, while still guaranteeing the convergence to the global optimum of F . For example, consider F ( x ) = 0 . 5 f 1 ( x ) + 0 . 5 f 2 ( x ) , where f 1 = -| x | (non-convex) and f 2 = 2 | x | . We have that F ( x ) = 1 2 | x | so x ⋆ = 0 . Now, choose h 1 ( x ) = max( f 1 ( x ) -f 1 ( x ⋆ ) , 0) = 0 and h 2 ( x ) = max( f 2 ( x ) -f 2 ( x ⋆ ) , 0) = 2 | x | . Hence, the hypotheses of the theorem are verified. Moreover, H ( x ) = 0 . 5 h 1 ( x ) + 0 . 5 h 2 ( x ) ≥ F ( x ) and H ( x ⋆ ) = 0 , so the theorem implies a convergence rate for the minimization of F ( x ) -F ( x ⋆ ) by using SPS + .

Remark 3. In the proof of Theorem 3, if one stops before taking expectations, one obtains a regret guarantee on a sequence of arbitrary losses h ( x , ξ t ) . Such regret scales as the sum of the loss in x ⋆ . This is exactly the L ⋆ bound that we mentioned in Section 2. Indeed, this kind of updates and guarantees were already obtained in the online learning literature for the special case of linear predictors by the Passive-Aggressive family of algorithms [11].

Besides covering a number of previous algorithmic variants, we also extend the previous known guarantees. In particular, Loizou et al. [29] only studied SPS in the non-smooth setting but did not include SPS max . 4 Theorem 3 shows for the first time that SPS max is adaptive to the entire range of upper curvature of the function, from Lipschitz to smooth functions. In Appendix D we also show additional results. Moreover, the second result in the smooth case is new, and it allows to recover the SGD guarantee on H when γ is sufficiently small. For example, we include a precise statement for SPS + when f ( x , ξ ) is L -self-bounded, that recovers the guarantee in Gower et al. [20, Corollary 2.3].

## 6 Neighbourhood of Convergence and Instability of the Polyak Stepsize

In Section 5 we demonstrate that a generalized version of the Polyak stepsize and existing variants can be viewed as GD on a function with approximate local curvature, with convergence to a neighbourhood of the optimal solution. This neighbourhood of convergence appears in our analysis just like in all existing variants, therefore suggesting it is unavoidable if

<!-- formula-not-decoded -->

In this section, we demonstrate that this neighbourhood of convergence is not an artifact of the analysis and indeed cannot be avoided in general. We also show that the positivity condition (5) can fundamentally change the dynamics of Algorithm 1, even in the deterministic setting, thus posing a challenge that is not just associated with interpolation.

Condition (5) occurs with SPS [29] without interpolation, or in the deterministic setting when the optimal value is underestimated, h ( x ) = f ( x ) -c where f ⋆ &gt; c . Convergence under this condition was first studied in the deterministic case in Polyak's original paper [36], where it is shown that if inf x h ( x ) = h ⋆ &gt; 0 then lim t →∞ min 1 ≤ s ≤ t h ( x t ) -h ⋆ ≤ h ⋆ . That is, the best iterate eventually enters a neighbourhood of the minima where the size of the neighbourhood is dependent on how much h ⋆ is underestimated by 0. In the stochastic case, convergence of the average iterate to a neighbourhood when understimating the minimum was also studied by Orvieto et al. [35] under SPS ℓ max . However, we demonstrate the consequence of condition (5) is far greater than existing results suggest, with instability of fixed points, potential cycles, lower bounds in the sub-optimality gap, and lack of convergence regardless of initialization.

Deterministic Setting. We first demonstrate that in the deterministic setting, for different classes of h , if h ⋆ = min x h ( x ) &gt; 0 , then the fixed points of

<!-- formula-not-decoded -->

are unstable. Intuitively, this can be explained via our surrogate function view: In fact, denoting the local curvature constant of the surrogate 1 2 ( h ( x ) -h ⋆ ) 2 around x t as λ t , we see that the stepsize η t in update (6) can be equivalently written as

<!-- formula-not-decoded -->

4 Although Loizou et al. [29] state that SPS analysis can be readily extended to SPS max this does not seem to be the case due to the non-convexity of the min function, as demonstrated by our different proof technique in Appendix C.

<!-- image -->

t

Figure 2: Trajectories under T (6) for h ( x ) = x 2 2 + a with an unstable fixed point at x ⋆ = 0 . Lack of convergence is observed for different values of a as predicted by Proposition 3.

If h is Lipschitz or self-bounded then η t → + ∞ as x t → x ⋆ . Therefore, if h possesses curvature, then x t +1 may move further away from x ⋆ within a neighborhood of x ⋆ . Indeed, in Proposition 1 we show that for all self-bounded functions with a quadratic growth condition, the fixed point of T in (6) is unstable. A similar result can also be shown if h is L -Lipschitz and has a sharp mininum (see Proposition 6 in the Appendix).

̸

Proposition 1 (Unstable fixed point) . Suppose h is convex, strictly positive, L -self-bounded, and satisfies the quadratic growth condition h ( x ) -h ⋆ ≥ µ 2 ∥ x -x ⋆ ∥ 2 , where x ⋆ = arg min x h ( x ) is the only fixed point of T , defined in (6) . Then, for any point x ∈ S = { y : y = x ⋆ , h ( y ) -h ⋆ &lt; h ⋆ µ 8 L -µ } we have

<!-- formula-not-decoded -->

Note that this reinforces the need to clip the stepsize as proposed in ALI-G and SPS max . However, clipping will not remove this behaviour unless the maximum value is taken to be small enough. In Proposition 8 we show there is always a subregion where the stepsize is bounded, and this subregion can be made arbitrarily large within the unstable region; therefore, if the clipped value is too large instability is unavoidable.

The importance of h &gt; 0 in update (6) has also been studied in Bauschke et al. [3], where they demonstrate with examples that T can fail to be quasi-firmly non-expansive if h ⋆ &gt; 0 . Propositions 1, and 6 provide extra insight on this phenomenon as they automatically prove T cannot be quasi-firmly non-expansive and therefore we have the following remark.

Remark 4. If h &gt; 0 , and either of the following conditions hold:

- h is convex, self-bounded, and satisfies the quadratic growth condition,
- h is convex, Lipschitz, and has a sharp minimum,

then T from (6) is not quasi-firmly non-expansive.

While Propositions 1 and 6 establish that minima can be unstable, this property may not fully describe the dynamics of update (6). In fact, instability can admit convergence in the average iterate or last iterate if the local critical neighborhood is skipped. So, in Proposition 2 we provide an example of a function h , which satisfies the assumptions of Proposition 1, where the iterates cycle and never reach the minimum in best iterate or on average .

Proposition 2 (Cycling and failure to converge) . There exists a strictly positive smooth and strongly convex function h , and an initial point x 1 such that iterates from update (6) cycle and satisfy the inequality h ( 1 t ∑ t i =1 x i ) -h ⋆ ≥ δ &gt; 0 for all t .

Note that since the cycle in Proposition 2 consists of a finite number of points, clipping will not necessarily remove this behaviour (e.g. if the clipped value is taken to be larger than any of the seen stepsizes). In Proposition 2, a specific initialization was chosen to construct a cycle that would not converge. However, in Proposition 3 we show that for 1-d quadratics the lack of convergence is true for all initializations and values of h ⋆ up to a set of measure zero .

Proposition 3 (The set of good initializations can have measure zero) . Let h : R → R , defined as h ( x ) = x 2 2 + a for a &gt; 0 , where x t +1 = x t -h ( x t ) ∥∇ h ( x t ) ∥ 2 ∇ h ( x t ) , and x 1 is randomly initialized. Then, P { lim t →∞ x t = x ⋆ } = 0 . In other words, the set of initializations that can converge to the optimal solution has measure zero.

̸

Proof. Note h is 1-smooth and 1-strongly convex and therefore satisfies the conditions of Proposition 1 with µ = L = 1 and an unstable unique fixed point. Let T be such that x t +1 = T ( x t ) . T ( x ) = x ( 1 2 -a x 2 ) if x = 0 and 0 otherwise. With inverse T -1 ( S ) = { x ± √ x 2 +2 a : x ∈ S } . Therefore, T -k ( { x ⋆ } ) has at most 2 k points which has measure zero for all k . By Lemma 5 in Appendix E, the result follows.

Remark 5. Note that, by Lemma 5, Proposition 3 can be extended much more generally if T from (6) is shown to satisfy the Lusin ( N -1 ) property (see Definition 7) [23, Definition 4.12].

In the stochastic case with SPS , condition (5) is due to lack of interpolation, and Orvieto et al. [35] show that it can change the expected fixed point. In contrast, in the deterministic setting we have shown lack of convergence despite the fixed point being x ⋆ . Therefore the issue here stems from the instability of the method due to the underestimation of h ⋆ and not the bias of the expected update.

Stochastic Setting. In the stochastic setting, the positivity condition (5) can occur despite min x h ( x , ξ ) = 0 , such as in SPS ( h ( x , ξ ) = f ( x ) -min x f ( x , ξ ) ) without interpolation. Orvieto et al. [35] demonstrate that without interpolation SPS can fail to converge in a 1-d quadratic and has an expected fixed point different than min x F ( x ) = min x E ξ ∼ D [ f ( x , ξ )] . Similarly to the deterministic setting, we can show that SPS can have a random walk between a finite number of points.

Proposition 4 (Failure to converge) . There exist f 1 and f 2 quadratic 1-d functions and a starting point x 1 such that SPS on F ( x ) = 0 . 5( f 1 ( x ) + f 2 ( x )) satisfies

<!-- formula-not-decoded -->

Proof. Let f 1 = x 2 + 2 x + 5 and f 2 ( x ) = 2 x 2 -4 x + 10 . Let's start from x 1 = 1 where F ( x 1 ) = 8 . If we draw f 1 , x 2 = -1 , while if we draw f 2 then x 2 = 1 because f ′ 2 (1) = 0 . Hence, E [ F ( x 2 )] = 0 . 5 f 1 (1)+ f 2 (1) 2 +0 . 5 f 1 ( -1)+ f 2 ( -1) 2 = 9 . Iterating, we have that x 3 has equal probability to be equal to 1 and -1. Hence, again we have E [ F ( x 3 )] = 9 . So, we have that this holds for any t . Moreover, we have that min x F ( x ) = 44 / 6 .

## 7 Discussion and Limitations

We have shown that the design, properties, and failure of the (variants) of the Polyak stepsize can be easily derived through the lens of the minimization of a surrogate objective function. This framework also provides a new and natural explanation on the adaptivity of the stepsize via the local curvature of the surrogate. We believe this framework has the promise to design new variants, by simply designing surrogate functions with the required properties. Furthermore, with our perspective we have provided new insight on the challenge of controlling neighbourhoods of convergence that often appear in variants of the Polyak stepsize. We demonstrate that this neighbourhood is unavoidable and a fundamental issue causing instability. Moreover, we show that this issue is not due to the lack of interpolation, as commonly believed, but instead because the minimum of the surrogate loss is not zero more generally.

The limitations of our framework include the assumption of convex h in the generalized surrogate that must be assumed apriori. It is unclear if this framework can be extended to the more general case of noncovex surrogate functions. The class of such surrogates that admit fast rates and tight neighbourhoods of convergence remains an open question that we leave to future work.

## Acknowledgments

We acknowledge the use of Gemini 2.5 in developing the proof of Proposition 2. We also thank Mehdi Inane Ahmed for helpful discussions. Ryan D'Orazio's work is funded by Ioannis Mitliagkas' CIFAR chair.

## References

- [1] Farshed Abdukhakimov, Chulu Xiang, Dmitry Kamzolov, and Martin Takáˇ c. Stochastic gradient descent with preconditioned Polyak step-size. Computational Mathematics and Mathematical Physics , 64(4):621-634, 2024.
- [2] Heinz H Bauschke and Patrick L Combettes. Convex Analysis and Monotone Operator Theory in Hilbert Spaces . Springer, 2017.
- [3] Heinz H Bauschke, Caifang Wang, Xianfu Wang, and Jia Xu. Subgradient projectors: extensions, theory, and characterizations. Set-Valued and Variational Analysis , 26:1009-1078, 2018.
- [4] Amir Beck. First-order methods in optimization . SIAM, 2017.
- [5] Leonard Berrada, Andrew Zisserman, and M. Pawan Kumar. Training neural networks for and by interpolation. In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning , volume 119 of Proceedings of Machine Learning Research , pages 799-809. PMLR, 13-18 Jul 2020.
- [6] Dimitri Bertsekas. Nonlinear Programming , volume 2. Athena Scientific, 1999.
- [7] Ulf Brännlund. On relaxation methods for nonsmooth convex optimization, 1993.
- [8] Andrzej Cegielski. Iterative methods for fixed point problems in Hilbert spaces , volume 2057. Springer, 2012.
- [9] Yair Censor. Iterative methods for the convex feasibility problem. In M. Rosenfeld and J. Zaks, editors, Annals of Discrete Mathematics (20): Convexity and Graph Theory , volume 87 of North-Holland Mathematics Studies , pages 83-91. North-Holland, 1984.
- [10] Patrick L. Combettes. Fejér monotonicity in convex optimization , pages 1016-1024. Springer US, Boston, MA, 2009.
- [11] Koby Crammer, Ofer Dekel, Joseph Keshet, Shai Shalev-Shwartz, and Yoram Singer. Online passive-aggressive algorithms. Journal of Machine Learning Research , 7(19):551-585, 2006.
- [12] Damek Davis, Dmitriy Drusvyatskiy, Kellie J MacPhee, and Courtney Paquette. Subgradient methods for sharp weakly convex functions. Journal of Optimization Theory and Applications , 179(3):962-982, 2018.
- [13] Ryan D'Orazio, Nicolas Loizou, Issam H. Laradji, and Ioannis Mitliagkas. Stochastic mirror descent: Convergence analysis and adaptive variants via the mirror stochastic Polyak stepsize. Transactions on Machine Learning Research , 2023. ISSN 2835-8856.
- [14] Yu. M. Ermol'ev. Methods of solution of nonlinear extremal problems. Cybernetics , 2(4):1-14, July 1966. ISSN 1573-8337. doi: 10.1007/BF01071403.
- [15] Guillaume Garrigos, Robert M. Gower, and Fabian Schaipp. Function value learning: Adaptive learning rates based on the Polyak stepsize and function splitting in ERM. arXiv preprint arXiv:2307.14528 , 2023.
- [16] Jean-Louis Goffin and Krzysztof C. Kiwiel. Convergence of a simple subgradient level method. Mathematical Programming , 85(1):207-211, May 1999. ISSN 1436-4646. doi: 10.1007/ s101070050053.
- [17] Eduard Gorbunov, Nazarii Tupitsa, Sayantan Choudhury, Alen Aliev, Peter Richtárik, Samuel Horváth, and Martin Takáˇ c. Methods for convex ( L 0 , L 1 ) -smooth optimization: Clipping, acceleration, and adaptivity. In The Thirteenth International Conference on Learning Representations , 2025.
- [18] Baptiste Goujaud, Adrien Taylor, and Aymeric Dieuleveut. Optimal first-order methods for convex functions with a quadratic upper bound. arXiv preprint arXiv:2205.15033 , 2022.
- [19] Robert M Gower, Aaron Defazio, and Michael Rabbat. Stochastic Polyak stepsize with a moving target. arXiv preprint arXiv:2106.11851 , 2021.

- [20] Robert M. Gower, Guillaume Garrigos, Nicolas Loizou, Dimitris Oikonomou, Konstantin Mishchenko, and Fabian Schaipp. Analysis of an idealized stochastic Polyak method and its application to black-box model distillation. arXiv preprint arXiv:2504.01898 , 2025.
- [21] Charles Guille-Escuret, Manuela Girotti, Baptiste Goujaud, and Ioannis Mitliagkas. A study of condition numbers for first-order optimization. In Arindam Banerjee and Kenji Fukumizu, editors, Proceedings of The 24th International Conference on Artificial Intelligence and Statistics , volume 130 of Proceedings of Machine Learning Research , pages 1261-1269. PMLR, 13-15 Apr 2021.
- [22] Elad Hazan and Sham Kakade. Revisiting the Polyak step size. arXiv preprint arXiv:1905.00313 , 2022.
- [23] Stanislav Hencl and Pekka Koskela. Lectures on mappings of finite distortion , volume 2096. Springer, 2014.
- [24] Samuel Horváth, Konstantin Mishchenko, and Peter Richtárik. Adaptive learning rates for faster stochastic gradient methods. arXiv preprint arXiv:2208.05287 , 2022.
- [25] Xiaowen Jiang and Sebastian U Stich. Adaptive SGD with Polyak stepsize and line-search: Robust convergence and variance reduction. In Thirty-seventh Conference on Neural Information Processing Systems , 2023.
- [26] Pooria Joulani, András György, and Csaba Szepesvári. A modular analysis of adaptive (non)convex optimization: Optimism, composite objectives, variance reduction, and variational bounds. Theoretical Computer Science , 808:108-138, 2020. ISSN 0304-3975. Special Issue on Algorithmic Learning Theory.
- [27] Krzysztof C. Kiwiel. Convergence of approximate and incremental subgradient methods for convex optimization. SIAM Journal on Optimization , 14(3):807-840, 2004.
- [28] Xiaoyu Li and Francesco Orabona. On the convergence of stochastic gradient descent with adaptive stepsizes. In Kamalika Chaudhuri and Masashi Sugiyama, editors, Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics , volume 89 of Proceedings of Machine Learning Research , pages 983-992. PMLR, 16-18 Apr 2019.
- [29] Nicolas Loizou, Sharan Vaswani, Issam Hadj Laradji, and Simon Lacoste-Julien. Stochastic Polyak step-size for SGD: An adaptive learning rate for fast convergence. In Arindam Banerjee and Kenji Fukumizu, editors, Proceedings of The 24th International Conference on Artificial Intelligence and Statistics , volume 130 of Proceedings of Machine Learning Research , pages 1306-1314. PMLR, 13-15 Apr 2021.
- [30] Aaron Mishkin, Ahmed Khaled, Yuanhao Wang, Aaron Defazio, and Robert M. Gower. Directional smoothness and gradient methods: Convergence and adaptivity. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [31] Angelia Nedic and Dimitri P. Bertsekas. Incremental subgradient methods for nondifferentiable optimization. SIAM Journal on Optimization , 12(1):109-138, 2001.
- [32] Yurii Nesterov. Introductory lectures on convex optimization: A basic course , volume 87. Springer, 2004.
- [33] Dimitris Oikonomou and Nicolas Loizou. Stochastic Polyak step-sizes and momentum: Convergence guarantees and practical performance. In The Thirteenth International Conference on Learning Representations , 2025.
- [34] Francesco Orabona. A modern introduction to online learning. arXiv preprint arXiv:1912.13213 , 2019. Version 7.
- [35] Antonio Orvieto, Simon Lacoste-Julien, and Nicolas Loizou. Dynamics of SGD with stochastic Polyak stepsizes: Truly adaptive variants and convergence to exact solution. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems , volume 35, pages 26943-26954. Curran Associates, Inc., 2022.

- [36] Boris Teodorovich Polyak. Minimization of unsmooth functionals. USSR Computational Mathematics and Mathematical Physics , 9(3):14-29, 1969.
- [37] Boris Teodorovich Polyak. Introduction to Optimization . Translations Series in Mathematics and Engineering. Optimization Software, Inc., 1987.
- [38] Mariana Prazeres and Adam M. Oberman. Stochastic gradient descent with Polyak's learning rate. Journal of Scientific Computing , 89(1):25, Sep 2021.
- [39] Michal Rolinek and Georg Martius. L4: Practical loss-based stepsize adaptation for deep learning. Advances in neural information processing systems , 31, 2018.
- [40] Fabian Schaipp, Ruben Ohana, Michael Eickenberg, Aaron Defazio, and Robert M Gower. MoMo: Momentum models for adaptive learning rates. In International Conference on Machine Learning , pages 43542-43570. PMLR, 2024.
- [41] Hanif D Sherali, Gyunghyun Choi, and Cihan H Tuncbilek. A variable target value method for nondifferentiable optimization. Operations Research Letters , 26(1):1-8, 2000.
- [42] Yuki Takezawa, Han Bao, Ryoma Sato, Kenta Niwa, and Makoto Yamada. Parameter-free clipped gradient descent meets Polyak. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [43] Xiaoyu Wang, Mikael Johansson, and Tong Zhang. Generalized Polyak step size for first order optimization with momentum. In International Conference on Machine Learning , pages 35836-35863. PMLR, 2023.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: To the best of our knowledge our framework in Section 5 that both generalizes and analyzes the Polyak stepsize is novel. Additionally, we have included novel negative results in Section 6.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: We discuss the limitations of our framework and assumptions in Section 7. Guidelines:

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

Justification: All theoretical statements outline assumptions used in their respective proof. Complete proofs are either presented directly in the main body or in the appendix. We do not provide proof sketches for proofs not in the main body but do provide intuition and discussion in such cases.

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

Justification: We do not include any experiments in our paper.

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

Justification: We do not include any experiments in our paper.

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

Justification: We do not include any experiments in our paper.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: We do not include any experiments in our paper.

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

Justification: We do not include any experiments in our paper.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: As our paper is theoretical we do not have any concerns regarding: research with human subjects, and data concerns. Nor do we foresee any societal impact.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Due to the fundamental research of our paper we do not foresee broader societal impacts as per the guidelines below.

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

Justification: We do not use nor provide any models or data.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: No assets of the kind are used.

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

Justification: We do not use or release any such assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: No human subjects are used.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: No human subjects are used.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: An LLM was used to help derive result but not our core results or analyses.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Proofs for the Surrogate ϕ

Theorem 1 (Curvature of the Polyak surrogate) . Let f ( x ) be convex and define x ⋆ ∈ argmin x f ( x ) . Define ϕ ( x ) = 1 2 ( f ( x ) -f ( x ⋆ )) 2 . Then, we have

- ϕ is ∥ g y ∥ 2 -LSUC around any y for any g y ∈ ∂f ( y ) .
- If f is s -sharp, then ϕ has s 2 -quadratic growth.
- If f has µ -quadratic growth and L -self bounded, then ϕ satisfies a local quadratic growth:

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

where the inequality is due to the convexity of f and the fact that f ( y ) -f ( x ⋆ ) ≥ 0 . For the second property, we have

<!-- formula-not-decoded -->

For the third property we have

<!-- formula-not-decoded -->

Remark 6. Convergence of the last iterate follows from a classic argument with Fejér monotone sequences. From Lemma 1 we have that the distance to any solution is decreasing ∥ x t +1 -x ⋆ ∥ ≤ ∥ x t -x ⋆ ∥ for any minimizer x ⋆ of ϕ , that is, { x t } t ≥ 0 is a Fejér monotone sequence with respect to the solution set. Since we have ϕ ( x t ) → 0 and ϕ is continuous then for every limit point x ′ of the sequence it also holds that ϕ ( x ′ ) = 0 implying x ′ is also a minimizer of ϕ . Therefore we can use the fact that if { x t } t ≥ 0 is Fejér monotone with respect to the solution set and the set contains all the limit points of the sequence then the sequence must converge to a point in the solution set (see Theorem 8.16 in Beck [4]).

## B Relationship between Star Upper Curvature and Upper Quadratic Growth

For a function f , denote by X ⋆ := { x : f ( x ) = min x f ( x ) } . In Guille-Escuret et al. [21], they define the following function class.

Definition 5. A function f is L -quadratically upper bounded (denoted L -QG + ) if for all x ∈ R d :

<!-- formula-not-decoded -->

We now show that convex L -QG + are globally L -star upper curved, while the other direction is true for the local version of the two definitions.

Theorem 5. Let f be a convex L -QG + function, then f is globally L -star upper curved. On the other hand, let f be L x -LSUC, then for all x we have

<!-- formula-not-decoded -->

Proof. Assume that f is L -QG + . Then, we have

<!-- formula-not-decoded -->

where the first inequality is due to convexity and g ∈ ∂f ( y ) . Now, set x = x ⋆ + 1 L g for any x ⋆ ∈ X ⋆ , to have

<!-- formula-not-decoded -->

Now, assume that f is λ x -LSUC and set g ∈ ∂f ( x ) . For any x ⋆ ∈ argmin x f ( x ) , using CauchySchwarz's inequality, we have

<!-- formula-not-decoded -->

Given that this holds for all x ⋆ ∈ X ⋆ , it implies

<!-- formula-not-decoded -->

## C Proofs for the Stochastic Surrogate ψ

Theorem 3. Let h : R d ×S → R ≥ 0 be convex. Denote by H ( x ) = E ξ ∼ D [ h ( x , ξ )] . Then, setting η t = min ( 1 ∥ g t ∥ 2 , γ h ( x t , ξ t ) ) in Algorithm 1, we have

- If h ( · , ξ t ) is L -self bounded, we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

- If h ( · , ξ t ) is G -Lipschitz, then we have

<!-- formula-not-decoded -->

- If h ( · , ξ ) is L -self-bounded and H ( x ) has µ -quadratic growth, then

<!-- formula-not-decoded -->

where a = µ 2 min ( 1 2 L , γ ) and b = 2 γ -min ( 1 2 L , γ ) .

Proof. For simplicity, denote by h t ( x ) = h ( x , ξ t ) .

From the Lemma 3, we have

<!-- formula-not-decoded -->

For the last term in the r.h.s., we have

<!-- formula-not-decoded -->

Observe that if h t is L -self bounded then ∥ g t ∥ 2 ≤ 2 L ( h t ( x ) -inf x h t ( x )) ≤ 2 Lh t ( x ) . Therefore, we have

<!-- formula-not-decoded -->

Now, since η t ≤ 1 / ∥ g t ∥ 2 the second term on the r.h.s can be discarded because it's negative. Taking expectations, we have the first stated bound.

For the second result, bring on the l.h.s. the terms η t 2 h 2 t ( x t ) . Taking expectations, we have the stated bound.

For the third result, first of all observe that for any a, b &gt; 0 we have

<!-- formula-not-decoded -->

Hence, we have

<!-- formula-not-decoded -->

Now, observe that the function B ( x ) = x 2 x + γG 2 is convex x ≥ 0 , because B ′′ ( x ) = 2 γ 2 G 4 ( γG 2 + x ) 3 . So, summing over time and using Jensen's inequality, we have

<!-- formula-not-decoded -->

Note that B -1 ( x ) = x + √ x 2 +4 xγG 2 2 ≤ x + G √ xγ 5 , that is an increasing concave function for x ≥ 0 . So, inverting B and taking expectation, we have

<!-- formula-not-decoded -->

For the smooth and quadratic growth case, we have

<!-- formula-not-decoded -->

Taking expectations and using the quadratic growth assumption on H , we have

<!-- formula-not-decoded -->

5 Using the inequality √ z + y ≤ √ z + √ y ∀ z, y ≥ 0 .

Hence, we obtain where a = µ 2 min ( 1 2 L , γ ) and b = ( 2 γ -min ( 1 2 L , γ )) H ( x ⋆ ) . Note we have 0 ≤ a ≤ µ / 4 L ≤ 1 . From this inequality, it is immediate to obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## D Additional Convergence Result for Generalized Polyak Stepsize

Corollary 1. Let f : R d × S be convex and L -self-bounded. Define x ⋆ = argmin x ( E ξ ∼ D [ f ( x , ξ )] := F ( x )) . Let h ( x , ξ ) = ( f ( x , ξ ) -f ( x ⋆ , ξ )) + . Then, running Algorithm 1 with γ = ∞ , we have

<!-- formula-not-decoded -->

where ¯ x T = 1 T ∑ T t =1 x t or ¯ x T = argmin x ∈{ x 1 ,..., x T } F ( x ) .

Proof. Given the definition of h , we have that H ( x ⋆ ) = 0 . Moreover, h ( x , ξ ) ≥ f ( x , ξ ) -f ( x ⋆ , ξ ) , hence E [ H ( x t )] ≥ E [ F ( x t )] -F ( x ⋆ ) for any t .

We have that

<!-- formula-not-decoded -->

Hence, for all g t ∈ ∂h ( x t , ξ t ) we have

<!-- formula-not-decoded -->

So, using this inequality in Lemma 3 gives

<!-- formula-not-decoded -->

Now, from Cauchy-Schwarz inequality, for any non-negative random variable Y and random variable X , we have E [ X 2 /Y ] ≥ ( E [ X ]) 2 / E [ Y ] . Denote by f ⋆ t = inf x f ( x , ξ t ) . Given that f t ( x ⋆ ) -f ⋆ t ≥ 0 , if f t ( x ⋆ ) -f ⋆ t = 0 with probability 1, i.e., F ( x ⋆ ) -E [inf x f ( x , ξ )] = 0 , then the expectation of the l.h.s. of the previous inequality is E [ h t ( x t )] . Otherwise, if we assume F ( x ⋆ ) -E [inf x f ( x , ξ )] &gt; 0 , we have

<!-- formula-not-decoded -->

Hence, in all cases we have the last expression is a lower bound to the l.h.s. of (8). We now can proceed as in the proof of the Lipschitz case in Theorem 3, to have the stated bound.

We now extend Theorem 3 to Hölder-self-bounded functions.

Definition 6. We say that f is ( L ν , ν ) Hölder-self-bounded if there exits ν ∈ [0 , 1] and L ν such That

<!-- formula-not-decoded -->

This definition is weaker than both Lipschitz and smoothness and it is easy to see that L -smooth functions satisfies this condition with ν = 1 and L 1 = L .

The following theorem generalizes both the Lipschitz and the smooth case, recovering both bounds up to constant factors.

Theorem 7. Let h : R d ×S → R ≥ 0 be convex. Denote by H ( x ) = E ξ ∼ D [ h ( x , ξ )] . Assume that h ( · , ξ t ) is ( L ν , ν ) -Hölder-self bounded. Then, setting η t = min ( 1 ∥ g t ∥ 2 , γ h ( x t , ξ t ) ) in Algorithm 1, we have

<!-- formula-not-decoded -->

where Q ( y ) = 2 y + L ν (2 γy ) 1+ ν 2 ( 1 + 1 ν )

<!-- formula-not-decoded -->

Proof. For simplicity, denote by h t ( x ) = h ( x , ξ t ) .

From the Lemma 3, we have

<!-- formula-not-decoded -->

For the last term in the r.h.s., we have

<!-- formula-not-decoded -->

Observe that if h t is ( L ν , ν ) -Hölder-self-bounded then

<!-- formula-not-decoded -->

where K ν = ( 1 + 1 ν ) 2 ν 1+ ν L 2 1+ ν ν . Therefore, we have

<!-- formula-not-decoded -->

As before, we lower bound the minimum with the convex function B ( x ) = x 2 1+ ν x 1 -ν 1+ ν + γK ν :

<!-- formula-not-decoded -->

As before, this allows us to use Jensen's inequality:

<!-- formula-not-decoded -->

For simplicity of calculations, we now lower bound B ( x ) ,

<!-- formula-not-decoded -->

Note that C ( x ) is invertible and its inverse is

<!-- formula-not-decoded -->

Taking expectations and using Jensen's inequality gives the stated bound.

## E Proofs for Section 6

Lemma 4. Let f : R n → R + where f ⋆ = inf x f ( x ) . Then for any c ≥ 0 the following are equivalent:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

Proposition 1. Suppose h is convex, strictly positive, L -self-bounded, and satisfies the quadratic growth condition h ( x ) -h ⋆ ≥ µ 2 ∥ x -x ⋆ ∥ 2 , where x ⋆ = arg min x h ( x ) is the only fixed point of T (6) . Then for any point x ∈ S = { y : h ( y ) -h ⋆ &lt; h ⋆ µ 8 L -µ } we have

<!-- formula-not-decoded -->

Proof. Let x t be in S then by definition of T we have

<!-- formula-not-decoded -->

If h ( x t ) -h ⋆ &lt; h ⋆ µ 8 L -µ then we have by Lemma 4

<!-- formula-not-decoded -->

Consequently,

<!-- formula-not-decoded -->

Where the last inequality follows from h being self-bounded and convex,

<!-- formula-not-decoded -->

Proposition 6. Suppose h is convex, strictly positive, L -Lipschitz, and has a µ -sharp minimum h ( x ) -h ⋆ ≥ µ ∥ x -x ⋆ ∥ , where x ⋆ = arg min x h ( x ) is the only fixed point of T (6) . Then for any point x ∈ S = { y : y = x ⋆ , h ( y ) -h ⋆ &lt; h ⋆ µ 2 L -µ } we have

<!-- formula-not-decoded -->

Proof. Let x t ∈ S then by following similar steps to Lemma 1 we have

<!-- formula-not-decoded -->

̸

By Lemma 4, we have

<!-- formula-not-decoded -->

Therefore, by sharpness and Lipschitz property of h , we have

<!-- formula-not-decoded -->

Proposition 2 (Cycling and failure to converge) . There exists a strictly positive smooth and strongly convex function h , and initial point x 1 such that iterates from update (6) cycle and satisfy the inequality h ( 1 t ∑ t i =1 x i ) -h ⋆ ≥ δ &gt; 0 for all t .

Proof. The proof is constructive: consider h : R → R , h ( x ) = x 2 +1 , so h ⋆ = 1 . Observe that the update is

<!-- formula-not-decoded -->

Now, we want to choose x 1 so that we oscillate between 3 possible values.

Set x 1 = cot θ where θ has to be determined. The update becomes

<!-- formula-not-decoded -->

where in the last equality we used the identity for cot . Hence, we have x t = cot(2 t θ ) . Given that we want to oscillate between 3 values, we want x t +3 = x t , that is, cot(2 t +3 θ ) = cot(2 t θ ) . We can achieve it if we select θ = π/ 7 . Indeed, we have

<!-- formula-not-decoded -->

Finally, one can verify numerically that f ( 1 t ∑ t i =1 x t ) -f ⋆ &gt; 0 . 77 .

Proposition 8. There exists subregions within the unstable regions in Propositions 1 and 6 where the stepsizes are upper bounded.

Proof. By convexity of h we have h -h ⋆ ≤ ⟨ g t , x t -x ⋆ ⟩ ≤ ∥ g t ∥∥ x t -x ⋆ ∥ , so ∥ g t ∥ ≥ h ( x ) -h ⋆ ∥ x t -x ⋆ ∥ . By assumption in Lemmas 1 and 6 the unstable region is S = { x : h ( x ) -h ⋆ &lt; ch ⋆ } where c depends on the properties of h . Therefore, for x ∈ S and denoting g ∈ ∂h ( x ) as any subgradient at x , we have

<!-- formula-not-decoded -->

If h has a sharp minimum, h ( x ) -h ⋆ ≥ µ ∥ x -x ⋆ ∥ , then we have h ( x ) ∥ g t ∥ 2 &lt; c +1 µ 2 h ⋆ . Therefore, the stepsizes are always bounded within S.

Now consider the subregion S k = { x : h ( x ) -h ⋆ &lt; c k h ⋆ } for some k &gt; 1 . Consider x ∈ S \ S k , that is (1 + c k ) h ∗ ≤ h ( x ) ≤ (1 + c ) h ∗ . If h statisfies the quadratic growth condition h ( x ) -h ⋆ ≥ µ 2 ∥ x -x ⋆ ∥ 2 then

<!-- formula-not-decoded -->

Where the last inequality follows since x / ∈ S k . Therefore, stepsize is bounded within S \ S k , and grows as we increase k .

Definition 7 (Lusin ( N -1 ) condition) . Let T : R d → R k . We define T -1 over a set S ⊆ R k as

<!-- formula-not-decoded -->

We say that T satisfies ( N -1 ) condition if for every set E of measure zero we have that T -1 ( E ) also has measure zero.

̸

Lemma 5. Let T : R n → R n with a unique fixed point x ⋆ . If x ⋆ is unstable, that is, there exists δ such that x = x ⋆ and ∥ x -x ⋆ ∥ ≤ δ , then ∥ T ( x ) -x ⋆ ∥ &gt; ∥ x -x ⋆ ∥ . Define T -1 over a set S ⊆ R n as T -1 ( S ) = { x : T ( x ) ∈ S } . If T -k ( { x ⋆ } ) is of measure zero for any k , then

<!-- formula-not-decoded -->

In other words, the set of initializations that can converge to the fixed point has measure zero.

Proof. We divide R n into two sets, S = ⋃ ∞ k =1 T -k ( { x ⋆ } ) , and its compliment S c . S represents the points that can exactly reach the unique minimizer x ⋆ . If T k ( { x ⋆ } ) is a null set for every k then so is S since the countable union of null sets is a null set.

̸

Now we show that for all initializations in x 1 ∈ S c , x 1 cannot converge to x ⋆ . Suppose the contrary, lim t →∞ x t = x ⋆ . Let B = { y : y = x ⋆ , ∥ x -x ⋆ ∥ ≤ δ } . Since x t → x ⋆ there exists a step n where { x t } t ≥ n ⊆ B . Similarly, there exists n ′ ≥ n where ∥ x t -x ⋆ ∥ ≤ ∥ x n -x ⋆ ∥ for all t ≥ n ′ . This is a contradiction as we have that ∥ x n -x ⋆ ∥ &lt; ∥ x n +1 -x ⋆ ∥ &lt; · · · &lt; ∥ x n ′ -x ⋆ ∥ . Therefore, the initializations that allow for x t → x ⋆ coincide exactly with the null set S .