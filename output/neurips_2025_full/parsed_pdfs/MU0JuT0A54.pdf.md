## Convergence Rates for Gradient Descent on the Edge of Stability for Overparametrised Least Squares

Lachlan Ewen MacDonald

Hancheng Min ∗ Leandro Palma ∗ Salma Tarmoun ∗ Ziqing Xu ∗ René Vidal

Innovation in Data Engineering and Science (IDEAS)

University of Pennsylvania Pennsylvania, PA 19104 lemacdonald@protonmail.com

## Abstract

Classical optimisation theory guarantees monotonic objective decrease for gradient descent (GD) when employed in a small step size, or 'stable", regime. In contrast, gradient descent on neural networks is frequently performed in a large step size regime called the 'edge of stability", in which the objective decreases non-monotonically with an observed implicit bias towards flat minima. In this paper, we take a step toward quantifying this phenomenon by providing convergence rates for gradient descent with large learning rates in an overparametrised least squares setting. The key insight behind our analysis is that, as a consequence of overparametrisation, the set of global minimisers forms a Riemannian manifold M , which enables the decomposition of the GD dynamics into components parallel and orthogonal to M . The parallel component corresponds to Riemannian gradient descent on the objective sharpness, while the orthogonal component is a bifurcating dynamical system. This insight allows us to derive convergence rates in three regimes characterised by the learning rate size: (a) the subcritical regime, in which transient instability is overcome in finite time before linear convergence to a suboptimally flat global minimum; (b) the critical regime, in which instability persists for all time with a power-law convergence toward the optimally flat global minimum; and (c) the supercritical regime, in which instability persists for all time with linear convergence to an orbit of period two centred on the optimally flat global minimum.

## 1 Introduction

The well-known gradient descent (GD) iteration

<!-- formula-not-decoded -->

for minimisation of an objective function ℓ : R p → R , with learning rate η &gt; 0 , is the foundation of most practical algorithms for training deep neural networks (DNNs). Despite the apparent simplicity of the algorithm and practical ease with which DNNs can be trained using GD, theoretically GD on DNNs remains poorly understood. In addition to the well-known non-convexity of DNN loss functions ℓ , our theoretical understanding is hampered by the fact that in practice DNNs tend to be trained with larger learning rates than permissible according to standard optimisation theory.

Specifically, classical optimisation theory requires that the learning rate η be less than twice the reciprocal of the largest eigenvalue λ (the sharpness ) of the Hessian ∇ 2 ℓ . The reason for this is easily

∗ Equal contribution, alphabetical order

seen by considering the simplest quadratic objective ℓ ( θ ) := 1 2 λθ 2 of a single variable θ : if η &gt; 2 /λ , then the iterates θ t +1 = (1 -ηλ ) θ t of GD ℓ ( η, · ) grow in magnitude like | 1 -ηλ | t leading to rapid divergence. In contrast, DNNs are typically trained in a regime called the edge of stability (EOS) [13], in which η is often (pointwise) strictly larger than 2 /λ . Miraculously, despite this large learning rate, GD is capable of converging non-monotonically to a global minimum of the loss, exhibiting an apparent bias toward less sharp ( flatter ) minima.

Despite significant theoretical progress in the past few years, GD at the edge of stability is still poorly understood when compared with its small learning rate counterpart. In particular, while convergence rates for GD in the small learning rate regime can be obtained for DNNs using standard methods [25], convergence rates for GD at the edge of stability have so far only been obtained in settings wherein monotonic loss decrease can be eventually guaranteed [42, 41, 11, 32].

Our purpose in this paper is to provide convergence rates in a least squares setting, wherein sustained monotonic decrease of the loss may never occur . To our knowledge, these are the first rates of this kind to be provided in the literature.

Our analysis synthesises Riemannian geometry and dynamical systems theory to formalise an intuition that has been folklore in the literature for some time [16, 12], namely that GD with a large step size implicitly favours flatter global minima. This formalisation is achieved for codimension 1, overparametrised least squares problems , including deep scalar factorisation, in which the solutions (global minima) are guaranteed to form a p -1 -dimensional Riemannian submanifold M of the p -dimensional Euclidean parameter space R p . Inspired by the 'self-stabilisation' idea proposed in [16], we introduce new coordinates ( θ ∥ , θ ⊥ ) for a neighbourhood of M , where θ ∥ is a point in M and θ ⊥ coordinatises the direction orthogonal to M , with respect to which the GD map ( θ ∥ , θ ⊥ ) ↦→ (GD ∥ ( θ ∥ , θ ⊥ ) , GD ⊥ ( θ ∥ , θ ⊥ )) has approximately the following form:

<!-- formula-not-decoded -->

The left map GD λ M is Riemannian gradient descent on λ along M (see (4)), with step size η ( θ ⊥ ) 2 / (2 c ( η, θ ∥ )) for some strictly positive number c ( η, θ ∥ ) ; the right is the normal form for a period-doubling bifurcation [27]. Our analysis thereby clarifies the implicit bias of GD toward flat minima:

In a neighbourhood of the solution manifold, GD with a large learning rate implicitly performs Riemannian GD on the sharpness along the solution manifold, and oscillates as a bifurcating dynamical system in the direction orthogonal to the solution manifold.

Paper contributions. In the context of codimension 1 least squares problems, we:

1. Make the implicit bias of GD toward flat minima explicit: overparametrisation enables the decomposition of GD into an explicitly sharpness-minimising component along the solution manifold, controlled by an oscillating component orthogonal to the solution manifold.
2. Prove that a class of non-convex scalar factorisation problems fit into our framework. In particular, we prove that despite the non-convexity of the landscape itself, the sharpness in such examples is geodesically strongly convex along a geodesic ball in the solution manifold. To our knowledge, this observation has not previously been made.
3. Identify and prove convergence and implicit bias results for large step size GD in three regimes:
4. (a) the subcritical regime, in which the iterates can be guaranteed to (eventually) decrease monotonically at a linear rate until convergence to a suboptimally flat minimum;
5. (b) the critical regime, in which the iterates converge non-monotonically with a power law rate to the sharpness-minimising global minimum;
6. (c) the supercritical regime, in which the iterates converge linearly to a stable period-two orbit orthogonal to the solution manifold at the point of minimum sharpness.

Although the former and the latter have already been observed in some examples, to our knowledge the middle regime has not yet been identified. Moreover, our work is the first to be able to theoretically quantify the rate of convergence and implicit bias in all three regimes.

4. Verify our theory against numerical experiments.

## 2 Related work

Overparametrisation and convergence guarantees for DNNs. Convergence analyses of GD for non-convex loss landscapes such as DNNs are by now well-established [19, 18, 3, 35, 34, 36, 10]. Although none of these works are able to account for non-monotonic convergence of GD using a large step size, they do point to overparametrisation as a key feature of DNN loss landscapes which enables a (local) Polyak-Łojasiewicz inequality and thus convergence guarantees in the absence of convexity. Specifically, overparametrisation enables the map f sending parameters to network outputs to be submersive , in the sense that the derivative matrix Df of f is pointwise full-rank. Much of the analysis involved in the aforementioned papers is aimed at quantifying this submersivity by lower-bounding the smallest eigenvalue of the 'neural tangent kernel' Df Df T [24, 10]. Overparametrisation in this sense plays a key role in our work too, since it guarantees that the global minima form a smooth manifold with a well-defined normal direction, which is the foundation for Equation (2) and all the analysis that follows.

GD with large learning rates. Recent work in convex optimisation has demonstrated surprising benefits of large step sizes [21, 22, 23, 4, 5], however the mechanism in these works appears to be distinct from the EOS phenomenon in deep learning. Concerning deep learning, although local stability analysis had already hinted at a relation between step size and sharpness prior to 2020 [43], the seminal work of [13] in 2021 was the first to conduct a systematic empirical study of large step size GD. In particular, [13] exposed the fact that GD on DNNs typically uses a larger step size than admissible by classical theory. Moreover, [13] made the empirical observation that this larger step size implicitly regularises sharpness during the late stages training, and coined the term 'edge of stability' for this phase. An explosion of empirical and theoretical work has since been produced in an attempt to account for these empirical observations [2, 7, 40, 38, 39, 16, 30, 44, 1, 12, 26, 42, 41, 11, 17, 32, 20]. At a high level, work up to this point on large step size GD can be classified into one of two categories: general analysis attempting to outline the essence of the mechanism of stability in this regime while abstracting from specific architectures [16, 14]; and specific analysis focusing on precise architectural details in an attempt to derive more fine-grained results on convergence [42, 41, 11, 32] or implicit bias [38, 39, 12, 17]. The advantage of the former approach is generality and clarity of insight, with the disadvantage of less facility in the proof of quantitative results. The advantage of the latter is the wealth of tools for the proof of quantitative results, with the disadvantage of obfuscating essential mathematical structures with inessential architectural detail.

Our work attempts to achieve the strengths of both approaches with a middle ground of abstraction. On the one hand, in common with and inspired by the seminal work [16], our formulation identifies and hinges on minimal mathematical structures which underlie the dynamics of GD with large step size, independent of architectural details; on the other hand, in common with the latter works, we retain enough mathematical structure in our assumptions to cover some of the specific examples already studied in prior work and obtain quantitative convergence rates and implicit bias characterisations. In particular, we are able to provide convergence rates outside of the 'eventually monotonically stable' regimes considered in prior work [42, 41, 11, 32], as well as rigorously quantify oscillatory implicit biases of large step size GD observed empirically and partially accounted for theoretically in [12, 17].

## 3 Theoretical setting

## 3.1 Notation

Given a smooth (i.e., infinitely differentiable, C ∞ ) manifold M without boundary, and a point θ ∈ M , we will use T θ M to denote the tangent space to M at θ , and TM := ⊔ θ ∈ M T θ M to denote its tangent bundle. Given a smooth map g : M → N of smooth manifolds, we will denote by D M g : TM → TN its derivative, acting at each point θ ∈ M as a linear map D M g ( θ ) : T θ M → T g ( θ ) N between tangent spaces.

A Riemannian metric on M is a smoothly-varying inner product ⟨· , ·⟩ θ on each T θ M ; we will usually drop the subscript and denote the metric by simply ⟨· , ·⟩ . Given Riemannian metrics on M and N , any higher derivative D k M g : TM ⊗ k → TN is also defined, and at each point θ ∈ M acts a symmetric, multilinear map D k M g ( θ ) : T θ M ⊗ k → T g ( θ ) N whose value on vectors v 1 , . . . , v k ∈ T θ M we denote D k M g ( θ )[ v 1 , . . . , v k ] (respectively, D k M g ( θ )[ v ⊗ k ] if v i = v ∀ i ). This D k M g may also act on vector fields V 1 , . . . , V k : M → TM to give a map D k M g [ V 1 , . . . , V k ] : M → TN defined by the formula

D k M g [ V 1 , . . . , V k ]( θ ) := D k M g ( θ )[ V 1 ( θ ) , . . . , V k ( θ )] (resp. D k M g [ V ⊗ k ]( θ ) := D k M g ( θ )[ V ( θ ) ⊗ k ] if V i = V ∀ i ) for θ ∈ M .

If g : M → R is scalar-valued, then ∇ k M g : TM ⊗ k -1 → TM will be used to denote the map obtained by dualising one of the inputs of D k M g using the Riemannian metric. Specifically, given any θ ∈ M and tangent vectors v 2 , . . . , v k ∈ T θ M , ∇ k M g ( θ )[ v 2 , . . . , v k ] is the unique element of T θ M satisfying

<!-- formula-not-decoded -->

where ⟨· , ·⟩ is the Riemannian metric in T θ M . The objects ∇ M g and ∇ 2 M g are the Riemannian gradient and Riemannian Hessian respectively. In particular, D k R p g and ∇ k R p g will be denoted simply D k g and ∇ k g ; ∇ g and ∇ 2 g then are the ordinary Euclidean gradient and Hessian respectively.

A geodesic is a locally length-minimising curve in M (for instance, geodesics in Euclidean space are just straight lines). We will use d M ( θ, θ ′ ) to denote the geodesic distance between θ, θ ′ ∈ M , which is the length of the shortest geodesic connecting θ to θ ′ ; d M makes M into a metric space. For any θ ∈ M and any sufficiently small v ∈ T θ M , there is a unique geodesic γ : [0 , 1] → M such that γ (0) = θ and ˙ γ (0) = v ; the exponential map on ( θ, v ) is then by definition exp θ ( v ) := γ (1) ; in Euclidean space exp θ ( v ) = θ + v . See Appendix A for more details. Any smooth function g : M → R is associated to a Riemannian gradient descent map ( η, θ ) ↦→ GD g M ( η, θ ) ∈ M defined in an open neighbourhood of { 0 } × M ⊂ R ≥ 0 × M by

<!-- formula-not-decoded -->

This reduces to the familiar ( η, θ ) ↦→ θ -η ∇ g ( θ ) in the Euclidean setting, wherein it is denoted simply GD g with the subscript dropped.

Recall that a regular value of a function f : R p → R d is a point y ∈ R d such that f -1 { y } is nonempty and f is C ∞ in a neighbourhood of f -1 { y } with derivative Df ( θ ) : T θ R p → T f ( θ ) R d surjective for all θ ∈ f -1 { y } . Moreover, if y is a regular value of f , then M := f -1 { y } is a smooth submanifold of R p , and if f is also analytic in a neighbourhood of M , then M is an analytic manifold.

## 3.2 Problem setting

Given a natural number p &gt; 1 , function f : R p → R and target value y ∈ R , we aim to solve the codimension 1 least squares problem

<!-- formula-not-decoded -->

using gradient descent with constant step size η &gt; 0 :

<!-- formula-not-decoded -->

We will make the following assumptions on f and y .

Assumption 3.1 (Regularity and analyticity) . The point y ∈ R is a regular value of f , and f is analytic in a neighbourhood of the pre-image M := f -1 { y } .

The assumption that y is a regular value of f is a strong version of overparametrisation : it is equivalent to assuming that the 'neural tangent kernel" Df Df T = ∥ Df ∥ 2 is non-vanishing along M , as is often insisted upon in overparametrisation theory for deep learning [24, 10]. Analyticity of f in a neighbourhood of M := f -1 { y } will be used to invoke powerful results from holomorphic dynamics [6] in our convergence theorems.

The key consequence of Assumption 3.1 is that, by the regular value theorem, the solution set M of (5) is a ( p -1) -dimensional analytic submanifold of R p . Points in M will be denoted θ ∥ in what follows. For any θ ∥ ∈ M , the tangent space T θ ∥ M to M at θ ∥ is the kernel of Df ( θ ∥ ) , and any singular value decomposition of Df ( θ ∥ ) admits precisely one nonzero singular value corresponding to a singular vector orthogonal to T θ ∥ M . This singular vector is the normal vector n ( θ ∥ ) := ∇ f ( θ ∥ ) / ∥∇ f ( θ ∥ ) ∥ .

It follows from the chain rule and the fact that f ≡ y along M that the Hessian ∇ 2 ℓ of ℓ satisfies the identity

<!-- formula-not-decoded -->

along M . We denote by λ the largest eigenvalue of ∇ 2 ℓ , which is equal to ∥∇ f ∥ 2 . We assume M to be equipped with the Riemannian metric inherited from its embedding into the Euclidean space R p .

Our new coordinate representation (2) for gradient descent is essentially a consequence of coordinatising a tubular neighbourhood N of M by a ∥ -coordinate along M , and a ⊥ -coordinate orthogonal to M (Figure 1). However, the rigorous form of (2) requires two additional assumptions on the solution manifold M . The first of these provides an invariant line segment about which GD can oscillate, and enables the correct decay rate of the error terms in the ∥ -component of GD.

Assumption 3.2 (Orthogonal stability) . There is θ ∥ ∗ ∈ M and a line segment L through θ ∥ ∗ orthogonal to M which is invariant under gradient descent on ℓ (see Figure 1).

The second assumption necessary for the rigorous form of (2), concerning the ⊥ -equation, is a standard assumption from bifurcation theory enabling the realisation of the ⊥ -equation in a wellknown normal form which is easily analysed [27, Theorem 4.3].

Assumption 3.3 (Genericity) . Recall the normal vector field n := ∇ f/ ∥∇ f ∥ . At each point θ ∥ ∈ M , for all η in a neighbourhood of 2 /λ ( θ ∥ ) , one has

<!-- formula-not-decoded -->

Assumptions 3.1, 3.2 and 3.3 are sufficient to obtain the rigorous form of (2) (Theorem 4.1). As is evident from Equation (2), GD with a large step size behaves like gradient descent on the sharpness along M . To utilise this fact in our convergence theorems, we will make the following additional assumptions on M and the sharpness λ .

Assumption 3.4 (Strong convexity of λ ) . There is a ball B M ( θ ∥ ∗ , r ) ⊂ M , centred on θ ∥ ∗ and of radius r &gt; 0 with respect to the geodesic distance, which is geodesically convex in the sense that any two points in B M ( θ ∥ ∗ , r ) are connected by a unique distance-minimising geodesic, over which the sharpness λ : M → R is µ -geodesically strongly convex and has L -geodesically Lipschitz gradients , in the sense that the Riemannian Hessian ∇ 2 M λ of λ satisfies

<!-- formula-not-decoded -->

Figure 1: M = { ( x, y ) : xy = 1 } with tubular neighbourhood N (shaded) and line L (dotted). Inside N , any point θ is closest to a unique point θ ∥ on M , with θ -θ ∥ = θ ⊥ n ( θ ∥ ) orthogonal to M for some θ ⊥ ∈ R . Assumption 3.2 says that ℓ ( θ ′ ) is parallel to at any point θ ′ .

<!-- image -->

∇ L ∈ L uniformly over B M ( θ ∥ ∗ , r ) , where I TM is the identity map on TM . We assume moreover that λ achieves its minimum value λ ( θ ∥ ∗ ) in B M ( θ ∥ ∗ , r ) uniquely at θ ∥ ∗ , where one moreover has

<!-- formula-not-decoded -->

for some ν &gt; 0 , where I TM ( θ ∥ ∗ ) is the identity map on the tangent space T θ ∥ ∗ M of M at θ ∥ ∗ .

These assumptions need not be satisfied for general functions f . The geodesic strong convexity of λ in particular seems at first to be an alarmingly strong assumption, since it is well-known that DNN loss landscapes are non -convex. Surprisingly, these non-trivial assumptions can be guaranteed to hold for the following class of toy examples of deep learning non-convex loss landscapes.

̸

Multilayer scalar factorisation. The map f : R p → R defined by f ( θ 1 , . . . , θ p ) := θ p · · · θ 1 corresponds to a linear network of depth p and width 1 on a single input datum. Any nonzero target value y = 0 is a regular value for f , and f -1 { y } is then a union of hypersurfaces (Proposition B.1). Assuming without loss of generality that y &gt; 0 and setting θ ∥ ∗ := y 1 p 1 p , one takes M to be the connected component of θ ∥ ∗ . The line L is given by the span of the ones-vector 1 p (Proposition B.2). Assumption 3.3 holds by Proposition B.3, while Proposition B.6 demonstrates that λ is geodesically strongly convex on a geodesic ball centred at θ ∥ ∗ , with

<!-- formula-not-decoded -->

so that Assumption 3.4 holds. One may also take f ( θ 1 , . . . , θ p ) := θ p ϕ ( θ p -1 ϕ ( · · · ( ϕ ( θ 1 )) · · · ) , with ϕ any nonlinearity that is the identity on a neighbourhood of y 1 /p and still satisfy all assumptions.

Some thought reveals that these assumptions can be relaxed in more-or-less straightforward ways to deep linear networks of sufficient constant width on multi-point datasets. However, the scalar case we consider in this paper is already sufficiently instructive and non-trivial that we leave a more general elaboration of this framework to future work.

## 4 A normal form for gradient descent about the solution manifold

In this section we state a rigorous form (Theorem 4.1) of the approximate update equations alluded to in Equation (2), and give an idea of its proof. This section makes use only of Assumptions 3.1, 3.2 and 3.3. The geodesic convexity of Assumption 3.4 is not needed to derive the normal form of GD about M , and is necessary only for the convergence theorems to be given in the next section.

Our analysis of the dynamics of GD is inspired by that of [16]; it proceeds from a Taylor expansion 2 . However, while [16] Taylor expands around a set of points with sharpness ≤ 2 /η , we Taylor expand around the solution manifold M . Our derivation is novel and provides a number of advantages: the assumptions are more transparent and easily checked, and we are guaranteed that M is a Riemannian manifold, while the set of points with sharpness ≤ 2 /η considered in [16] need not be.

Specifically, every point θ sufficiently close to M admits a unique nearest point θ ∥ on M . The difference θ -θ ∥ is then a scalar multiple θ ⊥ n ( θ ∥ ) of the normal vector n ( θ ∥ ) = ∇ f ( θ ∥ ) / ∥∇ f ( θ ∥ ) ∥ at θ ∥ . Thus, pairs ( θ ∥ , θ ⊥ ) with θ ∥ ∈ M and sufficiently small θ ⊥ ∈ R suffice to completely coordinatise a neighbourhood of M in R p known as a tubular neighbourhood N [28, p. 147]. Our first result is then the following characterisation of the GD dynamics in terms of θ ∥ , θ ⊥ , the sharpness λ and its Riemannian gradient descent map GD λ M .

Theorem 4.1 (Normal form for GD about M ) . There is an analytic change of coordinates ( η, θ ∥ , θ ⊥ ) ↦→ ( η, θ ) for a tubular neighbourhood N of M in which the gradient descent map GD ℓ : ( η, θ ) ↦→ ( η, θ -η ∇ ℓ ( θ ∥ ) ) takes the form ( η, θ ∥ , θ ⊥ ) ↦→ ( η, GD ∥ ( η, θ ∥ , θ ⊥ ) , GD ⊥ ( η, θ ∥ , θ ⊥ ) ) , where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with c ( η, θ ∥ ) defined in Assumption 3.3.

Outline of proof. Recalling the normal vector field n := ∇ f/ ∥∇ f ∥ , we approximate the gradient ∇ ℓ ( θ ) at θ ∈ N by the Taylor series

<!-- formula-not-decoded -->

Since M consists of global minima for ℓ , ∇ ℓ ( θ ∥ ) = 0 . Additionally, since n ( θ ∥ ) is the sole eigenvector along which ∇ 2 ℓ ( θ ∥ ) = ∇ f ( θ ∥ ) ∇ f ( θ ∥ ) T is nontrivial, with eigenvalue λ ( θ ∥ ) , one has ∇ 2 ℓ ( θ ∥ )[ n ( θ ∥ )] = λ ( θ ∥ ) n ( θ ∥ ) . One thus has:

<!-- formula-not-decoded -->

Plugging this into the gradient descent update formula, applying the orthogonal projection P TM of T R p onto TM and n T of T R p onto the normal direction respectively, and noting that

<!-- formula-not-decoded -->

2 The idea of higher order terms inducing implicit bias is also present in literature on Sharpness-Aware Minimisation [8].

since ∇ ( n T n ) = ∇ (1) = 0 , yields the following formulae for the GD update in ( θ ∥ , θ ⊥ ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The O (( θ ⊥ ) 3 ) error in the ∥ -update is tightened to O ( ( θ ⊥ ) 3 min { 1 , d M ( θ ∥ , θ ∥ ∗ ) } ) using Assumption 3.2, and θ ∥ -η ( θ ⊥ ) 2 2 ∇ M λ ( θ ∥ ) is the first order approximation of exp θ ∥ ( -η ( θ ⊥ ) 2 2 ∇ M λ ( θ ∥ ) ) by the Taylor expansion of the exponential map [33]. The final form is obtained by invoking Assumption 3.3 and applying a standard transformation from bifurcation theory [27, Theorem 4.3]. See Theorem C.6 for a rigorous proof.

We can immediately make the following qualitative observations of the dynamics in Theorem 4.1, with which we will be able to give an idea of how our convergence theorems work. Let ( θ ∥ t , θ ⊥ t ) denote the t th iterate of gradient descent in the ( θ ∥ , θ ⊥ ) coordinates of Theorem 4.1.

The ∥ -component of the GD dynamics is a perturbed version of Riemannian gradient descent on the sharpness along the solution manifold M , with time-varying step size determined by the magnitude of the ⊥ -iterates θ ⊥ t . Recalling that we assume λ to be geodesically strongly convex as in Assumption 3.4, we can be assured of descent guarantees for λ provided | θ ⊥ t | can be controlled.

The behaviour of | θ ⊥ t | is governed by the map (13), which is a perturbed, time-varying version of the simpler 'flip bifurcation"

<!-- formula-not-decoded -->

from bifurcation theory [27]. The iterates x t of (19) admit the following dynamics.

1. When η &lt; 2 /λ ′ , 0 is a hyperbolic attractor, and the iterates | x t | go to zero exponentially fast. Were this to hold also for the iterates | θ ⊥ t | of (13), θ ∥ t would evolve like gradient descent on λ with exponentially decaying step size, hence would converge exponentially fast to a suboptimally flat point.
2. When η = 2 /λ ′ , 0 is a parabolic attractor, and the iterates | x t | go to zero like Θ( t -1 / 2 ) . Were this to hold also for the iterates | θ ⊥ t | of (13), θ ∥ t would evolve like gradient descent on λ with power-law-decaying stepsize, hence would converge with a power-law rate to the optimally flat point.
3. When η &gt; 2 /λ ′ , there is a hyperbolically attracting orbit of period two with amplitude ≈ √ ηλ ′ -2 to which the iterates x t converge exponentially fast. Were the same to be true for the iterates θ ⊥ t of (13), θ ∥ t would evolve like gradient descent on λ with step size Θ ( ( ηλ ′ -2) ) , hence would converge exponentially to the optimally flat point.

In the next section, we provide theorems showing that these intuitions gathered from the unperturbed, time-invariant (19) can be rigorously carried over to the perturbed, time-varying system defined by Theorem 4.1.

## 5 Convergence theorems

In this section, we state our convergence theorems and provide numerical demonstrations of them on a multilayer scalar factorisation problem. In addition to the Assumptions 3.1, 3.2 and 3.3 required for the normal form (Theorem 4.1), we now also invoke Assumption 3.4 to provide rates of convergence. Iterates will be denoted θ t (respectively ( θ ∥ t , θ ⊥ t ) ) for t ∈ N ∪{ 0 } . For convenience, our | θ ⊥ t | plots concern the orthogonal coordinate of the intermediate Proposition C.5 rather than that of the final Theorem 4.1; since the transformation going between them is analytic (Theorem C.6), this causes no difference in the convergence rate.

## 5.1 Subcritical regime

The subcritical regime is when 2 /λ ( θ ∥ ∗ ) &gt; η &gt; 2 /λ ( θ ∥ 0 ) . In this regime, considered also in certain examples in prior works [26, 32], the orthogonal component θ ⊥ t of the iterates exhibits initial, transient oscillation driven by (13) of Theorem 4.1; during this time, the sharpness values λ ( θ ∥ t ) are monotonically decreasing according to Theorem 4.1 to the point of stability η &lt; 2 /λ ( θ ∥ t ) in a finite number of steps. Following the achievement of stability, θ ⊥ t decays exponentially fast, resulting in exponentially less aggressive steps in θ ∥ t to decrease λ ( θ ∥ t ) , ultimately resulting in convergence to a suboptimally-flat global minimum of ℓ . See Figure 2. The theorem below appears as Theorem D.4 in the appendix.

̸

Theorem 5.1. Assume that η &lt; 2 /λ ( θ ∥ ∗ ) . Then there is a constant γ &gt; 0 such that for all θ ⊥ 0 sufficiently close to zero and all θ ∥ 0 = θ ∥ ∗ ∈ B M ( θ ∥ ∗ , r ) , if η &gt; 2 /λ ( θ ∥ 0 ) is sufficiently small, then there is

<!-- formula-not-decoded -->

such that η &lt; 2 /λ ( θ ∥ t ) for all t ≥ τ and η ≥ 2 /λ ( θ ∥ t ) for all t &lt; τ . Consequently, setting β := 1 -(2 -ηλ ( θ ∥ τ )) &lt; 1 , the iterates ( θ ∥ t , θ ⊥ t ) converge to a global minimum ( θ ∥ ∞ , 0) of ℓ in M with suboptimal sharpness at a rate O ( β t ) .

<!-- formula-not-decoded -->

Figure 2: Log-scale plots of distance from θ ∥ t to θ ∥ ∗ (left), magnitude of θ ⊥ t (centre) and sharpness suboptimality gap (right) for gradient descent on depth 5 scalar factorisation in the subcritical regime . Trajectories from five different initialisations shown. Initial instability in | θ ⊥ t | (top) is overcome in finite time with rapid convergence to a suboptimally flat global minimum (bottom).

<!-- image -->

## 5.2 Critical regime

The critical regime happens when η = 2 /λ ( θ ∥ ∗ ) . To our knowledge, this regime has not been observed previously in the literature. In this regime, the dynamics of GD are non-hyperbolic, with θ ⊥ t exhibiting 2-periodic decay to zero with a rate Θ( t -1 2 ) . Consequently, the dynamics of θ ∥ t are essentially those of gradient descent on λ with a step size decaying according to a power-law. Ultimately then, θ t does converge to the optimally flat global minimum of ℓ , but does so at a slow rate. See Figure 3. The theorem below appears as Theorem D.9 in the appendix.

Theorem 5.2. Assume that η = 2 /λ ( θ ∥ ∗ ) and that ν/ ( c (2 /λ ∗ , θ ∥ ∗ ) λ ∗ ) &lt; 1 , where ν and c ( · , · ) are defined as in Assumptions 3.3 and 3.4 respectively. Then for all θ ∥ 0 sufficiently close to θ ∥ ∗ and all θ ⊥ 0 = 0 sufficiently small, one has d M ( θ ∥ t , θ ∥ ∗ ) = Θ( t -1 / 2 ) and | θ ⊥ t | = Θ( t -1 / 2 ) .

̸

Figure 3: Logy -scale plots of distance from θ ∥ t to θ ∥ ∗ (left), magnitude of θ ⊥ t (centre) and sharpness suboptimality gap (right) on depth 5 scalar factorisation in the critical regime . Trajectories from five different initialisations shown. The iterates | θ ⊥ t | may or may not initially increase ; in both cases, asymptotic, power law decrease of | θ ⊥ t | and ∥ θ ∥ t -θ ∥ ∗ ∥ to zero indicates power law convergence to the optimally flat global minimum.

<!-- image -->

Some further remarks about the critical regime are necessary. First, the additional hypothesis that ν/ ( c (2 /λ ∗ , θ ∥ ∗ ) λ ∗ ) &lt; 1 (which is satisfied for multilayer scalar factorisation, see Proposition B.7) is not strictly necessary for convergence; power-law convergence still occurs to the same minimum without this assumption, but is in that case faster for d M ( θ ∥ t , θ ∥ ∗ ) than the Θ( t -1 / 2 ) rate given in the theorem statement. However, we are unaware of examples having this faster rate. Second, the | θ ⊥ t | iterates may initially increase substantially as in Figure 3 depending on the initialisation; our theorem accommodates this behaviour explicitly (see Theorem D.8).

## 5.3 Supercritical regime

The supercritical regime is when η &gt; 2 /λ ( θ ∥ ∗ ) . In this case, when η &gt; 2 /λ ( θ ∥ ∗ ) is sufficiently small, Equation (13) exhibits linear convergence to a stable orbit of period two with amplitude ≈ ( ηλ ( θ ∥ ∗ ) -2) 1 2 . It follows that the step sizes in θ ∥ t in its descent on λ are of approximately constant size, so that θ ∥ t will converge linearly to the optimally flat θ ∥ ∗ . However, the complete iterates θ t of GD do not converge to a global minimum, but asymptotically oscillate orthogonally to M about the point θ ∥ ∗ along the line L of Assumption 3.2. See Figure 4. Although this regime has been observed in prior works [12, 17, 20], no general quantitative convergence theorem in this regime has yet been proved. The theorem below appears as Theorem D.10 in the appendix.

Theorem 5.3. Assume that η &gt; 2 /λ ( θ ∥ ∗ ) is sufficiently small. Then there are positive constants C 1 , C 2 , C 3 and a stable orbit of period two of the form ( θ ∥ ∗ , ± ( ηλ ( θ ∥ ∗ ) -2) 1 / 2 + O ( ηλ ( θ ∥ ∗ ) -2) ) to which the iterates ( θ ∥ t , θ ⊥ t ) starting from any ( θ ∥ 0 , θ ⊥ 0 ) satisfying 0 &lt; | θ ⊥ 0 | ≤ C 1 ( ηλ ( θ ∥ ∗ ) -2) 1 / 2 and d M ( θ ∥ 0 , θ ∥ ∗ ) ≤ C 2 ( ηλ ( θ ∥ ∗ ) -2) 1 / 2 converge with rate O ( (1 -C 3 ( ηλ ( θ ∥ ∗ ) -2)) t ) .

## 6 Limitations, discussion and conclusion

Our work has a number of limitations, all of which point toward avenues for further exploration along the lines we have developed.

Higher order orthogonal dynamics. Prior work has indicated that beyond the 2-periodic behaviour we studied in this work, higher-order periodicity and chaos emerge as the learning rate increases [17]. Our Theorem 4.1 provides new insight into this behaviour as a manifestation of a bifurcating system oscillating orthogonally to the solution manifold. Our analysis only handles the simplest (namely 2-periodic) case of this supercritical behaviour. We expect our framework to be able to admit convergence proofs of higher oscillatory and chaotic behaviour also, using ideas from dynamical systems theory, however we do not expect this to be easy and have not attempted it in this paper.

Figure 4: Log-scale plots of distance from θ ∥ t to θ ∥ ∗ (left), magnitude of θ ⊥ t (centre) and sharpness suboptimality gap (right) on depth 5 scalar factorisation in the supercritical regime . Trajectories from five different initialisations shown. The iterates | θ ⊥ t | converge to a stable, period-two orbit driving linear convergence of θ ∥ t to the optimally flat global minimum θ ∥ ∗ .

<!-- image -->

Higher codimension least squares. Our theory only allows us to treat codimension 1 problems. This is clearly a severe limitation, ruling out for instance overparametrised regression on multiple datapoints or multiple outputs. It is not difficult to extend our definitions to higher codimension, however proving anything in this setting seems very challenging, as it would require an understanding of higher-dimensional bifurcating dynamical systems. We leave this question to future work.

Geodesic convexity of sharpness for more general solution manifolds. One of the most surprising results of our work is that the sharpness of the loss is geodesically strongly convex over a geodesic ball in the solution manifold for a number of overparametrised non-convex problems. Should this prove to be a more general fact, which we suspect may be the case, it would go a long way to explaining the still-mysterious ease of optimisation and implicit bias of GD on DNNs. We leave a more thorough investigation of this question to future work.

Finding the tubular neighbourhood. Our theory is premised on the assumption that GD is initialised in a tubular neighbourhood of the solution manifold. While there is no reason to think this is the case with standard initialisation schemes used in practice, we conjecture that GD does converge rapidly to a tubular neighbourhood from a standard initialisation during the phase known as progressive sharpening [13], after which the familiar dynamics described here would apply. We leave exploration of this question to future work.

Relation to stochasticity. Prior works [9, 15, 31] studying stochastic gradient descent have demonstrated an implicit bias toward flat minimisers arising from stochasticity. All of these works consider a smaller learning rate than those considered in our work; consequently, the mechanism behind the implicit bias considered in [9, 15, 31] is fundamentally different than that considered in our work. Studying how these distinct implicit biases interact is an important direction for future research.

## Acknowledgments and Disclosure of Funding

This research was supported by NSF 2031985 and Simons Foundation 814201 (Theorinet), ONR MURI 503405-78051 and University of Pennsylvania Startup Funds.

## References

- [1] Atish Agarwala, Fabian Pedregosa, and Jeffrey Pennington. Second-order regression models exhibit progressive sharpening to the edge of stability. In ICML , 2023.
- [2] K. Ahn, J. Zhang, and S. Sra. Understanding the unstable convergence of gradient descent. In ICML , 2022.
- [3] Z. Allen-Zhu, Y. Li, and Z. Song. A Convergence Theory for Deep Learning via OverParameterization. In ICML , pages 242-252, 2019.
- [4] J. M. Altschuler and P. Parrilo. Acceleration by Stepsize Hedging: Multi-Step Descent and the Silver Stepsize Schedule. Journal of the ACM , 2023.

- [5] J. M. Altschuler and P. Parrilo. Acceleration by stepsize hedging: Silver Stepsize Schedule for smooth convex optimization. Mathematical Programming , pages 1-14, 2024.
- [6] M. Arizzi and J. Raissy. On Écale-hakim's theorems on holomorphic dynamics. Frontiers in Complex Dynamics , 2014.
- [7] S. Arora, Z. Li, and A. Panigrahi. Understanding Gradient Descent on Edge of Stability in Deep Learning. In ICML , 2022.
- [8] P. L. Bartlett, P. M. Long, and O. Bousquet. The Dynamics of Sharpness-Aware Minimization: Bouncing Across Ravines and Drifting Towards Wide Minima. Journal of Machine Learning Research , 24:1-36, 2023.
- [9] G. Blanc, N. Gupta, G. Valiant, and P. Valiant. Implicit regularization for deep neural networks driven by an Ornstein-Uhlenbeck like process. In COLT , 2020.
- [10] S. Bombari, M. H. Amani, and M. Mondelli. Memorization and Optimization in Deep Neural Networks with Minimum Over-parameterization. In NeurIPS , 2022.
- [11] Y. Cai, J. Wu, S. Mei, M. Lindsey, and P. L. Bartlett. Large Stepsize Gradient Descent for Non-Homogeneous Two-Layer Networks: Margin Improvement and Fast Optimization. In NeurIPS , 2024.
- [12] Lei Chen and Joan Bruna. Beyond the Edge of Stability via Two-step Gradient Updates. In ICML , 2023.
- [13] J. Cohen, S. Kaur, Y. Li, J. Zico Kolter, and A. Talwalkar. Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability. In ICLR , 2021.
- [14] Jeremy Cohen, Alex Damian, Ameet Talwalkar, J Zico Kolter, and Jason D. Lee. Understanding Optimization in Deep Learning with Central Flows. In ICLR , 2025.
- [15] A. Damian, T. Ma, and J. Lee. Label Noise SGD Provably Prefers Flat Global Minimizers. In NeurIPS , 2021.
- [16] A. Damian, E. Nichani, and J. Lee. Self-Stabilization: The Implicit Bias of Gradient Descent at the Edge of Stability. In ICLR , 2023.
- [17] Dayal Singh Kalra and Tianyu He and Maissam Barkeshli. Universal Sharpness Dynamics in Neural Network Training: Fixed Point Analysis, Edge of Stability, and Route to Chaos. In ICLR , 2025.
- [18] S. S. Du, J. Lee, H. Li, L. Wang, and X. Zhai. Gradient Descent Finds Global Minima of Deep Neural Networks. In ICML , pages 1675-1685, 2019.
- [19] S. S. Du, X. Zhai, B. Poczos, and A. Singh. Gradient Descent Provably Optimizes Overparameterized Neural Networks. In ICLR , 2019.
- [20] Avrajit Ghosh, Soo Min Kwon, Rongrong Wang, Saiprasad Ravishankar, and Qing Qu. Learning dynamics of deep matrix factorization beyond the edge of stability. In ICLR , 2025.
- [21] B. Grimmer. Provably faster gradient descent via long steps. SIAM Journal on Optimization , 34:2588-2608, 2024.
- [22] B. Grimmer, K. Shu, and A. L. Wang. Accelerated gradient descent via long steps. arXiv:2309.09961, 2023.
- [23] B. Grimmer, K. Shu, and A. L. Wang. Accelerated objective gap and gradient norm convergence for gradient descent via long steps. INFORMS Journal on Optimization , 7:156-169, 2025.
- [24] A. Jacot, F. Gabriel, and C. Hongler. Neural Tangent Kernel: Convergence and Generalization in Neural Networks. In NeurIPS , pages 8571-8580, 2018.
- [25] H. Karimi, J. Nutini, and M. Schmidt. Linear Convergence of Gradient and Proximal-Gradient Methods Under the Polyak-Łojasiewicz Condition. In ECML PKDD , pages 795--811, 2016.

- [26] Itai Kreisler, Mor Shpigel Nacson, Daniel Soudry, and Yair Carmon. Gradient descent monotonically decreases the sharpness of gradient flow solutions in scalar networks and beyond. In ICML , 2023.
- [27] Y. A. Kuznetsov. Elements of Applied Bifurcation Theory, 4th Edition . Springer, 2023.
- [28] J. M. Lee. Introduction to Smooth Manifolds, Second Edition . Springer, 2013.
- [29] J. M. Lee. Introduction to Riemannian Manifolds, Second Edition . Springer, 2018.
- [30] S. Lee and C. Jang. A new characterization of the edge of stability based on a sharpness measure aware of batch gradient distribution. In ICLR , 2023.
- [31] Z. Li, T. Wang, and S. Arora. What Happens after SGD Reaches Zero Loss? -A Mathematical Framework. In ICLR , 2022.
- [32] Liming Liu, Zixuan Zhang, Simon Du, and Tuo Zhao. A minimalist example of edge-of-stability and progressive sharpening, 2025.
- [33] Maria G Monera, A Montesinos-Amilibia, and Esther Sanabria-Codesal. The taylor expansion of the exponential map and geometric applications. Revista de la Real Academia de Ciencias Exactas, Fisicas y Naturales. Serie A. Matematicas , 108:881-906, 2014.
- [34] Q. Nguyen. On the Proof of Global Convergence of Gradient Descent for Deep ReLU Networks with Linear Widths. In NeurIPS , 2021.
- [35] Q. Nguyen and M. Mondelli. Global Convergence of Deep Networks with One Wide Layer Followed by Pyramidal Topology. In NeurIPS , 2020.
- [36] Q. Nguyen, M. Mondelli, and G. Montufar. Tight Bounds on the Smallest Eigenvalue of the Neural Tangent Kernel for Deep ReLU Networks. In ICML , 2021.
- [37] P. Petersen. Riemannian Geometry . Springer, 2016.
- [38] Yuqing Wang, Minshuo Chen, Tuo Zhao, and Molei Tao. Large Learning Rate Tames Homogeneity: Convergence and Balancing Effect. In ICLR , 2022.
- [39] Yuqing Wang, Zhenghao Xu, Tuo Zhao, and Molei Tao. Good regularity creates large learning rate implicit biases: edge of stability, balancing, and catapult. In NeurIPS 2023 Workshop on Mathematics of Modern Machine Learning , 2023.
- [40] Z. Wang, Z. Li, and J. Li. Analyzing Sharpness along GD Trajectory: Progressive Sharpening and Edge of Stability. In NeurIPS , 2022.
- [41] J. Wu, P. L. Bartlett, M. Telgarsky, and B. Yu. Large Stepsize Gradient Descent for Logistic Loss: Non-Monotonicity of the Loss Improves Optimization Efficiency. In COLT , 2024.
- [42] J. Wu, V. Braverman, and J. Lee. Implicit Bias of Gradient Descent for Logistic Regression at the Edge of Stability. In NeurIPS , 2023.
- [43] L. Wu, C. Ma, and W. E. How SGD Selects the Global Minima in Over-parameterized Learning: A Dynamical Stability Perspective. In NeurIPS , 2018.
- [44] X. Zhu, Z. Wang, X. Wang, M. Zhou, and R. Ge. Understanding Edge-of-Stability Training Dynamics with a Minimalist Example. In ICLR , 2023.

## A Differential geometry background

The purpose of this section is to give a brief overview of essential notions from Riemannian geometry. A good source for the first subsection is [29]. We have not been able to locate a good source for the latter subsection; in particular, to our knowledge no attempt has yet been made to consider the Riemannian Hessian of the square gradient norm of a function defining a hypersurface, however the calculation is elementary.

## A.1 Riemannian metrics, connections and geodesics

Recall that a Riemannian metric on a manifold M is a smoothly-varying family of positive-definite inner products ⟨· , ·⟩ θ on the tangent spaces T θ M of M , with associated norm ∥· ∥ θ . Any submanifold M of a Riemannian manifold ( N,g N ) inherits a Riemannian metric from N ; precisely, given tangent vectors v, w ∈ T θ M ⊂ T θ N , one defines

<!-- formula-not-decoded -->

Since we work with submanifolds of Euclidean space, whose tangent spaces are all themselves subspaces of Euclidean space equipped with the Euclidean inner product, we will often use the notation ⟨· , ·⟩ and ∥ · ∥ without further decoration when considering the metric of a submanifold of Euclidean space.

Denote by T ∗ M the cotangent bundle of M , whose fibre over θ ∈ M is the space of linear functionals T θ M → R . Given ( k, l ) ∈ N ∪ { 0 } a ( k, l ) -tensor at the point θ ∈ M is an element of T θ M ⊗ k ⊗ T ∗ θ M ⊗ l , and a ( k, l ) -tensor field is a smooth section of the bundle TM ⊗ k ⊗ T ∗ M ⊗ l , that is, a smooth assignment of a tensor T ( θ ) ∈ T θ M ⊗ k ⊗ T ∗ θ M ⊗ l to each θ ∈ M . The space of such sections will be denoted Γ( TM ⊗ k ⊗ T ∗ M ⊗ l ) .

Whether or not M carries a Riemannian metric, the derivative of a map f : M → R makes sense as a map D M f : TM → T R which is fibrewise-linear in the sense that its restriction D M f ( θ ) : T θ M → T f ( θ ) R to each fibre is a linear map. The derivative D M f can in this sense be thought of as a (0 , 1) -tensor field. Higher derivatives D k M f of f are thus derivatives of tensor fields, and can be defined using a Riemannian metric. Specifically, assuming that M has a Riemannian metric, there is a distinguished derivative operator D M : Γ( TM ⊗ k ⊗ T ∗ M ⊗ l ) → Γ( TM ⊗ k ⊗ T ∗ M ⊗ l +1 ) defined for all ( k, l ) ∈ N ∪ { 0 } called the Levi-Civita connection . The additional T ∗ M -slot gained by applying D M to a tensor T is to be thought of as a direction in which T is differentiated by D M . In particular, given T ∈ Γ( TM ⊗ k ⊗ T ∗ M ⊗ l ) and X ∈ Γ( TM ) , the notation D M T [ X ] ∈ Γ( TM ⊗ k ⊗ T ∗ M ⊗ l ) is the directional derivative of T in the direction X . Note that when M = R p , D := D R p is the usual multivariate derivative, and the higher derivatives D k := D k R p coincide with the usual higher-order derivatives in Euclidean space. If g : M → R is a smooth, scalar-valued function, then ∇ k M g ∈ Γ( TM ⊗ T ∗ M ⊗ k -1 ) will be used to denote D k M g with one of the T ∗ M -slots dualised to a TM -slot as in (3). In particular, ∇ M g and ∇ 2 M g are the Riemannian gradient and Riemannian Hessian respectively; they coincide with the usual gradient ∇ g and Hessian ∇ 2 g respectively when M = R p .

Aspecial case of the Levi-Civita connection is that inherited by a submanifold. If M is a submanifold of a Riemannian manifold N , then the Levi-Civita connection associated to the induced metric (22) is D M := P TM D N , where P TM is the orthogonal projection TN → TM induced by the metric on N .

Given ( θ, v ) ∈ TM , classical ordinary differential equation (ODE) theory guarantees the existence of a unique M -valued solution γ to the TM -valued second order ODE

<!-- formula-not-decoded -->

on some interval [0 , ϵ ] . This unique solution is called the geodesic through ( θ, v ) . In particular, for all v ∈ T θ M sufficiently small, γ (1) makes sense and is the value of the exponential map exp θ ( v ) . The exponential map exp θ : B T θ M (0 , R ) → M is invertible onto its image exp θ ( B T θ M (0 , R ) ) ⊂ M for all R &gt; 0 sufficiently small, and its inverse, denoted log θ : exp θ ( B T θ M (0 , R ) ) → B T θ M (0 , R ) , is called the logarithm. When M = R p with the Euclidean metric, exp θ ( v ) is defined for all ( θ, v ) ∈ T R p and is equal to θ + v . The length of a geodesic γ on [0 , b ] is given by ∫ b 0 ∥ ˙ γ ( t ) ∥ dt , and the geodesic distance d M ( θ, θ ′ ) between two points θ, θ ′ ∈ M is the length of the shortest

geodesic connecting θ to θ ′ . Associated to any geodesic γ : [0 , b ] → M and any t ∈ [0 , b ] is a unique orthogonal map Π( γ ) t : T γ (0) M → T γ ( t ) M such that Π( γ ) : t ↦→ Π( γ ) t satisfies the ODE

<!-- formula-not-decoded -->

This map Π( γ ) t is called the parallel transport morphism, and extends to a map T γ (0) M ⊗ k ⊗ T ∗ γ (0) M ⊗ l → T γ ( t ) M ⊗ k ⊗ T ∗ γ ( t ) M ⊗ l of arbitrary tensors. In particular, if θ, θ ′ ∈ M admit a unique geodesic γ : [0 , b ] → M with γ (0) = θ and γ ( b ) = θ ′ , Π( γ ) b : T θ M → T θ ′ M will be denoted simply Π θ → θ ′ .

Tensor fields admit covariant Taylor expansions on M defined as follows. Fix θ ∈ M and suppose that θ ′ ∈ M is sufficiently close to θ that there is a unique geodesic γ : [0 , 1] → M such that γ (0) = θ and γ (1) = θ ′ . Consider then the function ˜ T ( t ) := Π( γ ) -1 t T ( γ ( t )) ∈ T θ M ⊗ k ⊗ T ∗ θ M ⊗ l . Taylor's theorem in a single variable implies that

<!-- formula-not-decoded -->

Since ( D M Π( γ ) ) [ ˙ γ ] = 0 and ( D M ˙ γ )[ ˙ γ ] = 0 , one has:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substituting these formulae into (25)yields the covariant Taylor expansion

<!-- formula-not-decoded -->

## A.2 Riemannian geometry of submanifolds and hypersurfaces in Euclidean space

When M is a hypersurface Euclidean space R p , with normal vector field n , the Hessian of a smooth function takes the following form.

Lemma A.1. Let M be a smooth hypersurface in R p with normal vector field n . The Riemannian Hessian ∇ 2 M λ of a smooth map λ : R p → R along M is the p × p matrix-valued function on M .

<!-- formula-not-decoded -->

Proof. When M is an embedded submanifold of R p , the action of the Levi-Civita connection on a function λ (i.e., the Riemannian gradient operator) is given by P TM ∇ λ , where ∇ is the ordinary Euclidean gradient. The action of the Levi-Civita connection on a vector field X : R p → R p is given by P TM DXP TM . Thus:

<!-- formula-not-decoded -->

where the final line follows from

<!-- formula-not-decoded -->

for any vector v .

<!-- formula-not-decoded -->

We will be particularly interested in the case of a hypersurface M = f -1 { y } ⊂ R p , where f : R p → R is a smooth submersion, with metric g inherited from the ambient Euclidean space. Specifically, when n is the unit normal vector field defined by

<!-- formula-not-decoded -->

the Riemannian metric g on M is defined by

<!-- formula-not-decoded -->

where we use P TM ( θ ) to denote the projection onto T θ M . We will be particularly interested in computing the Riemannian Hessian of the function λ := ∥∇ f ∥ 2 , which admits the following formula entirely in terms of f and its derivatives.

Lemma A.2. For a hypersurface M := f -1 { y } of Euclidean space defined by a smooth function f : R p → R , the Riemannian Hessian of λ := ∥∇ f ∥ 2 is given by

<!-- formula-not-decoded -->

where ∇ 3 f is regarded as a map R p → R p × p , or as the map sending a direction vector v to the directional derivative of the Hessian matrix ∇ 2 f in the direction v .

Proof. By Lemma A.1, we have that

<!-- formula-not-decoded -->

so it remains only to compute the various quantities. The following identities are elementary to verify:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

The result now follows.

Our next lemma allows us to lower-bound the convexity radius conv ( θ ) at θ ∈ M which is the largest r &gt; 0 such that the geodesic ball B M ( θ, r ) is geodesically convex. This will be used in verifying Assumption 3.4 for multilayer scalar factorisation. If M is a hypersurface in R p with normal vector field n , then the second fundamental form of M is the map h : TM ⊗ TM → R defined by

<!-- formula-not-decoded -->

Lemma A.3. Let M be a smooth hypersurface in R p with normal vector field n , and fix θ ∗ ∈ M . Assume that for all R &gt; 0 , there exists C R &gt; 0 such that

<!-- formula-not-decoded -->

for all θ ∈ B M ( θ ∗ , 2 R ) and all orthonormal vectors u, v ∈ T θ M . Then for all R &gt; 0 , one has

<!-- formula-not-decoded -->

Proof. The bound follows from a well-known relationship between the convexity radius and the injectivity radius , which, at a point θ ∈ M , is the largest r &gt; 0 such that exp θ : B T θ M (0 , r ) → M is injective. We thus first bound the injectivity radius. Fix R &gt; 0 . By the Gauss equation and the hypothesis, at any point θ ∈ B ( θ ∗ , 2 R ) , the sectional curvature K ( θ ) of M satisfies

<!-- formula-not-decoded -->

for all unit, orthogonal pairs u, v ∈ T θ M . Consequently, by Klingenberg's formula [37, Lemma 6.4.7], the injectivity radius inj ( θ ) at θ , namely the largest value of r such that exp θ : B T θ M (0 , r ) → M is injective, satisfies

<!-- formula-not-decoded -->

for all θ ∈ M and v, w ∈ T θ M .

where L ( θ ) is the length of the shortest geodesic loop based at θ . If L ( θ ) ≥ 2 R , then this simply yields inj ≥ min { π/C R , R } . If L ( θ ) &lt; 2 R , then the shortest geodesic loop γ : [0 , L ( γ )] → M at θ is contained entirely in B ( θ ∗ , 2 R ) . Fenchel's theorem then combines with the hypothesis to give

<!-- formula-not-decoded -->

so that (1 / ) L ( θ ) ≥ π/C R . Consequently, one has the bound in general.

Now, setting

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

one sees that the hypotheses of [37, Theorem 6.4.8] hold, implying that B ( θ ∗ , r R ) is geodesically convex.

## B Case study: deep scalar factorisation

In this section, we verify that all of our assumptions are satisfied for deep scalar factorisation, where f : R p → R is defined by

For θ ∈ R p having all entries nonzero, it will be convenient to denote

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We first demonstrate that Assumption 3.1 is satisfied.

Proposition B.1. Any y = 0 is a regular value of f . Consequently, Assumption 3.1 is satisfied.

̸

̸

Proof. Since y = 0 , any point θ ∥ ∈ f -1 { y } has all coordinates being nonzero. Hence Df ( θ ∥ ) = y v ( θ ∥ ) = 0 making y a regular value of f .

̸

̸

The pre-image manifold f -1 { y } of any y = 0 has several connected components. Without loss of generality, we will assume from hereon that y &gt; 0 and denote by M the component of f -1 { y } contained in the positive orthant. We next demonstrate that Assumption 3.2 is satisfied.

Proposition B.2. Set θ ∥ ∗ := y 1 /p 1 p , where 1 p is the vector of 1s. Then the line spanned by the normal vector n ( θ ∥ ∗ ) is invariant under gradient descent on ℓ = (1 / 2)( f -y ) 2 for any η , so that Assumption 3.2 is satisfied.

Proof. At θ ∥ ∗ := y 1 /p 1 p , one has n ( θ ∥ ∗ ) = p -1 / 2 1 p . For any α ∈ R , one sees that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is a multiple of n ( θ ∥ ∗ ) , implying that the span of n ( θ ∥ ∗ ) is invariant under gradient descent on ℓ .

We will continue to denote θ ∥ ∗ := y 1 /p 1 p , which is often referred to in the literature as the 'balanced" solution. We next come to Assumption 3.3.

Proposition B.3. Assumption 3.3 holds along M . In particular, at θ ∥ ∗ ,

<!-- formula-not-decoded -->

where we use the convention ( p 3 ) = 0 if p &lt; 3 .

Proof. Given symmetric, multilinear functionals A,B taking k and l inputs respectively, let A ⊙ B denote their symmetric product:

<!-- formula-not-decoded -->

where Perm ( k + l ) is the group of permutations of k + l elements. Since ℓ ≡ (1 / 2)( f -y ) 2 , one has Dℓ = ( f -y ) Df ⇒ Dℓ | M = 0 , (56)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

One then sees that

<!-- formula-not-decoded -->

By continuity, it suffices to prove the claim at η = 2 /λ ( θ ∥ ) = 2 / ∥ Df ( θ ∥ ) ∥ 2 ; substituting this into the above yields

<!-- formula-not-decoded -->

One computes

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consequently,

<!-- formula-not-decoded -->

̸

If p = 2 , then, since D 3 f ≡ 0 , Assumption 3.3 trivially holds for all θ ∥ ∈ M and all η close to 2 /λ ( θ ∥ ) . Let us assume then that p ≥ 3 . One has:

<!-- formula-not-decoded -->

Thus one has

<!-- formula-not-decoded -->

Substituting then yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where S i ( θ ∥ ) are the elementary symmetric means in the variables θ -2 l . Newton's inequality for the elementary symmetric means gives S 2 ( θ ∥ ) 2 ≥ S 1 ( θ ∥ ) S 3 ( θ ∥ ) , so that c (2 /λ ( θ ∥ ) , θ ∥ ) &gt; 0 since ( p 2 ) 2 = p 2 ( p -1) 2 / 4 &gt; p 2 ( p -1)( p -2) / 6 = p ( p 3 ) .

To complete the proof, we evaluate at θ ∥ ∗ = y 1 /p 1 p . One has:

<!-- formula-not-decoded -->

from which it follows that

<!-- formula-not-decoded -->

thus yielding the claimed formula.

We finally come to verifying Assumption 3.4. This requires us both to demonstrate the existence of a geodesically convex ball containing θ ∥ ∗ , and to demonstrate that λ is geodesically strongly convex in this ball.

We first give an explicit ball about θ ∥ ∗ which is geodesically convex using Lemma A.3.

Lemma B.4. The geodesic ball of radius

<!-- formula-not-decoded -->

centred on θ ∥ ∗ is geodesically convex.

Proof. That r can be taken to be ∞ if p = 2 follows from the fact that M in this case is a copy of the real line.

<!-- formula-not-decoded -->

Suppose that p ≥ 3 . Observe that for any θ ∥ = ( θ 1 , . . . , θ p ) ∈ M and any u, v ∈ T θ ∥ M , one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

since n ( θ ∥ ) T w = 0 for any vector w ∈ T θ ∥ M . Consequently, for any R ∈ [0 , (1 / 2) y 1 /p ) , since B M ( θ ∥ ∗ , 2 R ) ⊂ B R p ( θ ∥ ∗ , 2 R ) by the distance minimising property of geodesics, if θ ∥ ∈ B M ( θ ∥ ∗ , 2 R ) then θ i ∈ [ y 1 /p -2 R,y 1 /p +2 R ] for all i , so that

<!-- formula-not-decoded -->

By Lemma A.3, for all R ∈ [0 , (1 / 2) y 1 /p ) the ball of radius

<!-- formula-not-decoded -->

is geodesically convex; it thus remains only to optimise this r R . Since R ↦→ R is monotonically increasing and R ↦→ π/C R is monotonically decreasing, it suffices to find the first smallest value of R for which π/C R = R . This equality gives a quadratic whose solution is

<!-- formula-not-decoded -->

from which the result follows.

We next use Lemma A.2 to compute the Riemannian Hessian of λ along M .

Lemma B.5. For θ ∥ = ( θ 1 , . . . , θ p ) ∈ M , define s 1 ( θ ∥ ) := ∑ i θ -2 i and s 2 ( θ ∥ ) := ∑ i θ -4 i . Then the function λ has Riemannian Hessian

<!-- formula-not-decoded -->

In particular, at θ ∥ ∗ ∈ U ,

<!-- formula-not-decoded -->

Moreover, if p = 2 , then λ is 2-geodesically strongly convex over all of M ; if p &gt; 2 , then for any δ ∈ [0 , 2 -√ 3) , λ is 2 y 2 -4 /p (1+ δ ) -4 (1 -δ ) -2 ( 3(1 -δ ) 2 -(1+ δ ) 2 ) -geodesically strongly convex over the closed geodesic-distance ball B M ( θ ∥ ∗ , δy 1 /p ) of radius δy 1 /p about θ ∥ ∗ in M .

Proof. For notational convenience, set V 1 := diag ( v ⊙ 2 ) and V 2 := diag ( v ⊙ 4 ) , where ⊙ denotes the Hadamard (componentwise) product and v is the vector field defined in (51). From the identities (64), (65) and (66), one sees that along M one has

<!-- formula-not-decoded -->

The pointwise-bilinear map ∇ 3 f is zero if p = 2 , and if p &gt; 2 it acts on a vector z to give the matrix with components

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the fourth line follows from the fact that (1 -δ ij ) A ij = ( A -diag ( A )) ij for any matrix A and the fact that diag ( v ( V 1 z ) T ) = diag (( V 1 z ) v T ) = diag ( v ⊙ 3 ⊙ z ) . Substituting z = ∇ f = y v one sees that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

so that, by Lemma A.2,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

if p &gt; 2 and

<!-- formula-not-decoded -->

if p = 2 . Now, let τ be any tangent vector field to M . Since v is normal to M one has v T τ = 0 , so that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

are both multiples of the normal vector v ; it follows that

<!-- formula-not-decoded -->

as claimed. In either case, evaluating this matrix at θ ∥ ∗ = y 1 /p 1 p gives the claimed identity

<!-- formula-not-decoded -->

In particular, if p = 2 , evaluating ∇ 2 M λ on the unit tangent vector field ˆ τ : ( θ 1 , θ 2 ) ↦→ ( -θ 1 , θ 2 ) / √ θ 2 1 + θ 2 2 yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any θ ∥ = ( θ 1 , θ 2 ) ∈ M , so that λ is 2-geodesically strongly convex over all of M as claimed.

We now assume that p &gt; 2 and come to determining a neighbourhood of θ ∥ ∗ on which ∇ 2 M λ is positive definite. It is clear that a sufficient condition for ∇ 2 M λ to be positive-definite is to have all entries of the diagonal matrix 3 V 2 -( s 2 /s 1 ) V 1 be positive; we will show that this sufficient condition can be guaranteed over a geodesic ball in M centred at θ ∥ ∗ . Since the geodesic ball of radius r at a point in M is contained in the Euclidean ball of radius r at the same point by the distance-minimising property of geodesics, it suffices to show that this sufficient condition can be guaranteed over a Euclidean ball centred at θ ∥ ∗ .

Fix δ ∈ [0 , 1) : for any θ = ( θ 1 , . . . , θ p ) in the Euclidean ball B R p ( θ ∥ ∗ , δy 1 /p ) , one has θ i ∈ [(1 -δ ) y 1 /p , (1 + δ ) y 1 /p ] . Consequently, for any θ ∈ B R p ( θ ∥ ∗ , δy 1 /p ) one has

<!-- formula-not-decoded -->

It follows that for all θ ∈ B R p ( θ ∥ ∗ , δy 1 /p ) one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which is strictly greater than zero provided that 3(1 -δ ) 2 -(1 + δ ) 2 &gt; 0 , which holds provided that 0 ≤ δ &lt; 2 -√ 3 . This proves the result.

<!-- formula-not-decoded -->

Note that λ is not globally convex over M in general. For instance, when p = 3 , along the family θ ∥ ( ϵ ) := ( ϵ, 1 , yϵ -1 ) ∈ M , elementary calculations show that ∇ 2 M λ ( θ ∥ ( ϵ )) exhibits negative eigenvalues as ϵ → 0 + .

Combining Lemmas B.4 and B.5 one obtains explicit bounds under which Assumption 3.4 holds for multilayer scalar factorisation.

Proposition B.6. If p = 2 , then λ is 2-geodesically strongly convex over the geodesically convex set M . If p ≥ 3 , then λ is 1 . 33 y 2 -4 /p -geodesically strongly convex over the geodesically convex ball B M ( θ ∥ ∗ , 0 . 15 y 1 /p ) . In all cases, one has

<!-- formula-not-decoded -->

Proof. Numerical computation reveals that the radius of Lemma B.4 is lower-bounded by 0 . 15 y 1 /p . Substituting δ = 0 . 15 into the strong convexity bound of Lemma B.5 and rounding down to two decimal places yields the claimed result.

We conclude the section by proving that the additional hypothesis required for Theorem 5.2 is satisfied.

Proposition B.7. For any p ≥ 2 , with ν = 4 y 2 -4 /p as in Proposition B.6, one has

<!-- formula-not-decoded -->

so that the additional hypothesis of Theorem 5.2 holds.

Proof. Since λ ( θ ∥ ∗ ) = ∥ Df ( θ ∥ ∗ ) ∥ 2 = py 2 -2 /p , one calculates using Proposition B.3 that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any p ≥ 2 . The result follows.

## C The normal form of gradient descent

To prove Theorem 4.1 rigorously requires a number of steps. First , one must give leading order formulae for the expression of gradient descent in tubular neighbourhood coordinates. It is essential here that the errors in these leading order formulae are appropriately sharpened from their naive estimates using the structure of the tubular neighbourhood map (Lemma C.2) and the orthogonal stability assumption (Lemma C.4); without this sharpening our convergence theorems are impossible. These considerations culminate in an intermediate coordinate expression for gradient descent in Proposition C.5. Second , one must invoke Assumption 3.3 to transform the ⊥ -dynamics of Proposition C.5 into the easily-analysed normal form quoted in Theorem 4.1. This is formalised in Theorem C.6 and concludes this section.

Throughout this section, given functions g, h 1 , . . . , h s : R k → R l we will use the notation

<!-- formula-not-decoded -->

to mean that there exist constants C 1 , . . . , C s &gt; 0 such that

<!-- formula-not-decoded -->

for all w sufficiently close to zero.

In addition to this asymptotic notation, we will continue the notation of the main body of the paper, denoting by M := f -1 { y } the submanifold of R p obtained as the preimage of a regular value y ∈ R of an analytic map f : R p → R . We will also denote by n := ∇ f/ ∥∇ f ∥ the unit normal along M . Assumptions 3.1, 3.2 and 3.3 are assumed in all of what follows. Assumption 3.4 is not needed for this section.

Consider the map E : M × R → R p defined by

Observe that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where I TM ( θ ∥ ) : T θ ∥ M → T θ ∥ M is the identity map. The derivative DE ( θ ∥ , θ ⊥ ) acts as the linear map

<!-- formula-not-decoded -->

between tangent spaces, and can equivalently be written in terms of the shape operator S ( θ ∥ ) : T θ ∥ M ∋ v ↦→-Dn ( θ ∥ )[ v ] ∈ T θ M as

Observe that since n ( θ ∥ ) is orthogonal to T θ ∥ M , DE ( θ ∥ , θ ⊥ ) is invertible whenever | θ ⊥ | &lt; ∥ S ( θ ∥ ) ∥ -1 , where ∥ S ( θ ∥ ) ∥ denotes the operator norm of S ( θ ∥ ) : T θ ∥ M → T θ ∥ M with respect to the Riemannian metric. Applying the inverse function theorem proves the following.

<!-- formula-not-decoded -->

Proposition C.1. The map E : M × R → R p is invertible on an open neighbourhood U of M ×{ 0 } . The image of U in R p is a tubular neighbourhood N of M , and we denote the inverse of E thereon by E -1 .

The coordinates E : ( θ ∥ , θ ⊥ ) ↦→ E ( θ ∥ , θ ⊥ ) are called tubular neighbourhood coordinates . The next step is to derive a formula for E -1 and thereafter derive a formula for the conjugate of gradient descent in the coordinates ( θ ∥ , θ ⊥ ) induced by E . The following technical lemma will be necessary to get the correct asymptotics for the error terms in our formula for E -1 .

Lemma C.2. Let U ⊂ R m 1 × R m 2 be an open set and let g : U ∋ ( x, y ) ↦→ g ( x, y ) ∈ R p be a C 3 function. Assume that D k y g ≡ 0 on U for k = 2 , 3 and that g is invertible on U . Then for any ( x, y ) ∈ U and all v ∈ R p sufficiently small, setting ( v x , v y ) = Dg ( x, y ) -1 [ v ] and z := g ( x, y ) , the point z + v is contained in the domain of g -1 and one has

<!-- formula-not-decoded -->

Proof. Since g is invertible on U and U is open, g ( U ) = domain ( g -1 ) is also open. Fix ( x, y ) ∈ U and set z := g ( x, y ) . Then since g ( U ) is open, z + v ∈ g ( U ) for all sufficiently small v so that g -1 ( z + v ) makes sense. Moreover, for all such v , there exists u ∈ R m 1 × R m 2 such that g (( x, y ) + u ) = z + v so that g -1 ( z + v ) = ( x, y ) + u = g -1 ( z ) + u . Hence one has the following formula for the remainder R ( v ) of the Taylor expansion of g -1 ( z + v ) about z :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To bound R ( v ) , therefore, we seek a formula u ( v ) for u in terms of v .

From g (( x, y ) + u ) = z + v , one obtains

<!-- formula-not-decoded -->

where the remainder ˜ u ↦→ S (˜ u ) satisfies S (˜ u ) = O ( ∥ ˜ u x ∥ 3 , ∥ ˜ u x ∥ 2 ∥ ˜ u y ∥ ) since D k y g ≡ 0 for k = 2 , 3 by hypothesis. Since Dg ( x, y ) is invertible by the invertibility of g on U , G is smoothly invertible on a neighbourhood V of 0 by the inverse function theorem. Consider now the second-order approximation

<!-- formula-not-decoded -->

to u . Since u ′ = O ( ∥ v ∥ ) , taking v smaller if necessary we are guaranteed that u ′ ∈ V . Thus there is C &gt; 0 such that

<!-- formula-not-decoded -->

We will show that the right side of (130) admits a sufficiently tight bound b ( v ) that u ( v ) = u ′ ( v ) + O ( b ( v )) suffices to give the desired decay in R ( v ) .

For notational convenience, set w := (1 / 2) Dg ( x, y ) -1 D 2 g ( x, y ) [ Dg ( x, y ) -1 [ v ] ⊗ 2 ] . Then u ′ = Dg ( x, y ) -1 [ v ] -w , and substituting this expression into (128) yields

<!-- formula-not-decoded -->

Since D 2 y g ≡ 0 by hypothesis, one has w = O ( ∥ v x ∥ 2 , ∥ v x ∥∥ v y ∥ ) as well as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, applying the chain rule to the equation g -1 ◦ g = Identity shows that Dg -1 ( z ) = Dg ( x, y ) -1 and D 2 g -1 ( z ) = -Dg ( x, y ) -1 D 2 g ( x, y )[ Dg ( x, y ) -1 , Dg ( x, y ) -1 ] . It thus follows from substituting (137) into (127) that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

as claimed.

and

Thus, by (130),

A formula can now be given for E -1 using a Taylor expansion as follows.

Proposition C.3. For all θ ∥ ∈ M and all v = v ∥ + v ⊥ n ( θ ∥ ) ∈ T θ ∥ M ⊕ span ( n ( θ ∥ )) = T θ ∥ R p sufficiently small, one has

<!-- formula-not-decoded -->

Proof. The result follows from a second-order Taylor expansion with remainder:

<!-- formula-not-decoded -->

Since the θ ⊥ -derivatives of E ( θ ∥ , θ ⊥ ) = θ ∥ + θ ⊥ n ( θ ∥ ) vanish for all orders &gt; 1 , by Lemma C.2 error ( θ ∥ , v ) , satisfies

<!-- formula-not-decoded -->

It thus remains only to calculate the leading order terms. One easily computes

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any ( θ ∥ , θ ⊥ ) ∈ M × R and ( v ∥ , v ⊥ ) ∈ T θ ∥ M × R . Recalling that E is invertible when restricted to an open neighbourhood U of M × { 0 } in M × R , by differentiating the identity E -1 ◦ E = Identity U ⊂ M × R one computes

<!-- formula-not-decoded -->

where P TM ( θ ∥ ) : T θ ∥ R p → T θ ∥ M is the orthogonal projection, so that

<!-- formula-not-decoded -->

Again differentiating the identity E -1 ◦ E = Identity U yields

<!-- formula-not-decoded -->

so that from (148) one obtains

<!-- formula-not-decoded -->

with the final equality being a consequence of n T Dn = -( Dn ) T n = -n T Dn = 0 as follows from differentiating the identity n T n = 1 . The result follows.

The next result characterises the scaling of the ∥ -component of ∇ ℓ in a neighbourhood of θ ∥ ∗ . It is essential for obtaining the correct asymptotic decay in the error terms.

Lemma C.4. Assumption 3.2 implies that for all θ ∥ ∈ M sufficiently close to θ ∥ ∗ and all θ ⊥ ∈ R sufficiently small, one has P TM ( θ ∥ ) ∇ ℓ ( E ( θ ∥ , θ ⊥ ) ) = O ( d M ( θ ∥ , θ ∥ ∗ )) , where P TM ( θ ∥ ) : T θ ∥ R p → T θ ∥ M is the orthogonal projection onto T θ ∥ M .

Proof. By Assumption 3.2, one has

<!-- formula-not-decoded -->

for all θ ⊥ ∈ R sufficiently small. Since for all sufficiently small θ ⊥ the map θ ∥ ↦→ P TM ( θ ∥ ) ∇ ℓ ( E ( θ ∥ , θ ⊥ ) ) is smooth, the result follows.

We are now able to give the tubular neighbourhood coordinate expression for gradient descent.

Proposition C.5. In the tubular neighbourhood coordinates E : ( θ ∥ , θ ⊥ ) ↦→ E ( θ ∥ , θ ⊥ ) for a tubular neighbourhood of M , the gradient descent map GD ℓ R p ( η, · ) : θ ↦→ θ -η ∇ ℓ ( θ ) is conjugated to the analytic map

<!-- formula-not-decoded -->

Proof. Since f is analytic by Assumption 3.1, so too are GD ℓ R p ( η, · ) and E and hence, since the inverse function theorem preserves analyticity, so too is E -1 . Hence the conjugate E -1 ◦ GD ℓ R p ( η, · ) ◦ E is analytic wherever defined. Recalling the open neighbourhood U of M ×{ 0 } in M × R from Proposition C.1 on which E is a diffeomorphism, given ( θ ∥ , θ ⊥ ) ∈ U

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Applying Proposition C.3 with v = θ ⊥ n ( θ ∥ ) -η ∇ ℓ ( θ ∥ + θ ⊥ n ( θ ∥ )) , one then obtains

<!-- formula-not-decoded -->

Note the crucial role played by the estimate on the error in Proposition C.3 here: were there to be a pure O ( ∥ v ⊥ ∥ k ) term in the error for some k ≥ 1 , then the same error term would have appeared in the the formula (157) and, as we shall now see, would have prevented the ∥ -component of the dynamics from having an O ( d M ( θ ∥ , θ ∥ ∗ )) error term.

We now compute formulae for v ∥ and v ⊥ . Using the fact that ∇ ℓ ( θ ∥ ) = 0 , one has the Taylor expansion

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second line follows from the fact that n ( θ ∥ ) is an eigenvector of ∇ 2 ℓ ( θ ∥ ) with eigenvalue λ ( θ ∥ ) . Taking the inner-product of (159) with n ( θ ∥ ) then reveals that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

One obtains v ∥ by projecting (159) onto T θ ∥ M :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second line follows from Lemma C.4, the third line follows from the definition ∇ M = P TM ∇ of the Levi-Civita connection on M ⊂ R p , and the final line follows from the identity ∇ 2 ℓ [ n, n ] ≡ λ that holds along M . Substituting these expressions into (157) yields the result.

By finally invoking Assumption 3.3, we may now transform the tubular neighbourhood coordinate expression of gradient descent from Proposition C.5 into the normal form expression of Theorem 4.1.

Theorem C.6. Recall the function c : R × M → R &gt; 0

<!-- formula-not-decoded -->

from Assumption 3.3. There is an open neighbourhood W of { (2 /λ ( θ ∥ ) , θ ∥ , 0) : θ ∥ ∈ M } in R × M × R on which φ : W → R × M × R defined by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is an analytic diffeomorphism onto its image. Moreover, the composite ( I R × E ) ◦ φ : W → R p × R , where I R is the identity map, conjugates ( I R , GD ℓ R p ) : R × R p → R × R p to the map

̸

Proof. That an open neighbourhood W of { (2 /λ ( θ ∥ ) , θ ∥ , 0) : θ ∥ ∈ M } exists on which φ is a diffeomorphism follows from the fact that c ( η, θ ∥ ) -1 / 2 = 0 for all θ ∥ ∈ M and η in a neighbourhood of 2 /λ ( θ ∥ ) and the inverse function theorem. That φ is analytic is a consequence of the analyticity of f and the functions used in defining φ in terms of ℓ = (1 / 2)( f -y ) 2 .

The formula (169) pertains to the conjugation

<!-- formula-not-decoded -->

of GD ℓ R p . Proposition C.5 already gives us a formula for ( I R × E -1 ) ◦ GD ℓ R p ◦ ( I R × E ) , so it suffices merely to conjugate this formula by φ .

Since φ ∥ ( η, θ ∥ , θ ⊥ ) = θ ∥ , the ∥ -component of (170) is given by substituting φ ⊥ ( η, θ ∥ , θ ⊥ ) in place of θ ⊥ in the ∥ -component of (154), which yields the map

<!-- formula-not-decoded -->

As demonstrated in [33], however, one has

<!-- formula-not-decoded -->

since ∇ M λ ( θ ∥ ∗ ) = 0 . Thus the ∥ -component of (170)may be written as

<!-- formula-not-decoded -->

as claimed.

Finally, the ⊥ -component of (169) follows from the same argument used in [27, Theorem 4.3].

## D Convergence theorems

The purpose of this section is to provide proofs of the main convergence theorems presented in the paper. In addition to assuming Assumptions 3.1, 3.2 and 3.3 as in the previous section, in this section we will also assume Assumption 3.4, which states that there is a geodesically convex ball B M ( θ ∥ ∗ , r ) centred on θ ∥ ∗ over which λ is geodesically smooth, geodesically strongly convex, and has unique global minimum at θ ∥ ∗ with Riemannian Hessian at θ ∥ ∗ being a multiple of the identity.

## D.1 Subcritical regime

Our convergence theorem in the subcritical regime has two components. First, it is proved that gradient descent implicitly descends λ ( θ ∥ t ) from being initially &gt; 2 /η to eventually being &lt; 2 /η . For this, it is necessary to have a descent lemma for the perturbed Riemannian gradient descent component of Theorem 4.1 (Lemma D.1) as well as upper and lower bounds on | θ ⊥ t | during this phase (Lemma D.3). Following this transient phase, convergence is easily proved using the fact that | θ ⊥ t | contracts exponentially to zero.

Lemma D.1 (Descent lemma for perturbed Riemannian gradient descent) . For the µ -geodesically strongly convex, L -geodesically smooth function λ on the geodesically convex ball B M ( θ ∥ ∗ , r ) centred on θ ∥ ∗ in M , consider the map

<!-- formula-not-decoded -->

of Theorem C.6. Then for any η &gt; 0 and all θ ⊥ sufficiently close to zero, GD ∥ ( η, · , θ ⊥ ) maps B M ( θ ∥ ∗ , r ) into B M ( θ ∥ ∗ , r ) . Moreover, for any η &gt; 0 , any θ ∥ ∈ B M ( θ ∥ ∗ , r ) and all θ ⊥ sufficiently close to zero one has

<!-- formula-not-decoded -->

Proof. For θ ⊥ sufficiently small, there is unique δ ( θ ∥ , θ ⊥ ) = O ( ( θ ⊥ ) 3 min { 1 , d M ( θ ∥ , θ ∥ ∗ ) } ) such that

<!-- formula-not-decoded -->

To see that GD ∥ ( η, · , θ ⊥ ) preserves the ball B M ( θ ∥ ∗ , r ) , consider the function ρ ( θ ∥ ) := (1 / 2) d M ( θ ∥ , θ ∥ ∗ ) 2 . By the Gauss lemma [29, Corollary 6.10], ∇ M ρ ( θ ∥ ) = -log θ ∥ ( θ ∥ ∗ ) . Fixing θ ∥ ∈ B M ( θ ∥ ∗ , r ) , for any vector w ∈ T θ ∥ M one therefore has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the third line follows from Lemma E.2. Thus, for all θ ⊥ sufficiently small, we are assured that ρ ( GD ∥ ( η, θ ∥ , θ ⊥ ) ) &lt; ρ ( θ ∥ ) , implying that GD ∥ ( η, θ ∥ , θ ⊥ ) ∈ B M ( θ ∥ ∗ , r ) .

We now come to the descent lemma. Since B M ( θ ∥ ∗ , r ) is geodesically convex, we may perform a covariant Taylor expansion and invoke the geodesic smoothness of λ to obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From Cauchy-Schwarz and the estimate 0 ≤ ( ϵ 1 / 2 a -ϵ -1 / 2 b ) 2 (implying that ab ≤ (1 / 2)( ϵa 2 + ϵ -1 b 2 ) ) with ϵ = η ( θ ⊥ ) 2 / (4 c ( η, θ ∥ )) , one obtains

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking | θ ⊥ | sufficiently small that η ( θ ⊥ ) 2 L/ (4 c ( η, θ ∥ )) ≤ 1 / 4 , therefore, one obtains

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Subtracting λ ( θ ∥ ∗ ) from both sides and applying geodesic strong convexity ( ∥∇ M λ ( θ ∥ ) ∥ 2 ≥ 2 µ ( λ ( θ ∥ ) -λ ( θ ∥ ∗ )) by Lemma E.3) yields

<!-- formula-not-decoded -->

Finally, invoking geodesic strong convexity once more to estimate d M ( θ ∥ , θ ∥ ∗ ) 2 ≤ (2 /µ ) ( λ ( θ ∥ ) -λ ( θ ∥ ∗ ) ) as in Lemma E.1 and taking θ ⊥ sufficiently small allows one to absorb the remainder term on the left side into the λ ( θ ∥ ) -λ ( θ ∥ ∗ ) factor and yields the result..

The next result will be used in giving uniform bounds on the magnitudes of the θ ⊥ iterates.

Lemma D.2. For α ∈ R , define f α : R → R by

<!-- formula-not-decoded -->

For α 0 , α 1 ∈ R , consider the composite f α 1 α 0 := f α 1 ◦ f α 0 . Then for all γ &gt; 0 sufficiently small and all α 0 , α 1 ∈ [0 , γ ] :

1. f α 1 α 0 is monotonically increasing on [ -2 √ γ, 2 √ γ ] .
2. f α 1 α 0 admits the sole fixed points 0 ,

<!-- formula-not-decoded -->

in the interval [ -2 √ γ, 2 √ γ ] .

Proof. For γ &gt; 0 small and all α 0 , α 1 ∈ [0 , γ ] , one computes

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In particular, for all | x | ≤ 2 √ γ , one has

<!-- formula-not-decoded -->

for all γ sufficiently small, so that f α 1 α 0 | [ -2 √ γ, 2 √ γ ] is monotonically increasing. Set β := ( α 0 + α 1 ) / 2 for notational convenience, and denote

<!-- formula-not-decoded -->

Fixed points of f α 1 α 0 are precisely roots of Φ , one of which is clearly x = 0 . Moreover, if β = 0 , then Φ( x ) = -2 x 3 + O ( x 4 ) so that in this case x = 0 is the only root of Φ on [ -2 √ γ, 2 √ γ ] for all γ sufficiently small. Suppose now that β &gt; 0 . We will demonstrate the existence of a unique root ξ + of Φ in (0 , 2 √ γ ] with the claimed formula. Since 2 βx -2 x 3 &gt; 0 for all 0 &lt; x ≤ 2 -1 / 2 √ β , we see that for all γ sufficiently small, Φ( x ) is strictly positive on (0 , 2 -1 / 2 √ β ] and so cannot admit any roots therein. However, observe that for all γ sufficiently small one has

<!-- formula-not-decoded -->

for all 2 -1 / 2 √ β ≤ x ≤ 2 √ γ , implying that Φ is monotonically decreasing on [2 -1 / 2 √ β, 2 √ γ ] , and thus admits at most one root therein. Since moreover Φ(2 -1 / 2 √ β ) &gt; 0 and Φ(2 √ β ) = -12 β 3 / 2 + O ( β 2 ) &lt; 0 , by the intermediate value theorem Φ admits precisely one root ξ + in [2 -1 / 2 √ β, 2 √ γ ] . By the mean value theorem, there is z between √ β and ξ + ≤ 2 √ β such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so that, since | D Φ( z ) | ≥ 4 β + O ( β 3 / 2 ) = Ω( β ) and Φ( √ β ) = O ( β 2 ) , as claimed. A similar argument can be used to show there exists a unique root ξ -= - √ β + O ( β ) of Φ in [ -2 √ γ, 0) , thus proving the lemma.

Lemma D.3 (Upper and lower bounds for T ⊥ iterates) . Let { λ t } t ∈ N be a monotonically decreasing sequence of numbers. Then for all η &gt; 0 such that ηλ 0 &gt; 2 is sufficiently small, there is a constant C &gt; 0 such that for any θ ⊥ 0 sufficiently small, the iterates defined by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all t ∈ N such that ηλ t ≥ 2 .

Proof. We prove the upper bound first. For each s ∈ N such that ηλ s &gt; 2 , denote α s := ηλ s -2 and denote by f s the map

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Supposing that s ∈ N is such that both ηλ s , ηλ s +1 ≥ 2 , since { λ t } t ∈ N is monotonically decreasing Lemma D.2 guarantees that f s +1 ,s := f s +1 ◦ f s admits attractive fixed points ξ s +1 ,s, ± = ± √ ( α s +1 + α s ) / 2 + O ( α s +1 + α s ) and is monotonically increasing on [ -2 √ α 0 , 2 √ α 0 ] provided α 0 = ηλ 0 -2 is sufficiently small. Suppose now that θ ⊥ 0 ∈ [ -2 √ α 0 , 2 √ α 0 ] \ { 0 } . The iterates θ ⊥ t as defined in (199) satisfy

<!-- formula-not-decoded -->

satisfy the bounds

Assuming without loss of generality that θ ⊥ 0 &gt; 0 , we will prove by induction that θ ⊥ 2 t ≤ 2 √ α 0 for all t ∈ N such that α s ≥ 0 for all s ≤ 2 t -1 . Clearly θ ⊥ 0 ≤ 2 √ α 0 by assumption. Suppose as an inductive hypothesis that 0 &lt; θ ⊥ 2 t ≤ 2 √ α 0 and that α 2 t +1 , α 2 t ≥ 0 . Using the monotonicity of f 2 t +1 , 2 t from Lemma D.2 and the fact that ξ 2 t +1 , 2 t, + is its sole fixed point in (0 , 2 √ α 0 ] , either

<!-- formula-not-decoded -->

or

<!-- formula-not-decoded -->

by the inductive hypothesis; in either case θ ⊥ 2 t +2 ≤ 2 √ α 0 , thus proving the claim. A symmetric argument shows that the odd iterates satisfy | θ ⊥ 2 t +1 | ≤ 2 √ α 0 for all t ∈ N such that α s ≥ 0 for all s ≤ 2 t , thus proving the upper-bound in (200).

We now prove the lower bound. Invoking the upper bound, α 0 can be taken sufficiently small such that the recursive estimate

<!-- formula-not-decoded -->

holds as long as α t ≥ 0 . By taking α 0 even smaller if necessary, squaring, taking the reciprocal of both sides and applying 1 / (1 -x ) 2 ≤ 1 + 3 x for x sufficiently small one deduces that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

from which it follows by a telescoping sum that

<!-- formula-not-decoded -->

for all t such that α t ≥ 0 . This proves the claim.

Theorem D.4. Assume that η &lt; 2 /λ ( θ ∥ ∗ ) , and let 0 &lt; c, C &lt; ∞ satisfy c ≤ c ( η, θ ∥ ) ≤ C for all θ ∥ ∈ B M ( θ ∥ ∗ , r ) . Then for all θ ⊥ 0 sufficiently close to zero and all θ ∥ 0 = θ ∥ ∗ ∈ B M ( θ ∥ ∗ , r ) , if η &gt; 2 /λ ( θ ∥ 0 ) is sufficiently small, then there is τ ∈ N , with

̸

<!-- formula-not-decoded -->

such that η &lt; 2 /λ ( θ ∥ t ) for all t ≥ τ and η ≥ 2 /λ ( θ ∥ t ) for all t &lt; τ . Consequently, setting β := 1 -(2 -ηλ ( θ ∥ τ )) &lt; 1 , the iterates ( θ ∥ t , θ ⊥ t ) converge to a global minimum ( θ ∥ ∞ , 0) of ℓ in M with suboptimal sharpness at a rate O ( β t ) .

<!-- formula-not-decoded -->

̸

Proof. For any θ ∥ 0 = θ ∥ ∗ , take θ ⊥ 0 = 0 sufficiently close to zero that the hypotheses of Lemma D.1 and Lemma D.3 are satisfied, with λ 0 in Lemma D.3 taken to be equal to λ ( θ ∥ 0 ) , and assume that ηλ 0 &gt; 0 is sufficiently small as in Lemma D.3. Then we may assume that λ t := λ ( θ ∥ t ) is monotonically decreasing and the bounds of Lemma D.3 apply to θ ⊥ t . Since { λ t } t ∈ N is monotonically decreasing, any τ satisfying η &lt; 2 /λ ( θ ∥ t ) for all t ≥ τ and η ≥ 2 /λ ( θ ∥ t ) for all t &lt; τ is necessarily unique. We claim that this τ satisfies the upper bound τ ≤ τ ( θ ∥ 0 , θ ⊥ 0 ) . Indeed, suppose for a contradiction that τ &gt; τ ( θ ⊥ 0 , θ ∥ 0 ) . Then for all t &lt; τ , denoting λ ∗ := λ ( θ ∥ ∗ ) , one would have

̸

<!-- formula-not-decoded -->

by Lemmas D.1 and D.3. Invoking c ( η, θ ∥ t ) ≤ C , taking logarithms, applying the upper-bound ln(1 -x ) ≤ -x and applying a telescoping sum then yield

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now come to the final convergence and sharpness suboptimality claim. For all t ≥ τ , Lemma D.1 continues to imply that λ t &lt; 2 /η decreases monotonically, while θ ⊥ t contracts exponentially according to the estimate by Lemma E.4, so that

However, plugging in t = τ ( θ ⊥ 0 , θ ∥ 0 ) then reveals that λ t &lt; 2 /η , thus contradicting the assumption that τ &gt; τ ( θ ⊥ 0 , θ ∥ 0 ) and implying that τ ≤ τ ( θ ⊥ 0 , θ ∥ 0 ) as claimed.

<!-- formula-not-decoded -->

This proves the claim that the iterates ( θ ∥ t , θ ⊥ t ) converge to a global minimum θ ∥ ∞ of ℓ at rate O ( (1 -(2 -ηλ τ )) t ) . We now turn to proving the suboptimality of the sharpness at θ ∥ ∞ . The geodesic strong convexity of λ in the form of Lemma E.2 implies that

<!-- formula-not-decoded -->

for any θ ∥ ∈ B M ( θ ∥ ∗ , r ) , so that the ∥ -update can be written

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consequently, invoking the geodesic strong convexity of λ once again in the form of Lemma E.1,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for sufficiently small 2 √ ηλ 0 -2 &gt; | θ ⊥ t | . Subtracting λ ∗ from both sides and invoking the Lipschitz gradients ( ∥∇ M λ ( θ ∥ ) ∥ ≤ Ld M ( θ ∥ , θ ∥ ∗ ) ) and strong convexity ( ( µ/ 2) d M ( θ ∥ , θ ∥ ∗ ) 2 ≤ λ ( θ ∥ ) -λ ∗ from Lemma E.1) properties of λ then yield

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in the final line we have invoked c ( η, θ ∥ t ) ≥ c . It follows that

<!-- formula-not-decoded -->

Taking 2 √ ηλ 0 -2 &gt; | θ ⊥ τ | smaller if necessary and using the bound ln(1 -x ) ≥ -2 x valid for small x one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

from which the result follows.

## D.2 Critical regime

The convergence theorem in the critical regime is the most difficult to prove. This is because in this regime, the fixed point to which the iterates converge is merely parabolic rather than hyperbolic as is (at least asymptotically) true in the subcritical and supercritical regimes. The convergence theorem in the critical regime will follow from a careful analysis of the following parabolic system.

Fix 0 &lt; a &lt; c and b &gt; 0 . Let A ( x, y ) = O ( x, y ) , B ( x ) = O ( x ) and C ( y ) = O ( y ) be analytic functions, and consider the analytic functions

<!-- formula-not-decoded -->

Denote by T = ( T x , T y ) : R 2 → R 2 the corresponding analytic map. Clearly 0 is a fixed point of T ; we will be concerned with proving convergence of the iterates of T to this fixed point. The following result guarantees that T admits an invariant manifold approaching the origin. This result is essential, since our final convergence proof is a two-stage proof; first, convergence to a neighbourhood of the invariant manifold is proved, following which, second, convergence to the origin is easy.

Lemma D.5. There exists r &gt; 0 and a function u : { x &gt; 0 : | x 2 -r | &lt; r } → R whose graph Γ u := { ( x, u ( x )) : x ∈ dom ( u ) } is invariant under T , and for which u ( x ) = √ b c -a x + O ( x 2 ) .

Proof. The analytic map T : R 2 → R 2 extends uniquely to a holomorphic map ˜ T defined on a neighbourhood of 0 in C 2 . Like T , at the origin the map ˜ T is the identity to first order, vanishes at order two, and has third order Taylor coefficients given by

<!-- formula-not-decoded -->

with all other third order coefficients being zero. Observe that on the vector v given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

one has

<!-- formula-not-decoded -->

so that v is a non-degenerate characteristic direction of ˜ T [6, Definition 4.1], which is moreover attracting since -6 a √ b/ ( c -a ) &lt; 0 . By [6, Section 6], therefore, there exists r &gt; 0 and a holomorphic function u : { x ∈ C : | x 2 -r | &lt; r } → C such that u ( x ) = √ b c -a x + O ( x 2 ) whose graph is invariant under T . Restricting this u to the intersection of its domain with the positive real axis yields the result.

The next lemma will be key in establishing convergence to the invariant manifold.

Lemma D.6. Let u ( x ) = √ b c -a x + O ( x 2 ) be the function from Lemma D.5, and define ∆( x, y ) := y -u ( x ) . Then ˜ ∆( x, y ) := ∆( x, y ) /x satisfies

<!-- formula-not-decoded -->

for all sufficiently small x &gt; 0 , y ≥ 0 .

Proof. We estimate ˜ ∆ ◦ T ˜ ∆ . Define

<!-- formula-not-decoded -->

Observe that both D and N are analytic on the domain of u . Observe moreover that

<!-- formula-not-decoded -->

so that estimation of ˜ ∆ ◦ T ˜ ∆ is reduced to estimating D -N D . We first estimate D -N . For this, observe that since T ( x, 0) = ( x, 0) , one has D ( x, 0) -N ( x, 0) = 0 for all x ∈ dom ( u ) . Thus, by analyticity of D and N , there is a unique analytic function H on dom ( u ) × R such that

<!-- formula-not-decoded -->

We compute H ( x, y ) to second order. For notational convenience, set

<!-- formula-not-decoded -->

so that u ( x ) = κx + O ( x 2 ) . Differentiate (239) with respect to y and evaluate at y = 0 to obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

while, from N = (∆ ◦ T ) / ∆ , one has

<!-- formula-not-decoded -->

From ∆( x, y ) = y -u ( x ) , is easily seen that

<!-- formula-not-decoded -->

while

<!-- formula-not-decoded -->

so that (243) yields

<!-- formula-not-decoded -->

To compute the second order component of H , differentiate (239) twice with respect to y and evaluate at y = 0 to obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Observe that

One sees that

while

<!-- formula-not-decoded -->

Since

<!-- formula-not-decoded -->

one obtains

Thus

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We deduce from (246), (253) and (239) that

<!-- formula-not-decoded -->

It follows that for all sufficiently small x &gt; 0 , y ≥ 0 , one has D ( x, y ) -N ( x, y ) ≥ ( c -a ) 2 y 2 and 1 ≥ D ( x, y ) &gt; 0 so that

<!-- formula-not-decoded -->

as claimed.

Lemma D.7 below gives uniform upper-bounds on the magnitudes of iterates, which are used in enabling certain estimates in the convergence theorem.

Lemma D.7. Define W ( x, y ) := bx 2 + a 2 y 2 . Then for all ( x 0 , y 0 ) in a sufficiently small neighbourhood of 0 , with y 0 &gt; 0 , the sequence x t is monotonically non-increasing and the sequence y t satisfies the uniform bound y 2 t ≤ 2 W ( x 0 , y 0 ) /a for all t ∈ N .

Proof. In any sufficiently small neighbourhood U of 0 ∈ R 2 ≥ 0 , one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all ( x, y ) ∈ U , so that W is non-increasing as a function of the iterates of the map close to zero.

Now fix ( x 0 , y 0 ) sufficiently small that, with W 0 := W ( x 0 , y 0 ) one has ( x 0 , √ 2 W 0 /a ) contained in U . Then in particular ( x 0 , y 0 ) is contained in U since y 0 ≤ √ 2 W 0 /a . Suppose as an inductive hypothesis that y t ≤ √ 2 W 0 /a , x t ≤ x 0 and W t ≤ W 0 . Shrinking U yet further if necessary, one sees that

<!-- formula-not-decoded -->

thus yielding x t +1 &lt; x t ≤ x 0 . Moreover, the inductive hypothesis implies that ( x t , y t ) ∈ U ; thus, denoting W t := W ( x t , y t ) and W t +1 := W ( x t +1 , y t +1 ) , one sees that

<!-- formula-not-decoded -->

so that y t +1 ≤ √ 2 W t +1 /a ≤ √ 2 W 0 /a . This completes the proof.

Finally we can prove convergence of the iterates of T toward zero.

Theorem D.8. Set κ := √ b/ ( c -a ) . For all sufficiently small x 0 ≥ 0 and y 0 &gt; 0 and all 0 &lt; ϵ &lt; κ , there is so that for all t ≥ τ , one has

and

<!-- formula-not-decoded -->

In particular, x t , y t = Θ( t -1 / 2 ) .

Proof. Using Lemma D.7, we may take ( x 0 , y 0 ) sufficiently small that the quantity ˜ ∆ t := ˜ ∆( x t , y t ) of Lemma D.6 satisfies the recursion

<!-- formula-not-decoded -->

for all t ∈ N . By taking ( x 0 , y 0 ) yet smaller if necessary we may also assume that y t satisfies the lower-bound

<!-- formula-not-decoded -->

Squaring and taking the reciprocal yields

<!-- formula-not-decoded -->

and, taking ( x 0 , y 0 ) yet smaller if necessary so that 1 / (1 -2 cy 2 t ) ≤ 1 + 6 y 2 t for all t , one obtains the recursive estimate

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all t ∈ N by telescoping. Plugging this into the recursion for ˜ ∆ and taking logarithms, one has

<!-- formula-not-decoded -->

Taking a telescoping sum yields

<!-- formula-not-decoded -->

by Lemma E.4. Thus

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

from which one obtains the bound

<!-- formula-not-decoded -->

and for any 0 &lt; ϵ &lt; √ b/ ( c -a ) one sees that for any t greater than or equal to

<!-- formula-not-decoded -->

one has | ˜ ∆ t | &lt; ϵ/ 4 . It follows from the definition of ˜ ∆( x, y ) := ( y -κx + O ( x 2 )) /x that t ≥ τ ( y 0 , ϵ ) implies

<!-- formula-not-decoded -->

where, shrinking x 0 , y 0 if necessary, the final containment is obtained using Lemma D.7. With t ≥ τ ( y 0 , ϵ ) and the bounds ( κ + ϵ/ 2) x t ≥ y t ≥ ( κ -ϵ/ 2) x t in hand, one obtains

<!-- formula-not-decoded -->

again shrinking x 0 , y 0 and invoking Lemma D.7 if necessary to deal with the higher-order terms. Inverting and squaring yield the bounds

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

again shrinking x 0 , y 0 and invoking Lemma D.7 if necessary to obtain the final upper bound in (276), so that

<!-- formula-not-decoded -->

from which telescoping summation yields the bounds (263). The corresponding bounds for y t now follow from the limits of (274). This completes the proof.

Finally, we may prove our convergence theorem in the critical regime using Theorem D.8.

Theorem D.9. Assume that η = 2 /λ ( θ ∥ ∗ ) . Assume in addition that ν/ ( c (2 /λ ( θ ∥ ∗ ) , θ ∥ ∗ ) λ ( θ ∥ ∗ )) &lt; 1 , where ν is defined in Assumption 3.4 and c is defined in Assumption 3.3. Then for all θ ∥ 0 sufficiently close to θ ∥ ∗ and all θ ⊥ = 0 sufficiently small, one has d M ( θ ∥ t , θ ∥ ∗ ) = Θ( t -1 / 2 ) and | θ ⊥ t | = Θ( t -1 / 2 ) .

̸

Proof. Denote λ ∗ := λ ( θ ∥ ∗ ) and c ∗ := c (2 /λ ∗ , θ ∥ ∗ ) for notational convenience. By Theorem 4.1, for all ( θ ∥ , θ ⊥ ) sufficiently close to ( θ ∥ ∗ , 0) , GD has the coordinate expression

<!-- formula-not-decoded -->

We first derive a formula for d M ( GD ∥ ( η, θ ∥ , θ ⊥ ) , θ ∥ ∗ ) = ∥ log θ ∥ ∗ (GD ∥ ( η, θ ∥ , θ ⊥ )) ∥ . Using a covariant Taylor expansion (29), observe that:

<!-- formula-not-decoded -->

by Assumption 3.4, where Π θ ∥ ∗ → θ ∥ is parallel transport from θ ∥ ∗ to θ ∥ . Similarly, observe that

<!-- formula-not-decoded -->

We may thus write

<!-- formula-not-decoded -->

Hence:

<!-- formula-not-decoded -->

We now derive a formula for | GD ⊥ ( η, θ ∥ , θ ⊥ ) | . Perform a covariant Taylor expansion (29) on λ ( θ ∥ ) to obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

by Assumption 3.4. Then one sees that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so that

<!-- formula-not-decoded -->

provided d M ( θ ∥ , θ ∥ ∗ ) and | θ ⊥ | are sufficiently small. The result now follows by invoking Theorem D.8.

## D.3 Supercritical regime

Our convergence theorem in the supercritical regime is the easiest of the three. The orbit to which the iterates are attracted is hyperbolic unlike in the critical case, and the transient phenomena are relatively simple compared with those of the subcritical case.

Theorem D.10. Assume that η &gt; 2 /λ ( θ ∥ ∗ ) is sufficiently small. Then there are positive constants C 1 , C 2 , C 3 and a stable orbit of period two of the form ( θ ∥ ∗ , ± ( ηλ ( θ ∥ ∗ ) -2) 1 / 2 + O ( ηλ ( θ ∥ ∗ ) -2) ) to which the iterates ( θ ∥ t , θ ⊥ t ) starting from any ( θ ∥ 0 , θ ⊥ 0 ) satisfying 0 &lt; | θ ⊥ 0 | ≤ C 1 ( ηλ ( θ ∥ ∗ ) -2) 1 / 2 and d M ( θ ∥ 0 , θ ∥ ∗ ) ≤ C 2 ( ηλ ( θ ∥ ∗ ) -2) 1 / 2 converge with rate O ( (1 -C 3 ( ηλ ( θ ∥ ∗ ) -2)) t ) .

Proof. Arguing using covariant Taylor expansions as in the proof of Theorem D.9, one sees that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Consequently, the result follows from Theorem D.11 below.

Theorem D.11. Given α &gt; 0 and constants a &gt; 0 and b &gt; 0 , consider the functions

<!-- formula-not-decoded -->

Then, there are constants C 1 , C 2 , C 3 &gt; 0 such that for all α sufficiently small, the map F = ( F x , F y ) admits a stable, period-2 orbit { (0 , ξ + ) , (0 , ξ -) } with ξ ± = ± √ α + O ( α ) , and the iterates ( x t +1 , y t +1 ) := F ( x t , y t ) starting from any ( x 0 , y 0 ) with 0 &lt; | y 0 | ≤ C 1 √ α and | x 0 | ≤ C 2 √ α converge to this orbit at a rate of O ( (1 -C 3 α ) t ) .

Proof. The line x = 0 is preserved by F , and thereon one sees that F y takes the form

<!-- formula-not-decoded -->

̸

which, by Lemma D.2, admits a period-two orbit { ξ + , ξ -} of the form ξ ± = ± √ α + O ( α ) . We will show that the iterates of the square F 2 of F converge to (0 , ξ ± ) for all initial points ( x 0 , y 0 ) sufficiently small, y 0 = 0 . This will be achieved by showing that ∥ DF 2 ( x, y ) ∥ &lt; 1 uniformly over a neighbourhood of (0 , ξ ± ) , followed by proving a guarantee of convergence to this neighbourhood in finite time.

Via a routine calculation one sees that

<!-- formula-not-decoded -->

The derivative of F 2 is then given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In particular, for all y satisfying

<!-- formula-not-decoded -->

the matrix DF 2 ( x, y ) has entries with magnitudes upper-bounded by for all α sufficiently small. Using the bound ∥ A ∥ 2 ≤ √ ∥ A ∥ 1 ∥ A ∥ ∞ for a matrix A , where ∥ · ∥ 1 and ∥ · ∥ ∞ are the maximum column 1-norm and maximum row 1-norm respectively, one sees then that there are C 2 , C 3 &gt; 0 such that for all α sufficiently small, all x satisfying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Letting V be the neighbourhood of the two-point set { (0 , ξ ± ) } enclosed by the bounds (301) and (303), one sees that the iterates ( x 2 t , y 2 t ) of F 2 starting at any point ( x 0 , y 0 ) ∈ V converge toward (0 , ξ sign ( y 0 ) ) with rate ∥ ( x 2 t , y 2 t -ξ ± ) ∥ ≤ (1 -C 3 α ) t ∥ ( x 0 , y 0 -ξ sign ( y 0 ) ) ∥ .

and all y satisfying (301), one has ∥ DF 2 ( x, y ) ∥ 2 ≤ 1 -C 3 α .

We complete the proof by showing that the iterates starting from any x 0 satisfying (303) and any y 0 satisfying

<!-- formula-not-decoded -->

are eventually drawn into V . This, however, is easy: for any x satisfying (303) and any y satisfying y 2 ≤ α/ ( (1 + α )(1 + (1 + α ) 2 ) ) , one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all α sufficiently small. It follows that the iterates y 2 t = ( F 2 t ) y ( x 0 , y 0 ) satisfy the recursion | y 2( t +1) | ≥ ( 1+(1 / 2) α ) | y 2 t | as long as y 2 2 t ≤ α/ ( (1 + α )(1 + (1 + α ) 2 ) ) . We claim now that there exists τ ∈ N satisfying

<!-- formula-not-decoded -->

such that y 2 τ ≥ α/ ( (1 + α )(1 + (1 + α ) 2 ) ) . Suppose for a contradiction that this were not the case; then

<!-- formula-not-decoded -->

which is a contradiction. This proves the result.

## E Elementary lemmas

In this section, for ease of reference, we collect several well-known facts and their elementary proofs. Recall that a subset M ′ of a Riemannian manifold M is geodesically convex if any two points of M ′ can be connected by a unique geodesic contained entirely in M ′ .

Lemma E.1. Let λ : M ′ → R be a C 2 function on a geodesically convex subset M ′ of a Riemannian manifold M . If λ is µ -geodesically strongly convex in the sense that ∇ 2 M λ ⪰ µI TM at all points on M ′ , then

<!-- formula-not-decoded -->

for all x, y ∈ M ′ .

Proof. Fix x, y ∈ M ′ . Since M ′ is geodesically convex, there exists a unique geodesic γ : [0 , 1] → M ′ such that γ (0) = x and γ (1) = y . This γ satisfies

<!-- formula-not-decoded -->

for all t ∈ [0 , 1] , where Π( γ ) t : T γ (0) M → T γ ( t ) M is parallel transport. Define φ ( t ) := λ ( γ ( t )) . Then using (23) and the chain rule,

<!-- formula-not-decoded -->

for all t ∈ [0 , 1] . Thus, by Taylor's theorem, there is s ∈ [0 , 1] such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

as claimed.

Lemma E.2. Let λ : M ′ → R be a C 2 function on a geodesically convex subset M ′ of a Riemannian manifold M . If λ is µ -geodesically strongly convex in the sense that ∇ 2 M λ ⪰ µI TM at all points on M ′ , then

<!-- formula-not-decoded -->

for all x, y ∈ M ′ , where Π x → y : T x M → T y M is parallel transport.

Proof. Fix x, y ∈ M ′ . From Lemma E.1, one has the estimates

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since parallel transport is an orthogonal transformation, one has

<!-- formula-not-decoded -->

Substituting this into (317) and rearranging yields

<!-- formula-not-decoded -->

while (318) may be rearranged to give

<!-- formula-not-decoded -->

Adding these estimates then yields the result.

Lemma E.3. Let λ : M ′ → R be a C 2 function on a geodesically convex subset M ′ of a Riemannian manifold M . If λ is µ -geodesically strongly convex in the sense that ∇ 2 M λ ⪰ µI TM at all points on M ′ and admits a critical point x ∗ in M ′ , then x ∗ is the unique global minimum of λ in M ′ and

<!-- formula-not-decoded -->

̸

for all x ∈ M ′ .

Proof. That the global minimum x ∗ is unique follows from Lemma E.1: for any x = x ∗ in M , one has

<!-- formula-not-decoded -->

thus proving the first claim. To prove the second claim, fix x ∈ M ′ . Lemma E.1 then applies to yield

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

from which the result follows.

Lemma E.4. For any c &gt; 0 , and any integers 0 ≤ s &lt; t , one has

<!-- formula-not-decoded -->

Proof. Since x ↦→ 1 ( c + x ) is monotonically decreasing along the positive reals, one has

<!-- formula-not-decoded -->

for any natural number n , from which it follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Integrating both sides now yields the result.

## F Experimental supplement

Our experiments were conducted for a depth 5, width 1 scalar factorisation problem, corresponding to the function f : R p → R defined by

<!-- formula-not-decoded -->

with target value y = 1 . Our code 3 runs gradient descent on this problem, and can be executed in seconds on a single cpu. At each iterate, the projection onto the solution manifold M := f -1 { y } is computed as follows.

The KKT conditions for the constrained optimisation problem

<!-- formula-not-decoded -->

3 Available at https://github.com/lemacdonald/eos-convergence-rates-codimension-1

are given by

<!-- formula-not-decoded -->

where α is the constraint parameter. These equations admit the quadratic solutions

<!-- formula-not-decoded -->

It is easily verified α ↦→ θ ∥ i ( α ) , is monotonically decreasing (when (334) is taken with the + sign) or monotonically increasing (when (334) is taken with the -sign); consequently, the map ϕ : α ↦→ ∏ p i =1 θ ∥ i ( α ) is also monotonic provided the signs are consistent among the θ ∥ i ( α ) .

The sign conventions we use are as follows. If ∏ p i =1 θ i &lt; y , then we must take the + sign for all θ ∥ i ( α ) . If ∏ p i =1 θ i ≥ y , then, noting that 2 -p ∏ p i =1 θ i is the minimum value possible for ∏ p i =1 θ ∥ i ( α ) when all θ ∥ i ( α ) are taken with the positive sign in (334), either:

1. y ≥ 2 -p ∏ p i =1 θ i , in which case we take the positive sign in (334) for all i .

2. y &lt; 2 -p ∏ p i =1 θ i , in which case we take the negative sign in (334) for all i .

Having determined the sign to use uniformly over all i = 1 , . . . , p in (334), we are assured that α ↦→ ϕ ( α ) = ∏ p i =1 θ ∥ i ( α ) is monotonic, hence that ϕ ( α ) = y admits a unique solution which we obtain using the bisection method.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: See the 'Paper contributions" section in the Introduction, which outlines our contributions in the context of codimension 1 least squares problems.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations are discussed in detail in the Discussion/Conclusion section under several bold headings.

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

Justification: Complete proofs of all theoretical results are provided in several appendices with appropriate referencing.

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

Justification: We included the code to produce our plots with the submission.

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

Justification: We provide open access to our code; see experimental supplement.

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

Justification: All relevant hyperparameters etc are contained in the code.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: Our experiments are deterministic.

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

Justification: See the 'Experimental supplement" appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have reviewed and conformed with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our work is primarily theoretical and concerned with understanding in simple settings what practitioners have already been doing at scale for over 10 years. No forseeable societal impacts.

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

Justification: We considered only toy models in our experiments.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: No existing assets used.

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

Justification: No new assets introduced.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: No crowdsourcing or research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: No crowdsourcing or research with human subjects.

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

## Answer: [NA]

Justification: LLMs were not used outside of creation of code for toy experiments and minimal assistance with proving theorems. All material was thoroughly checked and revised if necessary.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.