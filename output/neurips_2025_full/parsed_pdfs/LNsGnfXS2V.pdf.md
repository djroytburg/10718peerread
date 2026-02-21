## An Ellipsoid Algorithm for Online Convex Optimization

## Zakaria Mhammedi ∗

mhammedi@google.com

## Abstract

We study the problem of Online Convex Optimization (OCO) over a convex set K ⊂ R d , accessed via a separation oracle. While classical projection-based algorithms such as projected Online Gradient Descent (OGD) achieve the optimal O ( √ T ) regret, they require computing Euclidean projections onto K whenever an iterate falls outside the feasible set. These projections can be computationally expensive, especially for complex or high-dimensional sets. Projection-free algorithms address this by replacing projections with alternative oracle-based procedures, such as separation or linear optimization oracles. However, the regret bounds of existing separation-based methods scale poorly with the set's asphericity κ , defined as the ratio between the radii of the smallest enclosing ball and the largest inscribed ball in K ; for ill-conditioned sets, κ can be arbitrarily large.

We introduce a new separation-based algorithm for OCO that achieves a regret bound of ̃ O ( √ dT + d 2 ) , with only logarithmic dependence on κ . This removes a key limitation of prior work and eliminates the need for costly geometric pre-processing, such as transforming K into isotropic position. Our algorithm is based on a novel reduction to online optimization over a sequence of dynamically updated ellipsoids, inspired by the classical ellipsoid method for convex optimization. It requires only ̃ O ( 1 ) separation oracle calls per round, on par with existing separation-based approaches. These advances make our method particularly well suited for online optimization over geometrically complex feasible sets.

## 1 Introduction

Convex optimization plays a central role in many areas of modern machine learning and related fields. While classical algorithms are designed for settings with a fixed objective function, recent work has increasingly considered the more general framework of Online Convex Optimization (OCO) [10]. OCO models sequential decision-making problems where a possibly different convex function is revealed at each round, potentially chosen in an adversarial manner. This framework captures a broad range of learning scenarios and provides a unified lens for analyzing them. Standard online-to-batch conversion techniques [3, 22, 4] allow regret guarantees in OCO to be translated into convergence rates for offline (fixed-objective) and stochastic convex optimization problems. In contrast, the analysis of traditional offline methods does not always extend to stochastic settings. Even stochastic nonconvex optimization has benefited from the OCO framework; [5] showed that stochastic non-convex optimization can be reduced to OCO, and that this reduction leads to the best-known convergence rates for finding stationary points. Given its broad applicability, designing computationally efficient OCO algorithms with strong regret guarantees remains an important direction of research.

In constrained optimization settings where the decision variable must lie in a feasible set K , the classical (projected) Online Gradient Descent (OGD) [25] requires a Euclidean projection onto K

∗ Google Reseach, NYC.

whenever an iterate falls outside K . This step can be computationally expensive when K has a complex structure, and may limit the practicality of OGD. To address this, projection-free algorithms have been proposed. These methods avoid explicit projections; for example, the Frank-Wolfe algorithm [7] uses linear optimization over K . More recent approaches rely on membership or separation oracles to enforce constraints [18, 8, 17, 9]. In particular, separation oracles can often be more tractable to implement and use in settings where projections are costly.

The aforementioned projection-free algorithms are analyzed in the OCO framework, where at each round t the algorithm selects a point w t from a convex feasible set K ⊂ R d . It then incurs a loss f t ( w t ) , where f t is a convex function that may be chosen adversarially. The goal is to ensure a small regret sup u ∈K ∑ T t = 1 ( f t ( w t ) -f t ( u )) . While projected OGD achieves the optimal O ( √ T ) regret, projection-free methods based on linear optimization oracles have not improved beyond O ( T 3 / 4 ) [12] without additional structural assumptions on the OCO problem. Separation oracle-based methods proposed by [18, 8] achieve the optimal T -dependence with a O ( κ √ T ) regret bound, where κ ∶= R / r denotes the set's asphericity , defined as the ratio of the radii of the smallest enclosing and largest inscribed balls in K . However, the linear dependence on κ is not desirable, as κ can be arbitrarily large for ill-conditioned sets. Although preprocessing K into isotropic position can ensure κ ≤ d [6], this transformation is computationally expensive-requiring up to ̃ O ( d 4 ) separation oracle calls [16]-and does not fully resolve the issue: the dependence on κ may reappear implicitly through changes in the Lipschitz constant of the loss functions after the isotropic reparametrization. A more recent method [19] improves this by achieving a regret bound of ̃ O ( √ dT + κd ) , making the κ term independent of T . However, a residual dependence on the potentially large asphericity remains.

Contributions. This paper introduces a new projection-free algorithm for OCO based on separation oracles. Our key contribution is an algorithm that achieves a regret bound with only logarithmic dependence on the feasible set's asphericity κ . Specifically, it guarantees a regret of ̃ O ( √ dT + d 2 ) without requiring expensive geometric preprocessing of the set K . Thus, this approach is more practical and robust to the feasible set's ill-conditioning compared to prior separation-based methods.

The algorithm preserves the oracle complexity of existing separation-based approaches, requiring only ̃ O ( 1 ) separation oracle calls per round. The total computational cost per iteration is ̃ O ( C sep + d ω ) , where C sep denotes the cost of a single separation oracle call and ω is the matrix multiplication exponent. Our approach integrates ideas from the classical ellipsoid method with a reduction to online exp-concave optimization over a sequence of dynamically updated ellipsoids. A slight modification of the standard online-to-batch conversion yields a convergence rate of ̃ O ( σ √ d / T + d 2 / T ) for Stochastic Convex Optimization (SCO), where σ 2 denotes the gradient variance, and a rate of ̃ O ( d 2 / T ) in the offline setting ( σ = 0 ). A detailed comparison with prior results is presented in Table 1.

Limitations. We do not provide regret lower bounds or experimental validation for our algorithm.

Table 1: Comparison of projection-free algorithms for OCO and Stochastic Convex Optimization (SCO). κ = R / r is the asphericity of the feasible set K . σ 2 is the variance in SCO.

| Papers     | Regret bound in OCO   | Convergence rate in SCO ( σ ≥ 0 )   | Oracle type         | Number of oracle calls per round   | Runtime per round   |
|------------|-----------------------|-------------------------------------|---------------------|------------------------------------|---------------------|
| [12]       | O ( T 3 / 4 )         | O ( 1 T 1 / 3 )                     | Linear optimization | 1                                  | C linOpt + d        |
| [18, 8]    | O ( κ √ T )           | O ( κ √ T )                         | Separation          | O ( 1 ) - ̃ O ( 1 )                | C sep + d           |
| [19]       | ̃ O ( √ dT + κ d )    | ̃ O ( σ √ d T + κ d T )             | Separation          | ̃ O ( 1 )                          | C sep + d 2         |
| This paper | ̃ O ( √ dT + d 2 )    | ̃ O ( σ √ d T + d 2 T )             | Separation          | ̃ O ( 1 )                          | C sep + d ω         |

Related works. This work builds upon the recent line of research on separation oracle-based projection-free algorithms for OCO. The prior art established methods achieving O ( κ √ T ) [18, 17, 8] and subsequently ̃ O ( √ dT + κd ) regret bounds [19]. A key limitation of these results is the dependence on the asphericity κ . Our work directly addresses this limitation by providing an algorithm whose regret guarantee depends on κ only logarithmically .

Our algorithmic approach is inspired by the classical ellipsoid method [14, 23, 2], which we adapt to the online optimization setting. While the classical method is designed for offline problems, our algorithm addresses the sequential nature of OCO by combining ellipsoid updates with a reduction to online optimization over a sequence of changing ellipsoids that always contain the feasible set K . Our reduction shares conceptual links with the one presented in [19], but crucially, our method updates these ellipsoids adaptively to better 'approximate' the shape of the feasible set, unlike prior approaches using a fixed ball. This ability to adapt to the feasible set's geometry on the fly is a key feature distinguishing our method from previous separation-based approaches.

## 2 Preliminaries

In Section 2.1, we formally introduce the OCO setup along with the notation used throughout the paper. Section 2.2 presents key concepts we rely on such as Gauge distance and Gauge projections.

## 2.1 Setup and Notation

Let K be a closed convex subset of the Euclidean space R d , where we assume d ≥ 2 throughout. 2 We consider the standard framework of OCO over K , in which an algorithm generates a sequence of decisions ( w t ) t ≥ 1 within K . On each round t ∈ [ T ] , the algorithm selects a point w t ∈ K and incurs a loss f t ( w t ) , where f t ∶ K → R is a convex loss function that may depend adversarially on past decisions. As is standard in this setting, we assume access to a subgradient g t ∈ ∂f t ( w t ) rather than the full loss function. The performance of the algorithm is measured by its regret after T rounds, defined as Reg T ∶= ∑ T t = 1 f t ( w t ) -inf u ∈K ∑ T t = 1 f t ( u ) . Due to the convexity of each f t , the regret can be upper bounded by the so-called linearized regret: ∑ T t = 1 ⟨ g t , w t ⟩ -inf u ∈K ∑ T t = 1 ⟨ g t , u ⟩ . Hence, it suffices to control the linearized regret to bound Reg T .

Building on recent projection-free methods, this work aims to develop an efficient OCO algorithm that achieves sublinear linearized regret with only logarithmic dependence on the asphericity κ , while requiring a logarithmic number of separation oracle calls per round.

Definition 2.1 (Separation oracle). A Separation oracle Sep C for a set C is an oracle that given u ∈ R d returns ( b, v ) ∈ { 0 , 1 } × B ( 1 ) (where B ( 1 ) denotes the unit Euclidean ball in R d ), such that

- b = 0 and v = 0 , if u ∈ C ; and otherwise,
- b = 1 and ⟨ v , u ⟩ &gt; ⟨ v , w ⟩ , for all w ∈ K .

We denote by C sep (C) the computational cost of one call to this oracle.

In addition to the standard OCO setup, we impose two common assumptions. To present these assumptions, we let ∥ ⋅ ∥ denote the Euclidean norm and use B ( c , γ ) ⊂ R d to denote the ball of radius γ &gt; 0 centered at c ∈ R d . With this, the first assumption asserts that each loss function f t is G -Lipschitz with respect to ∥ ⋅ ∥ , for some constant G &gt; 0 . The second assumption requires that the feasible set K is bounded and lies between two Euclidean balls.

Assumption 2.1. There is some G &gt; 0 , such that for all t ≥ 1 , the function f t ∶ K → R is convex and for all w ∈ K and g ∈ ∂f t ( w ) , we have ∥ g ∥ ≤ G .

Assumption 2.2. The set C satisfies B ( c 0 , r ) ⊆ K ⊆ B ( R ) for some r, R &gt; 0 , and c 0 ∈ B ( R ) .

Prior work [18, 19] assumes the feasible set K satisfies B ( r ) ⊆ K ⊆ B ( R ) for r, R &gt; 0 . This paper relaxes the condition B ( r ) ⊆ K . Requiring K to contain a positive-radius ball centered at the origin is restrictive for general convex sets; such a ball may only exist after translation, its radius r can be small even then, and computing an optimal translation can be computationally expensive. Our algorithm eliminates the need to compute this translation, dynamically adapting to the geometry of K instead. We also note that both Assumption 2.2 and the assumptions in [18, 19] require K to have non-empty interior. This is without loss of generality, as any set K contained in a lower-dimensional affine subspace can be reparametrized as a full-dimensional set with non-empty interior.

Throughout, we let κ ∶= R / r denote the asphercity of K with r, R as in Assumption 2.2.

2 In one dimension ( d = 1 ), computing Euclidean projections is straightforward, making projected gradient methods both efficient and optimal.

Algorithm 1 GaugeDist ( w ; C , H, c , δ ) : Approximate value and subgradient of the Gauge distance.

```
require: Separation oracle Sep C , input vector w ∈ R d , H ∈ R d × d PSD, c ∈ R d , and δ > 0 . returns If w / ∈ C and c ∈ C , then S ≈ S Cc ( w -c ) and s ≈ ∂S Cc ( w -c ) . 1: Set ( b, v ) ← Sep C ( w ) . // b = 1 if w ∈ C ; and 0 , otherwise. 2: if b = 1 then // This corresponds to the case where w ∈ C . 3: Set ( S, s ) ← ( 0 , 0 ) . 4: return ( S, s ) . 5: Set ( b, v ) ← Sep C ( c ) . // b = 1 if c ∈ C ; and 0 , otherwise. 6: if b = 0 then 7: Set ( S, s ) ← ( 0 , 3 d v / √ v ⊺ H v ) . 8: return ( S, s ) . 9: Set α ← 0 , β ← 1 , and µ ←( α + β )/ 2 . 10: while β -α > δ 8 d 2 do 11: Set ( b, v ) ← Sep C ( µ ( w -c ) + c ) . // b = 1 if µ ( w -c ) + c ∈ C ; and 0 , otherwise. 12: if β ⋅ v ⊺ ( w -c ) < 1 2 d √ v ⊺ H v then break 13: Set α ← µ if b = 1 ; and β ← µ otherwise. 14: Set µ ←( α + β )/ 2 . 15: Set S ← α -1 -1 and s ← v β ⋅ v ⊺ ( w -c ) . 16: return ( S, s ) .
```

Additional notation. We denote by K ○ ∶= { x ∈ R d ∶ ⟨ x , y ⟩ ≤ 1 , ∀ y ∈ K} the polar set of K [13]. We let S d × d &gt; 0 denote the set of d × d positive-definite matrices and ∥ x ∥ H ∶= √ x ⊺ H x for x ∈ R d and H ∈ R d × d . For c ∈ R d , H ∈ S d × d &gt; 0 , we define the ellipsoid E( c , H ) ∶= { u ∈ R d ∣ ( u -c ) ⊺ H -1 ( u -c ) ≤ 1 } . We use ̃ O (⋅) to hide poly-log factors in parameters appearing in the expression.

## 2.2 Gauge Function, Distance, and Projection

We now introduce the concepts of the Gauge function, distance, and projection [18, 19], which are central to our algorithm and existing separation-based methods. Throughout this section, these concepts are defined using a convex set C ⊆ R d containing the origin. Here, C is used for definitional purposes only and does not necessarily correspond to the feasible set K in our OCO problem.

<!-- formula-not-decoded -->

The Gauge function γ C can be viewed as a 'pseudo' norm induced by the convex set C ; it becomes a true norm when C is centrally symmetric (i.e., C = -C ). With this, we define the Gauge distance.

Definition 2.3 (Gauge distance). The Gauge distance function S C corresponding to the set C is

<!-- formula-not-decoded -->

When the set C contains the origin in its interior, [18] showed that S C has the more distance-like expression S C ( u ) ∶= inf x ∈C γ C ( u -x ) , for all u ∈ R d . From this expression, we see that the Gauge distance generalizes the Euclidean distance, which is obtained by replacing γ C [resp. C ] with the Euclidean norm ∥ ⋅ ∥ [resp. B ( 1 ) ]. Observe that when u ∈ C , then S C ( u ) = 0 . Otherwise, S C ( u ) &gt; 0 .

Gauge projection. Similar to [18, 19], our algorithm relies on Gauge projections to ensure feasible iterates. The Gauge projection operator Π gau C induced by C is the mapping: Π gau C ( u ) ∶= arg min x ∈C γ C ( u -x ) , for all u ∈ R d . [18] showed that when 0 lies in the interior of C , the gauge projection admits the closed-form expression Π gau C ( u ) = u 1 +S C ( u ) . This makes the gauge projection particularly useful in our setting, as it can be approximated efficiently (by approximating the gauge distance S C ) using a logarithmic number of separation oracle calls.

In this paper, we use Algorithm 1 to approximate both the Gauge distance and its subgradients, which are required by our OCO algorithm. The algorithm, similar to [19, Algorithm 1], uses calls to a separation oracle for C and binary search to approximate the distance function. In a nutshell, given input (C , H, w , c ) such that c ∈ C and w ∉ C , Algorithm 1 returns a pair ( S, s ) , where S is

an approximation of the Gauge distance S Cc ( w -c ) , and s is an approximate subgradient of this function at w -c . Algorithm 1 is slightly more general than [19, Algorithm 1] in that it allows for c ≠ 0 and can handle c ∉ C . This feature is important because our projection-free algorithm will operate on a sequence of potentially uncentered ellipsoids. The next lemma is proven in Appendix A.

Lemma 2.1. Let δ ∈ ( 0 , 1 ) , w , c ∈ R d , C ⊆ R d convex, and H ∈ S d × d &gt; 0 be given such that C ⊆ E ∶= { x ∣ ( x -c ) ⊺ H -1 ( x -c ) ≤ 1 } and w ∈ E . Consider a call to Algorithm 1 with input (C , H, c , w , δ ) and let ( S, s ) be its output. Then, either ∥ H 1 / 2 s ∥ &gt; 2 d ; or we have for all u ∈ R d :

<!-- formula-not-decoded -->

Further, Algorithm 1 makes at most 2 + log 2 ( 8 d 2 / δ ) calls to the separation oracle Sep C .

As we will see in Section 4.1, the case where ∥ H 1 / 2 s ∥ &gt; 2 d in Lemma 2.1 corresponds to a situation where the set H -1 / 2 (C c ) is too 'thin' in the direction H 1 / 2 s . We will later use this test to dynamically update ellipsoids containing the feasible set, adapting them to its geometry.

## 3 Overview and Limitations of Previous Approaches

In this section, we briefly review existing separation-based approaches that our work builds on and highlight their main limitations that we address in this paper.

[18, 19] reduce OCO over K to OCO over a ball B ( R ) ⊇ K . For this reduction, they rely on an inner OCO algorithm that produces iterates ( u t ) in B ( R ) (where Euclidean projections are cheap), from which feasible iterates ( w t ) are extracted using Gauge projections (also cheap when given a separation oracle for K ; see Section 2.2). The feedback vector fed to the inner algorithm is given by

<!-- formula-not-decoded -->

where s t ∈ ∂S K ( w t ) . The subgradient s t can be computed efficiently using the GaugeDist subroutine in Algorithm 1; see Section 2.2. This choice of ̃ g t guarantees that

<!-- formula-not-decoded -->

which represents a key result in their reduction; see [19, Lemma 4.1]. The inequality in (2) is useful because it allows one to bound the instantaneous regret of their algorithm (left-hand side of (2)) in terms of that of the inner algorithm (right-hand side of (2)). Thus, if the inner algorithm achieves low regret over the ball B ( R ) , this guarantee essentially carries over to their final algorithm via (2).

The key issue with this approach is that standard OCO algorithms typically have regret bounds that scale with the maximum norm of the feedback vectors. Thus, if the inner algorithm is one such algorithm, its regret will scale linearly with max t ∈[ T ] ∥̃ g t ∥ . Due to the presence of s t in the definition of ̃ g t in (1), and because ∥ s t ∥ can be as large as the asphericity κ = R / r (see [19, Lemma 4.1]), the resulting regret bound is of order O ( κ √ T ) (in our work, we aim for a logarithmic dependence on κ ).

In a follow-up work, [19] manages to move the dependence on κ into a lower-order term, achieving a final regret bound of ̃ O ( √ dT + κd ) by using a variant of ONS as the inner algorithm. We now sketch why ONS was key to achieving this result; this is relevant as we also use ONS as the inner algorithm.

Advantage of ONS. The regret of ONS with learning rate η can be bounded by (see [19, Thm. 3.1])

<!-- formula-not-decoded -->

On the other hand, as observed by [19], as long as η ≤ ( max t ∈[ T ] ∣⟨̃ g t , u t -u ⟩∣) -1 , Eq. (2) implies

<!-- formula-not-decoded -->

Combining this with (3) and rearranging terms implies that the regret ∑ T t = 1 ⟨ g t , w t -u ⟩ is bounded by η 2 ∑ T t = 1 ⟨ g t , w t -u ⟩ 2 + O ( 1 ) ⋅ d log ( BT ) η , for all u ∈ K . Tuning η optimally in the interval between 0 and ( max t ∈[ T ] ∣⟨ u t -u , ̃ g t ⟩∣) -1 yields the regret bound

<!-- formula-not-decoded -->

for all u ∈ K . The key takeaways from (4) are two-fold: I) Unlike the regret bound of [18] where the dependence on B is linear, the B term in (4), which grows with the maximum norm of the vectors ( s t ) , appears only inside a logarithmic term; and II) although the scale of the vectors ( s t ) still affects the bound in (4) outside logarithms through max t ∈[ T ] ∣⟨ u t -u , ̃ g t ⟩∣ , this quantity can be significantly smaller than ∥ s t ∥ itself, as we will see in the analysis. [19] show that (4) can be bounded from above by ̃ O ( √ dT + κd ) . However, the κd term, while independent of T , can still be arbitrarily large for ill-conditioned sets, and we would like to remove it.

Limitations of reparametrization. Although any set K can be put into isotropic position so that its asphericity κ is reduced to at most O ( d ) [6], computing an approximate isotropic transformation may require up to ̃ O ( d 4 ) calls to a separation oracle [16], which can be computationally prohibitive in many practical applications.

## 4 Algorithm and Guarantees

In Section 4.1, we present our algorithm and outline the core ideas behind its design and how they allow us to overcome the limitations of previous approaches discussed in Section 3. In Section 4.2, we formally state our algorithm guarantees.

```
Algorithm 2 Ellipsoid algorithm for online convex optimization. require: Rounds T , a feasible point c ∈ K , r, R , and G,η > 0 . 1: Set κ ← R r , ε ← 1 κ 18 T 2 , β ← ηG 2 , c 1 ← c , H 1 ← R 2 I , Σ 1 ← βI , ( λ min , λ max ) ← ( ε 24 κ 2 , 40 d 2 κ ε ) . 2: for t = 1 , . . . , T do 3: Set ( S t , s t ) ← GaugeDist ( u t ; K , H t , c t , ε ) . // S t ≈ S Kc t ( u t -c t ) ; s t ≈ ∂S Kc t ( u t -c t ) . 4: if ∥ H 1 / 2 t s t ∥ > 2 d then 5: Play w t = c and observe g t ∈ ∂f t ( w t ) . 6: Set ̃ g t ← 0 . // Update the ellipsoid containing K . 7: Update c t + 1 ← c t -1 2 d + 2 ⋅ H t s t √ s ⊺ t H t s t . 8: Update H t + 1 ← 4 d 2 -1 4 d 2 -4 ⋅ ( H t -2 d 2 d 2 + d -1 H t s t s ⊺ t H t s ⊺ t H t s t ) . 9: else 10: Play w t = u t -c t 1 + S t + c t and observe g t ∈ ∂f t ( w t ) . // w t ≈ Π gau Kc t ( u t -c t ) + c t . 11: Set ̃ g t = g t -I {⟨ g t , u t -c t ⟩ < 0 } ⋅ ⟨ g t , w t -c t ⟩ ⋅ s t . 12: Update c t + 1 ← c t and H t + 1 ← H t . /* Do one ONS in E t + 1 ∶= { u ∣ ( u -c t + 1 ) ⊺ H -1 t + 1 ( u -c t + 1 ) ≤ 1 } given 'loss vector' ̃ g t */ 13: Set Σ t + 1 ← Σ t + 1 + η ̃ g t ̃ g ⊺ t . 14: Set z t + 1 ← u t -Σ -1 t + 1 ̃ g t . // Approximate u t + 1 ← arg min u ∈E t + 1 ( z t + 1 -u ) ⊺ Σ t + 1 ( z t + 1 -u ) with PoE (Algorithm 3). 15: Set u t + 1 ← PoE ( z t + 1 , c t + 1 , βR 2 Σ -1 t + 1 , H t + 1 ; λ min , λ max , ε ) .
```

## 4.1 Algorithm Overview and Design Rationale

Our main algorithm (Algorithm 2) effectively reduces OCO over the feasible set K to OCO over a sequence of ellipsoids

<!-- formula-not-decoded -->

each containing K , with centers ( c t ) ⊂ R d and positive definite matrices ( H t ) ⊆ S d × d &gt; 0 . Intuitively, with each update, the ellipsoid increasingly pulls in around the feasible set, yielding a progressively tighter approximation of K and allowing us to bypass the need for any expensive pre-processing of K , even when K is ill-conditioned. The advantage of working with ellipsoids containing K instead of K itself is that projections onto the former can be carried out more efficiently; we will see that the complexity of doing so is almost independent of the geometry of K . The specifics of how the ellipsoids are constructed and when updates occur will be explained in detail in the sequel.

Inner algorithm. Similar to prior works [18, 19], our reduction relies on an inner OCO algorithm whose iterate at round t lies within the ellipsoid E t . For clarity of exposition, we directly instantiate

the inner algorithm with a variant of ONS (see Line 13 and Line 15) that is adequate to achieve our desired regret bound. The version of ONS we use is similar to the original algorithm proposed by [11], with the key difference is that we perform generalized projections over a sequence of varying sets, namely the ellipsoids (E t ) , rather than projecting onto a fixed set; the generalized projection solves the problem inf u ∈C ∥ u -c t ∥ 2 H for a convex set C and a matrix H ∈ S d × d &gt; 0 .

Efficient 'projections.' In Algorithm 2, we use the PoE subroutine (described in Appendix C) to efficiently perform generalized projections onto the ellipsoids (E t ) (see Line 15); this costs ̃ O ( d ω ) . The iterates ( u t ) produced by ONS are then projected onto the set K using Gauge projections (Line 3 and Line 10), which we showed can be implemented using logarithmically many calls to a separation oracle for K (see Section 2.2). This yields a sequence of iterates ( w t ) that are guaranteed to be in K .

Feedback to inner algorithm. The feedback given to the inner ONS algorithm is a modified gradient vector ̃ g t (reminiscent of the choice of ̃ g t in prior works displayed in (1)) defined as

<!-- formula-not-decoded -->

where c t is the center of the ellipsoid at iteration t , and s t is an approximate subgradient of the Gauge distance function S Kc t at u t -c t . This subgradient is computed efficiently (see Line 3) using the GaugeDist subroutine from Section 2.2. Similar to (2), this choice of ̃ g t ensures that

<!-- formula-not-decoded -->

where g t ∈ ∂f t ( w t ) is a subgradient of the loss function at the outer iterate w t , and u t is the iterate of the inner ONS algorithm. This inequality is useful because it allows us to relate the instantaneous regret of Algorithm 2 to that of the inner ONS algorithm. Thus, as long as ONS, run over the sequence of ellipsoids containing K , guarantees low regret, we get low regret for Algorithm 2.

As mentioned earlier, one advantage of working with ellipsoids that contain K is that projections onto them can be performed more efficiently than projections onto K itself. However, this alone does not explain our use of a sequence of varying ellipsoids, rather than a fixed ball as in [18, 19]. The key reason is that varying ellipsoids allow us to dynamically adapt to the geometry of K , and avoid any linear dependence in κ in our final regret bound.

Mitigating κ -dependence. As discussed in Section 3, prior methods incur a κ term due to either a large norm of s t ∈ ∂ S K ( w t ) (as in [18]) or a large inner product ∣⟨ u t -u , ̃ g t ⟩∣ for some t and comparator u (as in [19]). In the latter case, observe that ∣⟨ u t -u , ̃ g t ⟩∣ becomes large only if ∣⟨ u t -u , s t ⟩∣ is large, due to the expression for ̃ g t in (1) (also the case for our choice of ̃ g t in (5)).

As we will clarify shortly, a large value of ∣⟨ u t -u , s t ⟩∣ indicates that K is too 'thin' in a certain direction. Our approach exploits this fact to adapt to the geometry of K whenever such thin directions are detected. On the other hand, our analysis shows that these detections can happen at most ̃ O ( d 2 ) times, and outside of the 'detection rounds', the inner product ∣⟨ u t -u , ̃ g t ⟩∣ is independent of κ , and the regret incurred by the inner ONS algorithm is well controlled.

Geometry-driven ellipsoid updates. We now examine the condition that determines when the ellipsoid is updated in Algorithm 2. From Line 4-Line 8, we see that an update is triggered only when ∥ H 1 / 2 t s t ∥ &gt; 2 d ; in (9) below, we relate ∥ H 1 / 2 t s t ∥ to the magnitude of ∣⟨ u t -u , s t ⟩∣ in the previous paragraph. Here, s t is an approximate subgradient of S Kc t at u t -c t . The condition ∥ H 1 / 2 t s t ∥ &gt; 2 d can be interpreted as a test for whether K is thin in some direction relative to the ellipsoid

<!-- formula-not-decoded -->

We now clarify this claim. Suppose that ∥ H 1 / 2 t s t ∥ &gt; 2 d . Since s t ∈ ∂ S Kc t ( u t -c t ) (ignoring any approximations for now), it follows that s t ∈ (Kc t ) ○ (see Lemma G.5). Therefore, by the definition of the polar set (K c t ) ○ , we have

<!-- formula-not-decoded -->

Define the unit vector v t ∶= H 1 / 2 t s t /∥ H 1 / 2 t s t ∥ . Then, by (7) and the assumption that ∥ H 1 / 2 t s t ∥ &gt; 2 d , we have for all u ∈ K c t , ⟨ H -1 / 2 t u , v t ⟩ = 1 ∥ H 1 / 2 t s t ∥ ⋅ ⟨ u , s t ⟩ ≤ 1 ∥ H 1 / 2 t s t ∥ ≤ 1 2 d . Equivalently, we have

<!-- formula-not-decoded -->

which implies that the reparametrized set H -1 / 2 t (K c t ) is thin in the direction of v t . When this occurs, we work with a new ellipsoid E t + 1 that better aligns with the geometry of K in that direction. Specifically, we set E t + 1 = E( c t + 1 , H t + 1 ) , where c t + 1 and H t + 1 are as in Line 7 and Line 8 of Algorithm 2, respectively. We now motivate these choices for c t + 1 and H t + 1 .

Suppose that ∥ H 1 / 2 t s t ∥ &gt; 2 d , and therefore (8), holds. Then, by (8) and the fact that K ⊆ E t , we have H -1 / 2 t (K c t ) ⊆ { u ∈ B ( 1 ) ∣ u ⊺ v t ≤ 1 / 2 d } . Thus, by Lemma B.1 (an adapted result from the analysis of the ellipsoid method; see, e.g., [2, Lem. 2.3]), we have:

<!-- formula-not-decoded -->

It can be verified that c t + H 1 / 2 t E ̃ t + 1 = E( c t + 1 , H t + 1 ) , with c t + 1 and H t + 1 as in Line 7 and Line 8, which implies that K ⊆ E t + 1 . Lemma B.1 also implies that vol (E t + 1 ) ≤ exp (-1 8 d ) ⋅ vol (E t ) ; that is, we not only maintain the invariant K ⊆ E t + 1 , but we also ensure that the volume of the ellipsoid shrinks with each update, providing a tighter approximation of K . Since K ⊆ E t , for all t ≥ 1 , the volumes ( vol (E t )) cannot shrink indefinitely, enabling us to bound the number of ellipsoid updates.

Bounding the number of ellipsoid updates. By Assumption 2.2 and that vol (E t + 1 ) ≤ e -1 8 d vol (E t ) , we can show that the number of updates is bounded by O ( d 2 log ( R / r )) (recall that B ( c 0 , r ) ⊆ K ⊆ B ( R ) ). In other words, the condition on Line 4 can be satisfied at most O ( d 2 log ( R / r )) times.

Regret on update rounds. In our analysis, whenever the ellipsoid is updated at round t (i.e., the condition ∥ H 1 / 2 t s t ∥ &gt; 2 d on Line 4 is satisfied), the algorithm simply plays a default vector c ∈ K . The cumulative cost of playing c over these rounds is at most O ( d 2 log ( R / r )) , due to the bound on the number of updates; we are willing to absorb this overhead into the overall regret bound.

Regret on no-update rounds. Now, we argue that when the ellipsoid is not updated at round t (i.e., when ∥ H 1 / 2 t s t ∥ ≤ 2 d ), the inner product ∣⟨ u t -u , ̃ g t ⟩∣ remains bounded independently of κ for all u ∈ K , which ensures that the regret of the inner ONS algorithm on these rounds remains small.

Suppose that ∥ H 1 / 2 t s t ∥ ≤ 2 d (i.e., no ellipsoid update). In this case, by Hölder's inequality:

<!-- formula-not-decoded -->

where the last inequality follows from the assumption that ∥ H 1 / 2 t s t ∥ ≤ 2 d and that both u t and u lie in E t . Indeed, u t ∈ E t because the inner ONS algorithm outputs iterates in E t , and u ∈ E t because our construction maintains the invariant K ⊆ E t . The inequality in (9) implies that for all u ∈ K , we have ∣⟨ u t -u , ̃ g t ⟩∣ ≤ O ( Gd ) (by the definition of ̃ g t in (5) and Assumption 2.1), which, from (4), is what we need to ensure the regret bound of the inner ONS scales only logarithmically in κ .

## 4.2 Algorithm Guarantees

We now formalize the claims made in Section 4.1 and present the regret guarantee of our algorithm. We start with the invariants that Algorithm 2 maintains. The proof of the lemma is in Appendix E.1.

Lemma 4.1. Let T ≥ 1 , η &gt; 0 , and c ∈ K be given and suppose that Assumption 2.2 and Assumption 2.1 hold with 0 &lt; r ≤ R and G &gt; 0 , respectively. Consider a call to Algorithm 2 with input ( T, c , r, R, G, η ) . Then, for any subgradients ( g t ) , the variables in Algorithm 2 satisfy for all t ∈ [ T ] :

1. K ⊆ E t ∶= { u ∈ R d ∣ ( u -c t ) ⊺ H -1 t ( u -c t ) ≤ 1 } ; √
3. vol (E t ) ≤ e -Nt 8 d ⋅ R 2 d , where N t ∶= ∑ t -1 τ = 1 I { √ s ⊺ τ H τ s τ &gt; 2 d } ;
2. If s ⊺ t H t s t ≤ 2 d , then c t ∈ K ;
4. N t ≤ 8 d 2 log ( R / r ) ;
5. σ min ( H t ) ≥ r 2 and σ max ( H t ) ≤ ( 1 + 2 / d 2 ) N t R 2 ≤ κ 16 R 2 .

Consistent with the claim made in Section 4.1, Item 1 of the lemma establishes that the set K is always contained within the ellipsoids (E t ) . Item 2 shows that c t ∈ K whenever ∥ H 1 / 2 t s t ∥ ≤ 2 d , ensuring that when the ellipsoid is not updated, the center c t of the ellipsoid E t lies within K . This is crucial for the correctness of our reduction, and in particular for the validity of (6). Item 3 shows that the volume of the ellipsoid shrinks by a factor of e -1 /( 8 d ) whenever an update occurs. Item 4 bounds the total number of such updates by O ( d 2 log ( R / r )) .

The next lemma formalizes our reduction result in (6), showing that the instantaneous regret of Algorithm 2 is bounded by the instantaneous regret of the inner ONS algorithm. The lemma also bounds key quantities discussed in Section 4.1 that appear in the ONS regret bound.

Lemma 4.2. Let T ≥ 2 , η &gt; 0 , and c ∈ K be given and suppose that Assumption 2.2 and Assumption 2.1 hold with 0 &lt; r ≤ R and G &gt; 0 , respectively. Consider a call to Algorithm 2 with input ( T, c , r, R, G, η ) . Then, for any subgradients ( g t ) , the variables in Algorithm 2 satisfy for all t ∈ [ T ] :

1. ∥̃ g t ∥ ≤ G ( 1 + 4 κd ) ;
2. w t ∈ K ;
3. For all u ∈ K , ∣⟨ u -u t , ̃ g t ⟩∣ ≤ 2 RG ( 1 + 4 d ) ;

<!-- formula-not-decoded -->

The proof of the lemma is in Appendix E.1. Item 1 bounds the norm of the feedback vectors (̃ g t ) passed to the inner ONS algorithm. While the norm of ̃ g t can be as large as O ( Gκd ) , this is acceptable because the regret bound for ONS only involves ∥̃ g t ∥ inside logarithmic terms (see (4) and recall that B = max t ∈[ T ] ∥̃ g t ∥ ). This is precisely why the choice of ONS as the inner algorithm is essential to our approach. Item 2 guarantees the feasibility of the iterates of Algorithm 2. Item 3 shows that the inner product ∣⟨ u t -u , ̃ g t ⟩∣ is at most O ( RGd ) for all u ∈ K , and crucially, this bound is independent of κ . This is key to ensuring that the regret of the inner ONS algorithm does not depend on κ outside of logarithmic terms; see (4). Finally, Item 4, a consequence of our specific choice of ̃ g t in (5), shows that, on rounds where no thin direction is detected, the instantaneous regret of Algorithm 2 is bounded by that of the inner ONS algorithm (up to a small additive term).

With these results in place, we now proceed to bound the regret of the inner ONS algorithm. The bound in the next lemma should be reminiscent of the ONS bound displayed in (3) (proof in Appendix D).

Lemma 4.3 (ONS Regret). Let T ≥ 2 , η &gt; 0 , and c ∈ K be given and suppose that Assumption 2.2 and Assumption 2.1 hold with 0 &lt; r ≤ R and G &gt; 0 , respectively. Consider a call to Algorithm 2 with input ( T, c , r, R, G, η ) . Then, for any ( g t ) , the vectors (̃ g t , u t , c t , H t ) in Algorithm 2 satisfy

<!-- formula-not-decoded -->

and u t ∈ E t ∶= E( c t , H t ) , ∀ t ≥ 1 . Further, given ̃ g t , computing u t + 1 costs at most O ( d ω log ( TR / r )) .

The proof closely follows the original analysis of the ONS algorithm by [11], despite the fact that we are working with a sequence of varying feasible sets (E t ) ; what is important here is that we have K ⊆ E t , for all t ≥ 1 . One key difference from the analysis of [11] is that we do not assume access to an exact oracle for generalized projection, which involves solving inf u ∈C ∥ u -c t ∥ 2 H for a convex set C and matrix H ∈ S d × d &gt; 0 . Instead, we use the PoE subroutine (Algorithm 3), which approximates this projection in O ( d ω log ( TR / r )) time (see Lemma C.1) using binary search when C = E t , for t ≥ 1 . Note that this translates into a ̃ O (C sep (K)+ d ω ) per-round cost for our OCO algorithm (Algorithm 2). We formalize this next and give the main regret guarantee for Algorithm 2 (proof in Appendix E.2).

Theorem 4.1 (Main Regret). Let T ≥ 2 and c ∈ K be given and suppose that Assumption 2.2 and Assumption 2.1 hold with 0 &lt; r ≤ R and G &gt; 0 , respectively. Consider a call to Algorithm 2 with input ( T, c , r, R, G, η ) , for η ≤ 1 10 dGR . Then, for any ( g t ) , the iterates ( w t ) of Algorithm 2 satisfy

<!-- formula-not-decoded -->

for all u ∈ K . Further, the per-round cost of the algorithm is at most O (( C sep (K)+ d ω )⋅ log ( TR / r )) .

Regret bound after tuning η . By Assumption 2.1 and the fact that w t ∈ K ⊆ B ( R ) , we have that ∣⟨ g t , w t -u ⟩∣ ≤ 2 RG , for all t ∈ [ T ] . Thus, setting η = ( GR √ T log ( Tκ )) -1 ∧ ( 10 dGR ) -1 in Theorem 4.1 gives a O ( GR √ dT log ( κT ) + d 2 log ( κT )) regret bound for Algorithm 2 as desired.

Rates for stochastic optimization. By combining this regret bound with a standard online-to-batch conversion technique (e.g., [10]), we obtain a convergence rate of ̃ O ( √ d / T + d 2 / T ) for stochastic convex optimization. However, due to the presence of the 'second-order' term ∑ T t = 1 ⟨ g t , w t -u ⟩ 2 in (10), we can achieve an improved rate of ̃ O ( σ √ d / T + d 2 / T ) , which simplifies to ̃ O ( d 2 / T ) when the gradient noise σ is zero; see Theorem F.1 for a formal statement.

## 5 Discussion on Computational Efficiency

A key consideration for our algorithm is its computational runtime, particularly in comparison to classical projection-based methods and other projection-free approaches. The per-iteration computational cost of our proposed algorithm is on the order of ̃ O ( d ω + C sep ) , where ω ≈ 2 . 372 is the exponent of matrix multiplication, and this cost arises from solving a d × d linear system. The C sep term is the cost of a single call to the separation oracle.

While the d ω term appears significant (larger than the d 2 in [19], for instance), it is crucial to contextualize this cost. In many large-scale optimization settings, the separation cost C sep is the dominant computational bottleneck. A canonical example is optimization over a polytope defined by m linear constraints in R d : K = { x ∈ R d ∣ a ⊺ i x ≤ b i , ∀ i ∈ [ m ]} .

In this common setup, a call to the separation oracle reduces to membership testing, which requires computing m inner products. The separation cost is therefore C sep = O ( md ) . The d ω term in our complexity is dominated by this separation cost as soon as m ≥ d ω -1 ≈ d 1 . 37 . In this regime, our per-iteration cost becomes ̃ O ( md ) , which is cheaper than Euclidean projection as we discuss next.

## 5.1 Cost of Euclidean Projection in the Polytope Setting

We compare the projection cost of two standard methods in the canonical polytope setting.

Using an Interior-Point Method (IPM), the cost of a single projection is ̃ O ( √ m ⋅ ( d ω + md 2 )) (see, e.g., [1]); here, √ m is the number of IPM iterations, and d ω + md 2 is the per-iteration cost, which stems from computing the Hessian of the logarithmic barrier ( md 2 ) and inverting it ( d ω ).

Another alternative for projection is to use a cutting-plane method [24]. The complexity of this approach is ̃ O ( d ⋅ C sep + d 4 ) per projection. In the polytope setting, this becomes ̃ O ( md 2 + d 4 ) .

We see that the computational cost of both projection methods is significantly higher than our method's per-iteration cost of ̃ O ( md + d ω ) when m ≥ d .

While per-iteration cost is important, a more holistic measure of efficiency is the overall runtime required to reach a target ε -average regret. We analyze this next.

## 5.2 Runtimes to Achieve an ε -Average Regret

We compare our method's runtime for achieving an ε -average regret against OGD with projections performed with either IPM or a cutting-plane method in the polytope setting.

Since OGD has a O ( √ T ) regret, the total runtime of OGD to achieve an ε -average regret is bounded by O ( C Proj / ε 2 ) , where C Proj is the cost of projection. Thus, instantiating the projection costs for the IPM and cutting-planes methods in Section 5.1, we get the runtimes:

<!-- formula-not-decoded -->

The comparable runtime for our proposed approach is ̃ O ( md 2 + d ω + 1 ) ⋅ ε -2 . Thus, our method's runtime is faster than OGD with the projection approaches we discussed, as long as m ≥ d .

We note, however, that newer cutting-plane methods exist with an improved complexity, running in ̃ O ( md 2 + d 3 ) time per projection [15]. Consequently, pairing OGD with these state-of-the-art methods would result in a runtime to achieve an ε -average regret that is comparable to our approach, or potentially slightly better due to an improved lower-order term ( d 3 instead of d ω + 1 ). However, these new methods are significantly more involved and often hide large constant factors in the ̃ O (⋅) notation, which may limit their practical applicability.

Frank-Wolfe (FW) Methods. Finally, we position our work relative to projection-free methods based on linear optimization, such as Frank-Wolfe (FW). These methods are indeed very useful and practical, and their dimension-free guarantees are particularly appealing. We view these approaches as complementary to ours. Our algorithm can be more competitive in popular settings where, in the absence of additional structure (like a well-studied combinatorial structure), the linear optimization oracle required by FW becomes computationally expensive; in some cases, comparable to a full projection such as in the canonical polytope case discussed above.

## Acknowledgments and Disclosure of Funding

We thank the reviewers for their constructive feedback on the presentation of our computational efficiency analysis.

## References

- [1] Stephen P Boyd and Lieven Vandenberghe. Convex optimization . Cambridge university press, 2004.
- [2] Sébastien Bubeck. Convex optimization: Algorithms and complexity, 2015.
- [3] Nicolo Cesa-Bianchi and Gábor Lugosi. Prediction, learning, and games . Cambridge university press, 2006.
- [4] Ashok Cutkosky. Anytime online-to-batch, optimism and acceleration. In International Conference on Machine Learning , pages 1446-1454. PMLR, 2019.
- [5] Ashok Cutkosky, Harsh Mehta, and Francesco Orabona. Optimal stochastic non-smooth non-convex optimization through online-to-non-convex conversion, 2023.
- [6] Abraham D Flaxman, Adam Tauman Kalai, and H Brendan McMahan. Online convex optimization in the bandit setting: gradient descent without a gradient. In Proceedings of the sixteenth annual ACM-SIAM symposium on Discrete algorithms , pages 385-394, 2005.
- [7] Marguerite Frank, Philip Wolfe, et al. An algorithm for quadratic programming. Naval research logistics quarterly , 3(1-2):95-110, 1956.
- [8] Dan Garber and Ben Kretzu. New projection-free algorithms for online convex optimization with adaptive regret guarantees. In Conference on Learning Theory , pages 2326-2359. PMLR, 2022.
- [9] Benjamin Grimmer. Radial duality part i: foundations. Mathematical Programming , 205(1):3368, 2024.
- [10] Elad Hazan. Introduction to online convex optimization. Foundations and Trends® in Optimization , 2(3-4):157-325, 2016.
- [11] Elad Hazan, Amit Agarwal, and Satyen Kale. Logarithmic regret algorithms for online convex optimization. Machine Learning , 69(2-3):169-192, 2007.
- [12] Elad Hazan and Satyen Kale. Projection-free online learning. In Proceedings of the 29th International Coference on International Conference on Machine Learning , pages 1843-1850, 2012.
- [13] Jean-Baptiste Hiriart-Urruty and Claude Lemaréchal. Fundamentals of convex analysis . Springer Science &amp; Business Media, 2004.
- [14] DB Iudin and Arkadi S Nemirovskii. Evaluation of informational complexity of mathematicalprogramming programs. Matekon , 13(2):3-25, 1977.
- [15] Yin Tat Lee, Aaron Sidford, and Santosh S. Vempala. Efficient convex optimization with membership oracles, 2017.
- [16] László Lovász and Santosh Vempala. Simulated annealing in convex bodies and an o*(n4) volume algorithm. Journal of Computer and System Sciences , 72(2):392-417, 2006. JCSS FOCS 2003 Special Issue.
- [17] Zhou Lu, Nataly Brukhim, Paula Gradu, and Elad Hazan. Projection-free adaptive regret with membership oracles. In International Conference on Algorithmic Learning Theory , pages 1055-1073. PMLR, 2023.
- [18] Zakaria Mhammedi. Efficient projection-free online convex optimization with membership oracle. In Conference on Learning Theory , pages 5314-5390. PMLR, 2022.

- [19] Zakaria Mhammedi. Online convex optimization with a separation oracle. arXiv preprint arXiv:2410.02476 , 2024.
- [20] Marco Molinaro. Curvature of feasible sets in offline and online optimization. arXiv preprint arXiv:2002.03213 , 2020.
- [21] Phan Phien. Some quantitative results on lipschitz inverse and implicit functions theorems. arXiv preprint arXiv:1204.4916 , 2012.
- [22] Shai Shalev-Shwartz et al. Online learning and online convex optimization. Foundations and trends in Machine Learning , 4(2):107-194, 2011.
- [23] Naum Z Shor. Cut-off method with space extension in convex programming problems. Cybernetics , 13(1):94-96, 1977.
- [24] Pravin M Vaidya. A new algorithm for minimizing convex functions over convex sets. Mathematical programming , 73(3):291-341, 1996.
- [25] Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In International Conference on Machine Learning , pages 928-936, 2003.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: All the claims made in the abstract are formally stated in Section 4 and proven in the appendix.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: After the contribution paragraph in the introduction, we discuss the limitations of our work; we do not provide regret lower bounds or experimental validation for our algorithm.

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

Justification: The paper contains the full set of assumptions and formal result statements. The proofs will be included in the appendix.

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

Justification: This is a purely theoretical paper.

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

Justification: The paper does not include experiments.

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

Answer: [NA] .

Justification: The paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA] .

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

Justification: Reviewed code of ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: No societal impact; purely theoretical paper.

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

Justification: No risks. Purely theoretical paper.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: No use of existing assets.

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

Justification: No release of new assests.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: No crowdsourcing involvement.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: No crowdsourcing or research on human subjects.

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

## Contents of Appendix

| A   | Computing the Gauge Distance (Proofs of Lemma 2.1)           |   21 |
|-----|--------------------------------------------------------------|------|
| B   | Ellipsoid Updates                                            |   23 |
| C   | Generalized Projections onto Ellipsoids                      |   25 |
| D   | ONS Analysis (Proof of Lemma 4.3)                            |   30 |
| E   | OCOAnalysis                                                  |   33 |
| E.1 | Algorithm Invariants (Proofs of Lemma 4.1 and Lemma 4.2)     |   33 |
| E.2 | OCO Regret (Proofs of Theorem 4.1) . . . . . . . . . . . . . |   37 |
| F   | Rates for Stochastic Convex Optimization                     |   38 |
| G   | Helper Lemmas                                                |   40 |

## A Computing the Gauge Distance (Proofs of Lemma 2.1)

For the proof of Lemma 2.1, we need the next intermediate result showing that the output ( S, s ) of Algorithm 1 is such that ( u -c ) ⊺ s ≤ 1 for all u ∈ C and input vector c .

Lemma A.1. Let δ ∈ ( 0 , 1 ) , w , c ∈ R d , C ⊆ R d convex, and H ∈ R d × d PSD be given, and consider a call to Algorithm 1 with input (C , H, c , w , δ ) . Then, the output ( S, s ) of Algorithm 1 satisfies

<!-- formula-not-decoded -->

Proof. We first establish (11). If w ∈ C , then the 'if' condition on Line 2 evaluates to 'true', and so the algorithm returns the pair ( S, s ) = ( 0 , 0 ) , which clearly satisfies (11). Now, if c / ∈ C , then the 'if' condition on Line 6 evaluates to 'true', and so the algorithm returns the pair ( S, s ) = ( 0 , 3 d v / √ v ⊺ H v ) , where v is the unit-norm vector corresponding to the hyperplane separating c from C . Thus, by definition of a separating hyperplane, we have that for all u ∈ C , u ⊺ v ≤ c ⊺ v , which implies (11) after using that s = 3 d v / √ v ⊺ H v .

Now, suppose that w / ∈ C and c ∈ C . In this case, Algorithm 1 returns ( S, s ) , where s = v β ⋅( w -c ) ⊺ v and v is the vector returned by the last call to Sep C ( µ ( w -c ) + c ) before the algorithm returns; here, β and µ are as in Algorithm 1. We first verify that β ⋅ ( w -c ) ⊺ v &gt; 0 so that s is well defined. Since v is the vector returned by the last call to Sep C ( µ ( w -c ) + c ) , we have

<!-- formula-not-decoded -->

Instantiating this with u = c and using that µ = α + β 2 ≤ β , we get that

<!-- formula-not-decoded -->

and so s is well-defined in R d . Dividing both sides of (12) by β ⋅ ( w -c ) ⊺ v and using that s = v β ⋅( w -c ) ⊺ v , we get

<!-- formula-not-decoded -->

where the last inequality follows by the fact that µ = α + β ≤ β (see Algorithm 1).

## Proof of Lemma 2.1. We consider cases.

Case where c ∉ C . If c / ∈ C , then the 'if' condition on Line 6 evaluates to 'true', and so the algorithm returns the pair ( S, s ) = ( 0 , 3 d v / √ v ⊺ H v ) , where v is the vector returned by the call to Sep C ( c ) on Line 5. In this case, we clearly have that ∥ H 1 / 2 s ∥ &gt; 2 d .

For the rest of this proof, we assume that c ∈ C .

Case where c , w ∈ C . If w ∈ C , then the 'if' condition on Line 2 of Algorithm 1 evaluates to 'true', and so the algorithm returns the pair ( S, s ) = ( 0 , 0 ) . Since w ∈ C , we have γ Cc ( w -c ) = inf { λ ≥ 0 ∣ w -c ∈ λ ⋅ (C c )} ≤ 1 , and so, we have for all u ∈ R d :

<!-- formula-not-decoded -->

This implies the desired claim.

Case where c ∈ C and w ∉ C : approximate gauge value. For the rest of this proof, we assume that c ∈ C and w / ∈ C , and let α , β , µ , v , S , and s be as in Algorithm 1 when the algorithm returns.

If the condition on Line 12 of Algorithm 1 is satisfied, then by definition of s in Algorithm 1, we have that

<!-- formula-not-decoded -->

Moving forward, we consider the alternative case where ∥ H 1 / 2 s ∥ ≤ 2 d , which, by contrapositive, implies that the condition on Line 12 is never satisfied during the call to Algorithm 1. Thus, by design, when Algorithm 1 returns, we have

<!-- formula-not-decoded -->

2

<!-- formula-not-decoded -->

with the convention that 1 / 0 = +∞ . By definition of s in Algorithm 1, we also have that

<!-- formula-not-decoded -->

where the last inequality follows by the fact that ∥ H 1 / 2 s ∥ ≤ 2 d and the assumption that w ∈ E in the lemma statement. Using (14), ∣ β -α ∣ ≤ δ 8 d 2 , and the fact that δ ∈ ( 0 , 1 ) implies that α &gt; 0 . Further, (14) with ∣ β -α ∣ ≤ δ 8 d 2 also imply that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (15) follows by the fact that 1 1 -x ≤ 1 + 2 x , for all x ≤ 1 2 ; we instantiate the latter with x = δ 8 βd 2 which clearly satisfies x ≤ 1 2 since 1 β ≤ 2 d and δ ∈ ( 0 , 1 ) . Combining (16) with (13) and using that S = α -1 -1 (see Algorithm 1), we get

<!-- formula-not-decoded -->

This together with the facts that S Cc ( w -c ) = max ( 0 , γ Cc ( w -c )-1 ) and γ Cc ( w -c ) ≥ 1 (since w / ∈ C and c ∈ C ) implies that S Cc ( w -c ) ≤ S ≤ S Cc ( w -c ) + δ , as desired.

Case where c ∈ C and w ∉ C : approximate gauge subgradient. Still in the same case as the previous paragraph, we now show that the second output s of Algorithm 1 is an approximate subgradient of the gauge distance function at w .

By Lemma A.1, we have

<!-- formula-not-decoded -->

This implies that s ∈ (C c ) ○ (by definition of the polar set) and so by Lemma G.5.a, we have

<!-- formula-not-decoded -->

On the other hand, combining (16) with (13), we get

<!-- formula-not-decoded -->

where the equality uses the expression of s . Combining (17) and (18) implies that

<!-- formula-not-decoded -->

which, in turn, implies (through the change of variable u ← u -c )

<!-- formula-not-decoded -->

Thus, subtracting 1 from both sides and using that S Cc ( w -c ) = γ Cc ( w -c ) -1 (since w / ∈ C and c ∈ C ), we get that

<!-- formula-not-decoded -->

Thus, we have shown that when ∥ H 1 / 2 s ∥ ≤ 2 d , the outputs ( S, s ) of Algorithm 1 are such that S approximates the Gauge distance S Cc ( w -c ) and s is an approximate subgradient of S Cc at w -c , as desired

Number of oracle calls. The number of oracle calls is bounded by the number of iterations of the 'while' loop on Line 10 plus the two calls to Sep C before the while loop. Since Algorithm 1 implements a bisection, the number of iterations in the while loop is at most log 2 ( 8 d 2 δ ) . Thus, the total number of calls to the oracle is at most log 2 ( 8 d 2 δ -1 ) + 2 .

## B Ellipsoid Updates

In this section, we provide the intuition behind the ellipsoid update rule used in Algorithm 2. As discussed in Section 4.1, when the condition for updating the current ellipsoid E t = E( c t , H t ) ⊇ K is satisfied at round t of Algorithm 2 (i.e., when Line 4 is satisfied), we have:

<!-- formula-not-decoded -->

We show that under this condition, an updated ellipsoid E t + 1 can be constructed such that it still contains K and its volume is reduced relative to E t .

To achieve this, we consider the more abstract problem of identifying the minimum volume ellipsoid that encloses the set { u ∈ B ( 1 ) ∣ u ⊺ v ≤ ε } for some unit-norm vector v ; this is the same set as in (19) but with ( v t , 1 /( 2 d )) replaced by ( v , ε ) . The construction of this ellipsoid follows a similar approach to that found in the classical ellipsoid method (see , e.g., [2]). A sketch of this construction, which mirrors the steps outlined in [2], is provided below.

Ellipsoid parameterization. Fix t &gt; 0 and ε ∈ ( 0 , 1 ) , and define H -1 a,b = a vv ⊺ + b ⋅ ( I d -vv ⊺ ) , for a, b &gt; 0 . We will start by solving the intermediate problem of finding parameters a, b ∈ R such that the ellipsoid E t,a,b ∶= { x ∈ R d ∣ ( x + t v ) ⊺ H -1 a,b ( x + t v ) ≤ 1 } contains the set given by

<!-- formula-not-decoded -->

We will derive expressions for a and b as a function of t , then tune t so that the corresponding ellipsoid has the smallest volume.

Choosing a and b . Observe that to satisfy the requirement that { u ∈ B ( 1 ) ∣ v ⊺ u ≤ ε } ⊆ E t,a,b , it suffices to choose parameters a, b ∈ R such that the boundary ∂ E t,a,b of the ellipsoid E t,a,b contains -v (which is in ∂ B ( 1 ) ) and the intersection set B ( 1 ) ∩ { u ∈ R d ∣ v ⊺ u = ε } . This requirement translates to the following constraints:

<!-- formula-not-decoded -->

Solving (20) for a, b , we get

<!-- formula-not-decoded -->

Tuning t for the smallest-volume ellipsoid. Now that we have derived expressions for a and b as a function of t such that { u ∈ B ( 1 ) ∣ v ⊺ u ≤ ε } ⊆ E t,a,b , we will tune t to obtain the smallest-volume ellipsoid E t,a,b . Note that the volume of the ellipsoid E t,a,b satisfies

<!-- formula-not-decoded -->

For ε = 1 2 d and ( a, b ) as in (21), the parameter t that minimizes the right-hand side of (22) is given by

<!-- formula-not-decoded -->

This leads to H -1 a,b = H -1 = 4 d 2 -4 4 d 2 -1 ( I d + 2 d ( 2 d + 1 )( d -1 ) vv ⊺ ) and by Sherman-Morisson:

<!-- formula-not-decoded -->

Further, by (22), it can be shown that with the choices of ( a, b ) and t as in (21) and (23), respectively, we have

<!-- formula-not-decoded -->

We formalize this result in Lemma B.1. Before presenting this lemma, we give a few comments.

The informal sketch presented above shows that when H -1 / 2 t (K c t ) ⊆ { u ∈ B ( 1 ) ∣ u ⊺ v t ≤ 1 / 2 d } (a condition satisfied whenever the ellipsoid update in Algorithm 2 is triggered), then

<!-- formula-not-decoded -->

Further, it can be verified that c t + H 1 / 2 t E ̃ t + 1 = E( c t + 1 , H t + 1 ) , with c t + 1 and H t + 1 as defined on Line 7 and Line 8 of Algorithm 2. Thus, K ⊆ E t + 1 ∶= E( c t + 1 , H t + 1 ) as discussed in Section 4.1.

We also note that the choice of ε scaling inversely with d in the construction just described was necessary to show the volume decrease in (24). This puts a limit on the maximum value ε can take and contributes to the additive ̃ O ( d 2 ) term in the final regret bound of Algorithm 2.

We now formalize the guarantee of our construction.

Lemma B.1. Let v ∈ R d be such that ∥ v ∥ = 1 , and define H ∶= 4 d 2 -1 4 d 2 -4 ⋅ ( I d -2 d 2 d 2 + d -1 vv ⊺ ) and

<!-- formula-not-decoded -->

Then, we have

<!-- formula-not-decoded -->

Furthermore, it holds that vol (E) ≤ e -1 8 d ⋅ vol ( B ( 1 )) .

Proof. We first prove (25). Let H be as in the lemma statement. By Sherman-Morisson, we have

<!-- formula-not-decoded -->

Thus, for all u ∈ R d , we have

<!-- formula-not-decoded -->

We need to prove that the right-hand side is at most 1 for all u ∈ { x ∈ B ( 1 ) ∣ v ⊺ x ≤ 1 2 d } . Note that for any such u , we have u ⊺ v ∈ [-1 , 1 2 d ] . Therefore, we have that for all u ∈ { x ∈ B ( 1 ) ∣ v ⊺ x ≤ 1 2 d } :

<!-- formula-not-decoded -->

Now, note that the function

<!-- formula-not-decoded -->

is convex, and so its maximum in the interval [-1 , 1 2 d ] must be attained at the endpoints -1 and 1 2 d . Therefore, we have that for all u ∈ { x ∈ B ( 1 ) ∣ v ⊺ x ≤ 1 2 d } :

<!-- formula-not-decoded -->

Combining this with (27) shows that ( u + 1 2 ( d + 1 ) v ) ⊺ H -1 ( u + 1 2 ( d + 1 ) v ) ≤ 1 for all u ∈ { x ∈ B ( 1 ) ∣ v ⊺ x ≤ 1 2 d } , and so

<!-- formula-not-decoded -->

Volume decrease. It remains to prove that vol (E) ≤ e -1 8 d ⋅ vol ( B ( 1 )) . The volume of the ellipsoid E is given by

<!-- formula-not-decoded -->

where Γ is the Gamma function . We now compute the determinant of H -1 . By (26), we can write

<!-- formula-not-decoded -->

where a = 4 ( 1 + d ) 2 ( 1 + d ) 2 and b = 4 d 2 -4 4 d 2 -1 ⋅ ( d -1 )( 2 d + 1 ) 2 d . Thus, for any z ⊥ v , we have H -1 z = b z . In addition, we have H -1 v = a v because ∥ v ∥ = 1 . Therefore, a and b are the only eigenvalues of H -1 with b having multiplicity d -1 , and so det ( H -1 ) = a d -1 b . Therefore, by (28)

<!-- formula-not-decoded -->

where the last inequality follows from the fact that

<!-- formula-not-decoded -->

and we are assuming that d ≥ 2 in this paper (see Section 2.1). Combining (29) with the fact that vol ( B ( 1 )) = π d / 2 / Γ ( d / 2 + 1 ) completes the proof.

## C Generalized Projections onto Ellipsoids

In this section, we consider the generalized projection problem

<!-- formula-not-decoded -->

for z ∈ R d and Q ∈ S d × d &gt; 0 in the special case where E is an ellipsoid; that is:

<!-- formula-not-decoded -->

for some H ∈ S d × d &gt; 0 and c ∈ R d . By Lagrangian duality, the problem in (30) is equivalent to the max-min problem:

<!-- formula-not-decoded -->

## Algorithm 3 PoE : Projection onto ellipsoid via binary search.

```
require: z , c ∈ R d , Q,H ∈ R d × d , and parameters λ -, λ + , δ > 0 . /* If z is almost in the ellipsoid E( c , H ) , then return a slightly 'scalled down' version of it ( z -c )/( 1 + δ ) + c . */ 1: if ( z -c ) ⊺ H -1 ( z -c ) ≤ 1 + δ then 2: return ( z -c )/( 1 + δ ) + c . /* Approximate u ⋆ ← arg min u ∈E( c ,H ) ( z -u ) ⊺ Q -1 ( z -u ) via binary search */ 3: Set α ← λ -, β ← λ + , and µ ←( α + β )/ 2 . 4: Set u ⋆ µ = ( HQ -1 + µI ) -1 ( HQ -1 z + µ c ) . 5: Set D = ( u ⋆ µ -c ) ⊺ H -1 ( u ⋆ µ -c ) . 6: while D ∉ [ 1 -δ, 1 ] do 7: if D < 1 -δ then 8: Set β ← µ and µ ←( α + β )/ 2 . 9: else 10: Set α ← µ and µ ←( α + β )/ 2 . 11: Set u ⋆ µ = ( HQ -1 + µI ) -1 ( HQ -1 z + µ c ) . 12: Set D = ( u ⋆ µ -c ) ⊺ H -1 ( u ⋆ µ -c ) . 13: return u ⋆ µ .
```

For a fixed λ ≥ 0 , the inner minimization problem has a closed-form solution:

<!-- formula-not-decoded -->

By leveraging this structure, we show that the problem in (31) can be solved efficiently via binary search over the dual variable λ . In this paper, we do this using Algorithm 3. This algorithm takes inputs z , c , Q , and H , and outputs an approximate solution to the problem in (30). We now state the guarantee of the algorithm.

Lemma C.1 (Projection onto ellipsoid). Let R &gt; 0 , δ &gt; 0 be given. Let z , c ∈ B ( R ) , and Q,H ∈ S d × d &gt; 0 be such that ( z -c ) ⊺ H -1 ( z -c ) &gt; 1 + δ and σ min ( H ) ≤ R 2 . Further, define E( ν ) ∶= { u ∈ R d ∣ ( u -c ) ⊺ H -1 ( u -c ) ≤ 1 -ν } , for ν ∈ R , and consider a call to Algorithm 3 with input ( z , c , Q, H, λ -, λ + , δ ) , where 0 ≤ λ -≤ δ ⋅ σ min ( H ) 2 24 R 2 ⋅ σ max ( Q ) and λ + ≥ 4 σ max ( H )⋅ R σ min ( H ) 1 / 2 ⋅ σ min ( Q ) . Then, Algorithm 3 returns u ⋆ ∈ R d such that:

<!-- formula-not-decoded -->

The number of iterations in the while loop on Line 6 of Algorithm 3 is at most O ( log 2 ( Lλ + δλ -)) , where

<!-- formula-not-decoded -->

Finally, each iteration of the while loop in Algorithm 3 can be performed in O ( d ω ) time, where ω is the matrix multiplication exponent.

Remark C.1 (Computational cost of PoE ). By Lemma C.1, the number of iterations in the whileloop of Algorithm 3 is bounded by log ( Lλ + δλ -) . We later show that for our application, we have Lλ + δλ -≤ poly ( R r , T ) . Note also that the cost per iteration of Algorithm 3 is bounded by the cost of solving a linear system (which costs O ( d ω ) ), and so total cost of running Algorithm 3 is bounded by O ( d ω log Lλ + δλ -) . It is possible to implement Algorithm 3 so that the total cost is bounded by O ( d 2 log Lλ + δλ -+ d 3 ) instead, where now the dominant term O ( d 3 ) is independent of any logarithmic factor. 3 This can be done as follows:

3 Even though O ( d ω log Lλ + δλ -) is technically better than O ( d 2 log Lλ + δλ -+ d 3 ) asymptotically (since ω &lt; 3 ), the O (⋅) notation in the former typically hides large constants making the new implementation described in the remark more favorable in practice.

- Compute Cholesky decompositions of Q and H (costs O ( d 3 ) ); that is, compute lowertriangular matrices L Q and L H such that Q = L Q L ⊺ Q and H = L H L ⊺ H .
- Compute SVD decomposition of L ⊺ H L -⊺ Q L -1 Q L H (costs O ( d 3 ) ); that is, compute ( Λ , U ) such that M = U Λ U ⊺ , UU ⊺ = I , and Λ = diag ( ρ 1 , . . . , ρ d ) .
- Compute inverses L -1 H and Q -1 (costs O ( d 3 ) ).

With this, we have for any µ ≥ 0 :

<!-- formula-not-decoded -->

Thus, given ( L H , L Q , L -1 H , Q -1 , Λ , U ) , we can compute ( HQ -1 + µI ) -1 ( HQ -1 z + µ c ) in O ( d 2 ) (matrix-vector multiplication costs) for any µ ≥ 0 . Thus, the total cost PoE with this implementation is bounded by O ( d 2 log Lλ + δλ -+ d 3 ) because the operations described in the bullet points (which cost O ( d 3 ) ) need to be performed only once.

The computational cost of a Euclidean projection onto an arbitrary set K can be much worse than that of PoE in Remark C.1. For example, using state-of-the-art ellipsoid methods to project a point onto a set K specified by a separation oracle can incur a cost of up to ̃ O ( d ⋅ C sep (K) + d 3 ) , where C sep denotes the cost of a single separation oracle call [15]. Moreover, the ̃ O (⋅) notation often conceals large constants, which can render these methods impractical. Alternatively, a Euclidean projection can be formulated as a quadratic program and solved using an interior point method. This approach requires a self-concordant barrier for the set K whose gradients and Hessians are inexpensive to compute. However, even under favorable conditions, the associated cost typically remains higher than the costs detailed in Remark C.1.

Proof of Lemma C.1. Let z , c ∈ B ( R ) and Q,H ∈ S d × d &gt; 0 be given. For λ ≥ 1 , let us define

<!-- formula-not-decoded -->

Setting the gradient of the objective to zero and solving for u , we obtain

<!-- formula-not-decoded -->

We now study how the 'constraint' objective g ( λ ) ∶= ( u ⋆ λ -c ) ⊺ H -1 ( u ⋆ λ -c ) varies as a function of λ . Taking the derivative of g gives

<!-- formula-not-decoded -->

On the other hand, by the expression of u ⋆ λ in (32), we have

<!-- formula-not-decoded -->

Plugging this into (33), we get that

<!-- formula-not-decoded -->

where the last inequality follows by the fact that the matrix H -1 ( Q -1 + λH -1 ) -1 H -1 is positive definite. Therefore, g ( λ ) is non-increasing in λ , and so we can find λ ⋆ using binary search. To show that this can be done efficiently, it remains to identify a reasonably small interval for the values of λ ⋆ .

Upper bound on λ ⋆ . Note that

<!-- formula-not-decoded -->

For λ ≥ 4 σ max ( H ) R σ min ( H ) 1 / 2 σ min ( Q ) , we have by the stability of the inverse operator (Lemma G.1)

<!-- formula-not-decoded -->

where we used that σ min ( H ) 1 / 2 ≤ R , by assumption. Using (35) and (36), we get

<!-- formula-not-decoded -->

where the last inequality uses that z , c ∈ B ( R ) . Thus, for all λ ≥ 4 σ max ( H ) R σ min ( H ) 1 / 2 σ min ( Q ) , we have g ( λ ) ≤ 1 . Since g ( λ ) is non-increasing, and we know that g ( λ ⋆ ) = 1 , we must have that

<!-- formula-not-decoded -->

Lower bound on λ ⋆ . For the other direction, we have

<!-- formula-not-decoded -->

And so, by Lemma G.1, as long as λ ≤ ∥ QH -1 ∥ -1 op / 2 , we have

<!-- formula-not-decoded -->

Moving forward, we let E = λQH -1 -λ ( HQ -1 + λI ) -1 and assume that λ ≤ λ ⋆ (recall that λ ⋆ ≤ ∥ QH -1 ∥ -1 op / 2 ). With this and (35), we have that

<!-- formula-not-decoded -->

Now, let E ′ ∶= -λQH -1 + E , and note that by the triangle inequality and (37), we have

<!-- formula-not-decoded -->

where the last inequality follows from the fact that λ ≤ ∥ QH -1 ∥ -1 op / 2 . On the other hand, by (38):

<!-- formula-not-decoded -->

Thus, by (39) and the fact that z , c ∈ B ( R ) , we have that

<!-- formula-not-decoded -->

Therefore, if λ ≤ δ ⋅ σ min ( H ) 2 24 R 2 ⋅ σ max ( Q ) , we get that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and so by the assumptions on λ -and λ + it holds that

<!-- formula-not-decoded -->

Now that we have established that λ ⋆ is in between λ -and λ + , we can use binary search to find λ ⋆ such that g ( λ ⋆ ) ≈ 1 . The number of iterations of the binary search will depend on the ratio λ + / λ -, the precision δ , and the Lipschitz constant of g .

Thus, we must have

Lipschitz constant of g . Since u ⋆ λ = ( HQ -1 + λI ) -1 ( HQ -1 z + λ c ) and z , c ∈ B ( R ) , we have that

<!-- formula-not-decoded -->

On the other hand, we have that

<!-- formula-not-decoded -->

and from (34)

<!-- formula-not-decoded -->

Therefore, by the triangle inequality and the fact that z , c ∈ B ( R ) , we get

<!-- formula-not-decoded -->

## Number of iterations of binary search. Given that

- 0 &lt; λ -≤ λ ⋆ ≤ λ + ,
- g ( λ + ) ≤ g ( λ ⋆ ) = 1 , and
- g is non-increasing in λ and L -Lipschitz,

Algorithm 3 finds a µ ∈ [ λ -, λ + ] such that

<!-- formula-not-decoded -->

after at most N = O ( 1 ) ⋅ log 2 ( λ + L λ -δ ) iterations of binary search on Line 6. Note that as soon as Algorithm 3 finds such a µ , it returns u = u ⋆ µ (see Line 6 and Line 13 of Algorithm 3).

Optimality of u ⋆ . Let µ ∶= 1 - ( u ⋆ -c ) ⊺ H -1 ( u ⋆ -c ) , where u ⋆ = u ⋆ µ is the vector returned by the call to Algorithm 3, and note that we have just shown that ν ∈ [ 0 , δ ] (see (40)). Now, consider the problem in the lemma statement:

<!-- formula-not-decoded -->

This problem can equivalently be written as:

<!-- formula-not-decoded -->

where f 0 ( u ) ∶= ( u -z ) ⊺ Q -1 ( u -z ) and f ( u ) ∶= ( u -c ) ⊺ H -1 ( u -c ) . By the definition of u ⋆ , it can be verified that

<!-- formula-not-decoded -->

Furthermore, we have f ( u ⋆ ) = g ( µ ) = 1 -ν (by (40) and the definition of ν ). Therefore, the primal-dual pair ( u ⋆ , µ ) satisfies the KKT conditions for the convex problem in (41), and so we have

<!-- formula-not-decoded -->

as desired. This completes the proof of Lemma C.1.

## D ONS Analysis (Proof of Lemma 4.3)

Proof. Let ( H t , c t , z t , u t , Σ t , ̃ g t , η ) be as in Algorithm 2. First, note that for any t ∈ [ T ] such that √ s ⊺ t H t s t &gt; 2 d , we have ̃ g t = 0 . Fix t ∈ [ T ] such that √ s ⊺ t H t s t ≤ 2 d . By Lemma 2.1 and Assumption 2.2, we have

<!-- formula-not-decoded -->

For the rest of this proof, we fix u ∈ K . By definition of z t + 1 in Algorithm 2, we have

<!-- formula-not-decoded -->

Multiplying both sides by Σ 1 / 2 t + 1 , we get

<!-- formula-not-decoded -->

Taking the norm of both sides and squaring leads to

<!-- formula-not-decoded -->

Using that ∥ u t -u ∥ 2 Σ t + 1 = ∥ u t -u ∥ 2 Σ t + η ⟨ u t -u , ̃ g t ⟩ 2 (since Σ t + 1 = Σ t + η ̃ g t ̃ g ⊺ t ) and rearranging (42) gives

<!-- formula-not-decoded -->

We will apply Lemma C.1 (guarantee of PoE ) to bound the term 1 2 ∥ z t + 1 -u ∥ 2 Σ t + 1 on left-hand side of (43). Then, summing the resulting inequality over t = 1 , . . . , T will give us the desired result.

Invoking the guarantee of PoE . Note that at iteration t , Algorithm 2 calls PoE with input ( z , c , Q, H, λ -, λ + , δ ) = ( z t + 1 , c t + 1 , βR 2 Σ -1 t + 1 , H t + 1 , λ min , λ max , ε ) , where ε = 1 κ 18 T 2 , λ min = ε 24 κ 2 , λ max = 40 d 2 κ ε , and β = ηG 2 ( G is as in Assumption 2.1). To invoke Lemma C.1, we need to check that the conditions on λ -and λ + are satisfied. By Lemma 4.1, we have that

<!-- formula-not-decoded -->

On the other hand, since Σ t + 1 = βI + η ∑ t τ = 1 ̃ g τ ̃ g ⊺ τ and ∥̃ g t ∥ ≤ G ( 1 + 4 κd ) (by Lemma 4.2), we have

<!-- formula-not-decoded -->

Therefore, since ( Q,H,δ ) = ( βR 2 Σ -1 t + 1 , H t + 1 , ε ) , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, the condition on λ -in Lemma C.1 is satisfied. Now, we show that the condition on λ + is satisfied. We have

<!-- formula-not-decoded -->

Thus, the condition on λ + is also satisfied. Therefore, we can apply Lemma C.1 in this proof. In particular, Lemma C.1 implies that

<!-- formula-not-decoded -->

which we will use in the sequel.

Computational cost. The cost of computing the new iterate u t + 1 given g t is dominated by the cost of the PoE call on Line 15 of Algorithm 2. Thus, by Lemma C.1, this cost is at most O ( d ω ⋅ log ( λ max L λ min ε )) , where

<!-- formula-not-decoded -->

and ( Q,H,δ ) = ( βR 2 Σ -1 t + 1 , H t + 1 , ε ) . Using (44) and (45), we get that λ max L λ min ε ≤ poly ( T,R / r ) , and so the cost of computing u t + 1 given ̃ g t is at most O ( d ω ⋅ log ( TR / r )) .

Bounding the instantaneous ONS regret. We now bound the instantaneous regret ⟨ u t -u , ̃ g t ⟩ of the inner ONS algorithm within Algorithm 2. We consider cases based on the value of

<!-- formula-not-decoded -->

Case where ∥ z t + 1 -c t + 1 ∥ 2 H -1 t + 1 ≤ 1 + ε . If ( z t + 1 -c t + 1 ) ⊺ H -1 t + 1 ( z t + 1 -c t + 1 ) ≤ 1 + ε , then PoE returns u t + 1 = ( z t + 1 -c t + 1 )/( 1 + ε ) + c t + 1 ∈ E t + 1 . In this case, by the triangle inequality, we have

<!-- formula-not-decoded -->

where the last inequality follows by (44) and (45). Now, by (46), we have u t + 1 ∈ E t + 1 . On the other hand, by Lemma 4.1, K ⊆ E t + 1 and so u ∈ E t + 1 . Therefore, ∥ u t + 1 -u t ∥ H -1 t + 1 ≤ 2 , and so

<!-- formula-not-decoded -->

where the last inequality follows by (45) and (44). Squaring (47) and using (48), we get

<!-- formula-not-decoded -->

Case where ∥ z t + 1 -c t + 1 ∥ 2 H -1 t + 1 &gt; 1 + ε . Now, suppose that ( z t + 1 -c t + 1 ) ⊺ H -1 t + 1 ( z t + 1 -c t + 1 ) &gt; 1 + ε . In this case, by Lemma C.1, there exists ν ∈ [ 0 , ε ] such that

<!-- formula-not-decoded -->

where E t + 1 ( ν ) ∶= { x ∈ R d ∣ ( x -c t + 1 ) ⊺ H -1 t + 1 ( x -c t + 1 ) ≤ 1 -ν } ⊆ E t + 1 . Fix such a ν . By Lemma 2.1, we have that K ⊆ E t + 1 , and so u ∈ E t + 1 . This in turn implies that

<!-- formula-not-decoded -->

where we used that 1 1 + x ≤ 1 -x 2 , for x ∈ [ 0 , 1 / 2 ] . Thus, by (50) and Lemma G.2, we have

<!-- formula-not-decoded -->

On the other hand, by the expression of u ν and the triangle inequality, we have that

<!-- formula-not-decoded -->

where the last inequality follows by (45) and (44). Similarly, we have that

<!-- formula-not-decoded -->

Thus, combining (51) and (52), we get

<!-- formula-not-decoded -->

Taking the square in (53) and using (48), we get

<!-- formula-not-decoded -->

Plugging (49) and (54) into (43) yields

<!-- formula-not-decoded -->

So far, we have considered rounds t ∈ [ T ] satisfying √ s ⊺ t H t s t ≤ 2 d . As remarked at the beginning of this proof, when √ s ⊺ t H t s t &gt; 2 d , we have ̃ g t = 0 , and so (55) remains true since Σ t + 1 = Σ t and u t + 1 = u t .

Bounding the ONS regret. We now bound the regret of ONS (not just the instantaneous regret). Summing (55) over t ∈ [ T ] and telescoping the terms (∥ u t -u ∥ 2 ) , we get

Σ t

<!-- formula-not-decoded -->

On the other hand, by [11, Lemma 11] and Lemma 4.2, we have that

<!-- formula-not-decoded -->

Plugging this into (56) and using that β = ηG 2 and ε = 1 T 2 κ 18 , we get

<!-- formula-not-decoded -->

This completes the proof.

## E OCOAnalysis

## E.1 Algorithm Invariants (Proofs of Lemma 4.1 and Lemma 4.2)

Proof of Lemma 4.1. We will show the claim via induction over t = 1 , . . . , T . The base case holds trivially because H 1 = R 2 I and K ⊆ B ( R ) = E 1 , where the set inclusion follows by Assumption 2.2. Now, suppose that the claim holds for t ∈ [ T -1 ] and we show that it holds for t + 1 . Note that ( c t + 1 , H t + 1 ) ≠ ( c t , H t ) only if √ s ⊺ t H t s t &gt; 2 d . Suppose that √ s ⊺ t H t s t &gt; 2 d . By Lemma A.1, we have that

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

This implies that for the unit-norm vector v t = H 1 / 2 t s t √ s ⊺ t H t s t , we have that

<!-- formula-not-decoded -->

And, by the induction hypothesis, we also have that H -1 / 2 t (K c t ) ⊆ B ( 1 ) . Combining this with (57) implies that

<!-- formula-not-decoded -->

Now, by Lemma B.1, we have that

<!-- formula-not-decoded -->

where ̃ H t + 1 ∶= 4 d 2 -1 4 d 2 -4 ⋅ ( I d -2 d 2 d 2 + d -1 v t v ⊺ t ) . Combining (59) with (58) implies that

<!-- formula-not-decoded -->

This means that

<!-- formula-not-decoded -->

and by using the definitions of ̃ H t + 1 , H t + 1 , v t , and c t + 1 , we have

<!-- formula-not-decoded -->

Thus, (60) implies that K ⊆ E t + 1 .

Showing Item 2. This item follows from Lemma 2.1.

Showing Item 3. If √ s ⊺ t H t s t ≤ 2 d , then ( c t + 1 , H t + 1 ) = ( c t , H t ) and Item 3 follows immediately by the induction hypothesis. Now, assume that √ s ⊺ t H t s t &gt; 2 d . In this case, we have H t + 1 = 4 d 2 -1 4 d 2 -4 ⋅ H 1 / 2 t ( I -2 d 2 d 2 + d -1 H 1 / 2 t s t s ⊺ t H 1 / 2 t s ⊺ t H t s t ) H 1 / 2 t . Thus,

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

and so by Lemma B.1, we have that

On the other hand, we have

<!-- formula-not-decoded -->

where the last equality follows by the fact that vol (E t ) = π d / 2 Γ ( d / 2 + 1 ) 1 √ det ( H -1 t ) and ∣ det ( H -1 / 2 t )∣ = √

det ( H -1 t ) because H t is positive definite. Now, using that vol ( B ( 1 )) = π d / 2 / Γ ( d / 2 + 1 ) , we get that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining this with (61), we get as desired. This proves Item 3.

Showing Item 4. By Item 1 and Assumption 2.2, we have that B ( c 0 , r ) ⊆ K ⊆ E t + 1 . Therefore, vol (E t + 1 ) ≥ r d . On the other hand, by Item 3, we have vol (E t + 1 ) ≤ e -Nt 8 d ⋅ R d , and so N t ≤ 8 d 2 log ( R / r ) ; otherwise, we would contradict vol (E t + 1 ) ≥ r d .

Showing Item 5. We now prove Item 5 in the lemma statement. By Assumption 2.2, there exists c 0 ∈ B ( R ) such that B ( c 0 , r ) ⊆ K , and by Item 1 we have B ( c 0 , r ) ⊆ E t + 1 . Thus, by Lemma G.3, we have

<!-- formula-not-decoded -->

Let z t + 1 be the unit-norm eignevector corresponding to the largest eigenvalue of H -1 t + 1 . Since r z t + 1 + c t + 1 ∈ B ( c t + 1 , r ) ⊆ E t + 1 , we have that

<!-- formula-not-decoded -->

Rearranging implies that σ min ( H t + 1 ) ≥ r 2 . It remains to bound σ max ( H t + 1 ) . By Lemma G.4, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, by the induction hypothesis, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (62) follows from ( 1 + x / n ) n ≤ e x for all n ≥ 1 and ∣ x ∣ ≤ n .

Proof of Lemma 4.2. Fix t ∈ [ T ] , and let S t , s t , u t , ̃ g t , and w t be as in Algorithm 2. When the condition in Line 4 of Algorithm 2 is satisfied (i.e., when √ s ⊺ t H t s t &gt; 2 d ), we simply have ̃ g t = 0 and w t = c ∈ K , in which case all the items hold. For the rest of the proof, we assume that

<!-- formula-not-decoded -->

By Lemma 2.1, we have that c t ∈ K . Using this, the triangle inequality, Assumption 2.2, and Hölder's inequality, we have

<!-- formula-not-decoded -->

where the last inequality follows from (63) and the fact that σ min ( H t ) ≥ r 2 by Lemma 4.1. This shows Item 1.

Proving Item 2. By definition of w t , we have w t = u t -c t 1 + S t + c t (recall that we are in the case where √ s ⊺ t H t s t ≤ 2 d ). Therefore, by the homogeneity of the Gauge function (see Lemma G.5), we have

<!-- formula-not-decoded -->

where the last inequality follows from S Kc t ( u t -c t ) ≤ S t by Lemma 2.1. Eq. (64) implies that w t -c t ∈ K c t by definition of the Gauge function (see Definition 2.2), which is equivalent to w t ∈ K .

Proving Item 3. We now show Item 3. Fix u ∈ K . Using the expression of ̃ g t and the triangle inequality, we have that

<!-- formula-not-decoded -->

and since c t ∈ K (Lemma 4.1) and w t ∈ K ⊆ B ( R ) by Item 2 and Assumption 2.2, we have

<!-- formula-not-decoded -->

where (65) follows by the fact that K ⊆ E t (Lemma 4.1) and that u t ∈ E t ; this is because u t is the output of A which is constrained to output a vector in E t at round t .

and so by Item 4, we have

Proving Item 4. We now prove that

<!-- formula-not-decoded -->

For this, define the surrogate loss function ℓ t :

<!-- formula-not-decoded -->

Since the pair ( S t , s t ) is the output of GaugeDist ( u t , K , H t , c t , ε ) with ε = 1 κ 8 T 2 , we have by Lemma 2.1:

<!-- formula-not-decoded -->

Now, since w t = ( u t -c t )/( 1 + S t ) + c t (see Algorithm 2) and S t ≥ S Kc t ( u t -c t ) ≥ 0 (by Lemma 2.1), we have that -I {⟨ g t , u t -c t ⟩ &lt; 0 } ⋅ ⟨ g t , w t -c t ⟩ ≥ 0 . And so, using (67) and the definition of ℓ t , we get

<!-- formula-not-decoded -->

where the last inequality uses that w t ∈ K ⊆ B ( R ) (by Item 2 and Assumption 2.2), c t ∈ K (by Lemma 4.1), and ∥ g t ∥ ≤ G (Assumption 2.1).

It remains to prove that ⟨ g t , w t -u ⟩ ≤ ℓ t ( u t ) -ℓ t ( u ) , for all u ∈ K . First, note that we have for all u ∈ K , S Kc t ( u -c t ) = max ( 0 , γ Kc t ( u -c t ) -1 ) = 0 (by definition of the Gauge function), and so

<!-- formula-not-decoded -->

We will now compare ⟨ g t , w t ⟩ to ℓ t ( u t ) by considering cases. Suppose that S t = 0 . In this case, we have w t = u t and so ⟨ g t , w t ⟩ = ⟨ g t , u t ⟩ = ℓ t ( u t ) . Now suppose that S t &gt; 0 and ⟨ g t , u t -c t ⟩ ≥ 0 . In this case, since w t = u t -c t 1 + S t + c t , we have

<!-- formula-not-decoded -->

Now suppose that S t &gt; 0 and ⟨ g t , u t -c t ⟩ &lt; 0 . Again, using that w t = u t -c t 1 + S t + c t , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality follows from the fact that σ max ( H t ) ≤ κ 16 R 2 (by Lemma 4.1) and ∥ g t ∥ ≤ G (Assumption 2.1). Rearranging (71), we get

<!-- formula-not-decoded -->

By combining (68), (69), (70), and (72), we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## E.2 OCORegret (Proofs of Theorem 4.1)

Proof. Fix u ∈ K . By Lemma 4.2, the sequence of vectors (̃ g t ) satisfies (̃ g t ) ⊂ B ( ̃ G ) with ̃ G = G ⋅ ( 1 + 4 κd ) . Thus, by invoking Lemma 4.3, we get

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where in the last step we used that η ≤ 1 10 dGR . We now prove that

<!-- formula-not-decoded -->

which together with (73) would complete the proof. Using Lemma 4.1, we have ∀ t ∈ [ T ] , ∣⟨̃ g t , u t -u ⟩∣ ≤ 2 RG ( 1 + 4 d ) . (74)

Combining this with the facts that:

- I { √ s ⊺ t H t s t ≤ 2 d } ⋅ ⟨ g t , w t -u ⟩ ≤ ⟨̃ g t , u t -u ⟩ + 3 GR T , for all t ∈ [ T ] (by Lemma 4.2); · x → x -η 2 x 2 in non-decreasing for all x ≤ 1 η (we instantiate this with x = I { √ s ⊺ t H t s t ≤ 2 d } ⋅ ⟨ g t , w t -u ⟩ and x = ⟨̃ g t , u t -u ⟩ + 3 GR T ); and
- η ≤ 1 10 dGR ;

we get that for all t ∈ [ T ]

<!-- formula-not-decoded -->

where the last step follows by (74) and η ≤ 1 10 dGR . Summing this over t = 1 , . . . , T , we obtain

<!-- formula-not-decoded -->

where the last inequality follows by the fact that ∑ T t = 1 I { √ s ⊺ t H t s t ≤ 2 d } ≤ 8 d 2 log ( R / r ) (by Lemma 4.1); u t , u ∈ K ⊆ B ( R ) ; and g t ∈ B ( G ) , for all t ∈ [ T ] . Rearranging (75) and combining with (73), we get the desired result.

Computational cost. The per-round computational cost of Algorithm 2 is dominated by the calls of the GaugeDist (Algorithm 1) and PoE (Algorithm 3) subroutines. As established in Lemma 2.1, the cost of a single call to GaugeDist is at most O ( C sep (K) ⋅ log ( TR / r )) . Similarly, Lemma 4.3 show that the cost of a single call to PoE is bounded by O ( d ω ⋅ log ( TR / r )) . Consequently, the total per-round cost of Algorithm 2 is O (( d ω + C sep (K)) ⋅ log ( TR / r )) .

## F Rates for Stochastic Convex Optimization

In this section, we use our regret bound from Theorem 4.1 to derive a state-of-the-art convergence rate for projection-free stochastic convex optimization that only depends on the asphercity κ of the set K logarithmically. We start by stating our assumptions for the stochastic optimization setting.

Assumption F.1. There is a function f ∶ K → R and parameters σ ≥ 0 and G &gt; 0 such that the loss vector g t that the algorithm receives at round t ≥ 1 is of the form g t = ¯ g t + ξ t , where

- For all t ≥ 1 , ¯ g t ∈ ∂f ( w t ) , where w t is the output of the algorithm at round t ;
- ( ξ t ) ⊂ R d are i.i.d. noise vectors such that E [ ξ t ] = 0 and E [∥ ξ t ∥ 2 ] ≤ σ 2 , for all t ≥ 1 ; and
- For all t ≥ 1 , ∥ ¯ g t ∥ ≤ G .

The conditions in Assumption F.1 are standard in the stochastic convex optimization literature; see, e.g., [19]. Under Assumption F.1, we now state the guarantee of Algorithm 2 when setting the input parameter η to

<!-- formula-not-decoded -->

where κ ∶= R / r and r, R &gt; 0 are as in Assumption 2.2.

Theorem F.1. Let T ≥ 2 and c ∈ K be given and suppose that Assumption 2.2 and Assumption 2.1 hold with 0 &lt; r ≤ R and G &gt; 0 , respectively. Consider a call to Algorithm 2 with input ( T, c , r, R, G, η ) , for η as in (76). Then, for κ ∶= R / r , we have

<!-- formula-not-decoded -->

where ̂ w T ∶= 1 T ∑ T t = 1 w t and ( w t ) are the iterates of Algorithm 2. The computational cost is at most

<!-- formula-not-decoded -->

The proof of Theorem F.1 is very similar to that of [19, Theorem 5.1].

Proof. Let u ⋆ ∈ arg min u ∈K f ( u ) . Using Jensen's inequality, we get

<!-- formula-not-decoded -->

Now, by instantiating the bound in Theorem 4.1 with comparator u ⋆ ∈ K , we get:

<!-- formula-not-decoded -->

Now, by Assumption F.1 (in particular, the fact that g t = ¯ g t + ξ t ) together with the fact that ( a + b ) 2 ≤ 2 a 2 + 2 b 2 and w t , u ⋆ ∈ B ( R ) , we have

<!-- formula-not-decoded -->

On the other hand, by definition of u ⋆ and the facts that w , w 1 , w 2 , ⋅ ⋅ ⋅ ∈ B ( R ) , we have for all t ∈ [ T ] :

<!-- formula-not-decoded -->

where the last inequality follows by the fact that w t ∈ K (Lemma 4.2) and that u ⋆ is the minimizer of f within K . Note that (80) implies that for all t ∈ [ T ] ,

<!-- formula-not-decoded -->

Picking up from (79), we get

<!-- formula-not-decoded -->

Plugging this into (78) and rearranging, we get

<!-- formula-not-decoded -->

Taking the expectation on both sides and using that E [ g t ] = ¯ g t and E [∥ ξ t ∥ 2 ] ≤ σ 2 , we get

<!-- formula-not-decoded -->

where the last inequality follows by the fact that η ≤ 1 4 GR and (77). Now, dividing by T 2 on both sides and rearranging, we get

<!-- formula-not-decoded -->

where the last inequality follows by η ≤ 1 10 dGR . Note that the optimal tuning of η in (82) is given by

<!-- formula-not-decoded -->

We now consider cases.

Case where η ⋆ ≤ 1 10 dGR . First, note that this implies that η = η ⋆ . And so, plugging this into (82), we get

<!-- formula-not-decoded -->

Case where η ⋆ ≥ 1 10 dGR . In this case, we have η = 1 10 dGR . Now, using that η ⋆ ≥ 1 10 dGR and the expression of η ⋆ , we have

<!-- formula-not-decoded -->

Plugging η = 1 10 dGR into (82) and using (84), we get

<!-- formula-not-decoded -->

where the last inequality follows by (84). Thus, combining (83) and (85), we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This proves the desired convergence rate.

## G Helper Lemmas

Lemma G.1 ([21]). Let A,E ∈ R d × d be such that A is invertible and r = ∥ A -1 E ∥ &lt; 1 . Then,

<!-- formula-not-decoded -->

Lemma G.2. Let C be a convex set and let H ∈ S d × d &gt; 0 and z ∈ R d be given. Further, let u ⋆ ∈ arg min u ∈C ∥ u -z ∥ H . Then,

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

Lemma G.3. Let r &gt; 0 , c 0 , c ∈ R d , and H ∈ S d × d &gt; 0 be given, and define E ∶= { u ∈ R d ∣ ( u -c ) ⊺ H -1 ( u -c ) ≤ 1 } . If B ( c 0 , r ) ⊆ E , then B ( c , r ) ⊆ E .

Proof. Let X + ∶= B ( c 0 , r ) and suppose that X + ⊆ E . Since E is centrally-symmetric around c (i.e., -( u -c ) + c ∈ E for all u ∈ E ), we have -( B ( c 0 , r ) -c ) + c ⊆ E . Since B ( r ) = -B ( r ) , this implies that

<!-- formula-not-decoded -->

Now, since E is convex, we have that

<!-- formula-not-decoded -->

Fix z ∈ B ( r ) . We have that z -c 0 + 2 c ∈ X -and z + c 0 ∈ X + and so by (86), we have z + c ∈ E , which establishes that B ( c , r ) ⊆ E and completes the proof.

Lemma G.4. Let H ∈ R d × d be a positive semi-definite matrix, and let v ∈ R d ∖{ 0 } be given. Define

<!-- formula-not-decoded -->

Then, we have σ max ( H v ) ≤ ( 1 + 2 / d 2 ) ⋅ σ max ( H ) .

Proof of Lemma G.4. Since ( H -2 d 2 d 2 + d -1 H vv ⊺ H v ⊺ H v ) /uni2AAF H , we have that

<!-- formula-not-decoded -->

where the last inequality follows from the fact that 4 d 2 -1 4 d 2 -4 ≤ 1 + 2 d 2 for d ≥ 2 and that H is positive semi-definite. This implies that σ max ( H v ) ≤ ( 1 + 2 / d 2 ) ⋅ σ max ( H ) .

We need the following properties of the Gauge function (see e.g. [20] for a proof).

Lemma G.5. Let w ∈ R d ∖ { 0 } and 0 &lt; r ≤ R . Further, let C be a closed convex set such that B( r ) ⊆ C ⊆ B( R ) . Then, the following properties hold:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->