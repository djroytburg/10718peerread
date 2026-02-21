## A Single-Loop First-Order Algorithm for Linearly Constrained Bilevel Optimization

## Wei Shen

University of Virginia zyy5hb@virginia.edu

## Minhui Huang

Meta mhhuang@meta.com

## Jiawei Zhang

University of Wisconsin-Madison jzhang2924@wisc.edu

## Cong Shen

University of Virginia cong@virginia.edu

## Abstract

Westudy bilevel optimization problems where the lower-level problems are strongly convex and have coupled linear constraints. To overcome the potential nonsmoothness of the hyper-objective and the computational challenges associated with the Hessian matrix, we utilize penalty and augmented Lagrangian methods to reformulate the original problem as a single-level one. Especially, we establish a strong theoretical connection between the reformulated function and the original hyper-objective by characterizing the closeness of their values and derivatives. Based on this reformulation, we propose a single-loop, first-order algorithm for linearly constrained bilevel optimization (SFLCB). We provide rigorous analyses of its non-asymptotic convergence rates, showing an improvement over prior double-loop algorithms - form O ( ϵ -3 log( ϵ -1 )) to O ( ϵ -3 ) . The experiments corroborate our theoretical findings and demonstrate the practical efficiency of the proposed SFLCB algorithm. Simulation code is provided at https://github.com/ShenGroup/SFLCB .

## 1 Introduction

In recent years, bilevel optimization (BLO) has gained significant popularity for addressing a wide range of modern machine learning problems, such as hyperparameter optimization [36, 7, 33], data hypercleaning [38], meta learning [37, 13], reinforcement learning [41, 11] and neural architecture search [26, 23]; see survey papers [52, 27, 40] for additional discussions. While numerous works for unconstrained BLO problems have been proposed [9, 30, 15, 5, 11, 22], studies focusing on constrained BLO problems are relatively limited.

In this paper, we consider the following BLO problem where the lower-level (LL) problem has coupled constraints :

<!-- formula-not-decoded -->

The upper-level (UL) objective function f : R d x × R d y → R and the lower-level objective function g : R d x × R d y → R are continuously differentiable. Moreover, we assume that g ( x, y ) is strongly convex with respect to y . The feasible sets are defined as X = R d x , Y ( x ) = { y ∈ R d y | h ( x, y ) ≤ 0 } where h : R d x × R d y → R d h .

For this setting, we develop a single loop algorithm for the special case h ( x, y ) = Bx + Ay -b , where B ∈ R d h × d x and A ∈ R d h × d y . This special class of constrained BLO problems covers a wide class of applications, including distributed optimization [48], hyperparameter optimization of constrained learning problems [46] and adversarial training [53] and draw significant attentions [42, 18, 20].

A popular approach for solving unconstrained BLO is implicit gradient descent [9, 15, 14, 4, 19]. For constrained BLO, several studies have extended this approach to accommodate different constraint settings [42, 18, 46, 45]. However, these implicit gradient-based methods in constrained BLO necessitate computing the Hessian matrix of the lower-level problem [42, 18, 46, 45]. The potential computational challenges associated with the Hessian matrix limit their practical applicability for large-scale problems.

Recently, some first-order methods [21, 50, 49, 16, 20] have been proposed for addressing constrained BLO problems. Most of those works considered transforming the original problem (1) into a singlelevel one and trying to find the stationary points of the reformulated problem. For example, [50, 49] reformulated the original problem into some approximated functions and proposed single-loop algorithms for finding the stationary point of the approximated problem. However, neither [50] nor [49] establishes clear relationships between the stationary points of their approximated problems and the original one.

Works most closely related to ours are those by [21] and [16], both of which reformulated the problem (1) as

<!-- formula-not-decoded -->

and considered optimizing the following function with a penalty parameter δ :

<!-- formula-not-decoded -->

where Φ δ ( x, y, z ) = f ( x, y ) + 1 δ [ g ( x, y ) -g ( x, z )] . Based on this reformulation, [21] proposed algorithms for solving BLO with LL constraints y ∈ Y and [16] proposed algorithms for solving coupled LL constraints Y ( x ) = { y ∈ R d y | h ( x, y ) ≤ 0 } . However, [21] only considered the LL constraints Y that are independent of x and their methods require projection oracle to Y at each iteration. Algorithms in [16] require complex double or triple loops, resulting in sub-optimal convergence rates and difficult implementation. Moreover, the connection between the stationary point of the reformulated function Φ δ and the original hyper-objective Φ is not discussed in [21, 16] for coupled constraints Y ( x ) .

To address these limitations, in this paper, we establish a rigorous theoretical justification for this reformulation (3) and propose a single-loop Hessian-free algorithm for the linearly constrained cases. Our main contributions can be summarized as follows.

- We establish a rigorous theoretical connection between the reformulated function Φ δ and the original hyper-objective Φ by proving the closeness of their values and derivatives under coupled constraints Y ( x ) = { y ∈ R d y | h ( x, y ) ≤ 0 } with certain assumptions, which provides strong justifications for the reformulation (3).
- Based on this reformulation and equipped with augmented Lagrangian methods, we proposed SFLCB, a single-loop, first-order algorithm for linearly constrained bilevel optimization problem, and provide rigorous analyses of its non-asymptotic convergence rates, achieving an improvement in the convergence rate from O ( ϵ -3 log( ϵ -1 )) to O ( ϵ -3 ) compared to prior works (See Table 1 for a more comprehensive comparison of our work with previous studies). The simple single-loop structure also makes our algorithm easier to implement in practice compared to [16].
- Our experiments on hyperparameter optimization in the support vector machine (SVM) and transportation network design problems validate the practical effectiveness and efficiency of the proposed SFLCB algorithm.

## 2 Related works

BLO without constraints. One popular approach for solving unconstrained BLO is to use implicit gradient descent methods [36]. It is well established that when the LL problem is strongly convex

Table 1: Comparison of our paper with [21, 16]. More detailed introductions and discussions of other related works can be found in Section 2. Here, the 'Complexity' means the iteration complexity needed to achieve the ϵ -stationary point of Φ δ (3).

| Methods                                               | LL Constraint                                                                                                                                                                                             | Complexity                                                                                                  | Loops                                            |
|-------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|--------------------------------------------------|
| [21] [16] [16] SFLCB (ours) SFLCB (ours) SFLCB (ours) | y ∈ Y , Y is a convex and compact set h ( x, y ) ≤ 0 , LICQ holds B ( x )+ A ( x ) y ≤ 0 , A ( x ) is full row rank Bx + Ay - b ≤ 0 , A is full row rank Ay ≤ 0 , LICQ holds at the initial points Ay ≤ 0 | O ( ϵ - 3 log( ϵ - 1 )) O ( ϵ - 5 log( ϵ - 1 )) O ( ϵ - 3 log( ϵ - 1 )) O ( ϵ - 3 ) O ( ϵ - 3 ) O ( ϵ - 4 ) | single/double triple double single single single |

and unconstrained, y ∗ ( x ) = argmin y g ( x, y ) exists and is differentiable, and the gradient of the hyper-objective can be calculated by ∇ Φ( x ) = ∇ x f ( x, y ) + ( ∇ y ∗ ( x )) ⊤ ∇ y f ( x, y ∗ ( x )) [9]. Later works improved the convergence rates and studied the gradient descent methods under various settings [15, 14, 4, 19, 47]. Another popular approach is based on iterative differentiation, which iteratively solves the LL problems and computes ∇ y ∗ ( x ) to approximate the hypergradient [34, 10, 29, 2]. Recently, penalty-based methods have gained traction as a promising approach for solving BLO. Those works usually reformulate the original BLO as the single-level one and use the first-order methods to find the stationary point of the reformulated problems [28, 35, 25, 22, 39, 8, 31, 32].

BLO with constraints. There are two primary types of methods for solving constrained bilevel optimization problems. One is based on the implicit gradient method. Generally, when the LL problem has constraints, the differentiabilities of y ∗ ( x ) and Φ( x ) are not guaranteed [18]. [42] proved the existence of ∇ Φ( x ) under additional assumptions for linearly constraint Ay ≤ b and proposed an implicit gradient-type double-loop algorithm. [18] proposed a perturbation-based smoothing technique to compute the approximate implicit gradient for linearly constraint Ay ≤ b . [46] used Clarke subdifferential to approximate the non-differentiable implicit function Φ . However, they only provided an asymptotic convergence analysis of their algorithm. [45] proved the existence of ∇ ϕ where the LL has equality constraints Ay + H ( x ) = c , and introduced an alternating projected SGD approach to solve this problem. However, these implicit gradient-type algorithms [42, 18, 46, 45] require the computations for the Hessian matrix of the LL problems, which potentially limit their practical applicability for large-scale problems.

Another commonly used approach for solving constrained BLO problems is based on penalty reformulation. For example, [32] reformulated unconstrained and constrained BLO problems as structured minimax problems and introduced first-order methods with guarantees for finding ϵ -KKT solutions. [50] reformulated the original problem into a proximal Lagrangian value function and proposed a single-loop, first-order method to find the stationary points of the reformulated value function. However, their algorithm requires the implementation of the projection operator on C = { x, y | h ( x, y ) ≤ 0 } at each iteration, which can be potentially costly. [49] reformulated the original problem into a doubly regularized gap function and proposed a single-loop, first-order algorithm. Compared to [50], [49] did not need the projection operator to the coupled constraint set. However, both [50] and [49] did not establish very clear relationships between the stationary points of their approximated problems and the original one. For example, [49] only provided an asymptotic relationship between the original problem and their reformulated one, i.e., as their penalty parameter approaches infinite, their reformulated problem is equivalent to the original one. Recently, [43, 17] proposed algorithms based on barrier approximation approach for constrained BLO problems. However, their algorithms also require the computations for the Hessian matrix.

[21] and [16] considered the same reformulation as ours. [21] studied the case where the LL variables y ∈ Y are independent of x and characterized the conditions under which the values and derivatives of Φ and Φ δ can be O ( δ ) -close for y ∈ Y constraints. Compared with [21], we prove similar results under coupled constraints Y ( x ) = { y ∈ R d y | h ( x, y ) ≤ 0 } . Moreover, the algorithms in [21] require the implementation of the projection operator to Y at each iteration, which can be costly. [16] studied the coupled constraints Y ( x ) = { y ∈ R d y | h ( x, y ) ≤ 0 } . While [16] considered more general constraints than ours, however, it did not characterize the gap between the stationary point of the reformulated function Φ δ and the original hyper-objective Φ . Our Theorem 4.9 provides further

justifications for their reformulation in coupled constraints. Moreover, compared with the double- and triple-loop algorithms in [21], we propose a single-loop algorithm SFLCB and prove an improvement in the convergence rate from O ( ϵ -3 log( ϵ -1 )) to O ( ϵ -3 ) .

Recently, [20] also proposed first-order methods for linearly constrained BLO. Especially, they proved a nearly optimal convergence rate ˜ O ( ϵ -2 ) for linear equality constraints and proposed algorithms that can attain ( δ, ϵ ) -Goldstein stationarity for linear inequality constraints. However, their convergence rates for linear inequality constraints either have additional dependence on dimension d (such as ˜ O ( dδ -1 ϵ -3 ) ) or need additional assumptions to access the exact optimal dual variable (such as ˜ O ( δ -1 ϵ -4 ) ), while we do not require the exact optimal dual variable assumption. Compared with the double-loop algorithms in [20], our proposed single-loop one is easier to implement in practice. Moreover, our techniques are also different from theirs under linear inequality constraints, thereby highlighting the distinct contributions and independent interests of our work.

## 3 Preliminaries

Notation. For vectors a, b ∈ R d , we denote a ≤ b if for all i ∈ [ d ] , a i ≤ b i . We use ∥ · ∥ to denote the l 2 norm of a vector and the spectral norm of a matrix. We define the projection operator that project x to a set P as Π P ( x ) = argmin x ′ ∈P 1 2 ∥ x -x ′ ∥ 2 . We denote the projection operator that projects a x ∈ R d to the set R d -as Π -( x ) .

We state the following assumptions for problem (1), which are commonly used in the theoretical studies of BLO.

Assumption 3.1. For any x ∈ X , Y ( x ) is nonempty, closed, and convex and Φ( x ) is lower bounded by a finite, Φ ∗ = inf x ∈X Φ( x ) ≥ -∞ .

Assumption 3.2. f, ∇ f, ∇ g are Lipschitz continuous with l f, 0 , l f, 1 , l g, 1 respectively, jointly over X × Y ( x ) .

Assumption 3.3. For any fixed x ∈ X , g ( x, y ) is µ g -strongly convex with respect to y ∈ Y ( x ) .

We introduce the standard definition of the ϵ -stationary point for a differentiable function.

Definition 3.4. We say ˆ x is an ϵ -stationary point of a differentiable function f if ∥∇ f (ˆ x ) ∥ ≤ ϵ .

## 4 Reformulation

In this section, we provide a theoretical justification for our reformulation (3) and establish the conditions under which the function values and gradients of the reformulated function Φ δ and the original hyper-objective Φ become sufficiently close. Note that in this section, we considered general coupled constraints Y ( x ) = { y ∈ R d y | h ( x, y ) ≤ 0 } which include, but are not limited to, the linear constraint case. The complete proofs for the lemmas and theorems in this section can be found in Appendix C.

First, we assume δ ≤ µ g 2 l f, 1 and introduce the following notations:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similar to Theorem 3.8 in [21], we have the following theorem to bound the difference between Φ and Φ δ , as well as y ∗ ( x ) and y ∗ δ ( x ) in the coupled constraints.

Theorem 4.1. When Assumption 3.1, 3.2 and 3.3 hold, we have

<!-- formula-not-decoded -->

Theorem 4.1 characterizes how the difference in function values and the optimal LL variables between the reformulated problem and the original one are controlled by the penalty parameter δ . Therefore, by choosing a sufficiently small δ , i.e., δ = O ( ϵ ) , we can treat the reformulated problem min x ∈X Φ δ ( x ) as an approximation of the original problem and solve this approximated problem instead. In the following lemmas, we will provide the conditions under which the reformulated function Φ δ ( x ) is differentiable. Before that, we first introduce the well-known and commonly used Linear Independence Constraint Qualification (LICQ) condition.

Definition 4.2 (Active set) . We denote I y ⊆ [ d h ] as the active set of y , i.e. I y = { i ∈ [ d h ] | h i ( x, y ) = 0 } .

Definition 4.3 (LICQ) . We say a point y satisfy the LICQ condition if, for all i ∈ I y , ∇ y h i ( x, y ) are linearly independent.

Then, similar to Lemmas 2 and 3 in [16], we have the following lemma.

Lemma 4.4. When Assumption 3.1, 3.2, 3.3 hold and δ ≤ µ g / (2 l f, 1 ) , if, for all x ∈ X , the LICQ condition (Definition 4.3) holds for y ∗ ( x ) and y ∗ δ ( x ) , then there exist the corresponding unique Lagrangian multipliers λ ∗ ( x ) ∈ R d h and λ ∗ δ ( x ) ∈ R d h such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, we have

<!-- formula-not-decoded -->

While the gradients of Φ δ ( x ) exists under LICQ conditions, for general problem (1), Φ( x ) is not guaranteed to be differentiable. For example, [18] provides an example where the LICQ condition holds, Φ( x ) is non-differentiable at some points. However, if a given x satisfies the following conditions, then ∇ Φ( x ) exists at x .

Assumption 4.5 (Strict Complementarity) . Let λ ∗ ( x ) be the Lagrange multipliers for y ∗ ( x ) (4). For any i ∈ I y ∗ ( x ) , [ λ ∗ ( x )] i &gt; 0 .

Assumption 4.6. ∇ 2 f, ∇ 2 g are Lipschitz continuous with l f, 2 , l g, 2 respectively, jointly over X × Y ( x ) . For i ∈ [ d h ] , h i ( x, y ) is convex with respect to y , h i , ∇ h i , ∇ 2 h i are respectively Lipschitz continuous with l h, 0 , l h, 1 , l h, 2 jointly over X × Y ( x ) .

Note that Assumption 4.5, 4.6 are commonly used in constrained BLO literature [42, 18, 46, 17, 21] to ensure the existence of ∇ Φ( x ) .

Lemma 4.7 (Theorem 2 in [46]) . When Assumption 3.1, 3.2, 3.3, 4.6 hold, if, for a given x , Assumption 4.5 and LICQ (Definition 4.3) condition hold for y ∗ ( x ) , then ∇ Φ( x ) exists at x .

Moreover, with additional assumptions, we can establish a non-asymptotic bound for ∥∇ Φ( x ) -∇ Φ δ ( x ) ∥ .

Assumption 4.8. For any t ∈ [0 , δ ] ,

- (1) y ∗ t ( x ) satisfies the LICQ condition (Definition 4.3) with the same active set as y ∗ ( x ) . Denote this active set as I . Let λ ∗ t ( x ) be the Lagrange multiplier for y ∗ t ( x ) in (5). For any i ∈ I , [ λ ∗ t ( x )] i &gt; 0 (Strict Complementarity). We assume ∥ λ ∗ t ( x ) ∥ ≤ Λ , where Λ is an O (1) constant.
- (2) Denote ∇ y ¯ h ( x, y ∗ t ( x )) = ∇ y [ h ( x, y ∗ t ( x ))] I . The singular values of ∇ y ¯ h ( x, y ∗ t ( x )) satisfy σ max ([ ∇ y ¯ h ( x, y ∗ t ( x ))) ≤ s max , σ min ( ∇ y ¯ h ( x, y ∗ t ( x ))) ≥ s min &gt; 0 , where s max , s min are O (1) constants.

Assumption 4.8 is made for t ∈ [0 , δ ] . When δ is sufficiently small, i.e., δ = O ( ϵ ) , y ∗ ( x ) and y ∗ t ( x ) are very close according to Theorem 4.1. Thus, we expect that for t ∈ [0 , δ ] , y ∗ t ( x ) will have similar properties as y ∗ ( x ) . Similar assumptions have also been used in [21] to establish the non-asymptotic bound for ∥∇ Φ( x ) -∇ Φ δ ( x ) ∥ .

Theorem 4.9. When Assumption 3.1, 3.2, 3.3, 4.6 hold and δ ≤ µ g / (2 l f, 1 ) , if Assumption 4.8 holds for a given x , we have

<!-- formula-not-decoded -->

Similar non-asymptotic bound for ∥∇ Φ( x ) -∇ Φ δ ( x ) ∥ has been established in [21]; however, their bound is established only for the LL constraints Y that do not depend on x . Our Theorem 4.9 provides a more general theoretical justification for the validity of the reformulation (3) for coupled constraints Y ( x ) = { y ∈ R d y | h ( x, y ) ≤ 0 } .

## 5 The SFLCB Algorithm

In the last section, we have justified the validity of our reformulation for coupled constrained BLO. In this section, we focus on a special and important case where the LL constraints are h ( x, y ) = Bx + Ay -b . This particular category of constrained BLO problems encompasses a broad range of applications, including distributed optimization [48, 18], adversarial training [53, 18], and hyperparameter optimization for constrained learning tasks such as hyperparameter optimization in SVM (see Section 6). For this special case h ( x, y ) = Bx + Ay -b , we introduce a novel single-loop, first-order algorithm SFLCB, which achieves an improvement in the convergence rate compared to prior works [21, 16].

First, we introduce the following slackness parameters α, β ∈ R d h -and define y ′ = ( y ⊤ , α ⊤ ) ⊤ , z ′ = ( z ⊤ , β ⊤ ) ⊤ . With these slackness parameters, we can convert the original inequality constraints to equality constraints, i.e. we can reformulate min x ∈X ,y ∈Y ( x ) max z ∈Y ( x ) ϕ δ ( x, y, z ) as:

<!-- formula-not-decoded -->

where P y = { y ∈ R d y , α ∈ R d h -} , S y ( x ) = { y, α ∈ P y | h ( x, y ) -α = 0 } . The Lagrangian of (6) with multiplier u, v ∈ R d h is

<!-- formula-not-decoded -->

According to Proposition 5.3.4 in [1], we know that

<!-- formula-not-decoded -->

Note that when δ ≤ µ g / (2 l f, 1 ) , ϕ δ is µ g / 2 -strongly convex with respect y . However, L δ ( x, y ′ , z ′ , u, v ) is only convex with respect y ′ and concave with respect z ′ . To make the objective function strongly convex with respect to y ′ and strongly concave with respect to z ′ , we can construct an augmented Lagrangian K :

<!-- formula-not-decoded -->

With 0 ≤ ρ 1 ≤ µ g -δl f, 1 σ 2 max ( A ) and 0 ≤ ρ 2 ≤ µ g σ 2 max ( A ) , according to Lemma D.1, K is strongly convex with respect to y ′ and strongly concave with respect to z ′ . Moreover, we have

<!-- formula-not-decoded -->

Note that L δ and K have the same optimal points and same optimal function value. Thus, we can reformulate the problem (6) to the minimax optimization problem over K :

<!-- formula-not-decoded -->

Motivated by these theoretical analyses, and applying gradient descent ascent (GDA) over problem (7), we propose SFLCB. A compact description can be found in Algorithm 1.

## Algorithm 1 SFLCB

```
Input: δ , ρ 1 , ρ 2 , η x , η y , η z , η v , η u , T Initialize: x 0 ∈ X , y ′ 0 , z ′ 0 ∈ P y , u 0 , v 0 ∈ R d h for t = 0 , 1 , ..., T -1 do u t +1 = u t + η u ( h ( x t , y t ) -α t ) v t +1 = v t + η v ( h ( x t , z t ) -β t ) x t +1 = x t -η x ∇ x K ( x t , y ′ t , z ′ t , u t +1 , v t +1 ) y ′ t +1 = Π P y { y ′ t -η y ∇ ′ y K ( x t , y ′ t , z ′ t , u t +1 , v t +1 ) z ′ t +1 = Π P y { z ′ t + η z ∇ ′ z K ( x t , y ′ t , z ′ t , u t +1 , v t +1 )
```

```
} } end for
```

## 5.1 Convergence results

In this section, we provide the non-asymptotic convergence results of SFLCB (Algorithm 1) for two constraint settings: 1) h ( x, y ) = Bx + Ay -b , where A is full row rank, and 2) h ( y ) = Ay -b , where A is not required to be full row rank.

Note that when the LICQ condition (Definition 4.3) holds for y ∗ ( x ) and y ∗ δ ( x ) , the optimal Lagrangian multipliers of y ∗ ( x ) and y ∗ δ ( x ) are unique. Thus, we first introduce the following lemma and notations for these optimal Lagrangian multipliers.

Lemma 5.1. When the LICQ condition (Definition 4.3) holds for y ∗ ( x ) and y ∗ δ ( x ) , the optimal Lagrangian multipliers of y ∗ ( x ) and y ∗ δ ( x ) are unique, and we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof of Lemma 5.1 can be found in Appendix E.

Next, we introduce the following notations:

<!-- formula-not-decoded -->

Then, we present the convergence results of SFLCB (Algorithm 1) for coupled constraints h ( x, y ) = Bx + Ay -b , where A is full row rank. Note that BLOCC in [16] that achieves the complexity of O ( ϵ -3 log( ϵ -1 )) also needs the matrix A to be full row rank (See Table 1). For A that is not full row rank and B = 0 , we provide the convergence results in Theorem 5.4 and Corollary 5.5.

Theorem 5.2. When h ( x, y ) = Bx + Ay -b , A is full row rank, Assumption 3.1, 3.2, 3.3 hold and δ = Θ( ϵ ) ≤ µ g / (2 l f, 1 ) , if we apply Algorithm 1 with appropriate parameters (see Appendix E), then we can find an ϵ -stationary point of Φ δ with a complexity of O ( ϵ -4 ) .

Moreover, if we have initial points x 0 , y 0 , z 0 , u 0 , v 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then we can find an ϵ -stationary point of Φ δ with a complexity of O ( ϵ -3 ) .

The formal statement and the complete proof of Theorem 5.2 can be found in Appendix E.

Proof sketch for Theorem 5.2 . The key new idea in our proof is the construction of a novel potential function V t and prove the descent lemma of V t (Lemma E.4). V t is defined as:

<!-- formula-not-decoded -->

where d ( x, z ′ , u, v ) = K ( x, y ′∗ δ ( x, u ) , z ′ , u, v ) and q ( x, v ) = ϕ δ ( x, y ∗ δ ( x ) , z ∗ ( x, v )) -v ⊤ ( A ′ z ′∗ ( x, v ) -b ) -ρ 2 2 ∥ A ′ z ′∗ ( x, v ) -b ∥ 2 . To prove Lemma E.4, we need to first prove several novel error bounds in Lemma D.2, Lemma E.3 and Lemma E.2. Those error bounds may be of

independent interest for solving other similar problems. The full row rank property of A is used in Lemma E.2 to bound ∥ u t +1 -u ∗ δ ( x t ) ∥ and ∥ v t +1 -v ∗ ( x t ) ∥ .

Since A has full row rank, then according to Theorem 6 in [16], we can easily find initial points satisfying (8)-(9) with a complexity of O (log( ϵ -1 )) and we have the following corollary.

Corollary 5.3. When h ( x, y ) = Bx + Ay -b , A has full row rank, Assumption 3.1, 3.2, 3.3 hold, and δ = Θ( ϵ ) ≤ µ g / (2 l f, 1 ) , if we apply projected gradient descent (PGD) for max v ∈ R d h + min z ∈ R dy g ( x 0 , z ) + v ⊤ ( Bx 0 + Az -b ) with a fixed x 0 , we can find ˆ v , ˆ z such that ∥ ˆ v -v ∗ ( x 0 ) ∥ ≤ δ and ∥ ˆ z -z ∗ ( x 0 ) ∥ ≤ δ with a complexity of O (log( ϵ -1 )) . Set y 0 = z 0 = ˆ z , u 0 = v 0 = ˆ v , α 0 = h ( x 0 , y 0 ) , β 0 = h ( x 0 , z 0 ) . With x 0 , y ′ 0 , z ′ 0 , u 0 , v 0 as initial points and applying Algorithm 1, we can find an ϵ -stationary point of Φ δ with a complexity of O ( ϵ -3 ) . Thus, the total complexity is O ( ϵ -3 +log( ϵ -1 )) = O ( ϵ -3 ) .

The proof of Corollary 5.3 can be found in Appendix E. Thus, compared to [16], we achieve an improvement in the convergence rate from O ( ϵ -3 log( ϵ -1 )) to O ( ϵ -3 ) for the coupled linear constraint (See Table 1).

Additionally, we have the following convergence results for constraints h ( y ) = Ay -b , where A is not required to have a full row rank.

Theorem 5.4. When h ( x, y ) = Ay -b , Assumption 3.1, 3.2, 3.3 hold, and δ = Θ( ϵ ) ≤ µ g / (2 l f, 1 ) , if we apply Algorithm 1 with appropriate parameters (see Appendix D), then we can find an ϵ -stationary point of Φ δ with a complexity of O ( ϵ -4 ) .

Moreover, if we have initial points x 0 , y ′ 0 , z ′ 0 , u 0 , v 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then we can find an ϵ -stationary point of Φ δ with a complexity of O ( ϵ -3 ) .

The formal statement and the complete proof of Theorem 5.4 can be found in Appendix D.

Proof sketch for Theorem 5.4 . The general proof flow of Theorem 5.4 is similar to that of Theorem 5.2. However, since here we do not have coupled constraints, ∇ x K has no relationship to u or v , and according to the Danskin's theorem, ∇ ϕ δ ( x ) = δ ∇ x f ( x, y ∗ δ ( x )) + ∇ x g ( x, y ∗ δ ( x )) -∇ x g ( x, z ∗ ( x )) also has no relationship to u ∗ δ ( x ) or v ∗ ( x ) . Thus, we do not require Lemma E.2 or the full row-rank assumption on A in this setting.

Next, we show that, as long as the LICQ condition (Definition 4.3) holds for the initial y ∗ ( x 0 ) and y ∗ δ ( x 0 ) , we can find initial points satisfying (10)-(11) with a complexity of O ( ϵ -2 ) and we have the following corollary.

Corollary 5.5. When h ( x, y ) = Ay -b , Assumption 3.1, 3.2, 3.3 hold, and δ = Θ( ϵ ) ≤ µ g / (2 l f, 1 ) , for a given initial point x 0 , if the LICQ condition (Definition 4.3) holds at y ∗ ( x 0 ) and y ∗ δ ( x 0 ) , we can apply Algorithm 1 with fixed x 0 . Then for a sufficiently small ϵ (see Appendix D), we can find ˆ y ′ , ˆ z ′ , ˆ u, ˆ v such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with a complexity of O ( ϵ -2 ) . Set y ′ 0 = ˆ y ′ , z ′ 0 = ˆ z ′ , u 0 = ˆ u , v 0 = ˆ v . With x 0 , y ′ 0 , z ′ 0 , u 0 , v 0 as initial points, we can find an ϵ -stationary point of Φ δ with a complexity of O ( ϵ -3 ) . Thus, the total complexity is O ( ϵ -3 + ϵ -2 ) = O ( ϵ -3 ) .

The proof of Corollary 5.5 can be found in Appendix D. The key to proving Corollary 5.5 lies in Lemma D.4. In Lemma D.4, we prove that, without the full row-rank assumption on A , we can bound ∥ u t +1 -u ∗ δ ( x 0 ) ∥ and ∥ v t +1 -v ∗ ( x 0 ) ∥ with a fixed x 0 . Thus, we can use our algorithm SFLCB to find suitable initial points with a fixed x 0 .

Note that in Corollary 5.5, we only need the LICQ condition holds for the initial y ∗ ( x 0 ) and y ∗ δ ( x 0 ) , and we can achieve a total complexity of O ( ϵ -3 ) . Compared to the decoupled constrained setting in [21], we achieve an improvement in the convergence rate from O ( ϵ -3 log( ϵ -1 )) to O ( ϵ -3 ) (See Table 1).

## 6 Experiments

In this section, we evaluate the performance of our SFLCB algorithm on three tasks: a toy example, hyperparameter optimization for SVM, and a transportation network design problem. These experiments demonstrate the practical effectiveness and efficiency of SFLCB. Additional hype-parameter sensitivity analysis experiments and detailed experimental settings can be found in Appendix F.

## 6.1 Toy example

Weconsider the same constrained BLO problem that was studied in [16], which is:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Y ( x ) = { y ∈ R | y ≤ x } . Note that this problem satisfies Assumption 3.1, 3.2, 3.3. Moreover, we have y ∗ ( x ) = x and Equation (12) is equivalent to min x ∈ [0 , 3] f ( x, y ) | y = x . In Figure 1, we plot the hyper-objective function f ( x, y ∗ ( x )) . The red points indicate the converged solutions obtained by our al-

Figure 1: Toy example.

<!-- image -->

gorithm with 200 different initialization values. We notice that SFLCB consistently finds the local minima of the hyper-objective function, which validates the effectiveness of SFLCB.

## 6.2 Hyperparameter optimization in SVM

Hyperparameter optimization in SVM is a wellknown real-world application for constrained BLO problems that has been used in many prior works [46, 50, 49, 16]. Here we consider the same problem formulation as in [16], which formulates this problem as a coupled linearly constrained BLO problem. We conduct experiments comparing our SFLCB algorithm with GAM [46], LV-HBA [50], BLOCC [16], and BiC-GAFFA [49] on the diabetes dataset [6]. Results are plotted in Figure 2. We notice that our SFLCB algorithm converges significantly faster than other algorithms, which demonstrates the practical efficiency of the proposed SFLCB algorithm.

## 6.3 Transportation network design

We further conduct experiments on a transportation network design problem, following the same setting as in [16]. In this setting, we act as the operator, whose profit serves as the upper-level objective and is influenced by passenger behavior, which is modeled in the lower-level problem. Detailed formulations and settings can be found in Appendix F. We consider the two synthetic networks of 3 and 9 nodes, same as those considered in [16]. We compare SFLCB with BLOCC [16]. Results are plotted in Figure 3, which indicate that SFLCB significantly outperforms BLOCC on this network design task.

## 6.3.1 Sensitivity analysis of δ

We also conduct the sensitivity analysis of the δ in SFLCB for the 3-node network. We set ρ 1 = ρ 2 = 1000 , η x = η y = η z = η u = η v = 3 e -4 , and T = 20000 . Then, we test different δ values from { 0 . 01 , 0 . 05 , 0 . 1 , 0 . 5 , 1 } . For each δ , we test with three different random seeds. The final average results and one standard deviation are reported in Figure 4. As can be seen, larger values of

Figure 2: Hyperparameter optimization in SVM.

<!-- image -->

Figure 3: Results of the transportation experiments on 3 nodes and 9 nodes settings. Larger UL utility indicates better performance.

<!-- image -->

δ lead to a faster initial decrease in the loss (increase in UL utility). In contrast, very small δ (e.g., δ = 0 . 01 ) results in significantly slower convergence overall. However, an overly large δ (e.g., δ = 1 ) can lead to large approximation errors in later stages, causing deviation from the true optimization objective and ultimately poor performance. We observe that moderate values of δ (such as 0.05, 0.1, and 0.5) achieve relatively good final performance. These observations are consistent with our theoretical predictions. For example, our theory indicates that the convergence rate of SLFCB is inversely proportional to δ : smaller δ leads to slower convergence but smaller approximation error, whereas larger δ improves convergence speed towards the approximate problem but incurs greater approximation error. Figure 4 indicates a properly chosen δ thus can balance convergence speed and approximation error.

Figure 4: Comparison of different δ in SFLCB for the 3 nodes network.

<!-- image -->

## 7 Conclusions and future directions

In this paper, for coupled constrained BLO problem in Equation (1), we theoretically analyzed the relationship between the original hyper-objective Φ and the reformulated function Φ δ in Equation (3), providing a solid justification for the validity of the reformulation. Especially, for the linearly constrained case, we proposed SFLCB, a single-loop, Hessian-free algorithm, improving the convergence rate from O ( ϵ -3 log( ϵ -1 )) to O ( ϵ -3 ) over previous works [21, 16]. Our experiments on hyperparameter optimization for SVM and the transportation network design problem validated the practical efficiency of the proposed SFLCB algorithm. One limitation of our work is that the analysis is restricted to deterministic and linearly constrained settings. A promising direction for future research is to extend the current results to stochastic environments or more general constraint structures. Moreover, since the best-known complexity for first-order methods in unconstrained BLO [12] is O ( ϵ -2 ) , it is also an interesting problem whether we can achieve this optimal rate in the constrained cases.

## Acknowledgements

The work of Wei Shen and Cong Shen was supported in part by the US National Science Foundation (NSF) under awards 2143559 and 2332060. The work of Jiawei Zhang was supported by the Office of the Vice Chancellor for Research and Graduate Education at the University of Wisconsin-Madison with funding from the Wisconsin Alumni Research Foundation.

## References

- [1] Dimitri Bertsekas. Convex Optimization Theory , volume 1. Athena Scientific, 2009.
- [2] Jérôme Bolte, Edouard Pauwels, and Samuel Vaiter. Automatic differentiation of nonsmooth iterative algorithms. Advances in Neural Information Processing Systems , 35:26404-26417, 2022.
- [3] Congliang Chen, Jiawei Zhang, Li Shen, Peilin Zhao, and Zhiquan Luo. Communication efficient primal-dual algorithm for nonconvex nonsmooth distributed optimization. In International conference on artificial intelligence and statistics , pages 1594-1602. PMLR, 2021.
- [4] Tianyi Chen, Yuejiao Sun, Quan Xiao, and Wotao Yin. A single-timescale method for stochastic bilevel optimization. In International Conference on Artificial Intelligence and Statistics , pages 2466-2488. PMLR, 2022.
- [5] Mathieu Dagréou, Pierre Ablin, Samuel Vaiter, and Thomas Moreau. A framework for bilevel optimization that enables stochastic and global variance reduction algorithms. Advances in Neural Information Processing Systems , 35:26698-26710, 2022.
- [6] Dheeru Dua, Casey Graff, et al. UCI machine learning repository. 2017.
- [7] Luca Franceschi, Paolo Frasconi, Saverio Salzo, Riccardo Grazzi, and Massimiliano Pontil. Bilevel programming for hyperparameter optimization and meta-learning. In International Conference on Machine Learning , pages 1568-1577. PMLR, 2018.
- [8] Lucy L Gao, Jane Ye, Haian Yin, Shangzhi Zeng, and Jin Zhang. Value function based difference-of-convex algorithm for bilevel hyperparameter selection problems. In International Conference on Machine Learning , pages 7164-7182. PMLR, 2022.
- [9] Saeed Ghadimi and Mengdi Wang. Approximation methods for bilevel programming. arXiv preprint arXiv:1802.02246 , 2018.
- [10] Riccardo Grazzi, Luca Franceschi, Massimiliano Pontil, and Saverio Salzo. On the iteration complexity of hypergradient computation. In International Conference on Machine Learning , pages 3748-3758. PMLR, 2020.
- [11] Mingyi Hong, Hoi-To Wai, Zhaoran Wang, and Zhuoran Yang. A two-timescale stochastic algorithm framework for bilevel optimization: Complexity analysis and application to actorcritic. SIAM Journal on Optimization , 33(1):147-180, 2023.
- [12] Feihu Huang. Optimal Hessian/Jacobian-free nonconvex-PL bilevel optimization. arXiv preprint arXiv:2407.17823 , 2024.
- [13] Kaiyi Ji, Jason D Lee, Yingbin Liang, and H Vincent Poor. Convergence of meta-learning with task-specific adaptation over partial parameters. Advances in Neural Information Processing Systems , 33:11490-11500, 2020.
- [14] Kaiyi Ji, Mingrui Liu, Yingbin Liang, and Lei Ying. Will bilevel optimizers benefit from loops. Advances in Neural Information Processing Systems , 35:3011-3023, 2022.
- [15] Kaiyi Ji, Junjie Yang, and Yingbin Liang. Bilevel optimization: Convergence analysis and enhanced design. In International Conference on Machine Learning , pages 4882-4892. PMLR, 2021.

- [16] Liuyuan Jiang, Quan Xiao, Victor M Tenorio, Fernando Real-Rojas, Antonio G Marques, and Tianyi Chen. A primal-dual-assisted penalty approach to bilevel optimization with coupled constraints. arXiv preprint arXiv:2406.10148 , 2024.
- [17] Xiaotian Jiang, Jiaxiang Li, Mingyi Hong, and Shuzhong Zhang. Barrier function for bilevel optimization with coupled lower-level constraints: Formulation, approximation and algorithms. arXiv preprint arXiv:2410.10670 , 2024.
- [18] Prashant Khanduri, Ioannis Tsaknakis, Yihua Zhang, Jia Liu, Sijia Liu, Jiawei Zhang, and Mingyi Hong. Linearly constrained bilevel optimization: A smoothed implicit gradient approach. In International Conference on Machine Learning , pages 16291-16325. PMLR, 2023.
- [19] Prashant Khanduri, Siliang Zeng, Mingyi Hong, Hoi-To Wai, Zhaoran Wang, and Zhuoran Yang. A near-optimal algorithm for stochastic bilevel optimization via double-momentum. Advances in neural information processing systems , 34:30271-30283, 2021.
- [20] Guy Kornowski, Swati Padmanabhan, Kai Wang, Zhe Zhang, and Suvrit Sra. First-order methods for linearly constrained bilevel optimization. arXiv preprint arXiv:2406.12771 , 2024.
- [21] Jeongyeol Kwon, Dohyun Kwon, Stephen Wright, and Robert Nowak. On penalty methods for nonconvex bilevel optimization and first-order stochastic approximation. arXiv preprint arXiv:2309.01753 , 2023.
- [22] Jeongyeol Kwon, Dohyun Kwon, Stephen Wright, and Robert D Nowak. A fully first-order method for stochastic bilevel optimization. In International Conference on Machine Learning , pages 18083-18113. PMLR, 2023.
- [23] Hanwen Liang, Shifeng Zhang, Jiacheng Sun, Xingqiu He, Weiran Huang, Kechen Zhuang, and Zhenguo Li. Darts+: Improved differentiable architecture search with early stopping. arXiv preprint arXiv:1909.06035 , 2019.
- [24] Tianyi Lin, Chi Jin, and Michael I Jordan. Near-optimal algorithms for minimax optimization. In Conference on Learning Theory , pages 2738-2779. PMLR, 2020.
- [25] Bo Liu, Mao Ye, Stephen Wright, Peter Stone, and Qiang Liu. Bome! bilevel optimization made easy: A simple first-order approach. Advances in neural information processing systems , 35:17248-17262, 2022.
- [26] Hanxiao Liu, Karen Simonyan, and Yiming Yang. Darts: Differentiable architecture search. arXiv preprint arXiv:1806.09055 , 2018.
- [27] Risheng Liu, Jiaxin Gao, Jin Zhang, Deyu Meng, and Zhouchen Lin. Investigating bi-level optimization for learning and vision from a unified perspective: A survey and beyond. IEEE Transactions on Pattern Analysis and Machine Intelligence , 44(12):10045-10067, 2021.
- [28] Risheng Liu, Xuan Liu, Xiaoming Yuan, Shangzhi Zeng, and Jin Zhang. A value-function-based interior-point method for non-convex bi-level optimization. In International conference on machine learning , pages 6882-6892. PMLR, 2021.
- [29] Risheng Liu, Yaohua Liu, Shangzhi Zeng, and Jin Zhang. Towards gradient-based bilevel optimization with non-convex followers and beyond. Advances in Neural Information Processing Systems , 34:8662-8675, 2021.
- [30] Risheng Liu, Pan Mu, Xiaoming Yuan, Shangzhi Zeng, and Jin Zhang. A generic first-order algorithmic framework for bi-level programming beyond lower-level singleton. In International Conference on Machine Learning , pages 6305-6315. PMLR, 2020.
- [31] Songtao Lu. Slm: A smoothed first-order lagrangian method for structured constrained nonconvex optimization. Advances in Neural Information Processing Systems , 36, 2024.
- [32] Zhaosong Lu and Sanyou Mei. First-order penalty methods for bilevel optimization. SIAM Journal on Optimization , 34(2):1937-1969, 2024.

- [33] Matthew MacKay, Paul Vicol, Jon Lorraine, David Duvenaud, and Roger Grosse. Self-tuning networks: Bilevel optimization of hyperparameters using structured best-response functions. arXiv preprint arXiv:1903.03088 , 2019.
- [34] Dougal Maclaurin, David Duvenaud, and Ryan Adams. Gradient-based hyperparameter optimization through reversible learning. In International Conference on Machine Learning , pages 2113-2122. PMLR, 2015.
- [35] Akshay Mehra and Jihun Hamm. Penalty method for inversion-free deep bilevel optimization. In Asian conference on machine learning , pages 347-362. PMLR, 2021.
- [36] Fabian Pedregosa. Hyperparameter optimization with approximate gradient. In International Conference on Machine Learning , pages 737-746. PMLR, 2016.
- [37] Aravind Rajeswaran, Chelsea Finn, Sham M Kakade, and Sergey Levine. Meta-learning with implicit gradients. Advances in Neural Information Processing Systems , 32, 2019.
- [38] Amirreza Shaban, Ching-An Cheng, Nathan Hatch, and Byron Boots. Truncated backpropagation for bilevel optimization. In The 22nd International Conference on Artificial Intelligence and Statistics , pages 1723-1732. PMLR, 2019.
- [39] Han Shen and Tianyi Chen. On penalty-based bilevel gradient descent method. In International Conference on Machine Learning , pages 30992-31015. PMLR, 2023.
- [40] Ankur Sinha, Pekka Malo, and Kalyanmoy Deb. A review on bilevel optimization: From classical to evolutionary approaches and applications. IEEE Transactions on Evolutionary Computation , 22(2):276-295, 2017.
- [41] Richard S Sutton. Reinforcement learning: An introduction. A Bradford Book , 2018.
- [42] Ioannis Tsaknakis, Prashant Khanduri, and Mingyi Hong. An implicit gradient-type method for linearly constrained bilevel problems. In 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 5438-5442. IEEE, 2022.
- [43] Ioannis Tsaknakis, Prashant Khanduri, and Mingyi Hong. An implicit gradient method for constrained bilevel problems using barrier approximation. In 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 1-5. IEEE, 2023.
- [44] Gerd Wachsmuth. On LICQ and the uniqueness of lagrange multipliers. Operations Research Letters , 41(1):78-80, 2013.
- [45] Quan Xiao, Han Shen, Wotao Yin, and Tianyi Chen. Alternating projected sgd for equalityconstrained bilevel optimization. In International Conference on Artificial Intelligence and Statistics , pages 987-1023. PMLR, 2023.
- [46] Siyuan Xu and Minghui Zhu. Efficient gradient approximation method for constrained bilevel optimization. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 37, pages 12509-12517, 2023.
- [47] Haikuo Yang, Luo Luo, Chris Junchi Li, and Michael I Jordan. Accelerating inexact hypergradient descent for bilevel optimization. arXiv preprint arXiv:2307.00126 , 2023.
- [48] Shuoguang Yang, Xuezhou Zhang, and Mengdi Wang. Decentralized gossip-based stochastic bilevel optimization over communication networks. Advances in Neural Information Processing Systems , 35:238-252, 2022.
- [49] Wei Yao, Haian Yin, Shangzhi Zeng, and Jin Zhang. Overcoming lower-level constraints in bilevel optimization: A novel approach with regularized gap functions. arXiv preprint arXiv:2406.01992 , 2024.
- [50] Wei Yao, Chengming Yu, Shangzhi Zeng, and Jin Zhang. Constrained bi-level optimization: Proximal lagrangian value function approach and Hessian-free algorithm. arXiv preprint arXiv:2401.16164 , 2024.

- [51] Jiawei Zhang and Zhi-Quan Luo. A global dual error bound and its application to the analysis of linearly constrained nonconvex optimization. SIAM Journal on Optimization , 32(3):2319-2346, 2022.
- [52] Yihua Zhang, Prashant Khanduri, Ioannis Tsaknakis, Yuguang Yao, Mingyi Hong, and Sijia Liu. An introduction to bilevel optimization: Foundations and applications in signal processing and machine learning. IEEE Signal Processing Magazine , 41(1):38-59, 2024.
- [53] Yihua Zhang, Guanhua Zhang, Prashant Khanduri, Mingyi Hong, Shiyu Chang, and Sijia Liu. Revisiting and advancing fast adversarial training through the lens of bi-level optimization. In International Conference on Machine Learning , pages 26693-26712. PMLR, 2022.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We clearly state this paper's contributions and scope in the abstract and introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discussed the limitations of this work in Section 7.

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

Justification: We clearly state all the assumptions for each theorem. The proofs can be found in Appendix.

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

Justification: We provide detailed experimental settings in Appendix F.

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

## Answer: [Yes]

Justification: We provide the open access to our code and provide detailed experimental settings in Appendix F.

## Guidelines:

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

Justification: We provide detailed experimental settings in Appendix F.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We report the error bar in Figure 2. The definition of the error bar can be found in Appendix F.

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

Justification: We provide the information on the computer resources in Appendix F.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: To the best of our knowledge, the research conducted in this paper fully conforms with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This work is theoretical in nature and does not have immediate direct societal impact.

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

Justification: To the best of our knowledge, we do not think our paper poses such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We used diabetes dataset and cited the original paper.

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

Justification: Our paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects. Guidelines:

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

## Appendix

The Appendix is organized as follows. In Appendix A, we introduce some useful lemmas that will be utilized in the subsequent proofs. In Appendix B, we introduce some notations that will be used throughout the Appendix. In Appendix C, we provide the proofs for the lemmas and theorems in Section 4. In Appendix D, we provide proofs for Theorem 5.4 and Corollary 5.5. In Appendix E, we provide proofs for Theorem 5.2 and Corollary 5.3. In Appendix F, we present the detailed experimental settings along with additional hyperparameter sensitivity analysis experiments.

## A Useful Lemmas

Lemma A.1 (Lemma 12 in [3]) . Suppose f ( · ) is l -smooth and µ -strongly convex, X is a convex closed set, and η ≤ 1 /l . Define x ∗ = argmin x ∈X f ( x ) and x + = Π X ( x -η ∇ f ( x )) . Then, we have

<!-- formula-not-decoded -->

Lemma A.2 (Lemma 23 in [24]) . Suppose for any fixed y ∈ Y , f ( x, y ) is µ -strongly convex w.r.t. x and suppose for any y 1 , y 2 ∈ Y , x ∈ X , ∥∇ x f ( x, y 1 ) - ∇ x f ( x, y 2 ) ∥ ≤ l ∥ y 1 -y 2 ∥ . Define x ∗ ( y ) = argmin x ∈X f ( x, y ) . Then, we have

<!-- formula-not-decoded -->

Lemma A.3 (Theorem 4.1 in [51]) . Suppose f ( x ) : R d 1 → R is l -smooth and µ -strongly convex, P = { x | Cx ≤ e } , S ( r ) = { x ∈ P| Ax -b = r } , where C, A ∈ R d 2 × d 1 , e, b, r ∈ R d 2 . Define

<!-- formula-not-decoded -->

where u ∈ R d 2 is the Lagrange multiplier. We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

¯ M is the set of all submatrices of M with full row rank.

Lemma A.4. Suppose f ( x ) : R d x → R is µ -strongly convex w.r.t. x . Define L ( x, α, u ) = f ( x )+ u ⊤ ( Ax -b -α )+ ρ 2 ∥ Ax -b -α ∥ 2 , where α, u, b ∈ R d y , A ∈ R d y × d x , ρ ∈ (0 , µ/σ 2 max ( A )) . Denoting x ′ = ( x ⊤ , α ⊤ ) ⊤ , L ( x ′ , u ) = L ( x, α, u ) , we have L ( x ′ , u ) is µ x -strongly convex w.r.t. x ′ , where µ x = min { µ -ρσ 2 max ( A ) , ρ 2 } .

where

Proof. For any x 1 , x 2 ∈ R d x , α 1 , α 2 ∈ R d y , denoting x ′ 1 = ( x ⊤ 1 , α ⊤ 1 ) ⊤ , x ′ 2 = ( x ⊤ 2 , α ⊤ 2 ) ⊤ , A ′ = ( A, -I ) , we have

<!-- formula-not-decoded -->

where µ x = min { µ -ρσ 2 ( A ) , }

<!-- formula-not-decoded -->

## B Notations

Denote l δ = µ g / 2 . We introduce the following notations.

<!-- formula-not-decoded -->

When δ ≤ µ g / 2 l f, 1 , we have δl f, 1 ≤ µ g / 2 = l δ and thus, g δ ( x, y ) is L g -smooth and ϕ δ ( x, y, z ) is L ϕ -smooth.

## C Reformulation

In this section, we provide the proofs for the lemmas and theorems in Section 4.

## Theorem 4.1

When Assumption 3.1, 3.2 and 3.3 hold, we have

<!-- formula-not-decoded -->

Proof. Similar results and proofs of Theorem 4.1 can also be found in [21, 16]. For completeness, we also provide our proofs for Theorem 4.1 here.

Note that y ∗ δ ( x ) satisfy

<!-- formula-not-decoded -->

Since g ( x, · ) is µ g -strongly concave and l g, 1 -smooth and L ϕ ≥ l g, 1 , according to Lemma A.1, we have

<!-- formula-not-decoded -->

For Φ δ ( x ) , we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where the first equality is due to the quadratic growth of a strongly convex function, the last equality is due to ax 2 + bx ≥ -b 2 / (4 a ) .

## Lemma 4.4

When Assumption 3.1, 3.2, 3.3 hold and δ ≤ µ g / (2 l f, 1 ) , if, for all x ∈ X , LICQ condition (Definition 4.3) hold for y ∗ ( x ) and y ∗ δ ( x ) , then there exist the corresponding unique Lagrangian multipliers λ ∗ ( x ) ∈ R d h and λ ∗ δ ( x ) ∈ R d h such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, we have

<!-- formula-not-decoded -->

Proof. The uniqueness of Lagrangian multipliers is a direct result from the LICQ condition [44]. According to Lemmas 2 and 3 in [16]. We know that when Assumption 3.1, 3.2, 3.3 hold, δ ≤

µ g / (2 l f, 1 ) , if, for all x ∈ X , LICQ condition (Definition 4.3) holds for y ∗ ( x ) and y ∗ δ ( x ) , defining ψ ( x ) = g ( x, y ∗ ( x )) and ψ δ ( x ) = g δ ( x, y ∗ δ ( x )) , we have

<!-- formula-not-decoded -->

Since Φ δ ( x ) = f ( x, y ∗ δ ( x )) + 1 δ ( g δ ( x, y ∗ δ ( x )) -g ( x, y ∗ ( x ))) , we have

<!-- formula-not-decoded -->

## Lemma 4.7

When Assumption 3.1, 3.2, 3.3, 4.6 hold, if, for a given x , Assumption 4.5 and the LICQ condition (Definition 4.3) holds for y ∗ ( x ) , then ∇ Φ( x ) exists at x and can be expressed as

<!-- formula-not-decoded -->

where ∇ y ∗ ( x ) can be calculated according to (20) .

Proof. According to Theorem 2 in [46], we know that when Assumption 3.1, 3.2, 3.3, 4.6 hold, if, for a given x , Assumption 4.5 and the LICQ (Definition 4.3) condition holds for y ∗ ( x ) , then ∇ Φ( x ) exists at x .

Moreover, we can give the explicit expression of ∇ Φ( x ) .

With λ ∈ R d h , we have the following Lagrangian function

<!-- formula-not-decoded -->

Denote λ ∗ ( x ) as the optimal Lagrangian multiplier, I x ⊆ [ d h ] as the active set of y ∗ ( x ) , i.e. I x = { i ∈ [ d h ][ h ( x, y )] i = 0 } Denote ¯ h ( x, y ∗ ( x )) = [ h ( x, y ∗ ( x ))] I x and ¯ λ ∗ ( x ) = [ λ ∗ ( x )] I x . We have the following KKT conditions:

<!-- formula-not-decoded -->

Differentiating the KKT conditions with respect to x , we have

<!-- formula-not-decoded -->

Thus, ∇ y ∗ ( x ) and λ ∗ ( x ) satisfy the following equation.

<!-- formula-not-decoded -->

Denote

<!-- formula-not-decoded -->

Since g is strongly convex, h is convex, ∇ 2 yy g ( x, y ∗ ( x )) ≻ 0 , ∇ 2 yy [ ¯ h ( x, y ∗ ( x ))] i ⪰ 0 . Moreover, since ¯ λ ∗ ( x ) &gt; 0 , we have ∇ 2 yy g ( x, y ∗ ( x )) + ∇ 2 yy ¯ h ( x, y ∗ ( x )) ⊤ ¯ λ ∗ ( x ) ≻ 0 . Additionally,

∇ y ¯ h ( x, y ∗ ( x )) has full row rank. Thus, according to Lemma A.2 in [20], H is invertible. We can calculate ∇ y ∗ ( x ) and λ ∗ ( x ) with the following equation.

<!-- formula-not-decoded -->

According to [9], we have

<!-- formula-not-decoded -->

## Theorem 4.9

When Assumption 3.1, 3.2, 3.3, 4.6 hold, and δ ≤ µ g / (2 l f, 1 ) , if Assumption 4.8 holds for a given x , we have

<!-- formula-not-decoded -->

Proof. With λ ∈ R d h , we have the following Lagrangian function

<!-- formula-not-decoded -->

Denote λ ∗ δ ( x ) as the optimal Lagrangian multiplier. We have the following KKT conditions.

<!-- formula-not-decoded -->

Differentiating the KKT conditions with respect to δ , we have

<!-- formula-not-decoded -->

We have the following equation:

<!-- formula-not-decoded -->

Denote

<!-- formula-not-decoded -->

We can notice that H δ =0 = H , where H is defined in (19).

Then, we have

<!-- formula-not-decoded -->

where we denote

<!-- formula-not-decoded -->

Note that according to (20), we have

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

Note that, according to Lemma 4.4, we have

<!-- formula-not-decoded -->

Then, we consider to bound ∇ Φ δ ( x ) -∇ Φ( x ) .

<!-- formula-not-decoded -->

where the last equality is due to (23).

For the first term in (24), we have

<!-- formula-not-decoded -->

For the remaindering terms in (24), we have

<!-- formula-not-decoded -->

For the first term in (25), we have

<!-- formula-not-decoded -->

where the last equality is due to Lemma C.1. Similarly, for the third, fifth terms in (25), we have

<!-- formula-not-decoded -->

For the second term in (25), we have

<!-- formula-not-decoded -->

where the last equality is due to Lemma C.1.

Similarly, for the fourth, sixth terms in (25), we have

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

Lemma C.1. When Assumption 3.1, 3.2, 3.3, 4.6 hold, and δ ≤ µ g / (2 l f, 1 ) , if Assumption 4.8 holds for a given x , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where C y , L y are O (1) constants.

Proof. According to (21), we have where

<!-- formula-not-decoded -->

According to Lemma C.2, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Denote

<!-- formula-not-decoded -->

Thus

Denote

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, we have

Moreover, we have

<!-- formula-not-decoded -->

where L y = ( C 2 H l f, 0 M 2 H + C H l f, 1 ) .

Lemma C.2. For H δ defined in (22) , we have

<!-- formula-not-decoded -->

where C H is an O (1) constant depending on µ g , l g, 1 , l f, 1 , d h , l h, 1 , s min , s max , Λ .

Proof. Denote A = ∇ 2 yy g δ ( x, y ∗ δ ( x )) + ∇ 2 yy ¯ h ( x, y ∗ δ ( x )) ⊤ ¯ λ ∗ δ ( x ) , C = ∇ y ¯ h ( x, y ∗ δ ( x )) . We have

<!-- formula-not-decoded -->

According to Assumption 4.8, we know that ∥ λ ∗ δ ( x ) ∥ ≤ Λ . Thus, 0 ≤ [ λ ∗ δ ( x )] i ≤ Λ . We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have ∥ B ∥ ≤ C B , ∥ D ∥ ≤ C D , ∥ E ∥ ≤ C E and so that ∥ H -1 δ 1 ∥ ≤ C H , where C B , C D , C E , C H are O (1) constants depending on µ g , l g, 1 , l f, 1 , d h , l h, 1 , s min , s max , Λ .

<!-- formula-not-decoded -->

## D Proofs of Theorem 5.4 and Corollary 5.5

In this section, we provide proofs for Theorem 5.4 and Corollary 5.5. We first introduce the additional notations and lemmas that will be used in this section.

## Notations

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma D.1. When δ ≤ µ g / (2 l f, 1 ) , 0 ≤ ρ 1 ≤ µ g -δl f, 1 σ 2 max ( A ) and 0 ≤ ρ 2 ≤ µ g σ 2 max ( A ) , K ( x, y ′ , z ′ , u, v ) is µ y -strongly convex w.r.t. y ′ , µ z -strongly concave w.r.t. z ′ , and L K -smooth w.r.t. x, y ′ , z ′ .

Proof. According to Lemma A.4, we know that K ( x, y ′ , z ′ , u, v ) is µ y -strongly convex w.r.t. y ′ , µ z -strongly concave w.r.t. z ′ . Moreover

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, K ( x, y ′ , z ′ , u, v ) is L K -smooth w.r.t. x, y ′ , z ′ .

<!-- formula-not-decoded -->

Lemma D.2. When δ ≤ µ g / (2 l f, 1 ) , 0 ≤ ρ 1 ≤ µ g -δl f, 1 σ 2 max ( A ) and 0 ≤ ρ 2 ≤ µ g σ 2 max ( A ) , η y , η z ≤ 1 /L K , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where σ yu = σ max ( A ′ ) µ y , σ yx = L K µ y , σ zv = σ max ( A ′ ) µ z , σ zx = L K µ z , σ ye = 2 µ y , σ ze = 2 µ z , σ α = 2 µ y η y , σ β = 2 µ z η z , σ ys = 2 L K µ g , σ zs = L K µ g ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

¯ M is the set of all submatrices of M with full row rank.

Proof. (26), (27), (28), (29), (30), (31) is due to Lemma A.2. (34), (35), (36), (37) is due to Lemma A.1. (32), (33) is due to Lemma A.3.

## D.1 Potential function

In this subsection, we will prove the following descent lemma for V t .

Lemma D.3. When δ ≤ µ g / (2 l f, 1 ) , 0 ≤ ρ 1 ≤ µ g -δl f, 1 σ 2 max ( A ) , 0 ≤ ρ 2 ≤ µ g σ 2 max ( A ) , η y = 1 / (4 L K ) , η z = 2 / ( L K + 4 L d ) , η x = min { η y µ 2 y / (512 L 2 ϕ ) , η z µ 2 z / (96 L 2 ϕ ) , η u / (64 σ 2 y L 2 ϕ ) , η v / (4 σ 2 z L 2 ϕ ) , 2 / ( L K + 4 L d +8 L q ) } , η u = η y µ 2 y / (32 σ 2 max ( A )) , η v = η z µ 2 z / (32 σ 2 max ( A )) , we have

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

Proof. First, for function d , we have

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

Thus, according to Lemma D.2, we know that ∇ x d ( x, z ′ , u, v ) is ( L ϕ + L ϕ σ yx ) -continuous w.r.t. x, z ′ and ∇ ′ z d ( x, z ′ , u, v ) is L K -continuous w.r.t. x, z ′ . Define L d = max { L ϕ + L ϕ σ yx , L K } . We have

<!-- formula-not-decoded -->

Then, for function q , we have

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

Thus, according to Lemma D.2, q ( · , v ) is L q = ( L ϕ + L ϕ σ zx + L ϕ σ ys ) -smooth. We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, for function K , we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Thus, for V t , we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where the last equality is due to Lemma D.2.

Thus, when δ ≤ µ g / (2 l f, 1 ) , 0 ≤ ρ 1 ≤ µ g -δl f, 1 σ 2 max ( A ) , 0 ≤ ρ 2 ≤ µ g σ 2 max ( A ) , η y = 1 / (4 L K ) , η z = 2 / ( L K + 4 L d ) , η x = min { η y µ 2 y / (512 L 2 ϕ ) , η z µ 2 z / (96 L 2 ϕ ) , η u / (64 σ 2 y L 2 ϕ ) , η v / (4 σ 2 z L 2 ϕ ) , 2 / ( L K +4 L d +8 L q ) } , η u = η y µ 2 y / (32 σ 2 max ( A )) , η v = η z µ 2 z / (32 σ 2 max ( A )) , we have

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

## Proof of Theorem 5.4

When δ ≤ µ g / (2 l f, 1 ) , 0 ≤ ρ 1 ≤ µ g -δl f, 1 σ 2 max ( A ) , 0 ≤ ρ 2 ≤ µ g σ 2 max ( A ) , η y = 1 / (4 L K ) , η z = 2 / ( L K +4 L d ) , η x = min { η y µ 2 y / (512 L 2 ϕ ) , η z µ 2 z / (96 L 2 ϕ ) , η u / (64 σ 2 y L 2 ϕ ) , η v / (4 σ 2 z L 2 ϕ ) , 2 / ( L K +4 L d +8 L q ) } , η u = η y µ 2 y / (32 σ 2 max ( A )) , η v = η z µ 2 z / (32 σ 2 max ( A )) , according to (40), we have

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

Therefore, when δ = Θ( ϵ ) , with T = O ( ϵ -4 ) , we have t ∈ [ T ] , such that ∥∇ Φ δ ( x t ) ∥ = ∥ 1 δ ∇ ϕ δ ( x t ) ∥ ≤ ϵ .

Moreover, if we have x 0 , y 0 , z 0 , u 0 , v 0 such that

<!-- formula-not-decoded -->

Then, we have

<!-- formula-not-decoded -->

where G = max { g ( x 0 , y 0 ) , g ( x 0 , z 0 ) , g ( x 0 , y ∗ δ ( x 0 ) , g ( x 0 , z ∗ ( x 0 , v 0 )) , g ( x 0 , y ∗ δ ( x 0 , u 0 ) } , C = { y ∈ R d y | g ( x 0 , y ) ≤ G } , C g = sup y ∈C ∇ y g ( x 0 , y ) . Since g ( x 0 , y ) is strongly convex w.r.t y , its sub-level

set C is compact and convex. Moreover, since g is Lipschitz smoothness, its gradient in this compact set C is upper bounded by an O (1) constant C g .

We can notice that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Therefore, we can find an ϵ -stationary point of Φ δ ( x ) with a complexity of O ( ϵ -3 ) .

## Proof of Corollary 5.5

For fixed x 0 , define W t = 1 4 K ( x 0 , y ′ t , z ′ t , u t , v t ) + 2 q ( x 0 , v t ) -d ( x 0 , z ′ t , u t , v t ) . According to (39), with appropriate parameters, we have

<!-- formula-not-decoded -->

Thus, when T = O ( ϵ 2 ) , we can find t ∈ T such that

<!-- formula-not-decoded -->

Denote the active set at y ∗ δ ( x 0 ) as I α , J α = [ d h ] / I . Define ∆ α = min i ∈J α | [ α ∗ δ ( x 0 )] i | . Denote the active set of z ∗ ( x 0 ) as I β , J β = [ d h ] / I . Define ∆ β = min i ∈J β | [ β ∗ ( x 0 )] i | . Set ϵ ≤ min { ∆ α / (6 σ α ) , ∆ α / (6 σ y ) , ∆ β / (6 σ β ) , ∆ β / (6 σ z ) } . According to Lemma D.4, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, if we set y ′ 0 = y ′ t , z ′ 0 = z ′ t , u 0 = u t +1 , v 0 = v t +1 , with x 0 , y ′ 0 , z ′ 0 , u 0 , v 0 as initial points, according to Theorem 5.4, we can find an ϵ -stationary point of Φ δ with a complexity of O ( ϵ -3 ) .

Thus, the total complexity is O ( ϵ -3 + ϵ -2 ) = O ( ϵ -3 ) .

Lemma D.4. For a fixed x 0 , denote the active set at y ∗ δ ( x 0 ) as I α , J α = [ d h ] / I . We have [ α ∗ δ ( x 0 )] I α = 0 and [ α ∗ δ ( x 0 )] J α &lt; 0 . Suppose s α min = σ min ( A I α ) &gt; 0 . Define ∆ α = min i ∈J α | [ α ∗ δ ( x 0 )] i | . When ∥ y ′ t +1 -y ′ t ∥ ≤ ∆ α / (6 σ α ) , ∥ A ′ y ′∗ δ ( x 0 , u t +1 ) -b ∥ ≤ ∆ α / (6 σ y ) , we have

<!-- formula-not-decoded -->

Similarly, for a fixed x 0 , denote the active set of z ∗ ( x 0 ) as I β , J β = [ d h ] / I . Suppose s β min = σ min ( A I β ) &gt; 0 . Define ∆ β = min i ∈J β | [ β ∗ ( x 0 )] i | . When ∥ z ′ t +1 -z ′ t ∥ ≤ ∆ β / (6 σ β ) , ∥ A ′ z ′∗ δ ( x 0 , v t +1 ) -b ∥ ≤ ∆ β / (6 σ z ) , we have

<!-- formula-not-decoded -->

Proof. Denote the active set of y ∗ δ ( x 0 ) as I , J = [ d h ] / I . We have [ α ∗ δ ( x 0 )] I = 0 and [ α ∗ δ ( x 0 )] J &lt; 0 . Note that

<!-- formula-not-decoded -->

Define ∆ = min i ∈J | [ α ∗ δ ( x 0 )] i | . When ∥ y ′ t +1 -y ′ t ∥ ≤ ∆ / (6 σ α ) , ∥ A ′ y ′∗ δ ( x 0 , u t +1 ) -b ∥ ≤ ∆ / (6 σ y ) , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, there are no projection in the update of α t +1 and we have

<!-- formula-not-decoded -->

Moreover, for I , we have

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

Suppose σ min ( A I ) = s min , we have

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similar conditions and conclusions also hold for ∥ v t +1 -v ∗ ( x 0 ) ∥ .

## E Proofs of Theorem 5.2 and Corollary 5.3

In this section, we provide proofs for Theorem 5.2 and Corollary 5.3. We first introduce the additional notations and lemmas that will be used in this section.

Thus, for J , we have

## Notations

<!-- formula-not-decoded -->

## Lemma 5.1

When the LICQ condition (Definition 4.3) holds for y ∗ ( x ) and y ∗ δ ( x ) , the optimal Lagrangian multipliers of y ∗ ( x ) and y ∗ δ ( x ) are unique and we have

<!-- formula-not-decoded -->

Proof. Suppose

The KKT conditions for v 1 are

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The KKT conditions for v 2 are

<!-- formula-not-decoded -->

Note that these two KKT conditions are equivalent. Moreover, since the LICQ condition (Definition 4.3) holds for z ∗ ( x ) , we have v 1 = v 2 . Similar conditions and conclusions also hold for u ∗ δ ( x ) .

Lemma E.1. K ( x, y ′ , z ′ , u, v ) is µ y -strongly w.r.t. y ′ , µ z -strongly concave w.r.t. z ′ , and L K -smooth w.r.t. x, y ′ , z ′ .

Proof. According to Lemma A.4, we know that K ( x, y ′ , z ′ , u, v ) is µ y -strongly convex w.r.t. y ′ , µ z -strongly concave w.r.t. z ′ . Moreover

<!-- formula-not-decoded -->

Thus, K ( x, y ′ , z ′ , u, v ) is L K -smooth w.r.t. x, y ′ , z ′ .

## Lemma E.2.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. By the optimality condition at y ∗ δ ( x t ) , we have

<!-- formula-not-decoded -->

The update rule of y t :

<!-- formula-not-decoded -->

Putting together, we have

<!-- formula-not-decoded -->

Since A has full row rank, we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Moreover,

<!-- formula-not-decoded -->

where the last equality is due to Lemma D.2.

Thus,

<!-- formula-not-decoded -->

Similarly, we have

<!-- formula-not-decoded -->

## Lemma E.3.

Thus,

<!-- formula-not-decoded -->

where σ zb = σ max ( A ) σ z + σ zx , σ yb = σ max ( A ) σ y + σ yx , σ vb = l g, 1 σ zb σ min ( A ) , σ ub = L g σ yb σ min ( A ) .

Proof. Here, we introduce an additional notation: z ′∗ ( x ; w ) = argmin z ′ ∈G ( w ) g ( x, z ) + ρ 2 2 ∥ Bx + A ′ z ′ -b ∥ 2 , where G ( w ) = { z ′ ∈ P y | Bw + A ′ z ′ -b = 0 } . We can notice that z ′∗ ( x ; x ) = z ′∗ ( x ) . According to Lemma A.3, we have ∥ z ′∗ ( x 1 ; x 1 ) -z ′∗ ( x 1 ; x 2 ) ∥ ≤ σ z ∥ Ax 1 -Ax 2 ∥ . Moreover, we have ∥ z ′∗ ( x 1 ; x 2 ) -z ′∗ ( x 2 ; x 2 ) ∥ ≤ σ zx ∥ x 1 -x 2 ∥ . Thus, ∥ z ′∗ ( x 1 ) -z ′∗ ( x 2 ) ∥ ≤ [ σ max ( A ) σ z + σ zx ] ∥ x 1 -x 2 ∥ . Moreover,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## E.1 Potential function

In this subsection, we will prove the following descent lemma for V t .

Lemma E.4. When δ ≤ µ g / (2 l f, 1 ) , 0 ≤ ρ 1 ≤ µ g -δl f, 1 σ 2 max ( A ) , 0 ≤ ρ 2 ≤ µ g σ 2 max ( A ) , η y = 1 / (4 L K ) , η z = 2 / ( L K + 4 L d ) , η x = min { η y µ 2 y / (640 L 2 K ) , η z µ 2 z / (640 L 2 K ) , η u / (240( σ 2 y + σ 2 u 2 + σ 2 u 1 ) L 2 K ) , η v / (240( σ 2 z + σ 2 v 2 + σ 2 v 1 ) L 2 K ) , 2 / ( L K + 4 L d + 8 L q ) , 1 / (1920 η y L 2 K σ 2 uy ) , 1 / (1920 η z L 2 K σ 2 uz ) } , η u = η y µ 2 y / (256 σ 2 max ( A )) , η v = η z µ 2 z / (256 σ 2 max ( A )) , we have

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

Proof. First, for function d , we have

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

Thus, according to Lemma D.2, we know that ∇ x d ( x, z ′ , u, v ) is ( L K + L K σ yx ) -continuous w.r.t. x, z ′ and ∇ ′ z d ( x, z ′ , u, v ) is L K -continuous w.r.t. x, z ′ . Define L d = max { L K + L K σ yx , L K } . We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, for function q , we have

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

Thus, according to Lemma D.2 and E.3, q ( · , v ) is L q = ( L K + L K σ zx + L K σ yb + σ max ( B ) σ ub ) -smooth. We have

<!-- formula-not-decoded -->

Finally, for function K , we have

K ( x t , y ′ t , z ′ t , u t , v t ) -K ( x t , y ′ t , z ′ t , u t +1 , v t +1 ) = -η u ∥ Bx t + A ′ y ′ t -b ∥ 2 + η v ∥ Bx t + A ′ z ′ t -b ∥ 2 , and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, for V t , we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where the last equality is due to Lemma D.2, E.2 and ∇ x K ( x t , y ′∗ δ ( x t ) , z ′∗ ( x t ) , u ∗ ( x t ) , v ∗ ( x t )) = ∇ ϕ δ ( x ) .

## Proof of Theorem 5.2

When δ ≤ µ g / (2 l f, 1 ) , 0 ≤ ρ 1 ≤ µ g -δl f, 1 σ 2 max ( A ) , 0 ≤ ρ 2 ≤ µ g σ 2 max ( A ) , η y = 1 / (4 L K ) , η z = 2 / ( L K + 4 L d ) , η x = min { η y µ 2 y / (640 L 2 K ) , η z µ 2 z / (640 L 2 K ) , η u / (240( σ 2 y + σ 2 u 2 + σ 2 u 1 ) L 2 K ) , η v / (240( σ 2 z + σ 2 v 2 + σ 2 v 1 ) L 2 K ) , 2 / ( L K + 4 L d + 8 L q ) , 1 / (1920 η y L 2 K σ 2 uy ) , 1 / (1920 η z L 2 K σ 2 uz ) } , η u = η y µ 2 y / (256 σ 2 max ( A )) , η v = η z µ 2 z / (256 σ 2 max ( A )) , according to (43), we have

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

Therefore, when δ = Θ( ϵ ) , with T = O ( ϵ -4 ) , we have t ∈ [ T ] , such that ∥∇ Φ δ ( x t ) ∥ = ∥ 1 δ ∇ ϕ δ ( x t ) ∥ ≤ ϵ .

Moreover, if we have x 0 , y 0 , z 0 , u 0 , v 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, set α = h ( x , y ) , β = h ( x , z ) , we can easily prove that

0 0 0 0 0 0 ∥ Bx 0 + A ′ y ′∗ δ ( x 0 , u 0 ) -b ∥ ≤ ∥ Bx 0 + A ′ y ′∗ δ ( x 0 , u ∗ δ ( x 0 )) -b ∥ + σ max ( A ′ ) σ yu ∥ u 0 -u ∗ δ ( x 0 ) ∥ ≤ O ( δ ) ∥ Bx 0 + A ′ y ′ 0 -b ∥ ≤ σ max ( A ′ ) ∥ y ′ 0 -y ′∗ ( x 0 ) ∥ ≤ O ( δ ) and similarly,

<!-- formula-not-decoded -->

Then, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where G = max { g ( x 0 , y 0 ) , g ( x 0 , z 0 ) , g ( x 0 , y ∗ δ ( x 0 ) , g ( x 0 , z ∗ ( x 0 , v 0 )) , g ( x 0 , y ∗ δ ( x 0 , u 0 ) } , C = { y ∈ R d y | g ( x 0 , y ) ≤ G } , C g = sup y ∈C ∇ y g ( x 0 , y ) . Since g ( x 0 , y ) is strongly convex w.r.t y , its sub-level set C is compact and convex. Moreover, since g is Lipschitz smooth, its gradient in this compact set C is upper bounded by an O (1) constant C g .

We can notice that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Therefore, we can find an ϵ -stationary point of Φ δ ( x ) with a complexity of O ( ϵ -3 ) .

## Proof of Corollary 5.3

When h ( x, y ) = Bx + Ay -b , A is full row rank and under Assumption 3.1, 3.2, 3.3 and δ = O ( ϵ ) ≤ µ g / (2 l f, 1 ) if we apply PGD for max v ∈ R + min z ∈ R dy g ( x 0 , y ) + v ⊤ ( Bx 0 + Az -b ) with a fixed x 0 , then according to Theorem 6 in [16], we can find ˆ v , ˆ z such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with a complexity of O (log( ϵ -1 )) . Set y 0 = z 0 = ˆ z , u 0 = v 0 = ˆ v , α 0 = h ( x 0 , y 0 ) , β 0 = h ( x 0 , z 0 ) . We have ∥ y 0 -y ∗ δ ( x 0 ) ∥ ≤ O ( δ ) ∥ Bx 0 + A ′ y ′∗ δ ( x 0 , u 0 ) -b ∥ ≤ O ( δ ) ∥ Bx 0 + A ′ y ′ 0 -b ∥ ≤ O ( δ ) ∥ u 0 -u ∗ δ ( x 0 ) ∥ ≤ O ( δ ) ∥ Bx 0 + A ′ z ′∗ ( x 0 , v 0 ) -b ∥ ≤ O ( δ ) ∥ Bx 0 + A ′ z ′ 0 -b ∥ ≤ O ( δ )

with x 0 , y ′ 0 , z ′ 0 , u 0 , v 0 as initial points and apply Algorithm 1, according to Theorem 5.2, we can find an ϵ -stationary point of Φ δ with a complexity of O ( ϵ -3 ) .

Thus, the total complexity is O ( ϵ -3 +log( ϵ -1 )) = O ( ϵ -3 ) .

## F Detailed Experimental Settings and Additional Experiments

We adapt and modify the code from [16]. The experiments on the toy example and hyperparameter optimization for SVM are conducted on an AMD EPYC 9554 64-Core Processor. The experiments on transportation network design are conducted on an Intel(R) Xeon(R) Platinum 8375C CPU. For the toy example, we set the hyperparameters for our algorithm as δ = 0 . 1 , η x = η y = η z = η u = η v = 0 . 01 , ρ 1 = ρ 2 = 1 .

## F.1 Hyperparameter optimization in SVM

Support Vector Machines (SVMs) construct a machine learning model by identifying the best possible hyperplane that maximizes the separation margin between data points from different classes. In a hard-margin SVM, no misclassification is allowed, ensuring that all samples are correctly classified. Conversely, soft-margin SVMs allow certain samples to be misclassified to accommodate cases where a perfect separation is not feasible. To achieve this, slack variables ξ are introduced to quantify classification violations for each sample.

We consider the same problem formulation as in [16], which is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where D tr ≜ { ( z tr ,i , l tr ,i ) } |D tr | i =1 is the training dataset, D val ≜ { ( z val ,i , l val ,i ) } |D val | i =1 is the validation dataset, with z tr ,i ( z val ,i ) being the features of sample i and l tr ,i ( l val ,i ) being its corresponding labels. The hyperparameter c i is introduced to bound the soft margin violation ξ i . Note that the LL problem is equivalent to

<!-- formula-not-decoded -->

Then upper level variables are c , and the lower level variables are w,b . Thus, we have a coupled linear constraint for the lower level problem.

We compared our algorithm SFLCB with GAM [46], LV-HBA [50], BLOCC [16], and BiC-GAFFA [49] on the diabetes dataset. For GAM, we follow the same implementation approach and hyperparameters as [16], setting α = 0 . 05 , ϵ = 0 . 005 and using a different formulation as introduced

in [46]. For BLOCC and SFLCB, we set the LL objective as 1 2 ∥ w ∥ 2 + µ 2 b 2 , where µ = 0 . 01 serves as a regularization term to make the LL problem strongly convex w.r.t w,b . For LV-HBA, we set α = 0 . 01 , γ 1 = 0 . 1 , γ 2 = 0 . 1 , η = 0 . 001 . For BLOCC, we set γ = 12 , η = 0 . 01 , T = 20 , T y = 20 , η 1 g = 0 . 001 , η 1 F = 0 . 00001 , η 2 g = 0 . 0001 , η 2 F = 0 . 0001 . For BiC-GAFFA, we set p = 0 . 3 , γ 1 = 10 , γ 2 = 0 . 01 , η = 0 . 01 , α = 0 . 001 , β = 0 . 001 , c 0 = 10 , R = 10 . For our algorithm SFLCB, we set δ = 0 . 01 , η x = η y = η z = η u = η v = 0 . 001 , ρ 1 = ρ 2 = 0 . 01 . The hyperparameters of LV-HBA, BLOCC are the same as those used in [16] and the hyperparameters of BiC-GAFFA are the same as those used in their original paper [49]. The experiments are conducted across 10 different random train-validation-test splits, and the average results along with one standard deviation are reported in Figure 2.

## F.2 Transportation network design

We consider the same setting as in [16]. In this setting, we act as the operator to design a new network that connects a set of stations S . Passengers then decide whether to use this network based on their rational decisions (lower level). The objective is to maximize the operator's benefit (upper level). The operator can select a set of potential links A ⊆ S × S , and for each link ( i, j ) ∈ A , determine its capacity x ij . A link is constructed if x ij &gt; 0 ; a larger x ij attracts more travelers, generating more revenue but incurring higher construction costs c ij . Passenger demand is defined over a set of origin-destination pairs K ⊆ S × S . For each ( o, d ) ∈ K , there is a known traffic demand w od and existing travel times t ext od . We assume a single existing network. The fraction of passengers choosing the new network for each ( o, d ) pair is denoted by y od , and y ij od represents the proportion using link ( i, j ) .

We summarize the notation as follows. We keep the notation the same as in [16].

- x ij ∈ R + , the capacity of the new network for the link ( i, j ) ∈ A .
- y od ∈ [0 , 1] , the proportion of passengers from ( o, d ) ∈ K choosing the new network for their travel.
- y od ij ∈ [0 , 1] , the proportion of passengers from ( o, d ) ∈ K choosing the new network and use the link ( i, j ) ∈ A
- x = { x ij } ∀ ( i,j ) ∈A are the upper-level variables to be optimized.
- X = R |A| + represents the domain of x .
- y = { y od , { y od ij } ∀ ( i,j ) ∈A } ∀ ( o,d ) ∈K are the lower-level variables to be optimized
- Y = [ ε, 1 -ε ] |K| × [ ε, 1 -ε ] |A||K| , where ε is a small positive number, represents the domain of y .
- w od , the total estimated demand for ( o, d ) ∈ K .
- m od , the revenue obtained by the operator from a passenger in ( o, d ) ∈ K .
- c ij , the construction cost per passenger for link ( i, j ) ∈ A .
- t ij , the travel time for link ( i, j ) ∈ A .
- t od ext , travel time on the existing network for passengers in ( o, d ) ∈ K .
- ω t &lt; 0 , the coefficient associated with the travel time for passengers.

With these notions, we can introduce the bilevel formulations of this problem. At the upper level, the network operator seeks to maximize the overall profit by attracting more passengers while minimizing construction costs; thus, its objective is

<!-- formula-not-decoded -->

where y od ∗ ( x ) are optimal lower-level passenger flows. For the lower-level, the objective is defined as finding the flow variables that maximize passenger utility and minimize flow entropy cost.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (47) are the flow-conservation constraints and (48) are the capacity constraints.

We consider the same 3 nodes and 9 nodes settings as in [16] and use the same lower level feasibility criteria. The hyperparameter settings used in Figure 3 are listed below. For the 3 nodes setting, we set the hyperparameters of our method SFLCB as: δ = 0 . 1 , ρ 1 = ρ 2 = 1000 , η x = η y = η z = η u = η v = 3 e -4 , and T = 30000 . For the 9 nodes setting, we set the hyperparameters of our method SFLCB as: δ = 0 . 25 , ρ 1 = ρ 2 = 50 , η x = η y = η z = η u = η v = 3 e -5 , and T = 300000 . For BLOCC, we used the same hyperparameters as those used in [16].