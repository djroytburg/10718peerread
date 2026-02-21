## Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates

## Andrew Lowy ∗

CISPA Helmholtz Center for Information Security lowy.andrew1@gmail.com

## Abstract

Bilevel optimization, in which one optimization problem is nested inside another, underlies many machine learning applications with a hierarchical structure-such as meta-learning and hyperparameter optimization. Such applications often involve sensitive training data, raising pressing concerns about individual privacy. Motivated by this, we study differentially private bilevel optimization. We first focus on settings where the outer-level objective is convex , and provide novel upper and lower bounds on the excess empirical risk for both pure and approximate differential privacy. These bounds are nearly tight and essentially match the optimal rates for standard single-level differentially private ERM, up to additional terms that capture the intrinsic complexity of the nested bilevel structure. We also provide population loss bounds for bilevel stochastic optimization. The bounds are achieved in polynomial time via efficient implementations of the exponential and regularized exponential mechanisms. A key technical contribution is a new method and analysis of log-concave sampling under inexact function evaluations, which may be of independent interest. In the non-convex setting, we develop novel algorithms with state-of-the-art rates for privately finding approximate stationary points. Notably, our bounds do not depend on the dimension of the inner problem.

## 1 Introduction

Bilevel optimization has emerged as a key tool for solving hierarchical learning and decision-making problems across machine learning and beyond. In a bilevel optimization problem, one task (the upper-level problem) is constrained by the solution to another optimization problem (the lower-level problem). This nested structure arises naturally in a variety of settings, including meta-learning [43], hyperparameter optimization and model selection [21, 29], reinforcement learning [27], adversarial training [49], and game theory [47], where the solution to one problem depends implicitly on the outcome of another. Formally, a bilevel problem can be written as:

<!-- formula-not-decoded -->

where x and y are the upper- and lower-level variables respectively, F is the upper-level objective, G is the lower-level objective, X ⊂ R d x is a domain. Solving (1) is challenging due to the dependency of y ∗ ( x ) on x . The study of algorithms and complexities for solving (1) has received a lot of attention from the optimization and ML communities in recent years [23, 30, 10, 16, 20, 39, 31, 33, 34, 14].

∗ Authors listed in reverse alphabetical order. Part of this work was completed while the first author was at University of Wisconsin-Madison.

## Daogao Liu

Google Research liudaogao@gmail.com

In many applications where bilevel optimization can be useful, data privacy is of critical importance. Machine learning models can leak sensitive training data [45, 13, 42]. Differential privacy (DP) [18] mitigates this by ensuring negligible dependence on any single data point.

While differentially private optimization has been extensively studied in a variety of settings [9, 6, 4, 8, 22, 35], the community's understanding of DP BLO is limited. Indeed, we are only aware of two prior works on DP BLO [15, 28]. The work of [15] considers local DP [26] and does not provide guarantees in the important privacy regime ε = O (1) . On the other hand, [28] provides guarantees for central DP nonconvex BLO with any ε &gt; 0 , which we improve over in this work.

In this work, we provide DP algorithms and error bounds for two fundamental BLO problems. The first BLO problem we study is bilevel empirical risk minimization (ERM) w.r.t. data set Z = ( z 1 , . . . , z n ) ∈ Z n :

<!-- formula-not-decoded -->

where f : X × R d y ×Z → R and g : X × R d y ×Z → R are smooth upper- and lower-level loss functions. Second, we consider bilevel stochastic optimization (SO) :

<!-- formula-not-decoded -->

We assume, as is standard, that g ( x, · , z ) is strongly convex, so ∀ x there are unique ̂ y ∗ Z ( x ) and y ∗ ( x ) .

A fundamental open problem in DP BLO is to determine the minimax optimal error rates for solving problems Bilevel ERM and Bilevel SO. A natural first step is to consider the convex case:

Question 1. What are the optimal error rates for solving problem Bilevel ERM with DP when ̂ Φ Z is convex?

Convex ̂ Φ Z , Φ arise in a variety of applications [33], including few-shot meta-learning with a shared embedding model [11], biased regularization in hyperparameter optimization [25], fair resource allocation in communication networks [46], and bilevel optimization with smooth convex f ( · , y ) and quadratic g ( x, · ) [33].

Contribution 1. We give a (nearly) complete answer to Question 1 for both pure ε -DP and approximate ( ε, δ ) -DP, by providing nearly tight upper and lower bounds : see Section 3. Our results show that if the smoothness, Lipschitz, and strong convexity parameters are constants, then it is possible to achieve the same rates as standard single-level convex DP-ERM [9], despite the more challenging bilevel setting (e.g., O ( d x /εn ) for ε -DP bilevel ERM). On the other hand, our lower bound establishes a novel separation between standard single-level DP optimization and DP BLO , showing that the error of any algorithm for DP BLO must necessarily depend on the complexity parameters of the lower-level problem (e.g. the Lipschitz parameter of g ( x, · , z ) ). Our algorithms are built on the exponential mechanism [40] for ε -DP and the regularized exponential mechanism [24] for ( ε, δ ) -DP. We provide efficient (i.e. polynomial-time) implementations of these mechanisms for DP BLO and a novel analysis of how function evaluation errors affect log-concave sampling algorithms. Additionally, we provide upper and lower bounds on the excess population risk for DP Bilevel SO.

DP Nonconvex BLO. The recent work of [28] provided an ( ε, δ ) -DP algorithm A capable of finding approximate stationary points of nonconvex ̂ Φ Z such that

<!-- formula-not-decoded -->

If d y is large, bound (2) suffers: e.g., if d y ≥ d x , then the bound is ≳ ( √ d y /εn ) 1 / 3 . This leads us to:

Question 2. Can we improve over the state-of-the-art bound in (2) for DP stationary points in nonconvex Bilevel ERM?

Contribution 2: We give a positive answer to Question 2 in Section 4, developing novel DP algorithms that improve over the bound in (2). Our first algorithm A 1 is a simple and efficient second-order DP BLO method that achieves an improved d y -independent bound of

<!-- formula-not-decoded -->

Second, we provide an (inefficient) algorithm A 2 that uses the exponential mechanism to 'warm start' A 1 using the framework of [37] to obtain a further improved bound in the parameter regime d x &lt; nε :

<!-- formula-not-decoded -->

As detailed in Appendix C.3, our results imply a new state-of-the-art upper bound for DP non-convex bilevel finite-sum optimization:

<!-- formula-not-decoded -->

## 1.1 Technical overview

We develop and utilize several novel algorithmic and analytic techniques to obtain our results.

Techniques for convex DP BLO: Our algorithms are built on the exponential and regularized exponential mechanisms [40, 24]. A key challenge is to implement these algorithms efficiently in BLO, where one lacks access to ̂ y ∗ Z ( x ) and hence cannot directly query ̂ Φ Z ( x ) . To overcome this challenge, we provide a novel analysis of log-concave sampling with inexact function evaluations, building on the grid-walk algorithm of [3] and the approach of [9]. To do so, we prove a bound on the conductance of the perturbed Markov chain arising from the grid-walk with perturbed/inexact function evaluation, as well as a bound on the relative distance between the original and perturbed stationary distributions. We believe these techniques and analyses may be of independent interest, since there are many problems beyond BLO where access to exact function evaluations is unavailable.

To prove our lower bounds, we construct a novel bilevel hard instance with linear upper-level f and quadratic lower-level g . This allows us to chain together the x and y variables, control ̂ y ∗ Z ( x ) , and reduce BLO to mean estimation. By carefully scaling our hard instance, we obtain our lower bound.

Techniques for nonconvex DP BLO: In the nonconvex setting, our algorithm uses a secondorder approximation ∇ ̂ F Z ( x t , y t +1 ) ≈ ∇ ̂ Φ Z ( x t ) in order to approximate gradient descent run on ̂ Φ Z . A key insight is that we can obtain a better bound by getting a high-accuracy non-private approximate solution y t +1 ≈ ̂ y ∗ Z ( x t ) and then noising ∇ ̂ F Z ( x t , y t +1 ) , rather than privatizing y t +1 . To prove such an approach can be made DP, we require a careful sensitivity analysis that leverages perturbation inequalities from numerical analysis. Further, we build a two-step algorithm on our novel second-order algorithm by leveraging the warm-start framework of [38].

## 2 Preliminaries

Notation and assumptions. Let f : X × R d y × Z → R and g : X × R d y × Z → R be loss functions, with X ⊂ R d x being a closed convex set of ℓ 2 -diameter D x ∈ [0 , ∞ ] . The data universe Z can be any set. P denotes any data distribution on X . Let ∥ · ∥ denote the ℓ 2 norm when applied to vectors. When applied to matrix A , ∥ A ∥ := s max ( A ) = √ λ max ( AA T ) denotes the ℓ 2 operator norm, which is the largest singular value of A . Function h : X → R is L -Lipschitz if | h ( x ) -h ( x ′ ) | ≤ L ∥ x -x ′ ∥ for all x, x ′ ∈ X . Function h : X → R is µ -strongly convex if h ( αx +(1 -α ) x ′ ) ≤ αh ( x ) +(1 -α ) h ( x ′ ) -α (1 -α ) µ 2 ∥ x -x ′ ∥ 2 for all α ∈ [0 , 1] and all x, x ′ ∈ X . If µ = 0 , we say h is convex . The excess (population) risk of a randomized algorithm A with

output ̂ x = A ( Z ) on loss function h ( x, z ) is E A ,Z [ H ( ̂ x )] -H ∗ , where H ( x ) = E z ∼ P [ h ( x, z )] and H ∗ := inf x H ( x ) . If ̂ H Z ( x ) = 1 n ∑ n i =1 h ( x, z i ) is an empirical loss function w.r.t. data set Z , then the excess empirical risk of A is E A [ ̂ H Z ( ̂ x )] -̂ H ∗ . Denote a ∧ b := min( a, b ) . For functions φ and ψ of input parameters θ , we write φ ≲ ψ if there is an absolute constant C &gt; 0 such that φ ( θ ) ≤ Cψ ( θ ) for all permissible values of θ . We use ˜ O to hide logarithmic factors. Denote by ∇ J ( x, y ( x ) , z ) = ∇ x J ( x, y ( x ) , z ) + ∇ y ( x ) T ∇ y J ( x, y ( x ) , z ) the gradient of function J w.r.t. x .

We assume, as is standard in DP optimization, that the loss functions are Lipschitz continuous, and that g ( x, · , z ) is strongly convex-a standard assumption in the BLO literature:

Assumption 2.1. 1. f ( · , y, z ) is L f,x -Lipschitz in x for all y, z .

2. f ( x, · , z ) is L f,y -Lipschitz in y for all x, z .
3. g ( x, · , z ) is µ g -strongly convex in y .
4. There exists a compact set Y ⊂ R d y with { ̂ y ∗ Z ( x ) } x ∈X ⊆ Y for ERM or { y ∗ ( x ) } x ∈X ⊆ Y for SO such that g ( x, · , z ) is L g,y -Lipschitz on Y .

Note that D y := diam ( Y ) ≤ L g,y µ g by Assumption 2.1. Some of our algorithms additionally require: Assumption 2.2. For all x, x ′ y, y ′ , z we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assumption 2.2 is standard for second-order optimization methods and is essentially the same as the [28, Assumptions 2.5 and 2.6], but we define the different smoothness parameters at a more granular level to get more precise bounds. As discussed in [23], these assumptions are satisfied in important applications of BLO, such as model selection and hyperparameter tuning with logistic loss (or another loss with bounded gradient and Hessian) and some Stackelberg game models.

Differential Privacy. Differential privacy prevents any adversary from inferring much more about any individual's data than if that data had not been used for training.

̸

Definition 2.3 (Differential Privacy) . Let ε ≥ 0 , δ ∈ [0 , 1) . Randomized algorithm A : Z n →W is ( ε, δ ) -differentially private (DP) if for any two datasets Z = ( z 1 , . . . , z n ) and Z ′ = ( z ′ 1 , . . . , z ′ n ) that differ in one data point (i.e. z i = z ′ i , z j = z ′ j for j = i ) and any measurable set S ⊂ X , we have

̸

<!-- formula-not-decoded -->

Algorithmic preliminaries on DP are given in Appendix A.

## 3 Private convex bilevel optimization

In this section, we characterize the optimal excess risk bounds for DP convex bilevel ERM:

Theorem 3.1 (Convex DP BLO - Informal) . Let ̂ Φ Z and Φ be convex ( ∀ Z ∈ Z n ) and grant Assumption 2.1. Then, there is an efficient ε -DP algorithm with output ̂ x such that

<!-- formula-not-decoded -->

If Assumption 2.2 parts 3-4 hold, then there is an efficient ( ε, δ ) -DP algorithm with output ̂ x s.t.

<!-- formula-not-decoded -->

Moreover, the above upper bounds are tight (optimal) up to logarithmic factors.

The following subsections contain formal statements capturing the precise dependence on the problem parameters given in Assumptions 2.1 and 2.2 and runtime bounds. We also provide bounds for DP convex bilevel SO.

## 3.1 Conceptual algorithms and excess risk upper bounds

This section contains our conceptual algorithms (ignoring efficiency considerations) and precise excess risk upper bounds.

Pure ε -DP. Consider the following sampler for DP bilevel ERM, which is an instantiation of the exponential mechanism [40]: Given Z ∈ Z n , sample ̂ x = ̂ x ( Z ) ∈ X with probability

<!-- formula-not-decoded -->

Theorem 3.2. Grant Assumption 2.1 and suppose ̂ Φ Z is convex. The Algorithm in (4) is ε -DP and

<!-- formula-not-decoded -->

We defer the proof to Appendix B and describe the efficient implementation in Section 3.2. If Φ is not convex, then privacy still holds and the same excess risk holds up to logarithmic factors. The key step in the privacy proof is to upper bound the sensitivity of the score function ̂ Φ Z ( x ) by s , by leveraging Assumption 2.1 and the fact that ∥ ̂ y ∗ Z ( x ) -̂ y ∗ Z ′ ( x ) ∥ ≤ 2 L g,y /µ g n for adjacent Z ∼ Z ′ .

Approximate ( ε, δ ) -DP. Consider the following instantiation of the regularized exponential mechanism [24]: Given Z , sample ̂ x = ̂ x ( Z ) from probability density function

<!-- formula-not-decoded -->

where µ is an algorithmic parameter that we will assign (not to be confused with µ g ).

Theorem 3.3 (Informal) . Grant Assumption 2.1 and parts 3 and 4 of Assumption 2.2. Assume ̂ Φ Z and Φ are convex for all Z ∈ Z n . There exists a choice of µ and k such that Algorithm (5) is ( ε, δ ) -DP and achieves excess empirical risk

<!-- formula-not-decoded -->

Further, if Z ∼ P n are independent samples, the excess population risk with a different choice of k, µ is

<!-- formula-not-decoded -->

Refer to Theorem B.8 in Appendix B for the precise population loss bound with proper units. The main idea of the privacy proof (in Appendix B) is to show that ̂ Φ Z -̂ Φ Z ′ is 2( L f,x n + L f,y β g,xy µ g n + L g,y β f,xy nµ g ) -Lipschitz and then compare the privacy curve [5] between the distributions Q and Q ′ (corresponding to (5) with data Z and Z ′ respectively) to the privacy curve between two Gaussians, by leveraging [24, Theorem 4.1].

Obtaining our dimension-independent generalization error bound involves a fairly long 'ghost sample' argument that leverages the Wasserstein distance bound for log-concave distributions, KantorovichRubinstein duality, and the Efron-Stein inequality. One can also obtain an O ( √ d/n ) generalization error bound by a uniform convergence argument; we omit those details here.

Remark 3.4 (Near-optimality for ERM) . The bounds in Theorem 3.2 and 3.3 nearly match the optimal bounds for standard single-level DP ERM [9], e.g. Θ( L f,x D x √ d x log(1 /δ ) /εn ) for ( ε, δ ) -DP ERM [9, 48], except for the addition of two terms capturing the complexity of the bilevel problem: For ε -DP ERM, the additional terms are O ( L f,y D y d x /εn ) and O (( L f,y L g,y /µ g ) d x /εn ) . Our lower bound in Theorem 3.10 shows that the first additional term is necessary. We conjecture that the second additional term is also necessary and that our upper bound is tight up to an absolute constant . This conjecture is clearly true in the parameter regime L g,y /µ g ≈ D y . For ( ε, δ ) -DP, the additional terms scale with O (( L f,y β g,xy /µ g + L g,y β f,xy /µ g ) D x ) . Our lower bound in Theorem 3.10 shows that dependence on L f,y is necessary and that the L f,y β g,xy /µ g term is tight in the parameter regime D y ≈ D x β g,xy /µ g . If also D x β g,xy / ≲ L g,y , then the bounds in Theorem 3.3 are tight up to an absolute constant factor.

Remark 3.5 (Suboptimality for SO) . There is a gap between our population risk upper bound in Theorem 3.3 and the lower bounds in [6] and Remark 3.11 for single-level SCO and bilevel SO respectively. We conjecture that our lower bound in Remark 3.11 is nearly tight and that our upper bound is suboptimal. We leave it as future work to investigate this conjecture.

## 3.2 Efficient implementation of conceptual algorithms

In many practical applications of optimization and sampling algorithms, we face unavoidable approximation errors when evaluating functions. Given any x , we may not get the exact ̂ y ∗ Z ( x ) in solving the low-level optimization, which means we may introduce a small error each time we compute the function value of f ( x, ̂ y ∗ Z ( x ) , z ) . This section analyzes how such small function evaluation errors affect log-concave sampling algorithms. We establish bounds on the impact of errors bounded by ζ on the conductance, mixing time, and distributional accuracy of Markov chains used for sampling. We then develop an efficient implementation based on the [9] approach that maintains polynomial time complexity while providing formal guarantees on sampling accuracy in the presence of function evaluation errors. As a corollary of our developments, we obtain Theorem 3.1.

Our approach builds on the classic Grid-Walk algorithm of [3] for sampling from log-concave distributions. Let F ( · ) be a real positive-valued function defined on a cube A = [ a, b ] d in R d . Let f ( θ ) = -log F ( θ ) and suppose there exist real numbers α, β such that:

<!-- formula-not-decoded -->

for all x, y ∈ A and λ ∈ [0 , 1] . The algorithm of [3], detailed in Appendix B.3.1 for completeness, samples from a distribution ν on the continuous domain A such that for all θ ∈ A , | ν ( θ ) -cF ( θ ) | ≤ ζ , where c is a normalization constant and ζ &gt; 0 . The algorithm defines a random walk (which is a Markov Chain) on the centers of small subcubes that partition A and form the state space Ω ⊂ A . The final output of the algorithm is a point x ∈ A , returned with probability close to F ( x ) .

Next, we briefly outline our analysis how the Grid-Walk algorithm behaves when the function F can only be evaluated with some bounded error, resulting in a 'perturbed' Markov chain.

Conductance bound with function evaluation errors. For a Markov chain with state space Ω , transition matrix P and stationary distribution q , its conductance φ measures how well the chain mixes, i.e. how quickly it converges to its stationary distribution:

<!-- formula-not-decoded -->

We analyze how function evaluation errors affect Grid-Walk conductance:

Lemma 3.6 (Conductance with Function Evaluation Errors) . Let P be the transition matrix of the original Markov chain in the grid-walk algorithm of Section B.3.1 based on function f , with state space Ω and conductance φ . Let P ′ be the transition matrix of the perturbed chain based on f ′ where f ′ ( θ ) = f ( θ ) + ζ ( θ ) with | ζ ( θ ) | ≤ ζ for all θ ∈ Ω , where ζ ( · ) is a bounded error function. Then the conductance φ ′ of the perturbed chain satisfies:

<!-- formula-not-decoded -->

Relative distance bound between F and F ′ . We now analyze how function evaluation errors affect the distributional distance between the original and perturbed stationary distributions.

Lemma 3.7 (Distance Between F and F ′ ) . Let F ( θ ) = e -f ( θ ) and F ′ ( θ ) = e -f ′ ( θ ) where f ′ ( θ ) = f ( θ ) + ζ ( θ ) with | ζ ( θ ) | ≤ ζ for all θ ∈ A . Then,

<!-- formula-not-decoded -->

Furthermore, if we define the distributions π ( θ ) ∝ F ( θ ) and π ′ ( θ ) ∝ F ′ ( θ ) , then:

<!-- formula-not-decoded -->

Mixing time analysis. For a Markov chain with state space Ω , transition matrix P , and stationary distribution π , the mixing time t mix ( ϵ ) with respect to the L ∞ -distance is defined as:

<!-- formula-not-decoded -->

for any ϵ ≥ 0 . We determine the number of steps required for L ∞ convergence with perturbed F :

Lemma 3.8 (Impact on Mixing Time) . The mixing time t ′ mix ( ϵ ) of the perturbed chain to achieve L ∞ -distance ϵ to its stationary distribution satisfies:

<!-- formula-not-decoded -->

Efficient implementation. Leveraging our analysis of how function evaluation errors affect conductance, mixing time, and distributional distance, we develop an efficient algorithm for sampling from log-concave distributions in the presence of such errors. Our approach builds upon the framework developed by [9], extending it to handle approximation errors.

Theorem 3.9 (Log-Concave Sampling with Function Evaluation Error) . Let C ⊂ R d be a convex set and f : C → R be a convex, L -Lipschitz function. Suppose we have access to an approximate function evaluator that returns f ′ ( θ ) = f ( θ ) + ζ ( θ ) where | ζ ( θ ) | ≤ ζ for all θ ∈ C , and ζ = O (1) is a constant independent of dimension. There exists an efficient algorithm that outputs a sample θ ∈ C from a distribution µ ′ such that:

<!-- formula-not-decoded -->

where π ( θ ) ∝ e -f ( θ ) is the target log-concave distribution and δ &gt; 0 is an arbitrarily small constant. This algorithm runs in time O ( e 12 ζ · d 3 · poly ( L, ∥ C ∥ 2 , 1 /ξ )) .

The efficiency claims in Theorem 3.1 follow as corollaries of Theorem 3.9: see Appendix B.3.

## 3.3 Excess risk lower bounds

If the problem parameters (e.g., Lipschitz, smoothness) are constants, then the upper bounds in Theorems 3.2 and 3.3 are tight and match known lower bounds for standard single-level DP ERM and SCO [9, 6]. In this section, we go a step further and provide novel lower bounds illustrating that the dependence of our bounds on L f,y D y (or a quantity larger than this) is necessary, thereby establishing a novel separation between single-level DP optimization and DP BLO :

Theorem 3.10 (Excess risk lower bounds for DP ERM) . 1. Let A be ε -DP. Then, there exists a data set Z ∈ Z n and a convex bilevel ERM problem instance satisfying Assumptions 2.1 and 2.2 with µ g = Θ( L g,y /D y ) such that

<!-- formula-not-decoded -->

2. Let A be ( ε, δ ) -DP with 2 -Ω( n ) ≤ δ ≤ 1 /n 1+Ω(1) . Then, there exists a data set Z ∈ Z n and a convex bilevel ERM problem instance satisfying Assumptions 2.1 and 2.2 with µ g = Θ( L g,y /D y ) such that

<!-- formula-not-decoded -->

By comparing the lower bounds in Theorem 3.10 with the bounds in [9], one sees that the DP bilevel ERM is harder (in terms of minimax error) than standard single-level DP ERM if L f,y D y &gt; L f,x D x .

See Appendix B.4 for the proof. A key challenge is in constructing the right f and g to chain together the x and y variables and obtain the desired L f,y D y scaling term.

Remark 3.11 (Bilevel SO lower bounds) . One can obtain lower bounds on the excess population risk that are larger than the excess empirical risk bounds in Theorem 3.10 by an additive L f,x D x (1 / √ n ) , via the reduction in [7].

## 4 Private non-convex bilevel optimization

In this section, we provide novel algorithms with state-of-the-art guarantees for privately finding approximate stationary points of non-convex ̂ Φ Z (see (3)).

## 4.1 An iterative second-order method

Assume for simplicity that X = R d x so that the optimization problem is unconstrained. 2 A natural approach to solving Bilevel ERM is to use a gradient descent scheme, where we iterate

<!-- formula-not-decoded -->

By the implicit function theorem, we have (c.f. [23]):

<!-- formula-not-decoded -->

Define the following approximation to ∇ ̂ Φ Z ( x ) at ( x, y ) :

<!-- formula-not-decoded -->

Note that ∇ ̂ F Z ( x, y ) = ∇ ̂ Φ Z ( x ) if y = ̂ y ∗ Z ( x ) .

Then to approximate (8) (non-privately), we can iterate (c.f. [23]):

<!-- formula-not-decoded -->

A naive approach to privatizing the iterations (10) is to solve y t +1 ≈ ̂ y ∗ Z ( x t ) = argmin y ̂ G Z ( x t , y ) privately at each step (e.g., by running DP-SGD), and then add noise to ∇ ̂ F Z ( x t , y t +1 ) before taking a step of noisy GD. (This is similar to how [28] privatized the penalty-based bilevel optimization algorithm of [30].) However, this approach results in a bound E ∥∇ ̂ Φ Z ( ̂ x ) ∥ ≤ O ( √ d x + d y /εn ) 1 / 2 that depends on d y due to the bias ∥∇ ̂ F Z ( x t , y t +1 ) -∇ ̂ Φ Z ( x t ) ∥ that results from using private y t +1 . To mitigate this issue and obtain state-of-the-art utility independent of d y , we propose an alternative approach in Algorithm 1: we find an approximate minimizer of ̂ G Z ( x t , · ) non-privately in line 3.

Algorithm 1: A Second-Order DP Bilevel Optimization Algorithm

1 Input: Dataset D = ( Z 1 , . . . , Z n ) , noise scale σ , initial points x 0 , y 0 ∈ X × Y , parameter α ; 2 for i = 0 , . . . , T -1 do

<!-- formula-not-decoded -->

- 4 end
- 5 Output: ̂ x T ∼ Unif ( { x t } T t =1 ) .

Since ̂ G Z ( x t , · ) is a smooth, strongly convex ERM function, we can implement line 3 efficiently using a non-private algorithm such as SGD or Katyusha [2].

Denote L := L f,x + β g,xy L f,y µ g , which is an upper bound on ∥∇ f ( x, ̂ y ∗ Z ( x ) , z ) ∥ , and

<!-- formula-not-decoded -->

which satisfies ∥∇ ̂ Φ Z ( x ) -∇ ̂ F Z ( x, y ) ∥ ≤ C ∥ ̂ y ∗ Z ( x ) -y ∥ for any x, y by [23, Lemma 2.2]. Let

<!-- formula-not-decoded -->

Lemma 4.1 (Sensitivity Bound for Algorithm 1) . For any fixed x t , define the query q t : Z n → R d ,

<!-- formula-not-decoded -->

where y t +1 = y t +1 ( Z ) is given in Algorithm 1. If α ≤ K Cn where C and K are defined in Equations (11) and (12) , then the ℓ 2 -sensitivity of q t is upper bounded by 4 K n .

The proof of this lemma-in Appendix C-is long. It uses the operator norm perturbation inequality ∥ M -1 -N -1 ∥ ≤ ∥ M -1 ∥∥ N -1 ∥∥ M -N ∥ to bound the sensitivity of [ ∇ 2 yy ̂ G Z ( x t , y y +1 )] -1 in (9).

Now we can state the main result of this subsection:

Theorem 4.2 (Guarantees of Algorithm 1 for Non-Convex Bilevel ERM - Informal) . Grant Assumptions 2.1 and 2.2. Set σ = 32 K √ T log(1 /δ ) /nε . Denote the smoothness parameter of ̂ Φ Z by β Φ , given in Lemma C.2. There are choices of α, η s.t. Algorithm 1 is ( ε, δ ) -DP and has output satisfying

<!-- formula-not-decoded -->

The privacy proof leverages Lemma 4.1. Utility is analyzed through the lens of gradient descent with biased, noisy gradient oracle. We choose small α so the bias is negligible and use smoothness of ̂ Φ Z .

## 4.2 'Warm starting' Algorithm 1 with the exponential mechanism

This subsection provides an algorithm that enables an improvement over the utility bound given in Theorem 4.2 in the parameter regime d x &lt; nε . Our algorithm is built on the 'warm start' framework of [37]: first, we run the exponential mechanism (4) with privacy parameter ε/ 2 to obtain x 0 ; then, we run ( ε/ 2 , δ ) -DP Algorithm 1 with 'warm' initial point x 0 . See Algorithm 2 in Appendix C.2.

Theorem 4.3 (Guarantees of Algorithm 2 for Non-Convex Bilevel ERM) . Grant Assumptions 2.1 and 2.2. Assume that there is a compact set X ⊂ R d x of diameter D x containing an approximate global minimizer ̂ x such that ̂ Φ Z ( ̂ x ) -̂ Φ ∗ Z ≤ Ψ d εn , where Ψ := L f,x D x + L f,y D y + L f,y L g,y µ g . Then, there exists an ( ε, δ ) -DP instantiation of Algorithm 2 with output satisfying

<!-- formula-not-decoded -->

2 Our approach and results readily extend to constrained X by incorporating proximal steps and measuring utility in terms of the norm of the proximal gradient mapping.

In Appendix C.3, we explain how to deduce the upper bound in (3) by combining Theorems 4.2 and 4.3 with the exponential mechanism using cost function ∥∇ ̂ Φ Z ( x ) ∥ .

## 5 Conclusion and discussion

We provided novel algorithms and lower bounds for differentially private bilevel optimization, with near-optimal rates for the convex setting and state-of-the-art rates for the nonconvex setting. There are some interesting open problems for future work to explore: (1) What are the optimal rates for DP bilevel convex SO ? As discussed in Remark 3.5, we believe that it should be possible, though challenging, to obtain an improved O (1 / √ n ) generalization error bound nearly matching the lower bound in Remark 3.11. (2a) What are the optimal rates for DP nonconvex bilevel ERM and SO ? Since the optimal rates for standard single-level DP nonconvex ERM and SO are still unknown, a first step would be to answer: (2b) Can we match the SOTA rate for single-level non-convex ERM [38] in BLO ? Incorporating variance-reduction in DP BLO seems challenging. (3) This work was focused on fundamental theoretical questions about DP BLO, but another important direction is to provide practical implementations and experimental evaluations .

## References

- [1] A. Ajalloeian and S. U. Stich. On the convergence of sgd with biased gradients. arXiv preprint arXiv:2008.00051 , 2020.
- [2] Z. Allen-Zhu. Katyusha: The first direct acceleration of stochastic gradient methods. The Journal of Machine Learning Research , 18(1):8194-8244, 2017.
- [3] D. Applegate and R. Kannan. Sampling and integration of near log-concave functions. In Proceedings of the twenty-third annual ACM symposium on Theory of computing , pages 156163, 1991.
- [4] H. Asi, V. Feldman, T. Koren, and K. Talwar. Private stochastic convex optimization: Optimal rates in ℓ 1 geometry. In ICML , 2021.
- [5] B. Balle and Y.-X. Wang. Improving the gaussian mechanism for differential privacy: Analytical calibration and optimal denoising. In International Conference on Machine Learning , pages 394-403. PMLR, 2018.
- [6] R. Bassily, V. Feldman, K. Talwar, and A. Thakurta. Private stochastic convex optimization with optimal rates. In Advances in Neural Information Processing Systems , volume 32, 2019.
- [7] R. Bassily, V. Feldman, K. Talwar, and A. Thakurta. Private stochastic convex optimization with optimal rates. In Advances in Neural Information Processing Systems , volume 32, pages 11282-11291, 2019.
- [8] R. Bassily, C. Guzmán, and M. Menart. Differentially private algorithms for the stochastic saddle point problem with optimal rates for the strong gap. In The Thirty Sixth Annual Conference on Learning Theory , pages 2482-2508. PMLR, 2023.
- [9] R. Bassily, A. Smith, and A. Thakurta. Private empirical risk minimization: Efficient algorithms and tight error bounds. In 2014 IEEE 55th Annual Symposium on Foundations of Computer Science , pages 464-473. IEEE, 2014.
- [10] K. P. Bennett, J. Hu, X. Ji, G. Kunapuli, and J.-S. Pang. Model selection via bilevel optimization. In The 2006 IEEE International Joint Conference on Neural Network Proceedings , pages 1922-1929. IEEE, 2006.
- [11] L. Bertinetto, J. F. Henriques, P. H. Torr, and A. Vedaldi. Meta-learning with differentiable closed-form solvers. arXiv preprint arXiv:1805.08136 , 2018.
- [12] O. Bousquet and A. Elisseeff. Stability and generalization. The Journal of Machine Learning Research , 2:499-526, 2002.

- [13] N. Carlini, F. Tramer, E. Wallace, M. Jagielski, A. Herbert-Voss, K. Lee, A. Roberts, T. B. Brown, D. Song, U. Erlingsson, et al. Extracting training data from large language models. In USENIX Security Symposium , volume 6, pages 2633-2650, 2021.
- [14] L. Chen, J. Xu, and J. Zhang. On finding small hyper-gradients in bilevel optimization: Hardness results and improved analysis. In The Thirty Seventh Annual Conference on Learning Theory , pages 947-980. PMLR, 2024.
- [15] Z. Chen and Y. Wang. Locally differentially private decentralized stochastic bilevel optimization with guaranteed convergence accuracy. In Forty-first International Conference on Machine Learning , 2024.
- [16] B. Colson, P. Marcotte, and G. Savard. An overview of bilevel optimization. Annals of operations research , 153:235-256, 2007.
- [17] E. De Klerk and M. Laurent. Comparison of lasserre's measure-based bounds for polynomial optimization to bounds obtained by simulated annealing. Mathematics of Operations Research , 43(4):1317-1325, 2018.
- [18] C. Dwork, F. McSherry, K. Nissim, and A. Smith. Calibrating noise to sensitivity in private data analysis. In Theory of cryptography conference , pages 265-284. Springer, 2006.
- [19] C. Dwork and A. Roth. The Algorithmic Foundations of Differential Privacy , volume 9. Now Publishers, Inc., 2014.
- [20] J. E. Falk and J. Liu. On bilevel programming, part i: general nonlinear cases. Mathematical Programming , 70:47-72, 1995.
- [21] L. Franceschi, P. Frasconi, S. Salzo, R. Grazzi, and M. Pontil. Bilevel programming for hyperparameter optimization and meta-learning. In International conference on machine learning , pages 1568-1577. PMLR, 2018.
- [22] C. Gao, A. Lowy, X. Zhou, and S. Wright. Private heterogeneous federated learning without a trusted server revisited: Error-optimal and communication-efficient algorithms for convex losses. In Forty-first International Conference on Machine Learning , 2024.
- [23] S. Ghadimi and M. Wang. Approximation methods for bilevel programming. arXiv preprint arXiv:1802.02246 , 2018.
- [24] S. Gopi, Y. T. Lee, and D. Liu. Private convex optimization via exponential mechanism. In Conference on Learning Theory , pages 1948-1989. PMLR, 2022.
- [25] R. Grazzi, L. Franceschi, M. Pontil, and S. Salzo. On the iteration complexity of hypergradient computation. In International Conference on Machine Learning , pages 3748-3758. PMLR, 2020.
- [26] S. P. Kasiviswanathan, H. K. Lee, K. Nissim, S. Raskhodnikova, and A. Smith. What can we learn privately? SIAM Journal on Computing , 40(3):793-826, 2011.
- [27] V. Konda and J. Tsitsiklis. Actor-critic algorithms. Advances in neural information processing systems , 12, 1999.
- [28] G. Kornowski. Differentially private bilevel optimization. arXiv preprint arXiv:2409.19800 , 2024.
- [29] G. Kunapuli, K. P. Bennett, J. Hu, and J.-S. Pang. Bilevel model selection for support vector machines. Data mining and mathematical programming , 45:129, 2008.
- [30] J. Kwon, D. Kwon, S. Wright, and R. D. Nowak. A fully first-order method for stochastic bilevel optimization. In International Conference on Machine Learning , pages 18083-18113. PMLR, 2023.
- [31] J. Kwon, D. Kwon, S. Wright, and R. D. Nowak. On penalty methods for nonconvex bilevel optimization and first-order stochastic approximation. In The Twelfth International Conference on Learning Representations , 2024.

- [32] Y. T. Lee, A. Sidford, and S. C.-w. Wong. A faster cutting plane method and its implications for combinatorial and convex optimization. In 2015 IEEE 56th Annual Symposium on Foundations of Computer Science , pages 1049-1065. IEEE, 2015.
- [33] Y. Liang et al. Lower bounds and accelerated algorithms for bilevel optimization. Journal of machine learning research , 24(22):1-56, 2023.
- [34] B. Liu, M. Ye, S. Wright, P. Stone, and Q. Liu. Bome! bilevel optimization made easy: A simple first-order approach. Advances in neural information processing systems , 35:17248-17262, 2022.
- [35] A. Lowy, D. Liu, and H. Asi. Faster algorithms for user-level private stochastic convex optimization. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [36] A. Lowy and M. Razaviyayn. Output perturbation for differentially private convex optimization: Faster and more general. arXiv preprint arXiv:2102.04704 , 2021.
- [37] A. Lowy, J. Ullman, and S. Wright. How to make the gradients small privately: Improved rates for differentially private non-convex optimization. In Forty-first International Conference on Machine Learning , 2024.
- [38] A. Lowy, J. Ullman, and S. J. Wright. How to make the gradients small privately: Improved rates for differentially private non-convex optimization. arXiv preprint arXiv:2402.11173 , 2024.
- [39] Z. Lu and S. Mei. First-order penalty methods for bilevel optimization. SIAM Journal on Optimization , 34(2):1937-1969, 2024.
- [40] F. McSherry and K. Talwar. Mechanism design via differential privacy. In 48th Annual IEEE Symposium on Foundations of Computer Science (FOCS'07) , pages 94-103. IEEE, 2007.
- [41] B. Morris and Y. Peres. Evolving sets, mixing and heat kernel bounds. Probability Theory and Related Fields , 133(2):245-266, 2005.
- [42] M. Nasr, J. Rando, N. Carlini, J. Hayase, M. Jagielski, A. F. Cooper, D. Ippolito, C. A. ChoquetteChoo, F. Tramèr, and K. Lee. Scalable extraction of training data from aligned, production language models. In The Thirteenth International Conference on Learning Representations , 2025.
- [43] A. Rajeswaran, C. Finn, S. M. Kakade, and S. Levine. Meta-learning with implicit gradients. Advances in neural information processing systems , 32, 2019.
- [44] S. Shalev-Shwartz, O. Shamir, N. Srebro, and K. Sridharan. Stochastic convex optimization. In COLT , volume 2, page 5, 2009.
- [45] R. Shokri, M. Stronati, C. Song, and V. Shmatikov. Membership inference attacks against machine learning models. In 2017 IEEE symposium on security and privacy (SP) , pages 3-18. IEEE, 2017.
- [46] R. Srikant and L. Ying. Communication networks: An optimization, control and stochastic networks perspective . Cambridge University Press, 2014.
- [47] H. Stackelberg. The Theory of the Market Economy . Oxford University Press, 1952.
- [48] T. Steinke and J. Ullman. Between pure and approximate differential privacy. Journal of Privacy and Confidentiality , 7(2), 2016.
- [49] Y. Zhang, G. Zhang, P. Khanduri, M. Hong, S. Chang, and S. Liu. Revisiting and advancing fast adversarial training through the lens of bi-level optimization. In International Conference on Machine Learning , pages 26693-26712. PMLR, 2022.

## Appendix

## A More privacy preliminaries

Definition A.1 (Sensitivity) . Given a function q : Z n → R k the ℓ 2 -sensitivity of q is defined as

<!-- formula-not-decoded -->

where the supremum is taken over all pairs of datasets that differ in one data point.

Definition A.2 (Gaussian Mechanism) . Let ε &gt; 0 , δ ∈ (0 , 1) . Given a function q : Z n → R k with ℓ 2 -sensitivity ∆ , the Gaussian Mechanism M is defined by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma A.3 (Privacy of Gaussian Mechanism [19]) . The Gaussian Mechanism is ( ε, δ ) -DP.

If we adaptively query a data set T times, then the privacy guarantees of the T -th query is still DP and the privacy parameters degrade gracefully:

Lemma A.4 (Advanced Composition Theorem [19]) . Let ε ≥ 0 , δ, δ ′ ∈ [0 , 1) . Assume A 1 , · · · , A T , with A t : Z n × X → X , are each ( ε, δ ) -DP ∀ t = 1 , · · · , T . Then, the adaptive composition A ( Z ) := A T ( Z, A T -1 ( Z, A T -2 ( X, · · · ))) is ( ε ′ , T δ + δ ′ ) -DP for

<!-- formula-not-decoded -->

## B Proofs for Section 3.1

## B.1 Conceptual algorithms and excess risk upper bounds

Pure ε -DP. We restate and prove the guarantees of the ε -DP exponential mechanism for BLO below:

Theorem B.1 (Re-statement of Theorem 3.2) . Grant Assumption 2.1 and suppose ̂ Φ Z is convex. The Algorithm in 4 is ε -DP and achieves excess empirical risk

<!-- formula-not-decoded -->

̸

Proof. Privacy: First, notice that the distribution induced by the exponential weight function in 4 is the same if we use exp ( -ε 2 s [ ̂ Φ Z ( x ) -̂ Φ Z ( x 0 )] ) for some arbitrary point x 0 ∈ X . To establish the privacy guarantee, it suffices to show that the sensitivity of ̂ Φ Z ( x ) -̂ Φ Z ( x 0 ) is upper bounded by s for any x . Now, let Z ∼ Z ′ be any adjacent data sets differing in z 1 = z ′ 1 and let x ∈ X . Then the sensitivity of ̂ Φ Z ( x ) -̂ Φ Z ( x 0 ) is upper bounded by

<!-- formula-not-decoded -->

Now, for any x , we have

Similarly for x ′ :

<!-- formula-not-decoded -->

by [44, 36]. Together with L f,y -Lipschitz continuity of f ( x, · , z ) , we can then obtain the desired sensitivity bound.

Excess risk: This is immediate from Lemma B.2 (stated below) in the convex case. For nonconvex ̂ Φ Z , the same excess risk bound holds up to logarithmic factors by [40].

Lemma B.2 (Utility Guarantee, [17, Corollary 1]) . Suppose k &gt; 0 and F is a convex function over the convex set K ⊆ R d . If we sample x according to distribution ν whose density is proportional to exp( -kF ( x )) , then we have

<!-- formula-not-decoded -->

Next, we turn to the ( ε, δ ) -DP case.

Approximate ( ε, δ ) -DP. We define the privacy curve first:

Definition B.3 (Privacy Curve) . Given two random variables X,Y supported on some set Ω , define the privacy curve δ ( X ∥ Y ) : R ≥ 0 → [0 , 1] as:

<!-- formula-not-decoded -->

We have the following theorem from [24]:

Theorem B.4 (Regularized Exponential Mechanism, [24]) . Given convex set K ⊆ R d and µ -strongly convex functions F, ˜ F over K . Let P, Q be distributions over K such that P ( x ) ∝ e -F ( x ) and Q ( x ) ∝ e -˜ F ( x ) . If ˜ F -F is G -Lipschitz over K , then for all z ∈ [0 , 1] ,

<!-- formula-not-decoded -->

It suffices to bound the Lipschitz constant of ̂ Φ Z ( x ) -̂ Φ Z ′ ( x ) . We have the following technical lemma:

Lemma B.5. Let ̂ y ∗ Z ( x ) = argmin y ∈Y ̂ G Z ( x, y ) where ̂ G Z ( x, y ) = 1 n ∑ n i =1 g ( x, y, z i ) . If g ( x, · , z ) is µ g -strongly convex in y and ∥∇ y ̂ G Z ( x, y ) -∇ y ̂ G Z ( x ′ , y ) ∥ ≤ β g,xy ∥ x -x ′ ∥ for all x, y, z , then

<!-- formula-not-decoded -->

Proof. Since ̂ y ∗ Z ( x ) is the minimizer of ̂ G Z ( x, y ) , the first-order optimality condition gives:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By µ g -strong convexity of ̂ G Z ( x ′ , · ) and the first-order optimality condition, we have:

<!-- formula-not-decoded -->

Therefore:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma B.6. Grant Assumption 2.1 and additionally assume ∥∇ x f ( x, y, z ) - ∇ x f ( x, y ′ , z ) ∥ ≤ β f,xy ∥ y -y ′ ∥ and ∥∇ y ̂ G Z ( x, y ) -∇ y ̂ G Z ( x ′ , y ) ∥ ≤ β g,xy ∥ x -x ′ ∥ for all x, x ′ , y, y ′ , z . Then, for any datasets Z, Z ′ ∈ Z n differing in one element, ̂ Φ Z -̂ Φ Z ′ is 2( L f,x n + L f,y β g,xy µ g n + L g,y β f,xy nµ g ) -Lipschitz.

̸

Proof. Suppose without loss of generality that Z and Z ′ differ only at the first element z 1 = z ′ 1 . Then:

<!-- formula-not-decoded -->

For i = 1 , we have

<!-- formula-not-decoded -->

where the last inequality follows from Lemma B.5. The same argument works for z ′ 1 .

For each i ≥ 2 (where z i is the same in both datasets), recalling that ∥ ̂ y ∗ Z ( x ) -̂ y ∗ Z ′ ( x ) ∥ ≤ 2 L g,y µ g n , we have

<!-- formula-not-decoded -->

Using the smoothness of ∇ x f with respect to y :

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

A similar analysis applies to the term involving z 1 and z ′ 1 , with an additional constant accounting for the difference between functions.

Therefore,

<!-- formula-not-decoded -->

## B.2 Generalization error of the regularized exponential mechanism for bilevel SCO

Another advantage of the Regularized Exponential Mechanism is that it can have a good generalization error.

Lemma B.7. If we sample the solution from density π Z ( x ) ∝ exp( -k ( ̂ Φ Z ( x ) + µ ∥ x ∥ 2 / 2) , the excess population loss is bounded as

<!-- formula-not-decoded -->

where

Proof. We have

<!-- formula-not-decoded -->

By Lemma B.2, we have which leads to that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next we bound the generalization error. Note that

<!-- formula-not-decoded -->

Recall that Z = { z 1 , · · · , z n } . Suppose we replace z i ∈ Z by a fresh independent sample z ′ i from P and get Z ′ . Then we have

<!-- formula-not-decoded -->

where the first equality follows by the argument in [12, Lemma 7]. Recall that in the proof of Theorem 3.2, we show that for any x ,

<!-- formula-not-decoded -->

By Lemma B.5, we know for any Z, z i , we have

<!-- formula-not-decoded -->

Moreover, by Lemma B.6 and [24], we can show that W 2 ( π Z , π Z ′ ) ≤ L µ with L = 2( L f,x n + L f,y β g,xy µ g n + L g,y β f,xy nµ g ) . Hence we have

<!-- formula-not-decoded -->

Now, by Lipschitz continuity, it remains to bound the right-hand side of

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For any x , by the strong convexity of g , we have

<!-- formula-not-decoded -->

By the first-order optimality, we know that ∇ y G ( x, y ∗ ( x )) = 0 , and hence we know

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, we have and

Next, we bound

<!-- formula-not-decoded -->

Let v ( Z, x ) := ∇ y G ( x, ̂ y ∗ Z ( x )) -∇ y ̂ G Z ( x, ̂ y ∗ Z ( x )) . Using the law of total variance, we have

<!-- formula-not-decoded -->

Note that we have

<!-- formula-not-decoded -->

where we apply Lemma B.5 for the last step. Similarly, we also have

<!-- formula-not-decoded -->

For Term A, for any dataset Z , π Z is kµ -strongly log-concave, and function v is Lipschitz in x with Lipschitz constant ( β g,xy + β g,yy β g,xy µ g ) specified in Equation 18. By Poincaré inequality, we have

<!-- formula-not-decoded -->

As for the Term B, let v ( Z ) = E x ∼ π Z v ( Z, x ) . By the Efron-Stein Inequality, we have

<!-- formula-not-decoded -->

where Z ′ j is the dataset with z j replaced by a fresh sample. Then we have

<!-- formula-not-decoded -->

By Kantorovich-Rubinstein duality and Cauchy-Schwartz, we have

<!-- formula-not-decoded -->

for any 1 -Lipschitz function h and any distributions π, π ′ ∈ L 2 . By Lemma B.5, the above fact implies

<!-- formula-not-decoded -->

A similar argument can be used to bound

∥ E x ∼ π Z ∇ y G ( x, ̂ y ∗ Z ( x )) -E x ′ ∼ π Z ′ ∇ y G ( x ′ , ̂ y ∗ Z ( x ′ )) ∥ ≤ β g,xy W 2 ( π Z , π Z ′ ) ≤ β g,xy L/µ, and likewise for ∇ y ̂ G Z ( x, ̂ y ∗ Z ( x )) . Hence

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

for C defined in the lemma statement.

We already bounded Var( v ( Z )) . As E ∥ v ( Z ) ∥ 2 = Var( v ( Z )) + ∥ E v ( Z ) ∥ 2 . It remains to bound Term C: ∥ E v ( Z ) ∥ 2 . For this, we have

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

Taking square roots and combining all the pieces above yields the lemma.

Theorem B.8 (Precise Statement of Theorem 3.3) . Grant Assumption 2.1 and parts 3 and 4 of Assumption 2.2. Assume ̂ Φ Z and Φ are convex for all Z . Sampling ̂ x from a distribution proportional to exp( -k ( ̂ Φ Z ( x ) + µ ∥ x ∥ 2 / 2)) with k = O ( µn 2 ε 2 G 2 log(1 /δ ) ) and G = ( L f,x + L f,y β g,xy µ g + L g,y β f,xy µ g ) is ( ε, δ ) -DP. Moreover,

- setting µ = G √ d x log(1 /δ ) nD x ε , we achieve excess risk

<!-- formula-not-decoded -->

- setting

<!-- formula-not-decoded -->

for C := β g,xy (1 + κ g ) with κ g = β g,yy /µ g , the population loss has the following guarantee:

<!-- formula-not-decoded -->

Proof. The privacy guarantee follows from the privacy curve of Gaussian variables and Lemma 6.3 in [24].

When setting µ = G √ d log(1 /δ ) nD x ε , Lemma B.2 gives us that

<!-- formula-not-decoded -->

The population loss guarantee follows from plugging the prescribed µ and k into Lemma B.7.

## B.3 Efficient implementation of conceptual algorithms

In many practical applications of optimization and sampling algorithms, we face unavoidable approximation errors when evaluating functions. Given any x , we may not get the exact ̂ y ∗ Z ( x ) in solving the low-level optimization, which means we may introduce a small error each time we compute the function value of f ( x, ̂ y ∗ Z ( x ) , z ) . This section analyzes how such small function evaluation errors affect log-concave sampling algorithms. We establish bounds on the impact of errors bounded by ζ on the conductance, mixing time, and distributional accuracy of Markov chains used for sampling. We then develop an efficient implementation based on the [9] approach that maintains polynomial time complexity while providing formal guarantees on sampling accuracy in the presence of function evaluation errors.

## B.3.1 Original Grid-Walk Algorithm for Log-Concave Sampling

We first state the classic Grid-Walk algorithm from Applegate and Kannan [3] on sampling from log-concave distributions.

Let F ( · ) be a real positive-valued function defined on a cube A = [ a, b ] d in R d , where [ a, b ] d represents a hypercube with side length κ := b -a . Let f ( θ ) = -log F ( θ ) and suppose there exist real numbers α, β such that:

<!-- formula-not-decoded -->

for all x, y ∈ A and λ ∈ [0 , 1] .

Let γ ≤ 1 / (2 α ) be a discretization parameter. The following algorithm samples from a distribution ν on the continuous domain A such that for all θ ∈ A , | ν ( θ ) -cF ( θ ) | ≤ ϵ , where c is a normalization constant:

1. Divide the cube A into small cubes { C x } of side length γ , with centers { x } . Let Ω be the set of all such centers.
2. If κ &lt; 1 /α , then pick a point θ uniformly from A and output θ with probability F ( θ ) / ( e max x ∈ A F ( x )) ; otherwise restart.
3. For κ ≥ 1 /α , proceed as follows:
4. (a) Choose a starting point x 0 ∈ Ω arbitrarily.
5. (b) Define a random walk on the centers of the small cubes as follows:
- i. At a state (cube center) x , stay at x with probability 1/2.
- ii. Otherwise (with probability 1/2), choose a direction u ∈ {± e 1 , · · · , ± e d } uniformly at random (each chosen with probability 1 / 2 d ), where e i is the standard basis vector in the i -th coordinate.
- iii. If the adjacent cube in that direction is not in A , stay at x .
- iv. Otherwise, move to the center y of that adjacent cube with probability min { 1 , F ( y ) /F ( x ) } ; with probability 1 -min { 1 , F ( y ) /F ( x ) } , remain at x .
10. (c) Run this random walk for T steps. Let x be the final state.
11. (d) Pick a point θ uniformly from the cube C x .
12. (e) Output θ with probability F ( θ ) / ( eF ( x )) ; otherwise, restart from step 3(a) with a new recursive call.

For implementation details, we refer to the original paper [3]. In the subsections that follow, we analyze how this algorithm behaves when the function F can only be evaluated with some bounded error, a common scenario in practical applications.

## B.3.2 Conductance Bound with Function Evaluation Errors

The conductance of a Markov chain measures how well the chain mixes, specifically how quickly it converges to its stationary distribution. For a Markov chain with state space Ω , transition matrix P

and stationary distribution q , the conductance φ is defined as:

<!-- formula-not-decoded -->

Higher conductance implies faster mixing, while lower conductance suggests the presence of bottlenecks in the state space.

We now analyze how small errors in function evaluation affect the conductance of the Markov chain used in log-concave sampling, described in Section B.3.1. This analysis is central to understanding the robustness of sampling algorithms in the presence of approximation errors.

Lemma B.9 (Re-statement of Lemma 3.6) . Let P be the transition matrix of the original Markov chain in the grid-walk algorithm of Section B.3.1 based on function f , with state space Ω and conductance φ . Let P ′ be the transition matrix of the perturbed chain based on f ′ where f ′ ( θ ) = f ( θ ) + ζ ( θ ) with | ζ ( θ ) | ≤ ζ for all θ ∈ Ω , where ζ ( · ) is an arbitrary bounded error function and ζ &gt; 0 is an upper bound on its magnitude. Then the conductance φ ′ of the perturbed chain satisfies:

<!-- formula-not-decoded -->

Proof. Fix any subset S of the state space. The conductance of S in the original chain is:

<!-- formula-not-decoded -->

where q is the stationary distribution and P xy are the transition probabilities.

̸

In the grid-walk algorithm, we know P xy = 0 if x = y are not adjacent; for adjacent points x and y :

<!-- formula-not-decoded -->

̸

and remarkably P xx = 1 -∑ x = y P xy .

For the perturbed chain with adjacent x, y :

<!-- formula-not-decoded -->

Since f ′ ( y ) -f ′ ( x ) = f ( y ) -f ( x ) + ( ζ ( y ) -ζ ( x )) and | ζ ( y ) -ζ ( x ) | ≤ 2 ζ , we have:

<!-- formula-not-decoded -->

This implies:

<!-- formula-not-decoded -->

Therefore:

<!-- formula-not-decoded -->

The stationary distributions q and q ′ satisfy:

<!-- formula-not-decoded -->

Since F ′ ( x ) = e -f ′ ( x ) = e -( f ( x )+ ζ ( x )) = e -f ( x ) e -ζ ( x ) = F ( x ) e -ζ ( x ) , we have:

<!-- formula-not-decoded -->

The normalization ratio satisfies:

<!-- formula-not-decoded -->

Therefore:

And:

Therefore:

<!-- formula-not-decoded -->

Since φ = min S φ S and φ ′ = min S φ ′ S , we have:

<!-- formula-not-decoded -->

## B.3.3 Relative Distance Bound Between F and F ′

We now analyze how function evaluation errors affect the distributional distance between the original and perturbed stationary distributions.

For distributions, we define the L ∞ distance (or log-ratio distance) between distributions µ and ν on A as:

<!-- formula-not-decoded -->

Lemma B.10 (Re-statement of Lemma 3.7) . Let F ( θ ) = e -f ( θ ) and F ′ ( θ ) = e -f ′ ( θ ) where f ′ ( θ ) = f ( θ ) + ζ ( θ ) with | ζ ( θ ) | ≤ ζ for all θ ∈ A . Then the relative distance between F and F ′ is bounded by:

<!-- formula-not-decoded -->

Furthermore, if we define the distributions π ( θ ) ∝ F ( θ ) and π ′ ( θ ) ∝ F ′ ( θ ) , then the infinity-distance between them is bounded by:

<!-- formula-not-decoded -->

Proof. For any θ ∈ A , we have:

<!-- formula-not-decoded -->

Since | ζ ( θ ) | ≤ ζ , we have:

<!-- formula-not-decoded -->

Using the bounds on transition probabilities and stationary distributions:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore:

This gives:

<!-- formula-not-decoded -->

Since e -ζ ≤ e -ζ ( z ) ≤ e ζ for all z ∈ A , we have:

<!-- formula-not-decoded -->

Therefore:

<!-- formula-not-decoded -->

Thus, the L ∞ -distance between π ′ and π is bounded by:

<!-- formula-not-decoded -->

## B.3.4 Mixing Time Analysis and Implementation Details

For a Markov chain with state space Ω , transition matrix P , and stationary distribution π , the mixing time t mix ( ϵ ) with respect to the L ∞ -distance is defined as:

<!-- formula-not-decoded -->

for any ϵ ≥ 0 . For efficient implementation of the grid-walk algorithm, we utilize the results of [9]. Following their approach, we can determine the number of steps required for L ∞ convergence using:

Lemma B.11 (Mixing time for relative L ∞ convergence [41]) . Let P be a lazy, time-reversible Markov chain over a finite state space Γ with stationary distribution π . Then, the mixing time of P w.r.t. L ∞ distance is at most

<!-- formula-not-decoded -->

where φ ( x ) = inf { φ S : π ( S ) ≤ x } , φ S denotes the conductance of the set S ⊆ Γ , and π ∗ = min x ∈ Γ π ( x ) is the minimum probability assigned by the stationary distribution.

We now provide a bound on how function evaluation errors affect the mixing time.

Lemma B.12 (Re-statement of Lemma 3.8) . The mixing time t ′ mix ( ϵ ) of the perturbed chain to achieve L ∞ -distance ϵ to its stationary distribution satisfies:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the normalized distributions, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. For a log-concave function F ( x ) = e -f ( x ) where f is α -Lipschitz, we set the grid spacing parameter γ = ϵ 2 α √ d . Using the conductance bound from previous analysis [9], we can derive the lower bound on conductance:

<!-- formula-not-decoded -->

By Lemma 3.6,the conductance φ ′ for the perturbed chain F ′ satisfies

<!-- formula-not-decoded -->

By the Lipschitz assumption, we have that

<!-- formula-not-decoded -->

Hence, by the lower bounds of conductance and minimum probability in the state space and Lemma B.11, we complete the proof.

Building upon our analysis of how function evaluation errors affect conductance, mixing time, and distributional distance, we now develop an efficient algorithm for sampling from log-concave distributions in the presence of such errors. Our approach builds upon the framework developed by Bassily, Smith, and Thakurta [9], extending it to handle approximation errors with formal guarantees.

Theorem B.13 (BST14-Based Implementation) . Let C ⊂ R d be a convex set and f : C → R be a convex, L -Lipschitz function. There exists an efficient algorithm that, when given exact function evaluations, outputs a sample θ ∈ C from a distribution µ such that the relative distance between µ and the target log-concave distribution π ( θ ) ∝ e -f ( θ ) can be made arbitrarily small, i.e., Dist ∞ ( µ, π ) ≤ ξ for any desired ξ &gt; 0 . This algorithm runs in time O ( d 3 · poly ( L, ∥ C ∥ 2 , 1 /ξ )) , which is polynomial in the dimension d , the diameter of C , the Lipschitz constant L , and the accuracy parameter 1 /ξ .

The key techniques in this implementation include:

1. Extending the function f beyond the convex set C to a surrounding cube A
2. Using a gauge penalty function to reduce the probability of sampling outside C
3. Implementing an efficient grid-walk algorithm to sample from the resulting distribution

We now formally incorporate the effect of function evaluation errors into this framework:

Theorem B.14 (Re-statement of Theorem 3.9) . Let C ⊂ R d be a convex set and f : C → R be a convex, L -Lipschitz function. Suppose we have access to an approximate function evaluator that returns f ′ ( θ ) = f ( θ )+ ζ ( θ ) where | ζ ( θ ) | ≤ ζ for all θ ∈ C , and ζ = O (1) is a constant independent of dimension. There exists an efficient algorithm that outputs a sample θ ∈ C from a distribution µ ′ such that:

<!-- formula-not-decoded -->

where π ( θ ) ∝ e -f ( θ ) is the target log-concave distribution and δ &gt; 0 is an arbitrarily small constant.

This algorithm runs in time O ( e 12 ζ · d 3 · poly ( L, ∥ C ∥ 2 , 1 /ξ )) . When ζ = O (1) is a constant, this remains O ( d 3 · poly ( L, ∥ C ∥ 2 , 1 /ξ )) with the same asymptotic complexity as the exact evaluation algorithm, differing only by a constant factor e 12 ζ in the running time.

Proof. We follow the approach of [9] with appropriate modifications to account for function evaluation errors:

1. Enclose the convex set C in an isotropic cube A with edge length τ = ∥ C ∥ ∞ .

2. Construct a convex Lipschitz extension f of the function f over A using:

<!-- formula-not-decoded -->

This extension preserves the Lipschitz constant L and the convexity of f .

3. Define a gauge penalty function using the Minkowski functional of C :

<!-- formula-not-decoded -->

where ψ ( θ ) := inf { r &gt; 0 : θ ∈ rC } is the Minkowski norm of θ with respect to C , and α is a parameter set to ensure correct sampling properties.

4. Define the target sampling distribution:

<!-- formula-not-decoded -->

5. In the presence of function evaluation errors, for θ ∈ A , the algorithm samples from:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

6. By Lemma 3.7 on the relative distance between distributions, we have:

<!-- formula-not-decoded -->

7. For the sampling algorithm's computational efficiency, we note that by Corollary 3.8, the mixing time increases by a factor of e 12 ζ . and the modified algorithm's running time becomes O ( e 12 ζ · d 3 · poly ( L, ∥ C ∥ 2 , 1 /ξ )) .

If ζ is a constant, then the factor e 12 ζ is also a constant. Therefore, the algorithm maintains the same asymptotic polynomial complexity in d as the exact evaluation algorithm, with only the leading constant factor affected by the approximation error.

Theorem B.15 (Exponential Mechanism Implementation) . Under Assumptions 2.1, for any constants ε = O (1) , there is an efficient sampler to solve DP-bilevel ERM with the following guarantees:

- The scheme is ( ε, 0) -DP;
- The expected loss is bounded by ˜ O ( d x εn [ L f,x D x + L f,y D y + L f,y L g,y µ g ]) ;
- The running time is O ( d 6 n · poly( L, D x , 1 /ε, log( dL 2 f,y /µ g )) ∧ d 4 n · L g,y µ g · poly( L, D x , 1 /ε ) ) .

Proof. Privacy: Let Z and Z ′ be adjacent data sets. Consider the exponential mechanism and the probability density π Z proportional to exp( -ε ′ 2 s ̂ Φ Z ( x )) . We set ζ = ξ = ε ′ / 6 . Let the π ′ Z be the probability density of the final output of the sampler. Then by Theorem B.14, we know

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and setting ε ′ = ε/ 2 completes the proof of the privacy guarantee.

Excess risk: The excess risk bound follows from Theorem 3.2 and the assumption that ε = O (1) .

By Theorem 3.2, we have

Hence we know

Time complexity: Given a function value query of ̂ Φ Z ( x ) , we need to return a value of error at most ζ . By the Lipschitz of f ( x, · , z ) , it suffices to find a point y such that

<!-- formula-not-decoded -->

By the strong convexity of g ( x, · , z ) , there are multiple ways to find the qualified y . In our case, we can simply apply the cutting plane method [32], which can be implemented in O ( d 3 n poly(log( dL 2 f,y /ζµ g )) . Alternatively, we could apply the subgradient method to ̂ G Z ( x, · ) , which can be implemented in O ( dn ( L g,y µ g ζ )) . Combining the query complexity in Theorem B.14 gives the total running time complexity.

With Theorem B.8 and a similar argument on the implementation, we can get the following result of the Regularized Exponential Mechanism.

Theorem B.16 (Regularized Exponential Mechanism Implementation) . Grant Assumptions 2.1 and additionally assume ∥∇ x f ( x, y, z ) - ∇ x f ( x, y ′ , z ) ∥ ≤ β f,xy ∥ y -y ′ ∥ and ∥∇ y ̂ G Z ( x, y ) -∇ y ̂ G Z ( x ′ , y ) ∥ ≤ β g,xy ∥ x -x ′ ∥ for all x, x ′ , y, y ′ , z . Given ε = O (1) and 0 &lt; δ &lt; 1 / 10 , there is an efficient sampler to implement the Regularized Exponential Mechanism and solve DP-bilevel ERM with the following guarantees:

- The scheme is ( ε, δ ) -DP;
- The expected empirical loss is bounded by O ( ( L f,x + L f,y β g,xy µ g + L g,y β f,xy µ g ) D x √ d x log(1 /δ ) n ) .

<!-- formula-not-decoded -->

With a different parameter setting, we can get the ( ε, δ ) -DP sampler with the same running time and achieve the expected population loss as O ( ( L f,x + L f,y β g,xy µ g + L g,y β f,xy µ g ) D x ( √ d x log(1 /δ ) n + 1 √ n )) .

## B.4 Excess risk lower bounds

Theorem B.17 (Re-statement of Theorem 3.10) . 1. Let A be ε -DP. Then, there exists a data set Z ∈ Z n and a convex bilevel ERM problem instance satisfying Assumptions 2.1 and 2.2 with µ g = Θ( L g,y /D y ) such that

<!-- formula-not-decoded -->

2. Let A be ( ε, δ ) -DP with 2 -Ω( n ) ≤ δ ≤ 1 /n 1+Ω(1) . Then, there exists a data set Z ∈ Z n and a convex bilevel ERM problem instance satisfying Assumptions 2.1 and 2.2 with µ g = Θ( L g,y /D y ) such that

<!-- formula-not-decoded -->

Proof. Case 1: Suppose L f,x D x ≲ L f,y D y . Then we will show ̂ Φ Z ( A ( Z )) -̂ Φ ∗ Z = Ω ( ( L f,y D y ) min { 1 , d nε }) with probability at least 1 / 2 for pure ε -DP A and ̂ Φ Z ( A ( Z )) -̂ Φ ∗ Z = Ω ( ( L f,y D y ) min { 1 , √ d nε }) with probability at least 1 / 3 for ( ε, δ ) -DP A .

Let f ( x, y, z ) = -⟨ y, z ⟩ , which is convex and 1 -Lipschitz in x and y if X = Y = B are unit balls in R d , d = d x = d y , and Z = {± 1 / √ d } d . Let g ( x, y, z ) = 1 2 ∥ y -ζx ∥ 2 for ζ &gt; 0 to be chosen later. Note ̂ F Z ( x, y ) = -⟨ y, Z ⟩ , where Z = 1 n ∑ n i =1 z i , ̂ y ∗ Z ( x ) = ζx , and ̂ Φ Z ( x ) = ̂ F Z ( x, ̂ y ∗ Z ( x )) = ⟨-ζx, Z ⟩ = ⇒ ̂ x ∗ ( Z ) = argmin x ∈X ̂ Φ Z ( x ) = Z ∥ Z ∥ . Therefore, for any x ∈ X , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

since ∥ x ∥ , ∥ ̂ x ∗ ( Z ) ∥ ≤ 1 . Now, recall the following result, which is due to [9, Lemma 5.1] and [48, Theorem 1.1]:

Lemma B.18 (Lower bounds for 1-way marginals) . Let n, d ≥ 1 , ε &gt; 0 , 2 -Ω( n ) ≤ δ ≤ 1 /n 1+Ω(1) .

1. ε -DP algorithms: There is a number M = Ω(min( n, d/ε )) such that for every ε -DP A , there is a data set Z = ( z 1 , . . . , z n ) ⊂ {± 1 / √ d } d with ∥ Z ∥ ∈ [( M -1) /n, ( M +1) /n ] such that, with probability at least 1 / 2 over the algorithm random coins, we have

<!-- formula-not-decoded -->

2. ( ε, δ ) -DP algorithms: There is a number M = Ω(min( n, √ d log(1 /δ ) /ε )) such that for every ( ε, δ ) -DP A , there is a data set Z = ( z 1 , . . . , z n ) ⊂ {± 1 / √ d } d with ∥ Z ∥ ∈ [( M -1) /n, ( M +1) /n ] such that, with probability at least 1 / 3 over the algorithm random coins, we have

<!-- formula-not-decoded -->

We claim there exists Z ∈ Z n with ∥ Z ∥ ∈ [( M -1) /n, ( M +1) /n ] such that

<!-- formula-not-decoded -->

with probability at least 1 / 2 . Suppose for the sake of contradiction that ∀ Z ∈ Z n with ∥ Z ∥ ∈ [( M -1) /n, ( M +1) /n ] , we have

<!-- formula-not-decoded -->

with probability at least 1 / 2 . Let c ∈ [ -1 /n, 1 /n ] such that ∥ Z ∥ = M/n + c . Then for the ε -DP algorithm ˜ A ( Z ) := M n A ( Z ) , we have

<!-- formula-not-decoded -->

which implies ∥ ˜ A ( Z ) -Z ∥ ≪ 1 ∧ d εn , contradicting Lemma B.18. By combining the claim (25) with inequality (24), we conclude that if x = A ( Z ) is ε -DP, then

<!-- formula-not-decoded -->

Next, we scale our hard instance to obtain the ε -DP lower bound. Define the scaled parameter domains ˜ X = D x B , ˜ Y = D y B , ˜ Z = Z = {± 1 / √ d } d , and denote ˜ x = D x x, ˜ y = D y y for any x, y ∈ X × Y = B 2 . Define ˜ f : ˜ X × ˜ Y × ˜ Z → R by

<!-- formula-not-decoded -->

which is convex and L f,y -Lipschitz in y for any permissible ˜ x, ˜ z . Define ˜ g : ˜ X × ˜ Y × ˜ Z → R by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then ˜ g is µ g -strongly convex in y and 2 L g,y -Lipschitz, since L g,y ≥ µ g D y . Now,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where and

Also,

<!-- formula-not-decoded -->

Thus, for any ε -DP A , there exists a dataset Z = ˜ Z such that the following holds with probability at least 1 / 2 , where we denote ˜ x = A ( ˜ Z ) :

<!-- formula-not-decoded -->

The argument for the ( ε, δ ) -DP case is identical to the above, except we invoke part 2 of Lemma B.18 instead of part 1.

Finally, it is easy to verify that Assumptions 2.1 and 2.2 are satisfied, with β g,xy ≤ µ g D y D x , C g,xy = C g,yy = M g,yy = M g,xy = 0 = β f,xx = β f,xy = β f,yy .

Case 2: L f,y D y ≲ L f,x D x . In this case, the desired lower bounds follow from a trivial reduction to the single-level DP ERM lower bounds of [9, Theorems 5.2 and 5.3]: take Y = { y 0 } for some y 0 ∈ R d with ∥ y 0 ∥ ≤ D y , X = D x B , Z = {± 1 / √ d } d , and let f ( x, y, z ) = -L f,x ⟨ x, z ⟩ and g ( x, y, z ) = µ g 2 ∥ y ∥ 2 . Then f and g satisfy Assumption 2.1, ̂ y ∗ Z ( x ) = y 0 , ̂ F Z ( x ) = ̂ Φ Z ( x ) = -L f,x ⟨ x, Z ⟩ . Thus, the lower bounds on the excess risk ̂ F Z ( x ) -̂ F ∗ Z for DP x given in [9, Theorems 5.2 and 5.3] apply verbatim to the excess risk ̂ Φ Z ( x ) -̂ Φ ∗ Z . This completes the proof.

## C Proofs for Section 4

## C.1 An iterative second-order method

We have the following key lemma, which will be needed for proving Theorem 4.2.

Lemma C.1 (Re-statement of Lemma 4.1) . For any fixed x t , define the query q t : Z n → R d ,

<!-- formula-not-decoded -->

where y t +1 = y t +1 ( Z ) is given in Algorithm 1. If α ≤ K Cn where C and K are defined in Equations (11) and (12) , then the ℓ 2 -sensitivity of q t is upper bounded by 4 K n .

Proof. We will need the following bound due to [23, Lemma 2.2]: for any x, y ∈ X × Y , for

<!-- formula-not-decoded -->

Now, denoting y t +1 = y t +1 ( Z ) and y ′ t +1 = y t +1 ( Z ′ ) , the sensitivity of the the query q t is bounded by

<!-- formula-not-decoded -->

where we used the bound (26) and our choice of α , for K defined in the theorem statement. Next, we claim

<!-- formula-not-decoded -->

This will follow from a rather long calculation that uses Assumption 2.2 repeatedly, along with the perturbation inequality ∥ M -1 -N -1 ∥ ≤ ∥ M -1 ∥∥ M -N ∥∥ N -1 ∥ which holds for any invertible matrices M and N . Let us now prove the bound (27). In what follows, the notation ∇ denote the derivative of the function w.r.t. x (accounting for the dependence of the function on ̂ y ∗ Z ( x ) via the chain rule) and denote

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

̸

where we assumed WLOG that z 1 = z ′ 1 and used the smoothness assumption in the last inequality above. Now, recall that

<!-- formula-not-decoded -->

and note that

<!-- formula-not-decoded -->

by the chain rule and ( β g,xy /µ g ) -Lipschitz continuity of ̂ y ∗ Z (see [23] for a proof of this result). Thus,

<!-- formula-not-decoded -->

Next, we bound

<!-- formula-not-decoded -->

It remains to bound

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in the second-to-last inequality we used the operator norm inequality

<!-- formula-not-decoded -->

which holds for any invertible matrices M and N of compatible shape.

Combining the above pieces completes the proof.

We have the following refinement of [23, Lemma 2.2c], in which we correctly describe the precise dependence on the smoothness, Lipschitz, and strong convexity parameters of f and g :

Lemma C.2 (Smoothness of ̂ Φ Z ) . Grant Assumptions 2.1 and 2.2. Then, for any x 1 , x 2 ,

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Proof. Recall that where

<!-- formula-not-decoded -->

Also, ̂ y ∗ Z is β g,xy µ g -Lipschitz (c.f. [23, Lemma 2.2b]). Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we used the operator norm inequality

<!-- formula-not-decoded -->

which holds for any invertible matrices M and N of compatible shape. Using the Lipschitz continuity of ̂ y ∗ Z one last time completes the proof.

Theorem C.3 (Precise version of Theorem 4.2) . Grant Assumptions 2.1 and 2.2. Set σ = 32 K √ T log(1 /δ ) /nε and

<!-- formula-not-decoded -->

for C defined in Equation (11) in Algorithm 1, where

<!-- formula-not-decoded -->

Then, Algorithm 1 is ( ε, δ ) -DP. Further, choosing η = 1 / 2 β Φ and T = ⌈ nε √ d x log(1 /δ ) √ β Φ ( ̂ Φ Z ( x 0 ) -̂ Φ ∗ Z ) K ⌉ for β Φ defined in Equation (28) , the output of Algorithm 1 satisfies

<!-- formula-not-decoded -->

Proof. Privacy: By Lemma 4.1, the ℓ 2 -sensitivity of ∇ ̂ F Z ( x t , y t +1 ) is upper bounded by 4 K/n . Thus, by the privacy guarantee of the gaussian mechanism and the advanced composition theorem, our prescribed choice of σ ensures that all T iterations of Algorithm 1 satisfy ( ε, δ ) -DP. Hence ̂ x T is ( ε, δ ) -DP by post-processing.

Utility: We will need the following descent lemma for gradient descent with biased, noisy gradient oracle:

Lemma C.4. [1, Lemma 2] Let H be β -smooth, x t +1 = x t -η ˜ ∇ H ( x t ) , where ˜ ∇ H ( x t ) = ∇ H ( x t ) + b t + N t is a biased, noisy gradient such that E [ N t | x t ] = 0 , ∥ E [ b t | x t ] ∥ ≤ B , and E [ ∥ N t ∥ 2 | x t ] ≤ Σ 2 . Then for any stepsize η ≤ 1 2 β , we have

<!-- formula-not-decoded -->

We will apply Lemma C.4 to H = ̂ Φ Z which is β Φ -smooth by Lemma C.2, ˜ ∇ H ( x t ) = ∇ ̂ F Z ( x t , y t +1 ) + u t with bias b t = ∇ ̂ F Z ( x t , y t +1 ) -∇ ̂ Φ Z ( x t ) and noise N t = u t :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any η ≤ 1 2 β Φ .

Now, [23, Lemma 2.2a] tells us that

<!-- formula-not-decoded -->

for C defined in Equation (11). Therefore,

<!-- formula-not-decoded -->

by our choice of α . Further,

<!-- formula-not-decoded -->

Plugging these values into (30) and choosing η = 1 / (2 β Φ ) , we obtain

<!-- formula-not-decoded -->

Plugging in the prescribed T from the theorem statement and then using Jensen's inequality completes the proof.

## C.2 'Warm starting' Algorithm 1 with the exponential mechanism

Algorithm 2: Warm-Start Meta-Algorithm for Bilevel ERM

- 1 Input: Data Z ∈ Z n , loss functions f and g , privacy parameters ( ε, δ ) , warm-start DP-ERM algorithm A , DP-ERM stationary point finder B ;
- 2 Run ( ε/ 2 , δ/ 2) -DP A on ̂ Φ Z ( · ) to obtain x 0 ;
- 3 Run ( ε/ 2 , δ/ 2) -DP B on ̂ Φ Z ( · ) with initialization x 0 to obtain x priv ;
- 4 Return: x priv .

We instantiate this framework by choosing A as the exponential mechanism (4) and B as Algorithm 1 to obtain the following result:

Theorem C.5 (Re-statement of Theorem 4.3) . Grant Assumptions 2.1 and 2.2. Assume that there is a compact set X ⊂ R d x of diameter D x containing an approximate global minimizer ̂ x such that ̂ Φ Z ( ̂ x ) -̂ Φ ∗ Z ≤ Ψ d εn , where

<!-- formula-not-decoded -->

Then, there exists an ( ε, δ ) -DP instantiation of Algorithm 2 with output satisfying

<!-- formula-not-decoded -->

Proof. Privacy: This is immediate from basic composition, since A is ε/ 2 -DP and B is ( ε/ 2 , δ/ 2) -DP.

Utility: First, note that the output x 0 of the exponential mechanism in (4) satisfies

<!-- formula-not-decoded -->

with probability ≥ 1 -ζ for any ζ &gt; 0 that is polynomial in all problem parameters, by [19, Theorem 3.11]. Let us say x 0 is good if the above excess risk bound holds. Now, by Theorem 4.2, the output of Algorithm 1 satisfies

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

Now, since ̂ Φ Z ( x ) -̂ Φ ∗ Z ≤ LD x for any x ∈ X , the law of total expectation implies

<!-- formula-not-decoded -->

where the final inequality follows by choosing ζ sufficiently small.

## C.3 Deducing the upper bound in (3) .

We prove in Lemma B.6 that sup Z ∼ Z ′ ,x ∥∇ ̂ Φ Z ( x ) -∇ ̂ Φ Z ′ ( x ) ∥ ≤ 2 G n , where G is defined in (5). Thus, by similar arguments used to prove the results in Section 3.1, one can show that sampling ̂ x proportional to the following density is ε -DP:

<!-- formula-not-decoded -->

Moreover, the output of this sampler satisfies

<!-- formula-not-decoded -->

Further, outputting arbitrary x 0 ∈ X trivially achieves ∥∇ ̂ Φ Z ( x 0 ) ∥ ≤ L f,x with 0 -DP. By combining these upper bounds with our results in Theorems 4.2 and 4.3, we deduce the novel state-of-the-art upper bound in (3) for DP nonconvex bilevel ERM (with constant problem parameters).

## D Limitations

While our work provides near-optimal rates and efficient algorithms for differentially private bilevel optimization (DP BLO), several limitations remain that should be considered when interpreting our theoretical and practical contributions.

Assumptions on Problem Structure. Our results rely on several assumptions that may not hold in all practical settings. For convex DP BLO, we assume that the lower-level problem is strongly convex and that the loss functions are Lipschitz continuous with bounded gradients (and, for some of our algorithms, bounded and/or Lipschitz Hessians). These structural assumptions are standard in bilevel optimization theory but may not accurately capture real-world scenarios where lower-level problems are ill-conditioned, non-convex, or lack smoothness. Violations of these assumptions could degrade both utility and privacy guarantees, as our sensitivity and excess risk bounds depend critically on these properties.

Scalability and Computational Efficiency. Although most of our algorithms are polynomialtime, they may still incur significant computational costs, especially in high-dimensional settings. Our efficient implementations rely on sampling techniques (e.g., grid-walk) whose runtime scales polynomially with the dimension. This may limit practicality on large-scale or high-dimensional problems. Additionally, the warm-start algorithm for nonconvex DP BLO is inefficient. We leave it for future work to develop algorithms with improved computational complexity guarantees.

Lack of Empirical Validation. This paper focuses on theoretical analysis and does not include experimental results. While our theoretical rates are nearly optimal, empirical performance can depend on implementation details, constant factors, and practical optimization challenges not captured in our analysis. We defer empirical validation, including runtime measurements and real-data utility evaluation, to future work.

## E Broader Impacts

This work advances algorithms for protecting the privacy of individuals whose data is used in bilevel learning applications, such as meta-learning and hyperparameter tuning. Privacy protection is widely regarded as a societal good and is enshrined as a fundamental right in many legal systems. By improving our theoretical understanding of privacy-preserving bilevel optimization, this work contributes to the development of machine learning methods that respect individual privacy.

However, there are trade-offs inherent in the use of differentially private (DP) methods. Privacy guarantees typically come at the cost of reduced model utility, which may lead to less accurate predictions or suboptimal decisions. For example, if a differentially private bilevel model is deployed in a sensitive application-such as medical treatment planning or environmental risk assessment-reduced accuracy could lead to unintended negative outcomes. While these risks are not unique to bilevel learning, they highlight the importance of transparency when communicating the limitations of DP models to stakeholders and decision-makers.

We also note that the performance of bilevel optimization algorithms depends on problem-specific factors such as the conditioning of the lower-level problem, the smoothness of the loss functions, and the dimensionality of the parameter spaces. Practitioners should carefully evaluate these factors when applying our methods in practice.

Finally, while this work focuses on theoretical developments and does not include empirical evaluation or deployment, we believe that the dissemination of privacy-preserving algorithms-alongside clear communication of their trade-offs-ultimately serves the public interest by empowering researchers and practitioners to build more responsible and privacy-aware machine learning systems.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: All claims are rigorously proved.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.

- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: See Appendix D.

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

Justification: All theorems are carefully proved in the Appendices with all assumptions clearly stated.

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

Justification: No experiments-this is a theoretical work.

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

Justification: Paper does not include experiments requiring code.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not

including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

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

Justification: Research conforms to the Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: See Appendix E.

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

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

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

- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.