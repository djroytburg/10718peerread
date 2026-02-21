## Local Curvature Descent: Squeezing More Curvature out of Standard and Polyak GD

Peter Richtárik 1 , 2

peter.richtarik@kaust.edu.sa

Dymitr Lubczyk 1 , 4

dymitr.lubczyk@student.uva.nl

Simone Maria Giancola 1 , 3

simone.giancola@kaust.edu.sa

Robin Yadav 1 , 5

robiny12@student.ubc.ca

1 KAUST, Saudi Arabia; 2 AI Initiative, Saudi Arabia; 3 Laboratoire de Mathématiques d'Orsay, Université Paris-Saclay, CNRS, France;

4 University of Amsterdam, Netherlands; 5 University of British Columbia, Canada

## Abstract

We contribute to the growing body of knowledge on more powerful and adaptive stepsizes for convex optimization, empowered by local curvature information . We do not go the route of fully-fledged second-order methods which require the expensive computation of the Hessian. Instead, our key observation is that, for some problems (e.g., when minimizing the sum of squares of absolutely convex functions), certain local curvature information is readily available, and can be used to obtain surprisingly powerful matrix-valued stepsizes, and meaningful theory. In particular, we develop three new methodsLCD1 , LCD2 and LCD3 -where the abbreviation stands for local curvature descent . While LCD1 generalizes gradient descent with fixed stepsize, LCD2 generalizes gradient descent with Polyak stepsize. Our methods enhance these classical gradient descent baselines with local curvature information, and our theory recovers the known rates in the special case when no curvature information is used. Our last method, LCD3 , is a variable-metric version of LCD2 ; this feature leads to a closed-form expression for the iterates. Our empirical results are encouraging, and show that the local curvature descent improves upon gradient descent.

## 1 Introduction

In this work we revisit the standard optimization problem

<!-- formula-not-decoded -->

where f : R d → R is a continuous convex function with a nonempty set of minimizers X ⋆ . Further, we denote the optimal function value by f ⋆ := f ( x ⋆ ) , where x ⋆ ∈ X ⋆ .

## 1.1 First-order methods

First-order methods of the Gradient Descent ( GD ) and Stochastic Gradient Descent ( SGD ) variety have been widely adopted to solve problems of type (1) [Polyak, 1963, Robbins and Monro, 1951]. Due to their simplicity and relatively low computational cost, these methods have seen great success across many machine learning applications, and beyond. Nonetheless, GD , performing iterations of the form

<!-- formula-not-decoded -->

where γ k &gt; 0 is a learning rate (stepsize), suffers from several well-known drawbacks. For example, for convex and L -smooth objectives, GD converges provided that 1

<!-- formula-not-decoded -->

[Nesterov, 2004]. For many problems, L is very large and/or unknown, and estimating its value is a non-trivial task. Overestimation of the smoothness constant leads to unnecessarily small stepsizes, which degrades performance, both in theory and in practice.

Polyak stepsize. When the optimal value f ⋆ is known, a very elegant solution to the above-mentioned problems was provided by Polyak [1987], who proposed the use of what is now known as the Polyak stepsize:

<!-- formula-not-decoded -->

It is known that if f is convex and L -smooth, then γ k ≥ 1 2 L for all k ≥ 0 . So, unlike strategies based on the recommendation provided by (3), Polyak stepsize can never be too small compared to the upper bound from (3). In fact, it is possible for γ k to be larger than 1 L , which leads to practical benefits. Moreover, this is achieved without having to know or estimate L , which is a big advantage. Since the function value f ( x k ) and the gradient ∇ f ( x k ) are typically known, the only price for these benefits is the knowledge of the optimal value f ⋆ . This may or may not be a large price to pay, depending on the application.

Malitsky-Mishchenko stepsize. In the case of convex and locally smooth objectives, Malitsky and Mishchenko [2020] recently proposed an ingenious adaptive stepsize rule that iteratively builds an estimate of the inverse local smoothness constant from the information provided by the sequence of iterates and gradients. Furthermore, they prove their methods achieve the same or better rate of convergence as GD , without the need to assume global smoothness. For a review of further approaches to adaptivity, we refer the reader to Malitsky and Mishchenko [2020], and for several extensions of this line of work, we refer to Zhou et al. [2024].

Adaptive stepsizes in deep learning. When training neural networks and other machine learning models, issues related to the appropriate selection of stepsizes are amplified even further. Optimization problems appearing in deep learning are not convex and may not even be L -smooth, or L is prohibitively large, and tuning the learning rate usually requires the use of schedulers or a costly grid search. In this domain, adaptive stepsizes have played a pivotal role in the success of first-order optimization algorithms. Adaptive methods such as Adam , RMSProp , AMSGrad , and Adagrad scale the stepsize at each iteration based on the gradients [Kingma and Ba, 2017, Hinton, 2014, Reddi et al., 2019, Duchi et al., 2011]. Although Adam has seen great success empirically when training deep learning models, there is very little theoretical understanding of why it works so well. On the other hand, Adagrad converges at the desired rate for smooth and Lipschitz objectives but is not as successful in practice as Adam [Duchi et al., 2011].

## 1.2 Second-order methods

When f is twice differentiable and L -smooth, L can be seen as a global upper bound on the largest eigenvalue of the Hessian of f . So, there are close connections between the way a learning rate should be set in GD -type methods and the curvature of f .

Newton's method. Perhaps the most well-known second-order algorithm is Newton's method:

<!-- formula-not-decoded -->

When it works, it converges in a few iterations only. However, it may fail to converge even on convex objectives 2 . It needs to be modified in order to converge from any starting point, say by adding a damping factor [Hanzely et al., 2022] or regularization [Mishchenko, 2023]. However, under suitable assumptions, Newton's method converges quadratically when started close enough to the solution. The key difficulty in performing a Newton's step is the computation of the Hessian and performing a linear solve. In analogy with (2), it is possible to think of ( ∇ 2 f ( x k ) ) -1 as a matrix-valued stepsize.

1 It is possible to use slightly larger stepsizes, by at most a factor of 2, but this is not relevant to our narrative.

2 A well-known example is the function f ( x ) = ln( e -x + e x ) with x 0 = 1 . 1 .

Quasi-Newton methods. To reduce the computational cost, quasi-Newton methods such as LBFGS utilize an approximation of the inverse Hessian that can be computed from gradients and iterates only, typically using the approximation ∇ 2 f ( x k +1 )( x k +1 -x k ) ≈ ∇ f ( x k +1 ) -∇ f ( x k ) , which makes sense under appropriate assumptions when ∥ x k +1 -x k ∥ is small Nocedal and Wright [2006], Al-Baali et al. [2014], Al-Baali and Khalfan [2007], Dennis and Moré [1977]. Until very recently, quasi-Newton methods were merely efficient heuristics, with very weak theory beyond quadratics [Kovalev et al., 2021, Rodomanov and Nesterov, 2021]. Furthermore, recent work has shown that the Hessian of deep neural networks exhibits a block diagonal structure [Dong et al., 2025]. A deeper understanding of these diagonal patterns can facilitate the design of more efficient adaptive algorithms for deep networks.

Polyak stepsize with second-order information. Li et al. [2022] recently proposed extensions of the Polyak stepsize, named SP2 and SP2+ , that incorporate second-order information. SP2 can also be derived similarly to the Polyak stepsize. While SP2 can be utilized in the non-convex stochastic setting, it only has convergence theory for quadratic functions and can often be very unstable in practice. Furthermore, the quadratic constraint defined for SP2 does not guarantee that iterates move closer to the set of minimizers. Instead, we propose an assumption similar to earlier works Karimireddy et al. [2018], Gower et al. [2019], with the aim of using second-order information rigorously.

## 1.3 Notation

All vectors are in R d unless explicitly stated otherwise. We use X ⋆ to denote the set of minimizes of f . Matrices are uppercase and bold (e.g., A , C ), the d × d zero (resp. identity) matrix is denoted by 0 (resp. I ), and S d + is the set of d × d positive semi-definite matrices. The standard Euclidean inner product is denoted with ⟨· , ·⟩ . For A ∈ S d + , we let ∥ x ∥ 2 A := ⟨ A x, x ⟩ . By ∥ x ∥ p := ( ∑ d i =1 | x i | p ) 1 /p we denote the L p norm in R d . The Löwner order for positive semi-definite matrices is denoted with ⪯ .

## 2 Summary of Contributions

In this work we contribute to the growing body of knowledge on more powerful and adaptive stepsizes, empowered by local curvature information . We do not go the route of fully-fledged second-order methods which require the expensive computation of the Hessian.

Instead, our key observation is that, for some problems, certain local curvature information is readily available, and can be used to obtain powerful matrix-valued stepsizes.

The examples mentioned above, and discussed in detail in Sections 6 and 7 lead to the following abstract assumption, which at the same time defines what we mean by the term local curvature :

Assumption 2.1 (Convexity and smoothness with local curvature) . There exists a curvature mapping/metric/matrix C : R d → S d + and a constant L C ≥ 0 such that the inequalities

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

hold for all x, y ∈ R d .

Assumption 2.1 defines a new class of functions. Note that with the specific choice C ( y ) ≡ 0 , (5) reduces to convexity, and (6) reduces to L -smoothness, with L = L C . Note that any function satisfying (5) is necessarily convex, and similarly, any L -smooth function satisfies (6) with any curvature mapping C and L C = L . However, the converse is not true: a function satisfying (6) is not necessarily L -smooth for any finite L . Further, note that if f is µ -strongly convex, then it satisfies (5) with curvature mapping C ( y ) ≡ µ I . The class of convex and L -smooth functions is one

of the most studied functional classes in optimization. Furthermore, we emphasize our new class is a strict and, as we shall see, useful generalization. We also show that many practical examples satisfy Assumption 2.1 with a diagonal curvature mapping, which is particularly relevant in machine learning applications as it enables efficient computation of the update steps for our algorithms.

Additionally, we want to stress that (5) generalizes convexity, which is a global property, and thus, (5) is also a global property, captured by the fact that we require it to hold for all x ∈ R d . However, the curvature matrix depends on y , which we will see shortly, refers to the current iterate in our methods. So, it depends on the 'locus' of the algorithm, i.e. the current iterate. This is why we use the terminology 'local'. This is not to be misunderstood or misinterpreted for 'local' reach. We elaborate on the connections between our assumptions and related notions in the literature in Appendix A. We now provide a brief overview of our theoretical and empirical contributions:

## 2.1 Local curvature and a new function class

We define a new function class, described by Assumption 2.1, extending the classical class of convex and L -smooth functions. Further, we show that there are problems which satisfy Assumption 2.1 with nontrivial and easy-to-compute curvature mapping C (see Section 6 and Section 7).

## 2.2 Three new algorithms

We propose three novel algorithms for solving problem (1) for function f satisfying Assumption 2.1: Local Curvature Descent 1 ( LCD1 ), Local Curvature Descent 2 ( LCD2 ) and Local Curvature Descent 3 ( LCD3 ). First, LCD1 generalizes GD with constant stepsize: one moves from point y to the point obtained by minimizing the upper bound (6) on f in x . Indeed, if C ( y ) ≡ 0 , this algorithmic design strategy leads to gradient descent with stepsize 1 /L , where L = L C . Second, LCD2 generalizes GD with Polyak stepsize: one moves from point y to the Euclidean projection of y onto the ellipsoid:

<!-- formula-not-decoded -->

Indeed, if C ( y ) ≡ 0 , this algorithmic design leads to GD with stepsize (4). Computing the projection involves finding the unique root of a scalar equation in variable, which can be executed efficiently. Third, LCD3 is obtained from LCD2 by replacing the Euclidean projection with the projection defined by the local curvature matrix C . The projection problem then has a closed-form solution.

## 2.3 Theory

We prove convergence theorems for LCD1 (Theorem 4.1) and LCD2 (Theorem 4.2), with the same O (1 /k ) worst case rate of GD with constant and Polyak stepsize, respectively. Previous work on preconditioned Polyak stepsize [Abdukhakimov et al., 2023] fails to provide convergence theory and uses matrix stepsizes based on heuristics. In contrast, LCD2 utilizes local curvature from Assumption 2.1, and enjoys strong convergence guarantees.

## 2.4 Experiments

We demonstrate superior empirical behavior of LCD2 over the GD with Polyak stepsize across several standard machine learning problems to which our theory applies. The presence of local curvature in our algorithms boosts their empirical performance when compared to their counterparts not taking advantage of local curvature.

## 3 Three Flavors of Local Curvature Descent

We now describe our methods.

## 3.1 Local Curvature Descent 1

Our first method, LCD1 is obtained by minimizing the upper bound from Assumption 2.1 where y = x k , and letting x k +1 be the minimizer:

<!-- formula-not-decoded -->

The derivation is routine; nevertheless, the detailed steps behind Equation ( LCD1 ) can be found in Appendix B.1. If C ( x ) ≡ 0 and we let L = L C , we recover GD with the constant stepsize γ k = 1 L . Note that just like GD , LCD1 is not adaptive to the smoothness parameter L C ; this parameter is needed to perform a step.

## 3.2 Local Curvature Descent 2

Given any y ∈ R d , let us define the localization set

<!-- formula-not-decoded -->

Due to (5), we have X ⋆ ⊂ L C ( y ) , which justifies the use of the word 'localization'. Furthermore, y ∈ X ⋆ if and only if y ∈ L C ( y ) . Therefore, L C ( x k ) separates R d in two regions: one containing X ⋆ , the other the current iterate y = x k . This allows us to design our second algorithm, LCD2 : we simply project the current iterate x k into the localization set L C ( x k ) , bringing it closer to the set of optimal points X ⋆ :

<!-- formula-not-decoded -->

It turns out that this projection problem has an implicit parametric solution of the form

<!-- formula-not-decoded -->

where β k &gt; 0 . Importantly, we show in Appendix C.1 that the structure of the problem is easy: the parameter 1 /β k is the unique root of a scalar equation, solvable efficiently. Moreover, if C ( x k ) is a rankone matrix or a multiple of I , a closed-form solution exists. We present the details in Appendix C.3.

When C ( x ) ≡ 0 , the localization set defined in (7) reduces to a half-space, and LCD2 becomes GD with Polyak stepsize. In general, LCD2 can be seen as a variant of GD with Polyak stepsize, enhanced with local curvature . The method no longer points in the negative gradient direction anymore, of course. We argue that one step of LCD2 improves on one step of GD with Polyak stepsize. Indeed, since L C ( x k ) ⊆ L 0 ( x k ) , with equality if and only if C ( x k ) = 0 , the point x k +1 obtained by LCD2 is closer to X ⋆ than what is achieved by a single step of GD with Polyak stepsize.

## 3.3 Local Curvature Descent 3

Our last method, LCD3 , was born out of the desire to remove the need for the univariate root-finding subroutine in order to execute the projection defining LCD2 . This can be achieved by projecting using the norm given by the local curvature matrix C ( x k ) instead:

<!-- formula-not-decoded -->

If C is invertible 3 , this projection problem admits the closed-form solution

<!-- formula-not-decoded -->

The full derivation of this fact can be found in Appendix D.1. Although LCD3 uses the same localization set as LCD2 , we do not provide any convergence theorem for this method. The variable metric nature of the projection makes it technically difficult to provide a meaningful analysis of this method. Nevertheless, we justify the introduction of LCD3 via its promising experimental behavior in Section 8 and Appendix G.

## 4 Convergence Rates

Having described the methods, this appears to be the right moment to present our main convergence results for LCD1 and LCD2 .

3 We assume this for simplicity only.

Theorem 4.1 (Convergence of LCD1 ) . Let Assumption 2.1 be satisfied. For all k ≥ 1 , the iterates of LCD1 satisfy

<!-- formula-not-decoded -->

Theorem 4.2 (Convergence of LCD2 ) . Let Assumption 2.1 be satisfied. For all k ≥ 1 , the iterates of LCD2 satisfy

<!-- formula-not-decoded -->

The proofs of these results can be found in Appendix B.2 and Appendix C.2, respectively. It is possible to derive linear convergence results under the assumption that C ( x ) ⪰ µ I for all x ∈ R d and some µ &gt; 0 ; however, we refrain from listing these for brevity reasons.

If C ( x ) ≡ 0 , and we let L = L C , these theorems recover the standard rates known for GD with the stepsize 1 /L and GD with Polyak stepsize, respectively. So, we generalize these earlier results. However, it is possible for a function to satisfy Assumption 2.1 and not be L -smooth. In this sense, our results extend the reach of the classical theorems beyond the class of convex and L -smooth functions. On the other hand, if f is convex and L -smooth, it may be possible that it satisfies Assumption 2.1 with some nonzero local curvature mapping C , in which case we can choose L C such that L C ≤ L . Indeed,

<!-- formula-not-decoded -->

where λ min ( · ) (resp. λ max ( · ) ) represents the smallest (resp. largest) eigenvalue of the argument, confirming L C ≤ L . However, it may be that L C ≪ L , in which case our result leads to improved complexity. Nevertheless, the main allure of our methods is their attractive empirical behavior.

Convex quadratics. For convex quadratics, Assumption 2.1 is satisfied with C ( x ) = ∇ 2 f ( x ) and L C = 0 . In this case, both LCD1 and LCD2 reduce to Newton's method, and converge in a single step. Moreover, Theorem 4.1 and Theorem 4.2 predict this one-step convergence behavior.

To validate our theoretical setting, we will show that functions satisfying Assumption 2.1 are easy to construct, well-behaved, and practically interesting.

## 5 Local Curvature Calculus

We now mention a couple basic properties of functions that satisfy Inequalities (5) and (6).

Lemma 5.1. Let α, β ∈ R with β ≥ 0 . Suppose functions f and g satisfy inequality (5) with curvature mappings C 1 and C 2 respectively. Then:

<!-- formula-not-decoded -->

satisfy Inequality (5) with curvature mappings C 1 , β C 1 , and C 1 + C 2 respectively.

The proof of the lemma can be found in Appendix E.1. A particularly useful instantiation of Lemma 5.1 is presented in the following corollary.

Corollary 5.1. If f satisfies (5) and g is convex, then h := f + g also satisfies (5) .

Corollary 5.1 enables us to derive a variety of examples of functions satisfying inequality (5) by summing convex functions with instances from our class. Moreover, we can also show that inequality (5) is preserved under pre-composition with linear functions. Additional results for functions satisfying Assumption 2.1 can be found in Appendix E.

## 6 Examples of Functions Satisfying Assumption 2.1

We first list three examples that satisfy both inequalities in Assumption 2.1. Firstly, observe that if a function is L -smooth, then it satisfies inequality (6) since C ( x ) is assumed to be a positive semi-definite matrix. We aim to find convex functions that satisfy our assumption in a non-trivial manner, i.e., C ( x ) ̸≡ 0 and C ( x ) ̸≡ µ I for some µ &gt; 0 .

Example 6.1 (Huber loss) . Let δ &gt; 0 and consider the Huber loss function h : R → R given by

<!-- formula-not-decoded -->

Then f = h 2 satisfies Assumption 2.1 with constant L C = 2 δ 2 and curvature mapping

<!-- formula-not-decoded -->

Example 6.1 is particularly interesting because C ( x ) + 2 δ 2 ≤ 3 δ 2 for any x ∈ R . By computing the second derivative of f , we can obtain the tightest L -smoothness constant; it is equal to 3 δ 2 . Therefore, the variable bound we derived is at least as good as the L -smoothness bound.

Example 6.2 (Squared p norm) . Let p ≥ 2 and define f : R d → R as f ( x ) = ∥ x ∥ p . Then f 2 satisfies Assumption 2.1 with either of the two curvature mappings,

<!-- formula-not-decoded -->

and constant L C = 2( p -1) .

Example 6.3 ( L p regression) . Suppose A ∈ R n × d and b ∈ R n . For p ≥ 2 , the function f ( x ) = ∥ A x -b ∥ 2 p , satisfies Assumption 2.1 as a precomposition of Example 6.2 with an affine function.

Therefore, linear regression in the squared L p norm satisfies our assumption. The L p regression problem has several applications in machine learning [Dasgupta et al., 2009, Musco et al., 2022, Yang et al., 2018]. This includes low-rank matrix approximation, sparse recovery, data clustering, and learning tasks [Adil et al., 2023]. In general, convex optimization in non-Euclidean geometries is a well-studied and important research direction. This motivates us to study L p norms further and understand how they can fit within our assumptions.

We can perform other simple modifications of L p norm that satisfy only inequality (5).

Example 6.4. Let p ≥ 2 . Then f ( x ) = ∥ x ∥ p p satisfies (5) with either of the curvature mappings

<!-- formula-not-decoded -->

We postpone comments to Appendix E.3. Using Corollary 5.1 and the above examples, we can construct regularized convex problems that satisfy our assumptions. For instance, we can add the square of an L p norm to the logistic loss function to obtain an objective function that satisfies (5), with the mapping from the regularizer. The objective function will be L -smooth, so it also satisfies inequality (6).

## 7 Absolutely Convex Functions

In addition to the examples from Section 6, we now introduce the class of absolutely convex functions, and the problem of minimizing the sum of squares of absolutely convex functions. In this setting, as we shall show, the curvature mapping C satisfying Inequality (5) is readily available. Absolutely convex functions are defined as follows.

Definition 7.1 (Absolute convexity) . A function ϕ : R d → R is absolutely convex if

<!-- formula-not-decoded -->

Above, ∇ ϕ ( y ) refers to a subgradient of ϕ at y . Geometrically, (8) means that linear approximations of ϕ are always above the graph of -ϕ in addition to being below the graph of ϕ (same as convexity),

<!-- formula-not-decoded -->

Thus, any absolutely convex function is necessarily convex and non-negative. A constant function is absolutely convex if and only if it is non-negative. A linear function is absolutely convex if and only if it is constant and non-negative. Moreover, the absolute value of any affine function is absolutely convex; that is, ϕ ( x ) = | ⟨ a, x ⟩ + b | is absolutely convex. We avoid stating basic calculus rules as in Lemma 5.1, and opt to present only one interesting property, and one notable example. Many others can be found in Appendix F.

Lemma 7.1. Absolutely convex functions have bounded subgradients.

Example 7.1. If p ≥ 1 , then ϕ ( x ) = ∥ x ∥ p is absolutely convex.

## 7.1 Minimizing the sum of squares of absolutely convex functions

To conclude, we present the derivation of the curvature mapping C for the sum of squares of absolutely convex functions. Consider the optimization problem

<!-- formula-not-decoded -->

where each ϕ i is absolutely convex and a solution, x ⋆ , is assumed to exist. Let f i := ϕ 2 i , so that ∇ f i ( x ) = 2 ϕ i ( x ) ∇ ϕ i ( x ) . The gradient of f is given by

<!-- formula-not-decoded -->

Since ϕ i is absolutely convex, f i is necessarily convex. Indeed, by squaring both sides of the defining inequality (8), we get that for all x, y ∈ R d ,

<!-- formula-not-decoded -->

Summing these inequalities across i and taking the average, we find that the curvature mapping can be set to,

<!-- formula-not-decoded -->

In Appendix G, we provide experiments on objective functions that are in this class.

## 8 Experiments

Figure 1: Logistic regression on a2a dataset with L 2 regularization.

<!-- image -->

Figure 2: Logistic regression on mushrooms dataset with L 2 regularization.

<!-- image -->

Through our experiments, we highlight the effectiveness of our methods on standard convex optimization problems, especially when employing diagonal curvature matrices. We use a MacBook Pro with Apple M1 chip annd 8GB of RAM. The datasets are from LibSVM [Chang and Lin, 2011]. First, let us focus on solving the binary classification problem:

<!-- formula-not-decoded -->

where a i ∈ R d and b i ∈ {-1 , 1 } are the data samples. The regularization weight λ is proportional to the L -smoothness constant of the logistic regression instance.

In the first experiment, we use L 2 regularization. Therefore, f is L -smooth and µ -strongly-convex, so C ( x ) ≡ µ I . This experiment demonstrates our LCD2 and LCD3 outperform standard adaptive methods for convex optimization, such as the Barzilai-Borwein [Barzilai and Borwein, 1988] and Malitsky-Mischenko [Malitsky and Mishchenko, 2020] step sizes. As mentioned previously, in this setting, LCD1 recovers GD and LCD2 has a closed-form solution coinciding with LCD3 .Figures 1-2 show that LCD2 consistently outperforms Polyak. As expected, the gap increases with λ because C ( x ) only stores information about the regularizer. Thus, increasing λ shrinks the localization set of LCD2 so its improvement over Polyak grows. Importantly, since LCD2 has a closed form solution, its cost-per-iteration is the same as Polyak.

In the next experiment, we use L 3 regularization. In Example 6.4 we propose two C ( x ) matrix candidates for ∥ x ∥ p p . Here we decide on the diagonal variant C 1 ( x ) . The objective function is no longer L -smooth, due to the non-smooth regularizer. As a result, we run LCD1 with the smallest L C such that the method converges. Additionally, LCD2 no longer has a closed form solution, so the projection algorithm must be deployed. To perform a fair comparison of our algorithms, we show both time and iteration plots.

Figure 3: Logistic regression on mushrooms dataset with L 3 regularization - iteration convergence.

<!-- image -->

Figure 4: Logistic regression on mushrooms dataset with L 3 regularization - time convergence.

<!-- image -->

Figure 3 displays similar to the L 2 case improvement of LCD2 over Polyak, which grows with λ . Our heuristic LCD3 can produce satisfying results, experimentally. However, its convergence cannot be guaranteed. In fact, as λ increases it becomes unstable. LCD1 converges at comparable pace with the other three methods at initial steps, yet the limited adaptiveness slows it down later on. Figure 4 shows convergence of our methods in time. One may point that the plots look almost identical to the iteration counterpart. The main reason is the cost of computing the gradient, which is O ( nd ) . All other operations performed by LCD3 and LCD1 are O ( d ) . The method with the most expensive update rule is LCD2 . At every step it performs around 5 rounds of the projection algorithm, each costing O ( d ) . We conclude that all the methods have comparable computational cost per iteration, as the main expense is the gradient evaluation. While the complexities discussed above are for diagonal matrices, we remark that the general O ( d 3 ) cost is bearable when n ≫ d . Moreover, our examples usually allow cheap diagonal matrix methods. Further experiments are in Appendix G.

## 9 Conclusion

We explored adaptive matrix-valued stepsizes under novel assumptions that reinforce convexity and L -smoothness with extra curvature information. Under our assumptions, we proposed LCD1 and LCD2 , which generalize GD with constant stepsize and Polyak stepsize, respectively. Moreover, we provided convergence theorems for both of these algorithms. We also proposed LCD3 which displays

promising experimental behavior. Our key insight is that, for some problems, we have certain local curvature information that can be readily exploited. We tested the methods on these problems using a variety of datasets, demonstrating strong empirical performance. The main limitation of our analysis is the restriction to a deterministic setting. We also acknowledge that the assumption has yet to be explored in its entirety. The most natural extension of the present work is including stochasticity and understanding the full potential of Assumption 2.1.

## 10 Acknowledgements

The work of all authors was supported by the KAUST Baseline Research Fund awarded to Peter Richtárik, who additionally acknowledges support from the SDAIA-KAUST Center of Excellence in Data Science and Artificial Intelligence. The work of Simone Maria Giancola, Dymitr Lubczyk and Robin Yadav was supported by the Visiting Student Research Program (VSRP) at KAUST. All authors are thankful to the KAUST AI Initiative for office space and administrative support.

## References

- F. Abdukhakimov, C. Xiang, D. Kamzolov, and M. Takáˇ c. Stochastic dradient descent with preconditioned Polyak step-size, preprint arXiv:2310.02093, 2023.
- D. Adil, R. Kyng, R. Peng, and S. Sachdeva. Fast algorithms for ℓ p -regression, preprint arXiv:2211.03963, 2023.
- M. Al-Baali and H. Khalfan. An overview of some practical quasi-Newton methods for unconstrained optimization. Sultan Qaboos University Journal for Science , 12(2):199, 2007. ISSN 2414-536X, 1027-524X. doi: 10.24200/squjs.vol12iss2pp199-209.
- M. Al-Baali, E. Spedicato, and F. Maggioni. Broyden's quasi-Newton methods for a nonlinear system of equations and unconstrained optimization: a review and open problems. Optimization Methods and Software , 29(5):937-954, 2014. ISSN 1055-6788, 1029-4937. doi: 10.1080/10556788.2013. 856909.
- J. Barzilai and J. M. Borwein. Two-Point Step Size Gradient Methods. IMA Journal of Numerical Analysis , 8(1):141-148, Jan. 1988. ISSN 0272-4979. doi: 10.1093/imanum/8.1.141. URL https://doi.org/10.1093/imanum/8.1.141 .
- H. H. Bauschke, J. Bolte, and M. Teboulle. A descent lemma beyond lipschitz gradient continuity: First-order methods revisited and applications. 42(2):330-348, 2017. ISSN 0364-765X. doi: 10. 1287/moor.2016.0817. URL https://pubsonline.informs.org/doi/10.1287/moor.2016. 0817 . Publisher: INFORMS.
7. C.-C. Chang and C.-J. Lin. LIBSVM: A library for support vector machines. ACM Transactions on Intelligent Systems and Technology , 2(3):27:1-27:27, 2011. ISSN 2157-6904. doi: 10.1145/ 1961189.1961199.
- F. H. Clarke. Optimization and nonsmooth analysis . Society for Industrial and Applied Mathematics, 1990. ISBN 978-0-89871-256-8 978-1-61197-130-9.
- A. Dasgupta, P. Drineas, B. Harb, R. Kumar, and M. W. Mahoney. Sampling algorithms and coresets for ℓ p regression. SIAM Journal on Computing , 38(5):2060-2078, 2009. ISSN 0097-5397, 1095-7111. doi: 10.1137/070696507.
- J. E. Dennis, Jr. and J. J. Moré. Quasi-Newton methods, ,otivation and theory. SIAM Review , 19(1): 46-89, 1977. ISSN 0036-1445. doi: 10.1137/1019005.
- Z. Dong, Y. Zhang, Z.-Q. Luo, J. Yao, and R. Sun. Towards Quantifying the Hessian Structure of Neural Networks, May 2025. URL http://arxiv.org/abs/2505.02809 . arXiv:2505.02809 [cs].
12. R.-A. Dragomir, A. B. Taylor, A. dâC™Aspremont, and J. Bolte. Optimal complexity and certification of bregman first-order methods, 2022. ISSN 1436-4646. URL https://doi.org/10.1007/ s10107-021-01618-1 .

- J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research , 12(61):2121-2159, 2011.
- R. M. Gower, D. Kovalev, F. Lieder, and P. Richtárik. RSN: Randomized subspace Newton, 2019.
- S. Hanzely, D. Kamzolov, D. Pasechnyuk, A. Gasnikov, P. Richtárik, and M. Takáˇ c. A damped newton method achieves global O ( 1 k 2 ) and local quadratic convergence rate. In A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, editors, Advances in Neural Information Processing Systems , 2022.
- G. E. Hinton. Neural networks for machine learning. Lecture slides, CSC 321, University of Toronto, 2014.
- S. M. Kakade, S. Shalev-Shwartz, and A. Tewari. Regularization techniques for learning with matrices. Journal of Machine Learning Research , 13(59):1865-1890, 2012.
- S. P. Karimireddy, S. U. Stich, and M. Jaggi. Global linear convergence of Newton's method without strong-convexity or Lipschitz gradients, 2018.
- D. P. Kingma and J. Ba. Adam: a method for stochastic optimization, preprint arXiv:1412.6980, 2017.
- D. Kovalev, R. M. Gower, P. Richtárik, and A. Rogozin. Fast linear convergence of randomized BFGS, preprint arXiv:2002.11337, 2021.
- S. Li, W. J. Swartworth, M. Takáˇ c, D. Needell, and R. M. Gower. SP2: a second order stochastic Polyak method, preprint arXiv:2207.08171, 2022.
- Y. Malitsky and K. Mishchenko. Adaptive gradient descent without descent. In Proceedings of the 37th International Conference on Machine Learning , pages 6702-6712. Proceedings of Machine Learning Research, 2020.
- K. Mishchenko. Regularized Newton method with global O (1 /k 2 ) convergence. SIAM Journal on Optimization , 33(3):1440-1462, 2023. ISSN 1052-6234, 1095-7189. doi: 10.1137/22M1488752.
- C. Musco, C. Musco, D. P. Woodruff, and T. Yasuda. Active linear regression for ℓ p norms and beyond. In 2022 IEEE 63rd Annual Symposium on Foundations of Computer Science (FOCS) , pages 744-753, 2022. doi: 10.1109/FOCS54457.2022.00076.
- I. E. Nesterov. Introductory lectures on convex optimization: a basic course . Applied Optimization. Kluwer Academic Publishers, 2004. ISBN 978-1-4020-7553-7.
14. Nocedal and Wright. Numerical optimization . Springer Series in Operations Research and Financial Engineering. Springer, 2006. ISBN 978-0-387-30303-1.
- B. Polyak. Gradient methods for the minimisation of functionals. USSR Computational Mathematics and Mathematical Physics , 3(4):864-878, 1963. ISSN 00415553. doi: 10.1016/0041-5553(63) 90382-3.
- B. Polyak. Introduction to optimization . Translations Series in Mathematics and Engineering. Optimization Software, Inc. Publications Division, New York, 1987. ISBN 0-911575-116.
- Z. Qu, P. Richtarik, M. Takac, and O. Fercoq. SDNA: Stochastic dual newton ascent for empirical risk minimization. In Proceedings of The 33rd International Conference on Machine Learning , pages 1823-1832. PMLR, 2016. URL https://proceedings.mlr.press/v48/qub16.html . ISSN: 1938-7228.
- S. J. Reddi, S. Kale, and S. Kumar. On the convergence of Adam and beyond, preprint arXiv:1904.09237, 2019.
- H. Robbins and S. Monro. A stochastic approximation method. The Annals of Mathematical Statistics , 22(3):400-407, 1951. ISSN 0003-4851. doi: 10.1214/aoms/1177729586.
- A. Rodomanov and Y. Nesterov. Greedy quasi-Newton methods with explicit superlinear convergence. SIAM Journal on Optimization , 31(1):785-811, 2021. ISSN 1052-6234, 1095-7189. doi: 10.1137/ 20M1320651.

- J. Yang, Y.-L. Chow, C. Ré, and M. W. Mahoney. Weighted SGD for ℓ p regression with randomized preconditioning. Journal of Machine Learning Research , 18(211):1-43, 2018.
- D. Zhou, S. Ma, and J. Yang. AdaBB: adaptive Barzilai-Borwein method for convex optimization, preprint arXiv:2401.08024, 2024.

## Table of Contents

| A Related Assumptions   | A Related Assumptions                  | A Related Assumptions                  |   14 |
|-------------------------|----------------------------------------|----------------------------------------|------|
| B                       | Local Curvature Descent 1 ( LCD1 )     | Local Curvature Descent 1 ( LCD1 )     |   15 |
|                         | B.1                                    | Derivation . . . . . . . . . . . .     |   15 |
|                         | B.2                                    | Convergence proof . . . . . . .        |   15 |
| C                       | Local Curvature Descent 2 ( LCD2 )     | Local Curvature Descent 2 ( LCD2 )     |   18 |
|                         | C.1                                    | Derivation . . . . . . . . . . . .     |   18 |
|                         | C.2                                    | Convergence proof . . . . . . .        |   20 |
|                         | C.3                                    | Closed-form solutions . . . . . .      |   21 |
| D                       | Local Curvature Descent 3 ( LCD3 )     | Local Curvature Descent 3 ( LCD3 )     |   23 |
|                         | D.1                                    | Derivation . . . . . . . . . . . .     |   23 |
|                         | D.2                                    | Convergence for quadratics . . .       |   24 |
| E                       | Properties &Examples                   | Properties &Examples                   |   25 |
|                         | E.1                                    | On the lower bound . . . . . . .       |   25 |
|                         | E.2                                    | On the upper bound . . . . . . .       |   28 |
|                         | E.3                                    | Lower bound examples . . . . .         |   30 |
|                         | E.4                                    | Lower and upper bound examples         |   36 |
| F                       | Absolutely Convex Functions            | Absolutely Convex Functions            |   41 |
|                         | F.1                                    | Examples . . . . . . . . . . . .       |   41 |
|                         | F.2                                    | Functions with zero minimum .          |   44 |
|                         | F.3                                    | Real valued functions . . . . . .      |   47 |
|                         | F.4                                    | Multivariable functions . . . . .      |   51 |
| G                       | Extra Experiments                      | Extra Experiments                      |   53 |
|                         | G.1 Regression with squared Huber loss | G.1 Regression with squared Huber loss |   53 |
|                         | G.2                                    | Ridge regression . . . . . . . .       |   53 |

## Appendix

## A Related Assumptions

In this section, we highlight the main differences and advantages of Assumption 2.1 over other related notions in literature such as relative smoothness and relative convexity [Bauschke et al., 2017, Dragomir et al., 2022, Gower et al., 2019, Qu et al., 2016].

Bauschke et al. [2017] proposed the notion of smoothness relative to a kernel function. A kernel function h : R d → R on closed convex subset C of R d is defined as follows,

Definition A.1. (Kernel function). A function h : R d → R is called a kernel function if,

- (i) h is a closed convex proper,
- (ii) dom h = C ,
- (iii) h is continuously differentiable and strictly convex on int dom h = ∅ .

̸

The Bregman distance of a kernel function is defined as,

<!-- formula-not-decoded -->

A function f is relatively smooth with respect to h if there exists a constant L such that Lh -f is convex on dom h . This implies the following inequality,

<!-- formula-not-decoded -->

Notably, if we choose the Euclidean kernel function h ( x ) = 1 2 ∥ x ∥ 2 , we recover the classic L -smooth assumption. Suppose M ∈ S d + is a positive semi-definite matrix. If we select h ( x ) = 1 2 ∥ x ∥ 2 M , we get an inequality that seems similar to Equation (6),

<!-- formula-not-decoded -->

Furthermore, Qu et al. [2016] propose a corresponding lower bound assumption on f based on another positive semi-definite matrix G ∈ S d + ,

<!-- formula-not-decoded -->

The crucial difference between Assumption 2.1 and Equation (10) and Equation (11) is that the curvature matrix in Assumption 2.1 can vary with y which allows it to capture a large class of functions. For instance, a simple function such as the square of the Huber loss given in Example 6.1 does not satisfy Equation (11) for any G ̸≡ 0 but does satisfy Assumption 2.1. Furthermore, the motivation for our assumption is to leverage local curvature information to develop adaptive algorithms. Hence, the fact that C ( y ) varies with y , which is the current iterate in our algorithms, is crucial to the performance as demonstrated by our numerical experiments.

Additionally, due to the variable nature of the norm in Assumption 2.1, it is not clear that there exists an elementary kernel function h such that LD h ( x, y ) = L 2 ∥ x -y ∥ 2 C ( y )+ LI . The difficulty arises from the fact that the matrix C ( y ) varies only with y . Therefore, we believe that Assumption 2.1 is not captured by the relative smoothness framework.

The Hessian of f has also been used to define other notions of relative convexity and relative smoothness. Specifically, there exists constants ˆ L ≥ ˆ µ &gt; 0 such that for all x, y ∈ R d ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assumption 2.1 does not restrict the curvature matrix to be the Hessian. Besides, our goal is to develop adaptive algorithms without going the route of fully-fledged second-order methods. Thus, leveraging readily available local curvature information without expensive computation of the Hessian is an advantage of our assumption and methods. Further, Li et al. [2022] recently proposed extensions of the Polyak step size, SP2 and SP2+ , that utilize the Hessian but can be unstable in practice. Therefore, we develop our assumption with the aim to rigorously utilize second-order information without the need to compute the Hessian.

## B Local Curvature Descent 1 ( LCD1 )

## B.1 Derivation

Suppose that the upper bound in (6) from Assumption 2.1) holds. Then, at a given point x k +1 ∈ R d , we have:

<!-- formula-not-decoded -->

Minimizing the right hand side with respect to x k +1 we find that:

<!-- formula-not-decoded -->

In particular, the matrix that pre-multiplies the vector is always invertible, since C ( x ) is positive semi-definite for each x ∈ R d .

## B.2 Convergence proof

Lemma B.1. Let Assumption 2.1 hold. For all k ≥ 0 , the sequence ( x k ) k ∈ N of LCD1 is such that:

<!-- formula-not-decoded -->

Proof. The proof is achieved by carefully bounding terms. For this reason, we split it into three steps. We seek a connection between the two distances in the geometry induced by ˜ C ( x k ) := C ( x k )+ L C I :

<!-- formula-not-decoded -->

Rearranging the terms we obtain

<!-- formula-not-decoded -->

In particular, we wish to bound the inner products.

Rearranging the lower bound (5) in Assumption 2.1 for the pair ( x k , x ⋆ ) :

<!-- formula-not-decoded -->

In a similar way, massaging the upper bound (6) of Assumption 2.1 for the pair ( x k +1 , x k ) one can derive:

<!-- formula-not-decoded -->

Combining two previous steps we find:

<!-- formula-not-decoded -->

The positive term ∥ x k -x ⋆ ∥ 2 C ( x k ) is on both sides of the inequality, so we can cancel it out:

<!-- formula-not-decoded -->

Having almost removed all the C ( x k ) norms, it suffices to apply the crude bound:

<!-- formula-not-decoded -->

which holds since L C I ⪯ C ( x k ) + L C I = ˜ C ( x k ) .

Therefore, we obtain

<!-- formula-not-decoded -->

Lemma B.2. For any k ∈ N , the iterations of LCD1 satisfy:

<!-- formula-not-decoded -->

Proof. Let us remind the form of the updates for each k ∈ N

<!-- formula-not-decoded -->

where ˜ C ( x k ) = C ( x k ) + L C I .

By Assumption 2.1, we know that

<!-- formula-not-decoded -->

and the claim follows by simple rearrangement.

Having the lemmas established, let us proceed to the proof of Theorem 4.1. We want to show that if f : R d → R satisfies Assumption 2.1 then for any k ∈ N , the iterates of LCD1 are such that:

<!-- formula-not-decoded -->

Proof. We use a standard Lyapunov function proof technique. For completeness, let us report it.

By Lemma B.2, function values get closer to f ⋆ across iterations. By Lemma B.1, the vectors get closer in norm to an optimum.

Then, we can combine the two positive decreasing terms L C ∥ x k -x ⋆ ∥ 2 and f ( x k ) -f ( x ⋆ ) into a Lyapunov energy function:

<!-- formula-not-decoded -->

In particular, E 0 = L C ∥ x 0 -x ⋆ ∥ 2 , and we claim that E k is a decreasing function. To see this, we start by rewriting the difference:

<!-- formula-not-decoded -->

It is evident that we can apply our Lemmas as follows:

<!-- formula-not-decoded -->

Putting everything together:

<!-- formula-not-decoded -->

showing that E k is decreasing. As a particular case, we then find:

<!-- formula-not-decoded -->

which reordered recovers the rate of GD with stepsize 1 L C , i.e.

<!-- formula-not-decoded -->

Remark. For quadratic functions Assumption 2.1 is satisfied with C ( x ) equal to the Hessian, and L C = 0 . Thus, LCD1 convergences in one step for this class of functions.

## C Local Curvature Descent 2 ( LCD2 )

## C.1 Derivation

Consider the minimization problem for the update step of LCD2 :

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

If C ( x k ) is the zero matrix, we know this problem has a closed-form solution. Therefore, we focus on the case where C ( x k ) is a non-zero matrix. Moreover, we assume that x k ̸∈ X ⋆ . The Lagrangian of this problem is:

<!-- formula-not-decoded -->

where β ≥ 0 . For optimal ¯ x and ¯ β we have that ∇ x L (¯ x, ¯ β ) = 0 . Therefore,

<!-- formula-not-decoded -->

Isolating for ¯ x , we find that:

̸

<!-- formula-not-decoded -->

We can see ¯ β = 0 so the constraint is tight. The next step would be to substitute ¯ x into the constraint and solve for ¯ β :

<!-- formula-not-decoded -->

Despite the left-hand side being a scalar function of ¯ β , we cannot obtain a closed-form solution for ¯ β . However, we can use an iterative root-finding sub-routine such as Newton's method to get an approximation of ¯ β cheaply and effectively. By substituting in the value of ¯ x , we see that we need to find the root of the following function:

<!-- formula-not-decoded -->

To simplify notation, let C := C ( x k ) , g := ∇ f ( x k ) and ∆ k := f ( x k ) -f ⋆ .

In the following proposition, we confirm that H has a root in the interval [0 , ∞ ) . We also show that H is convex and monotonically decreasing on that interval. Therefore, Newton's method is guaranteed to converge to the root of H at a quadratic rate. In particular, we do not need H to be monotonically decreasing; nonetheless, it is an interesting property of the problem.

Proposition C.1 (Properties of H ) . Let H be defined as in Equation (22) . Then, for β ≥ 0 :

<!-- formula-not-decoded -->

Proof. W.l.o.g. assume that C is a symmetric matrix. As a result, C is orthogonally diagonalizable so we let C = QDQ ⊤ where D is a diagonal matrix, and Q is an orthogonal matrix such that QQ ⊤ = I . Manipulating the inverse matrix in the definition of H , we find:

<!-- formula-not-decoded -->

Let ˜ g := Q ⊤ g . Let D i represent the i th entry of the diagonal of D and ˜ g i represent the i th entry in ˜ g . We rewrite H as:

<!-- formula-not-decoded -->

By inspection, H is a rational function and the derivative is easily found;

<!-- formula-not-decoded -->

̸

Since C is a positive semi-definite matrix, D i ≥ 0 for all i . Thus for β ≥ 0 , we have H ′ ( β ) ≤ 0 . Since the null-space of an orthogonal matrix is the singleton of the zero vector, the product ˜ g = Q ⊤ g is different than zero when g = 0 , which holds by the assumption x k / ∈ X ⋆ . Therefore, there is at least one ˜ g i that is non-zero and thus, H ′ ( β ) = 0 . The second derivative of H is,

̸

<!-- formula-not-decoded -->

By similar arguments used for the first derivative, we can show that H ′′ ( β ) &gt; 0 . To conclude, we will show that lim β →∞ H ( β ) &lt; 0 . We discuss two cases separately. C i D D = 0

Suppose is not invertible. Then, there exists an entry of such that i . Without loss of generality, suppose that the last entry D d , is equal to 0 . The same reasoning will apply if more than one entry is equal to 0 . Taking the limit:

<!-- formula-not-decoded -->

Now suppose that C is invertible. Then, D i &gt; 0 for all i . Differently from before:

<!-- formula-not-decoded -->

Recalling our definitions, the right hand side is:

<!-- formula-not-decoded -->

By Lemma E.5, lim β →∞ H ( β ) ≤ 0 . The inequality is strict when f ( x k ) -f ⋆ = 1 2 ∥∇ f ( x k ) ∥ 2 C -1 . In the case where equality holds, we have that lim β →∞ H ( β ) = 0 . Therefore, H does not have a root in the interval [0 , ∞ ) but the solution to the optimization problem is obtain when β = ∞ . This corresponds to the following optimal solution ¯ x :

̸

<!-- formula-not-decoded -->

Interestingly, under the same condition, LCD3 takes a step in the form x k +1 = x k -C -1 ( x k ) ∇ f ( x k ) . An example of a setting where the equality condition holds is when f is a convex quadratic and C is the Hessian of f . One can see that the update step of LCD3 and LCD2 are equivalent to Newton's method for that case so they both converge in one iteration.

It may seem that using Newton's root finding method is impractical because computing H defined in Equation (22) for a given β requires performing a matrix inversion. However, this can be avoided by computing the eigendecomposition of C ( x k ) at the beginning of each step of LCD2 . Then each subsequent evaluation of H done by Newton's method sub-routine only requires inverting a diagonal matrix and not the full matrix. Thus, the main cost at each step of LCD2 is computing the eigendecomposition of C ( x k ) once, which in practice is much faster than computing the inverse. Furthermore, if C ( x k ) is a diagonal matrix, the eigendecomposition of C ( x k ) is itself so each step of LCD2 becomes even cheaper. Also, in practice, Newton's method for root-finding is terminated when | H | &lt; ϵ . Therefore, in the case where

<!-- formula-not-decoded -->

the method will run until a large enough β is obtained and the step will become numerically equivalent to ¯ x = x k -C -1 ( x k ) ∇ f ( x k ) .

On a related note, Newton's method is used to solve a similar constrained optimization problem for trust region methods, namely, the trust region sub-problem [Nocedal and Wright, 2006]. Practical versions of such algorithms do not iterate until convergence but are content with an approximate solution that can be obtained in two or three iterations.

## C.2 Convergence proof

Lemma C.1. Let Assumption 2.1 hold. For all k ≥ 0 , the sequence ( x k ) k ∈ N of LCD2 obeys the recursion:

<!-- formula-not-decoded -->

Hence, for any k ≥ 1 , we have

<!-- formula-not-decoded -->

Proof. Let us write down the first-order optimality conditions for the optimization problem at Step 3 of LCD2 :

<!-- formula-not-decoded -->

Since x ⋆ ∈ L C ( x k ) , for any k ≥ 0 we have

<!-- formula-not-decoded -->

Summing up these inequalities for k = 0 , . . . , K -1 , we obtain (25).

<!-- formula-not-decoded -->

Let us proceed to the proof of Theorem 4.2 for LCD2 . We show that if f : R d → R satisfies Assumption 2.1 for any k ≥ 1 the iterates of LCD2 are such that:

<!-- formula-not-decoded -->

Proof. Since x k +1 ∈ L C ( x k ) , we have

<!-- formula-not-decoded -->

By rearranging the above inequality and applying (25) from Lemma C.1, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark. For quadratic functions Assumption 2.1 is satisfied with C ( x ) equal to the Hessian, and L C = 0 . Thus, LCD2 convergences in one step for this class of functions.

## C.3 Closed-form solutions

In the main text, we argued that the update step of LCD2 has a closed-form solution in certain special cases. One interesting case is when C ( x k ) is a rank one matrix. In the setting of minimizing the sum of squares of absolutely convex functions, we present a special rank one matrix and the corresponding update step of LCD2 . For general rank one matrices, the update step is not as interpretable or insightful so we leave out the computation.

Let f ( x ) = ∑ d i =1 ϕ 2 i ( x ) where ϕ i : R d → R is absolutely convex. Then f satisfies inequality (5) with the following curvature mapping:

<!-- formula-not-decoded -->

If we use the localization set defined by this curvature mapping, we can obtain a closed-form solution to the LCD2 update step. To simplify notation, let g k := ∇ f ( x k ) , f k := f ( x k ) , ∆ k := f ( x k ) -f ⋆ , D := D ( x k ) and Q = Q ( x k ) . Consider the orthogonal decomposition of C ( x k ) :

<!-- formula-not-decoded -->

where ˆ g k, 1 , . . . , ˆ g k,d -1 are d -1 orthogonal eigenvectors that are all also orthogonal to g k .

From Appendix C, we know that to obtain a closed-form solution of LCD2 , we must find the positive root of the following function:

<!-- formula-not-decoded -->

The second equality comes from simplifying the matrix multiplications and observing that,

<!-- formula-not-decoded -->

Let v = g ⊤ k g k . Since C is a rank-one matrix, we can simplify H and realize that it is a quadratic function of α ,

<!-- formula-not-decoded -->

Therefore, we will have at most two roots and there must exist a unique positive root that corresponds to the solution of the problem (the optimalLagrange multiplier). We can solve this quadratic to obtain an interpretable update step for LCD2 . To start, we multiply the entire equation by (2 f k + αv ) 2 and simplify to get,

<!-- formula-not-decoded -->

By observing that ∆ k = f k -f ⋆ we can simplify the expression further,

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

To determine which root is positive we can rearrange the terms to see that,

<!-- formula-not-decoded -->

For α to be positive we must select the positive sign. Now recall from Appendix C, that the update step of LCD2 is defined as follows,

<!-- formula-not-decoded -->

where γ k = α 2 f k 2 f k + αv . We substitute

<!-- formula-not-decoded -->

into γ k to get

<!-- formula-not-decoded -->

Therefore, we conclude that the update step has the following form:

<!-- formula-not-decoded -->

The update step of LCD2 differs from the classic Polyak stepsize in that we have √ f ( x k ) f ⋆ instead of f ⋆ and we multiply by 2 .

Remark. In case C ( x ) = c I , for c &gt; 0 , LCD2 reduces to LCD3 . Thus, the closed-form solution exists.

## D Local Curvature Descent 3 ( LCD3 )

## D.1 Derivation

Suppose the optimal value f ⋆ is known. While the update step of LCD2 does not have a closed-form solution, the following update step does:

<!-- formula-not-decoded -->

where L C ( x k ) is the same localization set defined in (21). Instead of using the L 2 norm, we can use the norm induced by C ( x k ) . Hence, this algorithm is referred to as LCD3 . The benefit of using a different norm is that we can obtain a closed-form solution to this constrained optimization algorithm.

The Lagrangian of this problem is

<!-- formula-not-decoded -->

If α is the optimal multiplier, then for optimal ¯ x we get ∇ x L (¯ x, ¯ α ) = 0 . The gradient is:

<!-- formula-not-decoded -->

Isolating ¯ x , we get:

<!-- formula-not-decoded -->

̸

Let t := ¯ α 1+¯ α . If ¯ α = 0 , x k ∈ L C ( x k ) , which means that the algorithm converged since x k ∈ L C ( x k ) if and only if x k ∈ X ⋆ . Then, for a generic update, we will have ¯ α = 0 . Imposing ∇ α L (¯ x, ¯ α ) = 0 , which means that the constraint must be tight:

<!-- formula-not-decoded -->

Plugging ¯ x ≡ ¯ x ( t ) ≡ ¯ x (¯ α ) into the equation gives

<!-- formula-not-decoded -->

The two inner products are norms of the form ∥∇ f ( x k ) ∥ 2 [ C ( x k )] -1 , and with more compact notation we can write:

<!-- formula-not-decoded -->

This equation has two roots summing up to 2, but only one of them can be of the form t = ¯ α 1+¯ α since only one of them can be smaller than 1, with expression:

<!-- formula-not-decoded -->

Substituting back t = ¯ α 1+¯ α into (28), where t is given by (29), leads to the method

<!-- formula-not-decoded -->

To realize that the scalar component of the stepsize is well-defined, it suffices to show that:

<!-- formula-not-decoded -->

which follows by reordering the result of Lemma E.5 for the pair ( x k , x ⋆ ) .

Then the update rule has a closed-form solution:

<!-- formula-not-decoded -->

In particular, the argument of the square root is always positive, making LCD3 well-defined. Routine (31) is promising: we apply a scalar stepsize γ LCD3 k that is similar in spirit to Polyak's in Equation (4), and 'reorient' the gradient according to C -1 k = [ C ( x k )] -1 .

Moreover, at each step, we aim to be as close as possible according to local upper-lower bounds on f . Experiments in section 8 and G, show that the algorithm converges, but is slower than LCD2 .

## D.2 Convergence for quadratics

Despite not converging in general, in special cases LCD3 reduces to Newton's method. Below, we show that the update rule (31) takes the form of Hessian times gradient.

Let ϕ i ( x ) = | a ⊤ i x -b i | , where a i ∈ R d and b i ∈ R , for i ∈ { 1 , . . . , n } . We know from Example 7.1 that ϕ i is absolutely convex. Then problem (9) becomes

<!-- formula-not-decoded -->

If x is such that ϕ i ( x ) = 0 for all i , then ∇ ϕ i ( x ) = a ⊤ i x -b i ϕ i ( x ) a i . Therefore, in view of the computation in Section 7 we get

̸

<!-- formula-not-decoded -->

Therefore, for least-squares problems, the LCD3 method of (31) moves in Newton's direction. Furthermore, γ LCD3 k = 1 since for quadratics we have the identity

<!-- formula-not-decoded -->

Indeed, this follows from Lemma E.5 and the fact that for quadratics, equation (5) is an identity. So, for least-squares problems, LCD3 reduces to Newton's method, and converges in one step.

## E Properties &amp; Examples

## E.1 On the lower bound

For clarity, the statements are repeated, but correspond to Lemma 5.1 and Corollary 5.1.

Lemma E.1. Suppose f, f 1 , f 2 : R d → R satisfy Equation (5) with curvature mappings C , C 1 , C 2 : R d → S d + , respectively. Then, the following functions satisfy Equation (5) :

<!-- formula-not-decoded -->

- (2) αf for α ≥ 0 , with α C ( · ) ;

<!-- formula-not-decoded -->

Proof.

(1) For any x, y ∈ R

We prove each statement separately. d , it holds:

<!-- formula-not-decoded -->

- (2) Similarly, for all x, y ∈ R d , one has:

<!-- formula-not-decoded -->

- (3) Concluding, for arbitrary vectors:

<!-- formula-not-decoded -->

Corollary E.1. Suppose f : R d → R satisfies the lower bound of Equation (5) with curvature mapping C : R d → S d + . Let g : R d → R be a convex function. Then, h ( x ) = f ( x ) + g ( x ) satisfies the lower bound with matrix C ( y ) .

Proof. Since g is convex it satisfies the lower bound with matrix C ( y ) ≡ 0 . By Lemma E.1, h satisfies the lower bound with C ( y ) + 0 = C ( y ) .

Another lemma used to construct functions that satisfy the lower bound is the following.

Lemma E.2. Suppose f : R d → R satisfies Equation (5) with the curvature mapping C : R d → S d + . Let A ∈ R d × m and b ∈ R d . Then g : R m → R where g ( x ) := f ( A x + b ) satisfies Equation (5) with curvature mapping ˜ C ( y ) = A ⊤ C ( A y + b ) A .

Proof. Without loss of generality, we can assume that C ( y ) is symmetric. Then,

<!-- formula-not-decoded -->

Considering the right hand side and the left hand side, we have recovered the claimed expression. The only missing detail is proving that ˜ C ( · ) is positive semi-definite. Let z ∈ R m . Since C ( y ) is symmetric and positive semi-definite then C ( y ) = C ( y ) 1 2 C ( y ) 1 2 and C ( y ) 1 2 is also symmetric. Therefore:

<!-- formula-not-decoded -->

By the arbitrariness of z

<!-- formula-not-decoded -->

, the matrix is positive semi-definite.

Lemma E.3. Suppose f : R d → R satisfies Equation (5) with curvature mapping C : R d → S d + . Then the following inequalities hold for any x, y ∈ R d ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Let us present one proof in detail. The other two follow trivially. (1) By the definition of Bregman divergence:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma E.4. Suppose f : R d → R satisfies

<!-- formula-not-decoded -->

for all x, y ∈ R d with curvature mapping C : R d → S d + . Then f satisfies Equation (5) with curvature mapping C ( · ) .

Proof. By the fundamental theorem of calculus:

<!-- formula-not-decoded -->

Rearranging the terms we obtain that D f ( x, y ) ≥ 1 ∥ x -y ∥ as desired.

2 2 C ( y )

Lemma E.5. Suppose that f : R d → R satisfies Equation (5) with curvature mapping C : R d → S d + . Suppose that f is differentiable and C is non-singular. Then

<!-- formula-not-decoded -->

Proof. Fix x ∈ R d . Suppose y ∈ R d is arbitrary. Let φ ( y ) := f ( y ) - ⟨∇ f ( x ) , y ⟩ . By construction, ∇ φ ( y ) = ∇ f ( y ) - ∇ f ( x ) . Using this fact, it can be shown that for any u, v ∈ R d , D f ( u, v ) ≥ 1 2 ∥ u -v ∥ 2 C ( v ) . Therefore, φ satisfies Equation (5) with curvature mapping C . Hence, for v ∈ R d we have that φ ( y ) ≥ G ( y ) where G ( y ) is defined as

<!-- formula-not-decoded -->

Observe that ∇ φ ( x ) = 0 . Since φ is convex, x is a minimizer of φ , and we the inequality below holds:

<!-- formula-not-decoded -->

By computing the gradient of G and setting it to zero, we find y = -C ( v ) -1 ∇ φ ( v ) + v such that ∇ G ( y ) = 0 . Therefore,

<!-- formula-not-decoded -->

By rearranging the terms, we find our result since x and v are arbitrary:

<!-- formula-not-decoded -->

Lemma E.6. Suppose that f : R d → R satisfies Equation (5) with curvature mapping C : R d → S d + . If f is twice continuously differentiable, then

<!-- formula-not-decoded -->

for all x ∈ R d .

Proof. Let x, y ′ ∈ R d and λ &gt; 0 . Since f satisfies Equation (5) we can substitute x + λ ( y ′ -x ) and λ ( y ′ -x ) into the first inequality described in Lemma E.3, to find:

<!-- formula-not-decoded -->

By the Fundamental Theorem of Calculus, we further have that:

<!-- formula-not-decoded -->

Dividing the first inequality by λ 2 on both sides we obtain an intermediate inequality:

<!-- formula-not-decoded -->

from which we take λ → 0 of both sides to get an inequality between norms,

<!-- formula-not-decoded -->

Thus, ∥ y ′ -x ∥ 2 C ( x ) ≤ 〈 ∇ 2 f ( x )( y ′ -x ) , y ′ -x 〉 . Since x, y ′ are arbitrary this implies that C ( x ) ⪯ ∇ 2 f ( x ) .

## E.2 On the upper bound

We provide analogous lemmas involving functions that satisfy Equation (6).

Lemma E.7. Suppose f : R d → R satisfies Equation (6) . Then for all x, y ∈ R d we have,

<!-- formula-not-decoded -->

Proof. Take the sum of the two Bregmann divergences:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma E.8. Suppose f : R d → R satisfies the following inequality with constant L C &gt; 0 and curvature mapping C : R d → S d + ,

<!-- formula-not-decoded -->

Then f satisfies Equation (6) with curvature mapping C and constant L C .

Proof. We invoke the Fundamental Theorem of Calculus:

<!-- formula-not-decoded -->

Rearranging the inequality above we get our result, D f ( x, y ) ≤ 1 ∥ x -y ∥ 2 .

2 C ( y )+ L I

Lemma E.9. Suppose that f : R d → R is convex and satisfies Equation (6) . Also, assume that f is differentiable. Then,

<!-- formula-not-decoded -->

Proof. Fix x ∈ R d . Suppose y ∈ R d . Let φ ( y ) := f ( y ) - ⟨∇ f ( x ) , y ⟩ . By construction, ∇ φ ( y ) = ∇ f ( y ) -∇ f ( x ) .

Using the above fact, we can show that φ is convex and that for any u, v ∈ R d , we have D φ ( u, v ) ≤ 1 2 ∥ u -v ∥ 2 C ( v )+ L C I . Therefore φ satisfies Equation (6). Now, let v ∈ R d be arbitrary. Since φ satisfies Equation (6), φ ( y ) ≤ G ( y ) where

<!-- formula-not-decoded -->

Moreover, x is a minimizer of φ because ∇ φ ( x ) = 0 and φ is convex. Combining the last two facts, φ ( x ) = inf y φ ( y ) ≥ inf y G ( y ) .

We minimize G with respect to y by finding a ¯ y ∈ R d such that ∇ G (¯ y ) = 0 . Since C ( v ) is positive semi-definite, C ( v ) + L C I is non-singular. Then ¯ y = v -( C ( v ) + L C I ) -1 φ ( v ) . Therefore,

<!-- formula-not-decoded -->

Rearranging the terms we obtain

<!-- formula-not-decoded -->

Since v, x were arbitrary, the claim is true.

Lemma E.10. Suppose that f : R d → R is convex and satisfies Equation (6) . Also, assume that f is twice differentiable. Then,

<!-- formula-not-decoded -->

Proof. Suppose x, y ′ ∈ R d and λ &gt; 0 . Since f satisfies Equation (6), we can substitute x + λ ( y ′ -x ) and λ ( y ′ -x ) into Lemma E.7,

<!-- formula-not-decoded -->

The following equality is a direct application of the fundamental theorem of calculus:

<!-- formula-not-decoded -->

Dividing the inequality by λ 2 on both sides:

<!-- formula-not-decoded -->

It suffices to take limits λ → 0 to get that:

<!-- formula-not-decoded -->

allowing us to conclude with:

<!-- formula-not-decoded -->

Since y ′ , x ∈ R d are arbitrary, we proved the claim: ∇ 2 f ( x ) ⪯ C ( x ) + L C I .

## E.3 Lower bound examples

Lemma E.11. Suppose p ≥ 2 . Let f : R d → R where f ( x ) = ∥ x ∥ p p . Then f satisfies Equation (5) in Assumption 2.1 with curvature mapping

<!-- formula-not-decoded -->

Proof. When p = 2 , we have that C ( y ) = 2 I . Then f satisfies Equation (5) because ∥ x ∥ 2 is 2 -strongly-convex.

Now suppose p &gt; 2 . For arbitrary x, y ∈ R d , an application of Young's Inequality yields

<!-- formula-not-decoded -->

Rearranging, we obtain:

and thus,

<!-- formula-not-decoded -->

By applying Hölder's inequality, we get:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By adding ∥ x ∥ p p + ( p 2 -1 ) ∥ y ∥ p p to both sides of Equation (34) and using Equation (33) we get,

<!-- formula-not-decoded -->

To derive the result, we begin by rearranging the above inequality:

<!-- formula-not-decoded -->

After reordering, we find:

<!-- formula-not-decoded -->

By performing some basic algebra and observing that ∂f ∂y i = py i | y i | p -2 , we obtain our result:

<!-- formula-not-decoded -->

Lemma E.12. Suppose p ≥ 2 . Let f : R d → R where f ( x ) = ∥ x ∥ p p . Then f satisfies Equation (5) in Assumption 2.1 with curvature mapping C ( y ) were the ( i, j ) th entry of C ( y ) is

<!-- formula-not-decoded -->

or alternatively, in matrix form:

<!-- formula-not-decoded -->

Proof. When p = 2 we get that C ( y ) = 2 ∥ y ∥ 2 yy ⊤ . Since ∥ x ∥ 2 is the square of ∥ x ∥ which is absolutely convex, f ( x ) = ∥ x ∥ 2 satisfies Equation (5) because the curvature mapping C corresponds to the mapping obtained from absolute convexity. Now suppose p &gt; 2 . Again by Hölder's Inequality, we have that

<!-- formula-not-decoded -->

We can lower bound the left-hand side in the following manner:

<!-- formula-not-decoded -->

Combining this inequality with Inequality (35) and squaring both sides we get,

<!-- formula-not-decoded -->

Then we multiply both sides by -2 ,

<!-- formula-not-decoded -->

From Lemma E.11, we know an application of Young's Inequality with some rearranging yields the following:

<!-- formula-not-decoded -->

Now multiply both sides by ∥ y ∥ p p ,

<!-- formula-not-decoded -->

Adding ∥ x ∥ p p ∥ y ∥ p p + ( p 2 -1 ) ∥ y ∥ 2 p p to both sides of Equation (36) and together with Equation (37) we have that,

<!-- formula-not-decoded -->

Rearranging this inequality and proceeding with the following steps we obtain the claim.

<!-- formula-not-decoded -->

The last term seems complicated but can be expressed as a matrix inner product. Continuing,

<!-- formula-not-decoded -->

To finalize, we proceed with the last few equalities:

<!-- formula-not-decoded -->

Lemma E.13. Suppose p ≥ 2 . The function f ( x ) = ∥ x ∥ p satisfies Equation (5) with either of the two curvature mappings:

<!-- formula-not-decoded -->

Proof. (1) . When p = 2 , we have C ( y ) = 2 I . Therefore, f satisfies Equation (5) because ∥ x ∥ 2 is 2 -strongly-convex.

Now suppose p &gt; 2 . By applying Young's Inequality we get that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We get our result from the above inequality and by observing that ∇ f ( y ) = p ∥ y ∥ p -2 2 y .

<!-- formula-not-decoded -->

(2) When p = 2 , we have C ( y ) = p ∥ y ∥ 2 2 yy ⊤ . Since ∥ x ∥ 2 is a square of an absolutely convex function, it satisfies Equation (5) with curvature mapping C ( y ) . For more details, refer to section 7 and F. Suppose that p &gt; 2 . As done previously, we can use Young's Inequality to obtain:

<!-- formula-not-decoded -->

Moreover, by Cauchy-Schwarz:

<!-- formula-not-decoded -->

Multiplying both sides by -p 2 ∥ y ∥ p -4 we get,

<!-- formula-not-decoded -->

Adding ∥ x ∥ p 2 + ( p 2 -1 ) ∥ y ∥ p 2 to both sides and by using Equation (38),

<!-- formula-not-decoded -->

and rearranging

We can reorder the terms to obtain the result:

<!-- formula-not-decoded -->

Lemma E.14. Suppose p ≥ 2 and let f : R d → R be defined as f ( x ) = ∥ x ∥ 2 p . Then f satisfies Equation (5) with the following curvature mapping:

<!-- formula-not-decoded -->

Proof. Using Holder's inequality we can see that,

<!-- formula-not-decoded -->

We raise both sides to the power of 1 p and proceed by rearranging some terms:

<!-- formula-not-decoded -->

We can arrive at our result by realizing that the last three terms are equal to ∥ x -y ∥ 2 C ( y ) and the middle two terms are equal to ⟨∇ f ( y ) , x -y ⟩ . Observe that ∂f ∂y i = 2 ∥ y ∥ p -2 p y i | y i | p -2 . Therefore,

<!-- formula-not-decoded -->

Lemma E.15. Suppose p ≥ 1 . Let g : R d → R be g ( x ) = ∥ x ∥ p . Then f := g 2 satisfies Equation (5) with the following curvature mapping:

<!-- formula-not-decoded -->

Proof. By Lemma F.2, g is absolutely convex for p ≥ 1 . Therefore, g 2 satisfies Equation (5) with curvature mapping C .

## E.4 Lower and upper bound examples

Lemma E.16. Let G be a symmetric positive semi-definite matrix. Let f : R d → R where f ( x ) = ∥ x ∥ 2 G . Then f satisfies Assumption 2.1 with curvature mapping C ( y ) ≡ 2 G and constant L C = 0 .

Proof. We start by computing:

<!-- formula-not-decoded -->

Rearranging the terms we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma E.17. Let p ≥ 2 . Suppose g : R d → R with g ( x ) = ∥ x ∥ 2 p . Let f := g 2 . Then f satisfies Assumption 2.1 with constant L C = 2( p -1) and either of the two curvature mappings:

<!-- formula-not-decoded -->

Proof. In Lemma E.14, we proved that f satisfies inequality (5) with the abovementioned curvature mappings. Now we will show that f is smooth so it satisfies inequality (6). For p = 2 . it is clear that f is 2 -smooth. We focus on the case where p &gt; 2 .

Since p &gt; 2 we have that 1 &lt; q &lt; 2 where q = p p -1 . Kakade et al. [2012] proved that h ( x ) = 1 2 ∥ x ∥ 2 q is strongly-convex with respect to the L q norm with µ = q -1 . We also know that L p norms are a decreasing function of p . Therefore, h ( x ) = 1 2 ∥ x ∥ 2 q is also strongly-convex with respect to the L 2 norm because ∥ x -y ∥ 2 2 ≤ ∥ x -y ∥ 2 q :

<!-- formula-not-decoded -->

The Frenchel conjugate of 1 2 ∥·∥ 2 q is 1 2 ∥·∥ 2 p because the dual norm of ∥·∥ q is ∥·∥ p . Kakade et al. [2012] showed that if h is µ -strongly-convex then the Frenchel conjugate of h is 1 µ -smooth. Therefore, 1 2 ∥ x ∥ 2 p is L -smooth with L = 1 q -1 = p -1 . Thus, f ( x ) = ∥ x ∥ 2 p is smooth with constant 2( p -1) .

̸

Lemma E.18. Suppose a, b ∈ R and a = 0 and b &gt; 0 . The function f : R → R defined as

<!-- formula-not-decoded -->

satisfies the upper and lower bounds in Assumption 2.1 with C ( y ) = 2 ay 2 f ( y ) and L C = √ 8 a .

Proof. Observe that x 2 + y 2 ≥ 2 x 2 y 2 . Multiply both sides by ab we get ab ( x 2 + y 2 ) ≥ 2 abx 2 y 2 . Then we add a 2 x 4 y 4 + b 2 to both sides,

<!-- formula-not-decoded -->

We can write this equivalently as,

<!-- formula-not-decoded -->

Then taking the square root of both sides,

<!-- formula-not-decoded -->

Rearranging the terms,

<!-- formula-not-decoded -->

Divide both sides by √ ay 4 + b to obtain our result,

<!-- formula-not-decoded -->

To compute L C , note that f is L -smooth so we can find an upper bound on f ′′ which is given by L C = √ 8 a .

Lemma E.19. Suppose δ &gt; 0 . Suppose h : R → R is such that:

<!-- formula-not-decoded -->

Let f = h 2 . Then f satisfies (2.1) with constant L C = 2 δ 2 and curvature mapping

<!-- formula-not-decoded -->

Notice that L C is less than the L -smoothness constant of f , which is 3 δ 2 (the tightest bound on the second derivative of f ).

Proof. First, we will prove that f satisfies inequality (5). We will split the proof into four cases and prove the inequality holds in each case.

Case | x | , | y | ≤ δ . We know x 4 + y 4 ≥ 2 x 2 y 2 . We divide both sides by 4 and rearrange the terms,

<!-- formula-not-decoded -->

We have our result because f ′ ( y ) = y 3 for | y | ≤ δ .

Case | x | ≥ δ and | y | ≤ δ . For any | y | ≤ δ define r : R → R as the following,

<!-- formula-not-decoded -->

First, we need to show that r ( x ) ≥ 0 . Suppose x ≥ δ . When x = δ ,

<!-- formula-not-decoded -->

Therefore, for x &gt; δ , if we show that r ′ ( x ) ≥ 0 then r ( x ) ≥ 0 . By a simple computation, we get that

<!-- formula-not-decoded -->

Since x ≥ δ then obviously x ≥ 2 3 δ so 3 2 x -δ ≥ 0 . Rearranging the terms and multiplying the entire inequality by δ we get that

<!-- formula-not-decoded -->

It is easy to show that -y 2 ≥ -δ 2 because | y | ≤ δ . Therefore,

<!-- formula-not-decoded -->

Now suppose x ≤ -δ . Observe that r ( -δ ) = r ( δ ) ≥ 0 . Thus, if we show that for x ≤ -δ , r ′ ( x ) ≤ 0 then r ( x ) ≥ 0 . For x ≤ -δ , we have that

<!-- formula-not-decoded -->

Notice that y 2 ≤ δ 2 because | y | ≤ δ . Then we have that 2 δ 2 -y 2 2 ≥ 3 δ 2 2 . Multiplying both sides of this inequality by x we obtain,

<!-- formula-not-decoded -->

The inequality was reversed because x ≤ -δ &lt; 0 and the second inequality also follows from x ≤ -δ . Rearranging the terms in this inequality we see that

<!-- formula-not-decoded -->

Therefore, we have shown that r ( x ) ≥ 0 for arbitrary absy ≤ δ . Then by rearranging the terms in r , we have that

<!-- formula-not-decoded -->

The left-hand side is equal to δ 2 ( | x | -1 2 δ ) = f ( x ) . In the previous case, we showed that the right-hand side is equal to 1 4 y 4 -y 3 ( x -y ) + 1 2 y 2 ( x -y ) 2 . Therefore, we have our result.

Case | x | , | y | ≥ δ . First, we show that the following inequality holds:

<!-- formula-not-decoded -->

For x, y ≥ δ and x, y ≤ -δ it is easy to show because ( x -y ) 2 ≥ 0 . Now suppose x ≥ δ and y ≤ -δ . Then we must show that

<!-- formula-not-decoded -->

Since y ≤ -δ we have that ( x -y ) 2 ≥ ( x + δ ) 2 . Therefore,

<!-- formula-not-decoded -->

Now suppose x ≤ -δ and y ≥ δ . Similar to before, we need to show

<!-- formula-not-decoded -->

Since y ≥ δ we obtain 4 xy ≤ 4 δx by multiplying both sides by 4 x and reversing the inequality because x ≤ -δ &lt; 0 . Therefore,

<!-- formula-not-decoded -->

As a result, we have shown inequality (39) holds. We can rewrite the inequality as

<!-- formula-not-decoded -->

Moving some terms to the right-hand side we get,

<!-- formula-not-decoded -->

Recall, that f ′ ( y ) = 2 δ 2 ( | y | -1 2 δ ) y | y | . Observe that we can factor the left-hand side of the inequality and after multiplying both sides by δ 2 we get our result:

<!-- formula-not-decoded -->

The case where | x | ≤ δ, | y | ≥ δ is similar to the previous cases. Using some elementary calculus, one can show that

<!-- formula-not-decoded -->

Rearranging the terms above directly leads to the result.

Now we show that f satisfies inequality (6) with L C = 2 δ 2 . In the case where | y | ≥ δ , L C + C ( y ) = 3 δ 2 is the L -smoothness constant of f so the inequality holds. We consider the case where | y | ≤ δ .

Case | x | , | y | ≤ δ . Then xy ≤ | xy | ≤ δ 2 so x 2 + xy ≤ 2 δ 2 . Adding y 2 to both sides and multiplying by ( x -y ) 2 we obtain,

<!-- formula-not-decoded -->

By Lemma E.8 we have our result.

Case | x | ≥ δ, | y | ≤ δ . We must show that the following inequality holds:

<!-- formula-not-decoded -->

We leave out the details of the calculations. The proof is similar to the same case for showing the lower bound. We can define a polynomial in x for arbitrary | y | ≤ δ . Then we show that this polynomial is less than 0 for x ≥ δ by computing the value at δ and show that the derivative is negative for x ≥ δ We proceed similarly for x ≤ -δ . By rearranging and manipulating the terms in the above inequality we can arrive at our result. These calculations are similar to the previous cases so we exclude them for brevity.

## F Absolutely Convex Functions

Discussing the theory constructions derived from Assumption 2.1, we introduced a stand-alone class of functions, satisfying an absolute convexity condition. In this section, we derive more properties and examples. Let us remind that a function ϕ : R d → R is absolutely convex if and only if:

<!-- formula-not-decoded -->

Our first statement is a Lemma that establishes calculus in the spirit of Lemma 5.1 in the main text. Lemma F.1. Let ϕ, ϕ 1 , ϕ 2 : R d → R be absolutely convex, and let A ∈ R d × m , b ∈ R d and α ≥ 0 . Then

- (i) ϕ + α is absolutely convex.
- (ii) αϕ is absolutely convex.
- (iii) ϕ 1 + ϕ 2 is absolutely convex.
- (iv) ϕ ( A x + b ) is absolutely convex.

Proof. We prove each statement:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (iv) ψ ( x ) = ϕ ( A x + b ) ≥ | ϕ ( A y + b ) + ⟨∇ ϕ ( A y + b ) , A x + b -( A y + b ) ⟩ | = | ϕ ( A y + b ) + 〈 A ⊤ ∇ ϕ ( A y + b ) , x -y 〉 | = | ϕ ( A y + b ) + 〈 A ⊤ ∇ ϕ ( A y + b ) , x -y 〉 | = | ψ ( y ) + ⟨∇ ψ ( y ) , x -y ⟩ | .

## F.1 Examples

Lemma F.2. Let p ≥ 1 . Then ϕ : R d → R where ϕ ( x ) = ∥ x ∥ p is absolutely convex.

Proof. We already know that ϕ is convex so we show that

<!-- formula-not-decoded -->

where ∂ϕ ( y ) ∂y i = y i | y i | p -2 ∥ y ∥ p -1 p for x, y ∈ R d .

Observe that

<!-- formula-not-decoded -->

We make use of Holder's inequality which is stated below for r, s ≥ 0 ,

<!-- formula-not-decoded -->

For any x, y ∈ R d and with r = 1 and s = p -1 we have that ,

<!-- formula-not-decoded -->

Simplifying this expression we obtain,

<!-- formula-not-decoded -->

We can obtain a lower bound on the term,

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

Now add ∥ y ∥ p p = ∑ d i =1 | y i | p to both sides of the inequality we get

<!-- formula-not-decoded -->

Now divide both sides of this inequality by ∥ y ∥ p -1 p ,

<!-- formula-not-decoded -->

By rearranging the terms we get our desired inequality

<!-- formula-not-decoded -->

Lemma F.3. There exists an absolutely convex function ϕ : R → R such that the derivative of f := ϕ 2 is not Lipschitz continuous. Namely,

<!-- formula-not-decoded -->

Proof. Firstly, by a simple computation we can show that ϕ ′′ ≥ 0 so ϕ is convex. Observe that | f ′ | ≤ 3 2 and x ⋆ = 0 . Also notice that ϕ is bounded below by 3 2 | x | . Therefore, by Lemma F.16, ϕ is absolutely convex.

From a brief computation, we can obtain,

<!-- formula-not-decoded -->

Now suppose by contradiction that h ′ is Lipschitz continuous. Then there exists an L &gt; 0 such that for all x, y ∈ R d ,

<!-- formula-not-decoded -->

Specifically, this holds for 0 &lt; x &lt; 1 and y = 0 so we have

<!-- formula-not-decoded -->

We can find an x small enough such that 3 √ x &gt; L . Therefore, the inequality cannot hold. As a consequence, f ′ is not Lipschitz continuous.

Lemma F.4. Let δ &gt; 0 . Then f, ϕ : R → R defined below are absolutely convex.

<!-- formula-not-decoded -->

Note that f is the pseudo-Huber loss function and ϕ is the Huber loss function.

Proof. Both the Huber loss and pseudo-Huber loss functions are well-known examples of convex loss functions. The minimizer of f and ϕ is x ⋆ = 0 .

From a simple computation we obtain that | f ′ ( x ) | ≤ 1 for all x ∈ R ,

<!-- formula-not-decoded -->

Also, we know that for all x ∈ R ,

<!-- formula-not-decoded -->

Therefore, by Lemma F.16, f is absolutely convex.

By computing the derivative of ϕ , we can show that | ϕ ′ ( x ) | ≤ δ for all x ∈ R . Now we show that ϕ ( x ) ≥ δ | x | .

When x &gt; δ , we can simply observe that ϕ ( x ) = δ | x | . Now consider the case where x ≤ δ . We have that,

Expanding the square we get,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Rearranging the terms and dividing by 2 we see that,

<!-- formula-not-decoded -->

Hence, by Lemma F.16, ϕ is absolutely convex.

Lemma F.5. Let a &gt; 0 and b ≥ 0 . Then ϕ : R → R where ϕ ( x ) = √ ax 2 + b is absolutely convex.

Proof. Notice that x ⋆ = 0 . Also by a simple computation we know can show that f is convex,

<!-- formula-not-decoded -->

We can compute an upper bound on the derivative of f ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then,

Since ϕ ( x ) ≥ √ a | x | , by Lemma F.16, f is absolutely convex.

Lemma F.6. The function ϕ : R → R , defined as follows

<!-- formula-not-decoded -->

is absolutely convex.

Proof. Observe that x ⋆ = 0 . By a simple computation, we can show that ϕ ′′ ( x ) ≥ 0 for all x ∈ R ,

<!-- formula-not-decoded -->

Similarly to the previous examples, we compute an upper bound on ϕ ′ ,

<!-- formula-not-decoded -->

It is clear that | ϕ ′ ( x ) | ≤ 1 for all x ∈ R . It is easy to shoiw ϕ ( x ) ≥ | x | . For x ≥ 0 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For x &lt; 0 , we have

Therefore, by Lemma F.16, ϕ is absolutely convex.

## F.2 Functions with zero minimum

Absolutely convex functions have some interesting properties when their minimum is 0 . Lemma F.7. If ϕ is absolutely convex, then the following statements are equivalent:

1. ϕ (0) = 0 ,
2. ϕ ( x ) = ⟨∇ ϕ ( x ) , x ⟩ ,

3. ϕ is homogeneous of degree 1.

Proof. We establish three implications:

- ( i ) ⇒ ( ii ) Pick any y . If ϕ (0) = 0 , then using Equation (40) with x = 0 leads to | ϕ ( y ) + ⟨∇ ϕ ( y ) , -y ⟩| ≤ 0 , which implies ϕ ( y ) = ⟨∇ ϕ ( y ) , y ⟩ .

( ii ) ⇒ ( iii ) We start by substituting ϕ ( y ) = ⟨∇ ϕ ( y ) , y ⟩ and ϕ ( x ) = ⟨∇ ϕ ( x ) , x ⟩ into ϕ ( x ) ≥ ϕ ( y ) + ⟨∇ ϕ ( y ) , x -y ⟩ to get,

<!-- formula-not-decoded -->

This means that,

So for some m ∈ R ≥ 0 .

Proof. We have that f ( x ) = ϕ ( x + x ⋆ ) is absolutely convex. It is also homogeneous of degree one since f (0) = 0 . Define U 1 = { x ∈ R | x &gt; 0 } and U 2 = { x ∈ R | x &lt; 0 } .

By homogeneity, for any t &gt; 0 and x ∈ U 1 we have that f ( tx ) = tf ( x ) . Differentiating both sides with respect to x we get f ′ ( tx ) = f ′ ( x ) . This means that for any x ∈ U 1 we have that f ′ ( x ) = m 1 for some m 1 ∈ R . Since U 1 is a connected open set, we have that f ( x ) = m 1 x for x ∈ U 1 .

By similar reasoning, for x ∈ U 2 , f ′ ( x ) = m 2 for some m 2 ∈ R . Also then, f ( x ) = m 2 x for x ∈ U 2 .

Now consider x ∈ U 1 and y ∈ U 2 . By absolute convexity we know that

<!-- formula-not-decoded -->

Then m 1 ≥ | m 2 | x | x | = | m 2 | which comes from the fact that x ∈ U 1 so x &gt; 0 and thus x | x | = 1 . Similarly,

<!-- formula-not-decoded -->

Then | m 1 | ≤ m 2 y | y | = -m 2 because y ∈ U 2 so y &lt; 0 and thus y | y | = -1 .

Since 0 ≤ | m 1 | ≤ -m 2 we get m 2 ≤ 0 . Also because | m 2 | ≤ m 1 it must be that m 2 = -m 1 where m 1 ≥ 0 .

<!-- formula-not-decoded -->

where Q := {∇ ϕ ( y ) : y ∈ R d } . As a consequence, for any t ≥ 0 and x ∈ R d we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma F.8. Let ϕ : R d → R be absolutely convex. Suppose there exists an x ⋆ ∈ R d such that ϕ ( x ⋆ ) = 0 . Then

<!-- formula-not-decoded -->

Proof. From absolute convexity, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Simply rearranging we get our result,

<!-- formula-not-decoded -->

Lemma F.9. Let ϕ : R → R be absolutely convex. Suppose there exists an x ⋆ ∈ R such that ϕ ( x ⋆ ) = 0 . Also, suppose that ϕ is differentiable everywhere but at x ⋆ . Then it must be that

<!-- formula-not-decoded -->

For brevity, we set m = m 1 and so we have that

<!-- formula-not-decoded -->

By definition we have that ϕ ( x ) = f ( x -x ⋆ ) . Therefore, we obtain our desired result

<!-- formula-not-decoded -->

Lemma F.10. Suppose that ϕ : R d → R is absolutely convex and ϕ (0) = 0 . Then ϕ is sub-additive,

<!-- formula-not-decoded -->

Proof. By Lemma F.7, we know ϕ is positively homogeneous of degree one. Also because ϕ is convex we have that for any 0 ≤ α ≤ 1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By combining the previous inequality with inequality (42) and multiplying by 2 we get our result.

Lemma F.11. Suppose that ϕ : R d → R is absolutely convex and ϕ (0) = 0 . Then the epigraph of ϕ is a convex cone.

Proof. Now we show that epi ϕ is a convex cone. Suppose ( x, µ 1 ) ∈ epi ϕ and ( y, µ 2 ) ∈ epi ϕ . Suppose α ≥ 0 and β ≥ 0 . Then

<!-- formula-not-decoded -->

The first inequality is from sub-additivity and the first equality is from homogeneity, Therefore, ( αx + βy, αµ 1 + βµ 2 ) ∈ epi ϕ so epi ϕ is a convex cone.

Lemma F.12. Suppose that ϕ : R d → R is absolutely convex and ϕ (0) = 0 . Then ϕ is even, i.e.

<!-- formula-not-decoded -->

Proof. By absolute convexity we have that for any x, y ∈ R d

<!-- formula-not-decoded -->

Similarly,

<!-- formula-not-decoded -->

Now subtitute y = -x into (44),

<!-- formula-not-decoded -->

where the last equality is because absolutely convex functions are non-negative. Substituting y = -x into (45),

<!-- formula-not-decoded -->

Therefore, ϕ ( x ) = ϕ ( -x ) .

Selecting α = 1 2 ,

By homogeneity of ϕ ,

## F.3 Real valued functions

Simple absolutely convex functions from the real numbers to the real numbers are an instructive playground to understand how to finalize the generalized proofs. Below, we report some properties of such sub-class. In particular, we prove bounded subgradients in Lemma F.15 and a useful result for validating examples in Lemma F.16.

̸

Lemma F.13. Let f : R → R defined as follows, f ( x ) = | a + b ( x -x 0 ) | for some a, b, x 0 ∈ R with b = 0 . Then for any c &gt; 0 . There exists x 1 , x 2 ∈ R with x 2 ≥ x 1 , f ( x 1 ) = f ( x 2 ) = c , | x 2 -x 1 | = 2 c | b | and f ′ ( x 2 ) ≥ 0 and f ′ ( x 1 ) ≤ 0 .

Proof. By simply solving the equation f ( x ) = c , we get that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

Suppose without loss of generality that b &gt; 0 . Then f ′ ( x 2 ) = cb &gt; 0 and f ′ ( x 1 ) = -cb &lt; 0 .

̸

Lemma F.14. Suppose ϕ : R → R is a convex function that is not constant with minimizer x ⋆ . Suppose ϕ lower bounded by f ( x ) = | a + b ( x -x 0 ) | for a, b, x 0 ∈ R and b = 0 . For any c &gt; ϕ ( x ⋆ ) there exists an x ′ such that ϕ ( x ′ ) = c and | x ′ -x ⋆ | ≤ 2 c | b | .

Proof. By Lemma F. 13 , there exists x 1 , x 2 ∈ R such that | x 2 -x 1 | = 2 c | b | and f ( x 1 ) = f ( x 2 ) = c .

We also know that x 2 &gt; x 1 and that f ′ ( x 2 ) ≥ 0 and f ′ ( x 1 ) ≤ 0 . We will show that x 1 &lt; x ⋆ &lt; x 2 . Suppose by contradiction that x ⋆ ≥ x 2 . Since f is a linear function with slope f ′ ( x 2 ) on the interval [ x 2 , ∞ ) we get

<!-- formula-not-decoded -->

which is a contradiction because f is supposed to be a lower bound on ϕ . The first inequality follows from the fact that f ′ ( x 2 )( x ⋆ -x 2 ) ≥ 0 . A similar argument follows if the assumption x ⋆ ≤ x 1 is made.

So c = f ( x 1 ) ≤ ϕ ( x 1 ) and ϕ ( x ⋆ ) &lt; c . By the Intermediate Value Theorem, there exists x ′ ∈ ( x 1 , x ⋆ ) such that ϕ ( x ′ ) = c .

<!-- formula-not-decoded -->

Lemma F.15. Suppose ϕ : R → R is absolutely convex and has a minimizer x ⋆ . Then there exists an M ∈ R such that | ϕ ′ ( x ) | ≤ M for all x ∈ R i.e. | ϕ ′ | is bounded.

Proof. If ϕ is constant then its derivative is bounded so we consider the case where ϕ is not constant.

We will do a proof by contradiction. Let c ∈ R such that c &gt; ϕ ( x ⋆ ) . Let ϵ = c -ϕ ( x ⋆ ) 2 . Note that | ϕ ( x ⋆ ) -c | &gt; ϵ .

By continuity of ϕ at x ⋆ there must be a δ &gt; 0 such that if | x -x ⋆ | ≤ δ then | ϕ ( x ) -ϕ ( x ⋆ ) | ≤ ϵ .

Suppose that | ϕ ′ | is unbounded. Therefore, there exists a sequence of numbers y n ∈ R such that

<!-- formula-not-decoded -->

For any n , let f n ( x ) = | ϕ ( y n ) + ϕ ′ ( y n )( x -y n ) | . Since ϕ is absolutely convex, f n must be a lower bound on ϕ . By Lemma F.14 there exists an x n such that ϕ ( x n ) = c and | x n -x ⋆ | ≤ 2 c | ϕ ′ ( y n ) | for any y n .

Since | ϕ ′ | is unbounded we can choose a N such that 2 c | ϕ ′ ( y N ) | ≤ δ . Thus | x N -x ⋆ | ≤ δ . Therefore, | ϕ ( x ⋆ ) -ϕ ( x N ) | = | ϕ ( x ⋆ ) -c | ≤ ϵ .

.

But this contradicts the fact that | ϕ ( x ⋆ ) -c | &gt; ϵ

Lemma F.16. Suppose ϕ : R → R is convex and has a minimizer at x ⋆ = 0 . Suppose ϕ ′ is bounded. If ϕ can be lower bounded by h ( x ) = m | x | where | ϕ ′ ( y ) | ≤ m for all y ∈ R then ϕ is absolutely convex.

Proof. Suppose without loss of generality y &lt; x ⋆ = 0 . Since ϕ is convex we have that ϕ ′ ( y ) ≤ 0 by monotoncity of | ϕ ′ | . Since m is a bound on ϕ ′ we get

<!-- formula-not-decoded -->

Since h is a lower bound on ϕ we also know that .

<!-- formula-not-decoded -->

Now let f ( x ) = | ϕ ( y ) + ϕ ′ ( y )( x -y ) | . Denote x v as the point where f ( x v ) = 0 . On the interval ( -∞ , x v ] , f is equal to the line tangent to ϕ at point y . So by convexity, f is a lower bound on ϕ on the interval ( -∞ , x v ] . It remains to show that f lower bounds ϕ on the interval [ x v , ∞ ) .

First, we show that x v = -ϕ ( y ) ϕ ′ ( y ) + y ≥ x ⋆ = 0 . We can take the reciprocal of inequality (46) to obtain, 1 m ≤ -1 ϕ ′ ( y ) . Multiply this inequality by (47) to get, -y ≤ -ϕ ( y ) ϕ ′ ( y ) . We can do this because the terms on both sides of the inequalities are positive. Rearranging we can see that x v ≥ 0 .

Suppose x ∈ [ x v , ∞ ) . On this interval, f is a line with slope -ϕ ′ ( y ) passing through the point ( x v , 0) . Thus we can rewrite f as f ( x ) = -ϕ ′ ( y )( x -x v ) . Since x v ≥ 0 we know that x -x v ≤ x . Multiply this inequality by m ≥ -ϕ ′ ( y ) ≥ 0 to get that -ϕ ′ ( y )( x -x v ) ≤ mx . Since h ( x ) = mx is a lower bound on ϕ we have that -ϕ ′ ( y )( x -x v ) ≤ ϕ ( x ) . Therefore, f is a lower bound on ϕ for arbitrary y &lt; 0 so we are done.

A similar argument can be made for y ≥ 0 , the signs will be flipped at each step.

A direction of potential interest for future developments is how to 'absolutely-convexify' a given function. Below, we prove the propotypical case of functions from R to R . In words, any convex function lifted high enough is absolutely convex.

Lemma F.17. Suppose f : R → R is a convex function. Then ϕ ( x ) = f ( x ) + β is non-negative for x ∈ [ a, b ] with α = max {| f ′ ( a ) | , | f ′ ( b ) |} and β = α ( b -a ) 2 -f ( a )+ f ( b ) 2 .

Proof. It is sufficient to show that ϕ ( x ) ≥ 0 for x ∈ [ a, b ] :

<!-- formula-not-decoded -->

Lemma F.18. Suppose f : R → R is a convex function. Then for any interval [ a, b ] there exists β , such that ϕ ( x ) = f ( x ) + β is absolutely convex on [ a, b ] .

Proof. As ϕ is convex by convexity of f , it is sufficient to show that for every x, y ∈ [ a, b ] :

<!-- formula-not-decoded -->

From Lemma F.17 we have ϕ ( x ) ≥ 0 , so it is sufficient to consider the case when ϕ ′ ( y )( x -y ) ≤ 0 . Having α, β from Lemma F.17.

Case 1: ϕ ′ ( y ) ≤ 0 and x ≥ y .

From the convexity of ϕ it follows that:

<!-- formula-not-decoded -->

Hence, -f ′ ( a ) ≤ α by construction, implying:

<!-- formula-not-decoded -->

It remains to show that:

<!-- formula-not-decoded -->

Case 2: ϕ ′ ( y ) ≥ 0 and x ≤ y .

In analogy with the previous case:

<!-- formula-not-decoded -->

Therefore, it is sufficient to show that:

<!-- formula-not-decoded -->

It is useful to remind the fllowing standard result for sub-gradients. The proof is in the referenced book.

Lemma F.19 (Lebourg Mean Value Theorem [Clarke, 1990]) . Suppose ϕ : R → R is Lipschitz on any open set containing the line segment [ x, y ] . Then there exists an a ∈ ( x, y ) such that

<!-- formula-not-decoded -->

Lemma F.20. Suppose ϕ : R → R is absolutely convex. Let M ∈ R be the bound on the subgradient of ϕ , i.e. | ϕ ′ ( x ) | ≤ M . Fix a y ∈ R . Then for any x ≥ y we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We prove the first claim. Suppose x ≥ y .

Since ϕ is convex, it is locally Lipschitz i.e. Lipschitz on [ y, x ] . By Lebourg's MVT we know there exists a c ∈ ( y, x ) such that ϕ ( x ) -ϕ ( y ) = g ( x -y ) where g ∈ ∂ϕ ( c ) . Note that g ≤ M . So, g ( x -y ) ≤ M ( x -y ) because x -y ≥ 0 . Therefore,

<!-- formula-not-decoded -->

Rearranging this expression we obtain ϕ ( x ) ≤ ϕ ( y ) + M ( x 0 -y ) .

The proof of the second claim follows the same format and uses the fact that g ≥ -M instead.

Similarly, for any x ≤ y ,

<!-- formula-not-decoded -->

Lemma F.21. Suppose ϕ : R → R is absolutely convex. Let M ∈ R be such that | ϕ ′ | ≤ M . Then the following limits exist and are equal

<!-- formula-not-decoded -->

Proof. First, we show that lim x →∞ | ϕ ′ ( x ) | exists.

Let { x n } be an arbitrary sequence such that x n →∞ . Since ϕ is convex the sequence {| ϕ ′ ( x n ) |} is monotonically increasing. Also, it is bounded above. Therefore, by the Monotone Convergence Theorem there exists an L 1 ∈ R

<!-- formula-not-decoded -->

Since x n is arbitrary it must be that lim x →∞ | ϕ ′ ( x ) | = L 1 . A similar argument can demonstrate that there exists an L 2 ∈ R with | L 2 | ≥ | ϕ ′ | and lim x →-∞ | ϕ ′ ( x ) | = | L 2 | .

Now we show that L 1 = | L 2 | . We will do this by contradiction. Suppose without loss of generality that L 1 &gt; | L 2 | and that L 2 &lt; 0 .

There exists an ϵ &gt; 0 such that L 1 -ϵ &gt; | L 2 | . Now there also exists an N such that for any y &gt; N we have that | ϕ ′ ( y ) -L 1 | &lt; ϵ . So L 1 -ϵ &lt; ϕ ′ ( y ) .

By absolutely convexity of ϕ , function f ( x ) = | ϕ ( y ) + ϕ ′ ( y )( x -y ) | is a lower bound on ϕ . Let x v be the value where f ( x v ) = 0 .

Define l ( x ) = -ϕ ′ ( y )( x -x v ) , which is the line passing through the point ( x v , 0) with slope -ϕ ′ ( y ) . Observe that for x ∈ ( -∞ , x v ] , l ( x ) = f ( x ) so l is a lower bound on ϕ in that interval. Define h 1 ( x ) = -( L 1 -ϵ )( x -x v ) to be the line passing through ( x v , 0) with slope -( L 1 -ϵ ) . For x ≤ x v , h 1 ( x ) &lt; l ( x ) because L 1 -ϵ &lt; ϕ ′ ( y ) . Therefore, h 1 ( x ) &lt; ϕ ( x ) for x ∈ ( -∞ , x v ] .

Define h 2 ( x ) = ϕ ( x v ) + L 2 ( x -x v ) . Since L 2 &lt; 0 , by Lemma F.20, the function h 2 is an upper bound on ϕ for any x &lt; x v .

By calculation we can determine an x i such that h 1 ( x i ) = h 2 ( x i ) where

<!-- formula-not-decoded -->

Now we show that x i ≤ x v . Note that ϕ is absolutely convex so -ϕ ( x 0 ) ≤ 0 . By adding ( L 2 + L 1 -ϵ ) x v to both sides of this inequality we obtain

<!-- formula-not-decoded -->

Dividing by L 2 + L 1 -ϵ we get

<!-- formula-not-decoded -->

Let x &lt; x i . So h 1 ( x ) = h 1 ( x i ) -( L 1 -ϵ )( x -x i ) and h 2 ( x ) = h 1 ( x i ) + L 2 ( x -x i ) . Observe that -( L 1 -ϵ ) &lt; L 2 because L 2 &lt; 0 and L 1 -ϵ &gt; | L 2 | . Multiplying both sides of this inequality by x -x i which is less than 0 and then adding h ( x i ) to both sides again we see that h 1 ( x ) &gt; h 2 ( x ) .

Therefore, for any x &lt; x i &lt; x v , we have that h 1 ( x ) &gt; h 2 ( x ) . This is a contradiction because on the interval ( -∞ , x v ) , h 1 is a lower bound so ϕ ( x ) ≥ h 1 ( x ) and h 2 is an upper bound so h 2 ( x ) ≥ ϕ ( x ) .

Lemma F.22. Suppose ϕ : R → R is absolutely convex. Suppose it has a minimum point x ⋆ . Suppose there exists a y 1 , y 2 ∈ R such that y 1 &lt; x ⋆ &lt; y 2 and for every y ≤ y 1 , ϕ ′ ( y ) = m 1 and for every y ′ ≥ y 2 , ϕ ′ ( y ′ ) = m 2 . Then m 1 = m 2 .

Proof. Note that since ϕ is convex it has monotonicly increasing derivative. Therefore, m 1 &lt; 0 since y &lt; x ⋆ and m 2 &gt; 0 since y ′ &gt; x ⋆ . Let f ( x ) = | ϕ ( y 1 ) + m 1 ( x -y 1 ) | be the tangent cone to y 1 . Then define line h ( x ) = | m 1 | ( x + ϕ ( y 1 ) m 1 -y 1 ) which is a line that has the same slope as f and intersects the vertex of f . Note that h is a lower bound on ϕ by absolute convexity.

Lemma F.23. Suppose ϕ : R → R be absolutely convex and assume ϕ ∗ = inf x ∈ R ϕ ( x ) &lt; ∞ . Then there exists an x ⋆ ∈ R such that ϕ ( x ⋆ ) ≤ ϕ ( x ) for any x ∈ R .

Proof. Suppose x ∈ R is arbitrary. Choose a y ∈ R and by absolute convexity we have that | ϕ ′ ( y ) + ϕ ′ ( y )( x -y ) | ≤ ϕ ( x ) . We can rearrange the terms on the left side and take the limit to see that

<!-- formula-not-decoded -->

since only the last term which is linear depends on x . Therefore, we have that lim | x |→∞ ϕ ( x ) = ∞ .

This demonstrates that ϕ is coercive. Now let x n be a sequence such that ϕ ( x n ) → f ∗ . Suppose that lim n →∞ | x n | = ∞ . Then by coercivity, we get that lim | x n |→∞ ϕ ( x n ) = infty which is a contradiction with the fact that ϕ ∗ ≤ ∞ . Thus it must be that lim n →∞ | x n | = r for some r ∈ R . Let B r = { x ∈ R | | x | ≤ r } which is compact because it is closed and bounded. Since x n ∈ B r and every sequence in a compact set has a convergent subsequence, so there exists an x ⋆ ∈ B r s.t. x n k → x ⋆ . By continuity of ϕ (because it is a convex function) we obtain f ∗ = lim k →∞ f ( x n k ) = f ( x ⋆ ) .

## F.4 Multivariable functions

Having analyzed the easy case, we move to general instances of absolutely convex functions. In particular, we prove that gradients of absolutely convex functions are bounded. The first statement is a rewriting of Lemma 7.1 in the main text.

Lemma F.24. Suppose ϕ : R d → R is absolutely convex and has a minimizer x ⋆ . Then there exists a M ∈ R such that ∥∇ ϕ ( x ) ∥ 2 ≤ M for all x ∈ R d .

Proof. If ϕ is the constant function then its gradient is bounded so we consider the case where ϕ is not constant.

We will do a proof by contradiction. Let c &gt; ϕ ( x ⋆ ) . Let ϵ = c -ϕ ( x ⋆ ) 2 . Note that | ϕ ( x ⋆ ) -c | &gt; ϵ . Suppose δ &gt; 0 . Observe that since ϕ is convex on R d it is continuous on R d and in particular it is continuous at x ⋆ . Therefore, there exists a δ &gt; 0 such that if | x ⋆ -x | ≤ δ then | ϕ ( x ⋆ ) -ϕ ( x ) | ≤ ϵ .

Suppose that |∇ ϕ | is unbounded. So, there exists a sequence of points y n ∈ R d such that

<!-- formula-not-decoded -->

Let f n ( x ) = | ϕ ( y n ) + ⟨∇ ϕ ( y n ) , x -y n ⟩| . Since ϕ is absolutely convex, f n must be a lower bound on ϕ . We can proceed similarly to the proof of bounded gradients in R (Lemma F.15) by considering the restriction of f and ϕ to specific lines. This allows us to find a sequence of points x n that lie on those lines and x n → x ⋆ .

Define L n to be the line that passes through x ⋆ in the direction of ∇ ϕ ( y n ) . Let ϕ ∣ ∣ L n : R → R be the restriction of ϕ to the line L n and similarly, f n ∣ ∣ L n : R → R be the restriction of f n to L n . Note that the function f n ∣ ∣ L n is of the form | a + b ( x -x 0 ) | where b = ∥∇ ϕ ( y n ) ∥ 2 for some a, x 0 ∈ R . Let ¯ x ∗ ∈ R be the minimizer of ϕ ∣ ∣ L n . Then by Lemma, there exists a point, ¯ x n ∈ R such that ϕ ∣ ∣ L n (¯ x n ) = c and | ¯ x ∗ -¯ x n | ≤ 2 c ∥∇ ϕ ( y n ) ∥ 2 . Observe that ¯ x ∗ corresponds to x ⋆ . Also, ¯ x n can be mapped to a point x n ∈ R d which lies on the line L n and ϕ ( x n ) = c and | x ⋆ -x n | ≤ 2 c ∥∇ ϕ ( y n ) ∥ 2 . This holds for each y n .

Since ∥∇ ϕ ( y n ) ∥ 2 is unbounded we can find an N such that | x ⋆ -x N | &lt; δ . Therefore, | ϕ ( x ⋆ ) -ϕ ( x n ) | = | ϕ ( x ⋆ ) -c | ≤ ϵ . However, this contradicts the fact that | ϕ ( x ⋆ ) -c | &gt; ϵ .

Lemma F.25. The maximum of a constant and an absolutely convex function is absolutely convex.

Proof. Let α ∈ R and f be absolutely convex and g := max { f, α } . We split the argument in some sub-steps.

(Trivial case) Since absolutely convex functions are always positive, it follows that if α ≤ 0 then g ( x ) = max { f ( x ) , α } ≡ f ( x ) and f = g is absolutely convex.

(Second case) Let α &gt; 0 . Since f is absolutely convex, it is convex and g is by construction. Therefore, the positive side of the inequality in absolute convexity needs not to be verified. It reamains to show that:

<!-- formula-not-decoded -->

For convenience, we will show the last version is positive for different choices of x, y . Recall that f is always positive so for arbitrary ( x, y ) there are four regions identified by the strips [0 , α ); [ α, ∞ ) iover which the values f ( x ) , f ( y ) can fall.

Additionally, recognize that for z ∈ R d we have ∇ g ( z ) = ∇ max { f ( z ) , a } = 1 f ( z ) &gt;a ∇ f ( z ) . Let us treat all the cases in separate ways. 4

If f ( x ) ≤ α and f ( y ) &lt; α we have the expression 2 α ≥ 0 by construction.

If f ( x ) &gt; α and f ( y ) &lt; α we have the expression:

<!-- formula-not-decoded -->

again, by construction.

If f ( x ) &lt; α and f ( y ) &gt; α we have a + f ( y )+ ⟨∇ f ( y ) , x -y ⟩ &gt; f ( x )+ f ( y )+ ⟨∇ f ( y ) , x -y ⟩ ≥ 0 since we assumed f is absolutely convex.

If f ( x ) ≥ α and f ( y ) &gt; α one has:

<!-- formula-not-decoded -->

which follows by the assumed strong convexity of f .

̸

̸

remark Let x ⋆ = arg min f , for an absolutely convex function f . Observe that one can always use, for any y = x ⋆ , x = x ⋆ :

<!-- formula-not-decoded -->

Lemma F.26. Suppose v ∈ R d . A function ϕ : R d → R is absolutely convex if and only if the function f : R → R defined as f ( t ) = ϕ ( x + tv ) is absolutely convex for all x ∈ R d .

Proof.

( ⇒ ) Suppose f ( t ) is an absolutely convex function for any x, v ∈ R d . We already know that ϕ will be convex so we only need to show that for all y, z ∈ R d ,

<!-- formula-not-decoded -->

Note that f ′ ( t ) = ⟨∇ ϕ ( x + tv ) , v ⟩ . Select x = y and v = z -y . Since f is absolutely convex, the following inequality will hold,

<!-- formula-not-decoded -->

We have our result because f (1) = ϕ ( y + v ) = ϕ ( z )

( ⇐ ) Suppose ϕ is absolutely convex. Let x, v ∈ R d be arbitrary. We know already that f is convex. So we just need to show that for any s, t ∈ R we have that

<!-- formula-not-decoded -->

By absolute convexity of ϕ we know that,

<!-- formula-not-decoded -->

Since f ( t ) = ϕ ( x + tv ) we have our result.

4 In principle, at z = a there is a singularity, we avoid doing this computation since the non-differentiable definition of absolute convexity is satisfied for the max function.

## G Extra Experiments

## G.1 Regression with squared Huber loss

In this experiment we optimize the function

<!-- formula-not-decoded -->

where a i ∈ R d , b i ∈ R are the data samples associated with a regression problem, and h δ is the Huber loss function. We run the experiments with C ( x ) for absolutely convex functions 7.1.

<!-- formula-not-decoded -->

Figure 5: Regression on housing dataset.

<!-- image -->

Figures 5-6 show that the algorithms using the C ( x ) matrix perform much better than the Polyak method. We observe very fast convergence of both LCD3 and LCD2 , regardless of δ . Contrary, as δ increases LCD1 loses in comparison with the other two matrix methods. The most likely reason is increasing part of the objective, which is quartic, as it requires extra adaptiveness on the smoothness constant.

## G.2 Ridge regression

We consider the following objective function:

<!-- formula-not-decoded -->

where a i ∈ R d , b i ∈ R are the data samples associated with a regression problem. By L we understand the smoothness constant of a linear regression instance, excluding the regularizer.

In Figures 7-8, C ( x ) is associated with the regularizer, and it becomes a multiple of I . As discussed in the main text, in this case LCD2 has closed-form solution, which coincides with LCD3 . The LCD1 algorithm becomes GD . We can see, similar behavior to logistic regression with L 2 regularizer that is consistent improvement of LCD2 over the Polyak's method.

Figures 9-10 show the results with C ( x ) = 2 n A ⊤ A , which is a lower bound on the main part of the objective. In this circumstance, LCD1 becomes the Newton's method, and converges in one step. As anticipated in the main text, LCD3 can diverge. Finally, LCD2 performs in a very consistent way, and converges in exactly 15 steps across all the setups.

103

102

101

100

10-1

10-2

10-3

10-4

10-5

10-6

103

102

101

=

100

1

10-1

10-2

10-3

I1xk

10-4

10-5

106

0

0

0

50

(a)

50

(a)

5

100

λ

100

λ

10

15

20

25

30

Iterations

(a)

λ

=

L

10

·

3

-

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should

150

200

Iterations

=

L

250

·

150

200

Iterations

=

L

·

-

10

250

-

10

300

3

300

3

Polyak

LCD1

LCD2

LCD3

Figure 7: Ridge regression on housing dataset; C ( x ) = 2 λ .

103

102

101

100

=

10-1

10-2

10-3

10-4

10-5

10-6

103

102

101

100

10-1

10-2

10-3

10-4

10-5

106

350

Polyak

10

20

(c)

λ

LCD3

Iterations

25

50

75

(b)

100

=

λ

·

LCD1

150

125

-

10

175

2

LCD2

Figure 9: Ridge regression on housing dataset; C ( x ) = 2 n A ⊤ A .

<!-- image -->

Figure 10: Ridge regression on mpg dataset; C ( x ) = 2 n A ⊤ A .

108

106

104

102

zll.x.

#

100

10-2

10-4

10-6

106

104

102

100

10-2

10-4

10-6

L

3

N

350

Polyak

25

75100125

150175

Iterations

(b)

λ

=

·

LCD1

10

2

-

LCD2

10

20

(c)

λ

LCD3

Figure 8: Ridge regression on mpg dataset; C ( x ) = 2 λ .

103

102

101

=

100

101

10-2

10-3

10-4

10-5

10-6

103

102

101

=

100

10-1

10-2

10-3

10-4

10-5

10-6

L

3

0

0

0

5

(b)

50

10

Iterations

λ

=

15

L

3

·

20

10

25

-

2

200

30

1

106

104

102

zll.x

100

10-2

10-4

106

0

0

0

5

(c)

30

40

50

Iterations

=

L

·

10

30

40

50

60

-

60

Iterations

=

L

·

10

Iterations

λ

=

15

L

20

·

25

-

10

2

10

-

70

2

70

2

80

80

30

follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (1âC'2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

IMPORTANT, please:

- Delete this instruction block, but keep the section heading 'NeurIPS Paper Checklist" ,
- Keep the checklist subsection headings, questions/answers and guidelines below.
- Do not modify the questions and only use the provided macros for your answers .

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: All claims made in the abstract about our methods have proofs and our empirical results are encouraging.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We point out assumptions and limitations in the conclusion.

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

Justification: All assumptions are stated in the introduction and proofs are provided in appendix.

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

Justification: The paper provides a thorough description of the experimental setup including the type of model that was used and all of the hyperparameters.

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

Answer: [No]

Justification: Although we don't provide code, we use common and standard experimental setups in ML such as logistic regression. Additionally, we provide a thorough description of our methods and the experimental setup so the results can be faithfully reproduced.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so âC´ sNoâCt' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: All experimental details are in the experiments section and additional technical details are in the appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: Our paper focuses on convex optimization algorithms, which typically do not employ statistical significant tests. We demonstrate the empirical performance of our methods across various ML setups against baselines.

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

Justification: Compute resources used are provided in the experiments section.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research is in convex optimization theory and adheres to the NeurIPS code of conduct.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

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

Justification: Not applicable to our research.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: We do not use existing assets.

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

Justification: We do not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our work does not involve crowdsourcing.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our paper does not involve research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: Does not involve LLMs.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.