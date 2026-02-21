## Quantum Speedups for Minimax Optimization and Beyond

## Chengchang Liu

The Chinese University of Hong Kong 7liuchengchang@gmail.com

## Jialin Zhang

## Zongqi Wan ∗

Great Bay University zqwan@gbu.edu.cn

## Xiaoming Sun

Institute of Computing Technology, CAS zhangjialin@ict.ac.cn

Institute of Computing Technology, CAS sunxiaoming@ict.ac.cn

## John C.S. Lui

The Chinese University of Hong Kong cslui@cse.cuhk.edu.hk

## Abstract

This paper investigates convex-concave minimax optimization problems where only the function value access is allowed. We introduce a class of Hessianaware quantum zeroth-order methods that can find the /epsilon1 -saddle point within ˜ O ( d 2 / 3 /epsilon1 -2 / 3 ) function value oracle calls. This represents an improvement of d 1 / 3 /epsilon1 -1 / 3 over the O ( d/epsilon1 -1 ) upper bound of classical zeroth-order methods, where d denotes the problem dimension. We extend these results to µ -stronglyconvex µ -strongly-concave minimax problems using a restart strategy, and show a speedup of d 1 / 3 µ -1 / 3 compared to classical zeroth-order methods. The acceleration achieved by our methods stems from the construction of efficient quantum estimators for the Hessian and the subsequent design of efficient Hessian-aware algorithms. In addition, we apply such ideas to non-convex optimization, leading to a reduction in the query complexity compared to classical methods.

## 1 Introduction

We consider the following unconstrained minimax problem:

<!-- formula-not-decoded -->

where h ( · , · ) is convex in x and concave in y . Let d = m + n , z = ( x /latticetop , y /latticetop ) /latticetop ∈ R d , and define f ( z ) /defines h ( x , y ) . The above problem has received considerable attention in the field of machine learning due to its wide applications, including fairness-aware learning [52, 32], AUC maximization [51, 20], robust optimization [3], game theory [48], and reinforcement learning [15, 43].

There are numerous classical algorithms to solve the convex-concave minimax problem (1). Firstorder methods, such as the optimistic gradient descent ascent (OGDA) [44, 36], extra gradient method (EG) [27, 38], along with their variants [40, 39] find an /epsilon1 -saddle point within O ( /epsilon1 -1 ) gradient oracle queries when h ( · , · ) is smooth. When ∇ 2 h ( · , · ) is Lipschitz continuous, second-order

∗ The corresponding author.

methods offer faster convergence rates. The Newton proximal extra gradient method (NPE) [37] and its cubic-regularized realization [31, 22] require O ( /epsilon1 -2 / 3 ) queries to the second-order oracle. Very recently, Chen et al. [11] further improved the second-order oracle complexity to ˜ O ( /epsilon1 -4 / 7 ) . These results have been generalized to the cases where the p -th order derivative of h ( · , · ) is Lipschitz, with query complexity of O ( /epsilon1 -p p +1 ) to the p -th order oracle [30, 23, 24].

However, access to the gradient oracle or higher-order oracles of h ( · , · ) is not always available, as the calculation of the exact gradient (or higher order) information can be expensive or even infeasible [42, 35, 25, 53]. This necessitates the design of efficient derivative-free algorithms to solve equation (1). Beznosikov et al. [4] proposed a gradient-free algorithm with O ( d/epsilon1 -2 ) query complexity to h ( · , · ) for the smooth convex-concave minimax problem (1). Subsequently, this rate was improved to O ( d/epsilon1 -1 ) by Sadiev et al. [45], whose dependence on /epsilon1 -1 matches the lower bound for first-order methods [54].

The aforementioned methods for minimax problems are all designed for classical computing machines. However, the advantage of quantum computing has been investigated for various optimization problems. Accessing quantum counterparts of the classical oracles always leads to better query complexities or even a breakthrough over the classical lower bounds, including convex [47, 8, 9, 55] or non-convex optimization [46, 19], semi-definite programming [5, 6], non-smooth optimization [8, 33, 28], stochastic optimization [41, 55, 46], and so on. There are also some previous studies on some specific minimax problems, including the zero-sum game [29, 16] and minimizing the maximal loss [49]. However, it remains an open question whether quantum speedup is available for general convex-concave minimax problems. Thus, the first question we aim to address is:

Can quantum zeroth-order methods be designed to surpass the O ( d/epsilon1 -1 ) query complexity of classical zeroth-order methods, thereby demonstrating a quantum speedup for general convex-concave minimax problems?

A natural approach to achieve this is to leverage the fast gradient estimator [26], which can approximate the gradient of a smooth objective function within ˜ O (1) queries to the quantum function value oracle, based on the quantum Fourier transform [50]. In convex optimization, employing the fast gradient estimator within cutting-plane algorithms and gradient descent leads to improved query complexities compared to classical zeroth-order methods. Specifically, the quantum cuttingplane method [8] and the quantum gradient descent method [2] achieve query complexities of ˜ O ( d ) and ˜ O ( /epsilon1 -1 ) , respectively, improving upon the ˜ O ( d 2 ) and ˜ O ( d/epsilon1 -1 ) complexities of classical zerothorder methods by a factor of d . For non-convex optimization, a similar strategy has been used in the design of quantum perturbed AGD [19], resulting in an improved query complexity of ˜ O ( /epsilon1 -7 / 4 ) compared to the ˜ O ( d/epsilon1 -7 / 4 ) complexity of the classical zeroth-order method [53]. These methods use the quantum gradient estimators to achieve the same query complexities as classical first-order methods by accessing only zeroth-order oracles, which reduces the dependency on d . On the other hand, the oracle complexity of classical second-order methods enjoys a better dependency on /epsilon1 -1 when compared to the classical first-order methods. This motivates us to ask:

Can we go beyond quantum estimation for the gradient and design efficient Hessian-aware quantum zeroth-order algorithms with better dependence on both /epsilon1 -1 and d ?

In this paper, we provide an affirmative answer to the above two questions. To this end, we develop a quantum estimator for the Hessian matrix using the finite difference method and design a novel Hessian-aware zeroth-order optimization framework. We summarize our contributions as follows.

- For the convex-concave problem, we propose a Hessian-aware quantum zeroth-order method (HAQZO), with query complexity of ˜ O ( d/epsilon1 -2 / 3 ) to find the /epsilon1 -saddle point, which surpasses the classical zeroth-order method by a factor of /epsilon1 -1 / 3 . We further accelerate such query complexity to ˜ O ( d + d 2 / 3 /epsilon1 -2 / 3 ) by proposing a double-loop Hessian-aware quantum method (HAQZO + ) that can reuse the Hessian estimators. HAQZO + accelerates the classical algorithms in terms of /epsilon1 -1 and d . We compare HAQZO and HAQZO + with the existing method in Table 1. The detailed analysis of HAQZO and HAQZO + can be found in Sections 4.1 and 4.2, respectively.
- For the strongly-convex-strongly-concave problem, we apply the restart strategy in HAQZO + and propose Restart-HAQZO + . We prove that Restart-HAQZO + finds the /epsilon1 -point with query complexity of ˜ O ( d + d 2 / 3 ( L 2 /µ ) 2 / 3 ) , outperforms the classical method by a factor d 1 / 3 µ -1 / 3 .

Table 1: We summarize the complexities of function value oracles to find the /epsilon1 -saddle point (c.f. Section 2) for the convex-concave minimax problem (1).

| Methods             | Oracle    | Query Complexity                  | Reference             |
|---------------------|-----------|-----------------------------------|-----------------------|
| ZOSPA               | classical | O ( d/epsilon1 - 2 )              | Beznosikov et al. [4] |
| ZOVIA               | classical | O ( d/epsilon1 - 1 )              | Sadiev et al. [45]    |
| HAQZO Algorithm 3   | quantum   | ˜ O ( d/epsilon1 - 2 / 3 )        | Theorem 4.3           |
| HAQZO + Algorithm 4 | quantum   | ˜ O d + d 2 / 3 /epsilon1 - 2 / 3 | Theorem 4.5           |

(

)

Table 2: We summarize the complexities of function value oracles to find the /epsilon1 -point for µ -stronglyconvexµ -strongly-concave minimax problem (1), i.e. ‖ z -z ∗ ‖ 2 ≤ /epsilon1 . We use L i ( i = 1 , 2 ) denotes the Lipschitz continuous parameter of i -th order derivatives of f ( · ) .

(

| Methods                     | Oracle    | Query Complexity                 | Reference          |
|-----------------------------|-----------|----------------------------------|--------------------|
| ZOVIA                       | classical | ˜ O ( dL 1 /µ )                  | Sadiev et al. [45] |
| Restart-HAQZO + Algorithm 5 | quantum   | ˜ O d + d 2 / 3 ( L 2 /µ ) 2 / 3 | Theorem 4.8        |

)

Table 3: We summarize the complexities of function value oracles to find the /epsilon1 -stationary point of non-convex minimization problem (10), i.e. ‖∇ f ( z ) ‖ ≤ /epsilon1 . We use d to denote the dimension of the problem.

| Methods          | Oracle    | Query Complexity                        | Reference                 |
|------------------|-----------|-----------------------------------------|---------------------------|
| GFM              | classical | O ( d/epsilon1 - 7 / 4 )                | Zhang and Gu [53]         |
| DF-CNM           | classical | ˜ O ( d 2 /epsilon1 - 3 / 2 )           | Cartis et al. [7]         |
| Zero-Order CNM   | classical | ˜ O ( d 2 + d 3 / 2 /epsilon1 - 3 / 2 ) | Doikov and Grapiglia [13] |
| Q-Perturbed-AGD  | quantum   | ˜ O ( /epsilon1 - 7 / 4 )               | Gong et al. [19]          |
| QCNM Algorithm 6 | quantum   | ˜ O d + d 1 / 2 /epsilon1 - 3 / 2       | Theorem 5.2               |

(

)

The comparison of the query complexities can be found in Table 2 and the detailed analysis is presented in Section 4.3.

- We further generalize the design of Hessian-aware quantum methods to solve non-convex problems with Lipschitz continuous Hessian. We propose the quantum cubic regularized-Newton method (QCNM) with query complexity of ˜ O ( d + d 1 / 2 /epsilon1 -3 / 2 ) to find the /epsilon1 -stationary point, which is better than all classical zeroth-order algorithms. The proposed QCNM method also enjoys an improved quantum query complexity over the existing state-of-the-art quantum algorithm when d = O ( /epsilon1 -1 / 2 ) , demonstrating the power of designing Hessian-aware quantum algorithms. We compare QCNM with existing classical algorithms and quantum algorithms in Table 3 and present the results in Section 5.

## 2 Preliminaries

We make the following assumptions on f ( z ) /defines h ( x , y ) .

Assumption 2.1. We assume f ( z ) = h ( x , y ) is convex in x and concave in y .

Assumption 2.2. We assume the f ( · ) , ∇ f ( · ) , ∇ 2 f ( · ) are L 0 , L 1 , and L 2 -Lipschitz continuous, respectively, i.e. we have | f ( z ) -f ( z ′ ) | ≤ L 0 ‖ z -z ′ ‖ , ‖∇ f ( z ) -∇ f ( z ′ ) ‖ ≤ L 1 ‖ z -z ′ ‖ , and for any z , z ′ ∈ R d .

<!-- formula-not-decoded -->

We aim to find the approximate saddle point [39, 31, 12], which is defined as follows.

be the saddle point of function f ( · ) . For a given point ˆ z /defines [ ˆ x ˆ y ] , we let β sufficiently large such that max {‖ ˆ x -x ∗ ‖ , ‖ ˆ y -y ∗ ‖} ≤ β holds, we define the restricted gap function as

Definition 1 (Nesterov [39]) . Let B β ( w ) be the ball centered at w with radius β . Let z ∗ /defines [ x ∗ y ∗ ]

<!-- formula-not-decoded -->

We call ˆ z an /epsilon1 -saddle point if Gap(ˆ z ; β ) ≤ /epsilon1 and β = Ω( ‖ z 0 -z ∗ ‖ ) .

diag ( I m , -I n ) . The Jacobian of F ( · ) can be written as ∇ F ( x ) = J ∇ 2 F ( x ) . The following proposition shows that F ( · ) is monotone if f ( · ) satisfies Assumption 2.1.

In the following context, we define F ( · ) as F ( z ) /defines J ∇ f ( z ) = [ ∇ x h ( x , y ) -∇ y h ( x , y ) ] , where J =

Proposition 2.3 (Lemma 2.7 [30]) . If f satisfies Assumption 2.1, then for all z , z ′ ∈ R d it holds that 〈 F ( z ) -F ( z ′ ) , z -z ′ 〉 ≥ 0 . For a given ˆ z ∈ R d , its gap can be bounded by Gap(ˆ z ; β ) ≤ max z ∈ B √ 2 β ( z ∗ ) 〈 F (ˆ z ) , ˆ z -z 〉 .

We define the quantum evaluation oracle for a function f ( · ) .

Definition 2.4 (Quantum Function Evaluation Oracle) . A quantum evaluation oracle for a function f is defined as the following unitary transformation

<!-- formula-not-decoded -->

Here ⊕ is the bit-wise XOR operation. We say that we have a quantum evaluation oracle for f with accuracy /epsilon1 0 if we have a quantum evaluation oracle for f , such that | f ( z ) -f ( z ) | ≤ /epsilon1 0 for all z .

˜ ˜ Remark 2.5 . The quantum advantages are stated in terms of query complexity on the function evaluation oracle. In many situations, query complexity dominates the computational complexity of the algorithm, which is a natural setting in both classical and quantum optimization. For example, considering the generalized linear model such that f ( x ) = h ( A T x ) where A ∈ R d × n . The circuit implementation of the oracle of f may involve dominating computational complexity if n /greatermuch d in this example, and our algorithm achieves meaningful quantum speedups under such a setting.

## 3 Gradient and Hessian Estimation via Quantum Function Evaluation Oracle

Before introducing our quantum algorithms, we first introduce the quantum estimators for the gradient and Hessian of the objective function by using the quantum evaluation oracle on f ( · ) , which are the critical components of our methods. These results are natural and direct extensions of Jordan's method [26] for the smooth objective. We do not consider them to be our primary technical contribution, but state them for completeness.

## 3.1 Quantum Gradient Estimator

Quantum gradient estimator is first proposed in [26] for the smooth objective, and its rigorous statement is given by Gilyén et al. [18], Chakrabarti et al. [8], van Apeldoorn et al. [47]. The following is one of the statements.

Lemma 3.1 (Lemma 2.2 [8]) . Let f be an L 0 -Lipschitz continuous and L 1 -smooth function. Given the access to a quantum evaluation oracle of f with /epsilon1 0 accuracy, then for /epsilon1 g ≥ /epsilon1 0 there is a quantum algorithm A ( f, /epsilon1 g , L 0 , L 1 , z ) which outputs an estimate ˜ ∇ f ( z ) of ∇ f ( z ) , satisfying that ∀ i ∈ [ d ] , Pr (∣ ∣ ∣ [ ˜ ∇ f ( z ) ] i -[ ∇ f ( z )] i ∣ ∣ ∣ ≥ 1500 √ L 1 d/epsilon1 g ) ≤ 1 3 . Moreover, the A algorithm uses O (1) queries to the quantum evaluation oracle and O ( d log L 0 dL 1 /epsilon1 g ) quantum gates.

The following lemma allows for an arbitrarily small failure probability δ ∈ (0 , 1) to the quantum gradient estimator, which generalizes the results above.

Lemma 3.2 (Quantum Gradient Estimator) . Let f ( · ) be a L 0 -Lipschitz continuous and L 1 -smooth function. Given the access to a quantum evaluation oracle of f with /epsilon1 0 accuracy, then for /epsilon1 g ≥ /epsilon1 0 , there exists a quantum algorithm QuantumGradient ( f, /epsilon1 g , L 0 , L 1 , z , δ ) which outputs an estimate ˜ ∇ f ( z ) of ∇ f ( z ) , satisfying

<!-- formula-not-decoded -->

∥ ˜ ∥ Moreover, QuantumGradient uses O (log( d δ )) queries to U f and O ( d log( L 0 dL 1 /epsilon1 g ) log( d δ ) ) gates.

## 3.2 Quantum Hessian-vector Estimator and Quantum Hessian Estimator

In this section, we show that the Hessian vector product of a smooth object function can also be constructed within the ˜ O (1) quantum function evaluation oracle. Furthermore, since ∇ 2 f ( z ) = [ ∇ 2 f ( z ) e 1 , · · · , ∇ 2 f ( z ) e d ] , the Hessian of a smooth object function can be constructed within the ˜ O ( d ) quantum function evaluation oracle.

We formally present our construction of the quantum Hessian vector product estimator in Algorithm 1 and state its complexity in the following lemma.

## Algorithm 1 QuantumHessianVector( f, /epsilon1 hv , L 0 , L 1 , L 2 , z , v , δ )

- 1: M = ‖ v ‖ 2
- 3: ˜ ∇ f ( z ) := QuantumGradient ( f, /epsilon1 hv , L 0 , L 1 , z , δ/ 2) 4: ∇ f ( z +∆ v ) := QuantumGradient ( f, /epsilon1 hv , L 0 , L 1 , z +∆ · v , δ/ 2)
- 2: ∆ = 20 √ 15 /epsilon1 1 / 4 hv M -1 / 2 L -1 / 2 2 L 1 / 4 1 d 1 / 2
- ˜ 5: Return 1 ∆ ( ˜ ∇ f ( x +∆ v ) -˜ ∇ f ( x ) )

<!-- formula-not-decoded -->

Lemma 3.3 (Quantum Hessian Vector Estimator) . Suppose f satisfies Assumption 2.2. Given the access to U f with /epsilon1 0 accuracy, let hv = QuantumHessianVector ( f, /epsilon1 hv , L 0 , L 1 , L 2 , z , v , δ ) be the output of Algorithm 1 where /epsilon1 hv ≥ /epsilon1 0 , then it holds that

Remark 3.4 . He et al. [21] proposed a quantum estimator for a row of a Hessian, which can be viewed as a special case of our quantum Hessian vector estimator. Besides, they do not provide results on the query complexity and gate complexity.

∥ ∥ Moreover, Algorithm 1 uses O (log( d δ )) queries to U f and O ( d log( L 0 dL 1 /epsilon1 hv ) log( d δ ) ) gates.

Given the Hessian vector estimator, we are ready to construct the Hessian estimator by calculating the estimators of the Hessian vector set {∇ 2 f ( x ) e i } i ∈ [ d ] , which is formally given in Algorithm 2. The following results show how well the output of Algorithm 2 approximates ∇ 2 f ( z ) .

<!-- formula-not-decoded -->

∥ ˜ ∥ Moreover, Algorithm 2 uses O ( d log( d δ )) queries to U f and O ( d 2 log( L 0 dL 1 /epsilon1 H ) log( d δ ) ) gates.

Lemma 3.5 (Quantum Hessian Estimator) . Suppose f satisfies Assumption 2.2. Given access to a quantum evaluation oracle of f with /epsilon1 0 accuracy and /epsilon1 H ≥ /epsilon1 0 , let ˜ ∇ 2 f ( z ) = QuantumHessian ( f, /epsilon1 H , L 0 , L 1 , L 2 , z , δ ) be the output of Algorithm 2, then it holds that

## Algorithm 2 QuantumHessian( f, /epsilon1 H , L 0 , L 1 , L 2 , z , δ )

- 1: H = 0 d × d
- 3: H [ i, :] = QuantumHessianVector( f, /epsilon1 H , L 0 , L 1 , L 2 , z , e i , δ/d )
- 2: for i ∈ [ d ]
- 4: ˜ H = 1 2 ( H + H /latticetop )
- 5: Return ˜ H

## Algorithm 3 HAQZO ( z 0 , T, L 0 , L 1 , L 2 , δ )

- 1: for t = 0 , · · · , T -1 do
- 2: Choose /epsilon1 1 ,t &gt; 0 and /epsilon1 H ,t &gt; 0
- 3: ˜ g t = QuantumGradient ( f, /epsilon1 1 ,t , L 0 , L 1 , z t , δ/ (3 T )) and g t = J ˜ g t
- 5: Compute the inexact cubic step i.e. find z t +1 / 2 that satisfies
- 4: ˜ H t = QuantumHessian ( f, /epsilon1 H ,t , L 0 , L 1 , L 2 , z t , δ/ (3 T )) and H t = J ˜ H t

<!-- formula-not-decoded -->

- 6: λ t = 6 ( L 2 ‖ z t -z t +1 / 2 ‖ + √ 1500 d 1 / 2 L 1 / 4 1 /epsilon1 1 / 4 1 ,t + √ 1500 d 2 L 1 / 4 1 L 1 / 2 2 /epsilon1 1 / 4 H ,t ) ) . 7: Choose /epsilon1 2 ,t &gt; 0
- 8: ˜ v t +1 / 2 = QuantumGradient ( f, /epsilon1 2 ,t , L 0 , L 1 , z t +1 / 2 , δ/ (3 T )) and v t = J ˜ v t
- 9: z t +1 = z t -λ -1 t v t .
- 10: end for
- 11: return ¯ z T = 1 ∑ T -1 t =0 λ -1 t ∑ T -1 t =0 λ -1 t z t +1 / 2 .

Remark 3.6 . We note an independent work by Zhang and Shao [56], who also employed the finite difference method to construct a Hessian estimator for the more general class of complex analytical functions. In contrast to our estimator, which is designed for smooth real functions, their approach utilizes the more sophisticated spectral method to handle the complex case. On the other hand, our theoretical error bound is measured using the spectral norm ( ‖ · ‖ 2 ), while the bound in [56] is given in the infinity norm ( ‖ · ‖ ∞ ).

## 4 Quantum Speedups for Minimax Optimization

In this section, we introduce quantum algorithms to find the /epsilon1 -saddle point for general convexconcave minimax problems. In Section 4.1, we propose a Hessian-aware algorithm with ˜ O ( d/epsilon1 -2 / 3 ) queries to the quantum function evaluation oracle, which outperforms the classical state-of-the-art algorithm by a factor of /epsilon1 -1 / 3 . We further improve such query complexity to ˜ O ( d 2 / 3 /epsilon1 -2 / 3 ) , which outperforms the classical algorithm by a factor of d 1 / 3 /epsilon1 -1 / 3 , by proposing a double-loop algorithm that reuses the Hessian estimators in Section 4.2. In Section 4.3, we generalize our results to stronglyconvex-strongly-concave problems.

## 4.1 Hessian-Aware Quantum Algorithm with Better Dependency on /epsilon1 -1

Our idea is to use the quantum gradient estimator and the quantum Hessian estimator to obtain a close approximation of F ( z ) and ∇ F ( z ) and then apply the Newton proximal extragradient framework [37]. We present our Hessian-aware quantum zeroth-order method (HAQZO) in Algorithm 3.

To analyze Algorithm 3, we first consider the following generalized NPE update:

where g t , v t , and H t are some approximations to F ( z t ) , F ( z t +1 / 2 ) , and ∇ F ( z t ) , which satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The following lemma shows that the update of (6) recovers the convergence rates of the NPE method if δ 1 ,t , δ 2 ,t , and δ H ,t are small enough.

```
Algorithm 4 HAQZO + ( z 0 , T, L 0 , L 1 , L 2 , M, m, δ ) 1: for t = 0 , · · · , T -1 do 2: if t mod m = 0 do 3: Choose /epsilon1 H > 0 4: ˜ H = QuantumHessian ( f, /epsilon1 H ,t , L 0 , L 1 , L 2 , z t , δ/ (3 T )) and H = J ˜ H 5: end if 6: Choose /epsilon1 1 ,t > 0 7: ˜ g t = QuantumGradient ( f, /epsilon1 1 ,t , L 0 , L 1 , z t , δ/ (3 T )) and g t = J ˜ g t 8: Compute the inexact cubic step i.e. find z t +1 / 2 that satisfies g t + ( H +6( M ‖ z t -z t +1 / 2 ‖ + √ 1500 d 1 / 2 L 1 / 4 1 /epsilon1 1 / 4 1 ,t + √ 1500 d 2 L 1 / 4 1 L 1 / 2 2 /epsilon1 1 / 4 H ) I ) ( z t +1 / 2 -z t )= 0 9: λ t = 6 ( M ‖ z t -z t +1 / 2 ‖ + √ 1500 d 1 / 2 L 1 / 4 1 /epsilon1 1 / 4 1 ,t + √ 1500 d 2 L 1 / 4 1 L 1 / 2 2 /epsilon1 1 / 4 H ) ) . 10: Choose /epsilon1 2 ,t > 0 11: ˜ v t +1 / 2 = QuantumGradient ( f, /epsilon1 2 ,t , L 0 , L 1 , z t +1 / 2 , δ/ (3 T )) and v t = J ˜ v t 12: z t +1 = z t -λ -1 t v t . 13: end for 14: return ¯ z T = 1 ∑ T -1 t =0 λ -1 t ∑ T -1 t =0 λ -1 t z t +1 / 2 .
```

<!-- formula-not-decoded -->

Remark 4.2 . We note that some prior works have also studied the inexact NPE methods [31, 1]. However, these methods only consider the case where the Hessian is inexact, while our Lemma 4.1 allows inexactness from both the gradient and the Hessian.

Because the iteration rule of Algorithm 3 can be interpreted as the generalized NPE update in (6) with high probability, we can determine the query complexity of Algorithm 3 by incorporating the quantum gradient and Hessian estimators. This result is formally stated in the following theorem.

Theorem 4.3. Under Assumptions 2.1 and 2.2, let R = Ω( ‖ z 0 -z ∗ ‖ ) , given desired accuracy /epsilon1 &gt; 0 , we run Algorithm 3 with

<!-- formula-not-decoded -->

then with probability at 1 -δ , Algorithm 3 finds the /epsilon1 -saddle point of f ( · ) with ˜ O ( dL 2 / 3 2 R 2 /epsilon1 -2 / 3 ) queries to U f , where ˜ O ( · ) hides the polylogarithm dependency on d , L 0 , L 1 , L 2 , /epsilon1 -1 , δ -1 , and R .

## 4.2 Hessian-Aware Quantum Algorithm with Better Dependency on /epsilon1 -1 and d

In this section, we further improve the query complexity of HAQZO by proposing a double-loop Hessian-aware quantum method HAQZO + in Algorithm 4, which is inspired by the recent advance in lazy Hessian methods [14, 12, 10, 34].

The main difference between Algorithm 4 and Algorithm 3 is that we eliminate calling quantumHessian in every iteration, but only call it at the snapshot point in iterations t when t mod m = 0 , and reuse such Hessian estimator in the next m iterations. In addition, we replace L 2 with a larger parameter M ≥ L 2 on Line 9 and Line 10 and tune it to guarantee convergence. We

Algorithm 5 Restart-HAQZO + ( z 0 , T, L 0 , L 1 , L 2 , M, m, S, δ )

- 1: z (0) = z 0
- 2: for s = 0 , · · · , S -1
- 3: z ( s +1) = HAQZO + ( z ( s ) , T, L 0 , L 1 , L 2 , M, m, δ/S )
- 4: end for
- 5: return z ( S )

first consider the following iteration rule

<!-- formula-not-decoded -->

where π ( t ) /defines t -( t mod m ) and g t , v t , H π ( t ) are some approximations to F ( z t ) , F ( z t +1 / 2 ) , ∇ F ( z π ( t ) ) such that

<!-- formula-not-decoded -->

The following lemma shows that the update of (8) still enjoys the rate of T -3 / 2 if the regularization term λ t is chosen large enough and δ 1 ,t , δ 2 ,t , and δ H are small.

<!-- formula-not-decoded -->

The iteration rule of Algorithm 4 can also be interpreted as (8) with high probability. At each iteration, the algorithm calls QuantumGradient with ˜ O (1) quantum function evaluation queries to obtain g t and v t . Every m iterations, the algorithm calls QuantumHessian with ˜ O ( d ) quantum function evaluation queries to obtain H t . The following theorem provides the query complexity of Algorithm 4 with a proper choice of m = d .

Theorem 4.5. Under Assumptions 2.1 and 2.2, let R = Ω( ‖ z 0 -z ∗ ‖ ) , given desired accuracy /epsilon1 &gt; 0 , we run Algorithm 4 with

<!-- formula-not-decoded -->

then Algorithm 4 finds the /epsilon1 -saddle point of f ( · ) with ˜ O ( d + d 2 / 3 L 2 / 3 2 R 2 /epsilon1 -2 / 3 ) queries to U f with probability at 1 -δ , where ˜ O ( · ) hides the polylogarithm dependency on d , L 0 , L 1 , L 2 , /epsilon1 -1 , δ -1 , and R .

## 4.3 Restarted Hessian-Aware Quantum Algorithm for Strongly-Convex Strongly-Concave Minimax Optimization

In this section, we generalize our results to solve strongly-convex-strongly-concave minimax problems. We make the following assumption on f ( · ) , which is stronger than Assumption 2.1.

Assumption 4.6. We assume f ( z ) = h ( x , y ) is µ -strongly-convex in x and µ -strongly-concave in y for some µ &gt; 0 .

We apply the restart strategy which is widely used in minimization [17] and minimax optimization [22, 30, 12] on our HAQZO + and propose Restart-HAQZO + in Algorithm 5. To avoid confusion, we use the superscript ( s ) to denote the parameters in the HAQZO + subroutine in the s -th iteration of Algorithm 5. The following lemma shows that by properly choosing the parameter in HAQZO + , ‖ z ( s +1) -z ∗ ‖ 2 will descend linearly with high probability.

Lemma 4.7. Under Assumptions 2.2 and 4.6, set the parameter in subroutine HAQZO + in the s -th iteration of Algorithm 5 as follows:

<!-- formula-not-decoded -->

Lemma 4.7 means it is enough to set S = /ceilingleft log(1 //epsilon1 ) /ceilingright to obtain some z ( S ) such that ‖ z ( S ) -z ∗ ‖ 2 ≤ /epsilon1 . Given this, we are ready to present the query complexity of Algorithm 5.

Theorem 4.8. Under Assumptions 2.2 and 4.6, set the parameter in subroutine HAQZO + in the s iteration of Algorithm 5 as in Lemma 4.7 with m = d , and set S = /ceilingleft log( ‖ z (0) -z ∗ ‖ 2 //epsilon1 ) /ceilingright , then with probability at least 1 -δ , the output of Algorithm 5 satisfies that ‖ z ( S ) -z ∗ ‖ 2 ≤ /epsilon1 with ˜ O ( d + d 2 / 3 L 2 / 3 2 µ -2 / 3 ) queries to U f , where ˜ O ( · ) hides the polylogarithm dependency on d , L 0 , L 1 , L 2 , /epsilon1 -1 , δ -1 .

## 5 Extension to Non-convex Optimization

In the previous section, we have shown that, using quantum Hessian estimators, it is possible to design fast quantum algorithms which outperform the classical algorithms in terms of accuracy /epsilon1 -1 and dimension d . Wehighlight that the quantum estimators designed in Section 3 are not restricted to convex-concave minimax problems. In this section, we extend the idea of designing Hessian-aware quantum zeroth-order methods to non-convex problems

<!-- formula-not-decoded -->

Definition 5.1. We say z is an /epsilon1 -stationary point of the nonconvex minimization problem (10) if it holds that ‖∇ f ( z ) ‖ ≤ /epsilon1 .

where f ( · ) is smooth but possibly not convex. We aim to find the /epsilon1 -stationary point of (10).

Wepresent the quantum cubic-regularized Newton methods in Algorithm 6, which replace the classical gradient and Hessian estimators in the zeroth-order CNM method [13] by the quantum estimators designed in Section 3. The following theorem gives the query complexity of Algorithm 6 to find the /epsilon1 -stationary point of f ( · ) .

<!-- formula-not-decoded -->

Theorem 5.2. Under Assumption 2.2 and suppose f ∗ /defines min z ∈ R d f ( z ) &gt; -∞ , given desired accuracy /epsilon1 &gt; 0 , we run Algorithm 6 with

<!-- formula-not-decoded -->

then with probability at least 1 -δ , the output of Algorithm 6 finds the /epsilon1 -stationary point of problem 10 with ˜ O ( d + d 1 / 2 L 1 / 2 2 ( f ( z 0 ) -f ∗ ) /epsilon1 -3 / 2 ) queries to U f , where ˜ O ( · ) hides the polylogarithm dependency on d , L 0 , L 1 , L 2 , /epsilon1 -1 , δ -1 .

## 6 Conclusion

In this paper, we have proposed quantum algorithms to speed up training for minimax optimization problems. Our Hessian-aware quantum zeroth-order method reduces the query complexity of the function evaluation oracle of the classical methods by a factor of d 1 / 3 /epsilon1 -1 / 3 and d 1 / 3 µ -1 / 3 for convex-concave and strongly-convex-strongly-concave problems, respectively. Moreover, we find that the proposed quantum oracles for estimating the Hessian matrix can be used to solve other important optimization problems, i.e. non-convex optimization. However, the query complexity of the proposed Hessian-aware quantum zeroth-order methods still depends on the dimension, and the quantum lower bound for this question is still unknown. We leave this for future work.

```
Algorithm 6 QCNM ( z 0 , T, L 0 , L 1 , L 2 , M, m, /epsilon1 g , /epsilon1 H , δ ) 1: δ g = 1500 dL 1 / 2 1 /epsilon1 1 / 2 g , δ H = 1500 1 / 2 d 2 L 1 / 4 1 L 1 / 2 2 /epsilon1 1 / 4 H 2: for t = 0 , · · · , T -1 do 3: if t mod m = 0 do 4: H = QuantumHessian ( f, /epsilon1 H , L 0 , L 1 , L 2 , z t , δ/ (2 T )) 5: end if 6: g t = QuantumGradient ( f, /epsilon1 g , L 0 , L 1 , z t , δ/ (2 T )) 7: Compute the cubic step i.e. find z t +1 that satisfies z t +1 = arg min z ∈ R d { 〈 g t , z -z t 〉 + 1 2 〈 H · ( z -z t ) , z -z t 〉 + M 6 ‖ z -z t ‖ 3 } 8: end for 9: return z out uniformly from { z i } T i =1
```

## Acknowledgment

We thank the anonymous reviewers for their helpful suggestions. Zongqi Wan, Jialin Zhang, and Xiaoming Sun are supported by the National Natural Science Foundation of China Grants No. 62325210 and 12447107. Chengchang Liu is supported by the National Natural Science Foundation of China (624B2125). John C.S. Lui is supported in part by the GRF-14207721 and SRFS21224S02.

## References

- [1] Artem Agafonov, Petr Ostroukhov, Roman Mozhaev, Konstantin Yakovlev, Eduard Gorbunov, Martin Takác, Alexander Gasnikov, and Dmitry Kamzolov. Exploring jacobian inexactness in second-order methods for variational inequalities: lower bounds, optimal algorithms and quasinewton approximations. Advances in Neural Information Processing Systems , 37:115816115860, 2024.
- [2] Brandon Augustino, Dylan Herman, Enrico Fontana, Junhyung Lyle Kim, Jacob Watkins, Shouvanik Chakrabarti, and Marco Pistoia. Fast convex optimization with quantum gradient methods. arXiv preprint arXiv:2503.17356 , 2025.
- [3] Aharon Ben-Tal, Laurent El Ghaoui, and Arkadi Nemirovski. Robust optimization . Princeton university press, 2009.
- [4] Aleksandr Beznosikov, Abdurakhmon Sadiev, and Alexander Gasnikov. Gradient-free methods with inexact oracle for convex-concave stochastic saddle-point problem. In International Conference on Mathematical Optimization Theory and Operations Research , pages 105-119. Springer, 2020.
- [5] Fernando GSL Brandao and Krysta M Svore. Quantum speed-ups for solving semidefinite programs. In 2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS) , pages 415-426. IEEE, 2017.
- [6] Fernando GSL Brandão, Amir Kalev, Tongyang Li, Cedric Yen-Yu Lin, Krysta M Svore, and Xiaodi Wu. Quantum sdp solvers: Large speed-ups, optimality, and applications to quantum learning. In 46th International Colloquium on Automata, Languages, and Programming (ICALP 2019) . Schloss-Dagstuhl-Leibniz Zentrum für Informatik, 2019.
- [7] Coralia Cartis, Nicholas IM Gould, and Philippe L Toint. On the oracle complexity of firstorder and derivative-free algorithms for smooth nonconvex minimization. SIAM Journal on Optimization , 22(1):66-86, 2012.
- [8] Shouvanik Chakrabarti, Andrew M Childs, Tongyang Li, and Xiaodi Wu. Quantum algorithms and lower bounds for convex optimization. Quantum , 4:221, 2020.

- [9] Shouvanik Chakrabarti, Andrew M Childs, Shih-Han Hung, Tongyang Li, Chunhao Wang, and Xiaodi Wu. Quantum algorithm for estimating volumes of convex bodies. ACM Transactions on Quantum Computing , 4(3):1-60, 2023.
- [10] Lesi Chen, Chengchang Liu, Luo Luo, and Jingzhao Zhang. Computationally faster newton methods by lazy evaluations. arXiv preprint arXiv:2501.17488 , 2025.
- [11] Lesi Chen, Chengchang Liu, Luo Luo, and Jingzhao Zhang. Solving convex-concave problems with ˜ O ( /epsilon1 -4 / 7 ) second-order oracle complexity. The 38th Annual Conference on Learning Theory , 2025.
- [12] Lesi Chen, Chengchang Liu, and Jingzhao Zhang. Second-order min-max optimization with lazy hessians. In The Thirteenth International Conference on Learning Representations , 2025.
- [13] Nikita Doikov and Geovani Nunes Grapiglia. First and zeroth-order implementations of the regularized newton method with lazy approximated hessians. Journal of Scientific Computing , 103(1):32, 2025.
- [14] Nikita Doikov, El Mahdi Chayti, and Martin Jaggi. Second-order optimization with lazy hessians. In ICML , 2023.
- [15] Simon S. Du, Jianshu Chen, Lihong Li, Lin Xiao, and Dengyong Zhou. Stochastic variance reduction methods for policy evaluation. In ICML , 2017.
- [16] Minbo Gao, Zhengfeng Ji, Tongyang Li, and Qisheng Wang. Logarithmic-regret quantum learning algorithms for zero-sum games. Advances in Neural Information Processing Systems , 36:31177-31203, 2023.
- [17] Saeed Ghadimi, Han Liu, and Tong Zhang. Second-order methods with cubic regularization under inexact information. arXiv preprint arXiv:1710.05782 , 2017.
- [18] András Gilyén, Srinivasan Arunachalam, and Nathan Wiebe. Optimizing quantum optimization algorithms via faster quantum gradient computation. In Proceedings of the Thirtieth Annual ACM-SIAM Symposium on Discrete Algorithms , pages 1425-1444. SIAM, 2019.
- [19] Weiyuan Gong, Chenyi Zhang, and Tongyang Li. Robustness of quantum algorithms for nonconvex optimization. In The Thirteenth International Conference on Learning Representations , 2025.
- [20] James A. Hanley and Barbara J. McNeil. The meaning and use of the area under a receiver operating characteristic (ROC) curve. Radiology , 143(1):29-36, 1982.
- [21] Jianhao He, Chengchang Liu, Xutong Liu, Lvzhou Li, and John C.S. Lui. Quantum algorithm for online exp-concave optimization. In International Conference on Machine Learning , pages 17946-17971. PMLR, 2024.
- [22] Kevin Huang and Shuzhong Zhang. An approximation-based regularized extra-gradient method for monotone variational inequalities. arXiv preprint arXiv:2210.04440 , 2022.
- [23] Kevin Huang, Junyu Zhang, and Shuzhong Zhang. Cubic regularized Newton method for saddle point models: a global and local convergence analysis. arXiv preprint arXiv:2008.09919 , 2020.
- [24] Ruichen Jiang and Aryan Mokhtari. Generalized optimistic methods for convex-concave saddle point problems. arXiv preprint arXiv:2202.09674 , 2022.
- [25] Gangshan Jing, He Bai, Jemin George, Aranya Chakrabortty, and Piyush K Sharma. Asynchronous distributed reinforcement learning for lqr control via zeroth-order block coordinate descent. IEEE Transactions on Automatic Control , 2024.
- [26] Stephen P Jordan. Fast quantum algorithm for numerical gradient estimation. Physical review letters , 95(5):050501, 2005.
- [27] G. M. Korpelevich. An extragradient method for finding saddle points and for other problems. Matecon , 12:747-756, 1976.

- [28] Jiaqi Leng, Yufan Zheng, Zhiyuan Jia, Lei Fan, Chaoyue Zhao, Yuxiang Peng, and Xiaodi Wu. Quantum hamiltonian descent for non-smooth optimization. arXiv preprint arXiv:2503.15878 , 2025.
- [29] Tongyang Li, Chunhao Wang, Shouvanik Chakrabarti, and Xiaodi Wu. Sublinear classical and quantum algorithms for general matrix games. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 35, pages 8465-8473, 2021.
- [30] Tianyi Lin and Michael I. Jordan. Perseus: A simple high-order regularization method for variational inequalities. arXiv preprint arXiv:2205.03202 , 2022.
- [31] Tianyi Lin, Panayotis Mertikopoulos, and Michael I. Jordan. Explicit second-order min-max optimization methods with optimal convergence guarantee. arXiv preprint arXiv:2210.12860 , 2022.
- [32] Chengchang Liu, Shuxian Bi, Luo Luo, and John C.S. Lui. Partial-quasi-newton methods: Efficient algorithms for minimax optimization problems with unbalanced dimensionality. In Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining , pages 1031-1041, 2022.
- [33] Chengchang Liu, Chaowen Guan, Jianhao He, and John Lui. Quantum algorithms for nonsmooth non-convex optimization. Advances in Neural Information Processing Systems , 37: 35288-35312, 2024.
- [34] Chengchang Liu, Luo Luo, and John C.S. Lui. An enhanced Levenberg-Marquardt method via gram reduction. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 39, pages 18772-18779, 2025.
- [35] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations , 2018.
- [36] Aryan Mokhtari, Asuman Ozdaglar, and Sarath Pattathil. A unified analysis of extra-gradient and optimistic gradient methods for saddle point problems: Proximal point approach. In AISTATS , 2020.
- [37] Renato DC Monteiro and Benar F Svaiter. Iteration-complexity of a newton proximal extragradient method for monotone variational inequalities and inclusion problems. SIAM Journal on Optimization , 22(3):914-935, 2012.
- [38] Arkadi Nemirovski. Prox-method with rate of convergence o (1/t) for variational inequalities with lipschitz continuous monotone operators and smooth convex-concave saddle point problems. SIAM Journal on Optimization , 15(1):229-251, 2004.
- [39] Yurii Nesterov. Dual extrapolation and its applications to solving variational inequalities and related problems. Mathematical Programming , 109(2-3):319-344, 2007.
- [40] Yurii Nesterov and Laura Scrimali. Solving strongly monotone variational and quasivariational inequalities. Discrete and Continuous Dynamical Systems , 31(4):1383-1396, 2007.
- [41] Guneykan Ozgul, Xiantao Li, Mehrdad Mahdavi, and Chunhao Wang. Quantum speedups for markov chain monte carlo methods with application to optimization. arXiv preprint arXiv:2504.03626 , 2025.
- [42] Nicolas Papernot, Patrick McDaniel, Ian Goodfellow, Somesh Jha, Z Berkay Celik, and Ananthram Swami. Practical black-box attacks against machine learning. In Proceedings of the 2017 ACM on Asia conference on computer and communications security , pages 506-519, 2017.
- [43] Santiago Paternain, Miguel Calvo-Fullana, Luiz FO Chamon, and Alejandro Ribeiro. Safe policies for reinforcement learning via primal-dual methods. IEEE Transactions on Automatic Control , 68(3):1321-1336, 2022.

- [44] Leonid Denisovich Popov. A modification of the arrow-hurwicz method for search of saddle points. Mathematical notes of the Academy of Sciences of the USSR , 28(5):845-848, 1980.
- [45] Abdurakhmon Sadiev, Aleksandr Beznosikov, Pavel Dvurechensky, and Alexander Gasnikov. Zeroth-order algorithms for smooth saddle-point problems. In International Conference on Mathematical Optimization Theory and Operations Research , pages 71-85. Springer, 2021.
- [46] Aaron Sidford and Chenyi Zhang. Quantum speedups for stochastic optimization. In Thirtyseventh Conference on Neural Information Processing Systems , 2023.
- [47] Joran van Apeldoorn, András Gilyén, Sander Gribling, and Ronald de Wolf. Convex optimization using quantum oracles. Quantum , 4:220, 2020.
- [48] John Von Neumann and Oskar Morgenstern. Theory of games and economic behavior . Princeton university press, 2007.
- [49] Hao Wang, Chenyi Zhang, and Tongyang Li. Near-optimal quantum algorithm for minimizing the maximal loss. In The Twelfth International Conference on Learning Representations .
- [50] Yaakov S Weinstein, MA Pravia, EM Fortunato, Seth Lloyd, and David G Cory. Implementation of the quantum fourier transform. Physical review letters , 86(9):1889, 2001.
- [51] Yiming Ying, Longyin Wen, and Siwei Lyu. Stochastic online AUC maximization. NIPS , 2016.
- [52] Brian Hu Zhang, Blake Lemoine, and Margaret Mitchell. Mitigating unwanted biases with adversarial learning. In AIES , 2018.
- [53] Hualin Zhang and Bin Gu. Faster gradient-free methods for escaping saddle points. In The Eleventh International Conference on Learning Representations , 2022.
- [54] Junyu Zhang, Mingyi Hong, and Shuzhong Zhang. On lower iteration complexity bounds for the convex concave saddle point problems. Mathematical Programming , 194(1-2):901-935, 2022.
- [55] Yexin Zhang, Chenyi Zhang, Cong Fang, Liwei Wang, and Tongyang Li. Quantum algorithms and lower bounds for finite-sum optimization. arXiv preprint arXiv:2406.03006 , 2024.
- [56] Yuxin Zhang and Changpeng Shao. Quantum spectral method for gradient and hessian estimation. arXiv preprint arXiv:2407.03833 , 2024.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

## Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss them in Section 6.

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

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https://nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

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

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This paper focus on the theory of quantum complexities to solve minimax problem.

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

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

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

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Auxiliary Lemmas

Lemma A.1. Given a positive sequence { λ t } T -1 t =0 , if ∑ T -1 t =0 λ 2 t ≤ C , then we have ∑ T -1 t =0 1 λ t ≥ T 3 / 2 √ C .

Proof. By Holder's inequality, we have that

Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma A.2 (Lemma 4.2, [12]) . For any sequence of positive numbers { r t } t ≥ 0 , it holds for any m ≥ 2 that ∑ m -1 t =1 ( ∑ t -1 i =0 r i ) 2 ≤ m 2 2 ∑ m -1 t =0 r 2 t .

Lemma A.3 (Lemma 4.2, [14]) . For any sequence of positive numbers { r t } t ≥ 0 , it holds for any m ≥ 1 that ∑ m -1 t =1 ( ∑ t -1 i =0 r i ) 3 ≤ m 3 3 ∑ m -1 t =0 r 3 t .

## B The Proof of Section 3

## B.1 The Proof of Lemma 3.2

Proof. Running A ( f, /epsilon1 g , L 0 , L 1 , x ) for M times. Let ˜ ∇ ( m ) f ( z ) denotes the output estimates for m -th running. Then for each coordination, take the median of the output estimates, and return the resulting vector as the output of QuantumGradient ( f, /epsilon1 g , L 0 , L 1 , z , δ ) , denoted as ˜ ∇ f ( z ) .

<!-- formula-not-decoded -->

For any i ∈ [ d ] , m ∈ [ M ] , X i,m denotes the indicator random variable of the event

∣ ∣ By Lemma 3.1, we have Pr( X m,i ) ≥ 2 3 and { X m,i } M m =1 are independent random variables. By Chernoff's bound,

Let Y i denote the event that ∑ M m =1 X i,m ≥ 5 9 M , then Pr( Y i ) ≥ 1 -e -13 M 500 . If Y i happens, we have ∣ ∣ ∣ [ ˜ ∇ f ( vz )] i -[ ∇ f ( z )] i ∣ ∣ ∣ ≤ 1500 √ L 1 d/epsilon1 g . By union bound, Pr( ∩ d i =1 Y i ) ≥ 1 -d · e -13 M 500 . Under event ∩ d i =1 Y i , we have ‖ ˜ ∇ f ( z ) - ∇ f ( z ) ‖ 2 ≤ √ 1500 2 d · L 1 d/epsilon1 g = 1500 d √ L 1 /epsilon1 g . Set M := O (log( d δ )) , we have Pr( ∩ d i =1 Y i ) ≥ 1 -δ .

<!-- formula-not-decoded -->

Since we have invoked A for M times, the total query complexity is O (log( d δ )) and the total gate complexity is O ( d log L 0 dL 1 /epsilon1 g log d δ ) by Lemma 3.1.

## B.2 The Proof of Lemma 3.3

Proof. Let ˜ ∇ f ( x ) and ˜ ∇ f ( x + ∆) be the quantum gradient estimates with probability at least (1 -δ/ 2) . By Lemma 3.2 and union bound, we have

<!-- formula-not-decoded -->

‖ ˜ ∇ f ( x ) -∇ f ( x ) ‖ 2 ≤ 1500 d √ L 1 /epsilon1 hv and ‖ ˜ ∇ f ( x +∆) -∇ f ( x +∆) ‖ 2 ≤ 1500 d √ L 1 /epsilon1 hv hold with probability at least 1 -δ . Condition on this good event, we have

Since we have choose ∆ = 20 √ 15 /epsilon1 1 / 4 hv M -1 / 2 L -1 / 2 2 L 1 / 4 1 d 1 / 2 , it holds that

<!-- formula-not-decoded -->

## B.3 The Proof of Lemma 3.5

Proof. It holds that H [ i, :] estimates ∇ 2 f ( z ) e i with failure probability δ/d , then by Lemma 3.3, holds with probability 1 -δ . Under this event, we have

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

∥ ∥ ∥ -∇ ∥ ∥ ∥ 2 ≤ ∥ ∥ -∇ ∥ ∥ ≤ √ ∥ ∥ -∇ ∥ ∥ ≤ √ holds with probability at least 1 -δ .

## C The Proof of Section 4

## C.1 The Proof of Lemma 4.1

Proof. The iteration rule (6) means

Thus, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality is due to the choice of λ t . We also have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since ‖ z t +1 / 2 -( z t -1 λ t v t ) ‖ = ‖ z t +1 / 2 -z t +1 ‖ , plugging the bound of (14) into (15) and using the fact that ‖ v t -F ( z t +1 / 2 ) ‖ ≤ δ 2 ,t , we have

We let R ≥ 10 ‖ z 0 -z ∗ ‖ and the choice of δ 1 ,t and δ 2 ,t means that we have

Let z = z ∗ in (16) and due to 〈 F ( z k +1 / 2 ) , z k +1 / 2 -z ∗ 〉 ≥ 0 for all k , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

Summing up the (16) from t = 0 to t = T -1 , for all z ∈ B √ 6 R ( z ∗ ) , we have

<!-- formula-not-decoded -->

We then bound the regularization term ∑ T -1 t =0 λ 2 t by

Thus, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality is from Proposition 2.3, ∗ is due to the monotone of F ( · ) , and the last inequality is due to ∑ T -1 t =0 1 λ t ≥ T 3 / 2 √ (864 L 2 2 +4) R 2 according to Lemma A.1.

## C.2 The Proof of Theorem 4.3

Proof. According to Lemmas 3.2 and 3.5, we know that g t , H t , v t can be constructed within ˜ O (1) , ˜ O ( d ) , and ˜ O (1) quantum function evaluation oracle, respectively. The following statements

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

hold with probability at least ( 1 -δ T ) .

<!-- formula-not-decoded -->

Let δ 1 ,t = R 2 10 T , δ H ,t = R √ T , and δ 2 ,t = min { λ t R 2 10 T ( ‖ z t +1 / 2 -z 0 ‖ + R ) , δ 1 ,t 2 } , we know that the condition of Lemma 4.1 holds with probability at least (1 -δ ) . Thus, the output ¯ z T of Algorithm 3 holds that with probability at least (1 -δ ) . The total query of quantum evaluation oracle can be bounded by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C.3 The Proof of Lemma 4.4

Proof. The iteration of (8) means

Thus, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality is due to the choice of λ t . On the other hand, it holds that

<!-- formula-not-decoded -->

We denote r t /defines ‖ z t +1 -z t ‖ , since r 2 t ≤ 2 ‖ z t +1 / 2 -z t ‖ 2 +2 ‖ z t +1 -z t ‖ 2 and that then it holds that

<!-- formula-not-decoded -->

Summing up the above inequality from i = π ( t ) to π ( t ) + s -1 where 1 ≤ s ≤ m and take M ≥ mL 2 √ 3 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality is due to Lemma A.2. Combining the error from v t , we have that

<!-- formula-not-decoded -->

Similar to the proof in Lemma 4.1, we let R ≥ 10 ‖ z 0 -z ∗ ‖ and the choice of δ 1 ,t and δ 2 ,t means

Let z = z ∗ in (25), we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

Summing up the batch of (25), for all z ∈ B ( z ∗ , 3 √ 2 R ) , we have

We then bound the regularization term T -1 t =0 λ 2 t can be bounded by

Finally, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality is due to Lemma A.1.

## C.4 The Proof of Theorem 4.5

Proof. According to Lemmas 3.2 and 3.5, we know that g t , H , v t can be constructed within ˜ O (1) , ˜ O ( d ) , and ˜ O (1) quantum function evaluation oracle, respectively and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and hold with probability at least ( 1 -δ T ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let δ 1 ,t = R 2 10 T , δ H ,t = R √ T , and δ 2 ,t = min { λ t R 2 10 T ( ‖ z t +1 / 2 -z 0 ‖ + R ) , δ 1 ,t 2 } , we know that the condition of Lemma 4.4 holds with probability at least (1 -δ ) . Thus, the output ¯ z T of Algorithm 4 satisfies that with probability at least (1 -δ ) . The total number of queries to the quantum evaluation oracle can be bounded by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C.5 The Proof of Lemma 4.7

Proof. We can obtain a good approximation of F ( z ( s ) i +1 / 2 ) , F ( z ( s ) i ) , and ∇ F ( z ( s ) π ( t ) ) with probability at least (1 -δ/S ) for all i ∈ [ T -1] . Recalling the proof of Lemma 4.4 in Appendix C.3, from (25), we have holds with probability at least (1 -δ/S ) . Assumption 4.6 means that

<!-- formula-not-decoded -->

Let z = z ∗ in (30), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Summing up above inequality from i = 0 to i = T -1 , we have

The choice of δ ( s ) 1 ,t and δ ( s ) 2 ,t guarantees that

Thus we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

inequality, we have

Since z ( s +1) = ¯ z ( s ) T = ∑ T -1 i =0 z ( s ) i λ ( s ) i ∑ T -1 i =0 1 λ ( s ) i and z ( s ) = z ( s ) 0 , using the convexity of ‖ · ‖ 2 and by Jensen's

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We then bound the term ∑ T -1 i =0 ( λ ( s ) i ) 2 by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

it enough to guarantee the linear decent on ‖ z ( s ) -z ∗ ‖ such that

## C.6 The Proof of Theorem 4.8

Proof. Using Lemma 4.7, we have that ‖ z ( s +1) -z ∗ ‖ 2 ≤ 1 2 ‖ z ( s ) -z ∗ ‖ 2 holds for all s ∈ [ S -1] with probability at least 1 -δ . Then we have ‖ z ( S ) -z ∗ ‖ 2 ≤ ( 1 2 ) S ‖ z (0) -z ∗ ‖ 2 ≤ /epsilon1 . For each call of the subroutine HAQZO + , the query complexity can be bounded by

<!-- formula-not-decoded -->

Thus, the total query complexity can be bounded by

<!-- formula-not-decoded -->

## D The Proof of Section 5

We first present some useful results for one step of lazy CRN [14, 13]:

<!-- formula-not-decoded -->

where g t and H π ( t ) are good estimations to ∇ f ( z t ) and ∇ 2 f ( z π ( t ) ) , respectively, such that

<!-- formula-not-decoded -->

Lemma D.1 (Theorem 2.4 [13]) . Consider the cubic regularization step in (33) where g t and H π ( t ) satisfy (34) , then it holds that

<!-- formula-not-decoded -->

Then we know that by properly choosing δ g , δ H , the CRN step can make the gradient small with a rate of O ( T -3 / 2 ) .

Lemma D.2. Under Assumptions 2.2, let ¯ z be uniformly chosen from { z i } T i =1 , generated by (33) , then it holds that

<!-- formula-not-decoded -->

Proof. Summing up (35) from k = π ( t ) to t , then it holds that

<!-- formula-not-decoded -->

where the last inequality is due to Lemma A.3 and M = 30 mL 2

<!-- formula-not-decoded -->

Summing up (36) from 0 to T -1 , we have and

<!-- formula-not-decoded -->

Now we are ready to prove Theorem 5.2.

## D.1 The Proof of Theorem 5.2

Proof. The choice of /epsilon1 g and /epsilon1 H means that

<!-- formula-not-decoded -->

hold with probability at least (1 -δ ) for all t ∈ [ T ] . Using Lemma D.2, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

due to the convexity of x 3 / 2 , we have

<!-- formula-not-decoded -->

We require using ˜ O (1) query complexity of function evaluation oracle to construct the gradient estimator for all T iterations and using ˜ O ( d ) query complexity to construct the Hessian estimator for T m +1 iterations at the snapshot point. Thus, the total query complexity of Algorithm 6 can be bounded by

<!-- formula-not-decoded -->