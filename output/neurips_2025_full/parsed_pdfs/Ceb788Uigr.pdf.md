## On the Convergence of Stochastic Smoothed Multi-Level Compositional Gradient Descent Ascent

## Xinwen Zhang

Temple University Philadelphia, PA, USA ellenz@temple.edu

## Hongchang Gao ∗

Temple University Philadelphia, PA, USA hongchang.gao@temple.edu

## Abstract

Multi-level compositional optimization is a fundamental framework in machine learning with broad applications. While recent advances have addressed compositional minimization problems, the stochastic multi-level compositional minimax problem introduces significant new challenges-most notably, the biased nature of stochastic gradients for both the primal and dual variables. In this work, we address this gap by proposing a novel stochastic multi-level compositional gradient descentascent algorithm, incorporating a smoothing technique under the nonconvex-PL condition. We establish a convergence rate to an ( ϵ, ϵ/ √ κ ) -stationary point with improved dependence on the condition number at O ( κ 3 / 2 ) , where ϵ denotes the solution accuracy and κ represents the condition number. Moreover, we design a novel stage-wise algorithm with variance reduction to address the biased gradient issue under the two-sided PL condition. This algorithm successfully enables a translation from and ( ϵ, ϵ/ √ κ ) -stationary point to an ϵ -stationary point. Finally, extensive experiments validate the effectiveness of our algorithms.

## 1 Introduction

This paper investigates the stochastic multi-level compositional minimax optimization problem:

<!-- formula-not-decoded -->

where f ( G ( x ) , y ) = E [ f ( G ( x ) , y ; ζ )] and ζ denotes a random variable. The function G ( x ) ≜ g ( K ) ( · · · ( g (1) ( x ))) is a K -level compositional function with K &gt; 1 , where each inner-level function g ( k ) ( · ) = E [ g ( k ) ( · ; ξ ( k ) )] depends on the random sample ξ ( k ) for k ∈ { 1 , · · · , K } . The function f ( · , · ) is referred to as the outer-level objective. In this paper, we consider the general nonconvex-PL setting, where f ( G ( x ) , y ) is nonconvex in the primal variable x and satisfies the Polyak-Lojasiewicz (PL) condition with respect to the dual variable y .

Multi-level compositional optimization has emerged as a vital framework in machine learning, with broad applications across numerous domains. In meta-learning, it enhances model adaptability across tasks [12, 22]; in finance, it supports risk-averse portfolio optimization under uncertainty [3, 20]; and in reinforcement learning, it aids policy evaluation and decision refinement [8, 24]. The widespread impact of the multi-level compositional structure highlights its importance in handling complex and structured optimization problems. Moreover, the scope of multi-level compositional optimization extends naturally to the minimax setting, with applications in areas such as deep AUC maximization [33], multi-instance learning [40], and multi-objective learning [19], etc. Despite the importance of these applications, the stochastic multi-level compositional minimax optimization problem remains

∗ Corresponding author

Table 1: The comparison of convergence rate between our algorithms and existing stochastic compositional minimax algorithms.

| Algorithms                                                                                                     | Convergence Rate                                                                              | Assumption                                                                                                                            | Level                                                           |
|----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| SCGDA [17] SCGDAM [33] CODA-Primal [9] NSTORM [26] Smoothed-SMCGDA-VR (Thm. 4.1) Onestage-SMCGDA-VR (Thm. C.1) | O ( κ 4 /ϵ 3 ) O ( κ 4 /ϵ 4 ) O ( κ 4 /ϵ 4 ) O ( κ 3 /ϵ 3 ) O ( κ 3 / 2 /ϵ 3 ) O ( κ 3 /ϵ 3 ) | Nonconvex-strongly-concave Nonconvex-strongly-concave Nonconvex-strongly-concave Nonconvex-strongly-concave Nonconvex-PL Nonconvex-PL | Two-level Two-level Two-level Two-level Multi-level Multi-level |

largely underexplored. This gap in the literature motivates our study, which aims to develop an effective algorithmic solution for this challenging class of problems.

Solving multi-level compositional problems is challenging, even in the minimization setting. In particular, when the inner-level functions are nonlinear, the stochastic gradient is no longer an unbiased estimator of the full gradient. Recent research has proposed new algorithms to address this issue. Notably, [31] introduced a K -level stochastic compositional gradient descent algorithm, and subsequent efforts [5, 23, 35] have developed algorithms specifically tailored to address the biased characteristics inherent in the stochastic multi-level compositional framework. Unfortunately, these minimization-targeted algorithms cannot directly address the stochastic multi-level compositional minimax optimization problem in Eq. (1), as the stochastic gradients for both primal and dual variables are biased estimators in stochastic multi-level compositional minimax problems-posing greater algorithmic and theoretical challenges.

In addition, although [17, 26, 9] investigate the two-level compositional minimax problem, it remains within the classical minimax framework and does not consider more advanced techniques that can improve convergence. In contrast, recent progress in classical minimax optimization has demonstrated that smoothed techniques can significantly improve convergence. For instance, [36] proposed a smoothed alternating gradient method for general nonconvex-concave problems, which achieves superior performance compared to conventional approaches. Building on this, [30] further applied this technique to the nonconvex-PL setting-a milder condition than strong concavity-and showed that stochastic smoothed techniques yield improved complexity bounds with better dependence on the condition number κ . However, despite the clear benefits of smoothed techniques in traditional minimax optimization, its application to multi-level compositional minimax problems remains unexplored. This observation motivates a key question: Can smoothed techniques be effectively integrated into the multi-level compositional framework to improve convergence performance?

Addressing this question is not straightforward and presents substantial algorithmic and theoretical challenges. On the one hand, while existing studies [36, 30] demonstrate that smoothing techniques are effective for unbiased stochastic gradient estimators, the biased nature of stochastic gradients for both the primal and dual variables in Eq. (1) introduces uncertainty regarding the effectiveness of applying such techniques. It remains unclear whether their use may lead to additional convergence issues. Therefore, it is essential to develop new algorithms that can accommodate biased gradient estimators and guarantee convergence when using smoothing techniques for Eq. (1). On the other hand, as demonstrated in [30], smoothed algorithms typically guarantee an ( ϵ 1 , ϵ 2 ) -stationary point, rather than a standard ϵ -stationary point, which are defined in Definition 3.5. This necessitates a translation between the two measures. While such a translation introduces negligible iteration complexity in classical minimax problems when using an unbiased gradient estimator as shown in [30], there is no known algorithm capable of performing this translation in the context of multi-level compositional minimax optimization. In particular, it remains unclear how to design a translation algorithm without degrading the iteration complexity of the smoothed algorithm in the presence of a multi-level compositional structure. Therefore, these challenges motivate us to address the problem through the following contributions:

- We develop a novel smoothed multi-level compositional minimax optimization algorithm for Eq. (1) by leveraging the variance reduction technique to mitigate the biased gradient estimator issue, and establish a convergence rate of O ( κ 3 / 2 /ϵ 3 ) to an ( ϵ, ϵ/ √ κ ) -stationary point. Compared to existing algorithms, our method achieves a better dependence on the

condition number κ : improving over the O ( κ 3 ) rate of standard two-level compositional minimax algorithms. √

- To bridge the gap between an ( ϵ, ϵ/ κ ) -stationary point and a standard ϵ -stationary point, we further propose a stage-wise variance-reduced algorithm for Eq. (1) under the two-sided PL condition. We show that the algorithm achieves a convergence rate of O (1 /ϵ 2 ) to an ϵ -stationary point. As a result, the iteration complexity from the translation is dominated by the complexity of finding an ( ϵ, ϵ/ √ κ ) -stationary point.
- Meanwhile, we obtain two additional results, which may be of independent interest: the convergence rates for the multi-level compositional minimax problem under the nonconvex-PL and two-sided-PL assumptions without using the smooth technique, as summarized in Table 1.
- We conduct extensive experiments to validate the effectiveness of our proposed algorithms, demonstrating superior performance compared to existing baselines.

## 2 Related Work

## 2.1 Stochastic Compositional Minimization Optimization

Recently, a general class of stochastic compositional gradient descent methods [29, 18, 32, 15, 16] was developed for two-level compositional minimization problems and established convergence rates for nonconvex loss functions. Aiming to address practical problems with a more general stochastic compositional structure, the stochastic two-level compositional problem has been extended to the stochastic multi-level compositional problem. Stochastic multi-level compositional learning has various applications, including multi-step model-agnostic meta-learning [12], the stochastic training of graph neural networks [6], the neural networks with batch-normalization [25], etc. Consequently, a series of stochastic multi-level compositional minimization algorithms [31, 2, 5, 35, 23, 13, 14] have been developed to solve this important problem. Notably, [31] introduced the first stochastic multilevel compositional gradient descent algorithm. Then, [2] employed a moving-average estimator, and [5] used the STORM variance-reduction estimator [7] for each inner-level function, achieving a convergence rate of O (1 /ϵ 4 ) . Later, [35] improved the sample complexity to O (1 /ϵ 3 ) by applying the SPIDER variance-reduction technique [11, 27] to both the inner-level function and Jacobian matrix at each level. Nevertheless, the large batch size required by this method makes it impractical for large-scale models, and the learning rate must be sufficiently small to maintain Lipschitz continuity of the variance-reduced gradient. By applying the STORM variance-reduction approach to both the function value and its Jacobian matrix at each level, [23] developed a convergence rate of O (1 /ϵ 3 ) for the stochastic multi-level compositional problem with a mini-batch size of O (1) . More recently, for the first time, [13] showed that the variance-reduction estimator is not necessary for the Jacobian matrix in each level to achieve a convergence rate of O (1 /ϵ 3 ) . However, these stochastic multilevel compositional algorithms focus exclusively on minimization problems and therefore cannot be directly applied to multi-level compositional minimax problems.

## 2.2 Stochastic Compositional Minimax Optimization

Stochastic compositional minimax optimization [17, 33, 9, 26, 37, 38] has attracted increasing attention due to its important applications in machine learning. To solve the two-level compositional minimax problem, [17] developed the first compositional minimax algorithm based on the mini-batch compositional gradient, achieving a convergence rate of O ( κ 4 /ϵ 4 ) for nonconvex-strongly-concave loss functions. [33] incorporated the momentum technique to reduce the mini-batch size to O (1) while achieving the same convergence rate as [17]. Similarly, [9] used a variance-reduced estimator for the inner-level function, also reducing the mini-batch size to O (1) and achieving the same convergence rate as [17]. [26] introduced the STORM technique for estimating the inner-level function and gradient, achieving a convergence rate of O ( κ 3 /ϵ 3 ) . Recently, [9] claimed to achieve a convergence rate of O ( κ 2 /ϵ 3 ) . However, this convergence rate is established with respect to the stationary point of the Moreau envelope of the primal function, rather than that of the original primal function. As a result, it corresponds to the convergence rate for a strongly-convex-strongly-concave loss function, rather than for nonconvex-strongly-concave or nonconvex-PL loss functions. More recently, [37] developed the first stochastic multi-level compositional minimax algorithm for nonconvex-stronglyconcave loss functions in the federated learning setting. However, its convergence rate O (1 /ϵ 4 ) is suboptimal compared to the multi-level compositional minimization algorithm. On the other hand, the smoothed technique was first introduced for nonconvex-concave minimax problems in [36], where the convergence rate of a full alternating gradient descent ascent method was established. Later, [30]

extended this technique to the nonconvex-PL setting and further investigated the relationship between two stationarity measures. However, none of these algorithms are equipped to handle the challenges posed by multi-level compositional minimax problems, which remain largely unexplored.

## 3 Preliminaries

## 3.1 Notations

We begin by simplifying the complex formulation in Eq. (1) to facilitate analysis:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The partial gradients of the objective function can then be expressed as follows:

<!-- formula-not-decoded -->

Following prior works [36, 30], we introduce an auxiliary variable z alongside the primal variable x as part of the smoothed technique, and define the smoothed loss function as:

<!-- formula-not-decoded -->

where ω &gt; 0 is a constant and f ω ( G ( x ) , y ; z ) is strongly convex with respect to x by selecting an appropriate ω . Using the smoothed loss, we can derive stochastic estimators of the compositional gradients with respect to the primal and dual variables at the t -th iteration:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 3.2 Assumptions

We next introduce the following standard assumptions, which are commonly used in stochastic compositional optimization [17, 26, 9, 38, 37, 30].

## Assumption 3.1. (Smoothness):

- For any k ∈ { 1 , 2 , · · · , K } , g ( k ) ( · ) and g ( k ) ( · ; ξ ) are C g -Lipschitz continuous, ∇ g ( k ) ( · ) and ∇ g ( k ) ( · ; ξ ) are L g -Lipschitz continuous,where C g &gt; 0 and L g &gt; 0 ;
- f ( · , · ) and f ( · , · ; ζ ) are C f -Lipschitz continuous, ∇ f ( · , · ) and ∇ f ( · , · ; ζ ) are L f -Lipschitz continuous, where C f &gt; 0 and L f &gt; 0 .

## Assumption 3.2. (Variance):

- For any k ∈ { 1 , · · · , K } , the stochastic gradients ∇ g ( k ) ( · ; ξ ( k ) ) and ∇ f ( · , · ; ζ ) have upper bounded variance σ 2 , where σ &gt; 0 .

## Assumption 3.3. (PL Condition):

- For any fixed x ∈ R d x , the maximization problem y ∗ = max y ∈ R dy f ( G ( x ) , y ) has a non-empty solution set and a finite optimal value. Moreover, for all x ∈ R d x , there exists a constant value µ &gt; 0 such that ∥∇ y f ( G ( x ) , y ) ∥ 2 ≥ 2 µ ( f ( G ( x ) , y ∗ ) -f ( G ( x ) , y )) .

Here, we define ℓ = max { L f , C 2 g K L f + C f ∑ K -1 k =0 L f C K -1+ k g } , and κ = ℓ µ denotes the condition number. Then, when ω &gt; ℓ , f ω ( G ( x ) , y ; z ) is strongly convex with respect to x . We also introduce the following definitions.

## Definition 3.4. (Two-sided PL Condition):

- f ( x, y ) satisfies the two-sided PL condition, if there exist constants µ x &gt; 0 and µ y &gt; 0 such that f ( · , y ) is µ x -PL for any y ∈ R d y , and -f ( x, · ) is µ y -PL for any x ∈ R d x .

Definition 3.5. (Stationarity measures):

- ( x, y ) is an ( ϵ 1 , ϵ 2 ) -stationary point of f ( · , · ) , if ∥∇ x f ( G ( x ) , y ) ∥ ≤ ϵ 1 and ∥∇ y f ( G ( x ) , y ) ∥ ≤ ϵ 2 .
- x is an ϵ -stationary point of Φ( · ) , if ∥∇ Φ( x ) ∥ ≤ ϵ , where Φ( x ) = f ( G ( x ) , y ∗ ) and y ∗ = arg max y ∈ R dy f ( G ( x ) , y ) .

## 3.3 Challenges

From the algorithmic design perspective, one of the primary challenges in incorporating smoothed techniques is managing the intrinsic bias of stochastic gradients for both the primal and dual variables . Specifically, as shown in Eq.(3), the partial gradient regarding the dual variable y relies on the stochastic estimator of K -level function G ( K ) ( · ) , while that regarding the primal variable x depends on the stochastic estimator of both G ( K ) ( · ) and ∇ G ( K ) ( · ) . In the stochastic setting, however, computing the stochastic estimator for both the k -th level function and its corresponding gradient introduces bias, as illustrated below:

̸

<!-- formula-not-decoded -->

̸

As a result, the stochastic gradients with respect to both primal and dual variables are biased estimators of the full gradient. Moreover, as shown in Eq. (6), the estimation biases accumulate across all compositional levels when estimating both the inner-level functions and their gradient. This accumulation of bias introduces greater complexity compared to the two-level case and raises concerns about whether the deeper compositional structure might undermine the effectiveness of smoothed techniques, as all existing smoothed minimax methods handle deterministic gradients or unbiased stochastic gradients .

From the theoretical analysis perspective, a major challenge arises from the gap between different stationarity measures induced by smoothed techniques . As demonstrated in [30], a translation is required from an ( ϵ 1 , ϵ 2 ) -stationary point to an ϵ -stationary point. In standard minimax settings, this can be achieved by applying a stochastic gradient descent-ascent algorithm to the auxiliary problem min x ∈ R dx max y ∈ R dy f ( x, y ) + ℓ ∥ x -˜ z ∥ 2 , where ˜ z is the output of the smoothed algorithm. Owing to the fact that this formulation satisfies the the PL condition in both x and y , with an iteration complexity of ˜ O (1 /ϵ 2 ) . Therefore, if the cost of this translation remains lower than that of the smoothed algorithm itself, it does not affect the overall complexity. However, for multi-level compositional minimax problems, there do not exist algorithms for handling the two-sided PL condition to complete the translation, and it is unclear whether the iteration complexity of the translation is smaller than that of the smoothed algorithm or not . In particular, the existing study [31] showed that the standard compositional gradient descent algorithm can only achieve a convergence rate with an exponential dependence on the number of levels, even for strongly convex loss functions. As a result, the complexity of the translation phase could dominate the overall complexity. Therefore, it remains unclear whether there exists an efficient algorithm to translate from an ( ϵ 1 , ϵ 2 ) -stationary point to an ϵ -stationary point for multi-level compositional minimax problems.

## 4 Algorithm 1: Smoothed-SMCGDA-VR

## 4.1 Algorithmic Design

To address the smoothed loss in Eq. (4), we design a novel algorithm, named stochastic smoothed multi-level compositional gradient descent ascent with variance reduction (Smoothed-SMCGDA-VR), as presented in Algorithm 1. To mitigate the accumulation of bias at each compositional level, our method incorporates a STORM-like variance-reduced estimator. Specifically, for each inner-level function g ( k ) ( · ) , where k ∈ { 1 , . . . , K } , we apply a recursive step that updates the estimator h ( k ) while controlling variance. This variance reduction technique is also employed for the stochastic gradients: ∇ x f ω ( · , · ; · ; ˆ ξ t , ζ t ) and ∇ y f ω ( · , · ; · ; ˆ ξ t , ζ t ) .

More concretely, the variance-reduced estimator for each levelk function is computed as:

<!-- formula-not-decoded -->

where h (0) t +1 = x t +1 when k = 0 , and α &gt; 0 is a hyperparameter such that αη 2 ∈ (0 , 1) .

Algorithm 1 Stochastic Smoothed Multi-Level Compositional Gradient Descent Ascent with Variance Reduced ( Smoothed -SMCGDA-VR)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 1: for t = 0 , · · · , T -1 do
- 2: Update x and y : x t +1 = x t -γ x ηp t , y t +1 = y t + γ y ηq t ,
- 3: Update z : z t +1 = z t + γ z η ( x t +1 -z t ) ,
- 4: h (0) t +1 = x t +1 ,
- 5: for k = 1 , · · · , K do
- 6: Compute k -th inner-level function: h ( k ) t +1 = g ( k ) ( h ( k -1) t +1 ; ξ ( k ) t +1 ) + (1 -αη 2 )( h ( k ) t -g ( k ) ( h ( k -1) t ; ξ ( k ) t +1 ))
- 7: end for
- 8: Compute stochastic compositional gradient u t +1 and v t +1 : u t +1; t +1 = ∇ g (1) ( h (0) t +1 ; ξ (1) t +1 ) · · · ∇ g ( K -1) ( h ( K -2) t +1 ; ξ ( K -1) t +1 ) ∇ g ( K ) ( h ( K -1) t +1 ; ξ ( K ) t +1 ) × ∇ 1 f ( h ( K ) t +1 , y t +1 ; ζ t +1 ) + ω ( x t +1 -z t +1 ) , ( K )

t

v

+1;

t

+1

=

∇

2

f

(

h

t

+1

, y

t

+1

;

ζ

t

+1

)

,

- 9: Compute variance-reduced gradient p t +1 and q t +1 :

<!-- formula-not-decoded -->

For the outer-level update, we compute the stochastic gradient of the smoothed loss defined in Eq. (5), based on the variance-reduced estimator { h ( k ) t +1 } K k =1 of the inner-level function, as presented in Step 8. Here, u t +1; t +1 denotes the stochastic compositional gradient regarding primal variable, where the first index indicates the t +1 -th iteration of the variable, and the second reflects the sample indices ˆ ξ t +1 = {{ ξ ( k ) t +1 } K k =1 , ζ t +1 } . Similarly, we compute the stochastic gradient with respect to the dual variable based on the variance-reduced estimator h ( K ) t +1 of the inner-level function. The algorithm then performs STORM-like updates on p t +1 and q t +1 , as presented in Step 9, where ρ x &gt; 0 and ρ y &gt; 0 are two hyperparameters such that ρ x η 2 ∈ (0 , 1) and ρ y η 2 ∈ (0 , 1) .

## 4.2 Theoretical Analysis

We derive the convergence rate of Algorithm 1 in the following theorem 2 .

Theorem 4.1. Given Assumptions 3.1-3.3, when ρ x &gt; 0 , ρ y &gt; 0 , α &gt; 0 , ω = O ( ℓ ) , and the hyperparameter conditions in Eq. (94) are satisfied, Algorithm 1 achieves the following convergence upper bound:

<!-- formula-not-decoded -->

where P 0 = f ω ( G ( x 0 ) , y 0 ; z 0 ) -2 f ω,d ( y 0 ; z 0 ) + 2 g ( z 0 ) , with the definitions of the involved terms provided in Eq. (25).

Corollary 4.2. Given Assumptions 3.1-3.3, by setting γ x = O (1) , γ y = O (1) , γ z = O (1 /κ ) , η = O ( ϵ/κ 1 / 2 ) , ρ x = O (1) , ρ y = O (1) , α = O (1) , S = O ( κ 1 / 2 /ϵ ) , T = O ( κ 3 / 2 /ϵ 3 ) ,

2 Due to space limitations, the theorem with the full hyperparameter conditions is provided in the Appendix B.2.

Algorithm 1 can achieve the O ( ϵ, ϵ/ √ κ ) -stationary solution, where ϵ &gt; 0 denotes the solution accuracy, and S is the batch size in the initial iteration.

Note that our Theorem 4.1 provides the convergence rate in terms of the stationary point of the original loss function f ( G ( x t ) , y t ) , rather than that of the smoothed loss function f ω ( G ( x ) , y ; z ) . Therefore, this result corresponds to the convergence rate for a nonconvex-PL loss function, rather than for a two-sided-PL loss function. As a result, the comparison of convergence rates with existing methods in Table 1 is fair and consistent with the comparison made in the context of classical smoothed minimax optimization in [30].

Proof Sketch. To establish the convergence rate of Algorithm 1, we propose a novel potential function as follows:

<!-- formula-not-decoded -->

where the coefficient ν a , ν b and { λ k } K k =1 are positive, where the notations of ∇ x f ω ( H ( x t ) , y t ; z t ) and ∇ y f ω ( H ( x t ) , y t ; z t ) can be found in Eq. (23).

To analyze the descent of the potential function, we decompose and bound each term through a sequence of lemmas. First, we bound:

<!-- formula-not-decoded -->

which characterizes the optimization error introduced by the smoothed technique. Each component of P t depends on x , y and z , and the compositional gradient introduces additional bias:

- We first derive upper bounds for each component in P t .
- We then combine these bounds in Appendix B.2.1 to analyze and quantify their dependence, providing a clear characterization of how the three terms interact.

Second, three additional terms in Eq. (9) arise from the gradient errors regarding x and y , and the inner-level estimation error in the multi-level compositional loss.

Third, the four terms in H t are interdependent. We analyze these dependencies in Appendix B.2.2 and show that H t satisfies a sufficient descent property, i.e. , H t +1 -H t can be bounded under suitable hyperparameter conditions, ensuring convergence to an ( ϵ, ϵ/ √ κ ) -stationary point. The complete proof is provided in Appendix B.

## 5 Algorithm 2: Stagewise-SMCGDA-VR

## 5.1 Algorithmic Design

| Algorithm 2 Stagewise-SMCGDA-VR                                                              | Algorithm 2 Stagewise-SMCGDA-VR                                                                                                                    |
|----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| Input: ρ x > 0 , ρ y > 0 , α > 0 , η x,r > 0 , η y,r > 0 . for Stage r = 0 , · · · ,R - 1 do | Input: ρ x > 0 , ρ y > 0 , α > 0 , η x,r > 0 , η y,r > 0 . for Stage r = 0 , · · · ,R - 1 do                                                       |
| 1: 2:                                                                                        | x r, 0 = ˜ x r , y r, 0 = ˜ y r , h ( k ) r, 0 = ˜ h ( k ) r for k ∈ { 0 , · · · ,K - 1 } , p r, 0 = ˜ p r , q r, 0 = ˜ q r .                      |
| 3:                                                                                           | for t = 0 , · · · ,T r - 1 , do                                                                                                                    |
| 4:                                                                                           | Perform one iteration t of SMCGDA-VR update                                                                                                        |
| 5:                                                                                           | Randomly select (˜ x r +1 , ˜ y r +1 , ˜ h ( k ) r +1 , ˜ p r +1 , ˜ q r +1 ) from { ( x r,t , y r,t ,h ( k ) r,t , p r,t , q r,t ) } T r - 1 t =0 |
| 6:                                                                                           | end for                                                                                                                                            |
| 7:                                                                                           | end for                                                                                                                                            |

However, to facilitate a fair comparison between the convergence rate of Algorithm 1 and existing stochastic two-level compositional minimax methods, which establish the rate in terms of ϵ -stationary point instead of ( ϵ 1 , ϵ 2 ) -stationary point, it is necessary to convert the ( ϵ, ϵ/ √ κ ) -stationary solution into an ϵ -stationary solution. As discussed in Section 3.3, making this translation is challenging for

multi-level compositional minimax optimization problems. Specifically, in the classical minimax setting, [30] showed that the standard stochastic gradient descent ascent (SGDA) algorithm is sufficient for the translation by solving a strongly-convex-strongly-concave problem, since its convergence rate is only ˜ O (1 /ϵ 2 ) , which is dominated by that of the smoothed algorithm. However, this approach does not work for the multi-level compositional minimax optimization problem. Specifically, even for the multi-level compositional minimization optimization problem, the classical stochastic compositional gradient descent algorithm can only achieve a convergence rate with an exponential dependence on the number of levels for strongly convex loss functions, as shown in [31].

The aforementioned challenge motivates the development of a new algorithm to handle the translation from an ( ϵ, ϵ/ √ κ ) -stationary solution into an ϵ -stationary solution. To this end, we aim to develop a new algorithm to solve the multi-level compositional minimax optimization problem that satisfies the two-sided PL condition. Specifically, assume ˜ z is the output of Algorithm 1, then we complete the translation by solving the following problem.

<!-- formula-not-decoded -->

Note that ˜ z is the output x from Algorithm 1 and it is fixed when solving this problem. Moreover, since ω is selected such that ˆ f ( G ( x ) , y ) is strongly convex with respect to x , ˆ f ( G ( x ) , y ) naturally satisfies the two-sided PL condition. Then, our next goal is to develop an efficient algorithm to solve Eq. (11) such that its iteration complexity is better than that of Algorithm 1, i.e., the translation does not hurt the overall convergence rate .

To this end, we propose a novel stage-wise algorithm, named Stagewise-SMCGDA-VR, as shown in Algorithm 2 ( Note that a more general algorithm is presented in Algorithm 3 for the multi-level compositional minimax optimization problem satisfying the two-sided PL condition. This algorithm may be of independent interest, beyond its use for the translation phase .). The overall optimization is divided into R stages, and in each stage, we run the SMCGDA-VR algorithm without updating z (i.e., removing the component highlighted in blue) and replacing z with ˜ z in Step 8. At the end of each stage r , the algorithm randomly selects a tuple from the set { ( x r,t , y r,t , h ( k ) r,t , p r,t , q r,t ) } T r -1 t =0 , where k ∈ { 1 , . . . , K } , to be used as the initialization for the next stage r +1 . A complete description of the algorithm is given in the Appendix C.

## 5.2 Theoretical Analysis

We establish the convergence rate of Algorithm 2 in the following theorem. More general results for the extended Algorithm 3, which may be of independent interest, are presented in Theorems C.1-C.2.

Theorem 5.1. Given Assumption 3.1-3.4, by setting c 0 = 25 L 2 f µ 2 , ρ x = 6400 c 0 L 2 β , ρ y = 640 L 2 β , α = 640 c 0 L 2 β , η y, 0 = 1 20 L β , T 0 = max { 225 , 16 V 0 L β σ 2 } , and for r ≥ 1 , η x,r = O ( µ 2 / ( √ 2 r -1 L β )) , η y,r = O (1 / ( √ 2 r -1 L β )) , T r = O ( c 0 / ( µ × 2 r -1 )) , after running Algorithm 2 for the total number of iterations (not stages) O (1 /ϵ 2 ) , we can get E [ ∥∇ Φ(˜ x R ) ∥ 2 ] ≤ ϵ 2 .

Remark 5.2. From Theorem 5.1, it can be observed that the iteration complexity O (1 /ϵ 2 ) of the translation phase is much smaller than that of Algorithm 1. Therefore, the translation does not hurt the overall convergence rate.

Remark 5.3. Since the overall iteration complexity is determined by Algorithm 1, we can conclude that our algorithm achieves an iteration complexity of T = O ( κ 3 / 2 /ϵ 3 ) , improving upon the O ( κ 4 /ϵ 4 ) complexity of the two-level compositional minimax problem in [17, 9] by offering better dependence on both κ and ϵ and the O ( κ 3 /ϵ 3 ) complexity in [26] by a better dependence on κ . To the best of our knowledge, this is the first algorithm to achieve an O ( κ 3 / 2 ) dependence for (multi-level compositional) minimax problems under the nonconvex-PL setting.

Proof Sketch. To prove Theorem 5.1, we use an induction approach to handle the stage-wise structure of Algorithms 2. We introduce two metrics to facilitate convergence analysis:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where V t denotes the optimization error, and U t is similar to the last three terms of Eq. (9), c 0 is a positive constant such that c 0 η x,t η y,t = 1 10 . Importantly, following Lemma C.6 and C.7, we establish how V t and U t affect each other across stages and derive the following inequalities in Appendix C.3:

<!-- formula-not-decoded -->

These bounds differ only by a factor of 1 /µ . Using induction, at the r -th stage, we assume

<!-- formula-not-decoded -->

where ϵ r &gt; 0 is a constant. Finally, by selecting appropriate hyperparameters, we prove that

<!-- formula-not-decoded -->

As such, we establish the desired convergence rate. The complete proof is provided in Appendix C.

## 6 Experiment

## 6.1 Deep AUC Maximization

In the deep AUC maximization problem, applying K -step gradient descent to minimize the crossentropy loss function results in a K -level inner function G ( · ) in Eq. (1), with a detailed discussion provided in Appendix A. We compare our smoothed method with three baselines: SCGDA [17], SCGDAM [33], and NSTORM [26] across three datasets: CATvsDOG, CIFAR10 and STL10. Imbalanced binary datasets are generated following the approach described in [33], with an imbalance ratio of 0.05. ResNet20 is employed as the model. For all algorithms, we set both the learning rate and the momentum or variance reduction coefficient to 0.1. In our proposed method, we employ smoothed techniques during the first 90 epochs, followed by stage-wise updates for the remaining 10 epochs.

Figure 1: The test AUC score versus the number of stochastic first-order gradient evaluations.

<!-- image -->

We conduct experiments using our smoothed method for both K = 1 and K = 5 , with results presented in Figure 1. Notably, NSTORM applies STORM-like updates to the two-level( K = 1 ) compositional minimax problem without smoothed techniques. Our results show that the smoothed approach consistently outperforms all baselines. Moreover, as the number of levels increases, the smoothed method does not degrade the performance, demonstrating its robustness to increased compositional levels. This improvement is observed consistently across all datasets, highlighting the effectiveness of incorporating deeper compositional structures. Additional experiments with varying K are provided in the Appendix A.

## 6.2 Multi-Instance Learning

Following [39], multi-instance learning can be reformulated as a multi-level compositional minimax problem as shown in Eq. 1, with details provided in Appendix A. For multi-instance learning tasks, our proposed approach utilizes two types of stochastic pooling operations: log-sum-exp (smx) pooling and attention-based (att) pooling. We compare the performance of our smoothed methods against six baseline methods: MIDAM(smx) and MIDAM(att) [40], both utilizing stochastic pooling

Figure 2: The test AUC score versus the number of epochs for Tabular Datasets.

<!-- image -->

operations; DAM(mean), DAM(max), DAM(smx), and DAM(att), all of which update the AUC loss with traditional PESG optimizer [34].

We conduct experiments on five commonly used tabular benchmark datasets [10, 1] for MIL tasks MUSK1, MUSK2, Fox, Tiger, and Elephant - as well as one histopathological image dataset, namely Breast Cancer. For the tabular datasets, we use a two-layer feed-forward neural network with tanh activation and a sigmoid output for AUC loss normalization. For the Breast Cancer dataset, each image is divided into 32 × 32 patches and treated as a bag of 672 local patches to enable efficient multi-instance processing, using ResNet20 as the model. All datasets are randomly split into training and testing sets with a 0.9/0.1 ratio.

For the tabular datasets, we perform 5fold cross-validation, repeating each run with three random seeds. For the image dataset, we use two random seeds. The learning rate for the primal variables is tuned within the set {1e-1, 1e-2, 1e3}, while the learning rate for the dual variables is fixed at 1. We vary the value of K from 1 to 5 and ultimately fix it at 3 to achieve more stable performance. We present the experimental results on the tabular datasets in Figure 2, and on

Figure 3: The test AUC score versus the number of epochs for Breast Cancer Dataset.

<!-- image -->

the image dataset in Figure 3. For the tabular datasets, to ensure clearer visualizations, we omit error bars in the plots and instead report both the mean and standard deviation of the results in Table 2, as shown in Appendix A. For the image dataset, we focus our comparison on the softmax-based and attention-based methods. In both experimental settings, our proposed algorithms consistently outperform all baseline methods, demonstrating superior optimization behavior and generalization performance across a range of tasks and datasets.

## 7 Conclusion

In this work, we addressed the challenging problem of stochastic multi-level compositional minimax optimization by proposing a smoothed variance-reduced algorithm. Our theoretical analysis demonstrates that the proposed smoothed method achieves a convergence rate of O ( κ 3 / 2 /ϵ 3 ) to an ( ϵ, ϵ/ √ κ ) -stationary point. Furthermore, to bridge the gap between different stationarity measures, we developed a stage-wise algorithm under the two-sided PL condition, enabling a translation to an ϵ -stationary point. Extensive experiments on deep AUC maximization and multi-instance learning tasks validate the superior performance of our approach.

## Acknowledcements

We thank anonymous reviewers for constructive comments. X. Zhang and H. Gao were partially supported by U.S. NSF CAREER 2339545, NSF IIS 2416607, NSF CNS 2107014.

## References

- [1] S. Andrews, I. Tsochantaridis, and T. Hofmann. Support vector machines for multiple-instance learning. Advances in neural information processing systems , 15, 2002.
- [2] K. Balasubramanian, S. Ghadimi, and A. Nguyen. Stochastic multilevel composition optimization algorithms with level-independent convergence rates. SIAM Journal on Optimization , 32(2):519-544, 2022.
- [3] S. Bruno, S. Ahmed, A. Shapiro, and A. Street. Risk neutral and risk averse approaches to multistage renewable investment planning under uncertainty. European Journal of Operational Research , 250(3):979-989, 2016.
- [4] L. Chen, B. Yao, and L. Luo. Faster stochastic algorithms for minimax optimization under polyak-{\ L } ojasiewicz condition. Advances in Neural Information Processing Systems , 35:13921-13932, 2022.
- [5] T. Chen, Y. Sun, and W. Yin. Solving stochastic compositional optimization is nearly as easy as solving stochastic optimization. IEEE Transactions on Signal Processing , 69:4937-4948, 2021.
- [6] W. Cong, M. Ramezani, and M. Mahdavi. On the importance of sampling in training gcns: Tighter analysis and variance reduction. arXiv preprint arXiv:2103.02696 , 2021.
- [7] A. Cutkosky and F. Orabona. Momentum-based variance reduction in non-convex sgd. Advances in neural information processing systems , 32, 2019.
- [8] C. Dann, G. Neumann, J. Peters, et al. Policy evaluation with temporal differences: A survey and comparison. Journal of Machine Learning Research , 15:809-883, 2014.
- [9] Y. Deng, F. Qiao, and M. Mahdavi. Stochastic compositional minimax optimization with provable convergence guarantees. In International Conference on Artificial Intelligence and Statistics , pages 3835-3843. PMLR, 2025.
- [10] T. G. Dietterich, R. H. Lathrop, and T. Lozano-Pérez. Solving the multiple instance problem with axis-parallel rectangles. Artificial intelligence , 89(1-2):31-71, 1997.
- [11] C. Fang, C. J. Li, Z. Lin, and T. Zhang. Spider: Near-optimal non-convex optimization via stochastic path-integrated differential estimator. Advances in neural information processing systems , 31, 2018.
- [12] C. Finn, P. Abbeel, and S. Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning , pages 1126-1135. PMLR, 2017.
- [13] H. Gao. Decentralized multi-level compositional optimization algorithms with level-independent convergence rate. In International Conference on Artificial Intelligence and Statistics , pages 4402-4410. PMLR, 2024.
- [14] H. Gao. A doubly recursive stochastic compositional gradient descent method for federated multi-level compositional optimization. In Forty-first International Conference on Machine Learning , 2024.
- [15] H. Gao and H. Huang. Fast training method for stochastic compositional optimization problems. Advances in Neural Information Processing Systems , 34:25334-25345, 2021.
- [16] H. Gao, J. Li, and H. Huang. On the convergence of local stochastic compositional gradient descent with momentum. In International Conference on Machine Learning , pages 7017-7035. PMLR, 2022.

- [17] H. Gao, X. Wang, L. Luo, and X. Shi. On the convergence of stochastic compositional gradient descent ascent method. In Thirtieth International Joint Conference on Artificial Intelligence , 2021.
- [18] S. Ghadimi, A. Ruszczynski, and M. Wang. A single timescale stochastic approximation method for nested stochastic optimization. SIAM Journal on Optimization , 30(1):960-979, 2020.
- [19] A. Gu, S. Lu, P. Ram, and T.-W. Weng. Min-max multi-objective bilevel optimization with applications in robust machine learning. In The Eleventh International Conference on Learning Representations , 2022.
- [20] R. Huang, S. Qu, X. Yang, and Z. Liu. Multi-stage distributionally robust optimization with risk aversion. Journal of Industrial &amp; Management Optimization , 17(1), 2021.
- [21] M. Ilse, J. Tomczak, and M. Welling. Attention-based deep multiple instance learning. In International conference on machine learning , pages 2127-2136. PMLR, 2018.
- [22] K. Ji, J. Yang, and Y. Liang. Multi-step model-agnostic meta-learning: Convergence and improved algorithms. arXiv preprint arXiv:2002.07836 , 2, 2020.
- [23] W. Jiang, B. Wang, Y. Wang, L. Zhang, and T. Yang. Optimal algorithms for stochastic multilevel compositional optimization. In International Conference on Machine Learning , pages 10195-10216. PMLR, 2022.
- [24] C. J. Li, M. Wang, H. Liu, and T. Zhang. Near-optimal stochastic approximation for online principal component estimation. Mathematical Programming , 167:75-97, 2018.
- [25] X. Lian and J. Liu. Revisit batch normalization: New understanding from an optimization view and a refinement via composition optimization. arXiv preprint arXiv:1810.06177 , 2018.
- [26] J. Liu, X. Pan, J. Duan, H.-D. Li, Y . Li, and Z. Qu. Faster stochastic variance reduction methods for compositional minimax optimization. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pages 13927-13935, 2024.
- [27] L. M. Nguyen, J. Liu, K. Scheinberg, and M. Takáˇ c. Sarah: A novel method for machine learning problems using stochastic recursive gradient. In International conference on machine learning , pages 2613-2621. PMLR, 2017.
- [28] J. Ramon, L. De Raedt, and S. Kramer. Multi instance neural networks. In Proceedings of the ICML-2000 workshop on attribute-value and relational learning , pages 53-60, 2000.
- [29] M. Wang, E. X. Fang, and H. Liu. Stochastic compositional gradient descent: algorithms for minimizing compositions of expected-value functions. Mathematical Programming , 161:419449, 2017.
- [30] J. Yang, A. Orvieto, A. Lucchi, and N. He. Faster single-loop algorithms for minimax optimization without strong concavity. In International Conference on Artificial Intelligence and Statistics , pages 5485-5517. PMLR, 2022.
- [31] S. Yang, M. Wang, and E. X. Fang. Multilevel stochastic gradient methods for nested composition optimization. SIAM Journal on Optimization , 29(1):616-659, 2019.
- [32] H. Yuan and W. Hu. Stochastic recursive momentum method for non-convex compositional optimization. arXiv preprint arXiv:2006.01688 , 2020.
- [33] Z. Yuan, Z. Guo, N. Chawla, and T. Yang. Compositional training for end-to-end deep auc maximization. In International Conference on Learning Representations , 2021.
- [34] Z. Yuan, Y. Yan, M. Sonka, and T. Yang. Large-scale robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 3040-3049, 2021.
- [35] J. Zhang and L. Xiao. Multilevel composite stochastic optimization via nested variance reduction. SIAM Journal on Optimization , 31(2):1131-1157, 2021.

- [36] J. Zhang, P. Xiao, R. Sun, and Z. Luo. A single-loop smoothed gradient descent-ascent algorithm for nonconvex-concave min-max problems. Advances in neural information processing systems , 33:7377-7389, 2020.
- [37] X. Zhang, A. Payani, M. Lee, R. Souvenir, and H. Gao. A federated stochastic multi-level compositional minimax algorithm for deep AUC maximization. In Forty-first International Conference on Machine Learning , 2024.
- [38] X. Zhang, Y. Zhang, T. Yang, R. Souvenir, and H. Gao. Federated compositional deep auc maximization. Advances in Neural Information Processing Systems , 36:9648-9660, 2023.
- [39] D. Zhu, G. Li, B. Wang, X. Wu, and T. Yang. When auc meets dro: Optimizing partial auc for deep learning with non-convex convergence guarantee. In International Conference on Machine Learning , pages 27548-27573. PMLR, 2022.
- [40] D. Zhu, B. Wang, Z. Chen, Y. Wang, M. Sonka, X. Wu, and T. Yang. Provable multi-instance deep auc maximization with stochastic pooling. In International Conference on Machine Learning , pages 43205-43227. PMLR, 2023.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claims are clearly stated.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations of previous work are discussed appropriately.

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

Justification: Every relevant detail is covered.

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

Justification: Every relevant detail is covered.

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

Justification: The dataset is open access and the code will be shared after acceptance.

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

Justification: Every relevant detail is covered.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The multi-instance task includes error bars.

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

Justification: Every relevant detail is covered.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: All ethical standards are satisfied.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This is a theoretical paper and not relevant to societal impacts.

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

Answer: [Yes]

Justification: Every relevant detail is covered.

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

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

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

Justification: The core method development in this research does not involve LLMs.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Contents

| 1 Introduction   | 1 Introduction                   | 1 Introduction                                                            | 1     |
|------------------|----------------------------------|---------------------------------------------------------------------------|-------|
| 2                | Related Work                     | Related Work                                                              | 3     |
|                  | 2.1                              | Stochastic Compositional Minimization Optimization                        | 3     |
|                  | 2.2                              | Stochastic Compositional Minimax Optimization . .                         | 3     |
| 3                | Preliminaries                    | Preliminaries                                                             | 4     |
|                  | 3.1                              | Notations . . . . . . . . . . . . . . . . . . . . . .                     | 4     |
|                  | 3.2                              | . Assumptions . . . . . . . . . . . . . . . . . . . . . .                 | 4     |
|                  | 3.3                              | Challenges . . . . . . . . . . . . . . . . . . . . . . .                  | 5     |
| 4                | Algorithm 1: Smoothed-SMCGDA-VR  | Algorithm 1: Smoothed-SMCGDA-VR                                           | 5     |
|                  | 4.1                              | Algorithmic Design . . . . . . . . . . . . . . . . . .                    | 5     |
|                  | 4.2                              | Theoretical Analysis . . . . . . . . . . . . . . . . .                    | 6     |
| 5                | Algorithm 2: Stagewise-SMCGDA-VR | Algorithm 2: Stagewise-SMCGDA-VR                                          |       |
|                  | 5.1                              | Algorithmic Design . . . . . . . . . . . . . . . . . .                    | 7 7   |
|                  | 5.2                              | Theoretical Analysis . . . . . . . . . . . . . . . . .                    |       |
|                  |                                  |                                                                           | 8     |
| 6                | Experiment                       | Experiment                                                                | 9     |
|                  | 6.1                              | Deep AUC Maximization . . . . . . . . . . . . . . .                       | 9     |
|                  | 6.2                              | Multi-Instance Learning . . . . . . . . . . . . . . .                     | 9     |
| 7                | Conclusion                       | Conclusion                                                                | 10    |
| A                | Applications A.1                 | and Experiments Deep AUC Maximization . . . . . . . . . . . . . . .       | 21 21 |
|                  |                                  | Multi-Instance Learning                                                   | 21    |
|                  | A.2                              | . . . . . . . . . . . . . . .                                             |       |
|                  | A.3                              | More Experimental Results . . . . . . . . . . . . . .                     | 22    |
| B                | Appendix: Smoothed-SMCGDA-VR     | Appendix: Smoothed-SMCGDA-VR                                              | 22    |
|                  | B.1                              | Useful Lemmas . . . . . . . . . . . . . . . . . . . .                     | 23    |
|                  | B.2                              | Proof of the Theorem 4.1 . . . . . . . . . . . . . . .                    | 28    |
|                  |                                  | B.2.1 Bound P t +1 -P t . . . . . . . . . . . . . . .                     | 29    |
|                  |                                  | B.2.2 Bound H t +1 -H t . . . . . . . . . . . . . .                       | 32    |
| C                | Appendix: Stagewise-SMCGDA-VR    | Appendix: Stagewise-SMCGDA-VR                                             | 38    |
|                  | C.1 Useful Lemmas                | . . . . . . . . . . . . . . . . .                                         | 38    |
|                  | . . . C.2                        | . . . . . . . . . . . . . . .                                             | 47    |
|                  | C.3 Proof                        | Proof of the Theorem C.1 of the Theorem C.2 . . . . . . . . . . . . . . . | 47    |
|                  | C.4                              | Proof of the Theorem 5.1 . . . . . . . . . . . . . . .                    | 53    |

## A Applications and Experiments

## A.1 Deep AUC Maximization

AUC(Area under the ROC curve) is widely used to evaluate the classifiers for binary classification with imbalanced data. [33] reformulated the AUC maximization problem as the following two-level compositional minimax problem:

<!-- formula-not-decoded -->

where w ∈ R d are model parameters while ( a, b, α ) are parameters for AUC loss, ( x, y ) represents feature and label of a sample.

Here, L CE indicates the standard cross-entropy loss function, w -˜ η ∇ w L CE denotes using the gradient descent approach on cross-entropy loss to update the model parameters, where ˜ η &gt; 0 is the learning rate. Then, the obtained model parameter ˜ w can be optimized through the AUC loss. The following serves as a generic representation of Eq. (16) as a two-level compositional minimax optimization problem:

<!-- formula-not-decoded -->

Figure 4: Different K on CIFAR10. where g denotes the inner-level function with one-step gradient descent and f denotes the outer-level function. Inspired

<!-- image -->

by the achievements in addressing the multi-level compositional minimization problem, we extend the one-step gradient descent for the inner-level function to a multi-step update. In detail, for k ∈ { 1 , · · · , K } , the k -th inner-level function is defined as:

̸

<!-- formula-not-decoded -->

where ˜ g refers to g ( k -1) ( · ) when k ∈ { 2 , · · · , K } , ξ ( k ) represents the data distribution for the k -th level function. The learning rate for the inner-level functions is denoted by ˜ η . Consequently, Eq. (17) can be reformulated as a multi-level compositional minimax optimization problem exactly as the Eq. (1).

## A.2 Multi-Instance Learning

Multi-instance learning [10] is designed for tasks with training data structured into bags containing many instances, with only bag-level labels known. The symmetric function, also known as the pooling operation, is a critical component of multi-instance learning. Diverse pooling strategies have been investigated, including mean pooling, max pooling, and softmax pooling [28] and attention-based pooling [21]. Then, to address memory concerns, [40] provided a class of variance-reduced stochastic pooling approaches by reformulating the AUC loss function with the pooled prediction as a three-level compositional minimax function as follows:

<!-- formula-not-decoded -->

where X i = { x 1 i , · · · , x n i i } denotes a bag of data instances, D + represents only containing positive bags with label y i = 1 , D -represents only containing negative bags with label y i = 0 . The pooled prediction h ( w ; X i ) = f 2 ( f 1 ( w ; X i )) denotes the predicted score of the bag i over all its instance, which is a two-level compositional function. For example, for the log-sum-exp(smx) pooling, we have:

<!-- formula-not-decoded -->

For the attention-based (att) pooling, we have:

<!-- formula-not-decoded -->

Similarly, the three-level compositional minimax problem in Eq. (19) can be reformulated as a stochastic multi-level compositional minimax problem by integrating it with cross-entropy loss minimization, as in Eq. (16), after computing the predicted score h ( w ; X i ) . In particular, applying K inner gradient steps to optimize the cross-entropy loss results in a K -level inner function. Consequently, Eq.(19) can be expressed in the unified form of Eq. (1), which corresponds to a stochastic ( K +3) -level compositional minimax problem.

## A.3 More Experimental Results

Here, we provide additional empirical results. Specifically, for the deep AUC maximization task, we perform experiments to evaluate the impact of the number of levels K on performance. As shown in Figure 4, increasing the number of inner levels leads to further improvements in testing performance. For the multi-instance learning task, we report both the mean and standard deviation of the results on tabular datasets in Table 2.

Table 2: The test AUC score of different methods on all Tabular Datasets.

| Methods             | MUSK1                     | MUSK2                     | Fox                                                 | Tiger                     | Elephant                  |
|---------------------|---------------------------|---------------------------|-----------------------------------------------------|---------------------------|---------------------------|
| Ours(att) Ours(smx) | 0.942(0.039) 0.921(0.047) | 0.965(0.029) 0.939(0.025) | 0.738(0.018) 0.770(0.034) 0.702(0.056) 0.686(0.050) | 0.942(0.017) 0.928(0.026) | 0.931(0.034) 0.942(0.026) |
|                     | 0.841(0.142)              | 0.868(0.087)              |                                                     |                           |                           |
| MIDAM(att)          |                           |                           | 0.718(0.078)                                        | 0.918(0.030)              | 0.919(0.029)              |
| MIDAM(smx)          | 0.841(0.142)              | 0.905(0.117)              |                                                     | 0.909(0.031)              | 0.903(0.039)              |
| DAM(att)            | 0.770(0.143)              | 0.782(0.075)              |                                                     | 0.870(0.027)              | 0.861(0.022)              |
| DAM(smx)            | 0.802(0.175)              | 0.847(0.116)              | 0.684(0.049)                                        | 0.889(0.014)              | 0.908(0.025)              |
| DAM(max)            | 0.745(0.112)              | 0.822(0.123)              | 0.591(0.082)                                        | 0.895(0.047)              | 0.875(0.028)              |
| DAM(mean)           | 0.795(0.138)              | 0.826(0.072)              | 0.653(0.103)                                        | 0.855(0.021)              | 0.895(0.020)              |

## B Appendix: Smoothed-SMCGDA-VR

To begin with, we introduce the following terminology to simplify the complex expressions, which will be useful in the subsequent analysis:

<!-- formula-not-decoded -->

Therefore, for the smoothed loss, we have

<!-- formula-not-decoded -->

Moreover, we introduce C 2 p as follows:

<!-- formula-not-decoded -->

Following [30], we introduce the following auxiliary functions for convergence analysis:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof Structure. Our proof consists of two key components. The first component, including Lemma B.3 and Lemma B.4, addresses the smoothing technique. The second component, comprising Lemma B.5, Lemma B.6, and Lemma B.7, deals with the multi-level compositional structure. In Section B.2, we complete the proof by carefully combining these two components while addressing their interdependence.

## B.1 Useful Lemmas

Lemma B.1. Given Assumptions 3.1-3.3, we can know

1. G ( k ) ( x ) is C k g -Lipschitz continuous for k ∈ { 1 , · · · , K -1 } and G ( x ) is C G -Lipschitz continuous where C G = C K g ;
2. ∇ G ( x ) is L G -Lipschitz continuous where L G = ∑ K -1 j =0 L g C K -1+ j g ;
3. ∇ x f ( G ( x ) , y ) is ˆ L -Lipschitz continuous where ˆ L = C 2 G L f + C f L G ;
4. Φ( x ) ≜ max y ∈ R dy f ( G ( x ) , y ) , Φ( x ) is L Φ -Lipschitz continuous where L Φ = 2 C 2 G L 2 f µ + C f L G .

Proof. The first three properties follow from Lemma B.1. in [37]. The last property is based on Lemma A.3. in [30] and can be established by showing that ∥ Φ( x 2 ) -Φ( x 1 ) ∥ ≤ ( 2 C 2 G L 2 f µ + C f L G ) ∥ x 2 -x 1 ∥ .

Lemma B.2. [30] Given Assumptions 3.1-3.3, the following inequality holds:

<!-- formula-not-decoded -->

where C x 1 yz = ω + ℓ ω -ℓ , C x 2 yz = ω ω -ℓ , and C x z = ω ω -ℓ and ω &gt; ℓ .

Lemma B.3. Given Assumptions 3.1-3.3, and γ z η ≤ 1 , the following inequality holds:

1. The smoothed function f ω ( G ( x t ) , y t ; z t ) satisfies:

<!-- formula-not-decoded -->

2. The dual function E [ h ω,d ( y t ; z t )] satisfies:

<!-- formula-not-decoded -->

3. The function h ( z t ) satisfies:

<!-- formula-not-decoded -->

Proof. (1). From the update rule z t +1 = z t + γ z η ( x t +1 -z t ) , we obtain

<!-- formula-not-decoded -->

where the last step holds uses the fact that γ z η ≤ 1 .

Proof. (2). Since the dual function h ω,d ( y t ; z t ) is L ω,d -smooth, it satisfies that

<!-- formula-not-decoded -->

On the other hand, we have

<!-- formula-not-decoded -->

where we use the fact ⟨ a -b, a + b ⟩ = ∥ a ∥ 2 -∥ b ∥ 2 in the second-to-last step. By combining the above two inequalities, the proof is complete.

Proof. (3). From the definition of h ( z t ) , we obtain

<!-- formula-not-decoded -->

where we use the fact ⟨ a -b, a + b ⟩ = ∥ a ∥ 2 -∥ b ∥ 2 in the last step.

Lemma B.4. Given Assumptions 3.1-3.3, when η ≤ 1 2 γ x ( ω + ℓ ) , and γ z η ≤ 1 , the following inequalities hold:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. First, from Lemma B.1, it follows that the smoothed loss function f ω ( G ( x t ) , y t ; z t ) is ( ω + ℓ ) -smooth with respect to x . Therefore, we have

<!-- formula-not-decoded -->

where the second-to-last step holds due to η ≤ 1 2 γ x ( ω + ℓ ) .

Similarly, since f ω ( G ( x t ) , y t ; z t ) is ( ω + ℓ ) -smooth with respect to y , we obtain

<!-- formula-not-decoded -->

By combining the inequalities above with Lemma B.3, the proof is complete.

Lemma B.5. Given Assumptions 3.1-3.3, the following inequality holds:

1. The estimation error between ∇ x f ( G ( x t ) , y t ) and ∇ x f ( H ( x t ) , y t ) satisfies:

<!-- formula-not-decoded -->

2. The bounded error between ∇ x f ω ( H ( x t ) , y t ; z t ; ˆ ξ t +1 ) and ∇ x f ω ( H ( x t ) , y t ; z t ) satisfies: E [ ∥∇ x f ω ( H ( x t ) , y t ; z t ; ˆ ξ t +1 ) -∇ x f ω ( H ( x t ) , y t ; z t ) ∥ 2 ] ≤ C 2 p σ 2 . (38)
3. For any k ∈ { 1 , · · · , K } , the descent error between h ( k ) t +1 and h ( k ) t satisfies:

<!-- formula-not-decoded -->

4. For any λ k &gt; 0 where k ∈ { 1 , · · · , K } , the estimation error between g ( k ) ( h ( k -1) t ) and h ( k ) t satisfies:

<!-- formula-not-decoded -->

Proof. First, we have

<!-- formula-not-decoded -->

where the last step follows from Lemma B.2, Eq. (25) in [37]. Similarly,

<!-- formula-not-decoded -->

where the last step follows from Lemma B.2, Eq. (28) in [37]. From the definition of C 2 p , the proof is complete.

Subsequently, the remaining inequalities follow from Lemma B.4 and Lemma B.5 in [37].

Lemma B.6. Given Assumptions 3.1-3.3, we derive

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

where the last step holds due to Lemma B.5 and third step holds due to the following inequality:

<!-- formula-not-decoded -->

Next, we bound T 1 as follows:

<!-- formula-not-decoded -->

Combining this with the previous inequalities completes the proof.

Lemma B.7. Given Assumption 3.1-3.3, we derive:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

by applying Lemma B.5, the proof is complete.

## B.2 Proof of the Theorem 4.1

Theorem B.8. (Restatement of Theorem 4.1) Given Assumptions 3.1-3.3, when ρ x &gt; 0 , ρ y &gt; 0 , α &gt; 0 , ω = O ( ℓ ) , and the hyperparameter conditions are satisfied:

<!-- formula-not-decoded -->

Algorithm 1 achieves the following convergence upper bound:

<!-- formula-not-decoded -->

where P 0 = f ω ( G ( x 0 ) , y 0 ; z 0 ) -2 f ω,d ( y 0 ; z 0 ) + 2 g ( z 0 ) , with the definitions of the involved terms provided in Eq. (25).

Proof. To establish the convergence rate of Algorithm 1, we propose a novel potential function as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the coefficient ν a , ν b and { λ k } K k =1 are positive.

## B.2.1 Bound P t +1 -P t

First, we aim to derive an upper bound for P t +1 -P t . To this end, we begin by applying Lemmas B.3 and B.4, from which we obtain

<!-- formula-not-decoded -->

Next, we derive

<!-- formula-not-decoded -->

where the third step follows from Lemma B.2.

Additionally, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where c &gt; 0 is a constant, and the last step follows from

<!-- formula-not-decoded -->

Setting c = 1 8 , we obtain

<!-- formula-not-decoded -->

Furthermore, due to the strong convexity of f ω ( G ( x t ) , y t ; z t ) regarding x , we obtain

<!-- formula-not-decoded -->

In addition, by introducing

<!-- formula-not-decoded -->

we obtain

<!-- formula-not-decoded -->

Moreover, we derive

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, we obtain the following upper bound for P t +1 -P t :

<!-- formula-not-decoded -->

## B.2.2 Bound H t +1 -H t

In the following, we aim to derive an upper bound for H t +1 -H t :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We consider the following choice for bounding E [ ∥∇ x f ω ( G ( x t ) , y t ; z t ) ∥ 2 ] and E [ ∥∇ y f ω ( G ( x t ) , y t ; z t ) ∥ 2 ] :

<!-- formula-not-decoded -->

Since γ z η ≤ 1 and C x 1 yz = ω + ℓ ω -ℓ , we set

<!-- formula-not-decoded -->

Additionally, we consider the following choice for bounding E [ ∥ z t +1 -z t ∥ 2 ] , we set

<!-- formula-not-decoded -->

Specifically, we enforce

<!-- formula-not-decoded -->

Then, based on Eq. (64), from C x z = C x 2 yz and η &lt; 1 , we obtain

<!-- formula-not-decoded -->

To remove the term E [ ∥∇ x f ω ( H ( x t ) , y t ; z t ) -p t ∥ 2 ] , we impose

<!-- formula-not-decoded -->

From this, we obtain the parameter choice

<!-- formula-not-decoded -->

Similarly, to remove the term E [ ∥∇ y f ω ( H ( x t ) , y t ; z t ) -q t ∥ 2 ] , we impose

<!-- formula-not-decoded -->

From the second inequality in Eq. (63) and definition of c γ y , we have

<!-- formula-not-decoded -->

As a result, we require which leads to the parameter choice

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Plugging the value of ν a and ν b , we obtain

<!-- formula-not-decoded -->

To analyze this, we first simplify the expression:

<!-- formula-not-decoded -->

Due to αη 2 ≤ 1 , we enforce the following to be non-positive:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, we obtain the parameter choice for any λ k where k ∈ { 1 , · · · , K } :

<!-- formula-not-decoded -->

Moreover, we enforce

<!-- formula-not-decoded -->

for k ∈ { 1 , · · · , K } , which leads to

<!-- formula-not-decoded -->

To guarantee that E [ ∥ p t ∥ 2 ] cancels out, we enforce

<!-- formula-not-decoded -->

This is equivalent to enforce

<!-- formula-not-decoded -->

Specifically, we enforce

<!-- formula-not-decoded -->

To solve the first third inequalities, we obtain

<!-- formula-not-decoded -->

For the last inequality, it is equivalent to enforce

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Specifically, we enforce

<!-- formula-not-decoded -->

For the first inequality, since d k is independent of hyperparameters, we obtain

<!-- formula-not-decoded -->

For the remaining inequalities, we obtain

<!-- formula-not-decoded -->

As for E [ ∥ q t ∥ 2 ] , we enforce

<!-- formula-not-decoded -->

This is equivalent to enforce

<!-- formula-not-decoded -->

Specifically, we enforce

<!-- formula-not-decoded -->

To solve these inequalities, we obtain

<!-- formula-not-decoded -->

In summary, by setting

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we obtain

<!-- formula-not-decoded -->

From

<!-- formula-not-decoded -->

by summing over t from 0 to T -1 and reformulate it, we obtain

<!-- formula-not-decoded -->

When t = 0 , we derive

<!-- formula-not-decoded -->

Finally, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C Appendix: Stagewise-SMCGDA-VR

Note that in this section, we provide a general algorithm for the multi-level compositional minimax optimization problem satisfying the two-sided PL condition. Specifically, we aim to solve the following problem:

<!-- formula-not-decoded -->

where f ( · , · ) satisfies the two-sided PL condition. To simplify the analysis, we further assume that the loss function satisfies the same continuity, smoothness, and bounded variance assumptions as stated in the main text.

In this general setting, we can obtain two results which may be of independent interest beyond their use for the translation phase.

First, when the number of stages is just one, i.e., R = 1 , we can obtain the convergence rate for the multi-level compositional minimax optimization problem satisfying the nonconvex-PL assumption in Theorem C.1.

Theorem C.1. Given Assumption 3.1-3.3, when R = 1 , by setting η y, 1 = O ( ϵ/κ ) , η x, 1 = O ( ϵ/κ 3 ) , and the initial batch size as O ( κ/ϵ ) , after running Algorithm 3 for T 1 = O ( κ 3 /ϵ 3 ) total iterations, we have 1 T 1 ∑ T 1 -1 t =0 E [ ∥∇ Φ( x 1 ,t ) ∥ 2 ] ≤ ϵ 2 .

Note that in the proof of Theorem C.1, we do not use the PL condition with respect to x . Therefore, the result provides a convergence rate for the nonconvex-PL minimax problem. In addition, this convergence rate corresponds to the standard compositional minimax algorithm without the use of the smoothing technique. Therefore, in Table C, we compare the convergence rate and learning rate with and without the use of the smoothing technique. It can be seen that we should use a smaller learning rate for x compared to y when not using the smoothing technique, as the condition number κ &gt; 1 .

Table 3: The comparison of the convergence rate and learning rate with and without the use of the smoothing technique. LRx denotes the learning rate for x , and LRy denotes that for y .

| Algorithms                                                  | Convergence Rate   | LR- x           | LR- y           | LR- x / LR- y   |
|-------------------------------------------------------------|--------------------|-----------------|-----------------|-----------------|
| Smoothed-SMCGDA-VR (Thm. 4.1) Onestage-SMCGDA-VR (Thm. C.1) | O ( κ 3 / 2 /ϵ 3 ) | O ( ϵ/κ 1 / 2 ) | O ( ϵ/κ 1 / 2 ) | O (1) O (1 /κ 2 |
|                                                             | O ( κ 3 /ϵ 3 )     | O ( ϵ/κ 3 )     | O ( ϵ/κ )       | )               |

Second, when the number of stages is greater than one, i.e., R &gt; 1 , we can obtain the convergence rate for the multi-level compositional minimax optimization problem satisfying the two-sided PL condition in Theorem C.2.

Theorem C.2. Given Assumption 3.1-3.4, by setting c 0 = 25 L 2 f µ 2 , ρ x = 6400 c 0 L 2 β , ρ y = 640 L 2 β , α = 640 c 0 L 2 β , η y, 0 = 1 20 L β , T 0 = max { 225 , 16 V 0 L β σ 2 } , and for r ≥ 1 , η x,r = O ( µ 2 / ( √ 2 r -1 L β )) , η y,r = O (1 / ( √ 2 r -1 L β )) , T r = O ( c 0 / ( µ × 2 r -1 )) , after running Algorithm 3 for O ( κ 6 /ϵ ) total iterations, we can get E [Φ(˜ x R ) -Φ( x ∗ )] ≤ ϵ .

## C.1 Useful Lemmas

Lemma C.3. Given Assumptions 3.1-3.3 and η x,r ≤ 1 2 L Φ , we know

<!-- formula-not-decoded -->

Algorithm 3 Stagewise Stochastic Multi-level Compositional Gradient Descent Ascent with Variance Reduced Algorithm (Stagewise-SMCGDA-VR)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

4:

x

r,t

+1

=

x

r,t

-

η

y

r,t

+1

=

y

r,t

+

η

x,r y,r

p

q

r,t r,t

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

7:

Compute

k

-th inner-level function:

<!-- formula-not-decoded -->

- 8: end for
- 9: Compute stochastic compositional gradient u r,t +1 and v r,t +1

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

v

r,t

+1;

t

+1

=

∇

(

K

)

2

f

(

h

r,t

+1

, y r,t

+1

;

ζ

r,t

)

,

Compute variance-reduced gradient

10:

p

r,t

+1

and

q

r,t

+1

<!-- formula-not-decoded -->

- 11: end for

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 13: end for

<!-- formula-not-decoded -->

## Proof.

<!-- formula-not-decoded -->

,

.

:

where the second-to-last step holds due to η x,r ≤ 1 2 L Φ .

Lemma C.4. Given Assumption 3.1-3.3 , η y,r ≤ 1 ℓ , we have

<!-- formula-not-decoded -->

Proof. First, from Lemma B.1, we obtain

<!-- formula-not-decoded -->

Moreover, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the sixth step holds due to η y,r ≤ 1 ℓ , the fourth step follows from the following inequality:

<!-- formula-not-decoded -->

By combining these two inequalities, the proof is complete.

Lemma C.5. Given Assumption 3.1-3.3 , η x,r ≤ 1 16 ℓ , we have

<!-- formula-not-decoded -->

Proof. In terms of Lemma C.3 and Lemma C.4, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last step follows from ∥∇ x f ( G ( x r,t ) , y r,t ) ∥ 2 ≤ 2 ∥∇ Φ( x r,t ) ∥ 2 +2 ∥∇ x f ( G ( x r,t ) , y r,t ) -∇ Φ( x r,t ) ∥ 2 and η x,r ≤ 1 16 ℓ .

Lemma C.6. Given Assumption 3.1-3.3 , by setting

<!-- formula-not-decoded -->

where ˜ λ k is defined in Eq. (116), L β is defined in Eq. (132), such that η x,r = η y,r 10 c 0 , we have

<!-- formula-not-decoded -->

Proof. We first propose a novel Lyapunov function as follows:

<!-- formula-not-decoded -->

where η x,r = η y,r 10 c 0 . Then, from Lemma C.5, B.5, B.6 and B.7, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since η x,r = η y,r 10 c 0 , we enforce

<!-- formula-not-decoded -->

By solving this, we obtain

<!-- formula-not-decoded -->

where k ∈ { 1 , · · · , K } and λ ′ k = max { λ k, 1 , λ k, 2 , λ k, 3 , λ k, 4 } . Given that λ k can be organized as:

<!-- formula-not-decoded -->

Moreover, we enforce

<!-- formula-not-decoded -->

for k ∈ { 1 , · · · , K } , which leads to

<!-- formula-not-decoded -->

Additionally, by plugging the value of λ k , we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To guarantee that E [ ∥ p r,t ∥ 2 ] cancels out, we enforce

<!-- formula-not-decoded -->

This can be done by setting

<!-- formula-not-decoded -->

For the first inequality, we enforce

<!-- formula-not-decoded -->

It is easy to obtain

<!-- formula-not-decoded -->

For the second and third inequalities, we obtain

<!-- formula-not-decoded -->

Similarly, to guarantee that E [ ∥ q r,t ∥ 2 ] cancels out, we enforce

<!-- formula-not-decoded -->

With η x,r = η y,r 10 c 0 , we obtain

We obtain

<!-- formula-not-decoded -->

To solve this inequality, we enforce

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In summary, the hyperparameters should be set as follows:

<!-- formula-not-decoded -->

Moreover, by setting c 0 = 25 ℓ 2 µ 2 , we get

<!-- formula-not-decoded -->

Then from Lemma A.7 in [4] , we have ( η x,r + 4 c 0 η 2 x,r η y,r ) E [ ∥∇ Φ( x r,t ) -∇ x f ( G ( x r,t ) , y r,t ) ∥ 2 ] ≤ c 0 η x,r 16 E [ ∥∇ y f ( G ( x r,t ) , y r,t ) ∥ 2 ] .

As a result, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last step holds due to αη 2 x,r ≤ 1 , and -η x,r 4 ≥ -η y,r 8 since η x,r = η y,r 10 c 0 , c 0 = 25 ℓ 2 µ 2 . Then, we set

<!-- formula-not-decoded -->

where λ ′ k is defined in Eq.(115). It is easy to verify that the conditions in Eq. (129) are satisfied. Meanwhile, this indicates ρ x = 10 c 0 ρ y , α = c 0 ρ y .

By summing t from 0 to T r -1 , we obtain

<!-- formula-not-decoded -->

where V r, 0 = E [Φ( x r, 0 )] -Φ( x ∗ ) + c 0 η x,r η y,r ( E [Φ( x r, 0 )] -E [ f ( g ( x r, 0 ) , y r, 0 )]) , σ x r, 0 = E [ ∥∇ x f ( H ( x r, 0 ) , y r, 0 ) -p r, 0 ∥ 2 ] , σ y r, 0 = E [ ∥∇ y f ( H ( x r, 0 ) , y r, 0 ) -q r, 0 ∥ 2 ] and σ h,k r, 0 = E [ ∥ g ( k ) ( h ( k -1) r, 0 ) -h ( k ) r, 0 ∥ 2 ] , σ h r, 0 = ∑ K k =1 λ ′ k σ h,k r, 0 .

## C.2 Proof of the Theorem C.1

Proof. Based on Lemma C.6, we have

<!-- formula-not-decoded -->

Since c 0 = O ( κ 2 ) , then it is easy to verify that by setting by setting η y, 1 = O ( ϵ/κ ) , η x, 1 = O ( ϵ/κ 3 ) , T 1 = O ( κ 3 /ϵ 3 ) , and the initial batch size as O ( κ/ϵ ) , we have

<!-- formula-not-decoded -->

As a result, we can conclude 1 T 1 ∑ T 1 -1 t =0 E [ ∥∇ Φ( x 1 ,t ) ∥ 2 ] ≤ ϵ 2 .

## C.3 Proof of the Theorem C.2

Lemma C.7. Assumption 3.1-3.4 , we have

<!-- formula-not-decoded -->

Proof. In the following, we will bound σ x r, 0 + σ y r, 0 +56 σ h r, 0 . At first, from Lemma B.5, we get

<!-- formula-not-decoded -->

where the last step holds due to αη 2 x,r ≤ 1 . Then, according to the random sampling operation, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last step holds due to ρ y = 640 L 2 β .

Then, based on Lemma B.6, we have

<!-- formula-not-decoded -->

where the last step holds due to the definition of λ ′ k and αη 2 x,r ≤ 1 . Then, due to the random sampling in each outer iteration, it is easy to know

<!-- formula-not-decoded -->

where the last step holds due to ρ y = 640 L 2 β , c 0 &gt; 1 and 0 x,r η y,r = 1 10 . Similarly, from Lemma B.7, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second step holds due to the definition of λ ′ k , and the last step holds due to αη 2 x,r ≤ 1 . Then, due to the randomly sampling operation in each outer iteration, it is easy to know

<!-- formula-not-decoded -->

where the last step holds due to ρ y = 640 L 2 β .

Then, we combine these three inequalities together as follows:

<!-- formula-not-decoded -->

Then, we need to bound 1 T r ∑ T r -1 t =0 E [ ∥ p r,t ∥ 2 ] + c 0 η y,r η x,r 1 T r ∑ T r -1 t =0 E [ ∥ q r,t ∥ 2 ] . In particular, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By plugging Eq. (131), we obtain

<!-- formula-not-decoded -->

By summing up t from 0 to T r -1 , we get

<!-- formula-not-decoded -->

By plugging this inequality to Eq. (143), we get

<!-- formula-not-decoded -->

In the following, we prove Theorem C.2.

Proof. Under the two-sided PL condition, we get

<!-- formula-not-decoded -->

we set

<!-- formula-not-decoded -->

Due to the random sampling in each outer iteration, we get

<!-- formula-not-decoded -->

Therefore, when r = 0 , we have σ x 0 , 0 + σ y 0 , 0 +56 σ h 0 , 0 = 58 L 2 β σ 2 . Based on Eq. (149) and Lemma C.7, we have

<!-- formula-not-decoded -->

When r = 0 , by setting η y, 0 = 1 30 L β and R 0 = max { 225 , 16 V 0 , 0 L β σ 2 } , we have

<!-- formula-not-decoded -->

where the second step holds due to ρ y = 640 L 2 β .

Therefore, we denote ϵ 1 ≜ 500 c 0 L 2 β σ 2 /µ such that

<!-- formula-not-decoded -->

In the following, we use the inductive approach to prove the desired result. Specifically, suppose σ x r, 0 + σ y r, 0 +56 σ h r, 0 ≤ µϵ r and V r, 0 ≤ ϵ r , we will prove σ x r +1 , 0 + σ y r +1 , 0 +56 σ h r +1 , 0 ≤ µϵ r / 2 and V r +1 , 0 ≤ ϵ r / 2 . At first, we have

<!-- formula-not-decoded -->

To make σ x r +1 , 0 + σ y r +1 , 0 +56 σ h r +1 , 0 ≤ µϵ r / 2 , we enforce each term to be smaller than ϵ r / 6 . In particular, by setting

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It is easy to verify that ρ y η 2 y,r &lt; 1 for t ≥ 1 .

By setting we get

By setting we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, by setting η y,r = √ µϵ r 1140 √ c 0 L 2 β σ and T r = O ( c 0 L 2 β √ c 0 σ µ √ µϵ r ∨ c 2 0 L 4 β σ 2 µϵ r ) , we get

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Since ϵ r = ϵ 1 2 r -1 = 500 c 0 L 2 β σ 2 2 r -1 µ , we get

<!-- formula-not-decoded -->

Therefore, we set T r = O ( c 0 µ × 2 r -1 ) . Finally, to achieve V R, 0 ≤ ϵ , we need ϵ 1 2 ( R -1) = ϵ so that R = log 2 2 ϵ 1 ϵ . As such, the total number of iterations is

<!-- formula-not-decoded -->

where the second step holds due to

<!-- formula-not-decoded -->

Moreover, we get

<!-- formula-not-decoded -->

and it is easy to know η x,r = O ( µ 2 / 2 r -1 L β ) .

## C.4 Proof of the Theorem 5.1

Proof. Because ˆ f ( G ( x ) , y ) is strongly convex with respect to x and satisfies the PL condition with respect to y , we have

<!-- formula-not-decoded -->

where we set ω = 2 ℓ such that ˆ f ( G ( x ) , y ) is ℓ -strongly convex with respect to x , and we define ˆ Φ( x ) = max y ˆ f ( G ( x ) , y ∗ ) with y ∗ = arg max y ∈ R dy ˆ f ( G ( x ) , y ) . Then, according to Proposition 2.1 in [30], to guarantee E [ ∥ x ˜ R -x ∗ ∥ 2 ] ≤ O ( ϵ 2 ) such that E [ ∥∇ Φ(˜ x R ) ∥ 2 ] ≤ O ( ϵ 2 ) , we can enforce E [ ˆ Φ( x ˜ R ) -ˆ Φ( x ∗ )] ≤ O ( ϵ 2 ) . Then, from Theorem C.2, it is easy to see that after running Algorithm 2 for the total number of iterations O (1 /ϵ 2 ) (Note that 1 /ϵ is usually large in practice [30] so that we omit other factors.), we have E [ ∥ x ˜ R -x ∗ ∥ 2 ] ≤ O ( ϵ 2 ) and then E [ ∥∇ Φ(˜ x R ) ∥ 2 ] ≤ O ( ϵ 2 ) .