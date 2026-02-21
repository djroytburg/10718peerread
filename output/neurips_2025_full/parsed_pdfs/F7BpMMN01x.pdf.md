## Finite-Sample Analysis of Policy Evaluation for Robust Average Reward Reinforcement Learning

## Yang Xu

## Washim Uddin Mondal

Purdue University West Lafayette, IN 47907, USA xu1720@purdue.edu

Indian Institute of Technology Kanpur Kanpur, UP, India 208016 wmondal@iitk.ac.in

## Vaneet Aggarwal

Purdue University West Lafayette, IN 47907, USA vaneet@purdue.edu

## Abstract

We present the first finite-sample analysis of policy evaluation in robust averagereward Markov Decision Processes (MDPs). Prior work in this setting have established only asymptotic convergence guarantees, leaving open the question of sample complexity. In this work, we address this gap by showing that the robust Bellman operator is a contraction under a carefully constructed semi-norm, and developing a stochastic approximation framework with controlled bias. Our approach builds upon Multi-Level Monte Carlo (MLMC) techniques to estimate the robust Bellman operator efficiently. To overcome the infinite expected sample complexity inherent in standard MLMC, we introduce a truncation mechanism based on a geometric distribution, ensuring a finite expected sample complexity while maintaining a small bias that decays exponentially with the truncation level. Our method achieves the order-optimal sample complexity of ˜ O ( /epsilon1 -2 ) for robust policy evaluation and robust average reward estimation, marking a significant advancement in robust reinforcement learning theory.

## 1 Introduction

Reinforcement learning (RL) has achieved notable success in domains such as robotics [33], finance [17], healthcare [44], transportation [1], and large language models [30] by enabling agents to learn optimal decision-making strategies through interaction with an environment. However, in many real-world applications, direct interaction is impractical due to safety concerns, high costs, or limited data collection budgets [32, 19]. This challenge is particularly evident in scenarios where agents are trained in simulated environments before being deployed in the real world, such as in robotic control and autonomous driving. The mismatch between simulated and real environments, known as the simulation-to-reality gap, often leads to performance degradation when the learned policy encounters unmodeled uncertainties. Robust reinforcement learning (robust RL) addresses this challenge by formulating the learning problem as an optimization over an uncertainty set of transition probabilities, ensuring reliable performance under worst-case conditions. In this work, we focus on the problem of evaluating the robust value function and robust average reward for a given policy using only data sampled from a simulator (nominal model), aiming to enhance generalization and mitigate the impact of transition uncertainty in real-world deployment.

Reinforcement learning problems under infinite time horizons are typically studied under two primary reward formulations: the discounted-reward setting, where future rewards are exponentially

discounted, and the average-reward setting, which focuses on optimizing long-term performance. While the discounted-reward formulation is widely used, it may lead to myopic policies that underperform in applications requiring sustained long-term efficiency, such as queueing systems, inventory management, and network control. In contrast, the average-reward setting is more suitable for environments where decisions impact long-term operational efficiency. Despite its advantages, robust reinforcement learning under the average-reward criterion remains largely unexplored. Existing works on robust average-reward RL primarily provide asymptotic guarantees [37, 39, 38], lacking algorithms with finite-time performance bounds. This gap highlights the need for principled approaches that ensure robustness against model uncertainties while maintaining strong long-term performance guarantees.

Solving the robust average-reward reinforcement learning problem is significantly more challenging than its non-robust counterpart, with the primary difficulty arising in policy evaluation. Specifically, the goal is to compute the worst-case value function and worst-case average reward over an entire uncertainty set of transition models while having access only to samples from a nominal transition model. In this paper, we investigate three types of uncertainty sets: Contamination uncertainty sets, total variation (TV) distance uncertainty sets, and Wasserstein distance uncertainty sets. Unlike the standard average-reward setting, where value functions and average rewards can be estimated directly from observed trajectories [41, 2, 12, 13, 14], the robust setting introduces an additional layer of complexity due to the need to optimize against adversarial transitions. Consequently, conventional approaches based on direct estimation such as [41, 2, 12, 13, 14] immediately fail, as they do not account for the worst-case nature of the problem. Overcoming this challenge requires new algorithmic techniques that can infer the worst-case dynamics using only limited samples from the nominal model.

## 1.1 Challenges and Contributions

A common approach to policy evaluation in robust RL is to solve the corresponding robust Bellman operator. However, robust average-reward RL presents additional difficulties compared to the robust discounted-reward setting. In the discounted case, the presence of a discount factor induces a contraction property in the robust Bellman operator [40, 46], facilitating stable iterative updates. In contrast, the average-reward Bellman operator lacks a contraction property with respect to any norm even in the non-robust setting [45], making standard fixed-point analysis inapplicable. Due to this fundamental limitation, existing works on robust average-reward RL such as [39] rely on asymptotic techniques, primarily leveraging ordinary differential equation (ODE) analysis to examine the behavior of temporal difference (TD) learning. These methods exploit the asymptotic stability of the corresponding ODE [6] to establish almost sure convergence but fail to provide finite-sample performance guarantees. Addressing this limitation requires novel analytical tools and algorithmic techniques capable of providing explicit finite-sample bounds for robust policy evaluation and optimization.

In this work, we first establish and exploit a key structural property of the robust average-reward Bellman operator with uncertainty set P under the ergodicity of the nominal model: it is a contraction under some semi-norm, denoted as ‖ · ‖ P , where the detailed construction is specified in Theorem 4.2 and (15). Constructing ‖ · ‖ P is not straightforward, because ergodicity alone only guarantees that the chain mixes over multiple steps and fails to produce a single-step contraction for familiar measures such as the span semi-norm. To overcome this, we group together all the worst-case transition dynamics under uncertainty into one compact family of linear mappings, and observe that their 'worst-case gain' over any number of steps stays strictly below 1 . From this we build an extremal norm, which by construction shrinks every non-constant component by the same fixed factor in a single step. Finally, we add a small 'quotient' correction that exactly annihilates constant shifts, producing a semi-norm that vanishes only on constant functions but still inherits the one-step shrinkage. The above construction yields a uniform, strict contraction for the robust Bellman operator.

This fundamental result above enables the use of stochastic approximation techniques similar to [45] to analyze and bound the error in policy evaluation, overcoming the lack of a standard contraction property that has hindered prior finite-sample analyses. Building on this insight, we develop a novel stochastic approximation framework tailored to the robust average-reward setting. Our approach simultaneously estimates both the robust value function and the robust average reward, leading to

an efficient iterative procedure for solving the robust Bellman equation. A critical challenge in this framework under TV and Wasserstein distance uncertainty sets is accurately estimating the worstcase transition effects, which requires computing the support function of the uncertainty set. While previous works [4, 5, 39] have leveraged Multi-Level Monte Carlo (MLMC) for this task, their MLMC-based estimators suffer from infinite expected sample complexity due to the unbounded nature of the required geometric sampling, leading to only asymptotic convergence. To address this, we introduce a truncation mechanism based on a truncated geometric distribution, ensuring that the sample complexity remains finite while maintaining an exponentially decaying bias. With these techniques, we derive the first finite-sample complexity guarantee for policy evaluation in robust average-reward RL, achieving an optimal ˜ O ( /epsilon1 -2 ) sample complexity bound. The main contributions of this paper are summarized as follows:

- We prove that under the ergodicity assumption of the nominal model, the robust average-reward Bellman operator is a contraction with respect to a suitably constructed semi-norm (Theorem 4.2). This key result enables the application of stochastic approximation techniques for policy evaluation.
- Weprove the convergence of stochastic approximation under the semi-norm contraction and under i.i.d. with noise with non-zero bias (Theorem B.1) as an intermediate result.
- We develop an efficient method for computing estimates for the robust Bellman operator under TV distance and Wasserstein distance uncertainty sets. By modifying MLMC with a truncated geometric sampling scheme, we ensure finite expected sample complexity while keeping variance controlled and bias decaying exponentially with truncation level (Theorem 5.1-5.4).
- We propose a novel temporal difference learning method that iteratively updates the robust value function and the robust average reward, facilitating efficient policy evaluation in robust averagereward RL. We establish the first non-asymptotic sample complexity result for policy evaluation in robust average-reward RL, proving an order-optimal ˜ O ( /epsilon1 -2 ) complexity for policy evaluation (Theorem 6.1), along with a ˜ O ( /epsilon1 -2 ) complexity for robust average-reward estimation (Theorem 6.2).

## 2 Related Work

The theoretical guarantees of robust average-reward reinforcement learning have been studied by the following works. [37] takes a model-based perspective, approximating robust average-reward MDPs with discounted MDPs and proving uniform convergence of the robust discounted value function as the discount factor approaches one, employing dynamic programming and Blackwell optimality arguments to characterize optimal policies. [39] proposes a model-free approach by developing robust relative value iteration (RVI) TD and Q-learning algorithms, proving their almost sure convergence using stochastic approximation, martingale theory, and Multi-Level Monte Carlo estimators to handle non-linearity in the robust Bellman operator. While these studies provide fundamental insights into robust average-reward RL, they do not establish explicit convergence rate guarantees due to the lack of contraction properties in the robust Bellman operator. In addition, [31, 35] study the policy optimization of average-reward robust MDPs assuming direct queries of the sub-gradient information.

Policy evaluation in robust discounted-reward reinforcement learning with finite sample guarantees has been extensively studied, with the key recent works [40, 46, 25, 24, 23] focusing on solving the robust Bellman equation by finding its fixed-point solution. This approach is made feasible by the contraction property of the robust Bellman operator under the sup-norm, which arises due to the presence of a discount factor γ &lt; 1 . However, this fundamental approach does not directly extend to the robust average-reward setting, where the absence of a discount factor removes the contraction property under any norm. As a result, existing robust discounted methods cannot be applied in the robust average-reward RL setting.

Recently, a growing body of concurrent work has established finite-sample guarantees for robust average-reward reinforcement learning. Model-based approaches include [29, 10], and [28] develops a model-free value-iteration method under contamination and /lscript p -ball uncertainty sets. While these results significantly advance the area, the specific problem of policy evaluation in robust average-reward MDPs has not yet been addressed in terms of sample complexity. Our work targets this gap.

## 3 Formulation

## 3.1 Robust average-reward MDPs.

For a robust MDP with state space S and action space A while |S| = S and |A| = A , the transition kernel is assumed to be in some uncertainty set P . At each time step, the environment transits to the next state according to an arbitrary transition kernel P ∈ P . In this paper, we focus on the ( s, a ) -rectangular compact uncertainty set [27, 22], i.e., P = ⊗ s,a P a s , where P a s ⊆ ∆( S ) , and ∆ denotes the probability simplex. Popular uncertainty sets include those defined by the contamination model [21, 40], total variation [26], and Wasserstein distance [15].

We investigate the worst-case average-reward over the uncertainty set of MDPs. Specifically, define the robust average-reward of a policy π as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where κ = ( P 0 , P 1 ... ) ∈ ⊗ n ≥ 0 P . It was shown in [37] that the worst case under the time-varying model is equivalent to the one under the stationary model:

Therefore, we limit our focus to the stationary model. We refer to the minimizers of (2) as the worst-case transition kernels for the policy π , and denote the set of all possible worst-case transition kernels by Ω π g , i.e., Ω π g /defines { P ∈ P : g π P = g π P } , where g π P denotes the average reward of policy π under the single transition P ∈ P :

<!-- formula-not-decoded -->

We focus on the model-free setting, where only samples from the nominal MDP denoted as ˜ P (the centroid of the uncertainty set) are available. We investigate the problem of robust policy evaluation and robust average reward estimation, which means for a given policy π , we aim to estimate the robust value function and the robust average reward. Throughout this paper, we make the following standard assumption regarding the structure of the induced Markov chain.

Assumption 3.1. The Markov chain induced by π is irreducible and aperiodic for the nominal model ˜ P .

In contrast to many current works on robust average-reward RL [37, 39, 38, 31, 28], Assumption 3.1 requires only that the center of the uncertainty set be irreducible and aperiodic. We note that when the radius of uncertainty sets is small enough, Assumption 3.1 can ensure that P π is irreducible and aperiodic for all P ∈ P . This ensures that, under any transition model within the uncertainty set, the policy π induces a single recurrent communicating class. A well-known result in average-reward MDPs states that under Assumption 3.1, the average reward is independent of the starting state, i.e., for any P ∈ P and all s, s ′ ∈ S , we have g π P ( s ) = g π P ( s ′ ) . Thus, we can drop the dependence on the initial state and simply write g π P as the robust average reward. We now formally define the robust value function V π P V by connecting it with the following robust Bellman equation:

Theorem 3.2 (Robust Bellman Equation, Theorem 3.1 in [39]) . If ( g, V ) is a solution to the robust Bellman equation

<!-- formula-not-decoded -->

where σ P a s ( V ) = min p ∈P a s p /latticetop V is denoted as the support function, then the scalar g corresponds to the robust average reward, i.e., g = g π P , and the worst-case transition kernel P V belongs to the set of minimizing transition kernels, i.e., P V ∈ Ω π g , where Ω π g /defines { P ∈ P : g π P = g π P } . Furthermore, the function V is unique up to an additive constant, where if V is a solution to the Bellman equation,

then we have V = V π P V + c e , where c ∈ R and e is the all-ones vector in R S , and V π P V is defined as the relative value function of the policy π under the single transition P V as follows:

<!-- formula-not-decoded -->

Theorem 3.2 implies that the robust Bellman equation (4) identifies both the worst-case average reward g and a corresponding value function V that is determined only up to an additive constant. In particular, σ P a s ( V ) represents the worst-case transition effect over the uncertainty set P a s . Unlike the robust discounted case, where the contraction property of the Bellman operator under the sup-norm enables straightforward fixed-point iteration, the robust average-reward Bellman equation does not induce contraction under any norm, making direct iterative methods inapplicable. Throughout the paper, we denote e as the all-ones vector in R S . We now characterize the explicit forms of σ P a s ( V ) for different compact uncertainty sets as follows:

Contamination Uncertainty Set The contamination uncertainty models outliers or rare faults [7]. Specifically, the δ -contamination uncertainty set is P a s = { (1 -δ ) ˜ P a s + δq : q ∈ ∆( S ) } , where 0 &lt; δ &lt; 1 is the radius. Under this uncertainty set, the support function can be computed as

<!-- formula-not-decoded -->

and this is linear in the nominal transition kernel ˜ P a s .

Total Variation Uncertainty Set. The total variation (TV) distance uncertainty set models categorical misspecification or discretization error [18], and is characterized as P a s = { q ∈ ∆( |S| ) : 1 2 ‖ q -˜ P a s ‖ 1 ≤ δ } , define ‖ · ‖ sp as the span semi-norm and the support function can be computed using its dual function [22]:

<!-- formula-not-decoded -->

Wasserstein Distance Uncertainty Sets. The Wasserstein distance uncertainty Models smooth model drift when states have a geometry [11]. Consider the metric space ( S , d ) by defining some distance metric d . For some parameter l ∈ [1 , ∞ ) and two distributions p, q ∈ ∆( S ) , define the l -Wasserstein distance between them as W l ( q, p ) = inf µ ∈ Γ( p,q ) ‖ d ‖ µ,l , where Γ( p, q ) denotes the distributions over S × S with marginal distributions p, q , and ‖ d ‖ µ,l = ( E ( X,Y ) ∼ µ [ d ( X,Y ) l ]) 1 /l . The Wasserstein distance uncertainty set is then defined as

The support function w.r.t. the Wasserstein distance set, can be calculated as follows [15]:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 3.2 Robust Bellman Operator

Motivated by Theorem 3.2, we define the robust Bellman operator, which forms the basis for our policy evaluation procedure.

Definition 3.3 (Robust Bellman Operator, [39]) . The robust Bellman operator T g is defined as:

<!-- formula-not-decoded -->

The operator T g transforms a value function V by incorporating the worst-case transition effect. A key challenge in solving the robust Bellman equation is that T g does not satisfy contraction under standard norms, preventing the use of conventional fixed-point iteration. To cope with this problem, we establish that T g is a contraction under some constructed semi-norm. This allows us to further develop provably efficient stochastic approximation algorithms.

## 4 Semi-Norm Contraction of Robust Bellman Operators

Under Assumption 3.1, we are able to establish the semi-norm contraction property. For motivation, we first establish the semi-norm contraction property of the non-robust average-reward Bellman operator for a policy π under transition P defined as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 4.1. Let S be a finite state space, and let π be a stationary policy. If the Markov chain induced by π under the transition P is irreducible and aperiodic, there exists a semi-norm ‖·‖ P with kernel { c e : c ∈ R } and a constant β ∈ (0 , 1) such that for all V 1 , V 2 ∈ R S and any g ∈ R ,

Proof Sketch Under ergodicity, the one-step transition matrix (denoted as P π ) has a unique stationary distribution d π , define the stationary projector E = e /latticetop d π , then the fluctuation matrix (defined as Q π = P π -E ) has all eigenvalues strictly inside the unit circle. Standard finite-dimensional theory (via the discrete Lyapunov equation [20]) would produce a norm ‖· ‖ Q on R S such that there is a constant α ∈ (0 , 1) such that for any x ∈ R S , ‖ Q π x ‖ Q ≤ α ‖ x ‖ Q . We then build the semi-norm as follows:

<!-- formula-not-decoded -->

so that its kernel is exactly the constant vectors (the second term vanishes only on shifts of e ) and the first term enforces a one-step shrinkage by β = α + /epsilon1 &lt; 1 . A short calculation then shows ‖ P π x ‖ P ≤ β ‖ x ‖ P , yielding the desired contraction, which leads to the overall result.

The concrete proof of Lemma 4.1 including the detailed construction of the semi-norm ‖ · ‖ P is in Appendix A.1, where the properties of irreducible and aperiodic finite state Markov chain are utilized. Thus, we show the (non-robust) average-reward Bellman operator T P g is a strict contraction under ‖ · ‖ P . Based on the above motivations, we now formally establish the contraction property of the robust average-reward Bellman operator by leveraging Lemma 4.1 and the compactness of the uncertainty sets.

Theorem 4.2. Under Assumption 3.1, if P is compact, with certain restrictions on the radius of the uncertainty sets, there exists a semi-norm ‖ · ‖ P with kernel { c e : c ∈ R } such that the robust Bellman operator T g is a contraction. Specifically, there exist γ ∈ (0 , 1) such that

<!-- formula-not-decoded -->

Proof Sketch For any P ∈ P , the one-step transition matrix P π has a unique stationary projector E P due to ergodicity. Since P is compact, the family of fluctuation matrices { Q π P = P π -E P : P ∈ P} has joint spectral radius strictly less than 1 . By Lemma F.1in [3], one is able to construct an 'extremal norm' (denoted as ‖·‖ ext ) under which every Q π P contracts by a uniform factor α ∈ (0 , 1) . Mimicking the non-robust case in Lemma 4.1, we similarly define

<!-- formula-not-decoded -->

The supremum term zeros out if x ∈ { c e : c ∈ R } , and it inherits the uniform one-step shrinkage by α . Adding the small quotient term fixes the kernel without spoiling γ = α + /epsilon1 &lt; 1 , so one shows at once

<!-- formula-not-decoded -->

The above leads to the desired results.

The concrete proof of Theorem 4.2 along with the detailed construction of the semi-norm ‖ · ‖ P and the specific radius restrictions on various uncertainty sets are in Appendix A.2. Since all the uncertainty sets listed in Section 3.1 are closed and bounded in a real vector space, these uncertainty sets are all compact and satisfy the contraction property in Theorem 4.2. We also note that the contraction factor γ relates to the joint spectral gap of the family { Q π P : P ∈ P} .

## 5 Efficient Estimators for Uncertainty Sets

To utilize the contraction property in Section 4 to obtain convergence rate results, our idea is perform the following iterative stochastic approximation:

<!-- formula-not-decoded -->

where the learning rate η t would be specified in Section 6. The detailed analysis and complexities of the general stochastic approximation in the form of (17) is provided in Appendix B. Theorem B.1 implies that if ˆ T g ( V ) , being an estimator of T g ( V ) , could be constructed with bounded variance and small bias, V t converges to a solution of the Bellman equation in (4). However, the challenge of constructing our desired ˆ T g ( V ) lies in the construction of the support function estimator ˆ σ P a s ( V ) .

In this section, we aim to construct an estimator ˆ σ P a s ( V ) for all s ∈ S and a ∈ A in various uncertainty sets. Recall that the support function σ P a s ( V ) represents the worst-case transition effect over the uncertainty set P a s as defined in the robust Bellman equation in Theorem 3.2. The explicit forms of σ P a s ( V ) for different uncertainty sets were characterized in (6)-(9). Our goal in this section is to construct efficient estimators ˆ σ P a s ( V ) that approximates σ P a s ( V ) while maintaining controlled variance and finite sample complexity.

Linear Contamination Uncertainty Set Recall that the δ -contamination uncertainty set is P a s = { (1 -δ ) ˜ P a s + δq : q ∈ ∆( S ) } , where 0 &lt; δ &lt; 1 is the radius. Since the support function can be computed by (6) and the expression is linear in the nominal transition kernel ˜ P a s . A direct approach is to use the transition to the subsequent state to construct our estimator:

<!-- formula-not-decoded -->

where s ′ is a subsequent state sample after ( s, a ) . Hence, the sample complexity of (18) is just one. Lemma F.3 from [39] states that ˆ σ P a s ( V ) obtained by (18) is unbiased and has bounded variance as follows:

Nonlinear Contamination Sets Regarding TV and Wasserstein distance uncertainty sets, they have a nonlinear relationship between the nominal distribution ˜ P a s and the support function σ P a s ( V ) . Previous works such as [4, 5, 39] have proposed a Multi-Level Monte-Carlo (MLMC) method for obtaining an unbiased estimator of σ P a s ( V ) with bounded variance. However, their approaches require drawing 2 N +1 samples where N is sampled from a geometric distribution Geom(Ψ) with parameter Ψ ∈ (0 , 0 . 5) . This operation would need infinite samples in expectation for obtaining each single estimator as E [2 N +1 ] = ∑ ∞ N =0 2 N +1 Ψ(1 -Ψ) N = ∑ ∞ N =0 2Ψ(2 -2Ψ) N → ∞ . To handle the above problem, we aim to provide an estimator ˆ σ P a s ( V ) with finite sample complexity and small enough bias. We construct a truncated-MLMC estimator under geometric sampling with parameter Ψ = 0 . 5 as shown in Algorithm 1.

<!-- formula-not-decoded -->

In particular, if n &lt; N max , then { N ′ = n } = { N = n } with probability ( 1 2 ) n +1 , while { N ′ = N max } has probability ∑ ∞ m = N max (1 / 2) m +1 = 2 -N max . After obtaining N ′ , Algorithm 1 then collects a set of 2 N ′ +1 i.i.d. samples from the nominal transition model to construct empirical estimators for different transition distributions. The core of the approach lies in computing the support function estimates for TV and Wasserstein uncertainty sets using a correction term ∆ N ′ ( V ) , which accounts for the bias introduced by truncation. This correction ensures that the final estimator maintains a low bias while achieving a finite sample complexity. This truncation technique has been widely used in prior work across different settings such as [36, 14, 43]. We now present several crucial properties of Algorithm 1.

Theorem 5.1 (Finite Sample Complexity) . Under Algorithm 1, denote M = 2 N ′ +1 as the random number of samples (where N ′ = min { N,N max } ). Then

<!-- formula-not-decoded -->

The proof of Theorem 5.1 is in Appendix C.1, which demonstrates that setting the geometric sampling parameter to Ψ = 0 . 5 ensures that the expected number of samples follows a linear growth

## Algorithm 1 Truncated MLMC Estimator for TV and Wasserstein Uncertainty Sets

- 1: Sample N ∼ Geom(0 . 5)

Input : s ∈ S , a ∈ A , Max level N max , Value function V

- 2: N ′ ← min { N,N max } 3: Collect 2 N ′ +1 i.i.d. samples of { s ′ i } 2 N ′ +1 i =1 with s ′ i ∼ ˜ P a s for each i

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

9:

else if

Wasserstein then

Obtain

σ

ˆ

a,

1

s,N

′

+1

a

s,N

′

+1

P

a,E

s,N

′

+1

P

a,O

s,N

′

+1

<!-- formula-not-decoded -->

pattern rather than an exponential one. This choice precisely cancels out the effect of the exponential sampling inherent in the truncated MLMC estimator, preventing infinite expected sample complexity. This result shows that the expected number of queries grows only linearly with N max , ensuring that the sampling cost remains manageable even for large truncation levels. The key factor enabling this behavior is setting the geometric distribution parameter to 0 . 5 , which balances the probability mass across different truncation levels, preventing an exponential increase in sample complexity.

Theorem 5.2 (Exponentially Decaying Bias) . Let ˆ σ P a s ( V ) be the estimator of σ P a s ( V ) obtained from Algorithm 1 then under TV uncertainty set, we have:

where δ denotes the radius of TV distance. Under Wasserstein uncertainty set, we have:

<!-- formula-not-decoded -->

∣ ∣ E [ ˆ σ P a s ( V ) -σ P a s ( V ) ]∣ ∣ ≤ 6 · 2 -N max 2 ‖ V ‖ sp (22) Theorem 5.2 establishes that the bias of the truncated MLMC estimator decays exponentially with N max , ensuring that truncation does not significantly affect accuracy. This result follows from observing that the deviation introduced by truncation can be expressed as a sum of differences between support function estimates at different level, and each of which is controlled by the /lscript 1 -distance between transition distributions. Thus, we can use binomial concentration property to ensure the exponentially decaying bias.

The proof of Theorem 5.2 is in Appendix C.2. One important lemma used in the proof is the following Lemma 5.3, where we show the Lipschitz property for both TV and Wasserstein distance uncertainty sets.

Lemma 5.3. For any p, q ∈ ∆( S ) , let P TV and Q TV denote the TV distance uncertainty set with radius δ centering at p and q respectively, and let P W and Q W denote the Wasserstein distance uncertainty set with radius δ centering at p and q respectively. Then for any value function V , we have:

<!-- formula-not-decoded -->

We refer the proof of Theorem 5.2 to Appendix C.3.

Theorem 5.4 (Linear Variance) . Let ˆ σ P a s ( V ) be the estimator of σ P a s ( V ) obtained from Algorithm 1 then under TV distance uncertainty set, we have:

<!-- formula-not-decoded -->

P

(

V

)

, σ

ˆ

P

(

V

)

, σ

ˆ

(

V

)

, σ

ˆ

(

V

)

from (9)

and under Wasserstein distance uncertainty set, we have:

<!-- formula-not-decoded -->

Theorem 5.4 establishes that the variance of the truncated MLMC estimator grows linearly with N max , ensuring that the estimator remains stable even as the truncation level increases. The proof of Theorem 5.4 is in Appendix C.4, which follows from bounding the second moment of the estimator by analyzing the variance decomposition across different MLMC levels. Specifically, by expressing the estimator in terms of successive refinements of the transition model, we show that the variance accumulates additively across levels due to the binomial concentration property.

## 6 Robust Average-Reward TD Learning

Equipped with the methods of constructing ˆ σ P a s ( V ) for all s ∈ S and a ∈ A , we now present the formal algorithm for robust policy evaluation and robust average reward for a given policy π in Algorithm 2. Algorithm 2 presents a robust temporal difference (TD) learning method for policy evaluation in robust average-reward MDPs. This algorithm builds upon the truncated MLMC estimator (Algorithm 1) and the biased stochastic approximation framework in Section B, ensuring both efficient sample complexity and finite-time convergence guarantees.

The algorithm is divided into two main phases. The first phase (Lines 1-7) estimates the robust value function. The noisy Bellman operator is computed using the estimator ˆ σ P a s ( V t ) obtained depending on the uncertainty set type. Then the iterative update follows a stochastic approximation scheme with stepsize η t , ensuring convergence while maintaining stability. Finally, the value function is centered at an anchor state s 0 to remove the ambiguity due to its additive invariance. The second phase (Lines 8-14) estimates the robust average reward by utilizing V T from the output of the first phase. The expected Bellman residual δ t ( s ) is computed across all states and averaging it to obtain ¯ δ t . Aseparate stochastic approximation update with stepsize β t is then applied to refine g t , ensuring convergence to the robust worst-case average reward. By combining these two phases, Algorithm 2 provides an efficient and provably convergent method for robust policy evaluation under averagereward criteria, marking a significant advancement over prior methods that only provided asymptotic guarantees.

## Algorithm 2 Robust Average-Reward TD

```
Input : Policy π , Initial values V 0 , g 0 = 0 , Stepsizes η t , β t , Max level N max , Anchor state s 0 ∈ S 1: for t = 0 , 1 , . . . , T -1 do 2: for each ( s, a ) ∈ S × A do 3: if Contamination then Sample ˆ σ P a s ( V t ) according to (18) 4: else if TV or Wasserstein then Sample ˆ σ P a s ( V t ) according to Algorithm 1 5: end if 6: end for 7: ˆ T g 0 ( V t )( s ) ← ∑ a π ( a | s ) [ r ( s, a ) -g 0 + ˆ σ P a s ( V t ) ] , ∀ s ∈ S 8: V t +1 ( s ) ← V t ( s ) + η t ( ˆ T g 0 ( V t )( s ) -V t ( s ) ) , ∀ s ∈ S 9: V t +1 ( s ) = V t +1 ( s ) -V t +1 ( s 0 ) , ∀ s ∈ S 10: end for 11: for t = 0 , 1 , . . . , T -1 do 12: for each ( s, a ) ∈ S × A do 13: if Contamination then Sample ˆ σ P a s ( V t ) according to (18) 14: else if TV or Wasserstein then Sample ˆ σ P a s ( V t ) according to Algorithm 1 15: end if 16: end for 17: ˆ δ t ( s ) ← ∑ a π ( a | s ) [ r ( s, a ) + ˆ σ P a s ( V T ) ] -V T ( s ) , ∀ s ∈ S 18: ¯ δ t ← 1 S ∑ s ˆ δ t ( s ) 19: g t +1 ← g t + β t ( ¯ δ t -g t ) 20: end for return V T , g T
```

To derive the sample complexity of robust policy evaluation, we utilize the semi-norm contraction property of the Bellman operator in Theorem 4.2, and fit Algorithm 2 into the general biased stochastic approximation result in Theorem B.1 while incorporating the bias analysis characterized in Section 5. Since each phase of Algorithm 2 contains a loop of length T with all the states and actions updated together, the total samples needed for the entire algorithm in expectation is 2 SAT E [ N max ] , where E [ N max ] is one for contamination uncertainty sets and is O ( N max ) from Theorem 5.1 for TV and Wasserstein distance uncertainty sets.

Theorem 6.1. If V t is generated by Algorithm 2 and satisfying Assumption 3.1, then if the stepsize η t := O ( 1 t ) , we require a sample complexity of O ( SAt 2 mix /epsilon1 2 (1 -γ ) 2 ) for contamination uncertainty set and a sample complexity of ˜ O ( SAt 2 mix /epsilon1 2 (1 -γ ) 2 ) for TV and Wasserstein distance uncertainty set to ensure an /epsilon1 convergence of V T . Moreover, these results are order-optimal in terms of /epsilon1 .

Theorem 6.2. If g t is generated by Algorithm 2 and satisfying Assumption 3.1, then if the stepsize β t := O ( 1 t ) , we require a sample complexity of ˜ O ( SAt 2 mix /epsilon1 2 (1 -γ ) 2 ) for all contamination, TV, and Wasserstein distance uncertainty set to ensure an /epsilon1 convergence of g T .

The formal version of Theorems 6.1 and 6.2 along with the proofs are in Appendix D. Theorem 6.1 provides the order-optimal sample complexity of ˜ O ( /epsilon1 -2 ) for Algorithm 2 to achieve an /epsilon1 -accurate estimate of V T . Although Theorem 6.1 claims order-optimal in terms of /epsilon1 , we do not claim tightness in S , A and γ , and treat sharpening these dependencies as open. The proof of Theorem 6.2 extends the analysis of Theorem 6.1 to robust average reward estimation. The key difficulty lies in controlling the propagation of error from value function estimates to reward estimation. By again leveraging the contraction property and appropriately tuning stepsizes, we establish an ˜ O ( /epsilon1 -2 ) complexity bound for robust average reward estimation.

## 7 Conclusion

This paper provides the first finite-sample analysis for policy evaluation in robust average-reward MDPs, bridging a gap where only asymptotic guarantees existed. By introducing a biased stochastic approximation framework and leveraging the properties of various uncertainty sets, we establish finite-time convergence under biased noise. Our algorithm achieves an order-optimal sample complexity of ˜ O ( /epsilon1 -2 ) for policy evaluation, despite the added complexity of robustness.

A crucial step in our analysis is proving that the robust Bellman operator is contractive under our constructed semi-norm ‖· ‖ P , ensuring the validity of stochastic approximation updates. We further develop a truncated Multi-Level Monte Carlo estimator that efficiently computes worst-case value functions under total variation and Wasserstein uncertainty, while keeping bias and variance controlled. One limitation of this work is that the results require ergodicity to hold in the setting, as stated in Assumption 3.1. Additionally, scaling the algorithm and results in the paper via function approximations remains an important open problem.

## Acknowledgments

We would like to thank Zaiwei Chen of Purdue University for assistance with identifying relevant literature and for constructive feedback. We also thank Zijun Chen and Nian Si of The Hong Kong University of Science and Technology for helpful discussions regarding Appendix A. Finally, we thank the anonymous reviewers for insightful comments that substantially improved the paper.

## References

- [1] Abubakr O Al-Abbasi, Arnob Ghosh, and Vaneet Aggarwal. Deeppool: Distributed model-free algorithm for ride-sharing using deep reinforcement learning. IEEE Transactions on Intelligent Transportation Systems , 20(12):4714-4727, 2019.
- [2] Qinbo Bai, Washim Uddin Mondal, and Vaneet Aggarwal. Regret analysis of policy gradient algorithm for infinite horizon average reward markov decision processes. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pages 10980-10988, 2024.

- [3] Marc A Berger and Yang Wang. Bounded semigroups of matrices. Linear Algebra and its Applications , 166:21-27, 1992.
- [4] Jose H Blanchet and Peter W Glynn. Unbiased Monte Carlo for optimization and functions of expectations via multi-level randomization. In 2015 Winter Simulation Conference (WSC) , pages 3656-3667. IEEE, 2015.
- [5] Jose H Blanchet, Peter W Glynn, and Yanan Pei. Unbiased multilevel Monte Carlo: Stochastic optimization, steady-state simulation, quantiles, and other applications. arXiv preprint arXiv:1904.09929 , 2019.
- [6] Vivek S Borkar and Vivek S Borkar. Stochastic approximation: a dynamical systems viewpoint , volume 100. Springer, 2008.
- [7] Mengjie Chen, Chao Gao, and Zhao Ren. A general decision theory for huber's /epsilon1 -contamination model. Electronic Journal of Statistics , 10:3752-3774, 2016.
- [8] Zaiwei Chen, Siva Theja Maguluri, Sanjay Shakkottai, and Karthikeyan Shanmugam. Finitesample analysis of contractive stochastic approximation using smooth convex envelopes. Advances in Neural Information Processing Systems , 33:8223-8234, 2020.
- [9] Zaiwei Chen, Sheng Zhang, Zhe Zhang, Shaan Ul Haque, and Siva Theja Maguluri. A nonasymptotic theory of seminorm lyapunov stability: From deterministic to stochastic iterative algorithms. arXiv preprint arXiv:2502.14208 , 2025.
- [10] Zijun Chen, Shengbo Wang, and Nian Si. Sample complexity of distributionally robust averagereward reinforcement learning. arXiv preprint arXiv:2505.10007 , 2025.
- [11] Julien Grand Clement and Christian Kroer. First-order methods for wasserstein distributionally robust mdp. In International Conference on Machine Learning , pages 2010-2019. PMLR, 2021.
- [12] Swetha Ganesh and Vaneet Aggarwal. Regret analysis of average-reward unichain mdps via an actor-critic approach. Advances in Neural Information Processing Systems , 2025.
- [13] Swetha Ganesh, Washim Uddin Mondal, and Vaneet Aggarwal. Order-optimal regret with novel policy gradient approaches in infinite-horizon average reward mdps. In International Conference on Artificial Intelligence and Statistics , pages 3421-3429. PMLR, 2025.
- [14] Swetha Ganesh, Washim Uddin Mondal, and Vaneet Aggarwal. A sharper global convergence analysis for average reward reinforcement learning via an actor-critic approach. In Fortysecond International Conference on Machine Learning , 2025.
- [15] Rui Gao and Anton Kleywegt. Distributionally robust stochastic optimization with wasserstein distance. Mathematics of Operations Research , 48(2):603-655, 2023.
- [16] Stéphane Gaubert and Zheng Qu. Dobrushin's ergodicity coefficient for markov operators on cones. Integral Equations and Operator Theory , 81(1):127-150, 2015.
- [17] Ben Hambly, Renyuan Xu, and Huining Yang. Recent advances in reinforcement learning in finance. Mathematical Finance , 33(3):437-503, 2023.
- [18] Chin Pang Ho, Marek Petrik, and Wolfram Wiesemann. Partial policy iteration for l1-robust markov decision processes. Journal of Machine Learning Research , 22(275):1-46, 2021.
- [19] Sebastian Höfer, Kostas Bekris, Ankur Handa, Juan Camilo Gamboa, Melissa Mozifian, Florian Golemo, Chris Atkeson, Dieter Fox, Ken Goldberg, John Leonard, et al. Sim2real in robotics and automation: Applications and challenges. IEEE transactions on automation science and engineering , 18(2):398-400, 2021.
- [20] Roger A Horn and Charles R Johnson. Matrix analysis . Cambridge university press, 2012.
- [21] Peter J Huber. A robust version of the probability ratio test. The Annals of Mathematical Statistics , pages 1753-1758, 1965.

- [22] Garud N Iyengar. Robust dynamic programming. Mathematics of Operations Research , 30(2):257-280, 2005.
- [23] Yufei Kuang, Miao Lu, Jie Wang, Qi Zhou, Bin Li, and Houqiang Li. Learning robust policy against disturbance in transition dynamics via state-conservative policy optimization. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 36, pages 7247-7254, 2022.
- [24] Navdeep Kumar, Esther Derman, Matthieu Geist, Kfir Y Levy, and Shie Mannor. Policy gradient for rectangular robust markov decision processes. Advances in Neural Information Processing Systems , 36:59477-59501, 2023.
- [25] Yan Li, Guanghui Lan, and Tuo Zhao. First-order policy optimization for robust markov decision process. arXiv preprint arXiv:2209.10579 , 2022.
- [26] Shiau Hong Lim, Huan Xu, and Shie Mannor. Reinforcement learning in robust markov decision processes. Advances in Neural Information Processing Systems , 26, 2013.
- [27] Arnab Nilim and Laurent El Ghaoui. Robustness in markov decision problems with uncertain transition matrices. Advances in neural information processing systems , 16, 2003.
- [28] Zachary Roch, Chi Zhang, George Atia, and Yue Wang. A finite-sample analysis of distributionally robust average-reward reinforcement learning. arXiv preprint arXiv:2505.12462 , 2025.
- [29] Zachary Andrew Roch, George K Atia, and Yue Wang. A reduction framework for distributionally robust reinforcement learning under average reward. In Forty-second International Conference on Machine Learning , 2025.
- [30] Saksham Sahai Srivastava and Vaneet Aggarwal. A technical survey of reinforcement learning techniques for large language models. arXiv preprint arXiv:2507.04136 , 2025.
- [31] Zhongchang Sun, Sihong He, Fei Miao, and Shaofeng Zou. Policy optimization for robust average reward mdps. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [32] Niko Sünderhauf, Oliver Brock, Walter Scheirer, Raia Hadsell, Dieter Fox, Jürgen Leitner, Ben Upcroft, Pieter Abbeel, Wolfram Burgard, Michael Milford, et al. The limits and potentials of deep learning for robotics. The International journal of robotics research , 37(4-5):405-420, 2018.
- [33] Chen Tang, Ben Abbatematteo, Jiaheng Hu, Rohan Chandra, Roberto Martín-Martín, and Peter Stone. Deep reinforcement learning for robotics: A survey of real-world successes. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 39, pages 28694-28698, 2025.
- [34] Jinghan Wang, Mengdi Wang, and Lin F Yang. Near sample-optimal reduction-based policy learning for average reward mdp. arXiv preprint arXiv:2212.00603 , 2022.
- [35] Qiuhao Wang, Yuqi Zha, Chin Pang Ho, and Marek Petrik. Provable policy gradient for robust average-reward mdps beyond rectangularity. In Forty-second International Conference on Machine Learning , 2025.
- [36] Yudan Wang, Shaofeng Zou, and Yue Wang. Model-free robust reinforcement learning with sample complexity analysis. In Uncertainty in Artificial Intelligence , pages 3470-3513. PMLR, 2024.
- [37] Yue Wang, Alvaro Velasquez, George Atia, Ashley Prater-Bennette, and Shaofeng Zou. Robust average-reward markov decision processes. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 37, pages 15215-15223, 2023.
- [38] Yue Wang, Alvaro Velasquez, George Atia, Ashley Prater-Bennette, and Shaofeng Zou. Robust average-reward reinforcement learning. Journal of Artificial Intelligence Research , 80:719803, 2024.

- [39] Yue Wang, Alvaro Velasquez, George K Atia, Ashley Prater-Bennette, and Shaofeng Zou. Model-free robust average-reward reinforcement learning. In International Conference on Machine Learning , pages 36431-36469. PMLR, 2023.
- [40] Yue Wang and Shaofeng Zou. Policy gradient method for robust reinforcement learning. In International conference on machine learning , pages 23484-23526. PMLR, 2022.
- [41] Chen-Yu Wei, Mehdi Jafarnia Jahromi, Haipeng Luo, Hiteshi Sharma, and Rahul Jain. Modelfree reinforcement learning in infinite-horizon average-reward markov decision processes. In International conference on machine learning , pages 10170-10180. PMLR, 2020.
- [42] Fabian Wirth. The generalized spectral radius and extremal norms. Linear Algebra and its Applications , 342(1-3):17-40, 2002.
- [43] Yang Xu and Vaneet Aggarwal. Accelerating quantum reinforcement learning with a quantum natural policy gradient based approach. In Forty-second International Conference on Machine Learning , 2025.
- [44] Chao Yu, Jiming Liu, Shamim Nemati, and Guosheng Yin. Reinforcement learning in healthcare: A survey. ACM Computing Surveys (CSUR) , 55(1):1-36, 2021.
- [45] Sheng Zhang, Zhe Zhang, and Siva Theja Maguluri. Finite sample analysis of average-reward td learning and q -learning. Advances in Neural Information Processing Systems , 34:12301242, 2021.
- [46] Ruida Zhou, Tao Liu, Min Cheng, Dileep Kalathil, PR Kumar, and Chao Tian. Natural actorcritic for robust reinforcement learning with function approximation. Advances in neural information processing systems , 36, 2024.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The claims made in the abstract and introduction accurately reflects the novelty and contributions of this paper. All details can be found either in the rest of the main text or in the appendix.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The discussion of limitations can be found in the conclusion.

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

Justification: The assumptions are clearly stated in the main text, the theorem statements and some proof sketches are also included in the main text. The formal theorem statements and their complete proofs are in the appendix.

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

Justification: This is a theoretical paper and does not include experiments. The minor numerical examples added during the rebuttal will be released.

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

Justification: This is a theoretical paper and does not include experiments. The minor numerical examples added during the rebuttal will be released.

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

Justification: This is a theoretical paper and does not include experiments. The minor numerical examples added during the rebuttal will be released.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: This is a theoretical paper and does not include experiments. The minor numerical examples added during the rebuttal will be released.

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

Justification: This is a theoretical paper and does not include experiments. The minor numerical examples added during the rebuttal will be released. The minor numerical examples added during the rebuttal will be released.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in the paper is theoretical and conform, in every respect, the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This is a theoretical work and the algorithm could be applied in different applications. There is nothing specific that can be highlighted.

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

Justification: This is a theoretical work and the algorithm could be applied in different applications. There is nothing specific that can be highlighted.

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

Justification: The paper does not release new assets.the paper does not release new assets.

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

## A Semi-Norm Contraction Property of the Bellman Operator

## A.1 Proof of Lemma 4.1

For any V 1 , V 2 ∈ R S and define ∆ = V 1 -V 2 . Denote P π as the transition matrix under policy π and the unique stationary distribution d π , and denote E as the matrix with all rows being identical to d π . We further define Q π = P π -E . Thus, we would have,

<!-- formula-not-decoded -->

which implies

We now discuss the detailed construction of the semi-norm ‖ · ‖ P . Since P π is ergodic, according to the Perron-Frobenius theorem, P π has an eigenvalue λ 1 = 1 of algebraic multiplicity exactly one, with corresponding right eigenvector e . Moreover, all other eigenvalues λ 2 ≥ . . . ≥ λ S of P π satisfies | λ i | &lt; 1 for all i ∈ { 2 , . . . , S } .

<!-- formula-not-decoded -->

Lemma A.1. All eigenvalues of Q π lies strictly inside the unit circle.

Proof. Since E = e d π /latticetop , E is a rank-one projector onto the span of e . Hence the spectrum of E is { 1 , 0 , . . . , 0 } . In addition, we can show P π and E commute by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, by the Schur's theorem, P and E are simultaneously upper triangularizable. In a common triangular basis, the diagonals of P and E list their eigenvalues in descending orders, which are { λ 1 , λ 2 , . . . , λ S } and { 1 , 0 , . . . , 0 } respectively. Thus, in that same basis, Q π = P π -E is also triangular, with diagonal entries being { λ 1 -1 , λ 2 -0 , . . . , λ S -0 } . Since λ 1 = 1 , we have the spectrum of Q π is exactly { λ 2 , . . . , λ S , 0 } . Since we already have | λ i | &lt; 1 for all i ∈ { 2 , . . . , S } , we conclude the proof.

Define ρ ( · ) to be the spectral radius of a matrix, then Lemma A.1 implies that ρ ( Q π ) &lt; 1 . Hence by equivalence of norms in R |S| it is possible to construct a vector norm ‖ · ‖ Q so that the induced operator norm of Q π is less than 1 , specifically

<!-- formula-not-decoded -->

A concrete construction example is to leverage the discrete-Lyapunov equation [20] of solving M on the space of symmetric matrices for any ρ ( Q π ) &lt; α &lt; 1 as follows:

<!-- formula-not-decoded -->

Define B := α -1 Q π , then ρ ( B ) = α -1 ρ ( Q π ) &lt; 1 . We can express M in the form of Neumann series as

We now show that M is bounded. Write B = SJS -1 , where J = diag ( J m 1 ( λ 1 ) , . . . , J m r ( λ r ) ) is the Jordan normal form. By the Jordan block power formula [20],

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with N m the nilpotent matrix having ones on the first superdiagonal and N m m = 0 . Then B k = SJ k S -1 and J k = diag ( J m 1 ( λ 1 ) k , . . . , J m r ( λ r ) k ) . For each block and each integer k ≥ m , by the binomial theorem we have N m m = 0 and

<!-- formula-not-decoded -->

For k ≥ m , use ( k j ) ≤ k j j ! and factor | λ | k :

where c m,λ := ∑ m -1 j =0 | λ | -j j ! ‖ N j m ‖ 2 . Thus, let s = max i m i be the size of the largest Jordan block of B . Since J k is block diagonal,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since similarity does not change eigenvalues but may scale norms by the condition number, we can derive that for all k ≥ s , where C J := ∑ r i =1 c m i ,λ i and ρ ( B ) = max i | λ i | .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where κ ( S ) := ‖ S ‖ 2 ‖ S -1 ‖ 2 . By choosing the appropriate constant, the same bound holds for all k ≥ 0 :

In spectral norm, this implies

<!-- formula-not-decoded -->

Thus, the scalar series ∑ ∞ k =0 k 2( s -1) ρ ( B ) 2 k is in the form of polynomial times geometric with ratio less than 1 , which converges, and the partial sum expression in (32) converges absolutely as a geometric-type series.

Also, since each term in (32) is positive semi-definite, and the first term α -2 I being positive definite, we can conclude that M being the summation is well-defined and is positive definite. Thus, using the positive definite M defined in (31), we can define our desired norm ‖ · ‖ Q as which implies

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

/negationslash

Where ( a ) is because for any x = 0 , from (31) we have

/negationslash

<!-- formula-not-decoded -->

Since ‖ x ‖ 2 2 is always non-negative dividing both sides of the second equation of (40) by x /latticetop Mx and further taking the square root on both sides yields the inequality of ( a ) .

Based on the above construction of the norm ‖ · ‖ Q , define the operator ‖ · ‖ P as where 0 &lt; /epsilon1 &lt; 1 -α .

<!-- formula-not-decoded -->

Lemma A.2. The operator ‖ · ‖ P is a valid semi-norm with kernel being exactly { c e : c ∈ R } . Furthermore, for all x ∈ R S , we have ∥ ∥ P π x ∥ ∥ P ≤ ( α + /epsilon1 ) ∥ ∥ x ∥ ∥ P .

<!-- formula-not-decoded -->

Proof. Regarding positive homogeneity and nonnegativity, for any scalar λ and x ∈ R S , and clearly ‖ x ‖ P ≥ 0 , with equality only when both ‖ Q π x ‖ Q = 0 and inf c ‖ x -c e ‖ Q = 0 . Regarding triangle inequality, for any x, y ∈ R S ,

<!-- formula-not-decoded -->

Regarding the kernel, if x = k e for some k ∈ R , then we have

<!-- formula-not-decoded -->

On the other hand, if x / ∈ { c e : c ∈ R } , we know that

<!-- formula-not-decoded -->

Thus, the kernel of ‖ · ‖ P is exactly { c e : c ∈ R } . We now show that, for any x ∈ R S ,

<!-- formula-not-decoded -->

Let β = α + /epsilon1 , by (30) and (41), we have α ∈ (0 , 1) and /epsilon1 ∈ (0 , 1 -α ) . Thus, β ∈ (0 , 1) and combining β with the semi-norm ‖ · ‖ P confirms Lemma 4.1.

## A.2 Proof of Theorem 4.2

We override the terms α, λ and /epsilon1 from the previous section. For any V 1 , V 2 and s ∈ S , where ˜ p ( V 1 ,V 2 ) ( ·| s, a ) = arg max p ∈P a s [ ∑ s ′ ∈S p ( s ′ ) V 1 ( s ′ ) -∑ s ′ ∈S p ( s ′ ) V 2 ( s ′ )] and each ˜ p ( V 1 ,V 2 ) ∈ P for all V 1 , V 2 . We now discuss the construction of the desired semi-norm ‖ · ‖ P .

<!-- formula-not-decoded -->

## A.2.1 Joint Spectral Radius of Q π P

For any P ∈ P , denote P π as the transition matrix under policy π and the unique stationary distribution d π P , and denote E P as the matrix with all rows being identical to d π P (we will provide the conditions for all P π having a unique stationary distribution later). We further define the following:

<!-- formula-not-decoded -->

To obtain the desired one-step contraction result under Assumption 3.1 along with proper radius restrictions, we need to show the conditions of the radius under the different uncertainty sets such that the joint spectral radius ˆ ρ ( Q π P ) defined in Lemma F.1 satisfies ˆ ρ ( Q π P ) &lt; 1 , which is necessary to establish the desired one-step contraction. We first provide an upper bound of the joint spectral radius as follows:

Lemma A.3. Define the Dobrushin's coefficient of an n dimensional Markov matrix P as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then the joint spectral radius of the family Q π P is upper bounded by the following:

Proof. We start by first connecting ˆ ρ ( Q π P ) to the joint spectral radius of the family { P π : P ∈ P} . Define H := { x ∈ R S : e /latticetop x = 0 } to be the zero-sum subspace where the space spanned by e is removed. Furthermore, choose an orthonormal basis U = [ u 0 U H ] ∈ R S × S with

<!-- formula-not-decoded -->

where Π is the orthogonal projector onto H . Since U is orthogonal, U /latticetop = U -1 . With the above notations, for any Q π P ∈ Q π P , we can construct a similar matrix ˜ Q π P as

Equivalently, define T P := Π P π ∣ ∣ H , which operates entirely on H . Then B P is the matrix of T P in the basis of U H . Since E P = e ( d π P ) /latticetop and U /latticetop H e = 0 , we have U /latticetop H E P U H = 0 . Hence, the lower-right block in U /latticetop H Q π P U H is just B P . Consequently, for any sequence P π 1 , . . . , P π k ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, by block upper-triangularity [20], the spectral radius of Q π P k · · · Q π P 1 on R S equals the spectral radius of B P k · · · B P 1 on H :

Thus, the spectral radius of the family Q π P k · · · Q π P 1 on R S equals the spectral radius of P π k · · · P π 1 on H :

Given (52), we now study the joint spectral radius of P H = { Π P π : P ∈ P} on H .

<!-- formula-not-decoded -->

From Lemma F.1, we have that for an arbitrary norm ‖ · ‖ H → H on H ,

<!-- formula-not-decoded -->

Divide k into m partitions of blocks with length q and a residue r as k = qm + r , where q, m, r ∈ N , 0 &lt; q ≤ k and 0 ≤ r &lt; q . Furthermore, let M m := sup P i ∈P ‖ T P m . . . T P 1 ‖ H → H and K &lt;m := max 0 ≤ r&lt;m sup P i ∈P ‖ T P r . . . T P 1 ‖ H → H , then by the submultiplicity of operator norm, we have that on H ,

<!-- formula-not-decoded -->

taking power of 1 k and let k →∞ implies

<!-- formula-not-decoded -->

Since q = /floorleft k m /floorright , we have lim k →∞ q k = 1 m and lim k →∞ 1 k = 0 , which suggests that for any positive integer m we have,

<!-- formula-not-decoded -->

which implies for any norm ‖ · ‖ H → H on H , we have

<!-- formula-not-decoded -->

From [16], the Dobrushin's coefficient is a valid norm (the induced matrix span norm) on the zerosum subspace H , which yields (48).

Lemma A.3 provides a quantitative method to relate the joint spectral radius of the family Q π P and the Dobrushin's coefficient of the family P . Under Assumption 3.1, we next discuss the radius restrictions of contamination, TV and Wasserstein distance uncertainty sets such that ˆ ρ ( Q π P ) &lt; 1 is satisfied.

## A.2.2 Discussions on Radius Restrictions

We provide the following Lemma A.4-A.6, which quantifies the radius restrictions regarding all three uncertainty sets of interests for obtaining the desired results.

Contamination Uncertainty Regarding contamination uncertainty, where the uncertainty set is characterized as

For a fixed policy π , the induced state-transition matrix P π is expressed as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Define the induced uncertainty set P π := { P π : P ∈ P} and define ˜ P π ( s, s ′ ) := ∑ a π ( a | s ) ˜ P ( s ′ | s, a ) . Then (58) can be expressed as

Lemma A.4. Under the contamination uncertainty set, if the centroid ˜ P π is irreducible and aperiodic, then the joint spectral radius of Q π P defined in (46) is strictly less than 1 . Furthermore, P π is irreducible and aperiodic for all P ∈ P .

Proof. Since ˜ P π is irreducible and aperiodic. Then there exists an m ∈ N such that all entries in ( ˜ P π ) m are strictly positive. For any P π ∈ P π we can write P π = (1 -δ ) ˜ P π + δq π with q π being row-stochastic, so by multinomial expansion,

<!-- formula-not-decoded -->

Hence ( P π ) m is strictly positive for all P ∈ P , which implies every P π ∈ P π is primitive with the same exponent m , which further implies P π is irreducible and aperiodic for all P ∈ P .

<!-- formula-not-decoded -->

To bound the joint spectral radius, for the same integer m , define the m -step overlap constant of the centroid as

For any lengthm product P π m · · · P π 1 with P π t ∈ P π , the same entrywise bound (60) gives P π m · · · P π 1 ≥ (1 -δ ) m ( ˜ P π ) m , whence for all i = j , we have

/negationslash

<!-- formula-not-decoded -->

By the definition of the Dobrushin's coefficient in (47), the above yields

<!-- formula-not-decoded -->

By Lemma A.3,

<!-- formula-not-decoded -->

Therefore, if the center ˜ P π is primitive and 0 ≤ δ &lt; 1 , without having any additional restrictions on the radius, we have that all induced kernels in P π are irreducible and aperiodic. Furthermore, the joint spectral radius of Q π P satisfies ˆ ρ ( Q π P ) &lt; 1 .

Total Variation (TV) Distance Uncertainty Regarding TV uncertainty, where the uncertainty set is characterized as

<!-- formula-not-decoded -->

where TV( p, q ) := 1 2 ‖ p -q ‖ 1 . For a fixed policy π , the induced state-transition matrix P π is expressed as

Then for each state s ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

by convexity of TV( · , · ) in each argument. Hence

Lemma A.5. Under the TV distance uncertainty set, if the centroid ˜ P π is irreducible and aperiodic, then there exists m ∈ N such that ( ˜ P π ) m is strictly positive. Define b 0 = min i,s (( ˜ P π ) m ) is &gt; 0 , then if the radius satisfies δ &lt; b 0 m , the joint spectral radius of Q π P defined in (46) is strictly less than 1 . Furthermore, P π is irreducible and aperiodic for all P ∈ P .

<!-- formula-not-decoded -->

Proof. Define the m -step constant a 0 as

<!-- formula-not-decoded -->

then a 0 ≥ S b 0 where S = |S| .

Regarding the joint spectral radius, for any lengthm product P π m · · · P π 1 with P π t ∈ P π . By a telescoping expansion and nonexpansiveness of TV under right-multiplication by a Markov kernel,

Then, for all i = j ,

<!-- formula-not-decoded -->

/negationslash by the triangle inequality in TV. Hence

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By setting δ &lt; a 0 2 m , we have sup P i ∈P τ (( P π m · · · P π 1 )) &lt; 1 , and by Lemma A.3,

Similarly, the same perturbation bound yields

<!-- formula-not-decoded -->

so by setting δ &lt; b 0 m , we have that ( P π ) m is strictly positive for every P ∈ P ; hence all induced kernels are irreducible and aperiodic. Since a 0 ≥ Sb 0 , we have a 0 2 m ≥ b 0 m for S ≥ 2 . Therefore the condition that δ &lt; b 0 m satisfies both requirements.

Wasserstein Distance Uncertainty Regarding Wasserstein uncertainty with p ≥ 1 , let ( S , d ) be the finite metric space. The uncertainty set can be characterized as

<!-- formula-not-decoded -->

For a fixed policy π , the induced state-transition matrix P π is expressed as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For each state s , by joint convexity of W p p ( · , · ; d ) , hence W p ( P π ( s, · ) , ˜ P π ( s, · ); d ) ≤ δ for all s , i.e.

We now draw connection between (70) and the TV version in (65). Since the state space is finite, denote δ min := min x = y d ( x, y ) &gt; 0 . Then, for any distributions u, v , we have

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore we can reduce (70) into a TV distance uncertainty set characterized as follows:

<!-- formula-not-decoded -->

/negationslash

Lemma A.6. Under the Wasserstein distance uncertainty set, if the centroid ˜ P π is irreducible and aperiodic, then there exists m ∈ N such that ( ˜ P π ) m is strictly positive. Define b 0 = min i,s (( ˜ P π ) m ) is &gt; 0 and δ min := min x = y d ( x, y ) &gt; 0 , then if the radius satisfies δ &lt; δ min b 0 m , the joint spectral radius of Q π P defined in (46) is strictly less than 1 . Furthermore, P π is irreducible and aperiodic for all P ∈ P .

Proof.

This is a direct corollary of Lemma A.5 under the condition of (72).

Remarks. (i) If d is normalized so δ min = 1 , the thresholds simplify accordingly. (ii) One can also argue via W p p ≥ δ p min TV , which gives the alternative (more conservative when ε is small) choice r = δ p /δ p min ; the linear reduction r = δ/δ min above is sharper and suffices for the bounds.

## A.2.3 Extremal Norm Construction

Under the radius conditions of Lemma A.4-A.6, we have that :

<!-- formula-not-decoded -->

We follow similar process for constructing our desired semi-norm ‖ · ‖ P as in Appendix A.1 by first constructing a norm such that all Q π P are strictly less then one under that norm. We choose α ∈ ( r ∗ , 1) and we follow the approach in [42] by constructing an extremal norm ‖· ‖ ext as follows:

<!-- formula-not-decoded -->

Note that we follow the convention that ‖ Q k Q k -1 . . . Q 1 x ‖ 2 = ‖ x ‖ 2 when k = 0 .

Lemma A.7. Under Assumption 3.1 and the radius conditions of Lemma A.4-A.6, the operator ‖ · ‖ ext is a valid norm with ‖ Q π P ‖ ext &lt; 1 for all P ∈ P

Proof. We first prove that ‖· ‖ ext is bounded. Following Lemma F.1 and choosing λ ∈ ( r ∗ , α ) , then there exist a positive constant C &lt; ∞ such that

<!-- formula-not-decoded -->

Hence for each k and for all x ∈ R S ,

<!-- formula-not-decoded -->

Thus the double supremum in (74) is over a bounded and vanishing sequence, so ‖ · ‖ ext bounded.

To check that ‖ · ‖ ext is a valid norm, note that if x = 0 , ‖ x ‖ ext is directly 0 . On the other hand, if ‖ x ‖ ext = 0 , we have

<!-- formula-not-decoded -->

Regarding homogeneity, observe that for any c ∈ R and x ∈ R S ,

<!-- formula-not-decoded -->

Regarding triangle inequality, using ‖ Q k . . . Q 1 ( x + y ) ‖ 2 ≤ ‖ Q k . . . Q 1 x ‖ 2 + ‖ Q k . . . Q 1 y ‖ 2 for any x, y ∈ R S , we obtain,

<!-- formula-not-decoded -->

For any P ∈ P , we have

<!-- formula-not-decoded -->

Since P is arbitrary, (80) implies that for any P ∈ P ,

/negationslash

<!-- formula-not-decoded -->

## A.2.4 Semi-Norm Contraction for Robust Bellman Operator

We now follow the same method as (41) to construct the semi-norm ‖·‖ P . Define the operator ‖·‖ P as where 0 &lt; /epsilon1 &lt; 1 -α .

<!-- formula-not-decoded -->

Lemma A.8. The operator ‖·‖ P is a valid semi-norm with kernel being exactly { c e : c ∈ R } under Assumption 3.1 and the radius conditions of Lemma A.4-A.6. Furthermore, for all x ∈ R S , we have ∥ ∥ P π x ∥ ∥ P ≤ ( α + /epsilon1 ) ∥ ∥ x ∥ ∥ P for all P ∈ P .

<!-- formula-not-decoded -->

Proof. Regarding positive homogeneity and nonnegativity, for any scalar λ and x ∈ R S , and ‖ x ‖ ext ≥ 0 . Regarding triangle inequality, for any x, y ∈ R S , note that for any P ∈ P ,

<!-- formula-not-decoded -->

Taking supremum over P on both sides yields

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

Regarding the kernel, if x = k e for some k ∈ R , then similar to (42), we have

On the other hand, if x / ∈ { c e : c ∈ R } , we know that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, the kernel of ‖ · ‖ P is exactly { c e : c ∈ R } . We now show that, for any x ∈ R S and P ∈ P ,

<!-- formula-not-decoded -->

Since α ∈ (0 , 1) and /epsilon1 ∈ (0 , 1 -α ) . Thus, let γ = α + /epsilon1 then γ ∈ (0 , 1) . Substituting the above result back to (45), we obtain

<!-- formula-not-decoded -->

## B Biased Stochastic Approximation Convergence Rate

In Section 4, we established that the robust Bellman operator is a contraction under the semi-norm ‖·‖ P , ensuring that policy evaluation can be analyzed within a well-posed stochastic approximation framework. However, conventional stochastic approximation methods typically assume unbiased noise, where variance diminishes over time without introducing systematic drift. In contrast, the noise in robust policy evaluation under TV and Wasserstein distance uncertainty sets exhibits a small but persistent bias, arising from the estimators of the support functions ˆ σ P a s ( V ) (discussed in Section 5). This bias, if not properly addressed, can lead to uncontrolled error accumulation, affecting the reliability of policy evaluation. To address this challenge, this section introduces a novel analysis of biased stochastic approximation, leveraging properties of dual norms to ensure that the bias remains controlled and does not significantly impact the convergence rate. Our results extend prior work on unbiased settings and provide the first explicit finite-time guarantees, which are further used to establish the sample complexity of policy evaluation in robust average-reward RL. Specifically, we analyze the iteration complexity for solving the fixed equivalence class equation H ( x ∗ ) -x ∗ ∈ E where E := { c e : c ∈ R } with e being the all-ones vector. The stochastic approximation iteration being used is as follows:

<!-- formula-not-decoded -->

with η t &gt; 0 being the step-size sequence. We assume that there exist γ ∈ (0 , 1) such that

<!-- formula-not-decoded -->

We also assume that the noise terms ω t are i.i.d. and have bounded bias and variance

E [ ‖ w t ‖ 2 P |F t ] ≤ A + B ‖ x t -x ∗ ‖ 2 P and ∥ ∥ E [ w t |F t ] ∥ ∥ P ≤ ε bias (91) Theorem B.1. If x t is generated by (89) with all assumptions in (90) and (91) satisfied, then if the stepsize η t := O ( 1 t ) , where x sup := sup x ‖ x ‖ P is the upper bound of the ‖ · ‖ P semi-norm for all x t .

<!-- formula-not-decoded -->

Theorem B.1 adapts the analysis of [45] and extends it to a biased i.i.d. noise setting. To manage the bias terms, we leverage properties of dual norms to bound the inner product between the error term and the gradient, ensuring that the bias influence remains logarithmic in T rather than growing

unbounded, while also carefully structuring the stepsize decay to mitigate long-term accumulation. This results in an extra ε bias term with logarithmic dependence of the total iteration T .

We perform analysis of the biased-noise extension to the semi-norm stochastic approximation (SA) problem by constructing a smooth convex semi-Lyapunov function for forming the negative drift [45, 9] and using properties in dual norms for managing the bias.

## B.1 Proof of Theorem B.1

## B.1.1 Setup and Notation.

In this section, we override the notation of the semi-norm ‖ · ‖ P by re-writing it as the norm ‖ · ‖ N (defined in (96)) to the equivalence class of constant vectors. For any norm ‖ · ‖ c and equivalence class E , define the indicator function δ E as

<!-- formula-not-decoded -->

then by [45], the semi-norm induced by norm ‖·‖ c and equivalence class E is the infimal convolution of ‖ · ‖ c and the indicator function δ E can be defined as follows

<!-- formula-not-decoded -->

where ∗ inf denotes the infimal convolution operator. Throughout the remaining section, we let E := { c e : c ∈ R } with e being the all-ones vector. Since ‖ · ‖ P constructed in (82) is a semi-norm with kernel being E , we can construct a norm ‖ · ‖ N such that

<!-- formula-not-decoded -->

We construct ‖ · ‖ N as follows:

where Q π P and /epsilon1 are defined in (82).

<!-- formula-not-decoded -->

Lemma B.2. The operator ‖ · ‖ N defined in (96) is a norm satisfying (95) .

/negationslash

Proof. We first verify that ‖ · ‖ N is a norm. Regarding positivity, since all terms in (96) are nonnegative, ‖ x ‖ N ≥ 0 for all x ∈ R S and ‖ 0 ‖ N = 0 . If x = 0 , since ‖ · ‖ ext is a valid norm and /epsilon1 &gt; 0 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Regarding homogeneity, For any λ ∈ R , we have

Regarding triangle inequality, for any x, y ∈ R S , we have

<!-- formula-not-decoded -->

We now show that since Q π P e = 0 for all P ∈ P , by the definition of infimal convolution, we have that for all x ∈ R S ,

<!-- formula-not-decoded -->

We thus restate our problem of analyzing the iteration complexity for solving the fixed equivalence class equation H ( x ∗ ) -x ∗ ∈ E , with the operator H : R n → R n satisfying the contraction property as follows:

<!-- formula-not-decoded -->

The stochastic approximation iteration being used is as follows

We assume:

- E [ ‖ w t ‖ 2 N ,E |F t ] ≤ A + B ‖ x t -x ∗ ‖ 2 N ,E (In the robust average-reward TD case, B = 0 ).

<!-- formula-not-decoded -->

- η t &gt; 0 is a chosen stepsize sequence (decreasing or constant).
- ∥ ∥ E [ w t |F t ] ∥ ∥ N ,E ≤ ε bias .

Note that beside the bias in the noise, the above formulation and assumptions are identical to the unbiased setups in Section B of [45]. Thus, we emphasize mostly on managing the bias.

## B.1.2 Semi-Lyapunov M E ( · ) and Smoothness.

By [45, Proposition 1-2], using the Moreau envelope function M ( x ) in Definition 2.2 of [8], we define so that there exist c l , c u &gt; 0 with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and M E is L -smooth w.r.t. another semi-norm ‖ · ‖ s,E . Concretely, L -smoothness means:

<!-- formula-not-decoded -->

Moreover, the gradient of M E satisfies 〈∇ M E ( x ) , c e 〉 = 0 for all x , and the dual norm denoted as ‖ · ‖ ∗ ,s,E is also L-smooth:

<!-- formula-not-decoded -->

Note that since ‖ · ‖ s,E and ‖ · ‖ N ,E are semi-norms on a finite-dimensional space with the same kernel, there exist ρ 1 , ρ 2 &gt; 0 such that

<!-- formula-not-decoded -->

Likewise, their dual norms (denoted ‖ · ‖ ∗ ,s,E and ‖ · ‖ ∗ , N ,E ) satisfy the following:

<!-- formula-not-decoded -->

## B.1.3 Formal Statement of Theorem B.1

By L -smoothness w.r.t. ‖ · ‖ s,E in (100), for each t ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where x t +1 -x t = η t [ ̂ H ( x t ) -x t ] = η t [ H ( x t ) + w t -x t ] . Taking expectation of the second term of the RHS of (104) conditioned on the filtration F t we obtain,

To analyze the additional bias term 〈∇ M E ( x t -x ∗ ) , E [ ω t |F t ] 〉 , we use the fact that for any (semi)norm ‖ · ‖ with dual (semi-)norm ‖ · ‖ ∗ (defined by ‖ u ‖ ∗ = sup {〈 u, v 〉 : ‖ v ‖ ≤ 1 } ), we have the general inequality

In the biased noise setting, u = ∇ M E ( x t -x ∗ ) and v = E [ w t |F t ] , with ‖ · ‖ = ‖ · ‖ N ,E . So

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since ‖ E [ w t |F t ] ‖ N ,E ≤ ε bias, it remains to bound ‖∇ M E ( x t -x ∗ ) ‖ ∗ , N ,E . By setting y = 0 in (101), we get

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

By (103), we know that there exists 1 ρ 2 ≤ α ≤ 1 ρ 1 such that

<!-- formula-not-decoded -->

Thus, combining (109) and (110) would give:

<!-- formula-not-decoded -->

By (102), we know that ‖ x ‖ s,E ≤ ‖ x ‖ N ,E , thus we have:

<!-- formula-not-decoded -->

Hence, combining the above with (107), there exist some such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining (114) with (105) we obtain

<!-- formula-not-decoded -->

To bound the first term in the RHS of (115), note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( a ) follows from the convexity of M ¯ E , ( b ) follows from x ∗ belonging to a fixed equivalence class with respect to H and ( c ) follows from the contraction property of H . Combining (116). (115) and Lemma F.2 with (104), we arrive as follows:

Where α 2 := (1 -γ √ c u /c l ) , α 3 := (8 + 2 B ) c u ρ 2 L and α 4 := Aρ 2 L . We now present the formal version of Theorem B.1 as follows:

<!-- formula-not-decoded -->

Theorem B.3 (Formal version of Theorem B.1) . let α 2 , α 3 and α 4 be defined in (117) , if x t is generated by (98) with all assumptions in B.1.1 satisfied, then if the stepsize η t := 1 α 2 ( t + K ) while K := max { α 3 /α 2 , 3 } , where C 1 = G (1+2 x sup ) , C 2 = 1 K +log ( T -1+ K K ) , G is defined in (113) and x sup := sup ‖ x ‖ N ,E is the upper bound of the ‖ · ‖ P semi-norm for all x t .

Proof. This choice η t satisfies α 3 η 2 t ≤ α 2 η t . Thus, by (117) we have we define Γ t := Π t -1 i = o (1 -α 2 η t ) and further obtain the T -step recursion relationship as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

also, R 2 can be bounded by a logarithmic dependence of T

<!-- formula-not-decoded -->

Combining (121) and (122) with (120) would obtain the following:

<!-- formula-not-decoded -->

Combining (123) with (99) yields (118).

## C Uncertainty Set Support Function Estimators

## C.1 Proof of Theorem 5.1

We have

<!-- formula-not-decoded -->

## C.2 Proof of Theorem 5.2

denote ˆ σ ∗ P a s ( V ) as the untruncated MLMC estimator obtained by running Algorithm 1 when setting N max to infinity. From [39], under both TV uncertainty sets and Wasserstein uncertainty sets, we have ˆ σ ∗ P a s ( V ) as an unbiased estimator of σ P a s ( V ) . Thus,

For each ∆ n ( V ) , the expectation of absolute value can be bounded as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

∣ ∣ ∣ ∣ By the binomial concentration and the Lipschitz property of the support function as in Lemma 5.3, we know for TV distance uncertainty, we have

<!-- formula-not-decoded -->

and for Wasserstein disance uncertainty, we have

<!-- formula-not-decoded -->

Thus, for TV distance uncertainty, we have

<!-- formula-not-decoded -->

and for Wasserstein distance uncertainty, we have

<!-- formula-not-decoded -->

## C.3 Proof of Lemma 5.3

For TV uncertainty sets, for a fixed V , for any p ∈ ∆( S ) , define f p ( µ ) := p ( V -µ ) -δ ‖ V -µ ‖ sp and µ ∗ p := arg max µ ≥ 0 f p ( µ ) . Thus, we have

<!-- formula-not-decoded -->

since, µ ∗ p and µ ∗ q are maximizers of f p and f q respectively, we further have

<!-- formula-not-decoded -->

Combing (131) and (132) we thus have:

<!-- formula-not-decoded -->

Note that σ P TV ( V ) can also be expressed as σ P TV ( V ) = p x ∗ -δ ‖ x ∗ ‖ sp where x ∗ := arg max x ≤ V ( p x -δ ‖ x ‖ sp ) . Let M := max s x ∗ ( s ) and m := min s x ∗ ( s ) , then ‖ x ‖ sp = M -m . Denote e as the all-ones vector, then x = min s V ( s ) · e is a feasible solution. Thus,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since p is a probability vector, p x ∗ ≤ M , using the fact that δ &gt; 0 , we then obtain

<!-- formula-not-decoded -->

Since x ∗ is a feasible solution, we have

<!-- formula-not-decoded -->

Combining (135) and (136) we obtain

<!-- formula-not-decoded -->

Where the last inequality is from M ≥ min s V ( s ) , which is a direct result of (135) and the term δ ( M -m ) being positive. We finally arrive with

Thus, ‖ x ∗ ‖ sp ≤ (1 + 1 δ ) ‖ V ‖ sp , which leads to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining (139) with (133) we obtain the first part of (23).

For Wasserstein uncertainty sets, note that for any p ∈ ∆( S ) and value function V ,

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

where the first inequality is because λd ( S, y ) l ≥ 0 for any d and l . We can then bound φ by the span of V as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We then further have that for any p, q ∈ ∆( S ) and λ ≥ 0 , using (143) and the fact that | f ( λ ) -g ( λ ) | ≤ /epsilon1 ⇒ | sup λ f ( λ ) -sup λ g ( λ ) | ≤ /epsilon1 , we obtain the second part of (23).

## C.4 Proof of Theorem 5.4

For all p ∈ ∆( S ) , we have σ p ( V ) ≤ ‖ V ‖ sp , leading to

To bound the second moment, note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Under TV distance uncertainty set, by (127), we further have

<!-- formula-not-decoded -->

Under Wasserstein distance uncertainty set, by (128), we further have

<!-- formula-not-decoded -->

## D Convergence for Robust TD

## D.1 Formal Statement of Theorem 6.1

The first half of Algorithm 2 (line 1 - line 7) can be treated as a special instance of the SA updates in (98) with the bias and variance of the i.i.d. noise term specified in Section 5. To facilitate deriving

the bounds of the noise terms, we first analyze the bounds in terms of the l ∞ norm, and then translate the bounds in terms of the ‖ · ‖ P semi-norm to obtain the final results.

<!-- formula-not-decoded -->

We start with analyzing the bias and variance of ˆ T g 0 ( V t ) for each t . Recall the definition of ˆ T g 0 ( V t ) is as follows:

Thus, we have for all s ∈ S ,

<!-- formula-not-decoded -->

Which further implies the bias of ˆ T g 0 ( V t ) is bounded by the bias of ˆ σ P a s ( V t ) as follows:

Regarding the variance, note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To create an upper bound of ‖ V ‖ sp for all possible V , define the mixing time of any p ∈ P to be where p π is the finite state Markov chain induced by π , µ 0 is any initial probability distribution on S and ν is its invariant distribution. By Assumption 3.1, and Lemma F.4,and for any value function V , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we define t mix := sup p ∈P t p mix , then t mix is also finite due to the compactness of P . We now derive the bounds of biases and variances for the three types of uncertainty sets. Regarding contamination uncertainty sets, according to Lemma F.3, ˆ σ P a s ( V ) is unbiased and has variance bounded by ‖ V ‖ 2 . Thu, define t mix according to Lemma F.4 and combining the above result with Lemma F.4, we obtain that ˆ T g 0 ( V t ) is also unbiased and the variance satisfies

Regarding TV distance uncertainty sets, using the property of the bias and variance of ˆ σ P a s ( V ) in Theorem 5.2 and Theorem 5.4 while combining them with Lemma F.4, we have

<!-- formula-not-decoded -->

Similarly, for Wasserstein distance uncertainty sets, we have and

<!-- formula-not-decoded -->

In order to translate the above bounds from the l ∞ norm into the ‖ · ‖ P norm, recall that in line 7 of Algorithm 2, we chose an anchor state s 0 set V t ( s 0 ) = 0 for all t to avoid ambiguity. We thus can draw the following relationship:

<!-- formula-not-decoded -->

Lemma D.1. Let x ∈ R S satisfy x i = 0 for some fixed index i . Then

<!-- formula-not-decoded -->

Moreover, since all semi-norms with the same kernel spaces are equivalent, there are constants c P , C P &gt; 0 so that then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Since x i = 0 , for every j we have -‖ x ‖ ∞ ≤ x j ≤ ‖ x ‖ ∞ . Hence and so

‖ x ‖ ∞ = max { max j x j -0 , 0 -min j x j } ≤ ‖ x ‖ sp = max j x j -min j x j ≤ ‖ x ‖ ∞ -( -‖ x ‖ ∞ ) = 2 ‖ x ‖ ∞ .

Since ‖·‖ sp and ‖·‖ P both have the same kernel of { c e : c ∈ R } , by the equivalence of semi-norms, it follows that there exists c P and C P such that as claimed.

<!-- formula-not-decoded -->

With the relationship established above, line 1 - line 7 of Algorithm 2 can be formally treated as a special instance of the SA updates in (98) with B = 0 . We now provide the bias and variance of the i.i.d. noise for the different uncertainty sets discussed using Lemma D.1 and the estimation bounds in (153)-(157). For contamination uncertainty sets, we have

<!-- formula-not-decoded -->

for TV distance uncertainty sets, we have and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem D.2 (Formal version of Theorem 6.1) . Let α 2 := (1 -γ √ c u /c l ) , α 3 := 8 c u ρ 2 L and α 4 := ρ 2 L , if V t is generated by Algorithm 2. Define V ∗ to be the anchored robust value function V ∗ = V π P V + c e for some c such that V ∗ ( s 0 ) = 0 , then under Assumption 3.1 and the radius conditions of Lemma A.4-A.6, if the stepsize η t := 1 α 2 ( t + K ) while K := max { α 3 /α 2 , 3 } , then for contamination uncertainty sets,

<!-- formula-not-decoded -->

for TV distance uncertainty sets,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for Wasserstein distance uncertainty sets,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the ε and A terms are defined in (159) -(163) , C 2 = 1 K +log ( T -1+ K K ) , C 3 = G (1+8 C P t mix ) , γ is defined in (14) , c u , c l are defined in (99) , ρ 2 is defined in (102) , G is defined in (113) , and C P , c P are defined in Lemma D.1.

<!-- formula-not-decoded -->

Proof. By Lemma D.1 and (152), we have that for any value function V its ‖ · ‖ P norm is bounded as follows:

<!-- formula-not-decoded -->

Substituting the terms of (159)-(163), (169), and Theorem 4.2 to Theorem B.3, we would have for contamination uncertainty sets,

<!-- formula-not-decoded -->

for TV distance uncertainty sets,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for Wasserstein distance uncertainty sets,

<!-- formula-not-decoded -->

where the ε and A terms are defined in (159)-(163), C 2 = 1 K +log ( T -1+ K K ) , C 3 = G (1+8 C P t mix ) , γ is defined in (14), c u , c l are defined in (99), ρ 2 is defined in (102), G is defined in (113), and C P is defined in Lemma D.1. We now translate the result back to the standard l ∞ norm by applying Lemma D.1 again to the above, we obtain the desired results.

<!-- formula-not-decoded -->

## D.2 Proof of Theorem 6.1

Weuse the result from Theorem D.2, to set E [ ‖ V T -V ∗ ‖ 2 ∞ ] ≤ /epsilon1 2 . For contamination uncertainty set we set T = O ( t 2 mix /epsilon1 2 (1 -γ ) 2 ) , resulting in O ( SAt 2 mix /epsilon1 2 (1 -γ ) 2 ) sample complexity. For TV and Wasserstein uncertainty set, we set N max = O ( log √ St mix /epsilon1 (1 -γ ) ) and T = O ( t 2 mix /epsilon1 2 (1 -γ ) 2 log √ St mix /epsilon1 (1 -γ ) ) , combining with Theorem 5.1, this would result in O ( SAt 2 mix /epsilon1 2 (1 -γ ) 2 log 2 √ St mix /epsilon1 (1 -γ ) ) sample complexity.

To show order-optimality, we provide the standard mean estimation as a hard example. Consider the TD learning of the MDP with only two states S = { s 1 , s 2 } , and Pr ( s → s 1 ) = p , Pr ( s → s 2 ) = 1 -p for each s ∈ S with p ∈ (0 , 1) . Thus, this MDP is indifferent from the actions chosen and we further define r ( s 1 ) = 1 , r ( s 2 ) = 0 . Thus, estimating the relative value functions is equivalent of estimating p . By the Cramér-Rao or direct variance argument for Bernoulli( p ) estimation, we have that to achieve | ˆ p N -p | 2 ≤ /epsilon1 2 requires N ≥ 1 / 2 /epsilon1 2 = Ω( /epsilon1 -2 ) .

## D.3 Formal Statement of Theorem 6.2

To analyze the second part (line 8 - line 14) of Algorithm 2 and provide the provide the complexity for g t , we first define the noiseless function ¯ δ ( V ) as

Thus, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ν t is the noise term with bias equal to the bias ˆ σ P a s ( V T )

By the Bellman equation in Theorem 3.2, we have g π P = ¯ δ ( V ∗ ) , which implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Where the last inequality is by Lemma D.1. Thus, the following recursion can be formed

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, taking expectation conditioned on the filtration F t yields

By letting ζ t := Π t -1 i =0 (1 -β t ) , we obtain the T -step recursion as follows:

<!-- formula-not-decoded -->

By setting β t := 1 t +1 , we have ζ T = 1 T +1 ≤ 1 T and ∑ T -1 t =0 β t ≤ 2 log T , (181) implies

<!-- formula-not-decoded -->

Theorem D.3 (Formal version of Theorem 6.2) . Following all notations and assumptions in Theorem D.2, then for contamination uncertainty sets,

<!-- formula-not-decoded -->

For TV distance uncertainty sets,

<!-- formula-not-decoded -->

(184)

For Wasserstein distance uncertainty sets, where all the above variables are defined the same as in Theorem D.2.

Proof. By Theorem D.2, taking square root on both side and utilizing the concavity of square root function, we have for contamination uncertainty sets,

<!-- formula-not-decoded -->

for TV distance uncertainty sets,

<!-- formula-not-decoded -->

for Wasserstein distance uncertainty sets,

<!-- formula-not-decoded -->

Regarding the bound for the absolute bias of ˆ σ P a s , from Lemma F.3, we have for contamination uncertainty,

In addition, combining (129)-(130) with Lemma F.4, we have for for TV distance uncertainty,

<!-- formula-not-decoded -->

and for Wasserstein distance uncertainty, we have

<!-- formula-not-decoded -->

Combining (186)-(191) with (182) gives the desired result.

<!-- formula-not-decoded -->

## D.4 Proof of Theorem 6.2

We use the result from Theorem D.3, to set E [ | g T -g π P | ] ≤ /epsilon1 . For contamination uncertainty sets we set T = O ( t 2 mix /epsilon1 2 (1 -γ ) 2 log t mix /epsilon1 (1 -γ ) ) , resulting in O ( SAt 2 mix /epsilon1 2 (1 -γ ) 2 log t mix /epsilon1 (1 -γ ) ) sample complexity. For TV and Wasserstein uncertainty set, we set N max = O ( log √ St mix /epsilon1 (1 -γ ) ) and T = O ( t 2 mix /epsilon1 2 (1 -γ ) 2 log 3 √ St mix /epsilon1 (1 -γ ) ) , combining with Theorem 5.1, this would result in O ( SAt 2 mix /epsilon1 2 (1 -γ ) 2 log 4 √ St mix /epsilon1 (1 -γ ) ) sample complexity.

<!-- formula-not-decoded -->

(185)

## E Numerical Validations for Semi-Norm Contractions

In this section, we provide numerical examples that directly verify the one-step strict contraction across the settings studied. These results empirically support the key structural claims used by our analysis.

## E.1 Evaluations of Lemma 4.1

Lemma A.2 is the technical backbone for Lemma 4.1, as Lemma A.2 constructs the fixed-kernel semi-norm and provides the one-step contraction for a given P . Therefore, we perform numerical evaluations on Lemma A.2 to demonstrate the one-step contraction property for Lemma 4.1.

For a kernel P with stationary distribution d , we follow the steps in Appendix A.1 and construct ‖·‖ P in (41) with where Q = P -e d /latticetop . Then the one-step contraction factor is β = α + /epsilon1 &lt; 1 .

<!-- formula-not-decoded -->

To generate ergodic matrices with dimension n , let I n be the identity matrix and S n be the cyclic shift matrix defined by S n e i = e i +1 mod n . We provide the following four examples:

- P 1 = 0 . 5 I 5 +0 . 5 S 5
- P 2 = 0 . 6 I 6 +0 . 4 S 6
- P 3 = 0 . 55 I 7 +0 . 45 S 7
- P 4 = 0 . 6 I 8 +0 . 3 S 8 +0 . 1 S 2 8

We generate 1000 random unit vectors x and compute each ratio ‖ P i x ‖ P ‖ x ‖ P . The empirical results are summarized below.

Table 1: Empirical one-step contraction ratios for ergodic kernels using the fixed-kernel semi-norm ‖·‖ P .

| matrix   |   n | max span ratio   | ρ ( Q )   | α      |   /epsilon1 |      β |   ratio min |   ratio median |   ratio p90 |   ratio max |
|----------|-----|------------------|-----------|--------|-------------|--------|-------------|----------------|-------------|-------------|
| P1       |   5 | 1 0.8090         | 0.9045    |        |      0.0239 | 0.9284 |      0.3824 |         0.795  |      0.8077 |      0.809  |
| P2       |   6 | 1                | 0.8718    | 0.9359 |      0.016  | 0.9519 |      0.5197 |         0.851  |      0.8687 |      0.8718 |
| P3       |   7 | 1 0.9020         |           | 0.9510 |      0.0122 | 0.9633 |      0.5861 |         0.8799 |      0.8976 |      0.9014 |
| P4       |   8 | 1                | 0.8700    | 0.9350 |      0.0162 | 0.9513 |      0.4855 |         0.8226 |      0.8604 |      0.8685 |

## E.2 Evaluations of Theorem 4.2

Lemma A.8 is the key step for Theorem 4.2, as Lemma A.8 proves a uniform one-step contraction across all P in the uncertainty set. We therefore perform numerical evaluations on Lemma A.8 to demonstrate the one-step contraction property for Theorem 4.2 under contamination, total variation (TV), and Wasserstein-1 uncertainty. We select the same P 1 , P 2 , P 3 , and P 4 in Appendix E.1 as four examples of the nominal model.

To numerically approximate ‖·‖ P defined in (82), we approximate ‖·‖ P by (i) discretizing the uncertainty set and (ii) using a finite product to approximate the extremal norm. First, we sample a family { P ( i ) } i m =1 ⊂ P of size m and form

<!-- formula-not-decoded -->

We set α = min { 0 . 99 , (1+ˆ r ) / 2 } and choose /epsilon1 ∈ (0 , 1 -α ) . To approximate the extremal norm, we build a library of scaled products of the Q i 's up to maximum length K : for each k = 0 , 1 , . . . , K we draw products M k,j = Q i k · · · Q i 1 ; the number of such draws at each k is the 'products per length' (denoted samples\_per\_k in the tables). This defines the surrogate

<!-- formula-not-decoded -->

We then set

<!-- formula-not-decoded -->

We generate 50 random unit vectors x for each sampled uncertainty matrix in { P ( i ) } i m =1 ⊂ P , and compute the ratios ‖ P ( i ) x ‖ P ‖ x ‖ P . The empirical results of the uncertainty sets studied in our settings are summarized below.

| nominal ˜ P   | n      |   δ m |   K |   samples_per_k max | ratio ˆ r   |      α |   /epsilon1 |      γ |   ratio min |   ratio median |   ratio p90 |   ratio max |
|---------------|--------|-------|-----|---------------------|-------------|--------|-------------|--------|-------------|----------------|-------------|-------------|
| P1            | 5 0.15 |    30 |   3 |                  25 | 1 0.8138    | 0.9069 |      0.0233 | 0.9302 |      0.221  |         0.6396 |      0.8057 |      0.8134 |
| P2            | 6 0.15 |    30 |   3 |                  25 | 1 0.8807    | 0.9403 |      0.0149 | 0.9553 |      0.2957 |         0.6581 |      0.863  |      0.8785 |
| P3            | 7 0.15 |    30 |   3 |                  25 | 1 0.9067    | 0.9534 |      0.0117 | 0.965  |      0.3217 |         0.6683 |      0.87   |      0.8901 |
| P4            | 8 0.15 |    30 |   3 |                  25 | 1 0.8812    | 0.9406 |      0.0148 | 0.9555 |      0.3739 |         0.5964 |      0.8207 |      0.8624 |

Table 2: Empirical one-step contraction ratios under contamination uncertainty.

Table 3: Empirical one-step contraction ratios under total variation (TV) uncertainty.

| nominal ˜ P   | n      |   δ m |   K |   samples_per_k max |   span ratio |    ˆ r |      α |   /epsilon1 |      γ |   ratio min |   ratio median |   ratio p90 |   ratio max |
|---------------|--------|-------|-----|---------------------|--------------|--------|--------|-------------|--------|-------------|----------------|-------------|-------------|
| P1            | 5 0.15 |    30 |   3 |                  25 |            1 | 0.828  | 0.914  |      0.0215 | 0.9355 |      0.3239 |         0.7609 |      0.8239 |      0.828  |
| P2            | 6 0.15 |    30 |   3 |                  25 |            1 | 0.9013 | 0.9507 |      0.0123 | 0.963  |      0.4021 |         0.7904 |      0.8862 |      0.9006 |
| P3            | 7 0.15 |    30 |   3 |                  25 |            1 | 0.9175 | 0.9588 |      0.0103 | 0.9691 |      0.4457 |         0.7918 |      0.8962 |      0.9162 |
| P4            | 8 0.15 |    30 |   3 |                  25 |            1 | 0.8805 | 0.9403 |      0.0149 | 0.9552 |      0.4497 |         0.7503 |      0.8449 |      0.8739 |

Table 4: Empirical one-step contraction ratios under Wasserstein-1 uncertainty.

| nominal ˜ P   | n      |   δ m |   K |   samples_per_k max span | ratio ˆ r   |      α |   /epsilon1 |      γ |   ratio min |   ratio median |   ratio p90 |   ratio max |
|---------------|--------|-------|-----|--------------------------|-------------|--------|-------------|--------|-------------|----------------|-------------|-------------|
| P1            | 5 0.15 |    30 |   3 |                       25 | 1 0.8184    | 0.9092 |      0.0227 | 0.9319 |      0.3698 |         0.7569 |      0.8141 |      0.8188 |
| P2            | 6 0.15 |    30 |   3 |                       25 | 1 0.8900    | 0.945  |      0.0138 | 0.9587 |      0.4257 |         0.7889 |      0.8774 |      0.8894 |
| P3            | 7 0.15 |    30 |   3 |                       25 | 1 0.9110    | 0.9555 |      0.0111 | 0.9666 |      0.4262 |         0.7818 |      0.8827 |      0.908  |
| P4            | 8 0.15 |    30 |   3 |                       25 | 1 0.8758    | 0.9379 |      0.0155 | 0.9534 |      0.4413 |         0.7224 |      0.8518 |      0.8689 |

## E.3 Interpretations

Note that for the above tables, max span ratio denotes the largest one-step span contraction coefficient over the sampled families. This value equals to 1 for all the settings, meaning no strict one-step contraction in the span. In contrast, the quantities ratio min , ratio median , ratio p90 , and ratio max summarize the empirical one-step ratios || Px || P || x || P (in the robust case) and || Px || P || x || P (in the non-robust case) computed under the constructed semi-norms over all sampled kernels P and random unit directions x . They report, respectively, the minimum, median, 90th percentile, and maximum observed value across those tests. In every table we have ratio max &lt; 1 , so we observe empirical one-step contraction under our semi-norms even when span does not contract. Moreover, ratio max ≤ γ (robust case) or ratio max ≤ β (non-robust case), which is consistent with the corresponding theoretical contraction factor guaranteed by our constructions.

## F Some Auxiliary Lemmas for the Proofs

Lemma F.1 (Theorem IV in [3]) . Let Q be a bounded set of square matrix such that ρ ( Q ) &lt; ∞ for all Q ∈ Q where ρ ( · ) denotes the spectral radius. Then the joint spectral radius of Q can be defined as

<!-- formula-not-decoded -->

where ‖ · ‖ is an arbitrary norm.

<!-- formula-not-decoded -->

Lemma F.2 (Lemma 6 in [45]) . Under the setup and notation in Appendix B.1.1, if assuming the noise has bounded variance of E [ ‖ w t ‖ 2 N ,E |F t ] ≤ A + B ‖ x t -x ∗ ‖ 2 N ,E , we have

Lemma F.3 (Theorem D.1 in [39]) . The estimator ˆ σ P a s ( V ) obtained by (18) for contamination uncertainty sets is unbiased and has bounded variance as follows:

<!-- formula-not-decoded -->

Lemma F.4 (Ergodic case of Lemma 9 in [34]) . For any average-reward MDP with stationary policy π and the mixing time defined as

<!-- formula-not-decoded -->

where P π is the finite state Markov chain induced by π , µ 0 is any initial probability distribution on S and ν is its invariant distribution. If P π is irreducible and aperiodic, then τ mix &lt; + ∞ and for the value function V , we have

<!-- formula-not-decoded -->