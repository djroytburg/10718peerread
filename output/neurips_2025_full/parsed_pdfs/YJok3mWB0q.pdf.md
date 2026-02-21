## Finite Sample Analyses for Continuous-time Linear Systems: System Identification and Online Control

## Hongyi Zhou ∗

IIIS, Tsinghua University

Shanghai Qizhi Institute zhouhong24@mails.tsinghua.edu.cn

## Jingwei Li ∗

IIIS, Tsinghua University Shanghai Qizhi Institute ljw22@mails.tsinghua.edu.cn

## Jingzhao Zhang †

IIIS, Tsinghua University

Shanghai Qi zhi Institute jingzhaoz@mail.tsinghua.edu.cn

## Abstract

Real world evolves in continuous time but computations are done from finite samples. Therefore, we study algorithms using finite observations in continuoustime linear dynamical systems. We first study the system identification problem, and propose a first non-asymptotic error analysis with finite observations. Our algorithm identifies system parameters without needing integrated observations over certain time intervals, making it more practical for real-world applications. Further we propose a lower bound result that shows our estimator is provably optimal up to constant factors. Moreover, we apply the above algorithm to online control regret analysis for continuous-time linear system. Our system identification method allows us to explore more efficiently, enabling the swift detection of ineffective policies. We achieve a regret of O ( √ T ) over a single T -time horizon in a controllable system, requiring only O ( T ) observations of the system.

## 1 Introduction

Finding optimal control policies requires accurately modelling the system [18]. However, realworld environments often involve unknown system parameters. In such cases, estimating unknown parameters from exploration becomes essential to identify the unseen dynamics. This process is recognized as system identification, a fundamental tool employed in various research fields, including time-series analysis [20], control theory [21], robotics [16], and reinforcement learning [28].

The identification of linear systems has long been studied because linear systems, as one of the most fundamental systems in both theoretical frameworks and practical applications, has wide applications ranging from natural physical processes to robotics. Most classical results provide only asymptotic convergence guarantees for parameter estimation [3, 24, 5].

On the other hand, with the rapid increase in data scale, there is a growing concern for statistical efficiency. Consequently, the non-asymptotic convergence of discrete-time linear system identification has emerged as another pivotal topic in this field. Investigations into this matter delve into understanding how estimation confidence is influenced by the sample complexity of trajectories [9], or the running time on a single trajectory [33, 29]. Furthermore, many of these studies operate

∗ Equal Contribution

† Corresponding Author

under the common assumption of stochastic noise, there has been a parallel exploration into the identification of discrete-time linear dynamical systems with diverse setups. This includes scenarios where perturbations are adversarial [15] or when only black-box access is available [7].

In contrast to studies in discrete time system, there have been relatively fewer non-asymptotic results addressing parameter identification for continuous-time systems . Two problems exist for continuous time analysis. First, nonasymptotic analysis in continuous system without noise can be degenerate, as a short time interval can contain infinite pieces of information. Second, if we consider the nondegenerate case when finite noisy observations are available, then the analyses require concentration results that become known only as in [33, 9, 29]. Recently [4] provides novel analyses for estimating system parameters, which relies on continuous data collection and interaction with the environment. Motivated by progress in these works, our first goal is to answer the question below:

Can we design a continuous-time stochastic system identification algorithm that provides nonasymptotic error bounds with only a finite number of samples?

We will introduce our system identification algorithms tailored to meet the above requirements. As expected, we discretize time into small intervals, thereby reducing the problem to a discrete system. The interesting part involves ensuring that the discretization remains bijective and that the inversion is unbiased. Our algorithm identifies the continuous system using only a finite number of samples from the discrete system. We further propose a information theoretic lower bound that shows our algorithm is optimal.

As an application of our system identification methods, we study an online continuous-time linear control problem as introduced in [30]. In this context, exploration is essential for estimating unknown parameters, with the goal of identifying a more optimal control policy that narrows the performance gap. The primary challenge involves finding the right balance between exploration and exploitation. Leveraging our identification method for more efficient parameter estimation allows us to effectively manage exploration and exploitation, achieving an expected regret of O ( √ T ) over a single trajectory with only O ( T ) samples in time horizon T . This surpasses the previously best known result of O ( √ T log( T )) , which needs continuous data collection from the system.

We summarize our contributions below.

1. When the system can be stabilized by a known controller, we establish an algorithm with O ( T ) samples that achieves estimation error O ( √ 1 /T ) on a single trajectory with running time T , which is shown in Theorem 2. We also provide Theorem 4 which shows that the estimation error of our system identification method is optimal up to constant factors.
2. When a stable controller is not available, we can use N independent short trajectories to obtain estimators with error O ( √ 1 /N ) , as is shown in Theorem 5 .
3. We apply our system identification method to an online continuous linear control algorithm, which only requires O ( T ) samples and achieves O ( √ T ) regret on a single trajectory with lasting time T (Theorem 6), improving upon the best known result O ( √ T log( T )) in [30].

## 2 Related Works

Control of both discrete and continuous linear dynamical systems have been extensively studied in various settings, such as linear quadratic optimal control [27], H 2 stochastic control [10], H ∞ robust control [34, 17] and system identification [21, 24]. Below we introduce some of the important results on both system identification and optimal control for linear dynamical systems.

System Identification Earlier literature focused primarily on the asymptotic convergence of system identification [6, 25]. Recently, there has been a resurgence of interest in non-asymptotic system identification for discrete-time systems. [9] studied the sample complexity of multiple trajectories, with O ( √ 1 /N ) estimation error on N independent trajectories. For systems with dynamics x t +1 = Ax t + w t (without controllers), [33] established an analysis for O ( √ 1 /T ) estimation error on a single stable trajectory with running time T , while [13] and [29] extended to more general discrete-time systems.

Non-asymptotic analyses for continuous-time linear system are less studied. Recently, [4] examined continuous-time linear quadratic control systems with standard brown noise and unknown system dynamics. Our algorithm is specifically designed for finite observations, achieving an error rate that cannot be attained through the direct discretization of integrals as done in [4].

Regret Analysis of Online Control In online control, if the system's parameters are known, achieving the optimal control policy in this setup can be straightforward [34, 35]. However, when the system parameters are unknown, identifying the system incurs regret. [1] achieved an O ( √ T ) regret for discrete-time online linear control, which has been proven optimal in T under that setting in [32]. Subsequent works have extended this setup, focusing on worst-case analysis with adversarial noise and cost, including [26, 8, 22, 32, 2]. These analyses are limited to discrete systems. For continuous-time systems, works of [31, 30, 23] established algorithms for online continuous control that achieves O ( √ T log( T ) ) regret.

## 3 Problem Setups and Notations

In this section, we introduce the background and notation for linear dynamical systems and online control.

## 3.1 Linear Dynamical Systems

We first introduce discrete-time linear dynamical systems as follows: Let x k ∈ R d represent the state of the system at time k , and let u k ∈ R p denote the action at time k . Then, for some linear time-invariant dynamics characterized by A ∈ R d × d and B ∈ R d × p , the transition of the system to the next state can be represented as:

<!-- formula-not-decoded -->

where w k ∈ R d are i.i.d. Gaussian random vectors with zero means and certain covariance.

Similarly, a continuous-time linear dynamical system with stochastic disturbance at time t is defined by a differential equation, instead of a recurrence relation:

<!-- formula-not-decoded -->

In this context, we use X t and U t to represent the state and action in the continuous-time linear system, distinguishing them from x t and u t in discrete-time systems. W t denotes the stochastic noise, which is modeled by standard Brownian motion.

For a continuous control problem, an important question of a linear dynamical system is whether such system can be stably controlled. Below we define the concepts of stable dynamics and stabilizers.

Definition 1. For any square matrix A , define α ( A ) = max i {ℜ ( λ i ) | λ i ∈ λ ( A ) } , where ℜ ( λ ) represents the real part of complex number λ , λ ( A ) is the set of all eigenvalues of A .

Definition 2. A matrix A ∈ R d × d is stable if α ( A ) &lt; 0 . A control matrix K ∈ R p × d is said to be a stabilizer for system ( A,B ) if A + BK is stable.

Under the above definition, a stable dynamic guarantees that the state can automatically go to the origin when no external forces are added, while applying a stabilizer as the dynamic for controller will also ensure that the state does not diverge.

## 3.2 Continuous-time LQR Problems and Optimal Control

For continuous-time linear systems disturbed by stochastic noise, as introduced in 3.1, we denote the strategy of applying control to such systems through a specific causal policy, f : X → U . This policy maps states X to control inputs U , where the policy at time t can only depend on the states and actions prior to t .

The optimal controls in linear systems are often linear [34, 35], which takes the following form

<!-- formula-not-decoded -->

where K t ∈ R p × d represents the linear parameterization at time t under some policy f ( X ) = KX . Additionally, we define the cost function of applying the action U t = K t X t with linear quadratic regulator (LQR) control. Given predefined symmetric positive definite matrices Q ∈ R d × d and R ∈ R p × p , along with the initial state X 0 , the cost during t ∈ [0 , T ] is denoted by J T , as represented in the following equation:

<!-- formula-not-decoded -->

Here the expectation is taken over the randomness of X t .

Among all the polices there exists an optimal mapping f ∗ which minimizes J T . When the system is dominated by dynamics ( A,B ) , with the state transits according to (2), such optimal K t can be computed via the Lyapunov matrix P t that solves the Ricatti differential equation [35]:

<!-- formula-not-decoded -->

Then, under f ∗ the action dynamic is set to be K t = -R -1 B T P t .

When T → + ∞ , the starting dynamic P 0 converges to some special dynamic P ∗ satisfying

<!-- formula-not-decoded -->

and the optimal control policy for infinite time horizon is by setting K ∗ = -R -1 B T P ∗ and apply the action by U t = K ∗ X t .

Online Control Problems. Online learning aims to find a strategy to output a sequence of controls { U t } that minimizes the cost J T without knowing the system parameters A,B . In this scenario, the algorithms explore to obtain valuable information, such as estimators ( ˆ A, ˆ B ) for ( A,B ) , while simultaneously exploit gathered information to avoid large instantaneous cost.

To quantify the progress in an online learning problem with horizon T , one quantity of interest is the regret R T , which quantifies the performance gap between the control taken U t = f ( X t ) and a baseline optimal policy which takes U t = K ∗ X t = -R -1 B T P ∗ X t , where K ∗ is defined in (5). Formally, by denoting J T be the expected cost under f , and J ∗ T be the expected cost under the baseline optimal policy, the regret R T is represented as:

<!-- formula-not-decoded -->

Other Notations Denote the d-dimensional unit sphere S d -1 = { v ∈ R d , ∥ v ∥ 2 = 1 } , where ∥ · ∥ 2 is the L 2 norm. For any matrix A ∈ R m × n , denote ∥ A ∥ be the spectral norm of A , or equivalently,

<!-- formula-not-decoded -->

## 4 The Proposed System Identification Method

In this section we propose our system identification method. Before presenting our method, we first introduce the formal definition of system identification and the finite observation setting.

## 4.1 System Identification and Finite Observation

We start with the definition of system identification.

Definition 3 (System Identification) . The system identification task aims to recover the true system dynamics matrices A and B by observing the system's response over time. Specifically, one selects a time horizon T and a sequence of actions U , observes the resulting states X , and computes estimates ˆ A and ˆ B of the true dynamics. The goal is to design an effective algorithm that achieves the following non-asymptotic estimation bound:

<!-- formula-not-decoded -->

for some function f depending on T . In particular, as T →∞ , we expect the estimation error f ( T ) to converge to zero.

Next, we introduce the finite observation assumption . Under this setting, the number of observed states N grows at most linearly with the trajectory running time T . In other words, for any trajectory of length T , we can only access a finite set of states { X 1 , X 2 , . . . , X N } to identify the system, where N = O ( T ) and does not exhibit superlinear growth.

To analyze the continuous-time system, we need to discretize it. Prior works [4, 31, 30] commonly approximate the dynamics using

<!-- formula-not-decoded -->

However, this approximation introduces a discretization error between the approximated and true dynamics. The error term, characterized by ( e hA -I ) /h -A , is of order O ( h ) . Consequently, the sampling interval h must be chosen as O (1 / √ T ) to ensure that discretization error does not dominate. This leads to a super-linear sampling complexity of m = T/h = Ω( T 3 / 2 ) , which violates the finite observation assumption and significantly increases computational demands.

In contrast, our method overcomes this limitation by directly estimating the matrix exponential e Ah in Lemma 1, and subsequently recovering ( A,B ) from this estimate. As a result, our approach avoids discretization error entirely, allowing the sampling interval to depend solely on system parameters rather than the total sampling time T . This innovation reduces the sampling complexity to grow linearly with T , offering significant computational advantages.

## 4.2 Algorithm Design

Then we introduce our algorithm. We choose a small sampling time interval h across a single trajectory of time length T . We then divide the time into small intervals and consider the state evolution within each interval. We get the following Lemma:

Lemma 1. In the time interval [ t, t + h ] , the following transition function holds:

<!-- formula-not-decoded -->

Here, w t is Gaussian noise N (0 , Σ) with covariance Σ = ∫ h s =0 e As e A T s ds . The formal proof of this Lemma is deferred to the Appendix A.2.

This transition equation connects continuous-time and discrete-time systems. In our method, the whole trajectory is partitioned into intervals with proper determined length h . During time t ∈ [ kh, ( k + 1) h ] , we observe a state x k at time t = kh , and fix the action U t ≡ u k in this interval. Denoting A ′ = e Ah and B ′ = [ ∫ h s =0 e A ( h -s ) ds ] B , then the set of observations { x k | k = 0 , 1 , 2 , ... } and actions { u k | k = 0 , 1 , 2 , ... } follow the standard discrete-time linear dynamical system:

<!-- formula-not-decoded -->

Then we can apply the system identification method of discrete-time system [33, 9]. However, different from classical discrete-time systems, continuous-time systems present new challenges. The crucial one is that knowing e Ah is not sufficient to determine A , because the matrix exponential function f ( X ) = e X is not one-to-one. This means we might obtain an incorrect estimator ˆ A by solving e ˆ Ah = M , where M is the estimate of e Ah . From the above analysis, we introduce our assumptions of the algorithm.

Assumption 1 (Assumptions for Algorithm 1 and Theorem 2) . We assume

1. The linear dynamic A is stable, with α ( A ) &lt; 0 (see Definition 1). This is equivalent to assuming the existence of a stable controller K and then set A ← A + BK .
2. ∥ A ∥ ≤ κ A , ∥ B ∥ ≤ κ B for some known κ A , κ B ( κ A , κ B need not be closed to ∥ A ∥ , ∥ B ∥ ).
3. The sample interval h is chosen to be h = 1 15 κ A .

With the above assumptions, we design our algorithm as described in Algorithm 1. In the k -th interval of length h , the state x k is observed at the beginning, and a randomly selected action u k is applied

## Algorithm 1 System identification algorithm for stable system

Input: Running time T , sample interval h satisfying the condition in Assumption 1. Define the number of samples T 0 = ⌈ T/h ⌉ . for k = 0 , . . . , T 0 -1 do Sample the action u k i.i.d. ∼ N (0 , I p ) . Use the action U t ≡ u k during the time period t ∈ [ kh, ( k +1) h ] . Observe the new state x k +1 at time ( k +1) h . end for Compute system estimates ( ˆ A, ˆ B ) via (8).

uniformly throughout the interval. The state-action pair x k , u k is then used to estimate the discretized dynamics via:

<!-- formula-not-decoded -->

The continuous-time dynamics ( A,B ) are then recovered from ( ˜ A, ˜ B ) . Under the condition ∥ A ∥ h ≪ 1 , we employ Taylor expansion to compute ˆ Ah = log( ˜ A ) , approximating Ah . The estimators ( ˆ A, ˆ B ) are given by:

<!-- formula-not-decoded -->

We now establish the efficiency of our algorithm and derive the main theorem as follows.

Theorem 2 (Upper bound) . In Algorithm 1, there exists a constant C ∈ poly ( | α ( A ) | -1 , κ A , κ B ) such that, ∀ 0 &lt; δ &lt; 1 2 , when T ≥ C ( ∥ X 0 ∥ 2 2 +log 2 1 /δ ) , with probability at least 1 -δ , we have:

<!-- formula-not-decoded -->

We defer the proof of the theorem to Appendix A.4 and highlight the key idea below. The key idea of the proof is to analyze the error transformation from the discrete system to the original system. We prove Lemma 3, which shows that the errors in the discrete and original systems differ only by a constant factor. This allows us to focus solely on the discrete system identification problem.

Lemma 3. In Algorithms 1, suppose we obtain the relative error ∥ ˜ A -A ′ ∥ , ∥ ˜ B -B ′ ∥ ≤ ϵ for some ϵ ≤ 1 15 and ∥ Ah ∥ ≤ 1 15 . Then, the relative error in the original system satisfies:

<!-- formula-not-decoded -->

From this lemma, it follows that if we develop a system identification algorithm for the discrete system that produces dynamics estimates ˜ A and ˜ B with minimal error, we can obtain accurate estimates for the original system. The remaining task is to analyze the discrete system with the transition function x k +1 = Ax k + Bu k + w k , which has been discussed in previous works such as [33].

## 4.3 Lower Bound

In this section, we discuss the lower bound of the problem. We prove Theorem 4 and establish that this method has already attained the optimal convergence rate for parameter estimation. The theorem primarily asserts that, given a single trajectory lasting for time T , any algorithm that estimates system parameters solely based on an arbitrarily large number of finite observed states cannot guarantee an estimation error of o ( √ 1 /T ) .

Theorem 4 (Lower bound) . Suppose T ≥ 1 be the running time of a single trajectory of continuoustime linear differential system, represented as in (2) . Then there exist constants c 1 , c 2 independent

of d such that, for any finite set of observed points { t 0 = 0 , t 1 , t 2 , ..., t N = T } , and any (possibly randomized) estimator function ϕ : { X t 0 , X t 1 , ..., X t N } → R d × d , there exists system parameter A,B satisfying P [ ∥ ϕ ( { X t i } i ≤ N ) -A ∥ ≥ c 1 √ T ] ≥ c 2 . Here the probability is with respect to noise.

In Theorem 4, the mapping ϕ can refer to the output of any algorithm that exclusively relies on the finite set of states X t 0 , X t 1 , ..., X t N . The interesting observation is that the lower bound does not decrease with a larger observation number N .

We defer the proof of the theorem to the Appendix A.6 and provide a proof sketch below. We consider two sets of dynamics, ( A, 0) and ( ¯ A, 0) , where both A and ¯ A are stable, and | A -¯ A | = 2 c 1 √ T . Our key observation is that for the two distributions of observed states S k = { X t 0 , X t 1 , ..., X t k } and ¯ S k = { ¯ X t 0 , X t 1 , ..., X t k } , where X corresponds to the linear dynamic A and ¯ X corresponds to ¯ A , the KL divergence between S k +1 and ¯ S k +1 increases by at most c T ( t k +1 -t k ) . Here, c is a universal constant independent of t k and t k +1 . Thus, regardless of how the observation times are selected, the KL divergence between the observed states remains bounded.

Remark 1 (The Discussion of Lower Bound) . The construction in Theorem 4 involves matrices A and ¯ A that depend on T , specifically with ∥ A -¯ A ∥ = 2 c 1 √ T . One might be concerned that such a T -dependent construction lacks interpretability since the true system parameters are independent of T . However, as shown in Appendix A.6, the matrices are taken as A = -I d and ¯ A = -I d -U , where U has only one nonzero entry at position (1 , 1) equal to 1 5 √ T . For these matrices, the key constants in the upper bound remain uniformly bounded: the inverse stability margin 1 | α ( A ) | equals 1 for A = -I d and is at most 1 1+ 1 5 √ T ≤ 1 for ¯ A ; the condition number κ ( A ) equals 1 for A = -I d and is at most

1 + 1 5 √ T ≤ 2 for ¯ A . Thus both quantities are controlled by universal constants, independent of T , ensuring that the lower and upper bounds are comparable up to a constant factor.

## 4.4 Finding an Initial Stable Controller

While previous work on continuous-time system identification [4, 30] always assumes a known stable controller, our method extends to cases where a stabilizer is not known in advance. For general ( A,B ) , where a stabilizer is not predetermined, relying on a single trajectory is not feasible, as the state may diverge rapidly before obtaining a stable controller is obtained. Instead, we first find a stable controller K using multiple short-interval trajectories and then employ it in Algorithm 3 for online control. Below, we list the assumptions on system parameters.

Assumption 2 (Assumptions for Algorithm 2 and Theorem 5) . We assume

1. The constants κ A , κ B , h follow the same assumptions as in 1.
2. The running time T for each trajectory is small, say, T = T 0 h where T 0 ∈ N and T 0 ≤ 10 .

Then, we employ multiple short trajectories to identify A and B as outlined in Algorithm 2. Similar to what is demonstrated in [9], this procedure results in an O ( H -1 / 2 ) estimation error on the trajectory number H .

Theorem 5. In Algorithm 2, there exists a constant C ∈ poly ( κ A , κ B ) such that w.p. at least 1 -δ , the estimation error of ( ˆ A, ˆ B ) from H trajectories satisfies:

<!-- formula-not-decoded -->

The proof details are shown in the Appendix A.5.

## 5 A Continuous Online Control Algorithm with Improved Regret

In this section, we apply our system identification method to a continuous LQR online control algorithm. Recall the setup introduced in Section 3.2 where we want to minimize the regret R T defined in (6). We will show in this section that with O ( T ) samples, our algorithm achieves O ( √ T )

```
Input: T , T 0 , h as in Assumption 2, number of trajectories H . for l = 1 , . . . , H do for k = 0 , . . . , T 0 -1 do Sample the action u l k i.i.d. ∼ N (0 , I p ) , use the action U t ≡ u l k during t ∈ [ kh, ( k +1) h ] . Observe the new state x l k +1 at time ( k +1) h . end for end for Compute ( ˜ A, ˜ B ) by ( ˜ A, ˜ B ) ∈ arg min ( A,B ) 1 2 ∑ H l =1 ∥ ∥ x l T 0 -Ax l T 0 -1 -Bu l T 0 -1 ∥ ∥ 2 2 . Compute ¯ A, ¯ B as in (8), let ( ˆ A, ˆ B ) = ( ¯ A, ¯ B ) be estimates for system dynamics ( A,B ) .
```

Algorithm 2 Multi-trajectory system identification algorithm

expected regret on a single trajectory, thereby improving upon the previous O ( √ T log( T ) ) result. We list the assumption for the online LQR problems below.

Assumption 3 (Assumptions for Algorithm 3 and Theorem 6) . We assume that:

1. A stabilizer K for ( A,B ) (see Definition 2) with α ( A + BK ) &lt; 0 is known in advance.
2. Sample distance h satisfies h = 1 15 κ , where κ ≥ ∥ A ∥ + ∥ B ∥∥ K ∥ ≥ ∥ A + BK ∥ is known.
3. Denote P ∗ be the solution in (5) and K ∗ = -R -1 B T P ∗ be the baseline control dynamic.
4. Q,R are positive-definite symmetric matrices with bounded spectral norms ∥ Q ∥ , ∥ R ∥ ≤ M and for some µ &gt; 0 , µI ⪯ Q,µI ⪯ R .

## 5.1 An O ( √ T ) Regret Algorithm for Continuous Online Control

Our online continuous control algorithm is outlined in Algorithm 3, and we provide a detailed description below. Algorithm 3 is divided into two phases, exploration and exploitation. For the first exploration phase, a previously known stabilizer K is applied to prevent the state from diverging. During the k -th interval, by setting U t = KX t + u k , the state X t transits according to

<!-- formula-not-decoded -->

Since A + BK is stable, through replacing A in Theorem 2 by A + BK in Algorithm 3, we can obtain a set of estimators ( ˆ A, ˆ B ) for ( A,B ) with small error. This further allows us to accurately estimate ( A,B ) , thereby a controller ¯ K = -R -1 ( ˆ B ) T P closed to K ∗ is obtained.

During exploitation phase, the near-optimal controller is deployed to minimize the cost, resulting in a regret of O ( √ T ) (see Theorem 6). However, as we lack direct feedback on whether ¯ K is a stabilizer, we need to detect its stability. Our approach involves replacing it with the known stabilizer K whenever the state deviates too far. Then we introduce the theorem of the regret analysis:

Theorem 6. Let J T be the expected LQR cost introduced in (3) that takes the action U t as in Algorithm 3. Then for some constant C ∈ poly ( κ, M, µ -1 , | α ( A + BK ) | -1 , | α ( A + BK ∗ ) | -1 ) , the regret satisfies:

<!-- formula-not-decoded -->

Proof Sketch of Theorem 6 We analyze the two phases of our algorithm. During the exploration phase, the stabilizing controller K effectively bounds the trajectory's radius, ensuring the average cost per unit time is within O (1) , resulting in a total exploration cost of C √ T . In the subsequent exploitation phase, we analyze two scenarios separately. The first scenario occurs when the estimators ( ˆ A, ˆ B ) have large errors or when ∥ X t ∥ 2 ≥ T 1 / 5 for some t ∈ [ √ T, T ] . This situation is rare and contributes a limited expected cost that can be bounded by a constant. The second scenario occurs when ( ˆ A, ˆ B ) are accurately estimated, and the control U t = -R -1 ( ˆ B ) T PX t is applied throughout the exploitation phase. In this case, the trajectory's performance is straightforward to analyze, and the expected cost is bounded by O ( √ T ) + J ∗ T .

Algorithm 3 Continuous online control algorithm

Input: K,h which follows Assumption 3, running time T √

<!-- formula-not-decoded -->

Sample the action u k i.i.d. ∼ N (0 , I p ) .

For

t

∈

[

kh,

(

k

+1)

h

]

, set

Observe the new state

## end for

## Do system identification and estimate dynamics:

Compute ( ˜ A, ˜ B ) according to (7) by using { x k , u k } .

Compute ¯ A, ¯ B by (8) with ˜ A, ˜ B , and estimators ( ˆ A, ˆ B ) by ˆ A = ¯ A -¯ BK , ˆ B = ¯ B .

If ˆ A is stable, compute P by (5) with estimated ˆ A, ˆ B , and set ¯ K = -R -1 ( ˆ B ) T P .

If ˆ A is not stable or P computed above satisfies ∥ P ∥ ≥ T 1 5 , then set ¯ K = K .

## Perform exploitation: √

For t ∈ [ T, T ] , set U t = ¯ KX t .

Detect bad policy and prevent the trajectory from diverging: √

If for some t 0 ≥ T , ∥ X t 0 ∥ ≥ T 1 5 , then set U t = KX t for t ∈ [ t 0 , T ] .

By summing the expected costs, the total exploration cost is bounded by O ( T ) , and the exploitation cost is bounded by J ∗ T + O ( √ T ) . By the definition of regret, R T = J T -J ∗ T , the total regret is O ( √ T ) , leading to the result of Theorem 6.

√

Our result is closely related to the result in [30], along with its similar version [12]. They achieve O ( √ T log( T )) regret. However, they further assumes a known stabilization set for obtaining a stable controller, which is stronger compared with ours. Such difference exists because our approach detects divergence and avoids sticking to a controller which is not stable. Morever, in [30], the exploration and exploitation is simultaneous, where a random matrix is added to the near-optimal controller so that both A and B can be identified. This causes an extra log( T ) factor to the regret. In contrast, our algorithm follows an explore-then-commit structure, which is enabled by the efficient system identification results presented previously. Finally, we additionally considered the setup of finite observation, which is not discussed in [30].

## 5.2 Experiments

In this section, we conduct simulation experiments for the baseline algorithm and our proposed algorithm. The baseline algorithm follows the work of [30]. We set d = p = 3 for simplicity. Each element of A is sampled uniformly from [ -1 , 1] , making A unstable with high probability. The matrix B , Q , R are set as the identity matrix I 3 . The sampling interval is set to h = 1 30 .

First, we run Algorithm 1 for system identification. We plot the expected Frobenius norms of the error matrices ∥ ˆ A -A ∥ 2 F and ∥ ˆ B -B ∥ 2 F . The results demonstrate that our algorithm can identify A and B within sufficient running time or number of trajectories.

Next, we compare Algorithm 3 with the baseline algorithm. We analyze the normalized regret R ( T ) /T 1 / 2 for different t ∈ [600 , 10000] and plot the results in Figure 1. The results show that our online control algorithm with system identification achieves constant normalized regret (i.e., O ( √ T ) regret) and outperforms the baseline algorithm when T is sufficiently large.

## 6 Conclusions, Limitations and Future Directions

In this work, we establish a novel system identification method for continuous-time linear dynamical systems. This method only uses a finite number of observations and can be applied to an algorithm for online LQR continuous control which achieves O ( √ T ) regret on a single trajectory. Compared with existed works, our work not only eases the requirement for data collection and computation, but achieves fast convergence rate in identifying the unknown dynamics as well.

x

k

U

+1

t

=

KX

at time

t

(

k

+

u

k

+1)

.

h

.

Figure 1: The empirical validation of our algorithm. Left: Identification of system dynamics using a single trajectory. Right: The normalized regret R ( T ) /T 1 / 2 of the baseline algorithm and our algorithm. The results show that our algorithm achieves small identification error and is more efficient than the baseline algorithm.

<!-- image -->

Although our method achieves near-optimal results in system identification and LQR online control for continuous systems with stochastic noise, many questions remain unsolved. First, it is unclear whether our system identification approach can be extended to more challenging setups, such as deterministic or adversarial noise. Additionally, many practical models are non-linear, raising the question of under what conditions discretization methods are effective. We believe these questions are crucial for real-world applications.

## References

- [1] Yasin Abbasi-Yadkori and Csaba Szepesvári. Regret bounds for the adaptive control of linear quadratic systems. In Proceedings of the 24th Annual Conference on Learning Theory , pages 1-26. JMLR Workshop and Conference Proceedings, 2011.
- [2] Naman Agarwal, Brian Bullins, Elad Hazan, Sham Kakade, and Karan Singh. Online control with adversarial disturbances. In International Conference on Machine Learning , pages 111-119. PMLR, 2019.
- [3] Karl Johan Åström and Peter Eykhoff. System identification-a survey. Automatica , 7(2):123162, 1971.
- [4] Matteo Basei, Xin Guo, Anran Hu, and Yufei Zhang. Logarithmic regret for episodic continuoustime linear-quadratic reinforcement learning over a finite-time horizon, 2022.
- [5] Marco C Campi and PR Kumar. Adaptive linear quadratic gaussian control: the cost-biased approach revisited. SIAM Journal on Control and Optimization , 36(6):1890-1907, 1998.
- [6] Marco C Campi and PR Kumar. Adaptive linear quadratic gaussian control: the cost-biased approach revisited. SIAM Journal on Control and Optimization , 36(6):1890-1907, 1998.
- [7] Xinyi Chen and Elad Hazan. Black-box control for linear dynamical systems. In Conference on Learning Theory , pages 1114-1143. PMLR, 2021.
- [8] Alon Cohen, Tomer Koren, and Yishay Mansour. Learning linear-quadratic regulators efficiently with only √ T regret. In International Conference on Machine Learning , pages 1300-1309. PMLR, 2019.
- [9] Sarah Dean, Horia Mania, Nikolai Matni, Benjamin Recht, and Stephen Tu. On the sample complexity of the linear quadratic regulator, 2018.
- [10] Vasile Dragan, Toader Morozan, and Adrian Stoica. H2 optimal control for linear stochastic systems. Automatica , 40(7):1103-1113, 2004.
- [11] R Durrett. Probability: Theory and examples, cambridge series in statistical and probabilistic mathematics, 2010.

- [12] Mohamad Kazem Shirani Faradonbeh. Regret analysis of certainty equivalence policies in continuous-time linear-quadratic systems. In 2022 26th International Conference on System Theory, Control and Computing (ICSTCC) , pages 368-373. IEEE, 2022.
- [13] Mohamad Kazem Shirani Faradonbeh, Ambuj Tewari, and George Michailidis. Finite time identification in unstable linear systems. Automatica , 96:342-353, 2018.
- [14] Gene H Golub and Charles F Van Loan. Matrix computations . JHU press, 2013.
- [15] Elad Hazan, Sham Kakade, and Karan Singh. The nonstochastic control problem. In Algorithmic Learning Theory , pages 408-421. PMLR, 2020.
- [16] Rolf Johansson, Anders Robertsson, Klas Nilsson, and Michel Verhaegen. State-space system identification of robot manipulator dynamics. Mechatronics , 10(3):403-418, 2000.
- [17] IS Khalil, JC Doyle, and K Glover. Robust and optimal control . Prentice hall, 1996.
- [18] Donald E Kirk. Optimal control theory: an introduction . Courier Corporation, 2004.
- [19] David Kleinman. On an iterative technique for riccati equation computations. IEEE Transactions on Automatic Control , 13(1):114-115, 1968.
- [20] Michael J Korenberg. A robust orthogonal algorithm for system identification and time-series analysis. Biological cybernetics , 60(4):267-276, 1989.
- [21] PR Kumar. Optimal adaptive control of linear-quadratic-gaussian systems. SIAM Journal on Control and Optimization , 21(2):163-178, 1983.
- [22] Sahin Lale, Kamyar Azizzadenesheli, Babak Hassibi, and Anima Anandkumar. Explore more and improve regret in linear quadratic regulators. arXiv , 2020.
- [23] Jingwei Li, Jing Dong, Can Chang, Baoxiang Wang, and Jingzhao Zhang. Online control with adversarial disturbance for continuous-time linear systems. Advances in Neural Information Processing Systems , 37:48130-48163, 2024.
- [24] Lennart Ljung. System identification. In Signal analysis and prediction , pages 163-173. Springer, 1998.
- [25] Lennart Ljung. System identification . Springer, 1998.
- [26] Horia Mania, Stephen Tu, and Benjamin Recht. Certainty equivalence is efficient for linear quadratic control. Advances in Neural Information Processing Systems , 32, 2019.
- [27] Volker Ludwig Mehrmann. The autonomous linear quadratic control problem: theory and numerical solution . Springer, 1991.
- [28] Stephane Ross and J Andrew Bagnell. Agnostic system identification for model-based reinforcement learning. arXiv preprint arXiv:1203.1007 , 2012.
- [29] Tuhin Sarkar and Alexander Rakhlin. Near optimal finite time identification of arbitrary linear dynamical systems. In International Conference on Machine Learning , pages 5610-5618. PMLR, 2019.
- [30] Mohamad Kazem Shirani Faradonbeh and Mohamad Sadegh Shirani Faradonbeh. Online reinforcement learning in stochastic continuous-time systems. In Gergely Neu and Lorenzo Rosasco, editors, Proceedings of Thirty Sixth Conference on Learning Theory , volume 195 of Proceedings of Machine Learning Research , pages 612-656. PMLR, 12-15 Jul 2023.
- [31] Mohamad Kazem Shirani Faradonbeh, Mohamad Sadegh Shirani Faradonbeh, and Mohsen Bayati. Thompson sampling efficiently learns to control diffusion processes. Advances in Neural Information Processing Systems , 35:3871-3884, 2022.
- [32] Max Simchowitz and Dylan J. Foster. Naive exploration is optimal for online lqr, 2023.
- [33] Max Simchowitz, Horia Mania, Stephen Tu, Michael I. Jordan, and Benjamin Recht. Learning without mixing: Towards a sharp analysis of linear system identification, 2018.

- [34] Robert F Stengel. Optimal control and estimation . Courier Corporation, 1994.
- [35] Jiongmin Yong and Xun Yu Zhou. Stochastic controls: Hamiltonian systems and HJB equations , volume 43. Springer Science &amp; Business Media, 1999.

## A System Identification for Continuous-time Linear System

In this section, we analysis our system identification method in Algorithm 1 and Algorithm 2. As a preparation, we establish some properties of matrix exponentials and their inverses.

## A.1 Matrix Exponential

For a matrix exponential e At , where the largest real component of A 's eigenvalues is denoted by α ( A ) , the spectral norm of e At can be well-bounded [14], as demonstrated in Lemma 7.

Lemma 7. Suppose an n × n matrix A satisfies that 0 &gt; α ( A ) = max {ℜ ( λ i ) | λ i ∈ λ ( A ) } . Let Q H AQ = diag( λ i ) + N be the Schur decomposition of A , and let M S ( t ) = ∑ n -1 k =0 ∥ Nt ∥ k 2 k ! . Then for t &gt; 0 , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In a special case where α ( A ) ≤ 0 , since M ( t ) ≥ 1 for all t

<!-- formula-not-decoded -->

We also show some properties of matrix inverse in the following Lemma 8.

Lemma 8 (Matrix inverse) . For any A ∈ R d × d and t such that 0 &lt; ∥ At ∥ ≤ 1 10 , we have the following estimation of e At :

<!-- formula-not-decoded -->

and if we denote A 1 = e At , then A also satisfies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We expand e At by which follows that

<!-- formula-not-decoded -->

Since ∥ A 1 -I d ∥ &lt; 1 , the progression A 2 = ∑ k ≥ 1 ( -1) k +1 kt ( A 1 -I d ) k converges, and thus e A 2 t = e At . Furthermore, it can be computed that

<!-- formula-not-decoded -->

Now we show that A 2 = A . We have already known that ∥ At ∥ and ∥ A 2 t ∥ are small. We also note that the function f : X → e X ( ∥ X ∥ ≤ 1 8 ) constitutes a one-to-one mapping. This assertion is supported by the observation that for any X 1 , X 2 such that ∥ X 1 ∥ , ∥ X 1 + X 2 ∥ ≤ 1 8 , we have ∥ X 2 ∥ ≤ 1 4 , implying that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then ∥ ∥ e X 1 + X 2 -e X 1 ∥ ∥ ≥ 1 2 ∥ X 2 ∥ , which means f is one-to-one, and thereby leading that A 2 = A .

## A.2 Proof of Lemma 1

Lemma 1. In the time interval [ t, t + h ] , the following transition function holds:

<!-- formula-not-decoded -->

Proof. Using Newton-Leibniz formula, we have

<!-- formula-not-decoded -->

Let w t + t 1 = BU t + t 1 + dW t + t 1 dt , we have:

<!-- formula-not-decoded -->

where the last equality we use the Fubini theorem to change the integral order of t 1 and t 2 to calculate the second term.

Suppose we already have the following equality for integer m (The case m = 2 has been checked above):

<!-- formula-not-decoded -->

Then, replace X t + t m -X t by ∫ t m 0 [ AX t + A ( X t + t m +1 -X t ) + w t m +1 ] d t m +1 , we get:

<!-- formula-not-decoded -->

Using the property that

<!-- formula-not-decoded -->

We finally get

<!-- formula-not-decoded -->

In the calculation of the second term we use the Fubini theorem to change the integral order of t m +1 and t 1 , t 2 ...t m .

So the induction hypothesis is true. For any positive integer m, we have the following equality:

<!-- formula-not-decoded -->

For the time interval ˜ t ∈ [ t, t + h ] , by the continuity of X ˜ t we know that X ˜ t is uniformly bounded by some constant C . Therefore we have the convergence of the third term in the RHS:

<!-- formula-not-decoded -->

Therefore, we finally get:

<!-- formula-not-decoded -->

Now, we use w t + s = BU t + s + dW t + s dt , we get:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where w t is Gaussian noise N (0 , Σ) with covariance Σ = ∫ h 0 e As e A T s ds .

## A.3 Proof of Lemma 3

We restate Lemma 3 and provide the proof here.

Lemma 3 In Algorithm 1, 2, suppose we have obtained the relative error ∥ ˜ A -A ′ ∥ , ∥ ˜ B -B ′ ∥ ≤ ϵ for some ϵ ≤ 1 15 and ∥ Ah ∥ ≤ 1 15 , then we have the following relative error of the primal system:

<!-- formula-not-decoded -->

where C is a constant independent of h .

Proof. Firstly, according to Lemma 8, the estimated ˜ A is not too far away from I d , as we have:

<!-- formula-not-decoded -->

Then, from (8) we can bound the matrix norm ∥ ∥ ∥ ˆ Ah ∥ ∥ ∥ by

<!-- formula-not-decoded -->

Now, let's denote A 1 = Ah and A 2 = ˆ Ah -A 1 , satisfying the relations A ′ = e A 1 and ˜ A = e A 1 + A 2 . It is given that ∥ A 1 ∥ ≤ 1 15 and ∥ A 2 ∥ ≤ ∥ A 1 ∥ + ∥ ˆ Ah ∥ ≤ 1 4 , so by (13), we obtain that ∥ ˆ A -A ∥ h = ∥ A 2 ∥ ≤ 2 ∥ ˜ A -A ′ ∥ , which follows that ∥ ˆ A -A ∥ ≤ 2 h ∥ ˜ A -A ′ ∥ ≤ 2 h ϵ .

Next, we will upper bound the estimation error of B . Let A h = ∫ h t =0 e At dt and ¯ A h = ∫ h t =0 e ˆ At dt , satisfying

<!-- formula-not-decoded -->

This follows that

<!-- formula-not-decoded -->

Since B and its estimator ˆ B satisfy that

<!-- formula-not-decoded -->

we can upper bound the estimation error ∥ ∥ ∥ ˆ B -B ∥ ∥ ∥ by

<!-- formula-not-decoded -->

where the last inequality is because ∥ B ′ ∥ ≤ ∥ A h ∥∥ B ∥ ≤ 2 h ∥ B ∥ .

Since 2 ∥ B ∥ ≤ 2 κ B ≤ 1 h · 2 κ B 15 κ A ≤ κ B κ A , we obtain Lemma 3.

## A.4 Proof of Theorem 2

In this section, we derive the proof of Theorem 2. We upper bound the estimation errors of intermediate dynamics ( A ′ , B ′ ) , obtained as in (7). We first prove Lemma 9 below, providing system identification results on a single trajectory with a stable controller.

Lemma 9. Consider the trajectory x k +1 = Ax k + Bu k + w k with A ∈ R d × d , ∥ A ∥ &lt; 1 , B ∈ R d × p ; u k ∼ N (0 , I p ) and w k ∼ N (0 , Σ) are i.i.d. random variables. Suppose we compute ( ˆ A, ˆ B ) by

<!-- formula-not-decoded -->

Then there exists a constant C (depending only on A , B , d , p and Σ ) such that for T ≥ C ( ∥ X 0 ∥ 2 2 +log 2 (1 /δ ) ) , w.p. at least 1 -δ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We first provide Lemma 10, which is used as the base of Lemma 9.

Lemma 10. Consider A ∈ R d × d such that ρ ( A ) &lt; 1 and the system X k +1 = AX k + w k with w k ∼ N (0 , Σ) be i.i.d. random variables. Suppose we estimate A as in (7) . Then there exists a constant C depending on A , Σ and d such that for T ≥ C ( ∥ X 0 ∥ 2 2 +log(1 /δ )) , w.p. at least 1 -δ , we have:

<!-- formula-not-decoded -->

The work of [33] has discussed such systems in their Theorem 2.4, and we list it below:

Theorem 11. Fix ϵ, δ ∈ (0 , 1) , T ∈ N and 0 ≺ Γ sb ≺ ¯ Γ . Then if ( X t , Y t ) t ≥ 1 ∈ ( R d × R n ) T is a random sequence such that (a) Y t = A ∗ X t + η t , where η t |F t is σ 2 -sub-Gaussian and mean zero, (b) X 1 , ..., X T satisfies the ( k, Γ sb , p ) -small ball condition, and (c) such that P [ ∑ T t =1 X t X T t ̸⪯ T ¯ Γ ] ≤ δ . Then if

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

Here, the ( k, Γ sb , p ) -small ball condition is defined as follows. Let ( Z t ) t ≥ 1 be an F tt ≥ 1 -adapted random process taking values in R . We say ( Z t ) t ≥ 1 satisfies the ( k, ν, p ) -block martingale small-ball (BMSB) condition if, for any j ≥ 0 , one has 1 k ∑ k i =1 P ( | Z j + i | ≥ ν |F j ) ≥ p almost surely. Given a process ( X t ) t ≥ 1 taking values in R d , we say that it satisfies the ( k, Γ sb , p ) -BMSB condition for Γ sb ≻ 0 if, for any fixed w ∈ S d -1 , the process Z t := ⟨ w,X t ⟩ satisfies ( k, √ w T Γ sb w,p ) -BMSB.

In the work of [33], they have discussed the case when X 0 = 0 , and now we modify it to a general starting state X 0 . From (7), we derive the estimation error of A as

<!-- formula-not-decoded -->

For the first term, consider any v ∈ S d -1 , we lower bound v T ( ∑ T -1 k =0 X k X T k ) v . Let a k = v T X k , then a k = v T AX k -1 + v T w k . We claim that for any k ≥ 1 , P [ | a k | ≥ 1 2 | X k -1 ] ≥ 1 2 . Let b k = v T w k , which is independent of X k -1 . It suffices to show that for any c ∈ R , P [ b k ∈ [ c, c +1]] ≤ 1 2 . Since ∥ v ∥ 2 = 1 and w k ∼ N (0 , I d ) , we have b k ∼ N (0 , 1) , from which we estimate the probability as

<!-- formula-not-decoded -->

Based on (22), we can simply choose k = 1 , Γ sb = 1 4 I d and p = 1 2 , then the random sequence ( X i ) i ≥ 0 satisfies the ( k, Γ sb , p ) -BMSB condition. It remains to choose a proper ¯ Γ that meets the condition (c) in Theorem 11.

Since X k = A k X 0 + ∑ k i =1 A k -i w i , we have:

<!-- formula-not-decoded -->

Let Γ ∞ = ∑ k ≥ 0 A k Σ( A k ) T which is bounded and C 1 be a constant such that C 1 ≥ ∑ k ≥ 0 ∥ A k ∥ 2 . We then show that for ¯ Γ = ( C 1 ∥ X 0 ∥ 2 2 T dI d + d ∥ Γ ∞ ∥ I d ) /δ , the condition (c) in Theorem 11 is satisfied. This is because E [ tr ( ∑ T -1 k =0 X k X T k )] = tr ( E [ ∑ T -1 k =0 X k X T k ]) ≤ Tδ d tr( ¯ Γ) so that P [ tr( ∑ T -1 k =0 X k X T k ) ≥ 1 d T tr( ¯ Γ) ] ≤ δ . Furthermore, a necessary condition for ∑ T -1 k =0 X k X T ̸≺ T ¯ Γ is tr( ∑ T -1 k =0 X k X T ) ≥ 1 d T tr( ¯ Γ) .

Now, we apply such ¯ Γ to Theorem 11. It can be computed that

<!-- formula-not-decoded -->

Then when T ≥ C 1 ∥ X 0 ∥ 2 as well as T ≥ 40 (2 d log(20) + d log(4 d (1 + ∥ Γ ∞ ∥ )) + 2 d log(1 /δ )) , we have:

<!-- formula-not-decoded -->

This implies our Lemma 10.

Proof of Lemma 9 As for the estimation error ∥ ˆ A -A ∥ , let w ′ k = Bu k + w k ∼ N (0 , Σ+ BB T ) , which form a sequence of i.i.d random variables. With the results in Lemma 10, there exist some constants C 1 , C 2 such that, as long as T ≥ C 1 ( ∥ X 0 ∥ 2 2 +log(1 /δ ) ) we have:

<!-- formula-not-decoded -->

Now we upper bound the estimation error ∥ ˆ B -B ∥ . With the expression in (7), we obtain:

<!-- formula-not-decoded -->

For the quantities λ -1 min ( ∑ T -1 k =0 u k u T k ) and ∥ ∑ T -1 k =0 u k w T k ∥ , we apply Lemma 2.1. and Lemma 2.2. in the work of [9], where they present the following results.

Lemma 12. Let N ≥ 2 log(1 /δ ) . Suppose f k ∈ R m , g k ∈ R n are independent vectors such that f k ∼ N (0 , Σ f ) and g k ∼ N (0 , Σ g ) for 1 ≤ k ≤ N . With probability at least 1 -δ ,

<!-- formula-not-decoded -->

Lemma 13. Let X ∈ R N × n have i.i.d. N (0 , 1) entries. With probability at least 1 -δ ,

<!-- formula-not-decoded -->

With these two lemmas, we can conclude that if T ≥ 32( d + p ) log(4 /δ ) , then both λ min ( u k u T k ) ≥ 1 2 T and ∥ ∥ ∥ ∑ T -1 k =0 u k w T k ∥ ∥ ∥ ≤ 4 ∥ Σ ∥ 1 / 2 2 √ T ( d + p ) log(18 /δ ) , w.p. at least 1 -δ .

Now we concentrate on the term ∥ ∥ ∥ ∑ T -1 k =0 u k X T k ∥ ∥ ∥ . Since w ′ i = Bu i + w i ∼ N (0 , Σ+ BB T ) , it can be directly computed that, w.p. at least 1 -δ/T , ∥ ∥ ∥ w ′ i ∥ ∥ ∥ 2 ≤ 2 ∥ ∥ d (Σ + BB T ) ∥ ∥ 1 / 2 2 √ log( T/δ ) . Then by union bound we get P [ sup 0 ≤ i ≤ T -1 ∥ ∥ ∥ w ′ i ∥ ∥ ∥ 2 ≤ 2 ∥ Σ+ BB T ∥ 1 / 2 2 √ d log( T/δ ) ] ≤ δ . Furthermore, when sup 0 ≤ i ≤ T -1 ∥ ∥ ∥ w ′ i ∥ ∥ ∥ 2 ≤ 2 ∥ ∥ Σ+ BB T ∥ ∥ 1 / 2 2 √ d log( T/δ ) , we must have

<!-- formula-not-decoded -->

For any u ∈ S p -1 and v ∈ S d -1 , let x i = u T u i (0 ≤ i ≤ T -1) . Then, x i follows a normal distribution x i ∼ N (0 , 1) and { x i } is a sequence of independent random variables. Furthermore, x k is also independent of ( X i ) 0 ≤ i ≤ k . On the other hand, denote y k = X T k v , (23) implies that w.p. at least 1 -δ , for all k we have | y k | ≤ ∥ X 0 ∥ 2 + 2 1 -∥ A ∥ ∥ ∥ Σ+ BB T ∥ ∥ 1 / 2 2 √ d log( T/δ ) := Y . Let

<!-- formula-not-decoded -->

and let F 0 , F 1 , ..., F T be the filtration of X 0 , X 1 , ..., X T , then for any α ≥ 0 ,

<!-- formula-not-decoded -->

implying that E [ e αZ k +1 Y ] ≤ e 1 2 α 2 E [ e αZ k +1 Y ] So we have: E [ e αZ T -1 Y ] ≤ e 1 2 α 2 T . By choosing α = ± √ 1 T , we obtain that

<!-- formula-not-decoded -->

For T d be a 1 4 -net of S d -1 and T p be a 1 4 -net of S p -1 , we use union bound on them and obtain that, w.p. at least 1 -δ

<!-- formula-not-decoded -->

Where the last inequality is because |T p | ≤ 9 p and |T d | ≤ 9 d

Next we upper bound ∥ ∥ ∥ ∑ T -1 k =0 u k X T k ∥ ∥ ∥ . For any u ∗ ∈ S p -1 and v ∗ ∈ S p -1 , with some u ∈ T p and v ∈ T d s.t. ∥ u -u ∗ ∥ 2 , ∥ v -v ∗ ∥ 2 ≤ 1 2 , we have:

<!-- formula-not-decoded -->

This leads ∥ ∥ ∥ ∑ T -1 k =0 u k X T k ∥ ∥ ∥ ≤ 2 sup u ∈T p ,v ∈T d ∣ ∣ ∣ u T ( ∑ T -1 k =0 u k X T k ) v ∣ ∣ ∣ . Therefore, for any δ ∈ (0 , 1 2 ) , we have:

<!-- formula-not-decoded -->

We choose constant C depending on A,B,d,p such that for all T ≥ C ( ∥ X 0 ∥ 2 2 +log 2 (1 /δ ) ) ,

<!-- formula-not-decoded -->

and we further have: whenever T ≥ C ( ∥ X 0 ∥ 2 2 +log 2 (1 /δ ) ) , w.p. at least 1 -3 δ ,

<!-- formula-not-decoded -->

Finally, when T ≥ max ( C ( ∥ X 0 ∥ 2 2 +log 2 (1 /δ ) ) , 32( d + p ) log(4 /δ ) ) , we combine this upper bound with P ( λ min ( ∑ T -1 k =0 u k u T k ) ≤ 1 2 T ) ≤ δ , and obtain Lemma 9.

## A.5 Proof of Theorem 5

Now, we aim to establish Theorem 5. The analysis of system identification for discrete-time linear dynamical systems with multiple trajectories has been thoroughly investigated by [9]. We hereby cite their findings, denoting the relevant result as Lemma 14.

Lemma 14. Suppose we have N i.i.d. trajectories X i k , each is defined by X i ( k +1) h = AX i k + Bu i k + w i k , where T 0 is any integer, u i k ∼ N (0 , I p ) and w i k ∼ N (0 , Σ) are two sets of i.i.d. random variables. Then, for the estimator ( ˆ A, ˆ B ) of

<!-- formula-not-decoded -->

with probability at least 1 -δ , we have:

<!-- formula-not-decoded -->

Combining Lemma 14 with Lemma 3, we directly obtain Theorem 5.

## A.6 Lower Bound of System Identification with Finite Observation

We restate and provide the proof of Theorem 4.

Theorem 4 Suppose T ≥ 1 be the running time of a single trajectory of continuous-time linear differential system, represented as in (2). Then there exist constants c 1 , c 2 independent of d such that, for any finite set of observed points { t 0 = 0 , t 1 , t 2 , ..., t N = T } , and any (possibly randomized) estimator function ϕ : { X t 0 , X t 1 , ..., X t N } → R d × d , there exists bounded A,B satisfying P [ ∥ ϕ ( { X i } i ≤ N ) -A ∥ ≥ c 1 √ T ] ≥ c 2 . Here the probability corresponds to the dynamical system dominated by ( A,B ) .

Proof. Firstly, we consider a special case where d = 1 , and let A = [ -1] and ¯ A = [ -1 -δ ] . We show that when δ = 1 5 √ T , for the two dynamical systems ψ θ : dX t = AX t dt + dW t and ψ ¯ θ : dX t = ¯ AX t dt + dW t , any algorithm A that outputs according only to { X t 0 , X t 1 , ..., X t N } satisfies:

<!-- formula-not-decoded -->

̸

We note that this special case can be easily generalized to any dimension d , since we can consider A = -I d and ¯ A satisfies ¯ A 1 , 1 = A 1 , 1 -δ , and for any ( i, j ) = (1 , 1) , ¯ A i,j = A i,j . In this case the last d -1 dimension is independent of the first dimension, so it is essentially the same as the simplest one-dimensional case.

Denote X = { X t 0 , X t 1 , ..., X t N } and g ( X ) , ¯ g ( X ) be the probability density of ψ θ and ψ ¯ θ , respectively. For these two probability densities we have:

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next we show that ∣ ∣ ∣ ln ( g ( X ) ¯ g ( X ) )∣ ∣ ∣ is not large with high probability when X follows the probability density of g . Consider the following subsets of X : E 1 = { X ∣ ∣ ∣ ∣ ∣ ∑ N i =1 -ln( γ i ) + 1 2 ( γ 2 i -1) α 2 i ∣ ∣ ∣ ≤ 1 } .

E 2 = { X ∣ ∣ | ∑ N i =1 γ 2 i α i β i | ≤ 1 } and E 3 = { X ∣ ∣ 1 2 ∑ N i =1 γ 2 i β 2 i ≤ 1 } . When X lies in the intersection of these three sets, ∣ ∣ ∣ ln ( g ( X ) ¯ g ( X ) )∣ ∣ ∣ is guaranteed to be not very large.

Let P be the probability with respect to density g . We will explicitly show that P [ X ∈ E k ] ≥ 5 6 ( k = 1 , 2 , 3) .

Lower bound P [ X ∈ E 1 ] Firstly, we estimate ∑ N i =1 1 2 ( γ 2 i -1) -ln( γ i ) . We first prove the following inequality:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The left hand side of this inequality is because Γ t ≥ ¯ Γ t , due to the reason that e -2 s ≥ e -(2+2 δ ) s for all s ≥ 0 and when f ( x ) ≥ g ( x ) for any x ∈ I we have: ∫ x ∈ I f ( x ) dx ≥ ∫ x ∈ I g ( x ) dx . Now we consider the right hand side of the inequality.

Case 1: When t ≥ 1 , we directly use the fact that 1 -e -2 t ≤ 1 -e -(2+2 δ ) t and obtain γ i ≤ 1 + δ .

Case 2: When t ∈ (0 , 1] , it suffices to show that

<!-- formula-not-decoded -->

Let h ( t ) = (1 + δ )(1 -e -2 t ) -(1 + 2 δt )(1 -e -(2+2 δ ) t ) , then

<!-- formula-not-decoded -->

Where for the first inequality we use the relation that e -2 δt ≤ 1 1+2 δt . The second inequality is obtained by the relation that e -2 t ≥ 1 -2 t .

Now we bound 1 2 ( γ 2 i -1) -ln( γ i ) . We first show that

<!-- formula-not-decoded -->

Let x = γ 2 i -1 and we obtain 1 2 ( γ 2 i -1) -ln( γ i ) = 1 2 [ x -ln(1 + x )] , and the inequality is obtained directly since we have x ≥ ln(1 + x ) ≥ x -1 2 x 2 ( x ≥ 0) .

Then we can bound ∑ N i =1 1 2 ( γ 2 i -1) -ln( γ i ) as

<!-- formula-not-decoded -->

Now we bound ∑ N i =1 1 2 ( γ 2 i -1)( α 2 i -1) . Notice that this variable has zero mean, so we can bound its variance and then apply Markov inequality to obtain a high probability bound.

At first, consider the variance of α 2 i -1 , denoted as V ar ( α 2 i -1) . By noticing that α i ∼ N (0 , 1) , we can directly calculate that

<!-- formula-not-decoded -->

Since all the α i 's are independent, we have:

<!-- formula-not-decoded -->

By Markov inequality, we have:

<!-- formula-not-decoded -->

Finally, for the subset E 1 = { X ∣ ∣ ∣ ∣ ∣ ∑ N i =1 -ln( γ i ) + 1 2 ( γ 2 i -1) α 2 i ∣ ∣ ∣ ≤ 1 } , we have:

<!-- formula-not-decoded -->

Lower bound P [ X ∈ E 2 ] Since all the α i 's are independent, and α i is independent of { β 1 , ..., β i } and { γ 1 , ..., γ N } , we obtain that

<!-- formula-not-decoded -->

We have shown that γ 2 i ≤ 1 + 2 δ . Then for T ≥ 1 we have: γ 4 i ≤ (1 + 2 5 ) 2 ≤ 2 . Therefore, we obtain:

<!-- formula-not-decoded -->

Now we upper bound E [ β 2 i ] , where β i = √ 1 Γ( t i -t i -1 ) ( e -( t i -t i -1 ) -e -(1+ δ )( t i -t i -1 ) ) X t i -1 Firstly, we show that

<!-- formula-not-decoded -->

Again denote t = t i -t i -1 . By using Γ t = 1 2 (1 -e -2 t ) , it suffices to show that

<!-- formula-not-decoded -->

By multiplying e t on both sides, the inequality is equivalent to

<!-- formula-not-decoded -->

This is true since e -δt ≥ 1 -δt , and e 2 t ≥ 1 + 2 t , implying that

<!-- formula-not-decoded -->

With this result, we can upper bound 2 ∑ N i =1 E [ β 2 i ] by

<!-- formula-not-decoded -->

Finally, since X t ∼ N (0 , Γ( t )) , for all t ≥ 0 ,

<!-- formula-not-decoded -->

Therefore, we obtain

<!-- formula-not-decoded -->

Again by using Markov inequality, we obtain:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lower bound P [ X ∈ E 3 ] We have shown that γ 2 i ≤ 2 , ∀ i and ∑ N i =1 E [ β 2 i ] ≤ δ 2 T . Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Which follows that

And we also have

Now we come back to prove the theorem. With lower bounds of P [ X ∈ E 1 ] , P [ X ∈ E 2 ] , P [ X ∈ E 3 ] , we have

<!-- formula-not-decoded -->

With this bound, we have:

<!-- formula-not-decoded -->

Where the second inequality is because ∥ ϕ ( X ) -A ∥ + ∥ ϕ ( X ) -¯ A ∥ ≥ ∥ A -¯ A ∥ = 1 5 √ T so we cannot have both ∥ ϕ ( X ) -A ∥ ≤ 1 10 √ T and ∥ ϕ ( X ) -¯ A ∥ ≤ 1 10 √ T . The third inequality is because for any X ∈ E 1 ∩ E 2 ∩ E 3 , we have

<!-- formula-not-decoded -->

implying that ¯ g ( X ) ≥ 1 e 3 g ( X ) .

Therefore, we have:

<!-- formula-not-decoded -->

This means that for any algorithm, it cannot achieve 1 10 √ T estimation error with success probability 1 -1 4 e 3 for at least one of the systems controlled by ( A, 0) and ( ¯ A, 0) .

## B Regret Analysis

Having demonstrated the results of system identification for continuous-time linear systems, we leverage these findings to establish upper bounds on the regret for Algorithm 3. Elaborations on the details will be presented in the subsequent sections.

## B.1 Convergence of P and the Estimation Error of K

In this section we provide the following Lemma 15, along with its proof, which shows that ∥ P -P ∗ ∥ converges at the same speed as ∥ ˆ A -A ∥ + ∥ ˆ B -B ∥ .

Lemma 15. There exist constants ϵ 0 &gt; 0 and C 2 &gt; 0 such that as long as ∥ ˆ A -A ∥ , ∥ ˆ B -B ∥ ≤ ϵ for some 0 &lt; ϵ &lt; ϵ 0 , with P obtained from (5) we have:

<!-- formula-not-decoded -->

Recall that the optimal dynamic is K ∗ = -R -1 B T P ∗ with P ∗ obtained from equation (5). Now we consider the distance between it and the sub-optimal dynamic ¯ K = -R -1 B T P with P obtained from (5) with ( ˆ A, ˆ B ) . Denote ∆ A = ˆ A -A and ∆ B = ˆ B -B , along with ∥ ∆ A ∥ , ∥ ∆ B ∥ ≤ ϵ where ϵ ∈ [0 , ϵ 0 ] with some ϵ 0 determined later. We establish the proof by constructing a sequence of matrices ( P k ) k ≥ 0 , and we will prove that such sequence converges to the unique symmetric solution P satisfying

<!-- formula-not-decoded -->

At first we introduce a solution of a particular kind of matrix equation [19].

Lemma 16. Suppose A satisfies α ( A ) = max {ℜ ( λ i ) | λ i ∈ λ ( A ) } &lt; 0 . Q is a symmetric matrix. Consider such a function

<!-- formula-not-decoded -->

Then, the unique symmetric solution X of this equation can be expressed as:

<!-- formula-not-decoded -->

Now we consider the relation between P and P ∗ . The core is iteratively constructing a sequence of matrices P k such that P 0 = P ∗ and lim k → + ∞ P k = P . Such matrices follows the relation P k +1 = P k +∆ P k where ∆ P k converges rapidly. As for the starting case, consider the expansion

<!-- formula-not-decoded -->

Define

<!-- formula-not-decoded -->

We set ∆ P 0 be a solution of

<!-- formula-not-decoded -->

which satisfies that (see Lemma 16)

<!-- formula-not-decoded -->

This ∆ P 0 also satisfies

<!-- formula-not-decoded -->

An important thing is to guarantee that A 0 is stable, and | α ( A 0 ) | can not be too closed to zero. For any ϵ 1 ∈ (0 , 1) and C 1 = ∥ R -1 ∥∥ P ∗ ∥ +1+2 ∥ BR -1 ∥∥ P ∗ ∥ , as long as ϵ ≤ ϵ 1 , ∥ A 0 -( A -BR -1 B T P ∗ ) ∥ ≤ C 1 ϵ . Furthermore, there exists ϵ 2 &gt; 0 such that if ∥ X -( A -R -1 B T P ∗ ) ∥ ≤ ϵ 2 , then α ( X ) ≤ 1 2 α ( A -R -1 B T P ∗ ) (the work of [30] shows this result). We can further let this ϵ 2 satisfies that, as long as ∥ ∆ A ∥ , ∥ ∆ B ∥ , ∥ ∆ P ∥ ≤ ϵ 2 , we always have:

<!-- formula-not-decoded -->

Now we additionally set ϵ 1 satisfying ϵ 1 ≤ 1 2 C 1 ϵ 2 and ∥ R -1 ∥ ϵ 1 ≤ 1 , then for all ϵ ≤ ϵ 1 ,

<!-- formula-not-decoded -->

Denote P 1 = P 0 +∆ P 0 , C 2 = 2 -α ( A -BR -1 B T P ∗ ) ∥ P ∗ ∥ 2 (1 + ∥ BR -1 ∥ ) , and set some constant C 3 satisfying C 3 ≥ ∥ BR -1 B T ∥ +2 ∥ BR -1 ∥ + ∥ R -1 ∥ . We then inductively define P k +1 and ∆ P k ( k ≥ 1) . For defined ∆ P k -1 , we set P k = P k -1 +∆ P k -1 , which satisfies

<!-- formula-not-decoded -->

Then we denote A k = A +∆ A -( B +∆ B ) R -1 ( B +∆ B ) T P k , and set ∆ P k satisfying:

<!-- formula-not-decoded -->

By the hypothesis of ϵ 2 , as long as ∥ P k -P ∗ ∥ ≤ ϵ 2 , we have α ( A k ) ≥ 1 2 α ( A -BR -1 B T P ∗ ) . By using (29) we obtain that ∥ ∆ P k ∥ ≤ C 4 ∥ ∆ P k -1 ∥ 2 , where C 4 = 2 -α ( A -BR -1 B T P ∗ ) C 3 . Now if we define P k +1 = P k +∆ P k , P k +1 also satisfies:

<!-- formula-not-decoded -->

Then these sequences ∆ P k and P k are well defined, along with the relation that P k +1 = P k +∆ P k . Furthermore, when ∥ P k -P ∗ ∥ ≤ ϵ 2 , we have ∥ ∆ P k +1 ∥ ≤ C 4 ∥ ∆ P k ∥ 2 . Note that for the base case we have ∥ ∆ P 0 ∥ ≤ C 2 ϵ .

Finally, it remains to constrain ∥ P k -P ∗ ∥ . By choosing ϵ ≤ min( 1 2 C 2 C 4 , 1 2 C 2 ϵ 2 , 1) , we obtain ∥ ∆ P 0 ∥ ≤ C 2 ϵ . We can also see that if for all 0 ≤ k ≤ m , ∥ ∆ P k ∥ ≤ 2 -k C 2 ϵ , then ∥ P m -P ∗ ∥ ≤ 2(1 -2 -m +1 ) C 2 ϵ ≤ ϵ 2 so that ∥ ∆ P m +1 ∥ ≤ C 4 ∥ ∆ P m ∥ 2 ≤ 2 -m -1 C 2 ϵ . So by induction we see that ∥ ∆ P k ∥ ≤ 2 -k C 2 ϵ for any k .

On the other hand, since ∥ ∆ P k ∥ ≤ 2 -k ∥ ∆ P 0 ∥ , lim k → + ∞ P k = P ∞ exists, and such P ∞ is the unique symmetric solution of

<!-- formula-not-decoded -->

such that ( A +∆ A ) -( B +∆ B ) R -1 ( B +∆ B ) T P is stable (recall the stable margin in (30), which implies that ( A +∆ A ) -( B +∆ B ) R -1 ( B +∆ B ) T P ∞ is stable).

So P ∞ is exactly P , satisfying ∥ P -P ∗ ∥ ≤ 2 C 2 ϵ .

Therefore, we conclude that there exists some ϵ 0 &gt; 0 and constant C , both depending on A,B,K,d,p such that for any ϵ ∈ [0 , ϵ 0 ] , ∥ P -P ∗ ∥ ≤ Cϵ as long as ∥ ˆ A -A ∥ , ∥ ˆ B -B ∥ ≤ ϵ .

Then we apply our results for system identification to establish an upper bound for ∥ ¯ K -K ∗ ∥ .

Based on Lemma 15, fix constant ϵ 1 &gt; 0 and constant C 1 ≥ 0 so that we have ∥ P -P ∗ ∥ ≤ C 1 ( ∥ ˆ A -A ∥ + ∥ ˆ B -B ∥ ) whenever ∥ ˆ A -A ∥ + ∥ ˆ B -B ∥ ≤ ϵ 1

We set C 2 ≥ 1 be two times the constant C in Lemma 9, and obtain that, when log 2 (1 /δ ) ≤ T 1 / 2 C 2 and T 1 / 2 ≥ C 2 ∥ X 0 ∥ 2 2 , we have:

<!-- formula-not-decoded -->

Then, for log(1 /δ ) ≤ min { Tϵ 2 1 4 C 2 2 , T 1 / 4 C 1 / 2 2 } ≤ T 1 / 4 ϵ 2 1 4 C 2 2 , we have:

<!-- formula-not-decoded -->

Finally, since ¯ K = -R -1 ( ˆ B ) T P , K ∗ = -R -1 B T P ∗ , we have:

<!-- formula-not-decoded -->

Wecan reset C 1 such that ∥ ¯ K -K ∗ ∥ ≤ C 1 ( ∥ ˆ A -A ∥ + ∥ ˆ B -B ∥ ) whenever ∥ ˆ A -A ∥ + ∥ ˆ B -B ∥ ≤ ϵ 1 , and combine this with (31), we have: for any log(1 /δ ) ≤ T 1 / 4 ϵ 2 1 4 C 2 2

<!-- formula-not-decoded -->

With this probability bound on ∥ ¯ K -K ∗ ∥ , we can further upper bound the regret, shown in the following part.

## B.2 Key Lemmas

We first upper bound the radius of a single trajectory with stable controller, for which we introduce and provide a proof for the following lemma:

Lemma 17. Consider the continuous system dX t = AX t dt + dW t such that α ( A ) &lt; 0 where α ( A ) is the largest real component of A and W is a standard Brownian noise. Then, w.p. at least 1 -δ :

<!-- formula-not-decoded -->

Then we concentrate on how the error ∥ P -P ∗ ∥ will influence the regret during the exploitation phase. For a dynamic U with α ( A + BU ) &lt; 0 , we define a cost function:

<!-- formula-not-decoded -->

The convergence rate of this cost function is stated in the following lemma:

Lemma 18. Let U ∗ minimize cost ( U ) . Then, there exists ϵ 0 ≥ 0 such that for any ∥ ∆ U ∥ = 1 and ϵ ∈ [0 , ϵ 0 ] , we have:

<!-- formula-not-decoded -->

The above result shows the average cost per unit time when applying fixed controller for infinite time.

Then we further consider the case when the running time is finite. We derive the following lemma:

Lemma 19. Let U ∗ follows the same definition as in Lemma 18. Then, for some ϵ &gt; 0 , there exist constants C 2 and C 3 (independent of U ) such that for all T &gt; 0 and any U such that ∥ U -U ∗ ∥ ≤ ϵ ,

<!-- formula-not-decoded -->

Here J T is the expected cost of the policy that takes action by U t = UX t ( t ∈ [0 , T ]) , with initial state X 0 = x .

With this lemma, by definition of U ∗ , we actually have U ∗ = K ∗ , where K ∗ = -R -1 B T P ∗ and P ∗ is the solution of (4). Since such C 2 , C 3 also satisfy:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so it follows that

## B.3 Proof of Lemma 17

We first upper bound the radius of a single trajectory with stable controller, for which we introduce and provide a proof for the following lemma:

Lemma 17. Consider the continuous system dX t = AX t dt + dW t such that α ( A ) &lt; 0 where α ( A ) is the largest real component of A and W is a standard Brownian noise. Then, w.p. at least 1 -δ :

<!-- formula-not-decoded -->

Proof. The trajectory X t with differential equation dX t = AX t + dW t can be derived as

<!-- formula-not-decoded -->

Lemma 7 tells that when A is stable, ∥ ∥ e At X 0 ∥ ∥ 2 ≤ e α ( A ) t ∥ X 0 ∥ 2 . So it suffices to show that

<!-- formula-not-decoded -->

Let T = T 0 h with T 0 be an integer. We first consider the set of points { X kh } . Denote w k := ∫ kh t =0 e A ( kh -t ) dW t , then w k ∼ N (0 , Σ k ) with Σ k = ∫ kh t =0 e At e A T t dt . This Σ h also satisfies

<!-- formula-not-decoded -->

Which follows that sup 0 ≤ k ≤ T 0 ∥ w k ∥ 2 ≤ 2 √ d | α ( A ) | log((1 + T 0 ) /δ ) , w.p. at least 1 -δ .

Next we consider any X kh + t with t ∈ [0 , h ] . Bounding such terms requires the Doob's martingale inequality [11], stated as in Lemma 20. We denote x k t = ∫ t s =0 e A ( t -s ) dW kh + s ds with corresponding filtration F t . We also define Z k t := e λ ∥ e -At x k t ∥ 2 2 with λ ≥ 0 . Then Z k t is a submartingale under the filtration F t , since for any t ≥ s ,

<!-- formula-not-decoded -->

Where we notice that E [ ∥ ∥ ∥ e -As x k s + ∫ t t 1 = s e -At 1 dW kh + t 1 ∥ ∥ ∥ 2 2 ∣ ∣ x k s ] ≥ ∥ ∥ e -As x k s ∥ ∥ 2 2 , and apply Jensen's inequality on the non-decreasing convex function f ( x ) = e λx to obtain the above inequality. Now we apply Lemma 20 and get

<!-- formula-not-decoded -->

We next estimate E ( Z k h ) . Since e -Ah x k h = ∫ h t =0 e -At dW kh + t , we obtain that e -Ah x k h ∼ N (0 , ¯ Σ) , where

<!-- formula-not-decoded -->

By setting λ = 1 4 ∥ ¯ Σ ∥ , it can be computed that

<!-- formula-not-decoded -->

where the last inequality is because I d -2 λ ¯ Σ ⪰ 1 2 I d .

We combine this result with (34) and obtain:

<!-- formula-not-decoded -->

Finally, since X kh + t = e A ( kh + t ) X 0 + e At w k + x k t , it follows that

<!-- formula-not-decoded -->

By applying union bound on ∥ w k ∥ 2 and ∥ ∥ x k t ∥ ∥ 2 we finally obtain Lemma 17.

Lemma20 (Doob's martingale inequality) . Let X 1 , . . . , X n be a discrete-time submartingale relative to a filtration F 1 , . . . , F n of the underlying probability space, which is to say:

<!-- formula-not-decoded -->

The submartingale inequality says that

<!-- formula-not-decoded -->

for any positive number C .

Moreover, let X t be a submartingale indexed by an interval [0 , T] of real numbers, relative to a filtration F t of the underlying probability space, which is to say:

<!-- formula-not-decoded -->

for all s &lt; t . The submartingale inequality says that if the sample paths of the martingale are almost-surely right-continuous, then

<!-- formula-not-decoded -->

for any positive number C .

## B.4 Proof of Lemma 18

In this section, we proof Lemma 18 which refers to the convergence rate of the cost function:

Lemma 18. Let U ∗ minimize cost ( U ) . Then, there exists ϵ 0 ≥ 0 such that for any ∥ ∆ U ∥ = 1 and ϵ ∈ [0 , ϵ 0 ] , we have:

<!-- formula-not-decoded -->

Proof. For any ∥ ∆ U ∥ = 1 and ϵ &gt; 0 , consider U = U ∗ + ϵ ∆ U , we show that as ϵ → 0 , there exists V ∈ R d such that tr( V ) = 0 , and

<!-- formula-not-decoded -->

Let D ( ϵ, t ) = e ( A + B ( U ∗ + ϵ ∆ U )) t -e ( A + BU ∗ ) t . The most important intuition is that D ( ϵ, t ) can be represented by the form of D ( ϵ, t ) = ϵD 1 ( t ) + ϵ 2 D 2 ( ϵ, t ) , where D 1 ( t ) does not depend on ϵ , and the residual D 2 ( ϵ, t ) can be well bounded. Now we find such D 1 ( t ) and upper bound ∥ D 2 ( ϵ, t ) ∥ . For t ≤ t 0 = 1 max {∥ A + BU ∗ ∥ , ∥ B ∥} and ϵ &lt; 1 , the Taylor expansion of e ( A + B ( U ∗ + ϵ ∆ U )) t can be represented as follows:

<!-- formula-not-decoded -->

where D 1 ( ϵ, k ) is the residual of ( A + BU + ϵB ∆ U ) k -( A + BU ) k with order at least ϵ 2 . This sequence of matrices are expressed and bounded as follows.

<!-- formula-not-decoded -->

Thus we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Define E ( t ) and E 1 ( ϵ, t ) as follows: for 0 ≤ t ≤ t 0 , let

<!-- formula-not-decoded -->

and for t ∈ [ 1 2 t 0 , t 0 ] , l ≥ 1 , we inductively define E (2 l t ) and E 1 (2 l t ) as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then we have the relation that e ( A + BU ∗ + B ∆ U ) t -e ( A + BU ∗ ) t = ϵE ( t ) + ϵ 2 E 1 ( ϵ, t ) .

Now we upper bound ∥ E ( t ) ∥ and ∥ E 1 ( ϵ, t ) ∥ . When t ≤ t 0 :

<!-- formula-not-decoded -->

For t ≥ t 0 , let t = 2 l 1 t 1 , with l 1 be an integer and t 1 ∈ ( 1 2 t 0 , t 0 ] , then

<!-- formula-not-decoded -->

where the last inequality is because for any x, a &gt; 0 , xe -ax ≤ 1 ae , and thus for any t ≥ 0 , ∥ E ( t ) ∥ ≤ C = 4 -α ( A + BU ∗ ) t 0 .

When t ≥ 2 -α ( A + BU ∗ ) , we additionally have

<!-- formula-not-decoded -->

Now we consider E 1 ( ϵ, t ) . When t ≤ t 0 ,

<!-- formula-not-decoded -->

When t &gt; t 0 , with t = 2 l t 1 and t 1 ∈ ( 1 2 t 0 , t 0 ] , we obtain:

<!-- formula-not-decoded -->

Now, we show that ∥ ∥ E 1 ( ϵ, 2 l t 1 ) ∥ ∥ converges exponentially eventually. The proof consists of two parts: first, for t which is not too large, ∥ E 1 ( ϵ, t ) ∥ can be bounded uniformly over all possible ∆ U and any constrained ϵ . Then, for larger t we can utilize the construction of ∥ E 1 ( ϵ, t ) ∥ to estimate its convergence speed.

Let ϵ ≤ -α ( A + BU ∗ ) t 0 (64 C ) 2 , l 0 = 1 + ⌊ log 2 4 -α ( A + BU ∗ ) t 0 ⌋ . We first inductively show that for any l ≤ l 0 , ∥ ∥ E 1 ( ϵ, 2 l t 1 ) ∥ ∥ ≤ (2 l +3 -4) C 2 . The base case where l = 0 is certainly true. Suppose we already have ∥ ∥ E 1 ( ϵ, 2 l -1 t 1 ) ∥ ∥ ≤ (2 l +2 -4) C 2 . Then for the case of l , we obtain:

<!-- formula-not-decoded -->

where for the first inequality we use the inductive hypothesis that

<!-- formula-not-decoded -->

along with facts that ∥ ∥ E (2 l -1 t 1 ) ∥ ∥ ≤ C and 2 e α ( A + BU ∗ )2 l -1 t 1 ≤ 2 . Specifically, we have ∥ ∥ E 1 ( ϵ, 2 l 0 t 1 ) ∥ ∥ ≤ 64 C 2 -α ( A + BU ∗ ) t 0 .

Now, we consider l &gt; l 0 . We first show that for all such l , ∥ ∥ E 1 ( ϵ, 2 l t 1 ) ∥ ∥ ≤ 64 C 2 -α ( A + BU ∗ ) t 0 . Since 2 l -1 t 1 ≥ 2 l 0 -1 t 0 ≥ 2 -α ( A + BU ∗ ) , we have 2 e α ( A + BU ∗ )2 l t 1 ≤ 2 e -2 , and thus

<!-- formula-not-decoded -->

which holds for all l ≥ l 0 with induction on l . Now we reuse the above expression and obtain that ∥ l ∥

<!-- formula-not-decoded -->

Let l ∗ be the smaller integer greater than l 0 +1 which satisfies:

<!-- formula-not-decoded -->

Then by using the relation that 2 ϵ 2 ∥ ∥ E 1 ( ϵ, 2 l -1 t 1 ) ∥ ∥ 2 ≤ 2 ϵ 2 ( 64 C 2 -α ( A + BU ∗ ) t 0 ) 2 ≤ 1 4 , we have:

<!-- formula-not-decoded -->

Now we inductively show that for all k ≥ 0 ,

<!-- formula-not-decoded -->

By using the hypothesis for k and 2 ϵ 2 ≤ 1 4 , we obtain:

<!-- formula-not-decoded -->

leading to the claim. This means there exist some constants C 1 , c 1 &gt; 0 depending on α ( A + BU ∗ ) such that for all t ≥ 0 , ∥ E 1 ( ϵ, t ) ∥ ≤ C 1 e -c 1 t .

Finally, we consider ∫ t ≥ 0 e ( A + BU ) T t ( Q + U T RU ) e ( A + BU ) t dt . Since e ( A + BU ∗ + ϵ ∆ U ) t = e ( A + BU ∗ ) t + ϵE ( t ) + ϵ 2 E 1 ( ϵ, t ) , with ∥ E ( t ) ∥ ≤ 8 -α ( A + BU ∗ ) t 0 e 1 4 α ( A + BU ∗ ) t and bounded E 1 ( ϵ, t ) , we obtain:

<!-- formula-not-decoded -->

Where the last term O ( ϵ 2 ) contains any terms with order at least ϵ 2 , whose norm is at most C 2 ϵ 2 for any ϵ ∈ [0 , ϵ 0 ) and ∥ ∆ U ∥ = 1 , where the constant C 2 depends on A,B,α ( A + BU ∗ ) and ϵ 0 is some small constant.

For any ∥ ∆ U ∥ = 1 , define V by

<!-- formula-not-decoded -->

then cost ( U ) = cost ( U ∗ ) + ϵ tr( V ) + O ( ϵ 2 ) .

Since U ∗ minimizes cost ( U ) , tr( V ) = lim ϵ → 0 ϵ -1 ( cost ( U ∗ + ϵ ∆ U ) -cost ( U ∗ )) = 0 . Therefore, we obtain that cost ( U ) = cost ( U ∗ ) + O ( ϵ 2 ) .

## B.5 Proof of Lemma 19

In this section, we proof Lemma 19.

Lemma 19. Let U ∗ follows the same definition as in Lemma 18. Then, for some ϵ &gt; 0 , there exist constants C 2 and C 3 (independent of U ) such that for all T &gt; 0 and any U such that ∥ U -U ∗ ∥ ≤ ϵ ,

<!-- formula-not-decoded -->

Here J T is the expected cost of the policy that takes action by U t = UX t ( t ∈ [0 , T ]) , with initial state X 0 = x .

Proof. By definition of J T , we have:

<!-- formula-not-decoded -->

Since the state transits according to dX t = AX t dt + BUX t dt + dW t , we can derive the expression of X t by X t = e ( A + BU ) t X 0 + ∫ t s =0 e ( A + BU )( t -s ) dW s . Then by utilizing this expression we obtain:

<!-- formula-not-decoded -->

Then, the expected cost on a trajectory lasting for time T can be computed as:

<!-- formula-not-decoded -->

Here the first term satisfies

<!-- formula-not-decoded -->

and the latter two integral terms can be bounded as follows.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, for C 2 ≥ -1 2 α ( A + BU ) and C 3 ≥ d ∥ Q + U T RU ∥ 2 α 2 ( A + BU ) , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B.6 Proof of Lemma 21

Finally, we prove Lemma 21. In this part we suppose T ≥ T 0 , where T 0 ≥ 1 is a constant depending on some hidden constants and ∥ X 0 ∥ 2 2 .

Lemma 21. regret Let U t be the action applied as in Algorithm 3. Then there exists a constant C ∈ poly ( κ, M, µ -1 , | α ( A + BK ) | -1 , | α ( A + BK ∗ ) | -1 ) such that for sufficiently large T :

<!-- formula-not-decoded -->

Define the following events where the stabilizing controller K might ever be applied during the exploitation phase. Let E 1 = { ∥ X √ T ∥ 2 ≥ 1 2 T 1 / 5 } , E 2 = { ∥ X t ∥ 2 ≥ T 1 / 5 for some t ∈ [ √ T, T ] } , and E 3 = { ∥ ¯ K -K ∗ ∥ ≤ ϵ 3 } , where ϵ 3 &gt; 0 depends on the constant ϵ 0 in Lemma 18, which will be determined later. In this part, we again let C 1 , C 2 be the same as in (32), and denote C 3 be the constant C 1 in Lemma 18. We firstly analyze these three events.

Upper Bound of P [ E 1 ] By Lemma 17, we can find some constant C 0 depending on ∥ A ∥ , ∥ B ∥ , ∥ K ∥ , d, p, h such that

<!-- formula-not-decoded -->

This is because we have the recursive function of { X kh } that

<!-- formula-not-decoded -->

from which we can derive that

X kh

<!-- formula-not-decoded -->

Then, for sufficiently large T , ∥ ∥ ∥ e ( A + BK ) √ T X 0 ∥ ∥ ∥ 2 can be bounded by 1 , and from the proof in Lemma 17 we can apply similar idea to upper bound the norm of the last two terms. So we can obtain the probability bound on ∥ X √ T ∥ 2 .

By setting δ = 2 T · e -T 1 / 5 4 C 2 0 , we obtain that P [ E 1 ] ≤ 2 T · e -T 1 / 5 4 C 2 0 .

Upper Bound of P [ E C 3 ] By (31), we obtain that, for ϵ 3 ≤ C 1 ϵ 1 T 1 / 8 4 C 2 2 , we have:

<!-- formula-not-decoded -->

and we also have: P [ E C 3 ] ≤ e -T 1 / 2 ϵ 2 3 4 C 2 1 C 2 2 .

By setting ϵ 3 = C 1 ϵ 1 T 1 / 8 4 C 2 2 , we have: P [ E C 3 ] ≤ e -T 1 / 4 ϵ 2 1 64 C 2 2 .

Upper Bound of P [ E 2 ] Consider any ∥ X √ T ∥ 2 ≤ 1 2 T 1 / 5 and any ∥ ¯ K -K ∗ ∥ ≤ ϵ 3 , we claim that P [ E 2 ∣ ∣ X √ T , ¯ K ] ≤ e -Ω( T 1 / 5 ) .

As what have discussed in Lemma 15 (see the discussion about stable margin near (30)), such ¯ K satisfies α ( A + B ¯ K ) ≤ 1 2 α ( A + BK ∗ ) .

Then by Lemma 17 we can derive that, for some constant C ,

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now we come to estimate the expected cost of Algorithm 3, as well as bound the regret. We separately calculate the cost during the two phases.

Cost During Exploration Phase For ( k +1) h ≤ √ T and t ∈ [0 , h ] , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the inequality is because u k is independent of X kh and W s ( s ∈ [ kh, kh + t ]) .

For the first term, we first upper bound E [ ∥ X kh + t ∥ 2 2 ] .

Denote w k,t = ∫ kh + t s = kh e ( A + BK )( kh + t -s ) dW s + ( ∫ t s =0 e ( A + BK ) s ds ) u k , which is a Gaussian variable with zero mean and is independent of X kh . Then

<!-- formula-not-decoded -->

For E [ ∥ X kh ∥ 2 2 ] , since

<!-- formula-not-decoded -->

We have

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

Where C 3 is a constant depending on α ( A + BK ) and d .

For the second term E [ ∥ w k,t ∥ 2 2 ] , can follow the same process of the above bound and obtain E [ ∥ w k,t ∥ 2 2 ] ≤ C 3 . Therefore, E [ ∥ X kh + t ∥ 2 2 ] ≤ 2 C 3 .

Now we can upper bound E [ X T kh + t QX kh + t + U T kh + t RU kh + t ] . We have

<!-- formula-not-decoded -->

We also have E [ u T k Ru k ] = tr ( R ) and the following inequality:

<!-- formula-not-decoded -->

We can conclude that there exists constant C 4 depending on A,B,K,Q,R,d,p,h such that

<!-- formula-not-decoded -->

Therefore, the cost during exploration phase can be bounded as

<!-- formula-not-decoded -->

Cost During Exploitation Phase We first concentrate on E 2 , which is the hardest event for the analysis of the cost. Consider the following two cases:

Case 1: ∥ X √ T ∥ 2 ≥ T 1 / 5 . In this case, the action is applied by U t = KX t , t ∈ [ √ T, T ] .

Case 2: ∥ X √ T ∥ 2 &lt; T 1 / 5 . In this case, the trajectory is unfortunately controlled by a bad controller, and suffers from large risk of diverging.

We first consider Case 1 . By ( ?? ) we can derive that

<!-- formula-not-decoded -->

Then, we have:

<!-- formula-not-decoded -->

Therefore, for some constants C 5 , C 6 , we have:

<!-- formula-not-decoded -->

Now we consider Case 2 . Let t 0 = inf t {∥ X t ∥ 2 ≥ T 1 / 5 , t ≥ √ T } , then ∥ X t 0 ∥ 2 = T 1 / 5 almost surely.

For t ∈ [ √ T, t 0 ] , since we always have

<!-- formula-not-decoded -->

the cost satisfies:

<!-- formula-not-decoded -->

Where C 7 is a constant depending on B,R,K,P .

For t ∈ [ t 0 , T ] , the trajectory X t satisfies

<!-- formula-not-decoded -->

Similar to the analysis for Case 1 , we have:

<!-- formula-not-decoded -->

Combining them, we can conclude that for some constant C 8 , no matter whether E 2 happens, we always have:

<!-- formula-not-decoded -->

Now we establish the upper bound for the regret. Since

<!-- formula-not-decoded -->

For the first term, we can upper bound it by

<!-- formula-not-decoded -->

Here the first inequality is because 1 E C 1 ∩E 3 = 1 E C 1 ∩E C 2 ∩E 3 +1 E C 1 ∩E 2 ∩E 3 and 1 E C 1 ∩E 2 ∩E 3 ≤ 1 E C 1 ∩E 3 . For the second inequality, the first term is because we can assume a situation that we do not change the dynamic when E 2 happens, and that will not make the expectation smaller. By applying the results of Lemma 18 and Lemma 19 we can get this term, where the constant C 9 is related to constants in these two lemmas. The last inequality is obtained from these two lemmas and the definitions of E 1 , E 2 , E 3 .

As for E [ ∥ ¯ K -K ∗ ∥ 2 · 1 E 3 ] , we use the bound that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For E [ 1 E C 1 ∩E 2 ] , we directly have E [ 1 E C 1 ∩E 2 ] ≤ P [ E 2 ] ≤ e -Ω( T 1 / 5 ) . Combining these results and Lemma 19 we obtain that for some constant C ,

<!-- formula-not-decoded -->

For the second term E [ ∫ T t = √ T ( X T t QX t + U T t RU t ) dt · 1 E 1 ] , given any X √ T , we always have

<!-- formula-not-decoded -->

So we can upper bound E [ ∫ T t = √ T ( X T t QX t + U T t RU t ) dt · 1 E 1 ] by

<!-- formula-not-decoded -->

where for the last inequality we apply the upper bound of P [ E 1 ] shown before.

For E [ ∥ X √ T ∥ 2 2 · 1 E 1 ] , we can apply Lemma 17 and obtain that for some constant c &gt; 0 , for any x ≥ 1 2 T 1 / 5 , we have

<!-- formula-not-decoded -->

and compute that

Thus we have:

<!-- formula-not-decoded -->

Therefore, we have E [ ∫ T t = √ T ( X T t QX t + U T t RU t ) dt · 1 E 1 ] ≤ O (1)

Finally, for the last term E [ ∫ T t = √ T ( X T t QX t + U T t RU t ) dt · 1 E C 1 ∩E C 3 ] , when condition on any ∥ X √ T ∥ 2 ≤ 1 2 T 1 / 5 , estimator ( ˆ A, ˆ B ) and X t 0 , where t 0 = inf t ≥ √ T ( ∥ X t ∥ 2 ≥ T 1 / 5 ) , we still have:

<!-- formula-not-decoded -->

So we can upper bound it by

<!-- formula-not-decoded -->

Combining them we finally obtain Lemma 21.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We clarify our contributions and basic problem setups in both abstract and introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations of the work in Section 6.

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

Justification: Assumptions can be found just near the main theorems. The complete proof is contained in our Appendix.

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

Justification: We disclose the experiment details in Section 5.3.

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

## Answer: [No]

Justification: Our code is very simple, just use a simulation experiment of 3*3 matrix. Our main contribution is the theoretical analysis.

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

Justification: We specify all the training and test details in Section 5.3.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We compute the average regret in our experiment.

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

Answer: [No]

Justification: Our code is very simple, which can run on CPU of a computer.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We conform with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: : Our work is about the theory on online control and system identification, which does not seem to have evident societal impacts.

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

## Answer: [No]

Justification: The LLM is used only for editing the paper of grammar mistake.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.