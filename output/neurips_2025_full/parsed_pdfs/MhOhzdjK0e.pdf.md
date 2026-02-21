## Deep Learning for Continuous-Time Stochastic Control with Jumps

Patrick Cheridito Department of Mathematics

Jean-Loup Dupret ∗ Department of Mathematics

ETH Zurich, Switzerland

ETH Zurich, Switzerland

Donatien Hainaut LIDAM-ISBA UCLouvain, Belgium

{patrickc,jdupret}@ethz.ch, {donatien.hainaut}@uclouvain.be

## Abstract

In this paper, we introduce a model-based deep-learning approach to solve finite-horizon continuous-time stochastic control problems with jumps. We iteratively train two neural networks: one to represent the optimal policy and the other to approximate the value function. Leveraging a continuous-time version of the dynamic programming principle, we derive two different training objectives based on the Hamilton-Jacobi-Bellman equation, ensuring that the networks capture the underlying stochastic dynamics. Empirical evaluations on different problems illustrate the accuracy and scalability of our approach, demonstrating its effectiveness in solving complex high-dimensional stochastic control tasks. Code is available at https://github.com/jdupret97/ Deep-Learning-for-CT-Stochastic-Control-with-Jumps .

## 1 Introduction

A large class of dynamic decision-making problems under uncertainty can be modeled as continuoustime stochastic control problems. In this paper, we introduce two neural network-based numerical algorithms for such problems in high dimensions with finite time horizon and jumps. More precisely, we consider control problems of the form

<!-- formula-not-decoded -->

for a finite horizon T &gt; 0 , where the supremum is over predictable control processes α = ( α t ) 0 ≤ t ≤ T taking values in a subset A ⊆ R m . The controlled process X α evolves in a subset D ⊆ R d according to

<!-- formula-not-decoded -->

for an initial condition x ∈ D , an n -dimensional Brownian motion W and a controlled random measure N α on E × R + , with E = R l \ { 0 } , and suitable functions β : [0 , T ] × D × A → R d , σ : [0 , T ] × D × A → R d × n and γ : [0 , T ] × D × E × A → R d . The functions f : [0 , T ] × D × A → R and F : D → R model the running and final rewards, respectively. We assume the controlled random measure is given by N α ( B × [0 , t ]) = ∑ M α t j =1 1 { Z j ∈ B } for measurable subsets B ⊆ E , where M α is a Cox process with a stochastic intensity of the form λ ( t, X α t -, α t ) and Z 1 , Z 2 , . . . are i.i.d. E -valued random vectors such that, conditionally on α , the random elements W,M α and Z 1 , Z 2 , . . . are independent. Our goal is to find an optimal control α ∗ and the corresponding value of problem

∗ Corresponding author.

(1). In view of the Markovian nature of the dynamics (2), we work with feedback controls of the form α t = α ( t, X α t -) for measurable functions α : [0 , T ) × D → A and consider the value function V : [0 , T ] × D → R given by

<!-- formula-not-decoded -->

Under suitable assumptions 2 , V is the unique solution of the following Hamilton-Jacobi-Bellman (HJB) equation

<!-- formula-not-decoded -->

for the Hamiltonian H : [0 , T ) × D ×V × A → R given 3 by

<!-- formula-not-decoded -->

where ∇ x V , ∇ 2 x V denote the gradient and Hessian of V with respect to x . Our approach consists in iteratively training two neural networks to approximate the value function V and an optimal control α ∗ attaining V . It has the following features:

- It yields accurate results for high-dimensional continuous-time stochastic control problems in cases where the underlying system dynamics are known.
- It can effectively handle a combination of diffusive noise and random jumps with controlled intensities.
- It can handle general situations where the optimal control is not available in closed form but has to be learned together with the value function.
- It approximates both the value function and optimal control at any point ( t, x ) ∈ [0 , T ) × D .

While methods like finite differences, finite elements and spectral methods work well for solving partial (integro-)differential equations P(I)DEs in low dimensions, they suffer from the curse of dimensionality and, as a consequence, become infeasible in high dimensions. Recently, different deep learning based approaches for solving high-dimensional PDEs have been proposed (Raissi et al., 2017, 2019; Han et al., 2017, 2018; Sirignano &amp; Spiliopoulos, 2018; Berg &amp; Nyström, 2018; Beck et al., 2021; Lu et al., 2021; Bruna et al., 2024). They can directly be used to solve continuous-time stochastic control problems that admit an explicit solution for the optimal control in terms of the value function since in this case, the expression for the optimal control can be plugged into the HJB equation, which then reduces to a parabolic PDE. On the other hand, if the optimal control is not available in closed form, it cannot be plugged into the HJB equation, but instead, has to be approximated numerically while at the same time solving a parabolic PDE. Such implicit optimal control problems can no longer be solved directly with one of the deep learning methods mentioned above but require a specifically designed iterative approximation procedure. For low-dimensional problems with nonexplicit optimal controls, a standard approach from the reinforcement learning (RL) literature is to use generalized policy iteration (GPI), a class of iterative algorithms that simultaneously approximate the value function and optimal control (Jacka &amp; Mijatovi´ c, 2017; Sutton &amp; Barto, 2018). Particularly popular are actor-critic methods going back to Werbos (1992), which have a separate memory structure to represent the optimal control independently of the value function. However, classical GPI schemes become impractical in high dimensions as the PDE and optimization problem both have to be discretized and solved for every point in a finite grid. This raises the need for meshfree methods to solve implicit continuous-time stochastic control problems in high dimensions with continuous action space. Several local approaches based on a time-discretization of (2) have been explored by e.g. Han &amp;E(2016), Nüsken &amp; Richter (2021), Huré et al. (2021), Bachouch et al. (2022), Ji et al. (2022), Li et al. (2024), Domingo-Enrich et al. (2024a), Domingo-Enrich et al. (2024b). Alternatively, (deep) RL methods can be used, for instance, Q-learning type algorithms such as DQN (Mnih et al., 2013) or C51 (Bellemare et al., 2017), see also Wang et al. (2020), Jia &amp; Zhou (2023), Gao et al. (2024), Szpruch et al. (2024); or actor critic approaches such as DDPG (Lillicrap et al., 2019), SAC (Haarnoja

2 see e.g. Soner (1988)

3 By V we denote the set of all functions in C 1 , 2 ([0 , T ) × R d ) such that the expectation in (4) is finite for all t, x and a .

et al., 2018), A2C/A3C (Mnih et al., 2016), PPO (Schulman et al., 2017) or TRPO (Schulman et al., 2015). However, these RL algorithms are model-free, that is, they do not explicitly take into account the underlying dynamics (2) of the control problem (1) but instead, solely rely on sampling from the environment. As a result, they are less accurate in cases where the system dynamics are known (see Figure 3 below). On the other hand, the difficulty of solving high-dimensional PDEs is further exacerbated if the system dynamics includes stochastic jumps since in this case, the HJB equation (3) requires the computation of the jump expectations E V ( t, x + γ ( t, x, Z 1 , a )) for every space-time point ( t, x ) sampled from the domain.

In this paper, we introduce a deep model-based approach for stochastic control problems with jumps that takes the system dynamics (2) into account by leveraging the HJB equation (3). This removes the need to simulate the underlying jump-diffusion (2) and, as a result, avoids discretization errors. Our approach combines GPI with a PIDE solving method. It approximates the value function and optimal control in an actor-critic fashion with two neural networks trained iteratively on sampled data from the space-time domain. As such, it has the advantage that it provides global approximations of the value function and optimal control available for all space-time points, which can be evaluated rapidly in online applications. We develop two related algorithms. The first one, GPI-PINN, approximates the value function by training a neural network to minimize the residuals of the HJB equation (3), following a physics-inspired neural network (PINN) approach (Raissi et al., 2017, 2019) while leveraging Proposition 3.1 below to avoid the direct computation of the gradient ∇ x V and Hessian ∇ 2 x V in the Hamiltonian (4). GPI-PINN can be viewed as an extension of the method proposed by Duarte et al. (2024) adapted to a finite-horizon setup with time-dependence and a terminal condition in the HJB equation, leading to time-dependent optimal control strategies and value function, see also Dupret &amp; Hainaut (2024). It works well in high dimensions for control problems without jumps in the underlying dynamics (2) ( γ = 0) , but becomes inefficient in the presence of jumps as it requires the computation of the jump-expectation E V ( t, x + γ ( t, x, Z 1 , a )) at numerous sample points ( t, x ) in every iteration of the algorithm. To address this, our second algorithm, GPI-CBU, relies on a continuous-time Bellman updating rule to approximate the value function, thereby circumventing the computation of gradients, Hessians and jump-expectations altogether. This makes it highly efficient for high-dimensional stochastic control problems with jumps, even when the control is not available in closed form.

We illustrate the accuracy and scalability of our approach in different numerical examples and provide comparisons with popular RL and deep-learning control methods. Proofs of theoretical results and additional numerical experiments are given in the Appendix.

## 2 General approach

Let α : [0 , T ) × D → A be a feedback control such that equation (2) has a unique solution X α and consider the corresponding value function

<!-- formula-not-decoded -->

Under appropriate assumptions, one obtains the following two results from standard arguments 4 .

Theorem 2.1 (Feynman-Kac Formula) . V α satisfies the PIDE

<!-- formula-not-decoded -->

Theorem 2.2 (Verification Theorem) . Let v ∈ V ∩ C ([0 , T ] × D ) be a solution of the HJB equation (3) such that there exists a measurable mapping ˆ α : [0 , T ) × D → A satisfying

<!-- formula-not-decoded -->

and the controlled jump-diffusion equation (2) admits a unique solution for each initial condition x ∈ D . Then v = V and ˆ α is an optimal control.

Based on Theorems 2.1 and 2.2, we iteratively approximate the value function V and optimal control α ∗ with neural networks 5 V θ : [0 , T ] × D → R and α φ : [0 , T ] × D → A . For given α φ , we train V θ

4 see e.g. the arguments in the proofs of Theorems 1.3.1 and 2.2.4 in Bouchard (2021).

5 Using a C 2 -activation function in the network V θ ensures that it belongs to C 2 ([0 , T ] × R d ) .

so as to solve the controlled HJB equation

<!-- formula-not-decoded -->

while for given V θ , α φ is trained with the goal to maximize the Hamiltonian H ( t, x, V θ , α φ ( t, x )) .

In the following, we introduce two different training objectives for updating the value network V θ , leading respectively to the algorithms GPI-PINN and GPI-CBU. GPI-PINN uses a PINN-type loss together with a trick adapted from Duarte et al. (2024) to bypass the explicit computation of gradients and Hessians, whereas GPI-CBU relies on a continuous-time Bellman updating rule with an expectation-free version of the Hamiltonian, thereby also avoiding the computation of the jump-expectations in (4).

## 3 GPI-PINN

GPI-PINN, described in Algorithm 1 below, relies on a PINN approach to minimize the residuals of the controlled HJB equation (5) in the value function approximation step. To avoid explicit computations of the gradient ∇ x V θ ( t, x ) and Hessian ∇ 2 x V θ ( t, x ) , which appear in the Hamiltonian (4), we use the following trick, adapted from Duarte et al. (2024).

Proposition 3.1. Consider a function v ∈ V together with a pair ( t, x ) ∈ [0 , T ) × D . Define the function ψ : R → R by

<!-- formula-not-decoded -->

where σ i ( t, x, a ) is the i th column of the d × n matrix σ ( t, x, a ) . Then,

<!-- formula-not-decoded -->

Proposition 3.1 makes it possible to replace the computation of gradients and Hessians of v by evaluating the univariate function ψ ′′ (0) , the cost of which, using automatic differentiation, is a small multiple of n · cost ( v ) .

To formulate GPI-PINN, we need the extended Hamiltonian

<!-- formula-not-decoded -->

which by Proposition 3.1, can be written as

<!-- formula-not-decoded -->

In Algorithm 1, we simplify the notation by using H ( t, x, θ, φ ) := H ( t, x, V θ , α φ ( t, x )) .

The loss function L 1 in (7) consists of two terms. The first represents the expected PIDE residual in the interior of the space-time domain with respect to a suitable measure µ on [0 , T ) × D . The second term penalizes violations of the terminal condition according to a measure ν on D . Hence, L 1 measures how well the function V θ satisfies the controlled HJB equation (5) corresponding to a control α φ . In every epoch k , the goal in Step 1 is therefore to find a parameter vector θ such that the value network V θ minimizes the error L 1 ( θ, φ ( k ) ) . We do this with a mini-batch stochastic gradient method which updates the measures µ and ν according to the residual-based adaptive distribution (RAD) method of Wu et al. (2023), as it is known to significantly improve the accuracy of PINNs. In Step 2, we minimize L 2 ( θ ( k +1) , φ ) with respect to φ . This corresponds to choosing the control α φ so as to maximize the extended Hamiltonian H (or equivalently H ); see Duan et al. (2023), Dupret &amp; Hainaut (2024) and Cohen et al. (2025) for theoretical convergence results supporting this approach.

Since the expectations in L 1 , L 2 and the extended Hamiltonian (6) are typically not available in closed form, we replace them by sample-based estimates. First, we estimate H ( t, x, θ, φ ) with ̂ H ( t, x, θ, φ ) by approximating the jump-expectation

<!-- formula-not-decoded -->

## Algorithm 1 GPI-PINN

Initialize admissible weights θ (0) for V θ and φ (0) for α φ . Choose proportionality factors ξ 1 , ξ 2 &gt; 0 and set epoch k = 0 .

repeat

Step 1: Update the value network V θ ( k +1) for a given control α φ ( k ) by minimizing the loss

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Step 2: Update the control network α φ ( k +1) for a given value network V θ ( k +1) by minimizing the loss

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where k ← k +1

until some convergence criterion is satisfied.

return V θ ( k ) and α φ ( k ) and set k ∗ ← k .

for points ( z j ) J j =1 in E sampled from the distribution Z of Z 1 . Then, we approximate L 1 and L 2 in every time step by

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where ( t m , x m ) M 1 m =1 ∈ [0 , T ) × D and ( y m ) M 2 m =1 ∈ D are sampled from µ and ν , respectively. In every epoch k = 0 , . . . , k ∗ -1 , we initialize θ ( k ) 0 := θ ( k ) and make N 1 gradient steps

<!-- formula-not-decoded -->

i = 0 , . . . , N 1 -1 , to obtain θ ( k +1) = θ ( k ) N 1 . Then, we initialize φ ( k ) 0 = φ ( k ) and perform N 2 gradient steps

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

GPI-PINN yields global approximations V θ ( k ∗ ) and α φ ( k ∗ ) of the value function and optimal control on the whole space-time domain [0 , T ] × D . By using Proposition 3.1, it avoids the computation of the gradients and Hessians appearing in the Hamiltonian (4). However, it still has two drawbacks that make it inefficient for high-dimensional control problems with jumps. First, it has to approximate the jump-expectations E [ V θ ( t m , x m + γ ( t m , x m , Z 1 , α φ ( t m , x m ))) ] for all sample points ( t m , x m ) , m = 1 , . . . , M 1 , in each of the gradient steps (10)-(11) and all epochs k = 0 , 1 , . . . , k ∗ ; see (9). Secondly, since the Hamiltonian is already a second-order integro-differential operator, the gradient steps ∇ θ ̂ H in (10) require the computation of third order derivatives, which is numerically costly.

## 4 GPI-CBU

GPI-CBU addresses the shortcomings of GPI-PINN by using a value function updating rule based on the expectation-free operator G ζ : [0 , T ] × D × E ×V × A → R given by

<!-- formula-not-decoded -->

for a scaling factor ζ ∈ R .

Proposition 4.1. Let X α be a solution of the jump-diffusion equation (2) corresponding to a feedback control α : [0 , T ) × D → A with associated value function V α ∈ C 1 , 2 ([0 , T ] × D ) . For given t ∈ [0 , T ) , let Y t be a D -valued random variable independent of Z 1 such that E G 2 ζ ( t, Y t , Z 1 , V α , α ( t, Y t )) &lt; ∞ . Then V α ( t, Y t ) = g ( Y t ) for the Borel measurable function g : D → R minimizing the mean squared error

<!-- formula-not-decoded -->

Proposition 4.1 suggests to update 6 the value function parameters according to

<!-- formula-not-decoded -->

By adding a penalty term enforcing the terminal condition, we obtain the recursive scheme

<!-- formula-not-decoded -->

for the loss

<!-- formula-not-decoded -->

where we use the notation G ζ ( t, x, z, θ, φ ) := G ζ ( t, x, z, V θ , α φ ( t, x )) . To implement (13), we approximate L ( k ) 1 ( θ ) with

<!-- formula-not-decoded -->

for ( t m , x m , z m ) M 1 m =1 ∈ [0 , T ) × D × E sampled from µ ⊗Z and ( y m ) M 2 m =1 ∈ D sampled from ν . In every epoch k = 0 , . . . , k ∗ -1 , we initialize θ ( k ) 0 = θ ( k ) and perform N 1 gradient steps

<!-- formula-not-decoded -->

i = 0 , ..., N 1 -1 , to obtain θ ( k +1) = θ ( k ) N 1 . To update the control network α φ , we introduce the operator

<!-- formula-not-decoded -->

which is an expectation-free version of the extended Hamiltonian H that, instead of the jump-expectation E V θ ( t, x + γ ( t, x, Z 1 , a φ ( t, x ))) , only contains a single jump V θ ( t, x + γ ( t, x, z, a φ ( t, x ))) , z ∈ E . GPI-CBU updates the parameters of the control network α φ according to

<!-- formula-not-decoded -->

6 By (22), the update rule (12) corresponds to V θ ( k +1) = T α V θ ( k ) := V θ ( k ) + ζ H ( · , V θ ( k ) , α ( · )) for the continuous-time Bellman updating (CBU) operator T α associated with the feedback control α and fixed point T α V α = V α .

for the loss

<!-- formula-not-decoded -->

We hence minimize the sample estimate

<!-- formula-not-decoded -->

by setting φ ( k ) 0 = φ ( k ) and making N 2 gradient steps

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Algorithm 2 GPI-CBU

Initialize admissible neural weights θ (0) for V θ and φ (0) for α φ . Choose learning rates η 1 , η 2 , proportionality factors ξ 1 , ξ 2 and numbers of gradient steps N 1 , N 2 . Set epoch k = 0 . repeat

Step 0: Generate M 1 sample points ( t m , x m , z m ) ∈ [0 , T ) × D × E from µ ⊗Z and M 2 sample points y m ∈ D from ν .

Step 1: Update V θ ( k +1) by minimizing the loss (14)

<!-- formula-not-decoded -->

with N 1 gradient steps (15).

Step 2: Update α φ ( k +1) by minimizing the loss (16)

<!-- formula-not-decoded -->

with N 2 gradient steps (17).

<!-- formula-not-decoded -->

until some convergence criterion is satisfied.

return V θ ( k ) and α φ ( k ) and set k ∗ ← k .

Like GPI-PINN, GPI-CBU leverages Proposition 3.1 to bypass the computation of the gradients ∇ x V θ and Hessians ∇ 2 x V θ . In addition, it avoids the costly computation of the jump-expectations (9) of V θ at each sample point ( t m , x m ) . Also, it it does not have to compute third-order derivatives like (10) when updating the value network since only ∇ θ V θ is needed in (15). This is a consequence of the fact that the value update rule (15) of GPI-CBU is recursive as G ζ depends on the value weights θ ( k ) computed in the previous epoch in contrast to the residual-based GPI-PINN. On the other hand, since GPI-PINN averages over different jumps in each update, it exhibits more stable convergence than GPI-CBU; see Figure 1 below. The proportionality factors ξ 1 , ξ 2 and the scaling factor ζ are hyperparameters. ξ 1 and ξ 2 can be fine-tuned following Wang et al. (2022). While Proposition 4.1 holds for any scaling factor ζ ∈ R , its choice affects the numerical performance of GPI-CBU. In the numerical experiments of this paper, we set ζ = 1 as it provides a good trade-off between convergence speed and accuracy of the improvements in GPI-CBU. On the other hand, negative scaling factors failed to converge to the true solution in all our experiments with exploding losses ̂ L ( k ) 1 and ̂ L ( k ) 2 after only a few epochs. Alternatively, one could consider an adaptive scaling factor ζ k &gt; 0 depending on the epoch k . More details on hyperparameter fine-tuning are given in Appendix B. Finally, we note that the policy updating rule of GPI-CBU (17) is equivalent to that of GPI-PINN (11), except that it circumvents the computation of the jump-expectations in ̂ H since it uses the expectation-free operator G .

## 5 Numerical experiments

In our numerical experiments, we use the Deep Galerkin Method (DGM) architecture of Sirignano &amp; Spiliopoulos (2018) for both the value and optimal control networks as it has been shown to

empirically improve PINN performance. Details of the network design and hyperparameters are given in Appendix B.

## 5.1 Linear-quadratic regulator with jumps

We first consider the linear-quadratic regulator (LQR) problem with jumps

<!-- formula-not-decoded -->

where the infimum is over d -dimensional predictable processes ( α t ) 0 ≤ t ≤ T and X α is a d -dimensional process with controlled dynamics

<!-- formula-not-decoded -->

for a d × d matrix B , a d -dimensional Brownian motion W , a Cox process M α with intensity λ α t = Λ 1 +Λ 2 ‖ α t ‖ 2 2 for constants Λ 1 , Λ 2 ≥ 0 and independent i.i.d. zero-mean square integrable d -dimensional random vectors Z j with E [ ‖ Z j ‖ 2 2 ] = υ , j = 1 , 2 , . . .

Problem (18) is of the general form (1) if instead of the infimum, one considers the supremum of the negative of the expectation. It can be reduced to a one-dimensional ODE which has a closed form solution in the special case Λ 2 = 0 and admits a very precise numerical solution via a standard ODE solver, such as Runge-Kutta, if Λ 2 &gt; 0 (see Appendix C). This provides highly accurate reference solutions V and α ∗ for the value function and optimal control.

Figure 1 compares the performances of GPI-PINN and GPI-CBU on a 10-dimensional ( d = 10) LQR problem of the form (18) with T = 1 , B = I d , c 1 = 1 , c 2 = 1 / 4 without jumps ( Λ 1 = Λ 2 = 0 ) and with jumps ( Λ 1 = 0 , Λ 2 = 2 and d -dimensional standard normal jumps Z j ). It shows mean absolute errors of V θ ( k ) with respect to V given by MAE V = 1 M ∑ M i =1 | V θ ( k ) ( t i , x i ) -V ( t i , x i ) | on a test set of size M uniformly sampled from [0 , 1] × [ -2 . 5 , 2 . 5] d together with runtimes as functions of the number of epochs k . It can be seen that GPI-PINN and GPI-CBU exhibit similar convergence in the number of epochs. The residual-based approach of GPI-PINN makes it more stable than GPI-CBU; see also Baird (1995). On the other hand, GPI-CBU has a lower computational cost. This is already the case without jumps (left plot of Figure 1) since it avoids the numerical evaluation of third-order derivatives but becomes much more significant with jumps (right plot of Figure 1) as it also circumvents the numerical integration of the jumps.

<!-- image -->

Epoch k

Epoch k

Figure 1: Comparison of MAE V (blue) and runtime in seconds (green) of GPI-PINN (solid lines) and GPI-CBU (dashed lines) for a 10-dimensional LQR problem (18) without jumps (left) and with jumps (right).

Figure 2 shows results of GPI-CBU for the same LQR problem with jumps in d = 50 dimensions. While the high dimensionality renders GPI-PINN infeasible, GPI-CBU achieves high accuracy in the approximation of the value function as well as the optimal policy. Additional results for up to 150-dimensional LQR problems are reported in Appendix C.

<!-- image -->

X1

Figure 2: Value function V (0 , x ) (left) and first component of the optimal control α ∗ 1 (0 , x ) (right) for x = ( x 1 , 0 , . . . , 0) with x 1 ∈ [ -2 . 5 , 2 . 5] for a 50-dimensional LQR problem with jumps. Orange dotted lines: numerical results of GPI-CBU with ± 1 standard deviation given by orange shaded area. Blue lines: reference solutions V and α ∗ .

Figure 3: log MAE V of different deep-learning methods for a 10-dimensional LQR problem with jumps.

<!-- image -->

Figure 3 shows the accuracy of GPI-CBU for the 10dimensional LQR problem with jumps from the right panel of Figure 1 compared with the two popular modelfree RL algorithms PPO and SAC as well as the modelbased discrete-time approach of Han &amp; E (2016) applied to a time-discretization of the state dynamics (19). It can be seen that in this setup, PPO and SAC cannot compete with the two model-based approaches since they do not explicitly use the dynamics (19) but instead only rely on sampling from the environment. The method of Han &amp; E (2016) outperforms PPO and SAC but does not achieve the same accuracy as GPI-CBU due to discretization errors and since it does not generalize well to unseen points in the test set. Trying to cover the space-time domain well, we ran the method of Han &amp; E (2016) from several randomly sampled starting points x 0 ∈ D . But being a local method, it tends to learn the optimal control only along the optimal state trajectories ( t, X α ∗ t ) 0 ≤ t ≤ T , which results in poor performance in parts of the spacetime domain that are not explored well. Additional results are discussed in Appendix C.

## 5.2 Optimal consumption-investment with jumps

As a second example, we consider an economic agent who consumes at relative rate c t and invests in n financial assets according to a strategy 7 ( π 1 t , . . . , π n t ) so as to maximize

<!-- formula-not-decoded -->

for two utility functions 8 u, U : R + → R , where Y α t is the wealth process evolving like

<!-- formula-not-decoded -->

for a stochastic interest rate r t , expected return rates µ i , stochastic volatilities σ i t , an n -dimensional Brownian motion W with correlation matrix Σ , Cox processes M i with stochastic jump intensities λ i t and random jump variables Z i j . We consider strategies of the form and c ( t, σ t , λ t , r t , Y α t -) and π i ( t, σ t , λ t , r t , Y α t -) . The problem has d = 2 n +2 state variables, consisting of σ i t , λ i t , r t , Y α t and

7 π i t describes the fraction of the agent's total wealth held in the i th asset at time t .

8 In our numerical experiments, we choose CRRA utility functions.

n +1 decision variables c t , π i t , i = 1 , . . . , n . In general, the corresponding HJB equation, given in (35) in Appendix D.2, does not have an analytical solution. But it can be seen in Figure 4 that GPI-CBU converges numerically for n = 25 financial assets. More details about this consumption-investment problem with stochastic coefficients are provided in Appendix D.2.

In Appendix D.1, we consider a simplified version of the problem where the coefficients σ i t , λ i t , r t are constant. This case can be reduced to a one-dimensional ODE that can be solved with a standard Runge-Kutta scheme to obtain reference solutions. GPI-CBU yields numerical solutions that are virtually indistinguishable from the Runge-Kutta results.

<!-- image -->

Epoch k

Figure 4: Training losses ̂ L ( k ) 1 ( θ ( k +1) ) (left) and ̂ L ( k +1) 2 ( φ ( k +1) ) (right) of GPI-CBU as functions of the epoch k for a consumption-investment problem of the form (20)-(21) with n = 25 financial assets. The blue curve in the left plot represents the interior loss of ̂ L ( k ) 1 and the orange curve its boundary part; see (14).

## 6 Conclusions, limitations and future work

In this paper, we have introduced two iterative deep learning algorithms for solving finite-horizon stochastic control problems with jumps. Both use an actor-critic approach and train two neural networks to approximate the value function and optimal control, providing global solutions over the entire space-time domain without requiring simulation or discretization of the underlying state dynamics. Our first algorithm, GPI-PINN, works well for high-dimensional problems without jumps but becomes computationally infeasible in the presence of jumps. The second algorithm, GPICBU, leverages an efficient expectation-free updating rule based on Proposition 4.1 which makes it particularly well-suited for high-dimensional problems with jumps. Both algorithms are model-based. As such, they outperform model-free RL methods in cases where the underlying state dynamics are known. The accuracy and scalability of the two algorithms has been demonstrated in different numerical examples.

A limitation of our approach lies in the need to know the underlying dynamics of the state process, which are not always available in real-world applications. While it is reasonable to assume that physical systems obey known laws of motion, dynamics in economics and finance typically need to be inferred from data. However, in such cases, they can be learned in a preliminary step using e.g. recent model-learning algorithms such as Brunton et al. (2016) or Champion et al. (2019). Once an approximate model has been learned from data, one of the proposed algorithms, GPI-PINN or GPI-CBU, can be applied to efficiently solve the resulting stochastic control problem.

## Acknowledgments

Financial support from Swiss National Science Foundation Grant 10003723 is gratefully acknowledged. We thank the reviewers for their helpful comments and constructive suggestions.

## References

Bachouch, A., Huré, C., Langrené, N., and Pham, H. Deep neural networks algorithms for stochastic control problems on finite horizon: numerical applications. Methodology and Computing in Applied Probability , 24(1):143-178, 2022.

- Baird, L. Residual algorithms: reinforcement learning with function approximation. International Conference on Machine Learning , 12:30-37, 1995.
- Beck, C., Becker, S., Cheridito, P., Jentzen, A., and Neufeld, A. Deep splitting method for parabolic PDEs. SIAM Journal on Scientific Computing , 43(5):A3135-A3154, 2021.
- Bellemare, M. G., Dabney, W., and Munos, R. A distributional perspective on reinforcement learning. International Conference on Machine Learning , 34:449-458, 2017.
- Berg, J. and Nyström, K. A unified deep artificial neural network approach to partial differential equations in complex geometries. Neurocomputing , 317:28-41, 2018.
- Bouchard, B. Introduction to Stochastic Control of Mixed Diffusion Processes, Viscosity Solutions and Applications in Finance and Insurance . Ceremade Lecture Notes, 2021.
- Bruna, J., Peherstorfer, B., and Vanden-Eijnden, E. Neural Galerkin schemes with active learning for high-dimensional evolution equations. Journal of Computational Physics , 496, 2024.
- Brunton, S. L., Proctor, J. L., and Kutz, J. N. Discovering governing equations from data by sparse identification of nonlinear dynamical systems. Proceedings of the National Academy of Sciences , 113(15):3932-3937, 2016.
- Champion, K., Lusch, B., Kutz, J. N., and Brunton, S. L. Data-driven discovery of coordinates and governing equations. Proceedings of the National Academy of Sciences , 116(45):22445-22451, 2019.
- Cohen, S. N., Hebner, J., Jiang, D., and Sirignano, J. Neural actor-critic methods for HamiltonJacobi-Bellman PDEs: asymptotic analysis and numerical studies. arXiv Preprint 2507.06428 , 2025.
- Domingo-Enrich, C., Drozdzal, M., Karrer, B., and Chen, R. T. Adjoint matching: fine-tuning flow and diffusion generative models with memoryless stochastic optimal control. arXiv Preprint 2409.08861 , 2024a.
- Domingo-Enrich, C., Han, J., Amos, B., Bruna, J., and Chen, R. T. Stochastic optimal control matching. Advances in Neural Information Processing Systems , 37:112459-112504, 2024b.
- Dormand, J. R. and Prince, P. J. A family of embedded Runge-Kutta formulae. Journal of Computational and Applied Mathematics , 6(1):19-26, 1980.
- Duan, J., Li, J., Ge, Q., Li, S. E., Bujarbaruah, M., Ma, F., and Zhang, D. Relaxed actor-critic with convergence guarantees for continuous-time optimal control of nonlinear systems. IEEE Transactions on Intelligent Vehicles , 8(5):3299-3311, 2023.
- Duarte, V., Duarte, D., and Silva, D. H. Machine learning for continuous-time finance. The Review of Financial Studies , 37(11):3217-3271, 2024.
- Dupret, J.-L. and Hainaut, D. Deep learning for high-dimensional continuous-time stochastic optimal control without explicit solution. Technical report, Université Catholique de Louvain, Institute of Statistics, Biostatistics and Acturial Sciences, 2024.
- Durrett, R. Probability: Theory and Examples . Cambridge University Press, fifth edition, 2019.
- Gao, X., Li, L., and Zhou, X. Y. Reinforcement learning for jump-diffusions with financial applications. arXiv Preprint 2405.16449 , 2024.
- Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. Soft actor-critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor. International Conference on Machine Learning , 35: 1861-1870, 2018.
- Han, J. and E, W. Deep learning approximation for stochastic control problems. arXiv Preprint 1611.07422 , 2016.

- Han, J., Jentzen, A., and E, W. Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations. Communications in Mathematics and Statistics , 5(4):349-380, 2017.
- Han, J., Jentzen, A., and E, W. Solving high-dimensional partial differential equations using deep learning. Proceedings of the National Academy of Sciences , 115(34):8505-8510, 2018.
- Huré, C., Pham, H., Bachouch, A., and Langrené, N. Deep neural networks algorithms for stochastic control problems on finite horizon: convergence analysis. SIAM Journal on Numerical Analysis , 59(1):525-557, 2021.
- Jacka, S. D. and Mijatovi´ c, A. On the policy improvement algorithm in continuous time. Stochastics , 89(1):348-359, 2017.
- Ji, S., Peng, S., Peng, Y ., and Zhang, X. Solving stochastic optimal control problem via stochastic maximum principle with deep learning method. Journal of Scientific Computing , 93(1):30, 2022.
- Jia, Y . and Zhou, X. Y. q-learning in continuous time. Journal of Machine Learning Research , 24 (161):1-61, 2023.
- Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 , 2014.
- Li, X., Verma, D., and Ruthotto, L. A neural network approach for stochastic optimal control. SIAM Journal on Scientific Computing , 46(5):C535-C556, 2024.
- Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y ., Silver, D., and Wierstra, D. Continuous control with deep reinforcement learning. arXiv Preprint 1509.02971 , 2019.
- Lu, L., Meng, X., Mao, Z., and Karniadakis, G. E. Deepxde: A deep learning library for solving differential equations. SIAM Review , 63(1):208-228, 2021.
- Mnih, V., Kavukcuoglu, K., Silver, D., Alex, G., Antonoglou, I., Wierstra, D., and Riedmiller, M. Playing Atari with deep reinforcement learning. arXiv Preprint 1312.5602 , 2013.
- Mnih, V., Puigdomènech Badia, A., Mirza, M., Graves, A., Lillicrap, T. P., Harley, T., Silver, D., and Kavukcuoglu, K. Asynchronous methods for deep reinforcement learning. arXiv Preprint 1602.01783 , 2016.
- Nüsken, N. and Richter, L. Solving high-dimensional Hamilton-Jacobi-Bellman PDEs using neural networks: perspectives from the theory of controlled diffusions and measures on path space. Partial Differential Equations and Applications , 2(4):48, 2021.
- Øksendal, B. and Sulem, A. Applied Stochastic Control of Jump Diffusions . Springer, third edition, 2007.
- Raissi, M., Perdikaris, P., and Karniadakis, G. E. Physics informed deep learning (Part 1): data-driven solutions of nonlinear partial differential equations. arXiv preprint arXiv:1711.10561 , 2017.
- Raissi, M., Perdikaris, P., and Karniadakis, G. E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics , 378:686-707, 2019.
- Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. Trust region policy optimization. International Conference on Machine Learning , 32:1889-1897, 2015.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv Preprint 1707.06347 , 2017.
- Sirignano, J. and Spiliopoulos, K. DGM: A deep learning algorithm for solving partial differential equations. Journal of Computational Physics , 375:1339-1364, 2018.
- Soner, H. M. Optimal control of jump-markov processes and viscosity solutions. In Stochastic Differential Systems, Stochastic Control Theory and Applications , pp. 501-511. Springer, 1988.

- Sutton, R. S. and Barto, A. G. Reinforcement Learning: An Introduction . The MIT Press, second edition, 2018.
- Szpruch, L., Treetanthiploet, T., and Zhang, Y . Optimal scheduling of entropy regularizer for continuoustime linear-quadratic reinforcement learning. SIAM Journal on Control and Optimization , 62 (1):135-166, 2024.
- Wang, H., Zariphopoulou, T., and Zhou, X. Y. Reinforcement learning in continuous time and space: a stochastic control approach. Journal of Machine Learning Research , 21(198):1-34, 2020.
- Wang, S., Yu, X., and Perdikaris, P. When and why PINNs fail to train: a neural tangent kernel perspective. Journal of Computational Physics , 449:110768, 2022.
- Werbos, P. Approximate dynamic programming for real-time control and neural modeling. Handbook of Intelligent Control: Neural, Fuzzy and Adaptative Approaches , pp. 493-526, 1992.
- Wu, C., Zhu, M., Tan, Q., Kartha, Y., and Lu, L. A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks. Computer Methods in Applied Mechanics and Engineering , 403:115671, 2023.

## A Proofs

## A.1 Proof of Proposition 3.1

Denoting v i ( · ) = v ( t + h 2 2 n , x + h √ 2 σ i ( t, x, a ) + h 2 2 n β ( t, x, a ) ) , the second-order derivative of ψ ( h ) := ∑ n i =1 v i ( · ) is given by

<!-- formula-not-decoded -->

Evaluating it at h = 0 gives

<!-- formula-not-decoded -->

which proves the proposition.

## A.2 Proof of Proposition 4.1

Since Y t is independent of Z 1 , one obtains from the definition of G ζ that

<!-- formula-not-decoded -->

where the last equality follows from Theorem 2.1. On the other hand, it is well known that

<!-- formula-not-decoded -->

for the Borel measurable function g : D → R minimizing the mean squared error

<!-- formula-not-decoded -->

see e.g. Theorem 4.1.15 of Durrett (2019). This completes the proof.

## B DGMArchitecture

Figure 5: DGM architecture of the value neural network with L = 3 ( i.e. 4 hidden layers).

<!-- image -->

Each DGM layer in Figure 5 in the value network V θ is of the following form

<!-- formula-not-decoded -->

where the number of hidden layers is L +1 , · denotes matrix multiplication and glyph[circledot] element-wise multiplication. The DGM parameters of the value network are

<!-- formula-not-decoded -->

The number of units in each layer is N . σ V : R N → R N is a twice-differentiable element-wise nonlinearity and σ V out : R N → V is the output activation function, also twice-differentiable, where V ⊆ R is chosen so as to satisfy possible restrictions on the value function's output, resulting from the form of the stochastic control problem (e.g. non-negative value function). The control network is designed following the same DGM architecture with σ α out : R N → A . Throughout the numerical examples of the paper, we use L = 3 with N = 50 neurons in each of the DGM layers, see Figure 5. In the value network, we use tanh for the activation function σ V and softplus for σ V out . For the control network, we adopt tanh for σ α , while as output activation σ α out we choose the identity in Example 5.1 and softplus in Example 5.2.

Unless stated otherwise, we use a number of sample points M 1 , M 2 equal to 256 , a number of gradient steps N 1 , N 2 equal to 64 and a maximum number of epochs k ∗ = 1500 . The network parameters are updated using Adam (Kingma &amp; Ba, 2014) with constant learning rates η 1 , η 2 equal to 0 . 001 . Both GPI-PINN and GPI-CBU are fairly robust with respect to these choices. In our setting, the most critical hyperparameters for convergence are the proportionality factors ξ 1 , ξ 2 , which we determined using the approach of Wang et al. (2022), together with the scaling rate ζ (specific to GPI-CBU). In our experiments, we set ζ = 1 as it provides a good trade-off between convergence speed and accuracy of the improvements in GPI-CBU. Alternatively, one could consider an adaptive scaling factor ζ k &gt; 0 depending on the epoch k . On the other hand, negative scaling factors resulted in non-convergence in all our experiments with exploding losses ̂ L ( k ) 1 and ̂ L ( k ) 2 after only a few epochs.

Algorithms 1 and 2 were implemented using TensorFlow and Keras with GPU acceleration on an NVIDIA RTX 4090.

## C Linear-quadratic regulator with jumps: detailed results

The value function of the LQR problem with jumps (18) is given by

<!-- formula-not-decoded -->

The corresponding HJB equation is

<!-- formula-not-decoded -->

with terminal condition V ( T, x ) = c 2 ‖ x ‖ 2 2 . Using the Ansatz

<!-- formula-not-decoded -->

it is straightforward to see that g is of the form

<!-- formula-not-decoded -->

for the unique solution of the non-linear ODE

<!-- formula-not-decoded -->

(25) can efficiently be solved using a numerical method such as Runge-Kutta of order 5(4) ; see e.g. Dormand &amp; Prince (1980). The optimal control α ∗ is given in terms of h by

<!-- formula-not-decoded -->

Note that in the constant intensity case Λ 1 &gt; 0 , Λ 2 = 0 , (24)-(25) admit the explicit solutions

<!-- formula-not-decoded -->

But also in the case Λ 2 &gt; 0 , numerical solutions of (24)-(25) obtained with Runge-Kutta of order 5(4) provide very precise reference solutions V and α ∗ to the LQR problem (18).

GPI-PINN and GPI-CBU both need random points ( t m , x m ) ∈ [0 , T ) × R d and ( y m ) ∈ R d sampled from two distributions µ on [0 , T ) × R d and ν on R d , which we chose according to Bachouch et al. (2022).

In the following we consider problem (18) with T = 1 , B = I d , c 1 = 1 , c 2 = 1 / 4 and d -dimensional standard normal jumps Z j . Figures 6-8 show results obtained with GPI-CBU in d = 50 dimensions for the constant jump intensity case Λ 1 = 1 / 4 and Λ 2 = 0 , which admits analytical reference solutions.

<!-- image -->

X1

X1

Figure 6: Value function V (0 , x ) (left) and first component of the optimal control α ∗ 1 (0 , x ) (right) for x = ( x 1 , 0 , . . . , 0) with x 1 ∈ [ -2 . 5 , 2 . 5] . Orange dotted lines: numerical results of GPI-CBU with ± 1 standard deviation given by orange shaded area. Blue lines: analytical solution (26). d = 10 , Λ = 1 / 4 , Λ 2 = 0 , k ∗ = 1500 .

<!-- image -->

t

t

Figure 7: Value function V ( t, x ) (left) and first component of the optimal control α ∗ 1 ( t, x ) (right) for t ∈ [0 , 1] and x = 1 50 . Orange dotted lines: numerical results from GPI-CBU with ± 1 standard deviation in orange shaded area. Blue lines: analytical solution (26). d = 50 , Λ 1 = 1 / 4 , Λ 2 = 0 , k ∗ = 1500 .

<!-- image -->

X1

X1

Figure 8: Heatmaps of MAE V (left) and MAE α (right) for t ∈ [0 , 1] and x = ( x 1 , 1 , . . . , 1) with x 1 ∈ [ -2 . 5 , 2 . 5] obtained from GPI-CBU for the LQR problem with d = 50 , Λ 1 = 1 / 4 and Λ 2 = 0 . k ∗ = 1500 .

MAE α in Figure 8 is defined, analogously to MAE V in Section 5.1, by

<!-- formula-not-decoded -->

for a test set of size M uniformly sampled from [0 , 1] × [ -2 . 5 , 2 . 5] d .

Next, we consider the same LQR problem (18) as above in 50 dimensions with a controlled jump intensity. Figures 9-10 (and 2), show results of GPI-CBU for Λ 1 = 0 and Λ 2 = 2 . Reference solutions were computed with Runge-Kutta of order 5(4).

<!-- image -->

Figure 9: Value function V ( t, x ) (left) and first component of the optimal control α ∗ 1 ( t, x ) (right) for t ∈ [0 , 1] and x = 1 50 . Orange dotted lines: numerical results of GPI-CBU with ± 1 standard deviation in orange shaded area. Blue lines: reference solution. d = 50 , Λ 1 = 0 , Λ 2 = 2 , k ∗ = 1500 .

<!-- image -->

X1

X1

Figure 10: Heatmaps of MAE V (left) and MAE α (right) for t ∈ [0 , 1] and x = ( x 1 , 1 49 ) with x 1 ∈ [ -2 . 5 , 2 . 5] obtained from GPI-CBU for the LQR problem with d = 50 , Λ 1 = 1 / 4 and Λ 2 = 0 . k ∗ = 1500 .

These results illustrate the accuracy of GPI-CBU for high-dimensional control problems with controlled jumps. It can be seen that the optimal control α ∗ is lower in magnitude compared to the case of constant jump intensity since now, exerting control increases the likelihood of jumps. But in both cases, it is optimal to steer the state process X α t towards 0 as t approaches the time horizon T .

Table 1 summarizes results of GPI-CBU for higher-dimensional LQR problems with the same parameters as above with controlled jump intensity Λ 1 = 0 , Λ 2 = 2 obtained over k ∗ = 5000 training epochs. This demonstrates the scalability of GPI-CBU.

Table 1: GPI-CBU performance metrics as a function of the state dimension d for LQR problems of the form (18) with Λ 1 = 0 , Λ 2 = 2 and k ∗ = 5000 training epochs.

|   Dimensions d |   MAE V |   MAE α |   Loss ̂ L 1 ( θ k ∗ ) |   Loss ̂ L 2 ( θ k ∗ ) | Time (sec)   |
|----------------|---------|---------|------------------------|------------------------|--------------|
|              5 |  0.0023 | 0.0041  |                 0.0237 |                -0.1332 | 6,410        |
|             10 |  0.0025 | 0.0049  |                 0.0314 |                -0.1972 | 8,093        |
|             50 |  0.0147 | 0.0075  |                 0.1267 |                -0.4206 | 16,129       |
|            100 |  0.0492 | 0.00539 |                 0.697  |                -0.4461 | 24,359       |
|            150 |  0.0979 | 0.0096  |                 3.721  |                -0.4671 | 33,120       |

## D Optimal consumption-investment problem

## D.1 Constant coefficients

We first study the optimal consumption-investment problem in a model with constant coefficients, where the risk-free asset evolves according to

<!-- formula-not-decoded -->

for a constant interest rate r ∈ R , and there are n stocks with prices following

<!-- formula-not-decoded -->

for constant drifts µ i ∈ R and volatilities σ i ∈ R + , an n -dimensional Brownian motion W with correlation matrix Σ , independent Poisson processes M i with constant intensities λ i ≥ 0 and i.i.d. normal random variables Z i 1 , Z i 2 , ... ∼ N ( µ i Z , σ i Z ) with µ i Z ∈ R and σ i Z ∈ R + . Suppose an investor starts with an initial endowment of Y 0 &gt; 0 , consumes at rate c t Y α t -and invests π i t Y α t -, i = 1 , . . . n , in the n stocks. If the risk-free asset is used to balance the transactions, the resulting wealth evolves according to

<!-- formula-not-decoded -->

Let us assume the investor attempts to maximize

<!-- formula-not-decoded -->

for two utility functions u, U : R + → R and a discount factor ρ ∈ R . Since the driving noise in (29) has stationary and independent increments, it is enough to consider strategies of the form c t = c ( t, Y α t -) and π i t = π i ( t, Y α t -) for functions c, π i : [0 , T ) × R + → R + , i = 1 , . . . , n . We write this n +1 -dimensional control as α t = ( c t , π 1 t , . . . , π n t ) glyph[latticetop] ∈ A := R n +1 + . Assuming constant relative risk aversion (CRRA) utility functions u, U with relative risk aversion γ &gt; 0 and denoting δ = 1 -γ , the resulting reward functional is

<!-- formula-not-decoded -->

Finally, writing µ = ( µ 1 , . . . , µ n ) glyph[latticetop] , σ = diag ( σ 1 , . . . , σ n ) , µ Z = ( µ 1 Z , . . . , µ n Z ) glyph[latticetop] , σ Z = ( σ 1 Z , . . . , σ n Z ) glyph[latticetop] , λ = ( λ 1 , . . . , λ n ) glyph[latticetop] , π = ( π 1 , . . . , π n ) glyph[latticetop] , the associated HJB equation for the value function V ( t, y ) = sup α V α ( t, y ) satisfies 9 for all ( t, y ) ∈ [0 , T ) × R + ,

<!-- formula-not-decoded -->

with V ( T, y ) = y δ /δ . This stochastic control problem does not admit an analytical solution but we can instead characterize its solution in terms of a PIDE, so as to assess the accuracy of the proposed Algorithms 1 and 2. Following Øksendal &amp; Sulem (2007), we assume the value function is of the form V ( t, y ) = A ( t ) y δ /δ . Then optimizing the HJB equation (31) leads to first order conditions showing that the optimal consumption rate c ∗ is independent from the wealth y and given by

<!-- formula-not-decoded -->

and the optimal investment strategy is given by a vector of constants π ∗ ( t, y ) = π ∗ := ( π ∗ , 1 , . . . , π ∗ ,n ) glyph[latticetop] satisfying

<!-- formula-not-decoded -->

Plugging these optimal controls back into the HJB equation (31), we find that the function A ( t ) satisfies the ODE

<!-- formula-not-decoded -->

with A ( T ) = 1 , which can be solved numerically using again the Runge-Kutta method of order 5(4). For both Algorithms 1 and 2, we sample the time and space points independently from the uniform distributions U [0 ,T ] and U [0 ,y b ] , respectively. The parameters of the optimal investment problem below are as follows: T = 1 , y b = 150 , r = 0 . 02 , ρ = 0 . 045 , δ = 0 . 7 , λ = 0 . 45 · 1 n , µ Z = 0 . 25 · 1 n , σ Z = 0 . 2 · 1 n , µ = 0 . 032 · 1 n , σ = I n , Σ = 0 . 2 · 1 n × n with diag (Σ) = 1 n . Note that, compared to the LQR problem in Section 5.1, the proportionality factor ξ is now set to 10 . Moreover, since A = R n +1 + , σ α out is chosen to be the softplus activation function. Instead of the DGM architecture, we here train a classical feedforward neural network with 4 hidden layers, each of 128 neurons. Finally, the MAE V and MAE α metrics are again defined as in Section 5.1 on a test set of size M , uniformly sampled from [0 , 1] × [0 , y b ] with V and α ∗ obtained from the Runge-Kutta method described above.

In Figure 11 we again compare MAE V and runtime of GPI-PINN and GPI-CBU as functions of the number of epochs k for n = 10 stocks in the consumption-investment problem (30) with and without jumps. We again see that the residual-based GPI-PINN Algorithm 1 tends to be more stable and accurate for larger epochs, despite having a higher runtime. When accounting for jumps in the dynamics (29), GPI-PINN becomes numerically very time-consuming, even for a 10-dimensional control problem. This issue is even amplified compared with the LQR problem (see Figure 1), as the jump size now depends on the control π in the wealth dynamics (29). In contrast, GPI-CBU again handles these jumps far more efficiently.

9 Note that the term -ρV ( t, y ) in (31) is not contained in our general HJB equation (3). But GPI-PINN and GPI-CBU still work if it is added.

<!-- image -->

Epoch k

Epoch k

Figure 11: Comparison of MAE V (blue) and runtime in seconds (green) of GPI-PINN (solid line) and GPICBU (dashed line) as functions of the number of epochs k for the optimal consumption-investment problem without jumps (left) and with jumps (right) for n = 10 stocks.

Next, we address a higher-dimensional optimal consumption-investment problem with jumps ( n = 50 stocks), where only GPI-CBU is implemented since GPI-PINN becomes then numerically infeasible. Figures 12 and 13 again confirm the accuracy of GPI-CBU for both the value function and the optimal consumption rate c ∗ ( t ) , with the standard deviation across 10 independent runs of GPI-CBU being virtually imperceptible. The optimal wealth allocations π ∗ , being constant, are not depicted. However, the MAE α heatmap in Figure 14 corroborates our method's accuracy in determining both c ∗ and π ∗ . We also observe that the higher absolute errors tend to occur at the boundaries of the domain.

Figure 12: Value function V ( t, y ) for t = 0 , y ∈ [0 , 150] (left) and t ∈ [0 , 1] , y = 50 (right) for the optimal consumption-investment problem (20) with constant coefficients and n = 50 stocks. Orange dotted lines: results of GPI-CBU with ± 1 standard deviation given by orange shaded area ( k ∗ = 1000 ). Blue lines: reference solution from Runge-Kutta applied to (32)-(33).

<!-- image -->

Figure 13: Optimal relative consumption rate c ∗ ( t ) for t ∈ [0 , 1] for the optimal consumption-investment problem (20) with constant coefficients and n = 50 stocks. Orange dotted line: results of GPI-CBU with ± 1 standard deviation given by orange shaded area ( k ∗ = 1000 ). Blue line: reference solution from Runge-Kutta applied to (32)-(33).

<!-- image -->

<!-- image -->

Wealth y

Figure 14: Heatmaps of MAE V (left) and MAE α (right) for t ∈ [0 , 1] and y ∈ [0 , 150] , for GPI-CBU applied to the optimal consumption-investment problem (20) with constant coefficients and n = 50 stocks.

## D.2 Stochastic coefficients

Now, we study a more complex optimal consumption-investment problem in a realistic market of the form (27)-(28) with a stochastic interest rate

<!-- formula-not-decoded -->

with stochastic (Heston) volatility models for the n stock prices

<!-- formula-not-decoded -->

for σ i t = √ v i t , where v i t follows

<!-- formula-not-decoded -->

and with stochastic jump intensities

<!-- formula-not-decoded -->

We denote W = ( W S, 1 , . . . , W S,n , W v, 1 , . . . , W v,n , W λ, 1 , . . . , W λ,n , W r ) glyph[latticetop] a (3 n + 1) -dimensional Brownian motion with correlation matrix Σ . In this case, the wealth process can still be described as follows

<!-- formula-not-decoded -->

and the strategies should be of the form c ( t, v t , λ t , r t , Y α t -) and π i ( t, v t , λ t , r t , Y α t -) , with α t = ( c t , π 1 t , . . . , π n t ) glyph[latticetop] ∈ A := R n +1 + . Hence, the dynamics of the (2 n +2) -dimensional process X α t = ( v t , λ t , r t , Y α t -) are given by

<!-- formula-not-decoded -->

with µ X ( · ) ∈ R 2 n +2 , Σ X ( · ) ∈ R (2 n +2) × (3 n +1) + and γ X ( · ) ∈ R 2 n +2 such that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

with terminal condition V ( T, x ) = y δ /δ . We consider a portfolio of n = 25 stocks, resulting in a 52 -dimensional value function and a 26 -dimensional control process, with the same parameters as in Section D.1. This version of the consumption-investment problem is significantly more complex than the one discussed in Section D.1, with the value function's dimensionality increasing from 2 to 52. Consequently, Runge-Kutta can no longer be used to solve the associated HJB equation (35), making it impossible to compute the MAE V and MAE α metrics. However, despite this lack of a reference for comparison, GPI-CBU still produces results that appear reasonable. We indeed first observe in Figure 4 (main manuscript) that the losses ̂ L ( k ) 1 (14) and ̂ L ( k ) 2 (16) converge as the number of epoch k increases. Note that the orange curve in the left plot represents the interior loss (first right-hand term) of ̂ L ( k ) 1 , while the blue curve is the boundary loss (second right-hand term) of ̂ L ( k ) 1 , see Eq. (14).

The following Figures 15 and 16 confirm that the results of GPI-CBU for both the value function and optimal control are reasonable, as they closely resemble those of Figures 12 and 13. For the value functions of Figure 15, the standard deviations across 10 independent runs of GPI-CBU remain very low, confirming the stability of the approximations. For the optimal consumption rate in Figure 16 (left plot), the standard deviation across the 10 runs tends to be higher for values of time t close to 0, although being still reasonable. Finally, varying the first dimension of the intensity λ 1 mainly impacts the corresponding fraction of wealth π ∗ , 1 t , while its effect on the other proportions and consumption rate is more moderate (since arising from the correlation matrix Σ of the Brownian motion W ).

Figure 15: Value function V ( t, x ) for t = 0 and x = (0 . 15 · 1 10 , 1 10 , 0 . 02 , y ) with y ∈ [0 , 150] (left) and for t ∈ [0 , 1] and x = (0 . 15 · 1 10 , 1 10 , 0 . 02 , 50) (right), for the optimal consumption-investment problem (20) with stochastic coefficients and n = 25 stocks. Blue lines: results of GPI-CBU with ± 1 standard deviation given by orange shaded area ( k ∗ = 1000 ).

<!-- image -->

Figure 16: Optimal policy for the optimal consumption-investment problem (20) with stochastic coefficients and n = 25 stocks. Left plot: optimal consumption rate c ∗ ( t, x ) for t ∈ [0 , 1] and x = (0 . 15 · 1 10 , 1 10 , 0 . 02 , 50) , results of GPI-CBU in blue and ± 1 standard deviation given by orange shaded area ( k ∗ = 1000 ). Right plot: optimal consumption rate c ∗ ( t, x ) and first five optimal fractions of wealth π ∗ ( t, x ) for t = 0 and x = (0 . 15 · 1 10 , λ 1 , 1 9 , 0 . 02 , 50) for λ 1 ∈ [0 , 5] .

<!-- image -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Every claim in the abstract and introduction is either theoretically or numerically supported in the paper. Similarly, the Conclusion, limitations and future work Sections discuss the limitations and possible extensions of the current work.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The main limitation of our approach lies in the need to know the underlying dynamics of the state process, which is not always available in real-world applications. While it is reasonable to assume that physical systems obey known laws of motion, the dynamics in economics and finance often need to be estimated. Moreover, theoretical convergence guarantees for the GPI-PINN and GPI-CBU still need to be established in future research. This has been emphasized as a separate part of the conclusion.

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

Justification: In the novel Propositions 3.1 and 4.1, all assumptions required to establish the results are clearly stated (or corresponding references provided). Similarly, for the well-known (standard) Theorem 2.2.

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

Justification: Sections 3-4 describes how to implement the two algorithms proposed in this paper with detailed pseudo-code. Sufficient information are also disclosed in Section 5 and Appendix B for reproducing the associated numerical results. The code will be made open access upon acceptance of the paper.

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

Justification: The code is available at https://github.com/jdupret97/Deep-Learningfor-CT-Stochastic-Control-with-Jumps .

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so No is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: The training and testig details are specified in Section 5 and the Appendix B.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All relevant figures are presented with ± 1 -sigma confidence intervals. This is stated every time needed and the computation of these standard errors is explained in Section 5.

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

Justification: Please consult the supplementary material in Appendix B.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conforms with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This paper presents work whose goal is to advance the field of Machine Learning by proposing a new deep-learning approach for solving stochastic optimal control problems. It has therefore no direct societal impact.

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

Justification: The paper poses no such risks

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: We do not use any existing assets.

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

Justification: The paper does not release for now new assets. The related code will be made open access upon acceptance of the paper.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve any crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve any crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: This research does not involve LLMs as any important, original, or nonstandard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.