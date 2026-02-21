## FEAT: Free energy Estimators with Adaptive Transport

Yuanqi Du 2 ∗ &amp;Jiajun He 1 ∗ , Francisco Vargas 3 , Yuanqing Wang 4 , Carla P. Gomes 2 , José Miguel Hernández-Lobato 1 , Eric Vanden-Eijnden 4 , 5

1 University of Cambridge, 2 Cornell University, 3 Xaira Therapeutics, 4 ML Lab, Capital Fund Management, 5 Courant Institute of Mathematical Sciences, NYU

## Abstract

We present Free energy Estimators with Adaptive Transport (FEAT), a novel framework for free energy estimation-a critical challenge across scientific domains. FEAT leverages learned transports implemented via stochastic interpolants and provides consistent, minimum-variance estimators based on escorted Jarzynski equality and controlled Crooks theorem, alongside variational upper and lower bounds on free energy differences. Unifying equilibrium and non-equilibrium methods under a single theoretical framework, FEAT establishes a principled foundation for neural free energy calculations. Experimental validation on toy examples, molecular simulations, and quantum field theory demonstrates promising improvements over existing learning-based methods. Our PyTorch implementation is available at https://github.com/jiajunhe98/FEAT .

## 1 Introduction

Estimating free energy is fundamental across machine learning (appearing as normalization factors and the model evidence), statistical mechanics (partition functions), chemistry, and biology [Chipot and Pohorille, 2007, Lelièvre et al., 2010, Tuckerman, 2023]. The free energy is expressed as:

<!-- formula-not-decoded -->

where Ω ⊆ R d , U : Ω → R is the energy function, assumed to be such that Z &lt; ∞ , and β = 1 /k B T combines the Boltzmann constant k B and temperature T .

Rather than calculating F directly, one typically estimates the free energy difference between systems (or states) S a and S b with energies U a and U b , which is essential for biological conformational changes, ligand-macromolecule binding, and chemical reaction mechanisms [Wang et al., 2015]:

<!-- formula-not-decoded -->

This computational challenge has driven numerous approaches. Zwanzig [1954] reformulated the problem as importance sampling , where one system serves as the proposal, enabling free energy difference estimation via Monte Carlo sampling. This free energy perturbation (FEP) method, however, suffers from high variance when the energies U a and U b of systems S a and S b differ significantly, particularly in high-dimensional spaces.

To mitigate this issue, targeted FEP [Jarzynski, 2002] learns an invertible mapping between two distributions to increase their overlap. From a complementary angle, Bennett [1976] generalized

∗ The authors contributed equally to this work. The order is randomly assigned and randomly reshuffled in different version of the paper to reflect this equal contribution. Corresponding to &lt;jh2383@cam.ac.uk&gt; , &lt;yuanqidu@cs.cornell.edu&gt; , and &lt;eve2@cims.nyu.edu&gt; .

FEP to create a minimum-variance free energy estimator -the Bennett acceptance ratio (BAR) method. However, BAR still requires sufficient distribution overlap. Shirts and Chodera [2008] addressed this limitation with the multistate Bennett acceptance ratio (MBAR), introducing multiple intermediate system to improve overlap. Thermodynamic integration [TI, Kirkwood, 1935] takes a different approach by constructing a continuous path of systems S t with energies U t , with t ∈ [0 , 1] , connecting S a and S b . The free energy difference is defined by integrating instantaneous energy differences along this path using samples from each intermediate systems S t -effectively performing infinitely many consecutive FEPs between infinitesimally close distributions.

The methods above rely on equilibrium averages, requiring exact samples drawn from the distributions of each considered state. Jarzynski equality [Jarzynski, 1997] offered a breakthrough alternative based on non-equilibrium trajectories. Similar to TI, methods based on this equality utilize a path of intermediate systems S t , but only require exact samples from one endpoint, S a or S b , allowing the law of the transported samples to deviate from the law of the intermediate systems. The escorted Jarzynski equality [Vaikuntanathan and Jarzynski, 2008] further refined this approach by introducing additional control that reduce estimator variance by constraining these deviations.

Recent advances have leveraged the capacity of neural networks to approximate high-dimensional distributions for improved free energy estimation. In Targeted FEP approaches, researchers have developed invertible maps using normalizing flows [Wirnsberger et al., 2020] and flow matching [Zhao and Wang, 2023, Erdogan et al., 2025]. In the context of TI, Máté et al. [2024a,b] introduced energy-parameterized diffusion [Song et al., 2021] and stochastic interpolant models [Albergo et al., 2023] to capture energy interpolants between endpoint distributions.

Despite recent advances, non-equilibrium approaches and their connections to other methods remain under-explored in the deep learning landscape for free energy estimation. Our work addresses this gap by introducing Free energy Estimators with Adaptive Transport (FEAT). Using samples from both endpoints, FEAT constructs non-equilibrium transport via stochastic interpolants. This learned transport is then leveraged through the escorted Jarzynski equality and Crooks theorem to obtain both variational bounds and a consistent, minimum-variance estimator for free energy differences.

FEAT capitalizes on the key advantage of non-equilibrium transport: eliminating the need for exact samples at intermediate distributions, thereby enabling more efficient computation. It facilitates faster numerical simulations without costly divergence evaluations while demonstrating enhanced robustness to discretization and network learning errors. Notably, our framework subsumes existing equilibrium-based methods as special cases, revealing a larger design space with greater flexibility and performance potential. Experimental results confirm that FEAT significantly outperforms leading baselines, including targeted FEP and neural thermodynamic integration.

## 2 Background and Motivation

Here we summarize key free energy estimators most relevant to our approach. While this material is well-established in the computational physics literature, it may be less familiar to machine learning audiences. For the reader's convenience, we provide a more comprehensive discussion and derivations of these results in Appendix B. For simplicity, hereafter, we set the combination of Boltzmann constant and temperature k B T = β -1 = 1 to absorb it into the definitions of the potential and free energy.

## 2.1 Free energy perturbation and Bennett acceptance ratio

Free energy perturbation [FEP, Zwanzig, 1954] estimates free energy differences between systems S a and S b through importance sampling, using samples from one system as proposals for the other. This gives the following expression for the free energy difference ∆ F :

<!-- formula-not-decoded -->

where we use E a to denote the expectation with respect to the equilibrium distribution µ a (d x ) = Z -1 a e -U a ( x ) d x of system S a .

Bennett acceptance ratio [BAR, Bennett, 1976] extends FEP with a minimal variance estimator (specifically, minimal relative mean squared error):

<!-- formula-not-decoded -->

where E a and E b denote expectations with respect to the equilibrium distributions of systems S a and S b , and ϕ ( x ) is any function satisfying ϕ ( x ) /ϕ ( -x ) = e -x : the usual choice is the Fermi function ϕ ( x ) = 1 / (1 + e x ) . The constant C can be optimized to minimizes the variance of the estimator: this gives C = ∆ F . Since ∆ F is unknown initially, in practice it is determined iteratively by updating C to ∆ ˆ F , where ∆ ˆ F is computed from Equation (4) in the previous iteration.

Both FEP and BAR rely on importance sampling and become ineffective when the distributions of systems S a and S b have minimal overlap. Targeted FEP [Jarzynski, 2002] addresses this limitation by designing an invertible transformation T that maps samples between states. The free energy difference is then calculated through importance sampling with change of variable:

<!-- formula-not-decoded -->

where ∇ T ( x ) denotes the Jacobian matrix of the map T and |∇ T ( x ) | its determinant. This approach naturally integrates with neural density estimators. Recent works have implemented this transformation using normalizing flows [Wirnsberger et al., 2020], computing the Jacobian via an invertible network, and flow matching [Lipman et al., 2023, Zhao and Wang, 2023, Erdogan et al., 2025], computing the Jacobian via the instantaneous change of variable formula.

## 2.2 Jarzynski equality, Crooks fluctuation theorem, and their escorted variations

Let U t be a smooth energy interpolating between U t =0 = U a and U t =1 = U b , and consider the stochastic process governed by the Langevin equation over this evolving potential:

<!-- formula-not-decoded -->

where σ t ≥ 0 is the volatility , X 0 ∼ µ a indicates that X 0 is sampled from the distribution µ a (d x ) = Z -1 a e -U a ( x ) d x , and the arrow over the Brownian motion B t indicates that this equation must be solved forward in time. Because U t is time-dependent, the law of X t is not the Gibbs distribution associated with U t . Yet, Jarzynski equality [Jarzynski, 1997] shows that we can correct for these non-equilibrium effects and relate the (equilibrium) free energy difference to the work W :

<!-- formula-not-decoded -->

where E - → P denotes expectation over the path measure - → P of the solutions to Equation (6). Note that, if we set σ t = 0 in Equation (6), the samples do not move so that we have X t = X 0 ∼ µ a and W a → b ( X ) = ∫ 1 0 ∂ t U t ( X t ) d t = U b ( X 0 ) -U a ( X 0 ) , so that Equation (7) reduces to Equation (3). The interest of Equation (7) is that it also works when σ t &gt; 0 .

We can also express and interpret Jarzynski equality through Crooks fluctuation theorem [CFT, Crooks, 1999]. Specifically, consider the following backward SDEs with path measure ← -P :

<!-- formula-not-decoded -->

where X 1 ∼ µ b indicates that X 1 is sampled from the distribution µ b (d x ) = Z -1 b e -U b ( x ) d x and the arrow over the Brownian motion B t indicates that this equation must be solved backward in time. Assuming that σ t &gt; 0 , the Radon-Nikodym derivative between the path measure of forward and the backward processes solutions to Equation (6) and Equation (8), respectively, can be expressed in terms of the free energy difference and the work as

<!-- formula-not-decoded -->

- →

<!-- formula-not-decoded -->

Jarzynski equality can be recovered from this expression by noting that its expectation over P is 1. Vaikuntanathan and Jarzynski [2008] add a control term v t to the drift in Equation (6), whose aim is to better align the law of X t with the Gibbs distribution associated with U t :

Let - → P v be the path measure of the solution to this SDE and define the generalized work W v as:

<!-- formula-not-decoded -->

where ∇· represents the divergence operator, i.e., trace of Jacobian.

Escorted Jarzynski equality expresses the free energy difference in terms this generalized work as:

<!-- formula-not-decoded -->

This expression remains valid if we use σ t = 0 in Equation (10) and it reduces to the ODE

<!-- formula-not-decoded -->

By chain rule, we have ∇ U t ( X t ) · v t ( X t ) + ∂ t U t ( X t ) = (d / d t ) U t ( X t ) , and hence the generalized work in Equation (12) becomes W v ( X ) = -∫ 1 0 ∇· v t ( X t ) d t + U b ( X 1 ) -U a ( X 0 ) . In this case, this expression becomes an implementation of Equation (5) in which we construct the map T via solution of the ODE (13) by setting T ( X 0 ) = X t =1 and hence log |∇ T | = ∫ 1 0 ∇· v t ( X t ) d t . This ODE-based mapping is also known as the continuous normalizing flow [CNF, Chen et al., 2018]. We will come back to this connection in Section 3.5.

It can also be shown [Lelièvre et al., 2010, Heng et al., 2021, Arbel et al., 2021, Vargas et al., 2024, Albergo and Vanden-Eijnden, 2024] that the law of the solution to Equation (6) is precisely µ t (d x ) = Z -1 t e -U t ( x ) d x if and only if v t ( x ) satisfies

<!-- formula-not-decoded -->

In this case, the generalized work defined in Equation (11) is determinitic and given by W v ( X ) = ∫ 1 0 ∂ t F t d t = ∆ F , and hence the escorted Jarzynski equality becomes a practical way to implement thermodynamic integration (TI). We elaborate on this connection in Section 3.5.

When σ t &gt; 0 , we can also establish the controlled Crooks fluctuation theorem [Vargas et al., 2024] by considering the backward SDE:

<!-- formula-not-decoded -->

Denoting the path measure of the solution to this SDE as ← -P v , we have:

<!-- formula-not-decoded -->

The expectation of this expression over - → P v is 1, which recovers Equation (12).

## 3 Methods

To leverage the escorted Jarzynski equality effectively, we need a control term v t that enables Equation (10) to transport samples from S a to S b accurately. Recent neural samplers [Vargas et al., 2024, Albergo and Vanden-Eijnden, 2024] approach this by first defining an energy interpolant (e.g., U t = (1 -t ) U a + tU b ), then optimizing a neural network to learn v t in Equation (10). These methods address the challenging scenario where only samples from one endpoint are available, making their performance sensitive to the choice of interpolating energy U t [Syed et al., 2022, Máté and Fleuret, 2023, Phillips et al., 2024] and requiring Langevin dynamics trajectory simulation during training.

Our setting is different: like BAR and other established methods, we assume that we have access to samples from both systems S a and S b , and our goal is solely to compute the free energy difference between these endpoints. This simplifies matters and renders the specific choice of energy interpolant U t less critical. In particular, it allows us to leverage stochastic interpolants framework [Albergo and Vanden-Eijnden, 2023, Albergo et al., 2023] to develop an efficient and scalable method for simultaneously learning the transport between the two marginal distributions and the associated energy function U t . This learning-based strategy is explained next and summarized in Algorithm 1.

## 3.1 Learning a Transport with Stochastic Interpolants

Given samples from systems S a and S b , stochastic interpolants [Albergo and Vanden-Eijnden, 2023, Albergo et al., 2023] provide a simple and efficient way to effectively learn a transport between these states in the form of Equation (10). We first define an interpolant between endpoint samples:

<!-- formula-not-decoded -->

where α 0 = 1 , α 1 = 0 ; β 0 = 0 , β 1 = 1 ; and γ 0 = γ 1 = 0 ensure proper boundary conditions: I t =0 = x a and I t =1 = x b . From the results of Albergo et al. [2023], we know that the law of I t is, at any time t ∈ [0 , 1] , the same as the law of the solution to Equation (10) if we use

<!-- formula-not-decoded -->

where the dot denotes the time derivative and E [ ·| I t = x ] denotes expectation over the law of I t conditional on I t = x . Using the L 2 formulation of the conditional expectation, we can write objective functions for the function v t and ∇ U t defined in Equation (18); if we parametrize these functions as neural networks v ψ t ( x ) and U θ t ( x ) , depending on both t and x , this leads to the losses:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where DSM stands for denoising score matching [Vincent, 2011], and λ t and η t are weighting functions to balance optimization across different times. In practice, λ t = 1 and η t = γ t work well.

Once we have learned v ψ t and ∇ U θ t we can use these functions in any of the estimators presented in Section 2 via computation of the generalized work in Equation (11) or the forward-backward Radon-Nikodym derivative (FB RND) in Equation (16). Note that, Equation (11) requires ∂ t U t , necessitating an energy-parameterized network U θ t , which is known to be difficult to learn via score matching [Zhang et al., 2022]. In contrast, the FB RND form only requires the score function, allowing direct parameterization of ∇ U θ t as a score network. In fact, using FB RND formulation also offers additional benefits, which we will discuss in Section 3.4.

We stress that the estimators presented in Section 2 are asymptotically unbiased even if we use them with functions v ψ t and U θ t that have been learned imperfectly, as long as we satisfy the boundary conditions U θ t =0 = U a and U θ t =1 = U b . In practice, however, imposing these boundary conditions puts constraints on the neural network used to parameterize U θ t or ∇ U θ t , which may limit its expressivity and impede training convergence. For these reasons, in FEAT we choose to not impose the boundary conditions by the parameterization design, but rather learn U θ t at the endpoints as well. To this end, we enhance the denoising score matching loss in Equation (20) with target score matching [TSM, De Bortoli et al., 2024]:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This objective formulation was introduced by Máté et al. [2024b] to improve energy estimation accuracy in neural thermodynamic integration (TI). Importantly, TSM does not increase target energy evaluation costs, as the gradients ∇ U a ( x a ) and ∇ U b ( x b ) can be precomputed and stored during molecular dynamics (MD) simulations alongside collected samples.

Unfortunately, even though TSM largely increases the accuracy of the boundary conditions, error still exists due to imperfect learning. Next, we discuss how to generalize the escorted Jarzynski estimators in Section 2 to account for the mismatch between U θ t and U a and U b at the endpoints.

## 3.2 Estimating the Free Energy Difference with Escorted Jarzynski Equality

̸

Suppose that we have learned a transport with stochastic interpolants for which the boundary conditions are not necessarily satisfied, meaning U θ 0 = U a and U θ 1 = U b . This boundary mismatch requires a correction term, as specified by the following result:

̸

Corollary 3.1 (Escorted Jarzynski Equality with imperfect boundary conditions) . Given v ψ t and U θ t , consider the forward and backward SDEs:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where σ t ≥ 0 and µ a and µ b denote the distributions associated with the energies U a and U b , respectively. Define also the 'corrected generalized work":

<!-- formula-not-decoded -->

Using the generalized work with correction, we have the same escorted Jarzynski equality as before:

<!-- formula-not-decoded -->

where - → P v and ← -P v are the path measures over the solutions to Equations (23) and (24) , respectively.

The proof of this proposition is given in Appendix C.3. Note that the corrected generalized work in Equation (25) coincides with the generalized work in Equation (11) if U θ 0 = U a and U θ 1 = U b , but we stress again the proposition remains valid even if these equalities do not hold.

The escorted Jarzynski equality in Equation (26) can be used to estimate the free energy difference: Denoting by - → X (1: N ) ∼ - → P v and ← -X (1: N ) ∼ ← -P v N independent realizations of the solutions to the forward Equation (23) and the backward Equation (24), respectively, we have

<!-- formula-not-decoded -->

and these expressions become unbiased in the limit as N →∞ . These finite sample size estimators coincide with the importance-weighted autoencoder [IWAE, Burda et al., 2015], and they give us bounds on the free energy difference in expectation:

<!-- formula-not-decoded -->

These bounds are much tighter in general than the usual variational bounds:

<!-- formula-not-decoded -->

which are also known as the evidence lower and upper bounds (ELBO and EUBO) in variational inference [Blei et al., 2017, Ji and Shen, 2019, Blessing et al., 2024] and were applied to free energy estimation by Wirnsberger et al. [2020], Zhao and Wang [2023] in their targeted FEP. We prove the IWAE bounds and the variational bounds in Appendix C.1.

## 3.3 Minimizing Variance with Non-equilibrium Bennett Acceptance Ratio

Equation (27) provides two estimators for the free energy difference. While simply averaging them can reduce variance, we can achieve optimal variance reduction using an approach similar to Bennett's Acceptance Ratio [BAR, Bennett, 1976].

Proposition 3.2 (Minimum variance non-equilibrium free energy estimator) . Let ˜ W v be the work terms defined in Corollary 3.1. The minimum-variance estimator is given by:

<!-- formula-not-decoded -->

where ϕ is the Fermi function ϕ ( z ) = 1 / (1 + exp( z )) . In addition, the minimum variance estimator is obtained with C = ∆ F .

In practice, we initialize C as the mean of Equation (27), then iteratively update ∆ F using C set to the current estimate until convergence. This estimator was originally introduced by Bennett [1976] for equilibrium averages, with non-equilibrium variants later developed by Shirts et al. [2003], Hahn and Then [2009], Minh and Chodera [2009], Vaikuntanathan and Jarzynski [2011] using work likelihood or 'density of trajectory" concepts. In Appendix C.2, we provide an alternative derivation based on the Radon-Nikodym derivative between path measures.

## 3.4 Improving Accuracy and Efficiency of Free Energy Estimation with FB RND

We now turn our attention to estimators of the free energy difference using forward-backward Radon-Nikodym derivative (FB RND), which is enabled by the (controlled) Crooks fluctuation theorem. This approach allows us to address an issue we have left open: in practice, numerical integration of Equations (23) and (24) requires time discretization, introducing additional error. We demonstrate below that FB RND-based estimators yield asymptotically unbiased estimates of free energy differences, even in the presence of this discretization error.

We first describe the calculation of the 'corrected generalized work" with discretized FB RND: let us discretize the time interval [0 , 1] into M steps t 0 = 0 &lt; t 1 &lt; · · · &lt; t M -1 &lt; t M = 1 with step size ∆ t , and denote by N + and N -the transition kernels under Euler-Maruyama discretization for the forward and backward SDEs in Equations (23) and (24):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The 'corrected generalized work" in the FB RND form can be calculated as:

<!-- formula-not-decoded -->

Even though the quantities at the right hand-sides are only approximate expressions for those at the left hand-side, they give consistent estimators for the free energy:

Proposition 3.3 (Discretized controlled Crooks theorem with imperfect boundary conditions) . Let N + and N -be as in Equations (31) and (32) , and define the forward/backward discretized paths:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we assume that σ t &gt; 0 , ∆ t i = t i +1 -t i and η i ∼ N (0 , Id ) , independent for each i = 0 , . . . , M . Then, we have:

<!-- formula-not-decoded -->

̸

The proof is given in Appendix C.3. This proposition shows that the FB RND yields estimators that are asymptotically unbiased in the limit of infinite sample size, even when the time-step size is finite . Furthermore, imperfect boundary conditions do not alter the form of the FB RND; Equation (36) hold whether U θ 0 = U a , U θ 1 = U b or U θ 0 = U a , U θ 1 = U b . This results in a more compact formulation for calculating the generalized work compared to Equation (25).

̸

The FB RND approach also offers another advantage in terms of computational efficiency as it allows direct parameterization of the score ∇ U θ t without divergence calculations, eliminating backpropagation needs during training and sampling, thus enhancing efficiency and scalability. In contrast, the calculation based on Equation (25) requires both time derivative ∂ t U θ t and divergence ∇· v ψ t , complicating training and sampling.

In summary, to estimate the free energy difference between two systems S a and S b , our approach learns stochastic interpolants using their samples. This yields forward and backward SDEs, as detailed in Equations (23) and (24), with potentially imperfect boundary conditions as discussed, which approximately transport samples between the two states. We then map samples from both states using these SDEs, and compute the 'generalized work" via FB RND as in Equation (33). This computation results in a consistent, minimal-variance estimator for the free energy difference as defined in Equation (30). Since our method relies on the escorted Jarzynski equality with a learned escorting term adaptive to the data, we dub it the Free energy Estimators with Adaptive Transport (FEAT). An outline of FEAT is provided in Algorithm 1.

## 3.5 Limiting Cases of FEAT with Connections to Other Approaches

Our algorithm generalizes several existing approaches and reduces to them under specific conditions. We illustrate these relationships in Figure 1 and elaborate below, focusing on connections beyond the already established link to the (escorted) Jarzynski equality.

FEP and Target FEP . Our method generalizes targeted FEP with flow matching [Zhao and Wang, 2023] when setting σ t = 0 . Specifically, escorted Jarzynski equality becomes equivalent to targeted FEP with instantaneous change of variables [Chen et al., 2018] in this deterministic limit, even with imperfect boundary conditions as in Corollary 3.1. We show this equivalence in Appendix C.4. Taking further reduction, our method also generalizes standard FEP when both σ t = 0 and v t = 0 .

Figure 1: Connection between our proposed algorithm and other free-energy estimation approaches.

<!-- image -->

Table 1: Comparative free energy difference ( ∆ F , unitless) estimation using targeted FEP [Zhao and Wang, 2023] and our proposed approach. MBAR results serve as reference values for LJ and ALDP systems. We report the mean and std across three runs with different seeds. (*) indicates prohibitively expensive computation due to divergence operations.

| Method         | GMM (16 modes to 40 modes)   | GMM (16 modes to 40 modes)   | LJ (w/o to w. LJ interaction)   | LJ (w/o to w. LJ interaction)   | LJ (w/o to w. LJ interaction)   | ALDP -S                    | ALDP -T                       |
|----------------|------------------------------|------------------------------|---------------------------------|---------------------------------|---------------------------------|----------------------------|-------------------------------|
|                | d = 40                       | d = 100                      | d = 55 × 3                      | d = 79 × 3                      | d = 128 × 3                     | d = 22 × 3                 | d = 22 × 3                    |
| Reference      | 0                            | 0                            | 234.77 ± 0 . 09 232.06 ± 0 .    | 357.43 ± 3 . 43 *               | 595.98 ± 0 . 58 *               | 29.43 ± 0 . 01 29.47 ± 0 . | -4.25 ± 0 . 05 -4.78 ± 0 . 32 |
| TargetFEP (FM) | 0.09 ± 0 . 26                | -17.96 ± 1 . 49              | 03                              |                                 |                                 | 22                         |                               |
| FEAT (ours)    | 0.04 ± 0 . 04                | -5.34 ± 1 . 52               | 232.47 ± 0 . 15                 | 356.74 ± 0 . 79                 | 595.04 ± 6 . 52                 | 29.38 ± 0 . 04             | -4.56 ± 0 . 08                |

In this scenario, the dynamic transport vanishes completely, and we revert to simple importance sampling using equilibrium samples from both endpoints.

Bennett Acceptance Ratio (BAR). Similar to FEP, our method recovers BAR when setting both σ t = 0 and v t = 0 and applying the minimum-variance estimator in Equation (30).

Thermodynamic Integration (TI). When U θ t and v ψ t are optimally learned, our method recovers TI. Specifically, this occurs when the distribution of X t simulated from Equation (23) exactly matches the density defined by energy U t . We derive this equivalence in Appendix C.5.

This connection reveals a limitation of TI: the energy function must precisely match the sample distribution. In neural TI [Máté et al., 2024a,b], the energy network must accurately capture the data density at every time step t , or the estimator becomes significantly biased. Our approach, based on escorted Jarzynski equality which accommodates non-equilibrium trajectories, remains effective even with imperfect learning, similar to Vargas et al. [2024], Albergo and Vanden-Eijnden [2024].

## 3.6 One-sided FEAT

In the last section, we focus on estimating the free energy difference between two states by learning a stochastic interpolant model. Notably, the same formulation also applies to estimating the absolute free energy, by choosing one of the states to be a reference distribution with known free energy, such as a Gaussian. In this case, we do not need to learn both the vector field and the score independently, as they are related in closed form [Albergo et al., 2023]. In fact, in this case, the one-sided stochastic interpolant recovers a diffusion model [Song et al., 2021]. The diffusion model learns a score network s θ t , from which one can easily recover the vector field. We refer to this variant as one-sided FEAT .

To estimate the free energy difference between two arbitrary states, one can also apply one-sided FEAT to each state and then compute the difference between their estimated absolute free energies. This formulation can be viewed as an extension of DeepBAR [Ding and Zhang, 2021], which estimates absolute free energies using a normalizing flow with the BAR equation. In contrast, our approach replaces the normalizing flow with FEAT, enabling more flexible and scalable modeling. Empirically, we found DeepBAR with one-sided FEAT achieves better performance compared to using one FEAT directly bridging two states, especially on large systems. A potential reason is learning the transport from a Gaussian distribution to a complex target is simpler than learning it directly between two different complex targets.

Figure 2: (a) Two states of ALDP-T. S a : ϕ ∈ (0 , 2 . 15) ; S b : ϕ ∈ [ -π, 0] ∪ [0 , π ) ; (b) (c) Estimators with eqs. (27) and (30) and their dynamics along training in LJ-128 and ALDP-S.

<!-- image -->

Figure 3: Eight example lattice configurations. We can see that the lattices are either mostly positive or negative.

<!-- image -->

## 4 Experiments

We evaluate FEAT on a diverse range of systems, from toy examples to molecular simulations and quantum field theory. Detailed experimental setups and hyperparameters appear in Appendix F.

Comparison with Target FEP. We benchmark our approach against targeted FEP with flow matching [Zhao and Wang, 2023] across four systems of increasing complexity: (1) Gaussian mixtures, (2) Lennard-Jones potentials with varying parameters, (3) alanine dipeptide in vacuum vs. implicit solvent (ALDP-S), and (4) alanine dipeptide in two metastable phases (ALDP-T, illustrated in Figure 2a). For fair comparison, all methods use identical model architectures, and we apply the minimum-variance estimator to both targeted FEP and our approach. Reference values for the last three systems are obtained using MBAR [Shirts and Chodera, 2008]. Results in Table 1 demonstrate that our approach consistently outperforms Target FEP. Our method's advantage over targeted FEP likely stems from the latter's reliance on instantaneous changes of variables, making it more susceptible to discretization errors. As discussed in Section 3.4, our approach offers inherent robustness to such errors while also eliminating divergence calculations for improved computational efficiency.

Comparison with Neural TI [Máté et al., 2024a,b]. We report the results of Neural TI on GMM-40 and LJ-79. For the accuracy of Neural TI, it is crucial to ensure the learned energy matches the sample distribution along the entire interpolant path. Therefore, Máté et al. [2024b] proposed to parameterize the energy network using the energy of state A and B as preconditioning. To have a fair comparison, we report Neu-

Table 2: Neural TI with and without preconditioning.

|                        | GMM-40                          | LJ-79                             |
|------------------------|---------------------------------|-----------------------------------|
| w/ Precond w/o Precond | 0 . 1 ± 0 . 2 - 181 . 6 ± 6 . 7 | 356 . 9 ± 1 . 8 468 . 8 ± 391 . 2 |

ral TI both with and without preconditioning. Details of the preconditioning design are provided in Appendix F.5. Table 2 shows that Neural TI relies heavily on such problem-specific preconditioning, which constrains flexibility and can be costly when the energy evaluation is expensive.

Different estimators and training dynamics. We visualize the estimates using only forward or backward simulation in Equation (27), and the minimum-variance form in Equation (30) throughout training for LJ-128 and ALDP in Figure 2. Our method converges rapidly on both systems, with the minimum-variance estimator clearly outperforming the forward or backward-only estimators.

Application on umbrella sampling. A valuable application of our method is umbrella sampling for free energy surface estimation (potential of mean force) in collective variable (CV) space. Traditionally addressed via weighted histogram analysis method [WHAM, Kumar et al., 1992], this

Figure 4: Umbrella samples and reweighted histogram. We denote the two umbrellas as u a and u b .

<!-- image -->

approach requires defining a sequence of 'umbrellas" by adding harmonic potentials along the CV dimension to the target potential, then sampling from these umbrellas using MCMC. To correctly aggregate samples from different umbrellas projected onto CV space, we must estimate relative free energies between umbrella potentials for proper reweighting. Our approach integrates naturally into this pipeline by efficiently estimating free energy differences between umbrella pairs.

We demonstrate this with φ 4 quantum field theory, also studied in Albergo and Vanden-Eijnden [2024] for sampling tasks. The variables are field configurations φ ∈ R L × L and we estimate the average magnetization histogram (see Appendix F.1.5 for energy details). The lattice exists in an ordered phase where neighboring sites correlate with the same sign and magnitude, creating the bimodal distribution shown in Figure 3. We estimate the magnetization histogram by performing two umbrella sampling runs-one biased toward negative magnetization and another toward positive magnetization-then combine them by reweighting according to the free energy difference estimated by our method (calculation details in Appendix F.1.5). As illustrated in Figure 4, the reweighted average magnetization distribution successfully recovers the symmetrical nature of the φ 4 energy.

One-sided FEAT and large-scale experiments. We evaluate FEAT on two larger systems-alanine tetrapeptide (ALA-4) and Chignolin-to estimate the solvation free energy. The standard stochastic interpolant struggles to fit larger molecular systems, so we adopt one-sided FEAT to learn the absolute free energies of each system and take their difference, as described in Section 3.6. Leveraging the diffusion-model design of

Table 3: One-sided FEAT on ALA-4 and CHIG.

|           | ALA-4             | CHIG              |
|-----------|-------------------|-------------------|
| Reference | 107.5             | 320.19            |
| FEAT      | 109 . 91 ± 2 . 55 | 320 . 02 ± 0 . 70 |

Karras et al. [2022], our model fits both systems well. As shown in Table 3, the proposed FEAT delivers accurate predictions, demonstrating its scalability and strong potential.

Runtime discussion . We report FEAT's inference time in Table 4. Relative to Target FEP (FM), FEAT avoids costly divergence evaluations, significantly improving efficiency.

## 5 Conclusions and Limitations

Free energy difference estimation between states remains a fundamental challenge with wide-ranging applications, yet research in the modern machine learning context has predominantly focused on equilibrium approaches, leaving non-equilibrium methods largely unexplored. Our Free Energy Estimators with Adaptive Transport (FEAT) address this gap by leveraging the stochastic interpolant framework to learn transports that permit both equilibrium and non-equilibrium estimation of the free energy difference through the escorted Jarzynski equality and the Crooks theorem.

One caveat of FEAT is the variance of estimator can be large even with the minimum variance estimator, especially for larger-scale systems. This is because FEAT is based on importance sampling over the path space , while target FEP is based on importance sampling in the state space . By the data-processing inequality, the overlap between two distributions will not decrease when lifted from state space to path space. Therefore, FEAT and Target FEP with flow matching may be understood as different points on a bias-variance spectrum: FEAT is asymptotically unbiased but tends to exhibit higher variance. Recent work by Schebek et al. [2025] evaluated FEAT and Target FEP for condensed-phase systems, and highlights this point with a detailed discussion.

Our current approach requires access to samples from both states of interest. Future work could explore approaches that relax this requirement, such as those proposed by Vargas et al. [2024], Albergo and Vanden-Eijnden [2024], potentially enabling free energy estimation in settings with limited sampling access. However, these sampling techniques still face notable challenges in scalability, stability, and mode collapsing. How to resolve these issues efficiently remains an open problem. Investigating the generalizability of our approach is another promising direction, potentially with transferable networks as demonstrated by Klein and Noé [2024]. We present a primary demonstration on a toy example in Appendix E.3, and leave molecular systems exploration to future works.

Finally, recent work has investigated the computational complexity of the Jarzynski equality [Guo et al., 2025]. Extending their analysis to our framework would provide additional insight into requirements for reliable estimation.

## Broader Impact

This work aims to estimate the free energy difference, a core quantity in studying chemical reactions, phase transitions, and biomolecular conformational changes. We expect it to have positive impacts in accelerating drug and materials discovery. However, more efficient free energy estimation also carries the risk of enabling the development of harmful chemicals or toxins. We therefore advocate for their responsible use.

## Acknowledgments

We thank Yanze Wang for the discussions on MBAR setup and John D. Chodera for pointing us to relevant history of MBAR. JH acknowledges support from the University of Cambridge Harding Distinguished Postgraduate Scholars Programme. CPG and YD acknowledge the support of Schmidt Sciences programs, an AI2050 Senior Fellowship and Eric and Wendy Schmidt AI in Science Postdoctoral Fellowships; the National Science Foundation (NSF); the National Institute of Food and Agriculture (USDA/NIFA); the Air Force Office of Scientific Research (AFOSR). YW acknowledges support from the Schmidt Science Fellowship, in partnership with the Rhodes Trust, as well as the Simons Center for Computational Physical Chemistry at New York University. YW thanks the highperformance computing resources at Memorial Sloan Kettering Cancer Center and the Washington Square and Abu Dhabi campuses of New York University-we are especially grateful to the technical supporting teams. YW has limited financial interests in Flagship Pioneering, Inc. and its subsidiaries. JMHL acknowledges support from a Turing AI Fellowship under grant EP/V023756/1.

## References

- M. S. Albergo and E. Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In 11th International Conference on Learning Representations, ICLR 2023 , 2023.
- M. S. Albergo and E. Vanden-Eijnden. Nets: A non-equilibrium transport sampler. arXiv preprint arXiv:2410.02711 , 2024.
- M. S. Albergo, N. M. Boffi, and E. Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797 , 2023.
- M. Arbel, A. Matthews, and A. Doucet. Annealed flow transport monte carlo. In International Conference on Machine Learning , pages 318-330. PMLR, 2021.
- C. H. Bennett. Efficient estimation of free energy differences from monte carlo data. Journal of Computational Physics , 22(2):245-268, 1976. ISSN 0021-9991. doi: https://doi.org/10.1016/ 0021-9991(76)90078-4. URL https://www.sciencedirect.com/science/article/pii/ 0021999176900784 .
- D. M. Blei, A. Kucukelbir, and J. D. McAuliffe. Variational inference: A review for statisticians. Journal of the American statistical Association , 112(518):859-877, 2017.
- D. Blessing, X. Jia, J. Esslinger, F. Vargas, and G. Neumann. Beyond elbos: A large-scale evaluation of variational methods for sampling. arXiv preprint arXiv:2406.07423 , 2024.
- Y. Burda, R. Grosse, and R. Salakhutdinov. Importance weighted autoencoders. arXiv preprint arXiv:1509.00519 , 2015.
- R. T. Chen, Y. Rubanova, J. Bettencourt, and D. K. Duvenaud. Neural ordinary differential equations. Advances in neural information processing systems , 31, 2018.
- C. Chipot and A. Pohorille. Free energy calculations , volume 86. Springer, 2007.
- J. Chodera, A. Rizzi, L. Naden, K. Beauchamp, P. Grinaway, M. Henry, I. Pulido, J. Fass, A. Wade, G. A. Ross, A. Kraemer, H. B. Macdonald, jaimergp, B. Rustenburg, D. W. Swenson, I. Zhang, D. Rufa, A. Simmonett, M. J. Williamson, hb0402, J. Fennick, S. Roet, B. Ries, I. Kenney, I. Alibay, R. Gowers, and SimonBoothroyd. choderalab/openmmtools: 0.24.1, Jan. 2025. URL https://doi.org/10.5281/zenodo.14782825 .

- G. E. Crooks. Entropy production fluctuation theorem and the nonequilibrium work relation for free energy differences. Physical Review E , 60(3):2721, 1999.
- V. De Bortoli, M. Hutchinson, P. Wirnsberger, and A. Doucet. Target score matching. arXiv preprint arXiv:2402.08667 , 2024.
- X. Ding and B. Zhang. Deepbar: A fast and exact method for binding free energy computation. The Journal of Physical Chemistry Letters , 12(10):2509-2515, 2021. doi: 10.1021/acs.jpclett.1c00189. URL https://doi.org/10.1021/acs.jpclett.1c00189 . PMID: 33719449.
- W. Du, H. Zhang, Y. Du, Q. Meng, W. Chen, N. Zheng, B. Shao, and T.-Y. Liu. Se (3) equivariant graph neural networks with complete local frames. In International Conference on Machine Learning , pages 5583-5608. PMLR, 2022.
- P. Eastman, R. Galvelis, R. P. Peláez, C. R. Abreu, S. E. Farr, E. Gallicchio, A. Gorenko, M. M. Henry, F. Hu, J. Huang, et al. Openmm 8: molecular dynamics simulation with machine learning potentials. The Journal of Physical Chemistry B , 128(1):109-116, 2023.
- E. Erdogan, R. Ralev, M. Rebensburg, C. Marquet, L. Klein, and H. Stark. Freeflow: Latent flow matching for free energy difference estimation. 2025.
- W. Guo, M. Tao, and Y. Chen. Complexity analysis of normalizing constant estimation: from jarzynski equality to annealed importance sampling and beyond. arXiv preprint arXiv:2502.04575 , 2025.
- A. M. Hahn and H. Then. Characteristic of bennett's acceptance ratio method. Phys. Rev. E , 80: 031111, Sep 2009. doi: 10.1103/PhysRevE.80.031111. URL https://link.aps.org/doi/10. 1103/PhysRevE.80.031111 .
- J. Heng, A. Doucet, and Y. Pokern. Gibbs flow for approximate transport with applications to bayesian computation. Journal of the Royal Statistical Society Series B: Statistical Methodology , 83(1):156-187, 2021.
- J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems , 33:6840-6851, 2020.
- C. Jarzynski. Nonequilibrium equality for free energy differences. Physical Review Letters , 78(14): 2690, 1997.
- C. Jarzynski. Targeted free energy perturbation. Physical Review E , 65(4):046122, 2002.
- C. Ji and H. Shen. Stochastic variational inference via upper bound. arXiv preprint arXiv:1912.00650 , 2019.
- W. Kabsch. A solution for the best rotation to relate two sets of vectors. Acta Crystallographica Section A , 32(5):922-923, 1976. doi: https://doi.org/10.1107/S0567739476001873. URL https: //onlinelibrary.wiley.com/doi/abs/10.1107/S0567739476001873 .
- T. Karras, M. Aittala, T. Aila, and S. Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems , 35:26565-26577, 2022.
- J. G. Kirkwood. Statistical mechanics of fluid mixtures. The Journal of chemical physics , 3(5): 300-313, 1935.
- L. Klein and F. Noé. Transferable boltzmann generators. arXiv preprint arXiv:2406.14426 , 2024.
- L. Klein, A. Krämer, and F. Noé. Equivariant flow matching. Advances in Neural Information Processing Systems , 36:59886-59910, 2023.
- H. W. Kuhn. The hungarian method for the assignment problem. Naval Research Logistics (NRL) , 52, 1955. URL https://api.semanticscholar.org/CorpusID:9426884 .
- H. W. Kuhn. Variants of the hungarian method for assignment problems. Naval Research Logistics Quarterly , 3(4):253-258, 1956. doi: https://doi.org/10.1002/nav.3800030404. URL https: //onlinelibrary.wiley.com/doi/abs/10.1002/nav.3800030404 .

- S. Kumar, J. M. Rosenberg, D. Bouzida, R. H. Swendsen, and P. A. Kollman. The weighted histogram analysis method for free-energy calculations on biomolecules. i. the method. Journal of Computational Chemistry , 13(8):1011-1021, 1992. doi: https://doi.org/10.1002/jcc.540130812. URL https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.540130812 .
- T. Lelièvre, M. Rousset, and G. Stoltz. Free Energy Computations: A Mathematical Perspective . World Scientific, 2010.
- Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations , 2023.
- B. Máté and F. Fleuret. Learning interpolations between boltzmann densities. arXiv preprint arXiv:2301.07388 , 2023.
- B. Máté, F. Fleuret, and T. Bereau. Neural thermodynamic integration: Free energies from energybased diffusion models. The Journal of Physical Chemistry Letters , 15(45):11395-11404, 2024a.
- B. Máté, F. Fleuret, and T. Bereau. Solvation free energies from neural thermodynamic integration. arXiv preprint arXiv:2410.15815 , 2024b.
7. X.-L. Meng and W. H. Wong. Simulating ratios of normalizing constants via a simple identity: a theoretical exploration. Statistica Sinica , pages 831-860, 1996.
- L. I. Midgley, V. Stimper, G. N. Simm, B. Schölkopf, and J. M. Hernández-Lobato. Flow annealed importance sampling bootstrap. In The Eleventh International Conference on Learning Representations , 2023.
- D. D. Minh and J. D. Chodera. Optimal estimators and asymptotic variances for nonequilibrium path-ensemble averages. The Journal of chemical physics , 131(13), 2009.
- A. Phillips, H.-D. Dau, M. J. Hutchinson, V. De Bortoli, G. Deligiannidis, and A. Doucet. Particle denoising diffusion sampler. In International Conference on Machine Learning , pages 4068840724. PMLR, 2024.
- V. G. Satorras, E. Hoogeboom, and M. Welling. E (n) equivariant graph neural networks. In International conference on machine learning , pages 9323-9332. PMLR, 2021.
- M. Schebek, J. He, E. Hoffmann, Y. Du, F. Noé, and J. Rogal. Assessing generative modeling approaches for free energy estimates in condensed matter. arXiv preprint arXiv:2512.23930 , 2025.
- M. R. Shirts and J. D. Chodera. Statistically optimal analysis of samples from multiple equilibrium states. The Journal of chemical physics , 129(12), 2008.
- M. R. Shirts, E. Bair, G. Hooker, and V. S. Pande. Equilibrium free energies from nonequilibrium measurements using maximum-likelihood methods. Physical review letters , 91(14):140601, 2003.
- Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations , 2021.
- S. Syed, A. Bouchard-Côté, G. Deligiannidis, and A. Doucet. Non-reversible parallel tempering: a scalable highly parallel mcmc scheme. Journal of the Royal Statistical Society Series B: Statistical Methodology , 84(2):321-350, 2022.
- Z. Tan. On a likelihood approach for monte carlo integration. Journal of the American Statistical Association , 99(468):1027-1036, 2004. ISSN 01621459.
- Y. Tian, N. Panda, and Y. T. Lin. Liouville flow importance sampler. In Proceedings of the 41st International Conference on Machine Learning , pages 48186-48210, 2024.
- A. Tong, K. Fatras, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, G. Wolf, and Y. Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. Transactions on Machine Learning Research , pages 1-34, 2024.

- M. E. Tuckerman. Statistical mechanics: theory and molecular simulation . Oxford university press, 2023.
- S. Vaikuntanathan and C. Jarzynski. Escorted free energy simulations: Improving convergence by reducing dissipation. Physical Review Letters , 100(19):190601, 2008.
- S. Vaikuntanathan and C. Jarzynski. Escorted free energy simulations. The Journal of chemical physics , 134(5), 2011.
- F. Vargas, S. Padhy, D. Blessing, and N. Nüsken. Transport meets variational inference: Controlled monte carlo diffusions. The Twelfth International Conference on Learning Representations , 2024.
- P. Vincent. A connection between score matching and denoising autoencoders. Neural computation , 23(7):1661-1674, 2011.
- L. Wang, Y. Wu, Y. Deng, B. Kim, L. Pierce, G. Krilov, D. Lupyan, S. Robinson, M. K. Dahlgren, J. Greenwood, et al. Accurate and reliable prediction of relative ligand binding potency in prospective drug discovery by way of a modern free-energy calculation protocol and force field. Journal of the American Chemical Society , 137(7):2695-2703, 2015.
- P. Wirnsberger, A. J. Ballard, G. Papamakarios, S. Abercrombie, S. Racanière, A. Pritzel, D. Jimenez Rezende, and C. Blundell. Targeted free energy estimation via learned mappings. The Journal of Chemical Physics , 153(14), 2020.
- M. Zhang, O. Key, P. Hayes, D. Barber, B. Paige, and F.-X. Briol. Towards healing the blindness of score matching. arXiv preprint arXiv:2209.07396 , 2022.
- L. Zhao and L. Wang. Bounding free energy difference with flow matching. Chinese Physics Letters , 40(12):120201, 2023.
- R. W. Zwanzig. High-temperature equation of state by a perturbation method. i. nonpolar gases. The Journal of Chemical Physics , 22(8):1420-1426, 1954.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We proposed Free energy Estimators with Adaptive Transport (FEAT), based on learning stochastic interpolants and Escorted Jarzynski equality, showing promising improvement over previous neural network-based free energy estimation methods. Our abstract and introduction accurately reflect the paper's contributions and scope.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: See Section 5.

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

Justification: In Section C.

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

Justification: Information on targets, hyperparameters and baseline settings can be found in Section F.

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

Justification:

https://github.com/jiajunhe98/FEAT

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

## Answer: [Yes]

Justification: Information on targets, hyperparameters, and baseline settings can be found in Section F.

## Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We repeat each setting 3 times and report mean &amp; std.

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

Justification: The details on computing resources used can be found in F.4.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We confirm the research has no ethical concerns.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: On Page 11.

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

Justification: We rely on our implementation. We cite and state the license for the used packages in Section F.

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

Answer: [Yes]

Justification: The dataset used will be released along the code.

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

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## FEAT: Free energy Estimators with Adaptive Transport Appendix

| A   | Algorithm                                              | Algorithm                                                                                                                                                         |   23 |
|-----|--------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
| B   | Extended review of free energy estimation methods      | Extended review of free energy estimation methods                                                                                                                 |   23 |
| C   | Proofs                                                 | Proofs                                                                                                                                                            |   24 |
|     | C.1                                                    | Variational Bounds (Equation (29)) and IWAE Bounds (Equation (28)) . . . . . . .                                                                                  |   24 |
|     | C.2                                                    | Minimum Variance Non-equilibrium Free Energy Estimator (Proposition 3.2) . . .                                                                                    |   26 |
|     | C.3                                                    | Escorted Jarzynski and Controlled Crooks with Imperfect Boundary Conditions (Corollary 3.1 and Proposition 3.3) . . . . . . . . . . . . . . . . . . . . . . . . . |   29 |
|     | C.4                                                    | Escorted Jarzynski Equality with ODE Transport . . . . . . . . . . . . . . . . . .                                                                                |   31 |
|     | C.5                                                    | Equivalence between TI and Our Approach with Perfect Transport . . . . . . . . .                                                                                  |   32 |
| D   | SE(3)-equivariant and -invariant Graph Neural Networks | SE(3)-equivariant and -invariant Graph Neural Networks                                                                                                            |   32 |
| E   | Additional Experimental Results                        | Additional Experimental Results                                                                                                                                   |   33 |
|     | E.1                                                    | Runtime Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                            |   33 |
|     | E.2                                                    | Robustness of FEAT . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                              |   33 |
|     | E.3                                                    | Demonstration of Transferable FEAT . . . . . . . . . . . . . . . . . . . . . . . . .                                                                              |   34 |
| F   | Additional Experimental Details                        | Additional Experimental Details                                                                                                                                   |   34 |
|     | F.1                                                    | Systems . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                           |   34 |
|     | F.2                                                    | Hyperparameter . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                            |   36 |
|     | F.3                                                    | Hyperparameters and Settings for One-sided FEAT . . . . . . . . . . . . . . . . .                                                                                 |   36 |
|     | F.4                                                    | Computing Resources . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                             |   37 |
|     | F.5                                                    | Baseline and Reference Settings . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                             |   37 |

## A Algorithm

Algorithm 1 Free energy estimation with FEAT

Input: Energy function of both states U a and U b and their samples x (1: N ) a ∼ µ a , x (1: N ) b ∼ µ b ; learning rate η lr, SDE solver step size ∆ t .

Output: Free energy difference estimator ∆ ˆ F ≈ -log Z b /Z a .

# initialization:

t t t # training:

Randomly initialize vector field u ψ and score network s θ (or energy network U θ );

<!-- formula-not-decoded -->

Simulate the forward eq. (34) from - → X (1: N ) 0 = x (1: N ) a ; calculate ˜ W v ( - → X (1: N ) ) by eq. (33); Simulate the backward eq. (35) from ← -X (1: N ) 1 = x (1: N ) b ; calculate ˜ W v ( ← -X (1: N ) ) by eq. (33); # estimation:

initialize C

repeat calculate ∆ F with eq. (30); and set C ← ∆ F ; until convergence.

to be the average value of the estimators in eq. (27); ˆ ˆ

## B Extended review of free energy estimation methods

Here, we provide an extended review of free energy estimation methods. This section complements the background in Section 2 by offering more details on FEP, target FEP, and BAR, along with a review of MBAR and TI.

Free energy perturbation [FEP, Zwanzig, 1954] leverages importance sampling to estimate free energy difference:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where E a denotes expectation w.r.t. the equilibrium distribution µ a (d x ) = Z -1 a e -U a ( x ) d x of state S a , and p a is the density of µ a .

Bennett acceptance ratio [BAR, Bennett, 1976] extends the FEP equation into a general form:

<!-- formula-not-decoded -->

where it has been proven the minimum variance free energy estimator can be obtained by setting ϕ = 1 / (1 + exp( x )) and C = ∆ F . This estimator was first proposed by Bennett [1976] without detailed proof. Meng and Wong [1996] termed the ratio estimator as 'bridge sampling" and proved that the BAR form minimizes MSE using Cauchy-Schwarz inequality. Generalizing this estimator and the proof by Meng and Wong [1996] to non-equilibrium settings, we obtain Proposition 3.2.

Multi-state Bennett acceptance ratio [MBAR, Shirts and Chodera, 2008] method K &gt; 2 states, which are typically chosen as interpolants between two ends, so that each adjacent pair of distributions have enough overlap. Specifically, we are interested in estimating the free energy difference between all pairs of distributions:

<!-- formula-not-decoded -->

To this end, MBAR bases on the detailed balance between all pairs:

<!-- formula-not-decoded -->

Assume for each distribution i , we use N i samples { X in } n =1 ··· N i . The optimal form of α ij is

<!-- formula-not-decoded -->

This is similar to BAR, where the minimal-variance estimator of the free energy differences contains the value to be estimated. Therefore, MBAR also takes an iterative form. Making all the necessary substitutions, we obtain:

<!-- formula-not-decoded -->

Tan [2004] first generalized the 'bridge sampling" framework [Meng and Wong, 1996] to multiple states and derived the optimal form of α . Shirts and Chodera [2008] then combined this formulation with the BAR estimator across multiple states, leading to the well-known MBAR method.

Targeted FEP [Jarzynski, 2002] method aims to design an invertible transformation T that maps x a ∼ µ a to µ b ′ such that µ b ′ largely overlaps with µ b . This allows us to use the change of variable formula to track the free energy difference:

<!-- formula-not-decoded -->

where |∇ T ( x ) | is the determinant of the Jacobian matrix of the function T . Similarly, as the transformation is invertible, we can write in the reverse direction:

<!-- formula-not-decoded -->

Thermodynamic Integration (TI) approach also introduces a sequence of distributions that connects the two marginal distributions and estimate free energy difference using the following equality:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C Proofs

## C.1 Variational Bounds (Equation (29)) and IWAE Bounds (Equation (28))

We restate the bounds for easier reference:

Let ˜ W v denote the generalized work associated with samples from the forward and backward SDEs:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We then have

<!-- formula-not-decoded -->

and which finishes the proof.

The IWAE bound can then be obtained by Jensen's inequality. The green color indicates the terms in Equations (28) and (29) for a clearer visualization.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We hence can see that IWAE bounds are tighter compared to the ELBO and EUBO bounds.

<!-- formula-not-decoded -->

Proof. These bounds are corollary based on the escorted Crooks theorem, which says

<!-- formula-not-decoded -->

We first look at the ELBO and EUBO bounds, and then we prove the IWAE bounds.

Taking the expectations over escorted Crooks theorem, we obtain

<!-- formula-not-decoded -->

We recognize their LHS are negative KL divergences, which are always non-positive. Therefore,

<!-- formula-not-decoded -->

Rearrange these equations, we obtain

<!-- formula-not-decoded -->

## C.2 Minimum Variance Non-equilibrium Free Energy Estimator (Proposition 3.2)

Proposition 3.2 (Minimum variance non-equilibrium free energy estimator) . Let ˜ W a → b and ˜ W b → a be the work terms defined in Corollary 3.1. The minimum-variance estimator is given by:

<!-- formula-not-decoded -->

where ϕ is the Fermi function ϕ ( · ) = 1 / (1 + exp( · )) and C = ∆ F in optimal.

Proof. Our proof follows Bennett [1976], Meng and Wong [1996] closely, with a slight extension to path measures. To increase the readability of the proof, we first summarize the entire structure:

1. express the normalizing factor ratio;
2. express MSE 2 of the normalizing factor ratio estimator, and approximate it with δ -method;
3. apply Cauchy-Schwartz inequality to obtain a lower bound of MSE 2 . This gives an optimal condition to minimize MSE 2 .
4. plug the condition back to the normalizing factor ratio expression and finish the proof.

## 1. express the normalizing factor ratio:

First, we have the following equality:

<!-- formula-not-decoded -->

where α is an arbitrary function, and g is any function such that g ( r ) /g ( -r ) = exp( -r ) . Our goal is to find a form of α to minimize the variance (MSE) of the estimator of the ratio between normalization factors. Also, to make things clearer, we explicitly write the direction in the path measure P a → b , P b → a , and note that we drop the superscript ( v ) for simplicity.

The equality Equation (71) holds, because:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 2. express MSE 2 of the normalizing factor ratio estimator, and approximate it with δ -method:

We now write down its Monte Carlo form. For simplicity, we assume we use the same number of samples from X (1: N ) a → b ∼ P a → b and X (1: N ) b → a ∼ P b → a .

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

We consider the MSE of ˆ r :

The variances are given by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and hence we have

<!-- formula-not-decoded -->

We now look at the second term in the numerator:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To simply the calculation of MSE, ket ¯ η 1 , ¯ η 2 be respectively the numerator and denominator of ˆ r :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and we denote η 1 and η 2 as the true value:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we write E b → a as a shorthand of E P b → a . Note that

<!-- formula-not-decoded -->

Following Meng and Wong [1996], we use δ -method to approximate the MSE:

<!-- formula-not-decoded -->

Therefore, we can write the MSE as

MSE 2

<!-- formula-not-decoded -->

## 3. apply Cauchy-Schwartz inequality to obtain a lower bound of MSE 2 :

We denote

We hence have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We can write the denominator as

<!-- formula-not-decoded -->

and the entire MSE as

<!-- formula-not-decoded -->

Following Meng and Wong [1996], we apply the Cauchy-Schwartz inequality:

<!-- formula-not-decoded -->

Due to the property of g , we can see B is always positive:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that E b → a [ B ] does not depend on α . Hence, we found a lower bound of the MSE w.r.t α . The equality holds when A ∝ B , i.e.,

<!-- formula-not-decoded -->

## 4. plug the condition back to the normalizing factor ratio expression and finish the proof:

Plugging α back to Equation (78), we obtain

<!-- formula-not-decoded -->

We now use the property of g , namely g ( r ) /g ( -r ) = exp( -r ) , to simplify both the numerator and the denominator:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking log on both sides of Equation (105) and plugging in the simplified numerator and denominator, we obtain

<!-- formula-not-decoded -->

Let ϕ ( · ) = 1 / (1 + exp( · )) as the Fermi function, we have

<!-- formula-not-decoded -->

and which finishes the proof.

## C.3 Escorted Jarzynski and Controlled Crooks with Imperfect Boundary Conditions (Corollary 3.1 and Proposition 3.3)

Corollary 3.1 (Escorted Jarzynski with imperfect boundary conditions) . Given v ψ t and U θ t , consider the forward and backward SDEs:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where σ t ≥ 0 and µ a and µ b denote the distributions associated with the energies U a and U b , respectively. Define also the 'corrected generalized work":

<!-- formula-not-decoded -->

Using the generalized work with correction, we have the same escorted Jarzynski equality as before:

<!-- formula-not-decoded -->

where - → P v and ← -P v are the path measures over the solutions to the forward and backward SDE, respectively.

Proposition 3.3 (Discretized Controlled Crooks theorem with imperfect boundary conditions) . Let N + and N -be as in Equations (31) and (32) , and define the forward and backward discretized paths via

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we assume that σ t &gt; 0 , ∆ t i = t i +1 -t i and η i ∼ N (0 , Id ) , independent for each i = 0 , . . . , M . Then, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We first prove Corollary 3.1.

Proof. First, we note that the escorted Jarzynski ∆ F = -log E - → P v [exp( -˜ W v )] = log E ← -P v [exp( ˜ W v )] can be obtained from controlled Crooks theorem. Therefore, to show Equation (119), we only need to prove:

<!-- formula-not-decoded -->

Consider the SDEs as Equations (23) and (24). However, instead of starting from U a and U b , we start from U θ 0 and U θ 1 . We define their path measures as - → P v ′ and ← -P v ′ , and we have

<!-- formula-not-decoded -->

According to the controlled Crooks theorem (as in Equation (16)) applied to the transport between U θ 1 and U θ 0 , we have

<!-- formula-not-decoded -->

Take logarithm of Equation (125) and add with Equation (126), we obtain

<!-- formula-not-decoded -->

which finishes the proof of Corollary 3.1.

Next, we look at Appendix C.3.

Proof. We will only provide proof for Equation (122) in the following, and Equation (123) follows exactly the same principle.

As the Gaussian kernel in Equations (120) and (121) is normalized, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C.4 Escorted Jarzynski Equality with ODE Transport

We restate the equivalence of escorted Jarzynski equality with ODE transport and target FEP formula with the instantaneous change of variables in the following:

Let X t evolve according to the ODE

<!-- formula-not-decoded -->

̸

For a smooth energy function U t with, potentially U 0 = U a and U 1 = U b , the escorted Jarzynski equation with imperfect boundary conditions (Corollary 3.1) holds:

̸

<!-- formula-not-decoded -->

This is equivalent to the target FEP formula with the instantaneous change of variables:

<!-- formula-not-decoded -->

̸

We note that the proof for escorted Jarzynski by Vargas et al. [2024], Albergo and Vanden-Eijnden [2024] requires σ t = 0 . Concretely, the proof by Vargas et al. [2024] relies on the FB RND, which applies only to SDEs. On the other hand, while Proposition 3 by Albergo and Vanden-Eijnden [2024] states that σ t ≥ 0 , the proof essentially requires σ t = 0 in order to eliminate σ t from both sides in Equation (63) on page 19. One valid proof for σ t = 0 was provided by Tian et al. [2024] using the generalized Liouville equation. Here, we consider a more straightforward derivation, which also directly showcases the equivalence to target FEP formula with the instantaneous change of variables.

̸

Proof. To prove Equation (135), we directly show the equivalence between Φ and ˜ W v . To do so, we consider the total derivative of U t ( X t ) :

<!-- formula-not-decoded -->

The second equality is due to the ODE Equation (134). We therefore have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which finishes the proof.

## C.5 Equivalence between TI and Our Approach with Perfect Transport

When our method is optimally trained-such that the density of samples simulated from the SDE matches the energy U t at every time step-it becomes equivalent to TI. We only prove this equivalence using forward work, while the backward work will follow the same argument. To show this, let p t denote the sample density at time step t . At optimality, following Albergo and Vanden-Eijnden [2024], we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, the generalized work

<!-- formula-not-decoded -->

i.e., ˜ W v will always be a constant under perfect transport. Therefore, we can write the escorted Jarzynski as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## D SE(3)-equivariant and -invariant Graph Neural Networks

For n -particle problem and molecular systems, we adopt E(n)-equivariant graph neural networks (EGNN) proposed in [Satorras et al., 2021]. Given a 3D graph G = ( V, E, X, H ) where V is a set of

vertices, E ⊆ V × V is a set of edges, X ∈ R N × 3 is a set of atomic coordinates and H ∈ R N × K is a set of node features with feature dimension K . The procedure of EGNN is the following:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we discard the coordinate update when we only need SE(3) invariance. To break the reflection symmetry, we introduce a cross-product during the position update which is reflectionantisymmetric [Du et al., 2022].

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ϕ e , ϕ h , ϕ x and ϕ c are different neural networks to encode edge, node, relative direction and cross direction scalar features.

## E Additional Experimental Results

## E.1 Runtime Analysis

| Table 4: Inference time for FEAT and Target FEP (FM).   | Table 4: Inference time for FEAT and Target FEP (FM).   | Table 4: Inference time for FEAT and Target FEP (FM).   | Table 4: Inference time for FEAT and Target FEP (FM).   |
|---------------------------------------------------------|---------------------------------------------------------|---------------------------------------------------------|---------------------------------------------------------|
| Name                                                    | GMM(d=100)                                              | LJ-55                                                   | ALDP-T                                                  |
| FEAT (ours)                                             | 8 s                                                     | 2 min                                                   | 20 s                                                    |
| Target FEP (FM)                                         | 40 s                                                    | >10 h                                                   | 40 min                                                  |

Besides the results in Table 4, we also include a brief discussion against neural TI and MBAR below:

- Versus neural TI, FEAT performs transport from both sides, which can roughly double its runtime. However, we observed that Neural TI requires a much larger sample size, and preconditioning for the network to achieve ideal performance, which largely increase its inference time.
- Versus MBAR, FEAT is a neural network approach that requires training. But it does not need intermediate samples. We here take an example using ALDP. This can make it more favorable when sampling from the intermediate is expensive. For ALDP, generating samples for each target requires about 1 day on our machine. Collecting all the targets for MBAR can hence take between 1-10 days, depending on whether the simulations are run in parallel. In contrast, our training process takes only 1-2 hours, which is significantly faster than sampling from all intermediate densities. However, this comparison can vary depending how the sample collection pipeline are implemented.

## E.2 Robustness of FEAT

In this section, we analyze the robustness of FEAT and Target FEP with flow matching against different sample size and number of discretization steps on GMM with different dimensionalities. We can see FEAT achieves greater robustness toward less steps and less samples. This results also reflect our discussion in Proposition 3.3 for discretization errors.

Table 5: FEAT

|   #step |   sample size | GMM-40D      | GMM-100D     |
|---------|---------------|--------------|--------------|
|      50 |          5000 | -0.05 ± 0.33 | -4.04 ± 1.92 |
|     100 |          5000 | 0.12 ± 0.07  | -6.43 ± 1.72 |
|     500 |          5000 | 0.00 ± 0.06  | -3.59 ± 1.06 |
|     500 |           500 | 0.13 ± 0.09  | -5.56 ± 1.87 |
|     500 |          1000 | -0.04 ± 0.05 | -6.46 ± 1.86 |
|     500 |          5000 | 0.00 ± 0.06  | -3.59 ± 1.06 |

## E.3 Demonstration of Transferable FEAT

FEAT can be trained on several datasets from multiple targets, with conditions on the target parameters. Once trained, this model will allow us to apply FEAT to similar systems without re-training, similar to what has been demonstrated in transferable Boltzmann generators [Klein and Noé, 2024].

We showcase this potential on GMM-40 with different scaling factor. Concretely, we scale the state B with a scalar 0 . 5 , 0 . 7 , 0 . 9 , 1 . 1 , 1 . 3 , 1 . 5 when training. The network also takes the scalar as an input. After training, we evaluate on unseen scalars and report the average error and standard deviation across 3 runs. As we can see, the transferable FEAT model achieves good accuracy across a range of unseen targets. Also, as expected, interpolation yields better performance than extrapolation.

Table 7: Transferable FEAT on GMM with different scalars unseen during training.

| Scalar      | 0.45         | 0.6          | 0.8          | 1.0          | 1.2         | 1.4          | 1.55        |
|-------------|--------------|--------------|--------------|--------------|-------------|--------------|-------------|
| Error ± std | -3.52 ± 1.13 | -0.07 ± 0.05 | -0.02 ± 0.06 | -0.01 ± 0.06 | 0.01 ± 0.04 | 0.003 ± 0.04 | 0.03 ± 0.13 |

## F Additional Experimental Details

## F.1 Systems

## F.1.1 Gaussian Mixtures

We consider estimating the free energy differences between two Gaussian Mixtures in 40/100dimensional space. Our implementation is based on the code by Midgley et al. [2023]. Below are parameters for S a and S b :

- S a : 16 mixture components, components mean ∼ U ( -2 , 2) , std softplus ( -3) , random seed 10.
- S b : 40 mixture components, components mean ∼ U ( -2 , 2) , std softplus ( -2) , random seed 0.

## F.1.2 Lennard-Jones (LJ) particles

We consider alchemical free energy for N = 55 / 79 / 128 LJ particles. We obtain samples from states S a and S b with the Metropolis-adjusted Langevin algorithm (MALA) for 100,000 steps. We remove the first 20,000 samples as the burn-in period. The step size is dynamically adjusted on the fly to ensure the acceptance rate is roughly 0.6. Below are detailed settings for S a and S b :

- S a : LJ-potential with the harmonic oscillator:

̸

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

In our experiments, we set σ = ϵ = 1 .

Table 6: Target FEP

|   #step |   sample size | GMM-40D      | GMM-100D      |
|---------|---------------|--------------|---------------|
|      50 |          5000 | -0.98 ± 0.23 | -15.44 ± 6.02 |
|     100 |          5000 | -0.48 ± 0.17 | -12.10 ± 4.80 |
|     500 |          5000 | -0.12 ± 0.08 | -13.72 ± 2.46 |
|     500 |           500 | -0.16 ± 0.37 | -22.24 ± 3.43 |
|     500 |          1000 | 0.02 ± 0.18  | -19.73 ± 2.13 |
|     500 |          5000 | -0.12 ± 0.08 | -13.72 ± 2.46 |

- S b : the harmonic oscillator:

<!-- formula-not-decoded -->

where X n ∈ R 3 is the coordinate for n -th particle in the system.

## F.1.3 Alanine dipeptide - solvation (ALDP-S)

We consider the solvation free energy between ALDP in the vacuum environment and with implicit solvent, defined with AMBER ff96 classical force field. Specifically, the samples were gathered from a 5 microsecond simulation under 300K with Generalized Born implicit solvent implemented in openmmtools Chodera et al. [2025]. The Langevin middle integrator implemented in Eastman et al. [2023] with a friction of 1/picosecond and a step size of 2 femtoseconds was used to harvest a total of 250,000 samples. The same sampling protocol was used in the following paragraph as well. Below are settings for S a and S b :

- S a : ALDP in the vacuum environment;
- S b : ALDP in implicit solvent.

We also rescale each target scale by 20, i.e., we define the energy as U ( x 20 ) . Note that this will only change the scale of input and the score, with no influence on the free energy difference as long as we apply the same scaling to both targets.

## F.1.4 Alanine dipeptide - transition (ALDP-T)

We consider Alanine dipeptide in the vacuum. As shown in Figure 2a, there are two metastable states in this system. We therefore consider the transition free energy between them. Below are settings for S a and S b :

- S a : ALDP in the vacuum environment, ϕ ∈ (0 , 2 . 15) ;
- S b : ALDP in the vacuum environment, ϕ ∈ [ -π, 0] ∪ [0 , π ) .

Similar to the solvation case, we rescale each target scale by 20.

## F.1.5 φ 4 lattice field theory

For φ 4 experiments, we consider reweighting the histograms obtained from two umbrella samplings with the free energy estimated by our approach. The random variables here are field configurations φ ∈ R L × L , and the energy function is defined as

<!-- formula-not-decoded -->

Here, we use φ x to represent the value of φ at index x . We choose m 2 = -1 , λ = 0 . 8 following Albergo and Vanden-Eijnden [2024]. The two states are defined with two different umbrella samplings:

- S a : U a ( φ ) = U ( φ ) + k 1 2 ( 1 L 2 ∑ x φ x -µ 1 ) 2 ;
- S b : U b ( φ ) = U ( φ ) + k 2 2 ( 1 L 2 ∑ x φ x -µ 2 ) 2 .

where k 1 = k 2 = 10 , and µ 1 = -0 . 3 , µ 2 = 0 . 6 . We deliberately choose asymmetric values, as a symmetric setup will render the analysis less complicated.

We then estimate the free energy difference ∆ F = F b -F a with our proposed approach. With this estimate, we construct the histogram of the average magnetization by reweighting the samples from U a and U b . Concretely, for each bin ξ in the histogram, we compute its reweighted probability as

<!-- formula-not-decoded -->

where N a denotes the total number of samples from u a , and n a is the number of those samples falling in bin ξ .

## F.2 Hyperparameter

We include hyperparameters for model training and evaluation in Table 8. We explain some hyperparameters in the following:

- α t , β t , γ t : these parameters define the interpolant: X t = α t X 0 + β t X 1 + γ t ϵ , where ϵ ∼ N (0 , I ) ;
- OT pair: To facilitate training, instead of randomly sampling a pair ( X 0 , X 1 ) , we compute an optimal transport (OT) plan to select pairs of data points that are closer to each other. We use the implementation by Tong et al. [2024] (MIT License) to find the OT pair.

Additionally, we note that standard OT chooses the closest pair in the Euclidean distance, which does not directly apply to our rotation-invariant alanine dipeptide and permutation/rotation-invariant Lennard-Jones data. To solve this problem, we first canonicalize all data points. Specifically, we select one sample as the reference system and rotate all other samples from both states to align with this reference using the Kabsch algorithm [Kabsch, 1976]. For the Lennard-Jones system, we additionally canonicalize atom permutations by applying the Hungarian algorithm [Kuhn, 1955, 1956] to find the optimal assignment that minimizes the distance matrix between each sample and the reference. Similar approaches were also adopted by Klein et al. [2023]. In practice, we found this significantly accelerates the training and enhances the performance for larger systems.

- OT batch size: instead of running OT on the entire dataset, we run OT within a much smaller batch to ensure a low running cost. Note that this number is different from the 'batch size".
- FM warm up: we found it is helpful to warm up the vector field network with flow matching, especially for GMM in high-dimensional space. If we use this warm-up, we will put the iteration number in the table; otherwise, we will leave '-".
- σ t : recall that during simulation, we run the forward and backward SDEs as defined in Equations (23) and (24). σ t is the diffusion term in these SDEs. We note that σ t is not the noise level for the stochastic interpolants (which is γ t ).

Table 8: Hyperparameters of our experiments.

| Hyperparameters                                         | GMM                           | GMM                           | LJ                            | LJ                            | LJ                            | ALDP-S/T                      |
|---------------------------------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|
|                                                         | d = 40                        | d = 100                       | d = 55 × 3                    | d = 79 × 3                    | d = 128 × 3                   | d = 22 × 3                    |
| Model and Interpolant choices                           | Model and Interpolant choices | Model and Interpolant choices | Model and Interpolant choices | Model and Interpolant choices | Model and Interpolant choices | Model and Interpolant choices |
| Network architecture Network size                       | MLP 5,                        |                               |                               |                               | SE(3)-GNN 4, 64               |                               |
| α t                                                     | 400                           |                               |                               |                               |                               |                               |
|                                                         |                               |                               |                               | 1 - t                         |                               |                               |
| β t                                                     |                               |                               |                               | t                             |                               |                               |
| γ t                                                     |                               |                               | √ at (1                       | - t ) , a = 0 .               | 05                            |                               |
| Training                                                | Training                      | Training                      | Training                      | Training                      | Training                      | Training                      |
| learning rate batch size                                | 1,000                         | 1,000                         | 100                           | 0.001 30                      | 20                            | 500                           |
| iteration number                                        | 50,000                        | 200,000                       | 10,000                        | 20,000                        | 40,000                        | 20,000                        |
| OT pair                                                 | No                            | Yes                           | No                            | Yes                           | Yes                           | Yes                           |
| OT batch size                                           | -                             | 1,000                         | -                             | 500                           | 500                           | 500                           |
| FM warm up                                              | -                             | 50,000                        | -                             | -                             | -                             | -                             |
| Simulation and Estimation                               | Simulation and Estimation     | Simulation and Estimation     | Simulation and Estimation     | Simulation and Estimation     | Simulation and Estimation     | Simulation and Estimation     |
| number of discretization steps evaluating sample size ϵ | 1,000                         | 5,000                         | 1,000                         | 500 1,000 0.01                | 1,000                         | 1,000                         |

## F.3 Hyperparameters and Settings for One-sided FEAT

Network. For one-sided FEAT, we only need one network to parameterize the score/mean/noise for the diffusion process. We adopt the parameterization (precisely, c in , c skip , c out ) following Karras et al. [2022], with an EGNN (fully-connected) as the network to prediction the mean E [ X 0 | X t ] . We increase the EGNN size with 5 hidden layers, each with 256 hidden units.

Training details. We train the network for 200,000 iterations, with a batch size of 20 for both Chignolin and Ala-4. We observe a better convergence for larger batch size, while we choose 20 to fit in our GPU. We keep an EMA with rate 0.99.

Estimation details. We estimate the free energy for each states with 500 discretization steps. We use 1,000 samples for ALA-4 and 3,000 samples for Chignolin. Additionally, we found it is more stable to use DDPM kernel [Ho et al., 2020] as the denoising and noising kernel instead of EM discretization kernel. This will only slightly change the formulation of the Gaussian expression, with other key components of FEAT unchanged.

Precisely, the forward SDE is defined following Karras et al. [2022] as:

<!-- formula-not-decoded -->

where µ is the target system distribution. The backward SDE is defined as:

<!-- formula-not-decoded -->

where ∇ U θ t is the learned score network. The score and the mean-prediction E [ X 0 | X t ] are connected with Tweedie's formula:

<!-- formula-not-decoded -->

The DDPM kernels for this pair of SDEs are:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## F.4 Computing Resources

All experiments are run on a single 80G NVIDIA H100.

## F.5 Baseline and Reference Settings

Target FEP. We use the same parameter as Table 8. Note that the interpolant becomes X t = α t X 0 + β t X 1 , and the simulation process will be ODE. We do not align the iteration number to be the same as SI. Instead, we run the training until convergence.

Neural TI. We use the same parameter as Table 8. We parameterize the energy network instead of the score network in order to use the TI formula. Specifically, we take the output of the network, and take an inner product with the input to form a scalar as the energy.

Neural TI Preconditioning Design. We include preconditioning for the energy network to ensure boundary conditions and also to increase the accuracy of the learned energy U t .

For GMM, we set the energy network to be

<!-- formula-not-decoded -->

where a t = exp[ f θ ( t ) -f θ (0)] · (1 -t ) , b t = exp[ g θ (1) -g θ ( t )] · t and c t = exp( h θ ( t )) · t · (1 -t ) . f θ , g θ , h θ , U θ are neural networks. We can exactly ensure the boundary condition by this.

For LJ, U A and U B is more sensitive to noisy x t . Therefore, inspired by Máté et al. [2024b], we set the energy network to be

<!-- formula-not-decoded -->

where r t is the radius parameter in LJ, ranging from 0 to 1. a t = exp( α θ ( t )) · t · (1 -t ) is a smooth parameter, as used by Máté et al. [2024b]. b t = 1 -exp( β θ ( t )) · t · (1 -t ) and c t = exp( γ θ ( t )) · t · (1 -t ) are scalar to ensure boundary conditions.

Due to this specific choice of smoothing parameters, we failed to design a stable preconditioning for ALDP and hence did not compare FEAT with neural TI on ALDP.

MBAR. We use MBAR (with pymbar ) to obtain the reference value for the LJ system and ALDP:

- LJ: for LJ-potential, we create N distributions as follows:

̸

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

We set σ 1 = ϵ 1 = 1 , σ N = ϵ N = 0 , and let σ n = ϵ n vary as a linear interpolant between 1 and 0 For LJ-55/79, we use N = 41 distributions, and for LJ-128, we choose N = 81 . We run the Metropolis-adjusted Langevin algorithm (MALA) to obtain 20,000 samples for each marginal, and then randomly choose 1,000 of them for each marginal to decorrelate the samples.

- ALDP-S: We set 11 distributions by modulating the solvation parameter with a factor λ ∈ [0 , 1] that scales the charges in the ' CustomGBForce ' in the force field with OpenMM . Specifically, we modify the force by the following code:

```
1 for force in system.getForces(): 2 if force.__class__.__name__ == 'CustomGBForce': 3 for idx in range(force.getNumParticles()): 4 charge , sigma , epsilon = force.getParticleParameters(idx) 5 force.setParticleParameters(idx, (charge*lamb, sigma , epsilon)) 6
```

Listing 1: Python code example for changing the solvation strength.

The 11 distributions are set with λ = 0 . 0 , 0 . 1 , . . . , 1 . 0 . We run MD for each distributions using the setup described in Appendix F.1.3, and randomly choose 2,000 of them for each marginal to decorrelate the samples to run MBAR.

- ALDP-T: we define three distinct distributions: (1) a distribution containing only S a , where the energy in regions corresponding to state S b are set to + ∞ , (2) a distribution containing only S b , where the energy in regions corresponding to state S a are set to + ∞ ; and (3) a full distribution that includes both S a and S b .