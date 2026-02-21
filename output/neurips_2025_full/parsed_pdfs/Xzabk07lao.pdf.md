## On scalable and efficient training of diffusion samplers

Minkyu Kim 1 ∗ Kiyoung Seong 1 ∗ Dongyeop Woo 1 Sungsoo Ahn 1 Minsu Kim 1 ,

1 Korea Advanced Institute of Science and Technology (KAIST) 2 Mila - Quebec AI Institute

## Abstract

We address the challenge of training diffusion models to sample from unnormalized energy distributions in the absence of data, the so-called diffusion samplers . Although these approaches have shown promise, they struggle to scale in more demanding scenarios where energy evaluations are expensive and the sampling space is high-dimensional. To address this limitation, we propose a scalable and sample-efficient framework that properly harmonizes the powerful classical sampling method and the diffusion sampler. Specifically, we utilize Monte Carlo Markov chain (MCMC) samplers with a novelty-based auxiliary energy as a Searcher to collect off-policy samples, using an auxiliary energy function to compensate for exploring modes the diffusion sampler rarely visits. These off-policy samples are then combined with on-policy data to train the diffusion sampler, thereby expanding its coverage of the energy landscape. Furthermore, we identify primacy bias, i.e., the preference of samplers for early experience during training, as the main cause of mode collapse during training, and introduce a periodic re-initialization trick to resolve this issue. Our method significantly improves sample efficiency on standard benchmarks for diffusion samplers and also excels at higher-dimensional problems and real-world molecular conformer generation.

## 1 Introduction

Inference in unnormalized densities is a central challenge in machine learning, underlying probabilistic deep learning [19, 25] and many scientific applications [33, 9]. Traditionally, Markov chain Monte Carlo (MCMC) methods have been used, most prominently Metropolis-adjusted Langevin algorithms (MALA) [37] and Hamiltonian Monte Carlo (HMC) [15], but they incur repeated energy-gradient evaluations per sample. Amortized inference instead trains deep generative models to map noise to samples, enabling evaluation-free generation at test time and promising orders-of-magnitude speedups once the model is trained.

Researchers have recently focused on diffusion samplers, which parameterize continuous-time diffusion processes with neural networks, an approach inspired by successes in high-dimensional settings like image and text generation. The leading methods include flow-annealed importance sampling bootstrap (FAB) [30], generative flow networks (GFlowNets) [3], denoising diffusion samplers (DDS) [43], controlled Monte Carlo diffusion (CMCD) [44], and iterative denoising energy matching (iDEM) [1]. Because samples from the target distribution are unavailable, these samplers iterate between: (1) sample from the neural diffusion model, (2) query the energy, and (3) update the model to better match the target distribution.

Despite their promise, diffusion-based samplers struggle in high dimensions. Early in training, the neural proposal is effectively random and is not aligned with the energy landscape, leading to sample-inefficient exploration. This is in contrast with the classic training-free samplers, e.g., MALA, which leverage gradient information to steer proposals toward low-energy modes from the start.

∗ Equal contribution, correspondence to: {minkyu-kim, kiyoung.seong}@kaist.ac.kr

2

Techniques to improve the sample efficiency of diffusion samplers, like replay buffers [3] and local energy-guided refinements [21], yield only marginal gains and fail to overcome the poor quality of the initial diffusion samples. Indeed, He et al. [18] recently showed that nearly all effective neural samplers rely on Langevin parametrization, i.e., incorporating energy gradients at inference, which erodes the primary efficiency benefit of amortized sampling.

Moreover, diffusion samplers are prone to mode collapse: training on their own outputs leads to overfitting to dominant modes, and the model 'locks in' prematurely. Reinforcement learning exploration bonuses [36] can broaden coverage, but at the cost of biasing the sampler's target distribution. Local perturbations [21] help, but require many expensive iterations in large state spaces.

Contribution. We propose search-guided diffusion samplers ( SGDS ), a simple yet powerful framework that enables scalable and unbiased training of diffusion samplers in high-dimensional problems. A training-free Markov-chain 'Searcher' explores the target density augmented with an explicit exploration reward to discover underexplored modes. The diffusion 'Learner' then distills these trajectories through the trajectory balance objective [28], preserving theoretical guarantees while incorporating exploration.

At a high level, our SGDS operates in two stages. Stage 1 : the Searcher collects informative samples from the target (optionally with exploration incentives) to overcome the random initialization of the Learner. The Learner is trained off-policy via trajectory balance on a mixture of Searcher- and self-generated trajectories, rapidly improving sample efficiency. Stage 2 : the Searcher employs random network distillation (RND) bonuses [10] to probe modes the Learner has not yet covered; the Learner then ingests these enriched trajectories using trajectory balance with weight re-initialization to counter primacy bias [32].

We show that SGDS , despite its simplicity, produces substantial gains over baseline diffusion samplers across benchmarks: classical Gaussian mixtures and the Manywell task; particle simulation problems like LJ-13 and LJ-55; and real-world molecules, Alanine Di-, Tri-, and Tetra-peptide. Our method significantly improves sample efficiency and scalability, marking a practical path towards highdimensional diffusion-based inference.

## 2 Preliminaries

## 2.1 Diffusion samplers as controlled neural SDEs

Let E : R 𝑑 → R be an energy function defining an unnormalized density 𝑅 ( 𝑥 ) = exp GLYPH&lt;0&gt; -E( 𝑥 ) GLYPH&lt;1&gt; . Sampling from the corresponding Boltzmann distribution 𝑝 target ( 𝑥 ) = 𝑅 ( 𝑥 )/ 𝑍 , with partition function 𝑍 = ∫ 𝑅 ( 𝑥 ) 𝑑𝑥 , can be formulated as controlling the stochastic differential equation (SDE)

<!-- formula-not-decoded -->

where 𝑤 𝑡 is standard 𝑑 -dimensional Brownian motion, 𝑢 𝜃 is the drift function parameterized by 𝜃 e.g., neural networks, and 𝑔 is the diffusion function. The goal is to choose 𝜃 such that the terminal distribution 𝑝 𝜃 1 induced by Equation (1) matches the target, i.e., 𝑝 𝜃 1 ( 𝑥 ) ∝ 𝑅 ( 𝑥 ) ).

Euler-Maruyama discretization. With 𝑇 uniform steps of size Δ 𝑡 : = 1 / 𝑇 , the SDE Equation (1) is discretized via the Euler-Maruyama scheme

<!-- formula-not-decoded -->

which defines Gaussian forward kernels 𝑃 𝐹 ( 𝑥 𝑡 + Δ 𝑡 | 𝑥 𝑡 ; 𝜃 ) . Analogously, one defines reference backward kernels 𝑃 𝐵 ( 𝑥 𝑡 -Δ 𝑡 | 𝑥 𝑡 ) . Common choices for 𝑃 𝐵 include Brownian motion d 𝑥 𝑡 = 𝛽 ( 𝑡 ) d ¯ 𝑤 𝑡 for variance-exploding (VE) processes, the time-reversed Ornstein-Uhlenbeck (OU) kernel d 𝑥 𝑡 = -𝛽 ( 𝑡 ) 𝑥 𝑡 d 𝑡 + √︁ 2 𝛽 ( 𝑡 ) d ¯ 𝑤 𝑡 for variance-preserving (VP) processes, and the Brownian bridge d 𝑥 𝑡 = 𝑥 𝑡 𝑡 d 𝑡 + 𝜎 d ¯ 𝑤 𝑡 , where ¯ 𝑤 𝑡 is time-reversed Brownian motion.

The forward and backward policies for the complete trajectory 𝜏 = ( 𝑥 0 → 𝑥 Δ 𝑡 →··· → 𝑥 1 ) , denoted by 𝑃 𝐹 ( 𝜏 ; 𝜃 ) and 𝑃 𝐵 ( 𝜏 | 𝑥 1 ) , repectively, are defined as compositions of these kernels across discrete time steps:

<!-- formula-not-decoded -->

## Algorithm 1 Training search-guided diffusion samplers ( SGDS )

```
1: Q buffer ←∅ ; fix random target net 𝑓 rnd ; initialize predictor ˆ 𝑓 𝜙 and Learner ( 𝑃 𝐹 ( 𝜏 ; 𝜃 ) , log 𝑍 𝜃 ) 2: for 𝑟 = 1 , . . . , 𝑁 round do ⊲ outer rounds 3: // Searcher: gradient-guided MCMC 4: ˜ E( 𝑥 ) ← ( E( 𝑥 ) , 𝑟 = 1 , E( 𝑥 ) -𝛼 GLYPH<13> GLYPH<13> 𝑓 rnd ( 𝑥 ) -ˆ 𝑓 𝜙 ( 𝑥 ) GLYPH<13> GLYPH<13> 2 2 , 𝑟 > 1 5: Obtain { 𝑥 ( 𝑖 ) 1 } 𝑀 chain 𝑖 = 1 and log ˆ 𝑍 by running 𝑀 chain parallel MCMC for 𝑀 iter steps on ˜ E( 𝑥 ) 6: Q buffer ←Q buffer ∪ { 𝑥 ( 𝑖 ) 1 , E( 𝑥 ( 𝑖 ) 1 )} 𝑀 chain 𝑖 = 1 7: // Learner: 𝐼 inner iterations (even iterations: on-policy, odd iterations: off-policy) 8: for 𝑖 = 1 , . . . , 𝐼 do 9: if 𝑖 mod 2 = 0 then ⊲ on-policy 10: Sample { 𝜏 𝑘 } 𝐵 𝑘 = 1 ∼ 𝑃 𝐹 ( 𝜏 ; 𝜃 ) 11: X ← { 𝑥 1 from 𝜏 𝑘 } 12: else ⊲ off-policy 13: Sample X = { 𝑥 1 } 𝐵 off ℓ = 1 ∼ 𝑃 (· | Q buffer ) 14: Generate { 𝜏 ℓ } ∼ 𝑃 𝐵 ( 𝜏 | 𝑥 1 ) 15: end if 16: L TB = 1 𝐵 ˝ 𝑘 GLYPH<2> log 𝑍 𝜃 𝑃 𝐹 ( 𝜏 𝑘 ; 𝜃 ) 𝑅 ( 𝑥 1 ) 𝑃 𝐵 ( 𝜏 𝑘 | 𝑥 1 ) GLYPH<3> 2 17: 𝜃 ← Minimize (L TB ) ⊲ diffusion sampler update 18: 𝜙 ← Minimize GLYPH<16> 1 | X| ˝ 𝑥 1 ∈X ∥ 𝑓 rnd ( 𝑥 1 ) -ˆ 𝑓 𝜙 ( 𝑥 1 )∥ 2 2 GLYPH<17> ⊲ RND predictor update 19: end for 20: Re-initialize 𝑃 𝐹 (· | 𝜃 ) but retain log 𝑍 𝜃 ⊲ Periodic partial re-initialization 21: end for
```

Stochastic control of neural SDEs. Diffusion models typically minimize the forward Kullback-Leibler (KL) divergence

<!-- formula-not-decoded -->

which presupposes abundant samples from 𝑥 1 ∼ 𝑝 target . When such data are unavailable, e.g., in scientific domains, one may instead minimize the reverse KL divergence

<!-- formula-not-decoded -->

using samples from 𝑥 1 ∼ 𝑃 𝐹 . Notable methods that optimize this objective include the path-integral sampler (PIS) [47], which employs a VE Brownian-motion reference process, and denoising diffusion samplers (DDS) [43], which use a VP OU reference process.

## 2.2 Continuous GFlowNet objective for diffusion samplers

Following Sendera et al. [39], Euler-Maruyama samplers can be interpreted as continuous generative flow networks (GFlowNets) [24]. GFlowNets [3, 4] are off-policy reinforcement-learning algorithms for sequential decision making samplers. Treating the initial state 𝑥 0 as a point mass at the origin, the forward policy 𝑃 𝐹 acts as an agent that sequentially constructs a trajectory 𝜏 . The trajectory balance (TB) criterion [28] guarantees that the density induced by 𝑃 𝐹 matches the target distribution:

<!-- formula-not-decoded -->

where 𝑍 𝜃 is a learnable scalar that approximates the unknown partition function 𝑍 . Existing GFlowNet-based samplers [46, 39] often adopt Brownian-bridge kernels for 𝑃 𝐵 .

Applying the TB condition to sub-trajectory of 𝜏 yields the sub-trajectory balance objective [27, 35, 46]. While this variant can improve credit assignment, it estimates marginal densities at intermediate states with higher bias compared to the global TB estimates [39].

Off-policy property of GFlowNet-based diffusion samplers. In contrast to KL-based objectives such as PIS or DDS, using on-policy training, GFlowNet objectives can be optimized with off-policy trajectories drawn from any proposal distribution with full support. This flexibility enables richer exploration strategies-noisy roll-outs [24], replay buffers, and MCMC-based local search [39]-that are crucial for efficient sampling from multimodal distributions.

## 3 Method

## 3.1 Search-guided diffusion samplers ( SGDS ): overall framework

In this section, we describe the overall framework of the search-guided diffusion samplers ( SGDS ). Our SGDS combines the strengths of off-policy training from GFlowNet diffusion samplers with the exploratory power of gradient-guided MCMC. We follow the setting of Sendera et al. [39] for modeling GFlowNet-based diffusion samplers. Each round alternates between two roles:

Searcher (gradient-informed MCMC). The Searcher uses gradient information ∇ log 𝜋 ( 𝑥 ) to efficiently generate representative samples from the target distribution. These samples populate a replay buffer and simultaneously provide an estimate of the log partition function, log 𝑍 . Exploration is guided by an intrinsic reward from random network distillation (RND) [10], which identifies underexplored modes using a form of self-supervised learning.

Learner (diffusion sampler). Learner, a neural diffusion sampler, is trained by minimizing trajectory balance loss [24], blending (i) on-policy trajectories generated from its current policy and (ii) off-policy trajectories replayed from the buffer. Periodic re-initialization of the Learner mitigates primacy bias, enhancing sample efficiency.

This round repeats until the Learner alone generates high-quality samples. For simple targets, training may converge within a single round, while complex targets typically benefit from multiple rounds.

The SGDS tackles two critical challenges in existing diffusion sampling approaches:

Scalability. In high-dimensional spaces, diffusion samplers frequently miss low-energy modes, as their generated samples rarely visit unexplored modes. The Searcher, operating as parallel gradient-informed chains, rapidly identifies these modes. Although the samples collected from the Searcher are biased, the trajectory balance objective enables unbiased training of the Learner.

Sample efficiency. Each expensive gradient evaluation is amortized across multiple Learner updates through off-policy replay. The RND-driven intrinsic rewards direct the Searcher towards under-explored areas, maximizing the informativeness of new samples. Periodic Learner re-initialization prevents overfitting to initial samples and maintains replay buffer diversity. Collectively, these components significantly enhance the efficiency of gradient computations.

Algorithmic details for each component follow in subsequent sections and Algorithm 1.

## 3.2 Searcher

The Searcher identifies low-energy modes using parallel gradient-guided Markov chains. Methods such as annealed importance sampling (AIS) [31], Metropolis-adjusted Langevin algorithms (MALA) [37], or molecular dynamics (MD) are suitable candidates. These methods generate samples by transporting prior samples in the direction of the target density (or its tempered density) via several Markov chains. We use AIS and MALA for synthetic energy functions, and MD for all-atom systems.

In the initial step of the algorithm, we run 𝑀 chain parallel chains, estimating log ˆ 𝑍 which is explained in Appendix A. The Searcher then stores the collected samples in a replay buffer and passes the estimated log ˆ 𝑍 to the Learner model. In subsequent rounds, we incorporate exploration uncertainty from the Learner via intrinsic rewards for exploration, modifying the Searcher's energy landscape as:

<!-- formula-not-decoded -->

Here, 𝑟 intrinsic ( 𝑥 ) highlights underexplored modes based on previous Learner experiences, and the gradient is used in the drift function of SDEs. Adding a repulsive term for exploration resembles the core idea of metadynamics, which biases sampling away from the modes that have already been well captured.

Random network distillation (RND). To efficiently guide exploration, we employ RND [10] to quantify state novelty, steering the Searcher towards underexplored consists of a fixed, randomly initialized network 𝑓 ( 𝑥 ) and a trainable predictor network ˆ 𝑓 ( 𝑥 ; 𝜙 ) trained by minimizing:

<!-- formula-not-decoded -->

and, for the Searcher in the next round, we utilize this loss as the intrinsic reward given by:

<!-- formula-not-decoded -->

High prediction errors indicate novel states. RND training uses replay buffer samples and online trajectories, assigning high novelty to underexplored modes.

## 3.3 Learner

With the replay buffer initialized by Searcher's samples, the Learner minimizes the trajectory balance objective through iterative training, combining online and replay trajectories. The training incorporates:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here 𝑃 ( 𝑥 1 | D buffer ) denotes a rank-based sampling distribution [42] that assigns higher probability to lower energy samples stored in the buffer, focusing replay on promising modes.

We leverage both on-policy and off-policy training signals from online trajectories and replayed samples, with a replay ratio 𝛾 determining the frequency of replay updates (default: 𝛾 = 1 ).

Re-initialization. Learner re-initialization mitigates primacy bias commonly observed in reinforcement learning scenarios. Primacy bias [32] refers to the model's tendency to rely excessively on early experiences, being trapped in low-reward or biased samples generated at initial stages, thereby hindering the discovery of high-reward samples and underexplored modes. Periodically re-initializing the Learner model 𝑃 𝐹 (·| 𝜃 ) alleviates this bias by resetting parameters strongly influenced by early samples, allowing faster adaptation to recent, higher-quality experiences. Crucially, we retain the previously learned log 𝑍 𝜃 parameter and the replay buffer, preserving the accumulated knowledge while allowing the network to recalibrate based on updated experiences.

## 4 Related works

Classical samplers. Classical sampling approaches primarily rely on MCMC methods. This includes gradient-based algorithms like MALA [37] and HMC [15]. Annealing-based techniques, such as AIS [31] and SMC [12], introduce intermediate distributions to gradually approximate complex targets, mitigating mode collapse. While these MCMC-based methods enable sampling from the complex unnormalized density, they require long trajectories and extensive energy evaluations.

Neural amortized inference. Neural amortized inference methods aim to bypass costly MCMC by training neural samplers that generate approximate samples in one or a few forward passes. Diffusionbased neural samplers learn stochastic differential equations parameterized by neural networks to map simple priors to complex targets [47, 43], and GFlowNets train stochastic policies whose marginal visitation probabilities match an unnormalized density [3, 13]. Boltzmann Generators (BG) is another line of works to amortize inference, such as molecular dynamics simulation. BG utilizes normalizing flows trained on simulated data to sample from the Boltzmann distribution and estimate density, enabling statistical reweighting for unbiased estimates [34, 14, 30, 23, 41].

Diffusion-based neural samplers. Diffusion-based samplers aim to sample from unnormalized target distributions in data-free settings. Several approaches [47, 43, 44, 2, 5] formulate the sampling objective via KL divergence in path measure space. Akhound-Sadegh et al. [1] further introduces off-policy training via replay buffers. Recent works [8, 11] also explore controllable dynamics, offering improved exploration in complex energy landscapes. While these methods often improve mode coverage by learning reverse-time dynamics, they remain computationally intensive, hindering scalability in high-dimensional settings.

Generative Flow Networks. GFlowNets was originally introduced by Bengio et al. [3] and Bengio et al. [4] on discrete spaces where the probability of each outcome is proportional to a given reward signal. Subsequent extensions have connected GFlowNets to continuous space [24], enabling sampling from unnormalized densities in high-dimensional spaces [13, 29]. Recent work has also explored enhancements to off-policy training strategies [39] and incorporated local search mechanisms [21], allowing GFlowNets to more effectively navigate continuous energy landscapes. Additionally, adaptive reward design has emerged as a promising direction for improving mode coverage during training [22], especially in tasks that require structured exploration or sparse supervision.

Figure 1: Mode coverage comparison using 2D projections of 10,000 samples on Manywell-128.

|               | Manywell ( 𝑑 = 64 )   | Manywell ( 𝑑 = 64 )   | Manywell ( 𝑑 = 64 )   | Manywell ( 𝑑 = 64 )   | Manywell ( 𝑑 = 128 )   | Manywell ( 𝑑 = 128 )   | Manywell ( 𝑑 = 128 )   | Manywell ( 𝑑 = 128 )   |
|---------------|-----------------------|-----------------------|-----------------------|-----------------------|------------------------|------------------------|------------------------|------------------------|
| Method        | ELBO ↑                | EUBO ↓                | EUBO - ELBO ↓         | Energy calls          | ELBO ↑                 | EUBO ↓                 | EUBO - ELBO ↓          | Energy calls           |
| PIS+LP        | 300 . 57 ± 0 . 37     | 347 . 48 ± 0 . 26     | 46 . 91 ± 0 . 55      | 130M                  | 601 . 01 ± 0 . 94      | 697 . 32 ± 0 . 49      | 96 . 31 ± 0 . 71       | 130M                   |
| TB+LP         | 306 . 47 ± 0 . 23     | 351 . 98 ± 0 . 46     | 45 . 52 ± 0 . 51      | 180M                  | 612 . 45 ± 0 . 65      | 706 . 73 ± 2 . 59      | 94 . 28 ± 3 . 00       | 300M                   |
| FL-SubTB+LP   | 306 . 14 ± 0 . 71     | 352 . 22 ± 0 . 62     | 46 . 08 ± 0 . 26      | 330M                  | 609 . 85 ± 0 . 48      | 709 . 96 ± 2 . 10      | 99 . 61 ± 1 . 83       | 330M                   |
| TB+LS+LP      | 312 . 66 ± 2 . 66     | 339 . 34 ± 1 . 02     | 26 . 68 ± 3 . 37      | 320M                  | 592 . 52 ± 2 . 25      | 693 . 65 ± 1 . 40      | 101 . 81 ± 3 . 62      | 320M                   |
| TB+Expl+LP    | 306 . 54 ± 0 . 23     | 351 . 91 ± 0 . 53     | 45 . 37 ± 0 . 66      | 180M                  | 611 . 98 ± 0 . 34      | 705 . 35 ± 1 . 05      | 93 . 37 ± 1 . 22       | 240M                   |
| TB+Expl+LS+LP | 300 . 10 ± 1 . 05     | 344 . 85 ± 0 . 41     | 44 . 75 ± 1 . 39      | 320M                  | 591 . 47 ± 0 . 36      | 694 . 93 ± 0 . 54      | 103 . 45 ± 0 . 88      | 320M                   |
| PIS           | 321 . 87 ± 0 . 05     | 2026 . 11 ± 408 . 98  | 1704 . 91 ± 408 . 49  | 100M                  | 643 . 30 ± 0 . 09      | 1159 . 60 ± 48 . 53    | 516 . 30 ± 49 . 67     | 100M                   |
| TB            | 317 . 35 ± 6 . 01     | 853 . 94 ± 43 . 35    | 544 . 36 ± 29 . 85    | 100M                  | 637 . 01 ± 2 . 14      | 1423 . 35 ± 292 . 15   | 786 . 35 ± 290 . 46    | 100M                   |
| TB+LS         | 314 . 94 ± 4 . 60     | 357 . 40 ± 4 . 36     | 42 . 91 ± 9 . 15      | 290M                  | 573 . 13 ± 73 . 49     | 738 . 07 ± 10 . 77     | 164 . 95 ± 62 . 71     | 290M                   |
| TB+Expl+LS    | 265 . 99 ± 95 . 39    | 361 . 00 ± 16 . 58    | 41 . 46 ± 15 . 47     | 290M                  | 589 . 49 ± 7 . 25      | 698 . 24 ± 2 . 81      | 108 . 74 ± 10 . 07     | 290M                   |
| GAFN [36]     | 320 . 88 ± 0 . 36     | 573 . 68 ± 29 . 02    | 252 . 80 ± 30 . 87    | 100M                  | ∗                      | ∗                      | ∗                      | 100M                   |
| AT [22] + LP  | 281 . 56 ± 2 . 21     | 353 . 64 ± 3 . 48     | 72 . 48 ± 2 . 97      | 370M                  | 462 . 61 ± 6 . 67      | 739 . 93 ± 4 . 97      | 277 . 32 ± 2 . 46      | 370M                   |
| iDEM [1]      | 268 . 99 ± 1 . 22     | 414 . 18 ± 1 . 06     | 145 . 20 ± 1 . 60     | 300M                  | 494 . 28 ± 2 . 94      | 817 . 32 ± 3 . 22      | 323 . 04 ± 5 . 69      | 300M                   |
| SGDS          | 320 . 25 ± 0 . 13     | 336 . 51 ± 0 . 11     | 16 . 26 ± 0 . 22      | 20M                   | 614 . 41 ± 3 . 44      | 684 . 76 ± 1 . 30      | 70 . 35 ± 4 . 31       | 20M                    |

Table 1: ELBO, EUBO, their gap, and energy calls across high-dimensional Manywell distributions. We use MALA as the local search algorithm. We consume 6M energy calls per searcher (12M total for 2 rounds) and 8M energy calls for the learner. Bold indicates the best performance per metric, and * indicates large absolute values of metrics.

<!-- image -->

Connection to previous works. Using gradient-guided MCMC for improving exploration in offpolicy diffusion samplers is not new. Lemos et al. [26] employed gradient-guided MCMC to populate replay buffers for GFlowNet diffusion sampler training. Sendera et al. [39] applied parallel MALA initialized from diffusion sampler states, similar to discrete local search GFlowNet methods [21]. Our approach extends the multiple-round algorithm of Lemos et al. [26], incorporating RL techniques to boost efficiency. It can be viewed as a deeper but shorter-cycle alternative to Sendera et al. [39], whose frequent diffusion-based re-initializations overly depend on sampler performance (see comparison with TB + LS at Table 1, Table 2, and Figure 4a).

Leveraging Learner uncertainty to guide exploration aligns with active learning and related GFlowNet approaches [36, 22]. Following generative augmented flow network (GAFN) [36], direct injection of intrinsic reward was effective, similar to our idea (see comparison with GAFN at Table 1). While Kim et al. [22] introduced additional neural samplers called adaptive teachers (AT) as Searchers to covers high loss region it is highly unstable in large scale due to Searcher's adversarial behavior with non-stationary objective, where our method efficiently employs MCMC-based exploration without additional neural network (see comparison with AT + LP at Table 1).

## 5 Experiments

In this section, 1 our primary goal is to demonstrate the performance and efficiency of our proposed framework through several experiments. Specifically, we aim to showcase the sample efficiency and scalability of our method, as well as to validate the effectiveness of the various training strategies we introduced. We focus on presenting results on high-dimensional tasks. In all the experiments, we use four different random seeds and average the results of each run. We provide details of experimental settings in Appendix A.4, Table 4, and Table 5, and additional results in Appendix B.

## 5.1 Main results

Settings. In this work, we compare the performance of our proposed framework against baselines on multiple benchmark tasks, including 40GMM, Manywell-32/64/128, LJ-13, and LJ-55. We

1 Source code: https://github.com/minkyu1022/SGDS

| Method            | ELBO ↑              | EUBO ↓             |
|-------------------|---------------------|--------------------|
| AIS+MLE           | 244 . 85 ± 10 . 34  | 840 . 45 ± 11 . 59 |
| Fine-tuning       | 554 . 17 ± 107 . 70 | 707 . 07 ± 8 . 49  |
| Re-init w/o log 𝑍 | 583 . 65 ± 16 . 69  | 691 . 37 ± 1 . 40  |
| w/o RND-searcher  | 608 . 01 ± 4 . 18   | 686 . 63 ± 1 . 32  |
| SGDS              | 614 . 41 ± 3 . 44   | 684 . 76 ± 1 . 30  |

Figure 2: Trade-off between EUBO-ELBO gap and energy calls in Manywell-128 (left) and LJ-55 (middle). The results of ablation study on Manywell-128 (right) show the performance of AIS using the same total energy calls with MLE amortizing, taking 2 rounds with fine-tuning instead of re-initialization, and using the Searcher with no RND rewards. All methods use 20M energy calls.

<!-- image -->

Table 2: The gap between ELBO and GLYPH&lt;155&gt; EUBO, (equivariant) position Wasserstein-2, energy histogram Wasserstein-2, and energy calls across Lennard-Jones potential. We denote GLYPH&lt;155&gt; EUBO as the EUBO metrics calculated by the reference samples provided by [1], which are not exact samples from the target distribution. For iDEM, we reproduce the results with the hyperparameter setting which can be found in Appendix A.4. The results of Adjoint sampling are obtained from [17]. Bold indicates the best performance, * indicates large absolute values of metrics, - indicates inaccessible value from their papers, and "Div." indicates divergent training.

|                       | LJ-13 ( 𝑑 = 39 )         | LJ-13 ( 𝑑 = 39 )   | LJ-13 ( 𝑑 = 39 )   | LJ-13 ( 𝑑 = 39 )   | LJ-55 ( 𝑑 = 165 )        | LJ-55 ( 𝑑 = 165 )   | LJ-55 ( 𝑑 = 165 )   | LJ-55 ( 𝑑 = 165 )   |
|-----------------------|--------------------------|--------------------|--------------------|--------------------|--------------------------|---------------------|---------------------|---------------------|
| Method                | GLYPH<155> EUBO - ELBO ↓ | W 2 ↓              | 𝐸 (·)W 2 ↓         | Energy calls       | GLYPH<155> EUBO - ELBO ↓ | W 2 ↓               | 𝐸 (·)W 2 ↓          | Energy calls        |
| PIS                   | 2 . 04 ± 0 . 32          | 1 . 57 ± 0 . 03    | 2 . 10 ± 0 . 46    | 370K               | 45 . 53 ± 5 . 58         | 4 . 28 ± 0 . 05     | 40 . 63 ± 13 . 68   | 45K                 |
| TB                    | 12 . 53 ± 3 . 43         | 1 . 71 ± 0 . 07    | 10 . 99 ± 3 . 14   | 370K               | Div.                     | Div.                | Div.                | 45K                 |
| TB+Expl+LS            | 11 . 99 ± 0 . 52         | 2 . 02 ± 0 . 04    | 18 . 69 ± 4 . 49   | 3M                 | ∗                        | 12 . 46 ± 0 . 19    | 123 . 02 ± 21 . 72  | 1M                  |
| iDEM [1]              | 112 . 37 ± 7 . 63        | 4 . 27 ± 0 . 02    | 39 . 92 ± 3 . 24   | 300M               | ∗                        | 17 . 17 ± 0 . 32    | 210 . 87 ± 6 . 26   | 120M                |
| Adjoint Sampling [17] | -                        | 1 . 67 ± 0 . 01    | 2 . 40 ± 1 . 25    | 1M                 | -                        | 4 . 50 ± 0 . 05     | 58 . 04 ± 20 . 98   | 1M                  |
| SGDS                  | 1 . 53 ± 0 . 25          | 1 . 56 ± 0 . 04    | 0 . 89 ± 0 . 19    | 370K               | 33 . 01 ± 0 . 93         | 4 . 25 ± 0 . 08     | 32 . 17 ± 12 . 90   | 45K                 |

evaluate methods using three metrics: the Evidence Lower Bound (ELBO), Evidence Upper Bound (EUBO) [7], and the EUBO -ELBO gap. A smaller gap between ELBO and EUBO indicates a more accurate approximation of the target distribution.

For fair comparison on the number of energy calls, we train the methods until convergence of ELBO and EUBO. To determine convergence, we evaluate based on the moving average of the metrics over the 10 consecutive evaluations, where we evaluate the model every 100 training steps. If a method does not converge within the maximum number of epochs, we report the metrics at the final step.

Baselines. For the manywell potential, the baselines are primarily selected based on their strong performance demonstrated in prior work [39], as well as their methodological relevance [36, 22] or their different framework [1]. Specifically, iDEM [1] utilizes trajectories of length 𝑇 = 1 , 000 for SDE integration, whereas other baselines, including PIS [47], TB [28], AT [22], and GAFN [36], employ shorter diffusion trajectories ( 𝑇 = 100 ) with distinct optimization objectives. We further evaluate enhanced variants of these methods incorporating LP, such as PIS+LP, TB+LP, and FL-SubTB+LP, along with exploration-enhanced (+Expl) or local search (+LS) variants introduced by Sendera et al. [39]. We describe the details of the abbreviations related to baselines in Appendix A.1. For the LJ potentials, we compare against Adjoint Sampling [17], as well as iDEM, PIS, TB, and its variant TB+Expl+LS. We omit LP-based methods due to their divergent training.

Results. As shown in Table 1 and Table 2, our proposed framework consistently achieves superior performance across all high-dimensional tasks: Manywell-64, Manywell-128, and LJ-55. Especially, our method demonstrates the best trade-off between performance and efficiency of energy call.

In Figure 1, one can observe that our method better captures the modes in Manywell-128 when compared to the baselines. As illustrated in Figure 2a and Figure 2b, even increasing the energy budget of baselines does not allow them to surpass the performance of our proposed approach. Also, as shown in Figure 3, our framework generates high quality samples with low energy. Furthermore, for the LJ-55 potential, the distribution of interatomic distances is similar to the ground truth

Figure 3: Histograms for LJ-13/55 energy densities and LJ-55 interatomic distances.

<!-- image -->

distribution. Additionally, our method obtains competitive results with significantly fewer samples in lower-dimensional tasks such as 40GMM, Manywell-32 (see Appendix B.1), and LJ-13 (see Table 2).

## 5.2 Ablation study

MCMC sampler with the same budget. In our method, we consume energy calls during both Searcher sampling and Learner training. To evaluate the efficiency of the Learner's on/off-policy mixing training scheme, we conduct a controlled comparison where the total energy call budget (20M) is entirely allocated to AIS on Manywell-128. We run 200 chains on the trajectories with 𝑇 = 10 , 000 for AIS. As a result, even though high-reward samples were collected using AIS with much longer trajectories, the MLE Learner failed to perform amortized learning as shown in Figure 2c.

Periodic re-initialization and pre-trained flow. We perform an ablation study to evaluate two design components of our method when proceeding to the next training round: (1) re-initializing the Learner model, and (2) retaining the pre-trained log 𝑍 parameter from the previous round. Specifically, to assess the benefit of re-initialization in mitigating primacy bias, we compare our method against a fine-tuning baseline where the second-round Learner continues training from the first-round model weights without re-initialization. To isolate the effect of retaining the estimated log 𝑍 value, we compare against a variant where the log 𝑍 parameter is also re-initialized at the start of the second round. As shown in Figure 2c, our full method outperforms both ablation variants, confirming that re-initialization is beneficial for mitigating primacy bias, and that employing the log 𝑍 parameter across rounds leads to better training stability and performance.

Novelty-based reward in Searcher sampling. We assess the effectiveness of incorporating the novelty-based intrinsic reward derived by RND [10] into the Searcher sampling process in later training rounds. In our framework, starting from the second round, the Searcher sampler drives prior samples in the direction of the target distribution and exploration signal derived from a previously trained RND module, which prioritizes underexplored modes by the Learner sampler. These dynamics guide the Searcher to focus sampling efforts on modes that remain novel and close to the target distribution across rounds. As shown in Figure 2c, at the end of round 2, our RND-augmented approach yields a smaller EUBO-ELBO gap compared to a way of repeating the same Searcher sampling without exploration. These results demonstrate that using intrinsic rewards to adaptively bias Searcher sampling toward novel modes improves overall distributional coverage across rounds.

## 5.3 Application to molecular conformer generation.

We also consider three real-world systems, Alanine Di-, Tri-, and Tetra-peptide, consisting of 23, 33, 43 atoms in vacuum at a temperature of 300K. While some previous works show promising results in sampling conformation of Alanine Dipeptide, they rely on low-dimensional descriptors such as rotatable torsion angles [45]. Solving these three peptides at all-atom resolutions remains a challenge for existing diffusion-based neural samplers.

Settings. To accurately evaluate molecular energies, we employ TorchANI [16], a PyTorch implementation of ANI deep learning potentials trained on quantum-mechanical reference data. For the Searcher, we run four parallel 55ps Langevin dynamics simulations under the TorchANI potential. In the first round, simulations are performed at 600 K to efficiently sample slow degrees of freedom; in the second round, we use 300 K to capture faster motions and collect high-reward samples. The Learner and RND models use the 𝐸 ( 3 ) -equivariant graph neural network (EGNN) architecture [38] based on atomic coordinates. We provide details in Appendix A.4.

Figure 4: Qualitative results of methods in three peptides. (a) Ramachandran plot of Alanine Dipeptide with two backbone torsion angles ( 𝜙, 𝜓 ) and (b) 3D visualization of generated conformations.

<!-- image -->

Baselines. We compare our method with maximum-likelihood estimation (MLE) as well as diffusionbased neural samplers: PIS, TB, and TB+Expl+LS. We generate reference ensemble using 100 ns Langevin-dynamics simulation at 300K. In MLE, the likelihood of forward path distribution is maximized over the samples from backward path distribution. To match the number of energy calls of MLE with our method, we collect 2.5 times more samples than our method, using the same searcher as our method without RND. TB+Expl+LS utilizes the same langevin dynamics for local search algorithm. We omit the LP methods since they have large absolute values of EUBO and ELBO. We exclude comparison with Volokhova et al. [45], as they consider only rotatable torsion angles, and with Midgley et al. [30], which employs a discrete normalizing flow on internal coordinates, whereas our method utilizes a diffusion model in atomic coordinates.

Results. As shown in Table 3, our method outperforms diffusion-based neural samplers and MLE on three peptides. As illustrated in Figure 4, both our method and MLE capture the free energy landscape and generate physically plausible conformations by leveraging high-fidelity samples from the Langevin dynamics Searcher. By contrast, PIS, TB, and TB+Expl+LS fail to reconstruct the target free-energy surface or to produce realistic structures, since their forward policies insufficiently explore the complex landscape. We note that the Langevin dynamics used

Table 3: EUBO-ELBO gaps on peptides in units of 10 3 . Bold indicates the best performance. * indicates large absolute values of metrics, and "Div." indicates divergent training.

| Method          | Dipeptide        | Tripeptide       | Tetrapeptide     |
|-----------------|------------------|------------------|------------------|
| PIS (1M)        | 84 . 65 ± 9 . 93 | 35 . 23 ± 2 . 43 | ∗                |
| TB (1M)         | Div.             | Div.             | Div.             |
| TB+Expl+LS (3M) | 18 . 86 ± 0 . 19 | 6 . 41 ± 0 . 34  | Div.             |
| MLE (1M)        | 17 . 41 ± 0 . 14 | 6 . 11 ± 0 . 02  | 29 . 07 ± 0 . 56 |
| SGDS (1M)       | 17 . 11 ± 0 . 16 | 5 . 60 ± 0 . 03  | 26 . 60 ± 0 . 21 |

in the Searcher yields higher-quality samples than those obtained by local searches of TB+Expl+LS from forward-policy outputs. Furthermore, our approach refines the biased samples from the hightemperature Searcher through an unbiased TB objective, improving performance compared to MLE.

## 6 Conclusion

We have proposed a scalable and sample-efficient sampling framework SGDS that integrates an MCMCSearcher with a diffusion Learner. By leveraging high-quality samples from replay buffers and training the Learner model via on/off-policy TB objectives, our method effectively bridges classical sampling with neural amortization. The inclusion of novelty-based intrinsic rewards by RND further enhances the exploration of the Searcher, enabling informed guidance to underexplored modes throughout multiple rounds.

Our work opened promising directions for integrating learning-based amortization with classical sampling, particularly for tasks where both diversity and precision are crucial. Future extensions include designing multi-agent search systems that leverage classical sampling methods for cooperative strategic exploration in high-dimensional spaces and developing advanced off-policy learning schemes, such as adaptive filtering strategies for the replay buffer.

## Acknowledgements

This work was partly supported by Institute for Information &amp; communications Technology Planning &amp;Evaluation(IITP) grant funded by the Korea government(MSIT) (RS-2019-II190075, Artificial Intelligence Graduate School Support Program(KAIST)), National Research Foundation of Korea(NRF) grant funded by the Ministry of Science and ICT(MSIT) (No. RS-2022-NR072184), GRDC(Global Research Development Center) Cooperative Hub Program through the National Research Foundation of Korea(NRF) grant funded by the Ministry of Science and ICT(MSIT) (No. RS-2024-00436165), and the Institute of Information &amp; Communications Technology Planning &amp; Evaluation(IITP) grant funded by the Korea government(MSIT) (RS-2025-02304967, AI Star Fellowship(KAIST)). Minsu Kim was supported by KAIST Jang Yeong Sil Fellowship.

## References

- [1] Tara Akhound-Sadegh, Jarrid Rector-Brooks, Joey Bose, Sarthak Mittal, Pablo Lemos, ChengHao Liu, Marcin Sendera, Siamak Ravanbakhsh, Gauthier Gidel, Yoshua Bengio, et al. Iterated denoising energy matching for sampling from boltzmann densities. In International Conference on Machine Learning (ICML) , 2024.
- [2] Michael S Albergo and Eric Vanden-Eijnden. Nets: A non-equilibrium transport sampler. arXiv preprint arXiv:2410.02711 , 2024.
- [3] Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, and Yoshua Bengio. Flow network based generative models for non-iterative diverse candidate generation. Neural Information Processing Systems (NeurIPS) , 2021.
- [4] Yoshua Bengio, Salem Lahlou, Tristan Deleu, Edward J Hu, Mo Tiwari, and Emmanuel Bengio. GFlowNet foundations. Journal of Machine Learning Research , 24(210):1-55, 2023.
- [5] Julius Berner, Lorenz Richter, and Karen Ullrich. An optimal control perspective on diffusionbased generative modeling. Transactions on Machine Learning Research (TMLR) , 2024. ISSN 2835-8856.
- [6] Julius Berner, Lorenz Richter, Marcin Sendera, Jarrid Rector-Brooks, and Nikolay Malkin. From discrete-time policies to continuous-time diffusion samplers: Asymptotic equivalences and faster training. arXiv preprint arXiv:2501.06148 , 2025.
- [7] Denis Blessing, Xiaogang Jia, Johannes Esslinger, Francisco Vargas, and Gerhard Neumann. Beyond ELBOs: A large-scale evaluation of variational methods for sampling. In International Conference on Machine Learning (ICML) , 2024.
- [8] Denis Blessing, Julius Berner, Lorenz Richter, and Gerhard Neumann. Underdamped diffusion bridges with applications to sampling. In International Conference on Learning Representations (ICLR) , 2025.
- [9] Ignasi Buch, Toni Giorgino, and Gianni De Fabritiis. Complete reconstruction of an enzymeinhibitor binding process by molecular dynamics simulations. Proceedings of the National Academy of Sciences , 108(25):10184-10189, 2011.
- [10] Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. In International Conference on Learning Representations (ICLR) , 2019.
- [11] Junhua Chen, Lorenz Richter, Julius Berner, Denis Blessing, Gerhard Neumann, and Anima Anandkumar. Sequential controlled langevin diffusions. International Conference on Learning Representations (ICLR) , 2025.
- [12] Pierre Del Moral, Arnaud Doucet, and Ajay Jasra. Sequential Monte Carlo samplers. Journal of the Royal Statistical Society Series B: Statistical Methodology , 68(3):411-436, 2006.
- [13] Tristan Deleu, António Góis, Chris Emezue, Mansi Rankawat, Simon Lacoste-Julien, Stefan Bauer, and Yoshua Bengio. Bayesian structure learning with generative flow networks. Conference on Uncertainty in Artificial Intelligence (UAI) , 2022.

- [14] Manuel Dibak, Leon Klein, Andreas Krämer, and Frank Noé. Temperature steerable flows and boltzmann generators. Physical Review Research , 4(4):L042005, 2022.
- [15] Simon Duane, A.D. Kennedy, Brian J. Pendleton, and Duncan Roweth. Hybrid Monte Carlo. Physics Letters B , 195(2):216-222, 1987.
- [16] Xiang Gao, Farhad Ramezanghorbani, Olexandr Isayev, Justin S Smith, and Adrian E Roitberg. Torchani: a free and open source pytorch-based deep learning implementation of the ani neural network potentials. Journal of chemical information and modeling , 60(7):3408-3415, 2020.
- [17] Aaron J Havens, Benjamin Kurt Miller, Bing Yan, Carles Domingo-Enrich, Anuroop Sriram, Daniel S. Levine, Brandon M Wood, Bin Hu, Brandon Amos, Brian Karrer, Xiang Fu, GuanHorng Liu, and Ricky T. Q. Chen. Adjoint sampling: Highly scalable diffusion samplers via adjoint matching. In International Conference on Learning Representations (ICLR) , 2025.
- [18] Jiajun He, Yuanqi Du, Francisco Vargas, Dinghuai Zhang, Shreyas Padhy, RuiKang OuYang, Carla Gomes, and José Miguel Hernández-Lobato. No trick, no treat: Pursuits and challenges towards simulation-free training of neural samplers. arXiv preprint arXiv:2502.06685 , 2025.
- [19] Geoffrey E Hinton. Training products of experts by minimizing contrastive divergence. Neural computation , 14(8):1771-1800, 2002.
- [20] Emiel Hoogeboom, Vıctor Garcia Satorras, Clément Vignac, and Max Welling. Equivariant diffusion for molecule generation in 3d. In International Conference on Machine Learning (ICML) , 2022.
- [21] Minsu Kim, Taeyoung Yun, Emmanuel Bengio, Dinghuai Zhang, Yoshua Bengio, Sungsoo Ahn, and Jinkyoo Park. Local search GFlowNets. International Conference on Learning Representations (ICLR) , 2024.
- [22] Minsu Kim, Sanghyeok Choi, Taeyoung Yun, Emmanuel Bengio, Leo Feng, Jarrid RectorBrooks, Sungsoo Ahn, Jinkyoo Park, Nikolay Malkin, and Yoshua Bengio. Adaptive teachers for amortized samplers. In International Conference on Learning Representations (ICLR) , 2025.
- [23] Leon Klein and Frank Noé. Transferable boltzmann generators. Neural Information Processing Systems (NeurIPS) , 2024.
- [24] Salem Lahlou, Tristan Deleu, Pablo Lemos, Dinghuai Zhang, Alexandra Volokhova, Alex Hernández-Garcıa, Léna Néhale Ezzine, Yoshua Bengio, and Nikolay Malkin. A theory of continuous generative flow networks. International Conference on Machine Learning (ICML) , 2023.
- [25] Yann LeCun, Sumit Chopra, Raia Hadsell, M Ranzato, Fujie Huang, et al. A tutorial on energy-based learning. Predicting structured data , 1(0), 2006.
- [26] Pablo Lemos, Nikolay Malkin, Will Handley, Yoshua Bengio, Yashar Hezaveh, and Laurence Perreault-Levasseur. Improving gradient-guided nested sampling for posterior inference. In International Conference on Machine Learning (ICML) , 2024.
- [27] Kanika Madan, Jarrid Rector-Brooks, Maksym Korablyov, Emmanuel Bengio, Moksh Jain, Andrei Nica, Tom Bosc, Yoshua Bengio, and Nikolay Malkin. Learning GFlowNets from partial episodes for improved convergence and stability. International Conference on Machine Learning (ICML) , 2022.
- [28] Nikolay Malkin, Moksh Jain, Emmanuel Bengio, Chen Sun, and Yoshua Bengio. Trajectory balance: Improved credit assignment in gflownets. Neural Information Processing Systems (NeurIPS) , 2022.
- [29] Nikolay Malkin, Salem Lahlou, Tristan Deleu, Xu Ji, Edward Hu, Katie Everett, Dinghuai Zhang, and Yoshua Bengio. GFlowNets and variational inference. International Conference on Learning Representations (ICLR) , 2023.

- [30] Laurence Illing Midgley, Vincent Stimper, Gregor N. C. Simm, Bernhard Schölkopf, and José Miguel Hernández-Lobato. Flow annealed importance sampling bootstrap. In International Conference on Learning Representations (ICLR) , 2023.
- [31] Radford M Neal. Annealed importance sampling. Statistics and computing , 11:125-139, 2001.
- [32] Evgenii Nikishin, Max Schwarzer, Pierluca D'Oro, Pierre-Luc Bacon, and Aaron Courville. The primacy bias in deep reinforcement learning. In International Conference on Machine Learning (ICML) , 2022.
- [33] Frank Noé, Christof Schütte, Eric Vanden-Eijnden, Lothar Reich, and Thomas R Weikl. Constructing the equilibrium ensemble of folding pathways from short off-equilibrium simulations. Proceedings of the National Academy of Sciences , 106(45):19011-19016, 2009.
- [34] Frank Noé, Simon Olsson, Jonas Köhler, and Hao Wu. Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning. Science , 365(6457):eaaw1147, 2019.
- [35] Ling Pan, Nikolay Malkin, Dinghuai Zhang, and Yoshua Bengio. Better training of GFlowNets with local credit and incomplete trajectories. International Conference on Machine Learning (ICML) , 2023.
- [36] Ling Pan, Dinghuai Zhang, Aaron Courville, Longbo Huang, and Yoshua Bengio. Generative augmented flow networks. In International Conference on Learning Representations (ICLR) , 2023.
- [37] Peter J Rossky, Jimmie D Doll, and Harold L Friedman. Brownian dynamics as smart monte carlo simulation. The Journal of Chemical Physics , 69(10):4628-4633, 1978.
- [38] Vıctor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E (n) equivariant graph neural networks. In International Conference on Machine Learning (ICML) , pages 9323-9332. PMLR, 2021.
- [39] Marcin Sendera, Minsu Kim, Sarthak Mittal, Pablo Lemos, Luca Scimeca, Jarrid Rector-Brooks, Alexandre Adam, Yoshua Bengio, and Nikolay Malkin. Improved off-policy training of diffusion samplers. Neural Information Processing Systems (NeurIPS) , 2024.
- [40] Yuji Sugita and Yuko Okamoto. Replica-exchange molecular dynamics method for protein folding. Chemical physics letters , 314(1-2):141-151, 1999.
- [41] Charlie B. Tan, Joey Bose, Chen Lin, Leon Klein, Michael M. Bronstein, and Alexander Tong. Scalable equilibrium sampling with sequential boltzmann generators. In Frontiers in Probabilistic Inference: Learning meets Sampling , 2025.
- [42] Austin Tripp, Erik Daxberger, and José Miguel Hernández-Lobato. Sample-efficient optimization in the latent space of deep generative models via weighted retraining. Neural Information Processing Systems (NeurIPS) , 2020.
- [43] Francisco Vargas, Will Grathwohl, and Arnaud Doucet. Denoising diffusion samplers. International Conference on Learning Representations (ICLR) , 2023.
- [44] Francisco Vargas, Shreyas Padhy, Denis Blessing, and Nikolas Nüsken. Transport meets variational inference: Controlled Monte Carlo diffusions. International Conference on Learning Representations (ICLR) , 2024.
- [45] Alexandra Volokhova, Michał Koziarski, Alex Hernández-García, Cheng-Hao Liu, Santiago Miret, Pablo Lemos, Luca Thiede, Zichao Yan, Alán Aspuru-Guzik, and Yoshua Bengio. Towards equilibrium molecular conformation generation with gflownets. Digital Discovery , 3 (5):1038-1047, 2024.
- [46] Dinghuai Zhang, Ricky Tian Qi Chen, Cheng-Hao Liu, Aaron Courville, and Yoshua Bengio. Diffusion generative flow samplers: Improving learning signals through partial trajectory optimization. International Conference on Learning Representations (ICLR) , 2024.
- [47] Qinsheng Zhang and Yongxin Chen. Path integral sampler: a stochastic control approach for sampling. International Conference on Learning Representations (ICLR) , 2022.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: See the methods Section 3 and experiments Section 5 sections.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: See Appendix C. where relevant.

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

Answer: [NA]

Justification: No new theoretical assumptions or results.

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

Justification: See experiment sections Section 5 and references to appendix material Appendix A.

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

Justification: We provide the code to reproduce all of our experimental results in Section 5. Guidelines:

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

Justification: See the experiment sections Section 5 and references to appendix material Appendix A.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All experimental tables include standard deviation and indicate significance of the best metric.

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

Justification: See experiment sections and references to appendix material.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We believe there are no violations of the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The paper studies a ML problem with no immediate societal impacts.

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

Justification: The paper studies a ML problem with no immediate application to generation of new image or text content, nor other functions that have the potential for misuse, to the best of our knowledge.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We cite the works introducing all datasets we study.

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

Justification: No new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: No human studies.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: No human studies.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: LLMs were used only for writing and editing assistance, not in the design or implementation of the core methods.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Experiment details

Code is available at https://github.com/minkyu1022/SGDS .

And the reference samples can be downloaded from https://zenodo.org/records/15436773 .

## A.1 Description of notations

PIS (Path Integral Sampler) [47], TB (Trajectory Balanced) [28], and FL-SubTB (Forward-Looking SubTB) [39] denote the types of objectives. GAFN (Generative Augmented Flow Networks) [36] is a GFlowNet-based method that directly injects intrinsic rewards into TB loss. AT (Adaptive Teacher) [22] introduces an additional neural sampler (the Teacher) trained to focus on high-loss regions of the Student. LP (Langevin Parametrization) [39] refers to a drift construction technique where the model's drift term is combined with the gradient of the target energy function, helping the trained dynamics to better follow the target energy landscape. LS (Local Search) [39] denotes a refinement step where a short Markov chain guided by the target energy gradient is applied from the final state of the generated trajectory to improve sample quality. Expl (Exploration) [39] represents a noise-scheduling technique applied during trajectory generation, in which additional stochasticity is injected more in the early training phase to promote broad exploration and gradually reduced over time for exploitation later.

## A.2 MCMCsamplers for Searcher

Annealed importance sampling (AIS). Annealed importance sampling (AIS) [31] is an MCMC sampling method for estimating the partition functions of target distributions. AIS bridges between an easy-to-sample initial distribution 𝜋 0 ( 𝑥 ) and a target distribution 𝜋 𝑇 ( 𝑥 ) through a sequence of intermediate distributions { 𝜋 𝑡 ( 𝑥 )} 𝑇 𝑡 = 0 , where 𝑇 is the length of a trajectory or chain. Each intermediate distribution 𝜋 𝑡 ( 𝑥 ) typically has the form:

<!-- formula-not-decoded -->

where { 𝛽 𝑡 } is a predefined annealing schedule, and we use 𝛽 𝑡 = 𝑡 𝑇 in our framework. AIS generates samples through an MCMC transition kernel at each intermediate distribution with the following SDE simulation:

<!-- formula-not-decoded -->

where ∇ log 𝜋 𝑡 ( 𝑥 𝑡 ) = ( 1 -𝛽 𝑡 )∇ log 𝜋 0 ( 𝑥 𝑡 ) + 𝛽 𝑡 ∇ log 𝜋 𝑇 ( 𝑥 𝑡 ) is the score function of the annealed distribution (unnormalized). Then it accumulates importance weights given by:

<!-- formula-not-decoded -->

and the expectation of these weights provides an unbiased estimator of the partition function ratio between 𝜋 𝑇 ( 𝑥 ) and 𝜋 0 ( 𝑥 ) :

<!-- formula-not-decoded -->

where 𝑤 ( 𝑖 ) is the importance weight computed for the 𝑖 -th AIS trajectory, and 𝑁 is the total number of trajectories. We compute the unbiased estimation of the log scale of the partition function for Manywell experiments by

<!-- formula-not-decoded -->

where log 𝑍 0 = 0 because the initial distribution is Gaussian in our framework.

Metropolis-Adjusted Langevin Algorithm (MALA). The Metropolis-Adjusted Langevin Algorithm (MALA) is an MCMC method that uses the gradient of the energy function to generate samples from

a target distribution 𝜋 ( 𝑥 ) . MALA starts by sampling initial states 𝑥 0 ∼ 𝜋 0 ( 𝑥 0 ) , where 𝜋 0 (·) is some proposed initial distribution (in most cases, N( 0 , 𝜎 2 𝐼 ) ). It then iteratively proceeds transition from 𝑥 𝑡 to 𝑥 𝑡 + 1 by simulating the following Langevin dynamics:

<!-- formula-not-decoded -->

Here, 𝑥 𝑡 is the current state at time 𝑡 , 𝑊 𝑡 denotes the standard Brownian motion, and E( 𝑥 ) is the energy function of target distribution 𝜋 ( 𝑥 ) , i.e. -∇E( 𝑥 𝑡 ) = ∇ log 𝜋 ( 𝑥 𝑡 ) .

The proposed sample 𝑥 𝑡 + 1 is then accepted or rejected according to the Metropolis-Hastings acceptance probability:

<!-- formula-not-decoded -->

where 𝑞 (· | ·) denotes the Gaussian transition density induced by the Langevin proposal:

<!-- formula-not-decoded -->

The step size Δ 𝑡 is a key factor influencing the quality of sampling. For all tasks, we utilize the scheduling of step size, by comparison between the current acceptance rate and the target acceptance rate (57.4%). We use MALA as Searcher on 40GMM, LJ-13, and LJ-55.

Also, since a MALA trajectory forms a Markov chain, consecutive samples are still correlated and therefore { 𝑥 𝑖 } 𝑁 𝑖 = 1 are not strictly i.i.d. To reduce the most severe correlations we discard the first 𝑀 burn-in iterations as burn-in and use all subsequent states directly. We then compute a rough estimate

<!-- formula-not-decoded -->

where this estimator is biased since 𝑥 𝑖 ∼ 𝜋 ideally and E 𝜋 [ exp (-E( 𝑥 ))] = 𝑍 ∫ 𝜋 2 ( 𝑥 ) 𝑑𝑥 &lt; 𝑍 . Despite the bias, the estimation can provide a numerically reasonable heuristic value for the initialization of the Learner's log 𝑍 𝜃 .

Underdamped Langevin dynamics. For MCMC Searchers of three peptides, we adopt underdamped Langevin dynamics as our molecular dynamics (MD). This framework combines deterministic forces with stochastic fluctuations, which is essential for accurately capturing thermal motion and inertial effects of the molecules. The resulting dynamics are governed by the following system of stochastic differential equations:

<!-- formula-not-decoded -->

Here, 𝑥 𝑡 is the position at time 𝑡 , 𝑣 𝑡 is the velocity, 𝑀 is the mass matrix (symmetric positive definite), E( 𝑥 ) is the potential energy function, and ∇E( 𝑥 𝑡 ) is its gradient with respect to position, i.e., the negative force. The parameter 𝛾 is the friction coefficient, 𝑘 𝐵 is the Boltzmann constant, 𝑇 is the absolute temperature, and 𝑊 𝑡 denotes standard Brownian motion.

For peptides, we use underdamped Langevin dynamics as MD with high temperature(600K). We use Euler-Maruyama integration to discretize the Langevin dynamics. As in MALA, we compute log ˆ 𝑍 for the initialization of log 𝑍 𝜃 in Learner, using Equation (18).

## A.3 Metrics

In this subsection, we formally define the evaluation metrics used to assess the Learner's quality. All metrics are derived from the same importance-weight formulation based on the target partition function.

We begin with the exact log partition function log 𝑍 , which can be written using forward-path importance sampling. Let 𝜏 = ( 𝑥 0 , 𝑥 Δ 𝑡 , . . . , 𝑥 1 ) denote a sample trajectory drawn from the forward

policy 𝑃 𝐹 ( 𝜏 ) , and let 𝑅 ( 𝑥 1 ) be the reward associated with the final state 𝑥 1 . Then, the partition function can be expressed as

<!-- formula-not-decoded -->

where 𝑃 𝐵 ( 𝜏 | 𝑥 1 ) is the backward policy conditioned on the final state.

Since directly optimizing this quantity is intractable, we use two surrogate bounds. The first is the evidence lower Bound (ELBO), defined as

<!-- formula-not-decoded -->

By Jensen's inequality, ELBO is always a lower bound on the true log 𝑍 . It is commonly used as a training objective and can reflect how well the forward policy 𝑃 𝐹 concentrates on high-reward trajectories. However, ELBO can be misleading in practice. A high ELBO does not necessarily imply that all important modes are captured, as the forward policy may collapse to a small subset of modes while still achieving high reward [7].

To address this limitation, we also evaluate the evidence upper Bound (EUBO), which flips the sampling distribution:

<!-- formula-not-decoded -->

Unlike ELBO, EUBO acts as a diagnostic metric. It is an upper bound of log 𝑍 and penalizes missing probability mass. EUBO is driven to penalize missing probability mass and therefore exposes modecollapse that ELBO may hide [7]. And then, true log 𝑍 is consequently bounded by two bounds, i.e., ELBO ≤ log 𝑍 ≤ EUBO.

A smaller gap between the two bounds yields a tighter estimate of log 𝑍 , making this gap a useful indicator of the Learner's sampling quality.

Table 4: Searcher configurations of SGDS

| Benchmark       | 40GMM   | Manywell 32   | Manywell 64   | Manywell 128   | LJ-13   | LJ-55   | Peptides   |
|-----------------|---------|---------------|---------------|----------------|---------|---------|------------|
| Type            | MALA    | AIS           | AIS           | AIS            | MALA    | MALA    | MD         |
| # of Chains     | 300     | 60K           | 60K           | 60K            | 16      | 1       | 4          |
| Chain length    | 4K      | 100           | 100           | 100            | 4K      | 10K     | 110K       |
| Burn-in         | 2K      | -             | -             | -              | 2K      | 4K      | 10K        |
| init. step size | 1e-3    | 1e-3          | 1e-3          | 1e-3           | 1e-5    | 1e-5    | 0.5fs      |

Table 5: Learner configurations of SGDS

| Benchmark                 | 40GMM   | Manywell 32   | Manywell 64   | Manywell 128   | LJ-13   | LJ-55   | Peptides   |
|---------------------------|---------|---------------|---------------|----------------|---------|---------|------------|
| Brownian bridge std ( 𝜎 ) | 10.0    | 1.0           | 1.0           | 1.0            | 0.2     | 0.2     | 0.2        |
| Buffer size               | 600k    | 60k           | 60k           | 60k            | 50K     | 10K     | 800K       |
| Batch size                | 300     | 300           | 300           | 300            | 32      | 4       | 16         |
| Architecture              | MLP     | MLP           | MLP           | MLP            | EGNN    | EGNN    | EGNN       |
| hidden dim                | 256     | 256           | 256           | 256            | 64      | 64      | 128        |
| # of layers               | 2       | 2             | 2             | 2              | 5       | 5       | 5          |
| RND weight                | 100     | 100           | 100           | 100            | 10      | 1       | 10         |

## A.4 Experimental setup

We reproduce iDEM [1] with modified hyperparameter settings to ensure a comparable number of energy calls. Specifically, we adjust the number of MC samples for score estimation and the number of iterative loops. Furthermore, we exclude the additional refinement steps originally applied to the LJ-55 potential in iDEM to maintain consistency across all evaluated methods.

For the diffusion-based neural samplers on synthetic benchmarks, we follow the setup of [39].

Gaussian mixture model with 40 modes (40GMM). Training proceeds in one or two rounds. Our framework achieves competitive performance against baselines even with only a single round, and

shows marginal improvement with a second round. We use MALA as the Searcher, running 300 parallel chains of length 4K, discarding the first 2K steps as burn-in. We maintain a target acceptance rate of 57.4% through step size scheduling, resulting in a total of 2.4M energy evaluations. We use the Gaussian prior with a standard deviation of 21.0 for MALA.

All methods adopt the PIS architecture [47, 39], with a joint network consisting of a two-layer MLP with 256 hidden dimensions. The RND network consists of three layers in the predictor network and the target network, with 256 hidden dimensions. We adopt Brownian bridges as the backward process, with a Brownian motion coefficient of 10.0. We run 25K epochs in both the first round and the second round.

Manywell distributions. We proceed with one or two rounds for training on Manywell distributions. We use AIS as the Searcher, running 60K parallel chains (3K chains * 20 iterations) of length 100, only taking the final step samples. We use the Gaussian prior with a standard deviation of 1.0.

All methods adopt the PIS architecture [47, 39], with a joint network consisting of a two-layer MLP with 256 hidden dimensions. The RND network consists of three layers in the predictor network and the target network, with 256 hidden dimensions. We adopt Brownian bridges as the backward process, with a Brownian motion coefficient of 1.0. We run 25K epochs in the first round and 30K in the second round.

Lennard-Jones (LJ) potentials. Training proceeds in two rounds. We use MALA as the Searcher for two rounds: in LJ-13 we run 16 parallel chains of length 4K corresponding to 64K energy evaluations, discarding the first 2K steps as burn-in and retaining 57.4% accepted samples among remaining 32K samples; in LJ-55 we run a single chain of length 10K corresponding to 10K energy evaluations, discarding the first 4K steps and retaining 57.4% accepted samples among remaining 6K samples. We use the Gaussian prior with a standard deviation of 1.75 for MALA.

All methods utilize five EGNN layers with 64 hidden dimensions. Following [20, 23], we design an E(3)-equivariant generative model initialized from a Dirac delta at the origin, using a mean-free forward transition kernel in inference. The RND network comprises three layers in the predictor network and two in the target network. We adopt Brownian bridges as the backward process for diffusion-based neural samplers, with a Brownian motion coefficient of 0.2. For LJ-13, we run 5K epochs in the first round and 10K in the second round; for LJ-55, 10K and then 20K epochs.

Specifically, we note that the reported performance of the iDEM on Table 2 differs from the original paper [1] due to adjustments, except 𝜎 max and 𝜎 min of the noise scheduling, made to avoid significant discrepancies in energy call usage compared to our method. We reduce the EGNN hidden dimension to 64 and the batch size to 8, and limit the total number of training epochs, including both inner and outer loops, to 15K accordingly. And while the latest iDEM codebase employs 10 steps of Langevin dynamics refinement before evaluation, particularly for LJ-55, we omit this step for fair comparison and instead set the number of samples for MC estimation to 1K. While iDEM reports a lower bound of log 𝑍 computed via importance sampling with its learned proposal density 𝑞 ( 𝑥 ) given by OT-CFM model, we omit this result in our tables. We compute the lower bound based on trajectory-level estimators without training auxiliary models, i.e., CFM. Thus, our reported values are not directly comparable to those from iDEM.

Additionally, in LJ-55, we maximize the log-likelihood of the forward path distribution under the backward process for the first 5K epochs of each round, discretizing backward paths from Brownian bridges initialized with empirical samples collected by Searchers. We also use randomized time scheduling introduced in [6] for our method. We train PIS at a learning rate of 1 𝑒 -4 , TB at a learning rate of 2 𝑒 -4 , and SGDS at a learning rate of 5 𝑒 -4 . We use 4 and 32 batch sizes for all methods except PIS in LJ-13 and LJ-55, respectively. For PIS, we halve these sizes due to the memory limitation required by the forward SDE computational graph.

Peptides. We perform two rounds of search using under-damped Langevin dynamics. In each round, we run four parallel simulations of 55 ps each, with a time step of 0.5 fs, requiring 440K energy evaluations. We discard the first 5 ps of each trajectory as burn-in, then collect 400K samples. Each simulation starts from the same initial position drawn from a Dirac delta distribution, with all initial velocities set to zero. We integrate equations of motion using the Euler-Maruyama integrator, set the friction coefficient 𝛾 = 1 , and use temperature 𝑇 = 600 𝐾 for the first round Searcher and 𝑇 = 300 𝐾 for the second round Searcher.

Similar to LJ potentials, all models utilize five EGNN layers with 128 hidden dimensions. We use a Dirac delta prior distribution at the origin and a mean-free forward transition kernel to guarantee 𝐸 ( 3 ) -equivariance of the marginal density in inference. The Learner network comprises five EGNN layers, while the predictor network and target network in the RND framework contain three and two layers, respectively. As in LJ potentials, we use the Brownian motion coefficient of 0.2. We run 10K epochs in the first round and 20K epochs in the second. As in LJ-55, we maximize the log-likelihood for the first 5K epochs each round. We also utilize randomized time scheduling for our method. We train PIS at a learning rate of 1 𝑒 -4 and all other methods at 5 𝑒 -4 . We use a 16 batch size for all methods except PIS, which uses an 8 batch size due to the memory limitation required by the forward SDE computational graph.

In inference time, we follow [23]. We first align the topology of generated samples with the target bond graph since the architecture and machine learning potential have a degree of freedom in atom ordering. We first match the bond graphs of generated samples with a given bond graph of interest and then correct the chirality of the generated sample to fit the target molecular configuration. The generated sample is rejected if the bond graph is not isomorphic to the target bond graph.

## A.5 Task details

40-Component Gaussian Mixture Model (40GMM). The 40-component Gaussian Mixture Model (GMM) consists of a mixture distribution of 40 Gaussian components, each characterized by a distinct mean vector 𝜇 𝑖 . The energy function for the GMM is defined as:

<!-- formula-not-decoded -->

where 𝑛 = 40 , the weight of each 𝑖 -th Gaussian component is the same, and N( 𝑥 ; 𝜇 𝑖 , 𝜎 2 𝐼 ) is the probability density function of the multivariate Gaussian distribution.

ManyWell distributions. The Manywell potential describes a high-dimensional energy landscape containing multiple wells (local minima), each representing stable states with distinct energy levels. The energy function of Manywell distribution is given by:

<!-- formula-not-decoded -->

where 𝑛 = 𝑑 / 2 is the number of wells, and 𝑑 is the dimensionality of the landscape. Adjusting the dimensionality 𝑑 = 2 𝑛 allows varying the number of wells and complexity, creating tasks like Manywell-32, Manywell-64, and Manywell-128.

Lennard-Jones (LJ) potentials. The Lennard-Jones potential models the interactions between particles. The energy function is defined as:

<!-- formula-not-decoded -->

where 𝜖 and 𝜅 are parameters defining the depth of the potential well and the energy factor, respectively. 𝑟 𝑖 𝑗 = ∥ 𝑥 𝑖 -𝑥 𝑗 ∥ represents the Euclidean distance between particles 𝑖 and 𝑗 . 𝜎 is the characteristic distance at which the potential between two particles vanishes, often interpreted as the van der Waals radius. In our experiments, we set all parameters to 1 . 0 , i.e., 𝜅 = 𝜖 = 𝜎 = 𝜆 = 1 . 0 . Adjusting the number of particles creates tasks such as LJ-13 and LJ-55, increasing the complexity of the particle interactions and resulting in a rugged energy landscape.

TorchANI potential for peptides. We leverage TorchANI [16], a PyTorch implementation of ANI deep-learning potentials trained on quantum-mechanical reference data, to accurately calculate molecular energies. It provides transferable machine learning potential trained on organic molecules for efficient energy and force evaluation with accuracy comparable to density-functional theory (DFT). In particular, TorchANI excels at modeling small peptides.

Table 6: ELBO, EUBO, their gap, and Energy calls on 40GMM and Manywell-32.

|                | 40GMM ( 𝑑 = 2 )   | 40GMM ( 𝑑 = 2 )   | 40GMM ( 𝑑 = 2 )   | 40GMM ( 𝑑 = 2 )   | Manywell ( 𝑑 = 32 )   | Manywell ( 𝑑 = 32 )   | Manywell ( 𝑑 = 32 )   | Manywell ( 𝑑 = 32 )   |
|----------------|-------------------|-------------------|-------------------|-------------------|-----------------------|-----------------------|-----------------------|-----------------------|
| Method         | ELBO ↑            | EUBO ↓            | Gap ↓             | Energy calls      | ELBO ↑                | EUBO ↓                | Gap ↓                 | Energy calls          |
| PIS+LP         | - 1 . 32 ± 0 . 07 | 2 . 42 ± 0 . 20   | 3 . 75 ± 0 . 22   | 300M              | 160 . 83 ± 0 . 41     | 180 . 49 ± 4 . 76     | 19 . 66 ± 4 . 78      | 300M                  |
| TB+LP          | - 0 . 35 ± 0 . 03 | 0 . 53 ± 0 . 04   | 0 . 87 ± 0 . 03   | 160M              | 161 . 42 ± 0 . 40     | 195 . 89 ± 8 . 14     | 34 . 37 ± 8 . 15      | 300M                  |
| FL-SubTB+LP    | - 0 . 36 ± 0 . 01 | 0 . 58 ± 0 . 08   | 0 . 94 ± 0 . 07   | 260M              | 160 . 74 ± 0 . 15     | 215 . 93 ± 4 . 52     | 55 . 19 ± 4 . 52      | 330M                  |
| TB+LS+LP       | - 0 . 38 ± 0 . 03 | 0 . 32 ± 0 . 02   | 0 . 69 ± 0 . 02   | 320M              | 162 . 95 ± 0 . 08     | 166 . 30 ± 0 . 11     | 3 . 35 ± 0 . 14       | 320M                  |
| TB+Expl+LP     | - 0 . 37 ± 0 . 01 | 0 . 32 ± 0 . 02   | 0 . 69 ± 0 . 02   | 300M              | 160 . 76 ± 0 . 13     | 215 . 92 ± 14 . 90    | 55 . 16 ± 14 . 90     | 300M                  |
| TB+Expl+LS+LP  | - 0 . 37 ± 0 . 01 | 0 . 34 ± 0 . 02   | 0 . 71 ± 0 . 02   | 320M              | 162 . 97 ± 0 . 06     | 166 . 25 ± 0 . 10     | 3 . 28 ± 0 . 12       | 320M                  |
| PIS            | - 2 . 03 ± 0 . 22 | 55 . 48 ± 10 . 71 | 57 . 50 ± 9 . 02  | 100M              | 159 . 71 ± 1 . 70     | 333 . 79 ± 3 . 98     | 174 . 08 ± 4 . 33     | 100M                  |
| TB             | - 1 . 35 ± 0 . 04 | 99 . 04 ± 6 . 01  | 100 . 40 ± 5 . 67 | 100M              | 160 . 58 ± 0 . 87     | 439 . 28 ± 166 . 52   | 278 . 70 ± 166 . 49   | 100M                  |
| TB+LS          | - 0 . 38 ± 0 . 03 | 0 . 83 ± 0 . 46   | 1 . 21 ± 0 . 38   | 290M              | 163 . 12 ± 0 . 10     | 166 . 05 ± 0 . 12     | 2 . 93 ± 0 . 16       | 290M                  |
| TB+Expl+LS     | - 0 . 38 ± 0 . 05 | 0 . 58 ± 0 . 34   | 0 . 96 ± 0 . 34   | 290M              | 160 . 87 ± 3 . 31     | 168 . 27 ± 1 . 49     | 7 . 40 ± 3 . 63       | 290M                  |
| GAFN           | ∗                 | ∗                 | ∗                 | N/A               | 161 . 02 ± 0 . 05     | 282 . 40 ± 2 . 02     | 121 . 38 ± 2 . 02     | 100M                  |
| iDEM           | - 2 . 14 ± 0 . 45 | 12 . 75 ± 3 . 67  | 14 . 89 ± 3 . 70  | 300M              | 142 . 23 ± 0 . 40     | 211 . 56 ± 2 . 53     | 69 . 33 ± 2 . 56      | 300M                  |
| SGDS (round 1) | - 0 . 40 ± 0 . 01 | 0 . 33 ± 0 . 02   | 0 . 73 ± 0 . 02   | 6M                | 162 . 49 ± 0 . 05     | 166 . 60 ± 0 . 01     | 4 . 11 ± 0 . 05       | 9M                    |
| SGDS (round 2) | - 0 . 40 ± 0 . 03 | 0 . 33 ± 0 . 05   | 0 . 73 ± 0 . 05   | 12M               | 162 . 63 ± 0 . 01     | 166 . 48 ± 0 . 03     | 3 . 85 ± 0 . 03       | 20M                   |

Figure 5: Mode coverage comparison on 40GMM.

<!-- image -->

## B Additional experimental results

## B.1 Low-dimensional standard benchmarks

Baselines and settings. We benchmark our framework on two standard low-dimensional tasks: 40GMM and Manywell-32. Consistent with the high-dimensional experiments, we report ELBO, EUBO, their gap, and the number of energy calls required during training. We employ the same baseline methods and trajectory configurations (including trajectory length and training objectives) as in the high-dimensional settings. We provide detailed configurations, including diffusion scales for each task, in Table 5.

Results. As demonstrated in Table 6, our method achieves competitive performance on lowerdimensional standard tasks, producing EUBO and ELBO metrics comparable to the strongest baselines, while using significantly fewer energy calls. On the 40GMM task, despite some baselines reporting strong ELBO and EUBO scores, they notably fail to capture the mode located at the bottom-right corner (see Figure 5). In contrast, our framework reliably identifies all modes without sacrificing performance metrics. We report both the first-round and second-round performances of our method, showing that our method attains robust performance on low-dimensional tasks even in the first round, with a slight but consistent improvement observed in the second round.

## B.2 Debiasing of Learner from MCMC Searcher

To address potential biases inherent in MCMC sampling due to finite-length chains, our framework incorporates both off-policy TB training using samples from the Searcher and on-policy TB training. This design choice aims to mitigate biases arising from the Searcher samples alone by enabling the Learner model to adjust toward the target distribution.

To evaluate whether the Learner effectively debiases the samples collected by the Searcher, we compare kernel density estimations (KDE) of samples obtained by the AIS Searcher with those generated by the on/off-policy TB Learner on Manywell distributions. Figure 6 illustrates these KDE comparisons across dimensions 32, 64, and 128.

Due to the varying mode masses assigned in Manywell distributions, even when AIS successfully covers all modes with limited budgets, it struggles to precisely capture the relative mode masses.

None

3

2

1

0

1

2

3

kde

<!-- image -->

kde

Figure 6: KDE figures of AIS( 𝑇 = 100 ), ours, and true samples on Manywell-32/64/128.

Table 7: EUBO-ELBO gap at 20% of the second-round training for different RND weight values.

| RND weight   | 10 0      | 10 1          | 10 2      | 10 3      |
|--------------|-----------|---------------|-----------|-----------|
| Manywell-128 | 549.54    | 542.60        | 538 . 79  | 570.49    |
| LJ-13        | 3.12      | 2 . 67        | 3.90      | 4.77      |
| LJ-55        | 34 . 76   | 36.72         | 44.78     | 51.03     |
| ALDP         | 17,531.95 | 17 , 381 . 04 | 17,417.75 | 17,410.98 |

In contrast, the KDE of the samples generated by the Learner aligns more closely with the true density, effectively reflecting the relative importance of different modes. This result highlights the effectiveness of combining on- and off-policy training to achieve better density approximation than relying solely on finite-budget AIS samples.

## B.3 RND Weight Calibration

To evaluate the sensitivity of our method to the choice of the RND weight, we conducted additional experiments across approximately four different RND weight values per task. The results, summarized in Table Table 7, show that the performance is largely robust to this hyperparameter. The calibration of the RND weight is straightforward: we run 20% of the second-round training to estimate performance and select a reasonable value. As shown in the ELBO-EUBO gap measured at this stage, reported for Manywell-128, LJ potentials, and peptides, the variation across different RND weights is minor, indicating that extensive hyperparameter tuning is unnecessary.

Overall, these results suggest that the method performs consistently across a wide range of RND weights, with negligible degradation in stability or performance. Although a more systematic tuning strategy could be explored in future work, the current approach provides reliable results with minimal calibration effort.

## B.4 Consistency with parallel tempering MD

We further validate the consistency of our method by replacing the Searcher with parallel tempering MD[40]. Parallel tempering (or replica exchange) MD is an enhanced sampling method that runs

multiple independent simulations (or replicas) at different temperatures and periodically attempts to exchange them, allowing low-temperature simulations to overcome energy barriers and escape local minima. Also, we align the number of energy evaluations by collecting additional data for MLE, ensuring a fair comparison. Table 8 shows the ELBO, EUBO, and their gap, demonstrating that our method achieves better metrics than MLE. This result indicates that our approach consistently works even when combined with advanced sampling techniques such as parallel tempering.

Table 8: Comparison of main metrics between parallel tempering + MLE (forward KL) training and our method with parallel tempering Searcher. We align the number of energy calls by collecting more data for MLE.

| Method   | ELBO [ × 10 3 ] ↑ EUBO [ ×   | 10 3 ] ↓          | Gap [ × 10 3 ] ↓   |
|----------|------------------------------|-------------------|--------------------|
| MLE      | 520 . 71 ± 0 . 02            | 538 . 03 ± 0 . 00 | 17 . 32 ± 0 . 02   |
| SGDS     | 521 . 03 ± 0 . 01            | 538 . 02 ± 0 . 00 | 16 . 99 ± 0 . 01   |

## C Limitations

While our framework demonstrates strong empirical performance, several limitations remain.

First, the effectiveness of intrinsic rewards from RND depends on careful tuning of the novelty scale parameter 𝛼 . Poorly calibrated 𝛼 can overly emphasize exploration, producing noisy or irrelevant samples, or conversely yield overly conservative exploration. This could be mitigated by employing adaptive strategies that dynamically adjust 𝛼 during sampling based on diversity metrics or exploration progress signals.

Additionally, the quality of samples provided by the Searcher sets a fundamental exploration limit. If the Searcher fails to adequately explore challenging modes, the Learner will inevitably inherit these limitations, particularly in high-barrier energy landscapes. Introducing enhanced exploration strategies, such as parallel tempering or more advanced proposal schemes like HMC, could improve coverage of hard-to-sample modes.