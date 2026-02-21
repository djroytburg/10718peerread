## How to build a consistency model: Learning flow maps via self-distillation

Nicholas M. Boffi Carnegie Mellon University

## Michael S. Albergo

Harvard University

## Eric Vanden-Eijnden

Courant Institute of Mathematical Sciences

## Abstract

Flow-based generative models achieve state-of-the-art sample quality, but require the expensive solution of a differential equation at inference time. Flow map models, commonly known as consistency models, encompass many recent efforts to improve inference-time efficiency by learning the solution operator of this differential equation. Yet despite their promise, these models lack a unified description that clearly explains how to learn them efficiently in practice. Here, building on the methodology proposed in Boffi et al. (2024), we present a systematic algorithmic framework for directly learning the flow map associated with a flow or diffusion model. By exploiting a relationship between the velocity field underlying a continuous-time flow and the instantaneous rate of change of the flow map, we show how to convert any distillation scheme into a direct training algorithm via self-distillation, eliminating the need for pre-trained teachers. We introduce three algorithmic families based on different mathematical characterizations of the flow map: Eulerian, Lagrangian, and Progressive methods, which we show encompass and extend all known distillation and direct training schemes for consistency models. We find that the novel class of Lagrangian methods, which avoid both spatial derivatives and bootstrapping from small steps by design, achieve significantly more stable training and higher performance than more standard Eulerian and Progressive schemes. Our methodology unifies existing training schemes under a single common framework and reveals new design principles for accelerated generative modeling. Associated code is available at https://github.com/nmboffi/flow-maps.

## 1 Introduction

Generative models based on dynamical systems, such as flows and diffusions, have achieved remarkable successes in vision (Song et al., 2020; Rombach et al., 2022; Ma et al., 2024; Polyak et al., 2025), language (Lou et al., 2024), protein structure prediction (Abramson et al., 2024), weather forecasting (Price et al., 2024), and materials design (Zeni et al., 2025). While highly expressive, dynamical models leverage the solution of a differential equation for sample generation, which typically requires repeated evaluation of the learned model. This computational bottleneck has limited the application of flows and diffusions in domains where rapid inference is crucial, such as real-time control (Black et al., 2024; Chi et al., 2024) and image editing, and as a result has led to intense interest in accelerated inference. One particularly promising approach, which underlies consistency models (Song et al., 2023; Kim et al., 2024), is to estimate the flow map associated with the deterministic probability flow equation instead of the velocity field governing its instantaneous dynamics. The flow map is defined by the integrated flow and can be used to generate samples in as few as one model evaluation, leading to inference that can be 10 -100 × faster than traditional dynamical models. This dramatic speedup

March 14, 2023

A

<!-- image -->

x

t 0 1 x t = cos 2 ( πt )(1 t &lt; 1 2 x 0 +1 t &gt; 1 2 x 1 ) + t (1 t ) z Direct, with latent var. Gaussian handshake x t t x 0 tx 1 t t z x t = cos 2 ( πt )(1 t &lt; 1 2 x 0 +1 t &gt; 1 2 x 1 ) + t (1 t ) z Direct, with latent var. Gaussian handshake x t t x 0 tx 1 t t z x t = cos 2 ( πt )(1 t &lt; 1 2 x 0 +1 t &gt; 1 2 x 1 ) + t (1 t ) z Direct, with latent var. Gaussian handshake Figure 1: Overview. (A) Schematic of the two-time flow map X s,t and the tangent condition (Lemma 2.1), which provides a relation between the map and the drift of the probability flow. The flow map is composable, invertible, and has the property that as t → s , its time derivative recovers the drift b s from (2). (B) Illustration of our proposed parameterization. The function v s,t estimates the slope of the line drawn between two points on a trajectory of the probability flow, and can be directly trained efficiently via the tangent condition.

= (1 -

x

t

)

x

+

tx

t

+

t

(1 -

t

)

z

= (1 -

)

+

+

(1 -

)

= (1 - tz

)

+

+

(1 -

)

t = (1 -) 0 + One-sided x t = (1 t ) x 0 + tz One-sided x t = (1 t ) x 0 + tz One-sided potential motivates the central question: is there a principled methodology for training flow maps, and how can we do so efficiently in practice? In this work,

x

We introduce a direct training framework for flow maps, eliminating the need for pre-trained teacher models while maintaining the training stability of distillation.

March 14, 2023 March 14, 2023 Recently, there have been broad efforts to learn the flow map either directly or through distillation of a pre-trained model (Boffi et al., 2024; Frans et al., 2024; Zhou et al., 2025; Salimans et al., 2024). Distillation-based approaches perform well empirically, but require a two-phase learning setup in which the performance of the student is limited by the performance of the teacher. In these methods, the practitioner first learns a score (Song and Ermon, 2020; Ho et al., 2020) or flow (Lipman et al., 2022; Albergo and Vanden-Eijnden, 2022; Albergo et al., 2023; Liu et al., 2022) model, and then converts it into a flow map via a secondary training algorithm that considers the pre-trained model as a 'teacher' for the 'student' flow map. To avoid this complication, we aim to design learning schemes in which the flow map can be trained similarly to standard flow matching. In this endeavor, the fundamental challenge is a lack of a unified mathematical characterization that reveals how to learn the flow map efficiently, which has led to complex pipelines that require extensive engineering to overcome unstable optimization dynamics outside of the distillation setting (Lu and Song, 2025).

To address this challenge, we introduce a mathematical framework that exposes a landscape of novel training schemes. Our key insight is a simple relation, the tangent condition (Figure 1), that explicitly relates the velocity of the probability flow equation to the derivative of the flow map. Using this insight, we develop a self-distillation framework where the flow map is learned by simultaneously training and distilling its implicit velocity. The result is a simple pipeline that leverages off-the-shelf training procedures for flows to learn a model with accelerated inference. Our framework reveals the fundamental design principles for learning flow maps, enabling practitioners to build few-step generative models as systematically as standard flows. Our main contributions are:

1. Algorithmic framework. We provide three equivalent mathematical characterizations of the flow map, showing how consistency models and other recent few-step methods - including consistency trajectory models (Kim et al., 2024), shortcut models (Frans et al., 2024), mean flow (Geng et al., 2025), and align your flow (Sabour et al., 2025) - emerge as special cases of our methodology.
2. Self-distillation algorithms. Leveraging our description of the flow map, we introduce three new algorithmic families - Eulerian (ESD), Lagrangian (LSD), and Progressive Self-Distillation (PSD) - and discuss their connections to existing direct training schemes. We prove that each has the correct unique minimizer, and provide guarantees that the loss values bound the 2-Wasserstein error of the learned one-step model for ESD and LSD.
3. Empirical analysis. We study the performance of each method as a function of the number of spatial and time derivatives that appear in the objective function. We find that LSD, which avoids both spatial derivatives and self-consistent bootstrapping from smaller steps, attains the best performance across standard benchmarks including the synthetic checkerboard dataset, CIFAR-10, CelebA-64, and AFHQ-64.

## 2 Theoretical framework

In this work, we study the flow map of the probability flow equation, which is a function that jumps between points along a trajectory (Figure 1). Given access to the flow map, samples can be generated in a single step by jumping directly to the endpoint, or can be generated with an adaptive amount of computation at inference time by taking multiple steps. Below, we give a detailed mathematical description of the flow map, which we leverage to design a suite of novel training schemes. We begin with a review of stochastic interpolants, which we use to build efficient flow-based generative models.

## 2.1 Stochastic interpolants and probability flows

Let D = { x i 1 } n i =1 with each x i 1 ∈ R d , x i 1 ∼ ρ 1 denote a dataset drawn from a target density ρ 1 . Given D , our goal is to draw a fresh sample ˆ x 1 ∼ ˆ ρ 1 from a distribution ˆ ρ 1 ≈ ρ 1 learned to approximate the target. Recent methods for accomplishing this task leverage flows, which dynamically evolve samples from a simple base distribution ρ 0 such as a Gaussian until they resemble samples from ρ 1 .

Interpolants. To build a flow-based generative model, we leverage the stochastic interpolant framework (Albergo et al., 2023), which we now briefly recall. We define a stochastic interpolant as a stochastic process I : [0 , 1] × R d × R d → R d that combines samples from the target and the base,

<!-- formula-not-decoded -->

where α, β : [0 , 1] → [0 , 1] are continuously differentiable functions satisfying the boundary conditions α 0 = 1 , α 1 = 0 , β 0 = 0 , and β 1 = 1 . In (1), the pair ( x 0 , x 1 ) ∼ ρ ( x 0 , x 1 ) is drawn from a coupling satisfying the marginal constraints ∫ R d ρ ( x 0 , x 1 ) dx 0 = ρ 1 ( x 1 ) and ∫ R d ρ ( x 0 , x 1 ) dx 1 = ρ 0 ( x 0 ) . By construction, the probability density ρ t = Law ( I t ) defines a path in the space of measures between the base and the target. This path specifies a probability flow that pushes samples from ρ 0 onto ρ 1 ,

<!-- formula-not-decoded -->

which has the same distribution as the interpolant, x t ∼ ρ t for all t ∈ [0 , 1] . The drift b in (2) is given by the conditional expectation of the time derivative of the interpolant, b t ( x ) = E [ ˙ I t | I t = x ] , which averages the 'velocity' of all interpolant paths that cross the point x at time t . A standard choice of coefficients is α t = 1 -t and β t = t (Albergo and Vanden-Eijnden, 2022; Albergo et al., 2023), which recovers flow matching (Lipman et al., 2022) and rectified flow (Liu et al., 2022). Many other options have been considered in the literature, and in addition to flow matching, variance-preserving and variance-exploding diffusions can be obtained as particular cases.

Learning. By standard results in probability theory and statistics, the conditional expectation b t can be learned efficiently in practice by solving a square loss regression problem,

<!-- formula-not-decoded -->

Above, E x 0 ,x 1 denotes an expectation over the random draws of ( x 0 , x 1 ) in the interpolant (1).

Sampling. Given an estimate ˆ b obtained by minimizing (3) over a class of neural networks, we can generate an approximate sample ˆ x 1 by numerically integrating the learned probability flow ˙ ˆ x t = ˆ b t (ˆ x t ) until time t = 1 from an initial condition ˆ x 0 ∼ ρ 0 . This approach yields high-quality samples from complex data distributions in practice, but is computationally expensive due to the need to repeatedly evaluate the learned model during integration; here, we aim to avoid this solve.

## 2.2 Characterizing the flow map

The flow map X : [0 , 1] 2 × R d → R d is the unique map satisfying the jump condition

<!-- formula-not-decoded -->

where ( x t ) t ∈ [0 , 1] is any solution of (2). The condition (4) means that the flow map takes 'steps' of arbitrary size t -s along trajectories of the probability flow. In particular, a single application X 0 , 1 ( x 0 )

with x 0 ∼ ρ 0 yields a sample from ρ 1 , avoiding numerical integration entirely. Moreover, we may also increase the number of steps by composing X t i ,t i +1 over a grid 0 = t 0 &lt; t 1 &lt; ... &lt; t k = 1 in the presence of model errors, which enables us to trade inference-time compute for sample quality.

In what follows, we give three characterizations of the flow map that each lead to an objective for its estimation. As we now show, these characterizations are based on a simple but key result that shows we can deduce the corresponding velocity field b t from a given flow map X s,t .

Lemma 2.1 (Tangent condition) . Let X s,t denote the flow map. Then,

<!-- formula-not-decoded -->

i.e. the tangent vectors to the curve ( X s,t ( x )) t ∈ [ s, 1] give the velocity field b t ( x ) for every x .

As illustrated in Figure 1A, Lemma 2.1 highlights that there is a velocity model 'implicit' in a flow map. To leverage this algorithmically, we propose to adopt an Euler step-like parameterization that takes into account the boundary condition X s,s ( x ) = x ,

<!-- formula-not-decoded -->

In (6), v : [0 , 1] 2 × R d → R d is the function we will estimate parametrically. Despite its similarity to a first-order Taylor expansion, the representation (6) corresponds to a shift and rescaling of X s,t , and hence is without loss of expressivity. In addition to enforcing that X s,t recovers the identity on the diagonal s = t , (6) implies that lim s → t ∂ t X s,t ( x ) = v t,t ( x ) , which gives an elegant connection between v s,t and the drift field b t ,

<!-- formula-not-decoded -->

Geometrically, v s,t describes the 'slope' of the line drawn between x s and x t on a single ODE trajectory (Figure 1B). The condition (7) states that the slope between two infinitesimally-spaced points is precisely the velocity b t . A key insight is that this relation indicates v t,t can be estimated using the objective (3). To learn the map X s,t , it then remains to estimate v s,t away from the diagonal s = t . To this end, we leverage the following result, which relates v s,t to v t,t for s = t .

Proposition 2.2 (Flow map) . Assume that X s,t is given by (6) with v s,t satisfying (7), and assume that v s,t is continuous in both time arguments. Then, X s,t is the flow map defined in (4) if and only if any of the following conditions also holds:

- (i) (Lagrangian condition): X s,t solves the Lagrangian equation

<!-- formula-not-decoded -->

for all ( s, t ) ∈ [0 , 1] 2 and for all x ∈ R d .

- (ii) (Eulerian condition): X s,t solves the Eulerian equation

<!-- formula-not-decoded -->

for all ( s, t ) ∈ [0 , 1] 2 and for all x ∈ R d .

- (iii) (Semigroup condition): For all ( s, t, u ) ∈ [0 , 1] 3 and for all x ∈ R d ,

<!-- formula-not-decoded -->

The Lagrangian and Eulerian conditions in Proposition 2.2 categorize the flow map X s,t as the solution of an infinite system of ODEs or as the solution of a PDE, each of which describes transport along trajectories of the flow (2). The semigroup condition states that any two jumps can be replaced by a single jump. Sections B to D provide a review of the flow map matching framework (Boffi et al., 2024), and describe how these three characterizations are the basis for consistency (Song et al., 2023; Kim et al., 2024; Geng et al., 2024) and progressive distillation (Salimans and Ho, 2022a) schemes that have appeared in the literature. In the following, we show how each - and in fact how any distillation method that produces a flow map from a velocity field ˆ b - can be immediately converted into a direct training objective for a single network model X s,t via the concept of self-distillation.

̸

## 2.3 A framework for self-distillation

Our framework augments training v t,t on the diagonal s = t via the objective (3) and the identity (7) with a penalization term for one or more of the conditions in Proposition 2.2 along the off-diagonal s = t . This leads to a set of objectives that can each be used to learn the flow map.

Proposition 2.3 (Self-distillation) . The flow map X s,t defined in (4) is given for all 0 ⩽ s ⩽ t ⩽ 1 by X s,t ( x ) = x +( t -s ) v s,t ( x ) where v s,t ( x ) the unique minimizer over ˆ v of

̸

where L b (ˆ v ) is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and where L D (ˆ v ) is any of the following three objectives.

(i) The Lagrangian self-distillation (LSD) objective, which leverages (8),

<!-- formula-not-decoded -->

(ii) The Eulerian self-distillation (ESD) objective, which leverages (9),

<!-- formula-not-decoded -->

(iii) The progressive self-distillation (PSD) objective, which leverages (10),

<!-- formula-not-decoded -->

Above, ˆ X s,t ( x ) = x +( t -s )ˆ v s,t ( x ) and E x 0 ,x 1 denotes an expectation over the random draws of ( x 0 , x 1 ) in the interpolant defined in (1).

The proof follows directly from Proposition 2.2 and is given in Section E; the resulting algorithmic approach is summarized graphically in Figure 2. In Proposition 2.3, we focus on s ⩽ t , which is all that is required to generate data. Training over the entire unit square ( s, t ) ∈ [0 , 1] 2 enables jumping backwards from data to noise along trajectories of (2) in addition to standard generation from noise to data. The derivatives with respect to space and time required to implement the LSD and ESD losses can be computed efficiently via standard jvp implementations. As we will see in our experiments, an advantage of the schemes (13) and (15) is they naturally avoid derivatives with respect to space, which leads to significantly improved training stability.

We now provide theoretical guarantees that the ob- jective value bounds the accuracy of the model for LSD and ESD. We were unable to obtain a similar guarantee for PSD due to issues of compounding errors and distribution shift associated with bootstrapping small steps to large steps, which we believe to be a fundamental difficulty to the algorithm's construction. This difficulty is consistent with the observed reduced performance of PSD in comparison to LSD in our experiments (Section 5).

Proposition 2.4 (Wasserstein bounds) . Let ˆ X s,t ( x ) = x +( t -s )ˆ v s,t ( x ) denote a candidate flow map, let ˆ ρ 1 = ˆ X 0 , 1 ♯ρ 0 denote the corresponding one-step generated distribution, and let ˆ L denote the spatial Lipschitz constant of ˆ v t,t ( · ) uniformly in t . First assume L b (ˆ v ) + L LSD (ˆ v ) ⩽ ε . Then,

<!-- formula-not-decoded -->

Now assume that L b (ˆ v ) + L ESD (ˆ v ) ⩽ ε . Then,

<!-- formula-not-decoded -->

<!-- image -->

̸

Figure 2: Self-distillation. Our plug-and-play approach pairs any distillation objective L D on the off-diagonal s = t of the square [0 , 1] 2 with a flow matching objective L b on the diagonal s = t to obtain a direct training algorithm for the flow map.

&lt;latexi sh

1\_b

64="WJ8I

mRDF

3p oG

/Q

PqvY+

Uw

&gt;A

O

cdVTL

B0S2

E

C

y

n

z

j

N

k

u

75

f

XK

M

r

9

Z

H

g

L

L

L

&lt;latexi sh

1\_b

64="

P

gX

9SJ5Gr

MCw pQHc/Z

&gt;A

E23

dVNL

W8

j

y

F

Im

T

Uqn

+

v

K

f

o

Y7

k

u

z

0

DO

R

B

&lt;latexi sh

1\_b

64="jA

E

pVYXng

I

BR

McP

qdT

&gt;

5H

NL

CW8

y

ZF

S

u0

U

7

f

K

v

2

/Q

G

D

+

3

z

wO

r

o

m

J

k

9

Algorithm 1: Learning flow maps via self-distillation input: Dataset D ; interpolant coefficients α t , β t ; batch size M ; diagonal fraction η ; distillation method L D ∈ {L LSD , L ESD , L PSD } . repeat Sample M d = ⌊ ηM ⌋ pairs ( x i 0 , x i 1 ) ∼ ρ ( x 0 , x 1 ) and times t i ∼ U ([0 , 1]) ; Compute interpolants I t i = α t i x i 0 + β t i x i 1 and velocities ˙ I t i = ˙ α t i x i 0 + ˙ β t i x i 1 ; Compute diagonal loss: L b = 1 M d ∑ M d i =1 e -w t i ,t i | ˆ v t i ,t i ( I t i ) -˙ I t i | 2 + w t i ,t i ; Sample M o = M -M d pairs ( x j 0 , x j 1 ) ∼ ρ ( x 0 , x 1 ) and times ( s j , t j ) ∼ U od ; Compute interpolants I s j = α s j x j 0 + β s j x j 1 ; Compute distillation loss: L D = 1 M o ∑ M o j =1 e -w s j ,t j L s j ,t j D (ˆ v ) + w s j ,t j ; Compute L SD = L b + L D ; Update ˆ v and w using ∇L SD ; until converged ; output: Trained flow map ˆ X s,t ( x ) = x +( t -s )ˆ v s,t ( x )

The above result highlights that the model's accuracy improves systematically as the loss is minimized for both ESD and LSD. The proof follows by combining guarantees for distillation-based algorithms (Boffi et al., 2024) with guarantees for flow-based algorithms (Albergo et al., 2023), which can be stitched together by our assumption on the value of the loss.

## 3 Algorithmic aspects

We now provide practical numerical recommendations for an implementation of self-distillation. Our aim is not to provide a single best method, but to devise a general-purpose framework that can be used to build high-performing flow maps across data modalities. We provide a general algorithmic prescription in Algorithm 1, with specific instantiations for LSD, ESD, and PSD in Section F.6.

Choice of teacher. The self-distillation objectives in Proposition 2.3 are obtained by squaring the residuals of the properties in Proposition 2.2. While the minimizers are correct, the associated training dynamics may not be optimal because the losses are nonconvex in ˆ v . The flow of information should be from the diagonal ˆ v t,t , where there is an external learning signal via ˙ I t , to the off-diagonal ˆ v s,t , which bootstraps the signal in ˆ v t,t . To enforce that ˆ v s,t learns to match ˆ v t,t , rather than vice-versa, we use the stopgrad operator to match the distillation setting, where the off-diagonal would adapt entirely to an external teacher. Detailed descriptions of the recommended placement are given in Section F.4.

Relation to existing methods. The generic framework described by Proposition 2.3 and Algorithm 1 recovers most existing schemes for training consistency models and their extensions. In particular, by proper choice of the distillation objective and the teacher, we can obtain standard training for consistency models (Song et al., 2023), consistency trajectory models (Kim et al., 2024), and shortcut models (Frans et al., 2024). These connections are given in detail in Sections C and D.

Loss weighting. The loss (11) can be written explicitly as an integral over the upper triangle s &lt; t ,

<!-- formula-not-decoded -->

where L t b (ˆ v ) = E x 0 ,x 1 [ | ˆ v t,t ( I t ) -˙ I t | 2 ] denotes (12) restricted to t , and where L s,t D (ˆ v ) denotes the distillation term restricted ( s, t ) . We find that loss values at different pairs ( s, t ) can have gradient norms that differ significantly, introducing undesirable variance. To rectify this, we incorporate a learned weight w s,t , generalizing the EDM2 weight (Karras et al., 2024) to the two-time setting,

<!-- formula-not-decoded -->

In (19), w s,t can be interpreted as an estimate of the log-variance of the loss values; at the global minimizer, it ensures that all values of ( s, t ) contribute on a similar scale. We find that using w s,t significantly stabilizes the training dynamics and enables the use of larger learning rates.

Temporal sampling. In addition to the weight, we introduce a sampling distribution p s,t ,

<!-- formula-not-decoded -->

While the weight w s,t normalizes the variance across times, p s,t chooses how we select times randomly for each batch. Let U d denote the uniform distribution on the diagonal s = t , and let U od denote the uniform distribution on the upper triangle s &lt; t . In our experiments, we leverage the mixture distribution p s,t = ηU d +(1 -η ) U od , which places a fraction η of the batch uniformly at random on the diagonal and a fraction (1 -η ) uniformly at random in the upper triangle. Because our distillation losses reduce to the flow matching loss in the limit as s → t (Section F.2), we only use L b on the diagonal s = t . We found η = 0 . 75 to work well in early experiments, which puts the majority of the computational effort towards learning the flow and proportionally less towards distilling it. The distillation objectives in Proposition 2.3 are more expensive than the interpolant objective L b because they require multiple evaluations of the network and Jacobian-vector products. As a result, η can also be used to tune the cost of each training step (Section F.3).

PSD sampling. For PSD, we introduce a proposal distribution p u over the intermediate step,

<!-- formula-not-decoded -->

We parameterize u = γs +(1 -γ ) t as a convex combination for γ ∈ [0 , 1] and define the proposal distribution by sampling over γ . In our experiments, we compare uniform sampling (PSD-U) with γ ∼ U ([0 , 1]) to midpoint sampling where γ ∼ δ 1 / 2 so that γ = 1 / 2 deterministically (PSD-M).

PSD scaling. We show in Section F that (21) may be rewritten entirely in terms of ˆ v as

<!-- formula-not-decoded -->

The form (22) eliminates factors u -s , t -s , and t -u that appear due to the parameterization (6). We found these terms introduced higher gradient variance because they cause the loss to scale like ( t -s ) 2 , which changes the effective learning rate depending on the timestep ( t -s ) ; we drop this factor of ( t -s ) 2 in practice. (22) preconditions the loss and removes this additional source of variability, leading to improved training stability.

Conditioning and guidance. Flow maps can be made conditional by incorporating a conditioning argument c as X s,t ( x ; c ) = x +( t -s ) v s,t ( x ; c ) . We can use this observation to define classifier-free guided (CFG) flow maps, as we now show (Ho and Salimans, 2022). To do so, let c = ∅ correspond to unconditional generation, and let q t ( x ; α, c ) = b t ( x ; ∅ )+ α ( b t ( x ; c ) -b t ( x ; ∅ )) be the CFG velocity field at guidance strength α . This velocity has a flow map X s,t ( x ; α, c ) = x +( t -s ) v s,t ( x ; α, c ) satisfying v t,t ( x ; α, c ) = q t ( x ; α, c ) , which may be learned via self-distillation by incorporating additional random sampling over the guidance scale (Section F.5). In this work, we focus solely on unconditional, unguided generation and leave the usage of guidance for future study.

Multiple models and general representations. In Proposition 2.3, we leverage a single model ˆ X s,t defined in terms of ˆ v s,t . While this leads to greater efficiency through (7), it requires one model to learn both the velocity b and its flow map X , which may require more network capacity than traditional flow-based generative models. Instead, it is also possible to use two models - one parameterizing b and one parameterizing X - which can be trained simultaneously. Higher-order parameterizations can be designed that leverage both, such as ˆ X s,t ( x ) = x +( t -s ) ˆ b s ( x ) + 1 2 ( t -s ) 2 ˆ ψ s,t ( x ) , which can use a frozen pre-trained model ˆ b s or can train ˆ b from scratch in tandem with ˆ ψ . More generally, any parameterization ˆ X s,t satisfying ˆ X s,s ( x ) = x may be used in practice, where in this setting we use lim s → t ∂ t ˆ X s,t ( x ) in place of ˆ v t,t ( x ) in Algorithm 1. This may be computed via automatic differentiation as a jvp in t at s = t . In this work, we focus on the representation (6), which requires only a single model and gives a computationally efficient way to evaluate L b ; we leave these more general and higher-order parameterizations to future work.

## 4 Related work

Flow matching and diffusion models. Our approach builds directly on methods from flow matching and stochastic interpolants (Lipman et al., 2022; Albergo and Vanden-Eijnden, 2022; Albergo

Table 1: Benchmark results. Performance across sampling step counts for the low-dimensional checker dataset (KL divergence) and natural image datasets (FID). Best method per dataset and step count shown in bold .

| Dataset            | Method   | Step Count   | Step Count   | Step Count   | Step Count   | Step Count   |
|--------------------|----------|--------------|--------------|--------------|--------------|--------------|
| Dataset            | Method   | 1            | 2            | 4            | 8            | 16           |
|                    | LSD      | 0 . 086      | 0 . 077      | 0 . 071      | 0 . 070      | 0 . 071      |
|                    | ESD      | 0 . 098      | 0 . 092      | 0 . 083      | 0 . 082      | 0 . 075      |
|                    | PSD-M    | 0 . 146      | 0 . 089      | 0 . 081      | 0 . 072      | 0 . 069      |
|                    | PSD-U    | 0 . 111      | 0 . 107      | 0 . 075      | 0 . 073      | 0 . 068      |
| CIFAR-10 (FID ↓ )  | LSD      | 8 . 100      | 4 . 370      | 3 . 340      | 3 . 330      | 3 . 570      |
| CIFAR-10 (FID ↓ )  | PSD-M    | 12 . 810     | 8 . 430      | 5 . 960      | 5 . 070      | 4 . 640      |
| CIFAR-10 (FID ↓ )  | PSD-U    | 13 . 610     | 7 . 950      | 6 . 030      | 5 . 320      | 5 . 160      |
| CelebA-64 (FID ↓ ) | LSD      | 12 . 220     | 5 . 740      | 3 . 180      | 2 . 180      | 1 . 960      |
| CelebA-64 (FID ↓ ) | PSD-M    | 19 . 640     | 11 . 750     | 7 . 890      | 6 . 060      | 5 . 090      |
| CelebA-64 (FID ↓ ) | PSD-U    | 18 . 810     | 11 . 020     | 7 . 470      | 6 . 000      | 5 . 630      |
|                    | LSD      | 11 . 190     | 7 . 780      | 7 . 000      | 5 . 890      | 5 . 610      |
|                    | PSD-M    | 18 . 860     | 14 . 750     | 14 . 400     | 13 . 260     | 11 . 070     |
|                    | PSD-U    | 14 . 500     | 10 . 730     | 10 . 990     | 12 . 020     | 11 . 470     |

et al., 2023; Liu et al., 2022) as well as the probability flow equation associated with diffusion models (Song et al., 2020; Ho et al., 2020; Maoutsa et al., 2020; Boffi and Vanden-Eijnden, 2023). These methods define an ordinary differential equation whose solution evaluates the flow map at a single time. Due to the computational expense associated with solving these equations, a line of recent work asks how to resolve the flow more efficiently with higher-order numerical solvers (Dockhorn et al., 2022; Lu et al., 2022; Karras et al., 2022; Li et al., 2024) and parallel sampling schemes (Chen et al., 2024; Bortoli et al., 2025). Our approach instead estimates the flow map to enable accelerated sampling by avoiding the differential equation solve altogether.

Consistency models. Appearing under several names, the flow map has become a central object of study in recent efforts to obtain accelerated inference. Consistency models (Song et al., 2023; Song and Dhariwal, 2023) estimate the single-time flow map to jump from any time s to data, given by X s, 1 in our notation. Consistency trajectory models (Kim et al., 2024; Li and He, 2025; Luo et al., 2023) estimate the two-time flow map, enabling multistep sampling. Both approaches implicitly leverage the Eulerian characterization (9), which we find leads to gradient instability, explaining recent engineering efforts for stable training (Lu and Song, 2024). Progressive distillation (Salimans and Ho, 2022a) uses the semigroup condition (10) to train a model that can recursively replicate two steps of a pre-trained teacher. Progressive flow map matching (Boffi et al., 2024) enforces this iteratively over a flow map after pre-training, while shortcut models apply a discretized semigroup condition (Frans et al., 2024). In concurrent work, Geng et al. (2025); Sabour et al. (2025) introduce distillation and direct training schemes that reduce to a particular case of our Eulerian formulation. Details on these methods and their connection to our framework may be found in Sections B to D.

## 5 Numerical experiments

We test LSD, ESD, PSD-U, and PSD-M on the low-dimensional checkerboard dataset, as well as in the high-dimensional setting of unconditional image generation on CIFAR-10, CelebA-64, and AFHQ-64. In each case, we study performance at fixed training time to obtain a fair comparison. We emphasize that our aim is not to obtain state of the art performance, but to understand the trade-offs of each approach and compare them on an equal footing; with further engineering, quantitative metrics could be lowered significantly for all methods. For image datasets, we find ESD to be unstable due to the spatial gradient, leading to poor performance without gradient stabilization schemes. We find that LSD obtains uniformly the best performance on all problems tried. This is consistent with our theoretical results in Proposition 2.4, where we were able to obtain stronger theoretical guarantees for LSD than for PSD. Full network and training details are provided in Section G and Table 2.

Figure 3: Checker dataset. Qualitative results for the two-dimensional checker dataset. LSD performs the best across all step counts except N = 16 (Table 1). All methods improve as the number of steps increase. ESD and both PSD variants fail to capture the sharp boundaries at small N , introducing artifacts and driving KL higher.

<!-- image -->

Checkerboard. While synthetic, the checkerboard dataset exhibits multimodality, sharp boundaries, and low-dimensionality that make it a useful testbed for exact visualization of how few-step samplers capture complex features in the target. Qualitative results are shown in Figure 3, while quantitative results obtained by estimating the KL divergence (for details, again see Section G) between generated samples and the target are shown in Table 1. LSD performs best across all sampling steps tried except for N = 16 , where all methods perform well. The performance of LSD also saturates around N = 4 sampling steps. By contrast, ESD, PSD-U, and PSD-M all see increased performance up to N = 16 steps with reduced performance for fewer steps. The qualitative results in Figures 3 and 6 highlight that the higher KL values result from a failure to capture the sharp features present in the dataset, with ESD blurring the boundaries and the PSD methods introducing artifacts that connect the modes.

CIFAR-10. In Figure 4, we study the parameter gradient norm as a function of the training iteration on CIFAR-10. LSD and PSD, which avoid computing spatial derivatives of the network during training, maintain significantly more stable gradients than ESD even when using sg ( · ) . We found the high gradient norm of ESD to induce training instability, ultimately leading to divergence. This is consistent with earlier work on consistency models, where careful annealing schedules, clipping, and network design has been necessary to stabilize continuous-time training (Lu and Song, 2025).

We track the quantitative performance of each method as measured by FID in Table 1; we do not report FID values for ESD due to training instability. We find that LSD obtains the best performance across all step counts followed by PSD. PSD-U and PSD-M trade places depending on step count. A qualitative visualization of sample quality is shown in Figure 5 (Top) as a function of the number of sampling steps. We see that each method obtains improved quality as the number of steps increases, and that all methods produce similar images for fixed seed.

CelebA-64. As shown in Table 1, LSD also obtains the best performance across all step counts on CelebA-64 (Liu et al., 2015), with FID scores ranging from 12.22 at N = 1 to 1.96 at N = 16 . The gap between LSD and the PSD variants is more pronounced on CelebA-64 than on CIFAR-10, particularly for low step counts. PSD-U mostly outperforms PSD-M, with PSD-M only obtaining a higher-

Figure 4: CIFAR-10: Parameter gradient norms. Spatial and temporal representations in the flow map impact parameter gradient norms of self-distillation methods that require network time and space derivatives.

<!-- image -->

Figure 5: Progressive refinement. Sample quality as a function of sampling steps using the same eight fixed noise samples across all methods for fair comparison. (Top) CIFAR-10, (Middle) CelebA-64, (Bottom) AFHQ-64. LSD consistently produces coherent samples across all datasets and step counts.

<!-- image -->

performing 16 -step map. A qualitative visualization is shown in Figure 5 (Middle). All methods show systematic improvement as the step count increases, with faces becoming sharper and more detailed.

AFHQ-64. Finally, we evaluate on AFHQ-64, a more challenging dataset with greater visual diversity than CelebA-64 that includes variation across animal categories (Choi et al., 2020). As shown in Table 1, LSD again achieves the best FID scores across all step counts, ranging from 11.19 at N = 1 to 5.61 at N = 16 . PSD shows notably higher FID scores on this dataset, particularly PSD-M, which struggles at low step counts. PSD-U again mostly outperforms PSD-M, with PSD-M obtaining a slightly higher-performing 16 -step map but worse performance otherwise. Qualitative results are shown in Figure 5 (Bottom), where we again see that higher step counts lead to generated images with increasing levels of detail.

## 6 Conclusion

In this work, we expose and investigate the design space of a class of flow-based generative models with accelerated inference known as flow maps . These models generalize and extend consistency models to include multiple training paradigms and principled multistep inference. Rather than learning the velocity field typical of flows and diffusions, flow maps learn the solution operator of the probability flow equation, obviating the need to solve a differential equation for inference. We show that learning can be performed directly by pairing flow training with any of three characterizations of the flow map, an approach we refer to as self-distillation. Self-distillation can be incorporated with minimal additional overhead, making flow maps an appealing new paradigm. While we systematically categorize the design space of flow map models, the main limitation of our contribution is that we were unable to systematically test each component empirically due to the large associated computational expense. Critical aspects deserving further experimentation include ablations over the flow map parameterization and architecture; stabilization, annealing, and stopgradient schemes for training; and hybrid approaches that combine multiple of our self-distillation objectives.

## Acknowledgments

MSA is supported by a Junior Fellowship at the Harvard Society of Fellows as well as the National Science Foundation under Cooperative Agreement PHY-2019786 (The NSF AI Institute for Artificial Intelligence and Fundamental Interactions, http://iaifi.org/). NMB would like to thank Max Simchowitz, Andrej Risteski, Stephen Huan, Jerry Huang, Chaoyi Pan, Giri Anantharaman, and Gabe Guo for helpful conversations.

## References

- Nicholas M. Boffi, Michael S. Albergo, and Eric Vanden-Eijnden. Flow Map Matching: A unifying framework for consistency models. arXiv:2406.07507 , June 2024.
- Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv:2011.13456 , 2020.
- Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 10684-10695, 2022.
- Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, and Saining Xie. SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers. arXiv:2401.08740 , 2024.
- Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, David Yan, Dhruv Choudhary, Dingkang Wang, Geet Sethi, Guan Pang, Haoyu Ma, Ishan Misra, Ji Hou, Jialiang Wang, Kiran Jagadeesh, Kunpeng Li, Luxin Zhang, Mannat Singh, Mary Williamson, Matt Le, Matthew Yu, Mitesh Kumar Singh, Peizhao Zhang, Peter Vajda, Quentin Duval, Rohit Girdhar, Roshan Sumbaly, Sai Saketh Rambhatla, Sam Tsai, Samaneh Azadi, Samyak Datta, Sanyuan Chen, Sean Bell, Sharadh Ramaswamy, Shelly Sheynin, Siddharth Bhattacharya, Simran Motwani, Tao Xu, Tianhe Li, Tingbo Hou, WeiNing Hsu, Xi Yin, Xiaoliang Dai, Yaniv Taigman, Yaqiao Luo, Yen-Cheng Liu, Yi-Chiao Wu, Yue Zhao, Yuval Kirstain, Zecheng He, Zijian He, Albert Pumarola, Ali Thabet, Artsiom Sanakoyeu, Arun Mallya, Baishan Guo, Boris Araya, Breena Kerr, Carleigh Wood, Ce Liu, Cen Peng, Dimitry Vengertsev, Edgar Schonfeld, Elliot Blanchard, Felix Juefei-Xu, Fraylie Nord, Jeff Liang, John Hoffman, Jonas Kohler, Kaolin Fire, Karthik Sivakumar, Lawrence Chen, Licheng Yu, Luya Gao, Markos Georgopoulos, Rashel Moritz, Sara K. Sampson, Shikai Li, Simone Parmeggiani, Steve Fine, Tara Fowler, Vladan Petrovic, and Yuming Du. Movie gen: A cast of media foundation models. arXiv:2410.13720 , 2025.
- Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution. arXiv:2310.16834 , June 2024.
- Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf Ronneberger, Lindsay Willmore, Andrew J. Ballard, Joshua Bambrick, Sebastian W. Bodenstein, David A. Evans, Chia-Chun Hung, Michael O'Neill, David Reiman, Kathryn Tunyasuvunakool, Zachary Wu, Akvil˙ e Žemgulyt˙ e, Eirini Arvaniti, Charles Beattie, Ottavia Bertolli, Alex Bridgland, Alexey Cherepanov, Miles Congreve, Alexander I. Cowen-Rivers, Andrew Cowie, Michael Figurnov, Fabian B. Fuchs, Hannah Gladman, Rishub Jain, Yousuf A. Khan, Caroline M. R. Low, Kuba Perlin, Anna Potapenko, Pascal Savy, Sukhdeep Singh, Adrian Stecula, Ashok Thillaisundaram, Catherine Tong, Sergei Yakneen, Ellen D. Zhong, Michal Zielinski, Augustin Žídek, Victor Bapst, Pushmeet Kohli, Max Jaderberg, Demis Hassabis, and John M. Jumper. Accurate structure prediction of biomolecular interactions with alphafold 3. Nature , 630(8016):493-500, 2024.
- Ilan Price, Alvaro Sanchez-Gonzalez, Ferran Alet, Tom R. Andersson, Andrew El-Kadi, Dominic Masters, Timo Ewalds, Jacklynn Stott, Shakir Mohamed, Peter Battaglia, Remi Lam, and Matthew Willson. GenCast: Diffusion-based ensemble forecasting for medium-range weather, May 2024.
- Claudio Zeni, Robert Pinsler, Daniel Zügner, Andrew Fowler, Matthew Horton, Xiang Fu, Zilong Wang, Aliaksandra Shysheya, Jonathan Crabbé, Shoko Ueda, Roberto Sordillo, Lixin Sun, Jake

Smith, Bichlien Nguyen, Hannes Schulz, Sarah Lewis, Chin-Wei Huang, Ziheng Lu, Yichi Zhou, Han Yang, Hongxia Hao, Jielan Li, Chunlei Yang, Wenjie Li, Ryota Tomioka, and Tian Xie. A generative model for inorganic materials design. Nature , 639(8055):624-632, 2025.

- Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. π 0 : A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164 , October 2024.
- Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. arXiv:2303.04137 , 2024.
- Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency Models. arXiv:2303.01469 , 2023.
- Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion. arXiv:2310.02279 , 2024.
- Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. arXiv:2410.12557 , 2024.
- Linqi Zhou, Stefano Ermon, and Jiaming Song. Inductive moment matching. arXiv:2503.07565 , 2025.
- Tim Salimans, Thomas Mensink, Jonathan Heek, and Emiel Hoogeboom. Multistep Distillation of Diffusion Models via Moment Matching. arXiv:2406.04103 , June 2024.
- Yang Song and Stefano Ermon. Generative Modeling by Estimating Gradients of the Data Distribution. arXiv:1907.05600 , 2020.
- Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in neural information processing systems , volume 33, pages 6840-6851, 2020.
- Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations , 2022.
- Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In The Eleventh International Conference on Learning Representations , 2022.
- Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797 , 2023.
- Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations , 2022.
- Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv:2410.11081 , 2025.
- Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, and Kaiming He. Mean Flows for One-step Generative Modeling. arXiv:2505.13447 , May 2025.
- Amirmojtaba Sabour, Sanja Fidler, and Karsten Kreis. Align Your Flow: Scaling Continuous-Time Flow Map Distillation. arXiv:2506.14603 , June 2025.
- Zhengyang Geng, Ashwini Pokle, William Luo, Justin Lin, and J. Zico Kolter. Consistency Models Made Easy. arXiv:2406.14548 , 2024.
- Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv:2202.00512 , 2022a.

- Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and Improving the Training Dynamics of Diffusion Models. arXiv:2312.02696 , March 2024.
- Jonathan Ho and Tim Salimans. Classifier-Free Diffusion Guidance. arXiv:2207.12598 , July 2022.
- Dimitra Maoutsa, Sebastian Reich, and Manfred Opper. Interacting particle solutions of FokkerPlanck equations through gradient-log-density estimation. Entropy , 22(8):802, July 2020.
- Nicholas M. Boffi and Eric Vanden-Eijnden. Probability flow solution of the Fokker-Planck equation. Machine Learning: Science and Technology , 4(3):035012, July 2023.
- Tim Dockhorn, Arash Vahdat, and Karsten Kreis. Genie: Higher-order denoising diffusion solvers. arXiv:2210.05475 , 2022.
- Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. arXiv:2206.00927 , 2022.
- Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. arXiv:2206.00364 , 2022.
- Gen Li, Yu Huang, Timofey Efimov, Yuting Wei, Yuejie Chi, and Yuxin Chen. Accelerating convergence of score-based diffusion models, provably. arXiv:2403.03852 , 2024.
- Haoxuan Chen, Yinuo Ren, Lexing Ying, and Grant M. Rotskoff. Accelerating diffusion models with parallel sampling: Inference at sub-linear time complexity. arXiv:2405.15986 , 2024.
- Valentin De Bortoli, Alexandre Galashov, Arthur Gretton, and Arnaud Doucet. Accelerated diffusion models via speculative sampling. arXiv:2501.05370 , 2025.
- Yang Song and Prafulla Dhariwal. Improved Techniques for Training Consistency Models. arXiv:2310.14189 , 2023.
- Liangchen Li and Jiajun He. Bidirectional consistency models. arXiv:2403.18035 , 2025.
- Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv:2310.04378 , 2023.
- Cheng Lu and Yang Song. Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models. arXiv:2410.11081 , October 2024.
- Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV) , December 2015.
- Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. Stargan v2: Diverse image synthesis for multiple domains. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , 2020.
- Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. DreamFusion: Text-to-3D using 2D Diffusion. arXiv:2209.14988 , September 2022.
- Tim Salimans and Jonathan Ho. Progressive Distillation for Fast Sampling of Diffusion Models. arXiv:2202.00512 , 2022b.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Yes, we introduce a novel class of self-distillation algorithms for learning flow maps, and we study their performance numerically.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Yes, we discuss how the main limitation of our work is a lack of state of the art results due to limited computational capabilities, and that our conclusions can be architecture and dataset dependent. We plan to improve upon the quantitative values in the revision.

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

Justification: We include several theoretical results in the paper, each of which clearly states the assumptions and includes a correct proof.

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

Justification: We include full experimental details in the main text and appendix, and we plan to release our code, checkpoints, and configuration files.

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

Justification: Yes, we plan to release a well-documented open source release with the camera-ready version. Included in this will be an open-source jax implementation of the EDM2 neural network.

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

Justification: All experimental details are provided in the main text with further details in the appendix, and will be included in the released configuration files.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: We did not report error bars for our quantitative metrics, but we used a very large batch for calculating the estimated KL divergence, so did not observe any variability empirically. We reviewed the literature and found that it is common to not report error bars for FID values, but we are happy to include them in the revision.

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

Justification: We provided the number and type of GPUs used to run the experiments in the main text.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our work is primarily about the theoretical and empirical study of training moderate-scale generative models, and does not come with significant ethical implications outside of the usual caveats surrounding generative AI broadly speaking.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: As stated in the previous answer, outside of the usual caveats surrounding generative AI, we do not believe there to be significant ethical or broader impacts related to our work.

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

Justification: We do not believe that this paper poses such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: We do not use any existing assets outside of the EDM2 network, which is cited clearly.

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

Justification: We have released an open-source implementation of our method that reproduces all experimental results, which comes with documentation.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our work does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our work does not involve crowdsourcing nor research with humans subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: Our work does not involved LLMs as a core component.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Background on stochastic interpolants

For the reader's convenience, we now recall how to construct probability flow equations using stochastic interpolants. We remark that score-based diffusion models can also be cast in the form of a stochastic interpolant (1), though they do not usually satisfy the exact boundary conditions at t = 0 and t = 1 , and the direction of time is opposite by convention. For example, the variance exploding and EDM processes (Karras et al., 2022) naturally fit within this form as X t = X 0 + σ t z , while the variance preserving process can be cast in this form after solving the Ornstein-Uhlenbeck process dX t = -X t dt + √ 2 dW t in distribution as X t d = exp( -t ) X 0 + √ 1 -exp( -2 t ) z with z ∼ N (0 , I ) . In both cases, we may define I t = X T -t to flip the direction of time over the horizon T .

The key property of the interpolant construction is that solutions to the probability flow (2) push forward their initial conditions onto samples from the target by matching the time-dependent density of (1), as we now show.

Lemma A.1 (Transport equation) . Let ρ t = Law ( I t ) be the density of the stochastic interpolant (1) . Then ρ t satisfies the transport equation

<!-- formula-not-decoded -->

where ∇ denotes a gradient with respect to x , and where b t ( x ) = E [ ˙ I t | I t = x ] .

Proof. The proof proceeds via the weak form of (23). Let ϕ ∈ C 1 b ( R d ) denote an arbitrary continuously differentiable and compactly supported test function. By definition,

<!-- formula-not-decoded -->

where I t is given by (1). Taking the time derivative of this equality, we deduce that

<!-- formula-not-decoded -->

The first line follows by the chain rule, the second by the tower property of the conditional expectation and the definition of the drift b t , and the third by definition of ρ t . The last line is the weak form of the transport equation (23).

Anearly-identical derivation (simply dropping the tower property step) shows that the probability flow equation (2) satisfies Law ( x t ) = ρ t = Law ( I t ) . Together, these results imply that we can sample from any density ρ t solving a transport equation of the form (23) by solving the corresponding ordinary differential equation (2). In practice, we may implement this algorithmically by approximating b t with a neural network via minimization of (3) to obtain a model ˆ b t , and then solving the associated differential equation ˙ ˆ x t = ˆ b t (ˆ x t ) from t = 0 to t = 1 with an initial condition ˆ x 0 ∼ ρ 0 to obtain approximate samples from ρ 1 .

## B Background on flow map matching.

As discussed in Boffi et al. (2024), given a pre-trained velocity field ˆ b , we may leverage the three properties in Proposition 2.2 to design efficient distillation schemes by minimizing the corresponding square residual. In the following, we use the notation L ( ˆ X ; ˆ b ) or L ( ˆ X ; ˇ X ) to denote a loss function for the flow map ˆ X given the teacher (which remains frozen during training).

Stopgradients. Because ˆ b is a pre-trained teacher, its parameters are frozen during training. The self-distillation schemes we introduce in this work replace the teacher network ˆ b s by a self-consistent implicit teacher ˆ v s,s , eliminating the need for the pre-trained model entirely. Inspired by the distillation setting, in Section F.4, we will use a stopgradient operator sg ( · ) in the context of self-distillation schemes to create a similar effect to a frozen teacher and to control the flow of information within the model. Nevertheless, for training stability, it has been observed that it can be useful to use additional sg ( · ) operators even for distillation, which we discuss after introducing each loss.

## B.1 Lagrangian distillation.

The first approach is the Lagrangian map distillation (LMD) algorithm, which is based on (8) and is the basis for the LSD algorithm,

<!-- formula-not-decoded -->

The Lagrangian scheme (26) was introduced in Boffi et al. (2024), and to our knowledge has not appeared in other works. While ˆ b is frozen, the loss (26) is nonconvex in ˆ X due to the nonlinearity of ˆ b . Moreover, computing the gradient of (26) with respect to ˆ X (or its parameters) requires computing the spatial Jacobian of ˆ b , which has been observed to be problematic for large generative models such as image synthesis systems (Poole et al., 2022). For these reasons, it is common to use the modified loss function

<!-- formula-not-decoded -->

The effectiveness of (27) over (26) depends on the data modality and the neural network architecture, as the spatial Jacobian is only problematic in some contexts depending on the pre-trained teacher. We refer to the gradient of a loss function such as (27) - which includes the sg ( · ) operator - as a semigradient .

## B.2 Eulerian distillation.

A second scheme is the Eulerian map distillation (EMD) method based on (9),

<!-- formula-not-decoded -->

Unlike the Lagrangian approach, (28) is convex in ˆ X . Nevertheless, taking the gradient with respect to the parameters of ˆ X requires backpropagating through its spatial Jacobian, which can be similarly problematic as the setting described for (26). One fix is to use a semigradient based on

<!-- formula-not-decoded -->

which avoids backpropagating through the spatial Jacobian entirely. While this helps training stability, it has been observed by Boffi et al. (2024) that the Lagrangian schemes (26) and (27) are more stable than the Eulerian schemes (28) and (29), which is consistent with our experiments in Section 5.

## B.3 Progressive flow map matching.

We now describe the progressive flow map matching (PFMM) algorithm, which is inspired by progressive distillation (Salimans and Ho, 2022b) for diffusion models, but adapted to the stochastic interpolant and two-time flow map setting. Let ˇ X s,t denote a pre-trained teacher flow map, assumed to be valid over the range 0 ⩽ s ⩽ t ⩽ τ . To obtain such a map at initialization, we may take τ = ∆ t and set ˇ X s,t ( x ) = x +( t -s ) ˆ b s ( x ) with a pre-trained flow map ˆ b , corresponding to a single Euler step of size ( t -s ) ⩽ τ = ∆ t . Our aim is to 'extend' ˇ X over a larger range, say 0 ⩽ s ⩽ t ⩽ 2 τ , by training a second flow map ˆ X s,t to match two steps of ˇ X s,t . To do so, we consider the objective

<!-- formula-not-decoded -->

which is based on the semigroup property (10). In words, (30) teaches ˆ X to replicate two jumps of ˇ X in one larger jump. We may also apply (30) self-consistently, where ˆ X itself serves as the teacher,

<!-- formula-not-decoded -->

after the first round where ˆ b is used, and extend τ over the course of optimization according to a pre-defined annealing scheme. Our general self-distillation framework described in Section 2.3 may be obtained by using one of the above distillation schemes in tandem with direct training of ˆ v , and where we use ˆ v as the teacher velocity field for the student flow map model ˆ X .

## C Connection to consistency models.

The approaches (28) and (29) are directly related to consistency distillation in the continuous-time limit (Song and Dhariwal, 2023; Lu and Song, 2024). Consistency models estimate the singletime flow map from noise to data, which in our notation is given by X s, 1 . Consistency trajectory models (Kim et al., 2024) use the same approach to learn the two-time map X s,t ; for agreement with the main text, we focus on this setting here.

## C.1 Consistency distillation and Align Your Flow.

We first take a continuous-time limit of the discrete-time consistency distillation objective. Discretetime consistency distillation considers the loss

<!-- formula-not-decoded -->

In words, L CD aims to make ˆ X s,t 'consistent' on trajectories of the teacher's probability flow ˆ x t by using a shorter step of size ( t -s -∆ s ) as a teacher for a slightly larger step of size ( t -s ). Taking the gradient with respect to ˆ X s,t , we find

<!-- formula-not-decoded -->

To obtain the gradient with respect to the parameters θ of ˆ X , we have by the chain rule that ∇ θ L CD = ∇ θ ˆ X s,t δ L CD δ ˆ X s,t , so we focus on the functional derivative for notational simplicity. Taylor expanding, we find that

<!-- formula-not-decoded -->

In the last line, we used that ∆ s ∇ ˆ X s +∆ s,t ( I s ) = ∆ s ∇ ˆ X s,t ( I s ) + o (∆ s ) . With this, we find

<!-- formula-not-decoded -->

which is simply the negative Eulerian residual.

We now ask if the semigradient (35) can be obtained from the Eulerian distillation objective (28) with a certain choice of sg ( · ) . To do so, we consider the specific parameterization (6) given by ˆ X s,t ( x ) = x +( t -s )ˆ v s,t ( x ) . In this case, the Eulerian equation becomes

As a result, the Eulerian map distillation loss (28) becomes

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We consider a variant that avoids backpropagating through any spatial or temporal gradient

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This yields the semigradient,

<!-- formula-not-decoded -->

In the last line, we applied (36), which agrees with (35). Hence, the objective (38) is equivalent to the consistency distillation objective in the continuous-time limit after a suitable rescaling of gradients. We note that (39) is identical to the 'Align Your Flow' update considered by Sabour et al. (2025).

## C.2 Consistency training and mean flow.

Consistency training aims to train a model directly, avoiding access to a pre-trained teacher (Song and Dhariwal, 2023; Lu and Song, 2024). The associated loss follows from an identical derivation, except it uses two points on the same interpolant trajectory (rather than ˆ x s +∆ s , which requires access to ˆ b s ),

<!-- formula-not-decoded -->

In (40), x 0 and x 1 are shared between I s and I s +∆ s , which yields the second equality for I s +∆ s . Following the same steps as for consistency distillation, the final result is to replace the semigradient (39) by a Monte-Carlo approximation that leverages ˙ I s in place of the true vector field b s ( x ) = E [ ˙ I s I s = x ] ,

<!-- formula-not-decoded -->

For the parameterization (6), (41) becomes

<!-- formula-not-decoded -->

The semigradients (41) and (42) are higher-variance than (35) as Monte-Carlo approximations, but on average give access to the ideal flow b rather than the pre-trained, approximate flow ˆ b . The gradient (42) is identical to the 'mean flow' update recently considered by Geng et al. (2025).

## D Connection to shortcut models.

Shortcut models (Frans et al., 2024) correspond to a subset of our proposed PSD scheme (15), which itself is based on PFMM. To touch base with the formulation of PFMM in (31), as well as the discussion of PSD in the main text, we place shortcut models in our notation here.

Shortcut models consider a fixed grid of times 0 = t 0 &lt; t 1 &lt; . . . &lt; t N = 1 spaced dyadically, so that t i +2 -t i +1 = 2( t i +1 -t i ) . Observing a similar relation to the tangent identity (7), they train ˆ v t,t like a flow matching model and leverage (31) as a bootstrapping mechanism,

<!-- formula-not-decoded -->

Clearly, the second term in (43) reduces to (31) with s, t, u restricted to a fixed grid. Similarly, (43) corresponds to (15) with discretization in time and the specific proposal distribution p i u = δ t i . The second term in (43) can also be written entirely in terms of ˆ v using the preconditioning discussed later in Section F.1, which leads to the exact form of the objective discussed in Frans et al. (2024).

## E Proofs

In this work, we assume that all studied differential equations satisfy the following assumption.

<!-- formula-not-decoded -->

Assumption E.1. The drift satisfies the one-sided Lipschitz condition

<!-- formula-not-decoded -->

Under Assumption E.1, the classical Cauchy-Lipschitz theory guarantees that solutions exist and are unique for all x 0 ∈ R d and for all t ∈ [0 , 1] .

We first provide a self-contained proof of the following proposition, which first appeared in Boffi et al. (2024). We will then apply this result to prove the primary claims of the main text.

Proposition E.2. Let X s,t denote the flow map (4) for the probability flow equation ˙ x t = b t ( x t ) . Then X s,t satisfies the Lagrangian equation,

<!-- formula-not-decoded -->

the Eulerian equation,

<!-- formula-not-decoded -->

and the semigroup property

<!-- formula-not-decoded -->

Proof. Repeating (4) for ease of reading, the flow map satisfies the jump condition

<!-- formula-not-decoded -->

where x t denotes a trajectory of the probability flow (2). The proof of each condition relies on careful manipulation of this equation.

We first prove the semigroup condition. Observe that

<!-- formula-not-decoded -->

Because x s was arbitrary, the result follows.

We now prove the Lagrangian condition. Taking a derivative of (48), with respect to t and applying the probability flow (2), we find

<!-- formula-not-decoded -->

Because x s was arbitrary, we obtain the Lagrangian condition (45)

Last, we prove the Eulerian condition. Taking a total derivative of (48) with respect to s , we find that

<!-- formula-not-decoded -->

Again, because x s was arbitrary, the result follows.

We now provide a simple proof of the tangent condition we leverage in the main text.

Lemma 2.1 (Tangent condition) . Let X s,t denote the flow map. Then,

<!-- formula-not-decoded -->

i.e. the tangent vectors to the curve ( X s,t ( x )) t ∈ [ s, 1] give the velocity field b t ( x ) for every x .

Proof. By Proposition E.2, we have that the flow map satisfies the Lagrangian equation (45). Taking the limit as s → t , and assuming continuity of the flow map, we find

<!-- formula-not-decoded -->

Above, we used that X t,t ( x ) = x for all x ∈ R d and for all t ∈ [0 , 1] .

We now prove Proposition 2.2, which extends Proposition E.2 to the representation (6).

Proposition 2.2 (Flow map) . Assume that X s,t is given by (6) with v s,t satisfying (7), and assume that v s,t is continuous in both time arguments. Then, X s,t is the flow map defined in (4) if and only if any of the following conditions also holds:

- (i) (Lagrangian condition): X s,t solves the Lagrangian equation

<!-- formula-not-decoded -->

for all ( s, t ) ∈ [0 , 1] 2 and for all x ∈ R d .

- (ii) (Eulerian condition): X s,t solves the Eulerian equation

<!-- formula-not-decoded -->

for all ( s, t ) ∈ [0 , 1] 2 and for all x ∈ R d .

- (iii) (Semigroup condition): For all ( s, t, u ) ∈ [0 , 1] 3 and for all x ∈ R d ,

<!-- formula-not-decoded -->

Proof. We start with the Lagrangian condition (8). By assumption of (7), v t,t ( x ) = b t ( x ) , so that (8) is equivalent to (45). It follows that the flow map must satisfy (8) by Proposition E.2, which proves the forward implication. To prove the reverse implication, observe that by Assumption E.1, solutions to (8) are unique, so that any solution must be the flow map.

The proof of the Eulerian condition is similar. For the forward implication, we observe that (9) is equivalent to (46), so that the flow map solves (9). Now, let X solve (9) (along with (6) and (7)). We would like to prove that X is the flow map. Let us observe that by assumption,

<!-- formula-not-decoded -->

where x s is any solution of the probability flow. Integrating both sides with respect to s from s to t , we find that

<!-- formula-not-decoded -->

This is precisely the definition of the flow map.

Last, we prove the final property. By Proposition E.2, we have that the flow map satisfies (10), which proves the forward implication. To prove the reverse implication, let X be any map satisfying (6), (7) and (10). Define the notation ∂ t X t,t ( y ) = lim s → t ∂ t X s,t ( y ) = v t,t ( y ) = b t ( y ) . Then, consider a Taylor expansion of the infinitesimal semigroup condition for ( x, s, t ) ∈ R d × [0 , 1] 2 arbitrary,

<!-- formula-not-decoded -->

Note that the above Taylor expansion implicitly uses that v s,t is continuous in ( s, t ) to write v t,t + h = v t,t + O ( h ) . This rules out the discontinuous solution

<!-- formula-not-decoded -->

which corresponds to X s,t ( x ) = x for all ( x, s, t ) ∈ R d × [0 , 1] 2 and satisfies the semigroup condition trivially.

Re-arranging the last line of (55), we find that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so that

Equation (58) is precisely the Lagrangian equation, whose unique solution is the ideal flow map. This completes the proof.

̸

Given the above developments, we now recall our main proposition.

Proposition 2.3 (Self-distillation) . The flow map X s,t defined in (4) is given for all 0 ⩽ s ⩽ t ⩽ 1 by X s,t ( x ) = x +( t -s ) v s,t ( x ) where v s,t ( x ) the unique minimizer over ˆ v of where L b (ˆ v ) is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and where L D (ˆ v ) is any of the following three objectives.

(i) The Lagrangian self-distillation (LSD) objective, which leverages (8),

<!-- formula-not-decoded -->

- (ii) The Eulerian self-distillation (ESD) objective, which leverages (9),

<!-- formula-not-decoded -->

(iii) The progressive self-distillation (PSD) objective, which leverages (10),

<!-- formula-not-decoded -->

Above, ˆ X s,t ( x ) = x +( t -s )ˆ v s,t ( x ) and E x 0 ,x 1 denotes an expectation over the random draws of ( x 0 , x 1 ) in the interpolant defined in (1).

Proof. We first prove the statement for the LSD algorithm. Observe that for any ˆ b t and any ˆ X s,t ( x ) = x +( t -s )ˆ v s,t ( x ) , where b t ( x ) = E [ ˙ I t | I t = x ] is the ideal flow. This follows because L b is convex in ˆ b with unique global minimizer given by b , while L LSD is a square residual term on the Lagrangian relation (8). From this, we conclude

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Lemma 2.1 and Proposition 2.2, the ideal flow map X s,t satisfies

<!-- formula-not-decoded -->

From (61), we see that so that X s,t achieves the lower bound (60) and is therefore optimal. Moreover, any global minimizer must satisfy (61), and by Proposition 2.2 therefore must be the flow map.

<!-- formula-not-decoded -->

We now prove the statement for the ESD algorithm, which is similar. We first observe that for any ˆ v ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, by Lemma 2.1 and Proposition 2.2,

<!-- formula-not-decoded -->

From above, we conclude

From (65), it follows that the ideal flow map satisfies

<!-- formula-not-decoded -->

Equation (66) shows that X s,t achieves the lower bound (64) and hence is optimal. Moreover, any global minimizer must satisfy (65) and therefore by Proposition 2.2 is the ideal flow map.

Finally, we prove the result for the PSD approach. The proposition is stated for a uniform proposal distribution over u , but holds for any distribution with full support over [ s, t ] . First, we observe that

<!-- formula-not-decoded -->

By the semigroup property (10), we have that the true flow map satisfies

<!-- formula-not-decoded -->

so that X is optimal. Now, let X ∗ be any map satisfying

<!-- formula-not-decoded -->

i.e., any global minimizer of the PSD objective. It then necessarily follows that

<!-- formula-not-decoded -->

By Proposition 2.2, under the assumption that v ∗ is continuous, (70) implies that X ∗ is the ideal flow map X .

We now recall our theoretical error bounds for the LSD and ESD algorithms.

Proposition 2.4 (Wasserstein bounds) . Let ˆ X s,t ( x ) = x +( t -s )ˆ v s,t ( x ) denote a candidate flow map, let ˆ ρ 1 = ˆ X 0 , 1 ♯ρ 0 denote the corresponding one-step generated distribution, and let ˆ L denote the spatial Lipschitz constant of ˆ v t,t ( · ) uniformly in t . First assume L b (ˆ v ) + L LSD (ˆ v ) ⩽ ε . Then,

<!-- formula-not-decoded -->

Now assume that L b (ˆ v ) + L ESD (ˆ v ) ⩽ ε . Then,

<!-- formula-not-decoded -->

For ease of reading, we split the proof of Proposition 2.4 into two results, one for each algorithm. We begin with LSD.

Proposition E.3 (Lagrangian self-distillation) . Consider the Lagrangian self-distillation method,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let ˆ X denote a candidate flow map satisfying L SD ( ˆ X ) ⩽ ε , and let ˆ ρ 1 denote the corresponding pushforward ˆ ρ 1 = ˆ X 0 , 1 ♯ρ 0 . Let ˆ L denote the spatial Lipschitz constant of ˆ v t,t ( · ) uniformly in time, i.e.

Then,

<!-- formula-not-decoded -->

Proof. Observe that L SD ( ˆ X ) ⩽ ε implies that both L b (ˆ v ) ⩽ ε and L LSD (ˆ v ) ⩽ ε . We first note that

<!-- formula-not-decoded -->

In (74), we used that b t ( x ) = E [ ˙ I t | I t = x ] along with the tower property of the conditional expectation. It then follows that the L 2 error from the target flow b is bounded by

<!-- formula-not-decoded -->

We now observe that, again by the tower property of the conditional expectation,

<!-- formula-not-decoded -->

Equation (76) shows that the term subtracted in (75) is a conditional variance, and therefore is nonnegative. Combining the two, we find that

<!-- formula-not-decoded -->

We now consider the learned probability flow

<!-- formula-not-decoded -->

By Proposition 3 of Albergo and Vanden-Eijnden (2022), (77) implies that

<!-- formula-not-decoded -->

where ˆ ρ ˆ v 1 = Law (ˆ x ˆ v t ) . Now, by Proposition 3.7 of Boffi et al. (2024), L LSD (ˆ v ) ⩽ ε implies

<!-- formula-not-decoded -->

By the triangle inequality and Young's inequality, we then have

<!-- formula-not-decoded -->

This completes the proof.

We now prove a similar guarantee for the ESD method.

Proposition E.4 (Eulerian self-distillation) . Consider the Eulerian self-distillation method,

<!-- formula-not-decoded -->

Let ˆ X denote a candidate flow map with the same properties as in Proposition E.3. Then,

<!-- formula-not-decoded -->

Proof. As in Proposition E.3, our assumption L SD (ˆ v ) ⩽ ε implies that both L b (ˆ v ) ⩽ ε and L ESD (ˆ v ) ⩽ ε . Defining the flow ˙ ˆ x ˆ v t as in (78), we have a bound identical to (79) on W 2 2 ( ρ 1 , ˆ ρ ˆ v 1 ) . Now, leveraging Proposition 3.8 in Boffi et al. (2024), we have that

<!-- formula-not-decoded -->

Again applying the triangle inequality and Young's inequality yields the relation

<!-- formula-not-decoded -->

This completes the proof.

## F Further details on self-distillation

In this section, we collect some additional results and detail on some of the topics discussed in the main text.

## F.1 Semigroup parameterization for PSD.

By definition, we have that

We then also have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By the semigroup property (10), it follows that

<!-- formula-not-decoded -->

from which we see that

<!-- formula-not-decoded -->

Re-arranging and eliminating, we find that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which provides a direct signal for v s,t . Choosing u = γs +(1 -γ ) t for γ ∈ [0 , 1] leads to the simple relations which can be used to precondition the relation (90) as

<!-- formula-not-decoded -->

In the numerical experiments, we use (92) to define a training signal for ˆ v for the PSD algorithm.

## F.2 Limiting relations and annealing schemes.

Limiting relations. As shown in the proof of Proposition 2.2, application of the semigroup property with ( s, u, t ) = ( s, t, t + h ) for a fixed ( s, t ) recovers the Lagrangian equation at order h . As shown in the proof of the tangent condition Lemma 2.1, the Lagrangian condition recovers the velocity field in the limit as s → t . Similarly, if we consider the Eulerian equation in the limit as s → t ,

<!-- formula-not-decoded -->

so that ∂ s X s,s ( x ) = -v s,s ( x ) = -b s ( x ) . In this way, all three characterizations reduce to the flow matching objective for v t,t as the diagonal is approached.

Annealing and pre-training. As a result of (93), we can view training the flow ˆ v t,t only on the diagonal s = t as a pre-training scheme for the map ˆ X . This also means that we can initialize ˆ v t,t from a pre-trained model in a principled way via appropriate duplication of the time embeddings.

The relations (7) and (93) imply that the off-diagonal self-distillation terms represent a natural extension of the diagonal flow matching term. This suggests a simple two-phase curriculum in which the flow matching term is trained alone for N fm steps as a pre-training phase, followed by a smooth conversion from diagonal training into self-distillation by expanding the sampled range of | t -s | from 0 to 1 over the course of N anneal steps. This can be accomplished, for example, by drawing ( s, t ) uniformly on the off-diagonal and then clamping t = min( t, s + δ ( k )) where k denotes the iteration and δ ( k ) is the maximum value of | t -s | , for example δ ( k ) = k/N anneal. For simplicity, we trained directly without any annealing in our experiments, but expect this to simplify and speed up training for large datasets where overfitting is not a concern.

## F.3 Further details on loss sampling and computation

In this section, we provide further detail on how the choice of η ∈ [0 , 1] , which distributes the batch between the diagonal flow matching term and the off-diagonal self-distillation term, affects training time. The factor η can be chosen based on the available computational budget to systematically trade off the relative amount of direct training and distillation per gradient step. We focus here on the computational cost of a forward pass of the objective function; the complexity of a backward pass will depend on the specific choice of sg ( · ) operator used.

Flow matching. Evaluating the interpolant loss L b on a single sample requires a single neural network evaluation ˆ v t,t ( I t ) , leading to B network evaluations on a batch.

LSD. The LSD objective requires a single partial derivative evaluation ∂ t ˆ X s,t ( I s ) and two network evaluations - one for ˆ v t,t and one for ˆ X s,t ( I s ) - per sample. The time derivative is a constant factor C ≈ 1 . 5 more than a forward pass, and with standard computational tools such as jvp , can be computed at the same time as ˆ X s,t ( I s ) . The LSD objective thus requires (1 + C ) B network evaluations. Adding the diagonal and off-diagonal parts, we find a complexity of ((1 -η )(1+ C )+ η ) B for the full self-distillation objective.

PSD. The PSD objective requires three neural network evaluations, so that its expense is 3 B . Combining this with the diagonal component, we have (3(1 -η ) + η ) B network evaluations.

ESD. The ESD objective requires a partial derivative evaluation ∂ s X s,t ( I s ) , a neural network evaluation v s,s ( I s ) , and a Jacobian-vector product ∇ X s,t ( I s ) v s,s ( I s ) . Observing that ( ∂ s , ∇ ) can be used as one augmented ( d + 1) -dimensional gradient, and then observing that ∂ ˆ X ( I

s s,t s ) + ∇ ˆ X s,t ( I s )ˆ v s,s ( I s ) = ∇ s,x ˆ X s,t ( I s ) ( 1 ˆ v s,s ( I s ) ) , this can be computed as a single Jacobian-vector product. This gives a complexity of (1 + C ) B , identical to LSD. Adding the diagonal component, we find ((1 + C )(1 -η ) + η ) B .

## F.4 Stopgradient recommendations

The choice of sg ( · ) operator in the loss is delicate and empirical, as it is very difficult to ascertain the convergence properties of an algorithm operating on an objective leveraging sg ( · ) a-priori . Nevertheless, in practice, we find it critical for high-dimensional tasks such as images to use sg ( · ) to control the flow of information from the teacher network on the diagonal s = t to the off-diagonal. For large-scale neural networks, we find empirically that backpropagating through Jacobian-vector products - in particular spatial Jacobian-vector products - leads to significant instability, which can be avoided with sg ( · ) . For low-dimensional tasks with simple neural networks, we found instability to be less of a concern.

Following these observations, we found it useful to take insight from the distillation setting described in Section B, leading to the configurations

<!-- formula-not-decoded -->

It is also possible to avoid backpropagating through the partial derivative with respect to s and with respect to t in ESD and LSD by expanding the definition of ˆ X s,t ( x ) = x +( t -s )ˆ v s,t ( x ) as described in Section C. This reduces the memory overhead even further by avoiding backpropagating through a backward pass of the network.

EMAteacher. In addition to the use of sg ( · ) , an important consideration is the choice of parameters for the teacher, which provides an alternative perspective on and method to implement the sg ( · ) . Making explicit the student parameters θ and teacher parameters ϕ , we can write for L LSD (with analogous expressions for the other choices),

<!-- formula-not-decoded -->

The recommendation in (94) corresponds to taking the gradient of (95) with respect to θ and then evaluating the result at ϕ = θ . A second option would be to evaluate ϕ at an exponential moving average of θ ,

<!-- formula-not-decoded -->

where k denotes the optimization step and where δ denotes a forgetting factor such as δ = 0 . 9999 . While in practice we found improved samples by evaluating the learned flow map over EMA parameters (see Section G for exact values), we found that the use of EMA for the teacher parameters offered no gain and sometimes led to instability in early experiments. For this reason, we use the instantaneous parameters ϕ = θ , corresponding to (94) or δ = 0 in (96).

## F.5 Classifier-free guidance

In this section, we describe how to train a flow map with classifier-free guidance. For the derivation, we focus on the LSD algorithm to avoid replicating the loss functions in each case, but the other choices are identical. To this end, let b t ( x ; c ) denote a conditional velocity field. We first observe that we may train a conditional flow map via the objective function

<!-- formula-not-decoded -->

In (97), I c t denotes the conditional interpolant (i.e., with I c t = α ( t ) x 0 + β ( t ) x c 1 with x c 1 ∼ ρ 1 ( · | c ) drawn conditionally on c ), and E now includes an expectation over the value of c . To train a model that is both conditional and unconditional, we may include c = ∅ in the expectation.

As in the main text, let us now define the CFG velocity field at guidance strength α ∈ R as

<!-- formula-not-decoded -->

Given the second line of (98), we define the current estimate of the guided velocity,

<!-- formula-not-decoded -->

We now observe that because (99) is constructed entirely in terms of the known ˆ v t,t , we only need to modify the self-distillation term rather than the flow matching term to train a CFG flow map. To this end, we self-distill the guided velocity ˆ q t over a range of α . This leads to the objective function

<!-- formula-not-decoded -->

where ¯ α denotes a maximum guidance scale of interest. Following the same derivation, we may obtain the CFG ESD objective

<!-- formula-not-decoded -->

as well as the CFG PSD objective,

<!-- formula-not-decoded -->

## F.6 Detailed algorithms for each self-distillation method

Here, we provide detailed algorithmic implementations for each self-distillation method using the recommendations provided in (94). Each algorithm computes the flow matching loss L b (ˆ v ) over a batch of size ηM and the distillation loss L D (ˆ v ) over a batch of size (1 -η ) M , comprising a total batch size of M .

## Algorithm 2: Lagrangian Self-Distillation (LSD)

input: Distribution ρ ( x 0 , x 1 ) ; model ˆ v s,t ; coefficients α t , β t ; batch size M ; diagonal fraction η ; weight w s,t .

## repeat

Sample M d = ⌊ ηM ⌋ pairs ( x i 0 , x i 1 ) ∼ ρ ( x 0 , x 1 ) ; Sample M d times t i ∼ U ([0 , 1]) ; Compute interpolants I t i = α t i x i 0 + β t i x i 1 and velocities, ˙ I t i = ˙ α t i x i 0 + ˙ β t i x i 1 ; Compute diagonal loss L b = 1 M d ∑ M d i =1 e -w t i ,t i | ˆ v t i ,t i ( I t i ) -˙ I t i | 2 + w t i ,t i ; Sample M o = M -M d pairs ( x j 0 , x j 1 ) ∼ ρ ( x 0 , x 1 ) ; Sample M o pairs ( s j , t j ) ∼ U od ; Compute interpolants: I s j = α s j x j 0 + β s j x j 1 ; Compute simultaneously via jvp : ˆ X s j ,t j ( I s j ) , ∂ t ˆ X s j ,t j ( I s j ) ; Evaluate teacher at transported point: ˆ b t j = ˆ v t j ,t j ( ˆ X s j ,t j ( I s j )) ; Compute residual: r j = ∂ t ˆ X s j ,t j ( I s j ) -sg ( ˆ b t j ) ; Compute LSD loss L LSD = 1 M o ∑ M o j =1 ( e -w s j ,t j | r j | 2 + w s j ,t j ) ; Compute self-distillation loss L SD = L b + L LSD ; Update ˆ v and w using ∇L SD ; until converged ; output: Trained flow map ˆ X ( x ) = x +( t s )ˆ v ( x )

s,t -s,t

## Algorithm 3: Eulerian Self-Distillation (ESD)

input: Distribution ρ ( x 0 , x 1 ) ; model ˆ v s,t ; coefficients α t , β t ; batch size M ; diagonal fraction η ; weight w s,t .

## repeat

```
Sample M d = ⌊ ηM ⌋ pairs ( x i 0 , x i 1 ) ∼ ρ ( x 0 , x 1 ) ; Sample M d times t i ∼ U ([0 , 1]) ; Compute interpolants I t i = α t i x i 0 + β t i x i 1 and velocities ˙ I t i = ˙ α t i x i 0 + ˙ β t i x i 1 ; Compute diagonal loss: L b = 1 M d ∑ M d i =1 ( e -w t i ,t i | ˆ v t i ,t i ( I t i ) -˙ I t i | 2 + w t i ,t i ) ; Sample M o = M -M d pairs ( x j 0 , x j 1 ) ∼ ρ ( x 0 , x 1 ) ; Sample M o pairs ( s j , t j ) ∼ U od ; Compute interpolants: I s j = α s j x j 0 + β s j x j 1 ; Evaluate teacher velocities: ˆ b s j = ˆ v s j ,s j ( I s j ) ; Compute simultaneously via single augmented jvp : ∂ s ˆ X s j ,t j ( I s j ) , ∇ ˆ X s j ,t j ( I s j ) ; Compute Eulerian residual: r j = ∂ s ˆ X s j ,t j ( I s j ) + sg ( ∇ ˆ X s j ,t j ( I s j ) ˆ b s j ) ; Compute ESD loss: L ESD = 1 M o ∑ M o j =1 ( e -w s j ,t j | r j | 2 + w s j ,t j ) ; Compute self-distillation loss L SD = L b + L ESD ; Update ˆ v and w using ∇L SD ; converged ;
```

until output: Trained flow map ˆ X s,t ( x ) = x +( t -s )ˆ v s,t ( x )

## Algorithm 4: Progressive Self-Distillation (PSD)

```
input: Distribution ρ ( x 0 , x 1 ) ; model ˆ v s,t ; coefficients α t , β t ; batch size M ; diagonal fraction η ; weight w s,t ; sampling method p γ . repeat Sample M d = ⌊ ηM ⌋ pairs ( x i 0 , x i 1 ) ∼ ρ ( x 0 , x 1 ) ; Sample M d times t i ∼ U ([0 , 1]) ; Compute interpolants I t i = α t i x i 0 + β t i x i 1 and velocities ˙ I t i = ˙ α t i x i 0 + ˙ β t i x i 1 ; Compute diagonal loss: L b = 1 M d ∑ M d i =1 ( e -w t i ,t i | ˆ v t i ,t i ( I t i ) -˙ I t i | 2 + w t i ,t i ) ; Sample M o = M -M d pairs ( x j 0 , x j 1 ) ∼ ρ ( x 0 , x 1 ) ; Sample M o pairs ( s j , t j ) ∼ U od ; Sample intermediate fractions: γ j ∼ p γ (e.g., U ([0 , 1]) or δ 1 / 2 ); Compute intermediate times: u j = γ j s j +(1 -γ j ) t j ; Compute interpolants: I s j = α s j x j 0 + β s j x j 1 ; Evaluate model at student points: ˆ v s j ,t j ( I s j ) ; Evaluate model at first segment: ˆ v s j ,u j ( I s j ) ; Compute intermediate flow maps: ˆ X s j ,u j ( I s j ) = I s j +( u j -s j )ˆ v s j ,u j ( I s j ) ; Evaluate model at second segment: ˆ v u j ,t j ( ˆ X s j ,u j ( I s j )) ; Compute preconditioned teacher signals: ˆ v teacher s j ,t j = (1 -γ j )ˆ v s j ,u j ( I s j ) + γ j ˆ v u j ,t j ( ˆ X s j ,u j ( I s j )) ; Compute residuals: r j = ˆ v s j ,t j ( I s j ) -sg ( ˆ v teacher s j ,t j ) ; Compute PSD loss: L PSD = 1 M o ∑ M o j =1 ( e -w s j ,t j | r j | 2 + w s j ,t j ) ; Compute self-distillation loss: L SD = L b + L PSD ; Update ˆ v and w using ∇L SD ; until converged ; output: Trained flow map ˆ X s,t ( x ) = x +( t -s )ˆ v s,t ( x )
```

## G Further details on numerical experiments

Here, we provide a complete description of the numerical experiments performed in the main text. A concise summary of each experiment is given in Table 2.

## G.1 Checkerboard Details

Experimental setup. We compare the LSD, ESD, and PSD algorithms on the two-dimensional checkerboard dataset. For PSD, we evaluate both uniform sampling ( γ ∼ U ([0 , 1]) , denoted PSD-U) and midpoint sampling ( γ = 1 / 2 , denoted PSD-M). We generate a dataset with 10 7 samples and train for 150 , 000 steps with a batch size of 100 , 000 and a learning rate of 10 -3 with square root decay after 35 , 000 steps. We use a diagonal fraction of η = 0 . 75 , allocating 75% of each batch to the flow matching loss L b and 25% to the self-distillation loss. The network architecture consists of a 4-layer MLP with 512 neurons per hidden layer and GELU activation functions. We use the linear interpolant with α t = 1 -t and β t = t with a Gaussian base distribution x 0 ∼ N (0 , I ) with adaptive scaling to normalize by the variance of the target distribution. Times are sampled uniformly over the upper triangle without annealing, and we apply gradient clipping at 10 . 0 . All methods use the stopgradient configurations described in (94). We visualize model samples produced from an exponential moving average of the learned parameters with decay factor 0 . 999 . Each experiment was run on a single 40GB A100 GPU. A full qualitative visualization of the tabular results discussed in the main text is shown in Figure 6.

KL Computation. To compute the KL divergence, we leverage that (a) the checkerboard density is known analytically as a uniform density over the selected squares, (b) the low-dimensionality of the dataset means that histogramming the model samples gives a good approximation of the model density, and (c) the low-dimensionality implies that quadrature can be used to compute a high-accuracy, deterministic approximation of KL. To this end, we first compute 64 , 000 samples

Table 2: Experimental setup. Summary of experimental configurations across all datasets. All experiments use uniform sampling of ( s, t ) pairs over the upper triangle and leverage the sg ( · ) choices described in (94).

|                       | Checker                | CIFAR-10          | CelebA-64         | AFHQ-64           |
|-----------------------|------------------------|-------------------|-------------------|-------------------|
| Dataset Properties    |                        |                   |                   |                   |
| Dimensionality        | 2                      | 3 × 32 × 32       | 3 × 64 × 64       | 3 × 64 × 64       |
| Samples               | 10 7                   | 50 k              | 203 k             | 16 k              |
| Network               |                        |                   |                   |                   |
| Architecture          | 4-layer MLP            | EDM2              | EDM2              | EDM2              |
| Hidden/base channels  | 512                    | 128               | 128               | 128               |
| Channel multipliers   | -                      | [2, 2, 2]         | [1, 2, 3, 4]      | [1, 2, 3, 4]      |
| Residual blocks       | -                      | 4 per resolution  | 3 per resolution  | 3 per resolution  |
| Attention resolutions | -                      | 16 × 16           | 16 × 16 , 8 × 8   | 16 × 16 , 8 × 8   |
| Dropout               | -                      | 0.13              | 0.0               | 0.0               |
| Hyperparameters       |                        |                   |                   |                   |
| Batch size            | 100,000                | 512               | 256               | 256               |
| Training steps        | 150,000                | 400,000           | 800,000           | 800,000           |
| Total samples         | 25 × 10 9              | 204 . 8 × 10 6    | 204 . 8 × 10 6    | 204 . 8 × 10 6    |
| Optimizer             | RAdam                  | RAdam             | RAdam             | RAdam             |
| Learning rate         | 10 - 3                 | 10 - 2            | 10 - 2            | 10 - 2            |
| LR schedule           | Sqrt decay at 35k      | Sqrt decay at 35k | Sqrt decay at 35k | Sqrt decay at 35k |
| Gradient clipping     | 10.0                   | 1.0               | 1.0               | 1.0               |
| Diagonal fraction η   | 0.75                   | 0.75              | 0.75              | 0.75              |
| EMA decay             | 0.999                  | 0.9999            | 0.9999            | 0.9999            |
| Evaluation            |                        |                   |                   |                   |
| Metric                | KL divergence          | FID               | FID               | FID               |
| Sample count          | 64,000                 | 50,000            | 50,000            | 10,000            |
| Methods               |                        |                   |                   |                   |
| Algorithms            | LSD, ESD, PSD-U, PSD-M | LSD, PSD-U, PSD-M | LSD, PSD-U, PSD-M | LSD, PSD-U, PSD-M |

from each model for each number of steps N . We then histogram these samples using an M × M grid with M = 50 over the range [ -1 , 1] 2 . To approximate the KL, we use the quadrature formula

<!-- formula-not-decoded -->

where in (103) x ij denotes the center of bin ( i, j ) used to compute the histogram. We note that because of the uniformity of ρ 1 and ˆ ρ hist 1 , this quadrature rule is exact, i.e.

<!-- formula-not-decoded -->

## G.2 CIFAR-10 Details.

We evaluate the LSD, ESD, and PSD algorithms on the CIFAR-10 dataset. Again for PSD, we compare uniform sampling ( γ ∼ U ([0 , 1]) , denoted PSD-U) and midpoint sampling ( γ = 1 / 2 , denoted PSD-M). All methods use uniform sampling over the upper triangle ( s, t ) ∼ U od without annealing. We train for 400 , 000 steps with a batch size of 512 and an initial learning rate of 10 -2 with square root decay after 35 , 000 steps. We use a diagonal fraction of η = 0 . 75 , allocating 75% of each batch to the flow matching loss and 25% to the self-distillation loss. The network architecture is based on EDM2 in Configuration G (Karras et al., 2024) and NCSN++ (Song et al., 2020), using 128 base channels, channel multipliers [2 , 2 , 2] , and 4 residual blocks per resolution. We use positional embeddings for time, as we found that Fourier embeddings led to greater training instability (Lu and Song, 2025). We embed s and ( t -s ) rather than s and t , which we found to perform better in early experiments, add these embeddings together, and otherwise use standard FiLM conditioning in the EDM2 network. We apply attention at the 16 × 16 resolution and use dropout of 0 . 13 following EDM

Figure 6: Checker: full qualitative results Full visualization of sample quality as a function of number of steps on the two-dimensional checker dataset.

<!-- image -->

recommendations for CIFAR-10 (Karras et al., 2022). We employ a learned weight function w s,t with 128 channels to normalize gradient variance. The interpolant uses α t = 1 -t and β t = t with a Gaussian base distribution, setting the variance of the Gaussian adaptively to match the variance of the training data. We apply gradient clipping at 1 . 0 and use the stopgradient configurations described in (94) for LSD and PSD; ESD was unstable in every sg ( · ) configuration tried. We evaluate sample quality using FID computed on-the-fly every 10 , 000 steps with 10 , 000 generated samples, using NFE ∈ { 1 , 2 , 4 , 8 , 16 } for the flow map. Models were trained from random initialization without pre-training, and we track EMA parameters with decay factors 0 . 999 and 0 . 9999 . We re-compute FID over 50 , 000 generated samples for post-processing and take the best checkpoints for each number of sampling steps over the entire training range with an EMA factor 0 . 9999 . We use the RAdam optimizer with default settings. Minimal hyperparameter tuning was applied to the algorithms due to well-established training practices for CIFAR-10 available in the literature.

## G.3 CelebA-64 Details

We compare LSD and both PSD variants (uniform and midpoint) on the CelebA-64 dataset. As for CIFAR-10, we found ESD to be uniformly unstable and so do not report results. We train for 800 , 000 steps (corresponding to 204 . 8 Msamples) with a batch size of 256 and an initial learning rate of 10 -2 with square root decay after 35 , 000 steps. We use a diagonal fraction of η = 0 . 75 , allocating 75% of each batch to the flow matching loss and 25% to the self-distillation loss. The network architecture is based on EDM2 in Configuration G with 128 base channels, channel multipliers [1 , 2 , 3 , 4] , and 3 residual blocks per resolution, corresponding to the 'ImageNet-S' variant reduced from 192 channels to 128 . We apply attention at resolutions 16 × 16 and 8 × 8 , and do not use dropout. As with CIFAR-10, we use positional embeddings for time and embed s and ( t -s ) with standard FiLM conditioning. We use the linear interpolant with α t = 1 -t and β t = t with a Gaussian base distribution and adaptive scaling to normalize to the variance of the target density. Times points ( s, t )

are sampled uniformly over the upper triangle and no annealing or pretraining is used. We apply gradient clipping at 1 . 0 and leverage the stopgradient configuration (94) for all methods. We use the RAdam optimizer with default settings. We evaluate online sample quality using FID-10K computed every 10 , 000 steps, and then compute FID-50k post-hoc to find the best model, following the same steps as for CIFAR-10. Models were trained from random initialization without pre-training, and we track EMA parameters with decay factor 0 . 9999 . FID-50K scores were computed with this EMA factor.

## G.4 AFHQ-64 Details

We compare LSD and both PSD variants (uniform and midpoint) on the AFHQ-64 dataset. As with CelebA-64, we found ESD to be unstable and do not report results. We train for 800 , 000 steps (corresponding to 204 . 8 Msamples) with a batch size of 256 and an initial learning rate of 10 -2 with square root decay after 35 , 000 steps. We use a diagonal fraction of η = 0 . 75 , allocating 75% of each batch to the flow matching loss and 25% to the self-distillation loss. The network architecture is based on EDM2 in Configuration G with 128 base channels, channel multipliers [1 , 2 , 3 , 4] , and 3 residual blocks per resolution, matching the architecture used for CelebA-64. We apply attention at resolutions 16 × 16 and 8 × 8 , and do not use dropout. As with CIFAR-10, we use positional embeddings for time and embed s and ( t -s ) with standard FiLM conditioning. We use the linear interpolant with α t = 1 -t and β t = t with a Gaussian base distribution and adaptive scaling to normalize to the variance of the target density. Time points ( s, t ) are sampled uniformly over the upper triangle and no annealing or pretraining is used. We apply gradient clipping at 1 . 0 and leverage the stopgradient configuration (94) for all methods. We use the RAdam optimizer with default settings. We evaluate online sample quality using FID-10K computed every 10 , 000 steps, and then compute FID-50k post-hoc to find the best model, following the same steps as for CIFAR-10 and CelebA-64. Models were trained from random initialization without pre-training, and we track EMA parameters with decay factor 0 . 9999 . FID-50K scores were computed with this EMA factor.