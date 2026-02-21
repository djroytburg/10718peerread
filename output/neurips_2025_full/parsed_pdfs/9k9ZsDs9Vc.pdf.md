## Multitask Learning with Stochastic Interpolants

## Hugo Negrel

Capital Fund Management 23 Rue de l'Université, 75007 Paris hugonegrel13@gmail.fr

## Michael S Albergo

## Florentin Coeurdoux

Capital Fund Management 23 Rue de l'Université, 75007 Paris florentin.coeurdoux@cfm.com

## Eric Vanden-Eijnden

Society of Fellows, Harvard University malbergo@fas.harvard.edu

## Abstract

We propose a framework for learning maps between probability distributions that broadly generalizes the time dynamics of flow and diffusion models. To enable this, we generalize stochastic interpolants by replacing the scalar time variable with vectors, matrices, or linear operators, allowing us to bridge probability distributions across multiple dimensional spaces. This approach enables the construction of versatile generative models capable of fulfilling multiple tasks without task-specific training. Our operator-based interpolants not only provide a unifying theoretical perspective for existing generative models but also extend their capabilities. Through numerical experiments, we demonstrate the zero-shot efficacy of our method on conditional generation and inpainting, fine-tuning and posterior sampling, and multiscale modeling, suggesting its potential as a generic task-agnostic alternative to specialized models.

## 1 Introduction

Recent years have witnessed remarkable advances in generative modeling, with transport-based approaches such as normalizing flows (Lipman et al., 2022; Albergo and Vanden-Eijnden, 2022; Liu et al., 2022) and diffusion models (Ho et al., 2020; Song and Ermon, 2020; De Bortoli et al., 2021; Albergo et al., 2023a) emerging as state-of-the-art techniques across various application domains (Rombach et al., 2022; Mazé and Ahmed, 2023; Alverson et al., 2024). These methods have revolutionized our ability to generate high-quality images, text, and other complex data types by viewing these data as samples from an unknown target distribution and learning to transform simple (e.g., noise) distributions into this target. This transformation is effectively achieved via transport of the samples by a flow or diffusion process with a drift (or score) parameterized by neural networks and estimated via simulation-free quadratic regression, enabling highly efficient training.

Despite their impressive performance, these generative frameworks face a fundamental limitation: they are typically designed and trained for specific, predetermined tasks, with the generative objective specified before training. For example, a diffusion model trained to generate images cannot easily be repurposed to perform inpainting or other editing tasks without substantial modification or retraining.

∗ also at: Courant Institute of Mathematical Sciences, New York University, New York, NY 10012, USA, eve2@cims.nyu.edu

Machine Learning Lab Capital Fund Management 23 Rue de l'Université, 75007 Paris eric.vanden-eijnden@cfm.com ∗

While some flexibility can be achieved through conditioning variables or prompting, these approaches remain constrained within narrowly defined operational boundaries established beforehand. Recent attempts at multitask generation using approximated guidance strategies (Chung et al., 2023; Song et al., 2022; Wang et al., 2024) have shown promise, but rely on uncontrolled approximations that limit their theoretical guarantees and can lead to unpredictable results. These methods typically operate within a predefined space of capabilities and lack the flexibility to adapt to novel tasks without retraining, often requiring domain-specific architectural modifications or specialized training procedures that further limit their versatility.

In this paper, we introduce a novel framework for training truly multi-task generative models based on a generalized formulation of stochastic interpolants. Our key insight is to replace the scalar time variable traditionally used in transport-based models with a linear operator. These operator-based interpolants enable interpolation between random variables across multiple dimensional planes or setups, providing a unified mathematical formulation that treats various generative tasks as different ways of traversing the same underlying space, rather than as separate problems requiring distinct models. This dramatically expands the space of possible tasks that a single model can perform.

Our main contributions include theoretical advances that establish a framework for multiple generative applications:

- We extend traditional scalar interpolation in dynamical generative models to higher-dimensional structures, developing a unified mathematical formulation of operator-based interpolants that treats various generative tasks as different ways of traversing the same underlying space. This opens up fundamentally new ways of seeing how generative models can be structured to handle multiple objectives simultaneously.
- We show how this framework enables generative models with continual self-supervision over a wide purview of generative tasks, making possible: universal inpainting models that work with arbitrary masks, multichannel data denoisers with operators in the Fourier domain, posterior sampling with quadratic rewards, and test-time dynamical optimization with rewards and interactive user feedback, all with one pretrained model.
- We demonstrate these various tools on image-infilling, data de-corruption, statistical physics simulation, and dynamical robotic planning tasks across a number of datasets, showing that our method matches or surpasses existing approaches without being specifically tied to any single generative objective.

All common augmentations like text-conditioning and guidance can still be used just as before in our setup. While our approach increases the complexity of the initial learning problem, we provide arguments that this additional complexity can be addressed through scale. In essence, the 'pretraining' phase becomes more challenging, but the resulting model gains substantial flexibility and versatility that compensates for the pretraining costs. That is, our approach can be seen as a way of amortizing learning over a variety of tasks. This points toward a more general paradigm of universal generative models that can be trained once and then applied to a variety of objectives, potentially reducing computational and environmental costs associated with training separate models for each task.

## 1.1 Related works

Flow matching and diffusion models. Our approach extends the theoretical groundwork established in flow matching with stochastic interpolant and rectified flows (Lipman et al., 2022; Albergo and Vanden-Eijnden, 2022; Albergo et al., 2023a; Liu et al., 2022) as well as the probability flow formulations in diffusion models (Ho et al., 2020; Song et al., 2020). which established techniques for handling multiple target distributions simultaneously. Unlike data-dependent coupling approaches (Albergo et al., 2023b) that require constructing specific couplings for tasks like inpainting, our method learns a general operator space that naturally accommodates such tasks without additional coupling design. By introducing operator-valued interpolants, we enable a richer space of transformations between distributions, unlocking a flexible framework for multiple generative tasks.

Inverse problems and inpainting Our framework offers a unified approach to inverse problems, contrasting with traditional methods that require problem-specific variational optimization procedures Pereyra et al. (2015). Recent diffusion-based approaches Chung et al. (2023); Song et al. (2022); Kawar et al. (2022); Martin et al. (2025) typically need guided sampling trajectories tailored to

each task. Similarly, methods using MCMC/SMC sampling Coeurdoux et al. (2024); Sun et al. (2024), variational approximations Mardani et al. (2023); Alkan et al. (2023), or optimization-driven techniques Wang et al. (2024); Daras et al. (2022) remain fundamentally task-specific. Our approach encodes solution paths within the interpolant operator structure itself, enabling multiple inverse problems to be addressed through appropriate operator path selection during inference-without additional training. Our work can also be seen as a way to formalize methods that seek to clean data corrupted in various ways Bansal et al. (2022)

Multiscale and any-order generation Recent approaches to generative modeling have explored hierarchical strategies through progressive refinement. Visual Autoregressive Modeling (Tian et al., 2024) employs next-scale prediction, while Fractal Generative Models (Li et al., 2025) utilize selfsimilar structures for multiscale representations. These methods typically constrain generation to fixed paths established during training. In contrast, our framework decouples the training process from generation trajectories, allowing flexible path selection at inference time. This is conceptually related to optimizing generation order in discrete diffusion (Shi et al., 2025) and token ordering studies (Kim et al., 2025), but provides greater flexibility by enabling post-training optimization and dynamic, self-guided generation strategies that adapt based on intermediate results.

## 2 Theoretical framework

Imagine we want to create a single generative model capable of multiple tasks - sampling new data, inpainting, denoising, and more. To achieve this, we need to expand beyond the traditional 'single path" between noise and data. In this section, we develop the theoretical foundations of operator-based interpolants, which allow flexible navigation through a richer multidimensional space and will enable multitask generative capabilities discussed in Section 3.

## 2.1 Operator-based interpolants

Suppose that we are given a couple of random variables ( x 0 , x 1 ) both taking values in a Hilbert space H (for example R d ) and drawn from a joint distribution µ ( dx 0 , dx 1 ) . Our aim is to design a transport between a broad class of distributions supported on H involving mixtures of x 0 and x 1 . We will do so by generalizing the framework of stochastic interpolant.

Definition 2.1 (Operator-based interpolants) . Let B ( H ) be a connected set of bounded linear operators on H and let S ⊆ B ( H ) × B ( H ) , also connected. Given any pair of linear operators ( α, β ) ∈ S , the operator-based interpolant I ( α, β ) is the stochastic process given by

<!-- formula-not-decoded -->

where ( x 0 , x 1 ) ∼ µ . We will denote by µ α,β the probability distribution of I ( α, β ) .

If we picked e.g. α = (1 -t ) Id and β = t Id with t ∈ [0 , 1] , respectively, we would go back to a standard stochastic interpolant, but we stress that the operator-based interpolants from Definition 2.1 are much more general objects. For example, if H = R d , we could take B ( H ) to be a set of d × d matrices with real entries - choices of α , β tailored to several multitask generation will be discussed in Section 3. The main objective of this paper is to show how to exploit this flexibility of design. Specifically we will show that we can learn a model that can be used to transport samples of I ( α, β ) along any paths of ( α, β ) in S without having to choose any such path during training . This will mean that transport problems in a broad class associated to a variety of tasks, including inpainting and block generation of any order, fine-tuning, etc. will be pretrained into this model.

## 2.2 Multipurpose drifts and score

To proceed, we introduce two drifts, which are functions of S ×H taking value in H

Definition 2.2 (Multipurpose drift) . The drifts η 0 , η 1 : S ×H→H

: are given by

<!-- formula-not-decoded -->

where E [ · | I ( α, β ) = x ] denotes expectation over the coupling ( x 0 , x 1 ) ∼ µ conditioned on the event I ( α, β ) = x .

Using the L 2 characterization of the conditional expectation, these drifts can be estimated via solution of a tractable optimization problem with an objective function involving an expectation:

Lemma 2.3 (Drift objective) . Let ν ( dα, dβ ) be a probability distribution whose support is S . Then the drifts η 0 , 1 ( α, β, x ) in Definition 2.2 can be characterized globally for all ( α, β ) ∈ S and all x ∈ supp( µ α,β ) via solution of the optimization problems

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ∥ · ∥ denotes the norm in H .

This lemma is proven in Appendix A. Below we will use (3) and (4) to learn η 0 , 1 over a rich parametric class made of deep neural networks. Note that the drifts η 0 and η 1 are not linearly independent since the definition of I ( α, β ) in (9) together with the equality x = E [ I ( α, β ) | I ( α, β ) = x ] imply that

<!-- formula-not-decoded -->

Therefore we can obtain η 0 from η 1 if α is invertible and η 1 from η 0 if β is invertible. Note also that, when x 0 is Gaussian and x 0 ⊥ x 1 , η 0 is related to the score of the distribution of the stochastic interpolant:

Lemma 2.4 (Score) . Assume that H = R d and that the probability distribution µ α,β of the stochastic interpolant I ( α, β ) is absolutely continuous with respect to the Lebesgue measure with density ρ α,β ( x ) . Assume also that x 0 ∼ N (0 , Id ) and x 0 ⊥ x 1 . Then the score s α,β ( x ) = ∇ log ρ α,β ( x ) is related to the drift η 0 ( α, β, x ) via

<!-- formula-not-decoded -->

This lemma is proven in Appendix A.

## 2.3 Transport with flows and diffusions

We can now state the main theoretical results of this paper: if we are able to sample the stochastic interpolant I ( α 0 , β 0 ) at a specific value ( α 0 , β 0 ) ∈ S , then we can produce sample of I ( α t , β t ) along any curve ( α t , β t ) ∈ S by solving either a probability flow ODE or an SDE, assuming we have estimated the drifts η 0 , 1 ( α, β, x ) from Definition 2.2 along this curve:

Proposition 2.5 (Probability flow) . Let ( α t , β t ) t ∈ [0 , 1] be any one-parameter family of operators ( α t , β t ) ∈ S . Assume that α t , β t are differentiable for all t ∈ [0 , 1] . Then, for all t ∈ [0 , 1] , the law of I ( α t , β t ) is the same as the law of the solution X t to

<!-- formula-not-decoded -->

This proposition is proven in Appendix A. Similarly, for generation with an SDE, we have:

Proposition 2.6 (Diffusion) . Assume that H = R d and the probability distribution µ α,β of the stochastic interpolant I ( α, β ) is absolutely continuous with respect to the Lebesgue measure. Assume also that, in I ( α, β ) , x 0 is Gaussian and x 0 ⊥ x 1 . Then, under the same conditions as in Proposition 2.5, if α t is invertible, for all t ∈ [0 , 1] and any ϵ t ⩾ 0 , the law of I ( α t , β t ) is the same as the law of the solution X ϵ t to

<!-- formula-not-decoded -->

where W t is a Wiener process in R d

<!-- formula-not-decoded -->

This proposition is also proven in Appendix A. Note that the SDE (8) reduces to the ODE (7) if we set ϵ t = 0 . Note that if α t is positive-definite we can use ¯ ϵ t = ϵ t α -1 t as new diffusion coefficient, which set the noise term in (8) to √ 2 ϵ t α 1 / 2 t dW t ; this allows to extend this SDE to paths along which we can have α t = 0 .

<!-- image -->

## 3 Multitask generation

In this section we discuss how to use the theoretical framework introduced in Section 2 to perform multiple generative tasks without having to doing any retraining.

## 3.1 Self-supervised generation and inpainting

May 15, 2025 1 Inpainting-the task of filling in missing parts of an image-traditionally requires specialized training for each possible mask configuration. Our operator-based framework enables a fundamentally different approach: a single model that can perform inpainting with any arbitrary mask, chosen at inference time, or can generate samples from scratch in an arbitrary ordering of the generation. This includes standard generation of all dimensions at once, autoregressive generation dimension by dimension, blockwise fractal generation, and so forth. In particular, we may want to construct a generative model that fills in missing entries from a sample x 1 ∈ R d drawn from a data distribution µ 1 . We would like this model to be universal, in the sense that it can be used regardless of which entries are missing; their number and position can be arbitrary and changed post-training, allowing for flexible inpainting and editing (see Figure 1 for an illustration). This approach creates a natural self-supervision mechanism, as the model continuously tracks which parts have been generated and which remain to be filled.

To perform this task, assume that x 1 is drawn from the data distribution µ 1 of interest and x 0 drawn independently from N (0 , Id ) , so that µ = N (0 , Id ) × µ 1 , set β = 1 -α in the operator interpolant (1), and assume that α is a diagonal matrix. With a slight abuse of notations we can then identify the diagonal elements of the matrix α with a vector α ∈ R d and write (1) as

<!-- formula-not-decoded -->

where ⊙ denotes the Hadamard (i.e. entrywise) product. The drift to learn in this case is

<!-- formula-not-decoded -->

for α ∈ [0 , 1] d , and this learning can be done via solution of

<!-- formula-not-decoded -->

Denoting x 1 = ( x 1 1 , x 2 1 , . . . , x d 1 ) , suppose that we observe x i 1 for the entries with i ∈ σ ⊂ { 1 , . . . , d } and would like to infer the missing entries with i ∈ σ c = { 1 , . . . , d } \ σ . To perform this inpainting we can use the probability flow ODE (7) with a path α t such that α i t = 0 if i ∈ σ and α i t = 1 -t if i ∈ σ c . Note that this can be done for any choice of σ without retraining .

## 3.2 Multichannel denoising

Suppose that B 1 , B 2 , . . . B n are deterministic corruption operators that can be applied to the data. For example B 1 could be a high-pass filter, B 2 a motion blur, etc. and they could be defined primarily in the Fourier representation of the data. Similarly, if x 0 ∼ N (0 , Id ) , let A 1 , A 2 , . . . A m be operators

Figure 1: Multi-task, self-supervised sampling: Schematic representation of various sub-tasks that are captured by the minimizer of our learning objective using the Hadamard-product interpolant in (9). A generative task is chosen in a zero-shot manner by specifying α as a function of time after training. This α t serves as a continual self-supervision of what has been unmasked vs. what remains. Top : α t is chosen to generate pixels in an autoregressive fashion. Middle : α t is chosen to sample along a fractal morton order. Bottom : α t can be chosen to do zero-shot inpainting.

t

t

1

t

2

t

Figure 2: Multichannel denoising: Possible interpolations fulfilled by various choices of operators in (12). We present two such examples in the form of Gaussian and motion blurring, realized by interpolations defined in the Fourier domain.

<!-- image -->

that give some structure to this noise (e.g. some spatial correlation over the domain of the data). Set A 0 = B 0 = Id and take

<!-- formula-not-decoded -->

where a i , b i are nonegative scalars each taking values in some range that includes 0 and b 0 = 1 . With this choice we can find path ( α t , β t ) t ∈ [0 , 1] that bridges data x 1 ∼ µ corrupted in any possible channel as ∑ m i =1 a i A i x 0 + ∑ n i =1 b i B i x 1 for some choice of ( a 1 , a 2 , . . . , b 1 , b 2 , . . . ) back to the clean data via a path that bridges this choice of parameters to b 0 = 1 and a 0 = a 1 . . . = b 1 = . . . = 0 . See Figure 2 for an illustration of two possible corruption schemes.

## 3.3 Fine-tuning and posterior sampling

May 15, 2025

2 Suppose that we are given data from a distribution µ 1 ( dx ) , and that we would like to generate samples from µ r 1 ( dx ) = Z -1 e r ( x ) µ 1 ( dx ) where r : H → R is a reward function and Z = ∫ H e r ( x ) µ 1 ( dx ) is a normalization function, which is unknown to us but we assume finite - in the context of Bayesian inference, µ 1 plays the role of prior distribution, r is the likelihood, and µ r 1 is the posterior distribution. We will assume that the reward r is a quadratic function, i.e.

<!-- formula-not-decoded -->

where A is a definite negative bilinear operator on H , b ∈ H , and ⟨· , ·⟩ denotes the inner product on H . For simplicity we also assume that H = R d : the general case can be treated similarly.

In this context, assume that we have learned the drifts η 0 , 1 associated with the interpolant

<!-- formula-not-decoded -->

so that we can generate samples from the prior distribution. Our next result shows that this gives us access to the drifts η r 0 , 1 associated with the interpolant

<!-- formula-not-decoded -->

involving data x r 1 from the posterior distribution.

## Proposition 3.1. Let

<!-- formula-not-decoded -->

be the drifts associated with the interpolant (14) and

<!-- formula-not-decoded -->

be the drifts associated with the interpolant (15) . If α and β are invertible, then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

as long as we can find a pair ( α r , β r ) that satisfies

<!-- formula-not-decoded -->

= 0

=

3

=

3

= 1

and x r is given by

<!-- formula-not-decoded -->

This proposition is proven in Appendix A as a corollary of Proposition A.1 that relates the probability distribution of I r to that of I . Proposition 3.1 offers a way to sample the posterior distribution without retraining, by using the drifts (18) and (19) in the ODE (7) or the SDE (8).

## 3.4 Inference adaption

Suppose that we have learned the drifts η 0 , 1 in Definition 2.2 and wish to transport samples along a path ( α t , β t ) with fixed end points. We can leverage the flexibility of our formulation to perform inference adaptation , that is, optimize the path ( α t , β t ) used during generation to achieve specific objectives, such as minimizing computational cost, maximizing sample quality, or satisfying user constraints. This can be done in two ways: (1) offline optimization , where we pre-compute optimal paths for different scenarios using objectives like Wasserstein length minimization, and (2) online adaptation , where paths are dynamically adjusted during generation based on intermediate results or user feedback.

In the case of offline optimization, we could for example optimize the Wasserstein length of the path. That is, if we want to bridge the distributions µ α 0 ,β 0 of I ( α 0 , β 0 ) and µ α 1 ,β 1 of I ( α 1 , β 1 ) via µ α t ,β t with ( α t , β t ) ∈ S for all t ∈ [0 , 1] then the path that minimizes the Wasserstein length of the bridge distribution µ α t ,β t solves

<!-- formula-not-decoded -->

where the minimization is performed over paths ( α t , β t ) t ≡ ( α t , β t ) t ∈ [0 , 1] such that ( α t , β t ) ∈ S for all t ∈ [0 , 1] with their end points ( α 0 , β 0 ) and ( α 1 , β 1 ) prescribed and fixed.

## Algorithm 1: Multitask learner

input: Samples ( x 0 , x 1 ) ∼ µ ; choice of distribution ν ( dα, dβ ) and associated sampler.

## repeat

Draw batch ( x i 0 , x 1 i , α i , β i ) i M =1 ∼ µ × ν .

<!-- formula-not-decoded -->

Compute I i = α i x i 0 + β i x i 1 .

Take a gradient step on ˆ L to update ˆ η 0 and ˆ η 1 .

until converged ;

output:

Drifts ˆ η 0 and ˆ η 1 .

## Algorithm 2: Multitask generator

input: Drifts ˆ η 0 , ˆ η 1 ; choice of path ( α t β t ) t ∈ [0 , 1] tailored to the generation task; data I ( α 0 , β 0 ) = α 0 x 0 + β 0 x 1 ; diffusion coefficient ϵ t ⩾ 0 ; time step h = 1 /K with K ∈ N . initialize: ˆ X ϵ 0 = I ( α 0 , β 0 ) ;

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

output: ˆ X ϵ 1 d = I ( α 1 , β 1 ) (approximately)

## 4 Algorithmic aspects

The algorithmic aspects of our framework can be summarized in a few key steps. First, we define a connected set S of ( α, β ) such that the ensemble of different tasks we will want to perform correspond

to getting samples of I ( α, β ) at some value of ( α 0 , β 0 ) ∈ S and generating from them new data at another value of ( α 1 , β 1 ) ∈ S . Second, we specify a measure ν on S for the learning of the drifts η 0 and η 1 defined in (2). Third, we learn these drifts via minimization of the objectives in (3) and (4), using the procedure outlined in Algorithm 1. Note that we can possibly simplify this algorithm, learning only one of the two drifts and obtaining the other through the relation (5). Finally, given any pairs ( α 0 , β 0 ) , ( α 1 , β 1 ) ∈ S , we use a path ( α t , β t ) t ∈ [0 , 1] with α t , β t ∈ S for all t ∈ [0 , 1] and integrate the SDE (8) (or possibly the ODE (7) if we set ϵ t = 0 ) to perform the generation, as outlined in Algorithm 2. Note that this path could also be adapted on-the-fly during inference, using some feedback about the solution of the SDE.

## 5 Numerical experiments

Below we provide numerical realization of some of the various objectives that can be fulfilled with the multitask objective. Details of the experimental setup can be found in Appendix B.

## 5.1 Multitask inpainting and sequential generation

We evaluate our method on three datasets: MNIST, with images of size 28 × 28 , CelebA, resized to 128 × 128 , and of Animal FacesHQ focused on cat class, with images resized to 256 × 256 . Details of the experimental setup are standard and can be found in Appendix B. In these experiments, we use the Hadamard interpolant (9).

MNIST. We demonstrate the versatility of our operator-based interpolant framework through inpainting and sequential generation tasks on MNIST. The results are shown Figure 3 where all the generated images come from the same model without any retraining.

For inpainting (left panels), we replace masked regions with Gaussian noise (shown as pink for clarity), then generate only these regions while preserving unmasked pixels. This is achieved by setting the entries of α to 1 -t for masked pixels and 0 for unmasked ones. To preserve unmasked pixels, we apply a secondary mask setting η ( α, x ) to zero at these positions.

Sequential generation (right panels) reformulates image creation as progressive inpainting. Starting with pure Gaussian noise, we generate the image block-by-block by successively updating the operator masks. Unlike single-pass inpainting, this requires multiple forward passes-one per block. For each pass, we apply α = 1 -t only to pixels in the current generation block, maintaining appropriate values for previously generated and remaining noise regions.

CelebA and AFHQ-Cat. We present benchmark results for all methods across various image restoration tasks, evaluating the average peak signal-to-noise ratio (PSNR) and structural similarity index (SSIM) on 100 test images from each dataset: AFHQ-Cat ( 256 × 256 ) and CelebA ( 128 × 128 ). To assess the performance of our methodology, we employed two types of masking: square masks of sizes 40 × 40 and 80 × 80 with added Gaussian noise of standard deviation 0.05, and random masks covering 70% of image pixels with Gaussian noise of standard deviation 0.01. We benchmark our method against four state-of-the-art interpolant-based restoration methods: PnP-Flow Martin et al. (2025), Flow-Priors Zhang et al. (2024), D-Flow Ben-Hamu et al. (2024), OT-ODE Pokle et al. (2024).

As shown in Table 1, our method consistently ranks either first or second in both reconstruction metrics across all tasks and datasets (with all values except the last row taken from Martin et al. (2025)). Regarding visual quality (Fig. 4), our method generates realistic, artifact-free images, albeit with slight over-smoothing at times.

## 5.2 Posterior sampling in the ϕ 4 -model

We apply our approach in the context of the ϕ 4 model in d = 2 spacetime dimensions, a statistical lattice field theory where field configurations ϕ ∈ R L × L represent the lattice state ( L denotes spatiotemporal extent)-for details see Appendix B.2. This model poses sampling challenges due to its phase transition from disorder to full order, during which neighboring sites develop strong correlations in sign and magnitude Vierhaus (2010); Albergo et al. (2019).

Table 1: PSNR and SSIM metrics for image inpainting methods on CelebA and AFHQ-Cat datasets.

|                        | CelebA   | CelebA   | CelebA   | CelebA   | AFHQ-Cat   | AFHQ-Cat   | AFHQ-Cat   | AFHQ-Cat   | AFHQ-Cat   | AFHQ-Cat   |
|------------------------|----------|----------|----------|----------|------------|------------|------------|------------|------------|------------|
| Method                 | Random   | Random   | Block    |          | Random     | Random     | Block      | Block      | Block      |            |
|                        | PSNR     | SSIM     | PSNR     | SSIM     | PSNR       | SSIM       | PSNR       | SSIM       |            |            |
| Degraded               | 11.82    | 0.197    | 22.12    | 0.742    | 13.35      | 0.234      | 21.50      | 0.744      |            |            |
| Pokle et al. (2024)    | 28.36    | 0.865    | 28.84    | 0.914    | 28.84      | 0.838      | 23.88      | 0.874      |            |            |
| Ben-Hamu et al. (2024) | 33.07    | 0.938    | 29.70    | 0.893    | 31.37      | 0.888      | 26.69      | 0.833      |            |            |
| Zhang et al. (2024)    | 32.33    | 0.945    | 29.40    | 0.858    | 31.76      | 0.909      | 25.85      | 0.822      |            |            |
| Martin et al. (2025)   | 33.54    | 0.953    | 30.59    | 0.943    | 32.98      | 0.930      | 26.87      | 0.904      |            |            |
| Ours                   | 33.76    | 0.967    | 29.98    | 0.938    | 33.11      | 0.945      | 26.96      | 0.914      |            |            |

Figure 3: Left : In-painting on MNIST using various corruptions. Right : Image generation in arbitrary orders, starting from the same initial noise, with examples showing autoregressive, block-wise, and column-wise.

<!-- image -->

May 15, 2025 1 The ϕ 4 model is specified by the following probability distribution

<!-- formula-not-decoded -->

where Z = ∫ R L × L e -E ( ϕ ) dϕ is a normalization constant and E ( ϕ ) is an energy function defined as

<!-- formula-not-decoded -->

where a, b ∈ [0 , . . . , L -1] 2 denote the discrete positions on a 2 -dimensional lattice of size L × L , a ∼ b denotes neighboring sites on the lattice, and we assume periodic boundary conditions; χ &gt; 0 , κ ∈ R and γ &gt; 0 are parameters. We perform MCMC simulations to generate configuration in a parameter range close to the phase transition. We use these data to learn a stochastic interpolant of the form (9) which allows us to perform unconditional generation of new field configurations as well arbitrary inpainting (conditional generation given partially observed configurations), as reported in Appendix B.2. It also allows us to test the formalism of Section 3.3 and perform sampling of the posterior distribution defined by adding a applied field h ∈ R to the energy, i.e. using

<!-- formula-not-decoded -->

<!-- image -->

<!-- image -->

AFHQ-Cat

CelebA

Figure 4: Inpainting using various masks left panels : AFHQ-Cat ( 256 × 256 ). Right panels : CelebA ( 128 × 128 ). Fixing block and random corruptions are scored against related works in Table 1, showing competitive or superior performance in all metrics.

May 15, 2025 1 Figure 5: Simulating a lattice ϕ 4 theory. Top left : L = 32 × L = 32 lattice configurations at the phase transition. Bottom left : lattice examples with drift parameter h = 0 . 02 . Top middle : Generated lattice examples at phase transition. Bottom middle : generated lattice examples with field h = 0 . 02 . Right : magnetization of 2000 lattice configurations.

<!-- image -->

The additional term plays the role of a reward. The results of the generation based on Proposition 3.1 are shown on Figure 5, which indicate that our approach permits to valid sample configurations (as verified by their magnetization) of the posterior without retraining.

## 5.3 Planning and decision making in a maze

This section applies our framework to shortest path planning in maze environments, drawing from Janner et al. (2022) and Chen et al. (2024). Using the Hadamard product interpolant (9), we can impose that the paths pass through arbitrary locations in the maze (by setting α i = 0 at these locations), reformulating planning as a zero-shot inpainting problem. Unlike traditional reinforcement learning approaches that generate paths sequentially through Markov decision processes, our method therefore produces entire trajectories simultaneously. It also avoid additional guiding terms like Monte Carlo guidance used in Diffusion Forcing Chen et al. (2024).

For training, we use paths of length 300 randomly extracted from the trajectory of length 2,000,000 from Chen et al. (2024). For simplicity, we subsample these paths every six points, creating sparse paths of length 50, from which we can recover paths of length 300 through linear interpolation between consecutive points. At inference, we perform zero-shot generation between any two points in the maze by enforcing that the trajectory passes through these points: the length of the path between these locations can be varied by pinning the first point by setting α 1 = 0 , and the second point by setting α i = 0 with a value of i ∈ [2 , 50] that can be adjusted (see Appendix B.3 for details). Typical results are shown in Fig. 6. In terms of quality assessment, we check that the generated trajectories remain within allowed maze regions: all the 10,000 paths we generated between randomly chosen point pairs avoided the forbidden areas, demonstrating robust performance. More numerical experiments in Appendix B.3 demonstrate that with a similar strategy, one can impose the pathway to take detour at will, even if it implies generating a longer path.

Figure 6: One-shot generation of pathways between two arbitrary points in the maze. The path length is automatically tuned via a simple heuristic, see discussion in Appendix B.3.

<!-- image -->

## Acknowledgements

We would like to thank Yilun Du for many helpful discussions on the maze planning problem. MSA is supported by a Junior Fellowship at the Harvard Society of Fellows as well as the National Science Foundation under Cooperative Agreement PHY-2019786 (The NSF AI Institute for Artificial Intelligence and Fundamental Interactions, http://iaifi.org).

## References

- Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations , 2022.
- Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In The Eleventh International Conference on Learning Representations , 2022.
- Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations , 2022.
- Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in neural information processing systems , volume 33, pages 6840-6851, 2020.
- Yang Song and Stefano Ermon. Generative Modeling by Estimating Gradients of the Data Distribution. arXiv:1907.05600 , 2020.
- Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion schrödinger bridge with applications to score-based generative modeling. In Advances in Neural Information Processing Systems , volume 34, pages 17695-17709, 2021.
- Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797 , 2023a.
- Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 10684-10695, 2022.
- François Mazé and Faez Ahmed. Diffusion models beat gans on topology optimization. In Proceedings of the AAAI conference on artificial intelligence , volume 37, pages 9108-9116, 2023.
- Michael Alverson, Sterling G Baird, Ryan Murdock, Jeremy Johnson, Taylor D Sparks, et al. Generative adversarial networks and diffusion models in material discovery. Digital Discovery , 3 (1):62-80, 2024.
- Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In International Conference on Learning Representations , 2023. URL https://openreview.net/forum?id=OnD9zGAGT0k .
- Yang Song, Liyue Shen, Lei Xing, and Stefano Ermon. Solving inverse problems in medical imaging with score-based generative models. In International Conference on Learning Representations , 2022. URL https://openreview.net/forum?id=vaRCHVj0uGI .
- Hengkang Wang, Xu Zhang, Taihui Li, Yuxiang Wan, Tiancong Chen, and Ju Sun. DMPlug: A plug-in method for solving inverse problems with diffusion models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv:2011.13456 , 2020.
- Michael S. Albergo, Mark Goldstein, Nicholas M. Boffi, Rajesh Ranganath, and Eric Vanden-Eijnden. Stochastic interpolants with data-dependent couplings. arXiv:2310.03725 , 2023b.
- Marcelo Pereyra, Philip Schniter, Emilie Chouzenoux, Jean-Christophe Pesquet, Jean-Yves Tourneret, Alfred O Hero, and Steve McLaughlin. A survey of stochastic simulation and optimization methods in signal processing. IEEE Journal of Selected Topics in Signal Processing , 10(2):224-241, 2015.
- Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. Advances in Neural Information Processing Systems , 35:23593-23606, 2022.

- Ségolène Martin, Anne Gagneux, Paul Hagemann, and Gabriele Steidl. Pnp-flow: Plug-and-play image restoration with flow matching. In International Conference on Learning Representations , 2025.
- Florentin Coeurdoux, Nicolas Dobigeon, and Pierre Chainais. Plug-and-play split gibbs sampler: embedding deep generative priors in bayesian inference. IEEE Transactions on Image Processing , 2024.
- Yu Sun, Zihui Wu, Yifan Chen, Berthy T Feng, and Katherine L Bouman. Provable probabilistic imaging using score-based generative priors. IEEE Transactions on Computational Imaging , 2024.
- Morteza Mardani, Jiaming Song, Jan Kautz, and Arash Vahdat. A variational perspective on solving inverse problems with diffusion models. arXiv preprint arXiv:2305.04391 , 2023.
- Cagan Alkan, Julio Oscanoa, Daniel Abraham, Mengze Gao, Aizada Nurdinova, Kawin Setsompop, John M Pauly, Morteza Mardani, and Shreyas Vasanawala. Variational diffusion models for blind mri inverse problems. In NeurIPS 2023 workshop on deep learning and inverse problems , 2023.
- Giannis Daras, Yuval Dagan, Alexandros G Dimakis, and Constantinos Daskalakis. Score-guided intermediate layer optimization: Fast langevin mixing for inverse problems. arXiv preprint arXiv:2206.09104 , 2022.
- Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie S. Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Cold diffusion: Inverting arbitrary image transforms without noise, 2022. URL https://arxiv.org/abs/2208.09392 .
- Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction, 2024. URL https://arxiv.org/abs/ 2404.02905 .
- Tianhong Li, Qinyi Sun, Lijie Fan, and Kaiming He. Fractal generative models, 2025. URL https://arxiv.org/abs/2502.17437 .
- Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K. Titsias. Simplified and generalized masked diffusion for discrete data, 2025. URL https://arxiv.org/abs/2406. 04329 .
- Jaeyeon Kim, Kulin Shah, Vasilis Kontonis, Sham Kakade, and Sitan Chen. Train for the worst, plan for the best: Understanding token ordering in masked diffusions, 2025. URL https: //arxiv.org/abs/2502.06768 .
- Yasi Zhang, Peiyu Yu, Yaxuan Zhu, Yingshan Chang, Feng Gao, Ying Nian Wu, and Oscar Leong. Flow priors for linear inverse problems via iterative corrupted trajectory matching. Advances in Neural Information Processing Systems , 37:57389-57417, 2024.
- Heli Ben-Hamu, Omri Puny, Itai Gat, Brian Karrer, Uriel Singer, and Yaron Lipman. D-flow: Differentiating through flows for controlled generation. In Proceedings of the 41st International Conference on Machine Learning , volume 235 of Proceedings of Machine Learning Research , pages 3462-3483. PMLR, 2024.
- Ashwini Pokle, Matthew J. Muckley, Ricky T. Q. Chen, and Brian Karrer. Training-free linear image inverses via flows. Transactions on Machine Learning Research , 2024. ISSN 2835-8856.
- Ingmar Vierhaus. Simulation of ϕ 4 theory in the strong coupling expansion beyond the Ising Limit . PhD thesis, Humboldt University of Berlin, 07 2010.
- M. S. Albergo, G. Kanwar, and P. E. Shanahan. Flow-based generative models for markov chain monte carlo in lattice field theory. Phys. Rev. D , 100:034515, Aug 2019. doi: 10.1103/PhysRevD. 100.034515. URL https://link.aps.org/doi/10.1103/PhysRevD.100.034515 .
- Michael Janner, Yilun Du, Joshua B. Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis, 2022. URL https://arxiv.org/abs/2205.09991 .

Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion, 2024. URL https: //arxiv.org/abs/2407.01392 .

David Williams. Probability with Martingales . Cambridge Mathematical Textbooks. Cambridge University Press, Cambridge, 1991. ISBN 9780521406055. doi: 10.1017/CBO9780511813658.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017. URL https://arxiv.org/abs/1412.6980 .

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. CoRR , abs/1505.04597, 2015. URL http://arxiv.org/abs/1505. 04597 .

## A Proofs

Definition 2.2 (Multipurpose drift) . The drifts η 0 , η 1 : S ×H→H are given by

<!-- formula-not-decoded -->

where E [ · | I ( α, β ) = x ] denotes expectation over the coupling ( x 0 , x 1 ) ∼ µ conditioned on the event I ( α, β ) = x .

Lemma 2.3 (Drift objective) . Let ν ( dα, dβ ) be a probability distribution whose support is S . Then the drifts η 0 , 1 ( α, β, x ) in Definition 2.2 can be characterized globally for all ( α, β ) ∈ S and all x ∈ supp( µ α,β ) via solution of the optimization problems

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ∥ · ∥ denotes the norm in H .

Proof. The lemma is a simple consequence of the L 2 characterization of the conditional expectation as least-squares-best predictor, see e.g. Section 9.3 in Williams (1991).

Lemma 2.4 (Score) . Assume that H = R d and that the probability distribution µ α,β of the stochastic interpolant I ( α, β ) is absolutely continuous with respect to the Lebesgue measure with density ρ α,β ( x ) . Assume also that x 0 ∼ N (0 , Id ) and x 0 ⊥ x 1 . Then the score s α,β ( x ) = ∇ log ρ α,β ( x ) is related to the drift η 0 ( α, β, x ) via

<!-- formula-not-decoded -->

Proof. The lemma follows from Stein's lemma (aka Gaussian integration by parts formula) that asserts that

<!-- formula-not-decoded -->

as well as the definition of η 0 ( α, β, x )

in (16).

Proposition 2.5 (Probability flow) . Let ( α t , β t ) t ∈ [0 , 1] be any one-parameter family of operators ( α t , β t ) ∈ S . Assume that α t , β t are differentiable for all t ∈ [0 , 1] . Then, for all t ∈ [0 , 1] , the law of I ( α t , β t ) is the same as the law of the solution X t to

<!-- formula-not-decoded -->

Proof. From the framework of standard stochastic interpolants Albergo and Vanden-Eijnden (2022); Albergo et al. (2023a), we know that the law of I t = I ( α t , β t ) is the same for all t ∈ [0 , 1] as the law of X t , i.e. the solution to the probability flow ODE

<!-- formula-not-decoded -->

where and x r is given by

where

<!-- formula-not-decoded -->

By the chain rule ˙ I t = ˙ α t x 0 + ˙ β t x 1 so that

<!-- formula-not-decoded -->

where η 0 , 1 are the drifts defined in (16). This means that (27) is (7).

Proposition 2.6 (Diffusion) . Assume that H = R d and the probability distribution µ α,β of the stochastic interpolant I ( α, β ) is absolutely continuous with respect to the Lebesgue measure. Assume also that, in I ( α, β ) , x 0 is Gaussian and x 0 ⊥ x 1 . Then, under the same conditions as in Proposition 2.5, if α t is invertible, for all t ∈ [0 , 1] and any ϵ t ⩾ 0 , the law of I ( α t , β t ) is the same as the law of the solution X ϵ t to

<!-- formula-not-decoded -->

where W t is a Wiener process in R d .

Proof. From the framework of standard stochastic interpolants Albergo and Vanden-Eijnden (2022); Albergo et al. (2023a), we know that the law of the solution X t to the probability flow ODE (27) is the same for all t ∈ [0 , 1] as the law of X ϵ t solution to to SDE

<!-- formula-not-decoded -->

where s t ( x ) is the score of the probability density function of I t = I ( α t , β t ) . Since s t ( x ) = s α t ,β t ( x ) , from Lemma 2.4, we have

<!-- formula-not-decoded -->

If we insert this expression in (30) and use (29), we see that this SDE reduces to (8).

## Proposition 3.1. Let

<!-- formula-not-decoded -->

be the drifts associated with the interpolant (14) and

<!-- formula-not-decoded -->

be the drifts associated with the interpolant (15) . If α and β are invertible, then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

as long as we can find a pair ( α r , β r ) that satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We will prove this proposition as a corollary of:

Proposition A.1 (Posterior distributions) . Let µ α,β and µ r α,β be the probability distributions of the stochastic interpolants defined in (14) and (15) , respectively. Assume that α and β are invertible, that the equations (20) for α r , β r in Proposition 3.1 have a solution, and that x r is given by (21) . Then these distributions are related, up to a constant independent of x and x r , as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. By definition of the probability distribution µ r α,β ( dx ) of I r ( α, β ) , given any integrable and bounded test function ϕ : R d → R we have

<!-- formula-not-decoded -->

If instead of x 0 we use as new integration variable x = αx 0 + βx 1 , this becomes

<!-- formula-not-decoded -->

Similarly, for the probability distribution µ α,β ( dx ) of I ( α, β ) , we have

<!-- formula-not-decoded -->

If in this equation we replace α by α r , β by β r , x by x r , and ϕ ( x ) by ϕ ( x ) e R ( α,β,x ) , and multiply both side by | α r | / | α | it becomes:

<!-- formula-not-decoded -->

We can now require that the right hand side of (37) be the same as at the right hand-side of (35) (so that, µ r α,β ( dx ) = | α r || α | -1 e R ( α,β,x ) µ α r ,β r ( dx r ) ), we arrive at the requirement that

<!-- formula-not-decoded -->

where we used r ( x ) = 1 2 ⟨ x, Ax ⟩ + ⟨ b, x ⟩ . Since (38) must hold for all x 1 , we can expand both sides of this equation, and equate the coefficient of order 2 , 1 and 0 in x 1 . They are completely equivalent to (20), (21), and (33), respectively. So as long as we can find solutions to (20), (32) holds.

Proof of Proposition 3.1. By definition of the conditional expectation, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If in the first equality we replace α by α r , β by β r , and x by x r , and assume that α r , β r , and x r satisfy (38), by construction we obtain that (19) holds. To get (18), use (19) as well as relation (5) twice to deduce

<!-- formula-not-decoded -->

## B Experimental details

Details for the experiments in Section 5 are provided here.

May 15, 2025

1 Figure 7: ϕ 4 -model: Inpainting of three different configurations.

<!-- image -->

## B.1 Multitask inpainting and sequential generation

For all image generation experiments, the U-Net architecture originally proposed in Ho et al. (2020) is used. The specification of architecture hyperparameters as well as training hyperparameters are given in Table 2. Training was done for 200 epochs on batches comprised of 30 draws from the target, and 50 time slices. The objectives given in 3 and 4 were optimized using the Adam optimizer. The learning rate was set to .0001 and was dropped by a factor of 2 every 1500 iterations of training. To integrate the ODE/SDE when drawing samples, we used a simple Euler integrator.

In order to progressively explore the space of the hypercube of α and β , we first learn a model in the diagonal of the hypercube, i.e where all entries of α are all the same value. We then fine-tune the first model for matrices α t uniformly distributed in [0 , 1] d . We also fine-tune the first model for matrices α t decomposed by blocks of 4 × 4 where entries of each blocks contains the same values in [0 , 1] d .

Table 2: Hyperparameters and architecture for MNIST, ϕ 4 and maze datasets.

|                      | MNIST     | ϕ 4       | CelebA    | AFHQ-Cat   |
|----------------------|-----------|-----------|-----------|------------|
| Dimension            | 28 × 28   | 32 × 32   | 128 × 128 | 256 × 256  |
| # Training point     | 60,000    | 100,000   | 190,000   | 5,000      |
| Batch Size           | 50        | 100       | 128       | 64         |
| Training Steps       | 4 × 10 5  | 2 × 10 5  | 4 × 10 5  | 4 × 10 5   |
| Attention Resolution | 64        | 64        | 64        | 64         |
| Learning Rate (LR)   | 0 . 0002  | 0 . 0002  | 0 . 0001  | 0 . 0001   |
| LR decay (1k epochs) | 0.995     | 0.995     | 0.995     | 0.995      |
| U-Net dim mult       | [1,2,2,2] | [1,2,2,2] | [1,2,4,8] | [1,2,4,8]  |
| Learned t embedding  | Yes       | Yes       | Yes       | Yes        |
| # GPUs               | 1         | 1         | 4         | 4          |

## B.2 Details about the ϕ 4 Model

We define the discrete Fourier transform as

<!-- formula-not-decoded -->

where a, k ∈ [0 , . . . , L -1] d , we can write the energy (24) as E ( ϕ ) = E 0 ( ϕ ) + U ( ϕ ) with

<!-- formula-not-decoded -->

where ˆ e denotes the d basis vectors on the lattice and β 0 &gt; 0 is an adjustable parameter; and

<!-- formula-not-decoded -->

where ϕ and ˆ ϕ are Fourier transform pairs as defined in (42): the last term can be implemented via ∑ a ( ifft ( ˆ ϕ )) 4 ( a ) .

Sampling using the Langevin SDE: To obtain the ground-truth samples from the ϕ 4 model, one option is to use the SDE

<!-- formula-not-decoded -->

where we denote

<!-- formula-not-decoded -->

which can be implemented via fft (( ifft ( ˆ ϕ t )) 3 ) . This SDE may be quite stiff, however, a problem that can be alleviated by changing the mobility and using instead

<!-- formula-not-decoded -->

The discretized version of this equation reads

<!-- formula-not-decoded -->

where ˆ η n is the Fourier transform of η n ∼ N (0 , Id ) .

Computing the generator of the posterior distribution As illustrated with (25), one would like to sample a slightly different ϕ 4 model with energy function, noted E r . E and E r define respectively the prior and posterior distribution, via the Boltzmann's law (23). In this work, only relations of the form E r = E -⟨ ϕ, Aφ ⟩ - ⟨ h, φ ⟩ are studied, combining a linear and quadratic term. A is assumed to be definite negative.

Lemma B.1. Assume that β = 1 -α and let

<!-- formula-not-decoded -->

Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of lemma B.1. Solving (5) and (49) in ( η 0 , η 1 ) gives (50) and (51).

The idea is to use the drift of the prior to sample from the posterior. The following proposition makes it possible.

Proposition B.2 (Posterior drift) . Assume β = 1 -α with α diagonal and invertible, and let

<!-- formula-not-decoded -->

where the drifts η r 0 and η r 1 are defined in (18) and (19) , respectively. Assume also that A is diagonal, non-positive definite, and invertible. Then, β r = 1 -α r and η r can be expressed as:

<!-- formula-not-decoded -->

where η ( α, x ) is the drift of the prior defined in (49) , and α r and x r are given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of the Proposition B.2. Using (50), (51) in (18) and (19), one obtains:

<!-- formula-not-decoded -->

Then, take the difference, and regrouping terms together eventually yields:

<!-- formula-not-decoded -->

The linear case Assume for now that A = 0 and b = h . Note that one recovers the same case that in (25), that is, one applies a uniform magnetic field of magnitude h over the whole lattice. What follows is simply a corollary of B.2.

Proposition B.3. Assume E r = E -( ϕ, h ) . Then, α r = α , β r = β , and

<!-- formula-not-decoded -->

Also, the posterior drift η r writes:

<!-- formula-not-decoded -->

Proof of Proposition B.3. Since A = 0 , (20) directly implies α r = α and β r = β . Consequently, (56) follow from (21) and (57) from (53).

In summary, a simple shift proportional to h appears in the posterior field ϕ r . It clearly tends to favor the alignment with the magnetic field, which obeys common sense.

## The quadratic case

- Assume that b = 0 and A = -k 2 Id.

In a similar fashion to the linear case, one derives analytical expressions for the quantities of interest.

Proposition B.4. Assume E r = E -( ϕ, Aϕ ) = E + k 2 ∑ a ϕ ( a ) 2 , β r = 1 -α r and β = 1 -α .

Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of the Proposition B.4. Given the assumptions, (20) yields:

<!-- formula-not-decoded -->

Observing that α and A are diagonals, α r is also diagonal. Furthermore, assuming that α r is proportional to the identity, the above reduces to the scalar equation (keeping the same notation for conciseness):

<!-- formula-not-decoded -->

After a few elementary manipulations, one arrives at:

<!-- formula-not-decoded -->

This is a quadratic equation that admits two solutions. Only one is positive, and writes as:

<!-- formula-not-decoded -->

It is quite easy to check that the discriminant is always positive, so it does not pose any problem. Also, if k ⩾ 0 and α ∈ [0 , 1] , then α r ∈ [0 , 1] . This property is necessary, since α ↦→ η ( · , α ) has been trained in the hypercube [0 , 1] d .

After elementary simplifications and recalling that β r = 1 -α r and (21), one has:

<!-- formula-not-decoded -->

Since 1 -α r = 1 -2 α + α 2 k 2 + α 2 -α √ 1 -2 α + α 2 (1+ k 2 ) 1 -2 α + α 2 k 2 = 1 -2 α + α 2 (1+ k 2 ) -α √ 1 -2 α + α 2 (1+ k 2 ) 1 -2 α + α 2 k 2 , it yields:

<!-- formula-not-decoded -->

then factorizing by √ 1 -2 α + α 2 (1 + k 2 ) eventually gives (59).

Eventually, after replacing (62) and (59) into (53), (60) holds.

- Assume that b = 0 and A = k 2 Id.

In this case, the quadratic equation is:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

or otherwise stated:

The discriminant of this polynomial is ∆ = α 2 (1 -k 2 ) -2 α +1 . Assuming it strictly positive, among the two solutions, only one is positive:

<!-- formula-not-decoded -->

The polynomial inside the square root is positive if and only if α / ∈ [ 1 1+ k , 1 1 -k ] . To see that, see there exists always two real roots, since the discriminants is 4 k 2 &gt; 0 . Those roots are 1 1+ k &lt; 1 and 1 1 -k &gt; 1 . Since α r ∈ [0 , 1] must be respected for all α ∈ [0 , 1] , only k &lt; 1 can be considered with our method. Consequently, sampling using stochastic interpolants from α = 1 to α = 0 appears impossible with this method.

## B.3 Details about the maze experiment

We use the Hadamard interpolant (9) and estimate the drift η ( α, x ) defined in (10) by approximating it with a U-Net neural network Ho et al. (2020), trained with an Adam optimizer Kingma and Ba (2017). The U-Net comprises 4 stages with 48 , 80 , 160 , and 256 channels respectively for the encoding flow. The decoder has the same architecture as the encoder but in reverse order, with added residual connections Ronneberger et al. (2015). Each stage consists of 2 residual blocks, with the first concatenated with a self-attention block. The input vector has shape d × 2 , where row i contains the x and y coordinates of the i -th point in the trajectory.

In contrast to conventional U-Net architectures, we perform interpolation and max pooling operations independently on each coordinate column to increase and reduce dimensions only along the trajectory length axis. The convolution kernel size is 5 × 2 , processing each point's coordinates together with those of its two temporal predecessors and successors in the sequence. We add the necessary padding to maintain identical input and output dimensions, which amounts to padding by two rows at the top and bottom of the input vector.

Given a pair of randomly chosen points in the maze, we must determine where to constrain these points along the generated trajectory. If the constraint points are placed too far apart in the sequence (large index difference), the resulting path will likely not be the shortest route; conversely, if placed too close together (small index difference), the generated path has an increased chance of cutting through forbidden regions, making it inadmissible. To address this trade-off, we adopt the following heuristic. We fix the starting point at the beginning of the path (index i = 1 ) and employ a progressive search for the target point placement using the candidate indices [5 , 10 , 20 , 30 , 40 , 45 , 50] . We first generate a path with the target point constrained at index 5 (creating a short trajectory). If this path intersects forbidden regions, we increase the target index to 10 (allowing a longer path), and continue this process until we generate a valid path that successfully avoids all obstacles.

On Figure 8, we impose paths to go by the bottom-right corner, the constraint is visible as a small white dot. The path length adapts accordingly.

<!-- image -->

start

Figure 8: Two generated pathways, one per row. Left : No constraint is imposed on the path, other than joining the two endpoints. Right : An additional constraint is imposed: the path must pass through the bottom-right corner, represented by a white dot.

## C Additional experimental results

Here we provide additional infilling image results, given in Figure 9.

Figure 9: Additional images demonstrating the inpainting task: Block inpainting is shown in the top panels, while random inpainting is displayed in the bottom panels. The left panels depict images from the AFHQ dataset, and the right panels show images from the CelebA dataset.

<!-- image -->

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (1-2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

IMPORTANT, please:

- Delete this instruction block, but keep the section heading 'NeurIPS Paper Checklist" ,
- Keep the checklist subsection headings, questions/answers and guidelines below.
- Do not modify the questions and only use the provided macros for your answers .

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract claims that the formalism of stochastic interpolants have been adapted for zero-shot conditional generation, in-painting, and posterior sampling. The corresponding numerical experiments are explicitly displayed in their associated sections. Posterior sampling has been theoretically explored and numerically investigated in the ϕ 4 model with stochastic interpolants. We show that our model can learn to generate images at multiple scales: pixel-wise, blockwise, and in all dimensions simultaneously.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: A discussion on this matter is present in the conclusion.

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

Justification: All proofs, with their set of assumptions, are provided in the Appendix.

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

Justification: The details of all the numerical experiments are discussed either in the main text of the paper or in the Appendix, especially concerning the one-shot conditional generation and inpainting.

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

Justification: All the code and data used for the numerical experiments will be exposed in a Github repository. A set of instructions to fully reproduce the results will be provided in a README file.

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

Justification: All test details are provided in Appendix (e.g. Table 2).

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The full distribution of the average magnetization has been studied for our ϕ 4 model in Figure 5.

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

## Answer: [Yes]

Justification: A Table containing all relevant information will be provided in the Appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our paper presents theoretical and mathematical foundations for multitask generative modeling using operator-based interpolants. The research primarily consists of mathematical formulations and theoretical derivations. Our experiments are limited to publicly available benchmark datasets (MNIST digits) and physical simulation data, neither of which contain sensitive information or raise ethical concerns. We do not collect or use personal data, conduct experimentation on humans or animals, or develop technologies with potential for harm or misuse. All sources and prior work are properly cited, and our mathematical derivations, proofs, and experimental procedures are presented transparently. The research was conducted with scientific integrity, honesty, and transparency throughout the process, fully adhering to all aspects of the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

## Answer: [NA]

Justification: Our paper presents a theoretical framework for operator-based interpolants in generative modeling, which primarily advances fundamental research in this area. The work is largely theoretical and mathematical in nature, focusing on developing a universal approach to multitask learning rather than specific applications with direct societal implications. While we discuss potential technical benefits in the paper's conclusion, the theoretical nature and early stage of this research makes specific societal impacts, whether positive or negative, difficult to assess meaningfully at this time.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to

any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: This question is not applicable to our paper as we do not release any models or datasets that present a high risk for misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: In our paper, we properly credit the creators and original owners of all assets used, including the MNIST dataset and any referenced algorithms or methodologies. For the MNIST dataset, which is in the public domain, we acknowledge its source and cite the original publication. All assets are used in accordance with their intended research purposes and we have carefully respected all applicable terms of use and licensing requirements throughout our research process.

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

Justification: Our paper introduces new theoretical formulations and algorithms for operatorbased interpolants, which are thoroughly documented within the paper itself. The mathematical framework, definitions, lemmas, and propositions are rigorously presented with complete derivations and proofs (in the appendix). For our experimental implementations, we provide detailed descriptions of the model architectures, training procedures, and inference methods in the paper and will release accompanying code with comprehensive documentation that explains the implementation of our operator-based interpolant framework. This documentation includes clear instructions for reproducing our experiments, explanations of key parameters, and examples demonstrating how to apply our methods to various tasks. All assets are carefully documented to ensure transparency and reproducibility of our research.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This question is not applicable to our research as our paper does not involve any human subjects or study participants. Our experiments are conducted exclusively on standard benchmark datasets (MNIST) and physics simulation data, with no human participation involved at any stage. Since no human subjects were part of this research, no IRB approvals or equivalent reviews were necessary, and there were no study-related risks to disclose or manage.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 15. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

## Answer: [No]

Justification: No LLM or any kind of transformer architecture is involved in the numerical experiments displayed.

## Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.