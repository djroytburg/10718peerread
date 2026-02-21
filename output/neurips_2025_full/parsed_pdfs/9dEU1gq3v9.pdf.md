## Exploring the Design Space of Diffusion Bridge Models

## Shaorong Zhang, Yuanbin Cheng, Greg Ver Steeg

Unversity of California Riverside {szhan311, ychen871, gregoryv}@ucr.edu

## Abstract

Diffusion bridge models and stochastic interpolants enable high-quality imageto-image (I2I) translation by creating paths between distributions in pixel space. However, recent diffusion bridge models excel in image translation but suffer from restricted design flexibility and complicated hyperparameter tuning, whereas Stochastic Interpolants offer greater flexibility but lack essential refinements. We show that these complementary strengths can be unified by interpreting all existing methods within a single SI-based framework. In this work, we unify and expand the space of bridge models by extending Stochastic Interpolants (SIs) with preconditioning, endpoint conditioning, and an optimized sampling algorithm. These enhancements expand the design space of diffusion bridge models, leading to state-of-the-art performance in both image quality and sampling efficiency across diverse I2I tasks. Furthermore, we identify and address a previously overlooked issue of low sample diversity under fixed conditions. We introduce a quantitative analysis for output diversity and demonstrate how we can modify the base distribution for further improvements. Code is available at https://github.com/szhan311/ECSI .

## 1 Introduction

Denoising Diffusion Models (DDMs) and flow matching create a stochastic process to transition Gaussian noise into a target distribution [33, 14, 34, 19]. Building upon this, diffusion bridge-based models (DBMs) have been developed to transport between two arbitrary distributions, π T and π 0 , including I2SB [21], DSBM [39], DDBM [18], DBIM [42], Bridge Matching [28]. DBMs achieve superior image quality in I2I translation compared to DDMs [18, 21, 2], primarily because the distance between source and target image distributions is typically smaller than that between Gaussian and target distributions.

While DBMs like DDBM [39], DBIM [42], and I2SB [21] achieve state-of-the-art FID scores in image-to-image translation, they suffer from limited design flexibility, constrained bridge path formulations, and complex parameter tuning. In contrast, Stochastic Interpolants (SIs) [1, 2] offer a simpler and more flexible framework, but they have yet to integrate practical advances from recent diffusion bridge models, such as preconditioning. Besides, SIs require training two separate models, unlike the more efficient single-model setup in DDBM. Table 1 summarizes the key characteristics of these methods, highlighting that their complementary strengths had not yet been unified.

Another overlooked issue stemming from restrictive design choices in previous bridge models is the lack of diversity in outputs. While some image translation tasks are one-to-one, we find that in one-to-many translation tasks, like black and white edges to color images, previous methods produce limited variation in colors and textures. We refer to this as the conditional diversity problem and show that our approach leads to significant improvements.

<!-- image -->

Figure 1: The design space of bridge paths and samplers.

|                        | DDBM   | DBIM   | DSBM   | SI   | ECSI (ours)   |
|------------------------|--------|--------|--------|------|---------------|
| Endpoint conditioning  | ✓      | ✓      | ✗      | ✗    | ✓             |
| Uncoupled parameters   | ✗      | ✗      | ✗      | ✓    | ✓             |
| Extensive bridge paths | ✗      | ✗      | ✓      | ✓    | ✓             |
| Extensive samplers     | ✗      | ✗      | ✗      | ✓    | ✓             |
| Preconditioning        | ✓      | ✓      | ✗      | ✗    | ✓             |
| Modified base density  | ✗      | ✗      | ✗      | ✗    | ✓             |

Table 1: Characteristics of different bridge models.

<!-- image -->

Deblurring

Depth to RGB

<!-- image -->

<!-- image -->

Edges to Hangbags

Figure 2: Samples for I2I translation with our ECSI models: Deblurring, Depth-RGB, and Edges to Handbags. For each pair of images, we show the input image (upper) and the output image (bottom).

Our main contributions are as follows:

- We propose Endpoint-Conditioned Stochastic Interpolants (ECSI), which extend stochastic interpolants by incorporating endpoint conditioning and preconditioning. Previous bridge methods artificially coupled unrelated aspects of the transition kernel. ECSI introduces a decoupled parametrization that expands and simplifies the design space for bridge paths and samplers. To further improve sampling quality and efficiency, we develop a novel noise control scheme and an efficient sampling algorithm.
- We identify a previously overlooked issue: the low diversity of outputs conditioned on fixed source images. To address this, we propose modifying the base distribution. Furthermore, to quantitatively evaluate conditional output diversity, we introduce a new metric-Average Feature Diversity (AFD).
- Experimental results demonstrate our model's state-of-the-art performance in both image quality and sampling speed across various I2I tasks, including deblurring, edges-to-handbags translation, and depth-to-RGB conversion. Notably, for handbag generation, our approach yields significantly more diverse outputs with varied colors and textures.

## 2 Background

Notations Let π T , π 0 , and π 0 T represent the base distribution, the target distribution, and the joint distribution of them respectively. π cond and π data represent the distributions of the input and output data. Let p be the distribution of a diffusion process; we denote its marginal distribution at time t by p t , the conditional distribution at time t given the state at time s by p t | s , and the distribution at time t given the states at times 0 and T by p t | 0 ,T , i.e., the transition kernel of a bridge.

## 2.1 Denoising Diffusion Bridge Models

DDBMs [18] extend diffusion models to translate between two arbitrary distributions π 0 and π T given samples from them. Consider a reference process given by:

<!-- formula-not-decoded -->

whose transition kernel is given by q t | 0 ( x t | x 0 ) = N ( x t ; a t x 0 , σ 2 t I ) . This process can be conditioned (or "pinned") at both an initial point x 0 and a terminal point x T to construct a diffusion bridge. Under mild assumptions, the pinned process is given by Doob's h -transform [29]:

<!-- formula-not-decoded -->

where ∇ X t log p T | t ( x T | X t ) = ( a t /a T ) x T -X t σ 2 t (SNR t / SNR T -1) and SNR t := a 2 t /σ 2 t [18]. Eq. (2) is a stochastic process that transport from p 0 = π 0 and p t = π t , which is a valid bridge process. To sample from the conditional distribution p ( x 0 | x T ) , we can solve the reverse SDE or probability flow ODE from t = T to t = 0 [18]:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where X T = x T , s = ∇ X t log p T | t ( x T | X t ) , h = ∇ X t log p t | T ( X t | x T ) . Generally, the score ∇ x t log p t | T ( x t | x T ) in Eqs. (3) and (4) is intractable. However, it can be effectively estimated by denoising bridge score matching. Let ( x 0 , x T ) ∼ π 0 ,T ( x 0 , x T ) , x t ∼ p t | 0 ,T ( x t | x 0 , x T ) , t ∼ U (0 , T ) , and ω ( t ) be non-zero loss weighting term of any choice, then the score ∇ x t log p T | t ( x T | x t ) can be approximated by a neural network s θ ( x t , x T , t ) with denoising bridge score matching objective [18]:

<!-- formula-not-decoded -->

where E dentotes expectation over x t ∼ p t | 0 ,T ( x t , x 0 ) , ( x 0 , x T ) ∼ π 0 ,T , t ∼ U (0 , T ) .

## 2.2 Diffusion Bridge Implicit Models

The transition kernel of the bridge process in Eq. (2) is given by [18, 42]:

<!-- formula-not-decoded -->

where α t = a t (1 -SNR T SNR t ) , β t = a t a T SNR T SNR t , γ 2 t = σ 2 t (1 -SNR T SNR t ) . Suppose we sample in reverse time on the discretized timesteps 0 = t 0 &lt; t 1 &lt; · · · t N -1 &lt; t N = T . Then we can sample x 0 by the initial value x T and the updating rule:

<!-- formula-not-decoded -->

where ˆ x θ 0 ( x t , x T , t ) has the relation with the score function:

<!-- formula-not-decoded -->

## 3 Have the bridge paths been fully explored?

Given the forward process defined in Eq. (1), diffusion bridge models [18, 42, 12, 8] utilize Doob's h -transform to construct a corresponding bridge process (Eq. (2)). While the resulting process effectively bridges the initial distribution π T and the target distribution π 0 , such diffusion bridge approaches exhibit several limitations.

- Parameter coupling. Notice that the parameters a t and σ t are convolved in the transition kernel (Eq. (6)). Such coupling is unnecessary and decoupling those parameters is helpful for searching the 'best' bridge path.

- Limited design space. Despite Eq. (2) provides an infinite number of bridge paths by tuning a t and σ t , but the space of bridge paths is still artificially restricted.

In contrast, the stochastic interpolants [1] framework allows a larger design space of bridge path with more decoupled parameters. Specifically, stochastic interpolants build a bridge path directly via the flow map:

<!-- formula-not-decoded -->

where z ∼ N (0 , I ) . Eq. (9) builds a transport with π 0 and π T as boundary conditions if the kernel parameters satisfy [1]:

- α 0 = β T = 1 and α T = β 0 = γ 0 = γ 1 = 0 ;
- α t , β t , γ t &gt; 0 for t ∈ (0 , T ) .

The transition kernel of the stochastic interpolants in Eq. (9) is a Gaussian distribution: N ( x t ; α t x 0 + β t x T , γ 2 t I ) . Unlike DDBM, which is parameterized by only two variables a t and σ t , stochastic interpolants introduce decoupled parameters α t , β t , and γ t , offering a more flexible and expressive design space for constructing bridge paths.

A detailed discussion on the rationale behind the choices of α t , β t , and and an ablation study on the shape of γ t is provided in App. E. Notably, the DDBM-VP and DDBM-VE models presented in [18] can be considered as special cases by choosing different α t , β t , and γ t , see App. D for more details. In the experiments, we limit the scope to linear transition kernels and set T = 1 , i.e., p t | 0 ,T ( x t | x 0 , x T ) = N ( x t ; (1 -t ) x 0 + tx 1 , 4 γ 2 max t (1 -t ) I ) .

Stochastic interpolants expands the space of bridge paths and leads to decoupled parameters compared to DDBM and DBIM.

## 4 Has the sampler space been fully explored?

For diffusion models, EDM [17] demonstrated that the design of training and sampling schemes could be decoupled to significantly improve results. We now explore whether a similar decoupling is possible for bridge models, and what freedom we have to improve sampling quality with a given trained model.

## 4.1 Endpoint-Conditioning for Stochastic Interpolants (ECSI)

Given transition kernel p t | 0 ,T ( x t | x 0 , x T ) = N ( x t ; α t x 0 + β t x T ; γ t I ) , we can identify the training objective 11, reverse sampling SDEs (Eq. (10)), as demonstrated in Proposition 4.1, see App. C for the proof.

Proposition 4.1 (Endpoint-conditioned Stochastic Interpolants) . Suppose the transition kernel of a diffusion bridge process is given by p t | 0 ,T ( x t | x 0 , x T ) = N ( x t ; α t x 0 + β t x T , γ 2 t I ) , then the evolution of conditional probability q t ( X t | x T ) is given by the SDE:

<!-- formula-not-decoded -->

where b ( t, x t , x T ) = ˙ α t ˆ x 0 + ˙ β t x T +(˙ γ t + ϵ t γ t )ˆ z t , ˆ x 0 = E [ x 0 | x t , x T ] , ˆ z t =: ( x t -α t ˆ x 0 -β t x T ) /γ t . Besides, ˆ x 0 can be approximated by neural networks ˆ x θ 0 by minimizing a regression objective with the observed x 0 , x T as targets,

<!-- formula-not-decoded -->

where E denotes an expectation over ( x 0 , x T ) ∼ π ( x 0 , x T ) and x t ∼ p t ( x t | x 0 , x T ) .

Relation to Stochastic Interpolants (SI) . Both SI and ECSI in Prop. 4.1 can be seen as special cases of Conditioned SI. A key advantage of ECSI is its efficiency: while SI need to estimate two terms: E [ x 0 | x 0 ] and E [ x 1 | x t ] , ECSI only estimate E [ x 0 | x t , x 1 ] . A detailed comparison was demonstrated in App. B.

For training, we found that we could define an expanded space of bridge paths in terms of α t , β t , γ t , where γ t apparently controlled the stochasticity of the path. For sampling, we see from the proposition above that the sampling design space is expanded even further, as the sampling dynamics depend on α t , β t , γ t and ϵ t , where ϵ t appears as an additional degree of freedom to control stochasticity.

Training. Eq. (11) provides the training objective of the denoiser ˆ x θ 0 ( t, x t , x T ) . In the implementation, we include additional preconditioning as DDBM [18] and DBIM [42], see App. G for more details.

Sampling. We can generate samples from the conditional distribution q 0 | T ( x 0 | x T ) by solving the stochastic differential equation in Eq. (10) from t = T to t = 0 .

## 4.2 Existing samplers are a strict subset of ECSI samplers

We now show that existing samplers implement a strict subset of the ECSI samplers, see Figure 1.

DDBMsampler. When ϵ t = 0 , Eq. (10) reduces to a deterministic ODE. Setting ϵ t = γ t ˙ γ t -˙ α t α t γ 2 t recovers the sampling SDE used in DDBM [18]. However, DDBM only provides a single reverse SDE and a single corresponding reverse ODE; it does not explore alternative choices of ϵ t .

DBIMsampler. For small enough ∆ t and γ 2 t -∆ t -2 ϵ t ∆ t &gt; 0 , the sampling SDE can be discretized as:

<!-- formula-not-decoded -->

where ¯ z t ∼ N (0 , I ) , ˜ z = √ γ 2 t -∆ t -2 ϵ t ∆ t ˆ z t + √ 2 ϵ t ∆ t ¯ z t . Eq. (12) recover the DBIM sampler. Note that the condition γ 2 t -∆ t -2 ϵ t ∆ t &gt; 0 limits the design space of samplers. For example, our best result in the experiments is achieved by setting α t = 1 -t , γ t = γ 2 max 4 t (1 -t ) and ϵ t = γ t ˙ γ t -˙ α t α t γ 2 t , DBIM sampler fails under this setting since γ 2 t -∆ t -2 ϵ t ∆ t &gt; 0 cannot be guaranteed all the time.

I 2 SB sampler. When 2 ϵ t ∆ t = γ 2 t -∆ t -β 2 t -∆ t γ 2 t /β 2 t , the coefficient of x T in Eq. (12) vanishes. This special case corresponds to the Markovian bridge introduced in [42], and notably allows us to recover the sampling procedure of I2SB [21]. We provide a detailed derivation of this connection in Appendix D. The design space of the I 2 SB sampler is also limited, as it can be interpreted as a special case of the DBIM sampler.

Endpoint-Conditioned Stochastic Interpolants (Prop. 4.1) identify a class of sampling SDEs that share the same marginal distribution, but offer greater flexibility and a broader design space for sampler construction compared to DDBM, DBIM, and I2SB.

## 4.3 Our implementation

Our sampler based on Euler's discretization of the sampling SDE in Eq. (10):

<!-- formula-not-decoded -->

We set ϵ t = η ( γ t ˙ γ t -˙ α t α t γ 2 t ) , where η ∈ (0 , 1) is an interpolation parameter. This formulation provides continuous control over the sampling process, ranging from purely deterministic ODE sampling ( η = 0 ) to fully stochastic SDE sampling ( η = 1 ). In our implementation, we let ϵ t = 0 for the last two steps, Eq. (12) gets reduced to: x t -∆ t ≈ α t -∆ t ˆ x 0 + β t -∆ t x T + γ t -∆ t ˆ z t . For other steps, we apply Eq. (13) and let ϵ t = η ( γ t ˙ γ t -˙ α t α t γ 2 t ) , where η is a constant. Putting all ingredients together leads to our sampler outlined in Algorithm 1.

<!-- image -->

(a) Bridge paths in state space

(b) Bridge paths in density space

Figure 3: Modifying the base distribution corresponds to a lossy compression of the input that leads to a 'trade-off' between unconditional diffusion and diffusion bridge models.

## 5 Is there any benefit to modifying the starting point of a bridge?

We expanded the paths in distribution space connecting a base and target distribution, but so far left the endpoints fixed. While the target distribution should remain fixed, we could, in principle, modify the base distribution. At first glance this seems counter-intuitive - because of the data processing inequality we can only lose information about the target by modifying the base distribution. Hence, this angle has not been explored in the bridge literature. However, we found a surprising result modifying the base distribution can help significantly. The situation is analogous to the benefits of lossy compression in VAEs [6]. Information in the base distribution is not necessarily helpful, so by modifying the base distribution (which destroys some information) the model can align better with natural factors of variation.

## 5.1 Low conditional diversity in one-to-many translations

In our experiments (see Sec. 6), we observe that existing diffusion bridge models tend to produce low-diversity outputs under fixed conditioning. For instance, when generating handbags from a single edge map, the model is expected to produce varied outputs in terms of color, texture, and fine details. However, we find that current bridge models generate visually similar images across different sampling runs, despite the injection of different noise realizations during the diffusion process.

To address the issue of low output diversity, we propose modifying the base distribution used in the bridge model. Prior works [18, 1] typically treat the base distribution π T as equivalent to the input data distribution, denoted π cond . In contrast, our approach introduces a controlled perturbation by redefining the base distribution as π T = π cond ∗ N (0 , b 2 I ) , where b is a constant that governs the magnitude of noise added to the input distribution. This modification enables greater diversity in the generated outputs while maintaining conditional alignment.

Intuitively, this modification can be interpreted as a trade-off between standard diffusion models and traditional diffusion bridge models. As illustrated in Fig. 3, diffusion models typically generate samples starting from pure Gaussian noise, while diffusion bridge models begin sampling from fully conditioned inputs, such as edge maps. Our approach introduces an intermediate regime by

Table 2: Validation of our sampler via DDBM pretrained VP model (Evaluated by FID), where ϵ t = 0 . 3( γ t ˙ γ t -˙ α t α t γ 2 t ) .

|             | Edges → Handbags ( 64 × 64 )   | Edges → Handbags ( 64 × 64 )   | Edges → Handbags ( 64 × 64 )   | DIODE-Outdoor ( 256 × 256 )   | DIODE-Outdoor ( 256 × 256 )   | DIODE-Outdoor ( 256 × 256 )   |
|-------------|--------------------------------|--------------------------------|--------------------------------|-------------------------------|-------------------------------|-------------------------------|
| Sampler     | NFE=5                          | NFE=10                         | NFE=20                         | NFE=5                         | NFE=10                        | NFE=20                        |
| DDBM [18]   | 317.22                         | 137.15                         | 46.74                          | 328.33                        | 151.93                        | 41.03                         |
| DBIM [42]   | 3.60                           | 2.46                           | 1.74                           | 14.25                         | 7.98                          | 4.99                          |
| ECSI (Ours) | 2.36                           | 2.25                           | 1.53                           | 10.87                         | 6.83                          | 4.12                          |

sampling from noisy conditioned inputs, thereby blending the benefits of both paradigms-preserving conditional guidance while enhancing output diversity.

Modifying the base distribution with lossy compression can significantly improve the conditional diversity of the generated images.

## 5.2 How to measure the conditional diversity?

While existing metrics like FID implicitly capture the unconditional diversity of generated images, we need to capture the diversity of outputs (e.g. color images) for a single input image (a black and white edge map). To measure the conditional diversity, we will adopt Vendi Score (VS) [11] as a metric. Besides, We propose the Average Feature Distance (AFD) metric to quantify the conditional diversity among generated images. Initially, we select a group of source images { x ( i ) T } i M =1 . For each x ( i ) T , we then generate L distinct target samples. The j -th generated sample corresponding to the i -th source image is denoted by y ij . Then the AFD is calculated as follows:

̸

<!-- formula-not-decoded -->

where F ( · ) is a function that extracts the features of images, and ∥ · ∥ represents Euclidean norm. Intuitively, a larger AFD indicates the better conditional diversity. Here, F ( x ) can be x to evaluate the diversity directly in the pixel space. Alternatively, F ( · ) can be defined using the Inception-V3 model to assess the diversity in the latent space. In our experiments, we use AFD in latent space. Furthermore, we provide additional justification for the validity of our proposed metric in App. A.

A comparison between AFD and VS . Both AFD and the VS quantify diversity in the feature space of images, using features extracted from the Inception-V3 model. AFD measures the average pairwise Euclidean distance between feature vectors, making it sensitive to outliers. In contrast, the Vendi Score evaluates diversity by computing the effective number of unique feature patterns, based on the eigenvalues of the similarity matrix, emphasizing the overall structural diversity of the feature set. These metrics are complementary, capturing different aspects of diversity.

## 6 Experiments

In this section, we demonstrate how greatly expanding the space of bridge paths with ECSI leads to significantly improved performance for I2I translation tasks, in terms of sample efficiency, image quality and conditional diversity. We evaluate on I2I translation tasks on Edges → Handbags [16] scaled to 64 × 64 pixels and DIODE-Outdoor scaled to 256 × 256 [37], and Deblurring on ImageNet dataset [9]. For evaluation metrics, we use Fréchet Inception Distance (FID) [13] for all experiments, and additionally measure Inception Scores (IS) [3], Learned Perceptual Image Patch Similarity (LPIPS) [41], Mean Square Error (MSE), following previous works [42, 18]. In addition, we use VS and AFD, Eq. 14, to measure conditional diversity. Further details of the experiments and design guidelines are provided in Appendix G and E.

Sampler . We evaluate different sampling algorithms in Fig. 4 (a), the results demonstrate that setting ϵ t = 0 and using Eq. (12) for the last 2 steps can significantly improve sampled image

Figure 4: Ablation studies on discretization, γ max and ϵ t . (a). We evaluate different discretization schemes on Edges2handbags ( 64 × 64 ) dataset using DDBM-VP pretrained model, A represents simple Euler discretization in Eq. (13), B reprents setting ϵ t = 0 for the last 2 steps, C represents using Eq. (12) for ϵ t = 0 . (b). Ablation study on γ max evaluated by DIODE ( 64 × 64 ) dataset. (c). Ablation study on ϵ t through our ECSI model with Linear path on Edges2handbags ( 64 × 64 ) dataset, where ϵ t = η ( γ t ˙ γ t -˙ α t α t γ 2 t ) .

<!-- image -->

Table 3: Quantitative results in the I2I translation task Edges2handbags ( 64 × 64 ) and DIODE ( 256 × 256 ) datasets. Our results were achieved by Linear transition kernel and setting η = 1 .

|                          |        | Edges → handbags ( 64 × 64 )   | Edges → handbags ( 64 × 64 )   | Edges → handbags ( 64 × 64 )   | Edges → handbags ( 64 × 64 )   | DIODE-Outdoor ( 256 × 256 )   | DIODE-Outdoor ( 256 × 256 )   | DIODE-Outdoor ( 256 × 256 )   | DIODE-Outdoor ( 256 × 256 )   |
|--------------------------|--------|--------------------------------|--------------------------------|--------------------------------|--------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|
| Model                    | NFE    | FID ↓                          | IS ↑                           | LPIPS ↓                        | MSE                            | FID ↓                         | IS ↑                          | LPIPS ↓                       | MSE                           |
| Pix2Pix [16]             | 1      | 74.8                           | 3.24                           | 0.356                          | 0.209                          | 82.4                          | 4.22                          | 0.556                         | 0.133                         |
| DDIB [36]                | ≥ 40 † | 186.84                         | 2.04                           | 0.869                          | 1.05                           | 242.3                         | 4.22                          | 0.798                         | 0.794                         |
| SDEdit [25]              | ≥ 40   | 26.5                           | 3.58                           | 0.271                          | 0.510                          | 31.14                         | 5.70                          | 0.714                         | 0.534                         |
| Rectified Flow [22]      | ≥ 40   | 25.3                           | 2.80                           | 0.241                          | 0.088                          | 77.18                         | 5.87                          | 0.534                         | 0.157                         |
| I 2 SB [21]              | ≥ 40   | 7.43                           | 3.40                           | 0.244                          | 0.191                          | 9.34                          | 5.77                          | 0.373                         | 0.145                         |
| DDBM [18]                | 118    | 1.83                           | 3.73                           | 0.142                          | 0.040                          | 4.43                          | 6.21                          | 0.244                         | 0.084                         |
| DBIM [42]                | 20     | 1.74                           | 3.64                           | 0.095                          | 0.005                          | 4.99                          | 6.10                          | 0.201                         | 0.017                         |
| ECSI ( γ max = 0 . 125 ) | 5      | 0.89                           | 4.10                           | 0.049                          | 0.024                          | 12.97                         | 5.49                          | 0.269                         | 0.074                         |
|                          | 10     | 0.67                           | 4.11                           | 0.045                          | 0.024                          | 10.12                         | 5.56                          | 0.255                         | 0.076                         |
|                          | 20     | 0.56                           | 4.11                           | 0.044                          | 0.024                          | 8.62                          | 5.62                          | 0.248                         | 0.078                         |
|                          | 5      | 1.46                           | 4.21                           | 0.040                          | 0.016                          | 4.16                          | 5.83                          | 0.104                         | 0.029                         |
| ECSI ( γ max = 0 . 25 )  | 10     | 1.38                           | 4.22                           | 0.038                          | 0.017                          | 3.44                          | 5.86                          | 0.098                         | 0.029                         |
|                          | 20     | 1.40                           | 4.20                           | 0.038                          | 0.017                          | 3.27                          | 5.85                          | 0.094                         | 0.029                         |

quality compared with simple Euler discretization and DDBM sampler. Furtheremore, By specifically designing noise control during sampling, our sampler surpasses the sampling results by DDBM and DBIM with the same pretrained model. The results are demonstrated in Table 2. We set the number of function evaluations (NFEs) from the set [5 , 10 , 20] .

Bridge paths . We introduced an extensive bridge design space and begin by focusing on linear transition paths with different strength of maximum stochasticity, i.e., p t | 0 ,T ( x t | x 0 , x T ) = N ( x t ; (1 -t ) x 0 + tx T , 1 4 γ 2 max t (1 -t ) I ) . We conducted detailed ablation studies on γ max and η for the Linear path on DIODE ( 64 × 64 ) dataset, as shown in Fig. 4 (b) and (c). The optimal values for γ max were found to be 0 . 125 and 0 . 25 , while the best performance for η was achieved with η = 0 . 8 and η = 1 . 0 . Performance deteriorates when either parameter is too small or too large. Based on the results of these ablation studies, we further trained ECSI models on the Edges2handbags ( 64 × 64 ) and DIODE ( 256 × 256 ) datasets by taking γ max ∈ { 0 . 125 , 0 . 5 } and setting η = 1 . 0 . The results are presented in Table 3. Our models establish a new benchmark for image quality, as evaluated by FID, IS and LPIPS. Despite our models having slightly higher MSEs compared to the baseline DDBM and DBIM, we believe that a larger MSE indicates that the generated images are distinct from their references, suggesting a richer diversity.

Modifying base distribution . Through controlling noise in the base distribution, we achieved a more diverse set of sample images, while this diversity comes at the cost of slightly higher FID scores and slower sampling speed. We show generated images in Fig. 5. More visualization can be found in Appendix I, which shows that by introducing booting noise to the input data distribution, the model can generate samples with more diverse colors and textures. Further quantitative results are presented

DBIM

NFE=10,FID=2.46,AFD=5.20

DDBM

NFE=118,FID=1.83,AFD=6.99

<!-- image -->

Linear (b = 0.5)

NFE=10,FID=2.07,AFD=9.35

Figure 5: Visualization of conditional diversity via sampled images in a one-to-many translation task. While FID measures diversity within columns, AFD evaluates diversity across rows. The visualization further proved the effectiveness of AFD. More sampled images can be found in Appendix I.

Table 4: Quantitative results for Different denoisers and samplers on Edges2handbags ( 64 × 64 ). Our baseline is achieved by DDBM pretrained checkpoint and DBIM sampler.

| Method                       | FID ↓   | FID ↓   | FID ↓   | AFD ↑   | AFD ↑   | AFD ↑   | VS ↑   | VS ↑   | VS ↑   |
|------------------------------|---------|---------|---------|---------|---------|---------|--------|--------|--------|
|                              | NFE=5   | NFE=10  | NFE=20  | NFE=5   | NFE=10  | NFE=20  | NFE=5  | NFE=10 | NFE=20 |
| DDBM (pre) + DBIM sampler    | 3.60    | 2.46    | 1.74    | 5.63    | 5.20    | 5.84    | 1.16   | 1.23   | 1.26   |
| A: DDBM (pre) + ECSI sampler | 2.36    | 2.25    | 1.53    | 5.11    | 5.70    | 6.04    | 1.15   | 1.20   | 1.23   |
| B: ECSI (pre) + ECSI sampler | 0.89    | 0.67    | 0.56    | 6.00    | 6.05    | 6.25    | 1.22   | 1.25   | 1.28   |
| B + Modified base density    | 3.31    | 2.07    | 1.74    | 8.53    | 9.35    | 9.65    | 1.48   | 1.63   | 1.69   |

in Table 4, confirming that our model surpasses the vanilla DDBM in terms of image quality, sample efficiency, and conditional diversity.

Deblurring on ImageNet Dataset . We evaluate our models for Gaussian deblurring applying a Gaussian kernel with σ = 10 and Uniform deblurring, shown in Table 5. The results demonstrates that our ECSI models achieve much lower FID score.

## 7 Related Work

Diffusion Bridge Models . Diffusion bridges are faster diffusion processes that could learn the mapping between two random target distributions [39, 35], demonstrating significant potential in various areas, such as protein docking [32], mean-field game [20], I2I translation [21, 18]. According to different design philosophies, DBMs can be divided into two groups: bridge matching and stochastic interpolants. The idea of bridge matching was first proposed by Peluchetti et al. [28], and can be viewed as a generalization of score matching [34]. Based on this, diffusion Schrödinger bridge matching (DSBM) has been developed for solving Schrödinger bridge problems [35, 39]. In addition, Liu et al. [21] utilize bridge matching to perform image restoration tasks and noted benefits of noise empirically, the experiments shows the new model is more efficient and interpretable than score-based generative models [21]. Furthermore, our benchmark DDBM [18] achieve significant improvement for various I2I translation tasks, DBIM [42] improved the sampling algorithm for DDBM, significantly reducing sampling time while maintaining the same image quality.

Image-to-Image Translations . While diffusion models are strong at generating images, applying them to image-to-image (I2I) translation is more difficult due to artifacts in the output. DiffI2I improves quality and alignment with fewer diffusion steps [5]. In latent space, S2ST speeds up translation and reduces memory use [27]. Other methods improve guidance using features like frequency control [26, 15, 38]. A common challenge is that many models require joint training on both source and target domains, raising privacy concerns. Injecting-Diffusion tackles this by isolating shared content for unpaired translation [24]. SDDM improves interpretability by breaking down the score function across diffusion steps [30].

Table 5: Deblurring results with respect to different kernels, evaluated by FID on the 10k ImageNet ( 256 × 256 ) validation subset. Our results are achieved by 20 NFEs.

| Kernel   |   DDRM |   DDNM |   Pallette |   CDSB |   I 2 SB |   ECSI (ours) |
|----------|--------|--------|------------|--------|----------|---------------|
| Uniform  |    9.9 |    3   |        4.1 |   15.5 |      3.9 |          1.11 |
| Gaussian |    6.1 |    2.9 |        3.1 |    7.7 |      3   |          0.41 |

## 8 Conclusion

We introduced Endpoint-Conditioned Stochastic Interpolants (ECSI)-an improved version of stochastic interpolants that adds endpoint conditioning, modifies the base distribution, and uses discretization to explore the design space of Diffusion Bridge Models (DBMs). We highlighted a key issue often overlooked: one-to-many image translation tasks lack conditional diversity. Our findings show that resolving this requires adjusting the starting distribution, not the path or sampler. ECSI sets new benchmarks in image quality, sampling efficiency, and conditional diversity on tasks like 64 × 64 edges2handbags, 256 × 256 DIODE-outdoor, and ImageNet deblurring.

Limitations . (i) We note that optimal path design may vary by task, leaving room for future refinement. (ii) Incorporating guidance techniques may further enhance model performance.

## References

- [1] Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797 , 2023.
- [2] Michael S Albergo, Mark Goldstein, Nicholas M Boffi, Rajesh Ranganath, and Eric Vanden-Eijnden. Stochastic interpolants with data-dependent couplings. arXiv preprint arXiv:2310.03725 , 2023.
- [3] Shane Barratt and Rishi Sharma. A note on the inception score. arXiv preprint arXiv:1801.01973 , 2018.
- [4] David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin A Raffel. Mixmatch: A holistic approach to semi-supervised learning. Advances in neural information processing systems , 32, 2019.
- [5] Bin Xia, Yulun Zhang, Shiyin Wang, Yitong Wang, Xiaohong Wu, Yapeng Tian, Wenge Yang, Radu Timotfe, and Luc Van Gool. DiffI2I: Efficient Diffusion Model for Image-to-Image Translation. arXiv.org , 2023.
- [6] Christopher P. Burgess, Irina Higgins, Arka Pal, Loic Matthey, Nick Watters, Guillaume Desjardins, and Alexander Lerchner. Understanding disentangling in β -VAE. arXiv preprint arXiv:1804.03599 , 2018.
- [7] Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey E Hinton. Big self-supervised models are strong semi-supervised learners. Advances in neural information processing systems , 33:22243-22255, 2020.
- [8] Valentin De Bortoli, Guan-Horng Liu, Tianrong Chen, Evangelos A Theodorou, and Weilie Nie. Augmented bridge matching. arXiv preprint arXiv:2311.06978 , 2023.
- [9] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition , pages 248-255. Ieee, 2009.
- [10] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems , 34:8780-8794, 2021.
- [11] Dan Friedman and Adji Bousso Dieng. The vendi score: A diversity evaluation metric for machine learning. arXiv preprint arXiv:2210.02410 , 2022.

- [12] Nikita Gushchin, David Li, Daniil Selikhanovych, Evgeny Burnaev, Dmitry Baranchuk, and Alexander Korotin. Inverse bridge matching distillation. arXiv preprint arXiv:2502.01362 , 2025.
- [13] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems , 30, 2017.
- [14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems , 33:6840-6851, 2020.
- [15] Hyunsoo Lee, Minsoo Kang, and Bohyung Han. Conditional Score Guidance for Text-Driven Image-to-Image Translation. Neural Information Processing Systems , 2023.
- [16] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 1125-1134, 2017.
- [17] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems , 35:26565-26577, 2022.
- [18] Linqi Zhou, Aaron Lou, Samar Khanna, and Stefano Ermon. Denoising Diffusion Bridge Models. arXiv.org , 2023.
- [19] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 , 2022.
- [20] Guan-Horng Liu, Tianrong Chen, Oswin So, and Evangelos Theodorou. Deep generalized schrödinger bridge. Advances in Neural Information Processing Systems , 35:9374-9388, 2022.
- [21] Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A Theodorou, Weili Nie, and Anima Anandkumar. I 2 sb: Image-to-image schrödinger bridge. arXiv preprint arXiv:2302.05872 , 2023.
- [22] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003 , 2022.
- [23] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081 , 2024.
- [24] Luying Li and Lizhuang Ma. Injecting-Diffusion: Inject Domain-Independent Contents into Diffusion Models for Unpaired Image-to-Image Translation. IEEE International Conference on Multimedia and Expo , 2023.
- [25] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073 , 2021.
- [26] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation. Computer Vision and Pattern Recognition , 2023.
- [27] Or Greenberg, Eran Kishon, and Dani Lischinski. S2ST: Image-to-Image Translation in the Seed Space of Latent Diffusion. arXiv.org , 2023.
- [28] Stefano Peluchetti. Non-denoising forward-time diffusions. arXiv preprint arXiv:2312.14589 , 2023.
- [29] L Chris G Rogers and David Williams. Diffusions, Markov processes, and martingales: Itô calculus , volume 2. Cambridge university press, 2000.
- [30] Shurong Sun, Longhui Wei, Junliang Xing, Jia Jia, and Qi Tian. SDDM: Score-Decomposed Diffusion Models on Manifolds for Unpaired Image-to-Image Translation. International Conference on Machine Learning , 2023.

- [31] Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin A Raffel, Ekin Dogus Cubuk, Alexey Kurakin, and Chun-Liang Li. Fixmatch: Simplifying semisupervised learning with consistency and confidence. Advances in neural information processing systems , 33:596-608, 2020.
- [32] Vignesh Ram Somnath, Matteo Pariset, Ya-Ping Hsieh, Maria Rodriguez Martinez, Andreas Krause, and Charlotte Bunne. Aligned diffusion schrödinger bridges. In Uncertainty in Artificial Intelligence , pages 1985-1995. PMLR, 2023.
- [33] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems , 32, 2019.
- [34] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 , 2020.
- [35] Stefano Peluchetti. Diffusion Bridge Mixture Transports, Schr\"odinger Bridge Problems and Generative Modeling. arXiv.org , 2023.
- [36] Xuan Su, Jiaming Song, Chenlin Meng, and Stefano Ermon. Dual diffusion implicit bridges for image-to-image translation. arXiv preprint arXiv:2203.08382 , 2022.
- [37] Igor Vasiljevic, Nick Kolkin, Shanyi Zhang, Ruotian Luo, Haochen Wang, Falcon Z Dai, Andrea F Daniele, Mohammadreza Mostajabi, Steven Basart, Matthew R Walter, et al. Diode: A dense indoor and outdoor depth dataset. arXiv preprint arXiv:1908.00463 , 2019.
- [38] Xiang Gao, Zhengbo Xu, Junhan Zhao, and Jiaying Liu. Frequency-Controlled Diffusion Model for Versatile Text-Guided Image-to-Image Translation. AAAI Conference on Artificial Intelligence , 2024.
- [39] Yifeng Shi, Valentin De Bortoli, Andrew T. Campbell, and Arnaud Doucet. Diffusion Schr\"odinger Bridge Matching. Neural Information Processing Systems , 2023.
- [40] Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stéphane Deny. Barlow twins: Selfsupervised learning via redundancy reduction. In International conference on machine learning , pages 12310-12320. PMLR, 2021.
- [41] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 586-595, 2018.
- [42] Kaiwen Zheng, Guande He, Jianfei Chen, Fan Bao, and Jun Zhu. Diffusion bridge implicit models. arXiv preprint arXiv:2405.15885 , 2024.

Table 6: Evaluation for generative models: ImageNet-1-mode, ImageNet-2-modes, ImageNet-5modes, and ImageNet-10-modes.

| Model   |   ImageNet-1-mode |   ImageNet-2-modes |   ImageNet-5-modes |   ImageNet-10-modes |
|---------|-------------------|--------------------|--------------------|---------------------|
| FID     |              58.3 |              57.34 |              57.78 |               57.26 |
| AFD     |               0   |               8.14 |              12.84 |               14.47 |

## A AFD validation

In this section, we thoroughly validate the effectiveness of our proposed metric, AFD, for measuring conditional diversity and demonstrate its role as a complementary metric to FID. In unconditional generation scenarios, the FID is widely used to evaluate the diversity of generated images. While low FID scores generally indicate high diversity across the entire dataset, they do not necessarily imply high conditional diversity. For instance, we observed that samples generated by the DDBM model often lack diversity when conditioned on edge images, despite achieving very low FID scores. To address this limitation, we introduce the concept of conditional diversity and propose a corresponding metric to quantify it.

The first question is why FID failed to measure the conditional diversity. To illustrate the limitations of FID in capturing conditional diversity, consider an extreme case: if the images generated by a generative model are identical to a set of baseline images, the FID score can be very low since the two distributions are indistinguishable. However, this scenario does not reflect diversity within the conditional outputs.

To further support our point, we designed two classes of pseudo-generative models capable of controlling the diversity of the generated images, which are further validated by FID and AFD. The experiments are evaluated on Imagenet dataset [9].

## A.1 Pseudo-generative models by random selection

We designed four pseudo-generative models: ImageNet-1-mode, ImageNet-2-modes, ImageNet-5modes, and ImageNet-10-modes. The experimental setup is as follows:

- We selected 11,000 samples from the ImageNet validation dataset, randomly choosing 11 images per class.
- From these, we designated 1,000 images as the "real" set, while the remaining images served as the source pool for the generative models.
- Each ImageNet-k-modes model simulates a generative process by randomly sampling images from a pool of k distinct images within a given class.

We present sampled images in Fig. 6, where it is evident that the ImageNet-10-modes model generates images with the highest conditional diversity. To quantify this, we conducted experiments to calculate both FID and AFD for the four generative models. The results are summarized in Table 6. While the FID scores are nearly identical across all models, the AFD values increase as the conditional diversity of the generative models improves. This highlights that AFD is a more effective metric for capturing conditional diversity than FID.

## A.2 Pseudo-generative models by strong augmentation

Strong augmentation has been widely used in computer vision to generate synthetic data while preserving its underlying semantics [7, 40, 31, 4]. The intensity of augmentation can be adjusted, with higher intensities producing more diverse images. To further validate our proposed metric, AFD, as a measure of diversity, we construct pseudo-generative models using strong augmentation.

We selected 1,000 images from the ImageNet-1k dataset, one from each category. These images were subjected to data augmentation, specifically using ColorJitter, with varying magnitudes to enhance diversity. For each image, the augmentation was applied 16 times, creating an augmented dataset for

ImageNet-1-mode:FID=58.30,AFD=0

<!-- image -->

ImageNet-2-modes: FID=57.34, AFD=8.14

<!-- image -->

ImageNet-5-modes: FID=57.78,AFD=12.84

ImageNet-10-modes: FID=57.26, AFD=14.47

<!-- image -->

Figure 6: Sampled images from 4 generative models: ImageNet-1-mode, ImageNet-2-modes, ImageNet-5-modes, ImageNet-10-modes.

<!-- image -->

each magnitude setting. We then calculated the AFD for these augmented datasets to evaluate the relationship between dataset diversity (as influenced by augmentation magnitude) and the AFD value.

Table 7 summarizes the AFD results across various augmentation magnitude settings. The results show that as diversity increases, AFD values also rise, further confirming that the proposed AFD metric is a reliable indicator of image diversity.

## B Relation to Stochastic Interpolants

Conditioned Stochastic Interpolants build a marginal probability path p t | y using a mixture of interpolating densities: p t | y ( x ) = ∫ p t ( x t | x 0 , x 1 ) π ( x 0 , x 1 | y ) dx 0 dx 1 , where π ( x 0 , x 1 | y ) is a joint distribution with marginals π 0 | y ( x 0 | y ) and π 1 | y ( x 1 | y ) . For linear interpolants given by: X t = α t X 0 + β t X 1 + γ t z . The conditional kernel p t ( x t | x 0 , x 1 ) is given by a Gaussian distribution:

Table 7: AFD results across different augmentation magnitudes

| Augmentation magnitude   |   0.1 |   0.2 |   0.3 |   0.4 |   0.5 |   0.6 |   0.7 |   0.8 |
|--------------------------|-------|-------|-------|-------|-------|-------|-------|-------|
| AFD                      |  2.16 |  3.77 |  5.13 |  6.16 |  6.98 |  7.63 |  8.22 |  9.01 |
| FID                      |  0.2  |  2.95 |  7.02 | 11.62 | 16.33 | 20.84 | 25.12 | 28.89 |

p t ( x t | x 0 , x 1 ) = N ( α t x 0 + β t x 1 , σ 2 t I ) , ∀ t ∈ [0 , 1] . Then we can sample from the conditional distribution p 0 | y ( x 0 | y ) by running a stochastic process p t | y ( x t | y ) from time t = 1 to t = 0 , which is given by the following SDE:

<!-- formula-not-decoded -->

where the drift term b ( t, x, y ) is:

<!-- formula-not-decoded -->

As y represents null conditioning, Eq. (15) recover the original sampler of Stochastic Interpolants. In the drift term, E [ x 0 | x, y ] , E [ x 1 | x, y ] and E [ z | x, y ] are unknown, but we only need to estimate two of them, since

<!-- formula-not-decoded -->

We can further reduce the number of unknown term by endpoint-conditioning. Here as we replace condition y to be endpoint x 1 , the term E [ x 1 | x, x 1 ] = x 1 . So we have:

<!-- formula-not-decoded -->

This is exactly the sampler for ECSI in Prop. 4.1. Therefore, both SI and ECSI can be seen as special cases of Conditioned SI. A key advantage of ECSI is its efficiency: while SI need to estimate two terms: E [ x 0 | x t ] and E [ x 1 | x t ] , ECSI only estimates E [ x 0 | x t , y ] .

## C Proofs

There are infinitely many pinned processes characterized by the Gaussian transition kernel p t | 0 ,T ( x t | x 0 , x T ) = N ( x t ; α t x 0 + β t x T , γ 2 t I ) . Specifically, we formalize the pinned process as a linear Itô SDE, as presented in Lemma C.1.

Lemma C.1. There exist a linear Itô SDE

<!-- formula-not-decoded -->

where f t = ˙ α t α t , s t = ˙ β t -˙ α t α t β t , g t = √ 2( γ t ˙ γ t -˙ α t α t γ 2 t ) , that has a Gaussian marginal distribution N ( x t ; α t x 0 + β t x T , γ 2 t I ) .

Proof. Let m t denote the mean function of the given Itô SDE, then we have d m t dt = f t m t + s t x T . Given the transition kernel, the mean function m t = α t x 0 + β t x T , therefore,

<!-- formula-not-decoded -->

Matching the above equation:

<!-- formula-not-decoded -->

Further, For the variance γ 2 t of the process, the dynamics are given by:

Solving for g 2 t , we substitute f t = ˙ α t α t :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Given the pinned process (18), we can sample from the conditional distribution p 0 | T ( x 0 | x T ) by solving the reverse SDE or ODE from t = T to t = 0 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the score ∇ X t log p t ( X t | x T ) can be estimated by score matching objective (5).

For dynamics described by ODE d X t = u t dt , we can identify the entire class of SDEs that maintain the same marginal distributions, as detailed in Lemma C.2. This enables us to control the noise during sampling by appropriately designing ϵ t .

Lemma C.2. Consider a continuous dynamics given by ODE of the form: d X t = u t dt , with the density evolution p t ( X t ) . Then there exists forward SDEs and backward SDEs that match the marginal distribution p t . The forward SDEs are given by: d X t = ( u t + ϵ t ∇ log p t ) dt + √ 2 ϵ t d W t , ϵ t &gt; 0 . The backward SDEs are given by: d X t = ( u t -ϵ t ∇ log p t ) dt + √ 2 ϵ t d W t , ϵ t &gt; 0 .

Proof. For the forward SDEs, the Fokker-Planck equations are given by:

<!-- formula-not-decoded -->

This is exactly the Fokker-Planck equation for the original deterministic ODE d X t = u t dt . Therefore, the forward SDE maintains the same marginal distribution p t ( X t ) as the original ODE.

Now consider the backward SDEs, the Fokker-Planck equations become:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This is again the Fokker-Planck equation corresponding to the original deterministic ODE d X t = u t dt . Therefore, the backward SDE also maintains the same marginal distribution p t ( X t ) .

Therefore,

<!-- formula-not-decoded -->

Lemma C.3. Let ( x 0 , x T ) ∼ π 0 ( x 0 , x T ) , x t ∼ p t ( x | x 0 , x T ) , Given the transition kernel: p ( x t | x 0 , x T ) = N ( x t ; α t x 0 + β t x T , γ 2 t I ) , if ˆ x 0 ( x t , x T , t ) is a denoiser function that minimizes the expected L 2 denoising error for samples drawn from π 0 ( x 0 , x T ) :

<!-- formula-not-decoded -->

then the score has the following relationship with ˆ x 0 ( x t , x T , t ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we can minimize L ( D ) by minimizing L ( D ; x t , x T ) independently for each { x t , x T } pair.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus we conclude the proof.

Proof of Prop. 4.1.

Proof.

Table 8: Specify design choices for different model families. In the implementation, σ t = t for EDM, σ t = t, a t = 1 for DDBM-VE, σ t = √ e 1 2 β d t 2 + β min t -1 and a t = 1 / √ e 1 2 β d t 2 + β min t for DDBM-VP, where β d and β min are parameters. We include details and proofs in Appendix D.

̸

|                                 | I2SB                                                    | DDBM                                                                                                                                                                  | DBIM                                              | EDM                         | Ours                                          |
|---------------------------------|---------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|-----------------------------|-----------------------------------------------|
| Transition kernel α t β t γ 2 t | 1 - σ 2 t /σ 2 T σ 2 t /σ 2 T σ 2 t (1 - σ 2 t /σ 2 T ) | a t (1 - a 2 T σ 2 t / ( σ 2 t a 2 t )) a t (1 - a 2 T σ 2 t / ( σ 2 t a T σ 2 t / ( σ 2 t a t ) a T σ 2 t / ( σ 2 t a t σ 2 t (1 - a 2 T σ 2 t / ( σ 2 t a 2 t 2 2 2 | a 2 t )) ) )) σ t (1 - a T σ t / ( σ 2 t a 2 t )) | 1 0 σ 2 t                   | 1 - t t γ 2 max 4 t (1 - t )                  |
| Sampling SDEs ϵ t               | γ 2 t - ∆ t β 2 t - β 2 t - ∆ t γ 2 t 2 β 2 t ∆ t       | η ( γ t ˙ γ t - ˙ α t α t γ 2 t ) η = 0 or η = 1                                                                                                                      | { γ 2 t - ∆ t 2∆ t , t = 0 0 , t = 0              | ¯ β t σ 2 t -               | η ( γ t ˙ γ t - ˙ α t α t γ 2 t ) η ∈ [0 , 1] |
| Base distribution π T           | π cond                                                  | π cond                                                                                                                                                                | π cond                                            | π cond π cond ∗N (0 , b 2 I | )                                             |
| Discretization -                | Euler Eq. (12)                                          | Euler Eq. (13)                                                                                                                                                        | Euler Eq. (12)                                    | Heun -                      | Euler Eqs. (13) and (12)                      |

Proof. Recall Eqs. (24) (25) and Lemma C.2,

<!-- formula-not-decoded -->

Next we take the reparameterized score in Eq. (34) into Eq. (48):

<!-- formula-not-decoded -->

## D Reframing previous methods in our framework

We draw a link between our framework and the diffusion bridge models used in DDBM.

## D.1 DDBM-VE

DDBM-VE can be reformulated in our framework as we set :

<!-- formula-not-decoded -->

Proof. In the origin DDBM paper, the evolution of conditional probability q ( x t | x T ) has a time reversed SDE of the form:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and an associated probability flow ODE

<!-- formula-not-decoded -->

Compare Eqs. (54) and 55 with Lemma C.1. We only need to prove:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the original paper,

Therefore,

<!-- formula-not-decoded -->

In our framework, f t , s t , g 2 t can be calculated:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

which matches the formulation in DDBM.

## D.2 DDBM-VP

DDBM-VP can be reformulated in our framework as we set :

<!-- formula-not-decoded -->

Proof. In the original DDBM-VP setting,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

In our framework, f t , s t , g 2 t can be calculated:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which matches the formulation in DDBM.

## D.3 EDM

ODE formulation . The ODE formulation in EDM can be formlated in our framework as we set α t = 1 , β t = 0 , γ t = σ t .

Proof. Recall 25, the ODE formulation is given by:

<!-- formula-not-decoded -->

Therefore,

where f t = ˙ α t α t , s t = ˙ β t -˙ α t α t β t , g t = √ 2( γ t ˙ γ t -˙ α t α t γ 2 t ) . As α t = 1 , β t = 0 , γ t = σ t , The sampling ODE is given by:

<!-- formula-not-decoded -->

Sampling SDEs with noise added . Recall Proposition 4.1, as α t = 1 , β t = 0 , γ t = σ t , then the SDE has the form:

<!-- formula-not-decoded -->

Now we recover the stochastic sampling SDE in original EDM paper.

## D.4 I2SB

I2SB can be reformulated in our framework as we let:

<!-- formula-not-decoded -->

where σ 2 t := ∫ t 0 β τ dτ .

When 2 ϵ t ∆ t = γ 2 t -∆ t -β 2 t -∆ t γ 2 t /β 2 t , the coefficient of x T in Eq. (12) vanishes. Thus, Eq. (12) can be simplified as:

<!-- formula-not-decoded -->

Using discretization in Eq. (85):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the I2SB paper, define a 2 n := ∫ t n +1 t n β τ d τ , σ 2 n := ∫ t n 0 β τ d τ . Therefore,

<!-- formula-not-decoded -->

Thus, we reproduce the sampler of I2SB.

Figure 7: Ablation study on the shape of γ t .

<!-- image -->

Figure 8: Sampling paths with dfferent choices of γ t . As γ t extreamly low, e.g, γ max = 0 . 025 , the model will be failed to construct details of images.

<!-- image -->

## E Additional design guideline

α t and β t . Theoretically, α t and β t can be freely designed, and future work may explore alternative design choices. However, in this paper, we focus on the simple case where α t = 1 -t and β t = t . The rationale is as follows: consider the scenario where α t = 1 -β t , which represents an interpolation along the line segment between x 0 and x 1 . For the path p (1) t ( x ) = N ((1 -β t ) x 0 + β t x 1 , γ 2 t I ) , where β t is invertible, it is straightforward to construct another path p (2) t ( x ) = N ((1 -t ) x 0 + tx 1 , γ 2 β -1 t I ) , which achieves the same objective function but uses a different distribution of t during training. Based on this equivalence, setting α t = 1 -t and β t = t is a reasonable choice.

The shape of γ t . We conducted an ablation study on γ t with different shapes. Specifically, we assumed γ t has the form γ t = 2 γ max √ t k (1 -t k ) , as shown in Fig. 7, γ t will have different shape as we set different k . The results indicate that the best performance is achieved when k = 1 , which is the exact setting used in this paper.

γ max . Our ablation studies on γ max demonstrate that the optimal values of γ max are approximately 0 . 125 or 0 . 25 . Furthermore, the sampling paths corresponding to different choices of γ t are shown in Fig. 8. Adding an appropriate amount of noise to the transition kernel helps in constructing finer details.

ϵ t . We use the setting ϵ t = η ( γ t ˙ γ t -˙ α t α t γ 2 t ) . The ablation studies on ϵ t demonstrate that the optimal choice of η for the DDBM-VP model is approximately 0 . 3 , while the best choice for the ECSI model with a Linear Path is around 1 . 0 . Additionally, we present sample paths and generated images under different η settings to illustrate heuristic parameter tuning techniques. The results are shown in Figures 10, 11, and 12. Too small a value of η results in the loss of high-frequency information, while too large a value of η produces over-sharpened and potentially noisy sampled images.

(a) Yt.

Figure 9: An illustration of design choices of transition kernels and how they affect the I2I translation process. α t and β t define the interpolation between two images, while γ t controls the noise added to the process. ntuitively, the DDBM-VE model introduces excessive noise in the middle stages, which is unnecessary for effective image translation and may explain its poor performance. In contrast, our Linear path results in a symmetrical noise schedule, ensuring a more balanced process. On the other hand, the DDBM-VP path adds more noise near x T , , indicating that during training, more computational resources are focused around x 0 .

<!-- image -->

Figure 10: Sampling path with dfferent choices of ϵ t . As ϵ t = 0 , the generated images lack details, as ϵ t too large, the sampled images are over-sharpening. The best choices of ϵ t are around ϵ t = 0 . 8 and ϵ t = 1 . 0 .

<!-- image -->

Figure 11: Comparison of sampled images with different ϵ t for ECSI model, where ϵ t = η ( γ t ˙ γ t -˙ α t α t γ 2 t ) , γ max = 0 . 25 , b = 0 .

<!-- image -->

## F Impact Statement

Our method can improve image translation and solving inverse problem, which may benefit applications in medical imaging. However, it is important to note that as with many generative and restoration models, our method could be misused for malicious image manipulation.

Figure 12: Comparison of sampled images with different ϵ t for DDBM-VP pretrained model, where ϵ t = η ( γ t ˙ γ t -˙ α t α t γ 2 t ) .

<!-- image -->

## G Experiment Details

Architecture . We maintain the architecture and parameter settings consistent with [18], utilizing the ADM model [10] for 64 × 64 resolution, modifying the channel dimensions from 192 to 256 and reducing the number of residual blocks from three to two. Apart from these changes, all other settings remain identical to those used for 64 × 64 resolution.

Training . We include additional pre- and post-processing steps: scaling functions and loss weighting, the same ingredient as [17]. Let D θ ( x t , x T , t ) = c skip ( t ) x t + c out( t ) ( t ) F θ ( c in ( t ) x t , c noise ( t )) ,

where F θ is a neural network with parameter θ , the effective training target with respect to the raw network F θ is: E x t , x 0 , x T ,t [ λ ∥ c skip ( x t + c out F θ ( c in x t , c noise ) -x 0 ∥ 2 ] . Scaling scheme are chosen by requiring network inputs and training targets to have unit variance ( c in , c out ) , and amplifying errors in F θ as little as possible. Following reasoning in [18],

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where σ 2 0 , σ 2 T , and σ 0 T denote the variance of x 0 , variance of x T and the covariance of the two, respectively.

Wenote that TrigFlow [23], adopts the same score reparameterization and pre-conditioning techniques. It can be considered a special case of our framework by setting α t = cos( t ) , β t = 0 , γ t = σ 0 sin( t ) , t ∈ [0 , π 2 ] . In this case, σ T = 0 , σ 0 T = 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then we recover TrigFlow.

In our implementation, we set σ 0 = σ T = 0 . 5 , σ 0 T = σ 2 0 / 2 for all training sessions. Other setting are shown in Table 9.

Table 9: Training settings

| Model   | Dataset η γ max                                   | edges → handbags 0 0 . 125             | edges → handbags 0 0 . 25             | edges → handbags 0 . 5 0 . 125         |
|---------|---------------------------------------------------|----------------------------------------|---------------------------------------|----------------------------------------|
| Setting | GPU Batch size Learning rate epochs Training time | 1 A6000 48G 32 1 × 10 - 5 2078 42 days | 1 H100 96G 128 5 × 10 - 5 2106 8 days | 1 H100 96G 200 1 × 10 - 4 1443 11 days |
| Model   | Dataset η γ max                                   | DIODE ( 256 × 256 ) 0 0 . 125          | DOIDE ( 256 × 256 ) 0 0 . 25          |                                        |
| Setting | GPU Batch size Learning rate epochs Training time | 1 H100 96G 16 2 × 10 - 5 2617 17 days  | 1 H100 96G 16 2 × 10 - 5 1745 25 days |                                        |

Sampling . We use the same timesteps distributed according to EDM [17]: ( t 1 /ρ max + i N ( t 1 /ρ min -t 1 /ρ max )) ρ , where t min = 0 . 001 and t max = 1 -10 -4 . The best performance achieved by setting ρ = 0 . 6 for Edges2handbags and ρ = 0 . 8 for DIODE datasets.

## H Licenses

- Edges → Handbags [16]: BSD license.
- DIODE-Outdoor [37]: MIT license.

Figure 13: ECSI model and sampler ( γ max = 0 . 125 , η = 1 , b = 0 , NFE= 5 , FID= 0 . 89 ).

<!-- image -->

## I Additional visualizations

Figure 14: DDBM model and Our sampler (NFE=20, FID=1.53).

<!-- image -->

Figure 15: DDBM model and ECSI sampler ( η = 0 . 3 , NFE= 20 , FID= 4 . 12 ). Samples for DIODE dataset (conditoned on depth images).

<!-- image -->

Figure 16: ECSI model and sampler ( γ max = 0 . 25 , η = 1 . 0 , b = 0 , NFE=5, FID = 4.16).

<!-- image -->

Figure 17: ECSI model and sampler ( γ max = 0 . 25 , η = 1 . 0 , b = 0 , NFE=20, FID = 3.27).

<!-- image -->

Figure 18: DDBM model and DBIM sampler (NFE=10, FID = 2.46, AFD=5.20).

<!-- image -->

Figure 19: DDBM model and sampler (NFE=118, FID = 1.83, AFD=6.99).

<!-- image -->

Figure 20: ECSI model and sampler ( γ max = 0 . 125 , b = 1 . 0 , NFE=10, FID = 2.07, AFD=9.35).

<!-- image -->

Figure 21: DDBM model and ECSI sampler on 446 test images. (NFE=20, FID = 52.01, AFD=5.60).

<!-- image -->

Figure 22: ECSI model and sampler on 446 test images. ( γ max = 0 . 125 , b = 0 . 5 , NFE=20, FID = 55.93, AFD=7.39).

<!-- image -->

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (1-2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

## IMPORTANT, please:

- Delete this instruction block, but keep the section heading 'NeurIPS Paper Checklist" ,
- Keep the checklist subsection headings, questions/answers and guidelines below.
- Do not modify the questions and only use the provided macros for your answers .

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We ensure that the abstract and introduction clearly summarize the proposed method.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations in the Conclusion in Sec. 8. For example, we acknowledge that the optimal paths may vary from one scenario to another, indicating a rich avenue for further exploration and refinement in future work.

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

Justification: We include proofs in App. C.

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

Justification: We include detailed experiment information in Sec. 6 and App. G.

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

Justification: We include anonymous code access in the Abstract.

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

Justification: We include experiments in Sec. 6 and additional experiment details in App. G. Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: Not include in the current version. We acknowledge that formal error bars or statistical tests (e.g., t-tests) are not included in the current draft.

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

Justification: We include sufficient information on the computer resources in App. G.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have carefully reviewed the NeurIPS Code of Ethics and confirm that our research adheres to all its principles.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discuss both the potential positive and negative societal impacts in App. F. Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: We use standard, publicly available datasets (e.g., Edges2handbags, DIODE, Imagenet), and the risk of misuse is minimal. As such, no specific safeguards were deemed necessary.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We include Licenses section in App. H.

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

Justification: We introduce new code for our proposed method. We include access of an anonymous repository in the Abstract.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our work does not involve any human subjects, user studies, or crowdsourcing experiments. All experiments are conducted using synthetic or publicly available datasets without human participation.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our research does not involve human subjects, user studies, or crowdsourced participation. Therefore, no risks were incurred, and IRB approval was not required.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: Large language models (LLMs) were not used in any way that affects the core methodology, experiments, or results presented in this paper. Any assistance from tools such as ChatGPT was limited to minor writing edits or formatting suggestions, which do not influence the scientific contributions or conclusions of the work.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.