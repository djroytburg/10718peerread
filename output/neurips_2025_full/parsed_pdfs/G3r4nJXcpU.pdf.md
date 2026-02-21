## Proper Hölder-Kullback Dirichlet Diffusion: A Framework for High Dimensional Generative Modeling

Wanpeng Zhang 1 , Yuhao Fang 2 , Xihang Qiu 1 , Jiarong Cheng 1 , Jialong Hong 1 , Bin Zhai 1 , Qing Zhou 2, 3 , Yao Lu 2 , Ye Zhang 1, 2 ∗ , Chun Li 2 ∗

1 Beijing Institute of Technology, Beijing, 100081, China

2 Shenzhen MSU-BIT University, Shenzhen, 518172, China

Lomonosov Moscow State University, Moscow, 119991, Russia lichun2020@smbu.edu.cn https://github.com/TudoWay/PHKDif

## Abstract

Diffusion-based generative models have long depended on Gaussian priors, with little exploration of alternative distributions. We introduce a Proper Hölder-Kullback Dirichlet framework that uses time-varying multiplicative transformations to define both forward and reverse diffusion processes. Moving beyond conventional reweighted evidence lower bounds (ELBO) or Kullback-Leibler upper bounds (KLUB), we propose two novel divergence measures: the Proper Hölder Divergence (PHD) and the Proper Hölder-Kullback (PHK) divergence, the latter designed to restore symmetry missing in existing formulations. When optimizing our Dirichlet diffusion model with PHK, we achieve a Fréchet Inception Distance (FID) of 2.78 on unconditional CIFAR-10. Comprehensive experiments on natural-image datasets validate the generative strengths of model and confirm PHK's effectiveness in model training. These contributions expand the diffusion-model family with principled non-Gaussian processes and effective optimization tools, offering new avenues for versatile, high-fidelity generative modeling.

## 1 Introduction

Diffusion-based generative models are revolutionizing deep learning by simulating gradual noise addition and removal processes to transform simple priors into complex data distributions. From high-fidelity image synthesis and editing [1-5] to video generation [6, 7] and protein design [8-10]. As such, there is great interest in applying diffusion models and improving them further in terms of distribution quality, training cost, and image performance.

Recent efforts have investigated non-Gaussian diffusion processes, extending to categorical distribution [11-14], Poisson distribution [15], and Beta distribution [16]. However, Beta diffusion incurs limitation that it applies one-dimensional Beta tr ansitions independently along each coordinate, preventing it from capturing interdependencies among components in a high-dimensional simplex. Traditional divergence measures such as the Kullback-Leibler (KL) divergence suffer from asymmetry and computational complexity.

To address these limitations, in this work, we proposed Proper Hölder-Kullback Dirichlet Diffusion, which is a fundamentally novel framework and can more faithfully captures the simplex-structured target space.

∗ Corresponding author

3

Its main contributions are summarized as follows:

1. K -dimensional Dirichlet diffusion on simplex is established. Beta diffusion is generalized to a K -dimensional Dirichlet process that preserves the simplex structure at every noising step.
2. Strictly proper Hölder training objectives are proposed. Closed-form losses based on the Proper Hölder Divergence (PHD) and Proper Hölder-Kullback (PHK) are obtained, and analytic gradients under the Dirichlet exponential-family log-normalizer are guaranteed.
3. Variance-reduced gradient estimates are proven. It is shown that PHD yields lowervariance gradients compared to pseudo-divergence objectives, improving training stability and efficiency.

State-of-the-art non-Gaussian performance is demonstrated on unconditional CIFAR-10: an FID of 2.78 is achieved, outperforming all prior non-Gaussian diffusion methods and most Gaussian baselines, while PHK concomitantly reduces approximately a 9 % training cost compared with KLUB. Additional uncurated samples trained on CIFAR-10, AFHQ, FFHQ and CVUSA dataset exhibit both high fidelity and diversity. By breaking free from Gaussian noise and KL-centric optimization, Proper Hölder-Kullback Dirichlet Diffusion opens up a richer design space for generative modeling, and offering both rigorous theoretical guarantees and clear empirical gains.

## 2 Background

Diffusion-based deep generative models have shown remarkable potential for development. Both Langevin dynamics and variational inference interpretations of diffusion models fundamentally depend on the properties of Gaussian noise employed in training and sampling pipelines [17]. Initially inspired by diffusion models utilizing Gaussian noise, various Gaussian diffusion models have emerged [1, 18, 2, 19, 3]. These models use Gaussian Markov chains to progressively diffuse images into Gaussian noise during training. Conversely, the learned reverse diffusion process, defined by reversing the Gaussian Markov chain, iteratively refines the noisy inputs to generate clean, realistic images.

Gaussian diffusion models can be further categorized from perspectives such as score matching [20, 21, 18, 19] and stochastic differential equations [22]. These models have demonstrated notable successes across diverse tasks, including image generation and editing [4, 5, 23-26], 2D-to-3D conversion [27, 28], calibrated Bayesian inference [29], audio synthesis [30-32], conditional video generation [6, 7], reinforcement learning [33-35], few-shot learning [36], protein design [8-10], and cross-modal visual generation and editing [37, 38].

Typically, diffusion-based generative models are built following a standard procedure [1-3, 39], involving three core steps: (1) defining a forward diffusion process that progressively degrades data by introducing noise and reducing the signal-to-noise ratio (SNR) from time 0 to 1; (2) specifying a reverse diffusion process that denoises the data in reverse time, from 1 to 0; and (3) discretizing the time interval [0,1] into finite steps, with the forward and reverse processes serving as a fixed inference network and a learnable generator, respectively. Variational autoencoder inference [40, 41] is subsequently applied to optimize generator parameters by minimizing a weighted negative evidence lower bound (ELBO), comprising Kullback-Leibler (KL) divergence losses at each discretized reverse step.

In this work, we explore various effective distributions and divergences to investigate the multiple possibilities of the model. Our focus has progressively shifted towards diffusion models based on the Dirichlet distribution and Hölder divergence. Unlike traditional Gaussian diffusion, our approach is based on models employing non-traditional distributions and the deliberate incorporation of specifically tailored loss functions. Specifically, the adoption of Dirichlet distribution and Hölder divergence, which eschews the reliance on Gaussian distributions during both training and testing phases, enables us to explore alternative distributions and divergences, thereby unlocking new pathways for generative modeling. This innovative approach has the potential to engender models with distinct theoretical properties and to stimulate the emergence of innovative research trajectories.

## 3 Method

## 3.1 From Beta distribution to Dirichlet distribution

The Beta distribution is a continuous probability distribution defined on the interval [0 , 1] , commonly used to model probabilities or proportions. Its probability density function is given by [42]:

<!-- formula-not-decoded -->

where B( α, β ) is the Beta function, and α and β are shape parameters. Beta function can be expressed in terms of the Gamma function: B( α, β ) = Γ( α )Γ( β ) Γ( α + β ) .

For the Beta distribution, the mean and variance are:

<!-- formula-not-decoded -->

In Beta diffusion [16], for scalar data x 0 ∈ [0 , 1] , the forward diffusion process is defined as: z t | x 0 ∼ Beta( ηα t x 0 , η (1 -α t x 0 )) , where α t is a time-dependent parameter, and η is a scaling parameter. When t = T , z T approaches uniform noise. During training, the parameters of the reverse process are obtained by minimizing the Kullback-Leibler (KL) divergence to obtain a parameterized model f θ ( z t , t ) :

<!-- formula-not-decoded -->

The Dirichlet distribution is a high-dimensional generalization of the Beta distribution, used to model distributions on the simplex. Its probability density function is given by [43]:

<!-- formula-not-decoded -->

where S K denotes the K -dimensional simplex, and α i is the shape parameter for the i -th dimension. For the Dirichlet distribution, the mean and variance are [43]:

<!-- formula-not-decoded -->

It is worth noting that the Dirichlet function can also be expressed in terms of the Gamma function. Compared to the Beta function, it extends to K dimensions in the same form:

<!-- formula-not-decoded -->

Based on the above formulas and derivations, the one-dimensional diffusion in Beta diffusion can be extended to K -dimensional diffusion to reasonably implement the diffusion process on highdimensional simplices:

<!-- formula-not-decoded -->

where α t,i are time-dependent parameter functions. By scheduling α t,i , z 0 = x 0 can be nearly noise-free, while z T approaches a uniform distribution. The expected value is:

<!-- formula-not-decoded -->

Dirichlet distribution enables direct handling of high-dimensional data without decomposing the highdimensional problem into multiple one-dimensional problems. It offers significant advantages in modeling data distributions on high-dimensional simplices and naturally extends from low-dimensional to high-dimensional diffusion processes. This extension makes the Dirichlet distribution more flexible and efficient in dealing with high-dimensional data.

## 3.2 Training via Proper Hölder-Kullback Divergence

For the diffusion process of the Dirichlet distribution on high-dimensional simplices, to ensure that the objective function has a closed-form solution under the Dirichlet distribution, this study considers introducing an objective function based on the Hölder divergence.

Two objective functions that meet the criteria are available for further experimental validation: the Hölder Statistical Pseudo-Divergence (HPD) and the Proper Hölder Divergence (PHD) [44].

The objective function based on HPD is expressed as:

<!-- formula-not-decoded -->

For the Dirichlet distribution, F ( θ ) has a closed-form expression, where θ i = α i -1 . If

<!-- formula-not-decoded -->

let θ p,i = a p,i -1 , θ q,i = a q,i -1 , then D H α ( p : q ) can be computed analytically. The training objective is:

<!-- formula-not-decoded -->

The objective function based on PHD is expressed as:

<!-- formula-not-decoded -->

Similarly, a closed-form solution can be obtained. The training objective is:

<!-- formula-not-decoded -->

Theory and experiments indicate that PHD yields superior performance, an advantage associated with the strict propriety of PHD. A detailed explanation is provided in Appendix A.

Furthermore, recognizing the unique advantages of PHD, combined with the mature foundation of KL theory in the diffusion model domain, KLUB [16], this study introduces the Proper Hölder-Kullback Divergence (PHK), which also has a closed-form solution for the Dirichlet distribution. Given δ, ϵ ∈ [0 , 1] as weight coefficients, the objective function based on PHK is expressed as:

<!-- formula-not-decoded -->

We summarize the training and sampling algorithms in Algorithms 1 and 2, respectively.

## Algorithm 1 Training

Require: Dataset: D = { { X m n } M m =1 , y n } N n =1 , Mini-batch size B , concentration parameter η = 10000 , data shifting parameter S hift = 0 . 6 , data scaling parameter S cale = 0 . 39 , generator f θ , time reversal coefficient π = 0 . 95 , and a sigmoid schedule defined by α t = 1 / (1 + e -c 0 -( c 1 -c 0 ) t ) given t ∈ [0 , 1] , where c 0 = 10 and c 1 = 13

## 1: repeat

- 2: Draw a mini-batch X 0 = { x ( i ) 0 } B i =1 from D
- 3: for i = 1 to B do
- 4: t i ∼ Unif (1 e -5 , 1)
- 5: s i = πt i
- 6: Compute α s i and α t i
- 7: x ( i ) 0 = x ( i ) 0 ∗ S cale + S hift

<!-- formula-not-decoded -->

- 9: ˆ x ( i ) 0 = f θ ( z t i , t i ) ∗ S cale + S hift
- 10: compute the loss L i
- 11: end for
- 13: until converge
- 12: Perform SGD with 1 B ∇ θ ∑ B i =1 L i

▷ can be run in parallel

## Algorithm 2 Sampling

```
Require: Number of function evaluations (NFE) J = 200 , generator f θ , and timesteps { t j } J j =0 : t j = 1 -(1 -1 e -5) ∗ ( J -j ) / ( J -1) for j = 1 , . . . , J and t 0 = 0 1: if NFE > 350 then 2: α t j = 1 / (1 + e -c 0 -( c 1 -c 0 ) t j ) 3: else 4: α t j = (1 / (1 + e -c 1 )) t j 5: end if 6: Initialize ˆ x 0 = E [ x 0 ] ∗ S cale + S hift 7: z t J ∼ Dir ( ηα t J ˆ x 0 , . . . , η 1 K -1 (1 -α t J ˆ x 0 )) 8: for j = J to 1 do 9: ˆ x 0 = f θ ( z t j , α t j ) ∗ S cale + S hift 10: p ( t j -1 ← t j ) ∼ Dir ( η ( α t j -1 -α t j )ˆ x 0 , . . . , η 1 K -1 (1 -α t j -1 ˆ x 0 )) 11: z t j -1 = z t j +(1 -z t j ) p ( t j -1 ← t j ) 12: end for 13: return (ˆ x 0 -S hift ) /S cale or ( z t 0 /α t 0 -S hift ) /S cale
```

## 3.3 Forward and Reverse Diffusion

Figure 1: Demonstration of the forward diffusion process on five samples. the first column displays the original iamge, and the subsequent 21 columns illustrate progressively noising and masking images at time steps t = 0 , 0 . 05 , . . . , 1 .

<!-- image -->

This study conducts a analysis of the forward and reverse diffusion processes by incorporating process images. First, the Dirichlet forward diffusion process is visualized by displaying the real image x and its noise-corrupted version during the forward diffusion process. Specifically, the images with noise and masking at times t = 0 , 0 . 05 , 0 . 1 , . . . , 1 are represented as:

<!-- formula-not-decoded -->

As can be clearly seen from Figure 1, during the forward diffusion process, the image becomes increasingly noisy and sparse with the time, and is ultimately almost estroyed. It is evident that the forward process of Dirichlet diffusion involves simultaneous noising and masking of pixels. This is distinct from traditional Gaussian diffusion, as the forward diffusion process in Gaussian diffusion gradually applies additive random noise and eventually concludes with Gaussian random noise.

Similarly, the reverse diffusion process of traditional Gaussian diffusion is considered a denoising process, whereas the reverse diffusion process of Dirichlet diffusion involves simultaneous denoising and demasking of the data. The images representing denoising and demasking are expressed as:

<!-- formula-not-decoded -->

Figure 2: Demonstration of the reverse diffusion process on five examples. The first column displays the initial noisy, and the subsequent 21 columns illustrate the progressively denoising and demasking at time steps t = 1 , 0 . 9 , . . . , 0 .

<!-- image -->

where z t -1 = z t + (1 -z t ) p ( t j → t j -1 ) is iteratively computed according to Algorithm 2. Highresolution images require more computational resources. Final generated image is represented as:

<!-- formula-not-decoded -->

The generated image data will more accurately approximate the real iamge data when θ increasingly approaches its theoretical optimal value θ ∗ , such that f θ ∗ ( z t j , t j ) = E [ x 0 | z t j ] .

As shown in Figure 2, starting from random noise

<!-- formula-not-decoded -->

most of the pixel values will be entirely black. Dirichlet diffusion gradually denoises and restores the image to a clean state through multiplicative transformations, as demonstrated in Algorithm 2.

Figure 3: Comparison of generated samples from the Beta Diffusion model and Dirichlet Diffusion model trained on unconditional CIFAR-10 image dataset. (L) Beta Diffusion and (R) Dirichlet Diffusion. This figure presents side-by-side examples of images synthesized on the CIFAR-10 image dataset by two diffusion frameworks.

<!-- image -->

## 4 Experiments

## 4.1 Comparison Study

This study offers a comprehensive numerical comparison with diffusion models based on traditional Gaussian distributions and those based on non-Gaussian or quasi-Gaussian distributions that have

been publicly released in recent years at NeurIPS, ICLR, ICML, etc. Table 1 encompasses a wide range of diffusion models. The results demonstrate that the new diffusion model proposed in this study achieved a Fréchet Inception Distance (FID) [45] score of 2.78 on the unconditional CIFAR-10 dataset [46], surpassing all non-Gaussian-based diffusion models, including the cold diffusion model based on deterministic diffusion [17], the Inverse Heat Dispersion based on deterministic diffusion [47], the D3PM diffusion model based on categorical distribution [12], and the JUMP diffusion model based on Poisson distribution [15]. Compared with the family of Gaussian diffusion models, the new diffusion model in this study outperformed the majority of Gaussian-like models, including NCSNv2 [19], DDPM [2], DiffuEBM [48], VDM [3], Improved DDPM [49], TDPM+ [50], Soft Diffusion [51], Blurring Diffusion [52], etc. It thus serves as a highly competitive alternative when employing diffusion models. As a newly proposed diffusion model, constrained by limitations in computational resources and training time, this study has not yet been able to conduct a detailed investigation and global search of all model parameters, nor has it introduced targeted enhancement modules. As a novel perspective on diffusion models, this study holds broad potential for future research.More hyperparameter settings and training details of the experiments are provided in Appendix F and G.

Table 1: FID scores for various generative models trained on the CIFAR-10 dataset. Lower FID indicates better image fidelity and diversity.

| Distribution   | Model                          | Year       | FID ( ↓ )                            |
|----------------|--------------------------------|------------|--------------------------------------|
| Gaussian       | DDPM [2]                       | NeurIPS'20 | 3.17                                 |
| Gaussian       | NCSNv2 [19]                    | NeurIPS'20 | 10.87                                |
| Gaussian       | DiffuEBM [48]                  | ICML'21    | 9.58                                 |
| Gaussian       | VDM[3]                         | NeurIPS'21 | 4.00                                 |
| Gaussian       | Improved DDPM [49]             | ICML'21    | 2.94                                 |
|                | TDPM+ [50]                     | ICLR'23    | 2.83                                 |
|                | Soft Diffusion [51]            | TMLR'23    | 3.86                                 |
|                | Blurring Diffusion [52]        | ICLR'23    | 3.17                                 |
|                | Blackout Diffusion [53]        | ICML'23    | 4.58                                 |
|                | GET [54]                       | NeurIPS'24 | 5.49                                 |
|                | Diff-Instruct [55]             | NeurIPS'24 | 4.53                                 |
|                | UniPC with optimized step [56] | CVPR'24    | 3.13                                 |
|                | RDUOT [57]                     | ECCV'24    | 2.95                                 |
| Deterministic  | Cold Diffusion [17]            | NeurIPS'23 | 80.08 (deblurring) 8.92 (inpainting) |
| Deterministic  | Inverse Heat Dispersion [47]   | ICLR'23    | 18.96                                |
| Categorical    | D3PM [12]                      | NeurIPS'21 | 7.34                                 |
| Beta           | Beta Diffusion [16]            | NeurIPS'23 | 3.06                                 |
| Poisson        | JUMP [15]                      | ICML'23    | 4.80                                 |
| Dirichlet      | Ours                           | -          | 2.78                                 |

## 4.2 Ablation Study

In this study, we meticulously designed and conducted a series of ablation experiments to systematically validate the effectiveness of the proposed PHD and PHK based on Hölder divergence. Through these experiments, we investigated the performance and potential advantages of these two methods.

As shown in Table 2, the experimental results clearly reveal the superiority of PHD and PHK over traditional methods. Specifically, compared with the widely used -ELBO, PHD achieves better generation performance and demonstrates unique capabilities in image processing. Moreover, PHK, as a hybrid optimization method, can effectively integrate the advantages of different divergences. It processes image data more comprehensively than other divergences, thereby achieving optimal performance. This result further validates the effectiveness and superiority of our proposed methods, indicating that PHD and PHK hold broad research prospects and application potential in the field of image generation.

Table 2: Ablation study on CIFAR-10 varying the number of function evaluations (NFE) and mini-batch size B , identifying the optimal NFEB configuration that maximizes performance per computation and guides efficient resource allocation.

|   Loss B |   - ELBO 512 |   - ELBO 288 |   KLUB 512 |   KLUB 288 |   PHD 512 |   PHD 288 |   PHK 512 |   PHK 288 |
|----------|--------------|--------------|------------|------------|-----------|-----------|-----------|-----------|
|       20 |        15.81 |        16.02 |      16.62 |      16.35 |     16.84 |     16.95 |     15.92 |     16.53 |
|       50 |         6.53 |         6.65 |       6.16 |       6.2  |      6.48 |      6.93 |      5.87 |      6.15 |
|      200 |         4.49 |         4.68 |       3.46 |       3.55 |      3.85 |      4.43 |      3.2  |      3.29 |
|      500 |         4.33 |         4.42 |       3.39 |       3.46 |      3.72 |      4.23 |      2.92 |      2.99 |
|     1000 |         4.28 |         4.42 |       3.32 |       3.4  |      3.52 |      4.19 |      2.84 |      2.91 |
|     2000 |         4.31 |         4.39 |       3.25 |       3.21 |      3.51 |      4.21 |      2.82 |      2.81 |

We also include Figure 3 to visually compare generated images under Beta Diffusion and Dirichlet Diffusion on CIFAR-10, Figure 4 to visually compare generated images under Dirichlet Diffusion on FFHQ [58] and AFHQ [59]. Owing to the considerable time required for each training iteration and the computationally demanding nature of FID assessment, we have not yet explored the optimization of these hyperparameter combinations, given the constraints of our current computational resources. Consequently, although the results presented in this paper illustrate that Dirichlet diffusion can achieve competitive performance in image generation, they do not fully capture the potential capabilities of Dirichlet diffusion. Further enhancements to these results may be attainable through optimized hyperparameter configurations or network architectures specifically designed for Dirichlet diffusion. We defer these investigations to our future work.

<!-- image -->

(b) AFHQ

Figure 4: Generated images from the Dirichlet Diffusion model on the FFHQ and AFHQ datasets. (a) FFHQ samples.(b) AFHQ samples.These results demonstrate that Dirichlet Diffusion produces high-fidelity, diverse images, capturing both fine-grained human facial details and the varied textures and structures of animals.

<!-- image -->

## 5 Related Work and Future Directions

Various diffusion processes, including Gaussian diffusion, Beta diffusion, Poisson diffusion, and Dirichlet diffusion, employ specific distributions in both forward and reverse sampling. Gaussianbased diffusion models initiate their reverse process from the standard normal distribution N (0 , 1) , whereas Poisson, Beta, and Dirichlet diffusion all start from 0. The reverse sampling of Dirichlet diffusion is a monotonically non-decreasing process, similar to Poisson diffusion. However, while Poisson diffusion involves discrete jumps in count values, Dirichlet diffusion employs continuous jumps in probability values, consistent with Beta diffusion.

In recent years, several works have actively explored diffusion models based on non-traditional Gaussian distributions. Bansal et al. [17] proposed cold diffusion, which constructs models based on arbitrary image transformations rather than Gaussian noise. Rissanen et al. [47] introduced Inverse Heat Dispersion, which generates images by stochastically reversing the heat equation. Austin et al. [12] proposed the D3PM diffusion model based on the Categorical distribution. Chen et al. [15] proposed the JUMP diffusion model based on the Poisson distribution. In comparison, Dirichlet diffusion, which innovates in terms of noise distribution and training loss, has achieved a new state-of-the-art FID for non-Gaussian models on the unconditional CIFAR-10 dataset.

There have also been several works in the quasi-Gaussian diffusion domain in recent years. Hoogeboom and Salimans [60] proposed blurring diffusion, which shows that blurring can be equivalently defined using a Gaussian diffusion process with anisotropic noise and proposed incorporating blurring into Gaussian diffusion. Daras et al. [51] proposed soft diffusion, which employs linear erosion processes such as Gaussian blurring and masking. Rissanen et al. [47] proposed a diffusion process based on inverse heat diffusion, which reverses the heat equation using the inductive bias from Gaussian diffusion models. These diffusion processes share similarities with Gaussian diffusion in terms of loss definition and the use of Gaussian-based reverse diffusion for generation. In contrast, Dirichlet diffusion differs from them in terms of noise distribution and training loss and outperforms most quasi-Gaussian models on the unconditional CIFAR-10 dataset.

For a long time, the standard diffusion models have involved adding noise and reversing the degradation of image processing. Bansal et al. [17] observed that the generative behavior of diffusion models does not strongly depend on the choice of image degradation. Hoogeboom et al. [52] showed that blurring can equivalently be defined through a Gaussian diffusion process with anisotropic noise, serving as an alternative to isotropic Gaussian diffusion. This has sparked the potential consideration of Dirichlet+Blurring. By establishing this connection, it is hoped to bridge the gap between reverse heat diffusion and denoising diffusion, leading to a generalized diffusion model that can reverse any process and thereby effectively eliminate the dependence on a single type of noise.

The improvement of neural image compression and reconstruction is also a direction worth discussing. The general consensus is that replacing the decoder with a conditional diffusion model can enhance the perceptual quality of neural image compression and reconstruction. However, their lack of inductive bias for image data limits their ability to achieve state-of-the-art perceptual performance. Khoshkhahtinat et al. [61] proposed that employing an anisotropic diffusion model on the decoder side can disentangle frequency content, thereby facilitating the generation of high-quality images. Dirichlet diffusion, which has room for approximate improvement on the decoder side, provides the capability for seamless integration within the operational space of neural networks, thus further optimizing image generation quality.

## 6 Limitations and Conclusions

Existing diffusion models predominantly rely on traditional Gaussian distributions for both the forward and reverse processes. In this work, we demonstrate that it is possible to integrate the non-traditional Dirichlet distribution and the Hölder-based Divergence within the framework of diffusion models. Our experimental results confirm its superior performance in image generation tasks. with highly competitive quantitative metrics. as well as its unique qualities when applied to generative modeling of high-dimensional simplex data. Additionally, these results highlight the effectiveness of PHD and PHK in optimizing Dirichlet diffusion. This framework offers a broader and more in-depth space for the application of more diverse diffusion models that transcend the Gaussian noise paradigm, including but not limited to image generation.

Despite its effectiveness, the method has some limitations: 1. It is sensitive to the selection of hyperparameters; the existing experiments have not covered all possible combinations of hyperparameters, necessitating a broader range of experiments and validation. 2. The training cost is high; processing 200 million CIFAR-10 images using four Nvidia L40S GPUs takes approximately 64 hours with a batch size B = 512 , and a more substantial computational resource is required given the multitude of parameter quantities and extensive tuning scenarios.

## 7 Acknowledgments

This research was supported by Guangdong Basic and Applied Basic Research Foundation (No. 2024A1515011774), the National Key Research and Development Program of China (No. 2022YFC3310300), the National Natural Science Foundation of China (No. 12171036), Shenzhen Sci-Tech Fund (Grant No. RCJC20231211090030059).

## References

- [1] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015 , pages 2256-2265, 2015.
- [2] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual , 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html .
- [3] Diederik P. Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual , 34:21696-21707, 2021. URL https://proceedings.neurips.cc/paper\_files/paper/ 2021/file/b578f2a52a0229873fefc2a4b06377fa-Paper.pdf .
- [4] Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion Models Beat GANs on Image Synthesis. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual , pages 8780-8794, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ 49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html .
- [5] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded Diffusion Models for High Fidelity Image Generation. Journal of Machine Learning Research , 23:47-1, 2022.
- [6] Qi Tang, Yao Zhao, Meiqin Liu, and Chao Yao. Seeclear: Semantic distillation enhances pixel condensation for video super-resolution. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 -15, 2024 , 2024. URL http://papers.nips.cc/paper\_files/paper/2024/hash/ f358b2a880adf34939d2d6f926e54d2a-Abstract-Conference.html .
- [7] Giannis Daras, Weili Nie, Karsten Kreis, Alex Dimakis, Morteza Mardani, Nikola B. Kovachki, and Arash Vahdat. Warped diffusion: Solving video inverse problems with image diffusion models. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 15, 2024 , volume 37, pages 101116-101143, 2024. URL http://papers.nips.cc/paper\_files/paper/2024/hash/ b736c4b0b38876c9249db9bd900c1a86-Abstract-Conference.html .
- [8] Chence Shi, Shitong Luo, Minkai Xu, and Jian Tang. Learning gradient fields for molecular conformation generation. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event , volume 139 of Proceedings of Machine Learning Research , pages 9558-9568. PMLR, 2021. URL http://proceedings.mlr. press/v139/shi21b.html .
- [9] Shitong Luo, Yufeng Su, Xingang Peng, Sheng Wang, Jian Peng, and Jianzhu Ma. AntigenSpecific Antibody Design and Optimization with Diffusion-Based Generative Models for Protein Structures. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November

28 - December 9, 2022 , 2022. URL http://papers.nips.cc/paper\_files/paper/2022/ hash/3fa7d76a0dc1179f1e98d1bc62403756-Abstract-Conference.html .

- [10] Bowen Jing, Gabriele Corso, Jeffrey Chang, Regina Barzilay, and Tommi S. Jaakkola. Torsional Diffusion for Molecular Conformer Generation. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 -December 9, 2022 , 2022. URL http://papers.nips.cc/paper\_files/paper/2022/hash/ 994545b2308bbbbc97e3e687ea9e464f-Abstract-Conference.html .
- [11] Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual , pages 12454-12465, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ 67d96d458abdef21792e6d8e590244e7-Abstract.html .
- [12] Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual , pages 17981-17993, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ 958c530554f78bcd8e97125b70e6973d-Abstract.html .
- [13] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022 , pages 10686-10696. IEEE, 2022. doi: 10.1109/CVPR52688.2022.01043. URL https://doi.org/10.1109/CVPR52688.2022.01043 .
- [14] Minghui Hu, Yujie Wang, Tat-Jen Cham, Jianfei Yang, and Ponnuthurai N. Suganthan. Global context with discrete diffusion in vector quantised modelling for image generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022 , pages 11492-11501. IEEE, 2022. doi: 10.1109/CVPR52688.2022.01121. URL https://doi.org/10.1109/CVPR52688.2022.01121 .
- [15] Tianqi Chen and Mingyuan Zhou. Learning to Jump: Thinning and Thickening Latent Counts for Generative Modeling. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA , volume 202 of Proceedings of Machine Learning Research , pages 5367-5382. PMLR, 2023.
- [16] Mingyuan Zhou, Tianqi Chen, Zhendong Wang, and Huangjie Zheng. Beta Diffusion. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023 , 2023. URL http://papers.nips.cc/paper\_files/paper/2023/hash/ 5fe1b43c882d746c187456eb4c8cdf52-Abstract-Conference.html .
- [17] Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, New Orleans, LA, USA, December 10 - 16, 2023 , 2023. URL http://papers.nips.cc/paper\_files/paper/2023/hash/ 80fe51a7d8d0c73ff7439c2a2554ed53-Abstract-Conference.html .
- [18] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada , pages 11895-11907, 2019. URL https://proceedings.neurips.cc/paper/ 2019/hash/3001ef257407d5a371a96dcd947c7d93-Abstract.html .

- [19] Yang Song and Stefano Ermon. Improved techniques for training score-based generative models. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 612, 2020, virtual , 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 92c3b916311a5517d9290576e3ea37ad-Abstract.html .
- [20] Aapo Hyvärinen. Estimation of Non-Normalized Statistical Models by Score Matching. Journal of Machine Learning Research , 6(24):695-709, 2005. URL http://jmlr.org/papers/v6/ hyvarinen05a.html .
- [21] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural Computation , 23(7):1661-1674, 2011.
- [22] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 37, 2021 . OpenReview.net, 2021. URL https://openreview.net/forum?id=PxTIG12RRHS .
- [23] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125 , 2022.
- [24] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 1824, 2022 , pages 10674-10685. IEEE, 2022. doi: 10.1109/CVPR52688.2022.01042. URL https://doi.org/10.1109/CVPR52688.2022.01042 .
- [25] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L. Denton, Seyed Kamyar Seyed Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022 , 2022. URL http://papers.nips.cc/paper\_files/paper/2022/hash/ ec795aeadae0b7d230fa35cbaf04c041-Abstract-Conference.html .
- [26] Zhendong Wang, Yifan Jiang, Yadong Lu, Yelong Shen, Pengcheng He, Weizhu Chen, Zhangyang (Atlas) Wang, and Mingyuan Zhou. In-Context Learning Unlocked for Diffusion Models. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023 , 2023. URL http://papers.nips.cc/paper\_files/paper/2023/hash/ 1b3750390ca8b931fb9ca988647940cb-Abstract-Conference.html .
- [27] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. DreamFusion: Text-to-3D using 2D Diffusion. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023 . OpenReview.net, 2023. URL https://openreview. net/forum?id=FjNys5c7VyY .
- [28] Mohammadreza Armandpour, Huangjie Zheng, Ali Sadeghian, Amir Sadeghian, and Mingyuan Zhou. Re-imagine the negative prompt algorithm: Transform 2D diffusion into 3D, alleviate Janus problem and beyond. arXiv preprint arXiv:2304.04968 , 2023. URL https://perp-neg. github.io/ . (the first three authors contributed equally).
- [29] Daniela de Albuquerque and John M. Pearson. Inflationary Flows: Calibrated Bayesian Inference with Diffusion-Based Models. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 -15, 2024 , 2024. URL http://papers.nips.cc/paper\_files/paper/2024/hash/ af85ade5e70a6e1eb07a9541fb529baf-Abstract-Conference.html .
- [30] Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, and William Chan. WaveGrad: Estimating Gradients for Waveform Generation. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021 . OpenReview.net, 2021. URL https://openreview.net/forum?id=NsMLjcFaO8O .

- [31] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. DiffWave: A Versatile Diffusion Model for Audio Synthesis. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021 . OpenReview.net, 2021.
- [32] Dongchao Yang, Jianwei Yu, Helin Wang, Wen Wang, Chao Weng, Yuexian Zou, and Dong Yu. Diffsound: Discrete Diffusion Model for Text-to-Sound Generation. IEEE ACM Trans. Audio Speech Lang. Process. , 31:1720-1733, 2023. URL https://doi.org/10.1109/TASLP. 2023.3268730 .
- [33] Michael Janner, Yilun Du, Joshua B. Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. In International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA , volume 162 of Proceedings of Machine Learning Research , pages 9902-9915. PMLR, 2022.
- [34] Zhendong Wang, Jonathan J. Hunt, and Mingyuan Zhou. Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023 . OpenReview.net, 2023.
- [35] Tim Pearce, Tabish Rashid, Anssi Kanervisto, David Bignell, Mingfei Sun, Raluca Georgescu, Sergio Valcarcel Macua, Shan Zheng Tan, Ida Momennejad, Katja Hofmann, and Sam Devlin. Imitating Human Behaviour with Diffusion Models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023 . OpenReview.net, 2023.
- [36] Baoquan Zhang, Chuyao Luo, Demin Yu, Xutao Li, Huiwei Lin, Yunming Ye, and Bowen Zhang. Metadiff: Meta-learning with conditional diffusion for few-shot learning. In ThirtyEighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada , pages 16687-16695. AAAI Press, 2024. doi: 10.1609/AAAI.V38I15.29608. URL https://doi.org/10.1609/aaai.v38i15.29608 .
- [37] Ling Yang, Zhilong Zhang, Zhaochen Yu, Jingwei Liu, Minkai Xu, Stefano Ermon, and Bin Cui. Cross-Modal Contextualized Diffusion Models for Text-Guided Visual Generation and Editing. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024 , 2024. URL https://openreview.net/forum?id=nFMS6wF2xq .
- [38] Ye Tian, Ling Yang, Haotian Yang, Yuan Gao, Yufan Deng, Xintao Wang, Zhaochen Yu, Xin Tao, Pengfei Wan, Di Zhang, and Bin Cui. VideoTetris: Towards Compositional Text-to-Video Generation. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024 , 2024. URL http://papers.nips.cc/paper\_files/paper/ 2024/hash/345208bdbbb6104616311dfc1d093fe7-Abstract-Conference.html .
- [39] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 -December 9, 2022 , 2022. URL http://papers.nips.cc/paper\_files/paper/2022/hash/ a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html .
- [40] Diederik P. Kingma and Max Welling. Auto-Encoding Variational Bayes. In 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings , 2014.
- [41] Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic Backpropagation and Approximate Inference in Deep Generative Models. In Proceedings of the 31th International Conference on Machine Learning , volume 32 of JMLR Workshop and Conference Proceedings , pages 1278-1286, 2014.
- [42] Karl Pearson. Contributions to the mathematical theory of evolution. Philosophical Transactions of the Royal Society of London. A , 185:71-110, 1894.

- [43] Kai Wang Ng, Guo-Liang Tian, and Man-Lai Tang. Dirichlet and related distributions: Theory, methods and applications . John Wiley &amp; Sons, United States, 2011.
- [44] Frank Nielsen, Ke Sun, and Stéphane Marchand-Maillet. On hölder projective divergences. Entropy , 19(3):122, 2017.
- [45] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA , pages 6626-6637. Curran Associates Inc., 2017. URL https://proceedings.neurips.cc/paper/2017/ hash/8a1d694707eb0fefe65871369074926d-Abstract.html .
- [46] A. Krizhevsky and G. Hinton. Learning multiple layers of features from tiny images. Technical Report TR-2009, University of Toronto, Toronto, Canada, 2009. URL https://www.cs. toronto.edu/~kriz/cifar.html .
- [47] Severi Rissanen, Markus Heinonen, and Arno Solin. Generative Modelling with Inverse Heat Dissipation. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023 . OpenReview.net, 2023.
- [48] Ruiqi Gao, Yang Song, Ben Poole, Ying Nian Wu, and Diederik P. Kingma. Learning EnergyBased Models by Diffusion Recovery Likelihood. In International Conference on Learning Representations, Virtual Event, Austria, May 3-7, 2021 , 2021.
- [49] Alexander Quinn Nichol and Prafulla Dhariwal. Improved Denoising Diffusion Probabilistic Models. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event , volume 139 of Proceedings of Machine Learning Research , pages 8162-8171. PMLR, 2021.
- [50] Huangjie Zheng, Pengcheng He, Weizhu Chen, and Mingyuan Zhou. Truncated Diffusion Probabilistic Models and Diffusion-based Adversarial Auto-Encoders. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023 . OpenReview.net, 2023. URL https://openreview.net/forum?id=HDxgaKk956l .
- [51] Giannis Daras, Mauricio Delbracio, Hossein Talebi, Alex Dimakis, and Peyman Milanfar. Soft Diffusion: Score Matching with General Corruptions. Transactions on Machine Learning Research , 2023. ISSN 2835-8856.
- [52] Emiel Hoogeboom and Tim Salimans. Blurring diffusion models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023 . OpenReview.net, 2023.
- [53] Javier E. Santos, Zachary R. Fox, Nicholas Lubbers, and Yen Ting Lin. Blackout diffusion: Generative diffusion models in discrete-state spaces. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA , volume 202, pages 9034-9059. PMLR, 2023. URL https://proceedings.mlr.press/v202/santos23a.html .
- [54] Zhengyang Geng, Ashwini Pokle, and J Zico Kolter. One-step diffusion distillation via deep equilibrium models. Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024 , 36:41914-41931, 2024.
- [55] Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diffinstruct: A universal approach for transferring knowledge from pre-trained diffusion models. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 15, 2024 , 2024.
- [56] Shuchen Xue, Zhaoqiang Liu, Fei Chen, Shifeng Zhang, Tianyang Hu, Enze Xie, and Zhenguo Li. Accelerating diffusion sampling with optimized time steps. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024 , pages 8292-8301. IEEE, 2024. doi: 10.1109/CVPR52733.2024.00792. URL https: //doi.org/10.1109/CVPR52733.2024.00792 .

- [57] Quan Dao, Binh Ta, Tung Pham, and Anh Tran. A high-quality robust diffusion framework for corrupted dataset. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part LXXXIV , volume 15142, pages 107-123. Springer, 2024. doi: 10.1007/978-3-031-72907-2\_7. URL https://doi.org/10.1007/ 978-3-031-72907-2\_7 .
- [58] Tero Karras, Samuli Laine, and Timo Aila. A Style-Based Generator Architecture for Generative Adversarial Networks. In Conference on Computer Vision and Pattern Recognition , pages 4401-4410, 2019. doi: 10.1109/CVPR.2019.00453.
- [59] Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. StarGAN v2: Diverse Image Synthesis for Multiple Domains. In Conference on Computer Vision and Pattern Recognition , pages 8185-8194, 2020. doi: 10.1109/CVPR42600.2020.00821.
- [60] Emiel Hoogeboom and Tim Salimans. Blurring Diffusion Models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023 , 2023.
- [61] Atefeh Khoshkhahtinat, Ali Zafari, Piyush M. Mehta, and Nasser M. Nasrabadi. Laplacianguided entropy model in neural codec with blur-dissipated synthesis. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 3045-3054, 2024. URL https://api.semanticscholar.org/CorpusID:268680730 .
- [62] Paul W Holland and Samuel Leinhardt. An exponential family of probability distributions for directed graphs. Journal of the American Statistical Association , 76(373):33-50, 1981.
- [63] Charles Elkan. Clustering documents with an exponential-family approximation of the dirichlet compound multinomial distribution. In William W. Cohen and Andrew W. Moore, editors, Machine Learning, Proceedings of the Twenty-Third International Conference (ICML 2006), Pittsburgh, Pennsylvania, USA, June 25-29, 2006 , volume 148 of ACMInternational Conference Proceeding Series , pages 289-296. ACM, 2006. doi: 10.1145/1143844.1143881. URL https: //doi.org/10.1145/1143844.1143881 .
- [64] David Blackwell and Lester E. Dubins. A converse to the dominated convergence theorem. Illinois Journal of Mathematics , 7:508-514, 1963. URL https://api.semanticscholar. org/CorpusID:122751594 .
- [65] Brian DO Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications , 12(3):313-326, 1982.
- [66] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2009), 20-25 June 2009, Miami, Florida, USA , pages 248-255. IEEE Computer Society, 2009. doi: 10.1109/CVPR.2009.5206848. URL https: //doi.org/10.1109/CVPR.2009.5206848 .
- [67] Krishna Regmi and Ali Borji. Cross-view image synthesis using conditional gans. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition , pages 3501-3510, 2018.
- [68] Hao Tang, Dan Xu, Nicu Sebe, Yanzhi Wang, et al. Multi-channel attention selection gan with cascaded semantic guidance for cross-view image translation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 2417-2426, 2019.
- [69] Taesung Park, Alexei A Efros, Richard Zhang, and Jun-Yan Zhu. Contrastive learning for unpaired image-to-image translation. In Computer Vision-ECCV 2020: 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part IX 16 , pages 319-345. Springer, 2020.
- [70] Aysim Toker, Qunjie Zhou, Maxim Maximov, and Laura Leal-Taixé. Coming down to earth: Satellite-to-street view synthesis for geo-localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 6488-6497, 2021.

- [71] Divya Kothandaraman, Tianyi Zhou, Ming Lin, and Dinesh Manocha. Aerial diffusion: Text guided ground-to-aerial view synthesis from a single image using diffusion models. In SIGGRAPH Asia 2023 Technical Communications , pages 1-4. 2023.
- [72] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 18392-18402, 2023.
- [73] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) , pages 3836-3847, October 2023.
- [74] Gaurav Parmar, Taesung Park, Srinivasa Narasimhan, and Jun-Yan Zhu. One-step image translation with text-to-image models. arXiv preprint arXiv:2403.12036 , 2024.
- [75] Ahmad Arrabi, Xiaohan Zhang, Waqas Sultan, Chen Chen, and Safwan Wshah. Cross-view meets diffusion: Aerial image synthesis with geometry and text guidance. arXiv preprint arXiv:2408.04224 , 2024.
- [76] Junyan Ye, Jun He, Weijia Li, Zhutao Lv, Jinhua Yu, Haote Yang, and Conghui He. Skydiffusion: Street-to-satellite image synthesis with diffusion models and bev paradigm. arXiv preprint arXiv:2408.01812 , 2024.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction articulate the contributions and scope of this study, including the methods, experimental results, and impacts.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We address the limitations of this research and the directions for future work in Section 5 and 6.

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

Justification: Every theoretical result has an associated proof in the appendix.

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

Justification: The paper fully describes the experimental information, including the algorithm steps, hyperparameter settings, training details, and so on, in the main text and the appendix F and G.

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

Justification: The datasets used in the experiments are publicly available. Code will be released with instructions.

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

Justification: The paper presents all the training and testing details. The information is introduced both in the main text and appendix G.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: Consistent with previous work, error bars are not reported. The related work is conducted in accordance with precedent.

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

Justification: For all experiments, the paper supplies details regarding the computational infrastructure employed.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in this paper adheres to the Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The broader impacts are discussed in appendix H

Guidelines:

- The answer NA means that there is no societal impact of the work performed.

- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [Yes]

Justification: The image generation model presented in this paper is equipped with essential protective measures, mandating that users comply with the usage guidelines for generative models to ensure controlled application of the model. The datasets used in the experiments are publicly available.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We cite both the paper and the license of each model and dataset we use.

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

Justification: Assets are documented with README files and the documentation is provided alongside the assets.

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

## Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Theoretical Analysis

Definition 1 (Proper vs. Improper Divergence [44]) . Let P denote the set of all probability densities on a measurable space (Ω , ν ) . A nonnegative functional D : P × P → R ≥ 0 is strictly proper if

<!-- formula-not-decoded -->

̸

If there exist distinct densities p = q such that D ( p : q ) = 0 , then D is improper .

Definition 2 ( Hölder Statistical Pseudo-Divergence (HPD) [44] ) . Let (Ω , F , µ ) be a measurable space with Lebesgue measure µ . Fix conjugate exponents α, β &gt; 0 satisfying 1 /α +1 /β = 1 . For densities p ∈ L α (Ω , µ ) and q ∈ L β (Ω , µ ) , each absolutely continuous w.r.t. µ , the HPD is

<!-- formula-not-decoded -->

Definition 3 ( Proper Hölder Divergence (PHD) [44] ) . Using the same exponents α, β &gt; 0 and a parameter γ &gt; 0 , the proper Hölder divergence between densities p, q ∈ L γ (Ω , µ ) is

<!-- formula-not-decoded -->

Definition 4 ( Exponential Family of Distribution [62] ) . The probability density function of the Dirichlet distribution is expressed as follows: p ( x ; θ ) = exp { θ ⊤ T ( x ) -F ( θ ) + B ( x ) } , where θ is the natural parameter, T ( x ) is the sufficient statistic, F ( θ ) is the log-normalizer, and B ( x ) is the base measure.

Definition 5 ( Dirichlet Distribution [43] ) . The Dirichlet distribution of order K (where K ≥ 2 ) with parameters α i &gt; 0 , i = 1 , 2 , 3 ..., K is defined by a probability density function with respect to Lebesgue measure on the Euclidean space R K -1 as follows:

<!-- formula-not-decoded -->

where x i ∈ S K , and S K is the standard K -1 dimentional simplex, namely,

<!-- formula-not-decoded -->

and Γ( . ) is the gamma function, defined as: Γ( s ) = ∫ ∞ 0 x s -1 e -x d x, s &gt; 0 .

Definition 6 ( The Exponential form of the Dirichlet Distribution [63] ) . Exponential formulation of the Dirichlet distribution probability density function can be rewrite as exp { ∑ K i =1 ( α i -1) log x i -[ ∑ K i =1 log Γ( α i ) -log Γ ( ∑ K i =1 α i )]} . Allowing us to obtain the

<!-- formula-not-decoded -->

K ∑ i =1 lnΓ( α i ) -ln Γ( K ∑ i =1 α i ) , B ( x ) = -ln( x ) , and ψ is the digamma function, defined as: ψ ( x ) = d d x ln Γ( x ) .

Assumption 1. The model density is Dirichlet

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assumption 2 (Integrability for PHD) . For γ &gt; 0 .

- (a) If 0 &lt; γ ≤ 1 , no extra condition is needed (because p γ , q γ ∈ L 1 automatically for probability densities p, q ).
- (b) If γ &gt; 1 , assume for every model instance θ that each Dirichlet concentration satisfies: α i ( θ ) &gt; 1 -1 γ , ∀ i. This guarantees p γ , q γ ∈ L 1 .

Theorem 1 (HPD is improper; PHD is strictly proper) . Let α, β &gt; 0 with α -1 + β -1 = 1 and γ &gt; 0 . Then:

̸

- (i) HPD is improper. There exist densities p = q with D H α ( p : q ) = 0 .
- (ii) PHD is strictly proper for α &gt; 1 . For all densities p, q ,

<!-- formula-not-decoded -->

- (iii) Boundary γ = 1 . Statement (ii) still holds without any extra condition.
- (iv) Right-limit α → 1 + . Define D H 1 + ,γ ( p : q ) := lim α → 1 D H α,γ ( p : q ) (the limit exists). If D H 1 + ,γ ( p : q ) = 0 , then p = q a.e. Thus strict propriety extends to the α → 1 limit.

Proof. We write the two divergences in their 'normalised Hölder form':

<!-- formula-not-decoded -->

̸

(i) HPD is improper. By Hölder's inequality, ∫ p q ≤ ∥ p ∥ α ∥ q ∥ β , so D H α ( p : q ) ≥ 0 . Equality in Hölder holds iff p α = c q β a.e. for some c &gt; 0 . One can pick p = q satisfying that proportionality (e.g. on [0 , 1] , take p ( x ) = 2 x and q ( x ) = 3 x 2 for ( α, β ) = (3 / 2 , 3) ); then D H α ( p : q ) = 0 although p = q . Hence HPD is improper.

̸

(ii) PHD is strictly proper when α &gt; 1 . Apply Hölder with f = p γ/α and g = q γ/β :

<!-- formula-not-decoded -->

Thus D H α,γ ( p : q ) ≥ 0 . Assume D H α,γ ( p : q ) = 0 . Then we are in the equality case of Hölder, so

<!-- formula-not-decoded -->

for some c &gt; 0 . Hence p γ = c q γ a.e., i.e. p = c 1 /γ q a.e. Since p and q are densities, 1 = ∫ p = c 1 /γ ∫ q = c 1 /γ , so c = 1 , giving p = q a.e. Conversely, if p = q , the ratio is 1 , hence D H α,γ ( p : q ) = 0 .

(iii) The boundary γ = 1 . Put γ = 1 in the PHD definition. The same equality-case argument applies verbatim (no extra integrability is needed), so strict propriety still holds.

- (iv) Right-limit α → 1 + . For α &gt; 1 define

<!-- formula-not-decoded -->

The maps α ↦→ p γ/α , α ↦→ q γ/β are pointwise continuous for α &gt; 1 , and all integrands are dominated by integrable functions thanks to Assumption 2. Therefore, each integral appearing in Φ( α ) is continuous in α , and so is Φ .

̸

If p = q , the argument in (ii) shows that for every α &gt; 1 , Φ( α ) &gt; 0 (because equality in Hölder would force p = q ). By continuity on (1 , ∞ ) , there exists ϵ &gt; 0 such that Φ( α ) ≥ ϵ for all α sufficiently close to 1 . Consequently, lim α → 1 Φ( α ) ≥ ϵ &gt; 0 . Hence D H 1 + ,γ ( p : q ) = 0 when p = q . If p = q , clearly Φ( α ) ≡ 0 for all α , whence the limit is 0 . Thus the limit divergence D H 1 + ,γ remains strictly proper.

̸

̸

This distinction explains why, in Dirichlet diffusion training, PHD supplies a reliable approximation signal, while HPD may induce a 'false convergence' phenomenon.

Lemma 1. Let Z ∼ Dir( α ) with α 0 = ∑ K i =1 α i . For any model parameter vector θ ,

<!-- formula-not-decoded -->

Proof. The Dirichlet density on the ( K -1) -simplex S K is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Because ∂ x log Γ( x ) = ψ ( x ) (the digamma function),

<!-- formula-not-decoded -->

Differentiate (33) and group like terms:

<!-- formula-not-decoded -->

Insert the duplicated sum in the second term and combine:

<!-- formula-not-decoded -->

Finally evaluate at the random variable Z to obtain the stated identity.

Lemma 2. Let Z ∼ Dir( α ) on the ( K -1) -simplex S K , with α = ( α 1 , . . . , α K ) and α 0 = ∑ K i =1 α i . Fix an index i ∈ { 1 , . . . , K } and any λ &gt; -1 . Define µ := ψ ( α i ) -ψ ( α 0 ) , where ψ is the digamma function. Then:

<!-- formula-not-decoded -->

Moreover, using ψ ′ ( x ) ≤ 1 /x for x &gt; 0 and the mean-value bound | ψ ( u ) -ψ ( v ) | ≤ | u -v | / min { u, v } , we obtain the (loose but handy) upper bound:

<!-- formula-not-decoded -->

Proof. Step 1: Reduce to a one-dimensional Beta integral. The joint pdf of Z is

<!-- formula-not-decoded -->

Because we only involve Z i , integrate out z -i to use the marginal:

<!-- formula-not-decoded -->

Hence

i.e. Z i ∼ Beta( α i , α 0 -α i ) . Therefore

<!-- formula-not-decoded -->

Step 2: Notation and Beta derivatives. Set a := α i + λ, b := α 0 -α i , B ( a, b ) := ∫ 1 0 t a -1 (1 -t ) b -1 d t = Γ( a )Γ( b ) Γ( a + b ) . Then

<!-- formula-not-decoded -->

Define J ( a, b ; µ ) := ∫ 1 0 t a -1 (1 -t ) b -1 (log t -µ ) 2 d t. We need J ( a, b ; µ ) in closed form. Recall the well-known identities:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 3: Express J via B and its derivatives. Expand the square: (log t -µ ) 2 = (log t ) 2 -2 µ log t + µ 2 . Hence

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substitute the derivative formulas:

<!-- formula-not-decoded -->

Step 4: Plug back and simplify. Therefore

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 5: A convenient upper bound. Use ψ ′ ( x ) ≤ 1 /x for x &gt; 0 and, by the mean-value theorem, | ψ ( u ) -ψ ( v ) | ≤ | u -v | min { u,v } ( u, v &gt; 0) . Here u = α i + λ , v = α i or u = α 0 + λ , v = α 0 , so the squared bracket term is O ( λ 2 / [( α i + λ ) 2 ] ) + O ( λ 2 / [( α 0 + λ ) 2 ] ) . Absorbing absolute constants, we obtain the loose bound (38):

<!-- formula-not-decoded -->

Lemma 3 (Gradient representations for HPD and PHD) . Assume Assumptions 1-2 . Define the normalising constants

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let model score is and introduce the probability weights

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Throughout, the interchange of ∇ θ and ∫ S K is justified by the dominated-convergence theorem [64] because Assumption 1 places θ in a compact set and the Dirichlet density is C ∞ in θ . (i) HPD. Write

<!-- formula-not-decoded -->

Only A and C depend on θ . Using

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we obtain

Hence

<!-- formula-not-decoded -->

which is Equ. (56).

(ii) PHD. Similarly,

Then hence

<!-- formula-not-decoded -->

Only T 1 and T 3 involve θ . Chain-rule differentiation gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substituting into the derivative of D H α,γ

<!-- formula-not-decoded -->

which is exactly Equ. (57).

Theorem 2 (Variance domination of PHD) . Adopt Assumptions 1-2 and suppose in addition that the Hölder scale satisfies γ ≥ 1 . Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Step 1 Gradient decompositions. Equs. (56)-(57) give

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 2 Two-variance trick. For any random vectors X,Y one has Var( X -Y ) ≤ 2 ( Var X +Var Y ) . Hence

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 3 Bounding the four variances. Lemma 2 with λ = β ( α i -1) yields

<!-- formula-not-decoded -->

with α min := min i,θ α i ( θ ) . The same calculation (setting β = 1 ) gives Var w (1) [ g θ ] ≤ Kc 2 / ( 2 α min ) . Thus

<!-- formula-not-decoded -->

For the PHD weights choose λ = γ ( α i -1) . Because γ ≥ 1 , Assumption 2 ensures λ &gt; -1 . Lemma 2 gives

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

Step 4 Ratio comparison. We can obtain:

<!-- formula-not-decoded -->

because (i) α min ≤ α , (ii) β ≥ 1 , and (iii) 1 + γ ≥ β when γ ≥ 1 . Multiplying through by Var( G H ) establishes the theorem.

Lemma 4 (HPD and PHD for Conic or Affine Exponential Family [44]) . For distributions p ( x ; θ p ) and q ( x ; θ q ) in the same exponential family with a conic or affine natural parameter space, the Hölder statistical pseudo-divergence and its generalization admit closed-form expressions:

<!-- formula-not-decoded -->

Here F ( θ ) is the log-normalizer (cumulant generating function) of the exponential family.

Lemma 5 (Symmetric HPD and PHD for Conic or Affine Exponential Family [44]) . Under the same assumptions, the symmetric versions of these divergences also have closed-form:

<!-- formula-not-decoded -->

Set A = ∑ K i =1 a i , B = ∑ K i =1 b i , α -1 + β -1 = 1 , α, β &gt; 0 , γ &gt; 0 . Via the Lemmas 4-5 We can obtain:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ¯ α = β and ¯ β = α .

Assumption 3 (Smooth Dirichlet model) . There exists c &gt; 0 such that sup θ,i ∥ ∂ θ α i ( θ ) ∥ ≤ c .

Assumption 4 (Admissible γ ) . If γ &gt; 1 , then α i ( θ ) &gt; 1 -1 γ for all i and all θ . (No extra condition is needed when 0 &lt; γ ≤ 1 .)

Theorem 3 (Uniform variance domination of PHD over HPD) . Let p be the target density and q θ ( z ) = Dir ( z | α ( θ ) ) the model, with α i ( θ ) &gt; 0 and α 0 ( θ ) = ∑ K i =1 α i ( θ ) , satisfying Assumptions 3-4. Let

1 /α +1 /β = 1 with α, β &gt; 0 , and γ ≥ 1 . Denote g ( z ) = ∇ log q ( z ) . Then, for every θ ,

<!-- formula-not-decoded -->

Moreover, α 0 +1 α 0 + γ ≤ 1 and decreases monotonically to 0 as γ →∞ , implying a strict improvement for any γ &gt; 1 .

Lemma 6 (Dirichlet score) . If Z ∼ Dir( α ) with α 0 = ∑ K i =1 α i , then

<!-- formula-not-decoded -->

where ψ is the digamma function.

Lemma 7 (Gradient forms of HPD and PHD) . Let

<!-- formula-not-decoded -->

Define weights w (1) = pq θ A , w (2) = q β θ C , ˜ w (1) = p γ/α q γ/β θ T 1 , ˜ w (2) = q γ θ T 3 . Then:

<!-- formula-not-decoded -->

Lemma 8 (Second-moment bound under powered Dirichlet) . Let Z ∼ Dir( α ) , α i &gt; 0 , and λ &gt; -1 . Let µ i = ψ ( α i ) -ψ ( α 0 ) . Then

<!-- formula-not-decoded -->

where ∆ i ( λ ) = ψ ( α i + λ ) -ψ ( α 0 + λ ) -µ i and ψ ′ is the trigamma function. Moreover,

<!-- formula-not-decoded -->

Proof. We split the proof into four steps. Throughout, Z ∼ Dir( α ) , α 0 = ∑ K i =1 α i , and g θ ( z ) = ∇ θ log q θ ( z ) .

Step 0: Notation and two gradient decompositions. From Lemma 7, define

<!-- formula-not-decoded -->

Step 1: A variance upper bound for differences. For any random vectors X,Y , V ar ( X -Y ) ≤ 2 ( V arX + V arY ) . Applying this to G H and G P gives:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus it suffices to bound the four weighted variances of g θ .

Step 2: Bounding V ar [ g θ ] under the four weights. Lemma 6 yields

<!-- formula-not-decoded -->

Let a = ( ∂ θ α 1 , . . . , ∂ θ α K ) ⊤ and u ( z ) = ( u 1 ( z ) , . . . , u K ( z )) ⊤ . Then

<!-- formula-not-decoded -->

Using λ max (Cov[ u ]) ≤ tr(Cov[ u ]) = ∑ i V ar [ u i ] , we get

<!-- formula-not-decoded -->

where Assumption 3 gives ∥ a ∥ 2 2 ≤ Kc 2 .

Under each weight, the marginal of Z i is a (powered) Dirichlet/Beta, so Lemma 8 applies with:

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

<!-- formula-not-decoded -->

Step 3: Comparing bounds and simplifying constants. Divide (102) by (101):

<!-- formula-not-decoded -->

We now relate this to the desired factor. Because 1 /α +1 /β = 1 , we have β ≥ 1 , α ≤ 1 , hence αβ ≤ β . Moreover, for any α 0 &gt; 0 and γ ≥ 1 ,

<!-- formula-not-decoded -->

(A short algebraic verification is given in Appendix A.2.) Therefore,

<!-- formula-not-decoded -->

and multiplying both sides by V ar ( G H ) yields (82). Step 4: Tightness and monotonicity. When p = q θ , both gradients vanish in expectation, and the ratio approaches the constant factor (tight up to trigamma bounds). The factor ( α 0 +1) / ( α 0 + γ ) ≤ 1 is decreasing in γ and tends to 0 as γ →∞ , giving strict improvement for any γ &gt; 1 .

Its inequality form gives:

Hence

Plugging into (88)-(89):

## B SDE Formulation

Data sample: x 0 ∼ p data ( x ) . Time: t ∈ [0 , 1] (or [0 , T ] in other conventions). State: x t ∈ R d . { W t } t ≥ 0 : standard Wiener process. Score function: ∇ x log p t ( x ) , the gradient of the log-density of x t . Forward Diffusion SDE The forward (noising) process is described by an Itô SDE [22].

<!-- formula-not-decoded -->

The corresponding density p t ( x ) evolves according to the Fokker-Planck equation

<!-- formula-not-decoded -->

Reverse-Time SDE (Generation Process) By Anderson's time-reversal theorem (1982) [65], the time-reversed process t : 1 → 0 is also an SDE:

<!-- formula-not-decoded -->

where ¯ W t is another Wiener process (independent of W t ), and d t &lt; 0 when integrating backward in time. In practice, we reparameterize time to integrate forward (e.g., τ = 1 -t ) and approximate the score with a neural network s θ ( x, t ) ≈ ∇ x log p t ( x ) .

Forward Simulation (Training-Time Noise Sampling) Using Euler-Maruyama,

<!-- formula-not-decoded -->

Reverse-Time Generation Discretizing gives

<!-- formula-not-decoded -->

SDE for Proper Hölder-Kullback Dirichlet Diffusion Let x 0 ∼ p data ( x ) and x t ∈ ∆ K -1 := { x ∈ R K ≥ 0 | 1 ⊤ x = 1 } for t ∈ [0 , 1] . We denote by { W t } t ≥ 0 a K -dimensional Wiener process. For a time-varying Dirichlet schedule α ( t ) = ( α 1 ( t ) , . . . , α K ( t )) ⊤ with α i ( t ) &gt; 0 and α 0 ( t ) = ∑ K i =1 α i ( t ) , define the simplex projection covariance

<!-- formula-not-decoded -->

Let s ⋆ t ( x ) = ∇ x log p t ( x ) be the true score of the forward marginal p t ( x ) , approximated in practice by a neural network s θ ( x, t ) trained with the PHK objective.

Forward (Noise-Injection) SDE We adopt a Wright-Fisher/Dirichlet-type diffusion to ensure x t remains on ∆ K -1 . For a positive noise schedule β ( t ) &gt; 0 :

<!-- formula-not-decoded -->

The drift pulls x t toward the instantaneous Dirichlet mean α ( t ) /α 0 ( t ) , while the diffusion term Σ( x t ) 1 / 2 keeps trajectories inside the simplex.

Reverse (Generation) SDE By Anderson's time-reversal theorem, the reverse-time process t satisfies

<!-- formula-not-decoded -->

where ¯ W t is another Wiener process (independent of the forward one) and d t &lt; 0 when integrating backward. In practice we reparameterize time by τ = 1 -t to integrate forward in τ .

Discrete-Time Euler-Maruyama Schemes For a small step ∆ t :

Forward simulation (training-time noising):

<!-- formula-not-decoded -->

Reverse sampling (generation):

<!-- formula-not-decoded -->

Higher-order SDE/ODE solvers (e.g., Heun, DPM-Solver, UniPC) can be used in place of Euler steps.

Score Learning with Proper Hölder-Kullback (PHK) Objective PHK combines Proper Hölder Divergence (PHD) [44] and KL-based terms to stabilize and symmetrize training:

<!-- formula-not-decoded -->

where q ⋆ t denotes the 'teacher' (true or target) conditional and q θ is the Dirichlet model induced by the network. Minimizing (116) yields s θ ( x, t ) ≈ s ⋆ t ( x ) , which is then plugged into (113).

## C HPD Impropriety Counter Example

The zero condition for the Hölder projective divergence (HPD) is

<!-- formula-not-decoded -->

which is equivalent to

Hence

̸

<!-- formula-not-decoded -->

Corrected counter-example (discrete case). Let p = ( p 1 , . . . , p K ) be any non-uniform probability vector and choose α &gt; 1 (thus β = α/ ( α -1) ). Define

<!-- formula-not-decoded -->

Then p = q (unless p is uniform), yet

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

and by the equality condition of Hölder's inequality the numerator and denominator of HPD are identical, yielding D H α ( p : q ) = 0 although p = q . This establishes the impropriety (failure of identity of indiscernibles) of HPD as stated in [44]

Numerical illustration. For example, let α = 1 . 5 and p = (0 . 7 , 0 . 3) . Then

<!-- formula-not-decoded -->

and D H α ( p : q ) evaluates to 0 up to machine precision.

## D Aditional Comparison

As shown in Table 3, PHK-Dir represents a 12 % improvement over DDPM's 3.17 [2] When B=128 and NFE=1000. As shown in Table 4, PHK-Dir represents 18.98 FID on the CVUSA[66] dataset.

Table 3: FID scores for DDPM and PHK-optimized Dirichlet Diffusion on the CIFAR-10 image dataset with varying NFEs under several different mini-batch size B .

| Model   |   B | NFE=20   | 50   | 200   | 500   |   1000 | 2000   |
|---------|-----|----------|------|-------|-------|--------|--------|
| PHK-Dir | 512 | 15.92    | 5.87 | 3.20  | 2.92  |   2.84 | 2.82   |
| PHK-Dir | 288 | 16.53    | 6.15 | 3.29  | 2.99  |   2.91 | 2.81   |
| PHK-Dir | 128 | 16.85    | 6.14 | 3.25  | 2.83  |   2.79 | 2.78   |
| DDPM    | 128 | -        | -    | -     | -     |   3.17 | -      |

Table 4: FID and LPIPS scores for different models on the CVUSA[66] dataset.

| Model             |   FID ( ↓ ) |   LPIPS ( ↓ ) |
|-------------------|-------------|---------------|
| X-Seq [67]        |      161.16 |         0.706 |
| SelGAN [68]       |      116.57 |         0.742 |
| CUT [69]          |       72.83 |         0.687 |
| CDTE [70]         |      122.84 |         0.694 |
| Aerial Diff [71]  |      136.18 |         0.855 |
| Instr-p2p [72]    |       38.01 |         0.697 |
| ControlNet [73]   |       32.45 |         0.65  |
| I2I-Turbo [74]    |       77.95 |         0.685 |
| GPG2A [75]        |       58.8  |         0.691 |
| BEV [76]          |       29.18 |         0.635 |
| PHK-Dir(20% data) |       21.8  |         0.599 |

## E Compositional Data

As shown in Figure 5, Dirichlet ELBO and Dirichlet PHK achieve better performance in the compositional data. Dirichlet ELBO is better than Gauss ELBO. Dirichlet PHK is better than Dirichlet ELBO.

Figure 5: Comparison of true probability mass function (PMF) and empirical PMFs of three methods-Gauss ELBO, Dirichlet ELBO, and Dirichlet PHK-computed. Each true PMF is depicted with red square. Each empirical PMF is computed based on 100k data generated by the model trained after 400k iterations with 200 Numbers of Function Evaluations (NFE).

<!-- image -->

## F Hyperparameter Settings

In the experiments, the neural network function ( f θ in equations) is implemented as a UNet architecture, as is typical in modern diffusion models [2]. we utilize the parameterization of Beta Diffusion [16] as the code base.

We demonstrate the intuition behind the setting of model parameters, including η , S hift , S cale , c 0 , c 1 , α , and β . Given the limitations of computational resources, the meticulous tuning of these model parameters is left for future work.

For the diffusion concentration parameter η , we set a moderately high value of η = 1000 . A larger value of η provides greater discriminative power among different pixel values but requires more discretization steps during sampling, leading to slower training speeds.

We set the linear scaling and shifting parameters as S cale = 0 . 39 and S hift = 0 . 60 , and the sigmoidbased schedule parameters as c 0 = 10 and c 1 = -13 . These settings are based on the work of Kingma et al. [3] and Zhou et al. [16].

We set the shape parameters of the Proper Hölder Divergence as α = 2 and β = α α -1 = 2 . This choice is motivated by an intuitive understanding of the Hölder Divergence, where a smaller α would lead to the loss of symmetry.

As shown in Table 7, the training-time cost of KLUB is substantially more expensive. When all other conditions are identical and B = 512 , KLUB incurs approximately a 15 % additional training-time cost compared with PHD.

Table 5: FID scores for negative ELBO, KLUB, PHD and PHK-optimized Dirichlet Diffusion on the CIFAR-10 image dataset with varying NFEs under several different combinations of concentration parameter η and mini-batch size B .

| Loss   |   η × 10 - 4 |   B |   NFE = 20 |   50 |   200 |   500 |   1000 |   2000 |
|--------|--------------|-----|------------|------|-------|-------|--------|--------|
| - ELBO |          1   | 512 |      15.81 | 6.53 |  4.49 |  4.33 |   4.28 |   4.31 |
| - ELBO |          1   | 288 |      16.02 | 6.65 |  4.68 |  4.42 |   4.42 |   4.39 |
| KLUB   |          1   | 512 |      16.62 | 6.16 |  3.46 |  3.39 |   3.32 |   3.25 |
| KLUB   |          1   | 288 |      16.35 | 6.23 |  3.58 |  3.46 |   3.4  |   3.21 |
| PHD    |          1   | 512 |      16.84 | 6.48 |  3.85 |  3.72 |   3.52 |   3.51 |
| PHD    |          1   | 288 |      16.95 | 6.93 |  4.43 |  4.23 |   4.19 |   4.21 |
| PHK    |          1   | 512 |      15.92 | 5.87 |  3.2  |  2.92 |   2.84 |   2.82 |
| PHK    |          0.5 | 512 |      17.08 | 6.6  |  3.55 |  3.09 |   3.1  |   3.14 |
| PHK    |          0.1 | 512 |      20.67 | 9.67 |  5.66 |  4.54 |   4.58 |   4.53 |
| PHK    |          1   | 288 |      16.53 | 6.15 |  3.29 |  2.99 |   2.91 |   2.81 |
| PHK    |          0.5 | 288 |      16.94 | 6.71 |  3.61 |  3.09 |   3.03 |   3.02 |
| PHK    |          0.1 | 288 |      20.33 | 9.56 |  5.71 |  4.77 |   4.75 |   4.72 |
| PHK    |          1   | 128 |      16.85 | 6.14 |  3.25 |  2.83 |   2.79 |   2.78 |
| PHK    |          0.5 | 128 |      16.32 | 6.4  |  3.57 |  3.23 |   3.09 |   3.06 |
| PHK    |          0.1 | 128 |      20.22 | 9.73 |  6    |  5    |   4.93 |   5    |

Table 6: Frechet Inception Distance (FID) scores for PHK-optimized Dirichlet Diffusion on the unconditional CIFAR-10 dataset with varying Numbers of Function Evaluations (NFE) under several different combinations of the key loss weight coefficients δ and ϵ with B = 288 .

| δ      | ϵ       |   NFE=20 |   50 |   200 |   500 |   1000 |   2000 |
|--------|---------|----------|------|-------|-------|--------|--------|
| δ =0.5 | ϵ =-0.1 |    16.62 | 6.38 |  3.78 |  3.56 |   3.42 |   3.35 |
| δ =0.5 | ϵ =0    |    16.14 | 6.17 |  3.66 |  3.43 |   3.34 |   3.29 |
| δ =0.5 | ϵ =0.1  |    16.71 | 6.16 |  3.25 |  2.92 |   2.82 |   2.84 |
| δ =0.5 | ϵ =0.15 |    16.74 | 6.06 |  3.2  |  3    |   2.86 |   2.78 |
| δ =0.5 | ϵ =0.2  |    16.53 | 6.15 |  3.29 |  2.99 |   2.91 |   2.81 |
| δ =0.5 | ϵ =0.25 |    16.58 | 6.25 |  3.47 |  3.12 |   2.98 |   2.93 |
| δ =0.5 | ϵ =0.5  |    16.07 | 6.23 |  3.61 |  3.39 |   3.35 |   3.31 |

## G Training Details

In the course of our model training, we employed the Adam optimization algorithm with specific parameter settings: the learning rate lr = 5 × 10 -4 , the exponential decay rate for the first moment estimates β 1 = 0 . 9 , the exponential decay rate for the second moment estimates β 2 = 0 . 999 , and the numerical stability parameter ϵ = 1 × 10 -8 .

For the purpose of training the model, we utilized 200 million images. Specifically, we leveraged four Nvidia L40s GPUs for computation, and with a batch size B = 512 , processing 1000 images of size 32 × 32 × 3 required approximately 1.16 seconds. The processing of 200 million CIFAR-10 images took approximately 64 hours.

During the training process, we saved a checkpoint every 25,000 steps and selected the model with the best performance based on the Fréchet Inception Distance (FID) scores obtained from these checkpoints. We assessed the FID scores using 50,000 samples, a practice consistent with previous research endeavors.

Table 7: The training time cost of the KLUB, PHD, and PHK-optimized Dirichlet Diffusion models on the unconditional CIFAR-10 dataset across two mini-batch sizes B with four Nvidia L40s GPUs.

|   B | KLUB   | PHD   | PHK   |
|-----|--------|-------|-------|
| 512 | 70h    | 61h   | 64h   |
| 288 | 73h    | 64h   | 68h   |

Table 8: The training time cost of the PHK-optimized Dirichlet Diffusion on the CIFAR10-32 × 32, AFHQ-64 × 64, FFHQ-64 × 64 datasets across two mini-batch sizes B with four Nvidia L40s GPUs.

| B       | CIFAR10-32 × 32   | AFHQ-64 × 64    | FFHQ-64 × 64    |
|---------|-------------------|-----------------|-----------------|
| 512 288 | 64 hours 68 hours | 10 days 11 days | 11 days 13 days |

Table 9: The Sampling time cost of the PHK-optimized Dirichlet Diffusion on the unconditional CIFAR10 at varying Numbers of Function Evaluations (NFE) with two or four Nvidia L40s GPUs.

| Nvidia L40s GPUs   | NFE = 20   | 50      | 200        | 500          | 1000        | 2000         |
|--------------------|------------|---------|------------|--------------|-------------|--------------|
| 2 4                | 11m 7m     | 21m 10m | 1h 13m 37m | 3h 2m 1h 33m | 6h 4m 3h 5m | 12h 7m 6h 9m |

## H Broader Impacts

This study's effective diffusion model is capable of generating images required for practical applications and further developing based on this foundational model, which will have a positive social impact on related fields of work. Technologies such as virtual try-on and video generation, formed based on diffusion models, provide more convenient development space for people's social lives.

This research may also raise concerns about the potential negative social impacts when training on maliciously curated image datasets, which is a shared challenge in the field of generative models. It is crucial to consider how to use these models responsibly to improve society, prepare comprehensive contingency plans, and mitigate any potential adverse consequences.

## I Uncurated Samples

Figures 6, 7, 8 9 show more uncurated samples from our trained models on CIFAR-10, FFHQ, AFHQ and CVUSA.Through these extensive results, the model's capacity to generate diverse and high-fidelity images without the need for manual selection or management can be clearly elucidated. This feature not only underscores the model's efficacy in image generation tasks but also further corroborates its generalization and adaptability on complex datasets.

Figure 6: Additional uncurated samples generated by our model trained on CIFAR-10 dataset.

<!-- image -->

Figure 7: Additional uncurated samples generated by our model trained on FFHQ dataset.

<!-- image -->

Figure 8: Additional uncurated samples generated by our model trained on AFHQ dataset.

<!-- image -->

Figure 9: Additional uncurated samples generated by our model trained on CVUSA dataset.

<!-- image -->