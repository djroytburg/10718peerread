## Zero-shot Denoising via Neural Compression: Theoretical and algorithmic framework

Ali Zafari ∗

ali.zafari@rutgers.edu

Xi Chen ∗

xi.chen15@rutgers.edu

Rutgers University New Brunswick, NJ, USA

## Abstract

Zero-shot denoising aims to denoise observations without access to training samples or clean reference images. This setting is particularly relevant in practical imaging scenarios involving specialized domains such as medical imaging or biology. In this work, we propose the Zero-Shot Neural Compression Denoiser (ZS-NCD), a novel denoising framework based on neural compression. ZS-NCD treats a neural compression network as an untrained model, optimized directly on patches extracted from a single noisy image. The final reconstruction is then obtained by aggregating the outputs of the trained model over overlapping patches. Thanks to the built-in entropy constraints of compression architectures, our method naturally avoids overfitting and does not require manual regularization or early stopping. Through extensive experiments, we show that ZS-NCD achieves state-of-the-art performance among zero-shot denoisers for both Gaussian and Poisson noise, and generalizes well to both natural and non-natural images. Additionally, we provide new finite-sample theoretical results that characterize upper bounds on the achievable reconstruction error of general maximum-likelihood compressionbased denoisers. These results further establish the theoretical foundations of compression-based denoising. Our code is available at: https://github.com/ Computational-Imaging-RU/ZS-NCDenoiser .

## 1 Introduction

Background and motivation Denoising is a fundamental problem in classical signal processing and has recently gained renewed attention from the machine learning community. Let x = ( x 1 , . . . , x n ) ∈ R n + denote a non-negative signal of length n , where signal x is not observable in many systems. Instead, we observe a noisy version y = ( y 1 , . . . , y n ) , where the observations are conditionally independent given x , and each entry is distributed according to a common conditional distribution:

<!-- formula-not-decoded -->

We assume that the noise mechanism is memoryless (independent across coordinates) and homogeneous (identical across entries). The goal of a denoising algorithm is to estimate x from the noisy observations y . Given its prevalence in imaging and data acquisition systems, denoising has been a central topic in signal processing for decades. Classical denoising methods rely on explicit structural assumptions about the underlying signal x , often hand-crafted by domain experts [1, 2, 3, 4, 5, 6, 7, 8]. In contrast, recent advances in machine learning have enabled a new class of data-driven denoising

∗ Equal contribution.

Shirin Jalali shirin.jalali@rutgers.edu

<!-- image -->

Zero-shotDenoisingPhase

Figure 1: Zero-Shot Neural Compression Denoiser (ZS-NCD). Learning phase: a neural compression model (architecture shown in Fig. 5 of the supplementary material) is trained on overlapping patches extracted from a single noisy image. Denoising phase: each pixel is reconstructed by averaging predictions across neighboring patches processed by the trained model.

algorithms. These methods learn the optimal denoising function from data, leveraging statistical patterns directly from signal and noise distributions.

While learning-based approaches achieve state-of-the-art performance and often outperform classical methods in controlled settings, they face significant challenges in practice:

1. Supervision requirement: Most learning-based methods require training set of paired samples { ( y i , x i ) } i m =1 , where x i is clean signal and y i is its noisy counterpart. In practical scenarios such as medical imaging, { x i } i m =1 are unavailable or prohibitively expensive to obtain.
2. Data efficiency: These methods usually need lots of training data. Acquiring sufficient samples is difficult or costly, particularly in domains with strict data acquisition constraints.

To mitigate the reliance on paired clean and noisy samples, several self-supervised denoising methods have been developed that learn directly from noisy observations, without access to clean ground truth signals [9, 10, 11, 12, 13]. While these approaches alleviate the supervision requirement, they typically depend on access to large collections of noisy data and often yield suboptimal performance compared to methods trained with clean targets. Moreover, the absence of clean supervision necessitates the use of complex neural architectures and training schemes, which can make these methods computationally demanding and difficult to optimize in practice.

These challenges have sparked growing interest in zero-shot denoisers , which aim to recover clean signals from noisy observations without access to paired data or extensive noisy training data. Such methods are particularly appealing in domains where acquiring clean data is infeasible, and they offer the potential for deployable denoisers that adapt to individual inputs with general purpose.

From neural compression to zero-shot denoising Denoising algorithms-ranging from classical signal processing techniques to deep learning methods-fundamentally rely on the assumption that real-world signals are highly structured. Compression-based denoising leverages this same principle, but rather than directly solving the inverse problem, it instead performs lossy compression on the

Figure 2: Zero-shot denoising of Kodim05 with AWGN ( σ = 25 ). Left : PSNR versus training iterations for zero-shot denoisers. Performance of BM3D [18] and Restormer [19] are included as a classical baseline and as a supervised empirical upper bound, respectively. Right : Visual reconstructions with PSNR above each image. Compression-based denoising based on JPEG2K [20] achieves inferior performance. Learning-based zero-shot denoisers often struggle with either overfitting or high bias. DIP [21] and DD [22] require early stopping to avoid overfitting. ZS-N2S [12] and S2S [23] struggle with high-resolution color images, and ZS-N2N [24] often produces noisy outputs with potential overfitting. BM3D tends to oversmooth the denoised image. In contrast, ZS-NCD avoids these issues.

<!-- image -->

noisy observation y , under the hypothesis that the clean signal x lies in a lower-complexity subspace and is therefore more compressible.

In lossy compression, the goal is to represent signals from a target class using discrete encodings with minimal distortion. When applied to noisy data, the intuition is that a lossy compressor-operating at a distortion level matched to the noise-will favor reconstructions close to the original clean signal. While this approach has a strong theoretical foundation [14, 15], classical compression-based denoisers have shown limited empirical success, particularly for natural image denoising.

In this work, we revisit this idea in light of recent progress in neural compression , where learned encoders and decoders have demonstrated strong rate-distortion performance across a variety of image domains [16, 17]. Building on this foundation, we propose a zero-shot denoising method that we call the Zero-Shot Neural Compression Denoiser (ZS-NCD) . Unlike traditional neural compression models that are trained on large corpora of clean high-resolution images, ZS-NCD learns directly from a single noisy input image. Specifically, we extract overlapping patches from the noisy image, and train a neural compression network on those patches alone-without any clean supervision or prior dataset. Once trained, the denoiser is applied to all patches from the same image, and the final output is obtained by averaging the predictions in overlapping regions. This approach is illustrated in Figure 1.

Despite relying solely on the noisy input and operating without supervision, ZS-NCD achieves state-of-the-art performance among zero-shot denoising methods across diverse noise models, and remains robust even on inputs that lie outside the natural image distribution. We compare it against the baselines in Figure 2, ZS-NCD shows superior performance in denoising and training stability.

Paper contributions This paper introduces a zero-shot image denoising framework based on neural compression. Our main contributions are:

- A zero-shot denoising algorithm using neural compression. We propose a fully unsupervised method that trains a neural compression network on image patches of the noisy input. It does not rely on clean images, paired datasets, or prior training on the target distribution. It is architectureagnostic and leverages only the structure present in the observed noisy image.
- Theoretical results connecting denoising and compression performance. We establish finitesample upper bounds on the reconstruction error of the proposed compression-based maximum likelihood denoisers, for both Gaussian and Poisson noise models.

- Extensive empirical validation. We demonstrate that our method achieves state-of-the-art performance among zero-shot denoising techniques across a range of noise models and datasets.

## 2 Related work

Self-supervised and zero-shot denoising Supervised learning-based denoisers such as DnCNN [25] and Restormer [19] achieve state-of-the-art performance across various noise models, but require large datasets of paired clean and noisy images-often impractical in real-world settings. To avoid clean images, self-supervised methods have been proposed, including Noise2Noise [10], Noise2Self [12], Noise2Void [11], Noise2Same[26] and Noise2Score [13], which only use noisy images for training. However, their reliance on large noisy datasets remains a limitation. Zero-shot denoisers address this by training on a single noisy image. These include (i) untrained networks like DIP [27] and Deep Decoder [22], and (ii) single-image adaptations of self-supervised methods, e.g., N2F [28], ZS-N2N [24], ZS-N2S [12] and its augmented variant with ensembling, S2S [23]. More recently, DS-N2N [29] improves ZS-N2N by further upsampling the downsampled paired noisy images. Pixel2Pixel [30] boosts the performance by using non-local similarity approach. DIP-based models avoid masking and leverage full-image context, but require early stopping or under-parameterization to avoid overfitting. Self-supervised variants suffer from masking-induced information loss. Hybrid approaches, such as masked pretraining-based method [31], uses external datasets for training and perform zero-shot inference, thus falling outside the zero-shot setting studied here.

Neural compression Learning-based lossy compression, often referred to as neural compression, uses an autoencoder architecture combined with an entropy model to estimate and constrain the bitrate at the bottleneck [32, 33]. These methods have significantly outperformed traditional codecs, particularly in image [32, 34, 35, 36, 37] and video [38, 39, 40] compression. In addition to these standard settings, several works have also explored applying neural compression to noisy data, either by adapting neural compression models for more efficient encoding of noisy images [41, 42, 43, 44].

Compression-based denoising Compression-based denoising leverages the insight that structured signals are inherently more compressible than their noisy counterparts. This connection was formalized by Donoho [14], who introduced the minimum Kolmogorov complexity estimator, and further refined by Weissman et al. [15], showing that, under certain conditions on both the signal and the noise, optimal lossy compression of a noisy signal-followed by suitable post-processing-can asymptotically achieve optimal denoising performance. Prior to these theoretical developments, early empirical methods such as wavelet-based schemes [45, 46, 47] and MDL-inspired heuristics [48] had explored this principle. Nevertheless, traditional compression-based denoisers have generally underperformed in high-dimensional settings such as image denoising.

Learning-based joint compression and denoising using neural compression has been explored in recent works [49, 50], where the goal is to achieve lower rate in compression. The empirical application of training neural compression for AWGN denoising was also proposed in [51]. Training the neural compression models in these works requires a dataset of images. In contrast, our proposed ZS-NCD is a two-step denoiser based on neural compression, trained on a single noisy image. It achieves state-of-the-art performance across both AWGN and Poisson noise models. Moreover, we contribute new theoretical results that advance the foundations of compression-based denoising.

## 3 Compression-based denoising: Theoretical foundations

Lossy compression Let Q ⊂ R n denote the signal class of interest, such as vectorized natural images of a fixed size. A lossy compression code for Q is defined by an encoder-decoder pair ( f, g ) , f : Q → { 1 , . . . , 2 R } , and g : { 1 , . . . , 2 R } → R n . The performance of a lossy code is characterized by: i) Rate R , indicating the number of distinct codewords; ii) Distortion δ , defined as the worst-case per-symbol mean squared error (MSE) over the signal class:

<!-- formula-not-decoded -->

The set of reconstructions produced by the decoder forms the codebook:

<!-- formula-not-decoded -->

Compression-based denoising We propose compression-based denoising as a structured maximum likelihood (ML) estimation. Given a noisy observation y ∼ ∏ n i =1 p ( y i | x i ) and a a lossy compression code ( f, g ) for Q , the compression-based ML denoiser solves

<!-- formula-not-decoded -->

This formulation leverages the fact that clean signals, by virtue of their structure, are more compressible than their noisy counterparts. Therefore, the most likely codeword under the noise model, when selected from a codebook designed to represent clean signals, serves as a natural denoising estimate. This ML-based view unifies denoising across noise models and provides a principled way to select reconstructions from a discrete, structure-aware prior.

In the case of AWGN: y = x + z , where z ∼ N ( 0 , σ 2 z I n ) , the described denoiser simplifies to:

<!-- formula-not-decoded -->

That is, denoising corresponds to projecting the noisy observation onto the nearest codeword.

Poisson noise commonly arises in low-light and photon-limited imaging scenarios. In this setting, each y i is modeled as a Poisson random variable with mean αx i : y i ∼ Poisson( αx i ) . Under this model, the compression-based ML denoiser simplifies to

<!-- formula-not-decoded -->

While the loss function in (2) is statistically well-motivated, it is more sensitive to optimization issues than its Gaussian counterpart due to the curvature and nonlinearity of the log term. To improve robustness and simplify optimization, we also consider an alternative loss based on a normalized squared error between c and the rescaled observations:

<!-- formula-not-decoded -->

Theoretical analysis We begin by analyzing the performance of compression-based ML denoising under AWGN. The following result provides a non-asymptotic upper bound on the reconstruction error in terms of the compression rate and distortion. All proofs can be found in Appendix A.

Theorem 1. Assume that x ∈ Q and let ( f, g ) denote a lossy compression for Q that operates at rate R and distortion δ . Consider y = x + z , where z ∼ N ( 0 , σ 2 z I n ) . Let ˆ x denote the output of the compression-based denoiser defined by ( f, g ) as in (1) . Then,

<!-- formula-not-decoded -->

with a probability larger than 1 -2 -ηR +2 .

This bound decomposes the denoising error into two terms: a distortion term √ δ , which reflects the approximation quality of the compression code, and a rate-dependent term that scales with the square root of the code rate R . The latter captures the likelihood concentration around the clean signal in high-probability regions of the noise distribution. Notably, the result holds non-asymptotically and does not assume the code is optimal, only that it provides a distortionδ covering of Q . This highlights that even non-ideal compression codes can enable effective denoising, provided the rate-distortion tradeoff is well-calibrated.

To better understand the implications of Theorem 1, in the following corollary, we focus on the special case of k -sparse signals.

Corollary 1 (AWGN, sparse signals) . Let Q n denote the set of k -sparse vectors in R n satisfying ∥ x ∥ ∞ ≤ 1 . Fix a parameter η ∈ (0 , 1) , and suppose y = x + z where z ∼ N (0 , σ 2 z I n ) . Then, there exists a family of compression codes such that, when used with the denoiser defined in (1) , with a probability larger than 1 -e -ηk log( n/k ) , the estimate ˆ x satisfies

<!-- formula-not-decoded -->

where C = 2(1 + 2 √ η ) √ 2 ln 2 and γ n = k log 2 k 2 n + log 2 k + k (log 2 e+1) n .

Corollary 1 provides a high-probability bound on the normalized error 1 √ n ∥ ˆ x -x ∥ 2 . Squaring both sides and having ( √ a + √ b ) 2 ≤ 2( a + b ) , it follows that, with high probability

<!-- formula-not-decoded -->

Thus, up to universal constants, the dominant term in the upper bound scales as σ 2 z k n log( n k ) . This matches the known minimax rate for estimating k -sparse signals in Gaussian noise when k/n → 0 ; see [52]. Determining whether the residual term 2 n is an artifact of our proof or a real barrier to optimality is an interesting problem. Finally, while the comparison above is high-probability rather than in expectation, one can integrate the tail bound from the proof to obtain an expected-risk bound.

We next extend our analysis to signal-dependent noise model. Poisson noise is particularly relevant in imaging applications such as microscopy and astronomy, where photon counts vary with signal intensity. Unlike Gaussian noise, Poisson observations induce a non-linear likelihood surface, making analysis more delicate. Theorem 2 and 3 establish performance guarantees for compression-based Poisson denoising, using both exact ML formulation and a practical squared-error surrogate.

Theorem 2. Consider the same setup of lossy compression as in Theorem 1. Assume that for any x ∈ Q , x i ∈ ( x min , x max ) , where 0 &lt; x min &lt; x max &lt; 1 . Assume that y 1 , . . . , y n are independent with y i ∼ Poisson( αx i ) . Let ˆ x denote the solution of (2) . Let C 1 = x 5 max / ( x 2 min ) and C 2 = x 2 max x 3 min β √ ( 4 ln 2 )( √ 1 + η + √ η ) . Then, with a probability larger than 1 -2 -ηR +2 ,

<!-- formula-not-decoded -->

Theorem 3. Consider the same setup as in Theorem 2. Let ˆ x denote the solution of (3) . Let C = 4 √ ln 2( √ 1 + η + √ η +1) . Then, with a probability larger than 1 -2 -ηR +2 ,

<!-- formula-not-decoded -->

Remark 1. Theorems 2 and 3 show that, in the case of Poisson noise, minimizing either the ML loss function or the computationally efficient MSE loss function can recover the signal. This result is also consistent with our simulations reported later in Section 5.

## 4 Zero-shot compression-based denoiser

We refer to a general class of learning-based denoisers that operate by compressing noisy images using neural compression as the Neural Compression Denoiser (NCD). In this framework, denoising is achieved by identifying a low-complexity reconstruction from the output of a neural compression model. In the previous section, we characterized the performance of such denoisers in a setting where the compression code is fixed in advance, either learned from external data or designed using classical methods, and applied independently of the noisy input. This setup is not zero-shot, as it relies on prior training or code design. Inspired by this idea, we now propose a fully unsupervised variant: the Zero-Shot Neural Compression Denoiser (ZS-NCD) . In ZS-NCD, a neural compression network is trained directly on patches extracted from a single noisy image, without access to clean targets or external data. This section describes the ZS-NCD architecture and optimization procedure in detail.

Proposed zero-shot denoiser: ZS-NCD Let P ( i,j ) : R h × w → R k × k denote the patch extraction operator, which returns a k × k patch whose top-left corner is at pixel ( i, j ) . Let f θ 1 and g θ 2 denote the encoder and decoder networks, parameterized by weights θ 1 and θ 2 , respectively. Define I as the set of all coordinates ( i, j ) ∈ { 1 , . . . , h -k +1 } × { 1 , . . . , w -k +1 } from which a valid k × k patch can be extracted.

Given a single noisy image y , the ZS-NCD is trained to minimize the following patchwise objective:

<!-- formula-not-decoded -->

where P ( f θ 1 ( P ( i,j ) ( y )) ) denotes the likelihood (or entropy model) of the latent code produced by the encoder, and λ &gt; 0 is a hyperparameter controlling the trade-off between fidelity and compressibility.

Figure 3: Effect of λ in denoising Mouse Nuclei image.

<!-- image -->

<!-- image -->

In (8), K = k 2 , and the function L K : R K × R K → R + is a distortion loss determined by the noise model, as defined in Section 3. For example, in the AWGN case, L K corresponds to the squared ℓ 2 norm of the distance between a noisy patch and its neural compression reconstruction. Note that f θ 1 maps the input into a discrete latent space, which is non-differentiable and thus incompatible with standard gradient-based optimization. To address this, we follow the neural compression framework of [32], using a continuous relaxation during training (e.g., uniform noise injection) and applying actual discretization only at test time. The entropy term P is modeled using a factorized, non-parametric density [34].

After training, the denoised image is obtained by applying the encoder and decoder to each patch and averaging the overlapping outputs. For each pixel ( i, j ) , let I ( i,j ) ⊂ I denote the set of patch locations such that P ( i ′ ,j ′ ) includes the pixel ( i, j ) . The final estimate at location ( i, j ) is given by

<!-- formula-not-decoded -->

where |I ( i,j ) | denotes the number of patches covering pixel ( i, j ) , and ·| ( a,b ) denotes the ( a, b ) -th pixel of the patch output. For interior pixels away from the boundary, |I ( i,j ) | = K . As shown later in Section 5, this aggregating of reconstructed patches significantly enhances denoising performance.

Setting the hyperparameter λ The ZS-NCD objective in (8) includes a hyperparameter, λ , which balances reconstruction fidelity and compressibility. Interpreted through the lens of lossy compression, varying λ allows the model to explore different rate-distortion trade-offs. However, in the context of denoising, our goal is not compression but accurate signal recovery from the noisy observation y . This raises the central question: how should λ be selected to optimize denoising performance? In the following, we explain our approach for setting λ under both AWGN model and Poisson noise model.

Case I: Gaussian noise. Let ˆ x denote the output of the ZS-NCD denoiser, and consider the AWGN model y = x + z with z ∼ N (0 , σ 2 z I n ) . Then,

<!-- formula-not-decoded -->

the first term is the noise variance, and the second is the true denoising error. While z and ˆ x are not fully independent, they are intuitively weakly correlated in successful denoising regimes, where the estimate ˆ x depends only indirectly on the noise. Thus, the cross term is expected to be small: 1 n E [ z ⊤ ( x -ˆ x ) ] ≈ 0 . This approximation suggests that, ideally, under low-noise regimes, 1 n ∥ y -ˆ x ∥ 2 2 is expected to be close to σ 2 z . Based on this insight, we propose a simple and effective heuristic for choosing λ : select λ such that 1 n ∥ y -ˆ x ∥ 2 2 is closest to the known noise variance σ 2 z . This procedure can be implemented efficiently via a tree-based search strategy, as described in Algorithm 1. To apply Algorithm 1, one needs an estimate of the noise power σ 2 z . This is a well-studied problem and there exist robust algorithms for estimating the variance of noise [2, 53]. For example, in [2], it is shown that the noise power can be estimated from the median of the absolute differences of wavelet coefficients.

Finally, we observe that the performance of ZS-NCD is relatively robust to the choice of λ . For instance, on the Nuclei dataset (Figure 3), ZS-NCD outperforms the state-of-the-art zero-shot learning-

based denoiser, ZS-Noise2Noise, across a wide range of λ values, for both σ z = 10 and σ z = 20 . A similar approach can also be applied to the case of Poisson noise, as well.

Case II: Poisson noise In the case of Poisson noise, in addition to estimating λ , we need to estimate α , which is used to normalize the measurements, in both the MSE-based and the MLE-based methods. Note that in the case of Poisson noise, E [ y i ] = αx i , and therefore, with high probability, 1 n ∑ n i =1 y i ≈ α 1 n ∑ n i =1 x i . Using this observation, and assuming that 1 n ∑ n i =1 x i ≈ 0 . 5 , we estimate α as ˆ α = 2 n ∑ n i =1 y i . We then use the estimated noise level ˆ α to normalize both MSE and MLE based optimization for denoising Poisson noise. See Table 1 for the result of this noise parameter estimation on a sample image. Having an estimate of noise parameter α , we then follow a similar procedure we used in the case of AWGN to set the parameter λ . Specifically, we write the MSE between normalized y and the denoised image ˆ x as

<!-- formula-not-decoded -->

Again, assuming that we are in a low-noise regime, i.e., the second and the third terms in (11) are close to zero, and using 1 n ∑ n i =1 x i ≈ 0 . 5 approximation, it follows that 1 n E [ ∥ y /α -ˆ x ∥ 2 2 ] ≈ 1 2 α . This implies that, in the case of Poisson noise, we can still use Algorithm 1 to find λ , after updating ∥ ˆ x ( k ) -y ∥ 2 &gt; nσ 2 z to ∥ ˆ x ( k ) -y ∥ 2 &gt; 1 2 α . We empirically observe that MSE( y /α , ˆ x ) in training our networks is close to 1 2 α as reported in Table 1, which indicates that 1 2 α is a good approximation of the MSE that can be used as a threshold for selecting λ . When α is not known, we obtain the estimate ˆ α , and use 1 2ˆ α as a valid threshold to set λ .

Table 1: Analyzing the estimation of Poisson noise parameter for Barbara image in Set11 (MSE values are reported in terms of PSNR).

|   true α |   estimated ˆ α |   empirical MSE( y /α , ˆ x ) [dB] |   1 / (2 α ) [dB] |   1 / (2ˆ α ) [dB] |
|----------|-----------------|------------------------------------|-------------------|--------------------|
|       25 |           23.02 |                              17.12 |             16.98 |              16.63 |
|       50 |           46.05 |                              19.66 |             20    |              19.64 |

## 5 Experiments

In this section, we evaluate the denoising performance of ZS-NCD on both synthetic and real-world noise, across natural and microscopy images. We compare against representative zero-shot denoisers, including both traditional and learning-based methods. All baselines are dataset-free, i.e., they operate solely on the noisy image to be denoised. For non-learning methods, we include JPEG-2K and BM3D. Although rarely used as a denoising baseline, JPEG-2K provides a useful point of comparison from the perspective of compression-based denoising, as it represents a fixed, pre-defined compression code. For learning-based methods, we evaluate Deep Image Prior (DIP) [21], Deep Decoder (DD) [22], Zero-Shot Noise2Self (ZS-N2S) [12], Self2Self (S2S) [23], and Zero-Shot Noise2Noise (ZS-N2N) [24].

Due to instability in training for several baselines, we report their best achieved performance (with early stopping or model selection), whereas ZS-NCD is evaluated at its final training iteration , without manual tuning or stopping criteria.

Natural images with synthetic noise We consider two synthetic noise models, AWGN N (0 , σ 2 z ) , where σ z is the standard deviation of the Gaussian distribution, and Poisson noise defined as Poisson( αx ) , where α is the scale factor. Note that Poisson noise is signal-dependent noise with E [ y ] = α x . To re-scale the noisy image to the range of clean image, we followed the literature by assuming that the the scale α is known, and normalize the noisy image as y /α in the experiments of this section. We evaluate on grayscale Set11 [25], RGB Set13 [54] (center-cropped to 192 × 192 ) and Kodak24 [55] datasets. Table 2 presents the denoising performance of various methods. BM3D achieves the strongest results on grayscale images, though it relies on accurate knowledge of the noise power parameter. Existing learning-based zero-shot denoisers, in contrast, often exhibit inconsistent

Table 2: Denoising performance comparison under AWGN and Poisson Noise, average PSNR(dB) and SSIM are reported. Best results are in bold , second-best are underlined.

| Noise Parameter   |        | AWGN, N (0 ,σ 2 )   | AWGN, N (0 ,σ 2 )   | AWGN, N (0 ,σ 2 )   | Poisson, Poisson( α x ) /α   | Poisson, Poisson( α x ) /α   | Poisson, Poisson( α x ) /α   |
|-------------------|--------|---------------------|---------------------|---------------------|------------------------------|------------------------------|------------------------------|
| σ or α            | Method | Set11               | Set13               | Kodak24             | Set11                        | Set13                        | Kodak24                      |
| 15                | JPEG2K | 27.45 / 0.7699      | 26.69 / 0.7543      | 27.86 / 0.7457      | 22.35 / 0.5882               | 21.76 / 0.5494               | 22.56 / 0.5249               |
| 15                | BM3D   | 32.22 / 0.8991      | 31.15 / 0.8808      | 32.37 / 0.8754      | 26.66 / 0.7505               | 25.64 / 0.6912               | 27.04 / 0.6900               |
| 15                | DIP    | 29.11 / 0.7990      | 30.31 / 0.8570      | 31.42 / 0.8454      | 23.69 / 0.5863               | 25.14 / 0.6916               | 26.37 / 0.6761               |
| 15                | DD     | 28.83 / 0.8215      | 29.22 / 0.8371      | 28.71 / 0.8016      | 24.37 / 0.6629               | 24.96 / 0.7006               | 25.59 / 0.6679               |
| 15                | S2S    | 26.81 / 0.8158      | 20.61 / 0.6879      | 23.08 / 0.7695      | 21.75 / 0.6872               | 19.23 / 0.6553               | 22.52 / 0.7418               |
| 15                | ZS-N2S | 28.92 / 0.8495      | 18.18 / 0.5690      | 18.68 / 0.5540      | 25.06 / 0.7051               | 21.23 / 0.6066               | 22.24 / 0.6170               |
| 15                | ZS-N2N | 30.01 / 0.8169      | 30.95 / 0.8701      | 32.30 / 0.8650      | 24.04 / 0.5766               | 25.37 / 0.6878               | 26.80 / 0.6757               |
| 15                | ZS-NCD | 31.35 / 0.8580      | 31.93 / 0.8983      | 33.18 / 0.9026      | 25.65 / 0.7132               | 26.44 / 0.7434               | 27.64 / 0.7432               |
| 25                | JPEG2K | 24.91 / 0.6997      | 24.32 / 0.6676      | 25.43 / 0.6550      | 23.03 / 0.6108               | 22.65 / 0.5952               | 23.58 / 0.5680               |
| 25                | BM3D   | 29.79 / 0.8523      | 28.81 / 0.8213      | 29.98 / 0.8092      | 22.70 / 0.5741               | 22.17 / 0.5992               | 24.13 / 0.5931               |
| 25                | DIP    | 26.60 / 0.7128      | 27.85 / 0.7837      | 28.90 / 0.7738      | 24.94 / 0.6512               | 26.13 / 0.7289               | 27.49 / 0.7243               |
| 25                | DD     | 26.93 / 0.7530      | 27.40 / 0.7832      | 27.62 / 0.7496      | 25.48 / 0.7022               | 26.04 / 0.7373               | 26.56 / 0.7060               |
| 25                | S2S    | 23.32 / 0.7306      | 17.95 / 0.5998      | 20.69 / 0.6949      | 23.40 / 0.7355               | 20.18 / 0.6927               | 23.09 / 0.7674               |
| 25                | ZS-N2S | 27.30 / 0.7971      | 20.39 / 0.6200      | 20.89 / 0.6156      | 26.01 / 0.7478               | 21.19 / 0.6312               | 21.47 / 0.6277               |
| 25                | ZS-N2N | 27.18 / 0.7173      | 28.36 / 0.8001      | 29.54 / 0.7798      | 25.40 / 0.6432               | 26.75 / 0.7455               | 28.21 / 0.7374               |
| 25                | ZS-NCD | 28.93 / 0.8079      | 29.33 / 0.8351      | 30.60 / 0.8144      | 27.10 / 0.7431               | 27.60 / 0.7827               | 28.77 / 0.7677               |
| 50                | JPEG2K | 22.05 / 0.5794      | 21.43 / 0.5295      | 22.17 / 0.5055      | 24.77 / 0.6811               | 24.25 / 0.6696               | 25.52 / 0.6608               |
| 50                | BM3D   | 26.56 / 0.7619      | 25.78 / 0.7134      | 27.06 / 0.7047      | 23.09 / 0.5787               | 23.00 / 0.6281               | 24.49 / 0.6008               |
|                   | DIP    | 23.46 / 0.5783      | 24.82 / 0.6748      | 25.90 / 0.6494      | 26.30 / 0.7004               | 27.72 / 0.7845               | 29.12 / 0.7845               |
|                   | DD     | 24.01 / 0.6584      | 24.56 / 0.6779      | 24.98 / 0.6413      | 26.87 / 0.7455               | 27.43 / 0.7867               | 27.71 / 0.7543               |
|                   | S2S    | 17.41 / 0.5200      | 14.21 / 0.3938      | 17.00 / 0.5325      | 25.70 / 0.7896               | 21.75 / 0.7365               | 23.88 / 0.8014               |
|                   | ZS-N2S | 24.74 / 0.6883      | 20.62 / 0.5880      | 20.05 / 0.5774      | 27.08 / 0.7855               | 20.75 / 0.6033               | 20.25 / 0.5993               |
|                   | ZS-N2N | 23.52 / 0.5457      | 24.67 / 0.6444      | 25.82 / 0.6151      | 27.26 / 0.7216               | 28.57 / 0.8112               | 30.13 / 0.8076               |
|                   | ZS-NCD | 25.58 / 0.7144      | 25.87 / 0.7269      | 27.89 / 0.7464      | 28.44 / 0.7914               | 29.09 / 0.8223               | 30.60 / 0.8235               |

performance across noise levels and image resolutions. For example, ZS-N2S and Self2Self degrade on high-resolution images, likely due to the limitations of training with masked pixels. ZS-N2N performs well on high-resolution images from Kodak24 but suffers on lower-resolution images in Set13 ( 192 × 192 ), as it is trained to map between two downscaled versions of the same noisy image. In comparison, ZS-NCD maintains robust performance across different noise levels and image sizes. The more realistic case of not having access to the noise parameter α was discussed in Section 4. In both noise regimes, we use MSE as the loss function. However, for Poisson noise, minimizing the negative log-likelihood is also a natural choice. We defer the results using this loss to Appendix B.3.

Table 3: First 2 rows: Denoising performance (average PSNR / SSIM of 6 images) under AWGN N (0 , σ 2 I ) on Mouse Nucle fluorescence microscopy images (image size 128 × 128 ). Noise levels are 10 and 20. Last row: Real camera denoising performance on camera image dataset: PolyU. The images are cropped into size of 512 × 512 . We report the average PSNR / SSIM of 6 random images.

| σ       | JPEG2K         | BM3D           | DIP            | DD             | ZS-N2N         | ZS-N2S         | S2S            | ZS-NCD         |
|---------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| 10      | 32.89 / 0.8294 | 38.65 / 0.9640 | 36.43 / 0.8789 | 37.33 / 0.9533 | 36.17 / 0.9319 | 31.26 / 0.8812 | 12.63 / 0.2966 | 38.23 / 0.9508 |
| 20      | 28.57 / 0.6986 | 34.96 / 0.9296 | 32.32 / 0.7889 | 33.50 / 0.9092 | 32.25 / 0.8532 | 30.41 / 0.8600 | 10.09 / 0.1559 | 34.71 / 0.9093 |
| Unknown | 32.89 / 0.8294 | 35.71 / 0.9506 | 35.43 / 0.9408 | 34.83 / 0.9395 | 34.07 / 0.9028 | 23.61 / 0.8344 | 35.66 / 0.9527 | 35.84 / 0.9534 |

Fluorescence microscopy and real camera images To evaluate performance in low-data and domain-shift settings, we test ZS-NCD on Mouse Nuclei fluorescence microscopy images [56], which differ significantly from natural images in structure and texture. We also assess real-world denoising using the PolyU dataset [57], which contains high-resolution images captured by Canon, Nikon, and Sony cameras. Ground-truth images are obtained by averaging multiple captures, while the noisy inputs are single-shot acquisitions. Results are shown in Table 3. ZS-NCD consistently outperforms other learning-based zero-shot denoisers, demonstrating robustness to unknown noise models and non-natural image distributions.

Table 4: Comparison of AWGN denoising of Conv and MLP based ZS-NCD on Set11.

| ZS-NCD   | σ = 25         | σ = 50         |
|----------|----------------|----------------|
| Conv     | 28.93 / 0.8079 | 25.58 / 0.7144 |
| MLP      | 29.52 / 0.8363 | 25.89 / 0.7306 |

<!-- image -->

1

2

3

4

5

6

7

8

Figure 4: Denoising Parrot using ZS-NCD, where only a single pixel from each overlapping patch (stride 1) is retained after compression. (AWGN, σ z = 25 .) Each heatmap value indicates the PSNR achieved when denoising is based solely on the pixel at that specific location within each patch.

Robustness to overfitting. Most learning-based zero-shot methods are prone to overfitting due to the lack of clean targets and the use of overparameterized networks. In contrast, ZS-NCD, grounded in compression-based denoising theory, overcomes this issue given the entropy constraint. To further highlight this key aspect of ZS-NCD, we replace the convolutional encoder-decoder ( ≈ 0 . 4 M params) with a fully connected MLP ( ≈ 2 . 3 Mparams) and observe that, instead of degradation, the performance improves using the same λ (see Table 4).

Effect of overlapping patch aggregation. As described in Section 4 and illustrated in Fig. 1, ZS-NCD denoises each pixel by aggregating outputs from overlapping patches, where each patch is first compressed and then decompressed using a learned neural compression model. Intuitively, one might expect the most accurate reconstruction for a given pixel to come from the patch in which it lies at the center, as this location benefits from the largest available spatial context, which has been observed in [58].

This observation leads to the question: Does averaging over overlapping reconstructions improve denoising quality, or would it suffice to use only the patch where pixel appears at a fixed position (e.g., the center)? From a computational perspective, both strategies are equivalent, since in both methods every patch is processed, but in averaging scheme, each patch contributes to all the pixels it covers.

To investigate this, we conducted an ablation in which, instead of averaging, each pixel ( i, j ) is reconstructed solely from one of the k × k patches in which it appears, using a fixed location in the patch (e.g., top-left, center, etc.). The results are shown in Fig. 4, where each heatmap entry reports the PSNR obtained by using only that specific location in the patch for reconstruction. As expected, performance is best when the pixel is centrally located, and degrades as it moves toward the patch boundaries.

However, the key observation is that averaging across all overlapping reconstructions yields a substantial performance gain. For instance, in denoising Parrot (from Set11 dataset), the best singlelocation reconstruction achieves 25.90 dB (center), while averaging achieves 28.14 dB, a gain of over 2 dB. This highlights the denoising benefit of combining multiple noisy views of each pixel, consistent with principles from ensembling and variance reduction.

## 6 Conclusions

We have studied maximum likelihood compression-based denoising, and provided theoretical characterization of its performance under both AWGN and Poisson noise. Furthermore, we introduced ZS-NCD, a new zero-shot neural-compression-based denoising and demonstrated that it achieves state-of-the-art performance among zero-shot methods, in both AWGN and Poisson denoising.

The presented theoretical results are derived by assuming a fixed (e.g., pre-trained/defined) compression code. Extending these results to the case of zero-shot learned compression codes is an interesting direction for future research.

## Acknowledgment

A.Z., X.C., S.J. were supported by NSF CCF-2237538.

## References

- [1] Norbert Wiener. Extrapolation, interpolation, and smoothing of stationary time series . The MIT press, 1964.
- [2] David L Donoho and Iain M Johnstone. Ideal spatial adaptation by wavelet shrinkage. biometrika , 81(3):425-455, 1994.
- [3] Stéphane Mallat. A wavelet tour of signal processing . Elsevier, 1999.
- [4] D. L. Donoho. De-noising by soft-thresholding. IEEE transactions on information theory , 41(3):613-627, 2002.
- [5] Javier Portilla, Vasily Strela, Martin J Wainwright, and Eero P Simoncelli. Image denoising using scale mixtures of gaussians in the wavelet domain. IEEE Transactions on Image processing , 12(11):1338-1351, 2003.
- [6] Michael Elad and Michal Aharon. Image denoising via sparse and redundant representations over learned dictionaries. IEEE Transactions on Image processing , 15(12):3736-3745, 2006.
- [7] Stefan Roth and Michael J Black. Fields of experts. International Journal of Computer Vision , 82:205-229, 2009.
- [8] Shuhang Gu, Lei Zhang, Wangmeng Zuo, and Xiangchu Feng. Weighted nuclear norm minimization with application to image denoising. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 2862-2869, 2014.
- [9] Shakarim Soltanayev and Se Young Chun. Training deep learning based denoisers without ground truth data. Advances in neural information processing systems , 31, 2018.
- [10] Jaakko Lehtinen, Jacob Munkberg, Jon Hasselgren, Samuli Laine, Tero Karras, Miika Aittala, and Timo Aila. Noise2noise: Learning image restoration without clean data. arXiv preprint arXiv:1803.04189 , 2018.
- [11] Alexander Krull, Tim-Oliver Buchholz, and Florian Jug. Noise2void-learning denoising from single noisy images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 2129-2137, 2019.
- [12] Joshua Batson and Loic Royer. Noise2self: Blind denoising by self-supervision. In International conference on machine learning , pages 524-533. PMLR, 2019.
- [13] Kwanyoung Kim and Jong Chul Ye. Noise2score: tweedie's approach to self-supervised image denoising without clean images. Advances in Neural Information Processing Systems , 34:864-874, 2021.
- [14] David Leigh Donoho. The kolmogorov sampler . Department of Statistics, Stanford University, 2002.
- [15] Tsachy Weissman and Erik Ordentlich. The empirical distribution of rate-constrained source codes. IEEE transactions on information theory , 51(11):3718-3733, 2005.
- [16] Johannes Ballé, Philip A Chou, David Minnen, Saurabh Singh, Nick Johnston, Eirikur Agustsson, Sung Jin Hwang, and George Toderici. Nonlinear transform coding. IEEE Journal of Selected Topics in Signal Processing , 15(2):339-353, 2020.
- [17] Yibo Yang, Stephan Mandt, Lucas Theis, et al. An introduction to neural data compression. Foundations and Trends® in Computer Graphics and Vision , 15(2):113-200, 2023.
- [18] Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, and Karen Egiazarian. Image denoising by sparse 3-d transform-domain collaborative filtering. IEEE Transactions on image processing , 16(8):2080-2095, 2007.
- [19] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: Efficient transformer for high-resolution image restoration. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 5728-5739, 2022.

- [20] David S Taubman, Michael W Marcellin, and Majid Rabbani. Jpeg2000: Image compression fundamentals, standards and practice. Journal of Electronic Imaging , 11(2):286-287, 2002.
- [21] Ulyanov Dmitry, Andrea Vedaldi, and Lempitsky Victor. Deep image prior. International Journal of Computer Vision , 128(7):1867-1888, 2020.
- [22] Reinhard Heckel and Paul Hand. Deep decoder: Concise image representations from untrained non-convolutional networks. In International Conference on Learning Representations , 2019.
- [23] Yuhui Quan, Mingqin Chen, Tongyao Pang, and Hui Ji. Self2self with dropout: Learning self-supervised denoising from single image. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 1890-1898, 2020.
- [24] Youssef Mansour and Reinhard Heckel. Zero-shot noise2noise: Efficient image denoising without any data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 14018-14027, 2023.
- [25] Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang. Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. IEEE transactions on image processing , 26(7):3142-3155, 2017.
- [26] Yaochen Xie, Zhengyang Wang, and Shuiwang Ji. Noise2same: Optimizing a self-supervised bound for image denoising. Advances in neural information processing systems , 33:2032020330, 2020.
- [27] Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Deep image prior. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 9446-9454, 2018.
- [28] Jason Lequyer, Reuben Philip, Amit Sharma, Wen-Hsin Hsu, and Laurence Pelletier. A fast blind zero-shot denoiser. Nature Machine Intelligence , 4(11):953-963, 2022.
- [29] Jibo Bai, Daqi Zhu, and Mingzhi Chen. Dual-sampling noise2noise: Efficient single image denoising. IEEE Transactions on Instrumentation and Measurement , 2025.
- [30] Qing Ma, Junjun Jiang, Xiong Zhou, Pengwei Liang, Xianming Liu, and Jiayi Ma. Pixel2pixel: A pixelwise approach for zero-shot single image denoising. IEEE Transactions on Pattern Analysis and Machine Intelligence , 2025.
- [31] Xiaoxiao Ma, Zhixiang Wei, Yi Jin, Pengyang Ling, Tianle Liu, Ben Wang, Junkang Dai, and Huaian Chen. Masked pre-training enables universal zero-shot denoiser. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [32] Johannes Ballé, Valero Laparra, and Eero P. Simoncelli. End-to-end optimized image compression. In International Conference on Learning Representations , 2017.
- [33] Lucas Theis, Wenzhe Shi, Andrew Cunningham, and Ferenc Huszár. Lossy image compression with compressive autoencoders. In International Conference on Learning Representations , 2017.
- [34] Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, and Nick Johnston. Variational image compression with a scale hyperprior. In International Conference on Learning Representations , 2018.
- [35] David Minnen, Johannes Ballé, and George D Toderici. Joint autoregressive and hierarchical priors for learned image compression. In Advances in neural information processing systems , volume 31, 2018.
- [36] Yinhao Zhu, Yang Yang, and Taco Cohen. Transformer-based transform coding. In International conference on learning representations , 2022.
- [37] Jinming Liu, Heming Sun, and Jiro Katto. Learned image compression with mixed transformercnn architectures. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 14388-14397, 2023.

- [38] Eirikur Agustsson, David Minnen, Nick Johnston, Johannes Balle, Sung Jin Hwang, and George Toderici. Scale-space flow for end-to-end optimized video compression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 8503-8512, 2020.
- [39] Fabian Mentzer, George Toderici, David Minnen, Sergi Caelles, Sung Jin Hwang, Mario Lucic, and Eirikur Agustsson. VCT: A video compression transformer. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems , 2022.
- [40] Jiahao Li, Bin Li, and Yan Lu. Neural video compression with feature modulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 26099-26108, 2024.
- [41] Michela Testolina, Evgeniy Upenik, and Touradj Ebrahimi. Towards image denoising in the latent space of learning-based compression. In Applications of Digital Image Processing XLIV , volume 11842, pages 412-422. SPIE, 2021.
- [42] Saeed Ranjbar Alvar, Mateen Ulhaq, Hyomin Choi, and Ivan V Baji´ c. Joint image compression and denoising via latent-space scalability. Frontiers in Signal Processing , 2:932873, 2022.
- [43] Benoit Brummer and Christophe De Vleeschouwer. On the importance of denoising when learning to compress images. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision , pages 2440-2448, 2023.
- [44] Yuxin Xie, Li Yu, Farhad Pakdaman, and Moncef Gabbouj. Joint end-to-end image compression and denoising: Leveraging contrastive learning and multi-scale self-onns. arXiv preprint arXiv:2402.05582 , 2024.
- [45] Naoki Saito. Simultaneous noise suppression and signal compression using a library of orthonormal bases and the minimum description length criterion. In Wavelet Analysis and Its Applications , volume 4, pages 299-324. Elsevier, 1994.
- [46] S Grace Chang, Bin Yu, and Martin Vetterli. Image denoising via lossy compression and wavelet thresholding. In Proceedings of International Conference on Image Processing , volume 1, pages 604-607. IEEE, 1997.
- [47] S Grace Chang, Bin Yu, and Martin Vetterli. Adaptive wavelet thresholding for image denoising and compression. IEEE transactions on image processing , 9(9):1532-1546, 2000.
- [48] Balas K Natarajan. Filtering random noise from deterministic signals via data compression. IEEE transactions on signal processing , 43(11):2595-2605, 1995.
- [49] Léo Larigauderie, Michela Testolina, and Touradj Ebrahimi. On combining denoising with learning-based image decoding. In Applications of Digital Image Processing XLV , volume 12226, pages 193-206. SPIE, 2022.
- [50] Zhihao Li, Yufei Wang, Alex Kot, and Bihan Wen. Compress clean signal from noisy raw image: A self-supervised approach. In Forty-first International Conference on Machine Learning , 2024.
- [51] Ali Zafari, Xi Chen, and Shirin Jalali. Decompress: Denoising via neural compression. arXiv preprint arXiv:2503.22015 , 2025.
- [52] Iain M. Johnstone. Gaussian estimation: Sequence and wavelet models . Unpublished Book, 2017.
- [53] Guangyong Chen, Fengyuan Zhu, and Pheng Ann Heng. An efficient statistical method for image noise level estimation. In Proceedings of the IEEE international conference on computer vision , pages 477-485, 2015.
- [54] Roman Zeyde, Michael Elad, and Matan Protter. On single image scale-up using sparserepresentations. In International conference on curves and surfaces , pages 711-730. Springer, 2010.
- [55] Eastman Kodak. Kodak lossless true color image suite (PhotoCD PCD0992). URL http://r0k.us/graphics/kodak , 6:2, 1993.

- [56] Tim-Oliver Buchholz, Mangal Prakash, Deborah Schmidt, Alexander Krull, and Florian Jug. Denoiseg: joint denoising and segmentation. In European Conference on Computer Vision , pages 324-337. Springer, 2020.
- [57] Jun Xu, Hui Li, Zhetong Liang, David Zhang, and Lei Zhang. Real-world noisy image denoising: A new benchmark. arXiv preprint arXiv:1804.02603 , 2018.
- [58] Dailan He, Yaoyan Zheng, Baocheng Sun, Yan Wang, and Hongwei Qin. Checkerboard context model for efficient learned image compression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 14771-14780, 2021.

## A Proofs

## A.1 Auxiliary lemmas

Before stating the proofs of the mains theorems, here we state some lemmas that will be used later in the proofs.

Lemma 1. Assume that 0 &lt; α m ≤ α 1 , α 2 ≤ α M &lt; ∞ . Then,

<!-- formula-not-decoded -->

Lemma 2. Consider independent Poisson random variables Y 1 , . . . , Y n , where Y i ∼ Poisson( α i ) . Consider w 1 , . . . , w n ∈ R . Let σ 2 n = ∑ n i =1 w 2 i α i and w M ≜ max i ∈{ 1 ,...,n } | w i | . Then, for any t ∈ [0 , 3 σ 2 n 2 w M ] ,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

## A.1.1 Proof of Lemma 1

Proof.

<!-- formula-not-decoded -->

Using the Taylor's theorem,

Letting u = α 2 -α 1 α 1 , for α ∈ (0 , u ) ,

<!-- formula-not-decoded -->

Combining (14), (15) and (16) yields the desired result.

## A.1.2 Proof of Lemma 2

Proof. Define and

<!-- formula-not-decoded -->

where f ( u ) = log(1 + u ) and α ∈ (0 , u ) . Note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consider s &gt; 0 , then using the Chernoff bound, we have

<!-- formula-not-decoded -->

Note that for u ∈ ( -1 , 1) , e u -1 -u ≤ u 2 2(1 -u/ 3) . Assuming that s ≤ 1 w M , then | sw i | &lt; 1 , for all i . Therefore,

<!-- formula-not-decoded -->

Evaluating this bound at s = t σ 2 n + w M t/ 3 , since 1 -sw M / 3 = σ 2 n σ 2 n + w M t/ 3 , it follows that

<!-- formula-not-decoded -->

To derive the other bound, we can follow the same steps and apply Chernoff bound as done in (17) to get

<!-- formula-not-decoded -->

We now use the inequality for u ∈ ( -1 , 1) :

<!-- formula-not-decoded -->

Assume s ≤ 1 w M so that s | w i | ≤ 1 for all i . Then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Setting s = 3 t 2 σ 2 n , which satisfies s ≤ 1 w M ,

<!-- formula-not-decoded -->

Hence,

## A.2 Proof of Theorem 1

Proof. Recall that y = x + z , with z is i.i.d. N (0 , σ 2 z ) , and

<!-- formula-not-decoded -->

Since both ˆ x , ˜ x are in C , and

Therefore, and finally,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let e = ˆ x -x denote the error of the compression-based estimate of ground truth x from its noisy version y , and d = ˜ x -x denote the distortion from the compressing the ground truth x with the compression code C , then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For any possible reconstruction c ∈ C , we define error vector e ( c ) = c -x . Given t 1 , t 2 &gt; 0 , define event E 1 and E 2 as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

respectively. Conditioned on E 1 ∩ E 2 , it follows from (30) that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last line follows because √ a + b ≤ √ a + √ b , for all a, b &gt; 0 . To finish the proof we need to bound P (( E 1 ∩ E 2 ) c ) and set parameters t 1 and t 2 .

Note that for each c , e ( c ) ∥ e ( c ) ∥ is a unit vector in R n . Therefore, ∑ n i =1 z i e ( c ) i ∥ e ( c ) ∥ ∼ N (0 , σ 2 z ) . Hence,

<!-- formula-not-decoded -->

Therefore, applying the union bound and noting that |C| ≤ 2 R ,

<!-- formula-not-decoded -->

and

For η ∈ (0 , 1) , set and

Then, bits.

2. Quantize the values of the non-zero coordinates such that overall distortion δ is achieved. To achieve this goal it quantizes each non-zero coordinate of x into b bits. Let [ x i ] b denote the b -bit quantized version of x i . Then, | x i -[ x i ] b | ≤ 2 -b . Therefore, the overall ℓ 2 distortion can be bounded as

Choosing

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using the selected values of t 1 and t 2 in (36) yields the desired result, i.e.,

<!-- formula-not-decoded -->

where we have used the fact that 1 n ∥ d ∥ 2 2 ≤ δ .

## A.3 Proof of Corollary 1

Proof. We start by designing a lossy compression code for the set of signals in Q n , defined as

<!-- formula-not-decoded -->

For a k -sparse x ∈ Q n , let x ( k ) ∈ R k denote the k -dimensional vector derived from the non-zero coordinates of x . We define a lossy compression code ( f, g ) that achieves distortion δ . Specifically, given a k -sparse x ∈ Q n , the encoder operates as follows:

1. Encode the number of non-zero entries and their locations. This requires at most

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It can be observed that overall the number of bits required for describing the signals in Q n within distortion δ can be bounded as

<!-- formula-not-decoded -->

where c k = log 2 k + k (log 2 e + 1) . Using the defined lossy compression code to solve (1) and applying Theorem 1, it follows that, with a probability larger than 1 -2 -ηR +2 ,

<!-- formula-not-decoded -->

where C = 2(1 + 2 √ η ) √ 2 ln 2 . Let

Then, where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, since R ≥ k log 2 ( n k ) , and

<!-- formula-not-decoded -->

## A.4 Proof of Theorem 2

Proof. Recall that arg min c ∈C L ( c ; y ) . Let

<!-- formula-not-decoded -->

Since both ˆ x and ˜ x are in C , we have L (ˆ x ; y ) ≤ L (˜ x ; y ) , or

<!-- formula-not-decoded -->

Given the input signal x ∈ R n and c ∈ C , let Poisson( α x ) and Poisson( α c ) denote the distributions corresponding to independent Poisson random variables with respective means αx i and αc i . Note that

<!-- formula-not-decoded -->

Adding ∑ i ( -αx i + αx i log x i ) to the both sides of (42), it follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Given t 1 , t 2 &gt; 0 , define events E 1 and E 2 as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

respectively. Conditioned on E 1 ∩ E 2 ,

D KL (Poisson( α x ) ∥ Poisson( α ˆ x )) ≤ D KL (Poisson( α x ) ∥ Poisson( α ˜ x )) + t 1 + t 2 , (48) and consequently from Lemma 1,

<!-- formula-not-decoded -->

To finish the proof, we bound P ( E 1 ∩ E 2 ) and set t 1 and t 2 .

To bound P ( E c 1 ) , we apply Lemma 2, where for each c , we set w i ( c ) = log 1 c i . Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and where

Therefore, using the union bound, it follows that

<!-- formula-not-decoded -->

To bound P ( E c 2 ) , we again apply Lemma 2, with w i = log 1 ˜ x i , and derive

<!-- formula-not-decoded -->

Setting t 1 and t 2 such that they are both smaller than 3 nαβ , and noting that x max &lt; 1 , we have

<!-- formula-not-decoded -->

Choosing t 1 = β √ 4 ln 2 nR (1 + η ) α and t 2 = β √ 4 ln 2 nRηα , it follows that

<!-- formula-not-decoded -->

## A.5 Proof of Theorem 3

Proof. Recall that

<!-- formula-not-decoded -->

Following the similar setup as in Section A.2, we get

<!-- formula-not-decoded -->

Defining e , d and e ( c ) , c ∈ C , as done in the proof of Theorem 1, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Conditioned on E 1 ∩ E 2 , it follows that

<!-- formula-not-decoded -->

Using Lemma 2 with y i ∼ Poisson( αx i ) and w i = e ( c ) i , it follows that

<!-- formula-not-decoded -->

where and

Define events and

and

For η ∈ (0 , 1) , set and

Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using the union bound and noting that |C| ≤ 2 R , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Setting t 1 and t 2 such that they are both smaller than 3 nx 2 max α , and noting that x max &lt; 1 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using the selected values of t 1 and t 2 in (57) yields the desired result, i.e.,

<!-- formula-not-decoded -->

Figure 5: Neural compression network used for denoising. Conv and FC denote the convolutiona and fully connected layer, respectively. GDN and ReLU are activation functions.

<!-- image -->

## B Additional experiments and experimental settings

In this section, we provide the details of the networks structures and experimental settings. We also present more experiments for Poisson denoising using MLE and MSE loss functions with unknown noise level.

## B.1 Network structure

For our experiments we used 3 convolutional layers in the encoder with 128 number of channels for the first two layers in the encoder (and the last two layers of decoder), see Figure 5. For color images we choose the number of channels in the last encoder (and first decoder) layer equals to 32, and for grayscale images equals to 16. The MLP-based network of the ablation study in Section 5 has 3 fully connected layers in the encoder with 1024 hidden units for the first two layers in the encoder (and the last two layers of decoder). The number of hidden units in the last encoder (and first decoder) layer equals to 16. As activation function we use GDN [32] for Conv network and ReLU for MLP.

## B.2 More recent improvements on Zero-shot Noise2Noise

Zero-shot Noise2Noise (ZS-N2N) [24] learns a mapping between two downsampled noisy images extracted from the original image. This strategy of generating noisy samples was extended in two recent concurrent works. Dual-sampling Noise2Noise (DS-N2N) [29] identifies a key sub-optimality in ZS-N2N: the network learns to denoise at a lower resolution but is then directly applied to the larger, original resolution. To address this, the authors proposed to apply an additional bicubic upsampling to the downsampled images and train the network on both the low-resolution and the upsampled pairs, which boosts the performance of ZS-N2N.

In Pixel2Pixel [30], the authors identified the local sampling in ZS-N2N insufficient to break the spatial correlation of real-world noise . To address this issue, Pixel2Pixel first finds a 'pixel bank' for each pixel based on the non-local similar patches. Then, by randomly choosing pixels from the bank, it generates multiple noisy pairs to be used as a pseudo-training dataset for zero-shot denoising.

We have compared the performance of both methods under AWGN and Poisson noise in Tables 5 and 6, respectively. Each paper's official code was used to report the numbers in the tables. In AWGN, our compression-based denoiser ZS-NCD outperforms both methods while achieving the second-best performance in Poisson denoising. Pixel2Pixel performs well in Poisson denoising.

Table 5: Average PSNR(dB)/SSIM denoising performance on Kodak24 dataset under AWGN N (0 , σ 2 z ) . Best results are in bold , second-best are underlined.

|                    | σ z = 15       | σ z = 25       | σ z = 50       |
|--------------------|----------------|----------------|----------------|
| ZS-N2N (2023)      | 32.30 / 0.8650 | 29.54 / 0.7798 | 25.82 / 0.6151 |
| DS-N2N (2025)      | 32.31 / 0.8803 | 29.64 / 0.8044 | 25.42 / 0.6378 |
| Pixel2Pixel (2025) | 31.31 / 0.8707 | 29.89 / 0.8098 | 26.55 / 0.6873 |
| ZS-NCD             | 33.18 / 0.9026 | 30.60 / 0.8144 | 27.89 / 0.7464 |

Table 6: Average PSNR(dB)/SSIM denoising performance on Kodak24 dataset under Poison noise, Poisson( α x ) /α . Best results are in bold , second-best are underlined.

|                    | α = 15         | α = 25         | α = 50         |
|--------------------|----------------|----------------|----------------|
| ZS-N2N (2023)      | 26.80 / 0.6757 | 28.21 / 0.7374 | 30.13 / 0.8076 |
| DS-N2N (2025)      | 27.29 / 0.7016 | 28.50 / 0.7540 | 30.27 / 0.8250 |
| Pixel2Pixel (2025) | 28.22 / 0.7390 | 29.26 / 0.7891 | 30.41 / 0.8372 |
| ZS-NCD             | 27.64 / 0.7432 | 28.77 / 0.7677 | 30.60 / 0.8235 |

## B.3 MSE and likelihood estimation under Poisson noise without knowing true α

We compare the MSE and MLE distortion for Poisson denoising in Table 7 using the estimated ˆ α as explained in Section 4.

Table 7: Minimizing Poisson negative log-likelihood (NLL) vs. MSE with estimated ˆ α for Cameraman image in Set11. PSNR / SSIM are reported here.

|   α | MSE (with estimated ˆ α )   | NLL (with estimated ˆ α )   |
|-----|-----------------------------|-----------------------------|
|  15 | 23.41 / 0.7554              | 23.13 / 0.7567              |
|  50 | 25.22 / 0.7961              | 24.88 / 0.7460              |

## B.4 Study on factors in patch-wise compression affecting denoising

In this section, we explain the intuition behind why learning compression networks and denoising on overlapped patches is feasible. The centered pixels in the patches are better compressed as empirically observed in [58], thus they can provide better denoising performance. To study the contribution of each patch containing the single pixel to be denoised we design the experiment that, in the denoising phase, we denoise the overlapped patches, but only a single pixel at the same location from each patch is used to construct the final denoised image, instead of averaging all of them as in (9). We show the denoising performance of each pixel location in Figure 6. The PSNR at each pixel denotes the denoising performance of only using the specific pixel of each overlapped patches with stride 1. We can find that the boundary pixels give lower PSNR, which is consistent with previous research findings that the centered pixels are better compressed. Next, we analyze the effect of patch size in both learning and denoising phases. Given that scaler quantization is applied and the entropy model is learned on latent code of the patches, the compression performance on the latent code is affected by both the patch size and the number of downsampling operations in CNN-based encoder. We design the experiment that 3 downsampling operators are applied to patch size 8 and 16, where the latent code sizes are 1 × 1 × n b and 2 × 2 × n b respectively, where the denoising performance at each pixel location is in Figure 6 ( Left ) and ( Middle ), and if we increase the downsampling to 4 for patch size 16, which results in the latent code size to be 1 × 1 × n b , the denoising performance is in Figure 6 ( Right ). We find that spatial size of the latent code to be quantized matters given the scaler quantization limitation, the reconstructed output by the decoder will be restricted by the only correlated latent code as we can observe. Motivated from this, we perform the learning and denoising phases both patch-wise with proper networks structure, all pixels in each patch are used and the overlapped areas are averaged properly to reduce the variance of the compression-based estimates.

## C Additional numerical results

In this section, we provide the full denoising numerical results of the denoisers on all the test images. All the experiments were run on Nvidia RTX 6000 Ada with 48 GB memory. It takes 40 minutes to denoise a grayscale image of size 256 × 256 , and 50 minutes for an RGB image of size 512 × 768 . Adam optimizer is used for training the networks over 20K steps, with initial learning rate of 5 × 10 -3 decreased to 5 × 10 -4 after 16K steps for the Conv-based network. The learning rate for MLP-based networks is 1 × 10 -3 .

<!-- image -->

1

2

3

4

5

6

7

8

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

Figure 6: Denoising AWGN ( σ z = 25 ) of image Parrot by only compressing a single pixel in each overlapped patches with stride 1. The PSNR at each pixel denotes the final denoising performance by only compressing the pixel at that specific location in each patch. Left : patch size 8 × 8 , with downsampling factor equals 8 in f θ 1 ; Middle : patch size 16 × 16 , with downsampling factor equals 8 in f θ 1 ; Right : patch size 16 × 16 , with downsampling factor equals 16 in f θ 1 .

## C.1 Set11 Dataset

For noise levels (15 , 25 , 50) we set λ = (300 , 850 , 3000) . Similar to Kodak and other experiments we set training epochs to have 20K steps of gradient back propagation. For Poisson denoising α = (15 , 25 , 50) the λ = (3000 , 1500 , 1000) . We report the detailed results of AWGN denoising in Table 8, and Poisson noise denoising in Table 9.

Table 8: Set11 Denoising performance comparison under AWGN N (0 , σ 2 z I ) .

|    |              | 256 × 256    | 256 × 256    | 256 × 256    | 256 × 256    | 256 × 256                 | 256 × 256    | 256 × 256    | 512 × 512    | 512 × 512    | 512 × 512    | 512 × 512    |              |
|----|--------------|--------------|--------------|--------------|--------------|---------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| σ  | Method       | C.man        | House        | Peppers      | Starfish     | Monarch                   | Airplane     | Parrot       | Barbara      | Boats        | Pirate       | Couple       | Average      |
|    | BM3D         | 31.84/0.8974 | 34.94/0.8870 | 32.82/0.9118 | 31.16/0.9082 | 31.92/0.9409 26.74/0.8166 | 31.09/0.9034 | 31.47/0.9032 | 33.04/0.9253 | 32.12/0.8604 | 31.94/0.8726 | 32.11/0.8795 | 32.22/0.8991 |
|    | JPEG2K       | 27.12/0.7474 | 29.48/0.7621 | 27.96/0.7907 | 26.75/0.8077 |                           | 26.58/0.7664 | 27.30/0.7778 | 26.76/0.7690 | 27.87/0.7390 | 27.92/0.7471 | 27.44/0.7449 | 27.45/0.7699 |
|    | DIP          | 27.94/0.7417 | 31.39/0.8111 | 29.80/0.8273 | 29.58/0.8605 | 29.93/0.8767              | 28.14/0.8047 | 28.37/0.7794 | 27.65/0.7538 | 29.48/0.7798 | 29.27/0.7817 | 28.65/0.7727 | 29.11/0.7990 |
|    | DD           | 29.41/0.8099 | 32.83/0.8406 | 26.97/0.8488 | 29.39/0.8739 | 30.01/0.8957              | 26.44/0.8228 | 29.32/0.8447 | 24.48/0.7089 | 29.45/0.7883 | 29.78/0.8085 | 29.06/0.7938 | 28.83/0.8215 |
| 15 | ZS-N2N       | 30.14/0.8133 | 32.19/0.8138 | 30.58/0.8264 | 29.52/0.8639 | 30.15/0.8551              | 29.98/0.8298 | 30.19/0.8290 | 27.70/0.7772 | 30.06/0.7900 | 30.06/0.7957 | 29.59/0.7913 | 30.01/0.8169 |
|    | ZS-N2S       | 27.66/0.8272 | 31.08/0.8442 | 29.46/0.8675 | 28.83/0.8810 | 28.77/0.8961              | 27.34/0.8591 | 27.67/0.8528 | 28.75/0.8534 | 29.52/0.8139 | 29.41/0.8181 | 29.60/0.8311 | 28.92/0.8495 |
|    | S2S          | 20.29/0.6769 | 32.96/0.8633 | 23.96/0.8387 | 25.50/0.8250 | 30.05/0.9269              | 28.10/0.8611 | 20.20/0.7132 | 30.35/0.8865 | 27.74/0.7871 | 29.97/0.8192 | 25.82/0.7754 | 26.81/0.8158 |
|    | ZS-NCD       | 30.83/0.8554 | 34.45/0.8835 | 32.20/0.8844 | 31.34/0.8749 | 31.83/0.8966              | 30.07/0.8552 | 30.40/0.8464 | 31.14/0.8826 | 31.09/0.8014 | 30.82/0.8302 | 30.67/0.8279 | 31.35/0.8580 |
|    | ZS-NCD (MLP) | 31.18/0.8680 | 34.86/0.8887 | 32.43/0.9009 | 31.37/0.9053 | 32.04/0.9263              | 30.79/0.8848 | 31.02/0.8839 | 32.55/0.9123 | 31.82/0.8468 | 31.35/0.8552 | 31.55/0.8638 | 31.91/0.8851 |
|    | BM3D         | 29.54/0.8499 | 32.79/0.8561 | 30.13/0.8705 | 28.58/0.8578 | 29.36/0.9046              | 28.50/0.8584 | 28.94/0.8561 | 30.63/0.8887 | 29.87/0.8039 | 29.64/0.8082 | 29.74/0.8206 | 29.79/0.8523 |
|    | JPEG2K       | 24.49/0.6976 | 27.26/0.7269 | 24.93/0.7206 | 24.18/0.7167 | 24.06/0.7561              | 23.91/0.7126 | 24.38/0.7162 | 24.09/0.6825 | 25.61/0.6577 | 25.76/0.6566 | 25.28/0.6534 | 24.91/0.6997 |
|    | DIP          | 25.23/0.6043 | 28.93/0.7545 | 27.39/0.7579 | 26.39/0.7777 | 27.47/0.8169              | 25.57/0.6983 | 26.29/0.7409 | 24.75/0.6356 | 27.05/0.6843 | 27.06/0.6857 | 26.52/0.6847 | 26.60/0.7128 |
|    | DD           | 27.24/0.7521 | 30.48/0.8023 | 25.39/0.7591 | 26.86/0.8051 | 27.69/0.8526              | 24.93/0.7120 | 27.29/0.7863 | 23.81/0.6455 | 27.49/0.7163 | 27.87/0.7380 | 27.13/0.7131 | 26.93/0.7530 |
| 25 | ZS-N2N       | 27.32/0.7089 | 29.36/0.7276 | 27.46/0.7240 | 26.61/0.7821 | 27.20/0.7634              | 27.02/0.7463 | 27.16/0.7149 | 25.49/0.6854 | 27.26/0.6779 | 27.48/0.6931 | 26.63/0.6673 | 27.18/0.7173 |
|    | ZS-N2S       | 26.24/0.7843 | 29.23/0.8073 | 27.77/0.8233 | 27.61/0.8463 | 27.35/0.8569              | 25.86/0.8023 | 26.27/0.7997 | 26.43/0.7759 | 28.23/0.7580 | 27.52/0.7526 | 27.74/0.7617 | 27.30/0.7971 |
|    | S2S          | 16.93/0.5998 | 29.12/0.8275 | 21.88/0.7666 | 21.14/0.6974 | 25.93/0.8606              | 24.12/0.7350 | 17.09/0.6069 | 25.79/0.7980 | 23.94/0.7061 | 27.32/0.7403 | 23.29/0.6979 | 23.32/0.7306 |
|    | ZS-NCD       | 28.78/0.8237 | 32.14/0.8547 | 29.62/0.8406 | 28.48/0.8134 | 29.02/0.8494              | 27.77/0.8126 | 28.14/0.8007 | 28.39/0.8192 | 28.85/0.7444 | 28.64/0.7630 | 28.37/0.7648 | 28.93/0.8079 |
|    | ZS-NCD (MLP) | 29.08/0.8259 | 32.63/0.8525 | 29.85/0.8574 | 28.73/0.8547 | 29.42/0.8861              | 28.37/0.8477 | 28.75/0.8431 | 30.01/0.8658 | 29.59/0.7843 | 29.15/0.7859 | 29.10/0.7958 | 29.52/0.8363 |
|    | BM3D         | 26.56/0.7813 | 29.61/0.8029 | 26.85/0.7911 | 25.07/0.7508 | 25.82/0.8192              | 25.29/0.7713 | 26.02/0.7809 | 27.02/0.7888 | 26.76/0.7003 | 26.75/0.6962 | 26.37/0.6977 | 26.56/0.7619 |
|    | JPEG2K       | 21.49/0.5880 | 24.24/0.6444 | 21.72/0.6077 | 21.39/0.5784 | 20.86/0.6414              | 21.11/0.6021 | 21.29/0.6035 | 21.65/0.5377 | 22.83/0.5318 | 23.29/0.5356 | 22.67/0.5025 | 22.05/0.5794 |
|    | DIP          | 22.73/0.5846 | 25.67/0.6475 | 23.81/0.5987 | 22.99/0.6406 | 23.06/0.6293              | 22.64/0.5522 | 23.02/0.5811 | 22.38/0.5316 | 23.90/0.5371 | 24.43/0.5524 | 23.40/0.5064 | 23.46/0.5783 |
|    | DD           | 23.89/0.6487 | 27.27/0.7282 | 22.95/0.7276 | 23.44/0.6700 | 23.55/0.7319              | 22.52/0.6652 | 23.87/0.6471 | 22.72/0.5980 | 24.47/0.6050 | 25.19/0.6340 | 24.30/0.5872 | 24.01/0.6584 |
| 50 | ZS-N2N       | 23.36/0.5324 | 25.17/0.5167 | 23.86/0.5669 | 22.92/0.6186 | 22.95/0.6010              | 23.39/0.5988 | 22.87/0.5136 | 22.62/0.5150 | 23.93/0.5138 | 24.30/0.5330 | 23.30/0.4930 | 23.52/0.5457 |
|    | ZS-N2S       | 24.65/0.6966 | 26.72/0.7091 | 25.24/0.7297 | 24.05/0.7102 | 24.82/0.7618              | 24.04/0.7467 | 24.00/0.7078 | 22.81/0.5916 | 25.46/0.6512 | 25.55/0.6469 | 24.80/0.6197 | 24.74/0.6883 |
|    | S2S          | 14.23/0.4809 | 21.14/0.6396 | 17.80/0.5763 | 15.71/0.4176 | 18.33/0.5955              | 15.70/0.4828 | 13.66/0.4446 | 17.60/0.4883 | 18.69/0.5264 | 19.55/0.5354 | 19.12/0.5325 | 17.41/0.5200 |
|    | ZS-NCD       | 25.55/0.7616 | 28.62/0.7995 | 26.31/0.7604 | 24.59/0.6925 | 25.53/0.7585              | 24.65/0.7338 | 25.31/0.7228 | 24.06/0.6525 | 25.61/0.6538 | 25.92/0.6707 | 25.19/0.6519 | 25.58/0.7144 |
|    | ZS-NCD (MLP) | 25.61/0.7342 | 29.13/0.7881 | 26.30/0.7702 | 24.85/0.7426 | 25.12/0.7795              | 24.84/0.7497 | 25.24/0.7596 | 25.55/0.7146 | 26.26/0.6698 | 26.25/0.6717 | 25.65/0.6560 | 25.89/0.7306 |

Table 9: Set11 Denoising performance comparison under Poisson noise Poisson( α x ) /α .

|    |        | 256 × 256    | 256 × 256    | 256 × 256    | 256 × 256                 | 256 × 256                              | 256 × 256                                           | 256 × 256    | 512 × 512                 | 512 × 512    | 512 × 512    | 512 × 512    |                           |
|----|--------|--------------|--------------|--------------|---------------------------|----------------------------------------|-----------------------------------------------------|--------------|---------------------------|--------------|--------------|--------------|---------------------------|
| α  | Method | C.man        | House        | Peppers      | Starfish                  | Monarch                                | Airplane                                            | Parrot       | Barbara                   | Boats        | Pirate       | Couple       | Average                   |
|    | BM3D   | 26.64/0.7651 | 29.39/0.7668 | 27.13/0.7914 | 24.93/0.7519              | 26.32/0.8265 25.37/0.7968 21.81/0.7813 | 24.79/0.6730 22.22/0.6055 23.05/0.7051 16.18/0.5010 | 26.26/0.7866 | 27.24/0.7860 21.94/0.5688 | 26.82/0.6977 | 27.07/0.7048 | 26.67/0.7056 | 26.66/0.7505 22.35/0.5882 |
|    | JPEG2K | 21.98/0.6032 | 24.35/0.6106 | 22.12/0.6213 | 21.52/0.5887              | 21.24/0.6493                           | 20.87/0.5818                                        | 22.02/0.6378 |                           | 23.09/0.5346 | 23.75/0.5510 | 23.01/0.5237 |                           |
|    | DIP    | 22.85/0.5382 | 26.32/0.6528 | 24.23/0.6138 | 23.23/0.6696              | 23.54/0.6875                           | 22.07/0.4938                                        | 22.81/0.5723 | 22.59/0.5503              | 24.18/0.5533 | 24.95/0.5811 | 23.83/0.5362 | 23.69/0.5863              |
|    | DD     | 24.45/0.6261 | 27.59/0.7453 | 23.20/0.7269 | 23.86/0.7164              | 24.66/0.7286                           |                                                     | 24.38/0.6830 | 22.89/0.6081              | 24.64/0.6023 | 25.61/0.6516 | 24.58/0.5986 | 24.37/0.6629              |
| 15 | ZS-N2N | 24.19/0.5818 | 25.41/0.5346 | 24.65/0.6016 | 23.12/0.6520              | 23.92/0.6441                           | 23.12/0.5565                                        | 23.83/0.5821 | 23.05/0.5503              | 24.40/0.5403 | 24.87/0.5684 | 23.94/0.5305 | 24.04/0.5766              |
|    | ZS-N2S | 24.94/0.7241 | 27.29/0.7317 | 25.71/0.7431 | 24.41/0.7417              |                                        |                                                     | 24.98/0.7315 | 22.87/0.6087              | 26.08/0.6696 | 25.94/0.6655 | 25.09/0.6389 | 25.06/0.7051              |
|    | S2S    | 23.53/0.7325 | 22.01/0.7409 | 22.73/0.7300 | 18.20/0.6010              |                                        |                                                     | 20.27/0.7304 | 22.10/0.7261              | 23.49/0.6529 | 24.72/0.6782 | 24.17/0.6843 | 21.75/0.6872              |
|    | ZS-NCD | 25.73/0.7660 | 28.87/0.8015 | 26.54/0.7745 | 24.65/0.6988              | 25.86/0.7791                           | 24.21/0.6568                                        | 25.52/0.7356 | 24.11/0.6562              | 25.48/0.6510 | 25.93/0.6705 | 25.29/0.6552 | 25.65/0.7132              |
|    | BM3D   | 22.69/0.5154 | 22.82/0.4765 | 22.74/0.5930 | 22.11/0.6947              | 23.44/0.7213                           | 19.60/0.3788                                        | 23.05/0.5991 | 23.09/0.6508              | 22.77/0.5123 | 23.89/0.5979 | 23.45/0.5753 | 22.70/0.5741              |
|    | JPEG2K | 22.54/0.6267 | 24.97/0.6773 | 22.87/0.6076 | 22.26/0.6378              | 22.55/0.6641                           | 21.59/0.5649                                        | 22.62/0.6373 | 22.55/0.5976              | 23.71/0.5685 | 24.20/0.5801 | 23.49/0.5566 | 23.03/0.6108              |
|    | DIP    | 24.21/0.5976 | 27.06/0.6553 | 25.76/0.6945 | 24.41/0.7312              | 25.21/0.7384                           | 23.58/0.6290                                        | 24.69/0.6608 | 23.11/0.5903              | 25.24/0.6130 | 26.10/0.6528 | 24.91/0.5998 | 24.94/0.6512              |
|    | DD     | 25.59/0.6695 | 28.47/0.7606 | 24.20/0.7348 | 25.14/0.7667              | 26.16/0.8022                           | 23.24/0.6306                                        | 25.89/0.7289 | 23.27/0.6257              | 25.84/0.6517 | 26.76/0.6961 | 25.77/0.6573 | 25.48/0.7022              |
| 25 | ZS-N2N | 25.54/0.6334 | 27.14/0.6234 | 25.82/0.6522 | 24.33/0.7158              | 25.51/0.7109                           | 24.53/0.6274                                        | 25.55/0.6617 | 24.09/0.6173              | 25.60/0.6018 | 26.11/0.6354 | 25.16/0.5958 | 25.40/0.6432              |
|    | ZS-N2S | 26.22/0.7776 | 27.81/0.7643 | 26.55/0.7768 | 24.82/0.7795              | 26.48/0.8254                           | 24.77/0.7463                                        | 25.44/0.7839 | 23.24/0.6387              | 27.25/0.7143 | 27.13/0.7200 | 26.44/0.6986 | 26.01/0.7478              |
|    | S2S    | 25.09/0.7572 | 24.10/0.7398 | 24.91/0.7733 | 19.11/0.6491              | 23.64/0.8226                           | 17.93/0.6279                                        | 21.13/0.7692 | 24.01/0.7860              | 25.30/0.7058 | 26.45/0.7232 | 25.77/0.7360 | 23.40/0.7355              |
|    | ZS-NCD | 27.17/0.7635 | 30.09/0.8109 | 27.92/0.7932 | 26.27/0.7600              | 27.28/0.8093                           | 24.93/0.6116                                        | 26.74/0.7551 | 26.24/0.7393              | 27.30/0.6993 | 27.32/0.7192 | 26.83/0.7123 | 27.10/0.7431              |
|    | BM3D   | 22.94/0.5314 | 22.89/0.4548 | 23.22/0.5844 | 23.06/0.7150              | 23.87/0.6990                           | 20.51/0.4115                                        | 23.65/0.6136 | 23.54/0.6545              | 22.82/0.5205 | 23.97/0.6087 | 23.47/0.5723 | 23.09/0.5787              |
|    | JPEG2K | 24.23/0.6635 | 26.87/0.6796 | 24.96/0.7042 | 24.08/0.7240              | 24.05/0.7568                           | 23.40/0.6387                                        | 24.57/0.7077 | 23.95/0.6804              | 25.37/0.6414 | 25.73/0.6483 | 25.25/0.6475 | 24.77/0.6811              |
|    | DIP    | 25.34/0.6348 | 28.88/0.7369 | 27.59/0.7559 | 26.18/0.7891              | 26.58/0.7728                           | 24.69/0.6457                                        | 26.04/0.7062 | 23.88/0.6158              | 26.61/0.6712 | 27.21/0.7015 | 26.34/0.6747 | 26.30/0.7004              |
|    | DD     | 27.24/0.7398 | 30.16/0.7784 | 25.44/0.7615 | 26.78/0.8127              | 27.82/0.8527                           | 24.39/0.6568                                        | 27.39/0.7735 | 23.86/0.6518              | 27.30/0.7096 | 28.03/0.7476 | 27.16/0.7162 | 26.87/0.7455              |
| 50 | ZS-N2N | 27.63/0.7210 | 29.30/0.7113 | 27.92/0.7424 | 26.35/0.7857 26.62/0.8243 | 27.38/0.7723 28.03/0.8624              | 26.26/0.7037                                        | 27.68/0.7443 | 25.50/0.6883 23.60/0.6606 | 27.31/0.6832 | 27.68/0.7075 | 26.84/0.6783 | 27.26/0.7216              |
|    | ZS-N2S | 26.82/0.8041 | 29.32/0.7832 | 27.75/0.8192 |                           |                                        | 26.05/0.8097                                        | 26.56/0.8196 |                           | 27.92/0.7493 | 27.68/0.7557 | 27.54/0.7521 | 27.08/0.7855              |
|    | S2S    | 26.72/0.8220 | 27.19/0.8106 | 27.83/0.8300 | 20.47/0.7126              | 26.38/0.8780                           | 21.09/0.6607                                        | 22.61/0.8001 | 27.07/0.8457              | 27.39/0.7620 | 28.31/0.7754 | 27.64/0.7885 | 25.70/0.7896              |
|    |        |              |              | 29.44/0.8410 |                           |                                        |                                                     |              |                           | 28.16/0.7273 |              |              |                           |
|    | ZS-NCD | 28.24/0.8093 | 31.90/0.8488 |              | 28.02/0.8064              | 28.78/0.8504                           | 26.99/0.7422                                        | 27.93/0.7961 | 27.60/0.7920              |              | 28.11/0.7466 | 27.72/0.7456 | 28.44/0.7914              |

## C.2 Set13 Dataset

All images are center-cropped at size of 192 × 192 . For this set of images we set λ = (100 , 200 , 800) and for noise levels σ z = (15 , 25 , 50) and for Poisson denoising we have λ = (900 , 500 , 200) for noise levels α = (15 , 25 , 50) . We report the detailed results of AWGN denoising in Table 10, and Poisson noise denoising in Table 11.

Table 10: Set13 Denoising performance comparison under AWGN N (0 , σ 2 z I ) .

<!-- image -->

Table 11: Set13 Denoising performance comparison under Poisson noise Poisson( α x ) /α .

<!-- image -->

## C.3 Kodak24 Dataset

For Gaussian denoising λ = (75 , 150 , 750) for noise levels σ z = (15 , 25 , 50) and for Poisson denoising λ = (750 , 300 , 150) for α = (15 , 25 , 50) . For BM3D Poisson denoising of α = (15 , 25 , 50) we set σ BM3D = (50 , 25 , 15) . We report the detailed results of AWGN denoising in Table 12, and Poisson noise denoising in Table 13.

Table 12: Kodak24 Denoising performance comparison under AWGN denoising N (0 , σ 2 z I ) .

<!-- image -->

| Method ( σ ) 01 02 JPEG2K (15) 25.20/0.7120 29.23/0.6996 BM3D (15) 29.46/0.8549 33.06/0.8266 DIP (15) 29.91/0.8724 31.98/0.7986 DD (15) 26.10/0.7586 29.56/0.7297 ZS-N2N (15) 31.13/0.8951 33.19/0.8418 ZS-N2S (15) 18.03/0.4389 25.76/0.6821 S2S (15) 25.73/0.7941 25.01/0.7193 ZS-NCD (15) 31.28/0.9059 33.93/0.8669 JPEG2K (25) 22.70/0.5639 27.54/0.6312 BM3D (25) 26.98/0.7554 31.29/0.7717 DIP (25) 26.90/0.7759 29.87/0.7204 DD (25) 25.26/0.7193 29.52/0.7298 ZS-N2N (25) 28.30/0.8249 30.67/0.7618 ZS-N2S (25) S2S (25) ZS-NCD (25)   | 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 Average 29.81/0.7678 29.09/0.7299 25.55/0.7728 26.71/0.7343 28.97/0.7880 25.24/0.7730 29.29/0.7756 29.27/0.7540 27.34/0.7033 29.28/0.7219 24.73/0.7477 26.65/0.7137 29.23/0.7490 28.29/0.7205 28.86/0.7709 26.54/0.7195 27.76/0.7400 29.28/0.8055 27.21/0.7749 27.81/0.6983 30.82/0.8039 26.38/0.7215 27.86/0.7457 35.19/0.9096 33.30/0.8583 30.37/0.9023 31.00/0.8664 34.59/0.9384 30.37/0.9019 34.59/0.9083 34.38/0.8952 31.70/0.8475 34.12/0.8611 27.97/0.8295 30.60/0.8425 33.84/0.8781 32.69/0.8647 33.46/0.8898 30.48/0.8558 32.18/0.8568 33.84/0.8936 31.46/0.8932 31.60/0.8355 35.90/0.9205 30.83/0.8800 32.37/0.8754 33.53/0.8588 32.07/0.8180 30.00/0.8875 30.85/0.8605 33.21/0.8851 29.54/0.8838 33.00/0.8551 32.71/0.8348 30.58/0.8144 32.79/0.8203 28.16/0.8619 30.37/0.8531 32.22/0.8231 32.26/0.8486 31.99/0.8492 29.78/0.8235 30.96/0.8111 32.66/0.8676 30.77/0.8396 30.82/0.8107 34.09/0.8716 29.94/0.8403 31.42/0.8454 30.98/0.8323 30.11/0.8023 26.17/0.8114 27.28/0.7715 30.92/0.8905 24.65/0.7766 31.34/0.8565 31.57/0.8480 28.45/0.7793 32.09/0.8177 23.42/0.6831 27.77/0.7891 29.31/0.7730 29.31/0.7730 30.93/0.8688 27.14/0.7826 28.25/0.7786 28.87/0.8549 28.03/0.8158 29.23/0.7820 31.88/0.8775 25.72/0.7853 28.71/0.8016 34.04/0.8582 32.93/0.8463 30.38/0.8890 31.96/0.8826 33.51/0.8872 30.77/0.9045 33.75/0.8532 33.01/0.8367 32.15/0.8571 33.48/0.8290 28.98/0.8885 31.14/0.8681 32.96/0.8401 33.23/0.8687 33.21/0.8743 30.44/0.8587 32.65/0.8736 33.86/0.8856 32.51/0.8849 31.83/0.8416 33.85/0.8395 30.29/0.8548 32.30/0.8650 17.06/0.6117 25.30/0.6915 18.35/0.5545 10.16/0.3461 22.51/0.7272 17.54/0.5125 15.42/0.6011 19.79/0.6228 23.15/0.6176 8.06/0.2929 19.24/0.3889 22.86/0.6371 23.91/0.7191 23.05/0.6010 16.88/0.5174 21.44/0.5637 11.59/0.5257 5.47/0.0930 22.77/0.6687 20.22/0.5615 22.33/0.7443 17.36/0.5761 18.68/0.5540 23.30/0.8177 27.41/0.8106 22.73/0.7274 19.32/0.7144 29.75/0.9184 19.82/0.7318 29.36/0.8927 25.72/0.8723 22.71/0.7244 22.15/0.7985 19.75/0.5547 24.55/0.7461 17.09/0.7066 27.15/0.8337 21.30/0.7254 24.15/0.7016 28.58/0.7811 10.63/0.6711 23.49/0.8304 23.60/0.7693 22.17/0.8469 18.37/0.7790 23.08/0.7695 35.61/0.9215 34.19/0.8876 31.63/0.9286 32.07/0.9009 35.34/0.9382 31.27/0.9222 35.03/0.9088 34.71/0.9097 32.40/0.8895 34.89/0.8939 28.61/0.8931 31.95/0.8978 34.37/0.8951 34.04/0.9004 34.04/0.9089 31.25/0.8860 33.26/0.8956 34.13/0.9096 32.45/0.8973 32.52/0.8762 35.83/0.9236 31.48/0.9046 33.18/0.9026 27.86/0.7130 27.10/0.6704 25.55/0.7728 24.21/0.6104 26.01/0.6863 22.11/0.6519 26.88/0.7440 26.57/0.7022 24.92/0.5912 27.20/0.6589 21.73/0.5638 24.45/0.5960 26.89/0.7150 26.60/0.6330 26.37/0.6632 24.10/0.5928 25.43/0.6575 26.43/0.7676 24.76/0.7002 25.40/0.5919 28.75/0.7779 23.70/0.6015 25.43/0.6550 32.74/0.8618 31.23/0.7994 27.56/0.8236 28.42/0.7789 31.87/0.9026 27.74/0.8497 32.20/0.8715 31.92/0.8480 29.29/0.7690 32.07/0.8068 25.21/0.6973 28.19/0.7513 31.75/0.8337 30.33/0.7813 31.03/0.8382 27.82/0.7616 30.14/0.7936 31.72/0.8522 28.89/0.8366 29.37/0.7491 33.59/0.8899 28.09/0.7966 29.98/0.8092 30.91/0.7859 30.03/0.7510 26.40/0.7875 27.91/0.7697 30.77/0.8496 26.51/0.8059 30.41/0.7959 30.28/0.7784 27.77/0.7115 30.79/0.7586 25.52/0.7763 27.74/0.7625 30.16/0.7568 29.96/0.7803 29.72/0.7923 26.80/0.7322 28.77/0.7591 30.54/0.8216 28.41/0.7963 28.38/0.7238 31.72/0.8261 27.00/0.7388 28.90/0.7738 29.58/0.7641 28.73/0.7327 25.97/0.7902 26.42/0.7245 29.27/0.8238 23.99/0.7493 29.50/0.7825 29.41/0.7732 27.37/0.7223 30.11/0.7407 22.97/0.6570 26.79/0.7397 29.39/0.7752 28.17/0.7014 29.63/0.8146 26.21/0.7263 26.99/0.7077 28.03/0.8136 27.01/0.7501 28.00/0.7173 30.10/0.8084 25.12/0.7364 27.62/0.7496 31.10/0.7676 30.28/0.7562 27.66/0.8231 29.01/0.7975 30.48/0.8057 27.82/0.8385 30.81/0.7545 30.24/0.7380 29.31/0.7645 30.69/0.7276 26.66/0.8113 28.32/0.7801 30.39/0.7501 30.46/0.7826 30.62/0.8000 28.07/0.7713 29.76/0.7902 30.95/0.8091 29.45/0.8093 29.09/0.7451 30.98/0.7461 27.99/0.7707 29.54/0.7798 17.06/0.4493 26.46/0.6886 26.94/0.7692 24.96/0.7041 20.70/0.6776 21.31/0.5206 20.89/0.6756 16.54/0.5188 20.46/0.6864 22.14/0.7248 23.98/0.6349 10.42/0.4087 17.29/0.4015 22.67/0.6045 25.05/0.7175 25.18/0.6550 24.09/0.7115 21.99/0.5551 20.16/0.6362 5.76/0.3971 21.54/0.6619 22.72/0.5972 24.98/0.7995 18.52/0.5813 20.89/0.6156 23.60/0.7118 20.04/0.6040 21.16/0.7692 23.86/0.7267 19.02/0.5867 17.49/0.7154 23.04/0.8244 17.92/0.6895 27.46/0.8618 24.39/0.8275 19.65/0.6347 21.04/0.7575 18.72/0.4512 22.24/0.5982 15.36/0.6307 25.01/0.7566 17.56/0.6245 20.93/0.5421 25.52/0.7214 10.09/0.6607 22.32/0.7695 22.60/0.6994 20.52/0.8077 17.67/0.7054 20.72/0.6949 28.88/0.8364 31.87/0.7865 32.78/0.8226 30.38/0.7368 28.81/0.8615 29.58/0.8146 32.01/0.8524 28.87/0.8661 32.17/0.8171 32.18/0.8286 30.03/0.7961 32.64/0.8248 26.88/0.8315 29.50/0.8210 31.75/0.7962 31.14/0.7939 31.33/0.8108 29.02/0.8012 30.71/0.8008 31.77/0.8283 29.79/0.7852 30.14/0.7875 33.06/0.8351 29.21/0.8189 30.60/0.8144   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

Table 13: Kodak24 Denoising performance performance under Poisson noise Poisson( α x ) /α .

| Method ( α ) 01 JPEG2K (15) 20.82/0.4172 BM3D (15) 24.18/0.5811 DIP (15) 24.26/0.6524 DD (15) 23.52/0.6317 ZS-N2N (15) 25.37/0.7129 ZS-N2S (15) 18.17/0.4762 S2S (15) 25.03/0.6917 ZS-NCD (15) JPEG2K (25) BM3D (25) DIP (25) DD (25) ZS-N2N (25) ZS-N2S (25) S2S (25) ZS-NCD (25)   | 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 Average 25.15/0.6187 24.68/0.6894 23.43/0.5872 20.65/0.5202 21.66/0.4510 23.38/0.5545 19.29/0.5129 23.08/0.4707 23.18/0.4742 22.71/0.4834 23.41/0.6520 19.63/0.3773 22.63/0.5025 22.96/0.5573 24.58/0.5536 23.96/0.6000 22.39/0.5079 22.53/0.4785 21.03/0.5747 22.38/0.5103 22.79/0.4402 23.87/0.5752 21.32/0.4894 22.56/0.5249 29.14/0.7007 30.21/0.8007 28.57/0.7097 24.18/0.6679 25.36/0.5866 28.76/0.8351 24.16/0.7193 29.14/0.7945 28.80/0.7620 26.59/0.6664 29.27/0.7134 22.06/0.4881 25.54/0.6146 28.84/0.7103 27.81/0.6562 28.18/0.7652 24.88/0.6056 27.66/0.7234 27.53/0.6116 25.90/0.7294 27.02/0.6352 30.39/0.8314 24.84/0.6514 27.04/0.6900 28.16/0.6587 28.70/0.7194 27.69/0.6423 24.37/0.7237 25.08/0.6237 27.75/0.7301 23.07/0.6912 27.66/0.6731 27.22/0.6660 25.76/0.6102 28.09/0.6682 22.64/0.6166 25.45/0.6635 28.03/0.6902 27.81/0.6691 27.56/0.7317 25.14/0.6761 25.90/0.6539 27.38/0.7276 25.82/0.7273 26.11/0.6091 28.85/0.7469 24.23/0.6551 26.37/0.6761 27.46/0.6668 28.10/0.7644 26.64/0.6408 24.10/0.7155 24.21/0.5925 26.69/0.7177 22.50/0.6769 26.61/0.7161 26.03/0.6516 25.62/0.6349 25.96/0.5870 21.86/0.5989 25.00/0.6526 27.46/0.7212 26.63/0.6179 27.29/0.7523 25.14/0.6708 24.86/0.6072 26.23/0.7622 24.78/0.6287 26.01/0.6168 27.85/0.7655 23.62/0.6385 25.59/0.6679 29.02/0.7162 28.48/0.6631 27.79/0.6496 25.11/0.7471 25.80/0.6537 27.80/0.6982 24.17/0.7357 27.61/0.6054 27.20/0.5986 26.81/0.6704 27.29/0.5673 23.90/0.7183 25.92/0.6925 28.08/0.6750 27.91/0.6668 28.38/0.7375 26.19/0.7150 26.65/0.6696 27.83/0.7085 26.47/0.6953 26.44/0.6213 27.93/0.6353 25.10/0.6638 26.80/0.6757 23.25/0.5907 27.16/0.7496 24.53/0.6645 17.77/0.5489 22.42/0.5608 25.09/0.7159 15.75/0.3956 14.63/0.4183 22.84/0.6716 23.80/0.5989 24.71/0.6360 19.27/0.4577 22.51/0.5822 25.66/0.7098 26.49/0.6555 25.15/0.7713 22.36/0.6068 22.47/0.6694 24.92/0.7338 23.38/0.7072 23.98/0.6279 17.06/0.6786 20.40/0.5816 22.24/0.6170 28.20/0.7579 24.92/0.8189 26.33/0.7688 22.75/0.7643 17.10/0.6688 27.81/0.8779 17.18/0.6822 29.57/0.8712 24.12/0.7921 26.54/0.7360 18.80/0.7180 18.86/0.5165 23.71/0.6809 16.07/0.7693 24.51/0.7464 28.09/0.8149 23.76/0.7199 21.90/0.7145 10.74/0.7206 22.82/0.7762 22.77/0.6811 20.68/0.8252 18.13/0.6895 22.52/0.7418 25.57/0.6860 29.16/0.7145 29.78/0.7999 29.01/0.7387 26.00/0.7874 26.21/0.6907 29.49/0.8493 25.77/0.7955 29.52/0.8022 29.13/0.7845 27.18/0.7206 29.44/0.7200 24.25/0.6846 26.65/0.7160 29.33/0.7604 26.32/0.7011 28.24/0.7903 25.60/0.6883 28.05/0.7459 27.71/0.6871 27.18/0.7588 27.51/0.6865 29.98/0.8425 25.69/0.7370 27.64/0.7432 21.37/0.4580 25.46/0.5785 25.60/0.6174 24.96/0.5626 22.01/0.6152 22.53/0.5087 24.46/0.6331 20.53/0.5871 24.45/0.5683 24.50/0.5692 23.64/0.5353 23.96/0.4892 20.38/0.4859 23.24/0.5303 24.52/0.6224 24.67/0.5549 25.02/0.6481 23.29/0.5701 23.46/0.5558 22.91/0.6487 23.16/0.5770 23.80/0.5007 25.79/0.6751 22.15/0.5401 23.58/0.5680 23.34/0.6677 26.19/0.6206 26.16/0.6566 25.34/0.5730 24.78/0.7690 22.26/0.5441 25.47/0.6135 22.19/0.7149 23.16/0.4315 24.18/0.5200 25.92/0.6327 20.97/0.3004 22.27/0.6244 24.40/0.6526 23.17/0.5810 25.33/0.5530 26.47/0.7299 26.02/0.7379 24.06/0.5471 20.88/0.3715 23.44/0.5145 24.16/0.5468 24.92/0.6625 23.97/0.6702 24.13/0.5931 25.53/0.7181 28.94/0.6890 29.84/0.7694 28.81/0.6975 25.70/0.7741 25.92/0.6635 29.02/0.7859 24.40/0.7418 28.92/0.7396 28.41/0.7008 26.77/0.6669 28.88/0.6895 23.78/0.7052 26.65/0.7228 29.15/0.7423 28.60/0.7042 28.70/0.7762 26.27/0.7327 27.24/0.6992 28.76/0.7739 26.81/0.7320 27.00/0.6533 30.31/0.8012 25.41/0.7034 27.49/0.7243 24.50/0.6806 28.46/0.6993 28.85/0.7584 27.68/0.6746 24.91/0.7541 25.37/0.6639 27.98/0.7857 23.26/0.7120 27.68/0.7230 27.41/0.6725 26.63/0.6822 27.09/0.6650 22.51/0.6348 26.11/0.7108 28.33/0.7418 27.30/0.6488 28.62/0.7882 25.93/0.7216 25.81/0.6453 26.91/0.7499 25.76/0.6668 26.93/0.6640 28.85/0.7997 24.57/0.7007 26.56/0.7060 26.74/0.7730 30.22/0.7610 29.87/0.7276 29.15/0.7157 26.59/0.8015 27.29/0.7272 29.32/0.7582 25.78/0.7910 29.07/0.6747 28.63/0.6673 28.19/0.7284 28.70/0.6385 25.27/0.7764 27.30/0.7542 29.35/0.7294 29.35/0.7345 29.69/0.7854 27.48/0.7688 28.20/0.7411 29.34/0.7639 28.05/0.7667 27.73/0.6866 29.42/0.7026 26.39/0.7239 28.21/0.7374 19.86/0.5123 27.15/0.6869 27.66/0.7773 25.01/0.6794 16.99/0.5539 20.20/0.5373 22.75/0.6374 18.34/0.5867 23.41/0.6613 23.10/0.6715 23.82/0.6477 08.29/0.4261 18.89/0.4529 22.97/0.6329 25.75/0.7498 25.59/0.6726 24.26/0.7312 22.07/0.5632 20.01/0.6144 23.02/0.7161 19.75/0.6136 23.78/0.6429 17.86/0.7240 14.82/0.5742 21.47/0.6277 26.06/0.7395 29.57/0.7862 25.56/0.8400 27.51/0.7961 23.50/0.8066 17.38/0.7114 29.35/0.9017 17.50/0.7124 27.17/0.8418 24.95/0.8210 27.41/0.7671 19.83/0.7251 19.36/0.5811 24.83/0.7263 16.35/0.7570 25.28/0.7770 29.17/0.8445 24.14/0.7567 22.71/0.7435 10.55/0.7026 23.30/0.8048 23.17/0.7129 21.39/0.8396 18.05/0.7221 23.09/0.7674 27.28/0.7950 28.73/0.6916 31.52/0.8338 29.67/0.7383 28.10/0.8620 27.49/0.7226 30.51/0.8113 26.62/0.8213 29.38/0.6909 30.35/0.7690 29.00/0.7796 27.91/0.6114 25.54/0.7812 28.23/0.7943 30.12/0.7591 30.14/0.7763 30.21/0.8430 28.22/0.8104 29.15/0.7564 27.91/0.6373 28.02/0.7346 28.70/0.7297 31.31/0.8256 27.82/0.8012 28.77/0.7677   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## C.4 Microscopy Mouse Nuclei Dataset

For these images with noise level σ z = (10 , 20) we set λ = (200 , 600) , we train the networks for 20K steps to obtain the results. We report the detailed denoising performance in Table 14 and 15 respectively.

Table 14: Denoising performance under AWGN N (0 , σ 2 I ) on fluorescence microscopy dataset: Mouse Nuclei. Images are cropped into 128 × 128 . Noise level σ z = 10 .

| #       | JPEG2K       | BM3D         | DIP          | DD           | ZS-N2N       | ZS-N2S       | S2S          | ZS-NCD       |
|---------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| 1       | 32.90/0.7954 | 38.88/0.9631 | 37.31/0.8973 | 37.73/0.9464 | 36.37/0.9356 | 34.70/0.9410 | 10.88/0.1687 | 39.03/0.9556 |
| 2       | 32.32/0.8300 | 37.53/0.9613 | 35.93/0.8909 | 36.46/0.9560 | 35.26/0.9345 | 28.78/0.8504 | 13.08/0.4000 | 36.83/0.9546 |
| 3       | 32.97/0.8584 | 38.43/0.9690 | 36.17/0.8482 | 37.03/0.9631 | 35.86/0.9405 | 31.53/0.9307 | 12.76/0.3374 | 37.81/0.9634 |
| 4       | 32.57/0.8418 | 38.05/0.9605 | 35.82/0.9107 | 36.70/0.9478 | 34.86/0.9066 | 32.13/0.8688 | 14.42/0.3639 | 37.51/0.9303 |
| 5       | 34.54/0.7646 | 41.53/0.9596 | 38.09/0.8268 | 40.02/0.9438 | 39.30/0.9252 | 29.75/0.7976 | 10.22/0.1165 | 40.93/0.9420 |
| 6       | 32.02/0.8860 | 37.49/0.9703 | 35.24/0.8997 | 36.05/0.9628 | 35.38/0.9491 | 30.63/0.8989 | 14.42/0.3931 | 37.26/0.9588 |
| Average | 32.89/0.8294 | 38.65/0.9640 | 36.43/0.8789 | 37.33/0.9533 | 36.17/0.9319 | 31.26/0.8812 | 12.63/0.2966 | 38.23/0.9508 |

Table 15: Denoising performance under AWGN N (0 , σ 2 I ) on fluorescence microscopy dataset: Mouse Nuclei. Images are cropped into 128 × 128 . Noise level σ z = 20 .

| #       | JPEG2K       | BM3D         | DIP          | DD           | ZS-N2N       | ZS-N2S       | S2S          | ZS-NCD       |
|---------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| 1       | 28.37/0.6337 | 35.10/0.9211 | 33.09/0.8485 | 33.70/0.8938 | 32.32/0.8609 | 32.59/0.8763 | 9.30/0.0240  | 34.98/0.8843 |
| 2       | 27.97/0.7255 | 33.80/0.9410 | 31.41/0.7986 | 32.39/0.9239 | 31.07/0.8421 | 28.42/0.8328 | 10.73/0.2383 | 33.75/0.9172 |
| 3       | 28.42/0.7121 | 34.45/0.9352 | 31.47/0.7642 | 32.64/0.9096 | 31.63/0.8660 | 31.08/0.9003 | 10.12/0.1807 | 34.25/0.9216 |
| 4       | 29.31/0.7557 | 34.30/0.9245 | 31.02/0.7598 | 32.71/0.9008 | 31.12/0.8168 | 30.60/0.8551 | 11.33/0.1947 | 33.87/0.8947 |
| 5       | 29.62/0.5932 | 38.50/0.9158 | 35.45/0.7763 | 37.18/0.9068 | 35.90/0.8585 | 32.89/0.8640 | 8.23/0.0650  | 37.70/0.9137 |
| 6       | 27.71/0.7713 | 33.61/0.9399 | 31.48/0.7860 | 32.41/0.9206 | 31.43/0.8750 | 26.87/0.8312 | 10.83/0.2328 | 33.72/0.9245 |
| Average | 28.57/0.6986 | 34.96/0.9296 | 32.32/0.7889 | 33.50/0.9092 | 32.25/0.8532 | 30.41/0.8600 | 10.09/0.1559 | 34.71/0.9093 |

## C.5 Real Camera Noise Dataset PolyU

For these images with unknown noise model/level λ = 25 . Also for BM3D the best peroformance was achieved with setting σ BM3D = 15 . We report the detailed denoising performance in Table 16.

Table 16: Real camera denoising performance on camera image dataset: PolyU. The dataset includes photos taken from 3 brands of cameras. Randomly selected 6 images are cropped into 512 × 512 .

| Models   | C.plug11       | C.bike10       | N.flower1      | N.plant10      | S.plant13      | S.door10       | Average        |
|----------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| JPEG2K   | 36.26 / 0.9615 | 34.23 / 0.9371 | 33.55 / 0.9194 | 36.74 / 0.9157 | 30.39 / 0.9001 | 34.84 / 0.9012 | 34.33 / 0.9225 |
| BM3D     | 37.15 / 0.9758 | 34.85 / 0.9615 | 35.81 / 0.9504 | 38.40 / 0.9410 | 31.65 / 0.9465 | 36.43 / 0.9285 | 35.71 / 0.9506 |
| DIP      | 37.62 / 0.9724 | 34.85 / 0.9534 | 34.93 / 0.9396 | 37.64 / 0.9256 | 31.50 / 0.9396 | 36.02 / 0.9145 | 35.43 / 0.9408 |
| DD       | 36.79 / 0.9722 | 34.73 / 0.9566 | 34.85 / 0.9366 | 37.84 / 0.9327 | 30.91 / 0.9305 | 33.88 / 0.9084 | 34.83 / 0.9395 |
| ZS-N2N   | 36.30 / 0.9621 | 33.18 / 0.8853 | 33.28 / 0.8974 | 36.21 / 0.8862 | 30.57 / 0.9052 | 34.89 / 0.8804 | 34.07 / 0.9028 |
| ZS-N2S   | 22.76 / 0.9119 | 20.36 / 0.8133 | 25.20 / 0.8670 | 33.63 / 0.8920 | 21.33 / 0.8256 | 18.39 / 0.6966 | 23.61 / 0.8344 |
| S2S      | 37.75 / 0.9765 | 33.56 / 0.9545 | 35.78 / 0.9537 | 38.30 / 0.9398 | 31.93 / 0.9483 | 36.65 / 0.9433 | 35.66 / 0.9527 |
| ZS-NCD   | 36.99 / 0.9763 | 34.79 / 0.9586 | 35.43 / 0.9489 | 38.65 / 0.9449 | 31.79 / 0.9464 | 37.42 / 0.9451 | 35.84 / 0.9534 |

## D Visual Comparisons

In this section, we provide more visualization comparison of the zero-shot denoisers. The reconstruction PSNR and SSIM are above the images.

## D.1 Kodak24

Figure 7: Kodim24 under additve white Gaussian noise ( σ z = 25 ).

<!-- image -->

Figure 8: Kodim24 under Poisson noise ( α = 25 ).

<!-- image -->

## D.2 Mouse Nuclei

Figure 9: Mouse nuclei reconstruction comparison under additive white Gaussian noise ( σ z = 20 ).

<!-- image -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

## Justification:

- We claim that our proposed method is zero-shot, the details of how to crop patches from the single given noisy image can be found in Figure 1 and Section 4.
- We provide the theory connects compression and denoising, which can be found in Section 3, we characterize the performance for both AWGN and Poisson noise.
- We claim that the proposed ZS-NCD is the state-of-the-art zero-shot denoiser over both natural and non-natural images, this can be found empirically in Table 2 and 3.

## Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We have a section 6 which includes the discussion on the limitation of our work.

## Guidelines:

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

## Justification:

- We assume that the compression code for Q is defined by an encoder-decoder pair ( f, g ) characterized by R and δ , these appear in the theoretical results we provide.
- The proof of all the proposed theorems and corollary are provided in Appendix A.

## Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

## Justification:

- The neural networks structures we used are provided in Appendix B. The learning parameters including learning rate and optimizer are also included. The hyperparameter λ for each noise level are introduced in Appendix C.
- The code of our proposed method is included in the supplementary.

## Guidelines:

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

Justification:

- We provide the implementation code of our method in ZS-NCD. We provide the instruction and comments in how to run the code. We also provide the requirements of the environments to run the code.

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

Justification: We provide the networks structures, hyperparameter λ , learning rate and optimizer in training the neural networks.

## Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No] .

Justification: Our experiments follow standard practice in image denoising, where evaluation is performed on fixed noisy images and results are reported using metrics such as PSNR.

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

Justification: We provide the information of computing resources and the running time in Appendix C.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our research on zero-shot neural compression based denoising conducted in the paper conform with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The paper proposes a novel way of denoising. Zero-shot denoiser has great potential to be used for imaging systems without ground truth and large amount of noisy images, such as microcopy and medical imaging. Providing the good denoised version of the acquired noisy images is beneficial to the downstream tasks.

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

Justification: Our paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We have cited properly about the datasets we used in the paper.

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

Justification: Our paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in our research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.