## From Softmax to Score: Transformers Can Effectively Implement In-Context Denoising Steps

Paul Rosu ∗ Duke University paul.rosu@duke.edu

Lawrence Carin Duke University lcarin@duke.edu

## Abstract

Transformers have emerged as powerful meta-learners, with growing evidence that they implement learning algorithms within their forward pass. We study this phenomenon in the context of denoising, presenting a unified framework that shows Transformers can implement (a) manifold denoising via Laplacian flows, (b) score-based denoising from diffusion models, and (c) a generalized form of anisotropic diffusion denoising. Our theory establishes exact equivalence between Transformer attention updates and these algorithms. Empirically, we validate these findings on image denoising tasks, showing that even simple Transformers can perform robust denoising both with and without context. These results illustrate the Transformer's flexibility as a denoising meta-learner. Code available at https://github.com/paulrosu11/Transformers\_are\_ Diffusion\_Denoisers

## 1 Introduction

Two dominant paradigms in generative modeling-diffusion models and Transformer-based sequence models-have independently achieved remarkable success. Denoising diffusion probabilistic models (DDPMs) generate data by reversing a stochastic noise process, reaching state-of-the-art results in image synthesis [14, 2]. Transformers, trained autoregressively, power large language models (LLMs) that excel at text generation [15, 16]. This convergence prompts a natural question: can attention mechanisms perform diffusion-like denoising , and what does this imply about the nature of generative modeling?

Recent work shows these paradigms are deeply connected. Diffusion Transformers are competitive with U-Nets as backbones of diffusion models [14, 2], while diffusion-based language models like Diffusion-LM [12] offer controllable text generation, suggesting attention can drive iterative denoising in both vision and language. Beyond empirical advances, Transformers have demonstrated remarkable capabilities in meta-learning, particularly through in-context learning, where they emulate learning algorithms during inference without explicit parameter updates [3, 4]. Recent studies show that Transformers can approximate gradient-based updates in their forward pass [7, 1], effectively bridging meta-learning and classical optimization [6]. From a theoretical standpoint, Transformer dynamics have recently been linked to continuous-time probability flows and Wasserstein gradient flows [8], offering a principled lens for understanding attention-driven generation. Yet our theoretical understanding of Transformers as denoisers remains limited. How could attention layers collectively implement a stepwise denoising process-a generative flow through high-dimensional space? While invertible attention has been proposed in flow-based models [19], a general framework for attention as a diffusion operator is still lacking.

We seek to develop theoretical insight into how Transformers can perform iterative denoising, by analyzing the capability of Transformers to implement various denoising algorithms, and drawing

∗ Correspondence: paul.rosu@duke.edu, xiang.cheng@duke.edu

Xiang Cheng ∗ Duke University xiang.cheng@duke.edu

connections to their meta-learning behavior in denoising contexts. Understanding attention as a generative flow could unify the strengths of both paradigms, and illuminate why Transformers are such powerful generative learners.

Denoising is central to generative modeling, yet theory on in-context learning (ICL) has largely focused on supervised settings with clean labels. Real contexts are noisy or partially observed, motivating unsupervised ICL. We ask whether Transformers can perform stepwise denoising during inference, and show that attention naturally implements structured denoising updates and, when trained end-to-end, can learn geometry-aware anisotropic diffusion. This yields a unified view of score-based diffusion and manifold learning as Transformer-implemented denoising and connects to recent in-context diffusion generation (e.g., [20]).

## 1.1 Outline and Main Contributions

The main contributions of our work are as follows:

1. Transformers Implementing Manifold Denoising. In Section 3, we consider the problem of in-context manifold denoising [10]: One is presented a set of points sampled from an arbitrary, context-dependent manifold perturbed by Gaussian noise, and the goal is to denoise these points. We show in Lemma 3 that the Transformer can implement a Laplacian-based manifolddenoising algorithm that iteratively applies a kernel-weighted update [10]. This result provides a clear theoretical link between manifold denoising and the attention mechanism of Transformers, highlighting their potential in learning implicit manifold structures in-context.
2. Transformers as Exact Score-Based Diffusion Denoisers. Section 4 examines the role of Transformers in the context of score-based diffusion generative models [11, 17]. We show rigorously that the Transformer architecture can implement the exact score-based denoising algorithm through a suitable cross-attention construction (Lemma 4). In Section 4.2, we further explore the Transformer's performance in an in-context denoising setup, demonstrating empirically that it effectively generalizes from context-free to context-dependent denoising scenarios , thus unifying these two important perspectives through the same attention-based mechanism.
3. Efficient Approximate Score-Based Denoising via Learnable Witness Tokens. Recognizing the impracticality of exact score-based denoising, Section 4.3 introduces an approach utilizing a small set of learnable tokens ('witnesses') that approximate the exact score computation. This significantly improves both accuracy and computational efficiency, as demonstrated in our empirical results in Section 4.3. The use of learnable witnesses bears a connection to several established methods in kernel approximation.
4. Generalization to Anisotropic Diffusion and Learned Attention Parameters. In Section 4.4, we generalize beyond isotropic kernels and standard diffusion models by studying anisotropic diffusion processes . We theoretically prove (Lemma 5) that Transformers, when their Query, Key, and Value parameters are suitably aligned with the diffusion coefficient matrix, can exactly implement the reverse ODE of anisotropic diffusion. This theoretical insight is substantiated empirically in Section 4.5, demonstrating that the Transformer can learn more efficient, geometryadaptive denoising algorithms beyond the standard score-based denoising framework . We visualize this process in Figure 1.

Collectively, these contributions elucidate how attention and cross-attention can naturally realize a broad spectrum of denoising algorithms. Our results reveal the remarkable flexibility and effectiveness of Transformers for both context-dependent and context-independent denoising.

## 2 Transformer and Kernel Weighted Update

Given n tokens z (1) , . . . , z ( n ) ∈ R d , we define the matrix Z 0 := [ z (1) , . . . , z ( n ) ] ∈ R d × n . For a L -layer Transformer, we let W V ℓ , W Q ℓ , W K ℓ ∈ R d × d denote the value, query and key parameter matrices of layer ℓ , and let W S ℓ ∈ R d × d parameterize a linear module. For convenience, let W ℓ := { W V ℓ , W Q ℓ , W K ℓ , W S ℓ } , and let W := { W ℓ } ℓ =0 ,...,L . Attn std denotes the standard attention:

<!-- formula-not-decoded -->

Figure 1: Visualization of diffusion steps of a denoising algorithm implemented by a trained Transformer with diagonal W V , W Q , W K matrices. The Transformer is modelled by the score-based anisotropic diffusion reversal algorithm in Lemma 5, Section 4.5. The first 2 columns show the clean and noisy images. Each subsequent column shows the output at a given layer of a 6-layer Transformer. We visualize the difference between consecutive layers' output at the bottom. Notice that each Transformer layer denoises a roughly non-overlapping patch.

<!-- image -->

where smax applies column-wise softmax. We use a diagonal mask so that each token does not attend to itself. This is standard when disallowing self-attention (diagonal mask) and is also common in manifold denoising, where the update at each position is informed by its neighbors rather than itself (see [10], Section 3.1). For notational simplicity, we omit the explicit mask matrix. Let rbf ( · , · ) : R d × n × R d × n → R n × n be the matrix valued RBF kernel, defined as [ rbf ( U, V )] ij = exp ( -1 / 2 · ∥ U ( i ) -V ( j ) ∥ 2 2 ) . Let [ logrbf ( U, V )] ij := [log( rbf ( U, V ))] ij . We define Attn rbf as

<!-- formula-not-decoded -->

We highlight that the difference between the standard Attn std and Attn rbf as follows:

<!-- formula-not-decoded -->

Given Attn ∈ { Attn std , Attn rbf } , we define the Transformer via the iterative update:

<!-- formula-not-decoded -->

In many subsequent applications of interest, ∥ W Q ℓ z ( j ) ∥ 2 2 and ∥ W K ℓ z ( j ) ∥ 2 2 are constant, or close to constant, due to concentration of norms of high-dimensional Gaussian vectors. In this case, we verify that Attn std and Attn rbf behave identically:

Lemma 1 (Equivalence of Attn std and Attn rbf on the sphere (with W Q , W K identity-scaled)) . If ∥ z ( i ) ℓ ∥ 2 = C for some constant C , and if W Q ℓ and W K ℓ are scalings of the identity matrix, then Attnstd ( Z ℓ , W ) = Attnrbf ( Z ℓ , W ) .

We defer proof of Lemma 1 to Appendix A. We also empirically verify the similar performance of Attn std and Attn rbf in Figure 3.

## 2.1 Transformer Construction for Kernel-Weighted Update.

Given tokens { z ( i ) } i =1 ,...,n ⊂ R d , let F = { z (1) , . . . , z ( m ) } denote the subset of frozen tokens. Much of our discussion in this paper revolves around the following kernel-weighted-update scheme:

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

and z ( j ) ℓ +1 = z ( j ) ℓ for j ∈ F . Motivated by (4), we implement this update using a single attention module together with a fixed mask that freezes the columns in F .

Let Z ℓ := [ z (1) ℓ , . . . , z ( n ) ℓ ] ∈ R d × n , and let M F ∈ R n × n denote the diagonal freeze mask with ( M F ) jj := { j / ∈ F} . We define the Transformer with frozen-token mask by

<!-- formula-not-decoded -->

We verify below that (6) can exactly implement the kernel-weighted update (4) (proof deferred to Appendix A):

Lemma 2 (Transformer implements kernel-weighted update.) . By choosing W Q ℓ = 1 /σ ℓ I d × d , W K ℓ = 1 /σ ℓ I d × d , W V ℓ = γ ℓ I d × d , W S ℓ = (1 -α ℓ -γ ℓ ) I d × d , the masked Transformer in (6) exactly implements the kernel-weighted update in (4) .

Cross-attention form. When a subset of tokens is permanently frozen (e.g., a large training set), it is convenient to store them separately and update only the remaining tokens. Let Z F := [ z (1) , z (2) , . . . , z ( m ) ] ∈ R d × m denote the matrix whose columns are the frozen tokens. Let Z := [ z ( m +1) , z ( m +2) , . . . , z ( n ) ] denote the matrix of not-frozen tokens. Let cross attention modules CrAttn std ( U, Z ; W ℓ ) := W V ℓ U smax ( U ⊤ W K ℓ ⊤ W Q ℓ Z ) and CrAttn rbf ( U, Z ; W ℓ ) := W V ℓ U smax ( logrbf ( W K ℓ U, W Q ℓ Z )) be defined analogously to Attn std and Attn rbf . We consider the Transformer with cross attention:

<!-- formula-not-decoded -->

We will use TF ℓ ( Z, Z F ; W ) to denote the value of Z ℓ , when evolved according to (7), initialized at Z 0 = Z and Z F ; i.e. the output of the ℓ -layer Transformer in (7). In the single-query setting used for score-based denoising (Section 4), (7) coincides with (6) (the self-attention term vanishes), so we will use the cross-attention notation for convenience.

## 3 Manifold Denoising via Laplacian-based ODE

In this section, we are motivated by the following in-context manifold denoising problem, originally proposed in [10]: Let M be a m -dimensional manifold embedded via an embedding T : M→ R d . We identify a point a ∈ M with its d -dimensional embedding T ( a ) . Let x (1) , . . . , x ( n ) be sampled from M , and we observe z ( i ) = x ( i ) + ε ( i ) , where ε ( i ) ∼ N (0 , σ 2 I d × d ) . The goal is to recover the x ( i ) 's. This problem is particularly challenging because (1) the manifold M is unobserved, and (2) the manifold M is context-dependent . Therefore, to solve this problem in-context , the Transformer needs to be able to learn (implicitly or explicitly) the manifold structure, from the observations z ( i ) 's. [10] propose the following manifold-denoising algorithm based on the Laplace-Beltrami operator: Let the adjacency matrix W ∈ R n × n be defined as W ij := exp( -∥ z ( i ) -z ( j ) ∥ 2 2 / (2 σ 2 )) and set W ii = 0 to remove self-loops. Let D be the diagonal matrix with D ii := ∑ n j =1 W ij . The RBF Laplacian is defined as L = I n × n -D -1 W . The denoising algorithm is based on a Laplacian-based flow ∂ t z ( i ) ( t ) = -γ ∑ n j =1 L ij z ( j ) ( t ) , whose Euler discretization gives the discrete-time algorithm:

<!-- formula-not-decoded -->

In Lemma 3, we verify that (8) is an instance of the kernel-weighted update (4) (with F = ∅ ), and is therefore implemented by the masked-attention Transformer construction in Lemma 2.

Lemma 3 (Reformulation of Manifold Denoising) . The manifold denoising algorithm in (8) is exactly equivalent to (4) , with F = ∅ , α ℓ = 0 , σ ℓ = σ , and γ ℓ = δ ℓ . Consequently, the algorithm (8) is implemented by the Transformer construction in Lemma 2.

The proof of Lemma 3 is just by algebra; we defer it to Appendix B.

## 3.1 Experiments for Manifold Denoising

We present experimental evidence demonstrating the Transformer construction in Lemma 2 can indeed act as an effective manifold denoiser. The problem setup is very similar to the above: Let x (1) , . . . , x ( n ) be clean images sampled from a training set. We perturb each image via z ( i ) = x ( i ) + ε ( i ) , where ε ( i ) ∼ N (0 , σ 2 I ) , for some noise level σ . The input matrix Z 0 = [ z (1) , . . . , z ( n ) ] and the Transformer TF ℓ ( Z ; W ℓ ) are as defined in Section 2. Motivated by Lemma 2, we impose the constraint that W V ℓ , W Q ℓ , W K ℓ , W S ℓ are each a (trainable) scalar multiple of identity. The training loss is given by E Z 0 [ 1 n ∑ n i =1 ∥ [ TF L ( Z 0 ; W )] ( i ) -x ( i ) ∥ 2 2 ] . Throughout, we report noise levels via a signal-to-noise ratio (SNR) defined as SNR := σ noise /σ data , i.e., the standard deviation of the additive Gaussian noise divided by that of the clean data distribution.

Figure 2a shows that the test loss decreases with the context length; this agrees with the fact that the discrete RBF Laplacian L becomes a better approximation to the Laplace-Beltrami operator with increasing number of samples. Figure 2b shows the test loss of the intermediate layer outputs of a 6-layer Transformer; the test loss decreases with number of layers, which is consistent with (8), as each layer implements one more discretization step of (8).

<!-- image -->

(a) Loss Against Context Length

n

.

<!-- image -->

(b) Loss Against Layer

ℓ

. Context size n =100.

Figure 2: (a) Test-set RMSE on MNIST versus context size n for the Laplacian-based denoising Transformer under a noise level of SNR ≈ 1. Each point shows the converged loss of a separately trained 6-layer model; (b): Test RMSE of intermediate output at layer ℓ with fixed n = 100 .

## 4 Score-Based Diffusion Denoising

As in standard score-based diffusion models [11, 18, 17], our training objective is defined over the empirical distribution ˆ p (equivalently denoted p 0 below), the uniform distribution over training samples { x ( i ) } n i =1 . We assume these samples are drawn i.i.d. from an unknown population distribution p . While optimization is with respect to ˆ p , the modeling goal is to generalize to p . Unless otherwise stated, all evaluations reported in this section are computed on unseen test examples.

Score-based diffusion generative models [11, 18] motivate the following problem: Let p 0 denote the uniform distribution over a (large) training set of samples { x (1) , . . . , x ( n ) } . Let p t = p 0 ∗ N (0 , tI ) , where ∗ denotes convolution. As we review in Section 4.1, getting access to ∇ log p t ( x ) allows one to turn an (almost pure) Gaussian noise x into a sample from p 0 by reversing a diffusion process. In practice, one trains a neural network s θ ( x, t ) , parameterized by weights θ , to approximate the (scaled) score, by minimizing the denoising objective E [ ∥ s θ ( x ; t ) -t ∇ log p t ( x ) ∥ 2 2 ] .

Conveniently, ∇ log p t ( x ) is exactly equal to a weighted-average between x and each { x ( i ) } i =1 ,...,n . Consequently, we verify in Lemma 4 that the Transformer (7) can implement score-based denoising.

## 4.1 Standard Score-based Diffusion Denoising

Consider the variance-exploding SDE, defined as

<!-- formula-not-decoded -->

where B t denotes the standard Brownian motion. Let p t denote the distribution of x t . We verify that p t = p 0 ∗ N (0 , tI ) . By the Fokker Planck equation, p t evolves as

<!-- formula-not-decoded -->

Consequently, if z 0 ∼ p 0 and dz t = -1 2 ∇ log p t ( z t ) dt , then z t has the same distribution as x t in (9). The above follows directly from integration-by-parts. The diffusion process of (9) can be reversed by reversing the velocity field v t ( x ) . Concretely, for any fixed time T , let y 0 ∼ p T , and dy s = 1 2 ∇ log p T -s ( y s ) ds , then y s ∼ p T -s . The Euler discretization of the ODE for y s is given by

<!-- formula-not-decoded -->

where σ 2 k is the time T -s at step k (equivalently, the Gaussian variance of ¯ y ℓ ).

From the definition of p 0 and p t , it follows that

<!-- formula-not-decoded -->

where Z t is the normalization constant for N (0 , tI ) . Observe that the RHS of (12) is exactly the kernel-weighted average κ from (4). Combining (12) and (11) then implies that the score-based denoising algorithm (11) is equivalent to (4), which we formalize in the following lemma:

Lemma 4 (Score-based Denoising as Kernel-Weighted Update) . The score-based denoising algorithm (11) is equivalent to (4) with z ( n +1) ℓ := ¯ y ℓ , F = { 1 , . . . , n } , α ℓ = 0 , and γ ℓ = δ ℓ / (2 σ 2 ℓ ) . Consequently, the Transformer construction in Lemma 2 can implement score-based denoising (11) .

Wedefer the proof to Appendix C. We highlight an important difference from the manifold denoising setup in Section 3: here, the set of training samples is typically large, and does not vary with context . Instead, the training samples encode global context-free knowledge . On the other hand, the presence of similar tokens in-context can indeed improve the denoising accuracy; we show this empirically in Figure 5 in the next section. Thus the same attention architecture can be well-suited for both in-context, and context-free learning .

In Section 4.3, we discuss a generalization of the construction of Lemma 4; where we train the crossattention tokens in (7) to learn an approximate score that significantly improves generalization error and computational cost. In Section 4.4, we discuss a generalization of the score-based denoising sequence (11) itself, which also leads to significant accuracy improvements.

## 4.2 Experiments for Score-Based Denoising

We empirically validate Lemma 4. Our experiment verifies that the Transformer construction of Lemma 4 does indeed implement exact score-based denoising. The setup is as follows: for each input context, we sample x (1) , . . . , x ( n ) clean images from a training set. We let z ( i ) = x ( i ) . The query image is x ( n +1) . We perturb the query image by Gaussian noise: z ( n +1) = x ( n +1) + ε ( n +1) , where ε ( n +1) ∼ N (0 , σ 2 I ) , where σ is chosen noise level. The Transformer TF ℓ ( Z ; W ℓ ) is as defined in (7). Motivated by Lemma 2, we impose the constraint that W V ℓ , W Q ℓ , W K ℓ , W S ℓ are each a (trainable) scalar multiple of identity. The training loss is given by the reconstruction error E Z 0 [ ∥ [ TF L ( Z 0 ; W )] ( n +1) -x ( n +1) ∥ 2 2 ] .

Single-query score-based denoising. Figure 3 shows two architectures: { 'RBF', 'STD' } refer to Transformers that use Attn rbf (2) and Attn std (1) respectively. For each architecture, 'trained' means the Transformer is trained on the above loss; 'theory' means the Transformer parameters are fixed at the construction in Lemma 4 (for a geometrically decreasing σ k schedule). Figure 3 plots the test loss of the per-layer output of 6-layer Transformers. We observe the following:

1. The reconstruction error decreases monotonically with layers, suggesting that the Transformer is implementing an iterative denoising algorithm at training convergence. 'RBF-trained' appears to denoise more aggressively in the first few layers compared to the theory constructions.

2. RBF-theory and STD-theory have almost identical losses, supporting our claim in Lemma 1 that the RBF and STD attentions behave similarly under our constructions.

In Figure 4, we visualize the clean and noisy query images, as well as the per-layer intermediate outputs of the RBF-trained and RBF-theory Transformers from Figure 3. In Figure 4a and 4b, we see that the trained Transformer can exactly recover the clean training set image , as does the exact score-denoising algorithm. In Figure 4c and 4d, we see that test images are not being recovered; instead the denoising algorithm produces a similar image from the training set. Though undesired, this is the expected behavior for exact score-denoising. In Figure 4e and 4f, we see the 'generation' capabilities of each Transformer from pure noise; again, the 'generated' image is in fact a sample from the training set (as expected of exact score denoising). The generated outputs of RBF-trained and RBF-theory are almost identical in all cases.

<!-- image -->

CIFAR-10DenoisingProgressionperLayer(L=6)

30

25

20

Test Loss

15

10

光

1

XRBF theory

+

STD theory

RBF trained (mean ±1o, n=3)

STD trained (mean ±1o, n=3)

3

4

5

6

LayerIndex

(b) CIFAR10

Figure 3: (a) Test loss of intermediate output at layer ℓ versus ℓ on MNIST, (b) CIFAR-10, at an injected noise level corresponding to SNR ≈ 3 for (a) and ≈ 5 for (b). All Transformers have 6layers total. For CIFAR-100 results and FID on CIFAR-10/100, see App. E.2.1.

In-context score-based denoising. In Figures 5 and 6, we present a different experiment which highlights the in-context aspect of score-based denoising. In this setup, the input consists of the noiseless training samples { x (1) , . . . , x ( n ) } , as well as m contextual images { x ( n +1) , . . . , x ( n + m ) } . The query image is x ( n + m +1) . Crucially, all m contextual images belong to the same class as the query image x ( n + m +1) . The Transformer input is z ( i ) = x ( i ) for i = 1 , . . . , n + m , and the query point is perturbed with Gaussian noise: z ( n + m +1) = x ( n + m +1) + N (0 , σ 2 I ) . Wecall m the 'context length'. The query x ( m + n +1) attends to the training set x (1) , . . . , x ( n ) via the CrAttn module in (7), while the context samples + query x ( n +1) , . . . , x ( n + m ) attend to each other via the Attn module in (7) (with appropriate masking). Experiment details are in Appendix E.4.

The training loss is again the reconstruction loss. In Figure 5, we plot the test reconstruction loss against the context length. The presence of a few contextual examples leads to a significant improvement in reconstruction error. This demonstrates that the Transformer is indeed implementing a denoising algorithm that can adapt to its context (at least partially). Figure 6 visualizes this phenomenon. The injected noise has high SNR=5, so that the original image is difficult to recover. In the leftmost block, we see that providing same-class contextual images guided the model to correctly generate '6' (even though the intermediate output appears to tend towards '5'). In the middle block, the contextual set consisted of images from random classes, and the Transformer had difficulty recovering the image. In the right block, a Transformer trained without context generated a different digit from the same starting noisy image.

## 4.3 Approximate Score-Based Denoising With Learnable Witnesses

In the preceding section, we show that exact score-based denoising can be implemented via crossattention between the noisy sample, and every other sample in the training dataset. This is in generally practically infeasible. On the flip-side, the training set is typically not context-dependent . Therefore, one can hope to approximate the exact score cross-attending to a small set of representative samples . Concretely, let Ψ = { ψ (1) , . . . , ψ ( τ ) } ⊂ R d denote a set of witnesses ψ ( i ) , for some

2

<!-- image -->

(e) MNIST (random noise query)

(f) CIFAR10 (random noise query)

Figure 4: Visualization of denoised images. Query image from { train, test, random noise } . In each sub-figure, the 1st row shows the trained Transformer, and the 2nd row shows the Transformer with theoretical weights. First two columns in each figure shows the { clean, noisy } query image respectively. Columns 2-7 show the per-layer intermediate outputs of a 6-layer Transformer.

<!-- image -->

(a) MNIST: Test Loss vs Context Length

(b) CIFAR-10: Test Loss vs Context Length

<!-- image -->

Figure 5: Test-set RMSE for a 6-layer RBF-attention denoiser as a function of the number of clean, same-class context images n ctx. The query token is severely noised (SNR ≈ 15 for MNIST, SNR ≈ 10 for CIFAR-10), while the full training set is available through cross-attention. Adding just a few class-matched examples dramatically lowers the loss from the nearly unrecoverable zero-context case. For CIFAR-100 results and FID on CIFAR-10/100, see App. E.3

τ &lt;&lt; n . The goal is to choose a Ψ which well-approximates the kernel-weighted update from (4).

<!-- formula-not-decoded -->

Observe that the 2 is equivalent to the κ -averaged term in (4). In the setting of score-based denoising, z (1) , . . . , z ( n ) is drawn from p 0 independently. An intuitively simple choice of Ψ is then to sub-sample τ points from { z (1) , . . . , z ( n ) } ; under this choice, the LHS of (13) is simply a MonteCarlo estimate of the RHS of (13). Assume that p 0 is compactly supported, we can show via basic algebra that E [ ∥ 1 -2 ∥ 2 2 ] = O ( τ -1 ) . We provide a short proof in Lemma 7 in Appendix C. Much more sophisticated approaches exist, and the problem of picking a good witness set Ψ has deep connections to a number of areas such as Nystrom's method for kernel approximation [21, 9], kernel mean embeddings [13] and kernel herding [5].

<!-- image -->

<!-- image -->

<!-- image -->

Figure 6: Two leftmost images are the clean and noisy (SNR ≈ 15 ) query images respectively. We show three sets of six intermediate layer outputs. Left: a Transformer trained using with a 100image contextual set is presented contextual images from the correct class at test time, and generates the correct image. Middle: the same Transformer, when provided contextual images from random classes, fails to generate the right image. Right: a different Transformer, trained without context, generates the digit 2 from the same input. All three scenarios start from the identical noisy input.

Motivated by the above, we treat Ψ as a learnable parameter , alongside the other Transformer parameters. Lemma 4 showed that the score-based denoising algorithm (11) can be implemented by the Transformer defined in (6) (equivalently (7)). The witness-based Transformer is then defined as

<!-- formula-not-decoded -->

where U ℓ ∈ R d × τ is the only difference from (7). The full set of parameters of (14) is { U ℓ , W V ℓ , W Q ℓ , W K ℓ } ℓ =1 ,...,L . In words, column i of U ℓ represents ψ ( i ) in the above discussion, and U ℓ can differ from layer to layer (analogous to Ψ changing with each denoising step.) We initialize U ℓ with randomly sampled vectors from { z ( i ) } i =1 ,...,n . We show in Figure 7 that (14) has significant advantages in computation and generalization error (see Section 4.5 for discussion).

## 4.4 Anisotropic Diffusion and its Reverse ODE

Consider a generalization of the standard diffusion process in (9): instead of dx t = dB t , let A t : R + → R d × d denote a time-parameterized family of matrices. For simplicity, we assume that A t is symmetric. The anisotropic diffusion is described by

<!-- formula-not-decoded -->

Throughout Section 4.4 we restrict to diffusion schedules satisfying the normalization ∫ T 0 A 2 s ds = TI . Let p t denote the distribution of x t . Let T be some final time, and let p T = p 0 ∗ N (0 , T I ) . In Appendix C.1, we show that (15) admits a simple continuous-time reverse ODE y t ∼ p T -t . Its Euler discretization gives a denoising algorithm that can be implemented by a Transformer:

Lemma 5 (Anisotropic Denoising with Transformer) . Let x (1) , . . . , x ( n ) denote a training set. Let z ( i ) = x ( i ) for i = 1 , . . . , n . Let z ( n +1) = x ( n +1) + N (0 , T I ) denote the query token. Consider the anisotropic diffusion in (15) , where A t is a family of arbitrary symmetric matrices satisfying ∫ T 0 A 2 s ds = TI . Let t 0 := T , t ℓ +1 := t ℓ -δ ℓ , M 0 := TI , M ℓ +1 = M ℓ -δ ℓ A 2 t ℓ , where we identify A ℓ with A t ℓ . Then there exists a discrete-time denoising algorithm, based on the time-reversal ODE of the anisotropic diffusion in (15) . This denoising algorithm can be implemented by the Transformer in (7) , with F = { 1 , . . . , n } and parameters

<!-- formula-not-decoded -->

The proof of Lemma 5 uses the same tools from Section 4.1, and we defer it to Appendix C. The importance of Lemma 5 lies in generalizing beyond the isotropic RBF kernel used in score-based denoising, and justifies the choice of non-identity W S , W Q , W K , W V matrices , which enables the Transformer to implement an anisotropic denoising algorithm that adapts to the distribution at step ℓ . In Section 4.5, we evaluate a witness-variant of the Transformer from Lemma 5.

## 4.5 Experiments for Witness and Anisotropic Denoising

In the following, we present a set of experiments that serves two purposes: First, it demonstrates the advantage of the Transformer with learnable witnesses from Section 4.3. Second, it studies the anisotropic diffusion Transformer construction from Lemma 5, with general learnable

W Q , W K , W V matrices. The setup is identical to the single-query score-based denoising experiment in in Section 3.1. The one difference is in the architecture: let the Transformer TF ℓ ( z 0 ; U ℓ , W ℓ ) be as defined in (14), where U ℓ denotes the set of learnable witnesses and W ℓ denotes the set of parameters at layer ℓ . The training loss is given by E Z 0 [ ∥ TF L ( z ; U, W ) -x ∥ 2 2 ] .

We initialize the witnesses (U) by sampling a random subset of training examples to set the initial columns of (U). During training, we treat (U) as a continuous, fully learnable parameter matrix and optimize it jointly with (W) via gradient descent. Consequently, at convergence the columns of (U) need not coincide with any training example (e.g., a column of (U) may represent an average or a denoising-friendly summary of multiple samples). This makes witness selection differentiable end-to-end and connects to classical kernel approximation methods (e.g., Nystrom and related constructions) that summarize large kernels with a small set of inducing points [21, 9, 5, 13].

In Figure 7, we compare the test loss of three models: the Baseline model is the Transformer from Lemma 4 that exactly implements the score-based denoising algorithm in (11). The 'Witness+RBF' model implements the witness-Transformer from (14), and its weights W are (trainable) scalar multiples of identity . The 'Witness+anisotropic' model is motivated by Lemma 5; we do not assign the Transformer parameters as done in (16); instead, we simply make { W S ℓ , W V ℓ , W Q ℓ , W K ℓ } trainable diagonal matrices. We provide details of the implementation in Appendix E.4.

With a moderate number of witnesses, both witness-based Transformers significantly outperform the baseline Transformer that implements exact score matching. Furthermore, the anisotropic denoising Transformer shows a significant advantage over the RBF (isotropic) denoising Transformer.

In Figure 1, we visualize the per-layer denoising progress of a 6-layer Witness+anisotropic Transformer (1000 witnesses). In contrast to the standard denoising sequence in Figure 4, where noise is removed uniformly in space, the denoising sequence in Figure 1 proceeds patch-wise . Such a patch-wise denoising procedure coincides with the anisotropic denoising algorithm when A ℓ for each step ℓ is a sparse diagonal matrix, whose non-zero entries coincide with the pixel locations of a localized image patch.

Across datasets and witness counts, the witness-based Transformer achieves lower test reconstruction error than the Baseline model that implements exact score-based denoising. Although the Baseline can perfectly fit the empirical train distribution ( ˆ p ), its unconstrained attention tends to generalize poorly. In contrast, the Witness model's learnable summary witnesses provide an inductive bias that improves generalization to the population distribution ( p ).

<!-- image -->

(a) MNIST

(b) CIFAR10

Figure 7: Test loss at SNR ≈ 3 for (a) and ≈ 5 for (b) with a 6-layer model as a function of the number of trainable witness tokens τ . The horizontal line shows the performance of the exact-score denoiser that uses the full training set, while the two curves plot witness-based approximations (14): one based on anisotropic diffusion (Section 4.4), and one based on standard RBF kernel. All reported metrics are computed on unseen test data to assess generalization. For CIFAR-100 results and FID on CIFAR-10/100, see App. E.3.1.

All experiments are done on a single A5000 GPU, with each experiment taking at most a few hours.

## References

- [1] E. Aky¨ urek, D. Schuurmans, and J. Andreas. Transformers learn in-context by gradient descent. In Proceedings of the 40th International Conference on Machine Learning . PMLR, 2023.
- [2] J. Bao, C. Dong, D. Zhang, L. Zhang, J. Zhang, H. Liu, Y. Yang, L. Yuan, D. Chen, and F. Wen. All are worth words: A vit backbone for diffusion models. arXiv preprint arXiv:2209.12152 , 2022.
- [3] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165 , 2020.
- [4] S. C. Y. Chan, A. Santoro, A. K. Lampinen, J. X. Wang, A. Singh, P. H. Richemond, J. McClelland, and F. Hill. Data distributional properties drive emergent in-context learning in transformers. arXiv preprint arXiv:2205.05055 , 2022.
- [5] Y. Chen, M. Welling, and A. Smola. Super-samples from kernel herding. arXiv preprint arXiv:1203.3472 , 2012.
- [6] D. Dai, Y. Sun, L. Dong, Y. Hao, S. Ma, Z. Sui, and F. Wei. Why can gpt learn in-context? language models implicitly perform gradient descent as meta-optimizers. arXiv preprint arXiv:2212.10559 , 2022.
- [7] S. Garg, D. Tsipras, P. Liang, and G. Valiant. What can transformers learn in-context? a case study of simple function classes. arXiv preprint arXiv:2208.01066 , 2022.
- [8] B. Geshkovski, C. Letrouit, Y. Polyanskiy, and P. Rigollet. A mathematical perspective on transformers. arXiv preprint arXiv:2312.10794 , 2023.
- [9] A. Gittens and M. W. Mahoney. Revisiting the nystr¨ om method for improved large-scale machine learning. The Journal of Machine Learning Research , 17(1):3977-4041, 2016.
- [10] M. Hein and M. Maier. Manifold denoising. Advances in neural information processing systems , 19, 2006.
- [11] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems , 33:6840-6851, 2020.
- [12] X. L. Li, J. Thickstun, I. Gulrajani, P. Liang, and T. B. Hashimoto. Diffusion-lm improves controllable text generation. arXiv preprint arXiv:2205.14217 , 2022.
- [13] K. Muandet, K. Fukumizu, B. Sriperumbudur, B. Sch¨ olkopf, et al. Kernel mean embedding of distributions: A review and beyond. Foundations and Trends® in Machine Learning , 10(12):1-141, 2017.
- [14] W. Peebles and S. Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748 , 2022.
- [15] A. Radford, K. Narasimhan, T. Salimans, I. Sutskever, et al. Improving language understanding by generative pre-training. 2018.
- [16] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog , 1(8):9, 2019.
- [17] J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 , 2020.
- [18] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 , 2020.
- [19] R. Sukthanker, Q. Huang, H. Wang, Z. S. Liu, and E. P. X. Wang. Generative flows with invertible attentions. arXiv preprint arXiv:2106.03959 , 2021.

- [20] Z. Tang, Z. Yang, M. Khademi, Y. Liu, C. Zhu, and M. Bansal. Codi-2: In-context interleaved and interactive any-to-any generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 27425-27434, June 2024.
- [21] C. Williams and M. Seeger. Using the Nystr¨ ommethod to speed up kernel machines. Advances in neural information processing systems , 13, 2000.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We have a subsection dedicated to discussing limitations. See Section G.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: We have several lemmas, and all details and assumptions connected to them are discussed. All proofs are provided and to the best of our knowledge are correct.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: All details of our experiments are provided. All code will be released upon publication.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes] X

Justification: All data we have considered here is publicly available, and in the Appendix we provide details for reproduction of synthetic data.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes] X

Justification: All experimental details are provided and code will be released upon publication.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We explain what type of GPU was used for our computations, and needed compute time above the Limitations Section.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have read the code of conduct, and have complied with it.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA] .

Justification: We discuss some broader impact of the work in introduction. The work is mainly foundational and has no direct social impact.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: There are no safety issues with the data that we have considered.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: We are using no code or data that would require this.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: The main new asset is the related code we have developed. It will be freely released and is well documented with its license.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

[NA]

Justification: No experiments of this type were performed.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: None of our work requires an IRB.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: This work does not involve any non-standard usages of LLMs.

## A Theory for Transformer Construction

Proof of Lemma 1. Recall from Section 2 the following expression for smax :

<!-- formula-not-decoded -->

In particular, the ∝ symbol is applied for fixed column j , across rows i .

Under our assumptions that W K ℓ = aI d × d and W Q ℓ = bI d × d , we can simplify

<!-- formula-not-decoded -->

Carefully writing the expression for the smax term in Attn rbf gives

̸

<!-- formula-not-decoded -->

̸

̸

This concludes our proof.

Proof of Lemma 2. Recall (6), where ( M F ) jj = ✶ { j / ∈ F} . If j ∈ F , then ( M F ) jj = 0 and the update gives Z ( j ) ℓ +1 = Z ( j ) ℓ , matching the frozen-token rule in (4).

Now fix j / ∈ F and let z ( j ) ℓ denote the j th column of Z ℓ . Since ( M F ) jj = 1 , (6) gives

<!-- formula-not-decoded -->

With W Q ℓ = W K ℓ = 1 σ ℓ I d × d , the RBF similarity between key i and query j is exp( -∥ z ( i ) ℓ -z ( j ) ℓ ∥ 2 2 / (2 σ 2 ℓ )) , and the column-wise softmax therefore equals κ ℓ ( i, j ) in (5). Thus, using W V ℓ = γ ℓ I d × d ,

̸

<!-- formula-not-decoded -->

Using W S ℓ = (1 -α ℓ -γ ℓ ) I d × d , we obtain

<!-- formula-not-decoded -->

which is exactly (4).

̸

̸

## B Theory for Manifold Denoising

Proof of Lemma 3. Recall L = I n × n - D -1 W , where W ij := exp( -∥ z ( i ) -z ( j ) ∥ 2 2 / (2 σ 2 )) and W ii = 0 , and D ii = ∑ n j =1 W ij . Then for each i ,

̸

<!-- formula-not-decoded -->

̸

Plugging this into (8) gives

<!-- formula-not-decoded -->

̸

Since ∑ j = i W ij / D ii = 1 , this can be rewritten as

̸

<!-- formula-not-decoded -->

Noting that W ij / D ii = κ ℓ ( j, i ) with σ ℓ = σ , we obtain (4) with F = ∅ , α ℓ = 0 , and γ ℓ = δ ℓ .

## C Theory for Score-Based Diffusion (Section 4)

Proof of Lemma 4. Let x (1) , . . . , x ( n ) denote the training set of unperturbed samples. Let x ( n +1) denote the query sample. Let z ( i ) 0 = x ( i ) for i = 1 , . . . , n . Let z ( n +1) 0 = x ( n +1) + N (0 , σ 2 0 I ) , where σ 2 0 := T is the initial noise level. From (11),

<!-- formula-not-decoded -->

Where the second equality is by (11), and the third equality is by the expression of the score in (12). Comparing (17) to (4), we see that it is equivalent to letting F = { z (1) , . . . , z ( n ) } , and σ ℓ = σ ℓ , and γ ℓ = δ ℓ 2 σ 2 ℓ . Consequently, let

<!-- formula-not-decoded -->

From (7),

<!-- formula-not-decoded -->

where we leave out the Attn rbf term because Z ℓ contains the single token query z ( n +1) ℓ , and we assumed in Section 2 that Attn does not have tokens attending to themselves.

By the lemma statement, we choose

<!-- formula-not-decoded -->

̸

Recall from Section 2.1 that CrAttn rbf ( U, Z ; W ℓ ) := W V ℓ U smax ( logrbf ( W K ℓ U, W Q ℓ Z )) . Here, U = Z F = {[ z (1) , . . . , z ( n ) ]} and Z = Z ℓ . Thus

<!-- formula-not-decoded -->

Under the definition of CrAttn , Z F , and Z ℓ = [ z ( n +1) ℓ ] , we verify that the above is identical to (17).

Lemma 6 (Continuity Equation) . Let p t = p 0 ∗ N (0 , tI ) . Its continuity equation is satisfied by v t = -1 2 ∇ log p t ( x ) , as defined in (10) .

Proof of Lemma 6. Because the forward SDE is standard Brownian motion, its density satisfies the heat equation 2 ∂ t p t = 1 2 ∆ p t . Define the velocity field v t ( x ) := -1 2 ∇ log p t ( x ) . Then, using the identity p t ∇ log p t = ∇ p t ,

<!-- formula-not-decoded -->

which is exactly (10).

Lemma 7. Consider the setting in Section 4.3. Assume that ∥ ∥ z ( i ) ∥ ∥ 2 ≤ R for all i . Then for any ∥ z ∥ ≤ R , we have

<!-- formula-not-decoded -->

Proof. For convenience,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let A τ := 1 τ ∑ τ j =1 e -∥ ψ ( j ) -z ∥ 2 2 / (2 σ 2 ) ( ψ ( j ) -z ) and B τ = 1 τ ∑ τ j =1 e -∥ ψ ( j ) -z ∥ 2 2 / (2 σ 2 ) . Analogously let A n := 1 n ∑ n j =1 e -∥ z ( j ) -z ∥ 2 2 / (2 σ 2 ) ( z ( j ) -z ) and B n := 1 n ∑ n k =1 e -∥ z ( k ) -z ∥ 2 2 / (2 σ 2 ) . Under our assumptions, there exists constants C, a, a ′ such that e -∥ z ( i ) -z ∥ 2 2 / (2 σ 2 ) ( z ( j ) -z ) ≤ C , and a ≤ e -∥ z ( i ) -z ∥ 2 2 / (2 σ 2 ) ≤ a ′

We can decompose

<!-- formula-not-decoded -->

The first term can be bound as

<!-- formula-not-decoded -->

2 For unit-variance Brownian motion the infinitesimal generator is 1 2 ∆ , hence ∂ t p t = 1 2 ∆ p t .

The second term can be bound using a Taylor expansion of f ( r ) = 1 /r :

<!-- formula-not-decoded -->

Thus we can bound the second term of (18) as

<!-- formula-not-decoded -->

## C.1 Anisotropic Diffusion

Proof of Lemma 5. Let x 0 ∼ p 0 , and dx t = A t dB t for some matrix-valued function t → A t . For simplicity, assume that A t is symmetric, so that A t A ⊤ t = A 2 t . By the Fokker Planck equation,

<!-- formula-not-decoded -->

From the RHS, we verify that the continuity equation is satisfied by dx t = -1 2 A 2 t ∇ log p t ( x ) . The forward ODE is thus given by

<!-- formula-not-decoded -->

Let p t denote the distribution of x t . Notice that the total Gaussian covariance under dx t = A t dB t is M t := ∫ t 0 A 2 s ds . Thus p t = p 0 ∗ N (0 , M t ) The explicit form of ∇ log p t ( x ) is

<!-- formula-not-decoded -->

Therefore, the forward ODE is also more explicitly written as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let y t = x T -t denote the time-reversal of the forward ODE. Then the Euler-discretization of y t is given by

<!-- formula-not-decoded -->

In the above, A t ℓ is a sequence that we assume is known a priori. M t ℓ can be computed from A t ℓ via the last line. M 0 denotes the initial Gaussian noise covariance, thus M 0 = TI . Subsequently, we simplify the notation M t ℓ → M ℓ and A t ℓ → A ℓ .

In the score-based denoising setup, we have p 0 be a discrete distribution over the training set x (1) , . . . , x ( n ) . Thus

<!-- formula-not-decoded -->

Let Z F := [ z (1) , . . . , z ( n ) ] , and Z ℓ := [ z ( n +1) ℓ ] . Observe that we are trying to denoise the query z ( n +1) ℓ , so z ( n +1) ℓ is y ℓ in (19). Therefore, (19) is equivalent to

<!-- formula-not-decoded -->

where the last line can be verified by definition of smax and rbf . Observe that z ( n +1) ℓ above is equal to Z ℓ .

By pattern matching, we see that the above coincides with the choices of

<!-- formula-not-decoded -->

which is exactly the parameter setting in the paper.

## D Miscellaneous Theory

The following standard result guarantees that Gaussian distributions, and the sum between a Gaussian distribution and an arbitrary unit-length vector, lie on a sphere.

Lemma 8 (Radius concentration under Gaussian perturbation.) . Let v ∈ S d -1 . Let u ∼ N (0 , cI d × d ) . Then for t ∈ (0 , 1) ,

<!-- formula-not-decoded -->

Proof of Lemma 8. The first equality is simply follows by (2 √ d, 4) -sub-exponential concentration of d -dimensional Gaussians.

The second equality follows by decomposing u into u ∥ = 〈 u, v ∥ v ∥ 2 〉 v ∥ v ∥ 2 and u ⊥ = u -u ∥ By the same sub-exponential bound applied to u ⊥ , we have

<!-- formula-not-decoded -->

On the other hand, we have

<!-- formula-not-decoded -->

The second inequality then follows by union bound.

## E Experiments

## E.1 General Experiment Details

In our experiments, we use both the MNIST (60,000 samples, 10 classes) and CIFAR10 (50,000, 10 classes) datasets. In each case, we use a train/test split of 9:1. All experiments are run on a single A5000 GPU.

In all our experiments, we do not use any positional encoding or tokenization of the image. Instead, we simply represent each image as a vector of all its pixel values. For MNIST, the vector dimension is 784 = 28 ∗ 28 . For CIFAR10 and CIFAR100, the vector dimension is 3072 = 3 ∗ 32 ∗ 32 .

## E.2 Additional CIFAR-10/100 Metrics and FID

FID protocol (minimal). We report Fr´ echet Inception Distance (FID) using the Inception-V3 pool 3 features (2048-D), with feature normalization enabled. Real images are the held-out test split ; fake images are generated by the same pipeline used for the corresponding figure (e.g., perlayer denoising outputs when the figure plots layerwise losses). We compute FID per configuration and report the mean ± standard deviation over k independent runs (same seeds used for CIFAR-10 and CIFAR-100 for parity). Unless otherwise noted, we use batch sizes matching the main experiments and evaluate on the full test split.

## E.2.1 Addendum to Fig. 3: score test loss - CIFAR-100 and FID

What we add. (i) CIFAR-100 analog of the loss curves (see Fig. 8); (ii) FID for CIFAR-10 and CIFAR-100 at each layer for the two trained variants (RBF, STD). We follow the same evaluation protocol as in App. E.2.

Figure 8: CIFAR-100 RMSE per layer for the curves in Fig. 3.

<!-- image -->

Table 1: Per-layer FID and RMSE for CIFAR-10 and CIFAR-100. Mean ± std over seeds.

| Dataset   | Metric   |   Layer | RBF trained           | STD trained           |
|-----------|----------|---------|-----------------------|-----------------------|
| CIFAR10   | FID      |       1 | 347 . 2712 ± 2 . 4329 | 403 . 9372 ± 0 . 9406 |
|           |          |       2 | 198 . 1683 ± 1 . 4204 | 321 . 1340 ± 1 . 1317 |
|           |          |       3 | 99 . 7453 ± 1 . 6279  | 233 . 5431 ± 1 . 5499 |
|           |          |       4 | 43 . 6949 ± 1 . 2320  | 163 . 7732 ± 0 . 9192 |
|           |          |       5 | 21 . 8445 ± 0 . 4051  | 114 . 8899 ± 0 . 5935 |
|           |          |       6 | 19 . 7110 ± 0 . 0940  | 78 . 7409 ± 0 . 1956  |
|           | RMSE     |       1 | 20 . 4389 ± 0 . 3717  | 30 . 8847 ± 0 . 0038  |
|           |          |       2 | 11 . 0475 ± 0 . 0544  | 18 . 2189 ± 0 . 0016  |
|           |          |       3 | 9 . 9866 ± 0 . 0112   | 12 . 4526 ± 0 . 0037  |
|           |          |       4 | 10 . 0475 ± 0 . 0148  | 10 . 3822 ± 0 . 0064  |
|           |          |       5 | 10 . 1168 ± 0 . 0144  | 9 . 8443 ± 0 . 0080   |
|           |          |       6 | 10 . 1454 ± 0 . 0141  | 9 . 7725 ± 0 . 0087   |
| CIFAR100  | FID      |       1 | 327 . 7921 ± 5 . 2556 | 390 . 9332 ± 1 . 5852 |
|           |          |       2 | 181 . 1612 ± 3 . 7589 | 300 . 4900 ± 2 . 2285 |
|           |          |       3 | 95 . 6904 ± 2 . 5033  | 212 . 3552 ± 2 . 2666 |
|           |          |       4 | 47 . 0779 ± 1 . 8721  | 150 . 9074 ± 1 . 3448 |
|           |          |       5 | 25 . 1974 ± 1 . 0298  | 107 . 3123 ± 1 . 1695 |
|           |          |       6 | 22 . 5363 ± 0 . 4453  | 77 . 9261 ± 0 . 9152  |
|           | RMSE     |       1 | 20 . 4177 ± 0 . 3744  | 30 . 8679 ± 0 . 0044  |
|           |          |       2 | 10 . 9805 ± 0 . 0717  | 18 . 1712 ± 0 . 0147  |
|           |          |       3 | 9 . 8963 ± 0 . 0532   | 12 . 3639 ± 0 . 0331  |
|           |          |       4 | 9 . 9517 ± 0 . 0570   | 10 . 2620 ± 0 . 0485  |
|           |          |       5 | 10 . 0194 ± 0 . 0577  | 9 . 7092 ± 0 . 0565   |
|           |          |       6 | 10 . 0477 ± 0 . 0578  | 9 . 6317 ± 0 . 0600   |

Takeaways FID decreases monotonically with layer index on both datasets, indicating progressive denoising. Across layers, RBF trained attains consistently lower (better) FID than STD trained . For RMSE, both datasets converge near 10 at the final layer; STD trained ends slightly lower than RBF trained at layer 6 (CIFAR-10: 9 . 77 vs. 10 . 15 ; CIFAR-100: 9 . 63 vs. 10 . 05 ).

## E.3 Addendum to Fig. 5: Context vs. Loss on CIFAR-100 and FID on CIFAR-10/100

We report test-set FID and RMSE across context lengths. Values are mean ± one standard deviation over three seeds. Results mirror the trend in Fig. 5: modest context substantially lowers error, with diminishing returns at larger context.

Table 2: Addendum to Fig. 5: FID and RMSE vs. context length ( n ctx). Mean ± 1 σ over 3 seeds.

CIFAR-10

| n ctx     | FID                 | RMSE                   |
|-----------|---------------------|------------------------|
| 0         | 368 . 79 ± 22 . 66  | 133 . 9387 ± 6 . 7723  |
| 2         | 102 . 12 ± 130 . 07 | 10 . 3142 ± 0 . 7683   |
| 6         | 14 . 06 ± 0 . 71    | 11 . 1293 ± 0 . 0403   |
| 16        | 30 . 14 ± 14 . 65   | 10 . 3618 ± 0 . 4681   |
| 32        | 31 . 80 ± 11 . 76   | 10 . 2821 ± 0 . 4452   |
| 60        | 47 . 28 ± 0 . 81    | 9 . 8351 ± 0 . 0090    |
| CIFAR-100 | CIFAR-100           | CIFAR-100              |
| n ctx     | FID                 | RMSE                   |
| 0         | 333 . 26 ± 77 . 20  | 150 . 0745 ± 36 . 1599 |
| 2         | 16 . 55 ± 1 . 23    | 11 . 0319 ± 0 . 1125   |
| 8         | 21 . 16 ± 4 . 74    | 10 . 7499 ± 0 . 2683   |
| 16        | 21 . 94 ± 4 . 06    | 10 . 6655 ± 0 . 2589   |
| 32        | 42 . 78 ± 1 . 26    | 9 . 9464 ± 0 . 0324    |
| 48        | 44 . 02 ± 0 . 40    | 9 . 9238 ± 0 . 0206    |

Note. FID decreases sharply with small context; RMSE shows smaller but consistent gains with additional context. CIFAR-100 exhibits the same qualitative pattern as CIFAR-10.

## E.3.1 Addendum to Fig. 7: witness/anisotropic - CIFAR-100 and FID

What we add. (i) CIFAR-100 loss vs. #witness tokens; (ii) FID and RMSE on CIFAR-10 and CIFAR-100 vs. τ for the Witness+RBF and Witness+anisotropic models. Baseline exact-score model metrics from Fig. 7 are also provided for comparison.

Table 3: Fig. 7 addendum: FID and RMSE at the final layer vs. #witness tokens ( τ ) for CIFAR-10 . Mean ± std over k seeds.

|                   | Witness+RBF   | Witness+RBF     | Witness+Anisotropic   | Witness+Anisotropic   |
|-------------------|---------------|-----------------|-----------------------|-----------------------|
| # Witnesses ( τ ) | FID           | RMSE            | FID                   | RMSE                  |
| 20                | 339.00 ± 0.70 | 7.8919 ± 0.0020 | 306.09 ± 0.34         | 7.4240 ± 0.0016       |
| 40                | 360.82 ± 0.13 | 7.5876 ± 0.0034 | 240.09 ± 0.60         | 7.2523 ± 0.0017       |
| 100               | 312.56 ± 0.02 | 7.3838 ± 0.0026 | 193.31 ± 0.56         | 7.1351 ± 0.0007       |
| 200               | 267.91 ± 1.01 | 7.2784 ± 0.0019 | 173.66 ± 0.24         | 7.0598 ± 0.0005       |
| 400               | 219.15 ± 0.86 | 7.2307 ± 0.0016 | 161.80 ± 0.70         | 7.0427 ± 0.0012       |
| 800               | 239.34 ± 0.18 | 7.2414 ± 0.0032 | 168.74 ± 0.64         | 7.0427 ± 0.0019       |
| 1000              | 271.73 ± 0.51 | 7.2924 ± 0.0014 | 168.89 ± 0.37         | 7.0462 ± 0.0003       |
| Baseline (Exact)  | 21.57         | 10.0805         | 23.57                 | 9.5456                |

Table 4: Fig. 7 addendum: FID and RMSE at the final layer vs. #witness tokens ( τ ) for CIFAR-100 . Mean ± std over k seeds.

|                   | Witness+RBF   | Witness+RBF     | Witness+Anisotropic   | Witness+Anisotropic   |
|-------------------|---------------|-----------------|-----------------------|-----------------------|
| # Witnesses ( τ ) | FID           | RMSE            | FID                   | RMSE                  |
| 20                | 313.06 ± 0.16 | 7.8954 ± 0.0020 | 284.66 ± 6.37         | 7.4390 ± 0.0055       |
| 40                | 337.84 ± 0.84 | 7.7204 ± 0.0027 | 221.24 ± 0.46         | 7.2714 ± 0.0035       |
| 100               | 306.16 ± 0.14 | 7.4170 ± 0.0027 | 181.34 ± 0.36         | 7.1272 ± 0.0032       |
| 200               | 262.20 ± 0.22 | 7.2780 ± 0.0022 | 168.53 ± 0.06         | 7.0650 ± 0.0019       |
| 400               | 222.98 ± 0.98 | 7.2226 ± 0.0038 | 164.91 ± 0.22         | 7.0323 ± 0.0036       |
| 800               | 236.45 ± 0.14 | 7.2424 ± 0.0038 | 163.78 ± 0.34         | 7.0244 ± 0.0048       |
| 1000              | 258.02 ± 0.27 | 7.2733 ± 0.0034 | 167.02 ± 0.26         | 7.0415 ± 0.0036       |
| Baseline (Exact)  | 26.64         | 9.5048          | 26.53                 | 9.5242                |

We discuss a few key observations from Tables 3 and 4:

Exact Baseline: low FID, high RMSE. RMSE measures per-image reconstruction, whereas FID measures distribution-level realism. The Baseline (Exact) model implements the exact score-based denoising via cross-attention to the full training set (Section 4.2); at test time, this dynamics pulls the noisy query toward an actual training example. This behavior is visible in Figure 4, where exact score denoising reproduces training images rather than the held-out target. Thus the model essentially 'memorizes' the training set of real images, yielding excellent FID. However, this same behavior results in a worse RMSE - when the noisy test image's ground truth is not in the training set, the model selects a different point from the data manifold (i.e., a different training image), which is often far from the clean test image.

Witness models: high FID, low RMSE. In contrast, the Witness+RBF and Witness+Anisotropic models replace full-set cross-attention with a compact, learned witness set (Section 4.3 - E.4.2). This induces a geometry-aware score approximation that avoids instance-level copying, so the denoising trajectory tracks the actual test instance rather than a nearest training neighbor, i.e. better generalization, and thus lower test RMSE. The cost is higher FID: outputs are less tightly matched to the empirical sample distribution measured by Inception features. Witness+Anisotropic consistently outperforms Witness+RBF , reflecting the benefit of geometry-aware updates.

These results highlight a generalization-realism trade-off: the Exact Baseline optimizes distributional realism (FID) by memorizing the training set, whereas Witness models trade some FID for stronger generalization to unseen test inputs.

## E.4 Implementation Details for Specific Experiments

In this section, we describe details of several experiment setups and Transformer implementations:

1. The in-context score-based denoising experiment from Section 4.2 is explained in Appendix E.4.1.
2. The experiments comparing Witness-based denoising and Anisotropic Denoising from Section 4.5 is explained in Appendix E.4.2.

## E.4.1 Implementation Details for In-Context Score-based Denoising

In this section, we provide further details on the in-context score-based denoising experiment outlined in Section 4.2. For ease of reference, we repeat below the setup described in the main paper:

Input consists of the noiseless training samples { x (1) , . . . , x ( n ) } , as well as m contextual images { x ( n +1) , . . . , x ( n + m ) } . The query image is x ( n + m +1) . Importantly, all m contextual images belong to the same class as the query image x ( n + m +1) . The Transformer input is z ( i ) = x ( i ) for i = 1 , . . . , n + m , and the query point is perturbed with Gaussian noise: z ( n + m +1) = x ( n + m +1) + N (0 , σ 2 I ) . Wecall m the 'context length'. In terms of implementation, the query x ( n + m +1) attends to the training set x (1) , . . . , x ( n ) via the CrAttn module in (7), while the context samples + query x ( n +1) , . . . , x ( n + m ) attend to each other via the Attn module in (7) (with appropriate masking).

The Transformer architecture is as defined in (7):

<!-- formula-not-decoded -->

We let

<!-- formula-not-decoded -->

denote the context plus queries, and let

<!-- formula-not-decoded -->

denote the frozen training set. Since the context tokens in the first m columns of Z 0 are noiseless, we enforce that they are not updated using an additional mask. In summary,

<!-- formula-not-decoded -->

Note that Z ( i ) ℓ +1 denotes the i th column of Z ℓ and corresponds to token z ( n + i ) .

Each parameter is constrained to be a scalar multiple of identity, i.e.

<!-- formula-not-decoded -->

so the total trainable parameters are { w S ℓ , w Q ℓ , w K ℓ , w V ℓ , w Q ℓ ′ , w K ℓ ′ , w V ℓ ′ } ℓ =1 ,...,L ⊂ R .

## E.4.2 Implementation Details for Witness and Anisotropic+Witness Experiments

In this section, we provide further details on the in-context score-based denoising experiment outlined in Section 4.5. For ease of reference, we repeat below the setup described in the main paper:

Input consists of the noiseless training samples { x (1) , . . . , x ( n ) } . The query image is x ( n +1) . The Transformer input is z ( i ) = x ( i ) for i = 1 , . . . , n , and the query point is perturbed with Gaussian noise: z ( n +1) = x ( n +1) + N (0 , σ 2 I ) . In terms of implementation, the query x ( n +1) attends to the training set x (1) , . . . , x ( n ) via the CrAttn module in (14).

The Transformer architecture is as defined in (14):

<!-- formula-not-decoded -->

in the above, we leave out the Attn module since it is not used. For all subsequent discussion, U ℓ ∈ R d × S is a trainable parameter, where S is the number of witnesses in one layer. We initialize each U ℓ by setting its i th column to be a randomly drawn training sample from x (1) , . . . , x ( n ) . Note that each layer ℓ has a separate U ℓ .

For Witness+RBF, we have

<!-- formula-not-decoded -->

where Z (1) ℓ +1 corresponds to the query token z ( n +1) . Each parameter is constrained to be a scalar multiple of identity, i.e.

<!-- formula-not-decoded -->

so the total trainable parameters are the learned witnesses { U ℓ } ℓ =1 ,...,L ⊂ R d × S , and the (scalar) parameters { w S ℓ , w Q ℓ ′ , w K ℓ ′ , w V ℓ ′ } ⊂ R .

<!-- formula-not-decoded -->

For Witness+Anisotropic, we have

<!-- formula-not-decoded -->

where Z (1) ℓ +1 corresponds to the query token z ( n +1) . Each parameter is constrained to be a d × d diagonal matrix, so the total trainable parameters are the learned witnesses { U ℓ } ℓ =1 ,...,L ⊂ R d × S and the diagonal parameter matrices { W S ℓ , W Q ℓ ′ , W K ℓ ′ , W V ℓ ′ } ℓ =1 ,...,L ⊂ R d × d (each parameterized by the d diagonal scalars).

## F Future Directions

Parameter-efficient denoising. Our characterization of anisotropic denoising with diagonal attention weights suggests that even highly constrained parameterizations can learn powerful denoising algorithms. This opens up opportunities for efficient Transformer variants in generative modelsfor example, using low-rank or diagonal attention in diffusion Transformers, or fine-tuning these components in parameter-efficient learning settings.

Localized, interpretable attention. We observe that trained Transformers often specialize different layers (or heads) to denoise approximately non-overlapping semantic patches (Figure 1). This suggests a promising direction for designing sparse, locality-aware attention mechanisms in structured domains like vision. Such mechanisms may be more interpretable and robust to noisy or incomplete context, and they align with the geometry-aware updates formalized in our analysis.

Multi-modal Contexts Incorporating conditioning signals (e.g., text) via cross-attention into our kernel-weighted update view provides a principled path to conditional denoising and controllable generation, and can be combined with parameter-efficient anisotropic modules.

## G Limitations:

Our analysis and experiments deliberately focus on small-scale image datasets and small-depth, single-head Transformers. While this choice isolates the denoising mechanisms we study, it leaves open how the same constructions behave on larger, more varied data (e.g., high-resolution images, audio, or text) and in deeper, multi-stage architectures. Second, we evaluate only pixel-space RMSE; assessing perceptual quality, class fidelity, or downstream generation tasks remains future work.