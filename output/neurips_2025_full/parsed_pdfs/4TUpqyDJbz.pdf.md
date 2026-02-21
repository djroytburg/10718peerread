## When and How Unlabeled Data Provably Improve In-Context Learning

Yingcong Li 1 , 4 Xiaofeng Liu 1

Xiangyu Chang 2 Amit Roy-Chowdhury 2

Muti Kara 3 Samet Oymak 1

1 University of Michigan 2 University of California, Riverside 3 Bilkent University 4 NJIT

## Abstract

Recent research shows that in-context learning (ICL) can be e ff ective even when demonstrations have missing or incorrect labels. To shed light on this capability, we examine a canonical setting where the demonstrations are drawn according to a binary Gaussian mixture model (GMM) and a certain fraction of the demonstrations have missing labels. We provide a comprehensive theoretical study to show that: (1) The loss landscape of one-layer linear attention models recover the optimal fully-supervised estimator but completely fail to exploit unlabeled data; (2) In contrast, multilayer or looped transformers can e ff ectively leverage unlabeled data by implicitly constructing estimators of the form P i ≥ 0 ai ( X ⊤ X ) i X ⊤ y with X and y denoting features and partially-observed labels (with missing entries set to zero). Wecharacterize the class of polynomials that can be expressed as a function of depth and draw connections to Expectation Maximization, an iterative pseudo-labeling algorithm commonly used in semi-supervised learning. Importantly, the leading polynomial power is exponential in depth, so mild amount of depth / looping su ffi ces. As an application of theory, we propose looping o ff -the-shelf tabular foundation models to enhance their semi-supervision capabilities. Extensive evaluations on real-world datasets show that our method significantly improves the semisupervised tabular learning performance over the standard single pass inference.

## 1 Introduction

In-context learning (ICL) is an intriguing capability of modern language models and has enjoyed remarkable empirical success (Brown et al., 2020; Min et al., 2022). This success is also being extended to multimodal scenarios (Zhou et al., 2024) as well as other modalities such as tabular data (Hollmann et al., 2022). The push toward test-time scaling and long-context models (Snell et al., 2024; Guo et al., 2025) has further boosted the benefits of ICL by allowing the model to ingest a large number of demonstrations. For instance, in 'Many-shot in-context learning' paper, Agarwal et al. (2024) demonstrate that pushing more examples into context window can substantially boost the accuracy. MAPLE (Chen et al., 2025) improves many-shot ICL by pseudo-labeling highimpact unlabeled examples and incorporating them into the prompt. The many-shot ICL setting naturally raises the question of when and how ICL can succeed with weaker supervision. As we can harness longer context models to boost predictive accuracy, we may indeed run out of high-quality demonstrations with verified answers / chain-of-thoughts and may want to utilize weaker data sources. This motivates our central question:

## Q: When and how can transformers learn in context from unlabeled data?

We primarily investigate this question under a semisupervised ICL (SS-ICL) setting with Gaussian mixture models (GMMs). Formally, given a prompt containing a dataset of feature-label pairs ( x i , yi ) n i = 1 ∈ R d × R as demonstrations and a query feature x (see Eq. (3)), a model trained for ICL

learns to predict the corresponding output y given prompt. For ICL with a supervised binary GMM model, we have x i ∼ N ( µ yi , σ 2 I ) and yi ∈ {-1 , 1 } , i ∈ [ n ], and the component means µ ± 1 that parameterize the classification task are sampled from a prior task distribution. This prompt model is well studied under various fully-supervised settings (Garg et al., 2022; Von Oswald et al., 2023; Ahn et al., 2023; Akyürek et al., 2023; Mahankali et al., 2024; Collins et al., 2024; Shen et al., 2024) where each demonstration includes a clearly labeled output. In our SS-ICL setting, only m out of n total samples have correct labels ( m ≤ n ) either -1 or 1, and remaining labels are unknown and fed to the model as yi = 0.

In this work, we provide a comprehensive theoretical and empirical study of attention models with varying depths when trained with SS-ICL. Our analysis reveals the importance of depth : Despite being able to implement the optimal fully-supervised estimator, single-layer linear attention completely fails to leverage unlabeled examples. In contrast, deeper or looped transformer architectures can emulate strong semi-supervision algorithms, approaching the performance of the Bayes-optimal classifier as depth increases. Informed by the importance of depth / looping, we also devise semisupervision strategies for tabular foundation models. Our specific contributions are:

- ⋄ Landscape of one-layer linear attention ( § 3): We study the optimization landscape of singlelayer linear attention for the SS-ICL problem under an isotropic task prior. We prove that the global minimum of the loss function returns the plug-in estimator (see Eq. (SPI)), i.e., ˆ y = sgn( x ⊤ ˆ µ ) with ˆ µ = X ⊤ y , where X ∈ R n × d represents features and y ∈ R n denotes partially-observed labels (with missing entries set to zero) of the ICL demonstrations. This implies that 1-layer model learns Bayes-optimal classifier in the fully-supervised setting, but completely fails to make use of unlabeled data.
- ⋄ Depth is crucial but shallow can su ffi ce (§4): We show that multilayer linear attention can emulate semisupervised learners by implementing polynomial estimators of the form

<!-- formula-not-decoded -->

Crucially, an L -layer (or looped) attention can express up to K = O (3 L ) powers, highlighting that logarithmic depth su ffi ces to represent high-degree monomials. We provide characterizations of the set of expressible polynomials through di ff erent constructions (where each layer gets to update the features or labels of the previous layer). Corroborating these, experiments reveal that shallow models with L ≥ 2 already achieve strong results and their performance can be approximately predicted through an eigen-estimator combining i = 0 and ∞ (see (SSPIk )).

- ⋄ What learner attention emulates? In Section 4.3, we describe how each attention block can update the label estimates by emulating expectation-maximization (for linear attention) or belief propagation (for softmax attention). For instance (1) can be interpreted as the model implicitly conducting an Expectation-Maximization algorithm: Starting with the supervised estimator ˆ µ 0 = X ⊤ y , each term ( X ⊤ X ) i X ⊤ y can be viewed as a sequence of pseudo-labeling (expectation) ˆ y i = X ˆ µ i -1 and training (maximization) ˆ µ i = X ⊤ ˆ y i steps. Corroborating this, we show that softmaxattention and softmax-transformer models similarly benefit from increasing depth and can emulate semisupervised learners competitive with Bayes limit (see Fig. 2c).
- ⋄ Applications to Tabular FMs (§5): Tabular foundation models such as TabPFN (Hollmann et al., 2022, 2025), TabICL (Qu et al., 2025) and TabDPT (Ma et al., 2025) represent a suitable application of theory as they also model the ICL examples with a single token. To harness unlabeled examples, we propose a novel strategy that iteratively creates soft pseudo-labels by explicitly looping the tabular FM while controlling validation risk. Focusing on the few-shot learning setting where TabPFN-v2 (Hollmann et al., 2025) excels, we demonstrate that our approach can significantly improve predictive performance on various real-world datasets.

## 1.1 Related Work

Theoretical Analysis of In-Context Learning Recent work has developed theoretical frameworks for understanding in-context learning in transformers. Akyürek et al. (2023), Von Oswald et al. (2023) and Dai et al. (2023) demonstrated that transformers emulate gradient descent during ICL. Xie et al. (2022) o ff ered a Bayesian perspective, while Zhang et al. (2024) showed transformers learn linear models in-context. Ahn et al. (2023) established they implement preconditioned gradient

descent, and Mahankali et al. (2024) proved one-step gradient descent is optimal for single-layer linear attention. Multiple works (Li et al., 2023; Yang et al., 2024; Li et al., 2024; Bai et al., 2023; Shen et al., 2024) studied the generalization capability of transformers. However, these exclusively focus on fully-supervised settings, leaving a critical gap in understanding how transformers handle partially labeled data-a common real-world scenario. Our work addresses this gap by providing the first theoretical characterization of semi-supervised in-context learning. Wang et al. (2024) considers a setting where the model observes demonstrations of the form (query, response i , reward i ) and aims to correct its response based on the reward sequence. Our work has a di ff erent focus as it highlights that the model can correct / impute the missing labels using implicit feedback from labeled demonstrations.

Semi-Supervised Learning Traditional semi-supervised learning (SSL) aims to leverage unlabeled data to improve classifier performance. For linear classifiers, Oymak &amp; Gulcu (2021) characterized self-training iterations and demonstrated rejecting low-confidence samples; further theoretical analyses of self-training / pseudo-labeling cover deep networks (Wei et al., 2020). For Gaussian Mixture Models (GMMs), Lelarge &amp; Miolane (2019) quantified maximal improvement from unlabeled data, while Krishnapuram et al. (2004) developed graph-based priors. Learning GMMs via Expectation-Maximization (EM) or pseudo-labeling, especially with few labels, is well-studied. Ratsaby &amp; Venkatesh (1995) provided early PAC-style bounds for GMMs learned from few labeled and many unlabeled points. Balakrishnan et al. (2017) o ff ered further statistical guarantees for EM. Nigam et al. (2000) demonstrated empirically that EM (viewable as iterative pseudo-labeling Xu et al. (2024)) with pseudo-labels significantly reduces text classification error using unlabeled documents. These foundational works, with ongoing research in areas like agnostic learning (Kwon &amp; Caramanis, 2020) underpin many SSL concepts. While these works established fundamental principles, they did not consider how these concepts apply to in-context learning with transformers. A most recent concurrent work (Liu &amp; Yang, 2026) makes a similar observation to ours, showing that softmax attention approximates an EM estimator in a sem-supervised ICL setting, but with a di ff erent focus on the underlying model and data regime. Our contribution bridges this gap by showing how transformer depth enables e ff ective utilization of unlabeled examples within the prompt, essentially implementing semi-supervised learning without parameter updates.

## 2 Problem Setup and Preliminaries

We study ICL in the setting of semi-supervised classification, where the in-context demonstrations are drawn from a binary Gaussian mixture model (GMM). We begin by introducing the following core notation: Denote the set { 1 , 2 , · · · , n } as [ n ] and use bold letters, such as x and X , to represent vectors and matrices, respectively. Let Q ( · ) function return the right tail of the standard normal distribution.

<!-- formula-not-decoded -->

## 2.1 Semi-supervised Data Model

Consider a d -dimensional semi-supervised binary GMM with n examples ( x i , yi ) n i = 1 , where x i ∈ R d denotes the feature vector and yi ∈ {-1 , 0 , 1 } represents the corresponding observed label, with yi = 0 indicating a missing label, and each label is revealed independently with probability p ∈ [0 , 1]. Specifically, the data is generated as follows (for each i ∈ [ n ]):

<!-- formula-not-decoded -->

Here µ ∼ Unif( S d -1 ) denotes the task mean, which is sampled uniformly from the unit sphere, and ξ i ∼ N (0 , σ 2 I ) is the random noise with σ ≥ 0 being the noise level that controls the variability of x i around its mean. y c i denotes the true class label that is uniform over {-1 , 1 } . Observe that p = 1 corresponds to fully-supervised learning and p = 0 corresponds to fully-unsupervised learning.

## 2.2 In-context Learning and Linear Attention

We build on the setting of (Garg et al., 2022; Mahankali et al., 2024; Zhang et al., 2024; Li et al., 2024) and construct the in-context prompts with examples drawn from the model (2) as follows.

Prompt Generation Given a task vector µ ∼ Unif( S d -1 ), we sample ( n + 1) in-context demonstrations ( x i , yi ) n + 1 i = 1 according to (2) and construct the prompt

<!-- formula-not-decoded -->

We will investigate training a transformer such that given Z as prompt, it correctly predicts the label y : = y c n + 1 of the query x : = x n + 1 through ICL.

Model Architecture Our work primarily focuses on training of linear attention models. Given any prompt Z ∈ R ( n + 1) × ( d + 1) , which can be treated as a sequence of ( d + 1)-dimensional tokens, the linear attention mechanism outputs

<!-- formula-not-decoded -->

where W : = { W k , W q , W v ∈ R ( d + 1) × ( d + 1) } denotes the set of the key, query and value weight matrices. Therefore, given the prompt matrix Z ∈ R ( n + 1) × ( d + 1) as input, the attention mechanism outputs a ( n + 1)-length sequence (i.e., att( Z ; W ) ∈ R ( n + 1) × ( d + 1) ). Note that the label for the query x is excluded from the prompt Z . Similar to Ahn et al. (2023), we consider a training objective with a mask

M = " I n 0 0 0 # to prevent input tokens from attending to the queries. To ensure that all in-context examples are treated equally and that the model remains invariant to their order / position, we do not apply a causal mask following Ahn et al. (2023). In contrast, Li et al. (2025) explores the use of causal masking in multi-layer linear attention and analyzes its impact on the final prediction.

Building upon the single-layer linear attention mechanism of (4), we can extend our model to multiple layers to capture more complex patterns. Consider optimizing an L -layer linear attention model and let Z ℓ be the input of ℓ th layer, ℓ ∈ [ L ]. Additionally, let W ℓ : = { W k ℓ , W q ℓ , W v ℓ ∈ R ( d + 1) × ( d + 1) } be the corresponding weight matrices of ℓ th layer. Then, recalling the attention mechanism (4), the input prompt of ℓ th layer is defined by

<!-- formula-not-decoded -->

and Z 1 = Z . We focus on the next-token prediction setting, where the model makes a prediction based on the final query token [ x ⊤ 0] ⊤ . Let h ∈ R d + 1 denote the linear prediction head. We define the output of the L -layer linear attention model at the last (query) token as

<!-- formula-not-decoded -->

Recalling the sign function, the predicted label for x is given by y attL ( Z ) = sgn( f attL ( Z )).

Model Training With our attention-based architecture established, we now turn to the training procedure and evaluation metrics. Consider the ICL setting where each input prompt Z (cf. (3)) corresponds to a randomly sampled task vector µ ∼ Unif( S d -1 ) and let ℓ ( · ) : R → R be the loss function. Additionally, define the set of attention weights W ( L ) : = ∪ L ℓ = 1 W ℓ ∈ ( R ( d + 1) × ( d + 1) ) 3 L . The objective of L -layer linear atention takes the following form:

<!-- formula-not-decoded -->

Here, y = y c n + 1 and the expectation subsumes the randomness of µ and ( ξ i , yi ) n + 1 i = 1 . The search space for W ( L ) is ( R ( d + 1) × ( d + 1) ) 3 L , and for h is R d + 1 .

## 3 Loss Landscape of One-layer Linear Attention under SS-ICL

Previous work (Ahn et al., 2023; Li et al., 2024; Mahankali et al., 2024) has shown that an optimized single-layer linear attention implements a form of preconditioned gradient descent over the linear in-context demonstrations provided within the prompt. However, to the best of our knowledge, prior studies have not addressed the semi-supervised setting, where some in-context labels are missing. In this section, we analyze the optimization behavior of single-layer linear attention under the semisupervised binary GMM setting described in Section 2, and demonstrate that the single-layer model learns the optimal fully-supervised learner, but fails to utilize the unlabeled data.

We begin with the following optimal supervised label estimator under our problem setting.

Supervised Plug-in (SPI) Estimator The plug-in method is a classical approach for supervised classification problems, aiming to find a linear combination of features that separates di ff erent categories. Under our problem setting, it also serves as the asymptotically Bayes-optimal estimator given only labeled data (Hastie et al., 2009; Devroye et al., 2013). Consider the binary semi-supervised GMMproblem described in (2) with dataset ( x i , yi ) n i = 1 , and let I ⊂ [ n ] represent the indices of labeled samples, e.g., yi , 0 for i ∈ I . The SPI estimator returns the task mean

<!-- formula-not-decoded -->

We next present the following theorem establishes that, under isotropic task prior, optimal single-layer linear attention is equivalent to the SPI estimation.

Theorem 1 Let the prompt (cf. (3) ) be generated as described in Section 2.2. Consider the objective (cf. (7) ) with L = 1 and squared loss function ℓ ( y , ˆ y ) = ( y -ˆ y ) 2 , and denote the optimal prediction as y ⋆ att1 ( Z ) . Let ˆ µ s represent the SPI estimator defined in (SPI) . Then, for any Z from (3) , we have

<!-- formula-not-decoded -->

Additionally, its classification error obeys

<!-- formula-not-decoded -->

where we define εσ = σ/ √ np and X 2 d defines chi-squared distribution with d degrees of freedom.

The proof of Theorem 1 is deferred to Appendix B. Eq. (8) shows that one-layer linear attention model indeed implements the optimal supervised predictor, assuming access to np labeled examples. Therefore, the classification error corresponds exactly to that of the SPI estimator. The supervised classification problem has been extensively studied (Bartlett et al., 2006; Belkin et al., 2018; Montanari et al., 2019; Thrampoulidis et al., 2020; Chatterji &amp; Long, 2021; Cao et al., 2021; Wang &amp; Thrampoulidis, 2022; Deng et al., 2022), with most existing work focusing on a single classification task in asymptotic data or overparameterized regimes. In contrast, within the ICL framework considered in our setting, the task mean µ is randomly sampled, and the classification error is computed by averaging over random draws of Z , y , and µ . Accordingly, in (9), we express the error in a simplified form as an expectation.

The experimental results in Figure 1 support Theorem 1, where dark blue circular markers represent the performance of the single-layer linear attention model, blue curves show the classification accuracy of the SPI estimator, and the red dotted curves depict the accuracy 1 -P ( y ⋆ att-1 ( Z ) , y ) as computed from (9). The alignments of these curves empirically validate Theorem 1. Implementation details and further discussion are provided in Section 5. Based on these results, we reach the following conclusion:

1-layer linear attention learns optimal supervised estimator but doesn't benefit from unlabeled data.

As shown in Figs 1b and 1c, when the number of labeled samples ( np = 10) is fixed, increasing the number of unlabeled examples (even up to ∼ 10000) has no e ff ect on performance, as the dark blue markers remain at the same level.

At first glance, this may seem counterintuitive-while the data is unlabeled, it still contains information about the classification feature. For instance, the mean of the data points carries relevant information, and one might expect the model to extract and leverage this for better predictions. This expectation is particularly reasonable when a large amount of unlabeled data is available, as the sample covariance matrix approximates the population covariance, i.e., E [ X ⊤ X / n ] = µµ ⊤ + σ 2 I where X = [ x 1 , x 2 , · · · , x n ] ⊤ ∈ R n × d . The key insight into why single-layer attention fails to leverage unlabeled data lies in the expectation structure. In our isotropic GMM setting where µ ∼ Unif( S d -1 ), the sample covariance matrix converges to E [ X ⊤ X / n ] = E [ µµ ⊤ ] + σ 2 I = (1 / d + σ 2 ) I , which contains no task-specific information. The expectation across multiple tasks loses the signal from µ . This

Figure 1: Experimental results support our theoretical findings presented in Sections 3 and 4. In all three subfigures, blue, green, and orange markers represent the results of 1-, 2-, and 5-layer linear attention models, respectively. The SPI estimator (cf. (SPI)), SSPI-1, and SSPI-∞ (cf. (SSPIk )) are shown as blue solid, green solid, and green dotted curves, respectively. The red dotted curves in all subfigures correspond to the singlelayer / SPI results described in Eq. (9) of Theorem 1, while the black dotted line in Fig. 1c corresponds to Eq. (13) of Theorem 2. Additional details and discussion can be found in Sections 3, 4, and 5.

<!-- image -->

explains why single-layer attention, operating in a meta-learning framework across many tasks rather than optimizing for a single fixed task, cannot extract useful information from unlabeled data.

In the following section, we study multi-layer linear attention and demonstrate that it has the ability to propagate X ⊤ X into deeper layers, thereby enabling the model to utilize the unlabeled data.

## 4 Multi-layer Attention and the Benefits of Depth

In this section, we explore how deeper attention models can e ff ectively utilize the unlabeled data. Let

<!-- formula-not-decoded -->

## 4.1 L -layer Linear Attention can Implement DegreeO (3 L ) Polynomials in X ⊤ X

We first present the following propositions to show that multi-layer as well as looped linear attention can be expressed as a polynomial function of X ⊤ X . This structure allows the models to leverage unlabeled data to improve the estimation of the task mean µ .

Proposition 1 Given an L-layer linear attention model described in Section 2.2 with input prompt Z defined in (3) , one can construct the key, query, value weight matrices and the linear prediction head such that the model outputs (cf. (6) )

<!-- formula-not-decoded -->

Then, the following A matrices are achievable via label and feature updates:

- Label propagation: A = c Q L -1 ℓ = 1 GLYPH&lt;0&gt; I + c ℓ X ⊤ X GLYPH&lt;1&gt; for arbitrary constants { c , c 1 , · · · , cL -1 } ;
- Feature propagation: A = c GLYPH&lt;0&gt; X ⊤ X GLYPH&lt;1&gt; 3 L -1 -1 for an arbitrary constant c.

Proposition 2 Consider the same setting as in Proposition 1. There exists a single-layer linear attention model whose parameters can be constructed such that, when looped L times, its output reproduces that of (11) , with c ℓ ≡ c ′ for some arbitrary constant c ′ .

The proofs of Proposition 1 and 2 are deferred to Appendix C.1 and C.2. In the following, we provide further clarification on the label and feature propagation.

1. The final prediction of the label propagation process can be rewritten as

<!-- formula-not-decoded -->

with y 1 = y . Here, y ℓ can be interpreted as the soft pseudo-labels input to the ℓ th layer, and each c ℓ is parameterized by the attention mechanism in the corresponding layer. Although not

exactly equivalent, the L -layer linear attention process shares similarities with the ExpectationMaximization (EM) algorithm for semi-supervised learning, with L iterations of pseudo-labeling and a di ff erent label update strategy.

2. In contrast, the feature propagation process yields the final prediction

<!-- formula-not-decoded -->

with X 1 = X and x 1 = x . Here, ( X ℓ , x ℓ ) can be viewed as the input features at the ℓ th layer, encoding exponentially higher-order powers of X ⊤ X . This result highlights that a linear attention model requires only O (log K ) layers to represent polynomial functions of degree K .

Our construction for label propagation is inherently related to the gradient descent emulation capability of linear attention Ahn et al. (2023). However, the feature propagation construction is fundamentally di ff erent and underscores the transformer's capability to implement rapid power iteration over the empirical covariance X ⊤ X . In the above constructions, each attention block with residual connections updates features or labels using one parameter, namely mappings of the form X → X + α XX ⊤ X or y → y + β XX ⊤ y . The lemma below shows that, even if the multilayer model can express polynomials of X ⊤ X with exponential degrees in depth, the expressible manifold of polynomials has dimensionality linear in depth.

Lemma 1 (Label + Feature Propagation) For an L-layer linear attention model, the resulting eventual prediction corresponds to the matrix A in Proposition 1 of the form

<!-- formula-not-decoded -->

The coe ffi cients a : = [ a 0 a 1 · · · a (3 L -3) / 2 ] ⊤ lie on a manifold of dimension at most 2 L as a can be expressed as a = g ( c ) for some smooth function g : R 2 L → R (3 L -3) / 2 with c representing the parameters of individual layers.

## 4.2 Which Semi-supervised Algorithm Does Multi-layer Attention Approximate?

Recall the SPI estimator ˆ µ s from (SPI), and that y denotes the visible labels defined in Section 2.1 and (10). We have ˆ µ s = 1 |I| X ⊤ y . Motivated by Proposition 1 that multi-layer linear attention can implement higher-degree polynomials of X ⊤ X , we introduce the following SSPI estimator, which makes predictions based on the supervised estimate ˆ µ s combined with higher-order debiased term of the form ( X ⊤ X / n -σ 2 I ) k .

Semisupervised Plug-in (SSPI) Estimator Observe that the feature covariance satisfies E [ X ⊤ X ] / n = µµ ⊤ + σ 2 I , and the top eigenvector of the centered covariance matrix ( X ⊤ X / n -σ 2 I ) asymptotically aligns with either µ or -µ . Therefore, with a substantial amount of unlabeled data, we propose the semisupervised plug-in (SSPI) estimator as follows:

<!-- formula-not-decoded -->

where ˆ µ s is the SPI estimator (cf. (SPI)), and α ∈ [0 , 1] controls the trade-o ff between the fullysupervised and semi-supervised estimators. The optimal choice of α depends on the problem parameters n , d and p . Note that as k →∞ , the term ( X ⊤ X / n -σ 2 I ) k converges (up to scaling) to a rank-one projection onto the top eigenvector of the debiased covariance matrix, e ff ectively serving as an estimator for µ (up to sign).

In Figure 1, we present the prediction accuracies of 2-layer and 5-layer linear attention models, shown by green and orange markers, respectively. We also evaluate the SSPI algorithm with varying k values, where the green solid curve corresponds to SSPI-1, and the green dotted represents SSPI-∞ , both using their respective optimal choices of α . Details on selecting the optimal α values are provided in Section A.1 and illustrated in Figure 2. The results reveal a close alignment between multi-layer linear attention and SSPI estimators. Notably, the 2-layer model outperforms SSPI-1, due to its ability to implement higher-degree polynomials of X ⊤ X (cf. Proposition 1 and Equation (12)). When the sample size is su ffi ciently large (e.g., n &gt; 50 in Figure 1b), the top eigenvector provides a more accurate estimate of the task mean, enabling SSPI-∞ to achieve higher accuracy. Furthermore, since

the 5-layer model is capable of representing higher-order functions than the 2-layer model, it can better estimate the top eigenvector, resulting in performance that closely matches that of SSPI-∞ .

In the following, we analyze the optimal classifier of the form sgn( x ⊤ A ˆ µ s ) for a GMM, and provide insights into its behavior in the asymptotic regime as n →∞ .

Theorem 2 Consider a binary GMM defined in Section 2.1 and suppose that ( x i , yi ) n + 1 i = 1 is generated using a fixed µ following (2) . Given matrix A ∈ R d × d , define prediction

<!-- formula-not-decoded -->

where ˆ µ s is the SPI estimator defined in (SPI) . Let A ⋆ : = min A ∈ R d × d P (ˆ y A , y ) be its optimal solution set. Then, µµ ⊤ ∈ A ⋆ . Additionally, it obeys

<!-- formula-not-decoded -->

Note that, P (ˆ y µµ ⊤ , y ) depends on np and σ only, regardless of µ and d .

Theorem 3 Let the prompt Z be generated as described in Section 2.2, and consider an L-layer linear attention model with L ≥ 2 and n = ∞ . Additionally, let ˆ µ s be the SPI estimator defined in (SPI) . There exist model constructions such that for any Z following (3) , its prediction satisfies

<!-- formula-not-decoded -->

The proof follows directly from Proposition 1 (label propagation), which shows that multi-layer linear attention can output x ⊤ ( X ⊤ X / n -σ 2 I ) ˆ µ s . As n →∞ , the empirical covariance converges to its expectation, i.e., X ⊤ X / n -σ 2 I → µµ ⊤ . The results in Figure 1c validate Theorem 3, showing that as n becomes large enough (i.e., n = 10000), the predictions from both 2-layer and 5-layer linear attention models, as well as the SSPI-1 and SSPI-∞ estimators, closely align with the classification error characterized in Theorem 2, depicted by the black dotted line.

Theorem 3 establishes that, with infinitely many unlabeled samples, an L -layer linear attention model (for L ≥ 2) can implement the predictor characterized in Theorem 2 using the optimal choice of A , thereby achieving the classification error specified in (13). In the following, we shift to the non-asymptotic setting where n is finite and analyze the model's performance in this regime.

Theorem 4 Let the prompt Z be generated as described in Section 2.2. Consider an L-layer linear attention model with L ≥ 2 and denote its optimal prediction as y ⋆ att-L ( Z ) . Additionally, let ˆ µ s be the SPI estimator defined in (SPI) . Suppose that the number of labeled samples satisfies np ≥ 8 d σ 2 and n &gt; O ( d ) is su ffi ciently large. Then, there exists a universal constant C &gt; 0 such that the classification error satisfies

<!-- formula-not-decoded -->

The proof is deferred to Appendix C.5. Note that when n ≫ d , the classification error approaches the Bayes error, i.e., P ( y ⋆ attL ( Z ) , y ) ≈ Q (1 /σ ).

## 4.3 Multi-layer Attention as Expectation Maximization and Belief Propagation

In Section 4.1, we discussed how multi-layer linear attention can express polynomial functions of X ⊤ X . Here, we further explore the connection between multi-layer attention and the Expectation Maximization (EM) algorithm for semi-supervised learning. Beyond linear attention, we also highlight key di ff erences between linear and softmax-based attention mechanisms, particularly in how they implement labeling strategies analogous to those in the EM algorithm.

Consider the following construction of the ℓ -th layer attention weights:

<!-- formula-not-decoded -->

We examine both linear and softmax attention mechanisms. Let S ( · ) denotes the softmax operation that applies on the rows of a matrix. With this, the data update defined in (5) becomes:

Linear attention:

$$y ℓ + 1 = y ℓ + c ℓ XX ⊤ y ℓ$$

Softmax attention:

$$y ℓ + 1 = y ℓ + c ℓ S ( XX ⊤ ) y ℓ$$

In the case of linear attention, given the pseudo-labels y ℓ = [ y ℓ 1 , y ℓ 2 , . . . , y ℓ n ] at layer ℓ , the model estimates the task mean using the SPI algorithm (cf. (SPI)) as ˆ µ ℓ = X ⊤ y ℓ . The attention then updates each pseudo-label through the residual rule:

<!-- formula-not-decoded -->

where c ℓ is a layer-specific coe ffi cient. Owing to the linearity of this mechanism, the resulting pseudo-labeling strategy aligns with a linear EM-style update.

In contrast, softmax attention computes pairwise similarities via the softmax of dot products. Define si j = e x ⊤ i x j P j ≤ n e x ⊤ i x j , then each pseudo-label is updated via:

<!-- formula-not-decoded -->

This update is a nonlinear, similarity-weighted pseudo-labeling strategy which can also be viewed as belief propagation . The nonlinear nature highlights a key distinction between softmax and linear attention in how they emulate EM-like updates ( si j = x ⊤ i x j for linear attention).

## 5 Experiments

In Sections 3 and 4, we introduced Figure 1 and demonstrated its consistency with our theoretical results. In this section, we describe the experimental setup and implementation details. Motivated by Proposition 2, which suggests that looping can help leverage unlabeled data, Section 5.1 introduces an algorithm based on the TabPFN, showing how it can enhance prediction performance by incorporating a small amount of unlabeled data and iterative pseudo-labeling through model looping. Additionally, we present further empirical findings to investigate additional questions of interest in Section A.1.

Experimental Setup Following Section 2, set d = 10 and noise level σ = 1. All models are trained using Adam optimizer with a learning rate of 10 -3 for 40,000 epochs, with a batch size of 512. We use logistic loss in our experiments. Since our study focuses on the optimization landscape and model expressivity, and experiments are implemented via gradient descent, we repeat 10 trainings from random initialization and results are presented as the maximal test accuracy among those 10 trails.

## 5.1 Tabular Experiments

To investigate how model looping (Proposition 2) can improve label prediction, we propose the LoopTabFM algorithm that addresses unlabeled data by iteratively assigning pseudo-labels. More details of the algorithm are deferred to Section A.2 and Algorithm 1. 1

We evaluated the e ff ectiveness of our proposed looping strategy by iteratively applying TabPFN-v2 on real-world binary classification benchmarks used in Hollmann et al. (2025). The results are summarized in Table 1, where each entry represents an average over 100 random splits of the dataset, with 80% of the data used as the test set in each split.

For each experiment, we randomly sample 10 labeled and 10 unlabeled examples, ensuring that the labeled set includes at least one example from each class. As a baseline (Loop-0), we apply TabPFNv2 using only the labeled data. The corresponding test accuracies are reported in the 'Loop-0' column of Table 1. We compare this to models updated through up to k ≤ 5 iterations of pseudo-label update, with results shown in the 'Loopk ' columns. The final column reports the relative improvement (Rel. Imp.) over the baseline. Our results demonstrate that the looping strategy can significantly improve test accuracy. For instance, on OpenML datasets 1049, 1464, 40701, and 40983, accuracy improves by more than 10% over the baseline using only 10 additional unlabeled samples. The last row of

1 Our code is available at https://github.com/xiaofengliu-water/LoopTabFM .

Table 1: Comparison of test accuracy (%) between the baseline (Loop-0) and LoopTabFM (Algorithm 1) after 1 to 5 iterations using TabPFN-v2. Each result is averaged over 100 random trials. The highest test accuracy for each dataset is highlighted in bold. The final column reports the relative improvement (%) of Loop-5 over the baseline, computed as (Loop-5 -Loop-0) / Loop-0 × 100%. Positive signs indicate a performance improvement over the baseline, while negative signs indicate a performance drop.

| OpenML ID   | # of features   | # of samples   | Class imbalance   |   Loop-0 |   Loop-1 |   Loop-2 |   Loop-3 |   Loop-4 |   Loop-5 | Rel. Imp. (%)   |
|-------------|-----------------|----------------|-------------------|----------|----------|----------|----------|----------|----------|-----------------|
| 3           | 36              | 3196           | 1.09              |    58.62 |    58.63 |    58.45 |    58.69 |    59    |    58.97 | 0.60 ( + )      |
| 31          | 20              | 1000           | 2.33              |    66.18 |    65.95 |    66.05 |    65.58 |    65.52 |    65.07 | 1.68 ( - )      |
| 1049        | 37              | 1458           | 7.19              |    72    |    75.62 |    79.48 |    80.31 |    81.49 |    81.4  | 13.06 ( + )     |
| 1067        | 21              | 2109           | 5.47              |    73.12 |    76.59 |    77.94 |    77.92 |    78.57 |    78.6  | 7.50 ( + )      |
| 1464        | 4               | 748            | 3.20              |    60.46 |    63.96 |    70.2  |    71.29 |    72.26 |    72.18 | 19.38 ( + )     |
| 1487        | 72              | 2534           | 14.84             |    82.54 |    87.67 |    88.57 |    88.27 |    89.85 |    89.56 | 8.51 ( + )      |
| 1489        | 5               | 5404           | 2.41              |    66.4  |    67.62 |    68.3  |    68.14 |    68.21 |    68.18 | 2.69 ( + )      |
| 1494        | 41              | 1055           | 1.96              |    62.24 |    63.05 |    64.62 |    65.94 |    66.07 |    66.05 | 6.12 ( + )      |
| 40701       | 20              | 5000           | 6.07              |    66.45 |    70.65 |    75.99 |    78.18 |    78    |    77.7  | 16.93 ( + )     |
| 40900       | 36              | 5100           | 67                |    98.53 |    98.41 |    98.39 |    98.39 |    98.27 |    98.26 | 0.28 ( - )      |
| 40981       | 14              | 690            | 1.25              |    73.56 |    74.41 |    74.67 |    74.99 |    74.93 |    74.94 | 1.88 ( + )      |
| 40983       | 5               | 4839           | 17.54             |    79.71 |    85.04 |    89.36 |    92.94 |    92.9  |    92.75 | 16.35 ( + )     |
| 41143       | 144             | 2984           | 1                 |    64.64 |    64.8  |    65.06 |    65.17 |    65.29 |    65.13 | 0.76 ( + )      |
| 41144       | 259             | 3140           | 1.01              |    50.7  |    50.63 |    50.68 |    50.67 |    50.71 |    50.77 | 0.14 ( + )      |
| 41145       | 308             | 5832           | 1                 |    56.16 |    56.28 |    56.21 |    56.24 |    56.19 |    56.22 | 0.12 ( + )      |
| 41146       | 20              | 5124           | 1                 |    71.26 |    73.9  |    75.39 |    75.84 |    76.02 |    77.07 | 8.51 ( + )      |
| 41156       | 48              | 4147           | 3.03              |    67.74 |    69.78 |    70.64 |    71.82 |    71.72 |    71.74 | 5.90 ( + )      |
| Average     |                 |                |                   |    68.84 |    70.76 |    72.35 |    72.96 |    73.24 |    73.21 | 6.35 ( + )      |

the table reports average performance across datasets, revealing that the majority of performance gains occur in the first two iterations. This observation aligns with our synthetic experiments using multi-layer models (Figure 1), where the improvement from 1-layer to 2-layer is substantially greater than the improvement from 2-layer to 5-layer. These findings highlight that explicitly looping the tabular foundation model to iteratively refine soft pseudo-labels of unlabeled data using only a few iterations can substantially enhance performance.

As shown and discussed, our LoopTabFM algorithm enhances model performance. However, this improvement is not consistent across all datasets. For example, performance drops on the OpenML datasets with IDs 31 and 40900. This may be attributed to factors such as noise levels in the raw data, class imbalance, or other dataset-specific characteristics. In contrast to our synthetic experimental setting, where the model is pretrained in a meta-learning fashion on the distribution of the given dataset, TabPFN is used as a general-purpose pretrained foundation model and applied directly to target datasets in a single-shot inference setting. Prior work (Ye et al., 2025) has also shown that TabPFN can be sensitive to input length, which may further a ff ect performance consistency. Despite these limitations, our experiments with TabPFN o ff er an initial insight into how unlabeled data and iterative looping can be leveraged to improve predictive performance. These findings suggest promising future directions, such as designing data-aware looping algorithms that adapt to dataset-specific properties.

## 6 Discussion and Limitations

Our paper introduces a theoretical study of semisupervised in-context learning and characterizes how transformer, specifically linear attention, models can harness unlabeled data in their context window to make inference. We show that depth is crucial to go beyond supervised estimation and utilize unlabeled data, and the latter is achieved by constructing estimators of the form ˆ µ = P K i = 0 ai ( X ⊤ X ) i X ⊤ y . log K depth su ffi ces to express a K th order polynomial which is in line with our synthetic and real experiments that corroborate that mild amount of depth / looping already achieves most of the benefit. Our core theoretical results are limited to linear attention models and it is important to understand the capabilities of the full transformer architecture. Indeed, transformer (MLP + softmax) empirically outperforms a linear attention model with equal number of layers, well approximating the Bayes optimal semisupervised estimator. It would also be exciting to go beyond the classification setting and examine how self-generated CoT rationales, as in (Wu et al., 2023), can enhance ICL capabilities for tasks that require reasoning / autoregression. Additionally, our proposed LoopTabFM algorithm demonstrates that iteratively pseudo-labeling unlabeled data can indeed enhance predictive performance for tabular tasks. However, there remains significant potential for developing more intelligent, data-specific algorithms that more e ff ectively leverage unlabeled data to further improve model performance.

## Acknowledgements

This work was supported in part by the National Science Foundation grants CCF-2046816, CCF2403075, CCF-2008020, the O ffi ce of Naval Research grant N000142412289, and by gifts / awards from Open Philanthropy, Amazon Research, and Google Research.

## References

- Rishabh Agarwal, Avi Singh, Lei M Zhang, Bernd Bohnet, Stephanie Chan, Ankesh Anand, Zaheer Abbas, Azade Nova, John D Co-Reyes, Eric Chu, et al. Many-shot in-context learning. arXiv preprint arXiv:2404.11018 , 2024.
- Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra. Transformers learn to implement preconditioned gradient descent for in-context learning. Advances in Neural Information Processing Systems , 36, 2023.
- Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models. In The Eleventh International Conference on Learning Representations , 2023. URL https://openreview.net/forum?id= 0g0X4H8yN4I .
- Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, and Song Mei. Transformers as statisticians: Provable in-context learning with in-context algorithm selection. Advances in neural information processing systems , 36:57125-57211, 2023.
- Sivaraman Balakrishnan, Martin J Wainwright, and Bin Yu. Statistical guarantees for the em algorithm: From population to sample-based analysis. The Annals of Statistics , 45(1):77-120, 2017.
- Peter L Bartlett, Michael I Jordan, and Jon D McAuli ff e. Convexity, classification, and risk bounds. Journal of the American Statistical Association , 101(473):138-156, 2006.
- Mikhail Belkin, Daniel J Hsu, and Partha Mitra. Overfitting or perfect fitting? risk bounds for classification and regression rules that interpolate. Advances in Neural Information Processing Systems , 31, 2018.
- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems , 33:1877-1901, 2020.
- Yuan Cao, Quanquan Gu, and Mikhail Belkin. Risk bounds for over-parameterized maximum margin classification on sub-gaussian mixtures. Advances in Neural Information Processing Systems , 34: 8407-8418, 2021.
- Niladri S Chatterji and Philip M Long. Finite-sample analysis of interpolating linear classifiers in the overparameterized regime. Journal of Machine Learning Research , 22(129):1-30, 2021.
- Zihan Chen, Song Wang, Zhen Tan, Jundong Li, and Cong Shen. Maple: Many-shot adaptive pseudo-labeling for in-context learning. arXiv preprint arXiv:2505.16225 , 2025.
- Liam Collins, Advait Parulekar, Aryan Mokhtari, Sujay Sanghavi, and Sanjay Shakkottai. In-context learning with transformers: Softmax attention adapts to function lipschitzness. arXiv preprint arXiv:2402.11639 , 2024.
- Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, and Furu Wei. Why can GPT learn in-context? language models secretly perform gradient descent as meta-optimizers. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Findings of the Association for Computational Linguistics: ACL 2023 , pp. 4005-4019, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653 / v1 / 2023.findings-acl.247. URL https://aclanthology.org/2023.findings-acl.247 .
- Zeyu Deng, Abla Kammoun, and Christos Thrampoulidis. A model of double descent for highdimensional binary linear classification. Information and Inference: A Journal of the IMA , 11(2): 435-495, 2022.

- Luc Devroye, László Györfi, and Gábor Lugosi. A probabilistic theory of pattern recognition , volume 31. Springer Science &amp; Business Media, 2013.
- Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes. Advances in Neural Information Processing Systems , 35:30583-30598, 2022.
- Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 , 2025.
- Trevor Hastie, Robert Tibshirani, and Jerome H Friedman. The elements of statistical learning: data mining, inference, and prediction , volume 2. Springer, 2009.
- Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. Tabpfn: A transformer that solves small tabular classification problems in a second. arXiv preprint arXiv:2207.01848 , 2022.
- Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nature , 637(8045):319-326, 2025.
- Balaji Krishnapuram, David Williams, Ya Xue, Lawrence Carin, Mário Figueiredo, and Alexander Hartemink. On semi-supervised classification. Advances in neural information processing systems , 17, 2004.
- Jeongyeol Kwon and Constantine Caramanis. The em algorithm gives sample-optimality for learning mixtures of well-separated gaussians. In Conference on Learning Theory , pp. 2425-2487. PMLR, 2020.
- Beatrice Laurent and Pascal Massart. Adaptive estimation of a quadratic functional by model selection. Annals of statistics , pp. 1302-1338, 2000.
- Marc Lelarge and Léo Miolane. Asymptotic bayes risk for gaussian mixture in a semi-supervised setting. In 2019 IEEE 8th International Workshop on Computational Advances in Multi-Sensor Adaptive Processing (CAMSAP) , pp. 639-643. IEEE, 2019.
- Yingcong Li, Muhammed Emrullah Ildiz, Dimitris Papailiopoulos, and Samet Oymak. Transformers as algorithms: Generalization and stability in in-context learning. In International Conference on Machine Learning , pp. 19565-19594. PMLR, 2023.
- Yingcong Li, Ankit S Rawat, and Samet Oymak. Fine-grained analysis of in-context linear estimation: Data, architecture, and beyond. Advances in Neural Information Processing Systems , 37:138324138364, 2024.
- Yingcong Li, Davoud Ataee Tarzanagh, Ankit Singh Rawat, Maryam Fazel, and Samet Oymak. Gating is weighting: Understanding gated linear attention through in-context learning. arXiv preprint arXiv:2504.04308 , 2025.
- Renpu Liu and Jing Yang. Unlabeled data can provably enhance in-context learning of transformers. arXiv preprint arXiv:2601.10058 , 2026.
- Junwei Ma, Valentin Thomas, Rasa Hosseinzadeh, Alex Labach, Jesse C Cresswell, Keyvan Golestan, Guangwei Yu, Anthony L Caterini, and Maksims Volkovs. Tabdpt: Scaling tabular foundation models on real data. In The Thirty-ninth Annual Conference on Neural Information Processing Systems , 2025.
- Arvind V. Mahankali, Tatsunori Hashimoto, and Tengyu Ma. One step of gradient descent is provably the optimal in-context learner with one layer of linear self-attention. In The Twelfth International Conference on Learning Representations , 2024. URL https://openreview.net/forum?id= 8p3fu56lKc .
- Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? arXiv preprint arXiv:2202.12837 , 2022.

- Andrea Montanari, Feng Ruan, Youngtak Sohn, and Jun Yan. The generalization error of max-margin linear classifiers: High-dimensional asymptotics in the overparametrized regime. arXiv preprint arXiv:1911.01544 , 7, 2019.
- Ojash Neopane. Lecture notes on high-dimensional statistics. https://www.stat.cmu.edu/ ~arinaldo/Teaching/36709/S19/Scribed\_Lectures/Feb26\_Ojash.pdf , 2018.
- Kamal Nigam, Andrew Kachites McCallum, Sebastian Thrun, and Tom Mitchell. Text classification from labeled and unlabeled documents using em. Machine Learning , 39(2-3):103-134, 2000. doi: 10.1023 / A:1007692713085.
- Samet Oymak and Talha Cihad Gulcu. A theoretical characterization of semi-supervised learning with self-training for gaussian mixture models. In International Conference on Artificial Intelligence and Statistics , pp. 3601-3609. PMLR, 2021.
- Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. Tabicl: A tabular foundation model for in-context learning on large data. arXiv preprint arXiv:2502.05564 , 2025.
- Joel Ratsaby and Santosh S. Venkatesh. Learning from a mixture of labeled and unlabeled examples with parametric side information. In Proceedings of the Eighth Annual Conference on Computational Learning Theory (COLT '95) , pp. 412-417. ACM, 1995. doi: 10.1145 / 225298.225348.
- Wei Shen, Ruida Zhou, Jing Yang, and Cong Shen. On the training convergence of transformers for in-context classification. arXiv preprint arXiv:2410.11778 , 2024.
- Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more e ff ective than scaling model parameters. arXiv preprint arXiv:2408.03314 , 2024.
- Christos Thrampoulidis, Samet Oymak, and Mahdi Soltanolkotabi. Theoretical insights into multiclass classification: A high-dimensional asymptotic view. Advances in Neural Information Processing Systems , 33:8907-8920, 2020.
- Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In International Conference on Machine Learning , pp. 35151-35174. PMLR, 2023.
- Ke Wang and Christos Thrampoulidis. Binary classification of gaussian mixtures: Abundance of support vectors, benign overfitting, and regularization. SIAM Journal on Mathematics of Data Science , 4(1):260-284, 2022.
- Yifei Wang, Yuyang Wu, Zeming Wei, Stefanie Jegelka, and Yisen Wang. A theoretical understanding of self-correction through in-context alignment. In Advances in Neural Information Processing Systems , volume 37, pp. 89869-89912, 2024.
- Colin Wei, Kendrick Shen, Yining Chen, and Tengyu Ma. Theoretical analysis of self-training with deep networks on unlabeled data. arXiv preprint arXiv:2010.03622 , 2020.
- Jingfeng Wu, Difan Zou, Zixiang Chen, Vladimir Braverman, Quanquan Gu, and Peter L Bartlett. How many pretraining tasks are needed for in-context learning of linear regression? arXiv preprint arXiv:2310.08391 , 2023.
- Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. In International Conference on Learning Representations , 2022. URL https://openreview.net/forum?id=RdJVFCHjUMI .
- Moucheng Xu, Yukun Zhou, Chen Jin, Marius de Groot, Daniel C. Alexander, Neil P. Oxtoby, Yipeng Hu, and Joseph Jacob. Expectation maximisation pseudo labels. Medical Image Analysis , 94:103125, 2024. ISSN 1361-8415. doi: https: // doi.org / 10.1016 / j.media.2024.103125. URL https://www.sciencedirect.com/science/article/pii/S1361841524000501 .
- Tong Yang, Yu Huang, Yingbin Liang, and Yuejie Chi. In-context learning with representations: Contextual generalization of trained transformers. arXiv preprint arXiv:2408.10147 , 2024.
- Han-Jia Ye, Si-Yang Liu, and Wei-Lun Chao. A closer look at tabpfn v2: Strength, limitation, and extension. arXiv preprint arXiv:2502.17361 , 2025.

- Ruiqi Zhang, Spencer Frei, and Peter L Bartlett. Trained transformers learn linear models in-context. Journal of Machine Learning Research , 25(49):1-55, 2024.
- Yucheng Zhou, Xiang Li, Qianning Wang, and Jianbing Shen. Visual in-context learning for large vision-language models. In Findings of the Association for Computational Linguistics ACL 2024 , pp. 15890-15902, 2024.

## Appendix

## Table of Contents

| A Additional Experiments and Algorithmic Details   | A Additional Experiments and Algorithmic Details   | A Additional Experiments and Algorithmic Details   |   15 |
|----------------------------------------------------|----------------------------------------------------|----------------------------------------------------|------|
|                                                    | A.1                                                | Additional Observations . . . . . . . . .          |   15 |
|                                                    | A.2                                                | Algorithmic Details of Tabular Experiments         |   17 |
| B                                                  | Analysis of Single-layer Linear Attention          | Analysis of Single-layer Linear Attention          |   17 |
|                                                    | B.1                                                | Supporting Lemmas . . . . . . . . . . . .          |   17 |
|                                                    | B.2                                                | Proof of Theorem 1 . . . . . . . . . . . .         |   20 |
| C                                                  | Analysis of Multi-layer Linear Attention           | Analysis of Multi-layer Linear Attention           |   22 |
|                                                    | C.1                                                | Proof of Proposition 1 . . . . . . . . . . .       |   22 |
|                                                    | C.2                                                | Proof of Proposition 2 . . . . . . . . . . .       |   24 |
|                                                    | C.3                                                | Proof of Lemma 1 . . . . . . . . . . . . .         |   24 |
|                                                    | C.4                                                | Proof of Theorem 2 . . . . . . . . . . . .         |   25 |
|                                                    | C.5                                                | Proof of Theorem 4 . . . . . . . . . . . .         |   27 |

Figure 2: Additional experimental results. (a)&amp;(b): Analysis of the optimal α values for the SSPI estimator (cf. (SSPIk )) under varying ( n , p , k ). Green solid and dotted curves represent optimal α values for SSPI-1 and SSPI-∞ , respectively. The SSPI results shown in Figure 1 use the corresponding α values from Figs. 2a and 2b. (c): Comparison of di ff erent model architectures for the SS-ICL problem. Dark blue and orange curves show results for 1-layer and 5-layer attention models, with solid and dashed curves representing linear and softmax attention, respectively. Cyan curves correspond to 5-layer Transformers. The black dotted curve shows the asymptotic Bayes-optimal error (cf. Lelarge &amp; Miolane (2019)). Results suggest the performance ordering: Transformer &gt; linear attention &gt; softmax attention. Further details are provided in Section 5.

<!-- image -->

## A Additional Experiments and Algorithmic Details

## A.1 Additional Observations

Exploration of Optimal α Values In Section 4, we introduced the SSPIk estimator (cf. (SSPIk )), but did not discuss the choice of the mixing parameter α , which plays a crucial role in balancing the contribution of the supervised estimator ˆ µ s . Specifically, α controls how much weight is given to the purely supervised signal. In the fully supervised case, the optimal choice is α = 1, as ˆ µ s corresponds to the optimal estimator.

In Figures 2a and 2b, we empirically examine the optimal values of α . Given µ ∼ Unif( S d -1 ), we define the optimal α as the minimizer of the following cosine similarity-based objective:

<!-- formula-not-decoded -->

## Algorithm 1 LoopTabFM : Looping Tabular FM with Soft Pseudo-labels and Risk-aware Updates

```
Require: Dataset D lab , D unlab, looping iterations K 1: procedure Looping ( D lab , D unlab , K ) 2: FM 0 ← TabPFN-v2( D lab ) ▷ FM k corresponds to model of Loopk . 3: D unlab ← FM 0 ( D unlab ) ▷ Assign pseudo labels via ˆ y soft ← FM 0 ( x ∈ D unlab ). 4: FM best ← FM 0 5: R val = Val_Risk ( D unlab ) 6: for Looping iteration k = 1 , . . . , K do 7: FM k ← TabPFN-v2( D lab ∪ D unlab ) 8: D unlab ← FM k ( D unlab ) ▷ Update pseudo labels via ˆ y soft ← FM k ( x ∈ D unlab ). 9: if Val_Risk ( D unlab ) < R val then 10: FM best ← FM k 11: R val = Val_Risk ( D unlab ) 12: end if 13: end for 14: return FM best 15: end procedure 16: procedure Val_Risk ( D unlab ) 17: return 1 |D unlab | P i min( | ˆ y soft i -1 | , | ˆ y soft i + 1 | ) 18: ▷ ˆ y soft corresponds to the assigned soft label for feature in D unlab . 19: end procedure
```

For each setting, we optimize α using the Adam optimizer for 10,000 epochs with a batch size of 128 and a learning rate of 0.01. The results are shown in Figs 2a and 2b.

In Figure 2a, for both SSPI-1 and SSPI-∞ , the optimal α starts near zero when the number of labeled examples is small, reflecting the limited utility of ˆ µ s in low-supervision regimes. As the number of labeled samples increases, α grows approximately linearly and approaches 1 when the problem becomes fully supervised. In Figure 2b, when n = 10 and p = 1 (i.e., all examples are labeled), the optimal α begins at 1. As n increases and the fraction of unlabeled data grows, α decreases significantly. This trend indicates that as the volume of unlabeled data increases, the SSPI estimator adaptively reduces reliance on the supervised component ˆ µ s and increases reliance on the semi-supervised component, which leverages the structure of the unlabeled data through X ⊤ X .

Comparison Across Di ff erent Model Architectures Beyond linear attention, we investigate additional model architectures under our SS-ICL setting. The comparison results are presented in Fig. 2c. The softmax attention model uses the same structure described in Section 2.2, with the only di ff erence being the addition of a softmax operation in Eq. (4). The Transformer model introduces further nonlinearity and capacity by incorporating multi-layer perceptrons (MLPs) and layer normalization. The Transformer experiments are conducted with 5-layer models.

When comparing weaker models-such as 1-layer linear (dark blue solid) and softmax (dark blue dashed) attention-we observe that softmax attention consistently underperforms linear attention. Notably, softmax attention fails to match the performance of the optimal supervised estimator, even when all labels are observed (i.e., when the number of labeled samples equals n = 50). Furthermore, increasing the depth of softmax attention (orange dashed curve for 5-layer softmax) still does not surpass the performance of 5-layer linear attention (orange solid curve). Among all architectures, the Transformer achieves the best performance due to its increased model capacity and expressiveness. Compared with Fig. 1a, where the orange and dark blue markers (linear attention) are identical, the Transformer significantly improves accuracy. This improvement highlights that SSPI, while e ff ective, is not the optimal semi-supervised estimator. Although our semi-supervised setting assumes isotropic data, the characterization of its optimal algorithm remains an open and foundational problem for future exploration. In the figure, we also include the asymptotic Bayes-optimal curve (black dotted; derived from Lelarge &amp; Miolane (2019)) . As the number of samples increases, the results from linear attention, softmax attention, and Transformer all converge toward this optimal curve. We attribute the initial performance gap, particularly at low values along x -axis (e.g., np = 1), to the scarcity of labeled data.

## A.2 Algorithmic Details of Tabular Experiments

In this section, we provide additional details regarding the tabular experiments discussed in Section 5.1. We propose the LoopTabFM algorithm with its details outlined in Algorithm 1. Suppose that we are given labeled D lab and unlabeled D unlab datasets. The overall workflow of the algorithm proceeds as follows:

1. Base Model: Perform ICL using TabPFN on the labeled dataset D lab and treat the resulting model as the base model (Loop-0). The corresponding test accuracies are reported in Table 1.
2. Pseudo-Label Assignment: Using the current model (e.g., Loopk ) to generate predictions for the unlabeled data D unlab. Assign soft pseudo-labels based on these predictions. Note that the model outputs are scalars (i.e., elements of R ) and can be interpreted as soft labels.
3. Model Update: Construct a new prompt by combining the labeled examples with their true labels and the unlabeled examples with their assigned soft pseudo-labels. Perform ICL using TabPFN on this combined prompt to obtain an updated model (Loop-( k + 1)). Repeat this process from Step 2 until the maximum number of looping iterations is reached.
4. ⋆ Model Validation: To improve the stability of the looping process, we introduce an additional validation step and retain the model with the lowest validation risk as the final (best) model. Specifically, after assigning soft pseudo-labels to the unlabeled data, i.e., D unlab = { ( x i , ˆ y soft i ) n i = 1 } , we compute the validation risk over these pseudo-labeled examples as follows:

<!-- formula-not-decoded -->

which penalizes predictions that deviate from confident binary labels ± 1.

## B Analysis of Single-layer Linear Attention

## B.1 Supporting Lemmas

Recap the SPI estimator from (SPI). Given a semi-supervised dataset ( x i , yi ) n i = 1 as described in Section 2.1, let I denote the token indices set corresponding to the labeled demonstrations, that is, we have

<!-- formula-not-decoded -->

Then, the SPI estimates the task mean via

<!-- formula-not-decoded -->

Let W ∈ R d × d be the preconditioning matrix. We define the following objective:

<!-- formula-not-decoded -->

Here, we set ( x , y ) to be the query feature and its corresponding true label. The expectation subsumes the randomness in ( x i , yi ) , ( x , y ) as described in Section 2.1.

In the following, we provide a lemma that establishes equivalence between optimizing L att-1 ( W (1) , h ) (cf. (7) and choosing L = 1) and ˜ L ( W ).

Lemma 2 Consider ICL problem described in Section 2.2 with prompt defined in (3) . Consider training a single-layer linear attention with squared loss, that is, L = 1 and ℓ ( y , ˆ y ) = ( y -ˆ y ) 2 . Recall the objectives from (7) and (15) , and let L ⋆ att1 and ˜ L ⋆ : = ˜ L ( W ⋆ ) be their corresponding optimal losses where W ⋆ is defined in (15) . Then, we have

<!-- formula-not-decoded -->

Additionally, let f ⋆ att1 : R ( n + 1) × ( d + 1) → R denote the optimal prediction (associated with the optimal loss L ⋆ att1 ). We have that f ⋆ att1 is unique and for any prompt Z (cf. (3) )

<!-- formula-not-decoded -->

Proof. Recap the single-layer linear attention model and its prediction from (4) and (6). We have ⊤ ⊤ ⊤

<!-- formula-not-decoded -->

with W : = { W q , W k , W v } being the set of the query, key and value matrices of the attention. Since W and h are tunable parameters, without loss of generality and for simplicity, let

<!-- formula-not-decoded -->

Following the proof of Li et al., 2024, Proposition 1, similarly, we denote

<!-- formula-not-decoded -->

where ¯ W ∈ R d × d , w 1 , w 2 , h 1 ∈ R d , and w , h ∈ R .

Additionally, let I denote the token indices set corresponding to the labeled demonstrations (cf. (14)). Recall the prompt Z from (3), and X = [ x 1 · · · x n ] ⊤ ∈ R n × d and y = [ y 1 · · · yn ] ⊤ ∈ R n from (10). Then we get

<!-- formula-not-decoded -->

Combining (18) and (19) together, we can rewrite the one-layer linear prediction as

f

<!-- formula-not-decoded -->

⊤

⊤

¯

where ˜ W : = h ¯ W + w 1 h ⊤ 1 and we define m : = |I| .

Next, recall the loss from (7) and consider the squared loss function, ℓ ( y , ˆ y ) = ( y -ˆ y ) 2 . We have

<!-- formula-not-decoded -->

For simplicity and without loss of generality, we omit y and use x to represent y x . Note that the distribution of (updated) x is not conditioned on its class and given mean vector µ , it follows x ∼ N ( µ , σ 2 I ). Similarly, let x i represent y c i x i . We can then write

<!-- formula-not-decoded -->

We start with showing that for any given parameters W ∈ R ( d + 1) × ( d + 1) , h ∈ R d + 1 , the component E [( x ⊤ ˜ W P i ∈I x i -1)( x ⊤ ( ¯ WX ⊤ Xh 1 + mh w 1 ))] = 0. To prove it, we first expand

<!-- formula-not-decoded -->

In the following, we consider the expectations of ( a ) , ( b ) , ( c ) , ( d ) sequentially, all of which take the value zero. First note that since µ ∼ Unif( S d -1 ) and ( ξ i ) n i = 1 , ξ ∼ N (0 , σ 2 I ), the odd moments of µ , ξ and ξ i , i ∈ [ n ] are all zeros.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, loss in (20) returns

<!-- formula-not-decoded -->

Here, the first term E [( x ⊤ ˜ W P i ∈I x i -1) 2 ] = ˜ L ( ˜ W ) where ˜ L ( ˜ W ) is defined in (15). Recall that ˜ W = h ¯ W + w 1 h ⊤ 1 . Then for any ˜ W ∈ R d × d , setting h 1 = w 1 = 0 d and h = 1 returns E GLYPH&lt;20&gt; GLYPH&lt;16&gt; x ⊤ GLYPH&lt;16&gt; ¯ WX ⊤ Xh 1 + mh w 1 GLYPH&lt;17&gt;GLYPH&lt;17&gt; 2 GLYPH&lt;21&gt; = 0, and then

<!-- formula-not-decoded -->

Therefore, optimizing L att-1 ( W (1) , h ) returns the same minima as optimizing ˜ L ( W ), which completes the proof of (16). Note that optimal loss L ⋆ att-1 depends on the labeled data i ∈ I only.

Furthermore, since ˜ L ( W ) is strongly convex (see (21)), W ⋆ exists and is unique. Therefore, (16) and uniqueness of W ⋆ leads to the conclusion (17).

Lemma 3 Consider the objective defined in (15) with semi-supervised data following Section 2. Then the optimal solution W ⋆ satisfies

<!-- formula-not-decoded -->

for some c &gt; 0 .

Proof. Recap the Objective (15) and its optimal solution W ⋆ . Let I be the index set corresponding the labeled in-context examples, and |I| = m . Note that, m is also a random variable, independent of x i , y c i , x , y .

As in the proof of Lemma 2, we use x to represent y x and x i to represent y c i x i for simplicity, where (updated) x i , x ∼ N ( µ , σ 2 I ). Letting ξ ′ , ξ , ξ i ∼ N (0 , σ 2 I ) be independent, we obtain

<!-- formula-not-decoded -->

Di ff erentiating it results in

<!-- formula-not-decoded -->

Setting ∇ W ˜ L ( W ) = 0, we obtain the optimal W ⋆

<!-- formula-not-decoded -->

which leads to the conclusion that W ⋆ = c I , for c = 1 (1 + σ 2 ) E [ m 2 ] / E [ m ] + σ 2 + σ 4 d &gt; 0. It completes the proof.

## B.2 Proof of Theorem 1

Proof. Note that (8) can be easily proven using Lemmas 2 and 3. Then, we focus on proving (9). Given that (8) holds, we can rewrite its classification error as

<!-- formula-not-decoded -->

where ˆ µ s = 1 |I| P i ∈I yi x i defined in (SPI) and I is the index set of labeled samples. Let m = |I| . Recall from Section 2.1 where x ∼ N ( y · µ , σ 2 I ). We can rewrite

<!-- formula-not-decoded -->

Then for any given µ , ˆ µ s , we get

<!-- formula-not-decoded -->

Here Q -function is the tail distribution function of the standard normal distribution. Next, similarly, given that x i ∼ N ( yi · µ , σ 2 I ) for i ∈ I , we can rewrite

<!-- formula-not-decoded -->

Then combining (22) and (23), we have

<!-- formula-not-decoded -->

Note that for any µ with ∥ µ ∥ ℓ 2 = 1, we have µ ⊤ g 2 ∼ N (0 , 1). Therefore, we can write

<!-- formula-not-decoded -->

and let U ∈ R d × d be a unitary matrix with first row being µ . We can write

<!-- formula-not-decoded -->

Here, X 2 d -1 denotes chi-squared distribution with ( d -1) degrees of freedom. Then, we get

<!-- formula-not-decoded -->

where εσ : = σ/ √ m . It completes the proof of (9).

Next, we derive an upper bound for P ( y ⋆ att-1 ( Z ) , y ). Let c : = ε -1 σ . Then we have

<!-- formula-not-decoded -->

where the inequality comes from the fact that P ( g ≤ -c / 2) = Q ( c / 2) and Q ( x ) ≤ 1 for any x ∈ R . Next, we have

<!-- formula-not-decoded -->

Here the first inequality comes from that 1 √ 1 + x ≥ 1 -1 2 x and the second utilizes that g ≥ -c 2 .

Since h ∼ X 2 d -1 , from the Laurent-Massart inequality (Laurent &amp; Massart, 2000), we have that

<!-- formula-not-decoded -->

Therefore, we have that with probability at least 1 -e -t 1

<!-- formula-not-decoded -->

Setting t 1 = d , we get with probability at least 1 -e -d

<!-- formula-not-decoded -->

Combining the result with (24), since Q ( x ) ≤ 1 for x ∈ R and Q ( x ) ≤ e -x 2 / 2 for x &gt; 1, we get that

<!-- formula-not-decoded -->

It completes the proof.

## C Analysis of Multi-layer Linear Attention

## C.1 Proof of Proposition 1

Proof. We consider the following model constructions for the attention matrices in the ℓ th layer, ℓ ∈ [ L ] and the final linear prediction head:

<!-- formula-not-decoded -->

Suppose the input to ℓ th layer is

<!-- formula-not-decoded -->

Recapping the model construction from (25), the ℓ th layer output returns

<!-- formula-not-decoded -->

Therefore, following (5), the input of ( ℓ + 1)th layer is

<!-- formula-not-decoded -->

- Label propagation: We first focus on deriving label propagation results. Suppose that we have a ℓ = 0 for ℓ ∈ [ L ] .

Then following (26), the output of ℓ 'th layer takes the following form:

<!-- formula-not-decoded -->

Here, the first d coordinates of each token's output are zeros, and therefore, the corresponding input coordinates remain unchanged, and we have

<!-- formula-not-decoded -->

The prediction (based on the last token output and after applying prediction head) is given by

<!-- formula-not-decoded -->

We next focus on obtaining y L . From (27), we have

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

Combining with (28) results in

<!-- formula-not-decoded -->

It completes the proof.

- Feature propagation: We now focus on the feature propagation setting. In contrast to the label propagation, let us assume that

<!-- formula-not-decoded -->

The prediction (following (26), based on the last token output and after applying prediction head) is given by

<!-- formula-not-decoded -->

We first obtain y L . From (27) (since b ℓ → 0), we have

<!-- formula-not-decoded -->

Therefore,

## C.3 Proof of Lemma 1

Proof. In the proof of Proposition 1, we showed how to derive the label and feature propagation results by restricting the construction to either a ℓ ≡ 0 (for label propagation) or ( a ℓ →∞ , b ℓ → 0) (for feature propagation). Here, we consider a propagation process without imposing restrictions on the choices of ( a ℓ , b ℓ ), and study the form of the final prediction returned by the model.

To avoid the notation conflict, we express the matrix A in (12) as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, we focus on X L , x L . From (27), as a ℓ →∞ , we have

<!-- formula-not-decoded -->

Therefore, and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining all together with (29), we have that

<!-- formula-not-decoded -->

It completes the proof.

## C.2 Proof of Proposition 2

Proof. The proof follows directly by adopting the same model construction and proof strategy as in Proposition 1, under the additional assumption that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Recall the same model construction used in the proof of Proposition 1, defined in (25). From (26), we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

ℓ

<!-- formula-not-decoded -->

At each layer, the operations performed are linear combinations and multiplications involving X ⊤ ℓ X ℓ and identity matrices scaled by the parameters ( a ℓ , b ℓ ). Thus, each coe ffi cient ek of ( X ⊤ X ) k depends smoothly on the scalar parameters ( a ℓ , b ℓ ).

From (26) and (27), we have that

<!-- formula-not-decoded -->

That is, in the final f attL ( Z ) expression, the coe ffi cients corresponding to di ff erent degrees of ( X ⊤ X ) k depend on the model parameters cbL and ( a ℓ , b ℓ ) L -1 ℓ = 1 , which together have at most 2 L -1 degrees of freedom. Let c = [ cbL a 1 · · · aL -1 b 1 · · · bL -1 ] ⊤ . This means there exists a smooth function g : R 2 L -1 → R K such that: e = g ( c ).

It remains to show that an L -layer linear attention model can produce terms involving powers of X ⊤ X up to degree (3 L -3) / 2.

Let f ( Z ) be a function that contains terms of the form x ⊤ ( X ⊤ X ) k X ⊤ y for various powers k . Define P ( f ( Z )) as the projection that extracts the highest degree k present in f ( Z ). For example, P GLYPH&lt;0&gt; x ⊤ ( I + ( X ⊤ X ) 2 ) X ⊤ y GLYPH&lt;1&gt; = 2. Then from (30), we have

<!-- formula-not-decoded -->

It completes the proof.

## C.4 Proof of Theorem 2

Proof. Let ξ ∼ N (0 , I ) and rewrite y x = µ + σ ξ . For any matrix A ∈ R d × d , the prediction error of ˆ y A = sgn( x ⊤ A ˆ µ s ) given ˆ µ s returns

<!-- formula-not-decoded -->

where following (27), we have

For any A ∈ R d × d , we can decompose it as

<!-- formula-not-decoded -->

where u 1 = µ , ∥ u i ∥ ℓ 2 = 1 and u ⊤ i u j = 0 for any i , j . Let λ 1 &gt; 0. Then, we get

<!-- formula-not-decoded -->

Now consider ∥ A ˆ µ s ∥ ℓ 2 where we have

<!-- formula-not-decoded -->

Since u i , i , 1 is orthogonal to µ , λ 1 µ v ⊤ 1 ˆ µ s is orthogonal to P d i = 2 λ i u i v ⊤ i ˆ µ s . Therefore, given ∥ u i ∥ ℓ 2 = 1 for all i ∈ [ d ], it obeys

<!-- formula-not-decoded -->

For simplicity, define

<!-- formula-not-decoded -->

where ∆ ( · ) is a function of λ 1 and ( λ i , v i )'s for i ≥ 2, and we have

<!-- formula-not-decoded -->

Recall that ˆ µ s is the SPI estimator (cf. (SPI)). Let |I| = m . We can write ˆ µ s = µ + ξ ′ / √ m where ξ ′ ∼ N (0 , σ 2 I ).

Using (31), (32) and (33), the classification error becomes

<!-- formula-not-decoded -->

First, note that for any x &gt; 0, Q ( x ) &lt; 0 . 5 &lt; Q ( -x ). Therefore, the optimal choice of v 1 ∈ R d that minimizes P (ˆ y A , y ) is contained within the set of v 1 values that maximize P ( v ⊤ 1 ˆ µ s &gt; 0). Let v ⋆ 1 : = arg max v 1 ∈ R d P ( v ⊤ 1 ˆ µ s &gt; 0). Given that ˆ µ s ∼ N ( µ , σ 2 / m I ), we have that v ⋆ 1 = c µ for c &gt; 0. Let c = 1 and therefore, v ⋆ 1 = µ without loss of generality (since λ 1 can be any positive scalar). Then we obtain

<!-- formula-not-decoded -->

Let f ( ˆ µ s ) be the probability density function of ˆ µ s . Since ˆ µ s ∼ N ( µ , σ 2 / m I ), then it satisfies f ( ˆ µ s ) ≥ f ( -ˆ µ s ) for any µ ⊤ ˆ µ s &gt; 0 . (34)

Therefore, the classification error becomes

<!-- formula-not-decoded -->

Following (34), to minimize the error, we need minimize Q µ ⊤ ˆ µ s σ √ ( µ ⊤ ˆ µ s ) 2 +∆ ( ˆ µ s ) ! for µ ⊤ ˆ µ s &gt; 0, which can be easily done by choosing λ i = 0 for i ≥ 2. Then we get ∆ ( ˆ µ s ) ≡ 0. Therefore, the optimal solution set A ⋆ defined in Theorem 2 satisfies:

<!-- formula-not-decoded -->

Combining all together, we obtain

<!-- formula-not-decoded -->

It completes the proof.

## C.5 Proof of Theorem 4

Proof. Recap from Proposition 1. For any L -layer attention model with L ≥ 2, it can output

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let with f attL ( Z ) defined in (35). Then we have

<!-- formula-not-decoded -->

Therefore, in the following, we focus on upper-bounding the classification error P (ˆ y , y ) corresponding to (35). Given that the optimal prediction under the form sgn( x ⊤ A ˆ µ s ) is given by ˆ y µµ ⊤ : = sgn( x ⊤ µµ ⊤ ˆ µ s ) (cf. Theorem 2), with its corresponding error presented in (13). To analyze the performance of ˆ y , we study its di ff erence from the prediction ˆ y µµ ⊤ .

To begin with, let g i = ξ i /σ ∼ N (0 , I ) and g = P n i = 1 ξ i /σ √ n ∼ N (0 , I ). For simplicity, let A : = X ⊤ X / n -σ 2 I . We get

<!-- formula-not-decoded -->

Recall (31) from the proof of Theorem 2. Our goal is to bound

<!-- formula-not-decoded -->

Define

<!-- formula-not-decoded -->

From the Laurent-Massart inequality (Laurent &amp; Massart, 2000), we have that with probability at least 1 -e -t 1 (assuming t 1 ≥ d ), the first term of (36) can be bounded by

<!-- formula-not-decoded -->

Additionally, from Neopane (2018), we have that with probability at least 1 -e -t 2 (assuming t 2 ≥ d ), the second term of ∆ (cf. (36)) is bounded by (with a universal constant C &gt; 0)

<!-- formula-not-decoded -->

Combining (37) and (38), we get with probability at least 1 -2 e -t (for t ≥ d )

<!-- formula-not-decoded -->

We also bound ∥ ˆ µ s ∥ as follows. Let ˆ µ s = µ + σ/ √ m g ′ ∼ N ( µ , σ 2 m I ), similar to (37), with probability at least 1 -e -t 3 (assuming 2 d ≤ t 3 ≤ m / 4 σ 2 ), we can bound

<!-- formula-not-decoded -->

Then consider a significantly large n (to ensure that ∥ ∆ ∥ ≤ 1 / 12, e.g., n ≥ (12 C 1 ) 2 t ). With probability at least 1 -3 e -min( t , t 3) and suppose that µ ⊤ ˆ µ s &gt; 0 . 5, we can bound

<!-- formula-not-decoded -->

Now, we are ready to bound the classification error, where we get

<!-- formula-not-decoded -->

Choosing t = t 3 = 2 d , since m / 4 σ 2 ≥ 2 d , we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It completes the proof.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction accurately reflect the paper's contributions and scope.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and / or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Our assumptions underlying the algorithm and their necessity are discussed. Also, the limitation is discussed in the supplementary material.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational e ffi ciency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

## Answer: [Yes]

Justification: Our assumptions underlying the analysis and algorithm are discussed. Detailed proofs can be found in the supplementary material.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it a ff ects the main claims and / or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: All the information needed to reproduce the main experimental results of the paper are provided, either in the main paper or in the supplementary material.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and / or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might su ffi ce, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with su ffi cient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We have attached the code for implementing the algorithm and reproducing the experiments in the supplementary material.

## Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting / details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: All the information needed to reproduce the main experimental results of the paper are provided, either in the main paper or in the supplementary material.

## Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Detailed experiment results with errors is included in the supplementary.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train / test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide su ffi cient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: Details can be found in the supplementary material.

## Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have reviewed and confirmed that the research conducted in the paper conform with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: There is no societal impact of the work performed.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the e ffi ciency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: This paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing e ff ective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith e ff ort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We cited the original paper that produced the code package or dataset.

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
- Researchers should communicate the details of the dataset / code / model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
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

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval / review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: This research does not involve LLMs as any important components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.