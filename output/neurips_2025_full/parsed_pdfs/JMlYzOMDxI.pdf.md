## Contrastive Learning with Data Misalignment: Feature Purity, Training Dynamics and Theoretical Generalization Guarantees

## Jiawei Sun

Rensselaer Polytechnic Institute sunj11@rpi.edu

## Hongkang Li

University of Pennsylvania lihk@seas.upenn.edu

## Shuai Zhang

New Jersey Institute of Technology sz457@njit.edu

## Meng Wang

Rensselaer Polytechnic Institute wangm7@rpi.edu

## Abstract

Contrastive learning is a powerful framework for learning discriminative representations from image-text pairs. Despite its success, its theoretical foundations, especially when the image-text pair exhibits misalignment, remain underexplored. This paper provides the first theoretical analysis of contrastive learning under data misalignment, proving how the ground-truth modality-paired features are amplified while spurious features are suppressed through the training dynamics analysis. Specifically, we study two nonlinear encoders trained jointly with a contrastive loss and demonstrate that noisy (or misaligned) data pairs result in mixed representations and degrade the model's generalization ability. In contrast, recaptioning and filtering improve the data alignment, which in turn purifies the features learned by neurons and subsequently enhances generalization. Our analysis identifies feature purity as a key factor in the success of contrastive learning and offers insights into how data quality and training procedures impact representation learning and downstream generalization. Theoretical insights are supported by experiments on standard benchmarks.

## 1 Introduction

Vision-language models (VLMs) have achieved strong performance across diverse multimodal tasks such as vision-language understanding and generation. State-of-the-art methods like CLIP [37] and SimVLM [50] use contrastive learning to train dual encoders on large-scale image-text pairs scraped from the web, aligning embeddings by pulling paired samples closer in a shared space. These models excel in zero-shot scenarios, requiring no task-specific fine-tuning.

However, web-sourced captions are often noisy or misaligned, containing irrelevant or spurious details that hinder cross-modal alignment and reduce representation quality. For example, [34] cites an image of a blue Mercedes-Benz in a parking lot paired with the caption: "2003 Mercedes-Benz C240 sedan, Leather, MUST BE SEEN - $6199." The price information in this caption is only superficially correlated with the image and does not contribute meaningfully to understanding the image context. To mitigate this issue, many works [13, 34, 46, 3, 38, 16, 45] adopt text generation methods during VLM training to produce high-quality synthetic captions more faithful to the corresponding images. Models like LaCLIP [13] and BLIP [25] show that such recaptioning improves both the quality and diversity of training data, leading to significantly better performance. Further, [34] demonstrates that the cosine similarities between BLIP2 generated captions [24] and their paired images is higher than

Table 1: Comparison with existing theoretical works on contrastive learning.

| Work                                                                                                                                                   | Train Dyn.   | Nonlinear   | Zero-shot Gen.   | Recaption   | Multi- modal   | Joint Encoder   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|-------------|------------------|-------------|----------------|-----------------|
| (Wen &Li, 2021) [51] (Nakada et al., 2023) [33] (Chen et al., 2024) [10] (Lee et al., 2021) [21] (Zhang et al., 2023a) [60] (Pareek et al., 2025) [35] | ✓ ✓ ✓        | ✓ ✓         | ✓                |             | ✓ ✓ ✓ ✓ ✓      | ✓               |
| This paper                                                                                                                                             | ✓            |             |                  | ✓           |                | ✓               |
|                                                                                                                                                        | ✓            | ✓           | ✓                | ✓           | ✓              | ✓               |

that of raw captions. [22] analyzes conformity on MSCOCO and finds that it correlates with how common or rare an image-caption embedding is, reflecting its degree of alignment within the dataset.

Despite the impressive success of VLMs and the practical advancements driven by recaptioned texts, their theoretical foundations remain relatively underdeveloped. Several critical questions remain mostly open:

How do contrastively pre-trained VLMs align modalities, extract feature representations, and achieve zero-shot capabilities? How does text recaptioning on noisy image-text pairs provably enhance generalization performance?

Notably, even the theory of vanilla multimodal contrastive learning is still incomplete. For instance, [15, 60] extend spectral contrastive loss to multimodal settings, showing that the objective can be related to matrix factorization. [35] provides a theoretical characterization of when data filtering improves multimodal contrastive learning, offering a complementary, data-centric perspective to objective-level analyses. Also, [17, 21, 59] show that, under certain conditions, multimodal models outperform unimodal ones with better representations. However, these works assume an optimal solution to the non-convex problem without analyzing the training dynamics that lead to strong generalization. The zero-shot ability of VLMs also lacks full theoretical study. To the best of our knowledge, only [10] analyzes CLIP's zero-shot performance, showing it learns shared features while ignoring modality-specific ones. Yet, their setup does not consider real-world issues like misalignment between image and text. Beyond standard contrastive learning, [33] proposes a modified loss using unpaired data to detect ground-truth pairs and improve results, but only for linear models. So far, no work has theoretically studied the effect of text recaptioning on VLMs.

Contributions: To the best of our knowledge, this is the first theoretical work explaining why text recaptioning improves zero-shot generalization in VLMs, especially under image-text misalignment, where text may include spurious or missing features. We analyze the training dynamics of stochastic gradient descent (SGD) in multimodal contrastive learning and derive the generalization behavior of the learned model. Our analysis uses a one-hidden-layer ReLU network, which remains the state-of-the-art model in theoretical studies of contrastive [51] and supervised learning [2, 61]. All findings are validated empirically on practical VLMs like CLIP. A comparison to prior theory works is shown in Table 1. Key contributions include:

1. Theoretical training dynamics and generalization analysis of contrastive learning in nonlinear VLMs. We provide a theoretical analysis of jointly training two nonlinear encoders with contrastive loss. Prior works on training dynamics in contrastive learning [51, 10, 33] either analyze a single encoder or are restricted to linear neural networks. In contrast, our analysis captures the joint learning behavior of both nonlinear encoders with ReLU activation functions.

2. Theoretical characterization of the impact of misaligned image-text pairs on pre-training performance. We analyze a data model with modality misalignment, where some texts may contain features spuriously correlated with the image and others may omit relevant features. We show that spurious and missing features cause neurons to entangle true and irrelevant representations, which hinders the ability of the vision-language model to disentangle semantic components, ultimately degrading generalization performance.

3. Theoretical justification of enhanced out-of-domain generalization through pre-training with text recaptioning. We first analyze the training dynamics of the text generation process and formally prove that the resulting text after recaptioning has reduced spurious correlation and enhanced semantic relevance with the corresponding images. When these filtered texts are used for contrastive pre-training, the resulting model exhibits improved feature purity and succeeds in out-of-domain zero-shot classification, whereas the model trained on raw data provably fails.

## 1.1 Related Works

Vision-Language Models: VLMs [58, 48, 37, 19, 28, 26] are trained via contrastive learning on large web-sourced image-text pairs. Following CLIP, later models [30, 1, 56] aim to boost zero-shot performance. Data quality has become a key bottleneck, leading to recent filtering efforts [13, 24, 49, 20, 27]. For example, LaCLIP [13] uses LLM-generated caption rewrites as augmentation, and BLIP [25] leverages synthetic captions to drop noisy pairs, enhancing feature quality and robustness.

Theoretical Exploration on Contrastive Learning. Recent studies explore why contrastive learning yields effective representations. [47] identifies alignment and uniformity as key properties of contrastive loss. [15] shows that solving auxiliary prediction tasks improves contrastive representations. [44] highlights the role of inductive biases in shaping learning dynamics. [29] proves that multimodal contrastive learning can recover shared latent factors under a generative model.

Generalization analyses of Neural Networks (NNs). Various approaches have been developed to analyze the generalization of feedforward NNs. The neural tangent kennel (NTK) approach shows that overparameterized networks can be approximated by kernel methods in the limiting case [18, 2]. The model estimation approach assumes the existence of a ground-truth one-hidden-layer model with desirable generalization and estimates the model parameters using the training data [63]. The feature learning approach analyzes how a shallow NN learns important features during training and thus achieves desirable generalization [31, 23, 43].

## 2 Problem Formulation and Algorithm

VLMs leverage large-scale web-based datasets containing paired visual and textual data to pre-train two separate encoders: an image encoder f and a text encoder h , parameterized by weights W and V , respectively. Contrastive learning serves as the core framework, ensuring the learned embeddings of matching pairs are closer while separating mismatched pairs.

̸

Specifically, let S be the indices of the image-text pairs, e.g., ( x p , y p ) with p ∈ S . ( x p , y p ) is referred to as a positive pair, while ( x p , y n ) with p = n is referred to as a negative pair. We minimize the following spectral loss function:

<!-- formula-not-decoded -->

where the hyper-parameter τ &gt; 0 is referred as the temperature. The spectral contrastive loss L in (1) has been extensively utilized in recent theoretical works [15, 42, 60]. Although it differs from the commonly used SimCLR [9] in practice, the spectral contrastive loss closely resembles SimCLR numerically, as shown in [15].

## 2.1 Training Framework

Let S = S h ∪ S w include human-annotated high-quality image-text pairs with indices in S h and noisy web low-quality dataset with indices in S w . Due to the inherently noisy nature of web data, the learned embeddings from (1) may be suboptimal. To mitigate this, many practical training methods [25, 13] incorporate recaptioned text to improve the quality and diversity of image-text pairs. While specific implementations vary, most frameworks follow a similar four-stage approach:

(S1) Image-text contrastive pre-training (ITCP) on raw data: The image encoder f and text encoder h are trained using the image-text pairs { ( x p , y p ) } p ∈ S by minimizing the contrastive loss as in (1). Let W and V denote the learned weights in f and h . We then estimate the image and text

embeddings of ( x p , y p ) by z ′ x p = f W ( x p ) and z ′ y p = h V ( y p ) . Due to the low-quality data in S w when training the encoders, these estimations might not be accurate.

(S2) Generating text captions: The high-quality data pairs in S h are used to finetune an imagegrounded text decoder G , which maps an image x p to text through G ( x p ) . Then, the learned G is applied to every image x p in S w to generate a synthetic caption ˆ y p = G ( x p ) . Next, the estimated text embedding of ˆ y p is computed as ˆ z y p = h V (ˆ y p ) = h V ( G ( x p )) , where V represents the weights of h learned from Stage (S1).

(S3) Filtering: For every ( x p , y p ) in S w , we compute the cosine similarity between the image embedding z ′ x p and the text embeddings of the original caption z ′ y p and the synthetic caption ˆ z y p , respectively. If the pair ( z ′ x p , ˆ z y p ) has higher similarity to each other than the pair ( z ′ x p , z ′ y p ) , ( x p , y p ) is replaced with ( x p , ˆ y p ) . Let ˜ S w denote the index set of the resulting data pairs. By filtering noisy captions in S w with synthetic captions that better align with image embeddings, ˜ S w becomes a cleaner dataset.

(S4) ITCP on filtered data: The image encoder f and text encoder h are trained by minimizing the contrastive loss in (1), repeating the procedure from Stage (S1) with the only difference being that the original dataset S is replaced by ˜ S = S h ∪ ˜ S w . The resulting loss is denoted by ˜ L ( f, h ) . Let ˜ W and ˜ V denote the resulting learned weights. f ˜ W and g ˜ V can produce improved embeddings compared with f W and g V .

We employ stochastic gradient descent (SGD) with step size η and batch size B , following standard practice. Despite the non-convexity of (1), we present a detailed analysis of the resulting training dynamics and establish convergence guarantees in Section 4. This stands in contrast to existing works [39, 53, 10] that assume the attainability of a global optimum.

## 2.2 Downstream Tasks

As a demonstration of the performance of the learned model ( f ˜ W , g ˜ V ) , we consider a downstream image classification task in a zero-shot setting. Unlike the regression and binary classification tasks to evaluate the uni-modal contrastive learning in [51], we consider a K -classification problem for any constant K ≥ 2 . Each class label is associated with a given text prompt y k , where k ∈ [ K ] . For any image x with its ground-truth label l x ∈ [ K ] , the zero-shot predicted label by the pre-trained models ( f ˜ W , g ˜ V ) is computed as arg max k ∈ [ K ] ⟨ f ˜ W ( x ) , g ˜ V ( y k ) ⟩ . This approach follows the typical setting of zero-shot image classification using VLMs [10, 19, 25]. The prediction is considered accurate if and only if arg max k ∈ [ K ] ⟨ f ˜ W ( x ) , g ˜ V ( y k ) ⟩ = l x .

## 3 Technical Assumptions and Setups

We introduce a set of assumptions that are either derived conceptually from the real data distribution or follow existing approaches in contrastive learning theory.

## 3.1 Backbone of the Encoders

We use a two-layer neural network with ReLU activation functions as the image and text encoders, respectively. Formally, we have

Definition 3.1. The image encoder f W : R d 1 → R m and text encoder h V : R d 1 → R m is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Because deep neural networks are highly nonlinear, analyzing the training dynamics and resulting generalization performance of learned models remains challenging. As a result, existing theoretical studies are largely limited to one-hidden-layer neural networks [2, 61, 51, 33], where the learning problem is already nonconvex. In this paper, we extend this line of research to a more complex setting, where two such encoders are jointly trained for image and text modalities.

## 3.2 Data Model for ITCP

Our data model in Assumption 3.1 builds on the sparse coding framework, which has been widely used in both uni-modal contrastive learning for images [2, 51] and multi-modal image-text contrastive learning [10]. This sparse coding model has been employed in theoretical analyses [4, 7, 14] because it effectively models the practical NLP [5, 6, 36, 11] and image data [55, 52, 54].

Assumption 3.1 (Sparse coding model for image-text pairs) . Each image-text pair ( x p , y p ) , p ∈ S , is generated i.i.d. from the following sparse coding form:

<!-- formula-not-decoded -->

where x p , y p ∈ R d 1 , z x p , z y p ∈ R d , and d 1 = poly( d ) . We assume:

- (a) Image dictionary: M = [ M 1 , . . . , M d ] ∈ R d 1 × d is column-orthonormal.
- (b) Text dictionary: H = [ H 1 , . . . , H d ] ∈ R d 1 × d is column-orthonormal.
- (c) Additive noise: ξ x p , ξ y p ∼ N (0 , σ 2 ξ I d 1 ) with ω (1 /d 1 ) ≤ σ 2 ξ ≤ O (√ log d/d 1+ c 0 ) .

(d) Sparse latent vector: z x p = ( z 1 x p , . . . , z d x p ) with z j x p ∈ { 0 , ± 1 } , where | z j x p | ∼ Bernoulli ( C z /d ) .

̸

Notably, we operate in a regime where the noise magnitude can dominate the signal: since ω (1 /d 1 ) &lt; σ 2 ξ ≤ O (√ log d/d 1+ c 0 ) 1 , we have ∥ ξ ∥ 2 2 ≫ Θ(1) ≫∥ M z ∥ 2 , indicating that the overall noise energy significantly exceeds that of the signal. Nevertheless, we will show that contrastive learning remains effective even under such high-noise conditions, due to the encoders' ability to extract denoised and purified features, as characterized in Theorem 4.4. An intuitive explanation for why feature recovery is still possible lies in the different alignment properties of the signal and noise: for any active feature z j = 0 , the signal aligns well with its corresponding basis: |⟨ M z, M j ⟩| = Θ(1) , while the noise contribution remains small, |⟨ ξ, M j ⟩| ≤ O (1 / √ d ) .

We introduce Assumptions 3.2 and 3.3 to capture the characteristics of the dataset S = S h ∪ S w . Notably, the number of high-quality pairs in S h may be significantly fewer than that of low-quality pairs in S w , with | S h | = Θ( d 2 ) and | S w | = poly ( d ) ≫ ω ( d 2 ) .

Assumption 3.2 (High-quality image-text pairs) . Every high-quality image-text pair ( x p , y p ) with p ∈ S h satisfies z x p = z y p , i.e., the image and text have the same latent vector.

Compared to high-quality pairs in S h , low-quality pairs in S w show modality misalignment due to spurious image-text correlations and missing descriptions of key visual features.

Assumption 3.3 (Low-quality misaligned image-text pairs) . There exists a constant C s ∈ ( ω (1 / log d ) , 1 / 2) such that for every low-quality pair ( x p , y p ) in S w and every image feature M j ( j ∈ [ d ] ) in x p , we have

<!-- formula-not-decoded -->

̸

where the first term in (5) is the probability that a text feature H j ′ ( j ′ = j ) is spuriously correlated to the image feature M j , and the second term is the probability that H j is missing in the text while the image feature M j exists.

Consider the blue Mercedes-Benz example from [34]. Here, M j denotes the car's visual feature, while H j ′ refers to unrelated price information spuriously correlated with M j , illustrating the first term in (5). The correct text feature 'Mercedes-Benz' is H j ; its absence reflects the omission of a relevant feature, as captured by the second term in (5). We focus on a single spurious pair ( j, j ′ ) for simplicity. Since our analysis depends on the total spurious feature probability (bounded by C s ), the results extend to multiple spurious features as long as their total probability stays within C s .

## 3.3 Image-Grounded Text Decoder G in Stage (S2)

Recall that G is employed in Stage (S2) to generate synthetic text captions. In practice, the core idea behind the widely adopted approaches [25, 58, 48] is to train the encoder-decoder model G and

√

1 The columns M j and H j are column-orthonormal with each entry bounded by ˜ O (1 / d 1 ) , ensuring small inner products with isotropic noise.

leverage the high-quality image-text pairs S h to improve its performance. In this paper, we consider a simplified form of G , given by:

<!-- formula-not-decoded -->

where σ denotes the ReLU function. The parameters W and V are learned by solving

<!-- formula-not-decoded -->

initialized at W and V , using SGD with step size η . Although G in (6) is a conceptual simplification, where σ ( W x p ) acts as the encoder and V ⊤ as the decoder, it serves as a realistic abstraction to illustrate the underlying advantages of synthetic text caption generation.

## 3.4 Zero-Shot Generalization on Image Classification

We consider an out-of-domain (OOD) setting for testing images and text prompts as follows.

Image: Each test image x can be approximated by a sparse coding model with dictionary M ′ ,

<!-- formula-not-decoded -->

where M ′ = MP 1 , and max i,j | ( P 1 ) ij -δ ij | ≤ O (1 / √ d ) . The noise ξ x matches the training distribution (Assumption 3.1(d)) and δ ij denotes the Kronecker delta function.

Text: Each class k ∈ [ K ] has a prompt that has a sparse decomposition

<!-- formula-not-decoded -->

If x belongs to class k , then among all K binary vectors z ′ y k ′ , z ′ x is maximally aligned with z ′ y k ,

̸

<!-- formula-not-decoded -->

This formulation reflects the intuition that x belongs to class k if its sparse representation is most similar to the sparse representation of class k 's text prompt.

## 4 Main Results

## 4.1 Intuition and Informal Insights

Before presenting our main results, we first offer an intuitive explanation of the encoder-learner's success. To learn the latent representation z from input pair ( x, y ) , a well-trained image encoder f and text encoder g must ensure that each feature pair ( M j , H j ) is captured by at least one neuron pair ( w i , v i ) , without interference from spurious signals. We call this a purified feature , meaning the neuron pair encodes only one true feature with no contamination. In this case, ⟨ w i , x ⟩ ≈ z j x and ⟨ v i , y ⟩ ≈ z j y , so f and g recover the full latent space z . But in real data, where high-quality pairs in S h are rare and noisy pairs with misaligned image-text pairs in S w dominate, achieving this is difficult. See Appendix B.1 for proof sketches and we summarize main findings below:

(I) SGD provably solves the nonconvex training problems (1). The existing training dynamics and convergence analyses are limited to either single-modal contrastive learning [51] or linear networks [10, 33]. Theorem 4.1 provides a convergence analysis of SGD for solving the nonconvex ITCG problem when the network contains nonlinear activations for both modalities.

(II) Failure of learning due to spurious correlations. Theorem 4.2 provides a negative result: if f and g are directly trained on the raw data S , the model inevitably learns M j and M j ′ together via some w i , and H j and H j ′ together via some v i . As a result, the model fails to distinguish between these spuriously correlated features.

(III) Successful learning with recaptioning and filtering . Theorem 4.3 demonstrates that recaptioned texts significantly suppress spurious features and enhance relevant feature alignment. Building on this, Theorem 4.4 states that training f and g on the recaptioned data ˜ S enables the resulting encoder pair to learn purified representations of M j and H j accurately, as if trained solely on sufficient high-quality data. This highlights the advantage of leveraging the recaptioned data ˜ S w .

(IV) Enhanced zero-shot image classification accuracy due to text recaptioning . The advantage of using synthetic text captions is further validated in downstream tasks. As shown in Theorem 4.5, for a zero-shot out-of-domain multi-class image classification task, ITCP trained using ˜ S achieves high accuracy, whereas ITCP directly using S fails to generalize accurately.

## 4.2 Feature Purity Improvements in Converged Models via Recaptioned Data

We first characterize the training dynamics and convergence of solving (1) using SGD in Stage (S1) and (S4) in Section 2.1. Let L ∗ and ˜ L ∗ denote the optimal values of the contrastive loss on the raw dataset S and the filtered dataset ˜ S , respectively. Note that ( W , V ) and ( ˜ W , ˜ V ) are the converged weights from contrastive training on S and ˜ S in Stage (S1) and (S4), respectively.

Theorem 4.1 ( Convergence of ITCP ) . Suppose Assumptions 3.1 to 3.3 hold. Let the model complexity be m = d 1 . 01 , initialized at w (0) i , v (0) i ∼ N (0 , σ 2 0 I d 1 ) , where σ 2 0 = Θ ( 1 d 1 poly ( d ) ) . After T = Θ ( d 2 log d ) iterations with batch size B = Ω( d ) and η = O (1) , the returned weights achieve a loss that is sufficiently close to the optimal loss in Stage (S1) and (S4), respectively, i.e.,

<!-- formula-not-decoded -->

Remark 4.1. Theorem 4.1 demonstrates that SGD iterations can converge to weights that achieve a near optimal loss of (1), respectively. This result is of independent interest, as existing training dynamics and convergence analyses for contrastive loss are limited to linear networks. Here, we extend such analysis to nonconvex optimization settings where the network contains nonlinear ReLU activations. Next, we characterize the feature purity of the learned models.

Theorem 4.2 ( Unsuccessful learning of ITCP on raw data S with low feature purity ) . For each neuron pair ( ¯ w i , ¯ v i ) in ( W , V ) , there exists a spurious feature pair ( j, j ′ ) ∈ [ d ] such that

<!-- formula-not-decoded -->

where α 2 i,j , α 2 i,j ′ = Θ ( ∥ ¯ w i ∥ 2 2 + ∥ ¯ v i ∥ 2 2 ) and ∥ r i ∥ 2 2 , ∥ s i ∥ 2 2 ≤ O (( ∥ ¯ w i ∥ 2 2 + ∥ ¯ v i ∥ 2 2 ) /d ) . Moreover, for every spuriously correlated pair ( j, j ′ ) , there exist at least Ω(1) neuron pairs ( ¯ w i , ¯ v i ) that primarily learn the mixed feature pair ( M j , M j ′ , H j , H j ′ ) .

Remark 4.2. Theorem 4.2 indicates that the model learned by ITCP on raw data achieves only limited feature purity. Specifically, a neuron pair ( ¯ w i , ¯ v i ) learns a mixture of image and text features, respectively. M j and M j ′ are always mixed together, as are H j and H j ′ . As a result, the learned weights W and V fail to produce purified representations, making it difficult to distinguish between features j and j ′ , which ultimately degrades downstream performance shown in (15).

Theorem 4.3 ( Spurious feature suppression and relevant feature preservation by recaptioned texts ) . After T = Θ( d log d ) steps of SGD, the decoder G in (6) , finetuned by solving (7) , converges to weights ( ˆ W , ˆ V ) with expected loss L C ≤ Θ(1 /d ) . The recaptioned texts in ˜ S w are computed by ˆ y p = G ( x p ) . Then for any index j ∈ [ d ] such that | z j x p | = 1 , the decoder output satisfies:

̸

<!-- formula-not-decoded -->

Remark 4.3. After captioning and filtering, the resulting text contains fewer spurious features and more aligned feature pairs than raw data. Compared with Assumption 3.3, the probability of spurious features can be reduced from a constant C s in S w to Θ(1 /d ) in ˜ S w , while the probability of retaining all aligned feature pairs increases from C s in S w to 1 -Θ(1 /d ) in ˜ S w . The resulting dataset ˜ S = S h ∪ ˜ S w has better-aligned image-text pairs, enabling higher feature purity in contrastive training. We next show how ITCP trained on ˜ S improves feature purity.

Theorem 4.4 ( Successful learning of ITCP on filtered data ˜ S with high feature purity ) . For each neuron pair ( ˜ w i , ˜ v i ) in ( ˜ W , ˜ V ) , there exists j ∈ [ d ] such that ( ˜ w i , ˜ v i ) primarily learns ( M j , H j )

<!-- formula-not-decoded -->

where ˜ α 2 i,j = Θ( ∥ ˜ w i ∥ 2 2 + ∥ ˜ v i ∥ 2 2 ) and ∥ ˜ r i ∥ 2 2 , ∥ ˜ s i ∥ 2 2 ≤ O ( ( ∥ ˜ w i ∥ 2 2 + ∥ ˜ v i ∥ 2 2 ) /d ) . Moreover, for every feature j ∈ [ d ] , there exist at least Ω(1) neuron pairs ( ˜ w i , ˜ v i ) that primarily learn purified feature pair ( M j , H j ) .

Remark 4.4. Theorem 4.4 indicates that the model learned by ITCP on filtered data achieves a purified representation. Specifically, a neuron pair ( ˜ w i , ˜ v i ) learns one single feature pair ( M j , H j ) , respectively. As a result, ˜ W and ˜ V yield purified representations that effectively separate individual features, enabling improved downstream performance shown in (16).

## 4.3 Performance Comparison on Downstream Tasks

We next compare the performance of the models ( f W , g V ) and ( f ˜ W , g ˜ V ) on the zero-shot image classification problem with out-of-domain data described in Sections 2.2 and 3.4.

Theorem 4.5 (Zero-Shot Image Classification) . For the OOD zero-shot K -class image classification problem, the model ( f W , g V ) from ITCP using raw data has a constant failure probability:

<!-- formula-not-decoded -->

In contrast, the model ( f ˜ W , g ˜ V ) from ITCP using filtered caption succeeds with high probability:

<!-- formula-not-decoded -->

Remark 4.5. Theorem 4.5 first demonstrates that the zero-shot performance of ( f W , g V ) is unsatisfactory, resulting from the low feature purity in ( f W , g V ) , as established in Theorem 4.2. Theorem 4.5 further shows that ( f ˜ W , g ˜ V ) achieves accurate classification. This success is attributed to high feature purity in ( f ˜ W , g ˜ V ) , as described in Theorem 4.4. Note that Theorem 4.5 holds for image data with a distribution shift from the training data.

## 5 Experiment

## 5.1 Simulated Experiment

Figure 1: Performance comparison of ITCP on raw data and filtered (recaptioned) data when the probability of spurious correlation C s changes. (a) Number of features that have purified representation in the model (b) Average magnitude of purified presentations (c) Zero-shot out-ofdomain classification accuracy (d) Silhouette Score with cosine distance.

<!-- image -->

̸

Experiment Setup. Wefirst validate our results via simulated experiments, using the same framework from Section 2.1. We adopt a more general spurious correlation model than Assumption 3.3, allowing each M j to be spuriously linked with multiple H j ′ ( j ′ = j ), while keeping the total spurious correlation probability at C s . We set d 1 = 2500 , d = 50 , | S w | = 5000 , | S h | = 1000 , and use m = 80 neurons. Matrices M , H are drawn from standard Gaussians and orthonormalized via QR decomposition. Sparse codes z x follows Bernoulli (0 . 1) Noise variance σ 2 ξ = 1 /d . SGD runs with batch size 500 and step size 0.001. Downstream evaluation uses 5-way classification with test z x ∼ Bernoulli (0 . 2) ; class codes z y k partition the d -dim space. Results are averaged over 20 trials. Models ( W , V ) and ( ˜ W , ˜ V ) are trained on raw and filtered data, respectively.

Improved feature representation using filtered (recaptioned) data. We say a weight ¯ w i learn a purified representation of M j if its projection along M j achieves the largest magnitude and satisfies |⟨ ¯ w i , M j ⟩| / ∥ ¯ w i ∥ &gt; 0 . 5 . The same applies to ( ˜ W , ˜ V ) . Figure 1(a) shows the number of features M j (out of d = 50 total features) for which at least one neuron in W (or ˜ W , respectively) learns a purified

Table 2: of CLIP and LaCLIP on Accuracy (%) and Silhouette Score.

|                | Food-101   | Food-101   | CIFAR-10   | CIFAR-10   | Caltech-101   | Caltech-101   | CIFAR-100   | CIFAR-100   | Pets   | Pets    | STL-10   | STL-10   |
|----------------|------------|------------|------------|------------|---------------|---------------|-------------|-------------|--------|---------|----------|----------|
| Model          | Acc        | SS         | Acc        | SS         | Acc           | SS            | Acc         | SS          | Acc    | SS      | Acc      | SS       |
| CC12M CLIP     | 50.8       | 0.034      | 64.9       | 0.113      | 77.4          | 0.225         | 38.5        | 0.005       | 64.1   | 0.069   | 91.0     | 0.195    |
| CC12M LaCLIP   | 60 . 7     | 0 . 038    | 75 . 1     | 0 . 157    | 83 . 3        | 0 . 276       | 43 . 9      | 0 . 029     | 72 . 4 | 0 . 070 | 95 . 1   | 0 . 273  |
| RedCaps CLIP   | 81.5       | 0.125      | 70.4       | 0.100      | 72.8          | 0.210         | 39.9        | - 0 . 002   | 82 . 7 | 0 . 091 | 92 . 8   | 0.226    |
| RedCaps LaCLIP | 85 . 0     | 0 . 175    | 74 . 8     | 0 . 107    | 76 . 4        | 0 . 233       | 40 . 7      | 0 . 011     | 78.2   | 0.074   | 91.4     | 0 . 275  |
| LAION CLIP     | 85.5       | 0.116      | 93.0       | 0.181      | 91.2          | 0.258         | 71.7        | 0.078       | 90.1   | 0.122   | 97.3     | 0.223    |
| LAION LaCLIP   | 86 . 5     | 0 . 148    | 93 . 5     | 0 . 215    | 92 . 4        | 0 . 306       | 73 . 9      | 0 . 108     | 90 . 9 | 0 . 152 | 98 . 4   | 0 . 260  |

representation. The results show that ITCP trained on filtered data learns purified representations for nearly all features, even at high spurious correlation levels ( C s = 0 . 3 ). In contrast, ITCP on raw data degrades significantly, with purity dropping faster as C s increases. Moreover, Figure 1(b) shows the average of the largest projection magnitudes among neurons that learn purified features. The magnitude from ˜ W (ITCP on filtered data) is consistently higher than that from W , indicating stronger purified representations. This aligns with Theorems 4.2, 4.4 and Remark 4.4.

Improved zero-shot out-of-domain performace using filtered (recaptioned) data. Figure 1(c) compares the classification accuracy of both models on zero-shot out-of-domain data. The model trained on filtered data consistently outperforms the one trained on raw data, with the performance gap widening as spurious correlations in the raw data increase. We also adopt the widely used Silhouette Score (SS) with cosine distance [57, 32, 62] to evaluate feature embedding quality in different clusters, as shown in Figure 1(d). A higher SS indicates better intra-class alignment and inter-class orthogonality, reflecting more purified representations. These results verify Theorem 4.5.

Impact of feature purity. When C s reaches 0.35 in Figure 1, even the filtered data fails to maintain full feature purification: the number of neurons learning disentangled representations of all d = 50 features drops significantly (Figure 1(a)), and the SS (Figure 1(d)) and classification accuracy (Figure 1(c)) both decline sharply. This highlights that feature purity -the extent to which each neuron aligns to a single semantic direction-is a key bottleneck in contrastive pretraining and downstream generalization. We provide extra results in Appendix A.1.

## 5.2 Experiments on Practical Data and Models

LaCLIP improves generalization over CLIP via recaption. Tables 2 compare CLIP [37] and LaCLIP [13], which share the same architecture and datasets, except LaCLIP replaces part of the original captions with LLM-generated rewrites. 'CC12M CLIP' denotes a CLIP model pretrained on raw CC12M [8], while 'CC12M LaCLIP' uses the same model and data but with LLM-rewritten captions. Other models are obtained similarly using RedCaps [12] and LAION [40] datasets. We evaluate their zero-shot classification accuracy and Silhouette Scores on various downstream datasets. LaCLIP generally outperforms CLIP in both metrics, empirically validating that higher-quality captions improve zero-shot generalization. Additional ImageNet results are reported in Table 3 of Appendix A.2.

Next, we study the feature purity using a CLIP model pretrained on CC3M [41]. Both the image and text encoders are 12-layer transformers that produce features in R 768 , which are subsequently projected into a shared embedding space of R 512 through final linear projection layers, as illustrated in Figure 6 of Appendix A.2. The final linear projection layer has 512 neurons and is functionally aligned with V in our theoretical model. We now present two key findings from this setting:

Purified neurons enhance generalization. To investigate the effect of feature purity on generalization, we prune the neurons in the final linear layer in different ways and evaluate the resulting zero-shot classification performance. Specifically, we rank the 512 neurons by their average pairwise absolute cosine similarity to all other neurons, from lowest to highest. The absolute cosine similarity of neurons v j , v j ′ is computed as |⟨ v j , v j ′ ⟩| / ∥ v j ∥∥ v j ′ ∥ for all j, j ′ ∈ { 1 , 2 , . . . , 512 } . A lower average indicates higher feature purity (i.e., more orthogonal representations), while a higher value suggests feature mixing. We evaluate three pruning strategies: (1) retaining high-purity neurons, i.e., with lowest similarity, (2) retaining low-purity neurons, i.e., with highest similarity, and (3) retaining a random subset of neurons. The number of retained neurons is varied from 200 to 500. As shown in Figure 2 (a-c,e-g), downstream performance is the best when retaining high-purity neurons,

followed by random selection, with low-purity neurons performing the worst. These results highlight the critical role of purified features in downstream generalization.

Data misalignment reduces feature purity. To study how image-text misalignment affects feature purity, we randomly shuffling texts across image-text pairs in CC3M with probability C m , as illustrated in Figure 7 of Appendix A.2, thereby introducing a controlled probability of modality misalignment. We then use the shuffled dataset to fine-tune the last linear projection layer only of the pretrained CLIP model, freezing other layers. We then compute the cosine similarities of all 512 neuron weight vectors v j ∈ R 768 of the fine-tuned model. Figure 2 (d) reports the average absolute cosine similarity of all neuron pairs, while (h) presents a histogram of cosine similarity ⟨ v j , v j ′ ⟩ / ( ∥ v j ∥∥ v j ′ ∥ ) . One can see that as C m increases, the average absolute cosine similarity increases, and the neurons become less orthogonal to each other and tend to encode mixed representations, resulting in lower feature purity. This coincides with the decreases classification accuracy in downstream tasks, as shown in Table 4 of Appendix A.2.

Figure 2: Left (a-c,e-g): Retaining high-purity neurons outperform random and low-purity neurons in downstream tasks. More datasets shown in Figure 8. Right(d,h): When C m increases, the neurons have higher cosine similarity and reduced feature purity.

<!-- image -->

## 6 Conclusion

This paper provides a theoretical analysis of contrastive learning with nonlinear networks, linking training dynamics to generalization. We identify feature purity as central to generalization and show that text recaptioning enhances purity and zero-shot performance. The theory is empirically validated on benchmarks. Future work includes extending to Transformer models and tasks like retrieval and visual question answering.

## Acknowledgments

This work was supported by National Science Foundation (NSF) #2430223, Army Research Office (ARO) W911NF-25-1-0020, and the Rensselaer-IBM Future of Computing Research Collaboration (http://airc.rpi.edu). The work of Shuai Zhang was supported by NSF #2349879. We also thank all anonymous reviewers for their constructive comments.

## References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems , 35:23716-23736, 2022.

- [2] Zeyuan Allen-Zhu and Yuanzhi Li. Feature purification: How adversarial training performs robust deep learning. In 2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS) , pages 977-988. IEEE, 2022.
- [3] Chenxin An, Jiangtao Feng, Kai Lv, Lingpeng Kong, Xipeng Qiu, and Xuanjing Huang. Cont: Contrastive neural text generation. Advances in Neural Information Processing Systems , 35:2197-2210, 2022.
- [4] Sanjeev Arora, Rong Ge, and Ankur Moitra. New algorithms for learning incoherent and overcomplete dictionaries. In Conference on Learning Theory , pages 779-806. PMLR, 2014.
- [5] Sanjeev Arora, Yuanzhi Li, Yingyu Liang, Tengyu Ma, and Andrej Risteski. A latent variable model approach to pmi-based word embeddings. Transactions of the Association for Computational Linguistics , 4:385-399, 2016.
- [6] Sanjeev Arora, Yuanzhi Li, Yingyu Liang, Tengyu Ma, and Andrej Risteski. Linear algebraic structure of word senses, with applications to polysemy. Transactions of the Association for Computational Linguistics , 6:483-495, 2018.
- [7] Boaz Barak, Jonathan A Kelner, and David Steurer. Dictionary learning and tensor decomposition via the sum-of-squares method. In Proceedings of the forty-seventh annual ACM symposium on Theory of computing , pages 143-151, 2015.
- [8] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 3558-3568, 2021.
- [9] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning , pages 1597-1607. PMLR, 2020.
- [10] Zixiang Chen, Yihe Deng, Yuanzhi Li, and Quanquan Gu. Understanding transferable representation learning and zero-shot transfer in clip. In Proceedings of the 12th International Conference on Learning Representations (ICLR) , Vienna, Austria, 2024.
- [11] Mingyang Deng, Lucas Tao, and Joe Benton. Measuring feature sparsity in language models. arXiv preprint arXiv:2310.07837 , 2023.
- [12] Karan Desai and Justin Johnson. Redcaps: Web-curated image-text data for robust image captioning. In NeurIPS , 2021.
- [13] Lijie Fan, Dilip Krishnan, Phillip Isola, Dina Katabi, and Yonglong Tian. Improving clip training with language rewrites. Advances in Neural Information Processing Systems , 36:35544-35575, 2023.
- [14] Karol Gregor and Yann LeCun. Learning fast approximations of sparse coding. In Proceedings of the 27th international conference on international conference on machine learning , pages 399-406, 2010.
- [15] Jeff Z HaoChen, Colin Wei, Adrien Gaidon, and Tengyu Ma. Provable guarantees for selfsupervised deep learning with spectral contrastive loss. Advances in Neural Information Processing Systems , 34:5000-5011, 2021.
- [16] Xiaowei Hu, Zhe Gan, Jianfeng Wang, Zhengyuan Yang, Zicheng Liu, Yumao Lu, and Lijuan Wang. Scaling up vision-language pre-training for image captioning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 17980-17989, 2022.
- [17] Yu Huang, Chenzhuang Du, Zihui Xue, Xuanyao Chen, Hang Zhao, and Longbo Huang. What makes multi-modal learning better than single (provably). Advances in Neural Information Processing Systems , 34:10944-10956, 2021.
- [18] Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems , volume 31. Curran Associates, Inc., 2018.

- [19] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning , pages 4904-4916. PMLR, 2021.
- [20] Wonjae Kim, Sanghyuk Chun, Taekyung Kim, Dongyoon Han, and Sangdoo Yun. Hype: Hyperbolic entailment filtering for underspecified images and texts. In European Conference on Computer Vision , pages 247-265. Springer, 2025.
- [21] Jason D Lee, Qi Lei, Nikunj Saunshi, and Jiacheng Zhuo. Predicting what you already know helps: Provable self-supervised learning. Advances in Neural Information Processing Systems , 34:309-323, 2021.
- [22] Meir Yossef Levi and Guy Gilboa. The double-ellipsoid geometry of clip. arXiv preprint arXiv:2411.14517 , 2024.
- [23] Hongkang Li, Songtao Lu, Xiaodong Cui, Pin-Yu Chen, and Meng Wang. Understanding mamba in in-context learning with outliers: A theoretical generalization analysis. In Highdimensional Learning Dynamics 2025 , 2025.
- [24] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning , pages 19730-19742. PMLR, 2023.
- [25] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping languageimage pre-training for unified vision-language understanding and generation. In International conference on machine learning , pages 12888-12900. PMLR, 2022.
- [26] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems , 34:9694-9705, 2021.
- [27] Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Omer Levy, Luke Zettlemoyer, Jason Weston, and Mike Lewis. Self-alignment with instruction backtranslation. In Proceedings of the International Conference on Learning Representations (ICLR) , 2024.
- [28] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al. Oscar: Object-semantics aligned pre-training for vision-language tasks. In Computer Vision-ECCV 2020: 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part XXX 16 , pages 121-137. Springer, 2020.
- [29] Xuechen Li, Tengyu Ma, and Sanjeev Arora. Identifiability results for multimodal contrastive learning. In International Conference on Learning Representations (ICLR) , 2023.
- [30] Yangguang Li, Feng Liang, Lichen Zhao, Yufeng Cui, Wanli Ouyang, Jing Shao, Fengwei Yu, and Junjie Yan. Supervision exists everywhere: A data efficient contrastive language-image pre-training paradigm. In International Conference on Learning Representations , 2022.
- [31] Yuanzhi Li and Yingyu Liang. Learning overparameterized neural networks via stochastic gradient descent on structured data. Advances in neural information processing systems , 31, 2018.
- [32] Yujie Mo, Zhihe Lu, Runpeng Yu, Xiaofeng Zhu, and Xinchao Wang. Revisiting self-supervised heterogeneous graph learning from spectral clustering perspective. Advances in Neural Information Processing Systems , 37:43133-43163, 2024.
- [33] Ryumei Nakada, Halil Ibrahim Gulluk, Zhun Deng, Wenlong Ji, James Zou, and Linjun Zhang. Understanding multimodal contrastive learning and incorporating unpaired data. In International Conference on Artificial Intelligence and Statistics , pages 4348-4380. PMLR, 2023.
- [34] Thao Nguyen, Samir Yitzhak Gadre, Gabriel Ilharco, Sewoong Oh, and Ludwig Schmidt. Improving multimodal datasets with image captioning. Advances in Neural Information Processing Systems , 36, 2024.

- [35] Divyansh Pareek, Sewoong Oh, and Simon Shaolei Du. Understanding the gain from data filtering in multimodal contrastive learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems .
- [36] Victor Prokhorov, Yingzhen Li, Ehsan Shareghi, and Nigel Collier. Learning sparse sentence encoding without supervision: An exploration of sparsity in variational autoencoders. In Proceedings of the 6th Workshop on Representation Learning for NLP (RepL4NLP-2021) , pages 34-46, Online, 2021. Association for Computational Linguistics.
- [37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning , pages 8748-8763. PMLR, 2021.
- [38] Noam Rotstein, David Bensaïd, Shaked Brody, Roy Ganz, and Ron Kimmel. Fusecap: Leveraging large language models for enriched fused image captions. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision , pages 5689-5700, 2024.
- [39] Nikunj Saunshi, Jordan Ash, Surbhi Goel, Dipendra Misra, Cyril Zhang, Sanjeev Arora, Sham Kakade, and Akshay Krishnamurthy. Understanding contrastive learning requires incorporating inductive biases. In International Conference on Machine Learning , pages 19250-19286. PMLR, 2022.
- [40] Christoph Schuhmann, Robert Beaumont, Richard Vencu, et al. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs, 2021. arXiv preprint arXiv:2111.02114.
- [41] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of ACL , 2018.
- [42] Kendrick Shen, Robbie M Jones, Ananya Kumar, Sang Michael Xie, Jeff Z HaoChen, Tengyu Ma, and Percy Liang. Connect, not collapse: Explaining contrastive learning for unsupervised domain adaptation. In International conference on machine learning , pages 19847-19878. PMLR, 2022.
- [43] Jiawei Sun, Hongkang Li, and Meng Wang. Theoretical learning performance of graph neural networks: The impact of jumping connections and layer-wise sparsification. arXiv preprint arXiv:2507.05533 , 2025.
- [44] Yuandong Tian, Lantao Yu, Xinlei Chen, and Surya Ganguli. Understanding self-supervised learning with dual deep networks. arXiv preprint arXiv:2010.00578 , 2020.
- [45] Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, and Lijuan Wang. Git: A generative image-to-text transformer for vision and language. arXiv preprint arXiv:2205.14100 , 2022.
- [46] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In International conference on machine learning , pages 23318-23340. PMLR, 2022.
- [47] Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In Advances in Neural Information Processing Systems (NeurIPS) , volume 33, pages 16070-16080, 2020.
- [48] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for vision and vision-language tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 19175-19186, 2023.
- [49] Yiping Wang, Yifang Chen, Wendan Yan, Alex Fang, Wenjing Zhou, Kevin Jamieson, and Simon Shaolei Du. Cliploss and norm-based data selection methods for multimodal contrastive learning. arXiv preprint arXiv:2405.19547 , 2024.

- [50] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. Simvlm: Simple visual language model pretraining with weak supervision. arXiv preprint arXiv:2108.10904 , 2021.
- [51] Zixin Wen and Yuanzhi Li. Toward understanding the feature learning process of self-supervised contrastive learning. In International Conference on Machine Learning , pages 11112-11122. PMLR, 2021.
- [52] Pan Xiao, Peijie Qiu, Sung Min Ha, Abdalla Bani, Shuang Zhou, and Aristeidis Sotiras. Sc-vae: Sparse coding-based variational autoencoder with learned ista. Pattern Recognition , 161:111187, 2025.
- [53] Yihao Xue, Kyle Whitecross, and Baharan Mirzasoleiman. Investigating why contrastive learning benefits robustness against label noise. In International Conference on Machine Learning , pages 24851-24871. PMLR, 2022.
- [54] Jianchao Yang, Kai Yu, Yihong Gong, and Thomas Huang. Linear spatial pyramid matching using sparse coding for image classification. In 2009 IEEE Conference on computer vision and pattern recognition , pages 1794-1801. IEEE, 2009.
- [55] Meng Yang, Lei Zhang, Jian Yang, and David Zhang. Robust sparse coding for face recognition. In CVPR 2011 , pages 625-632. IEEE, 2011.
- [56] Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. Filip: Fine-grained interactive language-image pre-training. In International Conference on Learning Representations , 2022.
- [57] Chunlin Yu, Ye Shi, and Jingya Wang. Contextually affinitive neighborhood refinery for deep clustering. Advances in Neural Information Processing Systems , 36:5778-5790, 2023.
- [58] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917 , 2022.
- [59] Amir Zadeh, Paul Pu Liang, and Louis-Philippe Morency. Foundations of multimodal colearning. Information Fusion , 64:188-193, 2020.
- [60] Qi Zhang, Yifei Wang, and Yisen Wang. On the generalization of multi-modal contrastive learning. In International Conference on Machine Learning , pages 41677-41693. PMLR, 2023.
- [61] S. Zhang, M. Wang, P.-Y. Chen, S. Liu, S. Lu, and M. Liu. Joint edge-model sparse learning is provably efficient for graph neural networks. In The Eleventh International Conference on Learning Representations , 2023.
- [62] Xiang Zhang, Ziyuan Zhao, Theodoros Tsiligkaridis, and Marinka Zitnik. Self-supervised contrastive pre-training for time series via time-frequency consistency. Advances in neural information processing systems , 35:3988-4003, 2022.
- [63] Kai Zhong, Zhao Song, Prateek Jain, Peter L Bartlett, and Inderjit S Dhillon. Recovery guarantees for one-hidden-layer neural networks. In International conference on machine learning , pages 4140-4149. PMLR, 2017.

The overall structure of the appendix is as follows. Each appendix provides supplementary information that supports the main content of this document but is not included in the main body to maintain clarity and flow.

## · Appendix A: Extra Experiments

Additional experiments including both synthetic simulations and CLIP/LaCLIP evaluations on omitted datasets.

## - A.1 Extra Simulated Experiment

Complements Section 5.1 with further analysis of neuron behavior trained on simulated data.

## - A.2 Extra CLIP/LaCLIP Experiment

Complements Section 5.2 by evaluating on datasets omitted due to space.

## · Appendix B: Preliminaries

Mathematical preliminaries and notation used throughout the paper. A proof sketch is also provided to outline the key ideas behind the main results.

## · Appendix C: Technical Lemmas

Full statements and proofs of supporting lemmas used in the theoretical analysis.

## · Appendix D-J: Proofs and Theoretical Analysis

## - Appendix D-F: ITCP on Raw Data (Phase I-III)

Theoretical proof of ITCP across three training phases on raw data.

## - Appendix G: Captioning

Theoretical proof of reception using high quality data.

## - Appendix H: Filtering

Theoretical proof of filtering noisy caption-text pairs.

## - Appendix I: ITCP on Synthetic (Recaptioned) Data

Theoretical proof of training dynamics when using synthetic recaptions.

## - Appendix J: Downstream Task Evaluation

Theoretical implications for performance on downstream tasks.

## · Checklist

## A Extra Experiment

All experiments were conducted on an internal compute cluster using 8 NVIDIA A5000 GPUs with 24 GB memory each, and each run completed within 50 GPU-hours. No large-scale pretraining or resource-intensive tuning was performed beyond the reported experiments.

## A.1 Extra Simulated Experiment

This section extends the analysis in Section 5.1 by providing additional simulated experiments on neuron behavior under synthetic data training.

̸

Neurons trained on filtered data exhibit a more concentrated distribution. Figure 3 visualizes the histograms of |⟨ ¯ v i , H j ⟩| / ∥ ¯ v i ∥ and |⟨ ˜ v i , H j ⟩| / ∥ ˜ v i ∥ for all i ∈ [ m ] and j ∈ [ d ] . The values of |⟨ ˜ v i , H j ⟩| / ∥ ˜ v i ∥ are more concentrated, typically around 0 . 05 and 0 . 7 . In contrast, the values for |⟨ ¯ v i , H j ⟩| / ∥ ¯ v i ∥ are less concentrated. This phenomenon is consistent with Theorem 4.4, which indicates that for every H j , certain neurons ˜ v i in ˜ V predominately learns H j . In such cases, |⟨ ˜ v i , H j ⟩| approaches 1 , while |⟨ ˜ v i , H j ′ ⟩| / ∥ ˜ v i ∥ approaches 0 for j ′ = j . The concentrated values of 0 . 05 and 0 . 7 observed in Figure 3 are due to noise in the data. In contrast, feature alignment is less significant for V , leading to less concentration of the corresponding values. Similar results are obtained for image encoder |⟨ w i , M j ⟩| , deferred to Figure 4.

Figure 3: Histogram of |⟨ ¯ v i , H j ⟩| / ∥ ¯ v i ∥ for ITCP on raw data and |⟨ ˜ v i , H j ⟩| / ∥ ˜ v i ∥ for ITCP on filtered data (split into two figures to highlight the significant differences in the value distributions).

<!-- image -->

Figure 4: Histogram of |⟨ ¯ w i , M j ⟩| / | ¯ w i | for ITCP on raw data and |⟨ ˜ w i , M j ⟩| / ˜ w i for ITCP on filtered data (split into two figures to highlight the significant differences in the value distributions).

<!-- image -->

Enhanced class separation of downstream tasks by ITCP with recaptioned data . Figure 5 visualizes the t-distributed stochastic neighbor embedding (t-SNE) of the feature embeddings generated by the two models, computed as f W ( x p ) and f ˜ W ( x p ) for each x p , respectively. The t-SNE method projects the high-dimensional embeddings onto a two-dimensional map. One can see that the embeddings from different groups are more distinctly separated in the model trained using ITCP on recaptioned data, indicating that this approach achieves better feature alignment.

## A.2 Extra Experiment on CLIP and LaCLIP

To complement the results in Section 5.2, we report additional experiments on CLIP and LaCLIP using datasets omitted from the main text due to space constraints.

ImageNet Results. The LaCLIP variants consistently surpass their CLIP counterparts on both Top-1 and Top-5 accuracy. Higher silhouette scores further indicate cleaner feature separation after recaptioning, in line with our theoretical predictions.

CLIP architecture. Figure 6 illustrates the CLIP architecture used in our experiments. Both image and text inputs are independently encoded by 12-layer transformer backbones, each producing a

Figure 5: t-SNE visualization of text embedding with spurious correlation probability C s .

<!-- image -->

Table 3: Comparison of CLIP and LaCLIP on ImageNet: Top-1 (%), Top-5 (%), and Silhouette Score.

| Model             | Top-1 (%)   | Top-5 (%)     | Silhouette   |
|-------------------|-------------|---------------|--------------|
| CC12M CLIP        | 35.04       | 62.10 70 . 17 | -0.014639    |
| CC12M LaCLIP      | 42 . 62     |               | - 0 . 008141 |
| LAION-400M CLIP   | 58.34       | 84.73         | -0.029893    |
| LAION-400M LaCLIP | 62 . 27     | 86 . 34       | - 0 . 056593 |
| RedCaps CLIP      | 37.66       | 63.31         | -0.022045    |
| RedCaps LaCLIP    | 39 . 66     | 66 . 06       | - 0 . 012269 |

768-dimensional feature vector. These features are then projected into a shared 512-dimensional embedding space through learned linear projection matrices W ∈ R 768 × 512 and V ∈ R 768 × 512 , corresponding to the image and text encoders in our theorem, defined in Eq. (2). The resulting embeddings are aligned via a contrastive loss that maximizes similarity for matched image-text pairs while minimizing similarity for unmatched pairs. This architecture forms the foundation for our analyses on neuron selection and feature purity in the shared embedding space.

Figure 6: Architecture of CLIP used in our experiments. Both image and text encoders are 12-layer transformers that output features in R 768 , which are then projected into a shared R 512 embedding space via final linear projection layers W and V , corresponding to Eq. (2) and Eq. (3) in our theoretical analysis. Contrastive loss is computed between the resulting image and text embeddings.

<!-- image -->

Simulating Modality Misalignment via Caption Shuffling. Figure 7 illustrates how modality misalignment is introduced by randomly shuffling text captions across image-text pairs with probability C m , resulting in noisy supervision for contrastive learning.

Purified neuron selection enhances generalization. Figure 8 presents additional experimental results on CIFAR-100, Pets, and STL-10, complementing the main results reported in Figure 2. Due to space constraints, we include only Food-101, CIFAR-10, and Caltech-101 in the main text. All experiments follow the same protocol, evaluating zero-shot classification accuracy and Silhouette Score under different neuron selection strategies. These results consistently support our core finding: selecting high-purity neurons leads to improved downstream performance across diverse datasets.

Higher shuffling probability leads to reduced generalization and feature purity. Table 4 presents additional experimental results on CLIP models finetuned with different levels of randomly shuffling probability C m to simulate spurious correlation, showing that both accuracy and Silhouette Score consistently decrease as C m increases.

Figure 7: Simulating Modality Misalignment via Caption Shuffling. Starting from original aligned image-text pairs, a controlled probability C m of misalignment is introduced by randomly shuffling the text captions. This results in noisy pairs that reflect varying levels of spurious correlations.

<!-- image -->

Figure 8: Zero-shot classification accuracy (top) and Silhouette Score (bottom) under different neuron selection strategies for CIFAR-100, Pets, and STL-10 datasets.

<!-- image -->

## B Preliminaries

We first restate some important notations used in the Appendix, which are summarized in Table 5.

## B.1 Proof Scratch

Theorem 4.1 is proven by integrating the convergence analyses in Appendix F and Appendix I. Appendix F establishes convergence for ITCP on raw data, while Appendix I extends the convergence result to ITCP on synthetic data. Together, they verify that SGD with ReLU networks achieves near-optimal contrastive loss on both datasets.

Table 4: Accuracy (%) and Silhouette Score of CLIP models finetuned with varying C m on six datasets.

| Dataset    | C m = 0   | C m = 0   | C m = 0 . 1   | C m = 0 . 1   | C m = 0 . 3   | C m = 0 . 3   | C m = 0 . 5   | C m = 0 . 5   | C m = 0 . 8   | C m = 0 . 8   |
|------------|-----------|-----------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
|            | Acc       | SS        | Acc           | SS            | Acc           | SS            | Acc           | SS            | Acc           | SS            |
| Caltech101 | 59.7      | 0.160     | 48.2          | 0.124         | 47.9          | 0.121         | 43.6          | 0.117         | 44.5          | 0.115         |
| CIFAR-10   | 57.9      | 0.030     | 50.7          | 0.012         | 49.5          | 0.013         | 46.5          | 0.013         | 44.1          | 0.011         |
| CIFAR-100  | 26.4      | - 0 . 038 | 19.5          | - 0 . 042     | 17.8          | - 0 . 043     | 17.4          | - 0 . 044     | 16.2          | - 0 . 048     |
| Food-101   | 12.9      | - 0 . 073 | 10.9          | - 0 . 052     | 10.9          | - 0 . 056     | 11.1          | - 0 . 057     | 11.1          | - 0 . 059     |
| Pets       | 13.9      | - 0 . 005 | 13.3          | - 0 . 006     | 13.2          | - 0 . 009     | 13.4          | - 0 . 011     | 12.6          | - 0 . 012     |
| STL-10     | 86.3      | 0.164     | 79.8          | 0.103         | 79.2          | 0.102         | 78.8          | 0.100         | 78.3          | 0.097         |

Table 5: Summary of Notations

| Notations                     | Annotation                                                                                                                   |
|-------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| M ∈ R d 1 × d , H ∈ R d 1 × d | M is the image dictionary matrix, H is the text dictionary matrix.                                                           |
| W ∈ R m × d 1 , V ∈ R m × d 1 | W is the weight of image encoder, V is the weight of text encoder.                                                           |
| x p ∈ R d 1 , y p ∈ R d 1     | x p and y p represent an image and a text data, respectively.                                                                |
| z x p , z y p ∈ R d           | z x p and z y p are the sparse signals of image and text, respectively. z y k is the sparse signal for the text prompt y k . |
| z j x p , z j y p             | z j x p is the j -th coordinate of z x p ; z j y p is the j -th coordinate of z y p .                                        |
| L , L C                       | L is the loss for ITCP; L C is the loss for Image-grounded Text De- coding.                                                  |
| S = S h ∪ S w                 | S w is the noisy web low-quality dataset; S h is the human-annotated high-quality dataset.                                   |
| ˜ S = S h ∪ ˜ S w             | ˜ S w replaces noisy captions in S w with synthetic captions.                                                                |
| T 1                           | Phase I of ITCP with b ( t ) i = 0 .                                                                                         |
| T 2                           | Phase II of ITCP with b ( t +1) i = (1+ η d ) b ( t ) i .                                                                    |
| T 3                           | Phase III of ITCP with b ( t +1) i = b ( T 2 ) i .                                                                           |
| T C                           | Stage of training caption generators.                                                                                        |
| S j, sure                     | The set of well-initialized neurons ( w i , v i ) on features ( M j , H j ) .                                                |

Theorem 4.2 is proven across Appendix D, Appendix E, and Appendix F. Specifically, Appendix D models Phase I training ( t ≤ T 1 ) and proves that neurons simultaneously align with true features and spuriously correlated features due to comparable gradient contributions, preventing pure feature separation. Appendix E analyzes Phase II training ( T 1 &lt; t ≤ T 2 ) and shows that this spurious alignment continues to strengthen, as neurons with initial mixed alignment further amplify their entanglement during continued SGD updates. Appendix F establishes the convergence behavior during Phase III ( T 2 &lt; t ≤ T 3 ), showing that the network stabilizes into mixed solutions where each neuron represents a combination of multiple features. These detailed stages collectively prove the failure of purified feature alignment as formalized in Theorem 4.2.

Theorem 4.3 is proven across Appendix G and Appendix H. Specifically, Appendix G analyzes the captioning stage, where the decoder is fine-tuned on clean data to generate synthetic captions. It proves that for neurons aligned with true features, the alignment towards the true features grows exponentially while the alignment towards spurious features remains negligible. This ensures that the synthetic captions preserve relevant features and suppress spurious ones. Appendix H then formalizes the filtering process, demonstrating that after replacing noisy captions with synthetic ones, the resulting dataset satisfies much stronger feature purity conditions, with spurious correlations

suppressed to Θ(1 /d ) and true features preserved with probability 1 -Θ(1 /d ) . These results directly support the purified feature learning described in Theorem 4.3.

Theorem 4.4 is proven in Appendix I, which integrates the proofs of Phase I, Phase II, and Phase III for ITCP on synthetic data. Specifically, Appendix I first establishes in Phase I that purified training pairs allow neurons aligned with true features to grow exponentially without spurious interference. It then shows in Phase II that these alignments continue to strengthen while suppressing non-informative neurons, leading to clear feature separation. Finally, it proves in Phase III that the model converges, achieving a bounded final loss and dominant true feature alignment. Since the overall proof structure closely mirrors that of Theorem 4.2 (which was proven separately across Appendix D, Appendix E, and Appendix F), we consolidate all stages into a single appendix for brevity and clarity.

Theorem 4.5 is proven in Appendix J, which analyzes the downstream zero-shot classification. Appendix J shows that for ITCP on raw data, spurious features cause a constant classification error, while for ITCP on synthetic data, true and spurious features become separable with high probability, leading to an o (1) error rate. This directly supports the main text conclusion on downstream generalization.

## B.2 Feature Coupling and Expected Values in S w

The following Assumption B.1 corresponds to the more specific forms of Assumptions 3.2 and 3.3 discussed earlier.

Assumption B.1 (High and low quality pairs) . The high-quality image-text pairs in S h have size | S h | = Θ( d 2 ) . The low-quality image-text pairs in S w have size | S w | = poly ( d ) |≫ ω ( d 2 ) .

In S h , for a positive pair ( x p , y p ) , we assume perfect alignment, meaning z x p = z y p . Consequently, the following holds:

̸

<!-- formula-not-decoded -->

To model the misaligned features in low-quality pairs in S w , where spurious misalignment occurs at a non-negligible level, we assume [ d ] can be divided into d/ 2 disjoint sets, each containing exactly two entries. Let ( j, j ′ ) ⊂ [ d ] denote one such set, referred to as a 'spuriously correlated set. ' The following assumptions capture the nature of spurious and true alignments:

<!-- formula-not-decoded -->

̸

These assumptions imply that true alignment dominates, with Pr( | z j y p | = 1 | | z j x p | = 1) &gt; 1 2 , while spurious alignment exists at a constant percentage level, making it non-negligible. The intuition behind this assumption is that each feature j is paired with exactly one spuriously correlated feature j ′ , ensuring that j is not associated with any other feature j ′′ = j ′ . This design simplifies the analysis while effectively capturing the key challenges posed by low-quality data.

Then, for a positive pair ( x p , y p ) with p in S w , we have:

<!-- formula-not-decoded -->

where ( j, j ′ ) is a spuriously correlated set.

̸

For negative pairs ( x p , y q ) , where p = q , and p, q ∈ S , we have:

<!-- formula-not-decoded -->

In S w , mismatched text and image pairs are prevalent compared to S h . For a postive pair ( x p , y p ) , we assume log(1 /c 0 ) 2 log d &lt; Pr( | z j ′ y p | = 1 | | z j x p | = 1) &lt; 1 2 . To model this, we assume that for each primary

̸

<!-- formula-not-decoded -->

feature j ∈ [ d ] , there exists exactly one spurious feature j ′ such that j and j ′ are uniquely coupled. This implies that j cannot be associated with any other feature j ′′ = j ′ . Mathematically, the coupling is defined as:

For a positive pair ( x p , y p ) in S w , the probabilities of spurious and aligned features are further constrained:

<!-- formula-not-decoded -->

The lower bound is established in Lemma C.8.

and:

<!-- formula-not-decoded -->

Under these assumptions, the expected values for the aligned and spurious features are calculated as follows:

For the aligned feature j , we have:

<!-- formula-not-decoded -->

For the spurious feature j ′ , we have:

<!-- formula-not-decoded -->

The total expected value across both aligned and spurious features satisfies:

<!-- formula-not-decoded -->

Here, j ′ denotes the spurious feature associated with j .

## B.3 Gradient

The contrastive loss in vision-language models (VLM) is defined as follows:

<!-- formula-not-decoded -->

where τ &gt; 0 is a temperature parameter.

We perform stochastic gradient descent (SGD) on this contrastive loss. Let f ( t ) and h ( t ) be the image encoder and text encoder networks at iteration t , respectively. The network parameters are updated as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where b ( t ) i , the bias term, is manually tuned during training and thus excluded from gradient updates.

The gradient of L ( f ( t ) , h ( t ) ) with respect to w ( t ) i at iteration t is given by:

<!-- formula-not-decoded -->

Similarly, the empirical gradient of L ( f ( t ) , h ( t ) ) with respect to v ( t ) i is:

<!-- formula-not-decoded -->

## B.4 Alignment Updates

We analyze how each neuron i ∈ [ m ] aligns with the feature M j during each iteration of SGD. The alignment can be described by the following update rule:

<!-- formula-not-decoded -->

Similarly, for ⟨ v ( t +1) i , H j ⟩ , the update rule becomes:

<!-- formula-not-decoded -->

Using Lemma C.6, we know that with high probability, ∑ x n ∈ N ⟨ f ( t ) ( x n ) ,h ( t ) ( y p ) ⟩ τ ≤ O ( 1 d ) , so in Eq (30) the sum of second term and third term is always less than the first term, until ⟨ f ( t ) ( x n ) , h ( t ) ( y p ) ⟩ = Θ( d ) .

The updates for the components ⟨ w ( t +1) i , M j ⟩ , ⟨ v ( t +1) i , H j ⟩ , ⟨ w ( t +1) i , M j ′ ⟩ , and ⟨ v ( t +1) i , H j ′ ⟩ (where j ′ represents the spurious aligned feature corresponding to j ) can be expressed concisely in matrix form as follows:

<!-- formula-not-decoded -->

where the coefficients are defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

This matrix representation highlights the interactions between the alignment of true and spurious features during SGD updates. The diagonal elements a dominate the contribution from existing alignments, while the off-diagonal terms b, c capture the mutual influence between paired features and spurious alignments. Note that if c is very small, it indicates that the spurious alignment ( j ′ ) has minimal influence, allowing w i to focus on learning purified features. Conversely, if c is large, the spurious alignment could significantly interfere with the learning process, hindering the purification of features. The error term Err t accounts for higher-order noise or unmodeled effects in the update process.

Assuming a single spurious feature is a simplification for presentation that was made for ease of presentation in the proof and can be extended to a more general setting without altering the underlying insights. If each feature j has K -1 spurious correlates, (34) becomes a 2 K × 2 K matrix, and N i = j, j ′ in the last sentence of Theorem 4.2 contains j and other K -1 features. Our analysis relies on the total spurious feature probability (bounded by C s ), not the number of correlated features, so as long as the sum of all spurious feature probabilities is upper bounded by C s , the core mechanism and insights of the theorem remain unchanged.

## C Technical Lemmas

Definition C.1 (Neuron Characterization) . Let us define a few notations to characterize each neuron w ( t ) i 's behavior. For every constant c 0 ∈ (0 , 1) and γ ∈ (0 , 0 . 1) , by choosing c 1 = 2 + 2(1 -γ ) c 0 and c 2 = γc 0 , we define:

1. Let S ( t ) j, sure ⊆ [ m ] be those neurons i ∈ [ m ] satisfying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

2. Let S ( t ) j, pot ⊆ [ m ] be those neurons i ∈ [ m ] satisfying

<!-- formula-not-decoded -->

Lemma C.1 (Geometry at initialization) . We initialize the parameters by w (0) i ∼ N (0 , σ 2 0 I d 1 ) , where σ 2 0 = Θ ( 1 d 1 poly ( d ) ) . We have with probability ≥ 1 -o (1 /d 3 ) over the random initialization, for all j ∈ [ d ] :

<!-- formula-not-decoded -->

Proof. If g is standard Gaussian, then for every t &gt; 0 ,

<!-- formula-not-decoded -->

We initialize the parameters by w (0) i ∼ N (0 , σ 2 0 I d 1 ) , where σ 2 0 = Θ ( 1 d 1 poly ( d ) ) . We have 1 n ∑ n i =1 ⟨ w (0) i , M i ⟩ ∼ N ( 0 , σ 2 0 n ) .

Therefore, for every i ∈ m and j ∈ d , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let S (0) j, sure ⊆ [ m ] be those neurons i ∈ [ m ] satisfying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By concentration with respect to all m choices of i ∈ [ m ] , we know with probability at least 1 -o ( 1 d 3 ) it satisfies ∣ ∣ ∣ S (0) j, sure ∣ ∣ ∣ = Ω ( d γ 4 c 0 ) .

Let S (0) j, pot ⊆ [ m ] be those neurons i ∈ [ m ] satisfying

<!-- formula-not-decoded -->

By concentration with respect to all m choices of i ∈ [ m ] , we know with probability at least 1 -o ( 1 d 3 ) it satisfies ∣ ∣ ∣ S (0) j, pot ∣ ∣ ∣ = O ( d 2 γc 0 ) .

More details of the proof can be found in Lemma B.2 of [2].

Lemma C.2. With high probability 1 -1 poly ( d ) , for every i ∈ [ m ] , the following holds:

<!-- formula-not-decoded -->

Lemma C.3. With high probability 1 -1 poly ( d ) , for every i ∈ [ m ] , the following holds:

<!-- formula-not-decoded -->

Proof. Let X ∼ χ 2 n . By standard properties of the chi-squared distribution, we know that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

In our case, we consider ∥ MM ⊤ w (0) i ∥ 2 2 + ∥ HH ⊤ v (0) i ∥ 2 2 σ 2 0 ∼ χ 2 2 d . Setting δ = 1 poly ( d ) , we have n = 2 d , and thus, with high probability 1 -1 poly ( d ) , the following holds:

<!-- formula-not-decoded -->

Rearranging and incorporating the scaling factor σ 2 0 , we get:

<!-- formula-not-decoded -->

Lemma C.4 (Noise Projection Bound) . For the spurious dense noise ξ x p ∼ N (0 , σ 2 ξ I d 1 ) , where the variance satisfies ω ( 1 d 1 ) ≤ σ 2 ξ ≤ O ( 1 d ) , the following holds with high probability 1 -e -Ω( d 1 ) :

<!-- formula-not-decoded -->

Proof. For all j ∈ [ d 1 ] , by the properties of the Gaussian distribution, we have:

<!-- formula-not-decoded -->

Now, consider the term |⟨ w i , ξ ⟩| 2 . We decompose it as:

<!-- formula-not-decoded -->

For the first term, since |⟨ M j , ξ ⟩| 2 ≤ O ( 1 d 1+ c 0 ) with high probability, we have:

<!-- formula-not-decoded -->

Similarly, for the second term:

<!-- formula-not-decoded -->

Combining these, we have:

<!-- formula-not-decoded -->

Since ∥ MM ⊤ w i ∥ 2 2 + ∥ M ⊥ M ⊥ ⊤ w i ∥ 2 2 = ∥ w i ∥ 2 2 , we conclude:

<!-- formula-not-decoded -->

Thus, the lemma holds.

Lemma C.5 (Tail Bound for Matrix Product) . Let Q ∈ R n × n be a symmetric matrix, and let w,v be independent zero-mean Gaussian random vectors with covariance matrix I n . Define

<!-- formula-not-decoded -->

Then, for any δ &gt; 0 , the following tail bound holds:

<!-- formula-not-decoded -->

Lemma C.6 (Bound Inner Product) . Consider the inner product between the feature vectors at initialization:

<!-- formula-not-decoded -->

Here, using Lemma C.5, Q = xy ⊤ , with ∥ Q ∥ op = Θ(1) , ∥ Q ∥ F = Θ(1) and σ 2 0 = Θ ( 1 d 1 poly ( d ) ) . Then, at initialization ( t = 0 ), the following holds:

<!-- formula-not-decoded -->

Lemma C.7 (Concentration bound for empirical loss and gradients) . There exist N ≥ poly ( d ) for some sufficiently large polynomial and all ∥ w i ∥ 2 ≤ O ( d ) , i ∈ [ m ] , it satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. The proof can be done by trivial VC dimension or Rademacher complexity arguments similarly to Lemma A.2. [2].

Lemma C.8 (Misalignment Probability Bound) . The probability of spurious alignment satisfies:

<!-- formula-not-decoded -->

Proof. By concentration over all m choices of i ∈ [ m ] , we find that with probability at least 1 -o ( 1 d 3 ) , the number of neurons satisfying:

<!-- formula-not-decoded -->

is o (1) .

In addition, for all neurons, we have:

<!-- formula-not-decoded -->

Define:

Thus:

<!-- formula-not-decoded -->

We begin by expressing a + b -c and a + b + c as functions of P 1 = Pr( | z j y p | = 1 | | z j ′ x p | = 1) and P 2 = Pr( | z j y p | = 1 | | z j x p | = 1) , where P 1 + P 2 = 1 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using Eq (62), Eq (35) and Eq (36), we derive:

<!-- formula-not-decoded -->

Substituting back, we find:

<!-- formula-not-decoded -->

For example, setting c 0 = 0 . 1 , γ = 0 . 005 , d = 100 , and d 1 = 10000 , we calculate:

<!-- formula-not-decoded -->

This concludes the proof by bounding Pr( | z j y p | = 1 | | z j ′ x p | = 1) under the given conditions.

## D ITCP on Raw Data I

In this section we analyze Phase I of ITCP on Raw Data as the training iterations t ≤ T 1 , where T 1 = Θ ( d log d η ) is the iteration when all ∥ w ( T 1 ) i ∥ 2 2 + ∥ v ( T 1 ) i ∥ 2 2 2 ≥ ∥ w (0) i ∥ 2 2 + ∥ v (0) i ∥ 2 2 . When t ≤ T 1 , we set b ( t ) i = 0 . For every neuron i ∈ [ m ] , the weights w i and v i exhibit an increase in alignment along the direction of informative features M and H , while showing negligible increase in alignment along the direction of noise features M ⊥ and H ⊥ .

Based on subsection B.2, we have Pr( | z j y p | = 1 | | z j ′ x p | = 1) = Θ(1) , so E [ z j x z j y ] and E [ z j x z j ′ y ] both in Θ ( 1 d ) . In this case, w ( t +1) i is jointly influenced by M j and M j ′ , with both features contributing comparably to the updates.

To simplify our analysis, we consider the worse case where Pr( | z j ′ y p | = 1 | | z j x p | = 1) = Pr( | z j y p | = 1 | | z j x p | = 1) = 1 2 such that E [ z j x z j y ] = E [ z j x z j ′ y ] = C z 2 d , so using Eq (35), Eq (36) and b ( t ) i = 0 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This represents the worst-case scenario as the contributions of the aligned feature E [ z j x z j y ] and the spurious feature E [ z j x z j ′ y ] are identical. Under real circumstances, we expect E [ z j x z j y ] &lt; E [ z j x z j ′ y ] , which would result in ⟨ w ( t +1) i , M j ⟩ &gt; ⟨ w ( t +1) i , M j ′ ⟩ . However, in this worst-case scenario, the equality of contributions prevents the network from prioritizing purified features, resulting in equal magnitudes for ⟨ w ( t +1) i , M j ⟩ and ⟨ w ( t +1) i , M j ′ ⟩ , thereby hindering effective feature separation.

We first provide a lower bound for ∥ MM ⊤ w ( t ) i ∥ 2 2 for iterations t ≤ t 1 . From Eq (122) and Eq (69) we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The detailed proof of Eq (71) can be found in Hypothesis C.4 of [51].

A similar result holds for ∥ HH ⊤ v ( t ) i ∥ 2 2 and ∥ H ⊥ ( H ⊥ ) ⊤ v ( t ) i ∥ 2 2 .

Eq (70) and Eq (71) shows that the image and text dictionary features M , H can grow exponentially, while the noisy features M ⊥ , H ⊥ remain almost unchanged when t ≤ T 1 .

For M ⊥ j where j ∈ [ d 1 ] \ [ d ] , using Eq (71), we obtain:

<!-- formula-not-decoded -->

This result demonstrates that the noisy features M ⊥ j experience nearly no increase during this phase, remaining insignificant in their contribution to the alignment of w i .

## D.1 Lower Bound of Alignment for i ∈ S j, sure

This section provides a analysis of the alignment growth for neurons i ∈ S j, sure . Specifically, we demonstrate that for every j ∈ [ d ] , if i ∈ S j, sure, the alignment ⟨ M j , w ( t ) i ⟩ 2 and its spurious alignment ⟨ M ′ j , w ( t ) i ⟩ 2 increase exponentially when t ≤ T 1 .

We now prove the lower bound of |⟨ w ( T 1 ) i , M j ⟩| 2 for i ∈ S j, sure :

<!-- formula-not-decoded -->

In ♢ we use Definition C.1. In ♡ we use Eq (70). In ♣ we use ∥ w ( T 1 ) i ∥ 2 2 + ∥ v ( T 1 ) i ∥ 2 2 2 ≥ ∥ w (0) i ∥ 2 2 + ∥ v (0) i ∥ 2 2 . In ♠ we use c 1 + c 2 &gt; 2(1 + c 0 -γc 0 ) .

Similarly, |⟨ w ( T 1 ) i , M j ′ ⟩| 2 have the same lower bound.

## D.2 Upper Bound of Alignment for i / ∈ S j, pot

In this subsection, we analyze the alignment of neuron i / ∈ S j, pot with the feature M j and provide an upper bound for |⟨ w ( T 1 ) i , M j ⟩| 2 . While neurons i / ∈ S j, pot still exhibit exponential growth in their alignment, their weaker initialization results in significantly smaller alignment compared to neurons in S j, sure, limiting their contribution to learning the feature M j .

To establish the bound, we begin with the following expression:

<!-- formula-not-decoded -->

Here, in ♢ , we use Lemma C.1, which captures the reduced alignment for neurons outside S j, pot . Similar to the analysis for i ∈ S j, sure, the alignment strength for i / ∈ S j, pot is weaker, as c 1 -c 2 is less than 2(1 + c 0 -γc 0 ) , leading to:

<!-- formula-not-decoded -->

This inequality highlights the slower alignment for neurons outside S j, pot, distinguishing their behavior from neurons in S j, sure. Consequently, i / ∈ S j, pot contributes less significantly to the alignment of M j , reinforcing the importance of initial affinity for effective alignment.

## D.3 Summary

At this stage ( t ≤ T 1 ), we do not consider the worst-case scenario where the probability bounds for feature coupling satisfy

<!-- formula-not-decoded -->

(as assumed in SubSection B.2). Thus, we summarize the results when t ≤ T 1 as follows:

1. For i ∈ S j, sure, the alignment strength satisfies:

<!-- formula-not-decoded -->

where j ′ represents the corresponding spurious alignment feature.

2. For i / ∈ S j, pot, the alignment strength satisfies:

<!-- formula-not-decoded -->

3. For M ⊥ j where j ∈ [ d 1 ] \ [ d ] , we have:

<!-- formula-not-decoded -->

These results demonstrate that when t ≤ T 1 , all features in M increase, but the alignment for i ∈ S j, sure, including the corresponding spurious alignment, grows significantly larger due to favorable initialization. In contrast, noisy features M ⊥ remain unchanged.

## E ITCP on Raw Data II

The Phase II of ITCP on Raw Data is defined as the training iterations T 1 &lt; t ≤ T 2 , where T 2 -T 1 = Θ ( d log d η ) .

At the beginning of this phase, we set the bias threshold as:

<!-- formula-not-decoded -->

During training, the bias threshold is iteratively updated as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In this phase, the dynamics of alignment vary depending on whether a neuron belongs to S j, sure or not:

- For i / ∈ S j, pot : The weights w i and v i show negligible alignment growth with both the informative features M j , H j and the noise features M ⊥ , H ⊥ . This is due to their weaker initialization, as shown in Phase I, and the effect of the indicator function when t ≥ T 1 which prevents them from being activated. As a result, their capacity to learn meaningful alignments during this phase is significantly limited.
- For i ∈ S j, sure : The weights w i and v i exhibit continued alignment growth with the informative features M j , H j . Additionally, their alignment with the corresponding spurious features M j ′ , H j ′ also increases due to their strong initialization, as shown in Phase I, and the effect of the indicator function when t ≥ T 1 , which ensures they are always activated.

By the end of this stage ( t = T 2 ), the weights w i , v i will predominantly focus on the features M j , H j if i ∈ S j, sure, while largely ignoring the features M j , H j if i / ∈ S j, pot, as well as the noise features M ⊥ , H ⊥ . This separation lays the foundation for the Phase II of ITCP on Raw Data, where spurious alignments are expected to further diminish due to the dominance of true feature alignments.

Similarly to the proof of t ≤ T To simplify our analysis, we still consider the worse case where

<!-- formula-not-decoded -->

1 ′ [ ′ ]

## E.1 Alignment for i ∈ S j, sure

This section provides a analysis of the alignment growth for neurons i ∈ S j, sure . Specifically, we demonstrate that for every j ∈ [ d ] , if i ∈ S j, sure, the alignment ⟨ M j , w ( t ) i ⟩ 2 and its spurious alignment ⟨ M ′ j , w ( t ) i ⟩ 2 increase exponentially when T 1 &lt; t ≤ T 2 .

For i ∈ S j, sure, using Lemma C.4, the following holds with high probability 1 -e -Ω( d 1 ) when T 1 &lt; t ≤ T 2 :

<!-- formula-not-decoded -->

Therefore, with high probability 1 -e -Ω( d 1 ) , using Eq (76) and Eq (79) the indicator function satisfies the condition when t = T 1 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using Eq (116) we know that ( 1 + η C z 2 d ) &gt; ( 1 + η d ) and using Eq (34) we have

<!-- formula-not-decoded -->

This implies that when t &gt; T 1 , the alignment strength of informative features surpasses the updated bias threshold b ( t ) i . Consequently, the indicator functions become consistently activated T 1 &lt; t ≤ T 2 such that

<!-- formula-not-decoded -->

until all neurons satisfy:

we can ensure that:

Using Eq (34), the weight dynamics for |⟨ w ( t +1) i , M j ⟩| can be expressed as when T 1 &lt; t ≤ T 2 :

<!-- formula-not-decoded -->

Similarly, |⟨ w ( T 1 ) i , M j ′ ⟩| 2 have the same result.

## E.2 Alignment for i / ∈ S j, pot

In this section, we analyze the alignment behavior for neurons i / ∈ S j, pot. Specifically, we demonstrate that for every j ∈ [ d ] , if i / ∈ S j, pot, the alignment ⟨ M j , w ( t ) i ⟩ 2 exhibits negligible growth during the interval T 1 &lt; t ≤ T 2 .

For i / ∈ S j, pot, using Eq (156), Eq (79) and Eq (76), we have with high probability 1 -e -Ω( d 1 ) , similarly to the proof of i ∈ S j, sure, the indicator function satisfies the condition when t = T 1 :

<!-- formula-not-decoded -->

We can ensure that:

<!-- formula-not-decoded -->

Using Eq (116) we know that ( 1 + o ( η d 2 ) ) &lt; ( 1 + η d ) and using Eq (34) we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This implies that when t &gt; T 1 , the alignment strength of informative features does not surpass the updated bias threshold b ( t ) i . Consequently, the indicator functions become consistently not activated T 1 &lt; t ≤ T 2 such that

<!-- formula-not-decoded -->

Using Eq (34), the weight dynamics for |⟨ w ( t +1) i , M j ⟩| can be expressed as when T 1 &lt; t ≤ T 2 :

<!-- formula-not-decoded -->

Because ( 1 + o ( η d 2 )) T 2 ≤ 1 + o ( 1 d ) , the growth in |⟨ w ( T 2 ) i , M j ⟩| is negligible. Consequently, we have:

## E.3 Summary

When T 2 = Θ ( d log d η ) , we know ( 1 + η C z d ) T 2 = poly ( d ) . Using Eq (76), we can ensure that when all neurons satisfy the following condition:

<!-- formula-not-decoded -->

we terminate the training process at T 2 = Θ ( d log d η ) . This ensures that the alignment has sufficiently progressed for effective learning.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, at this stage ( T 1 &lt; t ≤ T 2 ), we do not consider the worst-case scenario where the probability bounds for feature coupling satisfy

<!-- formula-not-decoded -->

We summarize the results when T 1 &lt; t ≤ T 2 as follows:

1. For i ∈ S j, sure, the alignment strength satisfies:

<!-- formula-not-decoded -->

where j ′ represents the corresponding spurious alignment feature.

2. For i / ∈ S j, pot, the alignment strength satisfies:

<!-- formula-not-decoded -->

3. For M ⊥ j where j ∈ [ d 1 ] \ [ d ] , we have:

<!-- formula-not-decoded -->

These results demonstrate that when T 1 &lt; t ≤ T 2 , the alignment for i ∈ S j, sure, including the corresponding spurious alignment, grows significantly larger. In contrast, the alignment strength for i / ∈ S j, pot and noisy features M ⊥ remains unchanged. Similar results also hold for v i .

## F ITCP on Raw Data III Convergence

In the previous section, we demonstrated that for t ≤ T 2 , the neurons ( w i , v i ) are sparsely activated and remain consistently activated for i ∈ S j, sure. Building on this result, this section establishes the convergence of these neurons to sparse solutions, providing a detailed analysis of their behavior during Phase III of ITCP on Raw Data. The following theorem outlines the convergence guarantees under these conditions.

The Phase III of ITCP on Raw Data is defined as the training iterations T 2 &lt; t ≤ T 3 , where T 3 -T 2 = Θ( d ) . At the beginning of this phase, we fix the bias threshold as b ( t ) i = b T 2 i for T 2 &lt; t ≤ T 3 . Because b ( T 2 ) i = ( 1 + η d ) Θ( d log d/η ) b ( T 1 ) i , it is easy to know that for t ≥ T 2 , only when ( x p , y p ) and ( x n , y n ) contain the true feature j and its corresponding spurious feature j ′ , the indicator functions remain consistently activated for i ∈ S j, sure .

Consequently, using Eq (27), Eq (30), and Eq (31), the loss function L becomes convex with respect to w i and v i independently when ( x p , y p ) and ( x n , y n ) contain the true feature j and its corresponding spurious feature j ′ .

At the end of Phase II, using Eq (81) , we know that ∥ w ( T 2 ) i ∥ 2 ≥ Ω( d ) . Consequently, we cannot only consider -⟨ f ( t ) ( x p ) , h ( t ) ( y p ) ⟩ , and the error term Err t becomes non-negligible.

Specifically, based on Eq (27), it can be observed that the term -⟨ f ( t ) ( x p ) , h ( t ) ( y p ) ⟩ is convex and l i,j, 1 = ∥ x p ∥ 2 ∥ y p ∥ 2 = Θ(1) -smooth. This ensures that the true features contribute consistently to the optimization process.

Additionally, L i,j, 2 = ( ⟨ f ( t ) ( x n ) ,h ( t ) ( y p ) ⟩ ) 2 2 τ is also convex, and we further establish its smoothness to provide a rigorous understanding of its behavior.

To analyze the l i,j, 2 -smoothness, we aim to find an upper bound that satisfies:

<!-- formula-not-decoded -->

The gradient difference for w i is given by:

<!-- formula-not-decoded -->

where l w i , 1 = ∥ x n ∥ 2 2 ∥ y p ∥ 2 2 ∥ v i, 1 ∥ 2 ∥ v i, 2 ∥ 2 ≤ O ( d ) and l w i , 2 = ∥ x n ∥ 2 2 ∥ y p ∥ 2 2 ( ∥ v i, 1 ∥ 2 ∥ w i, 2 ∥ 2 + ∥ w i, 1 ∥ 2 ∥ v i, 1 ∥ 2 ) ≤ O ( d ) .

Similarly, the gradient difference for v i is:

<!-- formula-not-decoded -->

Combining the results, we find:

<!-- formula-not-decoded -->

Thus, the total smoothness constant is:

<!-- formula-not-decoded -->

These results demonstrate that the loss function L remains convex and l i,j -smooth for neurons ( w i , v i ) when ( x p , y p ) and ( x n , y n ) contain the true feature j and its corresponding spurious feature j ′ during Phase III of ITCP on Raw Data, ensuring their convergence to sparse solutions while maintaining consistency in their activation patterns.

We verify that the following inequality holds

<!-- formula-not-decoded -->

Let L = max i ∈ m ( l i,j / (2 τ )) = Θ(1) and η = 1 L to ensure a monotonic decrease, plug Eq (28) and Eq (29) into Eq (178), we have

<!-- formula-not-decoded -->

Under our data assumptions for S w and conclusion in Eq (96) , we define w ∗ i = α ∗ i,j M j + α ∗ i,j ′ M j ′ , v ∗ i = α ∗ i,j H j + α ∗ i,j ′ H j ′ . Thus, L j ( w ∗ i , v ∗ i ) captures both the alignment with the true feature M j , H j and the spurious feature M j ′ , H j ′ , representing the minimal loss achievable under the influence of both true and spurious features in the optimization process. Using Eq (81) , we know w ( T 2 ) i = Θ( d ) , so L j ( w ∗ i , v ∗ i ) = -Θ( d ) .

By the property of smoothness, we have

<!-- formula-not-decoded -->

Take the telescope sum of from T 2 to T 3 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Generalized to every j ∈ d , the same convergence holds for all i ∈ S j, sure when ( x p , y p ) and ( x n , y n ) contain feature j, j ′ . For all ( x p , y p ) and ( x n , y n ) in S w , the following inequality holds:

<!-- formula-not-decoded -->

As a result, the relative difference is bounded by:

<!-- formula-not-decoded -->

## F.1 Summary

ITCP trained on raw data S undergoes Stages D-F. After T = Θ( d 2 log d ) SGD iterations with batch size B = Ω( d ) and learning rate η = O (1) , the resulting weights ( W , V ) minimize the contrastive loss in Eq. (1) up to a vanishing relative error:

<!-- formula-not-decoded -->

However, each neuron pair ( ¯ w i , ¯ v i ) in ( W , V ) , for i ∈ [ m ] , predominantly encodes a mixture of features indexed by a subset N i ⊆ [ d ] , with | N i | ≥ 2 . Specifically, we have:

<!-- formula-not-decoded -->

where α 2 i,j = Θ( ∥ ¯ w i ∥ 2 2 + ∥ ¯ v i ∥ 2 2 ) , and the interference from other features is small: β i,j /α i,j ≤ O (1 / √ d ) , γ i,j /α i,j ≤ O (1 / √ d 1 ) .

Moreover, for every spuriously correlated feature pair ( j, j ′ ) satisfying Assumption 3.3, there exists at least an Ω(1) many of neurons i ∈ [ m ] with N i = { j, j ′ } , indicating the prevalence of feature mixing due to data misalignment.

## G Captioning

In this stage, the model fine-tunes the pre-trained encoder parameters W and V to obtain the updated parameters ˆ W and ˆ V through Image-Text Contrastive Pre-training (ITCP) on raw data.

Given an image-text pair ( x p , y p ) in S w , the decoder generates synthetic captions ˆ y p = ˆ V T σ ( ˆ W x p ) , where σ ( · ) denotes the activation function. The Image-Grounded Text Decoder, initialized with W and V from the pre-trained encoders, is fine-tuned on S h by minimizing the following loss function:

<!-- formula-not-decoded -->

where ∥ · ∥ 2 denotes the Euclidean norm. This fine-tuning process refines the model to generate captions that are more closely aligned with the target text data in S h .

During the captioning, we sample a batch of image-text pairs S ( t ) h = { ( x p , y p ) } B p =1 ⊆ S h . We perform stochastic gradient descent on L C . At each iteration, we update as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

At the beginning of this phase, we set the bias threshold as:

<!-- formula-not-decoded -->

During training, the bias threshold is iteratively updated as:

<!-- formula-not-decoded -->

The gradient of L C with respect to w ( t ) i , v ( t ) i , W , and V at iteration t is given by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The alignment can be described by the following update rule:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## G.1 Alignment for i ∈ S j, sure

This section analyzes the alignment growth for neurons i ∈ S j, sure. Specifically, we show that when t ≤ T C , the alignment with the true feature M j grows exponentially if x p contains the true feature M j . In contrast, the alignment with the spurious feature M j ′ exhibits negligible growth, even for neurons i ∈ S j, sure. Specially,

1. For the true feature M j , based on the result in Eq (96) and the bias threshold in Eq (115), the indicator functions are always activated. This ensures that the neuron can consistently increase its alignment in the direction of the true feature M j .

2. For the spurious feature M j ′ , based on the result in Eq (96) and the bias threshold in Eq (115), the indicator functions remain non-activated. This prevents the neuron from increasing its alignment in the direction of the spurious feature M j ′ .

The details of proof as follow:

Using Eq (95), we know

<!-- formula-not-decoded -->

Using Eq (35) and Eq (36), we have

<!-- formula-not-decoded -->

Using Eq (40) and ( a + b -c ) T 1 + T 2 ≥ Ω( d 2 ) , with high probability 1 -O ( 1 √ d ) we have,

<!-- formula-not-decoded -->

Therefore, with high probability 1 -O ( 1 √ d ) we have

<!-- formula-not-decoded -->

We set b (0) i = √ ∥ w ( T 2 ) i ∥ 2 2 -∥ w ( T 1 ) i ∥ 2 2 2 , and using Eq (124), so similarly to the proof of Eq (86) we can prove:

1. For i ∈ S j, sure and x p contain the true feature M j , with high probability 1 -O ( 1 √ d ) the indicator functions become consistently activated 0 ≤ t ≤ T C such that:

<!-- formula-not-decoded -->

2. For i ∈ S j, sure and x p contain the corresponding spurious aligned feature M j ′ , with high probability 1 -O ( 1 √ d ) the indicator functions become consistently activated 0 ≤ t ≤ T C such that:

<!-- formula-not-decoded -->

3. For i / ∈ S j, pot and M ⊥ j where j ∈ [ d 1 ] \ [ d ] , we have:

<!-- formula-not-decoded -->

For the residual loss in Eq (119) and Eq (120), we bound the difference if 1 ∣ ∣ ∣ 〈 w ( t ) i ,x p 〉∣ ∣ ∣ ≥ b ( t ) i = 1 :

<!-- formula-not-decoded -->

̸

In ♢ , we employ the approximation y p x ⊤ p M j ≈ H j z j x p z j y p , based on the observation that z j x p z j ′ y p ≪ z j x p z j y p when j = j ′ . In ♡ , we utilize Eq (38). There are at most O ( d γc 0 ) neurons capable of learning M j , which satisfy the condition 1 ⟨ w i ,x p ⟩≥ b .

For i ∈ S j, sure and for x p contain M j , using Eq (128), Eq (119) and Eq (126) we have:

<!-- formula-not-decoded -->

Similar to Eq (35), we have

<!-- formula-not-decoded -->

Similarly, for i ∈ S j, sure and x p contain the corresponding spurious aligned feature M j ′ , because Pr[ 1 ∣ ∣ ∣ 〈 w ( t ) i ,x p 〉∣ ∣ ∣ ≥ b ( t ) i = 0] ≥ 1 -O ( 1 √ d ) , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

At T C = Θ ( d log( d ) η ) , we have:

<!-- formula-not-decoded -->

Therefore, we summarize that when t = T C , the alignment with the true feature M j dominates, satisfying:

<!-- formula-not-decoded -->

highlighting the significant separation between the true feature M j and the spurious feature M j ′ for neurons i ∈ S j, sure. A similar result holds for v i , where the alignment with the true feature H j similarly dominates over the spurious feature H j ′ .

## G.2 Convergence

For i ∈ S j, sure, when x p , y p contains the true feature j , the indicator functions remain consistently activated. Consequently, the loss function L C becomes convex with respect to w i and v i independently. We verify that the following inequality holds

<!-- formula-not-decoded -->

where l i = O ( C z d 2 γc 0 )( ∥ ∥ ∥ v ( t ) i ∥ ∥ ∥ 2 2 ∥ x p ∥ 2 2 + ∥ ∥ ∥ v ( t ) i ∥ ∥ ∥ 2 2 ∥ x p ∥ 2 2 ) = Θ(1) . This means L C,j ( w ( t ) i , v ( t ) i ) is l i -smooth for all i ∈ S j, sure when x p , y p contains the true feature j . Let L = max i ∈ m ( l i ) = Θ(1) Let η = 1 L to ensure a monotonic decrease, plug Eq (117) and Eq (118) into Eq (135), we have

<!-- formula-not-decoded -->

By the property of smoothness, we have

<!-- formula-not-decoded -->

Take the telescope sum of from 0 to T C , we have

<!-- formula-not-decoded -->

where ∆ 0 = L C,j ( w (0) i , v (0) i ) -L C,j ( w ∗ i , v ∗ i ) . In ♢ , we use T C = Θ( d ) , and ∥ w ( t ) i ∥ 2 2 = ∥ v ( t ) i ∥ 2 2 = Θ(1) . In ♡ , we use w ∗ i = α ∗ i,j M j , V ∗ i = α ∗ i,j H j and L C,j ( w ∗ i , v ∗ i ) = Θ( 1 d ) if x p contains the true feature M j .

Therefore, for all j ∈ d and all ( x p , y p ) ∈ S h , when T C = Θ( d 2 ) , we can ensure

<!-- formula-not-decoded -->

## G.3 Summary

After T C iterations, the parameters W and V are updated to W T C = ˆ W and V T C = ˆ V , respectively, using the dataset S h . The generated caption is given by:

<!-- formula-not-decoded -->

where the expected loss satisfies:

<!-- formula-not-decoded -->

1. For i ∈ S j, sure, the alignment strength satisfies:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and where j ′ represents the corresponding spurious alignment feature.

2. For i / ∈ S j, pot, the alignment strength satisfies:

<!-- formula-not-decoded -->

3. For M ⊥ j where j ∈ [ d 1 ] \ [ d ] , we have:

<!-- formula-not-decoded -->

## H Filtering

During filtering, we sample the synthetic image-text pair ( x p , ˆ y p ) in ˆ S w and the corresponding image-text pair ( x p , y p ) in S w . The image encoder f and text encoder h trained on raw data are employed to obtain the corresponding embeddings.

<!-- formula-not-decoded -->

Then, we calculate the cosine similarity of ⟨ z ′ x p , ˆ z y p ⟩ and ⟨ z ′ x p , z ′ y p ⟩ , and select the image-text pair with higher cosine similarity denoted as ( x, ˜ y ) . In this way, we replace the noisy pairs in S w with synthetic pairs in ˆ S w . The resulting dataset is denoted as ˜ S = ˜ S w ∪ S h .

The decoder generates synthetic captions ˆ y p = ˆ V T σ ( ˆ W x p ) . Using Eq (141), for each data pair ( x p , y p ) which contain feature ( M j , H j ) in S h we have

<!-- formula-not-decoded -->

Therefore, using ∥ H j ∥ 2 = 1 and z x p = z y p in S h , we have

<!-- formula-not-decoded -->

Base on Assumption B.1 z j x p ∼ Bernoulli ( C z d ) , we have

<!-- formula-not-decoded -->

Using Eq (134) and Eq (149), we have

<!-- formula-not-decoded -->

Therefore, after replace all noisy text y p in S w by synthetic caption ˆ y p in ˆ S w

1. for a positive pair ( x p , y p ) , we have

<!-- formula-not-decoded -->

̸

2. for negative pairs ( x p , y q ) , where p = q , we have:

<!-- formula-not-decoded -->

̸

## I ITCP on Synthetic (Recaptioned) Data

During ITCP on Raw Data, we use a noisy dataset S . Based on SubSection B.2, we have E [ z j x z j y ] and E [ z j x z j ′ y ] both in Θ ( 1 d ) . In this scenario, for i ∈ S j, sure , w ( t ) i is jointly influenced by M j and M j ′ , with both features contributing comparably to the updates. However, during ITCP on recaptioned data, we sample image-text pairs from the dataset ˜ S . Using Eq. (151), we find that E [ z j ˜ x p z j ′ ˜ y p ] = Θ ( 1 d 2 ) . In this case, for i ∈ S j, sure , w ( t ) i is influenced solely by M j , without interference from spurious features, ensuring purified representations.

The only difference between ITCP on Raw Data and Data lies in the E [ z j ˜ x p z j ′ ˜ y p ] ; all other training processes remain largely the same. Therefore, we simplify our proof accordingly.

## I.1 Phase I of ITCP on Synthetic Data

The Phase I of ITCP on Data is defined as the training iterations t ≤ T 1 , where T 1 = Θ ( d log d η ) is the iteration when all ∥ w ( T 2 ) i ∥ 2 2 = 2 ∥ w (0) i ∥ 2 2 . Before T 1 , we set b ( t ) i = 0 . For every neuron i ∈ [ m ] , the weights w i , v i will mostly ignore the noise features M ⊥ , H ⊥ and learn to emphasize the features M , H .

<!-- formula-not-decoded -->

In this case, w ( t +1) i is predominantly influenced by M j , with minimal contributions from M j ′ . The updates are thus primarily driven by the single feature M j , ensuring that spurious interactions from M j ′ are negligible.

<!-- formula-not-decoded -->

i ∈ S j, sure :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

i / ∈ S j, sure :

<!-- formula-not-decoded -->

|⟨ w ( t +1) i , M ⊥ j ⟩| 2 ≤ O ( 1 d 1 ) ∥ w ( T 1 ) i ∥ 2 2 + ∥ v ( T 1 ) i ∥ 2 2 2

## I.2 Phase II:

The Phase II of ITCP on Synthetic Data is defined as the training iterations T 1 ≤ t ≤ T 2 , where T 2 -T 1 = Θ ( d log d η ) is the iteration.

We set b ( t ) i = √ log d d · ∥ w ( T 1 ) i ∥ 2 2 + ∥ v ( T 1 ) i ∥ 2 2 2 and b ( t +1) i = (1 + η d ) b ( t ) i until all ∥∥ w ( T 2 ) i ∥ 2 ≥ Ω( d ) ∥ w ( T 1 ) i ∥ 2 , . In this phase, the weights ( w i , v i ) will mostly ignore the features M j , H j if i / ∈ S j, sure and the noise features M ⊥ , H ⊥ , and learn to emphasize the features M j , H j if i ∈ S j, sure . For i ∈ S j, sure, using Lemma C.4, the following holds with high probability 1 -e -Ω( d 1 ) when T 1 &lt; t ≤ T 2 :

<!-- formula-not-decoded -->

Under the assumption that, with high probability, the indicator function satisfies the condition when t = T 1 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The weight dynamics for |⟨ w ( t +1) i , M j ⟩| can be expressed as:

<!-- formula-not-decoded -->

Given that ( 1 + η C z d ) &gt; ( 1 + η d ) , and ⟨ w ( t ) i , M j ⟩ + ⟨ v ( t ) i , H j ⟩ 2 &gt; b ( t ) i , it follows that:

<!-- formula-not-decoded -->

Thus, with high probability, for t ≤ T 2 , we have:

<!-- formula-not-decoded -->

we can ensure that:

so for T 1 &lt; t ≤ T 2 we have

<!-- formula-not-decoded -->

For i / ∈ S j, sure, the projection of weights onto a generic feature ξ at iteration T 1 satisfies:

<!-- formula-not-decoded -->

We can ensure that:

<!-- formula-not-decoded -->

The weight dynamics for |⟨ w ( t +1) i , M j ⟩| can now be expressed as:

<!-- formula-not-decoded -->

Given that ( 1 + o ( η d 2 )) &lt; ( 1 + η d ) , and ⟨ w ( t ) i , M j ⟩ + ⟨ v ( t ) i , H j ⟩ 2 &lt; b ( t ) i , it follows that:

|⟨

w

(

i

,

M

j

⟩|

&lt; b

(

i

.

(166)

If |⟨ w ( T 1 ) i , M j ⟩| &lt; b ( T 1 ) i , then |⟨ w ( t ) i , M j ⟩| &lt; b ( t ) i for t ≤ T 2 . Thus, with high probability, for t ≤ T 2 , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

There exists T 2 = Θ ( d log d η ) such that the following conditions hold:

<!-- formula-not-decoded -->

indicating that |⟨ w ( t +1) i , M j ⟩| for i ∈ S j, sure increase iteratively until:

<!-- formula-not-decoded -->

while, for i / ∈ S j, sure, the updates diminish, such that:

<!-- formula-not-decoded -->

indicating negligible growth in |⟨ w ( t +1) i , M j ⟩| . Thus we have

<!-- formula-not-decoded -->

Finally, for i / ∈ S j, sure, we have:

<!-- formula-not-decoded -->

and for noise components:

t

+1)

t

+1)

<!-- formula-not-decoded -->

We summarize the results when T 1 &lt; t ≤ T 2 as follows:

1. For i ∈ S j, sure, the alignment strength satisfies:

<!-- formula-not-decoded -->

without j ′ that represents the corresponding spurious alignment feature.

2. For i / ∈ S j, pot, the alignment strength satisfies:

<!-- formula-not-decoded -->

3. For M ⊥ j where j ∈ [ d 1 ] \ [ d ] , we have:

<!-- formula-not-decoded -->

Similar results also hold for v i .

## I.3 Phase III Convergence of ITCP on Synthetic Data

Similarly to convergence Phase III in ITCP on Raw Data when T 2 ≤ t ≤ T 3 , using Eq (27), Eq (30), and Eq (31), the loss function L becomes convex with respect to w i and v i independently when ( x p , y p ) and ( x n , y n ) contain the true feature j .

We verify that the following inequality holds

<!-- formula-not-decoded -->

Let L = max i ∈ m ( l i,j / (2 τ )) = Θ(1) and η = 1 L to ensure a monotonic decrease, plug Eq (28) and Eq (29) into Eq (178), we have

<!-- formula-not-decoded -->

Under our data assumptions for S w and conclusion in Eq (96) , we define w ∗ i = α ∗ i,j M j , v ∗ i = α ∗ i,j H j . Thus, L j ( w ∗ i , v ∗ i ) captures both the alignment with the true feature M j , H j and the spurious feature M j ′ , H j ′ , representing the minimal loss achievable under the influence of both true and spurious features in the optimization process. Using Eq (81) , we know w ( T 2 ) i = Θ( d ) , so L j ( w ∗ i , v ∗ i ) = -Θ( d ) .

By the property of smoothness, we have

<!-- formula-not-decoded -->

Take the telescope sum of from T 2 to T 3 , we have

<!-- formula-not-decoded -->

where ∆ 0 = L j ( w ( T 1 ) i , v ( T 1 ) i ) -L j ( w ∗ i , v ∗ i ) = Θ(1) . In ♢ , we use T 2 = Θ( d ) , and L = Θ( 1 d ) .

Generalized to every j ∈ d , the same convergence holds for all i ∈ S j, sure when ( x p , y p ) and ( x n , y n ) contain feature j, j ′ . For all ( x p , y p ) and ( x n , y n ) in S w , the following inequality holds:

<!-- formula-not-decoded -->

## I.4 Summary

ITCP trained on recaptioned data ˜ S proceeds according to Eq. (1). After T = Θ( d 2 log d ) SGD iterations with batch size B = Ω( d ) and learning rate η = O (1) , the returned weights ( ˜ W , ˜ V ) achieve a contrastive loss that is asymptotically optimal:

<!-- formula-not-decoded -->

Each neuron pair ( ˜ w i , ˜ v i ) in ( ˜ W , ˜ V ) , for i ∈ [ m ] , primarily encodes a single aligned feature indexed by a set ˜ N i ⊆ [ d ] , with | ˜ N i | = 1 . Specifically, we have:

<!-- formula-not-decoded -->

where ˜ α 2 i,j = Θ( ∥ ˜ w i ∥ 2 2 + ∥ ˜ v i ∥ 2 2 ) , and the residual terms satisfy ˜ β i,j / ˜ α i,j ≤ O (1 / √ d ) , ˜ γ i,j / ˜ α i,j ≤ O (1 / √ d 1 ) .

Moreover, for every feature index j ∈ [ d ] , there exists an Ω(1) many of neurons i ∈ [ m ] such that ˜ N i = { j } , indicating that each semantic concept is distinctly captured by dedicated neuron pairs.

## J Downstream Task

We consider the same zero-shot classification task as in Section 3.4, where the image x and the class-wise text prompts { y k } K k =1 are given. Each prompt y k corresponds to one of K class labels, and the goal is to classify x into the class with the best matching prompt.

Each text prompt y k is generated as:

<!-- formula-not-decoded -->

Each test image x is generated as:

<!-- formula-not-decoded -->

where M ′ = MP 1 , and

<!-- formula-not-decoded -->

If x belongs to class k , then:

<!-- formula-not-decoded -->

̸

Using Eq. (96) and Eq. (144), let f ( x ) and h ( y ) represent the image encoder and text encoder of ITCP on raw data, respectively. Given a data sample x containing M j and y containing H j ′ , where j ′ is the spurious feature corresponding to j , it holds with high probability that:

<!-- formula-not-decoded -->

This result implies that the image and text encoders of ITCP on raw data struggle to distinguish between features j and j ′ , leading to misclassification caused by spurious correlations.

However, using Eq. (175) and Eq. (176), let ˜ f ( x ) and ˜ g ( y k ) denote the image and text encoders of ITCP on recaptioned data. Given x containing M j and y containing spurious H j ′ , it holds with high probability 1 -Θ ( 1 d ) that:

<!-- formula-not-decoded -->

This result implies that the image and text encoders of ITCP on synthetic data are capable of effectively distinguishing the true feature from the spurious feature.

Because K = Θ(1) and ∥ z y k ∥ 0 = Θ(1) , we only have constant class classification and constant features in images. Thus, we have:

1. For the image encoder f ( x ) and text encoder h ( y k ) of ITCP on raw data:

<!-- formula-not-decoded -->

2. For the image encoder ˜ f ( x ) and text encoder ˜ g ( y k ) of ITCP on synthetic data:

<!-- formula-not-decoded -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The claims regarding the improvement of feature purity and zero-shot performance through recaption are stated clearly in the abstract and introduction, and validated by theoretical and empirical analysis.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations in the Conclusion.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: All theoretical results are stated with full assumptions, and complete proofs are provided in the Appendix. Intuitive proof sketches are also included to aid understanding.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We outline all datasets, models, and training procedures in Section 5.1.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We will provide public GitHub access post-review.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We provide the details of data generation, dimensionality, noise setup, model size, and SGD training parameters including batch size and learning rate in Section 5.1.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Each reported metric (accuracy, Silhouette Score, feature purity, random selection of neurons) is averaged over 20 random seeds with 1-sigma standard deviation in Section 5.1.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: The experimental resources are introduced in Appendix A.2.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We comply with all ethical guidelines. All data used is publicly available under appropriate licenses; no human subjects or sensitive data are involved.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: There is no societal impact of the work performed.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: Our work does not involve generative models or web-scraped datasets with safety concerns.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We cite and respect the licenses of models and datasets.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: This paper does not release new assets.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: No human subject research is conducted.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research?

Answer: [NA]

Justification: No LLMs were used in this work.