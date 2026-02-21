## Theory-Driven Label-Specific Representation for Incomplete Multi-View Multi-Label Learning

Quanjiang Li 1 †

liquanjiang@nudt.edu.cn

Tianxiang Xu 2

†

xtx09@hotmail.com

Tingjin Luo 1

∗

tingjinluo@hotmail.com

Yan Zhong 2

zhongyan@stu.pku.edu.cn

Yang Li 1

liyang\_albert@foxmail.com

Yiyun Zhou 3

yiyunzhou@zju.edu.cn

## Chenping Hou 1

hcpnudt@hotmail.com

1 National University of Defense Technology, 2 Peking University, 3 Zhejiang University

## Abstract

Multi-view multi-label learning typically suffers from dual data incompleteness due to limitations in feature storage and annotation costs. The interplay of heterogeneous features, numerous labels, and missing information significantly degrades model performance. To tackle the complex yet highly practical challenges, we propose a Theory-Driven Label-Specific Representation (TDLSR) framework. Through constructing the view-specific sample topology and prototype association graph, we develop the proximity-aware imputation mechanism, while deriving class representatives that capture the label correlation semantics. To obtain semantically distinct view representations, we introduce principles of information shift, interaction and orthogonality, which promotes the disentanglement of representation information, and mitigates message distortion and redundancy. Besides, labelsemantic-guided feature learning is employed to identify the discriminative shared and specific representations and refine the label preference across views. Moreover, we theoretically investigate the characteristics of representation learning and the generalization performance. Finally, extensive experiments on public datasets and real-world applications validate the effectiveness of TDLSR.

## 1 Introduction

The popularity of multi-view learning stems from its ability to provide comprehensive representations of samples [29, 9]. Integrating multi-source information effectively reveals latent semantic associations in multimodal data, thereby enriching the feature space and improving model accuracy and generalization [10]. With the advancement of information technology, single-label proves inadequate in meeting the object labeling and recognition demands [3]. In practice, objects frequently fall under multiple categories simultaneously, with a document in text classification [40] being assigned labels like topic, intended audience, sentiment and so on. Benefiting from the holistic characterization offered by multi-view approaches [41] and the capacity of multi-label methods [31] to capture all sample attributes, diverse views and labels are emerging as the dominant form of training data. Consequently, multi-view multi-label classification (MvMLC) has attracted significant research attention, as it delivers a refined understanding of the complexity and diversity inherent in real-world situations.

† Equal contribution

∗ Corresponding Author

Existing MvMLC methods are capable of implementing both multivariate feature fusion and multiobjective joint discrimination. For example, LVSL [43] performed view-specific label learning and leveraged low-rank label structures to enhance performance. E F 2 FS [12] proposed an embedded feature selection model that integrated feature aggregation and enhancement. However, these ideal methods assume the presence of both features and labels, whereas limitations in feature collection techniques and annotation complexity result in the unavailability of partial views and tags [37], motivating researchers to explore the incomplete multi-view multi-label classification (iMvMLC) approach. Early efforts like iMVWL [33] employed a joint learning strategy to refine the shared subspace and enhance the robustness of the weak label classifier. NAIM3L [21] combined matrix factorization with global high-rank and local low-rank constraints to obtain a shared label space. Given the ability to capture complex semantic information, deep learning based methods have demonstrated promising performance. The pioneering work DICNet [25] introduced an incomplete instance-level contrastive learning method for improved consensus representation, while LMVCAT [26] designed two transformer-based modules for cross-view feature aggregation and category awareness.

Despite the emergence of various effective iMvMLC methods, they remain deficient in feature information reconstruction, semantic distinctness of extracted representations and the coupling between label correlation semantics with feature connotation. (i) The absence of views severely undermines both the capture of cross-view dependencies and the stability of downstream modules. Approaches [4] that only mask missing samples tend to circumvent the challenge, rather than engaging in robust information recovery. AIMNet [23] attempted to perform missing imputation via cross-view global attention computation. However, the integration of a global weighting scheme may further exacerbate reconstruction noise since not all samples are strongly correlated. (ii) The essence of multi-view learning lies in extracting the complementarity and consistency across diverse views [36]. Additionally, increased attention should be devoted to feature separation in multi-label scenarios [42], as distinct labels exhibit varying sensitivities towards particular features. DIMC [38], TSIEN [34] and SIP [27] primarily focused on shared subspaces while overlooking the unique characteristics of individual views. MTD [24] attempted to obtain shared and specific representations through geometric distance constraints. However, this linear interaction pattern struggled to reflect the intricate interview relationships. (iii) Label relevance is fundamental to multi-label learning and distinguishes it from multi-class problem [44], which makes multiple independent binary classifications insufficient as done in some methods [24]. Besides, instead of being considered in isolation, mutually dependent label semantics should interact seamlessly with feature information to facilitate both the perception of label-specific features and the selection of the preferred labels corresponding to those features.

To tackle these issues, we propose a Theory-Driven Label-Specific Representation framework named TDLSR. The motivation behind TDLSR is to minimize view reconstruction errors, improve the semantic discriminability of extracted representations and strengthen the interaction between label-relevant semantics and feature information. We first construct view-specific instance relation graphs using the attention mechanism integrated with neighborhood-aware selection. Without any discrepancy arising from network parameter updates, the reconstruction risk is minimized by propagating highly relevant sample information across views. By introducing the principles of information shift, interaction and orthogonality, we develop a mutual information optimization model that aligns the core constituents of representations with their respective views, facilitates interaction among shared representations, suppresses cross-talk between private components and enforces orthogonality of shared and specific information from the same view. Label correlation information is transmitted via graph-induced relational network modeling, which results in the emergence of interdependent category representatives. Through discrete engagements between each class prototype and semantically distinct features, the most sensitive feature ensemble for each label across both shared and specific feature pools can be identified. We further establish the generalization error bound via error decomposition, showing that feature disentanglement maximizes the mutual information between representations and objects and reduces generalization error. The main contributions of our work are summarized as follows:

- We propose a general multi-view representation extraction model inspired by information theory. This model guarantees unbiased representations through the constraint of information shift, separates shared and specific semantic via the regulation of information interaction, and eliminates representation redundancy by strengthening information orthogonality.
- TDLSR enhances the propagation of feature dependencies and label correlation semantics by constructing relational graphs. Besides, it introduces label-specific shared and private feature

learning for the first time. Theoretically, we prove the discriminability and effectiveness of feature extraction and derive the generalization error bound.

- Extensive experimental results across diverse public available datasets, along with applications on real-world NBA data, validate the effectiveness and robustness of our method.

## 2 Method

In this section, we present the following critical components of our TDLSR as shown in Fig. 1: proximity-aware graph attention recovery mechanism, universal view extraction framework under mutual information constraints and multi-label semantic and label-specific representation learning.

## 2.1 Problem definition

We define { X ( v ) } V v =1 as original multi-view setting, where X ( v ) = { x ( v ) i } N i =1 ∈ R N × d v represents d v dimensional feature matrix of the v -th view. The label matrix Y ∈ { 0 , 1 } N × C corresponds to C categories, with Y i,j = 1 if sample i is tagged as class j . Besides, we let W ∈ { 0 , 1 } N × V and G ∈ { 0 , 1 } N × C denote the missing indicator for views and labels, respectively. Specifically, W i,j = 1 if the j -th view of the i -th sample is available, otherwise W i,j = 0 . Similarly, G i,j = 1 or 0 reflects the certainty of the corresponding label. Our goal is to train an end-to-end neural network capable of performing classification inference on incomplete multi-view weak multi-label data.

Figure 1: The main framework of our proposed TDLSR. Different shapes signify different samples.

<!-- image -->

## 2.2 Proximity-aware Graph Attention Recovery Mechanism

The lack of diverse views limits the capacity of deep neural models to extract high-level representations [20]. Consequently, data augmentation through reconstruction techniques is essential for improving model accuracy and stability. To mitigate information deficiency arising from incompleteness, we propose an attention-based relational graph construction strategy to propagate similarity signals across samples for missing imputation. For any view v , the attention score for an instance pair is computed by B ( v ) i,j = e h ( x ( v ) i ) h ( x ( v ) T j ) /τ , where τ is the temperature parameter and h ( · ) denotes the normalization function. Each row of the matrix B ( v ) quantifies the degree of similarity between the corresponding sample with all other samples [23], which facilitates the identification of the k -nearest neighbors for each instance. Therefore, we construct the view-specific graph ˆ S ( v ) ∈ R N × N through attention-induced proximity awareness, where ˆ S ( v ) i,j = 1 means W i,v W j,v = 1 and B ( v ) i,j is one of the topk largest elements in the i -th row, i.e., x ( v ) j is the neighbor of x ( v ) i . Considering similarity relations between instances in existing views are applicable to the missing views, the transferred graph for finding the available instances related to the missing ones can be obtained:

̸

<!-- formula-not-decoded -->

Figure 2: Depiction of information shift, interaction, and orthogonality.

<!-- image -->

where diag( · ) creates a diagonal matrix and W : ,v is the v -th column. Besides, the contribution of available samples to reconstruct missing ones is governed by the maximal computable correlation from alternative views:

<!-- formula-not-decoded -->

where ¯ B i,j encapsulates the influence of the j -th sample in the recovery process of the i -th sample. By treating K as the adjacency matrix and ¯ B as the edge weight, we can get the reconstructed data after message propagation:

<!-- formula-not-decoded -->

Since ˆ X ( v ) serves as an approximate proxy of missing instances, we combine it with the original view to generate the final recovery matrix for downstream tasks:

<!-- formula-not-decoded -->

## 2.3 Universal View Extraction Framework under Mutual Information Constraints

̸

Previous studies [27] have primarily concentrated on enabling networks to extract shared information, while overlooking the contributions of each unique view. Moreover, the reliance on linear geometric constraints in early view-specific representation learning [24] falls short of capturing complex feature interactions. To effectively assess both the consistency and the complementarity of distinct features, we propose a general model that directly constrains the mutual information between representations for precisely measuring their interaction degree. Two groups of multi-layer perceptrons { M S v } V v =1 and { M O v } V v =1 are employed as shared and view-private channel to extract the corresponding representations, i.e., { M S v : Z ( v ) → S ( v ) } V v =1 and { M O v : Z ( v ) → O ( v ) } V v =1 . Taking the image-text retrieval task [32] for explanation, as illustrated in the Fig. 2, an image is typically paired with multiple textual descriptions that emphasize on individual relationships and environmental context, respectively. It requires the model to simultaneously extract the consistent shared representations (central actions) and unique modality-specific elements (e.g., distinct individual identities and background details) for holistic multimodal perception. To achieve this, maximizing I ( s ( v ) ; z ( v ) ) is critical for guaranteeing that the representations stay consistent with the core semantics of raw modalities, such as people, kit, background, etc. Besides, comprehensive feature disentanglement is facilitated by maximizing the mutual information I ( s ( v ∗ ) | z ( v ∗ ) ; s ( v ) | z ( v ) ) and minimizing I ( o ( v ∗ ) | z ( v ∗ ) ; o ( v ) | z ( v ) ) between representations from any pair of views v and v ∗ (1 ≤ v = v ∗ ≤ V ) , which focuses on capturing the central characteristics of the actions and preserving unique information embedded in each modality, respectively. Furthermore, establishing complete representation orthogonality from the same modality, as indicated by I ( s ( v ) | z ( v ) ; o ( v ) | z ( v ) ) = 0 , effectively minimizes semantic redundancy, such as the identification of the individual and their actions.

As mentioned above, feature separation is driven by the following three criteria: (i) prevent information shift, (ii) optimize information interaction and (iii) promote information orthogonality. Under

the condition of mutual information constraints, the universal model can be expressed as

̸

<!-- formula-not-decoded -->

By employing the Lagrange multiplier method, the equality constraint can be appropriately scaled. Direct computation of mutual information is impractical due to the challenges of distribution inference and the complexity of high-dimensional integrals. Thus, stability and feasibility are typically ensured by refining estimable mutual information bounds [35]. Regarding the information shift term, its lower bound is commonly represented through a reconstruction loss[15], where the representation s ( v ) is decoded via q ( z ( v ) | s ( v ) ) to faithfully preserve the original view:

<!-- formula-not-decoded -->

For the second term, based on the definition expansion and the non-negativity of entropy, we can derive the following lower bound by introducing another variational distribution q ( s ( v ) | s v ∗ ) :

<!-- formula-not-decoded -->

where p ( s ( v ) | z ( v ) / s ( v ∗ ) | z ( v ∗ ) ) refers to the conditional distribution between the shared representations from z ( v ) and z ( v ∗ ) . Leveraging the non-negativity of Kullback-Leibler divergence between the distributions p ( s ( v ) | z ( v ) / s ( v ∗ ) | z ( v ∗ ) ) and q ( s ( v ) | s ( v ∗ ) ) , the tighter lower bound is further obtained. Similarly, upper bounds for the remaining two negative terms can be derived following the same rationale. Therefore, the problem (16) is transformed into optimizing the following objective:

<!-- formula-not-decoded -->

where β is the Lagrange multiplier. Minimizing L IB promotes effective distinction between shared and specific feature semantic, with implementation details provided in the Appendix.

## 2.4 Multi-Label Semantic and Label-Specific Representation Learning

In multi-label classification tasks, binary encoding struggles to capture the underlying label semantics [19]. To overcome this limitation, we employ a data-driven approach to learn label prototypes [27], which provides a clear insight into label structures and enhances effective semantic associations with features. After initializing the one-hot vector b i ∈ R C for each category, we utilize two stochastic encoders to model the prototype distribution i.e., h i ∼ N ( µ i , σ 2 i I ) , where the mean µ i and variance σ i are determined by the encoders g µ ( b i ) and g σ 2 ( b i ) . Then, we sample from the distribution with the reparameterization trick to obtain the prototype representation h 0 i = 1 s ∑ s d =1 ( µ i + σ i ⊙ δ d ) [7], where δ d means the d -th individual sampling and ⊙ denotes the element-wise product.

The prototype representations are independently learned for each label, yet investigating intrinsic label correlation remains a core challenge in multi-label learning [3]. For bridging this gap, graph neural networks are leveraged to propagate prior correlation information and refine the label representations to encapsulate the inherent relationships of label semantics. Prototypes are regraded as a set of nodes positioned on the label relation graph, with edge weights reflecting the correlation between corresponding label pairs. Besides, the label correlation matrix A quantitatively characterized on the training data serves as the appropriate substitute for the adjacency matrix. Rather than utilizing co-occurrence frequency to evaluate correlation degree, we use the Jaccard distance [30] calculated over positive labels, as we are only concerned with the categories assigned to each instance. By computing the intersection and union of two classes regarding positive values, we can obtain

<!-- formula-not-decoded -->

where A ii is set to 0 to eliminate self-dependency. Given an aggregated matrix H ∈ R C × d , the label embeddings corresponding to each row can be updated by passing through the GIN layer with propagated correlation information [11], i.e., E = f [(1 + ϵ ) H + AH ] , where f ( · ) denotes a fully-connected layer followed by Batch Normalization and Leaky ReLU activation, and ϵ is a learnable scalar that controls the influence of node's own features. To reinforce cohesion between relevant prototype representations and distinguish unrelated ones, we employ the following objective that aligns representation similarity with label correlation:

<!-- formula-not-decoded -->

where cos ( E i , E j ) is the cosine similarity and ˆ A = A + I with I denoting an identity matrix. Multilabel classification tasks often involve labels with varying sensitivities to different feature subsets [14]. Consequently, label-specific feature learning has become a widely adopted technique to select the most relevant features tailored to classifying each label. However, label-specific disentangled feature learning remains underexplored, despite its potential to boost model performance. For instance, in image recognition, shared information capture general visual cues, while private features highlight other distinctive traits tied to each label, such as breed for "dog" and texture for "cat". Thus, we treat activated label prototypes as feature importance scores and engage them with both shared and private representations to discern the label-specific view embeddings:

<!-- formula-not-decoded -->

where σ S is the Sigmoid function. According to the Eq. (11), we can obtain the label-specific shared and private features, i.e., { S ( v ) → ˆ U ( v ) ∈ R N × C × d } V v =1 and { O ( v ) → ˆ V ( v ) ∈ R N × C × d } V v =1 . The interaction between label semantics and view representations supports a bidirectional selection mechanism, in which discriminative views are assigned to specific labels, while information-related label subsets are uncovered associated with distinct views. Processing through the linear classifiers, view-specific predictions U ( v ) ∈ R N × C and V ( v ) ∈ R N × C are generated. Given the variability in reconstruction quality, views with higher recovery accuracy should be emphasized in the fusion process. For this aspect, certain reconstructed views characterized by low attention scores in relation to their associated samples naturally exhibit reduced confidence [23]. Thus, we calculate the maximum original attention for each instance with respect to other instances as its confidence score:

<!-- formula-not-decoded -->

where ¯ B is computed by Eq. (2) and Q ∈ R N × V stores the confidence score for individual instances. Since Q is tailor-made for the missing samples, the confidence matrix is updated as Q ′ = (1 -W ) ⊙ Q + W . During the late fusion, we combine the feature reconstruction efficiency to obtain the final prediction:

<!-- formula-not-decoded -->

Then, we employ the weighted cross-entropy loss to mitigate the impact of missing labels:

<!-- formula-not-decoded -->

By incorporating λ 1 and λ 2 to balance the loss effects, our total training loss can be expressed as

<!-- formula-not-decoded -->

## 2.5 Theoretical Results

In this subsection, we aim to theoretically explore the fundamental mechanisms that contribute to the model performance and the generalization capability of TDLSR. Through rigorous derivations (proofs in the Appendix), we obtain the following theorems:

̸

Theorem 1. ( Discriminability of Label-specific Representation. ) For label prototypes E j and E k such that k = j for all k , and view representations X ( v ) and X ( v ∗ ) such that v ∗ = v for all v ∗ , the discriminability of ˆ P ( v ) for class j necessitates that either of the following conditions be satisfied:

̸

<!-- formula-not-decoded -->

Theorem 2. ( Effectiveness of Disentangled Representation. ) Let the disentangled representation be denoted as R = ( S (1) , . . . , S ( V ) , O (1) , . . . , O ( V ) ) , where the information entropy of each representation is assumed to be fixed, i.e., H ( S ( v ) ) = H ( O ( v ) ) = H 0 (1 ≤ v ≤ V ) . Then, in the case where each shared and specific representation is indispensable for prediction, I ( R ; Y ) will attain its maximum when R = R ∗ , with R ∗ being the optimal solution of the problem (16).

Theorem 3. ( Generalization Error Bound. ) Our model is designed to learn a vector-valued function f = ( f 1 , . . . , f C ) : X ↦→ R C . The expected risk and empirical risk w.r.t. the training dataset D are denoted as R ( f ) = E ( X , Y ) ∼X×Y [ ℓ ( f av ( X , Q ) , Y )] and ̂ R D ( f ) = 1 NC ∑ N i =1 ∑ C c =1 ℓ ( ∑ V v =1 ( Q i,v f c ( x ( v ) i )) , Y i,c ) , where f av ( · ) refers to the late fusion of multiple views. With probability at least 1 -δ , we have the following generalization error bound:

<!-- formula-not-decoded -->

where ˜ K 1 = ˜ K 3 = O ( C ) , ˜ K 2 = O ( √ C ) , ˜ K 4 is constant of order ˜ O (1) as N,V → ∞ , and gen rec ( Q , X , Y ) is the generalization error related to the view reconstruction quality. Moreover, the generalization error bound becomes increasingly tighter during the optimization of the problem (16).

## 3 Experiments

## 3.1 Datasets and metrics

In our experiments, we utilize six popular multi-view multi-label datasets to validate the performance of our TDLSR, i.e., Corel 5k [5], ESPGame [1], IAPRTC12 [8], Mirflickr [16], Pascal07 [6], OBJECT [13]. In accordance with [25, 38], we select six metrics to construct a comprehensive evaluation system, i.e., Hamming Loss (HL), Ranking Loss (RL), OneError (OE), Coverage (Cov), Average Precision (AP), and Area Under Curve (AUC). To facilitate comparison, we present 1-HL, 1-OE, 1-Cov, and 1-RL values in the report, where higher values correspond to better performance.

## 3.2 Comparison methods

To measure the advancement of our TDLSR, nine state-of-the-art methods are selected for comparison experiments, i.e., AIMNet [23], DICNet [25], DIMC [38], iMVWL [33], LMVCAT [26], MTD [24], SIP [27], LVSL [43], DM2L [28]. Specifically, the first seven methods can simultaneously address the issues of missing views and labels. LVSL is a MvMLC method unable to handle missing data.

Thus, we use the mean of available instances to complete the missing views and fill the unknown labels with "0". DM2L is a kernel-based nonlinear method for incomplete multi-label learning. Then, we concatenate all views into a single-view representation for the execution of DM2L. All parameters of compared methods are configured as the recommended values in their original codes.

## 3.3 Implementation details

Each dataset is divided into training, validation and test sets in the ratio of 7:1:2. To simulate the partial view setting, a specified proportion of instances based on the Partial Example Ratio (PER), are randomly marked as unavailable in each view. Additionally, we ensure that each sample contains at least one complete view to avoid invalid cases. For weak supervision, we introduce label omissions for both positive and negative tags in each category applying the same proportion determined by the Label Missing Ratio (LMR). The process of constructing incomplete data is repeated multiple times to mitigate the impact of experimental randomness. Our model is implemented by PyTorch on one NVIDIA GeForce RTX 4090 GPU of 24GB memory.

## 3.4 Experimental results and analysis

Figure 3: Experimental results on four datasets with PER and LMR changing from 30% to 90% .

<!-- image -->

To evaluate the effectiveness of our TDLSR in handling absent views and labels, we benchmark it against nine closely related algorithms across six datasets with varying levels of data sparsity. The proportion of missing views (PER) and labels (LMR) encompasses values of { 30% , 50% , 70% , 90% } . The mean and standard deviation of the results with PER and LMR fixed at 50% are reported in Table 1. Besides, the average ranking of each algorithm based on the six metrics is calculated to perform a thorough assessment. Fig. 3 illustrates the variation in AP as PER and LMR changes from 30% to 90% . The other relevant results will be presented in the Appendix.

Drawing from the comparison results, we have the following observations: (i) Our method exhibits outstanding performance on almost all metrics across all six datasets. As shown in Table 1, despite the fluctuating rankings of other methods, TDLSR consistently holds the top position. Therefore, our method effectively addresses the iMvMLC problem and maintains stable outcomes. (ii) SIP and MTD are top-performing methods that always appear among the top three. The reason our method surpasses these leading approaches lies in the mutual information optimization, which constrains complex interactions between representations that are insufficiently addressed by MTD. It also overcomes the limitations of SIP by accounting for the impact of private features and transmitting label correlation information to refine label prototypes. Compared to AIMNet that similarly engages in view recovery, our method achieves an improvement in AP from 0.40 to 0.45 on Corel 5k, which demonstrates that our proximity-aware strategy can greatly suppress reconstruction noise. Achieving over a 10% performance gain against traditional multi-label methods like DM2L and deep learning frameworks such as DIMC and DICNet that disregard label correlation, our approach highlights its strength in capturing high-level view representations and leveraging label dependencies to enhance overall performance. (iii) As depicted in Fig. 3, our method exhibits remarkable performance and strong robustness across a wide range of missing ratios. Moreover, our method is particularly well-suited for highly incomplete settings. For instance, when PER reaches 90% on Corel5k and ESPGame, all baseline methods collapse, while our approach continues to deliver commendable results.

Table 1: Experimental results of nine methods on the six datasets with 50% PER and 50% LMR. 'Ave.R' refers to the mean ranking of the corresponding method across all six metrics.

| DATA   | METRIC                          | AIMNet                                                                                    | DICNet                                                                                    | DIMC                                                                                      | DM2L                                                                                      | iMVWL                                                                                     | LMVCAT                                                                                    | LVSL                                                                                      | MTD                                                                                       | SIP                                                                                       | TDLSR                                                                                           |
|--------|---------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| COR    | 1-HL 1-OE 1-Cov 1-RL AP AUC AVE | 0.988 0 . 000 0.478 0 . 011 0.766 0 . 004 0.900 0 . 002 0.404 0 . 005 0.903 0 . 002 3.5   | 0.987 0 . 000 0.460 0 . 012 0.726 0 . 007 0.881 0 . 004 0.381 0 . 006 0.883 0 . 004 5     | 0.987 0 . 000 0.446 0 . 009 0.709 0 . 008 0.874 0 . 004 0.370 0 . 005 0.877 0 . 004 7.333 | 0.987 0 . 000 0.378 0 . 014 0.640 0 . 007 0.843 0 . 004 0.318 0 . 005 0.846 0 . 004 9     | 0.978 0 . 000 0.308 0 . 017 0.701 0 . 003 0.864 0 . 002 0.281 0 . 005 0.867 0 . 002 9.5   | 0.986 0 . 000 0.448 0 . 011 0.720 0 . 006 0.876 0 . 004 0.379 0 . 006 0.879 0 . 003 6.833 | 0.987 0 . 000 0.353 0 . 017 0.720 0 . 005 0.879 0 . 003 0.311 0 . 005 0.882 0 . 002 7.333 | 0.988 0 . 000 0.492 0 . 011 0.754 0 . 005 0.893 0 . 004 0.410 0 . 007 0.895 0 . 003 3.167 | 0.988 0 . 000 0.492 0 . 014 0.780 0 . 004 0.908 0 . 003 0.414 0 . 006 0.910 0 . 002 2.333 | 0.988 0 . 000 0.541 0 . 014 0.801 0 . 009 0.917 0 . 004 0.450 0 . 006 0.919 0 . 004 1           |
| ESP    | 1-HL 1-OE 1-Cov 1-RL AP AUC AVE | 0.983 0 . 000 0.442 0 . 006 0.621 0 . 003 0.845 0 . 002 0.305 0 . 003 0.850 0 . 001 3.833 | 0.983 0 . 000 0.440 0 . 009 0.601 0 . 003 0.836 0 . 002 0.300 0 . 003 0.841 0 . 002 4.5   | 0.983 0 . 000 0.431 0 . 009 0.586 0 . 004 0.830 0 . 002 0.294 0 . 003 0.835 0 . 002 5.833 | 0.983 0 . 000 0.302 0 . 008 0.532 0 . 003 0.804 0 . 002 0.229 0 . 003 0.808 0 . 001 9.667 | 0.972 0 . 000 0.343 0 . 010 0.548 0 . 004 0.807 0 . 002 0.243 0 . 004 0.813 0 . 002 9.167 | 0.982 0 . 000 0.431 0 . 006 0.587 0 . 003 0.827 0 . 002 0.293 0 . 003 0.832 0 . 001 7.333 | 0.983 0 . 000 0.365 0 . 006 0.578 0 . 002 0.829 0 . 001 0.266 0 . 003 0.834 0 . 001 7.167 | 0.983 0 . 000 0.452 0 . 007 0.617 0 . 004 0.843 0 . 002 0.309 0 . 003 0.847 0 . 002 3.833 | 0.983 0 . 000 0.450 0 . 006 0.622 0 . 004 0.847 0 . 002 0.309 0 . 004 0.851 0 . 002 2.667 | 0.983 0 . 000 0.477 0 . 007 0.646 0 . 004 0.859 0 . 002 0.328 0 . 004 0.863 0 . 002 1           |
| IAP    | 1-HL 1-OE 1-Cov 1-RL AP AUC AVE | 0.981 0 . 000 0.457 0 . 007 0.675 0 . 004 0.884 0 . 001 0.329 0 . 003 0.885 0 . 001 4     | 0.981 0 . 000 0.464 0 . 008 0.649 0 . 005 0.874 0 . 002 0.326 0 . 003 0.876 0 . 002 4.333 | 0.981 0 . 000 0.454 0 . 006 0.630 0 . 005 0.868 0 . 002 0.318 0 . 002 0.870 0 . 001 6     | 0.980 0 . 000 0.378 0 . 008 0.555 0 . 005 0.837 0 . 002 0.254 0 . 002 0.838 0 . 001 8.833 | 0.969 0 . 000 0.351 0 . 008 0.565 0 . 004 0.833 0 . 002 0.236 0 . 002 0.835 0 . 001 9.833 | 0.980 0 . 000 0.433 0 . 009 0.646 0 . 004 0.868 0 . 002 0.313 0 . 004 0.870 0 . 002 6.833 | 0.981 0 . 000 0.377 0 . 007 0.605 0 . 004 0.857 0 . 002 0.262 0 . 002 0.859 0 . 001 8     | 0.981 0 . 000 0.479 0 . 007 0.670 0 . 004 0.882 0 . 002 0.340 0 . 002 0.883 0 . 002 2.833 | 0.981 0 . 000 0.459 0 . 005 0.678 0 . 003 0.886 0 . 001 0.331 0 . 003 0.887 0 . 001 2.833 | 0.981 0 . 000 0.491 0 . 008 0.706 0 . 005 0.899 0 . 002 0.358 0 . 004 0.899 0 . 002 1           |
| MIR    | 1-HL 1-OE 1-Cov 1-RL AP AUC     | 0.890 0 . 001 0.646 0 . 009 0.673 0 . 003 0.874 0 . 002 0.599 0 . 003 0.861 0 . 001       | 0.890 0 . 001 0.647 0 . 010 0.661 0 . 004 0.869 0 . 003 0.595 0 . 007 0.855 0 . 002       | 0.890 0 . 001 0.645 0 . 008 0.657 0 . 003 0.867 0 . 003 0.592 0 . 006 0.854 0 . 002 6.167 | 0.876 0 . 001 0.533 0 . 008 0.615 0 . 002 0.835 0 . 001 0.519 0 . 003 0.828 0 . 001 9     | 0.840 0 . 004 0.511 0 . 016 0.588 0 . 013 0.809 0 . 014 0.494 0 . 017 0.801 0 . 017 10    | 0.880 0 . 004 0.639 0 . 009 0.665 0 . 002 0.862 0 . 003 0.589 0 . 004 0.852 0 . 003 6.667 | 0.877 0 . 001 0.609 0 . 007 0.624 0 . 002 0.847 0 . 001 0.548 0 . 003 0.839 0 . 001 8     | 0.893 0 . 001 0.667 0 . 006 0.681 0 . 002 0.878 0 . 001 0.614 0 . 004 0.864 0 . 001 2     | 0.890 0 . 001 0.654 0 . 007 0.668 0 . 006 0.873 0 . 002 0.603 0 . 005 0.859 0 . 002       | 0.896 0 . 001 0.690 0 . 009 0.694 0 . 003 0.888 0 . 002 0.631 0 . 004 0.875 0 . 001             |
|        | AVE 1-HL 1-OE 1-Cov 1-RL        | 3.833 0.948 0 . 001 0.619 0 . 015 0.806 0 . 006 0.888 0 . 005                             | 4.667 0.948 0 . 001 0.601 0 . 011 0.794 0 . 006 0.876 0 . 004                             | 0.947 0 . 001 0.594 0 . 012 0.793 0 . 006 0.875 0 . 004                                   | 0.935 0 . 000 0.537 0 . 011 0.768 0 . 005 0.860 0 . 004                                   | 0.899 0 . 002 0.465 0 . 018 0.744 0 . 008 0.833 0 . 006                                   | 0.940 0 . 003 0.604 0 . 016 0.796 0 . 008 0.878 0 . 006                                   | 0.935 0 . 001 0.450 0 . 008 0.759 0 . 006 0.850 0 . 004 0.537 0 . 008                     | 0.949 0 . 001 0.627 0 . 011 0.812 0 . 006 0.890 0 . 005 0.649 0 . 009                     | 3.667 0.948 0 . 001 0.626 0 . 009 0.809 0 . 006 0.889                                     | 1                                                                                               |
| OBJ    | AP AUC AVE                      | 0.639 0 . 010 0.897 0 . 004 4 0.931                                                       | 0.627 0 . 009 0.886 0 . 004 5.833 0.931 0 . 000 0.443 0 . 007                             | 0.623 0 . 010 0.885 0 . 004 6.833 0.931 0 . 001                                           | 0.577 0 . 009 0.872 0 . 004 8.167 0.927 0 . 001                                           | 0.512 0 . 014 0.846 0 . 006 9.833 0.882 0 . 004 0.366 0 . 039                             | 0.630 0 . 012 0.888 0 . 006 5.333 0.915 0 . 005 0.433 0 . 017 0.759 0 . 006               | 0.864 0 . 004 9 0.928 0 . 001 0.418 0 . 008                                               | 0.900 0 . 005 2 0.933 0 . 001 0.473 0 . 008                                               | 0 . 004 0.649 0 . 009 0.898 0 . 004 3 0.932 0 . 001 0.468 0 . 008                         | 0.953 0 . 001 0.685 0 . 011 0.834 0 . 007 0.910 0 . 004 0.692 0 . 009 0.918 0 . 004 1 0.933 0 . |
| PAS    | 1-HL 1-OE 1-Cov 1-RL AP AUC AVE | 0 . 001 0.462 0 . 009 0.781 0 . 007 0.830 0 . 006 0.548 0 . 007 0.851 0 . 005 3.5         | 0.749 0 . 003 0.803 0 . 002 0.517 0 . 004 0.827 0 . 002 5.667                             | 0.435 0 . 010 0.738 0 . 010 0.792 0 . 008 0.510 0 . 008 0.817 0 . 008 7.167               | 0.419 0 . 006 0.720 0 . 004 0.778 0 . 003 0.482 0 . 005 0.806 0 . 003 8.667               | 0.674 0 . 011 0.736 0 . 011 0.438 0 . 022 0.767 0 . 011 10                                | 0.808 0 . 006 0.524 0 . 009 0.830 0 . 006 6                                               | 0.738 0 . 003 0.797 0 . 002 0.486 0 . 005 0.823 0 . 002 7.5                               | 0.790 0 . 006 0.836 0 . 005 0.562 0 . 005 0.855 0 . 005 2                                 | 0.778 0 . 004 0.828 0 . 004 0.552 0 . 006 0.848 0 . 005 3.5                               | 001 0.495 0 . 013 0.817 0 . 004 0.862 0 . 004 0.590 0 . 008 0.880 0 . 003 1                     |

## 3.5 Ablation Study

The ablation experiments are conducted to deeply investigate the effect of the three crucial modules of TDLSR, i.e., proximity-aware graph attention recovery mechanism ( S 1 ), information theory-driven representation extraction framework ( S 2 ), multi-label semantic and label-specific representation learning ( S 3 ). After individually removing S 1 , S 2 and S 3 , we use mean imputation for missing samples, rely solely on a single multilayer perceptron (MLP) for feature extraction while discarding L IB , and directly employ a classifier based on fully connected layers without exploring label semantics, respectively. Based on the ablation results provided in Table 2, we have the following observations: (i) When either module is removed, the performance declines, which indicates the effectiveness and thoughtful design of our TDLSR. (ii) The recovery mechanism is crucial for enhancing performance, as it provides downstream modules with rich feature information. Moreover, feature separation outperforms the single-channel representations and incorporating category semantic learning enhances performance beyond that of classifier-only approaches. It demonstrates our thorough consideration of feature extraction and label associations.

## 4 Application to Comprehensive Potential Prediction of Players

To validate the practical applicability of our TDLSR, we evaluate its ability to predict multiple attributes of NBA players under partial data missingness. The NBA dataset was collected from Basketball-Reference [2], which contains 16,992 player-season records from the 2002-2022 seasons. Each sample is structured across six principal statistical views including scoring efficiency, rebounding and physical metrics, technical statistics, advanced efficiency metrics, player background, and season context. The prediction tasks comprise career stage classification, positional identification and awards prediction. Career stages are partitioned into early (first 25%), peak (middle 50%), and late (final

Table 2: Ablation study on Pascal07, OBJECT and Mirflickr with PER= 50% and LMR= 50% . ' ! ' and ' % ' represent the used and not used corresponding item, respectively.

| S 1   | S 2   | S 3   | Pascal07   | Pascal07   | Pascal07   | Pascal07   | OBJECT   | OBJECT   | OBJECT   | OBJECT   | Mirflickr   | Mirflickr   | Mirflickr   | Mirflickr   |
|-------|-------|-------|------------|------------|------------|------------|----------|----------|----------|----------|-------------|-------------|-------------|-------------|
|       |       |       | AP         | AUC        | 1-RL       | 1-OE       | AP       | AUC      | 1-RL     | 1-OE     | AP          | AUC         | 1-RL        | 1-OE        |
| %     | !     | !     | 0.546      | 0.852      | 0.830      | 0.455      | 0.650    | 0.903    | 0 .894   | 0.633    | 0.594       | 0.859       | 0.872       | 0.649       |
| !     | %     | !     | 0.576      | 0.874      | 0.853      | 0.478      | 0.687    | 0.914    | 0.906    | 0.678    | 0.614       | 0.872       | 0.881       | 0.652       |
| !     | !     | %     | 0.582      | 0.874      | 0.857      | 0.486      | 0.690    | 0.912    | 0.904    | 0.680    | 0.616       | 0.870       | 0.882       | 0.659       |
| !     | !     | !     | 0.599      | 0.882      | 0.864      | 0.519      | 0.702    | 0.924    | 0.916    | 0.688    | 0.631       | 0.875       | 0.889       | 0.687       |

25%) phases according to each player's professional timeline. Player positions (PG, SG, SF, PF, C) are represented using one-hot encoding, and multiple binary indicators corresponding to honors such as MVP awards and Defensive Player of the Year, are included to provide multi-task objectives and comprehensive modeling of player achievements.

Across varying levels of data incompleteness, with PER and LMR ranging from 50% to 90%, our method consistently surpasses baseline approaches, demonstrating superior robustness and reliability in attribute prediction. Moreover, all comparison methods fail to surpass an AP of 0.6 at 90% missing ratio, whereas our TDLSR achieves 0.668. Despite incomplete technical statistics and constrained annotation resources, our method remains effective in predicting player potential, including career development and honor attainment, which offers considerable promise for real-world applications.

Figure 4: Overview of the NBA dataset and performance with missing views and labels.

<!-- image -->

## 5 Conclusion

In this paper, we propose a theory-driven label-specific representation (TDLSR) framework for addressing the iMvMLC problem. Specifically, structural dependencies are modeled through graph attention mechanism inside each view for recovery, with the reconstruction fidelity adaptively tailored to enhance classification efficacy. Meanwhile, we construct a universal feature extraction model, where mutual information optimization serves to regulate information shift, interaction and orthogonality between representations. On the basis of complete feature semantic separation, we independently interact each representation with label prototypes that encode correlation semantic, aiming to extract label-specific discriminative features and uncover representation-sensitive label subsets. Moreover, we theoretically validate the effectiveness of representation learning and its influence on the generalization performance. Finally, the superiority of TDLSR are validated through extensive experiments and application to the NBA dataset. In the future, we will explore leveraging the prior knowledge embedded in LLM to facilitate label semantic learning.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China under Grant No. 62376281, the NSF for Distinguished Young Scholars under Grant No. 62425607, and the Key NSF of China under Grant No. 62136005.

## References

[1] Ahn, L. V.; and Dabbish, L. 2004. Labeling images with a computer game. In SIGCHI Conference on Human Factors in Computing Systems , 319-326.

[2] Basketball Reference. 2025. NBA Player Totals Statistics. https://www.basketball-reference.com . Accessed: 2025-03-03.

[3] Chen, T.; Wang, W.; Pu, T.; Qin, J.; Yang, Z.; Liu, J.; and Lin, L. 2024. Dynamic correlation learning and regularization for multi-label confidence calibration. IEEE Transactions on Image Processing , 33: 4811-4823.

[4] Duan, K.; Cui, S.; Shinnou, H.; and Bao, S. 2025. View-Channel Mixer Network for Double Incomplete Multi-View Multi-Label learning. Neurocomputing , 617: 129013.

[5] Duygulu, P.; Barnard, K.; de Freitas, J. F.; and Forsyth, D. A. 2002. Object recognition as machine translation: Learning a lexicon for a fixed image vocabulary. In Proceedings of European Conference on Computer Vision , 97-112.

[6] Everingham, M.; Van Gool, L.; Williams, C. K.; Winn, J.; and Zisserman, A. 2010. The pascal visual object classes (voc) challenge. International Journal of Computer Vision , 88: 303-338.

[7] Federici, M.; Dutta, A.; Forré, P.; Kushman, N.; and Akata, Z. 2020. Learning Robust Representations via Multi-View Information Bottleneck. In International Conference on Learning Representations .

[8] Grubinger, M.; Clough, P.; Müller, H.; and Deselaers, T. 2006. The iapr tc-12 benchmark: A new evaluation resource for visual information systems. In International Workshop onto Image , volume 2, 1-11.

[9] Guan, R.; Liu, T.; Tu, W.; Tang, C.; Luo, W.; and Liu, X. 2025. Sampling Enhanced Contrastive Multi-View Remote Sensing Data Clustering with Long-Short Range Information Mining. IEEE Transactions on Knowledge and Data Engineering , 1-15.

[10] Guan, R.; Tu, W.; Wang, S.; Liu, J.; Hu, D.; Tang, C.; Feng, Y .; Li, J.; Xiao, B.; and Liu, X. 2025. StructureAdaptive Multi-View Graph Clustering for Remote Sensing Data. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 39, 16933-16941.

[11] Hang, J.-Y.; and Zhang, M.-L. 2021. Collaborative learning of label semantics and deep label-specific features for multi-label classification. IEEE Transactions on Pattern Analysis and Machine Intelligence , 44(12): 9860-9871.

[12] Hao, P.; Gao, W.; and Hu, L. 2025. Embedded feature fusion for multi-view multi-label feature selection. Pattern Recognition , 157: 110888.

[13] Hao, P.; Liu, K.; and Gao, W. 2024. Anchor-guided global view reconstruction for multi-view multi-label feature selection. Information Sciences , 679: 121124.

[14] Huang, J.; Li, G.; Huang, Q.; and Wu, X. 2016. Learning label-specific features and class-dependent labels for multi-label classification. IEEE Transactions on Knowledge and Data Engineering , 28(12): 3309-3323.

[15] Huang, W.; Yang, S.; and Cai, H. 2023. Generalized information-theoretic multi-view clustering. Advances in Neural Information Processing Systems , 36: 58752-58764.

[16] Huiskes, M. J.; and Lew, M. S. 2008. The mir flickr retrieval evaluation. In Proceedings of ACM International Conference on Multimedia Information Retrieval , 39-43.

[17] Jia, X.; Jing, X.-Y.; Zhu, X.; Chen, S.; Du, B.; Cai, Z.; He, Z.; and Yue, D. 2020. Semi-supervised multi-view deep discriminant representation learning. IEEE Transactions on Pattern Analysis and Machine Intelligence , 43(7): 2496-2509.

[18] Kawaguchi, K.; Deng, Z.; Luh, K.; and Huang, J. 2022. Robustness implies generalization via datadependent generalization bounds. In International conference on machine learning , 10866-10894. PMLR.

[19] Li, Q.; Luo, T.; Jiang, M.; Jiang, Z.; Hou, C.; and Li, F. 2025. Semi-Supervised Multi-View Multi-Label Learning with View-Specific Transformer and Enhanced Pseudo-Label. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 39, 18430-18438.

[20] Li, Q.; Luo, T.; Jiang, M.; Liao, J.; and Jiang, Z. 2024. Deep Incomplete Multi-View Network SemiSupervised Multi-Label Learning with Unbiased Loss. In Proceedings of ACM International Conference on Multimedia , 9048-9056.

[21] Li, X.; and Chen, S. 2021. A concise yet effective model for non-aligned incomplete multi-view and missing multi-label learning. IEEE Transactions on Pattern Analysis and Machine Intelligence , 44(10): 5918-5932.

[22] Lin, Y.; Gou, Y.; Liu, X.; Bai, J.; Lv, J.; and Peng, X. 2022. Dual contrastive prediction for incomplete multi-view representation learning. IEEE Transactions on Pattern Analysis and Machine Intelligence , 45(4): 4447-4461.

[23] Liu, C.; Jia, J.; Wen, J.; Liu, Y .; Luo, X.; Huang, C.; and Xu, Y. 2024. Attention-induced embedding imputation for incomplete multi-view partial multi-label classification. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, 13864-13872.

[24] Liu, C.; Wen, J.; Liu, Y.; Huang, C.; Wu, Z.; Luo, X.; and Xu, Y. 2023. Masked two-channel decoupling framework for incomplete multi-view weak multi-label learning. Advances in Neural Information Processing Systems , 36: 32387-32400.

[25] Liu, C.; Wen, J.; Luo, X.; Huang, C.; Wu, Z.; and Xu, Y. 2023. Dicnet: Deep instance-level contrastive network for double incomplete multi-view multi-label classification. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 37, 8807-8815.

[26] Liu, C.; Wen, J.; Luo, X.; and Xu, Y. 2023. Incomplete multi-view multi-label learning via label-guided masked view-and category-aware transformers. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 37, 8816-8824.

[27] Liu, C.; Xu, G.; Wen, J.; Liu, Y .; Huang, C.; and Xu, Y . 2024. Partial multi-view multi-label classification via semantic invariance learning and prototype modeling. In Forty-first International Conference on Machine Learning .

[28] Ma, Z.; and Chen, S. 2021. Expand globally, shrink locally: Discriminant multi-label learning with missing labels. Pattern Recognition , 111: 107675.

[29] Pan, E.; and Kang, Z. 2021. Multi-view contrastive graph clustering. Advances in Neural Information Processing Systems , 34: 2148-2159.

[30] Park, L. A.; and Read, J. 2019. A blended metric for multi-label optimisation and evaluation. In Machine Learning and Knowledge Discovery in Databases: European Conference , 719-734. Springer.

- [31] Si, C.; Jia, Y .; Wang, R.; Zhang, M.-L.; Feng, Y .; and Qu, C. 2023. Multi-label classification with high-rank and high-order label correlations. IEEE Transactions on Knowledge and Data Engineering , 36(8): 4076-4088.

[32] Song, Q.; Gong, T.; Gao, S.; Zhou, H.; and Li, J. 2024. QUEST: Quadruple Multimodal Contrastive Learning with Constraints and Self-Penalization. Advances in Neural Information Processing Systems , 37: 28889-28919.

- [33] Tan, Q.; Yu, G.; Domeniconi, C.; Wang, J.; and Zhang, Z. 2018. Incomplete multi-view weak-label learning. In International Joint Conference on Artificial Intelligence , 2703-2709.
- [34] Tan, X.; Zhao, C.; Liu, C.; Wen, J.; and Tang, Z. 2024. A two-stage information extraction network for incomplete multi-view multi-label classification. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, 15249-15257.
- [35] Tian, X.; Zhang, Z.; Wang, C.; Zhang, W.; Qu, Y.; Ma, L.; Wu, Z.; Xie, Y.; and Tao, D. 2023. Variational distillation for multi-view learning. IEEE Transactions on Pattern Analysis and Machine Intelligence , 46(7): 4551-4566.
- [36] Wang, S.; Liu, X.; Liao, Q.; Wen, Y.; Zhu, E.; and He, K. 2025. Scalable Multi-View Graph Clustering With Cross-View Corresponding Anchor Alignment. IEEE Transactions on Knowledge and Data Engineering , 2932-2945.
- [37] Wang, Y.; Li, Q.; Chang, D.; Wen, J.; Xiao, F.; and Zhao, Y . 2025. A Category-Driven Contrastive Recovery Network for Double Incomplete Multi-view Multi-label Classification. IEEE Transactions on Multimedia , 1-10.
- [38] Wen, J.; Liu, C.; Deng, S.; Liu, Y.; Fei, L.; Yan, K.; and Xu, Y . 2023. Deep double incomplete multi-view multi-label learning with incomplete labels and missing views. IEEE Transactions on Neural Networks and Learning Systems , 35(8): 11396-11408.
- [39] Wen, W.; Gong, T.; Dong, Y.; Yu, S.; and Zhang, W. 2025. Towards the Generalization of Multi-view Learning: An Information-theoretical Analysis. arXiv preprint arXiv:2501.16768 .

[40] Ye, H.; Sunderraman, R.; and Ji, S. 2024. MatchXML: an efficient text-label matching framework for extreme multi-label text classification. IEEE Transactions on Knowledge and Data Engineering , 36: 4781-4793.

[41] Yuan, H.; Sun, Y.; Zhou, F.; Wen, J.; Yuan, S.; You, X.; and Ren, Z. 2025. Prototype Matching Learning for Incomplete Multi-view Clustering. IEEE Transactions on Image Processing , 34: 828-841.

[42] Zhang, X.; Wang, R.; Chen, S.; Jia, Y.; and Wang, D. D. 2024. AME-LSIFT: Attention-Aware Multi-Label Ensemble With Label Subset-SpecIfic FeaTures. IEEE Transactions on Knowledge and Data Engineering , 7627-7642.

[43] Zhao, D.; Gao, Q.; Lu, Y.; and Sun, D. 2022. Non-aligned multi-view multi-label classification via learning view-specific labels. IEEE Transactions on Multimedia , 25: 7235-7247.

[44] Zhu, Y.; Kwok, J. T.; and Zhou, Z.-H. 2017. Multi-label learning with global and local label correlation. IEEE Transactions on Knowledge and Data Engineering , 30(6): 1081-1094.

## A Derivation and Implementation of the Loss Function for the Universal View Extraction Model

## A.1 Derivation of the Loss Function

The universal view extraction model under mutual information constraints is

̸

<!-- formula-not-decoded -->

By incorporating variational derivations from the information theory, we arrive at the following objective function for optimization:

<!-- formula-not-decoded -->

Proof. Due to the challenge of distribution inference and the complexity of high-dimensional integration, directly computing mutual information is impractical. Therefore, it is common to ensure stability and feasibility by optimizing an estimable bound of mutual information. For the information shift term, we have the following transformation based on the definitions of mutual information and information entropy:

<!-- formula-not-decoded -->

Since Eq. (18) is intractable, we approximate p ( z ( v ) | s ( v ) ) using a stochastic variational distribution q ( z ( v ) | s ( v ) ) , which can be reasonably estimated. Combing the fact that entropy H ( z ( v ) ) ≥ 0 , we can further obtain the lower bound:

<!-- formula-not-decoded -->

The Kullback-Leibler divergence is denoted as

Thus, we have

<!-- formula-not-decoded -->

For the second term, based on the definition and non-negativity of entropy, a lower bound can be derived by introducing another estimable variational distribution q ( s ( v ) | s ( v ∗ ) ) . By expanding the mutual information in its integral form, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Based on the definition of Kullback-Leibler divergence, we have

<!-- formula-not-decoded -->

∗ ∗ ∗

<!-- formula-not-decoded -->

For the third term I ( o v ∗ | z ( v ∗ ) ; o ( v ) | z ( v ) ) , we derive the following upper bound:

<!-- formula-not-decoded -->

Similarly, we can derive the upper bound for the last term, which exhibit structure analogous to that of Eq. (27):

<!-- formula-not-decoded -->

Combining Eqs. (21), (25), (27) and (28), the objective is naturally transformed into minimizing its upper bound:

<!-- formula-not-decoded -->

## A.2 Implementation of the Loss Function

Data-driven contrastive learning [22] is used to compute various complex distributions. Specifically, representations are treated as probability vectors over D classes (corresponding to d dimension) via a Softmax activation function, and the joint distribution matrix is obtained by

<!-- formula-not-decoded -->

Due to the strong coupling between s ( v ) and s ( v ∗ ) , the variational distribution q ( s ( v ) | s ( v ∗ ) ) can be effectively estimated by the obtained conditional distribution. In contrast, owing to the extremely

weak correlation between o ( v ) and o ( v ∗ ) , as well as o ( v ) and s ( v ) , the distributions q ( o ( v ) / o ( v ∗ ) ) and q ( s ( v ) / o ( v ) ) can be approximated by the corresponding marginal distribution. Since the first information shift term is equivalent to the reconstruction loss, we construct a decoder for each view v to obtain ˆ z v , which is used to approximate the original view. Subsequently, by converting the integral to a summation form, each term in Eq. (29) can be expressed as

<!-- formula-not-decoded -->

where the Lagrange multiplie β is fixed to 1, and the marginal distributions P ( v ) d and P ( v ∗ ) d ′ can be obtained by summing over the d -th row and the d ′ -th column of the joint distribution matrix P ( v,v ∗ ) , with other symbols defined as the same. In this conversion, we let q ( s v / s v ∗ ) = ( ψ ( Q ( v ) )) α and q ( z v / s v ) = ( ψ ( M ( s v ) )) α , where ψ is a fully connected layer, and α is a balance factor to preserve crucial information and ensure model stability [17]. We set the value of α to 10 in our experiments.

## B Detailed Proof for the Theoretical Results

In this section, we will provide a rigorous proof of the theoretical results mentioned in the main text.

## B.1 Proof of the Theorem 1

̸

Theorem 4. ( Discriminability of Label-specific Representation. ) For label prototypes E j and E k such that k = j for all k , and view representations X ( v ) and X ( v ∗ ) such that v ∗ = v for all v ∗ , the discriminability of ˆ P ( v ) for class j necessitates that either of the following conditions be satisfied:

̸

<!-- formula-not-decoded -->

Proof. Since our training data is multi-view and multi-label, the discriminative power of view v with respect to label j involves two key considerations. First, among all labels, view v yields the most accurate prediction for label j ; second, across all views, view v provides the prediction for label j with the highest confidence. Thus, we proceed with the proof from the following two aspects:

̸

(i) For E j and E k such that k = j for all k , the following inequality regrading view v holds:

<!-- formula-not-decoded -->

The j -th component of ˆ P ( v ) i (corresponding to label j ) is

<!-- formula-not-decoded -->

where ˆ P ( v ) i,j is a vector where each element is the dot product of the corresponding elements of σ S ( E j ) and x ( v ) i . Since a linear classifier (e.g., fully connected layers) is used for classification, the prediction score for sample i is

<!-- formula-not-decoded -->

where W ∈ R C × ( C × d ) is the weight matrix and b ∈ R C is the bias term. Give that ˆ P ( v ) i ∈ R C × d , we can flatten it into C individual vectors for each label. Besides, W has a block-diagonal structure, i.e., the weights for each label j only act on ˆ P ( v ) i,j . Then, the prediction score for label j simplifies to

<!-- formula-not-decoded -->

where w j ∈ R d is the classifier weight for label j . To express discriminability, we need to show

̸

<!-- formula-not-decoded -->

The expectation is obtained by averaging over all empirical samples:

<!-- formula-not-decoded -->

Assuming classifier weights w k are independent of input data (e.g., optimized during training) and biases b k are constants, the core comparison reduces to

<!-- formula-not-decoded -->

During training, the classifier weights w j tend to align with label prototypes E j since E j captures the semantic meaning of label j . Thus, we approximate w k ≈ E ⊤ k . Substituting it into Eq. (32), the comparison terms simplify to

<!-- formula-not-decoded -->

Using properties of dot products:

<!-- formula-not-decoded -->

Then, the comparison is further refined into

<!-- formula-not-decoded -->

Note that σ S ( E j,l ) ∈ (0 , 1) and is monotonically increasing. For positive samples of label i , E j and x ( v ) i are better aligned, meaning E j,l · x ( v ) i,l has a higher expectation. Thus, if the condition is

<!-- formula-not-decoded -->

Then, the following inequality holds:

<!-- formula-not-decoded -->

Assuming E j and E k have similar scales ( ∥ E j ∥ 2 2 ≈ ∥ E k ∥ 2 2 ), we conclude

<!-- formula-not-decoded -->

Furthermore, we have

Similarly, we conclude

<!-- formula-not-decoded -->

Furthermore, we have

̸

<!-- formula-not-decoded -->

Therefore, the prediction score for label j exceeds that of all other labels, demonstrating the discriminative capability of view v for class j .

̸

(ii) For view v and v ∗ such that v ∗ = v for all v ∗ , the following inequality regrading label j holds:

<!-- formula-not-decoded -->

The prediction score for label j assigned by view v and v ∗ , respectively, is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To express discriminability, we need to show

<!-- formula-not-decoded -->

̸

Following the same rationale, we can transform the comparison into

<!-- formula-not-decoded -->

Thus, if the condition is

<!-- formula-not-decoded -->

Then, the following inequality holds:

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Therefore, the prediction score of the v -th view for class j is higher than that of all other views, indicating that view v exhibits discriminative power for label j . Combining (i) and (ii), we complete the proof.

## B.2 Proof of the Theorem 2

Theorem 5. ( Effectiveness of Disentangled Representation. ) Let the disentangled representation be denoted as R = ( S (1) , . . . , S ( V ) , O (1) , . . . , O ( V ) ) , where the information entropy of each representation is assumed to be fixed, i.e., H ( S ( v ) ) = H ( O ( v ) ) = H 0 (1 ≤ v ≤ V ) . Then, in the case where each shared and specific representation is indispensable for prediction, I ( R ; Y ) will attain its maximum when R = R ∗ , with R ∗ being the optimal solution of the problem (16).

Proof. By integrating core features from the raw data to construct shared representations and leveraging classification loss to derive view-specific representations aligned with particular labels, our approach enables the shared features to encapsulate primary generalization information across multiple views, while the specific features are designed to capture discriminative information that is

<!-- formula-not-decoded -->

Figure 5: Feature extraction visualization. Fig. (a) illustrates the optimal solution obtained by our model, while the remaining four figures display cases where the constraints of information shift, interaction, and orthogonality are successively removed. The shaded areas represent the shared representations, whereas the other colored regions indicate the view-specific representations.

<!-- image -->

finely attuned to task-specific characteristics. Thus, the assumption that each shared and specific representation is indispensable for prediction aligns well with our model. Besides, as all representations are extracted into the same dimensional space, it is reasonable to posit that they encode equivalent quantities of information, i.e., H ( S ( v ) ) = H ( O ( v ) ) = H 0 (1 ≤ v ≤ V ) . Let define the optimal solution of the problem (16) as R ∗ = ( S (1) ∗ , . . . , S ( V ) ∗ , O (1) ∗ , . . . , O ( V ) ∗ ) . Then, we specify that identifying each class Y c requires a corresponding representation setting R Y c to be mined, i.e, H ( Y c ) = H ( R Y c ) , which indicates that the successful extraction of R Y c is sufficient for predicting Y c . Based on this, we establish the following formulation:

<!-- formula-not-decoded -->

where f v c and g v c are indicator functions representing the shared and specific information components that constitute R Y c . According to Eq. (36), all information contained in the extracted representations is essential for predicting the target labels. Next, we demonstrate from three perspectives that our feature extraction model is capable of maximizing the mutual information between the representations and the labels, i.e., the representations contain insufficient information for prediction, just enough information, or redundant information beyond what is necessary for prediction.

<!-- formula-not-decoded -->

In this case, we denote the mutual information between R ∗ and Y as I 0 , i.e.,

<!-- formula-not-decoded -->

where L and U denote the subsets of labels that are fully predictable and not adequately predictable, respectively, and |L| + |U| = C . When the core shared feature information is essential to all labels, reducing I ( S (1) ∗ ; . . . ; S ( V ) ∗ ) to enhance label-specific information associated with certain labels will not lead to an increase in I ( R ∗ ; Y ) . Since ∑ c ∈L H ( Y c ) is fixed, postulating the existence of an alternative representation R satisfying I ( R ; Y ) &gt; I ( R ∗ ; Y ) would inevitably contradict the

assumption that the information entropy of each representation is fixed at H 0 . Thus, we can obtain I ( R ∗ ; Y ) ≥ I ( R ; Y ) . Given the insufficiency of prediction information and a fixed amount of information encoded in the representations, enhancing performance necessitates incorporating usable content beyond raw information and introducing additional constraints offers no further gain in the mutual information between representations and labels.

<!-- formula-not-decoded -->

In this case, similarly, we have the following equality:

<!-- formula-not-decoded -->

If the information shift term is not sufficiently optimized as shown in Fig. 5.(b), there will exist a shared representation ˜ S ( m ) in the setting R such that

̸

<!-- formula-not-decoded -->

where T m denotes the set of labels whose prediction processes involve contributions from ˜ S ( m ) , while T n m is the set unaffected by ˜ S ( m ) . Since the change in information entropy occurs exclusively in ˜ S ( m ) , we have

̸

<!-- formula-not-decoded -->

Furthermore, we have the following comparison:

<!-- formula-not-decoded -->

̸

If the shared information interaction term is not sufficiently optimized (Fig. 5.(c)), there will exist a set of shared representations whose task-relevant information is significantly reduced due to ineffective interactions. Under such circumstances, it similarly follows that I ( R ; Y ) ≤ I ( R ∗ ; Y ) . Moreover, insufficient suppression of interactions among specific representations causes overlap, thereby reducing the useful information entropy of each representation (Fig. 5.(d)). Additionally, inadequate optimization of the orthogonality constraint can introduce redundancy between shared and specific representations, which in turn hinders the extraction of the specific representations (Fig. 5.(e)). For this two aspects, we also have I ( R ; Y ) ≤ I ( R ∗ ; Y ) .

<!-- formula-not-decoded -->

̸

In this case, we can obtain

<!-- formula-not-decoded -->

Given that ∑ C c =1 H ( Y c ) quantifies the total label information, the mutual information I ( R ; Y ) obtained from any alternative representation is inherently constrained by this upper bound. Therefore, we conclude that I ( R ; Y ) ≤ I ( R ∗ ; Y ) .

Combining (i) (ii) and (iii), we deduce that the maximization of I ( R ; Y ) is attained when R serves as the optimal solution R ∗ of the problem (16). Moreover, the principles of information shift, interaction, and orthogonality are all indispensable for achieving this.

## B.3 Proof of the Theorem 3

Theorem 6. ( Generalization Error Bound. ) Our model is designed to learn a vector-valued function f = ( f 1 , . . . , f C ) : X ↦→ R C . The expected risk and empirical risk w.r.t. the training dataset D are denoted as R ( f ) = E ( X , Y ) ∼X×Y [ ℓ ( f av ( X , Q ) , Y )] and ̂ R D ( f ) = 1 NC ∑ N i =1 ∑ C c =1 ℓ ( ∑ V v =1 ( Q i,v f c ( x ( v ) i )) , Y i,c ) , where f av ( · ) refers to the late fusion of multiple views. With probability at least 1 -δ , we have the following generalization error bound:

<!-- formula-not-decoded -->

where ˜ K 1 = ˜ K 3 = O ( C ) , ˜ K 2 = O ( √ C ) , ˜ K 4 is constant of order ˜ O (1) as N,V → ∞ , and gen rec ( Q , X , Y ) is the generalization error related to the view reconstruction quality. Moreover, the generalization error bound becomes increasingly tighter during the optimization of the problem (16).

Proof. The training dataset denoted as D = { ( x i , y i ) : i ∈ [ N ] } is drawn from a probability distribution over X ×Y . Each x i = ( x (1) i , . . . , x ( V ) i ) consists of V views and y i = ( Y i, 1 , . . . , Y i,C ) . Our strategy is to learn a vector-valued function f = ( f 1 , . . . , f C ) : X ↦→ R C and determine relevant labels by applying a thresholding criterion. The goal of learning is to find a hypothesis f ∈ F with good generalization performance by minimizing the loss ℓ on the dataset D . As missing labels inevitably degrade generalization, we focus on deriving the tightest generalization error bound under complete labeling. Thus, the expected risk and empirical risk w.r.t. the training dataset D are denoted as R ( f ) = E ( X , Y ) ∼X×Y [ ℓ ( f av ( X , Q ) , Y )] and ̂ R D ( f ) = 1 NC ∑ N i =1 ∑ C c =1 ℓ ( ∑ V v =1 ( Q i,v f c ( x ( v ) i )) , Y i,c ) , respectively, where f av ( · ) refers to the result fusion of multiple views, Q is the reconstruction quality score serving as the fusion weights. We define the function class of TDLSR as follows:

<!-- formula-not-decoded -->

where ϕ s and ϕ o denote the view-common and view-specific representation extractors, ζ j is a nonlinear mapping induced by the label-specific representation construction and w is the classification head. Besides, the disentangled representation R is expressed by R = ϕ ( X ) = ( ϕ s ( X ) , ϕ o ( X )) =

( S (1) , . . . , S ( V ) , O (1) , . . . , O ( V ) ) . Let multi-view data X = { X ( v ) } V v =1 be produced with a hidden label-specific function θ c by X = θ c ( Y c , V ) , where Y c is the randomly generated single label and V = { V ( v ) = ( V ( v ) 1 , . . . , V ( v ) d ) } V v =1 ∈ R m × d are nuisance variables. Denote the conditional random variables of X and R given the category Y c = y c as X y c and R y c , respectively. For any y c ∈ Y ∗ = {-1 , 1 } , the sensitivity c y c ϕ of the representation function ϕ = { ϕ ( v ) } V v =1 w.r.t the nuisance variable V ( v ) i is defined as

<!-- formula-not-decoded -->

where θ y c ( v ( v ) ) = θ ( y c , v ( v ) ) and p r ( r ) = P ( R = r ) . Based on Eq. (38), we set the global sensitivity of ϕ as c ϕ = sup c ∈ [ C ] c y c ϕ . Let R x c = { θ y c ( v ) , ϕ ◦ θ y c ( v ) : v ∈ V , y c ∈ Y ∗ } denote the complete set of multi-view data and their corresponding representations. For any γ &gt; 0 , we construct the following typical representation subset for each class [39]:

<!-- formula-not-decoded -->

Define the function h ′ y c ( v ) = -log p r | y c ( h ′′ y c ( v )) , where h ′′ y c ( v ) = ϕ ◦ ( θ y c ( v )) . Let p v ( v ) = P ( V = v ) and h -1 y c ( r ) = { v ∈ V : h y c ( v ) = r } , we have

<!-- formula-not-decoded -->

By applying McDiarmid's inequality on h ′ y c ( V ) , we get

<!-- formula-not-decoded -->

Taking δ = exp ( -2 ϵ 2 d ( c ϕ ) 2 ) , we have

<!-- formula-not-decoded -->

After setting δ = γ/ √ NV , i.e., P ( X , R / ∈ R x c,γ ) ≤ δ = γ √ NV , we can obtain

<!-- formula-not-decoded -->

Then, we further derive the following transformation:

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

Therefore, regarding the property of the typical subset, the following lemma can be derived:

Lemma 1. For any γ &gt; 0 and all v ∈ [ V ] , we have

<!-- formula-not-decoded -->

Besides, we need the following lemma for subsequent proof:

Lemma 2. [18] The vector X = ( X 1 , . . . , X k ) is defined to follow the multinomial distribution with parameters p = ( p 1 , . . . , p k ) . Let ¯ a 1 , . . . , ¯ a k ≥ 0 be fixed such that ∑ k i =1 ¯ a i p i = 0 . Then, for any ϵ &gt; 0 , the following inequality holds:

<!-- formula-not-decoded -->

where β = 2 ∑ k i =1 ¯ a 2 i p i .

We further define T y c = |R x c,γ | , where the typical subset R x c,γ consists of elements R x c,γ = { ( a x c, 1 , a s c, 1 , a o c, 1 ) , . . . , ( a x c,T , a s c,T , a o c,T ) } . Besides, we introduce the following label-specific sets:

<!-- formula-not-decoded -->

Then, we conduct an analysis of the classification generalization error:

<!-- formula-not-decoded -->

Since ℓ is a convex function, we can apply Jensen's inequality to obtain

<!-- formula-not-decoded -->

Thus, we define a instance-level loss difference term ∆ ℓ c i ( Q i, : , X i, : , Y i,c ) such that

<!-- formula-not-decoded -->

Define q = sup i ∈ [ N ] sup v ∈ [ V ] Q i,v as the maximum view reconstruction quality among all views. Regarding the cumulative loss calculation for individual views, we have

<!-- formula-not-decoded -->

̸

Furthermore, we can obtain the upper bound for the empirical risk:

<!-- formula-not-decoded -->

The random variables of the quality score of the v -th view and the c -th category are denoted as Q ( v ) and Y c . Similarly, the population risk can be decomposed as

<!-- formula-not-decoded -->

(46) Since E ( X , Y c ) ∼X×Y ∗ [ ℓ ( f av c ( X , Q ) , Y c )] = E ( X , Y c ) ∼X×Y ∗ [ ℓ ( ∑ V v =1 Q ( v ) f c ( X ( v ) ) , Y c ))] , we can further decompose it based on the convexity of ℓ :

<!-- formula-not-decoded -->

where ℓ c ( Q , Y c ) represents the change in loss ℓ associated with class c induced by view fusion. Then, we can obtain the following transformation combing Eqs. (46) and (47):

<!-- formula-not-decoded -->

Putting Eqs. (58) and (48) back into Eq. (41), we have

<!-- formula-not-decoded -->

Let S x,y = sup c ∈ [ C ] sup ( X , Y c ) ∈X×Y ∗ ∑ V v =1 Q ( v ) ℓ ( f c ( X ( v ) ) , Y c ) denote the maximum attainable losses among all classes and S s x,y = sup c ∈ [ C ] sup i ∈ [ n ] Q i,v ∑ V v =1 ℓ ( f c ( x ( v ) i ) , Y i,c ) represent the maximum instance-level losses. We consider the results of Eq. (49) as four terms, with the upper bound of the first term as follows:

<!-- formula-not-decoded -->

(50) Define p c k = P ( Y c = y c , X ( j ) = a c,x k , ϕ j ( X ( j ) ) = ( a c,s k , a c,o k )) for k ∈ [ T y c ] , p c T y c +1 = P ( Y c = y c , ( X ( j ) , ϕ j ( X ( j ) )) / ∈ R x c,γ ) and b c k = ℓ ( ψ c ( a c,s k , a c,o k ) , y c )) for k ∈ [ T y c + 1] . Then, we can obtain the following term:

<!-- formula-not-decoded -->

Applying Lemma 2, for any ϵ &gt; 0 and k ∈ [ T y c ] , we have

<!-- formula-not-decoded -->

Similarly, we get the following inequality regarding k = T y c +1 :

<!-- formula-not-decoded -->

Let δ be the right-hand side of Eq. (52) and (53), respectively, we can obtain the following variants:

<!-- formula-not-decoded -->

Taking union bounds over all y c ∈ y ∗ ( | y ∗ = 2 | ) and k ∈ [ T y c ] , we have

<!-- formula-not-decoded -->

Then, for any δ &gt; 0 , with probability at least 1 -δ , we have the following scaling regrading the second term of Eq. (49) by using Jensen's inequality:

<!-- formula-not-decoded -->

Similarly, for P c 3 ,k , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

̸

Next, we can derive the constraint result for the third term:

<!-- formula-not-decoded -->

The property of T y c is as follows:

<!-- formula-not-decoded -->

Combining Eqs. (58) and (59), we can get the following bound by using Jensen's inequality:

<!-- formula-not-decoded -->

By applying the chain rule, we obtain the following conclusion:

<!-- formula-not-decoded -->

Since our model achieves complete feature separation, with seamlessly integrated shared representations, no overlap among special representations, and no redundancy between shared and special features, we have the following equality:

<!-- formula-not-decoded -->

Thus, we can further obtain

<!-- formula-not-decoded -->

Putting the above back into Eq. 60, we have

<!-- formula-not-decoded -->

Regarding the fourth term, as the accuracy of the view reconstruction increases, indicated by a larger value of Q , the loss discrepancy ∆ ℓ c i ( Q i, : , X i, : , Y i,c ) becomes more significant, while the corresponding generalization error bound becomes tighter. Therefore, we interpret the fourth term as the generalization error induced by the quality of view reconstruction, i.e.,

<!-- formula-not-decoded -->

Therefore, with probability at least 1 -δ , the generalization error bound of our model is

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Since S ( v ) and O ( v ) are generated by X ( v ) , we have the following transformation:

<!-- formula-not-decoded -->

Based on Theorem 5, it can be inferred that during the model optimization process, I (( S (1) , . . . , S ( V ) , O (1) , . . . , O ( V ) ); Y ) continuously approaches its maximum value, leading to a decrease in ∑ C c =1 ∑ V v =1 I ( X ( v ) ; S ( v ) , O ( v ) | Y c ) , which indicates an improvement in generalization performance.

## C Experiment

## C.1 Experiment Setup

Table 3: Detailed information of datasets.

| View      | Object    | VOC 2007        | Corel 5k        | Esp Game        | IAPR TC-12      | MIR FLICKR      |
|-----------|-----------|-----------------|-----------------|-----------------|-----------------|-----------------|
| 1         | CH(64)    | DenseHue(100)   | DenseHue(100)   | DenseHue(100)   | DenseHue(100)   | DenseHue(100)   |
| 2         | CM(225)   | DenseSift(1000) | DenseSift(1000) | DenseSift(1000) | DenseSift(1000) | DenseSift(1000) |
| 3         | CORR(144) | GIST(512)       | GIST(512)       | GIST(512)       | GIST(512)       | GIST(512)       |
| 4         | EDH(73)   | HSV(4096)       | HSV(4096)       | HSV(4096)       | HSV(4096)       | HSV(4096)       |
| 5         | WT(128)   | RGB(4096)       | RGB(4096)       | RGB(4096)       | RGB(4096)       | RGB(4096)       |
| 6         | -         | LAB(4096)       | LAB(4096)       | LAB(4096)       | LAB(4096)       | LAB(4096)       |
| #Label    | 31        | 20              | 260             | 268             | 291             | 38              |
| #Instance | 6047      | 9963            | 4999            | 20770           | 19627           | 25000           |

Datasets and Comparison Methods. In our experiments, six public multi-view multi-label datasets are selected as shown in Table 3. Their specific descriptions are as follows. Corel 5k is composed of 4999 image samples and 260 words, where each word can be regarded as an annotation or label. IAPRTC12 comprises 19627 high-quality natural images and each image contains 261 labels, including sports, actions, animals, cities, and so on. ESPGame is a multi-view multi-label dataset containing 20770 images with 268 corresponding tags. Pascal07 is a widely utilized dataset for visual object detection and recognition, which contains 9963 images and 20 kinds of objects. Mirfickr consists of 25,000 images from the Flickr platform, annotated with a total of 38 tags. OBJECT has 6047 instances requiring recognition, which are characterized by five distinct perspectives and annotated with 31 attributes. To validate the effectiveness of TDLSR, we compare it with nine state-of-the-art approaches, i.e., AIMNet [23], DICNet [25], DIMC [38], iMVWL [33], LMVCAT [26], MTD [24], SIP [27], LVSL [43], DM2L [28]. We also provide a comprehensive overview of their sources and functions in Table 4.

Construction of the Application NBA Dataset. The NBA dataset crawled from BasketballReference [2] includes 16,992 player samples from the regular and playoff seasons spanning 2002 to 2022. Each sample comprises six views capturing different aspects of player performance and background information: 1) Scoring Statistics , 20 features including shooting attempts, shooting percentages (scaled by 1000), points, and per-minute scoring efficiency, representing scoring ability and shooting efficiency. 2) Rebounding and Physical Attributes , 14 features such as games played, rebounds, fouls, and playing time averages, highlighting physical competitiveness and playing consistency. 3) Technical Statistics , 15 features including assists, steals, blocks, turnovers, and tripledoubles, reflecting player's playmaking and defensive contributions. 4) Advanced Efficiency Metrics , 10 features measuring comprehensive performance like true shooting percentage, usage rate, player efficiency rating, and per-minute rates. 5) Player Background , 41 features encoding age and team membership via one-hot encoding. 6) Seasonal Context , 22 features encompassing season indicators, playoff status, also one-hot encoded. Player attributes are structured as an 18-dimensional multi-label vector per sample, which includes 10 award-related labels (e.g., All-Star selection, All-NBA teams, MVP nomination), 5 positional one-hot labels (PG, SG, SF, PF, C) and 3 career stage one-hot labels categorizing each player-season into Early Career (first 25%), Prime Career (middle 50%), and Late Career (last 25%) according to the player's elapsed and remaining career years.

Implementation Details. We employ Hamming Loss (HL), Ranking Loss (RL), OneError (OE), Coverage (Cov), Average Precision (AP), and Area Under Curve (AUC) as six metrics to unify the experimental standards. Higher AP and AUC values indicate better performance, while lower HL, RL, OE, and Cov values are preferred. Their evaluation contents are described below: 1) ACC measures the proportion of correctly predicted labels across all samples. 2) RL evaluates the accuracy of the model's ranking of predicted labels compared to true labels. 3) AP computes the area under the precision-recall curve, indicating the average precision achieved across all recall levels. 4) AUC quantifies the probability that a randomly selected positive instance is ranked higher by the model than a randomly selected negative instance across all possible threshold values. 5) OE evaluates whether the top-ranked label predicted by the model is incorrect. 6) Cov computes the number of labels the model needs to traverse to cover all true labels, reflecting the efficiency of the model's predicted label range. The neighbor number k is fixed to 10 for all datasets. Adam optimizer with the initial learning rate of 0.0001 is used for optimization of all datasets. All methods use the same

dataset partition when conducting experiments, while the locations of view missing and label absence are recorded and kept consistent.

Table 4: Detailed information of comparison methods. ! represent the method is able to handle the corresponding problem.

| Method   | Source   |   Year | Multi-label   | Multi-view   | Missing-view   | Missing-label   |
|----------|----------|--------|---------------|--------------|----------------|-----------------|
| iMVWL    | IJCAI    |   2018 | !             | !            | !              | !               |
| DM2L     | PR       |   2021 | !             | %            | %              | !               |
| LVLS     | TMM      |   2022 | !             | !            | %              | %               |
| DICNet   | AAAI     |   2023 | !             | !            | !              | !               |
| DIMC     | TNNLS    |   2023 | !             | !            | !              | !               |
| LMVCAT   | AAAI     |   2023 | !             | !            | !              | !               |
| AIMNet   | AAAI     |   2024 | !             | !            | !              | !               |
| MTD      | NeurIPS  |   2024 | !             | !            | !              | !               |
| SIP      | ICML     |   2024 | !             | !            | !              | !               |

## C.2 Experiment Result

Table 5: Experimental results of nine methods on the six datasets with 90% PER and 90% LMR. 'Ave.R' refers to the mean ranking of the corresponding method across all six metrics.

| DATA   | METRIC                          | AIMNet                                                                                          | DICNet                                                                                                  | DIMC                                                                                            | DM2L                                                                                                    | iMVWL                                                                                                                            | LMVCAT                                                                                                                                    | LVSL                                                                                                          | MTD                                                                                                     | SIP                                                                                                     | TDLSR                                                                                                   |
|--------|---------------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| COR    | 1-HL 1-OE 1-Cov 1-RL AP AUC AVE | 0.987 0 . 000 0.277 0 . 007 0.605 0 . 003 0.823 0 . 001 0.240 0 . 002 0.826 0 . 001 3.167       | 0.987 0 . 000 0.242 0 . 006 0.515 0 . 009 0.774 0 . 003 0.208 0 . 001 0.776 0 . 004 6.833               | 0.987 0 . 000 0.239 0 . 009 0.518 0 . 005 0.772 0 . 005 0.206 0 . 004 0.774 0 . 005 7.333 0.982 | 0.987 0 . 000 0.171 0 . 010 0.481 0 . 013 0.750 0 . 005 0.181 0 . 002 0.754 0 . 006 8.333 0.983 0 . 000 | 0.976 0 . 000 0.181 0 . 005 0.524 0 . 004 0.762 0 . 004 0.163 0 . 004 0.766 0 . 004 9                                            | 0.987 0 . 000 0.229 0 . 008 0.600 0 . 004 0.817 0 . 003 0.214 0 . 001 0.820 0 . 003 5                                                     | 0.987 0 . 000 0.247 0 . 006 0.608 0 . 002 0.823 0 . 001 0.228 0 . 001 0.827 0 . 001 3                         | 0.987 0 . 000 0.274 0 . 012 0.573 0 . 005 0.809 0 . 001 0.234 0 . 005 0.811 0 . 002 5.667               | 0.986 0 . 000 0.289 0 . 014 0.601 0 . 006 0.821 0 . 001 0.242 0 . 002 0.823 0 . 001 4.167               | 0.987 0 . 000 0.374 0 . 010 0.692 0 . 004 0.866 0 . 002 0.323 0 . 004 0.869 0 . 002 1                   |
| ESP    | 1-HL 1-OE 1-Cov 1-RL AP AUC AVE | 0.982 0 . 000 0.310 0 . 007 0.508 0 . 003 0.792 0 . 001 0.222 0 . 003 0.797 0 . 001 3.333       | 0.982 0 . 000 0.289 0 . 012 0.464 0 . 000 0.773 0 . 001 0.210 0 . 002 0.777 0 . 000 5.333 0.980 0 . 000 | 0 . 000 0.283 0 . 006 0.456 0 . 004 0.769 0 . 002 0.207 0 . 003 0.772 0 . 002 6.5 0.980 0 . 000 | 0.210 0 . 001 0.447 0 . 003 0.758 0 . 002 0.172 0 . 002 0.762 0 . 002 7.667 0.980 0 . 000 0.247 0 . 004 | 0.969 0 . 000 0.204 0 . 009 0.421 0 . 007 0.729 0 . 004 0.155 0 . 004 0.733 0 . 005 10 0.966 0 . 000 0.245 0 . 011 0.438 0 . 009 | 0.982 0 . 000 0.266 0 . 023 0.468 0 . 003 0.771 0 . 001 0.201 0 . 006 0.775 0 . 002 6.333 0.980 0 . 000 0.290 0 . 007 0.471 0 . 005 0.793 | 0.983 0 . 000 0.265 0 . 004 0.489 0 . 001 0.783 0 . 001 0.204 0 . 001 0.787 0 . 000 5.167 0.980 0 . 000 0.294 | 0.982 0 . 000 0.302 0 . 008 0.492 0 . 003 0.786 0 . 000 0.219 0 . 002 0.790 0 . 000 4.167 0.980 0 . 000 | 0.982 0 . 000 0.327 0 . 008 0.500 0 . 004 0.785 0 . 001 0.225 0 . 002 0.790 0 . 002 4 0.980 0 . 000     | 0.982 0 . 000 0.385 0 . 010 0.576 0 . 002 0.825 0 . 001 0.271 0 . 003 0.830 0 . 001 1.333 0.980 0 . 000 |
| IAP    | 1-HL 1-OE 1-Cov 1-RL AP AUC     | 0.980 0 . 000 0.342 0 . 004 0.521 0 . 002 0.818 0 . 003 0.229 0 . 002 0.822 0 . 002 2.5         | 0.333 0 . 009 0.472 0 . 004 0.799 0 . 004 0.222 0 . 004 0.801 0 . 003 4.833 0.879 0 . 001               | 0.318 0 . 002 0.468 0 . 002 0.795 0 . 002 0.215 0 . 002 0.798 0 . 002 5.833 0.877 0 . 002       | 0.443 0 . 005 0.780 0 . 003 0.186 0 . 003 0.783 0 . 003 7.667 0.876 0 . 000 0.439 0 . 005 0.572 0 . 003 | 0.761 0 . 005 0.167 0 . 004 0.766 0 . 005 10 0.827 0 . 004 0.406 0 . 023 0.530 0 . 012 0.765 0 . 011                             | 0 . 000 0.202 0 . 003 0.797 0 . 001 6.667 0.865 0 . 004 0.470 0 . 020 0.581 0 . 002                                                       | 0 . 004 0.496 0 . 002 0.807 0 . 001 0.208 0 . 001 0.811 0 . 001 5 0.874 0 . 000                               | 0.344 0 . 007 0.510 0 . 006 0.816 0 . 003 0.232 0 . 002 0.819 0 . 002 3.167                             | 0.355 0 . 005 0.519 0 . 003 0.817 0 . 003 0.235 0 . 002 0.820 0 . 002 2.333 0.875 0 . 002 0.540 0 . 009 | 0.397 0 . 009 0.616 0 . 001 0.860 0 . 001 0.278 0 . 003 0.861 0 . 001 1 0.885 0 . 002 0.612 0 . 006     |
|        | AVE                             | 0.875 0 . 001 0.506 0 . 023 0.598 0 . 006 0.827 0 . 005                                         | 0.533 0 . 005 0.594 0 . 003 0.828 0 . 001 0.512 0 . 001                                                 | 0.511 0 . 005 0.589 0 . 001 0.823 0 . 001 0.501 0 . 002 0.818 0 . 001                           | 0.809 0 . 002 0.467 0 . 004 0.805 0 . 000 8.333                                                         | 0.415 0 . 009 0.769 0 . 007 10                                                                                                   | 0.817 0 . 003 0.485 0 . 010                                                                                                               | 0.485 0 . 006 0.584 0 . 001 0.819 0 . 001 0.482 0 . 002                                                       | 0.880 0 . 001 0.535 0 . 008 0.606 0 . 006 0.834 0 . 003 0.519 0 . 006                                   | 0.604 0 . 006 0.830 0 . 002 0.519 0 . 002                                                               | 0.645 0 . 005 0.861 0 . 003 0.575 0 . 002 0.852 0 . 000                                                 |
| MIR    | 1-HL 1-OE 1-Cov 1-RL AP AUC AVE | 0.494 0 . 017 0.820 0 . 003 5.333                                                               | 0.823 0 . 001 4 0.938 0 . 001 0.453 0 . 009                                                             | 5.333 0.938 0 . 000 0.439 0 . 005 0.709 0 . 010                                                 | 0.935 0 . 000 0.415 0 . 009 0.682 0 . 002                                                               | 0.882 0 . 005 0.335 0 . 036                                                                                                      | 0.808 0 . 004 8 0.927 0 . 002 0.405 0 . 019                                                                                               | 0.816 0 . 000 7.333                                                                                           | 0.827 0 . 002 2.167                                                                                     | 0.823 0 . 001 3.333                                                                                     | 1                                                                                                       |
| OBJ    | 1-HL 1-OE 1-Cov                 | 0.937 0 . 001 0.468 0 . 005 0.727 0 . 009                                                       | 0.720 0 . 008 0.823 0 . 006                                                                             | 0.814 0 . 008                                                                                   | 0.800 0 . 000                                                                                           | 0.657 0 . 023 0.768 0 . 012                                                                                                      | 0.705 0 . 017 0.806 0 . 011                                                                                                               | 0.934 0 . 000 0.364 0 . 004 0.712 0 . 002 0.811 0 . 001                                                       | 0.938 0 . 000 0.474 0 . 004 0.740 0 . 007 0.835 0 . 003                                                 | 0.937 0 . 001 0.485 0 . 013 0.727 0 . 006 0.828 0 . 004                                                 | 0.946 0 . 001 0.603 0 . 020 0.795 0 . 003 0.881 0 . 002                                                 |
|        | 1-RL AP AUC AVE                 | 0.829 0 . 003 0.506 0 . 010 0.842 0 . 003                                                       | 0.502 0 . 006 0.836 0 . 005 4.667 0.927 0 . 001 0.402 0 . 002                                           | 0.489 0 . 010 0.828 0 . 008 5.833 0.927 0 . 001                                                 | 0.470 0 . 002 0.814 0 . 000 8.167 0.926 000                                                             | 0.394 0 . 026 0.784 0 . 012 10 0.871 0 . 003 0.306 0 . 037                                                                       | 0.476 0 . 009 0.821 0 . 011 8 0.921                                                                                                       | 0.446 0 . 001 0.827 0 . 001 7.667 0.926 0 . 000 0.415 0 . 001 0.654 0 . 005 0.726 0 . 004                     | 0.519 0 . 008 0.848 0 . 003 2.333 0.926 0 . 001 0.395 0 . 011 0.674 0 . 005                             | 0.522 0 . 009 0.840 0 . 004 3.667 0.923 0 . 003                                                         | 0.624 0 . 011 0.891 0 . 001 1                                                                           |
|        | 1-HL 1-OE 1-Cov 1-RL            | 3.667 0.923 0 . 003 0.382 0 . 023 0.658 0 . 002 0.727 0 . 006 0.440 0 . 009 0.754 0 . 003 5.333 | 0.636 0 . 013 0.710 0 . 011 0.440 0 . 005                                                               | 0.403 0 . 0.626 0 . 0.703 0 . 0.434 0 .                                                         | 0 . 0.380 0 . 007 0.636 0 . 009 0.698 0 . 008 0.421 0 . 006 0.733 0 . 007                               | 0.589 0 . 014 0.658 0 . 014 0.368 0 . 021 0.690 0 . 013                                                                          | 0 . 003 0.376 0 . 021 0.630 0 . 018 0.693 0 . 016 0.430 0 . 005 0.727 0 . 010 8.5                                                         |                                                                                                               | 0.740 0 . 003                                                                                           | 0.389 0 . 011 0.668 0 . 018 0.729 0 . 013 0.449 0 . 006                                                 | 0.927 0 . 000 0.415 0 . 016 0.753 0 . 006 0.808 0 . 007 0.504 0 . 009 0.831 0 . 006                     |
| PAS    | AP AUC AVE                      |                                                                                                 | 0.737 0 . 011 5.167                                                                                     | 002 013 010 004 0.727 0 . 012 6                                                                 | 7.167                                                                                                   | 10                                                                                                                               |                                                                                                                                           | 0.444 0 . 002 0.754 0 . 003 4.167                                                                             | 0.454 0 . 007 0.766 0 . 006 3.167                                                                       | 0.761 0 . 013 4.167                                                                                     | 1                                                                                                       |

Comparison Experiment. To validate that our method can adapt to varying degrees of data missing, we conduct comparison experiments with PER and LMR ranging from { 30% , 50% , 70% , 90% } . Table 5 presents the comparison results and algorithm rankings when PER and LMR are fixed at 90%. Fig. 8, 9 and 10 illustrate the distributional trends of all metrics as PER increases from 30% to 70%, with LER fixed at 30%, 50%, and 70%, respectively. It can be observed that our method

outperforms the other nine methods in almost all cases, which demonstrates the robustness of our TDLSR in handling incomplete multi-view multi-label problems. Moreover, the effectiveness of our method is especially evident when dealing with high levels of data unavailability. For instance, as shown in Fig. 8, on Corel 5k, the performance margin of our method over the second-best gradually widens. Additionally, as reported in Table 5, our method achieves more than a 10% improvement in the most representative metric AP on both OBJECT and Pascal07.

Parameter Determination and Convergence. The parameters λ 1 and λ 2 are used to balance the effects of L IB and L le . Two parameters are selected from the range of { 0 . 01 , 0 . 1 , 1 , 10 , 100 , 1000 } and the joint influence are presented in the heatmap as shown in Fig. 6. From the result, the performance of TDLSR exhibits variability under different parameter setting. Besides, the optimal results are typically achieved when λ 1 falls within the range of (0.01, 1). Overall, the parameter sensitivity is relatively low on certain datasets, such as on ESPgame and OBJECT. We also present the simultaneous evolution of training loss and performance metric on the validation set throughout the training process in Fig. 7. The results show that our TDLSR demonstrates great convergence and gradually approaches the optimal network parameters.

Figure 6: Parameter analysis of the trade-off parameters λ 1 and λ 2 .

<!-- image -->

Figure 7: The convergence behavior of our TDLSR during training.

<!-- image -->

Figure 8: Experimental results of ten methods on five datasets with PER varying from 50% to 90% while LMR = 50% .

<!-- image -->

Figure 9: Experimental results of ten methods on five datasets with PER varying from 50% to 90% while LMR = 70% .

<!-- image -->

Figure 10: Experimental results of ten methods on five datasets with PER varying from 50% to 90% while LMR = 90% .

<!-- image -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly describe the algorithm implementation, theoretical contributions, and comprehensive experiments presented in this paper, consistent with the subsequent Sections 2 and 3.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer:[Yes]

Justification: Sections 2 and 5 mention that the model needs to fully separate feature semantics in order to achieve optimal generalization performance, and the prior semantics among multiple labels also need to be leveraged, respectively, which are key challenges in model optimization and design.

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

Justification: The paper provide the full set of assumptions and a complete proof needed for each theoretical result in Section 2 and Appendix.

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

Justification: The network architecture, datasets, model parameters, and training approaches involved in the experimental implementation are described in detail in both the main text and the Appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in

some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [No]

Justification: We commit to sharing our code upon the acceptance of the paper.

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

Justification: Experimental settings, including data splits, hyper parameters, optimizer and learning rate are discussed in both the main text and Appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Fig 3 and Table 1 present the results as the mean and standard deviation over 10 random trials, reflecting the statistical variability.

## Guidelines:

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

Justification: Section 3 mentiones that our model is implemented by PyTorch on one NVIDIA GeForce RTX 4090 GPU of 24GB memory.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research focuses on algorithm design and theoretical analysis. It does not involve human subjects or ethically sensitive applications. We believe it complies with the NeurIPS Code of Ethics.

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

Answer: [No]

Justification: The paper properly cites the sources for existing assets, such as baseline methods and data sources. However, the specific licenses and terms of use for these assets are not explicitly stated in the main text or Appendix.

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

Answer: [No]

Justification: The paper does not release new assets.

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