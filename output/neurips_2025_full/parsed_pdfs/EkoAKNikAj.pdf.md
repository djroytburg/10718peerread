## Polyline Path Masked Attention for Vision Transformer

Zhongchen Zhao 1 , Chaodong Xiao 2 , 3 , Hui Lin 1 , Qi Xie 1 , ∗ , Lei Zhang 2 , 3 , Deyu Meng 1 , 4

1 Xi'an Jiaotong University 2 The Hong Kong Polytechnic University 3 OPPO Research 4 Pazhou Laboratory (Huangpu) Institute

③❤♦♥❣❝❤❡♥③❤❛♦❅/a115/a116✉✳①❥/a116✉✳❡❞✉✳❝♥✱ ①✐❡✳/a113✐❅♠❛✐❧✳①❥/a116✉✳❡❞✉✳❝♥

## Abstract

Global dependency modeling and spatial position modeling are two core issues of the foundational architecture design in current deep learning frameworks. Recently, Vision Transformers (ViTs) have achieved remarkable success in computer vision, leveraging the powerful global dependency modeling capability of the self-attention mechanism. Furthermore, Mamba2 has demonstrated its significant potential in natural language processing tasks by explicitly modeling the spatial adjacency prior through the structured mask. In this paper, we propose Polyline Path Masked Attention ( PPMA ) that integrates the self-attention mechanism of ViTs with an enhanced structured mask of Mamba2, harnessing the complementary strengths of both architectures. Specifically, we first ameliorate the traditional structured mask of Mamba2 by introducing a 2D polyline path scanning strategy and derive its corresponding structured mask, polyline path mask, which better preserves the adjacency relationships among image tokens. Notably, we conduct a thorough theoretical analysis on the structural characteristics of the proposed polyline path mask and design an efficient algorithm for the computation of the polyline path mask. Next, we embed the polyline path mask into the self-attention mechanism of ViTs, enabling explicit modeling of spatial adjacency prior. Extensive experiments on standard benchmarks, including image classification, object detection, and segmentation, demonstrate that our model outperforms previous state-of-the-art approaches based on both state-space models and Transformers. For example, our proposed PPMA-T/S/B models achieve 48.7% / 51.1% / 52.3% mIoU on the ADE20K semantic segmentation task, surpassing RMT-T/S/B by 0.7%/1.3%/0.3%, respectively. Code is available at ❤/a116/a116♣/a115✿✴✴❣✐/a116❤✉❜✳❝♦♠✴③❤♦♥❣❝❤❡♥③❤❛♦✴/a80/a80▼❆ .

## 1 Introduction

The research of foundational models has long been a cornerstone of deep learning. In computer vision, Convolutional Neural Networks (CNNs) [20, 19] and Vision Transformers (ViTs) [46, 9, 31, 22] currently represent the dominant architectures. Notably, ViTs have become the most mainstream architecture in large models through the powerful self-attention mechanism, which can capture the non-local self-similarity within global receptive fields. However, the quadratic complexity of the Transformer, when implementing self-attention, severely limits its application in large image processing models. Moreover, as shown in Fig. 1 (b), classic positional encoding methods [46, 31, 38] in ViTs lack the explicit modeling capability of spatial distance between image tokens, and largely ignore the important spatial adjacency priors in texture, shape, semantics, and so on. This increases the learning pressure and limits its capability for fine-grained image feature extraction.

Compared to CNNs and Transformers, the recently proposed Mamba [11] achieves linear complexity while maintaining global receptive fields, demonstrating strong potential as the next-generation architecture. Specifically, Mamba follows the State Space Models (SSMs) paradigm and employs the selective scan mechanism with the state transition matrix to recursively propagate dependencies among

∗ Corresponding author.

tokens in a sequence. Building on this foundation, Mamba2 [6] further refines the state transition matrix into a lightweight structured mask and introduces a unified theoretical framework, Structured State Space Duality (SSD), to bridge SSMs and attention variants. Under SSD, the core selective scan mechanism of Mamba2 can be reformulated as a form of structured masked attention, i.e. , a Linear Attention [23] element-wise multiplied by the structured mask , as illustrated in Fig. 1 (a). Notably, this structured mask explicitly encodes the sequence adjacency of tokens,

Figure 1: (a)-(b) Illustration of the modules in Mamba2 and ViT. (c) Our method adapts the structured mask of Mamba2 to 2D scanning and integrates it with ViT's self-attention.

<!-- image -->

enabling Mamba2 to match or surpass Transformers across various natural language processing (NLP) tasks. Following its success in NLP, Mamba [11] has been rapidly adapted to various visual domains, including: high-level tasks (classification, object detection, segmentation [58, 30, 49, 47]), low-level tasks (super-resolution, denoising, deraining [13, 12, 59]), image generation [43], video analysis [26], point cloud analysis [27, 51], and remote sensing images [54].

Although Mamba [11] has demonstrated impressive results on certain vision tasks, empirical results on high-level vision tasks demonstrate that even state-of-the-art (SOTA) Mamba-based backbones [30, 49, 47, 15] still underperform SOTA Transformer-based backbones [57, 10] with a substantial performance gap. As shown in Fig. 1 (a), this gap mainly stems from two issues: (I) 1D Scanning Issue. Mamba's 1D scanning strategy arranges the tokens of a 2D image into a 1D sequence, which inevitably disrupts the inherent spatial adjacency within 2D images and limits the effectiveness of its recursive selective scanning mechanism. (II) Weak Global Dependency Modeling Issue. The linear attention in Mamba2 omits the non-linear softmax layer, leading to a decrease in the precision and stability of global dependency modeling of images.

In this paper, we present Polyline Path Masked Attention ( PPMA ), a brand-new method that effectively combines the advantages of ViTs and Mamba2. Specifically, as illustrated in Fig. 1 (c), to address the 1D Scanning Issue when applying current Mamba to 2D images, we propose a novel 2D polyline path scanning strategy and derive an efficient calculation method for its corresponding structured mask, the polyline path mask. Then, we embed the polyline path mask as an explicit positional encoding into the ViT framework. This not only avoids the Weak Global Dependency Modeling Issue of Mamba2, but also alleviates the positional encoding issue of ViTs. As a result, our method fully leverages the powerful global context modeling capability of the self-attention mechanism in ViTs together with the explicit spatial adjacency modeling capability of the polyline path mask inspired by Mamba2, achieving SOTA performance on mainstream high-level vision tasks.

To the best of our knowledge, this is the first work to integrate Mamba2's structured mask mechanism into ViTs. The main contributions of this study are summarized as follows:

- We propose a 2D polyline path scanning strategy for visual Mamba, which better preserves the inherent 2D spatial structure of images compared to existing scanning strategies. Building on this, we further derive a novel structured mask, termed polyline path mask, which is more suitable for 2D images than the traditional structured mask used in Mamba2.
- We conduct a comprehensive theoretical analysis for the proposed 2D polyline path mask. Specifically, we theoretically prove that it can be decomposed into two 1D structured masks with clear physical meanings (i.e., horizontal and vertical scanning masks). More importantly, by leveraging this decomposability, we derive an efficient algorithm to reduce its computational complexity from O ( N 2 ) in the naive calculation to O ( N 3 2 ) .
- The polyline path mask can be seamlessly integrated into various attention variants in a plugand-play manner without introducing a substantial increase in computational complexity. In this paper, we incorporate it into the vanilla self-attention and criss-cross attention, deriving the Polyline Path Masked Attention (PPMA).

- Leveraging PPMA, we construct a hybrid Mamba2-Transformer model. Experimental results demonstrate that our model achieves SOTA performance on standard benchmarks for image classification, object detection, and segmentation.

## 2 Related Work

Vision Transformers. ViTs have become foundational in large-scale vision models such as SAM [24] and Sora [1], primarily due to their self-attention mechanism that effectively captures the long-range dependency. Moreover, the spatial structural information provided by positional encodings (e.g., APE [46], RPE [31], and RoPE [38]) is also crucial to ViTs. However, traditional positional encodings fail to explicitly encode spatial adjacency. Recent works, such as RMT [10] and VVT [39], porpose to incorporate RetNet's input-independent temporal decay mask [40] into ViTs for more explicit spatial modeling based on the Manhattan distance. In comparison, Mamba2's input-dependent selective structured mask not only explicitly encodes the relative positional information in the spatial space but also captures the semantic continuity in the feature space.

Mamba. As a state space model, Mamba introduces an input-dependent selection mechanism into the state transition matrix A , achieving Transformer-level performance with linear complexity on NLP tasks. Building on this foundation, Mamba2 [6] further simplifies the matrix A to a scalar a , enabling more hardware-efficient parallelizable training without sacrificing performance. Moreover, Mamba2 [6] demonstrates that its formulation is mathematically equivalent to a 1-semiseparable structured masked attention, and develops the State Space Duality (SSD) framework to connect structured SSMs and attention variants. Furthermore, Mamba2 points out that other potential structured masked attentions can also be integrated into the SSD framework.

In this paper, we introduce a novel structured masked attention, termed polyline path masked attention, tailored for vision tasks. Different from the previous 2D selective SSM framework [53] based on Mamba [11], our Mamba2-based polyline path mask is more lightweight and can be plugged seamlessly into various attention variants. Moreover, compared to MambaVision [17] which naively concatenates Mamba's blocks and ViT self-attention layers, our method more effectively harnesses complementary strengths of both architectures.

## 3 Preliminaries

Mamba2's Recurrent Form. Mamba2 [6] initially adopts a recurrent form with linear complexity for sequence modeling. Specifically, Mamba2 employs the selective state space models to map the input sequence x ∈ R N × C to the output sequence y ∈ R N × C , i.e., for i =1: N ,

<!-- formula-not-decoded -->

where x i , y i ∈ R 1 × C , h i ∈ R D × C denotes the hidden state, a i ∈ R and B i , C i ∈ R 1 × D are inputdependent parameters learned by multilayer perceptron (MLP) layers, the scalar a i serves as a decay factor bounded in [0 , 1] , N,C and D denote the sequence length, channel number, and hidden state dimension, respectively.

Mamba2's Attention Form. Leveraging the SSD framework in Mamba2 [6], the recurrent form of Mamba2 in Eq. (1) can be reformulated as its equivalent dual form, i.e., structured masked attention, by eliminating the hidden state h i via substitution:

<!-- formula-not-decoded -->

where ⊙ denotes the Hadamard (element-wise) product, B , C ∈ R N × D , and the 1D structured mask L 1 D ∈ R N × N is a 1-semiseparable matrix which can be efficiently calculated with a complexity of O ( N 2 ) by the chunkwise algorithm [6]. Mamba2's attention form (Eq. (2)) enables more efficient parallelizable training than its recurrent form (Eq. (1)). Notably, parameters C and B in Eq. (2) are learned analogously to the query Q and key K in ViTs, respectively. Thus, Eq. (2) reveals that the selective state transition function in Mamba2 is equivalent to the Hadamard product of a linear attention map CB ⊤ and a 1D structured mask L 1 D . Here, the structured mask can be interpreted as a form of relative positional encoding [6].

Figure 2: Compared to existing scanning strategies (a) and (b), which flatten 2D tokens into a 1D sequence, our polyline path scanning (c) better preserves the adjacency of 2D tokens.

<!-- image -->

In this work, we extend the structured masked attention in Mamba2 from 1D sequences to 2D images. Specifically, we extend the 1D structured mask L 1 D to the 2D polyline path mask L 2 D by introducing a novel 2D scanning strategy, and propose an efficient algorithm for computing and applying this polyline path mask L 2 D . The proposed L 2 D can be substituted into Eq. (2) for replacing L 1 D or adopted in ViTs as the explicit positional encoding.

## 4 Method

In this section, we introduce the idea of adapting the structured mask of Mamba2 to 2D scanning and integrating it into the self-attention mechanism of ViTs, achieving an explicit positional encoding. Specifically, we 1) introduce the definition of 2D polyline path mask in Sec. 4.1; 2) analyze the theoretical properties of the proposed polyline path mask and introduce an efficient algorithm for the proposed polyline path mask in Sec. 4.2; 3) apply the polyline path mask to standard self-attention and criss-cross attention of ViTs in Sec. 4.3.

## 4.1 Definition of Polyline Path Mask

As a sequence autoregressive framework, visual Mamba starts from employing a scanning strategy to flatten a 2D image into a 1D sequence of image tokens. This scanning strategy plays an important role in Mamba's performance, since the order of tokens is determined by it. As illustrated in Fig. 2, previous works [58, 30, 27] have proposed various scanning strategies for visual Mamba. However, these strategies fail to fully preserve the inherent spatial adjacency of 2D tokens. For example, as shown in Fig. 2 (a) and (b), for two tokens B and C which are close in an image, previous scanning strategies [30, 27] may cause them to be significantly farther apart in the 1D scanning path.

𝑥1,1 𝑥1,2 𝑥1,3 𝑥1,4 𝛼1,2 𝛼1,3 𝛼1,4 𝛽2,1 𝛽3,1 𝛽4,1 𝛼2,2 𝛼1,3 𝛼2,4 𝛼3,2 𝛼3,3 𝛼3,4 𝛼4,2 𝛼4,3 𝛼4,4 𝛽2,2 𝛽3,2 𝛽4,2 𝛽2,3 𝛽3,3 𝛽4,3 𝛽2,4 𝛽3,4 𝛽4,4 𝑥2,1 𝑥2,2 𝑥2,3 𝑥2,4 𝑥3,1 𝑥3,2 𝑥3,3 𝑥3,4 𝑥4,1 𝑥4,2 𝑥4,3 𝑥4,4 Polyline Path Scanning. To address this limitation, we design a 2D polyline path scanning strategy. Specifically, for each token pair ( x i,j , x k,l ) in the 2D grid, we define their scanning path as the Lshaped polyline connecting them, as shown in Fig. 2 (c). To ensure symmetry in mutual distances, we set two bidirectional polyline paths: vertical-then-horizontal path (V2H solid lines in Fig. 2 (c)) and horizontal-then-vertical path (H2V dotted lines in Fig. 2 (c)), and use their combination as the final scanning path. In this way, the adjacency relationship of 2D tokens can be strictly maintained under the Manhattan distance 2 . Intuitively speaking, tokens close (or far) to each other will be in close (or far) distance on the scanning path, and vice versa. As the example shown in Fig. 2, polyline scanning strategy better preserves the distance between token B and C compared to the other two strategies. An more intuitive example is shown in Fig. 8.

Figure 3: An intuitive example illustrating the polyline path mask on a 4 × 4 grid.

<!-- image -->

Definition of Polyline Path Mask. Based on the proposed polyline path scanning strategy, we introduce the polyline path mask. As an example shown in Fig. 3, we define the horizontal and vertical decay factors of each input token x i,j as α i,j and β i,j , respectively. In this paper, we employ two MLP layers to learn α i,j and β i,j , respectively. 3 Then, the decay weight of V2H polyline path

2 The Manhattan distance between two points x i,j and x k,l in a 2D plane is | i -k | + | j -l | .

3 We apply the ReLU and exponential operator after the MLP layers to ensure α i,j , β i,j ∈ [0 , 1] .

from x k,l to x i,j is defined as L i,j,k,l , which is the product of all decay factors along that path, i.e.,

<!-- formula-not-decoded -->

For example, as illustrated in Fig. 3, the V2H polyline path's decay weight from token x 4 , 4 to x 1 , 1 is L 1 , 1 , 4 , 4 = α 1 , 2 α 1 , 3 α 1 , 4 β 2 , 4 β 3 , 4 β 4 , 4 . Similarly, the decay weight along the H2V polyline path is defined as ˜ L i,j,k,l = α k,j : l β i : k,l . Due to the spatial symmetry, it is evident that ˜ L i,j,k,l = L k,l,i,j . By combining the V2H and H2V polyline paths, the final decay weight is

<!-- formula-not-decoded -->

Note that L , ˜ L and L 2 D are all 4D tensors of size R H × W × H × W , where H and W are the height and width of the feature map, respectively. The polyline path mask, a 2D matrix L 2 D ∈ R HW × HW , can be obtained by unfolding the decay weight tensor, i.e., for all i , j , k , and l ,

<!-- formula-not-decoded -->

For simplicity, we denote the above tensor-to-matrix unfolding operation as L 2 D =unfold( L 2 D ) , and its inverse operation as L 2 D =fold( L 2 D ) in the following sections. More details can be found in Appendix A.2.

## 4.2 Efficient Computation Theory of Polyline Path Mask

According to the definition (3), the direct approach to compute the polyline path mask L 2 D is to calculate each element individually. However, the large size of the mask and numerous multiplications for each element lead to a high computational cost in both calculating and applying L 2 D . To address this issue, we present a decomposition theorem for matrices structured as L 2 D . Based on this, we further design an efficient algorithm for performing multiplication on L 2 D . For simplicity, we focus our theoretical study on L , which is similar to the case of L 2 D . Complete proofs of the theorems are provided in Appendix A.3 and A.5.

Theorem 1 (Matrix Decomposition) . For any matrix M ∈ R HW × HW and M = fold ( M ) , if for ∀ i, j, k, l , ∃ A i ∈ R W × W and B l ∈ R H × H , s.t., M i,j,k,l = [ A i ] j,l × [ B l ] i,k , then M can be decomposed as:

<!-- formula-not-decoded -->

where M A , M B , ˆ M A , ˆ M B ∈ R HW × HW , which satisfy

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

As defined in Eq. (3), the polyline path mask L satisfies the conditions in Theorem 1 with [ A i ] j,l = α i,j : l and [ B l ] i,k = β i : k,l . Thus, based on Theorem 1, the polyline path mask L can be decomposed as L = L H × L V = ˆ L H ⊙ ˆ L V . Moreover, for the complexity of computing L , we have:

Corollary 1 (Mask Complexity) . The complexity of directly computing polyline path mask L with Eq. (3) and (5) is O ( N 5 2 ) , which can be reduced to O ( N 2 ) by applying Theorem 1, where N = H × W .

For matrices in the form of Eq. (7), when performing multiplication operations, we have:

Theorem 2 (Efficient Matrix Multiplication) . For matrices M A , M B defined in Eq. (7) , ∀ x ∈ R HW , the following equation holds:

<!-- formula-not-decoded -->

where y ∈ R HW , X =unvec( x ) ∈ R H × W , Y =unvec( y ) ∈ R H × W , Z ∈ R H × W , and the operator vec( · ) vectorizes a matrix by stacking its columns and unvec( · ) is its inverse operator.

Figure 4: Illustration of the efficient algorithm for utilizing the proposed polyline path mask. Left: Naive computation of matrix multiplication. Right: An intuitive illustration of Algorithm 1.

<!-- image -->

Based on Theorem 2, we can design Algorithm 1 for computing the matrix multiplication between polyline path mask L and the vector x . Note that the involved matrices A i and B l are symmetric matrices with lower triangular parts being 1-semiseparable, as defined in Mamba2 [6]. This will lead to a substantial reduction in complexity as stated in the following corollary.

## Algorithm 1: Efficient Masked Attention Computation.

Input: decay factors α, β of L , vector x ∈ R HW ;

- 1: Compute X =unvec( x ) ∈ R H × W ;
- 2: Compute B l ∈ R H × H , where for l =1: W, [ B l ] i,k = β i : k,l ;
- 3: Compute Z ∈ R H × W , where Z : ,l = B l × X : ,l ;
- 4: Compute A i ∈ R W × W , where for i =1: H, [ A i ] j,l = α i,j : l ;
- 5: Compute Y ∈ R H × W , where Y i, : = A i × Z i, : ;

Output:

y =vec( Y )

;

Corollary 2 (Masked Attention Complexity) . The computational complexity of the matrix multiplication between polyline path mask and vector x , i.e., y = Lx , can be reduced from O ( N 2 ) to O ( N 3 2 ) by Algorithm 1, and further reduced to O ( N ) by applying the chunkwise algorithm of Mamba2 [6] to steps 3 and 5 in Algorithm 1.

Remarks. Intuitively, as illustrated in Fig. 4, Algorithm 1 shows that the 2D polyline path scanning on 2D tokens (i.e., Lx ) can be decomposed as the 1D vertical scanning along each column of X (i.e., Z : ,l = B l × X : ,l ) followed by the 1D horizontal scanning along each row of Z (i.e., Y i, : = A i × Z i, : ). This equivalence offers an intuitive understanding of the physical meaning of the decomposed polyline path mask L = L H L V and enables its natural extension to 3D or higher-dimensional tokens, as detailed in Appendix C.2.

## 4.3 Polyline Path Masked Attention

The proposed polyline path mask can be seamlessly integrated into various attention variants in a plug-and-play manner. In this section, we integrate it into two softmax-based self-attention layers: vanilla attention [9] and criss-cross attention [22]. Notably, theorems and algorithm given in Sec. 4.2 guarantee that integration of polyline path mask does not substantially increase the computational complexity of the original attention mechanism. More applications, such as the polyline path masked linear attention with a complexity of O ( N ) , are provided in Appendix A.7.

Polyline Path Masked Vanilla Attention. The polyline path mask L 2 D is integrated into vanilla attention via a Hadamard product with the attention map, i.e., for query Q , key K , and value V :

<!-- formula-not-decoded -->

where Q , K , V ∈ R HW × C . Based on Corollary 1, Eq. (10) maintains the complexity of O ( N 2 ) .

Polyline Path Masked Criss-Cross Attention. The original criss-cross attention [22] employs the sparse attention over tokens located in the same row or column, achieving a complexity of O ( N 3 2 ) . In this work, we follow RMT [10] to decompose criss-cross attention into the vertical attention over each column followed by the horizontal attention over each row. The polyline path mask L 2 D is applied to the decomposed criss-cross attention through the Hadamard product, that is:

<!-- formula-not-decoded -->

where horizontal and vertical attention maps S H , S V ∈ R HW × HW satisfy the form in Eq. (7) with A i =softmax( Q i, : , : K ⊤ i, : , : ) and B l =softmax( Q : ,l, : K ⊤ : ,l, : ) , and Q , K ∈ R H × W × C are tensor forms

Figure 5: Overall architecture of the Polyline Path Masked Attention based Vision Transformer.

<!-- image -->

of Q , K , respectively [22]. Based on Theorem 1, we can reformulate the left part of Eq. (11) as:

<!-- formula-not-decoded -->

Note that matrices ˆ S H ⊙ ˆ L H and ˆ S V ⊙ ˆ L V also satisfy the form in Eq. (7). Thus, the computational complexity of Eq. (12) can be reduced to O ( N 3 2 ) by Algorithm 1. Similar conclusions can also be derived for the right part of Eq. (11). Thus, the complexity of Eq.(11) maintains O ( N 3 2 ) .

## 4.4 Overall Architecture

Based on the proposed Polyline Path Masked Attention, we construct a hybrid Mamba2-Transformer backbone for vision tasks. As illustrated in Fig. 5, our backbone adopts the four-stage hierarchical architecture. Following RMT [10], we employ Polyline Path Masked Criss-Cross Attention in the first three stages, and Polyline Path Masked Vanilla Attention in the final stage. Moreover, we develop our model in three scales: tiny (PPMA-T), small (PPMA-S), and base (PPMA-B).

## 5 Experiments

To validate the effectiveness of our method, we conduct a series of experiments on mainstream benchmarks for image classification (Sec. 5.1), object detection and instance segmentation (Sec. 5.2), and semantic segmentation (Sec. 5.3). Comparison methods include advanced CNN-based [34, 32, 42], SSM-based [30, 53, 49, 47], and Transformer-based backbones [31, 8, 17, 16, 57, 10]. For a fair comparison, we reproduce the experimental results of RMT [10] with the same experimental settings as ours. We also perform comprehensive ablation studies on the structured mask design in Sec. 5.4. More detailed experimental settings and results can be found in Appendix B.

## 5.1 Image Classification on ImageNet-1K

Settings. We evaluate the classification performance of our method on ImageNet-1K [7]. Following the same training strategy as in [10, 44], we train our models from scratch for 300 epochs with the input size of 224 × 224. We use the adaptive AdamW optimizer with a cosine decay learning rate scheduler (batch size=1024, initial learning rate=0.001, weight decay=0.05).

Results. The comparison results presented in Table 1 show that our method achieves state-of-the-art (SOTA) performance compared to other advanced models based on various architectures across tiny, small, and base scales. Specifically, PPMA-S achieves 84.2% top-1 accuracy, surpassing 2DMambaT [53] by 1.4% , MLLA-T [15] by 0.7% , MambaVision-T2 [17] by 1.5% , and RMT-S [10] by 0.2% with similar FLOPs. PPMA-T achieves 82.6% top-1 accuracy, outperforming the most competitive RMT-T [10] by 0.2% without extra training tricks. Moreover, our PPMA-B also surpasses other SOTA CNN-based, SSM-based, and Transformer-based backbones.

Table 1: Image classification performance on the ImageNet-1K validation set.

| Model                            | Arch.   |   #Param. (M) |   FLOPs (G) |   Top-1 (%) | Model                      | Arch.   |   #Param. (M) |   FLOPs (G) |   Top-1 (%) |
|----------------------------------|---------|---------------|-------------|-------------|----------------------------|---------|---------------|-------------|-------------|
| RegNetY-1.6G [34] EffNet-B3 [42] | CNN     |            11 |         1.6 |        78   | NAT-T [16] BiFormer-S [57] |         |            28 |         4.3 |        83.2 |
|                                  |         |            12 |         1.8 |        81.6 |                            | Trans.  |            26 |         4.5 |        83.8 |
| Vim-T [58]                       |         |             7 |         1.5 |        76.1 | RMT-S [10]                 |         |            27 |         4.5 |        84   |
| MSVMamba-M [36]                  | SSM     |            12 |         1.5 |        79.8 | PPMA-S                     |         |            27 |         4.9 |        84.2 |
| BiFormer-T [57] NAT-M [16]       | Trans.  |            13 |         2.2 |        81.4 | RegNetY-8G [34]            | CNN     |            39 |         8   |        81.7 |
|                                  | Trans.  |            20 |         2.7 |        81.8 | ConvNeXt-S [32]            |         |            50 |         8.7 |        83.1 |
| SMT-T [29]                       | Trans.  |            12 |         2.4 |        82.2 | EffNet-B5 [42]             |         |            30 |         9.9 |        83.6 |
| RMT-T [10]                       |         |            14 |         2.5 |        82.4 | VMamba-S [30]              | SSM     |            50 |         8.7 |        83.6 |
| PPMA-T                           |         |            14 |         2.7 |        82.6 | 2DMamba-S [53]             |         |            50 |         8.8 |        83.8 |
| RegNetY-4G [34]                  |         |            21 |         4   |        80   | GrootVL-S [49]             |         |            51 |         8.5 |        84.2 |
| ConvNeXt-T [32]                  | CNN     |            29 |         4.5 |        82.1 | MLLA-S [15]                |         |            43 |         7.3 |        84.4 |
| EffNet-B4 [42]                   |         |            19 |         4.2 |        82.9 | Spatial-Mamba-S [47]       |         |            43 |         7.1 |        84.6 |
| VMamba-T [30]                    |         |            30 |         4.9 |        82.6 | Swin-S [31]                | Trans.  |            50 |         8.7 |        83   |
| 2DMamba-T [53]                   |         |            31 |         4.9 |        82.8 | NAT-S [16]                 |         |            51 |         7.8 |        83.7 |
| GrootVL-T [49]                   | SSM     |            30 |         4.8 |        83.4 | CSWin-B [8]                |         |            78 |        15   |        84.2 |
| Spatial-Mamba-T [47]             |         |            27 |         4.5 |        83.5 | MambaVision-B [17]         |         |            98 |        15   |        84.2 |
| MLLA-T [15]                      |         |            25 |         4.2 |        83.5 | BiFormer-B [57]            |         |            57 |         9.8 |        84.3 |
| Swin-T [31]                      |         |            29 |         4.5 |        82.1 | iFormer-B [37]             |         |            48 |         9.4 |        84.6 |
| CSWin-T [8]                      | Trans.  |            23 |         4.3 |        82.7 | RMT-B [10]                 |         |            54 |         9.7 |        84.9 |
| MambaVision-T2 [17]              |         |            35 |         5.1 |        82.7 | PPMA-B                     |         |            54 |        10.6 |        85   |

Table 2: Object detection and instance segmentation performance with Mask R-CNN [18] detector on COCO val2017. FLOPs are calculated with input resolution of 1280 × 800 .

| Mask R-CNN 1 × schedule   | Mask R-CNN 1 × schedule   | Mask R-CNN 1 × schedule   | Mask R-CNN 1 × schedule   | Mask R-CNN 1 × schedule   | Mask R-CNN 1 × schedule   | Mask R-CNN 1 × schedule   | Mask R-CNN 1 × schedule   | Mask R-CNN 1 × schedule   |
|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|
| Backbone                  | #Param. (M)               | FLOPs (G)                 | AP b                      | AP b 50                   | AP b 75                   | AP m                      | AP m 50                   | AP m 75                   |
| Vim-T [58]                | -                         | -                         | 45.7                      | 63.9                      | 49.6                      | 39.2                      | 60.9                      | 41.7                      |
| MSVMamba-M [36]           | 32                        | 201                       | 43.8                      | 65.8                      | 47.7                      | 39.9                      | 62.9                      | 42.9                      |
| MPViT-XS [25]             | 30                        | 231                       | 44.2                      | 66.7                      | 48.4                      | 40.4                      | 63.4                      | 43.4                      |
| RMT-T [10]                | 33                        | 218                       | 46.7                      | 68.6                      | 51.6                      | 42.1                      | 65.3                      | 45.2                      |
| PPMA-T                    | 33                        | 218                       | 47.1                      | 68.7                      | 51.7                      | 42.4                      | 65.9                      | 45.7                      |
| ResNet-50 [19]            | 44                        | 260                       | 38.2                      | 58.8                      | 41.4                      | 34.7                      | 55.7                      | 37.2                      |
| ConvNeXt-T [32]           | 48                        | 262                       | 44.2                      | 66.6                      | 48.3                      | 40.1                      | 63.3                      | 42.8                      |
| MLLA-T [15]               | 44                        | 255                       | 46.8                      | 69.5                      | 51.5                      | 42.1                      | 66.4                      | 45.0                      |
| GrootVL-T [49]            | 49                        | 265                       | 47.0                      | 69.4                      | 51.5                      | 42.7                      | 66.4                      | 46.0                      |
| VMamba-T [30]             | 50                        | 271                       | 47.3                      | 69.3                      | 52.0                      | 42.7                      | 66.4                      | 45.9                      |
| Spatial-Mamba-T [47]      | 46                        | 216                       | 47.6                      | 69.6                      | 52.3                      | 42.9                      | 66.5                      | 46.2                      |
| Swin-T [31]               | 48                        | 267                       | 43.7                      | 66.6                      | 47.7                      | 39.8                      | 63.3                      | 42.7                      |
| CSWin-T [8]               | 42                        | 279                       | 46.7                      | 68.6                      | 51.3                      | 42.2                      | 65.6                      | 45.4                      |
| BiFormer-S [57]           | -                         | -                         | 47.8                      | 69.8                      | 52.3                      | 43.2                      | 66.8                      | 46.5                      |
| RMT-S [10]                | 46                        | 262                       | 48.8                      | 70.8                      | 53.4                      | 43.6                      | 67.4                      | 47.3                      |
| PPMA-S                    | 46                        | 263                       | 49.2                      | 70.7                      | 54.0                      | 43.8                      | 67.4                      | 47.1                      |
| ResNet-101 [19]           | 63                        | 336                       | 40.4                      | 61.1                      | 44.2                      | 36.4                      | 57.7                      | 38.8                      |
| ConvNeXt-S [32]           | 70                        | 348                       | 45.4                      | 67.9                      | 50.0                      | 41.8                      | 65.2                      | 45.1                      |
| GrootVL-S [49]            | 70                        | 341                       | 48.6                      | 70.3                      | 53.5                      | 43.6                      | 67.5                      | 47.1                      |
| VMamba-S [30]             | 70                        | 349                       | 48.7                      | 70.0                      | 53.4                      | 43.7                      | 67.3                      | 47.0                      |
| Spatial-Mamba-S [47]      | 63                        | 315                       | 49.2                      | 70.8                      | 54.2                      | 44.0                      | 67.9                      | 47.5                      |
| MLLA-S [15]               | 63                        | 319                       | 49.2                      | 71.5                      | 53.9                      | 44.2                      | 68.5                      | 47.2                      |
| Swin-S [31]               | 69                        | 359                       | 45.7                      | 67.9                      | 50.4                      | 41.1                      | 64.9                      | 44.2                      |
| CSWin-S [8]               | 54                        | 342                       | 47.9                      | 70.1                      | 52.6                      | 43.2                      | 67.1                      | 46.2                      |
| BiFormer-B [57]           | -                         | -                         | 48.6                      | 70.5                      | 53.8                      | 43.7                      | 67.6                      | 47.1                      |
| RMT-B [10]                | 73                        | 373                       | 50.7                      | 72.0                      | 55.7                      | 45.1                      | 69.2                      | 49.0                      |
| PPMA-B                    | 73                        | 374                       | 51.1                      | 72.5                      | 55.9                      | 45.5                      | 69.7                      | 49.1                      |
| Mask R-CNN 3 × schedule   | Mask R-CNN 3 × schedule   | Mask R-CNN 3 × schedule   | Mask R-CNN 3 × schedule   | Mask R-CNN 3 × schedule   | Mask R-CNN 3 × schedule   | Mask R-CNN 3 × schedule   | Mask R-CNN 3 × schedule   | Mask R-CNN 3 × schedule   |
| ConvNeXt-S [32]           | 70                        | 348                       | 47.9                      | 70.0                      | 52.7                      | 42.9                      | 66.9                      | 46.2                      |
| GrootVL-S [49]            | 70                        | 341                       | 50.1                      | 71.2                      | 54.9                      | 44.6                      | 68.7                      | 47.8                      |
| VMamba-S [30]             | 70                        | 349                       | 49.9                      | 70.9                      | 54.7                      | 44.2                      | 68.2                      | 47.7                      |
| MLLA-S [15]               | 63                        | 319                       | 50.5                      | 71.8                      | 55.2                      | 44.9                      | 69.1                      | 48.2                      |
| Spatial-Mamba-S [47]      | 63                        | 315                       | 50.6                      | 71.5                      | 55.4                      | 44.7                      | 68.6                      | 48.2                      |
| NAT-S [16]                | 70                        | 330                       | 48.4                      | 69.8                      | 53.2                      | 43.2                      | 66.9                      | 46.4                      |
| Swin-S [31]               | 69                        | 359                       | 48.5                      | 70.2                      | 53.5                      | 43.3                      | 67.3                      | 46.6                      |
| CSWin-S [8]               | 54                        | 342                       | 50.0                      | 71.3                      | 54.7                      | 44.5                      | 68.4                      | 47.7                      |
| RMT-B [10]                | 73                        | 373                       | 52.2                      | 72.9                      | 57.0                      | 46.1                      | 70.4                      | 49.9                      |
| PPMA-B                    | 73                        | 374                       | 52.6                      | 73.3                      | 57.5                      | 46.3                      | 70.3                      | 50.2                      |

Table 3: Semantic segmentation performance with UPerNet [48] segmentor on ADE20K val set. 'SS' and 'MS' represent single-scale and multi-scale testing, respectively.

| Backbone             | #Param. (M)   | FLOPs (G)   | mIoU(%)   | mIoU(%)   | Backbone             | #Param. (M)   | FLOPs (G)   | mIoU(%)   | mIoU(%)   |
|----------------------|---------------|-------------|-----------|-----------|----------------------|---------------|-------------|-----------|-----------|
| Backbone             | #Param. (M)   | FLOPs (G)   | SS        | MS        | Backbone             | #Param. (M)   | FLOPs (G)   | SS        | MS        |
| LocalVim-T [21]      | 36            | 181         | 43.4      | 44.4      | BiFormer-S [57]      | -             | -           | 49.8      | 50.8      |
| MSVMamba-M [36]      | 42            | 875         | 45.1      | 45.4      | RMT-S [10]           | 56            | 937         | 49.8      | 49.7      |
| NAT-M [16]           | 50            | 900         | 45.1      | 46.4      | PPMA-S               | 56            | 984         | 51.1      | 52.0      |
| RMT-T [10]           | 43            | 977         | 48.0      | 48.8      | ResNet-101 [19]      | 85            | 1030        | 42.9      | 44.0      |
| PPMA-T               | 43            | 983         | 48.7      | 49.1      | ConvNeXt-S [32]      | 82            | 1027        | 48.7      | 49.6      |
| ResNet-50 [19]       | 67            | 953         | 42.1      | 42.8      | VMamba-S [30]        | 82            | 1028        | 50.6      | 51.2      |
| ConvNeXt-T [32]      | 60            | 939         | 46.0      | 46.7      | Spatial-Mamba-S [47] | 73            | 992         | 50.6      | 51.4      |
| VMamba-T [30]        | 62            | 949         | 48.0      | 48.8      | GrootVL-S [49]       | 82            | 1019        | 50.7      | 51.7      |
| 2DMamba-T [53]       | 62            | 950         | 48.6      | 49.3      | Swin-S [31]          | 81            | 1039        | 47.6      | 49.5      |
| GrootVL-T [49]       | 60            | 941         | 48.5      | 49.4      | NAT-S [16]           | 82            | 1010        | 48.0      | 49.5      |
| Spatial-Mamba-S [47] | 57            | 936         | 48.6      | 49.4      | MambaVision-S [17]   | 84            | 1135        | 48.2      | -         |
| Swin-T [31]          | 60            | 945         | 44.4      | 45.8      | CSWin-S [8]          | 65            | 1027        | 50.4      | 51.5      |
| MambaVision-T [17]   | 55            | 945         | 46.6      | -         | BiFormer-B [57]      | -             | -           | 51.0      | 51.7      |
| NAT-T [16]           | 58            | 934         | 47.1      | 48.4      | RMT-B [10]           | 83            | 1051        | 52.0      | 52.1      |
| CSWin-S [8]          | 60            | 959         | 49.3      | 50.7      | PPMA-B               | 83            | 1137        | 52.3      | 53.0      |

## 5.2 Object Detection and Instance Segmentation on COCO

Settings. We evaluate our method for object detection and instance segmentation tasks on MSCOCO2017 [28] using the MMDetection library [2]. Following previous work [35], we initialize the backbone with ImageNet-1K pretrained weights and adopt Mask R-CNN [18] as the basic framework. The models are trained for 12 epochs (1 × schedule) and 36 epochs with multi-scale inputs (3 × schedule) using AdamW optimizer (batch size=16, learning rate=0.0001, weight decay=0.05).

Results. The results presented in Table 2 show that our model outperforms existing methods on most evaluation metrics. Under the same experimental settings, PPMA-T achieves a box mAP of 47.1% and a mask mAP of 42.4% , surpassing the SOTA Transformer-based backbone RMT-T [10] by 0.4% and 0.3% in the 1 × schedule, respectively. Moreover, PPMA-B achieves a box mAP of 51.1% and a mask mAP of 45.5% , surpassing the SOTA SSM-based backbone MLLA-S [15] by 1.9% and 1.3% in the 1 × schedule, respectively. Furthermore, PPMA-B maintains its superior performance under the 3× multi-scale training schedule.

## 5.3 Semantic Segmentation on ADE20K

Settings. We evaluate the semantic segmentation performance of our method on ADE20K [56] using the MMSegmentation library [4]. Following the settings in previous works [35], we initialize the backbone with ImageNet-1K pretrained weights and adopt UPerNet [48] as the basic framework. The input size of images is set to 512 × 512 and all models are trained for 160K iterations with AdamW optimizer (batch size=16, learning rate= 6 × 10 -5 , weight decay=0.05).

Results. The semantic segmentation results are summarized in Table 3. Our method consistently outperforms previous methods under all settings. Compared to SOTA Transformer-based counterparts, PPMA-T/S/B surpass RMT-T/S/B by 0.7% / 1.3% / 0.3% mIoU in the Single-Scale (SS) setting and 0.3% / 2.3% / 0.9% mIoU in the Multi-Scale (MS) setting. Compared to SOTA SSM-based methods, PPMA-T/S/B surpass them by at least 3.6% / 2.5% / 1.6% in SS mIoU, respectively.

## 5.4 Ablation Study

Polyline Path Mask Design. To verify the effectiveness of the proposed polyline path mask, we conduct an ablation study on ImageNet-1K and ADE20K using PPMA-T as the backbone. Under the same experimental settings, we compare various structured masks embedded into the softmax-based self-attention layers by Hadamard product, including: no mask (baseline), RMT decay mask [10], cross scan mask [30], Hilbert scan mask [27], V2H polyline path mask, and our final 2D polyline path mask. As shown in Fig. 6, our polyline path mask L 2 D , compared to the RMT decay mask, can selectively capture the semantic continuity in the image. Compared to the cross scan mask and Hilbert scan mask, the polyline path mask better preserves the spatial relationships between 2D tokens, alleviating the long-range forgetting issue. Experimental results in Table 4 show that the 2D

Table 4: Ablation study of structured mask designs in PPMA-T on ImageNet-1K and ADE20K.

̸

(b) W/o Mask

(d) Cross Scan Mask

(e) Hilbert Scan Mask

口

(c) RMT Decay Mask

Figure 6: Illustration of various structured masks.

<!-- image -->

Figure 7: Visualizations of the decay factors and the polyline path masked attention maps of the well-trained PPMA model. In each input image, the query token is marked by a red box.

polyline path mask L 2 D boosts the baseline by 0.32% top-1 accuracy on ImageNet-1K and 0.95% SS mIoU on ADE20K, respectively. Visualization results in Fig. 7 further demonstrate that our 2D polyline path mask L 2 D effectively suppresses the falsely highlighted areas in the original attention maps. More visualizations and detailed discussions are provided in Fig. 12 and Sec. C.1.

̸

Horizontal and Vertical Decay Factors. In our model, we employ different decay factors ( α i,j = β i,j ) to capture semantic similarity between adjacent tokens along horizontal and vertical directions, respectively. As illustrated in Fig. 7 (b) and (c), the learned decay factors α and β effectively capture semantic continuity in horizontal and vertical directions, respectively. Table 4 shows that replacing different decay factors with a shared decay factor ( α i,j = β i,j ) results in a significant performance drop, highlighting the importance of modeling horizontal and vertical decay factors separately.

## 6 Conclusion

In this paper, we argue that the key component of Mamba2 model is its structured mask, which explicitly encodes the spatial distance information through the recursive propagation mechanism and captures the semantic continuity in sequences through the selective mechanism. Building on this insight, we propose to extend the structured mask from 1D text sequences to 2D images. To this end, we propose a novel 2D polyline path scanning strategy with its corresponding structured mask tailed for images. To achieve SOTA performance on high-level vision tasks, we integrate the polyline path mask into the powerful self-attention mechanism of ViTs.

Limitations. Although the proposed efficient algorithm optimizes the integration complexity, it inevitably incurs additional GPU memory occupation and lower throughput, as shown in Table 4. We plan to alleviate this limitation through further engineering optimizations, such as CUDA-based or Triton-based implementations, in the future work.

| Structured Mask                           |   #Param. (M) FLOPs (G) Throughput (imgs/s) Top-1 (%) mIoU SS (%) |   #Param. (M) FLOPs (G) Throughput (imgs/s) Top-1 (%) mIoU SS (%) |   #Param. (M) FLOPs (G) Throughput (imgs/s) Top-1 (%) mIoU SS (%) |   #Param. (M) FLOPs (G) Throughput (imgs/s) Top-1 (%) mIoU SS (%) |   #Param. (M) FLOPs (G) Throughput (imgs/s) Top-1 (%) mIoU SS (%) |
|-------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------|
| Baseline (w/o mask)                       |                                                             14.33 |                                                              2.65 |                                                              1779 |                                                             82.28 |                                                             47.78 |
| + RMT Decay Mask                          |                                                             14.33 |                                                              2.65 |                                                              1650 |                                                             82.35 |                                                             48.01 |
| + Cross Scan Mask                         |                                                             14.34 |                                                              2.71 |                                                              1100 |                                                             82.44 |                                                             48.14 |
| + Hilbert Scan Mask                       |                                                             14.34 |                                                              2.71 |                                                              1091 |                                                             82.44 |                                                             48.14 |
| + V2H Polyline Path Mask                  |                                                             14.34 |                                                              2.71 |                                                              1203 |                                                             82.44 |                                                             48.57 |
| + 2D Polyline Path Mask                   |                                                             14.34 |                                                              2.71 |                                                              1034 |                                                             82.6  |                                                             48.73 |
| Shared Decay factors ( α i,j = β i,j )    |                                                             14.33 |                                                              2.71 |                                                              1124 |                                                             82.37 |                                                             48.27 |
| Different Decay factors ( α i,j = β i,j ) |                                                             14.34 |                                                              2.71 |                                                              1034 |                                                             82.6  |                                                             48.73 |

1.0

0.8

0.6

0.4

- 0.2

- 0.0

(a) Input Image

(f) Polyline Path Mask

## Acknowledgments

We would like to thank all anonymous reviewers for their constructive suggestions for improving this paper. This work was supported in part by the National Key R&amp;D Program of China under Grant 2024YFA1012000; in part by the Major Key Project of PCL under Grant PCL2024A06; in part by Tianyuan Fund for Mathematics of the National Natural Science Foundation of China (Grant No. 12426105); in part by the China NSFC projects under contract 62476214.

## References

- [1] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog , 1:8, 2024.
- [2] Kai Chen, Jiaqi Wang, Jiangmiao Pang, et al. MMDetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:1906.07155 , 2019.
- [3] Xiangxiang Chu, Zhi Tian, Bo Zhang, Xinlong Wang, and Chunhua Shen. Conditional positional encodings for vision transformers. In The Twelfth International Conference on Learning Representations , 2023.
- [4] MMSegmentation Contributors. Mmsegmentation, an open source semantic segmentation toolbox, 2020.
- [5] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, et al. Randaugment: Practical automated data augmentation with a reduced search space. In CVPRW , 2020.
- [6] Tri Dao and Albert Gu. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. In Proceedings of the IEEE international conference on computer vision , 2024.
- [7] Jia Deng, Wei Dong, Richard Socher, et al. Imagenet: A large-scale hierarchical image database. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2009.
- [8] Xiaoyi Dong, Jianmin Bao, Dongdong Chen, et al. Cswin transformer: A general vision transformer backbone with cross-shaped windows. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2022.
- [9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In The Tenth International Conference on Learning Representations , 2021.
- [10] Qihang Fan, Huaibo Huang, Mingrui Chen, Hongmin Liu, and Ran He. Rmt: Retentive networks meet vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 5641-5651, 2024.
- [11] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752 , 2023.
- [12] Hang Guo, Yong Guo, Yaohua Zha, Yulun Zhang, Wenbo Li, Tao Dai, Shu-Tao Xia, and Yawei Li. Mambairv2: Attentive state space restoration. arXiv preprint arXiv:2411.15269 , 2024.
- [13] Hang Guo, Jinmin Li, Tao Dai, Zhihao Ouyang, Xudong Ren, and Shu-Tao Xia. Mambair: A simple baseline for image restoration with state-space model. In Proceedings of the European conference on computer vision , pages 222-241. Springer, 2024.
- [14] Jianyuan Guo, Kai Han, Han Wu, Chang Xu, Yehui Tang, Chunjing Xu, and Yunhe Wang. Cmt: Convolutional neural networks meet vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2022.
- [15] Dongchen Han, Ziyi Wang, Zhuofan Xia, Yizeng Han, Yifan Pu, Chunjiang Ge, Jun Song, Shiji Song, Bo Zheng, and Gao Huang. Demystify mamba in vision: A linear attention perspective. arXiv preprint arXiv:2405.16605 , 2024.
- [16] Ali Hassani, Steven Walton, Jiachen Li, Shen Li, and Humphrey Shi. Neighborhood attention transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2023.
- [17] Ali Hatamizadeh and Jan Kautz. Mambavision: A hybrid mamba-transformer vision backbone. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2025.

- [18] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross B. Girshick. Mask r-cnn. In Proceedings of the IEEE international conference on computer vision , 2017.
- [19] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Sun Jian. Deep residual learning for image recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2016.
- [20] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 4700-4708, 2017.
- [21] Tao Huang, Xiaohuan Pei, Shan You, Fei Wang, Chen Qian, and Chang Xu. Localmamba: Visual state space model with windowed selective scan. arXiv preprint arXiv:2403.09338 , 2024.
- [22] Zilong Huang, Xinggang Wang, Yunchao Wei, Lichao Huang, Humphrey Shi, Wenyu Liu, and Thomas S. Huang. Ccnet: Criss-cross attention for semantic segmentation. IEEE transactions on pattern analysis and machine intelligence , pages 1-1, 2020.
- [23] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In Proceedings of the IEEE international conference on computer vision , pages 5156-5165, 2020.
- [24] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE international conference on computer vision , pages 4015-4026, 2023.
- [25] Youngwan Lee, Jonghee Kim, Jeffrey Willette, and Sung Ju Hwang. Mpvit: Multi-path vision transformer for dense prediction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2022.
- [26] Kunchang Li, Xinhao Li, Yi Wang, Yinan He, Yali Wang, Limin Wang, and Yu Qiao. Videomamba: State space model for efficient video understanding. In Proceedings of the European conference on computer vision , pages 237-255. Springer, 2024.
- [27] Dingkang Liang, Xin Zhou, Wei Xu, Xingkui Zhu, Zhikang Zou, Xiaoqing Ye, Xiao Tan, and Xiang Bai. Pointmamba: A simple state space model for point cloud analysis. In Advances in Neural Information Processing Systems , 2024.
- [28] Tsung-Yi Lin, Michael Maire, Serge Belongie, et al. Microsoft coco: Common objects in context. In Proceedings of the European conference on computer vision , 2014.
- [29] Weifeng Lin, Ziheng Wu, Jiayu Chen, Jun Huang, and Lianwen Jin. Scale-aware modulation meet transformer. In Proceedings of the IEEE international conference on computer vision , 2023.
- [30] Yue Liu, Yunjie Tian, Yuzhong Zhao, Hongtian Yu, Lingxi Xie, Yaowei Wang, Qixiang Ye, Jianbin Jiao, and Yunfan Liu. Vmamba: Visual state space model. In Advances in Neural Information Processing Systems , pages 103031-103063, 2024.
- [31] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE international conference on computer vision , 2021.
- [32] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, et al. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2022.
- [33] Boris T Polyak and Anatoli B Juditsky. Acceleration of stochastic approximation by averaging. arXiv preprint arXiv:1906.07155 , 2019.
- [34] Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dollár. Designing network design spaces. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2020.
- [35] Dai Shi. Transnext: Robust foveal visual perception for vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 17773-17783, 2024.
- [36] Yuheng Shi, Minjing Dong, and Chang Xu. Multi-scale vmamba: Hierarchy in hierarchy visual state space model. arXiv preprint arXiv:2405.14174 , 2024.
- [37] Chenyang Si, Weihao Yu, Pan Zhou, Yichen Zhou, Xinchao Wang, and Shuicheng YAN. Inception transformer. In Advances in Neural Information Processing Systems , 2022.

- [38] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing , 568:127063, 2024.
- [39] Weixuan Sun, Zhen Qin, Hui Deng, Jianyuan Wang, Yi Zhang, Kaihao Zhang, Nick Barnes, Stan Birchfield, Lingpeng Kong, and Yiran Zhong. Vicinity vision transformer. IEEE Transactions on Pattern Analysis and Machine Intelligence , 45(10):12635-12649, 2023.
- [40] Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive network: A successor to Transformer for large language models. ArXiv , abs/2307.08621, 2023.
- [41] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 2818-2826, 2016.
- [42] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In Proceedings of the IEEE international conference on computer vision , 2019.
- [43] Yao Teng, Yue Wu, Han Shi, Xuefei Ning, Guohao Dai, Yu Wang, Zhenguo Li, and Xihui Liu. Dim: Diffusion mamba for efficient high-resolution image synthesis. arXiv preprint arXiv:2405.14224 , 2024.
- [44] Hugo Touvron, Matthieu Cord, Matthijs Douze, et al. Training data-efficient image transformers &amp; distillation through attention. In Proceedings of the IEEE international conference on computer vision , 2021.
- [45] Zhengzhong Tu, Hossein Talebi, Han Zhang, Feng Yang, Peyman Milanfar, Alan Bovik, and Yinxiao Li. Maxvit: Multi-axis vision transformer. In Proceedings of the European conference on computer vision , 2022.
- [46] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, L ukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems , 30, 2017.
- [47] Chaodong Xiao, Minghan Li, Zhengqiang Zhang, Deyu Meng, and Lei Zhang. Spatial-mamba: Effective visual state space models via structure-aware state fusion. In The Fourteenth International Conference on Learning Representations , 2025.
- [48] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In Proceedings of the European conference on computer vision , 2018.
- [49] Yicheng Xiao, Lin Song, Shaoli Huang, Jiangshan Wang, Siyu Song, Yixiao Ge, Xiu Li, and Ying Shan. Grootvl: Tree topology is all you need in state space model. arXiv preprint arXiv:2406.02395 , 2024.
- [50] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, et al. Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE international conference on computer vision , 2019.
- [51] Guowen Zhang, Lue Fan, Chenhang He, Zhen Lei, ZHAO-XIANG ZHANG, and Lei Zhang. Voxel mamba: Group-free state space models for point cloud based 3d object detection. Advances in Neural Information Processing Systems , 37:81489-81509, 2024.
- [52] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, et al. mixup: Beyond empirical risk minimization. In The Seventh International Conference on Learning Representations , 2018.
- [53] Jingwei Zhang, Anh Tien Nguyen, Xi Han, Vincent Quoc-Huy Trinh, Hong Qin, Dimitris Samaras, and Mahdi S Hosseini. 2dmamba: Efficient state space model for image representation with applications on giga-pixel whole slide image classification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2025.
- [54] Sijie Zhao, Hao Chen, Xueliang Zhang, Pengfeng Xiao, Lei Bai, and Wanli Ouyang. Rs-mamba for large remote sensing image dense prediction. IEEE Transactions on Geoscience and Remote Sensing , 2024.
- [55] Zhun Zhong, Liang Zheng, Guoliang Kang, et al. Random erasing data augmentation. In AAAI , 2020.
- [56] Bolei Zhou, Hang Zhao, Xavier Puig, et al. Scene parsing through ade20k dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2017.
- [57] Lei Zhu, Xinjiang Wang, Zhanghan Ke, Wayne Zhang, and Rynson Lau. Biformer: Vision transformer with bi-level routing attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2023.

- [58] Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, and Xinggang Wang. Vision mamba: Efficient visual representation learning with bidirectional state space model. arXiv preprint arXiv:2401.09417 , 2024.
- [59] Zhen Zou, Hu Yu, Jie Huang, and Feng Zhao. Freqmamba: Viewing mamba from a frequency perspective for image deraining. In Proceedings of the 32nd ACM International Conference on Multimedia , pages 1905-1914, 2024.

## Supplementary Material

Appendix A Efficient Computation Theory for Polyline Path Mask Applications;

Appendix A.1 Notations; Appendix A.2 Definition of Polyline Path Mask; Appendix A.3 Theorems and Proofs; Appendix A.4 Complexity Analysis of Mamba2 Attention Form; Appendix A.5 Complexity Analysis of Polyline Path Mask; Appendix A.6 Complexity Analysis of Polyline Path Mask Multiplication; Appendix A.7 Applications of Polyline Path Masked Attention; Appendix B Experimental Details; Appendix B.1 Implementation Details; Appendix B.2 Training Settings for ImageNet-1K; Appendix B.3 Training Settings for Downstream Tasks; Appendix B.4 Throughput Comparison; Appendix B.5 Visualization; Appendix C Discussion; Appendix C.1 Selectivity of Polyline Path Mask; Appendix C.2 3D Extension of Polyline Path Mask;

Appendix C.3 Limitations;

## A Efficient Computation Theory for Polyline Path Mask Applications

## A.1 Notations

Following Mamba2 [6], we employ a large number of notations both for clarity and as a central tool in stating and proving our theorems, including:

- Dimensions. We generally use N , H , W , C , D as the superscript letters of R to denote the sequence length, the height of the feature map, the width of the feature map, channel number, and hidden state dimension, respectively. The sequence length of the 2D feature map (i.e., the number of tokens) is N = H × W .
- Matrices and Tensors. Following convention, we use non-bolded lowercase letters, bolded lowercase letters, bolded uppercase letters, and bolded calligraphy letters to denote scalars, vectors, matrices, and 3D or higher dimensional tensors, respectively.
- Indexing. We use indexing i : j to refer to the range i + 1 , i + 2 , . . . , j when i &lt; j and i, i -1 , . . . , j + 1 when i &gt; j . For example, for any scalar a , we let a i : j for i &lt; j denote the sequence ( a i +1 , a i +2 , . . . , a j ) . For shorthand, we let a × i : j for i &lt; j denote the product a i +1 × a i +2 ×··· × a j 4 . We let a × j : i = a × i : j for j &gt; i .
- Tensor Unfolding. We use the operator vec ( · ) to vectorize a matrix by stacking its columns and the operator unvec ( · ) as its inverse operation. We use the operator unfold ( · ) to unfolds a 4D tensor L ∈ R H × W × H × W to a 2D matrix L ∈ R HW × HW , where [ L ] ( i -1) × W + j, ( k -1) × W + l = L i,j,k,l , and the operator fold ( · ) as its inverse operation.
- Tensor Multiplication. For 2D matrices, we use the symbol × to denote the matrix multiplication and the symbol ⊙ to denote the Hadamard (element-wise) multiplication. For multiplication operations involving 3D or higher dimensional tensors, we use the Einstein summation notation einsum ( · ) to denote the tensor multiplication on the given dimension, which is commonly used in modern tensor libraries such as PyTorch. For example, einsum ( ′ nc , mc → nm ′ , Q , K ) denotes the matrix multiplication Q × K ⊤ .

Figure 8: An illustration of the V2H polyline path scanning on a 3 × 3 grid (with a total of 9 tokens). There are 81 scanning paths. Each scanning path (red polyline) corresponds to a decay weight in the polyline path mask L .

<!-- image -->

## A.2 Definition of Polyline Path Mask

For each token pair ( x i,j , x k,l ) in the 2D grid, the decay weight of the vertical-then-horizontal (V2H) polyline path from x i,j to x k,l is defined as L i,j,k,l , which is the product of all decay factors along that path, i.e.,

<!-- formula-not-decoded -->

where α i,j : l and β i : k,l are horizontal and vertical decay factors bounded in the range [0 , 1] . For convenience, we unfold the 4D tensor L ∈ R H × W × H × W into a 2D matrix as the polyline path mask L ∈ R HW × HW , i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

An intuitive example illustrating the polyline path scanning on a 3 × 3 grid is presented in Fig. 8. For the 9 tokens in the 2D grid, there are 81 V2H scanning paths connecting them. The V2H scanning path between each token pair is marked by the red polyline, which corresponds to a decay weight in

4 In some contexts, it is always clear that the notation a i : j means a × i : j , and the superscript is omitted.

Figure 9: An overall illustration of the efficient computation theory and corresponding applications.

<!-- image -->

the polyline path mask L . The V2H polyline path mask L ∈ R 9 × 9 , constructed based on the scanning paths in Fig. 8, is defined as:

```
L =           α 1 , 1:1 β 1:1 , 1 α 1 , 1:2 β 1:1 , 2 α 1 , 1:3 β 1:1 , 3 α 1 , 2:1 β 1:1 , 1 α 1 , 2:2 β 1:1 , 2 α 1 , 2:3 β 1:1 , 3 α 1 , 3:1 β 1:1 , 1 α 1 , 3:2 β 1:1 , 2 α 1 , 3:3 β 1:1 , 3 α 1 , 1:1 β 1:2 , 1 α 1 , 1:2 β 1:2 , 2 α 1 , 1:3 β 1:2 , 3 α 1 , 2:1 β 1:2 , 1 α 1 , 2:2 β 1:2 , 2 α 1 , 2:3 β 1:2 , 3 α 1 , 3:1 β 1:2 , 1 α 1 , 3:2 β 1:2 , 2 α 1 , 3:3 β 1:2 , 3 α 1 , 1:1 β 1:3 , 1 α 1 , 1:2 β 1:3 , 2 α 1 , 1:3 β 1:3 , 3 α 1 , 2:1 β 1:3 , 1 α 1 , 2:2 β 1:3 , 2 α 1 , 2:3 β 1:3 , 3 α 1 , 3:1 β 1:3 , 1 α 1 , 3:2 β 1:3 , 2 α 1 , 3:3 β 1:3 , 3 α 2 , 1:1 β 2:1 , 1 α 2 , 1:2 β 2:1 , 2 α 2 , 1:3 β 2:1 , 3 α 2 , 2:1 β 2:1 , 1 α 2 , 2:2 β 2:1 , 2 α 2 , 2:3 β 2:1 , 3 α 2 , 3:1 β 2:1 , 1 α 2 , 3:2 β 2:1 , 2 α 2 , 3:3 β 2:1 , 3 α 2 , 1:1 β 2:2 , 1 α 2 , 1:2 β 2:2 , 2 α 2 , 1:3 β 2:2 , 3 α 2 , 2:1 β 2:2 , 1 α 2 , 2:2 β 2:2 , 2 α 2 , 2:3 β 2:2 , 3 α 2 , 3:1 β 2:2 , 1 α 2 , 3:2 β 2:2 , 2 α 2 , 3:3 β 2:2 , 3 α 2 , 1:1 β 2:3 , 1 α 2 , 1:2 β 2:3 , 2 α 2 , 1:3 β 2:3 , 3 α 2 , 2:1 β 2:3 , 1 α 2 , 2:2 β 2:3 , 2 α 2 , 2:3 β 2:3 , 3 α 2 , 3:1 β 2:3 , 1 α 2 , 3:2 β 2:3 , 2 α 2 , 3:3 β 2:3 , 3 α 3 , 1:1 β 3:1 , 1 α 3 , 1:2 β 3:1 , 2 α 3 , 1:3 β 3:1 , 3 α 3 , 2:1 β 3:1 , 1 α 3 , 2:2 β 3:1 , 2 α 3 , 2:3 β 3:1 , 3 α 3 , 3:1 β 3:1 , 1 α 3 , 3:2 β 3:1 , 2 α 3 , 3:3 β 3:1 , 3 α 3 , 1:1 β 3:2 , 1 α 3 , 1:2 β 3:2 , 2 α 3 , 1:3 β 3:2 , 3 α 3 , 2:1 β 3:2 , 1 α 3 , 2:2 β 3:2 , 2 α 3 , 2:3 β 3:2 , 3 α 3 , 3:1 β 3:2 , 1 α 3 , 3:2 β 3:2 , 2 α 3 , 3:3 β 3:2 , 3 α 3 , 1:1 β 3:3 , 1 α 3 , 1:2 β 3:3 , 2 α 3 , 1:3 β 3:3 , 3 α 3 , 2:1 β 3:3 , 1 α 3 , 2:2 β 3:3 , 2 α 3 , 2:3 β 3:3 , 3 α 3 , 3:1 β 3:3 , 1 α 3 , 3:2 β 3:3 , 2 α 3 , 3:3 β 3:3 , 3           (16)
```

## A.3 Theorems and Proofs

In this section, we present three theorems and their proofs, which will be used to optimize the computational complexity in the following applications A.7. Specifically, we present a decomposition theorem 1 for matrices structured as the polyline path mask L . Based on Theorem 1, we present an efficient matrix multiplication theorem 2 for performing multiplication on the polyline path mask L . Then, we present an equivalent computation theorem 3 for the masked linear attention. An overview illustration is provided in Fig. 9, which summarizes the theorems and their corresponding applications in polyline path masked attention.

Note that the polyline path mask L defined in Eq. (15) is a matrix with special structures. Here, we present a decomposition theorem for matrices structured as L .

Theorem 1 (Matrix Decomposition) . For any matrix M ∈ R HW × HW and M = fold ( M ) , if for ∀ i, j, k, l , ∃ A i ∈ R W × W and B l ∈ R H × H , s.t., M i,j,k,l = [ A i ] j,l × [ B l ] i,k , then M can be decomposed as:

<!-- formula-not-decoded -->

where M A , M B , ˆ M A , ˆ M B ∈ R HW × HW , which satisfy

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

Proof. Let us first prove M A × M B = M in Eq. (17). For clarity, let i = ⌊ u / W ⌋ + 1 , j = u mod W , k = ⌊ v / W ⌋ +1 , l = v mod W , m = ⌊ w / W ⌋ +1 , n = w mod W . For u = 1 : HW and v = 1 : HW , we have

̸

<!-- formula-not-decoded -->

According to Eq. (19), we have

<!-- formula-not-decoded -->

Thus, Theorem 1 is proven.

For matrices of the form given in Eq. (18), when performing multiplication operations, we have: Theorem 2 (Efficient Matrix Multiplication) . For matrices M A , M B defined in Eq. (18) , ∀ x ∈ R HW , the following equation holds:

<!-- formula-not-decoded -->

Proof. The left part of Eq. (22) can be calculated by z = M B × x and y = M A × z . For clarity, let i = ⌊ u / W ⌋ +1 , j = u mod W , k = ⌊ v / W ⌋ +1 , l = v mod W . For u = 1 : HW , we have

̸

<!-- formula-not-decoded -->

The final results in Eq. (23) is equivalent to Z : ,j = B j × X : ,j . Then, for u = 1 : HW , we have

̸

<!-- formula-not-decoded -->

The final results in Eq. (24) is equivalent to Y i, : = A i × Z i, : . Thus, Theorem 2 is proven.

̸

<!-- formula-not-decoded -->

Proof. 1) the left part of the Eq. (25) can be rewritten as:

<!-- formula-not-decoded -->

2) the right part of the Eq. (25) can be rewritten as:

<!-- formula-not-decoded -->

Thus, Theorem 3 is proven. Here, the computational complexity of the left part of Eq. (25) is O ( N 2 ) . The computational complexity of the first and third lines in right part of Eq. (25) is O ( N ) . And the computational complexity of the second line in right part of Eq. (25) is O ( N 2 ) . Thus, if we can reduce the complexity of computing L KV from O ( N 2 ) to O ( N ) , then the complexity of computing Y = (( QK ⊤ ) ⊙ L ) V can be reduced to O ( N ) .

## A.4 Preliminaries: Complexity Analysis of Mamba2 Attention Form

In this section, we present the efficient algorithm proposed in Mamba2 [6] for its attention form, achieving a computational complexity of O ( N ) .

Mamba2's Attention Form. Mamba2's attention form (i.e., structured masked attention) given by the SSD framework [6] is formulated as:

<!-- formula-not-decoded -->

where X , Y ∈ R N × C are the input and output sequences, respectively, B , C ∈ R N × D are inputdependent parameters learned by multilayer perceptron (MLP) layers. The 1D structured mask L 1 D ∈ R N × N is a 1-semiseparable matrix [6], and the scalar a i serves as a decay factor bounded in the range [0 , 1] . In Mamba2, parameters C and B in Eq. (28) correspond to the query Q and key K in ViTs, respectively. Therefore, Eq. (28) reveals that the selective state transition function in Mamba2 is equivalent to the Hadamard product of a linear attention map CB ⊤ and a 1D structured mask L 1 D .

Naive Computation. As defined in Eq. (28), the straightforward computation of structured masked attention has a complexity of O ( N 2 ) . In contrast, the complexity of linear attention Y = ( CB ⊤ ) X = C ( B ⊤ X ) can be reduced from O ( N 2 ) to O ( N ) by the associative property of matrix multiplication. However, this approach is not directly applicable to Eq. (28) because of the introduction of the Hadamard product.

Lemma 1. Let L ∈ R N × N be a 1-semiseparable matrix and X ∈ R N × C be a matrix, the complexity of computing Y = LX can be reduced from O ( N 2 ) to O ( N ) by using the chunkwise algorithm in Mamba2 [6].

Efficient Computation. Based on Theorem 3, Eq. (28) can be computed as follows:

<!-- formula-not-decoded -->

Note that L 1 D is a 1-semiseparable matrix, and the second line of Eq. (29) can be reformulated as a matrix multiplication. Therefore, by applying Lemma 1, the complexity of computing L BX can be reduced from O ( N 2 ) to O ( N ) . Moreover, the complexity of computing L 1 D is O ( N ) . Consequently, the overall computational complexity of Eq. (28) is O ( N ) .

## A.5 Complexity Analysis of Polyline Path Mask

In this section, we analyze the complexity of computing the polyline path mask L , as stated in Corollary 1, with detailed explanation.

Corollary 1 (Mask Complexity) . The complexity of directly computing polyline path mask L via Eq. (16) and (14) is O ( N 5 2 ) , which can be reduced to O ( N 2 ) by applying Theorem 1, where N = H × W .

Naive Computation. According to the definition in Eq. (15), the polyline path mask L ∈ R HW × HW is large in size, and each element requires numerous multiplications, resulting in high computational cost. The most straightforward way to compute L is to calculate each element L u,v individually. Hence, the total complexity of computing matrix L is N 2 times the complexity of computing each element L u,v . As defined in Eq. (16) and Eq. (14), L u,v = L i,j,k,l = α i,j : l β i : k,l , where i = ⌊ u / W ⌋ +1 , j = u mod W and k = ⌊ v / W ⌋ +1 , l = v mod W . There are | l -j | and | k -i | multiplication operations in α i,j : l and β i : k,l , respectively. Here, i, k range from 1 to H , and j, l range from 1 to W . Therefore, the complexity of L u,v is O ( N 1 2 ) . Consequently, the overall complex of directly computing L is O ( N 5 2 ) .

Efficient Computation. The polyline path mask L satisfies the conditions in Theorem 1 with [ A i ] j,l = α k,j : l and [ B l ] i,k = β i : k,j . Thus, based on Theorem 1, the polyline path mask L can be decomposed as:

<!-- formula-not-decoded -->

For example, as illustrated in Fig. 10 (a), the polyline path mask L in Eq. (16) can be decomposed as:

<!-- formula-not-decoded -->

<!-- image -->

Figure 10: (a) Illustration of the decomposition of the polyline path mask L . (b) Illustration of the multiplication between the polyline path mask L and vector x . (Algorithm 2) .

<!-- image -->

Feature map resolution (H×W)

Figure 11: The comparison of the relative time consuming and memory usage between the naive computation and efficient computation (Algorithm 2) of Lx .

Note that there are H × W 2 non-zero elements in L H and each non-zero element α i,j : l requires O ( N 1 2 ) multiplication operations. Thus, the complexity of computing L H and L V is O ( N 2 ) . Similarly, the complexity of computing ˆ L H and ˆ L V is also O ( N 2 ) . Thus, the complexity of computing L can be reduced to O ( N 2 ) by Eq. (30).

## A.6 Complexity Analysis of Polyline Path Mask Multiplication

In this section, we analyze the complexity of computing the matrix multiplication between the polyline path mask L and the vector x , as stated in Corollary 2 and Algorithm 2, with detailed explanation. Fig. 11 presents the comparison of speed and memory usage between the naive computation and efficient computation of Lx . Compared to the naive computation approach, Algorithm 2 achieves substantial speed-up and significantly reduced GPU memory consumption, especially when the shape of L and x is large.

Corollary 2 (Masked Attention Complexity) . The computational complexity of the matrix multiplication between polyline path mask and vector x , i.e., y = Lx , can be reduced from O ( N 2 ) to O ( N 3 2 ) by Algorithm 2, and further reduced to O ( N ) by applying the chunkwise algorithm of Mamba2 [6] to steps 3 and 5 in Algorithm 2.

Naive Computation. Typically, the polyline path mask L ∈ R N × N is a rank-N matrix. Thus, the most direct approach to compute Lx requires a computational complexity of O ( N 2 ) .

Efficient Computation. As mentioned above, the polyline path mask L can be decomposed as L H × L V , where L H and L V satisfy the definition in Eq. (18) with [ A i ] j,l = α i,j : l and [ B l ] i,k = β i : k,l . Thus, based on Theorem 2, we can design Algorithm 2 for computing the matrix multiplication between polyline path mask L and the vector x . As shown in Algorithm 2, computing B l × X : ,l has a complexity of O ( H 2 ) . Thus, the complexity of computing Z (i.e., step 3 in Algorithm 2) is O ( H 2 W ) . Similarly, the complexity of computing Y (i.e., step 5 in Algorithm 2) is O ( HW 2 ) . Thus, the computational complexity of Algorithm 2 is O ( N 3 2 ) , where N = H × W .

As illustrated in Fig. 10 (b), the matrices A i and B l are symmetric matrices, and their lower triangular parts are both 1-semiseparable matrices as defined in Mamba2 [6]. Therefore, by applying Lemma 1 the complexity of computing B l × X : ,l and can be reduced from O ( H 2 ) to O ( H ) , and the complexity of

Algorithm 2: Efficient Masked Attention Computation.

Input: decay factors α, β of the polyline path mask L , vector x ∈ R HW ;

- 1: Compute X =unvec( x ) ∈ R H × W ;
- 2: Compute B l ∈ R H × H , where for l =1: W, [ B l ] i,k = β i : k,l ;
- 3: Compute Z ∈ R H × W , where Z : ,l = B l × X : ,l ;
- 4: Compute A i ∈ R W × W , where for i =1: H, [ A i ] j,l = α i,j : l ;

5: Compute

Y

∈

R

H

×

Output: y =vec( Y ) ;

computing A i × Z i, : can be reduced from O ( W 2 ) to O ( W ) . Consequently, the overall complexity of computing Lx is O ( N ) .

## A.7 Applications of Polyline Path Masked Attention

The proposed polyline path mask can be seamlessly integrated into various attention variants in a plug-and-play manner. As illustrated in Fig. 9, theorems and algorithm given in Sec. A.5 and Sec. A.6 ensure that integrating the polyline path mask does not substantially increase the computational complexity of the original attention mechanism. In this section, we introduce several Polyline Path Masked Attention (PPMA), including Polyline Path Masked Vanilla Attention (PPMVA), Polyline Path Masked Linear Attention (PPMLA), Polyline Path Masked Criss-Cross Attention (PPMCCA), and Polyline Path Masked Decomposed Attention (PPMDA).

Basic Paradigm. The basic Polyline Path Masked Attention (PPMA) is implemented by performing a Hadamard multiplication with the attention map. Specifically, given query Q , key K , and value V ∈ R HW × C , PPMA is formulated as:

<!-- formula-not-decoded -->

1) Polyline Path Masked Vanilla Attention. According to Eq. (32), the polyline path masked vanilla attention is formulated as:

<!-- formula-not-decoded -->

Based on Corollary 1, the computation of L 2 D has a complexity of O ( N 2 ) . Thus, Eq. (33) maintains the complexity of O ( N 2 ) .

2) Polyline Path Masked Linear Attention. Similar to Mamba2's attention form (Eq. (28)), the polyline path masked linear attention is formulated as:

<!-- formula-not-decoded -->

W

, where

Y

i,

:

=

A

i

×

Z

i,

:

;

Based on Theorem 3, we can compute ( ( QK ⊤ ) ⊙ L ) V as follows:

<!-- formula-not-decoded -->

Eq. (35) shows that the computational complexity of Eq. (34) depends on computing L KV . Based on Corollary 2 and Algorithm 2, the computational complexity of the second line in Eq. (35) can be reduced from O ( N 2 ) to O ( N ) . Thus, the computational complexity of Eq. (34) maintains O ( N ) .

3) Polyline Path Masked Criss-Cross Attention. The original criss-cross attention [22] employs sparse attention over tokens located in the same row or column, achieving a computational complexity of O ( N 3 2 ) . In this work, following RMT [10], we decompose criss-cross attention into vertical attention over each column followed by horizontal attention over each row. The polyline path masked criss-cross attention is formulated as:

<!-- formula-not-decoded -->

where horizontal and vertical attention maps S H , S V ∈ R HW × HW satisfy the form in Eq. (18) with A i =softmax( Q i, : , : K ⊤ i, : , : ) and B l =softmax( Q : ,l, : K ⊤ : ,l, : ) , and Q , K ∈ R H × W × C are tensor forms of Q , K , respectively [22]. Based on Theorem 1, we can reformulate the left part of Eq. (36) as:

<!-- formula-not-decoded -->

Note that matrices ˆ S H ⊙ ˆ L H and ˆ S V ⊙ ˆ L V also satisfy the form (i.e. M A and M B ) in Eq. (18). Thus, the computational complexity of Eq. (37) can be reduced to O ( N 3 2 ) by Algorithm 2. Similar conclusions can also be derived for the right part of Eq. (36). Thus, the overall computational complexity of Eq.(36) maintains O ( N 3 2 ) .

4) Polyline Path Masked Decomposed Attention. For general decomposable attention which can be decomposed as S = S 1 × S 2 , where S 1 ∈ R N × D and S 2 ∈ R D × N , the polyline path masked decomposed attention is formulated as:

<!-- formula-not-decoded -->

According to Theorem 3, we can compute (( S 1 × S 2 ) ⊙ L ) V as follows:

<!-- formula-not-decoded -->

Based on Corollary 2 and Algorithm 2, the computational complexity of Eq. (39) can be reduced from O ( N 2 ) to O ( ND ) . Thus, the computational complexity of Eq. (38) is O ( ND ) .

## B Experimental Details

## B.1 Architecture Details

As illustrated in Fig. 5, our backbone adopts the same four-stage hierarchical architecture as RMT [10], where the first three stages employ Polyline Path Masked Criss-Cross Attention and the final stage

employs the Polyline Path Masked Vanilla Attention. Moreover, we develop our model in three scales: tiny (PPMA-T), small (PPMA-S), and base (PPMA-B).

The detailed configurations of PPMA variants are provided in Tab. 5. Following RMT [10], the stem layer consists of five 3 × 3 convolution layers followed by GELU and batch normalization to embed the input image into 56 × 56 tokens. The downsampling layer consists of 3 × 3 convolution layers with stride 2 to reduce the feature map's resolution. Moreover, we follow RMT [10] and incorporate RoPE [38], CPE [3], and LCE [57] into the PPMA blocks. All other configurations also follow RMT [10]. Code is available at ❤/a116/a116♣/a115✿✴✴❣✐/a116❤✉❜✳❝♦♠✴③❤♦♥❣❝❤❡♥③❤❛♦✴/a80/a80▼❆ .

## B.2 Training Settings for ImageNet-1K

To ensure reproducibility and consistency with prior work, we follow the training strategy of RMT [10] and DeiT [44]. Specifically, we employ various data augmentation techniques, including RandAugment [5], Mixup [52] (prob=0.8), CutMix [50] (prob=1.0), Random Erasing [55] (prob=0.25). For model optimization, we use the AdamW optimizer with a cosine decay learning rate scheduler and train our model 300 epochs from scratch. The initial learning rate, weight decay, and batch size are set to 0.001, 0.05, and 1024, respectively. The drop path rates for PPMA-T, PPMA-S, and PPMA-B are set to 0.1, 0.15, and 0.4, respectively. We also adopt training techniques from RMT [10], including Label Smoothing (0.1) [41] and Exponential Moving Average (EMA) [33].

## B.3 Training Settings for Downstream Tasks

For experiments on the ADE20K [56] and MSCOCO2017 [28] datasets, we follow the training settings of TransNeXT [35], and utilize the MMDetection [2] and MMSegmentation [4] libraries for training. Specifically, in the MMDetection [2] library, we adopt Mask R-CNN [18] as the basic framework and use the AdamW optimizer with an initial learning rate of 0 . 0001 and a weight decay of 0.01. The model is trained for 12 epochs with a batch size of 16 using the standard 1 × schedule. In the MMSegmentation [4] library, we adopt UPerNet [48] as the basic framework and use the AdamW optimizer with the initial learning rate of 6 × 10 -5 and the weight decay of 0.01. All models are trained for 160K iterations with a batch size of 16 on the ADE20K dataset. The input size of images is set to 512 × 512 .

## B.4 Throughput Comparison

To evaluate the inference speed of our model, we measure the throughput of PPMA-T/S/B on an A800 GPUwith a batch size of 64 and the image resolution of 224 × 224 . As shown in Table 6, the inference throughput of PPMA-T/S/B decrease by 37%/30%/21% compared to RMT-T/S/B, respectively. This is mainly caused by the additional GPU kernel launches and memory transactions required to compute the polyline path mask. As shown in Table 6, the CUDA implementation of TransNeXt achieves a significant speedup over the PyTorch implementation. In our implementation, the polyline path mask is currently computed using PyTorch. Similar to TransNeXt, our implementation can also be optimized through engineering efforts, such as using CUDA or Triton-based implementations, to accelerate inference speed.

## B.5 Visualization

The visualizations of the polyline path masked attention map are shown in Fig. 12. Input images are taken from the ImageNet-1K validation set, and the query token is marked by a red box on each input

Table 5: Detailed Architectures of the Polyline Path Masked Attention based Vision Transformer.

| Model                | Blocks                                   | Channels                                                    | Heads                                      | Ratios                                 | #Param. (M)   | FLOPs (G)    |
|----------------------|------------------------------------------|-------------------------------------------------------------|--------------------------------------------|----------------------------------------|---------------|--------------|
| PPMA-T PPMA-S PPMA-B | [2, 2, 8, 2] [3, 4, 18, 4] [4, 8, 25, 8] | [64, 128, 256, 512] [64, 128, 256, 512] [80, 160, 320, 512] | [4, 4, 8, 16] [4, 4, 8, 16] [5, 5, 10, 16] | [3, 3, 3, 3] [4, 4, 3, 3] [4, 4, 3, 3] | 14 27 54      | 2.7 4.9 10.6 |

Table 6: Comparison of inference speed across different models on ImageNet-1K. Throughput is measured on an A800 GPU with a batch size of 64.

| Model                      | #Param. (M)   | FLOPs (G)           | Throughput (imgs/s)         | Top-1 (%)                     |
|----------------------------|---------------|---------------------|-----------------------------|-------------------------------|
| BiFormer-T [57] SMT-T [29] | 13 12 14      | 2.2 2.4 2.5 2.7 2.7 | 1602 636 1650 742 1299 1034 | 81.4 82.2 82.4 82.5 82.5 82.6 |
| RMT-T [10]                 |               |                     |                             |                               |
| TransNeXt-M (PyTorch) [35] | 13            |                     |                             |                               |
| TransNeXt-M (CUDA) [35]    | 13            |                     |                             |                               |
| PPMA-T                     | 14            | 2.7                 |                             |                               |
| CMT-S [14]                 | 25            | 4.0                 | 848                         | 83.5                          |
| MaxViT-T [45]              | 31            | 5.6                 | 826                         | 83.6                          |
| SMT-S [29]                 | 20            | 4.8                 | 356                         | 83.7                          |
| BiFormer-S [57]            | 26            | 4.5                 | 766                         | 83.8                          |
| RMT-S [10]                 | 27            | 4.5                 | 876                         | 84.0                          |
| TransNeXt-T (PyTorch) [35] | 28            | 5.7                 | 508                         | 84.0                          |
| TransNeXt-T (CUDA) [35]    | 28            | 5.7                 | 947                         | 84.0                          |
| PPMA-S                     | 27            | 4.9                 | 612                         | 84.2                          |
| SMT-B [29]                 | 32            | 7.7                 | 237                         | 84.3                          |
| BiFormer-B [57]            | 57            | 9.8                 | 498                         | 84.3                          |
| CMT-B [14]                 | 46            | 9.3                 | 447                         | 84.5                          |
| TransNeXt-T (PyTorch) [35] | 50            | 10.3                | 266                         | 84.7                          |
| TransNeXt-T (CUDA) [35]    | 50            | 10.3                | 436                         | 84.7                          |
| RMT-B [10]                 | 54            | 9.7                 | 457                         | 84.9                          |
| PPMA-B                     | 54            | 10.6                | 362                         | 85.0                          |

image. The decay factors and attention maps are generated by the second block of the first stage in the PPMA-T model trained on the ImageNet-1K training set.

Input-dependent Decay Factor. As shown in Fig. 12 (b) and (c), the decay factors α and β learned by the network can roughly capture the edge information of objects in the feature map: decay factors at edges tend to be smaller (approaching zero), whereas those in homogeneous regions tend to be larger (approaching one). Moreover, the supervised training encourages the decay factors α and β to focus on horizontal and vertical edge information, respectively.

Polyline Path Mask. As shown in Fig. 12 (e), the polyline path mask, generated by the cumulative multiplication of decay factors, effectively captures the semantic continuity in the feature space. It maintains continuity in homogeneous regions sharing the same semantics and shows discontinuity at the edges between regions of different semantics.

Polyline Path Masked Attention Map. Fig. 12 (d) shows that attention maps from shallow layers in typical ViT models often struggle to focus on tokens relevant to the query token. In contrast, Fig. 12 (f) demonstrates that integrating the polyline path mask L 2 D successfully suppresses interference from distant and irrelevant tokens, resulting in more semantically accurate masked attention maps.

## C Discussion

## C.1 Selectivity of Polyline Path Mask

Compared to previous state-space models (SSMs) such as RetNet [40], the primary contribution of Mamba [11] and Mamba2 [6] is the introduction of a selective mechanism into the structured mask, which leads to significant performance improvements. However, current studies still lack a deep understanding of this crucial selectivity mechanism.

In this work, we argue that the selective mechanism in Mamba explicitly models the semantic continuity in sequences, which corresponds to the local smoothness prior in images. Building on this insight, we adopt the selective structured mask of Mamba2 and naturally generalize it into a 2D polyline path mask for Vision Transformers (ViTs).

Semantic Continuity in Sequence. Similar to self-attention maps in ViTs, the structured mask L ∈ R N × N can also be viewed as a weighting matrix that maps input tokens X ∈ R N × C to output

Figure 12: Visualizations of the decay factors and the polyline path masked attention maps of the well-trained PPMA model. In each input image, the query token is marked by a red box.

<!-- image -->

tokens Y ∈ R N × C along the sequence length dimension. In this weighting matrix L , a larger decay weight L i,j indicates a greater influence of the input token X i on the output token Y j , and vice versa. In Mamba2, L i,j is computed as the cumulative multiplication of decay factors a × i : j to achieve linear complexity. As a result, if any factor is close to zero, L i,j approaches zero; conversely, L i,j approaches one only when all decay factors are close to one.

For most semantic-related tasks, an ideal structured mask should model semantic continuity in the sequence: it should maintain continuous between connected tokens with the same semantic, while breaking between tokens with different semantics. This enables the aggregation and separation of tokens according to their semantics. Accordingly, decay factors should ideally be larger in homogeneous regions and smaller at heterogeneous regions. As illustrated in Fig. 12 (b) and (c), the decay factors learned through supervised training align well with this assumption. ×2

Local Smoothness Prior in Images. In natural images, spatially adjacent patches are more likely to belong to the same object and share similar semantics. This local smoothness prior plays a crucial role in natural image processing tasks, especially those requiring fine-grained feature extraction. The selectivity of polyline path mask aligns naturally with this prior by modeling semantic continuity within homogeneous regions and allowing discontinuities at object edges. Experimental results also show that integrating the polyline path mask yields significant performance improvements on the ADE20K semantic segmentation task. 𝓧

## C.2 3D Extension of Polyline Path Mask

Based on the decomposability, we naturally extend the 2D polyline path mask to 3D applications. As illustrated in Fig. 13, the 3D polyline path mask L 3 D can be decomposed as the multiplication of three 1D structured masks, L H × L V × L D , representing the horizontal, vertical, and depth scanning masks, respectively. Specifically, for each token pair ( x i,j,k , x l,m,n ) in the 3D grid, the 3D polyline path mask is defined as:

<!-- formula-not-decoded -->

where L 3 D is the tensor form of matrix L 3 D , α , β , and γ are the decay factors along the horizontal, vertical, and depth axes, respectively. Compared to the cross-scanning strategy [30], the 3D polyline path scanning strategy better preserves the adjacency relationships of 3D tokens.

## C.3 Limitations

In this work, we introduce a learnable, input-dependent polyline path mask as the explicit positional encoding for ViTs, replacing the fixed decay mask in RMT [10]. Experiments on high-level tasks demonstrate the superiority of our method, particularly in fine-grained segmentation benchmarks, where PPMA-T/S/B outperform RMT-T/S/B by 0.7%/1.3%/0.3% SS mIoU on ADE20K, respectively.

Notably, our carefully designed polyline path mask L 2 D is decomposable as described in Eq. (30), enabling efficient computation via algorithms 2 to optimize the computational complexity. However, despite these optimizations, the large size of the mask L 2 D still inevitably incurs extra GPU memory occupation and slower inference speed compared to RMT [10]. As shown in Table 6, the inference throughput of PPMA-T/S/B decrease by 37%/30%/21% in comparison with RMT-T/S/B, respectively. This limitation can be mitigated through engineering optimizations, such as CUDA or Triton-based implementations, which we plan to investigate in the future work.

Figure 13: Illustration of the 3D extension of polyline path mask.

<!-- image -->

𝑨 𝑻

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction accurately describe the proposed method (Polyline Path Masked Attention), theoretical contributions (Efficient Computation Theory), and experimental results (experiments on image classification, object detection and segmentation tasks), which align with the content presented in the paper (Sections 4 and 5).

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations are discussed in the Appendix, where the authors acknowledge constraints such as the proposed model has lower throughput than some existing models with similar FLOPs.

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

Justification: All theoretical results in the paper are accompanied by a complete set of clearly stated assumptions and formal proofs. While the full detailed proofs are presented in the Appendix for readability, the main paper includes a simple version to aid understanding.

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

Justification: Yes. The paper provides sufficient details to ensure the reproducibility of its main experimental results. The overall architecture is clearly described in Section 4. Furthermore, Section 5 outlines the experimental settings in detail. These descriptions are sufficient for the reproduction to verify the main claims of the paper.

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

Justification: We provide a GitHub repository in the abstract in the anonymised version.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( ❤/a116/a116♣/a115✿✴✴♥✐♣/a115✳❝❝✴ ♣✉❜❧✐❝✴❣✉✐❞❡/a115✴❈♦❞❡❙✉❜♠✐/a115/a115✐♦♥/a80♦❧✐❝② ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( ❤/a116/a116♣/a115✿ ✴✴♥✐♣/a115✳❝❝✴♣✉❜❧✐❝✴❣✉✐❞❡/a115✴❈♦❞❡❙✉❜♠✐/a115/a115✐♦♥/a80♦❧✐❝② ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: All the training and test details are presented in Section 5.1, 5.2 and 5.3.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The experiment for this task is time-consuming. Referring to previous work, there is no such experimental data.

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

Justification: The computational cost have been presented in the Appendix for readability Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics ❤/a116/a116♣/a115✿✴✴♥❡✉/a114✐♣/a115✳❝❝✴♣✉❜❧✐❝✴❊/a116❤✐❝/a115●✉✐❞❡❧✐♥❡/a115 ?

Answer: [Yes]

Justification: We have carefully reviewed the NeurIPS Code of Ethics and confirm that our research fully adheres to its principles.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: This paper presents work whose goal is to advance the field of Deep Learning and Computer Vision. None of the potential societal consequences we feel must be specifically highlighted here.

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

Justification: Our work does not involve pretrained models, generative tools, or scraped datasets that carry a high risk of misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We have properly credited all existing assets used in our work, including publicly available datasets and code repositories.

Guidelines:

- The answer NA means that the paper does not use existing assets.

- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, ♣❛♣❡/a114/a115✇✐/a116❤❝♦❞❡✳❝♦♠✴❞❛/a116❛/a115❡/a116/a115 has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: The primary new asset is the source code for the proposed methods, which is open source. Documentation is assumed to be provided alongside the code in its repository.

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

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( ❤/a116/a116♣/a115✿✴✴♥❡✉/a114✐♣/a115✳❝❝✴❈♦♥❢❡/a114❡♥❝❡/a115✴✷✵✷✺✴▲▲▼ ) for what should or should not be described.