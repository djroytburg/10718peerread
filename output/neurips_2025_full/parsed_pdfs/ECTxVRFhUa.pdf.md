## Tensor Product Attention Is All You Need

Yifan Zhang ∗⋄ 1 , 4 Yifeng Liu ∗ 3 Huizhuo Yuan 3 Zhen Qin Yang Yuan 1 , 2 Quanquan Gu 3 Andrew Chi-Chih Yao 1 , 2 †

1 IIIS, Tsinghua University 2 Shanghai Qi Zhi Institute 3 University of California, Los Angeles 4 Princeton University yifzhang@princeton.edu, liuyifeng@cs.ucla.edu qgu@cs.ucla.edu, andrewcyao@tsinghua.edu.cn

## Abstract

Scaling language models to handle longer input sequences typically necessitates large key-value (KV) caches, resulting in substantial memory overhead during inference. In this paper, we propose T ensor P roduct A ttention (TPA), a novel attention mechanism that uses tensor decompositions to represent queries, keys, and values compactly, substantially shrinking the KV cache size at inference time. By factorizing these representations into contextual low-rank components and seamlessly integrating with RoPE and any possible position encoding mechanisms, TPA achieves improved model quality alongside memory efficiency. Based on TPA, we introduce the T ensor Produc T A TT en T ion T ransformer (T6), a new model architecture for sequence modeling. Through extensive empirical evaluation on language modeling tasks, we demonstrate that T6 surpasses or matches the performance of standard Transformer baselines, including Multi-Head Attention (MHA), Multi-Query Attention (MQA), Grouped-Query Attention (GQA), and Multi-Head Latent Attention (MLA) across various metrics, including perplexity and a range of established evaluation benchmarks. Notably, TPA's memory efficiency and computational efficiency at the decoding stage enable processing longer sequences under fixed resource constraints, addressing a critical scalability challenge in modern language models. Project Page: https://github.com/tensorgi/TPA .

## 1 Introduction

Large language models (LLMs) have revolutionized natural language processing, demonstrating exceptional performance across tasks [5, 12, 58, 6]. As these models evolve, their ability to process longer contexts becomes increasingly important for sophisticated applications such as document analysis, complex reasoning, and code completion. However, managing longer sequences during inference poses significant computational and memory challenges, particularly due to the storage of key-value (KV) caches [70, 34]. Because memory consumption grows linearly with sequence length, the maximum context window is limited by practical hardware constraints.

A variety of solutions have been explored to address this memory bottleneck. Some approaches compress or selectively prune cached states through sparse attention patterns [10] or token eviction strategies [70, 62, 42], though such methods risk discarding tokens that may later prove important. Other work proposes off-chip storage of key-value states [17], at the expense of increased I/O latency. Attention variants like Multi-Query Attention (MQA) [46] and Grouped-Query Attention (GQA) [2] reduce per-token cache requirements by sharing keys and values across heads, but often compromise flexibility or require significant architectural modifications. Meanwhile, low-rank weight factorization methods such as LoRA [20] effectively reduce fine-tuning memory, yet do not address the KV cache overhead that dominates inference at runtime. The recently introduced Multi-Head Latent Attention

∗ Equal contribution; ⋄ Project lead; † Corresponding author.

Figure 1: Tensor Product Attention (TPA) within the T ensor Produc T A TT en T ion T ransformer (T6). In each TPA layer, the input hidden state x t is processed by linear layers to produce latent factor matrices for query (e.g., A Q ( x t ) , B Q ( x t ) ), key (e.g., A K ( x t ) , B K ( x t ) ), and value (e.g., A V ( x t ) , B V ( x t ) ). Rotary Position Embedding (RoPE) is applied to the B Q ( x t ) and B K ( x t ) factors. The query, key, and value tensors for each attention head are then formed by the tensor product of these factor matrices (e.g., Q t = 1 R Q A Q ( x t ) ⊤ B Q ( x t ) ). Finally, the TPA output is computed using scaled dot-product attention, followed by a linear projection of the concatenated results from all heads.

<!-- image -->

(MLA) in Deepseek-V2 [32] caches compressed key-value representations but encounters difficulties with efficient Rotary Position Embedding (RoPE) [52] integration, necessitating additional positionencoded parameters per head.

To overcome the limitations of existing approaches, we introduce Tensor Product Attention (TPA), illustrated in Figure 1. TPA is a novel attention mechanism that employs tensor factorizations for queries (Q), keys (K), and values (V). By dynamically factorizing activations rather than static weights (as in LoRA), TPA constructs low-rank, contextual representations. This approach substantially reduces KV cache memory usage while offering improved representational capacity. In practice, TPA can decrease memory overhead by an order of magnitude compared to standard Multi-Head Attention (MHA), alongside achieving lower pretraining validation loss (perplexity) and better downstream performance. A key advantage of TPA is its native compatibility with rotary positional embeddings (RoPE) [52] and any possible position encodings, enabling a straightforward drop-in replacement for multi-head attention (MHA) layers in modern LLM architectures such as LLaMA [58], Qwen [3], and Gemma [56].

Our main contributions are summarized as follows:

1. We propose Tensor Product Attention (TPA) , a mechanism that factorizes Q , K , and V activations using contextual tensor decompositions. This achieves a substantial reduction in inferencetime KV cache size relative to standard attention mechanisms [60], MHA, MQA, GQA, and MLA, while also improving performance. In addition, we analyze existing attention mechanisms and reveal that MHA, MQA, and GQA can be expressed as non-contextual variants of TPA.
2. We introduce the T ensor Produc T A TT en T ion T ransformer (T6), a new TPA-based model architecture for sequence modeling. In language modeling experiments, T6 consistently improves or matches validation perplexity and downstream evaluation performance, all while maintaining a reduced KV cache size.
3. We demonstrate that TPA integrates seamlessly with RoPE [52] and any possible position encodings as well as output gate and KV shifting, facilitating its easy adoption in popular foundation model architectures like LLaMA, Gemma, and Qwen.

4. We develop FlashTPA Decoding , an efficient autoregressive inference algorithm for TPA. Our empirical results show that FlashTPA Decoding can be faster than optimized MHA, MQA, GQA, and MLA decoding methods, particularly for long sequences.

## 2 Background

In this section, we briefly review Scaled Dot-Product Attention, Multi-Head Attention [60], and introduce key notations. Other attention mechanisms like Multi-Query Attention (MQA) [46], Grouped Query Attention (GQA) [2], Multi-head Latent Attention (MLA) [32, 33], and Rotary Position Embedding (RoPE) [52] are further discussed in the Appendix F.

Notations. We use bold uppercase letters (e.g., X , Q ) for matrices, bold lowercase (e.g., a , b ) for vectors, and italic uppercase (e.g., W Q i ) for learnable parameter matrices. We denote by [ n ] the set { 1 , . . . , n } for some positive integer n . We use ⊤ to denote the transpose of a vector or a matrix. Let d model be the embedding dimension, h the number of attention heads, d h the dimension per head, x t ∈ R d model the input for the t -th token at a given attention layer, X ∈ R T × d model denotes the input embeddings for T tokens, and Q , K , V ∈ R T × h × d h denote the queries, keys, and values of h heads for T tokens. With a little abuse of notation, Q i , K i , V i ∈ R T × d h denote the i -th head of queries, keys, and values, and Q t , K t , V t ∈ R h × d h denote the heads of the query, key, and value for t -th token. Throughout the paper, W Q , W K , W V denote projection matrices for queries, keys, and values, respectively. In multi-head attention, each head is associated with its own set of W Q i , W i K , W V i , and each has dimension W Q i , W i K , W V i ∈ R d model × d h . 5 Similarly, we have an output projection matrix W O ∈ R ( h · d h ) × d model .

We define the tensor product of two vectors as follows: for vectors a ∈ R m , b ∈ R n , the tensor product of a and b is: a ⊗ b = C ∈ R m × n , with C ij = a i b j , where a i is the i -th element of a , b j is the j -th element of b , and C ij is the ( i, j ) -th entry of C . The vectorization of a matrix C ∈ R m × n , denoted vec ( C ) ∈ R mn , stacks the columns of C into a single column vector. For example, if C = [ c 1 , c 2 , . . . , c n ] where c j are columns, then vec ( C ) = [ c ⊤ 1 , c ⊤ 2 , . . . , c ⊤ n ] ⊤ .

## 2.1 Scaled Dot-Product Attention

Scaled dot-product attention [60] determines how to focus on different parts of an input sequence by comparing queries ( Q ) and keys ( K ). It produces a weighted combination of the values ( V ). Formally, the attention output is:

<!-- formula-not-decoded -->

where Q ∈ R n × d h , K ∈ R n × d h , and V ∈ R n × d v for n tokens. The softmax is applied row-wise over the n keys for each query.

## 2.2 Multi-Head Attention (MHA)

Multi-Head Attention (MHA) [60] extends scaled dot-product attention by dividing the model's internal representation into several heads . Each head learns different projections for queries, keys, and values, allowing the model to attend to different types of information from different representational subspaces. For each token embedding x t ∈ R d model , MHAcomputes each head i as follows:

<!-- formula-not-decoded -->

where W Q i , W i K , W V i ∈ R d model × d h are learnable projection matrices for the i -th head, and Q i , K i , V i ∈ R T × d h are the query, key, and value matrices for the i -th head over T tokens. After computing each head's attention output, the results are concatenated and mapped back to the model's original dimension via another learnable linear projection matrix W O ∈ R hd h × d model :

<!-- formula-not-decoded -->

MHA enables the model to capture a rich set of dependencies by allowing each head to focus on different aspects of the input sequence. We also discuss how MHA, MQA, and GQA relate to TPA in the Section 4.

5 Often, h × d h = d model, so each head has query/key/value dimension d h .

## 3 Tensor Product Attention

In this section, we provide a detailed description of our proposed Tensor Product Attention (TPA), which enables contextual low-rank factorization for queries, keys, and values. First, we explain how TPA factorizes these components, specifying tensor shapes. Next, we describe TPA's integration into the multi-head attention framework and its benefits for reducing KV cache memory consumption during inference. Finally, we demonstrate RoPE's seamless integration with TPA, including a pre-rotated variant for efficiency.

## 3.1 Tensor Factorization of Queries, Keys, and Values

̸

Let d attn := hd h denote the total attention projection dimension. Typically one sets d attn = d model, but this is not required: when d attn = d model, the projection matrices W Q , W K , W V map from R d model into R d attn and W O maps R d attn back to R d model . Standard attention projects the entire sequence into three tensors, Q , K , V ∈ R T × h × d h , where Q t , K t , V t ∈ R h × d h denote the slices for the t -th token.

Contextual Factorization. Instead of forming each head's query, key, or value via a single linear map, TPA factorizes each Q t , K t , V t into a sum of (contextual) tensor products whose ranks are R Q , R K , and R V , respectively, and may differ. Specifically, for each token t , with a small abuse of notation, we define:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where a Q r ( x t ) , a K r ( x t ) , a V r ( x t ) ∈ R h , b Q r ( x t ) , b K r ( x t ) , b V r ( x t ) ∈ R d h . Hence, for queries, each tensor product a Q r ( x t ) ⊗ b Q r ( x t ): R h × R d h → R h × d h contributes to the query slice Q t ∈ R h × d h . Analogous definitions apply to the key slice K t and value slice V t .

Latent Factor Maps. Each factor in the tensor product depends on the token's hidden state x t . For example, for queries, we can write:

<!-- formula-not-decoded -->

where W a Q r ∈ R h × d model and W b Q r ∈ R d h × d model are learnable weight matrices. Similar linear maps produce the factors for keys and values.

One often merges the rank index into a single output dimension. For instance, for queries:

<!-- formula-not-decoded -->

which are then reshaped into A Q ( x t ) ∈ R R Q × h and B Q ( x t ) ∈ R R Q × d h (where each row of A Q ( x t ) corresponds to an a Q r ( x t ) ⊤ and each row of B Q ( x t ) to a b Q r ( x t ) ⊤ ). The query tensor for token t can then be expressed as:

<!-- formula-not-decoded -->

This operation is equivalent to Q t = 1 R Q ∑ R Q r =1 a Q r ( x t )( b Q r ( x t )) ⊤ , where a Q r is the r -th column of A Q ( x t ) ⊤ and ( b Q r ) ⊤ is the r -th row of B Q ( x t ) . Repeating for all tokens reconstitutes Q ∈ R T × h × d h . Similar procedures are applied to obtain K and V with ranks R K and R V , respectively.

Scaled Dot-Product Attention. Once Q , K , V are factorized, multi-head attention proceeds as in standard Transformers. For each head i ∈ { 1 , . . . , h } :

<!-- formula-not-decoded -->

where Q i , K i , V i ∈ R T × d h are the slices along the head dimension. Concatenating these h heads along the last dimension yields an R T × ( h · d h ) tensor, which is projected back to R T × d model by an output weight matrix W O ∈ R ( h · d h ) × d model :

<!-- formula-not-decoded -->

Parameter Initialization. We use Xavier initialization [15] for the factor weight matrices; details are in the Appendix G.

## 3.2 RoPE Compatibility and Acceleration

In a typical workflow of adding RoPE to standard multi-head attention, one first computes Q t , K s ∈ R h × d h of the t -th token and s -th token and then applies:

<!-- formula-not-decoded -->

Direct Integration. A useful optimization is to integrate RoPE directly into the TPA factorization. For example, one can pre-rotate the token-dimension factors:

<!-- formula-not-decoded -->

yielding a pre-rotated key representation:

<!-- formula-not-decoded -->

Here, RoPE t is applied to each row of B K ( x t ) (i.e., to each b K r ( x t ) vector). Thus, each cached key factor corresponds to a RoPE-rotated key slice. This removes the need to rotate cached keys at decoding time; the current-step query (which is not cached) can still be rotated on the fly at negligible cost. Depending on hardware and performance requirements, different RoPE integration strategies can be adopted for training and inference.

Theorem 3.1 (RoPE's Compatibility with TPA) . Let Q t be factorized by TPA as

<!-- formula-not-decoded -->

where A Q ( x t ) ∈ R R Q × h and B Q ( x t ) ∈ R R Q × d h . Then we have:

<!-- formula-not-decoded -->

where ˜ B Q ( x t ) := B Q ( x t ) T t = RoPE t ( B Q ( x t ) ) (RoPE applied row-wise to B Q ( x t ) ). Furthermore, let ˜ Q t = RoPE t ( Q t ) = Q t T t and ˜ K s = RoPE s ( K s ) = K s T s be the RoPE-transformed query/key slices. Then RoPE's standard relative-position identity is preserved:

<!-- formula-not-decoded -->

where T t -s := T t T ⊤ s . In particular, for any head i (the i -th row), if q t,i , k s,i ∈ R 1 × d h and ˜ q t,i = q t,i T t , ˜ k s,i = k s,i T s , then ˜ q t,i ˜ k ⊤ s,i = q t,i T t -s k ⊤ s,i .

Theorem 3.1 indicates that TPA does not break RoPE's relative translational property. We prove it in the Appendix D.1.

## 3.3 KV Caching and Memory Reduction

In autoregressive decoding, standard attention caches K t , V t ∈ R h × d h for each past token t . This accumulates to R T × h × d h for keys and R T × h × d h for values, i.e., 2 T h d h total.

TPA Factorized KV Caching. Instead of storing the full K t and V t , TPA stores only their factor components. Specifically, for each past token t , we cache:

<!-- formula-not-decoded -->

where A K ( x t ) ∈ R R K × h , ˜ B K ( x t ) ∈ R R K × d h ( pre-rotated ) , A V ( x t ) ∈ R R V × h , B V ( x t ) ∈ R R V × d h .

Hence, the memory cost per token is R K ( h + d h ) ︸ ︷︷ ︸ for K + R V ( h + d h ) ︸ ︷︷ ︸ for V = ( R K + R V ) ( h + d h ) .

Compared to the standard caching cost of 2 hd h , the ratio is ( R K + R V ) ( h + d h ) 2 hd h . For large h and d h (typically d h = 64 or 128 ), setting R K , R V ≪ h (e.g., rank 1 or 2 ) often yields substantial reduction of KV cache size. Table 1 provides a comparative overview of different attention mechanisms, including TPA and its variants, focusing on KV cache size per token and the number of parameters in an attention layer.

Table 1: Comparison of different attention mechanisms. Here, R Q , R K , and R V denote the ranks for queries, keys, and values in TPA, respectively. Variants of TPA, such as TPA (KVonly), TPA (Noncontextual A), and TPA (Non-contextual B), are detailed in the Appendix G. For MLA, d R h and d h are the dimensions for RoPE and non-RoPE parts; d ′ c and d c are the dimensions of compressed vectors for query and key-value, respectively. The MLA parameter count includes the output projection.

| METHOD                 | KV CACHE                   | # PARAMETERS                                              | # QUERY HEADS   | # KV HEADS   |
|------------------------|----------------------------|-----------------------------------------------------------|-----------------|--------------|
| MHA                    | 2 hd h                     | 4 d model hd h                                            | h               | h            |
| MQA                    | 2 d h                      | 2 d model d h ( h +1)                                     | h               | 1            |
| GQA                    | 2 Gd h                     | 2 d model d h ( h + G ) d ′ c ( d model + hd h + hd R h ) | h               | G            |
| MLA                    | d c + d R h                | + d c ( d model +2 hd h ) + d model ( hd h + d R h )      | h               | h            |
| TPA                    | ( R K + R V )( h + d h ) ) | d model ( R Q + R K + R V )( h + d h )+ d model hd h      | h               | h            |
| TPA (KVonly)           | ( R K + R V )( h + d h     | d model ( R K + R V )( h + d h )+2 d model hd h           | h               | h            |
| TPA (Non-contextual A) | ( R K + R V ) d h          | ( R Q + R K + R V )( d model d h + h )+ d model hd h      | h               | h            |
| TPA (Non-contextual B) | ( R K + R V ) h            | ( R Q + R K + R V )( d model h + d h )+ d model hd h      | h               | h            |

## 4 Expressing MHA, MQA, GQA as Non-contextual TPA

We demonstrate that standard Multi-Head Attention (MHA), Multi-Query Attention (MQA), and Grouped-Query Attention (GQA) can be expressed as special, non-contextual variants of Tensor Product Attention (TPA). This is achieved by imposing specific constraints on the TPA factors, particularly by making the head-dimension factors ( a ) independent of the input token ( x t ).

## 4.1 MHAas Non-contextual TPA

Standard Multi-Head Attention (MHA) can be precisely formulated as a TPA where the rank is equal to the number of heads ( R Q = R K = R V = h ), and the head-dimension factors are fixed, non-contextual basis vectors. To recover MHA, we set the rank R Q = h and define the factors for each head i ∈ [ h ] as follows:

- Contextual token factor : This is the standard linear projection for the i -th head's query:

<!-- formula-not-decoded -->

- Non-contextual head factor : This factor is a scaled standard basis vector, independent of x t :

<!-- formula-not-decoded -->

where e i is the i -th standard basis vector (a vector of zeros with a one at the i -th position).

Substituting these into the TPA equation, the 1 /R Q = 1 /h scaling factor cancels with the scaling of the a Q i factor:

<!-- formula-not-decoded -->

The resulting tensor product, e i ⊗ b Q i ( x t ) , produces an h × d h matrix where only the i -th row is non-zero and contains the vector ( b Q i ( x t )) ⊤ . Summing these matrices for i = 1 , . . . , h assembles the complete query tensor Q t , where the i -th row is precisely the query vector for the i -th head in standard MHA. An analogous construction applies to the key ( K t ) and value ( V t ) tensors.

Thus, MHA is equivalent to a non-contextual TPA where the head-dimension factors are fixed and orthogonal, effectively assigning a dedicated rank component to each attention head.

## 4.2 MQAand GQA as Non-contextual TPA

Similarly, Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) can be seen as non-contextual TPAs where the key and value tensors are formed with a rank lower than the number of heads.

- MQAas Rank-1 TPA (for K and V). In MQA, all h query heads share a single key and value. This corresponds to a TPA with ranks R K = 1 and R V = 1 . The key tensor K t is formed using a

single, non-contextual head-dimension factor a K = 1 h (a vector of all ones) and a single contextual token-dimension factor b K ( x t ) = ( W K ) ⊤ x t :

<!-- formula-not-decoded -->

This creates an h × d h matrix where every row is the same shared key vector ( b K ( x t )) ⊤ . The same logic applies to the value tensor V t . The queries remain full-rank ( R Q = h ) as in MHA.

- GQAas Rank-G TPA (for K and V). GQAis an intermediate approach where h heads are divided into G groups, with heads in the same group sharing a key and value. This is equivalent to a TPA with ranks R K = G and R V = G . The key tensor is formed by summing G components:

<!-- formula-not-decoded -->

Here, b K j ( x t ) is the shared key vector for group j . The non-contextual factor a K j is a scaled mask vector, defined as a K j = G · mask j , where the mask j vector has ones for heads belonging to group j and zeros elsewhere. This scaling cancels the 1 /G pre-factor:

<!-- formula-not-decoded -->

For example, with h = 8 heads and G = 2 groups ( 2 KV heads), the factor for the first group of 4 heads would be a K 1 = 2 · [1 , 1 , 1 , 1 , 0 , 0 , 0 , 0] ⊤ . This construction correctly assembles the final key tensor by broadcasting each group's shared key to its designated heads without any unintended extra scaling.

This perspective highlights that MHA, MQA, and GQA are specific instances of a more general TPA framework, where expressiveness and parameter sharing are controlled by the rank and the nature (contextual vs. non-contextual) of the tensor factors.

## 4.3 Model Architectures

We propose a new architecture called T ensor Produc T A TT en T ion T ransformer (T6), which uses our Tensor Product Attention (TPA) in place of standard MHA (multi-head attention) or GQA (grouped-query attention). Building upon the query, key, and value tensors Q , K , V ∈ R T × h × d h defined in Section 3.1, T6 utilizes the overall architecture of LLaMA [58] while changing the selfattention block to our TPA-based version. The feed-forward network (FFN) adopts a SwiGLU layer, as in [47, 58].

Rotary Positional Embedding (RoPE). As discussed in Section 3.2, RoPE [52] is applied to the Q and K . Within TPA, we pre-rotate the factor b Q t ( x t ) and b K s ( x s ) directly, so that each K s is already rotated prior to caching, see Equation (3.5) and Theorem 3.1.

SwiGLU Feed-Forward Network. Following [47, 58], our T6 uses a SwiGLU-based FeedForward Network (FFN): FFN( x ) = [ σ ( x W 1 ) ⊙ ( x W 2 ) ] W 3 , where σ is the SiLU (a.k.a., swish) nonlinearity, ⊙ is element-wise product, and W 1 , W 2 , W 3 are learnable parameters. Note that other activation functions can also be used.

Overall T6 Block Structure. Putting everything together, one T6 block consists of:

<!-- formula-not-decoded -->

We place norm layers (e.g., RMSNorm) before each sub-layer. Stacking L such blocks yields a T6 model architecture with L layers.

## 5 FlashTPA Decoding Algorithm

For efficient autoregressive inference with Tensor Product Attention (TPA), we introduce FlashTPA Decoding. This algorithm is optimized for generating one token at a time by leveraging the factorized

representation of queries, keys, and values. The core idea, illustrated in Figure 2, is to perform attention computations using a sequence of Einstein summations ('einsum') that operate directly on these factorized components. This avoids materializing the full query, key, and value tensors, which is particularly beneficial as the Key-Value (KV) cache grows with sequence length. The detailed definitions of the input factorized components and the step-by-step pseudo-code for FlashTPA Decoding are provided in Algorithm 2. An optimized Triton kernel implementation is outlined in Algorithm 3 (see Appendix B.1).

Figure 2: Data flow diagram for FlashTPA Decoding. Rectangles represent tensors (blue for inputs, yellow for intermediates, red for final output), circles with ∑ or ⊙ denote Einstein summation contractions or element-wise products respectively, and the green rounded rectangle is the softmax operation. Shapes are shown for a single query ( N = 1 ) interacting with M cached items in the common rank-1 setting R K = R V = 1 . We use a head-first layout ( H,M ) for logits and attention weights; the cached head factors a K cache and a V cache are shown transposed relative to their natural token-major layout for readability. H is the number of heads, R Q is the query rank, and D,E are respective feature dimensions for the B Q / b K cache and b V cache factors. Scaling factors are omitted for visual clarity.

<!-- image -->

This sequence of factorized operations allows FlashTPA Decoding to compute the attention output efficiently. Consequently, TPA is not only memory-efficient due to its smaller KV cache footprint but can also be computationally efficient during inference. The experimental results for FlashTPA decoding time are presented in Section 6.2.

## 6 Experiments

## 6.1 Language Modeling Tasks

All experiments reported in this paper are implemented based on the nanoGPT codebase [24], and we pretrain our models using the FineWeb-Edu 100B dataset [37]. The dataset contains 100 billion tokens for training and 0.1 billion tokens for validation. We compare T6 against the baseline Llama architecture [58] with SwiGLU activation [47] and RoPE embeddings [52], as well as Llama variants that replace Multi-Head Attention (MHA; [60]) with Multi-Query Attention (MQA; [46]), Grouped Query Attention (GQA; [2]), or Multi-head Latent Attention (MLA; [32]). In our experiments, the number of heads h is adjusted for each attention mechanism to ensure that all attention mechanisms have the same number of parameters as the standard Multi-Head Attention (MHA), which has 4 d 2 model parameters per attention layer. We train models at four scales: small (124M parameters), medium (353M), large (773M), and XL (1.5B). We pretrain all models for 50B tokens (roughly half an epoch over FineWeb-Edu-100B). Details on architecture hyperparameters and training hardware are shown in Appendix H.1.

Training &amp; Validation Curves. Figure 4 compares validation loss curves for the medium (353M), large (773M), and XL (1.5B) models on FineWeb-Edu-100B. Training loss curves are provided in Appendix Figure 3. Overall, TPA (red curves) and its simpler variant TPA-KVonly (pink curves) (see Appendix G) converge as fast as or faster than the baselines (MHA, MQA, GQA, MLA) while also achieving visibly lower final validation losses. For instance, in Figure 4(b), TPA and TPA-KVonly remain below the MHA baseline in terms of validation loss at nearly all training stages. Meanwhile, Multi-Head Latent Attention (MLA) [32] (blue curves) generally trains more slowly and yields higher validation losses.

Validation Perplexity. Figure 9 (in the Appendix) shows the validation perplexities of the medium -and large -scale models. Mirroring the loss curves, TPA and TPA-KVonly steadily outperform MHA, MQA, GQA, and MLA over the course of training. By the end of pretraining (around 49 B tokens), TPA-based approaches achieve the lowest perplexities in most configurations.

Downstream Evaluation. We evaluate zero-shot and two-shot performance on standard benchmarks, including ARC [63], BoolQ [13], HellaSwag [64], OBQA [39], PIQA [4], WinoGrande [43],

and MMLU [18], using the lm-evaluation-harness codebase [14]. For ARC-E, ARC-C, HellaSwag, OBQA, PIQA, and SciQ, we report accuracy norm; for other tasks, we report standard accuracy. Due to the page limitation, we only display the zero-shot evaluation results of medium and large models here in Tables 2 and 3. Zero-shot evaluation of small and XL models are displayed in Tables 11 and 12 in the appendix. Moreover, we also present 2-shot evaluation results in Tables 13, 14, 15 and 16 in the appendix.

For the medium -size (353M) models (Table 2 for 0-shot and Table 14 in appendix for 2-shot), TPA generally ties or outperforms all competing methods, achieving, for example, an average of 51.41% in zero-shot mode versus MHA's 50.11%, MQA's 50.44%, and MLA's 50.13%. When given two-shot prompts, TPA again leads with 53.12% average accuracy. A similar trend appears for the large -size (773M) models (Table 3), where TPA-KVonly attains the highest average (53.52% zero-shot). For the XL size models (1.5B) (Table 12 in the appendix), TPA-KV only achieves the highest average (55.03% zero-shot). Our experiments confirm that TPA consistently matches or exceeds the performance of established attention mechanisms (MHA, MQA, GQA, MLA) across medium and large model scales.

Figure 3: The training loss of medium-size (353M), large-size (773M) as well as XL-size (1.5B) models, with different attention mechanisms on the FineWeb-Edu 100B dataset.

<!-- image -->

Figure 4: The validation loss of medium-size (353M), large-size (773M) as well as XL-size (1.5B) models, with different attention mechanisms on the FineWeb-Edu 100B dataset.

<!-- image -->

Table 2: The evaluation results of medium models with different attention mechanisms pre-trained using FineWeb-Edu 100B dataset (0-shot with lm-evaluation-harness). The best scores in each column are bolded . Abbreviations: HellaSw. = HellaSwag, W.G. = WinoGrande.

| Method     |   ARC-E |   ARC-C |   BoolQ |   HellaSw. |   OBQA |   PIQA |   W.G. |   MMLU |   SciQ |   Avg. |
|------------|---------|---------|---------|------------|--------|--------|--------|--------|--------|--------|
| MHA        |   59.51 |   29.52 |   59.6  |      45.68 |   34.2 |  68.82 |  53.43 |  23.33 |   76.9 |  50.11 |
| MQA        |   57.62 |   31.91 |   59.45 |      45.69 |   35.4 |  69.31 |  53.51 |  26.47 |   74.6 |  50.44 |
| GQA        |   58.67 |   31.48 |   58.29 |      45.45 |   35.2 |  68.5  |  54.46 |  24.58 |   76.5 |  50.35 |
| MLA        |   56.65 |   29.52 |   57.83 |      46.05 |   34.6 |  69.42 |  52.8  |  24.62 |   79.7 |  50.13 |
| TPA-KVonly |   58.01 |   30.12 |   58.01 |      45.95 |   35.6 |  69.1  |  53.12 |  25.39 |   75.1 |  50.04 |
| TPA        |   58.38 |   31.57 |   59.39 |      46.83 |   37   |  70.02 |  54.06 |  25.52 |   79.9 |  51.41 |

## 6.2 Experimental Results on FlashTPA Decoding

This section presents an evaluation of FlashTPA's decoding time in comparison to several other optimized attention mechanisms. We benchmark FlashTPA against FlashMHA [45], FlashGQA, FlashMQA, and FlashMLA [23]. It is important to note that our current FlashTPA implementation utilizes Triton [57]. While the compared methods are typically available as highly optimized CUDA kernels, these experiments provide initial insights into FlashTPA's potential. Development of a CUDAbased FlashTPA kernel is ongoing and is expected to yield further performance improvements.

Table 3: The evaluation results of large models with different attention mechanisms pre-trained using the FineWeb-Edu 100B dataset (0-shot with lm-evaluation-harness). The best scores in each column are bolded . Abbreviations: HellaSw. = HellaSwag, W.G. = WinoGrande.

| Method     |   ARC-E |   ARC-C |   BoolQ |   HellaSw. |   OBQA |   PIQA |   W.G. |   MMLU |   SciQ |   Avg. |
|------------|---------|---------|---------|------------|--------|--------|--------|--------|--------|--------|
| MHA        |   59.93 |   33.62 |   61.93 |      50.63 |   36   |  71.06 |  55.41 |  22.87 |   81.2 |  52.52 |
| MQA        |   60.73 |   33.62 |   57.34 |      50.09 |   37   |  69.97 |  55.49 |  25.3  |   79.6 |  52.13 |
| GQA        |   61.66 |   34.3  |   58.72 |      49.85 |   38.4 |  71.16 |  53.75 |  25.23 |   77.6 |  52.3  |
| MLA        |   63.55 |   32.85 |   60.95 |      51.72 |   38.8 |  70.51 |  55.01 |  24.55 |   81.9 |  53.32 |
| TPA-KVonly |   63.26 |   34.13 |   61.96 |      50.66 |   37.2 |  72.09 |  55.25 |  26.06 |   81.1 |  53.52 |
| TPA        |   63.22 |   35.58 |   60.03 |      51.26 |   36.8 |  71.44 |  55.56 |  24.77 |   79.6 |  53.1  |

Figure 5: Decoding time comparison of different attention mechanisms with an embedding dimension of 2048 and d h = 64 . The y-axis represents log 2 ( time ) in seconds, and the x-axis represents log 2 ( sequence length ) . Each subfigure corresponds to a different batch size.

<!-- image -->

The evaluations were performed with batch sizes selected from { 1 , 2 , 4 , 8 , 16 } , model embedding dimensions ( d model) chosen from { 1024 , 2048 , 3072 } , and sequence lengths ranging from 2 12 (4,096) to 2 19 (524,288). For all experiments, the dimension per head ( d h ) was fixed at 64. The ranks for TPA's factorized components ( R Q , R K , R V ) were set to (16 , 1 , 1) , and for GQA configurations, the number of key-value head groups was 4. The decoding time per token, measured as log 2 ( time ) in seconds, is plotted against log 2 ( sequence length ) . Lower values on the y-axis indicate faster decoding times. Results are presented in Figure 5 for an embedding dimension of 2048 (corresponding to 32 attention heads). Additional results for embedding dimensions of 1024 (16 heads, Figure 8) and 3072 (48 heads, Figure 7) are provided in Appendix B. Figure 5 depicts these speed comparisons for an embedding dimension of 2048. The results indicate that FlashTPA (blue line) is highly competitive and often outperforms other attention mechanisms, especially as the sequence length increases.

## 7 Conclusion

We introduced Tensor Product Attention (TPA), which factorizes query, key, and value matrices into rankR tensor products dependent on the token's hidden state. Storing only the factorized key/value components during autoregressive decoding substantially decreases the KV memory size with improved performance compared with MHA, MQA, GQA, and MLA. The approach is fully compatible with RoPE (and can store pre-rotated keys). Variants of TPA include factorizing only the key/value or sharing basis vectors across tokens. Overall, TPA offers a powerful mechanism for compressing KV storage while improving the model performance, thereby enabling longer sequence contexts under constrained memory.

## Acknowledgements

We thank the anonymous reviewers and area chairs for their helpful comments. We acknowledge the compute credits provided by Fetch.ai.

## References

- [1] Muhammad Adnan, Akhil Arunkumar, Gaurav Jain, Prashant Nair, Ilya Soloveychik, and Purushotham Kamath. Keyformer: Kv cache reduction through key tokens selection for efficient generative inference. Proceedings of Machine Learning and Systems , 6:114-127, 2024.
- [2] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. GQA: training generalized multi-query transformer models from multi-head checkpoints. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023 , pages 4895-4901. Association for Computational Linguistics, 2023.
- [3] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609 , 2023.
- [4] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: reasoning about physical commonsense in natural language. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020 , pages 7432-7439. AAAI Press, 2020.
- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems , 33:1877-1901, 2020.
- [6] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712 , 2023.
- [7] Kerim Büyükakyüz. Olora: Orthonormal low-rank adaptation of large language models. arXiv preprint arXiv:2406.01775 , 2024.
- [8] Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Baobao Chang, Junjie Hu, et al. Pyramidkv: Dynamic kv cache compression based on pyramidal information funneling. arXiv preprint arXiv:2406.02069 , 2024.
- [9] Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. Longlora: Efficient fine-tuning of long-context large language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024 , 2024.
- [10] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509 , 2019.
- [11] Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamás Sarlós, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J. Colwell, and Adrian Weller. Rethinking attention with performers. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021 , 2021.
- [12] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin,

Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways. J. Mach. Learn. Res. , 24:240:1-240:113, 2023.

- [13] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) , pages 2924-2936, 2019.
- [14] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 07 2024.
- [15] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics , pages 249-256. JMLR Workshop and Conference Proceedings, 2010.
- [16] Insu Han, R Jayaram, A Karbasi, V Mirrokno, D Woodruff, and A Zandieh. Hyperattention: Long-context attention in near-linear time. In International Conference on Learning Representations . International Conference on Learning Representations, 2024.
- [17] Jiaao He and Jidong Zhai. Fastdecode: High-throughput gpu-efficient llm serving using heterogeneous pipelines. arXiv preprint arXiv:2403.11421 , 2024.
- [18] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021 , 2021.
- [19] Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W Mahoney, Yakun Sophia Shao, Kurt Keutzer, and Amir Gholami. Kvquant: Towards 10 million context length llm inference with kv cache quantization. arXiv preprint arXiv:2401.18079 , 2024.
- [20] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022 , 2022.
- [21] Jingcheng Hu, Houyi Li, Yinmin Zhang, Zili Wang, Shuigeng Zhou, Xiangyu Zhang, and Heung-Yeung Shum. Multi-matrix factorization attention. arXiv preprint arXiv:2412.19255 , 2024.
- [22] Ting Jiang, Shaohan Huang, Shengyue Luo, Zihan Zhang, Haizhen Huang, Furu Wei, Weiwei Deng, Feng Sun, Qi Zhang, Deqing Wang, et al. Mora: High-rank updating for parameterefficient fine-tuning. arXiv preprint arXiv:2405.12130 , 2024.
- [23] Shengyu Liu Jiashi Li. Flashmla: Efficient mla decoding kernels. https://github.com/ deepseek-ai/FlashMLA , 2025.
- [24] Andrej Karpathy. NanoGPT. https://github.com/karpathy/nanoGPT , 2022.
- [25] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning , pages 5156-5165. PMLR, 2020.

- [26] Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jaewoong Sim. { InfiniGen } : Efficient generative inference of large language models with dynamic { KV } cache management. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24) , pages 155-172, 2024.
- [27] Xiaoyu Li, Yingyu Liang, Zhenmei Shi, and Zhao Song. A tighter complexity analysis of sparsegpt. arXiv preprint arXiv:2408.12151 , 2024.
- [28] Vladislav Lialin, Sherin Muckatira, Namrata Shivagunde, and Anna Rumshisky. Relora: Highrank training through low-rank updates. In The Twelfth International Conference on Learning Representations , 2023.
- [29] Yan-Shuo Liang and Wu-Jun Li. Inflora: Interference-free low-rank adaptation for continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 23638-23647, 2024.
- [30] Yingyu Liang, Heshan Liu, Zhenmei Shi, Zhao Song, Zhuoyan Xu, and Junze Yin. Conv-basis: A new paradigm for efficient attention inference and gradient computation in transformers. arXiv preprint arXiv:2405.05219 , 2024.
- [31] Yingyu Liang, Jiangxuan Long, Zhenmei Shi, Zhao Song, and Yufa Zhou. Beyond linear approximations: A novel pruning approach for attention matrix. arXiv preprint arXiv:2410.11261 , 2024.
- [32] Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434 , 2024.
- [33] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 , 2024.
- [34] Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. KIVI: A tuning-free asymmetric 2bit quantization for KV cache. In Fortyfirst International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024 , 2024.
- [35] I Loshchilov. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 , 2017.
- [36] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983 , 2016.
- [37] Anton Lozhkov, Loubna Ben Allal, Leandro von Werra, and Thomas Wolf. Fineweb-edu: the finest collection of educational content, 2024.
- [38] Sadhika Malladi, Alexander Wettig, Dingli Yu, Danqi Chen, and Sanjeev Arora. A kernel-based view of language model fine-tuning. In International Conference on Machine Learning , pages 23610-23641. PMLR, 2023.
- [39] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing , pages 2381-2391, 2018.
- [40] Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. fairseq: A fast, extensible toolkit for sequence modeling. In Waleed Ammar, Annie Louis, and Nasrin Mostafazadeh, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Demonstrations , pages 48-53. Association for Computational Linguistics, 2019.
- [41] Weijieying Ren, Xinlong Li, Lei Wang, Tianxiang Zhao, and Wei Qin. Analyzing and reducing catastrophic forgetting in parameter efficient tuning. arXiv preprint arXiv:2402.18865 , 2024.
- [42] Luka Ribar, Ivan Chelombiev, Luke Hudlass-Galley, Charlie Blake, Carlo Luschi, and Douglas Orr. Sparq attention: Bandwidth-efficient LLM inference. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024 , 2024.

- [43] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 34, pages 8732-8740. Association for the Advancement of Artificial Intelligence (AAAI), 2020.
- [44] Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast weight programmers. In International Conference on Machine Learning , pages 9355-9366. PMLR, 2021.
- [45] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. Advances in Neural Information Processing Systems , 37:68658-68685, 2024.
- [46] Noam Shazeer. Fast transformer decoding: One write-head is all you need. arXiv preprint arXiv:1911.02150 , 2019.
- [47] Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202 , 2020.
- [48] Yiming Shi, Jiwei Wei, Yujia Wu, Ran Ran, Chengwei Sun, Shiyuan He, and Yang Yang. Loldu: Low-rank adaptation via lower-diag-upper decomposition for parameter-efficient fine-tuning. arXiv preprint arXiv:2410.13618 , 2024.
- [49] Zhenmei Shi, Jiefeng Chen, Kunyang Li, Jayaram Raghuram, Xi Wu, Yingyu Liang, and Somesh Jha. The trade-off between universality and label efficiency of representations from contrastive learning. In The Eleventh International Conference on Learning Representations , 2023.
- [50] Prajwal Singhania, Siddharth Singh, Shwai He, Soheil Feizi, and Abhinav Bhatele. Loki: Low-rank keys for efficient sparse attention. arXiv preprint arXiv:2406.02542 , 2024.
- [51] Jianlin Su. The extreme pull between cache and effect: From MHA, MQA, GQA to MLA. https://spaces.ac.cn/archives/10091 , May 2024.
- [52] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing , 568:127063, 2024.
- [53] Hanshi Sun, Li-Wen Chang, Wenlei Bao, Size Zheng, Ningxin Zheng, Xin Liu, Harry Dong, Yuejie Chi, and Beidi Chen. Shadowkv: Kv cache in shadows for high-throughput long-context llm inference. arXiv preprint arXiv:2410.21465 , 2024.
- [54] Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive network: A successor to transformer for large language models. arXiv preprint arXiv:2307.08621 , 2023.
- [55] Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. QUEST: query-aware sparsity for efficient long-context LLM inference. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024 , 2024.
- [56] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295 , 2024.
- [57] Philippe Tillet, HT Kung, and David Cox. Triton: An intermediate language and compiler for tiled neural network computations. In ACM SIGPLAN International Workshop on Machine Learning and Programming Languages co-located with PLDI . Association for Computing Machinery, 2019.
- [58] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 , 2023.

- [59] Yao-Hung Hubert Tsai, Shaojie Bai, Makoto Yamada, Louis-Philippe Morency, and Ruslan Salakhutdinov. Transformer dissection: An unified understanding for transformer's attention via the lens of kernel. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) , pages 4344-4353, 2019.
- [60] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems , 30, 2017.
- [61] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models. In International Conference on Machine Learning , pages 38087-38099. PMLR, 2023.
- [62] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024 , 2024.
- [63] Vikas Yadav, Steven Bethard, and Mihai Surdeanu. Quick and (not so) dirty: Unsupervised selection of justification sentences for multi-hop question answering. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) , pages 2578-2589, 2019.
- [64] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics . Association for Computational Linguistics, 2019.
- [65] Yuchen Zeng and Kangwook Lee. The expressive power of low-rank adaptation. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024 , 2024.
- [66] Hengyu Zhang. Sinklora: Enhanced efficiency and chat capabilities for long-context large language models. arXiv preprint arXiv:2406.05678 , 2024.
- [67] Michael Zhang, Kush Bhatia, Hermann Kumbong, and Christopher Re. The hedgehog &amp; the porcupine: Expressive linear attentions with softmax mimicry. In The Twelfth International Conference on Learning Representations , 2024.
- [68] Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. Adaptive budget allocation for parameter-efficient fine-tuning. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023 . OpenReview.net, 2023.
- [69] Ruiqi Zhang, Spencer Frei, and Peter L Bartlett. Trained transformers learn linear models in-context. arXiv preprint arXiv:2306.09927 , 2023.
- [70] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, et al. H2o: Heavy-hitter oracle for efficient generative inference of large language models. Advances in Neural Information Processing Systems , 36:34661-34710, 2023.
- [71] Hongbo Zhao, Bolin Ni, Junsong Fan, Yuxi Wang, Yuntao Chen, Gaofeng Meng, and Zhaoxiang Zhang. Continual forgetting for pre-trained vision models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 28631-28642, 2024.

## Appendix

| A Toward Faster Computation Without Materializing Q , K and V   | A Toward Faster Computation Without Materializing Q , K and V   | A Toward Faster Computation Without Materializing Q , K and V   |   17 |
|-----------------------------------------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------|------|
|                                                                 | A.1                                                             | Direct computation in factor space . . . . . . . . .            |   18 |
|                                                                 | A.2                                                             | Complexity: materialized vs. specialized computation            |   18 |
|                                                                 | A.3                                                             | Complexity of the specialized path . . . . . . . . .            |   19 |
|                                                                 | A.4                                                             | Inference-time decoding cost across mechanisms .                |   19 |
| B                                                               | More on FlashTPA Decoding Algorithm                             | More on FlashTPA Decoding Algorithm                             |   24 |
|                                                                 | B.1                                                             | Triton FlashTPA Decoding Kernel . . . . . . . . .               |   25 |
|                                                                 | B.2                                                             | Additional Experimental Results . . . . . . . . . .             |   25 |
| C                                                               | Higher-Order Tensor Product Attention                           | Higher-Order Tensor Product Attention                           |   28 |
|                                                                 | C.1                                                             | RoPE Compatibility in Higher-Order TPA . . . . .                |   28 |
| D                                                               | Proofs of Theorems                                              | Proofs of Theorems                                              |   29 |
|                                                                 | D.1                                                             | Proof of Theorem 3.1 . . . . . . . . . . . . . . . .            |   29 |
|                                                                 | D.2                                                             | Proof of Theorem C.1 . . . . . . . . . . . . . . . .            |   30 |
| E                                                               | More Related Work                                               | More Related Work                                               |   32 |
| F                                                               | More on Attention Mechanisms                                    | More on Attention Mechanisms                                    |   32 |
|                                                                 | F.1                                                             | Multi-Query Attention (MQA) . . . . . . . . . . .               |   32 |
|                                                                 | F.2                                                             | Grouped Query Attention (GQA) . . . . . . . . . .               |   33 |
|                                                                 | F.3                                                             | Multi-head Latent Attention (MLA) . . . . . . . .               |   33 |
|                                                                 | F.4                                                             | Multi-matrix Factorization Attention (MFA) . . . .              |   34 |
|                                                                 | F.5                                                             | Rotary Position Embedding (RoPE) . . . . . . . .                |   34 |
| G                                                               | More on TPA                                                     | More on TPA                                                     |   35 |
| H                                                               | More on Experiments                                             | More on Experiments                                             |   36 |
|                                                                 | H.1 Experimental Settings .                                     | . . . . . . . . . . . . . . .                                   |   36 |
|                                                                 | H.2                                                             | Additional Experimental Results . . . . . . . . . .             |   36 |
|                                                                 | H.3                                                             | Ablation Studies on Learning Rates . . . . . . . .              |   39 |
| I Broader Impacts and Limitations                               | I Broader Impacts and Limitations                               | I Broader Impacts and Limitations                               |   39 |

## A Toward Faster Computation Without Materializing Q , K and V

Our objective in this section is to compute attention without explicitly forming Q , K , V , by contracting their factorized representations in a cache- and throughput-friendly order. Recall from Equation (3.1) that each per-token slice Q t , K t , V t ∈ R h × d h is a sum of rank1 outer products. Unless otherwise stated we use the per-factor normalizations s Q =1 /R Q , s K =1 /R K , s V =1 /R V .

We make the batch/time/head/rank/value dimensions explicit and introduce the shorthands D := d h and E := d v (typically E = D ):

<!-- formula-not-decoded -->

Indices b, q, k, h, r, s, u, d, e denote batch, query position, key position, head, query-rank, key-rank, value-rank, feature ( D ), and value feature ( E ). We write T := T q = T k for full-sequence attention; in decoding, T q =1 and we denote the cache length by M = T k .

Convention. For a single token, the main text defines A ∗ ( x t ) ∈ R R ∗ × H and B ∗ ( x t ) ∈ R R ∗ × D , with Q t = 1 R Q A Q ( x t ) ⊤ B Q ( x t ) . Accordingly, throughout this appendix we index A Q as A Q [ b, q, r, h ] (rank-major). Some implementations may store A ∗ transposed as ( H × R ∗ ) for memory layout, this is equivalent, since all uses contract over the rank index.

High-level idea. We first compute headshared feature-space dot products between B Q and B K , then mix them with head-specific A Q , A K to obtain logits, apply the masked softmax, and finally aggregate values via A V , B V . This ordering avoids materializing any T q × h × D queries/keys/values.

<!-- image -->

incl.

s

V

Figure 6: Specialized TPA computation without materializing Q , K , V . Phase 1 (top): compute head-shared feature-space dot products P [ b, q, k, r, s ]= ⟨ B Q [ b, q, r, :] , B K [ b, k, s, :] ⟩ and mix them with head-specific factors A Q , A K to obtain logits L [ b, h, q, k ] . Phase 2 (bottom): apply the causal/padding mask and softmax to get α [ b, h, q, k ] , then aggregate values via A V , B V . Scalings s Q , s K , s V and 1 / √ D are folded into the corresponding phases. Dropout is omitted for clarity. Batch B , heads H , ranks R Q , R K , R V , and feature dims D,E are indicated in the nodes.

## A.1 Direct computation in factor space

Single head. For a fixed head h ∈ [ H ] and token indices ( q, k ) , using s Q =1 /R Q and s K =1 /R K we have

<!-- formula-not-decoded -->

and for values (with s V =1 /R V ), V ( h ) k = 1 R V ∑ R V u =1 a V, ( u ) k,h ( x k ) b V, ( u ) ( x k ) . The per-head attention output at query position q is then ∑ k softmax ( 1 √ D [ Q ( h ) ( K ( h ) ) ⊤ ] q, : ) k V ( h ) k .

Multi-head with head-shared feature dot-products. Define head-shared feature-space dot products P [ b, q, k, r, s ] = ⟨ B Q [ b, q, r, :] , B K [ b, k, s, :] ⟩ . With S = s Q s K √ D , we compute

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here mask [ b, q, k ] ∈{ 0 , -∞} is an additive mask in logit space that enforces causality and padding. Eqs. (A.2)-(A.3) make explicit that (i) feature-space dot products P are head-shared , and (ii) the rank normalizations s Q , s K , s V can be absorbed into the corresponding factor tensors (or into the scalar prefactors) without changing the computed attention output.

## A.2 Complexity: materialized vs. specialized computation

We compare two execution strategies. (i) Naïve/materialized: form Q , K , V explicitly and call standard kernels. (ii) Specialized: compute via Eq. (A.2)-(A.3) using head-shared feature-dot products and per-head rank contractions.

Standard MHA (baseline). Ignoring projections, full-sequence attention uses Θ( BHT 2 D ) FLOPs for scores and Θ( BHT 2 D ) for value aggregation, i.e., F MHA = 2Θ( BHT 2 D ) .

TPA (materialized). Forming Q , K , V from factors costs Θ( BTHD ( R Q + R K + R V )) after the linear projections; subsequent attention uses the same 2 Θ( BHT 2 D ) as MHA.

TPA (specialized). Using Eqs. (A.2)-(A.3) and writing T q = T k = T , the dominant FLOPs are

<!-- formula-not-decoded -->

Compared to F MHA, the specialized path reduces FLOPs whenever

<!-- formula-not-decoded -->

Dividing by HD yields ( R Q R K /H ) + ( R Q R K /D ) + R V ( E/D ) &lt; 2 . For E = D and small ranks (e.g., R Q = R K = R V =1 ), the inequality holds for typical H,D ≥ 2 and the benefit grows with larger H or D .

Memory traffic and peak working set. For full-sequence attention the naive path streams Q , K , V of size Θ( BTHD ) each. The specialized path streams factors only and needs the head-shared P tiles of size Θ( BT tile q T tile k R Q R K ) plus per-head tiles for the rank combine/value aggregation. In decoding with cache length M , the factorized KV cache uses ( R K + R V )( h + D ) numbers per token (cf. Section 3.3), vs. 2 hD for MHA; this reduction directly lowers memory bandwidth pressure.

```
Algorithm 1 Specialized TPA (no explicit Q , K , V ; causal) Require: A Q ∈ R B × T q × R Q × H , B Q ∈ R B × T q × R Q × D Require: A K ∈ R B × T k × R K × H , B K ∈ R B × T k × R K × D Require: A V ∈ R B × T k × R V × H , B V ∈ R B × T k × R V × E Require: scales s Q =1 /R Q , s K =1 /R K , s V =1 /R V ; mask mask ∈ { 0 , -∞} B × T q × T k Ensure: O ∈ R B × T q × H × E 1: P ← einsum( "bqrd,bksd->bqkrs" , B Q , B K ) ▷ ∈ R B × T q × T k × R Q × R K 2: L ← ( s Q s K / √ D ) · einsum( "bqrh,bksh,bqkrs->bhqk" , A Q , A K , P ) 3: L ← L +broadcast( mask ) ▷ causal/padding mask 4: α ← Softmax k ( L ) ▷ ∈ R B × H × T q × T k ; online/LSE in practice 5: O ← s V · einsum( "bhqk,bkuh,bkue->bhqe" , α, A V , B V ) 6: return transpose( O , "bhqe" → "bqhe" )
```

## A.3 Complexity of the specialized path

Combining the terms above gives complexity F TPA-spec = Θ( BT 2 R Q R K D ) + Θ( BHT 2 R Q R K ) + Θ( BHT 2 R V E ) , with the speed condition Eq. (A.4).

For a single query ( T q =1 ) against a cache of length M , the specialized FLOPs are

<!-- formula-not-decoded -->

while MHA uses 2 Θ( BHMD ) . This matches the asymptotics embodied in FLASHTPA (Section 5) and explains the regimes where R Q ≪ D and R K = R V ∈ { 1 , 2 } yield the largest gains.

We apply the causal mask before softmax and use an online log-sum-exp update for numerical stability (as in FlashAttention). The intermediate P ∈ R B × T q × T k × R Q × R K is evaluated blockwise in T k to keep peak memory linear in the block size; the same blocking naturally fuses with the masked softmax and the value aggregation step.

The constants s Q , s K , s V can be absorbed into either A ( · ) or B ( · ) at training time. We expose them explicitly only to make Eq. (A.4) transparent; The choice has no effect on softmax invariance or gradients.

The Triton kernel in Section 5 implements the blocked computation of P , the masked online softmax over k , and the fused value aggregation, mirroring Algorithm 1. This avoids creating any Q , K , V or full T q × T k temporaries beyond working tiles.

Compared with 2 Θ( BHT 2 D ) for MHA, the specialized path improves with small ( R Q , R K , R V ) and benefits further from pre-rotating B K for RoPE (cf. Section 3.2), which removes per-step rotations in decoding. Practical speed also depends on tiling, memory bandwidth, and kernel fusion; our measured gains in Section 6.2 align with the regime predicted by Eq. (A.4).

## A.4 Inference-time decoding cost across mechanisms

In autoregressive decoding, we generate the output for the current token x T given cached keys and values from T -1 previous tokens. We analyze the FLOPs for computing the attention output for this single query token and use M for the current cache length. For all mechanisms, we analyze the total Floating Point Operations (FLOPs) and the number of parameters in the attention layer, including the cost of projecting the current token's hidden state x T into its respective Query, Key, and Value representations. The parameter count formulas are taken from Table 1.

For Multi-Head Attention (MHA) , with H query heads and H distinct Key/Value heads, the complexity is determined by the dot-product attention and value aggregation steps.

- Projection: Projecting x T to get a query, key, and value vector for each of the H heads costs Θ( d model Hd h ) .
- Attention: Dot products and value aggregation over a cache of length M cost Θ(2 MHd h ) (ignoring softmax constants).
- Total MHA: The complexity is Θ( d model Hd h +2 MHd h ) .

Multi-Query Attention (MQA) uses H query heads but shares a single Key/Value head ( H kv = 1 ). The arithmetic complexity remains the same as MHA for the same number of query heads.

- Projection: Projecting for H query heads and 1 shared K/V head costs Θ( d model ( Hd h +2 d h )) .
- Attention: The interaction with the cache costs Θ(2 MHd h ) .
- Total MQA: The complexity is Θ( d model d h ( H +2) + 2 MHd h ) .

Grouped-Query Attention (GQA) uses H query heads and G Key/Value head groups ( H kv = G ). The arithmetic complexity is also identical to MHA.

- Projection: Projecting for H query heads and G K/V head groups costs Θ( d model ( Hd h +2 Gd h )) .
- Attention: The interaction with the cache costs Θ(2 MHd h ) .
- Total GQA: The complexity is Θ( d model d h ( H +2 G ) + 2 MHd h ) .

MQAand GQA significantly reduce the KV cache size and memory bandwidth compared to MHA. While the arithmetic FLOP count for the core attention computation (dot products and weighted sums) is 2 MHd h for all three (for fixed H,d h ), practical speedups for MQA/GQA arise from improved memory locality due to smaller K/V caches.

Multi-Head Latent Attention (MLA) , as described in Appendix F.3, uses H heads. Each head's (up-projected) query/key vectors have dimension d h + d R h . During decoding, however, the score computation against the cache can be decomposed into (i) a dot product in the cached latent space R d c for the content part and (ii) an additional RoPE dot product in R d R h for the positional part. Concretely, MLA caches c KV s ∈ R d c per past token s , aggregates values in R d c , and then up-projects once per step.

- Cached state: MLA caches the compressed KV latent c KV s ∈ R d c and the shared RoPE key component k R s ∈ R d R h per past token s .
- Projection (current token): Computing the query latents and the new cache entry (up to constant factors) costs

<!-- formula-not-decoded -->

corresponding to forming c Q , Q C , Q R , and computing/storing c KV and k R for the current token.

- Attention (cache interaction): Using the identity q C ⊤ t,i k C s,i = ( W i UK q C t,i ) ⊤ c KV s , the score against each cached token can be computed via a dot product in R d c plus the RoPE dot product in R d R h . The latent value can be aggregated in R d c and then up-projected once. The dominant cache-dependent cost is

<!-- formula-not-decoded -->

up to lower-order per-step terms such as Θ( Hd c d h ) .

- Total MLA: Θ ( d model d ′ c + d ′ c H ( d h + d R h ) + d model ( d c + d R h ) + MH (2 d c + d R h ) ) .

TPA. We use the FlashTPA Decoding algorithm (Algorithm 2) for FLOPs analysis, with N = 1 query token, M cached items, D as feature dimension for B Q / b K (typically d h ), and E for b V (typically d h ). For ranks ( R Q , R K , R V ) :

- Projection: Projecting the current token x T to all Q/K/V factors costs Θ ( d model ( R Q + R K + R V )( H + d h ) ) .
- Attention (cache interaction): Using Algorithm 2 with cache length M , the dominant cachedependent FLOPs are

<!-- formula-not-decoded -->

up to lower-order terms (masking/element-wise products and online-softmax bookkeeping).

- Total for TPA decoding: Θ ( d model ( R Q + R K + R V )( H + d h ) + M ( R Q R K D + HR Q R K + HR V E ) ) .

## Example Comparison I.

We compare the total Floating Point Operations (FLOPs) required to process a single token during autoregressive inference. This analysis separates the initial, constant projection cost from the attention cost, which scales linearly with the cache length M .

The following parameters are used for the comparison:

- Model Dimension: d model = 2048
- Heads: H = 32
- Head Dimension: d h = 64 (so D = E = d h )

- GQA Groups: G = 4

<!-- formula-not-decoded -->

## MHA(16.8M parameters):

<!-- formula-not-decoded -->

## GQA( G = 4 , 9.4M parameters):

<!-- formula-not-decoded -->

## MLA(9.8M parameters):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## TPA ( R Q = 16 , R K = 2 , R V = 2 , 8.1M parameters):

<!-- formula-not-decoded -->

## TPA ( R Q = 8 , R K = 1 , R V = 1 , 6.2M parameters):

<!-- formula-not-decoded -->

## TPA ( R Q = 8 , R K = 2 , R V = 2 , 6.6M parameters):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The analysis shows that TPA with low ranks offers a favorable trade-off. Reducing the query rank ( R Q ) from 16 to 8 further decreases both the projection and attention costs, making the TPA ( R Q =8 , R K =1 , R V =1 ) configuration the most computationally efficient in this comparison. Increasing key/value ranks (e.g., to R K =2 , R V =2 ) raises the attention cost linearly, remaining competitive with MHA for sufficiently long contexts where kernel fusion and blocking amortize memory traffic.

## Example Comparison II.

We now repeat the analysis for a larger model configuration to observe how these trade-offs scale. The following parameters for a larger model are used for this comparison:

- Model Dimension: d model = 4096
- Heads: H = 32
- Head Dimension: d h = 128 (so D = E = d h )
- GQA Groups: G = 4
- MLA Dimensions: d c = 512 , d R h = 64 , and d ′ c = 1536

## MHA(67.1M parameters):

<!-- formula-not-decoded -->

## GQA( G = 4 , 37.7M parameters):

```
Parameters = d model d h (2 H +2 G ) = 4096 · 128 · (2 · 32 + 2 · 4) ≈ 37 . 7 × 10 6 Projection = 4096 · (32 + 8) · 128 ≈ 21 . 0 × 10 6 Attention = 2 · M · 32 · 128 = 8192 M
```

## MLA(39.1M parameters):

Parameters = 1536(4096 + 4096 + 2048) + 4096(64 + 4096) + 512(4096 + 8192) ≈ 39 . 1 × 10 6

<!-- formula-not-decoded -->

= 4096

≈

20

.

2

·

1536 + 1536

×

·

32

·

(128 + 64) + 4096

10

6

<!-- formula-not-decoded -->

TPA ( R Q = 16 , R K = 1 , R V = 1 , 28.6M parameters):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

·

(512 + 64) + 32

<!-- formula-not-decoded -->

·

512

·

128

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For this larger configuration, TPA ( R Q =8 , R K =1 , R V =1 ) remains the clear leader in computational efficiency, with the lowest projection and attention costs. This highlights the value of tuning TPA ranks to balance expressiveness against compute.

## Example Comparison III.

Then we analyze a very large model configuration (e.g. MoE model with 1 ∼ 2T parameters) to examine the scaling properties of each architecture, where d model = H · d h to align MLA with other attention mechanisms. We also denote the number of parameters in the attention part for each layer.

The following parameters are used for this comparison:

- Model Dimension: d model = 7168
- Heads: H = 64
- Head Dimension: d h = 128 (so D = E = d h )
- GQA Groups: G = 8
- MLA Dimensions: d c = 512 , d R h = 64 , and d ′ c = 1536

## MHA(235M parameters):

```
Parameters model h Projection = 3 · 7168 · 64 · 128 ≈ 176 . 2 × 10 6 Attention = 2 · M · 64 · 128 = 16384 M
```

<!-- formula-not-decoded -->

## GQA( G = 8 , 132M parameters):

```
Parameters = d model d h (2 H +2 G ) = 7168 · 128 · (2 · 64 + 2 · 8) ≈ 132 × 10 6 Projection = 7168 · (64 + 16) · 128 ≈ 73 . 4 × 10 6
```

Attention = 2 · M · 64 · 128 = 16384 M

## MLA(101M parameters):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

6

6

<!-- formula-not-decoded -->

TPA ( R Q = 8 , R K = 1 , R V = 1 , 72.5M parameters): TPA ( R Q = 8 , R K = 2 , R V = 2 , 75.2M parameters):

<!-- formula-not-decoded -->

At this very large scale, the cost of MHA projections becomes prohibitive. While MLA's projection cost can be competitive, its attention cost scales with (2 d c + d R h ) and exceeds MHA for long sequences. TPA with low ranks ( R Q =8 , R K =1 , R V =1 ) yields the lowest attention cost and a substantially smaller projection cost, strengthening its advantage as model size increases.

## B More on FlashTPA Decoding Algorithm

In this section, we present FlashTPA for decoding in a hardware-friendly, numerically stable form and extend it to general key/value ranks R K , R V ≥ 1 . The algorithm computes attention without materializing Q , K , V or the full N × M attention matrix, by (i) forming head-shared feature-space dot products, (ii) mixing them with head-specific factors to obtain logits as in Eq. (A.2), and (iii) aggregating values as in Eq. (A.3) in a single online softmax pass.

Notation and shapes. We allow N query positions but decoding uses N =1 . Let B be batch, M the cache length, H heads, R Q , R K , R V ranks, and D,E feature sizes (typically D = E = d h ). Inputs:

<!-- formula-not-decoded -->

We use scalings s Q =1 /R Q , s K =1 /R K , s V =1 /R V , and s total =1 / D . Let mask ∈ { 0 , -∞} B × N × M encode causality/padding. If RoPE pre-rotation is used (Section 3.2), B cache K already includes positional phases; otherwise apply RoPE to B K on load.

## Algorithm 2 FlashTPA Decoding (general R , R , masked, online-LSE)

- return O ← s V · s [ ...,None ] permuted to ( B,N,H,E
- K V Require: A Q , B Q , A cache K , B cache K , A cache V , B cache V , mask ; s Q , s K , s V , s total Ensure: O ∈ R B × N × H × E 1: Initialize y ← 0 B × H × N × E , s ← 0 B × H × N , m ← ( -∞ ) B × H × N ▷ s accumulates ∑ exp( · ) ; log-sum-exp is log s + m 2: for each cache block m : m +∆ m ≤ M do 3: Load B K, blk ∈ R B × ∆ m × R K × D , A K, blk ∈ R B × ∆ m × R K × H 4: Load A V, blk ∈ R B × ∆ m × R V × H , B V, blk ∈ R B × ∆ m × R V × E , mask blk ∈ R B × N × ∆ m 5: (1) Head-shared feature dots: P ← einsum( 'bnrd,bmsd → bnmrs' , B Q , B K, blk ) ▷ R B × N × ∆ m × R Q × R K 6: (2) Per-head rank mixing to logits: 7: L blk ← ( s total s Q s K ) · einsum( 'bnrh,bmsh,bnmrs → bhnm' , A Q , A K, blk , P ) ▷ R B × H × N × ∆ m 8: L blk ←L blk +broadcast( mask blk ) 9: (3) Online softmax update (no α materialization): 10: m blk ← max m ( L blk ) ; p blk ← exp( L blk -m blk ) ; s blk ← ∑ m p blk 11: (4) Block value aggregation (fused over m,u ): 12: y blk ← einsum( 'bhnm,bmuh,bmue → bhne' , p blk , A V, blk , B V, blk ) ▷ R B × H × N × E 13: (5) Fuse blocks with log-sum-exp: 14: m new ← max( m , m blk ) ; y ← exp( m -m new )[ ..., None ] ⊙ y + exp( m blk -m new )[ ..., None ] ⊙ y blk 15: s ← exp( m -m new ) ⊙ s + exp( m blk -m new ) ⊙ s blk ; m ← m new 16: end for 17: y )

Step (1)-(2) implements Eq. (A.2); step (4)-(5) implements Eq. (A.3) while fusing the masked softmax with value aggregation via online log-sum-exp (as in FlashAttention), thereby avoiding any α materialization. When R K = R V =1 the contractions reduce to the simpler einsums in Figure 2.

Complexity and working set. Per block of ∆ m cache items, the dominant FLOPs are

<!-- formula-not-decoded -->

matching the specialized analysis in Appendix A.2 and the decoding bounds in Appendix A.4. Peak memory scales with tiles of B K , A K , A V , B V and the small temporaries P and V blk; neither Q , K , V nor the full N × M attention matrix is formed.

RoPE and masking. If keys are pre-rotated (Eq. (3.5)), B cache K needs no decoding-time rotation. Otherwise apply RoPE to B K, blk row-wise before step (1). The mask mask (zeros or -∞ ) is added to logits in step (2) and supports both causal and padding masks.

## B.1 Triton FlashTPA Decoding Kernel

We implement the experiments using Triton [57]; Algorithm 3 sketches the kernel corresponding to Algorithm 2. The provided kernel outline specializes to the frequently used case R K = R V =1 ; general ranks follow by tiling over R K , R V and replacing the rank-1 vector-matrix products with the corresponding small GEMMs in steps S 1 /S 2 and the value mixing path.

## B.2 Additional Experimental Results

The following figures present additional speed comparisons for different embedding dimensions, with d h = 64 maintained. The y-axis represents log 2 ( time ) in seconds (lower is faster), and the x-axis represents log 2 ( sequence length ) .

Detailed Analysis of Figure 5 (Embedding Dimension 2048): Figure 5 in the main paper depicts speed comparisons for an embedding dimension of 2048. The results indicate that FlashTPA (blue line) is highly competitive. Across all tested batch sizes (1 to 16) for d model = 2048 :

- MHA(orange line) is consistently the slowest mechanism, with its decoding time increasing most rapidly with sequence length.
- MQA(purple line) and GQA (green line) offer significant speedups over MHA and perform very similarly to each other, often overlapping in the plots.
- MLA (blue line) demonstrates strong performance, generally being faster than GQA, particularly at longer sequence lengths.
- FlashTPA shows excellent scalability. While at very short sequence lengths (e.g., 2 12 to 2 13 ), its performance is comparable to MQA/GQA and MLA, its decoding time increases at a notably slower rate with sequence length. Consequently, FlashTPA becomes significantly faster than GQA for sequences longer than approximately 2 14 .
- Compared to MLA, FlashTPA is consistently among the top two performers. In many instances, particularly at sequence lengths greater than 2 14 or 2 15 , FlashTPA matches or slightly surpasses MLA in speed. The logarithmic scale for time suggests that these differences can be substantial in practice for very long contexts. For example, at a sequence length of 2 19 across various batch sizes, FlashTPA often shows a visible advantage over MLA.

Figure 7 (Embedding Dimension 3072): With a larger embedding dimension of 3072, the relative performance trends observed in Figure 5 largely persist.

- FlashTPA (red line) remains one of the most efficient decoding methods. MHA (orange line) is consistently the slowest, while MQA (purple line) and GQA (green line) offer considerable improvements over MHA.
- MLA (blue line) and FlashTPA are the top two performers. FlashTPA consistently matches or exceeds the speed of MLA, particularly at longer sequence lengths (e.g., beyond 2 15 or 2 16 depending on the batch size). Its advantage often becomes more pronounced at the longest sequences tested ( 2 19 ). For instance, in batch size 1, TPA is clearly faster than MLA for sequence lengths 2 16 and above. A similar trend is seen across other batch sizes, where TPA maintains a competitive edge or becomes superior at longer contexts.

```
Algorithm 3 Triton FlashTPA Decoding Kernel Require: Input Tensors: A Q ( B,N,R Q , H ) , a K ( B,M,H ) , a V ( B,M,H ) , B Q ( B,N,R Q , D ) , b K ( B,M,D ) , b V ( B,M,E ) Require: Scaling factors: s total , s Q , s K , s V ; Dimensions: B,N (= 1) , M, H, R Q , D, E Require: Kernel Block dims: B H , B R , B D , B E ; Sequence Blocking: M block , M chunk Require: Program IDs: p id B , p id H , p id M Ensure: Partial Output O partial ( B, Num M , N, H, E ) , Log-Sum-Exp LSE partial ( B, Num M , H ) 1: b ← p id B ; h start ← p id H · B H 2: m block_start ← p id M · M block ; m block_end ← min(( p id M +1) · M block , M ) 3: ▷ B H , B R , B D , B E are tile sizes for dimensions H, R, D, E respectively. 4: ▷ Initialize accumulators for the head block 5: o accum ← 0 ( E × B H ) ; m max ←-∞ ( B H ) ; s exp_sum ← 0 ( B H ) ; c scale ← s total · s Q · s K 6: ▷ Load query factors (fixed for this program as N=1) 7: Load A ( R Q × B H ) Q, local from A Q [ b, 0 , : , h start . . . ] 8: Load B ( D × R Q ) Q, local from B Q [ b, 0 , : , :] ▷ Dimensions may be transposed after loading for matmul 9: ▷ Iterate over M chunk-sized chunks within the K/V block 10: for m chunk_start from m block_start to m block_end -1 step M chunk do 11: m chunk_end ← min( m chunk_start + M chunk , m block_end ) 12: M curr_chunk ← m chunk_end -m chunk_start 13: ▷ Load K/V factors for the current chunk 14: Load a K chunk ( M curr_chunk , B H ) ; a V chunk ( M curr_chunk , B H ) ; b K chunk ( M curr_chunk , D ) ; b V chunk ( E,M curr_chunk ) ▷ Layouts optimized for memory access and matmuls 15: b V chunk ← b V chunk · s V 16: ▷ Core TPA Score Calculation for the chunk 17: S 1 chunk ← MatMul ( b K chunk , B Q, local ) ▷ Shape: ( M curr_chunk , R Q ) 18: S 2 chunk ← MatMul ( S 1 chunk , A Q, local ) ▷ Shape: ( M curr_chunk , B H ) 19: S 3 chunk ← S 2 chunk ⊙ a K chunk · c scale ▷ Shape: ( M curr_chunk , B H ) 20: ▷ Online Softmax Update for the chunk 21: m max_local ← max axis =0 ( S 3 chunk ) ▷ Shape: ( B H ) 22: m max_new ← max( m max , m max_local ) 23: p num ← exp( S 3 chunk -m max_new [ None , :]) 24: s exp_sum_local ← ∑ axis =0 ( p num ) 25: p weighted_av ← ( p num / s exp_sum_local [ None , :]) ⊙ a V chunk 26: o chunk ← MatMul ( b V chunk , p weighted_av ) ▷ Shape: ( E,B H ) 27: ▷ Update global (M-block level) accumulators 28: s exp_sum_prev_rescaled ← s exp_sum · exp( m max -m max_new ) 29: s exp_sum ← s exp_sum_prev_rescaled + s exp_sum_local 30: ratio ← s exp_sum_local / s exp_sum ▷ This is s exp_sum_local / s exp_sum_new 31: o accum ← (1 -ratio ) · o accum + ratio · o chunk 32: m max ← m max_new 33: end for 34: ▷ Store partial results for this program's (batch, head_block, M_block) 35: Store o accum into O partial [ b, p id M , 0 , h start . . . , :] 36: LSE val ← log( s exp_sum ) + m max 37: Store LSE val into LSE partial [ b, p id M , h start . . . ]
```

This suggests that FlashTPA's efficiency is well-maintained even as the model's embedding dimension increases.

Figure 8 (Embedding Dimension 1024): For a smaller embedding dimension of 1024, similar trends are observed:

- FlashTPA (red line) is highly competitive. MHA (orange line) remains the least performant. MQA (purple line) and GQA (green line) are faster than MHA.
- However, as sequence length increases, both MLA (blue line) and FlashTPA demonstrate superior scalability. FlashTPA generally matches or outperforms MLA, particularly for sequences longer than 2 15 . For example, with a batch size of 16, TPA shows a clear speed advantage over MLA for sequence lengths 2 16 and greater.

These results across different embedding dimensions highlight the robustness of FlashTPA's decoding speed advantages, especially for long sequences where it consistently ranks as one of the fastest, if not the fastest, attention mechanisms among those tested.

Figure 7: Decoding time comparison of different attention mechanisms with an embedding dimension of 3072 and d h = 64 .

<!-- image -->

Figure 8: Decoding time comparison of different attention mechanisms with an embedding dimension of 1024 and d h = 64 .

<!-- image -->

## C Higher-Order Tensor Product Attention

All prior discussions have focused on TPA where the query, key, and value matrices (e.g., Q t ∈ R h × d h ) are formed as a sum of R Q components. Each component is an outer product of two context-dependent vectors, one spanning the head dimension ( R h ) and the other spanning the featureper-head dimension ( R d h ), as detailed in Section 3.1 (e.g., Q t = 1 R Q A Q ( x t ) ⊤ B Q ( x t ) implies Q t = ∑ r a r b ⊤ r where a r are columns of A ⊤ Q and b ⊤ r are rows of B Q ). We now generalize this by introducing additional latent factors in the construction of the feature-per-head vectors, leading to what we term higher-order TPA. This approach allows for more complex interactions in forming these feature vectors.

For instance, in a third-order factorization, the query tensor Q t ∈ R h × d h for a single token t is constructed as:

<!-- formula-not-decoded -->

where a Q r ( x t ) ∈ R h . The term b Q r ( x t ) ∈ R d b and the newly introduced factor c Q r ( x t ) ∈ R d c first form a matrix b Q r ( x t ) ⊗ c Q r ( x t ) ∈ R d b × d c via an outer product (as defined in Section 2). This matrix is then vectorized by vec( · ) into a column vector of dimension d h = d b d c . The final query Q t is formed by the sum of outer products between a Q r ( x t ) and these resulting d h -dimensional vectors. Analogous expansions apply to K t and V t .

The additional factor c Q r ( x t ) can be viewed as a learnable, context-dependent modulation or gating term for the features generated by b Q r ( x t ) .

<!-- formula-not-decoded -->

This higher-order construction can enhance expressiveness. While introducing c Q r increases the parameter count for the factors, it might allow for the use of smaller base ranks ( R Q , R K , R V ) to achieve comparable representational power, thus offering a different design choice. One could also explore tying or sharing c Q r across queries, keys, and values to manage parameter overhead.

From a memory perspective, during inference, higher-order TPA maintains the benefit of factorized KV caching. Only the constituent factors a K ( x t ) , b K ( x t ) , c K ( x t ) (and similarly for values) for each past token need to be stored. A trade-off arises between model capacity and the overhead of memory and computation. Higher-order tensor decompositions can provide additional flexibility and potentially increased capacity.

## C.1 RoPE Compatibility in Higher-Order TPA

Rotary positional embeddings (RoPE) remain compatible with higher-order factorizations. In secondorder TPA, RoPE applies rotations to the d h -dimensional feature vectors. This compatibility extends to higher-order TPA. Consider the case where RoPE is intended to primarily rotate feature pairs derived from the b Q r ( x t ) components, while the structural influence of c Q r ( x t ) components on the d h -dimensional vector is preserved. More formally, RoPE acts on the d h -dimensional vector vec( b Q r ⊗ c Q r ) such that the transformation is equivalent to rotating b Q r to ˜ b Q r = R t b Q r (where R t is the RoPE rotation matrix for d b dimensions) and then forming vec( ˜ b Q r ⊗ c Q r ) . This is achieved by a specific RoPE transformation matrix T t acting on the full d h -dimensional vector, as stated in the following theorem.

Theorem C.1 (RoPE Compatibility in Higher-Order TPA) . Consider the higher-order (3-order) Tensor Product Attention (TPA) query factorization

<!-- formula-not-decoded -->

where a Q r ( x t ) ∈ R h , b Q r ( x t ) ∈ R d b , c Q r ( x t ) ∈ R d c , with d h = d b d c . Define the RoPE-transformed query as ˜ Q t = RoPE t ( Q t ) = Q t T t , where

<!-- formula-not-decoded -->

I d c is the identity matrix of size d c × d c , and R t ∈ R d b × d b ( d b ∈ Z + is even) is the standard RoPE block-diagonal matrix composed of 2 × 2 rotation matrices:

<!-- formula-not-decoded -->

for t ∈ { 1 , . . . , T } and j ∈ { 1 , . . . , d b / 2 } . The transformation T t = I d c ⊗ ( R t ) ⊤ operates on the d h -dimensional vectorized features by post-multiplication. This structure of T t ensures that the rotation effectively applied to the b Q r ( x t ) component (which is a column vector) corresponds to a pre-multiplication by R t , as detailed in the proof (Appendix D.2). This preserves the structure induced by c Q r ( x t ) while rotating b Q r ( x t ) .

Under these conditions, the RoPE-transformed query RoPE t ( Q t ) admits a higher-order TPA factorization of the same rank R Q :

<!-- formula-not-decoded -->

where ˜ b Q r ( x t ) = R t b Q r ( x t ) .

Please see Appendix D.2 for the proof. For fourth-order or higher, this result still holds.

To assess its empirical performance, we implemented third-order TPA. Table 4 lists the evaluation results for a small model. These results provide an initial indication of its viability. A comprehensive comparison with second-order TPA variants of similar parameter counts or ranks would be necessary to fully evaluate the trade-offs.

Table 4: The evaluation results of small models with third-order TPA pre-trained using FineWebEdu 100B dataset with lm-evaluation-harness. Abbreviations: HellaSw. = HellaSwag, W.G. = WinoGrande.

| Few-shot   |   ARC-E |   ARC-C |   BoolQ |   HellaSw. |   OBQA |   PIQA |   W.G. |   MMLU |   SciQ |   Avg. |
|------------|---------|---------|---------|------------|--------|--------|--------|--------|--------|--------|
| 0-shot     |   49.24 |   24.91 |   57.06 |      34.01 |   31.8 |  63.33 |  50.59 |  23.23 |   66.9 |  44.56 |
| 2-shot     |   53.37 |   25.34 |   48.78 |      34    |   29.2 |  62.79 |  52.33 |  26.41 |   75.3 |  45.28 |

## D Proofs of Theorems

## D.1 Proof of Theorem 3.1

Proof. Because RoPE is a linear orthogonal transform, we can write

<!-- formula-not-decoded -->

where T t is the block-diagonal matrix encoding RoPE. This allows us to define

<!-- formula-not-decoded -->

thereby obtaining where

<!-- formula-not-decoded -->

This equality confirms that the relative positional encoding between queries and keys is preserved under TPA's factorization and RoPE's rotation. Thus, TPA maintains compatibility with RoPE. This completes the proof of Theorem 3.1.

## D.2 Proof of Theorem C.1

Theorem C.1 addresses the compatibility of RoPE with higher-order (specifically, 3rd-order) Tensor Product Attention. The theorem considers the query factorization:

<!-- formula-not-decoded -->

where a Q r ( x t ) ∈ R h (column vector), b Q r ( x t ) ∈ R d b (column vector), c Q r ( x t ) ∈ R d c (column vector), and d h = d b d c . The term b Q r ( x t ) ⊗ c Q r ( x t ) is interpreted as the matrix M r =

<!-- formula-not-decoded -->

Similarly, for the key tensor K s , we have

<!-- formula-not-decoded -->

which defines and thus

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, consider the product of the rotated queries and keys:

<!-- formula-not-decoded -->

Since T t and T s encode positional rotations, the product T t T ⊤ s corresponds to a relative rotation T t -s . Therefore, we can express the above as

<!-- formula-not-decoded -->

This shows that

<!-- formula-not-decoded -->

Focusing on individual heads i , the above matrix equality implies:

<!-- formula-not-decoded -->

b Q r ( x t )( c Q r ( x t )) ⊤ ∈ R d b × d c . The notation a ⊗ v for a ∈ R h and v ∈ R d h (column vectors) implies the outer product av ⊤ . Thus, Q t = 1 R Q ∑ R Q r =1 a Q r ( x t )(vec( M r )) ⊤ .

The RoPE-transformed query is defined as ˜ Q t = RoPE t ( Q t ) = Q t T t . Crucially, for the theorem's conclusion to hold as intended (i.e., that the b Q r component is transformed by pre-multiplication with the standard RoPE matrix R t ), the global transformation matrix T t ∈ R d h × d h (that post-multiplies Q t ) is given by:

<!-- formula-not-decoded -->

where I d c is the d c × d c identity matrix, and R t ∈ R d b × d b is the standard RoPE block-diagonal matrix that pre-multiplies d b -dimensional column vectors (as defined explicitly in the theorem statement in Section C).

The theorem claims that, under these conditions, ˜ Q t admits a higher-order TPA factorization:

<!-- formula-not-decoded -->

where ˜ b Q r ( x t ) = R t b Q r ( x t ) .

Proof. Let a Q r ≡ a Q r ( x t ) , b Q r ≡ b Q r ( x t ) , and c Q r ≡ c Q r ( x t ) for brevity. Let M r = b Q r ( c Q r ) ⊤ ∈ R d b × d c . Let v r = vec( M r ) ∈ R d h be the column vector obtained by stacking the columns of M r . The query tensor is Q t = 1 R Q ∑ R Q r =1 a Q r ( v r ) ⊤ .

The RoPE transformation is ˜ Q t = Q t T t . Substituting the factorization and the revised definition of T t :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let's analyze the transformed vector part for the r -th component: ( v r ) ⊤ ( I d c ⊗ ( R t ) ⊤ ) . This row vector is the transpose of (( I d c ⊗ ( R t ) ⊤ ) ⊤ v r ) . Let's compute the pre-multiplying matrix:

<!-- formula-not-decoded -->

So, the column vector transformation is ( I d c ⊗ R t ) v r . Substitute v r = vec( M r ) = vec( b Q r ( c Q r ) ⊤ ) :

<!-- formula-not-decoded -->

We use the Kronecker product identity: ( B 0 ⊤ ⊗ A 0 ) vec( X 0 ) = vec( A 0 X 0 B 0 ) . To match our expression ( I d c ⊗ R t ) vec( M r ) , we identify: A 0 = R t , B 0 ⊤ = I d c = ⇒ B 0 = I d c , X 0 = M r = b Q r ( c Q r ) ⊤ . Applying the identity, we get:

<!-- formula-not-decoded -->

Let ˜ b Q r = R t b Q r . This is precisely the transformation for the b Q r component as claimed in the theorem. So the transformed column vector is vec( ˜ b Q r ( c Q r ) ⊤ ) . The corresponding row vector in the sum for ˜ Q t is therefore (vec( ˜ b Q r ( c Q r ) ⊤ )) ⊤ .

Substituting this back into the expression for ˜ Q t :

<!-- formula-not-decoded -->

This is equivalent to the theorem's claimed factorization, using the definition a ⊗ col \_ vec = a ( col \_ vec ) ⊤ :

<!-- formula-not-decoded -->

where ˜ b Q r = R t b Q r . This completes the proof, showing that RoPE can be consistently applied to higher-order TPA representations if the global RoPE transformation matrix T t (that post-multiplies Q t ) is appropriately defined as I d c ⊗ ( R t ) ⊤ , ensuring that the standard RoPE matrix R t effectively pre-multiplies the b Q r component.

## E More Related Work

Transformers and Attention. As a sequence-to-sequence architecture, Transformer [60] introduced Multi-Head Attention (MHA), enabling more effective capture of long-range dependencies. Subsequent work has explored a variety of attention mechanisms aimed at improving scalability and efficiency, including sparse patterns [10, 49, 16, 30, 27, 31], kernel-based projections [11], and linearized transformers [59, 25, 44, 69, 54, 67]. To decrease memory usage and circumvent the limitation of memory bandwidth in training, [46] proposed Multi-Query Attention (MQA) where multiple query heads share the same key head and value head. To tackle the issue of quality degradation and instability in training, Grouped-Query Attention (GQA) [2] divides queries into several groups, and each group of queries shares a single key head and value head. Recently, DeepSeek-V2 [32] applied multihead latent attention (MLA) to achieve better performance than MHA while reducing KV cache in inference time by sharing the same low-rank representation of key and value. Concurrently, [21] proposed Multi-matrix Factorization Attention (MFA), which can be simply seen as MQA with lowrank factorized Q. Compared to the approaches above, TPA applied contextual tensor decompositions to represent queries, keys, and values activations compactly, achieving better reduction on the size of KV cache with improved performance.

KV Cache Optimization. During the auto-regressive inference of Transformers, key and value (KV) tensors from previous tokens are cached to avoid recomputation, a technique first proposed by [40]. This Key-Value (KV) cache, while crucial for efficiency, consumes significant memory and can introduce latency bottlenecks due to memory bandwidth limitations [1]. Consequently, various studies have explored methods to mitigate these issues. These include KV cache eviction strategies that discard less significant tokens [70, 62, 8, 1], dynamic sparse attention mechanisms focusing on selected keys and values [42, 55, 50], offloading the KV cache to CPU memory [17, 26, 53], and quantizing the KV cache [61, 34, 19]. In contrast to these approaches, TPA focuses on reducing the intrinsic size of the KV cache by employing tensor-decomposed key and value representations.

Low-Rank Factorizations. Low-rank approximations are widely used to compress model parameters and reduce computational complexity. Notable examples include LoRA [20], which factorizes weight updates during fine-tuning, and its derivatives tailored for various training scenarios such as efficient pretraining (ReLoRA [28], MoRA [22]), long-context training (LongLoRA [9], SinkLoRA [66]), and continual training (InfLoRA [29], GS-LoRA [71], I-LoRA [41]). These methods generally produce static low-rank expansions that are independent of the input context. Theoretical justifications for the expressiveness of low-rank approximations have been provided by [38, 65]. Initialization strategies for these factorization matrices have also been explored: OLoRA [7] utilizes QR-decomposition of pretrained weights for improved language model performance, while LoLDU [48] employs LDU-decomposition to accelerate LoRA training. Furthermore, AdaLoRA [68] uses Singular Value Decomposition (SVD) on pretrained weights and introduces parameter importance scores to dynamically adjust ranks. TPA, in contrast, constructs Q, K, and V tensors using contextually-aware factorizations, allowing for dynamic adaptation based on the input.

## F More on Attention Mechanisms

## F.1 Multi-Query Attention (MQA)

Multi-Query Attention (MQA) [46] significantly reduces memory usage, particularly for the KV cache, by sharing a single key and value projection across all attention heads, while each head maintains a unique query projection. Given a sequence of input embeddings X ∈ R T × d model , the query, shared key, and shared value tensors are computed as:

<!-- formula-not-decoded -->

Thus, each head i uses a distinct query projection Q i ∈ R T × d h but shares the common key K shared ∈ R T × d h and value V shared ∈ R T × d h tensors. The weight matrices are:

<!-- formula-not-decoded -->

The resulting MQA operation is:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By sharing key and value projections, MQA substantially reduces memory demands, especially for the KV cache during autoregressive inference. However, this comes at the cost of reduced model expressivity, as all heads must utilize the same key and value representations.

## F.2 Grouped Query Attention (GQA)

Grouped Query Attention (GQA) [2] generalizes Multi-Head Attention (MHA) and MQA by dividing the total h attention heads into G groups. Within each group, heads share a common key and value projection, while each head maintains its own unique query projection. Formally, let g ( i ) denote the group index for head i ∈ { 1 , . . . , h } , where g ( i ) ∈ { 1 , . . . , G } . The projections are:

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Here, W g K and W V g are the shared weight matrices for group g , each in R d model × d h , and W Q i ∈ R d model × d h is the query weight matrix for head i . The complete output is again a concatenation of all heads:

<!-- formula-not-decoded -->

By varying G from 1 (equivalent to MQA) to h (equivalent to MHA), GQA offers a trade-off between memory efficiency and model capacity.

## F.3 Multi-head Latent Attention (MLA)

Multi-head Latent Attention (MLA), as used in DeepSeek-V2 [32] and DeepSeek-V3 [33], introduces low-rank compression for keys and values to reduce KV caching costs during inference.

<!-- formula-not-decoded -->

Here, W DKV ∈ R d model × d c projects to a compressed dimension d c , W UK ∈ R d c × ( d h h ) up-projects the compressed keys, W KR ∈ R d model × d R h projects to a residual key component for RoPE, and W UV ∈ R d c × ( d h h ) up-projects the compressed values. C KV ∈ R T × d c is the shared compressed KV latent (where d c ≪ d h h ). The RoPE transformation is applied to a separate key embedding K R ∈ R T × d R h . Thus, only C KV and K R are cached, reducing KV memory usage while largely preserving performance compared to standard MHA [60].

MLA also compresses the queries, lowering their training-time memory footprint:

<!-- formula-not-decoded -->

where

The weight matrices are W DQ ∈ R d model × d ′ c , W UQ ∈ R d ′ c × ( d h h ) , and W QR ∈ R d ′ c × ( d R h h ) . Here, C Q ∈ R T × d ′ c (where d ′ c ≪ d h h ) is the compressed query latent. The final query Q i for each head, formed by concatenating Q C i and Q R i , has a dimension of d h + d R h .

Given compressed queries, keys, and values, the final attention output for the t -th token is:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where V i is typically V C i as no residual value component is explicitly defined, and W O ∈ R ( d h h ) × d model is the output projection.

During inference, C KV and K R are cached to accelerate decoding. In detail, if RoPE were ignored for the compressed components, the inner product q ⊤ t,i k s,i (where q t,i , k s,i ∈ R d h ) of the i -th head between t -th token query and s -th token key could be calculated using the current hidden state x t ∈ R d model and the cached latent state c KV s ∈ R d c for the s -th token:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where W ( · ) i denotes the i -th head's portion of the respective weight matrix. The term [ W DQ i W UQ i ( W i UK ) ⊤ ] could be pre-computed for faster decoding. However, as noted by [51], this pre-computation strategy is not directly compatible with RoPE if RoPE were applied to these compressed representations. RoPE applies a rotation matrix T t ∈ R d h × d h based on position t (see Section F.5), satisfying T t T ⊤ s = T t -s (Equation F.4). If RoPE were applied to the up-projected Q C and K C :

<!-- formula-not-decoded -->

Unlike Equation (F.2), acceleration by pre-computing the term [ W DQ i W UQ i T t -s ( W i UK ) ⊤ ] is not possible because it depends on the relative position ( t -s ) and thus varies for different ( t, s ) pairs. To maintain RoPE compatibility while benefiting from compression, MLA introduces an additional, smaller key component K R (and similarly Q R ) to which RoPE is applied, while the main compressed components K C and V C (derived from C KV ) remain RoPE-free. As we will demonstrate in Section 3.2 of the main paper, TPA offers a different approach to integrate RoPE efficiently with factorized attention through its tensor product formulation.

## F.4 Multi-matrix Factorization Attention (MFA)

[21] proposed Multi-matrix Factorization Attention (MFA), which can be conceptualized as a variation of MQA where the shared key and value projections have a dimension d c , and the query projection for each head is low-rank factorized:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

## F.5 Rotary Position Embedding (RoPE)

Many recent LLMs use rotary position embedding (RoPE; 52) to encode positional information in the query/key vectors. Specifically, for a vector at position t , RoPE applies a rotation matrix T t ∈ R d × d (where d is the dimension of the query/key vectors, typically d h per head). T t is a block-diagonal

(

cos(

sin(

matrix composed of d/

2

rotation blocks of the form tθ

tθ

j

j

)

)

-

sin(

cos(

tθ

tθ

j

j

)

for

j

1

, . . . , d/

2

.

The frequencies { θ j } are typically defined as θ j = base -2 j/d , with a common base like 10000 . If q t ∈ R d is a query (or key) row vector for a specific head at position t , RoPE is applied as:

<!-- formula-not-decoded -->

)

)

∈ {

}

A key property of RoPE is that the inner product between RoPE-transformed vectors depends only on their relative position. For a query q t and key k s : ( q t T t )( k s T s ) ⊤ = q t T t T ⊤ s k ⊤ s = q t T t -s k ⊤ s . This relies on the property:

<!-- formula-not-decoded -->

which embeds relative positional information ( t -s ) into the attention scores.

## G More on TPA

Parameter Initialization for TPA Factors. We initialize the weight matrices for TPA factors, such as W a Q r , W a K r , W a V r , W b Q r , W b K r , and W b V r (or their combined forms W a Q , W b Q , etc.), using Xavier initialization [15]. Specifically, each entry of a weight matrix is drawn from a uniform distribution U ( -bound, bound ) , where bound = √ 6 / ( n in + n out ) . Here, n in and n out are the input and output dimensions of the respective weight matrix. This initialization strategy is chosen to help maintain the variance of activations and gradients as they propagate through the network layers, contributing to stable training.

TPA with Non-contextual B. In Section 4.1, we have introduced TPA with non-contextual A, where head-dimension factors a Q r , a K r , a V r ∈ R h are fixed. Conversely, one may fix the token-dimension factors b Q r , b K r , b V r ∈ R d h as learned parameters, while allowing a Q r ( x t ) , a K r ( x t ) , a V r ( x t ) to adapt to the input token x t . The key tensor for token t , K t ∈ R h × d h , would then be constructed as:

<!-- formula-not-decoded -->

A similar formulation applies to values. This configuration might be effective if the fundamental token-level features (captured by b r ) are relatively stable, while their combination across heads (captured by a r ( x t ) ) needs to adapt to the context. Performance comparisons for TPA with noncontextual A factors versus non-contextual B factors on small and medium-sized models are presented in Tables 5, 6, 7, and 8.

Table 5: Evaluation results of small models with TPA using non-contextual A or B factors, pre-trained on FineWeb-Edu 100B dataset (0-shot with lm-evaluation-harness). Abbreviations: HellaSw. = HellaSwag, W.G. = WinoGrande.

| Method          |   ARC-E |   ARC-C |   BoolQ |   HellaSw. |   OBQA |   PIQA |   W.G. |   MMLU |   SciQ |   Avg. |
|-----------------|---------|---------|---------|------------|--------|--------|--------|--------|--------|--------|
| TPA (non-ctx-A) |   50.17 |   25.6  |   57.95 |      36.13 |   31.4 |  64.8  |  49.57 |  24.88 |   64.8 |  45.03 |
| TPA (non-ctx-B) |   47.39 |   26.37 |   54.8  |      32.71 |   30.2 |  63.38 |  50.2  |  23.13 |   64.8 |  43.66 |

Table 6: Evaluation results of small models with TPA using non-contextual A or B factors, pre-trained on FineWeb-Edu 100B dataset (2-shot with lm-evaluation-harness). Abbreviations: HellaSw. = HellaSwag, W.G. = WinoGrande.

Table 7: Evaluation results of medium models with TPA using non-contextual A or B factors, pretrained on FineWeb-Edu 100B dataset (0-shot with lm-evaluation-harness). Abbreviations: HellaSw. = HellaSwag, W.G. = WinoGrande.

| Method          |   ARC-E |   ARC-C |   BoolQ |   HellaSw. |   OBQA |   PIQA |   W.G. |   MMLU |   SciQ |   Avg. |
|-----------------|---------|---------|---------|------------|--------|--------|--------|--------|--------|--------|
| TPA (non-ctx-A) |   55.09 |   27.65 |   53.82 |      36.24 |   30.2 |  64.53 |  50.75 |  26.01 |   78.6 |  46.99 |
| TPA (non-ctx-B) |   50.8  |   26.96 |   57.65 |      32.4  |   29.4 |  63.22 |  49.57 |  23.96 |   66.4 |  44.48 |

TPA KV Only. A simpler variant involves using a standard linear projection for queries,

| Method          |   ARC-E |   ARC-C |   BoolQ |   HellaSw. |   OBQA |   PIQA |   W.G. |   MMLU |   SciQ |   Avg. |
|-----------------|---------|---------|---------|------------|--------|--------|--------|--------|--------|--------|
| TPA (non-ctx-A) |   58.96 |   31.48 |   59.76 |      45.07 |   34.8 |  69.21 |  53.59 |  25.42 |   76.4 |  50.52 |
| TPA (non-ctx-B) |   55.43 |   29.69 |   58.32 |      40.77 |   34.4 |  66.92 |  51.38 |  25.66 |   71.1 |  48.19 |

<!-- formula-not-decoded -->

and factorize only the key and value tensors ( K t , V t ). This approach, termed TPA-KVonly, maintains the standard query projection mechanism but still achieves significant KV cache reduction through factorized key and value representations.

Table 8: Evaluation results of medium models with TPA using non-contextual A or B factors, pretrained on FineWeb-Edu 100B dataset (2-shot with lm-evaluation-harness). Abbreviations: HellaSw. = HellaSwag, W.G. = WinoGrande.

| Method          |   ARC-E |   ARC-C |   BoolQ |   HellaSw. |   OBQA |   PIQA |   W.G. |   MMLU |   SciQ |   Avg. |
|-----------------|---------|---------|---------|------------|--------|--------|--------|--------|--------|--------|
| TPA (non-ctx-A) |   65.45 |   33.79 |   56.88 |      45.23 |   33.6 |  68.61 |  54.22 |  25    |   85   |  51.98 |
| TPA (non-ctx-B) |   61.2  |   30.2  |   55.93 |      40.45 |   34.4 |  68.23 |  51.78 |  26.11 |   78.1 |  49.6  |

TPA KV with Shared B . Further parameter reduction can be achieved by sharing the tokendimension factors b r between keys and values:

<!-- formula-not-decoded -->

This sharing reduces both parameter count and the KV cache footprint. Although it constrains K t and V t to be constructed from the same token-level basis vectors, this variant can still offer strong performance with additional memory savings.

Nonlinear Head Factors. Instead of using purely linear transformations to derive the contextual head-dimension factors a Q r ( x t ) , a K r ( x t ) , a V r ( x t ) , one can introduce element-wise nonlinearities (e.g., sigmoid σ ( · ) or softmax). Applying softmax, for instance, to the coefficients that generate a r ( x t ) could be interpreted as a form of Mixture-of-Heads, where the network learns to dynamically weight different head configurations based on the input context.

Discussion. These variants highlight the flexibility of the TPA framework, allowing for different trade-offs between memory efficiency, computational cost, and model expressiveness. By carefully choosing which factor components (head-dimension or token-dimension) are contextual versus noncontextual, and by adjusting the ranks ( R Q , R K , R V ) , TPA can not only unify existing mechanisms like MHA, MQA, and GQA but also significantly reduce KV cache size-potentially by an order of magnitude-during autoregressive inference.

## H More on Experiments

## H.1 Experimental Settings

We list the main architecture hyper-parameters and training devices in Table 9. For all models, the head dimension d h is fixed at 64. Specific architectural choices include: 2 KV heads for GQA models; a residual key dimension d R h = 32 for MLA models; and ranks R K = R V = 2 and R Q = 6 for TPA and TPA-KVonly models, unless otherwise specified. Other relevant hyper-parameters are listed in Table 10.

Training Setup Details. We follow the nanoGPT training configuration [24]. In particular, we use the AdamW [35] optimizer with ( β 1 , β 2 ) = (0 . 9 , 0 . 95) , a weight decay of 0 . 1 , and gradient clipping at 1 . 0 . We follow the same setting as nanoGPT that the learning rate is managed by a cosine annealing scheduler [36] with 2 , 000 warmup steps and a (total) global batch size of 480 . For the small , medium , large and XL models, we set maximum learning rates of 6 × 10 -4 , 3 × 10 -4 , 2 × 10 -4 , and 1 × 10 -4 (respectively), and minimum learning rates of 3 × 10 -5 , 6 × 10 -5 , 1 × 10 -5 , and 1 × 10 -5 (respectively).

Table 9: The architecture hyper-parameters and training devices of models. Abbreviations: BS. = Batch Size, GAS. = Gradient Accumulation Steps.

| MODEL SIZE   | PARAMETERS   | DEVICES       |   MICRO BS. |   GAS. |   #LAYERS |   d MODEL |
|--------------|--------------|---------------|-------------|--------|-----------|-----------|
| SMALL        | 124M         | 4 × A100 GPUS |          24 |      5 |        12 |       768 |
| MEDIUM       | 353M         | 8 × A100 GPUS |          20 |      3 |        24 |      1024 |
| LARGE        | 772M         | 8 × A100 GPUS |          15 |      4 |        36 |      1280 |
| XL           | 1.55B        | 8 × A100 GPUS |           6 |     10 |        48 |      1600 |

## H.2 Additional Experimental Results

## H.2.1 Perplexity Curves

We display the perplexity curves for medium, large, and XL size models in Figure 9.

Table 10: The architecture hyper-parameters for different models.

| MODEL SIZE     |   SMALL |   MEDIUM |   LARGE |   XL |
|----------------|---------|----------|---------|------|
| h (MHA)        |      12 |       16 |      20 |   25 |
| h (MQA)        |      23 |       31 |      39 |   49 |
| h (GQA)        |      22 |       30 |      38 |   48 |
| h (MLA)        |      12 |       23 |      34 |   49 |
| h (TPA-KVONLY) |      22 |       29 |      37 |   47 |
| h (TPA)        |      34 |       47 |      61 |   78 |
| d c (MLA)      |     256 |      512 |     512 |  512 |
| d ′ c (MLA)    |     512 |     1024 |    1024 | 1024 |

Figure 9: The validation perplexity of medium-size (353M) models, large-size (773M), and XL-size (1.5B) models with different attention mechanisms on the FineWeb-Edu 100B dataset.

<!-- image -->

## H.2.2 Ablation Study on Different Ranks

Figure 10 illustrates the training loss, validation loss, and validation perplexity for XL-sized (1.5B parameters) TPA models with varying key/value ranks ( R K = R V = R , as indicated in the figure legend), trained on the FineWeb-Edu 100B dataset. Corresponding 0-shot evaluation results are presented in Table 12 (rows for TPA-KVonly with different R K,V ). These results indicate that increasing the ranks for key and value factorizations generally improves the performance of the TPA models.

Figure 10: The training loss, validation loss and validation perplexity curves of XL-size (1.5B) TPA models with different key/value ranks ( R K = R V = R ) on the FineWeb-Edu 100B dataset.

<!-- image -->

## H.2.3 0-shot Evaluation with lm-evaluation-harness

We present 0-shot evaluation results using the lm-evaluation-harness for small (124M parameters) and XL (1.5B parameters) models in Tables 11 and 12, respectively.

## H.2.4 2-shot Evaluation with lm-evaluation-harness

Similarly, 2-shot evaluation results are provided in Tables 13 (Small), 14 (Medium), 15 (Large), and 16 (XL).

Table 11: Evaluation results of small models (124M) with different attention mechanisms, pre-trained on FineWeb-Edu 100B dataset (0-shot with lm-evaluation-harness). The best scores in each column are bolded . Abbreviations: HellaSw. = HellaSwag, W.G. = WinoGrande.

| Method     |   ARC-E |   ARC-C |   BoolQ |   HellaSw. |   OBQA |   PIQA |   W.G. |   MMLU |   SciQ |   Avg. |
|------------|---------|---------|---------|------------|--------|--------|--------|--------|--------|--------|
| MHA        |   50.63 |   26.96 |   59.39 |      36.18 |   32   |  64.96 |  51.85 |  23.4  |   70.3 |  46.19 |
| MQA        |   49.62 |   25.34 |   55.72 |      35.94 |   31.4 |  64.85 |  51.3  |  23.37 |   68.7 |  45.14 |
| GQA        |   48.7  |   25.68 |   56.15 |      35.58 |   31.4 |  64.91 |  51.62 |  23.12 |   68.2 |  45.04 |
| MLA        |   50.21 |   26.71 |   58.01 |      36.25 |   32.8 |  64.69 |  50.59 |  24.67 |   71.9 |  46.2  |
| TPA-KVonly |   51.05 |   26.54 |   57.25 |      36.77 |   32.6 |  65.02 |  50.91 |  23.64 |   69.7 |  45.94 |
| TPA        |   51.26 |   27.39 |   57    |      36.68 |   32.8 |  64.47 |  49.72 |  24.61 |   72   |  46.21 |

Table 12: Evaluation results of XL models (1.5B) with different attention mechanisms, pre-trained on the FineWeb-Edu 100B dataset (0-shot with lm-evaluation-harness). The best scores in each column are bolded . Abbreviations: HellaSw. = HellaSwag, W.G. = WinoGrande. If not specified, TPA and TPA-KVonly models use R K = R V = 2 .

| Method                   |   ARC-E |   ARC-C |   BoolQ |   HellaSw. |   OBQA |   PIQA |   W.G. |   MMLU |   SciQ |   Avg. |
|--------------------------|---------|---------|---------|------------|--------|--------|--------|--------|--------|--------|
| MHA                      |   64.81 |   35.41 |   61.9  |      54.32 |   37.2 |  72.74 |  55.8  |  25.44 |   82.8 |  54.49 |
| MQA                      |   64.1  |   36.01 |   62.26 |      54.38 |   39   |  72.58 |  56.43 |  23.7  |   81.9 |  54.48 |
| GQA                      |   63.68 |   35.92 |   60.46 |      54.17 |   38.4 |  73.56 |  56.27 |  24.77 |   81.7 |  54.33 |
| MLA                      |   64.14 |   35.92 |   60.12 |      53.6  |   39.2 |  72.25 |  55.17 |  24.71 |   81.6 |  54.08 |
| TPA-KVonly               |   65.61 |   36.77 |   63.02 |      54.17 |   37   |  73.34 |  54.62 |  25.02 |   81.6 |  54.57 |
| TPA-KVonly ( R K,V = 4 ) |   64.52 |   37.03 |   63.27 |      54.89 |   39.8 |  72.91 |  56.51 |  24.74 |   81.6 |  55.03 |
| TPA-KVonly ( R K,V = 6 ) |   65.78 |   35.92 |   61.71 |      54.86 |   38.6 |  72.69 |  57.93 |  25.59 |   82.2 |  55.03 |
| TPA                      |   66.71 |   36.52 |   61.38 |      54.03 |   40.4 |  72.52 |  56.83 |  24.49 |   82.2 |  55.01 |

Table 13: Evaluation results of small models (124M) with different attention mechanisms, pre-trained on FineWeb-Edu 100B dataset (2-shot with lm-evaluation-harness). The best scores in each column are bolded . Abbreviations: HellaSw. = HellaSwag, W.G. = WinoGrande.

| Method     |   ARC-E |   ARC-C |   BoolQ |   HellaSw. |   OBQA |   PIQA |   W.G. |   MMLU |   SciQ |   Avg. |
|------------|---------|---------|---------|------------|--------|--------|--------|--------|--------|--------|
| MHA        |   57.66 |   28.24 |   57.28 |      36.43 |   29.6 |  64.09 |  51.14 |  26.57 |   82   |  48.11 |
| MQA        |   53.79 |   26.35 |   44.95 |      34.18 |   28.8 |  62.79 |  52.01 |  25.91 |   78.1 |  45.21 |
| GQA        |   55.01 |   25.94 |   55.72 |      35.68 |   31.8 |  65.29 |  51.93 |  25.27 |   77.8 |  47.16 |
| MLA        |   54.76 |   27.13 |   58.07 |      36.13 |   31.4 |  65.07 |  51.3  |  25.9  |   78.9 |  47.63 |
| TPA-KVonly |   54.25 |   27.9  |   57.06 |      36.36 |   31.8 |  64.31 |  53.59 |  26.18 |   79.2 |  47.85 |
| TPA        |   57.53 |   28.07 |   56.33 |      36.49 |   31.8 |  64.36 |  51.14 |  25.92 |   79.7 |  47.93 |

Table 14: Evaluation results of medium models (353M) with different attention mechanisms, pretrained on FineWeb-Edu 100B dataset (2-shot with lm-evaluation-harness, default LR 6 × 10 -4 ). The best scores in each column are bolded . Abbreviations: HellaSw. = HellaSwag, W.G. = WinoGrande.

| Method     |   ARC-E |   ARC-C |   BoolQ |   HellaSw. |   OBQA |   PIQA |   W.G. |   MMLU |   SciQ |   Avg. |
|------------|---------|---------|---------|------------|--------|--------|--------|--------|--------|--------|
| MHA        |   64.73 |   32.42 |   58.29 |      45.89 |   34.2 |  68.5  |  53.2  |  25.86 |   88   |  52.34 |
| MQA        |   64.98 |   33.62 |   55.02 |      45.81 |   34   |  69.59 |  53.43 |  24.3  |   85.2 |  51.77 |
| GQA        |   65.24 |   33.19 |   56.54 |      45.41 |   34.8 |  69.04 |  55.72 |  24.73 |   87.9 |  52.51 |
| MLA        |   64.98 |   33.62 |   53.52 |      45.94 |   33   |  68.55 |  51.85 |  25.46 |   89.1 |  51.78 |
| TPA-KVonly |   64.69 |   32.34 |   59.48 |      46.23 |   35.4 |  70.08 |  54.06 |  25.64 |   86.3 |  52.69 |
| TPA        |   67.97 |   34.56 |   57.22 |      46.87 |   34.6 |  69.91 |  52.01 |  25.07 |   89.9 |  53.12 |

Table 15: Evaluation results of large models (772M) with different attention mechanisms, pre-trained on the FineWeb-Edu 100B dataset (2-shot with lm-evaluation-harness). The best scores in each column are bolded . Abbreviations: HellaSw. = HellaSwag, W.G. = WinoGrande.

| Method     |   ARC-E |   ARC-C |   BoolQ |   HellaSw. |   OBQA |   PIQA |   W.G. |   MMLU |   SciQ |   Avg. |
|------------|---------|---------|---------|------------|--------|--------|--------|--------|--------|--------|
| MHA        |   67.85 |   36.35 |   59.82 |      50.22 |   35   |  70.67 |  53.35 |  23.92 |   91.1 |  54.25 |
| MQA        |   68.86 |   36.09 |   53.79 |      50.5  |   37   |  70.89 |  54.7  |  25.01 |   88   |  53.87 |
| GQA        |   69.15 |   36.09 |   58.84 |      50.29 |   36.2 |  70.73 |  54.22 |  26.08 |   90   |  54.62 |
| MLA        |   70.54 |   38.74 |   61.5  |      51.86 |   36   |  70.89 |  54.22 |  25.47 |   92.4 |  55.74 |
| TPA-KVonly |   71.34 |   37.71 |   59.76 |      51.1  |   36   |  71.49 |  54.62 |  25.83 |   90.1 |  55.33 |
| TPA        |   70.41 |   37.71 |   60.06 |      51.3  |   34   |  71.06 |  54.54 |  25.79 |   90.3 |  55.02 |

Table 16: Evaluation results of XL models (1.5B) with different attention mechanisms, pre-trained on the FineWeb-Edu 100B dataset (2-shot with lm-evaluation-harness). The best scores in each column are bolded . Abbreviations: HellaSw. = HellaSwag, W.G. = WinoGrande. If not specified, R K = R V = 2 for TPA and TPA-KVonly models.

| Method                   |   ARC-E |   ARC-C |   BoolQ |   HellaSw. |   OBQA |   PIQA |   W.G. |   MMLU |   SciQ |   Avg. |
|--------------------------|---------|---------|---------|------------|--------|--------|--------|--------|--------|--------|
| MHA                      |   70.83 |   39.93 |   59.85 |      54.05 |   36.2 |  72.52 |  55.17 |  25.42 |   91.7 |  56.18 |
| MQA                      |   71.34 |   39.76 |   58.93 |      54.27 |   39.4 |  72.96 |  57.38 |  24.74 |   91.9 |  56.74 |
| GQA                      |   71.17 |   39.08 |   60.18 |      54.05 |   37.4 |  73.07 |  56.35 |  24.87 |   92.2 |  56.49 |
| MLA                      |   70.79 |   37.54 |   50.83 |      53.33 |   40   |  72.09 |  56.51 |  24.93 |   91.8 |  55.31 |
| TPA-KVonly               |   72.85 |   39.68 |   60.92 |      53.81 |   37   |  73.34 |  56.83 |  26.19 |   91.3 |  56.88 |
| TPA-KVonly ( R K,V = 4 ) |   72.98 |   40.27 |   60.15 |      54.88 |   36.8 |  73.29 |  56.43 |  25.5  |   92.1 |  56.93 |
| TPA-KVonly ( R K,V = 6 ) |   73.95 |   39.76 |   58.99 |      54.73 |   36.8 |  72.91 |  59.04 |  24.93 |   92.9 |  57.11 |
| TPA                      |   71.76 |   39.16 |   61.25 |      53.74 |   37.8 |  72.8  |  55.49 |  23.86 |   90.7 |  56.28 |

## H.3 Ablation Studies on Learning Rates

To assess sensitivity to learning rates, we conducted parallel experiments on medium-sized models using a learning rate of 3 × 10 -4 (compared to the default 6 × 10 -4 used for other medium model results). The training loss, validation loss, and validation perplexity curves are shown in Figure 11. Performance on standard benchmarks for these models trained with the 3 × 10 -4 learning rate are reported in Tables 17 (0-shot) and 18 (2-shot). The results demonstrate that TPA and TPA-KVonly maintain their performance advantages over other attention mechanisms even with this alternative learning rate.

Figure 11: The training loss, validation loss, and validation perplexity of medium-size (353M) models (learning rate 3 × 10 -4 ) with different attention mechanisms on the FineWeb-Edu 100B dataset.

<!-- image -->

Table 17: The evaluation results of medium models (learning rate 3 × 10 -4 ) with different attention mechanisms pretrained using the FineWeb-Edu 100B dataset (0-shot with lm-evaluation-harness). The best scores in each column are bolded . Abbreviations: HellaSw. = HellaSwag, W.G. = WinoGrande.

| Method     |   ARC-E |   ARC-C |   BoolQ |   HellaSw. |   OBQA |   PIQA |   W.G. |   MMLU |   SciQ |   Avg. |
|------------|---------|---------|---------|------------|--------|--------|--------|--------|--------|--------|
| MHA        |   56.52 |   29.27 |   58.84 |      44.06 |   35   |  68.44 |  51.07 |  25.35 |   76.4 |  49.44 |
| MQA        |   55.68 |   28.24 |   60.86 |      44.17 |   35.2 |  68.66 |  52.72 |  25.14 |   72.9 |  49.29 |
| GQA        |   54.88 |   29.61 |   56.36 |      43.77 |   35.2 |  68.82 |  52.57 |  25.41 |   74.8 |  49.05 |
| MLA        |   59.64 |   29.78 |   60.73 |      45.17 |   34.2 |  68.66 |  52.8  |  25.34 |   75.7 |  50.22 |
| TPA-KVonly |   57.11 |   30.03 |   61.25 |      44.83 |   34.6 |  69.04 |  54.54 |  23.35 |   74.6 |  49.93 |
| TPA        |   59.3  |   31.91 |   60.98 |      45.57 |   34.6 |  69.48 |  53.91 |  24.93 |   77.2 |  50.88 |

## I Broader Impacts and Limitations

This work allows for the processing of much longer sequences of information with limited hardware resources by reducing the KV cache size. This could make advanced AI capabilities accessible to entities with limited computational budgets, potentially fostering improvement on downstream tasks, including in-depth document analysis, complicated-context reasoning, and code generation, promoting innovation across various sectors in fields of scientific research, education, and software development.

Table 18: The evaluation results of medium models (learning rate 3 × 10 -4 ) with different attention mechanisms pre-trained using the FineWeb-Edu 100B dataset (2-shot with lm-evaluation-harness). The best scores in each column are bolded . Abbreviations: HellaSw. = HellaSwag, W.G. = WinoGrande.

| Method     |   ARC-E |   ARC-C |   BoolQ |   HellaSw. |   OBQA |   PIQA |   W.G. |   MMLU |   SciQ |   Avg. |
|------------|---------|---------|---------|------------|--------|--------|--------|--------|--------|--------|
| MHA        |   64.44 |   32.85 |   59.05 |      44.18 |   33.2 |  68.72 |  50.12 |  26.01 |   87.4 |  51.77 |
| MQA        |   64.27 |   32.94 |   57.71 |      44.36 |   31.8 |  68.01 |  51.7  |  25.99 |   86   |  51.42 |
| GQA        |   61.7  |   32.17 |   52.81 |      43.99 |   33.8 |  68.5  |  53.35 |  24.44 |   86.4 |  50.8  |
| MLA        |   65.95 |   31.48 |   50.98 |      44.99 |   32.2 |  68.93 |  51.93 |  25.89 |   88.8 |  51.24 |
| TPA-KVonly |   65.99 |   33.7  |   57.49 |      44.47 |   34.2 |  69.53 |  53.28 |  24.23 |   86.5 |  52.15 |
| TPA        |   66.54 |   34.47 |   58.96 |      45.35 |   33   |  69.21 |  53.99 |  24.51 |   91.3 |  53.04 |

Although our work proposes a KV-cache efficient architecture for large language models, it may contain certain limitations. For instance, generalization to other modalities deserves more extensive investigation.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We describe all the contributions and scope in the abstract and introduction parts.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discussed the limitations in Appendix I.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be

used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: We list all the assumptions and proofs in Appendix D.

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

Justification: We listed all the experiment details in Section 6 for reproduction of our work.

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

Justification: The code and data required to reproduce the main experimental results are provided at https://anonymous.4open.science/r/T6-anonymous-2025 . The supplemental material will contain instructions for their use.

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

Justification: We just list all the training and test details in Section 6.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: The error bars are not reported because it would be too computationally expensive for repeated experiments on LLMs.

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

Justification: We just list all the computer resources in Section 6.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our research only explores a novel framework for large language models with better KV-Cache efficiency. Therefore, the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discussed the potential positive societal impacts and negative societal impacts in Appendix I.

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

Justification: Our work proposes a novel framework of large language models. To our knowledge, this work has no direct path to any negative applications.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We add the citation to all the codes (nanoGPT and lm-evaluation-harness: MIT License) and datasets (FineWeb-Edu-100B: odc-by) that we used in this work. No other models are included in our work.

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

Justification: The code implementing our proposed TPA model and experimental setup is released at https://anonymous.4open.science/r/T6-anonymous-2025 . This code will be documented to facilitate understanding and use by other researchers. No new datasets or pre-trained models are introduced beyond the code for the methods.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper only use open-source codes and datasets which do not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper only use open-source codes and datasets which do not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: This work aims at exploring more efficient architecture for large language models. Therefore, LLM architectures are well described in the main part of this paper.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.