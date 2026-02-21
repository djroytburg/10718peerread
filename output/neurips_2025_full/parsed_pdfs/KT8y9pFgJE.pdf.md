## Fixed-Point RNNs: Interpolating from Diagonal to Dense

Sajad Movahedi ∗ 1 , 2 , Felix Sarnthein ∗ 1 , 2 , Nicola Muça Cirone 3 , Antonio Orvieto 1 , 2

1 ELLIS Institute Tuebingen, 2 Max Planck Institute for Intelligent Systems, 3 Department of Mathematics, Imperial College London {sajad.movahedi, felix.sarnthein}@tue.ellis.eu

## Abstract

Linear recurrent neural networks (RNNs) and state-space models (SSMs) such as Mamba have become promising alternatives to softmax-attention as sequence mixing layers in Transformer architectures. Current models, however, do not exhibit the full state-tracking expressivity of RNNs because they rely on channel-wise (i.e. diagonal) sequence mixing. In this paper, we investigate parameterizations of a large class of dense linear RNNs as fixed-points of parallelizable diagonal linear RNNs. The resulting models can naturally trade expressivity for efficiency at a fixed number of parameters and achieve state-of-the-art results on the state-tracking benchmarks A 5 and S 5 , while matching performance on copying and other tasks.

## 1 Introduction

State-space models (SSMs) and other new efficient recurrent token mixers are becoming a popular alternative to softmax attention in language modeling (Gu &amp; Dao, 2024) as well as in other applications such as vision (Liu et al., 2024) and DNA processing (Nguyen et al., 2024). Inspired by linear input-controlled filtering, these models can be expressed as carefully parametrized linear recurrent neural networks (RNNs) with input-dependent, diagonal state transition:

<!-- formula-not-decoded -->

Compared to classical RNNs such as LSTMs (Hochreiter &amp; Schmidhuber, 1997), in Eq. (1) the relation between the previous hidden state h t -1 and the current h t is linear and its coefficient a t does not depend on the hidden states. These choices allow SSMs such as Mamba (Gu &amp;Dao, 2024) to be computed through efficient parallel methods during training. Furthermore, they are easier to optimize than classical RNNs,

Figure 1: Sequence length generalization at training length 16 (pink) for state-tracking on A 5 , with Transformer (brown) and LSTM (purple) as lower/upper bounds. Our Fixed-Point RNN (FPMamba-H) is trained at different maximum number of fixed-point iterations ℓ max: between 2 (green) and 16 (blue). Increasing the number of fixedpoint iterations allows the linear RNN to interpolate from diagonal to dense in a few iterations.

<!-- image -->

thanks to stable and efficient reparametrizations available for diagonal transitions (Orvieto et al., 2023; Zucchet &amp; Orvieto, 2024) - techniques that are significantly more difficult to apply effectively in the classical setting (Arjovsky et al., 2016; Helfrich et al., 2018). At test time, they are faster than classical Transformers on long sequences due to their recurrent nature.

∗ Equal contribution.

Though modern linear RNNs have shown promise in practice, recent theoretical studies suggest that using dense, input-dependent transition matrices (i.e. replacing diag ( a t ) with a dense A t ) could present an opportunity to improve expressivity and unlock performance on challenging tasks. In particular, Cirone et al. (2024b) prove that dense selective SSMs are endowed with the theoretical expressivity of classical non-linear RNNs such as LSTMs. As shown by Merrill et al. (2024) and Sarrof et al. (2024), such gained expressivity proves to be particularly useful in state-tracking applications where models are expected to maintain and extrapolate a complex state of the world. Since state-tracking is naturally expressed by non-linear RNNs but provably unavailable to channelwise sequence mixers such as SSMs or Transformers, Merrill &amp; Sabharwal (2023) speculate on a fundamental tradeoff between parallelism and expressivity . This discussion sparked interest in non-diagonal recurrences and parallelizable architectures capable of state-tracking (Grazzi et al., 2024; Terzic et al., 2025; Schöne et al., 2025; Peng et al., 2025; Siems et al., 2025).

When designing new architectures involving dense selective yet linear state transitions of the form h t = A t h t -1 + B t x t , two fundamental concerns arise:

1. What should the parametric form for A t , as a function of the input be? How can we guarantee this parametrization induces a stable recurrence, like in standard 2 SSMs?
2. How does a parametrization balance between expressivity and parallelism? Which assumptions on the structure of A t enable efficient computation, and how do they interact with expressivity?

Perhaps the first approach tackling the above questions was DeltaNet (Schlag et al., 2021a; Yang et al., 2024b) with a block-diagonal and orthogonal therefore, stable state transition structure, where each block is parametrized by a Householder matrix. The parallelizable algorithm, was then extended to include negative eigenvalues (Grazzi et al., 2024), gates (Yang et al., 2025), and most recently products of Householders (Siems et al., 2025). Such choices, leading to increased expressivity as exemplified by their state-tracking and length generalization capabilities, are motivated mainly by hardware considerations: Householder-based mixing can be implemented efficiently on GPUs as linear attention via WY-representations and the UT transform (Yang et al., 2024b).

While the works above offer exciting practical strategies for boosting capabilities at a relatively low additional computational cost, they fall short in exploring the sea of intriguing options for dense transitions and hence, in thoroughly answering questions (1) and (2) above.

Unfortunately, this is not an easy task: although linear recurrences are theoretically parallelizable across sequence length (Martin &amp; Cundy, 2018), parallelizing dense RNNs efficiently is not trivial due to increased memory I/O. These thoughts inspired us to change our viewpoint: instead of designing an algorithm which adds a fraction of non-diagonal processing to a model, here, we look for a strategy to navigate the parallelism tradeoff towards a truly dense object.

Motivated by the idea of designing a parallelizable general-purpose method to implement new

Figure 2: (a) State-tracking on A 5 at sequence length 16 , and (b) character accuracy of copying at 2 × sequence length generalization, trained on lengths ∈ [5 , 50] . Our single layer FP-Mamba-H with mixer reflections r ∈ { 1 , 2 , 4 } is compared to baselines of increasing depth ∈ { 1 , 2 , 4 , 6 , 8 } . FPMamba-H is the only model capable of solving both the state-tracking and the copy task.

<!-- image -->

dense RNN variations, in this paper we devise a new adaptive computation strategy which allows to interpolate between fast recurrent diagonal RNNs and dense recurrences with arbitrary preselected structure. Instead of parametrizing the dense RNN layer as an explicit function h = F θ ( x ) , we build on the literature of equilibrium/implicit models (Bai et al., 2019; Ghaoui et al., 2021) to parametrize it implicitly as a solution h ∗ to a fixed-point equation h = f θ ( x , h ) involving only a diagonal RNN. As described in Fig. 3a, we solve for h ∗ using a fixed-point iteration of diagonal RNN evaluations f θ .

2 Standard SSMs are diagonal and operate in polar coordinates, parametrizing directly the gap between eigenvalues and the stability threshold (Orvieto et al., 2023). This technique allows to increasing granularity near the identity, and to effectively normalize the forward pass (cf. √ 1 -| γ | 2 term in Griffin (De et al., 2024)).

A fundamental question some readers might rightfully ask, is the following: ' what is the advantage of iterating a single layer in depth compared to depth-stacking multiple SSM, e.g. Mamba layers? ' We claim one advantage comes from having access to the limiting dense object. As showcased by Fig. 2, this allows to adaptively provide the required expressivity for a fixed set of parameters without any a priori choice on the network size.

Summary. In this work, we propose a recipe to design a general class of dense linear RNNs as fixed points of corresponding diagonal linear RNNs. Our contributions are:

1. We develop the framework of Fixed-Point RNNs to adaptively trade parallelism for expressivity using the number of fixed-point iterations (Fig. 1).
2. We achieve a stable parametrization of a dense RNN via a carefully designed diagonal RNN.
3. The framework allows for easy integration of both non-linear hidden state dependence and linear attention based matrix-valued formulations. This way, our FP-Mamba unites previously isolated capabilities of recurrent computation and memory (Fig. 2).

## 2 Background

Since their introduction (Rumelhart et al., 1986; Elman, 1990), RNNs have significantly contributed to the evolution of machine learning methods for sequential data (Hochreiter &amp; Schmidhuber, 1997; Jaeger, 2001). But despite their theoretical promise of Turing-completeness (Siegelmann &amp; Sontag, 1992), recurrent models fell out of fashion due to two significant challenges: they are inherently sequential, and notoriously difficult to train (Hochreiter et al., 2001; Pascanu et al., 2013). The recent advancements of linear RNNs (Gu &amp; Dao, 2024) suggest a way forward to combine the scalability of Transformers (Vaswani et al., 2017) with the expressivity of classical RNNs (Cirone et al., 2024b). The key challenge here is the stable and efficient parametrization of a linear RNN layer with a time-varying recurrent transition matrix. In this paper we are exploring first steps towards this goal.

Dense Selective RNN. Traditionally, RNNs are parametrized as either time-invariant, non-linear, or element-wise system. To the best of our knowledge, a time-variant, dense, and linear RNN parametrization has been of mild interest at best. To understand why, consider the general form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where A t ∈ R d × d corresponds to the time-varying state transition matrix, B t ∈ R d × d is the input transformation matrix, h t ∈ R d denotes the hidden state, and x t ∈ R d is the input for t &lt; T steps. For a given sequence of A t , the complexity of a forward pass is O ( Td 2 ) in memory and O ( T ) sequential steps. Although such a linear RNN could also be computed in O (log T ) sequential steps using a parallel scan algorithm (Martin &amp; Cundy, 2018), this would require materializing matrixmatrix multiplications at cost O ( d 3 ) . An issue in both scenarios, however, is the parametrization of A t as time-varying, i.e. input- or even hidden state-dependent matrices. In general, this requires a map M : d ↦→ d 2 with potentially d × d 2 parameters, and O ( Td 3 ) time complexity. While structured dense matrix representations for A t could potentially present a remedy, they come with additional challenges: (1) In order to guarantee expressivity, the A t cannot be co-diagonalizable such as for example Toeplitz matrices (Cirone et al., 2024b). (2) In order to guarantee stability of the dynamical system, the spectral radius ρ ( A t ) needs to be less than, but still close to 1 for long-range interactions (Orvieto et al., 2023). (3) The matrix structure needs to be closed under multiplications to enable parallel scans without having to materialize dense representations at O ( Td 2 ) memory cost.

Related Works. Improving the trainability of classical non-linear RNNs has a long history. For example, Arjovsky et al. (2016) and Helfrich et al. (2018) investigate parameterizations to stabilize their spectral radius with structured matrix representations, while Lim et al. (2024) and Gonzalez et al. (2024) propose iterative methods to parallelize their computation. In this work, however, we focus on stabilizing and parallelizing a time-variant, dense, linear RNN. Improving the limited expressivity of existing diagonal linear RNNs is the focus of a few recent works, e.g. by Grazzi et al. (2024) and Siems et al. (2025). In contrast, we investigate a wide class of structured parameterizations for dense RNNs where the additional cost is adaptively chosen depending on the task. In concurrent work, Schöne et al. (2025) propose an iterative method similar to ours, but as opposed to our carefully designed implicit dense RNN layer, they focus on scaling implicit causal models of existing multi-layer architectures on language. For a more extensive literature review, we refer the reader to App. A.

Figure 3: (a) An overview of the proposed Fixed-Point RNN framework in Sec. 3. A diagonal RNN f θ consisting of a sequence mixer Λ t and a channel mixer Q t is iterated until convergence towards the hidden states of an implicitly dense RNN F θ . (b) FP-RNN variants with channel mixer introduced in Sec. 3.3 and 3.4 solve the state-tracking task A 5 up to various sequence lengths. (c) FP-RNNs adapt their computation time to the difficulty of the task by varying the number of fixed-point iterations ℓ ∗ .

<!-- image -->

## 3 Fixed-Points as an RNN Layer

In this section, we introduce an implicit parameterization for a family of dense RNNs F θ ( x ) which describes its output by a solution h ∗ ∈ R T × d to the fixed-point equation h = f θ ( x , h ) (Sec. 3.1). Then, we discuss how to find the solution h ∗ using fixed-point iterations (Sec. 3.2) and the algorithmic implications (Sec. 3.4) of the FP-RNN framework in light of the challenges outlined in Sec. 2. Finally, we briefly touch on how to train an implicitly dense model F θ ( x ) with gradient descent (Sec. 3.5).

## 3.1 From Explicit to Implicit Parameterization

We start by designing a diagonal RNN f θ ( x , h ) such that the solution h ∗ to its fixed-point equation h = f θ ( x , h ) implicitly represents a dense RNN h ∗ = F θ ( x ) . Consider the factorized parametrization of A t similar to the one introduced by Helfrich et al. (2018) for non-linear and time-invariant RNN:

<!-- formula-not-decoded -->

Separating A t into a diagonal matrix Λ t ∈ R d × d and a non-diagonal invertible mixing matrix Q t ∈ R d × d allows to describe h ∗ by only a diagonal transition Λ t by reformulating Eq. 3 to

<!-- formula-not-decoded -->

This means that the states h ∗ = F θ ( x ) of the dense linear RNN can be implicitly described by the fixed-point h ∗ = f θ ( x , h ∗ ) of a corresponding diagonal linear RNN of the following form:

<!-- formula-not-decoded -->

In other words, if we could find the fixed-point h ∗ = f θ ( x , h ∗ ) ∈ R T × d for the diagonal RNN defined in Eq. 5, then h ∗ would describe the states of a corresponding dense RNN h ∗ = F θ ( x ) . Motivated by this insight, in Sec. 3.2 we carefully parametrize the diagonal RNN f θ ( x , h ) and its channel mixer Q t such that a computable fixed-point exists.

## 3.2 The Fixed-Point Iteration

Solving fixed-point equations such as h = f θ ( x , h ) , is perhaps one of the most well-studied problems in mathematics (Granas et al., 2003). In the context of deep learning, the literature on Neural ODEs (Chen et al., 2018) and Deep Equilibrium Models (Bai et al., 2019; Ghaoui et al., 2021) investigates fixed-point methods for implicit parametrizations of neural networks. A straightforward, yet effective method computes the forward pass by simply rolling out the fixed-point iteration. In the context of solving h ∗ = f θ ( x , h ∗ ) , this corresponds to introducing an iteration in depth h ℓ = f θ ( x , h ℓ -1 ) . Denoting ℓ as the current iteration in depth (i.e., over the layer dimension), and t as the current iteration in time (i.e., over the sequence dimension), the iteration starts at h 0 t = 0 and proceeds with

<!-- formula-not-decoded -->

Intuitively, this iteration mixes information with interleaved channel mixing (with Q t ) and sequence mixing (with Λ t ) until convergence towards the hidden states of an implicit dense RNN F θ (cf. 3a).

The difficulty with such an iteration in depth and time is that the recurrent dynamics could explode without proper stabilization. While the recurrence in time can be stabilized with RNN techniques (Zucchet &amp; Orvieto, 2024) such as an input gate I -Λ t , the recurrence in depth, however, could still diverge if f θ ( x , h ) does not have an attracting fixed-point (Granas et al., 2003). In order to design a diagonal linear RNN f θ ( x , h ) which is guaranteed to have an attracting fixed-point, we make use of Banach (1922)'s theorem. In our context, the theorem states that f θ ( x , h ) converges to a fixed-point from any initialization h 0 if it has a Lipschitz constant &lt; 1 in h . For a fixed-point RNNs with input gate I -Λ , we present the following theorem:

Theorem 3.1. Let f θ ( x , h ) be the diagonal linear RNN with input-independent Λ and Q

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Intuitively, Thm. 3.1 states two conditions for stable parametrization of an implicitly dense RNN F θ : (1) the recurrence in time needs to be coupled with input normalization and contractive (i.e. ∥ Λ ∥ 2 &lt; 1 ). (2) The recurrence in depth acting on h , i.e. ( I -Q t ) , needs to be contractive. Together, this guarantees that all sequences h ℓ up to h ∗ throughout the fixed-point iteration do not explode without any explicit assumptions on the spectral radius on A (Arjovsky et al., 2016).

## 3.3 Parametrization of Q t and Λ t

To satisfy the assumptions required for expressivity in (Cirone et al., 2024b), the implicit transition matrix A t and therefore Λ t and Q t need to be input-controlled (i.e. selective), which could be realized through a linear mapping of the input, i.e. Q t = M ( x t ) := reshape ( W Q x t ) . However, this presents two challenges: how can stability be guaranteed (c.f. Thm. 3.1) and excessive computational cost due to the O ( d 3 ) parameters of W Q be avoided? A straight-forward solution lies in structured matrix representations for both the diagonal transition matrix Λ t and the channel mixer Q t .

Inspired by Helfrich et al. (2018), we aim for Q t to be approximately norm-preserving and Λ t to control the eigenvalue scale using a parametrization akin to Mamba or Griffin (Gu &amp; Dao, 2024; De et al., 2024) and normalization ( I -Λ t ) . For the channel mixers Q t , we consider the structures:

- Diagonal Plus Low Rank (DPLR): Q t = M ( x t ) := ( I -∑ r i =1 α it · ¯ u it ¯ u ⊤ it ) , for rank r .
- Householder Reflections (H): Q t = M ( x t ) := ∏ r i =1 ( I -α it · ¯ u it ¯ u ⊤ it ) , for r reflections.
- Kronecker (K): Q t = M ( x t ) := I -( ¯ K 1 t ⊗ ¯ K 2 t ) , where ⊗ denotes the Kronecker product.

This allows to reduce the size of the input-dependent parameters α it , ¯ u it , and ¯ K i t to O ( d ) , and consequently reduce the size of the linear map W Q to O ( d 2 r ) and O ( d 2 ) . In order to guarantee stability, the condition || I -Q || 2 &lt; 1 can be enforced by scaling α it , ¯ u it , and ¯ K i t appropriately. For more details about the channel mixer variants, please refer to App. C. Fig. 3b, we compare different channel mixer variants and observe that the Kronecker structure seems to be most appropriate the state-tracking task A 5 .

## 3.4 Algorithmic Implications

Recall from Sec. 2 that an explicitly parametrized dense selective RNN can only be parallelized under strict assumptions on its structure and runs otherwise in O ( T ) sequential steps. However, a parallelizable structure is given by the element-wise, diagonal transition Λ t of a diagonal RNN (Martin &amp; Cundy, 2018). Since such a diagonal RNN is called ℓ ∗ -times as a subroutine of the fixed-point iteration in Eq. 6, a fixed-point RNN runs in O ( ℓ ∗ · log T ) sequential steps. This means that the implicit parametrization -as opposed to explicit or non-linear parametrizations- allows to decouple the number of sequential steps ℓ ∗ from the sequence length T itself, and trade parallelism for expressivity.

This insight suggests an opportunity to introduce a non-linear computation for every sequential step, like in classical RNNs. Concretely, we investigate channel mixers M ( x t + h ℓ -1 t -1 ) which are a function of both the input x t and the hidden state h ℓ -1 t -1 from the previous iteration (in both time and depth) without degrading parallelizability. In Fig. 3b, we compare channel mixers with and without hidden state dependence and observe that this indeed improves sequence length generalization.

Summarizing the results so far, we arrive at an updated recurrence with hidden state dependence:

<!-- formula-not-decoded -->

where we use ⊙ to highlight the parallelizability of the element-wise product. We would like to note that due to the normalization ( I -Λ t ) , the corresponding dense RNN F θ is not explicitly representable anymore as discussed in App. B.2. Furthermore, for the time-varying parametrization in Eq. 8, the convergence guarantees may be weaker and solutions h ∗ could be non-unique due to the hidden state dependence. In practice, we iterate until || h ℓ -h ℓ -1 || ∞ || h ℓ || ∞ &lt; 0 . 1 and observe that the conditions of Thm. 3.1 are strong enough to reach convergence within a finite number of iterations ℓ ∗ as evidenced by Fig. 3c. Interestingly, the model navigates the parallelism tradeoff (Merrill &amp; Sabharwal, 2023) and adaptively increases its sequential computation for harder tasks.

## 3.5 Optimizing Fixed-Point RNNs

One advantage of converging to a fixed-point as opposed to general layer looping lies in model training. Since the gradient with respect to h 0 is not needed, implicit differentiation can be used to avoid storing and backpropagating through the computational graph of the fixed-point iteration, as discussed by Liao et al. (2018), Bai et al. (2019), and in App. B.3. In practice, truncated backpropagation of the last k iterations suffices to approximate the gradient through the full iteration J ∗ x ≈ J x ( h ℓ ∗ -k ) · . . . · J x ( h ℓ ∗ ) . For Fixed-Point RNNs we observe that computing the gradient only at the fixed-point ( k = 0 ), is enough to stabilize training. This means that compared to a single diagonal RNN layer, Fixed-Point RNNs incur no memory overhead and only sequential overhead in the forward pass but not in the backward pass.

We hypothesize that this is possible because f θ ( x , h ) is a mostly linear object as opposed to multilayer implicit models such as (Schöne et al., 2025). Furthermore, we observe that hidden state dependence M ( x t + h ℓ -1 t -1 ) particularly helps with gradient-based optimization. We credit this to the symmetry between the gradients w.r.t. x and h , and formalize this in the following theorem:

Theorem 3.2. Let f θ ( x , h ) have Lipschitz constant &lt; 1 and fixed-point h ∗ . If the Jacobians ∂f θ ∂ x ( x , h ) and ∂f θ ∂ h ( x , h ) are equal, then the gradient ∇ θ L ( f θ ( x , h ) , y ) of the loss L ( · , y ) for a target y at the fixed point h = h ∗ is a descent direction of L ( F θ ( x ) , y ) . Proof in App. B.4.

## 4 Fixed-Point Mamba

In the previous section we introduced the FP-RNN framework on a small RNN with vector hidden state. Now, we extend it to modern matrix state RNNs in Sec. 4.1 and parametrize a dense variant of Mamba (Gu &amp; Dao, 2024) in Sec. 4.2. A detailed description of the architecture is available in App. C.2. We compare the architecture to the baselines Mamba (Gu &amp; Dao, 2024), Mamba-2 (Dao &amp; Gu, 2024), Gated DeltaNet (Yang et al., 2025), and LSTM (Hochreiter &amp; Schmidhuber, 1997) on the copy task introduced by Jelassi et al. (2024) in Sec. 4.3 and state-tracking introduced by Merrill &amp; Sabharwal (2023) in Sec. 4.4. In order to keep the number of layers at the same order of magnitude, we use two layers for the diagonal linear RNN baselines and one layer for FP-Mamba and LSTM. Finally, we discuss the required number of fixed-point iterations in the context of state-tracking and language modeling in Sec. 4.5.

Figure 4: Length generalization on A 5 (a, c) and S 5 (b, d) beyond the train sequence length 16 (pink line). We compare a 1-layer FP-Mamba with mixer variants Q t to baselines with 2 layers.

<!-- image -->

## 4.1 Introducing Matrix States

Memory capacity is an important consideration in RNNs. In preliminary experiments, we notice a clear gap between the performance of a Fixed-Point RNNs and Mamba in terms of copying ability. We attribute this difference in performance to Mamba's state-expansion which endows it with matrix hidden states similar to linear attention, DeltaNet, or mLSTM (Katharopoulos et al., 2020; Schlag et al., 2021a; Beck et al., 2024). In simple terms, these models use an outer product of an inputdependent vector b t ∈ R d state (i.e. the key) and the input vector x t ∈ R d inner (i.e. the value) as an input to a matrix-valued recurrence with hidden state and transition gate H t , λ t ∈ R d state × d inner . The hidden state is then contracted with another input-dependent vector c t ∈ R d state (i.e. the query) to get the output y ⊤ t = c ⊤ t H t ∈ R d inner :

<!-- formula-not-decoded -->

This matrix-valued recurrence introduces some challenges to our fixed-point framework. Specifically, in order to mix all the channels over the entirety of the state elements, the mixer has to be a fourth-order tensor Q t ∈ R d state × d inner × d state × d inner in

<!-- formula-not-decoded -->

where · denotes the tensor contraction einsum( klij, ij → kl ) with fourth-order identity tensor I of the same shape as Q t . Certainly, computing the fixed-point introduced in Eq. 10 is very challenging both in terms of computation and memory. As we will confirm in Sec. 4.2, one solution is to pass the contracted output y t between fixed-point iterations

<!-- formula-not-decoded -->

This implicitly factorizes the tensor mixer Q t into separately mixing along dimension d inner which is used for better expressivity, and dimension d state which is used for better memory capacity.

## 4.2 FP-Mamba Iteration

Let us apply the the fixed-point RNN framework to the Mamba parametrization. We represent the hidden state as H ℓ t , where t is the token index (i.e., indexing over the sequence dimension), and ℓ is the fixed-point iteration index (i.e., indexing over the depth dimension). The same notation is used for other variables to emphasize when they depend on the input and hidden state of the current iteration. We propose the following iteration to adapt Mamba with notation from App. C.1 to the fixed-point mechanism for matrix state RNNs in Eq. 11:

<!-- formula-not-decoded -->

L2-normalizing ¯ b ℓ t and ¯ c ℓ t allows to limit the Lipschitz constant according to Theorem 3.1. Furthermore, we replace the normalization term ( 1 -λ t ) with Mamba's normalization term ∆ t . Expanding y ℓ -1 t yields the recurrence on the matrix state

<!-- formula-not-decoded -->

where the last term nicely illustrates the two components which mix the channels of the hidden states: the low-rank matrix ¯ b ℓ t ( ¯ c ℓ -1 t ) ⊤ mixes over the dimension d state, while ( I -Q ℓ t ) ⊤ mixes over the dimension d inner. This factorization significantly simplifies the fourth-order tensor mixer formulation introduced in Eq. 10, remains expressive as discussed in App. F, and performs well in practice.

Finally, Eq. 12 can be computed as Mamba with an adjusted input ˜ x ℓ t = Q ℓ t ( x t -y ℓ -1 t ) + y ℓ -1 t

,

<!-- formula-not-decoded -->

In other words, one fixed-point step consists of a channel mixing using Q t , followed by a sequence mixing using Mamba. This separation of concerns allows to speed up the parallel recurrence in time using the Mamba implementation. To find a fixed-point, the two phases are repeated until ∥ y ℓ -y ℓ -1 ∥ ∞ ∥ y ℓ ∥ ∞ &lt; 0 . 1 is satisfied. After these ℓ ∗ iterations, required for the model to converge to a fixed-point, H ∗ t and y ∗ t present the hidden state and output of the dense matrix-valued RNN F θ . Similar to Mamba, we apply a gated linear unit g t ∈ R d inner to the output, which we observe to provide a slight improvement in performance when present within the fixed-point loop: ˜ y ℓ t = g t ⊙ y ℓ -1 t .

Table 1: Effect of shifted hidden state dependence y ℓ -1 t -1 on copying at × 2 length generalization. Each column determines which inputdependent component of the recurrence in Eq. 12 also depends on y ℓ -1 t -1 . Performance is unlocked by including a hidden dependence for b t and c t .

| Dependence on y ℓ - 1 t - 1   | Dependence on y ℓ - 1 t - 1   | Dependence on y ℓ - 1 t - 1   | Dependence on y ℓ - 1 t - 1   | Test Accuracy                      | Test Accuracy                       |
|-------------------------------|-------------------------------|-------------------------------|-------------------------------|------------------------------------|-------------------------------------|
| λ t                           | Q t                           | b t                           | c t                           |                                    |                                     |
| ✓ ✓                           | ✓ ✓                           |                               |                               | 0 . 11 0 . 53 0 . 45               | ± 0 . 00 ± 0 . 02 ± 0 . 05 ± 0 . 05 |
| ✓                             | ✓                             | ✓ ✓ ✓                         | ✓ ✓ ✓ ✓                       | 0 . 55 0 . 81 0 . 88 0 . 86 0 . 94 | ± 0 . 01 ± 0 . 01 ± 0 . 02 ± 0 . 03 |
| ✓                             | ✓                             | ✓                             |                               |                                    |                                     |

## 4.3 Shifted Hidden State Dependence y ℓ -1 t -1

In preliminary experiments, we observe that even the Fixed-Point RNN with input-dependent parameters and matrix state akin to Mamba-1 is outperformed by Mamba-2 or DeltaNet (Dao &amp; Gu, 2024; Yang et al., 2024b) on a copy task. Inspired by the short convolution in Mamba, we investigate the effect of augmenting the input-dependence of parameters λ ℓ t , b ℓ t , c ℓ t , and Q ℓ t at iteration ℓ with a shifted hidden state dependence. In practice, this means that these are linear functions of x t as well as the shifted previous iterate in depth y ℓ -1 t -1 . We refer the reader to App. C.2 for the exact formulation of the dependency.

In Tab. 1, we ablate the hidden state dependence for various combinations of λ t , b t , c t , and a Householder Q t . Observe that the dependence of b t and c t is crucial to enable the model to copy. In App. C.4, we discuss why this dependence of b t and c t could be important for copying. If additionally λ t and Q t depend on y ℓ -1 t -1 , the copy task is essentially solvable at × 2 length generalization. We therefore adopt the hidden state dependence for all components in FP-Mamba.

In Fig. 5, we evaluate length generalization on the copying task. While the best-performing baseline Gated DeltaNet is specifically designed for associative recall tasks (Yang et al., 2025), both Mamba 1 and 2 struggle with × 2 generalization. FP-Mamba closes this gap and proves the effectiveness of our proposed modifications for better memory. We would like to highlight that the number of fixed-point iterations ℓ ∗ (gray vertical line) in FP-Mamba is well below the maximum sequence length.

## 4.4 State-Tracking

In Fig. 4, we evaluate the state-tracking capabilities of FP-Mamba with Kronecker, Householder, and DPLR channel mixers of r ∈ { 1 , 2 , 4 } reflections or ranks, respectively. In particular, we compare our FP-Mamba to the baselines with regards to their length generalization beyond the training sequence length 16 . As expected, LSTM solves A 5 and S 5 , while Mamba and Mamba-2 are not able to learn it even at the training sequence length. Similar to Fig. 3b, the Kronecker structure seems to be the most suitable for the task. But FP-Mamba based on Householders also improves in terms of sequence length generalization presumably due to its improved memory. A comparison to the recent DeltaProduct (Siems et al., 2025) on training sequence length 128 is available in App. E.2.

Figure 6: Length generalization as a function of training time on A 5 . Wall clock time is plotted against the longest test sequence length with &gt; 90% accuracy for every model. While baselines of increasing depth cannot generalize beyond the training sequence length 16 (horizontal pink line), our proposed framework allows to achieve much higher generalization by scaling training time through the number of fixed-point iterations ℓ .

<!-- image -->

Figure 5: Sequence length generalization on the copy task. A 1-layer FP-Mamba-H matches a 2layer GatedDeltaNet baseline. Note that the median number of fixed-point iterations at test time ℓ ∗ (gray vertical line) is well below the longest training sequence length (pink line).

<!-- image -->

## 4.5 Required Number of Iterations ℓ ∗

A fixed-point iteration in the forward pass inevitably introduces sequential overhead to the computation of a model. While this might be acceptable for sequential generation at test time, reduced parallelism can be inhibiting at training time. In Fig. 1, we therefore evaluate FP-Mamba-H on A 5 with limited number of fixed-point iterations at training time ℓ max ∈ { 2 , 4 , 8 , 16 } . We observe that the performance decreases once ℓ max is lower than the training sequence length of 16 . In Fig. 6, we confirm that the resulting longer training times are indeed required for good length generalization. However, as opposed to baselines of increasing depth ∈ { 1 , 2 , 4 , 6 , 8 } , fixed-point iterations gain from the additional training time. Furthermore, there is room to improve efficiency, as suggested by a simple randomization scheme

Figure 7: The effective number of fixed-point iterations for each layer of a FP-Mamba-H throughout language pretraining on FineWeb (Penedo et al., 2024) at context length 2048. The corresponding validation perplexities are available in App. E.1.

<!-- image -->

(gray stars) where ℓ max ∼ Γ(4 , 1) is sampled from a Gamma distribution with mean 4 for every batch. But most importantly, the effective number of fixed-point iterations depends on the difficulty of the task. Indeed, Fig. 7 shows that the model automatically adapts to using less fixed-point iterations on language pretraining at context length 2048. Similarly, on copying (Fig. 5) and and modular arithmetic (Fig. 10), we observe that the required number of fixed-point iterations ℓ ∗ is well below the sequence length T . This suggests that the model adapts to O ( T ) complexity on simpler tasks when the full state-tracking expressivity is not required.

## 5 Discussion

A fixed-point mechanism, such as the one introduced in this paper, endows a parallelizable, diagonal linear RNN

Table 2: Complexity of FP-Mamba in comparison to Mamba. The cost of channel mixing with structure Q t is denoted by C Q t .

|          | Forward                                | Backward      |
|----------|----------------------------------------|---------------|
| Mamba    | O ( T )                                | O ( T )       |
| FP-Mamba | O (( T + C Q t ) · min( ℓ ∗ , ℓ max )) | O ( T + C Q t |

with the ability to dynamically increase the sequential computation and describe a dense linear RNN in the limit. Our results show that such a paradigm can enable both strong state-tracking and memory capabilities with a constant number of parameters in a combined sequence and channel mixing layer (Fig. 2). In fact, the fixed-point iteration gradually transforms a diagonal (i.e., channel-wise) RNN into a dense (i.e., channel-mixing) RNN, thereby allowing to trade parallel computation for expressivity (Fig. 1) without incurring additional cost during backpropagation (cf. Tab. 2).

For Fixed-Point RNNs to become competitive in practice, it is important to further understand the trade-offs between parallel and sequential computation. In the worst case, as shown in Tab. 2, FPRNNs could behave like traditional, non-linear RNNs with quadratic runtime O ( T 2 ) if the sequential overhead ℓ ∗ is linear in the sequence length T . This, however, is not necessarily a disadvantage since FP-RNNs adapt ℓ ∗ to the difficulty of the task. In this paper, we focus on introducing the framework for FP-RNNs and leave the improvement of fixed-point convergence rates to future work.

Fixed-Point RNNs present an interesting opportunity to be fused into a single GPU kernel with reduced memory I/O. This is an inherent advantage from performing repeated computation on the same operands. Several open problems need to be solved to achieve that: (1) different implementations such as sequential, parallel, or chunk-wise should converge to the same fixed-points, (2) the memory footprint of the fixed-point iteration should satisfy current hardware limitations, and (3) alternative sequence or channel mixer structures could unlock higher efficiency. Future progress on these problems could enable significant speed-ups in practical implementations of Fixed-Point RNNs.

Conclusion In this paper, we presented a framework to cast a general class of dense linear RNNs as fixed-points of corresponding diagonal linear RNNs. Fixed-Point RNNs provide a mechanism to trade computation complexity for expressivity while uniting the expressivity of recurrent models with the improved memory of linear attention models. Following encouraging results on toy tasks specifically designed to assess these capabilities, we hope this paper enables more expressive sequence mixers.

## Acknowledgments and Disclosure of Funding

We would like to thank Riccardo Grazzi and Julien Siems for the helpful discussions and comments. Antonio Orvieto, Felix Sarnthein and Sajad Movahedi acknowledge the financial support of the Hector Foundation. Felix Sarnthein would also like to acknowledge the financial support from the Max Planck ETH Center for Learning Systems (CLS).

## References

- Ajroldi, N. plainlm: Language model pretraining in pytorch. https://github.com/ Niccolo-Ajroldi/plainLM , 2024.
- Arjovsky, M., Shah, A., and Bengio, Y. Unitary evolution recurrent neural networks. In Proceedings of The 33rd International Conference on Machine Learning , pp. 1120-1128, 2016. URL https: //arxiv.org/abs/1511.06464 .
- Arora, S., Eyuboglu, S., Zhang, M., Timalsina, A., Alberti, S., Zou, J., Rudra, A., and Ré, C. Simple linear attention language models balance the recall-throughput tradeoff. In Forty-first International Conference on Machine Learning, ICML 2024 , 2024. URL https://arxiv.org/abs/2402. 18668 .
- Bai, S., Kolter, J. Z., and Koltun, V . Deep equilibrium models. In Advances in Neural Information Processing Systems , pp. 688-699, 2019. URL https://arxiv.org/abs/1909.01377 .
- Bai, S., Koltun, V., and Kolter, J. Z. Stabilizing equilibrium models by jacobian regularization. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021 , pp. 554-565, 2021. URL https://arxiv.org/abs/2106.14342 .
- Banach, S. Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales. Fundamenta mathematicae , 3(1):133-181, 1922.
- Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., and Hochreiter, S. xlstm: Extended long short-term memory. In Advances in Neural Information Processing Systems 38, NeurIPS 2024 , 2024. URL https://arxiv.org/abs/2405.04517 .
- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., and Duvenaud, D. K. Neural ordinary differential equations. In Advances in Neural Information Processing Systems , 2018. URL https://arxiv. org/abs/1806.07366 .
- Chen, Y., Zeng, Q., Ji, H., and Yang, Y. Skyformer: Remodel self-attention with Gaussian kernel and Nystrom method. Advances in Neural Information Processing Systems , 2021. URL https: //arxiv.org/abs/2111.00035 .
- Choromanski, K. M., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J. Q., Mohiuddin, A., Kaiser, L., et al. Rethinking attention with performers. In International Conference on Learning Representations , 2020. URL https://arxiv.org/abs/2009.14794 .
- Cirone, N. M., Hamdan, J., and Salvi, C. Genus expansion for non-linear random matrix ensembles with applications to neural networks, 2024a. URL https://arxiv.org/abs/2407.08459 .
- Cirone, N. M., Orvieto, A., Walker, B., Salvi, C., and Lyons, T. Theoretical foundations of deep selective state-space models. In Advances in Neural Information Processing Systems , volume 37, pp. 127226-127272, 2024b. URL https://arxiv.org/abs/2402.19047 .
- Dao, T. and Gu, A. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. In Forty-first International Conference on Machine Learning, ICML 2024 , 2024. URL https://arxiv.org/abs/2405.21060 .
- De, S., Smith, S. L., Fernando, A., Botev, A., Muraru, G., Gu, A., Haroun, R., Berrada, L., Chen, Y., Srinivasan, S., Desjardins, G., Doucet, A., Budden, D., Teh, Y. W., Pascanu, R., de Freitas, N., and Gulcehre, C. Griffin: Mixing gated linear recurrences with local attention for efficient language models, 2024. URL https://arxiv.org/abs/2402.19427 .

- Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., and Kaiser, L. Universal transformers. In 7th International Conference on Learning Representations, ICLR 2019 , 2019. URL https: //arxiv.org/abs/1807.03819 .
- Elman, J. L. Finding structure in time. Cognitive science , 1990.
- Geiping, J., McLeish, S., Jain, N., Kirchenbauer, J., Singh, S., Bartoldson, B. R., Kailkhura, B., Bhatele, A., and Goldstein, T. Scaling up test-time compute with latent reasoning: A recurrent depth approach. In ES-FoMo III: 3rd Workshop on Efficient Systems for Foundation Models , 2025. URL https://arxiv.org/abs/2502.05171 .
- Ghaoui, L. E., Gu, F., Travacca, B., Askari, A., and Tsai, A. Y . Implicit deep learning. SIAM J. Math. Data Sci. , 3:930-958, 2021. URL https://arxiv.org/abs/1908.06315 .
- Giannou, A., Rajput, S., Sohn, J., Lee, K., Lee, J. D., and Papailiopoulos, D. Looped transformers as programmable computers. In International Conference on Machine Learning, ICML 2023 , pp. 11398-11442, 2023. URL https://arxiv.org/abs/2301.13196 .
- Gonzalez, X., Warrington, A., Smith, J. T., and Linderman, S. Towards scalable and stable parallelization of nonlinear RNNs. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024. URL https://arxiv.org/abs/2407.19115 .
- Granas, A., Dugundji, J., et al. Fixed point theory , volume 14. 2003.
- Graves, A. Adaptive computation time for recurrent neural networks, 2016. URL https://arxiv. org/abs/1603.08983 .
- Grazzi, R., Siems, J., Franke, J. K., Zela, A., Hutter, F., and Pontil, M. Unlocking state-tracking in linear RNNs through negative eigenvalues. In NeurIPS 2024 Workshop on Mathematics of Modern Machine Learning , 2024. URL https://arxiv.org/abs/2411.12537 .
- Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. In First Conference on Language Modeling , 2024. URL https://arxiv.org/abs/2312.00752 .
- Gu, A., Goel, K., and Ré, C. Efficiently modeling long sequences with structured state spaces. In The Tenth International Conference on Learning Representations, ICLR 2022 , 2022. URL https://arxiv.org/abs/2111.00396 .
- Hanson, J. and Raginsky, M. Universal simulation of stable dynamical systems by recurrent neural nets. In Learning for Dynamics and Control , 2020. URL https://proceedings.mlr.press/ v120/hanson20a.html .
- Helfrich, K., Willmott, D., and Ye, Q. Orthogonal recurrent neural networks with scaled Cayley transform. In Proceedings of the 35th International Conference on Machine Learning , pp. 19691978, 2018. URL https://arxiv.org/abs/1707.09520 .
- Hochreiter, S. and Schmidhuber, J. Long short-term memory. Neural computation , 1997.
- Hochreiter, S., Bengio, Y., Frasconi, P., et al. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies. A Field Guide to Dynamical Recurrent Neural Networks , 2001.
- Hopfield, J. J. Neural networks and physical systems with emergent collective computational abilities. Proceedings of the National Academy of Sciences , 79(8):2554-2558, 1982.
- Jaeger, H. The "echo state" approach to analysing and training recurrent neural networks-with an erratum note. German National Research Center for Information Technology GMD Technical Report , 2001.
- Jelassi, S., Brandfonbrener, D., Kakade, S. M., and Malach, E. Repeat after me: Transformers are better than state space models at copying. In Forty-first International Conference on Machine Learning, ICML 2024 , 2024. URL https://arxiv.org/abs/2402.01032 .
- Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F. Transformers are rnns: Fast autoregressive transformers with linear attention. In International Conference on Machine Learning , 2020. URL https://arxiv.org/abs/2006.16236 .

- Korsky, S. A. On the computational power of RNNs . PhD thesis, Massachusetts Institute of Technology, 2019. URL https://dspace.mit.edu/handle/1721.1/127704 .
- Liao, R., Xiong, Y., Fetaya, E., Zhang, L., Yoon, K., Pitkow, X., Urtasun, R., and Zemel, R. Reviving and improving recurrent back-propagation. In Proceedings of the 35th International Conference on Machine Learning , pp. 3082-3091, 2018. URL https://arxiv.org/abs/1803.06396 .
- Lim, Y. H., Zhu, Q., Selfridge, J., and Kasim, M. F. Parallelizing non-linear sequential models over the sequence length. In The Twelfth International Conference on Learning Representations, ICLR 2024 , 2024. URL https://arxiv.org/abs/2309.12252 .
- Liu, Y., Tian, Y., Zhao, Y., Yu, H., Xie, L., Wang, Y., Ye, Q., Jiao, J., and Liu, Y . Vmamba: Visual state space model. In Advances in Neural Information Processing Systems 38, NeurIPS 2024 , 2024. URL https://arxiv.org/abs/2401.10166 .
- Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In International Conference on Learning Representations , 2017. URL https://arxiv.org/abs/1711.05101 .
- Martin, E. and Cundy, C. Parallelizing linear recurrent neural nets over sequence length. In 6th International Conference on Learning Representations, ICLR 2018 , 2018. URL https: //arxiv.org/abs/1709.04057 .
- Merrill, W. and Sabharwal, A. The parallelism tradeoff: Limitations of log-precision transformers. Transactions of the Association for Computational Linguistics , 11:531-545, 2023. URL https: //arxiv.org/abs/2207.00729 .
- Merrill, W., Petty, J., and Sabharwal, A. The illusion of state in state-space models. In Forty-first International Conference on Machine Learning, ICML 2024 , 2024. URL https://arxiv.org/ abs/2404.08819 .
- Miyato, T., Löwe, S., Geiger, A., and Welling, M. Artificial kuramoto oscillatory neurons. In The Thirteenth International Conference on Learning Representations , 2025. URL https://arxiv. org/abs/2410.13821 .
- Nguyen, E., Poli, M., Durrant, M. G., Kang, B., Katrekar, D., Li, D. B., Bartie, L. J., Thomas, A. W., King, S. H., Brixi, G., et al. Sequence modeling and design from molecular to genome scale with Evo. Science , 2024. URL https://www.science.org/doi/10.1126/science.ado9336 .
- Orvieto, A., Smith, S. L., Gu, A., Fernando, A., Gulcehre, C., Pascanu, R., and De, S. Resurrecting recurrent neural networks for long sequences. In International Conference on Machine Learning , 2023. URL https://arxiv.org/abs/2303.06349 .
- Pascanu, R., Mikolov, T., and Bengio, Y. On the difficulty of training recurrent neural networks. In International Conference on Machine Learning , 2013. URL https://arxiv.org/abs/1211. 5063 .
- Penedo, G., Kydlíˇ cek, H., allal, L. B., Lozhkov, A., Mitchell, M., Raffel, C., Werra, L. V ., and Wolf, T. The fineweb datasets: Decanting the web for the finest text data at scale. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track , 2024. URL https://arxiv.org/abs/2406.17557 .
- Peng, B., Goldstein, D., Anthony, Q., Albalak, A., Alcaide, E., Biderman, S., Cheah, E., Du, X., Ferdinan, T., Hou, H., et al. Eagle and Finch: RWKV with matrix-valued states and dynamic recurrence. In First Conference on Language Modeling , 2024. URL https://arxiv.org/abs/ 2404.05892 .
- Peng, B., Zhang, R., Goldstein, D., Alcaide, E., Du, X., Hou, H., Lin, J., Liu, J., Lu, J., Merrill, W., Song, G., Tan, K., Utpala, S., Wilce, N., Wind, J. S., Wu, T., Wuttke, D., and Zhou-Zheng, C. Rwkv-7 "goose" with expressive dynamic state evolution. In Second Conference on Language Modeling , 2025. URL https://arxiv.org/abs/2503.14456 .
- Qin, Z., Yang, S., Sun, W., Shen, X., Li, D., Sun, W., and Zhong, Y. HGRN2: Gated linear RNNs with state expansion. In First Conference on Language Modeling , 2024. URL https: //arxiv.org/abs/2404.07904 .

- Rumelhart, D. E., Smolensky, P., McClelland, J. L., and Hinton, G. Sequential thought processes in pdp models. Parallel Distributed Processing: Explorations in the Microstructures of Cognition , 1986.
- Sarrof, Y., Veitsman, Y., and Hahn, M. The expressive capacity of state space models: A formal language perspective. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024. URL https://arxiv.org/abs/2405.17394 .
- Saunshi, N., Dikkala, N., Li, Z., Kumar, S., and Reddi, S. J. Reasoning with latent thoughts: On the power of looped transformers. In International Conference on Learning Representations (ICLR) , 2025. URL https://arxiv.org/abs/2502.17416 .
- Schlag, I., Irie, K., and Schmidhuber, J. Linear transformers are secretly fast weight programmers. In International Conference on Machine Learning , 2021a. URL https://arxiv.org/abs/2102. 11174 .
- Schlag, I., Munkhdalai, T., and Schmidhuber, J. Learning associative inference using fast weight memory. In International Conference on Learning Representations (ICLR) , 2021b. URL https: //openreview.net/forum?id=TuK6agbdt27 .
- Schwarzschild, A., Borgnia, E., Gupta, A., Huang, F., Vishkin, U., Goldblum, M., and Goldstein, T. Can you learn an algorithm? generalizing from easy to hard problems with recurrent networks. In Advances in Neural Information Processing Systems 34: NeurIPS 2021 , pp. 6695-6706, 2021. URL https://arxiv.org/abs/2106.04537 .
- Schöne, M., Rahmani, B., Kremer, H., Falck, F., Ballani, H., and Gladrow, J. Implicit language models are rnns: Balancing parallelization and expressivity. In Forty-second International Conference on Machine Learning , 2025. URL https://arxiv.org/abs/2502.07827 .
- Siegelmann, H. T. and Sontag, E. D. On the computational power of neural nets. In Proceedings of the fifth Annual Workshop on Computational Learning Theory , 1992.
- Siems, J., Carstensen, T., Zela, A., Hutter, F., Pontil, M., and Grazzi, R. Deltaproduct: Increasing the expressivity of deltanet through products of householders. In The Thirty-ninth Annual Conference on Neural Information Processing Systems , 2025. URL https://arxiv.org/abs/2502.10297 .
- Smith, J. T., Warrington, A., and Linderman, S. Simplified state space layers for sequence modeling. In International Conference on Learning Representations , 2023. URL https://arxiv.org/ abs/2208.04933 .
- Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., and Wei, F. Retentive network: A successor to transformer for large language models, 2023. URL https://arxiv.org/abs/ 2307.08621 .
- Tay, Y., Dehghani, M., Abnar, S., Shen, Y., Bahri, D., Pham, P., Rao, J., Yang, L., Ruder, S., and Metzler, D. Long range arena: A benchmark for efficient transformers. In International Conference on Learning Representations , 2020. URL https://arxiv.org/abs/2011.04006 .
- Terzic, A., Hersche, M., Camposampiero, G., Hofmann, T., Sebastian, A., and Rahimi, A. On the expressiveness and length generalization of selective state space models on regular languages. Proceedings of the AAAI Conference on Artificial Intelligence , pp. 20876-20884, 2025. URL https://arxiv.org/abs/2412.19350 .
- Trockman, A., Harutyunyan, H., Kolter, J. Z., Kumar, S., and Bhojanapalli, S. Mimetic initialization helps state space models learn to recall. In Workshop on Neural Network Weights as a New Data Modality , 2024. URL https://arxiv.org/abs/2410.11135 .
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need. Advances in Neural Information Processing Systems , 2017. URL https://arxiv.org/abs/1706.03762 .
- Waleffe, R., Byeon, W., Riach, D., Norick, B., Korthikanti, V., Dao, T., Gu, A., Hatamizadeh, A., Singh, S., Narayanan, D., et al. An empirical study of Mamba-based language models, 2024. URL https://arxiv.org/abs/2406.07887 .

- Wang, S., Li, B. Z., Khabsa, M., Fang, H., and Ma, H. Linformer: Self-attention with linear complexity, 2020. URL https://arxiv.org/abs/2006.04768 .
- Weston, J., Bordes, A., Chopra, S., Rush, A. M., van Merriënboer, B., Joulin, A., and Mikolov, T. Towards AI-complete question answering: A set of prerequisite toy tasks, 2015. URL https: //arxiv.org/abs/1502.05698 .
- Yang, S., Wang, B., Shen, Y., Panda, R., and Kim, Y. Gated linear attention transformers with hardware-efficient training. In Forty-first International Conference on Machine Learning, ICML 2024 , 2024a. URL https://arxiv.org/abs/2312.06635 .
- Yang, S., Wang, B., Zhang, Y., Shen, Y., and Kim, Y. Parallelizing linear transformers with the delta rule over sequence length. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024b. URL https://arxiv.org/abs/2406.06484 .
- Yang, S., Kautz, J., and Hatamizadeh, A. Gated delta networks: Improving mamba2 with delta rule. In The Thirteenth International Conference on Learning Representations , 2025. URL https://arxiv.org/abs/2412.06464 .
- Zucchet, N. and Orvieto, A. Recurrent neural networks: vanishing and exploding gradients are not the end of the story. Advances in Neural Information Processing Systems , pp. 139402-139443, 2024. URL https://arxiv.org/abs/2405.21064 .

## Appendices

| A   | Background and Literature Review (Sec. 2)   | Background and Literature Review (Sec. 2)                                                |   16 |
|-----|---------------------------------------------|------------------------------------------------------------------------------------------|------|
| B   | Fixed-Points as an RNN Layer (Sec. 3)       | Fixed-Points as an RNN Layer (Sec. 3)                                                    |   17 |
|     | B.1                                         | Proof for Theorem 3.1 (Lipschitz constant of f θ ( x , h ) is < 1 ) . . . . . . . . .    |   17 |
|     | B.2                                         | Effect of normalization factor ( I - Λ t ) on class of matrices A t . . . . . . . .      |   17 |
|     | B.3                                         | Implicit Differentiation for Optimizing Fixed-Point RNNs . . . . . . . . . .             |   18 |
|     | B.4                                         | Proof for Theorem 3.2 (Gradient of f θ ( x , h ) is a descent direction of F θ ( x ) ) . |   18 |
| C   | Fixed-Point Mamba (Sec. 4)                  | Fixed-Point Mamba (Sec. 4)                                                               |   19 |
|     | C.1                                         | Mamba: Selective SSMs . . . . . . . . . . . . . . . . . . . . . . . . . . . .            |   19 |
|     | C.2                                         | FP-Mamba Parametrization . . . . . . . . . . . . . . . . . . . . . . . . . . .           |   19 |
|     | C.3                                         | Parameterizing the mixers . . . . . . . . . . . . . . . . . . . . . . . . . . .          |   19 |
|     | C.4                                         | Dependence on H t - 1 in theory . . . . . . . . . . . . . . . . . . . . . . . . .        |   20 |
| D   | Evaluation                                  | Evaluation                                                                               |   21 |
|     | D.1                                         | Task Descriptions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .        |   21 |
|     | D.2                                         | Experimental Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .         |   21 |
|     | D.3                                         | Heuristics to reduce the number of fixed-point iterations . . . . . . . . . . .          |   22 |
| E   | Additional Experimental Results             | Additional Experimental Results                                                          |   23 |
|     | E.1                                         | Language Modeling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .          |   23 |
|     | E.2                                         | Long-Range State-Tracking . . . . . . . . . . . . . . . . . . . . . . . . . . .          |   23 |
|     | E.3                                         | Reasoning on CatbAbI . . . . . . . . . . . . . . . . . . . . . . . . . . . . .           |   24 |
|     | E.4                                         | Modular Arithmetic Task Results . . . . . . . . . . . . . . . . . . . . . . . .          |   25 |
|     | E.5                                         | Effect of ℓ max on test performance and number of iterations ℓ ∗ . . . . . . . .         |   26 |
|     | E.6                                         | Sequential vs Parallel Fixed-Point Iteration . . . . . . . . . . . . . . . . . .         |   26 |
| F   | Low-Rank Expressiveness                     | Low-Rank Expressiveness                                                                  |   27 |

## A Background and Literature Review (Sec. 2)

Since their introduction (Rumelhart et al., 1986; Elman, 1990), RNNs have significantly contributed to the evolution of machine learning methods for sequential data, marked by key innovations such as the LSTM (Hochreiter &amp; Schmidhuber, 1997) and Echo-State Networks (Jaeger, 2001). However, two significant challenges lead to the widespread adoption of the Transformer architecture (Vaswani et al., 2017): first, GPU hardware is optimized for large-scale matrix multiplications. Second, recurrent models are notoriously difficult to train due to vanishing and exploding gradients (Hochreiter et al., 2001; Pascanu et al., 2013).

Beyond softmax attention. The quadratic runtime complexity of Transformers motivated research on the linearization of its attention mechanism (Wang et al., 2020; Chen et al., 2021; Choromanski et al., 2020) - a technique that inevitably brings the sequence mixing mechanism closer to RNNlike processing (Katharopoulos et al., 2020; Schlag et al., 2021a). Recently, improvements on the long-range-arena benchmark (Tay et al., 2020) with state-space models (Gu et al., 2022; Smith et al., 2023) sparked a renewed interest in recurrent models (Gu &amp; Dao, 2024; Sun et al., 2023; De et al., 2024; Qin et al., 2024; Peng et al., 2024; Yang et al., 2024a). New efficient token mixing strategies such as Mamba (Gu &amp; Dao, 2024) showcase impressive results in language modeling (Waleffe et al., 2024) while offering linear runtime complexity. These models are fundamentally diagonal linear RNNs, which enables parallel algorithms such as parallel scans (Martin &amp; Cundy, 2018) and fast linear attention based implementations (Yang et al., 2024b; Dao &amp; Gu, 2024).

Expressivity of Diagonal vs. Dense RNNs. It was recently pointed out by Cirone et al. (2024b) that the diagonality in the hidden-to-hidden state transition inevitably causes expressivity issues, showcasing a stark distinction with classic dense nonlinear RNNs, known to be Turing-complete (Siegelmann &amp;Sontag, 1992; Korsky, 2019) and fully expressive in a dynamical systems sense (Hanson &amp; Raginsky, 2020). Merrill et al. (2024) pointed at a similar issue with diagonality using tools from circuit complexity: in contrast to e.g. LSTMs, diagonal linear RNNs can not express state-tracking algorithms. This issue sparked interest in designing fast non-diagonal recurrent mechanisms and, more generally, in providing architectures capable of solving state-tracking problems. The first example of such an architecture is DeltaNet (Yang et al., 2024b) employing a parallelizable Housholder reflection as a state transition matrix. Endowing this matrix with negative eigenvalues improves tracking in SSMs (Grazzi et al., 2024). In concurrent work, Siems et al. (2025) show that adding more reflections improves state-tracking.

Toy tasks. Several works propose toy tasks to identify specific shortcomings of modern architectures. Specifically, Beck et al. (2024) use the Chomsky hierarchy to organize formal language tasks, of which a modular arithmetic task remains unsolved. With similar motivations, Merrill &amp; Sabharwal (2023) introduce a set of word-problems for assessing state-tracking capabilities, among which the A 5 and S 5 tasks remain unsolved by Transformers and SSMs. Motivated by Transformers outperforming RNNs in memory capabilities, Jelassi et al. (2024) introduce a copying task as a fundamental benchmark for memory. We focus on these tasks to evaluate our Fixed-Point RNN framework.

Recurrence in Depth. Machine learning models that reduce an intrinsic energy through iterations have been an object of interest for decades (Hopfield, 1982; Miyato et al., 2025). For example, recurrence in depth can increase the expressivity of Transformers (Dehghani et al., 2019; Schwarzschild et al., 2021; Giannou et al., 2023; Geiping et al., 2025) and is sometimes also understood as adaptive compute time (Graves, 2016). Under certain assumptions, iterated blocks can converge to an equilibrium point where they implicitly describe an expressive function (Bai et al., 2019; Ghaoui et al., 2021). Recently, this technique has been used to approximate non-linear RNNs with a fixed-point iteration of parallelizable linear RNNs (Lim et al., 2024; Gonzalez et al., 2024). In concurrent work to ours, Schöne et al. (2025) apply an iteration in depth to Mamba-2 and Llama blocks to increase expressivity and show promising results of their implicit language models . In contrast, we derive an explicit fixed-point iteration towards a dense linear RNN with a theoretically motivated parameterization, and focus on theoretical toy tasks.

## B Fixed-Points as an RNN Layer (Sec. 3)

## B.1 Proof for Theorem 3.1 (Lipschitz constant of f θ ( x , h ) is &lt; 1 )

Theorem 3.1. Let f θ ( x , h ) be the diagonal linear RNN with input-independent Λ and Q

<!-- formula-not-decoded -->

If || Λ || 2 &lt; 1 and || I -Q || 2 &lt; 1 , then f θ ( x , h ) has a Lipschitz constant &lt; 1 in h . Proof in App. B.1.

We start the proof with the unrolled form of the linear RNN

<!-- formula-not-decoded -->

Note that in order to prove the theorem, we need to show that

<!-- formula-not-decoded -->

where h and h ′ are two arbitrary hidden states. From the unrolled form, this is equivalent to

<!-- formula-not-decoded -->

From the Cauchy-Schwarz inequality, we can upper-bound the LHS of Eq. 15 as

<!-- formula-not-decoded -->

where h ≤ t corresponds to the concatenation of the hidden states h τ for τ ≤ t . Now to prove this product is &lt; ∥ h -h ′ ∥ 2 , consider the terms individually. Since ∥ ∥ h ≤ t -h ′ ≤ t ∥ ∥ 2 ≤ ∥ h -h ′ ∥ 2 , the remaining terms need to be &lt; 1 . Assuming Λ is contractive, we use the Neumann series ∑ t τ =0 Λ t -τ ≤ ( I -Λ ) -1 and get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This condition can be satisfied if I -Q is contractive. This completes our proof.

## B.2 Effect of normalization factor ( I -Λ t ) on class of matrices A t

For the sake of exposition, the introduction of the implicit parametrization in Sec. 3.1 did not consider input normalization ( I -Λ t ) . However, as discussed in Sec. 3.2 this is a crucial component to stabilize the recurrence in time. To derive the representable dense matrices A t in the presence of the normalization factor ( I -Λ t ) , let us start by assuming a fixed-point was found according to Thm. 3.2:

<!-- formula-not-decoded -->

Rearranging the terms allows to move h ∗ t to the other side

<!-- formula-not-decoded -->

Moving ( I -( I -Λ t )( I -Q t )) back to other side yields

<!-- formula-not-decoded -->

Finally, it remains to show that

Following the standard assumptions that 0 ⪯ Λ t , Q t ⪯ I , the matrix ( I +( Λ -1 t -I ) Q t ) is full rank and ⪰ I . Therefore its inverse A t exists and is contractive. The expressivity of A t is only limited if Λ t ≈ I . This however would also be problematic for diagonal SSM and therefore the Mamba initialization is bias towards Λ t ≺ I . Thus, the normalization does not pose a significant problem for the expressivity of A t in practice.

## B.3 Implicit Differentiation for Optimizing Fixed-Point RNNs

One advantage of converging to a fixed-point over general layer looping lies in model training. Since the gradient with respect to h 0 is not needed, implicit differentiation can be used to avoid storing and backpropagating through the computational graph of the fixed-point iteration, as discussed by Liao et al. (2018), Bai et al. (2019). To see this, consider the Jacobian across ℓ iterations J ℓ x = ∂f θ ∂ x ( x , h ℓ -1 ) . Since h ℓ -1 depends on x as well, we can recursively express J ℓ x in terms of J ℓ -1 x and the Jacobians of a single iteration J x ( h ) = ∂f θ ∂ x ( x , h ) and J h = ∂f θ ∂ h ( x , h ) by applying the chain rule

<!-- formula-not-decoded -->

Instead of unrolling, we can implicitly differentiate h ∗ = f θ ( x , h ∗ ) w.r.t. x , which yields J ∗ x = J x ( h ∗ ) + J h ∗ · J ∗ x . Given the conditions on the Lipschitz constant of f θ ( x , h ) in h , we can assume J h ℓ to be contractive and therefore ( I -J h ℓ ) to be positive definite and invertible. This allows to reformulate as

<!-- formula-not-decoded -->

The case for J ∗ θ works analogously. This means that the gradient w.r.t. the input x and parameters θ can be computed at the fixed-point with the cost of solving ( I -J h ∗ ) -1 . Bai et al. (2021) and Schöne et al. (2025) approximate this inverse using the first terms of the Neumann series, which leads to a truncated backpropagation formulation or phantom gradients , incurring sequential overhead. For iteration with hidden state dependence, we can avoid this inversion altogether with Thm. 3.2:

Theorem 3.2. Let f θ ( x , h ) have Lipschitz constant &lt; 1 and fixed-point h ∗ . If the Jacobians ∂f θ ∂ x ( x , h ) and ∂f θ ∂ h ( x , h ) are equal, then the gradient ∇ θ L ( f θ ( x , h ) , y ) of the loss L ( · , y ) for a target y at the fixed point h = h ∗ is a descent direction of L ( F θ ( x ) , y ) . Proof in App. B.4.

In simple terms, Thm. 3.2 shows that parameterizing f θ ( x , h ) such that J x ( h ) = J h guarantees optimization progress even if the gradient is computed only at the fixed-point. In practice, we observe that adhering to this condition in the form of hidden state dependence speeds-up the convergence of the model during training.

## B.4 Proof for Theorem 3.2 (Gradient of f θ ( x , h ) is a descent direction of F θ ( x ) )

We start the proof by setting δ := ∂ L ∂f and J x := J x ( h ∗ ) . Then, we can write the backward propagation as ∂ L ∂ x = ( J ∗ x ) ⊤ δ . In order to prove that the gradient computed at the fixed-point is a descent direction, we need to show that J ⊤ x δ is in the direction of ( J ∗ x ) ⊤ δ , or in other words, we have δ ⊤ J ∗ x J ⊤ x δ ≥ 0 . This is equivalent to showing that the symmetric part of the matrix J ∗ x J ⊤ x is positive semi-definite.

Now note that from Eq. 17 we have: J ∗ x J ⊤ x = ( I -J h ) -1 J x J ⊤ x . From our assumption J x = J h := J , we need to show that the symmetric part of the matrix ( I -J ) -1 JJ ⊤ is positive semi-definite. Note that ( I -J ) -1 and J commute by application of the Neumann series

<!-- formula-not-decoded -->

which yields ( I -J ) -1 JJ ⊤ = J ( I -J ) -1 J ⊤ . Going back to the definition of positive semidefiniteness, we need to show that δ ⊤ J ( I -J ) -1 J ⊤ δ &gt; 0 for all δ . Setting ω = J ⊤ δ , this is equivalent to having ω ⊤ ( I -J ) -1 ω . Note that from our assumption for the Lipschitz constant of the function, we have ∥ J ∥ 2 &lt; 1 , which means ( I -J ) and ( I -J ) -1 have strictly positive eigenvalues. This completes our proof.

## C Fixed-Point Mamba (Sec. 4)

## C.1 Mamba: Selective SSMs

Mamba is a multi-layer network, with an embedding size of d model. A Mamba block is a matrix state diagonal linear RNN which first expands a sequence of embeddings by a factor of e to size d inner = e × d model, and then computes an element-wise recurrence on the matrix hidden states H t ∈ R d state × d inner as

<!-- formula-not-decoded -->

where λ t ∈ R d state × d inner is an input-dependent state transition vector, b t ∈ R d state an input transition vector, x t ∈ R d inner the input, and ∆ t ∈ R d inner × d inner a diagonal matrix which acts an input normalization term. The matrices are parameterized as:

<!-- formula-not-decoded -->

with ω ∈ R d state × d inner , W ∆ ∈ R d inner × d inner , W b ∈ R d state × d inner , and b ∆ ∈ R d inner . The output of a Mamba block y t ∈ R d inner is a contraction of the matrix hidden state with c t ∈ R d state

<!-- formula-not-decoded -->

for W c ∈ R d state × d inner . Note that Mamba proposes a skip connection of y t + D ⊙ x t , where D ∈ R d inner is an input-independent vector. Finally, the model output is usually scaled by a gated linear unit (GLU) as ˜ y t = g t ⊙ y t , where g t = SiLU ( W g x t ) is a non-linear function of the input.

## C.2 FP-Mamba Parametrization

In our design of FP-Mamba, we aim to minimize our interventions in the underlying architecture in order to showcase the adaptability of our proposed framework. Consequently, we do not modify the careful parameterization of λ and the weight-tied normalization factor ∆ t proposed in the original Mamba formulation, and instead rely on layer normalization to limit the Lipschitz constant of the Mamba function. Specifically, in the FP-Mamba model we redefine b t and c t as b ℓ t = W y b y ℓ -1 t -1 + W x b x t and c t = W y c y ℓ -1 t -1 + W x c x t . The remaining components, namely the state transition matrix λ t and the GLU component are parameterized identically to Mamba.

The normalization is applied to the output of the model y t after each iteration. While in theory projecting the output onto the unit sphere does not guarantee a Lipschitz constant &lt; 1 , we observe that in practice, this helps with stabilizing the forward and backward pass of the fixed-point RNN framework. We attribute this observation to the fact that achieving a &gt; 1 Lipschitz constant requires the output of the RNN to become its additive inverse after an iteration, which rarely happens in practice.

## C.3 Parameterizing the mixers

We parameterize the channel mixer variants as follows:

- Diagonal Plus Low Rank: we define u ℓ it = SiLU ( W x u i x t + W y u i y ℓ -1 t -1 ) and α it = σ ( ( w x α i ) ⊤ x t +( w y α i ) ⊤ y ℓ -1 t -1 + b α i ) , where SiLU ( . ) and σ ( . ) are the SiLU and the sigmoid functions, respectively.
- Householder Reflections: we define similar to the diagonal plus low-rank variant.
- Kronecker: we define D ℓ,n t = diag ( σ ( W x D n x t + W y D n y ℓ -1 t -1 + b D n )) and K n t = mat ( SiLU ( W x K n x t + W y K n y ℓ -1 t -1 + b K n )) for n = 1 , 2 , where diag ( . ) is the operator transforming a vector into a diagonal matrix, and mat ( . ) is the operator transforming a size d vector into a √ d × √ d matrix.

For the diagonal plus low rank and the Householder reflections mixers, we L2 normalize the vectors u it to achieve the unit vector formulation. Note that this does not guarantee a contractive diagonal plus low rank structure, which is why the first variant of the channel mixers are excluded form our

FP-RNN experiments. For the Kronecker variant, we define the matrices K n t as symmetric and positive semi-definite using the Cholesky decompositon structure, and normalize them by their largest eigenvalues. The largest eigenvalue is found using the power iterations method, which we found to be much more efficient for small-scale matrices compared to the functions in the PyTorch framework provided for this purpose.

In all of these parameterization, computing a matrix vector product for each fixed-point iteration can be performed in subquadratic time. Specifically, for the DPLR and the Householder formulation, the computation can be performed in linear time in state-size, while in the kronecker product variant, it can be performed in √ d × √ d for d state-size.

## C.4 Dependence on H t -1 in theory

We hypothesize that the dependence of the matrices λ t , b t , c t , and Q t may provide a mechanism for the model to retain and manipulate positional information over the sequence. Jelassi et al. (2024) and Trockman et al. (2024) show that position embeddings could play a crucial role in copy tasks by acting similar to hashing keys in a hashing table. We extend their mechanistic approach to understand why two-layers of linear attention could need H ℓ -1 t -1 to generate appropriate position embeddings for the hashing mechanism.

Specifically consider y ⊤ t = c ⊤ t H t with H t = H t -1 + b t x ⊤ t , assuming that a linear RNN with matrix-state can express linear attention by setting λ t ≈ 1 ∀ t . Upon receiving an input sequence { x 1 , x 2 , . . . , x δ } of length δ followed by a delimiter element x s , the model is expected to copy the input sequence autoregressively, i.e. to start producing { x 1 , x 2 , . . . , x δ } at output positions δ + 1 to 2 δ . Following Arora et al. (2024), the second layer could use position embeddings as hashing keys to detect and copy each token. More concretely, if the first layer receives a sequence { x 1 , x 2 , . . . , x δ , x s , x 1 , x 2 , . . . , x δ -1 } of size 2 δ and augments it with shifted position embeddings { p i } δ i =1 to produce the hidden sequence { x 1 + p 1 , x 2 + p 2 , . . . , x δ + p δ , x s + p 1 , x 1 + p 2 , . . . , x δ -1 + p δ } , then a second layer can act as a linear transformer and produce the sequence { x 1 , x 2 , . . . , x δ } at output positions δ +1 to 2 δ . In the following, we focus on the conditions for the first layer to produce the shifted position embeddings.

We start by assuming that the first layer has a skip-connection y ⊤ t = c ⊤ t H t + x ⊤ t . In this case, the model can augment the inputs with positional embeddings { p i } δ i =1 if it is able to produce shifted encodings p t -δ = p t for δ &lt; t using p ⊤ t = c ⊤ t H t . This condition can be unrolled as

<!-- formula-not-decoded -->

and is satisfied if the equations

<!-- formula-not-decoded -->

hold. Such conditions could only be true if b t and c t are a function of the previous hidden state H t -1 because they need to be able to retain information about { x i } t -1 i = t -δ +1 . While not an explicit mechanism for copying, this derivation provides insight into why a dependency on H t -1 could be helpful.

## D Evaluation

## D.1 Task Descriptions

In this section, we provide task descriptions for the tasks used in the main text.

State Tracking The task of tracking state in the alternating group on five elements ( A 5 ) is one of the tasks introduced in (Merrill et al., 2024) to show that linear RNNs and SSMs cannot solve state-tracking problems. A 5 is the simplest subset of S 5 , the word problem involving tracking the permutation of five elements. In these tasks, a model is presented with an initial state and a sequence of permutations. As the output, the model is expected to predict the state that results from applying the permutations to the initial state. Solving these task with an RNN requires either a dense transition matrix or the presence of non-linearity in the recurrence. It is therefore a good proxy to verify the state-tracking ability of FP-Mamba. In order to investigate the out-of-distribution generalization ability of the model, we train the model with a smaller train sequence length and evaluate for larger (more than × 3 ) sequence lengths.

Copying We use the copy task (Jelassi et al., 2024) in order to assess the memory capabilities of FP-Mamba. In this task, the model is presented with a fixed-size sequence of elements, and expected to copy a subsequence of it after receiving a special token signaling the start of the copying process. In order to investigate the out-of-distribution generalization ability of the model, we train the models with sequence length &lt; 50 , and assess the × 2 length generalization following Jelassi et al. (2024) and Trockman et al. (2024).

## D.2 Experimental Details

In this section, we will provide our experiment setup for the state tracking, copying, and mod arithmetic tasks. The code is available at github.com/dr-faustus/fp-rnn.

State tracking. We train all models for 5 epochs, with a batch size of 512 , 3 different random seeds, learning rate set to 0 . 0001 , weight decay set to 0 . 01 , gradient clipping 1 . 0 , and the AdamW optimizer (Loshchilov &amp; Hutter, 2017). For the train data, we sample 16 Mdatapoints from all the possible permutations for a sequence length of 16 , and split the data with a ratio of 4 to 1 for train and validation samples. For the test data, we sample 500 k sequences of length 50 . We use the implementation and the hyperparameters provided by Merrill et al. (2024) both for data generation and train/test. We train the model for sequence length 16 on the train sample, and evaluate for sequence lengths 2 through 50 on the test sample. Consequently, each epoch of training consists of 25428 iterations, making the total number of iterations during training to be around 1.25M. Note that the likelihood of overlap between the train and test samples is negligible since exhaustive generation of samples in S 5 and A 5 at sequence length k would amount to 60 k and 30 k , respectively.

Copying. We train all models for 10000 iterations, batch size 128 , 3 different random seeds, learning rate 0 . 00001 , weight decay 0 . 1 , gradient clipping 1 . 0 , the AdamW optimizer, and with linear learning rate decay after a 300 iterations warmup. The data is sampled randomly at the start of the training/evaluation. We use a vocab size of 29 , a context length of 256 , and train the model for copy sequence length in the range 5 to 50 , and evaluate for the range 5 to 100 . we use the implementation and the hyperparameters provided by Jelassi et al. (2024).

Mod arithmetic. Our models are trained for 100000 iterations, batch size 256 , learning rate 0 . 001 , weight decay 0 . 1 , and no gradient clipping. The learning rate is decayed using a cosine scheduling by a factor of 0 . 001 after 10000 iterations of warmup. The data is randomly sampled at the start of training/evaluation. We use a vocab size of 12 , with context length 256 , and train data sequence length in the range 3 to 40 , and the test/evaluation data in the range 40 to 256 . We use the implementation and the hyperparameters provided by Beck et al. (2024) and Grazzi et al. (2024), which are the same hyperparameters used for training and evaluating the baselines.

Language Modeling. For the language modeling task, we use the implmentation provided by Ajroldi (2024). We use a batchsize of 16 × 4 × 4 = 256 , training on 4 A100-80GB GPUs with 4 accumulation steps, which is the batchsize used in the 2.5B setting in (Gu &amp; Dao, 2024). The

learning rate is optimized for the Mamba model ( 0 . 004 ) and train all models with this learning rate, with cosine warmup with 0 . 1 steps. We use the AdamW optimizer with weight decay set to 0 . 1 and β 1 , β 2 set to 0 . 9 , 0 . 95 .

Training Time on A 5 . In order to compare the proposed model to the baselines in terms of computation time, we train all of the baselines and our proposed model using the same hardware (A100-80GB gpus) on the A 5 task. We present the results in Fig. 6. Our Fixed-Point Mamba is trained at different maximum number of fixed-point iterations: between 2 (green) and 16 (blue), or sampled from the Gamma distribution Γ(4 , 1) with mean 4 (gray).

catbAbI In this experiment, we use the setting provided by Schlag et al. (2021b). We optimize the learning rates on Mamba, and use the same learning rate to train FP-Mamba, which we found to be 5 × 10 -4 . We use a batch size of 256 , along with short convolutions, and 1 , 2 , or 4 layers. We set the maximum number of iterations ℓ max to 100 .

## D.3 Heuristics to reduce the number of fixed-point iterations

Given the importance of scalability in current machine learning research, an implicit network needs to be as efficiently designed and implemented as possible. While our theoretical framework improves upon the memory and computational requirements on the backward pass, the forward, and especially finding the fixed-point through fixed-point iterations needs further consideration. In our preliminary experiments, we discover two heuristics that can help with improving this aspect significantly.

The first heuristic is relaxing our definition of convergence to the fixed-point during training. We observe that the number of iterations required to find the fixed-point for the sequences in the model usually has a power-law distribution, with certain outliers in each batch elongating the convergence time. In our experiments, we notice very little difference in the performance of the converged model when we exclude these sequences from our stopping criterion. Consequently, during training, we continue the fixed-point iterations procedure until a certain percentage of the datapoints in the batch (usually set to 75%) satisfy our criteria for convergence.

The second heuristic involves using a momentum-like update rule to accelerate the convergence of fixed-point iterations for certain sequences. Specifically, we observe that by setting the fixed-point update rule to h ℓ +1 = δ · f θ ( x , h ℓ ) + (1 -δ ) · h ℓ for some δ ∈ [0 , 1] , we can accelerate the convergence for certain sequences that are particularly slow to converge. Since this update rule can result in a biased approximation of the fixed-point, we implement a patience-based system that starts with δ = 1 , and reduces the value of δ exponentially when the residues fail to improve.

## E Additional Experimental Results

## E.1 Language Modeling

Figure 8: The validation perplexity of the Mamba model vs. FP-Mamba-K and FP-Mamba-H with r ∈ { 1 , 2 } reflections. Note that all of the hyperparameters of the models are identical for fair comparison.

<!-- image -->

In order to confirm the utility of the fixed-point framework in non-state-tracking settings, we performed an experiment on language modeling. Specifically, we compare the performance of a Mamba with an FP-Mamba, with the same hidden size ( 768 ) and number of layers ( 12 ). The settings are selected according to the 2.5B setup introduced in Gu &amp; Dao (2024). We use a train subsample of the FineWeb dataset (Penedo et al., 2024) with 2B tokens, and a validation subsample with 200K tokens. We use a context length of 2048 for our experiment. For the FP-Mamba model, we use the Householder mixer with 1 and 2 reflections. We report the validation perplexity in Fig. 8.

As we can observe, the fixed-point framework does introduce a significant improvement to the performance of the model on perplexity. However, we note that this improvement cannot be only attributed to the multi-layer hypothesis of implicit models (Giannou et al., 2023), as increasing the number of Householder reflections does seem to be improving the perplexity further. Furthermore, we point out the practicality of the setup, as we can observe in Fig. 7 that in the absence of a state-tracking problem, the number of fixed-point iterations seems to be independent of the sequence length, and instead hover in the &lt; 10 range. Finally, fixed-point iterations are not required in the backward bass and therefore only increase training time moderately.

## E.2 Long-Range State-Tracking

In this section, we investigate the ability of our proposed method in doing state tracking on longer sequences. Specifically, we will use the A 5 and S 5 datasets and train on sequence length 128, while evaluating for sequence lengths in the range [2 , 512] . We also implement the proposed Fixed-Point framework on Mamba2 (Dao &amp; Gu, 2024), and we compared our method to DeltaProduct (Siems et al., 2025). In Fig. 9, we plot the test accuracy for these one-layer models.

Comparing our results to DeltaProduct, we can see that the non-linearity introduced by the FixedPoint dynamics allow for a slight improvement in the performance of the Householder products as the mixer components. Furthermore, we observe that the best performing mixer variant is still the Kroneckers model, which can successfully learn the state-tracking problem in all runs. Moreover, the FP-Mamba2 model demonsterates a better length generalization ability compared to FP-Mamba1, which we attribute to the improved underlying architecture used in the model. As shown in (Dao &amp; Gu, 2024), Mamba2 has better recall capabilities, which can help with length generalization.

<!-- image -->

(e) State Tracking on

A

5

- DeltaProduct

(f) State Tracking on

S

5

- DeltaProduct

Figure 9: The state-tracking experiment for train sequence length 128 and evaluation sequence length [2 , 512] . We omit the results of DPLR mixer due to poor performance. The figure presents (a, b) the results for FP-Mamba with Householder ( H ) and Kronecker ( K ) mixer, (c, d) the results for FP-Mamba2 with Householder ( H ) and Kronecker ( K ) mixer, and (e, f) the DeltaProduct method (Siems et al., 2025) for Householder mixers.

## E.3 Reasoning on CatbAbI

In order to investigate the state-tracking ability of the fixed-point framework in a natural language setting, we perform experiments on the catbAbI dataset (Schlag et al., 2021b). catbAbI (concatenatedbAbI) is a reprocessing of the bAbI QA benchmark (Weston et al., 2015), where individual bAbI stories are stitched into one long, continuous sequence, so models must keep track of state across story boundaries. The task tries to stress-test the long-range state tracking and associative inference capabilities of sequence models beyond short, isolated contexts. Each sample in this dataset is a short story. At the end of each story, the model needs to choose a single word that is the answer to the question corresponding to the story. The responses include yes/no responses and the names of characters or locations in the story. We present the results in Table 3.

In order to observe and compare the effect of more complex mixers with the number of layers, we use 1 , 2 , and 4 layers along with the Kronecker and Householder mixer with r ∈ { 1 , 2 , 3 } reflections. Our investigation shows that increasing the number of layers seems to be reaching the point of diminishing returns very fast, while the fixed-point framework improves the performance. This observation seems to be in line with the findings of Saunshi et al. (2025), where the looped architecture seems to be providing a very helpful inductive bias for solving reasoning tasks. Comparing the performance of mixers, we observe that the Kronecker mixer under-performs compared to the Householder

Table 3: Test accuracy of the Mamba model vs. the FP-Mamba model for the Kronecker ( K ) and the Householder ( H ) channel mixers with r ∈ { 1 , 2 , 3 } on the catbAbI dataset. We increase the number of layers to show the effect of having more layers on all models. The task benefits from the fixed-point dynamic, but increasing the number of layers seems to be suffering from diminishing returns.

| # Layers   | Mamba   | FP-Mamba-K   | FP-Mamba-H ( r = 1 )   | FP-Mamba-H ( r = 2 )   | FP-Mamba-H ( r = 3 )   |
|------------|---------|--------------|------------------------|------------------------|------------------------|
| 1 Layer    | 78.28%  | 79.93%       | 81.32%                 | 81.60%                 | 80.79%                 |
| 2 Layers   | 87.08%  | 84.16%       | 89.08%                 | 87.47%                 | 89.55%                 |
| 4 Layers   | 86.51%  | -            | -                      | -                      | -                      |

mixer, which we believe is in line with our observation in App. E.1, where the Kronecker mixer underperforms on tasks involving natural languages.

## E.4 Modular Arithmetic Task Results

Following Grazzi et al. (2024), we also evaluate FP-Mamba on the remaining unsolved task of the Chomsky Hierarchy of language problems introduced by Beck et al. (2024). Specifically, we focus on the mod arithmetic task with brackets. Following the setup of Grazzi et al. (2024), we train on sequence lengths 3 to 40 and report scaled accuracies on test sequences of lengths 40 to 256 . For FP-Mamba, we use a 2 -layer model with r = 4 reflections, i.e. the best performing model in the A 5 experiment.

In Tab. 4, we observe that a 2-layer FP-Mamba-H outperforms the baselines reported in (Grazzi et al., 2024) with a comparable number of parameters. In Fig. 10, we plot the validation accuracy as a function of the number of fixed-point iterations. We observe that the accuracy plateaus at 20 iterations, which is significantly less than the shortest and longest sequence in the validation set. Therefore, the number of iterations required by FP-Mamba-H to reach its fixed point clearly does not scale with the sequence length in this task.

Table 4: The accuracy of various models on modular arithmetic with brackets. We adopt the reported numbers in (Grazzi et al., 2024) evaluating baselines the extended [ -1 , 1] eigenvalue range. Scores are commonly used scaled accuracies between 1 . 0 and 0 . 0 (random guessing). Highlighted is the best performance in each category.

| Model                 |   Accuracy |
|-----------------------|------------|
| 2L Transformer        |      0.025 |
| 2L mLSTM              |      0.034 |
| 2L sLSTM              |      0.173 |
| 2L Mamba              |      0.136 |
| 2L DeltaNet           |      0.2   |
| 2L GatedDeltaProduct  |      0.342 |
| 2L FP-Mamba ( r = 4 ) |      0.384 |

Figure 10: Number of fixed-point iterations on the modular arithmetic task at test time. We report the validation accuracy after convergence for the number of fixed-point iterations caped at various values ranging from 2 to 512 . The pink dashed line denotes the maximum sequence length during validation.

<!-- image -->

## E.5 Effect of ℓ max on test performance and number of iterations ℓ ∗

Figure 11: The effect of ℓ max on the performance of the model ( (a) ), and on the number of iterations ℓ ∗ ( (b) ) on the A 5 task. The vertical line denotes the train sequence length. All of the experiments are performed on FP-Mamba1 with a Householder mixer with r = 1 reflections. Results are averaged across 4 runs.

<!-- image -->

In Fig. 11 we present the effect of the maximum number of iterations ℓ max during training on the accuracy and the number of iterations ℓ ∗ during inference. As we observe, the general trend is that increasing ℓ max improves the performance of the model. We attribute this observation to how well the model learns the task, as following Thm. 3.2, a condition for the gradients being a descent direction is for them to be computed at or close to the fixed-point. Consequently, we can see that when trained with a smaller number of iterations (small ℓ max), the model fails to fully utilize the fixed-point by adapting ℓ ∗ to the difficulty of the task.

## E.6 Sequential vs Parallel Fixed-Point Iteration

An important detail about the fixed-point framework proposed in this paper is that it is not convex. Therefore, the fixed-point is not necessarily unique, which can be problematic in autoregressive applications because there are no guarantee that the parallel fixed-point during training will be the same as the sequential fixed-point used during inference (Schöne et al., 2025). In order to investigate this issue, we trained an FP-Mamba-H model on the A 5 task and compared the fixed-point computed sequentially and in parallel. We report the results in Fig. 12. We observe that the fixed-points are extremely similar, providing the possiblity of computing the fixed-point sequentially during inference.

Figure 12: The difference between the fixed-point computed sequentially (i.e., computing the fixed-point for each token separately) and the fixed-point computed in parallel (i.e., computed through Eq. 12) on the A 5 task trained on sequence length 16 to convergence. The x-axis denotes the test sequence length, and the y-axis the normalized difference. The dashed gray line denotes the threshold for stopping the fixed-point iterations.

<!-- image -->

## F Low-Rank Expressiveness

In this section, we prove that SSMs with low-rank structure can be maximally expressive under weak assumptions on the growth of the rank with hidden dimension. To do this we first place ourselves in the general setting of (Cirone et al., 2024b), accordingly we consider models given by controlled differential equations of type 3 :

<!-- formula-not-decoded -->

Following the notation and methodology of Cirone et al. (2024b)[B.4] ), this can be written in terms of the Signature as

<!-- formula-not-decoded -->

where W d ω is the set of words in the alphabet [[ d ω ]] := { 1 , . . . , d ω } ( i.e. W d ω = ⋃ n ≥ 0 [[ d ω ]] n ) and for a given word I = i 1 . . . i n with S I ( ω ) [0 ,t ] we refer to the I th component of the signature tensor S ( ω ) [0 ,t ] i.e.

<!-- formula-not-decoded -->

It follows directly from Eq. 20 that any linear readout of Y t can be represented as a series in signature terms. As a result, these systems are fundamentally restricted to learning functions that closely approximate these convergent series.

Maximal expressivity is attained when any finite linear combination of signature terms can be approximated by a linear readout on Y t via suitable configurations of the matrices A i .

Definition F.1. Fix a set of paths X ⊆ C 1 -var ([0 , 1]; R d ) . We say that a sequence ( A N , Y N ) N ∈ N , where Y N ⊆ R N and A N ⊆ R N × N , achieves maximal expressivity for X whenever for any positive tolerance ϵ &gt; 0 and any finite linear combination coefficients α ∈ T ( R d ) there exist a choice of parameters v, ( A i ) , Y 0 in some R N , A N , Y N in the sequence such that v ⊤ Y (( A i ) , Y 0 , ω ) · is uniformly close to ⟨ α, S ( ω ) [0 , · ] ⟩ up to an error of ϵ i.e.

<!-- formula-not-decoded -->

If we are given a sequence of probabilities P N on A d N ×Y N such that ∀ ϵ &gt; 0 , ∀ α ∈ T ( R d ) it holds that

<!-- formula-not-decoded -->

then we say that ( A N , Y N , P N ) N ∈ N achieves maximal probabilistic expressivity for X .

As discussed in the main body of this work in (Cirone et al., 2024b) the authors prove that ( R N × N , R N , P N ) , where P N is a Gaussian measure corresponding to the classical Glorot initialization scheme in deep learning, achieves maximal probabilistic expressivity for compact sets.

Albeit expressiveness is thus maximally attained the resulting matrices A i are almost-surely dense, hence the models are not efficiently implementable. As the next result suggests, a possible alternative is given by low-rank matrices:

Proposition F.2. The sequence of triplets ( R N × N , R N , P N ) where P N is such that

3 For simplicity we have omitted the dξ term, as the results and proof change minimally in form but not in spirit.

- the initial value has independent standard Gaussian entries [ Y 0 ] α iid ∼ N (0 , 1) ,
- the weight matrices are distributed as A i iid ∼ 1 √ Nr N WM ⊤ with W and M independent N × r N matrices having entries [ W ] α,β , [ M ] α,β iid ∼ N (0 , 1) ,
- the rank parameter r N satisfies r N →∞ as N →∞

achieves maximal probabilistic expressivity for compact sets.

Proof. Following (Cirone et al., 2024b)[B.3.5] we only need to prove a bound of type

<!-- formula-not-decoded -->

as in the full-rank Gaussian case.

We will place ourselves in the graphical setting of (Cirone et al., 2024a) and leverage the fact that ( c.f. (Cirone et al., 2024a)[7.1]) their results and techniques naturally hold for rectangular matrices.

In our setting 1 N ⟨ A I Y 0 , A J Y 0 ⟩ R N corresponds to a product graph G I,J corresponding to a ladder having 2 | I | +2 | J | edges as shown in Fig. 13. We can then use (Cirone et al., 2024a)[Prop. 2] to compute the square of the L 2 norm in equation Eq. 22, the only difference from the dense case is that half of the vertices (excluding the "middle" one) correspond to a space of dimension r N while the rest to the standard N .

Since r N →∞ and given the scaling N -1 ( Nr N ) -| I | + | J | 2 , the admissible pairings of G I,J not of order o (1) are only the leading ones. These correspond to product graphs with | I | + | J | 2 r N -dimensional vertices and | I | + | J | 2 +1 N -dimensional vertices. By the same reasoning as in the full-rank case, these are found to be just the identity pairings.

Moreover, all pairings of G I,J ⊔ G I,J that do not result in an identity pairing in at least one of the two copies are O ( 1 N ∧ r N ) ( instead of O ( 1 N ) ). This follows as in the full-rank case.

Since the total number of admissible pairings of G I,J ⊔ G I,J is (4( | I | + | J | ))! ! , we conclude that equation 22 holds with κ = 4 and o (1) := O ( 1 √ N ∧ r N ) .

<!-- formula-not-decoded -->

Figure 13: The product graph G I,J for I = i 1 i 2 i 3 and J = j 1 .

Remark F.3 . Following (Cirone et al., 2024a)[6.1] it's possible to prove that the W and M can be taken as having iid entries from a centred, symmetric but heavy tailed distribution given finiteness of even moments. This distributional choice comes useful in controlling the eigenvalues of A = WM ⊤ . Remark F.4 . While the proof crucially uses the assumption r N →∞ as N →∞ , at the same time we have not provided an argument against r N not diverging. In Fig. 14 we present a counterexample, showing that if r N does not diverge then the asymptotics differ from the dense ones, in particular some symmetries are "lost", impossible to recover due to unavoidable noise.

<!-- image -->

̸

Figure 14: Admissible pairing different from the "identity" paring, but still leading to maximal asymptotic scaling in the bounded r N case. Here, I = 12 = 1112 = J , and we have highlighted in blue the vertices corresponding to the bounded dimension r N . Recall that edges without arrows correspond to the matrix I (matrix of ones), and that two edges corresponding to matrices A and B which share direction and terminal vertices can be merged into the edge A ⊙ B .

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: In the paper we make claims about the proposed method increasing the expressivity of the model, for which we provide experimental and theoretical justification.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Yes, we discuss the limitations in the discussion section of the paper.

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

Justification: Yes, we provide the set of assumptions in the description of the theories, and the correct proof in the appendix.

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

Justification: Yes, we provide the experimental setups in the paper. A full experimental setup description can be found in the appendix.

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

Justification: All of the datasets used in our paper are open access. We intend to provide the code for our paper after publication.

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

Justification: Yes, all of the details are available in the appendix, and the corresponding papers are cited.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All of the experiments either provide error bars, or the setting is specifically mentioned in the text and justified using cited material.

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

Justification: Yes, we provide the details of the hardware used in the experiments in the appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We believe our work conforms with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: As our paper is mostly concerned with expressivity of RNNs and theoretical and empirical justifications for it, we believe this issue does not apply to our work.

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

Justification: As our paper is mostly concerned with expressivity of RNNs and theoretical and empirical justifications for it, we believe this issue does not apply to our work.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Yes, we cite the papers providing or proposing the datasets and models used in our experiments in the paper.

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

Justification: Our paper does not involve crowd sourcing.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our paper does not involve crow sourcing or working with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: We do not use LLMs in this paper.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.