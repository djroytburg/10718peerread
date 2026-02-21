## Trained Mamba Emulates Online Gradient Descent in In-Context Linear Regression

Jiarui Jiang ∗ 1 , Wei Huang ∗ 2 , Miao Zhang † 1 , Taiji Suzuki 3 2 , Liqiang Nie 1

1 Harbin Institute of Technology, Shenzhen 2 RIKEN AIP

3 University of Tokyo jiaruij@outlook.com , wei.huang.vr@riken.jp , zhangmiao@hit.edu.cn taiji@mist.i.u-tokyo.ac.jp , nieliqiang@gmail.com

## Abstract

State-space models (SSMs), particularly Mamba, emerge as an efficient Transformer alternative with linear complexity for long-sequence modeling. Recent empirical works demonstrate Mamba's in-context learning (ICL) capabilities competitive with Transformers, a critical capacity for large foundation models. However, theoretical understanding of Mamba's ICL remains limited, restricting deeper insights into its underlying mechanisms. Even fundamental tasks such as linear regression ICL, widely studied as a standard theoretical benchmark for Transformers, have not been thoroughly analyzed in the context of Mamba. To address this gap, we study the training dynamics of Mamba on the linear regression ICL task. By developing novel techniques tackling non-convex optimization with gradient descent related to Mamba's structure, we establish an exponential convergence rate to ICL solution, and derive a loss bound that is comparable to Transformer's. Importantly, our results reveal that Mamba can perform a variant of online gradient descent to learn the latent function in context. This mechanism is different from that of Transformer, which is typically understood to achieve ICL through gradient descent emulation. The theoretical results are verified by experimental simulation.

## 1 Introduction

State-space models (SSMs), notably Mamba (Gu and Dao, 2024), have recently emerged as compelling alternatives to Transformer-based architectures (Vaswani et al., 2017). Mamba integrates gating, convolutions, and state-space modeling with selection mechanisms, enabling linear-time complexity. This effectively addresses the quadratic computational costs typically associated with self-attention mechanisms in Transformers. Consequently, Mamba demonstrates superior efficiency in processing long sequences while maintaining or even surpassing Transformer performance across diverse benchmarks (Gu and Dao, 2024; Dao and Gu, 2024; Patro and Agneeswaran, 2024; Liu et al., 2024; Ahamed and Cheng, 2024; Li et al., 2024a,b).

In-context learning (ICL) (Brown et al., 2020) is a powerful paradigm that enables models to generalize to unseen tasks by dynamically leveraging contextual examples (such as input-output pairs) without task-specific fine-tuning. This capability has become a defining characteristic of large foundation models, significantly enhancing their flexibility and adaptability. While extensive research has provided substantial insights into Transformer-based ICL mechanisms (Garg et al., 2022; Gatmiry et al., 2024; Sander et al., 2024; Zheng et al., 2024; Zhang et al., 2025), the principles underlying

∗ Equal contribution

† Corresponding author

Mamba's ability to perform in-context learning remain largely unexplored, highlighting a compelling research gap.

Recent empirical studies have examined Mamba's (ICL) capabilities, showing it matches Transformers on many standard ICL tasks, while surpassing them in specialized scenarios like sparse parity (Park et al., 2024; Grazzi et al., 2024). Bondaschi et al. (2025) theoretically analyzed its representational capacity for in-context learning of Markov chains, and Li et al. (2025a) investigated binary classification tasks with outliers. (Yang et al., 2024, 2025; Behrouz et al., 2025b,a) leverage the connection between SSMs and online learning to design new architectures. However, even the linear regression model, a canonical setting widely used for theoretical analysis of Transformer-based ICL mechanisms, remains theoretically underexplored in the context of Mamba. To fill this gap, we analyze Mamba's training dynamics on in-context linear regression tasks. More precisely, following the previous ICL analysis in Transformers (Garg et al., 2022; Zhang et al., 2024; Ahn et al., 2023), this paper focuses on a data generative model with N input-output pairs ( { x i , y i } N i =1 ) and a query input ( x q ) satisfying y = f ( x ) = w ⊤ x , where x denotes the input and y denotes the output, and w is randomly sampled from Gaussian distribution, termed the context . In this work, we develop a rigorous theoretical framework to analyze how randomly initialized Mamba models, when trained through gradient descent, evolve to implement in-context learning. We demonstrate that the trained Mamba architecture dynamically leverages the input context to perform implicit estimation of the vector w . This estimation is achieved through hidden state updates that mimic online gradient descent steps, finally implementing prediction for y q = f ( x q ) = w ⊤ x q . We also provide a loss bound that is comparable to Transformers'. Our contributions are summarized as follows:

- We construct a Mamba architecture (S6: S4 with selection) capable of ICL, establishing its exponential convergence rate to ICL solution, and further derive the loss bound after convergence. The loss matches that of Transformers.
- Technically, we develop novel techniques to address optimization challenges induced by random initialization and gradient descent, rigorously characterizing Mamba's training dynamics when trained from scratch.
- Wereveal how trained Mamba achieves in-context linear regression by progressively aligning its hidden states with the context through sequential token processing. This finding provides a new perspective for understanding Mamba's ICL mechanism, distinct from Transformerbased approaches. All the above results are verified by experiments.

## 2 Related Work

In-Context Learning The seminal work of Brown et al. (2020) demonstrated the in-context learning capability in Transformers, showing their ability to infer functional mappings from input-output exemplars without weight updates. Garg et al. (2022) initiated the investigation of ICL from the perspective of learning particular function classes. Following these, a line of research analyze this phenomenon through the lens of algorithm imitation: Transformers can be trained to implement various learning algorithms that can mimic the latent functions in context, including: a single step of gradient descent (Von Oswald et al., 2023; Akyürek et al., 2023), statistical algorithms (Bai et al., 2023), reinforcement learning algorithm (Lin et al., 2024), multi-step gradient descent (Gatmiry et al., 2024), mesa-optimization (Zheng et al., 2024), Newton's method (Giannou et al., 2025), weighted preconditioned gradient descent (Li et al., 2025b), in context classification (Bu et al., 2024; Shen et al., 2024; Bu et al., 2025) among others.

Recent work extends ICL analysis beyond Transformers: (Lee et al., 2024; Park et al., 2024) empirically compared popular architectures (e.g., RNNs, CNNs, SSMs, Transformers) on synthetic ICL tasks, identifying capability variations across model types and task demands. Tong and Pehlevan (2024) demonstrate that MLPs can learn in-context a series of classical tasks such as regression and classification with less computation than Transformers. Sushma et al. (2024) show that state space models augmented with local self-attention can learn linear regression in-context. Unlike existing research on ICL, this work focuses on the ICL mechanism of Mamba (specifically S4 with selection) and its training dynamics.

Theoretical Understanding of SSMs As Gu et al. (2022) introduce structured state spaces models in modeling long sequence and further be extended to Mamba (Gu and Dao, 2024), which gained

significant attention as alternatives to Transformers, extensive research has sought to theoretically understand the mechanisms and capabilities of state-spaces models (SSMs). Dao and Gu (2024) propose the framework of state space duality, which establishes a connection between SSMs and attention variants through the lens of structured matrices. Vankadara et al. (2024) provide a scaling analysis of signal propagation in SSMs through the lens of feature learning. Cirone et al. (2024) draw the link of SSMs to linear CDEs (controlled differential equations) and use tools from rough path theory to study their expressivity. Chen et al. (2025) establish the computational limits of SSMs and Mamba via circuit complexity analysis, questioning the prevailing belief that Mamba possesses superior computational expressivity compared to Transformers. Nishikawa and Suzuki (2025) demonstrate that state space models integrated with nonlinear layers achieve dynamic token selection capabilities comparable to Transformers. Different from the above, we provide theoretical understanding of Mamba from the perspective of ICL.

## 3 Problem Setup

In this section, we outline the ICL data model, the Mamba model, the prediction strategy, and the gradient descent training algorithm.

Data Model. We consider an in-context linear regression task where each prompt corresponds to a new function f ( x ) = w ⊤ x with weights w ∼ N (0 , I d ) and d &gt; 1 . For each task, we generate N i.i.d. input-output pairs { ( x i , y i ) } N i =1 and a query x q , where all inputs x i , x q ∼ N (0 , I d ) are independent Gaussian vectors, and the outputs satisfy y i = f ( x i ) . The goal is to predict y q = f ( x q ) for the query.

To enable sequential processing of prompts in the Mamba model, we implement an embedding strategy where:

1. The i -th context token is encoded as e i = ( x ⊤ i , y i ) ⊤ , formed by concatenating input x i with its corresponding label y i .
2. The query token is represented as e q = ( x ⊤ q , 0) ⊤ , masking the unknown target value with a zero placeholder.

In many theoretical analyses of Transformer-based in-context learning, token embeddings are conventionally concatenated into a single matrix to enable parallel computation of global attention (Zhang et al., 2024; Ahn et al., 2023; Huang et al., 2023; Mahankali et al., 2024; Wu et al., 2024). In contrast, since Mamba operates as a sequential model, we feed the embeddings of context tokens one by one, and finally the query token ( e 1 → e 2 →··· → e N → e q ).

Mamba Model. We consider a S6 layer of Mamba o 1: L = Mamba ( θ ; u 1: L ) with selection, discretization, and linear recurrence components, where u l , o l ∈ R d e . It can be described as follows:

<!-- formula-not-decoded -->

o ( i ) l = C ⊤ l h ( i ) l , C l ∈ R d h × 1 , (1b) B l = (∆ l A ) -1 (exp(∆ l A ) -I )∆ l B l ∈ R d h × 1 (2b) for i ∈ [ d e ] . Here, the superscript ( i ) denotes the i -th independent processing channel, where each channel operates on a unique feature dimension of the input u l and output o l vectors (i.e., u ( i ) l and o ( i ) l correspond to the i -th elements of u l and o l , respectively). The hidden state h ( i ) l is initialized as h ( i ) 0 = 0 and evolves according to A l ∈ R d h × d h , B l ∈ R d h × 1 and the input u ( i ) l . C l ∈ R d h × 1 maps the hidden state h ( i ) l to the output o ( i ) l . As shown in (2), A l and B l are computed using the zero-order hold (ZOH) discretization method applied to A ∈ R d h × d h , B l ∈ R d h × 1 and the timestep ∆ l ∈ R . Next, we describe the selection mechanism.

B l = W B u l + b B , (3) C l = W C u l + b C , (4) ∆ l = softplus( w ⊤ ∆ u l + b ∆ ) , (5) Here, softplus(x) = log(1 + exp( x )) . W B , W C ∈ R d h × d e , b B , b C ∈ R d h × 1 , w ∆ ∈ R d e × 1 , b ∆ ∈ R , along with A ∈ R d h × d h are the parameters of the Mamba model. We use θ to denote the collection of all the parameters.

Unlike previous work (Sushma et al., 2024) that introduce local self-attention component to augment SSMs, which may inherit the Transformer's ICL ability, our model adheres to Mamba's original selective state-space framework (Gu and Dao, 2024). This alignment ensures us to mechanistically analyze how Mamba's architecture enables in-context learning (ICL).

Linear Regression Prediction. In this work, we set d e = d +1 , enabling the Mamba model to process the embeddings e 1: N , e q . Given the prompt ( e 1 , . . . , e N , e q ) , the Mamba model will output a sequence o 1: N +1 = Mamba ( θ ; e 1 , . . . , e N , e q ) . The prediction for the linear regression target y q = w ⊤ x q is extracted from the terminal position of the output matrix (corresponding to the zero placeholder in the query token e q = ( x ⊤ q , 0) ⊤ ). Concretely, ˆ y q = o ( d +1) N +1 .

Training Algorithm. To train a Mamba model over the in-context linear regression task, we consider minimizing the following population loss:

<!-- formula-not-decoded -->

Given a Mamba model, we use gradient descent to minimize population loss L ( θ ) , and the update of trainable parameters θ ′ = { W B , W C , b B , b C } can be written as follows:

<!-- formula-not-decoded -->

## 4 Main Results

This section presents our main theoretical results that characterize the convergence state of Mamba and its final loss. We also compare the results with other models.

Assumption 4.1 (1) Matrix A = -I d h . (2) The vector w ∆ is fixed as zero 0 , and b ∆ is fixed as ln(exp((ln 2) /N ) -1) . (3) Matrices W B , W C are initialized with entries drawn i.i.d. from the standard Gaussian distribution N (0 , 1) . (4) The hidden state dimension satisfies: d h = ˜ Ω( d 2 ) . (5) The learning rate satisfies: η = O ( d -2 d -1 h ) . (6) Bias vectors b B , b C are initialized as zero 0 . (7) Token length N = Ω( d ) .

(1) The negative-definite matrix A = -I d h guarantees the stable convergence of hidden states h ( i ) l . (2) Given the zero-mean and symmetric distribution of embeddings, w ∆ can naturally converge to 0 during gradient descent, and we fix it as 0 for simplicity. We further fix b ∆ to an appropriate constant to maintain a suitable timestep ∆ l , enabling us to concentrate our theoretical analysis on W B , W C , b B , and b C . In prior works on Transformer-based in-context learning, merging key-query weights (e.g., W := W Q W K ) and specific initializations (e.g., W Q = W K = I ) are often adopted to simplify optimization analysis (Zhang et al., 2024; Ahn et al., 2023; Huang et al., 2023; Mahankali et al., 2024; Wu et al., 2024). (3, 4, 5) In contrast, our Gaussian initialization of W B and W C demonstrates more practicality, which requires a sufficiently large hidden state dimension d h and a sufficiently small learning rate η to ensure favorable loss landscape properties. Assumption (6) is intended to simplify the analysis. (7) Token length should be larger enough than the dimension of w to capture sufficient contextual information.

Theorem 4.1 Under Assumption 4.1, if the Mamba is trained with gradient descent, and given a new prompt ( e 1 , . . . , e N , e q ) , then with probability at least 1 -δ for some δ ∈ (0 , 1) , the trainable parameters θ ′ ( t ) = { W B ( t ) , W C ( t ) , b B ( t ) , b C ( t ) } converge as t →∞ to parameters that satisfies:

- (a) Projected hidden state: ( W ⊤ C ) [1: d, :] ( t ) h ( d +1) l = α ( W ⊤ C ( t )) [1: d, :] h ( d +1) l -1 +(1 -α ) βy l x l ,
- (b) Prediction for target: ˆ y q = x ⊤ q ∑ N -1 i =0 (1 -α ) α i +1 βy N -i x N -i ,
- (c) Population loss: L ( θ ( t )) ≤ 3 d ( d +1) 2 N ,

<!-- formula-not-decoded -->

Theorem 4.1 characterizes the in-context learning (ICL) mechanism of Mamba and establishes an upper bound on its population loss. Specifically, (Thm 4.1 (a)) shows how the hidden state is updated according the given prompt e l = ( x ⊤ l , y l ) ⊤ . (Thm 4.1 (b)) presents the final prediction given prompt ( e 1 , . . . , e N , e q ) . (Thm 4.1 (c)) provides the upper bound for the population loss, which is comparable to that of the Transformer (Zhang et al., 2024). Next, we'll discuss it in more detail.

Update of Hidden State. If we define ˜ h l := ( W ⊤ C ) [1: d, :] h ( d +1) l , then (Thm 4.1 (a)) can be rewritten as follows:

<!-- formula-not-decoded -->

We observe its intrinsic connection to online gradient descent , which updates the model parameters ( ˜ h l ) with only one currently arriving sample ( e l = ( x ⊤ l , y l ) ⊤ ) at each step. Specifically, the system gradually updates ˜ h l along the pseudo-gradient direction βy l x l , with a fixed step size (1 -α ) .

For a newly defined task f ( x ) = w ⊤ x , given that E [ y l x l ] = w , the direction of ˜ h l converges toward w as mamba processes multiple prompts. This demonstrates mamba's ability to internalize f ( x ) through prompt processing, which ultimately ensures that predictions for query token e q = ( x ⊤ q , 0) ⊤ closely approximate f ( x q ) .

Previous works have shown that Transformer can mimic a single step of gradient descent to achieve incontext learning ability (Zhang et al., 2024; Mahankali et al., 2024). Concretely, a trained Transformer can be described as follows

<!-- formula-not-decoded -->

Our theoretical analysis reveals that Mamba and Transformer have different in-context learning mechanisms. This divergence stems from their inherent architectural biases: Transformers process contexts globally through self-attention, while Mamba enforces local sequential dependencies via recurrent state transitions. These findings provide fundamental insights into the contrasting capabilities of Transformer-based and Mamba-based models for in-context learning. As experimental work shows, transformers can learn vector-valued MQAR tasks in the context which Mamba cannot, while Mamba succeeds in sparse-parity in-context learning tasks where Transformers fail(Park et al., 2024).

Prediction Outcome. Comparing equations Thm 4.1 (b) and (9), we found both similarities and distinctions in how Transformer and Mamba implement in-context learning (ICL). Both models leverage a weighted aggregation of y i x i , aligning with the intuition that learning f ( x ) = x ⊤ w from context reduces to estimating the latent parameter w , since E [ y i x i ] = w . Notably, their token weighting strategies diverge: Transformer's global attention mechanism implicitly assigns nearly uniform weights ( ∼ 1 N , where N is the token length) to all y i x i , while Mamba's linear recurrence imposes position-dependent weight variations. This difference arises from Mamba's iterative state update rule, where the influence of prompt tokens e i on the hidden state h l depends on their sequential placement, governed by the model's linear recurrence dynamics.

The derived upper bound (Thm 4.1 (c)) establishes an O (1 /N ) convergence rate for the loss (ignoring dimension factor), demonstrating that Mamba matches the sample complexity scaling of Transformers in linear regression ICL tasks (Zhang et al., 2024).

Compare with S4. Mamba extends the structured state space model (S4) (Gu et al., 2022) by integrating a selection mechanism, which is critical for enabling ICL. In S4 model, the matrices A ∈ R d h × d h , and B , C ∈ R d h × 1 are static, leading to a fixed linear combination of inputs:

<!-- formula-not-decoded -->

where the coefficients C ⊤ A l -j B are task-agnostic . This formulation inherently limits S4's ability to adapt to task-specific parameters w in ICL scenarios, as the model cannot adjust its inductive bias to match distinct w across different tasks. Therefore, the S4 model cannot truly learn in-context.

In contrast, Mamba's selection mechanism dynamically adjusts B l and C l (and optionally A l ) based on the input tokens ( u 1 , . . . , u N ) . This allows the model to implicitly adapt its hidden state to

align with the latent w of each task, effectively transforming the linear combination weights into context-dependent functions f ( x ) = x ⊤ w . Such adaptability is essential for ICL, as it enables Mamba to reconstruct diverse w from input prompts without task-specific fine-tuning.

## 5 Proof Sketch

This section outlines the main technical ideas to prove Theorem 4.1. The complete proofs are given in the appendix.

Linear Recurrence. To start with, we show how the hidden states update when receiving token e l = ( x ⊤ l , y l ) ⊤ . By (Eq. (5)) and Assumption 4.1(2), we have ∆ l = (ln 2) /N . Combining it with (Eq. (1)(2)) and get:

<!-- formula-not-decoded -->

where α := exp( -∆ l ) = exp(( -ln 2) /N ) , the second equality is by discretization rule (2), the third equality is by Assumption 4.1(2) and exp( -∆ l I ) = exp( -∆ l ) I .

Prediction Output. We next derive the expression of ˆ y q . By recurring (Eq.(11)), the hidden state after receiving the first l context prompts e 1: l is given by h ( d +1) l = (1 -α ) ∑ l -1 i =0 α i y l -i B l -i . Receiving all the prompt tokens e 1: N and the query token e q = ( x ⊤ q , 0) ⊤ , we have:

<!-- formula-not-decoded -->

Finally, the prediction output is as follows

<!-- formula-not-decoded -->

To handle W C e q and W B e N -i , we further decompose W B = [ Bb ] and W C = [ Cc ] , where B , C ∈ R d h × d , b , c ∈ R d h × 1 . Then we write another form of (Eq. (13)):

<!-- formula-not-decoded -->

The loss becomes:

<!-- formula-not-decoded -->

By computing the gradient of C , b C , B , b and b B with respect to L ( θ ( t )) , we derive the following update rule according to Eq. (7).

Lemma 5.1 (Update Rule) Let η be the learning rate and we use gradient descent to update the weights W B , W C , b B , b C , for t ≥ 0 we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Technical Challenges. Unlike many prior Transformer-based ICL analyses that simplify dynamics via merged weights or special initializations, our Gaussian-initialized W B , W C and discrete-time gradient descent introduces more complexity (cf. assumption 4.1). To solve the optimization problem described in Lemma 5.1, we have the following three questions to answer: (1) Convergence Target: Where do the parameters converge? (2) Convergence Proof: How to rigorously establish convergence? (3) Saddle Point Avoidance: How to avoid saddle points? To answer these three questions, we propose two key techniques: Vector-coupled Dynamic , Negative Feedback Convergence , and apply them with a Fine-grained Induction . We next describe them in detail.

## 5.1 Vector-coupled Dynamics

We can verify by Lemma 5.1 that C ⊤ B = Diag( a 1 , . . . , a d ) with a i ∈ { 0 , β 3 β 1 } , C ⊤ b = 0 are the fixed points for the parameters W B , W C .

Combining the loss function Eq. 15 and b B ( t ) = b C ( t ) = 0 in Lemma 5.1, the loss function can be rewritten as

<!-- formula-not-decoded -->

To minimize loss, the term (1 -α ) ∑ N -1 i =0 α i +1 ( x ⊤ q C ⊤ B y N -i x N -i + y 2 N -i x ⊤ q C ⊤ b ) should approximate w ⊤ x q . Given E [ y N -i x N -i ] = w and E [ y 2 N -i ] &gt; 0 , we derive that C ⊤ B should converge to β 3 β 1 I , while C ⊤ b converges to 0 to minimize the loss. However, as mentioned above, C ⊤ B = Diag( a 1 , . . . , a d ) with partial a i = 0 can also enable convergence, which is an undesirable scenario.

To analyze the convergence behavior of C ⊤ B and C ⊤ b , we introduce the Vector-coupled Dynamics technique, which studies the inner product dynamics between decomposed column vectors of B and C . Specifically, we decompose B and C into B = [ b 1 . . . b d ] , C = [ c 1 . . . c d ] . Then we have another form of Lemma 5.1 for B , C and b as the following lemma.

Lemma 5.2 (Vectors Update Rule) Let η be the learning rate and we use gradient descent to update the weights W B , W C , b B , b C , for i ∈ [ d ] , t ≥ 0 we have

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

With Lemma 5.2, we can further analyze the dynamics of the inner products c ⊤ i ( t ) b i ( t ) , c ⊤ i ( t ) b j ( t ) and c ⊤ i ( t ) b ( t ) , precisely characterizing the behavior of C ⊤ B and C ⊤ b . This technique helps answer the question " Where do the parameters converge? "

## 5.2 Negative Feedback Convergence

̸

As we discuss in Section 5.1, to minimize loss, the following conditions must be satisfied for all i, j ∈ [ d ] with i = j : c ⊤ i ( t ) b i ( t ) → β 3 β 1 , c ⊤ i ( t ) b j ( t ) → 0 , c ⊤ i ( t ) b ( t ) → 0 . To establish the convergence, we introduce the Negative Feedback Convergence technique. This technique leverages the negative feedback terms in the dynamical equations of c ⊤ i ( t ) b i ( t ) , c ⊤ i ( t ) b j ( t ) , and c ⊤ i ( t ) b ( t ) to derive an exponential convergence rate. Taking c ⊤ i ( t ) b i ( t ) as an example, we derive the following

update rule by Lemma 5.2.

̸

<!-- formula-not-decoded -->

̸

The term ( β 3 -β 1 c ⊤ i ( t +1) b i ( t +1) ) decomposes into its previous state ( β 3 -β 1 c ⊤ i ( t ) b i ( t ) ) (marked with underline) plus the remaining terms (increment terms). The increment terms includes a negative feedback term , which induces a tendency to drive ( β 3 -β 1 c ⊤ i ( t ) b i ( t ) ) to 0 ( c ⊤ i ( t ) b i ( t ) → β 3 β 1 ).

̸

Intuitively, b ⊤ i ( t ) b i ( t ) and c ⊤ i ( t ) c i ( t ) are much larger than b ⊤ k ( t ) b i ( t ) , c ⊤ i ( t ) c k ( t ) and b ⊤ i ( t ) b ( t ) at Gaussian initialization with high probability. Also, as c ⊤ i ( t ) b j ( t ) , b ⊤ i ( t ) b ( t ) → 0 with i = j and η is small enough, the effect of negative feedback term is the dominant term in the increment terms. Therefore, denoting y ( t ) = β 3 -β 1 c ⊤ i ( t ) b i ( t ) and ξ ( t ) = y ( t +1) -y ( t ) -negative feedback term we can model the update rule of (Eq. 16) as follows:

<!-- formula-not-decoded -->

Recur this formula from 0 to t , we have:

<!-- formula-not-decoded -->

Denoting γ = min { b ⊤ i ( s ) b i ( s ) , c ⊤ i ( s ) c i ( s ) } for s ∈ [0 , t ] , the first term on the RHS of (Eq. (17)) can be upper bounded by (1 -2 ηβ 1 γ ) t +1 y (0) . if ξ ( s ′ ) has an exponentially decaying upper bound (it can be proved when c ⊤ i ( t ) b j ( t ) → 0 , c ⊤ i ( t ) b ( t ) → 0 with an exponential rate), the second term on the RHS of (Eq. (17)) has an exponentially decaying upper bound. Therefore, we can establish an exponential convergence rate for c ⊤ i ( t ) b i ( t ) → β 3 β 1 . The similar method can be used on c ⊤ i ( t ) b j ( t ) → 0 , c ⊤ i ( t ) b ( t ) → 0 . This technique helps answer the question " How to rigorously establish convergence? "

## 5.3 Fine-grained Induction

The exponential convergence of c ⊤ i ( t ) b i ( t ) → β 3 β 1 under the Negative Feedback Convergence framework requires the following two conditions for all i, j ∈ [ d ] with i = j :

̸

- (1) b ⊤ i ( t ) b i ( t ) and c ⊤ i ( t ) c i ( t ) dominate b ⊤ i ( t ) b j ( t ) , c ⊤ i ( t ) c j ( t ) and b ⊤ i ( t ) b ( t ) in magnitude.
- (2) c ⊤ i ( t ) b j ( t ) → 0 , c ⊤ i ( t ) b ( t ) → 0 at an exponentially decaying rate.

On the one hand, condition (1) at initialization ( t = 0 ) can be established via concentration inequalities, and critically, the preservation of Condition (1) for t &gt; 0 relies on the rapid decay of c ⊤ i ( t ) b i ( t ) , c ⊤ i ( t ) b j ( t ) , and c ⊤ i ( t ) b ( t ) (condition (2)). On the other hand, under the framework of Negative Feedback Convergence , c ⊤ i ( t ) b j ( t ) → 0 in Condition (2) also relies on Condition (1) and the rapid decay of c ⊤ i ( t ) b i ( t ) → β 3 β 1 , c ⊤ i ( t ) b ( t ) → 0 . This implies mutual dependencies among the bounds of these Vector-coupled inner products.

To handle these dependencies and establish stable bounds, we introduce the technique Fine-grained Induction : Divide the inner products into three groups: (1) Squared norms: b ⊤ i ( t ) b i ( t ) , c ⊤ i ( t ) c i ( t ) , b ⊤ ( t ) b ( t ) . (2) Target terms: c ⊤ i ( t ) b i ( t ) , c ⊤ i ( t ) b j ( t ) , c ⊤ i ( t ) b ( t ) . (3) Crossinteractions: b ⊤ i ( t ) b j ( t ) , c ⊤ i ( t ) c j ( t ) , b ⊤ i ( t ) b ( t ) . And then carefully give bounds for them with an induction.

Specifically, denoting δ ( t ) = max s ∈ [0 ,t ] { 2 √ d h log(4 d (2 d +1) /δ ) , | b ⊤ i ( s ) b j ( s ) | , | c ⊤ i ( s ) c j ( s ) | , | b ⊤ i ( s ) b ( s ) |} and γ = 1 2 d h ≤ min t ≥ 0 { b ⊤ i ( t ) b i ( t ) , c ⊤ i ( t ) c i ( t ) , b ⊤ ( t ) b ( t ) } , we establish the following three properties A ( t ) , B ( t ) , and C ( t ) simultaneously for t ≥ 0 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The initial conditions A (0) , B (0) , and C (0) are established with high probability by concentration inequalities. We also provide the following claims to establish A ( t ) , B ( t ) , and C ( t ) for t ≥ 0 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This induction answers the question " How to avoid saddle points? " because B ( t ) guarantees that C ⊤ B → β 3 β 1 I and C ⊤ b → 0 , preventing stagnation of partial diagonal entries of C ⊤ B at zero. Theorem 4.1 can be proved by substituting C ⊤ B = β 3 β 1 I , C ⊤ b = 0 , b B = b C = 0 into (Eq. (11), (14), (15))

## 6 Experimental Results

We present simulation results on synthetic data to verify our theoretical results. More experimental results can be found in Appendix E.

Figure 1: (a) Post-training visualization of matrix product C ⊤ W B ; (b) Cosine similarity evolution between w and ˜ h l = ( W ⊤ C ) [1: d, :] h ( d +1) l across recurrent steps l (after processing prompts e 1: l ); (c) Test loss versus token sequence length N . Blue curve: experimental results; orange curve: theoretical upper bound.

<!-- image -->

Experiments Setting We follow Section 3 to generate the dateset and initialize the model. Specifically, we set dimension d = 4 , d h = 80 , prompt token length N = 50 , and train the Mamba model on 3000 sequences by gradient descent. After training, we save the model and test it on 1000 new generated sequences, tracking the cosine similarity between ˜ h l ( := ( W ⊤ C ) [1: d, :] h ( d +1) l ) and w . Moreover, we vary the length of the prompt token N from 4 to 80 and compare the test loss with the theoretical upper bound. For each N , we conduct 10 independent experiments and report the averaged results. All experiments are performed on an NVIDIA A800 GPU.

Experiment Result Recalling that we denote W B = [ Bb ] , Figure 1a reveals the convergence of C ⊤ B to a diagonal matrix and C ⊤ b to 0 , confirming the theoretical induction presented in Section 5, also consistent with (Thm 4.1 (b)). Figure 1b shows that the projected hidden state ˜ h l gradually aligns with w as more prompt tokens are processed, consistent with (Thm 4.1 (a)). Figure 1c demonstrates that the experimental loss has an upper bound 3 d ( d +1) 2 N that decays linearly with N, aligning with (Thm 4.1 (c)).

## 7 Conclusion

This paper study Mamba's in-context learning mechanism, and rigorously establish its convergence and loss bound. By analysing the Vector-coupled Dynamics , we provide an exponential convergence rate with Negative Feedback Convergence technique in a Fine-grained Induction , and finally establish a O (1 /N ) loss bound. The loss bound is comparable to that of Transformer. Our theoretical results reveal the different mechanism between Transformer and Mamba on ICL, where Mamba emulates a variant of online gradient descent to perform in-context, while Transformers approximate a single step of gradient descent. Furthermore, our comparison with the S4 model demonstrates that the selection components are essential for Mamba to perform ICL.

Limitations and Social Impact Our analysis focuses on one-layer Mamba model, thus the behavior of Mamba with multi-layer or augmented with other components such as MLP is still unclear. We believe that our work will provide insight for those cases and can be used to study more data models such as nonlinear features. This paper is mainly a theoretical investigation, and we do not see an immediate social impact.

## Acknowledgements

We thank the anonymous reviewers for their insightful comments to improve the paper. Miao Zhang was partially sponsored by the National Natural Science Foundation of China under Grant 62306084 and U23B2051, Shenzhen College Stability Support Plan under Grant GXWD20231128102243003, and Shenzhen Science and Technology Program under Grant ZDSYS20230626091203008 and KJZD20230923115113026.

## References

- Ahamed, M. A. and Cheng, Q. (2024). Timemachine: A time series is worth 4 mambas for long-term forecasting. In ECAI 2024: 27th European Conference on Artificial Intelligence, 19-24 October 2024, Santiago de Compostela, Spain-Including 13th Conference on Prestigious Applications of Intelligent Systems. European Conference on Artificial Intelli , volume 392, page 1688.
- Ahn, K., Cheng, X., Daneshmand, H., and Sra, S. (2023). Transformers learn to implement preconditioned gradient descent for in-context learning. Advances in Neural Information Processing Systems , 36:45614-45650.
- Akyürek, E., Schuurmans, D., Andreas, J., Ma, T., and Zhou, D. (2023). What learning algorithm is in-context learning? investigations with linear models. In The Eleventh International Conference on Learning Representations .
- Arora, S., Cohen, N., Golowich, N., and Hu, W. (2019). A convergence analysis of gradient descent for deep linear neural networks. In International Conference on Learning Representations .
- Bai, Y., Chen, F., Wang, H., Xiong, C., and Mei, S. (2023). Transformers as statisticians: Provable in-context learning with in-context algorithm selection. Advances in neural information processing systems , 36:57125-57211.
- Behrouz, A., Li, Z., Kacham, P., Daliri, M., Deng, Y., Zhong, P., Razaviyayn, M., and Mirrokni, V. (2025a). Atlas: Learning to optimally memorize the context at test time. arXiv preprint arXiv:2505.23735 .

- Behrouz, A., Razaviyayn, M., Zhong, P., and Mirrokni, V. (2025b). It's all connected: A journey through test-time memorization, attentional bias, retention, and online optimization. arXiv preprint arXiv:2504.13173 .
- Bondaschi, M., Rajaraman, N., Wei, X., Ramchandran, K., Pascanu, R., Gulcehre, C., Gastpar, M., and Makkuva, A. V. (2025). From markov to laplace: How mamba in-context learns markov chains. arXiv preprint arXiv:2502.10178 .
- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. (2020). Language models are few-shot learners. Advances in neural information processing systems , 33:1877-1901.
- Bu, D., Huang, W., Han, A., Nitanda, A., Suzuki, T., Zhang, Q., and Wong, H.-S. (2024). Provably transformers harness multi-concept word semantics for efficient in-context learning. Advances in Neural Information Processing Systems , 37:63342-63405.
- Bu, D., Huang, W., Han, A., Nitanda, A., Zhang, Q., Wong, H.-S., and Suzuki, T. (2025). Provable in-context vector arithmetic via retrieving task concepts. In Forty-second International Conference on Machine Learning .
- Chen, Y., Li, X., Liang, Y., Shi, Z., and Song, Z. (2025). The computational limits of state-space models and mamba via the lens of circuit complexity. In The Second Conference on Parsimony and Learning (Proceedings Track) .
- Cirone, N. M., Orvieto, A., Walker, B., Salvi, C., and Lyons, T. (2024). Theoretical foundations of deep selective state-space models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems .
- Dao, T. and Gu, A. (2024). Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. In International Conference on Machine Learning (ICML) .
- Du, S. and Hu, W. (2019). Width provably matters in optimization for deep linear neural networks. In International Conference on Machine Learning , pages 1655-1664. PMLR.
- Garg, S., Tsipras, D., Liang, P. S., and Valiant, G. (2022). What can transformers learn in-context? a case study of simple function classes. Advances in Neural Information Processing Systems , 35:30583-30598.
- Gatmiry, K., Saunshi, N., Reddi, S., Jegelka, S., and Kumar, S. (2024). Can looped transformers learn to implement multi-step gradient descent for in-context learning? In Proceedings of the 41st International Conference on Machine Learning , pages 15130-15152.
- Giannou, A., Yang, L., Wang, T., Papailiopoulos, D., and Lee, J. D. (2025). How well can transformers emulate in-context newton's method? In The 28th International Conference on Artificial Intelligence and Statistics .
- Grazzi, R., Siems, J. N., Schrodi, S., Brox, T., and Hutter, F. (2024). Is mamba capable of in-context learning? In Proceedings of the Third International Conference on Automated Machine Learning , volume 256 of Proceedings of Machine Learning Research , pages 1/1-26. PMLR.
- Gu, A. and Dao, T. (2024). Mamba: Linear-time sequence modeling with selective state spaces. In First Conference on Language Modeling .
- Gu, A., Goel, K., and Re, C. (2022). Efficiently modeling long sequences with structured state spaces. In International Conference on Learning Representations .
- Huang, Y., Cheng, Y., and Liang, Y. (2023). In-context convergence of transformers. arXiv preprint arXiv:2310.05249 .
- Lee, I., Jiang, N., and Berg-Kirkpatrick, T. (2024). Is attention required for icl? exploring the relationship between model architecture and in-context learning ability. In The Twelfth International Conference on Learning Representations .

- Li, H., Lu, S., Cui, X., Chen, P.-Y., and Wang, M. (2025a). Understanding mamba in in-context learning with outliers: A theoretical generalization analysis. In High-dimensional Learning Dynamics .
- Li, K., Chen, G., Yang, R., and Hu, X. (2024a). Spmamba: State-space model is all you need in speech separation. arXiv preprint arXiv:2404.02063 .
- Li, K., Li, X., Wang, Y., He, Y., Wang, Y., Wang, L., and Qiao, Y. (2024b). Videomamba: State space model for efficient video understanding. In European Conference on Computer Vision , pages 237-255. Springer.
- Li, Y., Tarzanagh, D. A., Rawat, A. S., Fazel, M., and Oymak, S. (2025b). Gating is weighting: Understanding gated linear attention through in-context learning. arXiv preprint arXiv:2504.04308 .
- Lin, L., Bai, Y ., and Mei, S. (2024). Transformers as decision makers: Provable in-context reinforcement learning via supervised pretraining. In The Twelfth International Conference on Learning Representations .
- Liu, Y., Tian, Y., Zhao, Y., Yu, H., Xie, L., Wang, Y., Ye, Q., Jiao, J., and Liu, Y . (2024). Vmamba: Visual state space model. Advances in neural information processing systems , 37:103031-103063.
- Mahankali, A. V., Hashimoto, T., and Ma, T. (2024). One step of gradient descent is provably the optimal in-context learner with one layer of linear self-attention. In The Twelfth International Conference on Learning Representations .
- Nishikawa, N. and Suzuki, T. (2025). State space models are provably comparable to transformers in dynamic token selection. In The Thirteenth International Conference on Learning Representations .
- Park, J., Park, J., Xiong, Z., Lee, N., Cho, J., Oymak, S., Lee, K., and Papailiopoulos, D. (2024). Can mamba learn how to learn? a comparative study on in-context learning tasks. In Proceedings of the 41st International Conference on Machine Learning , pages 39793-39812.
- Patro, B. N. and Agneeswaran, V. S. (2024). Mamba-360: Survey of state space models as transformer alternative for long sequence modelling: Methods, applications, and challenges. arXiv preprint arXiv:2404.16112 .
- Sander, M. E., Giryes, R., Suzuki, T., Blondel, M., and Peyré, G. (2024). How do transformers perform in-context autoregressive learning? In Proceedings of the 41st International Conference on Machine Learning , pages 43235-43254.
- Shen, W., Zhou, R., Yang, J., and Shen, C. (2024). On the training convergence of transformers for in-context classification of gaussian mixtures. arXiv preprint arXiv:2410.11778 .
- Sushma, N. M., Tian, Y., Mestha, H., Colombo, N., Kappel, D., and Subramoney, A. (2024). State-space models can learn in-context by gradient descent. arXiv preprint arXiv:2410.11687 .
- Tong, W. L. and Pehlevan, C. (2024). Mlps learn in-context on regression and classification tasks. arXiv preprint arXiv:2405.15618 .
- Vankadara, L. C., Xu, J., Haas, M., and Cevher, V. (2024). On feature learning in structured state space models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems .
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems , 30.
- Von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A., and Vladymyrov, M. (2023). Transformers learn in-context by gradient descent. In International Conference on Machine Learning , pages 35151-35174. PMLR.
- Wu, J., Zou, D., Chen, Z., Braverman, V., Gu, Q., and Bartlett, P. (2024). How many pretraining tasks are needed for in-context learning of linear regression? In The Twelfth International Conference on Learning Representations .

- Yang, S., Kautz, J., and Hatamizadeh, A. (2025). Gated delta networks: Improving mamba2 with delta rule. In The Thirteenth International Conference on Learning Representations .
- Yang, S., Wang, B., Shen, Y., Panda, R., and Kim, Y. (2024). Gated linear attention transformers with hardware-efficient training. In International Conference on Machine Learning , pages 56501-56523. PMLR.
- Zhang, R., Frei, S., and Bartlett, P. L. (2024). Trained transformers learn linear models in-context. Journal of Machine Learning Research , 25(49):1-55.
- Zhang, Y., Singh, A. K., Latham, P. E., and Saxe, A. (2025). Training dynamics of in-context learning in linear attention. arXiv preprint arXiv:2501.16265 .
- Zheng, C., Huang, W., Wang, R., Wu, G., Zhu, J., and Li, C. (2024). On mesa-optimization in autoregressively trained transformers: Emergence and capability. Advances in Neural Information Processing Systems , 37:49081-49129.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract reflects the paper's scope, and the introduction reflects the paper's contributions.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss our limitation in the conclusion Section.

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

Justification: We provide the assumptions in Assumption 4.1. In the main paper, we provide a proof sketch (Section 5). The detailed proofs are in the appendix.

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

Justification: The experiments follows our problem setup and assumption. We also provide the experiments setting in this paper, and the codes are in the supplementary material.

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

Justification: The codes are in the supplementary material. We also provide a readme file. Guidelines:

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

Justification: The experiments follows our problem setup and assumption. We also provide hyperparameters in the experiments setting.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We plot the 1-sigma error bar. Figure 1b, the error bar for experimental loss can be found in Figure 2c.

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

Justification: We report the computer resources in experiments setting. All experiments are performed on an NVIDIA A800 GPU within hours.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: This work focuses on theoretical study of Mamba's in-context learning. All the data is synthesized. We see no ethical or potential harms of our work. We will not violate the Code Of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discuss the societal impacts in Conclusion Section. We do not see an immediate social impact.

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

Justification: This paper is mainly a theoretical work. It poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: We perform experiments on synthetic data. It does not use existing assets.

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

Justification: We perform experiments on synthetic data. No new assets are introduced in the paper.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing or research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve crowdsourcing or research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: This paper is mainly a theoretical work. The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Contents

| A Basic Calculations   | A Basic Calculations            | A Basic Calculations            |   22 |
|------------------------|---------------------------------|---------------------------------|------|
|                        | A.1                             | Data Statistics . . . .         |   22 |
|                        | A.2                             | Output, Loss, Gradient          |   24 |
|                        | A.3                             | Training Dynamics .             |   25 |
| B                      | Proof of Theorem 4.1            | Proof of Theorem 4.1            |   27 |
| C                      | Complete Proof                  | Complete Proof                  |   31 |
|                        | C.1                             | Proof of Lemma A.2              |   32 |
|                        | C.2                             | Proof of Lemma A.3              |   33 |
|                        | C.3                             | Proof of Lemma A.4              |   36 |
|                        | C.4                             | Proof of claim B.1 . .          |   44 |
|                        | C.5                             | Proof of claim B.2 . .          |   50 |
|                        | C.6                             | Proof of claim B.3 . .          |   57 |
|                        | C.7                             | Bounds of η 2 terms .           |   63 |
| D                      | Discussion                      | Discussion                      |   66 |
| E                      | Additional Experimental Results | Additional Experimental Results |   67 |

## Appendix

Table 1: Key notations

| Symbols                                                                                                                             | Definitions                                                                                                                                                                                                                                                                                                 |
|-------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| x i , x q , w , y i , y q                                                                                                           | x i , x q , w are i.i.d. sampled from Gaussian distribution N (0 , I d ) . y i = w ⊤ x i , y q = w ⊤ x q .                                                                                                                                                                                                  |
| ¯ b i ( t ) , ¯ c i ( t ) , ¯ b ( t )                                                                                               | ¯ b i ( t ) = 1 η ( b i ( t +1) - b i ( t ) ) , ¯ c i ( t ) = 1 η ( c i ( t +1) - c i ( t ) ) , ¯ b ( t ) = 1 η ( b ( t +1) - b ( t ) ) .                                                                                                                                                                   |
| B , C , b , c , b i , c i                                                                                                           | Decompose the matrices W B , W C into colums of vectors: W B = [ Bb ] = [ b 1 , . . . , b d b ] , W C = [ Cc ] = [ c 1 , . . . , c d c ] where W B , W C ∈ R d h × ( d +1) , B , C ∈ R d h × d , b , c , b i , c i ∈ R d h × 1                                                                              |
| b i ( t ) ⊤ b j ( t ) , c i ( t ) ⊤ c j ( t ) , b ( t ) ⊤ b ( t ) c i ( t ) ⊤ b j ( t ) , b i ( t ) ⊤ b ( t ) , c i ( t ) ⊤ b ( t ) | inner product of the vectors b , b i , c i with i, j ∈ [1 ,d ] . e.g. b i ( t ) ⊤ b j ( t ) is the inner product of b i ( t ) and b i ( t ) .                                                                                                                                                               |
| α                                                                                                                                   | A factor, α := exp( - ∆ l ) = exp(( - ln 2) /N ) .                                                                                                                                                                                                                                                          |
| β 1 ,β 2 ,β 3                                                                                                                       | The factors appearing in the gradient equation. Specifically, β 1 = ( α 2 ( 1 - α N ) 2 + ( d +1) α 2 (1 - α ) ( 1 - α 2 N ) (1+ α ) ) , β 2 = ( d 2 α 2 ( 1 - α N ) 2 + (2 d 2 +6 d ) α 2 (1 - α ) ( 1 - α 2 N ) (1+ α ) ) , β 3 = α ( 1 - α N )                                                           |
| γ                                                                                                                                   | The lower bound of squared norms b ⊤ i ( t ) b i ( t ) , c ⊤ i ( t ) c i ( t ) , and b ⊤ ( t ) b ( t ) . Specifically, γ = 1 2 d h .                                                                                                                                                                        |
| δ ( T )                                                                                                                             | The upper bound of cross-interactions: b ⊤ i ( t ) b j ( t ) , c ⊤ i ( t ) c j ( t ) , and b ⊤ i ( t ) b ( t ) . Specifically, δ ( t ) = max s ∈ [0 ,t ] { 2 √ d h log(4 d (2 d +1) /δ ) , &#124; b ⊤ i ( s ) b j ( s ) &#124; , &#124; c ⊤ i ( s ) c j ( s ) &#124; , &#124; b ⊤ i ( s ) b ( s ) &#124;} . |

## A Basic Calculations

This Section provide the data statistics related to gaussian distribution, and compute the expressions for the output, loss, gradient, training dynamics (particularly Vector-coupled Dynamics ) of the Mamba model. Section B presents the Fine-grained Induction with Negative Feedback Convergence technique, and finally establish the results for Theorem 4.1. Section C details the complete proofs for Section A and Section B. In Section D, we discuss about orthogonal initialization and compare our framework with other techniques. In Section E, we give more experimental results.

## A.1 Data Statistics

Lemma A.1 (Concentration Inequalities) Let b i (0) be the i-th colum of B (0) , c i (0) be the i-th colum of C (0) , and suppose that δ &gt; 0 and d h = Ω(log(4(2 d +1) /δ )) , with probability at least 1 -δ , we have:

<!-- formula-not-decoded -->

̸

for i, j ∈ [ d ] , i = j .

Proof of Lemma A.1. By Bernstein's inequality, with probability at least 1 -δ/ 2(2 d +1) we have

<!-- formula-not-decoded -->

Therefore, as long as d h = Ω(log(4(2 d +1) /δ )) , we have 3 d h / 4 ≤ b i (0) ⊤ b i (0) ≤ 5 d h / 4 . Similarly, we have

̸

<!-- formula-not-decoded -->

For i, j ∈ [ d ] , i = j , By Bernstein's inequality, with probability at least 1 -δ/ 2 d (2 d +1) , we have

<!-- formula-not-decoded -->

We can apply a union bound to complete the proof.

Lemma A.2 If vectors x and w are iid generated from N (0 , I d ) , y = x ⊤ w we have the following expectations:

<!-- formula-not-decoded -->

The proof of lemma A.2 is in Section C.1.

Lemma A.3 If vectors x i and w are iid generated from N (0 , I d ) , y = x ⊤ i w we have the following expectations:

<!-- formula-not-decoded -->

The proof of lemma A.3 is in Section C.2.

## A.2 Output, Loss, Gradient

This section we derive the output of Mamba given sequence { e 1: N , e q } , and establishe the loss function formulation with its gradient expression.

Linear Recurrence. To start with, we show how the hidden states update when receiving token e l = ( x ⊤ l , y l ) ⊤ . By (Eq. (5)) and Assumption 4.1(2), we have ∆ l = softplus(ln(exp((ln 2) /N ) -1)) = (ln 2) /N . Combining it with (Eq. (1)(2)) and get:

<!-- formula-not-decoded -->

where α := exp( -∆ l ) = exp(( -ln 2) /N ) , the second equality is by discretization rule (2), the third equality is by Assumption 4.1(2) and exp ( -∆ l I ) = exp ( -∆ l ) I . (Eq. (18)) is similar to theorem 1 in Gu and Dao (2024)

Prediction Output. We next derive the expression of ˆ y q . Based on (Eq.(11)), the hidden state after receiving the first l context prompts e 1: l is given by:

<!-- formula-not-decoded -->

Receiving the query token e q = ( x ⊤ q , 0) ⊤ , we have:

<!-- formula-not-decoded -->

Finally, the prediction output is as follows

<!-- formula-not-decoded -->

To handle W C e q and W B e N -i , we further denote W B = [ Bb ] and W C = [ Cc ] , where B , C ∈ R d h × d , b , c ∈ R d h × 1 . Then we write another form of (Eq. (21)):

<!-- formula-not-decoded -->

The loss becomes:

<!-- formula-not-decoded -->

The following lemma provides the gradient of B , C , b , b B , b C with respect to loss (Eq. (23)).

Lemma A.4 (Gradient) The gradient of trainable parameters θ ′ = { B , C , b , b B , b C } with respect to loss (Eq. (23) ) are as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, we denote β 1 = ( α 2 ( 1 -α N ) 2 + ( d +1) α 2 (1 -α ) ( 1 -α 2 N ) (1+ α ) ) , β 2 = ( d 2 α 2 ( 1 -α N ) 2 + (2 d 2 +6 d ) α 2 (1 -α ) ( 1 -α 2 N ) (1+ α ) ) , β 3 = α ( 1 -α N ) for simplicity. The proof of lemma A.4 is in Section C.3.

## A.3 Training Dynamics

With the gradient in lemma A.4, we further provide the update rule for Mamba's parameters and the Vector-coupled Dynamics .

Using gradient descent algorithm θ ′ ( t +1) = θ ′ ( t ) -η ∇ θ ′ L ( θ ( t )) with training rate η , we have the following update rule base on lemma A.4.

Lemma A.5 (Update Rule, restatement of lemma 5.1) Let η be the learning rate and we use gradient descent to update the weights W B , W C , b B , b C , for t ≥ 0 we have

<!-- formula-not-decoded -->

We decompose B and C as B = [ b 1 . . . b d ] , C = [ c 1 . . . c d ] , and provide the update rule for b i , c i and b with i ∈ [1 : d ] as the following lemma.

Lemma A.6 (Vectors Update Rule, restatement of lemma 5.2) Let η be the learning rate and we use gradient descent to update the weights W B , W C , b B , b C , for i ∈ [ d ] , t ≥ 0 we have

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Here, we denote η ¯ b i ( t ) = b i ( t +1) -b i ( t ) , η ¯ c i ( t ) = c i ( t +1) -c i ( t ) , and η ¯ b ( t ) = b ( t +1) -b ( t ) for simplicity.

Next, we provide the dynamics for the inner products of these vectors.

Lemma A.7 (Vector-coupled Dynamics) Let η be the learning rate and we use gradient descent to update the weights W B , W C , b B , b C , we have

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

̸

̸

̸

̸

<!-- formula-not-decoded -->

Lemma A.7 is derive by calculating the inner products of the vectors update rule in lemma A.6. For example, b ⊤ ( t +1) b ( t +1) is derived as follow:

<!-- formula-not-decoded -->

The other equations are similar to it.

## B Proof of Theorem 4.1

In this section, we present the framework of Fine-grained Induction , and establish the results of Theorem 4.1 after convergence.

̸

̸

̸

̸

̸

Fine-gained Induction Specifically, denoting δ ( t ) = max s ∈ [0 ,t ] {| b ⊤ i ( s ) b j ( s ) | , | c ⊤ i ( s ) c j ( s ) | , | b ⊤ i ( s ) b ( s ) |} and γ = min t ≥ 0 { b ⊤ i ( t ) b i ( t ) , c ⊤ i ( t ) c i ( t ) , b ⊤ ( t ) b ( t ) } , we establish the following three properties A ( t ) , B ( t ) , and C ( t ) simultaneously for t ≥ 0 :

- A ( t ) :
- B ( t ) :
- C ( t ) :

<!-- formula-not-decoded -->

̸

Here, i, j ∈ [1 , d ] , i = j . The initial conditions A (0) , B (0) , and C (0) are established with high probability by concentration inequalities (lemma A.1). We also provide the following claims to establish A ( t ) , B ( t ) , and C ( t ) for t ≥ 0 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark. Property A ( t ) establishes the stability of quadratic norms:

<!-- formula-not-decoded -->

This norm lower bound induces two critical effects:

1. Convergence Rate: As we can see in property B ( t ) , The upper bound of c ⊤ i ( t ) b i ( t ) , c ⊤ i ( t ) b j ( t ) and c ⊤ i ( t ) b ( t ) is related to γ (lower bound of the squared norms), thus the stability of quadratic norms ensure a stable rapid convergence rate for property B ( t ) .
2. Saddle Point Avoidance: The strict positivity ( &gt; 0 ) of | b i | 2 and | c i | 2 prevents the dynamics collapse to undesirable solutions b i = c i = 0 , which would permanently make c ⊤ i b i = 0 (saddle points).

Property B ( t ) establishes a rapid exponential convergence rate:

<!-- formula-not-decoded -->

The rapid convergence rate ensures that the variations of Squared norms (in property A ( t ) ) and Crossinteractions (in property C ( t ) ) remain bounded, thereby establishing their constraints. For example, at initialization, b ⊤ i (0) b i (0) is bounded by 3 d h / 4 ≤ b ⊤ i (0) b i (0) ≤ 5 d h / 4 . Further, thanks to the exponential convergence rate in property B ( t ) , we can prove that | b ⊤ i ( t ) b i ( t ) -b ⊤ i (0) b i (0) | ≤ d h / 4 , and therefore d h / 2 ≤ b ⊤ i ( t ) b i ( t ) , c ⊤ i ( t ) c i ( t ) , b ⊤ ( t ) b ( t ) ≤ 3 d h / 2 ≤ 2 d h .

Property C ( t ) establishes the upper bound for the Cross-interactions . As we discuss in section 5.2, if the Squared norms ( c ⊤ i ( t ) b i ( t ) , c ⊤ i ( t ) b j ( t ) and c ⊤ i ( t ) b ( t ) ) are larger enough than the Crossinteractions ( b ⊤ i ( t ) b j ( t ) , c ⊤ i ( t ) c j ( t ) , and b ⊤ i ( t ) b ( t ) ), we can make use of the negative feedback term to establish an exponential convergence rate. Thus property C ( t ) is also important.

The proof of claim B.1, claim B.2, and claim B.3 are in section C.4, section C.5, and section C.6 respectively.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Theorem 4.1 After convergence ( t → 0 ), we will have C ⊤ B = β 3 β 1 I , C ⊤ b = 0 (by property B ( t ) ), and b B ( t ) = b C ( t ) = 0 (by lemma A.5). We will restate some equality for ease of reference.

Linear Recurrence (restatement of (Eq. (18)))

<!-- formula-not-decoded -->

Prediction Output (restatement of (Eq. (22)))

<!-- formula-not-decoded -->

Loss (restatement of (Eq. (23)))

<!-- formula-not-decoded -->

Based on (Eq. (24)), we have

<!-- formula-not-decoded -->

where the second equality is by selection rule B l = W B e l + b B (Eq. (3)), and W B = [ Bb ] , e l = ( x ⊤ l , y l ) ⊤ . The third equality is by b B ( t ) = 0 . The fourth equality is by C ⊤ B = β 3 β 1 I and C ⊤ b = 0 . (Eq. (27)) establish the first equation (Thm 4.1 (a)) of the Theorem.

Based on (Eq. (25)), we have

<!-- formula-not-decoded -->

where the second equality is by b B ( t ) = b C ( t ) = 0 . The fourth equality is by C ⊤ B = β 3 β 1 I and C ⊤ b = 0 . (Eq. (28)) establish the second equation (Thm 4.1 (b)) of the Theorem.

Based on (Eq. (25)), we have

<!-- formula-not-decoded -->

We compute terms ♠ and ♣ as follows:

<!-- formula-not-decoded -->

For the fourth equality, E x N -i , x N -j , w [ ∑ N -1 i =0 ∑ N -1 j =0 α i + j +2 y N -i y N -j x N -i x ⊤ N -j ] = ( α 2 ( 1 -α N ) 2 (1 -α ) 2 + ( d +1) α 2 ( 1 -α 2 N ) (1 -α )(1+ α ) ) · I by lemma A.3. The last equality is by β 1 = ( α 2 ( 1 -α N ) 2 + ( d +1) α 2 (1 -α ) ( 1 -α 2 N ) (1+ α ) )

<!-- formula-not-decoded -->

For the third equality, E x N -i , w [ ∑ N -1 i =0 α i +1 y N -i x N -i w ⊤ ] = α ( 1 -α N 1 -α ) I by lemma A.3. The last equality is by β 3 = α ( 1 -α N ) .

Substituting ♠ and ♣ into (Eq. (29)) and get:

<!-- formula-not-decoded -->

Recall β 1 = ( α 2 ( 1 -α N ) 2 + ( d +1) α 2 (1 -α ) ( 1 -α 2 N ) (1+ α ) ) , β 3 = α ( 1 -α N ) and α = exp(( -ln 2) /N ) . For the inequality, 1 -α = 1 -exp(( -ln 2) /N ) ≤ ln 2 N ≤ 1 N , β 1 ≥ α 2 ( 1 -α N ) 2 = 1 4 α 2 , 1 -α 2 N = 1 -1 4 = 3 4 , thus d ( d +1)(1 -α ) ≤ d ( d +1) N , α 2 β 1 ≤ 4 , ( 1 -α 2 N ) 2(1+ α ) ≤ 3 8(1+ α ) ≤ 3 8 . (Eq. (30)) establish the third equation (Thm 4.1 (c)) of the Theorem.

## C Complete Proof

This section presents the complete proof for the above results. To begin with, we provide the exact assumptions for N , η and d h as part of Assumption 4.1.

## Assumption

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that under the assumption of N ≥ max { 2 ln 2 ln 6 -ln 5 , 3( d +1)ln2 2 } and combining α = exp(( -ln 2) /N ) , we have the following:

<!-- formula-not-decoded -->

These condition will be use to prove some bounds.

## C.1 Proof of Lemma A.2

Lemma C.1 (restatement of lemma A.2) If vectors x and w are iid generated from N (0 , I d ) , y = x ⊤ w we have the following expectations:

<!-- formula-not-decoded -->

Proof. For ( i, j ) -th element of E [ xx ⊤ ww ⊤ xx ⊤ ] , we have:

<!-- formula-not-decoded -->

According to the distribution of w , we have E [ w [ k ] w [ l ] ] = δ kl , where δ kl is the Kronecker delta defined as:

̸

<!-- formula-not-decoded -->

By Isserlis Theorem, we have:

<!-- formula-not-decoded -->

Then we have:

<!-- formula-not-decoded -->

Then we have:

<!-- formula-not-decoded -->

̸

̸

̸

<!-- formula-not-decoded -->

## C.2 Proof of Lemma A.3

Lemma C.2 (restatement of lemma A.3) If vectors x i and w are iid generated from N (0 , I d ) , y = x ⊤ i w we have the following expectations:

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Notice that w appears three (odd) times in the second equality, if we define a function g ( w ) = x N -i x ⊤ N -i wx ⊤ N -j wx ⊤ N -j w , we can see that g ( -w ) = -g ( w ) , and further E w [ g ( w ) ] = 0 . Therefore, the above expectation equals to 0 . We will use the similar property in some of the following equations.

̸

̸

<!-- formula-not-decoded -->

where the second equality is by E [ y 2 N -i ] = E [ y 2 N -j ] = d and E [ y 4 N -i ] = 3 d ( d +2) (lemma A.2).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Notice that w appears three (odd) times in the second equality, thus this expectation equals to 0 .

̸

<!-- formula-not-decoded -->

Notice that x N -i appears three (odd) times in E [ x N -i x ⊤ N -i wx ⊤ N -i w ] , and x ⊤ N -j appears once (odd) in E [ wx ⊤ N -j w ] , thus this expectation equals to 0 .

̸

## C.3 Proof of Lemma A.4

Lemma C.3 (restatement of lemma A.4) The gradient of trainable parameters θ ′ = { B , C , b , b B , b C } with respect to loss (Eq. (23) ) are as follows:

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Notice that w appears three (odd) times in the second equality, thus this expectation equals to 0 .

̸

<!-- formula-not-decoded -->

̸

Notice that x ⊤ N -i appears once (odd) in E [ x ⊤ N -i wx ⊤ N -j w ] where i = j , thus ∑ i = j α i + j +2 E [ y N -i y N -j ] = 0 . Moreover, E [ y 2 N -i ] = d by lemma A.2.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of lemma C.3. Recalling the loss:

<!-- formula-not-decoded -->

We will compute the gradient of { B , C , b , b B , b C } with respect to L ( θ ) . Some expectation calculation are detailed in Section C.3.1.

<!-- formula-not-decoded -->

It is clear that E [ x q ] = 0 . Thus, if b C = 0 , then ∇ b C L ( θ ) = 0 . Notice that we assume b C (0) = 0 at initialization, so by induction, b C ( t ) = 0 and ∇ b C L ( θ ( t )) = 0 for t ≥ 0 . We will consider b C = 0 when computing other gradients.

<!-- formula-not-decoded -->

+

The last equality follows from lemma C.4 where we have: E [ x ⊤ q C ⊤ ( ∑ N -1 i =0 α i +1 y N -i ( Bx N -i y N -i b + b B ) ) · Cx q · ∑ N -1 i =0 α i +1 y N -i ] = dα 2 ( 1 -α 2 N ) (1 -α )(1+ α ) CC ⊤ b B , and E [ w ⊤ x q · Cx q · ∑ N -1 i =0 α i +1 y N -i ] = 0 . Similar to b C , notice that b B is initialized as 0 , thus by induction, b B ( t ) = 0 and ∇ b B L ( θ ( t )) = 0 for t ≥ 0 . We will consider b B = b C = 0 when computing other gradients.

<!-- formula-not-decoded -->

The last equality follows from lemma C.4, where we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The last equality follows from lemma C.4, where we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The last equality follows from lemma C.4, where we have:

<!-- formula-not-decoded -->

## C.3.1 Auxiliary Lemma for Lemma A.4

Lemma C.4 If vectors x i , x q and w are iid generated from N (0 , I d ) , y = x ⊤ i w we have the following expectations:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of lemma C.4. We will use the results of lemma A.3 to prove the above equation.

<!-- formula-not-decoded -->

The last equality follows from lemma A.3, where we have: E [ ∑ N -1 i =0 ∑ N -1 j =0 α i + j +2 y N -i y N -j x N -i x ⊤ N -j ] = ( α 2 ( 1 -α N ) 2 (1 -α ) 2 + ( d +1) α 2 ( 1 -α 2 N ) (1 -α )(1+ α ) ) I , E [ ∑ N -1 i =0 ∑ N -1 j =0 α i + j +2 y N -i y 2 N -j x N -i ] = 0 and E [ ∑ N -1 i =0 ∑ N -1 j =0 α i + j +2 y 2 N -i y 2 N -j ] = ( d 2 α 2 ( 1 -α N ) 2 (1 -α ) 2 + (2 d 2 +6 d ) α 2 ( 1 -α 2 N ) (1 -α )(1+ α ) ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The last equality follows from lemma A.3, where we have: E [ ∑ N -1 i =0 α i +1 y N -i x N -i w ⊤ ] = α ( 1 -α N 1 -α ) I , E [ ∑ N -1 i =0 α i +1 y 2 N -i w ] = 0 , and E [ x q x ⊤ q ] = I .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The last equality follows from lemma A.3, where we have: E [ ∑ N -1 i =0 ∑ N -1 j =0 α i + j +2 y N -i y N -j x N -i x ⊤ N -j ] = ( α 2 ( 1 -α N ) 2 (1 -α ) 2 + ( d +1) α 2 ( 1 -α 2 N ) (1 -α )(1+ α ) ) I , and E [ x q x ⊤ q ] = I .

<!-- formula-not-decoded -->

The last equality follows from lemma A.3, where we have: E [ ∑ N -1 i =0 α i +1 y N -i x N -i w ⊤ ] = α ( 1 -α N 1 -α ) I , and E [ x q x ⊤ q ] = I .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The last equality follows from lemma A.3, where we have: E [ ∑ N -1 i =0 ∑ N -1 j =0 α i + j +2 y N -i y 2 N -j x N -i ] = 0 , E [ ∑ N -1 i =0 ∑ N -1 j =0 α i + j +2 y 2 N -i y 2 N -j ] = ( d 2 α 2 ( 1 -α N ) 2 (1 -α ) 2 + (2 d 2 +6 d ) α 2 ( 1 -α 2 N ) (1 -α )(1+ α ) ) , and E [ x q x ⊤ q ] = I .

<!-- formula-not-decoded -->

The last equality follows from lemma A.3, where we have E [ ∑ N -1 i =0 α i +1 y 2 N -i w ] = 0 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The last equality follows from lemma A.3, where we have E [ ∑ N -1 i =0 α i +1 y N -i w ] = 0 .

## C.4 Proof of claim B.1

This Section presents the bounds for terms b ⊤ i ( T +1) b i ( T +1) , c ⊤ i ( Tt +1) c i ( T +1) and b ⊤ ( T + 1) b ( T +1) , establishing the property A ( T +1) .

Recurring the Vector-coupled Dynamics equations of b ⊤ i ( t +1) b i ( t +1) , c ⊤ i ( t +1) c i ( t +1) and b ⊤ ( t +1) b ( t +1) in lemma A.7, we have:

̸

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

<!-- formula-not-decoded -->

̸

To bound terms I - VI, we will use some inequalities from property B ( t ) and lemma C.7 as following with i, j, k ∈ [1 , d ] , i = j :

<!-- formula-not-decoded -->

Next we begin bounding terms I - VI.

Bound of term I: By ∣ ∣ ∣ β 3 -β 1 c ⊤ i ( s ) b i ( s ) ∣ ∣ ∣ ≤ δ ( s ) exp( -ηβ 1 γs ) , we have:

<!-- formula-not-decoded -->

The third inequality is by δ ( s ) ≥ 2 √ d h log(4 d (2 d +1) /δ ) ≥ 2 α = 2 α = 2exp(( -ln 2) /N ) . For the last inequality, as long as N ≥ 2 ln 2 ln 6 -ln 5 , we have 5 α 2 ≤ 6 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The second inequality is due to exp( -ηβ 1 γs ) is monotone decreasing.

## Bound of term II:

<!-- formula-not-decoded -->

The second inequality is due to exp( -2 ηβ 1 γs ) is monotone decreasing.

## Bound of term III:

<!-- formula-not-decoded -->

The second inequality is due to exp( -ηβ 1 γs ) is monotone decreasing.

## Bound of term IV:

<!-- formula-not-decoded -->

The second inequality is due to exp( -2 ηβ 2 γs ) , exp( -η ( β 1 + β 2 ) γs ) and exp( -2 ηβ 1 γs ) are monotone decreasing. The last inequality is by exp( η ( β 1 + β 2 ) γ ) ( β 1 + β 2 ) ≤ 2 and exp(2 ηβ 1 γ ) 2 β 1 β 2 ≤ 7 since exp(2 ηβ 1 γ ) ≤ exp( η ( β 1 + β 2 ) γ ) ≤ 2 and β 1 + β 2 ≥ 1 , β 1 β 2 ≥ 1 7 .

## Bound of term V:

<!-- formula-not-decoded -->

The second inequality is due to exp( -ηβ 2 γs ) and exp( -ηβ 1 γs ) are monotone decreasing.

## Bound of term VI:

<!-- formula-not-decoded -->

The second inequality is due to exp( -ηβ γs ) and exp( -ηβ γs ) are monotone decreasing.

2 1 We next use the bounds of I - VI to bound b ⊤ i ( T +1) b i ( T +1) , c ⊤ i ( T +1) c i ( T +1) and b ⊤ ( T 1) b ( T +1) .

Lower bound of b ⊤ ( T +1) b ( T

<!-- formula-not-decoded -->

̸

+

<!-- formula-not-decoded -->

The third inequality is by δ max = 3 √ d h log(4 d (2 d +1) /δ ) , exp( ηβ 1 γ ) ≤ exp(2 ηβ 1 γ ) ≤ 2 and γ = 1 2 d h . The last inequality follows from d h = ˜ Ω( d 2 ) ≥ ( 1728 log(4 d (2 d + 1) /δ ) + 576( d -1) β 1 log(4 d (2 d +1) /δ ) ) /β 1 .

Upper bound of b ⊤ ( T +1) b ( T

<!-- formula-not-decoded -->

̸

The third inequality is by δ max = 3 √ d h log(4 d (2 d +1) /δ ) , exp( ηβ 1 γ ) ≤ 2 and γ = 1 2 d h . The last inequality follows from

<!-- formula-not-decoded -->

Lower bound of c ⊤ i ( T +1) c i ( T +1)

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

The second inequality is by δ max = 3 √ d h log(4 d (2 d +1) /δ ) , exp( ηβ 1 γ ) ≤ exp(2 ηβ 1 γ ) ≤ 2 and γ = 1 2 d h . The last inequality follows from d h = ˜ Ω( d 2 ) ≥ ( 1728 log(4 d (2 d + 1) /δ ) + (576 d + 1872) β 1 log(4 d (2 d +1) /δ ) ) /β 1 .

Upper bound of ⊤

<!-- formula-not-decoded -->

̸

The second inequality is by δ max = 3 √ d h log(4 d (2 d +1) /δ ) , exp( ηβ 1 γ ) ≤ exp(2 ηβ 1 γ ) ≤ 2 and γ = 1 2 d h . The last inequality follows from

<!-- formula-not-decoded -->

Lower bound of b ⊤ ( T +1) b ( T +1)

<!-- formula-not-decoded -->

The third inequality is by δ max = 3 √ d h log(4 d (2 d +1) /δ ) and γ = 1 2 d h . The last inequality follows from d h = ˜ Ω( d 2 ) ≥ 2448 d log(4 d (2 d +1) /δ ) .

Upper bound of b ⊤ ( T +1) b ( T +1)

<!-- formula-not-decoded -->

The second inequality is by exp( ηβ 1 γ ) ≤ exp( ηβ 2 γ ) ≤ 2 The third inequality is by δ max = 3 √ d h log(4 d (2 d +1) /δ ) and γ = 1 2 d h . The last inequality follows from

d

h

=

˜

Ω(

d

2

)

<!-- formula-not-decoded -->

## C.5 Proof of claim B.2

This Section presents the exponential decay bounds for terms ( β 3 -β 1 c ⊤ i ( T + 1) b i ( T + 1) ) , c ⊤ i ( T +1) b j ( T +1) and c ⊤ i ( T +1) b ( T +1) , establishing the property B ( T +1) .

<!-- formula-not-decoded -->

Recall the following equation from lemma A.7.

̸

<!-- formula-not-decoded -->

Based on the above equation, we have:

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

̸

̸

<!-- formula-not-decoded -->

The term β 3 -β 1 c ⊤ i ( T ) b i ( T ) is highlighted with underline, and we collect its negative feedback terms together. The factor ( 1 -ηβ 1 ( b ⊤ i ( T ) b i ( T ) + c ⊤ i ( T ) c i ( T ) ) ) ≤ 1 will drive ( β 3 -β 1 c ⊤ i ( T + 1) b i ( T +1) ) to converge to zero.

By Recurring (Eq. (31)) from 0 to T , we have:

̸

<!-- formula-not-decoded -->

̸

Here ∏ T s =0 ( 1 -ηβ 1 ( b ⊤ i ( s ) b i ( s ) + c ⊤ i ( s ) c i ( s ) ) ) ≤ (1 -2 ηβ 1 γ ) T +1 since γ ≤ b ⊤ i ( s ) b i ( s ) , c ⊤ i ( s ) c i ( s ) . Besides, from property B (0) , . . . , B ( T ) and lemma C.7 we know that c ⊤ i ( s ) b k ( s ) , c ⊤ i ( s ) b ( s ) and ¯ c ⊤ i ( s ) ¯ b i ( s ) have bounds with exponential decreasing rate. Therefore, it is easy to derive an exponential decreasing upper bound for ∣ ∣ ∣ ( β 3 -β 1 c ⊤ i ( T +1) b i ( T +1) ) ∣ ∣ ∣ .

By substituting the bounds of c ⊤ i ( s ) b k ( s ) , c ⊤ i ( s ) b ( s ) , ¯ c ⊤ i ( s ) ¯ b i ( s ) , b ⊤ k ( s ) b i ( s ) , c ⊤ i ( s ) c k ( s ) and b ⊤ i ( s ) b ( s ) , we have:

<!-- formula-not-decoded -->

The notations ♠ , ♣ and ♢ highlight the corresponding terms between (Eq. (32)) and (Eq. (33)) for refference.

We further have the following:

<!-- formula-not-decoded -->

The second inequality is derived by factoring out the factors exp( -ηβ 1 γs ) and exp( -ηβ 2 γs ) . The third inequality is due to ∑ T s =0 exp(2 ηβ 1 γ ( s -T )) · exp( -ηβ 1 γs ) ≤ 2 ηβ 1 γ exp( -ηβ 1 γ ( T +1)) and ∑ T s =0 exp(2 ηβ 1 γ ( s -T )) · exp( -ηβ 2 γs ) ≤ 3 ηβ 2 γ exp( -ηβ 1 γ ( T + 1)) in lemma C.5. The fourth inequality is by δ (0) ≤ δ ( T ) , exp( -2 ηβ 1 γ ( T +1)) ≤ exp( -ηβ 1 γ ( T +1)) , and we consider β 3 = β 3 δ (0) · δ (0) ≤ β 3 δ (0) · δ ( T ) . The fifth inequality is by proving ( β 3 δ (0) + β 1 + 8 β 1 ( d -1) δ ( T ) γ + 2 δ ( T ) γ + 16 ηd h d 2 δ ( T ) γ + 6 β 1 δ ( T ) γ + 24 ηβ 1 d h dδ ( T ) γ ) ≤ 1 as follows:

<!-- formula-not-decoded -->

The first inequality is by β 1 ≤ 3 4 . The second inequality is by δ (0) ≥ 2 √ d h log(4 d (2 d +1) /δ ) , δ ( T ) ≤ 3 √ d h log(4 d (2 d +1) /δ ) and γ = 1 2 d h . The last inequality hold as long as d h =

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof for the bounds of c ⊤ i ( T + 1) b j ( T + 1) and c ⊤ i ( T + 1) b ( T + 1) are similar to that of ( β 3 -β 1 c ⊤ i ( T +1) b i ( T +1) ) . We presents the calculation as follows.

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

̸

̸

̸

̸

<!-- formula-not-decoded -->

This bound requires δ ( T ) γ ( 4 β 1 + 8( d -2) + 2 β 1 + 16 ηd h d 2 β 1 + 6 + 24 ηd h d ) ≤ 1 , which can be verified by d h = ˜ Ω( d 2 ) ≥ 36 log(4 d (2 d + 1) /δ ) ( 4 β 1 + 8( d -2) + 2 β 1 + 2 β 1 + 6 + 12 d ) 2 ≥ 36 log(4 d (2 d +1) /δ ) ( 4 β 1 +8( d -2) + 2 β 1 + 16 ηd h d 2 β 1 +6+24 ηd h d ) 2 . Bound of c ⊤ i ( T +1) b ( T +1)

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Property B ( T +1) is established by (Eq. (35)), (Eq. (36)) and (Eq. (37)).

## C.5.1 Auxiliary lemma

Lemma C.5 As long as 2 ηβ 1 γ ≤ ln 2 , and 2 ηβ 2 γ ≤ ln 2 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of lemma C.5.

<!-- formula-not-decoded -->

The first inequality is due to exp( ηβ 1 γs ) is monotone increasing. The last inequality is due to 2 ηβ 1 γ ≤ ln 2 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The first inequality is due to exp( -η ( β 2 -2 β 1 ) γs ) is monotone decreasing. The fourth inequality is due to β 2 ≥ 4 β 1 , thus 1 β 2 -2 β 1 ≤ 2 β 2 . The last inequality is due to ηβ 2 γ ≤ (ln 2) / 2 ≤ ln(3 / 2) .

<!-- formula-not-decoded -->

The first inequality is due to exp( η (2 β 2 -β 1 ) γs ) is monotone increasing. The last inequality is due to β 2 ≥ β 1 and 2 ηβ 2 γ ≤ ln 2 .

The proof of ∑ T s =0 exp(2 ηβ 2 γ ( s -T )) · exp( -ηβ 2 γs ) ≤ 2 ηβ 2 γ exp( -ηβ 2 γ ( T +1)) is similar to ∑ T s =0 exp(2 ηβ 1 γ ( s -T )) · exp( -ηβ 1 γs ) ≤ 2 ηβ 1 γ exp( -ηβ 1 γ ( T +1)) . Just replace β 1 with β 2 , and consider 2 ηβ 2 γ ≤ ln 2 .

## C.6 Proof of claim B.3

This Section presents the bounds for terms b ⊤ i ( T +1) b j ( T +1) , c ⊤ i ( T +1) c j ( T +1) and b ⊤ i ( T + 1) b ( T +1) with i, j ∈ [1 , d ] , i = j , establishing the property C ( T +1) .

̸

Recall the Vector-coupled Dynamics equations of b ⊤ i ( t + 1) b j ( t + 1) , c ⊤ i ( t + 1) c j ( t + 1) and b ⊤ i ( t +1) b ( t +1) in lemma A.7:

<!-- formula-not-decoded -->

̸

̸

̸

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

To give bounds for the above three terms, we will use the following bounds from property B ( t ) and lemma C.7:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Besides, by ∣ ∣ ∣ β 3 -β 1 c ⊤ i ( t ) b i ( t ) ∣ ∣ ∣ ≤ δ ( t ) exp( -ηβ 1 γt ) , we have:

<!-- formula-not-decoded -->

The third inequality is by δ ( s ) ≥ 2 √ d h log(4 d (2 d +1) /δ ) ≥ 2 α = 2 α = 2exp(( -ln 2) /N ) . For the last inequality, as long as N ≥ 2 ln 2 ln 6 -ln 5 , we have 5 α 2 ≤ 6 .

We will provide the upper bounds for ∣ ∣ ∣ b ⊤ i ( T +1) b j ( T +1) ∣ ∣ ∣ , ∣ ∣ ∣ c ⊤ i ( T +1) c j ( T +1) ∣ ∣ ∣ and ∣ ∣ ∣ b ⊤ i ( T + 1) b ( T +1) ∣ ∣ ∣ by substituting the above bounds into (Eq. 38), (Eq. 39) and (Eq. 40) respectively.

Bound of ∣ ∣ ∣ b ⊤ i ( T +1) b j ( T +1) ∣ ∣ ∣

<!-- formula-not-decoded -->

̸

̸

The first inequality is derived by triangle inequality. The second inequality is derived by ∣ ∣ ∣ b ⊤ i ( T ) b j ( T ) ∣ ∣ ∣ ≤ δ ( T ) and substituting the bounds of | β 3 -β 1 c ⊤ i ( t ) b i ( t ) | , | c ⊤ i ( t ) b j ( t ) | and ∣ ∣ ∣ ¯ b i ( t ) ⊤ ¯ b j ( t ) ∣ ∣ ∣ . The third inequality is derived by factoring out the common factor δ ( T ) . The last inequality is derived by δ ( T ) ≤ δ max and exp( -ηβ 1 γT ) ≤ 1 .

̸

̸

Bound of ∣ ∣ ∣ c ⊤ i ( T +1) c j ( T +1) ∣ ∣ ∣

<!-- formula-not-decoded -->

̸

̸

The first inequality is derived by triangle inequality. The second inequality is derived by ∣ ∣ ∣ b ⊤ i ( T ) b j ( T ) ∣ ∣ ∣ ≤ δ ( T ) and substituting the bounds of | β 3 -β 1 c ⊤ i ( t ) b i ( t ) | , | c ⊤ i ( t ) b j ( t ) | and ∣ ∣ ∣ ¯ c i ( t ) ⊤ ¯ c j ( t ) ∣ ∣ ∣ . The third inequality is derived by factoring out the common factor δ ( T ) . The last inequality is derived by δ ( T ) ≤ δ max , exp( -ηβ 1 γT ) ≤ 1 and exp( -ηβ 2 γT ) ≤ 1 .

̸

̸

Bound of ∣ ∣ ∣ b ⊤ i ( T +1) b ( T +1) ∣ ∣ ∣

̸

<!-- formula-not-decoded -->

̸

The first inequality is derived by triangle inequality. The second inequality is derived by ∣ ∣ ∣ b ⊤ i ( T ) b j ( T ) ∣ ∣ ∣ ≤ δ ( T ) and substituting the bounds of | β 3 -β 1 c ⊤ i ( t ) b i ( t ) | , | c ⊤ i ( t ) b j ( t ) | , | c ⊤ i ( t ) b i ( t ) | and ∣ ∣ ∣ ¯ b i ( t ) ⊤ ¯ b ( t ) ∣ ∣ ∣ . The third inequality is derived by factoring out the common factor δ ( T ) . The last inequality is derived by δ ( T ) ≤ δ max , exp( -ηβ 1 γT ) ≤ 1 and 2( β 1 + β 2 )( d -1) + 1 ≤ 4 β 2 d since β 2 ≥ β 1 and β 2 ≥ 1 .

̸

̸

̸

We next provide the upper bound for δ ( T +1) .

<!-- formula-not-decoded -->

This inequality can be verified by comparing with (Eq. (41)), (Eq. (42)), (Eq. (43)). To give more precise bound, we introduce the following lemma:

Lemma C.6 If y ( t +1) ≤ y ( t ) + cy ( t ) exp( -at ) + dy ( t ) exp( -bt ) , with a, b, c, d &gt; 0 , t ≥ 0 and a, b ≤ ln 2 , then y ( t ) satisfies:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of lemma C.6.

<!-- formula-not-decoded -->

The fourth arrow is due to ln(1 + x ) ≤ x for x ≥ 0 . The fifth arrow is due to exp( -as ) , exp( -bs ) are monotone decreasing. the 7-th arrow is due to a, b ≤ ln 2 and -exp( -at ) ≤ 0 , -exp( -bt ) ≤ 0 .

Lemma C.6 presents the core idea of establishing property C ( T +1) . If a ≫ c and b ≫ d in the above lemma, we will have y ( t +1) ≤ y (0) · O (1) . Similarly, as Mamba converges quickly ( C ⊤ B → β 3 β 1 I , C ⊤ b → 0 ), we can prove that | b ⊤ i ( t ) b j ( t ) | , | c ⊤ i ( t ) c j ( t ) | , | b ⊤ i ( t ) b ( t ) | hold their magnitudes around their initial states.

We next combine (Eq. (44)) and lemma C.6 to give bound for δ ( T +1) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The first inequality is derived by (Eq. (44)). The second inequality is derived by lemma C.6. The third inequality is derived by γ = 1 2 d h . The last inequality is derived by

<!-- formula-not-decoded -->

## C.7 Bounds of η 2 terms

This Section presents the bounds for ¯ b i ( t ) ⊤ ¯ b j ( t ) , ¯ c i ( t ) ⊤ ¯ c j ( t ) , ∥ ∥ ∥ ¯ b ( t ) ∥ ∥ ∥ 2 2 , ¯ c ⊤ i ( t ) ¯ b j ( t ) , ¯ c ⊤ i ( t ) ¯ b ( t ) , ¯ b ⊤ i ( t ) ¯ b ( t ) (these terms usually appear in the Vector-coupled Dynamics equations with a η 2 factor) with i, j ∈ [1 , d ] under the assumption that A ( t ) , B ( t ) , and C ( t ) hold.

Lemma C.7 Under the assumption that A ( t ) , B ( t ) , and C ( t ) hold, we have the following bounds:

<!-- formula-not-decoded -->

̸

where i, j ∈ [1 , d ] . Note that this lemma does not require i = j .

Firstly, recall the following dynamics equation in lemma A.6:

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Thus we have

<!-- formula-not-decoded -->

Recalling the properties A ( t ) and B ( t ) :

A ( t ) :

B ( t ) :

<!-- formula-not-decoded -->

We can derive the follow bounds for the norm of ¯ b i ( t ) , ¯ c i ( t ) and ¯ b ( t ) :

̸

<!-- formula-not-decoded -->

The last inequality is by β 1 ≤ 1 .

<!-- formula-not-decoded -->

The last inequality is by β 1 ≤ 1 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

<!-- formula-not-decoded -->

By multiplying them pairwise, we obtain:

∣ ∣ ∣ ¯ b i ( t ) ⊤ ¯ b j ( t ) ∣ ∣ ∣ ≤ ( 2 √ 2 d h dδ ( t ) exp( -ηβ 1 γt ) ) 2 ≤ 8 d h d 2 δ ( t ) 2 exp( -ηβ 1 γt ) The last inequality is by exp( -2 ηβ 1 γt ) ≤ exp( -ηβ 1 γt ) . ∣ ∣ ∣ ¯ c i ( t ) ⊤ ¯ c j ( t ) ∣ ∣ ∣ ≤ ( 2 √ 2 d h dδ ( t ) exp( -ηβ 1 γt ) + 2 √ 2 d h β 2 δ ( t ) exp( -ηβ 2 γt ) ) 2 = 8 d h d 2 δ ( t ) 2 exp( -2 ηβ 1 γt ) + 8 β 2 2 d h δ ( t ) 2 exp( -2 ηβ 2 γt ) +16 d h β 2 dδ ( t ) 2 exp( -η ( β 1 + β 2 ) γt ) ≤ 8 d h d 2 δ ( t ) 2 exp( -ηβ 1 γt ) + 40 β 2 2 d h δ ( t ) 2 exp( -ηβ 2 γt )

The last inequality is by β 2 d ≤ 2 β 2 2 because β 2 = Ω( d 2 ) ≥ d , and exp( -2 ηβ 1 γt ) ≤ exp( -ηβ 1 γt ) , exp( -2 ηβ 2 γt ) ≤ exp( -ηβ 2 γt ) , exp( -η ( β 1 + β 2 ) γt ) ≤ exp( -ηβ 2 γt ) .

<!-- formula-not-decoded -->

The last inequality is by β 2 ≤ β 2 2 because β 2 ≥ 1 , and exp( -2 ηβ 1 γt ) ≤ exp( -ηβ 1 γt ) , exp( -2 ηβ 2 γt ) ≤ exp( -ηβ 2 γt ) , exp( -η ( β 1 + β 2 ) γt ) ≤ exp( -ηβ 2 γt ) .

∣ ∣ ∣ ¯ c ⊤ i ( t ) ¯ b j ( t ) ∣ ∣ ∣ ≤ ∥ ∥ ∥ ¯ c i ( t ) ∥ ∥ ∥ · ∥ ∥ ∥ ¯ b j ( t ) ∥ ∥ ∥ ≤ ( 2 √ 2 d h dδ ( t ) exp( -ηβ 1 γt ) + 2 √ 2 d h β 2 δ ( t ) exp( -ηβ 2 γt ) ) · 2 √ 2 d h dδ ( t ) exp( -ηβ 1 γt ) ≤ 8 d h d 2 δ ( t ) 2 exp( -ηβ 1 γt ) + 8 β 2 d h dδ ( t ) 2 exp( -ηβ 2 γt ) The last inequality is by exp( -2 ηβ 1 γt ) ≤ exp( -ηβ 1 γt ) , exp( -η ( β 1 + β 2 ) γt ) ≤ exp( -ηβ 2 γt ) .

<!-- formula-not-decoded -->

The last inequality is by β 2 d ≤ 2 β 2 2 , β 2 ≤ β 2 2 , and exp( -2 ηβ 1 γt ) ≤ exp( -ηβ 1 γt ) , exp( -2 ηβ 2 γt ) ≤ exp( -ηβ 2 γt ) , exp( -η ( β 1 + β 2 ) γt ) ≤ exp( -ηβ 2 γt ) .

<!-- formula-not-decoded -->

The last inequality is by exp( -2 ηβ 1 γt ) ≤ exp( -ηβ 1 γt ) , exp( -η ( β 1 + β 2 ) γt ) ≤ exp( -ηβ 2 γt ) .

## D Discussion

In this section, we show that orthogonal initialization Mamba can be trained to ICL solution, and compare our method with some previous works.

Orthogonal Initialization Now we assume that each column of W B and W C are initialized with orthogonal columns of unit norm. Then we have

<!-- formula-not-decoded -->

Consider the following update rule as part of lemma 5.1.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By (Eq. (45)), (Eq. (46)) and (Eq. (47)), we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining B ⊤ (0) b (0) = C ⊤ (0) b (0) = 0 with induction, we can derive that B ⊤ ( t ) b ( t ) = C ⊤ ( t ) b ( t ) = 0 for t ≥ 0 . Thus we only need to consider the following dynamics.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Denote B ( t ) ⊤ B ( t ) = D ( t ) , C ( t ) ⊤ C ( t ) = E ( t ) and C ( t ) ⊤ B ( t ) = F ( t ) then by (Eq. (48)) and (Eq. (49)), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that D (0) = E (0) = I and F (0) = 0 . By induction we can see that D ( t ) , E ( t ) and F ( t ) are diagonal matrix for t &gt; 0 . Because of the symmetry, we have D ( t ) = E ( t ) . Now we denote D ( t ) = E ( t ) = g ( t ) I and F ( t ) = h ( t ) I . Then based on (Eq. (50)), (Eq. (51)) and (Eq. (52)), we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since g (0) = 1 and h (0) = 0 at initialization, h ( t ) will converge to β 3 β 1 (i.e. C ⊤ B → β 3 β 1 I ).

Compare with Other Techniques (Eq. (45)), (Eq. (46)) and (Eq. (47)) can be viewed as the gradient descent that minimize the following target:

<!-- formula-not-decoded -->

where X and Y satisfy:

<!-- formula-not-decoded -->

To establish convergence for this problem under gaussian initialization, Arora et al. (2019) require the standard deviation to be small enough, while Du and Hu (2019) require larger dimension d h because their method relies on the condition number of X . Our method balances the requirements on initialization and dimension. The fine-grained nature of our analysis (particularly the Vector-coupled Dynamics ) enables extension to various problem beyong (Eq. (55)).

## E Additional Experimental Results

(a) Initialization

<!-- image -->

(b) Trained parameter

Figure 2: (a) Visualization of matrix product C ⊤ W B before training; (b) Post-training visualization of matrix product C ⊤ W B ; (c) Test loss versus token sequence length N . Blue curve: experimental loss; orange dashed line: theoretical loss d 2 ( 1 -β 2 3 β 1 ) .

Experiments Setting We follow Section 3 to generate the dateset and initialize the model. Specifically, we set dimension d = 4 , d h = 80 , prompt token length N = 50 , and train the Mamba model on 3000 sequences by gradient descent. Moreover, we vary the length of the prompt token N from 4 to 80 and compare the test loss with the theoretical loss. For each N , we conduct 10 independent experiments and report the averaged results. All experiments are performed on an NVIDIA A800 GPU.

Experiment Result Figure 2a and Figure 2b show that C ⊤ B can be trained to diagonal matrix from random initialization. Figure 2c show that the experimental loss aligns with the theoretical loss L ( θ ) = d 2 ( 1 -β 2 3 β 1 ) , noting that the theoretical loss ( 1 -β 2 3 β 1 ) has an upper bound 3 d ( d +1) 2 N that decays linearly with N. These experimental results further verified our theoretical proof.

Mamba vs Linear Attention Optimal linear attention outperforms Mamba under our construction, and they have O (1 /N ) error upper bound with different constant factors. We provide a comprarison of loss between optimal Mamba (under our Assumption 4.1) with optimal linear attention as in Table 2 with setting d = 10 , N = 10 , 20 , . . . , 80 .

When N is smaller than d We also test the case when N ≤ d in Table 3 with setting d = 20 , N = 4 , 6 , . . . , 20 .

Convergence of w ∆ We set w ∆ = 0 in the assumption. Now we show that random initializd w ∆ = 0 can converge to 0 experimental. The results is in Table 4.

Different d h Table 5 shows the mean value and standard deviation of the loss for smaller d h (in 10 repeated experiments). We set d = 4 , N = 30 , and the theoretical loss is 0.2954.

Table 2: Comparison of Mamba and Linear Attention

| N                |     10 |     20 |     30 |     40 |     50 |     60 |     70 |     80 |
|------------------|--------|--------|--------|--------|--------|--------|--------|--------|
| Mamba            | 2.6671 | 1.8189 | 1.38   | 1.1117 | 0.9308 | 0.8005 | 0.7022 | 0.6254 |
| Linear Attention | 2.619  | 1.7742 | 1.3415 | 1.0784 | 0.9016 | 0.7746 | 0.679  | 0.6044 |

Table 3: Experiment for N ≤ d

| N                 |      4 |      6 |      8 |     10 |     12 |     14 |     16 |     18 |     20 |
|-------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Experimental Loss | 8.5911 | 7.8292 | 7.7009 | 6.8235 | 6.4004 | 6.0612 | 5.9689 | 5.6193 | 5.1426 |
| Theoretical Loss  | 8.4484 | 7.8425 | 7.3173 | 6.8579 | 6.4526 | 6.0926 | 5.7706 | 5.481  | 5.219  |

Table 4: Convergence of w ∆

| Epoch               |      0 |     10 |     20 |     30 |     40 |     50 |     60 |     70 |     80 |
|---------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| ∥ w ∆ ∥ 2 ∥ w ∆ ∥ 2 | 0.8883 | 0.7513 | 0.4821 | 0.3331 | 0.2444 | 0.2026 | 0.1868 | 0.1799 | 0.1773 |
| 2                   | 0.7891 | 0.5645 | 0.2324 | 0.1109 | 0.0597 | 0.041  | 0.0349 | 0.0324 | 0.0314 |

Table 5: Different d h

| d h        |      6 |      8 |     10 |     12 |     14 |     16 |     18 |     20 |
|------------|--------|--------|--------|--------|--------|--------|--------|--------|
| mean(loss) | 0.2912 | 0.2933 | 0.2899 | 0.2887 | 0.2929 | 0.2951 | 0.2967 | 0.2959 |
| std(loss)  | 0.0075 | 0.0055 | 0.0116 | 0.0052 | 0.0105 | 0.0097 | 0.011  | 0.0142 |