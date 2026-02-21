## When Do Transformers Outperform Feedforward and Recurrent Networks? A Statistical Perspective

Alireza Mousavi-Hosseini 1 , Clayton Sanford 2 , Denny Wu 3 , Murat A. Erdogdu 1

1 University of Toronto and Vector Institute, 2 Google Research, 3 New York University and Flatiron Institute

{mousavi,erdogdu}@cs.toronto.edu , chsanford@google.com , dennywu@nyu.edu

## Abstract

Theoretical efforts to prove advantages of Transformers in comparison with classical architectures such as feedforward and recurrent neural networks have mostly focused on representational power. In this work, we take an alternative perspective and prove that even with infinite compute, feedforward and recurrent networks may suffer from larger sample complexity compared to Transformers, as the latter can adapt to a form of dynamic sparsity . Specifically, we consider a sequence-tosequence data generating model on sequences of length N , where the output at each position only depends on q ≪ N relevant tokens, and the positions of these tokens are described in the input prompt. We prove that a single-layer Transformer can learn this model if and only if its number of attention heads is at least q , in which case it achieves a sample complexity almost independent of N , while recurrent networks require N Ω(1) samples on the same problem. If we simplify this model, recurrent networks may achieve a complexity almost independent of N , while feedforward networks still require N samples. Our proposed sparse retrieval model illustrates a natural hierarchy in sample complexity across these architectures.

## 1 Introduction

The Transformer [VSP + 17], a neural network architecture that combines attention and feedforward blocks, forms the backbone of large language models and machine learning approaches across many domains [RNSS18, DBK + 20, BMR + 20]. The theoretical efforts surrounding the success of Transformers have so far demonstrated various capabilities like in-context learning [ASA + 23, VONR + 23, BCW + 23, ZFB24, KNS24, and others] and chain-of-thought prompting along with its benefits [FZG + 23, MS24, LLZM24, KS24, and others] in various settings. There are fewer works that provide specific benefits of Transformers in comparison with feedforward and recurrent architectures. On the approximation side, there are tasks that Transformers can solve with size logarithmic in the input, while alternative architectures require polynomial size [SHT23, SHT24]. Based on these results, [WWHL24] showed a separation between Transformers and feedforward networks by providing further optimization guarantees for gradient-based training of Transformers on a sparse token selection task.

While most prior works focused on the approximation separation between Transformers and feedforward networks (FFNs), in this work we focus on a purely statistical separation, and ask:

What function class can Transformers learn with fewer samples compared to feedforward and recurrent networks, even with infinite computational resources?

[FGBM23] approached the above problem with random features, where the query-key matrix for the attention and the first layer weights for the two-layer feedforward network were fixed at random

Table 1: Summary of main contributions (see Theorem 1). ✓ indicates a sample complexity upper bound that is almost sequence length-free (up to polylogarithmic factors). ✗ indicates a lower bound of order N Ω(1) .

| Statistical Model   | Feedforward   | RNN           | Transformer   |
|---------------------|---------------|---------------|---------------|
| Simple- q STR       | ✗ (Theorem 9) | ✓ (Theorem 5) | ✓ (Theorem 3) |
| q STR               | ✗ (Theorem 9) | ✗ (Theorem 7) | ✓ (Theorem 3) |

initialization. However, this only presents a partial picture, as neural networks can learn a significantly larger class of functions once 'feature learning' is allowed, i.e., parameters are trained to adapt to the structure of the underlying task [Bac17, BES + 22, DLS22, BBSS22, DKL + 23, AAM23, MHWE24].

We evaluate the statistical efficiency of Transformers and alternative architectures by characterizing how the sample complexity depends on the input sequence length. A benign length dependence (e.g., sublinear) signifies the ability to achieve low test error in longer sequences, which intuitively connects to the length generalization capability [AWA + 22]. While Transformers have demonstrated this ability in certain structured logical tasks, they fail in other simple settings [ZBL + 23, LAG + 23]. Our generalization bounds for bounded-norm Transformers - along with our contrasts to RNNs and feedforward neural networks - provide theoretical insights into the statistical advantages of Transformers and lay the foundation for future rigorous investigations of length generalization.

## 1.1 Our Contributions

We study the q -Sparse Token Regression ( q STR ) data generating model, a sequence-to-sequence model where the output at every position depends on a sparse subset of the input tokens. Importantly, this dependence is dynamic, i.e., changes from prompt to prompt, and is described in the input itself. We prove that by employing the attention layer to retrieve relevant tokens at each position, singlelayer Transformers can adapt to this dynamic sparsity, and learn q STR with a sample complexity almost independent of the length of input sequence N , as long as the number of attention heads is at least q . On the other hand, we develop a new metric-entropy-based argument to derive norm and parameter-count lower bounds for RNNs approximating the q STR model. Thanks to lower bounds on weight norm, we also obtain a sample complexity lower bound of order N Ω(1) for RNNs. Further, we show that RNNs can learn a subset of q STR where the output is a constant sequence, which we call simpleq STR , with a sample complexity polylogarithmic in N . Finally, we develop a lower bound technique for feedforward networks (FFNs) that takes advantage of the fully connected projection of the first layer to obtain a sample complexity lower bound linear in N , even when learning simpleq STR models. The following theorem and Table 1 summarize our main contributions.

Theorem 1 (Informal) . We have the following hierarchy of statistical efficiency for learning q STR .

- A single-layer Transformer with H ≥ q heads can learn q STR with sample complexity almost independent of N , and cannot learn q STR when H &lt; q even with infinitely many samples.
- RNNs can learn simple -q STR with sample size almost independent of N , but require at least Ω( N c ) samples for some constant c &gt; 0 to learn a generic q STR model, regardless of their size.
- Feedforward neural networks, regardless of their size, require Ω( Nd ) samples to learn even simple -q STR models, where d is input token dimension.

We empirically validate the intuitions from Theorem 1 in Figure 1. Observe that on a 1STR task, both FFNs and RNNs suffer from a large sample complexity for larger N . However, for a simple1STR model, RNNs perform closer to Transformers with a much milder dependence on N than FFNs.

## 1.2 Related Work

While generalization is a fundamental area of study in machine learning theory, theoretical work on the generalization capabilities of Transformers remains relatively sparse. Some works analyze the inductive biases of self-attention through connections to max-margin SVM classifiers [VDT24]. Others quantify complexity in terms of the simplest programs in a formal language (such as the RASP model of [YCA23]) that solve the task and relate that to Transformer generalization [ZBL + 23, CS24]. The most relevant works to our own are [EGKZ22, TT23, Tru24], which employ covering numbers to bound the sample complexity of deep Transformers with bounded weights. They demonstrate a logarithmic scaling in the sequence length, depth, and width and apply their bounds

Figure 1: Number of samples required to reach a certain test MSE loss threshold while training with online AdamW. We consider (a) the 1STR model with loss threshold 0 . 7 and (b) the simple1STR model with loss threshold 0 . 02 , averaged over 5 experiments. We use a linear link function, standard Gaussian input, d = 10 and d e = ⌊ 5 log( N ) ⌋ . Positional encodings are sampled uniformly from the unit hypercube. Experimental details and additional results on the effect of q are provided in Appendix E.

<!-- image -->

to the learnability of sparse Boolean functions. We refine these covering number bounds to better characterize generalization in sequence-to-sequence learning with dynamic sparsity [SHT23]. Our problems formalize long-context reasoning tasks, extending beyond simple retrieval to include challenges like multi-round coreference resolution [VOT + 24].

Expressivity of Transformers. The expressive power of Transformers has been extensively studied in prior works. Universality results establish that Transformers can approximate the output of any continuous function or Turing machine [YBR + 19, WCM21], as well as measure-to-measure maps [GRRB24], and their memorization capacity is well-understood [MLT24]. However, complexity limitations remain for bounded-size models. Transformers with fixed model sizes are unable to solve even regular languages, such as Dyck and Parity [BAG20, Hah20]. Further work [e.g. MS23] relates Transformers to boolean circuits to establish the hardness of solving tasks like graph connectivity with even polynomial-width Transformers. Additionally, work on self-attention complexity explores how the embedding dimension and number of heads affects the ability of attention layers to approximate sparse matrices [LCW21], recover nearest-neighbor associations [AYB24], and compute sparse averages [SHT23]. The final task closely resembles our q STR model and has been applied to relate the capabilities of deep Transformers to parallel algorithms [SHT24]. Several works [e.g. JBKM24, BHBK24, WDL24] introduce sequential tasks where Transformers outperform RNNs or other state space models in parameter-efficient expressivity. We establish similar architectural separations with an added focus on generalization capabilities.

Statistical Separation. Our work is conceptually related to studies on feature learning and adaptivity in feedforward networks, particularly in learning models with sparsity and low-dimensional structures. Prior work has analyzed how neural networks and gradient-based optimization introduce inductive biases that facilitates the learning of low-rank and low-dimensional functions [LMZ18, WLLM19, CB20, MHPG + 23, OSSW24]. These studies often demonstrate favorable generalization properties based on certain structures of the solution such as large margin or low norm [BFT17, NLB + 18, OWSS19, WLLM19]. Our goal is to extend efficient learning of low-dimensional concepts to sequential architectures, ensuring sample complexity remains efficient in both input dimension d and context length N . Our approach, motivated by [SHT23, WWHL24], suggests that q STR is a sequential model whose sparsity serves as a low-dimensional structure, making it the primary determinant of generalization complexity for Transformers.

Notation. For a natural number n , define [ n ] := { 1 , . . . , n } . We use ∥·∥ p to denote the ℓ p norm of vectors. For a matrix A ∈ R m × n , ∥ A ∥ p,q := ∥ ∥ ( ∥ A : , 1 ∥ p , . . . , ∥ A : ,n ∥ p )∥ ∥ q , and ∥ A ∥ op denotes the operator norm of A . We use a ≲ b and a ≤ O ( b ) interchangeably, which means a ≤ Cb for some absolute constant C . We similarly define ≳ and Ω . ˜ O and ˜ Ω hide multiplicative constants that depend polylogarithmically on problem parameters. σ denotes the ReLU activation.

## 2 Problem Setup

Statistical Model. In this paper, we will focus on the ability of different architectures for learning the following data generating model.

Definition 2 ( q -Sparse Token Regression) . Suppose p , y ∼ P where

<!-- formula-not-decoded -->

t i ∈ [ N ] q and x i ∈ R d for i ∈ [ N ] . In the q -sparse token regression ( q STR) data generating model, the output is given by y = ( y 1 , . . . , y N ) ⊤ ∈ R N , where

<!-- formula-not-decoded -->

for some g : R qd → R . We call this model simple -q STR if the data distribution is such that t i = t for all i ∈ [ N ] and some t drawn from [ N ] q .

The above defines a class of sequence-to-sequence functions, where the label at position i in the output sequence depends only on a subsequence of size q of the input data, determined by the set of indices t i . p in the above definition denotes the prompt or context. Given the large context length of modern architectures, we are interested in a setting where q ≪ N . In this setting, the answer at each position only depends on a few tokens, however the tokens it depends on change based on the context. Therefore, we seek architectures that are adaptive to this form of dynamic sparsity in the true data generating process, with computational and sample complexity independent of N . As a special case, choosing g as the tokens' mean recovers the sparse averaging model proposed in [SHT23], where the authors separate the representational capacity of Transformers and other architectures.

While our main motivation for using the q STR model is the role of this model as a theoretical benchmark (cf. [SHT23, WWHL24]), we now present an example of how tasks similar to q STR can arise in natural language modeling. Consider the prompt ' For my vacation this summer, I'm considering either Paris or Tokyo. If I go to Paris, I want to visit their art museums, and if I end up in Tokyo, I want to try their cuisine. Can you tell me how much would my first and second option cost respectively? ' In this case, t 1 is the token first and refers to the tokens Paris and art museumes , while t 2 is the token second and refers to the tokens Tokyo and cuisine . Note that for either t 1 or t 2 , the answer to the prompt only depends on two tokens out of the entire context, thus this example demonstrates the case of q = 2 . We refer the interested readers to the multi-round conference resolution task of [VOT + 24] for more realistic examples in evaluating large models.

To obtain statistical guarantees, we will impose mild moment assumptions on the data.

Assumption 1. Suppose E [ ∥ x i ∥ r ] 1 /r ≤ √ C x dr and E [ | y i | r ] 1 /r ≤ √ C y r s for all r ≥ 1 , i ∈ [ N ] , and some absolute constants s ≥ 1 and C x , C y &gt; 0 .

We only require the above assumption to establish standard concentration bounds, and it is satisfied as soon as ∥ x ∥ is subGaussian and y is sub-Weibull (e.g. g grows at most like a polynomial of degree s ). Learning the q STR model requires two steps: ( i ) extracting the relevant tokens at each position, ( ii ) learning the link function g . We are interested in settings where the difficulty of learning is dominated by the first step, hence we assume g can be approximated by a two-layer feedforward network.

Assumption 2. There exist m g ∈ N , a g , b g ∈ R m g and W g ∈ R m g × qd , such that ∥ a g ∥ 2 ≤ r a / √ m g , and ∥ ( W g , b g ) ∥ F ≤ √ m g r w for some constants r a , r w &gt; 0 , and

<!-- formula-not-decoded -->

where C = 3 C x e and ε 2NN is some absolute constant.

Ideally, ε 2NN above is a small constant denoting the approximation error. This assumption can be verified using various universal approximation results for ReLU networks. For example, when g is an additive model of P Lipschitz functions, where each function depends only on a k -dimensional projection of the input, the above holds for every ε 2NN &gt; 0 and m g = ˜ O ( ( P/ √ ε 2NN ) k ) , r a = ˜ O ( ( P/ √ ε 2NN ) ( k +1) / 2 ) , and r w = 1 (we can always have r w = 1 by homogeneity) [Bac17].

Empirical Risk Minimization. While Empirical Risk Minimization (ERM) is a standard abstract learning algorithm to use for generalization analysis, its standard formalizations use risk functions for scalar-valued predictions. Before introducing the notions of ERM that we employ, we first state several sequential risk formulations to evaluate a predictor ˆ y arc ( · ; Θ ) ∈ F arc on i.i.d. training samples { p ( i ) , y ( i ) } n i =1 , where arc denotes a general architecture. We define the population risk , averaged empirical risk , and point-wise empirical risk respectively as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where { j ( i ) } n i =1 are i.i.d. position indices drawn from Unif([ N ]) .The goal is to minimize the population risk R arc ( Θ ) by minimizing some empirical risk, potentially with weight regularization. We use three formalizations of learning algorithms to prove our results.

1. Constrained ERM minimizes an empirical risk ˆ R arc n subject to the model parameters belonging on some (e.g., norm-constrained) set Θ . Concretely, let

<!-- formula-not-decoded -->

Theorem 3 considers constrained ERM algorithms for bounded-weight transformers with pointwise risk ˆ R TR n ( Θ ) , and Theorem 5 uses ˆ R RNN n ( Θ ) for RNNs. Note that upper bounds for training with point-wise empirical risk ˆ R arc n readily transfer to training with averaged empirical risk ˆ R arc n,N .

2. Min-norm ε -ERM minimizes the norm of the parameters, subject to sufficiently small loss:

<!-- formula-not-decoded -->

Theorem 7 uses min-norm ε -ERM to place a sample complexity lower bound ˆ R RNN n ( Θ ) .

3. Beyond ERM, Theorem 9 also considers stationary points of the averaged or point-wise loss, with ℓ 2 regularization. This learning algorithm is presented in greater detail in Definition 8.

If Θ is defined by a norm constraint, then min-norm ε -ERM with a proper ε can be seen as an instance of constrained ERM. All three formulations are motivated by practical optimization algorithms that either minimize an explicitly regularized loss, or have an implicit bias towards min-norm solutions.

## 3 Transformers

A single-layer Transformer is composed of an attention and a parallel feedforward layer. Given a sequence { z i } N i =1 of input embeddings where z i ∈ R D e with embedding dimension D e , a single head of attention outputs another sequence of length N in R D e , given by

<!-- formula-not-decoded -->

Where W K , W Q , W V are the key, query, and value projection matrices respectively. The output of H units of attention can be concatenated to form multi-head attention with output h ∈ R HD e . A two-layer neural network acts on h to generate the final output sequence via

<!-- formula-not-decoded -->

Our architectural choices are standard in theoretical studies of Transformers. We provide full details, including how to obtain input embeddings by positional encoding, in Appendix A.1.

## 3.1 Learning Guarantees for Multi-Head Transformers

We consider the following parameter class Θ TR = {∥ vec( Θ ) ∥ 2 ≤ R } and provide a learning guarantee for empirical risk minimizers over Θ TR , with its proof deferred to Appendix A.2.

Theorem 3. Let ˆ Θ = arg min Θ ∈ Θ TR ˆ R TR n ( Θ ) and m = m g . Suppose we set H = q and R 2 = ˜ Θ( r 2 a /m g + m g r 2 w + q 2 /d ) . Under Assumptions 1, 2 and 3, we have

<!-- formula-not-decoded -->

where C 1 = R 2 qd , with probability at least 1 -n -c for some absolute constant c &gt; 0 .

We make the following remarks.

- First, the sample complexity above depends on N only up to log factors. Second, we can remove the C 1 factor by performing a clipping operation with a large constant on the Transformer output. Note that the first and second terms in the RHS above denote the approximation and estimation errors respectively. Extending the above guarantee to cover m ≥ m g and H ≥ q is straightforward.
- This bound provides guidance on the relative merits of scaling the parameter complexity of the feedforward versus the attention layer (which is an active research area related to Transformer scaling laws [HSSL24, JMB + 24]), by highlighting the trade-off between the two to achieve minimal generalization error. Concretely, m g ≫ d + q represents a regime where the complexity is dominated by the feedforward layer learning the downstream task g , while m g ≪ d + q signifies dominance of the attention layer learning to retrieve the relevant tokens.

Finally, by incorporating additional structure in the ERM solution, it is possible to obtain improved sample complexities. A close study of the optimization dynamics may reveal such additional structure in the solution reached by gradientbased methods, pushing the sample complexity closer to the information-theoretic limit of Ω( qd ) . Figure 2 demonstrates that the attention weights achieved through standard optimization of a Transformer match our theoretical constructions see Equation (A.2) - even while maintaining separate W Q and W K during training (we use the 1STR setup of Figure 1 with N = 100 ). We leave the study of optimization dynamics and the resulting sample complexity for future work.

## 3.2 Limitations of Transformers with Few Heads

We establish the necessity of the linear dependence of H on q . In contrast to [AYB24], we do not put any assumptions on the rank of the key-query projections, i.e. our lower bound applies even when the key-query projection matrix is full-rank.

Proposition 4. Consider a q STR model where y i = 1 √ qd ∑ q j =1 ( ∥ x t ij ∥ 2 -E [ ∥ x t ij ∥ 2 ]) , x i ∼ N (0 , Σ i ) such that Σ i = I d for i &lt; N/ 2 and Σ i = 0 for i ≥ N/ 2 . Then, there exists a distribution over ( t i ) i ∈ [ N ] such that for any choice of Θ TR (including arbitrary { W ( h ) QK } h ∈ [ H ] ), we have

<!-- formula-not-decoded -->

Remark. We highlight the importance of the nonlinear dependence of y i on x for the above lower bound. In particular, for the sparse token averaging task introduced in [SHT23], a single-head attention layer with a carefully constructed embedding suffices for approximation.

The above proposition implies that given sufficiently large dimensionality d ≫ q , approximation alone necessitates at least H = Ω( q ) heads. In Appendix A.3, we present the proof of Proposition 4, along with Proposition 21 which establishes an exact lower bound H ≥ q for all d ≥ 1 , at the expense of additional restrictions on the query-key projection matrix.

Figure 2: Trained attention weights match our theoretical construction (A.2).

<!-- image -->

## 4 Recurrent Neural Networks

In this section, we first provide positive results for RNNs by proving that they can learn simpleq STR with a sample complexity only polylogarithmic in N , thus establishing a separation in their learning capability from feedforward networks. Next, we turn to general q STR , where we provide a negative result on RNNs, proving that to learn such models their sample complexity must scale with N Ω(1) regardless of model size, making them less statistically efficient than Transformers. Throughout this section, we focus on bidirectional RNNs, since the q STR model is not necessarily causal and the output at position i may depend on future tokens.

## 4.1 RNNs can learn simpleq STR

A bidirectional RNN maintains, for each position in the sequence, a forward and a reverse hidden state, denoted by ( h → i ) N i =1 and ( h ← i ) N i =1 , where h → i , h ← i ∈ R d h . These hidden states are obtained by initializing h → 1 = h ← N = 0 d h and recursively applying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Π r h : R d h → R d h is the projection Π r h h = (1 ∧ r h / ∥ h ∥ 2 ) h , and f → h and f ← h are implemented by feedforward networks, parameterized by Θ → h and Θ ← h respectively. Recall z i = ( x ⊤ i , enc( i, t i ) ⊤ ) ⊤ is the encoding of x i . We remark that while we add Π r h for technical reasons, it resembles layer normalization which ensures stability of the state transitions on very long inputs; a more involved analysis can replace Π r h with standard formulations of layer normalization. Additionally, directly adding h → i -1 and h ← i +1 to the output of transition functions represents residual or skip connections. The output at position i is generated by

<!-- formula-not-decoded -->

which is an L y -layer feedforward network. Specifically, we consider an RNN with deep transitions [PGCB13] and let f → h ( · ; Θ → h ) be an L h -layer feedforward network (see Appendix B.1 for complete definitions). We denote the complete output of the RNN via

<!-- formula-not-decoded -->

We have the following guarantee for RNNs learning simpleq STR models.

Theorem 5. Let ˆ Θ = arg min Θ ∈ Θ RNN ˆ R RNN n ( Θ ) (with Θ RNN defined in Equation (B.2) ). Suppose Assumptions 1, 2 and 3 hold with the simpleq STR model, i.e. t i = t for all i ∈ [ N ] and some t drawn from [ N ] q . Then, with L h , L y = O (1) , r h = ˜ Θ( √ qd ) , and proper hyperparameters in Θ RNN (see Appendix B.1), we obtain

<!-- formula-not-decoded -->

with probability at least 1 -n -c for some absolute constant c &gt; 0 .

As desired, the above sample complexity depends on N only up to polylogarithmic factors. The dimension and norm of RNN weights, implicit in the formulation above, must have a similar polynomial scaling as evident by the proof of the above theorem in Appendix B.

## 4.2 RNNs cannot learn general q STR

For our lower bound, we will consider a broad class of recurrent networks, without restricting to a specific form of parametrization. Specifically, we consider bidirectional RNNs chracterized by

<!-- formula-not-decoded -->

where f y : R d h × R d h × R d × [ N ] q +1 → R , f → h , f ← h : R d h × R d × [ N ] q +1 → R d h , U → , U ← ∈ R d h × d h , d h is the width of the model, and r h &gt; 0 is some constant. Moreover, proj r h : R d h → R d h

is any mapping that guarantees ∥ ∥ proj r h ( · ) ∥ ∥ 2 ≤ r h . As mentioned before, this operation mirrors the layer normalization to ensure that h i remains stable. Further, we assume f y ( · , x , t ) is L /r h -Lipschitz for all x ∈ R d and t ∈ [ N ] q . This formulation covers different variants of (bidirectional) RNNs used in practice such as LSTM and GRU, and includes the RNN formulation of Section 4.1 as a special case. Define U := ( U → , U ← ) ∈ R d h × 2 d h for conciseness. Note that in practice f y , f → h , f ← h are determined by additional parameters. However, the only weight that we explicitly denote in this formulation is U , since our lower bound will directly involve this projection, and we keep the rest of the parameters implicit for our representational lower bound.

Our technique for proving the RNN lower bound differs significantly from that of FFNs. In particular, we will control the representation cost of the q STR model, i.e., a lower bound on the norm of Θ RNN .

We will now present the RNN lower bound, with its proof deferred to Appendix B.5.

Proposition 6. Consider the 1STR model where x ∼ N (0 , I Nd ) with a linear link function, i.e. y j = 〈 u , x t j 〉 for some u ∈ S d -1 . Further, t i is drawn independently from the rest of the prompt and uniformly from [ N ] for all i ∈ [ N ] . Then, there exists an absolute constant c &gt; 0 , such that

<!-- formula-not-decoded -->

implies

<!-- formula-not-decoded -->

Remark. Note that the unboundedness of Gaussian random variables is not an issue for approximation here, since ( g ( x 1 ) , . . . , g ( x N )) is highly concentrated around S N -1 ( √ N ) . In fact, one can directly assume ( g ( x 1 ) , . . . , g ( x N )) ∼ Unif ( S N -1 ( √ N )) and derive a similar lower bound. The choice of Gaussian above is only made to simplify the presentation of the proof.

The above proposition has two implications. First, it has a computational consequence, implying that any RNN representing the q STR models requires a width that grows at least linearly with the context-length N . A similar lower bound in terms of bit complexity was derived in [SHT23] using different tools. More importantly, the norm lower bound ∥ U ∥ F ≥ ˜ Ω( √ N ) has a generalization consequence, which we discuss below.

To translate the above representational cost result to a sample complexity lower bound, we now introduce the parametrization of the output function f y . The exact parametrization of the transition functions will be unimportant, and we will use the notation f → h ( h , x , t ; Θ → h ) to denote a general parameterized function (similarly with f ← ). We will assume f y is given by a feedforward network,

<!-- formula-not-decoded -->

where h = ( h → , h ← ) ∈ R 2 d h , z = ( x i , f E ( t i , i )) ∈ R d + d E . Here, f E ( t i , i ) is an arbitrary encoding function with arbitrary dimension d E . Then Θ y = ( U , W y , b y , W 2 , b 2 , . . . , W L y ) , and Θ RNN = ( U , Θ y , Θ → h , Θ ← h ) . Note that thanks to the homogeneity of ReLU, we can always reparameterize the network by taking ¯ h = h /r h , ¯ W y = W y /r h , ¯ b y = b y /r h , and ¯ W 2 = W 2 /r h without changing the prediction function. Thus, in the following, we take r h = 1 without losing the expressive power of the network. We then have the following sample complexity lower bound.

Theorem 7. Consider the 1STR model of Proposition 6. Suppose the size of the hidden state, the depth of the prediction function, and the weight norm respectively satisfy d h ≤ e N c , 2 ≤ L y ≤ C , and ∥ vec( Θ RNN ) ∥ 2 ≤ e N c /L y for some absolute constants c &lt; 1 and C ≥ 2 , and recall we set r h = 1 due to homogeneity of the network. Let ˆ Θ ε be the min-norm ε -ERM of ˆ R RNN n , defined in (2.4) . Then, there exist absolute constants c 1 , c 2 , c 3 &gt; 0 such that if n ≤ O ( N c 1 ) , for any ε ≥ 0 , with probability at least c 2 over the training set,

<!-- formula-not-decoded -->

Remark. It is possible to remove the subexponential bound on ∥ vec( Θ RNN ) ∥ by allowing the learner to search over families of RNNs with arbitrary d h ≤ e N c rather than fixing a single d h . Additionally, one would avoid solutions that violate this norm constraint in practice due to numerical instability.

To prove the above theorem, we use the fact that an RNN that generalizes on the entire data distribution (hence approximates the 1STR model) requires a weight norm that scales with √ N , while overfitting on the n samples in the training set with zero empirical risk is possible with a poly( n ) weight norm. As a result, as long as n ≤ N c 1 for some small constant c 1 &gt; 0 , min-norm ε -ERM will choose models that overfit rather than generalize. A similar approach was taken in [POW + 24] to prove sample complexity separations between two and three-layer feedforward networks. The complete proof is presented in Appendix B.6.

## 5 Feedforward Neural Networks (FFNs)

In this section, we consider a general formulation of a feedforward network. Our only requirement will be that the first layer performs a fully-connected projection. The subsequent layers of the network can be arbitrarily implemented, e.g. using attention blocks or convolution filters. Specifically, the FFN implements the mapping p ↦→ f ( T , Wx ) where W ∈ R m 1 × Nd is the weight matrix in the first layer, x = ( x ⊤ 1 , . . . , x ⊤ N ) ⊤ ∈ R Nd , and f : [ N ] qN × R m 1 → R N implements the rest of the network. Unlike the Transformer architecture, here we give the network full information of T = ( t 1 , . . . , t N ) , and in particular the network can implement arbitrary encodings of the position variables t 1 , . . . , t N . This formulation covers usual approaches where encodings of t are added to or concatenated with x .

For our negative result on feedforward networks, we can further restrict the class of q STR models, and only look at simpleq STR where ˆ R n of (2.3) and ˆ R n,N of (2.2) will be equivalent. Additionally, the lower bound of this section holds regardless of the loss function used for training; for some arbitrary loss ℓ : R × R → R , we define the empirical risk of the FFN as

<!-- formula-not-decoded -->

where T ( i ) = ( t ( i ) 1 , . . . , t ( i ) N ) . We still use R FFN ( f, W ) for expected squared loss. Our lower bound covers a broad set of algorithms, characterized by the following definition.

Definition 8. Let A SP denote the set of algorithms that return a stationary point of the regularized empirical risk. Specifically, for every A ∈ A SP , A ( S n ) returns f A ( S n ) , W A ( S n ) , such that

<!-- formula-not-decoded -->

for some λ &gt; 0 depending on A . S n above denotes the training set. Let A ERM denote the set of algorithms that return the min-norm approximate ERM. Specifically, every A ∈ A ERM returns

<!-- formula-not-decoded -->

for some ε ≥ 0 . Define A := A SP ∪ A ERM .

In particular, A goes beyond constrained ERM in that it also includes the (ideal) output of first-order optimization algorithms with weight decay, or ERM with additional ℓ 2 penalty on the weights. The following minimax lower bound shows that all algorithms in class A fail to learn even the subset of simpleq STR models with a sample complexity sublinear in N .

Theorem 9. Suppose x ∼ N (0 , I Nd ) , and consider the simple1STR model with t i 1 = t 1 for all i ∈ [ N ] , where t 1 is drawn independently and uniformly in [ N ] , and a linear link function, i.e. y = ⟨ u , x t 1 ⟩ for some u ∈ S d -1 . Let A be the class of algorithms in Definition 8. Then,

<!-- formula-not-decoded -->

with probability 1 over the training set S n .

Remark. The above lower bound implies that learning the simple 1STR model with FFNs requires at least Nd samples. Note that here we do not have any assumption on m 1 , i.e. the network can have infinite width. This is a crucial difference with the lower bounds in [SHT23, WWHL24] which are computational, i.e., a similar model cannot be learned unless m 1 ≥ Nd .

The main intuition is that from the stationarity property of Definition 8, the rows of the trained W will always be in the span of the training data x ( i ) for i ∈ [ n ] . This is an n -dimensional subspace, and the best predictor that only depends on this subspace still has a loss determined by the variance of y conditioned on this subspace. By randomizing the target direction u , the label y can depend on all Nd target directions. As a result, as long as n &lt; Nd , this variance will be bounded away from zero, leading to the failure of FFNs, even with infinite compute/width. See Appendix D for detailed proof.

## 6 Conclusion

In this paper, we established a sample complexity separation between Transformers and baseline architectures, namely feedforward and recurrent networks, for learning sequence-to-sequence models where the output at each position depends on a sparse subset of input tokens described in the input itself, coined the q STR model. We proved that Transformers can learn such a model with sample complexity almost independent of the length of the input sequence N , while feedforward and recurrent networks have sample complexity lower bounds of N and N Ω(1) , respectively. Further, we established a separation between FFNs and RNNs by proving that recurrent networks can learn the subset of simpleq STR models where the output at all positions is identical, whereas feedforward networks require at least N samples. An important direction for future work is to develop an understanding of the optimization dynamics of Transformers to learn q STR models, and to study sample complexity separations that highlight the role of depth in Transformers.

## Acknowledgments and Disclosure of Funding

The authors thank Alberto Bietti and Song Mei for useful discussions. MAE was partially supported by the NSERC Grant [2019-06167], the CIFAR AI Chairs program, the CIFAR Catalyst grant, and the Ontario Early Researcher Award.

## References

- [AAM23] Emmanuel Abbe, Enric Boix Adsera, and Theodor Misiakiewicz, Sgd learning on neural networks: leap complexity and saddle-to-saddle dynamics , The Thirty Sixth Annual Conference on Learning Theory, PMLR, 2023, pp. 2552-2623.
- [ACDS23] Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra, Transformers learn to implement preconditioned gradient descent for in-context learning , Advances in Neural Information Processing Systems 37 (2023), 45614-45650.
- [ASA + 23] Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou, What learning algorithm is in-context learning? investigations with linear models , The Eleventh International Conference on Learning Representations, 2023.
- [AWA + 22] Cem Anil, Yuhuai Wu, Anders Andreassen, Aitor Lewkowycz, Vedant Misra, Vinay Ramasesh, Ambrose Slone, Guy Gur-Ari, Ethan Dyer, and Behnam Neyshabur, Exploring length generalization in large language models , Advances in Neural Information Processing Systems 35 (2022), 38546-38556.
- [AYB24] Noah Amsel, Gilad Yehudai, and Joan Bruna, On the benefits of rank in attention layers , arXiv preprint arXiv:2407.16153 (2024).
- [Bac17] Francis Bach, Breaking the curse of dimensionality with convex neural networks , Journal of Machine Learning Research 18 (2017), no. 19, 1-53.
- [BAG20] S. Bhattamishra, Kabir Ahuja, and Navin Goyal, On the ability and limitations of transformers to recognize formal languages , Conference on Empirical Methods in Natural Language Processing, 2020.
- [BBSS22] Alberto Bietti, Joan Bruna, Clayton Sanford, and Min Jae Song, Learning single-index models with shallow neural networks , Advances in Neural Information Processing Systems, 2022.

- [BCW + 23] Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, and Song Mei, Transformers as statisticians: Provable in-context learning with in-context algorithm selection , Advances in neural information processing systems 36 (2023).
- [BES + 22] Jimmy Ba, Murat A Erdogdu, Taiji Suzuki, Zhichao Wang, Denny Wu, and Greg Yang, High-dimensional Asymptotics of Feature Learning: How One Gradient Step Improves the Representation , arXiv preprint arXiv:2205.01445 (2022).
- [BFT17] Peter L Bartlett, Dylan J Foster, and Matus J Telgarsky, Spectrally-normalized margin bounds for neural networks , Advances in neural information processing systems 30 (2017).
- [BHBK24] Satwik Bhattamishra, Michael Hahn, Phil Blunsom, and Varun Kanade, Separations in the representational capabilities of transformers and recurrent architectures , 2024.
- [BMR + 20] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al., Language models are few-shot learners , Advances in neural information processing systems 33 (2020), 1877-1901.
- [CB20] Lenaic Chizat and Francis Bach, Implicit bias of gradient descent for wide two-layer neural networks trained with the logistic loss , 2020.
- [CLZ20] Minshuo Chen, Xingguo Li, and Tuo Zhao, On generalization bounds of a family of recurrent neural networks , Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, Proceedings of Machine Learning Research, vol. 108, PMLR, 2020, pp. 1233-1243.
- [CS24] Sourav Chatterjee and Timothy Sudijono, Neural networks generalize on low complexity data , ArXiv abs/2409.12446 (2024).
- [DBK + 20] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al., An image is worth 16x16 words: Transformers for image recognition at scale , arXiv preprint arXiv:2010.11929 (2020).
- [DKL + 23] Yatin Dandi, Florent Krzakala, Bruno Loureiro, Luca Pesce, and Ludovic Stephan, Learning two-layer neural networks, one (giant) step at a time , arXiv preprint arXiv:2305.18270 (2023).
- [DLS22] Alexandru Damian, Jason Lee, and Mahdi Soltanolkotabi, Neural Networks can Learn Representations with Gradient Descent , Conference on Learning Theory, 2022.
- [EGKZ22] Benjamin L Edelman, Surbhi Goel, Sham Kakade, and Cyril Zhang, Inductive biases and variable creation in self-attention mechanisms , International Conference on Machine Learning, PMLR, 2022, pp. 5793-5831.
- [FGBM23] Hengyu Fu, Tianyu Guo, Yu Bai, and Song Mei, What can a single attention layer learn? a study through the random features lens , Advances in Neural Information Processing Systems 36 (2023).
- [FZG + 23] Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang, Towards revealing the mystery behind chain of thought: a theoretical perspective , Advances in Neural Information Processing Systems 36 (2023).
- [GRRB24] Borjan Geshkovski, Philippe Rigollet, and Domènec Ruiz-Balet, Measure-to-measure interpolation using transformers , arXiv preprint arXiv:2411.04551 (2024).
- [Hah20] Michael Hahn, Theoretical limitations of self-attention in neural sequence models , Transactions of the Association for Computational Linguistics 8 (2020), 156-171.
- [HSSL24] Shwai He, Guoheng Sun, Zheyu Shen, and Ang Li, What matters in transformers? not all attention is needed , 2024.

- [JBKM24] Samy Jelassi, David Brandfonbrener, Sham M. Kakade, and Eran Malach, Repeat after me: Transformers are better than state space models at copying , ArXiv abs/2402.01032 (2024).
- [JMB + 24] Samy Jelassi, Clara Mohri, David Brandfonbrener, Alex Gu, Nikhil Vyas, Nikhil Anand, David Alvarez-Melis, Yuanzhi Li, Sham M. Kakade, and Eran Malach, Mixture of parrots: Experts improve memorization more than reasoning , 2024.
- [KNS24] Juno Kim, Tai Nakamaki, and Taiji Suzuki, Transformers are minimax optimal nonparametric in-context learners , ICML 2024 Workshop on In-Context Learning, 2024.
- [KS24] Juno Kim and Taiji Suzuki, Transformers provably solve parity efficiently with chain of thought , arXiv preprint arXiv:2410.08633 (2024).
- [LAG + 23] Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang, Exposing attention glitches with flip-flop language modeling , 2023.
- [LCW21] Valerii Likhosherstov, Krzysztof Choromanski, and Adrian Weller, On the expressive power of self-attention matrices , ArXiv abs/2106.03764 (2021).
- [LIPO23] Yingcong Li, Muhammed Emrullah Ildiz, Dimitris Papailiopoulos, and Samet Oymak, Transformers as algorithms: Generalization and stability in in-context learning , International Conference on Machine Learning, PMLR, 2023, pp. 19565-19594.
- [LLZM24] Zhiyuan Li, Hong Liu, Denny Zhou, and Tengyu Ma, Chain of thought empowers transformers to solve inherently serial problems , The Twelfth International Conference on Learning Representations, 2024.
- [LMZ18] Yuanzhi Li, Tengyu Ma, and Hongyang Zhang, Algorithmic regularization in overparameterized matrix sensing and neural networks with quadratic activations , Conference On Learning Theory, PMLR, 2018, pp. 2-47.
- [MHPG + 23] Alireza Mousavi-Hosseini, Sejun Park, Manuela Girotti, Ioannis Mitliagkas, and Murat A Erdogdu, Neural networks efficiently learn low-dimensional representations with sgd , The Eleventh International Conference on Learning Representations, 2023.
- [MHWE24] Alireza Mousavi-Hosseini, Denny Wu, and Murat A Erdogdu, Learning multiindex models with neural networks via mean-field langevin dynamics , arXiv preprint arXiv:2408.07254 (2024).
- [MLT24] Sadegh Mahdavi, Renjie Liao, and Christos Thrampoulidis, Memorization capacity of multi-head attention in transformers , The Twelfth International Conference on Learning Representations, 2024.
- [MS23] William Merrill and Ashish Sabharwal, The expressive power of transformers with chain of thought , 2023.
- [MS24] , The expressive power of transformers with chain of thought , The Twelfth International Conference on Learning Representations, 2024.
- [NLB + 18] Behnam Neyshabur, Zhiyuan Li, Srinadh Bhojanapalli, Yann LeCun, and Nathan Srebro, Towards understanding the role of over-parametrization in generalization of neural networks , arXiv preprint arXiv:1805.12076 (2018).
- [OSSW24] Kazusato Oko, Yujin Song, Taiji Suzuki, and Denny Wu, Pretrained transformer efficiently learns low-dimensional target functions in-context , arXiv preprint arXiv:2411.02544 (2024).
- [OWSS19] Greg Ongie, Rebecca Willett, Daniel Soudry, and Nathan Srebro, A function space view of bounded norm infinite width relu nets: The multivariate case , 2019.
- [PGCB13] Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, and Yoshua Bengio, How to construct deep recurrent neural networks , arXiv preprint arXiv:1312.6026 (2013).

- [POW + 24] Suzanna Parkinson, Greg Ongie, Rebecca Willett, Ohad Shamir, and Nathan Srebro, Depth separation in norm-bounded infinite-width neural networks , The Thirty Seventh Annual Conference on Learning Theory, PMLR, 2024, pp. 4082-4114.
- [RNSS18] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever, Improving language understanding by generative pre-training , OpenAI Blog (2018).
- [SHT23] Clayton Sanford, Daniel J Hsu, and Matus Telgarsky, Representational strengths and limitations of transformers , Advances in Neural Information Processing Systems 36 (2023).
- [SHT24] Clayton Sanford, Daniel Hsu, and Matus Telgarsky, Transformers, parallel computation, and logarithmic depth , Proceedings of the 41st International Conference on Machine Learning, 2024.
- [Tru24] Lan V Truong, On rank-dependent generalisation error bounds for transformers , arXiv preprint arXiv:2410.11500 (2024).
- [TT23] Jacob Trauger and Ambuj Tewari, Sequence length independent norm-based generalization bounds for transformers , 2023.
- [VDT24] Bhavya Vasudeva, Puneesh Deora, and Christos Thrampoulidis, Implicit bias and fast convergence rates for self-attention , ArXiv abs/2402.05738 (2024).
- [VONR + 23] Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov, Transformers learn in-context by gradient descent , International Conference on Machine Learning, PMLR, 2023, pp. 35151-35174.
- [VOT + 24] Kiran Vodrahalli, Santiago Ontanon, Nilesh Tripuraneni, Kelvin Xu, Sanil Jain, Rakesh Shivanna, Jeffrey Hui, Nishanth Dikkala, Mehran Kazemi, Bahare Fatemi, Rohan Anil, Ethan Dyer, Siamak Shakeri, Roopali Vij, Harsh Mehta, Vinay Ramasesh, Quoc Le, Ed Chi, Yifeng Lu, Orhan Firat, Angeliki Lazaridou, Jean-Baptiste Lespiau, Nithya Attaluri, and Kate Olszewska, Michelangelo: Long context evaluations beyond haystacks via latent structure queries , 2024.
- [VSP + 17] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin, Attention is all you need , Advances in Neural Information Processing Systems, vol. 30, 2017.
- [WCM21] Colin Wei, Yining Chen, and Tengyu Ma, Statistically meaningful approximation: a case study on approximating turing machines with transformers , 2021.
- [WDL24] Kaiyue Wen, Xingyu Dang, and Kaifeng Lyu, Rnns are not transformers (yet): The key bottleneck on in-context retrieval , 2024.
- [WLLM19] Colin Wei, Jason D Lee, Qiang Liu, and Tengyu Ma, Regularization matters: Generalization and optimization of neural nets vs their induced kernel , Advances in Neural Information Processing Systems 32 (2019).
- [WWHL24] Zixuan Wang, Stanley Wei, Daniel Hsu, and Jason D. Lee, Transformers provably learn sparse token selection while fully-connected nets cannot , Proceedings of the 41st International Conference on Machine Learning, 2024.
- [YBR + 19] Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J. Reddi, and Sanjiv Kumar, Are transformers universal approximators of sequence-to-sequence functions? , 2019.
- [YCA23] Andy Yang, David Chiang, and Dana Angluin, Masked hard-attention transformers recognize exactly the star-free languages , 2023.
- [ZBL + 23] Hattie Zhou, Arwen Bradley, Etai Littwin, Noam Razin, Omid Saremi, Josh Susskind, Samy Bengio, and Preetum Nakkiran, What algorithms can transformers learn? a study in length generalization , ArXiv abs/2310.16028 (2023).
- [ZFB24] Ruiqi Zhang, Spencer Frei, and Peter L Bartlett, Trained transformers learn linear models in-context , Journal of Machine Learning Research 25 (2024), no. 49, 1-55.

## A Details of Section 3

Here we present the omitted details and proofs of Section 3. We begin by presenting the architectural details before proving sample complexity upper bounds for Transformers.

## A.1 Transformer Architectural Definition

We formally introduce the single-layer H -headed Transformer that appears in all Section 3 proofs.

Positional encoding. To break the permutation equivaraince of Transformers, we append positional information to the input tokens. Given a prompt p , we consider an encoding given by

<!-- formula-not-decoded -->

where enc : [ N ] × [ N ] q → R d enc provides the encoding of the position and of t i , and D e := d + d enc . We use z i to refer to the i th column above. We remark that allowing enc to take t i as input allows specific encodings of the indices t i that take advantage of the q STR structure; examples of this have been considered in prior works [WWHL24]. In practice, we expect such useful encodings to be learned automatically by previous layers in the Transformer. We remark that for a fair comparison, in our lower bounds for other architectures we allow arbitrary processing of t i in their encoding procedure. To specify enc , we use a set of vectors { ω i } N i =1 in R d e that satisfy the following property.

̸

<!-- formula-not-decoded -->

Such a set of vectors can be obtained e.g., by sampling random Rademacher vectors from the unit cube {± 1 / √ d e } d e which will satisfy the assumption with high probability. We define

<!-- formula-not-decoded -->

hence d enc = ( q +1) d e and D e = d +( q +1) d e . The √ d/q prefactor ensures that x i and enc( i, t i ) will roughly have the same ℓ 2 norm, resulting in a balanced input to the attention layer.

Multi-head attention. Given a sequence { z i } N i =1 where z i ∈ R D e with D e as the embedding dimension, a single head of attention outputs another sequence of length N in R D e , given by

<!-- formula-not-decoded -->

Where W K , W Q , W V are the key, query, and value projection matrices respectively. We can simplify the presentation by replacing W ⊤ Q W K with a single parameterizing matrix for query-key projections denoted by W QK ∈ R D e × D e , and absorbing W V into the weights of the feedforward layer. This provides us with a simplified parameterization of attention, which we denote by f Attn ( p ; W QK ) . This simplification is standard in theoretical works (see e.g. [LIPO23, ACDS23, ZFB24, WWHL24]). Our main separation results still apply when maintaining separate trainable projections.

We can concatenate the output of H attention heads with separate key-query projection matrices to obtain a multi-head attention layer with H heads. We denote the output of head h ∈ [ H ] with f ( p ; W ( h ) ) . The output of the multi-head attention at position i is then given by

Attn QK

<!-- formula-not-decoded -->

We will denote by Θ QK = ( W (1) QK , . . . , W ( H ) QK ) the parameters of the multi-head attention.

Finally, a two-layer neural network acts on the output of the attention to generate labels. Given input h ∈ R HD e , the output of the network is given by

<!-- formula-not-decoded -->

where W 2NN ∈ R m × HD e are the first layer weights, b 2NN , a 2NN ∈ R m are the second layer weights and biases, and m is the width. We also use the summarized notation Θ 2NN = ( a 2NN , W 2NN , b 2NN ) to refer to the feedforward layer weights. The prediction of the transformer at position i is given by

<!-- formula-not-decoded -->

where Θ TR = ( Θ QK , Θ 2NN ) denotes the overall trainable parameters of the Transformer. We use the notation ˆ y TR ( p ; Θ TR ) = (ˆ y TR ( p ; Θ TR ) 1 , . . . , ˆ y TR ( p ; Θ TR ) N ) ⊤ ∈ R N to denote the vectorized output.

## A.2 Proof of Theorem 3

To prove Theorem 3, we will prove the more general theorem below.

Theorem 10. Let ˆ Θ := arg min Θ ∈ Θ TR ˆ R TR n ( Θ ) , where

<!-- formula-not-decoded -->

Suppose H = q , m = m g , and α = ˜ Θ(1) (given in Lemma 11). Then, under Assumptions 1, 2 and 3, with probability at least 1 -n -c for some absolute constant c &gt; 0 , we have

<!-- formula-not-decoded -->

where C 1 = qr 2 a r 2 w r 2 z .

We begin with a lemma establishing the capability of Transformers in approximating q STR models.

Lemma 11. Suppose Assumption 2 holds. Let r x = √ 3 C x ed log( nN ) . Assume H = q and m g = m . Then, there exists Θ TR such that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

for all h ∈ [ H ] .

Proof. In our construction, the goal of attention head h at position i will be to output z t ih . Namely, we want to achieve

<!-- formula-not-decoded -->

Note that to do so, for each key token z j , we only need to compute ⟨ ω t ih , ω j ⟩ . Therefore, most entries in W ( h ) QK can be zero. We only require a block of d e × d e , which corresponds to comparing ω j and ω t ih when comparing query z i and key z j . Thus, we let

<!-- formula-not-decoded -->

Then, we have 〈 z i , W ( h ) QK z j 〉 = α ⟨ ω t ih , ω j ⟩ d/q . We can then verify that

̸

<!-- formula-not-decoded -->

for every matrix A . We will specifically choose A to be the projection onto the first d coordinates in the following. Hence, α will control the error in the softmax attention approximating a 'hard-max' attention that would exactly choose z t ih .

To construct the weights of the feedforward layer a 2NN , W 2NN , b 2NN , we let a 2NN = a g and b 2NN = b g from Assumption 2, and define W 2NN by extending W g with zero entries such that

<!-- formula-not-decoded -->

Then ∥ W 2NN ∥ F = ∥ W g ∥ F . Notice that · ↦→ a ⊤ σ ( W ( · ) + b ) is r a r w Lipschitz. As a result, for any x with ∥ x ∥ ≤ r x we have

<!-- formula-not-decoded -->

where we recall

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where we recall Az j = x j . Thus, with

<!-- formula-not-decoded -->

we can guarantee the distance is at most 2 √ ε 2NN .

Before proceeding to obtain statistical guarantees, we will show that we can consider the encodings z ( i ) j to be bounded with high probability. This will be a useful event to consider throughout the proofs of various sections.

Lemma 12. Suppose { p ( i ) } n i =1 are n input prompts (not necessarily independent) drawn from the input distribution, with tokens denoted by { ( x ( i ) j ) N j =1 } n i =1 . Under Assumption 1, for any r x &gt; 0 we have

<!-- formula-not-decoded -->

In particular, for r x = √ 3 C x ed log( nN ) we have

<!-- formula-not-decoded -->

Proof. Via Markov's inequality, for any p &gt; 0 and r x &gt; 0 , we have

<!-- formula-not-decoded -->

Let p = r 2 x / ( C x ed ) . Then,

<!-- formula-not-decoded -->

which proves the first statement, and the second statement follows by plugging in the specific value of r x .

We are now ready to move to the generalization analysis of Transformers. First, we have to formally define the prediction function class of Transformers with a notation suitable for this section. We begin by defining the function class of attention. We have

<!-- formula-not-decoded -->

where we will later specify Θ QK . Additionally, we define F 2NN by

<!-- formula-not-decoded -->

where Θ 2NN = ( a 2NN , W 2NN , b 2NN ) , and we will later specify Θ 2NN . Then the class F TR can be defined as

<!-- formula-not-decoded -->

Recall we use the S n to denote the training set. To avoid extra indices, we will use the notation p , j ∈ S n to go over { p ( i ) , j ( i ) } n i =1 . We can then define the following distances on the introduced function classes

<!-- formula-not-decoded -->

We choose the radius √ Hr z for defining d 2NN ∞ since on the event of Lemma 12, this will be the norm bound on the output of the attention layer at every position.

Recall that for a distance d ∞ and a set F , an ϵ -covering ˆ F is a set such that for every f ∈ F , there exists ˆ f ∈ ˆ F such that d ∞ ( f, ˆ f ) ≤ ϵ . The ϵ -covering number of F , denoted by C ( F , d ∞ , ϵ ) , is the number of elements of the smallest such ˆ F . The following lemma relates the covering number of F TR to those of F Attn and F 2NN .

Lemma 13. Suppose f 2NN is L f Lipschitz for every f 2NN ∈ F 2NN . Then, for any ϵ 2NN , ϵ Attn &gt; 0 , on the event of Lemma 12 we have

<!-- formula-not-decoded -->

Proof. The proof simply follows from the triangle inequality, namely

<!-- formula-not-decoded -->

We have the following estimate for the covering number of F 2NN .

Lemma 14. Suppose ∥ vec( Θ RNN ) ∥ 2 ≤ R and ∥ ∥ ∥ z ( i ) j ∥ ∥ ∥ 2 ≤ R for all i ∈ [ n ] and j ∈ [ N ] . Then,

<!-- formula-not-decoded -->

This is a special case of Lemma 30, proved in Appendix B.

For the next step, define the distance

<!-- formula-not-decoded -->

on Θ QK , where we recall Θ QK = ( W (1) QK , . . . , W ( H ) QK ) ∈ R D e × HD e . The following lemma relates the covering number of the multi-head attention layer to the matrix covering number of the class of attention parameters.

Lemma 15. Suppose ∥ ∥ ∥ z ( i ) j ∥ ∥ ∥ 2 ≤ r z for all i ∈ [ n ] and j ∈ [ N ] . Then,

<!-- formula-not-decoded -->

Proof. We recall that Z ∈ R N × D e denotes the encoded prompt, and softmax is applied row-wise. For conciseness, Let ∆ := sup p ,j ∥ ∥ ∥ f ( H ) Attn ( p ; Θ QK ) j -f ( H ) Attn ( p ; ˆ Θ QK ) j ∥ ∥ ∥ 2 2 . Then we have

<!-- formula-not-decoded -->

where we used Lemma 39 for the last inequality. Moreover, by [EGKZ22, Corollary A.7],

<!-- formula-not-decoded -->

Consequently,

<!-- formula-not-decoded -->

which completes the proof.

Further, we have the following covering number estimate for Θ QK .

Lemma 16. Suppose Θ QK = {∥ Θ QK ∥ 2 , 1 ≤ R 2 , 1 , ∥ Θ QK ∥ F ≤ R F } and ∥ ∥ ∥ z ( i ) j ∥ ∥ ∥ 2 ≤ r z for all i ∈ [ n ] and j ∈ [ N ] . Then,

<!-- formula-not-decoded -->

Proof. The first estimate comes from Maurey's sparsification lemma [BFT17, Lemma 3.2], while the second estimate is based on the inequality

<!-- formula-not-decoded -->

and covering Θ QK with the Frobenius norm, see e.g. Lemma 41.

Finally, we obtain the following covering number for F TR .

Proposition 17. Suppose ∥ a 2NN ∥ 2 ≤ r m,a , ∥ ( W 2NN , b 2NN ) ∥ F ≤ R m,w , and ∥ ∥ ∥ W ( h ) QK ∥ ∥ ∥ 2 , 1 ≤ r QK for all h ∈ [ H ] . Further assume ∥ ∥ ∥ z ( i ) j ∥ ∥ ∥ 2 ≤ r z for all i ∈ [ n ] and j ∈ [ N ] . Let R := max( r m,a , R m,w , r z ) . Then, log C ( F TR , d F , ϵ ) ≲ m g HD e log(1 + R/ϵ )

<!-- formula-not-decoded -->

Proof. The proof follows from a number of observations. First, given the parameterization in the statement of the proposition, we have L f = r m,a R m,w in Lemma 13. Moreover, we have

R F ≤ √ Hr QK and R 2 , 1 ≤ Hr QK in Lemma 16. The rest follows from combining the statements of the previous lemmas.

Next, we will use the covering number bound to provide a bound for Rademacher complexity. Recall that for a class of loss functions L , the empirical and population Rademacher complexities are defined as

<!-- formula-not-decoded -->

respectively, where ( ξ i ) are i.i.d. Rademacher random variables. Let the class of loss functions be defined by

<!-- formula-not-decoded -->

for some constant τ &gt; 0 to be fixed later. We then have the following bound on Rademacher complexity.

Lemma 18. Suppose max i ∈ [ n ] ,j ∈ [ N ] ∥ ∥ ∥ z ( i ) j ∥ ∥ ∥ 2 ≤ r z . For the loss class L τ given by (A.3) , we have

<!-- formula-not-decoded -->

where C 1 = m g HD e , C 2 = r 6 z r 2 m,a R 2 m,w H 2 r 2 QK , and C 3 = HD 2 e .

Proof. Let C ( L , d L ∞ , ϵ ) denote the ϵ -covering number of L , where ℓ ( p , y , j ) = ( f ( p ) j -y j ) 2 ∧ τ and ℓ ′ ( p , y , j ) = ( f ′ ( p ) j -y j ) 2 ∧ τ . Then, for any α ≥ 0 , by a standard chaining argument,

<!-- formula-not-decoded -->

where ( C i ) 3 i =1 are given in the statement of the lemma and C 4 = Hr QK r 3 z r m,a R m,w . Choosing α = 1 / √ n completes the proof.

Using standard symmetrization techniques, the above immediately yields a high probability upper bound for the expected truncated loss of any estimator in Θ TR .

Corollary 19. Let ˆ Θ = arg min Θ ∈ Θ TR ˆ R TR n ( Θ ) , where Θ TR is described in Proposition 17. Define r z = √ r 2 x + d (1 + 1 /q ) where r x is defined in Lemma 12. Let C 1 , C 2 , and C 3 be defined as in Lemma 18. Then, with probability at least 1 -δ -( nN ) -1 / 2 over S n , we have

<!-- formula-not-decoded -->

Proof. The proof is a standard consequence of Rademacher-based generalization bounds, with the additional observation that

<!-- formula-not-decoded -->

The last step in the proof of the generalization bound is to bound R TR ( ˆ Θ ) with R TR τ ( ˆ Θ ) . This is achieved by the following lemma.

Lemma 20. Define κ 2 := Hr 2 m,a R 2 m,w r 2 z . Then, under Assumption 1, for τ ≍ κ 2 log( κ 2 N √ n ) + log( κ 2 √ n ) s , we have

<!-- formula-not-decoded -->

Proof. For conciseness, define ∆ y := ∣ ∣ ∣ ˆ y TR ( p ; ˆ Θ ) j -y j ∣ ∣ ∣ . By the Cuachy-Schwartz inequality, we have

Moreover,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Assumption 1, we have E [ y 4 j ] 1 / 2 ≲ 1 . Additionally, note that

<!-- formula-not-decoded -->

To bound max l ∈ [ N ] ∥ z l ∥ 2 , we use the subGaussianity of ∥ x l ∥ 2 characterized in Assumption 1. Specifically, for all r ≥ 1

<!-- formula-not-decoded -->

where the last inequality follows from choosing r = log N . As a result,

<!-- formula-not-decoded -->

We now turn to bounding the probability. We have

<!-- formula-not-decoded -->

where the second inequality follows from sub-Weibull concentration bounds for y and Lemma 12. Choosing τ = Θ( κ 2 log( κ 2 N √ n ) + log( κ 2 √ n ) s ) completes the proof.

Proof of Theorem 10. The theorem follows immediately from the approximation guarantee of Lemma 11, the generalization bound of Corollary 19, and the truncation control of Lemma 20.

## A.3 Details on Limitations of Transformers with Few Heads

While Proposition 4 is only meaningful in the setting of d = Ω( q ) , the following proposition provides an exact lower bound H ≥ q on the number of heads for all d , at the expense of additional restrictions on the attention matrix.

Proposition 21. Consider the q STR data model. Suppose d = 1 and y i = 1 √ q ∑ q j =1 ( x 2 t ij -E [ x 2 t ij ]) . Assume x i ∼ N (0 , σ 2 i ) independently, such that σ i = 1 for i &lt; N/ 2 and σ i = 0 for i ≥ N/ 2 . Further, assume the attention weights between the data and positional encoding parts of the tokens are fixed at zero, i.e. W ( h ) QK = ( W ( h ) x 0 d × ( q +1) d e 0 ( q +1) d e × d W ( h ) ω ) where W ( h ) x ∈ R d × d and W ( h ) ω ∈ R ( q +1) d e × ( q +1) d e are the attention parameters, for i ∈ [ H ] . Then, there exists a distribution over ( t i ) i ∈ [ N ] such that for any choice of Θ TR , we have

<!-- formula-not-decoded -->

Note that in our approximation constructions for learning q STR , we always fixed the attention weights between data and positional components to be zero, which is why we assume the same in Proposition 21.

Proof of Proposition 21. We will simply choose t i = (1 , . . . , q ) deterministically for i ≥ N 2 and draw t i from an arbitrary distribution for i &lt; N/ 2 . Note that we have

<!-- formula-not-decoded -->

Let ϕ : R HD e → R denote the mapping by the feedforward layer. Fix some i ≥ N/ 2 . Note that

<!-- formula-not-decoded -->

for some real-valued function ˜ ϕ , where

<!-- formula-not-decoded -->

are the attention scores. Let A ( i ) ∈ R H × q be the matrix such that A ( i ) hj = α ( h ) ij . Let x 1: q = ( x 1 , . . . , x q ) ⊤ ∈ R q . Then,

<!-- formula-not-decoded -->

where V ( i ) ∈ R H × q is a matrix whose rows form an orthonormal basis of span( α (1) i , . . . , α ( H ) i ) where α ( h ) i = ( α ( h ) i 1 , . . . , α ( h ) iq ) ⊤ ∈ R q (note that V ( i ) may have fewer than H rows, we consider the worst-case for the lower bound which is having H rows). The second inequality follows from the fact that z l is independent of x 1: q for l ≥ q +1 , and the fact that best predictor of y i (in L 2 error) given A ( i ) x 1: q is E [ y i | V ( i ) x 1: q ] .

Next, thanks to the structural property of W ( h ) QK in the assumption of the proposition and the fact that x i = 0 for i ≥ N/ 2 , α ( h ) ij does not depend on ( x l ) l ∈ [ q ] for all h ∈ [ H ] , i ≥ N/ 2 , and j ∈ [ q ] . As a result, V ( i ) is independent of x 1: q . Therefore,

<!-- formula-not-decoded -->

By Lemma 40, we have Var( ∥ x 1: q ∥ 2 | V ( i ) x 1: q ) = 2( q -H ) , which combined with (A.4) completes the proof.

We now present the similarly structured proof of Proposition 4.

Proof of Proposition 4. The choice of distribution over ( t i ) i ≥ N/ 2 is similar to the one presented above, i.e. we let t i = (1 , . . . , q ) deterministically for i ≥ N 2 . However, for i &lt; N 2 , we draw t i such that they are independent from x . Once again, we use the fact that

<!-- formula-not-decoded -->

Recall z i = ( x ⊤ i , enc( i, t i ) ⊤ ) . Fix some i ≥ N/ 2 , and define

<!-- formula-not-decoded -->

where we use the notation

<!-- formula-not-decoded -->

for the query-key matrix of each head. Recall that x i = 0 for i &lt; N/ 2 , thus the attention weights are given by

<!-- formula-not-decoded -->

Recall from the proof of Proposition 21 that we denote the feedforward layer by ϕ : R HD e → R . With this notation, we have

<!-- formula-not-decoded -->

Therefore, using the fact that z j and ˜ α ( h ) ij are independent of x 1: q for j ≥ l +1 , we have

<!-- formula-not-decoded -->

where α ( h,r ) i ∈ R qd such that

̸

<!-- formula-not-decoded -->

which yields 〈 α ( h,r ) i , x 1: q 〉 = ∑ q j =1 α ( h ) ij x jr , and w ( h ) i,j ∈ R qd such that

̸

<!-- formula-not-decoded -->

which yields 〈 w ( h ) i,j , x 1: q 〉 = 〈 W ( h,e,x ) ⊤ enc( i, t i ) , x j 〉 . Finally, V ( i ) is a matrix whose rows form an orthonormal basis of span ( ( α ( h,r ) i ) h = H,r = d h =1 ,r =1 , ( w ( h ) i,j ) h = H,j = q h =1 ,j =1 ) . Namely, V ( i ) has at most H ( d + q ) rows. Recall that

<!-- formula-not-decoded -->

Once again, by Lemma 40, we conclude that var( ∥ x 1: q ∥ 2 | V ( i ) x 1: q ) ≥ 2( qd -H ( q + d )) , which completes the proof.

## B Details and Proofs of Section 4

Before presenting the proofs, we state the omitted setup and parameterization of the network in the next section.

## B.1 Complete Setup of RNNs

When introducing RNNs in Section 4, we used L h -layer deep feedforward networks to implement the transitions f → h ( · ; Θ → h ) and f ← h ( · ; Θ ← h ) . These transitions are given by

<!-- formula-not-decoded -->

with Θ → h = ( W → 1 , b → 1 , . . . , W → L h -1 , b → L h -1 , W → L h ) and a similar equation for f ← ( · ; Θ ← h ) . Recall that the output of the RNN is denoted by

<!-- formula-not-decoded -->

We now define the constraint set of this architecture. Let

<!-- formula-not-decoded -->

where W → 1 ,h contains the first d h columns of W → 1 , and the conditions above are introduced to ensure f → h and f ← h are at most α N -Lipschitz with respect to the hidden state input. One way to meet this requirement is to multiply W → 1 ,h by a factor of α N / ∏ L h l =2 ∥ W → l ∥ op in the forward pass. Without this Lipschitzness constraint, current techniques for proving uniform RNN generalization bounds will suffer from a sample complexity linear in N , see e.g. [CLZ20].

For Theorem 5 we only require α N ≤ N -1 . In particular, we can choose α N = 0 and fix W → 1 ,h = W ← 1 ,h = 0 , which would simplify the parameterization of the network. Namely, in our construction f → and f ← do not need to depend on h → and h ← respectively.

## B.2 Overview of the Proof of Theorem 5

The following is the roadmap we will take for the proof of Section 4.1. The goal here is to implement a bi-directional RNN in such a way that and

Throughout this section, we will use the notation

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

✶ ✶ We can obtain the hidden states above through the following updates and

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we recall ω t = ( ω t 1 , . . . , ω t q ) , and σ is ReLU. As a result, our network must approximate

<!-- formula-not-decoded -->

A core challenge in this approximation is that if we simply control

<!-- formula-not-decoded -->

this error will propoagte through the forward pass, and we will have

<!-- formula-not-decoded -->

As a result, we would like an implementation that satisfies the following

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

Since for each l ∈ [ q ] , t l = j is possible for at most one j ∈ [ N ] , (B.4) implies

<!-- formula-not-decoded -->

for all i ∈ [ N ] , hence, we can avoid dependence on N .

We can implmenet f → h to satisfy (B.3) with a depth three network, where the first two layers implements 〈 ω i , ω t j 〉 (as a sum of Lipschitz 2-dimensional functions, an example of their approximation is given by [Bac17, Proposition 6]), and the third performs coordinate-wise product between x i and σ ( 〈 ω i , ω t j 〉 -1 / 2) (which for each coordinate is a Lipschitz two-dimensional function). To ensure f → h satisfies (B.4), we can pass the outputs to a fourth layer which rectifies its input near zero to be exactly zero using ReLU activations.

To generate y i from h → i and h ← i , we first calculate

<!-- formula-not-decoded -->

Finally, y i can be generated from h i by applying the two-layer neural network from Assumption 2 that approximates y i = g ( x t ) .

Note that the construction above has a complexity poly( d, q, log( nN )) (both in terms of number and weight of parameters), only depending on N up to log factors. As a result, by a simple parametercounting approach, the sample complexity of regularized ERM would also be (almost) independent of N . We also simply use the encoding

<!-- formula-not-decoded -->

for the RNN positive result. The scaling difference with the encoding for Transofrmers is only made to simplify the exposition, as we no longer keep explicit dependence on d and q .

## B.3 Approximations

As explained above, to implement f → h we first construct a depth three neural network (with two layers of non-linearity) which approximately performs the following mapping

<!-- formula-not-decoded -->

̸

The first mapping will be provided by

<!-- formula-not-decoded -->

where χ 0 = ( h ⊤ , x ⊤ , ω ⊤ i , ω ⊤ t 1 , . . . , ω ⊤ t q ) ⊤ ∈ R d h + d +( q +1) d e , W 1 ∈ R m 1 × ( d h + d +( q +1) d e ) , b 1 ∈ R m 1 , and A 1 ∈ R ( d + q ) × m 1 , with m 1 as the width of the first layer. We will use the notation

<!-- formula-not-decoded -->

to refer for the first d coordinates and the rest of the q coordinates of χ 1 respectively, thus ideally χ x 1 = x and χ ω 1 ( l ) = ⟨ ω i , ω t l ⟩ . The second mapping is provided by

<!-- formula-not-decoded -->

where W 2 ∈ R m 2 × ( d + q ) , b 2 ∈ R m 2 , and A 2 ∈ R dq × m 2 . We will similarly use the notation χ 2 = ( χ 2 (1) , . . . , χ 2 ( q )) , where our goal is to have χ 2 ( l ) ≈ 2 x σ ( ⟨ ω i , ω t l ⟩ -1 / 2) . To implement the first mapping, we rely on the following lemma.

Lemma 22. Let σ be the ReLU activation. For any ε &gt; 0 and positive integer d e , there exists m = O ( d 3 e (log( d e /ε ) /ε ) 2 ) , a ∈ R m , W ∈ R m × 2 d e , and b ∈ R m , such that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Proof. Consider the mapping e 1 j , e 2 j ↦→ e 1 j e 2 j . Note that when | e 1 j | ≤ 1 and | e 2 j | ≤ 1 , this mapping is √ 2 -Lipschitz, and the output is bounded between [ -1 , 1] . Then, by Lemma 42, for every ε j &gt; 0 , there exists m j ≤ O ((1 /ε j log(1 /ε j )) 2 ) , a j ∈ R m j , W j ∈ R m j × 2 d e , and b j ∈ R m j , such that

<!-- formula-not-decoded -->

∥ a j ∥ 2 ≤ O ( (log(1 /ε j ) /ε j ) 3 / 2 / √ m j ) , ∥ b j ∥ ∞ ≤ 1 , and ∥ w jl ∥ 1 ≤ 1 . Specifically, the only nonzero coordinates of w jl are the j th and d e + j th coordinates.

Let ε j = ε/d e and m = ∑ d e j =1 m j = O ( d 3 e (log( d e /ε ) /ε ) 2 ) . Construct a , b ∈ R m and W ∈ R m × 2 d e by concatenating ( a j ) , ( b j ) , and ( W j ) respectively. The resulting network satisfies

<!-- formula-not-decoded -->

while ∥ a ∥ 2 ≤ O ( d 5 / 2 e (log( d e /ε ) /ε ) 3 / 2 / √ m ) , ∥ b ∥ ∞ ≤ 1 , and ∥ ∥ ∥ W ⊤ ∥ ∥ ∥ 1 , ∞ ≤ 1 , completing the proof.

We can now specify A 1 , W 1 , and b 1 in our construction.

Lemma 23. For any ε &gt; 0 , let ¯ m 1 = O ( d 3 e (log( d e /ε ) /ε ) 2 ) and m 1 = 2 d + q ¯ m 1 . Then, there exist A 1 ∈ R ( d + q ) × m 1 , W 1 ∈ R m 1 × ( d h + d +( q +1) d e ) , and b 1 ∈ R m 1 , given by Equations (B.5) to (B.9) , such that

<!-- formula-not-decoded -->

for all h ∈ R d h , x ∈ R d , ω i , ( ω t j ) j ∈ [ q ] ∈ S d e -1 , and l ∈ [ q ] . Furthermore, we have the following guarantees

<!-- formula-not-decoded -->

Proof. We define the decompositions

<!-- formula-not-decoded -->

where W 11 ∈ R 2 d × ( d h + d + d e ) , W 12 ∈ R q ¯ m 1 × ( d h + d + d e ) , b 11 ∈ R 2 d , b 12 ∈ R q ¯ m 1 , A 11 ∈ R d × m 1 , and A 12 ∈ R q × m 1 . Let v 1 , . . . , v d denote the standard basis of R d , and notice that σ ( z ) -σ ( -z ) = z . Therefore, we can implement the identity part of the mapping by letting

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

as well as

Notice that ∥ ∥ ∥ W ⊤ 11 ∥ ∥ ∥ 1 , ∞ = 1 and ∥ ∥ ∥ A ⊤ 11 ∥ ∥ ∥ 1 , ∞ = 2 . To implement the inner product part of the mapping, we take the construction of weights, biases, and second layer weights from Lemma 22, and rename them as ˜ W 1 ∈ R ¯ m 1 × 2 d e , ˜ b 1 ∈ R ¯ m 1 , and ˜ a 1 ∈ R ¯ m 1 . Let us introduce the decomposition ˜ W 1 = ( ˜ W 11 ˜ W 12 ) , where ˜ W 11 , ˜ W 12 ∈ R ¯ m 1 × d e . With this decomposition, we can separate the projections applied to the first and second vectors in Lemma 22. We can then define

<!-- formula-not-decoded -->

as well as

<!-- formula-not-decoded -->

From Lemma 22, we have ∥ ∥ ∥ W ⊤ 12 ∥ ∥ ∥ 1 , ∞ ≤ 1 , ∥ b 12 ∥ ∞ ≤ 1 , and

<!-- formula-not-decoded -->

which completes the proof.

To introduce the construction of the next layer, we rely on the following lemma which establishes the desired approximation for a single coordinate, the proof of which is similar to that of Lemma 22.

Lemma 24. Let σ be the ReLU activation. Suppose | h | ≤ r h ∞ , | x | ≤ r x ∞ and | z | ≤ 1 . Let R := √ 1 + r x ∞ 2 + r h ∞ 2 . For any ε &gt; 0 , there exists m = O ( R 6 (log( R/ε ) /ε ) 3 ) , a ∈ R m , W ∈ R m × 2 , and b ∈ R m , such that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Additionally, if r h ∞ = 0 , we have the improved bounds

<!-- formula-not-decoded -->

̸

Proof. Note that ( h, x, z ) ↦→ h +2 xσ ( z -1 / 2) is 2 R -Lipschitz, and | h +2 xσ ( z -1 / 2) | ≤ R . The proof follows from Lemma 42 with dimension 3 when r h ∞ = 0 and dimension 2 otherwise.

With that, we can now construct the weights for the second mapping in the network.

Lemma 25. Suppose ∥ χ x 1 ∥ ∞ ≤ r x and max l | χ ω ( l ) | ≤ 1 . Let R := √ 1 + r 2 x . Then, for every ε &gt; 0 and absolute constant δ ∈ (0 , 1) , there exists ¯ m 2 ≤ O ( R 4 (log( R/ε ) /ε ) 3 / 2 ) , m 2 := qd ¯ m 2 , and A 2 ∈ R d h × m 2 , W 2 ∈ R m 2 × ( d + q ) , and b 2 ∈ R m 2 given by Equations (B.10) and (B.11) such that

<!-- formula-not-decoded -->

for all such χ 1 and l ∈ [ q ] , where we recall χ 2 = A 2 σ ( W 2 χ 1 + b 2 ) . Moreover, we have

<!-- formula-not-decoded -->

Proof. Let ˜ W = (˜ w 21 ˜ w 22 ) , ˜ b , and ˜ a be the weights obtained from Lemma 24, where ˜ w 21 , ˜ w 22 , ˜ b , ˜ a ∈ R ¯ m 2 . To construct W 2 and b 2 , we let

<!-- formula-not-decoded -->

where W 2 ( l, j ) ∈ R ¯ m 2 × ( d + q ) is given by

<!-- formula-not-decoded -->

and b 2 ( l, j ) = ˜ b 2 . Consequently, ∥ ∥ ∥ W ⊤ 2 ∥ ∥ ∥ 1 , ∞ ≤ 1 and ∥ b 2 ∥ ∞ ≤ 1 . Finally, we have

<!-- formula-not-decoded -->

Consequently, we obtain ∥ ∥ ∥ A ⊤ 2 ∥ ∥ ∥ 1 , ∞ ≤ O ( R 4 (log( R/ε ) /ε ) 3 / 2 ) , completing the proof.

We are now ready to provide the four-layer feedforward construction of f → ( h , x , t ; Θ → h ) .

Proposition 26. Let z = ( x , ω i , ω t 1 , . . . , ω t q ) . Then, for every ε &gt; 0 , there exists a feedforward network with L h = 4 layers given by

<!-- formula-not-decoded -->

where W i ∈ R m i × m i -1 , b i ∈ R i m for i ∈ { 2 , . . . , L h -1 } , W 1 ∈ R m 1 × d h + d +( q +1) d e , b 1 ∈ R m 1 , and W L h ∈ R d h × m L h -1 that satisfies the following:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all l ∈ [ q ] , h ∈ R d h and ∥ x ∥ 2 ≤ r x . Additionally ∥ W i ∥ F ≤ poly( r x , D e , ε -1 ) for all i ∈ [ L h ] and m i , ∥ b i ∥ 2 ≤ poly( r x , D e , ε -1 ) for all i ∈ [ L h -1] , where we recall D e = d +( q +1) d e .

<!-- formula-not-decoded -->

Proof. Let ˜ A 1 ∈ R ( d + q ) × m 1 , ˜ W 1 ∈ R m 1 × ( d h + d +( q +1) d e ) , ˜ b 1 ∈ R m 1 be given by Lemma 23 with error parameter ε 1 and ˜ A 2 ∈ R d h × m 2 , ˜ W 2 ∈ R m 2 × ( d + q ) , ˜ b 2 ∈ R m 2 be given by Lemma 25 with error parameter ε 2 . Recall that

<!-- formula-not-decoded -->

By the triangle inequality,

<!-- formula-not-decoded -->

where ¯ χ 1 = ( x ⊤ , ⟨ ω i , ω t 1 ⟩ , . . . , 〈 ω i , ω t q 〉 ) ⊤ . By letting ε 2 = ε/ 4 , we obtain

<!-- formula-not-decoded -->

Similarly, we can let ε 1 = ε/ ( 4 ∥ ∥ ∥ ˜ A 2 ∥ ∥ ∥ 1 , ∞ ∥ ∥ ∥ ˜ W 2 ∥ ∥ ∥ 1 , ∞ ) , which yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let

Then, satisfies ∥ χ 2 -Ψ( x , t , i ) ∥ ∞ ≤ ε/ 2 for all ∥ x ∥ 2 ≤ r x .

̸

Recall that when t l = i for some l ∈ [ q ] , we would like to guarantee the output of the network to be equal to Ψ( x , t , i ) l = 0 d . To do so, we rely on the fact that z ↦→ σ ( z -b ) -σ ( -z -b ) is zero for | z | ≤ b , and has an L ∞ distance of b from the identity, i.e. | z -σ ( z -b ) + σ ( -z -b ) | ≤ b . This mapping needs to be applied element-wise to χ 2 . Let ˜ W 3 ∈ R 2 d h × d h , b 3 ∈ R 2 d h , and W 4 ∈ R d h × 2 d h via

<!-- formula-not-decoded -->

As a result, χ 3 = W 4 σ ( ˜ W 3 χ 2 + b 3 ) satisfies

<!-- formula-not-decoded -->

̸

We thus make two observations. First, ∥ χ 3 -χ 2 ∥ ∞ ≤ ε/ 2 , and consequently ∥ χ 3 ( l ) -Ψ( x , t , i ) l ∥ ∞ ≤ ε for all l ∈ [ q ] . Second, when t l = i , we have Ψ( x , t , i ) l = 0 d and | χ 2 ( l ) j | ≤ ε/ 2 for all j ∈ [ d ] since ∥ χ 2 ( l ) -Ψ( x , t , i ) l ∥ ∞ ≤ ε/ 2 . Consequently, by the first case in (B.12), we have χ 3 ( l ) j = 0 for all j ∈ [ d ] . We can summarize these two observations as follows

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

which completes the proof.

With the above implementation of f → ( h , z ; Θ → h ) , we have the following guarantee on h → i for all i ∈ [ N ] .

Corollary 27. Let f → h be given by the construction in Proposition 26, and suppose r h ≥ √ q ( r x + √ dε ) . Then, h → i satisfies the following guarantees for all i ∈ [ N ] and l ∈ [ q ] :

1. If t l ≥ i , then h → i ( l ) = 0 d
2. If t l &lt; i , then ∥ h → i ( l ) -x t l ∥ ∞ ≤ ε .

Proof. We can prove the statement by induction. Note that it holds for i = 1 since h → 1 = 0 d . For the induction step, suppose it holds up to some i , and recall

<!-- formula-not-decoded -->

- If t l ≥ i +1 , then h → i ( l ) = 0 d and f → h ( h → j , z i ; Θ → h ) = 0 d by Proposition 26.
- If t l &lt; i &lt; i + 1 , then ∥ h → i ( l ) -x t l ∥ ∞ ≤ ε by induction hypothesis, and f → h ( h → j , z j ; Θ → h ) = 0 d .
- Finally, if t l = i &lt; i +1 , then h → i ( l ) = 0 and ∥ f → h ( h → i , z i ; Θ → h ) -x t l ∥ ∞ ≤ ε .

Note that since ∥ ∥ h → j ∥ ∥ 2 ≤ r h for all j ∈ [ N ] , the projection Π r h will always be identity through the forward pass, concluding the proof.

By symmetry, the same construction for f ← h would yield a similar guarantee on h ← j .

The last step is to design f y ( h → , h ← , z ; Θ y ) such that f y ( h → , h ← , z i ; Θ y ) ≈ g ( h → + h ← +( x ⊤ i ✶ [ t 1 = i ] , . . . , x ⊤ i ✶ [ t q = i ]) ⊤ ) . The following proposition provides the end-to-end RNN guarantee for approximating simple q STR models.

Proposition 28. Suppose g satisfies Assumption 2. Then there exist RNN weights Θ RNN with vec( Θ RNN ) ∈ R p (i.e. with p parameters) and r h ≥ √ qr x + √ ε 2NN / ( r a r w ) , such that

<!-- formula-not-decoded -->

for all t ∈ [ N ] q and ∥ x j ∥ 2 ≤ r x for all j ∈ [ N ] . Additionally, we have

<!-- formula-not-decoded -->

and f → h , f ← h do not depend on h → and h ← , namely the first d h columns of W → 1 and W ← 1 that are multiplied by h → and h ← respectively are zero.

Proof. As the proof of this proposition mostly follows from the previous proofs in this section, we only state the procedure for obtaining the desired weights.

Let ( v j ) d h j =1 denote the standard basis of R d h . Since σ ( z ) -σ ( -z ) = z , we can implement the identity mapping in R d h via a two-layer feedforward network with the following weights

<!-- formula-not-decoded -->

where W id ∈ R 2 d h × d h , b id ∈ R 2 d h , and A id ∈ R d h × 2 d h . Let W 1 , b 1 , ˜ A 1 , ˜ W 2 , b 2 , ˜ A 2 be given as in the proof of Proposition 26, for achieving an L ∞ error of ˜ ε , to be fixed later. Recall z i = ( x ⊤ i , ω ⊤ i , ω ⊤ t 1 , . . . , ω ⊤ t q ) ⊤ . In the following, we remove the zero columns of W 1 corresponding to the h part of the input (see Lemma 23), which does not change the resulting function. Our construction can then be denoted by

<!-- formula-not-decoded -->

Note that the addition above can be implemented exactly by using the fact that σ ( z 1 + z 2 + z 3 ) -σ ( -z 1 -z 2 -z 3 ) = z 1 + z 2 + z 3 . Specifically, the weights of this layer are given by

<!-- formula-not-decoded -->

where W add ∈ R 2 d h × 3 d h , b add ∈ R 2 d h , A add ∈ R d h × 2 d h .

Let Θ → h (and similarly Θ ← h ) be given by Proposition 26 with corresponding error ε h . Using the shorthand notation x t = ( x t 1 , . . . , x t q ) ∈ R dq and ˆ x t = h → i + h ← i + χ 2 , we have

<!-- formula-not-decoded -->

which holds for all input prompts p with ∥ x j ∥ 2 ≤ r x for all j ∈ [ N ] . Finally, we have

<!-- formula-not-decoded -->

Choosing ε h = √ ε 2NN / (4 √ qdr a r w ) and ˜ ε = √ ε 2NN / (2 √ qdr a r w ) , we obtain RNN weights that saitsfy ∥ vec( Θ RNN ) ∥ 2 ≤ poly( r x , D e , r a , r w , ε -1 2NN ) , completing the proof.

## B.4 Generalization Upper Bounds for RNNs

Recall the state transitions

<!-- formula-not-decoded -->

We will use the notation h → j ( p ; Θ → h ) and h ← j ( p ; Θ ← j ) to highlight the dependence of the hidden states on the prompt p and parameters Θ → h and Θ ← h . We then define the prediction function as F ( p ; Θ → h , Θ ← h , Θ y ) where

<!-- formula-not-decoded -->

We can now define the function class

<!-- formula-not-decoded -->

We can then define our distance function by going over { p , j ∈ S n } ,

<!-- formula-not-decoded -->

We will further use the notation

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

We similarly define F ← NN ,L h . The covering number of F RNN can be related to that of F y NN ,L y , F → NN ,L h , and F → NN ,L y , through the following lemma.

Lemma 29. Suppose for every Θ → h , Θ ← h , Θ y ∈ Θ RNN we have

∥ ∥ ∥ W y L y . . . W y 1 ∥ ∥ ∥ op ≤ C y W , ∥ ∥ W → L h ∥ ∥ op . . . ∥ ∥ W → 1 ,h ∥ ∥ op ≤ α N , ∥ ∥ W ← L h ∥ ∥ op . . . ∥ ∥ W ← 1 ,h ∥ ∥ op ≤ α N , where α N ≤ N -1 . Then,

<!-- formula-not-decoded -->

Proof. Throughout the proof, we will use the shorthand notation h → j = h → j ( p ; Θ → h ) and ˆ h → j = h → j ( p ; ˆ Θ → h ) , with similarly define h ← j and ˆ h ← j . We begin by observing

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Then, we observe that E 1 = d ∞ ( f y ( · ; Θ y ) , f y ( · ; ˆ Θ y )) .Thus, we can ensure E 1 ≤ ϵ/ 2 with a covering { ˆ Θ y } of size C ( F y NN ,L y , d ∞ , ϵ/ 2) . Hence, we move to E 2 .

Using the Lipschitzness of f y , we obtain

<!-- formula-not-decoded -->

Further, by Lipschitzness of Π r h , we have

<!-- formula-not-decoded -->

By the Lipschitzness of f → h , for the second term we have

<!-- formula-not-decoded -->

Moreover, we have E h 2 ≤ d ∞ ( f → h ( · ; Θ → h ) , f → h ( · ; ˆ Θ → h )) . Consequently, we obtain

<!-- formula-not-decoded -->

We can similarly obtain an upper bound on sup p ,j ∥ ∥ ∥ h ← j -ˆ h ← j ∥ ∥ ∥ 2 . Hence, we have

<!-- formula-not-decoded -->

Therefore, by constructing ϵ/ (2 eC y w N ) coverings { ˆ Θ → h } and { ˆ Θ ← h } which have sizes

<!-- formula-not-decoded -->

respectively, we complete the covering of F RNN .

The next step is to bound the covering number of the class of feedforward networks, as performed by the following lemma.

## Lemma 30. Let

<!-- formula-not-decoded -->

where Θ NN = ( W 1 , b 1 , . . . , W L -1 , b L -1 , W L ) and vec( Θ NN ) ∈ R p . Further, define the distance function

<!-- formula-not-decoded -->

Suppose ∥ W l ∥ F , ∥ b l ∥ 2 ≤ R for all l . Then, for any absolute constant depth L = O (1) , we have

<!-- formula-not-decoded -->

Proof. Let x 0 = x , x l = σ ( W l x l -1 + b l ) for l ∈ [ L -1] , and x L = W L x L -1 . Also let (ˆ x l ) be the corresponding definitions under weights and biases ( ˆ W l ) and ( ˆ b l ) . First, we remark that for l ∈ [ L -1] ,

<!-- formula-not-decoded -->

where we used the fact that L is an absolute constant. Next, for l ∈ [ L -1] , we have

<!-- formula-not-decoded -->

Once again, using the fact that L is an absolute constant and by expnaind the above inequality, we obtain

<!-- formula-not-decoded -->

Finally, we have the bound

<!-- formula-not-decoded -->

Consequently, we have

<!-- formula-not-decoded -->

where the last inequality follows from Lemma 41.

Therefore, we immediately obtain the following bound on the covering number of F RNN .

Corollary 31. Suppose Θ RNN ⊆ { Θ ∈ R p : ∥ vec( Θ ) ∥ 2 ≤ R } and ∥ ∥ ∥ z ( i ) j ∥ ∥ ∥ 2 ≤ R for all i ∈ [ n ] and j ∈ [ N ] . Then,

<!-- formula-not-decoded -->

We can now proceed with standard Rademacher complexity based arguments. Similar to the argument in Appendix A.2, we define a truncated version of the loss by considering the loss class

<!-- formula-not-decoded -->

where the constant τ &gt; 0 will be chosen later. We then have the following bound on the empirical Rademacher complexity of L RNN τ .

Lemma 32. In the same setting as Corollary 31 and with τ ≥ 1 , we have

<!-- formula-not-decoded -->

Proof. By a standard discretization bound for Rademacher complexity, for all ϵ &gt; 0 we have

<!-- formula-not-decoded -->

where the second inequality follows from Lipschitzness of ( · ) 2 ∧ τ . We conclude the proof by choosing ϵ = 1 / √ n .

We can directly turn the above bound on the empirical Rademacher complexity into a bound on generalization gap.

Corollary 33. Let ˆ Θ = arg min Θ ∈ Θ RNN ˆ R RNN n ( Θ ) . Suppose Θ RNN ⊆ { Θ ∈ R p : ∥ vec( Θ ) ∥ 2 ≤ R } , and additionally √ 3 C x ed log( nN ) + q +1 ≤ R . Then, for every δ &gt; 0 , with probability at least 1 -δ -( nN ) -1 / 2 over the training set, we have

<!-- formula-not-decoded -->

Proof. We highlight that for the specified R , Lemma 12 guarantees ∥ ∥ ∥ z ( i ) j ∥ ∥ ∥ 2 ≤ R for all i ∈ [ n ] and j ∈ [ N ] with probability at least 1 -( nN ) -1 / 2 . Standard Rademacher complexity generalization arguments applied to Lemma 32 complete the proof.

Note that ˆ R RNN τ ( ˆ Θ ) ≤ ˆ R RNN n ( ˆ Θ ) which is further controlled in the approximation section by Proposition 28. Therefore, the last step is to demonstrate that choosing τ = poly( d, q, log n ) suffices to achieve a desirable bound on R RNN ( ˆ Θ ) through R RNN τ ( ˆ Θ ) .

Lemma 34. Consider the setting of Corollary 33, and additionally assume R ≥ r h . Then, for some τ = poly( R, log n ) , we have

<!-- formula-not-decoded -->

.

Proof. The proof of this lemma proceeds similarly to the proof of Lemma 20. By defining

<!-- formula-not-decoded -->

and following the same steps (where we recall j ∼ Unif ([ N ]) ), we obtain where

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From Assumption 1, we have E [ y 4 j ] 1 / 2 ≲ 1 and P ( | y j | ≥ √ τ/ 2) ≤ e -Ω( τ 1 /s ) . For the prediction of the RNN, we have the following bound (see (B.16) for the derivation)

<!-- formula-not-decoded -->

As a result, then

<!-- formula-not-decoded -->

As a result, by the fact that r h ≤ R and Assumption 1, after taking an expectation, we immediately have

<!-- formula-not-decoded -->

On the other hand, from Lemma 12 (with n = N = 1 ), we obtain

<!-- formula-not-decoded -->

Therefore, for some τ = poly( R, log n ) we can obtain the bound stated in the lemma.

We can summarize the above facts into the proof of Theorem 5.

Proof of Theorem 5. From the approximation bound of Proposition 28, we know that for some R = poly( d, q, r a , r w , ε -1 2NN , log( nN )) and the constraint set

<!-- formula-not-decoded -->

with any α N ≤ N -1 , we have ˆ R RNN ( ˆ Θ ) ≲ ε 2NN . The proof is then completed by letting r h = √ qr x + √ ε 2NN / ( r a r w ) , invoking the generalization bound of Corollary 33, and the bound on truncation error given in Lemma 34, with R = poly( d, q, r a , r w , ε -1 2NN , log( nN )) .

## B.5 Proof of Proposition 6

The crux of the proof of Proposition 6 is to show the following position, which provides a lower bound on the prediction error at any fixed position in the prompt.

Proposition 35. Consider the same setting as in Proposition 6. There exists an absolute constant c &gt; 0 , such that for any fixed j ∈ [ N ] , if

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We shortly remark that the statement of Proposition 6 directly follows from that of Proposition 35.

Proof of Proposition 6. Let c be the constant given by Proposition 35. Suppose that

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

As a result, there exists some j ∈ [ N ] such that E [ (ˆ y RNN ( p ) j -y j ) 2 ] ≤ c . We can then invoke Proposition 35 to obtain lower bounds on d h and ∥ U ∥ op , completing the proof of Proposition 6. We now present the proof of Proposition 35.

Proof of Proposition 35. Let h j = ( U → h → j , U ← h ← j ) ∈ R 2 d h , and define

<!-- formula-not-decoded -->

In other words, Φ : R 2 d h → R N -1 captures all possible outcomes of ˆ y RNN ( p ) j depending on the value of t j (excluding the case where t j = j ). Ideally, we must have f y ( h j , x j , ( k ) , j ) ≈ g ( x k ) .

Let p (1) , . . . , p ( P ) be an i.i.d. sequence of prompts, then modify them to share the j th input token, i.e. x ( i ) j = x (1) j for all i ∈ [ P ] , with P to be determined later. Note that by our assumption on prompt distribution, this operation does not change the marginal distribution of each p ( i ) . Similarly, define

<!-- formula-not-decoded -->

for each prompt. We also let h ( i ) → j , h ( i ) ← j be the corresponding hidden states obtained from passing these prompts through the RNN, and define h ( i ) j using them. Note that g (1) , . . . , g ( P ) is an i.i.d. sequence of vectors drawn from N (0 , I N -1 ) .

We now define two events E 1 and E 2 , where

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and where δ ∈ (0 , 1) will be chosen later. In other words, E 1 is the event in which g ( i ) are 'packed' in the space, while E 2 is the event where the RNN will be 'wrong' at position j on at most 2 δ 2 fraction of the prompts. We will now attempt to lower bound P ( E 1 ∩ E 2 ) .

Note that g ( i ) -g ( k ) ( d ) = 2 g where g ∼ N (0 , I N -1 ) . By a union bound we have

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all ε g ≤ c √ 2 , where c &gt; 0 is an absolute constant such that c √ N -1 ≤ E [ ∥ g ∥ ] , and the last inequality holds by subGaussianity of the norm of a standard Gaussian random vector. From here on, we will choose ε g = c/ √ 2 (and simply denote ε g ≍ 1 ), which implies P ( E C 1 ) ≤ P 2 e -c 2 ( N -1) / 8 .

To lower bound P ( E 2 ) , consider a random prompt-label pair p , y and the corresponding g . Note that in the prompt p , the index t j is drawn independently of the rest of p , and has a uniform distribution

in [ N ] . Let p [ t j ↦→ k ] denote a modification of p where we set t j equal to k , and let y [ t j ↦→ k ] be the labels corresponding to this modified prompt. We then have

̸

<!-- formula-not-decoded -->

As a result, via a Markov inequality, we obtain

<!-- formula-not-decoded -->

Going back to our lower bound on P ( E 2 ) , define the Bernoulli random variable

<!-- formula-not-decoded -->

Note that ( z ( i ) ) are i.i.d. since h ( i ) j and g ( i ) do not depend on x j . Then, by Hoeffding's inequality,

<!-- formula-not-decoded -->

We now have our desired lower bound on P ( E 1 ∩ E 2 ) , given by

<!-- formula-not-decoded -->

Suppose δ ≥ e -c ′ N for some absolute constant c ′ &gt; 0 . Then, choosing P = ⌊ e c ′′ N ⌋ for some absolute constant c ′′ &gt; 0 would ensure P ( E 1 ∩ E 2 ) &gt; 0 , and allows us to look at this intersection.

̸

Let I = { i : z ( i ) = 0 } . On E 1 , and for i, k ∈ I with i = k we have

<!-- formula-not-decoded -->

Note that from the Lipschitzness of f y , we have ∥ ∥ ∥ Φ( h ( i ) j ) -Φ( h ( k ) j ) ∥ ∥ ∥ 2 ≤ L √ N r h ∥ ∥ ∥ h ( i ) j -h ( k ) j ∥ ∥ ∥ 2 . As a result, the set { h ( i ) j : i ∈ I } is an r h ε h -packing for { h : ∥ h ∥ 2 ≤ √ 2 ∥ U ∥ op r h } . Using Lemma 41, the log packing number can be bounded by

<!-- formula-not-decoded -->

On E 1 ∩ E 2 , we have I ≥ (1 -2 δ 2 ) P ≥ (1 -2 δ 2 ) e cN for some absolute constant c &gt; 0 . Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

Choosing δ = 1 / 2 and recalling ε g ≍ 1 , we obtain ε h ≳ (1 -Cε ) / L for some absolute constant C &gt; 0 , which concludes the proof.

## B.6 Proof of Theorem 7

We first provide an estimate for the capacity of two-layer feedforward networks to interpolate n samples.

Lemma 36. Suppose { x ( i ) } n i =1 i . i . d . ∼ N (0 , I d ) and let y ( i ) = ⟨ u , x t i ⟩ for arbitrary t i ∈ [ N ] and u ∈ S d -1 . Then, there exists an absolute constant c &gt; 0 such that for all m ≥ n and with probability at least c , there exist data dependent weights a , b ∈ R m and W ∈ R m × d , such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

Proof. The proof of Lemma 36 is an immediate consequence of two lemmas.

1. Lemma 37 shows that the inputs x (1) , . . . , x ( n ) can be projected to sufficiently separated scalar values with a unit vector v .
2. Lemma 38 perfectly fits n univariate samples using a two-layer ReLU neural network. When invoking this lemma, we use ∥ z ∥ 2 = O ( √ n ) and ϵ = Ω(1 /n 2 ) as given by Lemma 37.

The only missing piece is to upper bound ∥ y ∥ 2 appearing in the final bound of Lemma 38. To that end, we apply the following Markov inequality,

<!-- formula-not-decoded -->

As the statement of Lemma 37 holds with probability at least 1 3 , this suggests that the statement of Lemma 36 holds with probability at least 1 6 , concluding the proof.

̸

Lemma 37. Suppose { x ( i ) } n i =1 i . i . d . ∼ N (0 , I d ) . Then, with probability at least 1 / 3 , there exists some v ∈ S d -1 (dependent on { x ( i ) } ) such that for all i = j ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

Proof. The proof follows the probabilistic method. Sample v ∼ Unif ( S d -1 ) independent of { x ( i ) } . For each i = j , let

<!-- formula-not-decoded -->

and note that a i,j | v ∼ N (0 , 2) . We apply basic Gaussian anti-concentration to place a lower bound on the probability of any a i,j being close to zero,

̸

<!-- formula-not-decoded -->

̸

where the last inequality follows by taking ϵ = √ π/ (3 n 2 ) . Furthermore,

<!-- formula-not-decoded -->

by Markov's inequality. Combining the two events completes the proof.

̸

Lemma 38. Consider some z = ( z (1) , . . . , z ( n ) ) ⊤ ∈ R n and y = ( y (1) , . . . , y ( n ) ) ⊤ ∈ R n , such that ∣ ∣ z ( i ) -z ( j ) ∣ ∣ ≥ ϵ for all i = j . For simplicity, assume ϵ ≤ 1 . Then, there exists a two-layer ReLU neural network

<!-- formula-not-decoded -->

that satisfies g ( z ( i ) ) = y ( i ) for all i ∈ [ n ] , m = n , and

<!-- formula-not-decoded -->

Proof. Without loss of generality, we assume that z (1) ≤ · · · ≤ z ( n ) . Then, we define the neural network g as follows:

<!-- formula-not-decoded -->

One can verify by induction that g ( z ( i ) ) = y ( i ) for every i by noting that the slope of g is

<!-- formula-not-decoded -->

between ( z ( i -1) , y ( i -1) ) and ( z ( i ) , y ( i ) ) . From the above, we have w ′ i = 1 , ∥ ∥ b ′ ∥ ∥ 2 2 ≲ ∥ z ∥ 2 2 +1 , and ∥ a ′ ∥ 2 2 ≲ ∥ y ∥ 2 2 /ϵ 2 . For α = ( ( ∥ z ∥ 2 2 + n ) ϵ 2 / ∥ y ∥ 2 2 ) 1 / 4 , let u = α u ′ , w = w ′ /α , and b = b ′ /α . By homogeneity, the neural network with weights ( u , w , b ) has identical outputs to that of ( u ′ , w ′ , b ′ ) and satisfies (B.18), completing the proof.

We are now ready to present the proof of the sample complexity lower bound for RNNs.

Proof of Theorem 7. First, consider the case where d h &lt; n . Note that as a function of Uh = ( U → h → , U ← h ← ) , f y is L -Lipschitz with

<!-- formula-not-decoded -->

Using the AM-GM inequality,

<!-- formula-not-decoded -->

As a result, we have L ∥ U ∥ op ≤ e N c / 2 . By invoking Proposition 26, to obtain population risk less than some absolute constant c 3 &gt; 0 , we need

<!-- formula-not-decoded -->

This implies n ≥ d h ≥ Ω( N 1 -c ) . By taking c 1 in the theorem statement to be less than 1 -c , we obtain a contradiction. Therefore, we must have either a population risk at least c 3 or d h ≥ n .

Suppose now that d h ≥ n . We show that with constant probability, we can construct an RNN that interpolates the n training samples with norm independent of n . We simply let Θ → h = 0 , Θ ← h = 0 , U = 0 , and describe the construction of W L y , . . . , W 2 , W y , and ( b l ) in the following. Using the construction of Lemma 36, we can let

<!-- formula-not-decoded -->

where W ∈ R n × d , and a , b ∈ R n are given by Lemma 36. Then,

<!-- formula-not-decoded -->

For ( W l ) L y -1 l =3 , we let ( W l ) 11 = ( W l ) 22 = 1 , and choose the rest of the coordinates of W l to be zero. Therefore, the output of the l th layer is given by

<!-- formula-not-decoded -->

For the final layer, we let W L y = (1 , -1 , 0 , . . . , 0) . Using the fact that σ ( z ) -σ ( -z ) = z , we obtain

<!-- formula-not-decoded -->

We have found Θ such that ˆ R RNN n ( Θ ) = 0 and ∥ vec( Θ ) ∥ 2 2 ≤ O ( n 3 ) (recall that L y ≤ O (1) ). As a result, ˆ Θ ε must also satisfy ∥ ∥ ∥ vec( ˆ Θ ε ) ∥ ∥ ∥ 2 2 ≤ O ( n 3 ) .

On the other hand, notice that as a function of Uh = ( U → h → , U ← h ← ) , f y is L -Lipschitz with

<!-- formula-not-decoded -->

From Proposition 6, using the fact that ∥·∥ op ≤ ∥·∥ F and the AM-GM inequality, we obtain

<!-- formula-not-decoded -->

to achieve population risk less than some absolute constant c 3 &gt; 0 . Recall that log d h ≤ N c for some c &lt; 1 . The proof is completed by noticing that unless n ≥ Ω( N c 1 ) for some absolute constant c 1 &gt; 0 , ∥ ∥ ∥ vec( ˆ Θ ε ) ∥ ∥ ∥ 2 will always be less than the lower bound above, with some absolute constant probability c 2 &gt; 0 over the training set.

## C Auxiliary Lemmas

Lemma 39. Suppose A ∈ R d 1 × d 2 and B ∈ R d 2 × d 3 . Then, for all r, s ≥ 1 and p, q ≥ 1 such that 1 /p +1 /q = 1 , we have

<!-- formula-not-decoded -->

Proof. First, we note that for any vector b ∈ R d 2 we have

<!-- formula-not-decoded -->

where the last inequality holds for all conjugate indices p, q and follows from Hölder's inequality. We now have

<!-- formula-not-decoded -->

The next lemma follows from standard Gaussian integration.

<!-- formula-not-decoded -->

The following lemma combines two different techniques for establishing a packing number over the unit ball, the first construction uses volume comparison, whereas the second construction uses Maurey's sparsification lemma, both of which are well-established in the literature.

Lemma 41. Let P denote the ϵ -packing number of the unit ball in R d . We have

<!-- formula-not-decoded -->

Finally, the lemma below allows us to approximate arbitrary Lipschitz functions with two-layer feedforward networks.

Lemma 42 ([Bac17, Propositions 1 and 6]) . Suppose f : R d → R satisfies | f ( x ) | ≤ LR and | f ( x ) -f ( x ′ ) | ≤ L ∥ x -x ′ ∥ 2 for all x , x ′ ∈ R d with ∥ x ∥ 2 ≤ R and ∥ x ′ ∥ 2 ≤ R and some

constants L, R &gt; 0 . Then, for every ε &gt; 0 , there exists a positive integer m and W ∈ R m × d , b ∈ R m , and a ∈ R m , such that

<!-- formula-not-decoded -->

Additionally, we have

<!-- formula-not-decoded -->

## D Proof of Theorem 9

Let u be sampled uniformly from S d -1 independently from p = ( t 1 , x ) , and note that we have

<!-- formula-not-decoded -->

for all A ∈ A . From this point, we will simply use f for f A ( S n ) and W for W A ( S n ) . Next, we argue that the output weights of any algorithm in A satisfy

<!-- formula-not-decoded -->

for some coefficients ( α ( i ) k ) i ∈ [ n ] ,k ∈ [ m 1 ] . This is straightforward to verify for A ∈ A SP , as

<!-- formula-not-decoded -->

For A ∈ A ERM , note that ˆ L FFN only depends on w k through its projection on span( x (1) , . . . , x ( n ) ) . As a result, any minimum-norm ε -ERM would satisfy w k ∈ span( x (1) , . . . , x ( n ) ) .

Note that for n ≤ Nd , the span of x (1) , . . . , x ( n ) is n -dimensional with probability 1 over S n . Let v (1) , . . . , v ( n ) denote an orthonormal basis of span( x (1) , . . . , x ( n ) ) , and let V = ( v (1) , . . . , v ( n ) ) ⊤ ∈ R n × Nd . Recall that for the simple1STR model considered here, y j = y = 〈 u , x t q 〉 for j ∈ [ N ] . Then,

<!-- formula-not-decoded -->

where P t 1 ∈ R Nd × d has the form ( 0 d , . . . , I d ︸ ︷︷ ︸ t 1 , . . . , 0 d ) ⊤ . The conditioning above comes from the fact that via training, f and W can depend on u , but the prediction depends on x only through V x . Consequently, we replace the predicition of the FFN by the best predictor having access to u , t 1 , and V x . Note that t 1 , u , and V x are jointly independent, and the joint distribution ( ⟨ P t 1 u , x ⟩ , V x ) is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In particular, and

## E Experimental Details and Additional Results

In this section, we provide the details of our experimental setup, as well as additional results on the effect of q in Figure 3.

Architectures. We use a Transformer composed of a multihead attention layer with q heads, where each heads observes the entire d +( q +1) d e -dimensional input token, followed by a fully connected ReLU layer with width 100 . For the RNN, we use a simple bidirectional RNN with a hidden state size 500 × q , and a linear readout layer. For the FFN, we use a depth-3 fully connected ReLU network, where the first layer has width Ndq and the second layer has width 1000 . The output layer of the FFN has width N to match the input sequence.

Optimization. For Figures 1 and 3 we use online AdamW with weight decay 0 . 1 , where in Figure 1 we use a learning rate of 10 -3 and in Figure 3 we use a learning rate of 10 -4 . Each optimization step uses an independent batch size of 64 samples, and we track the test MSE loss using an independent set of 10 , 000 samples. For Figure 2 we use AdamW with weight decay 0 . 2 and learning rate 10 -3 on a fixed training set of 50 , 000 samples.

Data Generating Model. In all experiments, we sample x ∼ N ( 0 , I Nd ) . For Figures 1 and 2 we have q = 1 and define g ( x 1 ) = ⟨ u , x 1 ⟩ for a unit-norm u uniformly sampled from the unit sphere. For Figure 3 we let g ( x 1 , . . . , x q ) = 1 √ q ∑ q i =1 He 2 ( ⟨ u i , x i ⟩ ) where He 2 ( z ) = ( z 2 -1) / √ 2 is the normalized second Hermite polynomial. We use a non-linear g as this is a more challenging setting where e.g. Transformers require q heads by Theorem 4.

The code to reproduce all our experiments is provided at: https://github.com/mousavih/ transformers-separation .

Figure 3: Number of samples required to get to test MSE loss 0 . 88 while training with online AdamW for the quadratic q STR model explained in Appendix E with N = 7 . The gap increases with larger q . A closer theoretical analysis capturing the effect of large q can be an interesting direction for future work.

<!-- image -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We prove all the claims made in the abstract and introduction in this paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations throughout the paper and mostly in Section 6.

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

Justification: Our Definition 2 of the q STR model and our Assumptions 1, 2 and 3 are clearly stated.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA] .

Justification: Our contributions are theoretical and our limited numerical simulations are for illustration purposes only. We will include a link to the GitHub repository of our code in the de-anonymized version.

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

## Answer: [NA]

Justification: Our contributions are theoretical and our limited numerical simulations are for illustration purposes only. We will include a link to the GitHub repository of our code in the de-anonymized version.

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

Answer: [NA]

Justification: Our contributions are theoretical and our limited numerical simulations are for illustration purposes only. We will include a link to the GitHub repository of our code in the de-anonymized version.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA] .

Justification: Our contributions are theoretical and our limited numerical simulations are for illustration purposes only.

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

Answer: [NA]

Justification: Our contributions are theoretical and our limited numerical simulations are for illustration purposes only.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have followed the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA] .

Justification: Our contributions are theoretical and do not have immediate societal impacts.

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

Justification: Our contributions are theoretical.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: Our contributions are theoretical and we do not use any such assets.

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

Justification: Our contributions are theoretical and we do not introduce new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our contributions are theoretical and we do not perform this type of experiment. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our contributions are theoretical and we do not perform this type of experiment.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: We did not use LLMs.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.