## Vocabulary In-Context Learning in Transformers: Benefits of Positional Encoding

1

Qian Ma 1 Ruoxiang Xu 1 Yongqiang Cai 1 ∗ School of Mathematical Sciences, Beijing Normal University

## Abstract

Numerous studies have demonstrated that the Transformer architecture possesses the capability for in-context learning (ICL). In scenarios involving function approximation, context can serve as a control parameter for the model, endowing it with the universal approximation property (UAP). In practice, context is represented by tokens from a finite set, referred to as a vocabulary, which is the case considered in this paper, i.e. , vocabulary in-context learning (VICL). We demonstrate that VICL in single-layer Transformers, without positional encoding, does not possess the UAP; however, it is possible to achieve the UAP when positional encoding is included. Several sufficient conditions for the positional encoding are provided. Our findings reveal the benefits of positional encoding from an approximation theory perspective in the context of ICL.

## 1 Intruduction

Transformers have emerged as a dominant architecture in deep learning over the past few years. Thanks to their remarkable performance in language tasks, they have become the preferred framework in the natural language processing (NLP) field. A major trend in modern NLP is the development and integration of various black-box models, along with the construction of extensive text datasets. In addition, improving model performance in specific tasks through techniques such as in-context learning (ICL) [1, 2], chain of thought (CoT) [3, 4], and retrieval-augmented generation (RAG) [5] has become a significant research focus. While the practical success of these models and techniques is well-documented, the theoretical understanding of why they perform so well remains incomplete.

To explore the capabilities of Transformers in handling ICL tasks, it is essential to examine their approximation power. The universal approximation property (UAP) [6-9] has long been a key topic in the theoretical study of neural networks (NNs), with much of the focus historically on feed-forward neural networks (FNNs). Yun et al. [10] was the first to investigate the UAP of Transformers, demonstrating that any sequence-to-sequence function could be approximated by a Transformer network with fixed positional encoding. Luo et al. [11] highlighted that a Transformer with relative positional encoding does not possess the UAP. Meanwhile, Petrov et al. [12] explored the role of prompting in Transformers, proving that prompting a pre-trained Transformer can act as a universal functional approximator.

However, one limitation of these studies is that, in practical scenarios, the inputs to language models are derived from a finite set embedded in high-dimensional Euclidean space - commonly referred to as a vocabulary. Whether examining the work on prompts in [12] or the research on ICL in [13, 14], these studies assume inputs from the entire Euclidean space, which differs significantly from the discrete nature of vocabularies used in real-world applications.

∗ Email: caiyq.math@bnu.edu.cn. The first two authors made equal contributions to the paper.

## 1.1 Contributions

Starting with the connection between FNNs and Transformers, we turn to the finite restriction of vocabularies and study the benefits of positional encoding. Leveraging the UAP of FNNs, we explore the UAP of Transformers for ICL tasks in two scenarios: one where positional encoding is used and one where it is not. In both cases, the inputs are from a finite vocabulary. More specifically:

1. We first establish a connection between FNNs and Transformers in processing ICL tasks (Lemma 3). Using this lemma, we show that Transformers can function as universal approximators (Lemma 4), where the context serves as control parameters, while the weights and biases of the Transformer remain fixed.
2. When the vocabulary is finite and positional encoding is not used, we prove that single-layer Transformers cannot achieve the UAP for ICL tasks (Theorem 6).
3. However, when positional encoding is used, it becomes possible for single-layer Transformers to achieve the UAP (Theorem 7). In particular, for Transformers with ReLU or softmax activation functions, the conditions on the positional encoding are relaxed (Theorem 8).

## 1.2 Related Works

Universal approximation property. NNs, through multi-layer nonlinear transformations and feature extraction, are capable of learning deep feature representations from raw data. As neural networks gain prominence, theoretical investigations-especially into their UAP - have intensified. Related studies typically fall into two categories: those allowing arbitrary width with fixed depth [69], and those allowing arbitrary depth with bounded width [15-19]. Since our study builds on existing results regarding the approximation capabilities of FNNs, we focus on investigating the approximation abilities of single-layer Transformers in modulating context for ICL tasks. Consequently, our work relies more on the findings from the first category of research. The realization of the UAP depends on the architecture of the network itself, providing constructive insights for exploring the connection between FNNs and Transformers. Recently, Petrov et al. [12] also explored UAP in the context of ICL, but without considering vocabulary constraints or positional encodings.

Transformers. The Transformer is a widely used neural network architecture for modeling sequences [20-25]. This non-recurrent architecture relies entirely on the attention mechanism to capture global dependencies between inputs and outputs [20]. The neural sequence transduction model is typically structured using an encoder-decoder framework [26, 27].

Without positional encoding, the Transformer can be viewed as a stack of N blocks, each consisting of a self-attention layer followed by a feed-forward layer with skip connections. In this paper, we focus on the case of a single-layer self-attention sequence encoder.

In-context learning. The Transformer has demonstrated remarkable performance in the field of NLP, and large language models (LLMs) are gaining increasing popularity. ICL has emerged as a new paradigm in NLP, enabling LLMs to make better predictions through prompts provided within the context [2, 28-31]. ICL delivers high performance with high-quality data at a lower cost [32-34]. It enhances retrieval-augmented methods by prepending grounding documents to the input [35] and can effectively update or refine the model's knowledge base through well-designed prompts [36].

Positional Encoding. The following explanation clarifies the significance of incorporating positional encoding into the Transformer architecture. RNNs capture sequential order by encoding the changes in hidden states over time. In contrast, for Transformers, the self-attention mechanism is permutation equivariant, meaning that for any model f , any permutation matrix π , and any input x , the following holds: f ( π ( x )) = π ( f ( x )) .

We aim to explore the impact of positional encoding on the performance of a single-layer Transformer when performing ICL tasks with a finite vocabulary. Therefore, we focus on analyzing existing positional encoding methods. There are fundamental methods for encoding positional information in a sequence within the Transformer: absolute positional encodings (APEs), e.g. [37, 25, 38, 39], relative positional encodings (RPEs), e.g. [40, 41, 39] and rotary positional embedding (RoPE) [42]. The commonly used APE is implemented by directly adding the positional encodings to the word embeddings, and we follow this implementation.

UAP of ICL. To understand the mechanism of ICL, various explanations have been proposed, including those based on Bayesian theory [43, 44] and gradient descent theory [45]. Fine-tuning the Transformer through ICL alters the presentation of the input rather than the model parameters, which is driven by successful few-shot and zero-shot learning [46, 47]. This success raises the question of whether we can achieve the UAP through context adjustment.

Yun et al. [10] demonstrated that Transformers can serve as universal sequence-to-sequence approximators, while Alberti et al. [48] extended the UAP to architectures with non-standard attention mechanisms. However, their implementations allow the internal parameters of the Transformers to vary, which does not fully capture the nature of ICL. In contrast, Likhosherstov et al. [49] showed that while the parameters of self-attention remain fixed, various sparse matrices can be approximated by altering the inputs. Fixing self-attention parameters aligns more closely with practical scenarios and provides valuable insights for our work. However, this approach has the limitation of excluding the full Transformer architecture. Furthermore, Deora et al. [50] illustrated the convergence and generalization of single-layer multi-head self-attention models trained using gradient descent, supporting the feasibility of our research by emphasizing the robust generalization of Transformers. Nevertheless, Petrov et al. [51] indicated that the presence of a prefix does not alter the attention focus within the context, prompting us to explore variations in input context and introduce flexibility in positional encoding.

## 1.3 Outline

We will introduce the notations and background results in Section 2. Section 3 addresses the case where the vocabulary is finite and positional encoding is not used. Section 4 discusses the benefits of using positional encoding. A summary is provided in Section 5. All proofs of lemmas and theorems are provided in the Appendix.

## 2 Background Materials

We consider the approximation problem as follows. Given a fixed Transformer network, for any target continuous function f : K → R d y with a compact domain K ⊂ R d x , we aim to adjust the content of the context so that the output of the Transformer network can approximate f . First, we present the concrete forms and notations for the inputs of ICL, FNNs, and Transformers.

## 2.1 Notations

Input of in-context learning. In the ICL task, the given n demonstrations are denoted as z ( i ) = ( x ( i ) , y ( i ) ) for i = 1 , 2 , ..., n , where x ( i ) ∈ R d x and y ( i ) ∈ R d y . Unlike the setting in [13, 14] where y ( i ) was related to x ( i ) (for example y ( i ) = ϕ ( x ( i ) ) for some function ϕ ), we do not assume any correspondence between x ( i ) and y ( i ) , i.e. , x ( i ) and y ( i ) are chosen freely. To predict the target at a query vector x ∈ R d x or z = ( x, 0) ∈ R d x + d y , we define the input matrix Z as follows:

<!-- formula-not-decoded -->

Furthermore, let P : N + → R d x + d y represent a positional encoding function, and define P ( i ) := P ( i ) . Denote the demonstrations with positional encoding as z ( i ) P := z ( i ) + P ( i ) and z P := z + P ( n +1) . The context with positional encoding can then be represented as:

<!-- formula-not-decoded -->

Additionally, we denote:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Feed-forward neural networks. One-hidden-layer FNNs have sufficient capacity to approximate continuous functions on any compact domain. In this article, all the FNNs we refer to and use are one-hidden-layer networks. We denote a one-hidden-layer FNN with activation function σ as N σ , and the set of all such networks is denoted as N σ , i.e. ,

<!-- formula-not-decoded -->

For element-wise activations, such as ReLU, the above notation is well-defined. However, for nonelement-wise activation functions, which are not widely used but considered in this article, especially the softmax activation, we need to give more details for the notation:

<!-- formula-not-decoded -->

Transformers. We define the general attention mechanism following [13, 14] as:

<!-- formula-not-decoded -->

where V, Q, K are the value, query, and key matrices in R ( d x + d y ) × ( d x + d y ) , respectively. M = diag ( I n , 0) is the mask matrix in R ( n +1) × ( n +1) , and σ is the activation function. Note that the context vectors z ( i ) are asymmetric and do not include the query vector z itself; therefore, we introduce a mask matrix M following the design of [13, 14]. Here the softmax activation of a matrix G ∈ R m × n is defined as:

<!-- formula-not-decoded -->

With this formulation of the general attention mechanism, we can define a single-layer Transformer without positional encoding as:

<!-- formula-not-decoded -->

where [ a : b, c : d ] denotes the submatrix from the a -th row to the b -th row and from the c -th column to the d -th column. If a = b (or c = d ), the row (or column) index is reduced to a single number. Similarly to the notation for FNNs, T σ denotes the set of all T σ with different parameters.

Vocabulary. In the above notations, the tokens in context of ICL are general and unrestricted. When we refer to a 'vocabulary', we mean that the tokens are drawn from a finite set. More specifically, we refer to it as VICL if all input vectors z ( i ) come from a finite vocabulary V = V x ×V y ⊂ R d x × R d y . In this case, we use subscript ∗ , i.e. , T σ ∗ ( x ; X,Y ) , to represent the Transformer T σ ( x ; X,Y ) defined in equation (9), and denote the set of such Transformers as T σ ∗ :

<!-- formula-not-decoded -->

To facilitate the simplification of VICL analysis, we denote a FNN with a finite set of weights as N σ ∗ , and the corresponding set of such networks as N σ ∗ . More specifically, when the activation function is an elementwise activation, we denote:

<!-- formula-not-decoded -->

where A ⊂ R d y , W ⊂ R d x , and B ⊂ R are finite sets. When the activation function is softmax , we denote:

<!-- formula-not-decoded -->

where A , W and B are defined as in the previous context.

Positional encoding. When positional encoding P is involved, we add the subscript P , i.e. ,

<!-- formula-not-decoded -->

Note that the context length n in T , T ∗ and T ∗ , P

<!-- formula-not-decoded -->

We present all our notations in Table 1 in Appendix A for easy reference.

## 2.2 Universal Approximation Property

The vanilla form of the UAP for FFNs plays a crucial role in our study. Before we state this property as a formal lemma, we put forward the following assumption first, which is similar to the one in [14] and is used to simplify the analysis of Transformers.

Assumption 1. The matrices Q, K, V ∈ R ( d x + d y ) × ( d x + d y ) have the following sparse partition:

<!-- formula-not-decoded -->

where B, C, D ∈ R d x × d x , E ∈ R d x × d y , F ∈ R d y × d x and U ∈ R d y × d y . Furthermore, the matrices B, C and U are non-singular, and the matrix F = 0 .

In addition, we assume the element-wise activation σ is non-polynomial, locally bounded, and continuous. It is worth noting that a randomly initialized n × n matrix is non-singular with probability one, so it is acceptable to assume that the matrices B,C and U are non-singular. Moreover, the assumption F = 0 can be relaxed, which will be discussed in Appendix F. Here, we have slightly strengthened it for the sake of computational convenience.

Lemma 2 (UAP of FNNs [9]) . Let σ : R → R be a non-polynomial, locally bounded, piecewise continuous activation function. For any continuous function f : R d x → R d y defined on a compact domain K , and for any ε &gt; 0 , there exist k ∈ N + , A ∈ R d y × k , b ∈ R k , and W ∈ R k × d x such that

<!-- formula-not-decoded -->

The theorem presented above is well-known and primarily applies to activation functions operating element-wise. However, it can be readily extended to the case of the softmax activation function. In fact, this can be achieved using NNs with exponential activation functions. The specific approach for this generalization is detailed in Appendix B.

## 2.3 Feed-forward neural networks and Transformers

It is important to emphasize the connection between FNNs and Transformers, which will be represented in the following lemmas and are crucial for establishing our main theory.

Lemma 3. Let σ : R → R be a non-polynomial, locally bounded, piecewise continuous activation function, and T σ be a single-layer Transformer satisfying Assumption 1. For any one-hidden-layer network N σ : R d x -1 → R d y ∈ N σ with n hidden neurons, there exist matrices X ∈ R d x × n and Y ∈ R d y × n such that

<!-- formula-not-decoded -->

There is a difference in the input dimensions of T σ and N σ , as the latter includes a bias dimension absent in the former. To connect the two inputs, ˜ x and x , we use a tilde, where ˜ x is formed by augmenting x with an additional one appended to the end.

By employing the structure of K , Q and V in equation (14), the output forms of the Transformer T σ ( ˜ x ; X,Y ) can be simplified as follows:

<!-- formula-not-decoded -->

Comparing this with the output form of FNNs, i.e. , N σ ( x ) = Aσ ( Wx + b ) , it becomes evident that setting X = ( C ⊤ B ) -1 [ W b ] ⊤ and Y = U -1 A is sufficient to finish the proof.

It can be observed that the form in equation (17) exhibits the structure of an FNN. Consequently, Lemma 3 implies that single-layer Transformers T σ in the context of ICL and FNNs N σ are equivalent. However, this equivalence does not hold for the case of softmax activation due to differences in the normalization operations between FNNs and Transformers. Therefore, in the subsequent sections of this article, we employ different analytical methods to address the two types of activation functions.

Moreover, the equivalence in equation (16) suggests that the context in Transformers can act as a control parameter for the model, thereby endowing it with the UAP.

## 2.4 Universal Approximation Property of In-context Learning

We now present the UAP of Transformers in the context of ICL.

Lemma 4. Let σ : R → R be a non-polynomial, locally bounded, piecewise continuous activation function or softmax function, and T σ be a single-layer Transformer satisfying Assumption 1, and K be a compact domain in R d x -1 . Then for any continuous function f : K → R d y and any ε &gt; 0 , there exist matrices X ∈ R d x × n and Y ∈ R d y × n such that

<!-- formula-not-decoded -->

For the case of element-wise activation, the result follows directly by combining Lemma 2 and Lemma 3. However, for the softmax activation, the normalization operation requires an additional technique in the proof. The core idea is to construct an FNN with exponential activation functions, incorporating an additional neuron to handle the normalization effect. Detailed proofs are provided in Appendix B. Similar results have been obtained in recent work [12], though via different methodologies.

## 3 The Non-Universal Approximation Property of N σ ∗ and T σ ∗

One key aspect of ICL is that the context can act as a control parameter for the model. We now consider the case where the tokens in context are restricted to a finite vocabulary. A natural question arises: can single-layer Transformers with a finite vocabulary, i.e. , T σ ∗ , still achieve the UAP via ICL? We first analyze N σ ∗ for simplicity, and then use the established connection between FNNs and Transformers to extend the result to T σ ∗ . The answer is that N σ ∗ cannot achieve the UAP because of the restriction of finite parameters.

For element-wise activations, the span of N σ ∗ , span {N σ ∗ } , forms a finite-dimensional function space. According to results from functional analysis, span {N σ ∗ } is closed under the function norm (see e.g. Theorem 1.21 of [52] or Corollary C.4 of [53]). This implies that the set of functions that can be approximated by span {N σ ∗ } is precisely the set of functions within span {N σ ∗ } . Consequently, any function not in span {N σ ∗ } cannot be arbitrarily approximated, meaning that the UAP cannot be achieved.

For softmax networks, the normalization operation introduces further limitations. Even though N softmax ∗ consists of weighted units drawn from a fixed finite collection of basic units, normalization prevents these networks from being simple linear combinations of one another. While the span of N softmax ∗ might theoretically have infinite dimensionality, its expressive power remains constrained.

To better understand the functional behavior of N softmax ∗ , we introduce the structural Proposition 12 whose details are provided in Appendix C. The proposition characterizes the maximum number of zero points that functions in this class can exhibit, and the result can be established via mathematical induction. This observation motivates the following lemma, which formally states the non-universal approximation property of N σ ∗ .

Lemma 5. The function class N σ ∗ , with a non-polynomial, locally bounded, piecewise continuous element-wise activation function or softmax activation function σ , cannot achieve the UAP. Specifically, there exist a compact domain K ⊂ R d x , a continuous function f : K → R d y , and ε 0 &gt; 0 such that

<!-- formula-not-decoded -->

In the proof of Lemma 5, we demonstrated through Proposition 12 that the number of zeros of N softmax ∗ depends solely on a finite set of parameters and constitutes a bounded quantity. Functions

can be explicitly constructed whose number of zeros exceeds this bound, thereby preventing their approximation within N softmax ∗ .

By leveraging the connection between FNNs and Transformers, we establish Theorem 6 to demonstrate that T σ ∗ cannot achieve the UAP.

Theorem 6. The function class T σ ∗ , with a non-polynomial, locally bounded, piecewise continuous element-wise activation function or softmax activation function σ and every T σ ∈ T σ ∗ satisfies Assumption 1, cannot achieve the UAP. Specifically, there exist a compact domain K ⊂ R d x , a continuous function f : K → R d y , and ε 0 &gt; 0 such that

<!-- formula-not-decoded -->

The result for element-wise activations follows directly from the application of Lemma 3 and Lemma 5. However, the case of the softmax activation requires additional techniques to account for the normalization effect. The proof, which utilizes Proposition 12 once again, is presented in the Appendix C. It is worth noting that Theorem 6 holds even without imposing any constraints on the V , Q and K (e.g., the sparse partition described in equation (14)). Further details can be found in Appendix F.

## 4 The Universal Approximation Property of T σ ∗ , P

After establishing that neither N σ ∗ nor T σ ∗ can achieve the UAP, we aim to leverage a key feature of Transformers: their ability to incorporate APEs during token input. This motivates us to investigate whether T σ ∗ , P can realize the UAP.

The answer is affirmative. To support our constructive proof, we invoke the Kronecker Approximation Theorem as a key auxiliary tool, which will be stated as Lemma 13. This result ensures the density of certain structured sets in R n under mild arithmetic conditions. The formal statement and discussion of this theorem are provided in Appendix D.

Theorem 7. Let T σ ∗ , P be the class of functions T σ ∗ , P satisfying Assumption 1, with a non-polynomial, locally bounded, piecewise continuous element-wise activation function σ , the subscript refers to the finite vocabulary V = V x ×V y , P = P x ×P y represents the positional encoding map, and denote a set S as:

If S is dense in R d x , { 1 , -1 , √ 2 , 0 } d y ⊂ V y and P y = 0 , then T σ ∗ , P can achieve the UAP. More specifically, given a network T σ ∗ , P , then for any continuous function f : R d x -1 → R d y defined on a compact domain K and ε &gt; 0 , there always exist X ∈ R d x × n and Y ∈ R d y × n from the vocabulary V , i.e. , x ( i ) ∈ V x , y ( i ) ∈ V y , with some length n ∈ N + such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We provide a constructive proof in Appendix C, and here we only demonstrate the proof idea by considering the specific case of d y = 1 and assuming the matrice U in the Transformer T σ ∗ , P is an identity matrice. In this case, the Transformer can be simplified to an FNN N σ ∗ , that is

<!-- formula-not-decoded -->

which is similar to the calculation in equation (17). The UAP of FNNs shown in Lemma 2 implies that the target function f can be approximated by an FNN with k hidden neurons, that is

<!-- formula-not-decoded -->

Since we are considering a continuous activation function σ , we can conclude that slightly perturbing the parameters A and W will lead to new FNN that can still approximate f . This motivates us to construct a proof using the property that each ˜ w i ∈ R d x can be approximated by vectors

x P B ⊤ C, x P ∈ S = V x + P x , and each a i ∈ R can be approximated by q i √ 2 ± l i , with positive integers q i and l i .

For ease of exposition, we will first show how to construct X and Y , so as to approximate the first term in the summation in equation (24), namely a 1 σ ( ˜ w 1 · ˜ x ) . By Lemma 13, we may choose positive integers q and l such that q √ 2 ± l is sufficiently close to a 1 . Consider the first token in the context, since the positional encoding is fixed, i.e. , V x + P (1) is a finite set, then one of two cases must occur:

1. if there exists a token x (1) ∈ V x for which x (1) + P (1) is sufficiently close to ˜ w 1 , then we declare this position 'valid';

2. otherwise, we declare the position 'invalid', and choose any x (1) ∈ V x , and set y (1) = 0 so as to nullify its contribution in the sum.

We then proceed inductively: having handled the first token, we construct the second token in exactly the same manner, then the third, and so on, until we have identified q + l valid positions. Because S is dense in R d x and q, l are finite, this selection process necessarily terminates after finitely many steps. Finally, we assign y ( i ) = √ 2 for q of the valid positions and y ( i ) = ± 1 for other l valid positions. Up to now, we have built a partial context that enables the output of T σ ∗ , P to approximate a 1 σ ( ˜ w 1 · ˜ x ) with arbitrarily small error. Once we have approximated a 1 σ ( ˜ w 1 · ˜ x ) , we can then approximate a 2 σ ( ˜ w 2 · ˜ x ) , · · · , a k σ ( ˜ w k · ˜ x ) in finitely many steps, thereby completing the construction of the full context X and Y . In the proof idea above, we take the density of the set S in R d x as a fundamental assumption. V x contains only finitely many elements, rendering it bounded. For S to be dense in the entire space, P x must be unbounded. We further extend the above approach to discuss whether other forms of positional encoding can also achieve the UAP in Appendix E.4.

Next, we relax this requirement, eliminating the need for P x to be unbounded, making the conditions more aligned with practical scenarios. Particularly, we consider the specific activation function in the following theorem, where the notations not explicitly mentioned remain consistent with those in Theorem 7. We present an informal version, and the formal version is provided in Appendix E.

Theorem 8 (Informal Version) . If the set S is dense in [ -1 , 1] d x , then T ReLU ∗ , P is capable of achieving the UAP. Additionally, if S is only dense in a neighborhood B ( w ∗ , δ ) of a point w ∗ ∈ R d x with radius δ &gt; 0 , then the class of transformers with exponential activation, i.e. , T exp ∗ , P , is capable of achieving the UAP.

The density condition on S is significantly refined here, which we will discuss in the later remark. This improvement is possible because the proof of Theorem 7 relies directly on the UAP of FNNs, where the weights take values from the entire parameter space. However, for FNNs with specific activations, we can restrict the weights to a small set without losing the UAP.

For ReLU networks, we can use the positive homogeneity property, i.e. , A ReLU ( W ˜ x ) = λ -1 A ReLU ( λW ˜ x ) for any λ &gt; 0 , to restrict the weight matrix W . In fact, the restriction that all elements of W take values in the interval [ -1 , 1] does not affect the UAP of ReLU FNNs because the scale of W can be recovered by adjusting the scale of A via choosing a proper λ .

For exponential networks, the condition on S is much weaker than in the ReLU case. This relaxation is nontrivial, and the proof stems from a property of the derivatives of exponential functions. Consider the exponential function exp( w · x ) as a function of w ∈ B ( w ∗ , δ ) , and denote it as h ( w ) ,

<!-- formula-not-decoded -->

where w i and x i ∈ R are the components of w and x , respectively. Calculating the partial derivatives of h ( w ) , we observe the following relations:

<!-- formula-not-decoded -->

where α = ( α 1 , . . . , α d ) ∈ N d is the index vector representing the order of partial derivatives, and | α | := α 1 + · · · + α d . This relationship allows us to link exponential FNNs to polynomials since any polynomial P ( x ) can be represented in the following form:

<!-- formula-not-decoded -->

where a α are the coefficients of the polynomials, Λ is a finite set of indices, and the partial derivatives can be approximated by finite differences, which are FNNs. For example, the first-order partial derivative ∂h ∂w 1 ∣ ∣ w = w ∗ = x 1 h ( w ∗ ) can be approximated by the following difference with a small nonzero number λ ∈ (0 , δ ) ,

<!-- formula-not-decoded -->

This is an exponential FNN with two neurons. Finally, employing the well-known Stone-Weierstrass Theorem, which states that any continuous function f on compact domains can be approximated by polynomials, and combining the above relations between FNNs and polynomials, we can establish the UAP of exponential FNNs with weight constraints. When y ( i ) = f ( x ( i ) ) (referred to as meaningfully related), the conclusion still holds in standard ICL, provided that V y satisfies certain conditions. A brief proof is provided in Theorem 15.

Remark 9. When discussing density, one of the most immediate examples that comes to mind is the density of rational numbers in R . How can we effectively enumerate rational numbers? The work by [54] introduces an elegant method for enumerating positive rational numbers, synthesizing ideas from [55] and [56]. It demonstrates the computational feasibility of enumeration through an effective algorithm. Thus, we assume that positional encodings can be implemented using computer algorithms, such as iterative functions. Furthermore, since positional encodings vary across different positions, they encapsulate semantic information concerning both position and order.

## 5 Conclusion

In this paper, we establish a connection between FNNs and Transformers through ICL. By leveraging the UAP of FNNs, we demonstrate that the UAP of ICL holds when the context is selected from the entire vector space. When the context is drawn from a finite set, we explore the approximation power of VICL, showing that the UAP is achievable only when appropriate positional encodings are incorporated, highlighting their importance.

In our work, we consider Transformers with input sequences of arbitrary length, implying that the positional encoding P x consists of a countably infinite set of elements. In Theorem 7, we assume a strong density condition, which is later relaxed in Theorem 8. However, in practical applications, input sequences are finite and are typically truncated for computational feasibility. This shift allows our conclusions to be interpreted through an approximation lens, where the objective is to approximate functions within a specified error margin, rather than achieving infinitesimal precision. Additionally, to achieve the UAP, it is insightful to compare the function approximation capabilities of our approach (outlined in Lemma 4) with the direct use of FNNs, particularly when the Transformer parameters are trainable.

It is important to note that this paper is limited to single-layer Transformers with APEs, and the main results (Theorem 7 and Theorem 8) focus on element-wise activations. Future research should extend these findings to multi-layer Transformers, general positional encodings (such as RPEs and RoPE), and softmax activations. For softmax Transformers, our analysis in Sections 2 and 3 highlighted their connection to Transformers with exponential activations. However, extending this connection to the scenario in Section 4 proves challenging and requires more sophisticated techniques.

Although this paper primarily addresses theoretical issues, we believe our results provide valuable insights for practitioners. In Remark 9, we observe that algorithms using function composition to enumerate dense numbers in R could inspire positional encodings via compositions of fixed functions, similar to RNN approaches. RNNs capture the sequential nature of information by integrating word order. However, existing research on RNNs has not explored the denseness of the sets formed by their hidden states, which we hope will inspire future experimental research. Lastly, our construction for Theorem 7 relies on the sparse partition assumption in equation (14), whose practical validity remains uncertain and requires future exploration.

In fact, Tack et al. [57], Hao et al. [58] on continuous CoTs and continuous states are closely related to our work - specifically, leveraging positional encoding to enable Transformers to achieve the UAP for functions whose domain is a finite set while the range covers the entire Euclidean space. Moreover, Xiao et al. [59] propose an approach for automatically adjusting prompts for function fitting, which is also related to our theoretical findings. Therefore, with further research, our theory holds practical significance.

## Acknowledgements

This work was partially supported by the National Key R&amp;D Program of China under grants 2024YFF0505501 and the Fundamental Research Funds for the Central Universities. We thank the anonymous reviewers for their valuable comments and helpful suggestions. We gratefully acknowledge the Scholar Award from the Neural Information Processing Foundation.

## References

- [1] Q. Dong, L. Li, D. Dai, C. Zheng, J. Ma, R. Li, H. Xia, J. Xu, Z. Wu, B. Chang, X. Sun, L. Li, and Z. Sui, 'A survey on in-context learning,' arXiv preprint arXiv:2301.00234 , 2024.
- [2] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei, 'Language models are few-shot learners,' in Advances in Neural Information Processing Systems , 2020.
- [3] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. V. Le, and D. Zhou, 'Chainof-thought prompting elicits reasoning in large language models,' in Advances in Neural Information Processing Systems , 2022.
- [4] Z. Chu, J. Chen, Q. Chen, W. Yu, T. He, H. Wang, W. Peng, M. Liu, B. Qin, and T. Liu, 'Navigate through enigmatic labyrinth a survey of chain of thought reasoning: Advances, frontiers and future,' in Annual Meeting of the Association for Computational Linguistics , 2024.
- [5] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, M. Wang, and H. Wang, 'Retrieval-augmented generation for large language models: A survey,' arXiv preprint arXiv:2312.10997 , 2024.
- [6] G. Cybenko, 'Approximation by superpositions of a sigmoidal function,' Mathematics of control, signals and systems , vol. 2, pp. 303-314, 1989.
- [7] K. Hornik, M. Stinchcombe, and H. White, 'Multilayer feedforward networks are universal approximators,' Neural Networks , vol. 2, pp. 359-366, 1989.
- [8] K. Hornik, 'Approximation capabilities of multilayer feedforward networks,' Neural Networks , vol. 4, pp. 251-257, 1991.
- [9] M. Leshno, V. Y. Lin, A. Pinkus, and S. Schocken, 'Multilayer feedforward networks with a nonpolynomial activation function can approximate any function,' Neural Networks , vol. 6, pp. 861-867, 1993.
- [10] C. Yun, S. Bhojanapalli, A. S. Rawat, S. Reddi, and S. Kumar, 'Are transformers universal approximators of sequence-to-sequence functions?' in International Conference on Learning Representations , 2020.
- [11] S. Luo, S. Li, S. Zheng, T.-Y. Liu, L. Wang, and D. He, 'Your transformer may not be as powerful as you expect,' in Advances in Neural Information Processing Systems , 2022.
- [12] A. Petrov, P. Torr, and A. Bibi, 'Prompting a pretrained transformer can be a universal approximator,' in International Conference on Machine Learning , 2024.
- [13] K. Ahn, X. Cheng, H. Daneshmand, and S. Sra, 'Transformers learn to implement preconditioned gradient descent for in-context learning,' in Advances in Neural Information Processing Systems , 2024.
- [14] X. Cheng, Y. Chen, and S. Sra, 'Transformers implement functional gradient descent to learn non-linear functions in context,' in International Conference on Machine Learning , 2024.
- [15] Z. Lu, H. Pu, F. Wang, Z. Hu, and L. Wang, 'The expressive power of neural networks: A view from the width,' in Advances in Neural Information Processing Systems , 2017.
- [16] S. Park, C. Yun, J. Lee, and J. Shin, 'Minimum width for universal approximation,' in International Conference on Learning Representations , 2021.
- [17] Y. Cai, 'Achieve the minimum width of neural networks for universal approximation,' in International Conference on Learning Representations , 2023.
- [18] L. Li, Y. Duan, G. Ji, and Y. Cai, 'Minimum width of leaky-relu neural networks for uniform universal approximation,' in arXiv:2305.18460v3 , 2024.

- [19] Y. Cai, 'Vocabulary for universal approximation: a linguistic perspective of mapping compositions,' in International Conference on Machine Learning , 2024.
- [20] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. ukasz Kaiser, and I. Polosukhin, 'Attention is all you need,' in Advances in Neural Information Processing Systems , 2017.
- [21] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, 'Bert: Pre-training of deep bidirectional transformers for language understanding,' in Annual Meeting of the Association for Computational Linguistics , 2019.
- [22] Z. Yang, Z. Dai, Y. Yang, J. Carbonell, R. R. Salakhutdinov, and Q. V. Le, 'Xlnet: Generalized autoregressive pretraining for language understanding,' in Advances in Neural Information Processing Systems , 2019.
- [23] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, 'Exploring the limits of transfer learning with a unified text-to-text transformer,' Journal of Machine Learning Research , vol. 21, pp. 1-67, 2020.
- [24] L. Zhenzhong, C. Mingda, G. Sebastian, G. Kevin, S. Piyush, and S. Radu, 'Albert: A lite bert for selfsupervised learning of language representations,' in International Conference on Learning Representations , 2021.
- [25] X. Liu, H.-F. Yu, I. Dhillon, and C.-J. Hsieh, 'Learning to encode position for transformer with continuous dynamical model,' in International Conference on Machine Learning , 2020.
- [26] D. Bahdanau, K. Cho, and Y. Bengio, 'Neural machine translation by jointly learning to align and translate,' arXiv preprint arXiv:1409.0473 , 2014.
- [27] I. Sutskever, O. Vinyals, and Q. V. Le, 'Sequence to sequence learning with neural networks,' in Advances in Neural Information Processing Systems , 2014.
- [28] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, P. Schuh, K. Shi, S. Tsvyashchenko, J. Maynez, A. Rao, P. Barnes, Y. Tay, N. Shazeer, V. Prabhakaran, E. Reif, N. Du, B. Hutchinson, R. Pope, J. Bradbury, J. Austin, M. Isard, G. Gur-Ari, P. Yin, T. Duke, A. Levskaya, S. Ghemawat, S. Dev, H. Michalewski, X. Garcia, V. Misra, K. Robinson, L. Fedus, D. Zhou, D. Ippolito, D. Luan, H. Lim, B. Zoph, A. Spiridonov, R. Sepassi, D. Dohan, S. Agrawal, M. Omernick, A. M. Dai, T. S. Pillai, M. Pellat, A. Lewkowycz, E. Moreira, R. Child, O. Polozov, K. Lee, Z. Zhou, X. Wang, B. Saeta, M. Diaz, O. Firat, M. Catasta, J. Wei, K. Meier-Hellstern, D. Eck, J. Dean, S. Petrov, and N. Fiedel, 'Palm: Scaling language modeling with pathways,' Journal of Machine Learning Research , vol. 24, pp. 1-113, 2023.
- [29] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample, 'Llama: Open and efficient foundation language models,' arXiv preprint arXiv:2302.13971 , 2023.
- [30] OpenAI, J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, R. Avila, I. Babuschkin, S. Balaji, V. Balcom, P. Baltescu, H. Bao, M. Bavarian, J. Belgum, I. Bello, J. Berdine, G. Bernadett-Shapiro, C. Berner, L. Bogdonoff, O. Boiko, M. Boyd, A.-L. Brakman, G. Brockman, T. Brooks, M. Brundage, K. Button, T. Cai, R. Campbell, A. Cann, B. Carey, C. Carlson, R. Carmichael, B. Chan, C. Chang, F. Chantzis, D. Chen, S. Chen, R. Chen, J. Chen, M. Chen, B. Chess, C. Cho, C. Chu, H. W. Chung, D. Cummings, J. Currier, Y. Dai, C. Decareaux, T. Degry, N. Deutsch, D. Deville, A. Dhar, D. Dohan, S. Dowling, S. Dunning, A. Ecoffet, A. Eleti, T. Eloundou, D. Farhi, L. Fedus, N. Felix, S. P. Fishman, J. Forte, I. Fulford, L. Gao, E. Georges, C. Gibson, V. Goel, T. Gogineni, G. Goh, R. Gontijo-Lopes, J. Gordon, M. Grafstein, S. Gray, R. Greene, J. Gross, S. S. Gu, Y. Guo, C. Hallacy, J. Han, J. Harris, Y. He, M. Heaton, J. Heidecke, C. Hesse, A. Hickey, W. Hickey, P. Hoeschele, B. Houghton, K. Hsu, S. Hu, X. Hu, J. Huizinga, S. Jain, S. Jain, J. Jang, A. Jiang, R. Jiang, H. Jin, D. Jin, S. Jomoto, B. Jonn, H. Jun, T. Kaftan, Łukasz Kaiser, A. Kamali, I. Kanitscheider, N. S. Keskar, T. Khan, L. Kilpatrick, J. W. Kim, C. Kim, Y. Kim, J. H. Kirchner, J. Kiros, M. Knight, D. Kokotajlo, Łukasz Kondraciuk, A. Kondrich, A. Konstantinidis, K. Kosic, G. Krueger, V. Kuo, M. Lampe, I. Lan, T. Lee, J. Leike, J. Leung, D. Levy, C. M. Li, R. Lim, M. Lin, S. Lin, M. Litwin, T. Lopez, R. Lowe, P. Lue, A. Makanju, K. Malfacini, S. Manning, T. Markov, Y. Markovski, B. Martin, K. Mayer, A. Mayne, B. McGrew, S. M. McKinney, C. McLeavey, P. McMillan, J. McNeil, D. Medina, A. Mehta, J. Menick, L. Metz, A. Mishchenko, P. Mishkin, V. Monaco, E. Morikawa, D. Mossing, T. Mu, M. Murati, O. Murk, D. Mély, A. Nair, R. Nakano, R. Nayak, A. Neelakantan, R. Ngo, H. Noh, L. Ouyang, C. O'Keefe, J. Pachocki, A. Paino, J. Palermo, A. Pantuliano, G. Parascandolo, J. Parish, E. Parparita, A. Passos, M. Pavlov, A. Peng, A. Perelman, F. de Avila Belbute Peres, M. Petrov, H. P. de Oliveira Pinto, Michael, Pokorny, M. Pokrass, V. H. Pong, T. Powell, A. Power, B. Power, E. Proehl, R. Puri, A. Radford, J. Rae, A. Ramesh, C. Raymond, F. Real, K. Rimbach, C. Ross, B. Rotsted, H. Roussez, N. Ryder, M. Saltarelli,

T. Sanders, S. Santurkar, G. Sastry, H. Schmidt, D. Schnurr, J. Schulman, D. Selsam, K. Sheppard, T. Sherbakov, J. Shieh, S. Shoker, P. Shyam, S. Sidor, E. Sigler, M. Simens, J. Sitkin, K. Slama, I. Sohl, B. Sokolowsky, Y. Song, N. Staudacher, F. P. Such, N. Summers, I. Sutskever, J. Tang, N. Tezak, M. B. Thompson, P. Tillet, A. Tootoonchian, E. Tseng, P. Tuggle, N. Turley, J. Tworek, J. F. C. Uribe, A. Vallone, A. Vijayvergiya, C. Voss, C. Wainwright, J. J. Wang, A. Wang, B. Wang, J. Ward, J. Wei, C. Weinmann, A. Welihinda, P. Welinder, J. Weng, L. Weng, M. Wiethoff, D. Willner, C. Winter, S. Wolrich, H. Wong, L. Workman, S. Wu, J. Wu, M. Wu, K. Xiao, T. Xu, S. Yoo, K. Yu, Q. Yuan, W. Zaremba, R. Zellers, C. Zhang, M. Zhang, S. Zhao, T. Zheng, J. Zhuang, W. Zhuk, and B. Zoph, 'Gpt-4 technical report,' arXiv preprint arXiv:2303.08774 , 2024.

- [31] G. Xun, X. Jia, V. Gopalakrishnan, and A. Zhang, 'A survey on context learning,' IEEE Transactions on Knowledge and Data Engineering , vol. 29, pp. 38-56, 2017.
- [32] S. Wang, Y. Liu, Y. Xu, C. Zhu, and M. Zeng, 'Want to reduce labeling cost? gpt-3 can help,' in Empirical Methods in Natural Language Processing , 2021.
- [33] H. Khorashadizadeh, N. Mihindukulasooriya, S. Tiwari, J. Groppe, and S. Groppe, 'Exploring in-context learning capabilities of foundation models for generating knowledge graphs from text,' arXiv preprint arXiv:2305.08804 , 2023.
- [34] B. Ding, C. Qin, L. Liu, Y. K. Chia, B. Li, S. Joty, and L. Bing, 'Is gpt-3 a good data annotator?' in Annual Meeting of the Association for Computational Linguistics , 2023.
- [35] O. Ram, Y. Levine, I. Dalmedigos, D. Muhlgay, A. Shashua, K. Leyton-Brown, and Y. Shoham, 'In-context retrieval-augmented language models,' in Annual Meeting of the Association for Computational Linguistics , 2023.
- [36] N. De Cao, W. Aziz, and I. Titov, 'Editing factual knowledge in language models,' in Empirical Methods in Natural Language Processing , 2021.
- [37] P. He, X. Liu, J. Gao, and W. Chen, 'Deberta: Decoding-enhanced bert with disentangled attention,' in International Conference on Learning Representations , 2021.
- [38] B. Wang, L. Shang, C. Lioma, X. Jiang, H. Yang, Q. Liu, and J. G. Simonsen, 'On position embeddings in bert,' in International Conference on Learning Representations , 2021.
- [39] G. Ke, D. He, and T.-Y. Liu, 'Rethinking positional encoding in language pre-training,' in International Conference on Learning Representations , 2021.
- [40] P. Shaw, J. Uszkoreit, and A. Vaswani, 'Self-attention with relative position representations,' in Annual Meeting of the Association for Computational Linguistics , 2018.
- [41] Z. Dai, Z. Yang, Y. Yang, J. Carbonell, Q. Le, and R. Salakhutdinov, 'Transformer-xl: Attentive language models beyond a fixed-length context,' in Annual Meeting of the Association for Computational Linguistics , 2019.
- [42] J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu, 'Roformer: Enhanced transformer with rotary position embedding,' Neurocomputing , vol. 568, p. 127063, 2024.
- [43] S. M. Xie, A. Raghunathan, P. Liang, and T. Ma, 'An explanation of in-context learning as implicit bayesian inference,' in International Conference on Learning Representations , 2022.
- [44] X. Wang, W. Zhu, M. Saxon, M. Steyvers, and W. Y. Wang, 'Large language models are latent variable models: Explaining and finding good demonstrations for in-context learning,' in Advances in Neural Information Processing Systems , 2024.
- [45] D. Dai, Y. Sun, L. Dong, Y. Hao, S. Ma, Z. Sui, and F. Wei, 'Why can GPT learn in-context? language models secretly perform gradient descent as meta-optimizers,' in Annual Meeting of the Association for Computational Linguistics , 2023.
- [46] J. Wei, M. Bosma, V. Zhao, K. Guu, A. W. Yu, B. Lester, N. Du, A. M. Dai, and Q. V. Le, 'Finetuned language models are zero-shot learners,' in International Conference on Learning Representations , 2022.
- [47] T. Kojima, S. S. Gu, M. Reid, Y. Matsuo, and Y. Iwasawa, 'Large language models are zero-shot reasoners,' in Advances in Neural Information Processing Systems , 2022.
- [48] S. Alberti, N. Dern, L. Thesing, and G. Kutyniok, 'Sumformer: Universal approximation for efficient transformers,' Annual Workshop on Topology, Algebra, and Geometry in Machine Learning , pp. 72-86, 2023.

- [49] V. Likhosherstov, K. Choromanski, and A. Weller, 'On the expressive power of self-attention matrices,' arXiv preprint arXiv:2106.03764 , 2021.
- [50] P. Deora, R. Ghaderi, H. Taheri, and C. Thrampoulidis, 'On the optimization and generalization of multi-head attention,' Transactions on Machine Learning Research , 2024.
- [51] A. Petrov, P. Torr, and A. Bibi, 'When do prompting and prefix-tuning work? a theory of capabilities and limitations,' in International Conference on Learning Representations , 2024.
- [52] W. Rudin, Functional Analysis . McGraw-Hill Science, 1991.
- [53] P. Cannarsa and T. D'Aprile, Introduction to Measure Theory and Functional Analysis . Springer Cham, 2015.
- [54] N. J. Calkin and H. S. Wilf, 'Recounting the rationals,' The American Mathematical Monthly , vol. 107, pp. 360-363, 2000.
- [55] M. Stern, 'Ueber eine zahlentheoretische funktion,' Journal für die reine und angewandte Mathematik , vol. 1858, pp. 193-220, 1858.
- [56] B. C. Berndt, H. G. Diamond, H. Halberstam, and A. Hildebrand, Analytic Number Theory: Proceedings of a Conference in Honor of Paul T. Bateman . Birkhäuser, 1990.
- [57] J. Tack, J. Lanchantin, J. Yu, A. Cohen, I. Kulikov, J. Lan, S. Hao, Y. Tian, J. Weston, and X. Li, 'Llm pretraining with continuous concepts,' arXiv preprint arXiv:2502.08524 , 2025.
- [58] S. Hao, S. Sukhbaatar, D. Su, X. Li, Z. Hu, J. Weston, Tian, and Yuandong, 'Training large language models to reason in a continuous latent space,' arXiv preprint arXiv:2412.06769 , 2024.
- [59] T. Z. Xiao, R. Bamler, B. Schölkopf, and W. Liu, 'Verbalized machine learning: Revisiting machine learning with language models,' Transactions on Machine Learning Research , 2025.
- [60] T. M. Apostol, Modular Functions and Dirichlet Series in Number Theory (Graduate Texts in Mathematics, 41) . Springer, 1989.
- [61] G. Boole, A Treatise on the Calculus of Finite Differences . Cambridge University Press, 2009.

## A Table of Notations

We present all our notations in Table 1 for easy reference.

Table 1: Table of Notations

| Notations                                                                                                                       | Explanations                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|---------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| d x , d y P X,Y X P , Y P Z Z P V V x , V y σ # N σ , N σ T σ , T σ N σ ∗ , N σ ∗ T σ ∗ , T σ ∗ T σ ∗ , P , T σ ∗ , P ∥ · ∥ ˜ x | Dimensions of input and output. Positional encoding. Context without positional encoding. Context with positional encoding P . Input without positional encoding. Input with positional encoding P . Vocabulary. Vocabulary of x ( i ) and y ( i ) . Activation function. Cardinality of a set. One-hidden-layer FNN and its collection. Single-layer Transformer and its collection. One-hidden-layer FNN with a finite set of weights and its collection. Single-layer Transformer with vocabulary restrictions and its collection. Single-layer Transformer with positional encoding, vocabulary restrictions, and its collection. The uniform norm of vectors, i.e. , a shorthand for ∥ · ∥ ∞ . Append a one to the end of x , i.e. , ˜ x = [ x 1 ] . |

## B Proofs for Section 2

We provide detailed proofs of lemmas in Section 2. We will first directly prove Lemma 3 using Lemma 2. Next, by a similar method together with an additional technical refinement, we will establish Lemma 11. Finally, leveraging Lemma 11, we will prove Lemma 4.

## B.1 Proof of Lemma 3

Lemma 3. Let σ : R → R be a non-polynomial, locally bounded, piecewise continuous activation function, and T σ be a single-layer Transformer satisfying Assumption 1. For any one-hidden-layer network N σ : R d x -1 → R d y ∈ N σ with n hidden neurons, there exist matrices X ∈ R d x × n and Y ∈ R d y × n such that

<!-- formula-not-decoded -->

Proof. The output of T σ can be computed directly as

<!-- formula-not-decoded -->

(30)

One can easily observe that the output closely resembles that of a single-layer FNN. Suppose N σ ( x ) = Aσ ( Wx + b ) : R d x -1 → R d y is an arbitrary single-layer FNN with k hidden neurons, where W ∈ R k × ( d x -1) , A ∈ R d y × k and b ∈ R k . We construct the context by setting its length to k , i.e. , X ∈ R d x × k , Y ∈ R d y × k . A straightforward calculation shows that choosing

<!-- formula-not-decoded -->

Remark 10. It is worth noting that in the above proof, the matrix F was set to zero in accordance with Assumption 1. However, we emphasize that this is not a strict requirement. In fact, one can accommodate an arbitrary F by choosing Y = U -1 ( A -FX ) . The choice F = 0 is made purely for computational convenience under our current assumptions, which is also discussed in Appendix F.

## B.2 Proof of the UAP of Softmax FNNs

Before proving Lemma 4, we demonstrate the UAP of softmax FNNs as a supporting lemma.

Lemma 11 (UAP of Softmax FNNs) . For any continuous function f : R d x → R d y defined on a compact domain K , and for any ε &gt; 0 , there exists a network N softmax ( x ) : R d x → R d y satisfying

<!-- formula-not-decoded -->

Proof. We first build a bridge connecting softmax FNNs and target function f using Lemma 2. We can construct a network

<!-- formula-not-decoded -->

with k hidden neurons such that

<!-- formula-not-decoded -->

where a i ∈ R d y , w i ∈ R d x , b i ∈ R . It therefore suffices to construct a softmax FNN N softmax ( x ) that approximates N exp ( x ) . This task amounts to eliminating the effect of the normalization in the softmax output.

Consider a softmax FNN

<!-- formula-not-decoded -->

with k +1 hidden neurons, where w ′ k +1 = b ′ k +1 = 0 , b ′ i = b ′ i ( ε ) is sufficiently small such that

<!-- formula-not-decoded -->

and w ′ i = w i for i = 1 , 2 , · · · , k . This arrangement ensures that, in the denominators of each term in Equation (35), the first k entries are arbitrarily small, while the ( k +1 )-th entry is exactly one. We then simply adjust A ′ so that the numerators coincide with those in Equation (33), and this can be done by setting a ′ i = { a i e b i -b ′ i , i = 1 , 2 , · · · , k 0 , i = k +1 . With the formal construction now complete, we

present a more precise estimate of the approximation error as follows.

<!-- formula-not-decoded -->

This leads to the conclusion that ∥ N softmax ( x ) -f ( x ) ∥ &lt; ε for all x ∈ K , thus completing the proof.

## B.3 Proof of Lemma 4

Lemma 4. Let σ : R → R be a non-polynomial, locally bounded, piecewise continuous activation function or softmax function, and T σ be a single-layer Transformer satisfying Assumption 1, and K be a compact domain in R d x -1 . Then for any continuous function f : K → R d y and any ε &gt; 0 , there exist matrices X ∈ R d x × n and Y ∈ R d y × n such that

<!-- formula-not-decoded -->

Proof. For element-wise activation case, with the help of Lemma 2 and Lemma 3, the conclusion follows trivially.

Then we handle the softmax case. Similarly, for any ε &gt; 0 , we can construct a softmax FNN N softmax ( x ) with k hidden neurons, using Lemma 11 such that

<!-- formula-not-decoded -->

We need to approximate this softmax FNN with a softmax Transformer. The computation proceeds as follows:

<!-- formula-not-decoded -->

By comparing the output with that of the exponential FNN, we find that there is an additional bounded positive term t ( x ) := exp ( ˜ x ⊤ B ⊤ C ˜ x ) arising from the normalization process.

Choose the length of the context n = k +1 and matrices X,Y such that

<!-- formula-not-decoded -->

where 1 is all-ones vector and s is big enough, making

<!-- formula-not-decoded -->

Then X ⊤ B ⊤ C ˜ x = [ W b + s 1 0 s ][ x 1 ] = [ Wx + b + s 1 s ] , and we can compute the explicit form of T softmax (˜ x ; X,Y ) as:

<!-- formula-not-decoded -->

We estimate the upper bound of the distance between N softmax and T softmax , that is

<!-- formula-not-decoded -->

As a consequence, we have ∥ ∥ T σ ( ˜ x ; X,Y ) -f ( x ) ∥ ∥ &lt; ε for all x ∈ K , which finishes the proof.

## C Proofs for Section 3

In this appendix, we provide detailed proofs of Proposition 12, Lemma 5, and Theorem 6 presented in Section 3. We will first use induction to prove Proposition 12, and then employ this proposition together with a proof by contradiction to establish Lemma 5 and Theorem 6.

Proposition 12. The scalar function h k ( x ) = k ∑ i =1 a i e b i x , with a i , b i , x ∈ R , where the exponents b i are pairwise distinct, and at least one a i is nonzero, has at most k -1 zero points.

Proof. We prove this statement by induction. When k = 1 or 2 , this can be proven easily. We suppose h N ( x ) has at most N -1 zero points. Now consider the case k = N +1 . Let h N +1 ( x ) =

̸

∑ N +1 i =1 a i e b i x . Without loss of generality, assume that a N +1 = 0 . Thus, we can rewrite h N +1 ( x ) as

<!-- formula-not-decoded -->

Then we process by contradiction. Suppose h N +1 ( x ) has more than N zero points, which implies g ( x ) has more than N zero points. Then, according to Rolle's Theorem, g ′ ( x ) must have more than N -1 zero points, which contradicts our assumption. Thus, h N +1 have at most N zero points, and the proof is complete.

Lemma 5. The function class N σ ∗ , with a non-polynomial, locally bounded, piecewise continuous element-wise activation function or softmax activation function σ , cannot achieve the UAP. Specifically, there exist a compact domain K ⊂ R d x , a continuous function f : K → R d y , and ε 0 &gt; 0 such that

<!-- formula-not-decoded -->

Proof. For any element-wise activation σ , span {N σ } forms a finite-dimensional function space. span {N σ } is closed under the uniform norm as established by Theorem 2.1 from [52] and Corollary C.4 from [53]. This implies that the set of functions approximable by span {N σ } is precisely the set of functions within span {N σ } . Consequently, any function not in span {N σ } cannot be arbitrarily approximated, meaning that the UAP cannot be achieved.

Then we prove the softmax case. First, we simplify the problem to facilitate the construction of a function that cannot be approximated. We observe that it suffices to prove the UAP fails when the first input coordinate ranges over [0 , 1] and all other coordinates are held fixed. Indeed, we can find the compact set K ⊂ R d x containing ∏ d x i =1 [ l i , r i ] . If we can show that N softmax does not achieve the UAP on [ l 1 , r 1 ] × ∏ d x i =2 { l i } , then, by applying a suitable affine change of variables, it follows that UAP also fails on [0 , 1] × ∏ d x i =2 { l i } . Consider a continuous target function

<!-- formula-not-decoded -->

The reason why we consider such a target function is that every vector-valued function f ( x 1 , · · · , x d x ) can be represented as f ( x 1 , · · · , x d x ) = ( f 1 ( x 1 , · · · , x d x ) , · · · , f d y ( x 1 , · · · , x d x ) ) . If the UAP fails for f , it must fail for at least one of its scalar components. Hence it suffices to consider the onedimensional (scalar) case. Moreover, since the values of x 2 , · · · , x d x are fixed, the above reduction to a single-variable scalar function is justified. We only need to demonstrate that there exists at least one such function that cannot be approximated arbitrarily well by any N softmax ∗ ∈ N softmax ∗ .

Then we will use Proposition 12 to complete the remainder of this proof. Before that, we need to rewrite the form of the output of N softmax , which is

<!-- formula-not-decoded -->

where ( a i , w i , b i ) ∈ A × W × B is a finite set and k is the number of hidden neurons. Consequently, the set W×B is finite, and we denote it as N := #( W×B ) . By regrouping identical terms in the numerator, we can rewrite the equation as

<!-- formula-not-decoded -->

It is important to note that this transformation applies to any N softmax ∗ ∈ N softmax ∗ , ensuring that the number of summation terms in the numerator remains strictly bounded by N .

Finally, we construct a function which cannot be approximated by such softmax networks. Assume a continuous target function

<!-- formula-not-decoded -->

which has ( N +1) zero points. If N softmax ∗ achieves the UAP, we assume that N softmax ∗ ∈ N softmax ∗ which satisfies ∥ N softmax ∗ -g ∥ ≤ ε &lt; 1 10 . We denote z i = i N +1 for i = 0 , 1 , · · · , N +1 . It is easy to verify that g ( z i ) = 1 if i is even, and g ( z i ) = -1 if i is odd, which means N softmax ∗ ( z i ) &gt; 0 . 9 for even i and N softmax ∗ ( z i ) &lt; -0 . 9 for odd i . According to the Intermediate Value Theorem, N softmax ∗ has at least N +1 zero points, which contradicts Proposition 12. And we finish our proof.

We will use Figure 1 to provide readers with an intuitive illustration of why a class of functions whose number of zeros is bounded cannot achieve universal approximation.

Figure 1: An illustration of non-approximability. The black curve represents the target function, which has N +1 zero points. The red curve represents a sum of exponentials, which has no more than N zero points. If the UAP holds, then the red curve must pass near the N +2 marked extrema in the figure. By the Intermediate Value Theorem, the function represented by the red curve would then have N +1 zeros, which contradicts its intrinsic properties.

<!-- image -->

## C.1 Proof of Theorem 6

Theorem 6. The function class T σ ∗ , with a non-polynomial, locally bounded, piecewise continuous element-wise activation function or softmax activation function σ and every T σ ∈ T σ ∗ satisfies Assumption 1, cannot achieve the UAP. Specifically, there exist a compact domain K ⊂ R d x , a continuous function f : K → R d y , and ε 0 &gt; 0 such that

<!-- formula-not-decoded -->

Proof. For cases of element-wise activation, since T σ ∗ has a similar structure to N σ ∗ , we find that span { T σ ∗ } is also a finite-dimensional function space. Hence, the same argument from Lemma 5 can be applied here to complete the proof.

Then we prove the softmax case. Recall equation (40), the output of T softmax ∗ (˜ x ; X,Y ) can be viewed as

<!-- formula-not-decoded -->

where n denotes the context length and a i ∈ A , w i ∈ W , b i ∈ B for some finite sets A , W , B . This allows us to apply the same approach as in the proof of Lemma 5, which leads to the conclusion that T σ ∗ cannot achieve the UAP.

## D Kronecker Approximation Theorem

To facilitate our constructive proof, we introduce the Kronecker Approximation Theorem as an auxiliary tool to support the main theorem.

Lemma 13 (Kronecker Approximation Theorem [60]) . Given real n -tuples α ( i ) = ( α ( i ) 1 , α ( i ) 2 , · · · , α ( i ) n ) ∈ R n for i = 1 , · · · , m and β = ( β 1 , β 2 , · · · , β n ) ∈ R n , the following condition holds: for any ε &gt; 0 , there exist q i , l j ∈ Z such that

<!-- formula-not-decoded -->

if and only if for any r 1 , · · · , r n ∈ Z , i = 1 , · · · , m with

<!-- formula-not-decoded -->

the number n ∑ j =1 β j r j is also an integer. In the case of m = 1 and n = 1 , for any α, β ∈ R with α irrational and ε &gt; 0 , there exist integers l and q with q &gt; 0 such that | β -qα + l | &lt; ε .

Lemma 13 indicates that if the condition in equation (53) is satisfied only when all r i are zeros, then the set { Mq + l | q ∈ Z m , l ∈ Z n } is dense in R n , where the matrix M ∈ R n × m is assembled with vectors α ( i ) , i.e. , M = [ α (1) , α (2) , · · · , α ( m ) ] . In the case of m = 1 and n = 1 , let α = √ 2 , then Lemma 13 implies that the set { q √ 2 ± l | l ∈ N + , q ∈ N + } is dense in R . We will build upon this result to prove one of the main theorems in this article.

## E Proofs for Section 4

In this appendix, we lay the groundwork for the proof of Theorem 7 by first introducing Lemma 14. We then present Theorem 7 and provide its complete proof, demonstrating that T σ ∗ , P can realize the UAP. To facilitate understanding of Theorem 7, we provide a simple illustrative example. While the theorem assumes dense positional encodings, we relax this condition under specific activation functions, as formalized in Lemma 16 and Theorem 8.

## E.1 Lemma 14

Lemma 14. For a network with a fixed width and a continuous activation function, it is possible to apply slight perturbations within an arbitrarily small error margin. For any network N σ 1 ( x ) defined on a compact set K ⊂ R d x , with parameters A ∈ R d y × k , W ∈ R k × d x , b ∈ R k × 1 , there exists M &gt; 0 , M 1 &gt; 0 ( ∥ x ∥ &lt; M and ∥ a i ∥ &lt; M 1 , i = 1 , · · · , k ) , and for any ε &gt; 0 , there exists 0 &lt; δ &lt; ε 2 M 1 k and a perturbed network N σ 2 ( x ) with parameters ˜ A ∈ R d y × k , ˜ W ∈ R k × d x , ˜ b ∈ R k × 1 ( ∥ ∥ ∥ σ ( ˜ w i · x + ˜ b i ) ∥ ∥ ∥ &lt; M 1 , i = 1 , · · · , k ), such that if max {∥ a i -˜ a i ∥ , M ∥ w i -˜ w i ∥ + ∥ b -˜ b ∥ | i = 1 , · · · , k } &lt; δ , then

<!-- formula-not-decoded -->

where a i , ˜ a i are the i -th column vectors of A, ˜ A , respectively, w i , ˜ w i are the i -th row vectors of W, ˜ W , and b i , ˜ b i are the i -th components of b, ˜ b , respectively, for any i = 1 , · · · , k .

<!-- formula-not-decoded -->

) = k ∑ i =1 ˜ a i σ ( ˜ w i · x + ˜ b i ) , where ˜ a j ∈ R d y , ˜ w i ∈ R d x , ˜ b i ∈ R . For any x ∈ K , ∥ x ∥ &lt; M . There exists a constant M 1 &gt; 0 such that for any i = 1 , · · · , k , the following inequalities hold: ∥ a i ∥ &lt; M 1 and ∥ ∥ ∥ σ ( ˜ w i · x + ˜ b i ) ∥ ∥ ∥ &lt; M 1 .

Due to the continuity of the activation function, for any ε &gt; 0 , there exists 0 &lt; δ &lt; ε 2 M 1 k , such that if ∥ w i · x + b i -( ˜ w i · x + ˜ b i ) ∥ ≤ ∥ w i -˜ w i ∥∥ x ∥ + ∥ b i -˜ b i ∥ &lt; M ∥ w i -˜ w i ∥ + ∥ b -˜ b ∥ &lt; δ , then ∥ σ ( w i · x + b i ) -σ ( ˜ w i · x + ˜ b i ) ∥ &lt; ε 2 M 1 k , and ∥ a i -˜ a i ∥ &lt; δ , for any i = 1 , · · · , k .

Combining all these inequalities, we can further derive:

<!-- formula-not-decoded -->

## E.2 Proof of Theorem 7

Theorem 7. Let T σ ∗ , P be the class of functions T σ ∗ , P satisfying Assumption 1, with a non-polynomial, locally bounded, piecewise continuous element-wise activation function σ , the subscript refers the finite vocabulary V = V x ×V y , P = P x ×P y represents the positional encoding map, and denote a set S as:

<!-- formula-not-decoded -->

If S is dense in R d x , { 1 , -1 , √ 2 , 0 } d y ⊂ V y and P y = 0 , then T σ ∗ , P can achieve the UAP. More specifically, given a network T σ ∗ , P , then for any continuous function f : R d x -1 → R d y defined on a compact domain K and ε &gt; 0 , there always exist X ∈ R d x × n and Y ∈ R d y × n from the vocabulary V , i.e. , x ( i ) ∈ V x , y ( i ) ∈ V y , with some length n ∈ N + such that

<!-- formula-not-decoded -->

̸

Proof. Our conclusion holds for all element-wise continuous activation functions in T σ ∗ , P . We now assume d y = 1 for simplicity, and the case d y = 1 will be considered later.

We are reformulating the problem. Using Lemma 3, we have

<!-- formula-not-decoded -->

Since P y = 0 , it follows that Y P = Y . For any continuous function f : R d x -1 → R d y defined on a compact domain K and for any ε &gt; 0 , we aim to show that there exists T σ ∗ , P ∈ T σ ∗ , P such that:

<!-- formula-not-decoded -->

In the main text, for illustrative purposes, we consider the special case where U is the identity matrix to simplify the exposition. In the present analysis, we dispense with this assumption. We already have the Lemma 2 ensuring the existence of a one-hidden-layer network N σ (with activation function

σ satisfying the required conditions) that approximates f ( x ) . Our proof is divided into four steps, serving as a bridge built upon the Lemma 2:

<!-- formula-not-decoded -->

We present the specific details at each step.

Step (1): Approximating f ( x ) Using N σ ( x ) . Supported by Lemma 2, there exists a neural network N σ ( x ) = Aσ ( Wx + b ) = k ∑ i =1 a i σ ( w i · x + b i ) ∈ N σ , with parameters k ∈ N + , A ∈ R d y × k , b ∈ R k , and W ∈ R k × ( d x -1) ,

<!-- formula-not-decoded -->

Step (2): Approximating N σ ( x ) Using N ′ ( x ) . Using Lemma 13 and Lemma 14, a neural network N σ ( x ) = k ∑ i =1 a i σ ( w i · x + b i ) ∈ N σ can be perturbed into N ′ ( x ) = k ∑ i =1 ( q √ 2 ± l ) i σ ( ˜ w i · x + ˜ b i ) (with q i ∈ N + and l i ∈ N + , i = 1 , · · · , k ), such that for any ε &gt; 0 , there exists 0 &lt; δ &lt; ε 6 M 1 k satisfying:

<!-- formula-not-decoded -->

ensuring:

<!-- formula-not-decoded -->

Step (3): Approximating N ′ ( x ) Using N σ ∗ ( x ) . Next, we show that N σ ∗ ( x ) = n ∑ i =1 y ( i ) σ ( ˜ R i · ˜ x ) ∈

N σ ∗ can approximate N ′ ( x ) = k ∑ i =1 ( q √ 2 ± l ) i σ ( ˜ w i · ˜ x ) . As a demonstration, we approximate a single term ( q √ 2 ± l ) 1 σ ( ˜ w 1 · ˜ x ) . Since the positional encoding is fixed, i.e. , V x + P (1) is a finite set, one of two cases must occur:

1. Valid Position: If there exists x (1) ∈ V x where ( x (1) + P (1) ) ⊤ B ⊤ C ≈ ˜ w 1 ;
2. Invalid Position: Set y (1) = 0 to nullify contribution.

Since S is dense in R d x and B ⊤ C is non-singular, the set G := { ˜ R | ˜ R = X ⊤ P B ⊤ C, X P ⊂ 2 S } remains dense. Let K 1 denote the set of indices corresponding to all "valid" positions for ˜ w 1 . Since y ( i ) ∈ { 1 , -1 , √ 2 , 0 } , we require q 1 + l 1 elements from G that approximate ˜ w 1 , such that

<!-- formula-not-decoded -->

Here, #( K 1 ) = q 1 + l 1 and K 1 = Q 1 ⋃ L 1 , where Q 1 , L 1 are disjoint subsets of positive integer indices satisfying #( Q 1 ) = q 1 and #( L 1 ) = l 1 . For this construction, we assign y ( j ) = √ 2 for j ∈ Q 1 and y ( j ) = ± 1 for j ∈ L 1 . For j ∈ { 1 , 2 , 3 , · · · , max i { i ∈ K 1 }}\ K 1 , i.e. , for the Invalid Position , we set y ( j ) = 0 .

The multi-term approximation employs parallel construction via disjoint node subsets K i = Q i ∪ L i , where Q i ( q i nodes) and L i ( l i nodes) implement √ 2 and ± 1 coefficients respectively. For j / ∈ k ⋃ l =1 K l , we set y ( j ) = 0 . Each term achieves:

<!-- formula-not-decoded -->

We then define n = max { j | j ∈ k ⋃ l =1 K l } . The complete network combines these approximations through:

<!-- formula-not-decoded -->

Step (4): Combining Results. Combining all results, we have:

<!-- formula-not-decoded -->

The scalar-output results ( d y = 1 ) extend naturally to vector-valued functions via componentwise approximation. For any continuous f : R d x -1 → R d y on a compact domain K , uniform approximation is achieved by independently approximating each coordinate function f j with scalar networks N σ ∗ ,j ( x ) satisfying

<!-- formula-not-decoded -->

The full approximator is then obtained by concatenating the component networks.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where y ( i ) j is the j -th row of the y ( i ) . We require that the index sets satisfy K ( o ) i ∩ K ( u ) j = ∅ for all o, u, i, j ∈ N + , where K ( o ) i denotes the index set constructed for the i -th term approximation in the o -th output dimension. Furthermore, each y ( j ) must have at most one non-zero element across its dimensions. This ensures we achieve uniform approximation by independently handling each output dimension. The proof is complete.

## E.3 Example of Theorem 7

We present a concrete example with 2D input ( d x = 2 ) and 2D output ( d y = 2 ) to illustrate the universal approximation capability of our architecture. Consider a continuous function f : [0 , 1] 2 → R 2 defined by

<!-- formula-not-decoded -->

Our goal is to construct a module T σ ∗ , P such that

<!-- formula-not-decoded -->

Step (1): Component-wise Approximation. For each component f i , there exists a single-hiddenlayer neural network N σ i ( x ) = A i σ ( W i x + b i ) such that

<!-- formula-not-decoded -->

Step (2): Rational Perturbation. We approximate each N σ i by a rational network N ′ i :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˜ x = [ x 1 x 2 1] ⊤ , satisfying

<!-- formula-not-decoded -->

Step (3): Architecture Realization. We define a Transformer-like module N σ ∗ ( x ) with shared representation:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

such that

<!-- formula-not-decoded -->

Step (4): Error Analysis. The total approximation error satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We argue that in standard ICL, when y ( i ) = f ( x ( i ) ) (referred to as meaningfully related) and V y satisfies certain conditions, the UAP conclusion still holds. This conclusion relies on the density of S , and we provide a concise argument based on Theorem 7.

Theorem 15. Let T σ ∗ , P be the class of functions T σ ∗ , P satisfying Assumption 1, with a non-polynomial, locally bounded, piecewise continuous element-wise activation function σ , the subscript refers the finite vocabulary V = V x ×V y , P = P x ×P y represents the positional encoding map, and denote a set S as:

<!-- formula-not-decoded -->

If S is dense in R d x , and there exists a subset Y 0 ⊆ V y whose (finite) columnwise additive combinations contain the block-diagonal pattern in

<!-- formula-not-decoded -->

and P y = 0 , then T σ ∗ , P can achieve the UAP. More specifically, for any network T σ ∗ , P , and for any continuous function f : R d x -1 → R d y defined on a compact domain K and any ε &gt; 0 , there exist X ∈ R d x × n and Y ∈ R d y × n from the vocabulary V , i.e. , x ( i ) ∈ V x , y ( i ) ∈ V y , with some length n ∈ N + such that

<!-- formula-not-decoded -->

We reuse Steps (1) -(2) from the proof of Theorem 7, focusing on understanding the third step. There exists a subset Y 0 ⊆ V y whose additive combinations of columns contain the pattern in Eq. (83). The role of this structure is to enable cumulative approximation through additive combinations.

## E.4 Feasibility of UAP under Different Positional Encodings

We also pay attention to more dynamic positional encodings such as RoPE, and are currently exploring appropriate analytical methods for them. Our recent progress on APEs has given us greater confidence in studying RPEs. Our analytical framework mainly relies on achieving density of the set σ ( X ⊤ B ⊤ C ˜ x ) , in particular, on the richness of the term X ⊤ B ⊤ C . (See Lemma 3 and Theorem 7 for supporting arguments).

For RoPE, whose basic formulation is given by equation (16) in [42]

<!-- formula-not-decoded -->

applying our approach yields

<!-- formula-not-decoded -->

However, since the rotation operation in RoPE acts on distinct two-dimensional subspaces of d x , the induced family { B ⊤ ( R d Θ ,j ) ⊤ C } does not generate a dense subset; hence our density-based argument does not directly apply to RoPE. Consequently, our current method cannot be directly applied to prove that RoPE possesses similar approximation properties.

Likewise, other RPEs, such as the one defined in equation (4) of [40],

<!-- formula-not-decoded -->

cannot be analyzed using this approach either. Nevertheless, the encoding formulation in [41],

<!-- formula-not-decoded -->

can be accommodated within our framework and is compatible with the UAP result.

## E.5 Proof of Theorem 8

Before proving Theorem 8, we need to prove the following lemma with the help of the well-known Stone-Weierstrass theorem.

Lemma 16. For any continuous function f : R d x → R d y defined on a compact domain K , and for any ε &gt; 0 , there exists a network N exp ( x ) : R d x → R d y satisfying

<!-- formula-not-decoded -->

where b = 0 and all row vectors of W are restricted in a neighborhood B ( ω ∗ , δ ) with any fixed w ∗ ∈ R d x and radius δ &gt; 0 .

Proof. Assume f ( x ) = ( f 1 ( x ) , · · · , f d x ( x )) . According to Stone-Weierstrass theorem, for any ε &gt; 0 , there exist polynomials P i ( x ) satisfying

<!-- formula-not-decoded -->

Then we construct a single-layer FNN with exponential activation function to approximate P i ( x )e w ∗ · x . The multiple derivatives of h ( w ) := e w · x = exp( w 1 x 1 + · · · + w d x x d x ) with respect to w 1 , · · · , w d x are

<!-- formula-not-decoded -->

where α ∈ N d x represents the index and | α | := α 1 + · · · + α d x . Actually, the form of multiple derivative ∂ | α | h ∂w α is a polynomial of | α | degree with respect to x 1 , · · · , x d x times h ( w ) . Hence, each target term P i ( x )e w ∗ · x can be written as a linear combination of such multiple derivatives of h ( w ) , which allows us to approximate the required partials and thus complete the proof. Moreover, each mixed derivative can be approximated by a finite-difference scheme, which can be implemented using a single hidden layer.

Remark 17. We give two examples of approximating multiple derivatives of h ( w ) below.

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where e 1 = (1 , 0 , 0 , · · · , 0) , e 2 = (0 , 1 , 0 , · · · , 0) are unit vectors and R 1 ( λ, w ∗ ) , R 2 ( λ, w ∗ ) are error terms with respect to λ and w ∗ . According to Taylor's theorem, the error terms R 1 ( λ, w ∗ ) = λ ∂ 2 h ∂w 2 1 ∣ ∣ w = ξ for some ξ between w ∗ and w ∗ + λe 1 . It is obvious that the partial differential term is uniformly bounded, so the resulting error can be made arbitrarily small by a suitable choice of the parameter λ . The argument for R 2 ( λ, W ∗ ) is entirely analogous and is therefore omitted; see [61] for further details.

Since λ is very small and the exponential term e w ∗ · x only involves the parameters w ∗ , w ∗ + e 1 and w ∗ + e 2 , which all lie within a small neighborhood of w ∗ , the desired conclusion can be drawn, and this means we can in fact restrict all row vectors of W to lie within B ( W,δ ) .

Theorem 8 (Formal Version) . Let T σ ∗ , P be the class of functions T σ ∗ , P satisfying Assumption 1, with a non-polynomial, locally bounded, piecewise continuous element-wise activation function σ , the subscript refers to the finite vocabulary V = V x × V y , P = P x × P y represents the positional encoding map, and denote a set S as:

<!-- formula-not-decoded -->

If the set S is dense in [ -1 , 1] d x , then T ReLU ∗ , P is capable of achieving the UAP. Additionally, if S is only dense in a neighborhood B ( w ∗ , δ ) of a point w ∗ ∈ R d x with radius δ &gt; 0 , then the class of transformers with exponential activation, i.e. , T exp ∗ , P , is capable of achieving the UAP.

Proof. For the proof of the ReLU case, we follow the same reasoning as in the previous one, noting that ReLU( ax ) = a ReLU( x ) holds for any positive a . In the proof of Theorem 7, we construct a T ReLU ∗ , P (˜ x ; X,Y ) ∈ T ReLU ∗ , P to approximate a FNN A ReLU( Wx + B ) . Here we can do a similar construction to find another ˜ T ReLU ∗ , P (˜ x ; X,Y ) ∈ T ReLU ∗ , P to approximate λA ReLU ( λ -1 ( Wx + b ) ) as the Step (2) -(4) in Theorem 7, where λ is chosen sufficiently large such that the row vectors of λ -1 W become small enough to ensure that S = { x i + P | x i ∈ V , i, j ∈ N + } is dense in [ -1 , 1] d x and sufficient for our construction.

For exponential Transformers, by using Lemma 16, we can do the Step (2) -(4) in Theorem 7 again, which is similar to ReLU case.

## F Weakened Assumption and Generalized Conclusions

It is important to note that most of our conclusions remain valid even if Assumption 1 is weakened. Below we outline the reasoning.

In general, we decompose the matrices as follows:

<!-- formula-not-decoded -->

where O 11 , D ∈ R d x × d x , O 12 , E ∈ R d x × d y , O 21 , F ∈ R d y × d x , and O 22 , U ∈ R d y × d y . The attention mechanism can then be computed as:

<!-- formula-not-decoded -->

where O represents the matrix X ⊤ O 11 X + X ⊤ O 12 Y + Y ⊤ O 21 X + Y ⊤ O 22 Y . As a result, we have:

<!-- formula-not-decoded -->

for the case of element-wise activations, and:

<!-- formula-not-decoded -->

for the case of softmax activation.

By revisiting the definition of T σ and T σ ∗ , and comparing T σ presented here with those in the preceding section, it is clear that the only distinction lies in the specific matrices involved, and matrix O 11 and U are non-singular are the only conditions we need. Notably, the proof process for Theorem 6 does not rely on any assumptions, which means the conclusion stated in Section 3 can be further strengthened.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes] .

Justification: We outline and discuss the contributions and scope of our work at the end of the Abstract and in a dedicated subsection 1.1 of the Introduction 1.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes] .

Justification: We discuss the limitations of our work around line 345 in Section 5.

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

Answer: [Yes] .

Justification: Our assumptions are clearly stated as Assumption 1 in Section 2 and Appendix F, and the proofs are provided in Appendix B to Appendix E.

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

Justification: This paper does not include experiments.

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

Answer: [NA] .

Justification: This paper does not include experiments requiring code.

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

Answer: [NA] .

Justification: This paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA] .

Justification: This paper does not include experiments.

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

Answer: [NA] .

Justification: This paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes] .

Justification: This research is theoretical in nature and does not involve human subjects, personal data, or potentially harmful applications. All results are derived through mathematical analysis and do not raise ethical concerns as outlined in the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This paper focuses on the theoretical expressivity of Transformers under ICL and provides approximation results from a mathematical perspective. We believe that discussing societal impact falls outside the scope of this foundational contribution.

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

Justification: This paper does not release any data, models, or tools that pose risks of misuse or dual use. The work is purely theoretical and focuses on the UAP in VICL with single-layer Transformers.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper does not use any external assets such as datasets, models, or thirdparty code. The research is purely theoretical and does not rely on pre-existing software or data resources.

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

Justification: This paper is theoretical in nature and does not introduce or release any new datasets, models, or software assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve any crowdsourcing or experiments with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA] .

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA] .

Justification: The core method development in this research does not involve LLMs Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.