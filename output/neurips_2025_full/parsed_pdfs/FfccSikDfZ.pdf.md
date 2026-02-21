## SHAP Meets Tensor Networks: Provably Tractable Explanations with Parallelism

## Reda Marzouk ∗

LIRMM, UMR 5506, University of Montpellier, CNRS mohamed-reda.marzouk@umontpellier.fr

## Shahaf Bassan*

The Hebrew University of Jeursalem shahaf.bassan@mail.huji.ac.il

Guy Katz

The Hebrew University of Jeursalem g.katz@mail.huji.ac.il

## Abstract

Although Shapley additive explanations (SHAP) can be computed in polynomial time for simple models like decision trees, they unfortunately become NP-hard to compute for more expressive black-box models like neural networks - where generating explanations is often most critical. In this work, we analyze the problem of computing SHAP explanations for Tensor Networks (TNs) , a broader and more expressive class of models than those for which current exact SHAP algorithms are known to hold, and which is widely used for neural network abstraction and compression. First, we introduce a general framework for computing provably exact SHAP explanations for general TNs with arbitrary structures. Interestingly, we show that, when TNs are restricted to a Tensor Train (TT) structure, SHAP computation can be performed in poly-logarithmic time using parallel computation. Thanks to the expressiveness power of TTs, this complexity result can be generalized to many other popular ML models such as decision trees, tree ensembles, linear models, and linear RNNs, therefore tightening previously reported complexity results for these families of models. Finally, by leveraging reductions of binarized neural networks to Tensor Network representations, we demonstrate that SHAP computation can become efficiently tractable when the network's width is fixed, while it remains computationally hard even with constant depth . This highlights an important insight: for this class of models, width - rather than depth -emerges as the primary computational bottleneck in SHAP computation.

## 1 Introduction

Shapley additive explanations (SHAP) [76] represent a widely adopted method for obtaining posthoc explanations for decisions made by ML models. However, one of its main limitations lies in its computational intractability [22, 107, 83]. While some frameworks mitigate this intractability using sampling heuristics or approximations, such as KernelSHAP [76, 42], FastSHAP [63], DeepSHAP [37], Monte Carlo-based methods [31, 117, 86, 50, 51, 32], or leverage score sampling [89], others focus on exact SHAP computations, by leveraging structural properties of certain model families to derive polynomial-time algorithms, such as the popular TreeSHAP [77] algorithm, and its variants [115, 85, 113, 87].

However, exact SHAP frameworks are confined to relatively simple models, such as tree-based models. For more expressive models like neural networks, where explanations are often most critical,

∗ Equal Contribution

NP-Hardness results have unfortunately been shown to hold [107, 83]. In this work, we conduct a complexity-theoretic analysis of the problem of computing exact SHAP explanations for the class of Tensor Networks (TNs) , a broader and more expressive class of models compared to previously studied models such as tree-based models or linear models.

Tensor Networks. Originally introduced in the field of Quantum physics [24], TNs have gradually garnered attention of the ML community where they were employed to model various ML tasks, ranging from classification and regression tasks [90, 102, 99, 34, 36, 53] to probabilistic modeling [84, 94, 103, 97, 52, 91], dimensionality reduction [78] and model compression [54]. From the perspective of explainable AI, the study of TNs is interesting in two ways. First, TNs offer a powerful modeling framework with function approximation capabilities comparable to certain families of neural networks [4, 95]. Second, the structured physics-inspired architecture of TNs enables the derivation of efficient, theoretically grounded solutions of XAI-related problems [74, 96, 3]. These two properties make TNs an interesting family of models that enjoy two arguably sought-after desiderata of ML models in the field of explainability: (i) high expressiveness; and (ii) enhanced transparency [98, 70].

Contributions. In this work, we provide a computational complexity-theoretic analysis of the problem of SHAP computation for the family of TNs. Following the line of work in [77, 6, 116, 67, 82, 83, 58] on tractable SHAP computation for simpler models, our main technical contributions are as follows:

1. SHAP for general TNs. We introduce a framework for computing provably exact SHAP explanations for general TNs with arbitrary structure, offering the first exact algorithm for generating SHAP values in this model class.
2. SHAP for Tensor Trains (TTs). We provide a deeper computational study of the SHAP problem for the particular class of Tensor Trains (TTs) , a popular subfamily of TNs that exhibits better tractability properties than general TNs. Interestingly, we show that computing SHAP for the family of TTs is not only computable in polynomial-time, but also belongs to the complexity class NC , i.e. it can be computed in poly-logarithmic time using parallel computation. This complexity result bridges a significant expressivity gap by establishing the tractability of SHAP computation for a model family that is significantly more expressive than those previously known to admit exact SHAP algorithms, while also demonstrating its efficient parallelizability.
3. From TTs to improved complexity bounds for SHAP across additional ML models. Via reduction to TTs, we show that the NC complexity result can be extended to the problem of SHAP computation for a wide range of additional popular ML models, including tree ensembles, decision trees, linear RNNs, and linear models, across various distributions, thus tightening previously known complexity results. This advancement benefits SHAP computation for these models in two key ways: (i) it enables substantially more efficient computation through poly-logarithmic parallelism, and (ii) it broadens the class of distributions used to compute SHAP's expected value, which captures more complex feature dependencies than those employed in current implementations.
4. From TTs to SHAP for Binarized Neural Networks (BNNs). Finally, through reductions from TTs, we reveal new complexity results for computing SHAP for BNNs via parameterized complexity [48, 46], a framework for assessing how structural parameters impact computational hardness. We find that while SHAP remains hard when depth is fixed, it becomes polynomial-time computable when width is bounded - highlighting width as the main computational bottleneck. We then further strengthen this insight by proving that fixing both width and sparsity (via the reified cardinality) renders SHAP efficiently tractable , even for arbitrarily large networks. This opens the door to a new, relaxed class of neural networks that permit efficient SHAP computation.

Beyond these core complexity results, which form the central focus of our work, our contributions also shed light into two novel complexity-theoretic aspects of SHAP computation that, to the best of our knowledge, have not been explored in prior literature:

When is computing SHAP efficiently parallelizable? To the best of our knowledge, this is the first work to provide a complexity-theoretic analysis of the computational parallelizability of SHAP computation. Our work provides a tighter complexity bound for many previously known tractable SHAP configurations by investigating conditions under which SHAP computation can be parallelized to achieve polylogarithmic-time complexity. Notably, we show that this is achievable for several

classical ML models, including decision trees, tree ensembles, linear RNNs, and linear models paving the way for a line of future research in this direction.

What is the computational bottleneck of SHAP computation for neural networks? To the best of our knowledge, while prior work has established that computing SHAP for general neural networks is computationally hard [107, 83], our work presents the first fine-grained analysis of how different structural parameters influence this complexity. Focusing on binarized neural networks, we show that SHAP becomes efficiently computable when both the network's width and sparsity are fixed, whereas it remains hard even with constant depth. We believe this insight paves the way for further exploration of neural network relaxations that enable efficient SHAP computation and invites broader theoretical investigation into other structural parameters and architectures where SHAP may be tractable.

Due to space constraints, we include only brief outlines of some proofs in the main text, with complete proofs provided in the appendix.

## 2 Preliminaries

Notation. For integers ( i, n ) with i ≤ n , let e n i denote the one-hot vector of length n with a 1 in the i -th position and 0 elsewhere. The vector 1 n ∈ R n denotes the vector equal to 1 everywhere. For integers m and n , we use the notation m ⊗ n def = [ m ] × . . . × [ m ] ︸ ︷︷ ︸ where [ m ] = { 1 , 2 , . . . , m } .

n

times

Complexity classes. In this work, we will assume familiarity with standard complexity classes such as polynomial time ( P ), and nondeterministic polynomial time ( NP and coNP ). We further analyze the complexity class # P , which captures the number of accepting paths of a nondeterministic polynomialtime Turing machine, and is generally regarded as significantly 'harder' than NP [7]. Moreover, we analyze the complexity class NC , which includes problems solvable in poly-logarithmic time using a polynomial number of parallel processors, typically on a Parallel Random Access Machine (PRAM) [39] (see the appendix for a full formalization). Intuitively, a problem is in NC if it can be efficiently solved in parallel . While NC ⊆ P is known, it is widely believed that NC ⊊ P [7]. The class NC is further divided into subclasses NC k for some integer k . The parameter k designates the logarithmic order of circuits that can solve computational problems in NC k . For example, NC 1 contains problems solvable with circuits of logarithmic depth, while NC 2 allows circuits of quadratic logarithmic depth, capturing slightly more complex parallel computations.

Finally, our work also draws on concepts from parameterized complexity theory [46, 48]. In this framework, problems are evaluated based on two inputs: the main input size n and an additional measure known as the parameter k , with the aim of confining the combinatorial explosion to k rather than n . We focus on the three most commonly studied parameterized complexity classes: (i) FPT (Fixed-Parameter Tractable), comprising problems solvable in time g ( k ) · n O (1) for some computable function g , implying tractability when k is small; (ii) XP (slice-wise Polynomial), where problems can be solved in time n g ( k ) , with a polynomial degree that may grow with k , thus offering weaker tractability guarantees than FPT ; and (iii) para-NP , which captures problems with the highest sensitivity to k , where a problem is para-NP-hard if it remains NP-hard even when k is fixed to a constant . It is widely believed that FPT ⊊ XP ⊊ para-NP [46].

Shapley values. Let n in , n out ≥ 1 be two integers. Fix a discrete input space D = [ N 1 ] × [ N 2 ] × · · · × [ N n in ] , a model M : D → R n out , and a probability distribution P over D . For an input instance x = ( x 1 , x 2 , . . . , x n in ) ∈ D , the SHAP attribution vector assigned to the feature i ∈ [ n in ] is defined as:

<!-- formula-not-decoded -->

where W ( S ) def = | S | !( n in -| S |-1)! n in ! . We assume the common marginal (or interventional ) value function: V M ( x, S ; P ) := E x ′ ∼ P [ M ( x S , x ′ ¯ S ) ] [62, 104]. For j ∈ [ n out ] , the j -th component of ϕ i ( M,x,P ) represents the attribution of feature i to the j -th output of the model on input x .

## 3 Tensors, Tensor Networks and Binarized Neural Networks

## 3.1 Tensors

A tensor is a multi-dimensional array that generalizes vectors and matrices to higher dimensions, referred to as indices . The dimensionality of a tensor, i.e. the number of its indices, defines its order . Elements of a tensor T ∈ R d 1 × ... × d n are denoted T i 1 ,...,i n . For j ∈ [ n ] , the slice T i 1 ,...,i j -1 , : ,i j +1 ,...,i n is a vector in R d j whose k -th entry is equal to T i 1 ,...,i j -1 ,k,i j +1 ,...,i n for k ∈ [ d j ] . A key operation over tensors is the contraction operation , which generalizes matrix multiplication to high-order tensors. Formally, given two tensors T (1) ∈ R d 1 × ... × d n and T (2) ∈ R d ′ 1 × ... × d ′ m , and two indices ( i, j ) ∈ [ n ] × [ m ] , such that d i = d ′ j The contraction operation between T (1) and T (2) over their respective indices i and j produces another tensor, denoted T (1) × ( i,j ) T (2) , over R d 1 × ...d i -1 × d i +1 × ... × d n × d ′ 1 × ... × d ′ j -1 × d ′ j +1 × ... × d ′ m such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The contraction operation over a single pair of shared indices as defined above can be generalized to sets of shared indices in a natural fashion: For a set S ⊆ [ n ] × [ m ] , the multi-leg contraction operation between T (1) and T (2) over S , denoted T (1) × S T (2) applies leg contraction to all pairs of indices in S . Note that this operation is commutative - the result of the operation doesn't depend on the contraction order.

Figure 1: Illustrations of (1) A tensor of order 4, (2) A general TN comprising 4 tensors of order 4 and 3 free indices, and (3) A tensor train (TT).

<!-- image -->

## 3.2 Tensor Networks (TNs)

TNs provide a structured representation of high-order tensors by decomposing them into interconnected collections of lower-order tensors bound by the contraction operation. TNs can be modeled as a graph in which each node corresponds to a tensor and each edge corresponds to an index. Edges that are incident to only one node represent free indices of the overall tensor, while edges shared between two nodes represent contracted indices (see Figure 5b). The tensor encoded by the network is obtained by carrying out all contractions prescribed by the graph structure. This representation of high-order tensors can be advantageous because the storage requirements and computational cost scale with the ranks of the intermediate tensors and the network topology, rather than with the full dimensionality of the original tensor.

Tensor Trains (TTs). General TNs with arbitrary topologies can be computationally challenging to handle [111, 101]. TTs form a subclass of TNs with a one-dimensional topology that exhibits better tractability properties [93]. A TT decomposes a high-order tensor into a linear chain of third-order 'cores' (see Figure 5c). Formally, a TT T is a TN parametrized as J T (1) , . . . , T ( n ) K , where each core T ( i ) ∈ R d i -1 × N i × d i . The TT T corresponds to a tensor in R d 0 × N 1 × ... × N n × d n obtained by contraction.

<!-- formula-not-decoded -->

By convention, when d 0 (resp. d n ) is equal to 1 , the tensor core T (1) (resp. T ( n ) ) is a matrix (i.e., a Tensor of order 2).

## 3.3 Binarized Neural Networks (BNNs)

A Binarized Neural Network (BNN) is a neural network in which both the weights and activations are constrained to binary values, typically {-1 , +1 } . In a BNN, each layer performs a sequence of operations: (i) a linear transformation, (ii) batch normalization, and (iii) binarization. Formally, for an input vector x ∈ {-1 , +1 } n , the output y ∈ {-1 , +1 } m of a layer is computed as follows [41]:

<!-- formula-not-decoded -->

W ∈ {-1 , +1 } m × n , b ∈ R m , and γ, β, µ, σ ∈ R m are the batch normalization parameters. The sign function is applied element-wise, mapping positive inputs to +1 and non-positive inputs to -1 .

Reified Cardinality Representation of BNNs. In Section 6, we analyze how network sparsity affects the complexity of computing SHAP for BNNs, using the reified cardinality parameter from [64]. A reified cardinality constraint is a binary condition of the form ( ∑ n i =1 x i ≥ R ) , where x i are Boolean variables and R is the reified cardinality parameter. In BNNs, each neuron is encoded with such a constraint: for binary inputs x j and weights w ij ∈ {-1 , +1 } , define l ij = x j if w ij = +1 , and l ij = ¬ x j otherwise. The output y i is then: y i ← ( ∑ n j =1 l ij ≥ R i ) where R i is derived from the neuron's bias and normalization. Jia and Rinard [64] showed that all BNNs can be expressed this way, and that constraining R i during training improves formal verification efficiency.

## 4 Provably Exact SHAP Explanations for TNs: A General Framework

In this section, we introduce a general framework for computing provably exact SHAP explanations for TNs with arbitrary structures, which provides the backbone of more tractable SHAP computational solutions. We begin by formalizing the problem and transitioning from the classic SHAP formulation to an equivalent tensorized representation that facilitates our proofs.

A tensorized representation of Shapley values. We adopt an equivalent tensorized form of the SHAP formula in Equation (1) by defining the Marginal SHAP Tensor T ( M,P ) ∈ R D × n in × n out as:

<!-- formula-not-decoded -->

The Marginal SHAP Tensor summarizes the full SHAP information of the model M , and will play a crucial role in the technical development of this paper. Given V ( M,P ) , The SHAP Matrix of the input x = ( x 1 , . . . , x n in ) ∈ D is obtained using the straightforward contraction operation:

<!-- formula-not-decoded -->

where S def = { ( k, k ) : k ∈ [ n in +1] } .

## SHAP Computation (Problem Statement).

Given two integers n in , n out ≥ 1 , a finite domain D , a TN model T M ∈ R N 1 × ... × N n in × n out mapping D to R n out , a TN T P ∈ R N 1 ×···× N n in implementing a probability distribution over D , and an instance x ∈ D . The objective is to construct the SHAP Matrix Φ( T M , x, T P ) ∈ R n in × n out , where: Φ( T M , x, T P ) i, : def = ϕ i ( T M , x, T P ) (Equation (1)).

The complexity is measured in terms of the input and output dimensions n in , n out , the volume of the input space max i ∈ [ n in ] N i , and the total number of parameters of T M and T P .

The first step toward the derivation of exact SHAP explanations for general TNs builds on a reformulation of the Marginal SHAP Tensor (Equation (2)) highlighted in the following proposition:

Proposition 1. Define the modified Weighted Coalitional tensor ˜ W ∈ R n in × 2 ⊗ n in such that ∀ ( i, s 1 , . . . , s n in ) ∈ [ n in ] × [2] ⊗ n in it holds that ˜ W i,s 1 ,...,s n in def = -W ( s -1 n in ) if s i = 1 , and ˜ W i,s 1 ,...,s n in def = W ( s -1 n in ) otherwise. Moreover, define the marginal value tensor V ( M,P ) ∈

Algorithm 1 The construction of the Marginal SHAP Tensor

Input:

Two TNs T M and T P

Output:

The Marginal SHAP Tensor T ( M,P )

- 1: Construct the modified Weighted Coalitional Tensor ˜ W (Lemma 1)

- 2: Construct the Marginal Value Tensor V ( M,P ) (Lemma 2)

- 3: return ˜ W× S V ( M,P ) (Equation (12))

R D × 2 ⊗ n in × n out , such that ∀ ( x, s ) ∈ D × [2] ⊗ n it holds that V ( M,P ) x,s, : def = V M ( x, s ; P ) . Then we have that:

<!-- formula-not-decoded -->

Proposition 1 expresses the Marginal SHAP Tensor T ( M,P ) as a contraction of two tensors: the modified Weighted Coalitional Tensor and the Marginal Value Tensor . This leads to a natural algorithm to construct V ( M,P ) : First construct ˜ W and V ( M,P ) , then contract them as in Equation (12) (Algorithm 1). Figure 2a illustrates the resulting TN of this process.

To complete the picture, we need to show how to construct both tensors ˜ W (Step 1) and V ( M,P ) (Step 2). We split the remainder of this section into two segments, each of which is dedicated to outlining their structure and the running time of their construction.

Step 1: Constructing the modified Weighted Coalitional Tensor. The Tensor ˜ W simulates the computation of weights associated with each subset of the input features in the SHAP formulation (Equation (1)). A key observation consists of noting that this tensor admits an efficient representation as a (sparse) TT constructible in O (1) time using parallel processors:

Lemma 1. The modified Weighted Coalitional Tensor ˜ W admits a TT representation: J G (1) , . . . , G ( n in ) K , where G (1) ∈ R n in × 2 × n 2 in , G ( i ) ∈ R n 2 i × 2 × n 2 i for any i ∈ [2 , n in -1] , and G ( n in ) ∈ R n 2 in × 2 . Moreover, this TT representation is constructible in O (log( n in )) time using O ( n 3 in ) parallel processors.

Step 2: Constructing the Marginal Value Tensor. The goal of the second step is to construct the Marginal Value Tensor V ( M,P ) from the TNs T M and T P . The following lemma shows how this Tensor can be constructed by means of suitable TN contractions:

Lemma 2. Let T M and T P be TNs implementing the model M and the probability distribution P , respectively. Then, the marginal value tensor V ( M,P ) can be computed as:

<!-- formula-not-decoded -->

where S 1 and S 2 are instantiated such that for all k ∈ { 1 , 2 } it holds that: S k def = { (5 -k ) · i, i ) : i ∈ [ n in ] } , and for any i ∈ [ n in ] , the (sparse) tensor M ( i ) ∈ R N i × 2 × N ⊗ 2 i is constructible in O (1) time using O ( N 2 i ) parallel processors.

The collection of tensors {M ( i ) } i ∈ [ n in ] can be interpreted as routers simulating the interventional mechanism in the Marginal SHAP formulation: Depending on the value assigned to its third index (which is binary), it routes the feature value of either its value in the input instance to explain x or from one drawn from the data generating distribution T P to feed the model's input.

## 5 Provably Exact and Tractable SHAP for TTs, and Other ML Models

In this section, we show that SHAP values can be computed in polynomial time for the specific case of Tensor Train (TT) models. Interestingly, we prove that the problem also lies in NC , meaning it can also be solved in polylogarithmic time using parallel computation. This result is established in Subsection 5.1. Then, in Subsection 5.2, we demonstrate how this finding can be leveraged - via reduction - to tighten complexity bounds for several other popular ML models.

Figure 2: The construction of the T ( M,P ) TN for a model of 3 features. The general case: Both T M and T P are general TNs with arbitrary structures; The TT case: Both T M and T P are TTs.

<!-- image -->

## 5.1 Provably exact and tractable SHAP explanations for TTs

The computational complexity of computing SHAP for general TNs, as discussed in the previous section, naturally depends on the structural properties of the TNs T M and T P implementing the model to explain and the data generating distribution, respectively. We begin by presenting a general worst-case negative result, showing that the problem is #P-Hard when no structural constraints are imposed on the tensor networks.

Proposition 2. Computing Marginal SHAP values for general TNs is # P-Hard .

Proof Sketch. The result is obtained by reduction from the #CNF-SAT problem (The problem of counting the number of satisfying assignments of a CNF boolean formula ). Essentially, it leverages two facts: (i) the model counting problem is polynomially reducible to the SHAP problem [67]. (ii) A polynomial-time algorithm to construct an equivalent TN to a given CNF boolean formula. The full proof can be found in Appendix D.

Interestingly, the problem however, becomes tractable when we restrict both T M and T P to lie within the class of Tensor Trains (TTs). To establish this result, we begin with a key observation: the marginal SHAP tensor V ( M,P ) itself admits a representation as a TT. Formally:

Theorem 1. Let T M = J I (1) , . . . , I ( n in ) K and T P = J P (1) , . . . , P ( n in ) K be two TTs corresponding to the model to interpret and the data-generating distribution, respectively. Then, the Marginal SHAP Tensor T ( M,P ) can be represented by a TT parametrized as:

<!-- formula-not-decoded -->

where the collection of Tensors {G ( i ) } i ∈ [ n in ] and {M ( i ) } i ∈ [ n in ] are implicitly defined in Lemma 1 and Lemma 2, respectively.

Proof Sketch. The result is obtained by plugging the TT corresponding to ˜ W (Lemma 1) and V ( M,P ) (Lemma 2) into Equation 12, and performing a suitable arrangement of contraction ordering of the resulting TN. Figure 2b provides a visual description of how the TT structure of the SHAP Value Tensor V ( M,P ) emerges when both T M and T P are TTs.

Efficient parallel computation of SHAP for TTs. Theorem 1 shows that computing exact SHAP values for TTs reduces to contracting the Marginal SHAP Tensor - a TT - with a rank-1 tensor representing the input (Equation 3). TT contraction is a well-studied problem with efficient parallel algorithms that run in poly-logarithmic time [84, 81], typically using a parallel scan over adjacent tensors. Leveraging this and the fact that matrix multiplication is in NC [28], we can prove the following:

Proposition 3. Computing Marginal SHAP for the family of TTs lies in NC 2 .

Proof Sketch. A parallel procedure to solve this problem runs as follows: First, compute in parallel tensor cores in Equation (20). Given that Matrix Multiplication is in NC 1 , this operation is also in NC 1 . Second, following [81], a parallel scan strategy will be applied to contract the resulting TT

Figure 3: Conversion of a decision tree into an equivalent Tensor Train (TT). (a) A simple decision tree with two binary input features, (b) its equivalent lattice of finite-state automata, where each automaton encodes a distinct path leading to a leaf labeled 1 , and (c) the corresponding TT representation with three free legs: x 1 , x 2 ,for inputs and y for the output. The tensor cores G (1) ∈ R 2 × 2 and G (2) ∈ R 2 × 2 × 2 have ranks equal to the number of automata in the lattice, i.e rank ( G ( i ) ) = 2 for i ∈ { 1 , 2 }

<!-- image -->

using a logarithmic depth of matrix multiplication operations. This second operation is performed by a circuit whose depth scales as O (log 2 ( n in )) yielding the result (see Appendix D).

## 5.2 Tightening the complexity results of SHAP computations in many other ML models

Thanks to the expressive power of Tensor Trains (TTs), many widely used ML models - such as tree ensembles, decision trees, linear models, and linear RNNs - can be reduced to TTs. This reduction allows us to significantly tighten the known complexity bounds for these models. This improvement is crucial for two main reasons:

- (i) It shows that computing SHAP values for all these models is not only polynomial-time solvable but also in the complexity class NC, enabling efficient parallel computation .
- (ii) It demonstrates that Marginal SHAP can be computed under TT-based distributions - a class of distributions more expressive than those previously considered, englobing independent [6], empirical [107], Markovian [82], Hidden Markov distributions [83] and Born Machines [52].

We formalize this in the following theorem:

Theorem 2. Computing Marginal SHAP values for decision trees, tree ensembles, linear models, and linear RNNs under the distribution class of TTs lies in NC 2 .

The proof of Theorem 2 can be found in Appendix E. It proceeds by constructing NC-reduction procedures that transform each of these ML models (i.e. Decision Trees, Tree Ensembles, Linear Models and Linear RNNs) into equivalent TTs.

For illustrative purposes, we show in figure 3 an example of such construction for a simple binary decision tree converted into an equivalent TT representation. The construction proceeds through an intermediate automata-based representation, where the DT is transformed into a lattice of finite state automata which admits a natural and compact parametrization in TT format. This construction can be implemented by means of a uniform family of boolean circuits with polynomial size and poly-logarithmic depth (see Appendix E for the formal details of this construction).

We believe that this result could be of interest to practitioners willing to scale the computation of SHAP explanations to large dimensions by leveraging parallelization under data generating distributions that capture sophisticated dependencies between input features.

## 6 A Fine-Grained Analysis of SHAP Computation for BNNs

In this section, we reveal an additional intriguing connection between our complexity results for TNs and Binarized Neural Networks (BNNs). This connection enables what is, to the best of our knowledge, the first fine-grained analysis of how particular structural parameters of a neural network affect the complexity of computing SHAP values.

Webegin by noting that for a non-quantized neural network, even the seemingly simple case of a single sigmoid neuron with binary inputs has been shown to be intractable for SHAP computation [107, 83].

Figure 4: In Figure 4a, R ij denotes the reified cardinality parameters of the neurons. In Figures 4b, 4c, and 4d, the numbers above the edges indicate tensor index dimensionality.

<!-- image -->

In contrast, we show that by quantizing the weights and transitioning to a BNN, tractability can be achieved - though this tractability critically depends on the network's different structural parameters.

We carry out this analysis using the framework of parameterized complexity [48, 46], a standard approach in complexity theory for understanding how various structural parameters influence computational hardness. Specifically, we focus on three key parameters: (i) the network's width , (ii) its depth , and (iii) its sparsity . The main result of this analysis is captured in the following theorem:

Theorem 3. Let P be either the class of empirical distributions, independent distributions, or the class of TTs. We have that:

1. Bounded Depth: The problem of computing SHAP for BNNs under any distribution class P is PARA-NP-HARD with respect to the network's depth parameter.
2. Bounded Width: The problem of computing SHAP for BNNs under any distribution class P is in XP with respect to the width parameter.
3. Bounded Width and Sparsity. The problem of computing SHAP for BNNs under any distribution class P is in FPT with respect to the width and reified cardinality parameters.

Key takeaways from these fine-grained results. Theorem 3 captures several notable insights into the complexity of computing exact SHAP values for BNNs:

1. Even shallow networks are hard. Computing SHAP values for BNNs remains NP-Hard even when the network depth is fixed to a constant . In fact, intractability arises already with a single hidden layer, underscoring that reducing the network's depth does not alleviate the computational hardness of SHAP.
2. Narrow networks make it easier. However, when the width of the neural network is fixed to a constant, computing SHAP for a BNN becomes polynomial (as it falls within the class XP), highlighting width as a potential relaxation point for improving tractability.
3. Narrow and sparse networks are efficient. Finally, by fixing both the network's width and sparsity , we obtain an even stronger result: fixed-parameter tractability (FPT) . This implies that computing SHAP is efficiently tractable -even for arbitrarily large networks - so long as these parameters remain small, regardless of the network's depth, input dimensionality, or number of non-linear activations.

This shows that width drives the shift from intractability to tractability in SHAP for BNNs, and that adding sparsity bounds can fully tame complexity - even in high-dimensional settings.

Proof Sketch (Theorem 3). The first part follows from a direct reduction from 3SAT: any CNF formula can be encoded by a depth-2 BNN, and computing SHAP for CNF formulas is #P-Hard [107].

The second and third parts rely on a more involved construction: converting a BNN into an equivalent Tensor Train (TT), inspired by [73]. Each BNN layer is represented as a 2D tensor network, and these are contracted backwards into a TT. Figure 4 illustrates this process. The compilation of a BNN into an equivalent TT runs in O ( R W · poly ( D,n in , max i N i )) time, where W , D , and R are the width, depth, and reified cardinality of the BNN. The last two items of Theorem 3 follows immediately from this runtime complexity by definition of XP and FPT (see section 2), and the fact that SHAP for TTs is in NC (Proposition 3).

## 7 Limitations and Future Work

While our work represents a notable step toward understanding the computational landscape of SHAP, it remains focused on specific settings, and many other settings could naturally be explored. Future research could extend our analysis to additional model classes, such as Tree Tensor Networks [35], SHAP variants [104], and relaxations or approximations that enhance tractability for even more expressive models than those studied here. We also acknowledge existing critiques of SHAP [71, 57, 100, 49, 25]; our goal is not to defend its axiomatic foundations but to analyze the complexity of a widely adopted explanation method. Exploring the complexity of obtaining alternative value function definitions [72], SHAP variants [83], or other attribution indices [10, 114] presents exciting directions for future work.

## 8 Conclusion

In this work, we present the first provably exact algorithm for computing SHAP explanations for the class of Tensor Networks . Moreover, we prove that for the particular subclass of Tensor Trains , this computation can be carried out not only in polynomial time but also in polylogarithmic time using parallel processors. This result closes a significant expressivity gap in tractable SHAP computation by extending tractability to a significantly more expressive model family than previously known. Building on this expressivity, our approach also yields new insights into the complexity of computing SHAP for other popular ML models - including decision trees, linear models, linear RNNs, and tree ensembles - by enabling improved parallelizability-related complexity bounds and more expressive distribution modeling. Furthermore, these results offer, by reduction, a novel fine-grained analysis of the tractability barriers in computing SHAP for binarized neural networks, identifying width as a central computational bottleneck. Together, we believe that these findings significantly advance our understanding of the computational landscape of SHAP, highlighting both inherent limitations and new opportunities, and hence paving the way for future research.

## Acknowledgments

This work was partially funded by the European Union (ERC, VeriDeL, 101112713). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them. This research was additionally supported by a grant from the Israeli Science Foundation (grant number 558/24).

## References

- [1] Omer Abramovich, Daniel Deutch, Nave Frost, Ahmet Kara, and Dan Olteanu. Banzhaf Values for Facts in Query Answering. In Proc. 2nd ACM on Management of Data (PACMMOD) , pages 1-26, 2024.
- [2] Federico Adolfi, Martina Vilas, and Todd Wareham. The Computational Complexity of Circuit Discovery for Inner Interpretability. In Proc. 13th Int. Conf. on Learning Representations (ICLR) , 2025.
- [3] Borja Aizpurua, Samuel Palmer, and Román Orús. Tensor Networks for Explainable Machine Learning in Cybersecurity. Neurocomputing , 639:130211, 2025.

- [4] Mazen Ali and Anthony Nouy. Approximation Theory of Tree Tensor Networks: Tensorized Multivariate Functions, 2024. Technical Report. https://arxiv.org/abs/2101.11932 .
- [5] Guy Amir, Shahaf Bassan, and Guy Katz. Hard to Explain: On the Computational Hardness of In-Distribution Model Interpretation. In Proc. 27th European Conf. on Artificial Intelligence (ECAI) , pages 818-825, 2024.
- [6] Marcelo Arenas, Pablo Barcelo, Leopoldo Bertossi, and Mikael Monet. On the Complexity of SHAP-Score-Based Explanations: Tractability via Knowledge Compilation and NonApproximability Results. Journal of Machine Learning Research (JMLR) , 24(63):1-58, 2023.
- [7] Sanjeev Arora and Boaz Barak. Computational Complexity: A Modern Approach. 2009.
- [8] Gilles Audemard, Steve Bellart, Louenas Bounia, Frédéric Koriche, Jean-Marie Lagniez, and Pierre Marquis. On Preferred Abductive Explanations for Decision Trees and Random Forests. In Proc. 31st Int. Joint Conf. on Artificial Intelligence (IJCAI) , pages 643-650, 2022.
- [9] Gilles Audemard, Steve Bellart, Louenas Bounia, Frédéric Koriche, Jean-Marie Lagniez, and Pierre Marquis. Trading Complexity for Sparsity in Random Forest Explanations. In Proc. 36th AAAI Conf. on Artificial Intelligence , pages 5461-5469, 2022.
- [10] P Barceló, R Cominetti, and M Morgado. When is the Computation of a Feature Attribution Method Tractable? 2025. Technical Report. https://arXiv:2501.02356 .
- [11] Pablo Barceló, Mikaël Monet, Jorge Pérez, and Bernardo Subercaseaux. Model Interpretability Through the Lens of Computational Complexity. In Proc. 34th Conf. on Neural Information Processing Systems (NeurIPS) , pages 15487-15498, 2020.
- [12] Pablo Barceló, Alexander Kozachinskiy, Miguel Romero, Bernardo Subsercaseaux, and José Verschae. Explaining k-Nearest Neighbors: Abductive and Counterfactual Explanations. In Proc. 3rd ACM on Management of Data , number 2, pages 1-26, 2025.
- [13] Clark Barrett and Cesare Tinelli. Satisfiability Modulo Theories. Handbook of model checking , 2018.
- [14] Shahaf Bassan and Guy Katz. Towards Formal XAI: Formally Approximate Minimal Explanations of Neural Networks. In Proc. 29th Int. Conf. on Tools and Algorithms for the Construction and Analysis of Systems (TACAS) , pages 187-207, 2023.
- [15] Shahaf Bassan, Guy Amir, Davide Corsi, Idan Refaeli, and Guy Katz. Formally Explaining Neural Networks within Reactive Systems. In Proc. 23rd Conf. on Formal Methods in Computer-Aided Design (FMCAD) , pages 1-13, 2023.
- [16] Shahaf Bassan, Guy Amir, and Guy Katz. Local vs. Global Interpretability: A Computational Complexity Perspective. In Proc. 41st Int. Conf. on Machine Learning (ICML) , 2024.
- [17] Shahaf Bassan, Guy Amir, Meirav Zehavi, and Guy Katz. What makes an Ensemble (Un) Interpretable? In Proc. 42nd Int. Conf. on Machine Learning (ICML) , 2025.
- [18] Shahaf Bassan, Yizhak Yisrael Elboher, Tobias Ladner, Matthias Althoff, and Guy Katz. Explaining, Fast and Slow: Abstraction and Refinement of Provable Explanations. In Proc. 42nd Int. Conf. on Machine Learning (ICML) , 2025.
- [19] Shahaf Bassan, Ron Eliav, and Shlomit Gur. Explain Yourself, Briefly! Self-Explaining Neural Networks with Concise Sufficient Reasons. In Proc. 13th Int. Conf. on Learning Representations (ICLR) , 2025.
- [20] Shahaf Bassan, Shlomit Gur, Sergey Zeltyn, Konstantinos Mavrogiorgos, Ron Eliav, and Dimosthenis Kyriazis. Self-Explaining Neural Networks for Business Process Monitoring. 2025. Technical Report. https://arXiv:2503.18067 .
- [21] Shahaf Bassan, Michal Moshkovitz, and Guy Katz. Additive Models Explained: A Computational Complexity Approach. In Proc. 39th Conf. on Neural Information Processing Systems (NeurIPS) , 2025.

- [22] Leopoldo Bertossi, Jordan Li, Maximilian Schleich, Dan Suciu, and Zografoula Vagena. Causality-based Explanation of Classification Outcomes. In Proc. of the 4th Int. Workshop on Data Management for End-to-End Machine Learning , 2020.
- [23] Leopoldo Bertossi, Benny Kimelfeld, Ester Livshits, and Mikaël Monet. The Shapley Value in Database Management. ACM Sigmod Record , pages 6-17, 2023.
- [24] Jacob Biamonte. Lectures on Quantum Tensor Networks. 2020. Technical Report. https: //arxiv.org/abs/1912.10049 .
- [25] Gagan Biradar, Yacine Izza, Elita Lobo, Vignesh Viswanathan, and Yair Zick. Axiomatic Aggregations of Abductive Explanations. In Proc. of the 38th AAAI Conf. on Artificial Intelligence , number 10, pages 11096-11104, 2024.
- [26] Guy Blanc, Jane Lange, and Li-Yang Tan. Provably Efficient, Succinct, and Precise Explanations. In Proc. 35th Conf. on Neural Information Processing Systems (NeurIPS) , pages 6129-6141, 2021.
- [27] Sebastian Bordt and Ulrike von Luxburg. From Shapley values to Generalized Additive Models and Back. In Proc. 26th Int. Conf. on Artificial Intelligence and Statistics (AISTATS) , pages 709-745. PMLR, 2023.
- [28] Allan Borodin, Joachim von zur Gathen, and John E. Hopcroft. Fast Parallel Matrix and GCD Computations. Information and Control , 52(3):241-256, 1982.
- [29] Louenas Bounia and Frederic Koriche. Approximating Probabilistic Explanations via Supermodular Minimization. In Proc. 39th Conf. on Uncertainty in Artificial Intelligence (UAI) , pages 216-225, 2023.
- [30] Marco Calautti, Enrico Malizia, and Cristian Molinaro. On the Complexity of Global Necessary Reasons to Explain Classification. 2025. Technical Report. https://arXivpreprintarXiv: 2501.06766 .
- [31] Javier Castro, Daniel Gómez, and Juan Tejada. Polynomial Calculation of the Shapley Value Based on Sampling. Computers &amp; Operations Research , 36(5):1726-1730, 2009.
- [32] Siu Lun Chau, Robert Hu, Javier Gonzalez, and Dino Sejdinovic. RKHS-SHAP: Shapley Values for Kernel Methods. Proc. of the 36th Conf. on Neural Information Processing Systems (NeurIPS) , pages 13050-13063, 2022.
- [33] Siu Lun Chau, Krikamol Muandet, and Dino Sejdinovic. Explaining the Uncertain: Stochastic Shapley Values for Gaussian Process Models. In Proc. 37th Conf. on Neural Information Processing Systems (NeurIPS) , 2023.
- [34] Cong Chen, Kim Batselier, and Ngai Wong. Chapter 8 - Tensor Network Algorithms for Image Classification. In Yipeng Liu, editor, Tensors for Data Processing , pages 249-291. Academic Press, 2022.
- [35] Hao Chen and Thomas Barthel. Machine learning with tree tensor networks, cp rank constraints, and tensor dropout. IEEE Transactions on Pattern Analysis and Machine Intelligence , 46(12): 7825-7832, 2024. doi: 10.1109/TPAMI.2024.3396386.
- [36] Hao Chen and Thomas Barthel. Machine Learning With Tree Tensor Networks, CP Rank Constraints, and Tensor Dropout. IEEE Trans. Pattern Anal. Mach. Intell. , 46(12):7825-7832, December 2024. ISSN 0162-8828.
- [37] Hugh Chen, Scott Lundberg, and Su-In Lee. Explaining Models by Propagating Shapley Values of Local Components. Explainable AI in Healthcare and Medicine: Building a Culture of Transparency and Accountability , pages 261-270, 2021.
- [38] Lu Chen, Siyu Lou, Keyan Zhang, Jin Huang, and Quanshi Zhang. HarsanyiNet: Computing Accurate Shapley Values in a Single Forward Propagation. In Proc. 40th Int. Conf. on Machine Learning (ICML) , pages 4804-4825, 2023.

- [39] Stephen A Cook. Towards a Complexity Theory of Synchronous Parallel Computation. In Logic, Automata, and Computational Complexity: The Works of Stephen A. Cook , pages 219-244. 2023.
- [40] Martin C Cooper and João Marques-Silva. Tractability of Explaining Classifier Decisions. Artificial Intelligence , 2023.
- [41] Matthieu Courbariaux, Itay Hubara, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1. Technical Report. https://arxiv.org/abs/1602.02830 .
- [42] Ian Covert and Su-In Lee. Improving KernelSHAP: Practical Shapley Value Estimation via Linear Regression. Proc. 24th Int. Conf. on Artificial Intelligence and Statistics (AISTATS) , pages 1-9, 2021.
- [43] Adnan Darwiche and Auguste Hirth. On the Reasons Behind Decisions. In Proc. 24th European Conf. on Artifical Intelligence (ECAI) , pages 712-720, 2020.
- [44] Adnan Darwiche and Chunxi Ji. On the Computation of Necessary and Sufficient Explanations. In Proc. 36th AAAI Conf. on Artificial Intelligence , pages 5582-5591, 2022.
- [45] Daniel Deutch, Nave Frost, Benny Kimelfeld, and Mikaël Monet. Computing the Shapley Value of Facts in Query Answering. In Proc. Int. Conf. on Management of Data (MOD) , pages 1570-1583, 2022.
- [46] Rodney G Downey and Michael Ralph Fellows. Parameterized complexity . Springer Science &amp;Business Media, 2012.
- [47] James Enouen and Yan Liu. InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly. In Proc. 13th Int. Conf. on Learning Representations (ICLR) , 2025.
- [48] Jörg Flum and Martin Grohe. Parameterized Complexity Theory. pages 1-495, 2006.
- [49] Daniel Fryer, Inga Strümke, and Hien Nguyen. Shapley Values for Feature Selection: The Good, the Bad, and the Axioms. Ieee Access , 9:144352-144360, 2021.
- [50] Fabian Fumagalli, Maximilian Muschalik, Patrick Kolpaczki, Eyke Hüllermeier, and Barbara Hammer. SHAP-IQ: Unified Approximation of Any-Order Shapley Interactions. In Proc. 37th Conf. on Neural Information Processing Systems (NeurIPS) , pages 11515-11551, 2023.
- [51] Fabian Fumagalli, Maximilian Muschalik, Patrick Kolpaczki, Eyke Hüllermeier, and Barbara Hammer. KernelSHAP-IQ: Weighted Least Square Optimization for Shapley Interactions. In Proc. 41st Int. Conf. on Machine Learning (ICML) , pages 14308-14342, 2024.
- [52] Ivan Glasser, Ryan Sweke, Nicola Pancotti, Jens Eisert, and Ignacio Cirac. Expressive Power of Tensor-Network Factorizations for Probabilistic Modeling. In Proc. 33rd Conf. on Neural Information Processing Systems (NeurIPS) , 2019.
- [53] Carys Harvey, Richie Yeung, and Konstantinos Meichanetzidis. Sequence Processing with Quantum-Inspired Tensor Networks. Scientific Reports , 15(1):7155, 2025. doi: 10.1038/ s41598-024-84295-2.
- [54] Kohei Hayashi, Taiki Yamaguchi, Yohei Sugawara, and Shin-ichi Maeda. Exploring Unexplored Tensor Network Decompositions for Convolutional Neural Networks. In Proc. 33rd Conf. on Neural Information Processing Systems (NeurIPS) , 2019.
- [55] Tom Heskes, Evi Sijben, Ioan Gabriel Bucur, and Tom Claassen. Causal Shapley Values: Exploiting Causal Knowledge to Explain Individual Predictions of Complex Models. In Proc. 34th Conf. on Neural Information Processing Systems (NeurIPS) , pages 4778-4789, 2020.
- [56] John E. Hopcroft, Rajeev Motwani, and Jeffrey D. Ullman. Introduction to Automata Theory, Languages, and Computation. 2006.
- [57] Xuanxiang Huang and Joao Marques-Silva. On the Failings of Shapley Values for Explainability. Int. Journal of Approximate Reasoning (IJAR) , 171:109112, 2024.

- [58] Xuanxiang Huang and Joao Marques-Silva. Updates on the Complexity of SHAP Scores. In Proc. of the 33rd Int. Joint Conf. on Artificial Intelligence (IJCAI) , pages 403-412, 2024.
- [59] Alexey Ignatiev. Towards Trustable Explainable AI. In Proc. 29th Int. Joint Conf. on Artificial Intelligence (IJCAI) , pages 5154-5158, 2020.
- [60] Alexey Ignatiev, Nina Narodytska, and Joao Marques-Silva. Abduction-based Explanations for Machine Learning Models. In Proc. 33rd AAAI Conf. on Artificial Intelligence , pages 1511-1519, 2019.
- [61] Yacine Izza, Xuanxiang Huang, Antonio Morgado, Jordi Planes, Alexey Ignatiev, and Joao Marques-Silva. Distance-Restricted Explanations: Theoretical Underpinnings &amp; Efficient implementation. In Proc. 21st Int. Conf. on Principles of Knowledge Representation and Reasoning (KR) , pages 475-486, 2024.
- [62] Dominik Janzing, Lenon Minorics, and Patrick Blöbaum. Feature Relevance Quantification in Explainable AI: A causal Problem. In Proc. 23rd Int. Conf. on Artificial Intelligence and Statistics (AISTATS) , pages 2907-2916, 2020.
- [63] Neil Jethani, Mukund Sudarshan, Ian Connick Covert, Su-In Lee, and Rajesh Ranganath. FastSHAP: Real-time Shapley Value Estimation. In Proc. 9th Int. Conf. on Learning Representations (ICLR) , 2021.
- [64] Kai Jia and Martin Rinard. Efficient Exact Verification of Binarized Neural Networks. In Proc. 34th Conf. on Neural Information Processing Systems (NeurIPS) , pages 1782-1795, 2020.
- [65] Helen Jin, Anton Xue, Weiqiu You, Surbhi Goel, and Eric Wong. Probabilistic Stability Guarantees for Feature Attributions. 2025. Technical Report. https://arXiv:2504.13787 .
- [66] Ahmet Kara, Dan Olteanu, and Dan Suciu. From Shapley Value to Model Counting and Back. In Proc. 2nd ACM on Management of Data (PACMMOD) , pages 1-23, 2024.
- [67] Ahmet Kara, Dan Olteanu, and Dan Suciu. From Shapley Value to Model Counting and Back. Proc. ACM Manag. Data , 2(2), May 2024.
- [68] Pratik Karmakar, Mikaël Monet, Pierre Senellart, and Stéphane Bressan. Expected Shapleylike Scores of Boolean Functions: Complexity and Applications to Probabilistic Databases. In Proc. 2nd ACM on Management of Data (PACMMOD) , pages 1-26, 2024.
- [69] Guy Katz, Clark Barrett, David Dill, Kyle Julian, and Mykel J Kochenderfer. Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks. In Proc. 29th Int. Conf. on Computer Aided Verification (CAV) , pages 97-117, 2017.
- [70] Sven Kruschel, Nico Hambauer, Sven Weinzierl, Mathias Kraus, Sandra Zilker, and Patrick Zschech. Challenging the Performance-Interpretability Trade-Off: An Evaluation of Interpretable Machine Learning Models. Business &amp; Information Systems Engineering , 2025.
- [71] I Elizabeth Kumar, Suresh Venkatasubramanian, Carlos Scheidegger, and Sorelle Friedler. Problems with Shapley-Value-based Explanations as Feature Importance Measures. In Proc. 37th Int. Conf. on Machine Learning (ICML) , pages 5491-5500, 2020.
- [72] Olivier Létoffé, Xuanxiang Huang, and Joao Marques-Silva. Towards Trustable SHAP Scores. In Proc. 39th AAAI Conf. on Artificial Intelligence , number 17, pages 18198-18208, 2025.
- [73] Sujie Li, Feng Pan, Pengfei Zhou, and Pan Zhang. Boltzmann Machines as Two-Dimensional Tensor Networks. Phys. Rev. B , 104:075154, 2021.
- [74] Yuhan Liu, Xiao Zhang, Maciej Lewenstein, and Shi-Ju Ran. Entanglement-Based Feature Extraction by Tensor Network Machine Learning. In Frontiers in Applied Mathematics and Statistics , 2018.
- [75] Ester Livshits, Leopoldo Bertossi, Benny Kimelfeld, and Moshe Sebag. The Shapley Value of Tuples in Query Answering. Logical Methods in Computer Science , 2021.

- [76] Scott M. Lundberg and Su-In Lee. A Unified Approach to Interpreting Model Predictions. In Proc. 31st Conf. on Neural Information Processing Systems (NeurIPS) , pages 4765-4774, 2017.
- [77] Scott M. Lundberg, Gabriel G. Erion, Hugh Chen, Ankur De, Jonas J. Lee, Stephanie Mattox, Jared A. Harris, Karlina J. Grimsman, and et al. From Local Explanations to Global Understanding with Explainable AI for Trees. Nature Machine Intelligence , 2(1):56-67, 2020.
- [78] Linjian Ma and Edgar Solomonik. Cost-Efficient Gaussian Tensor Network Embeddings for Tensor-Structured Inputs. In Proc. 36th Conf. on Neural Information Processing Systems (NeurIPS) , pages 38980-38993, 2022.
- [79] Joao Marques-Silva. Logic-based Explainability in Machine Learning. In Proc. 18th Int. Summer School of Reasoning Web. Causality, Explanations and Declarative Knowledge , pages 24-104, 2023.
- [80] Joao Marques-Silva and Xuanxiang Huang. Explainability is NOT a Game. Communications of the ACM , pages 66-75, 2024.
- [81] Eric Martin and Chris Cundy. Parallelizing Linear Recurrent Neural Nets Over Sequence Length. In Proc. 6th Int. Conf. on Learning Representations (ICLR) , 2018.
- [82] Reda Marzouk and Colin De La Higuera. On the Tractability of SHAP Explanations under Markovian Distributions. In Proc. of the 41st Int. Conf. on Machine Learning (ICML) , pages 34961-34986, 2024.
- [83] Reda Marzouk, Shahaf Bassan, Guy Katz, and De la Higuera. On the Computational Tractability of the (Many) Shapley Values. In Proc. 28th Int. Conf. on Artificial Intelligence and Statistics (AISTATS) , 2025.
- [84] Jacob Miller, Guillaume Rabusseau, and John Terilla. Tensor Networks for Probabilistic Sequence Modeling. In Proc. 24th Int. Conf. on Artificial Intelligence and Statistics (AISTATS) , pages 3079-3087, 2021.
- [85] Rory Mitchell, Eibe Frank, and Geoffrey Holmes. GPUTreeShap: Massively Parallel Exact Calculation of SHAP Scores for Tree Ensembles. arXiv preprint arXiv:2010.13972 , 2020. Technical Report. https://arxiv.org/abs/2010.13972 .
- [86] Rory Mitchell, Joshua Cooper, Eibe Frank, and Geoffrey Holmes. Sampling Permutations for Shapley Value Estimation. Journal of Machine Learning Research (JMLR) , 23(43):1-46, 2022.
- [87] Maximilian Muschalik, Fabian Fumagalli, Barbara Hammer, and Eyke Hüllermeier. Beyond TreeSHAP: Efficient Computation of Any-Order Shapley Interactions for Tree Ensembles. In Proc. 38th AAAI Conf. on Artificial Intelligence , number 13, pages 14388-14396, 2024.
- [88] Maximilian Muschalik, Fabian Fumagalli, Paolo Frazzetto, Janine Strotherm, Luca Hermes, Alessandro Sperduti, Eyke Hüllermeier, and Barbara Hammer. Exact Computation of AnyOrder Shapley Interactions for Graph Neural Networks. In Proc. 13th Int. Conf. on Learning Representations (ICLR) , 2025.
- [89] Christopher Musco and R. Teal Witter. Provably Accurate Shapley Value Estimation via Leverage Score Sampling. In Proc. 13th Int. Conf. on Learning Representations (ICLR) , 2025.
- [90] Alexander Novikov, Dmitrii Podoprikhin, Anton Osokin, and Dmitry P. Vetrov. Tensorizing Neural Networks. In Proc. 29th Conf. on Neural Information Processing Systems (NeurIPS) , pages 442-450, 2015.
- [91] Georgii S Novikov, Maxim E Panov, and Ivan V Oseledets. Tensor-Train Density Estimation. In Proc. 37th Conf. on Uncertainty in Artificial Intelligence (UAI) , pages 1321-1331, 2021.
- [92] Sebastian Ordyniak, Giacomo Paesani, and Stefan Szeider. The Parameterized Complexity of Finding Concise Local Explanations. In Proc. 32nd Int. Joint Conf. on Artificial Intelligence (IJCAI) , pages 3312-3320, 2023.

- [93] V. Oseledets, I.˙Tensor-Train Decomposition. SIAM Journal on Scientific Computing , 33(5): 2295-2317, 2011.
- [94] Vasily Pestun and Yiannis Vlassopoulos. Tensor Network Language Model, 2017. Technical Report. https://arxiv.org/abs/1710.10248 .
- [95] Guillaume Rabusseau, Tianyu Li, and Doina Precup. Connecting Weighted Automata and Recurrent Neural Networks through Spectral Learning. In Proc. 22nd Int. Conf. on Artificial Intelligence and Statistics (AISTATS) , pages 1630-1639, 2019.
- [96] Shi-Ju Ran and Gang Su. Tensor Networks for Interpretable and Efficient Quantum-Inspired Machine Learning. Computing , 2:0061, 2023.
- [97] Martin Roa-Villescas, Xuanzhao Gao, Sander Stuijk, Henk Corporaal, and Jin-Guo Liu. Probabilistic Inference in the Era of Tensor Networks and Differential Programming. Phys. Rev. Res. , 6:033261, 2024.
- [98] Cynthia Rudin. Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead. Nature Machine Intelligence , 1(5):206-215, 2019.
- [99] Raghavendra Selvan and Erik B. Dam. Tensor Networks for Medical Image Classification. In Proc. Int. Conf. on Medical Imaging with Deep Learning (MIDL) , 2020.
- [100] Dylan Slack, Sophie Hilgard, Emily Jia, Sameer Singh, and Himabindu Lakkaraju. Fooling LIME and SHAP: Adversarial Attacks on Post Hoc Explanation Methods. In Proc. AAAI/ACM Conference on AI, Ethics, and Society (AIES) , pages 180-186, 2020.
- [101] Mihail Stoian, Richard Milbradt, and Christian Mendl. NP-Hardness of Optimal TensorNetwork Contraction and Polynomial-Time Algorithms for Tree Tensor Networks. Quantum , 6:e119, 2022.
- [102] E. M. Stoudenmire and David J. Schwab. Supervised Learning with Tensor Networks. In Proc. 30th Conf. on Neural Information Processing Systems (NeurIPS) , page 4806-4814, 2016.
- [103] Zhan Su, Yuqin Zhou, Fengran Mo, and Jakob Grue Simonsen. Language Modeling Using Tensor Trains, 2024. Technical Report. https://arxiv.org/abs/2405.04590 .
- [104] Mukund Sundararajan and Amir Najmi. The Many Shapley Values for Model Explanation. In Proc. 37th Int. Conf. on Machine Learning (ICML) , pages 9269-9277, 2019.
- [105] Mukund Sundararajan, Kedar Dhamdhere, and Ashish Agarwal. The Shapley Taylor Interaction Index. In Proc. 37th Int. Conf. on Machine Learning (ICML) , pages 9259-9268, 2020.
- [106] Muhammad Taufiq, Patrick Blöbaum, and Lenon Minorics. Manifold Restricted Interventional Shapley Values. In Proc. 26th Int. Conf. on Artificial Intelligence and Statistics (AISTATS) , pages 5079-5106, 2023.
- [107] Guy Van den Broeck, Anton Lykov, Maximilian Schleich, and Dan Suciu. On the Tractability of SHAP Explanations. Journal of Artificial Intelligence Research (JAIR) , 72:861-886, 2021.
- [108] Stephan Wäldchen, Jan Macdonald, Sascha Hauch, and Gitta Kutyniok. The Computational Complexity of Understanding Binary Classifier Decisions. Journal of Artificial Intelligence Research (JAIR) , 70:351-387, 2021.
- [109] Shiqi Wang, Huan Zhang, Kaidi Xu, Xue Lin, Suman Jana, Cho-Jui Hsieh, and J Zico Kolter. Beta-Crown: Efficient Bound Propagation with Per-Neuron Split Constraints for Neural Network Robustness Verification. In Proc. 35th Conf. on Neural Information Processing Systems (NeurIPS) , pages 29909-29921, 2021.
- [110] Haoze Wu, Omri Isac, Aleksandar Zelji´ c, Teruhiro Tagomori, Matthew Daggitt, Wen Kokke, Idan Refaeli, Guy Amir, Kyle Julian, Shahaf Bassan, et al. Marabou 2.0: A Versatile Formal Analyzer of Neural Networks. In Proc. 36th Int. Conf. on Computer Aided Verification (CAV) , pages 249-264, 2024.

- [111] Jianyu Xu, Wei Li, and Ming Zhao. Complexity of Optimal Tensor Network Contraction Sequences. Journal of Computational Physics , 480:112237, 2023.
- [112] Anton Xue, Rajeev Alur, and Eric Wong. Stability Guarantees for Feature Attributions with Multiplicative Smoothing. Proc. 37th Conf. on Neural Information Processing Systems (NeurIPS) , pages 62388-62413, 2023.
- [113] Jilei Yang, Rui Zhang, Cong Li, Meng Zhang, Chao Zhang, Tao Zhang, and Chao Zhang. Fast TreeSHAP: Accelerating SHAP Value Computation for Trees. 2021. Technical Report. https://arxiv.org/abs/2109.09847 .
- [114] Jinqiang Yu, Graham Farr, Alexey Ignatiev, and Peter J Stuckey. Anytime Approximate Formal Feature Attribution. In Proc. 27th Int. Conf. on Theory and Applications of Satisfiability Testing (SAT) , pages 30-1, 2024.
- [115] Peng Yu, Chao Xu, Albert Bifet, and Jesse Read. Linear TreeSHAP. In Proc. 36th Conf. on Neural Information Processing Systems (NeurIPS) , pages 32112-32123, 2022.
- [116] Artjom Zern, Klaus Broelemann, and Gjergji Kasneci. Interventional SHAP Values and Interaction Values for Piecewise Linear Regression Trees. In Proc. of the AAAI Conf. on Artificial Intelligence , number 9, pages 11164-11173, 2023.
- [117] Erik Štrumbelj and Igor Kononenko. Explaining Prediction Models and Individual Predictions with Feature Contributions. Knowledge and Information Systems , 41(3):647-665, 2014.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction directly represent the proofs developed in this work.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We provide a dedicated Limitations section.

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

Justification: Owing to space limitations, the main paper presents only a sketch of central proofs, with the complete versions of all proofs in the appendix.

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

Justification: This is a theoretical work, so experimental reproducibility is not relevant.

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

Justification: This is a theoretical work, so providing open access to data and code is not applicable.

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

Justification: This is a theoretical work, and no hyperparameter details are needed.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA] .

Justification: This is a theoretical work, so such statistical tests are not required.

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

Justification: This is a theoretical work, so no details on computational resources are required.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes] .

Justification: We have read the NeurIPS Code of Ethics Guidelines and ensured that our paper fully complies with them.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA] .

Justification: While we recognize the potential social implications of interpretability, our work is primarily theoretical in nature and, as such, does not pose any direct social consequences.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA] .

Justification: Our work is theoretical; therefore, safeguards are not applicable.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA] .

Justification: This work is theoretical; therefore, assets of this type are not relevant.

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

Answer: [NA] .

Justification: This work is theoretical; therefore, assets of this type are not relevant.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA] .

Justification: This work is theoretical; no such crowdsourcing experiments were conducted.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA] .

Justification: This work is theoretical; no such user study experiments were conducted.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA] .

Justification: This is a theoretical study and does not involve any non-standard use of LLMs.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

The appendix is organized as follows:

Appendix A provides extended related work that is relevant to this work.

Appendix B provides the technical background, including descriptions of the model families studied in this work, as well as other technical tools - such as Copy Tensors and Layered Deterministic Finite Automata (LDFAs) - used in the proofs of various results presented in the paper.

Appendix C provides proofs related to the exact SHAP computation for Tensor Networks with arbitrary structures, namely Proposition 1, Lemma 1, and Lemma 2.

Appendix D provides the proof of the #P-Hardness of computing SHAP for general TNs (Proposition 2), along with a proof of the NC complexity result for the case of TTs (Proposition 3).

Appendix E presents the proof of Theorem 2, detailing the NC reductions from the SHAP computation problem for decision trees, tree ensembles, linear models, and linear RNNs to the corresponding SHAP problem for Tensor Trains (TTs).

Appendix F provides proofs on the fine-grained analysis of BNNs and their connection to TNs (Theorem 3).

## A Extended Related Work

This section offers an expanded discussion of related work, with particular emphasis on key complexity results established in prior studies.

SHAP values. Building on the original SHAP framework introduced by [76] for explaining machine learning predictions, a substantial body of subsequent work has explored its application across a wide range of XAI settings. This includes the development of numerous SHAP variants [104, 62, 55], adaptations that align SHAP with underlying data manifolds [49, 106], and extensions that attribute higher-order feature interactions [50, 51, 105, 87]. From the computational perspective, a variety of statistical approximation techniques have been proposed to improve scalability [50, 105, 87, 88], alongside model-specific implementations for classes such as linear models [76], tree-based models [115, 85, 113, 87, 77], additive models [47, 21, 27], and kernel methods [32, 33]. Closer to our line of work, recent efforts aim to construct neural network architectures that support efficient computation of SHAP values [88, 38]. Finally, several parallel research threads have highlighted important limitations and potential failure modes of SHAP across different settings [49, 57, 71, 80].

The computational complexity of SHAP. Notably, [107] study SHAP when the value function is defined via conditional expectations and establish both tractability and intractability results across various ML model classes. Building on this, [6] generalize these findings, showing that tractability for SHAP using conditional expectations coincides precisely with the class of Decomposable Deterministic Boolean Circuits, and further prove that both decomposability and determinism are necessary for tractable computation. More recently, [82] extend the analysis beyond independent feature distributions to Markovian distributions, which was later generalized to HMM-distributed features [83], alongside an exploration of additional SHAP variants - such as baseline and marginal SHAP - as well as both local and global computation settings. Furthermore, [58, 83] distinguish between the complexity of SHAP for regression versus classification tasks. Other, more tangential but relevant extensions of these complexity studies include analyzing SHAP computations in database settings [45, 75, 23, 66, 68], as well as investigating the complexity of alternative attribution schemes beyond Shapley values - such as Banzhaf values [1] and other cooperative game-theoretic values [10]. In contrast, our work significantly broadens existing complexity results by considering much more expressive classes for both the underlying prediction models and the data distributions. Moreover, we are the first to provide a complexity-theoretic analysis of the parallelizability of SHAP, as well as fine-grained complexity bounds for computing SHAP values in families of neural networks.

## Appendix

Formal XAI. More broadly, this work lies within the emerging area of formal explainable AI (formal XAI) [79, 14], which seeks to derive explanations for machine learning models that come with mathematical guarantees [59, 14, 43, 44, 18, 60, 20, 8, 112, 65, 29]. Explanations in this setting are typically produced through automated reasoning techniques, such as SMT solvers [13] (e.g., for explaining tree ensembles [9]) and neural network verifiers [69, 110, 109] (e.g., for explaining neural networks [61, 15]). Since computing explanations with provable guarantees is often computationally intractable, a central line of work in formal XAI focuses on analyzing the computational complexity of explanation problems [11, 108, 40, 16, 26, 5, 19, 17, 2, 12, 30, 92].

## B Background

This section begins by formally introducing the ML models examined in this work. It then presents the concept of copy tensors, followed by an introduction to Layered Deterministic Finite Automata (LDFAs). Both copy tensors and LDFAs are structures used to support several proofs throughout the appendix.

## B.1 Model Formalization

## B.1.1 Decision Trees

We define a decision tree (DT) as a directed acyclic graph that represents a function f : D → [ c ] , where c ∈ N is the number of classes. The tree structure encodes this function as follows: (i) Each internal node v is associated with a unique binary input feature from { 1 , . . . , n } ; (ii) Each internal node has at most k outgoing edges, corresponding to values in [ k ] assigned to the feature at v ; (iii) Along any path α , each feature appears at most once; (iv) Each leaf node is labeled with a class from [ c ] . Given an input x ∈ D , the DT defines a unique path α from the root to a leaf, where f ( x ) is the class label at the leaf. The size of the DT, denoted | f | , is the total number of edges in the graph. To allow modeling flexibility, the order of variables can differ across paths α and α ′ , but no variable repeats within a single path.

## B.1.2 Decision Tree Ensembles

There are several well-established architectures for tree ensembles. Although these models mainly differ in how they are trained, our focus is on post-hoc interpretation, so we concentrate on differences in the inference phase. In particular, we analyze ensemble families that apply weighted-voting schemes during inference - this includes boosted ensembles such as XGBoost. When all weights are equal, our framework also captures majority-voting ensembles, like those used in bagging methods such as Random Forests. Conceptually, an ensemble tree model T is defined as a weighted sum of individual decision trees. Formally, it is specified by the tuple ⟨{ T i } i ∈ [ m ] , { w i } i ∈ [ m ] ⟩ , where { T i } i ∈ [ m ] denotes a collection of decision trees (the ensemble), and { w i } i ∈ [ m ] is a set of real-valued weights. The model T is applied to regression tasks and the function it computes is defined as follows:

<!-- formula-not-decoded -->

## B.1.3 Linear RNNs and Linear Regression Models.

Recurrent Neural Networks (RNNs) are neural models specifically designed to handle sequential data, making them ideal for tasks involving time-dependent or ordered information. They are widely used in applications such as language modeling, speech recognition, and time series prediction. A (second-order) linear RNN is a sub-class of RNNs, that models non-linear multiplicative interactions between the input and the RNN's hidden state. Formally, a second-linear RNN R is parametrized by the tuple &lt; h 0 , T , W, U, b, O &gt; where :

- h 0 ∈ R d : is the initial state vector.
- T ∈ R d × d × d : Second-order interaction tensor capturing the multiplicative interactions between x ( i ) t and h ( j ) t -1 .

- W ∈ R d × N : The input matrix
- U ∈ R d × d : The state transition matrix
- b ∈ R d : The bias vector
- O ∈ R N × n out : The observation matrix

The parameter d is referred to as the size of the RNN, and N is the vocabulary size.

The processing of a vector x = ( x 1 , . . . , x n in ) ∈ [ N ] n in by an RNN is performed sequentially from left to right according to the equation:

<!-- formula-not-decoded -->

Where h t -1 is the hidden state vector of the RNN after reading the prefix x 1: y of the input sequence, and y is the model's output for the input instance x .

Second-order linear RNNs, as defined above, generalize two classical ML models, namely first-order linear RNNs and Linear Regression Models . When the second-order interaction tensor T is set to the zero tensor, then the family of second-order linear RNNs is reduced to the family of first-order linear RNNs. If, in addition, the matrix U is equal to the identity matrix, then a second-order linear RNN is reduced to a linear regression model.

We use three main tensor operations in this work:

Figure 5: Illustrations of (1) A tensor of order 4, (2) A general TN comprising 4 tensors of order 4 and 3 free indices, and (3) A tensor train (TT).

<!-- image -->

## B.2 Elements of Tensor Algebra: Operations over Tensors, Copy Tensors

## B.2.1 Operations over Tensors

In this article, we mainly use three operations over Tensors:

Operation 1: Contraction. Given two Tensors T (1) ∈ R d 1 × ... × d n and T (2) ∈ R d ′ 1 × ... × d ′ m , and two indices ( i, j ) ∈ [ n ] × [ m ] , such that d i = d ′ j The contraction of Tensors T (1) and T (2) over indices i and j produces another tensor, denoted T (1) × ( i,j ) T (2) , over R d 1 × ...d i -1 × d i +1 × ... × d n × d ′ 1 × ... × d ′ j -1 × d ′ j +1 × ... × d ′ m such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We slightly abuse notation and generalize single-leg contraction to multi-leg contraction . For a set S ⊆ [ n ] × [ m ] , the multi-leg contraction T (1) × S T (2) applies leg contraction to all index pairs in S . This operation is commutative - the result doesn't depend on contraction order.

Operation 2: Outer Product. Given two Tensors T (1) ∈ R d 1 × ... × d n and T (2) ∈ R d ′ 1 × ... × d ′ m . The Tensor Outer product of T (1) and T (2) produces a Tensor, denoted T (1) ◦ T (2) , in R d 1 × ... × d n × d ′ 1 × ... × d ′ m whose parameters are given as follows:

<!-- formula-not-decoded -->

Operation 3: Reshape. Let T ∈ R d 1 × d 2 ×···× d n be a tensor of order n , and let [ m 1 , m 2 , . . . , m k ] be a list of positive integers such that m 1 + m 2 + · · · + m k = n . Define: p ( i ) def = 1 + ∑ i -1 l =1 m l , q ( i ) def = ∑ i l =1 m l and D i def = ∏ q ( i ) j = p ( i ) d j . Then, the reshape operation, denoted reshape ( T , [ m 1 , . . . , m k ]) produces a new tensor T ′ ∈ R D 1 × D 2 ×···× D k via a bijection between the multi-index ( i 1 , . . . , i n ) of T and ( J 1 , . . . , J k ) of T ′ . For each group i , the combined index is: J i = 1 + ∑ q ( i ) j = p ( i ) ( ( i j -1) ∏ q ( i ) l = j +1 d l ) , and we set T ′ ( J 1 , . . . , J k ) = T ( i 1 , . . . , i n ) . This mapping is bijective and preserves the number of elements.

## B.2.2 Copy Tensors

In the field of TNs, a copy tensor, also known as a delta tensor or Kronecker tensor [24], is a special type of tensor that enforces equality among its indices. It ensures that all connected indices must take the same value, making it essential for modeling constraints where multiple indices must be identical. This tensor is widely used in operations requiring synchronized index values across different parts of a network. In our context, it will be employed to simulate BNNs using TNs (Section F), as well as in the proof of the #P-Hardness of computing Marginal SHAP for General TNs (subsection D.1). Formally, for n ≥ 1 , the elements of the ordern copy tensor, denoted ∆ ( n ) , defined over the index set [ d ] are given by:

Figure 6: (a) The copy tensor ∆ (4) enforces index-equality among its n legs. (b) Its exact TT representation is a chain of 4 tensor cores, each a black node with one 'physical' leg (down) and two 'bond' legs (horizontal), enforcing i 1 = · · · = i n .

<!-- image -->

In TN diagrams, a copy tensor is traditionally depicted as a black dot with n edges, where each edge represents one of its indices (Figure 6).

Copy Tensors as Tensor Trains (TTs). The explicit construction of the copy tensor ∆ ( n ) scales exponentially with n . Fortunately, as shown in [73], the copy tensor ∆ ( n ) admits an exact length TT decomposition n .

<!-- formula-not-decoded -->

With the following elements:

<!-- formula-not-decoded -->

Where α 0 = α n = 1 , and each α k , i k ∈ { 1 , . . . , d } . Hence, we have the following:

<!-- formula-not-decoded -->

Since each core enforces α k -1 = i k = α k , the only nonzero term in the sum occurs when i 1 = · · · = i n , thereby exactly reproducing the copy tensor.

## B.3 Layered Deterministic Finite Automata (LDFAs).

A Layered Deterministic Finite Automaton (LDFA) is a restricted form of a deterministic finite automaton [56] with a strict layered structure. Let Σ = [ N ] be a finite input alphabet. An LDFA over Σ is defined as a tuple,

<!-- formula-not-decoded -->

- L ≥ 1 is the number of layers;
- For each l ∈ [ L ] , S l is a finite set of states at layer l , and the total set of states is Q = ⊎ L l =1 S l ;
- δ is a partial transition function:

<!-- formula-not-decoded -->

such that for each ( s, σ ) , δ ( s, σ ) is uniquely defined if a transition exists, and otherwise undefined;

- s 0 1 ∈ S 1 is the unique initial state, located in the first layer;
- F L ⊆ S L is the set of accepting states, all located in the final layer.

The LDFA processes sequences of length L . A sequence w = σ 1 σ 2 · · · σ L is accepted if there exists a sequence of states s 0 1 , s 1 2 , . . . , s L L +1 such that

<!-- formula-not-decoded -->

and s L L +1 ∈ F L . By construction, transitions only go forward one layer at a time, making the automaton acyclic and enforcing a strict pipeline topology.

LDFAs as TTs. Layered Deterministic Finite Automata (LDFAs) admit a natural and efficient representation in the Tensor Train (TT) format. Given an LDFA A = ( L, { S l } L l =1 , Σ , δ, s 0 1 , F L ) over the alphabet Σ = [ N ] , we construct a sequence of tensor cores {G ( l ) } L l =1 , where each core G ( l ) is a sparse binary 3-dimensional tensor encoding the transition map δ from layer l to l +1 . Each core G ( l ) has dimensions corresponding to | S l | × [ N ] ×| S l +1 | , where an entry G ( l ) ( s, σ, s ′ ) = 1 if and only if δ ( s, σ ) = s ′ (i.e. the transition from the state s to the state s ′ is valid after reading the inpit σ .

Using the TT representation of LDFAs, the acceptance of the input x = ( x 1 , . . . , x L ) can be alternatively computed as:

<!-- formula-not-decoded -->

where S def = { ( i, i ) : i ∈ [ L ] } .

It's worth noting that the conversion from an LDFA to its TT representation can be performed in polynomial time with respect to the number of layers L and the total number of states ∑ L l =1 | S l | , since each transition is represented by a single non-zero entry in a sparse tensor core. This TT encoding enables the efficient representation and manipulation of the automaton using TN operations.

Product Operation over LDFAs. Given two LDFAs A (1) = ( L, { S 1 l } L l =1 , Σ , δ 1 , s 0 , 1 1 , F 1 L ) and A (2) = ( L, { S 2 l } L l =1 , Σ , δ 2 , s 0 , 2 1 , F 2 L ) with identical input alphabet and number of input layers, the product of A (1) and A (2) , denoted A (1) ×A (2) , is a LDFA with the following parameterization:

<!-- formula-not-decoded -->

where for any l ∈ [ L ] and any pair of states ( s 1 l , s 2 l ) ∈ S 1 l × S 2 l , and σ ∈ Σ , we have:

<!-- formula-not-decoded -->

The function computed by the LDFA A (1) × A (2) is equal to f A (1) ×A (2) = f A (1) · f A (2) [82]. Concretely, the product operation on LDFAs computes the intersection of two LDFAs. Equivalently, where:

when LDFAs encode clauses, this operation corresponds to the conjunction of those clauses. Note that the product operation over LDFAs is commutative.

In the context of our work, we are interested in efficient encodings of the product of multiple LDFAs into a single TN. Fix an integer k ≥ 1 and let {A ( k ) } k ∈ [ K ] be K LDFAs sharing the same input alphabet and the same number of layers. We aim at constructing a TN that computes the function: f A (1) × . . . f A ( K ) . If we restrict our TN representation to be in TT format, a naive construction of an equivalent TT consists of applying the Kronecker product over transition maps of all these LDFAs [82]. The state space of the resulting LDFA from this operation scales exponentially with respect to K . However, for general TNs, one can prove that the product over LDFAs admits an efficient encoding in polynomial time:

Lemma 1. There exists a polynomial algorithm that takes as input a set of K LDFAs { A ( k ) } k ∈ [ K ] sharing the same input alphabet Σ , and the same number of layers L and outputs a TN equivalent to A (1) × . . . × A ( K ) . The complexity is measured in terms of K , the alphabet size, and the number of states of the input LDFAs.

The key building block for proving Lemma 1 is the copy tensor introduced in the previous section. Specifically, for each layer l ∈ [ L ] , the tensor obtained by contracting the copy tensor ∆ ( K ) with the l -th tensor cores of the TT representations of the LDFAs matches the transition map of the resulting LDFA produced by the product operation. We show this fact in the following proposition. To ease exposition, we restrict our result for the case K = 2 . Thanks to the commutative property of the product operation over LDFAs, the general case follows by induction.

Proposition 1. Let A (1) , and A (2) be two LDFAs (sharing the same input alphabet and the same number of layers) parametrized in TT format as J G (1 , 1) , . . . , G (1 ,L ) K and J G (2 , 1) , . . . , G (2 ,L ) K , respectively. Then, the TT is parametrized as:

<!-- formula-not-decoded -->

is equivalent to A (1) ×A (2) .

Proof. Let A (1) = ( L, { S l } L l =1 , Σ , δ 1 , s 0 , 1 1 , F 1 L ) and A (2) = ( L, { S ′ l } L l =1 , Σ , δ 2 , s ′ 0 , 1 1 , F ′ 1 L ) be two LDFAs sharing the same input alphabet Σ and the same number of layers L . We show that at each layer l ∈ [ L -1] , the TT whose parameterization is given in equation (11) simulates exactly the dynamics governed by the transition map of the product LDFA A (1) ×A (2) at layer l , namely δ 1 l × δ 2 l .

More formally, denote J G (1 , 1) , . . . , G (1 ,L ) and J G (2 , 1) , . . . , G (2 ,L ) K the TT parametrizations of A (1) and A (2) , respectively. We need to show that, for any ( i l , i l +1 , σ, j l , j l +1 ) ∈ [ S l ] × [ S l +1 ] × Σ × [ S ′ l ] × [ S ′ l +1 ]) , we have:

<!-- formula-not-decoded -->

where δ 1 , δ 2 , δ 1 × δ 2 refer to the transition map of A (1) , A (2) and A (1) ×A (2) , respectively. We have:

<!-- formula-not-decoded -->

Now, we are ready to prove lemma 1:

Proof. (Lemma 1) Proposition 1 shows that one can construct a TT equivalent to the product of K LDFAs {A ( k ) } k ∈ [ K ] whose tensor core at layer l are parametrized as:

<!-- formula-not-decoded -->

Figure 7: The structure of a TN equivalent to the product operation of 3 LDFAs (whose TT representations are depicted in green). The number of layers is equal to L = 4 . The tensors in the bottom (in black) correspond to the TT representation of the copy tensor.

<!-- image -->

Anaive construction of the tensor ∆ ( K ) scales exponentially with respect to K . However, constructing it using TT format (as discussed in the previous section) can be done in O ( poly ( | Σ | , K )) running time.

Using the TT representation of copy tensors, the resulting TN that computes the product LDFA A (1) × . . . ×A ( K ) has a 2-dimensional grid structure whose width is equal to the number of layers L and whose height is equal to K . This TN structure can be constructed in polynomial time with respect to the size of the LDFAs, K , | Σ | , and L . Figure 7 provides a graphical illustration of this construction.

## C Computing Marginal SHAP for General TNs

This section presents the proof of correctness of the framework introduced in section 4 to compute exactly Marginal SHAP scores of TNs with arbitrary structures. Specifically, we prove Proposition 1, Lemma 1, and Lemma 2. This section is organized into three parts, each devoted to one of these results.

## C.1 Proof of Proposition 1

Recall the statement of Proposition 1:

Proposition. Define the modified Weighted Coalitional tensor ˜ W ∈ R n in × 2 ⊗ n in such that ∀ ( i, s 1 , . . . , s n in ) ∈ [ n in ] × [2] ⊗ n in it holds that ˜ W i,s 1 ,...,s n in def = -W ( s -1 n in ) if s i = 1 , and ˜ W i,s 1 ,...,s n in def = W ( s -1 n in ) otherwise. Moreover, define the marginal value tensor V ( M,P ) ∈ R D × 2 ⊗ n in × n out , such that ∀ ( x, s ) ∈ D × [2] ⊗ n it holds that V ( M,P ) x,s, : def = V M ( x, s ; P ) . Then we have that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We first define the swap (.,.) operation. For n ∈ N , a binary vector s ∈ { 0 , 1 } n and i ∈ [ n ] , swap ( s, i ) returns a binary vector of dimension n equal to s everywhere except for the i -th element where it is equal to 1 .

We rewrite the vector ϕ i ( M,x,P ) (Equation (1)) as follows:

<!-- formula-not-decoded -->

where: ˜ W ( s, i ) def = { W ( s ) if s i = 1 (Feature i to explain is part of the coalition) -W ( s ) otherwise (Feature i to explain is not part of the coalition) .

Acontraction of the modified Weighted Coalitional Tensor ˜ W with the Marginal Value Tensor V ( M,P ) across the dimensions defined by the set S (defined in the proposition statement) straightforwardly yields Equation (12).

## C.2 The construction of the modified Weighted Coalitional Tensor ˜ W (Lemma 1)

Recall the statement of Lemma 1:

Lemma. The modified Weighted Coalitional Tensor ˜ W admits a TT representation: J G (1) , . . . , G ( n in ) K , where G (1) ∈ R n in × 2 × n 2 in , G ( i ) ∈ R n 2 i × 2 × n 2 i for any i ∈ [2 , n in -1] , and G ( n in ) ∈ R n 2 in × 2 . Moreover, this TT representation is constructible in O (log( n in )) time using O ( n 3 in ) parallel processors.

The definition of the Marginal SHAP Coalitional Tensor is:

<!-- formula-not-decoded -->

Remark. Note the existence of a coordinate shift from the boolean representation that lies in { 0 , 1 } to TN representation which lies in [2] , hence the subtraction by the vector 1 n in . To distinguish between both representations, we use the notation ˜ s in the sequel to refer to the TN representation of boolean values.

In the remainder of this section, we shall provide the exact construction of the tensor cores {G l } l ∈ [ n in ] implicitly mentioned in the lemma statement, followed by a proof of its correctness and its associated complexity. Before that, we introduce some notation.

Notation. For a vector ˜ s ∈ [2] n , we denote | ˜ s | the number of its elements equal to 1 , and ˜ s 1: k ∈ R k the vector composed of its first k elements. We define the function ∆ that takes as input a vector ˜ s ∈ [2] n and an integer i ∈ N , and outputs 1 if [(˜ s i = 1) ∨ ( n ≤ i -1)] , -1 otherwise. The function ∆( ., . ) can be alternatively defined using the recursive formula:

<!-- formula-not-decoded -->

where ∆( ϵ, i ) = 1 ( ϵ refers to the trivial vector of dimension 0 )

High-Level Steps of the Construction. The general idea of the construction is to build a TT that simulates the computation of the modified Weighted Coalitional Tensor by processing its input ( i, ˜ s 1 , ˜ s 2 , . . . , ˜ s n in ) from left to right in a similar fashion to state machines. The introduction of each new element ˜ s j in the sequence updates the state of the computation by adjusting the weight corresponding to the size of the coalition of features seen so far and retaining the relevant information required for the processing of subsequent elements.

Intuitively, two pieces of information need to be kept throughout the computation performed by the state machine:

- The feature to explain i : This information is provided at the beginning of the computation by the first leg of the tensor ˜ W and will be stored throughout the computation, using a simple copy operation. This information is needed to flip the sign when the position i in the sequence is reached (Equation 14).
- The size of the coalition k . Elements of the modified Weighted Coalitional Tensor depend on the size of the coalition. This information needs to be updated throughout the processing of the input sequence. Assume at step i ∈ [ n in ] , the size of the coalition formed from the first i features is equal to k :
- -If ˜ s i = 1 (feature i is not part of the coalition), we transition to the state k to the next step, and we normalize the weight of the coalition accordingly,
- -If ˜ s i = 2 (feature i is part of the coalition), we transition to the state k + 1 , and normalize the weight of the coalition accordingly.

Based on this description, we propose a TT representation of ˜ W that maintains an internal state encoded as a matrix G ( j ) ∈ R n in × n in . The semantics of the element G ( j ) i,k in the matrix reflect the description described above.

Construction of the core tensors {G ( i ) } i ∈ [ n in ] . For simplicity, we assume that the core tensors are of order 5 , so that the internal state encoding takes a matrix format and holds the semantics outlined in the above description. The transition to core tensors of order 3 as in the TT format can be performed through a suitable reshape operation. The parameterization of the core tensors {G ( i ) } i ∈ [ n in ] is given as follows:

- The core tensor G (1) :
1. If ˜ s 1 = 1 (feature 1 is not part of the coalition):

<!-- formula-not-decoded -->

2. If ˜ s j = 2 (feature 1 is part of the coalition):

<!-- formula-not-decoded -->

- The core tensors {G ( j ) } j ∈ [ n in ] \{ 1 } :
1. If ˜ s j = 1 (feature j is not part of the coalition):

<!-- formula-not-decoded -->

2. If ˜ s j = 2 (feature j is part of the coalition):

<!-- formula-not-decoded -->

The rightmost core tensor of the TT is obtained by contracting G ( n in ) with the ones matrix 1 n in × n in (The matrix whose all elements are equal to 1 ) at the pair of leg indices (4 , 1) and (5 , 2) .

Correctness. The proposed TT construction is designed in such a way that it captures the dynamics outlined in the description above. The state machine maintains a sparse matrix equal to zero everywhere except for the element ( i, k ) where i corresponds to the feature to explain, and k is the number of already processed features in the coalition. The following proposition formalizes this fact:

Proposition 2. Fix an integer n in ≥ 1 . For any j ∈ [ n in ] , i ∈ [ n in ] , ( l, k ) ∈ [ n in ] 2 and s = ( s 1 , · · · , s j ) ∈ [2] j , we have:

<!-- formula-not-decoded -->

where:

<!-- formula-not-decoded -->

Proof. Fix n in ≥ 1 . The proof proceeds by induction on j .

Base Case. Assume j = 1 . Let i ∈ [ n in ] , ( l, k ) ∈ [ n in ] 2 and ˜ s 1 ∈ [2] . Equation (17) holds by construction of G (1) .

General Case. Assume Equation (17) holds for j ∈ [ n in -1] , we need to show that it's also valid for j +1 .

Let ( l, k ) ∈ [ n in ] 2 and (˜ s 1 , . . . , ˜ s j +1 ) ∈ [2] j +1 . For better readability, we adopt the following notation in the sequel:

<!-- formula-not-decoded -->

By the induction assumption, we have:

<!-- formula-not-decoded -->

where the second equality is obtained by the induction assumption.

We first handle the case when H ( j +1) ˜ s 1 ,..., ˜ s j +1 ,l,k = 0 .

By construction of G ( j +1) , we have:

̸

<!-- formula-not-decoded -->

̸

Next, we analyze the case: l = i ∧ | s 1: j +1 | = k . We split into two cases:

Case s j +1 = 1 (The feature j +1 is not part of the coalition): In this case, note that: | s 1: j +1 | = | s 1: j | . We have:

<!-- formula-not-decoded -->

Case s j +1 = 2 (The feature j +1 is not part of the coalition). In this case, note that: | s 1: j +1 | = | s 1: j | +1 , and ∆( s 1: j +1 , i ) = ∆( s 1: j , i ) .

We have:

<!-- formula-not-decoded -->

Complexity. The core tensors {G ( j ) } j ∈ [ n in ] are extremely sparse. By leveraging this sparsity property, each core tensor can be constructed in O (1) time using O ( n 2 in ) parallel processors. By parallelizing the construction of all core tensors, this leads to a number of O ( n 3 in ) parallel processors. Yet, it should be observed that the construction of the leftmost tensor G (1) requires the computation of the factorial terms ( n in -1)! and ( n in -2)! . This operation can be performed using a parallel scan strategy using O ( n in ) parallel processors with O (log( n in )) running time.

To summarize, the total computation of the TT corresponding to the modified Weighted Coalitional Tensor requires O ( n 3 in ) parallel processors and runs in O (log( n in )) time.

## C.3 The construction of the Marginal SHAP Value Tensor V ( M,P ) (Lemma 2)

The objective of this section is to prove the Lemma 2. Recall the statement of this lemma:

Lemma. Let T M and T P be two TNs implementing the model M and the probability distribution P , respectively. Then, the marginal value tensor V ( M,P ) can be computed as:

<!-- formula-not-decoded -->

- S 1 and S 2 are instantiated such that for all k ∈ { 1 , 2 } it holds that: S k def = { (5 -k ) · i, i ) : i ∈ [ n in ] } ,
- For any i ∈ [ n in ] , ( σ, s, σ ′ , σ ′′ ) ∈ [ N i ] × [2] × [ N i ] ⊗ 2 , we have:

<!-- formula-not-decoded -->

Before providing the proof of Lemma 2, we first introduce an operator which will be useful for the proof:

The do operator [83]. Let S be a finite set. The do operator takes as input a triplet ( σ, s, σ ′ ) ∈ S × [2] × S and returns σ if s = 1 , σ ′ otherwise. The collection of Tensors M ( i ) parametrized as in Equation (18) can be seen as tensorized representations of the do operator [83]. When S = [ N i ] , we have, for any ( σ, s, σ ′ , σ ′′ ) ∈ [ N i ] × [2] × [ N i ] × [ N i ] , M ( i ) σ,σ ′ ,σ ′′ = 1 if and only if σ ′′ = do ( σ, s, σ ′ ) . By induction, for any ( x, s, x ′ , x ′′ ) ∈ D × [2] × D ⊗ 2 , we have:

<!-- formula-not-decoded -->

Now, we are ready to provide the proof of Lemma 2:

Proof. (Lemma 2) Let ( x, s, y ) ∈ D × [2] ⊗ n in × [ n out ] , we have:

<!-- formula-not-decoded -->

where:

Figure 8: An LDFA that accepts the set of satisfying assignments of the disjunctive clause: C = x 1 ∨ ¬ x 3 ∨ x 4 . The lowest states in the grid ( { s 2 1 , s 3 1 , s 4 1 , s 5 1 } tracks the satisfiability of the running assignment. Satisfying assignments are those that reach the state s 5 1

<!-- image -->

## D Computing Marginal SHAP for Tensor Trains

This section contains the proof of the #P-Hardness of computing SHAP for General TNs (Proposition 2), as well as a proof of the fact that SHAP for TTs is in NC.

## D.1 SHAP for general TNs is #P-Hard

Proposition 2 states the following:

Proposition. Computing Marginal SHAP scores for general TNs is # P-HARD .

Proof. The proof builds on known connections between model counting and the computation of SHAP values. Recall that, given a class of classifiers C , model counting refers to the problem of determining the number of inputs classified as positive by a classifier in C . The following classical lemma established by both [107] and [6]:

Lemma 2. Given a class of classifiers C , the model counting problem for C is polynomial-time reducible to the problem of computing Marginal SHAP values for the class C under the uniform distribution.

We note that although the original lemma addresses a different value function, specifically the Conditional SHAP variant based on conditional expectations - the Marginal SHAP and Conditional SHAP formulations coincide under the feature independence assumption [104]. Consequently, the hardness result established in that setting also applies to the general Marginal SHAP case.

Now, to prove Proposition 2 using Lemma 2, we use the class of CNF formulas as a proxy. It is a classical result in complexity theory that the model counting problem for CNFs is #P-Complete [7]. Our goal is to reduce this problem to the model counting problem for TNs:

Lemma 3. There exists a polynomial-time algorithm that takes as input an arbitrary CNF formula Φ and produces a TN that computes an equivalent Boolean function.

Proof. Let Φ be a CNF formula composed of L clauses. The reduction proceeds as follows: For each clause, construct an equivalent Layered Deterministic Finite Automaton (LDFA). This transformation can be performed in linear time with respect to the number of variables in the clause (see Figure 8 for an illustrative example). Then, construct the TN that corresponds to the product of the resulting LDFAs. This TN simulates the conjunction over all clauses of the formula Φ . The polynomial time complexity of this procedure is guaranteed by Lemma 1.

The proof of Lemma 3, combined with our earlier claims, completes the proof of Proposition 2:

Proof. ( Proposition 2) By Lemma 2, the model counting problem for CNFs is polynomially reducible to computing Marginal SHAP values under the uniform distribution. Lemma 3 ensures that any CNF formula is reducible in polynomial time into an equivalent TN. Moreover, the uniform input distribution can be encoded as a product of rank-one tensors in linear time with respect to the number of input variables; for instance, by the tensor 1 2 ( e 1 1 + e 1 2 ) ◦ . . . ◦ 1 2 ( e n 1 + e n 2 ) . This completes the reduction and proves the #P-Hardness of computing Marginal SHAP scores for general TNs.

## D.2 Marginal SHAP for TTs is in NC 2 .

In this section, we provide the details of the algorithmic construction to compute the exact marginal SHAP value for TTs in poly-logarithmic time using a polynomial number of parallel processors.

The algorithmic construction we propose stems its correctness from Theorem 1 which states the following:

Theorem. Let T M = J I (1) , . . . , I ( n in ) K and T P = J P (1) , . . . , P ( n in ) K be two TTs corresponding to the model to interpret and the data-generating distribution, respectively. Then, the Marginal SHAP Tensor T ( M,P ) can be represented by a TT parametrized as:

<!-- formula-not-decoded -->

where the collection of Tensors {G ( i ) } i ∈ [ n in ] and {M ( i ) } i ∈ [ n in ] are implicitly defined in Lemma 1 and Lemma 2, respectively.

Theorem 1 is a corollary of Proposition 1: Replace ˜ W with its TT parametrization (Lemma 1), and replace T M and T P with their corresponding TT parametrizations in the formulation of the Marginal Value Tensor in Lemma 2.

Next, our focus shall be placed on the computational aspect of computing Marginal SHAP scores for TTs by leveraging the result of Theorem 1 to show that this problem is in NC.

Denote:

- T M = q I (1) , . . . , I ( n in ) y a TT model such that, for each i ∈ [ n in ] , I ( i ) is in R d M,i × N i × d M,i +1 ( d M, 1 = 1 and d M,n in +1 = n out ),
- T P = q P (1) , . . . , P ( n in ) y a TT implementing a probability distribution over D such that, for each i ∈ [ n in ] , P ( i ) is in R d P,i × N i × d P,i +1 ( d P, 1 = d P,n in +1 = 1 ),
- An input instance x = ( x 1 , . . . , x n in ) ∈ D ,

In light of Theorem 1, a typical parallel scan procedure to compute the matrix Φ( T M , x, T P ) runs as follows:

- Level 0. Compute in parallel the following n in tensors. For i ∈ [ n in ] :

<!-- formula-not-decoded -->

- Level 1 to log( n in ) . At step j ∈ [ ⌊ log( n in ) ⌋ ] , perform a contraction operation over Neighboring Tensors. For i ∈ [ n in 2 j ] :

<!-- formula-not-decoded -->

where S def = { (7 -k, k ) : k ∈ [3] } .

By Theorem 1, the produced matrix at the last step is equal to Φ( T M , x, T P ) .

Complexity. Each tensor contraction operation in Equations (21) and (22) requires at most O ( n 4 in · max i ∈ [ n in ] d M, · max i ∈ [ n in ] d P,i ) . At level 0 , we need to perform in parallel n in operations of this kind upgrading the number of required parallel processors to O ( n 5 in · max i ∈ [ n in ] d M,i · max i ∈ [ n in ] d P,i ) . As for the running time, the depth of the circuit perform- ing the tensor contractions is bounded by ′ ( log( n in ) + log( max i ∈ [ n in ] d M,i ) + log( max i ∈ [ n in ] d P,i ) ) . The depth of the total circuit to compute Marginal SHAP for TTs is thus bounded by O ( log( n in ) · [log( n in ) + log( max i ∈ [ n in ] d M,i ) + log( max i ∈ [ n in ] d P,i )] ) .

## E Tightening Complexity Results for Other ML Models

In this section, we employ NC reductions to show that the problem of computing Marginal SHAP scores for certain popular model classes (Linear RNNs, decision trees, ensemble trees, and linear models) lies also in NC 2 (Theorem 2).

An NC reduction is a type of many-one reduction where the transformation function from instances of problem A to instances of problem B is computable by a uniform family of Boolean circuits with polynomial size and polylogarithmic depth. This ensures that the reduction itself can be performed efficiently in parallel. A crucial property of the class NC is its closure under NC reductions: if a problem A is NC-reducible to a problem B , and B is in NC, then A is also in NC [7].

## E.1 Reduction from Linear RNNs and Linear Models to TTs.

Weassume without loss of generality that N = N 1 = N 2 = . . . = N n in . This assumption reflects the practical use of RNNs, such as in Natural Language Processing (NLP) applications, where elements of the sequence are assumed to belong to a finite vocabulary.

Linear RNNs and TTs. General TTs can be seen as non-stationary generalizations of Linear RNNs at a fixed window. Interestingly, Rabusseau et al. [95] showed that stationary TTs are strictly equivalent to second-order linear RNNs (see section B for a formal definition of second-order linear RNNs). Indeed, reformulating the equations governing the dynamics of a second-order linear RNNs in a tensorized format, it can be shown using careful algebraic calculation that:

<!-- formula-not-decoded -->

Where it holds that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

All the other elements are set to 0 .

The additional dummy neuron at dimension d +1 (whose value is always equal to 1 ) is added to the model to account for additive terms present in the linear RNN dynamics equation.

Consequently, a second-order linear RNN R at a bounded window n in is equivalent to the stationary TT T R parametrized as:

<!-- formula-not-decoded -->

Reduction and Complexity. The SHAP computational problem for Linear RNNs under TT distributions takes as input an RNN R of size d mapping sequences of elements in [ N ] to R n out , an input sequence x = ( x 1 , . . . , x n in ) ∈ [ N ] ⊗ n in and T P and outputs the SHAP Matrix Φ( R,x, T P ) ∈ R n in × n out (Equation ).

<!-- formula-not-decoded -->

Fix an input instance ( R,x, T P ) . Our objective is to construct in poly-logarithmic time using a polynomial number of parallel processors an input instance of the SHAP problem for TTs: ( T R , ¯ x, T ¯ P ) such that Φ( R,x, T P ) = Φ( T R , ¯ x, T ¯ P ) . The detailed reduction strategy of the SHAP problem of Linear RNNs to the SHAP problem of TTs is detailed as follows:

- Reduction of x and T P : The input instance x and the tensor T P are mapped trivially to the instance ¯ x and the tensor T ¯ P of the input instance of the SHAP problem for TTs. This operation runs in constant time using a linear number of parallel processors with respect to n in and the size of T P
- Reduction of the linear RNN R : Equation (24) hints at a straightforward strategy to reduce linear RNNs into equivalent TTs at a bounded window:
- -Apply the tensor contraction operation: H = ˜ T × (2 , 1) I . This operation runs in O (log( N )) using O ( d 2 · N ) parallel processors
- -Compute Leftmost and Rightmost matrices:
* Leftmost Matrix H (1) : H (1) = H× (1 , 1) [ h 0 1 ] .

This operation runs in O (log( d )) using O ( d 2 · N ) parallel processors.

* Rightmost Matrix: H ( n in ) = H× (3 , 1) [ O 0 N +1 ] .

This operation runs in O (log( d )) using O ( d 2 · N ) parallel processors.

- -Run in parallel O ( n in · N · d 2 ) parallel workers to place in the input tape of the SHAP problem for TTs the following TT:

<!-- formula-not-decoded -->

The total reduction strategy runs in O (max(log( N ) , log( d ))) using O ( n in · N · d 2 + |T P | ) parallel processors.

## E.2 Reduction from Tree Ensembles to TTs.

The objective of this subsection is to show that computing SHAP for Tree Ensembles is NC-reducible to the SHAP problem for TTs.

We first note the fact that if computing SHAP for DTs is efficiently parallelizable, then so is Ensemble Trees. Indeed, by the linearity property of the SHAP score, computing SHAP for Ensemble Trees is obtained as a weighted sum of SHAP scores of the trees forming the ensemble. This operation adds O (log( N )) depth to the circuit that computes SHAP scores of single DTs, where N refers to the number of trees in the ensemble. This implies the following fact:

Fact. Fix a distribution class P . Computing Marginal SHAP for the class of Decision Trees under P is in NC implies computing Marginal SHAP for Ensemble Trees is also in NC .

In light of this fact, we dedicate the rest of this section to examining the specific case of DTs and their reduction into TTs.

Decision Trees as Disjoint DNFs. We first propose a representation of Tree-based models. This representation has been shown to be more amenable to parallelization and forms the core of the preprocessing step as the GPUTreeSHAP algorithm [85]. Define the following collection of predicates:

<!-- formula-not-decoded -->

where i ∈ [ n in ] , and j ∈ [ N i ] . A DT T can be equivalently represented as a disjoint DNF formula over the predicates { p ij } i ∈ [ n in ] ,j ∈ [ N i ] as [82]:

<!-- formula-not-decoded -->

where C L is a conjunctive clause.

Clauses { C j } j ∈ L satisfy the disjointness property: For two different clauses, the intersection of their satisfying assignments is empty. The conversion of Decision Trees into equivalent Disjoint DNFs has been shown to run in polynomial time with respect to the number of edges of the DT.

In the sequel, we assume that DTs are represented using the Disjoint DNF formalism. In practice, the conversion of a DT into this format can be performed offline (in polynomial time), and stored as such, in the same fashion of the GPUTreeSHAP Algorithm [85].

We begin by showing how DTs can be encoded in TT format. We assume an arbitrary ordering X 1 , . . . , X n in of input features of the DT:

Proposition 3. A DT T encoded as a disjoint DNF Φ T = C 1 ∨ . . . ∨ C L is equivalent to a TT parametrized as:

<!-- formula-not-decoded -->

such that for l ∈ [ n in ] , the tensor I ( l ) ∈ R L × N l × L is such that:

<!-- formula-not-decoded -->

Proof. Let T be a a DT, and Φ T = C 1 ∨ . . . ∨ C L be its equivalent representation into a disjoint DNF format. For k ∈ [ n in ] , we shall use the notation C k l to denote the restriction of the clause l to include only predicates in { p ij } i ≤ i .

Weclaim that for any k ∈ [ n in ] , and any x = ( x 1 , . . . , x N n in ) ∈ D , the TT (of length k ) parametrized as:

is such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If this claim holds, then for k = n , and by the disjointness property of Φ T , we have for any x = ( x 1 , . . . , x n in ) :

<!-- formula-not-decoded -->

which completes the proof of the proposition.

We prove this claim by induction on k .

<!-- formula-not-decoded -->

(1)

General Case. Assume the claim holds for k , we show that it also holds for k +1 .

Let ( x 1 , . . . x k +1 ) ∈ [ N 1 ] × . . . × [ N k +1 ] and l ∈ [ L ] . We have:

<!-- formula-not-decoded -->

Reduction and Complexity. Proposition 3 suggests the following strategy to reduce a DT encoded as a Disjoint DNF into an equivalent TT.

Fix a Disjoint DNF Φ T = C 1 ∨ . . . ∨ C L . The granularity of the parallelization strategy is at the literal level: Each parallel processor is dedicated to processing a specific literal p ij in a clause C l : If a literal p ij appears in the clause C l , the processor sets the value I ( i ) l,j,l to 1 . The correctness of this parallel schema is guaranteed by Proposition 3. The running time complexity of this parallel procedure is O (1) . The number of parallel processors is O ( L · n in ) .

## F Computing Marginal SHAP for BNNs

In this section, we present the proof of Theorem 3 which provides fine-grained parameterized complexity for the problem of computing SHAP for BNNs. Recall this theorem's statement:

Theorem. Let P be either the class of empirical distributions, independent distributions, or the class of TTs. We have that:

1. Bounded Depth: The problem of computing SHAP for BNNs under any distribution class P is PARA-NP-HARD with respect to the network's depth parameter.
2. Bounded Width: The problem of computing SHAP for BNNs under any distribution class P is in XP with respect to the width parameter.
3. Bounded Width and Sparsity. The problem of computing SHAP for BNNs under any distribution class P is in FPT with respect to the width and reified cardinality parameters.

We split this section into two parts: The first part is dedicated to proving that computing SHAP for BNNs with bounded depth remains intractable. The second part provides the details of a procedure that builds an equivalent TT to a given BNN. The complexity analysis of this procedure will yield the results of items (2) and (3) of the Theorem.

## F.1 SHAP for BNNs with constant depth is intractable

We will demonstrate that computing SHAP values for BNNs remains intractable even when the network is restricted to a depth of just one.

Proposition 4. Given a BNN B with one hidden layer, and some input x , then it holds that computing SHAP for B and x is #P-Hard.

Proof. We prove that computing SHAP values for a Binarized Neural Network (BNN) with just a single hidden layer is already #P-Hard. Specifically, this is shown via a reduction from the counting variant of the classic 3SAT problem - namely, #3SAT - which is known to be #P-Hard [7].

We begin by referencing the result of [6], which showed that computing SHAP values for a model f and input x , even under the simple assumption that the distribution D p is uniform , is as hard as model counting for f . Consequently, if model counting for f is #P-Hard, so is SHAP computation-this follows directly from the efficiency axiom under uniform distributions. Therefore, to establish #P-Hardness of SHAP for a model f , it suffices to reduce from a model with known #P-Hard model counting (e.g., a 3CNF formula). Since uniform distributions are a special case of independent distributions and significantly simpler than structured models like tensor trains (TTs), the hardness results extend naturally to our setting as well.

Consider a 3-CNF formula ϕ := t 1 ∧ t 2 ∧ . . . ∧ t m , where each clause t i is a disjunction of three literals: t i := x j ∨ x ℓ ∨ x k . We construct a Binarized Neural Network (BNN) B with a single hidden layer over the input space {-1 , 1 } n -asetting that already implies hardness for more general discrete domains. Each input neuron of the BNN corresponds to a variable in ϕ , so the input layer has n neurons. The hidden layer contains m neurons, one for each clause in ϕ . For a clause t i , if a variable x j appears positively, we connect input neuron j to hidden neuron i with weight +1 ; if it appears negatively, the weight is -1 . Each hidden neuron is assigned a bias of 3 2 . Finally, we add a single output neuron with a bias of -( m -1 2 ) .

Wenowshow that every satisfying assignment to ϕ yields a 'True' output in B , and every unsatisfying assignment yields 'False'. A clause t i is satisfied if at least one of its variables evaluates to True. This occurs either when a non-negated variable is assigned 1 (a 'double-positive' case) or when a negated variable is assigned -1 (a 'double-negative' case). In the BNN, both cases contribute 1 to the corresponding hidden neuron, since input and weight signs align. Thus, if any variable in t i satisfies the clause, the corresponding neuron receives at least one input of 1 . With a bias of 3 2 , even if the remaining two inputs contribute -1 each, the total input is 5 2 -2 = 1 2 &gt; 0 , so the step function outputs 1 . Therefore, satisfied clauses activate their corresponding neurons. Finally, since the output neuron has a bias of -( m -1 2 ) , it outputs 1 if and only if all m hidden neurons are active, meaning all clauses are satisfied.

Since we have shown that each satisfying assignment of the BNN B corresponds to a satisfying assignment of ϕ , it follows that the model counting problem is the same for both. Therefore, for this BNN with only a single hidden layer, and by the result of [6], computing SHAP is #P-Hard.

## F.2 Compiling BNNs into Equivalent TTs

In this section, we provide the details of the procedure of compiling BNNs into Equivalent TTs, to show that the problem of computing SHAP for BNNs in bounded width (resp. bounded width and sparsity) is in XP (resp. FPT).

## F.2.1 Warm Up: Compilation of a single-neuron BNN with Depth 1 into TT.

As an initial step, we present in this subsection a construction for representing the activation function of a single neuron in a BNN using TTs. This construction serves as a basic building block for simulating multi-layered BNN architectures by means of the TT formalism.

Let w = ( w 1 , . . . w n ) ∈ {-1 , 1 } n be the weight parameters of a BNN with a single output neuron n out . Its activation as computed using the reified cardinality representation is given as:

<!-- formula-not-decoded -->

where: R is its reified cardinality parameter, and l i def = { x i if w i = 1 ¬ x i otherwise .

To understand how the construction of an equivalent TT to simulate this BNN operates, we will employ the LDFA formalism previously introduced in subsection B.3.

The main idea of this construction is to build an LDFA whose states function as counters, tracking the number of satisfied literals { l i } i ∈ [ n ] (Equation (26). When this count exceeds the specified cardinality threshold R , the output neuron n out is activated; otherwise, it remains inactive. Crucially, the number of states at each layer in LDFA scales linearly with the parameter R .

Formally, the LDFA that simulates the neuron n out (Equation (26)) is given as follows:

- Number of layers: n ,
- State space: For each layer l ∈ [ n ] , S l = { 0 , R }
- The initial state: s 1 0 (The state at Layer 1 corresponding to R = 0 )
- Transition function: For a layer l ∈ [ n -1] ,

̸

<!-- formula-not-decoded -->

- The final state: s ( n ) R

## F.2.2 Compilation of a single Layer into TT.

In this section, we will move from BNNs with a single neuron into multi-layered BNNs. We will show how one can construct a TT equivalent to a given BNN in O ( R W · poly ( D, max i ∈ [ n in ] N i , n in ) , where R is the reified cardinality parameter of the network, W its width, and D its depth.

The construction will proceed in two steps:

1. From Multi-Layered BNN into a BNN with one hidden layer. In this step, we convert a deep BNN into an equivalent BNN with one hidden layer,
2. From BNN with one hidden layer into an equivalent TT. The resulting BNN with one hidden layer from the first step is then converted into an equivalent TT.

Next, we will provide details of how to perform each of these steps.

Multi-Layered BNNs into BNNs with one hidden layer. There are many ways to convert a multi-layered BNN into an equivalent BNN with one hidden layer. In the main paper, we propose a technique that collapses the network's depth progressively from the last layer up to the first layer. This technique has the advantage of better scaling with respect to the network's depth: Leveraging parallelization, this operation can be performed in O (log( D )) using a similar parallel scan strategy presented in section D. A simpler method consists at settling for building a lookup table that maps the set of possible activation values of neurons in the first layer to the model's output. This lookup table can be built by enumerating all possible activation patterns of the first hidden layer and evaluating the model's output via a network's forward propagation operations. This operation runs in O (2 W 1 · poly ( D )) time.

BNNs with one hidden layer into TTs. Given a BNN with one hidden layer whose width is equal to 1 . The conversion to an equivalent TT can be achieved by remarking that a BNN with one hidden layer can be compiled into an equivalent DNF formula whose number of clauses is O (2 W 1 ) .

Let B be a BNN with one hidden layer over n in variables, and W 1 be the width of the first hidden layer. Denote:

- n = ( n 1 , . . . , n W i ) : the set of its hidden layer neuron activation predicates,
- W ∈ { 0 , 1 } D × 2 ⊗ W 1 : A Binary Tensor that maps the input to the activation pattern of the first hidden layer, i.e.

<!-- formula-not-decoded -->

- T ∈ { 0 , 1 } 2 ⊗ n : A binary tensor that arranges the lookup table (computed in the first step) that maps each activation pattern n to the model's output.

The function computed by the BNN B can be written as:

<!-- formula-not-decoded -->

Equation (28) expresses the function computed by the BNN B as a contraction of two tensors: W and T . The tensor T is nothing but a rearrangement of the lookup table constructed in the first step in a tensorized format. The construction of T requires O (2 W 1 ) running time. On the other hand, even when the network's width is small, a naive construction of the tensor W (Equation (27)) may scale exponentially with the dimension of the input space. What we will show next is how, in bounded width and sparsity regime, this tensor admits a TT representation of size O ( R W 1 · poly ( n in , N )) .

Construction of the tensor W . A first observation about the tensor W is that it encodes a CNF formula whose clauses correspond to activations of neurons in n . On the other hand, in the context of BNNs, clauses are represented using reified cardinality constraints.

The construction procedure of a TT equivalent to the tensor W runs as follows:

1. For each neuron n j , we construct its equivalent LDFA A n j using the procedure outlined as a warm-up at the beginning of this section. The size of the layer's state space of each of these LDFAs is bounded by O ( R ) ( R refers to the network's reified cardinality parameter).

2. Construct the resulting TT corresponding to the product of the LDFAs {A n j } j ∈ [ W 1 ] (subsection B.3), while keeping the rightmost core tensor with legs open, thereby ignore the final states reached by the TT.

Leveraging the Kronecker product operation over Finite State Automata [82] to perform product operations, the TT equivalent to the product of W 1 LDFAs (whose state space is bounded by a constant R ) can be constructed in O ( R W 1 ) time. Consequently, the entire construction produces a TT whose size is bounded by O ( R W 1 · n in · N ) . The running time complexity of the procedure is also bounded by O ( R W 1 · n in · N ) .