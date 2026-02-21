## Revisiting 1 -peer exponential graph for enhancing decentralized learning efficiency

## Kenta Niwa

NTT Communication Science Laboratories &lt;kenta.niwa@ntt.com&gt;

## Guoqiang Zhang

University of Exeter &lt;G.Z.Zhang@exeter.ac.uk&gt;

## Yuki Takezawa

Kyoto University

Okinawa Institute of Science and Technology

&lt;yuki-takezawa@ml.ist.i.kyoto-u.ac.jp&gt;

## W. Bastiaan Kleijn

Victoria University of Wellington &lt;bastiaan.kleijn@vuw.ac.nz&gt;

## Abstract

For communication-efficient decentralized learning, it is essential to employ dynamic graphs designed to improve the expected spectral gap by reducing deviations from global averaging. The 1 -peer exponential graph demonstrates its finite-time convergence property-achieved by maximizing the expected spectral gap-but only when the number of nodes n is a power of two. However, its efficiency across any n and the commutativity of mixing matrices remain unexplored. We delve into the principles underlying the 1 -peer exponential graph to explain its efficiency across any n and leverage them to develop new dynamic graphs. We propose two new dynamic graphs: the k -peer exponential graph and the nullcascade graph. Notably, the null-cascade graph achieves finite-time convergence for any n while ensuring commutativity. Our experiments confirm the effectiveness of these new graphs, particularly the null-cascade graph, in most test settings. https://github.com/garden1984/NullCascadeGraph

## 1 Introduction

In the era of large-scale datasets, complex models (e.g., Deep Neural Networks: DNNs), and powerful computing resources capable of high throughput and simultaneous multicasting, decentralized learning of models over arbitrary network topologies (graphs)-for tasks such as natural language processing and image recognition-has been essential for advanced collective intelligence. The convergence rates of decentralized learning algorithms, including DSGD [20, 26], are primarily influenced by the graphs used for partial averaging of models within n ∈ N nodes [13]. Thus, optimizing the communication graphs to improve the expected spectral gap (by reducing deviations from global averaging), given a maximum degree k ∈ N (i.e., the maximum number of connections or multicast transmissions per node), is vital for communication-efficient decentralized learning.

Graphs used in decentralized learning are broadly categorized into static ( τ =1 ) and dynamic ( τ ≥ 2 ), with the dynamic graphs undergoing changes involving a periodic interval τ ∈ N . In contrast, static graphs, such as the ring, grid, torus [24], and exponential graph [1, 43], remain unchanged over time. Dynamic graphs rely on τ mixing matrices, specifying varying connection paths and weights. A noteworthy dynamic graph is the 1 -peer exponential graph [43], characterized by 1 -peer communication and known to achieve desirable finite-time convergence-i.e., the expected spectral gap of the product of τ mixing matrices is maximized when n is a power of two, as in Table 1.

The introduction of k -peer communication with k ≥ 2 offers the potential for improved decentralized learning, especially in settings where simultaneous multicasting is supported. This scenario has been

Table 1: Comparison of dynamic graphs. The expected spectral gap p and the periodic interval τ are defined in Assumption 4 in Sec. 2. The τ is estimated when n is factorized as n = ∏ λ i =1 ( ν i +1) , where ν 1 ≤ ... ≤ ν κ ≤ k &lt; ν κ +1 ≤ ... ≤ ν λ , and each ν i is a natural number, with κ factors being less than or equal to k . Uniform degree is satisfied ( ✓ ) when every node has the same number of connections.

| Dynamic graphs               | # of nodes n   | Max. degree k      | degree   | Uniform                                                                    | Expected spectral gap p Periodic interval τ   | Commu tativity   |
|------------------------------|----------------|--------------------|----------|----------------------------------------------------------------------------|-----------------------------------------------|------------------|
| 1 -peer hypercube [31]       | A power of 2   | 1                  | ✓        | 1 (finite-time convergence) for limited n                                  | log 2 ( n ) ⌈ log ( n ) ⌉                     | ✓                |
| 1 -peer exponential [43]     | ∀ n ∈ N        | 1                  | ✓        | Not analyzed for any n                                                     | 2                                             | ✓                |
| Base- ( k +1) [35]           | ∀ n ∈ N        | ∀ k ∈ N            | -        | 1 (finite-time convergence)                                                | κ + ∑ λ i = κ +1 2 ⌈ log k +1 ( ν i ) ⌉       | -                |
| k -peer exponential (Sec. 3) | ∀ n ∈ N        | ∀ k ∈ N            | ✓        | 1 - max i =1 ,...,n - 1 ∣ ∣ ∣ 1 ( k +1) τ sin( πi ( k +1) τ /n sin( πi/n ) | ) ∣ ∣ ∣ ⌊ log k +1 ( n ) ⌋                    | ✓                |
| Null-cascade (Sec. 4)        | ∀ n ∈ N        | ∀ k ∈ N s.t. k ≥ 2 | ✓        | 1 (finite-time convergence)                                                | κ + ∑ λ i = κ +1 ν i / (2 ⌊ k/ 2 ⌋ )          | ✓                |

explored, such as the base-( k +1) graph [35], which achieves finite-time convergence for any ( n, k ) configuration. Nevertheless, several challenges remain. First, the required periodic intervals τ can become large depending on ( n, k ) , and the emergence of nonuniform degrees (i.e., unequal numbers of connections per node) can hinder convergence rates in practice. Second, the base-( k +1) graph does not ensure commutativity of the τ mixing matrices (see Definition 4 in Sec. 2), necessitating strict coordination of periodic and synchronized communication cycles across n nodes.

For further communication-efficient decentralized learning by leveraging k -peer communication with k ≥ 2 , we propose two new dynamic graphs: the k -peer exponential graph and the null-cascade graph. Our key contributions are as follows:

Exploring principles underlying the 1 -peer exponential graph. We revisit the advantageous properties of the 1 -peer exponential graph. Ying et al. [43] proved its finite-time convergence only when n is a power of two. However, its efficiency across any n and the commutativity of its mixing matrices remain unexplored, warranting further investigation. In Sec. 3, we explore the extension to the k -peer exponential graph, which generalizes the 1 -peer exponential graph to allow k -peer communication. The k -peer exponential graph, with k ≥ 1 , is characterized by τ circulant mixing matrices of uniform weights. Their eigenvalues are analytically determined by the Discrete Fourier Transform (DFT) of expansions of Moving Average Filters (MAFs) with uniform weights. This allows us to analyze the efficiencies of the k -peer exponential graph for any n by utilizing established Digital Signal Processing (DSP) techniques. Our findings are twofold: i) finite-time convergence can be interpreted as isolating the Direct Current (DC) component and eliminating (nullifying) non-zero frequency components through a cascade of τ mixing matrices, and ii) the k -peer exponential graph functions effectively by isolating the DC component while nullifying non-zero frequency components. Although this suggests desiring finite-time convergence is only achieved when n is a power of k +1 .

Proposition of null-cascade graph. To achieve finite-time convergence for any n while preserving commutativity, our approach builds on the principles underlying the k -peer exponential graph. Our main idea is to incorporate not only MAFs with uniform weights but also Steerable Nulling Filters (SNFs) with non-uniform weights to nullify all non-zero frequencies, as detailed in Sec. 4. We refer to the resulting graph as the null-cascade graph, which ensures finite-time convergence for any n while maintaining commutativity. Limitations in the null-cascade graph are that i) it requires k ≥ 2 for forming SNFs and ii) it is most effective when the factorization of n does not include large prime numbers, as it yields a comparatively small increase in the periodic interval τ (see Table 1). The effectiveness of the proposed graphs, particularly the null-cascade graph, was confirmed through numerical experiments in Sec. 5.

## 2 Preliminaries

The simplest decentralized learning algorithm, DSGD [16], and its convergence rate are presented in Sec. 2.1. As a baseline graph, 1 -peer exponential graph is introduced in Sec. 2.2.

## 2.1 DSGD and its convergence rate

Consider a network consisting of n local nodes collaborating to solve an optimization problem:

<!-- formula-not-decoded -->

where f i : R d → R denotes the local loss function and differentiable. A fundamental algorithm for solving this problem is DSGD. It iteratively repeats two steps: (i) local updates of n local models X =[ x 1 , . . . , x n ] ∈ R d × n and (ii) partial mixing of local parameters using mixing matrices. Following [13, 43, 35], we consider dynamic communication among local nodes by periodically repeating a sequence of τ mixing matrices { W (0) , . . . , W ( τ -1) }∈ R ( n × n ) × τ ≥ 0 . Each mixing matrix is doubly stochastic, satisfying W ( ℓ ) 1 n = W ( ℓ ) ⊤ 1 n = 1 n . The update rules are given by:

<!-- formula-not-decoded -->

where r ∈{ 0 , . . . , R -1 } denotes the index of communication round, η is the learning rate, ∇ F ( X ; ξ ) ∈ R d × n represents stochastic gradients using local sampling data samples ξ , and mixing matrices are initially given. Representative dynamic graphs include the 1 -peer exponential graph [43], 1 -peer hypercube graph [31], 1 -peer EquiDyn graph [32], and base-( k +1) graph [35]. Conversely, static graphs, such as ring, grid, torus [24], and exponential graph [1] are available by setting τ =1 .

Associated with (2), a convergence analysis is provided in [13]. Under the Assumptions 1-4 listed below, a convergence rate of DSGD is given:

Assumption 1. There exists L ( &gt; 0) such that ∥∇ f i ( a ) -∇ f i ( b ) ∥ 2 ≤ L ∥ a -b ∥ 2 , ∀ a , b ∈ R d .

Assumption 2. There exists σ such that E ξ ∥∇ f i ( x ) -∇ F i ( x ; ξ ) ∥ 2 2 ≤ σ 2 , ∀ x ∈ R d .

Assumption 3. There exists ζ such that 1 n ∑ n i =1 ∥∇ f i ( x ) -∇ f ( x ) ∥ 2 2 ≤ ζ 2 , ∀ x ∈ R d .

Assumption 4 (Expected spectral gap) . There exists p ∈ (0 , 1] and τ ( ≥ 1) such that: E W ∥ XW -X ∥ 2 F ≤ (1 -p ) ∥ X -X ∥ 2 F , ∀ X ∈ R d × n , where W = W (0) · · · W ( τ -1) and X = 1 n X1 n 1 ⊤ n .

Theorem 1 (Complexity estimates of DSGD [13]) . Suppose that Assumptions 1-4 hold. For any target accuracy ϵ ( &gt; 0) , there exists a learning rate (potentially depending on ϵ ) such that the accuracy 1 R +1 ∑ R r =0 E ∥∇ f ( x ( r ) ) ∥ 2 2 ≤ ϵ can be reached after

<!-- formula-not-decoded -->

communication rounds, where f 0 = f ( x (0) ) -f ∗ and f ∗ denotes the minimum of f .

Theorem 1 highlights that maximizing the expected spectral gap ( p → 1) , and shortening the periodic interval τ are crucial for communication-efficient decentralized learning. To this end, various graphs have been explored, one of which is introduced in the next subsection.

## 2.2 1 -peer exponential graph

As a baseline dynamic graph, the 1 -peer exponential graph is introduced. It can be characterized by a sequence of circulant mixing matrices:

Definition 1 (Graph consisting of circulant mixing matrices) . Consider a graph defined by τ circulant mixing matrices, denoted by { W (0) , . . . , W ( τ -1) }∈ R ( n × n ) × τ ≥ 0 for each ℓ =0 , 1 , . . . , τ -1 , as

<!-- formula-not-decoded -->

where { h ( ℓ ) (0) , . . . , h ( ℓ ) ( n -1) } satisfies ∑ n -1 i =0 h ( ℓ ) ( i ) = 1 to be doubly stochastic.

Unless the graph consists of circulant mixing matrices, each node has the same number of connections (uniform degree). The 1 -peer exponential graph decomposes the connections of the exponential graph into τ = ⌈ log 2 ( n ) ⌉ mixing matrices, enabling 1 -peer communication as follows:

Definition 2 ( 1 -peer exponential graph [43]) . The 1 -peer exponential graph is a dynamic graph composed of τ = ⌈ log 2 ( n ) ⌉ mixing matrices ( ℓ ∈{ 0 , 1 , . . . , τ -1 } ) in the form (3) with:

<!-- formula-not-decoded -->

Definition 3 (Finite-time convergence) . A dynamic graph composed of τ mixing matrices { W (0) , . . . , W ( τ -1) } is considered a τ -finite time convergent graph if it satisfies: W (0) · · · W ( τ -1) = 1 n 1 n 1 ⊤ n . This indicates that the expected spectral gap in Assumption 4 becomes one ( p =1) .

In [43], it is proven that the 1 -peer exponential graph achieves finite-time convergence for limited n (a power of 2 ). An additional key property is the commutativity of τ mixing matrices. Specifically, their product-regardless of the permutation of the order-consistently satisfies the following:

Definition 4 (Commutativity of mixing matrices) . For any permutation ρ of indices { 0 , . . . , τ -1 } , the product W ( ρ (0)) · · · W ( ρ ( τ -1)) is equal to W (0) · · · W ( τ -1) .

As shown in Theorem 1, maximizing the expected spectral gap ( p → 1) and maintaining a small periodic interval are effective for communication-efficient decentralized learning. However, the 1 -peer exponential graph only achieves finite-time convergence for a limited n such that n is a power of two. Moreover, its efficiency across any n and the commutativity of τ mixing matrices remain unexplored, warranting further investigation.

## 3 Revising 1 -peer exponential graph to explore its underlying principles

The underlying principles of the 1 -peer exponential graph are explored. The previous study [43] established its finite-time convergence property for specific n . However, its efficiency across any n and the commutativity of τ mixing matrices remain unexplored. Further investigation may lead to more communication-efficient decentralized learning methods. Section 3.1 presents the k -peer exponential graph, a generalized form of the 1 -peer exponential graph supporting k -peer communications. In Sec. 3.2, we explore the conditions to satisfy finite-time convergence and the underlying principles of the k -peer exponential graph to explain its efficiency and commutativity.

## 3.1 k -peer exponential graph

The fact that the 1 -peer exponential graph achieves finite-time convergence for specific n can be extended to allow k -peer communication for specific n (a power of k +1 ). To illustrate this, we first introduce the k -peer exponential graph, defined as follows:

Definition 5 ( k -peer exponential graph) . Given ( n, k ) and τ = ⌊ log k +1 ( n ) ⌋ 1 , the k -peer exponential graph consists of τ circulant mixing matrices ( ℓ ∈ { 0 , 1 , . . . , τ -1 } ) , defined in the form of (3) with:

<!-- formula-not-decoded -->

An example is illustrated in Fig. 1(a); however, this does not achieve finite-time convergence since ( n, k ) = (15 , 2) . Atheoretical analysis of the k -peer exponential graph, including its expected spectral gap and commutativity, is provided in the following subsection and Appendix A.

## 3.2 Principles underlying k -peer exponential graph

This subsection presents two of our key findings: i) a necessary condition for a graph of τ circulant mixing matrices to achieve finite-time convergence, and ii) a description of the efficiency of the k -peer exponential graph, a generalization of the 1 -peer exponential graph, across any n .

i) Condition to be a finite-time convergent graph. To establish a condition for a graph composed of τ circulant mixing matrices to achieve finite-time convergence, we first analyze the target property of mixing matrix product; namely, the complete graph to connect all nodes: W comp = 1 n 1 n 1 ⊤ n . Since W comp is also a circulant mixing matrix, its eigenvalue decomposition is facilitated by the DFT [7] as W comp = D ⊤ diag([1 , 0 , . . . , 0]) D , where D ∈ C n × n denotes the DFT matrix. This reveals that averaging n local parameters is equivalent to isolating the DC component ( 0 -th frequency

1 In the 1 -peer exponential graph in Sec. 2.2, the ceiling function was used as τ = ⌈ log 2 ( n ) ⌉ ; however, we use the flooring function to derive expected spectral gap for any n in Sec. 3.2.

Figure 1: Proposed graphs: (a) k -peer exponential graph with ( n, k ) = (15 , 2) , which does not achieve finite-time convergence. (b) Null-cascade graph with ( n, k ) = (15 , 2) , which achieves finite-time convergence. Each graph consists of n = 15 nodes, depicted as black dots arranged in a circle and interconnected by lines. The colors of these lines indicate the mixing weights between the nodes in W ( ℓ ) . In (a), uniform weights ( W ( ℓ ) ij = 1 / 3) are employed (depicted in yellow), whereas in (b), n = 15 nodes when ℓ = 1 , 2 are interconnected with lines colored differently (red and green), indicating non-uniform weights specified by SNFs.

<!-- image -->

component) while nullifying all non-zero frequencies ( i ω ∈{ 1 , . . . , n -1 } ). Similarly, we analyze the product of τ circulant mixing matrices through eigenvalue decomposition:

<!-- formula-not-decoded -->

where H ( ℓ ) = diag([ H ( ℓ ) (0) , . . . , H ( ℓ ) ( n -1)]) denotes the eigenvalue matrix, of which the diagonal elements are in fact the frequency responses of { h ( ℓ ) (0) , . . . , h ( ℓ ) ( n -1) } : H ( ℓ ) ( i ω ) = ∑ n -1 i =0 h ( ℓ ) ( i ) e -2 π j i ω i/n , ( i ω ∈ { 0 , 1 , . . . , n -1 } ) , where the DC component is preserved as H ( τ -1) (0) · · · H (0) (0) = 1 due to the use of the doubly stochastic sequence ∑ n -1 i =0 h ( ℓ ) ( i ) = 1 . Additionally, since the results of (5) remain unchanged even when the order of τ circulant matrices is permuted, dynamic graphs defined in Definition 1 exhibit commutativity for any n .

From (5), the product of τ circulant mixing matrices can be interpreted as a cascade of τ (spatial) filters that maintain the DC component at 1 . Thus, a condition for finite-time convergence is given by the following equation, which ensures that all non-zero frequency components are nullified through the cascade of τ circulant mixing matrices: H ( τ -1) ( i ω ) · · · H (0) ( i ω ) = 0 , ( i ω =1 , . . . , n -1) . If at least one of the τ circulant mixing matrices has a zero response, H ( ℓ ) ( i ω ) = 0 for every non-zero frequency i ω . Let N ( ℓ ) denote the set of null frequency indices for the ℓ -th mixing matrix W ( ℓ ) . Thus, the condition for finite-time convergence can be equivalently expressed as follows: the cascade of τ circulant mixing matrices must form nulls at all non-zero frequency indices, ensuring:

<!-- formula-not-decoded -->

ii) Exploring principles underlying k -peer exponential graph. We next demonstrate the k -peer exponential graph, with k ≥ 1 , effectively increases the expected spectral gap p across any n .

We notice that (4) is an MAF with ( k +1) uniform weights and its expansions. 2 This observation motivates us to analyze the efficiencies of the k -peer exponential graph by employing DSP techniques:

Definition 6 (Frequency response of Moving Average Filter (MAF) [29]) . For any c ℓ ∈ N , frequency response of an MAF with ( c ℓ +1) -length is given by

<!-- formula-not-decoded -->

where ω ∈ [0 , 2 π ) denotes the angular frequency.

Definition 7 (Expanded sequence) . Given a sequence h (0) , . . . , h ( c ℓ ) and expansion order m ℓ ∈ N satisfying ( c ℓ +1) m ℓ ≤ n , the expanded sequence h ↑ m l (0) , . . . , h ↑ m ℓ ( n -1) is formed by separating each element of the original sequence by m ℓ -1 zeros:

<!-- formula-not-decoded -->

2 In the DSP field, "expansion" refers to the process of upsampling by inserting additional zeros between the original samples, thus lengthening the sequence for sampling rate conversion.

Lemma 1 (Frequency response of expanded sequence [29]) . Given an original sequence h ( i ) and its expansion h ↑ m ℓ ( i ) as in (7) , the frequency response of expanded sequence H ↑ m ℓ ( e j ω ) is represented as shifts in the frequency domain of the original sequence H ( e j ω ) : H ↑ m ℓ ( e j ω ) = H ( e j ωm ℓ ) .

When the original sequence is an MAF of length ( c ℓ +1) , i.e., h (0)= · · · = h ( c ℓ )=1 / ( c ℓ +1) , the frequency response of the m ℓ -order expended MAF can be derived by combining Definition 6 and Lemma 1, as follows:

<!-- formula-not-decoded -->

Recall that the k -peer exponential graph (Definition 5) consists of τ = ⌊ log k +1 ( n ) ⌋ circulant mixing matrices, where each matrix W ( ℓ ) is characterized by an MAF with a fixed communication order k and exponential expansion order m ℓ = ( k +1) ℓ . Using the DSP techniques discussed previously and the fact that the angular frequency ω and frequency index i ω are related by ω =2 πi ω /n , the expected spectral gap regarding the k -peer exponential graph is given as follows:

Theorem 2 (Expected spectral gap of k -peer exponential graph for any ( n, k ) ) . The expected spectral gap of k -peer exponential graph for any ( n, k ) with τ = ⌊ log k +1 ( n ) ⌋ is given by

<!-- formula-not-decoded -->

.

The proof is provided in Appendix A. It shows that the k -peer exponential graph achieves finite-time convergence when n is a power of k +1 . Moreover, the expected spectral gap p is increased to the order of 1 -O (1 / ( k +1) τ ) . This result theoretically supports the efficiency of the k -peer exponential graph across any n . In particular, p is efficiently maximized when n is equal to or slightly exceeds a power of ( k +1) , as illustrated in Appendix F. Furthermore, the convergence rate of DSGD using this graph is obtained by substituting p in Theorem 2 into Theorem 1, as summarized in Appendix E.

As a secondary outcome of our exploration in this subsection, we present a simplified algorithm for constructing the base-( k +1) graph [35], which achieves finite-time convergence for any ( n, k ) configuration. The details of this are provided in Alg. 5 in Appendix C. However, as noted earlier, the base-( k +1) graph does not support the commutative dynamic graph we are aiming to achieve. In the next subsection, we further extend the k -peer exponential graph to achieve finite-time convergence for any n while maintaining commutativity.

## 4 Null-cascade graph

We propose the null-cascade graph designed to achieve finite-time convergence for any n while maintaining commutativity. Recall that the k -peer exponential graph maintains commutativity but only achieves finite-time convergence for specific n (a power of k +1 ). This limitation stems from the fixed communication order k . Assuming that k ≥ 2 is available, our strategy involves the following steps: (Step 1) using MAFs with dynamic communication orders c ℓ ( ≤ k ) and dynamic expansion orders m ℓ to efficiently nullify as many non-zero frequency components as possible, and (Step 2) incorporating not only MAFs with uniform weights but also SNFs with non-uniform weights, which are defined later, to nullify remaining frequencies, as detailed in Alg. 1.

To implement this strategy, we first factorize n as n = ∏ λ i =1 ( ν i +1) , where ν 1 ≤ ... ≤ ν κ ≤ k &lt; ν κ +1 ≤ ... ≤ ν λ , with each ν i ∈ N (line 2 in Alg. 1). For Step 1, the null frequencies of expanded MAFs with communication order c ℓ ∈ N and expansion order m ℓ ∈ N are analytically derived from (8):

Lemma 2 (Null frequencies of expanded MAF) . Suppose an MAF with communication order c ℓ conforms to Definition 6. Its expansion with order m ℓ forms nulls at multiples of n ( c ℓ +1) m ℓ , excluding multiples of n m ℓ , within [0 , n ) .

The derivation is shown in Appendix B. In (6), the frequencies to be nullified are identified as non-zero frequencies i ω ∈{ 1 ,..., n -1 } , equivalently represented as multiples of b =1 within the interval [0 , n ) (line 1). To effectively nullify these frequencies, it is optimal to select m ℓ such that n ( c ℓ +1) m ℓ = b , yielding in m ℓ = n ( c ℓ +1) b (line 6). To ensure m ℓ be a natural number, ( c ℓ +1) must be

```
Algorithm 1 Null-cascade graph 1: ▷ Given n , k ( ≥ 2) , ℓ = 0 , b = 1 2: ▷ Factorization of n : n = ∏ λ i =1 ( ν i +1) , where ν 1 ≤···≤ ν κ ≤ k<ν κ +1 ≤···≤ ν λ , where ν i ∈ N 3: ▷ (Step 1) Null forming using expanded MAFs 4: for i = κ, . . . , 1 do 5: c ℓ = ν i 6: m ℓ = n ( c ℓ +1) b 7: h ( ℓ ) ( j ) = { 1 c ℓ +1 ( j =0 , . . . , c ℓ ) 0 (otherwise) 8: W ( ℓ ) = circ([ h ( ℓ ) ↑ m ℓ (0) , . . . , h ( ℓ ) ↑ m ℓ ( n -1)]) 9: ℓ = ℓ +1 /* mixing matrix index increment */ 10: b = b ( ν i +1) /*base of frequencies to be nullified*/ 11: end for 12: ▷ (Step 2) Null forming using expanded SNFs 13: for i = λ, . . . , κ +1 do 14: { ˜ γ j (0) ,..., ˜ γ j ( n -1) } j ∈{ 1 ,...,τ snf ( ν i ) } =SNF( n, k, ν i , b ) /* Alg. 3 in Appendix B */ 15: for j = 1 , . . . , τ snf ( ν i ) do 16: W ( ℓ ) = circ([ ˜ γ j (0) , . . . , ˜ γ j ( n -1)]) 17: ℓ = ℓ +1 /* mixing matrix index increment */ 18: end for 19: b = b ( ν i +1) /*base of frequencies to be nullified*/ 20: end for
```

a factor of n/b ; namely, c ℓ = ν i ( i ∈{ 1 ,..., κ } ) (line 5). In our implementation, as detailed in Alg. 1, we sequentially select from ν κ down to ν 1 as c ℓ ( ℓ ∈{ 0 ,..., κ -1 } ) (line 4), because many nulls can be efficiently formed by choosing the largest possible values for both c ℓ and m ℓ . After generating an expanded MAF, circulant mixing matrix W ( ℓ ) is computed (lines 7-8), and the frequencies still to be nullified can be updated to multiples of b ← b ( ν i +1) (line 10) because frequencies associated with a factor ν i are eliminated. This process can be incrementally repeated ( ℓ ∈{ 0 ,..., κ -1 } ) to nullify the frequencies associated with { ν κ ,..., ν 1 } and reaches b = ∏ κ i =1 ( ν i +1) in Step 1 (lines 4-11) in Alg. 1.

However, this approach is not feasible when factors ν i &gt;k exist, as c ℓ ≤ k . To address the challenge of nullifying the remaining frequencies for the factors { ν κ +1 ,..., ν λ } that exceed k , we introduce additional filters, referred to as SNFs, designed to cascade and achieve the complete nullification of non-zero frequencies in Step 2.

Definition 8 (Steerable Nulling Filter (SNF)) . When k ≥ 2 , c ℓ -peer communication with c ℓ ≤ k is available. An SNF with a length of c ℓ +1 can then be employed to form nulls at specific target frequencies. We derive the SNF by solving a polynomial specifically designed to form these nulls. By setting c ℓ to an even number c ℓ =2 ⌊ k/ 2 ⌋ , the polynomial is expressed with roots corresponding to frequency indices { q ℓ, 1 , . . . , q ℓ,c ℓ / 2 } and their conjugate frequency indices { q ∗ ℓ, 1 , . . . , q ∗ ℓ,c ℓ / 2 } , where q ∗ ℓ,ψ = n -q ℓ,ψ . The polynomial is expressed as

<!-- formula-not-decoded -->

This leads to a c ℓ -order polynomial equation of the form: γ (0) x c ℓ + · · · + γ ( c ℓ -1) x + γ ( c ℓ ) . To ensure the doubly stochastic property in a ( c ℓ +1) -length filter sequence, the normalized filter sequence is computed as γ ( i ) = γ ( i ) / ∑ c ℓ i =0 γ ( i ) for i ∈ { 0 , . . . , c ℓ } . This will result in a filter sequence consisting of non-uniform weights.

In (9), we set pairwise nulls in a conjugate relationship to ensure that SNFs are real numbers. This configuration requires k ≥ 2 . As stipulated in Definition 8, the SNF creates c ℓ =2 ⌊ k/ 2 ⌋ nulls at q ℓ,ψ and q ∗ ℓ,ψ ( ψ ∈{ 1 ,..., c ℓ / 2 } ) . The expanded SNF with order m ℓ leads the following null formation:

Lemma 3 (Null frequencies of expanded SNF) . Suppose an SNF with communication order c ℓ conforms to Definition 8. Its expansion with order m ℓ forms null frequencies at q ℓ,ψ + ϕn m ℓ and q ∗ ℓ,ψ + ϕn m ℓ , where ψ ∈{ 1 ,..., c ℓ / 2 } , and ϕ ∈{ 0 ,..., m ℓ -1 } .

Figure 2: Base-( k +1) graph with ( n, k ) = (15 , 2) , where nodes are depicted as black dots arranged in a circle and interconnected by lines. The colors of these lines indicate the mixing weights between nodes in W ( ℓ ) .

<!-- image -->

The derivation is provided in Appendix B. To nullify the remaining frequencies associated with { ν κ +1 ,..., ν λ } , selection of ( m ℓ , q ℓ,ψ ) is required in the SNF. According to Lemma 3, ( m ℓ , q ℓ,ψ ) must be selected such that q ℓ,ψ /m ℓ corresponds to the remaining frequency indices (multiples of b ). However, since this cannot be uniquely determined, an algorithm to select ( m ℓ , q ℓ,ψ ) is specified in Alg. 3 and 4 in Appendix B, ensuring the positive-definite property of the SNFs. The process of sequentially generating SNFs (line 14) and corresponding circulant matrices (line 16) to nullify the frequencies associated with { ν λ ,..., ν κ +1 } continues until b reaches n , as outlined in Step 2 (lines 13-20) of Alg. 1, thereby ensuring that nullification at all non-zero frequencies in (6) is accomplished.

Theorem 3 (Expected spectral gap of null-cascade graph) . The expected spectral gap of the nullcascade graph, specified by Alg. 1, is p = 1 (i.e., finite-time convergence) for any n with k ≥ 2 .

Discussion. An example of the null-cascade graph with ( n, k ) = (15 , 2) is depicted in Fig. 1(b). Since n can be factorized as n =15=(4+1)(2+1) , the factors are ν 1 = 2( ≤ k ) and ν 2 =4( &gt;k ) . For frequencies associated with ν 1 are nullified using an MAF, while those associated with ν 2 are nullified with two SNFs, resulting in a total of τ =3 mixing matrices. In Fig. 1(b), each matrix weights vary across each matrix ℓ ∈ { 0 , 1 , 2 } , affecting the color of lines connecting the nodes. For ℓ = 0 , an MAF with equal weights ( c 0 =2) is employed, coloring all connecting lines yellow. Conversely, for ℓ =1 , 2 , non-uniform weights ( c 1 =2 , c 2 =2) specified by SNFs are employed, resulting in lines colored differently (red and green). From Theorem 3, this achieves finite-time convergence.

Next, we present a brief comparison with the base-( k +1) graph, which also achieves finite-time convergence for any n . Figure 2 illustrates the base-( k +1) graph for ( n, k ) = (15 , 2) . Comparing this to Fig. 1(b), which depicts the null-cascade graph under the same ( n, k ) configuration, several differences emerge. Primarily, the null-cascade graph maintains a uniform degree across n nodes (every node has the same number of connections), whereas the base-( k +1) graph does not, leading to unbalanced parameter mixing within each subgroup and occasional parameter exchanges among subgroups ( ℓ ∈ { 0 , 1 , 2 } ) . This uniformity in local degree within the null-cascade graph supports stable parameter training. Moreover, the periodic interval τ , which is required for finite-time convergence, differs between null-cascade graph and base-( k +1) graph. For ( n, k ) = (15 , 2) , τ in the null-cascade graph ( τ =3) is smaller compared to that in the base-( k +1) graph ( τ =4) . However, this is not always the case as outlined in Appendix F. Including large prime numbers ν i in the factorization of n leads to an increase in τ for the null-cascade graph, since τ ≈ κ + ∑ λ i = κ +1 ν i / (2 ⌊ k/ 2 ⌋ ) , as shown in Table 1. While further discussion on the possibility of redundant counting of τ can be found in Appendix F, to demonstrate that the impact of this increase in τ is not significant, the next section includes experiments where n is a large prime number.

## 5 Numerical experiments

To show the effectiveness of the proposed graphs, we conducted two experimental tests. In Sec. 5.1, we examine consensus errors to asses the fundamental properties of the graphs. In Sec. 5.2, we evaluate the performance of decentralized learning of DNN models by combining DSGD with graphs.

## 5.1 Test 1: Consensus error investigation

Comparison graphs. We used three network configurations 3 : ( n, k ) = (15 , 2) , (17 , 2) , and (30 , 2) , all incorporating SNFs in mixing matrices, where n =17 is a large prime number. For comparison purposes, we employed both static and dynamic graphs, considering a fair communication degree

3 Other ( n, k ) -configurations are also tested in Appendix G.

Figure 3: Comparison of consensus error ∆ ( t ) to be minimized to zero for three ( n, k ) -configurations.

<!-- image -->

Figure 4: Convergence curves using test accuracy of global parameters for four configurations.

<!-- image -->

Table 2: Comparison of the highest test accuracy using global parameters.

| Dataset (heterogeneity level)      | CIFAR-10 ( α = 0 . 1 )   | CIFAR-10 ( α = 0 . 1 )   | CIFAR-10 ( α = 0 . 1 )   | CIFAR-10 ( α = 0 . 1 )   | CIFAR-10 ( α = 0 . 1 )   | CIFAR-10 ( α = 0 . 1 )   | CIFAR-100 ( α = 0 . 1 )   | CIFAR-100 ( α = 0 . 1 )   | CIFAR-100 ( α = 0 . 1 )   | CIFAR-100 ( α = 0 . 1 )   | CIFAR-100 ( α = 0 . 1 )   | CIFAR-100 ( α = 0 . 1 )   |
|------------------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|
| ( n,k ) -configuration             | (a1) (15 , 2)            | (a1) (15 , 2)            | (b1) (17 , 2)            | (b1) (17 , 2)            | (c1) (30 , 2)            | (c1) (30 , 2)            | (a2) (15 , 2)             | (a2) (15 , 2)             | (b2) (17 , 2)             | (b2) (17 , 2)             | (c2) (30 , 2)             | (c2) (30 , 2)             |
| Tested graphs                      | τ                        | test acc.                | τ                        | test acc.                | τ                        | test acc.                | τ                         | test acc.                 | τ                         | test acc.                 | τ                         | test acc.                 |
| (C1) Ring                          | 1                        | 0 . 8210                 | 1                        | 0 . 8773                 | 1                        | 0 . 7902                 | 1                         | 0 . 7140                  | 1                         | 0 . 7134                  | 1                         | 0 . 7192                  |
| (C2) Base- ( k +1)                 | 4                        | 0 . 8469                 | 5                        | 0 . 8727                 | 5                        | 0 . 8712                 | 4                         | 0 . 7145                  | 5                         | 0 . 7190                  | 5                         | 0 . 7275                  |
| (C3) Random                        | 1                        | 0 . 5815                 | 1                        | 0 . 4249                 | 1                        | 0 . 7828                 | 1                         | 0 . 1781                  | 1                         | 0 . 0835                  | 1                         | 0 . 4604                  |
| (C4) ( k +1) -partite random match | -                        | 0 . 8556                 | -                        | 0 . 8711                 | -                        | 0 . 8691                 | -                         | 0 . 7204                  | -                         | 0 . 7185                  | -                         | 0 . 7291                  |
| (P1) k -peer exponential           | 2                        | 0 . 8371                 | 2                        | 0 . 8609                 | 3                        | 0 . 8709                 | 2                         | 0 . 7163                  | 2                         | 0 . 7170                  | 3                         | 0 . 7273                  |
| (P2) Null-cascade                  | 3                        | 0 . 8660                 | 8                        | 0 . 8817                 | 4                        | 0 . 8784                 | 3                         | 0 . 7213                  | 8                         | 0 . 7203                  | 4                         | 0 . 7329                  |

( k = 2) . The static graphs consist of a (C1) ring graph, and a (C3) random graph modeled as in ErdosRényi random graph [5], adjusted to ensure k for each configuration. The dynamic graphs include a (C2) base-( k +1) graph and a (C4) ( k +1) -partite random match graph, which is a generalization of a bipartite graph, used in [43] to allow k -peer communication. The proposed graphs consist of (P1) k -peer exponential graph and (P2) null-cascade graph. Further details are provided in Appendix G.

Evaluation metric. We computed the consensus error at each iteration t as ∆ ( t ) = 1 n ∑ n i =1 ( x ( t ) i -x ) 2 , where x ( t ) i denotes local parameter ( d =1 ) at the i -th node, and x ( t ) = 1 n ∑ n i =1 x ( t ) i is global parameter. A desirable graph leads to a small consensus error in fewer iterations. For each i , x (0) i was drawn from Gaussian distribution N (0 , 1) , and performed 100 independent trials for each graph.

Experimental results. Figure 3 presents the consensus error for each ( n, k ) -configuration. As (C2) base-( k +1) graph and (P2) null-cascade graph exhibit finite-time convergence, their consensus errors drop to zero after τ iterations. As discussed in the previous section, τ for null-cascade graph tends to be small when the factorization of n does not include large prime numbers. As a result, null-cascade graph performs well in Figures 3(a) and (c). In contrast, (C2) base-( k +1) graph is preferable in Figure 3(b) because n = 17 is a large prime number, leading to a small τ . Following these two graphs in performance are (C4) ( k +1) -partite random match graph and (P1) k -peer exponential graph. When n is equal to or slightly exceeds a power of ( k +1) , as in (c) ( n, k ) = (30 , 2) , the consensus error of the k -peer exponential graph becomes small. This observation aligns with Theorem 2.

## 5.2 Test 2: Decentralized learning of DNN models

Dataset and model. We investigated decentralized learning performance of each graph using image classification benchmark tests using CIFAR-10 and CIFAR-100 with ResNet18 [8]. The

batch normalization layers were replaced by group normalization layers [41] to account for potential data heterogeneity in local datasets. To experimentally assess robustness against data heterogeneity, the training dataset was divided into n nodes to follow a Dirichlet distribution with concentration hyperparameter α [38]. We set α = 0 . 1 , representing a scenario with strong data heterogeneity. The data distributions across the n nodes are illustrated in Appendix G.

Update rules. The DSGD algorithm used in our experiments is outlined in Alg. 6 in Appendix G, which extends the update rules (2) to support multiple local parameter updates. The graphs used 3 are identical to those in Test 1. The learning rate η was pre-tuned, as provided in Appendix G. As the evaluation metric, we used the test accuracy of the global parameter x ( r ) = 1 n ∑ n i =1 x ( r ) i .

Experimental results. Figure 4 illustrates the convergence curves using test accuracy of the global parameter under several configurations in a strongly heterogeneous setting ( α =0 . 1) . The highest test accuracy during the training is summarized in Table 2. Among the tested ( n, k ) -configurations, the newly proposed (P2) null-cascade graph consistently achieved the highest test accuracy across most settings, including cases where n is a large prime number. As discussed in Sec. 4, both (P2) null-cascade graph and (C2) base-( k +1) graph ensure finite-time convergence. However, they differ in degree uniformity and the periodic interval τ . When the gap in τ is small, the non-uniform degree in (C2) base-( k +1) graph leads to unbalanced model mixing, which negatively affects training performance. In contrast, (C4) ( k + 1) -partite random match graph and (P1) k -peer exponential graph exhibit performance comparable to (C2) base-( k +1) graph, despite showing clear differences in consensus error in Test 1. This can be attributed to the fact that the degree of (C4) remains statistically uniform, while (P1) ensures strict degree uniformity. Although (C1) ring also has uniform degree, its performance is limited by a small spectral gap. Our empirical results highlight that the proposed dynamic graphs, particularly the null-cascade graph, enhance decentralized learning under heterogeneous data settings.

## 6 Conclusion

We proposed two dynamic graphs-the k -peer exponential graph and the null-cascade graph-as a result of revisiting the 1 -peer exponential graph. This reexamination revealed two key findings: (i) finite-time convergence can be achieved through a cascade of null formation at all non-zero frequencies using τ circulant matrices, and (ii) the expected spectral gap remains large for any n in the k -peer exponential graph with k ≥ 1 . Leveraging these discoveries, the null-cascade graph is constructed to ensure finite-time convergence and commutativity for any n ; however, this requires k ≥ 2 . Numerical experiments on image classification tasks demonstrated that the null-cascade graph achieved the highest test accuracy compared to conventional graphs across most test settings.

## References

- [1] Assran, M., Loizou, N., Ballas, N., and Rabbat, M. (2019). Stochastic gradient push for distributed deep learning. In International Conference on Machine Learning , pages 344-353. PMLR.
- [2] Chen, Y., Yuan, K., Zhang, Y., Pan, P., Xu, Y ., and Yin, W. (2021). Accelerating gossip SGD with periodic global averaging. In International Conference on Machine Learning , pages 1791-1802. PMLR.
- [3] Di Lorenzo, P. and Scutari, G. (2016). Next: In-network nonconvex optimization. IEEE Transactions on Signal and Information Processing over Networks , 2(2):120-136.
- [4] Ding, L., Jin, K., Ying, B., Yuan, K., and Yin, W. (2023). Dsgd-ceca: Decentralized sgd with communication-optimal exact consensus algorithm. In International Conference on Machine Learning , pages 8067-8089. PMLR.
- [5] Erdos, P., Rényi, A., et al. (1960). On the evolution of random graphs. Publ. math. inst. hung. acad. sci , 5(1):17-60.
- [6] Gao, H. and Huang, H. (2020). Periodic stochastic gradient descent with momentum for decentralized training. arXiv preprint arXiv:2008.10435 .
- [7] Gray, R. M. et al. (2006). Toeplitz and circulant matrices: A review. Foundations and Trends® in Communications and Information Theory , 2(3):155-239.
- [8] He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 770-778.
- [9] Horváth, S. and Richtárik, P. (2020). A better alternative to error feedback for communicationefficient distributed learning. arXiv preprint arXiv:2006.11077 .
- [10] Kairouz, P., McMahan, H. B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A. N., Bonawitz, K., Charles, Z., Cormode, G., Cummings, R., et al. (2021). Advances and open problems in federated learning. Foundations and trends® in machine learning , 14(1-2):1-210.
- [11] Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S., Stich, S., and Suresh, A. T. (2020). SCAFFOLD: Stochastic controlled averaging for federated learning. In International Conference on Machine Learning .
- [12] Koloskova, A., Lin, T., and Stich, S. U. (2021). An improved analysis of gradient tracking for decentralized machine learning. Advances in Neural Information Processing Systems , 34:1142211435.
- [13] Koloskova, A., Loizou, N., Boreiri, S., Jaggi, M., and Stich, S. (2020). A unified theory of decentralized SGD with changing topology and local updates. In International Conference on Machine Learning , pages 5381-5393. PMLR.
- [14] Koloskova, A., Stich, S., and Jaggi, M. (2019). Decentralized stochastic optimization and gossip algorithms with compressed communication. In International Conference on Machine Learning , pages 3478-3487. PMLR.
- [15] Kong, L., Lin, T., Koloskova, A., Jaggi, M., and Stich, S. (2021). Consensus control for decentralized deep learning. In International Conference on Machine Learning , pages 5686-5696. PMLR.
- [16] Lian, X., Zhang, C., Zhang, H., Hsieh, C.-J., Zhang, W., and Liu, J. (2017). Can decentralized algorithms outperform centralized algorithms? A case study for decentralized parallel stochastic gradient descent. In Advances in Neural Information Processing Systems .
- [17] Lian, X., Zhang, W., Zhang, C., and Liu, J. (2018). Asynchronous decentralized parallel stochastic gradient descent. In International Conference on Machine Learning , pages 3043-3052. PMLR.

- [18] Lin, T., Karimireddy, S. P., Stich, S. U., and Jaggi, M. (2021). Quasi-global momentum: Accelerating decentralized deep learning on heterogeneous data. arXiv preprint arXiv:2102.04761 .
- [19] Liu, Y., Stich, S. U., Lin Roger Wattenhofer, T., and Jaggi, M. (2021). Variance reduction in decentralized training over heterogeneous data. ETH Zurich Master's Thesis , 2020-May.
- [20] Lopes, C. G. and Sayed, A. H. (2008). Diffusion least-mean squares over adaptive networks: Formulation and performance analysis. IEEE Transactions on Signal Processing , 56(7):31223136.
- [21] Lu, Y. and De Sa, C. (2020). Moniqua: Modulo quantized communication in decentralized SGD. In International Conference on Machine Learning , pages 6415-6425. PMLR.
- [22] Marfoq, O., Xu, C., Neglia, G., and Vidal, R. (2020). Throughput-optimal topology design for cross-silo federated learning. Advances in Neural Information Processing Systems , 33:1947819487.
- [23] McMahan, B., Moore, E., Ramage, D., Hampson, S., and Arcas, B. A. (2017). Communicationefficient learning of deep networks from decentralized data. In International Conference on Artificial Intelligence and Statistics .
- [24] Nedi´ c, A., Olshevsky, A., and Rabbat, M. G. (2018). Network topology and communicationcomputation tradeoffs in decentralized optimization. Proceedings of the IEEE , 106(5):953-976.
- [25] Nedic, A., Olshevsky, A., and Shi, W. (2017). Achieving geometric convergence for distributed optimization over time-varying graphs. SIAM Journal on Optimization , 27(4):2597-2633.
- [26] Nedic, A. and Ozdaglar, A. (2009). Distributed subgradient methods for multi-agent optimization. IEEE Transactions on Automatic Control , 54(1):48-61.
- [27] Niwa, K., Harada, N., Zhang, G., and Kleijn, W. B. (2020). Edge-consensus learning: Deep learning on P2P networks with nonhomogeneous data. In International Conference on Knowledge Discovery and Data Mining .
- [28] Niwa, K., Zhang, G., Kleijn, W. B., Harada, N., Sawada, H., and Fujino, A. (2021). Asynchronous decentralized optimization with implicit stochastic variance reduction. In International Conference on Machine Learning , pages 8195-8204. PMLR.
- [29] Oppenheim, A. V. (1999). Discrete-time signal processing . Pearson Education India.
- [30] Pu, S. and Nedi´ c, A. (2021). Distributed stochastic gradient tracking methods. Mathematical Programming , 187(1):409-457.
- [31] Shi, G., Li, B., Johansson, M., and Johansson, K. H. (2015). Finite-time convergent gossiping. IEEE/ACM Transactions on Networking , 24(5):2782-2794.
- [32] Song, Z., Li, W., Jin, K., Shi, L., Yan, M., Yin, W., and Yuan, K. (2022). Communicationefficient topologies for decentralized learning with o (1) consensus rate. Advances in Neural Information Processing Systems , 35:1073-1085.
- [33] Takezawa, Y., Bao, H., Niwa, K., Sato, R., and Yamada, M. (2023). Momentum tracking: Momentum acceleration for decentralized deep learning on heterogeneous data. In Transactions on Machine Learning Research .
- [34] Takezawa, Y., Niwa, K., and Yamada, M. (2022). Theoretical analysis of primal-dual algorithm for non-convex stochastic decentralized optimization. In arXiv .
- [35] Takezawa, Y., Sato, R., Bao, H., Niwa, K., and Yamada, M. (2024). Beyond exponential graph: Communication-efficient topologies for decentralized learning via finite-time convergence. Advances in Neural Information Processing Systems , 36.
- [36] Tang, H., Gan, S., Zhang, C., Zhang, T., and Liu, J. (2018a). Communication compression for decentralized training. Advances in Neural Information Processing Systems , 31.

- [37] Tang, H., Lian, X., Yan, M., Zhang, C., and Liu, J. (2018b). D 2 : decentralized training over decentralized data. arXiv preprint arXiv:1803.07068 .
- [38] Vogels, T., He, L., Koloskova, A., Karimireddy, S. P., Lin, T., Stich, S. U., and Jaggi, M. (2021). Relaysum for decentralized deep learning on heterogeneous data. Advances in Neural Information Processing Systems , 34:28004-28015.
- [39] Vogels, T., Karimireddy, S. P., and Jaggi, M. (2020). Practical low-rank communication compression in decentralized deep learning. Advances in Neural Information Processing Systems , 33:14171-14181.
- [40] Wang, J., Sahu, A. K., Yang, Z., Joshi, G., and Kar, S. (2019). Matcha: Speeding up decentralized sgd via matching decomposition sampling. In 2019 Sixth Indian Control Conference (ICC) , pages 299-300. IEEE.
- [41] Wu, Y. and He, K. (2018). Group normalization. In European Conference on Computer Vision .
- [42] Xin, R. and Khan, U. A. (2019). Distributed heavy-ball: A generalization and acceleration of first-order methods with gradient tracking. IEEE Transactions on Automatic Control , 65(6):26272633.
- [43] Ying, B., Yuan, K., Chen, Y., Hu, H., Pan, P., and Yin, W. (2021). Exponential graph is provably efficient for decentralized deep training. Advances in Neural Information Processing Systems , 34:13975-13987.
- [44] You, R. and Pu, S. (2024). B-ary tree push-pull method is provably efficient for decentralized learning on heterogeneous data. arXiv preprint arXiv:2404.05454 .
- [45] Yu, H., Jin, R., and Yang, S. (2019). On the linear speedup analysis of communication efficient momentum sgd for distributed non-convex optimization. In International Conference on Machine Learning , pages 7184-7193. PMLR.
- [46] Yuan, K., Chen, Y., Huang, X., Zhang, Y., Pan, P., Xu, Y., and Yin, W. (2021). Decentlam: Decentralized momentum sgd for large-batch deep training. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 3029-3039.
- [47] Zhang, G. and Heusdens, R. (2017). Distributed optimization using the primal-dual method of multipliers. IEEE Transactions on Signal and Information Processing over Networks , 4(1):173187.
- [48] Zhao, H., Li, B., Li, Z., Richtárik, P., and Chi, Y . (2022). Beer: Fast O (1 /T ) rate for decentralized nonconvex optimization with communication compression. Advances in Neural Information Processing Systems , 35:31653-31667.
- [49] Zhu, T., He, F., Zhang, L., Niu, Z., Song, M., and Tao, D. (2022). Topology-aware generalization of decentralized sgd. In International Conference on Machine Learning , pages 27479-27503. PMLR.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification:

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Limitations are discussed in Appendix I.

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

Justification:

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

Justification: Source code is provided as a supplementary material.

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

Justification: We provide source code as supplementary material.

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

Justification: We provide experimental setup details in Appendix G.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We provide error bars in Appendix G.

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

Justification: Due to limitations in computing resources, we prioritized increasing the variety of benchmark tests. Additional experiments can be included in the revised version.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification:

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We note the impact statement in Appendix J.

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

Justification: Since our paper provides dynamic graphs for communication-efficient decentralized learning, data or models are not provided.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification:

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

Justification:

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our paper does not include crowdsourcing experiments and research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our paper does not include human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: We use LLMs as a check for spelling and grammar.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Technical details for the k -peer exponential graph in Sec. 3

Technical proofs for Sec. 3 are summarized in Sec. A.1 and A.2. After introducing preliminaries regarding the calculation of null frequencies using circulant mixing matrices in Sec. A.3, several examples of the k -peer exponential graph are illustrated in Sec. A.4.

## A.1 Proof of Lemma 1

Weinvestigated the frequency response of expanded sequences h ↑ m ( i ) . It is equivalently reformulated as h ↑ m ( i ) = ∑ ∞ s = -∞ h ( s ) δ ( i -ms ) using original sequence h ( i ) . Hence, applying DFT results in

<!-- formula-not-decoded -->

This indicates that the frequency response of the expanded sequence can be represented by shifting the frequency response of the original sequence H ( e j ω ) .

## A.2 Proof of Theorem 2

We computed the expected consensus rate of the k -peer exponential graph. As described in Sec. 3, the k -peer exponential graph (4) consists of τ = ⌈ log k +1 ( n ) ⌉ circulant mixing matrices W ( ℓ ) for ℓ ∈ { 0 , 1 , . . . , τ -1 } . It can be decomposed by using DFT matrix D as

<!-- formula-not-decoded -->

where the eigenvalue matrix is H ( ℓ ) ↑ m ℓ = diag([ H ( ℓ ) ↑ m ℓ (0) , . . . , H ( ℓ ) ↑ m ℓ ( n -1)]) , with

<!-- formula-not-decoded -->

In the k -peer exponential graph, we employed a filter sequence of length ( k +1) , where each sequence is expanded to the order m ℓ =( k +1) ℓ . This results in the sequence { h ( ℓ ) ↑ m ℓ (0) , . . . , h ( ℓ ) ↑ m ℓ ( n -1) } for each ℓ ∈ { 0 , 1 , . . . , τ -1 } . As described in (8), the frequency response (amplitude) of the expanded ( k +1) -length MAF with order m ℓ is given by

<!-- formula-not-decoded -->

As noted in Sec. 3, the angular frequency ω and frequency indices i ω ∈{ 0 , . . . , n -1 } are related by ω =2 πi ω /n . By substituting it into the above equation, we can compute the frequency response at frequency indices i ω ∈ { 0 , . . . , n -1 } as

<!-- formula-not-decoded -->

where the DC component holds ∣ ∣ ∣ H ( ℓ ) ↑ m ℓ (0) ∣ ∣ ∣ = 1 , which is the maximum eigenvalue of W ( ℓ ) , which is denoted by Λ 1 ( W ( ℓ ) ) . The expected consensus rate corresponds to the second-largest eigenvalue of W ( ℓ ) , which is denoted by Λ 2 ( W ( ℓ ) ) , and can be computed by selecting the maximum argument

of nonzero frequency responses (amplitude), formulated as:

<!-- formula-not-decoded -->

Thus, the expected consensus rate (product of τ mixing matrices) can be calculated as

<!-- formula-not-decoded -->

In our k -peer exponential graph, the exponential expansion m ℓ = ( k +1) ℓ is employed. Substituting m ℓ = ( k +1) ℓ into (10) results in:

<!-- formula-not-decoded -->

For specific n (a power of k +1 ; namely, n = ( k +1) τ ), substituting it into (11) results in:

<!-- formula-not-decoded -->

Otherwise, for any n , the expected consensus rate results in:

<!-- formula-not-decoded -->

These results are summarized in Theorem 2. Thanks to the exponential expansion m ℓ = ( k +1) ℓ , division is recursively simplified (via cancellation of units) in the first reformulation of (11), which we refer to as the exponential expansion trick . This is crucial for demonstrating the efficiency of the k -peer exponential graph across n and was not explored in the previous study [43].

## A.3 Calculation of null frequencies for circulant matrix employing MAFs

First, we review the principles underlying the 1 -peer exponential graph explored in Sec. 3.2. Consider a sequence of length ( c ℓ +1) , denoted by { h ( ℓ ) (0) , h ( ℓ ) (1) , . . . , h ( ℓ ) ( c ℓ ) } , for ℓ ∈ { 0 , 1 , . . . , τ -1 } . Following Definition 7, the expanded sequence with order m ℓ , resulting in an n -length sequence { h ( ℓ ) ↑ m ℓ (0) , h ( ℓ ) ↑ m ℓ (1) , . . . , h ( ℓ ) ↑ m ℓ ( n -1) } is

<!-- formula-not-decoded -->

A circulant mixing matrix employing this sequence is given by

<!-- formula-not-decoded -->

.

As performed in (5), eigenvalues of this is obtained by applying DFT, as

<!-- formula-not-decoded -->

where D denotes the DFT matrix and the eigenvalue matrix is H ( ℓ ) ↑ m ℓ = diag([ H ( ℓ ) ↑ m ℓ (0) , . . . , H ( ℓ ) ↑ m ℓ ( n -1)]) , with

<!-- formula-not-decoded -->

As mentioned in the main paper, the use of a doubly stochastic sequence ensures that ∑ n -1 i =0 h ( ℓ ) ↑ m ℓ ( i ) = 1 , preserving the DC component as H ( ℓ ) ↑ m ℓ (0) = 1 . This is the maximum eigenvalue, since the sequence { h ( ℓ ) ↑ m ℓ (0) , . . . , h ( ℓ ) ↑ m ℓ ( n -1) } is positive-definite.

## MAFis used as a sequence h ( ℓ ) ( i )

Next, we identify the null frequencies when applying the MAFs discussed in Sec. 3.2. Suppose that the MAF with ( c ℓ +1) -length is employed as filter sequence:

<!-- formula-not-decoded -->

Associated with this, the expanded sequence with order m ℓ results in an n -length filter sequence { h ( ℓ ) ↑ m ℓ (0) , h ( ℓ ) ↑ m ℓ (1) , . . . , h ( ℓ ) ↑ m ℓ ( n -1) } . As illustrated in (8), the frequency response regarding (continuous) angular frequency ω is given by

<!-- formula-not-decoded -->

where frequency index i ω and angular frequency ω are related by ω =2 πi ω /n . As proven in Sec. B.1, null frequencies can be identified as follows:

<!-- formula-not-decoded -->

## A.4 Examples of k -peer exponential graph

Example 1 ( k -peer exponential graph) . Given ( n, k ) = (16 , 1) , the 1 -peer exponential graph is constructed using τ =log 2 (16)=4 circulant matrices:

<!-- formula-not-decoded -->

Next, null indices are identified for each mixing matrix { 0 , 1 , 2 , 3 } ∈ ℓ using (13) , as:

<!-- formula-not-decoded -->

Combining the above four null frequency sets satisfies (6) , confirming that Example 1 satisfies finite-time convergence.

Figure 5: Mixing matrices of k -peer exponential graph with ( n, k ) = (16 , 1) . Each graph consists of n = 16 nodes, depicted as black dots arranged in a circle and interconnected by lines. The colors of these lines indicate the mixing weights between nodes in W ( ℓ ) .

<!-- image -->

Figure 6: Frequency characteristics (amplitude) of each sequence.

<!-- image -->

Example 2 ( k -peer exponential graph) . Given ( n, k ) = (27 , 1) , the 1 -peer exponential graph is constructed using τ = ⌊ log 2 (24) ⌋ =4 circulant matrices:

<!-- formula-not-decoded -->

However, null indices for each mixing matrix { 0 , 1 , 2 , 3 , 4 } ∈ ℓ using (13) form the following:

<!-- formula-not-decoded -->

Since (6) is not satisfied, Example 2 is not a finite-time convergence graph.

Figure 7: Mixing matrices of k -peer exponential graph with ( n, k ) = (27 , 1) . Each graph consists of n = 27 nodes, depicted as black dots arranged in a circle and interconnected by lines. The colors of these lines indicate the mixing weights between nodes in W ( ℓ ) .

<!-- image -->

Figure 8: Frequency characteristics (amplitude) of each sequence.

<!-- image -->

Example 3 ( k -peer exponential graph) . For ( n, k ) = (27 , 2) , the 2 -peer exponential graph consists of τ =log 3 (27)=3 mixing matrices:

<!-- formula-not-decoded -->

Next, null frequency sets are identified for each mixing matrix { 0 , 1 , 2 } ∈ ℓ using (13) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since this also fulfills (6) , Example 3 is confirmed to be a finite-time convergent graph.

Figure 9: Mixing matrices of k -peer exponential graph with ( n, k ) = (27 , 2) . Each graph consists of n = 27 nodes, depicted as black dots arranged in a circle and interconnected by lines. The colors of these lines indicate the mixing weights between nodes in W ( ℓ ) .

<!-- image -->

Figure 10: Frequency characteristics (amplitude) of each sequence.

<!-- image -->

## B Technical details for null-cascade graph in Sec. 4

Technical proofs of lemmas and theorems in Sec. 4 are shown in Sec. B.1-B.3. Section B.4 provides the invariant property regarding the circulant shifting, which involves assigning maximum weight to the targeted node. Subroutines used in Alg. 1 are detailed in Sec. B.5. Associated example graphs are illustrated in Sec. B.6.

## B.1 Proof of Lemma 2

In (8), the frequency response of MAF with communication order c ℓ and expansion order m ℓ is formulated. To determine the frequency indices at which the frequency response of the expanded MAF becomes zero (nulls), we need to find the angular frequencies ω such that the numerator of (8) equals zero; namely, sin( ωm ℓ ( c ℓ +1) / 2) = 0 . This satisfies when ωm ℓ ( c ℓ +1) / 2 = ϕπ where ϕ is an integer. This yields ω = 2 ϕπ m ℓ ( c ℓ +1) . Considering ϕ ranges from 0 to m ℓ ( c ℓ +1) -1 . However, since the DC component ( ω = 0) is not null and periodicity due to the expansion from Definition 7, ϕ can be determined such that multiples of 1 , excluding multiples of ( c ℓ + 1) , consisting of ϕ ∈ { 1 , . . . , ( c ℓ +1) -1 , ( c ℓ +1) + 1 , . . . , 2( c ℓ +1) -1 , 2( c ℓ +1) + 1 , . . . , ( m ℓ -1)( c ℓ +1) -1 , ( m ℓ -1)( c ℓ +1)+1 , . . . , m ℓ ( c ℓ +1) -1 } . Thus, nulls can be obtained at multiples of 2 π m ℓ ( c ℓ +1) , excluding multiples of 2 π m ℓ . From the fact that the angular frequency ω and frequency indices i ω are related by ω =2 πi ω /n , this can be reformulated: i ω can be determined such that multiples of n , excluding multiples of n within i ω ∈ [0 , 2 π ) :

m ℓ ( c ℓ +1) m ℓ

<!-- formula-not-decoded -->

## B.2 Proof of Lemma 3

Recall from Definition 8 the fundamental property of SNF, which forms c ℓ =2 ⌊ k/ 2 ⌋ nulls at q ℓ,ψ and their conjugates q ∗ ℓ,ψ = n -q ℓ,ψ for ψ ∈ { 1 ,..., c ℓ / 2 } . From Definition 7, the expanded SNF with order m ℓ , null frequencies can be shifted by 1 /m ℓ . Therefore, the null frequencies are identified as follows:

<!-- formula-not-decoded -->

## B.3 Proof of Theorem 3

First, we explain our proof strategy demonstrating that the null-cascade graph achieves finite-time convergence for any n . As explained in Sec 3, when a graph consists of τ circulant mixing matrices as in Definition 1, it achieves finite-time convergence when cascading these τ mixing matrices results in null formation at all non-zero frequencies i ω ∈ { 1 , . . . , n -1 } , as demonstrated in (6). Utilizing this principle, if we calculate a set of null frequency indices for each W ( ℓ ) ( ℓ ∈ { 0 , . . . , τ -1 } ) within the null-cascade graph and demonstrate that the union of these sets covers all non-zero frequencies, the finite-time convergence property in the null-cascade graph can be established.

Suppose n is factorized as n = ∏ λ i =1 ( ν i +1) , where ν 1 ≤ ... ≤ ν κ ≤ k &lt; ν κ +1 ≤ ... ≤ ν λ , with each ν i ∈ N (line 2 in Alg. 1). Initially, the remaining frequencies to be nullified, i ω ∈ { 1 , . . . , n -1 } , are equivalently represented as multiples of b =1 within the interval [0 , n ) (line 1 in Alg. 1). Given the factorization of n , the task of nullifying all non-zero frequencies can be interpreted as applying mixing matrices to form nulls at multiples of b , excluding multiples of b ( ν i +1) -referred to frequencies associated with ν i in the main paper. This process is repeated sequentially until b reaches n . Once b equals n , nullification at non-zero frequencies i ω ∈ { 1 , . . . , n -1 } is accomplished. Thus, the focus is on generating filters to form nulls specifically at multiples of b excluding multiples of b ( ν i +1) .

In Step 1 (lines 4-11) of Alg. 1, we utilize MAFs with dynamic selection of communication order c ℓ and expansion order m ℓ to nullify frequencies associated with { ν 1 , . . . , ν κ } satisfying ν i ≤ k . According to Lemma 2, forming nulls at frequencies associated with ν i ( ≤ k ) is straightforward. In

line 4 in Alg. 1, we sequentially select from ν κ down to ν 1 as c ℓ ( ℓ ∈{ 0 ,..., κ -1 } ) , because many nulls can be efficiently formed by choosing the largest possible values for both c ℓ and m ℓ . In lines 5 and 6 in Alg. 1, ( c 0 , m 0 ) are selected (with initialization b = 1 ) as

<!-- formula-not-decoded -->

After generating an expanded MAF with ( c 0 , m 0 ) , a corresponding mixing matrix W (0) is calculated (lines 7-8 in Alg. 1). According to Lemma 2, this mixing matrix enables to formation of nulls at frequencies that are multiples of n m 0 ( c 0 +1) = b , while excluding multiples of n m 0 = ( c 0 +1) b . Consequently, the base number to be nullified is updated to b ← b ( ν κ +1) = ν κ +1 (line 10 in Alg. 1). This process eliminates the frequencies associated with a factor ν κ .

In Step 1, this process is repeated to sequentially nullify the frequencies associated with factors { ν κ , . . . , ν 1 } satisfying ν i ≤ k . For each factor i ∈ { 1 , . . . , κ } , ( c i , m i , b ) given by Alg. 1 can be generalized as

<!-- formula-not-decoded -->

Repeating this process (Step 1) κ times results in

<!-- formula-not-decoded -->

This indicates that the remaining frequencies to be nullified after the cascade of MAFs are identified as multiples of b = ∏ i ∈{ 1 ,...,κ } ( ν i +1) .

To nullify the remaining frequencies associated with the factors { ν κ +1 , . . . , ν λ } satisfying ν i &gt; k , Step 2 involves generating SNFs with a fixed communication order c j = 2 ⌊ k 2 ⌋ as specified in Definition 8. According to Lemma 3, nulls can be formed at frequencies q ℓ,ψ + ϕn m ℓ and their conjugates q ∗ ℓ,ψ + ϕn m ℓ , where ψ = { 1 , . . . , c ℓ / 2 } and ϕ ∈ { 0 , . . . , m ℓ -1 } . While each SNF can generate many nulls depending on the choice of ( m ℓ , q ℓ,ψ ) , we particularly focus on the frequencies when setting ϕ = 0 ; specifically, ( q ℓ,ψ m ℓ , q ∗ ℓ,ψ m ℓ ) , where ( ψ ∈ { 1 , . . . , c ℓ / 2 } ) . The selection of ( m ℓ , q ℓ,ψ ) in Alg. 4 is strategically chosen to align q ℓ,ψ m ℓ with the remaining target frequencies associated with ν i ( &gt; k ) (multiples of b ). To generate nulls at multiples of b i , τ snf ( ν i ) = ⌈ ν i /c ℓ ⌉ SNFs are required to form nulls at remaining frequencies associated with ν i ( &gt; k ) , since c j = 2 ⌊ k 2 ⌋ . After these frequencies are nullified, the base number to be nullified is updated to b = ∏ i ∈{ 1 ,...,κ,λ -i +1 ,...,λ } ( ν i +1) (line 19 in Alg. 1). By repeating this nullification process with SNFs, b i eventually becomes reaches as b = ∏ i ∈{ 1 ,...,λ } ( ν i +1) = n . This comprehensive nullification at non-zero frequencies ensures that finite-time convergence is achieved within the null-cascade graph.

## B.4 Invariant frequency amplitude regarding circulant-shifting

When using non-uniform weights (e.g., SNFs introduced in Sec. 4), assigning the maximum weight to the local model held by each node would be effective in decentralized learning. To this aim, the following lemma is effective for circulant-shifting of the filter sequence (e.g., used in Alg. 3).

Lemma 4 (Invariant frequency amplitude regarding circulant shifting) . Suppose n -length sequence { h (0) , h (1) , . . . , h ( n -1) } and its frequency response H ( e j ω ) . Its circulant shifted sequence with order u is defined by h → u ( i ) = h (( i -u ) mod n ) and its frequency response is denoted by H → u ( e j ω ) . The frequency response (amplitude) of the circulant shifted sequence is invariant for any u :

<!-- formula-not-decoded -->

Proof. Suppose that n -length sequence { h (0) , h (1) , . . . , h ( n -1) } and its frequency response H ( e j ω ) are given by

<!-- formula-not-decoded -->

The circulant shift with shifting order u is defined by h → u ( i ) = h (( i -u ) mod n ) , which is is equivalent to generate the following sequence: { h → u (0) , h → u (1) , . . . , h → u ( n -1) } = { h ( u ) , h ( u + 1) , . . . , h ( n -1) , h (0) , . . . , h ( u -1) } . The frequency response of it is given by:

<!-- formula-not-decoded -->

Since | H → u ( e j ω ) | = | H ( e j ω ) e -ωu | = | H ( e j ω ) | , the frequency amplitude regarding circulant shifting is invariant.

Based on Lemma 4, Alg. 2 is constructed. This is a subroutine of Alg. 3.

## Algorithm 2 Circulant shifting function

- 1: function h → u ( i ) = CirculantShifting( h ( i ))
- 2: ▷ Ciurculant shifting to assign maximum weight to local node
- 3: u = arg max i ( h ( i ))
- 4: [ h → u (0) , h → u (1) , . . . , h → u ( n -1)] = [ h ( u ) , . . . , h ( n -1) , h (0) , . . . , h ( u -1)]

## B.5 Subroutines to compute SNFs

A subroutine used in Alg. 1 is outlined in Alg. 3. The aim of this subroutine is to nullify the remaining frequencies associated with the factors { ν κ +1 , . . . , ν λ } satisfying ν i &gt; k by generating SNFs. As described in Definition 8, SNFs that form nulls at target frequencies can be obtained by solving a polynomial expression. Specifically, line 2 (calling the subroutine in Alg. 4) selects ( m ℓ , p ℓ,ψ ) for ψ ∈ { 1 , . . . , c ℓ / 2 } , and lines 4-6 compute the conjugate frequency indices. Following (9), the polynomial is expressed with roots corresponding to frequency indices { q ℓ, 1 , . . . , q ℓ,c ℓ / 2 } and their conjugate frequency indices { q ∗ ℓ, 1 , . . . , q ∗ ℓ,c ℓ / 2 } in line 8, yielding a c ℓ -order polynomial equation of the form: γ (0) x c ℓ + · · · + γ ( c ℓ -1) x + γ ( c ℓ ) . To ensure the doubly stochastic property in a ( c ℓ +1) -length filter sequence, the normalized filter sequence is computed in lines 10-12. In line 14, thanks to the invariant frequency amplitude regarding circulant shifting, detailed in Lemma 4, the maximum weight assigned to the local model held by each node would be effective in decentralized learning.

Next, we briefly explain the core functionality of Alg. 4 for selecting ( m ℓ , p ℓ,ψ ) to nullify frequencies associated with ν i . As discussed in Sec. B.3, the selection of ( m ℓ , p ℓ,ψ ) is not uniquely determined. Therefore, an implementation for selecting ( m ℓ , p ℓ,ψ ) is introduced in Alg. 4, which can be improved in future work. As outlined in Definition 8, communication order is set as c j = 2 ⌊ k 2 ⌋ in line 6 for two reasons: (i) the desire to use an even value of c ℓ to form pairwise nulls in the conjugate relationship for real-valued filter sequence, and (ii) the goal of achieving as many nulls as possible.

According to Lemma 3, nulls can be formed at frequencies q ℓ,ψ + ϕn m ℓ and their conjugates q ∗ ℓ,ψ + ϕn m ℓ , where ψ = { 1 , . . . , c ℓ / 2 } and ϕ ∈ { 0 , . . . , m ℓ -1 } . While each SNF can generate many nulls depending on the choice of ( m ℓ , q ℓ,ψ ) , we particularly focus on the frequencies when setting ϕ = 0 ; specifically, ( q ℓ,ψ m ℓ , q ∗ ℓ,ψ m ℓ ) , where ( ψ ∈ { 1 , . . . , c ℓ / 2 } ) . To generate nulls at multiples of b i , τ snf ( ν i ) = ⌈ ν i /c ℓ ⌉ SNFs are required to form nulls at remaining frequencies associated with ν i ( &gt; k ) , since c j = 2 ⌊ k 2 ⌋ . After these frequencies are nullified, the base number to be nullified is updated to b = ∏ i ∈{ 1 ,...,κ,λ -i +1 ,...,λ } ( ν i +1) (line 19 in Alg. 1).

Finally, runtime cost and implementation of Alg. 1-4 is briefly noted. Firstly our source code associated with Alg. 1-4 is available on GitHub (see Abstract). Alg. 1- 3 are straightforward to implement and incur negligible runtime overhead. Alg. 4, which is used to select the roots q ℓ , expansions m ℓ , and communication orders c ℓ for computing SNFs, may be complex to implement and may introduce noticeable runtime costs. This potential complexity stems from the fact that combinations of ( q ℓ , m ℓ , c ℓ ) are not uniquely determined, requiring a greedy search to identify suitable candidates. While we have not exhaustively evaluated the runtime cost for all possible ( n, k ) -configurations, we confirm that for those configurations tested in our experiments (as reported in Section 5 and Appendix F), the runtime costs were negligible.

```
Algorithm 3 SNF: a subroutine of Alg. 1 1: function { ˜ γ j (0) ,..., ˜ γ j ( n -1) } j ∈{ 1 ,...,τ snf ( ν i ) } = SNF( n, k, ν i , b ) 2: { c j , m j , p j, 1 , . . . , p j,c j / 2 } j =1 ,...,τ snf ( ν i ) = SelectOrders( n, k, ν i , b ) /* Alg. 4 */ 3: for j = 1 , . . . , τ snf ( ν i ) do 4: for ψ = 1 , . . . , c j / 2 do 5: q ∗ j,ψ = n -q j,ψ /* conjugate frequency index to be nullified */ 6: end for 7: ▷ compute coefficients of a polynomial equation 8: ∏ c j / 2 ψ =1 ( x -e j2 πq j,ψ /n )( x -e j2 πq ∗ j,ψ /n ) = γ j (0) x c j + · · · + γ j ( c j -1) x + γ j ( c j ) 9: ▷ compute the normalized sequence 10: for i = 0 , . . . , c j do 11: γ j ( i ) = γ j ( i ) / ∑ c j l =0 γ j ( l ) 12: end for 13: ▷ circulant shifting after expansion with order m j 14: ˜ γ j ( i ) = CirculantShifting( γ j, ↑ m j ( i )) /* Alg. 2 */ 15: end for
```

## Algorithm 4 An implementation of subroutine to select roots, expansions, and communication orders

̸

```
1: function { c j , m j , q j, 1 , . . . , q j,c j / 2 } j =1 ,...,τ snf ( ν i ) = SelectOrders( n, k, ν i , b ) 2: β ∈ { n ν i +1 , 2 n ν i +1 , . . . , ⌊ ν i / 2 ⌋ n ν i +1 } /* frequency indices to be nullified */ 3: m base = n ( ν i +1) b /* base expansion order corresponding to factorization number ν i */ 4: j = 0 /* SNF index to be incremented */ 5: while (1) do 6: c j = 2 ⌊ k 2 ⌋ /* we use an even number less than k is used as communication order */ 7: m tmp = 1 /* temporary expansion order to incrementally search */ 8: ▷ compute roots of a polynomial equation and expansion order 9: { q j,ψ } ψ ∈{ 1 ,...,c j / 2 } = SelectFreqIndices( β, c j / 2) /* c j / 2 frequency indices to be nullified are selected from the higher indices of β . If | β | = c j / 2 , other frequencies such that interpolates β are selected. */ 10: while (1) do 11: if m tmp ≥ 2 then 12: for ψ = 1 , . . . , c j / 2 do 13: q j,ψ = m tmp · q j,ψ /* expanded SNF is used */ 14: end for 15: end if 16: for ψ = 1 , . . . , c j / 2 do 17: q ∗ j,ψ = n -q j,ψ /* conjugate frequency index to be nullified */ 18: end for 19: ▷ compute coefficients of a polynomial equation 20: ∏ c j / 2 ψ =1 ( x -e j2 πq j,ψ /n )( x -e j2 πq ∗ j,ψ /n ) = γ j (0) x c j + · · · + γ j ( c j -1) x + γ j ( c j ) 21: if { γ j (0) , . . . , γ j ( c j ) } are positive definite real numbers then 22: β = β \ { q j,ψ /m tmp } ψ ∈{ 1 ,...,c j / 2 } /* remove nullified frequency indices */ 23: break 24: else 25: m tmp = m tmp +1 /* expansion order increment */ 26: end if 27: end while 28: ▷ set expansion order m j 29: m j = m base · m tmp 30: j = j +1 /* SNF index increment */ 31: if m tmp ≥ 2 then 32: Factorization of m tmp = µ 1 , . . . , µ v , where µ 1 ≤ · · · ≤ µ v and µ 1 = 1 33: for s = v, . . . , 1 do 34: c j = 2 ⌊ k 2 ⌋ 35: m j = m base · µ s 36: for ψ = 1 , . . . , c j / 2 do 37: q j,ψ = q j -1 ,ψ 38: end for 39: β = β \ { q j,ψ /µ s } ψ ∈{ 1 ,...,c j / 2 } /* remove nullified frequency indices */ 40: j = j +1 /* SNF index increment */ 41: end for 42: end if 43: if | β | = 0 (If a set of frequencies to be nullified is empty) then 44: break 45: end if 46: end while 47: τ snf ( ν i ) = j /* number of mixing matrices employing SNFs */
```

## B.6 Examples of null-cascade graph

Example 4 (Null-cascade graph) . Given ( n, k ) = (15 , 2) , where n can be factorized as n = 15 = 5 × 3 . Since k = 2 , n is not a composite number that can be factorized using only integers less than or equal to k +1 = 3 . The null-cascade graph consists of τ = 4 circulant matrices, as

<!-- formula-not-decoded -->

where null responses for each ℓ ∈ { 0 , 1 , 2 } are obtained at following frequency indices:

<!-- formula-not-decoded -->

Combining the above four null frequency sets satisfies (6) , confirming that Example 4 satisfies finite-time convergence.

Figure 11: Mixing matrices of null-cascade graph with ( n, k ) = (15 , 2) . Each graph consists of n = 15 nodes, depicted as black dots arranged in a circle and interconnected by lines. The colors of these lines indicate the mixing weights between nodes in W ( ℓ ) .

<!-- image -->

Figure 12: Frequency characteristics (amplitude) of each sequence.

<!-- image -->

Example 5 (Null-cascade graph) . Given ( n, k ) = (17 , 2) , where n can be factorized as n = 17 . Since k = 2 , n is not a composite number that can be factorized using only integers less than or equal to k +1 = 3 . The null-cascade graph consists of τ = 8 circulant matrices, as

<!-- formula-not-decoded -->

where null responses for each ℓ ∈ { 0 , 1 , . . . , 7 } are obtained at the following frequency indices:

<!-- formula-not-decoded -->

Combining the above four null frequency sets satisfies (6) , confirming that Example 5 satisfies finite-time convergence.

Figure 13: Mixing matrices of null-cascade graph with ( n, k ) = (17 , 2) . Each graph consists of n = 17 nodes, depicted as black dots arranged in a circle and interconnected by lines. The colors of these lines indicate the mixing weights between nodes in W ( ℓ ) .

<!-- image -->

Figure 14: Frequency characteristics (amplitude) of each sequence.

<!-- image -->

Example 6 (Null-cascade graph) . Given ( n, k ) = (30 , 2) , where n can be factorized as n = 30 = 5 × 3 × 2 . Since k = 2 , n is not a composite number that can be factorized using only integers less than or equal to k +1 = 3 . The null-cascade graph consists of τ = 4 circulant matrices, as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where null responses for each ℓ ∈ { 0 , 1 , 2 , 3 } are obtained at the following frequency indices:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining the above four null frequency sets satisfies (6) , confirming that Example 6 satisfies finite-time convergence.

Q

<!-- image -->

0.05

Figure 15: Mixing matrices of null-cascade graph with ( n, k ) = (30 , 2) . Each graph consists of n = 30 nodes, depicted as black dots arranged in a circle and interconnected by lines. The colors of these lines indicate the mixing weights between nodes in W ( ℓ ) .

Figure 16: Frequency characteristics (amplitude) of each sequence.

<!-- image -->

Example 7 (Null-cascade graph) . Given ( n, k ) = (60 , 4) , where n can be factorized as n = 60 = 5 × 4 × 3 . Since k = 4 , n is a composite number that can be factorized using integers less than or equal to k +1 = 5 . The null-cascade graph consists of τ = 3 circulant matrices, as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where null responses for each ℓ ∈ { 0 , 1 , 2 } are obtained at the following frequency indices

<!-- formula-not-decoded -->

Combining the above three null frequency sets satisfies (6) , confirming that Example 7 satisfies finite-time convergence.

Figure 17: Mixing matrices of null-cascade graph with ( n, k ) = (60 , 4) . Each graph consists of n = 60 nodes, depicted as black dots arranged in a circle and interconnected by lines. The colors of these lines indicate the mixing weights between nodes in W ( ℓ ) .

<!-- image -->

Figure 18: Frequency characteristics (amplitude) of each sequence.

<!-- image -->

Example 8 (Null-cascade graph) . Given ( n, k ) = (60 , 3) , where n can be factorized as n = 60 = 5 × 4 × 3 . Since k = 3 , n is not a composite number that can be factorized using only integers less than or equal to k +1 = 4 . The null-cascade graph consists of τ = 4 circulant matrices, as

<!-- formula-not-decoded -->

where null responses for each ℓ ∈ { 0 , 1 , 2 , 3 } are obtained at the following frequency indices

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining the above four null frequency sets satisfies (6) , confirming that Example 8 satisfies finite-time convergence.

<!-- image -->

0.05

Figure 19: Mixing matrices of null-cascade graph with ( n, k ) = (60 , 3) . Each graph consists of n = 60 nodes, depicted as black dots arranged in a circle and interconnected by lines. The colors of these lines indicate the mixing weights between nodes in W ( ℓ ) .

Figure 20: Frequency characteristics (amplitude) of each sequence.

<!-- image -->

Example 9 (Null-cascade graph) . Given ( n, k ) = (60 , 2) , where n can be factorized as n = 60 = 5 × 3 × 2 × 2 . Since k = 2 , n is not a composite number that can be factorized using only integers less than or equal to k +1 = 3 . The null-cascade graph consists of τ = 5 circulant matrices, as

<!-- formula-not-decoded -->

where null responses for each ℓ ∈ { 0 , 1 , 2 , 3 , 4 } are obtained at the following frequency indices

<!-- formula-not-decoded -->

Combining the above five null frequency sets satisfies (6) , confirming that Example 9 satisfies finitetime convergence.

<!-- image -->

0.05

Figure 21: Mixing matrices of null-cascade graph with ( n, k ) = (60 , 2) . Each graph consists of n = 60 nodes, depicted as black dots arranged in a circle and interconnected by lines. The colors of these lines indicate the mixing weights between nodes in W ( ℓ ) .

Figure 22: Frequency characteristics (amplitude) of each sequence.

<!-- image -->

## C Simplified algorithm to construct base-( k +1) graph

This section consists of the following:

(i) Interpretation of the base-( k +1) graph. We summarized our interpretation of the base-( k +1) graph based on the principles underlying the 1 -peer exponential graph in Sec. C.1. We found that: i) applying the simple base-( k +1) graph , which is included in the base-( k +1) graph, effectively nullifies frequencies except for those that are multiples of a specific number, and ii) the remaining non-zero frequencies can be eliminated by utilizing circulant matrices employing MAFs with an appropriate choice of length c ℓ and expansion order m ℓ .

(ii) Simplified algorithm to construct the base-( k +1) graph : Based on the insights in Sec. C.1, a new and simple algorithm to compute mixing matrices in the base-( k +1) graph is developed in Sec. C.2.

## C.1 Our interpretation of the base-( k +1) graph

## (i-a) Brief introduction of the simple base-( k +1) graph

First, we briefly explain the simple base-( k +1) graph, which is included in the base-( k +1) graph. Since the algorithm to compute mixing matrices in the simple base-( k +1) graph is complex (see Alg. 2 in [34]), we aim to provide an intuitive explanation of the simple base-( k +1) graph through an illustrative example:

Example 10 (Simple base-( k +1) graph with ( n, k ) = (3 , 1) ) . First, let us decompose n using the power of ( k + 1) . Consider the case when ( n, k ) = (3 , 1) . Then, n can be decomposed as n = 3 = (1 + 1) 1 + (1 + 1) 0 = 2 1 + 2 0 , resulting in the generation of two subgroups: {{ x 1 , x 2 } ︸ ︷︷ ︸ S 1 :2 1 nodes , { x 3 } ︸︷︷︸ S 2 :2 0 node } . The simple base-( k +1) graph with ( n, k ) = (3 , 1) consists of τ = 3 mixing

<!-- formula-not-decoded -->

Figure 23: Mixing matrices of simple base-( k +1) graph with ( n, k ) = (3 , 1) . Each graph consists of n = 3 nodes, depicted as black dots arranged in a circle and interconnected by lines. The colors of these lines indicate the mixing weights between nodes in W ( ℓ ) .

<!-- image -->

Example 11 (Simple base-( k +1) graph with ( n, k ) = (5 , 3) ) . First, let us decompose n using the power of ( k + 1) . Consider the case when ( n, k ) = (5 , 3) . Then, n can be decomposed as n = 5 = (3 + 1) 1 + (3 + 1) 0 = 4 1 + 4 0 , resulting in the generation of two subgroups: {{ x 1 , x 2 , x 3 , x 4 } ︸ ︷︷ ︸ S 1 :4 1 nodes , { x 5 } ︸︷︷︸ S 2 :4 0 node } . The simple base-( k +1) graph with ( n, k ) = (5 , 3) consists of τ = 3

mixing matrices

{

W

(0)

,

W

(1)

,

W

(2)

}

:

<!-- image -->

Note that this example corresponds to Fig. 2 in Sec. 5.

Figure 24: Mixing matrices of simple base-( k +1) graph with ( n, k ) = (5 , 3) . Each graph consists of n = 5 nodes, depicted as black dots arranged in a circle and interconnected by lines. The colors of these lines indicate the mixing weights between nodes in W ( ℓ ) .

<!-- image -->

It is important to note that, in the simple base-( k +1) graph , the use of mixing matrices is not limited to circulant matrices. Thus, the commutativity of the τ mixing matrices does not hold.

Examples with other ( n, k ) configurations are illustrated in Sec. C.3. The simple base-( k +1) graph, included in the base-( k +1) graph, is basically structured by three steps: (1a): intra-subgroup averaging within each subgroup using the k -peer hypercube graph, which is implemented by W (0) , (2): inter-subgroup mixing of the models across subgroups, which is implemented by W (1) , and (1b): intra-subgroup averaging using the k -peer hypercube graph, which is implemented by W (2) .

Although the simple base-( k +1) graph achieves finite-time convergence for any ( n, k ) combination, two issues remain: i) the periodic interval τ tends to be large, mainly because the twice repetition of intra-subgroup averaging, resulting in O (2 ⌈ log k +1 ( n ) ⌉ ) 4 , and ii) commutativity among the mixing matrices does not hold.

## (i-b) Our interpretation of the base-( k +1) graph

Next, we delve into our interpretation of base-( k +1) graph. The algorithm currently used to compute mixing matrices within the base-( k +1) graph, including the simple base-( k +1) graph, is complex (refer to Alg. 3 in [34]). Our primary aim in this section is to develop a new algorithm in the following subsection that will calculate mixing matrices in the base-( k +1) graph. This new algorithm will be based on the principles underlying the 1 -peer exponential graph in Sec. 3, presenting a different formulation of base-( k +1) graph. As an initial step, we explain our interpretation regarding base-( k +1) graph in detail below.

Let us recall that the simple base-( k + 1) graph ensures finite-time convergence for any ( n, k ) configuration. The base-( k + 1) graph is employed to average local models within each subgroup. Suppose that n can be factorized as n = µ 1 µ 2 , which results in the formation of µ 1 subgroups, each containing µ 2 nodes. Assuming that local models within

4 The original paper [34] states the order of the periodic interval τ as O (log k +1 ( n )) . This representation remains correct since coefficients multiplied by the main component log k +1 ( n ) are general. However, we apply a coefficient of 2 to estimate τ more accurately, reflecting the twice iteration of intra-subgroup averaging.

each subgroup are averaged/identified using the simple base-( k + 1) graph, this results in { ˜ x 1 , ˜ x 1 , . . . , ˜ x 1 ︸ ︷︷ ︸ µ 2 nodes , ˜ x 2 , ˜ x 2 , . . . , ˜ x 2 ︸ ︷︷ ︸ µ 2 nodes , · · · , ˜ x µ 1 , ˜ x µ 1 , . . . , ˜ x µ 1 ︸ ︷︷ ︸ µ 2 nodes } . This is equivalent to a permutation that repeats µ 1 -length sequence { ˜ x 1 , ˜ x 2 , . . . , ˜ x µ 1 } by µ 2 times, referred to as repeated sequence. The arrangement appears as follows: { ˜ x 1 , ˜ x 2 , . . . , ˜ x µ 1 ︸ ︷︷ ︸ 1 time , ˜ x 1 , ˜ x 2 , . . . , ˜ x µ 1 ︸ ︷︷ ︸ 2 times , . . . , ˜ x 1 , ˜ x 2 , . . . , ˜ x µ 1 ︸ ︷︷ ︸ µ 2 times } .

For simple notation, suppose that µ 1 -length sequence { h (0) , h (1) , . . . , h ( µ 1 -1) } and its µ 2 -times repeated sequence { h ′ (0) , h ′ (1) , . . . , h ( µ 1 µ 2 -1) } = { h (0) , h (1) , . . . , h ( µ 1 -1) , h (0) , h (1) , . . . , h ( µ 1 -1) , . . . , h (0) , h (1) , . . . , h ( µ 1 -1) } are given. The DFT of the original sequence h ( i ) is given by

<!-- formula-not-decoded -->

where i ∈ { 0 , . . . , µ 1 -1 } in H ( i ) is frequency index. Since the repeated sequence can be equivalently formulated using the original sequence, as h ′ ( i ) = h ( i mod µ 1 ) . Thus, the DFT of the repeated sequence is given by

<!-- formula-not-decoded -->

where T 1 takes µ 2 when i is a multiple of µ 1 µ 2 ; otherwise, it is zero. Consequently, we obtain

<!-- formula-not-decoded -->

where i ∈ { 0 , . . . , µ 1 µ 2 -1 } in H ′ ( i ) denotes the frequency index. This indicates that many frequency components are nullified if n = µ 1 µ 2 -length sequence is composed of µ 2 times repeats of µ 1 -length sequence. This is applicable to the local models through the simple base-( k +1) graph, denoted by { ˜ x 1 , ˜ x 2 , . . . , ˜ x µ 1 ︸ ︷︷ ︸ 1 time , . . . , ˜ x 1 , ˜ x 2 , . . . , ˜ x µ 1 ︸ ︷︷ ︸ µ 2 times } . Since this is also a repeated sequence, we can recognize the effect of the simple base-( k +1) graph as null forming except the frequency indices satisfying ( i mod µ 2 = 0) .

Noticing that the remaining frequencies to be nullified are multiples of µ 2 , MAFs with a suitable choice of c ℓ -length and m ℓ -order expansion prove beneficial. Recall that the null frequency indices with MAFs with c ℓ -length and m ℓ -order expansion are theoretically analyzed in Sec. 3 and Appendix A, specifically in (13). Moreover, this approach is efficiently utilized in null-cascade graph in Sec. 4, with a dynamic/suitable choice of { c ℓ , m ℓ } , resulting in the forming of nulls at non-zero frequencies. This strategy is also applicable to nullify the remaining frequencies. In the next subsection, building on the ideas discussed so far, a new and simplified algorithm to construct mixing matrices in the base-( k +1) graph will be developed.

## C.2 Simplified algorithm to construct the base-( k +1) graph

The original algorithm to construct the base-( k +1) graph is outlined in Alg. 3 in [34]; however, it is somewhat complex. Assuming that the algorithm to construct the simple base-( k +1) graph is given

## Algorithm 5 New algorithm to construct base-( k +1) graph

- 1: ▷ Given ( n, k ) and ℓ = 0
- 2: ▷ Factorization of n as n = ∏ λ i =1 ( ν i +1) , where ν 1 ≤··· ≤ ν κ ≤ k ≤ ν κ +1 ≤··· ≤ ν λ , where ν i ∈ N .
- 3: ▷ (Step 1) Applying simple base-( k +1) graph to be repeated sequence (which is equivalent to form nulls as in (14))
- 4: for i = λ, ..., κ +1 do
- 5: ▷ Generate mixing matrices Z ∈ R ( n × n ) × τ sb + to be repeated sequence
- 6: { Z (0) , . . . , Z ( τ sb -1 ) } = SimpleBaseGraphRepeatedSequence( ν i +1 , n, k )
- 7: for j = 0 , ..., τ sb -1 do
- 8: W ( ℓ ) = Z ( j )
- 9: ℓ = ℓ +1 /* mixing matrix index increment */
- 10: end for
- 11: end for
- 12: ▷ Remaining frequency to be eliminated (multiples of b from (14))
- 13: b = ∏ λ i = κ +1 ( ν i +1) /* base of frequencies to be nullified */
- 14: ▷ (Step 2) Null forming using expanded MAFs
- 15: for i = κ, . . . , 1 do
- 16: ( c ℓ , m ℓ ) = ( ν i , n ( ν i +1) b )

<!-- formula-not-decoded -->

- 18: W ( ℓ ) = circ([ h ( ℓ ) ↑ m ℓ (0) , . . . , h ( ℓ ) ↑ m ℓ ( n -1)])
- 19: ℓ = ℓ +1 /* mixing matrix index increment */
- 20: b = b ( ν i +1) /* base of frequencies to be eliminated */
- 21: end for
- 22: function { Z (0) , . . . , Z ( τ sb -1) } = SimpleBaseGraphRepeatedSequence( ν j +1 , n, k )
- 23: ▷ Generate simple base-( k +1) graph Y ∈ R (( ν j +1) × ( ν j +1)) × τ sb + to make consensus among ( ν j +1) nodes.
- 24: { Y (0) , . . . , Y ( τ sb -1 ) } = SimpleBaseGraph( ν j +1 , k

<!-- formula-not-decoded -->

- 25: ▷ Generate mixing matrices Z ( r ) ∈ R ( n × n ) × τ sb + to be repeated sequence
- 26: for i = 0 , . . . , τ sb -1 n/ ( ν j +1) -1
- do

<!-- formula-not-decoded -->

## 28: end for

(Alg. 2 in [34]), a new and simplified algorithm for constructing the base-( k +1) graph is presented, leveraging the insights discussed in Sec. C.1.

Our new algorithm for computing the base-( k +1) graphis detailed in Alg. 5. In line 1, n is factorized into κ ( ≤ 1) components as n = ∏ κ -1 i =0 ( c i +1) , where c i ∈ N are ordered as c 0 ≤ · · · ≤ c ν -1 ≤ k ≤ c ν ≤ · · · ≤ c κ -1 . This factorization is critical to determine the integration strategy of the simple base-( k +1) graph and circulant matrices using MAFs. For c i &gt; k for i ∈ { ν, . . . , κ -1 } , the simple base-( k +1) graph is applied (step 1). Conversely, for c i ≤ k for i ∈ { 0 , . . . , ν -1 } , circulant matrices using MAFs are employed (step 2). Lines 3 to 11 compute mixing matrices based on the simple base-( k +1) graph . Through τ sb mixing matrices, a repeated sequence of local models { ˜ x 1 , ˜ x 2 , . . . , ˜ x n/ ( c r +1) , . . . , ˜ x 1 , ˜ x 2 , . . . , ˜ x n/ ( c r +1) } is generated. The application of

︸

1

time

(

c

r

+1)

times the simple base-( k +1) graph is repeated for c i &gt; k for i ∈ { ν, . . . , κ -1 } . From (14), this cascade of the simple base-( k +1) graph remains non-nullified frequency components i , which are multiples of b = ∏ κ -1 i = ν ( c i + 1) within i ∈ { 0 , . . . , n } . For nullifying the remaining frequencies in step 2,

︷︷

︸

︸

︷︷

︸

circulant matrices employing MAFs with c r -length and expansion order m r are generated in lines 14 to 21. We used the techniques in the null-cascade graph in Alg. 1.

Although our Alg. 5 diverges significantly from the original algorithm (Alg. 3 in [34]), they generate equivalent mixing matrices. These intriguing outcomes stem from our theoretical analysis of the principles underlying the 1 -peer exponential graph.

## C.3 Example illustrations

## C.3.1 Simple base-( k +1) graph

Figure 25: Mixing matrices of simple base-( k +1) graph with ( n, k ) = (5 , 1) . Each graph consists of n = 5 nodes, depicted as black dots arranged in a circle and interconnected by lines. The colors of these lines indicate the mixing weights between nodes in W ( ℓ ) .

<!-- image -->







<!-- image -->





Figure 26: Mixing matrices of simple base-( k +1) graph with ( n, k ) = (5 , 2) . Each graph consists of n = 5 nodes, depicted as black dots arranged in a circle and interconnected by lines. The colors of these lines indicate the mixing weights between nodes in W ( ℓ ) .

## C.3.2 Base-( k +1) graph

Following Alg. 5, several illustrations of the base-( k +1) graph are shown.

Example 14 (Base-( k +1) graph) . First, the factorization of n is given as: n = 15 = (2+1) · (4+1) ; namely, ν 1 = 2 , ν 2 = 4 . Since k = 2 , n includes a factorization number ν 2 = 4 such that ν i &gt; k . Following Alg. 5, the simple base-( k +1) graph with ( n, k ) = (5 , 2) is computed. From Example 13, { Y ( i ) } i ∈{ 0 , 1 , 2 } are prepared.

<!-- formula-not-decoded -->

The base-( k +1) graph consists of τ = 4 circulant matrices as

<!-- formula-not-decoded -->

where null responses for each ℓ ∈ { 0 , 1 , 2 , 3 , 4 } are obtained at following frequency indices

<!-- formula-not-decoded -->

(Null frequency indices are multiples of 1, excluding multiples of 5, from (14) characterizing the frequency response of the simple base-( k +1) graph. )

<!-- formula-not-decoded -->

Combining the above null frequency sets satisfies (6) , confirming that this satisfies finite-time convergence.

Figure 27: Mixing matrices of the simple base-( k +1) graph with ( n, k ) = (15 , 2) . Each graph consists of n = 15 nodes, depicted as black dots arranged in a circle and interconnected by lines. The colors of these lines indicate the mixing weights between nodes in W ( ℓ ) .

<!-- image -->

Example 15 (Base-( k +1) graph) . First, the factorization of n is given as: n = 30 = (1 + 1) · (2 + 1) · (4 + 1) ; namely, ν 1 = 1 , ν 2 = 2 , ν 3 = 4 . Since k = 2 , n includes a factorization number ν 3 = 4 such that ν i &gt; k . Following Alg. 5, the simple base-( k +1) graph with ( n, k ) = (5 , 2) is computed. From Example 13, { Y ( i ) } i ∈{ 0 , 1 , 2 } are prepared.

<!-- formula-not-decoded -->

The base-( k +1) graph consists of τ = 4 circulant matrices as

<!-- formula-not-decoded -->

where null responses for each ℓ ∈ { 0 , 1 , 2 , 3 , 4 } are obtained at following frequency indices

<!-- formula-not-decoded -->

(Null frequency indices are multiples of 1, excluding multiples of 5, from (14) characterizing the frequency response of the simple base-( k +1) graph. )

<!-- formula-not-decoded -->

Combining the above null frequency sets satisfies (6) , confirming that this satisfies finite-time convergence.

Figure 28: Mixing matrices of simple base-( k +1) graph with ( n, k ) = (30 , 2) . Each graph consists of n = 30 nodes, depicted as black dots arranged in a circle and interconnected by lines. The colors of these lines indicate the mixing weights between nodes in W ( ℓ ) .

<!-- image -->

## D A consideration regarding the benefit of commutativity

A brief consideration of commutativity (Definition 4) follows. Reducing the discrepancy between the global model x ( r ) and local models x ( r ) i , specifically, minimizing ∑ n i =1 ∥ x ( r ) -x ( r ) i ∥ is critical for accelerating decentralized learning. This term appears explicitly in the convergence analysis of DSGD (implicitly included in Theorem 1). Enforcing commutativity of the graph mixing matrices can reduce this discrepancy.

To illustrate this, we reformulate the update rules of DSGD recursively, as follows:

<!-- formula-not-decoded -->

Next, we investigate the property of ∏ r s 2 = s 1 W ( s 2 ) for finite-time convergent graph with periodic interval τ .

[Case 1: Commutativity of mixing matrices is satisfied] (e.g., k -peer exponential graph and null-cascade graph)

When commutativity of mixing matrices is satisfied, W ( ρ (0)) W ( ρ (1)) · · · W ( ρ ( τ -1)) = 1 n 1 n 1 ⊤ n for any permutation ρ . The product ∏ r s 2 = s 1 W ( s 2 ) can be categorized into two cases:

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

These results indicate that no discrepancy among n nodes can occur for terms with index s 1 ∈ [0 , r -τ +1] (in Case 1a). In contrast, discrepancy among n nodes can occur for terms with index s 1 ∈ [ r -τ +2 , r ] (in Case 1b), which leads to degradation in the convergence rate.

[Case 2: Commutativity of mixing matrices is NOT satisfied] (e.g., base-( k +1) graph)

When we denote mod ( r, τ ) = ς , the product ∏ r s 2 = s 1 W ( s 2 ) can be categorized into two cases:

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

These results indicate that no discrepancy among n nodes can occur for terms with index s 1 ∈ [0 , r -τ -ς +1] (in case 2a). In contrast, discrepancy among n nodes can occur for terms with index s 1 ∈ [ r -τ -ς +2 , r ] (in case 2b), which leads to degradation in the convergence rate.

Comparing these two cases, the components that contribute to discrepancies among the n nodes tend to be more significant when mod ( r, τ ) = ς is non-zero. Although this difference may be small and therefore difficult to observe empirically, we have theoretically demonstrated the advantage of using commutative mixing matrices.

## E Convergence rate comparison using various graphs

As a theoretical contribution, the convergence rates of DSGD over various graphs are summarized in Table 3. These rates are derived by substituting the expected spectral gap p and periodic interval τ specific to each graph into the convergence rate of DSGD in Theorem 1.

Table 3: Comparison of convergence rates of DSGD over various graphs.

| Graphs                 | Convergence rate of DSGD                                                                       | Maximum degree k   | # of nodes n               |
|------------------------|------------------------------------------------------------------------------------------------|--------------------|----------------------------|
| Ring [24]              | O ( σ 2 nϵ 2 + ζn 2 + σn ϵ 3 / 2 + n 2 ϵ ) · Lf 0 √                                            | 2                  | ∀ n ∈ N                    |
| Torus [24]             | O ( σ 2 nϵ 2 + ζn + σ n ϵ 3 / 2 + n ϵ ) · Lf 0                                                 | 4                  | ∀ n ∈ N                    |
| Exp. [43]              | O ( σ 2 nϵ 2 + ζ log 2 ( n )+ σ √ log 2 ( n ) ϵ 3 / 2 + log 2 ( n ) ϵ ) ·                      | ⌈ log 2 ( n ) ⌉    | ∀ n ∈ N                    |
| 1 -peer exp. [43]      |      O ( σ 2 nϵ 2 + ζ log 2 ( n )+ σ √ log 2 ( n ) ϵ 3 / 2 + log 2 ( n ) ϵ ) · n.a.       | 1                  |    A power of 2 ∀ n ∈ N |
| 1 -peer hypercube [31] | O ( σ 2 nϵ 2 + ζ log 2 ( n )+ σ √ log 2 ( n ) ϵ 3 / 2 + log 2 ( n ) ϵ ) ·                      | 1                  | A power of 2               |
| Base- ( k +1) [35]     | O ( σ 2 nϵ 2 + ζτ base + σ √ τ base ϵ 3 / 2 + τ base ϵ ) · Lf 0 ∗ 1                            | ∀ k ∈ N            | ∀ n ∈ N                    |
| k -peer exp.           | 2 nϵ 2 + ζ log k +1 ( n )+ σ √ p kpexp log k +1 ( n ) p kpexp ϵ 3 / 2 + log k +1 ( n p kpexp ϵ | ∀ k ∈ N            | ∀ n ∈ N                    |
| Null-cascade           | O ( σ 2 nϵ 2 + ζτ null + σ √ τ null ϵ 3 / 2 + τ null ϵ ) · Lf 0 ∗ 3                            | ∀ k ( ≥ 2)         | ∀ n ∈ N                    |

<!-- formula-not-decoded -->

*3: τ null = κ + ∑ λ i = κ +1 ν i / (2 ⌊ k/ 2 ⌋ ) , where n is factorized as n = ∏ λ i =1 ( ν i +1) , where ν 1 ≤ ... ≤ ν κ ≤ k &lt;ν κ +1 ≤ ... ≤ ν λ , and each ν i is a natural number, with κ factors being less than or equal to k .

## F Relationships between expected consensus rate, periodic interval and ( n, k ) configurations

In Sec. F.1, the relationships between the expected consensus rate p , the periodic interval τ , and the number of nodes n are illustrated for various values of k . These illustrations reveal that the periodic interval τ required for achieving finite-time convergence ( p =1) in the null-cascade graph tends to increase when the factorization of n includes large prime numbers, as discussed in Sec. 4. Nonetheless, experimental results presented in Sec. 5 indicate that performance is not compromised even under such scenarios. The primal reason is discussed in Sec. 5, with additional considerations provided in Sec. F.2.

## F.1 Illustrations of expected consensus rate p , periodic interval τ , and ( n, k ) configurations

The expected consensus rate p , the periodic interval τ , and ( n, k ) configurations are numerically investigated across several graphs. This examination covers a broad range of n and k = { 2 , 3 , 4 } , as depicted in Fig. 29. For the k -peer exponential graph, the expected consensus rate becomes one (indicating finite-time convergence) for specific n (a power of k +1 ). As n deviates from this condition, the expected consensus rate increases, as observed with configurations like ( n, k ) = (17 , 2) discussed in Sec. 5. The periodic interval τ for the k -peer exponential graph typically remains small, given by τ = ⌊ log k +1 ( n ) ⌋ . On the other hand, the consensus rate of both base-( k +1) graph and null-cascade graph consistently reaches one, indicating they achieve finite-time convergence for any n . However, the required periodic intervals τ to achieve finite-time convergence differ between these two graphs. When the factorization of n includes large prime numbers, τ in the null-cascade graph tends to increase; if not, τ in the null-cascade graph is generally equal to or less than that in the base-( k +1) graph.

## F.2 Additional discussion

As shown in Fig. 29, the periodic interval τ for the null-cascade graph increases when the factorization of n includes large prime numbers. Nonetheless, the experimental results presented in Sec. 5 demonstrated that the performance is not degraded even under such scenarios. The prime reason for this is discussed in the last paragraph of Sec. 4. This suggests that balanced mixing among n nodes in the null-cascade graph leads to better experimental results.

Additionally, there is the issue of potential redundant counting of τ in the null-cascade graph. This arises because τ in the null-cascade graph is counted as the number of mixing matrices required to achieve finite-time convergence. However, a smaller τ could potentially be set by accepting a modest expected consensus rate. Furthermore, repeating a graph that achieves finite-time convergence twice results in a doubled periodic interval, yet the expected consensus rate remains unchanged. Even though the periodic interval can be doubled, this does not necessarily slow the convergence rate. Such observations indicate that a rigorous determination of τ in Assumption 4 cannot yet be definitively established, pointing to a direction for future research.

Figure 29: Relationships between expected consensus rate p , periodic interval τ , and n with k = { 2 , 3 , 4 } .

<!-- image -->

## G Additional experiments

Data distributions of CIFAR-10 and CIFAR-100. As described in Sec. 5, the training dataset of CIFAR-10 and CIFAR-100 5 were divided into n local datasets D i ( i ∈ { 1 , . . . , n } ) to follow a Dirichlet distribution with concentration hyperparameter α , using the source code used in [38] 6 . We set α = 0 . 1 , representing a scenario with significant data heterogeneity. The distributions of the dataset for n = { 15 , 17 , 30 } are illustrated in Figs. 30 and 31.

Figure 30: Data distributions of CIFAR-10 training dataset when (a) n = 15 , (b) n = 17 , and (c) n = 30 .

<!-- image -->

Figure 31: Data distributions of CIFAR-100 training dataset when (a) n = 15 , (b) n = 17 , and (c) n = 30 .

<!-- image -->

Update rules based on DSGD. Our DSGD-based update rules that allow multiple local updates are summarized in Alg. 6. We set the number of inner multiple local updates to T = 100 for CIFAR-10 and T = 10 for CIFAR-100. The minibatch size to compute stochastic gradient was 64 . As noted in the main paper, we computed the test accuracy of the global parameter x = 1 n ∑ n i =1 x i once every dozen communication rounds, for the evaluation metric.

## Algorithm 6 DSGD used in the experiments in Sec. 5

- 1: ▷ Initialization x 1 = , . . . , = x n , η, R, T, D i ( i ∈ [ n ]) 2: ▷ Set mixing matrices { W (0) , . . . , W ( τ -1) } 3: for (Outer communication round) r = 0 , . . . , R -1 do 4: x ( r, 0) i = x ( r ) i 5: for (Inner local updates) t = 0 , . . . , T -1 do 6: x ( r,t +1) i = x ( r,t ) i -η ∇ f i ( x ( r,t ) i ; ξ ( r,t ) i ) /* minibatch sampling ξ ( r,t ) i ∼ D i */ 7: end for 8: x ( r + 1 2 ) i = x ( r,T ) i 9: ▷ Partial averaging 10: x ( r +1) i = ∑ n j =1 W (mod( r,τ )) ij x ( r + 1 2 ) i 11: end for

5 https://www.cs.toronto.edu/~kriz/cifar.html

6 https://github.com/epfml/relaysgd

Hyperparameter tuning. For Alg. 6, the learning rate η was initially pre-tuned for each datasetmodel combination. To maintain consistency across various graphs and ensure a fair comparison, we used a common η across all tested graphs. For this aim, we investigated η , which maximizes the test accuracy for each dataset-model combination, based on the results from single-node model training. Details of the associated hyperparameter settings are provided in Table 4. Through this hyperparameter tuning process, a learning rate of η = 0 . 01 , which gradually reduces to η = 0 . 001 through cosine annealing, was chosen for all combinations of datasets (CIFAR-10 / CIFAR-100) and the model (ResNet-18).

Table 4: Hyperparameter settings for CIFAR-10/CIFAR-100 with ResNet-18.

| Data augmentation Learning rate η   | RandomCrop, RandomHorizontalFlip, RandomErasing of PyTorch Grid search over { 0 . 1 , 0 . 01 , 0 . 001 }   |
|-------------------------------------|------------------------------------------------------------------------------------------------------------|

Graphs used in experiments. Several graphs used in the experiments are summarized as follows:

(C1) Ring graph. This is a static graph ( τ = 1 ) and each node connects with neighboring two nodes. The mixing matrix is given by

<!-- formula-not-decoded -->

where this holds k = 2 independently of n . Thus, we used this as a comparison graph for the configurations in the main paper configurations ( n, k ) = (15 , 2) , (17 , 2) , (30 , 2) .

(C1') Extended ring graph. This is a static graph ( τ = 1 ). To enable a fair comparison with other graphs that support k ≥ 2 connections, we extended the ring graph by connecting each node not only to its two immediate neighbors (i.e., k = 2 ) but also to additional adjacent nodes. For instance, when k = 3 , the mixing matrix is given by

<!-- formula-not-decoded -->

and when k = 4 , the mixing matrix is given by

<!-- formula-not-decoded -->

(C2) Base-( k +1) graph. This is a dynamic graph ( τ ≥ 2 ) and a related discussion is given in Sec. C.3. We computed the τ mixing matrix in the base-( k +1) graph using the source code on the website 7 .

(C3) Random graph. This is a static graph ( τ = 1 ), and is modeled as in the Erdos-Rényi random graph [5], adjusted to ensure k for each configuration.

(C4) ( k +1) -partite random match graph. This is a dynamic graph that is a generalization of the bipartite random match graph used as a comparison graph in [43], to accommodate k -peer communication. In the bipartite random match graph, two randomly selected nodes are connected with undirected edges, with the assumption that n is even. Unlike other graphs where connections follow a periodic pattern, the connections between two nodes vary over time without a fixed period, thus this graph does not maintain a consistent periodic interval τ . To generate the bipartite random match graph to allow any ( n, k ) configurations, i) the index ( 1 , 2 , . . . , n ) to ( ρ ( r ) (1) , ρ ( r ) (2) , . . . , ρ ( r ) ( n ) ) was randomly permuted, where ρ ( r ) ( · ) denotes the permutation function at communication round r , ii) the forming subgroups consisted of as many ( k +1) nodes as possible, and iii) partial average

7 https://github.com/yukiTakezawa/BaseGraph

within each subgroup after exchanging local parameters was performed. To the best of our knowledge, the expected consensus rate of the ( k +1) -partite random match graph, has not been investigated.

(P1) k -peer exponential graph. This is a dynamic graph, whose details are explained in Sec. 3 and Appendix A.

(P2) Null-cascade graph. This is a dynamic graph, whose details are explained in Sec. 4 and Appendix B.

Computing resource. We used computing servers employing 8 GPUs (NVIDIA RTX 6000 Ada (48 GB)) and 2 CPUs (AMD EPYC 9354, 3.25 GHz, 32-Core Processor).

Additional experimental results. Firstly, due to the space limitation, we picked several configurations in Fig. 4 in Sec. 5, its complete version is illustrated in Fig. 32. The corresponding highest test accuracy is summarized in Table 2.

Figure 32: Convergence curves using test accuracy of global parameters for three ( n, k ) -configurations with two datasets (CIFAR-10/CIFAR-100). This figure is the complete version of Fig. 4 in Sec. 5.

<!-- image -->

Next, additional ( n, k ) -configurations were tested in this appendix section, particularly using k ≥ 3 . We conducted five additional ( n, k ) -configurations; namely (d1) ( n, k ) = (15 , 3) , (e1) ( n, k ) = (17 , 3) , (f1) ( n, k ) = (30 , 3) , (g1) ( n, k ) = (30 , 4) , (h1) ( n, k ) = (31 , 4) using CIFAR-10 classification task. The corresponding results are summarized in Table 5 and Fig. 33. Even among the additional ( n, k ) -configurations with k ≥ 3 , the proposed (P2) null-cascade graph consistently illustrated strong performance, including cases where n is a large prime number. As discussed in the main paper, this can be attributed to the maximized expected spectral gap under uniform degree, although for some ( n, k ) -configurations, the periodic interval τ is increased. For ( n, k ) = (30 , 4) , we observed that the mixing matrices consisting of the base-( k +1) graph and the null-cascade graph are identical. As explained in Appendix C, this arises from the fact that the base-( k +1) graph for certain ( n, k ) configurations can be constructed following the design principle of the null-cascade

graph. Therefore, it is natural that the performance of the base-( k +1) graph and the null-cascade graph is nearly identical in this case. To further investigate this, we conducted additional experiments with ( n, k ) = (31 , 4) , where n is a large prime number. In this configuration, we observed that the null-cascade graph outperforms the base-( k +1) graph. These additional experimental results highlight the effectiveness of the null-cascade graph across various ( n, k ) -configurations.

Figure 33: Convergence curves using test accuracy of global parameters for five additional ( n, k ) -configurations.

| ( n,k ) -configuration            | (d1) ( n,k )=(15 , 3)   | (d1) ( n,k )=(15 , 3)   | (e1) ( n,k )=(17 , 3)   | (e1) ( n,k )=(17 , 3)   | (f1) ( n,k )=(30 , 3)   | (f1) ( n,k )=(30 , 3)   | (g1) ( n,k )=(30 , 4)   | (g1) ( n,k )=(30 , 4)   | (h1) ( n,k )=(31 , 4)   | (h1) ( n,k )=(31 , 4)   |
|-----------------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|
| Tested graphs                     | τ                       | test acc.               | τ                       | test acc.               | τ                       | test acc.               | τ                       | test acc.               | τ                       | test acc.               |
| (C1') Extended ring               | 1                       | 0 . 8676                | 1                       | 0 . 8756                | 1                       | 0 . 8605                | 1                       | 0 . 8544                | 1                       | 0 . 8958                |
| (C2) Base- ( k +1)                | 4                       | 0 . 8467                | 5                       | 0 . 8701                | 5                       | 0 . 8640                | 3                       | 0 . 8746                | 5                       | 0 . 8974                |
| (C3) Random                       | 1                       | 0 . 7560                | 1                       | 0 . 8749                | 1                       | 0 . 8747                | 1                       | 0 . 8237                | 1                       | 0 . 7576                |
| (C4) ( k +1) -partite rand. match | -                       | 0 . 8574                | -                       | 0 . 8760                | -                       | 0 . 8760                | -                       | 0 . 8788                | -                       | 0 . 8999                |
| (P1) k -peer exponential          | 2                       | 0 . 8532                | 2                       | 0 . 8789                | 3                       | 0 . 8713                | 2                       | 0 . 8780                | 2                       | 0 . 9031                |
| (P2) Null-cascade                 | 3                       | 0 . 8732                | 8                       | 0 . 8817                | 3                       | 0 . 8794                | 3                       | 0 . 8746                | 8                       | 0 . 9019                |

Table 5: Comparison of the highest test accuracy using global parameters for CIFAR-10 classification task for five additional ( n, k ) -configurations.

<!-- image -->

## Further additional tests using large n .

In the rebuttal phase, we performed additional experiments using large n . For ( n, k ) = (69 , 2) , data distribution of CIFAR-10 with α = 0 . 1 is illustrated as follow:

<!-- image -->

Node index (1-n)

Figure 34: Data distributions of CIFAR-10 training dataset when n = 69 .

The highest test accuracy and convergence curves are illustrated as follows:

Table 6: Comparison of the highest test accuracy using global parameters for CIFAR-10 classification task for ( n, k ) = (69 , 2) .

Figure 35: Convergence curves using test accuracy of global parameter for ( n, k ) = (69 , 2) .

| ( n,k ) -configuration            | ( n,k )=(69 , 2)   | ( n,k )=(69 , 2)   |
|-----------------------------------|--------------------|--------------------|
| Tested graphs                     | τ                  | test acc.          |
| (C1') Extended ring               | 1                  | 0 . 8805           |
| (C2) Base- ( k +1)                | 7                  | 0 . 9213           |
| (C3) Random                       | 1                  | 0 . 6267           |
| (C4) ( k +1) -partite rand. match | -                  | 0 . 9229           |
| (P1) k -peer exponential          | 4                  | 0 . 9225           |
| (P2) Null-cascade                 | 12                 | 0 . 9216           |

<!-- image -->

## H Related works

(i) Centralized learning. In distributed model parameter training, centralized learning such as Parallel SGD (PSGD), All-Reduce, and Federated Learning (FL) [23, 11, 10] often serve as the initial approach. FL, in particular, has seen significant efficiency gains due to client sampling by the center server and multiple local updates performed in each client. However, in scenarios involving extensive computing resources with many nodes/clients, a center server becomes a bottleneck due to the need to aggregate local model parameters from clients and distribute the global parameters back to them. To address this issue, research into decentralized learning approaches has been pursued, e.g., [16, 17].

(ii) Decentralized learning algorithms. The most widely used decentralized learning algorithm is DSGD [20, 26], and its extensions [16, 1, 2, 17]. Many researchers have integrated DSGD and well-known acceleration techniques, such as momentum [6, 18, 45, 46], communication compression techniques [9, 14, 21, 36, 39].

As illustrated in Theorem 1, the convergence rate of DSGD depends on data heterogeneity (Assumption 3). To mitigate this issue, many improved algorithms have been studied, such as Gradient Tracking (GT) and its extensions [3, 25, 30, 12, 33, 42, 48], D 2 [37], decentralized SCAFFOLD [19], primal-dual methods [47, 27, 28], etc.

In this paper, we primarily used DSGD as the fundamental decentralized learning algorithm. However, our proposed graphs are applicable to any decentralized learning algorithms.

(iii) Other graphs for decentralized learning. As illustrated in theoretical analysis for decentralized learning algorithms [13, 12, 15, 22, 40, 49, 33], network topology (graph) affects the convergence rate of many decentralized learning algorithms. Since the total communication costs are increased as the maximum degree k increases [32, 43], constructing graphs to increase/maximize the (expected) consensus rate while maintaining a small maximum degree k is essential for communication-efficient decentralized learning.

Several studies [32, 4, 40] have aimed to minimize communication costs by constraining the maximum degree of an underlying graph for communication-efficient decentralized learning. For instance, the ring graph, with its maximum degree k of only 2 , is simple yet communication-efficient. However, its consensus rate, p = O (1 /n 2 ) , as detailed in [24], suggests it performs poorly with large n . This observation aligns with the experimental findings presented in Sec. 5. When a larger k is available, the (static) exponential graph [1] may be a good option. Its consensus rate is discussed in Sec. 2.2; although k is dependent on n , limiting the flexibility of ( n, k ) combinations. In the recent study by [44], the BTPP utilizes two spanning trees (static graphs) as communication graphs. This study analytically demonstrates finite-time convergence through multiple repeats of communication. However, it is important to note that not all combinations of ( n, k ) are feasible with this method.

Nowadays, employing dynamic graphs for communication-efficient decentralized learning is a hot topic. We omit the discussion of works introduced in the main paper (e.g., [31, 43, 32, 4, 35]).

## I Limitations and future work

First, the k -peer exponential graph achieves finite-time convergence for limited n (a power of k +1 ). Despite this being a significant constraint, we calculated the expected consensus rate for the k -peer exponential graph for any ( n, k ) configuration, as detailed in Sec. 3 and illustrated in Appendix F. Figure 29 in Appendix F shows that the expected consensus rate increases when n deviates from conditions where n is a power of k +1 . Consequently, employing the k -peer exponential graph may be particularly advantageous for values of n that are a power of k +1 .

Second, it is necessary to have k ≥ 2 in the null-cascade graph, as discussed in Sec. 4. This requirement arises from setting pairwise nulls in a conjugate relationship, which results in the polynomial equation in (9), ensuring that SNFs are real numbers. In an era characterized by substantial computational resources spread across multi-location data centers, the availability of a large number of nodes n and a sufficiently large maximum degree k is feasible. Our null-cascade graph is designed to be well-suited for this future scenario.

Third, the periodic interval τ required for achieving finite-time convergence in the null-cascade graph is not always small, as discussed in Sec. 4. Including large prime numbers ν i in the factorization of n increases in the periodic interval τ needed for finite-time convergence. Nevertheless, its impact would not be significant for the following reasons: i) uniform degree: the null-cascade graph consists of circulant matrices, resulting in a uniform degree such that every node has the same number of connections. Although the balanced mixing due to this is empirically known to influence outcomes, it has not yet been considered in the advanced convergence analysis (e.g., [13]); addressing this remains work for the future. The additional reason, ii) the possibility of redundant counting of τ , can be found in Appendix F. Moreover, experimental results in Sec. 5 and Appendix G demonstrated that the null-cascade graph remains effective even when large prime numbers ν i are included in the factorization of n .

Fourth, we report convergence curves with respect to communication rounds rather than run-time. This choice reflects our compute constraints-eight GPUs on a single server without NVLink-requiring us to simulate multiple nodes per GPU (e.g., for n =30 , 3 or 4 nodes/GPU). Under this setup, accurately and fairly assessing communication overheads is difficult, and these costs grow especially large as the number of neighbors k increases.

Another promising direction for future work involves leveraging non-zero spatial frequency components for personalized model training. Although this study focuses on training the global parameter (DC component), non-zero frequency components also contain valuable information that could significantly enhance model personalization, thus presenting a valuable avenue for future exploration.

## J Impact statement

We present dynamic graphs for communication-efficient decentralized learning, which can be applied for training large-scale models (e.g., Large Language Models: LLMs) across extensive distributed computing resources, such as data centers. A potential risk of this technology is that it could enable a broader range of organizations to train large-scale models, which were previously restricted to organizations with access to substantial computing resources.