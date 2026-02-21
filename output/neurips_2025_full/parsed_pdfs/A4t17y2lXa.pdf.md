## Robust Distributed Estimation: Extending Gossip Algorithms to Ranking and Trimmed Means

## Anna van Elst Igor Colin Stephan Clémençon

LTCI, Télécom Paris, Institut Polytechnique de Paris

{anna.vanelst, igor.colin, stephan.clemencon}@telecom-paris.fr

## Abstract

This paper addresses the problem of robust estimation in gossip algorithms over arbitrary communication graphs. Gossip algorithms are fully decentralized, relying only on local neighbor-to-neighbor communication, making them well-suited for situations where communication is constrained. A fundamental challenge in existing mean-based gossip algorithms is their vulnerability to malicious or corrupted nodes. In this paper, we show that an outlier-robust mean can be computed by globally estimating a robust statistic. More specifically, we propose a novel gossip algorithm for rank estimation, referred to as GORANK, and leverage it to design a gossip procedure dedicated to trimmed mean estimation, coined GOTRIM. In addition to a detailed description of the proposed methods, a key contribution of our work is a precise convergence analysis: we establish an O (1 /t ) rate for rank estimation and an O (1 /t ) rate for trimmed mean estimation, where by t is meant the number of iterations. Moreover, we provide a breakdown point analysis of GOTRIM. We empirically validate our theoretical results through experiments on diverse network topologies, data distributions and contamination schemes.

## 1 Introduction

Distributed learning has gained significant attention in machine learning applications where data is naturally distributed across a network, either due to resource or communication constraints [7]. The rapid development of the Internet of Things has further amplified this trend, as the number of connected devices continues to grow, producing large volumes of data at the edge of the network. As a result, edge computing, where computation is performed closer to the data source, has emerged as a viable alternative to conventional cloud computing [6, 36]. By reducing reliance on centralized servers, edge computing offers several advantages, including lower energy consumption, improved data security, and reduced latency [6, 36]. In this context, several distributed learning frameworks, including Gossip Learning and Federated Learning, have been developed. Federated Learning depends on a central server for model aggregation [26], which introduces challenges such as communication bottlenecks and a single point of failure. In contrast, Gossip Learning provides a fully decentralized alternative where nodes communicate only with their nearby neighbors [18]. In addition, Gossip learning is particularly well-suited for situations where communication is constrained, such as in peer-to-peer or sensor networks.

One of the central challenges in distributed learning is robustness to corrupted nodes-scenarios where some nodes may contain contaminated data due to hardware faults, or even adversarial attacks [8, 39]. Consider a network of sensors deployed to monitor the temperature in a small region, as described in [5]. In idealized scenarios, sensor noise is typically modeled by a zeromean Gaussian. However, a recent survey (see [1]) mentions that sensor networks are especially prone to outliers due to their reliance on imperfect sensing devices. Indeed, in practice, sensors can malfunction (e.g., become miscalibrated, get stuck at constant values, or report extreme values

Figure 1: Illustration of a GOTRIM's iteration paired with GORANK on a communication graph G = ( V, E ) . Initially, each node k ∈ V stores the observation X k .

<!-- image -->

due to environmental interference). These issues introduce outliers in the dataset which can corrupt the estimated temperature. Specifically, we consider a setting in which a fraction of the nodes hold outlying data [14]. In Federated Learning, significant progress has been made in designing robust aggregation methods based on statistics, such as the median and trimmed mean [2, 3, 39]. These statistics are known to be more robust to outliers than the standard mean [19]. However, ensuring robustness in Gossip Learning remains challenging. Unlike Federated Learning, where a central server can enforce robust aggregation rules, gossip algorithms inherently rely on local communication. To the best of our knowledge, approaches relying on globally estimating robust statistics remain largely unexplored in the gossip literature.

In this paper, we address the problem of robust estimation in gossip algorithms over arbitrary communication graphs. Specifically, we show that an outlier-robust mean can be computed by globally estimating a robust statistic. Ranks play a crucial role in the framework we develop here, as they allow us to identify outliers and compute robust statistics on the one hand [30], and because they can be obtained by means of pairwise comparisons on the other. This allows us to exploit the gossip approach proposed in [10] to compute pairwise averages ( U -statistics) in a decentralized way. Note that both ranks and trimmed means are inherently global statistics, which makes them more challenging to estimate in a decentralized manner compared to statistics like averages, minima, or maxima that rely on local conservation principles.

Our main contributions are summarized as follows:

- We propose GORANK, a new gossip algorithm for rank estimation, and establish the first theoretical convergence rates for gossip-based ranking on arbitrary communication graphs, proving an O (1 /t ) convergence rate, where t ≥ 1 denotes the number of iterations.
- We introduce GOTRIM, the first gossip algorithm for trimmed mean estimation which can be paired with any ranking algorithm. GOTRIM does not rely on strong graph topology assumptions-a common restriction in robust gossip algorithms. We prove a competitive convergence rate of order O (1 /t ) and provide a breakdown point analysis, demonstrating the robustness of its estimate.
- Finally, we conduct extensive experiments on various contaminated data distributions and network topologies. Numerical results empirically show that (1) GORANK consistently outperforms previous work in large and poorly connected communication graphs, and (2) GOTRIM effectively handles outliers-its estimate quickly improves on the naive mean and converges to the true trimmed mean.

This paper is organized as follows. Section 2 introduces the problem setup and reviews the related works. In Section 3, we present our new gossip algorithm for ranking, along with its convergence analysis and supporting numerical experiments. Section 4 introduces our novel gossip algorithm for trimmed mean estimation, together with a corresponding convergence analysis and experimental results. Finally, Section 5 explores potential extensions of our work. Due to space constraints, technical details, further discussions, and results are deferred to the Supplementary Material.

## 2 Background and Preliminaries

This section briefly introduces the concepts of decentralized learning, describes the problem studied and the framework for its analysis.

## 2.1 Problem Formulation and Framework

Here, we formulate the problem using a rigorous framework and introduce the necessary notations.

Notation. Let n ≥ 1 . We denote scalars by normal lowercase letters x ∈ R , vectors (identified as column vectors) by boldface lowercase letters x ∈ R n , and matrices by boldface uppercase letters X ∈ R n × n . The set { 1 , . . . , n } is denoted by [ n ] , R n 's canonical basis by { e k : k ∈ [ n ] } , the indicator function of any event A by I A , the transpose of any matrix M by M ⊤ and the cardinality of any finite set F by | F | . By I n is meant the identity matrix in R n × n , by 1 n = (1 , . . . , 1) ⊤ the vector in R n whose coordinates are all equal to one, by ∥ · ∥ the usual ℓ 2 norm, by ⌊·⌋ the floor function, and by A ⊙ B the Hadamard product of matrices A and B . We model a network of size n &gt; 0 as an undirected graph G = ( V, E ) , where V = [ n ] denotes the set of vertices and E ⊆ V × V the set of edges. We denote by A its adjacency matrix, meaning that for all ( i, j ) ∈ V 2 , [ A ] ij = 1 iff ( i, j ) ∈ E , and by D the diagonal matrix of vertex degrees. The graph Laplacian of G is defined as L = D -A .

̸

Setup. We consider a decentralized setting where n ≥ 2 real-valued observations X 1 , . . . , X n are distributed over a communication network represented by a connected and non-bipartite graph G = ( V, E ) , see [12]: the observation X k is assigned to node k ∈ [ n ] . For simplicity, we assume no ties: for all k = l , X k = X l . Communication between nodes occurs in a stochastic and pairwise manner: at each iteration, an edge of the communication graph G is chosen uniformly at random, allowing the corresponding neighboring nodes to exchange information. This popular setup is robust to network changes and helps reduce communication overhead and network congestion [5, 10, 13]. We focus on the synchronous gossip setting, where nodes have access to a global clock and synchronize their updates [5, 10, 13]. We assume that a fraction 0 &lt; ε &lt; 1 / 2 of the data is corrupted and may contain outliers, as stipulated in Huber's contamination model, refer to [19].

̸

The Decentralized Estimation Problem. The goal pursued here is to develop a robust gossip algorithm that accurately estimates the mean over the network despite the presence of outliers, specifically the α -trimmed mean with α ∈ (0 , 1 / 2) , i.e., the average of the middle (1 -2 α ) -th fraction of the observations. This statistic, discarding the observations of greater or smaller rank, is a widely used location estimator when data contamination is suspected; see [31, 34]. Formally, let X n (1) ≤ X n (2) ≤ · · · ≤ X n ( n ) be the order statistics ( i.e., the observations sorted in ascending order), and define m = ⌊ αn ⌋ for α ∈ (0 , 1 / 2) . The α -trimmed mean is given by:

<!-- formula-not-decoded -->

Based on the rank of each node's observation X k , namely r k = 1 + ∑ n l =1 I { X k &gt;X l } in the absence of ties, for k ∈ [ n ] , it can also be formulated as a weighted average of the observations, just like many other robust statistics:

<!-- formula-not-decoded -->

where w n,α is a weight function defined as w n,α ( r k ) = ( n/ ( n -2 m )) I { r k ∈ I n,α } , with the inclusion interval given by I n,α = [ u, v ] where u = m +1 and v = n -m .

Remark 1. The framework can be extended to the case of ℓ ≥ 2 tied observations with the mid-rank method: the rank assigned to the ℓ tied values is the average of the ranks they would have obtained in absence of ties, that is, the average of p +1 , p +2 , . . . , p + ℓ , which equals p +( ℓ +1) / 2 , see [24]. To account for the possibility of non-integer rank estimates, one may use the adjusted inclusion interval to define the weights in (2) : I n,α = [ u -1 / 2 , v +1 / 2] . Note that when ranks are integers, this adjustment has no effect.

Remark 2. Our setup assumes honest nodes, meaning they perform updates correctly and consistently based on their local observations. However, under Huber's contamination model, these observations may include outliers. Note that this setup is fundamentally different from the Byzantine model where nodes may behave arbitrarily or maliciously, potentially sending incorrect updates with the intent to disrupt consensus or degrade performance. This model is still realistic in many practical settings: a sensor could be miscalibrated, stuck at a fixed value, or may consistently report incorrect readings due to environmental factors, without necessarily being attacked.

## 2.2 Related Works - State of the Art

The overview of related literature below highlights the novelty of the gossip problem analyzed here.

Distributed Ranking. Distributed ranking (or ordering) is considered in [9], where a gossip algorithm for estimating ranks on any communication graph is proposed. The algorithm is proved to converge, in the sense of yielding the correct rank estimate, in finite time with probability one. However, their work does not provide any convergence rate bound, and empirical results suggest that the algorithm is suboptimal in scenarios with long return times, i.e., when the expected time for a random walk on the graph to return to its starting node is long (see the Supplementary Material). Alternatively, here we take advantage of the fact that ranks can be calculated using pairwise comparisons I { X k &gt;X l } , so as to build on the GoSta approach in [10], originally introduced for the distributed estimation of U -statistics, with a proved convergence rate bound of order O (1 /t ) .

Robust Mean Estimation. To the best of our knowledge, the estimation of α -trimmed means has not yet been explored in the gossip literature. Several related works examine the estimation of medians and quantiles in sensor networks [15, 23, 33]. A key limitation of these works is their reliance on a special node, such as a base station or leader, which initiates queries, broadcasts information, and collects data from other sensors-an assumption that does not apply to our setting [15, 23, 33]. Another work has proposed an algorithm for estimating quantiles [16]; however, there seems to be no guarantees of convergence to the true quantiles. Moreover, their algorithm assumes a fully connected communication graph and the ability to sample from four nodes at each step and to sample K random nodes at the end of the protocol, whereas we consider the more challenging setting of arbitrary communication graphs and pairwise communication. Recently, He et al. [17] proposed a novel local aggregator, ClippedGossip , for Byzantine-robust gossip learning. In our pairwise setup with a fixed clipping radius, their approach-though reasonable for robust optimization-does not work for robust estimation. Specifically, ClippedGossip ultimately converges to the corrupted mean and, therefore, fails to reduce the impact of outliers. A detailed analysis is provided in the Supplementary Material.

The following table assesses whether each method is fully decentralized, whether the estimator is unbiased, and whether theoretical convergence rates exist (for both complete and arbitrary graphs).

| Method                                                                   | Decentralized?   | Unbiased?   | Rates on: Any Graph? Complete Graph?                                                                                                                     |
|--------------------------------------------------------------------------|------------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Chiuso et al. (Baseline) Baseline++ (ours) GoRank (ours) Haeupler et al. | × × ✓ ×          | ✓ ✓ ✓ ∼     | Complete Graph: O (exp( - t/ &#124; E &#124; )) Complete Graph: O (exp( - t/ &#124; E &#124; )) Any Graph: O (1 /t ) Complete Graph: O (log( n )) rounds |

## 3 A Gossip Algorithm for Distributed Ranking - GORANK

In this section, we introduce and analyze the GORANK algorithm. We establish that the expected estimates converge to the true ranks at a O (1 /ct ) rate, where the constant c &gt; 0 (given in Theorem 1 below) quantifies the degree of connectivity of G : the more connected the graph, the greater this quantity and the smaller the rate bound. We also prove that the expected absolute error decreases at a rate of O (1 / √ ct ) . In addition, we empirically validate these results with experiments involving graphs of different types, showing that the observed convergence aligns with the theoretical bounds.

## 3.1 Algorithm - Convergence Analysis

We introduce GORANK, a gossip algorithm for estimating the ranks of the observations distributed on the network, see Algorithm 1. It builds on GOSTA, an algorithm originally designed for estimating pairwise averages ( U -statistics of degree 2 ) proposed and analyzed in [10]. The GORANK algorithm exploits the fact that ranks can be computed by means of pairwise comparisons:

<!-- formula-not-decoded -->

| Algorithm 1 GoRank: a synchronous gossip algorithm for ranking.   | Algorithm 1 GoRank: a synchronous gossip algorithm for ranking.                                                            |
|-------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| 1:                                                                | Init: For each k ∈ [ n ] , initialize Y k ← X k and R ′ k ← 0 . // init(k)                                                 |
| 2:                                                                | for s = 1 , 2 , . . . do for k = 1 , . . .,n do ′                                                                          |
| 3:                                                                |                                                                                                                            |
| 4:                                                                | Update estimate: R k ← (1 - 1 /s ) R ′ k +(1 /s ) I { X k >Y k } . Update rank estimate: R k ← nR ′ k +1 . // update(k, s) |
| 5:                                                                |                                                                                                                            |
| 6:                                                                | end for                                                                                                                    |
| 7:                                                                | Draw ( i, j ) ∈ E uniformly at random.                                                                                     |
| 8:                                                                | Swap auxiliary observation: Y i ↔ Y j . // swap(i, j)                                                                      |
| 9:                                                                | end for                                                                                                                    |
| 10:                                                               | Output: Estimate of ranks R k .                                                                                            |

Let R k ( t ) and R ′ k ( t ) denote the local estimates of r k and r ′ k respectively at node k ∈ [ n ] and iteration t ≥ 1 . Each node maintains an auxiliary observation, denoted Y k ( t ) , which enables the propagation of observations across the network, despite communication constraints. For each node k , the variables are initialized as Y k (0) = X k and R ′ k (0) = 0 . At each iteration t ≥ 1 , node k updates its estimate R ′ k ( t ) by computing the running average of R ′ k ( t -1) and I { X k &gt;Y k ( t -1) } . The rank estimate is then computed as R k ( t ) = nR ′ k ( t ) + 1 . Next, an edge ( i, j ) ∈ E is selected uniformly at random, and the corresponding nodes exchange their auxiliary observations: Y i ( t ) = Y j ( t -1) and Y j ( t ) = Y i ( t -1) . This random swapping procedure allows each observation to perform a random walk on the graph, which is described by the permutation matrix W 1 ( t ) = I n -( e i -e j )( e i -e j ) ⊤ , which plays a key role in the convergence analysis [10]. By taking the expectation with respect to the edge sampling process, we obtain W 1 := E [ W 1 ( t )] = I n -(1 / | E | ) L . This stochastic matrix W 1 shares similarities with the transition matrix used in gossip averaging [5, 25, 32]: it is symmetric and doubly stochastic. Consequently, W 1 has eigenvector 1 n with eigenvalue 1 , resulting in a random walk with uniform stationary distribution. In other words, after sufficient iterations ( i.e., in a nearly stationary regime), each observation has (approximately) an equal probability of being located at any given node-a property that would naturally hold (exactly) without swapping if the communication graph were complete. In addition, it can be shown that, if the graph is connected and non-bipartite, the spectral gap of W 1 satisfies 0 &lt; c &lt; 1 and is given by c = λ 2 / | E | where λ 2 is the second smallest eigenvalue (or spectral gap) of the Laplacian [10]. This spectral gap plays a crucial role in the mixing time of the random walk, reflecting how quickly it (geometrically) converges to its stationary distribution [32]. In fact, the spectral gap of the Laplacian is also known as the graph's algebraic connectivity, a larger spectral gap meaning a higher graph connectivity [27].

Remark 3. We assume that the network size n ≥ 2 is known to the nodes. If not, it can be easily estimated by injecting a value +1 into the network, where all initial estimates are set to 0 , and applying the standard gossip algorithm for averaging [5], see the Supplementary Material. The quantities computed for each node will then converge to 1 /n .

Remark 4. The asynchronous extension of GORANK is straightforward: it replaces the global iteration counter s with a local counter C k maintained at each node k . The update rule then becomes (1 -1 /C k ) R ′ k +(1 /C k ) I X k &gt;Y k . Empirically, we find that Asynchronous GORANK converges slightly faster and is more efficient than its synchronous counterpart. Due to space constraints, the detailed algorithm and experimental results are deferred to Appendix H.

We now establish convergence results for GORANK, by adapting the analysis in [10], originally proposed to derive convergence rate bounds for GOSTA as follows. For k ∈ [ n ] , set h k = ( I { X k &gt;X 1 } , . . . , I { X k &gt;X n } ) ⊤ and observe that the true rank of observation X k is given by r k = h ⊤ k 1 n +1 . At iteration t = 1 , the auxiliary observation has not yet been swapped, so the expected estimate is updated as E [ R ′ k (1)] = h ⊤ k e k . At the end of the iteration, the auxiliary observation is randomly swapped, yielding the update: E [ R ′ k (2)] = (1 / 2) E [ R ′ k (1)]+(1 / 2) h ⊤ k W 1 e k , where E [ · ] denotes the expectation taken over the edge sampling process. Using recursion, the evolution of the estimates, for any t ≥ 1 and k ∈ [ n ] , is given by

<!-- formula-not-decoded -->

Note that E [ R ′ k ( t )] can be viewed as the average of t terms of the form I { X k &gt;X l } , where X l is picked at random. Observe also that R ( t ) = ( R 1 ( t ) , . . . , R n ( t )) is not a permutation of [ n ] in general: in particular, we initially have R k (1) = 1 for all k ∈ [ n ] . We now state a convergence result for GORANK, which claims that the expected estimates converge to the true ranks at a rate of O (1 /ct ) .

Theorem 1 (Convergence of Expected GORANK Estimates) . We have: ∀ k ∈ [ n ] , ∀ t ≥ 1 ,

<!-- formula-not-decoded -->

where the constant c = λ 2 / | E | represents the connectivity of the graph, with λ 2 being the spectral gap of the graph Laplacian, and the rank functional σ k = n 3 / 2 · ϕ (( r k -1) /n ) is determined by the score generating function ϕ : u ∈ (0 , 1) → √ u (1 -u ) .

More details, as well as the technical proof, are deferred to section C of the Supplementary Material. Theorem 1 establishes that, for each node, the estimates converge in expectation to the true ranks at O (1 /ct ) rate, where c is a constant depending on the graph's algebraic connectivity: higher network connectivity leads to faster convergence. In addition, the shape of the function ϕ involved in the bound (1), resp. of u ∈ (0 , 1) ↦→ √ u (1 -u ) , suggests that extreme ( i.e., the lowest and largest) values are intrinsically easier to rank than middle values, see also Fig. 2. Regarding its shape, observe that GORANK can be seen as an algorithm that estimates Bernoulli parameters (1 /n ) ∑ n l =1 I { X k &gt;X l } = ( r k -1) /n and, from this perspective, the σ k 's can be viewed as the related standard deviations, up to the factor n 3 / 2 .

We state an additional result below that builds upon the previous analysis, and establishes a bound of order O (1 / √ ct ) for the expected absolute deviation.

Theorem 2 (Expected Gap) . Let k ∈ [ n ] , and let c and σ k be as defined in Theorem 1. For all t ≥ 1 , we have: E [ | R k ( t ) -r k | 2 ] ≤ O (1 /ct ) · σ 2 k . Consequently,

<!-- formula-not-decoded -->

Refer to section C in the Supplementary Material for the technical proof. Theorem 2 shows that the expected error in absolute deviation decreases at a rate of O (1 / √ ct ) , similar to the convergence rate for the expectation E [ R ( t )] , but with a square root dependence. In addition, Theorem 2 can be combined with Markov's inequality to derive high-probability bounds on the ranking error. As will be shown in the next section, this is a key component in the convergence analysis of GOTRIM.

Building upon the previous analysis, we derive another gossip algorithm called Baseline++ , an improved variant of the one proposed by Chiuso et al. for decentralized rank estimation [9]. It is described at length in section A of the Supplementary Material. The algorithm's steps and propagation closely resemble those of GORANK. However, instead of estimating the means (3) based on pairwise comparisons directly, Baseline++ aims to minimize, for each node k ∈ [ n ] , a specific ranking loss function, namely the Kendall τ distance ϕ k ( X , R ) = ∑ n l =1 I { ( X k -X l ) · ( R k -R l ) &lt; 0 } , counting the number of discordant pairs among (( R k , X k ) , ( R l , X l )) with l = 1 , . . . , n . While GORANK provides a quick approximation of the ranks, especially in poorly connected graphs, Baseline++ , despite being slower at first, may ultimately achieve a lower overall error as it minimizes a discrete loss function. However, GORANK has a practical advantage: it does not require an initial ranking, which may not always be available in real-world settings. Moreover, it comes with convergence rate guarantees that apply to any graph topology. In contrast, analyzing the convergence of Baseline++ on arbitrary communication graphs is challenging and is left for future work.

## 3.2 Numerical Experiments

Setup. We conduct experiments on a dataset S = { 1 , . . . , n } with n = 500 , distributed across nodes of a communication graph. Our evaluation metric is the normalized absolute error between estimated and true ranks, i.e., for node k at iteration t , the error is defined as ℓ k ( t ) = | R k ( t ) -r k | /n . While more sophisticated ranking metrics such as Kendall τ exist, they are not relevant in our context, as we focus on individual rank estimation accuracy. We first examine the impact of the constants appearing in our theoretical bounds (see Theorem 1), the one related to graph connectivity and the one reflecting rank centrality. To this end, we consider three graph topologies: the complete

graph ( c = 4 . 01 × 10 -3 ), in which every node is directly connected to all others, yielding maximal connectivity; the two-dimensional grid ( c = 1 . 65 × 10 -5 ), where each node connects to its four immediate neighbors; and the Watts-Strogatz network ( c = 3 . 31 × 10 -4 ), a randomized graph with average degree k = 4 and rewiring probability p = 0 . 2 , offering intermediate connectivity between the complete and grid graphs. Then, we compare GORANK with the two other ranking algorithms Baseline and Baseline++ . Figure (a) was generated on a Watts-Strogatz graph, averaged over 1 e 3 trials. Figure (b) and (c) show the convergence of ranking algorithms with mean and standard deviation computed from 100 trials: figure (b) shows the convergence of GORANK on different graph topologies; figure (c) compares the convergence of the ranking algorithms on a 2D Grid graph. All experiments were run on a single CPU with 32 GB of memory for 8 e 4 iterations, with a total execution time of approximately two hours. The code for our experiments is publicly available. 1

Figure 2: Illustration of the behavior of GORANK: (a) shows how the absolute error of the rank estimates of GORANK aligns with the shape of the function ϕ , and highlights the role of the constant σ k for k ∈ [ n ] in the error bound; (b) compares the convergence rate of GORANK across different graph topologies ( i.e., levels of connectivity), illustrating the influence of the constant c in the bound; (c) compares GORANK with existing method ( Baseline ) and an alternative ranking algorithm developed in this work ( Baseline++ ).

<!-- image -->

Results. Together, the figures illustrate key properties of GORANK. Figure (a) empirically confirms that extreme ranks are easier to estimate than those in the middle-consistent with the shape of the theoretical bound. Figure (b) highlights the impact of graph topology: as connectivity decreases, convergence slows, in line with the theoretical bounds. Finally, Figure (c) compares the different methods. GORANK and Baseline++ provide fast rank approximations even for large, poorly connected graphs ( e.g., a 2D grid graph). Baseline++ appears to converge faster in the end, as it directly optimizes a discrete loss. Baseline , on the other hand, converges more slowly throughout, which aligns with its dependence on the return time ( i.e., the expected time for a random walk to revisit its starting node), a quantity known to be large for poorly connected graphs. More extensive experiments and discussion, confirming the results above, can be found in the Supplementary Material.

## 4 GOTRIM - A Gossip Algorithm for Trimmed Means Estimation

In this section, we present GOTRIM, a gossip algorithm for trimmed means estimation. We establish a convergence in O (1 /t ) with constants depending on the network and data distribution.

## 4.1 Algorithm - Convergence Analysis

We introduce GOTRIM, a gossip algorithm to estimate trimmed mean statistics (see Algorithm 2), which dynamically computes a weighted average using current estimated ranks. Notably, GOTRIM can be paired with any ranking algorithm, including GORANK consequently. Let Z k ( t ) and W k ( t ) denote the local estimates of the statistic and weight at node k and iteration t . First, by Equation (2), the α -trimmed mean can be computed via standard gossip averaging: for all k ∈ [ n ] , Z k ( t ) = (1 /n ) ∑ n l =1 W l ( t ) · X l , where W l ( t ) = w n,α ( R l ( t )) with w n,α ( · ) defined in Section 2. Secondly,

1 The code is available at github.com/anna-vanelst/robust-gossip.

## Algorithm 2 GoTrim: a synchronous gossip algorithm for estimating α -trimmed means.

- 1: Input: Trimming level α ∈ (0 , 1 / 2) , function w n,α defined in 2 and choice of ranking algorithm rank ( e.g., GoRank).
- 2: Init: For all node k , set Z k ← 0 , W k ← 0 and R k ← rank . init ( k ) .
- 3: for s = 1 , 2 , . . . do
- 4: for k = 1 , . . . , n do
- 5: Update rank: R k ← rank.update ( k, s ) .
- 7: Set Z k ← Z k +( W ′ k -W k ) · X k .
- 6: Set W ′ k ← w n,α ( R k ) .
- 8: Set W k ← W ′ k .
- 9: end for
- 10: Draw ( i, j ) ∈ E uniformly at random.
- 11: Set Z i , Z j ← ( Z i + Z j ) / 2 .
- 12: Swap auxiliary variables: swap(i, j)
- 13: end for
- 14: Output: Estimate of trimmed mean Z k .

since ranks R k ( t ) vary over iterations, the algorithm dynamically adjusts to correct past errors: at each step, it compensates by injecting ( W k ( t ) -W k ( t -1)) · X k into the averaging process. On the one hand, we have an averaging operation which is captured by the averaging matrix: W 2 ( t ) = I n -( e i -e j )( e i -e j ) ⊤ / 2 . Similarly to the permutation matrix (see the previous section), the expectation of the averaging matrix is symmetric, doubly stochastic and has spectral gap that satisfies 0 &lt; c 2 &lt; 1 and is given by c 2 = c/ 2 [5, 32]. On the other hand, we have a non-linear operation that depends on the estimated ranks: at each iteration t &gt; 0 , each node k is updated as Z k ( t ) = Z k ( t -1) + δ k ( t ) · X k , where δ k ( t ) = W k ( t ) -W k ( t -1) . Hence, the evolution of the estimates can be expressed as Z ( t ) = W 2 ( t ) ( Z ( t -1) + δ ( t ) ⊙ X ) , where Z ( t ) = ( Z 1 ( t ) , . . . , Z n ( t )) and δ ( t ) = ( δ 1 ( t ) , . . . , δ n ( t )) . Taking the expectation over the sampling process, the expected estimates are given by: E [ Z ( t )] = W 2 ( E [ Z ( t -1)] + ∆ w ( t ) ⊙ X ) with W 2 = I n -(1 / 2 | E | ) L and ∆ w ( t ) = E [ δ ( t )] . For t = 1 , since Z k (0) = 0 , we have E [ Z (1)] = W 2 ∆ w (1) ⊙ X . Recursively, for any t &gt; 0 ,

<!-- formula-not-decoded -->

We first state a lemma that claims that W k ( t ) converges in expectation to w n,α ( r k ) .

Lemma 1 (Convergence in Expectation of W k ( t ) ) . Let R ( t ) and W ( t ) be defined as in Algorithm 1 and Algorithm 2, respectively. For all k ∈ [ n ] and t &gt; 0 , we have:

<!-- formula-not-decoded -->

where γ k = min( | r k -a | , | r k -b | ) ≥ 1 / 2 with a = ⌊ αn ⌋ +1 / 2 and b = n -⌊ αn ⌋ +1 / 2 being the endpoints of interval I n,α . The constants c and σ k are those defined in Theorem 2.

Further details and the complete proof are provided in Section D of the Supplementary Material. Lemma 1 establishes that, for each node k , the estimates of the weight W k converge in expectation to the true weight w n,α ( r k ) at a rate of O ( 1 /γ 2 k ct ) . In addition, this lemma suggests that points closer to the interval endpoints are subject to larger errors, as reflected in the constant γ k , which is consistent with the intuition that these points require greater precision in estimating ranks.

Having established the convergence of the weights W k ( t ) , we now focus on the convergence of the estimates Z k ( t ) . The following theorem demonstrates the convergence in expectation of GOTRIM when paired with the GORANK ranking algorithm.

Theorem 3 (Convergence in Expectation of GOTRIM) . Let Z ( t ) be defined as in Algorithm 2, and assume the ranking algorithm is Algorithm 1. Then, for any t &gt; T ∗ = min { t &gt; 1 | ct &gt; 2 log( t ) } ,

<!-- formula-not-decoded -->

where K = ( σ 2 1 /γ 2 1 , . . . , σ 2 n /γ 2 n ) and c , σ k , γ k are the constants defined in Lemma 1.

See Section D of the Supplementary Material for the detailed proof. Theorem 3 shows that, for each node k , the estimate of the trimmed mean Z k converges in expectation to the true trimmed mean ¯ x α at a rate of O ( 1 /c 2 t ) . While the presence of the c 2 term may appear pessimistic, empirical evidence suggests that the actual convergence may be faster in practice. The vector K acts as a rank-dependent mask over X , modulating the contribution of each data point X k to the error bound. Specifically: (1) extreme values are more heavily penalized by the mask, as they come with better rank estimation; (2) values with ranks near the trimming interval endpoints are amplified, as small inaccuracies in rank estimation can lead to disproportionately larger errors.

## 4.2 Robustness Analysis - Breakdown Points

Consider a dataset S of size n ≥ 1 and T n = T n ( S ) a real-valued statistic based on it. The breakdown point for the statistic T n ( S ) is defined as ε ∗ = p ∞ /n , where p ∞ = min { p ∈ N ∗ : sup S ′ p | T n ( S ) -T n ( S ′ p ) | = ∞} , the supremum being taken over all corrupted datasets S ′ p obtained by replacing p samples in S by arbitrary samples. The breakdown point is a popular notion of robustness [20], corresponding here to the fraction of samples that need to be corrupted to make the statistic T n arbitrarily large ( i.e., "break down"). For example, the breakdown point of the α -trimmed mean is given by ⌊ αn ⌋ /n , which is approximately α . We consider here a generalization of this notion, namely the τ -breakdown point by replacing p ∞ with p τ = min { p ∈ N ∗ , sup S ′ p | T n ( S ) -T n ( S ′ p ) | ≥ τ } , where τ &gt; 0 is a threshold parameter. Since our algorithm does not compute the exact α -trimmed mean, we focus instead on determining the τ -breakdown point ε ∗ k ( t ) of the partial α -trimmed mean at iteration t for each node k , i.e., the estimate Z k ( t ) . Given that this quantity was previously shown to converge to the α -trimmed mean, we expect that ε ∗ k ( t ) ≤ α . The following theorem provides framing bounds for the breakdown point of the estimates of GOTRIM when paired with GORANK.

Theorem 4. Let τ &gt; 0 and δ, α ∈ (0 , 1) . With probability at least 1 -δ , the τ -breakdown point ε ∗ k ( t ) of the partial α -trimmed mean at iteration t &gt; T for any node k satisfies

<!-- formula-not-decoded -->

where K ( δ ) = O (1 /δ ) is a constant and T = O (log(1 /τδ )) represents the time allowed for the mean to propagate.

See Section E of the Supplementary Material for the technical proof, which relies on the idea that, when estimating a partial α -trimmed mean, there are two sources of error: (1) the uncertainty in rank estimation, which can lead to incorrect data points being included in the mean, and (2) the delay from the gossip averaging, which requires a certain propagation time T to update the network estimates. Note that, as t →∞ , Theorem 4 recovers the breakdown point of the exact α -trimmed mean.

## 4.3 Numerical Experiments

Setup. The experimental setup is identical to that of the previous section, with the key difference being the introduction of corrupted data. Specifically, the dataset S is contaminated by replacing a fraction ε = 0 . 1 of the values with outliers. We consider two types of corruption, each affecting ⌊ εn ⌋ randomly selected data points: (a) scaling, where a value x is changed to sx , and (b) shifting, where x becomes x + s . While this is a relatively simple form of corruption, it is sufficient to break down the classical mean. We measure performance using the absolute error between the estimated and true trimmed mean. For node k at iteration t , the error is given by ℓ ( t ) = (1 /n ) ∑ k | Z k ( t ) -¯ x α | . Experiments (a) and (b) are run on dataset S , corrupted with scaling s = 10 , using a Watts-Strogatz and a 2D grid graph, respectively. Experiment (c) uses the Basel Luftklima dataset, corrupted with shift s = 100 . This dataset includes temperature measurements from n = 105 sensors across Basel. A graph with connectivity c = 4 . 7 × 10 -4 is constructed by connecting sensors within 1 km of each other. The code and dataset for our experiments is publicly available. 2

Results. Figure (a) empirically confirms that the uncertainty in the weight estimates is highest near the boundaries of the interval, consistent with our theoretical bound. Figures (b) and (c) show that GOTRIM, when combined with GORANK or even alternative ranking methods, quickly approximates

2 The code and Basel Luktklima dataset are available at github.com/anna-vanelst/robust-gossip.

<!-- image -->

k

k

Figure 3: Convergence behavior of GOTRIM in combination with different ranking algorithms. Figure (a) illustrates how the constant in the bound reflects the error of the weight estimate of GOTRIM. Figures (b) and (c) demonstrate that, for α = 0 . 2 and ε = 1 , GOTRIM quickly improves on the naive corrupted mean and converges to the trimmed mean.

the trimmed mean and significantly improves over the corrupted mean (indicated by the black dashed line). Overall, GOTRIM quickly outperforms the naive mean under corruption and ClippedGossip which will ultimately converge to the corrupted mean. Additional experiments and implementation details are provided in the Supplementary Material.

## 5 Conclusion and Discussion

We introduced and analyzed two novel gossip algorithms: GORANK for rank estimation and GOTRIM for trimmed mean estimation. We proved convergence rates of O (1 /t ) and established robustness guarantees for GOTRIM through breakdown point analysis. Empirical results show both methods perform well on large, poorly connected networks: GORANK quickly estimates ranks, and GOTRIM is robust to outliers and improves on the naive mean.

Byzantine Robustness. In this work, we focused on robustness to data contamination in the sense of Huber's framework. Extending these results to the more adversarial setting of Byzantine robustness remains an interesting direction. Developing a rigorous theoretical foundation for this setting is still an open problem, and we plan to address it in future work.

Asynchronous Extension. Although our analysis focuses on the synchronous setting, real-world systems are often asynchronous. We present the asynchronous version of GORANK in Appendix H. The theoretical analysis in the asynchronous setting is carried out in an extension to this work [35].

Scalability. To demonstrate the scalability of our method on large networks, we repeated the experiments from Fig. (c) in Sections 3.2 and 4.3, originally conducted with n = 500 , on larger networks with n = 1000 and n = 5000 . The detailed results are provided in Appendix I.

Robustness to Network Disruptions. While robustness to data contamination is important, the robustness of our proposed algorithms to network disruptions (e.g., edge or node failures, network partitioning) is equally crucial in real-world applications. In Appendix J, we provide a detailed analysis of how our current framework can be extended.

Performance on Sparse Graphs. An interesting question is whether our algorithms perform well on sparse graphs. In practice, however, performance depends more on the graph's connectivity than on its sparsity. To illustrate this, we present experiments in Appendix K on sparse graphs with varying levels of connectivity.

Rank-based Statistics. GOTRIM naturally extends to the decentralized estimation of rank-based statistics. This includes rank statistics [21], which are key tools in data analysis-particularly for robust hypothesis testing-as well as L-statistics (such as the Winsorized mean).

Further Discussion. In appendix L, we provide extended discussion on several topics, including the optimality of the bounds, faster gossip algorithms, extension to multivariate data, and potential applications like robust decentralized optimization.

## Acknowledgments

This research was supported by the PEPR IA Foundry and Hi!Paris ANR Cluster IA France 2030 grants. The authors thank the program for its funding and support.

## References

- [1] Aya Ayadi, Oussama Ghorbel, Abdulfattah M Obeid, and Mohamed Abid. Outlier detection approaches for wireless sensor networks: A survey. Computer Networks , 129:319-333, 2017.
- [2] Gilad Baruch, Moran Baruch, and Yoav Goldberg. A little is enough: Circumventing defenses for distributed learning. Advances in Neural Information Processing Systems , 32, 2019.
- [3] Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer. Machine learning with adversaries: Byzantine tolerant gradient descent. Advances in neural information processing systems , 30, 2017.
- [4] Stephen Boyd, Persi Diaconis, and Lin Xiao. Fastest mixing markov chain on a graph. SIAM review , 46(4):667-689, 2004.
- [5] Stephen Boyd, Arpita Ghosh, Balaji Prabhakar, and Devavrat Shah. Randomized gossip algorithms. IEEE transactions on information theory , 52(6):2508-2530, 2006.
- [6] Keyan Cao, Yefan Liu, Gongjie Meng, and Qimeng Sun. An overview on edge computing research. IEEE access , 8:85714-85728, 2020.
- [7] Mingzhe Chen, Deniz Gündüz, Kaibin Huang, Walid Saad, Mehdi Bennis, Aneta Vulgarakis Feljan, and H Vincent Poor. Distributed learning in wireless networks: Recent progress and future challenges. IEEE Journal on Selected Areas in Communications , 39(12):3579-3605, 2021.
- [8] Yudong Chen, Lili Su, and Jiaming Xu. Distributed statistical machine learning in adversarial settings: Byzantine gradient descent. Proceedings of the ACM on Measurement and Analysis of Computing Systems , 1(2):1-25, 2017.
- [9] Alessandro Chiuso, Fabio Fagnani, Luca Schenato, and Sandro Zampieri. Gossip algorithms for distributed ranking. In Proceedings of the 2011 American Control Conference , pages 5468-5473. IEEE, 2011.
- [10] Igor Colin, Aurélien Bellet, Joseph Salmon, and Stéphan Clémençon. Extending gossip algorithms to distributed estimation of u-statistics. Advances in Neural Information Processing Systems , 28, 2015.
- [11] Igor Colin, Aurélien Bellet, Joseph Salmon, and Stéphan Clémençon. Gossip dual averaging for decentralized optimization of pairwise functions. In International conference on machine learning , pages 1388-1396. PMLR, 2016.
- [12] Reinhard Diestel. Graph Theory , volume 173. Graduate Texts in Mathematics, Springer, 2025.
- [13] John C Duchi, Alekh Agarwal, and Martin J Wainwright. Dual averaging for distributed optimization: Convergence analysis and network scaling. IEEE Transactions on Automatic control , 57(3):592-606, 2011.
- [14] Jiashi Feng, Huan Xu, and Shie Mannor. Distributed robust learning. arXiv preprint arXiv:1409.5937 , 2014.
- [15] Michael B Greenwald and Sanjeev Khanna. Power-conserving computation of order-statistics over sensor networks. In Proceedings of the twenty-third ACM SIGMOD-SIGACT-SIGART symposium on Principles of database systems , pages 275-285, 2004.
- [16] Bernhard Haeupler, Jeet Mohapatra, and Hsin-Hao Su. Optimal gossip algorithms for exact and approximate quantile computations. In Proceedings of the 2018 ACM Symposium on Principles of Distributed Computing , pages 179-188, 2018.

- [17] Lie He, Sai Praneeth Karimireddy, and Martin Jaggi. Byzantine-robust decentralized learning via clippedgossip. arXiv preprint arXiv:2202.01545 , 2022.
- [18] István Heged˝ us, Gábor Danner, and Márk Jelasity. Decentralized learning works: An empirical comparison of gossip learning and federated learning. Journal of Parallel and Distributed Computing , 148:109-124, 2021.
- [19] Peter J Huber. Robust estimation of a location parameter. In Breakthroughs in statistics: Methodology and distribution , pages 492-518. Springer, 1992.
- [20] Peter J Huber and Elvezio M Ronchetti. Robust statistics . John Wiley &amp; Sons, 2011.
- [21] Jaroslav Hájek and Zbynˇ ek Šidák. Theory of Rank Tests . Probability and Mathematical Statistics. Academic Press, San Diego, second edition edition, 1999.
- [22] Hamid Jalalzai, Stephan Clémençon, and Anne Sabourin. On binary classification in extreme regions. Advances in Neural Information Processing Systems , 31, 2018.
- [23] David Kempe, Alin Dobra, and Johannes Gehrke. Gossip-based computation of aggregate information. In 44th Annual IEEE Symposium on Foundations of Computer Science, 2003. Proceedings. , pages 482-491. IEEE, 2003.
- [24] Maurice G Kendall. The treatment of ties in ranking problems. Biometrika , 33(3):239-251, 1945.
- [25] Anastasia Koloskova, Sebastian Stich, and Martin Jaggi. Decentralized stochastic optimization and gossip algorithms with compressed communication. In International conference on machine learning , pages 3478-3487. PMLR, 2019.
- [26] Priyanka Mary Mammen. Federated learning: Opportunities and challenges. arXiv preprint arXiv:2101.05428 , 2021.
- [27] Bojan Mohar, Y Alavi, G Chartrand, and Ortrud Oellermann. The laplacian spectrum of graphs. Graph theory, combinatorics, and applications , 2(871-898):12, 1991.
- [28] Karl Mosler. Depth statistics. Robustness and complex data structures: Festschrift in Honour of Ursula Gather , pages 17-34, 2013.
- [29] Krishna Pillutla, Sham M Kakade, and Zaid Harchaoui. Robust aggregation for federated learning. IEEE Transactions on Signal Processing , 70:1142-1154, 2022.
- [30] Helmut Rieder. Qualitative robustness of rank tests. Ann. Statist. , 10(1):205 - 211, 1982.
- [31] Robert J Serfling. Approximation theorems of mathematical statistics . John Wiley &amp; Sons, 2009.
- [32] Devavrat Shah et al. Gossip algorithms. Foundations and Trends® in Networking , 3(1):1-125, 2009.
- [33] Nisheeth Shrivastava, Chiranjeeb Buragohain, Divyakant Agrawal, and Subhash Suri. Medians and beyond: new aggregation techniques for sensor networks. In Proceedings of the 2nd international conference on Embedded networked sensor systems , pages 239-249, 2004.
- [34] Aad W. Van der Vaart. Asymptotic statistics , volume 3. Cambridge university press, 2000.
- [35] Anna Van Elst, Igor Colin, and Stephan Clémençon. Asynchronous gossip algorithms for rank-based statistical methods. To appear in International Conference on Federated Learning Technologies and Applications (FLTA) , 2025.
- [36] Esther Villar-Rodriguez, María Arostegi Pérez, Ana I Torre-Bastida, Cristina Regueiro Senderos, and Juan López-de Armentia. Edge intelligence secure frameworks: Current state and future challenges. Computers &amp; Security , 130:103278, 2023.
- [37] Sissi Xiaoxiao Wu, Hoi-To Wai, Lin Li, and Anna Scaglione. A review of distributed algorithms for principal component analysis. Proceedings of the IEEE , 106(8):1321-1340, 2018.

- [38] Lin Xiao and Stephen Boyd. Fast linear iterations for distributed averaging. Systems &amp; Control Letters , 53(1):65-78, 2004.
- [39] Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett. Byzantine-robust distributed learning: Towards optimal statistical rates. In International conference on machine learning , pages 5650-5659. Pmlr, 2018.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We clearly state the contributions in the abstract and introduction: two new gossip algorithms with corresponding convergence rate bounds, as well as numerical results.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We address the limitations in the Conclusion section.

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

Justification: We provide a clear problem formulation and setup in the second section. All the proofs are detailed in the Supplementary Material.

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

Justification: The proposed algorithms and experimental setup (see subsections Numerical Experiments) are clearly detailed, allowing for full reproducibility. Code with detailed instructions and data are also released in this purpose.

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

Justification: The code and data are available in an anonymous Github.

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

Justification: We provide most details of the experiments in the subsections Numerical Experiments and the rest of the details is available in the Supplementary Material.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We compute the mean error and standard deviation over all the runs. The variability comes from the edge sampling process and the distribution of the data over the graph.

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

Justification: As mentioned in the paper and detailed in the Supplementary Material, the experiments are not computationally intensive and are run on a single CPU for a few of hours.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The paper's contribution is of methodological nature and regarding the experiments, we use either synthetic data or publicly available data with no sensitive information.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Our paper is mostly theoretical but our introduction mentions the positive impact of decentralized and robust learning, regarding trustworthiness and frugality.

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

Answer: [NA]

Justification: The paper does not use existing assets.

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

Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Outline of the Supplementary Material

The Supplementary Material is organized as follows. Section A introduces two alternative gossip algorithms for distributed ranking: Baseline and Baseline++ . In Section B, we present auxiliary results related to gossip matrices and the convergence analysis of the standard gossip algorithm for averaging. The detailed convergence analysis of GORANK, including the proofs of Theorems 1 and 2, is provided in Section C. Section D presents the convergence proofs for GOTRIM, specifically Lemma 1 and Theorem 3. In Section E, we provide the robustness analysis of GOTRIM ( i.e., the proof of Theorem 4). The convergence analysis of ClippedGossip is covered in Section F. Section G includes additional experiments and implementation details. Section H introduces Asynchronous GORANK, along with corresponding experimental results. Section I details experiments on larger networks. Section J describes how the current framework can be extended to include network disruptions. Section K provides experiments on sparse networks. Finally, Section L offers further discussion and outlines future work.

## A Gossip Algorithms for Ranking

Here, we detail two gossip algorithms for ranking, which can be considered natural competitors to GORANK. The algorithm presented first was proposed in [9] and, to our knowledge, is the only decentralized ranking algorithm documented in the literature. The algorithm presented next can be seen as a variant of the latter, incorporating the more efficient communication scheme of GORANK.

## A.1 Baseline: Algorithm from Chiuso et al.

Chiuso et al. propose a gossip algorithm for distributed ranking in a general (connected) network [9]. They demonstrate that this algorithm solves the ranking problem almost surely in finite time. However, they do not provide any convergence rate or non-asymptotic convergence results. The algorithm is outlined in Algorithm 3 and proceeds as follows. At each time step t , an edge ( i, j ) is selected, and the algorithm operates in three phases:

1. Ranking: The nodes check if both their local and auxiliary ranks are consistent with the corresponding local and auxiliary observations. If the ranks are inconsistent, the nodes exchange their local and auxiliary rank.
2. Propagation: The nodes swap all their auxiliary variables.
3. Local update: Each of the two nodes verifies if the auxiliary node has the same ID. If so, it updates its local rank estimate based on the auxiliary rank estimate.

## Algorithm 3 Algorithm from Chiuso et al. (Baseline)

- 1: Require: Each node with id I = k holds observation X .

```
k k 2: Init: Each node k initializes its ranking estimate R k ← k and its auxiliary variables R v k ← R k , X v k ← X k and I v k ← I k . 3: for t = 1 , 2 , . . . do 4: Draw ( i, j ) uniformly at random from E . 5: if ( X i -X j ) · ( R i -R j ) < 0 then 6: Swap rankings of nodes i and j : R i ↔ R j . 7: end if 8: if ( X v i -X v j ) · ( R v i -R v j ) < 0 then 9: Swap rankings of nodes i and j : R v i ↔ R v j . 10: end if 11: Swap auxiliary variables of nodes i and j : I v i ↔ I v j , R v i ↔ R v j and X v i ↔ X v j . 12: for p ∈ { i, j } do 13: if I v p = I p then 14: Update local ranking estimate: R p ← R v p . 15: end if 16: end for 17: end for 18: Output: Each node contains the estimate of the ranking.
```

## A.2 Baseline++ - Our Improved Variant Proposal

The algorithm selects an edge ( i, j ) at each step, and the corresponding nodes check if the auxiliary ranks are consistent with their auxiliary observations. If the ordering is inconsistent, the nodes swap their auxiliary ranks. Then, each node updates its local rank if the ordering of its local observation is inconsistent with the auxiliary observation. In contrast, the algorithm proposed by Chiuso et al. only updates the local estimates when the wandering estimate returns to its originating node. This design can significantly slow down convergence in graphs with low connectivity and long return times.

## Algorithm 4 Baseline++

```
1: Init: For all k , set R k ← k , R ′ k ← k and X ′ k ← X k . 2: for t = 1 , 2 , . . . do 3: Draw ( i, j ) ∈ E uniformly at random. 4: if ( X ′ i -X ′ j ) · ( R ′ i -R ′ j ) < 0 then 5: Swap rankings: R ′ i ↔ R ′ j . 6: end if 7: for p ∈ { i, j } do 8: if ( X ′ p -X p ) · ( R ′ p -R p ) < 0 or X ′ p = X p then 9: Update local rank: R p ← R ′ p . 10: end if 11: end for 12: Swap: R ′ i ↔ R ′ j and X ′ i ↔ X ′ j . 13: end for 14: Output: Estimate of ranks R k .
```

## B Auxiliary Results

In this section, we present key properties of the transition matrices that will be essential for the proofs of our main theorems. We also present the standard gossip algorithm for mean estimation, along with its convergence results.

## B.1 Properties of Gossip Matrices

Lemma (B.1) . Assume the graph G = ( V, E ) is connected and non-bipartite. Let t &gt; 0 . If at iteration t , edge ( i, j ) is selected with probability p = 1 | E | , then the transition matrices are given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For α ∈ { 1 , 2 } , denote W α ( t ) = I n -1 α ( e i -e j ) ( e i -e j ) ⊤ . The following properties hold:

- (a) The matrices are symmetric and doubly stochastic, meaning that

<!-- formula-not-decoded -->

- (b) The matrices satisfy the following equalities:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (d) The matrix W α is also doubly stochastic and it follows that 1 n is an eigenvector with eigenvalue 1.
- (c) For α ∈ { 1 , 2 } , we have

- (e) The matrix ˜ W α ≜ W α -1 n 1 ⊤ n n satisfies, by construction, ˜ W1 n = 0 , and it can be shown that ∥ ˜ W α ∥ op ≤ λ 2 ( α ) , where λ 2 ( α ) is the second largest eigenvalue of W α and ∥ · ∥ op denotes the operator norm of a matrix.
- (f) The eigenvalue λ 2 ( α ) satisfies 0 ≤ λ 2 ( α ) &lt; 1 and λ 2 ( α ) = 1 -λ 2 α | E | where λ 2 the spectral gap (or second smallest eigenvalue) of the Laplacian.

Proof. The proofs of (a), (b), (d), and (e) are omitted for brevity. For more details, we refer the reader to [5, 10]. The proof of (c) follows from the fact that

<!-- formula-not-decoded -->

and from the definition of the Laplacian matrix L = ∑ ( i,j ) ∈ E ( e i -e j ) ( e i -e j ) ⊤ . The proof of (f) can be found in [10] (see Lemmas 1, 2 and 3).

## B.2 Convergence Analysis of Gossip for Averaging

In the following section, we present classical results on the gossip algorithm for averaging.

Algorithm 5 Gossip algorithm for estimating the standard average [5]

- 1: Each node k initializes its estimate as Z k = X k .
- 2: for t = 1 , 2 , . . . do
- 3: Randomly select an edge ( i, j ) from the network.
- 4: Update estimates: Z i , Z j ← Z i + Z j 2 .
- 5: end for

Lemma (B.2) . Let us assume that G = ( V, E ) is connected and non bipartite. Then, for Z ( t ) defined in Algorithm 5, we have that for all k ∈ [ n ] :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. The proof is kept brief, as this is a standard result. For more details, we refer the reader to [5, 10]. To prove this lemma, we will primarily use points (e) and (f) from Appendix B.1.

Let t &gt; 0 . Recursively, we have E [ Z ( t )] = W t 2 X where W 2 = I n -1 2 | E | L .

Denote ˜ W 2 ≜ W 2 -1 n 1 ⊤ n n . Noticing that 1 n 1 ⊤ n n X = ¯ X n 1 n and that ˜ W t 2 = W t 2 -1 n 1 ⊤ n n , we have

<!-- formula-not-decoded -->

Moreover, for any t &gt; 0 ,

Since 1 ⊤ n ˜ W 2 = 0 , it follows that

<!-- formula-not-decoded -->

Setting c 2 = 1 -λ 2 (2) &gt; 0 finishes the proof, as 0 &lt; λ 2 (2) &lt; 1 .

Lemma (B.3) . Let us assume that G = ( V, E ) is connected and non bipartite. Then, for Z ( t ) defined in Algorithm 5, we have that for any t &gt; 0 :

<!-- formula-not-decoded -->

where c 2 = 1 -λ 2 (2) &gt; 0 , with λ 2 (2) = 1 -λ 2 2 | E | .

Proof. The original proof can be found in [5].

Let t &gt; 0 and ε &gt; 0 . At iteration t , we have Z ( t ) = W 2 ( t ) Z ( t -1) . Denoting Y ( t ) = Z ( t ) -¯ X n 1 n , it follows that Y ( t ) = W 2 ( t ) Y ( t -1) . Taking the conditional expectation, we get

<!-- formula-not-decoded -->

By repeatedly conditioning, we obtain the bound

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark. Using the Cauchy-Schwarz inequality E [ ∥ Y ∥ ] ≤ √ E [ ∥ Y ∥ 2 ] , we can also derive

<!-- formula-not-decoded -->

Setting c 2 = 1 -λ 2 (2) &gt; 0 finishes the proof, as 0 &lt; λ 2 (2) &lt; 1 .

## C Convergence Analysis of GORANK

In this section, we will prove Theorem 1 and 2.

## C.1 Proof of Theorem 1

Theorem 1. Let R ( t ) be defined in GoRank. We have that for all k ∈ [ n ] :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where c = λ 2 | E | , with λ 2 being the second smallest eigenvalue of the graph Laplacian (spectral gap), and σ k = n ˜ σ n ( r k -1 n ) , with ˜ σ n ( x ) = √ nx (1 -x ) .

Proof. To prove Theorem 1, we will primarily use points (e) and (f) from Appendix B.1.

Let k ∈ [ n ] , t &gt; 0 , and ¯ h k = 1 n h ⊤ k 1 n . From the previous derivation, one has:

<!-- formula-not-decoded -->

Denote ˜ W 1 ≜ W 1 -1 n 1 ⊤ n n . Noticing that h ⊤ k 1 n 1 ⊤ n n e k = ¯ h k and that ˜ W s 1 = W s 1 -1 n 1 ⊤ n n for 0 ≤ s ≤ t -1 , we have

<!-- formula-not-decoded -->

Since 1 ⊤ n ˜ W 1 = 0 , it follows that

<!-- formula-not-decoded -->

Finally,

Moreover, for any t &gt; 0 ,

Thus,

<!-- formula-not-decoded -->

since λ 2 (1) &lt; 1 and ∥ ∥ h k -¯ h k 1 n ∥ ∥ = √ ∑ i ( I { X k &gt;X i } -¯ h k ) 2 . Using 1 -λ 2 (1) = λ 2 | E | where λ 2 is the second smallest eigenvalue of the graph Laplacian, it follows that

<!-- formula-not-decoded -->

where ˜ σ n ( x ) = √ nx (1 -x ) . Plugging E [ R k ( t )] = n E [ R ′ k ( t )] + 1 and r k = n ¯ h k +1 finishes the proof:

<!-- formula-not-decoded -->

## C.2 Proof of Theorem 2

We now present a convergence result for the expected gap, which is key to proving the convergence of GOTRIM.

Theorem 2 (Expected Gap) . Let all k ∈ [ n ] , and let c and σ k be as defined in Theorem 1. Then, for all t ≥ 1 , we have: E [ | R k ( t ) -r k | 2 ] ≤ (3 /ct ) · σ 2 k . Consequently,

<!-- formula-not-decoded -->

Proof. The proof relies on the Cauchy-Schwarz inequality:

<!-- formula-not-decoded -->

Thus, to prove Theorem 2, a convergence result for the variance term E [ ( R k ( t ) -r k ) 2 ] or equivalently E [ ( R ′ k ( t ) -¯ h k ) 2 ] is required. This result is formalized in the following lemma.

Lemma (C.1) . We have

<!-- formula-not-decoded -->

where the constants are defined in Theorem 1.

Proof. Let t &gt; 0 and let k ∈ [ n ] . Using the update rule in GoRank , the estimated rank R ′ k ( t ) of agent k at iteration t can be expressed as follows:

<!-- formula-not-decoded -->

where for any 1 ≤ s ≤ t -1 , W ( s :) = W ( s ) . . . W (1) and W (0 :) = I n .

For any t ≥ 1 , let us define ˜ W 1 ( t ) as ˜ W 1 ( t ) ≜ W 1 ( t ) -1 n × n n . Note that W 1 ( t ) are a real symmetric matrices and that 1 n is always an eigenvector of such matrices. Therefore, for any 0 ≤ s ≤ t -1 , one has: ˜ W 1 ( s :) = W 1 ( s :) -1 n × n n . Using this decomposition in (17) yields

<!-- formula-not-decoded -->

since ¯ h k = h ⊤ k 1 n × n n e k and 1 ⊤ n ˜ W 1 ( s ) = 0 . The squared gap can thus be expressed as follows:

<!-- formula-not-decoded -->

or equivalently,

<!-- formula-not-decoded -->

Denoting ˜ h k = h k -¯ h k 1 n , define v ( s ) := ˜ h ⊤ k ˜ W 1 ( s ) ⊤ e k . We first consider the second term: 1 t 2 ∑ t -1 u =0 v ( u ) 2 . Note that

<!-- formula-not-decoded -->

Let J = 1 n 1 n 1 ⊤ n . Since ˜ h ⊤ k J = 0 , we can simplify:

<!-- formula-not-decoded -->

Since the product of two identical permutation matrices is equal to the identity matrix, we obtain W 1 ( u :) ⊤ W 1 ( u :) = I n and conclude that E [ v ( u ) 2 ] ≤ ∥ ˜ h k ∥ 2 . Now let's consider the first term: 2 t 2 ∑ s&lt;u v ( s ) v ( u ) . Note that for s &lt; u ,

<!-- formula-not-decoded -->

Taking expectation conditional on W 1 ( s :) , we have

<!-- formula-not-decoded -->

Since E [ ˜ W 1 ( u : s +1)] = ˜ W u -s 1 , we obtain the following bound:

<!-- formula-not-decoded -->

Using the previous derivation, we get:

<!-- formula-not-decoded -->

Combining the two terms, we obtain:

<!-- formula-not-decoded -->

Note that:

<!-- formula-not-decoded -->

Recalling c = 1 -λ 2 (1) ,

<!-- formula-not-decoded -->

Plugging ∥ ˜ h k ∥ 2 = σ n ( ¯ h k ) 2 and using E [( R ′ k ( t ) -¯ h k ) 2 ] = 1 n 2 E [( R k ( t ) -r k ) 2 ] finish the proof.

## D Convergence Proofs for GOTRIM

In this section, we will prove Lemma 1 and Theorem 3.

## D.1 Proof of Lemma 1

Lemma 1. Let R ( t ) and W ( t ) be defined as in Algorithm 1 and Algorithm 2, respectively. For all k ∈ [ n ] and t &gt; 0 , we have:

<!-- formula-not-decoded -->

where γ k = min( | r k -a | , | r k -b | ) ≥ 1 2 with a = ⌊ αn ⌋ + 1 2 and b = n - ⌊ αn ⌋ + 1 2 being the endpoints of interval I n,α . The constants c and σ n ( · ) are as defined in Theorem 2.

Proof of Lemma 1. Let k ∈ [ n ] and t &gt; 0 . Denoting p k ( t ) = E [ I { R k ( t ) ∈ I n,α } ] = P ( R k ( t ) ∈ I n,α ) , we have E [ W k ( t )] = p k ( t ) c n,α where c n,α = 1 -2 mn -1 with m = ⌊ αn ⌋ .

Hence, we need to show, for γ k &gt; 0 :

<!-- formula-not-decoded -->

The right-hand side of the inequality follows directly from Lemma C.1 via an application of Markov's inequality:

<!-- formula-not-decoded -->

For the left-hand side, we introduce the interval I n,α = [ a, b ] and define

<!-- formula-not-decoded -->

Observe that γ k ≥ 1 2 since r k is always discrete. We now analyze the three different cases.

Case 1: a ≤ r k ≤ b . Then, ∣ ∣ p k ( t ) -I { r k ∈ I n,α } ∣ ∣ = | 1 -p k ( t ) | = P ( R k ( t ) / ∈ I n,α ) . Since we have P ( | R k ( t ) -r k | ≤ γ k ) ≤ P ( R k ( t ) ∈ I n,α ) , it follows that the probability can be upper-bounded as P ( R k ( t ) / ∈ I n,α ) ≤ P ( | R k ( t ) -r k | &gt; γ k ) ≤ P ( | R k ( t ) -r k | ≥ γ k ) .

Case 2: r k &lt; a . Here, ∣ ∣ p k ( t ) -I { r k ∈ I n,α } ∣ ∣ = p k ( t ) = P ( R k ( t ) ∈ I n,α ) . Since it holds that P ( | R k ( t ) -r k | &lt; γ k ) ≤ P ( R k ( t ) / ∈ I n,α ) , we obtain P ( R k ( t ) ∈ I n,α ) ≤ P ( | R k ( t ) -r k | ≥ γ k ) .

Case 3: r k &gt; b This case follows symmetrically from the previous one.

In all cases, we have ∣ ∣ P ( R k ( t ) ∈ I n,α ) -I { r i ∈ I n,α } ∣ ∣ ≤ P ( | R k ( t ) -r k | ≥ γ k ) .

Finally, the result follows from E [ W k ( t )] = p k ( t ) c n,α and c n,α = 1 -2 mn -1 ≥ 1 -2 α .

## D.2 Proof of Theorem 3

Now, we will prove the convergence in expectation of the estimates of GOTRIM.

Theorem 3 (Convergence in Expectation of GoTrim) . Let Z ( t ) be defined GoTrim, and assuming the ranking algorithm is GoRank, we have that for all k ∈ [ n ] :

<!-- formula-not-decoded -->

Moreover, for any t &gt; T ∗ = min { t &gt; 1 | ct &gt; 2 log( t ) } , we have

<!-- formula-not-decoded -->

where K = [ σ 2 k γ 2 k ] n k =1 and c , σ k and γ k are constants defined in Lemma 1.

Proof of Theorem 3. Recall that for t &gt; 1 , the expected estimates are characterized recursively as:

<!-- formula-not-decoded -->

Denote ˜ W 2 ≜ W 2 -1 n 1 ⊤ n n and notice that ˜ W t 2 = W t 2 -1 n 1 ⊤ n n . Denoting S ( s ) = ∆ w ( s ) ⊙ X , we have have

<!-- formula-not-decoded -->

Since ∀ i, p i (0) = 0 , the first term can be rewritten as:

<!-- formula-not-decoded -->

where ∆ w i ( s ) = w i ( s ) -w i ( s -1) with w i ( s ) = E [ W i ( s )] . This leads to the bound:

<!-- formula-not-decoded -->

The first term simplifies as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where C i = 3 c · σ 2 k γ 2 i (1 -2 α ) comes from Lemma 1 and we denote C = [ C 1 , . . . , C n ] . To bound R ( t ) , we decompose it using an intermediate time step T = t -log( t ) /c 2 &gt; 0 , where c 2 = 1 -λ 2 (2) &gt; 0 . Let T ∗ = min { t &gt; 1 | c 2 t &gt; log( t ) } . We obtain for all t &gt; T ∗ ,

<!-- formula-not-decoded -->

Using Lemma 1, we derive for s &gt; 1 , | ∆ w k ( s ) | ≤ 2 C k s -1 . We obtain ∥ S (1) ∥ ≤ ∥ C ⊙ X ∥ and for s &gt; 1 ,

<!-- formula-not-decoded -->

Applying this bound and using c 2 = 1 -λ 2 (2) yields:

<!-- formula-not-decoded -->

Finally, we establish the following error bound:

<!-- formula-not-decoded -->

where c = 2 c 2 . This bound simplifies to

<!-- formula-not-decoded -->

Moreover, we have ∥ C ⊙ X ∥ = 3 c (1 -2 α ) ∥ K ⊙ X ∥ where K = [ σ 2 k γ 2 k ] n k =1 , which completes the proof.

## E Robustness Analysis: Breakdown Point of the Partial Trimmed Mean

Here, we prove provide a breakdown point analysis of GOTRIM and prove Theorem 4.

Theorem 4 (Breakdown Point) . Let τ &gt; 0 and δ, α ∈ (0 , 1) . Denote m = ⌊ αn ⌋ . With probability at least 1 -δ , the τ -breakdown point ε ∗ i ( t ) of the partial α -trimmed mean at iteration t &gt; T for any node i satisfies

<!-- formula-not-decoded -->

where T = 4 c log ( n εδ ) denote a propagation time with c being the connectivity constant and ε := τ/ max i | X i | denote a tolerance parameter. Moreover, we define K ( δ ) := c m ( 1 -1 n ) √ cδ where c m = √

<!-- formula-not-decoded -->

Proof. Applying Lemma 1, we estimate the breakdown point while accounting for the uncertainty in rank estimation, and combine it with Lemma 2 via a union bound. Lemma 2 introduces T the number of iterations required for the mean to propagate and the maximum delay T corresponds to the smallest possible ε , given by ε = τ B , where B = max i | X i | . Setting δ 1 = ( 1 -1 n ) δ and δ 2 = δ n complete the proof.

Lemma E.1 (Instant Breakdown Point). Let δ 1 &gt; 0 . Denote ˜ ε ∗ ( t ) the breakdown point at iteration t as the statistic defined as

<!-- formula-not-decoded -->

Then, with probability at least 1 -δ 1 ,

<!-- formula-not-decoded -->

where we define m = ⌊ nα ⌋ and c m = √ 3 n 3 / 2 · ϕ (( m -1) /n ) . As t →∞ , note that p = ⌊ 1 2 + m ⌋ = m , which allows us to recover the breakdown point of the α -trimmed mean.

Proof. We are interested in the observations X k with rank 1 ≤ r k ≤ m (the outliers that should be excluded from the mean), where m = ⌊ nα ⌋ . Note that we do not consider the data points with rank n -m +1 ≤ r k ≤ n , as this case is symmetrical for determining the breakdown point.

Let 1 ≤ r k ≤ m and consider the probability p k ( t ) of including X k in the mean. This probability satisfies: √

<!-- formula-not-decoded -->

where a = m + 1 2 is the left endpoint of the inclusion interval. The breakdown point can be interpreted as p n where p is the maximum rank required to "break" the mean and n is the sample size. To determine the maximum r k in [1 , m ] such that the probability of inclusion is lower than a certain confidence parameter δ , we solve for r k :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and obtain

Thus, with probability at least 1 -δ , the breakdown point is given by ˜ ε ∗ ( t ) = p n .

Lemma E.2 (Propagation Time). Let δ 2 &gt; 0 be a confidence parameter and ε &gt; 0 a tolerance parameter. We define Z ( · ) as the evolution of the standard mean estimate, initialized as Z (0) = X . Additionally, we introduce a perturbed estimate, ˜ Z ( · ) , which starts at ˜ Z (0) = X + B e j . At iteration t , we detect a mistake of magnitude B , we correct it by injecting -B , leading to the update

˜ Z ( t ) ← ˜ Z ( t ) -B e j . Then, it follows that, with probability at least 1 -δ 2 , for all s ≥ t + T , for any node i ,

<!-- formula-not-decoded -->

where T represents number of iterations required to correct a mistake with tolerance ε &gt; 0 and is given by T = 4 c log ( 1 εδ 2 ) with c is the connectivity of the graph.

Proof. At iteration t , the perturbed estimate is ˜ Z ( t ) = W 2 ( t :) ˜ Z (0) = Z ( t ) + W 2 ( t :) B e j since Z ( t ) = W 2 ( t :) Z (0) . Then, at iteration s &gt; t , we inject -B at node j which gives the following: ˜ Z ( s ) = W 2 ( s : t ) [ Z ( t ) + W 2 ( t :) B e j -B e j ] . Thus, for any s &gt; t ,

<!-- formula-not-decoded -->

Following the proof in [5], by repeatedly conditioning, we obtain

<!-- formula-not-decoded -->

Since ∥ ∆ ( s ) ∥ ≤ B and taking the expectation on both sides, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let δ 2 &gt; 0 . We need to solve 1 ε 2 e -c 2 T ≤ δ , which gives T = 4 c log ( 1 εδ 2 ) . It follows that with probability at least 1 -δ 2 , for s ≥ t + T , for all nodes i ,

<!-- formula-not-decoded -->

## F Convergence Analysis of ClippedGossip

In this section, we show that ClippedGossip , in our pairwise setup with a fixed clipping radius, ultimately converges to the corrupted mean. The following lemma describes the update rule of ClippedGossip through its transition matrix.

Lemma (Transition Matrix). Assume at iteration t , edge ( i, j ) ∈ E is selected. Then, the update using ClippedGossip rule with constant clipping radius τ is given by x t +1 = W ( t ) x t where W ( t ) ∈ R n × n denotes the transition matrix at iteration t which is given by

<!-- formula-not-decoded -->

where L ij := ( e i -e j )( e i -e j ) ⊤ is the elementary Laplacian associated with edge ( i, j ) and α t ij := 1 2 min ( 1 , τ / ∥ x t i -x t j ∥ ) a constant that depends on the previous estimates. Moreover, we have W ( t ) W ( t ) ⊤ = I n -2 α t ij (1 -α t ij ) L ij . In the special case α ij = 1 / 2 , the matrix reduces to the standard averaging and is a projection matrix: W ( t ) W ( t ) ⊤ = W ( t ) .

Proof. At iteration t , if edge ( i, j ) ∈ E is selected, the update rule is

<!-- formula-not-decoded -->

where the clipping operator is defined as CLIP( z, τ ) := min(1 , τ / ∥ z ∥ ) z. This update can be expressed in matrix form as x t +1 = W ( t ) x t . It equals the identity matrix except for a 2 × 2 block corresponding to nodes i and j , given by

<!-- formula-not-decoded -->

Finally, we derive

Observe that L 2 ij = 2 L 2 ij and conclude.

The next lemma allows us to bound the coefficients α t ij of the transition matrix.

Lemma (Lower Bound on Coefficient of the Transition Matrix). Let m t = max ( k,l ) ∈ E ∥ x t k -x t l ∥ . We have m t ≤ m and thus we derive

<!-- formula-not-decoded -->

where α := min(1 , τ /m ) / 2 with m = max ( k,l ) ∈ E ∥ x 0 k -x 0 l ∥ .

Proof. Let m 1 = max ( k,l ) ∈ E ∥ x 1 k -x 1 l ∥ . We can show that m 1 ≤ m . After update, we have x 1 i = (1 -α ij ) x 0 i + α ij x 0 j and x 1 j = (1 -α ij ) x 0 j + α ij x 0 i . Observe that the new estimates remain within the convex hull of the original points. Since only x i and x k have changed, the next maximum distance is:

̸

̸

̸

<!-- formula-not-decoded -->

̸

̸

Note: ∥ x 0 k -x 0 ℓ ∥ ≤ m by definition. So we just need to show that the other terms are smaller than m . For k = i, j , recall x 1 i = (1 -α ij ) x 0 i + α ij x 0 j , then we have

<!-- formula-not-decoded -->

̸

by the triangle inequality. Similarly, we derive for k = i, j , ∥ x 1 j -x 0 k ∥ ≤ m. Finally, since 0 &lt; 1 -2 α ij &lt; 1 , we have

<!-- formula-not-decoded -->

Recursively, we have for all t ≥ 1 , m t ≤ m , which finishes the proof.

The next lemma shows that at each iteration, the expected gap between the estimates and the corrupted mean will decrease.

Lemma (Contraction Bound). Let t &gt; 0 . Define y t = x t -¯ x 1 n where ¯ x := (1 /n ) ∑ n i =1 x 0 i is the initial average. We have the following bound:

<!-- formula-not-decoded -->

where β = 2 α (1 -α ) with α := min(1 , τ /m ) / 2 for m = max ( k,l ) ∈ E ∥ x 0 k -x 0 l ∥ and λ 2 denotes the spectral gap of the Laplacian.

Proof. Taking the conditional expectation over the sampling process, we have

<!-- formula-not-decoded -->

Since y t ⊥ 1 , and 1 is the eigenvector corresponding to the largest eigenvalue 1 of E [ W ( t ) ⊤ W ( t ) | y t ] ,

<!-- formula-not-decoded -->

Now from the first lemma, we have W ( t ) ⊤ W ( t ) = I n -β t ij L ij where β t ij = 2 α t ij (1 -α t ij ) . We see that β ≤ β t ij ≤ 1 / 2 where β = 2 α (1 -α ) . Observing that E [ W ( t ) ⊤ W ( t ) | y t ] = I n -(1 / | E | ) ˜ L t where we define the weighted Laplacian as

<!-- formula-not-decoded -->

Using the quadratic form of the weighted Laplacian:

<!-- formula-not-decoded -->

we derive β L ⪯ ˜ L t ⪯ (1 / 2) L , as β ≤ β t ij ≤ 1 / 2 . Using the Courant-Fischer Theorem, we obtain: βλ 2 ≤ ρ t ≤ λ 2 / 2 , where ρ t denotes the second smallest eigenvalue of ˜ L t and λ 2 is the second smallest eigenvalue of the (unweighted) Laplacian L . We therefore obtain a bound on the second largest eigenvalue of E [ W ( t ) ⊤ W ( t ) | y t ] :

<!-- formula-not-decoded -->

which finishes the proof.

Finally, the convergence of ClippedGossip estimates is established in the following proposition.

Proposition (Convergence of ClippedGossip Estimates). For each node k ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, we have where ¯ x := (1 /n ) ∑ n i =1 x 0 i is the initial average, β = 2 α (1 -α ) with α := min(1 , τ /m ) / 2 and m = max ( k,l ) ∈ E ∥ x 0 k -x 0 l ∥ and λ 2 denotes the spectral gap of the Laplacian.

Proof. Using the previous lemma, recursively, we obtain

<!-- formula-not-decoded -->

Recalling y t = x t -¯ x 1 n finishes the proof.

## G Additional Experiments and Implementation Details

This section provides additional experiments and more details on the Basel Luftklima dataset as well as compute resources.

## G.1 Experiments Compute Resources

The experiments are run on a single CPU with 32 GB of memory.

The execution time for each experiment is less than 30 minutes (except for the large-scale experiments). The details of a few experiments are given in Table 1.

Table 1: Execution times for all experiments

| Experiments                                                                                                | Execution Time                                              | Figure                                                                                             |
|------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| exp1+exp2+exp3 exp4+exp5+exp6 exp7+exp8+exp9 exp10+exp10a+exp10b exp11+exp12+exp13 exp14 exp15+exp16+exp17 | ∼ 30 min ∼ 5 min ∼ 15 min ∼ 50 min ∼ 15 min ∼ 5 min ∼ 5 min | Ranking (a) Ranking (b) Ranking (c) Trimmed Mean (a) Trimmed Mean (b) Trimmed Mean (c) Ranking (d) |

## G.2 Basel Luftklima Dataset

The dataset contains temperature measurements from 99 Meteoblue sensors across the Basel region, recorded between April 14 and April 15, 2025. For each sensor, only the first observation is used. A graph is built by connecting sensors that are within 1 km of each other, based on their geographic coordinates. Only the connected component of the graph is kept. To avoid ties during ranking, we add a small, imperceptible amount of noise to the data.

## G.3 Additional Experiments

Figures 4, 5, and 6 extend the experiments presented in sections 3 and 4.

<!-- image -->

Figure 4: Role of ϕ for three different communication graphs

Figure 5: Comparison of ranking algorithms on three different communication graphs

<!-- image -->

Figure 6: Comparison of gossip algorithms for robust mean estimation on three different graphs

<!-- image -->

## H Asynchronous Variant of GORANK

In this section, we propose an asynchronous variant of GORANK that operates without access to a global clock or a shared iteration counter. Instead, each node k maintains a local counter C k which tracks the number of times it has participated in an update. Asynchronous GORANK, outlined in Algorithm 6 proceeds as follows. When node k is selected, it increments C k and updates its local rank estimate using a running average, where the global iteration t is replaced by C k . As in the synchronous version of GoRank , selected nodes also exchange their auxiliary observations. Note that this asynchronous version is more efficient, as it significantly reduces the number of updates-performing only two updates per iteration instead of n .

## Algorithm 6 Asynchronous GoRank

```
1: Init: For each k ∈ [ n ] , Y k ← X k , R ′ k ← 0 , C k ← 0 . 2: for t = 1 , 2 , . . . do 3: Draw ( i, j ) ∈ E uniformly at random. 4: for p ∈ { i, j } do 5: Set C p ← C p +1 . 6: Set R ′ p ← (1 -1 /C p ) R ′ p +(1 /C p ) I { X p >Y p } . 7: Update rank estimate: R p ← nR ′ p +1 . 8: end for 9: Swap auxiliary observation: Y i ↔ Y j . 10: end for 11: Output: Estimate of ranks R k .
```

Figure 7 shows a comparison of Asynchronous GoRank with our two other ranking algorithms: Synchronous GoRank and Baseline++. The results suggest that Asynchronous GoRank converges slightly faster than Synchronous GoRank, thereby outperforming it in both speed and efficiency.

Figure 7: Comparison of Asynchronous GoRank with other ranking algorithms. Results show that Asynchronous GoRank slightly converges faster than the synchronous version.

<!-- image -->

Note that the convergence analysis in the asynchronous case is more complex because it requires analyzing the ratio of two statistics. Nonetheless, though technically more demanding, it is possible to derive convergence rate bounds using Taylor expansions techniques. We will carry out such an analysis in an extension to this work.

## I Large-scale Experiments

To demonstrate the scalability of our method on large networks, we repeated the experiments from Fig. (c) in Sections 3.2 and 4.3, originally conducted with n = 500 , on larger networks with n = 1000 and n = 5000 .

For the experiments related to Section 3.2, the results lead to the same conclusions: Figure 8 shows that GoRank continues to achieve a low error ( &lt; 0 . 1 ), and the overall performance trends of the other ranking algorithms remain similar on the Watts-Strogatz graph but are significantly worse on the 2D grid graph. These additional results further reinforce GoRank's scalability and robustness, particularly in comparison to the other methods.

For the experiments related to Section 5.4, Figure 9 shows that GoTrim + GoRank continues to converge efficiently to the trimmed mean, even on larger 2D grid graphs, clearly outperforming the corrupted mean. In contrast, GoTrim + Baseline++ converges significantly more slowly on the 2D grid (and does not even outperform the corrupted mean), although it remains efficient on Watts-Strogatz graphs.

Figure 8: Comparison of the performance of gossip algorithms for ranking across large networks.

<!-- image -->

Figure 9: Comparison of the performance for trimmed means estimation across large networks.

<!-- image -->

## J Robustness to Network Disruptions

While robustness to data contamination is important, the robustness of our proposed algorithms to network disruptions (e.g., edge/node failures, network partitioning) is equally crucial in real-world applications.

One natural extension is to introduce a fixed probability of failure for each node or edge (see [2]). This would modify the edge sampling: instead of a uniform distribution, the algorithm would sample edge e with probability p e . In our analysis, this change corresponds to replacing the normalized Laplacian L/ | E | with a weighted sum ∑ e p e L e , where ∑ e p e &lt; 1 due to edge failures and L e correspond to the elementary Laplacians. The spectral gap, which governs the convergence rate, would now correspond to the connectivity constant ˜ c of this weighted Laplacian instead of c = λ 2 / | E | . For instance, if each edge fails independently with probability 0.5, then p e = 1 / 2 | E | for all e , and the effective spectral gap becomes ˜ c = λ 2 / (2 | E | ) &lt; c , indicating a slower convergence rate. We emphasize that as long as p e &gt; 0 for all e , the weighted graph remains connected and non-bipartite if the original graph was.

Another interesting scenario is network partitioning. One way to model this is by designing graphs composed of tightly connected clusters with only a few inter-cluster edges that are prone to failure. In such a setup, the connectivity constant would degrade significantly, and we expect the convergence rate to reflect this bottleneck. We generated a graph consisting of 500 nodes organized into three well-connected clusters, with only five inter-cluster edges. As expected, we observed a very low connectivity constant c = 1 . 86 × 10 -6 and the convergence behavior of this graph is similar to that of a 2D grid graph, which is consistent with the low overall connectivity.

## K Experiments on Sparse Graphs

An interesting question is whether our algorithms performance well on sparse graphs. First, we note that the Watts-Strogatz and 2D grid graphs used in our main experiments are already quite sparse, with both containing fewer than 1000 edges. However, we would like to emphasize that

what primarily governs convergence behavior is not sparsity per se, but graph connectivity. For example, a dense graph composed of loosely connected clusters may have poor connectivity, while a sparse graph like a 3-regular graph can exhibit strong connectivity properties. A cycle graph, in addition to being sparse, has very low connectivity and serves as a useful pathological case. To better illustrate the relationship between topology and performance, we conducted additional experiments using three different sparse graphs: 1) Watts-Strogatz graph (connectivity c = 3 . 31 × 10 -4 , ˜ 1000 edges), 2) Cycle graph ( c = 3 . 16 × 10 -7 , 499 edges), 3) 3-regular graph ( c = 2 . 24 × 10 -4 , 750 edges). All experiments were run for 15 × 10 4 iterations. For GoRank, both the Watts-Strogatz and 3-regular graphs achieved fast convergence (absolute error below 0.05), consistent with their strong connectivity properties typical of expander graphs. The cycle graph, as expected, performed worse due to its very low connectivity, but still achieved an error below 0.1. For GoTrim, using the same corruption model as in Fig. 5b, the Watts-Strogatz and 3-regular graphs again performed very well, with errors close to 0. However, the cycle graph exhibited much slower convergence: after 8 × 10 4 iterations, the absolute error remained above 40. Doubling the number of iterations reduced the error to below 40. While this is indeed slow, it is important to note that performance remains better than the corrupted mean, and such a topology is highly atypical in real-world sensor networks. In practice, such scenarios would likely require gossip algorithms specifically designed for cycle graphs. These experiments support the conclusion that GoTrim still performs well on sparse graph with good connectivity properties.

Figure 10: Comparison of the performance of GoRank and GoTrim on different sparse graphs.

<!-- image -->

## L Further Discussion and Future Work

Our main focus was to develop foundational results for robust decentralized estimation. This section highlights directions for deepening our understanding of the algorithms' properties, extending them to more complex settings, and applying them to related problems.

Optimality of the Bounds. To the best of our knowledge, there are currently no established lower bounds on the convergence rate for the class of gossip-based algorithms applied to either decentralized ranking or trimmed mean estimation. Thus, the optimality of our algorithms remains an open question. In the special case of a complete graph, we observe that significantly faster convergence than the typical O (1 /t ) rate is possible. For example, in the Baseline++ algorithm, which performs direct ranking swaps, we can show exponential convergence of the form exp( -t/ | E | ) . However, this fast convergence critically relies on the high connectivity of the complete graph and does not generalize well to less connected graphs. In contrast, GORANK demonstrates more robust performance across general graphs, including those with limited connectivity, where achieving O (1 /t ) convergence is already non-trivial. Deriving a formal lower bound for arbitrary graphs remains an open and challenging problem. Nevertheless, it can be noted that with the current approach, the GORANK convergence rate of O (1 /t ) is tight in the sense that rank estimation involves averaging t random indicator functions. Finally, since GOTRIM relies on the rank estimates from GORANK, it inherits the O (1 /t ) convergence rate, which is already near-optimal.

Faster Gossip Algorithms. Designing faster gossip algorithms is an interesting direction, and we have also been exploring this aspect. One promising approach involves optimizing the edge sampling strategy based on the graph connectivity constant c (see [4, 38]). While this strategy improves gossip

performance in standard averaging tasks, our empirical experiments suggest that its impact on the GORANK algorithm is more limited. This is likely due to the relatively small variation in c , which is insufficient to significantly affect a rate in 1 /ct -though it has a more noticeable effect under a geometric rate.

Extension to Multivariate Data. Extending the proposed approach to multivariate data is an important direction. A natural extension involves ranking multivariate observations based on their norms. Specifically, one could define a suitable norm depending on the task (e.g., Euclidean), compute the norm for each observation, and then apply our univariate ranking method (GoRank) to these one-dimensional values. Observations with the largest norms can then be treated as potential outliers. Following this, a multivariate version of GoTrim can be defined by discarding the top k = ⌊ αn ⌋ observations with the largest norms, and computing the mean of the remaining points using the standard gossip algorithm. Alternatively, one could explore data depths as a generalization of ranking in multivariate settings (see [28]). Data depth provides a measure of centrality for multivariate data. While this approach is promising, it would require a more in-depth investigation beyond the scope of the current work, since depth computations usually require to solve computationally demanding optimization problems, just like alternative methods recently designed to define multivariate ranks (e.g. based on optimal transport). Finally, we could simply compute the coordinate-wise trimmed mean by applying GoTrim to each coordinate individually [29, 39].

Applications. The proposed methods offer a promising foundation for robust decentralized optimization. In particular, GOTRIM could be integrated into existing mean-based optimization algorithms to enhance robustness against outliers [11, 25, 37]. Realizing this integration, however, will require further algorithmic and theoretical development. Additionally, our ranking algorithms may prove valuable in extreme value theory, where identifying rare or extreme observations is critical-especially in high-stakes domains such as finance, insurance, and environmental science [22].