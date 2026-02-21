## Random Search Neural Networks for Efficient and Expressive Graph Learning

Michael Ito University of Michigan mbito@umich.edu

Danai Koutra University of Michigan dkoutra@umich.edu

## Abstract

Random walk neural networks (RWNNs) have emerged as a promising approach for graph representation learning, leveraging recent advances in sequence models to process random walks. However, under realistic sampling constraints, RWNNs often fail to capture global structure even in small graphs due to incomplete node and edge coverage, limiting their expressivity. To address this, we propose random search neural networks (RSNNs), which operate on random searches, each of which guarantees full node coverage. Theoretically, we demonstrate that in sparse graphs, only O (log | V | ) searches are needed to achieve full edge coverage, substantially reducing sampling complexity compared to the O ( | V | ) walks required by RWNNs (assuming walk lengths scale with graph size). Furthermore, when paired with universal sequence models, RSNNs are universal approximators. We lastly show RSNNs are probabilistically invariant to graph isomorphisms, ensuring their expectation is an isomorphism-invariant graph function. Empirically, RSNNs consistently outperform RWNNs on molecular and protein benchmarks, achieving comparable or superior performance with up to 16 × fewer sampled sequences. Our work bridges theoretical and practical advances in random walk based approaches, offering an efficient and expressive framework for learning on sparse graphs.

## 1 Introduction

Early work on random walk-based graph representations focused on using skip-gram objectives to learn node embeddings from sampled walks [1, 2]. Building on these ideas and leveraging recent advances in sequence modeling, random walk neural networks (RWNNs) have emerged as a powerful paradigm for modern graph learning [3-8], overcoming the limitations of message-passing neural networks (MPNNs) [9-11] and graph transformers [12-14] by representing graphs as collections of random walks processed by sequence models. This advancement aligns with the broader research goal of identifying effective and expressive methods for graph representation learning [15-17]. However, despite their success, RWNNs encounter critical expressivity challenges under realistic conditions due to incomplete node and edge coverage, limiting their capacity to capture structure even in small graphs (Figure 1). In our analysis, we establish that, under partial coverage, RWNNs are strictly less expressive than traditional MPNNs, highlighting the importance of complete coverage and bridging the theoretical expressivity of the two paradigms.

To illustrate the limitations of RWNNs, consider the graph composed of connected six-cycles and side chains shown in Figure 1. Capturing the full structure of this graph requires traversing every node and edge. However, since the node and edge cover times for a random walk can scale as O ( | V || E | ) [18], RWNNs require either prohibitively long walks or an impractically large number of samples to guarantee complete coverage. Under realistic sampling constraints where the walk's number of steps is significantly less than O ( | V || E | ) , random walks obtain only partial graph reconstruction: as shown in Figure 1(a), subgraphs induced by short random walks can miss critical structural components,

Jenna Wiens University of Michigan wiensj@umich.edu

## (a) Random Walk Extraction

Figure 1: RWNN and RSNN coverage differences. Random walks miss critical structure under realistic sampling constraints, wheras each individual search only misses single edges in cycles, enabling complete reconstruction across logarithmic sampling in | V | on sparse graphs.

<!-- image -->

such as the side chains connected to the six-cycles. This incomplete coverage significantly hinders RWNNexpressivity. Current methods attempt to address this limitation through non-backtracking walks [5, 6] and minimum-degree local rules (MDLR) [7], reducing node and edge cover time to O ( | V | 2 ) . Nonetheless, these approaches retain quadratic complexity with respect to graph size, making comprehensive coverage costly and impractical for even small and medium graphs.

To overcome these challenges in small and medium sized graphs, we introduce random search neural networks (RSNNs) , which represent graphs as collections of random searches. Critical to our analysis is the insight that subgraphs induced by searches are spanning trees as opposed to arbitrary subgraphs induced by random walks. Each spanning tree inherently ensures full node coverage, reducing the task to achieving edge coverage across the union of induced trees. Leveraging this insight, our analysis demonstrates that RSNNs require only a logarithmic number of searches for complete edge coverage, specifically in sparse graphs where such searches are computationally feasible. This is a substantial improvement over the linear number of walks required by RWNNs, assuming walk lengths scale with graph size. As shown in Figure 1(b), the union of just a few spanning trees enables complete reconstruction of the graph, including nodes and edges missed by walk-induced subgraphs. When equipped with maximally expressive sequence models, RSNNs achieve universal approximation efficiently. Furthermore, we show that RSNNs are probabilistically invariant to graph isomorphisms, ensuring their expectation is an isomorphism-invariant predictor. Empirically, we focus on sparse molecular and protein graph classification datasets, domains in which RWNNs have shown significant improvement over existing GNNs. Across both domains, we demonstrate that RSNNs consistently outperform existing RWNN approaches. In summary, we make the following contributions:

- Characterizing RWNN Expressive Limitations. Our analysis characterizes the expressive power of RWNNs, bridging the expressivity of RWNNs and MPNNs. We demonstrate that RWNNs under partial node and edge coverage are strictly less expressive than MPNNs, motivating the design of sampling strategies that guarantee full coverage.
- New Model: Random Search Neural Networks. We propose random search neural networks (RSNNs), a new approach that operates on random searches, whose induced subgraphs are spanning trees, substantially reducing the sample size required for complete node and edge coverage in sparse graphs.
- Efficient Coverage, Universal Approximation, &amp; Isomorphism Invariance. We demonstrate that RSNNs can achieve universal approximation efficiently with logarithmic sampling in sparse graphs. RSNNs are also probabilistically invariant to graph isomorphims, ensuring their expectation is an isomorphism-invariant function on graphs.
- Extensive Empirical Analysis. Focusing on sparse molecular and protein graph benchmarks, we demonstrate that RSNNs consistently outperform existing RWNNs.

## 2 Background and Preliminaries

We establish notation for graphs and random walks and next review MPNNs and RWNNs, the primary class of models under investigation. Importantly, we later bridge the expressivity of MPNNs and RWNNs. We lastly review random walk cover times, highlighting how RWNNs require prohibitively long walks or impractically large numbers of walks to guarantee full graph coverage.

## 2.1 Notation and Random Walks on Graphs

We define a graph G = ( V, A , X ) , where V is the set of nodes, A ∈ { 0 , 1 } | V |×| V | is the adjacency matrix representing the set of edges E , and X ∈ R | V |× d is the node feature matrix. For each node i ∈ V , we denote its feature vector as x i and its set of immediate (one-hop) neighbors as N ( i ) . We define the augmented neighborhood ˆ N ( i ) , obtained by adding a self-loop to node i .

A random walk of length ℓ on G produces a sequence of nodes W = ( w 0 , . . . , w ℓ ) by first sampling an initial node w 0 ∈ V according to a uniform distribution P 0 , and then iteratively transitioning to subsequent nodes by sampling neighbors according to a given random walk algorithm. We let W ℓ ( G ) denote the set of all possible random walks of length ℓ on G , and let P ( W ( G ) , P 0 ) represent a probability distribution over these walks. Lastly, we define P m ( W ( G )) = { W 1 , . . . , W m } as a realization of a set of m independently sampled random walks from P ( W ( G ) , P 0 ) .

## 2.2 Message-passing Neural Networks and GNN Expressivity

Standard GNNs adopt a message-passing approach, where each layer iteratively updates a node's representation by aggregating the features of its neighbors [19]. Formally, the initial message-passing layer can be defined as the following propagation rule at the node level for all i ∈ V ,

<!-- formula-not-decoded -->

where f agg is a permutation-invariant function. Because of this aggregation step, MPNNs incur fundamental expressivity limitations and cannot distinguish certain classes of non-isomorphic graphs [15, 20]. We compare the expressivity of GNNs by the pairs of graphs they can distinguish [21], introducing the following notation. For two GNNs f 1 and f 2 , we write

<!-- formula-not-decoded -->

̸

Thus, any pair indistinguishable by f 1 is also indistinguishable by f 2 , so f 1 is at least as expressive as f 2 . The relation is strict, f 2 ≺ f 1 , if f 2 ⪯ f 1 and there exist graphs G,H with f 1 ( G ) = f 1 ( H ) while f 2 ( G ) = f 2 ( H ) . f 1 and f 2 are equally expressive, written f 1 ≃ f 2 , if f 2 ⪯ f 1 and f 1 ⪯ f 2 . These relations coincide with notions of approximation power. For example, if f 2 ≺ f 1 , every target approximable by f 2 is approximable by f 1 , and there exist targets approximable by f 1 but not f 2 .

## 2.3 Random Walk Neural Networks

RWNNs are a novel class of neural network on graphs that leverage sequence models to process random walks sampled from the graph. Typically, an RWNN is characterized by four key components: (1) a random walk algorithm that generates node sequences, (2) a recording function that encodes the walks into structured representations, (3) a reader neural network that processes these representations, and (4) an aggregator that combines the representations or predictions from multiple walks.

For our analysis, we assume the following representative general version of RWNN [3-8]. Specifically, we consider the random walk algorithm as uniform random walks of fixed length ℓ , denoted by P m ( W ( G )) := P ( W ℓ ( G ) , U ( V )) , where U ( V ) denotes the uniform distribution over V . Given a sampled walk W ∈ P m ( W ( G )) , we define the recording function f emb : W ℓ ( G ) → R ℓ × d as follows:

<!-- formula-not-decoded -->

where h V : V → R d is a node embedding function. Here, h PE [ i ] serves as an optional position encoding that supplies extra structural context for each node in the walk (Appendix B); when such encoding is employed, it is further processed by the learnable projection mapping proj : R d pe → R d . Subsequently, we assume walk embeddings produced by f emb are processed by a sequence model,

denoted by f seq : R ℓ × d → R ℓ × d . Finally, embeddings from the sequence model are aggregated by a permutation-invariant function. The choice for the function can be simple functions such as taking the mean over random walk representations such as in Wang and Cho [3], Kim et al. [7], or it can be more complex as in Tönshoff et al. [5], Chen et al. [6], which updates a node's representation as the aggregation of its representations across all walks using the aggregation function f agg : R m × ℓ × d → R | V |× d :

<!-- formula-not-decoded -->

where N i ( P m ( W ( G ))) represents the number of occurrences of node i in the union of walks in P m ( W ( G )) . The RWNN layer is defined as the composition f l RWNN = f l agg ◦ f l seq , while the overall architecture f RWNN is defined as the stacking of RWNN layers. In the node classification setting, the final node representation f agg [ i ] produced by the last RWNN layer is directly utilized for predictions. In graph classification, an additional global pooling function aggregates these node representations into a single representation for the graph.

## 2.4 Random Walk Cover Times

RWNN expressivity depends on how much of the graph its random walks visit (Section 3). Here, we review results on random walk node cover times, C V ( G ) , the expected number of steps a walk takes to visit all nodes. For a connected graph G = ( V, E ) , the cover time of a general uniform random walk satisfies C V ( G ) = O ( | V || E | ) [22]; in particular, for sparse graphs ( | E | = Θ( | V | ) ) this gives C V ( G ) = O ( | V | 2 ) . Minimum-degree local rule (MDLR) walks further achieve C V ( G ) = O ( | V | 2 ) on all graphs, which is optimal among first-order walks [7, 23]. Non-backtracking walks can also empirically reduce cover times on graphs [5, 6]. Even with these improvements, guaranteeing full node and edge coverage by random walks can require prohibitively long walks or impractically large numbers of walks. We therefore replace walks entirely with searches (Section 4), significantly improving on the number of samples required for full coverage in comparison to random walks.

## 3 Expressive Power of Random Walk Neural Networks

In this section, we characterize the expressive power of RWNNs. Our main result establishes that without additional positional or structural encodings, RWNNs with access to the complete multiset of random walks whose lengths scale up to the cover time are exactly as expressive as MPNNs. In practice, however, such assumptions are unrealistic: guaranteeing full node and edge coverage requires walk lengths on the order of the cover time, rendering full coverage computationally infeasible. We then show that in the partial-coverage regime, RWNNs are strictly less expressive than MPNNs. This limitation motivates our random search neural network (RSNN), which achieves full coverage and thus maximal expressivity at significantly lower sampling cost.

## 3.1 The Role of Coverage: RWNNs vs. MPNNs

We first analyze the ideal setting in which the RWNN has access to complete walk sets up to the cover time. In this regime, RWNN expressive power matches that of MPNNs.

Theorem 3.1 (RWNN-MPNN Equivalence Under Full Coverage (FC)) . Let G be a graph. Let f FC RWNN denote an RWNN with injective f seq and f agg with no additional positional encodings, applied to the complete multiset of walks W ≤ ℓ ( G ) with lengths up to ℓ = C E ( G ) , the edge cover time of G . Let f MPNN be an MPNN with injective f agg . Then, for all graphs G,H ,

<!-- formula-not-decoded -->

Hence, f FC RWNN ≃ f MPNN (i.e., f FC RWNN and f MPNN are equal in expressive power).

Although Theorem 3.1 shows that full-coverage RWNNs and MPNNs are equal in expressivity, RWNNs under full coverage can be more effective empirically. RWNNs leverage expressive sequence models capable of capturing long-range dependencies when given full graph structure in complete sequences. MPNNs instead rely on iterative neighborhood aggregation and are limited in depth by oversmoothing [24] and oversquashing [25], which in practice reduce their expressivity and

capabilities to capture long-range signals. This contrasts our theoretical setup where we assume MPNNs have unlimited depth, allowing them to match full-coverage RWNN expressivity.

Constructing complete walk sets with lengths up to the cover time, however, is typically computationally infeasible. RWNNs can thus fall short of MPNNs under realistic budgets despite their inherent advantages. Indeed, as an immediate consequence of Theorem 3.1, when RWNNs operate under partial coverage, their expressive power is strictly less than that of MPNNs.

Corollary 3.2 (RWNNs Under Partial Coverage (PC)) . Let f PC RWNN denote an RWNN of the same architectural class as in Theorem 3.1 but applied to a multiset of random walks that attains only partial node/edge coverage of the input graph. Then, for all graphs G,H ,

<!-- formula-not-decoded -->

and there exist non-isomorphic graphs G ̸ ∼ = H such that

̸

<!-- formula-not-decoded -->

Hence, f PC RWNN ≺ f MPNN (partial-coverage RWNNs are strictly less expressive than MPNNs).

Corollary 3.2 reveals a fundamental limitation of RWNNs: under partial coverage, their expressive power falls below that of classical message passing. Thus, to attain maximal theoretical expressivity, it is essential to design sampling strategies that efficiently guarantee complete coverage. In order to realize the advantages of RWNNs while obtaining maximal expressivity, we introduce RSNNs (Section 4), which replace walks with searches to guarantee full node coverage by construction and achieve full edge coverage with a small number of searches on sparse graphs.

Insights of the analysis. In proving Theorem 3.1, we introduce a walk-based color refinement, Walk Weisfeiler-Lehman ( WWL ; Definition A.3), which updates each node using the multiset of walks that visit it. We demonstrate that WWL upper bounds RWNN expressivity (Lemma A.5). Next, we establish that WWL operates on the same object as classical WL : unfolding trees (Definition A.6). We lastly leverage this insight to establish that WWL and WL have equal distinguishing power (Theorem A.9). In essence, this construction aligns the Weisfeiler-Lehman hierarchy with RWNNs , unifying the expressive power of two seemingly distinct model classes: RWNNs, which process random walks with sequence models, and MPNNs, which process multisets of node neighborhoods with graph convolution. Formal definitions and details are in Appendix A.

## 4 Random Search Neural Networks (RSNNs)

Motivated by our analysis of RWNNs, we propose a new sampling strategy that efficiently achieves the necessary conditions for maximal expressivity: full node and edge coverage. Since random walks require either prohibitively long walks or an impractically large number of walks to guarantee full coverage, we introduce random search neural networks (RSNNs), which represent graphs as collections of random searches. Notably, a single search guarantees full node coverage, and under the sparse graph assumption, only O (log( | V | )) searches are needed to capture all edges. This significantly reduces the sampling complexity compared to the O ( | V | ) requirement for traditional RWNNs, assuming walk lengths scale on the order of O ( | V | ) . When paired with a maximally expressive sequence model, RSNNs emerge as universal approximators on graphs. Moreover, we provably show RSNNs are probabilistically invariant to graph isomorphisms. Hence, the predictor obtained by averaging over searches is an isomorphism-invariant graph function. While the computational cost of a full search can be significantly larger than a short random walk, we focus on sparse graphs where search is computationally feasible, addressing the limitations of RWNNs in these classes of graphs.

## 4.1 Search via Random DFS

RSNNs leverage a random depth-first search (DFS) procedure to obtain sequences from an input graph G . We utilize a DFS rather than a breadth-first search to better preserve continuity in the sequence. RSNN obtains a random DFS by sampling a uniform root and independent random neighbors at each vertex; this induces a distribution over DFS sequences we denote S DFS ( G ) . RSNN collects m independent searches to form the set P m ( S DFS ( G )) = { S 1 , . . . , S m } . Once these searches are obtained, RSNNs leverage all the advances of RWNNs but with new benefits. We apply the recording function (Equation (1)) to each search, including positional encodings from Tönshoff et al. [5] to distinguish between disconnected nodes and true connections in the sequence. Search embeddings are then processed with a sequence model and the node aggregation function (Equation (2)) (Figure 2).

Figure 2: Overview of an RSNN layer. Starting from an input graph, m random depth-first searches are extracted and encoded via f emb. Additional positional encodings indicate discontinuities in the sequence (e.g., -· -in search 1). These sequences are processed by a sequence model f seq, and final node representations are aggregated across sequences using f agg. We highlight in blue the flow of a selected node representation (shown as ) as it is tracked through each stage of the RSNN layer.

<!-- image -->

## 4.2 From Efficient Graph Coverage to Universal Approximation in RSNNs

In this section, we establish the theoretical foundations of RSNNs by demonstrating how our random search strategy efficiently obtains full graph coverage. Central to our analysis is the observation that the subgraphs induced by DFS sequences are spanning trees. Leveraging this insight, we prove the following key lemma showing that for sparse graphs with bounded degree, a logarithmic number of random searches is sufficient to guarantee full node and edge coverage with high probability.

Lemma 4.1 (Logarithmic Sampling Yields Full Edge Coverage) . Let G = ( V, E ) be a connected graph with | E | ≤ C | V | for some constant C and a bounded maximum degree d max . Let S 1 , S 2 , . . . , S m be m independent random searches sampled from G , and let T 1 , T 2 , . . . , T m be their corresponding induced spanning trees. Then, for small δ ≪ 1 , if the union of T 1 , T 2 , . . . , T m contains every edge in E with probability at least 1 -δ .

<!-- formula-not-decoded -->

In contrast to RWNNs, which require m = O ( | V | ) random walks of length ℓ = O ( | V | ) , RSNNs achieve complete coverage with m = O (log( | V | )) searches of length ℓ = O ( | V | ) . With full node and edge coverage, RSNNs are able to capture all the information necessary to represent any function on graphs. Intuitively, this means that under our sampling strategy, RSNNs are universal approximators: they can approximate any graph function arbitrarily well, provided they are paired with a universal sequence model such as transformers or LSTMs [26, 27].

Theorem 4.2 (Universal Approximation by RSNNs on Sparse Graphs with Bounded Degree) . Let ϵ &gt; 0 and let f : G → R d be any continuous graph-level function, where G is the space of sparse graphs with | E | = O ( | V | ) , maximum degree at most d max , and size at most | V | ≤ n max . Assume m satisfies Equation (3) , so that full coverage is achieved with probability at least 1 -δ . Then, with probability at least 1 -δ there exists an RSNN configuration such that

<!-- formula-not-decoded -->

## 4.3 From Expressivity to Invariance: Isomorphism Invariance of RSNNs

Having established the expressive capabilities of RSNNs, we now turn to invariance. For graphs, the target symmetry is isomorphism invariance: for all G ∼ = H , an isomorphism-invariant graph function satisfies f ( G ) = f ( H ) . Graph functions that capture the symmetry enjoy learning and generalization benefits. Because RSNNs are randomized functions, we adopt the notion of probabilistic invariance [7,

28]: for all G ∼ = H , the random outputs satisfy f ( G ) d = f ( H ) . Intuitively, a randomized graph function is probabilistically invariant to graph isomorphisms if its distribution is unchanged under any graph isomorphism. We demonstrate that the randomized DFS procedure used by RSNNs is probabilistically invariant; consequently, the RSNN predictor f RSNN is invariant in distribution, and its expectation Φ( G ) := E [ f RSNN ( G )] is an isomorphism-invariant function on graphs.

Theorem 4.3 (Probabilistic Isomorphism-Invariance of RSNN) . A randomized search procedure on a graph G produces a sequence S G = ( s G 0 , . . . , s G | V ( G ) | ) of visited vertices. We say the procedure is

probabilistically invariant to graph isomorphisms if for all graph isomorphisms π ,

<!-- formula-not-decoded -->

Learning the invariance. In addition to being invariant in expectation, we show that RSNNs can learn the optimal invariant predictor throughout training even under limited sampling budgets, where the expectation is only approximated (e.g., m = 1 sampled search for each forward pass in the parameter update). At inference, the invariant predictor can then be computed exactly or estimated by the Monte Carlo estimator. Our result follows Murphy et al. [29, 30]. For RSNN parameters W , define the model output on a graph G and a sampled search set S ∼ S DFS ( G ) as f RSNN ( G,S ; W ) .

The randomized DFS procedure used in RSNNs satisfies the above definition. Hence, RSNNs satisfy probabilistic invariance: for all G ∼ = H , f RSNN ( G ) d = f RSNN ( H ) , and the averaged predictor Φ( G ) := E [ f RSNN ( G ) ] is an invariant function on graphs: Φ( G ) = Φ( H ) for all G ∼ = H .

Corollary 4.4 (SGD converges to the invariant objective) . Let ℓ ( · , y ) be differentiable and define

At each step t , sample a mini-batch B t = { ( G ( i ) t , y ( i ) t ) } B i =1 i.i.d. from D and, for each i , draw a single S ( i ) t ∼ S DFS ( G ( i ) t ) independently of W t ; update

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then E [ 1 B ∑ B i =1 ∇ W ℓ ( f RSNN ( G ( i ) t , S ( i ) t ; W t ) , y ( i ) t ) ] = ∇ W L ( W t ) , i.e., the mini-batch gradient is an unbiased estimator of ∇ L ( W t ) . Under standard SGD conditions, W t converges almost surely to an optimizer W ⋆ of the invariant objective.

Inference. Given a fixed point W ⋆ and a new test graph G ′ , the invariant prediction is E S [ f RSNN ( G ′ , S ; W ⋆ )] , which can be exactly computed or approximated with the estimator 1 m ∑ m j =1 f RSNN ( G ′ , S j ; W ⋆ ) where S 1 , . . . , S m i.i.d. ∼ S DFS ( G ′ ) .

## 4.4 Runtime Complexity

We compare the sampling costs of RSNNs and RWNNs. In our approach, each random search corresponds to a DFS traversal. Assuming a sparse graph, a single DFS has a worst-case cost of O ( | V | ) , and obtaining m searches requires O ( m | V | ) time, efficient and computationally feasible in small to medium-sized graphs. In contrast, RWNNs generate m random walks of length ℓ , with total sampling cost O ( mℓ ) . When ℓ ≪| V | , random walk sampling can be faster than random search extraction. However, as we have shown, short walks fail to capture global structure, leading to reduced expressivity. Thus, while RSNN sampling is more expensive when ℓ is small, its increased coverage and performance can justify its cost, especially in graphs where full structure is critical.

## 5 Experiments &amp; Results

Through empirical evaluation we aim to answer the following research questions, extending our theory by testing RSNNs on datasets with factors not explicitly addressed in the theoretical analysis (e.g., class imbalance, rich node features), and testing RSNNs against models beyond our theory such as canonical approaches (e.g., SMILES, Fingerprints) used commonly in molecular analysis.

- RQ1 (Discriminative performance): How does RSNN discriminative performance compare to standard baselines and RWNNs across sparse graph benchmark tasks?
- RQ2 (Node and edge coverage): Do RSNNs achieve higher node and edge coverage than RWNNs as the number of sampled searches m increases, and does this increased coverage translate into improved task performance?
- RQ3 (Generalization to larger and denser graphs): How do RSNNs perform on larger and denser graphs, where attaining full edge coverage is computationally expensive?

## 5.1 Experimental Setup

Datasets. We focus our analysis on molecular and protein benchmarks, domains where RWNNs have demonstrated strong empirical performance and where efficient coverage, long-range dependencies, and high expressivity are essential [5, 7, 31]. Importantly, RSNNs are not intended as a solution across all domains, but as a principled alternative for sparse graphs requiring representations that capture global structure. Specifically, we evaluate on four small-scale molecular graph classification datasets from MoleculeNet [32]: CLINTOX , SIDER , TOX21 , and BBBP . These benchmarks span diverse molecular tasks such as toxicity and adverse reaction prediction, with graph sizes ranging from tens to hundreds nodes. We also include four protein graph classification datasets from ProteinShake [33]: EC Subclass , EC Mechanism , SC Class , and SC Family . Protein graphs are significantly larger than molecules, ranging up to thousands of nodes, making it more difficult to capture global structure. To assess scalability, we evaluate on large-scale molecular benchmarks with hundreds of thousands of graphs from Open Graph Benchmark [34]: PCBA-1030 , PCBA-1458 , and PCBA-4467 . Lastly, to test generalization to larger and denser graphs, we evaluate on NeuroGraph-task , a brain graph benchmark, where the task is to predict one of seven mental states (e.g., emotion processing). We provide descriptive statistics for all datasets in Tables 1 and 2.

Baselines. First, we compare to standard molecular learning baselines: (1) SMILES , a sequence model applied to canonical SMILES [35]; (2) GCN [36] and (3) GIN [15], message-passing GNNs; and (4) GT [12], a graph transformer model. In addition, we compare to (5) Fingerprint , a multilayer perceptron trained on hand-crafted chemical descriptors known to be effective in molecular tasks [37]. Importantly, SMILES and Fingerprint are not applicable in protein graphs. Second, we consider four RWNN variants as baselines for comparison: (6) RWNN-base , which employs uniform random walks of length ℓ with mean aggregation over walk representations [4]; (7) RWNN-anon , which augments the base model with a node anonymization strategy from Wang and Cho [3]; (8) RWNN-mdlr , which uses minimum-degree local rule walks from [7], anonymization, and mean aggregation; (9) CRAWL [5], which applies non-backtracking walks with node-level aggregation. We consider three sequence models for f seq: (a) GRU [38], (b) LSTM[26], and (c) transformer [27].

Training and Evaluation. To ensure fair comparisons, all RWNNs and RSNN are configured with the same number of samples m , and RWNN walk lengths are set to ℓ = | V | , the number of nodes per graph, so that asymptotic runtimes are equivalent across methods. On molecular benchmarks, we sample a new set of m walks for each forward pass during training, and on protein benchmarks, we precompute the set of m walks before training. Following each dataset's protocol, performance is computed as AUC or accuracy. We report median (min, max) performance over five random splits (60/20/20), which is more robust than mean and standard deviation for small sample sizes. All models are trained on a machine equipped with 8 × NVIDIA GeForce GTX 1080 Ti GPUs; if a model does not converge within 24 hours, we omit it from evaluation. All remaining details are in Appendix D 1 .

## 5.2 RQ1 &amp; RQ2: Discriminative Performance and Coverage

First, RSNNs significantly outperform standard baselines across all benchmarks, demonstrating their effectiveness for molecular and protein learning (Table 1). Notably, at m = 16 , RSNNs match or exceed the performance of Fingerprint models, which do not rely on learned representations and instead use features designed by domain experts. For all RWNNs and RSNN, we present results using GRU, which performs best empirically, and include additional results for LSTMs and transformers in the Appendix C, where we observe similar trends. Compared to existing RWNNs, RSNNs exhibit greater expressivity at low sampling budgets; with a single search ( m = 1 ), RSNN significantly outperforms all RWNN variants at the same budget. Moreover, across all molecular benchmarks, RSNNs at m = 1 match or exceed the best-performing RWNNs at m = 16 , highlighting their sample efficiency. While performance differences narrow at m = 16 on molecular benchmarks, RSNNs retain a substantial lead on larger protein graphs, underscoring their expressivity in structurally complex settings. On large-scale molecular benchmarks, training both RWNNs and RSNNs with m&gt; 1 becomes computationally infeasible, exceeding the 24-hour time budget. At m = 1 , however, RSNNs maintain strong performance and substantially outperform RWNNs (Table 2), demonstrating RSNNs' robustness under sampling constraints when computation is limited.

1 Code can be found at:

https://github.com/MLD3/RandomSearchNNs

Table 1: Median (min, max) of performance across test splits on molecular and protein benchmarks. We highlight in blue the best model for each value of m . We use "-" to indicate when a method is not applicable (Fingerprint/SMILES) or when training exceeds 24 hours (GT). RSNNs consistently outperform all RWNN variants at m = 1 . While RWNNs approach RSNN performance on molecular benchmarks at m = 16 , RSNNs outperform RWNNs across all m on protein benchmarks.

|        |                                                                      | Small Scale Molecular Benchmarks                                                                                              | Small Scale Molecular Benchmarks                                                                            | Small Scale Molecular Benchmarks                                                          | Small Scale Molecular Benchmarks                                                            | Protein Benchmarks (ACC ↑ )                                                               | Protein Benchmarks (ACC ↑ )                                                     | Protein Benchmarks (ACC ↑ )                                                               | Protein Benchmarks (ACC ↑ )                                                               |
|--------|----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
|        | Avg. &#124; E &#124;                                                 | CLINTOX 1.5K 26.1 28.0 2                                                                                                      | SIDER 1.5K 33.6 35.4 2                                                                                      | BBBP 2K 23.9 26.0                                                                         | TOX21 8K 18.6 16.9                                                                          | SC CL 10K 217.5 593.8 5                                                                   | SC FAM 10K 217.5 593.8 1000                                                     | EC SUB 15K 304.9 843.4 24                                                                 | ECMEC 15K 306.4 846.9 31                                                                  |
| NA     | # Graphs Avg. &#124; V &#124; # Classes Fingerprint SMILES GT (full) | (52.3, 74.9) (45.7, 68.6) (46.5, 73.5)                                                                                        | 70.4 (66.6, 74.5) 61.5 (57.6, 66.4) 64.3 (57.9, 69.0) 64.2 (62.4, 70.3)                                     | 2 (83.4, (65.5, (62.6,                                                                    | 2 79.1 (75.1, 81.0) 71.3 (66.4, 73.8) 67.8 (64.8, 73.9) 67.5 (63.1, 71.9) 66.9 (64.6, 73.4) | - - - 63.4 (62.8, 64.9) 68.0 (67.9,                                                       | - - - 3.9 (1.1,                                                                 | - - - 31.2 (28.0, 33.1)                                                                   | - - - 52.8 (51.9,                                                                         |
| m = 1  | GCN GIN CRAWL RSNN                                                   | 66.5 62.5 57.1 62.4 (56.9, 74.7) 59.7 (54.1, 72.4)                                                                            | 66.5 (64.0, 69.9)                                                                                           | 86.2 92.5) 71.9 75.3) 75.8 84.0) 73.9 (68.9, 81.4) 75.3 (49.4, 85.3) 74.1 (56.7, 82.8)    | 71.5 (68.8, 76.3) 71.2 (69.3, 75.0) 70.8 (66.6, 75.3)                                       | 69.2) 44.5 (42.9, 45.4) 45.4 (41.5, 45.9) 43.3 (42.9, 45.1)                               | 5.3) 10.4 (8.7, 11.7) 2.2 (1.6, 2.8) 4.6 (4.2, 5.8)                             | 37.2 (33.5, 38.3) 26.7 (24.8, 27.9) 26.9 (26.0, 28.7) 26.7 (26.5, 27.2) 28.7 (27.6, 29.6) | 53.1) 57.4 (56.1, 59.5) 47.3 (46.1, 47.1 (45.6,                                           |
|        | RWNN-base RWNN-anon RWNN-mdlr (ours) RWNN-base                       | 71.0 (54.9, 79.5) 68.2 (52.5, 87.2) 70.7 (60.4, 76.1) 70.0 (64.6, 73.6) 88.1 (84.9, 91.5) 83.6 (76.5, 86.7) 84.7 (80.3, 89.5) | 62.5 (55.9, 67.3) 64.1 (57.0, 67.3) 59.8 (57.0, 65.9) 64.2 (56.1, 67.2) 66.2 (63.0, 72.4) 64.4 (59.9, 71.9) | 74.8 (69.0, 82.6) 76.1 (72.1, 81.6) 77.6 (68.8, 81.5) 87.5 (80.3, 89.9) 84.2 (77.2, 87.0) | 71.7 (66.4, 75.3) 79.8 (77.2, 83.4) 76.3 (71.9, 80.9) 77.2 (73.5, 79.2)                     | 53.0 (50.7, 53.4) 62.2 (60.0, 65.6) 53.0 (52.5, 52.7 (51.7,                               | 4.5 (3.7, 4.7) 5.2 (3.4, 5.8) 13.9 (10.6, 14.9)                                 | 36.8 (36.5, 38.3)                                                                         | 48.4) 48.2) 47.2 (46.0, 48.2) 47.0 (46.2, 47.6) 49.8 (48.2, 50.8)                         |
| m = 4  | RWNN-anon RWNN-mdlr CRAWL RSNN (ours)                                | 82.9 (77.9, 90.4) 83.0 (76.6, 91.5)                                                                                           | 65.6 (61.5, 68.8) 65.5 (60.4, 72.4) 65.2 (59.5, 71.3) 67.0 (61.3, 71.1)                                     | 82.0 (77.1, 85.4) 81.9 (79.2, 88.0) 84.5 (80.7, 87.0) 88.0 (80.3, 90.5)                   | 76.9 (72.6, 80.2) 77.6 (75.6, 81.2) 80.3 (77.3, 84.2)                                       | 54.1) 53.1) 51.5 (50.2, 52.5) 67.0 (66.6, 67.9) 71.7 (70.5, 73.8)                         | 3.7 (3.3, 5.4) 6.4 (5.2, 7.5) 6.2 (5.4, 7.8) 10.8 (9.5, 11.4) 15.5 (14.4, 19.2) | 32.7 (32.1, 34.5) 32.9 (31.2, 34.2) 32.4 (30.6, 33.6) 38.2 (37.0, 39.9) 43.9 (41.7, 44.3) | 48.1 (47.1, 48.8) 47.9 (46.5, 50.3) 48.2 (47.3, 49.3) 50.7 (49.9, 51.7)                   |
|        |                                                                      | 89.1 (80.9, 91.7) (82.6, 88.7)                                                                                                | 65.2 (62.8, 70.2) 67.8 (60.3, 70.7) 64.9 (61.8, 69.1)                                                       | 84.1 (81.0, 91.1) 83.9 (78.2, 85.3) 84.9 (81.5, 86.7)                                     | 78.3 (72.1, 81.3) 78.9 (76.1, 82.0) (75.0, 79.0)                                            | 57.0 (55.5, 58.5) 55.0 (53.5, 58.4)                                                       | 6.1 (4.3, 6.9) 9.3 (8.6, 10.0)                                                  |                                                                                           | 54.8 (51.7, 55.8)                                                                         |
| m = 8  | RWNN-base RWNN-anon RWNN-mdlr CRAWL                                  | 85.0 86.6 (81.8, 92.7) 83.9 (78.0, 87.5) 86.5 (83.6, 91.4) 88.3 (80.1, 91.3)                                                  | 66.1 (62.1, 69.9) 67.6 (63.3, 69.2)                                                                         | 86.0 (82.8, 89.6) 88.6 (83.6, 90.3)                                                       | 77.6 79.1 (76.7, 82.1) 82.2 (77.3, 85.3)                                                    | 54.9 (52.1, 56.9) 72.7 (71.7, 73.3) 74.4 (74.1, 75.4)                                     | 9.2 (8.4, 10.7) 14.1 (10.2, 17.6) 16.0 (14.5, 19.2)                             | 35.5 (34.8, 36.9) 36.2 (35.8, 37.0) 35.5 (34.5, 36.7) 43.7 (43.0, 45.4)                   | 49.7 (48.2, 52.0) 49.3 (48.8, 49.6 (48.3, 54.7 (51.6,                                     |
|        | RSNN (ours)                                                          | 87.8 (82.6, 91.1) 85.9 (81.7, 91.8) 85.9 (81.5, 89.9)                                                                         | 67.2 (64.6, 71.4) 66.5 (61.1, 69.3) 65.7 (63.5, 70.1) 65.3 (61.4, 70.8)                                     | 86.0 (83.7, 85.8 (80.1, 85.4 (80.8, 87.0 (81.7, 89.4 (83.0,                               | (75.6, 81.8) 79.2 (75.9, 82.2) 79.1 (77.7, 83.0) 80.9 (77.4, 82.6) 82.2 (78.0, 84.1)        | 59.0 (58.4, 60.2) 60.1 (58.3, 61.5) 59.5 (56.7, 61.0) 76.2 (73.6, 77.4) 77.0 (75.0, 77.2) | 10.9 (9.6, 11.4) 10.2 (8.1, 12.4) 11.2 (9.4, 11.7) 15.5 (13.6, 16.0) 19.0       | 46.3 (46.0, 49.4) 37.2 (36.1, 39.3)                                                       | 50.3) 51.8) 55.0) 57.1 (56.5, 57.7) 51.7 (51.4, 53.0) 51.7 (50.4, 53.2) 51.3 (49.9, 51.9) |
| m = 16 | RWNN-base RWNN-anon RWNN-mdlr CRAWL RSNN (ours)                      | 89.1 (80.5, 91.1) 88.5 (82.0, 93.7)                                                                                           | 67.1 (65.0, 74.0)                                                                                           | 88.1) 88.1) 90.5) 90.3) 91.7)                                                             | 80.0                                                                                        |                                                                                           | (15.3, 20.1)                                                                    | 39.3 (38.5, 40.6) 39.1 (38.4, 40.2) 48.7 (46.1, 49.3) 50.0 (49.5, 52.0)                   | 57.4 (56.8, 58.6) 59.5 (57.1, 60.0)                                                       |

We compare how node/edge coverage and performance varies with the number of walks or searches for RSNNs and CRAWL, the strongest RWNN baseline (Figure 3). Across all benchmarks, we observe a strong correlation between coverage and model performance. On molecular graphs, RSNNs achieve full node and high edge coverage with a single search ( m = 1 ), resulting in strong initial performance. This aligns with our theoretical analysis: each RSNN search guarantees node coverage by construction, and

Table 2: Median (min, max) AUC on large scale molecular benchmarks. We highlight in blue the best model. RSNNs outperform all RWNNs across all tasks.

|       |                      | Large Scale Molecular Benchmarks (AUC ↑ )             | Large Scale Molecular Benchmarks (AUC ↑ )             | Large Scale Molecular Benchmarks (AUC ↑ )             |
|-------|----------------------|-------------------------------------------------------|-------------------------------------------------------|-------------------------------------------------------|
|       | # Graphs             | PCBA-1030 160K                                        | PCBA-1458 195K                                        | PCBA-4467 240K 25.3                                   |
|       | Avg. &#124; V &#124; | 24.3                                                  | 25.1                                                  |                                                       |
|       | Avg. &#124; E &#124; | 26.2                                                  | 27.1                                                  | 27.2                                                  |
| m = 1 | RWNN-mdlr CRAWL RSNN | 63.5 (62.3, 64.3) 64.2 (62.5, 64.5) 78.8 (78.1, 79.3) | 76.2 (75.4, 76.7) 77.0 (76.8, 77.2) 87.0 (86.7, 87.4) | 75.4 (75.4, 76.0) 75.6 (75.2, 75.7) 85.2 (84.3, 85.3) |

only a few searches are needed to achieve full edge coverage in sparse graphs. In contrast, CRAWL begins with low node and edge coverage and only reaches RSNN-level performance at m = 16 , once coverage converges, highlighting RWNN limitations under small sampling budgets. On larger protein graphs, both coverage and performance improve more gradually, but RSNNs retain a consistent performance advantage across all m , underscoring the benefit of efficient coverage in larger graphs.

## 5.3 RQ3: Generalization to Larger and Denser Graphs

Table 3: Median (min, max) of accuracy on NeuroGraph benchmark. We highlight in blue the best model. RSNNs outperform CRAWL across m = 4 , 16 .

|                      | NeuroGraph-task   |                    |                   |
|----------------------|-------------------|--------------------|-------------------|
| # Graphs             |                   | 7500               |                   |
| Avg. &#124; V &#124; |                   | 400                |                   |
| Avg. &#124; E &#124; |                   | 7029               |                   |
| Max degree           |                   | 153                |                   |
|                      | m = 1             | m = 4              | m = 16            |
| CRAWL                | 63.4 (59.4, 64.5) | 77.5, (74.4, 78.9) | 68.3 (30.1, 87.9) |
| RSNN                 | 58.9 (57.1, 61.5) | 80.4 (78.8, 82.6)  | 86.5 (76.5, 88.9) |

To assess generalization beyond small and sparse regimes, we evaluate on a NeuroGraph benchmark of brain graphs. These graphs are substantially larger than molecules and denser than proteins, making full edge coverage expensive for both walks and searches. We compare RSNN against CRAWL. RSNN outperforms CRAWL at m = 4 , 16 , indicating that RSNNs can leverage structure even when full coverage is expensive and that their performance advantage remains on larger and denser graphs.

<!-- image -->

(a) BBBP Molecular Graph

(b) Structural Family Protein Graph

(c) Enzyme Subclass Protein Graph

Figure 3: Coverage vs. performance across benchmarks. RSNNs achieve higher coverage and performance at low sample sizes, while CRAWL only approaches RSNN coverage and performance at m = 16 , highlighting a strong correlation between coverage and performance.

## 6 Discussion and Conclusion

We present the first theoretical analysis of RWNNs under realistic sampling constraints, showing that their expressivity is fundamentally limited without full node and edge coverage, even in small graphs. We prove that under partial coverage, RWNNs are strictly less expressive than traditional MPNNs. To address this, we introduce RSNNs, which use random depth-first search to guarantee full node coverage and edge coverage with only a logarithmic number of samples in sparse graphs. When paired with expressive sequence models, we show that RSNNs are universal approximators. Furthermore, RSNNs are also probabilistically invariant to graph isomorphisms. Empirically, RSNNs consistently outperform RWNNs on both molecular and protein benchmarks, requiring up to 16 × fewer samples to achieve comparable performance.

Our work builds on recent work in RWNNs that combines random walks with expressive sequence models [3-8]. These works explore various walk strategies, including uniform walks [3, 4], nonbacktracking walks [5, 6], minimum-degree local rule walks [7], and learnable walks [8], and propose architectural improvements to enhance expressivity and performance. We critically examine the expressivity of RWNNs under realistic sampling constraints, relaxing prior assumptions that walks are as long as cover times. Based on our analysis, we propose to replace random walks entirely with random searches, leading to RSNNs, a more sample-efficient and expressive alternative.

Our work is not without limitations. In particular, RSNNs are tailored to sparse, medium-sized graphs. How to scale RSNN to extremely large, densely connected graphs remains an open question. In such settings, full-depth searches may become prohibitively expensive, and edge coverage may scale less efficiently. A promising direction is to explore truncated searches that capture key structural signal while reducing computation. This raises new questions about how coverage and expressivity behave under partial searches, particularly in dense regimes where full coverage is infeasible.

Despite the focused scope, our results are promising: RSNNs match or exceed RWNN performance with significantly fewer samples and maintain a clear advantage across benchmarks. These findings underscore the value of replacing random walk sampling with search-based sampling in graph learning. More broadly, this work highlights the importance of moving beyond local neighborhoods toward sampling strategies that capture global structure. By leveraging efficient coverage through random searches, RSNNs offer a principled, expressive, and sample-efficient framework for learning on sparse graphs, laying the foundation for future exploration in other settings.

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research, Department of Energy Computational Science Graduate Fellowship under Award Number DE-SC0023112. It was also partially supported by National Science Foundation under Grants No. IIS 2212143 and IIS 2504090. We thank the anonymous reviewers and members of the MLD3 lab for their valuable feedback.

## References

- [1] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. Deepwalk: Online learning of social representations. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining , 2014.
- [2] Aditya Grover and Jure Leskovec. node2vec: Scalable feature learning for networks. In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining , 2016.
- [3] Yuanqing Wang and Kyunghyun Cho. Non-convolutional graph neural networks. In Advances in Neural Information Processing Systems , 2024.
- [4] Yanchao Tan, Zihao Zhou, Hang Lv, Weiming Liu, and Carl Yang. Walklm: A uniform language model fine-tuning framework for attributed graph embedding. In Advances in neural information processing systems , 2023.
- [5] Jan Tönshoff, Martin Ritzert, Hinrikus Wolf, and Martin Grohe. Walking out of the weisfeiler leman hierarchy: Graph learning beyond message passing. Transactions in Machine Learning Research , 2023.
- [6] Dexiong Chen, Till Hendrik Schulz, and Karsten Borgwardt. Learning long range dependencies on graphs via random walks. In International Conference on Learning Representations , 2025.
- [7] Jinwoo Kim, Olga Zaghen, Ayhan Suleymanzade, Youngmin Ryou, and Seunghoon Hong. Revisiting random walks for learning on graphs. In International Conference on Learning Representations , 2025.
- [8] Karolis Martinkus, Pál András Papp, Benedikt Schesch, and Roger Wattenhofer. Agent-based graph neural networks. In International Conference on Learning Representations , 2023.
- [9] Qimai Li, Zhichao Han, and Xiao-Ming Wu. Deeper insights into graph convolutional networks for semi-supervised learning. In Proceedings of the AAAI conference on artificial intelligence , volume 32, 2018.
- [10] Jiong Zhu, Yujun Yan, Lingxiao Zhao, Mark Heimann, Leman Akoglu, and Danai Koutra. Beyond homophily in graph neural networks: Current limitations and effective designs. In Advances in Neural Information Processing Systems , 2020.
- [11] Michael Ito, Danai Koutra, and Jenna Wiens. Understanding gnns and homophily in dynamic node classification. In International Conference on Artificial Intelligence and Statistics . PMLR, 2025.
- [12] Vijay Prakash Dwivedi and Xavier Bresson. A generalization of transformer networks to graphs. In AAAI Workshop on Deep Learning on Graphs: Methods and Applications , 2021.
- [13] Ladislav Rampášek, Michael Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Dominique Beaini. Recipe for a general, powerful, scalable graph transformer. In Advances in Neural Information Processing Systems , 2022.
- [14] Michael Ito, Jiong Zhu, Dexiong Chen, Danai Koutra, and Jenna Wiens. Learning laplacian positional encodings for heterophilous graphs. In International Conference on Artificial Intelligence and Statistics . PMLR, 2025.
- [15] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In International Conference on Learning Representations , 2019.
- [16] Xiyuan Wang and Muhan Zhang. How powerful are spectral graph neural networks. In International Conference on Machine Learning , 2022.
- [17] Liheng Ma, Chen Lin, Derek Lim, Adriana Romero-Soriano, Puneet K Dokania, Mark Coates, Philip Torr, and Ser-Nam Lim. Graph inductive biases in transformers without message passing. In International Conference on Machine Learning , 2023.

- [18] Romas Aleliunas, Richard M Karp, Richard J Lipton, László Lovász, and Charles Rackoff. Random walks, universal traversal sequences, and the complexity of maze problems. In 20th Annual Symposium on Foundations of Computer Science (sfcs 1979) , pages 218-223. IEEE Computer Society, 1979.
- [19] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In International Conference on Machine Learning , 2017.
- [20] Ryoma Sato. A survey on the expressive power of graph neural networks. arXiv preprint arXiv:2003.04078 , 2020.
- [21] Waiss Azizian and Marc Lelarge. Expressive power of invariant and equivariant graph neural networks. In International Conference on Machine Learning , 2021.
- [22] László Lovász. Random walks on graphs. Combinatorics, Paul erdos is eighty , 2(1-46):4, 1993.
- [23] Roee David and Uriel Feige. Random walks with the minimum degree local rule have o(nˆ2) cover time. SIAM Journal on Computing , 47(3):755-768, 2018.
- [24] Ming Chen, Zhewei Wei, Zengfeng Huang, Bolin Ding, and Yaliang Li. Simple and deep graph convolutional networks. In International conference on machine learning , pages 1725-1735. PMLR, 2020.
- [25] Kenta Oono and Taiji Suzuki. Graph neural networks exponentially lose expressive power for node classification. In International Conference on Learning Representations , 2020.
- [26] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation , 9(8): 1735-1780, 1997.
- [27] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems , 30, 2017.
- [28] Benjamin Bloem-Reddy and Yee Whye Teh. Probabilistic symmetries and invariant neural networks. Journal of Machine Learning Research , 21(90):1-61, 2020.
- [29] Ryan L Murphy, Balasubramaniam Srinivasan, Vinayak Rao, and Bruno Ribeiro. Janossy pooling: Learning deep permutation-invariant functions for variable-size inputs. In International Conference on Learning Representations , 2018.
- [30] Ryan Murphy, Balasubramaniam Srinivasan, Vinayak Rao, and Bruno Ribeiro. Relational pooling for graph representations. In International Conference on Machine Learning , pages 4663-4673. PMLR, 2019.
- [31] Vijay Prakash Dwivedi, Ladislav Rampášek, Michael Galkin, Ali Parviz, Guy Wolf, Anh Tuan Luu, and Dominique Beaini. Long range graph benchmark. In Advances in Neural Information Processing Systems , 2022.
- [32] Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S Pappu, Karl Leswing, and Vijay Pande. Moleculenet: a benchmark for molecular machine learning. Chemical science , 9(2):513-530, 2018.
- [33] Tim Kucera, Carlos Oliver, Dexiong Chen, and Karsten Borgwardt. Proteinshake: Building datasets and benchmarks for deep learning on protein structures. In Advances in Neural Information Processing Systems , 2023.
- [34] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. Advances in neural information processing systems , 33:22118-22133, 2020.
- [35] David Weininger. Smiles, a chemical language and information system. 1. introduction to methodology and encoding rules. Journal of chemical information and computer sciences , 28 (1):31-36, 1988.

- [36] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations , 2017.
- [37] David Rogers and Mathew Hahn. Extended-connectivity fingerprints. Journal of chemical information and modeling , 50(5):742-754, 2010.
- [38] Kyunghyun Cho, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoderdecoder for statistical machine translation. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing , 2014.
- [39] Christopher Morris, Martin Ritzert, Matthias Fey, William L Hamilton, Jan Eric Lenssen, Gaurav Rattan, and Martin Grohe. Weisfeiler and leman go neural: Higher-order graph neural networks. In Proceedings of the AAAI conference on artificial intelligence , volume 33, pages 4602-4609, 2019.
- [40] Christopher Morris, Gaurav Rattan, and Petra Mutzel. Weisfeiler and leman go sparse: Towards scalable higher-order graph embeddings. Advances in Neural Information Processing Systems , 33:21824-21840, 2020.
- [41] Nils M Kriege. Weisfeiler and leman go walking: Random walk kernels revisited. Advances in Neural Information Processing Systems , 35:20119-20132, 2022.
- [42] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations , 2015.
- [43] Luis Müller and Christopher Morris. Aligning transformers with weisfeiler-leman. In International Conference on Learning Representations , 2024.
- [44] Luis Müller, Daniel Kusuma, Blai Bonet, and Christopher Morris. Towards principled graph transformers. Advances in Neural Information Processing Systems , 37:126767-126801, 2024.
- [45] Caterina Graziani, Tamara Drucks, Fabian Jogl, Monica Bianchini, Franco Scarselli, T Gartner, et al. The expressive power of path-based graph neural networks. In International Conference on Machine Learning , volume 235, pages 16226-16249. ML Research Press, 2024.

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (1-2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

## IMPORTANT, please:

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Yes, they are properly reflected.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Yes, we discuss limitations of the work in Section 6.

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

Justification: Yes, all theoretical results are clearly stated and we place all the mathematical proofs in the Appendix.

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

Justification: All experimental results are described in the main paper and the Appendix

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

Justification: Yes, all data are publicly available and code is available at:

https://github.com/MLD3/RandomSearchNNs

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

Justification: We leave all training and test details in the main paper and the Appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All error bars are explained in Section 5

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

Justification: Yes, all details are in the Appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The paper conforms in every respect to the code of ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our work is a general graph representation learning framework with no positive/negative societal impacts beyond any general machine learning framework for graphs (message-passing graph neural networks, graph transformers, random walk neural networks).

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

Justification: No new data or models are released with a high risk of misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Yes, all assets are properly credited.

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

Justification: All assets are properly documented.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: No crowdsourcing or research with human subjects was used.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: All data used in this work are publicly available datasets.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: LLMs do not impact the core methodology of the research.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Mathematical Proofs

## A.1 Random Walk Neural Network Expressive Power

We begin with the 1-Weisfeiler-Lehman (WL) color refinement, which iteratively updates each node's color by hashing its current color together with the multiset of its neighbors' colors (Section A.1.1). WL is known to upper-bound MPNN expressivity [15, 39]. We then introduce Walk Weisfeiler-Lehman (WWL), a walk-based refinement that updates a node from the multiset of colored walks of length ≤ ℓ that terminates at it (Section A.1.2). We establish monotonicity of WWL in the number of refinement rounds t , the maximum walk length ℓ , and the initialization π (0) (i.e., richer initial features yield finer partitions). We further show that WWL upper-bounds the expressive power of RWNNs (without positional/ID signals). Finally, using unfolding trees , which simultaneously captures the nodes visible to t rounds of message passing and encodes all root-terminating walks of length ≤ t , we prove the main expressivity results that unify MPNNs and RWNNs: equivalence under full coverage and strict separation under partial coverage (Section A.1.3). We provide a more detailed review of existing WL variants and their relation to WWL in Appendix E.

## A.1.1 Weisfeiler-Lehman (WL)

We begin with the 1-dimensional Weisfeiler-Lehman (WL) color-refinement procedure, which upperbounds the expressive power of message-passing GNNs and, as we will show later, also upper-bounds RWNN expressive power. Intuitively, WL iteratively refines node labels by hashing each node's current label together with the multiset of its neighbors' labels.

Definition A.1 (1-WL color refinement) . Let G = ( V, E ) be an unlabeled graph and let N ( u ) denote the neighbors of u ∈ V . Initialize a coloring π (0) WL : V → Σ with a constant value (e.g., π (0) WL ( u ) = 1 for all u ). For t ≥ 0 , update

<!-- formula-not-decoded -->

where Hash is injective and maps pairs of the form (current color, neighbor colors) to Σ . The process stabilizes at the first t ⋆ for which π ( t ⋆ ) WL = π ( t ⋆ +1) WL ; we denote the stable coloring by π ( ∞ ) WL .

To compare two graphs G and H , run 1-WL on each. If the stable color multisets differ (e.g., some color has a different node count), the graphs are certified non-isomorphic. If the stable colorings agree, the test is inconclusive (the graphs may still be non-isomorphic). For the remainder of the analysis, we write α ⪯ β for node-level colorings α, β : V → Σ to mean that β refines α : if β ( u ) = β ( v ) then α ( u ) = α ( v ) . These notions coincide with graph-level distinguishability: applying an injective readout on multiset colors α and β for α ⪯ β yields graph-level functions f α and f β such that f α ⪯ f β . Thus, distinguishability at the node-level translates to distinguishability at the graph-level.

WLhas been used to quantify the expressive power of MPNNs. For any standard message-passing NN, its expressive power is no greater than that of WL. Moreover, if its aggregation function is injective on the multisets of node neighbors, its expressive power matches that of WL.

Lemma A.2 (MPNN vs. 1-WL Expressivity [15, 39]) . Let f MPNN be a MPNN with a permutationinvariant readout, and let π WL denote the 1-WL coloring. Then

<!-- formula-not-decoded -->

Moreover, if the multiset aggregator f agg used by f MPNN is injective, then

<!-- formula-not-decoded -->

## A.1.2 Walk Weisfeiler-Lehman (WWL)

Building on WL, we now align Weisfeiler-Lehman to random walk models by defining a node-level WWLscheme that refines a node's label from the multiset of colored walks incident to it.

Definition A.3 (WWL at length ℓ ) . Let G = ( V, E ) be a graph and ℓ ∈ N . For L ≥ 1 , let

<!-- formula-not-decoded -->

be the set of lengthL walks, and write W ≤ ℓ = ⋃ ℓ ℓ =1 W L , the union of all walks of length ≤ ℓ . For a node u ∈ V , define its terminating-walk neighborhood

<!-- formula-not-decoded -->

Given an initial coloring π (0) : V → Σ (e.g., uniform or from node features), define for any walk col ( t ) WWL ℓ ( W ) = ( π ( t ) WWL ℓ ( w 0 ) , . . . , π ( t ) WWL ℓ ( w L ) ) , the colored walk obtained by applying π ( t ) WWL ℓ to each node in the walk. The WWL ℓ update is, for all u ∈ V ,

<!-- formula-not-decoded -->

Lemma A.4 (Monotonicity in t , ℓ , and π 0 ) . Fix ℓ ≤ ℓ ′ , t ≤ t ′ , and initial colorings π 0 ⪯ π ′ 0 . Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consequently, combining each result yields π ( t ) WWL ℓ ( π 0 ) ⪯ π ( t ′ ) WWL ℓ ′ ( π ′ 0 ) for t ≤ t ′ , ℓ ≤ ℓ ′ , π 0 ⪯ π ′ 0 .

Proof. Monotonicity in t . At each step, π ( t +1) WWL ℓ ( u ) = Hash ( π ( t ) WWL ℓ ( u ) , · ) includes the current color as an input. By injectivity of Hash , if two nodes receive the same new color then they had the same current color. Thus π ( t ) WWL ℓ ⪯ π ( t +1) WWL ℓ , and induction gives the stated inequality for t ≤ t ′ .

Monotonicity in ℓ . Let ℓ ≤ ℓ ′ . For each node u , the multiset of colored terminating walks of lengths ≤ ℓ is obtained from the corresponding multiset for lengths ≤ ℓ ′ by the projection that discards all walks of length &gt; ℓ . Therefore, if two nodes are equal under WWL ℓ ′ , they are equal under WWL ℓ as well. Injectivity of Hash yields π ( t ) WWL ℓ ⪯ π ( t ) WWL ℓ ′ .

Monotonicity in the initialization π 0 . Assume π 0 ⪯ π ′ 0 . Then there exists a color-forgetting map ρ with π 0 = ρ ◦ π ′ 0 . Apply ρ pointwise to every color in each colored walk: for any terminating walk W = ( w 0 , . . . , w L ) ,

( π 0 ( w 0 ) , . . . , π 0 ( w L )) = ( ρ ( π ′ 0 ( w 0 )) , . . . , ρ ( π ′ 0 ( w L )) ) . Hence the multiset of π 0 -colored walks at any node is the image, under this deterministic transformation, of the multiset of π ′ 0 -colored walks. Consequently, equality of the π ′ 0 -based walk multisets implies equality of the π 0 -based walk multisets, and injectivity of Hash gives π (1) WWL ℓ ( π 0 ) ⪯ π (1) WWL ℓ ( π ′ 0 ) . The same argument iterates, since each WWL round recomputes colors from the previous round's coloring via the same construction, yielding π ( t ) WWL ℓ ( π 0 ) ⪯ π ( t ) WWL ℓ ( π ′ 0 ) for all t .

The following lemma is an analogue to expressive results on MPNNs and 1-WL. Intuitively, WWL ℓ upper bounds RWNN expressivity, and RWNNs can match WWL ℓ if their aggregator is injective.

Lemma A.5 (RWNN vs. WWL ℓ Expressivity) . Let f ℓ RWNN be a random walk neural network that, for each node u , aggregates over the multiset of all terminating walks of lengths 1 , . . . , ℓ ending at u , via a permutation-invariant aggregator and a sequence encoder applied to each walk. Let π WWL ℓ denote the WWL ℓ coloring.

1. ( Upper bound ) For any choice of encoders/aggregators,

<!-- formula-not-decoded -->

That is, if two graphs are indistinguishable by WWL ℓ , they are indistinguishable by f ℓ RWNN .

2. ( Tightness under injectivity ) Suppose the sequence encoder f seq is injective on length-aware color sequences and the nodewise multiset aggregator f agg is injective. Then

<!-- formula-not-decoded -->

Proof. (Upper bound). We prove by induction on t that π ( t ) WWL ℓ ( u ) = π ( t ) WWL ℓ ( v ) implies f ℓ,t RWNN ( u ) = f ℓ,t RWNN ( v ) .

Base case t = 0 . Both procedures start from the same initialization (e.g., uniform or fixed features), so the claim holds trivially.

Inductive step. Assume the claim holds at depth t . Take u, v with π ( t +1) WWL ℓ ( u ) = π ( t +1) WWL ℓ ( v ) . By injectivity of the WWL hash, the entire inputs to the hash coincide, hence

<!-- formula-not-decoded -->

where each π ( t ) WWL ℓ ( W ) is the length-aware color sequence along the walk W . Thus there is a bijection between terminating walks at u and v that preserves these sequences. By the induction hypothesis, matched nodes with equal WWL color at round t have equal RWNN representations at depth t . Therefore, for each matched walk pair, the inputs to the per-walk sequence encoder f seq agree elementwise, so per-walk encodings match; applying the same permutation-invariant multiset aggregator f agg yields f ℓ,t +1 RWNN ( u ) = f ℓ,t +1 RWNN ( v ) . This completes the induction and the upper bound.

̸

(Equivalence under injectivity). Assume f seq is injective on length-aware sequences and f agg is injective on multisets. Let u, v satisfy π ( t +1) WWL ℓ ( u ) = π ( t +1) WWL ℓ ( v ) . By injectivity of the WWL hash, either their current colors at round t differ, or their multisets { { π ( t ) WWL ℓ ( W ) : W ∈ W ≤ ℓ ( · ) } } differ. In the first case, including (an injective transform of) the current node state in the RWNN update separates u and v . In the second case, there is no bijection between the two multisets of colored sequences; since f seq is injective on sequences and f agg is injective on multisets, the aggregated RWNN representations must differ at round t +1 . Combining with the upper bound, we conclude f ℓ,t RWNN ≃ π ( t ) WWL ℓ for all t .

## A.1.3 RWNN-MPNN Equivalence Under Full Coverage (Theorem 3.1, Corollary 3.2)

Unfolding Trees. We introduce the unfolding tree from Morris et al. [40], Kriege [41], which makes explicit the bridge between Weisfeiler-Lehman (WL) refinement and random walks. For a node u in G , the unfolding tree at depth ℓ enumerates, with multiplicities, all vertices seen by successive layers of message passing around u . Equivalently, every leaf-to-root path in the unfolding tree corresponds to a walk in G that terminates at u . Hence the unfolding tree simultaneously encodes (i) all messages propagated in a message-passing view and (ii) all terminating walks of length ≤ ℓ at u . We will leverage this structure to relate the expressive power of WL and WWL.

Definition A.6 (Unfolding tree [40, 41]) . Let G = ( V, E ) be a graph, ℓ ∈ N , and u ∈ V . The unfolding tree of depth ℓ rooted at u , denoted T G [ ℓ, u ] , is the rooted tree defined recursively as follows:

- T G [0 , u ] consists of a single root node labeled by u .
- For ℓ ≥ 1 , T G [ ℓ, u ] has root labeled by u ; for each neighbor v ∈ N G ( u ) , attach as a child a fresh copy of T G [ ℓ -1 , v ] .

The first key fact ties WL colors to unfolding trees: WL's ℓ -round color of a node is exactly the isomorphism type of its depthℓ unfolding tree.

Lemma A.7 (WL ↔ unfolding tree [41]) . Let G,H be graphs, u ∈ V ( G ) , v ∈ V ( H ) , and ℓ ≥ 1 . Then

<!-- formula-not-decoded -->

Unfolding trees also capture terminating walks: every leaf-to-root path in T G [ ℓ, u ] reads off a unique lengthℓ walk in G ending at u , and conversely. Let

<!-- formula-not-decoded -->

Lemma A.8 (Leaf-to-root paths ↔ terminating walks [41]) . For any u ∈ V ( G ) and ℓ ≥ 1 ,

<!-- formula-not-decoded -->

Theorem A.9. For any graphs G,H and any ℓ ≥ 1 , WWL ℓ test has exactly the same distinguishing power as the classical 1-dimensional Weisfeiler-Lehman test. Formally,

<!-- formula-not-decoded -->

Proof. π ( ∞ ) WL ⪯ π ( ∞ ) WWL ℓ . For ℓ = 1 , the set of terminating length-1 walks at a node u is exactly its neighbor set N ( u ) . Hence the WWL 1 update coincides with the WL update, and for every round t

<!-- formula-not-decoded -->

By Lemma A.4 (monotonicity in ℓ ), π ( t ) WWL 1 ⪯ π ( t ) WWL ℓ for all ℓ ≥ 1 and all t . Passing to the limit,

<!-- formula-not-decoded -->

π ( ∞ ) WWL ℓ ⪯ π ( ∞ ) WL . Initializing WWL with the WL limit, π (0) = π ( ∞ ) WL , it suffices to show that one WWLupdate makes no further splits. Fix u ∈ V ( G ) and v ∈ V ( H ) with π ( ∞ ) WL ( u ) = π ( ∞ ) WL ( v ) . By Lemma A.7, there is a root-preserving isomorphism σ : T G [ ℓ, u ] ∼ = - - → T H [ ℓ, v ] . By Lemma A.8, leaf-to-root paths in these depthℓ trees biject with the terminating walks of lengths 1 , . . . , ℓ at u and v , respectively; σ also preserves WL ∞ colors at every node of two unfolding trees. To show this, suppose for contradiction, that there exists x ∈ T G [ ℓ, u ] with π ( ∞ ) 1-WL ( x ) = π ( ∞ ) 1-WL ( σ ( x )) . Since 1-WL stabilizes in finite time on the finite disjoint union G ⊎ H , there exists a finite witness round k ⋆ ∈ N such that π ( k ⋆ ) 1-WL ( x ) = π ( k ⋆ ) 1-WL ( σ ( x )) . Let d be the distance from x to the root u in T G [ ℓ, u ] . By the 1-WL update rule, a mismatch at a node at round k ⋆ forces a mismatch at its parent at round k ⋆ +1 (the multiset of child colors differs), and inductively a mismatch at the root after d further rounds:

̸

<!-- formula-not-decoded -->

This contradicts π ( ∞ ) 1-WL ( u ) = π ( ∞ ) 1-WL ( v ) . Hence σ must preserve 1-WL colors at every node. Consequently, the leaf-to-root paths in T G [ ℓ, u ] and T H [ ℓ, v ] correspond bijectively under σ with identical colored sequences. Thus, the corresponding multisets of WL ∞ -colored terminating-walk sequences at u and v coincide. Together with π (0) ( u ) = π (0) ( v ) , the entire inputs to the WWL hash agree at u and v , so by injectivity of Hash we obtain π (1) WWL ℓ ( π ( ∞ ) WL ) ( u ) = π (1) WWL ℓ ( π ( ∞ ) WL ) ( v ) . Hence π ( ∞ ) WL is a fixed point of WWL. By Lemma A.4, WWL is monotone in t and π 0 ; since uniform ⪯ π ( ∞ ) WL , it follows that π ( ∞ ) WWL ℓ ⪯ π ( ∞ ) WL . Combined with π ( ∞ ) WL = π ( ∞ ) WWL 1 ⪯ π ( ∞ ) WWL ℓ , we conclude π ( ∞ ) WWL ℓ ≃ π ( ∞ ) WL .

We now leverage the preceding results to prove the main expressivity statements.

Theorem A.10 (RWNN-MPNN equivalence under full coverage) . Let G be a graph. Let f FC RWNN denote an RWNN with injective f seq and f agg with no additional positional encodings, applied to the complete multiset of walks W ≤ ℓ ( G ) with lengths up to ℓ = C E ( G ) , the edge cover time of G . Let f MPNN be an MPNN with injective f agg . Then, for all graphs G,H ,

<!-- formula-not-decoded -->

Hence, f FC RWNN ≃ f MPNN (i.e., f FC RWNN and f MPNN are equal in expressive power).

Proof. By the standard 1-WL result for message passing (Theorem A.2), an MPNN with injective aggregation satisfies f MPNN ≃ π WL . By the RWNN/WWL correspondence (Lemma A.5), a fullcoverage RWNN with injective f seq and f agg satisfies f FC RWNN ≃ π WWL ℓ . Finally, by the equivalence π WWL ℓ ≃ π WL (Theorem A.9), we conclude f FC RWNN ≃ f MPNN .

Corollary A.11 (RWNNs under partial coverage) . Let f PC RWNN be an RWNN of the same architectural class as in Theorem 3.1, but applied to a multiset of terminating walks that achieves only partial node/edge coverage of the input. Then, for all graphs G,H ,

<!-- formula-not-decoded -->

̸

̸

and there exist non-isomorphic graphs G ̸ ∼ = H such that

̸

<!-- formula-not-decoded -->

Hence f PC RWNN ≺ f MPNN .

Proof. Coverage monotonicity (a direct consequence of injectivity and permutation invariance of the aggregator on multisets) implies that removing walks cannot increase distinguishing power, i.e., f PC RWNN ⪯ f FC RWNN ≃ f MPNN , which yields the implication f MPNN ( G ) = f MPNN ( H ) ⇒ f PC RWNN ( G ) = f PC RWNN ( H ) . For strictness, start from two isomorphic graphs and form G by adding one isolated vertex and H by adding one pendant vertex (a new vertex attached to an existing node). Then 1-WL (hence an MPNN) distinguishes G and H . However, if f PC RWNN is applied to walk multisets that exclude all walks visiting the added vertex in both graphs, the remaining covered walks coincide, so f PC RWNN ( G ) = f PC RWNN ( H ) . Thus f PC RWNN ≺ f MPNN .

## A.2 Random Search Neural Network Expressive Power (Lemma 4.1, Theorem 4.2)

We first establish a coverage lemma: for any edge e = { u, v } in a connected graph G with maximum degree d max , a randomized DFS (uniform start; i.i.d. tie-breaking) includes e in its spanning tree with probability at least 1 /d max , i.e., Pr [ e ∈ T DFS ( G ) ] ≥ 1 /d max . Building on this, we show that sampling O ( d max log | E | ) independent DFS trees suffices to achieve full edge coverage with high probability; in bounded-degree sparse graphs ( d max = O (1) and | E | = Θ( | V | ) ), this reduces to O (log | V | ) searches. Equipped with such full coverage, standard universal components, and shared anonymous integer tags, RSNNs are universal approximators on graphs in the specified family.

Lemma A.12 (Edge inclusion probability under random DFS) . Consider the following random-DFS procedure on a graph G : fix a uniform distribution over the root vertex; independently for each vertex x , draw a uniformly random permutation π x of its neighbors; run depth-first search that, upon first visiting x , explores neighbors in the order π x . Let T DFS be the resulting DFS spanning tree. For an edge e = ( u, v ) , define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Let A be the event that u is discovered by DFS before v . On A , when u is first processed, v is unvisited. The edge ( u, v ) will be taken as a tree edge iff, in the random neighbor order π u , the vertex v appears before all neighbors S u ( e ) that can lead from u to v without using e . The positions of the other neighbors of u are irrelevant: exploring any neighbor not on a path to v first cannot reach v before DFS returns to u . Since π u is a uniform permutation, the probability of this sufficient event is exactly 1 / ( τ u ( e ) + 1) . A symmetric argument on A c (i.e., when v is discovered before u ) gives the bound 1 / ( τ v ( e ) + 1) . Unconditionally,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma A.13 (Logarithmic Sampling Yields Full Edge Coverage) . Let G = ( V, E ) be a connected, unweighted graph with | E | ≤ C | V | for some constant C and a bounded maximum degree d max . Let S 1 , S 2 , . . . , S m be m independent random searches sampled from G , and let T 1 , T 2 , . . . , T m be their corresponding induced spanning trees. Then, for small δ ≪ 1 , if

<!-- formula-not-decoded -->

the union of T 1 , T 2 , . . . , T m contains every edge in E with probability at least 1 -δ .

Proof. By Lemma A.12 the probability that any edge e appears in any DFS is at least p e ≥ 1 d max . Hence, the probability that a single DFS tree does not contain e is at most 1 -p e ≤ 1 -1 d max . Since the spanning trees T 1 , T 2 , . . . , T m are sampled independently, the probability that e is missing from all m trees is at most ( 1 -1 d max ) m . By the union bound over all | E | edges, the probability that there exists at least one edge which is not covered by the union of the m trees is at most

<!-- formula-not-decoded -->

We require this probability to be at most δ :

<!-- formula-not-decoded -->

Taking the natural logarithm on both sides gives:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since ln ( 1 -1 d max ) &lt; 0 , dividing by this term (and reversing the inequality) yields

Thus, with m chosen accordingly, the union of the m spanning trees contains every edge of G with probability at least 1 -δ .

Definition A.14 (Anonymous identity and parent tags) . Let G = ( V, E, X ) be a graph and let S (1) be the first search sampled on G according to the RSNN search algorithm. Write S (1) = ( w 0 , w 1 , . . . , w T ) and let ( v (1) , v (2) , . . . , v ( n ) ) be the vertices of G ordered by their first-visit time along S (1) , i.e. v ( i ) is the i -th distinct vertex encountered in S (1) . Define the anonymous identity tag assignment τ : V → [ n ] by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We use the same tag assignment τ for all searches in the RSNN search set on G . Because S (1) is sampled in a manner equivariant to vertex relabellings, the induced random tag assignment τ is permutation-invariant in distribution . Now let S ( j ) be any search produced by the RSNN search algorithm, and write S ( j ) = ( u ( j ) 0 , u ( j ) 1 , . . . , u ( j ) T j ) . We define a parent-tag sequence as follows. For each index t , let q ( j ) t denote the vertex from which u ( j ) t is discovered by the RSNN search algorithm along a true edge of G ; when u ( j ) t is chosen as an initial vertex of a search, we set q ( j ) t := u ( j ) t . We then define

<!-- formula-not-decoded -->

In particular, the discovery vertex q ( j ) t need not equal the previous vertex u ( j ) t -1 in the recorded sequence: S ( j ) may contain discontinuities where consecutive vertices u ( j ) t -1 and u ( j ) t are not adjacent in G . The parent tag π ( j ) t always records, in anonymous coordinates given by τ , the vertex from which the search actually arrived at u ( j ) t along a true edge of G . Thus, the collection of identity and parent tags ( τ, ( π ( j ) ) j ) encodes all edges traversed by the RSNN searches, even across such discontinuities.

Theorem A.15 (Universal Approximation by RSNNs on Sparse Graphs with Bounded Degree) . Let ϵ &gt; 0 and let f : G → R d be any continuous graph-level function, where G is the space of sparse, unweighted graphs with | E | = O ( | V | ) , maximum degree at most d max , and size at most | V | ≤ n max . Assume f RSNN ( G ) uses (i) a universal set encoder f agg , (ii) a universal sequence encoder f seq , and (iii) anonymous identity and parent tags. Assume m satisfies Lemma A.13, so that full coverage is achieved with probability at least 1 -δ . Then, with probability at least 1 -δ there exists an RSNN configuration such that

<!-- formula-not-decoded -->

Proof. Let S FC ( G ) be the set of search sets of size m on G . Define a target on search sets by

<!-- formula-not-decoded -->

This ˜ f is well-defined (for any given input search set, there is a single unique output), and is permutation-invariant in the multiset argument. Because the input space (bounded-length sequences over a finite alphabet, aggregated into multisets of bounded size) is finite, the assumed universal sequence encoder and universal set aggregator can uniformly approximate ˜ f to error &lt; ε across ⋃ G ∈G ≤ n max S FC ( G ) . Therefore, with those parameters, for any G and any random S ( G ) , and hence unconditionally Pr( ∥ f RSNN ( S ) -F ( G ) ∥ &lt; ε ) ≥ 1 -δ .

<!-- formula-not-decoded -->

## A.3 Random Search Neural Network Invariance (Theorem 4.3, Corollary 4.4)

We next study invariance properties of RSNNs. Because RSNNs are randomized graph functions, we adopt a probabilistic notion of isomorphism invariance: if two graphs are isomorphic, the distributions of RSNN outputs coincide. As a consequence, the expected predictor Φ( G ) = E [ f RSNN ( G )] is an isomorphism-invariant graph function. Moreover, RSNNs learn this invariance via stochastic training: sampling a fresh search per step yields an unbiased gradient of the invariant risk, and under standard SGD conditions the parameters converge to a stationary point of the invariant objective. In practice, this justifies sampling with a small number of searches (e.g., m =1 ) in limited budget regimes.

Theorem A.16 (Isomorphism-Invariance of RSNN) . A randomized search procedure on a graph G produces a sequence S G = ( s G 0 , . . . , s G | V ( G ) | ) of visited vertices. We say the procedure is probabilistically invariant to graph isomorphisms if,

<!-- formula-not-decoded -->

The randomized DFS procedure used in RSNNs satisfies the above definition. Hence, RSNNs satisfy probabilistic invariance: for all G ∼ = H , f RSNN ( G ) d = f RSNN ( H ) , and the averaged predictor Φ( G ) := E [ f RSNN ( G ) ] is an invariant function on graphs: Φ( G ) = Φ( H ) for all G ∼ = H .

Proof. Write X DFS ( G ) = ( s 0 , . . . , s | V |-1 ) for the vertex sequence produced by the randomized DFS on G , and let H = π · G for an isomorphism π : V ( G ) → V ( H ) . The randomness comes from: (i) the root s 0 ∼ Unif( V ( G )) and (ii) an independent random order of neighbors at each vertex.

We prove by induction on t that the next state has the same pushforward conditional law under any isomorphism π :

π ( X DFS ( G )[ t ] ∣ ∣ x ) d = X DFS ( H )[ t ] ∣ ∣ π x , (10) for every valid DFS prefix x = ( s 0 , . . . , s t -1 ) on G (and its image π x on H ). Averaging over prefixes then yields πX DFS ( G )[ t ] d = X DFS ( H )[ t ] for each t , and thus πX DFS ( G ) d = X DFS ( H ) .

State, admissible set, and frontier. For a prefix x valid on G , let V vis ( G ; x ) = { s 0 , . . . , s t -1 } be the visited set and let top( G ; x ) be the current DFS stack top (the vertex whose adjacency list is being explored). Define the admissible neighbor set

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If A ( G ; x ) = ∅ , the rule 'pick the unvisited neighbor at random' makes the next vertex s t uniform on A ( G ; x ) . If A ( G ; x ) = ∅ , the next move is the (deterministic) backtrack to the parent of top( G ; x ) in the current DFS tree. Under an isomorphism π : G ∼ = H , relabeling preserves these invariants:

Base case ( t = 0 ). s 0 ∼ Unif( V ( G )) and πs 0 ∼ Unif( V ( H )) , so

<!-- formula-not-decoded -->

Induction step. Assume πX DFS ( G )[: t ] d = X DFS ( H )[: t ] . Fix any realization x of the prefix on G . There are two cases.

̸

(i) Expansion step: A ( G ; x ) = ∅ . Conditioned on x , X DFS ( G )[ t ] is uniform on A ( G ; x ) . Conditioned on π x , X DFS ( H )[ t ] is uniform on A ( H ; π x ) = πA ( G ; x ) . Pushing the uniform measure on A ( G ; x ) forward by π yields the uniform measure on πA ( G ; x ) , hence

<!-- formula-not-decoded -->

π ( X DFS ( G )[ t ] ∣ ∣ x ) = X DFS ( H )[ t ] ∣ ∣ π x , In both cases, the conditional laws match after applying π . Taking expectations over the distributions gives πX DFS ( G )[ t ] d = X DFS ( H )[ t ] for each t , which completes the induction and yields

(ii) Backtrack step: A ( G ; x ) = ∅ . The next state is the parent of top( G ; x ) in the DFS tree determined by x ; thus it is deterministic given x . Relabeling preserves parent/child relations in the explored DFS tree, so

<!-- formula-not-decoded -->

This proves probabilistic invariance of the randomized DFS. Since the RSNN output f RSNN is a deterministic function of the search sequence, it follows that f RSNN ( G ) d = f RSNN ( H ) , and the averaged predictor Φ( G ) = E [ f RSNN ( G )] is an invariant graph function.

Corollary A.17 (Stochastic training converges to the invariant objective) . Let ℓ ( · , y ) be a differentiable loss. Consider the expected risk

<!-- formula-not-decoded -->

At each SGD step t , sample ( G t , y t ) ∼ D and one search draw S t ∼ S DFS ( G t ) , and update

<!-- formula-not-decoded -->

Then E [ ∇ W ℓ ( f RSNN ( G t , S t ; W t ) , y t ) ] = ∇ W L ( W t ) , i.e., the single-sample gradient is an unbiased estimator of the invariant objective's gradient. Under standard SGD conditions, W t converges almost surely to the optimal W ⋆ of the invariant objective.

Proof sketch. This follows directly from the proof of Proposition A.1 in Murphy et al. [29], replacing permutations by RSNN searches: since the search randomness S ∼S DFS ( G ) is sampled independently of W and ℓ is differentiable with integrable gradient, we can exchange ∇ W and the expectations to get ∇ W L ( W ) = E ( G,y ) ∼D E S ∼S DFS ( G ) [ ∇ W ℓ ( f RSNN ( G,S ; W ) , y ) ] , so the single-sample stochastic gradient is unbiased; standard Robbins-Monro/Polyak supermartingale arguments then yield a.s. convergence of SGD to a stationary point (and to W ⋆ under convexity).

## B Additional Model Details: Positional Encodings and Sampling Algorithms

In this section, we provide additional details on the positional encoding scheme and sampling algorithms used in both RSNN and RWNN models. These components are essential not only for implementation but also for theoretical expressivity. We also present detailed descriptions of the sampling procedures for both random walks and random searches. For RWNNs, we outline the walk generation algorithm, including initialization, neighbor selection, and PE encoding. For RSNNs, we describe the random depth-first search strategy, including how spanning trees are constructed and how node visitation is handled.

## B.1 Positional Encodings

Identity and Adjacency Encodings. Tönshoff et al. [5] and Chen et al. [6] augment each walk with two binary feature matrices that inject explicit structural context. For a walk W = ( w 0 , . . . , w ℓ ) on graph G , the identity encoding id s W ∈ { 0 , 1 } ( ℓ +1) × s marks node repetitions within a sliding window of size s : for indices 0 ≤ i ≤ ℓ and 0 ≤ j ≤ s -1 we set

<!-- formula-not-decoded -->

and 0 otherwise. Thus column j signals whether the current node re-appeared exactly j steps earlier, explicitly encoding cycles of length j +1 . Second, the adjacency encoding adj s W ∈ { 0 , 1 } ( ℓ +1) × ( s -1) records edges among already-visited nodes that the walk does not traverse. We define

<!-- formula-not-decoded -->

and 0 otherwise. Here, E ( · ) denotes the edge set of the input. Consequently, for every pair of nodes that appears within the window, the encoding reveals whether they are adjacent in the underlying graph. The two blocks are concatenated to form a positional-encoding matrix h PE = [ id s W ∥ adj s W ] ∈ R ( ℓ +1) × d pe with d pe = 2 s -1 . Appending h PE to the raw node embeddings ensures that, once full node- and edge-coverage is achieved, the sequence model receives enough information to reconstruct the entire subgraph induced by the walk.

Anonymous Encodings. As an alternative to the identity-adjacency scheme, anonymous encodings have been proposed to capture graph structure by Wang and Cho [3] and Kim et al. [7]. For a walk W we create an integer vector ω anon ( W ) ∈ { 1 , . . . , ℓ +1 } ℓ +1 defined recursively:

<!-- formula-not-decoded -->

In words, the first time a node appears in the walk it is assigned the next unused label 1 , 2 , 3 , . . . ; every subsequent visit to that same node reuses the original label. Hence ω anon is invariant to the specific node IDs yet records the order in which unique vertices are discovered, providing topological context without relying on absolute labels.

Role in Expressivity. These positional encodings play a critical role in the expressive power of both RWNNs and RSNNs. They serve as the main mechanism by which the walk or search encodes structural information from the underlying graph. In particular, the identity and anonymous encodings, when combined with walks that achieve full edge coverage, allow for exact reconstruction of the input graph. Meanwhile, adjacency encodings with sufficient window size and identity/parent tags introduced in Definition A.14 enable full reconstruction even with only node coverage, as they record structural edges not explicitly traversed in the sequence. In our RSNN implementation, we rely on adjacency encodings. These are especially important for preserving expressivity in RSNNs: depth-first searches introduce disconnections in the sequence, where jumps between non-adjacent nodes may obscure structure. Consider for example nodes w i and w i +1 traversed adjacent to one another in a search sequence, but disconnected in the graph. With an appropriate window size, the adjacency encoding first signals the disconnection setting adj s W [ i + 1 , 1] = 0 , then identifies the connecting edge when it appeared in the sequence setting adj s W [ i +1 , j ] = 1 for ( w i , w i -j ) ∈ E ( G ) . This ensures that, once full edge coverage is achieved across searches, the sequence model receives all structural information necessary to reconstruct the graph. Thus, positional encodings are central to the theoretical guarantees of RSNN expressivity.

```
Input: Graph G = ( V, E ) , walk length l , window size s Output: Random walk W = ( w 0 , . . . , w l ) , encodings id s W , adj s W Sample initial node w 0 ∼ U ( V ) Initialize W ← [ w 0 ] for i ← 1 to l do Let N ( w i -1 ) be the neighbors of w i -1 Sample w i ∼ U ( N ( w i -1 )) Append w i to W for j ← 1 to s do if i -j ≥ 0 then id s W [ i, j ] ← 1 [ w i = w i -j ] // Identity encoding adj s W [ i, j ] ← 1 [( w i , w i -j ) ∈ E ] // Adjacency encoding return W , id s W , adj s W
```

Algorithm 1: Uniform Random Walk with Positional Encodings

```
Input: Graph G = ( V, E ) , window size s Output: Search sequence W = ( w 0 , . . . , w ℓ ) , adjacency encoding adj s W Sample initial node w 0 ∼ U ( V ) Initialize stack S ← [ w 0 ] , visited set V ← { w 0 } , walk W ← [ ] Initialize adj s W ← 0 | V |× ( s -1) while S is not empty do Pop u ←S Append u to W for j ← 1 to s -1 do if | W | > j then adj s W [ | W | -1 , j ] ← 1 [( u, W [ | W | -1 -j ]) ∈ E ] // Adjacency encoding Let N ( u ) be unvisited neighbors of u in random order foreach v ∈ N ( u ) do Push v onto S Add v to V return W , adj s W
```

Algorithm 2: Random Depth-First Search with Adjacency Encodings

## B.2 Sampling Algorithms

Random Walk Sampling. We adopt a standard uniform random walk procedure to extract sequences from a graph (Algorithm 1). The algorithm begins by uniformly sampling a starting node from the vertex set. At each step, it selects the next node uniformly at random from the current node's neighbors. As the walk progresses, we maintain a sliding window of fixed size s to compute identity and adjacency encodings for each step. The algorithm takes as input the graph G , walk length l , and window size s , and returns both the walk and the corresponding structural encodings.

Random Search Sampling. We implement random searches in RSNNs using a randomized depthfirst search (DFS) traversal (Algorithm 2). The algorithm begins by sampling a starting node uniformly at random from the vertex set. From there, we perform a standard DFS, visiting each neighbor in a random order to introduce stochasticity. As nodes are visited, they are recorded sequentially in the walk W , and only the adjacency-based positional encoding adj s W is computed using a sliding window of size s . Since DFS visits each node exactly once, identity encodings are unnecessary. The resulting walk and adjacency encoding together define the structural input for RSNNs.

## C Extended Results

We present two additional experiments to complement our main findings. First, we conduct an ablation study evaluating the impact of the sequence model architecture on performance by comparing CRAWL, the best performing RWNN, and RSNNs equipped with GRUs, LSTMs, and Transformers on molecular benchmarks. This experiment helps assess whether the RSNN framework is sensitive to the choice of sequence model. Second, we report runtime comparisons between RSNNs and RWNNs to evaluate computational efficiency. Specifically, we compare training times across varying sample sizes to understand how the two approaches scale under realistic computational budgets.

## C.1 Sequence Model Ablations

We evaluate the impact of sequence model architecture by comparing RSNNs and CRAWL equipped with GRUs, LSTMs, and Transformers (Table 1). Across all configurations, the trends from the main paper hold: RSNNs consistently outperform RWNNs at low sample sizes ( m = 1 ), regardless of sequence model. Notably, RSNNs with m = 1 often match or exceed the performance of RWNNs with m = 16 , reaffirming the sample efficiency advantages of random search. When m = 16 on the BACE dataset, CRAWL-LSTM and CRAWL-GRU slightly outperform their RSNN counterparts, however in the remaining comparisons RSNN always outperforms CRAWL across all m . Overall, GRUs and LSTMs perform comparably within both RSNN and RWNN variants, indicating that RSNN improvements are robust to the choice of sequence model, provided it has adequate recurrence-based inductive bias. In contrast, Transformers underperform relative to GRUs and LSTMs across most benchmarks and sample sizes. One possible explanation is that Transformers lack the hard-coded recurrence structure present in GRUs and LSTMs, relying instead on global attention mechanisms

Table 4: Median (min, max) of model AUC across test splits on molecular benchmarks. We report results for each model equipped with one of three sequence models: (1) GRU, (2) LSTM, or (3) Transformer (TRSF), as indicated by the suffix. The best model for each value of m is highlighted in blue . Trends from the main paper hold across architectures: RSNNs consistently outperform RWNNs at low sample sizes, with GRUs and LSTMs yielding similar performance, while Transformers underperform across most settings.

|        |                      | MoleculeNet Molecular Benchmarks (AUC ↑ )   | MoleculeNet Molecular Benchmarks (AUC ↑ )   | MoleculeNet Molecular Benchmarks (AUC ↑ )   | MoleculeNet Molecular Benchmarks (AUC ↑ )   | MoleculeNet Molecular Benchmarks (AUC ↑ )   | MoleculeNet Molecular Benchmarks (AUC ↑ )   |
|--------|----------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|
|        | # Graphs             | CLINTOX                                     | SIDER                                       | BACE                                        | BBBP                                        | TOX21 8K 18.6                               | TOXCAST                                     |
|        | # Graphs             | 1.5K                                        | 1.5K                                        | 1.5K                                        | 2K                                          |                                             | 9K                                          |
|        | Avg. &#124; V &#124; | 26.1                                        | 33.6                                        | 34.1                                        | 23.9                                        |                                             | 18.8                                        |
|        | Avg. &#124; E &#124; | 55.5                                        | 70.7                                        | 73.7                                        | 51.6                                        | 38.6                                        | 38.5                                        |
|        | # Classes            | 2                                           | 2                                           | 2                                           | 2                                           | 2                                           | 2                                           |
|        | CRAWL-TRSF           | 59.8 (48.1, 71.8)                           | 60.3 (57.2, 68.4)                           | 67.6 (65.2, 73.3)                           | 74.6 (66.1, 79.4)                           | 70.4 (65.3, 74.5)                           | 70.8 (65.4, 75.3)                           |
|        | CRAWL-LSTM           | 66.7 (40.4, 80.2)                           | 61.4 (57.4, 63.8)                           | 66.2 (60.7, 71.4)                           | 74.4 (68.4, 80.4)                           | 72.2 (67.6, 76.0)                           | 71.5 (67.7, 75.4)                           |
|        | CRAWL-GRU            | 70.0 (64.6, 73.6)                           | 64.2 (56.1, 67.2)                           | 62.5 (59.2, 70.8)                           | 77.6 (68.8, 81.5)                           | 71.7 (66.4, 75.3)                           | 72.8 (67.7, 76.7)                           |
| m = 1  | RSNN-TRSF            | 82.9 (59.8, 87.9)                           | 65.6 (63.1, 71.9)                           | 78.0 (71.3, 81.5)                           | 85.6 (77.6, 89.8)                           | 77.7 (73.8, 78.9)                           | 74.2 (70.8, 78.8)                           |
|        | RSNN-LSTM            | 87.2 (82.6, 89.4)                           | 66.8 (61.7, 72.2)                           | 78.2 (74.3, 84.3)                           | 87.1 (83.9, 89.5)                           | 79.5 (77.2, 83.7)                           | 75.6 (72.9, 80.6)                           |
|        | RSNN-GRU             | 88.1 (84.9, 91.5)                           | 66.2 (63.0, 72.4)                           | 79.7 (75.9, 83.6)                           | 87.5 (80.3, 89.9)                           | 79.8 (77.2, 83.4)                           | 74.6 (72.3, 79.7)                           |
|        | CRAWL-TRSF           | 69.4 (49.0, 79.0)                           | 64.7 (61.1, 69.5)                           | 73.7 (68.4, 75.4)                           | 82.6 (77.5, 87.7)                           | 74.5 (71.6, 78.6)                           | 71.3 (69.1, 80.0)                           |
|        | CRAWL-LSTM           | 80.4 (72.3, 83.8)                           | 66.3 (63.2, 68.8)                           | 72.7 (67.5, 78.5)                           | 84.0 (78.5, 88.6)                           | 77.5 (75.3, 79.9)                           | 74.6 (71.1, 79.9)                           |
|        | CRAWL-GRU            | 83.0 (76.6, 91.5)                           | 65.2 (59.5, 71.3)                           | 75.7 (71.0, 79.0)                           | 84.5 (80.7, 87.0)                           | 77.6 (75.6, 81.2)                           | 74.4 (69.2, 77.9)                           |
| m = 4  | RSNN-TRSF            | 84.2 (63.4, 87.0)                           | 67.1 (64.6, 70.8)                           | 79.8 (69.4, 82.5)                           | 85.6 (79.9, 90.7)                           | 78.0 (74.2, 83.0)                           | 76.6 (71.5, 81.2)                           |
|        | RSNN-LSTM            | 88.7 (81.2, 90.8)                           | 67.5 (64.1, 70.1)                           | 80.9 (75.3, 84.4)                           | 88.9 (82.7, 91.6)                           | 81.4 (76.3, 83.3)                           | 76.6 (73.8, 81.3)                           |
|        | RSNN-GRU             | 89.1 (80.9, 91.7)                           | 67.0 (61.3, 71.1)                           | 80.4 (76.5, 84.0)                           | 88.0 (80.3, 90.5)                           | 80.3 (77.3, 84.2)                           | 76.1 (72.2, 79.0)                           |
|        | CRAWL-TRSF           | 68.3 (53.1, 88.1)                           | 65.9 (62.6, 71.4)                           | 75.4 (66.6, 80.7)                           | 85.4 (79.2, 89.6)                           | 76.4 (71.8, 78.2)                           | 75.2 (72.0, 78.7)                           |
|        | CRAWL-LSTM           | 87.2 (78.3, 89.4)                           | 67.1 (63.6, 70.7)                           | 79.2 (76.8, 83.2)                           | 86.8 (79.5, 91.6)                           | 78.9 (76.0, 81.7)                           | 73.5 (68.9, 77.3)                           |
|        | CRAWL-GRU            | 86.5 (83.6, 91.4)                           | 66.1 (62.1, 69.9)                           | 80.3 (71.0, 82.5)                           | 86.0 (82.8, 89.6)                           | 79.1 (76.7, 82.1)                           | 75.5 (72.0, 78.6)                           |
| m = 8  | RSNN-TRSF            | 82.7 (51.8, 89.9)                           | 66.8 (62.5, 72.0)                           | 80.2 (73.3, 82.4)                           | 86.4 (79.8, 90.7)                           | 76.8 (75.4, 81.5)                           | 75.2 (71.5, 81.4)                           |
|        | RSNN-LSTM            | 88.4 (82.2, 90.6)                           | 67.2 (64.3, 74.6)                           | 80.7 (74.8, 87.1)                           | 88.1 (82.6, 91.4)                           | 81.1 (77.7, 85.2)                           | 75.9 (72.3, 82.2)                           |
|        | RSNN-GRU             | 88.3 (80.1, 91.3)                           | 67.6 (63.3, 69.2)                           | 80.0 (76.1, 85.1)                           | 88.6 (83.6, 90.3)                           | 82.2 (77.3, 85.3)                           | 75.7 (73.0, 78.9)                           |
|        | CRAWL-TRSF           | 69.6 (47.6, 86.9)                           | 65.1 (63.1, 70.1)                           | 78.8 (73.5, 79.7)                           | 85.2 (79.5, 89.3)                           | 77.7 (75.8, 81.9)                           | 74.8 (72.1, 80.0)                           |
|        | CRAWL-LSTM           | 87.8 (80.1, 89.5)                           | 65.7 (63.4, 69.0)                           | 79.5 (74.4, 86.0)                           | 87.1 (79.7, 93.9)                           | 79.2 (77.9, 82.3)                           | 76.2 (70.4, 79.0)                           |
|        | CRAWL-GRU            | 89.1 (80.5, 91.1)                           | 65.3 (61.4, 70.8)                           | 80.7 (76.1, 84.5)                           | 87.0 (81.7, 90.3)                           | 80.9 (77.4, 82.6)                           | 76.2 (72.7, 77.9)                           |
| m = 16 | RSNN-TRSF            | 84.4 (78.5, 91.7)                           | 66.6 (63.6, 73.6)                           | 81.0 (73.1, 82.8)                           | 86.0 (78.7, 90.7)                           | 77.6 (74.5, 82.1)                           | 76.4 (72.3, 79.2)                           |
|        | RSNN-LSTM            | 88.3 (81.9, 92.2)                           | 67.3 (64.8, 71.9)                           | 80.5 (79.0, 84.3)                           | 88.5 (83.8, 91.2)                           | 82.0 (78.8, 83.5)                           | 75.5 (72.9, 80.0)                           |
|        | RSNN-GRU             | 88.5 (82.0, 93.7)                           | 67.1 (65.0, 74.0)                           | 79.8 (76.8, 84.9)                           | 89.4 (83.0, 91.7)                           | 82.2 (78.0, 84.1)                           | 76.5 (73.4, 79.3)                           |

that may require more data to model sequential dependencies effectively, especially in low-sample regimes. These results suggest that recurrent sequence models are better suited for graph-based walk or search processing under constrained sampling budgets.

Figure 4: Training runtime (in seconds) of RSNN and CRAWL over 25 epochs on SIDER , BBBP , TOXCAST as a function of the number of samples m . Error bars represent standard deviation across 5 runs. At low sample counts, RSNNs exhibit comparable runtime to RWNNs; as m increases, RSNNs become faster despite longer sequence lengths. We hypothesize this is due to random walks repeatedly visiting high-degree nodes, incurring more computation per step, whereas DFS-based searches visit each node exactly once.

<!-- image -->

## C.2 Runtime Comparisons

Experimental Setup. To ensure a fair comparison between RSNNs and CRAWL, we fix all model components except for the sampling strategy. Both models use a GRU sequence model with hidden dimension 64 and are composed of 2 layers. We set the positional encoding window size to 8 and batch size to 64. For each graph, the random walk length is set to l = | V | , equal to the number of nodes, so that the number of sequence steps is identical between random walks and searches. As a result, RSNNs and CRAWL have equivalent asymptotic runtimes per sample. We measure training runtime over 25 epochs on three molecular benchmarks, SIDER , BBBP , and TOXCAST , across varying sample sizes m ∈ { 1 , 4 , 8 , 16 } . For each forward pass, a fresh set of m walks or searches is sampled per graph. All experiments are run on a single NVIDIA GeForce GTX 1080 Ti GPU, with sampling parallelized across 4 CPU workers to reflect practical deployment conditions.

Results. Empirically we observe that RSNN searches are never more expensive than CRAWL walks for any tested number of walks m , and that for larger m the RSNN implementation can even become faster (Figure 4). Although, each routine shares the same asymptotic cost, O ( | V | ) on our sparse graphs, they differ by constant factors that affect runtime comparisons in practice:

- Random Walks Revisit Nodes with Larger Degrees. A DFS visits each vertex exactly once, while a random walk visits nodes randomly, potentially revisiting many vertices with higher degrees. Consequently, searches and random walks visit different sets of nodes. This affects runtime since operations per node depend on their degrees (e.g., shuffling neighbors, random choices on neighbors, identity/adjacency checks), incurring more computation per-step and increasing runtimes for random walks.
- Per-step work. The DFS runs one for s loop that updates a single adjacency-encoding tensor. The RWNN walk performs an identical for s loop, but each iteration evaluates two conditions (identity &amp; adjacency encoding) and writes to two tensors, effectively doubling the cost of that inner loop per step.
- Neighbor handling. DFS shuffles the neighbor list once per new vertex, whereas random walks rebuilds a Python list and calls random.choice at every step, and if non-backtracking is enabled, creates an additional filtered list. These repeated list allocations and Python-level random picks inflate wall time.

Together, these constant-factor differences explain why the asymptotically identical O ( | V | ) algorithms show distinct wall-time profiles: RSNN remains competitive for all m , while CRAWL exhibit longer runtimes at larger m .

## D Experimental Details and Code

Training and Hyperparameter Selection. All models are trained by minimizing the binary crossentropy loss on molecular benchmarks and the negative log-likelihood loss on protein benchmarks. Training is performed for a maximum of 200 epochs with early stopping patience set to 25 epochs based on validation performance. The best-performing model on the validation set is selected for evaluation on the test set. We perform a grid search over the following hyperparameters for all RWNN and RSNN models:

- Number of layers: {1, 2, 3}
- Learning rate: {0.05, 0.01, 0.005, 0.001}
- Batch size: {32, 64, 128}
- Hidden dimension: {32, 64, 128}
- Global pooling: { mean , sum , max }
- Sequence model: { GRU , LSTM , Transformer }
- Number of samples m : {1, 4, 8, 16}

We fix the window size s = 8 for both CRAWL and RSNN models. All models are optimized using the Adam optimizer [42].

## E Extended Discussion

Background on WL and its Variants. The Weisfeiler-Lehman (WL) hierarchy has become a standard lens for characterizing graph model expressivity. Xu et al. [15] first established the equivalence between 1-dimensional WL and MPNNs, while Morris et al. [39], Azizian and Lelarge [21] generalized this perspective to higher-order GNNs via higher-order WL variants. Beyond MPNNs, recent work has aligned graph transformers with WL, clarifying their expressivity within the same hierarchy [43, 44]. In parallel, random walk kernels and path GNNs have been connected to WLas sequence-based representations [41, 45].

Our Walk Weisfeiler-Lehman (WWL) refinement builds directly on this line: we introduce a walkbased color refinement and show that, under full coverage, its distinguishing power coincides with 1-WL. In doing so, we place RWNNs firmly within the WL-centered expressivity landscape alongside MPNNs, graph transformers, and path-based GNNs, advancing a unified view of diverse graph learning architectures through the WL hierarchy.