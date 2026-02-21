## Differentially Private Gomory-Hu Trees

## Anders Aamand

BARC, University of Copenhagen aa@di.ku.dk

## Mina Dalirrooyfard

Morgan Stanley minad@mit.edu

## Yuriy Nevmyvaka

Morgan Stanley

## Justin Y. Chen

Massachusetts Institute of Technology justc@mit.edu

## Slobodan Mitrovi´ c

UC Davis smitrovic@ucdavis.edu

## Sandeep Silwal

UW-Madison yuriy.nevmyvaka@morganstanley.com

silwal@cs.wisc.edu

## Abstract

Given an undirected, weighted n -vertex graph G = ( V, E, w ) , a Gomory-Hu tree T is a weighted tree on V that preserves the Mins -t -Cut between any pair of vertices s, t ∈ V . Finding cuts in graphs is a key primitive in problems such as bipartite matching, spectral and correlation clustering, and community detection. We design a differentially private (DP) algorithm that computes an approximate Gomory-Hu tree. Our algorithm is ε -DP, runs in polynomial time, and can be used to compute s -t cuts that are ˜ O ( n/ε ) -additive approximations of the Mins -t -Cuts in G for all distinct s, t ∈ V with high probability. Our error bound is essentially optimal, since [29] showed that privately outputting a single Mins -t -Cut requires Ω( n ) additive error even with ( ε, δ ) -DP and allowing for multiplicative error. Prior to our work, the best additive error bounds for approximate all-pairs Mins -t -Cuts were O ( n 3 / 2 /ε ) for ε -DP [47] and ˜ O ( √ mn/ε ) for ( ε, δ ) -DP [66], both achieved by DP algorithms that preserve all cuts in the graph. To achieve our result, we develop an ε -DP algorithm for the Minimum Isolating Cuts problem with near-linear error, and introduce a novel privacy composition technique combining elements of both parallel and basic composition to handle 'bounded overlap' computational branches in recursive algorithms, which maybe of independent interest.

## 1 Introduction

Over the last two decades, there has been a significant attention to privatizing graph algorithms (see [70, 51, 57, 47, 10, 58, 37, 17, 78, 31, 39, 22, 32, 29, 66, 53] and references within). Graph algorithms are often applied to large data sets, such as social networks, containing sensitive information. It is well understood by now that even minor negligence in handling user data can severely impact privacy; see [8, 74, 59, 82, 26] for a few examples. Differential privacy (DP) , introduced by Dwork, McSherry, Nissim, and Smith in their seminal work [36], is a widely adopted standard for formalizing the privacy guarantees of algorithms. Informally, an algorithm is differentially private if the outputs for two given neighboring inputs are statistically indistinguishable.

A major part of the literature on private graph algorithms has been approximating cuts which is the setting of our paper. Given an undirected, weighted graph G = ( V, E, w ) with positive edge weights, a cut is a bipartition of vertices ( U, V \ U ) , and the value of the cut is the sum of the weights of edges crossing the bipartition. Given a pair of distinct vertices s, t ∈ V , the Mins -t -Cut is

Yinzhan Xu UC San Diego xyzhan@ucsd.edu

a minimum-valued cut ( U, V \ U ) where s ∈ U and t ∈ V \ U . Mins -t -Cut is dual to the Maxs -t -Flow problem, and the celebrated max-flow min-cut theorem states that the value of the Mins -t -Cut equals the value of the Maxs -t -Flow [41, 38]. Finding a Mins -t -Cut (or equivalently Maxs -t -Flow) is a fundamental problem in algorithmic graph theory, which has been studied for over seven decades (earliest references include [30, 50, 41, 38]), and has inspired ample algorithmic research and applications, including edge connectivity [71], bipartite matching (see, e.g., [25]), minimum Steiner cut [62], vertex-connectivity oracles [77], among others (see the survey [23]).

Over the recent years, nearly tight algorithms have been developed on outputting private Mins -t -Cut [29], private global Min Cut [46], and private All Cuts problems [47, 9, 37, 66, 67]. 1 . However, no work has been done on private All-Pairs Min Cut (APMC) which is our focus. Given an input graph, the goal of APMC is to output a Mins -t -Cut for all the pairs of vertices s and t in V . We fill this gap, and obtain a private algorithm on APMC with the same error guarantee as private Mins -t -Cut.

The definition of neighboring inputs depends on the specific application and yields semantically different privacy guarantees. We now specify the standard privacy model for graph cut problems which is used in ours and the aforementioned works. The graph's vertex set is publicly available, and two neighboring graphs differ in only one edge. If the graphs are weighted, two neighboring graphs are those whose total weights differ by at most one and in a single edge. Semantically, this protects privacy if the inclusion of any individual affects the weight of an edge by a bounded quantity. 2 Given neighboring inputs G and G ′ , and a subset of outputs O , an ( ε, δ ) -DP algorithm A satisfies P ( A ( G ) ∈ O ) ≤ e ε P ( A ( G ′ ) ∈ O ) + δ . When δ = 0 , the algorithm satisfies pure DP, and otherwise approximate DP. 3 There is always a trade-off between privacy and accuracy: algorithms for nontrivial problems satisfying DP must have errors.

Since the above privacy model is standard for analyzing cut problems under DP, it is natural to also adopt it for the APMC problem. To further motivate the study of APMC under this particular privacy model from a more practical perspective, note that the graph could correspond to a road network and the weight of an edge could correspond to the number of people who travel along the road corresponding to that edge. The above model then ensures that whether any particular individual uses any particular road cannot be deduced based on the output of a DP-algorithm run on the graph. Moreover, minimum cuts are a natural measure of bottlenecks in transportation networks, and it may therefore be of interest to compute them on such traffic graphs in a differentially private fashion.

We now discuss the relevant private algorithms for cut problems. Dalirrooyfard, Mitrovi´ c, and Nevmyvaka [29] recently gave the optimal ε -DP algorithm for the Mins -t -Cut problem with additive error O ( n/ε ) . They show an essentially matching Ω( n ) lower bound, even for algorithms which satisfy only approximate DP and allow both multiplicative and additive error.

For the problem of global Min Cut , where one seeks the cut minimizing Mins -t -Cut over all pairs of node s, t , Gupta et al. [46] gave an ε -DP algorithm with additive error O ( log n ε ) . Their algorithm runs in exponential time, but they also give a version running in polynomial time but only satisfying approximate DP. 4 The authors also show that there does not exist an ε -DP algorithm for global Min Cut incurring less than Ω(log n ) additive error.

The private All Cuts problem, where one seeks to output a synthetic, private graph which preserves the value of all the cuts, has been extensively studied. Given a graph G , the goal is to output a synthetic graph H on the same vertices such that each cut-value in H is the same as the corresponding cut-value in G , up to some additive error. Gupta, Roth, and Ullman [47] and, independently, Blocki, Blum, Datta, and Sheffet [9] gave algorithms for this problem with additive error of O ( n 1 . 5 /ε ) while satisfying pure and approximate DP, respectively. Eliáš, Kapralov, Kulkarni, and Lee [37] improved

1 Note that the focus of all of these works and our paper is on outputting the actual cut structure not the value of the cut.

2 We note that there are several other notions of neighboring datasets for other graph problems implying different privacy semantics (including allowing a vertex and all of its edges to change or fixing the unweighted topology of the graph and only allowing edge weights to change). The notion which we use on is the standard for cut problems (used in [46, 9, 37, 84, 29, 66]) as it is the strongest form of privacy that allows for any approximation of cuts or cut values. Moreover, this notion carries over to weight-DP, where in the two neighboring graphs all edges can change in total ℓ 1 distance 1. We include a detailed discussion of various forms of differential privacy on graphs in Appendix A.5.

3 The parameter δ corresponds to the small probability that an individual's data is leaked.

4 In this case the additive error has a dependency on δ , we do not mention this dependency for simplicity.

Table 1: State-of-the-art bounds for private cut problems. Dependencies on the approx. DP parameter δ are hidden. The APMC result with approx. DP follows from advanced composition by adding Lap ( O ( n/ε )) random noise to all ( n 2 ) true values. For APMC, the lower bound of Ω( n ) error [29] also applies as APMC generalizes Mins -t -Cut.

| Problem               | Additive Error                 | DP     | Output          | Runtime     |
|-----------------------|--------------------------------|--------|-----------------|-------------|
| Global Min Cut [46]   | Θ(log( n ) /ε )                | Pure   | Cut             | Exponential |
| Global Min Cut [46]   | Θ(log( n ) /ε )                | Approx | Cut             | Polynomial  |
| Min- s - t -Cut [29]  | O ( n/ε ) and Ω( n )           | Pure   | Cut             | Polynomial  |
| All Cuts [47]         | O ( n 3 / 2 /ε )               | Pure   | Synthetic Graph | Polynomial  |
| All Cuts [37, 66]     | ˜ O ( √ mn/ε ) and Ω( √ mn/ε ) | Approx | Synthetic Graph | Polynomial  |
| APMC Values (Trivial) | O ( n/ε )                      | Approx | Values Only     | Polynomial  |
| APMC (Our Work)       | ˜ O ( n/ε )                    | Pure   | GH-Tree         | Polynomial  |

on these results for sparse, unweighted (or small weight) graphs, achieving error ˜ O (√ mn ε ) 5 with approximate DP. The authors show that this error is essentially tight for algorithms with purely additive error (and no multiplicative approximation). In a follow-up work, Liu, Upadhyay, and Zou [66] extended these results to weighted graphs and gave an algorithm for releasing a synthetic graph for the All Cuts problem with ˜ O ( √ mn ε ) error. Recently, Liu et al. [67] gave an algorithm for the same problem with worse error ˜ O ( m ε ) but which runs in near-linear time.

One important cut problem absent in the study of private graph algorithms is All-Pairs Min Cut (APMC), which has been extensively studied in the graph algorithms community for over six decades. In a seminal paper, Gomory and Hu [44] showed that there is a tree representation for this problem, called a GH-tree or cut tree, which takes only n -1 Mins -t -Cut (Max Flow) calls to compute. Consequently, there are only n -1 different minimum cut values in an arbitrary graph with positive edge weights. There has been a long line of research in designing faster GH-tree algorithms (e.g. [49, 15, 2, 88, 63, 3, 5, 6], also see the survey [75]), culminating in an almost linear time algorithm for computing the GH-tree [6].

Beyond its importance in graph algorithms, the all-pairs aspect of APMC is especially important in the context of DP. Answering multiple queries degrades privacy, so a key feature of differential privacy is the ability to control this degradation through composition theorems (see, for instance, [34, 83]). Applying advanced composition of private mechanisms to the O ( n/ε ) error result of [29] for a single Mins -t -Cut implies that APMC can be solved with O ( n 2 /ε ) additive error while satisfying approximate DP. Given the structure of the APMC problem as characterized by the existence of GH-trees, it is natural to ask if one can improve upon black-box composition results. 6

Existing works provide a preliminary answer. Since the algorithm by [66] approximately preserves all cuts, it can also be used to solve APMC with approximate DP and additive error of ˜ O ( √ mn/ε ) . Additionally, the Ω( n ) lower bound for Mins -t -Cut of [29] also applies to APMC, as it is a harder problem. So, in contrast to computing global Min Cut [46], Mins -t -Cut [29], and All Cuts [37, 66], where the privacy/error tradeoff is tightly characterized up to poly (log n, 1 /ε ) factors, there remains a gap of ≈ √ m/n between the best known lower and upper bound for DP APMC, which can be as large as Ω( √ n ) in dense graphs. This motivates the following question, which is our focus:

## 1.1 Our Results

Our main contribution is an ε -DP algorithm for APMC with ˜ O ( n/ε ) additive error. Our algorithm privately outputs all the Mins -t -Cuts while incurring the same error, up to polylog( n ) factors, required to output a Mins -t -Cut for a single pair of vertices s and t . To achieve this result, we introduce an algorithm that solves the more general problem of privately generating an approximate Gomory-Hu tree (GH-tree). Gomory and Hu [44] showed that for any undirected graph G , there exists a tree T defined on the vertices of graph G such that for all pairs of vertices s, t , the Mins -t -Cut in T is also a Mins -t -Cut in G . We develop a private algorithm for constructing such a tree.

̸

Theorem 1.1. Given a weighted graph G = ( V, E, w ) with positive edge weights and a privacy parameter ε &gt; 0 , there exists an ε -DP algorithm that outputs an approximate GH-tree T with additive error ˜ O ( n/ε ) : for any s = t ∈ V , the Mins -t -Cut on T and the Mins -t -Cut on G differ in ˜ O ( n/ε ) in cut value with respect to edge weights in G . The algorithm runs in time ˜ O ( n 2 ) , and the additive error guarantee holds with high probability.

Theorem 1.1 essentially outputs a synthetic graph in a DP manner that approximates each Mins -t -Cut with an additive error of ˜ O ( n/ε ) . Since the GH-tree output by Theorem 1.1 is private, any postprocessing on this tree is also private. This yields the following corollary.

Corollary 1.1. Given a weighted graph G with positive edge weights and a privacy parameter ε &gt; 0 , there exists an ε -DP algorithm that outputs, for all the pairs of vertices s and t , a cut whose value is within ˜ O ( n/ε ) from the value of the Mins -t -Cut with high probability.

Another corollary of Theorem 1.1 is a polynomial-time, pure DP algorithm for global Min-Cut.

Corollary 1.2. Given a weighted graph G with positive edge weights and a privacy parameter ε &gt; 0 , there exists an ε -DP algorithm that outputs an approximate global Min-Cut of G in ˜ O ( n 2 ) time and has additive error ˜ O ( n/ε ) with high probability.

Prior work obtained an exponential-time pure DP algorithm and a polynomial-time approximate DP for global Min-Cut with error O (log n/ε ) [46]. It is an open question whether there exists an efficient algorithm which satisfies pure DP and outputs an approximage global Min-Cut with polylog( n ) /ε additive error.

Lastly, we note another application of Theorem 1.1 is a pure -DP algorithm for min k -cut 7 problem with multiplicative error 2 , additive error ˜ O ( nk/ε ) . No prior poly time pure DP algorithm can compute min k -cut on weighted graphs with near-linear in n error. See Corollary F.1) for details.

Tightness of Our Main Result Corollary 1.1 is tight up to polylog( n ) and 1 ε factors since any DP algorithm outputting the Mins -t -Cut for a single fixed pair of vertices s and t requires Ω( n ) additive error [29]. Thus, our result shows that we can compute all min-cuts privately with the same error required for a single cut up to log factors. We also note that the Ω( n ) lower bound for a single cut of [29] is on sparse graphs and applies to both pure and approx DP. Hence, there can be no polynomial improvement on our result even if the input is sparse or if we relax to approx DP.

Paper Organization In the rest of the main body, we give a detailed overview of the challenges in privately creating a GH-tree as well as a technical overview of our approach. Due to space constraints, all pseudocode and proofs are given in the appendix. Two technical ingredients which may be generally applicable are (1) an ε -DP, ˜ O ( n/ε ) additive error algorithm for Minimum Isolating Cuts [62, 3], a recently introduced problem which has found success as a subroutine in fast algorithms for cut problems, and (2) a general theorem for privacy composition on recursive algorithms where sensitive data is not partitioned into disjoint sets on each recursive call (in this setting, composition is straightforward), but rather the recursive subsets have 'bounded-overlap.'

Open Problems. We highlight three interesting open problems related to our work in Appendix G.

5 In this work, the notation ˜ O ( x ) stands for O ( x · polylog x ) .

6 A recent line of work on approximating all-pairs shortest path distances with differential privacy has the same goal of using graph structure to limit the error necessary to answer many queries [81, 39, 22, 13].

7 The goal is to partition the vertex set into k pieces and the cost of a partitioning is the total weight of all edges between different pieces in the partition. We wish to find the smallest cost solution.

## 2 Technical Overview

A greatly simplified view of a typical approach to designing a DP algorithm begins with a non-DP algorithm, which is modified to ensure privacy. The primary challenge lies in finding an appropriate method to privatizing the algorithm, if such a method even exists, and rigorously proving that satisfies DP. For instance, Gupta et al. [46] employ Karger's algorithm [56] to produce a set of cuts and then use the Exponential Mechanism [70] to choose one of those cut. This simple but clever approach results in an ( ε, δ ) -DP algorithm for global Min Cut, for δ = 1 poly( n ) , with O ( log n ε ) additive error. Dalirrooyfard et al. [29] show that the following simple algorithm yields ε -DP Mins -t -Cut with O ( n/ε ) additive error: for each vertex v , add an edge from v to s and from v to t with their weights chosen from the exponential distribution with parameter 1 /ε ; return the Mins -t -Cut on the modified graph. In the remainder of this section, we first explain why directly privatizing certain existing non-private algorithms fails to achieve the desired additive error. After, we describe our approach.

## 2.1 Obstacles in Privatizing the Algorithm of Gomory and Hu

The All-Pairs Min-Cut (APMC) problem produces cuts for ( n 2 ) pairs of vertices; it is known that these cuts have only O ( n ) distinct values. This property was leveraged in the pioneering work by Gomory and Hu [44], who introduced the Gomory-Hu tree (GH-tree), a structure that succinctly represents all the Mins -t -Cuts in a graph.

The original GH-tree algorithm uses a recursive construction, solving the Mins -t -Cut problem at each recursion step: In each step of the recursive call, the input will be a graph H and a special set of terminal vertices R ⊆ V . (1) The root of the recursion starts with H = G and R = V . (2) At each recursive step, select two arbitrary vertices s and t within R (if | R | = 1 , then the problem becomes trivial and the recursion stops). (3) Compute the Mins -t -Cut in the graph, and say the Mins -t -Cut is ( U, V \ U ) where s ∈ U, t ̸∈ U . (4) Create two graphs, H s and H t , where H s is the graph with V \ U contracted, and H t is the graph with U contracted. Then recursively solve the problem on H s with terminal set R ∩ U , and on H t with terminal set R ∩ ( V \ U ) . (5) Finally, we combine the two trees created by the two recursive calls, argue the correctness utilizing the submodularity of cuts. The GH-tree efficiently represents all pairwise Mins -t -Cuts in the graph by iterating through these steps until there is no supernode of size larger than 1 .

To privatize this algorithm, the Mins -t -Cut procedures can be replaced with the private Mins -t -Cut algorithm introduced in [29]. However, several challenges arise in ensuring low error with this approach. First, the algorithm of [29] modifies the graph. It is unclear whether these modifications should persist in each Mins -t -Cut call or if the graph should revert to its original form. Second, the recursion depth may reach O ( n ) . Even ignoring the propagation of error across recursive calls, dependent invocations of the DP Mins -t -Cut must use very small values of ε due to privacy composition. Under basic composition [35], each call to the algorithm of [29] must use ε ′ = O ( ε/n ) to guarantee ε -DP for the final algorithm. Even using advanced composition [34] would require ε ′ = O ( ε/ √ n log(1 /δ )) to achieve ( ε, δ ) -DP. The resulting error would still be higher than that achieved in the prior work preserving all cuts. Finally, we may hope that a single edge impacts only a small number of min-cuts, a property that could be leveraged in constructing a private GH-tree. However, as we depict in Figure 3, the ℓ 1 -sensitivity of changing a single edge is Ω( n ) , even when outputting only the values of the Mins -t -Cuts.

## 2.2 Towards Privatizing a Low-Depth Algorithm

The preceding discussion indicates that low recursive depth is a crucial property of a private algorithm for producing a GH-tree. While the canonical algorithm of [44] has a linear recursive depth, recent breakthroughs in fast GH-tree algorithms offer the additional advantage of polylogarithmic depth, e.g., [3, 61, 4, 64, 6]. At a high-level, our result derives from privatizing the algorithm described in [61, Section 4.5] (this same algorithm also appears in [4]). To replace components of this algorithm with differentially private counterparts necessitates adding noise, which introduces additive errors throughout the recursive algorithm. Tracking the propagation of this error throughout the algorithm requires careful accounting. Moreover, the specific steps of the non-private algorithm presents several fundamental challenges in creating a private version. The remainder of this subsection briefly summarizes the structure of this algorithm.

Figure 1: Recursive structure of our low-depth GH-tree. (a) The decomposition into ( S v ) v ∈ R and S large = V \ ⋃ v ∈ R S v defines the recursive subinstances. For each v ∈ R , S v constitutes a Mins -v -Cut. None of the S v or S large are too large, which leads to a polylogarithmic upper-bound on the recursion depth of the algorithm. (b) A subinstance obtained by contracting V \ S v 1 to a single vertex x v 1 . (c) A subinstance obtained by contracting S v to a single vertex y v for each v ∈ B .

<!-- image -->

Minimum Isolating Cuts. For a set U of vertices in G , let w ( U ) be the sum of the weights of the edges in G with exactly one endpoint in U . The first building block is a subroutine called Minimum Isolating Cuts , introduced in [62, 3].

Definition 2.1 (Min Isolating Cuts [62, 3]) . Given a set of terminals R ⊆ V , the Min Isolating Cuts problem asks to output a collection of sets { S v ⊆ V : v ∈ R } such that for each vertex v ∈ R , the set S v satisfies S v ∩ R = { v } , and it has the minimum value of w ( S ′ v ) over all sets S ′ v that S ′ v ∩ R = { v } . In other words, each S v is the minimum cut separating v from R \ { v } .

The Isolating Cuts Lemma [62, 3] shows how to solve Min Isolating Cuts in O (log n ) calls to Mins -t -Cut. At a high level, the Isolating Cuts Lemma is used to simultaneously find disjoint Mins -t -Cuts for several s, t pairs. More precisely, a single recursive step of the GH-tree algorithm of [61] uses Min Isolating Cuts to find a collection of disjoint subsets of vertices { S v ⊂ V } v ∈ R for some set of terminals R ⊂ V and a source vertex s / ∈ R which satisfy three properties:

- (a) The subsets correspond to Mins -v -Cuts: each subset S v is the v -side of a Mins -v -Cut.
- (b) The subsets are not too large: | S v | ≤ (1 -Ω(1)) n .
- (c) The union of the subsets is not too small: ∑ | S v | ≥ n/ polylog( n ) .

The algorithm is recursively applied to each of the subsets S v (with vertices outside of S v contracted) as well as the remainder of the vertices S large = V \ ⋃ S v (with each S v contracted). In concert, conditions (b) and (c) guarantee that the recursive depth will be polylogarithmic. Condition (a) allows us to stitch the outcome of these recursive calls together into a GH-tree that preserves Mins -t -Cuts. See Figure 1 for a visualization of this process. It remains to describe how to generate the source s , terminals R , and subsets { S v } v ∈ R to complete the high level description of the algorithm of [61]. We call a subset S v having properties (a) and (b) a good subset.

Choosing s . The vertex s is chosen uniformly due to the following: Let D ∗ be the set of all vertices v ∈ V \ { s } for which there exists a Mins -v -Cut with the v -side having cardinality at most n/ 2 , i.e., the v side of the cut satisfies condition (b). A lemma from [2] shows that, as long as s is chosen uniformly at random from V , we have E [ | D ∗ | ] = Ω( n ) : a constant fraction of vertices have small cardinality Mins -v -Cuts. This is important since the algorithm chooses R from D ∗ , i.e., all the subsets S v chosen by the algorithm have cardinality at most n/ 2 and hence they satisfy property (b).

Finding good subsets { S v } which cover D ∗ . After choosing s , the algorithm selects geometrically decreasing sets of potential terminals R 0 , . . . , R ⌊ lg n ⌋ where each vertex other than s is selected into R i uniformly at random with probability 2 -i . For each set of terminals R i , the algorithm runs Min Isolating Cuts on R i ∪{ s } and keeps the 'good' subsets { S v } v ∈ R i output by Min Isolating Cuts. Recall that a good subset S v has properties (a) and (b). To test property (a), we verify which subsets

S v correspond to Mins -v -Cuts by computing the single source Mins -v -Cut values from s to all v ∈ V \ { s } and comparing that to the value of the cuts S v . To satisfy property (b), we only keep S v if | S v | ≤ n/ 2 .

The first observation is that for any particular Mins -v -Cut with v -side S ∗ v , there is some sampling level i ∗ where there is a reasonable probability that the only terminal sampled in S ∗ v is v , i.e. R i ∗ ∩ S ∗ v = { v } . If this event occurs, then Min Isolating Cuts with terminals R i ∗ ∪ { s } will return S ∗ v as one of its outputs-the minimum cut separating v from the other terminals is the Mins -v -Cut. The second observation is that there is some sampling level i ≤ ⌊ lg n ⌋ where, when run on the corresponding terminals R i ∪ { s } , the expected size of the union of good subsets is Ω ( | D ∗ | log 2 n ) = Ω ( n log 2 n ) , i.e., satisfying (c). Thus, selecting the sampling level whose good subsets have maximum total cardinality produces a collection { S v } satisfying conditions (a), (b), (c). 8

## 2.3 Three Core Challenges

As the Min Isolating Cuts are a major building block of the algorithm of [61], our first challenge is finding a privatized version of it.

Challenge 1. Construct a differentially private, approximate Min Isolating Cuts algorithm.

Next, in one recursive step, the non-private algorithm compares the Min Isolating Cuts to single source Mins -v -Cut values. Making these procedures private will introduce additive errors, and this makes it difficult to satisfy conditions (b) and (c). Recall that in [61], at some sampling level Min Isolating Cuts will correspond to Mins -v -Cuts which are neither too large for condition (b) nor too small for condition (c). This convenient property is deduced from the cardinality of the true Mins -v -Cut and the fact that calls to Min Isolating Cuts will find exactly those Mins -v -Cuts. Without adjustment, plugging in additive approximate subroutines will fail to produce recursive sub-instances of a reasonable size, undermining the goal of low recursive depth.

Challenge 2. Design a recursive step that employs additive approximations for Mins -v -Cut values and Min Isolating Cuts while ensuring that (1) no sub-instance is too large and (2) the union of sub-instances is sufficiently large.

Even assuming that the above two challenges are resolved, it remains unclear how to account for the privacy loss of the final low-depth recursive algorithm. The difficulty lies in the fact that, despite the polylogarithmic recursion depth, a single edge of the original graph may appear in multiple sub-instances across a single recursive level. A given instance with terminal set R will have | R | +1 recursive sub-instances: | R | of them are obtained by contracting each of the vertex sets ( V \ S v ) v ∈ R into a single vertex x v , and one is obtained by contracting each S v , v ∈ R into a single vertex y v (see Figure 1 (b)). If an edge has both of its incident vertices lying in a single S v or in S large := V \ ⋃ v ∈ R S v , then that edge will only appear in the corresponding sub-instance. However, an edge with, say, one endpoint in S u and the other in S v for different u and v will appear in three recursive sub-instances: those obtained by contracting V \ S u , V \ S v , and V \ S large . Moreover, this edge could appear in up to O ( n ) sub-instances. So, if we naïvely apply basic composition, we need the computation on each sub-instance to be ( ε/n ) -DP. This would lead to a final ˜ O ( n 2 /ε ) additive error guarantee, which is again worse than previous work on preserving all cuts.

Challenge 3. Modify the algorithm so that the privacy loss of a given edge can be accounted for by the recursion depth.

In the following sections, we give a technical overview on how we overcome these challenges.

## 2.4 Addressing Challenge 1: Privatizing Min Isolating Cuts (Appendix B)

[62] and [3] independently introduce the Isolating Cuts Lemma, showing how to solve the Min Isolating Cuts problem using O (log | R | ) many Mins -t -Cut calls. [62] use it to find the global Min Cut in polylogarithmically many invocations of Max Flow and [3] use it to compute GH-tree in

8 Note that the final set of terminals R is inferred from the selection of { S v } , corresponding to the good subsets at a certain sampling level.

unweighted graphs. Subsequent algorithms for GH-tree also use the Isolating Cuts Lemma, including the almost linear time algorithm for weighted graphs [6]. The Isolating Cuts Lemma has been extended to obtain new algorithms for finding the non-trivial minimizer of a symmetric submodular function and solving the hypergraph minimum cut problem [72, 21]. We develop the first differentially private algorithm for finding Min Isolating Cuts.

Theorem 2.1. There is an ε -DP algorithm that given a graph G and a set of terminals R , outputs sets { S v : v ∈ R } , such that for each vertex v ∈ R , the set S v satisfies S v ∩ R = { v } , and w ( S v ) ≤ w ( S ∗ v ) + ˜ O ( n ε ) with high prob., where { S ∗ v : v ∈ R } are the Min Isolating Cuts for R .

Tackling this first challenge-obtaining a DP algorithm for constructing Min Isolating Cuts-turns out to be not too difficult. We repeatedly invoke the private Mins -t -Cut algorithm of [29] O (log n ) times. To establish the error bound, we carefully apply the cut submodularity property (Lemma A.1) to account for the approximate errors. Since we are separating a potentially large set of terminals rather than a single pair, we must ensure the total error, summed across all terminals, remains bounded by ˜ O ( n ε ) (see Lemma B.1 for this stronger result).

## 2.5 Addressing Challenge 2: Privatizing the Recursive Step (Appendix B and Appendix C)

The second challenge involves controlling the size of the sets produced by our DP Min Isolating Cuts algorithm. To address this, we use the following idea to bias the algorithm toward selecting smaller cardinality sets to satisfy that property (b). Consider a graph G with terminals s and t . We aim to find an approximate Mins -t -Cut where the s -side of the cut contains a small number of vertices, without significantly sacrificing accuracy. To achieve this, we can add edges from t to every vertex with a certain weight, penalizing the placement of vertices on the s -side of the cut. Applying this idea to our DP Min Isolating Cuts algorithm, we enforce that if a true minimum isolating cut of size at most n/ 2 exists, we will output an isolating cut of size at most 0 . 9 n while increasing the additive error of the algorithm by only a constant factor.

The above only takes care of property (b). A subtlety in the argument of [61] in making sure property (c) is satisfied over all good subsets is the following: Randomly sampling terminals R i will, with reasonable probability, mean that for some vertices v , its Min Isolating Cut S v will be the same as its Mins -v -Cut S ∗ v . In particular, this will be true if v ∈ R i is the only vertex sampled on its side of the Mins -v -Cut. In this case, S v is a good subset and is kept by the algorithm. Later, using that S v = S ∗ v , they argue that the union of good subsets is not too small.

Unfortunately, even though the Min Isolating Cuts output by our DP algorithm have small additive error, they can have significantly fewer nodes than the optimal cut S ∗ v . Essentially, although the true Mins -v -Cut may be large, there is no lower bound on the node size of approximate Mins -v -Cuts. This poses a challenge in showing that the union of the good sets S v retained by the algorithm is not too small (property (c)). To address this, we compare S v not to S ∗ v , but instead to the smallest cardinality 'approximate' Mins -v -Cut ˜ S v . The argument that, for some sampling level, there will be a reasonable probability that Min Isolating Cuts will return an S v which is an approximate Mins -v -Cut and has cardinality lower bounded by | ˜ S v | then goes through. For the complete argument, we adjust our notion of 'approximation" based on the size of ˜ S v , allowing for weaker approximations for smaller cardinality sets. This only degrades our approximation with respect to property (a) by log factors.

## 2.6 Addressing Challenge 3: Bounding Privacy along the Recursion Tree (Appendix D)

The third challenge involves controlling the privacy budget. We need to ensure that, for any two neighboring graphs, the output distributions across polylog( n ) recursive layers differ by, at most a e ε factor. To achieve this guarantee, we allocate the privacy budget across polylog( n ) recursive sub-instances. Recall that a recursion depth of polylog( n ) does not imply that the privacy budget can be evenly allocated across polylog( n ) instances. To recall the 3rd challenge, consider two neighboring instances, G and G ′ , that differ on edge xy . Suppose that in the first step of the algorithm, the good sets { S v } and S large are the same in both graphs G and G ′ . Suppose that x ∈ S v and y ∈ S u . In this case, the edge xy affects multiple recursive instances: specifically, in G v , G u , and G large (similarly G ′ v , G ′ u , and G ′ large ). Thus computations across multiple branches of the recursion depend on the same edge, meaning the privacy guarantee depends on more than just the recursion depth.

Figure 2: Bounded-overlap branching composition. Node u receives as input a set of sensitive data X u and a set of sanitized data Y u . Via a DP mechanism M recurse, it then computes the number of children of u , and for each child v , a set of sensitive indices I v and a set of sanitized indices J v . Importantly, the sets I v are disjoint, and an index j can appear in at most ℓ different sets J v (in the figure ℓ = 2 ). Node u may further return some other DP output a u . Each child v now receives as input the data ( X i ) i ∈ I v and the concatenation ( f ( Y ) , M sanitize ( X j ) j ∈ J v ) where M sanitize is a DP mechanism and f is some arbitrary function. Intuitively, if an element X i of the dataset is sanitized in a child v , then post-processing ensures that the entire computation in the subtree rooted at v is DP with respect to user i . Since user i 's data X i can only appear unsanitized in a single child, and is sanitized in at most ℓ children, we can apply basic composition over h computations of M recurse and ℓ · h computations of M sanitize, where h is the depth of the tree.

<!-- image -->

To address this issue, we introduce an additional step before recursing on certain new instances. For the instances where all vertices in V \ S v are contracted to a single vertex, we first add an edge from the contracted vertex to every vertex in S v with weights drawn from Lap (polylog( n ) /ε ) . Then we recurse on this altered graph. The intuition is that in neighboring instances, if x ∈ S v and y / ∈ S v , these noisy edges cancel the effect of xy , preventing its influence from propagating further along this branch of the recursion. As a result, xy impacts only G large. Fortunately, the added noise only contributes ˜ O ( n/ε ) to the final additive error, as noise is added to only O ( n ) edges.

Bounded-Overlap Branching Composition. The approach outlined above is an example of a more general privacy composition technique, which we develop and formalize (Appendix D.2). Here, we provide a high-level overview of the technique and refer to Figure 2 for an illustration. Consider an algorithm M branch that takes a sensitive dataset X as input, performs DP computation on X , and recurses on some subsets of X . Now, consider the computation tree corresponding to the recursive branching of M branch. If the subsets assigned to a node's children are chosen privately and the subsets are disjoint , privacy composition is straightforward, as outlined below.

Let h upper-bound the depth of the tree. If, at a single node, the release of the computation on X as well as the indices to the subsets of its children is ( ε, δ ) -DP, then the entire mechanism M branch ( X ) is ( hε, δh ) -DP. To see this, consider any pair of neighboring datasets X,X ′ which differ only on the coordinate i ∗ . As the subsets of indices assigned by each node to its children are disjoint, there is a unique path from the root of the tree along which nodes do computation on X i ∗ . All other nodes in the tree do not depend on the value of X i ∗ after conditioning on the release of the index sets of all children of nodes along the path. Composition along this path gives the result. This can be thought of as a novel combination of parallel 9 and basic composition, and post-processing.

The challenge in our setting is that each node in the recursion tree does not partition its data among its children (the same edge weight can appear in multiple subinstances). A single data point may be sent to multiple children, so the number of nodes in the tree whose input includes X i ∗ may be exponential in the depth h . We define a restricted class of recursive mechanisms, which take in both a sensitive dataset X and a sanitized dataset Y . The overall recursive mechanism M branch is formed using two private subroutines M recurse and M sanitize. At each recursive step, M recurse ( X,Y ) is used to generate some partial output, the number of recursive children, and a set of sensitive and sanitized index sets I = ( i 1 , . . . , i n ) and J = ( j 1 , . . . , j m ) for each recursive child. A child receives as input (1) a sensitive dataset X i 1 , . . . , X i n which is a subset of its parent's sensitive dataset X , and (2) a sanitized input, which is the concatenation of any function of its parent's sanitized dataset Y and the output of the private computation M sanitize ( X j 1 , . . . , X j m ) .

9 We use parallel composition to refer to the fact that the union of outputs of an ( ε, δ ) -DP mechanism applied separately to disjoint subsets of a sensitive dataset is itself ( ε, δ ) -DP.

The key property of the mechanisms we define is bounded-overlap , parameterized by some constant ℓ . For any index i into the sensitive dataset X of the parent: (1): There is at most one child whose sensitive index set contains i . (2): There are at most ℓ children whose sanitized index sets contain i .

Theorem 2.2 (Informal Version of Theorem D.2) . Let M branch be a recursive mechanism as described above with bounded-overlap and maximum depth h with subroutines M recurse and M sanitize which are ( ε 1 , δ 1 ) -DP and ( ε 2 , δ 2 ) -DP, respectively. Then, releasing the union of outputs of M recurse and M sanitize over the entire recursion tree is ( hε 1 +( h -1) ℓε 2 , hδ 1 +( h -1) ℓδ 2 ) -DP.

Note that this generalizes the above example, where all children get disjoint subsets of the parent's data by setting ℓ = 0 . Indeed, the high-level idea of the proof of this theorem follows the same argument, with more care taken to argue that, by sanitizing the overlapping data sent to a node's children, we can bound privacy along a single path from the root of the tree for any particular index i ∗ . We ultimately prove the privacy of our GH-tree algorithm by the application of this general theorem. As described earlier in this subsection, the sanitization procedure is to add Laplace noise from the contracted node to all other nodes in S v subinstances. Then, each edge is only part of the private input to a single recursive child. A key observation is that the recursive GH-tree algorithm has bounded overlap with ℓ = 2 as any given edge can only belong to two such S v instances.

## Acknowledgements

A. Aamand was supported by the VILLUM Foundation grant 54451. J. Chen was supported by an NSF Graduate Research Fellowship under Grant No. 17453. S. Mitrovi´ c was supported by the Google Research Scholar and NSF Faculty Early Career Development Program No. 2340048. Y. Xu was supported by NSF Grant CCF-2330048, HDR TRIPODS Phase II grant 2217058, and a Simons Investigator Award. Part of this work was conducted while J. Chen and S. Mitrovi´ c were visiting the Simons Institute for the Theory of Computing.

## References

- [1] Anders Aamand, Justin Y. Chen, Mina Dalirrooyfard, Slobodan Mitrovic, Yuriy Nevmyvaka, Sandeep Silwal, and Yinzhan Xu. Breaking the n 1.5 additive error barrier for private and efficient graph sparsification via private expander decomposition. In Proceedings of the 42nd International Conference on Machine Learning (ICML) , 2025. ↑ 38
- [2] Amir Abboud, Robert Krauthgamer, and Ohad Trabelsi. Cut-equivalent trees are optimal for min-cut queries. In Proceedings of the 2020 IEEE 61st Annual Symposium on Foundations of Computer Science (FOCS) , pages 105-118. IEEE, 2020. ↑ 3, ↑ 6, ↑ 17, ↑ 31
- [3] Amir Abboud, Robert Krauthgamer, and Ohad Trabelsi. Subcubic algorithms for GomoryHu tree in unweighted graphs. In Samir Khuller and Virginia Vassilevska Williams, editors, Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing (STOC) , pages 1725-1737. ACM, 2021. doi: 10.1145/3406325.3451073. URL https://doi.org/10. 1145/3406325.3451073 . ↑ 3, ↑ 4, ↑ 5, ↑ 6, ↑ 7, ↑ 21
- [4] Amir Abboud, Robert Krauthgamer, Jason Li, Debmalya Panigrahi, Thatchaphol Saranurak, and Ohad Trabelsi. Breaking the cubic barrier for all-pairs max-flow: Gomory-Hu tree in nearly quadratic time. In Proceedings of the 2022 IEEE 63rd Annual Symposium on Foundations of Computer Science (FOCS) , pages 884-895, 10 2022. ↑ 5, ↑ 24, ↑ 36
- [5] Amir Abboud, Robert Krauthgamer, and Ohad Trabelsi. APMF &lt; APSP? Gomory-Hu tree for unweighted graphs in almost-quadratic time. In Proceedings of the 2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS) , pages 1135-1146. IEEE, 2022. ↑ 3
- [6] Amir Abboud, Jason Li, Debmalya Panigrahi, and Thatchaphol Saranurak. All-pairs max-flow is no harder than single-pair max-flow: Gomory-Hu trees in almost-linear time. In Proceedings of the 2023 IEEE 64th Annual Symposium on Foundations of Computer Science (FOCS) , pages 2204-2212. IEEE, 2023. ↑ 3, ↑ 5, ↑ 8
- [7] Arik Azran. The rendezvous algorithm: Multiclass semi-supervised learning with markov random walks. In Proceedings of the 24th international conference on Machine learning , pages 49-56, 2007. ↑ 36

- [8] Lars Backstrom, Cynthia Dwork, and Jon Kleinberg. Wherefore art thou R3579X? anonymized social networks, hidden patterns, and structural steganography. In Proceedings of the 16th international conference on World Wide Web (WWW) , pages 181-190, 2007. ↑ 1
- [9] Jeremiah Blocki, Avrim Blum, Anupam Datta, and Or Sheffet. The Johnson-Lindenstrauss transform itself preserves differential privacy. In Proceedings of the 53rd Annual IEEE Symposium on Foundations of Computer Science (FOCS) , pages 410-419, 2012. doi: 10.1109/FOCS.2012.67. URL https://doi.org/10.1109/FOCS.2012.67 . ↑ 2
- [10] Jeremiah Blocki, Avrim Blum, Anupam Datta, and Or Sheffet. Differentially private data analysis of social networks via restricted sensitivity. In Robert D. Kleinberg, editor, Proceedings of the 13th Innovations in Theoretical Computer Science (ITCS) , pages 87-96. ACM, 2013. ↑ 1
- [11] Avrim Blum and Shuchi Chawla. Learning from labeled and unlabeled data using graph mincuts. In Proceedings of the Eighteenth International Conference on Machine Learning , pages 19-26, 2001. ↑ 36
- [12] Avrim Blum, John Lafferty, Mugizi Robert Rwebangira, and Rajashekar Reddy. Semisupervised learning using randomized mincuts. In Proceedings of the twenty-first international conference on Machine learning , page 13, 2004. ↑ 36
- [13] Greg Bodwin, Chengyuan Deng, Jie Gao, Gary Hoppenworth, Jalaj Upadhyay, and Chen Wang. The discrepancy of shortest paths. In Karl Bringmann, Martin Grohe, Gabriele Puppis, and Ola Svensson, editors, Proceedings of the 51st International Colloquium on Automata, Languages, and Programming (ICALP) , volume 297 of LIPIcs , pages 27:1-27:20. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2024. doi: 10.4230/LIPICS.ICALP.2024.27. URL https://doi.org/10.4230/LIPIcs.ICALP.2024.27 . ↑ 4, ↑ 37
- [14] Christian Borgs, Jennifer T. Chayes, Adam D. Smith, and Ilias Zadik. Revealing network structure, confidentially: Improved rates for node-private graphon estimation. In Mikkel Thorup, editor, Proceedings of the 59th IEEE Annual Symposium on Foundations of Computer Science (FOCS) , pages 533-543. IEEE Computer Society, 2018. ↑ 19
- [15] Glencora Borradaile, Piotr Sankowski, and Christian Wulff-Nilsen. Min st-cut oracle for planar graphs with near-linear preprocessing time. ACM Trans. Algorithms , 11(3):1-29, 2015. ↑ 3
- [16] Glencora Borradaile, David Eppstein, Amir Nayyeri, and Christian Wulff-Nilsen. All-pairs minimum cuts in near-linear time for surface-embedded graphs. In Sándor P. Fekete and Anna Lubiw, editors, Proceedings of the 32nd International Symposium on Computational Geometry (SoCG) , volume 51 of LIPIcs , pages 22:1-22:16. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2016. doi: 10.4230/LIPICS.SOCG.2016.22. URL https://doi.org/10.4230/ LIPIcs.SoCG.2016.22 . ↑ 17
- [17] Mark Bun, Marek Elias, and Janardhan Kulkarni. Differentially private correlation clustering. In Proceedings of the 38th International Conference on Machine Learning (ICML) , pages 1136-1146. PMLR, 2021. ↑ 1
- [18] Rishi Chandra, Michael Dinitz, Chenglin Fan, and Zongrui Zou. Differentially private multiway and k -cut. CoRR , abs/2407.06911, 2024. doi: 10.48550/ARXIV.2407.06911. URL https: //doi.org/10.48550/arXiv.2407.06911 . ↑ 36
- [19] Chandra Chekuri. Approximation Algorithms , Lecture 7, 2009. URL https://courses. grainger.illinois.edu/cs598csc/sp2009/lectures/lecture\_7.pdf . Url: https: //courses.grainger.illinois.edu/cs598csc/sp2009/lectures/lecture\_7.pdf , Accessed: August 1, 2024. ↑ 37
- [20] Kamalika Chaudhuri, Fan Chung, and Alexander Tsiatas. Spectral clustering of graphs with general degrees in the extended planted partition model. In Conference on Learning Theory , pages 35-1. JMLR Workshop and Conference Proceedings, 2012. ↑ 36
- [21] Chandra Chekuri and Kent Quanrud. Isolating cuts, (bi-)submodularity, and faster algorithms for connectivity. In Nikhil Bansal, Emanuela Merelli, and James Worrell, editors, Proceedings of the 48th International Colloquium on Automata, Languages, and Programming (ICALP) ,

volume 198 of LIPIcs , pages 50:1-50:20. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2021. doi: 10.4230/LIPICS.ICALP.2021.50. URL https://doi.org/10.4230/LIPIcs. ICALP.2021.50 . ↑ 8

- [22] Justin Y. Chen, Badih Ghazi, Ravi Kumar, Pasin Manurangsi, Shyam Narayanan, Jelani Nelson, and Yinzhan Xu. Differentially private all-pairs shortest path distances: Improved algorithms and lower bounds. In Nikhil Bansal and Viswanath Nagarajan, editors, Proceedings of the 2023 ACM-SIAM Symposium on Discrete Algorithms (SODA) , pages 5040-5067. SIAM, 2023. doi: 10.1137/1.9781611977554.CH184. URL https://doi.org/10.1137/1.9781611977554. ch184 . ↑ 1, ↑ 4, ↑ 20, ↑ 37
- [23] Li Chen, Rasmus Kyng, Yang P. Liu, Richard Peng, Maximilian Probst Gutenberg, and Sushant Sachdeva. Maximum flow and minimum-cost flow in almost-linear time. In Proceedings of the 63rd IEEE Annual Symposium on Foundations of Computer Science (FOCS) , pages 612-623. IEEE, 2022. doi: 10.1109/FOCS54457.2022.00064. URL https://doi.org/10. 1109/FOCS54457.2022.00064 . ↑ 2, ↑ 36
- [24] Vincent Cohen-Addad, Frederik Mallmann-Trenn, and David Saulpic. Community recovery in the degree-heterogeneous stochastic block model. In Conference on Learning Theory , pages 1662-1692. PMLR, 2022. ↑ 36
- [25] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. Introduction to algorithms . MIT press, 2022. ↑ 2
- [26] Chris Culnane, Benjamin I. P. Rubinstein, and Vanessa Teague. Stop the open data bus, we want to get off. CoRR , abs/1908.05004, 2019. URL http://arxiv.org/abs/1908.05004 . ↑ 1
- [27] William H. Cunningham. Minimum cuts, modular functions, and matroid polyhedra. Networks , 15(2):205-215, 1985. ↑ 17
- [28] Elias Dahlhaus, David S. Johnson, Christos H. Papadimitriou, Paul D. Seymour, and Mihalis Yannakakis. The complexity of multiterminal cuts. SIAM Journal on Computing , 23(4):864-894, 1994. ↑ 36
- [29] Mina Dalirrooyfard, Slobodan Mitrovi´ c, and Yuriy Nevmyvaka. Nearly tight bounds for differentially private multiway cut. In Proceedings of the Advances in Neural Information Processing Systems 36 (NeurIPS) , 2023. ↑ 1, ↑ 2, ↑ 3, ↑ 4, ↑ 5, ↑ 8, ↑ 19, ↑ 36, ↑ 37
- [30] George B. Dantzig. Application of the simplex method to a transportation problem. Activity analysis and production and allocation , 1951. ↑ 2
- [31] Laxman Dhulipala, Quanquan C. Liu, Sofya Raskhodnikova, Jessica Shi, Julian Shun, and Shangdi Yu. Differential privacy from locally adjustable graph algorithms: k-core decomposition, low out-degree ordering, and densest subgraphs. In Proceedings of the 2022 IEEE 63rd Annual Symposium on Foundations of Computer Science (FOCS) , pages 754-765. IEEE, 2022. ↑ 1
- [32] Michael Dinitz, Satyen Kale, Silvio Lattanzi, and Sergei Vassilvitskii. Improved differentially private densest subgraph: Local and purely additive. arXiv preprint arXiv:2308.10316 , 2023. ↑ 1
- [33] Cynthia Dwork and Jing Lei. Differential privacy and robust statistics. In Proceedings of the 41st Annual ACM Symposium on Theory of Computing (STOC) , pages 371-380, 2009. ↑ 18
- [34] Cynthia Dwork and Aaron Roth. The algorithmic foundations of differential privacy. Found. Trends Theor. Comput. Sci. , 9(3-4):211-407, 2014. ↑ 3, ↑ 5, ↑ 18, ↑ 37
- [35] Cynthia Dwork, Krishnaram Kenthapadi, Frank McSherry, Ilya Mironov, and Moni Naor. Our data, ourselves: Privacy via distributed noise generation. In Proceedings of the 24th Annual International Conference on the Theory and Applications of Cryptographic Techniques (EUROCRYPT) , pages 486-503. Springer, 2006. ↑ 5

- [36] Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam D. Smith. Calibrating noise to sensitivity in private data analysis. In Shai Halevi and Tal Rabin, editors, Proceedings of the 3rd Theory of Cryptography Conference (TCC) , volume 3876 of Lecture Notes in Computer Science , pages 265-284. Springer, 2006. doi: 10.1007/11681878\_14. URL https: //doi.org/10.1007/11681878\_14 . ↑ 1, ↑ 18
- [37] Marek Eliáš, Michael Kapralov, Janardhan Kulkarni, and Yin Tat Lee. Differentially private release of synthetic graphs. In Proceedings of the 14th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA) , pages 560-578, 2020. doi: 10.1137/1.9781611975994.34. URL https://doi.org/10.1137/1.9781611975994.34 . ↑ 1, ↑ 2, ↑ 3, ↑ 19, ↑ 38
- [38] Peter Elias, Amiel Feinstein, and Claude Shannon. A note on the maximum flow through a network. IRE Trans. Inf. Theory , 2(4):117-119, 1956. ↑ 2
- [39] Chenglin Fan, Ping Li, and Xiaoyun Li. Private graph all-pairwise-shortest-path distance release with improved error rate. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Proceedings of the Advances in Neural Information Processing Systems 35 (NeurIPS) , 2022. URL http://papers.nips.cc/paper\_files/paper/2022/ hash/71b17f00017da0d73823ccf7fbce2d4f-Abstract-Conference.html . ↑ 1, ↑ 4, ↑ 37
- [40] Gary William Flake, Steve Lawrence, and C Lee Giles. Efficient identification of web communities. In Proceedings of the sixth ACM SIGKDD international conference on Knowledge discovery and data mining , pages 150-160, 2000. ↑ 36
- [41] Lester Randolph Ford and Delbert R. Fulkerson. Maximal flow through a network. Can. J. Math. , 8:399-404, 1956. ↑ 2
- [42] Julia Gaudio and Nirmit Joshi. Community detection in the hypergraph sbm: Exact recovery given the similarity matrix. In The Thirty Sixth Annual Conference on Learning Theory , pages 469-510. PMLR, 2023. ↑ 36
- [43] Julia Gaudio, Miklos Z Racz, and Anirudh Sridhar. Exact community recovery in correlated stochastic block models. In Conference on Learning Theory , pages 2183-2241. PMLR, 2022. ↑ 36
- [44] Ralph E. Gomory and Tien Chung Hu. Multi-terminal network flows. J. Soc. Indust. Appl. Math. , 9(4):551-570, 1961. ↑ 3, ↑ 4, ↑ 5
- [45] Andrew Guillory and Jeff A Bilmes. Label selection on graphs. Advances in Neural Information Processing Systems , 22, 2009. ↑ 36
- [46] Anupam Gupta, Katrina Ligett, Frank McSherry, Aaron Roth, and Kunal Talwar. Differentially private combinatorial optimization. In Proceedings of the 21st Annual ACM-SIAM Symposium on Discrete Algorithms (SODA) , 2010. doi: 10.1137/1.9781611973075.90. URL https: //doi.org/10.1137/1.9781611973075.90 . ↑ 2, ↑ 3, ↑ 4, ↑ 5, ↑ 19, ↑ 37
- [47] Anupam Gupta, Aaron Roth, and Jonathan R. Ullman. Iterative constructions and private data release. In Ronald Cramer, editor, Proceedings of the 9th Theory of Cryptography Conference (TCC) , volume 7194 of Lecture Notes in Computer Science , pages 339-356. Springer, 2012. doi: 10.1007/978-3-642-28914-9\_19. URL https://doi.org/10.1007/ 978-3-642-28914-9\_19 . ↑ 1, ↑ 2, ↑ 3, ↑ 19, ↑ 37
- [48] Steve Hanneke. An analysis of graph cut size for transductive learning. In Proceedings of the 23rd international conference on Machine learning , pages 393-399, 2006. ↑ 36
- [49] Ramesh Hariharan, Telikepalli Kavitha, Debmalya Panigrahi, and Anand Bhalgat. An ˜ O ( mn ) Gomory-Hu tree construction algorithm for unweighted graphs. In Proceedings of the 39th Annual ACM Symposium on Theory of Computing (STOC) , pages 605-614, 2007. ↑ 3
- [50] T. E. Harris and F. S. Ross. Fundamentals of a method for evaluating rail net capacities. Technical report, RAND Corporation, Santa Monica, CA, 1955. ↑ 2

- [51] Michael Hay, Chao Li, Gerome Miklau, and David D. Jensen. Accurate estimation of the degree distribution of private networks. In Wei Wang, Hillol Kargupta, Sanjay Ranka, Philip S. Yu, and Xindong Wu, editors, Proceedings of the 9th IEEE International Conference on Data Mining (ICDM) , pages 169-178. IEEE Computer Society, 2009. ↑ 1
- [52] Chester Holtz, Pengwen Chen, Zhengchao Wan, Chung-Kuan Cheng, and Gal Mishne. Continuous partitioning for graph-based semi-supervised learning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024. URL https://openreview. net/forum?id=hCOuip5Ona . ↑ 36
- [53] Palak Jain, Iden Kalemaj, Sofya Raskhodnikova, Satchit Sivakumar, and Adam Smith. Counting distinct elements in the turnstile model with differential privacy under continual observation. Proceedings of the Advances in Neural Information Processing Systems 36 (NeurIPS) , 36, 2024. ↑ 1
- [54] Svante Janson. Tail bounds for sums of geometric and exponential variables. Stat. Probab. Lett. , 135, 09 2017. doi: 10.1016/j.spl.2017.11.017. ↑ 18
- [55] Thorsten Joachims. Transductive learning via spectral graph partitioning. In Proceedings of the 20th international conference on machine learning (ICML-03) , pages 290-297, 2003. ↑ 36
- [56] David R. Karger. Global min-cuts in RNC, and other ramifications of a simple min-cut algorithm. In Vijaya Ramachandran, editor, Proceedings of the 4th Annual ACM/SIGACTSIAM Symposium on Discrete Algorithms (SODA) , pages 21-30. ACM/SIAM, 1993. URL http://dl.acm.org/citation.cfm?id=313559.313605 . ↑ 5
- [57] Vishesh Karwa, Sofya Raskhodnikova, Adam D. Smith, and Grigory Yaroslavtsev. Private analysis of graph structure. ACM Trans. Database Syst. , 39(3):22:1-22:33, 2014. doi: 10.1145/ 2611523. URL https://doi.org/10.1145/2611523 . ↑ 1
- [58] Shiva Prasad Kasiviswanathan, Kobbi Nissim, Sofya Raskhodnikova, and Adam D. Smith. Analyzing graphs with node differential privacy. In Amit Sahai, editor, Proceedings of the 10th Theory of Cryptography Conference (TCC) , volume 7785 of Lecture Notes in Computer Science , pages 457-476. Springer, 2013. ↑ 1
- [59] Aleksandra Korolova. Privacy violations using microtargeted ads: A case study. In Proceedings of the 2010 IEEE International Conference on Data Mining Workshops (ICDMW) , pages 474-482. IEEE, 2010. ↑ 1
- [60] Samuel Kotz, Tomasz J. Kozubowski, and Krzysztof Podgórski. The Laplace Distribution and Generalizations: A Revisit with Applications to Communications, Economics, Engineering, and Finance . Birkhäuser, 2001. ISBN 9780817641665. Proposition 2.2.2, Equation 2.2.8. ↑ 26
- [61] Jason Li. Preconditioning and Locality in Algorithm Design. PhD thesis, Carnegie Mellon University, USA, 2021. ↑ 5, ↑ 6, ↑ 7, ↑ 8, ↑ 17, ↑ 23, ↑ 24, ↑ 29, ↑ 30, ↑ 31, ↑ 36
- [62] Jason Li and Debmalya Panigrahi. Deterministic min-cut in poly-logarithmic max-flows. In Sandy Irani, editor, Proceedings of the 61st IEEE Annual Symposium on Foundations of Computer Science (FOCS) , pages 85-92. IEEE, 2020. doi: 10.1109/FOCS46700.2020.00017. URL https://doi.org/10.1109/FOCS46700.2020.00017 . ↑ 2, ↑ 4, ↑ 6, ↑ 7, ↑ 17, ↑ 21, ↑ 25
- [63] Jason Li and Debmalya Panigrahi. Approximate Gomory-Hu tree is faster than n -1 max-flows. In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing (STOC) , pages 1738-1748, 2021. ↑ 3
- [64] Jason Li, Debmalya Panigrahi, and Thatchaphol Saranurak. A nearly optimal all-pairs min-cuts algorithm in simple graphs. In Proceedings of the 2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS) , pages 1124-1134. IEEE, 2022. ↑ 5
- [65] Daogao Liu. Better private algorithms for correlation clustering. In Po-Ling Loh and Maxim Raginsky, editors, Proceedings of the 35th Annual Conference on Learning Theory (COLT) , volume 178 of Proceedings of Machine Learning Research , pages 5391-5412. PMLR, 2022. URL https://proceedings.mlr.press/v178/liu22h.html . ↑ 36

- [66] Jingcheng Liu, Jalaj Upadhyay, and Zongrui Zou. Optimal bounds on private graph approximation. In Proceedings of the 2024 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA) , pages 1019-1049, 2024. doi: 10.1137/1.9781611977912.39. URL https://doi.org/10.1137/1.9781611977912.39 . ↑ 1, ↑ 2, ↑ 3, ↑ 19
- [67] Jingcheng Liu, Jalaj Upadhyay, and Zongrui Zou. Almost linear time differentially private release of synthetic graphs. arXiv preprint arXiv:2406.02156 , 2024. ↑ 2, ↑ 3
- [68] Xizhi Liu and Sayan Mukherjee. Tight query complexity bounds for learning graph partitions. In Conference on Learning Theory , pages 167-181. PMLR, 2022. ↑ 36
- [69] Konstantin Makarychev, Yury Makarychev, and Aravindan Vijayaraghavan. Correlation clustering with noisy partial information. In Conference on Learning Theory , pages 1321-1342. PMLR, 2015. ↑ 36
- [70] Frank McSherry and Kunal Talwar. Mechanism design via differential privacy. In Proceedings of the 48th Annual IEEE Symposium on Foundations of Computer Science (FOCS) , pages 94-103. IEEE, 2007. ↑ 1, ↑ 5
- [71] Karl Menger. Zur allgemeinen kurventheorie. Fundam. Math. , 10(1):96-115, 1927. ↑ 2
- [72] Sagnik Mukhopadhyay and Danupon Nanongkai. A note on isolating cut lemma for submodular function minimization. arXiv preprint arXiv:2103.15724 , 2021. ↑ 8
- [73] Ketan Mulmuley, Umesh V. Vazirani, and Vijay V. Vazirani. Matching is as easy as matrix inversion. Comb. , 7(1):105-113, 1987. doi: 10.1007/BF02579206. URL https://doi.org/ 10.1007/BF02579206 . ↑ 17
- [74] Arvind Narayanan and Vitaly Shmatikov. Robust de-anonymization of large sparse datasets. In Proceedings of the 2008 IEEE Symposium on Security and Privacy (SP) , pages 111-125. IEEE, 2008. ↑ 1
- [75] Debmalya Panigrahi. Gomory-Hu trees. In Encyclopedia of Algorithms , pages 858-861. Springer, 2016. doi: 10.1007/978-1-4939-2864-4\_168. URL https://doi.org/10.1007/ 978-1-4939-2864-4\_168 . ↑ 3
- [76] Richard Peng, He Sun, and Luca Zanetti. Partitioning well-clustered graphs: Spectral clustering works! In Conference on learning theory , pages 1423-1455. PMLR, 2015. ↑ 36
- [77] Seth Pettie, Thatchaphol Saranurak, and Longhui Yin. Optimal vertex connectivity oracles. In Stefano Leonardi and Anupam Gupta, editors, Proceedings of the 54th Annual ACM SIGACT Symposium on Theory of Computing (STOC) , pages 151-161. ACM, 2022. doi: 10.1145/ 3519935.3519945. URL https://doi.org/10.1145/3519935.3519945 . ↑ 2
- [78] Sofya Raskhodnikova, Satchit Sivakumar, Adam D. Smith, and Marika Swanberg. Differentially private sampling from distributions. In Marc'Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Proceedings of the Advances in Neural Information Processing Systems 34 (NeurIPS) , pages 28983-28994, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ f2b5e92f61b6de923b063588ee6e7c48-Abstract.html . ↑ 1
- [79] Shota Saito and Mark Herbster. Multi-class graph clustering via approximated effective p -resistance. In International Conference on Machine Learning , pages 29697-29733. PMLR, 2023. ↑ 36
- [80] Huzur Saran and Vijay V. Vazirani. Finding k -cuts within twice the optimal. SIAM J. Comput. , 24(1):101-108, 1995. ↑ 36, ↑ 37
- [81] Adam Sealfon. Shortest paths and distances with differential privacy. In Proceedings of the 35th ACM SIGMOD-SIGACT-SIGAI Symposium on Principles of Database Systems (PODS) , pages 29-41, 2016. ↑ 4, ↑ 20, ↑ 37
- [82] Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In Proceedings of the 2017 IEEE Symposium on Security and Privacy (SP) , pages 3-18. IEEE, 2017. ↑ 1

- [83] Thomas Steinke. Composition of Differential Privacy &amp; Privacy Amplification by Subsampling. arXiv e-prints , art. arXiv:2210.00597, October 2022. doi: 10.48550/arXiv.2210.00597. ↑ 3
- [84] Jalaj Upadhyay, Sarvagya Upadhyay, and Raman Arora. Differentially private analysis on graph streams. In The 24th International Conference on Artificial Intelligence and Statistics (AISTATS) , pages 1171-1179, 2021. URL http://proceedings.mlr.press/v130/ upadhyay21a.html . ↑ 2
- [85] Jan van den Brand, Yin Tat Lee, Yang P. Liu, Thatchaphol Saranurak, Aaron Sidford, Zhao Song, and Di Wang. Minimum cost flows, mdps, and ℓ 1 -regression in nearly linear time for dense instances. In Samir Khuller and Virginia Vassilevska Williams, editors, Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing (STOC) , pages 859-869. ACM, 2021. doi: 10.1145/3406325.3451108. URL https://doi.org/10.1145/3406325.3451108 . ↑ 36
- [86] Nate Veldt, David Gleich, and Michael Mahoney. A simple and strongly-local flow-based method for cut improvement. In International Conference on Machine Learning , pages 19381947. PMLR, 2016. ↑ 36
- [87] Jun Wang, Tony Jebara, and Shih-Fu Chang. Semi-supervised learning using greedy max-cut. The Journal of Machine Learning Research , 14(1):771-800, 2013. ↑ 36
- [88] Tianyi Zhang. Gomory-Hu trees in quadratic time. arXiv preprint arXiv:2112.01042 , 2021. ↑ 3

## A Preliminaries

## A.1 Notation

We use G = ( V, E, w ) to denote a weighted, undirected graph with vertex set V , edge set E and edge weights w . For a subset of vertices S ⊆ V , we use ∂ G S , or simply ∂S when G is clear from context, to denote the set of edges between S and V \ S . For a set of edges Q ⊆ E , we use w ( Q ) to denote the sum of the weights of the edges in Q . For a set of vertices S ⊆ V , we use w ( S ) to denote w ( ∂S ) , for simplicity.

For s, t ∈ V , we use λ G ( s, t ) to denote the Mins -t -Cut value in a specified graph G . When clear from context, the subscript G might be omitted. Throughout our computation, we assume that all Mins -t -Cuts or Min Isolating Cuts are unique. This is without loss of generality by adding small noise to the edges and applying the isolation lemma [73] (see also [16, 2]). 10 For n &gt; 0 , lg( n ) is logarithm base2 and ln( n ) is logarithm basee . Unless specified, high probability refers to failure probability of at most 1 /n O (1) where we can pick the O (1) factor to be an arbitrarily large constant of our choice.

Table 2: A summary of our notation throughout the paper. When clear from the context, we drop the subscript of the graph G = ( V, E, w ) .

| Notation            | Meaning                                  | Reference      |
|---------------------|------------------------------------------|----------------|
| ε                   | privacy parameter                        | Definition A.4 |
| ∂ G S,S ⊆ V         | set of edges between S and V \ S in G    | Appendix A.1   |
| w G ( Q ) ,Q ⊆ E    | sum of weights of edges in Q in G        | Appendix A.1   |
| w G ( S ) ,S ⊆ V    | sum of weights of edges in ∂S in G       | Appendix A.1   |
| λ G ( s, t )        | min s - t cut value in G                 | Appendix A.1   |
| S u                 | isolating cut for a vertex u             | Definition 2.1 |
| ˆ w G ( · ) , ˆ S u | privatized versions of w G ( · ) and S u | Algorithm 2    |

## A.2 Graph Cuts

In our algorithms, we use the notion of vertex contractions , which we formally define next.

Definition A.1 (Vertex contractions) . Let X ⊆ V be a subset of vertices of the graph G = ( V, E, w ) . Contracting the set X into a vertex is done as follows: we add a vertex x to the graph and remove all the vertices in X from the graph. Then for every vertex v ∈ V \ X , we add an edge from x to v with weight ∑ x ′ ∈ X w ( x ′ v ) . Note that if none of the vertices in X has an edge to v , then there is no edge from x to v .

We use the submodularity property of cuts in many of our proofs.

Lemma A.1 (Submodularity of Cuts [27]) . For any graph G = ( V, E, w ) , and any two subsets S, T ⊆ V , it holds

<!-- formula-not-decoded -->

Recall the definition of Min Isolating Cuts problem (see Definition 2.1). We use the following simple fact.

Fact A.1 ([62]) . Given a set of terminals R ⊆ V , there always exists a set of minimum isolating cuts { S v : v ∈ R } such that the cuts are disjoint.

Definition A.2 (Gomory-Hu Steiner tree [61]) . Given a graph G = ( V, E, w ) and a set of terminals U ⊆ V , the Gomory-Hu Steiner tree is a weighted tree T on the vertices U , together with a function

10 The isolation lemma involves adding noise to existing edge weights. This noise can be bounded in magnitude to be at most 1 / poly( n ) , so scaling the weights down by a (1+1 / poly( n )) factor yields the normal neighboring definition for DP.

f : V → U , such that: For all s, t ∈ U , consider the minimum-weight edge uv on the unique s -t path in T . Let U 0 be the vertices of the connected component of T -uv containing s . Then, the set f -1 ( U 0 ) ⊆ V is the Mins -t -Cut, and its value is w T ( uv ) .

Note that for U = V and f ( v ) = v the Gomory-Hu Steiner tree equals the Gomory-Hu tree.

## A.3 Concentration Inequalities

Theorem A.1 (Sums of Exponential Random Variables ([54, Theorem 5.1])) . Let X 1 , . . . , X N be independent random variables with X i ∼ Exp ( a i ) . Let µ = ∑ N i =1 1 a i be the expectation of the sum of the X i 's and let a ∗ = min i a i . Then, for any t ≥ 1 ,

<!-- formula-not-decoded -->

Corollary A.1. Let X 1 , . . . , X N be independent random variables with X i ∼ Exp ( a ) for some real number a &gt; 0 . Let µ = N/a be the expectation of the sum of the X i 's. Then, for any T ≥ 2 µ ,

<!-- formula-not-decoded -->

Proof. Apply Theorem A.1 with a ∗ = a and t = T/µ , we obtain

<!-- formula-not-decoded -->

□

## A.4 Differential Privacy

In this paper, we focus on a weighted version of edge differential privacy. At the end of this section, we discuss why this choice of neighboring graphs is made as well as some connections between notions of neighboring graphs.

̸

Definition A.3 (Edge-Neighboring Graphs) . Graphs G = ( V, E, w ) and G ′ = ( V, E ′ , w ′ ) are called edge-neighboring if there is uv ∈ V 2 such that | w G ( uv ) -w G ′ ( uv ) | ≤ 1 and for all u ′ v ′ = uv , u ′ v ′ ∈ V 2 , we have w G ( u ′ v ′ ) = w G ′ ( u ′ v ′ ) . Note that w ( u ′′ v ′′ ) = 0 for all non-edges u ′′ v ′′ .

Definition A.4 (Differential Privacy [36]) . A (randomized) algorithm A is ( ε, δ ) -private (or ( ε, δ ) -DP) if for any neighboring graphs G and G ′ and any set of outcomes O ⊂ Range ( A ) it holds

<!-- formula-not-decoded -->

When δ = 0 , algorithm A is pure differentially private , or ε -DP.

We now state some standard properties of differential privacy which we will utilize in our algorithm design and analysis.

Theorem A.2 (Basic composition [36, 33]) . Let ε 1 , . . . , ε t &gt; 0 and δ 1 , . . . , δ t ≥ 0 . If we run t (possibly adaptive) algorithms where the i -th algorithm is ( ε i , δ i ) -DP, then the entire algorithm is ( ε 1 + . . . + ε t , δ 1 + . . . + δ t ) -DP.

Theorem A.3 (Laplace mechanism [34]) . Consider any function f which maps graphs G to R d with the property that for any two neighboring graphs G,G ′ , ∥ f ( G ) -f ( G ′ ) ∥ 1 ≤ ∆ . Then, releasing

<!-- formula-not-decoded -->

where each X i is i.i.d. with X i ∼ Lap (∆ /ε ) satisfies ε -DP.

Figure 3: This example describes two weighted graphs, G and G ′ , that differ by a single edge { u, v } : G is a path, while G ′ is a cycle. The numbers next to the edges are their weights. Note that both graphs have n -1 distinct min-cut values. However, the min-cut values in G are { 2 , 4 , . . . , 2( n -1) } , while the min-cut values in G ′ are { 3 , 5 , . . . , 2( n -1) + 1 } . Thus, the min-cut value sets of these two neighboring graphs differ in n -1 entries.

<!-- image -->

Note that any w ( S ) for S ⊂ V has sensitivity ∆ ≤ 1 as changing one edge weight by one can change the sum of a subset of edge weights by at most one. We now state a result on privately releasing an approximate Mins -t -Cut for a single pair of vertices s, t .

̸

Theorem A.4 (Private Mins -t -Cut [29]) . Fix any ε &gt; 0 . There is an ( ε, 0) -DP algorithm PrivateMin-s-t-Cut ( G = ( V, E, w ) , s, t, ε ) for s = t ∈ V that reports an s -t cut for n -vertex weighted graphs that is within O ( n ε ) additive error from the Mins -t -Cut with high probability.

By standard techniques, we can also use Theorem A.4 to design an ( ε, 0) -DP algorithm for computing an approximate MinS -T -Cut for two disjoint subsets S, T ⊆ V that is within O ( n ε ) additive error from the actual MinS -T -Cut (e.g., by contracting all vertices in S and all vertices in T to two supernodes). Furthermore, our final algorithm is recursive with many calls to Private MinS -T -Cut for graphs with few vertices, and it is not enough to succeed with high probability with respect to n . The error analysis of [29] shows that the error is bounded by the sum O ( n ) random variables distributed as Exp ( ε ) . Using Corollary A.1 yields the following corollary:

Corollary A.2 (Private MinS -T -Cut) . Fix any ε &gt; 0 , there exists an ( ε, 0) -DP algorithm PrivateMin-S-T-Cut ( G = ( V, E, w ) , S, T, ε, β ) for disjoint S, T ⊆ V that reports a set C ⊆ V where S ⊆ C and C ∩ T = ∅ , and w ( ∂C ) is within O ( n +log(1 /β ) ε ) additive error from the true minS -T -cut with probability at least 1 -β .

## A.5 On various notions of neighboring graphs

For graph data, there are several choices of neighboring datasets with very different semantics for the privacy they correspond to.

At one extreme are vertex-neighboring graphs, where neighboring graphs differ arbitrarily in the edges incident on a single vertex, e.g., [14]. Semantically, each vertex corresponds to a person, and vertex differential privacy protects the data of that individual person. While this offers broad protection, this notion of privacy is simply too restrictive for cut problems. It has been found to be useful in simpler problems, such as estimating the edge density of random graphs. However, the value of any cut can change arbitrarily between neighboring graphs, so no approximation of any cut value is possible while maintaining privacy.

We consider edge-neighboring graphs, which is the standard for cut problems in the literature, see, e.g., [46, 47, 37, 66, 29]. It is the strongest form of privacy for which we can get a meaningful approximation to cut problems. For unweighted graphs, neighboring graphs are those in which a single edge has been added/removed. For weighted graphs, two related notions are considered: where a single edge can change in weight by 1 and where all edges can change in total ℓ 1 distance 1 11 . Semantically, these notions of privacy are meaningful if individuals impact the existence of edges and the size of edge weights. We use the former definition but note that, as is often the case, our result applies in both settings. We outline the reduction below.

11 Note that both of these capture the unweighted case as zero weight edges are equivalent to edges not belonging to the graph in the context of cut problems.

Let A be an algorithm satisfying the former notion of edge differential privacy (changing a single edge by weight 1 ) with an approximation error that depends linearly on 1 /ε and is scale-invariant in that it does not explicitly depend on the scale of the edge weights in the graph. Say two graphs have edge weight vectors w G , w G ′ ∈ R ( n 2 ) with ∥ w G -w G ′ ∥ 1 ≤ 1 . Let ∆ = w G -w G ′ , and let C be some constant such that C ∆ is integer-valued - we assume that such a constant exists either due to finite precision of the weights or by rounding the weights to finite precision with arbitrarily small loss. Note that ∥ C ∆ ∥ 1 ≤ C . By group privacy, running A on a graph with edge weights Cw G will preserve a Cε -DP guarantee with respect to all graphs which are formed by starting with Cw G and iteratively changing C edge weights by 1 . In particular, this holds for Cw G ′ . Rescaling the solution by 1 /C yields a reduction of error by a factor of 1 /C at the cost of increasing the privacy parameter by a factor of C . As the error of A scales linearly in 1 /ε , rescaling the privacy parameter by a factor of C yields an equivalent error/privacy tradeoff for the ℓ 1 notion of neighboring graphs.

We note that a yet more restrictive notion of privacy is used for privacy in the context of shortest path problems, e.g., [81, 22]. In the context of shortest paths, a zero-weight edge is not the same as a non-edge: a non-edge is equivalent to an edge of infinite weight. Therefore, standard notions of edge differential privacy do not allow for any approximation to path lengths. The notion considered in these shortest path problems is edge weight differential privacy, where the unweighted topology of the graph is fixed and made public while the weights on each edge are private. Specifically, neighboring graphs have the same edge set E but differ in weights on those edges in ℓ 1 distance at most 1 . This is in contrast to the former notion of edge differential privacy, where two graphs are neighboring if one contains an edge of weight 1 and the other has a non-edge in that location.

## B Private Min Isolating Cuts

In this section we prove Theorem 2.1. In fact, we prove a stronger version given by Lemma B.1. The steps in the algorithm that differ meaningfully from the non-private version are in color. We refer to Appendix A.1 and Table 2 for the definitions of our notation.

```
1 Initialize W r ← V for every r ∈ R 2 Identify R with { 0 , . . . , | R | -1 } 3 for i from 0 to ⌊ lg( | R | -1) ⌋ do 4 A i ←{ r ∈ R : r mod 2 i +1 < 2 i } 5 C i ← PrivateMin-S-T-Cut ( G,A i , R \ A i , ε/ (lg | R | +3) , β/ (lg | R | +3)) 6 W r ← W r ∩ C i for every r ∈ A i 7 W r ← W r ∩ ( V \ C i ) for every r ∈ R \ A i 8 end 9 for r ∈ R do 10 Let H r be G with all vertices in V \ W r contracted, and let t r be the contracted vertex 11 In H r , add weight B H · ( n +lg(1 /β )) lg 2 ( | R | ) } ε | U | between every vertex in W r ∩ U and t r for some sufficiently large constant B H 12 end 13 H ← ⋃ r ∈ R H r 14 C ← PrivateMin-S-T-Cut ( H , R, { t r } r ∈ R , ε/ (lg | R | +3) , β/ (lg | R | +3)) 15 return {C ∩ W r } r ∈ R
```

Algorithm 1: PrivateMinIsolatingCuts ( G = ( V, E, w ) , R, U, ε, β )

Lemma B.1. On a graph G with n vertices, a set of terminals R ⊆ V , another set of vertices U ⊆ V , and a privacy parameter ε , there is an ( ε, 0) -DP algorithm PrivateMinIsolatingCuts ( G,R,U,ε,β ) that returns a set of Isolating Cuts over terminals R . The total cut values of the Isolating Cuts is within additive error O (( n +lg(1 /β )) lg 2 ( | R | ) /ε ) from the Min Isolating Cuts with probability 1 -β .

Furthermore, suppose the Min Isolating Cut for any terminal r ∈ R contains at most 0 . 5 | U | vertices from U . In that case, the Isolating Cut for r returned by the algorithm will, with probability 1 -β , contain at most 0 . 9 | U | vertices from U .

Proof. The algorithm is presented in Algorithm 1. On a high level, the algorithm follows the nonprivate Min Isolating Cuts algorithm by [62, 3], but replacing all calls to MinS -T -Cut with private MinS -T -Cut from Corollary A.2. One added step is Line 10, which is used to provide the guarantee that if the Min Isolating Cut for a terminal r ∈ R contains a small number of vertices in U , then the isolating cut for terminal r returned by the algorithm also does.

Next, we explain the algorithm in more detail. For every r ∈ R , we maintain a set W r that should contain the r -side of a cut seperating r from R \ { r } obtained in the algorithm. In each of the ⌊ lg( | R | -1) ⌋ +1 iterations, we find a subset A i ⊆ R , and find a cut that separates A i from R \ A i . Let C i be the side of the cut containing A i . Then for every r ∈ A i , we update W r with W r ∩ C i ; for every r ∈ R \ A i , we update W r with W r ∩ ( V \ C i ) . The choice of A i is so that every pair r 1 , r 2 ∈ R are on different sides of the MinS -T -Cut in at least one iteration; as a result, W r ∩ R = { r } for every r ∈ R after all iterations.

Next, for every r ∈ R , the algorithm aims to compute a cut separating r from R \ { r } , where the side containing r is inside W r . This can be done by contracting all vertices outside of W r to a vertex t r and computing private Minr -t r -Cut. To incentivize cuts that contain fewer vertices in U on the side containing r , the algorithm adds an edge with positive weight from every vertex in W r ∩ U to t r . Finally, these private Minr -t r -Cut instances can be solved at once by combining them into a single graph H .

Privacy analysis. Considering the first for loop in the algorithm, The only parts that depend on the edges or edge weights are the calls to PrivateMin-S-T-Cut. Each call to PrivateMin-S-T-Cut is ( ε/ (lg | R | +3) , 0) -DP, and the number of calls is ⌊ lg( | R | -1) ⌋ +1 , so this part of the algorithm is log | R | +1 log | R | +3 ε -DP via basic composition (Theorem A.2).

Next, note that the sets W r are private since they are obtained from postprocessing the lg | R | privately computed min cuts C i . Furthermore, the sets W r form a partition of V . This implies that an edge in the initial graph contributes its weight to at most two edges in H . Namely, an edge internal to some W r appears only in H r and an edge between some W r and W r ′ is only contracted into an edge in H r and an edge in H r ′ . Thus, the sensitivity of H is 2 , so running PrivateMin-S-T-Cut on H is 2 ε/ (log | R | +3 -DP. Hence, the overall algorithm is ε -DP.

Error analysis. First, we analyze the error introduced by the for loop starting at Line 3. Let { S r } r ∈ R be the (non-private) Min Isolating Cuts for terminals in R , which are only used for analysis purposes. Recall by Fact A.1, we can assume { S r } r ∈ R are disjoint. Take an iteration i of the for loop and let { W r } r ∈ R be the values of W r 's before the start of the iteration, and let { W ′ r } r ∈ R denote the value of W r 's at the end of the iteration.

Recall that with probability 1 -β/ (lg( | R | )+3) , the PrivateMin-S-T-Cut algorithm Corollary A.2 has additive error O (( n +lg((lg( | R | ) + 3) /β ))(lg( | R | ) + 3) /ε ) = O (( n +lg(1 /β )) lg( | R | ) /ε ) . Hence, in the following analysis, we assume all calls of the PrivateMin-S-T-Cut algorithm have additive error O (( n +lg(1 /β )) lg( | R | ) /ε ) (which holds with probability 1 -β by union bound).

We show the following claim:

Claim 1. It holds that

<!-- formula-not-decoded -->

Proof. We first show ∑ r ∈ A i w ( W ′ r ∩ S r ) ≤ ∑ r ∈ A i w ( W r ∩ S r ) + O ( n lg( | R | ) /ε ) . Let S A i := ⋃ r ∈ A i ( W r ∩ S r ) . By Lemma A.1,

<!-- formula-not-decoded -->

Recall that with probability 1 -β/ (lg( | R | ) +3) , C i is within O (( n +lg((lg( | R | ) +3) /β ))(lg( | R | ) + 3) /ε ) = O (( n +lg(1 /β )) lg( | R | ) /ε ) of the minimum cut separating A i and R \ A i , by the guarantee of Corollary A.2, and note that S A i ∪ C i is also a cut separating A i and R \ A i . Therefore,

<!-- formula-not-decoded -->

Combining Equations (1) and (2), we get that

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

̸

By an analogous argument, we can show ∑ r ∈ R \ A i w ( W ′ r ∩ S r ) ≤ ∑ r ∈ R \ A i w ( W r ∩ S r ) + O ( n lg( | R | ) /ε ) . Summing up the two inequalities gives the desired claim. □

By applying Claim 1 repeatedly, we can easily show the following claim:

Claim 2. At the end of the for loop starting at Line 3, ∑ r ∈ R w ( W r ∩ S r ) ≤ ∑ r ∈ R w ( S r ) + O (( n + lg(1 /β )) lg 2 ( | R | ) /ε ) .

Proof. Before the for loop starting at Line 3, we have W r = V for every r ∈ R , so ∑ r ∈ R w ( W r ∩ S r ) = ∑ r ∈ R w ( S r ) . Claim 1 shows that after each iteration of the for loop, the quantity ∑ r ∈ R w ( W r ∩ S r ) does not increase by more than O (( n +lg(1 /β )) lg( | R | ) /ε ) . As there are O (lg( | R | )) iterations, we get that at the end of the for loop,

<!-- formula-not-decoded -->

The following claim is a simple observation:

Claim 3. At the end of the for loop starting at Line 3, r ∈ W r for every r ∈ R and distinct W r 's are disjoint.

Next, we show that the Minr -t r -Cut values in H r are close to the Min Isolating Cut values:

<!-- formula-not-decoded -->

Proof. We have that

<!-- formula-not-decoded -->

̸

̸

Because of Claim 4, and the guarantee of Corollary A.2, the final cuts C ∩ W r returned by the algorithm will have the property that

<!-- formula-not-decoded -->

Furthermore, w G ( C ∩ W r ) ≤ w H r ( C ∩ W r ) as we only add positive weights to H r compared to G , we further get

<!-- formula-not-decoded -->

which is the desired error bound.

Additional guarantee. Finally, we need to show that if | S r ∩ U | ≤ 0 . 5 U for some r ∈ R , then with probability 1 -β , the returned isolating cut by the algorithm C ∩ W r has |C ∩ W r ∩ U | ≤ 0 . 9 U . Again, we can assume all calls of the PrivateMin-S-T-Cut algorithm have additive error O (( n +lg(1 /β )) lg( | R | ) /ε ) . As C is a cut returned by the PrivateMin-S-T-Cut algorithm separating R and { t r } r ∈ R and ( C \ W r ) ∪ ( S r ∩ W r ) is also a cut separating R and { t r } r ∈ R , we have

<!-- formula-not-decoded -->

̸

By removing cut values contributed by H r ′ for r ′ = r from both sides, we get that

<!-- formula-not-decoded -->

Rewriting the cut values in terms of the edge weights of G instead of H r (recall the weights of G and H r are related as shown in Line 11), the above becomes

<!-- formula-not-decoded -->

Because w ( C ∩ W r ) ≥ w ( S r ) by definition of S r , the above implies

<!-- formula-not-decoded -->

as desired.

## C Core Recursive Step

We now describe a key subroutine, outlined as Algorithm 2, used to compute a DP Gomory-Hu tree. The high-level goal is to use Min Isolating Cuts to find minimum cuts that cover a large fraction of vertices in the graph. The overall structure of this algorithm follows that of the prior work [61] with several key changes to handle additive approximations and privacy. The inputs to Algorithm 2 are the weighted graph, a source vertex s , a set of active vertices U ⊆ V , a privacy parameter ε , and a failure probability β . The steps that differ meaningfully from the non-private version developed in [61] are in color. To obtain a DP version of this method, Algorithm 2 invokes the DP Mins -t -Cut and the DP Min Isolating Cuts algorithm; the latter primitive is developed in this work in Appendix B. In the original non-private algorithm, isolating cuts S i v are included in D i if the set S i v corresponds to the v side of the Mins -v -Cut, i.e., w ( S i v ) = λ ( s, v ) . The analysis in prior work relies on this equality, i.e., on w ( S i v ) and λ ( s, v ) being the same, in a crucial way. Informally speaking, it enables the selection of many Min Isolating Cuts of the right size. In our case, since the cuts and their values are released

□

Algorithm 2: PrivateGHTreeStep ( G = ( V, E, w ) , s, U, ε, β )

```
1 Γ iso ← O ( ( n +lg(1 /β )) lg 2 ( | U | ) ε ) and Γ values ← O ( | U | lg( | U | /β ) ε ) 2 ˆ λ ( s, v ) ← λ ( s, v ) + Lap ( 4( | U |-1) ε ) for all v ∈ U \ { s } 3 Initialize R 0 ← U 4 for i from 0 to ⌊ lg | U |⌋ do 5 Call PrivateMinIsolatingCuts ( G,R i , U, ε 2( ⌊ lg | U |⌋ +1) , β ⌊ lg | U |⌋ +1 ) (Algorithm 1) obtaining disjoint sets ˆ S i v ; /* v ranges over vertices in R i */ 6 ˆ w ( ˆ S i v ) ← w ( ˆ S i v ) + Lap ( 8( ⌊ lg | U |⌋ +1) ε ) for each v ∈ R i \ { s } 7 Let D i ⊆ U be the union of ˆ S i v ∩ U over all v ∈ R i \ { s } satisfying ˆ w ( ˆ S i v ) ≤ ˆ λ ( s, v ) + (2( ⌊ lg | U |⌋ -i ) + 1)Γ iso +Γ values and | ˆ S i v ∩ U | ≤ (9 / 10) | U | 8 R i +1 ← sample of U where each vertex in U \ { s } is sampled independently with probability 2 -i +1 , and s is sampled with probability 1
```

- 10 return D (the largest set D i ), R (the set of terminals v ∈ R i \ { s } satisfying the conditions on Line 7), and sets ˆ S i for v ∈ R .

```
9 end v
```

privately by random perturbations, it is unclear how to test that condition with equality. On the other hand, we still would like to ensure that many isolating cuts have 'the right' size. Among our key technical contributions is relaxing that condition by using a condition which changes from iteration to iteration of the for-loop. The actual condition we use is

<!-- formula-not-decoded -->

on Line 7 of Algorithm 2 where Γ iso and Γ values are upper bounds on the additive errors of the approximate Min Isolating Cuts and the approximate Mins -v -Cut values, respectively.

When using Equation (4), we also have to ensure that significant progress can still be made, i.e., to ensure that both (a) we will find a large set D i which is the union of approximate Min Isolating Cuts ˆ S i v satisfying the condition above and (b) none of the individual ˆ S which we return are too large as we will recurse within each of these sets. A new analysis uses this changing inequality to show that the former is true. For the latter, we utilize the special property of our PrivateMinIsolatingCuts in Appendix B that forces an approximate isolating cut to contain at most 0 . 9 | U | terminals if there exists an exact isolating cut of size at most | U | / 2 . We now turn to the analysis.

## C.1 Correctness

As in prior work [61, 4], let D ∗ ⊆ U \ { s } be the set of vertices v such that if S ∗ v is the v side of the Mins -v -Cut, | S ∗ v ∩ U | ≤ | U | / 2 .

Lemma C.1. PrivateGHTreeStep ( G,U,s,ε, β ) (Algorithm 2) has the following properties:

- Let Γ iso = C 1 ( n +lg(1 /β )) lg 2 ( | U | ) /ε and Γ values = C 2 | U | lg( | U | /β ) /ε for large enough constants C 1 , C 2 . Let { S i v } v ∈ R i be the optimal Min Isolating Cuts for terminals R i (by Fact A.1, these are disjoint without loss of generality). Let R ∗ be the set of vertices v for which Algorithm 2 returns ˆ S i v . Then with probability at least 1 -O ( β ) , the sets { ˆ S i v : v ∈ R ∗ } returned by Algorithm 2 are approximate Min Isolating Cuts and approximate Minv -s -Cuts:

and, for all v ∈ R ∗ ,

<!-- formula-not-decoded -->

- D returned by the algorithm satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To prove this, we will need the following helpful definition and lemma. Let X i v be a random variable for the number of vertices in U added to D i by a set ˆ S i v :

<!-- formula-not-decoded -->

Lemma C.2. Consider a vertex v ∈ D ∗ . Let S i v be the v part of the optimal solution to Min Isolating Cuts at stage i . Assume that | λ ( s, v ) -ˆ λ ( s, v ) | ≤ Γ values and | w ( ˆ S i v ) -ˆ w ( ˆ S i v ) | + | w ( S i v ) -w ( ˆ S i v ) | ≤ Γ iso for all i ∈ { 0 , . . . ⌊ lg | U |⌋} . Additionally, if | S i v ∩ U | ≤ | U | / 2 , assume ˆ S i v ≤ (9 / 10) | U | . Then, there exists an i ′ ∈ { 0 , . . . ⌊ lg | U |⌋} such that

<!-- formula-not-decoded -->

Proof. Consider a specific sampling level i ∈ { 0 , . . . , ⌊ lg | U |⌋} . We say that i is 'active' if there exists a set ˜ S i v ⊂ U containing v and not s such that | ˜ S i v ∩ U | ∈ [2 i , 2 i +1 ) and

<!-- formula-not-decoded -->

Note that this is a deterministic property regarding the existence of such a set ˜ S i v independent of the randomness used to sample terminals or find private Min Isolating Cuts.

Let i ′ be the smallest active i . Recall S ∗ v is the v side of the true Mins -v -Cut. Let i ∗ = ⌊ lg | S ∗ v ∩ U |⌋ . As w ( S ∗ v ) = λ ( s, v ) ≤ ˆ λ ( s, v ) + Γ values , i ∗ must be active, so i ′ is well-defined and i ′ ≤ i ∗ . As i ′ is active, there exists a set ˜ S i ′ v with | ˜ S i ′ v ∩ U | ∈ [2 i ′ , 2 i ′ +1 ) and with cost within 2( ⌊ lg( | U | ) ⌋ -i ′ )Γ iso + Γ values of ˆ λ ( s, v ) . On the other hand, as i ′ is the smallest active level, there is no set containing v but not s , whose intersection with U is less than 2 i ′ , and whose cost is within 2( ⌊ lg( | U | ) ⌋ -( i ′ -1))Γ iso +Γ values of ˆ λ ( s, v ) .

Consider the event that in R i ′ , we sample v but no other vertices in ˜ S i ′ v as terminals, i.e., R i ′ ∩ ˜ S i ′ v = { v } . Then, ˜ S i ′ v is a cut separating v and R i ′ \ { v } . By the assumed guarantee of in the lemma statement, the actual cut we output has approximated cost:

<!-- formula-not-decoded -->

For sake of contradiction, assume that | ˆ S i ′ v ∩ U | &lt; 2 i ′ . Using the fact that all solutions of this size have large cost, we can conclude that

<!-- formula-not-decoded -->

This contradicts the previous inequality that shows that ˆ w ( ˆ S i ′ v ) is upper bounded by this quantity, so | ˆ S i ′ v ∩ U | ≥ 2 i ′ as long as the sampling event occurs.

Next, we show that | ˆ S i ′ v ∩ U | ≤ (9 / 10) | U | . As v ∈ D ∗ , the true minimum cut S ∗ v has the property | S ∗ v ∩ U | ≤ | U | / 2 . Furthermore, by the isolating cuts lemma of [62], there is one minimum isolating cut solution for any set of terminals including v and s will have that the v part S v is a subset of S ∗ v (this is the basis for the isolating cuts algorithm). So, if v is sampled in R i , there will exist an optimal isolating cuts solution S i v with | S i v ∩ U | ≤ | U | / 2 . By the assumption in the lemma statement, | ˆ S i ′ v ∩ U | ≤ (9 / 10) | U | .

Overall, we can bound the contribution of ˆ S i ′ v to D i ′ as

<!-- formula-not-decoded -->

If i ′ = 0 , this evaluates to 1 . Otherwise, if i ′ ≥ 1 ,

<!-- formula-not-decoded -->

We are now ready to prove the main lemma of this section.

Proof of Lemma C.1. The first step of the proof will be to show that Γ iso and Γ values upper-bound the error of the approximate isolating cuts and min cut values used in the algorithm with probability 1 -O ( β ) . Applying the guarantee of Lemma B.1 and union bounding over all i , the following guarantee of the quality of ˆ S i v holds with probability 1 -β . If { S i v } are optimal Min Isolating Cuts for terminals R i :

<!-- formula-not-decoded -->

We remark that w ( ˆ S i v ) ≥ w ( S i v ) due to the optimality of S i v .

Now noting (from Line 6 of Algorithm 2) that ˆ w ( ˆ S i v ) -w ( ˆ S i v ) is a Laplace random variable Lap ( 8( ⌊ lg | U |⌋ +1) ε ) and that the absolute value of it is distributed as Exp ( ε 8( ⌊ lg | U |⌋ +1) ) [60].

Note that the expected value of ∑ v ∈ R i | w ( ˆ S i v ) -ˆ w ( ˆ S i v ) | is 8 | R i | ( ⌊ lg | U |⌋ +1) ε = O ( n log | U | ε ) . We then apply Corollary A.1 with a = ε 8( ⌊ lg | U |⌋ +1) and T = Θ( ( n +log(1 /β )) log | U | ε ) so that

<!-- formula-not-decoded -->

when the constant factor hidden in the bound for T is sufficiently large.

Therefore, with probability 1 -O ( β ) , for all i ∈ { 0 , . . . , ⌊ lg | U |⌋} , it holds

<!-- formula-not-decoded -->

This satisfies the approximate Min Isolating Cuts guarantee of the lemma.

Similarly, the additional guarantee in Lemma C.2 (if | S i v ∩ U | ≤ | U | / 2 , assume ˆ S i v ≤ (9 / 10) | U | ) also holds with probability 1 -O ( β ) by union bound.

For the approximate min cut values, by the tail of the Laplace distribution and a union bound, each ˆ λ ( s, v ) satisfies

<!-- formula-not-decoded -->

with probability 1 -β . We condition on these events moving forward.

Next, we show that sets ˆ S i v are only included in our output if they are close to the min cut value λ ( s, v ) . Specifically, for any set returned by our algorithm:

<!-- formula-not-decoded -->

Applying the error guarantees for ˆ w ( ˆ S i v ) , ˆ S i v , and ˆ λ ( s, v ) derived above,

<!-- formula-not-decoded -->

This completes the first part of the proof concerning the error of the returned sets. In the remainder, we focus on the cardinality of the output.

By definition of X i v , the size of D i is given by a sum over X i v :

<!-- formula-not-decoded -->

By linearity of expectation and as D ∗ ⊆ U \ { s } ,

<!-- formula-not-decoded -->

As we output the largest D i across all i , the output of our algorithm will have an expected size of at least

<!-- formula-not-decoded -->

Via Lemma C.2 (note that the error condition holds with probability 1 -O ( β ) from the first part of this proof), the expected output size will be at least

<!-- formula-not-decoded -->

## C.2 Privacy

We now analyze the privacy guarantee of our algorithm. A key technical observation behind our analysis is that the (approximate) isolating cuts found by Algorithm 2 are disjoint subsets of vertices, so any edge can only appear in at most 2 sets at any sampling level in the for loop starting at Line 4. This is formalized in the lemma below.

Lemma C.3. PrivateGHTreeStep (Algorithm 2) is ε -DP.

Proof. The algorithm PrivateGHTreeStep interacts with the sensitive edges only through calculations of approximate min cut values ˆ λ ( s, v ) , approximate min isolating cut values ˆ w ( ˆ S i v ) , and calls to PrivateMinIsolatingCuts (Algorithm 1). Otherwise, the computation only deals with the vertices of the graph, which are public. The calculation of each of the | U | -1 cut values is ε 4( | U |-1) -DP via the Laplace mechanism (Theorem A.3) as a change in any edge weight by 1 can affect a cut value by at most 1 . By basic composition (Theorem A.2), the total privacy of these calls is

<!-- formula-not-decoded -->

Via the privacy of PrivateMinIsolatingCuts, each call to that subroutine is ε 2( ⌊ lg | U |⌋ +1) -DP. By basic composition, the total privacy of these calls is

<!-- formula-not-decoded -->

Consider the vector x i ∈ R | R i |-1 where each entry in x i corresponds to w ( ˆ S i v ) for some v ∈ R i \{ s } . At any sampling level i , the approximate isolating cuts ˆ S i v are disjoint. Therefore, a change in any

□

edge weight by 1 can change at most two coordinates of x i each at most by 1 (namely, the coordinates corresponding to the sets ˆ S i v which contain the endpoints of the edge). So, x i has ℓ 1 -sensitivity 2 relative to the edge weights of the graph. By the Laplace mechanism, release of all noised entries of x i , given by ˆ w ( ˆ S i v ) , for any fixed i is ε 4( ⌊ lg | U |⌋ +1) -DP. Summing over all sampling levels via basic composition, these calls have privacy

<!-- formula-not-decoded -->

In total, this algorithm is ε -DP as ε 4 + ε 2 + ε 4 = ε .

## D Final Algorithm

Algorithm 3: PrivateGHTree ( G = ( V, E, w ) , ε ) 1 ( T, f ) ← PrivateGHSteinerTree ( G,V,ε, 0 , n ) 2 Add Lap ( 2( n -1) ε ) noise to each edge in T 3 return T

Algorithm 4: PrivateGHSteinerTree ( G = ( V, E, w ) , U, ε, t, n max )

```
1 t max ← Θ(lg 2 n max ) 2 if t ≥ t max then 3 return abort ; /* the privacy budget is exhausted */ 4 end 5 s ← uniformly random vertex in U 6 Call PrivateGHTreeStep ( G,s, U, ε 4 t max , 1 n 3 max ) to obtain D,R ⊆ U and disjoint sets ˆ S v for v ∈ R (recall D = ⋃ v ∈ R ˆ S v ∩ U ) 7 for each v ∈ R do 8 Let G v be the graph with vertices V \ ˆ S v contracted to a single vertex x v 9 Add edges with weight Lap ( 8 t max ε ) from x v to every other vertex in G v , truncating resulting edge weights to be at least 0 10 U v ← ˆ S v ∩ U 11 If | U v | > 1 , recursively set ( T v , f v ) ← PrivateGHSteinerTree ( G v , U v , ε, t +1 , n max ) ; otherwise, T v is a single node and f v is the identity map 12 end 13 Let G large be the graph G with (disjoint) vertex sets ˆ S v contracted to single vertices y v for all v ∈ R 14 U large ← U \ D 15 If | U large | > 1 , recursively set ( T large , f large ) ← PrivateGHSteinerTree ( G large , U large , ε, t +1 , n max ) ; otherwise, T large is a single node and f large is the identity map 16 return Combine (( T large , f large ) , { ( T v , f v ) : v ∈ R } , { ˆ S v : v ∈ R } )
```

Algorithm 5: Combine (( T large , f large ) , { ( T v , f v ) : v ∈ R } , { ˆ S v : v ∈ R ) }

- 1 Construct T by starting with the disjoint union T large ∪ ⋃ v ∈ R T v and for each v ∈ R , adding an edge between f v ( x v ) ∈ U v and f large ( y v ) ∈ U large
- 2 Construct f : V → U = U large ∪ ⋃ v ∈ R U v by f ( v ′ ) = f large ( v ′ ) if v ′ ∈ V \ ⋃ v ∈ R ˆ S v and f ( v ′ ) = f v ( v ′ ) if v ′ ∈ ˆ S v for some v ∈ R

In this section, we present the algorithm PrivateGHTree (Algorithm 3) for constructing an ε -DP approximate Gomory-Hu tree and analyze its approximation error and privacy guarantees. The steps

□

Figure 4: The Combine procedure of Algorithm 5. The computation on each recursive subinstance provides a Gomory-Hu Steiner tree on that subinstance. To stitch these together to a Gomory-Hu Steiner tree of the initial instance, we need to add edges between the solutions to the subinstances. For a node v ∈ R , such that x v is assigned to f v ( x v ) in S v and y v is assigned to f large ( y v ) in S large in the recursively obtained Gomory-Hu Steiner trees, we add an edge between f v ( x v ) and f large ( y v ) .

<!-- image -->

that differ meaningfully from the non-private version developed in [61] are in color. As in [61], we construct the slightly more general structure of a Gomory-Hu Steiner tree as an intermediate step in Algorithm 4.

Definition D.1. Let G = ( V, E, w ) be a weighted graph and U ⊆ V a set of terminals. A Γ -approximate Gomory-Hu Steiner tree is a weighted spanning tree T on U with a function f : V → U where f | U is the identity.

For all distinct s, t ∈ U , if ( u, v ) is the minimum weight edge on the unique path between s and t , in T , and if U ′ is the connected component of T \ { ( u, v ) } containing s , then f -1 ( U ′ ) is a Γ -approximate Mins -t -Cut with λ G ( s, t ) ≤ w T ( u, v ) = w G ( f -1 ( U ′ )) ≤ λ G ( s, t ) + Γ .

To construct the final approximate Gomory-Hu tree, we make a call to PrivateGHSteinerTree (Algorithm 4) with U = V , the entire vertex set. The algorithm PrivateGHSteinerTree is a private version of the GHTree algorithm in [61]. It computes several (approximate) min cuts from a randomly sampled vertex s ∈ U by making a call to PrivateGHTreeStep (Algorithm 2) to obtain D,R ⊆ U and disjoint sets ˆ S v (where D = ⋃ v ∈ R ˆ S v ∩ U ). For each of these cuts ˆ S v it constructs recursive sub-instances ( G v , U v ) where G v is obtained by contracting V \ ˆ S v to a single vertex x v and U v ← ˆ S v ∩ U . Moreover, it creates a sub-instance ( G large , U large ) by contracting each of ˆ S v to a single vertex y v for y ∈ R and setting U large = U ← D .

Notably, on Line 8, where the algorithm recurses on the graph G v with V \ ˆ S v contracted to a single vertex x v , we add noisy edges from x v to all other vertices of the graph. This ensures the privacy of any actual edge from x v in the entire recursive subtree of that instance without incurring too much error. This will imply that for any edge and any instance during the recursion, there is at most one sub-instance where the edge does not receive this privacy guarantee. If t is the depth of the recursion tree, this allows us to apply basic composition over only O ( t ) computations of the algorithm. Essentially, there is only one path down the recursion tree on which we need to track privacy for any given edge in the original graph. We enforce t &lt; t max , and as we will show, the algorithm successfully terminates with depth less than t max with high probability.

To combine the solutions to the recursive sub-problems, we use the Combine algorithm (Algorithm 5) from [61], which in turn is similar to the original Gomory-Hu tree combine step except that it combines more than two recursive sub-instances. See Figure 4 for an illustration of the Combine step.

Finally, Algorithm 3 calls Algorithm 4 with privacy budget ε/ 2 . To be able to output weights of the tree edges, it simply adds Laplace noise Lap ( 2( n -1) ε ) to value of the corresponding cuts in G , hence incurring error O ( n lg n ε ) with high probability. This also has privacy loss ε/ 2 by basic composition over the n -1 tree edges, so the full algorithm is ε -differentially private.

## D.1 Correctness

In this section, we analyze the approximation guarantee of our algorithm. The following main lemma states that Algorithm 4 outputs an O ( n polylog( n )) -approximate Gomory-Hu Steiner tree.

Lemma D.1. Let t max = C lg 2 n for a sufficiently large constant C . PrivateGHSteinerTree ( G,V,ε, 0 , n ) outputs an O ( n lg 8 n ε ) -approximate Gomory-Hu Steiner tree T of G with high probability.

We start by proving a lemma for analyzing a single recursive step of the algorithm. It is similar to [61, Lemma 4.5.4] but its proof requires a more careful application of the submodularity lemma. Lemma D.2. With high probability, for any distinct vertices p, q ∈ U large, we have that λ G ( p, q ) ≤ λ G large ( p, q ) ≤ λ G ( p, q ) + O ( n lg 5 n max ε ) . Also with high probability, for any v ∈ R and distinct vertices p, q ∈ U v , we have that λ G ( p, q ) ≤ λ G v ( p, q ) ≤ λ G ( p, q ) + O ( n lg 6 n max ) .

ε

Proof. Let us start by upper bounding how close the cuts ˆ S v are to being Mins -v -Cuts and how close the approximate min cut values ˆ w ( ˆ S v ) are to the true sizes w ( ˆ S v ) . Algorithm 4 calls Algorithm 2 with privacy parameter ε 1 = ε 4 t max = Θ ( ε lg 2 n max ) and error parameter β = 1 n 3 max where n max is the number of vertices in the original graph. By Lemma C.1 with privacy ε 1 and error parameter β , it follows that (a) the sets { ˆ S v } are approximate minimum isolating cuts with total error O ( ( n +log(1 /β )) lg 2 n ε 1 ) = O ( n lg 5 n max ε ) and (b) each set ˆ S v is an approximate Mins -v -Cut with error O ( n lg 6 n max ε ) . On any given call to Algorithm 2, these error bounds hold with probability 1 -n -3 max . As each call to Algorithm 2 ultimately contributes an edge to the final Gomory-Hu tree via Algorithm 5, there can be at most n max -1 calls throughout the entire recursion tree, resulting in failure probability of n -2 max overall after a union bound.

The fact that λ G ( p, q ) ≤ λ G large ( p, q ) follows since G large is a contraction of G . To prove the second inequality, let S be one side of the true Minp -q -Cut in G . Let R 1 = R ∩ S and R 2 = R \ S . We show that the cut S ∗ = ( S ∪ ⋃ v ∈ R 1 ˆ S v ) \ ( ⋃ v ∈ R 2 ˆ S v ) is an O ( n lg 6 n max /ε ) -approximate min cut. Since S ∗ is also a cut in G large, the desired bound on λ G large ( p, q ) follows.

Let v 1 , . . . , v | R 1 | be the vertices of R 1 in an arbitrary order. By | R 1 | applications of the submodularity lemma (Lemma A.1),

<!-- formula-not-decoded -->

Note that S ∪ ⋃ | R 1 | i =1 ˆ S v i is still a ( p, q ) -cut as p, q ∈ U large and the sets ˆ S v i are each disjoint from U large . Moreover, for each i , S contains v i and so ( S ∪ ⋃ j&lt;i ˆ S v j ) ∩ ˆ S v i isolates v i from all vertices in V \ ˆ S v i . Using the fact that { ˆ S v } are approximate minimum isolating cuts, the sum in the RHS above can be upper bounded by O ( n lg 5 n max ε ) . Letting S ′ = S ∪ ⋃ v ∈ R 1 ˆ S v , and S ′′ = ( V \ S ′ ) ∪ ⋃ v ∈ R 2 ˆ S v a similar argument but applied to V \ S ′ , shows that

<!-- formula-not-decoded -->

But S ∗ = V \ S ′′ , so we get that

<!-- formula-not-decoded -->

as desired.

For the case of p, q ∈ U v for some v , again the bound λ G ( p, q ) ≤ λ G u ( p, q ) is clear. Thus, it suffices to consider the upper bound on λ G v ( p, q ) . Let S be the side of the Minp -q -Cut in G which does not contain v . Assume first that s / ∈ S . By the submodularity lemma (Lemma A.1)

<!-- formula-not-decoded -->

By the approximation guarantees of Algorithm 2, w ( ˆ S v ) ≤ λ G ( s, v ) + O ( n lg 6 n max ε ) . Moreover, note that S ∪ ˆ S v is an ( s, v ) -cut of G , so w ( S ∪ ˆ S v ) ≥ λ G ( s, v ) . Thus,

<!-- formula-not-decoded -->

Since S ∩ ˆ S v is a ( p, q ) -cut of G v , we must have that w ( S ∩ ˆ S v ) ≥ λ G v ( p, q ) , so in conclusion λ G v ( p, q ) ≤ λ G ( p, q )+ O ( n lg 6 n max ε ) ignoring the added noisy edges to G v in Line 8 of Algorithm 4.

Adding the noisy edges can only increase the cost by O ( n lg 3 n max ε ) with high probability via Laplace tail bounds (note that there can be at most n max -1 noisy edges added as each time a noisy edge is added an edge is added to the final approximate Gomory Hu Steiner tree). This finishes the proof in the case s / ∈ S . A similar argument handles the case where s ∈ S but here we relate the value w ( V \ S ) to w ( V \ S ) ∩ ˆ S v ) . □

To bound the error of the algorithm, we need a further lemma bounding the depth of its recursion. The argument is similar to that of [61].

Lemma D.3. If t max = C lg 2 n for a sufficiently large constant C , then, with high probability, no recursive call to Algorithm 4 from PrivateGHTree ( G,ε ) aborts.

Proof. Each of the recursive instances ( G v , U v ) has | U v | ≤ 9 10 | U | by the way D is picked in Line 7 of Algorithm 2. Moreover, by [2, Corollary III.7], if s is picked uniformly at random from U , then E [ D ∗ ] = Ω( | U | -1) . By Lemma C.1, the expected size of D returned by a call to Algorithm 2 when picking s at random from | U | is then at least Ω ( | U |-1 lg n ) . By Line 14 of Algorithm 4, it follows that, E [ | U large | ] ≤ | U | (1 -Ω(1 / lg n )) when | U | &gt; 1 . Thus any instance at recursive depth t satisfies E [ | U large | ] ≤ n (1 -Ω(1 / lg n )) t ≤ n exp( -Ω( t/ lg n )) . If t = Ω(lg 2 n ) , the expectation is smaller than 1 / poly( n ) , so by Markov's inequality, any instance at recursive depth t satisfies | U | = 1 with high probability. Now note that at every recursive depth, we can only have at most n instances, since the sets U passed to the recursive calls of Algorithm 4 at depth t are disjoint. Thus, there are polynomially many recursive instances up to recursive depth t max , so we can union bound over all sub-instances. In conclusion, all sub-instances have | U | = 1 within O (lg 2 n ) recursive depth with high probability. □

We can now prove Lemma D.1. The argument is again similar to [61] except we have to incorporate the approximation errors.

Proof of Lemma D.1. By Lemma D.3, the algorithm does not abort with high probability.

Throughout the proof, n denotes the number of vertices in the input graph. Let ∆ = O ( n lg 6 n ε ) be such that with high probability λ G v ( p, q ) ≤ λ G ( p, q )+∆ for p, q ∈ U v and similarly λ G large ( p, q ) ≤ λ G ( p, q ) + ∆ for p, q ∈ U large . The existence of ∆ is guaranteed by Lemma D.2. We prove by induction on i = 0 , . . . , t max , that the output to the instances at level t max -i of the recursion are 2 i ∆ -approximate Gomory-Hu Steiner trees. This holds trivially for i = 0 as the instances on that level have | U | = 1 and the tree is the trivial one-vertex tree approximating no cuts at all. Let i ≥ 1 and assume inductively that the result holds for smaller i . In particular, if ( T, f ) is the output of an instance at recursion level i , then the trees ( T v , f v ) and ( T large , f large ) are 2( i -1)∆ -approximate Gomory-Hu Steiner trees of their respective G v or G large graphs.

Consider any internal edge ( a, b ) ∈ T large (without loss of generality, what follows also holds for ( a, b ) ∈ T v ) . Let U ′ and U ′ large be the connected component containing a after removing ( a, b ) from T and T large, respectively. By design of Algorithm 5, f -1 large ( U ′ large ) and f -1 ( U ′ ) are the same except each contracted vertex y v ∈ f -1 large ( U ′ large ) appears as ˆ S v ⊆ f -1 ( U ′ ) . It follows that w G large ( f -1 large ( U ′ large )) = w G ( f -1 ( U ′ )) . By the inductive hypothesis, ( T large , f large ) is an approximate Gomory-Hu Steiner tree, so w G large ( f -1 large ( U ′ large )) = w T large ( a, b ) . Therefore setting w T ( a, b ) = w T large ( a, b ) = w G ( f -1 ( U ′ )) has the correct cost for T according to the definition of an approximate Gomory-Hu Steiner tree.

Furthermore, on the new edges ( f v ( x v ) , f large ( y v )) , the weight w ( ˆ S v ) is the correct weight for that edge in T as ˆ S v is the f v ( x v ) side of the connected component after removing that edge. Finally, by

adding these new edges, the resulting tree is a spanning tree. So, the structure of the tree is correct, and it remains to argue that the cuts induced by the tree (via minimum edges on shortest paths) are approximate Mins -t -Cuts.

Consider any p, q ∈ U . Let ( a, b ) be the minimum edge on the shortest path in T . Note that it is always the case that w T ( a, b ) ≥ λ ( a, b ) as w T ( a, b ) corresponds to the value of a cut in G separating a and b . We will proceed by cases.

If ( a, b ) ∈ T large , then by induction, w T ( a, b ) = w T large ( a, b ) ≤ λ G large ( a, b ) + (2 i -2)∆ . By Lemma D.2, it follows that w T ( a, b ) ≤ λ G ( a, b ) + (2 i -1)∆ . The exact same argument applies if ( a, b ) ∈ T v for some v ∈ R .

The case that remains is if ( a, b ) is a new edge with a = f -1 v ( x v ) ∈ U v and b = f -1 large ( y v ) ∈ U large for some v ∈ R . Then, w T ( a, b ) = w G ( ˆ S v ) . By Lemma C.1, ˆ S v is an approximate Minv -s -Cut and w T ( a, b ) ≤ λ G ( v, s ) + ∆ . To connect this value to λ G ( a, b ) , note that by considering the choices of where v and s lie on the Mina -b -Cut, we can show

<!-- formula-not-decoded -->

Let S ′ a be the a side of the a -v cut induced by the approximate Gomory-Hu Steiner tree ( T v , f v ) . As a = f -1 v ( x v ) and s ∈ x v , S ′ a is also the s side of a v -s cut. Therefore, w G ( S ′ a ) ≥ λ G ( v, s ) . On the other hand, by our inductive hypothesis and Lemma D.2, this is an approximate Mina -v -Cut: w G ( S ′ a ) ≤ λ G v ( a, v )+(2 i -2)∆ ≤ λ G ( a, v )+(2 i -1)∆ . Hence, λ G ( v, s ) ≤ λ G ( a, v )+(2 i -1)∆ . The analogous argument holds to show λ G ( v, s ) ≤ λ G ( s, b ) + (2 i -1)∆ . Therefore,

<!-- formula-not-decoded -->

which further implies λ G ( v, s ) ≤ λ G ( a, b ) + (2 i -1)∆ by combining with Equation (7). Therefore,

<!-- formula-not-decoded -->

In all cases, w T ( a, b ) ≤ λ G ( a, b ) + 2 i ∆ . As the cut corresponding to the edge ( a, b ) is on the path from p to q , it is also a ( p, q ) -cut, so λ G ( p, q ) ≤ w T ( a, b ) . Furthermore, it must the case that there is an edge ( a ′ , b ′ ) along the path between p to q such that a ′ and b ′ are in different sides of the true Minp -q -Cut. Therefore, λ G ( p, q ) ≥ λ G ( a ′ , b ′ ) . As we chose ( a, b ) to be the minimum weight edge,

<!-- formula-not-decoded -->

This completes the induction. It follows that the call to PrivateGHSteinerTree ( G,V,ε, 0) outputs a 2 t max ∆ -approximate Gomory-Hu Steiner tree T . Substituting in the values t max = O (lg 2 n ) and ∆ = O ( n lg 6 n ε ) gives the approximation guarantee. □

We now state our main result on the approximation guarantee of Algorithm 3.

Theorem D.1. Let T = ( V T , E T , w T ) be the weighted tree output by PrivateGHTree ( G = ( V, E, w ) , ε ) on a weighted graph G . For each edge e ∈ E T , define S e to be the set of vertices of one of the connected components of T \ { e } . Let u, v ∈ V be distinct vertices and let e min be an edge on the unique u -v path in T such that w T ( e min ) is minimal. With high probability, S e min is an O ( n lg 8 n ε ) -approximate Minu -v -Cut and moreover, | λ G ( u, v ) -w T ( e min ) | = O ( n lg 8 n ε ) .

Proof. Note that for each edge e , the final tree weight w T ( e ) is obtained by adding noise Lap ( 2( n -1) ε ) to the cut value w ( S e ) . Thus, | w ( S e ) -w T ( e ) | = O ( n lg n ε ) with high probability for all e ∈ T . Now let e 0 be an edge on the unique u -v path in T such that w ( S e 0 ) is minimal. Then, by Lemma D.1, w ( S e 0 ) ≤ λ G ( u, v ) + O ( n lg 8 n ε ) with high probability. As e min was chosen as an edge on the u -v path in T of minimal weight, w T ( e min ) ≤ w T ( e 0 ) , and so

<!-- formula-not-decoded -->

On the the other hand, S e min defines a ( u, v ) -cut, so λ G ( u, v ) ≤ w ( S e min ) . This proves the first statement. Moreover, the string of inequalities above combined with λ G ( u, v ) ≤ w ( S e min ) in particular entails that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## D.2 Privacy via Bounded-Overlap Branching Composition

Consider a family of computation trees parameterized by two differentially private mechanisms, denoted M recurse and M sanitize. Through M recurse, each node in the tree privately produces both (1) the output of its computation and (2) the topology and description of its children's input. Each child's input consists of a combination of 'sensitive' data, transmitted in plaintext, and 'sanitized' data, provided as the output of the private mechanism M sanitize. Let σ ( u ) denote the set of children of a node u in a tree.

Specifically, upon receiving a sensitive dataset X u = X u 1 , . . . , X u s and a sanitized dataset Y u = Y u 1 , . . . , Y u t from its parent, a node u computes the following objects via M recurse ( X u , Y u ) (we only require that M recurse is DP with respect to its first input):

- A number of children d u
- For each child v ∈ σ ( u ) , a set of 'sensitive' indices I v = { i v 1 , . . . , i v n }
- For each child v ∈ σ ( u ) , a set of 'sanitized' indices J v = { j v 1 , . . . , j v m }
- An auxiliary output a u

Let v be one of the d u children of u . The input for v consists of a sensitive dataset X v = X u i v 1 , . . . X u i v n and a sanitized dataset, formed by concatenating any function of Y u with M sanitize ( X u j v 1 , . . . , X u j v m ) . This setup is illustrated in Figure 2.

̸

Additionally, we consider a privacy model where the indices of the data are public, and two neighboring data sets differ in at most one index. The theorem below holds regardless of how we define the neighborhood relation on a single index. For example, we could say that X and X ′ are neighboring if there exists at most one index i such that X i = X ′ i . However, for our main result on DP Gomory-Hu trees, X = ( X e ) e ∈ E = ( w ( e )) e ∈ E are the weights of edges of the input graph, and two data sets X,X ′ are neighboring if there exists an e ∈ E such that for all e ′ ∈ E \ { e } , X e ′ = X ′ e ′ and moreover, | X e -X ′ e | ≤ 1 .

Theorem D.2 (Bounded-Overlap Branching Composition) . Let M recurse and M sanitize be mechanisms as described, satisfying ( ε 1 , δ 1 ) -DP and ( ε 2 , δ 2 ) -DP, respectively. Let the subsets of indices { I v } and { J v } produced by M recurse satisfy the following conditions deterministically for all inputs:

- The index sets { I v } are disjoint.
- For all indices j , |{ v : j ∈ J v }| ≤ ℓ for a fixed constant ℓ .

Consider the following mechanism M branch that, as input, receives a sensitive dataset X and a maximum depth parameter h . M branch creates a tree T = ( V, E ) recursively using M recurse and M sanitize with X as the sensitive input to the root node; the root node does not receive any sanitized input. If the depth of T is greater than or equal to h , the mechanism outputs ⊥ . Otherwise, the mechanism releases T along with all outputs { ( Y u , d u , { I v } v ∈ σ ( u ) , { J v } v ∈ σ ( u ) , a u ) : u ∈ V } . This mechanism is ( hε 1 +( h -1) ℓε 2 , hδ 1 +( h -1) ℓδ 2 ) -DP.

Proof. Let N = | X | . For any t ∈ N , let M t branch be the mechanism which, given a sensitive dataset X , creates a tree by recursively using M recurse and M sanitize with X as the input to the root node, stopping after recursion depth t ; we adopt the convention that a tree with a single node has depth 0 . The output of M t branch is the tree T t = ( V t , E t ) with maximum depth t , and the sanitized inputs and node outputs { ( Y u , d u , { I v } v ∈ σ ( u ) , { J v } v ∈ σ ( u ) , a u ) : u ∈ V t } .

Let ε ( t ) , δ ( t ) be the privacy parameters for M t branch . The mechanism described in the theorem statement is ( ε ( h -1) , δ ( h -1)) -DP as checking whether the tree (with unbounded recursive depth) has

depth at most h can be verified by running the recursion for up to h steps and checking whether d u = 0 for every node u at depth h . Outputting either ⊥ or ( T, { ( Y u , d u , { I v } v ∈ σ ( u ) , { J v } v ∈ σ ( u ) , a u ) : u ∈ V t } ) is a post-processing of the output of M h -1 branch and therefore cannot increase the privacy loss. We prove by induction over t that the following conditions hold:

- (a) ε ( t ) ≤ ( t +1) ε 1 + tℓε 2
- (b) δ ( t ) ≤ ( t +1) δ 1 + tℓδ 2
- (c) Let S t ⊆ V be the subset of vertices in the tree produced by M t branch ( X ) at depth t . For any i ∈ [ N ] , there is at most one vertex u ∈ S t for which X i ∈ X u .

The theorem statement immediately follows from conditions (a) and (b) by plugging in t = h -1 .

For the base case, consider t = 0 . M t branch ( X ) releases a tree T 0 = ( V 0 , E 0 ) as well as { ( Y u , d u , { I v } v ∈ σ ( u ) , { J v } v ∈ σ ( u ) , a u ) : u ∈ V 0 } . The tree T 0 simply contains a single node independently of X . Let u be the single node in V 0 . The root node receives no sanitized dataset, so Y u = ∅ independently of X . The output ( d u , { I v } v ∈ σ ( u ) , { J v } v ∈ σ ( u ) , a u ) is produced by M recurse which is ( ε 1 , δ 1 ) -DP. Therefore, the entire mechanism M 1 branch is ( ε 1 , δ 1 ) -DP. Furthermore, condition (c) of the inductive hypothesis is trivially satisfied as there is a unique vertex.

Consider any t &gt; 0 . By the inductive hypothesis, releasing the tree T t -1 = ( V t -1 , E t -1 ) along with { ( Y u , d u , { I v } v ∈ σ ( u ) , { J v } v ∈ σ ( u ) , a u ) : u ∈ V t -1 } is ( tε 1 +( t -1) ℓε 2 , tδ 1 +( t -1) ℓδ 2 ) -DP. We will assume that these objects have been released and analyze the privacy of releasing the additional objects output by M t branch ( X ) . Note that the topology of the tree T t = ( V t , E t ) can already be calculated as post-processing from these objects as the structure of the t -th level of the tree is contained in the degrees of the leaves of T t -1 . Let S t = V t \ V t -1 . The additional objects output by M t branch ( X ) that cannot be obtained via post-processing of M t -1 branch ( X ) are { ( Y v , d v , { I w } w ∈ σ ( v ) , { J w } w ∈ σ ( v ) , a v ) : v ∈ S t } .

We will continue by bounding the privacy of releasing these outputs for any node at depth t with respect to the sensitive inputs to the node and its parent. Consider any v ∈ S t and let p ( v ) ∈ S t -1 be the parent of v in T t . The only part of Y v which cannot be obtained by post-processing Y p ( v ) is the output of M sanitize ( X p ( v ) j v 1 , . . . , X p ( v ) j v m ) . So releasing Y v is ( ε 2 , δ 2 ) -DP with respect to the subset of its parent's sensitive input ( X p ( v ) j v 1 , . . . , X p ( v ) j v m ) and (0 , 0) -DP with respect to X \ ( X p ( v ) j v 1 , . . . , X p ( v ) j v m ) . Given the release of Y v as well as I v and J v , releasing ( d v , { I w } w ∈ σ ( v ) , { J w } w ∈ σ ( v ) , a v ) , the output of M recurse ( X v = ( X p ( v ) i v 1 , . . . , X p ( v ) i v n ) , Y v ) , is ( ε 1 , δ 1 ) -DP with respect to X v and (0 , 0) -DP with respect to X \ X v .

̸

Consider any particular index i ∗ ∈ [ N ] corresponding to the sole index on which two neighboring datasets X and X ′ differ. By the inductive hypothesis, there is at most one node u ∗ ∈ S t -1 such that X i ∗ ∈ X u . Consider any node v ∈ S t which is not a child of u ∗ , p ( v ) = u ∗ . Note that X i ∗ / ∈ X p ( v ) implies that X i ∗ / ∈ X v as X v ⊆ X p ( v ) . By the argument in the paragraph above, conditioning on the release of M t -1 branch ( X ) , the release of ( Y v , d v , { I w } w ∈ σ ( v ) , { J w } w ∈ σ ( v ) , a v ) does not depend on X i ∗ and does not affect privacy.

Consider nodes v ∈ σ ( u ∗ ) , the children of u ∗ . If i ∗ / ∈ I v and i ∗ / ∈ J v , as above, conditioning on the release of M t -1 branch ( X ) , the release of ( Y v , d v , { I w } w ∈ σ ( v ) , { J w } w ∈ σ ( v ) , a v ) does not depend on X i ∗ and does not affect privacy. If i ∗ ∈ J v , the release of ( Y v , d v , { I w } w ∈ σ ( v ) , { J w } w ∈ σ ( v ) , a v ) is ( ε 2 , δ 2 ) -DP. If i ∗ ∈ I v , the release of ( Y v , d v , { I w } w ∈ σ ( v ) , { J w } w ∈ σ ( v ) , a v ) is ( ε 1 , δ 1 ) -DP.

By the condition on M recurse in the theorem statement, there is at most one child v ∈ σ ( u ∗ ) such that i ∗ ∈ I v and at most ℓ v ∈ σ ( u ∗ ) such that i ∗ ∈ J v . This implies that there is at most one node v ∈ S t where i ∗ ∈ I v , satisfying condition (c) of the inductive hypothesis. As this holds for any choice of i ∗ , conditioned on the release of M t -1 branch ( X ) , releasing { ( Y v , d v , { I w } w ∈ σ ( v ) , { J w } w ∈ σ ( v ) , a v ) : v ∈ S t } is ( ε 1 + ℓε 2 , δ 1 + ℓδ 2 ) -DP. By basic composition, summing over the privacy parameters of releasing M t -1 branch , ε ( t ) ≤ ( ε 1 + ℓε 2 ) + ( tε 1 +( t -1) ℓε 2 ) = ( t + 1) ε 1 + tℓε 2 and, likewise, δ ( t ) ≤ ( t +1) δ 1 + tℓδ 2 , satisfying conditions (a) and (b) of the inductive hypothesis. This completes the proof. □

We will use bounded-overlap branching composition to prove the privacy of our GH-tree algorithm. Theorem D.3. PrivateGHTree ( G,ε ) , i.e., Algorithm 3, is ε -DP.

Proof. We will first argue that releasing the unweighted tree returned by the call to PrivateGHSteinerTree ( G,V,ε, 0 , n ) is ( ε/ 2) -DP. To invoke Theorem D.2, we will describe the mechanisms M recurse and M sanitize for which the corresponding M branch mechanism simulates the computation done in this call to PrivateGHSteinerTree.

M recurse has a sensitive and sanitized input. Its sanitized input is a list of vertices and edges, edge weights for a subset of its edges, as well as U, ε, t, n max . Its sensitive input is a set of edge weights for the rest of its edges. Let G be the weighted graph which is formed by combining the two inputs. M recurse contracts some vertices (if it is itself a G large subinstance), picks a uniformly random vertex s , calculates t max , and runs PrivateGHTreeStep ( G,s, U, ε 4 t max , 1 n 3 max ) to obtain D,R ⊆ U , and { ˆ S v } v ∈ R . The output of M recurse is the following:

- Recursive degree | R | +1 .
- For each v ∈ R , a recursive child is created which gets sensitive indices corresponding to all edges between pairs of vertices in ˆ S v . In addition, this child gets sanitized indices corresponding to all edges between ˆ S v and V \ ˆ S v .
- An additional recursive child (corresponding to the G large subinstance) is created which gets sensitive indices corresponding to all edges whose endpoints do not both belong to the same set ˆ S v . This child has no sanitized indices.
- Auxiliary output R, { ˆ S v } v ∈ R .

By the privacy of PrivateGHTreeStep in Lemma C.3, M recurse is ( ε 4 t max ) -DP.

M sanitize takes as input a set of edge weights for edges between a set ˆ S v and its complement V \ ˆ S v . It contracts V \ ˆ S v to a vertex x v and outputs the resulting (summed) edge weight from edges between ˆ S v and x v plus Lap ( 8 t max ε ) noise. By the privacy of the Laplace mechanism Theorem A.3 and as the contraction/sum operation has sensitivity 1 , M sanitize is ( ε 8 t max ) -DP.

The unweighted tree output by PrivateGHSteinerTree ( G,V,ε/ 2 , 0 , n ) can be calculated via postprocessing of the recursive mechanism M branch parameterized by M recurse and M sanitize with maximum depth h = t max . While in Algorithm 4, we invoke the Combine step to construct the GH-tree, this can easily be simulated given the auxiliary outputs { ˆ S v } v ∈ R at each recursive node as these sets determine the GH-tree topology.

The final step to bound the overall privacy of outputting the unweighted tree is to ensure that M recurse satisfies the bounded-overlap condition. Consider the edge weights which are the sensitive input to a call to M recurse. Note that these edge weights are partitioned across the recursive children: either the weight corresponds to an edge with both endpoints in a single ˆ S v , in which case it is sent to the corresponding G v subinstance, otherwise, it is sent to the special G large subinstance. Furthermore, any edge weight only belongs to the sanitized indices of at most two recursive children. Each G v subinstance receives all edge weights for edges between ˆ S v and V \ ˆ S v as sanitized indices. Any particular edge has two endpoints and so can only belong to two such sets.

Plugging in ℓ = 2 to Theorem D.2, we get the following bound on the privacy of releasing the unweighted approximate GH-tree:

<!-- formula-not-decoded -->

In the final algorithm PrivateGHTree ( G,ε ) , the tree weights are set to be the corresponding cut value in G plus Lap ( 2( n -1) ε ) noise. Note that the choice of which cut values to calculate is a

post-processing of the privatized output of the unweighted tree. As any specific cut value can change by at most 1 in neighboring graphs, outputting a single tree edge weight is ( ε 2( n -1) ) -DP. Basic composition over all n -1 tree edges means that releasing the edge weights given the tree topology is ( ε/ 2) -DP. In total, releasing the weighted tree is ε -DP, as required. □

## D.3 Runtime

While runtime is not our main focus, as a final note, our algorithm can be implemented to run in near-quadratic time in the number of vertices of the graph. The runtime is inherited directly from prior work of [4], which utilizes the same recursive algorithm introduced in [61]. The overall structure of their main algorithm and subroutines remains in our work with changes of the form (a) altering runtime-independent conditions in if statements or (b) adding noise to cut values or edges in the graph. While left unspecified here, computation of single source Mins -v -Cuts in Algorithm 2 should be done via the runtime-optimized algorithm of prior work [85] to achieve the best bound. Then, via Theorem 1.3 of [4], Algorithm 3 runs in time ˜ O ( n 2 ) (note that if instead of [85], we use the almost-linear time algorithm for single source Mins -v -Cuts [23], the running time would be n 2+ o (1) for dense graphs).

## E Additional Related Work

Multiway cut is another cut problem that has been studied in the privacy setting. Given k terminals, the multiway cut problem seeks a partitioning of the graph's nodes into k parts such that (1) each part contains exactly one terminal, and (2) the sum of the weights of edges between parts, known as the cut value, is minimized [28]. The multiway cut problem is NP-hard for k ≥ 3 [28], implying that all non-private polynomial-time algorithms have approximation factors greater than 1 . Dalirrooyfard, Mitrovi´ c, and Nevmyvaka [29] present an ε -DP algorithm for the multiway cut problem with a multiplicative approximation factor of 2 and an additive error of O ( n log k/ε ) . Chandra et al. [18] provide an ε -DP algorithm achieving an additive error of ˜ O ( nk/ε ) and a multiplicative approximation ratio that matches the best-known non-private algorithm. They show that an additive error of Ω( n log k/ε ) is necessary for any ε -private algorithm for multiway cut. An open question is whether there exists an algorithm with an additive error of O ( n log k/ε ) and a multiplicative approximation ratio that matches the best-known non-private algorithm.

Computing cuts on (suitably defined) graphs is also a key primitive in a wide-range of applications beyond the aforementioned graph algorithms. For example in many clustering problems, cuts are explicitly computed, as in spectral [20, 76] or correlation clustering [69, 65], or the objective involves identifying a small cut implicity, such as in community detection in the stochastic block model and beyond [40, 24, 43, 42]. Cuts are also at the heart of many learning-theory tasks on graph data, such as learning graph partitions [68, 79], (semi)-supervised learning using cuts [12, 7, 87, 86, 52], active learning [11, 45], and transductive learning [55, 48], to name a few examples. We refer to the papers and references therein for details.

## F Minimum k -cut

Lastly, we note an application to the minimum k -cut problem. Here, the goal is to partition the vertex set into k pieces and the cost of a partitioning is the total weight of all edges between different pieces in the partition. We wish to find the smallest cost solution. It is known that simply removing the smallest k -1 edges of an exact GH tree gives us a solution to the minimum k -cut problem with a multiplicative approximation of 2 [80]. Since we compute an approximate GH tree with additive error ˜ O ( n/ε ) , we can obtain a solution to minimum k -cut with multiplicative error 2 and additive error ˜ O ( nk/ε ) . Our corollary is the following.

Corollary F.1. Given a weighted graph G with positive edge weights and a privacy parameter ε &gt; 0 , there exists an ε -DP algorithm that outputs a solution to the minimum k -cut problem on G in ˜ O ( n 2 ) time with multiplicative error 2 and additive error ˜ O ( nk/ε ) with high probability.

The only prior non-trivial DP algorithm for the minimum k -cut problem is given in [18]. Their pure DP algorithm obtains the optimal additive error Θ( k log( n ) /ε ) but requires the input graph to be

unweighted while also requiring exponential time. They also give a polynomial time algorithm which also has multiplicative error 2 with additive error ˜ O ( k 1 . 5 /ε ) which holds for weighted graphs, but only works in the approximate DP setting. Note that there are also trivial algorithms such as finding the minimum k -cut from a solution to the All Cuts problem, e.g., those from Table 1, but this strategy only gives additive error O ( kn 1 . 5 /ε ) for dense graphs.

Thus, to the best of our knowledge, no prior polynomial-time pure DP algorithm can compute the minimum k -cut on weighted graphs with near-linear in n error. Determining the limits of efficient and pure DP algorithms for the minimum k -cut problem is an interesting open question.

Proof of Corollary F.1. We follow the proof of [80] (via the lecture notes in [19]), replacing the exact GH-tree with our approximate version. The algorithm is simple: we cut the edges corresponding to the union of cuts given by the smallest k -1 edges of our approximate GH-tree T of Theorem 1.1. If this produces more than k pieces, arbitrarily add back cut edges until we reach a k -cut.

For the analysis, consider the optimal k -cut with partitions V 1 , . . . , V k and let w ( V 1 ) ≤ . . . ≤ w ( V k ) denote the weight of the edges leaving each partition without loss of generality. Since every edge in the optimum is adjacent to exactly two pieces of the partition, it follows that ∑ i w ( V i ) is twice the cost of the optimal k -cut. We will now demonstrate k -1 different edges in T which have cost at most ∑ i w ( V i ) , up to additive error O ( k ∆) = ˜ O ( nk/ε ) , where ∆ = ˜ O ( n/ε ) is the additive error from Theorem 1.1.

As in the proof in [80], contract the vertices in V i in T for all i . This may create parallel edges, but the resulting graph is connected since T was connected to begin with. Make this graph into a spanning tree by removing parallel edges arbitrarily, root this graph at V k , and orient all edges towards V k .

̸

Consider an arbitrary V i where i = k . The 'supernode' for V i has a unique edge leaving it, which corresponds to a cut between some vertex v ∈ V i and some vertex w ̸∈ V i . Since T is an approximateGHtree, the weight of this edge must be upper bounded by w ( V i ) (which is also a valid cut separating v and w ), up to additive error ∆ . The proof now follows by summing across V i . □

## G Discussion on Open Problems

An interesting open question is whether a better additive error can be achieved or a lower bound established in the setting where we only care about values of cuts and not the cuts (vertex bipartitions) themselves. To our knowledge, the best-known error bound for this All-Pairs Mins -t -Cut Values problem is O ( n/ε ) for approximate DP, obtained by a trivial algorithm that adds Lap ( n/ε ) noise to each of the ( n 2 ) true values; this method satisfies privacy through advanced composition and the Laplace mechanism [34]. Note that this algorithm does not leverage the fact that there are, in fact, at most n -1 distinct cut values or any of the structure of the graph. Before our work, the best algorithm for this problem in the pure DP setting solved the All Cuts problem, incurring an O ( n 3 / 2 /ε ) error [47]. Our work yields an improvement to ˜ O ( n/ε ) error with pure DP. Both of these solutions in the pure DP setting output the cuts and the values, so it seems probable that better error can be achieved. No non-trivial lower bound is known for this problem. The Ω( n ) lower bound of [29] applies to releasing an approximate Mins -t -Cut, whereas releasing a single Mins -t -Cut value can be achieved with an error of O (1 /ε ) using the Laplace mechanism. This question parallels the All-Pairs Shortest-Paths Distances problem studied in [81, 39, 22, 13], where sublinear additive error is achievable for releasing the values of all ( n 2 ) shortest-paths. In contrast, the linear error is required to release any shortest path itself. One might wonder about the sensitivity of a single edge on Mins -t -Cut values; specifically, whether two neighboring graphs differ in only a small number of Mins -t -Cut values. However, as illustrated in Figure 3, the Mins -t -Cut value sets of two neighboring graphs can differ by as many as n -1 entries.

An additional open question is whether there exists a polynomial-time ε -DP algorithm for the global Min Cut problem that achieves error below ˜ O ( n/ε ) . As noted in Corollary 1.2, the polynomial time algorithm from [46] is only approximate DP, though it can be made pure DP if allowed exponential runtime. The same question applies to minimum k -cut (see Corollary F.1): what are the limits of efficient, i.e., polynomial-time, algorithms that are also pure DP for this problem?

Finally, in the problem of outputting synthetic graphs that preserve all cuts, the work of [37] establishes a lower bound of Ω( √ mn/ε ) on the additive error. However, this result applies to algorithms that do not permit any multiplicative approximation. In the same paper, the authors discuss that an additive error of ˜ O ( n ) with a multiplicative error of 1 + η , for any constant η &gt; 0 , can be achieved in exponential time using the existence of linear-size cut-sparsifiers and the Exponential mechanism. Recently, Aamand et al. [1] gave a polynomial time algorithm for this problem with roughly n 1 . 25 additive error with constant multiplicative error. It remains an intriguing open question whether a synthetic graph that preserves all cuts within an additive error of roughly n and a constant multiplicative error can be generated in polynomial time.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We give full proofs of all of our theoretical results.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We fully specify the theoretical (differential privacy) setting for which our main results hold. Since this is a theoretical work, specific concerns about any practical applications of our work is beyond the scope of our paper.

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

Justification: Full proofs are given in the appendix.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA]

Justification: Our paper has no experiments.

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

Answer: [NA]

Justification: The paper has no experiments.

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

Justification: The paper has no experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The paper has no experiments.

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

Justification: The paper has no experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our paper complies with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: This paper presents work whose goal is to advance algorithmic and theoretical tools in the field of Machine Learning. There are many potential societal consequences of our work. The most direct consequence which we highlight in the paper is that we develop a better algorithm for a fundamental graph problem with differential privacy. This may allow more private/more accurate analysis of graph data.

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

Justification: Our work poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper does not use any existing assets.

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

Justification: All the assets of the papers (the proofs) are given in the appendix.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing or research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing or research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: Our paper is not related to LLMs in any meaningful way, and we did not use LLMs in creating the paper.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.