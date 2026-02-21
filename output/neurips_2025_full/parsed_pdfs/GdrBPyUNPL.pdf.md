## Non-monotone Submodular Optimization: p -Matchoid Constraints and Fully Dynamic Setting

Kiarash Banihashem ∗ University of Maryland College Park, MD, USA kiarash@umd.edu

## Samira Goudarzi ∗

University of Maryland College Park, MD, USA samirag@umd.edu

## Peyman Jabbarzade ∗

University of Maryland College Park, MD, USA peymanj@umd.edu

## MohammadTaghi Hajiaghayi ∗

University of Maryland College Park, MD, USA hajiagha@umd.edu

## Morteza Monemizadeh ∗

TU Eindhoven Eindhoven, The Netherlands M.Monemizadeh@tue.nl

## Abstract

Submodular maximization subject to a p -matchoid constraint has various applications in machine learning, particularly in tasks such as feature selection, video and text summarization, movie recommendation, graph-based learning, and constraintbased optimization. We study this problem in the dynamic setting, where a sequence of insertions and deletions of elements to a p -matchoid M ( V , I ) occurs over time and the goal is to e ffi ciently maintain an approximate solution. We propose a dynamic algorithm for non-monotone submodular maximization under a p -matchoid constraint. For a p -matchoid M ( V , I ) of rank k , defined by a collection of m matroids, our algorithm guarantees a (2 p + 2 p p ( p + 1) + 1 + ϵ )-approximate solution at any time t in the update sequence, with an expected amortized query complexity of O ( ϵ -3 pk 4 log 2 ( k )) per update.

## 1 Introduction

A p -matchoid M ( V , I ) consists of m matroids M 1 ( V 1 , I 1 ), M 2 ( V 2 , I 2 ), . . . , M m ( V m , I m ), where each element in the ground set V appears in the ground sets of at most p of these matroids. Additionally, every subset X ⊆ V is an independent set in I if X ∩V i ∈ I i for any matroid M i ( V i , I i ). In the context of submodular maximization subject to a p -matchoid constraint M ( V , I ), the objective is to find an independent set I ∗ ∈ I that maximizes f ( I ∗ ), where f : 2 V → R ≥ 0 is a (non-monotone) submodular function defined over subsets of the ground set V .

The p -matchoid constraint has various applications in machine learning, particularly in tasks involving optimization and constraint satisfaction including the followings:

Feature selection: In feature selection, where the goal is to pick a subset of features that works well while keeping things simple, p -matchoids can be used to model constraints on the relationships between features. For example, if certain features are highly correlated or redundant, a p -matchoid constraint can ensure that only a limited number of such features are chosen together Das and Kempe [2008], Khanna et al. [2017], Elenberg et al. [2016], Qian and Singer [2019], Quinzan et al. [2023].

Video summarization and movie recommendation tasks: Submodular maximization under p -matchoid constraints is crucial for video summarization Feldman et al. [2018], Mirzasoleiman et al. [2018], Gu et al. [2023] and movie recommendation Harshaw et al. [2022], Badanidiyuru et al. [2020],

∗ Equal Contribution

Liu et al. [2022], Tschiatschek et al. [2018] tasks. It helps select representative content and generate personalized and diverse recommendations, leading to improved user experience and engagement.

Text summarization: In text summarization Lin and Bilmes [2011], Bairi et al. [2015], Lin and Bilmes [2010], Liu et al. [2013], summarizing multiple documents together (e.g., news articles on the same topic or a survey paper) is often necessary. In such scenarios, p -matchoids help ensure that the resulting summary represents a balanced combination of information from each document.

Viral marketing campaigns: Viral marketing promotes products or services through social sharing and word-of-mouth for rapid, exponential message spread. p -matchoids can model relationships and interactions, capturing constraints on information dissemination and influence propagation Zhang et al. [2023], Jin et al. [2021], Cui et al. [2021], Kempe et al. [2003].

In general, p -matchoids provide a flexible way to add complex rules to optimization problems in machine learning. By setting p -matchoid constraints, practitioners can adjust optimization algorithms to meet specific requirements and constraints, resulting in more robust models Quinzan et al. [2021], Friedrich et al. [2019], Wei et al. [2015].

In this paper, we study the non-monotone submodular maximization problem under a p -matchoid constraint in the fully dynamic setting. This model considers a sequence of updates, each of which is either an insertion or a deletion. Using the notation Vt ⊆ V to represent the set of elements present at any given time t , this model assumes that Vt is derived from Vt -1 after the insertion or deletion of an element e at time t .

The ultimate objective of this problem is to maintain an independent set I ∗ ∈ I at each time t , such that f ( I ∗ ) = OPTt , where OPTt = max I ∈I , I ⊆ Vt f ( I ) is the optimal submodular value among any independent set in I present at time t . However, finding an exact solution for this problem, even in an o ffl ine setting where the elements are fixed and provided at once, is NP-hard as it generalizes problems such as unconstrained non-monotone submodular maximization and monotone submodular maximization under a cardinality constraint, both of which are known to be NP-hard Feige et al. [2011], Nemhauser et al. [1978]. Therefore, we focus on developing approximate dynamic algorithms and use their approximation ratios to evaluate their quality. Also, following prior work, we assume having an oracle access to the submodular function f , and measure the performance of our algorithms based on their query complexities.

To underscore some specific aspects of this problem that make it more challenging than some other similar problems in the literature, we highlight the following:

- The nature of the p -matchoid constraint, which is extremely versatile and generalizes even complex combinatorial structures such as the intersection of p matroids, p -uniform hypergraph matching, and by extension, matroid, and cardinality constraints, making it significantly more complex to handle.
- Function f not being monotone, as non-monotone submodular maximization is inherently more di ffi cult than its monotone counterpart, evidenced by the weaker approximation guarantees and increased complexity it typically yields across di ff erent settings and constraints, adding to the challenge.
- The dynamic setting of the problem, where elements get inserted or deleted adversarially, and it requires algorithms to adapt to real-time changes.

To date, the most significant advancement in non-monotone submodular maximization in the dynamic setting is a result presented at NeurIPS'23. In their pioneering study, Banihashem, Biabani, Goudarzi, Hajiaghayi, Jabbarzade, and Monemizadeh Banihashem et al. [2023] proposed a dynamic algorithm for non-monotone submodular maximization under a cardinality constraint k . Their algorithm achieves a (8 + ϵ )-approximation while requiring an expected amortized O ( ϵ -2 k 2 log 3 ( k )) oracle queries per update.

Despite this progress, the development of dynamic algorithms for non-monotone submodular maximization under a matroid constraint , or more generally a p-matchoid constraint , remains an open and unexplored challenge, one that we address in this study.

## 1.1 Our contribution and techniques

We present the main result of this paper in the following theorem.

Theorem 1.1 (Main Theorem) . Let 0 &lt; ϵ ≤ 1 be a parameter. Let M ( V , I ) be a p-matchoid consisting of m matroids M 1 ( V 1 , I 1 ) , M 2 ( V 2 , I 2 ) , . . . , M m ( V m , I m ) , and let k denote the size of the largest independent set in I . Let f : 2 V → R ≥ 0 be a (not necessarily monotone) submodular function defined over subsets of the ground set V . Then, there exists a dynamic algorithm for maximizing the function f subject to the p-matchoid constraint M ( V , I ) , which maintains a (2 p + 2 p p ( p + 1) + 1 + ϵ ) -approximate solution at any time t during the sequence of updates, while performing an expected amortized O ( ϵ -3 pk 4 log 2 k ) oracle queries per update.

As a byproduct, we obtain another result for the special case of cardinality k , as stated below.

Corollary 1.2. There exists a dynamic algorithm for non-monotone submodular maximization subject to a matroid constraint of rank k (including the cardinality constraint k), which achieves a (5 . 82 + ϵ ) -approximate solution using O ( ϵ -3 k 4 log 2 k ) oracle queries per update.

Our algorithm for the cardinality case is an improvement upon the approximation of the dynamic algorithm presented by Banihashem et al.Banihashem et al. [2023], which maintains a (8 + ε )-approximate solution with an expected amortized O ( ε -2 k 2 log 3 ( k )) oracle queries per update.

Under the assumption that the submodular function f is monotone, the approximation guarantee of Theorem 1.1 can be strengthened. An adaptation of our algorithm then yields the following side result:

Proposition 1.3. There exists a dynamic algorithm for monotone submodular maximization subject to a p-matchoid constraint of rank k with (4 p + ϵ ) approximation guarantee and O ( ε -3 pk 4 log 2 ( k )) update time.

## 1.1.1 Overview of our algorithm

We present a dynamic algorithm for non-monotone submodular maximization under p -matchoid constraints, where the ground set evolves through a sequence of insertions and deletions. Our approach builds on an o ffl ine recursive framework that incrementally constructs a feasible solution by combining two core ideas: filtering and selective sampling. The o ffl ine algorithm processes elements in multiple levels, iteratively refining the solution by selecting elements with high marginal gain relative to the structural cost imposed by the p -matchoid constraints. This recursive mechanism forms the backbone of our dynamic algorithm.

At each level, the algorithm maintains a candidate set and a partial solution. An element is promoted to the next level only if its marginal contribution outweighs the cost of maintaining feasibility across the constituent matroids. Within each level, one element is sampled from the surviving candidates and added to the solution, while conflicting elements are removed. This yields a layered recursive structure culminating in the final solution.

Our dynamic algorithm is inspired by the streaming algorithm of Feldman, Karbasi, and Kazemi Feldman et al. [2018], but technical modifications are required to support e ffi cient updates in a fully dynamic setting. Our key idea is to simulate the behavior of the o ffl ine algorithm on an evolving ground set, while ensuring that the amortized update cost remains low.

Our algorithm incorporates two randomized procedures. The first is an initial sampling phase that reduces the input size and transforms a non-monotone problem into a more tractable monotone-like instance. The second is used during updates to guard against adversarially ordered insertions and deletions. When an element is inserted, it is retained with some probability. If retained, it is evaluated for inclusion in the current solution, and a ff ected levels may be reconstructed. Similarly, deletions are handled by checking whether the element contributed to the current solution and selectively rebuilding the necessary components.

This design ensures that each update has low expected cost while maintaining near-optimal performance guarantees. A complete description of the algorithm appears in Section 4.

## 1.1.2 Comparison with prior dynamic and streaming algorithms

Several dynamic and streaming algorithms exist for submodular maximization under various constraints such as cardinality, matroid, and p -matchoid constraints. However, our algorithm di ff ers in several key aspects that we summarize below:

Streaming vs. dynamic models. The algorithm by Feldman et al. Feldman et al. [2018] is specifically designed for the insertion-only streaming setting, and it cannot naturally handle deletions. In the streaming model, once an element is deleted, we must reassess all previously seen elements to find a replacement-this typically requires Ω ( n ) oracle queries. In contrast, the dynamic model demands e ffi cient handling of both insertions and deletions. Applying the streaming algorithm in a dynamic setting would require reconstructing the solution from scratch after every deletion, resulting in linear query complexity per update, which is computationally prohibitive.

Matroid vs. p -matchoid constraints. p -matchoid constraints are a generalization of matroid constraints, which makes adapting existing algorithms more challenging. For example, the algorithm of Banihashem et al. Banihashem et al. [2024] depends on a monotonicity property essential for performing binary search over levels. This property holds in matroids but not in general p -matchoids. Hence, their approach cannot be extended to p -matchoids.

To illustrate the failure of monotonicity in p -matchoids, consider the following counterexample. Let V = { 1 , 2 , 3 } and define a 2-matchoid composed of two matroids. The first has ground set { 1 , 2 } and allows only one element, i.e., I 1 = {∅ , { 1 } , { 2 }} . The second has ground set { 1 , 3 } with I 2 = {∅ , { 1 } , { 3 }} . Thus, the independent sets of the matchoid are I = {∅ , { 1 } , { 2 } , { 3 } , { 2 , 3 }} .

Now consider an additive submodular function with f ( { 3 } ) &lt; f ( { 1 } ) &lt; f ( { 2 } ) / 10. Suppose A 1 = { 1 } is selected at level 1. At level 2, adding element 2 requires removing 1 (due to matroid 1), giving A 2 = { 2 } . Since 1 is now absent, we can include element 3 at level 3, resulting in A 3 = { 2 , 3 } . Thus, element 3 can be added at level 3 but not at level 1 due to conflict with 1, violating monotonicity assumptions. This shows the binary search technique used for matroids fails for p -matchoids.

Monotone vs. non-monotone functions. Our algorithm also handles non-monotone submodular functions, unlike the method of Monemizadeh Monemizadeh [2020], which is designed for monotone functions under cardinality constraints. Monemizadeh's algorithm samples O ( ϵ -2 log n ) elements to ensure a su ffi ciently good one is selected, whereas we require only a single sample due to our algorithmic structure and more refined analysis.

Cardinality vs. p -matchoid constraints. The algorithm by Banihashem et al. Banihashem et al. [2023] is specific to cardinality constraints and achieves an (8 + ϵ )-approximation for non-monotone submodular functions. Since cardinality is a special case of p -matchoid constraints, our algorithm generalizes their result while improving the approximation factor to (5 . 82 + ϵ ).

## 2 Related work

Submodular maximization under a cardinality constraint k in the dynamic model was first studied at NeurIPS'20 by Lattanzi, Mitrovic, Norouzi-Fard, Tarnawski, and Zadimoghaddam Lattanzi et al. [2020], and independently by Monemizadeh Monemizadeh [2020]. Both papers demonstrated how to maintain a (2 + ϵ )-approximate solution for this problem while having fast query time. The amortized query complexity of the dynamic algorithm presented in Lattanzi et al. [2020] is O ( ϵ -11 log 6 ( k ) log 2 ( n )) in expectation. And the algorithm presented in Monemizadeh [2020] has an expected amortized O ( ϵ -3 k 2 log 5 ( n )) query complexity. Later, Chen and Peng Chen and Peng [2022] showed that developing a c -approximation dynamic algorithm for any c &lt; 2 requires an amortized query complexity polynomial in the size of the ground set V .

Recently, at ICML 2023, Duetting, Fusco, Lattanzi, Norouzi-Fard, and Zadimoghaddam Duetting et al. [2023] studied monotone submodular maximization under a matroid constraint in the dynamic model and proposed a dynamic (4 + ϵ )-approximation algorithm with an amortized expected query complexity of O ( k 2 ϵ log( k ) log 2 ( n ) log 3 ( k ϵ )). Simultaneously, Banihashem, Biabani, Goudarzi, Hajiaghayi, Jabbarzade, and Monemizadeh Banihashem et al. [2024] tackled the same problem with the same approximation guarantee but with improved query complexity. Specifically, their algorithm achieves a worst-case expected query complexity of O ( k log( k ) log 3 ( k ϵ )). Additionally, the authors in Banihashem et al. [2024] improved the query complexity of the dynamic algorithm proposed in Monemizadeh [2020] for the cardinality constraint to an expected O ( k ϵ -1 log 2 ( k )).

All these results are for monotone submodular maximization problems. The only result for nonmonotone submodular maximization within the dynamic setting is a recent result presented at NeurIPS'23 by Banihashem, Biabani, Goudarzi, Hajiaghayi, Jabbarzade, and Monemizadeh Banihashem et al. [2023] who presented a dynamic algorithm for non-monotone submodular maximization under the cardinality constraint k . This dynamic algorithm maintains a (8 + ε )-approximate solution for cardinality constraint while having an expected amortized O ( ε -2 k 2 log 3 ( k )) oracle queries per update.

Submodular maximization problems have also been studied in the streaming model. Badanidiyuru, Mirzasoleiman, Karbasi, and Krause Badanidiyuru et al. [2014] investigated monotone submodular maximization under a cardinality constraint k and proposed an insertion-only streaming algorithm with a (2 + ϵ )-approximation guarantee. Chekuri, Gupta, and Quanrud Chekuri et al. [2015] presented a O ( p )-approximation single-pass streaming algorithm for maximizing non-monotone submodular functions subject to p -matchoid constraints using O ( k log k ) space. Feldman, Karbasi, and Kazemi Feldman et al. [2018] and Mirzasoleiman, Jegelka, and Krause Mirzasoleiman et al. [2018] subsequently developed streaming algorithms with improved approximation guarantees for maximizing non-monotone functions under p -matchoid constraints. Specifically, the streaming algorithm in Feldman et al. [2018], which was an inspiration for our dynamic algorithm has the same approximation guarantee of = 4 p + 2 -o (1) as our algorithm, and uses O ( k ) space and O ( km / p ) query calls per element.

## 3 Models and definitions

Let V denote a ground set. A function f : 2 V → R ≥ 0 is termed submodular if it satisfies the inequality f ( A ∪ { e } ) -f ( A ) ≥ f ( B ∪ { e } ) -f ( B ) for all A ⊆ B ⊆ V and e &lt; B . For a subset A ⊆ V and an element e ∈ V , the di ff erence ∆ ( e | A ) = f ( A + e ) -f ( A ) 2 is commonly referred to as the marginal gain of adding e to A . If f satisfies f ( A ∪ e ) ≥ f ( A ) for all choices of A and e , meaning that all marginal gains are non-negative, we refer to f as monotone . Conversely, we refer to f as non-monotone if the marginal gain of adding an element to a subset is not necessarily non-negative; that is, there could be a subset A and an element e &lt; A such that f ( A ∪ e ) &lt; f ( A ).

Matroids and p -matchoids. A matroid M ( V , I ) is defined by a ground set V and a nonempty downward-closed set system I ⊆ 2 V , where I consists of independent sets. It satisfies the exchange axiom : for any pair of independent sets A , B ∈ I with | A | &lt; | B | , there exists an element x ∈ B \ A such that A ∪{ x } ∈ I . Any subset of the ground set V that is not independent is labeled as dependent . A maximal independent set, which becomes dependent upon the addition of any other element, is known as a basis for the matroid M ( V , I ). Conversely, a circuit in a matroid M ( V , I ) is a minimal dependent subset of V , meaning a dependent set whose proper subsets are all independent. Given a subset A of V , the rank of A , denoted as rank ( A ), represents the maximum cardinality of an independent subset within A .

An p-matchoid M ( V , I ) consists of m matroids M 1 ( V 1 , I 1 ) , M 2 ( V 2 , I 2 ) , . . . , M m ( V m , I m ) such that each element in the ground set V appears in the ground sets of at most p of these matroids. The set I of independent sets is defined as the collection of all subsets of V whose projection onto any ground set V i is an independent set in I i . In other words, I = n S ⊆ 2 V : ∀ m i = 1 S ∩ V i ∈ I i o .

A classic example is the 2-matchoid representing the classical matching problem. In this case, given an unweighted graph G ( V , E ), a set M ⊆ E is a matching of G if and only if every vertex u ∈ V is incident to at most one edge of M . We can define a 2-matchoid M ( V , I ) for G with ground set V = E as the edge set of G and independent set I representing all valid matchings in G . Specifically, for each vertex v , a matroid M v ( V v , I v ) is defined, where V v represents the edges in V incident to v . An edge set X ⊆ V is independent in M v ( V v , I v ) if | X ∩ V v | ≤ 1. Thus, any edge set X ⊆ V is an independent set in the 2-matchoid M ( V , I ) (i.e., X ∈ I ) if and only if X is a valid matching in G .

Definition 3.1 ( p -matchoid submodular maximization) . Consider a submodular function f : 2 V → R ≥ 0 (not necessarily monotone). The goal in p-matchoid submodular maximization or submodular maximization subject to the p -matchoid constraint M ( V , I ) is to find an independent set I ∗ ∈ I that maximizes f ( I ∗ ). In other words, if we denote the maximum submodular value of an independent

2 For a set A and an element e , we may represent the union of sets A and { e } as A + e or A ∪ e for convenience. Similarly, when considering a set A with an element e ∈ A , we may denote A \ { e } as A -e or A \ e .

set in I as OPT = max I ∈I f ( I ), then I ∗ ∈ I is an independent set that achieves the optimal value f ( I ∗ ) = OPT .

Oracle queries. We assume that the access to matchoid M ( V , I ) is through an oracle that answers the following types of queries.

- Submodular value oracle: The oracle provides access to a submodular function f : 2 V → R ≥ 0 , allowing retrieval of the value f ( A ) for any subset A ⊂ V . In this query access model, computing the marginal gain f ( A ∪ { e } ) -f ( A ) is achieved through two queries: f ( A ∪ { e } ) and f ( A ), where A ⊆ V and e ∈ V .
- Matroid independence oracle: The oracle allows access to a matroid M i ( V i , I i ) of a p -matchoid M ( V , I ). For any subset S ⊆ V , the oracle answers whether S is an independent set in matroid M i ( V i , I i ) or if it is dependent. In other words, it determines if S ∈ I i .

We assess the time complexity of an algorithm that (approximately) solves p -matchoid submodular maximization in terms of its query complexity , defined as the number of queries made to either the submodular value oracle for f or the matroid independence oracle for I .

Dynamic Model. Consider a sequence S comprising insertions and deletions of elements from an underlying ground set V . Let S t denote the sequence of the first t updates (insertions or deletions) from S . The term time t refers to the moment after the first t updates of the sequence S have been executed. We define Vt as the set of elements that have been inserted until time t but have not been deleted since their latest insertion.

In dynamic p-matchoid submodular maximization , our goal is to have an approximate solution of OPTt = max I t ⊆ Vt : I t ∈I f ( I t ) at any time t . We use the term dynamic α -approximate algorithm to refer to such an algorithm whose output solution at every timestep t is guaranteed to have a function value that is at least α -1 · OPTt .

The performance of dynamic algorithms is evaluated based on their update time , which measures the computational e ff ort required to maintain a feasible solution between two consecutive updates. Accordingly, the query complexity of a dynamic α -approximate algorithm for p -matchoid submodular maximization denotes the number of oracle queries (value or independence) required to compute an independent set whose submodular value is an α -approximation of OPTt , given all computations performed up to time t -1.

Our dynamic algorithm operates within the oblivious adversarial model Carter and Wegman [1977]. In this model, the adversary determines the elements in the set V and their arrival order. However, the adversary remains unaware of the random bits used in the algorithm and, therefore, cannot adaptively choose updates in response to the algorithm's randomly guided choices. It is assumed that the adversary prepares the complete input sequence (of insertions and deletions) before the algorithm starts running.

## 4 O ffl ine and Dynamic algorithms for p -matchoid

First, we explain our o ffl ine method provided in Algorithm 1 that given a ground set V , computes an approximate solution for submodular maximization subject to p -matchoid constraints. This algorithm is fully executed by invoking the I nit ( V ) procedure, which solves the problem in the o ffl ine scenario or acts as a pre-processing step in the dynamic scenario, in which case V may be either empty or not. Then, we show how we handle the insertion and deletion of elements in Algorithm 2. Note that throughout the paper, we assume access to a parameter MAX that approximates the maximum submodular value up to a factor of 2; formally, max v ∈V f ( v ) ≤ MAX ≤ 2max v ∈V f ( v ). This is a standard assumption, which we show how to lift by maintaining parallel runs in Appendix B.

## 4.1 O ffl ine algorithm

In the beginning, our o ffl ine algorithm employs a random sampling procedure named R ate S ampling that intuitively transforms a non-monotone submodular maximization instance into a monotone one. Once the initial sampling is complete, the algorithm uses a recursive procedure to refine a solution from the remaining elements at di ff erent levels. At any level i , we maintain two sets Ai and Bi . The set Bi is the set of every element e ∈ Bi -1 whose marginal gain in case of addition to Ai -1 is large enough

## Algorithm 1 M atchoid C onstruction

## Procedure I nit ( V ):

- 1: A 0 ←∅ , and B 0 ← R ate S ampling ( V )
- 2: Return R ecursion ( A 0 , B 0 )

## Procedure R ate S ampling ( V ):

- 1: Sample each element e ∈ V i.i.d with probability q = ( p + p p 2 + p + 1) -1
- 2: Return sampled set B 0 in Step 1

## Procedure R ecursion ( Ai -1 , Bi -1 ):

- 1: Bi ← F iltering ( Ai -1 , Bi -1 )
- 2: if | Bi | &gt; 0 then
- 3: Sample an element ei ∈ Bi uniformly at random
- 4: Ai ← E xtension ( Ai -1 , ei )
- 5: Return R ecursion ( Ai , Bi )
- 6: Let i ∗ ← i -1
- 7: Return Ai ∗

## Procedure F iltering ( Ai -1 , Bi -1 ):

- 1: Bi ←∅
- 2: for every element e ∈ Bi -1 do
- 3: Ui ( e ) ← F ind S waps ( Ai -1 , e )
- 4: store Ui ( e ) for element e as it may be used later in Procedure E xtension
- 5: if Ui ( e ) , F ail then
- 6: Bi ← Bi ∪ e
- 7: Return Bi

to o ff set the loss of the elements we would need to delete from Ai -1 to preserve its independence, and the set Ai is the solution computed at the first i levels. In the end, we denote the number of the last constructed level by i ∗ , and Ai ∗ would be our final solution. This recursive procedure relies on a filtering process and a second sampling process designed to outmaneuver the adversary.

Initial sampling The first sampling process samples elements of V with probability q = ( p + p p 2 + p + 1) -1 ∈ O ( p -1 ) and discards elements that are not sampled. This simple sampling process is well-established Fahrbach et al. [2019], Mirzasoleiman et al. [2018], Feldman et al. [2018] as a technique to transform an instance of non-monotone submodular maximization into a monotone one. It even speeds up the algorithm as it reduces the number of elements to process.

## Algorithm 2 U pdate O perations

| Procedure I nsert (e): 1: With probability 1 - q Ignore e and return 2: B 0 ← B 0 + e 3: for i ← 1 to i ∗ do 4: U i ( e ) ← F ind S waps ( A i - 1 , e ) 5: if U i ( e ) = FAIL then 6: Return 7: B i ← B i + e 8: With probability r i = 1 &#124; B i &#124; : ① e i ← e ② A i ← E xtension ( A i - 1 , e i ) ③ Return R ecursion ( A i , B i )   | Procedure D elete ( e ): 1: for i ← 0 to i ∗ do 2: if e < B i then 3: Return 4: B i ← B i - e 5: if e i = e then 6: if &#124; B i &#124; > 0 then 7: Sample an element e i ∈ B i uniformly at random 8: A i ← E xtension ( A i - 1 , e i ) 9: Return R ecursion ( A i , B i ) 10: Let i ∗ ← i - 1 11: Return A i ∗   |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Procedure E xtension ( Ai -1 , ei ):

- 1: Set weight w ( ei ) ← ∆ ( ei | Ai -1 )
- 2: Ai ← Ai -1 -Ui ( ei ) + ei
- 3: Return Ai

## Procedure F ind S waps ( Ai -1 , e ):

- 1: Ui ( e ) ← D ependency D etection ( Ai -1 , e )
- 2: c ← q 1 + 1 p
- 3: τ ← max n (1 + c ) · w ( Ui ( e )) , ε MAX 2 k o
- 4: if ∆ ( e | Ai -1 ) ≥ τ then
- 5: Return Ui ( e )
- 6: else
- 7: Return F ail

Procedure D ependency D etection ( Ai -1 , e ):

- 1: Ui ( e ) ←∅
- 2: for j = 1 to m do
- 3: if ( Ai -1 + e ) ∩ V j &lt; I j then
- 5: xj ← arg min e ′ ∈ Xj w ( e ′ )
- 4: Xj ←{ e ′ ∈ Ai -1 | ( Ai -1 -e ′ + e ) ∩V j ∈ I j }
- 6: Ui ( e ) ← Ui ( e ) + xj
- 7: Return Ui ( e )

Filtering. The inputs to the filtering procedure at an arbitrary level i are the sets Ai -1 and Bi -1 . In this procedure, for each e ∈ Bi -1 we decide whether to keep or filter this element at this level based on its suitability for addition to Ai -1. Note that if e is later selected for addition to Ai -1, we fix its weight as w ( e ) = ∆ ( e | Ai -1), where ∆ ( e | A ) : = f ( A ∪{ e } ) -f ( A ) denotes the marginal gain of e with respect to A .

Adding e to Ai -1 may violate its independency. To address this, we compute a set Ui of elements that would need to be removed to make it independent again. Formally, we start with an empty set Ui , then for each matroid M j ( V j , I j ) of p -matchoid M ( V , I ) for which ( Ai -1 + e ) ∩ V j &lt; I j , we identify the element e ′ ∈ Ai -1 with minimum weight that ( Ai -1 + e -e ′ ) ∩ V j ∈ I j and add e ′ to Ui ( e ).

Once Ui ( e ) is computed, we consider two cases: If the marginal gain of adding e to Ai -1 is significant enough to o ff set deleting Ui ( e ), specifically if ∆ ( e | Ai -1 ) ≥ (1 + q 1 + 1 p ) · P e ′ ∈ Ui ( e ) w ( e ′ ), then e will be added to a survivor set Bi , which was initialized as an empty set at the beginning of this level. If this condition is not met, e gets filtered out.

Sampling and E xtension . At the end of the filtering step in level i , set Bi is the set of survivors. We sample one of the survivors, say ei ∈ Bi , and obtain the new solution set Ai = Ai -1 + ei -Ui ( ei ). In addition, we set the weight of sampled element ei to w ( ei ) = ∆ ( ei | Ai -1 ).

Our sampling strategy is similar to that of Monemizadeh Monemizadeh [2020] who used a similar method for dynamic (monotone) submodular maximization under a cardinality constraint. However, in that work, he samples O ( ϵ -2 log( n )) elements to guarantee that one of the sampled elements has a marginal gain above a predetermined threshold. In contrast, our sampling strategy requires sampling just one element, and our filtering guarantees that the marginal gain of this element is su ffi ciently large.

## 4.2 Insertion and deletion algorithms

Upon the insertion of an element e , we ignore e with probability 1 -q and terminate the insertion algorithm. Otherwise (i.e., with probability q ), we iterate through levels i ∈ { 1 , . . . , i ∗ } and check if we can add e to the survivor set Bi . If this is the case, we then reconstruct levels i , . . . , i ∗ with probability 1 | Bi | .

For deleting an element e , we iterate through levels { 1 , . . . , i ∗ } . At each level i , we check if e is the element that has been sampled at level i . If this is the case, similar to the insertion algorithm, we reconstruct levels i , . . . , i ∗ in a recursive manner.

## 5 Analysis

In this section, we provide a sketch of our analysis including some notations and definitions. The detailed proofs are provided in Appendix A.

Survivors are elements of a set Bi -1 that are included in set Bi . Formally, we define them as follows:

Definition 5.1 (Survivor) . Let 1 ≤ i ≤ i ∗ be a level. We call an element e , a survivor for level i if ∆ ( e | Ai -1 ) ≥ (1 + q 1 + 1 p ) · P e ′ ∈ Ui ( e ) w ( e ′ ).

In this definition, Ui ( e ) denotes the set of dependent elements that must get removed from the independent set Ai -1 so that e can be added to it. More precisely, for every matroid M j ( V j , I j ) in the matchoid M ( V , I ), if ( Ai -1 + e ) ∩ V j &lt; I j , we define Xj = { e ′ ∈ Ai -1 | (( Ai -1 -e ′ + e ) ∩ V j ) ∈ I j } , the set of elements whose removal would restore independence in matroid M j ( V j , I j ) after the addition of e . We then select the element xj = arg min e ′ ∈ Xj w ( e ′ ) with the smallest weight among those in Xj and add it to Ui ( e ).

In the analysis of our randomized algorithm, we denote the random variable itself as x for any variable x , and x represents its value during execution. The key random variables used in our analysis are as follows:

- We denote e i as the random variable corresponding to the sampled element ei at level i .
- B i represents the random variable corresponding to the set Bi .

- The random variable i ∗ corresponds to i ∗ , the index of the last non-empty level created.
- We define C i = ( e 1 , . . . , e i -1 , B 0 , . . . , B i ) as the random variable corresponding to configuration Ci = ( e 1 , . . . , ei -1 , B 0 , . . . , Bi ) up to level i .

When observing an update at time t , it is crucial to di ff erentiate between random variables e i , B i , i ∗ , and C i and their respective values before and after the update. To this end, we employ the notations Y -and Y -to represent a random variable and its value before time t . We continue to use Y and Y to denote them at the current time after the execution of the update.

We divide the analysis of the dynamic algorithm algorithm into several steps. We first define a set of invariants and we show that we can maintain them in the course of the dynamic algorithm. Having these invariants in hand, we prove the approximation guarantee and the query complexity of the algorithm.

Step 1: Invariants. Initially, we introduce the following set of invariants.

1. Survivor: Bi = { e ∈ Bi -1 : ∆ ( e | Ai -1 ) ≥ (1 + q 1 + 1 p ) · P e ′ ∈ Ui ( e ) w ( e ′ ) } for all 1 ≤ i ≤ i ∗ + 1.
2. Starter: B 0 = R ate S ampling ( V ) and A 0 = ∅
3. Weight: For 1 ≤ i ≤ i ∗ , ei ∈ Bi and w ( ei ) = ∆ ( ei | Ai -1 )
4. Independent: For 1 ≤ i ≤ i ∗ , Ai = Ai -1 + ei -Ui ( ei )
5. Terminator: Bi ∗ + 1 = ∅
6. Uniform invariant: P [ e i = e | i ∗ ≥ i and C i = Ci ] = 1 | Bi | · 1 [ e ∈ Bi ].

Step 2: Maintenance of invariants. Next, we demonstrate the validity of the invariants at the end of the execution of M atchoid C onstruction . We also show that these invariants remain preserved following each insertion and deletion operation. Specifically, we establish the following lemmas.

Lemma 5.2. At the end of Algorithm M atchoid C onstruction all invariants hold.

Lemma 5.3. If before the insertion or deletion of an element e, the invariants hold, then they also hold after the execution of I nsert ( e ) and D elete ( e ) , respectively.

Step 3: Query complexity. Subsequently, we establish bounds on the expected worst-case query complexity of both insertion and deletion operations. Formally, we prove the following result.

Theorem 5.4. The expected query complexity of each insert / delete for all runs is O ( ε -3 pk 4 log 2 ( k )) .

Step 4: Approximation guarantee. Finally, we demonstrate that, assuming the invariants hold, an independent set Ai ∗ ∈ I of the matchoid M ( V , I ) can be reported, with a submodular value approximating the optimal solution by (2 p + 2 p p ( p + 1) + 1 + ε ).

Theorem 5.5. Suppose that the invariants hold in every run of U pdate O perations . Let Ai ∗ be the independent set that Algorithm U pdate O perations returns. Then, the set Ai ∗ satisfies (2 p + 2 p p ( p + 1) + 1 + ε ) · f ( Ai ∗ ) ≥ OPT, where OPT = max I ∗ ∈I f ( I ∗ ) .

## Acknowledgements

The work is partially supported by DARPA QuICC, ONR MURI 2024 award on Algorithms, Learning, and Game Theory, Army-Research Laboratory (ARL) grant W911NF2410052, NSF AF:Small grants 2218678, 2114269, 2347322, and Royal Society grant IES \ R2 \ 222170.

## References

- A. Badanidiyuru, B. Mirzasoleiman, A. Karbasi, and A. Krause. Streaming submodular maximization: Massive data summarization on the fly. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining , pages 671-680, 2014.
- A. Badanidiyuru, A. Karbasi, E. Kazemi, and J. Vondrák. Submodular maximization through barrier functions. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in

Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual , 2020. URL https://proceedings. neurips.cc/paper/2020/hash/061412e4a03c02f9902576ec55ebbe77-Abstract.html .

- R. Bairi, R. K. Iyer, G. Ramakrishnan, and J. A. Bilmes. Summarization of multi-document topic hierarchies using submodular mixtures. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing of the Asian Federation of Natural Language Processing, ACL 2015, July 26-31, 2015, Beijing, China, Volume 1: Long Papers , pages 553-563. The Association for Computer Linguistics, 2015. doi: 10.3115 / V1 / P15-1054. URL https://doi.org/10.3115/ v1/p15-1054 .
- K. Banihashem, L. Biabani, S. Goudarzi, M. Hajiaghayi, P. Jabbarzade, and M. Monemizadeh. Dynamic non-monotone submodular maximization. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023 , 2023. URL http://papers.nips.cc/paper\_files/paper/ 2023/hash/387982dbf23d9975c7fc45813dd3dabc-Abstract-Conference.html .
- K. Banihashem, L. Biabani, S. Goudarzi, M. Hajiaghayi, P. Jabbarzade, and M. Monemizadeh. Dynamic algorithms for matroid submodular maximization. In Proceedings of the 2024 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA) , pages 3485-3533. SIAM, 2024.
- N. Buchbinder, M. Feldman, J. Naor, and R. Schwartz. Submodular maximization with cardinality constraints. In C. Chekuri, editor, Proceedings of the Twenty-Fifth Annual ACM-SIAM Symposium on Discrete Algorithms, SODA 2014, Portland, Oregon, USA, January 5-7, 2014 , pages 14331452. SIAM, 2014. ISBN 978-1-61197-338-9. doi: 10.1137 / 1.9781611973402.106. URL https://doi.org/10.1137/1.9781611973402.106 .
- L. Carter and M. N. Wegman. Universal classes of hash functions (extended abstract). In Proceedings of the 9th Annual ACM Symposium on Theory of Computing, May 4-6, 1977, Boulder, Colorado, USA , pages 106-112, 1977. doi: 10.1145 / 800105.803400. URL https://doi.org/10.1145/ 800105.803400 .
- C. Chekuri, S. Gupta, and K. Quanrud. Streaming algorithms for submodular function maximization. In M. M. Halldórsson, K. Iwama, N. Kobayashi, and B. Speckmann, editors, Automata, Languages, and Programming - 42nd International Colloquium, ICALP 2015, Kyoto, Japan, July 6-10, 2015, Proceedings, Part I , volume 9134 of Lecture Notes in Computer Science , pages 318-330. Springer, 2015. doi: 10.1007 / 978-3-662-47672-7 \ \_26. URL https://doi.org/10.1007/978-3-662-47672-7\_26 .
- X. Chen and B. Peng. On the complexity of dynamic submodular maximization. In Proceedings of the Fifty-Fourth Annual ACM on Symposium on Theory of Computing, STOC 2022, to appear , 2022. URL https://arxiv.org/abs/2111.03198 .
- S. Cui, K. Han, T. Zhu, J. Tang, B. Wu, and H. Huang. Randomized algorithms for submodular function maximization with a k-system constraint. In M. Meila and T. Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event , volume 139 of Proceedings of Machine Learning Research , pages 2222-2232. PMLR, 2021. URL http://proceedings.mlr.press/v139/cui21b.html .
- A. Das and D. Kempe. Algorithms for subset selection in linear regression. In Proceedings of the 40th Annual ACM Symposium on Theory of Computing, Victoria, British Columbia, Canada, May 17-20, 2008 , pages 45-54. ACM, 2008. doi: 10.1145 / 1374376.1374384. URL https: //doi.org/10.1145/1374376.1374384 .
- P. Duetting, F. Fusco, S. Lattanzi, A. Norouzi-Fard, and M. Zadimoghaddam. Fully dynamic submodular maximization over matroids. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA , volume 202 of Proceedings of Machine Learning Research , pages 8821-8835. PMLR, 2023. URL https://proceedings.mlr.press/v202/ duetting23a.html .

- E. R. Elenberg, R. Khanna, A. G. Dimakis, and S. N. Negahban. Restricted strong convexity implies weak submodularity. CoRR , abs / 1612.00804, 2016. URL http://arxiv.org/abs/ 1612.00804 .
- M. Fahrbach, V. S. Mirrokni, and M. Zadimoghaddam. Non-monotone submodular maximization with nearly optimal adaptivity and query complexity. In Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA , volume 97 of Proceedings of Machine Learning Research , pages 1833-1842. PMLR, 2019. URL http://proceedings.mlr.press/v97/fahrbach19a.html .
- U. Feige, V. S. Mirrokni, and J. Vondrák. Maximizing non-monotone submodular functions. SIAM J. Comput. , 40(4):1133-1153, 2011. doi: 10.1137 / 090779346. URL https://doi.org/10.1137/ 090779346 .
- M. Feldman, A. Karbasi, and E. Kazemi. Do less, get more: Streaming submodular maximization with subsampling. In S. Bengio, H. M. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada , pages 730-740, 2018. URL https://proceedings.neurips.cc/paper/ 2018/hash/d1f255a373a3cef72e03aa9d980c7eca-Abstract.html .
- T. Friedrich, A. Göbel, F. Neumann, F. Quinzan, and R. Rothenberger. Greedy maximization of functions with bounded curvature under partition matroid constraints. In The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019 , pages 2272-2279. AAAI Press, 2019. doi: 10.1609 / AAAI.V33I01.33012272. URL https://doi.org/10.1609/aaai.v33i01.33012272 .
- Y. Gu, C. Bian, and C. Qian. Submodular maximization under the intersection of matroid and knapsack constraints. In B. Williams, Y. Chen, and J. Neville, editors, Thirty-Seventh AAAI Conference on Artificial Intelligence, AAAI 2023, Thirty-Fifth Conference on Innovative Applications of Artificial Intelligence, IAAI 2023, Thirteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2023, Washington, DC, USA, February 7-14, 2023 , pages 3959-3967. AAAI Press, 2023. doi: 10.1609 / AAAI.V37I4.25510. URL https://doi.org/10.1609/aaai.v37i4.25510 .
- C. Harshaw, E. Kazemi, M. Feldman, and A. Karbasi. The power of subsampling in submodular maximization. Math. Oper. Res. , 47(2):1365-1393, 2022. doi: 10.1287 / MOOR.2021.1172. URL https://doi.org/10.1287/moor.2021.1172 .
- T. Jin, Y . Yang, R. Yang, J. Shi, K. Huang, and X. Xiao. Unconstrained submodular maximization with modular costs: Tight approximation and application to profit maximization. Proc. VLDB Endow. , 14(10):1756-1768, 2021. doi: 10.14778 / 3467861.3467866. URL http://www.vldb. org/pvldb/vol14/p1756-jin.pdf .
- E. Kazemi, M. Mitrovic, M. Zadimoghaddam, S. Lattanzi, and A. Karbasi. Submodular streaming in all its glory: Tight approximation, minimum memory and low adaptive complexity. In K. Chaudhuri and R. Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA , volume 97 of Proceedings of Machine Learning Research , pages 3311-3320. PMLR, 2019. URL http://proceedings. mlr.press/v97/kazemi19a.html .
- D. Kempe, J. M. Kleinberg, and É. Tardos. Maximizing the spread of influence through a social network. In L. Getoor, T. E. Senator, P. M. Domingos, and C. Faloutsos, editors, Proceedings of the Ninth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Washington, DC, USA, August 24 - 27, 2003 , pages 137-146. ACM, 2003. doi: 10.1145 / 956750. 956769. URL https://doi.org/10.1145/956750.956769 .
- R. Khanna, E. R. Elenberg, A. G. Dimakis, S. N. Negahban, and J. Ghosh. Scalable greedy feature selection via weak submodularity. In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, AISTATS 2017, 20-22 April 2017, Fort Lauderdale, FL, USA , volume 54 of Proceedings of Machine Learning Research , pages 1560-1568. PMLR, 2017. URL http://proceedings.mlr.press/v54/khanna17b.html .

- S. Lattanzi, S. Mitrovic, A. Norouzi-Fard, J. Tarnawski, and M. Zadimoghaddam. Fully dynamic algorithm for constrained submodular optimization. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual , 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 9715d04413f296eaf3c30c47cec3daa6-Abstract.html .
- H. Lin and J. A. Bilmes. Multi-document summarization via budgeted maximization of submodular functions. In Human Language Technologies: Conference of the North American Chapter of the Association of Computational Linguistics, Proceedings, June 2-4, 2010, Los Angeles, California, USA , pages 912-920. The Association for Computational Linguistics, 2010. URL https:// aclanthology.org/N10-1134/ .
- H. Lin and J. A. Bilmes. A class of submodular functions for document summarization. In D. Lin, Y. Matsumoto, and R. Mihalcea, editors, The 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, Proceedings of the Conference, 19-24 June, 2011, Portland, Oregon, USA , pages 510-520. The Association for Computer Linguistics, 2011. URL https://www.aclweb.org/anthology/P11-1052/ .
- S. Liu, N. Lu, C. Chen, and K. Tang. E ffi cient combinatorial optimization for word-level adversarial textual attack. IEEE ACM Trans. Audio Speech Lang. Process. , 30:98-111, 2022. doi: 10.1109 / TASLP.2021.3130970. URL https://doi.org/10.1109/TASLP.2021.3130970 .
- Y. Liu, K. Wei, K. Kirchho ff , Y. Song, and J. A. Bilmes. Submodular feature selection for highdimensional acoustic score spaces. In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2013, Vancouver, BC, Canada, May 26-31, 2013 , pages 7184-7188. IEEE, 2013. doi: 10.1109 / ICASSP.2013.6639057. URL https://doi.org/10.1109/ICASSP. 2013.6639057 .
- B. Mirzasoleiman, S. Jegelka, and A. Krause. Streaming non-monotone submodular maximization: Personalized video summarization on the fly. In S. A. McIlraith and K. Q. Weinberger, editors, Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018 , pages 1379-1386. AAAI Press, 2018. URL https://www.aaai.org/ocs/ index.php/AAAI/AAAI18/paper/view/17014 .
- M. Monemizadeh. Dynamic submodular maximization. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual , 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 6fbd841e2e4b2938351a4f9b68f12e6b-Abstract.html .
- G. L. Nemhauser, L. A. Wolsey, and M. L. Fisher. An analysis of approximations for maximizing submodular set functions - I. Math. Program. , 14(1):265-294, 1978. doi: 10.1007 / BF01588971. URL https://doi.org/10.1007/BF01588971 .
- S. Qian and Y. Singer. Fast parallel algorithms for statistical subset selection problems. In Proceedings of the 33rd International Conference on Neural Information Processing Systems , Red Hook, NY, USA, 2019. Curran Associates Inc.
- F. Quinzan, V. Doskoc, A. Göbel, and T. Friedrich. Adaptive sampling for fast constrained maximization of submodular functions. In A. Banerjee and K. Fukumizu, editors, The 24th International Conference on Artificial Intelligence and Statistics, AISTATS 2021, April 13-15, 2021, Virtual Event , volume 130 of Proceedings of Machine Learning Research , pages 964-972. PMLR, 2021. URL http://proceedings.mlr.press/v130/quinzan21a.html .
- F. Quinzan, R. Khanna, M. Hershcovitch, S. Cohen, D. G. Waddington, T. Friedrich, and M. W. Mahoney. Fast feature selection with fairness constraints. In F. J. R. Ruiz, J. G. Dy, and J. van de Meent, editors, International Conference on Artificial Intelligence and Statistics, 25-27 April 2023, Palau de Congressos, Valencia, Spain , volume 206 of Proceedings of Machine Learning Research , pages 7800-7823. PMLR, 2023. URL https://proceedings.mlr.press/v206/ quinzan23a.html .

- S. Tschiatschek, A. Sahin, and A. Krause. Di ff erentiable submodular maximization. In J. Lang, editor, Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, IJCAI 2018, July 13-19, 2018, Stockholm, Sweden , pages 2731-2738. ijcai.org, 2018. doi: 10.24963 / IJCAI.2018 / 379. URL https://doi.org/10.24963/ijcai.2018/379 .
- K. Wei, R. K. Iyer, and J. A. Bilmes. Submodularity in data subset selection and active learning. In F. R. Bach and D. M. Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015 , volume 37 of JMLR Workshop and Conference Proceedings , pages 1954-1963. JMLR.org, 2015. URL http://proceedings.mlr. press/v37/wei15.html .
- G. Zhang, N. Tatti, and A. Gionis. Ranking with submodular functions on the fly. In S. Shekhar, Z. Zhou, Y. Chiang, and G. Stiglic, editors, Proceedings of the 2023 SIAM International Conference on Data Mining, SDM 2023, Minneapolis-St. Paul Twin Cities, MN, USA, April 27-29, 2023 , pages 604-612. SIAM, 2023. doi: 10.1137 / 1.9781611977653.CH68. URL https://doi.org/10. 1137/1.9781611977653.ch68 .

## A Omitted proofs

We begin proving the main theorem of our paper by first introducing basic probabilistic terminology.

Conditional expectation. For a function x and a set A , we denote x [ A ] as the function x restricted to the domain A . For an event E , we define 1 [ E ] as the indicator function of E , i.e., 1 [ E ] is set to one if E holds and zero otherwise. When dealing with random variables and their values, we use bold and non-bold letters, respectively. For instance, a random variable is denoted as X , and its value is represented by X . We use the notations P [ X ] and E [ X ] for the probability and expectation of a random variable X . For events A and B , the notation P [ A | B ] denotes " the conditional probability of A given B " or " the probability of A under the condition B ". For an event A with nonzero probability and a discrete random variable X , we denote by E [ X | A ] the conditional expectation of X given A , i.e., E [ X | A ] = P x x · P [ X = x | A ]. Similarly, for discrete random variables X and Y , the conditional expectation of X given Y is denoted by E GLYPH&lt;2&gt; X | Y = y GLYPH&lt;3&gt; .

## A.1 Maintenance of survivor invariant

We first prove that the survivor invariant holds at the end of the algorithm M atchoid C onstruction . Later, we show that it holds after the execution of I nsert ( e ) and D elete ( e ) provided it was upheld before the insertion or deletion of an element e . We employ a similar methodology to demonstrate the preservation of the remaining invariants. In particular, we first prove them for the algorithm M atchoid C onstruction and then for I nsert ( e ) and D elete ( e ).

Survivor invariant for M atchoid C onstruction ( V ) : We inductively prove that survivor invariant holds for any element e and any level i &lt; i ∗ . For the base case i = 0, if any arbitrary element e ∈ V is sampled with probability q , element e will be in B 0. Otherwise, it will not be added to B 0 .

Now, let us consider an arbitrary element e that is in survivor set Bi -1 of a level i -1 where 1 ≤ i &lt; i ∗ . Recall that Ui ( e ) is the set of dependent elements that needs to be deleted from independent set Ai -1 so that e can be added to it. In the algorithm F iltering , we check if ∆ ( e | Ai -1 ) ≥ (1 + q 1 + 1 p ) · P e ′ ∈ Ui -1( e ) w ( e ′ ). If that is the case, e is added to Bi , otherwise, e is not added to Bi . Thus, the survivor invariant holds.

Survivor invariant for I nsert ( e ) : We first give a few useful facts and for that, define variables r and s as follows.

- ✰ Let s be the largest i ∈ [0 , i ∗-+ 1] such that e is added to B -i .
- ✰ If there is a level i ∈ [ i ∗-] in which ei = e , then we let r be i . Otherwise, we let r be i ∗-+ 1. We consider two cases.
- ☛ Case 1 happens if r ≤ i ∗-. Then, we know that Subroutines E xtension and R ecursion have been invoked. In addition, we have er = e and s = r .
- ☛ Case 2 occurs if r = i ∗-+ 1 which means Subroutines E xtension and R ecursion have never been invoked for I nsert ( e ). Thus, the last level i ∗ after inserting e is the same last level i ∗-before inserting e . That is, i ∗ = i ∗-. Note that, we must have s , i ∗-+ 1, since otherwise, we have | B -i ∗-+ 1 | = 0 and | Bi ∗ + 1 | = 1. Therefore, we have invoked Subroutines E xtension and R ecursion for this level what means i ∗ &gt; i ∗-which is not the case. Thus, s &lt; r = i ∗-+ 1.

We handle I nsert ( v ). Therefore, for any i &lt; r , we have not made any change in variables e -i , w ( ei ), or A -i because we have not invoked Subroutines E xtension and R ecursion for those levels upon insertion of e . Hence, the following facts are correct.

Fact A.1 . For any i ∈ [1 , r ) we have: ➊ ei = e -i ➋ w ( ei ) = w -( ei ) ➌ Ai = A -i . For i = 0, we have Ai = A -i .

By the definition of s , we add e to the set B -i , for each i ∈ [0 , s ] and we have s ≤ r . In addition, by invoking Subroutines E xtension and R ecursion for level r + 1, nothing happens to the variables in levels that are less than r + 1. Thus, we obtain the following fact.

Fact A.2 . For any i ∈ [1 , s ], we have Bi = B -i + e .

Fact A.3 . For any i ∈ [ s + 1 , r ], we have Bi = B -i .

glyph[negationslash]

glyph[negationslash]

<!-- image -->

Case 2 of inserting an element e

Figure 1: Visualisation of two cases upon I nsert ( e ).

Next, we show that the survivor invariant holds, i.e., Bi = { e ∈ Bi -1 : ∆ ( e | Ai -1 ) ≥ (1 + q 1 + 1 p ) · P e ′ ∈ Ui ( e ) w ( e ′ ) } for i ∈ [1 , i ∗ ].

First, let us consider i ∈ [1 , s ]. By the assumption, the survivor invariant holds before the the insertion of element e at time t . That is, B -i = { e ∈ B -i -1 : ∆ ( e | A -i -1 ) ≥ (1 + q 1 + 1 p ) · P e ′ ∈ Ui ( e ) w -( e ′ ) } for i ∈ [1 , s ]. Using Fact A.2, Bi = B -i + v holds for all i ∈ [1 , s ]. Thus,

<!-- formula-not-decoded -->

According to Fact A.1, for any i ∈ [1 , s ] ⊆ [1 , r ) we have: ➊ ei = e -i ➋ w ( ei ) = w -( ei ) ➌ Ai = A -i . Therefore, we obtain the following:

<!-- formula-not-decoded -->

Recall that if Case 1 occurs, r = s . Since in this case, Subroutines E xtension and R ecursion have been invoked, we know that the survivor invariant will be held for all level [ s + 1 , i ∗ ]. However, if Case 2 happens, we need to prove that the survivor invariant is held for all levels [ s + 1 , i ∗ ].

Thus, we are in Case 2 and we need to prove the survivor invariant for all levels [ s + 1 , i ∗ ]. We have r = i ∗-+ 1 = i ∗ + 1. According to Fact A.3, for any i ∈ [ s + 1 , i ∗ + 1], we have Bi = B -i .

<!-- formula-not-decoded -->

Once again, using Fact A.1, for any i ∈ [1 , r ) we have: ➊ ei = e -i ➋ w ( ei ) = w -( ei ) ➌ Ai = A -i what means

<!-- formula-not-decoded -->

This essentially means that the survivor invariant holds after I nsert ( e ).

Survivor invariant for D elete ( e ) : Similar to I nsert ( e ), We first define variables r and s as follows.

- ✰ Let s be the largest i ∈ [0 , i ∗-+ 1] from which e is deleted from B -i .
- ✰ If there is a level i ∈ [1 , i ∗-] in which ei = e , then we let r be i . Otherwise, we let r be i ∗-+ 1. We consider two cases.
- ☛ Case 1 happens if r ≤ i ∗-. Then, we know that Subroutines E xtension and R ecursion have been invoked. In addition, we have er = e and s = r .
- ☛ Case 2 occurs if r = i ∗-+ 1 which means Subroutines E xtension and R ecursion have never been invoked for I nsert ( e ). Thus, the last level i ∗ after deleting e is the same last level i ∗-before deleting e . That is, i ∗ = i ∗-. Note that, we must have s &lt; r = i ∗-+ 1, since otherwise, we have | B -i ∗-+ 1 | = 1 which cannot be the case.

Our update is D elete ( v ). Thus, in any level i ∈ [1 , i ∗ ), we do not make any change in in these levels other than removing the element e from B -i if i ≤ s .

Fact A.5 . For any i ∈ [0 , s ], it holds that Bi = B -i \{ v } and for any i ∈ [ s + 1 , r ], we have Bi = B -i .

Fact A.4 . For any i ∈ [1 , r ) we have: ➊ ei = e -i ➋ w ( ei ) = w -( ei ) ➌ Ai = A -i . For i = 0, we have Ai = A -i .

By the assumption, the survivor invariant holds before the the deletion of element e at time t . That is, B -i = { e ∈ B -i -1 : ∆ ( e | A -i -1 ) ≥ (1 + q 1 + 1 p ) · P e ′ ∈ Ui ( e ) w -( e ′ ) } for i ∈ [1 , s ]. Thus, using Fact A.5, for i ∈ [1 , s ]

<!-- formula-not-decoded -->

and for i ∈ [ s + 1 , r ],

<!-- formula-not-decoded -->

Since by Fact A.4, for any i ∈ [1 , r ) we have ➊ ei = e -i ➋ w ( ei ) = w -( ei ) ➌ Ai = A -i , we conclude that

<!-- formula-not-decoded -->

what proves that the survivor invariant holds for i ∈ [1 , r ) after the deletion of element e . Recall that if r ≤ i ∗-, Subroutines E xtension and R ecursion have been invoked, therefore, we know that the survivor invariant will be held for all levels [ r , i ∗ ].

## A.2 Maintenance of starter invariant

Starter invariant for I nsert ( e ) : Recall that in Subroutine R ate S ampling ( V ), we sample each element e ∈ V i.i.d with probability q = ( p + p p 2 + p + 1) -1 and let B 0 be the set of elements that are sampled. By our assumption, we have B -0 = R ate S ampling ( V ) and A -0 = 0.

Upon insertion of an element e , we add it to B -0 with probability q . Observe that we always add e to the ground set. Thus, V = V -+ e . Condition of this event which happens with probability q , we have B 0 = B -0 + e . Thus, B 0 = R ate S ampling ( V -+ e ) = R ate S ampling ( V ). In addition, using Fact A.1, we have A 0 = A -0 = 0.

Starter invariant for D elete ( e ) : By our assumption, we have B -0 = R ate S ampling ( V ) and A -0 = 0. Moreover, V 0 = V 0 \{ e } . If e ∈ B -0 (what happens with probability q ), we then remove it from B -0 . That is, B 0 = B -0 \{ e } = R ate S ampling ( V - \{ e } ) = R ate S ampling ( V ). Using Fact A.4, Ai = A -i = 0.

This proves that the starter invariant holds after insertion or deletion of an arbitrary element e .

## A.3 Maintenance of independent invariant

We assume that the independent invariant holds until time t . That is, for any level 1 ≤ i ≤ i ∗-, A -i = A -i -1 + e -i -Ui ( e -i ). Upon the insertion or deletion of an arbitrary element e at time t , we show that Ai = Ai -1 + ei -Ui ( ei ) for any level 1 ≤ i ≤ i ∗ .

Remarkably, we can establish this invariant for both insertions and deletions simultaneously, without the need for separate consideration of each operation. Indeed, according to Facts A.1 and A.4, upon the insertion or deletion of an arbitrary element e at time t , for any i ∈ [1 , r ) we have ➊ ei = e -i ➋ w ( ei ) = w -( ei ) ➌ Ai = A -i and for i = 0, we have Ai = A -i . Thus, Ai = A -i = A -i -1 + e -i -Ui ( e -i ).

Now, observe that for any j ≤ i -1 &lt; r , we have ➊ ej = e -j ➋ w ( ej ) = w -( ej ) ➌ Aj = A -j and for j = 0, we have Aj = A -j . This essentially means that Ai = Ai -1 + ei -Ui ( ei ) for any level 1 ≤ i ≤ i ∗ . Also, recall that if r ≤ i ∗-, Subroutines E xtension and R ecursion have been invoked, therefore, we know that the independent invariant will be held for all levels [ r , i ∗ ].

## A.4 Maintenance of weight invariant

Next, we show that the weight invariant holds. That is, assuming that for 1 ≤ i ≤ i ∗-, e -i ∈ B -i and w -( e -i ) = ∆ ( e -i | A -i -1 ). We show that upon the insertion or deletion of an arbitrary element e at time t , ei ∈ Bi and w ( ei ) = ∆ ( ei | Ai -1) for 1 ≤ i ≤ i ∗ .

Weight invariant for I nsert ( e ) : According to Fact A.2, for any i ∈ [1 , s ], we have Bi = B -i + e and for any i ∈ [ s + 1 , r ], we have Bi = B -i using Fact A.3. Thus, B -i ⊆ Bi .

Using Fact A.1, for any i ∈ [1 , r ) we have ➊ ei = e -i ➋ w ( ei ) = w -( ei ) ➌ Ai = A -i . For i = 0, we have Ai = A -i . Therefore, ei = e -i ∈ B -i ⊆ Bi for any i ∈ [1 , i ∗ ). This essentially means that ei ∈ Bi .

Next we need to show w ( ei ) = ∆ ( ei | Ai -1 ) = f ( Ai -1 + ei ) -f ( Ai -1 ). For any i ∈ [1 , r ), we have w ( ei ) = w -( ei ) and ei = e -i . Then, w -( ei ) = w -( e -i ). Moreover, the assumption of this lemma implies that w -( e -i ) = ∆ ( e -i | A -i -1 ) = f ( A -i -1 + e -i ) -f ( A -i -1 ). Since ei = e -i , we obtain f ( A -i -1 + e -i ) -f ( A -i -1 ) = f ( Ai -1 + ei ) -f ( Ai -1). Putting everything together, for any i ∈ [1 , r ) we have

<!-- formula-not-decoded -->

Weight invariant for D elete ( e ) : We next prove that the weight invariant holds upon deletion of an element e at time t . According to Fact A.4, for any i ∈ [1 , r ) we have ➊ ei = e -i ➋ w ( ei ) = w -( ei ) ➌ Ai = A -i . In addition, for i = 0, we have Ai = A -i . From the definition of r we know that e -i ∈ R -i \{ e } . Using Fact A.5, for any i ∈ [0 , s ], it holds that Bi = B -i \{ v } and for any i ∈ [ s + 1 , r ], we have Bi = B -i . Therefore, Bi = B -i \{ v } which means that ei = e -i ∈ B -i \{ e } = Bi .

By the assumption, we have w -( e -i ) = ∆ ( e -i | A -i -1 ) = f ( A -i -1 + e -i ) -f ( A -i -1 ) . Thus, for any i ∈ [1 , r ) we have w ( ei ) = w -( ei ) = w -( e -i ) = f ( A -i -1 + e -i ) -f ( A -i -1 ) = f ( Ai -1 + ei ) -f ( Ai -1 ) = ∆ ( ei | Ai -1 ) which completes the proof that the weight invariant holds upon deletion of an element e at time t .

## A.5 Maintenance of terminator invariant

Next, we prove that the terminator invariant holds. That is, upon the insertion or deletion of an arbitrary element e at time t , we have Bi ∗ + 1 = ∅ assuming that B -i ∗-+ 1 = ∅ holds.

By the definition of r , if r ≤ i ∗-we know that r is the level where er = e upon insertion or deletion of an element e at time t , we know that Subroutines E xtension and R ecursion have been invoked. In addition, we have s = r . If this happens, R ecursion subroutine terminates when | Bi | &gt; 0. As the result, we have i ∗ = i -1 which means that Bi ∗ + 1 = ∅ .

However, if r = i ∗-+ 1, we know that Subroutines E xtension and R ecursion have never been invoked for I nsert ( e ) and D elete ( e ) . Thus, the last level i ∗ after inserting e is the same last level i ∗-before inserting e . That is, i ∗ = i ∗-. Note that, we must have s , i ∗-+ 1, since otherwise, we have | B -i ∗-+ 1 | = 0 and | Bi ∗ + 1 | = 1. Therefore, we have invoked Subroutines E xtension and R ecursion for this level what means i ∗ &gt; i ∗-which is not the case. Thus, s &lt; r = i ∗-+ 1 which means Bi ∗ + 1 = B -i ∗-+ 1 = ∅ .

## A.6 Maintenance of uniform invariant

We now prove that the uniform invariant holds. That is, for any i ≤ i ∗ we have P [ e i = e | i ∗ ≥ i and C i = Ci ] = 1 | Bi | · 1 [ e ∈ Bi ]. Consider the time when the element ei is sampled. At this time, all of the random values in C i are realized. By construction, ei is sampled uniformly at random from Bi . Therefore, at this time the claim holds. Additionally, none of the values in C i are changed afterwards in the execution. Therefore, the claim holds at the end of the execution as well.

In this section, we prove the uniform invariant, i.e., we show that for any i ≤ i ∗ we have P [ e i = e | i ∗ ≥ i and C i = Ci ] = 1 | Bi | · 1 [ e ∈ Bi ]. We first that the invariant holds after calling I nit because we sample ei to be uniformly at random from Bi and none of the value of C i are changed after this sampling. The main thing to prove is that the invariant holds after each insertion and deletion operation.

Assume that the invariant holds before some update which is either the insertion or deletion of an element v . we will prove that it holds after the update as well. We use the superscript -to denote values before the update, e.g., e -i denotes the value of ei before the update. When no superscript is used, we are referring to values after the update. By assumption, the invariant holds before the update, i.e., for any i and e , and any Ci such that P h i ∗- ≥ i , C -i = Ci i &gt; 0 we have we have

<!-- formula-not-decoded -->

We need to show that after the update, for any arbitrary i and e , and any Ci such that P [ i ∗ ≥ i , C i = Ci ] &gt; 0 we have

<!-- formula-not-decoded -->

Note that P [ e i = e | i ∗ ≥ i , C i = Ci ], is only defined when P [ i ∗ ≥ i , C i = Ci ] &gt; 0, which means that given the input and considering the behavior of our algorithm including its random choices, it is possible to reach a state where i ∗ ≥ i and C i = Ci .

## A.6.1 Insertions

If v &lt; B 0, then it means that the data structure has ignored v , which in turn implies that C i = C -i and e i = e -i , and i ∗ = i ∗ , -which means the claim follows from the induction assumption. Formally, the following claim holds.

Claim A.6 . If v &lt; B 0, then events { i ∗ ≥ i , C i = Ci } and n i ∗ , - ≥ i , C -i -1 = Ci -1 , v &lt; B 0 o are equivalent. Furthermore, the event implies that e i = e -i .

Proof. The first direction of equivalency is clear; if we have { i ∗ ≥ i , C i = Ci } then the algorithm has ignored v because otherwise we would have v ∈ B0 = B 0. Therefore, we have i ∗ , -= i ∗ and C -i = C i . For the other direction, since v &lt; B 0 we know that the algorithm has ignored v which means that i ∗ , -= i ∗ and C -i = C i as before.

Since the event implies that the algorithm has ignored v , we have e -i = e i for all i . □

Given the above claim, for all Ci such that v &lt; B 0 ,

<!-- formula-not-decoded -->

Here, for the second equality we have used the fact that the realization of e -i is independent of whether v ∈ B 0. This is because the algorithm decides whether or not v ∈ B 0 by flipping a coin with probability 1 -q afte e -i is realized.

We therefore focus on Ci for which v ∈ B 0 .

For any i , let P i denote the random variable that is 1 if we invoke R ecursion ( Aj , Bj ) for some j ≤ i , and equals 0 otherwise. We first show that

<!-- formula-not-decoded -->

This is because e i is obtained by sampling uniformly at random from Bi and at the time of the sampling, the values in C i are realized and they do not change afterwards.

We now consider the case where P i -1 = 0. We claim that

<!-- formula-not-decoded -->

If v &lt; Bi , then we have B i = B -i and e i = e -i . Define C -i as C -i = ( B 0 \ { v } , . . . , Bi \ { v } , e 1 , . . . , ei -1 ). Define the event σ as

<!-- formula-not-decoded -->

Claim A.7 . For any Ci such that P [ i ∗ ≥ i , C i = Ci , P i -1 = 0] &gt; 0, the event σ is equivalent to { i ∗ ≥ i , C i = Ci , P i -1 = 0 }

Proof. Assume that the said event holds. We first note that i ∗ , - ≥ i ; if this is not the case then P i -1 = 0 implies that i ∗ &lt; i as well which is not possible. Additionally, we must have C i = C -i because P i -1 = 0 ensures that e j = e -j for all j ≤ i and the only di ff erence between B j and B -j is that the former may possibly contain v . Finally, we have v ∈ B 0 given the assumption on Ci made in the beginning of the proof. We note that since we assumed that P [ i ∗ ≥ i , C i = Ci , P i -1 = 0] &gt; 0, this further means that P [ σ, C i = Ci , ] &gt; 0.

For the converse, assume that σ holds. Since P i -1 = 0, we have i ∗ = i ∗ , -. We therefore need to show that C i = Ci . The values in e 1 , . . . , e i -1 are not changed so e j = e -j = ej by definition of C -i . Additionally, the values of B 0 , . . . , B i are now deterministic because we have fixed the realization of the coin flips occurring with probability 1 | Bj | by assuming P i -1 = 0, and we have assumed v ∈ B 0. Therefore, there is only one possible value for C i . Therefore, either P [ C i = Ci | σ ] = 0 or P [ C i = Ci | σ ] = 1. We have already proved that P [ C i = Ci , σ ] &gt; 0, which means we must have P [ C i = Ci | σ ] = 1, finishing the proof. □

<!-- formula-not-decoded -->

Proof. If σ holds then P i -1 = 0 which means that P i = 0 if and only if the algorithm flips the coin in Line 2 to invoke recursion. If v &lt; Bi , then this cannot happen because Line 2 would never be executed. If v ∈ Bi , then we note that this happens with probability 1 | Bi | because of Line 2. Note that at the time this line is executed, the values of i ∗ , -, C -i , B 0 , P i -1 . □

Combining the above claims we obtain

<!-- formula-not-decoded -->

where for the final equality we have used the fact that if R ecursion ( Ai , Bi ) is invoked (i.e., if P i = 1), then we must have e i = v . We next calculate P [ e i = e | σ, P i = 0]. We begin by observing that if P i = 0, then we must have e i = e i -1 which means

<!-- formula-not-decoded -->

Define the event σ ′ as σ ′ = n i ∗ , - ≥ i , C i = C -i , v ∈ B 0 o , i.e., σ ′ satisfies σ = { σ ′ , P i -1 = 0 } .

Claim A.9 . Conditioned on the event n i ∗ , - ≥ i , C i = C -i o , the events { v ∈ B 0 , P i = 0 } and e -i = e are independent.

Proof. In order to have v ∈ B 0 and P i = 0, the insertion algorithm should not ignore v and for all j such that we reach line 2, we need to flip the coin such that R ecursion is not invoked. The realization of e -i has no e ff ect on this. For B 0, the coint is always flipped with the same probability and for P i = 0, if the coins are flipped, they are always flipped with the same probability 1 | Bj | , and whether or not the coin is flipped (i.e., Line 2 is executed) depends only the value of B j and A j -1, neither of which are a ff ected by the realization of e -i . □

Given the above claim,

<!-- formula-not-decoded -->

Combined with Equation (4) and Equation (5), this gives us

<!-- formula-not-decoded -->

We therefore need to show that the above expression is the same as 1 [ e ∈ Bi ] | Bi | . This is easy to check however using a case by case analysis. If v &lt; Bi , then the first term vanishes and the second term becomes 1 [ e ∈ Bi ] | Bi | as required. If v ∈ Bi then the expression can be rewritten as

<!-- formula-not-decoded -->

Depending on whether e , v or e = v , either the first term or (respectively) the second term disappears and the other term equals 1 | Bi | . This proves Equation (5) which combined with Equation (4) finishes the proof.

## A.6.2 Deletion

For any i , let P i denote the random variable that is 1 if we invoke R ecursion ( Aj , Bj ) for some j ≤ i , and equals 0 otherwise. We first claim that

<!-- formula-not-decoded -->

As in the case of insertion, this is because e i is obtained by sampling uniformly at random from Bi and at the time of the sampling, the values in C i are realized and they do not change afterwards.

We next show that

<!-- formula-not-decoded -->

We start by evaluating the left hand side. We first claim that the event { i ∗ ≥ i , C i = Ci , P i = 0 } implies the event i ∗ , - ≥ i . This is because if i ∗ , -&lt; i , then P i = 0 means that R ecursion is not invoked by deletion, which in turn means that i ∗ = i ∗ , -, contradicting the assumption i ∗ ≥ i . Since i ∗ , - ≥ i , the value of C -i is defined. We condition on the value of C -i . Formally,

<!-- formula-not-decoded -->

where in the above expectation, the value of C -i is sampled from the distribution of C -i conditioned on { i ∗ ≥ i , C i = Ci , P i = 0 } . We note that all the values considered in the expectation satisfy P h i ∗ ≥ i , C i = Ci , P i = 0 , C -i = C -i i &gt; 0.

Consider some C -i such that P h i ∗ ≥ i , C i = Ci , P i = 0 , C -i = C -i i . We claim that rewrite the event n i ∗ ≥ i , C i = Ci , P i = 0 , C -i = C -i o is equivalent to n i ∗ , - ≥ i , C -i = C -i , P i = 0 o . It is clear that the first event implies the second event. As for the other direction, we note that the realization of C -i fully determines the realization of C i ; specifically, since P i = 0 we have B j = B -j \ { v } for all j ≤ i and e j = e -j for all j &lt; i . Therefore, P h i ∗ ≥ i , C i = Ci , P i = 0 | i ∗ , - ≥ i , C -i = C -i i is either 0 or 1. We know that it cannot be 0 however because by assumption, P h i ∗ ≥ i , C i = Ci , P i = 0 , C -i = C -i i &gt; 0. Therefore, it must be 1 which means C i = Ci . Finally, since P i = 0 we have i ∗ = i ∗ , - ≥ i . We note that P i = 0 itself implies that e i = e -i . Therefore, we can rewrite the probability inside the expectation in Equation (8) as

<!-- formula-not-decoded -->

We first claim that v , e -j for all j &lt; i . This is because if v = e -j for some j &lt; i , then P h P i = 0 | i ∗ , - ≥ i , C -i = C -i i = 0 because the deletion of v (which is e -j ) would invoke R ecursion . By assumption however, we only considered values of C -i for which P h C -i = C -i , P i = 0 i &gt; 0 in the expectation in Equation (8).

Given this, we have P i = 0 if and only if v , e -i . It follows that

<!-- formula-not-decoded -->

and

P GLYPH&lt;2&gt; P i = 0 | e -i = e , i ∗ , - ≥ i , C -i = C -i GLYPH&lt;3&gt; = P GLYPH&lt;2&gt; e -i , v | e -i = e , i ∗ , - ≥ i , C -i = C -i GLYPH&lt;3&gt; = 1 [ e , v ] . By induction hypothesis (Equation (1)),

<!-- formula-not-decoded -->

Putting it all together we obtain

<!-- formula-not-decoded -->

If v &lt; B -i , then we have Bi = B -i and the above expression turns into

<!-- formula-not-decoded -->

where for the equality we have used the fact that if e ∈ Bi we must have e , v since v &lt; Bi . If v ∈ Bi , then the above expression becomes

<!-- formula-not-decoded -->

where for the equality we have used the fact that e ∈ Bi { v } and e , v is equivalent to e ∈ Bi to rewrite the numerator and simplified ( | Bi | + 1 ) (1 -1 | Bi | + 1 to

<!-- formula-not-decoded -->

to rewrite the denominator. We have therefore proved Equation (7) which finishes the proof together with Equation (6).

## A.7 Query complexity

To analyze the query complexity of this algorithm, we first prove that checking if an element e can be added to level i requires O ( p · log( k )) oracle queries due to Lemma A.10. Consequently, the F iltering algorithm requires O ( | Bj | · p · log( k )) queries to operate on level j .

Therefore, running R ecursion on level i requires O ( i ∗ · | Bi | · p · log( k )) oracle calls to construct levels i , i + 1 , . . . , i ∗ . Given the uniform invariant, in each update, we reconstruct at level i with probability 1 / Bi . Thus, the expected number of oracle calls for reconstructing at level i is O ( i ∗ p log( k )).

Summing this up for all levels, the expected number of oracle calls is O (( i ∗ ) 2 p log( k )), which can be written as O ( ε -2 pk 4 log( k )) because of Lemma A.11.

Finally, as mentioned in Section B.2, each element is applied in O (log( k ε )) parallel runs, resulting in a total query complexity of O ( ε -3 pk 4 log 2 ( k )).

Lemma A.10 (Lemma 3.2 in Banihashem et al. [2024]) . Let I ∈ I j be an independent set of matroid M i ( V i , I i ) and e be an element such that I ∪ { e } &lt; I j . Define Xj : = { e ′ : I -e ′ + e ∈ I j } . Let w : I ∪ { e } → R ≥ 0 be an arbitrary weight function and define x j : = arg mine ′ ∈ Xjw ( e ′ ) . The element xj can be found using at most O (log( k )) oracle queries.

Lemma A.11. The number of levels i ∗ is at most O GLYPH&lt;16&gt; k 2 ε GLYPH&lt;17&gt; .

Proof. For each element e ∈ V , we have f ( e ) ≤ MAX. Therefore, because of submodularity, we have w ( e ) = f ( e | Ai ) ≤ f ( e ) ≤ MAX. Thus, we can conclude that w ( Ai ∗ ) = P e ∈ Ai ∗ w ( e ) ≤ k · MAX.

Moreover, if at level i , we add element ei , we have two cases:

- If ei does not replace any other elements, given the survivor invariant, w ( ei ) ≥ ε · MAX k , which means w ( Ai ) -w ( Ai -1 ) ≥ ε · MAX k .
- If the element removes a set of elements Ui ( ei ), its weight should be larger than (1 + q 1 + 1 p ) times the total weight of the removed elements in Ui ( ei ). This means removing elements of Ui ( ei ) and adding ei adds at least q 1 + 1 p · w ( Ui ( ei )) ≥ q 1 + 1 p · ε · MAX k ≥ ε · MAX k .

Since after each level, the value of w ( Ai ) increases by at least ε · MAX k and it cannot exceed k · MAX, the number of levels is at most

<!-- formula-not-decoded -->

□

Next, we analyze the query complexity of calling R ecursion on level i . Lemma A.12. The total cost of calling R ecursion ( i ) is at most O ( ε -1 pk 2 · | Bi | · log( k )) .

Proof. Based on Lemma A.10, inside the for loop of D ependency D etection requires O (log( k )) query calls to find a replacing element. Since each element is inside at most p matroids, we need to run inside that for loop p times, which makes the query complexity of D ependency D etection O ( p · log( k )).

Since F ind S waps calls D ependency D etection , and F iltering calls F ind S waps on every element of | Bi | , one step of R ecursion needs O ( | Bi | · p · log( k )) oracle calls.

As we proved in Lemma A.11, the depth of recursion in R ecursion is at most O GLYPH&lt;16&gt; k 2 ε GLYPH&lt;17&gt; . Therefore, the query complexity of R ecursion ( i ) is O GLYPH&lt;16&gt; ε -1 k 2 · | Bi | · p · log( k ) GLYPH&lt;17&gt; . □

Lemma A.13. For a specified value of MAX, each update operation in Algorithm U pdate O perations has query complexity at most O ( ε -2 pk 4 log( k )) .

Proof. Based on the uniform invariant, when we insert or delete an element, for each level i ≤ i ∗ , we call R ecursion ( i ) with probability 1 | Bi | · 1 [ e ∈ Bi ], which is at most 1 | Bi | . Using Lemma A.12, the

query complexity for calling R ecursion ( i ) is O ( ε -1 pk 2 · | Bi | · log( k )). Therefore, the expected number of queries caused by level i is bounded by

<!-- formula-not-decoded -->

As Lemma A.11 bounds the number of levels by i ∗ = O GLYPH&lt;16&gt; k 2 ε GLYPH&lt;17&gt; , we calculate the expected number of query calls for each update by summing the expected number of query calls at each level:

<!-- formula-not-decoded -->

□

To create an algorithm that operates independently of the value of MAX , we employ a strategy of guessing MAX up to a factor of 2 using parallel runs. Each element is inserted into only log( k /ε ) instances of the algorithm. Consequently, we achieve the total query complexity as claimed in Theorem 5.4.

## A.8 Approximation guarantee

In this section, we aim to establish the approximation guarantee of our algorithm. For this purpose, we make the assumption that elements are not randomly discarded at the beginning. Instead, we discard them with probability 1 -q when attempting to add them to our solution. While this adjustment does not a ff ect the solution itself, it simplifies the proof of the approximation factor. We denote the set of elements discarded in this approach as R , and we use M to represent the set of elements filtered in F ind S waps because their marginal gain was less than ε MAX k .

Before starting the proofs, we introduce some useful variables and notations. First, we define A ′ : = ∪ i ∗ i = 1 Ai as the union of solutions at all levels. Additionally, we introduce a crucial notation for our proofs:

Definition A.14. Let e ∈ V be an element and S ⊆ V be a set of elements. Let i ≥ 0 be the largest number such that e ∈ Bi . We define f ( e : S ) : = ∆ ( e | S \ Bi ), which calculates the marginal gain of adding e to a subset of S containing only elements that have been selected or filtered in levels preceding the level of selecting or filtering e . Note that w ( e ) = ∆ ( e | Ai -1 ) = ∆ ( e | Ai -1 \ Bi ) = f ( e : Ai -1 ), where w is the weight function defined in Algorithm 1. For sets T , S ⊆ V , we define f ( T : S ) : = P e ∈ T f ( e : S ).

Additionally, for every element e ∈ V , we define d ( e ) as the first level i such that e &lt; Ai ∪ Bi .

Now we represent several lemmas from Feldman et al. [2018] and utilize them in our proofs. In the following, we use the constant c , defined as c : = q 1 + 1 p .

Lemma A.15 (Lemma 7 in Feldman et al. [2018]) . f ( A ′ \ Ai ∗ : Ai ∗ ) ≤ f ( Ai ∗ ) c .

Lemma A.17 (Proposition 10 in Feldman et al. [2018]) . For every independent set S ⊆ V \ ( M ∪ R ) , there exists a mapping ϕ S from elements of S to multi-subsets of A ′ such that:

Lemma A.16 (Corollary 8 in Feldman et al. [2018]) . f ( A ′ ) ≤ c + 1 c · f ( Ai ∗ ) .

- Each element e ∈ Ai ∗ appears at most p times in the multi-sets of { ϕ S ( e ) | e ∈ S } .
- Each element e ∈ A ′ \ Ai ∗ appears at most p -1 times in the multi-sets of { ϕ S ( e ) | e ∈ S } .
- Each element e ∈ S \ A ′ satisfies w ( e ) ≤ (1 + c ) · P e ′ ∈ ϕ S ( e ) f ( e ′ : Ad ( e ′ ) -1 ) .
- Each element e ∈ S ∩ A ′ satisfies w ( e ) ≤ f ( e ′ : Ad ( e ′ ) -1 ) for every e ′ ∈ ϕ S ( e ) , and the multi-set ϕ S ( e ) contains exactly p elements (including repetitions).

Theorem A.18. E GLYPH&lt;2&gt; f ( A ′ ∪ OPT ) GLYPH&lt;3&gt; ≤ (1 + c ) 2 p c · E GLYPH&lt;2&gt; f ( Ai ∗ ) GLYPH&lt;3&gt; + ε f ( OPT ) .

Proof. First, we break A ′ ∪ OPT into a union of four sets:

<!-- formula-not-decoded -->

If we set S = OPT \ ( R ∪ M ), we can use Lemma A.17 to bound the second term of the last expression:

<!-- formula-not-decoded -->

Moreover,

<!-- formula-not-decoded -->

Combining all together, we get:

<!-- formula-not-decoded -->

We can bound the last term given that for each element e ∈ M , we have w ( e ) ≤ ε MAX k ≤ ε f ( OPT ) k and | OPT | ≤ k , we can conclude that

<!-- formula-not-decoded -->

Now, we want to show that the expectations of the second term and the third term are equal. Consider an arbitrary element e ∈ OPT . Based on our assumption discussed at the beginning of this section about discarding elements randomly, if they were going to be added to the solution instead of being discarded at the beginning, each time an element is considered for addition to the solution, with probability q it will be added to A ′ and with probability 1 -q it will be discarded, which means it will be added to R . Therefore, we have

<!-- formula-not-decoded -->

where 1 [ X ] is set to one if X holds and is set to zero otherwise. Rearranging the above equality, we have

<!-- formula-not-decoded -->

where the last equality comes from the fact that the equality 1 q -1 = (1 + c ) p holds for q = ( p + p p 2 + p + 1) -1 and c = q 1 + 1 p .

By combining the above, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma A.19 (Lemma 2.2 in Buchbinder et al. [2014]) . Let g : 2 V → R + be a submodular function. Let S be a random subset of V such that each element appears in S with probability at most q, not necessarily independently. We have E [ g ( S )] ≥ (1 -q ) g ( ∅ ) .

Proof of Theorem 5.5: Let's define g ( S ) = f ( S ∪ OPT ). Since g is a non-negative submodular function, and each element will be discarded with probability 1 -q , meaning it can appear in A ′ with probability at most q , we can use Lemma A.19 to conclude that

<!-- formula-not-decoded -->

where the last equality holds because the equation 1 -q = 1 c for q = ( p + p p 2 + p + 1) -1 and c = q 1 + 1 p . Combining this with Theorem A.18, we get

<!-- formula-not-decoded -->

or rearranging it as

<!-- formula-not-decoded -->

Proof of Proposition 1.3: The proof of this result is analogous to the proof of the main result 1.1, with the key di ff erence that its approximation guarantee follows directly from Theorem A.18, which relies solely on the equation 1 q -1 = (1 + c ) p , unlike the proof of Theorem 5.5, which also uses the equation 1 -q = 1 c . This flexibility allows the approximation ratio to be optimized by adjusting the parameters and setting c = 1 and q = (2 p + 1) -1 .

□

## B Removing known MAX assumption

In this section, we show how to remove the assumption that we know a parameter MAX satisfying max v ∈V f ( v ) ≤ MAX ≤ 2max v ∈V f ( v ). While the techniques here are standard in the literature for both dynamic and streaming Kazemi et al. [2019], Lattanzi et al. [2020], Banihashem et al. [2024] submodular optimization, we here provide the approach and the proof for completeness.

## B.1 Overview of the reduction

Let A denote a dynamic algorithm that operates without this knowledge, i.e., the algorithm presented in the paper. We will maintain parallel runs of the algorithm A in memory which try to guess the value MAX. Formally, for any integer i , possibly negative, let A i denote a run of the algorithm A with the parameter MAX set to 2 i . Note that we will not actually initialize these runs for all i and will only do so for some values based on the inserted elements.

When an element v is inserted, define I v as

<!-- formula-not-decoded -->

For each i ∈ I v , if the instance A i is not initialized, then we initialize it. Next, for all i ∈ I v , we insert v in A i .

When an element is deleted, we delete it from all A i for i ∈ I v .

We always output the maximum answer among all parallel runs that are initialized. We note that this does not require any queries.

## B.2 Analysis of query complexity

Since | I v | ≤ log( k /ε ), and initializing A i with an empty ground set does not require any queries, the expected query complexity of the algorithm is O (log( k /ε )) times the expected query complexity of A i .

## B.3 Analysis of approximation guarantee

We first consider an alternative version of the algorithm in which we set I v to be

<!-- formula-not-decoded -->

We claim that the output of this version is the same as the version described in Section B.1. Note that this new algorithm is used for the purpose of analysis only and is not actually used for implementation.

This is because if f ( v ) &lt; ε k 2 i , then the algorithm simply discards it given the check in F ind S waps . For the new version of the algorithm, set i ′ such that 2 i ′ -1 ≤ f ( v ) ≤ 2 i . Given the analysis in Section A, the output of A i ′ provides the desired approximation guarantee. Since we are always taking the output of the A i with the best output, our output provides the desired approximation guarantee as well.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly and accurately state the claims.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and / or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The assumptions made in the paper as well as the results are clearly stated and therefore the limitations are clear.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational e ffi ciency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: A complete proof of theoretical results is provided.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it a ff ects the main claims and / or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA]

Justification: The paper makes a significant theoretical contribution and does not include experimental results.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and / or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might su ffi ce, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with su ffi cient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: The paper makes a significant theoretical contribution and does not include experimental results.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting / details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Justification: The paper makes a significant theoretical contribution and does not include experimental results.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The paper makes a significant theoretical contribution and does not include experimental results.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train / test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide su ffi cient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justification: The paper makes a significant theoretical contribution and does not include experimental results.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Conforms to the guidelines

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The paper is theoretical in nature and does not have societal impact.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the e ffi ciency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.

- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing e ff ective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith e ff ort.

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

Justification: No new assets are released.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset / code / model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not use crowdsourcing.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval / review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: No LLMs were used in conducting the research or generating the results.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.