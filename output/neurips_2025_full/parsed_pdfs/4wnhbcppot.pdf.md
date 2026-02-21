## Multimodal Bandits: Regret Lower Bounds and Optimal Algorithms

## William Réveillard

Division of Decision and Control Systems KTH Royal Institute of Technology 11428 Stockholm, Sweden wilrev@kth.se

## Richard Combes

Laboratoire des signaux et systèmes Université Paris-Saclay, CNRS, CentraleSupélec 91190 Gif-sur-Yvette, France richard.combes@centralesupelec.fr

## Abstract

We consider a stochastic multi-armed bandit problem with i.i.d. rewards where the expected reward function is multimodal with at most m modes. We propose the first known computationally tractable algorithm for computing the solution to the Graves-Lai optimization problem, which in turn enables the implementation of asymptotically optimal algorithms for this bandit problem.

## 1 Introduction

We consider a stochastic multi-armed bandit with K ≥ 1 arms. At time t ∈ [ T ] , with T ≥ 1 , based on her previous observations, a learner selects an arm k ( t ) in [ K ] = { 1 , . . . , K } , and subsequently observes a random reward X k ( t ) ,t . The successive rewards ( X k,t ) t ∈ N obtained when sampling a given arm k ∈ [ K ] are drawn i.i.d. from a family of distributions ν k ( µ k ) parameterized by their expectation µ k . The vector of mean rewards 1 µ = ( µ k ) k ∈ [ K ] is unknown to the learner. The learner's goal is to select arms in order to discover the optimal arm k ⋆ ( µ ) = arg max k ∈ [ K ] µ k . More precisely, the learner aims at minimizing the regret

<!-- formula-not-decoded -->

which is the expected difference between the total reward obtained by an oracle who knows µ in advance and always chooses the arm with the largest mean reward, and the total reward obtained by the learner. If we were to assume that µ is arbitrary, then the problem at hand would reduce to the classical stochastic multi-armed bandit. Here we consider a structured stochastic multi-armed bandit, where the structure is encoded by a graph G .

We consider a graph G = ( V, E ) whose vertices are the arms V = [ K ] , and we say that arms k and ℓ are neighbors if and only if ( k, ℓ ) ∈ E . Arm k is a mode of µ with respect to G if and only if it has a strictly greater reward than all of its neighbors: µ k &gt; max ℓ :( k,ℓ ) ∈ E µ ℓ and we say that µ is m -modal with respect to G if it has m modes with respect to G .

In this work, we assume that µ is at most m -modal with respect to a graph G known to the learner. We emphasize that, while G is known to the learner, µ is not, so that finding the optimal arm requires sampling from suboptimal arms repeatedly. We note that for m = 1 , a 1 -modal vector is simply a unimodal vector, thus the problem reduces to unimodal bandits [8]. If G is a line graph, a mode is an arm whose reward is greater than that of its left and right neighbors. We will assume that G is a tree.

1 With a slight abuse of notation, we also refer to µ as the reward function of the bandit problem.

## 2 Related work and contribution

In the absence of a multimodal structure, our problem reduces to the classical multi-armed bandit studied by [13], and several asymptotically optimal algorithms are known for this problem, such as KL-UCB, proposed by [6], DMED, proposed by [9] and Thompson Sampling, as analyzed by [10].

When adding a multimodal structure with m = 1 mode, we obtain the unimodal bandit problem, originally studied by [8] and revisited by [25]. Several asymptotically optimal algorithms have been proposed for this problem including the KL-UCB style algorithm of [2], the Thompson Sampling style algorithm of [21] and the DMED style algorithm of [19]. A common feature of these algorithms is that they are all based on local search , where only arms that are neighbors of the optimal arm are selected a logarithmic amount of time. Local search is necessary for asymptotic optimality in unimodal bandits because the strategy that minimizes the Graves-Lai bound, as introduced by [7], is local.

Multimodal bandits with m ≥ 1 modes generalize unimodal bandits, and have been considered in [17] and [18]. These works explored local search strategies, which, as we shall see, are not necessarily asymptotically optimal. The main motivation behind multimodal bandits is the fact that many objective functions encountered in applications are not convex nor unimodal, such as the empirical risk of deep neural networks. Methods for optimizing or sampling (which are closely related) multimodal functions include bayesian methods studied by [3] and MCMC methods considered by [14]. Multimodal functions have also been considered in an active learning setting by [16]. An interesting application of bandit problems with multimodal rewards is pricing ([15, 24]).

For structured bandits, the Graves-Lai bound is an information-theoretic regret lower bound, stated as an optimization problem. Its optimal solution identifies strategies for optimal exploration , i.e., the rate at which suboptimal arms must be selected to ensure minimal regret. Several asymptotically optimal algorithms with regret matching the Graves-Lai bound have been proposed by [7], [1], [5], [22]. The main advantage of these algorithms is their universality in the sense that they apply to all structured bandits.

While the above algorithms are indeed universally asymptotically optimal, they often pose a tremendous computational challenge, because they must solve the Graves-Lai optimization problem. In some simple structures, the Graves-Lai optimization problem admits a closed-form solution, for instance: classical bandits ([13]), unimodal bandits ([2]), dueling bandits ([11]) to name a few. For some other structures such as combinatorial bandits ([4]), the Graves-Lai optimization problem can be solved with efficient iterative algorithms. For multimodal bandits, solving the Graves-Lai optimization problem is challenging, as we shall see, primarily due to the highly non-convex nature of its constraint set.

Our contribution. In this work, we provide the first known computationally tractable algorithm to solve the Graves-Lai optimization problem for multimodal bandits. The algorithm is involved and uses a combination of discretization, dynamic programming and projected subgradient descent in order to navigate the intricate structure of the constraint set. The algorithm applies to a wide variety of reward distributions, and any tree graph. We further demonstrate that local search strategies are suboptimal, which means that solving the Graves-Lai problem is unavoidable for optimality. The code for the proposed algorithms, which are involved, is publicly available at https://github. com/wilrev/MultimodalBandits .

## 3 Asymptotically optimal algorithms for multimodal bandits

In this section, we state our problem assumptions, present the Graves-Lai lower bound specialized to the case of multimodal bandits, and recall how solving the Graves-Lai optimization problem enables one to design asymptotically optimal algorithms, i.e., with regret matching the Graves-Lai lower bound.

Notation. To ease exposition, we use the following notation. All vectors are represented with bold symbols. We denote by e ( k ) ∈ R K the k -th canonical basis vector of R K , and by ∥ x ∥ p = ( ∑ k ∈ [ K ] | x | p ) 1 /p the L p norm. The closure of a set S ⊂ R K is denoted by S. We denote µ ⋆ = max k ∈ [ K ] µ k and µ ⋆ = min k ∈ [ K ] µ k the maximum and minimum mean reward, respectively. We

assume that the optimal arm k ⋆ ( µ ) = arg max k ∈ [ K ] µ k is unique. We define the vector of gaps ∆ = ( µ ⋆ -µ k ) k ∈ [ K ] and the minimal gap ∆ min = min k ∈ [ K ]:∆ k &gt; 0 ∆ k .

For a given tree G = ( V, E ) with V = [ K ] , we denote by diam( G ) its diameter and deg( G ) its maximal degree. M ( µ ) is the set of modes of µ with respect to G , so that µ is m -modal if |M ( µ ) | = m , and N ( µ ) is the set of modes and neighbors of modes of µ . We define F ≤ m (resp. F m ) the set of reward functions of R K with at most m modes (resp. exactly m modes) with respect to G . We assume that µ ∈ F ≤ m for some m&gt; 1 , and that m is known to the learner.

Finally, we sometimes consider the tree G to be directed. Then, for a given arm k ∈ [ K ] , we denote by C ( k ) the set of children of k , D ( k ) the set of descendants of k , p ( k ) the parent of k and p 2 ( k ) the grandparent of k (i.e., the parent of p ( k ) ).

Assumptions on reward distributions. The rewards from arm k ∈ [ K ] form an i.i.d. sample from distribution ν k ( µ k ) . Let d k ( µ k , λ k ) = D ( ν k ( µ k ) ∥ ν k ( λ k )) denote the relative entropy between the rewards distributions of arm k under parameters µ k and λ k . For µ , λ ∈ R K we use the vectorized notation d ( µ , λ ) = ( d k ( µ k , λ k )) k ∈ [ K ] . We make two assumptions regarding these relative entropies.

Assumption 1. For all µ ∈ R K and k ∈ [ K ] , λ k ↦→ d k ( µ k , λ k ) is strictly decreasing for λ k &lt; µ k and strictly increasing for λ k &gt; µ k .

Assumption 2. For all k ∈ [ K ] , µ ∈ R K and λ , λ ′ in [ µ ⋆ , µ ⋆ ] K we have ∥ d ( µ , λ ) -d ( µ , λ ′ ) ∥ 1 ≤ A ( µ ) ∥ λ -λ ′ ∥ 1 where A ( µ ) ≥ 0 can be understood as the Lipschitz constant of the relative entropy when its first argument is held constant.

The first assumption boils down to the relative entropy d k being unimodal, and its unique mode being the minimizer λ k = µ k . The second assumption is satisfied whenever the divergence is continuously differentiable over [ µ ⋆ , µ ⋆ ] K . For example, when ν k ( µ k ) = N (0 , 1) , it holds with A ( µ ) = µ ⋆ -µ ⋆ .

Regret lower bound. Proposition 1 states that the asymptotic regret of any uniformly good algorithm (i.e., whose regret scales subpolynomially on all problem instances) must be lower bounded by the value of the Graves-Lai optimization problem multiplied by ln T . We denote this optimization problem by P GL .

Proposition 1. Consider an algorithm such that lim T →∞ R ( µ ,T ) T α = 0 for all α &gt; 0 and all µ ∈ F ≤ m . Then its asymptotic regret is lower bounded as lim inf T →∞ R ( µ ,T ) ln T ≥ C ( m, µ ) for all µ ∈ F ≤ m where C ( m, µ ) is the value of:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

P GL is a semi-infinite linear program. There are infinitely many constraints, and these constraints are described by B ( m, µ ) , which is the set of confusing parameters . λ ∈ B ( m, µ ) is confusing to the learner because it cannot be distinguished from µ by selecting the optimal arm k ⋆ ( µ ) , thereby forcing the learner to explore suboptimal arms. In particular, for a fixed η ≥ 0 , we call most confusing parameter the minimizer λ ⋆ of λ ↦→ η ⊤ d ( µ , λ ) over B ( m, µ ) . The set B ( m, µ ) has a complicated structure, which is why solving P GL is non-trivial. Proposition 1 is a direct consequence of the general bound of Theorem 1 in [7] or alternatively the simpler lower bound of Theorem 1 in [1].

̸

Asymptotically optimal algorithms. We recall that, if P GL can be solved, then there exists a wide variety of algorithms that are asymptotically optimal, such as those presented by [7], [1], [5] and [22]. All these algorithms attempt to select arm k = k ⋆ ( µ ) a number of times η ⋆ k ln T + o (ln T ) when T →∞ , where η ⋆ is a solution to P GL . The only requirement is that one is able to solve P GL several times in order to decide which arm to explore.

Theorem 1 (Theorem 2 in [1]) . Consider Gaussian rewards with variance one and assume that for any µ ∈ F ≤ m , a solution to the Graves-Lai problem can be computed. Then, the OSSB algorithm

<!-- formula-not-decoded -->

For completeness, the pseudo-code of OSSB is provided in Appendix A.1. We now focus on how to solve P GL for multimodal bandit problems.

## 4 A computationally tractable algorithm to solve the Graves-Lai problem

In this section, we propose an algorithm to solve the Graves-Lai optimization problem in a computationally tractable manner, which constitutes our main contribution. The algorithm is rather intricate, and we go through its derivation step by step. The complete approach is summarized in Figure 1. For clarity, detailed proofs are provided in Appendix C.

Figure 1: Summary of the procedure to solve P GL .

<!-- image -->

## 4.1 Reducing the constraint of P GL to tractable subproblems

On the difficulty of computing the constraint. In the unimodal case where m = 1 , solving P GL can be done in closed form, however for m&gt; 1 this is much less straightforward. The main difficulty is to compute the value of the constraint inf λ ∈B ( m, µ ) η ⊤ d ( µ , λ ) . While λ ↦→ η ⊤ d ( µ , λ ) is usually a convex function, minimizing it over B ( m, µ ) is difficult, due to the particular structure of the set B ( m, µ ) . Indeed, B ( m, µ ) is not convex, nor is it connected. In Proposition 2 we show that, if one wanted to express B ( m, µ ) as a union of U ( K,m ) convex sets (so that we could minimize η ⊤ d ( µ , λ ) over each set using convex programming), then one would require U ( K,m ) to be exponentially large in K . For example, if G is a line graph with K = 100 nodes, and we consider multimodal functions with m = 5 modes, then the number of convex components U ( K,m ) must be greater than 10 5 .

Proposition 2. Assume that B ( m, µ ) can be written as a union of U ( K,m ) convex sets. Then for any m&gt; 1 , we have:

<!-- formula-not-decoded -->

hence U ( K,m ) grows exponentially with m .

In contrast to the unimodal case ( m = 1 ), where the most confusing parameter λ ⋆ is obtained by perturbing a single neighbor of the optimal arm k ⋆ ( µ ) (as shown by [2]), the multimodal setting ( m &gt; 1 ) introduces more complex possibilities. While one might expect the most confusing parameter to be such that λ k = µ ⋆ for a single k ∈ N ( µ ) \ k ⋆ ( µ ) , and equal to µ elsewhere, this intuition fails in general. Depending on the value of η , it may be more confusing to set λ k = µ ⋆ for k / ∈ N ( µ ) , and to ensure that λ ∈ F ≤ m , set λ k ′ = λ ℓ for some k ′ ∈ M ( µ ) \ k ⋆ ( µ ) and ℓ such that ( k ′ , ℓ ) ∈ E. Figure 2 provides a concrete illustration of

<!-- image -->

Arm Index

Figure 2: 2-modal example.

this phenomenon on a line graph, with λ ⋆ = arg min λ ∈B (2 , µ ) η ⊤ d ( µ , λ ) for µ = (1 , 2 , 4 , 2 , 3) , η = (0 . 01 , 0 . 25 , 1 , 0 . 25 , 1) , and Gaussian rewards with variance one.

̸

Restricting the constraint and solution spaces. Wefirst show some elementary properties of the solution, which will allow us to restrict both the spaces where η and λ lie. First, we decompose the constraint set as B ( m, µ ) = ∪ k = k ⋆ ( µ ) B k ( m, µ ) with B k ( m, µ ) = { λ ∈ F ≤ m , λ k ⋆ ( µ ) = µ ⋆ , k ⋆ ( λ ) = k } . Clearly, minimizing η ⊤ d ( µ , λ ) over λ ∈ B ( m, µ ) amounts to minimizing η ⊤ d ( µ , λ ) over λ ∈ B k ( m, µ ) for each k = k ⋆ ( µ ) . Proposition 3 states that this minimization problem is straightforward when µ has strictly less than m modes or when k is in the neighborhood of the modes of µ .

̸

̸

Proposition 3. Let η ≥ 0 and k = k ⋆ ( µ ) . If k ∈ N ( µ ) or |M ( µ ) | &lt; m

<!-- formula-not-decoded -->

which is attained for λ = µ +( µ ⋆ -µ k ) e ( k ) ∈ B k ( m, µ ) .

We now focus on the case |M ( µ ) | = m and k / ∈ N ( µ ) . Proposition 4 shows that the constraint λ ∈ B k ( m, µ ) is equivalent to a constraint on a compact set whose elements have entries comprised between the minimum and maximum of µ , and that have the same value µ ⋆ at k and k ⋆ ( µ ) .

Proposition 4. Let η ≥ 0 , k / ∈ N ( µ ) and B ′ k ( m, µ ) = { λ ∈ [ µ ⋆ , µ ⋆ ] K ∩F ≤ m , λ k = λ k ⋆ ( µ ) = µ ⋆ } . Then

<!-- formula-not-decoded -->

Proposition 5 shows that, in order to compute a solution to P GL , we can restrict our attention to a compact region, and that the entries of η cannot be larger than B ( µ ) . In turn, B ( µ ) may be interpreted as the regret predicted by the Lai-Robbins bound in absence of a multimodal structure, divided by the minimal gap.

Proposition 5. There is a solution η ⋆ of P GL such that η ⋆ ∈ [0 , B ( µ )] K where B ( µ ) = 1 ∆ min ∑ k :∆ k &gt; 0 ∆ k d k ( µ k ,µ ⋆ ) .

Location of modes in subproblems. We have shown in the previous section that the computation of the constraint of P GL can be reduced to solving the following subproblem for all η ≥ 0 and for each k ∈ [ K ] \ N ( µ ) :

<!-- formula-not-decoded -->

We now present the most important structural result about the solution to P GL ( k ) , which pertains to the location of its modes. Proposition 6 states that the modes of the solution to P GL ( k ) all lie in the set of modes of µ , apart from k , which must of course be a mode since it is a maximizer. This is in fact the reason why one is able to compute the solution to P GL ( k ) . While this will be made clearer by exhibiting an efficient algorithm to compute the solution, it is understood searching over λ is much easier when the location of its modes is known.

Proposition 6. Consider λ ⋆ the solution to P GL ( k ) . Then M ( λ ⋆ ) ⊂ M ( µ ) ∪ { k } .

̸

If |M ( µ ) | = m , we have |M ( µ ) ∪ { k }| = m +1 , which implies that there must exist a mode k ′ of µ that is not a mode of λ ⋆ , and all of the modes of λ ⋆ apart from k are modes of µ . Additionally, we can assume that k ′ = k ⋆ ( µ ) . Indeed, if k ′ = k ⋆ ( µ ) , the constraint λ k ⋆ ( µ ) = µ ⋆ would yield λ ℓ ≥ µ ⋆ for some neighbor ℓ of k ⋆ ( µ ) . This cannot improve upon the solution given by Proposition 3. This means that we can restrict our attention to the sets B k,k ′ ( m, µ ) for k ′ ∈ M ( µ ) \ { k ⋆ ( µ ) } with

<!-- formula-not-decoded -->

which is the set of vectors whose modes lie in M ( µ ) ∪ { k } \ { k ′ } and attain their maximum at k ⋆ ( µ ) and k , and that have the same value as µ at k ⋆ ( µ ) . We must solve the subproblems, for k ∈ [ K ] \ N ( µ ) , k ′ ∈ M ( µ ) \ { k ⋆ ( µ ) } :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Discretizing the subproblems. The last step before we can solve P GL ( k, k ′ ) is to discretize the space in which λ lies, which will allow us to design a discrete search procedure over λ . Proposition 7 states that discretizing each entry of λ ∈ B k,k ′ ( m, µ ) with a grid of n points D ( n, µ ) , incurs a small approximation error that can be controlled, and vanishes at a rate inversely proportional to n . It is noted that this result is non trivial in the sense that there exists sets of large volume whose intersection with some grid can be empty, and holds because of the particular structure of B k,k ′ ( m, µ ) . In essence, we can round λ to ensure both a small rounding error while leaving the set of modes of λ unchanged.

Proposition 7. Consider D ( n, µ ) the following uniform discretization of [ µ ⋆ , µ ⋆ ] with n discretization points:

<!-- formula-not-decoded -->

Then there exists ˜ λ ∈ B k,k ′ ( m, µ ) ∩ D ( n, µ ) K such that for any η ∈ [0 , B ( µ )] K :

<!-- formula-not-decoded -->

with C ( µ ) = diam( G )( µ ⋆ -µ ⋆ ) A ( µ ) B ( µ ) K.

## 4.2 Computing the constraint sets via dynamic programming

We now explain how to efficiently solve the discretized version of P GL ( k, k ′ ) , namely

<!-- formula-not-decoded -->

for ˜ B k,k ′ ( m, µ ) = B k,k ′ ( m, µ ) ∩ D ( n, µ ) K . We use a dynamic programming procedure which necessitates viewing G as a directed tree G k , obtained by performing depth-first search on the undirected tree G starting at node k (which is consequently the root of G k ). We recall the notations C ( ℓ ) , D ( ℓ ) , p ( ℓ ) and p 2 ( ℓ ) to denote the children, descendants, parent and grandparent of a node ℓ in G k . The high-level idea of the procedure is to compute the value of ˜ P GL ( k, k ′ ) recursively with a formula that relates the minimal obtainable value of ∑ j ∈D ( ℓ ) ∪{ ℓ } η j d j ( µ j , λ j ) to that of ∑ j ∈D ( ℓ ) η j d j ( µ j , λ j ) for any node ℓ . Note that when ℓ = k , the former is equal to the value of ˜ P GL ( k, k ′ ) , and when ℓ is a leaf of G k , the latter is equal to 0 . This recursion formula heavily relies on the fact that all modes of the solution to ˜ P GL ( k, k ′ ) are in M ( µ ) ∪ { k } \ { k ′ } .

̸

We now introduce some important quantities for our dynamic programming approach. For a node ℓ = k in G k , we define f ℓ ( z, u ) as the minimal obtainable value of

<!-- formula-not-decoded -->

subject to the constraints λ ∈ ˜ B k,k ′ ( m, µ ) , λ ℓ = z and λ ℓ &gt; λ p ( ℓ ) if u = 1 (resp. λ ℓ ≤ λ p ( ℓ ) if u = -1 ). To simplify the notations further, we introduce the following auxiliary functions 2 :

<!-- formula-not-decoded -->

which represent the minimal obtainable value of ∑ j ∈D ( ℓ ) ∪{ ℓ } η j d j ( µ j , λ j ) for λ ∈ ˜ B k,k ′ ( m, µ ) when λ ℓ &gt; λ p ( ℓ ) = z, λ ℓ ≤ λ p ( ℓ ) = z and λ p ( ℓ ) = z , respectively. Finally, to ensure that the constraint λ k ⋆ ( µ ) = µ ⋆ is satisfied during the dynamic programming procedure, we set 3 η k ⋆ ( µ ) = + ∞ , and we use the convention that η k ⋆ ( µ ) d k ⋆ ( µ ) ( µ ⋆ , z ) equals 0 if z = µ ⋆ and + ∞ otherwise.

Proposition 8. The functions f ℓ ( z, u ) for ℓ ∈ [ K ] , z ∈ D ( n, µ ) and u ∈ {-1 , +1 } obey the following recursion:

If ℓ ∈ M ( µ ) ∪{ k }\{ k ′ } : f ℓ ( z, u ) = η ℓ d ℓ ( µ ℓ , z ) + ∑ j ∈C ( ℓ ) f ⋄ j ( z ) , and if ℓ ̸∈ M ( µ ) ∪{ k }\{ k ′ } :

̸

<!-- formula-not-decoded -->

2 The minima are taken with the implicit constraint w ∈ D ( n, µ ) .

3 Recall that the value of η k ⋆ ( µ ) has no impact on the solution to ˜ P GL ( k, k ′ ) .

where g v ( z ) = min { f ⋆ v ( z, +1) , f v ( z, -1) } -f ⋄ v ( z ) , and the value of ˜ P GL ( k, k ′ ) equals f k ( µ ⋆ , u ) for any u ∈ {-1 , +1 } .

Since the discretized search space D ( n, µ ) is finite, we can straightforwardly compute the values of f ℓ ( z, u ) , f ⋆ ℓ ( z, u ) and f ⋄ ℓ ( z ) for each ℓ ∈ [ K ] , z ∈ D ( n, µ ) and u ∈ {-1 , +1 } with the dynamic programming equations of Proposition 8. The solution λ ⋆ of ˜ P GL ( k, k ′ ) can then be obtained recursively as in Corollary 1, in which the condition ℓ = arg min v ∈C ( p ( ℓ )) g v ( λ ⋆ p ( ℓ ) ) can be understood as ℓ being the children of p ( ℓ ) that induces the smallest cost when constrained by the value of p ( ℓ ) .

̸

Corollary 1. The solution λ ⋆ of ˜ P GL ( k, k ′ ) is such that λ ⋆ k = µ ⋆ and for ℓ = k :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, computing the solution to ˜ P GL ( k, k ′ ) using the procedure of Proposition 8 and Corollary 1 can be done in time and memory O ( nK ) .

Overall time complexity. This procedure allows us to solve the subproblems ˜ P GL ( k, k ′ ) for all k / ∈ N ( µ ) and k ′ ∈ M ( µ ) \{ k ⋆ ( µ ) } . By comparing these solutions with the trivial parameters from Proposition 3, we can find the most confusing parameter in B ( m, µ ) ∩ D ( n, µ ) K for any sampling rate η in time O ( K 2 mn ) . In practice, these subproblems are independent and can be solved in parallel. In Appendix E, we describe a more involved dynamic program that runs in time O ( Kn ) without requiring parallelism.

Illustration of the dynamic programming approach. We now illustrate the computation of the most confusing parameter in B ( m, µ ) ∩ D ( n, µ ) K with the line graph example of Figure 2. There, the divergence is d k ( λ k , µ k ) = 1 2 ( λ k -µ k ) 2 , the optimal arm is k ⋆ ( µ ) = 3 , the modes are M ( µ ) = { 3 , 5 } , and their neighborhood is N ( µ ) = { 2 , 3 , 4 , 5 } . If k = k ⋆ ( λ ) is chosen in N ( µ ) , applying Proposition 3 yields

<!-- formula-not-decoded -->

Otherwise, we must have k = 1 , and the only choice for the mode of µ to be removed is k ′ = 5 .

We can then solve ˜ P GL ( k, k ′ ) for ( k, k ′ ) = (1 , 5) by applying Proposition 8 as follows. We first form the directed tree G 1 as a line graph, rooted at 1 , with leaf node 5 . For each node ℓ and each grid value z ∈ D ( n, µ ) the program computes and stores in memory the values

<!-- formula-not-decoded -->

together with the auxiliary minima f ⋆ ℓ ( z, · ) and f ⋄ ℓ ( z ) , as defined in Proposition 8. The leaf entry is obtained directly from the divergence:

<!-- formula-not-decoded -->

and f ⋆ 5 ( z, · ) , f ⋄ 5 ( z ) are then computed by minimizing over the grid D ( n, µ ) . For an internal node ℓ = 5 , the recursion in Proposition 8 is applied: each entry f ℓ ( z, · ) is derived from the already computed child cost values as prescribed in the proposition. Finally, the value of the discretized subproblem ˜ P GL ( k, k ′ ) is read off at the root as f 1 ( µ ⋆ , +1) , where µ ⋆ = 4 in the present example. The minimizer is finally recovered by backtracking, as described in Corollary 1. In the limit n →∞ , this minimizer approaches λ ⋆ = (4 , 2 , 4 , 2 . 8 , 2 . 8) . This non-trivial confusing parameter turns out to be the global minimizer of P GL as its value approaches 0 . 145 &lt; 0 . 5 .

̸

## 4.3 Solving P GL via penalized subgradient descent

Weare now capable of computing a minimizer of η ⊤ d ( µ , λ ) over B ( m, µ ) ∩ D ( n, µ ) K , the constraint in the original problem P GL , with discretization. The last step to close the loop is to use an iterative procedure to derive an approximate solution to P GL . The simplest way to understand this procedure is to view it as a projected subgradient descent for the convex function

<!-- formula-not-decoded -->

which can be interpreted as the objective of P GL with a discretized constraint space and where the hard constraints have been replaced by a penalty similar to the hinge loss function, and where γ controls the magnitude of the penalty. The projection step is used to enforce the constraint η ≥ 0 . We show in Proposition 9 that the minimizer of h ( η ) subject to the constraint η ≥ 0 is an approximate solution to P GL .

Proposition 9. Consider a step size δ 2 = K B ( µ ) 2 t E ( µ ) 2 where t is the number of iterations, E ( µ ) = ∥ ∆ ∥ 2 + γK 3 / 2 A ( µ )( µ ⋆ -µ ⋆ ) , a penalty γ = 2max k, ∆ k &gt; 0 ∆ k d ( µ k ,µ ⋆ ) and an iterative procedure η (1) = 0 , and for s &lt; t :

<!-- formula-not-decoded -->

✶ where λ ( s ) ∈ arg min λ ∈B ( m, µ ) ∩ D ( n, µ ) K η ( s ) ⊤ d ( µ , λ ) and Π[ x ] = (max( x k , 0)) k ∈ [ K ] is the projection on the positive orthant. Define ¯ η ( t ) = (1 /t ) ∑ t s =1 η ( s ) the average iterate and a scaled version ˜ η ( t ) = ¯ η ( t ) ( 1 -C ( µ ) n -2 F ( µ ) γ √ t ) -1 for F ( µ ) = √ K B ( µ ) E ( µ ) . Then ˜ η ( t ) is a feasible solution to P GL with value at most:

<!-- formula-not-decoded -->

Putting it all together. We end this section by stating our main result, which is a direct consequence of the previous propositions, and propose a computationally tractable algorithm in order to compute an approximate solution to P GL . With the more intricate dynamic programming scheme presented in Appendix E, its time complexity can be improved to O ( Knt ) .

Theorem 2. Consider the algorithm which outputs ˜ η ( t ) after t iterations of the scheme described in Proposition 9 with n discretization points. At each step s ≤ t , λ ( s ) is computed by solving ˜ P GL ( k, k ′ ) for all k / ∈ N ( µ ) and all k ′ ∈ M ( µ ) \{ k ⋆ ( µ ) } using the dynamic programming scheme of Proposition 8. This algorithm runs in time O ( K 2 mnt ) and space O ( Knt ) and yields ˜ η ( t ) , a feasible solution to P GL with value at most:

<!-- formula-not-decoded -->

The parameters n and t can be chosen by the learner. Since the time complexity grows linearly in nt , and the optimization error is proportional to 1 /n +1 / √ t , given a time budget constraint nt = a , one may choose n = a 1 / 3 and t = a 2 / 3 , which yields an optimization error proportional to a -1 / 3 .

## 5 Local search strategies and peaked functions

In this section, we consider algorithms that primarily explore arms in N ( µ ) , where we recall that N ( µ ) is the set of all modes of µ and their neighbors. The proofs are deferred to Appendix D.

Local search. To analyze such algorithms within the Graves-Lai framework, we connect this behavior to the properties of the corresponding sampling rate vector η . Let N k ( t ) = ∑ t s =1 ✶ { k ( s ) = k } denote the number of times arm k has been selected by up to round t . The asymptotic sampling rate for arm k ∈ [ K ] is given by η k = lim sup T →∞ E [ N k ( T )] ln T . An algorithm performs local search if η k = 0 for all k / ∈ N ( µ ) . This leads directly to the following definition.

Definition 1. Consider η ∈ ( R + ) K a feasible solution to P GL . We say that η is a local search strategy if and only if η k = 0 for all k ̸∈ N ( µ ) . Further define C loc ( m, µ ) the optimal value of P GL restricted to the set of local search strategies.

The appeal of local search strategies stems from two key properties: they are provably optimal in the unimodal case ( m = 1 ), and are conceptually simpler than non-local strategies, which may explore all arms.

̸

Suboptimality of local search strategies. Unfortunately, and rather counterintuitively, not only are local search strategies suboptimal, the performance gap between local and non-local strategies is not upper bounded. More precisely, the ratio between the value of the best local strategy and the value of the best strategy can be arbitrarily large. Local search strategies are suboptimal because for every mode k = k ⋆ ( µ ) and every neighbor ℓ of k , they must be able to check that µ k &gt; µ ℓ , requiring a number of samples proportional to 1 d ℓ ( µ ℓ ,µ k ) , which can cause an arbitrarily large amount of regret if the function is flat in the neighborhood of k , so that µ k is very close to µ ℓ .

Theorem 3. Assume that |N ( µ ) | &lt; K . Then the following bounds hold:

<!-- formula-not-decoded -->

where δ k = min ℓ :( k,ℓ ) ∈ E | µ k -µ ℓ | &gt; 0 . As a consequence, sup µ ∈F m C loc ( m, µ ) C ( m, µ ) = + ∞ , i.e., the performance ratio between local and non-local strategies is unbounded.

In particular, this result implies that the IMED-MB algorithm from [18], which uses a local search strategy, cannot be asymptotically optimal, which contradicts the statement of their Theorem 2. This is rigorously shown in Appendix D.2.

Peaked reward functions. While in general local search strategies can be far from optimal, there exists a smaller subclass of reward functions on which they can be shown to be quasi-optimal, up to a constant multiplicative factor. We call these reward functions peaked in the sense that they are not flat around their modes.

Definition 2. A reward function µ ∈ R K is κ -peaked if and only if for all k ∈ M ( µ ) and all ℓ neighbor of k we have d k ( µ k , µ ⋆ ) ≤ κd k ( µ k , µ k -δ k / 2) and d ℓ ( µ ℓ , µ ⋆ ) ≤ κd ℓ ( µ ℓ , µ k -δ k / 2) .

In particular, when rewards are Gaussian, the condition above reduces to a simpler one: for all modes k , the gap between k and its neighbors should be at least proportional to the gap between k and the optimal arm, so that indeed, the function cannot be flat around the modes.

Proposition 10. Consider Gaussian rewards with fixed variance. Then µ is κ -peaked if and only if for all k ∈ M ( µ ) we have ∆ k ≤ δ k ( √ κ 2 -1) .

Proposition 11 shows that for κ -peaked reward functions, there exists local search strategies that are a factor κ from optimal.

Proposition 11. Assume that µ ∈ F m . Then the following bounds hold:

<!-- formula-not-decoded -->

and by corollary, if µ is κ -peaked, there exists local search strategies within a constant factor: sup µ ∈F m ,κ -peaked C loc ( m, µ ) C ( m, µ ) ≤ κ.

## 6 Numerical experiments

In this section, we conduct numerical experiments to demonstrate the benefit of properly exploiting the multimodal structure. To this end, we implement OSSB from [1] and use our approach to solve

P GL . At round t , OSSB samples as dictated by the solution η ⋆ ( t ) to the Graves-Lai problem for the empirical estimate ˆ µ ( t ) of µ , which is given by ˆ µ k ( t ) = ∑ t s =1 X k,s ✶ { k ( s )= k } max(1 ,N k ( t )) . We compare the cumulative regret of two algorithms:

- (i) Multimodal OSSB : the OSSB algorithm where the Graves-Lai problem is solved using our proposed method,
- (ii) Classical OSSB : the OSSB algorithm for unstructured bandits, which serves as a baseline.

The pseudo-code of OSSB and further details regarding the experiment are deferred to Appendix A.1.

G is chosen as a fixed binary tree of height two (resulting in K = 7 arms). We consider instances µ ∈ F 2 with rewards from arm k ∈ [ K ] drawn from a Gaussian distribution N ( µ k , 1) . The mean rewards µ k are generated as a sum of exponential functions centered on the modes in M ( µ ) :

<!-- formula-not-decoded -->

where ρ jk is the shortest path distance between nodes j and k in G . We choose M ( µ ) = { 4 , 6 } and k ⋆ ( µ ) = 6 . The parameter σ controls how peaked the reward function is: a small σ leads to sharp peaks with modes well-separated from their neighbors, whereas a large σ creates flatter modes.

We consider two instances: σ = 0 . 5 (easy instance) and σ = 4 (hard instance). We run the experiment up to a horizon of T = 10 , 000 . To reduce the computational burden of solving P GL at each round, we only update η ⋆ ( t ) when t = 2 k for k ∈ { 0 , . . . , ⌊ log 2 T ⌋} . The cumulative regret is averaged over 500 trials. The results are presented in Figure 3, where the shaded regions have radius one standard error. In both settings, multimodal OSSB exhibits superior performance over classical OSSB .

Figure 3: Cumulative regret as a function of the number of rounds.

<!-- image -->

Further experiments on the runtime of our dynamic programming approach are deferred to Appendices A.2 and E.8. The code used for the experiments is available at https://github.com/wilrev/ MultimodalBandits .

## 7 Conclusion

We have considered a stochastic multi-armed bandit with i.i.d. rewards and a multimodal reward structure, and have proposed the first known computationally tractable algorithm to solve the GravesLai optimization problem, which we have shown is a requirement to implement asymptotically optimal algorithms, as the performance ratio between local and non-local strategies can be arbitrarily large. We believe that an interesting direction for future work is to characterize the minimal computational complexity necessary to solve this problem in terms of the number of arms, modes and graph structure.

## References

- [1] Richard Combes, Stefan Magureanu, and Alexandre Proutiere. Minimal exploration in structured stochastic bandits. In Advances in Neural Information Processing Systems , 2017.
- [2] Richard Combes and Alexandre Proutiere. Unimodal bandits: Regret lower bounds and optimal algorithms. In Proceedings of the 31st International Conference on Machine Learning , 2014.
- [3] Emile Contal. Statistical learning approaches for global optimization . PhD thesis, Université Paris Saclay (COmUE), 2016.
- [4] Thibault Cuvelier, Richard Combes, and Eric Gourdin. Asymptotically optimal strategies for combinatorial semi-bandits in polynomial time. In ALT , 2021.
- [5] Remy Degenne, Han Shao, and Wouter M. Koolen. Adaptive algorithms for stochastic bandits. In Proceedings of ICML , 2020.
- [6] Aurelien Garivier and Olivier Cappe. The KL-UCB algorithm for bounded stochastic bandits and beyond. In Proceedings of COLT , 2011.
- [7] Todd L. Graves and Tze Leung Lai. Asymptotically efficient adaptive choice of control laws in controlled markov chains. SIAM Journal on Control and Optimization , 35(3):715-743, 1997.
- [8] Ulrich Herkenrath. The n-armed bandit with unimodal structure. Metrika , 30(1):195-210, 1983.
- [9] Junya Honda and Akimichi Takemura. An asymptotically optimal bandit algorithm for bounded support models. In Proceedings of COLT , 2010.
- [10] Emilie Kaufmann, Nathaniel Korda, and Rémi Munos. Thompson sampling: An asymptotically optimal finite-time analysis. In Proceedings of ALT , 2012.
- [11] Junpei Komiyama, Junya Honda, Hisashi Kashima, and Hiroshi Nakagawa. Regret lower bound and optimal algorithm in dueling bandit problem. In Proceedings of COLT , pages 1141-1154, 2015.
- [12] Dieter Kraft. A software package for sequential quadratic programming. Technical Report DFVLRFB 88-28, Deutsche Forschungs- und Versuchsanstalt für Luft- und Raumfahrt - Institut für Dynamik der Flugsysteme, Köln, Deutschland. , 1988.
- [13] Tze Lung Lai and Herbert Robbins. Asymptotically efficient adaptive allocation rules. Advances in Applied Mathematics , 6(1):4-22, 1985.
- [14] Holden Lee. MCMCalgorithms for sampling from multimodal and changing distributions . PhD thesis, Princeton University, 2019.
- [15] Kanishka Misra, Eric M. Schwartz, and Jacob Abernethy. Dynamic online pricing with incomplete information using multiarmed bandit experiments. Marketing Science , 38(2):226252, 2019.
- [16] Vivek Myers, Erdem Biyik, Nima Anari, and Dorsa Sadigh. Learning multimodal rewards from rankings. In Aleksandra Faust, David Hsu, and Gerhard Neumann, editors, Proceedings of the 5th Conference on Robot Learning , volume 164 of Proceedings of Machine Learning Research , pages 342-352, 08-11 Nov 2022.
- [17] Hassan Saber. Structure Adaptation in Bandit Theory . PhD thesis, Université de Lille, 2022.
- [18] Hassan Saber and Odalric-Ambrym Maillard. Bandits with multimodal structure. Reinforcement Learning Journal , 5:2400-2439, 2024.
- [19] Hassan Saber, Pierre Ménard, and Odalric-Ambrym Maillard. Indexed minimum empirical divergence for unimodal bandits. In Proceedings of NeurIPS , volume 34, pages 7346-7356, 2021.
- [20] Shai Shalev-Shwartz and Shai Ben-David. Understanding Machine Learning: From Theory to Algorithms . Cambridge University Press, 2014.

- [21] Cindy Trinh, Emilie Kaufmann, Claire Vernade, and Richard Combes. Solving bernoulli rank-one bandits with unimodal thompson sampling. In Proceedings of ALT , 2020.
- [22] Bart Van Parys and Negin Golrezaei. Optimal learning for structured bandits. Management Science , 2023.
- [23] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C. J. Carey, ˙ Ilhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. Scipy 1.0: fundamental algorithms for scientific computing in python. Nature Methods , 17(3):261-272, Mar 2020.
- [24] Yining Wang, Boxiao Chen, and David Simchi-Levi. Multimodal dynamic pricing. Manage. Sci. , 67(10):6136-6152, October 2021.
- [25] Jia Yuan Yu and Shie Mannor. Unimodal bandits. In Proceedings of ICML , page 41-48, 2011.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We indeed provide in the paper the first known computationally tractable algorithm to solve the Graves-Lai problem for multimodal bandits.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We state in the conclusion that interesting future work would be to characterize the minimal computational complexity required to solve the Graves-Lai problem.

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

Justification: Assumptions are clearly stated in Section 3 and detailed proofs of all results are provided in the appendices.

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

Justification: The dynamic programming procedure used is described fully in the main paper. Details on OSSB's implementation (along with the corresponding pseudo-code) and our experiments is provided in Appendix A.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed

instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: The code used for the experiments is available at https://github.com/ wilrev/MultimodalBandits .

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

Justification: The specific instances considered are provided explicitly in the paper.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Empirical 95% confidence intervals are provided for the regret results.

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

Justification: Hardware specifications are provided in Appendix A.2.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our research does not involve human subjects or sensitive data, and we do not believe our research findings to have any potential negative societal impact.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our work is mostly theoretical, and consequently has no immediate societal impact.

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

Justification: Our work poses such risk.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The code used is our own and the data is synthetic.

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

Justification: We introduce no such asset.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: We conduct no such experiment.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: We conduct no such experiment.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: LLMs were not used as a non-standard component in this work.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/ LLM ) for what should or should not be described.

## A Experimental details

## A.1 OSSB for multimodal bandits

̸

As explained in Section 3, asymptotically optimal algorithms for structured bandits attempt to sample k = k ⋆ ( µ ) a number of times η ⋆ k ln T + o (ln T ) when T →∞ , where η ⋆ is a solution to P GL . The OSSB algorithm of [1] does so by sampling as dictated by the solution to the Graves-Lai problem for the empirical estimate ˆ µ ( t ) of µ , updated at each round t . The pseudo-code of OSSB for multimodal bandits is given in Algorithm 1.

̸

## Algorithm 1 Optimal Sampling for Structured Bandits ( OSSB ) N k (1) ← 0 , ˆ µ k (1) ← 0 ∀ k ∈ [ K ] {Initialization} for t = 1 , . . . , T do Compute η ⋆ ( t ) the solution to P GL for ˆ µ ( t ) if ∀ k ∈ [ K ] , N k ( t ) ≥ η ⋆ k ( t ) ln t then k ( t ) ← arg max k ∈ [ K ] ˆ µ k ( t ) {Exploitation, ties broken arbitrarily} else k ( t ) ← arg min k ∈ [ K ] N k ( t ) η ⋆ k ( ˆ µ ( t )) {Exploration, ties broken arbitrarily} end if Observe X k, ( t ) ,t for k = k ( t ) do ˆ µ k ( t +1) ← ˆ µ k ( t ) N k ( t +1) ← N k ( t ) end for ˆ µ k ( t ) ( t +1) ← X k ( t ) ,t +ˆ µ k ( t ) ( t ) N k ( t ) ( t ) N k ( t ) ( t )+1 N k ( t ) ( t +1) ← N k ( t ) ( t ) + 1 end for

We implement two versions of OSSB . Each version is associated with a specific sampling strategy η ⋆ ( t ) , which it aims to follow at round t . Let ˆ ∆ k ( t ) = max k ′ ∈ [ K ] ˆ µ k ′ ( t ) -ˆ µ k ( t ) . These strategies are given by:

- the solution to P GL for ˆ µ ( t ) , as computed by our algorithm, with the convention η ⋆ k ( t ) = 0 if k ∈ arg max ˆ µ ( t ) ( Multimodal OSSB , Algorithm 1)
- η ⋆ k ( t ) = 1 d k (ˆ µ k ( t ) , ˆ µ ( t ) ⋆ ) ✶ { ˆ ∆ k ( t ) &gt; 0 } ( Classical OSSB , rates given by the Lai-Robbins bound [13]).

The specific instances µ ∈ F 2 considered in the experiment of Section 6 are shown in Figure 4.

Figure 4: 2 -modal reward instances on the binary tree G ( K = 7 , M ( µ ) = { 4 , 6 } , k ⋆ ( µ ) = 6) .

<!-- image -->

In this specific experiment, instead of the penalized subgradient subroutine described in Proposition 9, we used the Sequential Least SQuares Programming (SLSQP) method from [12] and implemented in SciPy by [23] for faster convergence towards a solution to P GL for each ˆ µ ( t ) , and we only solve P GL when t = 2 k for k ∈ { 0 , . . . , ⌊ log 2 T ⌋} . We used n = 100 discretization points.

## A.2 Runtime experiment

In this experiment, we evaluate the runtime of the algorithm of Theorem 2 with respect to the number of arms K . We generate line graphs with varying numbers of arms K ∈ { 20 , 25 , 30 , . . . , 70 } , number of modes | M ( µ ) | = m ∈ { 2 , 3 , 4 , 5 } , for n = 100 discretization points and t = 100 iterations of penalized subgradient descent. The reward instances µ are generated as in Section 6 with σ = 2 . To ensure that they are always m -modal, the position of k ⋆ ( µ ) and of the other modes of µ are chosen to be as spread out as possible. We perform 5 trials per configuration ( K,m ) . Figure 5 displays the average runtime as a function of K for each value of m on a log-log plot, as well as the corresponding slopes obtained from log-log regression. The runtime exhibits an approximately quadratic growth with the number of arms, which aligns with the time complexity O ( K 2 mnt ) stated in Theorem 2.

Figure 5: Runtime as a function of the number of arms.

<!-- image -->

All experiments were run on a single desktop PC equipped with an AMD Ryzen 7 5800X 8-core/16thread CPU @ 3.8 GHz, 16 GB DDR4 RAM.

## B Proofs of Section 3

Proof of Proposition 1. Applying Theorem 1 in [1] to the parameter set F ≤ m yields that the GravesLai constant C ( m, µ ) is the value of

<!-- formula-not-decoded -->

̸

with λ ( m, µ ) = { λ ∈ F ≤ m , d k ( µ k , λ k ) = 0 , k ⋆ ( λ ) = k } for k = k ⋆ ( µ ) . By Assumption 1, d k ( µ k , λ k ) = 0 holds if and only if µ k = λ k , which concludes the proof.

## C Proofs of Section 4

## C.1 Proof of Proposition 2

Assume that B ( m, µ ) = ∪ U ( K,m ) i =1 Z i where the Z i are disjoint convex sets. Denote by X ( m ) the set of independent sets of G of size m , and let us lower bound its size. Consider the process in which we first select a node x 1 ∈ [ K ] , and then for i = 2 , ..., m we select a node x i ∈ [ K ] in [ K ] \ ∪ i -1 j =1 N ( x j ) where N ( x j ) is x j and its neighbors. Then X = { x 1 , ..., x m } is an independent set of G of size m , and at step i of the process has at least

<!-- formula-not-decoded -->

choices, since | N ( x j ) | ≤ deg( G ) + 1 and i ≤ m . Hence, the number of independent sets of G of size m is at least

<!-- formula-not-decoded -->

For each independent set of size m , X ∈ X ( m ) , define the vector µ X ∈ R K such that µ X k = ✶ { k ∈ X } for k ∈ [ K ] . We can readily check that µ X is multimodal with m modes, and that its m modes are precisely the elements of X . Now consider another independent set of size m , X ′ ∈ X ( m ) and consider λ X,X ′ = (3 / 4) µ X +(1 / 4) µ X ′ a convex combination of µ X and µ X ′ . For k ∈ X we have that λ k = 3 / 4 and λ k ′ ≤ 1 / 4 for any k ′ that is a neighbor of k , and hence k is a mode of λ . On the other hand, assume that there exists k ∈ X ′ such that k is not a neighbor of a node in X . Then λ X,X ′ k = 1 / 4 and λ X,X ′ k ′ = 0 for any k ′ that is a neighbor of k so that k is a mode of λ X,X ′ . Putting it together, this means that, for λ X,X ′ to have m modes, one must make sure each element of X ′ is the neighbor of an element in X .

Consider X ∈ Z i for some i . For any X ′ ∈ ∪ Z i by convexity we must have λ X,X ′ ∈ Z i so that λ X,X ′ has m modes. By the above this means that all elements of X ′ must be neighbors of X , so |X ( m ) ∪ Z i | ≤ ( deg( G ) m m ) since X ′ has m elements and X has at most deg( G ) m neighbors. Since the Z i are disjoint,

<!-- formula-not-decoded -->

This yields the result: U ( K,m ) ≥ ((deg( G ) -1) m )! (deg( G ) m )! ( K -(deg( G ) + 1) m ) m .

## C.2 Proof of Proposition 3

By definition, for any λ ∈ B k ( m, µ ) , we must have λ k &gt; µ ⋆ , so that inf λ ∈ B ( m, µ ) η ⊤ d ( µ , λ ) ≥ η k d k ( µ k , µ ⋆ ) . Conversely, let ε &gt; 0 and λ ε = µ +( µ ⋆ + ε -µ k ) e ( k ) . If |M ( µ ) | &lt; m , M ( λ ε ) ⊂ M ( µ ) ∪ { k } ensures that |M ( λ ε ) | ≤ m and in turn λ ε ∈ B k ( m, µ ) . If k ∈ N ( µ ) , either k is a mode of µ and M ( λ ε ) = M ( µ ) , or k is a neighbor of a mode ℓ and M ( λ ε ) = M ( µ ) ∪ { k } \ { ℓ } . In any case, |M ( λ ε ) | ≤ m and in turn, λ ε ∈ B k ( m, µ ) . Consequently, inf λ ∈B k ( m, µ ) η ⊤ d ( µ , λ ) ≤ η ⊤ d ( µ , λ ε ) = η k d k ( µ k , µ ⋆ + ε ) . Letting ε → 0 yields the other side of the inequality. Finally, note that µ +( µ ⋆ -µ k ) e ( k ) = lim ε → 0 λ ε ∈ B k ( m, µ ) . This concludes the proof.

## C.3 Proof of Proposition 4

We first demonstrate the following intermediary result.

Lemma 1. The closure of B k ( m, µ ) in R K is given by

<!-- formula-not-decoded -->

Proof. Consider ( λ t ) t ≥ 0 a sequence of elements of B k ( m, µ ) with lim t →∞ λ t = λ ∞ and let us prove that λ ∞ ∈ F k . For all t ≥ 0 , we have λ t k ⋆ ( µ ) = λ ∞ k ⋆ ( µ ) = µ ⋆ . Furthermore, for all t ≥ 0 , k ⋆ ( λ t ) = k , hence λ t k &gt; λ t k ⋆ ( µ ) = µ ⋆ so that λ ∞ k ≥ µ ⋆ . Now, let i ∈ M ( λ ∞ ) , which implies λ ∞ i &gt; λ ∞ ℓ for all ℓ neighbor of i . Hence, for all t large enough, i ∈ M ( λ t ) . Repeating the same reasoning for all the modes of λ ∞ , for all t large enough, M ( λ ∞ ) ⊂ M ( λ t ) , and |M ( λ ∞ ) | ≤ |M ( λ t ) | ≤ m . We have proven that λ ∞ ∈ F k . Conversely, let λ ∈ F k . Either λ k &gt; µ ⋆ and λ ∈ B k ( m, µ ) ⊂ B k ( m, µ ) , or λ j ≤ µ ⋆ for all j . There are then two cases to distinguish.

̸

- (i) If k is a mode of λ , we can write λ = lim t →∞ λ t where λ t ∈ B k ( m, µ ) is defined by λ t k = µ ⋆ +1 /t and λ t j = λ j for j = k .

̸

- (ii) If k is not a mode of λ , as λ j ≤ µ ⋆ for all j , this must mean that there exists a neighbor ℓ of k such that λ ℓ = λ k = µ ⋆ . Note further that k / ∈ N ( µ ) ensures ℓ = k ⋆ ( µ ) . Then we can

<!-- formula-not-decoded -->

In any case, we have shown that λ ∈ B k ( m, µ )

, which concludes the proof.

We now prove Proposition 4. By Assumption 2, λ ↦→ η ⊤ d ( µ , λ ) is continuous, so that inf λ ∈B k ( m, µ ) η ⊤ d ( µ , λ ) = inf λ ∈B k ( m, µ ) η ⊤ d ( µ , λ ) . Consider now λ ∈ B k ( m, µ ) such that λ ℓ &gt; µ ⋆ for some ℓ and let ε = min ℓ,λ ℓ &gt;µ ⋆ ( λ ℓ -µ ⋆ ) &gt; 0 the minimal amount by which an entry of λ is larger than µ ⋆ . We will show that there exists λ ′ ∈ B k ( m, µ ) such that η ⊤ d ( µ , λ ′ ) &lt; η ⊤ d ( µ , λ ) .

̸

Consider a graph G ′ = ([ K ] , E ′ ) where ( i, j ) ∈ E ′ if and only if ( i, j ) ∈ E and λ i = λ j . Consider S ⊂ [ K ] the connected component of G ′ where ℓ lies. Consider the minimal gap between two neighboring arms: δ = min ( i,j ) ∈ E,λ i = λ j | λ i -λ j | . Consider λ ′ such that λ ′ i = λ i -(min( ε, δ ) / 2) ✶ { i ∈ S } . As the nodes are modified by strictly less than δ , we have M ( λ ′ ) = M ( λ ) . As they are modified by strictly less than ε , λ ′ k &gt; µ ⋆ . In turn, λ ′ ∈ B k ( m, µ ) . Furthermore, d ( µ , λ ′ ) &lt; d ( µ , λ ) since for all k , λ k ↦→ d k ( µ k , λ k ) is strictly increasing whenever λ k &gt; µ ⋆ ≥ µ k , which implies that η ⊤ d ( µ , λ ′ ) &lt; η ⊤ d ( µ , λ ) .

We may prove similarly that if λ ℓ &lt; µ ⋆ for some ℓ , there exists λ ′ ∈ B k ( m, µ ) such that η ⊤ d ( µ , λ ′ ) &lt; η ⊤ d ( µ , λ ) . This ensures that inf λ ∈B k ( m, µ ) η ⊤ d ( µ , λ ) = inf λ ∈B ′ k ( m, µ ) η ⊤ d ( µ , λ ) . Finally, as λ ↦→ η ⊤ d ( µ , λ ) is continuous on the compact set B ′ k ( m, µ ) , the infimum is attained. This concludes the proof.

## C.4 Proof of Proposition 5

Consider η defined as η k = 1 d k ( µ k ,µ ⋆ ) for all k = k ⋆ ( µ ) , and where the value of η k ⋆ ( µ ) is arbitrary. Let us check that η is a feasible solution to P GL . For any λ ∈ B ( m, µ ) , there must exist k = k ⋆ ( µ ) such that λ k &gt; µ ⋆ and therefore η ⊤ d ( µ , λ ) ≥ η k d k ( µ k , λ k ) &gt; η k d k ( µ k , µ ⋆ ) = 1 . This shows that inf λ ∈ B ( m, µ ) η ⊤ d ( µ , λ ) ≥ 1 hence η is indeed a feasible solution to P GL , so η must have a higher value than a solution η ⋆ of P GL . As long as η ⋆ k = 0 when ∆ k = 0 (note that this choice does not impact the value of P GL ), we get

̸

<!-- formula-not-decoded -->

hence ∥ η ⋆ ∥ ∞ ≤ B ( µ ) as announced.

## C.5 Proof of Proposition 6

̸

Consider i = k a mode i ∈ M ( λ ⋆ ) , and define the minimal difference between i and its neighbors δ i = min ( i,j ) ∈ E | λ ⋆ i -λ ⋆ j | ) For all δ ′ ∈ ( -δ i , δ i ) , it is noted that M ( λ ⋆ ) = M ( λ ⋆ + δ ′ e ( i ) ) , and so λ ⋆ + δ ′ e ( i ) ∈ B ′ k ( m, µ ) . Therefore, the function δ ′ ↦→ η d ( µ ⋆ , λ ⋆ + δ ′ e ( i ) ) must attain its minimum at δ ′ = 0 , which implies λ ⋆ i = µ i . Further consider ℓ a neighbor of i and δ ℓ = | λ ⋆ i -λ ⋆ ℓ | . For all δ ′ ∈ [0 , δ ℓ ) , it is noted that M ( λ ⋆ + δ ′ e ( ℓ ) ) ⊂ M ( λ ⋆ ) , so that λ ⋆ + δ ′ e ( ℓ ) ∈ B ′ k ( m, µ ) . Therefore, the function δ ′ ↦→ η d ( µ ⋆ , λ ⋆ + δ ′ e ( ℓ ) ) must attain its minimum at δ ′ = 0 , which implies λ ⋆ ℓ ≥ µ ℓ . Putting it together, we have proven that, if i ∈ M ( λ ⋆ ) , for all ( ℓ, i ) ∈ E we have µ ℓ ≤ λ ⋆ ℓ &lt; λ ⋆ i = µ i , and in turn i ∈ M ( µ ) , which concludes the proof.

## C.6 Proof of Proposition 7

Let us consider λ a minimizer of η ⊤ d ( µ , λ ) over B k,k ′ ( m, µ ) . We already know that λ ∈ [ µ ⋆ , µ ⋆ ] from Proposition 4. Consider G k the directed tree rooted at k and define ˜ λ a rounding of λ using the following recursive procedure: start at the root ˜ λ k ∈ arg min {| x -λ k | : x ∈ D ( n, µ ) } then for all ℓ , choose

<!-- formula-not-decoded -->

̸

where p ( ℓ ) is the parent of ℓ and sgn is the sign function. The idea is that, when rounding in this fashion, we both have the insurance that for any ( i, ℓ ) ∈ E | ˜ λ ℓ -˜ λ i | ≤ (1 /n )( µ ⋆ -µ ⋆ ) so that, by recursion over ℓ : ∥ ˜ λ -λ ∥ ∞ ≤ (1 /n )diam( G )( µ ⋆ -µ ⋆ ) and we also have that sgn( λ i -λ ℓ ) = sgn( ˜ λ i -˜ λ ℓ ) so that M ( ˜ λ ) = M ( λ ) . In essence, we round λ to ensure both a small rounding error while leaving the set of modes unchanged. We further have

<!-- formula-not-decoded -->

with C ( µ ) = diam( G )( µ ⋆ -µ ⋆ ) B ( µ ) A ( µ ) K using Assumption 2, the fact that ∥ η ∥ ∞ ≤ B ( µ ) and the previous bound. This concludes the proof.

## C.7 Proof of Proposition 8

We recall that we consider the graph G k , which is a directed tree rooted at k . Consider node ℓ and its parent p ( ℓ ) , assume that λ p ( ℓ ) is known, and we wish to minimize the value:

<!-- formula-not-decoded -->

which corresponds to η ⊤ d ( µ , λ ) restricted to ℓ and its descendants, subject to λ ∈ B k,k ′ ( m, µ ) . Then it suffices to optimize over λ ℓ and λ j for j ∈ D ( ℓ ) . Also, we may readily check that setting η k ⋆ ( µ ) = + ∞ is equivalent to enforcing the constraint λ k ⋆ ( µ ) = µ ⋆ . Of course, the sign of λ ℓ -λ p ( ℓ ) is important to ensure that the constraints are satisfied. Define f ℓ ( z, +1) the minimal value that can be obtained if selecting λ ℓ = z &gt; λ p ( ℓ ) and f ℓ ( z, -1) the minimal value that can be obtained if selecting λ ℓ = z ≤ λ p ( ℓ ) . Consider λ ℓ = z and λ p ( ℓ ) fixed, and let us examine how one can select λ j for j ∈ C ( ℓ ) the children of ℓ to ensure that λ ∈ B k,k ′ ( m, µ ) is respected.

(i) First consider ℓ ̸∈ M ( µ ) ∪ { k } \ { k ′ } , so that ℓ should not be a mode. If λ ℓ ≤ λ p ( ℓ ) then ℓ cannot be a mode anyways, so for each j ∈ C ( ℓ ) we have two choices: select λ j ≤ λ ℓ , which gives a minimal value of min w ≤ z f j ( w, -1) = f ⋆ j ( z, -1) , or select λ j &gt; λ ℓ , which gives a minimal value of min w&gt;z f j ( w, +1) = f ⋆ j ( z, +1) . The minimal value over these two choices is f ⋄ j ( z ) = min( f ⋆ j ( z, -1) , f ⋆ j ( z, +1)) . Therefore,

<!-- formula-not-decoded -->

(ii) If λ ℓ &gt; λ p ( ℓ ) , since ℓ should not be a mode, one must make sure that there exists at least one child v ∈ C ( ℓ ) of ℓ such that λ ℓ ≤ λ v , and the value of λ j for j ∈ C ( ℓ ) \ { v } can be chosen freely. Of course, if C ( ℓ ) = ∅ this is not possible and one has simply f ℓ ( z, +1) = + ∞ . If C ( ℓ ) = ∅ and v ∈ C ( ℓ ) , min { f ⋆ v ( z, +1) , f v ( z, -1) } represents the minimal cost of choosing a value of λ v such that λ ℓ ≤ λ v . By the same reasoning as before, the minimal obtainable value is therefore:

̸

<!-- formula-not-decoded -->

for g v ( z ) = min { f ⋆ v ( z, +1) , f v ( z, -1) } -f ⋄ v ( z ) , where we have minimized over the choice of v ∈ C ( ℓ ) .

(iii) Now consider the case ℓ ∈ M ( µ ) ∪ { k } \ { k ′ } . Since ℓ can either be a mode or not a mode, regardless of the sign of λ ℓ -λ p ( ℓ ) , we have no constraints on the choice of λ j for j ∈ C ( ℓ ) and the minimal value that can be obtained is

<!-- formula-not-decoded -->

for both u = -1 and u = +1 .

We have proven that f ℓ indeed obeys the dynamic programming equations. Since k is the root of the tree and we must have λ ⋆ k = µ ⋆ , then f k ( µ ⋆ , u ) is the value of P GL ( k, k ′ ) for any u .

## C.8 Proof of Corollary 1

From the definition of ˜ P GL ( k, k ′ ) , λ ⋆ k = µ ⋆ . Consider ℓ = k and its parent p ( ℓ ) in G k , and assume that λ ⋆ p ( ℓ ) is known. There are two cases to consider:

̸

- (i) If p ( ℓ ) / ∈ M ( µ ) ∪ { k } \ { k ′ } , λ ⋆ p ( ℓ ) &gt; λ ⋆ p 2 ( ℓ ) where p 2 ( ℓ ) is the parent of p ( ℓ ) , and ℓ = arg min v ∈C ( p ( ℓ )) g v ( z ) is the node that induces the smallest cost when constrained, we must have λ ⋆ ℓ ≥ λ ⋆ p ( ℓ ) to ensure that p ( ℓ ) is not a mode. There are two choices: either select λ ⋆ ℓ &gt; λ ⋆ p ( ℓ ) which yields value f ⋆ ℓ ( λ ⋆ p ( ℓ ) , +1) and dictates the choice λ ⋆ ℓ ∈ arg min z&gt;λ ⋆ p ( ℓ ) f ℓ ( z, +1) , otherwise select λ ⋆ ℓ = λ ⋆ p ( ℓ ) which yields value f ℓ ( λ ⋆ p ( ℓ ) , -1) . Taking the minimal value among the two choices gives:

<!-- formula-not-decoded -->

- (ii) Otherwise, there are no constraints on the value of λ ⋆ ℓ : either select λ ⋆ ℓ &gt; λ ⋆ p ( ℓ ) which yields value f ⋆ ℓ ( λ ⋆ p ( ℓ ) , +1) and dictates the choice λ ⋆ ℓ ∈ arg min z&gt;λ ⋆ p ( ℓ ) f ℓ ( z, +1) , otherwise select λ ⋆ ℓ ≤ λ ⋆ p ( ℓ ) which yields value f ⋆ ℓ ( λ ⋆ p ( ℓ ) , -1) and dictates the choice λ ⋆ ℓ ∈ arg min z ≤ λ ⋆ p ( ℓ ) f ℓ ( z, -1) . Taking the minimal value among the two choices gives:

<!-- formula-not-decoded -->

It is noted that storing the values of f ℓ ( z, u ) , f ⋆ ℓ ( z, u ) and f ⋄ ℓ ( z ) for ℓ ∈ [ K ] , z ∈ D ( n, µ ) and u ∈ {-1 , +1 } requires memory O ( nK ) since | D ( n, µ ) | = n . Furthermore, if f j ( z, u ) , f ⋆ j ( z, u ) and f ⋄ j ( z ) for j ∈ C ( ℓ ) , z ∈ D ( n, µ ) and u ∈ {-1 , +1 } , have been computed, then one may compute f ℓ ( z, u ) , f ⋆ ℓ ( z, u ) and f ⋄ ℓ ( z ) for z ∈ D ( n, µ ) and u ∈ {-1 , +1 } in time O ( n |C ( ℓ ) | ) from the dynamic programming equations. Since ∑ ℓ ∈ [ K ] |C ( ℓ ) | = K -1 , the complete procedure requires time O ( nK ) . This completes the proof.

## C.9 Proof of Proposition 9

To ease the notation, denote

<!-- formula-not-decoded -->

Recall that h ( η ) = η ⊤ ∆ + γ max[1 -g ( η , µ , n ) , 0] which is convex as a sum of a linear function, and a maximum of linear functions, with subgradients:

∆ -γd ( µ , λ ) ✶ { g ( η , µ , n ) &lt; 1 } = v ∈ ∂h ( η ) for any λ such that g ( η , µ , n ) = η ⊤ d ( µ , λ ) . The norm of a subgradient is upper bounded as

<!-- formula-not-decoded -->

since

<!-- formula-not-decoded -->

Hence, h is Lipschitz continuous with Lipschitz constant E ( µ ) . Let η ⋆ the minimizer of h over ( R + ) K , and let us prove that g ( η ⋆ , µ , n ) ≥ 1 . Any subgradient at η ⋆ must have positive entries so that:

̸

✶ where ˜ λ is such that g ( η ⋆ , µ , n ) = η ⋆ ⊤ d ( µ , ˜ λ ) . In turn, since ˜ λ ∈ B ( m, µ ) there must exist k = k ⋆ ( µ ) such that ˜ λ k ≥ µ ⋆ and hence:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

by definition of γ . This implies that η ⊤ d ( µ , ˜ λ ) ≥ 1 and so g ( η ⋆ , µ , n ) ≥ 1 . So η ⋆ minimizes η ⊤ ∆ over the set of η such that g ( η , µ , n ) ≥ 1 .

We may now analyze the iterative scheme in itself. Since h is convex and Lipschitz continuous with Lipschitz constant E ( µ ) , and our iterative scheme is a projected subgradient descent with step size δ , from [20][Lemma 14.1]:

<!-- formula-not-decoded -->

where we used Proposition 5 and setting δ 2 = K B ( µ ) 2 t E ( µ ) 2 to optimize the right hand side.

Let us now check that ˜ η ( t ) is an approximate solution to P GL . From the above

<!-- formula-not-decoded -->

using the definition of h , the above bound, the fact that η ⋆ ⊤ ∆ is the minimum of η ⊤ ∆ subject to g ( η , µ , n ) ≥ 1 , η ≥ 0 and the fact that C ( m, µ ) is the minimum of η ⊤ ∆ subject to g ( η , µ ) ≥ 1 , η ≥ 0 . Scaling the above on both sides gives a bound on the value of ˜ η ( t ) :

<!-- formula-not-decoded -->

We now have to check that ˜ η ( t ) is a feasible solution to P GL . Denote by ϕ = g ( ¯ η ( t ) , µ , n ) . Consider the case ϕ &lt; 1 , so that:

<!-- formula-not-decoded -->

where we used the fact that g is homogeneous, i.e. g ( η , µ , n ) ≥ ϕ if and only if g ( η /ϕ, µ , n ) ≥ 1 . This implies

<!-- formula-not-decoded -->

Rearranging the terms we get:

Therefore, using Proposition 7:

<!-- formula-not-decoded -->

And once again using the homogeneity of g we get

<!-- formula-not-decoded -->

which is the announced result and concludes the proof.

## D Proofs of Section 5

## D.1 Proof of Theorem 3

̸

Consider j ̸∈ N ( µ ) , k = k ⋆ ( µ ) a mode of µ and ℓ the neighbor of k which is the closest to k , that is arg min ℓ :( k,ℓ ) ∈ E | µ k -µ ℓ | . Define the parameter λ = µ +( µ ⋆ -µ j ) e ( j ) +( µ ℓ -µ k ) e ( k ) . One may check that λ ∈ B ( m, µ ) . For any feasible local search strategy η we have 1 ≤ η ⊤ d ( µ , λ ) = η j d j ( µ j , λ j )+ η k d k ( µ k , λ ℓ ) = η j d j ( µ j , µ ⋆ )+ η k d k ( µ k , µ k -δ k ) = η k d k ( µ k , µ k -δ k ) since η j = 0

<!-- formula-not-decoded -->

as j ̸∈ N ( µ ) and η is a local search strategy. Hence, ∆ k η k ≥ ∆ k d k ( µ k ,µ k -δ k ) for any mode k and summing over modes we get the announced result

<!-- formula-not-decoded -->

Now, since a multimodal bandit problem is also a classical bandit problem, we always have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

as there exists reward functions µ ∈ F m where δ k is arbitrarily small while µ ⋆ -µ k remains comparatively large for any k = k ⋆ ( µ ) , for instance consider a case where µ k = 1 if k = k ⋆ ( µ ) , µ k = ϵ if k ∈ M ( µ ) \ { k ⋆ ( µ ) } and µ k = 0 if k ̸∈ M ( µ ) . This function has exactly m modes and δ k = ϵ for any mode k = k ⋆ ( µ ) .

̸

## D.2 Suboptimality of IMED-MB

We argue that IMED-MB cannot be asymptotically optimal for all instances µ ∈ F ≤ m , contrary to the statement of Theorem 2 in [18]. Consider an instance µ ∈ F m for which the optimal value for local search strategies is strictly greater than the value of P GL , i.e. C loc ( m, µ ) &gt; C ( m, µ ) . Such an instance is guaranteed to exist by Theorem 3. Assume by contradiction that their Theorem 2 holds. Then, by their Proposition 1, IMED-MB is asymptotically optimal, so that

<!-- formula-not-decoded -->

Furthermore, by their Proposition 1 and Theorem 2 again, for k = k ⋆ ( µ ) and η k = lim T →∞ E [ N k ( T )] ln( T ) , we have η k = 1 d k ( µ k ,µ ⋆ ) if k ∈ N ( µ ) , and η k = 0 otherwise. In particular, η is a local search strategy (and IMED-MB is uniformly good), so that its regret must verify

<!-- formula-not-decoded -->

Combining these inequalities leads to:

<!-- formula-not-decoded -->

a contradiction.

## D.3 Proof of Proposition 10

Consider k a mode and ℓ one of its neighbors. For Gaussian rewards the conditions become ( µ ⋆ -µ k ) 2 ≤ κ ( δ k / 2) 2 and ( µ ⋆ -µ ℓ ) 2 ≤ κ ( µ k -δ k / 2 -µ ℓ ) 2 . These conditions are equivalent to ∆ k ≤ √ κδ k / 2 and ∆ ℓ ≤ √ κ (∆ ℓ -∆ k -δ k / 2) . This must be true for all ℓ neighbor of k and should hold when ∆ ℓ = ∆ k + δ k , therefore we must have ∆ k ≤ ( √ κ 2 -1) δ k . Hence, µ is κ -peaked if and only if ∆ k ≤ δ k ( √ κ 2 -1) for all k ∈ M ( µ ) .

## D.4 Proof of Proposition 11

By Proposition 3, for k ∈ N ( µ ) \ { k ⋆ ( µ ) } , inf λ ∈ B ( m, µ ) η ⊤ d ( µ , λ ) ≤ η k d k ( µ k , µ ⋆ ) , so for any η feasible solution to P GL we must have η k ≥ 1 d k ( µ k ,µ ⋆ ) . Summing over k ∈ N ( µ ) \ { k ⋆ ( µ ) } , the value of P GL is lower bounded as

<!-- formula-not-decoded -->

so that

̸

̸

Now, consider the local search strategy η defined as

<!-- formula-not-decoded -->

Let us check that η is feasible. Consider λ ∈ B ( m, µ ) attaining its maximum at k ∈ [ K ] \ { k ⋆ ( µ ) } . On the one hand, if k ∈ N ( µ ) , since λ k &gt; µ ⋆ we have η ⊤ d ( µ , λ ) ≥ η k d k ( µ k , λ k ) &gt; η k d k ( µ k , µ ⋆ ) ≥ 1 . On the other hand, if k ̸∈ N ( µ ) there must exist at least one j ∈ M ( µ ) such that j ̸∈ M ( λ ) , as otherwise λ would have more than m modes. In turn there must exist ℓ a neighbor of j such that λ j ≤ λ ℓ . This implies that either λ j ≤ µ j -δ j / 2 &lt; µ j and in that case η ⊤ d ( µ , λ ) ≥ η j d j ( µ j , λ j ) ≥ η j d j ( µ j , µ j -δ j / 2) ≥ 1 , or λ ℓ &gt; µ j -δ j / 2 &gt; µ ℓ and in that case η ⊤ d ( µ , λ ) ≥ η ℓ d ℓ ( µ ℓ , λ ℓ ) &gt; η ℓ d ℓ ( µ ℓ , µ j -δ j / 2) ≥ 1 . In all cases we have η ⊤ d ( µ , λ ) ≥ 1 for all λ ∈ B ( m, µ ) , hence η is feasible. This concludes the proof.

## E An improved dynamic programming procedure

In this section, we describe a more complex, but more computationally efficient dynamic program than that presented in Section 3, which enables solving P GL more efficiently. Throughout this section we view G as the directed tree G k ⋆ ( µ ) rooted at node k ⋆ ( µ ) .

## E.1 Dynamic programming variables

We describe a dynamic programming procedure to solve the optimization problem

̸

<!-- formula-not-decoded -->

̸

where ˜ B k ( m, µ ) = B ′ k ( m, µ ) ∩ D ( n, µ ) K (refer to Proposition 4 for the definition of B ′ k ( m, µ ) ). Consider λ ⋆ the optimal solution to this problem. Then there exists a node k = k ⋆ ( µ ) such that λ ⋆ k = µ ⋆ . With a slight abuse of notation, we denote this node by k ⋆ ( λ ⋆ ) . We distinguish two cases.

Case 1. If k = k ⋆ ( λ ⋆ ) ∈ N ( µ ) , from Proposition 3, the optimal solution is λ ⋆ = µ +( µ ⋆ -µ k ) e ( k ) , and the optimal value is η k d k ( µ k , µ ⋆ ) .

Case 2. If k = k ⋆ ( λ ⋆ ) ̸∈ N ( µ ) , from Proposition 6, the modes of λ ⋆ must be located at M ( λ ⋆ ) = M ( µ ) ∪ { k } \ { k ′ } , where k ′ ∈ M ( µ ) \ { k ⋆ ( µ ) } is the mode of µ that is not a mode of λ ⋆ .

Given a node ℓ of G , and flags ( z, a, b, c ) ∈ D ( n, µ ) × { 0 , 1 , 2 } × { 0 , 1 } × { 0 , 1 } we define h ℓ ( z, a, b, c ) as the minimal value of ∑ j ∈D ( ℓ ) ∪{ ℓ } η j d j ( µ j , λ j ) where M ( λ ⋆ ) = M ( µ ) ∪ { k ⋆ ( λ ⋆ ) } \ { k ′ } for some k ′ ∈ M ( µ ) \ { k ⋆ ( µ ) } , under four constraints:

(i)

λ

=

z

<!-- formula-not-decoded -->

{

ℓ

(iii) b = ✶ (iv) c = ✶ { k ′ ∈ D ( ℓ ) ∪ { ℓ }} We further define λ ⋆ ( ℓ, z, a, b, c ) as the corresponding optimal solution. Recall that G k ⋆ ( µ ) is a tree rooted at k ⋆ ( µ ) and that in case 2, k ⋆ ( µ ) must be a mode of λ ⋆ . Putting the two cases together, the optimal solution to ˜ P ′ GL is

k

⋆

(

λ

⋆

)

∈ D

(

ℓ

)

∪ {

ℓ

}}

<!-- formula-not-decoded -->

where k ⊥ ∈ arg min k ∈N ( µ ) η k d k ( µ k , µ ⋆ ) .

## E.2 Terminal conditions for leaves

̸

First consider ℓ a leaf of G . Then five terminal conditions must be satisfied: (i) Since a = 1 requires min v ∈C ( ℓ ) λ v ≥ λ ℓ , we must have a = 1 (ii) If b = 0 then ℓ = k ⋆ ( λ ⋆ ) , so that either a = 0 or ℓ ∈ M ( µ )

̸

̸

(iii) If b = 1 then ℓ = k ⋆ ( λ ⋆ ) , so that a = 0 and ℓ ̸∈ M ( µ )

̸

(iv) If c = 0 then ℓ = k ′ , so that either a = 0 or ℓ ̸∈ M ( µ )

̸

(v) If c = 1 then ℓ = k ′ , so that a = 0 and ℓ ∈ M ( µ ) so that

<!-- formula-not-decoded -->

## E.3 Dynamic programming rules for internal nodes

Now consider ℓ an internal node of G . We wish to compute the value of h recursively using dynamic programming. We first write

<!-- formula-not-decoded -->

where ( z v , a v , b v , c v ) v ∈C ( ℓ ) obeys a set of rules:

(i) If a = 0 then ℓ is a mode of λ , so z v &lt; z for all v ∈ C ( ℓ ) , and a v = 2 , because λ v &lt; λ ℓ = λ p ( v ) (ii) if a = 1 then ℓ has a child with higher value, so there must exist at least one v ∈ C ( ℓ ) with z v ≥ z (iii) If b = 0 then b v = 0 for all v ∈ C ( ℓ ) , since if the subtree of ℓ does not contain k ⋆ ( λ ⋆ ) , then none of the subtrees of v contain k ⋆ ( λ ⋆ )

(iv) If b = 1 , a = 0 and ℓ ̸∈ M ( µ ) , then ℓ = k ⋆ ( λ ⋆ ) , so that none of the subtrees of v contain k ⋆ ( λ ⋆ ) , i.e., b v = 0 for all v ∈ C ( ℓ ) .

̸

(v) If b = 1 and either a ∈ { 1 , 2 } or ℓ ∈ M ( µ ) , then ∑ v ∈C ( ℓ ) b v = 1 , since if the subtree of ℓ contains k ⋆ ( λ ⋆ ) , and ℓ = k ⋆ ( λ ⋆ ) then there must exist exactly one v whose subtree contains k ⋆ ( λ ⋆ ) (vi) If c = 0 and ℓ ∈ M ( µ ) then a = 0 , since if the subtree of ℓ does not contain k ′ then ℓ = k ′ , which gives ℓ ∈ M ( λ ⋆ )

̸

(vii) If c = 0 then c v = 0 for all v ∈ C ( ℓ ) , since if the subtree of ℓ does not contain k ′ then the subtree of all v does not contain k ′

(viii) If c = 1 then either a = { 1 , 2 } and ℓ ∈ M ( µ ) and we have ℓ = k ′ , which implies c v = 0 for all v ∈ D ( ℓ ) , or there must exist exactly one v whose subtree contains k ′ , so that ∑ v ∈D ( ℓ ) c v = 1 (ix) If z v ≤ z then a v = 2 , otherwise a v ∈ { 0 , 1 } .

## E.4 Recursive equations for internal nodes

Based on those rules we compute the value of h ℓ ( z, a, b, c ) recursively using dynamic programming. To do so we define the following auxiliary functions where, as in Section 4.2, the minima are taken with the implicit constraint w ∈ D ( n, µ ) :

<!-- formula-not-decoded -->

where it is noted that X ℓ ( s 1 , s 2 ) = ∅ if min( s 1 , s 2 ) &lt; 0 .

We have h ℓ ( z, a, b, c ) = + ∞ if any of the three conditions hold:

(i) a = 0 , ℓ ̸∈ M ( µ ) and b = 0

̸

(ii) a = 0 , ℓ ̸∈ M ( µ ) , and z = µ ⋆

(iii) a ∈ { 1 , 2 } and ℓ ∈ M ( µ ) and c = 0 .

Indeed, if a = 0 and ℓ ̸∈ M ( µ ) we must have ℓ = k ⋆ ( λ ⋆ ) , so that in turn b = 1 , and λ ℓ = µ ⋆ . Also, if a ∈ { 1 , 2 } and ℓ ∈ M ( µ ) then we must have ℓ = k ′ , which imposes c = 1 .

Otherwise, the value of h ℓ ( z, a, b, c ) is given by the following recursive equations, where by convention, minimization over an empty set yields + ∞ :

̸

h ℓ ( z, 0 , b, c ) = η ℓ d ℓ ( µ ℓ , z ) + min ( b , c ) ∈X ℓ ( b -✶ { ℓ ̸∈M ( µ ) } ,c ) { ∑ v ∈C ( ℓ ) h &lt; v ( z, b v , c v ) } h ℓ ( z, 1 , b, c ) = η ℓ d ℓ ( µ ℓ , z ) + min ( b , c ) ∈X ℓ ( b,c -✶ { ℓ ∈M ( µ ) } ) min w ∈C ( ℓ ) { h ≥ w ( z, b w , c w ) + ∑ v ∈C ( ℓ ) ,v = w h ⋆ v ( z, b v , c v ) } = η ℓ d ℓ ( µ ℓ , z ) + min ( b , c ) ∈X ℓ ( b,c -✶ { ℓ ∈M ( µ ) } ) min w ∈ W ℓ ( z ) { h ≥ w ( z, b w , c w ) + ∑ v ∈C ( ℓ ) ,v = w h ⋆ v ( z, b v , c v ) } h ℓ ( z, 2 , b, c ) = η ℓ d ℓ ( µ ℓ , z ) + min ( b , c ) ∈X ℓ ( b,c -✶ { ℓ ∈M ( µ ) } ) { ∑ v ∈C ( ℓ ) h ⋆ v ( z, b v , c v ) } with W ℓ ( z ) = ∪ ( b,c ) ∈{ 0 , 1 } 2 { arg min w ∈C ( ℓ ) [ h ≥ w ( z, b, c ) -h ⋆ w ( z, b, c ) ]} so that | W ℓ ( z ) | ≤ 4 and where we used the fact that arg min w ∈C ( ℓ ) { h ≥ w ( z, b w , c w ) + ∑ v ∈C ( ℓ ) ,v = w h ⋆ v ( z, b v , c v ) } = arg min w ∈C ( ℓ ) { h ≥ w ( z, b w , c w ) -h ⋆ w ( z, b w , c w ) + ∑ v ∈C ( ℓ ) h ⋆ v ( z, b v , c v ) } = arg min w ∈C ( ℓ ) { h ≥ w ( z, b w , c w ) -h ⋆ w ( z, b w , c w ) } ∈ W ℓ ( z )

̸

## E.5 Fast evaluation of recursive equations

We now propose an efficient strategy to compute the minimization problems in the recursive equations. For any function ϕ , we can minimize ∑ v ∈C ( ℓ ) ϕ v ( b v , c v ) over ( b , c ) ∈ X ℓ ( s 1 , s 2 ) in time and memory O ( |C ( ℓ ) | ) for any ( s 1 , s 2 ) ∈ { 0 , 1 } 2 using the following strategy. If ( s 1 , s 2 ) = (0 , 0) , then trivially

̸

min ( b , c ) ∈X ℓ (0 , 0) { ∑ v ∈C ( ℓ ) ϕ v ( b v , c v ) } = ∑ v ∈C ( ℓ ) ϕ v (0 , 0) If ( s 1 , s 2 ) = (1 , 0) , min ( b , c ) ∈X ℓ (1 , 0) { ∑ v ∈C ( ℓ ) ϕ v ( b v , c v ) } = min b ∈{ 0 , 1 } C ( ℓ ) : ∑ v ∈C ( ℓ ) b v =1 { ∑ v ∈C ( ℓ ) ϕ v ( b v , 0) } = min w ∈C ( ℓ ) { ϕ w (1 , 0) -ϕ w (0 , 0) } + ∑ v ∈C ( ℓ ) ϕ v (0 , 0) and by symmetry, if ( s 1 , s 2 ) = (0 , 1) , min ( b , c ) ∈X ℓ (0 , 1) { ∑ v ∈C ( ℓ ) ϕ v ( b v , c v ) } = min w ∈C ( ℓ ) { ϕ w (0 , 1) -ϕ w (0 , 0) } + ∑ v ∈C ( ℓ ) ϕ v (0 , 0) and finally, if ( s 1 , s 2 ) = (1 , 1) , min ( b , c ) ∈X ℓ (1 , 1) { ∑ v ∈C ( ℓ ) ϕ v ( b v , c v ) } = min(∆ , ∆ ′ ) + ∑ v ∈C ( ℓ ) ϕ v (0 , 0) with ∆ = min w ∈C ( ℓ ) { ϕ w (1 , 1) -ϕ w (0 , 0) } ∆ ′ = min w 1 ,w 2 ∈C ( ℓ ) 2 ,w 1 = w 2 { ϕ w 1 (1 , 0) -ϕ w 1 (0 , 0) + ϕ w 2 (0 , 1) -ϕ w 2 (0 , 0) }

̸

In all cases, one can compute the minimization in time O ( |C ( ℓ ) | ) . In particular ∆ ′ can be computed by realizing that the only pairs ( w 1 , w 2 ) that minimize the expression must be either the first or second smallest entry of ϕ v (1 , 0) -ϕ v (0 , 0) and ϕ v (0 , 1) -ϕ v (0 , 0) . We recall that, finding the first and second smallest entry of a vector can be done in time proportional to the vector size, by inspecting each entry at most twice.

## E.6 Retrieving the optimal solution

̸

Once the value of h ℓ ( z, a, b, c ) has been determined, we can retrieve the optimal solution λ ⋆ ( k ⋆ ( µ ) , µ ⋆ , 0 , 1 , 1) by retrieving the value of the flags ( z ⋆ ℓ , a ⋆ ℓ , b ⋆ ℓ , c ⋆ ℓ ) for all ℓ . Consider a node ℓ = k ⋆ ( µ ) , once its flags ( z ⋆ ℓ , a ⋆ ℓ , b ⋆ ℓ , c ⋆ ℓ ) have been computed, we compute the flags of its children as follows.

(i) If a ⋆ ℓ = 0 , then

<!-- formula-not-decoded -->

(ii) If a ⋆ ℓ = 1 , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

(iii) If a = 2

<!-- formula-not-decoded -->

Finally, a ⋆ v = 2 if z ⋆ ℓ ≥ z ⋆ v and a ⋆ v ∈ arg min a ∈{ 0 , 1 } h v ( z ⋆ v , a, b ⋆ v , c ⋆ v ) otherwise. In practice, these argmins can be stored during the forward pass (where we compute each h ℓ ( z, a, b, c ) ) and do not need to be recomputed.

## E.7 Computational complexity

Recall that z ∈ D ( µ , n ) , which is a grid of size n . If h v ( z, a, b, c ) has been computed for all ( z, a, b, c ) and all v ∈ C ( ℓ ) , then one can compute h &gt; v ( z, b, c ) , h = v ( z, b, c ) , h &lt; v ( z, b, c ) , h ≥ v ( z, b, c ) , h ⋆ v ( z, b, c ) for all ( z, a, b, c ) and all v ∈ C ( ℓ ) in time O ( n |C ( ℓ ) | ) . In turn, using the fast evaluation strategy explained above, one can compute h ℓ ( z, a, b, c ) for all ( z, a, b, c ) in time O ( n |C ( ℓ ) | ) . Therefore, the total time and memory required to solve ˜ P ′ GL with this strategy is O ( n ∑ ℓ |C ( ℓ ) | ) = O ( nK ) since G is a tree.

## E.8 Runtime comparison in practice

The improved dynamic program (DP) runs in time O ( Kn ) , compared to O ( K 2 mn ) for the procedure described in Section 4.2, which we will refer to as the original DP in what follows. In practice,

̸

̸

̸

̸

however, the improved DP may run slower than the original DP on tree instances with moderate values of K . To clarify when one should use each procedure, we report their average runtime over 50 trials on specific tree instances, with η uniformly sampled at random in [0 , 1] K , µ generated as in Section 6 with m = |M ( µ ) | = 3 , σ = 2 , and a discretization parameter of n = 100 . We pick a Gaussian divergence: d k ( λ k , µ k ) = ( λ k -µ k ) 2 / 2 for each k ∈ [ K ] . We perform two experiments:

- (i) We measure runtime on random trees as the number of nodes K increases, with K ∈ { 100 , 400 , 700 , 1000 , 1300 , 1600 , 1900 } .
- (ii) We measure runtime on balanced d -ary trees (i.e., each node has d children) of a fixed height h = 3 . We vary the branching factor d ∈ { 2 , 4 , 6 , 8 , 10 , 12 } , which implicitly varies the number of nodes K from 15 to 1885 .

The results are reported in Figure 6, along with 95% confidence intervals using bootstrap. The left panel shows that for random trees, the original DP is faster on average up to K = 1000 , after which the improved DP generally runs faster. The difference is more pronounced in the right panel, with the average runtime of the original DP increasing much faster with the branching factor d of the balanced tree.

Notably, the original DP exhibits higher runtime variance. This may be explained by an implementation trick we applied to reduce its runtime: specifically, before running the complete dynamic programming subroutine for each k / ∈ N ( µ ) , we check whether η k d k ( µ k , µ ⋆ ) ≥ min ℓ ∈N ( µ ) η ℓ d ℓ ( µ ℓ , µ ⋆ ) . If this holds, the subroutine will not find a parameter with a smaller value than the trivial solution of Proposition 3, hence it is skipped. The number of calls to the subroutine is therefore highly instance-dependent.

Figure 6: Average runtime of each dynamic program with respect to the number of nodes K or the branching factor d .

<!-- image -->