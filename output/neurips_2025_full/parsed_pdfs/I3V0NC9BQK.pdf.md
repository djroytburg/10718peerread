## Scalable Neural Incentive Design with Parameterized Mean-Field Approximation

## Nathan Corecco ∗

Department of Computer Science ETH Zurich nathan.corecco@inf.ethz.ch

## Vinzenz Thoma

Department of Computer Science ETH Zurich &amp; ETH AI Center vinzenz.thoma@ai.ethz.ch

## Batuhan Yardim *

Department of Computer Science ETH Zurich yardima@inf.ethz.ch

## Zebang Shen

Department of Computer Science ETH Zurich zebang.shen@inf.ethz.ch

## Niao He

Department of Computer Science ETH Zurich niao.he@inf.ethz.ch

## Abstract

Designing incentives for a multi-agent system to induce a desirable Nash equilibrium is both a crucial and challenging problem appearing in many decision-making domains, especially for a large number of agents N . Under the exchangeability assumption, we formalize this incentive design (ID) problem as a parameterized mean-field game (PMFG), aiming to reduce complexity via an infinite-population limit. We first show that when dynamics and rewards are Lipschitz, the finiteN ID objective is approximated by the PMFG at rate O ( 1 / √ N ) . Moreover, beyond the Lipschitz-continuous setting, we prove the same O ( 1 / √ N ) decay for the important special case of sequential auctions, despite discontinuities in dynamics, through a tailored auction-specific analysis. Built on our novel approximation results, we further introduce our Adjoint Mean-Field Incentive Design (AMID) algorithm, which uses explicit differentiation of iterated equilibrium operators to compute gradients efficiently. By uniting approximation bounds with optimization guarantees, AMID delivers a powerful, scalable algorithmic tool for many-agent (large N ) ID. Across diverse auction settings, the proposed AMID method substantially increases revenue over first-price formats and outperforms existing benchmark methods.

## 1 Introduction

Setting the right incentives in a game with many participants is a challenging and high-stakes problem. Policymakers must frequently make choices that affect millions, for instance, planners must design rules for curtailing city traffic [46], set pricing to maximize effective bandwidth in telecommunications networks [3], design spectrum auctions between telecom operators [41] or manage supply and demand in energy grids [52].

We study the incentive design (ID) problem. Given an objective G , a parameterized N -player game G , and player strategies π 1 , . . . , π N , ID solves the equilibrium constrained optimization problem:

∗ Equal contribution.

<!-- formula-not-decoded -->

where Nash G ( θ ) denotes the set of Nash equilibria (NE) of a game for the incentive parameter θ (which is to be learned). (ID) is also referred to in the literature as mathematical programming with equilibrium constraints (MPEC) [51]. Despite their relevance, MPECs are notoriously computationally challenging and in the general case NP-hard [39, 51]. Simply computing a Nash equilibrium for a fixed incentive parameter θ is also PPAD-complete [22], even for 2 players [17]. For games with many players, as in many real-world problems, the so-called curse of many agents [71] becomes an added challenge. In such cases, computing Nash G ( θ ) becomes prohibitively expensive, let alone tackling (ID).

In this work, we take the approach of mean-field approximation for games with agent exchangeability (i.e., symmetry) to tackle this problem. Instead of solving (ID) directly, we construct an appropriate mean-field game (MFG) approximation M to G and solve the mean-field ID (MID) problem:

<!-- formula-not-decoded -->

For the mean-field approximation (MID) to be meaningful, its solution should closely match that of the original N -player incentive design problem (ID), with the approximation error vanishing as N →∞ . We formalize this requirement in the following desideratum.

Desideratum D1 (Approximation) The solution of (MID) should be a good (approximate) solution for (ID) , in particular when N is large, with explicit guarantees.

MFGs are known to approximate finite-player NEs with explicit bounds for large N [63, 14, 19, 75]. Under Lipschitz continuity, a non-asymptotic bound of O ( 1 / √ N ) in exploitability is obtained with a propagation-of-chaos type argument. By contrast, our problem (MID) not only contains the classical MFG as a subproblem but also an incentive objective, accordingly, we must show that the approximation error still vanishes as N →∞ . Establishing this result is our first contribution.

Contribution 1: Lipschitz PMFGs and Approximation. We formalize (MID) as a parameterized MFG (PMFG), and show that for Lipschitz PMFGs the (MID) problem approximates (ID) with a rate of O ( 1 / √ N ) , both in exploitability and optimality of the design objective.

While Lipschitz continuous MFGs cover a broad class of real-world games and are well-studied, they exclude many important applications, notably large-scale auction design, where the transition dynamics are inherently correlated and non-Lipschitz. Motivated by the ubiquity and impact of auctions, we analyze (D1) beyond Lipschitz PMFGs, which is our second contribution.

Contribution 2: Approximation beyond Lipschitz. We identify a PMFG for sequential batched auctions (BA-MFG) with many bidders, and establish (D1) with an O ( 1 / √ N ) approximation rate, by identifying a set of 'well-behaved' policies that is dense in the set of Nash equilibria.

Although (D1) guarantees that the mean-field approximation captures the fundamental solution structure of the N -player incentive design problem, realizing practical benefits from this framework requires the following desideratum.

Desideratum D2 (Optimization) We would like (MID) to be computationally tractable or easier to solve than (ID) , specifically, admit an efficient first-order oracle that can be used for optimization.

Our algorithmic contribution is to satisfy this desideratum in both settings: (i) PMFGs under Lipschitz continuity assumptions, and (ii) mean-field auction design.

Contribution 3: Algorithmic Contribution. We formulate our adjoint mean-field incentive design (AMID) algorithm for solving (MID) efficiently. AMID is a modification of backpropagation based on the adjoint method for computing (approximate) derivatives with Nash equilibrium constraints. While naive autodiff-based approaches incur a large O ( T ) memory footprint, our reformulation of the gradient computations reduces the memory footprint to O ( √ T ) , with up to 80% savings in practice.

Contribution 4: Experimental Contribution. Finally, we use AMID to (i) solve congestion pricing in the classical beach bar MFG, and (ii) design revenue-optimal mechanisms across a variety of sequential auctions with a neural network parameterization. Our method consistently outperforms standard first-price mechanisms used in practice and other optimization approaches to solve (MID).

Table 1: Comparison of selected works on ID in literature. Large N : results scale to many-agent (symmetric) G , Dynamic: solves ID on games with dynamics, Explicit diff.: explicit differentiation for first-order information, Approx.: finite-agent approximation in N , Auctions: applies to auctions.

| Work   | Model      | Large N   | Dynamic   | Approx.       | Explicit diff.   | Auctions      |
|--------|------------|-----------|-----------|---------------|------------------|---------------|
| [48]   | VI         | ✗         | ✗         | -             | ✓                | ✗             |
| [50]   | VI         | ✓         | ✗         | -             | ✗                | ✗             |
| [23]   | Cont. time | ✓         | ✓         | ✓             | ✗                | ✗             |
| [54]   | LQ         | ✓         | ✓         | ✓             | ✗                | ✗             |
| [64]   | Maj.-min.  | ✓         | ✓         | ✓             | ✗                | ✗             |
| Ours   | PMFG       | ✓         | ✓         | ✓ - Theorem 1 | ✓ - Section 2.2  | ✓ - Section 3 |

## 1.1 Related Works

We present the works most relevant to this paper, complemented by the discussion in Appendix B.

Mathematical Programming with Equilibrium Constraints (MPEC). Several works have studied gradient-based approaches for MPEC [48, 50, 70, 47, 26]. Some of these assume that the equilibrium problem satisfies strong monotonicity to compute the gradients. Others use explicit differentiation, an approach we also follow in this work. Motivated by the success of reinforcement learning (RL), many works have focused on the optimal design of (multi-agent) RL environments [67, 16, 80, 28, 21, 69]. An important instance of designing games with desirable outcomes is automated mechanism design , capturing many real-world economic problems [18, 20, 49].

Steering and Equilibrium Selection. Complementary to ID problems, a related strand of work has focused on steering and equilibrium selection. For mean-field games, [33] considers a problem of choosing equilibria with high social welfare. Steering learning dynamics towards desirable equilibria was studied by [35, 11, 78] for Markovian and no-regret learners and extended to MFGs by [72].

Mean-Field Games (MFG). MFGs, first formulated in the seminal works of Lasry &amp; Lions [44] and Huang et al. [36], analyze symmetric competitive agents at the many-agent limit. Recently, many works have studied RL in mean-field settings, such as stationary MFGs [32, 77, 73, 19], monotone MFGs [58, 56, 57, 76], static MFGs [74], and mean-field control [15, 5]. While general MFGs remain a theoretical challenge [75], under structured settings, they have shown empirical and algorithmic efficiency [19, 15, 45]. MFGs have also been studied in Stackelberg equilibria, closer to our setting [13, 23, 1, 23]. While these works have similar objectives to ours, rather than letting a leader influence a population through interactions with a static environment, we aim to design parameterized MFGs directly by explicit differentiation. In this sense, (MID) can be seen to differ in objective from these works. Moreover, these results do not readily apply to auction design, a foundational problem for incentive design. In Table 1, we provide a comparison with selected works and our results.

## 2 Designing Games for Large Populations: Lipschitz Case

We first formalize parameterized N -player dynamic games and the corresponding ID problem.

Notation. We use S , A to denote (finite) state-actions spaces. For the horizon H , define policy space Π H := { π : [ H ] ×S → ∆ A } , abbreviate π h ( a | s ) := π ( h, s )( a ) and also treat Π H as a subspace of R [ H ] ×S×A . For a finite set X , define the 'empirical distribution' σ ( x )( x ′ ) = 1 / N ∑ N i =1 ✶ x i = x ′ . We also provide a full reference table for our notation in Appendix A.

Definition 1 (Parameterized Dynamic Games) A finite-horizon parameterized dynamic game (PDG) is a tuple G := ( N, S , A , H, µ 0 , Θ , { P h,θ } H -1 h =0 , { R h,θ } H -1 h =0 ) of players N ∈ N ≥ 1 , discrete state actions sets S , A , parameter space Θ , parameterized transition dynamics P h,θ : S N ×A N → ∆ S N , parameterized reward functions R h,θ : S N ×A N → [0 , 1] N , starting distribution µ 0 ∈ ∆ S , and time horizon H ∈ N &gt; 0 . For a strategy profile π ∈ Π N H , τ ≥ 0 and some θ ∈ Θ , the expected

(entropy-regularized) sum of rewards of player i ∈ [ N ] is defined as

<!-- formula-not-decoded -->

Define E τ G ( π | θ ) := max i ∈ [ N ] E τ,i G ( π | θ ) where E τ,i G ( π | θ ) := max π ′ ∈ Π J τ,i G (( π ′ , π -i ) | θ ) -J τ,i G ( π | θ ) , the exploitability. If E τ G ( π ∗ | θ ) = 0 for π ∗ ∈ Π N H , we call π ∗ a Nash equilibrium (DG-NE) with respect to parameter θ . The set of all Nash equilibria for θ ∈ Θ is denoted Nash τ G ( θ ) . One is typically interested in maximizing a function of the aggregated population behavior (e.g., revenue, negative congestion):

<!-- formula-not-decoded -->

given by some g : Θ × ∆ H S×A → R , with the constraint that π ∈ Π N H is an (approximate) Nash equilibrium under θ (ignoring multiplicities). The parameter space Θ and the parameterizations of P h,θ , R h,θ will dictate the implicit constraints on the design, such as the available information at time h . For such parameterizations, optimizing G will be nontrivial and incorporate an intractable many-agent NE computation as a subproblem. In the following, we reduce this problem to a lower-dimensional MFG (i.e. of size independent of N ) and propose tractable alternatives.

## 2.1 Parameterized Mean Field Game Design

Below, we formalize PMFGs. Definition 2 generalizes the standard definition of MFGs to a parametric family of MFGs, and approximates Definition 1 on a continuum of infinitely many players.

Definition 2 (Parameterized Mean-Field Games) A finite-horizon parameterized mean-field game (PMFG) is a tuple M := ( S , A , H, µ 0 , Θ , { P h,θ } H -1 h =0 , { R h,θ } H -1 h =0 ) of discrete state actions sets S , A , parameterized transition dynamics P h,θ : S × A × ∆ S×A → ∆ S , parameterized reward functions R h,θ : S × A × ∆ S×A → [0 , 1] , initial distribution µ 0 ∈ ∆ S , and horizon H ∈ N &gt; 0 . Define operators Γ h , Λ as Γ h ( L, π h | θ )( s ′ , a ′ ) := ∑ s,a L ( s, a ) P h,θ ( s ′ | s, a, L ) π h ( a ′ | s ′ ) and Λ( π | θ ) := { Γ h -1 ( ··· Γ 1 (Γ 0 ( µ 0 · π 0 , π 1 | θ ) | θ ) ··· , π h -1 ) | θ ) } H -1 h =0 , called population operators 2 . For π ∈ Π H , τ ≥ 0 and L = { L h } H -1 h =0 ∈ ∆ H S×A , the total expected (entropy regularized) reward is

<!-- formula-not-decoded -->

We define mean-field exploitability as E τ M ( π | θ ) := max π ′ ∈ Π V τ M (Λ( π ) , π ′ | θ ) -V τ M (Λ( π ) , π | θ ) . If E τ M ( π ∗ | θ ) = 0 for π ∗ ∈ Π H , we call π ∗ a MFG Nash equilibrium (MFG-NE) with respect to parameter θ . The set of all Nash equilibria for θ ∈ Θ is denoted Nash τ M ( θ ) .

In this section, we will make the following (standard) assumption on P θ ( s ′ | s, a, L ) , R θ ( s, a, L ) , which holds in many relevant applications.

Assumption 1 (Lipschitz continuity) For all s, s ′ ∈ S , a ∈ A , the functions P h,θ ( s ′ | s, a, L ) , R h,θ ( s, a, L ) , and g ( θ, L ) are Lipschitz continuous in θ, L .

Theorem 1 below demonstrates that by optimizing the objective g ( θ, Λ( π ∗ )) , one can obtain approximation guarantees (up to a bias of O ( 1 / √ N ) ) on the performance of the PDG that has independent and symmetric state transitions. We have therefore established (D1) for PMFGs.

Theorem 1 Let M be a PMFG, Assumption 1 hold, and G be the PDG such that P h,θ ( s , a ) := ⊗ i ∈ [ N ] P h,θ ( s i , a i , σ ( s , a )) and R i h,θ ( s , a ) = R h,θ ( s i , a i , σ ( s , a )) for all i . Let π ∗ ∈ Nash τ M ( θ ) and π ∗ := ( π ∗ , . . . , π ∗ ) ∈ Π N H . Then:

<!-- formula-not-decoded -->

2 Note that we define Λ( π | θ ) 0 := µ 0 · π 0

<!-- formula-not-decoded -->

Theorem 1 mirrors bounds in MFGs without ID [75] and Stackelberg MFGs in other settings [54]. For clarity, the theorem is stated as an approximation result for a PDG G that exactly satisfies agent exchangeability, which might not always be the case. In some applications, finding the mean-field formulation M given a PDF G might be nontrivial. The converse problem of constructing an appropriate PMFG M that approximates a given G was studied in [76], and this work can be trivially generalized to this case with an additional approximation bias due to asymmetries in G . The case of auction design, which also does not satisfy Assumption 1 and the assumption of P h,θ being a product measure, will require specific treatment in Section 3.

## 2.2 Approximating the First Order Derivatives

Having satisfied (D1) for ID with Lipschitz dynamics in the previous section, we turn to (D2)formulating algorithmic methods to solve (MID). We state the standard definitions of value and q-functions for PMFGs, which will be important for learning NEs:

<!-- formula-not-decoded -->

We define the commonly used online mirror descent update rule F : Θ × Π → Π

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some given learning rate η &gt; 0 and entropy regularization τ ≥ 0 . F omd has received particular attention in MFG literature due to its theoretical and empirical properties. Abbreviating F ( T ) omd ( θ, ζ ) := F omd ( θ, F omd ( θ, . . . F omd ( θ, ζ ) . . . )) , i.e., F omd ( θ, · ) applied T times, the repeated iterations F ( T ) omd for T &gt; 0 are known to convert to NE for monotone MFGs theoretically [56, 79, 37], and empirically find good approximations to NE [19] for general MFGs. Furthermore, any π ∗ ∈ Nash τ M ( θ ) is guaranteed to be a fixed point of the map F omd ( θ, · ) for some learning rate η . We formulate an explicit differentiation scheme for the PMFG using these properties of F omd. Defining the softmax transform softmax : R [ H ] ×S×A → Π as softmax( ζ )( h, s, a ) := exp { ζ h,s,a } ∑ a ′ exp { ζ h,s,a ′ } , the above OMD update rule can be reformulated in terms of log probabilities:

<!-- formula-not-decoded -->

For fixed θ , the repeated iterations F omd will converge (under technical conditions) to log π ∗ where π ∗ is an NE of the MFG induced by θ . With this motivation, we reformulate the PMFG design problem (MID) as a maximization of the T -step approximate objective

<!-- formula-not-decoded -->

G T approx in general is a well-defined differentiable function (see Lemma 4, Appendix D). In particular, standard autograd tools can be used to compute ∇ G T approx . While the behavior of ∇ G T approx when T → ∞ is not immediate, Lemma 1 below shows that under technical conditions G T approx is a meaningful objective function to maximize, and produces low-bias estimates of the derivatives of the true NE with respect to θ for sufficiently large T .

Lemma 1 (Differentiability of F ∞ omd ) Let ζ ∈ R [ H ] ×S×A , θ ∈ Θ be such that the following hold:

1. F ∞ omd ( θ ′ , ζ ) := lim T →∞ F ( T ) omd ( θ ′ , ζ ) for θ ′ ∈ U for a neighborhood U of θ ,
2. For ζ ∗ such that ζ ∗ = F ∞ omd ( θ, ζ ) , q τ h is C 1 in a neighborhood of ( θ, ζ ∗ ) , and ρ ( ∂ ζ q τ · ( · , ·| Λ(softmax( ζ ∗ )) , softmax( ζ ∗ ) , θ )) &lt; τ for all h ∈ [ H ] .

Then, softmax( F ∞ omd ( θ ′ , ζ )) ∈ Nash τ M ( θ ′ ) on θ ′ ∈ U , F ∞ omd is partially differentiable in θ at ( θ, ζ ) , and lim T →∞ ∂ θ F ( T ) omd ( θ, ζ ) = ∂ θ F ∞ omd ( θ, ζ ) .

Lemma 1 justifies the use of G T approx (and subsequently the explicit differentiation scheme) as an objective function under mild technical conditions. If G is also C 1 , ∇ G T approx converges to the derivative of a map θ ′ → G ( θ ′ , π θ ′ ) where π θ ′ is a function of θ ′ such that π θ ′ ∈ Nash τ M ( θ ′ ) locally, that is, for all θ ′ in some neighborhood of θ . Moreover, Lemma 1 provides intuition on how to tune the parameters η, T, τ and characterizes their impact on the quality of explicit differentiation.

One major challenge from a computational point of view of backpropagating ( T -approx.) will be the size of the computational graph, growing with O ( T ) . In many MFGs, finding a good approximate MFG-NE will require a large T , which will incur a large computational overhead.

Algorithm 1, which we call adjoint mean-field incentive design (AMID), reduces the potentially memory-intensive backpropagation through a complex computational graph to a simple forward and backward pass in t . Crucially, the update operator F is typically quite complicated for PMFGs: for instance, F omd itself involves solving forward ( Λ ) and backward equations ( q τ h ) in h . Therefore, for large T , naive autograd will be inefficient due to the storage of many intermediate values and a large graph.

## Algorithm 1 AMID

| Input:   | Input:                                                  | Update rule F , objective G , T, η, τ, θ, ζ 0           |
|----------|---------------------------------------------------------|---------------------------------------------------------|
| 1:       | for t ∈ 0 , . . .,T do ▷ Forward pass                   | for t ∈ 0 , . . .,T do ▷ Forward pass                   |
| 2:       | ζ t +1 = (1 - ητ ) ζ t + ηF ( θ, ζ t )                  | ζ t +1 = (1 - ητ ) ζ t + ηF ( θ, ζ t )                  |
| 3:       | end for                                                 | end for                                                 |
| 4:       | s T +1 = ∂ θ G ( θ, ζ T +1 ) , a T = - ∂ ζ G ( ζ T +1 ) | s T +1 = ∂ θ G ( θ, ζ T +1 ) , a T = - ∂ ζ G ( ζ T +1 ) |
| 5:       | for t ∈ T, .                                            | . . , 0 do ▷ Backward pass                              |
| 6:       | a t - 1 = (1 - ητ ) a t + ηa t ∂ ζ F ( θ, ζ t )         | a t - 1 = (1 - ητ ) a t + ηa t ∂ ζ F ( θ, ζ t )         |
| 7:       | s t - 1 = s t - ηa t ∂ θ F ( θ, ζ t )                   | s t - 1 = s t - ηa t ∂ θ F ( θ, ζ t )                   |
| 8:       | end for                                                 | end for                                                 |
| 9:       | return s 0                                              | return s 0                                              |

Lemma 2 (Adjoint method) Let Θ ⊂ R d ′ be an open set and F : Θ × R d → R d and G : Θ × R d → R d be differentiable functions. Assume Algorithm 1 is run with inputs F, G , and T ∈ N &gt; 0 , η, τ &gt; 0 , ζ 0 ∈ R d . Then its return value s 0 is equal to ∇ θ G ( θ, F ( T ) ( θ, ζ 0 )) .

Remark 1 In Algorithm 1, { ζ t } t will need to be stored for the backward pass in memory, however, the memory footprint can be reduced to O ( √ T ) by caching every O ( √ T ) timesteps of the forward process and recomputing ζ t as required, maintaining a time complexity of O ( T ) . Furthermore, Algorithm 1 can be generalized to arbitrary Bregman divergences, which permits a variety of inner loop operators to be used (Appendix D.6).

## 3 Beyond Lipschitz: Mechanism Design for Large-Scale Auctions

Auction design is a ubiquitous and well-studied problem of extraordinary economic interest [43]. To analyze auctions at the mean-field regime, we move beyond the Lipschitz PMFG assumptions. Designing auctions with maximum revenue in the resulting equilibrium can be framed as an instance of (ID) (see our discussion in Appendix C), and with an appropriate PMFG, tackled using AMID. Specifically, we consider the following sequential batched auction setting motivated by real-world formats used for selling government debt, broadcast licenses, mineral rights, art, fish, timber [27], and including transactions in mined Ethereum blocks [62].

(Parameterized) batched auctions. An H -round N -player batched auction with incentive parameter θ is a PDG G auc with state space S = V ∪ {⊥} (where the value space V := { 0 , ..., ( |V| 1) / |V| } represents possible valuations and ⊥ denotes non-participation in the current round) and action space A = { 0 , . . . , ( |A| 1) / |A| } (possible bids). Each bidder i ∈ [ N ] at round h ∈ [ H ] has a private state s i h ∈ S not revealed to the auctioneer or other bidders. Overall, the auctioneer sells at most ⌈ α max N ⌉ goods (for some α max ∈ (0 , 1) ) and chooses θ , parameterizing the allocations and payments as outlined below. The auction evolves for h ∈ [ H ] as follows:

1. Initial states at h = 0 are independently sampled from distribution µ 0 ∈ ∆ V .

̸

3. Observing the bid distributions ̂ ν -⊥ h := 1 N ∑ i ∈ [ N ] e a i h ✶ s i h = ⊥ , the mechanism decides on a ratio of goods to be sold this round, α θ h ( ̂ ν -⊥ h ) . Items are allocated to the highest ⌊ α θ h ( ̂ ν -⊥ h ) N ⌋ bidders, with ties broken uniformly at random.
2. At every round h ∈ [ H ] , bidders for which s i h = ⊥ submit their bids a i h ∈ A .

̸

4. Each bidder i who receives an item, makes a payment p i h = p θ h ( a i h , ̂ ν -⊥ h ) ∈ R ≥ 0 . A winning bidder receives utility u h ( s i h , p i h ) ∈ R and transitions to state ⊥ , while non-winning and nonparticipating bidders receive zero utility.

5. Before proceeding to round h +1 , each bidder transitions independently to a new state according to a dynamics function w h : S × ∆ S → ∆ S , which maps the agent's current state and the empirical population distribution (after the allocation at round h ) to a distribution over next states.

To ensure the mechanism can not sell more goods than are available ( α max), we assume that the parameterizations of α θ h are such that ∑ h α θ h ( ̂ ν -⊥ h ) ≤ α max almost surely. This is can be ensured, for example, by parameterizing α θ h as a fraction of remaining goods at every h . G auc allows for complex valuation dynamics, such as single-minded bidders (who stay in ⊥ ), time-dependent or population-dependent evolving valuations, as well as super and subadditive valuations over bundles of goods. Under these dynamics, we denote the expected utility of player i and exploitability as J τ,i auc , E τ,i auc respectively, as defined in Definition 1.

We note that parameterizing α θ h , p θ h fully captures the intuition of reserve prices in the BA-MFG setting. In many auction formats, a reserve price, i.e., a minimum price that bidders have to bid and pay to win, has been shown to increase revenue [55]. A reserve price r h ∈ A at round h can be implemented for example with α h ( ν ) = ∑ a ′ ≥ r h ν ( a ′ ) and p h ( a, ν ) = a .

## 3.1 Auctions at the Mean-Field Regime

From the above description, G auc clearly has exchangeable agents. However, the corresponding one-step state evolutions are not independent, making Theorem 1 inapplicable here. Motivated by the relevance of large-scale auction design, we show that PMFGs are still relevant and (D1) holds with a refined analysis of batched auctions in the following. We begin by defining the correct MFG that characterizes the batch auction at the limit N →∞ . While the definition is symbol-laden, we state it for completeness.

Definition 3 (BA-MFG) A Batched-Auction MFG (BA-MFG) is the PMFG M mfa := ( S , A , H, µ 0 , Θ , { P mfa h,θ } H -1 h =0 , { R mfa h,θ } H -1 h =0 ) , where R mfa θ,h , P mfa θ,h are given by:

<!-- formula-not-decoded -->

̸

where ν -⊥ ( L ) := ∑ s = ⊥ L ( s, · ) , ξ L,θ ∈ ∆ S such that ξ L,θ ( ⊥ ) = L ( ⊥ ) + ⟨ L, p win ( · , · , L, α θ h ( ν -⊥ ( L ))) ⟩ and ξ ( s ) = ⟨ L ( s, · ) , p win ( s, · , L, α θ h ( ν -⊥ ( L ))) ⟩ , and p win defined as 3

̸

We use V τ mfa , E τ mfa to denote expected reward and exploitability in M mfa, as in Definition 2.

<!-- formula-not-decoded -->

The intuition behind Definition 3 relies on the fact that the function p win approximately characterizes the marginal winning probability of an agent when N is large. In fact, Theorem 2 below shows that BA-MFG is indeed the correct model for auctions with large N . Existing approximation results (such as [63, 75] as well as Theorem 1) fundamentally are incompatible with this setting due to (1) the fact that transitions in (finite-player) auctions are not independent, and (2) due to the inherent jump discontinuities in both P auc θ,h , R auc θ,h . No zero-dominance (NZD) policies, defined below, identify a subset of policy space Π H where this difficulty can be circumvented.

Definition 4 (No zero-dominance (NZD)) Let M mfa be a BA-MFG and θ ∈ Θ . π ∈ Π H is said to satisfy the NZD property for θ if at induced L = { L h } H -1 h =0 = Λ mfa ( π | θ ) there exist no a ∈ A , h ∈ [ H ] such that ∑ s ∈V L h ( s, a ) = 0 and ∑ s ∈V ,a ′ &gt;a L h ( s, a ′ ) = α θ h ( ν -⊥ ( L )) .

While NZD is a technical condition, it is for instance satisfied by any entropy regularized MFG-NE of M mfa if τ &gt; 0 , therefore, contains ε -NE for arbitrarily small ε &gt; 0 . With this property, BA-MFGs satisfy (D1) as shown below, making it a relevant model for auction design.

3 In this definition we take ε / 0 = ∞ , for any ε &gt; 0 and 0 / 0 = 0 , for convenience.

Theorem 2 (Approximation for BA-MFG) Let M mfa be a BA-MFG with Lipschitz-continuous u h , w h , α θ h , p θ h and let g : Θ × ∆ H S×A → R be Lipschitz. Let π ∈ Π H be a policy that satisfies the no zero-dominance property with respect to θ ∈ Θ . Then, for π = ( π, . . . , π ) ∈ Π N H ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof sketch: The proof builds on (1) special handling of the correlated evolution of s i h at any round h and (2) showing that for non-zero dominant policies π , the dynamics are locally Lipschitz continuous. The two conclusions are proved separately in Appendices E.3 and E.4. □

Theorem 2 demonstrates convergence for a broad class of policies and relates to a large strand of work on equilibrium computation for auctions, which we discuss in Appendix B. While a true MF-NE does not necessarily satisfy the no zero-dominance property, an entropy-regularized MF-NE does. In this regard, the results above show that the BA-MFG essentially characterizes the limiting behavior of batched auctions.

Remark 2 In general, Theorem 2 incorporates a standard worst-case exponential bound in H , depending on w h , π . However, in certain cases, such as non-expansive or population-independent w h and π with full support, the bound becomes polynomial in H, |S| , |A| (see Appendix E.3). We later verify the quality of the bound in real-world experiments.

Finally, we state the following differentiability result of F τ omd , thus satisfying (D2) when combined with the adjoint method described in Section 2.2. This result permits mechanism design by backpropagation for any entropy regularization τ &gt; 0 , completing the motivation for BA-MFG.

Lemma 3 (Differentiability on G auc ) Let M mfa be a BA-MFG on an open parameter space Θ ⊂ R d , with Lipschitz u h , w h , α θ h , p θ h , then F τ omd is almost everywhere differentiable on R [ H ] ×S×A × Θ .

Equipped with an algorithmic tool to solve large-scale ID problems, we move to empirical demonstration on applications.

## 4 Experimental Results

Weevaluate our methodology on numerical examples of increasing complexity, using AMID to obtain gradient estimates and ADAM [40] as an update rule on parameters θ . All experiment details, including computational resources, can be found in Appendix F. We also provide reference implementations in JAX and PyTorch 4 .

First, we demonstrate the effectiveness of our approach on the prototypical MFG of the beach bar game [58]. We formulate the PMFG M bb, where a large population of beachgoers starting from µ 0 = Uniform( S ) can move left, stay, or move right ( A : {-1 , 0 , 1 } ) on a beach ( S := [ K ] ) over H steps, while trying to minimize their distance to the bar located at s bar = K / 2 and avoiding busy spots. We parameterize a pricing mechanism θ ∈ [0 , 1 / 2 ] S for spots on the beach to minimize congestion (the softmax of population flow):

<!-- formula-not-decoded -->

We report the training curves and induced flows in Figure 1, along with θ ∗ in Appendix F.

4 The PyTorch implementation was adapted from MFGLib[31].

Figure 1: Payment design with AMID in M bb. Left: objective and exploitability throughout training iterations. Middle-right: population flow in time before and after learning payments.

<!-- image -->

Dynamic auction settings. We move to the more challenging but relevant setting of designing neural network mechanisms BA-MFGs. We focus on risk-neutral bidders ( u h linear in payments) and on direct revelation mechanisms, i.e. S = A . 5 We set |S| = 100 , and maximize the revenue objective:

<!-- formula-not-decoded -->

The exact settings, labeled (A1)-(A3) are as follows:

- (A1) H = 4 , µ 0 = Uniform( S ) , α max = 0 . 8 , and single-minded bidders (after winning stay at state ⊥ ) with no evolution in valuations s i h otherwise.
- (A2) H = 4 , α max = 0 . 8 , µ 0 ( s ) ∝ γ s for γ = 0 . 9 , dynamic values with w ( s ′ | s ) ∝ exp { -(3 s -s ′ ) 2 / 2 σ 2 } for σ = 0 . 2 , bidders are single-minded.
- (A3) H = 5 , µ 0 uniformly sampled from ∆ S , α max ∼ Uniform ([0 . 6 , 1 . 2]) , participants re-enter with probability 0 . 3 each round. 6

We parameterize p θ h , α θ h with a residual neural network (architecture clarified in Appendix F) containing ≈ 2 × 10 5 parameters, with inputs e h , ν -⊥ h , and remaining unsold goods at round h , guaranteeing by parameterization that no more than α max goods are sold in total.

Baselines. We evaluate AMID against several benchmarks. First, we compare against the results of running a simple first-price auction (FIRSTPRICE), i.e., the highest bidders win and pay what they bid, to see how much more revenue we can achieve from optimizing over α θ h and p θ h . Second, we contrast with various methods without gradient information: two methods using two-point gradient estimators ( 0 -ADAM and 0 -SGD respectively), and a 0-order annealing strategy (ANNEAL) using random perturbations of θ . We use τ = 10 -3 , η = 10 and T = 400 for computing objective G T approx . We report the training curves in Figure 2, where we evaluate G T val approx throughout training T val = 500 for robustness. The results indicate the effectiveness of our method against zeroth-order methods across different auctions. Evaluations on a larger variety of settings and parameterizations (longer horizons H , nonlinear utilities, static mechanisms, other g ) are also reported in Appendix F.

Empirically Testing (D1) &amp; (D2) . Figure 3 illustrates that we fulfill (D1) &amp; (D2). Notably, the actual revenue in the N player auction is very close to the optimized g rev even for N ≈ 100 . Furthermore, the exploitability curve of OMD iterates at the optimized θ ∗ also suggests that the iterates are a good approximation of MF-NE, and empirically, the assumptions of Lemma 1 are valid. Namely, OMD iterations produce a valid approximate Nash equilibrium after the end-to-end optimization process with AMID, empirically verifying that the revenue at Nash is indeed optimized.

We further provide empirical evidence supporting (D2) by comparing the computational footprint of AMID against naive backpropagation through the full computational graph induced by OMD. In

5 The latter choice is motivated by the conceptual simplicity, widespread use in practice, and does not represent a significant restriction given the revelation principle [43].

6 The setting is more challenging for two reasons. First, the neural mechanism must generalize over α max, which it can observe. Second, it must generalize over all µ 0 , which it does not observe, but potentially infer from ν -⊥ ( L ) . This is also referred to as prior-free mechanism design [29, 34]

1

Figure 2: g rev throughout iterations of AMID and baseline algorithms in settings (A1-3), left to right.

<!-- image -->

Table 2, we report the time and memory usage of the two methods when solving (A1) with increasing time horizons H on a single H100. The results are reported for a single backpropagation step. The modest growth in memory and computation time observed for AMID as H increases highlights the scalability and practical suitability of our methodology for solving large-scale ID problems.

1

Figure 3: Left: deviation in revenue in M mfa vs N -player G auc at θ ∗ as functions of N , and middle: exploitability curve of OMD iterations F ( T ) omd ( θ ∗ , · ) at optimized θ ∗ in (A1), (A2), (A3). Right: mean bids of NE computed by F ( T ) omd for h ∈ [4] before and after optimization with AMID in ( A 1) .

<!-- image -->

Table 2: Empirical compute time and memory usage for single-step naive backpropagation vs. singlestep AMID across different problem horizons, in setting (A1). The rows with 'n/a' indicate the method did not run on a single H100.

| Horizon   | Naive (time, s)   | Naive (memory)   | AMID (time, s)    | AMID (memory)   |
|-----------|-------------------|------------------|-------------------|-----------------|
| H =5      | 0 . 19 ± 0 . 02 s | 2760 MiB         | 0 . 09 ± 0 . 01 s | 560 MiB         |
| H =10     | 0 . 25 ± 0 . 10 s | 8746 MiB         | 0 . 21 ± 0 . 08 s | 586 MiB         |
| H =25     | 0 . 71 ± 0 . 15 s | 16960 MiB        | 0 . 67 ± 0 . 12 s | 826 MiB         |
| H =50     | n/a               | n/a              | 1 . 72 ± 0 . 41 s | 1076 MiB        |

## 5 Conclusion

In this work, we presented a novel method for ID relying on PMFGs. In particular, we set forth two desiderata in order to use scalable first-order optimization to approximately solve ID problems. Through new analyses, we demonstrated that these conditions hold in both classical Mean Field Game (MFG) settings and batched auction environments. For both settings, we presented a unified algorithm, called AMID, which can solve a span of ID problems, such as congestion pricing or optimal auction design. Overall, the AMID framework offers a flexible foundation for diverse incentive design applications, paving the way for future extensions.

## Acknowledgments and Disclosure of Funding

This project was supported by Swiss National Science Foundation (SNSF) under the framework of NCCR Automation and SNSF Starting Grant. V. Thoma acknowledges funding from the Swiss National Science Foundation (SNSF) Project Funding No. 200021-207343 and is supported by an ETH AI Center Doctoral Fellowship.

## References

- [1] Alexander Aurell, Rene Carmona, Gokce Dayanikli, and Mathieu Lauriere. Optimal incentives to mitigate epidemics: a stackelberg mean field game approach. SIAM Journal on Control and Optimization , 60(2):S294-S322, 2022.
- [2] Santiago R. Balseiro, Omar Besbes, and Gabriel Y. Weintraub. Repeated auctions with budgets in ad exchanges: Approximations and design. Management Science , 61(4):864-884, 2015.
- [3] Tamer Basar and Rayadurgam Srikant. Revenue-maximizing pricing and capacity expansion in a many-users regime. In Proceedings. Twenty-First Annual Joint Conference of the IEEE Computer and Communications Societies , volume 1, pages 294-301. IEEE, 2002.
- [4] Alain Bensoussan, Michael HM Chau, and Sheung Chi Phillip Yam. Mean field stackelberg games: Aggregation of delayed instructions. SIAM Journal on Control and Optimization , 53(4):2237-2266, 2015.
- [5] Alain Bensoussan, Jens Frehse, Phillip Yam, et al. Mean field games and mean field type control theory , volume 101. Springer, 2013.
- [6] Martin Bichler, Maximilian Fichtl, Stefan Heidekrüger, Nils Kohring, and Paul Sutterer. Learning equilibria in symmetric auction games using artificial neural networks. Nature Machine Intelligence , 3(8):687-695, August 2021.
- [7] Martin Bichler, Maximilian Fichtl, and Matthias Oberlechner. Computing bayes nash equilibrium strategies in auction games via simultaneous online dual averaging. In Proceedings of the 24th ACM Conference on Economics and Computation , EC '23, page 294, New York, NY, USA, 2023. Association for Computing Machinery.
- [8] Vitor Bosshard, Benedikt Bünz, Benjamin Lubin, and Sven Seuken. Computing Bayes-Nash equilibria in combinatorial auctions with continuous value and action spaces. In Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI) , Melbourne, Australia, August 2017.
- [9] Vitor Bosshard, Benedikt Bünz, Benjamin Lubin, and Sven Seuken. Computing bayes-nash equilibria in combinatorial auctions with verification. Journal of Artificial Intelligence Research , 69:531-570, 2020.
- [10] Yang Cai and Christos Papadimitriou. Simultaneous bayesian auctions and computational complexity. In Proceedings of the Fifteenth ACM Conference on Economics and Computation , EC '14, page 895-910, New York, NY, USA, 2014. Association for Computing Machinery.
- [11] Ilayda Canyakmaz, Iosif Sakos, Wayne Lin, Antonios Varvitsiotis, and Georgios Piliouras. Learning and steering game dynamics towards desirable outcomes. In Proceedings of the 7th Annual Learning for Dynamics &amp; Control Conference , volume 283 of Proceedings of Machine Learning Research , pages 1512-1524. PMLR, 04-06 Jun 2025.
- [12] René Carmona and Gökçe Dayanıklı. Mean field game model for an advertising competition in a duopoly. International Game Theory Review , 23(04):2150024, 2021.
- [13] René Carmona, Gökçe Dayanıklı, and Mathieu Laurière. Mean field models to regulate carbon emissions in electricity production. Dynamic Games and Applications , 12(3):897-928, 2022.
- [14] René Carmona and François Delarue. Probabilistic analysis of mean-field games. SIAM Journal on Control and Optimization , 51(4):2705-2734, 2013.

- [15] René Carmona, Mathieu Laurière, and Zongjun Tan. Model-free mean-field reinforcement learning: mean-field mdp and mean-field q-learning. The Annals of Applied Probability , 33(6B):5334-5381, 2023.
- [16] Siyu Chen, Donglin Yang, Jiayang Li, Senmiao Wang, Zhuoran Yang, and Zhaoran Wang. Adaptive model design for Markov decision process. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning , volume 162 of Proceedings of Machine Learning Research , pages 3679-3700. PMLR, 17-23 Jul 2022.
- [17] Xi Chen, Xiaotie Deng, and Shang-Hua Teng. Settling the complexity of computing two-player nash equilibria. Journal of the ACM (JACM) , 56(3):1-57, 2009.
- [18] Vincent Conitzer and Tuomas Sandholm. Self-interested automated mechanism design and implications for optimal combinatorial auctions. In Proceedings of the 5th ACM Conference on Electronic Commerce , EC '04, page 132-141, New York, NY, USA, 2004. Association for Computing Machinery.
- [19] Kai Cui and Heinz Koeppl. Approximately solving mean field games via entropy-regularized deep reinforcement learning. In International Conference on Artificial Intelligence and Statistics , pages 1909-1917. PMLR, 2021.
- [20] Michael Curry, Vinzenz Thoma, Darshan Chakrabarti, Stephen McAleer, Christian Kroer, Tuomas Sandholm, Niao He, and Sven Seuken. Automated design of affine maximizer mechanisms in dynamic settings. Proceedings of the AAAI Conference on Artificial Intelligence , 38(9):9626-9635, March 2024.
- [21] Michael Curry, Alexander R Trott, Soham Phade, Yu Bai, and Stephan Zheng. Finding general equilibria in many-agent economic simulations using deep reinforcement learning, 2022.
- [22] Constantinos Daskalakis, Paul W Goldberg, and Christos H Papadimitriou. The complexity of computing a nash equilibrium. Communications of the ACM , 52(2):89-97, 2009.
- [23] Gokce Dayanikli and Mathieu Lauriere. A machine learning method for stackelberg mean field games. arXiv preprint arXiv:2302.10440 , 2023.
- [24] Greg d'Eon, Neil Newman, and Kevin Leyton-Brown. Understanding iterative combinatorial auction designs via multi-agent reinforcement learning. In Proceedings of the 25th ACM Conference on Economics and Computation , EC '24, page 1102-1130, New York, NY, USA, 2024. Association for Computing Machinery.
- [25] Stylianos Despotakis, R. Ravi, and Amin Sayedi. First-price auctions in online display advertising. Journal of Marketing Research , 58(5):888-907, October 2021.
- [26] Tanner Fiez, Benjamin Chasnov, and Lillian Ratliff. Implicit learning dynamics in stackelberg games: Equilibria characterization, convergence analysis, and empirical study. In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning , volume 119 of Proceedings of Machine Learning Research , pages 3133-3144. PMLR, 13-18 Jul 2020.
- [27] Ian Gale and Mark Stegeman. Sequential auctions of endogenously valued objects. Games and Economic Behavior , 36:74-103, 07 2001.
- [28] Matthias Gerstgrasser and David C. Parkes. Oracles &amp; followers: Stackelberg equilibria in deep multi-agent reinforcement learning. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 11213-11236. PMLR, 23-29 Jul 2023.
- [29] Andrew V. Goldberg, Jason D. Hartline, and Andrew Wright. Competitive auctions and digital goods. page 735 - 744, 2001. Cited by: 199.

- [30] Amy Greenwald, Jiacui Li, and Eric Sodomka. Approximating equilibria in sequential auctions with incomplete information and multi-unit demand. In F. Pereira, C.J. Burges, L. Bottou, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems , volume 25. Curran Associates, Inc., 2012.
- [31] X. Guo, A. Hu, M. Santamaria, M. Tajrobehkar, and J. Zhang. MFGLib: A library for mean field games. arXiv preprint arXiv:2304.08630 , 2023.
- [32] Xin Guo, Anran Hu, Renyuan Xu, and Junzi Zhang. Learning mean-field games. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 32. Curran Associates, Inc., 2019.
- [33] Xin Guo, Lihong Li, Sareh Nabi, Rabih Salhab, and Junzi Zhang. Mesob: Balancing equilibria &amp;social optimality, 2023.
- [34] Jason D. Hartline and Tim Roughgarden. Optimal mechanism design and money burning. In Proceedings of the Fortieth Annual ACM Symposium on Theory of Computing , STOC '08, page 75-84, New York, NY, USA, 2008. Association for Computing Machinery.
- [35] Jiawei Huang, Vinzenz Thoma, Zebang Shen, Heinrich H. Nax, and Niao He. Learning to steer markovian agents under model uncertainty. In The Thirteenth International Conference on Learning Representations , 2025.
- [36] Minyi Huang, Roland P Malhamé, and Peter E Caines. Large population stochastic dynamic games: closed-loop mckean-vlasov systems and the nash certainty equivalence principle. Communications in Information &amp; Systems , 6(3):221-252, 2006.
- [37] Noboru Isobe, Kenshi Abe, and Kaito Ariu. Last iterate convergence in monotone mean field games. arXiv preprint arXiv:2410.05127 , 2024.
- [38] Krishnamurthy Iyer, Ramesh Johari, and Mukund Sundararajan. Mean field equilibria of dynamic auctions with learning. Management Science , 60(12):2949-2970, dec 2014.
- [39] Robert G. Jeroslow. The polynomial hierarchy and a simple model for competitive analysis. Mathematical Programming , 32(2):146-164, June 1985.
- [40] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Yoshua Bengio and Yann LeCun, editors, 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings , 2015.
- [41] Paul Klemperer. How (not) to run auctions: The european 3g telecom auctions. European Economic Review , 46(4-5):829-845, May 2002.
- [42] Nils Kohring, Fabian Raoul Pieroth, and Martin Bichler. Enabling first-order gradient-based learning for equilibrium computation in markets. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 17327-17342. PMLR, 23-29 Jul 2023.
- [43] Vijay Krishna. Auction Theory . Academic Press, 2002.
- [44] Jean-Michel Lasry and Pierre-Louis Lions. Mean field games. Japanese journal of mathematics , 2(1):229-260, 2007.
- [45] Mathieu Laurière, Sarah Perrin, Julien Pérolat, Sertan Girgin, Paul Muller, Romuald Élie, Matthieu Geist, and Olivier Pietquin. Learning in mean field games: A survey. arXiv preprint arXiv:2205.12944 , 2022.
- [46] Ana Ley, Winnie Hu, and Keith Collins. Less traffic, faster buses: Congestion pricing's first week. The New York Times , January 2025. Accessed: 2025-15-01.
- [47] Jiayang Li, Jing Yu, Boyi Liu, Yu Nie, and Zhaoran Wang. Achieving hierarchy-free approximation for bilevel programs with equilibrium constraints. In Proceedings of the 40th International Conference on Machine Learning , ICML'23, Honolulu, Hawaii, USA, 2023. JMLR.org.

- [48] Jiayang Li, Jing Yu, Yu Nie, and Zhaoran Wang. End-to-end learning and intervention in games. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems , volume 33, pages 16653-16665. Curran Associates, Inc., 2020.
- [49] Anton Likhodedov and Tuomas Sandholm. Methods for boosting revenue in combinatorial auctions. In Proceedings of the 19th National Conference on Artifical Intelligence , AAAI'04, page 232-237, San Jose, California, 2004. AAAI Press.
- [50] Boyi Liu, Jiayang Li, Zhuoran Yang, Hoi-To Wai, Mingyi Hong, Yu Nie, and Zhaoran Wang. Inducing equilibria via incentives: Simultaneous design-and-play ensures global convergence. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems , volume 35, pages 29001-29013. Curran Associates, Inc., 2022.
- [51] Zhi-Quan Luo, Jong-Shi Pang, and Daniel Ralph. Mathematical Programs with Equilibrium Constraints . Cambridge University Press, nov 1996.
- [52] Sabita Maharjan, Quanyan Zhu, Yan Zhang, Stein Gjessing, and Tamer Basar. Dependable demand response management in the smart grid: A stackelberg game approach. IEEE Transactions on Smart Grid , 4(1):120-132, 2013.
- [53] Carlos Martin and Tuomas Sandholm. Finding mixed-strategy equilibria of continuous-action games without gradients using randomized policy networks. In Edith Elkind, editor, Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI-23 , pages 2844-2852. International Joint Conferences on Artificial Intelligence Organization, 8 2023. Main Track.
- [54] Jun Moon and Tamer Ba¸ sar. Linear quadratic mean field stackelberg differential games. Automatica , 97:200-213, 2018.
- [55] Roger B. Myerson. Optimal auction design. Mathematics of Operations Research , 6(1):58-73, 1981.
- [56] Julien Pérolat, Sarah Perrin, Romuald Elie, Mathieu Laurière, Georgios Piliouras, Matthieu Geist, Karl Tuyls, and Olivier Pietquin. Scaling mean field games by online mirror descent. In Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems , pages 1028-1037, 2022.
- [57] Sarah Perrin, Mathieu Laurière, Julien Pérolat, Romuald Élie, Matthieu Geist, and Olivier Pietquin. Generalization in mean field games by learning master policies. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 36, pages 9413-9421, 2022.
- [58] Sarah Perrin, Julien Pérolat, Mathieu Laurière, Matthieu Geist, Romuald Elie, and Olivier Pietquin. Fictitious play for mean field games: Continuous time analysis and applications. Advances in Neural Information Processing Systems , 33:13199-13213, 2020.
- [59] Fabian R. Pieroth, Nils Kohring, and Martin Bichler. Equilibrium computation in multi-stage auctions and contests. ArXiv , abs/2312.11751, 2023.
- [60] Zinovi Rabinovich, Victor Naroditskiy, Enrico H. Gerding, and Nicholas R. Jennings. Computing pure bayesian-nash equilibria in games with finite actions and continuous types. Artificial Intelligence , 195:106-139, February 2013.
- [61] Daniel M. Reeves and Michael P. Wellman. Computing best-response strategies in infinite games of incomplete information. In Proceedings of the 20th Conference on Uncertainty in Artificial Intelligence , UAI '04, page 470-478, Arlington, Virginia, USA, 2004. AUAI Press.
- [62] Tim Roughgarden. Transaction fee mechanism design. In Proceedings of the 22nd ACM Conference on Economics and Computation , EC '21, page 792, New York, NY, USA, 2021. Association for Computing Machinery.
- [63] Naci Saldi, Tamer Basar, and Maxim Raginsky. Markov-nash equilibria in mean-field games with discounted cost. SIAM Journal on Control and Optimization , 56(6):4256-4287, 2018.

- [64] Sina Sanjari, Subhonmesh Bose, and Tamer Ba¸ sar. Incentive designs for stackelberg games with a large number of followers and their mean-field limits. Dynamic Games and Applications , 15(1):238-278, 2025.
- [65] Vinzenz Thoma, Vitor Bosshard, and Sven Seuken. Computing perfect bayesian equilibria in sequential auctions with verification. Proceedings of the AAAI Conference on Artificial Intelligence , 39(13):14158-14166, April 2025.
- [66] Vinzenz Thoma, Michael Curry, Niao He, and Sven Seuken. Learning best response policies in dynamic auctions via deep reinforcement learning, 2023.
- [67] Vinzenz Thoma, Barna Pasztor, Andreas Krause, Giorgia Ramponi, and Yifan Hu. Contextual bilevel reinforcement learning for incentive alignment. In Advances in Neural Information Processing Systems , volume 37, pages 127369-127435, 2024.
- [68] Yevgeniy Vorobeychik and Michael P. Wellman. Stochastic search methods for nash equilibrium approximation in simulation-based games. In Proceedings of the 7th International Joint Conference on Autonomous Agents and Multiagent Systems - Volume 2 , AAMAS '08, page 1055-1062, Richland, SC, 2008. International Foundation for Autonomous Agents and Multiagent Systems.
- [69] Jing Wang, Meichen Song, Feng Gao, Boyi Liu, Zhaoran Wang, and Yi Wu. Differentiable arbitrating in zero-sum markov games. In Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems , AAMAS '23, page 1034-1043, Richland, SC, 2023. International Foundation for Autonomous Agents and Multiagent Systems.
- [70] Kai Wang, Lily Xu, Andrew Perrault, Michael K. Reiter, and Milind Tambe. Coordinating followers to reach better equilibria: End-to-end gradient descent for stackelberg games. Proceedings of the AAAI Conference on Artificial Intelligence , 36(5):5219-5227, June 2022.
- [71] Lingxiao Wang, Zhuoran Yang, and Zhaoran Wang. Breaking the curse of many agents: Provable mean embedding q-iteration for mean-field reinforcement learning. In International conference on machine learning , pages 10092-10103. PMLR, 2020.
- [72] Leo Widmer, Jiawei Huang, and Niao He. Steering no-regret agents in mfgs under model uncertainty, 2025.
- [73] Batuhan Yardim, Semih Cayci, Matthieu Geist, and Niao He. Policy mirror ascent for efficient and independent learning in mean field games. In International Conference on Machine Learning , pages 39722-39754. PMLR, 2023.
- [74] Batuhan Yardim, Semih Cayci, and Niao He. A variational inequality approach to independent learning in static mean-field games. ACM / IMS Journal of Data Science , 2(2), July 2025.
- [75] Batuhan Yardim, Artur Goldman, and Niao He. When is mean-field reinforcement learning tractable and relevant? In Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems , AAMAS '24, page 2038-2046, Richland, SC, 2024. International Foundation for Autonomous Agents and Multiagent Systems.
- [76] Batuhan Yardim and Niao He. Exploiting approximate symmetry for efficient multi-agent reinforcement learning. In Proceedings of the 7th Annual Learning for Dynamics &amp; Control Conference , volume 283 of Proceedings of Machine Learning Research , pages 31-44. PMLR, 04-06 Jun 2025.
- [77] Muhammad Aneeq Uz Zaman, Alec Koppel, Sujay Bhatt, and Tamer Basar. Oracle-free reinforcement learning in mean-field games along a single sample path. In International Conference on Artificial Intelligence and Statistics , pages 10178-10206. PMLR, 2023.
- [78] Brian Hu Zhang, Gabriele Farina, Ioannis Anagnostides, Federico Cacciamani, Stephen McAleer, Andreas Haupt, Andrea Celli, Nicola Gatti, Vincent Conitzer, and Tuomas Sandholm. Steering no-regret learners to a desired equilibrium. In Proceedings of the 25th ACM Conference on Economics and Computation , EC '24, page 73-74, New York, NY, USA, 2024. Association for Computing Machinery.

- [79] Fengzhuo Zhang, Vincent YF Tan, Zhaoran Wang, and Zhuoran Yang. Learning regularized monotone graphon mean-field games. arXiv preprint arXiv:2310.08089 , 2023.
- [80] Stephan Zheng, Alexander Trott, Sunil Srinivasa, David C. Parkes, and Richard Socher. The ai economist: Taxation policy design via two-level deep multiagent reinforcement learning. Science Advances , 8(18), May 2022.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The claimed convergence rate is proven in the paper and the experimental results are accurately described.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We state our assumptions in Section 2.

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

Justification: See Assumption 2 and Assumption 1 and the extensive appendix with all proofs.

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

Justification: We introduce and formally define our approximation framework in Definition 2 and describe the algorithmic procedure using pseudocode in Algorithm 1. The experimental setting (Batched Auction MFG) (Definition 3) and objectives are precisely specified. The various experiments where we apply our framework are clearly defined in Section 4, and full details are presented in the appendix.

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

Justification: We provide code with both JAX and PyTorch implementations, along with instructions to execute the experiments. All experiments are seeded to ensure reproducibility. The main experiments are implemented using the JAX codebase.

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

Justification: All experimental settings are documented in Section 4. Details on the hyperparameters and experimental setup are provided in Appendix F, and the accompanying code includes the full configuration used for each experiment.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All experiments include error bars.

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

Justification: We provide detailed information about computing resources in Appendix F.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We use neither human subjects nor sensitive data, and we do not foresee dual use of our work.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Not applicable, no immediate social impact.

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

Justification: Not applicable.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We used JAX, PyTorch, and adapted code from MFGLib. No additional third-party datasets or pretrained models were used. Everything is properly cited.

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

Justification: Not applicable.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Not applicable.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Not applicable.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: Not applicable.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Contents

| 1 Introduction                                                                         | 1   | 1 Introduction                                                                         |
|----------------------------------------------------------------------------------------|-----|----------------------------------------------------------------------------------------|
| 1.1 Related Works                                                                      | 3   | 1.1 Related Works                                                                      |
| Designing Games for Large Populations: Lipschitz Case                                  | 3   | Designing Games for Large Populations: Lipschitz Case                                  |
|                                                                                        | 4   |                                                                                        |
| 2.1 Parameterized Mean Field Game Design 2.2 Approximating the First Order Derivatives | 5   | 2.1 Parameterized Mean Field Game Design 2.2 Approximating the First Order Derivatives |
| Beyond Lipschitz: Mechanism Design for Large-Scale Auctions                            | 6   | Beyond Lipschitz: Mechanism Design for Large-Scale Auctions                            |
| 3.1 Auctions at the Mean-Field Regime                                                  | 7   | 3.1 Auctions at the Mean-Field Regime                                                  |
| 4 Experimental Results                                                                 | 8   | 4 Experimental Results                                                                 |
| 5 Conclusion                                                                           | 10  | 5 Conclusion                                                                           |
| A Frequently-Used Notation                                                             | 25  | A Frequently-Used Notation                                                             |
| B Extended Related Works                                                               | 26  | B Extended Related Works                                                               |
| C Automated Mechanism Design as ID                                                     | 26  | C Automated Mechanism Design as ID                                                     |
| D Mean-Field Mechanism Design                                                          | 27  | D Mean-Field Mechanism Design                                                          |
| D.1 Preliminary Lemmas                                                                 | 27  | D.1 Preliminary Lemmas                                                                 |
|                                                                                        | 28  |                                                                                        |
| D.2 Proof of Theorem 1 . . . D.3 Proof of Lemma 1 . . . .                              | 28  | D.2 Proof of Theorem 1 . . . D.3 Proof of Lemma 1 . . . .                              |
| D.4 Discussion of Assumptions of Lemma                                                 | 29  | D.4 Discussion of Assumptions of Lemma                                                 |
| D.6 Extension of Lemma 2 to General Mirror                                             | 31  | D.6 Extension of Lemma 2 to General Mirror                                             |
| Descent . . . Results on Batched Auctions                                              | 31  | Descent . . . Results on Batched Auctions                                              |
| E.1 Extended Definitions                                                               | 34  | E.1 Extended Definitions                                                               |
| E.3 Proof of Theorem 2, part 1 (Approximation in Exploitability)                       | 37  | E.3 Proof of Theorem 2, part 1 (Approximation in Exploitability)                       |
| E.4 Proof of Theorem E.5 Proof of Lemma Experiment Details                             | 50  | E.4 Proof of Theorem E.5 Proof of Lemma Experiment Details                             |
|                                                                                        | 54  |                                                                                        |
| F                                                                                      | F   | F                                                                                      |
| Bar Process                                                                            | 55  | Bar Process                                                                            |
| F.1 Additional Results on the Beach F.2 Additional Results on Auctions .               | 57  | F.1 Additional Results on the Beach F.2 Additional Results on Auctions .               |

## A Frequently-Used Notation

## General Notation

H ( u )

$$:= - ∑ x u ( x ) log u ( x ) , entropy$$

sigmoid( x )

:= 1 1+ e - x , sigmoid function

∇ f

∈

R

d

1

×

d

2

, Jacobian of function

f

:

R

d

2

→

R

d

1

S D - 1

:= { x ∈ R D : ∥ x ∥ = 1 } , ( D - 1) -dimensional unit sphere in R D

B D

:=

{

x

∈

R

D

:

∥

x

∥ ≤

1

}

,

D

-dimensional unit closed ball in

R

D

V ar

variance of random variable

e i

standard unit vector with i -th entry 1.

∆ X

:=

{

u

∈

R

X

:

∑

x

u

x

= 1

, u

x

≥

0

}

, probability simplex on

X

.

D kl ( u | v ) ⟨· , ·⟩

:=

∑

x

u

(

x

) log

u

(

x

)

/

v

(

x

)

, Kullback-Leibler divergence

dot product.

✶ ∥ · ∥ 1 ∥ · ∥

indicator function.

ℓ 1 norm (on Euclidean space R D )

2

ℓ 2 norm (on Euclidean space R D )

⊗

i

∈

[

N

]

m

i

product measure

σ ( x )

empirical counts of entries of some x ∈ X N , σ ( x ) := 1 / N ∑ N i =1 e x i spectral radius of matrix A ∈ R D,D

ρ ( A )

Marg Y ( d )

$$:= ∑ x ∈X d ( x, · ) ∈ ∆ Y , for d ∈ ∆ X×Y .$$

∥ A ∥ p → q

:= sup

∥

x

∥

p

≤

1

∥

Ax

∥

q

, operator norm of

A

with norms

∥ · ∥

p

,

∥ · ∥

q

## Generic PMFGs

G

## parameterized MFG

H

horizon (number of rounds) of auction

S

(finite) state space

A

(finite) action space

N

number of players/agents

g

ID objective

G

:= E [ g ( θ, ̂ L )] , true ID objective in N -player game

Nash τ ( G )

set of τ -regularized Nash equilibria

Θ

parameter space

θ

∈ Θ , ID design parameter

Π H

{

π : [ H ] ×S → ∆ A } , set of finite-horizon Markovian policies.

µ 0

initial state distribution

P h,θ

parameterized state transition dynamics in N -player DG

R h,θ

parameterized reward functions in N -player DG

τ

∈ R entropy regularization magnitude

π

$$∈ Π N H , N -tuples of policies$$

J

τ,i

G

expected reward of player i in dynamic game G (see Definition 1)

E

τ,i

G

exploitability of player i in dynamic game G (see Definition 1)

E

τ

G M

maximum exploitability in dynamic game G (see Definition 1)

parameterized mean-field game (PMFG)

P h,θ

parameterized state transition dynamics in PMFG

R h,θ

parameterized reward functions in PMFG

Γ h

one-step MFG forward flow (see Definition 2)

Λ

maps policies in Π H to H -step mean-field population flow in ∆ H S×A (see Definition 2)

## Adjoint method:

V

τ

h

state value function in the PMFG

q τ h

q-value value function (on state-action pairs) in the PMFG

F

: R [ H ] ×S×A × Θ → R [ H ] ×S×A generic policy update operator for computing

NE, defined in log policy space

F omd

mirror descent update, in policy space

F omd

mirror descent update, in log-policy space

## For auctions:

| G auc ⊥ V α α θ h p θ h u h w h P R J auc E τ,i M P h,θ   | batched Auction inactive state for agent not participating in the current value space parameter for maximal amount of goods   |
|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
|                                                           | round                                                                                                                         |
| max                                                       |                                                                                                                               |
|                                                           | parametrized item allocation function                                                                                         |
|                                                           | parametrized payment function                                                                                                 |
|                                                           | utility function of bidders                                                                                                   |
|                                                           | valuation dynamics function                                                                                                   |
| auc h,θ                                                   | transition dynamics in batched auction                                                                                        |
| auc h,θ                                                   | reward functions                                                                                                              |
| τ,i                                                       | entropy regularized ( τ ) sum of rewards for agent i ∈ [ N ]                                                                  |
| auc                                                       | entropy regularized ( τ ) exploitability for agent i ∈ [ N ]                                                                  |
| mfa                                                       | batched auction MFG                                                                                                           |
| mfa                                                       | BA-MFG transition dynamics                                                                                                    |
| R mfa h,θ                                                 | BA-MFG reward functions                                                                                                       |
| ν -⊥                                                      | action marginal of active (non- ⊥ ) states; maps state-action distributions to sub-probability distributions over actions     |
| p win                                                     | winning probability function, given a bid a , sold goods α and population bid distribution ν -⊥                               |
| ξ L,θ                                                     | post allocation state distribution                                                                                            |
| V τ mfa                                                   | entropy regularized sum of rewards for BA-MFG                                                                                 |
| E τ mfa                                                   | entropy regularized exploitability of rewards for BA-MFG                                                                      |
| Λ mfa                                                     | population operator for BA-MFG                                                                                                |
| g rev                                                     | revenue objective                                                                                                             |

## B Extended Related Works

Related works on Equilibrium Computation in Auctions. As outlined in Appendix C, many realworld auctions are not strategyproof. It is thus important to evaluate them at equilibrium-both from a predictive (how bidders will likely behave), as well as from a normative (bidding recommendations) standpoint. While some simple formats have been solved analytically [43], existing hardness results for computing exact equilibria in auctions [10] motivate approximate computational approaches. In single-round auctions, a strand of work has used iterated best-response computations to calculate equilibria [61, 68, 60, 8, 9]. Other approaches rely on gradient descent [7, 42] or deep learning [6, 53]. For multi-round auctions, [65] compute ε -perfect Bayesian equilibria, using best response dynamics. Others have used (deep) RL to find approximate Nash equilibria [30, 59, 24, 66]. In our work, by using mean-field approximations, we circumvent the curse of dimensionality inherent to these multi-agent RL approaches. Using mean-field approaches to solve auctions has been explored previously to some extent. [2, 32] study specific repeated ad auctions with budget constraints, and [38] studies dynamic auctions, where bidders iteratively learn about their own type.

Other MFG works. Stackelberg equilibria for MFGs have also been studied in the particular case of linear-quadratic models [54, 4]. Another relevant model in this setting is mean-field incentive design with major and minor players [64], where designing incentives for a leader is studied for the purpose of influencing a population. In continuous time, Stackelberg MFGs have been studied in applications such as regulating carbon markets [13], epidemics [1], and advertising markets [12].

## C Automated Mechanism Design as ID

We note that in contrast to our approach many works on automated mechanism design focus on designing strategyproof auctions, i.e. auctions in which bidders bid truthful in equilibrium, relying on the so-called revelation principle [20, 18, 49]. The revelation principle states that any nonstrategyproof equilibrium of an auction can be implemented as an outcome equivalent strategyproof equilibrium of an adapted auction [43]. Restricting to strategyproof mechanisms bypasses the need

to differentiate through an equilibrium. Instead of a problem like (ID) where the outer objective depends on an inner equilibrium solution, the inner solution is already known-bid truthfully-and instead the problem becomes one of constrained optimization problem, where the so-called incentive compatibility (IC) constraints ensure that bidding truthfully is in fact an equilibrium [18].

While restricting to strategyproof mechanisms foregoes the need to differentiate through an equilibrium, many real-world auctions are not strategyproof. In fact, in 2019 Google for example deliberately changed towards non-strategyproof first price auctions for selling ads, citing the increased transparency of simple, non-strategyproof format for the bidders [25]. In such cases without IC constraints, the question is how bidders will respond in equilibrium and in turn designing revenue-optimal auctions becomes an instance of (ID), which we tackle in Sections 3 and 4.

## D Mean-Field Mechanism Design

## D.1 Preliminary Lemmas

Theorem 3 (Rademacher) Let U ⊂ R m be an open set and f : U → R n be a Lipschitz continuous map. Then, f is almost everywhere differentiable on U , that is, the points on which f is not differentiable on U for a set of measure 0.

In some cases, an explicit differentiability assumption might be useful for PMFG dynamics, which we state below.

Assumption 2 (Differentiability) For all s, s ′ ∈ S , a ∈ A , the functions P θ ( s ′ | s, a, L ) , R θ ( s, a, L ) are differentiable on θ, L . Furthermore, g ( θ, L ) is differentiable on θ, L with bounded derivatives.

In particular, the following simple result is useful for the derivation of AMID.

Lemma 4 (Differentiability of operators) The maps Γ , Λ , q τ h , V τ h , F omd as well as the map θ, π → g ( θ, Λ( π )) are almost everywhere differentiable under Assumption 1 and differentiable everywhere under Assumption 2.

Proof: This result is a straightforward result of the definitions of the mentioned operators, in particular, when Assumption 2 is taken, the mentioned operators are also differentiable as they are the compositions of differentiable functions. In the case where only Assumption 1 holds, the above-mentioned functions are also Lipschitz on every bounded domain, which implies by Theorem 3 that they are almost everywhere differentiable. □

Finally, we state the following standard lemma from past work on the approximation of MFG dynamics by finite player games.

Lemma 5 Assume that the conditions of Theorem 1 hold for the PMFG M and DG G , let π, π ∈ Π H be two arbitrary policies, and let θ ∈ Θ be fixed. Let L π = { L π h } h = Λ( π | θ ) be the population flow induced by π on the PMFG with fixed parameter θ . Take the trajectories s i h , a i h , ̂ L h induced by the DG with parameter θ and policy profile ( π, π, . . . , π ) ∈ Π N H . Then,

<!-- formula-not-decoded -->

where ∆ h := sup s ∥ π h ( ·| s ) -π h ( ·| s ) ∥ 1 , and L pop ,µ is a uniform bound on the Lipschitz moduli of Γ h in L . Furthermore, denoting the random variables s h , a h as the distributions of state-action pairs in the PMFG dynamics with population flow L π and policy π ,

<!-- formula-not-decoded -->

where K L is the Lipschitz modulus of transition dynamics P θ,h in L .

Proof: The proof is a straightforward extension of the approximation results due to [75], to the case where transition dynamics and rewards also depend on the mean-field flow over actions. See in particular Theorem 3.2 in [75]. □

## D.2 Proof of Theorem 1

First, we present the bound on exploitability. As in the setting of Lemma 5, let L π = { L π h } h = Λ( π | θ ) be the population flow induced by π on the PMFG with fixed parameter θ . Take the trajectories s i h , a i h , ̂ L h induced by the DG with parameter θ and policy profile π := ( π, π, . . . , π ) ∈ Π N H . Similarly, denote the random variables s h , a h as the distributions of state-action pairs in the PMFG dynamics with population flow L π and policy π .

<!-- formula-not-decoded -->

Since | R h,θ ( L π h , s, a | θ ) + τ H ( π h ( s )) | ≤ 1+ τ log |A| , and R h,θ is Lipschitz in L (say with modulus K ), it holds that

<!-- formula-not-decoded -->

by an application of Lemma 5. The corresponding bound on E τ G ( π ∗ | θ ) is obtained by maximizing π as the best response to the population strategy profile in the DG, and setting π to be an MFG-NE π ∗ for the parameter θ .

We also prove a similar bound for the objective value. Assume now that all N players play policy π ∗ in the DG. Then, since g is Lipschitz continuous,

<!-- formula-not-decoded -->

where we denote the Lipschitz modulus of g with L h as K h (with respect to norm ∥ · ∥ 1 ). An application of the technical lemma Lemma 5 yields then the O ( 1 / √ N ) upper bound.

## D.3 Proof of Lemma 1

We first prove that if lim T →∞ F ( T ) omd ( θ ′ , ζ ) exists and F ∞ omd ( θ ′ , ζ ) = ζ ∗ , the softmax( ζ ∗ ) ∈ Nash τ M ( θ ′ ) . Since F omd is a continuous function, it must hold that

<!-- formula-not-decoded -->

therefore F omd ( θ ′ , ζ ∗ ) = ζ ∗ . Denoting π ∗ h ( ·| s ) := softmax( ζ ∗ ( h, s, · )) , we then have the relations

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, for any h , it holds that ζ ∗ ( h, s, a ) = τ -1 q τ h ( s, a | Λ( π ∗ ) , π ∗ , θ ) . We show that π ∗ is then the best response to Λ( π ∗ ) by backward induction. Denote for convenience L ∗ := { L ∗ h } h = Λ( π ∗ ) .

At time H -1 , ζ ∗ ( H -1 , s, a ) = τ -1 R h,θ ( s, a, L ∗ H -1 ) and π ∗ H -1 ( a | s ) = exp { τ -1 R h,θ ( s,a,L ∗ H -1 ) } ∑ a ′ exp { τ -1 R h,θ ( s,a ′ ,L ∗ H -1 ) } , therefore, by first order optimality conditions, π ∗ H -1 ( ·| s ) maximizes (uniquely) the strongly concave function u → ⟨ u, R h,θ ( s, a, L ∗ H -1 ) ⟩ + τ H ( u ) on the

simplex ∆ A . Hence, at every state s at time H -1 , the policy π ∗ H -1 ( ·| s ) is optimal. Assume now π ∗ h ′ ( ·| s ) is optimal for all h ′ &gt; h for some h ∈ { 0 , . . . , H -2 } . Then, by the inductive assumption, q τ h ′ ( s, a | Λ( π ∗ ) , π ∗ , θ ) is the optimal regularized q function for all h ′ ≥ h . Since ζ ∗ ( h, s, a ) = τ -1 q τ h ( s, a | Λ( π ∗ ) , π ∗ , θ ) , once again by first order optimality conditions, π ∗ h is also the optimal policy at time h .

We move on to the convergence of derivatives. Firstly, since q τ h is given to be C 1 in a neighborhood of ( θ, ζ ∗ ) and ρ ( ∂ ζ q τ · ( · , ·| Λ(softmax( ζ ∗ )) , softmax( ζ ∗ ) , θ )) &lt; τ , there exists an open neighborhood U of ( θ, ζ ∗ ) where ρ ( ∂ ζ q τ · ( · , ·| Λ(softmax( ζ ′ )) , softmax( ζ ′ ) , θ ′ )) &lt; τ -δ 1 for all ( θ ′ , ζ ′ ) ∈ U for some δ 1 &gt; 0 . Then, since

<!-- formula-not-decoded -->

on U it also holds that ρ ( ∂ ζ F omd ( θ, ζ )) &lt; 1 -δ 2 for some δ 2 &gt; 0 . On U , the map F ∞ omd is implicitly defined by F omd ( θ, F ∞ omd ( θ, ζ )) = F ∞ omd ( θ, ζ ) , therefore by the implicit function theorem F ∞ omd is differentiable in θ and

<!-- formula-not-decoded -->

Fixing some direction v ∈ Θ , define inductively the iterates

<!-- formula-not-decoded -->

Note that by the chain rule, x t +1 = ∇ u ( F ( T ) omd ( θ, ζ ′ )) v , therefore, if we show that lim t →∞ x t = ∇ θ ( F ( ∞ ) omd ( θ, ζ ′ )) v for any choice of v , we are done. Firstly, by the assumptions of the lemma, for sufficiently large T 0 , it holds that ζ t ∈ U for all t &gt; T 0 , and ζ ∗ = lim t →∞ ζ t . Defining x ∗ := ( I -∂ ζ F omd ( θ, ζ ∗ )) -1 ∂ θ F omd ( θ, ζ t ) v , which satisfies x ∗ := ∂ ζ F omd ( θ, ζ ∗ ) x ∗ + ∂ θ F omd ( θ, ζ t ) v . Therefore,

<!-- formula-not-decoded -->

which proves that x t → x ∗ as both ∂ θ F omd , ∂ ζ are continuous in ζ for ζ ∈ U and ζ t → ζ ∗ . By the implicit function theorem, x ∗ is the gradient of F ( ∞ ) omd , concluding the proof.

## D.4 Discussion of Assumptions of Lemma 1

We briefly discuss the ramifications of Lemma 1. Firstly, the result is useful only assuming that the OMDiterates converge to an NE of the MFG. NE computation in MFGs is a well-studied research topic on its own, and several positive results are known for various classes of MFGs. We take this for granted in this lemma, as NE computation is not the main goal.

Next, we note that the lemma suggests that the derivative of ( T -approx.) is a valid first-order oracle provided that the Jacobian of F omd has bounded spectral radius around the NE induced by a θ . Importantly, for any PMFG, T -step objective approximates the fixed-point gradient provided that τ is sufficiently large. For more structured settings, the result can be strengthened to permit τ = 0 , for instance, the derivations in [48] readily extend to the case where H = 1 , |S| = 1 , and the reward function R is monotone in population distribution. We leave as future work to generalize this for monotone PMFGs with dynamics (i.e., for PMFGs with H &gt; 1 ).

## D.5 Proof of Lemma 2

Let Θ = R D 1 , Z = R D 2 , and assume the functions F : Θ × Z → Z and g : Θ × Z → R are differentiable. We define ℓ : Θ → R

<!-- formula-not-decoded -->

for some ζ 0 constant. Clearly ℓ ( θ ) is differentiable; our goal is to efficiently compute ∇ θ ℓ ( θ ) . Define for any sequence of (row) vectors { a t } T t =0 for a t ∈ Θ the function

<!-- formula-not-decoded -->

Since for any { a t } T t =0 it holds that L ( θ, { a t } T t =0 ) = ℓ ( θ ) , we have the identities

<!-- formula-not-decoded -->

Therefore, for any sequence of functions a t ( θ ) that depend on θ , we have

<!-- formula-not-decoded -->

We will therefore compute ∂ θ L ( θ, { a t } T t =0 ) with a suitable choice of a t . By simple derivation:

<!-- formula-not-decoded -->

Since ∇ θ ζ 0 = 0 , as ζ 0 is constant, rearranging the terms yields:

<!-- formula-not-decoded -->

Therefore, we pick we obtain the equality

<!-- formula-not-decoded -->

Most importantly, Equation (2) permits the computation of ∇ θ ℓ with T with only caching the variables ζ t . Namely, the forward system Equation (1) can be used to iteratively compute ζ T +1 , and the backward system only requires evaluating Jacobians at the current step, meaning other than ζ t , the memory requirements are kept constant.

In case memory is a bottleneck, ζ t can be cached during the forward step every β √ T steps for some constant β , meaning only O ( √ T ) memory is needed. In this case, during the backward step, ζ t will need to be recomputed every O ( √ T ) steps for O ( √ T ) OMDiterations, maintaining the O ( T ) time complexity. β will introduce a tradeoff between time and space complexity.

<!-- formula-not-decoded -->

## D.6 Extension of Lemma 2 to General Mirror Descent

Lemma 2 analyzes the adjoint method under the specific entropy regularization scheme. Let h : R D 1 → R be a strongly convex distance generating function, and ∇ h : R D 1 → R D 1 the corresponding mirror map. Then, the generalized mirror descent update rule can be written as (in policy space Π H ⊂ R [ H ] ×S×A ):

<!-- formula-not-decoded -->

Defining the iterates ζ t := ∇ h ( π t +1 ) , we obtain the similar update rule

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which reduces to the case analyzed in Appendix D.5, by the definitions

<!-- formula-not-decoded -->

## E Results on Batched Auctions

This section presents the formal analysis of parameterized batched auctions introduced in the main text. We also provide rigorous statements and complete the proofs left out in the main text.

Appendix E.1 provides formal definitions of the settings omitted in the main text and useful auxiliary constructions to assist the proofs. Appendix E.2 presents a collection of auxiliary lemmas that support the subsequent analysis. In Appendix E.3, we prove the first part of Theorem 2, establishing an upper bound on the exploitability of mean field policies under the no zero-dominance condition. The second part of the theorem, which addresses convergence of the mechanism-level objective, is proved in Appendix E.4. Finally, Appendix E.5 provides the proof of Lemma 3, which establishes the Lipschitz continuity of the (entropy regularized) q values under full-support policies.

## E.1 Extended Definitions

Additional useful notation. To streamline the proofs, we also introduce some useful notation. For any arbitrary finite set X and a scalar β ∈ R ≥ 0 define the sets:

<!-- formula-not-decoded -->

For x ∈ R X , where X has a total order, denote the cumulative mass function S x ( d ) := ∑ x ′ ≥ x d ( x ′ ) . For some d ∈ ∆ X×Y , define the marginal distribution

<!-- formula-not-decoded -->

For α ∈ [0 , 1] , define the threshold bid operator Th α : ∆ ≥ 0 A →A as

<!-- formula-not-decoded -->

where a 0 is the smallest element of A . For d ∈ ∆ ≥ α V×A we also define Th α ( d ) := Th α (Marg S ( d )) .

We define the operator Ξ α : ∆ ≥ α V×A → R V×A , which maps a matrix state-action distribution L to a matrix L ′ , where each entry L ′ ( s, a ) represents the expected probability mass of agents in state s who chose action a and did not win an item this round, given that α goods are allocated. Formally:

<!-- formula-not-decoded -->

Ξ α is well-defined as ∑ s ′ ∈V d ( s ′ , Th α ( d )) &gt; 0 whenever α &gt; 0 by definition.

Using the operator Ξ α , we define the post allocation operator Υ α : ∆ ≥ α V×A → ∆ V , which maps a sub-probability distribution d over active state-action pairs to the post allocation state distribution. That is,

<!-- formula-not-decoded -->

The operator Γ α , which governs the transition of the population state distribution after item allocation, can be expressed in terms of Υ α by explicitly accounting for the inactive state ⊥ . Specifically, let d ∈ ∆ S×A be a state-action distribution and assume α ( ν -⊥ ( d )) ≤ ∥ ν -⊥ ( d ) ∥ 1 . Then Γ α : ∆ S×A → ∆ S is defined as

<!-- formula-not-decoded -->

where d -⊥ denotes the restriction of d to active states V .

The BA-MFG is a parametrized PMFG, where both the payment function and the allocation threshold function are parameterized as α θ h and p θ h . Throughout this section, we assume these parameters are fixed and omit them, writing without loss of generality α h , p h instead.

Definition 5 (Batched Auction MFG (BA-MFG)) A Batched-Auction MFG (BA-MFG) is a MFG defined by the tuple M mfa = ( S , A , H, µ 0 , { P mfa h } H -1 h =0 , { R mfa h } H -1 h =0 ) of discrete state space S = V ∪{⊥} , discrete action space A , horizon H ∈ N &gt; 0 , initial distribution µ 0 ∈ S , transition dynamics P mfa h : S × A × ∆ S×A → ∆ S and reward functions R mfa h : S × A × ∆ S×A → [0 , 1] . The transition dynamics P mfa and the reward functions R mfa depend from the allocation functions { α h } H -1 h =0 , the dynamic functions { w h } H -1 h =0 , the payment functions { p h } H -1 h =0 and the utility functions { u h } H -1 h =0 . Define the 'winning probability' as

̸

For the allocation α and valuation transition w , define the operators Γ α : ∆ S×A → ∆ S , Γ w : ∆ S × ∆ S as

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Define also Γ h , Λ mfa as 7

<!-- formula-not-decoded -->

The transition probability and rewards can be written as:

<!-- formula-not-decoded -->

For π ∈ Π H , τ ≥ 0 and L = { L h } H -1 h =0 , the total expected (entropy regularized) reward is

<!-- formula-not-decoded -->

7 Note that this way of writing Γ h , Λ mfa is consistent with the MFG definition.

For a policy π ∈ Π H we denote with L π = Λ mfa ( π ) the induced state-action distribution, with µ π the induced state distribution and with ξ π the induced hidden state distribution, with ξ π h = Γ α h ( L π h ) . Additionally, we define with P α the transition probabilities associated with the item allocation dynamics

<!-- formula-not-decoded -->

Then, the total expected (entropy regularized) reward can also be expressed as

<!-- formula-not-decoded -->

Definition 6 ( N -player Batched Auction ( N -BA)) An N - player batched auction ( N -BA) is a dynamic game G auc = ( N, S , A , H, µ 0 , { P auc h } H -1 h =0 , { R auc h } H -1 h =0 ) of discrete state space S = V ∪ {⊥} , discrete action space A , horizon H ∈ N &gt; 0 , starting distribution µ 0 ∈ S , transition dynamics P auc h : S N × A N → ∆ N S and rewards functions R auc h : S N × A N → [0 , 1] N . The transition dynamics P auc h and the reward functions R auc h depend from the allocation functions { α h } H -1 h =0 , the dynamic functions { w h } H -1 h =0 , the payment functions { p h } H -1 h =0 and the utility functions { u h } H -1 h =0 . Let s = ( s 1 , . . . , s N ) ∈ S N denote the joint state of all agents, and a = ( a 1 , . . . , a N ) ∈ A N the joint action profile. Assume items are allocated to the top bidders according to the submitted actions, with uniform random tie-breaking at the allocation threshold. This induces a joint allocation probability kernel P N,α : S N ×A N → ∆ S N formally defined as

<!-- formula-not-decoded -->

̸

̸

where ν = ∑ j e a j ✶ s j = ⊥ , a ∗ = max { a ∈ A : ∑ j ∈ [ N ] ✶ a j ≥ a,s j = ⊥ ≥ ⌊ α ( ν ) N ⌋} , T = { j ∈ [ N ] : a j = a ∗ , s j = ⊥} , W = { j ∈ [ N ] : a j &gt; a ∗ , s j = ⊥} and L = { j ∈ [ N ] : a j &lt; a ∗ , s j = ⊥} . The marginal winning probability and the rewards, similar as its corresponding MFG can be expressed as

̸

̸

̸

<!-- formula-not-decoded -->

We define with R mfa N,h the N-player discretization of R mfa h :

<!-- formula-not-decoded -->

Note that if L ( s, a ) = 0 , the reward R mfa N,h ( s, a, L ) is not defined. Additionally, observe that

<!-- formula-not-decoded -->

For a strategy profile π ∈ Π N H , τ ≥ 0 the (entropy regularized) sum of rewards of player i ∈ [ N ] is defined as

<!-- formula-not-decoded -->

exploitability E τ,i auc as E τ,i auc ( π ) := max π ′ ∈ Π H J τ,i auc ( π ′ , π -i ) -J τ,i auc ( π ) .

## E.2 Preliminary Lemmas

We present several important lemmas that will be used later to prove the main convergence theorem.

Lemma 6 (Sensitivity of Ξ α to the Population) Let α ∈ [0 , 1] and let d, d ′ ∈ ∆ ≥ α V×A . Then Ξ α is Lipschitz-continuous with respect to the ℓ 1 -norm, with constant 1:

<!-- formula-not-decoded -->

Proof: It is straightforward to verify that Ξ α is continuous. We verify that it is also Lipschitz continuous with modulus 1. Assume α &gt; 0 , as otherwise Ξ α ( d ) is the identity map and the claim is trivial. Denote the bids by A := { 1 , . . . , A } , and let S a ( d ) := ∑ s ′ ∑ a ′ ≥ a d ( s ′ , a ′ ) with S A +1 ( d ) := 0 , and r a ( d ) := ∑ s ′ ∈V d ( s ′ , a ) . For ¯ a ∈ { 1 , . . . , A +1 } , define the regions

<!-- formula-not-decoded -->

On the region R ¯ a , the map Ξ α ( d ) is differentiable, in fact, for d ∈ R ¯ a , it holds that ¯ a = Th α ( d ) and

<!-- formula-not-decoded -->

We calculate the Jacobian of Ξ α and upper bound its operator norm ∥∇ Ξ α ∥ 1 → 1 given by the max column sum ∥∇ Ξ α ∥ 1 → 1 = max s,a ∑ s ′ ,a ′ | ( ∇ Ξ α ) s ′ a ′ ,sa | . We upper bound the column sums corresponding to s, a .

̸

Case 1. If a &lt; ¯ a , then ∂ Ξ α ( d )( s ′ ,a ′ ) ∂d sa = 0 for any ( s ′ , a ′ ) = ( s, a ) and ∂ Ξ α ( d )( s,a ) ∂d sa = 1 , therefore the column sum is

<!-- formula-not-decoded -->

̸

Case 2. If a = ¯ a , then ∂ Ξ α ( d )( s ′ ,a ′ ) ∂d sa = 0 if a ′ = ¯ a , therefore

̸

<!-- formula-not-decoded -->

̸

since all terms in the absolute values are nonnegative if ¯ a = Th α ( d ) .

Case 3. If a &gt; ¯ a , then only the rows corresponding to the active action ¯ a has nonzero gradient, and

<!-- formula-not-decoded -->

To conclude, it holds that on any arbitrary region R ¯ a ,

<!-- formula-not-decoded -->

Therefore, ∥∇ Ξ α ∥ 1 → 1 ≤ 1 on all regions R ¯ a , and Ξ α is non-expansive in the ℓ 1 norm. □

Lemma 7 (Sensitivity of Ξ α to the Allocation Parameter) Let d ∈ ∆ ≤ 1 V×A arbitrary, let α 1 , α 2 ≤ ∥ d ∥ 1 arbitrary. Then

<!-- formula-not-decoded -->

Proof: As in Lemma 6, let A := { 1 , . . . , A } , and let S a ( d ) := ∑ s ′ ∑ a ′ ≥ a d ( s ′ , a ′ ) with S A +1 ( d ) := 0 , and r a ( d ) := ∑ s ′ ∈V d ( s ′ , a ) .

For ¯ a ∈ { 1 , . . . , A +1 } such that r ¯ a ( d ) &gt; 0 , define the partition of the interval [0 , ∥ d ∥ 1 ] into the intervals

<!-- formula-not-decoded -->

On the interval R ¯ a , the map Ξ α ( d ) is differentiable in α , and it holds that ¯ a = Th α ( d ) and

<!-- formula-not-decoded -->

Then, again, we upper bound the operator norm of the Jacobian (in this case, gradient):

<!-- formula-not-decoded -->

So ∥∇ α Ξ α ( d ) ∥ 1 → 1 ≤ 1 , implying the claim of the lemma.

□

For completeness, we state the simple corollaries of the above two sensitivity analyses in the following.

Corollary 1 (Lipschitz Continuity of Ξ ) Let d, d ′ ∈ ∆ ≤ 1 V×A , α ∈ [0 , ∥ d ∥ 1 ) , and α ′ ∈ [0 , ∥ d ′ ∥ 1 ) arbitrary, then

<!-- formula-not-decoded -->

Proof: Let d, d ′ ∈ ∆ ≤ 1 V×A , α ∈ [0 , ∥ d ∥ 1 ) , and α ′ ∈ [0 , ∥ d ′ ∥ 1 ) arbitrary. Without loss of generality assume ∥ d ∥ 1 ≥ ∥ d ′ ∥ 1 , then by applying triangular inequality we have

<!-- formula-not-decoded -->

Corollary 2 (Lipschitz Continuity of Υ α ) Let d, d ′ ∈ ∆ ≤ 1 V×A , α ∈ [0 , ∥ d ∥ 1 ) , and α ′ ∈ [0 , ∥ d ′ ∥ 1 ) arbitrary, then Υ α is non-expansive in the ℓ 1 norm, that is

<!-- formula-not-decoded -->

Proof: Let d, d ′ ∈ ∆ ≤ 1 , α ∈ [0 , ∥ d ∥ 1 ) , and α ′ ∈ [0 , ∥ d ′ ∥ 1 ) arbitrary, then

<!-- formula-not-decoded -->

The upper bound follows by the result of Corollary 1.

□

Corollary 3 (Lipschitz Continuity of Γ α ) Let d, d ′ ∈ ∆ S×A , and suppose the threshold function α : ∆ ≤ 1 A → [0 , 1] is Lipschitz continuous with constant K α , and satisfies the feasibility condition α ( ν ) ≤ ∥ ν ∥ 1 for all ν ∈ ∆ ≤ 1 A . Then,

<!-- formula-not-decoded -->

Proof: By Corollary 2 and the Lipschitz continuity of α , it follows that:

<!-- formula-not-decoded -->

In the next sequence of results, we deal with the stability of winning probabilities given by the function p win. In general, p win is easily seen to have discontinuous jumps, however, a local stability result can be shown if the NZD condition holds.

Lemma 8 (Stability of Winning Probabilities) Let A = { a 1 , a 2 , . . . , a K } be a finite set of actions with total order a 1 &lt; a 2 &lt; · · · &lt; a K , and let ν ∈ ∆ ≤ 1 A . For any α ∈ [0 , 1] , define the winning probability for action a ∈ A as

<!-- formula-not-decoded -->

Assume that α, ν satisfy the no zero-dominance property (i.e., for any a ∈ A , ν ( a ) = 0 ⇒ ∑ a ′ &gt;a ν ( a ′ ) &lt; α ( ν ) ) and ∥ ν ∥ 1 &gt; 0 . Then for all ν ′ ∈ ∆ ≤ 1 A , α ′ ∈ [0 , 1] , a ∈ A , p win satisfies

<!-- formula-not-decoded -->

where:

<!-- formula-not-decoded -->

̸

and where a ∗ := Th α ( ν ) is the threshold action, a -is the action just below the threshold ( a -:= a k -1 if a ∗ = a k for some k &gt; 1 and a -= ⊥ if a ∗ = a 1 ), and ∆ ν,α := min a ∈A ∣ ∣ ∣ α -∑ a ′ ⪰ a ν ( a ′ ) ∣ ∣ ∣ .

̸

̸

Proof: Importantly, C ν,α is finite in each case if ν, L satisfies the NZD condition. The proof, while notionally dense, works on a simple idea: in general, p win incorporates discontinuities where the winning probability of an action below the threshold might jump from 0 to 1 . However, this does not happen locally when there is some probability mass on a -just below the threshold action. Note that when ∥ ν ∥ 1 = 0 and NZD holds, it holds that ν ( a ∗ ) &gt; 0 , and ν ( a -) &gt; 0 if a ∗ = a 1 and α &gt; 0 . Furthermore, by NZD, if α = 0 , it must hold that ν ( a K ) &gt; 0 and a ∗ = a K by definition. Define the useful constant

̸

<!-- formula-not-decoded -->

which will be the radius of the open set around which there are no discontinuities of p win.

First, we show that | p win ( a, ν, α ) -p win ( a, ν, α ′ ) | ≤ C ν,α | α -α ′ | for any α ′ . Without loss of generality, we can assume that α ≤ ∥ ν ∥ 1 , α ′ ≤ ∥ ν ∥ 1 , as p win ( a, ν, α ′ ) = p win ( a, ν, min { α ′ , ∥ ν ∥ 1 } ) for any α ′ by definition, and | min { α, ∥ ν ∥ 1 } -min { α ′ , ∥ ν ∥ 1 }| ≤ | α -α ′ | . If | α -α ′ | &lt; δ , then α ′ ∈ ( α -δ, 0 , α + δ ) ∩ R ≥ 0 . On the interval ( α -δ, α + δ ) ∩ R ≥ 0 , p win is continuous for any a since

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, from above, p win is almost everywhere differentiable in α ′ in this interval with

̸

<!-- formula-not-decoded -->

as in this interval the 'active' threshold action will always be either a ∗ of a -. Therefore,

<!-- formula-not-decoded -->

̸

̸

If on the other hand | α -α ′ | ≥ δ , then

<!-- formula-not-decoded -->

So in both cases it holds that | p win ( a, ν, α ) -p win ( a, ν, α ′ ) | ≤ C ν,α | α -α ′ | .

Next, we show the stability in ν . Once again, assume that ∥ ν ′ -ν ∥ 1 &lt; δ . On the open set { ν ′ : ∥ ν ′ -ν ∥ 1 &lt; δ } the function p win ( a, · , α ) is once again continuous in ν ′ for any a , and almost everywhere differentiable with

̸

<!-- formula-not-decoded -->

for almost every ν ′ . Therefore, it holds that

<!-- formula-not-decoded -->

On the other hand, if ∥ ν ′ -ν ∥ 1 ≥ δ , then

<!-- formula-not-decoded -->

To complete the proof, we use the triangle inequality:

<!-- formula-not-decoded -->

Remark 3 The proof of Lemma 8 can be adapted to handle action distributions with full support. In this case, a refined version of the first part of the argument shows that the winning probability is Lipschitz-continuous with a constant bounded by 1 min { ν ( a -) ,ν ( a ∗ ) ,ν ( a + ) } , where a ∗ , a -, a + are actions around the threshold. Consequently, for policies with full support, i.e., π h ( a | s ) ≥ ϵ for all s ∈ V , a ∈ A , and some ϵ &gt; 0 , the deviation in winning probability is bounded by 1 (1 -α max ) ϵ ∥ ν -ν ′ ∥ 1 .

Lemma 9 Let µ, µ ′ ∈ ∆ S be two arbitrary state distributions, and let π, π ′ ∈ Π be two arbitrary policies. Define the corresponding state-action distributions L, L ′ ∈ ∆ S×A as

<!-- formula-not-decoded -->

Then, it holds that

<!-- formula-not-decoded -->

Proof: We compute the total variation distance:

<!-- formula-not-decoded -->

where the last line uses that ∑ a π ′ ( a | s ) = 1 for all s .

## E.3 Proof of Theorem 2, part 1 (Approximation in Exploitability)

The theorem considers BA-MFGs with Lipschitz-continuous reward, dynamics, and utility functions. Let K w ∈ [0 , 2] denote the Lipschitz modulus of the state dynamics w h , K p the Lipschitz modulus of the payment function p h , K u the Lipschitz modulus of the utility function u h with respect to its second argument, and K α be the Lipschitz modulus of the allocation funciton α h . That is, for any s ∈ V , a ∈ A , ν, ν ′ ∈ ∆ A , p, p ′ ∈ R ≥ 0 ,

<!-- formula-not-decoded -->

□

The composed function u h ( s, p h ( a, ν )) then is also Lipschitz with modulus K u K p as

<!-- formula-not-decoded -->

for all ν, ν ′ ∈ ∆ ≤ 1 A .

Additionally, let B p be an upper bound on the absolute value of payments, i.e., | p h ( a, ν ) | ≤ B p , and let B u be an upper bound on the absolute value of utilities, i.e., | u h ( s, p ) | ≤ B u , for all s, a, ν, h . Such an upper bound always exists if u h , p h are continuous in ν , since ∆ ≤ 1 A is a compact set.

The argument proceeds in three steps:

1. First, we bound the expected deviation between the empirical distributions and their mean field counterparts. That is, we show an upper bound on the deviation E [ ∥ L π h -̂ L h ∥ 1 ] .
2. Second, we show that for policies satisfying the no zero-dominance property, the expected transition probabilities associated with item allocation, under both the mean field and finite-population settings, differ proportionally to the deviation in population distributions.
3. Finally, we bound the exploitability of a single agent when all other agents follow a policy that satisfies the no zero-dominance property with respect to the mechanism.

In our analysis, we also make use of the population distribution ξ h after item allocation at round h . In the mean field setting, this is defined as ξ h := Γ α h ( L h ) , where L h ∈ ∆ S×A is the state-action distribution at round h , and the operator Γ α h captures the expected post-allocation state distribution under the mechanism (e.g., by marking winners as inactive). This quantity serves as the input to the state transition function w h in the MFG dynamics. In the finite-agent setting, we denote the analogous empirical quantity by ̂ ξ h , representing the empirical distribution over states immediately after allocation. We also define the random variables { z i h } as z i h = ⊥ if agent i was not active in round h (i.e. s i h = ⊥ ) or agent i won the the auction in round h , and z i h = s i h otherwise. With this definition,

<!-- formula-not-decoded -->

Finally, we define the constants used in our convergence analysis as follows. Define

<!-- formula-not-decoded -->

## E.3.1 Step 1: Expected Deviation of Empirical Distributions

We derive explicit bounds on the expected deviation between the empirical distributions and their mean field counterparts. In particular, we bound the deviations for the state distribution and the state-action distribution.

Lemma 10 Let M mfa = ( S , A , H, µ 0 , { P mfa h } H -1 h =0 , { R mfa h } H -1 h =0 ) define a BA-MFG. Consider the corresponding finite-agent Batched Auction model with N agents, G auc, which is approximated by M mfa. Let π = { π i h } h =0 ,...,H -1 , i ∈ [ N ] ∈ Π N H denote the joint policy of the population. Denote by ̂ µ h ∈ ∆ S the empirical state distribution and by ̂ L h ∈ ∆ S×A the empirical state-action distribution at round h .

Let π ∈ Π H arbitrary, and define the associated mean field state-action distribution flow L π := Λ mfa ( π ) , with corresponding marginal state distribution µ π h := ∑ a L π h ( · , a ) . Then, for all h ∈ { 0 , . . . , H -1 } , the following bound holds:

<!-- formula-not-decoded -->

Proof: We decompose the deviation:

<!-- formula-not-decoded -->

We bound the two terms separately. For ( □ ) , define L ∈ ∆ S×A as L ( s, a ) = ̂ µ h ( s ) π ( a | s ) = 1 N ∑ i ∈ N s i h = s π ( a | s ) . We (almost surely) have:

<!-- formula-not-decoded -->

For the second term ( △ ) , by applying Jensen's inequality, we have:

<!-- formula-not-decoded -->

Applying Cauchy-Schwarz's inequality, we get for any s ∈ S :

<!-- formula-not-decoded -->

By integrating this result into the previous computation and using Cauchy-Schwarz's inequality, we get:

<!-- formula-not-decoded -->

Combining the upper bounds derived for terms ( □ ) and ( △ ) , we obtain the desired result, as E [ ∥ L π h -̂ L h ∥ 1 ] = E [ E [ ∥ L π h -̂ L h ∥ 1 ∣ ∣ ∣ { s i h } i,h ]] . □

Lemma 11 (Deviation Between Empirical and Mean Field Population) Let M mfa = ( S , A , H, µ 0 , { P mfa h } H -1 h =0 , { R mfa h } H -1 h =0 ) be a BA-MFG. Let { α h } H -1 h =0 , { w h } H -1 h =0 , { p h } H -1 h =0 , and { u h } H -1 h =0 denote the allocation thresholds, transition dynamics, payment, and utility functions, respectively, from which { P mfa h } H -1 h =0 and { R mfa h } H -1 h =0 are derived. Assume these functions are Lipschitz continuous, with respective Lipschitz constants K α , K w , K p , and K u .

Consider the corresponding finite-agent Batched Auction model with N agents G auc, which is approximated by M mfa. Let π = { π i h } h =0 ,...,H -1 , i ∈ [ N ] ∈ Π N H denote the joint policy of the population. For

each round h , denote by ̂ L h ∈ ∆ S×A the empirical state-action distribution and by ̂ µ h ∈ ∆ S the corresponding empirical state distribution.

Let π ∈ Π H be an arbitrary policy, and define the associated mean field state-action distribution flow L π := Λ mfa ( π ) , with corresponding marginal state distribution µ π h := ∑ a L π h ( · , a ) . Then, for all h ∈ { 0 , . . . , H -1 } , it holds that

<!-- formula-not-decoded -->

̸

if K ξ (1 + K α ) = 1 , and

<!-- formula-not-decoded -->

if K ξ (1 + K α ) = 1 .

Proof: Let { s i h } h =0 ,...,H -1 ,i ∈ [ N ] , { a i h } h =0 ,...,H -1 ,i ∈ [ N ] and { z i h } h =0 ,...,H -1 ,i ∈ [ N ] as in Definition 6. We prove the lemma inductively over h . For h = 0 we have µ π 0 = µ 0 = E [ ̂ µ 0 ] . Let X s := ∑ i ∈ [ N ] ✶ { s i 0 = s } , which is by definition X s a binomial random variable with parameters N and µ 0 ( s ) . Since it is a sum of independent Bernoulli random variables, its variance is V ar[ X s ] = Nµ 0 ( s )(1 -µ 0 ( s )) . By using Jensen's, we can upper bound the expected absolute deviation for each state s ∈ V

<!-- formula-not-decoded -->

By summing over all states s ∈ V and applying Cauchy-Schwarz's inequality, we get:

<!-- formula-not-decoded -->

Next, for h ≥ 0 , we compute an upper bound for the deviation at step h +1 . In particular, we analyze the conditional expectation

<!-- formula-not-decoded -->

almost surely, where { z i h } N i =1 are the states of agents after the item allocation in round h as before. We upper bound the two terms ( □ ) and ( △ ) separately. For ( □ ) we have:

<!-- formula-not-decoded -->

For arbitrary s ∈ S , noting that ̂ ξ h is { z i h } -measurable,

<!-- formula-not-decoded -->

By the Cauchy-Schwarz inequality, we have

<!-- formula-not-decoded -->

Therefore, we (almost surely) have

<!-- formula-not-decoded -->

For ( △ ) it holds that

<!-- formula-not-decoded -->

where in the last step we applied Lemma 2.2 from [75]. Merging the upper bounds for ( △ ) and ( □ ) and taking expectations, we obtain

<!-- formula-not-decoded -->

Next, we bound the expected deviation between the post-allocation state distribution in the mean field model and its empirical counterpart in the finite-agent system. Specifically, we consider E [ ∥ ξ π h -̂ ξ h ∥ 1 ] , where ξ π h := Γ α h ( L π h ) is the mean field post-allocation distribution induced by policy π , and ̂ ξ h is the corresponding empirical distribution in the finite-agent auction, computed from realized allocations rather than via the operator Γ α h . Denote the σ -algebra induced by { s i h , a i h } i as F h for simplicity, then

<!-- formula-not-decoded -->

We upper-bound the two terms separately once again. For ( ♡ ) we have:

<!-- formula-not-decoded -->

We establish an upper bound on the absolute deviation for each state s ∈ V . Let s ∈ V be arbitrary. Given the empirical state-action distribution ̂ L h ∈ ∆ S×A , the corresponding empirical state distribution is given by marginalizing over actions: ̂ µ h = ∑ a ∈A ̂ L h ( · , a ) . By definition of ̂ µ h , there are N ̂ µ h ( s ) agents in state s at round h . Denoting these agents as i 1 , . . . , i N ̂ µ h ( s ) , we can express ̂ ξ h ( s ) as:

Two key observations can be made regarding these indicator variables:

<!-- formula-not-decoded -->

̸

1. The indicators are negatively correlated due to the structure of the auction. To illustrate this, assume without loss of generality that the first M agents have not won yet, i.e., s i h = ⊥ for all i ∈ [ M ] . Since the number of items in each round is fixed, when conditioning on ̂ L h , we have

<!-- formula-not-decoded -->

which implies that the indicator variables ✶ z i h = ⊥ are negatively correlated. Consequently, their complements ✶ z i h = s i h = 1 -✶ z i h = ⊥ are also negatively correlated. This implies that any subset of these indicator variables retains this negative correlation property. Specifically, for every state s , the random variables ✶ z i j h = s , j ∈ [ N ̂ µ h ( s )] , are negatively correlated.

2. Since these are Bernoulli random variables, their variance is at most 1 / 4 .

It follows from the two observations above that the variance of ̂ ξ h conditioned on F h can be upper bounded almost surely as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using this result, together with Cauchy-Schwarz's inequality, we can further bound ( ♡ ) :

<!-- formula-not-decoded -->

For the term ( ♢ ) , applying the result of Corollary 3 yields:

<!-- formula-not-decoded -->

where the second to last step follows from the Lipschitz continuity of α h , while the last step follows from ∥ ( ν -⊥ ( L π h ) -( ν -⊥ ( ̂ L h ) ∥ 1 ≤ ∥ L π h -̂ L h ∥ 1 . Additionally, ∥ E [ ̂ ξ h |F h ] -Γ α h ( ̂ L h ) ∥ 1 ≤ 1 N comes from Lemma 7, as |⌊ Nα h ( ν -⊥ ( L π h )) ⌋ -Nα h ( ν -⊥ ( L π h )) | ≤ 1 .

Combining the upper bounds for ( ♡ ) and ( ♢ ) we get:

<!-- formula-not-decoded -->

where the last step follows from Corollary 4.

Combining this result with the bound on E [ ∥ µ π h +1 -̂ µ h +1 ∥ 1 ] , we apply induction on h to conclude the proof of the lemma. □

Corollary 4 (Deviation Between Empirical and Mean Field State-Action Population) Under the conditions of Lemma 11, for all h ∈ { 0 , . . . , H -1 } , it holds that:

<!-- formula-not-decoded -->

̸

if K ξ (1 + K α ) = 1 , and

<!-- formula-not-decoded -->

if K ξ (1 + K α ) = 1 .

Proof: The upper bound is obtained easily from Lemmas 10 and 11.

## E.3.2 Step 2: Expected Deviation of Winning probabilities

We derive an explicit upper bound on the expected deviation in winning distributions between the mean field auction and its finite-agent counterpart, under a single-agent deviation from a common policy.

Lemma 12 (Expected Deviation in Allocation Dynamics) Let M mfa = ( S , A , H, µ 0 , { P mfa h } H -1 h =0 , { R mfa h } H -1 h =0 ) be a BA-MFG. Let { α h } H -1 h =0 , { w h } H -1 h =0 , { p h } H -1 h =0 , and { u h } H -1 h =0 denote the allocation thresholds, transition dynamics, payment, and utility functions, respectively, from which { P mfa h } H -1 h =0 and { R mfa h } H -1 h =0 are derived. Assume these functions are Lipschitz continuous, with respective Lipschitz moduli K α , K w , K p , and K u . Consider the corresponding finite-agent Batched Auction model with N agents approximated by M mfa . Let π = { π i h } h =0 ,...,H -1 , i ∈ [ N ] ∈ Π N H denote the joint policy of the population, and let ̂ L h ∈ ∆ S be the empirical state-action distribution at round h .

Let π ∈ Π H arbitrary, and define the associated mean field state-action distribution flow L π := Λ mfa ( π ) . Then, for all h ∈ { 0 , . . . , H -1 } , the following bounds hold:

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where for an arbitrary state-action distribution L ∈ ∆ S×A and α : ∆ ≤ 1 S → [0 , 1] , the constant C L,α is defined as in Lemma 8.

Proof: For the first inequality, we have

<!-- formula-not-decoded -->

where in the second-to-last step we used Lemma 8.

□

Additionally, for the allocation dynamics P N,α h , since each state s ∈ V can transition only either to itself or to the inactive state ⊥ , it follows that:

<!-- formula-not-decoded -->

Since the marginal probability of transitioning to the state ⊥ corresponds to the winning probability, the bound follows directly from the first inequality.

□

## E.3.3 Step 3: Exploitability Deviation for BA-MFG

Finally we prove the absolute difference in expected reward due to a single-side policy deviation.

Theorem 4 Let M mfa = ( S , A , H, µ 0 , { P mfa h } H -1 h =0 , { R mfa h } H -1 h =0 ) be a Batched Auction Mean Field Game (BA-MFG). Let { α h } H -1 h =0 , { w h } H -1 h =0 , { p h } H -1 h =0 , and { u h } H -1 h =0 denote the allocation thresholds, transition dynamics, payment, and utility functions, respectively, from which { P mfa h } H -1 h =0 and { R mfa h } H -1 h =0 are derived. Assume these functions are Lipschitz continuous, with respective Lipschitz constants K α , K w , K p , and K u .

Consider the corresponding finite-agent Batched Auction G auc with N agents, which is approximated by M mfa. Let π ∈ Π H an arbitrary policy satisfying the no-zero dominance property. Then, for any policy π ∈ Π H , τ ≥ 0 it holds

<!-- formula-not-decoded -->

Proof: Define the random variables { s i h , a i h , z i h } i ∈ [ N ] , h ∈{ 0 ,...,H -1 } , along with { ̂ L h } H -1 h =0 , { ̂ µ h } H -1 h =0 , and { ̂ ξ h } H -1 h =0 , as in the definition of the N -player Batched Auction (see Definition 6). Here, s i h denotes the state of agent i at round h , a i h its action, and z i h its hidden state following the allocation step. The random variables ̂ L h , ̂ µ h , and ̂ ξ h represent, respectively, the empirical state-action distribution, the empirical state distribution, and the empirical post-allocation state distribution at round h .

For the Mean-Field Batched Auction, define { s h , a h , z h } H -1 h =0 , where s h and a h represent the state and action of a representative agent at round h , and z h denotes its post-allocation hidden state. These evolve deterministically according to the mean-field population flows L π , µ π , ξ π induced by the population policy π .

We divide the proof into three steps:

1. We show that for every h ∈ { 0 , . . . , H -1 } we have: ∥ ∥ P [ s h = · ] -P [ s 1 h = · ] ∥ ∥ 1 ≤ ∑ h ′ &lt;h E [ ∥ L π h ′ -̂ L h ′ ∥ 1 ] ( K w ( K α +1) + 2 C L π h ′ ,α h ′ ) + 2 C L π h ′ ,α h ′ ( K α +1) N + hK w (√ |S| 2 √ N + 1 N )
2. We show that for every h ∈ { 0 , . . . , H -1 } we have:

<!-- formula-not-decoded -->

3. In the last step we combine the results of the previous steps to prove the man claim of the theorem.

Step 1: We prove the inequality by induction on h . For the base case h = 0 , we have ∥ P [ s 0 = · ] -P [ s 1 0 = · ] ∥ 1 = 0 , since both distributions are equal to the initial distribution µ 0 . Now, assuming the bound holds for some h ≥ 0 , we show that it also holds for round h +1 .

<!-- formula-not-decoded -->

By adding and subtracting ∑ z,ξ P [ z 1 h = z, ̂ ξ h = ξ ] w h ( z, ξ π h ) = ∑ z P [ z 1 h = z ] w h ( z, ξ π h ) , and applying triangular inequality, we get

<!-- formula-not-decoded -->

The term E [ ∥ ξ π h -̂ ξ h ∥ 1 ] , using the same derivation as in the inductive step of Lemma 11, can be further upper bounded as

<!-- formula-not-decoded -->

Finally applying a similar reasoning we can upper bound ∥ P [ z h = · ] -P [ z 1 h = · ] ∥ 1 .

<!-- formula-not-decoded -->

By adding and subtracting ∑ s , a P [ s h = s , a h = a ] P α h ( s 1 , a 1 , L π h ) = ∑ s,a P [ s 1 h = s, a 1 h = a ] P α h ( s, a, L π h ) , and applying triangular inequality, we get

<!-- formula-not-decoded -->

Applying Lemma 12 it follows

<!-- formula-not-decoded -->

By combining the two bounds, we obtain

<!-- formula-not-decoded -->

Applying the induction hypothesis completes the proof for round h +1 .

Step 2: We prove the inequality by using the result obtained in step 1. Let h ∈ { 0 , . . . , H -1 } arbitrary, then

<!-- formula-not-decoded -->

The first term ( □ ) can be upper bounded by

<!-- formula-not-decoded -->

For the second term ( △ ) we have:

<!-- formula-not-decoded -->

where in the second to last step we used the Lipschitz continuity of u ◦ p and Lemma 12. Combining both results we get:

<!-- formula-not-decoded -->

Step 3: We now combine the results from the previous two steps to establish the final bound stated in the theorem.

<!-- formula-not-decoded -->

We proceed by bounding each term individually for every round h . Let h ∈ { 0 , · · · H -1 } arbitrary, then

<!-- formula-not-decoded -->

The first term is bounded using the result from Step 2, while the second term can be handled as follows:

<!-- formula-not-decoded -->

Therefore the (entropy regularized) absolute difference in rewards at round h can be upper bounded by

<!-- formula-not-decoded -->

By applying the bound on | P [ s h = · ] -P [ s 1 h = · ] | 1 derived in Step 1, and summing the per-round deviations over all h , it follows:

<!-- formula-not-decoded -->

In particular, Corollary 4 implies that the total (entropy-regularized) reward difference is of order O ( 1 √ N ) . □

Conclusion and Statement of Result. Let M mfa be a BA-MFG with Lipschitz-continuous { u h } H -1 h =0 , { w h } H -1 h =0 , { α h } H -1 h =0 , { p h } H -1 h =0 Let π δ ∈ Π H be a policy that satisfy the no zerodominance property. Let further assume π δ is a δ -MFG-NE, namely

<!-- formula-not-decoded -->

Then, for π = ( π δ , . . . , π δ ) , we have:

<!-- formula-not-decoded -->

̸

In case K ξ ( K α +1) = 1 , the constants C 1 and C 2 are given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In case K ξ ( K α +1) = 1 , the constants C 1 and C 2 are given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

Remark 4 For policies with full support, the Lipschitz constant associated with the winning probability is of order 1 ϵ , as discussed in Remark 3. Consequently, the constants C 1 and C 2 scale as O ( 1 ϵ H 2 · 1 -( K ξ ( K α +1)) H 1 -K ξ ( K α +1) ) when K ξ ( K α +1) = 1 , and as O ( 1 ϵ H 3 ) when K ξ ( K α +1) = 1 .

Explanation of Constants. In the above expression, the constants represent key components of the BA-MFG dynamics:

- B u and K u are the bound and Lipschitz constant of the utility function u respectively.
- B p and K p are the bound and Lipschitz constant of the payment function p respectively.
- K w denotes the Lipschitz constant of the transition function w .
- K α is the Lipschitz constant of the allocation threshold function α .
- K s = sup s,s ′ ,ξ ∥ w ( s, ξ ) -w ( s ′ , ξ ) ∥ 1 .
- K ξ = K w + 1 2 K s .
- C L,α is the Lipschitz constant of the winning probability function evaluated at the distribution L , assuming L satisfies the no zero-dominance property. For its precise definition, see Lemma 8.
- τ is the entropy regularization parameter.

## E.4 Proof of Theorem 2, part 2 (Approximation in Objective)

We show that, under Lipschitz conditions, the objective computed under the mean field approximation closely matches its expected value under a finite population of agents.

Theorem 5 (Convergence of the Mechanism Objective) Let g : ∆ H S×A → R be a Lipschitzcontinuous objective defined over the class of Batched Auction Mean Field Games (BA-MFGs). Let M mfa be a BA-MFG with Lipschitz-continuous { u h } H -1 h =0 , { w h } H -1 h =0 , { α h } H -1 h =0 , { p h } H -1 h =0 . Let π = ( π, . . . , π ) be the joint population policy for some π ∈ Π H . Then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof: We use a decomposition over the support of ̂ L and apply the triangle inequality:

<!-- formula-not-decoded -->

where K g is the Lipschitz constant of g . The result follows by applying the bound from Corollary 4. □

Lemma 13 (Lipschitz Continuity of Expected Revenue) Let g rev denote the expected revenue objective. Let M mfa be a BA-MFG with Lipschitz-continuous payment functions { p h } H -1 h =0 and allocation functions { α h } H -1 h =0 . Let L = { L h } H -1 h =0 and L ′ = { L ′ h } H -1 h =0 be two arbitrary state-action distribution trajectories over H rounds. Then,

<!-- formula-not-decoded -->

where B p is a uniform bound on the absolute value of the payment functions, K p is their Lipschitz constant, and K α is the Lipschitz constant of the allocation threshold functions.

Proof: The expected revenue, for L = { L h } H -1 h =0 , can be rewritten using the operator Ξ

<!-- formula-not-decoded -->

To simplify notation, we define

<!-- formula-not-decoded -->

representing the residual (unallocated) mass at each state-action pair. Applying the triangle inequality:

<!-- formula-not-decoded -->

Using the boundedness of the payment function p and the Lipschitz continuity of the allocation operator Ξ α , the first term can be bounded by B p (2 + K α ) ∑ H -1 h =0 ∥ L h -L ′ h ∥ 1 . For the second term, the Lipschitz property of p implies a bound of ∑ H -1 h =0 K p ∥ L h -L ′ h ∥ 1 1 . Combining these, we conclude that the revenue objective g rev is Lipschitz continuous with constant B p (2 + K α ) + K p , and satisfies the bound

<!-- formula-not-decoded -->

Corollary 5 (Convergence of Expected Revenue) Let g rev the expected revenue objective. Let M mfa be a BA-MFG with Lipschitz-continuous { u h } H -1 h =0 , { w h } H -1 h =0 , { α h } H -1 h =0 , { p h } H -1 h =0 . Let π = ( π, . . . , π ) be the joint population policy for some π ∈ Π H . Then:

<!-- formula-not-decoded -->

Proof: The result follows by combining Theorem 5 and lemma 13.

## E.5 Proof of Lemma 3

In this section, we prove that the (entropy regularized) q-functions are Lipschitz continuous with respect to the population policy, assuming full support. We begin by showing that the population flow is Lipschitz in the policy. Next, we establish that both the transition dynamics and the reward function are Lipschitz continuous with respect to the population distribution. Finally, we combine these results to derive a bound on the Lipschitz constant of the (entropy regularized) q-functions.

□

## Lemma 14 (Lipschitz Continuity of Population Operator) Let M mfa

=

( S , A , H, µ 0 , { P mfa h } H -1 h =0 , { R mfa h } H -1 h =0 ) be a Batched Auction Mean Field Game (BA-MFG). Let { α h } H -1 h =0 , { w h } H -1 h =0 , { p h } H -1 h =0 , and { u h } H -1 h =0 denote the allocation thresholds, transition dynamics, payment, and utility functions, respectively, from which { P mfa h } H -1 h =0 and { R mfa h } H -1 h =0 are derived. Assume these functions are Lipschitz continuous. Consider two arbitrary policies π, π ′ ∈ Π H , then

<!-- formula-not-decoded -->

Proof: We prove the bound inductively. Let L π := Λ mfa ( π ) and L π ′ := Λ mfa ( π ′ ) denote the population flows induced by policies π and π ′ , respectively. Similarly, let µ π and µ π ′ denote the corresponding marginal state distributions.

For h = 0 we have µ π 0 = µ π ′ 0 = µ 0 . Therefore by Lemma 9 we have ∥ L π 0 -L π ′ 0 ∥ 1 ≤ ∥ π 0 -π ′ 0 ∥ 1 .

For h +1 &gt; 0 by applying the result of Lemma 9 we have

<!-- formula-not-decoded -->

We then bound the variational difference in state distribution:

<!-- formula-not-decoded -->

where in the last step we used Corollary 3. By induction over h the claim follows. □

## Lemma 15 (Lipschitz Continuity of Transitions and Rewards under Full-Support Policies)

Let M mfa be a BA-MFG with utility functions { u h } H -1 h =0 , transition dynamics { w h } H -1 h =0 , payment functions { p h } H -1 h =0 , and allocation thresholds { α h } H -1 h =0 , all of which are Lipschitz-continuous with constants K u , K w , K p , and K α , respectively. Consider two policies π, π ′ ∈ Π H with full support; that is, for all s ∈ S , a ∈ A , and h ∈ 0 , . . . , H -1 , we have π h ( a | s ) &gt; ϵ and π ′ h ( a | s ) &gt; ϵ for some constant ϵ &gt; 0 . Then

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Proof: Let π, π ′ ∈ Π H be two arbitrary policies with full support Let L π h = Λ mfa ( π ) and L π ′ h = Λ mfa ( π ′ ) . We then bound separately the rewards and the transition probabilities.

̸

We first bound bound the absolute difference in rewards, let s ∈ S , a ∈ A arbitrary, assume s = ⊥ , else the claim holds trivially, then, by applying triangular inequality we have:

<!-- formula-not-decoded -->

where the last step follows by Remark 3 and the Lipschitz continuity of p and u . By Lemma 14 the bound for the rewards follows.

Similarly for the transition probabilities we have:

<!-- formula-not-decoded -->

From Remark 3 and Corollary 3 it follows

<!-- formula-not-decoded -->

by Lemma 14 the bound follows.

□

Lemma 16 (Lipschitz Continuity of Regularized Value Functions) Let M mfa be a BA-MFG with utility functions { u h } H -1 h =0 , transition dynamics { w h } H -1 h =0 , payment functions { p h } H -1 h =0 , and allocation thresholds { α h } H -1 h =0 , all of which are Lipschitz-continuous with constants K u , K w , K p , and K α , respectively.

Let π, π ′ ∈ Π H be two policies with full support, i.e., for some ϵ &gt; 0 , π h ( a | s ) , π ′ h ( a | s ) &gt; ϵ , for all h, s, a . Let V τ h ( s | L π , π ) and V τ h ( s | L π ′ , π ′ ) denote the entropy-regularized value functions under policies π and π ′ , defined recursively as

<!-- formula-not-decoded -->

where { L π h } H -1 h =0 := Λ mfa ( π ) is the population distribution induced by policy π .

Then, there exists a constant C &gt; 0 such that for all h and all s ∈ S , the following bound holds:

<!-- formula-not-decoded -->

where C = O ( 1 ϵ H 2 ) when K ξ (1+ K α ) = 1 , and C = O ( 1 ϵ H 1 -( K ξ (1+ K α )) H 1 -K ξ (1+ K α ) ) when K ξ (1+ K α ) = 1 .

̸

Proof: We prove a stronger inductive bound that implies the result of the lemma. To simplify notation, we introduce the following constants:

- β := K ξ (1 + K α ) ,
- B := B u + τ log( |A| ) ,
- g 0 := B u (1 -α max ) ϵ + K u K p ,

<!-- formula-not-decoded -->

We also define the following h -dependent quantities, which will be used in the inductive argument:

- A h := τ (log( 1 ϵ ) + 1) + ( H -h ) B ,
- G h := g 0 + g 1 ( H -h -1) ,
- ∆ h := ∥ π h -π ′ h ∥ 1 .

We proceed by backward induction on the round h , and show that the following bound holds:

<!-- formula-not-decoded -->

For h = H by definition, the value function at round H is zero for all states, i.e.,

<!-- formula-not-decoded -->

Assume that for value function at time h +1 , V τ h +1 the upper bound holds. We now prove the same for V τ h . The regularized value function is:

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

For the entropy term ( □ ) we have

<!-- formula-not-decoded -->

By applying triangular inequality we can upper bound ( △ ) as follows:

<!-- formula-not-decoded -->

The first term can be bounded as a function of H and the maximal absolute utility B u :

<!-- formula-not-decoded -->

where the term τ log( |A| ) comes from the maximal additional reward from the entropy regularizer.s For the second term we bound the absolute difference in q functions:

<!-- formula-not-decoded -->

By combining all intermediate bounds and applying Lemma 15 we have

<!-- formula-not-decoded -->

by induction the bound holds. As next we compute the global Lipschitz constant.

<!-- formula-not-decoded -->

For ( □ ) we have:

For ( △ ) we have:

<!-- formula-not-decoded -->

̸

Using the definition of the geometric sum, and observing that the constants G h ′ as well as the nested geometric sum decrease as h ′ increases, we can further simplify the bound by pulling out the leading terms. For β = 1 :

<!-- formula-not-decoded -->

while for β = 1 :

<!-- formula-not-decoded -->

̸

Combining the results (for β = 1 ) of ( □ ) and ( △ ) we have:

<!-- formula-not-decoded -->

For β = 1 the geometric term is replaced by H . By the definitions of B,g 0 and g 1 the claim follows.

□

<!-- formula-not-decoded -->

## F Experiment Details

Implementation Details. The experiments were implemented in JAX and PyTorch, the code is provided in the supplementary material. We implement the adjoint method in JAX. For the PyTorch implementation, some code was adapted from [31]. All error bars in experiments are one standard deviation away from the mean.

Hardware and Compute Time. We run our experiments on a single NVIDIA H100 GPU with an AMD EPYC 16-core CPU. One run of AMID for 1000 iterations takes 6 minutes, apart from the experiment (A8) described below with a long time horizon H = 100 , which takes 20 minutes for 1000 iterations. The beach bar process experiments take roughly 3 minutes for 1000 iterations.

Parameterizing θ in M bb. For the beach bar process, we parameterize the per state payments as θ s = p max sigmoid( ξ s ) , where p max is the maximum per state payment and the unconstrained parameters ξ ∈ R S are learned via AMID.

Parameterizing θ in M mfa. We parameterize p θ h and sold goods α θ h as residual neural networks sharing a base. The base network, f θ base , has d in = H + |A| +1 inputs consisting of one-hot encoded time vector e h , |A| -dimensional vector of bid distribution ν -⊥ , and remaining goods at round h denoted r h , (given by α max -∑ h -1 h ′ =0 α h ′ ). For the input vector x in ∈ R d in , the base residual network is defined as

<!-- formula-not-decoded -->

The goods to be sold this round are then computed by:

<!-- formula-not-decoded -->

and the payments functions for bids is computed then by:

<!-- formula-not-decoded -->

Note that this parameterization ensures that the payment rule p θ h is a monotonic increasing function of bids a . The parameters θ of the mechanism overall are W (1) , V (2) ∈ R d hidden × ( H + d +1) , W (2) , W (3) ∈ R d hidden × d hidden , W (4) ∈ R ( A -1) × d hidden and b (1) , b (2) , c (2) , b (3) , w g ∈ R d hidden , b (4) ∈ R A -1 , b g ∈ R .

Baseline algorithms. For the zeroth-order baseline algorithms 0 -SGD and 0 -Adam, we use the standard 2-point (biased) gradient estimator

<!-- formula-not-decoded -->

where θ ∈ R D , z is uniformly distributed on the sphere S D -1 , and u zero is a tunable hyperparameter. This estimator satisfies the well-known property

<!-- formula-not-decoded -->

That is, ̂ ∇ θ is an unbiased estimator of the gradient of a smoothed version of the function G T approx . The bias is tunable by the parameter u zero, with smaller values corresponding to less bias but potentially higher variance in estimates. Since a single evaluation of this gradient estimator takes 2 forward passes over G T approx , its run time is comparable to that of AMID per iteration. For the baseline ANNEAL, each iteration, we sample a perturbation n from the D -dimensional standard normal distribution. After evaluating G T approx ( θ ) , G T approx ( θ + σ anneal n ) , G T approx ( θ -σ anneal n ) for the tunable hyperparameter σ anneal &gt; 0 , ANNEAL updates θ to be the best among θ, θ -σ anneal n, θ + σ anneal n .

Hyperparameters. All hyperparameters for the baselines as well as AMID are presented in Table 4. For a fair comparison, we perform a grid search on a range of values for the parameters for all baselines and take the best run after 10 repetitions. In our experiments, AMID is robust to hyperparameter choices while zeroth order methods require some tuning.

## F.1 Additional Results on the Beach Bar Process

As mentioned in the main body of the paper, we first present the payment function θ s learnt AMID after 1000 iterations. As before, we report these by using a slightly higher OMD iteration step T val = 500 than used for training, to demonstrate the robustness of our method.

Table 4: Hyperparameters for the experiments on auctions.

| Parameter   | Explanation                                                                                   | Values                                              |
|-------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------|
| η           | Adam/SGD learning rate                                                                        | 3e - 5 , 1e - 4 , 3e - 4 , 1e - 3 , 1e - 3 , 1e - 2 |
| u zero      | Noise magnitude for evaluating zeroth-order gradi- ent estimator ̂ ∇ θ , for 0-SGD and 0-ADAM | 1e - 3 , 1e - 2 , 3e - 2                            |
| σ anneal    | Perturbation magnitude for ANNEAL                                                             | 1e - 6 , 1e - 5 , 1e - 4 , 1e - 3 , 1e - 2 , 3e - 2 |
| τ           | Entropy regularization                                                                        | 1e - 3                                              |
| η OMD       | OMDlearning rate                                                                              | 10                                                  |
| T           | OMDiterations in ( T -approx.)                                                                | 400                                                 |
| T val       | OMDiterations for validation                                                                  | 500                                                 |
| d hidden    | Hidden dimension of residual network parameter- izing payments and sold goods                 | 256                                                 |

Figure 4: Payment function s → θ s learned after training with AMID, where payments are bounded on [0 , 1 / 2 ] .

<!-- image -->

A bottleneck in the beach bar experiment is the magnitude of payments θ s , which is restricted to be bounded on [0 , 1 / 2 ] . We also report the experiment when θ ∈ [0 , 4 / 5 ] S below (i.e., when p max = 0 . 8 ), in Figure 5. As expected, the population distributions L h are smoother and closer to uniform in this case. In both cases, AMID behaves as expected: the exploitability of T iterates of OMD is consistently low throughout training, suggesting that the T step approximation objective remains close to a NE throughout training iterations (with exploitability &lt; 0 . 02 ). The fact that the exploitability of the policy induced by the T repeated iterations of OMD is due to the fact that in the beach bar experiments, tuning payments yields a NE distribution closer to uniform, which is also the initialization of OMD iterates.

where we take β = 1 .

- (A6) Hyperbolic time discounting, which discounts future rewards, where the utility at time h is given by:

<!-- formula-not-decoded -->

where we take λ = 1 as the time discount factor.

Across utility functions, AMID manages to beat all baselines. In our experiments, we also observed significant qualitative changes in both the mean-field Nash equilibrium and the payment rule when nonlinear utility functions are used. We show NE and payment rules suggested by the neural mechanism as NE in Figure 7.

Experiments regarding the impact of horizon. We report the impact of agent regeneration and time horizon by evaluating AMID on the following auction setups:

- (A7) H = 6 , agents regenerate with probability 1 (that is, they never transition to ⊥ even when they win a round), linear utilities for bidders, α max = 0 . 8 , µ 0 ( s ) ∝ γ s for γ = 0 . 9 , dynamic values with w ( s ′ | s ) ∝ exp { -(1 . 2 s -s ′ ) 2 / 2 σ 2 } for σ = 0 . 2 .
- (A8) Long horizon H = 100 , α max = 5 , linear utilities for bidders, agents regenerate with probability 0 . 015 , µ 0 ( s ) ∝ γ s for γ = 0 . 9 , dynamic values with w ( s ′ | s ) ∝ exp { -(1 . 2 s -s ′ ) 2 / 2 σ 2 } for σ = 0 . 01 .

The results are reported on Figure 8.

τ

Figure 5: Payment design with AMID in M bb with larger payments. Left: objective and exploitability throughout training iterations. Middle: learned payment rule after training with AMID. Right: population flow in time after learning payments.

<!-- image -->

## F.2 Additional Results on Auctions

We first analyze 3 further auctions with nonlinear utility functions for bidders. Namely, we take (A1) presented in the main body of the paper where H = 4 , µ 0 = Uniform( S ) , α max = 0 . 8 , |S| = |A| = 100 and bidders are single-minded with no evolution in valuations s i h other than to transitions to ⊥ .

- (A4) Risk-averse utilities formulated by

<!-- formula-not-decoded -->

where we take β = 1 .

- (A5) Risk-seeking utilities formulated by

<!-- formula-not-decoded -->

Figure 6: g rev throughout iterations of AMID and baseline algorithms in settings with nonlinear utilities for bidders (A4-6), left to right.

<!-- image -->

Figure 7: Left: mean bids at NE at θ ∗ after training with AMID on the risk-seeking utility experiment (A5) for h ∈ { 0 , 1 , 2 , 3 } . Middle: mean bids at NE at θ ∗ after training with AMID on the hyperbolic time discounting utility experiment (A6) for h ∈ { 0 , 1 , 2 , 3 } . Right: payment function in setting (A5) at the bids induced by the NE policy at θ ∗ for h ∈ { 0 , 1 , 2 , 3 } .

<!-- image -->

General objectives. Finally, we explore the impact of optimizing over more general objectives other than revenue. We define the objective

<!-- formula-not-decoded -->

where we define g efficiency as:

<!-- formula-not-decoded -->

We modify experiment (A1) and evaluate our AMID on the following setting.

(A9) H = 4 , µ 0 = Uniform( S ) , α max = 0 . 8 , and single-minded bidders (after winning stay at state ⊥ ) with no evolution in valuations s i h otherwise. The objective function is g mix.

The results are reported on Figure 8.

Experiments with static α h and payment rules. Finally, to verify the impact of the mechanism having access to bids ν -⊥ h at round h , we run a final experiment where the mechanism is independent of bids, which we call a static mechanism . In this case, we simply parameterize

<!-- formula-not-decoded -->

are sold and the payments never exceed the bid. The parameter space is then θ := [ θ (1) , θ (2) ] , for θ (1) ∈ R [ H ] ×A , θ (2) ∈ R [ H ] . For static mechanisms, we observe much less significant improvement over the first price mechanism in general, which most likely originates from better allocation of goods over time when there are dynamics such as regeneration.

(A10) H = 4 , static mechanism parameterization (independent of ν -⊥ h ), each agent regenerates with probability 0 . 3 at the end of every round, and has a linear utility function.

The results in this setting are reported in Figure 9.

Figure 8: Left-middle: g rev throughout iterations of AMID and baseline algorithms in settings (A7-8). Right: g mix throughout iterations of AMID and baseline algorithms, in setting (A9).

<!-- image -->

Experiments on approximating exploitability. We also report our attempts to measure the true N player exploitability gap suggested by Theorem 2 when N is finite. For the first price mechanism, we compute the MFG-NE on the auction setting (A1) introduced in the main paper. Then, fixing N = 1000 , we simulate trajectories in the batched auction by setting the policies of 999 agents to the MFG-NE, and train PPO on the last bidder. In this setting, we were not able to achieve a better mean reward for the last bidder than the MFG-NE. This suggests the exploitability is close to 0 , we report the expected reward achieved by PPO throughout training in Figure 9.

Figure 9: Left: Revenue throughout training in the static mechanism setting (A10). Right: PPO episodic rewards trained on the batched auction (A1) with N = 1000 agents, all but one playing NE. Blue is the MFG best response expected reward, red is PPOs expected reward throughout training.

<!-- image -->